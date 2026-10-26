"""Enforced optimisation-stability instrumentation for r5 critic, advantage and BC fits.

Brief section 17 (stability and learning dynamics) and the r5 acceptance
harness. One :class:`StabilityMonitor` wraps the optimizer step of a fit:

1. **Finite updates.** Every step checks the loss, the global pre-clip gradient
   norm (which is nonfinite iff any gradient is), and after the step every
   parameter and every Adam moment. On a nonfinite value the parameters and
   the optimizer state are restored to the last good step, the learning rate is
   multiplied by ``lr_backoff`` (halved), and an event is recorded. ``K``
   consecutive failures raise :class:`StabilityAbort` with the receipt attached.
2. **Per-block dynamics.** Every ``log_every`` steps: per-head losses, per-block
   pre-clip gradient norms (frozen blocks such as the actor encoder must be
   exactly zero), the global clip multiplier and the per-block parameter-step
   norms. A :class:`DominanceDetector` watches each shared block (the task
   encoder shared by QoL and population-years; the global clip shared by the
   task critic and both costs) with a preregistered window and progress
   criterion.
3. **Bounded QoL saturation** (share of pre-activations with |z| > 6).
4. **TD(lambda) target drift** per refresh, with the divergence alarm.
5. **Decoded tail and upper-bin mass** of the distributional population head.
6. **Advantage overfit**: train versus held-out pair ranking per epoch.
7. **Privileged-token attention mass** and dead-feature counts (probe batch).

The monitor never changes a successful step's arithmetic: it reads gradients
and copies state, and the optimizer step and global clip are the same calls in
the same order as the unmonitored loops, so a monitored fit without failures
is bitwise identical to an unmonitored one (tested).

Every threshold in :class:`StabilityThresholds` is a preregistration
**proposal** for the user's review (``PROPOSED_THRESHOLDS``), except the |z| > 6
saturation definition (amendment r5-b item 4) and the 3-refresh TD growth
window (r5 freeze), which are frozen r5 text.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

STABILITY_SCHEMA = "labor-r5-stability-v1"
MANDATORY_CHECKS = (
    "finite_updates",
    "frozen_gradients_zero",
    "gradient_dominance",
    "qol_saturation",
    "td_divergence",
    "population_tail",
    "advantage_overfit",
    "privileged_attention",
)
ADVISORY_CHECKS = ("dead_features",)
STATUSES = ("PASS", "FAIL", "HOLD", "NOT_APPLICABLE")


class StabilityAbort(ValueError):
    """K consecutive nonfinite steps: the fit is aborted and its receipt attached."""

    def __init__(self, message: str, summary: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.summary = dict(summary)


@dataclass(frozen=True)
class StabilityThresholds:
    """Preregistered stability thresholds (proposals unless noted)."""

    # 1. Finite updates.
    max_consecutive_failures: int = 3  # K
    lr_backoff: float = 0.5  # "LR halving"
    max_total_rollbacks: int = 5  # a fit with more rollbacks is not admitted
    # 2. Dominance on a shared block (brief 17.2: window and progress criterion).
    dominance_window: int = 16  # log points (16 x 25 = 400 optimizer steps; user decision 26 Oct)
    dominance_share: float = 0.90  # one contender's share of the block's gradient norm
    dominance_persistence: float = 0.75  # share of window points above dominance_share
    dominance_min_progress: float = 0.01  # dominated contender's relative loss decrease
    # 3. Bounded QoL head: share of |z| > 6 (r5-b item 4 defines the monitor).
    #    Approved with relaxation (user 26 Oct): low-side saturation on rows whose bounded
    #    target fraction G/W is below ``saturation_expected_low_target_fraction`` is expected
    #    (no housing/clothing yet) and excluded from the alarm share; the first
    #    ``saturation_early_decisions`` decisions of each (world, camp, trajectory) use the
    #    relaxed ``saturation_alarm_early``; all other rows keep ``saturation_alarm``.
    saturation_alarm: float = 0.05
    saturation_alarm_early: float = 0.20
    saturation_early_decisions: int = 5
    saturation_expected_low_target_fraction: float = 0.05
    # 4. TD(lambda) drift: growth over 3 consecutive refreshes (r5 freeze).
    #    Approved trajectory rule (user 26 Oct): judge the trajectory, not an absolute bound.
    td_growth_refreshes: int = 3  # divergence: this many consecutive increases ...
    td_divergence_increase: float = 0.10  # ... each larger than this relative increase
    td_warmup_refreshes: int = 2  # refreshes ignored while drift and error may be large
    td_final_ratio: float = 0.5  # final value at most this x the post-warm-up maximum
    td_drift_floor: float = 1e-3  # values (fraction of W) below this count as converged
    td_bound_tolerance: float = 1e-6  # hard [0, W] bound slack, relative to W
    # 5. Population head decoded tail (reference-scale units; 10% of the 0.10 bias allowance).
    upper_excess_alarm: float = 0.01
    top_bin_mass_alarm: float = 1e-3
    # 6. Advantage head overfit.
    overfit_gap: float = 0.10  # final train - held-out ranking
    overfit_heldout_drop: float = 0.03  # best - final held-out ranking
    # 7. Privileged attention: village-token mass on privileged keys / uniform share.
    attention_ratio_floor: float = 0.10
    dead_unit_std: float = 1e-6

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)


PROPOSED_THRESHOLDS = tuple(
    name
    for name in StabilityThresholds.__dataclass_fields__
    if name not in ("td_growth_refreshes",)
)


@dataclass(frozen=True)
class StabilityConfig:
    """Monitor settings. ``inject_*`` are fault-injection hooks for tests only;
    a fit that used them is never admitted (the acceptance receipt FAILs it)."""

    enabled: bool = True
    log_every: int = 25
    jsonl: str | None = None
    probe_rows: int = 64
    thresholds: StabilityThresholds = field(default_factory=StabilityThresholds)
    inject_nonfinite_grad_steps: tuple[int, ...] = ()
    inject_nonfinite_param_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.log_every < 1:
            raise ValueError("stability log_every must be >= 1")
        if not 0.0 < self.thresholds.lr_backoff < 1.0:
            raise ValueError("stability lr_backoff must be in (0, 1)")
        if self.thresholds.max_consecutive_failures < 1:
            raise ValueError("stability K must be >= 1")

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "log_every": self.log_every,
            "jsonl": self.jsonl,
            "probe_rows": self.probe_rows,
            "thresholds": self.thresholds.state_dict(),
            "inject_nonfinite_grad_steps": list(self.inject_nonfinite_grad_steps),
            "inject_nonfinite_param_steps": list(self.inject_nonfinite_param_steps),
        }

    @property
    def fault_injection(self) -> bool:
        return bool(self.inject_nonfinite_grad_steps or self.inject_nonfinite_param_steps)


def resolve_stability(
    stability: StabilityConfig | str | None, *, r5: bool
) -> StabilityConfig | None:
    """``"auto"`` enables the monitor for r5 models (privileged block or bounded QoL head)."""
    if stability is None:
        return None
    if isinstance(stability, str):
        if stability != "auto":
            raise ValueError("stability must be a StabilityConfig, 'auto' or None")
        return StabilityConfig() if r5 else None
    if not isinstance(stability, StabilityConfig):
        raise ValueError("stability must be a StabilityConfig, 'auto' or None")
    return stability if stability.enabled else None


def is_r5_critic(task_critic: nn.Module) -> bool:
    return (
        getattr(task_critic, "privileged_layout", None) is not None
        or getattr(task_critic, "qol_head", None) is not None
    )


# ---------------------------------------------------------------------------
# Pure detectors (fixture-tested)
# ---------------------------------------------------------------------------


class DominanceDetector:
    """One contender's gradient dominating a shared block without others' progress.

    Each log point supplies, per contender, its gradient-norm contribution on
    the shared block and its current loss. Over every full window of
    ``window`` points, contender ``d`` dominates when its share
    ``g_d / sum(g)`` exceeds ``share`` on at least ``persistence`` of the
    points; the alarm fires when, in such a window, some other contender made
    less than ``min_progress`` relative loss progress (mean of the window's
    second half against its first half). Dominance with progress is recorded
    but is not an alarm: the brief's trigger is dominance *with failed
    other-head progress*.
    """

    def __init__(
        self,
        name: str,
        contenders: Sequence[str],
        *,
        window: int,
        share: float,
        persistence: float,
        min_progress: float,
    ) -> None:
        if len(contenders) < 2:
            raise ValueError("dominance needs at least two contenders")
        if window < 2:
            raise ValueError("dominance window must cover at least two log points")
        self.name = name
        self.contenders = tuple(contenders)
        self.window = int(window)
        self.share = float(share)
        self.persistence = float(persistence)
        self.min_progress = float(min_progress)
        self.points: list[dict[str, Any]] = []
        self.alarms: list[dict[str, Any]] = []
        self.dominant_windows = 0

    def update(
        self, step: int, norms: Mapping[str, float], losses: Mapping[str, float]
    ) -> dict[str, Any] | None:
        total = sum(max(0.0, float(norms[c])) for c in self.contenders)
        shares = {c: (float(norms[c]) / total if total > 0 else 0.0) for c in self.contenders}
        self.points.append(
            {
                "step": int(step),
                "shares": shares,
                "losses": {c: float(losses[c]) for c in self.contenders},
            }
        )
        if len(self.points) < self.window:
            return None
        recent = self.points[-self.window :]
        for dominant in self.contenders:
            above = sum(1 for p in recent if p["shares"][dominant] > self.share)
            if above < self.persistence * self.window:
                continue
            self.dominant_windows += 1
            half = self.window // 2
            stalled = {}
            for other in self.contenders:
                if other == dominant:
                    continue
                first = [p["losses"][other] for p in recent[:half]]
                second = [p["losses"][other] for p in recent[half:]]
                start = sum(first) / len(first)
                end = sum(second) / len(second)
                progress = (start - end) / abs(start) if start != 0 else 0.0
                if not math.isfinite(progress) or progress < self.min_progress:
                    stalled[other] = progress
            if stalled:
                alarm = {
                    "check": "gradient_dominance",
                    "block": self.name,
                    "step": int(step),
                    "dominant": dominant,
                    "dominant_share_points": above,
                    "window": self.window,
                    "stalled_progress": stalled,
                }
                self.alarms.append(alarm)
                return alarm
        return None

    def report(self) -> dict[str, Any]:
        return {
            "block": self.name,
            "contenders": list(self.contenders),
            "points": len(self.points),
            "dominant_windows": self.dominant_windows,
            "alarms": list(self.alarms),
            "last_shares": self.points[-1]["shares"] if self.points else None,
        }


def trajectory_report(series: Sequence[float], *, ratio: float, floor: float) -> dict[str, Any]:
    """Diminishing-trajectory test on a post-warm-up series (fractions of W).

    Passes when the series has converged below ``floor`` (its final value is under the
    floor), or when the least-squares slope of log(value) over the refresh index is
    negative and the final value is at most ``ratio`` x the series maximum. Fewer than
    three points cannot define a slope and are reported as ``insufficient`` (no alarm).
    """
    values = [float(v) for v in series]
    report: dict[str, Any] = {
        "series": values,
        "slope": None,
        "final_over_max": None,
        "status": "insufficient",
        "pass": True,
    }
    if not values:
        return report
    peak = max(values)
    report["final_over_max"] = values[-1] / peak if peak > 0 else 0.0
    if values[-1] < floor:
        report.update(status="converged", **{"pass": True})
        return report
    if len(values) < 3:
        return report
    logs = [math.log(max(v, 1e-12)) for v in values]
    n = len(logs)
    mean_x = (n - 1) / 2.0
    mean_y = sum(logs) / n
    slope = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(logs)) / sum(
        (i - mean_x) ** 2 for i in range(n)
    )
    report["slope"] = slope
    ok = slope < 0 and values[-1] <= ratio * peak
    report.update(status="diminishing" if ok else "not-diminishing", **{"pass": ok})
    return report


class TDDriftMonitor:
    """TD(lambda) target drift and error trajectory checks.

    Drift (mean |target change| per refresh) and error (mean |prediction - TD target| on
    the fit rows) are in units of the discount weight W. Rules (approved trajectory rule,
    user 26 Oct): the first ``warmup`` refreshes are ignored; afterwards each series must
    diminish (see :func:`trajectory_report`); a divergence alarm fires on ``growth``
    consecutive increases each larger than ``increase``; any bootstrapped target outside
    [0, W] by more than ``tolerance * W`` alarms at once (QoL cannot leave that range).
    """

    def __init__(
        self,
        weight: float,
        *,
        growth: int,
        floor: float,
        tolerance: float,
        warmup: int = 2,
        ratio: float = 0.5,
        increase: float = 0.10,
    ) -> None:
        self.weight = float(weight)
        self.growth = int(growth)
        self.floor = float(floor)
        self.tolerance = float(tolerance)
        self.warmup = int(warmup)
        self.ratio = float(ratio)
        self.increase = float(increase)
        self.refreshes: list[dict[str, Any]] = []
        self.alarms: list[dict[str, Any]] = []

    def _series(self, key: str) -> list[float]:
        """Post-warm-up values; the epoch-0 refresh replaces the Monte Carlo return and is
        not part of the trajectory (``growth_eligible`` false)."""
        values = [
            r[key] for r in self.refreshes if r.get(key) is not None and r["growth_eligible"]
        ]
        return values[self.warmup :]

    def update(
        self,
        epoch: int,
        previous: torch.Tensor | None,
        current: torch.Tensor,
        *,
        growth_eligible: bool = True,
        error: float | None = None,
        seconds: float | None = None,
    ) -> dict[str, Any] | None:
        """``growth_eligible=False`` records a refresh (e.g. the first one against the
        Monte Carlo return) without counting it toward the trajectory. ``error`` is the
        mean |prediction - TD target| on the fit rows, in the same units as the targets."""
        current = current.detach().double().reshape(-1)
        low = float(current.min()) if current.numel() else 0.0
        high = float(current.max()) if current.numel() else 0.0
        slack = self.tolerance * self.weight
        out_of_bounds = int(((current < -slack) | (current > self.weight + slack)).sum())
        record: dict[str, Any] = {
            "epoch": int(epoch),
            "target_min": low,
            "target_max": high,
            "weight": self.weight,
            "out_of_bounds": out_of_bounds,
            "drift": None,
            "error": None if error is None else float(error) / self.weight,
            "growth_eligible": bool(growth_eligible),
        }
        if seconds is not None:
            record["refresh_seconds"] = float(seconds)
        if previous is not None:
            change = (current - previous.detach().double().reshape(-1)).abs()
            record["drift"] = float(change.mean()) / self.weight if change.numel() else 0.0
            record["drift_max"] = float(change.max()) / self.weight if change.numel() else 0.0
        self.refreshes.append(record)
        alarm = None
        if out_of_bounds:
            alarm = {"check": "td_divergence", "reason": "targets-outside-[0,W]", **record}
        if alarm is None:
            for key in ("drift", "error"):
                tail = self._series(key)[-(self.growth + 1) :]
                if len(tail) == self.growth + 1:
                    rising = all(b > a * (1.0 + self.increase) for a, b in zip(tail, tail[1:]))
                    if rising and tail[-1] > self.floor:
                        alarm = {
                            "check": "td_divergence",
                            "reason": f"{key}-rose->{int(self.increase * 100)}%-for-"
                            f"{self.growth}-consecutive-refreshes",
                            "series": tail,
                            **record,
                        }
                        break
        if alarm is not None:
            self.alarms.append(alarm)
        return alarm

    def trajectory(self) -> dict[str, Any]:
        return {
            key: trajectory_report(self._series(key), ratio=self.ratio, floor=self.floor)
            for key in ("drift", "error")
        }

    def report(self) -> dict[str, Any]:
        trajectory = self.trajectory()
        alarms = list(self.alarms)
        for key, result in trajectory.items():
            if not result["pass"]:
                alarms.append(
                    {
                        "check": "td_divergence",
                        "reason": f"{key}-trajectory-not-diminishing",
                        **result,
                    }
                )
        return {
            "refreshes": list(self.refreshes),
            "alarms": alarms,
            "insufficient": sorted(
                key for key, result in trajectory.items() if result["status"] == "insufficient"
            ),
            "trajectory": trajectory,
            "warmup_refreshes": self.warmup,
            "final_ratio": self.ratio,
            "floor": self.floor,
        }


def saturation_alarm(share: float | None, threshold: float) -> bool:
    return share is not None and float(share) > float(threshold)


def saturation_components(
    z: Any,
    target_fraction: Any,
    early: Any,
    *,
    saz_threshold: float = 6.0,
    thresholds: StabilityThresholds | None = None,
) -> dict[str, Any]:
    """Stratified saturation shares for the bounded QoL head.

    ``z`` are pre-activations, ``target_fraction`` the bounded target G/W per row and
    ``early`` a boolean early-stratum mask. A low-side row (z < -6) with target fraction
    below the expected-low bound is expected and excluded from the unexpected share. The
    alarm fires if either stratum's unexpected share exceeds its threshold. Stratum shares
    are fractions of that stratum's rows; the others are fractions of all rows.
    """
    import torch

    th = thresholds or StabilityThresholds()
    z = torch.as_tensor(z).detach().reshape(-1).double()
    frac = torch.as_tensor(target_fraction).detach().reshape(-1).double()
    early = torch.as_tensor(early).reshape(-1).bool()
    if not (z.numel() == frac.numel() == early.numel()):
        raise ValueError("saturation components need one z, target fraction and flag per row")
    low = z < -saz_threshold
    high = z > saz_threshold
    expected = low & (frac < th.saturation_expected_low_target_fraction)
    unexpected = (low | high) & ~expected

    def share(mask: Any, stratum: Any = None) -> float:
        if stratum is None:
            n = z.numel()
            return float(mask.double().sum() / n) if n else 0.0
        n = int(stratum.sum())
        return float((mask & stratum).double().sum() / n) if n else 0.0

    rest = ~early
    early_unexpected = share(unexpected, early)
    rest_unexpected = share(unexpected, rest)
    return {
        "rows": int(z.numel()),
        "early_rows": int(early.sum()),
        "rest_rows": int(rest.sum()),
        "raw_share": share(low | high),
        "low_share": share(low),
        "high_share": share(high),
        "expected_low_share": share(expected),
        "early_expected_low_share": share(expected, early),
        "rest_expected_low_share": share(expected, rest),
        "early_unexpected_share": early_unexpected,
        "rest_unexpected_share": rest_unexpected,
        "threshold_early": th.saturation_alarm_early,
        "threshold_rest": th.saturation_alarm,
        "expected_low_target_fraction": th.saturation_expected_low_target_fraction,
        "early_decisions": th.saturation_early_decisions,
        "alarm": bool(
            early_unexpected > th.saturation_alarm_early
            or rest_unexpected > th.saturation_alarm
        ),
    }


def overfit_alarm(
    train: Sequence[float | None], heldout: Sequence[float | None], *, gap: float, drop: float
) -> dict[str, Any] | None:
    """Advantage-head overfit from per-epoch train and held-out pair ranking."""
    pairs = [(t, h) for t, h in zip(train, heldout, strict=True) if t is not None and h is not None]
    if not pairs:
        return None
    final_train, final_held = pairs[-1]
    best_held = max(h for _, h in pairs)
    reasons = []
    if final_train - final_held > gap:
        reasons.append(f"train-minus-heldout {final_train - final_held:.4f} > {gap}")
    if best_held - final_held > drop:
        reasons.append(f"heldout fell {best_held - final_held:.4f} from its best > {drop}")
    if not reasons:
        return None
    return {
        "check": "advantage_overfit",
        "final_train": final_train,
        "final_heldout": final_held,
        "best_heldout": best_held,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Probe diagnostics
# ---------------------------------------------------------------------------


def _privileged_rows(encoder: nn.Module) -> int:
    layout = getattr(encoder, "privileged_layout", None)
    return 0 if layout is None else int(sum(f.rows for f in layout.families))


def privileged_attention_mass(
    encoder: nn.Module,
    tensors: Mapping[str, torch.Tensor],
    privileged: Mapping[str, torch.Tensor] | None,
) -> dict[str, Any] | None:
    """Attention mass on privileged keys in every global attention layer.

    For the village token (query 0, the value heads' input) and for all present
    queries: the mean attention probability summed over present privileged
    keys, its uniform-attention baseline (present privileged / present keys)
    and their ratio. ``None`` for an encoder without a privileged block.
    """
    count = _privileged_rows(encoder)
    if count == 0 or privileged is None:
        return None
    layers: list[dict[str, Any]] = []
    pending: list[torch.Tensor | None] = []
    handles = []

    def pre(module, args, kwargs):
        kwargs = dict(kwargs)
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        pending.append(kwargs.get("key_padding_mask"))
        return args, kwargs

    def post(module, args, kwargs, output):
        # Reduce immediately: the full [B, heads, L, L] weights are not kept.
        weights, padding = output[1].detach(), pending.pop()
        present = (
            torch.ones(weights.shape[0], weights.shape[-1], dtype=torch.bool)
            if padding is None
            else ~padding.bool()
        )
        priv_present = present[:, -count:]
        uniform = priv_present.sum(1).double() / present.sum(1).clamp_min(1).double()
        village = weights[:, :, 0, -count:].double().sum(-1).mean(1)  # [B]
        queries = weights[:, :, :, -count:].double().sum(-1).mean(1)  # [B, L]
        q_mask = present.double()
        all_queries = (queries * q_mask).sum(1) / q_mask.sum(1).clamp_min(1)
        rows = uniform > 0
        layers.append(
            {
                "layer": len(layers),
                "village_mass": float(village[rows].mean()) if rows.any() else None,
                "all_query_mass": float(all_queries[rows].mean()) if rows.any() else None,
                "uniform_share": float(uniform[rows].mean()) if rows.any() else None,
            }
        )
        return (output[0], None)

    for layer in encoder.attention.layers:
        handles.append(layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(post, with_kwargs=True))
    was = encoder.training
    encoder.train()  # no dropout; keeps the eval-only fused path (no weights) off
    try:
        with torch.no_grad():
            encoder.context(tensors, privileged)
    finally:
        encoder.train(was)
        for handle in handles:
            handle.remove()
    for record in layers:
        u = record["uniform_share"]
        record["village_ratio"] = None if not u else record["village_mass"] / u
        record["all_query_ratio"] = None if not u else record["all_query_mass"] / u
    ratios = [r["village_ratio"] for r in layers if r["village_ratio"] is not None]
    all_ratios = [r["all_query_ratio"] for r in layers if r["all_query_ratio"] is not None]
    return {
        "privileged_rows": count,
        "rows": int(next(iter(tensors.values())).shape[0]),
        "layers": layers,
        # Village token in any layer, or any query (two-hop reading) in any layer.
        "max_village_ratio": max(ratios) if ratios else None,
        "max_all_query_ratio": max(all_ratios) if all_ratios else None,
    }


def dead_feature_counts(
    encoder: nn.Module,
    tensors: Mapping[str, torch.Tensor],
    privileged: Mapping[str, torch.Tensor] | None,
    *,
    std: float,
) -> dict[str, Any]:
    """Constant privileged input columns (present rows) and constant trunk units."""
    was = encoder.training
    encoder.train()
    try:
        with torch.no_grad():
            hidden = encoder.context(tensors, privileged)["hidden"].double()
    finally:
        encoder.train(was)
    unit_std = (
        hidden.std(0, unbiased=False) if hidden.shape[0] > 1 else torch.zeros(hidden.shape[1])
    )
    report: dict[str, Any] = {
        "trunk_units": int(hidden.shape[1]),
        "dead_trunk_units": int((unit_std <= std).sum()),
        "privileged": None,
    }
    layout = getattr(encoder, "privileged_layout", None)
    if layout is not None and privileged is not None:
        from .labor_model import PRIVILEGED_CRITIC_PREFIX

        families = {}
        for family in layout.families:
            values = privileged[PRIVILEGED_CRITIC_PREFIX + family.name].double()
            mask = privileged[PRIVILEGED_CRITIC_PREFIX + family.name + "_present"].bool()
            rows = values[mask]
            if rows.shape[0] < 2:
                families[family.name] = {
                    "columns": int(family.width),
                    "present_rows": int(rows.shape[0]),
                    "dead_columns": None,
                }
                continue
            column_std = rows.std(0, unbiased=False)
            families[family.name] = {
                "columns": int(family.width),
                "present_rows": int(rows.shape[0]),
                "dead_columns": int((column_std <= std).sum()),
            }
        report["privileged"] = families
    return report


def population_tail_report(
    task_critic: nn.Module, logits: torch.Tensor, observed_max: float
) -> dict[str, Any]:
    """Mean decoded tail terms of the distributional population head (reference scales)."""
    terms = task_critic.support_diagnostics("population_years", logits, observed_max)
    return {
        "observed_normalized_max": float(observed_max),
        **{name: float(value.mean()) for name, value in terms.items()},
        "upper_excess_max": float(terms["upper_excess"].max())
        if terms["upper_excess"].numel()
        else 0.0,
        "rows": int(logits.shape[0]),
    }


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


def _norm(tensors: Iterable[torch.Tensor | None]) -> float:
    total = 0.0
    for tensor in tensors:
        if tensor is not None:
            total += float(tensor.detach().double().square().sum())
    return math.sqrt(total)


def module_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(
            tensor.numpy().tobytes()
            if tensor.dtype != torch.bfloat16
            else tensor.float().numpy().tobytes()
        )
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    return value


class StabilityMonitor:
    """Guarded optimizer steps and the per-fit stability receipt.

    ``blocks`` names the trained parameter groups reported separately (their
    union must be exactly the clipped parameter list). ``frozen`` modules must
    receive no gradient at all (None or exactly zero) and keep their digest.
    ``shared`` maps a shared block to its contenders: each contender is either
    a head whose gradient on that block is computed from its own loss, or (for
    the global clip, ``contender_blocks``) a block whose own pre-clip norm is its
    contribution.
    """

    def __init__(
        self,
        config: StabilityConfig,
        *,
        fit: str,
        optimizer: torch.optim.Optimizer,
        parameters: Sequence[nn.Parameter],
        blocks: Mapping[str, Sequence[nn.Parameter]],
        frozen: Mapping[str, nn.Module] | None = None,
        shared_heads: Mapping[str, tuple[Sequence[nn.Parameter], Sequence[str]]] | None = None,
        contender_blocks: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
        applicable: Mapping[str, bool] | None = None,
    ) -> None:
        self.config = config
        self.t = config.thresholds
        self.fit = str(fit)
        self.optimizer = optimizer
        self.parameters = list(parameters)
        self.blocks = {name: list(params) for name, params in blocks.items()}
        covered = {id(p) for params in self.blocks.values() for p in params}
        if covered != {id(p) for p in self.parameters} or sum(
            len(p) for p in self.blocks.values()
        ) != len(self.parameters):
            raise ValueError("stability blocks must partition the trained parameters")
        self.frozen = dict(frozen or {})
        frozen_ids = {id(p) for m in self.frozen.values() for p in m.parameters()}
        if frozen_ids & covered:
            raise ValueError("a frozen module shares parameters with the trained blocks")
        self.frozen_digests = {name: module_digest(m) for name, m in self.frozen.items()}
        for module in self.frozen.values():
            module.zero_grad(set_to_none=True)
        self.shared_heads = {
            name: (list(params), tuple(heads))
            for name, (params, heads) in (shared_heads or {}).items()
        }
        self.contender_blocks = {
            k: {c: tuple(b) for c, b in v.items()} for k, v in (contender_blocks or {}).items()
        }
        for groups in self.contender_blocks.values():
            unknown = {b for blocks_ in groups.values() for b in blocks_} - set(self.blocks)
            if unknown:
                raise ValueError(f"contender blocks name unknown blocks {sorted(unknown)}")
        th = self.t
        self.detectors: dict[str, DominanceDetector] = {}
        for name, (_, heads) in self.shared_heads.items():
            self.detectors[name] = DominanceDetector(
                name,
                heads,
                window=th.dominance_window,
                share=th.dominance_share,
                persistence=th.dominance_persistence,
                min_progress=th.dominance_min_progress,
            )
        for name, contenders in self.contender_blocks.items():
            self.detectors[name] = DominanceDetector(
                name,
                tuple(contenders),
                window=th.dominance_window,
                share=th.dominance_share,
                persistence=th.dominance_persistence,
                min_progress=th.dominance_min_progress,
            )
        self.applicable = {name: False for name in MANDATORY_CHECKS + ADVISORY_CHECKS}
        self.applicable.update({"finite_updates": True, "frozen_gradients_zero": bool(self.frozen)})
        self.applicable["gradient_dominance"] = bool(self.detectors)
        self.applicable.update(dict(applicable or {}))
        self.step_index = 0
        self.successful_steps = 0
        self.rollbacks = 0
        self.consecutive = 0
        self.events: list[dict[str, Any]] = []
        self.alarms: list[dict[str, Any]] = []
        self.frozen_violations: list[dict[str, Any]] = []
        self.td: TDDriftMonitor | None = None
        self.saturation: list[dict[str, Any]] = []
        self.population_tail: list[dict[str, Any]] = []
        self.epoch_rankings: list[dict[str, Any]] = []
        self.probes: dict[str, Any] = {}
        self.final: dict[str, Any] = {}
        self.lr_initial = [float(g["lr"]) for g in optimizer.param_groups]
        self.started = time.time()
        self._jsonl = None
        if config.jsonl:
            path = Path(config.jsonl)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl = path.open("a", encoding="utf-8")
        self._snapshot()
        self.write(
            {
                "kind": "start",
                "schema": STABILITY_SCHEMA,
                "config": config.state_dict(),
                "blocks": {k: sum(p.numel() for p in v) for k, v in self.blocks.items()},
                "frozen": self.frozen_digests,
            }
        )

    # ---- state -----------------------------------------------------------
    def _snapshot(self) -> None:
        self._good_params = [p.detach().clone() for p in self.parameters]
        self._good_optimizer = copy.deepcopy(self.optimizer.state_dict())

    def _restore(self) -> None:
        with torch.no_grad():
            for parameter, saved in zip(self.parameters, self._good_params, strict=True):
                parameter.copy_(saved)
        lrs = [float(g["lr"]) for g in self.optimizer.param_groups]
        self.optimizer.load_state_dict(copy.deepcopy(self._good_optimizer))
        for group, lr in zip(self.optimizer.param_groups, lrs, strict=True):
            group["lr"] = lr * self.t.lr_backoff
        # The saved state keeps the halved rate on any later rollback.
        for group, saved in zip(
            self.optimizer.param_groups, self._good_optimizer["param_groups"], strict=True
        ):
            saved["lr"] = group["lr"]
        self.optimizer.zero_grad(set_to_none=True)
        for module in self.frozen.values():
            module.zero_grad(set_to_none=True)

    def _optimizer_finite(self) -> bool:
        for state in self.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    if not bool(torch.isfinite(value).all()):
                        return False
        return True

    def _parameters_finite(self) -> bool:
        return all(bool(torch.isfinite(p).all()) for p in self.parameters)

    def write(self, record: Mapping[str, Any]) -> None:
        if self._jsonl is None:
            return
        line = {"fit": self.fit, "t": round(time.time() - self.started, 3), **record}
        self._jsonl.write(json.dumps(_jsonable(line), sort_keys=True) + "\n")
        self._jsonl.flush()

    def close(self) -> None:
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None

    def _alarm(self, alarm: Mapping[str, Any]) -> None:
        record = {"step": self.step_index, **alarm}
        self.alarms.append(record)
        self.write({"kind": "alarm", **record})

    # ---- the guarded step --------------------------------------------------
    def is_log_step(self) -> bool:
        return self.step_index % self.config.log_every == 0

    def step(
        self,
        loss: torch.Tensor,
        *,
        max_grad_norm: float,
        head_losses: Mapping[str, torch.Tensor] | None = None,
        diagnostics: Callable[[], Mapping[str, Any]] | None = None,
    ) -> bool:
        """zero_grad was called by the caller; backward, clip, step, check, maybe roll back.

        ``head_losses`` are scalar losses (same graph as ``loss``) of the shared
        blocks' head contenders and of the contender blocks; they are read only
        on log steps. Returns False when the step was rolled back.
        """
        self.step_index += 1
        log = self.is_log_step()
        reason = None
        record: dict[str, Any] = {"kind": "step", "step": self.step_index}
        if not bool(torch.isfinite(loss.detach()).all()):
            reason = "nonfinite-loss"
        head_values: dict[str, float] = {}
        shared_norms: dict[str, dict[str, float]] = {}
        if reason is None and log and head_losses is not None:
            head_values = {k: float(v.detach()) for k, v in head_losses.items()}
            for block, (params, heads) in self.shared_heads.items():
                norms = {}
                for head in heads:
                    grads = torch.autograd.grad(
                        head_losses[head], params, retain_graph=True, allow_unused=True
                    )
                    norms[head] = _norm(grads)
                shared_norms[block] = norms
        if reason is None:
            loss.backward()
            if self.step_index in self.config.inject_nonfinite_grad_steps:
                with torch.no_grad():
                    self.parameters[0].grad = torch.full_like(self.parameters[0], math.nan)
            block_norms = (
                {name: _norm(p.grad for p in params) for name, params in self.blocks.items()}
                if log
                else {}
            )
            total = nn.utils.clip_grad_norm_(self.parameters, max_grad_norm)
            total = float(total)
            if not math.isfinite(total):
                reason = "nonfinite-gradient"
        frozen_norms: dict[str, float] = {}
        if reason is None:
            for name, module in self.frozen.items():
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                frozen_norms[name] = _norm(grads)
                if grads and any(bool((g != 0).any()) for g in grads):
                    violation = {"module": name, "step": self.step_index, "norm": _norm(grads)}
                    self.frozen_violations.append(violation)
                    self._alarm({"check": "frozen_gradients_zero", **violation})
                    module.zero_grad(set_to_none=True)
            self.optimizer.step()
            if self.step_index in self.config.inject_nonfinite_param_steps:
                with torch.no_grad():
                    self.parameters[0].view(-1)[0] = math.nan
            if not self._parameters_finite():
                reason = "nonfinite-parameter"
            elif not self._optimizer_finite():
                reason = "nonfinite-optimizer-state"
        if reason is not None:
            self.rollbacks += 1
            self.consecutive += 1
            self._restore()
            event = {
                "kind": "rollback",
                "step": self.step_index,
                "reason": reason,
                "consecutive": self.consecutive,
                "lr_after": [float(g["lr"]) for g in self.optimizer.param_groups],
                "injected": self.step_index
                in (
                    *self.config.inject_nonfinite_grad_steps,
                    *self.config.inject_nonfinite_param_steps,
                ),
            }
            self.events.append(event)
            self.write(event)
            if self.consecutive >= self.t.max_consecutive_failures:
                summary = self.summary()
                self.write({"kind": "abort", "summary": summary})
                self.close()
                raise StabilityAbort(
                    f"{self.fit}: {self.consecutive} consecutive nonfinite steps "
                    f"(K={self.t.max_consecutive_failures}); last {reason}",
                    summary,
                )
            return False
        self.consecutive = 0
        self.successful_steps += 1
        if log:
            saved = {id(p): s for p, s in zip(self.parameters, self._good_params, strict=True)}
            steps = {
                name: _norm(p.detach() - saved[id(p)] for p in params)
                for name, params in self.blocks.items()
            }
            record.update(
                {
                    "loss": float(loss.detach()),
                    "head_losses": head_values,
                    "grad_norm_preclip": block_norms,
                    "grad_norm_global": total,
                    "clip_multiplier": min(1.0, max_grad_norm / (total + 1e-6)),
                    "param_step_norm": steps,
                    # Measured before the step (must be exactly 0.0; a violation alarms).
                    "frozen_grad_norm": frozen_norms,
                    "shared_head_grad_norm": shared_norms,
                    "lr": [float(g["lr"]) for g in self.optimizer.param_groups],
                }
            )
            for block, norms in shared_norms.items():
                alarm = self.detectors[block].update(self.step_index, norms, head_values)
                if alarm:
                    self._alarm(alarm)
            contender_norms = {}
            for block, contenders in self.contender_blocks.items():
                norms = {
                    c: math.sqrt(sum(block_norms[b] ** 2 for b in members))
                    for c, members in contenders.items()
                }
                contender_norms[block] = norms
                if all(c in head_values for c in contenders):
                    alarm = self.detectors[block].update(
                        self.step_index, norms, {c: head_values[c] for c in contenders}
                    )
                    if alarm:
                        self._alarm(alarm)
            record["contender_grad_norm"] = contender_norms
            if diagnostics is not None:
                extra = dict(diagnostics())
                record.update(extra)
                self._record_diagnostics(extra)
            self.write(record)
        self._snapshot()
        return True

    def _record_diagnostics(self, extra: Mapping[str, Any]) -> None:
        if "qol_saturation" in extra and extra["qol_saturation"] is not None:
            self.saturation.append({"step": self.step_index, "share": extra["qol_saturation"]})
        if "population_tail" in extra and extra["population_tail"] is not None:
            self.population_tail.append({"step": self.step_index, **extra["population_tail"]})

    # ---- fit-level records -------------------------------------------------
    def _new_td(self, weight: float) -> TDDriftMonitor:
        return TDDriftMonitor(
            weight,
            growth=self.t.td_growth_refreshes,
            floor=self.t.td_drift_floor,
            tolerance=self.t.td_bound_tolerance,
            warmup=self.t.td_warmup_refreshes,
            ratio=self.t.td_final_ratio,
            increase=self.t.td_divergence_increase,
        )

    def td_refresh(
        self,
        epoch: int,
        previous: torch.Tensor | None,
        current: torch.Tensor,
        weight: float,
        *,
        growth_eligible: bool = True,
        error: float | None = None,
        seconds: float | None = None,
    ) -> None:
        if self.td is None:
            self.td = self._new_td(weight)
            self.applicable["td_divergence"] = True
        alarm = self.td.update(
            epoch,
            previous,
            current,
            growth_eligible=growth_eligible,
            error=error,
            seconds=seconds,
        )
        self.write({"kind": "td_refresh", **self.td.refreshes[-1]})
        if alarm:
            self._alarm(alarm)

    def epoch_ranking(
        self, epoch: int, train: float | None, heldout: float | None, **extra: Any
    ) -> None:
        self.applicable["advantage_overfit"] = True
        record = {"epoch": int(epoch), "train": train, "heldout": heldout, **extra}
        self.epoch_rankings.append(record)
        self.write({"kind": "epoch_ranking", **record})

    def record_probe(self, name: str, value: Any) -> None:
        self.probes[name] = value
        self.write({"kind": "probe", "name": name, "value": value})

    def record_final(self, name: str, value: Any) -> None:
        self.final[name] = value
        self.write({"kind": "final", "name": name, "value": value})

    # ---- resume ------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Monitor history for an exact mid-fit resume (plain Python values)."""
        return {
            "step_index": self.step_index,
            "successful_steps": self.successful_steps,
            "rollbacks": self.rollbacks,
            "consecutive": self.consecutive,
            "events": copy.deepcopy(self.events),
            "alarms": _jsonable(self.alarms),
            "frozen_violations": copy.deepcopy(self.frozen_violations),
            "saturation": copy.deepcopy(self.saturation),
            "population_tail": copy.deepcopy(self.population_tail),
            "epoch_rankings": copy.deepcopy(self.epoch_rankings),
            "td": None
            if self.td is None
            else {
                "weight": self.td.weight,
                "refreshes": _jsonable(self.td.refreshes),
                "alarms": _jsonable(self.td.alarms),
            },
            "detectors": {
                name: {
                    "points": copy.deepcopy(d.points),
                    "alarms": _jsonable(d.alarms),
                    "dominant_windows": d.dominant_windows,
                }
                for name, d in self.detectors.items()
            },
            "lr_initial": list(self.lr_initial),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key in ("step_index", "successful_steps", "rollbacks", "consecutive"):
            setattr(self, key, int(state[key]))
        for key in (
            "events",
            "alarms",
            "frozen_violations",
            "saturation",
            "population_tail",
            "epoch_rankings",
        ):
            setattr(self, key, list(state[key]))
        self.lr_initial = [float(v) for v in state["lr_initial"]]
        if state["td"] is not None:
            self.td = self._new_td(state["td"]["weight"])
            self.td.refreshes = list(state["td"]["refreshes"])
            self.td.alarms = list(state["td"]["alarms"])
            self.applicable["td_divergence"] = True
        for name, saved in state["detectors"].items():
            detector = self.detectors[name]
            detector.points = list(saved["points"])
            detector.alarms = list(saved["alarms"])
            detector.dominant_windows = int(saved["dominant_windows"])
        if self.epoch_rankings:
            self.applicable["advantage_overfit"] = True
        self._snapshot()
        self.write({"kind": "resumed", "step": self.step_index})

    # ---- verdict -----------------------------------------------------------
    def checks(self) -> dict[str, dict[str, Any]]:
        th = self.t
        out: dict[str, dict[str, Any]] = {}

        def put(name: str, status: str, **detail: Any) -> None:
            if not self.applicable.get(name, False):
                status = "NOT_APPLICABLE"
            out[name] = {
                "mandatory": name in MANDATORY_CHECKS,
                "status": status,
                **detail,
            }

        aborted = self.consecutive >= th.max_consecutive_failures
        put(
            "finite_updates",
            "FAIL" if aborted or self.rollbacks > th.max_total_rollbacks else "PASS",
            rollbacks=self.rollbacks,
            max_total_rollbacks=th.max_total_rollbacks,
            aborted=aborted,
            fault_injection=self.config.fault_injection,
        )
        digests_ok = all(
            module_digest(m) == self.frozen_digests[name] for name, m in self.frozen.items()
        )
        put(
            "frozen_gradients_zero",
            "PASS" if not self.frozen_violations and digests_ok else "FAIL",
            violations=self.frozen_violations,
            digests_unchanged=digests_ok,
            modules=sorted(self.frozen),
        )
        dominance_alarms = [a for d in self.detectors.values() for a in d.alarms]
        put(
            "gradient_dominance",
            "FAIL" if dominance_alarms else "PASS",
            detectors={k: d.report() for k, d in self.detectors.items()},
        )
        final_sat = self.final.get("qol_saturation")
        detail = self.final.get("qol_saturation_detail")
        if final_sat is None:
            sat_status = "HOLD"
        elif detail is not None:
            sat_status = "FAIL" if detail["alarm"] else "PASS"
        else:
            sat_status = "FAIL" if saturation_alarm(final_sat, th.saturation_alarm) else "PASS"
        put(
            "qol_saturation",
            sat_status,
            final_share=final_sat,
            threshold=th.saturation_alarm,
            threshold_early=th.saturation_alarm_early,
            components=detail,
            trajectory=self.saturation[-20:],
        )
        td = self.td.report() if self.td is not None else None
        # Too few post-warm-up refreshes to judge a trajectory is missing evidence: HOLD,
        # never a silent pass (FAIL still dominates).
        td_status = (
            "HOLD"
            if td is None
            else "FAIL"
            if td["alarms"]
            else "HOLD"
            if td["insufficient"]
            else "PASS"
        )
        put(
            "td_divergence",
            td_status,
            **(
                {
                    "refreshes": td["refreshes"],
                    "alarms": td["alarms"],
                    "trajectory": td["trajectory"],
                    "insufficient": td["insufficient"],
                    "warmup_refreshes": td["warmup_refreshes"],
                }
                if td
                else {}
            ),
        )
        tail = self.final.get("population_tail")
        put(
            "population_tail",
            "HOLD"
            if tail is None
            else "FAIL"
            if tail["upper_excess"] > th.upper_excess_alarm
            or tail["top_bin_mass"] > th.top_bin_mass_alarm
            else "PASS",
            final=tail,
            upper_excess_alarm=th.upper_excess_alarm,
            top_bin_mass_alarm=th.top_bin_mass_alarm,
            trajectory=self.population_tail[-20:],
        )
        overfit = overfit_alarm(
            [r["train"] for r in self.epoch_rankings],
            [r["heldout"] for r in self.epoch_rankings],
            gap=th.overfit_gap,
            drop=th.overfit_heldout_drop,
        )
        held_seen = any(r["heldout"] is not None for r in self.epoch_rankings)
        put(
            "advantage_overfit",
            "FAIL" if overfit else "PASS" if held_seen else "HOLD",
            alarm=overfit,
            epochs=self.epoch_rankings,
        )
        attention = self.probes.get("privileged_attention")
        if attention is None:
            status, low = "HOLD", None
        else:
            ratios = {}
            for name, rec in attention.items():
                values = (
                    []
                    if rec is None
                    else [
                        v
                        for v in (rec["max_village_ratio"], rec["max_all_query_ratio"])
                        if v is not None
                    ]
                )
                ratios[name] = max(values) if values else None
            low = {k: v for k, v in ratios.items() if v is None or v < th.attention_ratio_floor}
            status = "FAIL" if low else "PASS"
        put(
            "privileged_attention",
            status,
            encoders=attention,
            below_floor=low,
            floor=th.attention_ratio_floor,
        )
        dead = self.probes.get("dead_features")
        put("dead_features", "PASS" if dead is not None else "HOLD", encoders=dead)
        return out

    def summary(self) -> dict[str, Any]:
        checks = self.checks()
        mandatory = {
            k: v for k, v in checks.items() if v["mandatory"] and v["status"] != "NOT_APPLICABLE"
        }
        failed = sorted(k for k, v in mandatory.items() if v["status"] == "FAIL")
        held = sorted(k for k, v in mandatory.items() if v["status"] == "HOLD")
        return _jsonable(
            {
                "schema": STABILITY_SCHEMA,
                "fit": self.fit,
                "config": self.config.state_dict(),
                "proposed_thresholds": list(PROPOSED_THRESHOLDS),
                "steps": self.step_index,
                "successful_steps": self.successful_steps,
                "rollbacks": self.rollbacks,
                "lr_initial": self.lr_initial,
                "lr_final": [float(g["lr"]) for g in self.optimizer.param_groups],
                "events": self.events,
                "alarms": self.alarms,
                "checks": checks,
                "mandatory_failed": failed,
                "mandatory_hold": held,
                "admitted": not failed and not held and not self.config.fault_injection,
                "jsonl": self.config.jsonl,
                "fault_injection": self.config.fault_injection,
                "seconds": round(time.time() - self.started, 3),
            }
        )

    def finish(self) -> dict[str, Any]:
        summary = self.summary()
        self.write({"kind": "summary", "summary": summary})
        self.close()
        return summary


def parameter_blocks(named: Mapping[str, nn.Module]) -> dict[str, list[nn.Parameter]]:
    return {name: list(module.parameters()) for name, module in named.items()}


def split_frozen(modules: Mapping[str, nn.Module | None]) -> dict[str, nn.Module]:
    return {name: module for name, module in modules.items() if module is not None}


def stability_config_from_args(
    args: Any, default_jsonl: str | os.PathLike | None = None
) -> StabilityConfig | str | None:
    """CLI helper: ``--no-stability`` / ``--stability-log`` / ``--stability-every``."""
    if getattr(args, "no_stability", False):
        return None
    jsonl = getattr(args, "stability_log", None) or default_jsonl
    every = getattr(args, "stability_every", None)
    if jsonl is None and every is None:
        return "auto"
    return StabilityConfig(log_every=int(every or 25), jsonl=None if jsonl is None else str(jsonl))


def add_stability_arguments(parser: Any) -> None:
    parser.add_argument("--no-stability", action="store_true", help="disable the stability monitor")
    parser.add_argument("--stability-log", type=Path, default=None, help="stability JSONL path")
    parser.add_argument("--stability-every", type=int, default=None, help="log every N steps (25)")
