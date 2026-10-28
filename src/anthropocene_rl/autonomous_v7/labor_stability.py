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
4. **TD(lambda) target drift** per refresh, with the divergence alarm (trajectory rule v2,
   ``TD_RULE_V2``, for new fits since 28 Oct; resume states without a rule keep v1).
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
import itertools
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

from .labor_device import all_finite

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
# Live early stop (``StabilityConfig.early_stop``): conditions that are already a mandatory
# FAIL under the end-of-run judgement stop the fit the moment they trip, under the check id
# ``checks()`` FAILs for them: a gradient_dominance alarm, a live td_divergence alarm
# (rule v2: error growth ending above 1.5 x its stationary median; v1: growth after
# warm-up; a target outside its per-row admissible bound, a reward outside
# [0, dt], a discount other than exp(-elapsed / tau) or a decoded head value outside its
# range; receipts written before 27 Oct carry the flat ``targets-outside-[0,W]`` reason,
# which still stops and FAILs) and more than max_total_rollbacks
# rollbacks (finite_updates). QoL saturation and the population tail stay end-of-run
# judgements; the population tail adds a live, non-gating alarm.
EARLY_STOP_CHECKS = ("gradient_dominance", "td_divergence", "finite_updates")
# Held-out validation (monitoring only): the advisory alarm fires when the validation total
# loss rises on this many consecutive epochs while the training loss falls.
VALIDATION_DIVERGENCE_EPOCHS = 2


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
    td_bound_tolerance: float = 1e-6  # hard target-bound slack (relative), see TDDriftMonitor
    #    Rule v2 (user 28 Oct, ``TD_RULE_V2``): the v1 trajectory rule failed fits whose drift
    #    and error had reached a stationary noise floor. Warm-up by value; PASS on a declining
    #    or a stationary-below-floor ERROR series (drift advisory); divergence relative to the
    #    stationary median. ``td_rule`` selects the rule; the v1 fields above stay in force for
    #    v1 (resume states written before 28 Oct, and their receipts, keep their verdicts).
    td_rule: str = "v2-floor-or-declining"
    td_warmup_median_factor: float = 1.5  # warm-up ends at the first value <= this x median
    td_stationary_refreshes: int = 6  # K: the stationary window and the minimum usable series
    td_stationary_spread: float = 0.5  # max - min over the window <= this x the window median
    # Declared floors (fractions of W) of the stationary branch: approved 28 Oct, to be
    # recalibrated from the first clean fit.
    td_stationary_error_floor: float = 0.05
    td_stationary_drift_floor: float = 0.05
    td_divergence_median_factor: float = 1.5  # a rising run must end above this x median
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
    if name not in ("td_growth_refreshes", "td_rule")
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
    # Live early stop on would-FAIL conditions (EARLY_STOP_CHECKS); off in the library,
    # on in labor_r5_acceptance run. Same thresholds and definitions as ``checks()``.
    early_stop: bool = False
    # ``--dominance-excess-loss``: the shared-head dominance detector judges the bounded
    # logistic QoL head's progress on its excess BCE (BCE - H(p*)) instead of the raw BCE,
    # whose entropy floor makes relative progress structurally tiny. Off by default (the
    # approved check's measurement is unchanged); recorded only when on.
    dominance_excess_loss: bool = False

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
            # Recorded only when on: receipts of fits without it are unchanged.
            **({"early_stop": True} if self.early_stop else {}),
            **({"dominance_excess_loss": True} if self.dominance_excess_loss else {}),
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


# TD trajectory rules. v1 (user 26 Oct): count warm-up, diminishing-or-below-1e-3 on both
# series, growth alarm on either. v2 (user 28 Oct): see :func:`trajectory_report_v2`.
TD_RULE_V1 = "v1-diminishing"
TD_RULE_V2 = "v2-floor-or-declining"
TD_RULES = (TD_RULE_V1, TD_RULE_V2)
# v2: the error series gates; the drift series is reported with an advisory flag only.
TD_V2_GATED_SERIES = "error"
TD_V2_ADVISORY_SERIES = "drift"


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of an empty series")
    middle = n // 2
    return ordered[middle] if n % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def _log_slope(values: Sequence[float]) -> float | None:
    """Least-squares slope of log(value) over the index (the v1 slope definition)."""
    n = len(values)
    if n < 2:
        return None
    logs = [math.log(max(float(v), 1e-12)) for v in values]
    mean_x = (n - 1) / 2.0
    mean_y = sum(logs) / n
    return sum((i - mean_x) * (y - mean_y) for i, y in enumerate(logs)) / sum(
        (i - mean_x) ** 2 for i in range(n)
    )


def td_warmup_by_value(series: Sequence[float], factor: float) -> tuple[int, float | None]:
    """v2 warm-up: the number of leading refreshes dropped and the running median.

    The running median is the median of every (growth-eligible) value observed so far, so
    it is recomputed at each refresh while the fit runs and equals the whole-series median
    at the end. Leading values are dropped until the first one within ``factor`` x that
    median (value <= factor x median); later values are never dropped. A series that never
    starts high has no warm-up.
    """
    values = [float(v) for v in series]
    if not values:
        return 0, None
    median = _median(values)
    for index, value in enumerate(values):
        if value <= factor * median:
            return index, median
    return len(values), median  # unreachable: some value is <= the median


def trajectory_report_v2(
    series: Sequence[float],
    *,
    warmup_factor: float,
    window: int,
    spread: float,
    floor: float,
    ratio: float,
) -> dict[str, Any]:
    """Rule v2 on one growth-eligible series (fractions of W).

    Warm-up by value (:func:`td_warmup_by_value`). Fewer than ``window`` (K) usable
    refreshes after it: ``insufficient`` (HOLD, no alarm). Otherwise PASS when either
    (a) **declining**: the least-squares slope of log(value) over the post-warm-up series
    is <= 0 and the final value is <= ``ratio`` x the post-warm-up peak, or (b)
    **stationary**: over the last K refreshes max - min <= ``spread`` x their median and
    that median <= ``floor`` (the declared floor, in units of W). Else not passing.
    """
    values = [float(v) for v in series]
    dropped, median = td_warmup_by_value(values, warmup_factor)
    post = values[dropped:]
    report: dict[str, Any] = {
        "rule": TD_RULE_V2,
        "series": values,
        "warmup_dropped": dropped,
        "warmup_median": median,
        "post_warmup": post,
        "slope": None,
        "final_over_peak": None,
        "declining": None,
        "window": int(window),
        "window_median": None,
        "window_spread_over_median": None,
        "floor": float(floor),
        "stationary": None,
        "status": "insufficient",
        "pass": True,
    }
    if len(post) < int(window):
        return report
    peak = max(post)
    slope = _log_slope(post)
    final_over_peak = post[-1] / peak if peak > 0 else 0.0
    declining = slope is not None and slope <= 0 and post[-1] <= ratio * peak
    last = post[-int(window) :]
    window_median = _median(last)
    width = max(last) - min(last)
    stationary = width <= spread * window_median and window_median <= floor
    report.update(
        slope=slope,
        final_over_peak=final_over_peak,
        declining=bool(declining),
        window_median=window_median,
        window_spread_over_median=(width / window_median) if window_median > 0 else 0.0,
        stationary=bool(stationary),
    )
    if declining:
        report.update(status="declining", **{"pass": True})
    elif stationary:
        report.update(status="stationary-below-floor", **{"pass": True})
    else:
        report.update(status="not-declining-or-stationary", **{"pass": False})
    return report


def td_divergence_v2(
    series: Sequence[float],
    *,
    warmup_factor: float,
    growth: int,
    increase: float,
    median_factor: float,
) -> dict[str, Any] | None:
    """Rule v2 divergence on one growth-eligible series, or None.

    After the warm-up by value: the last ``growth`` refreshes each rose by more than
    ``increase`` (relative) and the last one ends above ``median_factor`` x the stationary
    median, the median of the post-warm-up values up to and including the base of the
    rising run (the level the series held before it started to rise).
    """
    values = [float(v) for v in series]
    dropped, _ = td_warmup_by_value(values, warmup_factor)
    post = values[dropped:]
    run = post[-(growth + 1) :]
    if len(run) < growth + 1:
        return None
    if not all(b > a * (1.0 + increase) for a, b in itertools.pairwise(run)):
        return None
    base = post[: len(post) - growth]
    reference = _median(base)
    if not run[-1] > median_factor * reference:
        return None
    return {"series": run, "stationary_median": reference, "warmup_dropped": dropped}


# Live td_divergence reasons of the hard checks, in the order they are reported (the
# first violated one names the alarm; every count is in the record).
TD_REASON_FLAT_BOUND = "targets-outside-[0,W]"  # historical; callers without a bound
TD_REASON_REWARD = "rewards-outside-[0,dt]"
TD_REASON_DISCOUNT = "discount-not-exp(-elapsed/tau)"
TD_REASON_HEAD_RANGE = "head-outside-range"
TD_REASON_ADMISSIBLE = "targets-outside-admissible-bound"
TD_HARD_REASONS = (
    TD_REASON_FLAT_BOUND,
    TD_REASON_REWARD,
    TD_REASON_DISCOUNT,
    TD_REASON_HEAD_RANGE,
    TD_REASON_ADMISSIBLE,
)


class TDDriftMonitor:
    """TD(lambda) target drift and error trajectory checks.

    Drift (mean |target change| per refresh) and error (mean |prediction - TD target| on
    the fit rows) are in units of the discount weight W. ``rule`` selects the trajectory
    rule; the hard bounds below are the same under both.

    Rule v1 (``TD_RULE_V1``, user 26 Oct; the constructor default, so direct callers and
    resume states without a recorded rule are unchanged): the first ``warmup`` refreshes
    are ignored; afterwards each series must diminish (see :func:`trajectory_report`); a
    divergence alarm fires on ``growth`` consecutive increases each larger than
    ``increase`` on either series.

    Rule v2 (``TD_RULE_V2``, user 28 Oct; what :class:`StabilityMonitor` uses by default):
    warm-up by value (:func:`td_warmup_by_value`); the ERROR series is gated by
    :func:`trajectory_report_v2` (declining, or stationary below ``error_floor``; fewer
    than ``window`` usable refreshes is insufficient: HOLD); the live divergence alarm
    (:func:`td_divergence_v2`) fires on ``growth`` consecutive rises of more than
    ``increase`` that end above ``median_factor`` x the stationary median. The DRIFT series
    gets the same report (floor ``drift_floor``) and the same divergence test, but only as
    advisories (``advisories``), never alarms.

    Hard bound (user decision 27 Oct). With ``admissibility`` (from
    ``labor_advantage.qol_td_admissibility``) every target must lie in [0, U_i], where
    U_i is the same backward TD(lambda) recursion as the target with the head's maximum
    for every bootstrapped V and max(tail, head maximum) for the tail after the last row;
    a target below ``-tolerance * W`` or above ``U_i (1 + tolerance)`` alarms at once
    (``targets-outside-admissible-bound``). The randomized-roulette tail is unbiased but
    unbounded (weights gamma / rho > 1), so a flat [0, W] bound fired on correct
    estimator noise; the per-row bound admits it, while the real bugs still alarm through
    the same record: a reward outside [0, dt], a discount other than exp(-elapsed / tau)
    and a decoded head value outside its range. The share of rows whose tail exceeds W
    and the largest tail / W are recorded as diagnostics, never gating. Without
    ``admissibility`` (historical callers) the flat [0, W] check runs unchanged under its
    original reason ``targets-outside-[0,W]``.
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
        rule: str = TD_RULE_V1,
        warmup_factor: float = 1.5,
        window: int = 6,
        spread: float = 0.5,
        error_floor: float = 0.05,
        drift_floor: float = 0.05,
        median_factor: float = 1.5,
    ) -> None:
        if rule not in TD_RULES:
            raise ValueError(f"unknown TD trajectory rule {rule!r}; expected one of {TD_RULES}")
        self.weight = float(weight)
        self.growth = int(growth)
        self.floor = float(floor)
        self.tolerance = float(tolerance)
        self.warmup = int(warmup)
        self.ratio = float(ratio)
        self.increase = float(increase)
        self.rule = rule
        self.warmup_factor = float(warmup_factor)
        self.window = int(window)
        self.spread = float(spread)
        self.error_floor = float(error_floor)
        self.drift_floor = float(drift_floor)
        self.median_factor = float(median_factor)
        self.refreshes: list[dict[str, Any]] = []
        self.alarms: list[dict[str, Any]] = []
        self.advisories: list[dict[str, Any]] = []

    def _eligible(self, key: str) -> list[float]:
        """Growth-eligible values; the epoch-0 refresh replaces the Monte Carlo return and
        is not part of the trajectory (``growth_eligible`` false)."""
        return [r[key] for r in self.refreshes if r.get(key) is not None and r["growth_eligible"]]

    def _series(self, key: str) -> list[float]:
        """v1: post-warm-up values (the first ``warmup`` eligible refreshes dropped)."""
        return self._eligible(key)[self.warmup :]

    def _v2_floor(self, key: str) -> float:
        return self.error_floor if key == "error" else self.drift_floor

    def update(
        self,
        epoch: int,
        previous: torch.Tensor | None,
        current: torch.Tensor,
        *,
        growth_eligible: bool = True,
        error: float | None = None,
        seconds: float | None = None,
        admissibility: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """``growth_eligible=False`` records a refresh (e.g. the first one against the
        Monte Carlo return) without counting it toward the trajectory. ``error`` is the
        mean |prediction - TD target| on the fit rows, in the same units as the targets.
        ``admissibility`` selects the per-row admissible bound (see the class docstring):
        ``upper`` (one U_i per target row), the violation counts ``reward_violations``,
        ``discount_violations`` and ``head_violations``, and optional ``diagnostics``."""
        current = current.detach().double().reshape(-1)
        low = float(current.min()) if current.numel() else 0.0
        high = float(current.max()) if current.numel() else 0.0
        slack = self.tolerance * self.weight
        violations: dict[str, int] = {}
        if admissibility is None:
            out_of_bounds = int(((current < -slack) | (current > self.weight + slack)).sum())
        else:
            upper = torch.as_tensor(admissibility["upper"]).detach().double().reshape(-1)
            if upper.shape != current.shape:
                raise ValueError("admissible bound needs one U_i per TD target row")
            out_of_bounds = int(
                ((current < -slack) | (current > upper * (1.0 + self.tolerance))).sum()
            )
            violations = {
                TD_REASON_REWARD: int(admissibility.get("reward_violations", 0)),
                TD_REASON_DISCOUNT: int(admissibility.get("discount_violations", 0)),
                TD_REASON_HEAD_RANGE: int(admissibility.get("head_violations", 0)),
                TD_REASON_ADMISSIBLE: out_of_bounds,
            }
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
        if admissibility is not None:
            record["bound"] = "admissible"
            record["upper_max"] = float(upper.max()) / self.weight if upper.numel() else 0.0
            record["reward_violations"] = violations[TD_REASON_REWARD]
            record["discount_violations"] = violations[TD_REASON_DISCOUNT]
            record["head_violations"] = violations[TD_REASON_HEAD_RANGE]
            if admissibility.get("diagnostics") is not None:
                record["diagnostics"] = dict(admissibility["diagnostics"])
        if seconds is not None:
            record["refresh_seconds"] = float(seconds)
        if previous is not None:
            change = (current - previous.detach().double().reshape(-1)).abs()
            record["drift"] = float(change.mean()) / self.weight if change.numel() else 0.0
            record["drift_max"] = float(change.max()) / self.weight if change.numel() else 0.0
        self.refreshes.append(record)
        alarm = None
        if admissibility is None:
            if out_of_bounds:
                alarm = {"check": "td_divergence", "reason": TD_REASON_FLAT_BOUND, **record}
        else:
            for reason, count in violations.items():
                if count:
                    alarm = {"check": "td_divergence", "reason": reason, **record}
                    break
        return self._judge_latest(record, alarm)

    def _judge_latest(
        self, record: Mapping[str, Any], alarm: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """The trajectory part of a refresh's live judgement (after its hard-bound alarm)."""
        if alarm is None and self.rule == TD_RULE_V1:
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
        if self.rule == TD_RULE_V2:
            for key in (TD_V2_GATED_SERIES, TD_V2_ADVISORY_SERIES):
                found = self._divergence_v2(key)
                if found is None:
                    continue
                entry = {
                    "check": "td_divergence",
                    "rule": TD_RULE_V2,
                    "reason": f"{key}-rose->{int(self.increase * 100)}%-for-"
                    f"{self.growth}-consecutive-refreshes-above-"
                    f"{self.median_factor:g}x-stationary-median",
                    **found,
                    **record,
                }
                if key == TD_V2_GATED_SERIES:
                    if alarm is None:
                        alarm = entry
                else:
                    self.advisories.append({**entry, "gating": False})
        if alarm is not None:
            self.alarms.append(alarm)
        return alarm

    def replay(self, refreshes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Re-judge recorded refreshes (``td_refresh`` log records or a resume state's
        ``refreshes``) under this monitor's rule, in order, as the live fit would have.

        Hard-bound alarms come from the recorded counts (the targets are not logged):
        ``out_of_bounds`` without ``bound`` is the flat [0, W] check; with
        ``bound == "admissible"`` the reward / discount / head / admissible counts. Returns
        the live alarms in refresh order; ``report()`` then gives the end-of-fit verdict.
        """
        fired = []
        for raw in refreshes:
            record = {k: v for k, v in raw.items() if k not in ("fit", "t", "kind")}
            self.refreshes.append(record)
            alarm = None
            if record.get("bound") == "admissible":
                counts = {
                    TD_REASON_REWARD: record.get("reward_violations", 0),
                    TD_REASON_DISCOUNT: record.get("discount_violations", 0),
                    TD_REASON_HEAD_RANGE: record.get("head_violations", 0),
                    TD_REASON_ADMISSIBLE: record.get("out_of_bounds", 0),
                }
                for reason, count in counts.items():
                    if count:
                        alarm = {"check": "td_divergence", "reason": reason, **record}
                        break
            elif record.get("out_of_bounds"):
                alarm = {"check": "td_divergence", "reason": TD_REASON_FLAT_BOUND, **record}
            alarm = self._judge_latest(record, alarm)
            if alarm is not None:
                fired.append(alarm)
        return fired

    def _divergence_v2(self, key: str) -> dict[str, Any] | None:
        return td_divergence_v2(
            self._eligible(key),
            warmup_factor=self.warmup_factor,
            growth=self.growth,
            increase=self.increase,
            median_factor=self.median_factor,
        )

    def trajectory(self) -> dict[str, Any]:
        if self.rule == TD_RULE_V2:
            return {
                key: {
                    **trajectory_report_v2(
                        self._eligible(key),
                        warmup_factor=self.warmup_factor,
                        window=self.window,
                        spread=self.spread,
                        floor=self._v2_floor(key),
                        ratio=self.ratio,
                    ),
                    "gating": key == TD_V2_GATED_SERIES,
                }
                for key in ("drift", "error")
            }
        return {
            key: trajectory_report(self._series(key), ratio=self.ratio, floor=self.floor)
            for key in ("drift", "error")
        }

    def report(self) -> dict[str, Any]:
        trajectory = self.trajectory()
        alarms = list(self.alarms)
        advisories = list(self.advisories)
        if self.rule == TD_RULE_V2:
            for key, result in trajectory.items():
                if result["pass"]:
                    continue
                entry = {
                    "check": "td_divergence",
                    "rule": TD_RULE_V2,
                    "reason": f"{key}-trajectory-not-declining-or-stationary",
                    **result,
                }
                if key == TD_V2_GATED_SERIES:
                    alarms.append(entry)
                else:
                    advisories.append({**entry, "gating": False})
            insufficient = (
                [TD_V2_GATED_SERIES]
                if trajectory[TD_V2_GATED_SERIES]["status"] == "insufficient"
                else []
            )
            return {
                "rule": TD_RULE_V2,
                "refreshes": list(self.refreshes),
                "alarms": alarms,
                "advisories": advisories,
                "insufficient": insufficient,
                "trajectory": trajectory,
                "gated_series": TD_V2_GATED_SERIES,
                "warmup": {"by": "value", "median_factor": self.warmup_factor},
                "warmup_refreshes": None,
                "final_ratio": self.ratio,
                "window": self.window,
                "spread": self.spread,
                "floors": {"error": self.error_floor, "drift": self.drift_floor},
                "divergence": {
                    "growth": self.growth,
                    "increase": self.increase,
                    "median_factor": self.median_factor,
                },
            }
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


def td_monitor_from_thresholds(
    weight: float, thresholds: StabilityThresholds, rule: str | None = None
) -> TDDriftMonitor:
    """The TD monitor :class:`StabilityMonitor` builds from ``thresholds`` (rule override)."""
    th = thresholds
    return TDDriftMonitor(
        weight,
        growth=th.td_growth_refreshes,
        floor=th.td_drift_floor,
        tolerance=th.td_bound_tolerance,
        warmup=th.td_warmup_refreshes,
        ratio=th.td_final_ratio,
        increase=th.td_divergence_increase,
        rule=th.td_rule if rule is None else rule,
        warmup_factor=th.td_warmup_median_factor,
        window=th.td_stationary_refreshes,
        spread=th.td_stationary_spread,
        error_floor=th.td_stationary_error_floor,
        drift_floor=th.td_stationary_drift_floor,
        median_factor=th.td_divergence_median_factor,
    )


def td_status(report: Mapping[str, Any] | None) -> str:
    """The td_divergence check's status from a TD report (as :meth:`StabilityMonitor.checks`)."""
    if report is None:
        return "HOLD"
    if report["alarms"]:
        return "FAIL"
    return "HOLD" if report["insufficient"] else "PASS"


def rejudge_td_refreshes(
    refreshes: Sequence[Mapping[str, Any]],
    *,
    rule: str = TD_RULE_V2,
    thresholds: StabilityThresholds | None = None,
) -> dict[str, Any]:
    """Re-judge a fit's recorded TD refreshes (a stability JSONL's ``td_refresh`` records)
    under ``rule``: the live alarms (the first is where an early stop would have fired),
    the end-of-fit report and the td_divergence status. Diagnostic: it never edits a
    receipt; a recorded receipt keeps the verdict of the rule it was judged under."""
    refreshes = [r for r in refreshes if r.get("kind", "td_refresh") == "td_refresh"]
    if not refreshes:
        return {"rule": rule, "status": "HOLD", "live_alarms": [], "report": None}
    monitor = td_monitor_from_thresholds(
        float(refreshes[0]["weight"]), thresholds or StabilityThresholds(), rule
    )
    live = monitor.replay(refreshes)
    report = monitor.report()
    return {
        "rule": rule,
        "status": td_status(report),
        "live_alarms": live,
        "early_stop_epoch": live[0]["epoch"] if live else None,
        "report": report,
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
    head_scale: float = 1.0,
) -> dict[str, Any]:
    """Stratified saturation shares for the bounded QoL head.

    ``z`` are pre-activations, ``target_fraction`` the target G/W per row (always in
    units of W, whatever the head range) and ``early`` a boolean early-stratum mask. A
    low-side row (z < -6) with target fraction below the expected-low bound is expected
    and excluded from the unexpected share. The alarm fires if either stratum's
    unexpected share exceeds its threshold. Stratum shares are fractions of that
    stratum's rows; the others are fractions of all rows.

    Head scale (27 Oct). A head of range (0, s W) fits p* = G / (s W); the expected-low
    rule is a statement about physical QoL (G < 0.05 W: no housing or clothing yet), so
    it is applied to G/W unchanged, which is p* < 0.05 / s in head units. The |z| > 6
    definition is frozen r5 text and is kept as is: at s = 2 a low-saturated row decodes
    below 2 sigmoid(-6) W = 0.0049 W (0.0025 W at s = 1), still far inside the
    expected-low stratum. ``head_scale`` other than 1 is recorded in the report.
    """
    import torch

    th = thresholds or StabilityThresholds()
    z = torch.as_tensor(z).detach().reshape(-1).double().cpu()
    frac = torch.as_tensor(target_fraction).detach().reshape(-1).double().cpu()
    early = torch.as_tensor(early).reshape(-1).bool().cpu()
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
        **(
            {
                "head_scale": float(head_scale),
                "expected_low_head_fraction": th.saturation_expected_low_target_fraction
                / float(head_scale),
            }
            if float(head_scale) != 1.0
            else {}
        ),
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


class ValidationMonitor:
    """Per-epoch held-out validation losses of a critic fit (monitoring only).

    Each record carries ``epoch``, the epoch's training loss ``train_loss`` (mean of its
    accepted minibatch losses) and the validation ``total``. The non-gating
    ``validation_divergence`` alarm fires when the validation total rose on each of the
    last ``epochs`` epochs while the training loss fell on each of them (strict changes).
    """

    def __init__(self, epochs: int = VALIDATION_DIVERGENCE_EPOCHS) -> None:
        if epochs < 1:
            raise ValueError("validation divergence needs at least one epoch")
        self.epochs = int(epochs)
        self.records: list[dict[str, Any]] = []
        self.alarms: list[dict[str, Any]] = []

    def update(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        self.records.append(dict(record))
        tail = [
            r
            for r in self.records
            if r.get("train_loss") is not None and r.get("total") is not None
        ][-(self.epochs + 1) :]
        if len(tail) < self.epochs + 1:
            return None
        pairs = list(zip(tail, tail[1:], strict=False))
        rising = all(b["total"] > a["total"] for a, b in pairs)
        falling = all(b["train_loss"] < a["train_loss"] for a, b in pairs)
        if not (rising and falling):
            return None
        alarm = {
            "check": "validation_divergence",
            "gating": False,
            "epoch": int(tail[-1]["epoch"]),
            "epochs": self.epochs,
            "validation_total": [float(r["total"]) for r in tail],
            "train_loss": [float(r["train_loss"]) for r in tail],
        }
        self.alarms.append(alarm)
        return alarm

    def report(self) -> dict[str, Any]:
        return {
            "gating": False,
            "rule": f"validation total rose on {self.epochs} consecutive epochs while the "
            "training loss fell",
            "epochs": list(self.records),
            "alarms": list(self.alarms),
            "status": "FLAG" if self.alarms else "OK",
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "records": _jsonable(self.records),
            "alarms": _jsonable(self.alarms),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.epochs = int(state["epochs"])
        self.records = [dict(r) for r in state["records"]]
        self.alarms = [dict(a) for a in state["alarms"]]


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
            torch.ones(weights.shape[0], weights.shape[-1], dtype=torch.bool, device=weights.device)
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
        hidden.std(0, unbiased=False)
        if hidden.shape[0] > 1
        else torch.zeros(hidden.shape[1], dtype=hidden.dtype, device=hidden.device)
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
        # Live early stop and non-gating live advisories (validation divergence, the
        # population tail); their summary keys appear only when used.
        self.early_stop: dict[str, Any] | None = None
        self._pending_stop: dict[str, Any] | None = None
        self.advisories: list[dict[str, Any]] = []
        self._tail_alarmed = False
        self.validation: ValidationMonitor | None = None
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
        # One host sync on an accelerator (labor_device.all_finite); per tensor on CPU.
        values = [
            value
            for state in self.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ]
        return all_finite(values)

    def _parameters_finite(self) -> bool:
        return all_finite(list(self.parameters))

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
        if alarm.get("check") in ("gradient_dominance", "td_divergence"):
            # Every such alarm is a mandatory FAIL in ``checks()`` (dominance: any detector
            # alarm; TD: any live alarm of TDDriftMonitor.update).
            self._request_stop(
                str(alarm["check"]), str(alarm.get("reason") or alarm["check"]), record
            )

    def _advisory(self, alarm: Mapping[str, Any]) -> None:
        """A live, non-gating alarm: logged and listed, never part of ``checks()``."""
        record = {"step": self.step_index, "gating": False, **alarm}
        self.advisories.append(record)
        self.write({"kind": "alarm", **record})

    def _request_stop(self, check: str, reason: str, evidence: Mapping[str, Any]) -> None:
        if self.config.early_stop and self._pending_stop is None:
            self._pending_stop = {
                "check": check,
                "reason": reason,
                "step": self.step_index,
                "evidence": dict(evidence),
            }

    def _maybe_stop(self) -> None:
        """Raise :class:`StabilityAbort` once a requested early stop is pending."""
        if self._pending_stop is None:
            return
        self.early_stop = {"stopped_early": True, **self._pending_stop}
        self._pending_stop = None
        summary = self.summary()
        self.write({"kind": "abort", "early_stop": self.early_stop, "summary": summary})
        self.close()
        raise StabilityAbort(
            f"{self.fit}: early stop at step {self.early_stop['step']}: "
            f"{self.early_stop['check']} ({self.early_stop['reason']})",
            summary,
        )

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
        head_grad_scales: Mapping[str, float] | None = None,
        progress_losses: Callable[[], Mapping[str, float]] | None = None,
        log_hook: Callable[
            [Mapping[str, Mapping[str, float]], Mapping[str, Mapping[str, float]]],
            Mapping[str, Any],
        ]
        | None = None,
    ) -> bool:
        """zero_grad was called by the caller; backward, clip, step, check, maybe roll back.

        ``head_losses`` are scalar losses (same graph as ``loss``) of the shared
        blocks' head contenders and of the contender blocks; they are read only
        on log steps. Returns False when the step was rolled back.

        ``head_grad_scales`` (per-head task loss weights): a head's loss enters ``loss``
        multiplied by its scale, so its effective gradient on a shared block is the
        unweighted head-loss gradient times the scale; the record's
        ``shared_head_grad_norm`` and the dominance detector then use the effective norms
        and ``shared_head_grad_norm_unweighted`` keeps the raw ones. ``progress_losses``
        (read on log steps, only with ``config.dominance_excess_loss``) replaces head
        losses in the shared-head detectors' progress criterion. ``log_hook(effective,
        raw)`` runs on successful log steps and its fields join the step record. None of
        them touches the step's arithmetic.
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
            if self.rollbacks > self.t.max_total_rollbacks:
                # The end-of-run finite_updates rule: more than max_total_rollbacks FAILs.
                self._request_stop(
                    "finite_updates",
                    f"rollbacks {self.rollbacks} > max_total_rollbacks "
                    f"{self.t.max_total_rollbacks}",
                    event,
                )
                self._maybe_stop()
            return False
        self.consecutive = 0
        self.successful_steps += 1
        if log:
            raw_norms = shared_norms
            if head_grad_scales is not None:
                shared_norms = {
                    block: {
                        head: value * float(head_grad_scales.get(head, 1.0))
                        for head, value in norms.items()
                    }
                    for block, norms in raw_norms.items()
                }
            detector_losses = head_values
            if self.config.dominance_excess_loss and progress_losses is not None:
                detector_losses = {**head_values, **dict(progress_losses())}
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
            if head_grad_scales is not None:
                record["shared_head_grad_norm_unweighted"] = raw_norms
            if detector_losses is not head_values:
                record["dominance_progress_losses"] = detector_losses
            for block, norms in shared_norms.items():
                alarm = self.detectors[block].update(self.step_index, norms, detector_losses)
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
            if log_hook is not None:
                record.update(dict(log_hook(shared_norms, raw_norms)))
            self.write(record)
        self._snapshot()
        # After the step is committed and logged: the stopped fit's state is its last step.
        self._maybe_stop()
        return True

    def _record_diagnostics(self, extra: Mapping[str, Any]) -> None:
        if "qol_saturation" in extra and extra["qol_saturation"] is not None:
            self.saturation.append({"step": self.step_index, "share": extra["qol_saturation"]})
        if "population_tail" in extra and extra["population_tail"] is not None:
            tail = extra["population_tail"]
            self.population_tail.append({"step": self.step_index, **tail})
            if self.config.early_stop:
                # Live, non-gating alarm (a user-escalation item, never a stop); the
                # end-of-run check still judges the final full pass. Alarms on entry only.
                exceeded = (
                    float(tail["upper_excess"]) > self.t.upper_excess_alarm
                    or float(tail["top_bin_mass"]) > self.t.top_bin_mass_alarm
                )
                if exceeded and not self._tail_alarmed:
                    self._advisory(
                        {
                            "check": "population_tail",
                            "live": True,
                            "upper_excess": float(tail["upper_excess"]),
                            "top_bin_mass": float(tail["top_bin_mass"]),
                            "upper_excess_alarm": self.t.upper_excess_alarm,
                            "top_bin_mass_alarm": self.t.top_bin_mass_alarm,
                            "note": "minibatch diagnostic; the end-of-run check judges the "
                            "final full pass",
                        }
                    )
                self._tail_alarmed = exceeded

    # ---- fit-level records -------------------------------------------------
    def _new_td(self, weight: float, rule: str | None = None) -> TDDriftMonitor:
        """``rule`` defaults to the configured ``td_rule`` (a new fit); a resume state
        passes its recorded rule (absent in states written before 28 Oct: v1)."""
        return td_monitor_from_thresholds(weight, self.t, rule)

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
        admissibility: Mapping[str, Any] | None = None,
    ) -> None:
        if self.td is None:
            self.td = self._new_td(weight)
            self.applicable["td_divergence"] = True
        advised = len(self.td.advisories)
        alarm = self.td.update(
            epoch,
            previous,
            current,
            growth_eligible=growth_eligible,
            error=error,
            seconds=seconds,
            admissibility=admissibility,
        )
        self.write({"kind": "td_refresh", **self.td.refreshes[-1]})
        # v2 drift-series divergence: a live advisory, never a stop or a FAIL.
        for advisory in self.td.advisories[advised:]:
            self._advisory(advisory)
        if alarm:
            self._alarm(alarm)
        self._maybe_stop()

    def record_validation(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        """One epoch's held-out validation losses (monitoring only, never gating)."""
        if self.validation is None:
            self.validation = ValidationMonitor()
        alarm = self.validation.update(record)
        self.write({"kind": "validation", **record})
        if alarm:
            self._advisory(alarm)
        return alarm

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
                "rule": self.td.rule,
                "refreshes": _jsonable(self.td.refreshes),
                "alarms": _jsonable(self.td.alarms),
                "advisories": _jsonable(self.td.advisories),
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
            "advisories": _jsonable(self.advisories),
            "tail_alarmed": self._tail_alarmed,
            "validation": None if self.validation is None else self.validation.state_dict(),
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
            # States written before rule v2 (28 Oct) carry no rule: they keep v1.
            self.td = self._new_td(state["td"]["weight"], state["td"].get("rule", TD_RULE_V1))
            self.td.refreshes = list(state["td"]["refreshes"])
            self.td.alarms = list(state["td"]["alarms"])
            self.td.advisories = list(state["td"].get("advisories") or [])
            self.applicable["td_divergence"] = True
        for name, saved in state["detectors"].items():
            detector = self.detectors[name]
            detector.points = list(saved["points"])
            detector.alarms = list(saved["alarms"])
            detector.dominant_windows = int(saved["dominant_windows"])
        if self.epoch_rankings:
            self.applicable["advantage_overfit"] = True
        # Resume files written before the live advisories existed lack these keys.
        self.advisories = list(state.get("advisories") or [])
        self._tail_alarmed = bool(state.get("tail_alarmed", False))
        if state.get("validation") is not None:
            self.validation = ValidationMonitor()
            self.validation.load_state_dict(state["validation"])
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
                    # Rule v2 fields; a v1 record is unchanged.
                    **(
                        {
                            "td_rule": td["rule"],
                            "gated_series": td["gated_series"],
                            "advisories": td["advisories"],
                            "floors": td["floors"],
                            "window": td["window"],
                        }
                        if td.get("rule") == TD_RULE_V2
                        else {}
                    ),
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
                # New fields appear only when used (non-gating, never part of ``checks()``).
                **({"early_stop": self.early_stop} if self.early_stop is not None else {}),
                **({"advisories": self.advisories} if self.advisories else {}),
                **(
                    {"validation": self.validation.report()}
                    if self.validation is not None
                    else {}
                ),
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
    """CLI helper: ``--no-stability`` / ``--stability-log`` / ``--stability-every`` /
    ``--stability-early-stop`` (which, like a log path, enables the monitor explicitly)."""
    early = bool(getattr(args, "stability_early_stop", False))
    if getattr(args, "no_stability", False):
        if early:
            raise ValueError("--stability-early-stop needs the stability monitor")
        if getattr(args, "dominance_excess_loss", False):
            raise ValueError("--dominance-excess-loss needs the stability monitor")
        return None
    jsonl = getattr(args, "stability_log", None) or default_jsonl
    every = getattr(args, "stability_every", None)
    excess = bool(getattr(args, "dominance_excess_loss", False))
    if jsonl is None and every is None and not early and not excess:
        return "auto"
    return StabilityConfig(
        log_every=int(every or 25),
        jsonl=None if jsonl is None else str(jsonl),
        early_stop=early,
        dominance_excess_loss=excess,
    )


def add_stability_arguments(parser: Any) -> None:
    parser.add_argument("--no-stability", action="store_true", help="disable the stability monitor")
    parser.add_argument("--stability-log", type=Path, default=None, help="stability JSONL path")
    parser.add_argument("--stability-every", type=int, default=None, help="log every N steps (25)")
    parser.add_argument(
        "--stability-early-stop",
        action="store_true",
        default=False,
        help="raise StabilityAbort the moment a would-FAIL condition trips (a gradient_dominance "
        "or td_divergence alarm, or more than max_total_rollbacks rollbacks); the same "
        "thresholds as the end-of-run checks (off by default)",
    )
