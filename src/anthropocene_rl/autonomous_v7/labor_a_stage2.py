"""Stage 2 of the labor A(s, a) head: from the teacher's Q to the actor's Q (library only).

Approved A(s,a) spec (rev 33, D1-D7), stage 2. PPO-loop integration is owned elsewhere;
this module is a set of pure, deterministic functions that loop can call. It does not
touch ``labor_ppo``, ``labor_advantage`` or ``labor_search`` and depends only on their
public API at 90eee94 (``LaborAdvantageHead``, ``REFERENCE_SCALES``, ``COMPONENTS``) and on
``labor_sa_advantage.SuccStream``/``successor_target`` as the branch record format.

Units. ``SuccStream`` rewards and the V values passed as ``v_succ`` are raw per-objective
units (as ``labor_sa_advantage.successor_target``). Paired targets are returned raw;
``to_head_units`` divides by ``labor_advantage.REFERENCE_SCALES`` (the units the head
predicts in) and ``to_raw_units`` undoes it. Policy-facing per-objective advantages stay
raw so they can go through ``labor_ppo.combine_labor_advantages`` unchanged.

Contents (spec item -> functions):

1. Adaptive branch selection (D6): ``head_distance``, ``calibrate_thresholds``,
   ``branch_probabilities``, ``select_branches``.
2. Paired targets: ``folded_lambda_return``, ``paired_target`` (k = 1..3, folded lambda
   return difference under CRN, weight 1/p), ``one_step_paired_signal`` (warm-up, k = 1),
   ``paired_policy_advantages``, ``single_action_target`` (G - V(s)).
3. Stage-2 fit: ``stage2_loss``/``stage2_fit_step`` (Huber on A(s,a) - A(s,b), weighted;
   optional single-action rows on the actor-centred A), ``actor_centred_advantage``.
4. Q-Prop-style per-head control variate: ``control_variate_advantages``,
   ``control_variate_correction``, ``per_head_surrogate``, ``categorical_expectation``.
5. Per-head gradient variance and D7: ``per_sample_gradients``, ``gradient_variance``,
   ``measure_head_variance``.
6. Warm-up switch: ``WarmupThresholds``, ``WarmupSwitch``, ``per_world_td``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import labor_advantage as la
from . import labor_sa_advantage as sa

COMPONENTS = la.COMPONENTS
assert tuple(sa.COMPONENTS) == tuple(COMPONENTS), "objective order differs between modules"

# Spec constants.
DISCOUNT_YEARS = 400.0  # gamma = exp(-years / 400) per step (0.951 for a 20-year turn)
LAMBDA = 0.95
MAX_PAIRED_HORIZON = 3  # stage-2 branches run 2-3 turns; k = 1..3

# D6: branch selection.
BRANCH_TAG = 0x4253  # "BS": keyed branch-selection draw
PROJECTION_TAG = 0x4250  # "BP": gradient-variance random projection
TARGET_BRANCH_RATE = 0.25  # about one state in four branches
FLOOR_BRANCH_RATE = 1.0 / 16.0  # about one in sixteen elsewhere

# D7: adopt A as a head's control variate at this gradient-variance reduction.
# TODO(D7): the spec sets the final value after the first round's measured V-only
# variance; 0.25 is the approved provisional constant.
D7_VARIANCE_REDUCTION = 0.25

# Distance kinds per policy head (see ``head_distance``).
DISTANCE_KINDS = ("categorical", "simplex", "counts", "scalar")


def _as_np(x: Any, dtype: Any = np.float64) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


# ---------------------------------------------------------------------------
# 1. Branch selection (D6, adaptive)
# ---------------------------------------------------------------------------


def head_distance(kind: str, actor: Any, h12: Any, *, scale: float = 1.0) -> np.ndarray:
    """Per-state distance [N] between the actor's action and H12's action on one head.

    ``kind``:

    * ``"categorical"`` (movement, fission, trade gate): integer codes ``[N]`` or ``[N, G]``
      (G groups, e.g. one gate per trade group). Distance = fraction of groups whose code
      differs (0 or 1 for a single group). Range [0, 1].
    * ``"simplex"`` (specialty share ``[N, K]``, activity shares ``[N, R, K]``): total
      variation 0.5 * sum |x - y| per row, averaged over the R rows. Range [0, 1].
    * ``"counts"`` (worker counts after largest-remainder apportionment, ``[N, K]`` or
      ``[N, R, K]``): sum |n_a - n_h12| / (2 * max(total actor workers, 1)), summed over
      rows: the fraction of workers reassigned when totals agree. Range [0, 1] when the
      two totals agree; larger if they differ.
    * ``"scalar"`` (trade value in the encoded log unit, trade share): mean over groups of
      |x - y| / ``scale``; ``[N]`` or ``[N, G]``.
    """
    if kind == "categorical":
        a, b = _as_np(actor, np.int64), _as_np(h12, np.int64)
        if a.shape != b.shape or a.ndim not in (1, 2):
            raise ValueError("categorical codes must be [N] or [N, G] and aligned")
        diff = (a != b).astype(np.float64)
        return diff if diff.ndim == 1 else diff.mean(axis=1)
    if kind == "simplex":
        a, b = _as_np(actor), _as_np(h12)
        if a.shape != b.shape or a.ndim not in (2, 3):
            raise ValueError("simplex shares must be [N, K] or [N, R, K] and aligned")
        tv = 0.5 * np.abs(a - b).sum(axis=-1)
        return tv if tv.ndim == 1 else tv.mean(axis=1)
    if kind == "counts":
        a, b = _as_np(actor), _as_np(h12)
        if a.shape != b.shape or a.ndim not in (2, 3):
            raise ValueError("counts must be [N, K] or [N, R, K] and aligned")
        flat_a = a.reshape(len(a), -1)
        flat_b = b.reshape(len(b), -1)
        total = np.maximum(flat_a.sum(axis=1), 1.0)
        return np.abs(flat_a - flat_b).sum(axis=1) / (2.0 * total)
    if kind == "scalar":
        if not scale > 0:
            raise ValueError("scalar distance scale must be positive")
        a, b = _as_np(actor), _as_np(h12)
        if a.shape != b.shape or a.ndim not in (1, 2):
            raise ValueError("scalars must be [N] or [N, G] and aligned")
        d = np.abs(a - b) / scale
        return d if d.ndim == 1 else d.mean(axis=1)
    raise ValueError(f"unknown distance kind {kind!r}; expected one of {DISTANCE_KINDS}")


def _stack_distances(distances: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    heads = list(distances)
    if not heads:
        raise ValueError("need at least one head distance")
    mat = np.stack([_as_np(distances[h]) for h in heads], axis=1)
    if mat.ndim != 2 or not np.isfinite(mat).all() or (mat < 0).any():
        raise ValueError("head distances must be finite, non-negative [N] vectors")
    return heads, mat


def far_mask(distances: Mapping[str, Any], thresholds: Mapping[str, float]) -> np.ndarray:
    """True where any head's distance strictly exceeds its threshold."""
    heads, mat = _stack_distances(distances)
    tau = np.asarray([float(thresholds[h]) for h in heads])
    return (mat > tau[None, :]).any(axis=1)


@dataclass(frozen=True)
class BranchCalibration:
    thresholds: dict[str, float]
    tail_mass: float  # per-head quantile level u: tau_h = quantile_h(1 - u)
    far_rate: float  # achieved fraction of states over some threshold
    expected_rate: float  # far_rate * far_probability + (1 - far_rate) * floor
    target_rate: float
    floor_rate: float
    far_probability: float
    head_triggers: dict[str, int]  # states over each head's own threshold
    states: int


def calibrate_thresholds(
    distances: Mapping[str, Any],
    *,
    target_rate: float = TARGET_BRANCH_RATE,
    floor_rate: float = FLOOR_BRANCH_RATE,
    far_probability: float = 1.0,
    iterations: int = 60,
) -> BranchCalibration:
    """Per-head thresholds such that the EXPECTED branch rate is ``target_rate``.

    Interpretation (spec "threshold tuned so about 1 state in 4 branches; floor rate about
    1 in 16 elsewhere"): the 0.25 is the overall expected rate, floor included, so the far
    fraction aimed at is ``(target - floor) / (far_probability - floor)`` (0.2 by default).
    All heads share one tail mass u (tau_h is head h's (1 - u) quantile), so heads with
    different distance scales trigger comparably; u is found by bisection (the far rate is
    monotone in u). Discrete distances (categorical) can make the target unreachable
    exactly: the closest achieved rate is returned and reported.
    """
    if not 0.0 <= floor_rate < far_probability <= 1.0:
        raise ValueError("need 0 <= floor_rate < far_probability <= 1")
    if not floor_rate <= target_rate <= far_probability:
        raise ValueError("target_rate must lie between floor_rate and far_probability")
    heads, mat = _stack_distances(distances)
    goal = (target_rate - floor_rate) / (far_probability - floor_rate)

    def thresholds_at(u: float) -> np.ndarray:
        level = min(1.0, max(0.0, 1.0 - u))
        return np.quantile(mat, level, axis=0, method="higher")

    def rate_at(u: float) -> float:
        return float((mat > thresholds_at(u)[None, :]).any(axis=1).mean())

    lo, hi = 0.0, 1.0
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        if rate_at(mid) < goal:
            lo = mid
        else:
            hi = mid
    u = lo if abs(rate_at(lo) - goal) <= abs(rate_at(hi) - goal) else hi
    tau = thresholds_at(u)
    far = mat > tau[None, :]
    far_rate = float(far.any(axis=1).mean())
    return BranchCalibration(
        thresholds={h: float(tau[i]) for i, h in enumerate(heads)},
        tail_mass=float(u),
        far_rate=far_rate,
        expected_rate=far_rate * far_probability + (1.0 - far_rate) * floor_rate,
        target_rate=float(target_rate),
        floor_rate=float(floor_rate),
        far_probability=float(far_probability),
        head_triggers={h: int(far[:, i].sum()) for i, h in enumerate(heads)},
        states=int(mat.shape[0]),
    )


def branch_probabilities(
    distances: Mapping[str, Any],
    thresholds: Mapping[str, float],
    *,
    floor_rate: float = FLOOR_BRANCH_RATE,
    far_probability: float = 1.0,
) -> np.ndarray:
    """Selection probability per state: ``far_probability`` if far, else ``floor_rate``."""
    if not 0.0 < floor_rate <= far_probability <= 1.0:
        raise ValueError("need 0 < floor_rate <= far_probability <= 1 (1/p must exist)")
    return np.where(far_mask(distances, thresholds), far_probability, floor_rate)


def select_branches(
    probabilities: Any, keys: Sequence[Sequence[int]], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keyed Bernoulli draw per state; returns (selected [N] bool, weight [N] = 1/p).

    ``keys[i]`` identifies the state (e.g. (world, turn, camp, decision)); the draw for a
    state depends only on (BRANCH_TAG, seed, key), not on batch order or size.
    """
    p = _as_np(probabilities)
    if p.ndim != 1 or len(keys) != len(p):
        raise ValueError("one key per probability")
    if not ((p > 0) & (p <= 1)).all():
        raise ValueError("selection probabilities must be in (0, 1]")
    draws = np.asarray(
        [
            np.random.default_rng([BRANCH_TAG, int(seed), *[int(x) for x in key]]).random()
            for key in keys
        ]
    )
    return draws < p, 1.0 / p


# ---------------------------------------------------------------------------
# 2. Paired targets
# ---------------------------------------------------------------------------


def folded_lambda_weights(k: int, lam: float = LAMBDA) -> tuple[float, float]:
    """(weight on the one-step term, weight on the k-step term).

    Only s_1 and s_k are recorded (D3), so the truncated lambda-return
    ``sum_{n<k} (1-lam) lam^(n-1) T_n + lam^(k-1) T_k`` is folded: 1 - lam on T_1 and the
    remaining lam on T_k (0.05 / 0.95 at lam = 0.95, as approved). k = 1 is T_1 alone.
    """
    if not 1 <= int(k) <= MAX_PAIRED_HORIZON:
        raise ValueError(f"k must be in 1..{MAX_PAIRED_HORIZON}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    return (1.0, 0.0) if int(k) == 1 else (1.0 - lam, lam)


def folded_lambda_return(
    stream: sa.SuccStream,
    k: int,
    v_succ: Mapping[str, np.ndarray],
    *,
    lam: float = LAMBDA,
    discount_years: float = DISCOUNT_YEARS,
) -> np.ndarray | None:
    """Folded lambda-return [4] (raw) of one branch stream, bootstrapped with V.

    Uses ``labor_sa_advantage.successor_target`` for T_1 and T_k, so the semantics are
    identical: a terminated focal camp closes with zero future; an inactive camp, a missing
    step or a missing successor row makes the target unavailable (None). A row must exist
    at decision k (export v2 records decision 1 and the last horizon).
    """
    w1, wk = folded_lambda_weights(k, lam)
    t1 = sa.successor_target(stream, 1, v_succ, discount_years)
    if t1 is None:
        return None
    if wk == 0.0:
        return t1
    tk = sa.successor_target(stream, int(k), v_succ, discount_years)
    if tk is None:
        return None
    return w1 * t1 + wk * tk


@dataclass(frozen=True)
class PairedTarget:
    """``G(s, a) - G(s, b)`` for the actor's two sampled actions at one state."""

    value: np.ndarray  # [4] raw, mean over shared CRN streams
    k: int
    streams: tuple[int, ...]
    weight: float  # 1 / selection probability


def _shared_streams(
    streams_a: Sequence[sa.SuccStream], streams_b: Sequence[sa.SuccStream]
) -> tuple[dict[int, sa.SuccStream], dict[int, sa.SuccStream], list[int]]:
    by_a = {s.stream: s for s in streams_a}
    by_b = {s.stream: s for s in streams_b}
    if len(by_a) != len(streams_a) or len(by_b) != len(streams_b):
        raise ValueError("duplicate stream index in a branch")
    return by_a, by_b, sorted(set(by_a) & set(by_b))


def paired_target(
    streams_a: Sequence[sa.SuccStream],
    streams_b: Sequence[sa.SuccStream],
    k: int,
    v_succ: Mapping[str, np.ndarray],
    *,
    selection_probability: float,
    lam: float = LAMBDA,
    discount_years: float = DISCOUNT_YEARS,
) -> PairedTarget | None:
    """Folded lambda-return difference between two CRN branches of the same state.

    Streams are paired by index (same stream = same common random numbers); the result is
    the mean over shared streams of ``G_a - G_b``. None when no stream is shared or any
    shared stream's target is unavailable on either side (strict: no partial averages).
    """
    if not 0.0 < selection_probability <= 1.0:
        raise ValueError("selection probability must be in (0, 1]")
    by_a, by_b, shared = _shared_streams(streams_a, streams_b)
    if not shared:
        return None
    diffs = []
    for s in shared:
        ga = folded_lambda_return(by_a[s], k, v_succ, lam=lam, discount_years=discount_years)
        gb = folded_lambda_return(by_b[s], k, v_succ, lam=lam, discount_years=discount_years)
        if ga is None or gb is None:
            return None
        diffs.append(ga - gb)
    return PairedTarget(
        np.mean(diffs, axis=0), int(k), tuple(shared), 1.0 / float(selection_probability)
    )


def one_step_paired_signal(
    streams_a: Sequence[sa.SuccStream],
    streams_b: Sequence[sa.SuccStream],
    v_succ: Mapping[str, np.ndarray],
    *,
    selection_probability: float,
    discount_years: float = DISCOUNT_YEARS,
) -> PairedTarget | None:
    """Warm-up signal ``(r_a + g V(s'_a)) - (r_b + g V(s'_b))`` (k = 1), weight 1/p."""
    return paired_target(
        streams_a,
        streams_b,
        1,
        v_succ,
        selection_probability=selection_probability,
        discount_years=discount_years,
    )


def paired_policy_advantages(diff: Any) -> tuple[Any, Any]:
    """Per-sample policy advantages of the two sampled actions from their difference.

    Leave-one-out baseline with two samples: the advantage of a is ``Q_a - Q_b = diff``
    and of b is ``-diff``; averaging the two score-function terms is an unbiased policy
    gradient (no state-value baseline involved, so never the single-path TD error).
    Works on arrays, tensors or per-objective dicts of either.
    """
    if isinstance(diff, Mapping):
        pairs = {name: paired_policy_advantages(value) for name, value in diff.items()}
        return {n: p[0] for n, p in pairs.items()}, {n: p[1] for n, p in pairs.items()}
    return diff, -diff


def single_action_target(
    streams: Sequence[sa.SuccStream],
    k: int,
    v_root: np.ndarray,
    v_succ: Mapping[str, np.ndarray],
    *,
    lam: float = LAMBDA,
    discount_years: float = DISCOUNT_YEARS,
) -> np.ndarray | None:
    """``G(s, a) - V(s)`` [4] raw for a single-action rollout (mean over its streams).
    ``v_root`` is V(s) per objective (raw, [4]). None if any stream is unavailable."""
    if not streams:
        return None
    values = []
    for s in streams:
        g = folded_lambda_return(s, k, v_succ, lam=lam, discount_years=discount_years)
        if g is None:
            return None
        values.append(g)
    return np.mean(values, axis=0) - np.asarray(v_root, dtype=np.float64)


def to_head_units(raw: Any) -> Any:
    """Raw per-objective values ([..., 4] array/tensor or dict) -> head (reference) units."""
    if isinstance(raw, Mapping):
        return {n: raw[n] / la.REFERENCE_SCALES[n] for n in COMPONENTS}
    scales = [la.REFERENCE_SCALES[n] for n in COMPONENTS]
    if isinstance(raw, torch.Tensor):
        return raw / torch.as_tensor(scales, dtype=raw.dtype, device=raw.device)
    return np.asarray(raw, dtype=np.float64) / np.asarray(scales)


def to_raw_units(head: Mapping[str, Any]) -> dict[str, Any]:
    """Head-unit predictions per objective -> raw units (for ``combine_labor_advantages``)."""
    return {n: head[n] * la.REFERENCE_SCALES[n] for n in COMPONENTS}


# ---------------------------------------------------------------------------
# 3. Stage-2 fit
# ---------------------------------------------------------------------------


@dataclass
class PairBatch:
    """Paired rows: two actor actions at one state (``root`` indexes the contexts)."""

    root: torch.Tensor  # [P] long
    features_a: torch.Tensor  # [P, ACTION_FEATURE_WIDTH]
    features_b: torch.Tensor  # [P, ACTION_FEATURE_WIDTH]
    target: torch.Tensor  # [P, 4] head units: G_a - G_b
    weight: torch.Tensor  # [P] (1/p, optionally times inverse noise variance)


@dataclass
class SingleBatch:
    """Single-action rows: target G(s, a) - V(s) for the actor-centred A."""

    root: torch.Tensor  # [M] long
    features: torch.Tensor  # [M, W]
    target: torch.Tensor  # [M, 4] head units
    weight: torch.Tensor  # [M]


@dataclass
class AnchorBatch:
    """The actor's sampled actions per state; their mean A is the baseline (anchor)."""

    root: torch.Tensor  # [S] long
    features: torch.Tensor  # [S, W]


def actor_centred_advantage(
    head: la.LaborAdvantageHead,
    contexts: Mapping[str, torch.Tensor],
    features: torch.Tensor,
    root: torch.Tensor,
    anchor: AnchorBatch,
) -> dict[str, torch.Tensor]:
    """A(s, a) - mean_{a' in anchor(s)} A(s, a') per objective (head units).

    Estimator of the stage-2 anchor E_{a~pi}[A(s, a)] = 0: the sample mean of A over the
    actor's own sampled actions at s (the two branch actions, plus any extra policy samples
    the caller scores; extra samples need no rollout). For a fixed A this mean is an
    unbiased estimate of E_pi[A(s, .)]; its noise falls as 1/(samples per state).
    Differences A(s,a) - A(s,b) are invariant to it.
    """
    rows = int(contexts[COMPONENTS[0]].shape[0])
    raw = head(contexts, features, root)
    anchor_raw = head(contexts, anchor.features, anchor.root)
    count = torch.zeros(rows, dtype=features.dtype).index_add(
        0, anchor.root, torch.ones(len(anchor.root), dtype=features.dtype)
    )
    if bool((count[root] == 0).any()):
        raise ValueError("every centred row needs at least one anchor sample at its state")
    out = {}
    for name in COMPONENTS:
        total = torch.zeros(rows, dtype=raw[name].dtype).index_add(0, anchor.root, anchor_raw[name])
        out[name] = raw[name] - (total / count.clamp_min(1))[root]
    return out


def _weighted_huber(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, delta: float
) -> torch.Tensor:
    if not delta > 0:
        raise ValueError("Huber delta must be positive")
    h = torch.nn.functional.huber_loss(pred, target, reduction="none", delta=float(delta))
    return (weight * h).sum() / weight.sum().clamp_min(1e-12)


@dataclass(frozen=True)
class Stage2Report:
    loss: float
    pair_loss: dict[str, float]
    single_loss: dict[str, float]
    pairs: int
    singles: int


def stage2_loss(
    head: la.LaborAdvantageHead,
    contexts: Mapping[str, torch.Tensor],
    pairs: PairBatch,
    *,
    delta: Mapping[str, float],
    single: SingleBatch | None = None,
    anchor: AnchorBatch | None = None,
    single_weight: float = 1.0,
    objectives: Sequence[str] = COMPONENTS,
) -> tuple[torch.Tensor, Stage2Report]:
    """Sum over objectives of weighted Huber(A(s,a) - A(s,b) - y) plus, optionally,
    ``single_weight`` * weighted Huber(A_centred(s,a) - (G - V(s))).

    ``delta[name]`` is the Huber threshold in head units (set from the target's noise SE
    by the caller). Contexts are detached inside the head, so no gradient reaches V.
    """
    for t in (pairs.weight,) + ((single.weight,) if single is not None else ()):
        if bool((t < 0).any()) or not bool(torch.isfinite(t).all()):
            raise ValueError("weights must be finite and non-negative")
    a = head(contexts, pairs.features_a, pairs.root)
    b = head(contexts, pairs.features_b, pairs.root)
    total = a[objectives[0]].sum() * 0.0
    pair_loss, single_loss = {}, {}
    for j, name in enumerate(COMPONENTS):
        if name not in objectives:
            continue
        term = _weighted_huber(a[name] - b[name], pairs.target[:, j], pairs.weight, delta[name])
        pair_loss[name] = float(term.detach())
        total = total + term
    if single is not None:
        if anchor is None:
            raise ValueError("single-action rows need the actor anchor samples")
        centred = actor_centred_advantage(head, contexts, single.features, single.root, anchor)
        for j, name in enumerate(COMPONENTS):
            if name not in objectives:
                continue
            term = _weighted_huber(centred[name], single.target[:, j], single.weight, delta[name])
            single_loss[name] = float(term.detach())
            total = total + float(single_weight) * term
    report = Stage2Report(
        float(total.detach()),
        pair_loss,
        single_loss,
        len(pairs.root),
        0 if single is None else len(single.root),
    )
    return total, report


def stage2_fit_step(
    head: la.LaborAdvantageHead,
    optimizer: torch.optim.Optimizer,
    contexts: Mapping[str, torch.Tensor],
    pairs: PairBatch,
    *,
    delta: Mapping[str, float],
    single: SingleBatch | None = None,
    anchor: AnchorBatch | None = None,
    single_weight: float = 1.0,
    frozen: Sequence[torch.nn.Module] = (),
) -> Stage2Report:
    """One optimizer step on ``stage2_loss``. ``frozen`` modules (the V critics) are
    checked after backward: any gradient reaching them raises (stop-gradient guard)."""
    owned = {id(p) for p in head.parameters()}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) not in owned:
                raise ValueError("the stage-2 optimizer may only hold A-head parameters")
    optimizer.zero_grad(set_to_none=True)
    loss, report = stage2_loss(
        head,
        contexts,
        pairs,
        delta=delta,
        single=single,
        anchor=anchor,
        single_weight=single_weight,
    )
    loss.backward()
    for module in frozen:
        for p in module.parameters():
            if p.grad is not None and bool((p.grad != 0).any()):
                raise RuntimeError("A's loss reached a frozen V parameter")
    optimizer.step()
    return report


# ---------------------------------------------------------------------------
# 4. Control variate (Q-Prop style, per policy head)
# ---------------------------------------------------------------------------
#
# Policy pi(a|s) = prod_h pi_h(a_h|s) (heads conditionally independent given s). For head h
# with a connection coefficient eta_h in {0, 1}, the per-sample policy gradient is
#
#   g_h = grad log pi_h(a_h|s) * [GAE - eta_h (A(s,a) - Abar(s))] + eta_h grad_h E_h(s),
#   E_h(s) = E_{a_h' ~ pi_h}[A(s, a_h', a_{-h})]   (a_{-h} the sampled other heads),
#   Abar(s) = E_{a ~ pi}[A(s, a)]                    (state-only; ~0 for the actor-centred A),
#
# where grad_h differentiates only through head h's distribution. Since
# E_{a_h}[grad log pi_h * A(s, a)] = grad_h E_h(s) and Abar(s) does not depend on a_h,
# E[g_h] equals the V-only GAE gradient for ANY A and ANY state-only Abar: a stale or wrong
# A only changes variance. Abar is subtracted (rather than E_h, also admissible) because
# GAE - A(s,a) + Abar(s) is close to pure noise when A is right, whereas GAE - A + E_h
# keeps the other heads' advantage. With the stage-2 anchor (A centred on the actor's own
# samples) Abar is the anchor mean, i.e. ~0. All quantities are in one combined unit
# (per-objective GAE and A, after ``to_raw_units``, through
# ``labor_ppo.combine_labor_advantages``).


def control_variate_advantages(
    gae: torch.Tensor,
    a_executed: torch.Tensor,
    a_policy_mean: torch.Tensor,
    eta: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    """Per-head score-function coefficient ``GAE - eta_h (A(s,a) - Abar(s))``, detached.

    ``a_policy_mean`` [N] is Abar(s) = E_{a~pi}[A(s, .)] (any state-only estimate keeps the
    estimator unbiased; its accuracy only affects variance). eta_h = 0 gives plain GAE.
    """
    out = {}
    residual = a_executed.detach() - a_policy_mean.detach()
    for h, e in eta.items():
        if float(e) not in (0.0, 1.0):
            raise ValueError("eta must be 0 (V-only) or 1 (A connected) per head")
        out[h] = gae.detach() - float(e) * residual
    return out


def control_variate_correction(
    expected: Mapping[str, torch.Tensor],
    eta: Mapping[str, float],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Analytic term ``mean_i sum_h eta_h E_h(s_i)`` to ADD to the policy objective.

    ``expected[h]`` must be differentiable in head h's distribution parameters only (the
    other heads' actions and A's parameters detached): e.g. ``categorical_expectation``
    for categorical heads, a reparameterised draw for continuous ones. ``weights`` are the
    per-row weights used for the surrogate's mean (e.g. 1/p); default uniform.
    """
    total = None
    for h, e in eta.items():
        if float(e) == 0.0:
            continue
        term = float(e) * expected[h]
        total = term if total is None else total + term
    if total is None:
        first = next(iter(expected.values()))
        return first.sum() * 0.0
    if weights is None:
        return total.mean()
    return (weights * total).sum() / weights.sum().clamp_min(1e-12)


def categorical_expectation(probs: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """E_{k ~ probs}[A_k] = sum_k p_k A_k, exact; ``values`` [N, K] are A at each category
    with the other heads fixed, detached (the gradient flows only through ``probs``)."""
    if probs.shape != values.shape or probs.ndim != 2:
        raise ValueError("probs and values must be aligned [N, K]")
    return (probs * values.detach()).sum(dim=1)


def per_head_surrogate(
    current_logp: Mapping[str, torch.Tensor],
    behaviour_logp: Mapping[str, torch.Tensor],
    advantages: Mapping[str, torch.Tensor],
    clip: float,
) -> torch.Tensor:
    """Per-row clipped surrogate with a separate advantage per head [N] (maximise).

    For head h the joint ratio exp(sum_j logp_j - logp_j_old) is formed with every other
    head's log-prob detached, so the gradient of head h's term flows only through
    log pi_h. Clipping uses the joint ratio, as ``labor_ppo._clipped_surrogate``. When all
    heads share one advantage the gradient equals the joint surrogate's gradient. Under
    ratio 1 (first epoch) the gradient is exactly sum_h grad log pi_h * c_h.
    """
    heads = list(advantages)
    if set(heads) != set(current_logp) or set(heads) != set(behaviour_logp):
        raise ValueError("advantages and log-probs must name the same heads")
    if not 0.0 < clip < 1.0:
        raise ValueError("clip must be in (0, 1)")
    old = sum(behaviour_logp[h].detach() for h in heads)
    total = None
    for h in heads:
        joint = current_logp[h] + sum(current_logp[j].detach() for j in heads if j != h)
        ratio = torch.exp(joint - old)
        adv = advantages[h].detach()
        term = torch.minimum(ratio * adv, ratio.clamp(1.0 - clip, 1.0 + clip) * adv)
        total = term if total is None else total + term
    return total


# ---------------------------------------------------------------------------
# 5. Per-head gradient variance (D7)
# ---------------------------------------------------------------------------


def per_sample_gradients(
    outputs: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    *,
    projection_dim: int | None = None,
    key: int = 0,
) -> torch.Tensor:
    """Per-row gradients [N, D] of ``outputs[i]`` (e.g. log pi_h(a_i|s_i), or E_h(s_i))
    with respect to ``params``; optionally projected onto ``projection_dim`` keyed Gaussian
    directions with E[P^T P] = I, which keeps the covariance trace unbiased. Use the same
    key for the V-only and the control-variate estimates (common projection)."""
    params = list(params)
    rows = []
    for i in range(len(outputs)):
        grads = torch.autograd.grad(outputs[i], params, retain_graph=True, allow_unused=True)
        rows.append(
            torch.cat(
                [
                    (g if g is not None else torch.zeros_like(p)).reshape(-1)
                    for g, p in zip(grads, params, strict=True)
                ]
            ).detach()
        )
    mat = torch.stack(rows) if rows else torch.zeros(0, sum(p.numel() for p in params))
    if projection_dim is None:
        return mat
    seed = int(np.random.default_rng([PROJECTION_TAG, int(key)]).integers(2**62))
    gen = torch.Generator().manual_seed(seed)
    proj = torch.randn(mat.shape[1], int(projection_dim), generator=gen, dtype=mat.dtype)
    return mat @ proj / math.sqrt(int(projection_dim))


def gradient_variance(
    score: torch.Tensor,
    coefficient: torch.Tensor,
    correction: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> float:
    """Trace of the covariance of the per-row estimate ``w_i (score_i * c_i + corr_i)``,
    with w normalised to mean 1 (the variance of a one-row estimate; divide by N for the
    batch mean)."""
    est = score * coefficient[:, None]
    if correction is not None:
        est = est + correction
    if weights is not None:
        est = est * (weights / weights.mean())[:, None]
    if len(est) < 2:
        raise ValueError("need at least two rows")
    return float(est.var(dim=0, unbiased=True).sum())


@dataclass(frozen=True)
class HeadVariance:
    head: str
    variance_v_only: float
    variance_cv: float
    reduction: float  # 1 - var_cv / var_v
    adopt: bool
    rows: int


def measure_head_variance(
    score: Mapping[str, torch.Tensor],
    gae: torch.Tensor,
    a_executed: torch.Tensor,
    a_policy_mean: torch.Tensor,
    correction: Mapping[str, torch.Tensor],
    *,
    weights: torch.Tensor | None = None,
    threshold: float = D7_VARIANCE_REDUCTION,
) -> tuple[dict[str, HeadVariance], dict[str, bool]]:
    """Per policy head: V-only vs A-control-variate gradient variance and the D7 decision.

    ``score[h]`` [N, P_h]: per-row grad log pi_h (``per_sample_gradients``); ``correction[h]``
    [N, P_h]: per-row grad_h E_h(s_i). Rows should be held-out actor rows. Returns the
    reports and the connection mask (True = connect A for that head, eta_h = 1).
    """
    reports, mask = {}, {}
    for h in score:
        coef = control_variate_advantages(gae, a_executed, a_policy_mean, {h: 1.0})[h]
        v = gradient_variance(score[h], gae.detach(), None, weights)
        c = gradient_variance(score[h], coef, correction[h], weights)
        reduction = 1.0 - c / v if v > 0 else 0.0
        adopt = bool(v > 0 and reduction >= threshold)
        reports[h] = HeadVariance(h, v, c, reduction, adopt, len(gae))
        mask[h] = adopt
    return reports, mask


def connection_eta(mask: Mapping[str, bool]) -> dict[str, float]:
    """D7 mask -> per-head eta for ``control_variate_advantages``/``..._correction``."""
    return {h: 1.0 if bool(m) else 0.0 for h, m in mask.items()}


# ---------------------------------------------------------------------------
# 6. Warm-up switch: one-step paired signals -> GAE
# ---------------------------------------------------------------------------

WARMUP_DECISIONS = 9  # the warm-up covers the first nine decisions (spec); see report notes
# V calibration gate on actor-visited states. The approved calibration thresholds (slope
# 0.8-1.2, |bias| < 0.25 of the realised spread) are reused for V provisionally.
# TODO(spec): confirm V's gate uses the same numbers as A's.
V_SLOPE_BAND = (0.8, 1.2)
V_BIAS_FRACTION = 0.25
# TODO(measure): per-world TD split bound (|mean TD| / spread per world); unset -> fails.
TD_SPLIT_BIAS_FRACTION: float | None = None
# TODO(measure): approx-KL bound for the K-epoch dry run; unset -> fails.
DRY_RUN_KL_BOUND: float | None = None
# TODO(measure): KL cap for the warm-up's small updates (used by the PPO loop, not here).
WARMUP_KL_CAP: float | None = None


@dataclass(frozen=True)
class WarmupThresholds:
    slope_band: tuple[float, float] = V_SLOPE_BAND
    bias_fraction: float = V_BIAS_FRACTION
    td_split_fraction: float | None = TD_SPLIT_BIAS_FRACTION
    kl_bound: float | None = DRY_RUN_KL_BOUND
    consecutive: int = 1  # passing reports needed in a row before switching


def per_world_td(world: Any, td: Any) -> dict[int, dict[str, float]]:
    """Per-world mean, standard deviation and count of TD errors (input to the TD split)."""
    w, t = _as_np(world, np.int64), _as_np(td)
    out = {}
    for key in sorted(set(w.tolist())):
        sel = t[w == key]
        out[int(key)] = {
            "mean": float(sel.mean()),
            "sd": float(sel.std(ddof=1)) if len(sel) > 1 else 0.0,
            "n": len(sel),
        }
    return out


@dataclass(frozen=True)
class WarmupReport:
    """Plain-number inputs for one switch evaluation (per objective where relevant)."""

    calibration_slope: Mapping[str, float]
    calibration_bias: Mapping[str, float]
    realised_spread: Mapping[str, float]
    td_by_world: Mapping[int, float]  # |mean TD| per world (any one unit)
    td_spread: float  # the same unit's realised spread
    dry_run_kl: Sequence[float]  # approx-KL per epoch, K epochs, updates not applied
    dry_run_clip_fraction: Sequence[float] = ()  # report only


@dataclass(frozen=True)
class WarmupDecision:
    phase: str  # "warmup" or "gae"
    passed: bool
    gates: dict[str, bool]
    reasons: list[str]
    streak: int


@dataclass
class WarmupSwitch:
    """Latching state machine: stays in ``warmup`` until ``consecutive`` reports in a row
    pass all three gates (V calibration, per-world TD split, dry-run KL), then ``gae``
    for good. An unset threshold fails its gate (fail closed)."""

    thresholds: WarmupThresholds = field(default_factory=WarmupThresholds)
    phase: str = "warmup"
    streak: int = 0
    history: list[WarmupDecision] = field(default_factory=list)

    def evaluate(self, report: WarmupReport) -> tuple[dict[str, bool], list[str]]:
        t = self.thresholds
        reasons: list[str] = []
        lo, hi = t.slope_band
        calibration = bool(report.calibration_slope)
        for name, slope in report.calibration_slope.items():
            if not (lo <= float(slope) <= hi):
                calibration = False
                reasons.append(f"V slope {name}={slope:.3g} outside [{lo}, {hi}]")
            spread = float(report.realised_spread[name])
            bias = abs(float(report.calibration_bias[name]))
            if not spread > 0 or bias >= t.bias_fraction * spread:
                calibration = False
                reasons.append(
                    f"V bias {name}={bias:.3g} >= {t.bias_fraction} x spread {spread:.3g}"
                )
        if not report.calibration_slope:
            reasons.append("no V calibration measured")
        if t.td_split_fraction is None:
            td = False
            reasons.append("TD split threshold unset (TODO measure)")
        else:
            bound = t.td_split_fraction * float(report.td_spread)
            bad = {w: v for w, v in report.td_by_world.items() if abs(float(v)) > bound}
            td = bool(report.td_by_world) and not bad and report.td_spread > 0
            if bad:
                reasons.append(f"TD split fails in worlds {sorted(bad)}")
            if not report.td_by_world:
                reasons.append("no per-world TD split")
        if t.kl_bound is None:
            kl = False
            reasons.append("dry-run KL bound unset (TODO measure)")
        else:
            kl = bool(report.dry_run_kl) and max(float(x) for x in report.dry_run_kl) <= t.kl_bound
            if not kl:
                reasons.append("dry-run approx-KL above bound or missing")
        return {"v_calibration": calibration, "td_split": td, "dry_run_kl": kl}, reasons

    def update(self, report: WarmupReport) -> WarmupDecision:
        gates, reasons = self.evaluate(report)
        passed = all(gates.values())
        if self.phase == "warmup":
            self.streak = self.streak + 1 if passed else 0
            if self.streak >= max(1, int(self.thresholds.consecutive)):
                self.phase = "gae"
        decision = WarmupDecision(self.phase, passed, gates, reasons, self.streak)
        self.history.append(decision)
        return decision
