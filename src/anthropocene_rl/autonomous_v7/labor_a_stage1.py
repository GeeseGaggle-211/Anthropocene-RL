"""Stage 1 of the A(s,a) design: search pretraining of the labor advantage head.

Spec: "A(s,a) target spec" rev 33, approved 4 Nov 2026 with recommendations D1-D7
(stage 1 only here; stage-2 library code lives elsewhere). ``V`` is the pretrained
critic (``LaborTaskCritic`` + ``LaborCostCritics``); ``A`` is
``labor_advantage.LaborAdvantageHead`` reading V's encoder context under
stop-gradient (``critic_contexts`` runs under ``no_grad`` and the head detaches it
again). An A fit never moves V's parameters or outputs; :func:`fit_stage1` checks
both bitwise (``v_guard``).

Targets (:func:`build_stage1_targets`)
--------------------------------------
Inputs: search export-v2 shards (``labor_search --successor-rows``): per first-action
candidate and evaluated stream, the raw per-step rewards ``step_rewards``/``step_years``
and the focal camp's critic input at decision 1 and at the root's last successor
decision K (``successor_decisions[-1]``), parsed by
``labor_sa_advantage.parse_successors``. ``V(s)`` is evaluated on those rows by a
critic checkpoint or an ensemble (per-row member mean) in raw units. All targets are
divided by the frozen reference scales (``labor_advantage.REFERENCE_SCALES``), the
units the A head predicts in.

Per candidate ``a``, objective ``j`` and stream ``s`` (``g_t = exp(-years_t / 400)``,
``labor_sa_advantage.successor_target``):

* one-step target  ``T1 = r_0 + g_0 V(s_1)``;
* k-step target    ``TK = sum_{t<K} (prod_{u<t} g_u) r_t + (prod_{t<K} g_t) V(s_K)``;
* a focal camp that terminates closes with zero future (the sum stops at the
  termination step); an ``inactive`` boundary, a missing step or a missing row
  makes that target unavailable.

Folded lambda-return (lambda = 0.95). With every branch state the lambda-return is
``G^lambda = (1 - lambda) sum_{n=1}^{K-1} lambda^{n-1} G^(n) + lambda^{K-1} G^(K)``.
Only s_1 and s_K are recorded, so the weight of every intermediate n-step return
(n >= 2) is folded onto the K-step return: ``G = 0.05 T1 + 0.95 TK``. A candidate is a
**survivor** when it has an evaluation at the final horizon K (it was kept by
successive halving to the last round, including a branch that terminated before K);
survivors use the folded target.

Degenerate folding for **non-survivors**: they have only s_1 (stream 0), so the K-step
term does not exist and its whole weight falls on the one-step term: weights (1, 0),
``G = T1``. Every candidate therefore contributes a one-step target; survivors are
never the only rows (selection bias).

Anchor. ``y_a = G_a - G_h12`` is computed on the same root, the same streams and the
**same estimator**: for a survivor, H12's folded target on each shared stream; for a
non-survivor, H12's one-step target on stream 0. The paired difference is therefore
exactly the folded lambda-return difference with the (0.05, 0.95) or (1, 0) weights on
both sides; ``y_h12 = 0`` and A is trained on ``A(a) - A(h12)``, so the prediction
``A(a) - A(h12)`` is exactly 0 for H12 (no centring).

Streams and noise. Survivors with >= 2 re-evaluation streams (1..S) shared with H12
use the mean paired difference over those streams; the per-candidate noise variance
is the sample variance of the paired differences (ddof 1) divided by their count,
floored at ``VARIANCE_FLOOR_FRACTION`` times the pooled single-stream variance divided
by the count (three streams give two degrees of freedom; an exactly-zero sample
variance would otherwise give an infinite weight). Otherwise (non-survivors, or
survivors without two shared re-evaluations) the target is the stream-0 paired
difference with the **pooled per-horizon variance**: the dof-weighted mean of the
single-stream paired-difference variances over all survivors' re-evaluation streams,
pooled separately for the one-step (horizon 1) and the folded (horizon K) estimator
and per objective. The re-evaluation streams are H12-dithered while stream 0 is the
deterministic H12 continuation; the pooled variance is therefore a proxy for stream 0's
noise. Every SE is at least ``SE_ABS_FLOOR``.

Unavailable targets (no successor streams, inactive or missing rows, H12 unavailable on
the needed stream) are dropped and counted by reason. Every row carries its survival
flag and estimator class; every evaluation is split by survival status.

Loss (D1): :func:`anchored_huber_loss`
--------------------------------------
Per objective ``j`` and row ``a``: ``r = (A_j(a) - A_j(h12)) - y_j(a)`` and
``loss = mean_a Huber_c(r / se_j(a))`` with ``c = 1.345`` (``HUBER_SE_MULTIPLIER``).
Since ``Huber_c(r / se) = Huber_{c se}(r) / se^2``, this is the Huber loss of the
residual with **delta = 1.345 x the row's target noise SE** (the 95%-Gaussian-
efficiency constant: quadratic within 1.345 noise SEs, linear beyond) and **weight =
1 / se^2**, the inverse noise variance. Every candidate is fitted against H12 on its
root; there is no 2-SE pair filter. Ranking metrics are reported only, never trained
on. ``labor_advantage.pairwise_loss`` stays callable for comparison.

Action encoding and masks
-------------------------
Every candidate's specialty block is encoded after native largest-remainder
apportionment (``labor_advantage.action_features(population=...)``; the root
population is required). The trade value and trade share blocks are never perturbed
by search (D5) and are masked out of A (zeroed before the action encoder, so A's
response to them is exactly zero). Each other block's support is the number of fit
rows whose block differs from H12's at their root; a block below
``support_floor`` (default ``BLOCK_SUPPORT_FLOOR`` = 30 rows) is masked too, held at
zero and reported.

Gates (:func:`stage1_gate_report`): thresholds are the approved values below; see the
report's ``definitions``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import labor_advantage as la
from .labor_advantage import (
    COMPONENTS,
    LAMBDA_TARGET_SOURCE,
    REFERENCE_SCALES,
    SE_ABS_FLOOR,
    VARIANCE_FLOOR_FRACTION,
    AdvantageFitConfig,
    CandidateSet,
    LaborAdvantageHead,
    Root,
)
from .labor_sa_advantage import SuccessorData, SuccStream, parse_successors, successor_target

TARGET_SCHEMA = "labor-a-stage1-targets-v1"
GATE_SCHEMA = "labor-a-stage1-gates-v1"
LAMBDA = 0.95
ONE_STEP_WEIGHT = 1.0 - LAMBDA  # 0.05
K_STEP_WEIGHT = LAMBDA  # 0.95
DISCOUNT_YEARS = 400.0
ONE_STEP, FOLDED = "one-step", "folded"

# Approved gate thresholds (spec table).
CALIBRATION_SLOPE_RANGE = (0.8, 1.2)
CALIBRATION_BIAS_FRACTION = 0.25  # |bias| < 0.25 x realised within-root spread
ORDERING_SIGN_ACCURACY = 0.65  # informative pairs, controls excluded
PATH_SIGN_ACCURACY_FLOOR = 0.5  # no world or family below
SEED_RANK_CORRELATION = 0.8
OVERFIT_GAP = 0.10
OVERFIT_HELDOUT_DROP = 0.03
# Evaluation conventions (not approved thresholds; stated in the report).
INFORMATIVE_PAIR_SE = 2.0  # |y_a - y_b| > 2 sqrt(se_a^2 + se_b^2)
BLOCK_DIFF_TOLERANCE = 1e-6
# PPO-readiness dry run (report only; sets K).
PPO_DRY_RUN_EPOCHS = 8
PPO_CLIP = 0.2
PPO_DRY_RUN_LR = 3e-4
PPO_DRY_RUN_ROOTS = 64
DRY_RUN_TAG = 0x4131  # "A1": keyed dry-run labor sample

ValuesFn = Callable[[Mapping[str, torch.Tensor]], Mapping[str, np.ndarray]]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class StageOneData:
    """Parsed export-v2 shards: candidates, successor streams and successor rows."""

    candidates: CandidateSet
    successors: dict[str, dict[int, dict[int, SuccStream]]]  # root -> cand -> stream
    final_decision: dict[str, int]  # root -> K
    successor_dataset: dict[str, torch.Tensor]  # merged tensors + privileged, global rows
    successor_rows: int


def stage1_data(shards: Sequence[Mapping[str, Any]]) -> StageOneData:
    """Strictly parse export-v2 shards (in order) for stage-1 targets."""
    if not shards:
        raise ValueError("no candidate shards")
    sets: list[CandidateSet] = []
    successors: dict[str, dict[int, dict[int, SuccStream]]] = {}
    final: dict[str, int] = {}
    parts: list[dict[str, Any]] = []
    offset = 0
    for index, shard in enumerate(shards):
        if "successor_row_count" not in shard or shard.get("successor_tensors") is None:
            raise ValueError("stage-1 targets need search export-v2 shards (--successor-rows)")
        sets.append(la.parse_candidate_shard(shard, shard_index=index))
        for raw in shard["roots"]:
            root_id = str(raw["root_id"])
            per = parse_successors(raw, offset)
            successors[root_id] = {
                cand: {stream.stream: stream for stream in streams} for cand, streams in per.items()
            }
            decisions = raw.get("successor_decisions") or sorted(
                {d for streams in per.values() for s in streams for d in s.row}
            )
            if not decisions:
                raise ValueError(f"root {root_id} declares no successor decisions")
            final[root_id] = int(max(int(d) for d in decisions))
        count = int(shard["successor_row_count"])
        if count:
            parts.append(
                SuccessorData._merge(shard["successor_tensors"], shard.get("successor_privileged"))
            )
        offset += count
    merged = la.merge_candidate_sets(sets)
    return StageOneData(merged, successors, final, SuccessorData._concat(parts), offset)


def load_stage1_data(paths: Sequence[Path | str]) -> StageOneData:
    """Export-v2 shards from files and directories (each directory's ``*.pt`` sorted, or
    ``advantage-candidates.pt`` found recursively), refusing a file reached twice."""
    files: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            found = sorted(path.glob("*.pt")) or sorted(path.glob("**/advantage-candidates.pt"))
            files.extend(found)
        else:
            files.append(path)
    if not files:
        raise ValueError("no candidate shard files found")
    resolved = [f.resolve() for f in files]
    if len(set(resolved)) != len(resolved):
        raise ValueError("candidate shard files listed more than once")
    return stage1_data([torch.load(f, map_location="cpu", weights_only=True) for f in files])


def critic_values(task: nn.Module, costs: nn.Module, batch_size: int = 256) -> ValuesFn:
    """V of one critic checkpoint on a dataset (raw units per head)."""
    from .labor_pretrain_eval import critic_predictions

    def values(data: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
        n = int(data["core"].shape[0]) if data else 0
        if n == 0:
            return {h: np.zeros(0) for h in COMPONENTS}
        preds = critic_predictions(task, costs, data, np.arange(n), batch_size)
        return {h: np.asarray(preds[h], dtype=np.float64) for h in COMPONENTS}

    return values


def ensemble_values(paths: Sequence[Path | str]) -> ValuesFn:
    """V of a critic ensemble: per-row member mean (``labor_sa_advantage._ensemble_values``)."""
    from .labor_sa_advantage import _ensemble_values

    def values(data: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
        _, members = _ensemble_values([Path(p) for p in paths], [data])
        return {h: np.mean([m[0][h] for m in members], axis=0) for h in COMPONENTS}

    return values


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

_ROW_FIELDS = (
    "root_id",
    "world",
    "candidate",
    "h12",
    "family",
    "control",
    "survivor",
    "target_class",
    "source",
    "streams",
    "y",
    "se",
)


@dataclass
class StageOneTargets:
    """One row per non-H12 candidate with an available target (reference-scale units)."""

    root_id: np.ndarray  # str
    world: np.ndarray
    candidate: np.ndarray
    h12: np.ndarray
    family: np.ndarray  # str
    control: np.ndarray  # bool: evaluator control or bookkeeping twin
    survivor: np.ndarray  # bool
    target_class: np.ndarray  # str: one-step / folded
    source: np.ndarray  # str: reeval / stream0
    streams: np.ndarray  # int: streams averaged
    y: np.ndarray  # [N, 4] G_a - G_h12
    se: np.ndarray  # [N, 4] noise SE
    dropped: dict[str, int] = field(default_factory=dict)
    candidates_seen: int = 0
    pooled_variance: dict[str, dict[str, float | None]] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.root_id.size)

    def select(self, mask: np.ndarray) -> StageOneTargets:
        mask = np.asarray(mask, dtype=bool)
        return StageOneTargets(
            **{name: getattr(self, name)[mask] for name in _ROW_FIELDS},
            dropped=dict(self.dropped),
            candidates_seen=self.candidates_seen,
            pooled_variance=self.pooled_variance,
        )

    def for_roots(self, root_ids: Sequence[str]) -> StageOneTargets:
        keep = {str(r) for r in root_ids}
        return self.select(np.asarray([r in keep for r in self.root_id.tolist()], dtype=bool))

    def summary(self) -> dict[str, Any]:
        return {
            "schema": TARGET_SCHEMA,
            "rows": len(self),
            "candidates_seen": int(self.candidates_seen),
            "dropped": dict(sorted(self.dropped.items())),
            "rows_by_class": dict(sorted(Counter(self.target_class.tolist()).items())),
            "rows_by_source": dict(sorted(Counter(self.source.tolist()).items())),
            "survivors": int(self.survivor.sum()),
            "non_survivors": int((~self.survivor).sum()),
            "pooled_single_stream_variance": self.pooled_variance,
            "lambda": LAMBDA,
            "weights": {"one_step": ONE_STEP_WEIGHT, "k_step": K_STEP_WEIGHT},
            "units": "reference-scale-normalized-return-difference",
        }


def _is_control(cand: la.CandidateRecord) -> bool:
    return cand.control is not None or cand.twin is not None or cand.family.startswith("control")


def _require_population(roots: Sequence[Root]) -> None:
    missing = [r.root_id for r in roots if r.population is None]
    if missing:
        raise ValueError(
            f"stage 1 encodes specialties after apportionment; roots without a population: "
            f"{missing[:3]}"
        )


def build_stage1_targets(
    data: StageOneData,
    values: ValuesFn,
    *,
    lam: float = LAMBDA,
    variance_floor_fraction: float = VARIANCE_FLOOR_FRACTION,
    se_abs_floor: float = SE_ABS_FLOOR,
    discount_years: float = DISCOUNT_YEARS,
) -> StageOneTargets:
    """Per-candidate, per-objective folded lambda-return targets ``y = G_a - G_h12``.

    See the module docstring for the definitions. ``values`` maps the successor dataset
    to raw-unit V per head (:func:`critic_values` / :func:`ensemble_values`).
    """
    roots = data.candidates.roots
    _require_population(roots)
    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    w1, wk = 1.0 - float(lam), float(lam)
    v_raw = (
        values(data.successor_dataset)
        if data.successor_rows
        else {h: np.zeros(0) for h in COMPONENTS}
    )
    for h in COMPONENTS:
        if len(v_raw[h]) != data.successor_rows:
            raise ValueError("V must give one value per successor row")
    scale = np.asarray([REFERENCE_SCALES[h] for h in COMPONENTS], dtype=np.float64)

    def targets_of(stream: SuccStream, k: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        t1 = successor_target(stream, 1, v_raw, discount_years)
        tk = successor_target(stream, k, v_raw, discount_years)
        return (
            None if t1 is None else t1 / scale,
            None if tk is None else tk / scale,
        )

    prelim: list[tuple[Root, la.CandidateRecord, bool, dict, dict, str | None]] = []
    pool: dict[str, list[tuple[np.ndarray, int]]] = {ONE_STEP: [], FOLDED: []}
    dropped: Counter[str] = Counter()
    seen = 0
    for root in roots:
        k = data.final_decision[root.root_id]
        streams = data.successors.get(root.root_id, {})
        h12 = root.candidates[root.h12]
        h_t = {s: targets_of(e, k) for s, e in streams.get(h12.index, {}).items()}
        for cand in root.candidates:
            if cand.is_h12:
                continue
            seen += 1
            survivor = any(h == k for (h, _) in cand.evaluations)
            if cand.index not in streams:
                prelim.append((root, cand, survivor, {}, {}, "no-successor-streams"))
                continue
            c_t = {s: targets_of(e, k) for s, e in streams[cand.index].items()}
            one, fold = {}, {}
            for s, (t1, tk) in c_t.items():
                if s not in h_t:
                    continue
                h1, hk = h_t[s]
                if t1 is not None and h1 is not None:
                    one[s] = t1 - h1
                    if tk is not None and hk is not None:
                        fold[s] = (w1 * t1 + wk * tk) - (w1 * h1 + wk * hk)
            reason = None
            if survivor:
                for cls, diffs in ((FOLDED, fold), (ONE_STEP, one)):
                    reeval = sorted(s for s in diffs if s >= 1)
                    if len(reeval) >= 2:
                        d = np.stack([diffs[s] for s in reeval])
                        pool[cls].append((d.var(axis=0, ddof=1), len(reeval) - 1))
            prelim.append((root, cand, survivor, one, fold, reason))

    pooled: dict[str, np.ndarray | None] = {}
    for cls, items in pool.items():
        dof = sum(n for _, n in items)
        pooled[cls] = None if dof == 0 else sum(v * n for v, n in items) / dof

    rows: dict[str, list[Any]] = defaultdict(list)
    for root, cand, survivor, one, fold, reason in prelim:
        tag = "survivor" if survivor else "non-survivor"
        if reason is not None:
            dropped[f"{tag}:{reason}"] += 1
            continue
        cls = FOLDED if survivor else ONE_STEP
        diffs = fold if survivor else one
        pooled_var = pooled[cls]
        reeval = sorted(s for s in diffs if s >= 1) if survivor else []
        if len(reeval) >= 2:
            d = np.stack([diffs[s] for s in reeval])
            n = len(reeval)
            floor = (
                np.zeros(len(COMPONENTS))
                if pooled_var is None
                else float(variance_floor_fraction) * pooled_var
            )
            y = d.mean(axis=0)
            se2 = np.maximum(d.var(axis=0, ddof=1), floor) / n
            source, used = "reeval", n
        elif 0 in diffs:
            y = diffs[0]
            se2 = np.zeros(len(COMPONENTS)) if pooled_var is None else pooled_var.copy()
            source, used = "stream0", 1
        else:
            dropped[f"{tag}:{cls}-target-unavailable"] += 1
            continue
        rows["root_id"].append(root.root_id)
        rows["world"].append(root.world)
        rows["candidate"].append(cand.index)
        rows["h12"].append(root.h12)
        rows["family"].append(cand.family)
        rows["control"].append(_is_control(cand))
        rows["survivor"].append(bool(survivor))
        rows["target_class"].append(cls)
        rows["source"].append(source)
        rows["streams"].append(used)
        rows["y"].append(y)
        rows["se"].append(np.maximum(np.sqrt(se2), float(se_abs_floor)))
    width = len(COMPONENTS)
    return StageOneTargets(
        root_id=np.asarray(rows["root_id"], dtype=object),
        world=np.asarray(rows["world"], dtype=np.int64),
        candidate=np.asarray(rows["candidate"], dtype=np.int64),
        h12=np.asarray(rows["h12"], dtype=np.int64),
        family=np.asarray(rows["family"], dtype=object),
        control=np.asarray(rows["control"], dtype=bool),
        survivor=np.asarray(rows["survivor"], dtype=bool),
        target_class=np.asarray(rows["target_class"], dtype=object),
        source=np.asarray(rows["source"], dtype=object),
        streams=np.asarray(rows["streams"], dtype=np.int64),
        y=np.stack(rows["y"]) if rows["y"] else np.zeros((0, width)),
        se=np.stack(rows["se"]) if rows["se"] else np.zeros((0, width)),
        dropped=dict(dropped),
        candidates_seen=seen,
        pooled_variance={
            cls: {"horizon": "1" if cls == ONE_STEP else "K"}
            | (
                {name: None for name in COMPONENTS}
                if pooled[cls] is None
                else {name: float(pooled[cls][j]) for j, name in enumerate(COMPONENTS)}
            )
            for cls in (ONE_STEP, FOLDED)
        },
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def anchored_huber_loss(
    a_i: torch.Tensor,
    a_h12: torch.Tensor,
    y: torch.Tensor,
    se: torch.Tensor,
    *,
    c: float = la.HUBER_SE_MULTIPLIER,
) -> torch.Tensor:
    """mean_a Huber_c((A_a - A_h12 - y_a) / se_a): Huber with delta = c x se_a and
    weight 1 / se_a^2 (inverse noise variance), see the module docstring."""
    if a_i.numel() == 0:
        return a_i.sum() * 0.0
    if bool((se <= 0).any()):
        raise ValueError("target noise SE must be positive")
    z = (a_i - a_h12 - y) / se
    absz = z.abs()
    return torch.where(absz <= c, 0.5 * z.square(), c * (absz - 0.5 * c)).mean()


# ---------------------------------------------------------------------------
# Support and masks
# ---------------------------------------------------------------------------


def _root_map(roots: Sequence[Root]) -> dict[str, Root]:
    return {r.root_id: r for r in roots}


def block_support(roots: Sequence[Root], targets: StageOneTargets) -> dict[str, int]:
    """Per action block: target rows whose block differs from H12's at their root."""
    slices = la.block_slices()
    by_id = _root_map(roots)
    counts = {name: 0 for name in slices}
    for root_id, c, h in zip(targets.root_id, targets.candidate, targets.h12, strict=True):
        root = by_id[root_id]
        f, fh = root.candidates[int(c)].features, root.candidates[int(h)].features
        for name, sl in slices.items():
            if float(np.max(np.abs(f[sl] - fh[sl]))) > BLOCK_DIFF_TOLERANCE:
                counts[name] += 1
    return counts


def masked_blocks_for(
    support: Mapping[str, int], config: AdvantageFitConfig
) -> tuple[list[str], list[str]]:
    """(all masked blocks, blocks masked for low support)."""
    low = sorted(
        name
        for name, n in support.items()
        if n < config.support_floor and name not in config.masked_blocks
    )
    return sorted(set(config.masked_blocks) | set(low)), low


# ---------------------------------------------------------------------------
# Prediction and metrics
# ---------------------------------------------------------------------------


def anchored_predictions(
    head: LaborAdvantageHead, contexts: Mapping[str, torch.Tensor], roots: Sequence[Root]
) -> dict[str, list[np.ndarray]]:
    """A(a) - A(h12) per component, one array per root (candidate order); H12 is 0."""
    from .labor_device import module_device

    flat = la._flatten(roots)
    device = module_device(head)
    with torch.no_grad():
        raw = head(contexts, flat.features.to(device), flat.root_index.to(device))
    out: dict[str, list[np.ndarray]] = {}
    for name in COMPONENTS:
        values = raw[name].double().cpu().numpy()
        out[name] = []
        for r, root in enumerate(roots):
            block = values[flat.offsets[r] : flat.offsets[r] + len(root.candidates)]
            out[name].append(block - block[root.h12])
    return out


def evaluation_rows(
    roots: Sequence[Root], targets: StageOneTargets, prediction: Mapping[str, list[np.ndarray]]
) -> dict[str, np.ndarray]:
    """Target rows with their anchored predictions, plus one H12 anchor row (y = 0,
    prediction 0) per (root, estimator class) group, so within-root comparisons never
    mix the one-step and folded estimators."""
    position = {r.root_id: i for i, r in enumerate(roots)}
    out: dict[str, list[Any]] = defaultdict(list)
    groups: dict[str, tuple[str, int, bool, str]] = {}
    for i in range(len(targets)):
        root_id = str(targets.root_id[i])
        r = position[root_id]
        cls = str(targets.target_class[i])
        group = f"{root_id}|{cls}"
        groups.setdefault(group, (root_id, int(targets.world[i]), bool(targets.survivor[i]), cls))
        out["group"].append(group)
        out["world"].append(int(targets.world[i]))
        out["family"].append(str(targets.family[i]))
        out["control"].append(bool(targets.control[i]))
        out["survivor"].append(bool(targets.survivor[i]))
        out["anchor"].append(False)
        out["pred"].append([prediction[n][r][int(targets.candidate[i])] for n in COMPONENTS])
        out["y"].append(targets.y[i])
        out["se"].append(targets.se[i])
    for group, (_root_id, world, survivor, _cls) in groups.items():
        out["group"].append(group)
        out["world"].append(world)
        out["family"].append("h12")
        out["control"].append(False)
        out["survivor"].append(survivor)
        out["anchor"].append(True)
        out["pred"].append([0.0] * len(COMPONENTS))
        out["y"].append(np.zeros(len(COMPONENTS)))
        out["se"].append(np.zeros(len(COMPONENTS)))
    width = len(COMPONENTS)
    return {
        "group": np.asarray(out["group"], dtype=object),
        "world": np.asarray(out["world"], dtype=np.int64),
        "family": np.asarray(out["family"], dtype=object),
        "control": np.asarray(out["control"], dtype=bool),
        "survivor": np.asarray(out["survivor"], dtype=bool),
        "anchor": np.asarray(out["anchor"], dtype=bool),
        "pred": np.asarray(out["pred"], dtype=np.float64).reshape(-1, width),
        "y": np.asarray(out["y"], dtype=np.float64).reshape(-1, width),
        "se": np.asarray(out["se"], dtype=np.float64).reshape(-1, width),
    }


def _subset(rows: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in rows.items()}


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return None if not vals else float(np.mean(vals))


def calibration_metrics(rows: Mapping[str, np.ndarray], j: int) -> dict[str, Any]:
    """World-balanced within-root calibration of objective ``j``: slope of realised on
    predicted after per-group demeaning (anchors included), MAE and bias over non-anchor
    rows, realised spread = RMS over groups of the within-group SD of y."""
    per_world: dict[int, dict[str, Any]] = {}
    for w in sorted({int(x) for x in rows["world"]}):
        m = rows["world"] == w
        p, y, g, anchor = rows["pred"][m, j], rows["y"][m, j], rows["group"][m], rows["anchor"][m]
        dp, dy, sds = p.copy(), y.copy(), []
        for gid in np.unique(g):
            k = g == gid
            dp[k] -= p[k].mean()
            dy[k] -= y[k].mean()
            if k.sum() >= 2:
                sds.append(float(np.std(y[k])))
        var = float(np.dot(dp, dp))
        err = (p - y)[~anchor]
        per_world[w] = {
            "rows": int((~anchor).sum()),
            "roots": len(np.unique(g)),
            "slope": None if var == 0 else float(np.dot(dp, dy) / var),
            "mae": None if err.size == 0 else float(np.mean(np.abs(err))),
            "bias": None if err.size == 0 else float(np.mean(err)),
            "realised_spread": None if not sds else float(np.sqrt(np.mean(np.square(sds)))),
        }
    wb = {
        key: _mean([v[key] for v in per_world.values()])
        for key in ("slope", "mae", "bias", "realised_spread")
    }
    wb["rows"] = int(sum(v["rows"] for v in per_world.values()))
    wb["worlds"] = len(per_world)
    spread = wb["realised_spread"]
    wb["abs_bias_over_spread"] = (
        None if wb["bias"] is None or not spread else abs(wb["bias"]) / spread
    )
    return {"world_balanced": wb, "per_world": {str(w): v for w, v in per_world.items()}}


def pair_metrics(
    rows: Mapping[str, np.ndarray], j: int, *, exclude_controls: bool = True
) -> dict[str, Any]:
    """Same-group pair sign accuracy of objective ``j`` over informative pairs
    (|y_a - y_b| > 2 sqrt(se_a^2 + se_b^2)); a predicted tie scores 0.5. World-balanced
    accuracy; per world; per family (pairs of a family member with H12)."""
    world_ok: Counter[int] = Counter()
    world_n: Counter[int] = Counter()
    fam_ok: Counter[str] = Counter()
    fam_n: Counter[str] = Counter()
    keep = ~rows["control"] if exclude_controls else np.ones(len(rows["group"]), dtype=bool)
    for gid in np.unique(rows["group"]):
        idx = np.nonzero((rows["group"] == gid) & keep)[0]
        for a_pos in range(len(idx)):
            for b_pos in range(a_pos + 1, len(idx)):
                a, b = idx[a_pos], idx[b_pos]
                dy = rows["y"][a, j] - rows["y"][b, j]
                se = math.hypot(rows["se"][a, j], rows["se"][b, j])
                if dy == 0.0 or abs(dy) <= INFORMATIVE_PAIR_SE * se:
                    continue
                dp = rows["pred"][a, j] - rows["pred"][b, j]
                score = 0.5 if dp == 0.0 else float(np.sign(dp) == np.sign(dy))
                w = int(rows["world"][a])
                world_ok[w] += score
                world_n[w] += 1
                if rows["anchor"][a] != rows["anchor"][b]:
                    fam = str(rows["family"][b if rows["anchor"][a] else a])
                    fam_ok[fam] += score
                    fam_n[fam] += 1
    per_world = {
        str(w): {"pairs": world_n[w], "accuracy": world_ok[w] / world_n[w]} for w in sorted(world_n)
    }
    per_family = {f: {"pairs": fam_n[f], "accuracy": fam_ok[f] / fam_n[f]} for f in sorted(fam_n)}
    return {
        "pairs": int(sum(world_n.values())),
        "accuracy": _mean([v["accuracy"] for v in per_world.values()]),
        "per_world": per_world,
        "per_family": per_family,
        "worst_world": min(per_world.items(), key=lambda kv: kv[1]["accuracy"])[0]
        if per_world
        else None,
        "worst_family": min(per_family.items(), key=lambda kv: kv[1]["accuracy"])[0]
        if per_family
        else None,
    }


def top1_metrics(
    rows: Mapping[str, np.ndarray], j: int, *, exclude_controls: bool = True
) -> dict[str, Any]:
    """Share of groups whose predicted best candidate is the realised best, over groups
    whose realised best beats the runner-up informatively (report only)."""
    ok: Counter[int] = Counter()
    n: Counter[int] = Counter()
    keep = ~rows["control"] if exclude_controls else np.ones(len(rows["group"]), dtype=bool)
    for gid in np.unique(rows["group"]):
        idx = np.nonzero((rows["group"] == gid) & keep)[0]
        if len(idx) < 2:
            continue
        y, se = rows["y"][idx, j], rows["se"][idx, j]
        order = np.argsort(-y, kind="mergesort")
        best, second = order[0], order[1]
        if y[best] - y[second] <= INFORMATIVE_PAIR_SE * math.hypot(se[best], se[second]):
            continue
        w = int(rows["world"][idx[0]])
        n[w] += 1
        ok[w] += int(int(np.argmax(rows["pred"][idx, j])) == int(best))
    per_world = {str(w): {"groups": n[w], "top1": ok[w] / n[w]} for w in sorted(n)}
    return {"groups": int(sum(n.values())), "top1": _mean([v["top1"] for v in per_world.values()])}


SPLITS = ("all", "survivors", "non_survivors")


def _splits(rows: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    return {
        "all": dict(rows),
        "survivors": _subset(rows, rows["survivor"]),
        "non_survivors": _subset(rows, ~rows["survivor"]),
    }


def sign_accuracy(
    head: LaborAdvantageHead,
    contexts: Mapping[str, torch.Tensor],
    roots: Sequence[Root],
    targets: StageOneTargets,
    name: str,
) -> float | None:
    """World-balanced informative-pair sign accuracy (controls excluded), one objective."""
    if not roots or not len(targets):
        return None
    rows = evaluation_rows(roots, targets, anchored_predictions(head, contexts, roots))
    return pair_metrics(rows, COMPONENTS.index(name))["accuracy"]


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def _critic_snapshot(task: nn.Module, costs: nn.Module) -> dict[str, torch.Tensor]:
    out = {f"task.{k}": v.detach().clone() for k, v in task.state_dict().items()}
    out.update({f"costs.{k}": v.detach().clone() for k, v in costs.state_dict().items()})
    return out


def _same(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> bool:
    return set(a) == set(b) and all(
        a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in a
    )


def fit_stage1(
    task_critic: nn.Module,
    cost_critics: nn.Module,
    candidates: CandidateSet,
    targets: StageOneTargets,
    config: AdvantageFitConfig | None = None,
    *,
    stability: Any = "auto",
    frozen_actor: nn.Module | None = None,
) -> tuple[LaborAdvantageHead, dict[str, Any]]:
    """Fit A on stage-1 targets of fitting worlds (anchored Huber, V frozen)."""
    from .labor_device import module_device
    from .labor_stability import StabilityMonitor, resolve_stability
    from .labor_value_calibration import labor_critic_digest

    config = config or AdvantageFitConfig(target_source=LAMBDA_TARGET_SOURCE)
    if config.target_source != LAMBDA_TARGET_SOURCE:
        raise ValueError("fit_stage1 needs target_source=LAMBDA_TARGET_SOURCE")
    la.require_advantage_fit_worlds(candidates.worlds)
    _require_population(candidates.roots)
    device = module_device(task_critic)
    digest_before = labor_critic_digest(task_critic, cost_critics)
    params_before = _critic_snapshot(task_critic, cost_critics)
    fit_worlds, held_worlds = la.split_worlds(
        candidates.worlds, config.holdout_fraction, config.seed
    )
    fit_set, held_set = candidates.subset(fit_worlds), candidates.subset(held_worlds)
    fit_t = targets.for_roots([r.root_id for r in fit_set.roots])
    held_t = targets.for_roots([r.root_id for r in held_set.roots])
    tensors, privileged = fit_set.root_tensors()
    v_before = la.state_values(task_critic, cost_critics, tensors, privileged)
    contexts = la.critic_contexts(task_critic, cost_critics, tensors, privileged)
    h_contexts = None
    if held_set.roots:
        h_tensors, h_priv = held_set.root_tensors()
        h_contexts = la.critic_contexts(task_critic, cost_critics, h_tensors, h_priv)
    support = block_support(fit_set.roots, fit_t)
    masked, low_support = masked_blocks_for(support, config)

    torch.manual_seed(config.seed)
    head = LaborAdvantageHead.for_critics(
        task_critic, cost_critics, embedding=config.embedding, hidden=config.hidden
    )
    head.set_masked_blocks(masked)
    head.to(device)
    flat = la._flatten(fit_set.roots)
    features = flat.features.to(device)
    root_index = flat.root_index.to(device)
    position = {r.root_id: i for i, r in enumerate(fit_set.roots)}
    by_root: dict[int, list[int]] = defaultdict(list)
    for row, root_id in enumerate(fit_t.root_id.tolist()):
        by_root[position[root_id]].append(row)
    row_i = np.asarray(
        [
            flat.offsets[position[r]] + int(c)
            for r, c in zip(fit_t.root_id, fit_t.candidate, strict=True)
        ],
        dtype=np.int64,
    )
    row_h = np.asarray(
        [flat.offsets[position[r]] + int(h) for r, h in zip(fit_t.root_id, fit_t.h12, strict=True)],
        dtype=np.int64,
    )
    y_all = torch.as_tensor(fit_t.y, dtype=torch.float32)
    se_all = torch.as_tensor(fit_t.se, dtype=torch.float32)
    trainable = sorted(by_root)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng([la.SHUFFLE_TAG, int(config.seed)])
    stability_config = resolve_stability(stability, r5=True)
    monitor = None
    if stability_config is not None:
        blocks = {"action_encoder": list(head.action_encoder.parameters())}
        for name in COMPONENTS:
            blocks[f"head:{name}"] = list(head.heads[name].parameters())
        frozen = {"task_critic": task_critic, "cost_critics": cost_critics}
        if frozen_actor is not None:
            frozen["actor"] = frozen_actor
        monitor = StabilityMonitor(
            stability_config,
            fit="advantage-fit-stage1",
            optimizer=optimizer,
            parameters=list(head.parameters()),
            blocks=blocks,
            frozen=frozen,
            shared_heads={
                "action_encoder": (blocks["action_encoder"], ("qol_years", "population_years"))
            },
            applicable={"advantage_overfit": True},
        )
    history: list[float] = []
    for epoch_number in range(config.epochs if trainable else 0):
        order = [trainable[i] for i in rng.permutation(len(trainable))]
        total = 0.0
        for start in range(0, len(order), config.batch_roots):
            batch = order[start : start + config.batch_roots]
            optimizer.zero_grad()
            members = torch.cat(
                [
                    torch.arange(
                        flat.offsets[r], flat.offsets[r] + len(fit_set.roots[r].candidates)
                    )
                    for r in batch
                ]
            )
            local = torch.full((len(flat.root_index),), -1, dtype=torch.long)
            local[members] = torch.arange(len(members))
            members_on = members.to(device)
            raw = head(contexts, features[members_on], root_index[members_on])
            rows = np.asarray([row for r in batch for row in by_root[r]], dtype=np.int64)
            li = local[torch.as_tensor(row_i[rows])].to(device)
            lh = local[torch.as_tensor(row_h[rows])].to(device)
            y_b = y_all[rows].to(device)
            se_b = se_all[rows].to(device)
            component_losses = {
                name: anchored_huber_loss(
                    raw[name][li],
                    raw[name][lh],
                    y_b[:, j],
                    se_b[:, j],
                    c=config.huber_se_multiplier,
                )
                for j, name in enumerate(COMPONENTS)
            }
            loss = sum(component_losses.values())
            if monitor is None:
                if not torch.isfinite(loss):
                    raise ValueError("advantage fit produced a nonfinite loss")
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), config.max_grad_norm)
                optimizer.step()
            elif not monitor.step(
                loss, max_grad_norm=config.max_grad_norm, head_losses=component_losses
            ):
                continue
            total += float(loss.detach()) * len(batch)
        history.append(total / max(1, len(order)))
        if monitor is not None:
            acc = {
                label: {
                    name: sign_accuracy(head, ctx, roots, t, name)
                    for name in ("qol_years", "population_years")
                }
                for label, roots, ctx, t in (
                    ("train", fit_set.roots, contexts, fit_t),
                    ("heldout", held_set.roots, h_contexts, held_t),
                )
            }
            monitor.epoch_ranking(
                epoch_number,
                acc["train"]["qol_years"],
                acc["heldout"]["qol_years"],
                population_years={k: acc[k]["population_years"] for k in acc},
                loss=history[-1],
            )
    v_after = la.state_values(task_critic, cost_critics, tensors, privileged)
    params_after = _critic_snapshot(task_critic, cost_critics)
    digest_after = labor_critic_digest(task_critic, cost_critics)
    v_guard = {
        "digest_identical": digest_after == digest_before,
        "parameters_identical": _same(params_before, params_after),
        "outputs_identical": all(torch.equal(v_before[n], v_after[n]) for n in COMPONENTS),
        "rows_checked": int(tensors["core"].shape[0]),
    }
    v_guard["identical"] = all(
        v_guard[k] for k in ("digest_identical", "parameters_identical", "outputs_identical")
    )
    if not v_guard["identical"]:
        raise AssertionError(f"stage-1 A fit changed the frozen critic: {v_guard}")
    summary: dict[str, Any] = {
        "target_source": LAMBDA_TARGET_SOURCE,
        "loss": "anchored-huber-inverse-noise-variance",
        "huber_se_multiplier": config.huber_se_multiplier,
        "fit_worlds": fit_worlds,
        "heldout_worlds": held_worlds,
        "fit_roots": len(fit_set.roots),
        "heldout_roots": len(held_set.roots),
        "trainable_roots": len(trainable),
        "fit_rows": len(fit_t),
        "fit_rows_by_survival": {
            "survivors": int(fit_t.survivor.sum()),
            "non_survivors": int((~fit_t.survivor).sum()),
        },
        "heldout_rows": len(held_t),
        "targets": targets.summary(),
        "block_support": support,
        "support_floor": config.support_floor,
        "masked_blocks": masked,
        "masked_for_low_support": low_support,
        "loss_history": history,
        "critic_digest": digest_before,
        "v_guard": v_guard,
        "parameters": la.parameter_counts(head),
    }
    if monitor is not None:
        summary["stability"] = monitor.finish()
    if held_set.roots and len(held_t):
        rows = evaluation_rows(
            held_set.roots, held_t, anchored_predictions(head, h_contexts, held_set.roots)
        )
        summary["heldout_sign_accuracy"] = {
            name: pair_metrics(rows, j)["accuracy"] for j, name in enumerate(COMPONENTS)
        }
    return head, summary


# ---------------------------------------------------------------------------
# Gate report
# ---------------------------------------------------------------------------


def block_response(
    head: LaborAdvantageHead, contexts: Mapping[str, torch.Tensor], roots: Sequence[Root]
) -> dict[str, dict[str, float]]:
    """Mean |A(a) - A(a with block b set to H12's)| per block and objective over every
    non-H12 candidate (exactly 0 for a masked block)."""
    from .labor_device import module_device

    flat = la._flatten(roots)
    device = module_device(head)
    h12_rows = torch.as_tensor(
        [flat.offsets[int(r)] + roots[int(r)].h12 for r in flat.root_index.tolist()]
    )
    non_h12 = torch.as_tensor(
        [
            i != flat.offsets[int(r)] + roots[int(r)].h12
            for i, r in enumerate(flat.root_index.tolist())
        ]
    )
    out: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        base = head(contexts, flat.features.to(device), flat.root_index.to(device))
        for name, sl in la.block_slices().items():
            swapped = flat.features.clone()
            swapped[:, sl] = flat.features[h12_rows][:, sl]
            moved = head(contexts, swapped.to(device), flat.root_index.to(device))
            out[name] = {
                n: float((base[n] - moved[n]).abs().cpu()[non_h12].mean()) if non_h12.any() else 0.0
                for n in COMPONENTS
            }
    return out


def seed_agreement(
    heads: Sequence[LaborAdvantageHead],
    contexts: Mapping[str, torch.Tensor],
    roots: Sequence[Root],
) -> dict[str, Any]:
    """Mean within-root Spearman correlation of anchored A between every pair of seeds
    (roots with >= 3 candidates and non-constant predictions); the minimum over pairs."""
    from .labor_sa_advantage import _corr, _rank

    preds = [anchored_predictions(h, contexts, roots) for h in heads]
    out: dict[str, Any] = {}
    for name in COMPONENTS:
        pairs = {}
        for a in range(len(heads)):
            for b in range(a + 1, len(heads)):
                corrs = []
                for r in range(len(roots)):
                    pa, pb = preds[a][name][r], preds[b][name][r]
                    if len(pa) < 3:
                        continue
                    c = _corr(_rank(pa), _rank(pb))
                    if c is not None:
                        corrs.append(c)
                pairs[f"{a}-{b}"] = {"roots": len(corrs), "spearman": _mean(corrs)}
        values = [v["spearman"] for v in pairs.values() if v["spearman"] is not None]
        out[name] = {"pairs": pairs, "min": None if not values else float(min(values))}
    return out


def ppo_dry_run(
    actor: nn.Module,
    head: LaborAdvantageHead,
    task_critic: nn.Module,
    cost_critics: nn.Module,
    candidates: CandidateSet,
    *,
    epochs: int = PPO_DRY_RUN_EPOCHS,
    clip: float = PPO_CLIP,
    lr: float = PPO_DRY_RUN_LR,
    max_roots: int = PPO_DRY_RUN_ROOTS,
    seed: int = 0,
) -> dict[str, Any]:
    """K PPO epochs on one fixed batch, on a COPY of the actor (the actor is untouched).

    The batch: up to ``max_roots`` roots; one keyed labor sample per root
    (``sample_labor_policy``); the sampled shares replace H12's labor blocks (specialty
    after apportionment at the root population; H12's movement, fission and trade kept);
    advantage = the root personality's utility of the anchored A, standardized over the
    batch. Per epoch, before that epoch's update: approx-KL (k3 estimator,
    mean((r - 1) - log r)) and clip fraction (|r - 1| > clip). Labor log-probabilities
    only (non-labor heads are not resampled). Report only; it sets K.
    """
    from .labor_device import module_device
    from .labor_model import labor_log_prob, sample_labor_policy
    from .labor_ppo import _clipped_surrogate

    roots = candidates.roots[:max_roots]
    _require_population(roots)
    subset = CandidateSet(
        roots, candidates.tensors, candidates.privileged, candidates.rows[: len(roots)]
    )
    tensors, privileged = subset.root_tensors()
    contexts = la.critic_contexts(task_critic, cost_critics, tensors, privileged)
    before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    work = copy.deepcopy(actor)
    work.train()
    device = module_device(work)
    tensors_on = {k: v.to(device) for k, v in tensors.items()}
    with torch.no_grad():
        out = work(tensors_on)
        sample = sample_labor_policy(
            out,
            tensors_on,
            generator=torch.Generator(device=device).manual_seed(
                int(np.random.default_rng([DRY_RUN_TAG, int(seed)]).integers(2**31))
            ),
        )
        old_logp = labor_log_prob(out, sample, tensors_on)["joint"].detach()
    spec = sample["specialty_allocation"].double().cpu().numpy()
    act = sample["activity_allocation"].double().cpu().numpy()
    slices = la.block_slices()
    features = []
    for r, root in enumerate(roots):
        vector = root.candidates[root.h12].features.astype(np.float64).copy()
        vector[slices["specialty_share"]] = la.apportioned_specialty(spec[r], root.population)
        vector[slices["activity_share"]] = act[r].reshape(-1)
        features.append(vector.astype(np.float32))
    head_device = module_device(head)
    feats = torch.as_tensor(np.stack(features)).to(head_device)
    index = torch.arange(len(roots), device=head_device)
    anchor = torch.as_tensor(np.stack([r.candidates[r.h12].features for r in roots])).to(
        head_device
    )
    with torch.no_grad():
        a = head(contexts, feats, index)
        a0 = head(contexts, anchor, index)
    utility = np.zeros(len(roots))
    for r, root in enumerate(roots):
        w = la.utility_weights(root.personality)
        utility[r] = sum(w[n] * float(a[n][r] - a0[n][r]) for n in COMPONENTS)
    std = float(utility.std())
    adv = torch.as_tensor(
        (utility - utility.mean()) / (std if std > 0 else 1.0), dtype=old_logp.dtype
    ).to(device)
    optimizer = torch.optim.Adam(work.parameters(), lr=lr)
    records = []
    for epoch in range(int(epochs)):
        optimizer.zero_grad()
        logp = labor_log_prob(work(tensors_on), sample, tensors_on)["joint"]
        ratio = torch.exp(logp - old_logp)
        with torch.no_grad():
            log_ratio = logp - old_logp
            records.append(
                {
                    "epoch": epoch,
                    "approx_kl": float(((ratio - 1.0) - log_ratio).mean()),
                    "clip_fraction": float(((ratio - 1.0).abs() > clip).double().mean()),
                }
            )
        loss = -_clipped_surrogate(ratio, adv, clip).mean()
        loss.backward()
        optimizer.step()
    after = actor.state_dict()
    return {
        "epochs": int(epochs),
        "clip": float(clip),
        "lr": float(lr),
        "roots": len(roots),
        "per_epoch": records,
        "actor_untouched": all(torch.equal(before[k], after[k]) for k in before),
        "note": "labor heads only; updates applied to a copy of the actor",
    }


def _json_safe(value: Any) -> Any:
    """Plain JSON values: numpy scalars to Python, non-finite floats to None."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _gate(value_pass: bool | None, **detail: Any) -> dict[str, Any]:
    status = "NO_SIGNAL" if value_pass is None else ("PASS" if value_pass else "FAIL")
    return {"status": status, **detail}


def stage1_gate_report(
    heads: Sequence[LaborAdvantageHead],
    task_critic: nn.Module,
    cost_critics: nn.Module,
    candidates: CandidateSet,
    targets: StageOneTargets,
    fit_summaries: Sequence[Mapping[str, Any]],
    *,
    actor: nn.Module | None = None,
    ppo_epochs: int = PPO_DRY_RUN_EPOCHS,
    stability_abort: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Every gate of the spec table on the evaluation roots (``heads[0]`` is the primary
    head; the others are the extra seeds). A gate with no informative data is
    ``NO_SIGNAL`` and listed, not passed; ``status`` is PASS only when every gate that
    has signal passes, every seed's V guard holds and the stability harness admitted."""
    from .labor_bc import SEALED_WORLDS
    from .labor_privileged import RESERVED_HELDOUT_WORLDS

    if not heads:
        raise ValueError("the gate report needs at least one fitted head")
    worlds = candidates.worlds
    if any(w in SEALED_WORLDS for w in worlds):
        raise ValueError("sealed worlds 3000-3059 are never evaluated here")
    fit_worlds = {w for s in fit_summaries for w in s.get("fit_worlds", [])}
    if fit_worlds & set(worlds):
        raise ValueError("evaluation worlds overlap the advantage fit worlds")
    _require_population(candidates.roots)
    targets = targets.for_roots([r.root_id for r in candidates.roots])
    head = heads[0]
    tensors, privileged = candidates.root_tensors()
    contexts = la.critic_contexts(task_critic, cost_critics, tensors, privileged)
    rows = evaluation_rows(
        candidates.roots, targets, anchored_predictions(head, contexts, candidates.roots)
    )
    splits = _splits(rows)
    gates: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    ordering: dict[str, Any] = {}
    for j, name in enumerate(COMPONENTS):
        calibration[name] = {s: calibration_metrics(r, j) for s, r in splits.items()}
        ordering[name] = {
            s: {
                "no_controls": pair_metrics(r, j, exclude_controls=True),
                "with_controls": pair_metrics(r, j, exclude_controls=False),
                "top1_no_controls": top1_metrics(r, j),
            }
            for s, r in splits.items()
        }
        wb = calibration[name]["all"]["world_balanced"]
        if wb["slope"] is None or not wb["realised_spread"]:
            ok = None
        else:
            ok = (
                CALIBRATION_SLOPE_RANGE[0] <= wb["slope"] <= CALIBRATION_SLOPE_RANGE[1]
                and wb["abs_bias_over_spread"] < CALIBRATION_BIAS_FRACTION
            )
        gates[f"calibration:{name}"] = _gate(
            ok,
            slope=wb["slope"],
            mae=wb["mae"],
            bias=wb["bias"],
            realised_spread=wb["realised_spread"],
            abs_bias_over_spread=wb["abs_bias_over_spread"],
        )
        order = ordering[name]["all"]["no_controls"]
        gates[f"ordering:{name}"] = _gate(
            None if order["accuracy"] is None else order["accuracy"] >= ORDERING_SIGN_ACCURACY,
            value=order["accuracy"],
            pairs=order["pairs"],
        )
        per_world = {w: v["accuracy"] for w, v in order["per_world"].items()}
        per_family = {f: v["accuracy"] for f, v in order["per_family"].items()}
        worst = [v for v in (*per_world.values(), *per_family.values())]
        gates[f"no_single_path:{name}"] = _gate(
            None if not worst else min(worst) >= PATH_SIGN_ACCURACY_FLOOR,
            worst_world=order["worst_world"],
            worst_world_accuracy=None if not per_world else min(per_world.values()),
            worst_family=order["worst_family"],
            worst_family_accuracy=None if not per_family else min(per_family.values()),
            per_world_calibration_slope={
                w: v["slope"] for w, v in calibration[name]["all"]["per_world"].items()
            },
        )
    # Per-family calibration (slope/MAE over the family's rows plus their anchors).
    family_calibration: dict[str, Any] = {}
    for fam in sorted({str(f) for f in rows["family"][~rows["anchor"]]}):
        groups = set(rows["group"][(rows["family"] == fam) & ~rows["anchor"]].tolist())
        m = ((rows["family"] == fam) | rows["anchor"]) & np.isin(rows["group"], list(groups))
        family_calibration[fam] = {
            name: calibration_metrics(_subset(rows, m), j)["world_balanced"]
            for j, name in enumerate(COMPONENTS)
        }
    # Block support and response.
    masked = set(head.masked_blocks)
    response = block_response(head, contexts, candidates.roots)
    support_fit = fit_summaries[0].get("block_support", {})
    support_eval = block_support(candidates.roots, targets)
    masked_zero = all(response[b][n] == 0.0 for b in masked for n in COMPONENTS)
    floor = fit_summaries[0].get("support_floor")
    unmasked_supported = all(
        support_fit.get(b, 0) >= (floor or 0) for b in response if b not in masked
    )
    gates["block_support"] = _gate(
        masked_zero and unmasked_supported,
        masked_blocks=sorted(masked),
        masked_for_low_support=fit_summaries[0].get("masked_for_low_support"),
        masked_response_exactly_zero=masked_zero,
        support_floor=floor,
        fit_support=support_fit,
        eval_support=support_eval,
        response=response,
    )
    guards = [s.get("v_guard", {}) for s in fit_summaries]
    gates["v_non_regression"] = _gate(
        bool(guards) and all(g.get("identical") for g in guards), per_seed=guards
    )
    if len(heads) >= 2:
        agreement = seed_agreement(heads, contexts, candidates.roots)
        mins = [v["min"] for v in agreement.values() if v["min"] is not None]
        gates["seed_agreement"] = _gate(
            None if not mins else min(mins) >= SEED_RANK_CORRELATION,
            seeds=len(heads),
            per_objective=agreement,
        )
    else:
        gates["seed_agreement"] = {"status": "NOT_RUN", "reason": "needs >= 2 seeds"}
    stability = []
    for s in fit_summaries:
        st = s.get("stability")
        if st is None:
            stability.append({"monitor": "off"})
            continue
        overfit = (st.get("checks") or {}).get("advantage_overfit", {})
        stability.append(
            {"admitted": st.get("admitted"), "advantage_overfit": overfit.get("status")}
        )
    stable = stability_abort is None and all(
        x.get("admitted") is True and x.get("advantage_overfit") == "PASS" for x in stability
    )
    if stability_abort is not None:
        stability_status = "FAIL"
    elif stability and all("monitor" in x for x in stability):
        stability_status = "NOT_RUN"
    else:
        stability_status = "PASS" if stable else "FAIL"
    gates["stability"] = {
        "status": stability_status,
        "per_seed": stability,
        "abort": stability_abort,
        "thresholds": {"overfit_gap": OVERFIT_GAP, "overfit_heldout_drop": OVERFIT_HELDOUT_DROP},
    }
    if actor is None:
        gates["ppo_readiness"] = {"status": "SKIPPED", "reason": "no actor checkpoint given"}
    else:
        gates["ppo_readiness"] = {
            **ppo_dry_run(actor, head, task_critic, cost_critics, candidates, epochs=ppo_epochs),
            "status": "REPORT_ONLY",
        }
    gating = {k: v["status"] for k, v in gates.items() if k != "ppo_readiness"}
    failed = sorted(k for k, s in gating.items() if s in ("FAIL", "NOT_RUN"))
    no_signal = sorted(k for k, s in gating.items() if s == "NO_SIGNAL")
    reserved = set(RESERVED_HELDOUT_WORLDS)
    return _json_safe(
        {
            "schema": GATE_SCHEMA,
            "worlds": worlds,
            "reserved_worlds_only": bool(worlds) and set(worlds) <= reserved,
            "roots": len(candidates.roots),
            "targets": targets.summary(),
            "thresholds": {
                "calibration_slope_range": list(CALIBRATION_SLOPE_RANGE),
                "calibration_abs_bias_over_spread": CALIBRATION_BIAS_FRACTION,
                "ordering_sign_accuracy": ORDERING_SIGN_ACCURACY,
                "no_single_path_sign_accuracy_floor": PATH_SIGN_ACCURACY_FLOOR,
                "seed_rank_correlation": SEED_RANK_CORRELATION,
                "overfit_gap": OVERFIT_GAP,
                "overfit_heldout_drop": OVERFIT_HELDOUT_DROP,
            },
            "definitions": {
                "rows": "non-H12 candidates with targets, plus one H12 anchor row (y = 0, "
                "prediction 0) per (root, estimator class) group; prediction = A(a) - A(h12)",
                "calibration": "world-balanced; slope of realised y on predicted after "
                "per-group demeaning; MAE/bias over non-anchor rows; realised spread = RMS "
                "within-group SD of y",
                "informative_pair": f"same group, |dy| > {INFORMATIVE_PAIR_SE} sqrt(se_a^2 + "
                "se_b^2); predicted tie scores 0.5",
                "per_family": "pairs of a family member with H12",
                "splits": "all / survivors (folded estimator) / non_survivors (one-step)",
                "gates_evaluated_on": "split 'all'",
            },
            "gates": gates,
            "calibration": calibration,
            "ordering": ordering,
            "family_calibration": family_calibration,
            "fit": [
                {k: v for k, v in s.items() if k not in ("loss_history", "stability")}
                for s in fit_summaries
            ],
            "failed": failed,
            "no_signal": no_signal,
            "status": "PASS" if not failed else "FAIL",
        }
    )


# ---------------------------------------------------------------------------
# Synthetic export-v2 shards (tests only; never real data)
# ---------------------------------------------------------------------------


def synthetic_stage1_shard(
    *,
    worlds: Sequence[int] = (90100, 90101, 90102, 90103, 90104),
    roots_per_world: int = 6,
    candidates: int = 8,
    survivors: int = 4,
    final_horizon: int = 3,
    streams: int = 3,
    population: int = 23,
    noise: float = 0.01,
    seed: int = 0,
    core_width: int = 10,
    structure_seed: int = 0,
) -> dict[str, Any]:
    """An export-v2 shard with a planted return structure, built on
    ``labor_advantage.synthetic_candidate_shard`` (actions, controls, twins, tensors).

    Candidate quality ``q`` (4 objectives) is a fixed random linear function (keyed by
    ``structure_seed``, shared across ``seed``s) of the post-apportionment action
    vector with zero weight on the trade value/share blocks.
    Per stream: every step's reward is ``0.1 q + root offset + noise`` (reference
    units x scale) and the successor row at decision d carries ``V = 0.5 q + offset``
    in its first four core columns (read by :func:`synthetic_values`). The first
    ``survivors`` candidates (H12 included) reach ``final_horizon`` with streams 0..S;
    the others stop at horizon 1 with stream 0 only.
    """
    base = la.synthetic_candidate_shard(
        worlds=worlds,
        roots_per_world=roots_per_world,
        candidates=candidates,
        horizons=(1,),
        streams=0,
        seed=seed,
        core_width=core_width,
    )
    direction = np.random.default_rng([0x5332, int(structure_seed)]).normal(
        size=(len(COMPONENTS), la.ACTION_FEATURE_WIDTH)
    )
    rng = np.random.default_rng([0x5331, int(seed)])
    slices = la.block_slices()
    for name in la.STAGE1_MASKED_BLOCKS:
        direction[:, slices[name]] = 0.0
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    scale = np.asarray([REFERENCE_SCALES[n] for n in COMPONENTS])
    families = ("transfer", "activity", "movement")
    succ_core: list[np.ndarray] = []
    years = 20.0
    k = int(final_horizon)
    for root in base["roots"]:
        root["population"] = int(population)
        root["successor_decisions"] = sorted({1, k})
        root["final_horizon"] = k
        offset = rng.normal(size=len(COMPONENTS)) * 0.5
        for cand in root["candidates"]:
            i = cand["index"]
            if not cand["is_h12"] and cand["control"] is None and cand["bookkeeping_twin"] is None:
                cand["family"] = families[i % len(families)]
            feats = la.action_features(cand["action"], population=population).astype(np.float64)
            q = 2.0 * direction @ feats
            if cand["control"] == "all-idle":
                q = np.array([-3.0, -3.0, 3.0, 3.0])
            survivor = i < survivors
            horizon = k if survivor else 1
            stream_ids = range(streams + 1) if survivor else (0,)
            evaluations, succ_streams = [], []
            for s in stream_ids:
                steps = [
                    ((0.1 * q + offset + rng.normal(scale=noise, size=len(COMPONENTS))) * scale)
                    for _ in range(horizon)
                ]
                boundaries = []
                for d in range(1, horizon + 1):
                    row = None
                    if d in (1, k):
                        v = 0.5 * q + offset + rng.normal(scale=noise, size=len(COMPONENTS))
                        succ_core.append(v)
                        row = len(succ_core) - 1
                    boundaries.append({"decision": d, "turn": 10 + d, "status": "live", "row": row})
                succ_streams.append(
                    {
                        "stream": s,
                        "step_rewards": [list(map(float, r)) for r in steps],
                        "step_years": [years] * horizon,
                        "terminated": False,
                        "boundaries": boundaries,
                    }
                )
                for h in sorted({1, horizon}) if s == 0 else (horizon,):
                    g, total = 1.0, np.zeros(len(COMPONENTS))
                    for r in steps[:h]:
                        total = total + g * r
                        g *= math.exp(-years / DISCOUNT_YEARS)
                    evaluations.append(
                        {
                            "horizon": h,
                            "reached_horizon": h,
                            "terminated": False,
                            "stream": s,
                            "returns": [float(x) for x in total],
                            "step_years": [years] * h,
                            "se": None,
                        }
                    )
            cand["evaluations"] = evaluations
            cand["successor_streams"] = succ_streams
        root["pair_sd"] = None
    rows = len(succ_core)
    tensors = la.synthetic_root_tensors(rows, core_width, seed + 1)
    if rows:
        tensors["core"][:, : len(COMPONENTS)] = torch.as_tensor(
            np.stack(succ_core), dtype=torch.float32
        )
    base["successor_tensors"] = tensors
    base["successor_privileged"] = None
    base["successor_row_count"] = rows
    return base


def synthetic_values(data: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Fake V for synthetic shards: core[:, j] x reference scale (raw units)."""
    core = data["core"].double().numpy() if data else np.zeros((0, len(COMPONENTS)))
    return {name: core[:, j] * REFERENCE_SCALES[name] for j, name in enumerate(COMPONENTS)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def argument_parser() -> argparse.ArgumentParser:
    from .labor_device import add_device_argument
    from .labor_stability import add_stability_arguments

    parser = argparse.ArgumentParser(
        description="Stage-1 search pretraining of the labor advantage head A(s,a)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit A on export-v2 search payloads and write the gate report")
    fit.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="frozen labor critic checkpoint (V and A's context)",
    )
    fit.add_argument(
        "--payloads", type=Path, nargs="+", required=True, help="fitting payload dirs/files"
    )
    fit.add_argument(
        "--eval-payloads",
        type=Path,
        nargs="*",
        default=None,
        help="gate payloads (reserved worlds 90154-90159); default: the held-out fit worlds",
    )
    fit.add_argument(
        "--value-checkpoints",
        default=None,
        help="comma-separated critic checkpoints whose per-row mean V bootstraps the "
        "targets (default: --checkpoint)",
    )
    fit.add_argument(
        "--out", type=Path, required=True, help="advantage-head checkpoint (primary seed)"
    )
    fit.add_argument("--report", type=Path, required=True, help="gate report JSON")
    fit.add_argument(
        "--seeds", default="0,1", help="comma-separated fit seeds (>= 2 for seed agreement)"
    )
    fit.add_argument("--epochs", type=int, default=50)
    fit.add_argument("--batch-roots", type=int, default=32)
    fit.add_argument("--lr", type=float, default=3e-4)
    fit.add_argument("--holdout-fraction", type=float, default=0.2)
    fit.add_argument("--support-floor", type=int, default=la.BLOCK_SUPPORT_FLOOR)
    fit.add_argument("--huber-se-multiplier", type=float, default=la.HUBER_SE_MULTIPLIER)
    fit.add_argument(
        "--actor-checkpoint",
        type=Path,
        default=None,
        help="actor for the PPO-readiness dry run (skipped without)",
    )
    fit.add_argument("--ppo-epochs", type=int, default=PPO_DRY_RUN_EPOCHS)
    add_stability_arguments(fit)
    add_device_argument(fit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from .labor_device import cpu_copy, device_record, setup
    from .labor_stability import StabilityAbort, stability_config_from_args
    from .labor_value_calibration import labor_critic_digest

    args = argument_parser().parse_args(argv)
    device = setup(getattr(args, "device", "cpu"))
    task, costs = la._load_critics(args.checkpoint)
    task.to(device)
    costs.to(device)
    values = (
        critic_values(task, costs)
        if not args.value_checkpoints
        else ensemble_values(args.value_checkpoints.split(","))
    )
    fit_data = load_stage1_data(args.payloads)
    fit_targets = build_stage1_targets(fit_data, values)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    if not seeds:
        raise ValueError("at least one seed")
    heads, summaries, abort = [], [], None
    for seed in seeds:
        config = AdvantageFitConfig(
            epochs=args.epochs,
            batch_roots=args.batch_roots,
            lr=args.lr,
            holdout_fraction=args.holdout_fraction,
            seed=seed,
            target_source=LAMBDA_TARGET_SOURCE,
            support_floor=args.support_floor,
            huber_se_multiplier=args.huber_se_multiplier,
        )
        try:
            head, summary = la.fit_advantage_head(
                task,
                costs,
                fit_data.candidates,
                config,
                stability=stability_config_from_args(args),
                frozen_actor=la._load_actor(args.checkpoint),
                targets=fit_targets,
            )
        except StabilityAbort as error:
            abort = {"seed": seed, "summary": error.summary}
            break
        if device.type != "cpu":
            summary["device"] = device_record(device)
        head.to("cpu")
        heads.append(head)
        summaries.append(summary)
        payload = la.advantage_checkpoint_payload(
            head,
            critic_digest=labor_critic_digest(task, costs),
            config=config,
            fit=summary,
            task_context=la.task_context_record(task),
        )
        payload["state_dict"] = cpu_copy(payload["state_dict"])
        path = (
            args.out
            if seed == seeds[0]
            else args.out.with_name(f"{args.out.stem}.seed{seed}{args.out.suffix}")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        head.to(device)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if abort is not None and not heads:
        args.report.write_text(
            json.dumps(
                {"schema": GATE_SCHEMA, "status": "FAIL", "stability_abort": _json_safe(abort)},
                indent=1,
            ),
            encoding="utf-8",
        )
        return 1
    if args.eval_payloads:
        eval_data = load_stage1_data(args.eval_payloads)
        eval_set = eval_data.candidates
        eval_targets = build_stage1_targets(eval_data, values)
    else:
        eval_set = fit_data.candidates.subset(summaries[0]["heldout_worlds"])
        eval_targets = fit_targets
    actor = None
    if args.actor_checkpoint is not None:
        actor = la._load_actor(args.actor_checkpoint)
    report = stage1_gate_report(
        heads,
        task,
        costs,
        eval_set,
        eval_targets,
        summaries,
        actor=actor,
        ppo_epochs=args.ppo_epochs,
        stability_abort=abort,
    )
    args.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "failed": report["failed"],
                "no_signal": report["no_signal"],
            },
            indent=1,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
