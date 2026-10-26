"""r5 trajectory diagnostics T1, T3 and T4 for the labor critics (privileged V plus A).

Approved design: r5 acceptance page, section 4a (26 Oct). These diagnostics are
read-only and **never gate**: they sit beside the gate verdict in every r5
acceptance receipt (``trajectory_diagnostics``) and are compared checkpoint to
checkpoint so a regression shows *which* failure it is. A T-metric that worsens
by more than its seed spread is flagged for review, nothing more. T2 (plan
candidates in search) is a separate exporter change and is not here.

* **T1, myopia.** For every same-root candidate pair of a
  ``labor-advantage-candidates-v1`` payload, the realised discounted return
  difference of the selection branch (stream 0, ``step_rewards``/``step_years``)
  is split into the first decision's part, decisions 2-5 and beyond.
  :func:`t1_advantage_report` reports how A's pair difference correlates with
  each part, least-squares loadings of dA on (first, delayed), and A's ranking
  on *sign-conflict* pairs (first part and delayed outcome disagree in sign).
  A critic that follows the first part on those pairs is myopic.
  :func:`td_error_split` applies the same split to V along a trajectory:
  ``G_t - V_t = sum_j Gamma_j delta_{t+j}`` (one-step TD errors), split into
  ``delta_t``, the discounted errors of decisions 2-5 and the rest.
* **T3, adaptation lag.** :func:`v_adaptation_report` gives V calibration
  (world-balanced nMAE / bias in reference scales) on teacher-visited rows
  (``row_source`` 0), actor-visited rows (``row_source`` 3, teacher
  continuations from DAgger learner states) and the beat-the-teacher subset
  (continuations whose actor branch beat the teacher's own return from the
  same start, :func:`branch_beat_keys`), per DAgger round.
  :func:`a_adaptation_report` gives A's ranking on payload roots by root source
  (``h12`` / ``actor``) and on roots where search beat H12, per labelled
  payload set (round). Per refit is checkpoint-to-checkpoint (:func:`compare`,
  :func:`adaptation_series`).
* **T4, value tracking.** The autonomous evaluator records, with
  ``--record-critic-trace`` (off by default), each head's predicted value and
  the realised rewards per decision (:data:`TRACE_SCHEMA`).
  :func:`t4_report` turns the traces into decision-by-decision curves of
  predicted value against realised discounted return, per-stage summaries
  (early, around a Farmer unlock, around a fission, late) and the decision at
  which the curves separate.

CLI: ``compare`` (two receipts or evidence files: flagged T-metrics),
``series`` (several checkpoints in order: the T3 lag series) and ``t4``
(a T4 report from autonomous reports carrying traces).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "labor-r5-trajectory-diagnostics-v1"
TRACE_SCHEMA = "labor-critic-rollout-trace-v1"
COMPARISON_SCHEMA = "labor-r5-trajectory-comparison-v1"
COMPONENTS = ("qol_years", "population_years", "feasibility", "settlement_guard")
TASK_HEADS = ("qol_years", "population_years")
DAGGER_ROW_SOURCE = 1  # labor_dagger.DAGGER_ROW_SOURCE (actor-only rows, no V target)
DISCOUNT_YEARS = 400.0
PARTS = ("first", "mid", "late")
PART_DEFINITION = {
    "first": "decision 1 (the root decision)",
    "mid": "decisions 2-5",
    "late": "decisions 6 onward (to the common reached horizon or the trajectory end)",
    "delayed": "mid + late",
}
T1_TARGETS = ("utility:root", "qol_years", "population_years")
GROUPS = ("teacher", "actor", "beat")

# Declared, non-gating constants (proposed, for review with the receipt).
SETTINGS: dict[str, dict[str, Any]] = {
    "first_decisions": {"value": 1, "status": "approved (section 4a)"},
    "delayed_split_decisions": {"value": 5, "status": "approved (section 4a: 2-5, then beyond)"},
    "conflict_min_abs": {
        "value": 0.01,
        "status": "proposed",
        "source": "a sign-conflict part must exceed the false-improvement margin "
        "(reference scales)",
    },
    "beat_margin": {
        "value": 0.01,
        "status": "proposed",
        "source": "actor/search beats H12 by more than the false-improvement margin "
        "(utility, reference scales; single-rollout and tail noise not removed)",
    },
    "separation_threshold": {
        "value": 0.10,
        "status": "proposed",
        "source": "mean |V - G| in reference scales, the calibration |bias| bound",
    },
    "separation_persist": {"value": 2, "status": "proposed", "source": "consecutive decisions"},
    "stage_window": {
        "value": 2,
        "status": "proposed",
        "source": "decisions either side of an event",
    },
    "lag_gap": {
        "value": 0.10,
        "status": "proposed",
        "source": "beat-minus-teacher nMAE above this at the latest refit, not shrinking: lagging",
    },
    "gating": {"value": False, "status": "approved (section 4a: never gate)"},
}


def setting(name: str) -> Any:
    return SETTINGS[name]["value"]


# ---------------------------------------------------------------------------
# Small numerics
# ---------------------------------------------------------------------------


def _reference_scales() -> dict[str, float]:
    from .labor_model import LABOR_COST_REFERENCE_SCALES, LABOR_OBJECTIVE_REFERENCE_SCALES

    return {**LABOR_OBJECTIVE_REFERENCE_SCALES, **LABOR_COST_REFERENCE_SCALES}


def _weights(target: str, personality: Sequence[float] | None) -> dict[str, float]:
    """Normalized-unit coefficients (as labor_advantage.target_weights)."""
    if target in COMPONENTS:
        return {name: float(name == target) for name in COMPONENTS}
    if target.startswith("utility"):
        q, p = (0.5, 0.5) if personality is None else (float(personality[0]), float(personality[1]))
        return {
            "qol_years": q,
            "population_years": p,
            "feasibility": -1.0,
            "settlement_guard": -1.0,
        }
    raise ValueError(f"unknown target {target!r}")


def _scalar(raw: np.ndarray, weights: Mapping[str, float], scales: Mapping[str, float]) -> float:
    return float(sum(weights[n] * float(raw[c]) / scales[n] for c, n in enumerate(COMPONENTS)))


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.size < 3:
        return None
    sx, sy = float(x.std()), float(y.std())
    if sx <= 1e-15 or sy <= 1e-15:
        return None
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def _agree(prediction: float, truth: float) -> float:
    if prediction == 0.0:
        return 0.5
    return float(np.sign(prediction) == np.sign(truth))


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def world_balanced(worlds: Sequence[int]) -> np.ndarray:
    worlds = np.asarray(worlds).reshape(-1)
    weights = np.zeros(len(worlds))
    unique = np.unique(worlds)
    for world in unique:
        rows = worlds == world
        weights[rows] = len(worlds) / (len(unique) * float(rows.sum()))
    return weights


def _wmean(values: np.ndarray, weights: np.ndarray) -> float | None:
    total = float(weights.sum())
    return None if total <= 0 else float((values * weights).sum() / total)


# ---------------------------------------------------------------------------
# T1: per-step contributions of candidate branches
# ---------------------------------------------------------------------------


def discounted_contributions(
    step_rewards: Sequence[Sequence[float]],
    step_years: Sequence[float],
    discount_years: float = DISCOUNT_YEARS,
) -> np.ndarray:
    """``D_t r_t`` per step (raw units, COMPONENTS order), ``D_0 = 1``, physical-time discount."""
    rewards = np.asarray(step_rewards, dtype=np.float64).reshape(-1, len(COMPONENTS))
    if len(step_years) != len(rewards):
        raise ValueError("per-step rewards and years differ in length")
    out = np.zeros_like(rewards)
    discount = 1.0
    for t, (reward, years) in enumerate(zip(rewards, step_years, strict=True)):
        out[t] = discount * reward
        discount *= math.exp(-float(years) / float(discount_years))
    return out


class StepTable:
    """Per (root_id, candidate) selection-branch contributions and termination.

    ``exact`` is False for a contribution vector rebuilt from cumulative
    horizon returns whose segments straddle a part boundary.
    """

    def __init__(self) -> None:
        self.entries: dict[tuple[str, int], tuple[np.ndarray, bool, bool]] = {}

    def add(
        self, root_id: str, index: int, contrib: np.ndarray, terminated: bool, exact: bool
    ) -> None:
        self.entries[(str(root_id), int(index))] = (contrib, bool(terminated), bool(exact))

    def get(self, root_id: str, index: int) -> tuple[np.ndarray, bool, bool] | None:
        return self.entries.get((str(root_id), int(index)))

    def __len__(self) -> int:
        return len(self.entries)


def step_table_from_shards(shards: Iterable[Mapping[str, Any]]) -> StepTable:
    """Contributions from the exporter's ``step_rewards``/``step_years`` candidate extension."""
    table = StepTable()
    for shard in shards:
        discount = float(shard.get("discount_years", DISCOUNT_YEARS))
        for root in shard.get("roots") or []:
            for cand in root.get("candidates") or []:
                if cand.get("plan_role") is not None:
                    continue  # T2 plan rows are never first-action candidates
                steps, years = cand.get("step_rewards"), cand.get("step_years")
                if steps is None or years is None:
                    continue
                table.add(
                    str(root["root_id"]),
                    int(cand["index"]),
                    discounted_contributions(steps, years, discount),
                    bool(cand.get("terminated", False)),
                    True,
                )
    return table


def load_step_table(paths: Sequence[Path | str]) -> StepTable:
    """Raw payload shards (files or directories of ``*.pt``) to a :class:`StepTable`."""
    import torch

    files: list[Path] = []
    for path in paths:
        path = Path(path)
        files.extend(sorted(path.glob("*.pt")) if path.is_dir() else [path])
    return step_table_from_shards(
        torch.load(f, map_location="cpu", weights_only=True) for f in files
    )


def _fallback_contributions(candidate: Any) -> tuple[np.ndarray, bool, bool] | None:
    """Increments of the stream-0 cumulative returns, placed at each horizon's last step."""
    by_h = sorted((h, ev) for (h, s), ev in candidate.evaluations.items() if s == 0)
    if not by_h or by_h[0][0] != 1:
        return None
    contrib = np.zeros((by_h[-1][0], len(COMPONENTS)))
    previous_h, previous = 0, np.zeros(len(COMPONENTS))
    exact = True
    boundary = int(setting("delayed_split_decisions"))
    for h, ev in by_h:
        contrib[h - 1] = ev.returns - previous
        if previous_h < boundary < h:
            exact = False
        previous_h, previous = h, ev.returns
    return contrib, False, exact


def _parts(contrib: np.ndarray, length: int) -> dict[str, np.ndarray]:
    padded = np.zeros((max(length, len(contrib)), len(COMPONENTS)))
    padded[: len(contrib)] = contrib
    split = int(setting("delayed_split_decisions"))
    return {
        "first": padded[0],
        "mid": padded[1:split].sum(axis=0),
        "late": padded[split:length].sum(axis=0),
    }


def _common_length(a: tuple[np.ndarray, bool], b: tuple[np.ndarray, bool]) -> int:
    """Decisions both branches are known for (a terminated branch earns zero afterwards)."""
    limits = [len(c) for c, terminated in (a, b) if not terminated]
    return min(limits) if limits else max(len(a[0]), len(b[0]))


def t1_pairs(
    roots: Sequence[Any],
    prediction: Mapping[str, Sequence[np.ndarray]],
    table: StepTable | None,
    target: str,
) -> dict[str, np.ndarray]:
    """Per-pair dA and the first / mid / late / delayed / total realised difference."""
    scales = _reference_scales()
    out: dict[str, list[float]] = defaultdict(list)
    for r, root in enumerate(roots):
        weights = _weights(target, root.personality if target == "utility:root" else None)
        predicted = sum(weights[n] * np.asarray(prediction[n][r], np.float64) for n in COMPONENTS)
        branches = []
        for cand in root.candidates:
            entry = None if table is None else table.get(root.root_id, cand.index)
            if entry is None:
                entry = _fallback_contributions(cand)
            branches.append(entry)
        for i in range(len(root.candidates)):
            for k in range(i + 1, len(root.candidates)):
                if branches[i] is None or branches[k] is None:
                    out["skipped"].append(1.0)
                    continue
                length = _common_length(branches[i][:2], branches[k][:2])
                if length < 2:
                    out["single_decision"].append(1.0)
                    continue
                pi, pk = _parts(branches[i][0], length), _parts(branches[k][0], length)
                d = {p: _scalar(pi[p] - pk[p], weights, scales) for p in PARTS}
                out["root"].append(r)
                out["world"].append(root.world)
                out["dA"].append(float(predicted[i] - predicted[k]))
                for p in PARTS:
                    out[p].append(d[p])
                out["delayed"].append(d["mid"] + d["late"])
                out["total"].append(d["first"] + d["mid"] + d["late"])
                out["exact"].append(float(branches[i][2] and branches[k][2]))
    keys = ("root", "world", "dA", *PARTS, "delayed", "total", "exact")
    result = {key: np.asarray(out[key], dtype=np.float64) for key in keys}
    result["skipped"] = np.asarray([len(out["skipped"]), len(out["single_decision"])])
    return result


def _loadings(da: np.ndarray, first: np.ndarray, delayed: np.ndarray) -> dict[str, Any]:
    if da.size < 3:
        return {"first": None, "delayed": None, "delayed_to_first_ratio": None}
    design = np.stack([first, delayed], axis=1)
    if np.linalg.matrix_rank(design) < 2:
        return {"first": None, "delayed": None, "delayed_to_first_ratio": None}
    (b_first, b_delayed), *_ = np.linalg.lstsq(design, da, rcond=None)
    ratio = None if abs(b_first) < 1e-12 else float(b_delayed / b_first)
    return {
        "first": float(b_first),
        "delayed": float(b_delayed),
        "delayed_to_first_ratio": ratio,
        "ratio_error": None if ratio is None else abs(1.0 - ratio),
    }


def t1_summary(pairs: Mapping[str, np.ndarray], *, min_abs: float | None = None) -> dict[str, Any]:
    """Correlations, loadings and the sign-conflict ranking of one target's pairs."""
    m = float(setting("conflict_min_abs") if min_abs is None else min_abs)
    da = pairs["dA"]
    first, delayed, total = pairs["first"], pairs["delayed"], pairs["total"]
    conflict = (np.abs(first) > m) & (np.abs(delayed) > m) & (np.sign(first) != np.sign(delayed))
    dominated = conflict & (np.abs(delayed) > np.abs(first))
    follows_delayed = [_agree(a, t) for a, t in zip(da[conflict], delayed[conflict], strict=True)]
    follows_first = [_agree(a, t) for a, t in zip(da[conflict], first[conflict], strict=True)]
    dominated_total = [_agree(a, t) for a, t in zip(da[dominated], total[dominated], strict=True)]
    informative = np.abs(total) > m
    ranking_total = [_agree(a, t) for a, t in zip(da[informative], total[informative], strict=True)]
    abs_first, abs_delayed = float(np.abs(first).sum()), float(np.abs(delayed).sum())
    return {
        "pairs": int(da.size),
        "roots": len(np.unique(pairs["root"])) if da.size else 0,
        "worlds": len(np.unique(pairs["world"])) if da.size else 0,
        "skipped_pairs_without_steps": int(pairs["skipped"][0]),
        "single_decision_pairs_excluded": int(pairs["skipped"][1]),
        "inexact_part_boundaries": int((pairs["exact"] == 0).sum()) if da.size else 0,
        "delayed_share_of_abs_difference": None
        if abs_first + abs_delayed <= 0
        else abs_delayed / (abs_first + abs_delayed),
        "correlation": {
            "first": pearson(da, first),
            "mid": pearson(da, pairs["mid"]),
            "late": pearson(da, pairs["late"]),
            "delayed": pearson(da, delayed),
            "total": pearson(da, total),
        },
        "loadings": _loadings(da, first, delayed),
        "ranking_total": {"accuracy": _mean(ranking_total), "pairs": len(ranking_total)},
        "sign_conflict": {
            "pairs": int(conflict.sum()),
            "follows_delayed": _mean(follows_delayed),
            "follows_first": _mean(follows_first),
            "delayed_dominated_pairs": int(dominated.sum()),
            "delayed_dominated_follows_total": _mean(dominated_total),
            "myopic": None
            if not follows_first
            else bool(np.mean(follows_first) > np.mean(follows_delayed)),
        },
        "min_abs": m,
    }


def t1_advantage_report(
    roots: Sequence[Any],
    prediction: Mapping[str, Sequence[np.ndarray]],
    table: StepTable | None,
    *,
    targets: Sequence[str] = T1_TARGETS,
) -> dict[str, Any]:
    return {
        "parts": PART_DEFINITION,
        "units": "reference scales; selection stream 0 (deterministic H12 continuation)",
        "step_source": "step_rewards/step_years (exact) with cumulative-horizon fallback",
        "targets": {t: t1_summary(t1_pairs(roots, prediction, table, t)) for t in targets},
    }


# ---------------------------------------------------------------------------
# T1: V's TD errors along trajectories
# ---------------------------------------------------------------------------


def _trajectory_groups(
    worlds: Sequence[int], camps: Sequence[int], trajectories: Sequence[int], turns: Sequence[int]
) -> list[list[int]]:
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, key in enumerate(zip(worlds, camps, trajectories, strict=True)):
        groups[(int(key[0]), int(key[1]), int(key[2]))].append(i)
    out = []
    for key in sorted(groups):
        indices = groups[key]
        indices.sort(key=lambda i: int(turns[i]))
        out.append(indices)
    return out


def td_error_split(
    values: Sequence[float],
    reward: Sequence[float],
    elapsed: Sequence[float],
    terminal: Sequence[bool],
    turns: Sequence[int],
    worlds: Sequence[int],
    camps: Sequence[int],
    trajectories: Sequence[int],
    codes: Sequence[int] | None = None,
    tails: Sequence[float] | None = None,
    *,
    zero_tail_declared: bool = False,
    discount_years: float = DISCOUNT_YEARS,
) -> dict[str, np.ndarray]:
    """Split each row's Monte Carlo error ``G_t - V_t`` into first / mid / late TD parts.

    ``delta_t = r_t + g_t V_{t+1} - V_t`` (``V_{t+1}`` = 0 after a terminal row, the
    declared tail after a living end); ``G_t - V_t = sum_j (prod_{m<j} g_{t+m})
    delta_{t+j}``. ``first`` is ``delta_t``, ``mid`` the discounted errors of
    decisions 2-5 and ``late`` the rest. A part reaching a living end without a
    declared tail is unavailable (never closed with zero) unless the dataset declares
    the truncated-zero method. Positive means V under-predicts.
    """
    from .labor_bc import TAIL_RANDOMIZED, TAIL_TERMINAL_EXACT

    n = len(values)
    split = int(setting("delayed_split_decisions"))
    delta = np.zeros(n)
    gamma = np.zeros(n)
    end_ok = np.ones(n, dtype=bool)
    parts = {p: np.zeros(n) for p in PARTS}
    available = {p: np.ones(n, dtype=bool) for p in PARTS}
    position = np.zeros(n, dtype=np.int64)
    for indices in _trajectory_groups(worlds, camps, trajectories, turns):
        size = len(indices)
        stop = size  # rows after a terminal row are not part of the trajectory
        for pos, i in enumerate(indices):
            if terminal[i]:
                stop = pos + 1
                break
        ok_end = True
        for pos, i in enumerate(indices[:stop]):
            position[i] = pos
            g = math.exp(-float(elapsed[i]) / discount_years)
            gamma[i] = g
            if terminal[i]:
                nxt = 0.0
            elif pos + 1 < stop:
                nxt = float(values[indices[pos + 1]])
            else:
                exact = codes is not None and codes[i] in (TAIL_TERMINAL_EXACT, TAIL_RANDOMIZED)
                if exact:
                    nxt = float(tails[i])
                elif zero_tail_declared:
                    nxt = 0.0
                else:
                    nxt, ok_end = 0.0, False
            delta[i] = float(reward[i]) + g * nxt - float(values[i])
        for pos, i in enumerate(indices[:stop]):
            discount = 1.0
            for j, row in enumerate(indices[pos:stop]):
                part = "first" if j == 0 else "mid" if j < split else "late"
                parts[part][i] += discount * delta[row]
                discount *= gamma[row]
            last = stop - 1 - pos  # offset of the trajectory's last row from this row
            if not ok_end:
                available["late"][i] = False
                if last < split:
                    available["mid"][i] = False
                if last == 0:
                    available["first"][i] = False
            end_ok[i] = ok_end
        for i in indices[stop:]:
            for p in PARTS:
                available[p][i] = False
    return {"parts": parts, "available": available, "position": position, "delta": delta}


def _part_stats(
    values: np.ndarray, available: np.ndarray, worlds: np.ndarray, keep: np.ndarray, scale: float
) -> dict[str, Any]:
    rows = np.flatnonzero(keep & available)
    if not len(rows):
        return {"rows": 0, "bias": None, "abs_bias": None, "mae": None}
    weights = world_balanced(worlds[rows])
    normalized = values[rows] / scale
    bias = _wmean(normalized, weights)
    return {
        "rows": len(rows),
        "bias": bias,
        "abs_bias": None if bias is None else abs(bias),
        "mae": _wmean(np.abs(normalized), weights),
    }


def _dataset_arrays(dataset: Mapping[str, Any]) -> dict[str, Any]:
    from .labor_bc import (
        RETURN_TAIL_RANDOMIZED,
        TAIL_METHOD_FIELD,
        TRAJECTORY_FIELD,
        critic_target_available,
        elapsed_years,
        return_tail_method,
    )

    def arr(key: str) -> np.ndarray:
        value = dataset[key]
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    n = int(dataset["row_world"].shape[0])
    randomized = return_tail_method(dataset) == RETURN_TAIL_RANDOMIZED
    quarantined = (
        arr("row_quarantined").reshape(-1).astype(bool)
        if "row_quarantined" in dataset
        else np.zeros(n, dtype=bool)
    )
    return {
        "n": n,
        "worlds": arr("row_world").reshape(-1),
        "camps": arr("row_camp").reshape(-1),
        "turns": arr("row_turn").reshape(-1),
        "trajectories": arr(TRAJECTORY_FIELD).reshape(-1)
        if TRAJECTORY_FIELD in dataset
        else np.full(n, -1),
        "sources": arr("row_source").reshape(-1) if "row_source" in dataset else np.zeros(n),
        "rounds": arr("row_round").reshape(-1) if "row_round" in dataset else np.zeros(n),
        "elapsed": elapsed_years(dataset).reshape(-1).double().numpy(),
        "terminal": arr("transition_terminal").reshape(-1).astype(bool),
        "randomized": randomized,
        "codes": arr(TAIL_METHOD_FIELD).reshape(-1) if randomized else None,
        "keep": ~quarantined & critic_target_available(dataset).numpy().astype(bool),
        "arr": arr,
    }


def t1_value_report(
    predictions: Mapping[str, np.ndarray], dataset: Mapping[str, Any]
) -> dict[str, Any]:
    """The T1 split of V's TD errors per head (``predictions``: decoded raw values per row)."""
    from .labor_bc import TAIL_VALUE_PREFIX

    a = _dataset_arrays(dataset)
    scales = _reference_scales()
    index = np.zeros(a["n"], dtype=np.int64)
    early = np.zeros(a["n"], dtype=bool)
    heads: dict[str, Any] = {}
    for name in COMPONENTS:
        reward_key = "reward_" + name
        if name not in predictions or reward_key not in dataset:
            continue
        tails = a["arr"](TAIL_VALUE_PREFIX + name).reshape(-1) if a["randomized"] else None
        split = td_error_split(
            np.asarray(predictions[name], np.float64),
            a["arr"](reward_key).reshape(-1).astype(np.float64),
            a["elapsed"],
            a["terminal"],
            a["turns"],
            a["worlds"],
            a["camps"],
            a["trajectories"],
            a["codes"],
            tails,
            zero_tail_declared=not a["randomized"],
        )
        index = split["position"]
        lengths = defaultdict(int)
        for indices in _trajectory_groups(a["worlds"], a["camps"], a["trajectories"], a["turns"]):
            for i in indices:
                lengths[i] = len(indices)
        early = np.array([index[i] < (lengths[i] + 1) // 2 for i in range(a["n"])], dtype=bool)
        scale = scales[name]
        report: dict[str, Any] = {}
        for part in PARTS:
            report[part] = _part_stats(
                split["parts"][part], split["available"][part], a["worlds"], a["keep"], scale
            )
            report[part]["early"] = _part_stats(
                split["parts"][part],
                split["available"][part],
                a["worlds"],
                a["keep"] & early,
                scale,
            )
            report[part]["late_turns"] = _part_stats(
                split["parts"][part],
                split["available"][part],
                a["worlds"],
                a["keep"] & ~early,
                scale,
            )
        total_available = (
            split["available"]["late"] & split["available"]["mid"] & split["available"]["first"]
        )
        total = split["parts"]["first"] + split["parts"]["mid"] + split["parts"]["late"]
        report["total"] = _part_stats(total, total_available, a["worlds"], a["keep"], scale)
        b_first = report["first"]["bias"]
        delayed = split["parts"]["mid"] + split["parts"]["late"]
        report["delayed"] = _part_stats(delayed, total_available, a["worlds"], a["keep"], scale)
        b_delayed = report["delayed"]["bias"]
        report["delayed_share_of_abs_bias"] = (
            None
            if b_first is None or b_delayed is None or abs(b_first) + abs(b_delayed) <= 0
            else abs(b_delayed) / (abs(b_first) + abs(b_delayed))
        )
        heads[name] = report
    return {
        "parts": PART_DEFINITION,
        "sign": "target minus V (positive: V under-predicts)",
        "units": "reference scales, world-balanced",
        "trajectory_keys": "world, camp, row_trajectory (as labor_bc.discounted_returns)",
        "heads": heads,
    }


# ---------------------------------------------------------------------------
# T3: adaptation lag
# ---------------------------------------------------------------------------


def _personality_rows(dataset: Mapping[str, Any], n: int) -> np.ndarray:
    value = dataset.get("labor_personality")
    if value is None:
        return np.full((n, 2), 0.5)
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return array.reshape(n, -1)[:, :2].astype(np.float64)


def _critic_returns(dataset: Mapping[str, Any], sources: np.ndarray) -> dict[str, np.ndarray]:
    """Discounted returns of the teacher/continuation rows only (DAgger rows never chain in)."""
    from .labor_bc import discounted_returns

    n = len(sources)
    rows = np.flatnonzero(sources != DAGGER_ROW_SOURCE)
    out = {name: np.zeros(n) for name in COMPONENTS}
    if not len(rows):
        return out
    if len(rows) == n:
        sub = dataset
    else:
        import torch

        index = torch.as_tensor(rows, dtype=torch.long)
        sub = {
            key: value[index]
            if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.shape[0]) == n
            else value
            for key, value in dataset.items()
        }
    for name, value in discounted_returns(sub).items():
        out[name][rows] = value.double().numpy()
    return out


def branch_beat_keys(
    datasets: Sequence[Mapping[str, Any]],
    *,
    target: str = "utility",
) -> dict[tuple[int, int, int, int, int], float]:
    """Actor-visited continuation states whose actor branch beat the teacher from its start.

    Joins teacher rows (``row_source`` 0, ``row_trajectory`` -1), DAgger branch rows
    (source 1, rewards of the actor's decisions) and teacher continuations (source 3)
    over (world, camp, round, start_index, branch_step). For a continuation after
    ``b`` actor decisions from start turn ``s``: ``G_actor = sum_{j<b} D_j r_j + D_b
    G_cont`` against the teacher's own return ``G_teacher(s)``. Returns the key ->
    utility difference (reference scales) for every joined branch; the beating ones
    are those above ``beat_margin`` (applied by :func:`v_adaptation_report`).
    """
    from .labor_bc import CONTINUATION_ROW_SOURCE

    scales = _reference_scales()
    teacher: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    branch: dict[tuple[int, int, int, int, int], tuple[np.ndarray, float, int]] = {}
    continuation: dict[tuple[int, int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for dataset in datasets:
        a = _dataset_arrays(dataset)
        n = a["n"]
        personality = _personality_rows(dataset, n)
        arr = a["arr"]
        sources = a["sources"]
        has_returns = bool(((sources == 0) | (sources == CONTINUATION_ROW_SOURCE)).any())
        returns = None
        if has_returns:
            try:
                returns = _critic_returns(dataset, sources)
            except (KeyError, ValueError):
                returns = None
        start_index = arr("row_start_index").reshape(-1) if "row_start_index" in dataset else None
        branch_step = arr("row_branch_step").reshape(-1) if "row_branch_step" in dataset else None
        for i in range(n):
            if not a["keep"][i] and sources[i] != 1:
                continue
            world, camp, turn = int(a["worlds"][i]), int(a["camps"][i]), int(a["turns"][i])
            if sources[i] == 0 and int(a["trajectories"][i]) == -1 and returns is not None:
                g = np.array([float(returns[name][i]) for name in COMPONENTS])
                teacher[(world, camp, turn)] = (g, personality[i])
            elif sources[i] == 1 and start_index is not None and branch_step is not None:
                r = np.array([float(arr("reward_" + name).reshape(-1)[i]) for name in COMPONENTS])
                branch[
                    (world, camp, int(a["rounds"][i]), int(start_index[i]), int(branch_step[i]))
                ] = (
                    r,
                    float(a["elapsed"][i]),
                    turn,
                )
            elif (
                sources[i] == CONTINUATION_ROW_SOURCE
                and returns is not None
                and start_index is not None
                and branch_step is not None
            ):
                g = np.array([float(returns[name][i]) for name in COMPONENTS])
                key = (world, camp, int(a["rounds"][i]), int(start_index[i]), int(branch_step[i]))
                continuation[key] = (g, personality[i])
    out: dict[tuple[int, int, int, int, int], float] = {}
    for key, (g_cont, pers) in continuation.items():
        world, camp, rnd, start, steps = key
        if steps < 1:
            continue
        g_actor = np.zeros(len(COMPONENTS))
        discount = 1.0
        start_turn = None
        complete = True
        for j in range(steps):
            entry = branch.get((world, camp, rnd, start, j))
            if entry is None:
                complete = False
                break
            reward, elapsed, turn = entry
            if j == 0:
                start_turn = turn
            g_actor += discount * reward
            discount *= math.exp(-elapsed / DISCOUNT_YEARS)
        if not complete or start_turn is None or (world, camp, start_turn) not in teacher:
            continue
        g_actor += discount * g_cont
        g_teacher, _ = teacher[(world, camp, start_turn)]
        weights = _weights(target if target in COMPONENTS else "utility", pers)
        out[key] = _scalar(g_actor - g_teacher, weights, scales)
    return out


def _calibration(
    pred: np.ndarray, target: np.ndarray, worlds: np.ndarray, scale: float
) -> dict[str, Any]:
    if not len(pred):
        return {"rows": 0, "worlds": 0, "nmae": None, "bias": None, "abs_bias": None}
    weights = world_balanced(worlds)
    bias = _wmean((pred - target) / scale, weights)
    return {
        "rows": len(pred),
        "worlds": len(np.unique(worlds)),
        "nmae": _wmean(np.abs(pred - target) / scale, weights),
        "bias": bias,
        "abs_bias": None if bias is None else abs(bias),
    }


def v_adaptation_report(
    sets: Sequence[tuple[str, Mapping[str, Any], Mapping[str, np.ndarray]]],
    *,
    beat: Mapping[tuple[int, int, int, int, int], float] | None = None,
    margin: float | None = None,
) -> dict[str, Any]:
    """V calibration on teacher, actor-visited and beat-the-teacher rows, per DAgger round.

    ``sets`` is (label, dataset, decoded predictions per head for every row). Teacher
    rows are ``row_source`` 0, actor-visited rows are teacher continuations from
    learner states (``row_source`` 3); DAgger rows (source 1) carry no V target. The
    beat subset is the actor-visited rows whose branch key is in ``beat`` above
    ``margin``.
    """
    from .labor_bc import CONTINUATION_ROW_SOURCE

    m = float(setting("beat_margin") if margin is None else margin)
    beat = beat or {}
    scales = _reference_scales()
    pooled: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    labels = []
    for label, dataset, predictions in sets:
        a = _dataset_arrays(dataset)
        sources = a["sources"]
        if not bool(((sources == 0) | (sources == CONTINUATION_ROW_SOURCE)).any()):
            continue
        labels.append(label)
        returns = _critic_returns(dataset, sources)
        arr = a["arr"]
        start = arr("row_start_index").reshape(-1) if "row_start_index" in dataset else None
        step = arr("row_branch_step").reshape(-1) if "row_branch_step" in dataset else None
        is_beat = np.zeros(a["n"], dtype=bool)
        if start is not None and step is not None:
            for i in np.flatnonzero(sources == CONTINUATION_ROW_SOURCE):
                key = (
                    int(a["worlds"][i]),
                    int(a["camps"][i]),
                    int(a["rounds"][i]),
                    int(start[i]),
                    int(step[i]),
                )
                is_beat[i] = beat.get(key, -math.inf) > m
        masks = {
            "teacher": a["keep"] & (sources == 0),
            "actor": a["keep"] & (sources == CONTINUATION_ROW_SOURCE),
            "beat": a["keep"] & is_beat,
        }
        for group in ("actor", "beat"):
            for rnd in np.unique(a["rounds"][masks[group]]):
                masks[f"{group}:round-{int(rnd)}"] = masks[group] & (a["rounds"] == rnd)
        for group, mask in masks.items():
            rows = np.flatnonzero(mask)
            for name in COMPONENTS:
                if name not in predictions:
                    continue
                pooled[group][name + "/pred"].append(np.asarray(predictions[name])[rows])
                pooled[group][name + "/target"].append(returns[name][rows])
            pooled[group]["worlds"].append(a["worlds"][rows])
    heads: dict[str, Any] = {}
    for name in COMPONENTS:
        groups = {}
        for group, parts in pooled.items():
            if name + "/pred" not in parts:
                continue
            groups[group] = _calibration(
                np.concatenate(parts[name + "/pred"]),
                np.concatenate(parts[name + "/target"]),
                np.concatenate(parts["worlds"]),
                scales[name],
            )
        if not groups:
            continue

        def gap(group: str, key: str, groups: Mapping[str, Any] = groups) -> float | None:
            g, t = groups.get(group, {}).get(key), groups.get("teacher", {}).get(key)
            return None if g is None or t is None else g - t

        heads[name] = {
            "groups": dict(sorted(groups.items())),
            "actor_minus_teacher_nmae": gap("actor", "nmae"),
            "beat_minus_teacher_nmae": gap("beat", "nmae"),
            "beat_minus_teacher_abs_bias": gap("beat", "abs_bias"),
        }
    return {
        "datasets": labels,
        "groups": {
            "teacher": "row_source 0: teacher (H12) trajectory rows",
            "actor": "row_source 3: teacher continuations from DAgger learner-visited states",
            "beat": "actor-visited rows whose actor branch beat the teacher's own return "
            "from the same start by more than beat_margin (utility)",
        },
        "beat_branches_joined": len(beat),
        "beat_branches_above_margin": int(sum(1 for v in beat.values() if v > m)),
        "beat_margin": m,
        "units": "reference scales, world-balanced; bias = V - G",
        "heads": heads,
    }


def _subset(prediction: Mapping[str, Sequence[np.ndarray]], index: Sequence[int]) -> dict:
    return {name: [prediction[name][i] for i in index] for name in prediction}


def search_beat_roots(roots: Sequence[Any], *, margin: float | None = None) -> dict[int, int]:
    """Roots where an admitted non-H12 candidate beats H12 on root utility: root -> best."""
    from .labor_advantage import build_pairs

    m = float(setting("beat_margin") if margin is None else margin)
    pairs = build_pairs(roots, "utility", personality="root")
    best: dict[int, tuple[int, float]] = {}
    for idx in np.flatnonzero(pairs.admitted):
        r = int(pairs.root[idx])
        h12 = roots[r].h12
        i, k, delta = int(pairs.i[idx]), int(pairs.k[idx]), float(pairs.delta[idx])
        if h12 not in (i, k):
            continue
        other, gain = (k, -delta) if i == h12 else (i, delta)
        if gain > m and gain > best.get(r, (-1, -math.inf))[1]:
            best[r] = (other, gain)
    return {r: c for r, (c, _) in best.items()}


def a_adaptation_report(
    roots: Sequence[Any],
    prediction: Mapping[str, Sequence[np.ndarray]],
    *,
    margin: float | None = None,
) -> dict[str, Any]:
    """A ranking on teacher-source, actor-source and search-beat-the-teacher roots."""
    from .labor_advantage_eval import pairwise_ranking

    beat = search_beat_roots(roots, margin=margin)
    groups = {
        "teacher": [r for r, root in enumerate(roots) if root.source == "h12"],
        "actor": [r for r, root in enumerate(roots) if root.source == "actor"],
        "beat": sorted(beat),
    }
    out: dict[str, Any] = {}
    for group, index in groups.items():
        sub = [roots[i] for i in index]
        pred = _subset(prediction, index)
        entry: dict[str, Any] = {"roots": len(sub)}
        if sub:
            for target, pers in (("utility", "root"), ("qol_years", None)):
                key = "utility:root" if target == "utility" else target
                result = pairwise_ranking(sub, pred, target, personality=pers)
                entry[key] = {"accuracy": result["accuracy"], "pairs": result["pairs"]}
        if group == "beat" and sub:
            hits = []
            for r in index:
                root = roots[r]
                weights = _weights("utility:root", root.personality)
                a = sum(weights[n] * np.asarray(prediction[n][r]) for n in COMPONENTS)
                hits.append(_agree(float(a[beat[r]] - a[root.h12]), 1.0))
            entry["beat_candidate_above_h12"] = _mean(hits)
        out[group] = entry
    return {
        "groups": out,
        "beat_definition": "an admitted pair (|dG| > 2 SE) where a non-H12 candidate beats H12 "
        "on root utility by more than beat_margin",
    }


# ---------------------------------------------------------------------------
# T4: value-tracking curves along autonomous rollouts
# ---------------------------------------------------------------------------


def traces_from_reports(reports: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Critic traces from autonomous reports (``runs[*].critic_trace``) or bare traces."""
    out = []
    for report in reports:
        if report.get("schema") == TRACE_SCHEMA:
            out.append(dict(report))
            continue
        for run in report.get("runs") or []:
            trace = run.get("critic_trace")
            if trace is not None:
                out.append(dict(trace, world=run.get("world"), mode=run.get("mode")))
    return out


def _trace_rows(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per (camp, decision) rows with realised discounted returns, closed and truncated."""
    by_camp: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for d, decision in enumerate(trace.get("decisions") or []):
        index = int(decision.get("decision", d))
        for camp in decision.get("camps") or []:
            by_camp[int(camp["camp"])].append(dict(camp, decision=index, turn=decision.get("turn")))
    final = {int(c["camp"]): c for c in (trace.get("final") or {}).get("camps") or []}
    rows = []
    for camp, entries in by_camp.items():
        entries.sort(key=lambda e: e["decision"])
        stop = len(entries)
        for pos, entry in enumerate(entries):
            if entry.get("terminal"):
                stop = pos + 1
                break
        entries = entries[:stop]
        ended = bool(entries and entries[-1].get("terminal"))
        tail = final.get(camp)
        g_closed = np.array(
            [0.0 if ended or tail is None else float(tail["values"][n]) for n in COMPONENTS]
        )
        g_trunc = np.zeros(len(COMPONENTS))
        remaining = 1.0
        closed_ok = ended or tail is not None
        for entry in reversed(entries):
            gamma = math.exp(-float(entry["elapsed"]) / DISCOUNT_YEARS)
            reward = np.array([float(entry["rewards"][n]) for n in COMPONENTS])
            g_closed = reward + gamma * g_closed
            g_trunc = reward + gamma * g_trunc
            remaining = 0.0 if ended else remaining * gamma
            rows.append(
                {
                    "camp": camp,
                    "decision": int(entry["decision"]),
                    "society": entry.get("society"),
                    "parent_society": entry.get("parent_society"),
                    "farmer_unlocked": entry.get("farmer_unlocked"),
                    "pred": np.array([float(entry["values"][n]) for n in COMPONENTS]),
                    "closed": g_closed.copy() if closed_ok else None,
                    "truncated": g_trunc.copy(),
                    "truncation_weight": remaining,
                }
            )
    return rows


def _stage_tags(trace: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[set[str]]:
    window = int(setting("stage_window"))
    decisions = [r["decision"] for r in rows]
    last = max(decisions) if decisions else 0
    first_seen: dict[int, int] = {}
    unlock: dict[int, int] = {}
    previous: dict[int, Any] = {}
    for r in sorted(rows, key=lambda r: (r["camp"], r["decision"])):
        camp = r["camp"]
        first_seen.setdefault(camp, r["decision"])
        if previous.get(camp) is False and r["farmer_unlocked"] is True and camp not in unlock:
            unlock[camp] = r["decision"]
        previous[camp] = r["farmer_unlocked"]
    fission_events: list[tuple[int, int, int]] = []  # (decision, daughter, parent society)
    for r in rows:
        parent = r.get("parent_society")
        if (
            parent is not None
            and int(parent) > 0
            and first_seen[r["camp"]] > 0
            and r["decision"] == first_seen[r["camp"]]
        ):
            fission_events.append((r["decision"], r["camp"], int(parent)))
    tags = []
    for r in rows:
        t: set[str] = set()
        if r["decision"] < (last + 1) / 3:
            t.add("early")
        elif r["decision"] >= 2 * (last + 1) / 3:
            t.add("late")
        else:
            t.add("middle")
        if r["camp"] in unlock and abs(r["decision"] - unlock[r["camp"]]) <= window:
            t.add("unlock")
        for when, daughter, parent in fission_events:
            if abs(r["decision"] - when) <= window and (
                r["camp"] == daughter or r.get("society") == parent
            ):
                t.add("fission")
        tags.append(t)
    return tags


def separation_decision(
    decisions: Sequence[int],
    gaps: Sequence[float | None],
    *,
    threshold: float | None = None,
    persist: int | None = None,
) -> int | None:
    """The first decision from which |gap| exceeds ``threshold`` for ``persist`` decisions."""
    t = float(setting("separation_threshold") if threshold is None else threshold)
    p = int(setting("separation_persist") if persist is None else persist)
    run = 0
    for pos, gap in enumerate(gaps):
        if gap is not None and abs(gap) > t:
            run += 1
            if run >= p:
                return int(decisions[pos - p + 1])
        else:
            run = 0
    return None


def t4_report(
    traces: Sequence[Mapping[str, Any]],
    *,
    threshold: float | None = None,
    persist: int | None = None,
) -> dict[str, Any]:
    """Decision-by-decision V against realised return, per-stage summaries and separation.

    ``closed`` realised returns bootstrap a living end with the critic's own value of
    the final state (recorded after the last decision); ``truncated`` returns stop at
    the rollout end and carry the remaining discount weight. Curves average camps
    within a world, then worlds (world-balanced). Gap = (V - G) / reference scale.
    """
    scales = _reference_scales()
    per_world_rows: dict[Any, list[tuple[dict[str, Any], set[str]]]] = defaultdict(list)
    critic_digests = sorted({str(t.get("critic_digest")) for t in traces})
    for trace in traces:
        rows = _trace_rows(trace)
        for row, tags in zip(rows, _stage_tags(trace, rows), strict=True):
            per_world_rows[(trace.get("world"), trace.get("mode"))].append((row, tags))
    heads: dict[str, Any] = {}
    for c, name in enumerate(COMPONENTS):
        scale = scales[name]
        curve: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        stage_values: dict[str, list[tuple[Any, float]]] = defaultdict(list)
        per_world_sep = {}
        for world, rows in per_world_rows.items():
            by_decision: dict[int, list[tuple[float, float, float | None]]] = defaultdict(list)
            for row, tags in rows:
                pred = row["pred"][c] / scale
                closed = None if row["closed"] is None else row["closed"][c] / scale
                trunc = row["truncated"][c] / scale
                by_decision[row["decision"]].append((pred, trunc, closed))
                if closed is not None:
                    for tag in (*tags, "all"):
                        stage_values[tag].append((world, pred - closed))
            world_decisions = sorted(by_decision)
            world_gaps = []
            for d in world_decisions:
                values = by_decision[d]
                curve[d]["pred"].append(float(np.mean([v[0] for v in values])))
                curve[d]["truncated"].append(float(np.mean([v[1] for v in values])))
                closed = [v[2] for v in values if v[2] is not None]
                if closed:
                    curve[d]["closed"].append(float(np.mean(closed)))
                    world_gaps.append(
                        float(np.mean([v[0] for v in values if v[2] is not None]))
                        - float(np.mean(closed))
                    )
                else:
                    world_gaps.append(None)
            per_world_sep[f"{world[0]}/{world[1]}"] = separation_decision(
                world_decisions, world_gaps, threshold=threshold, persist=persist
            )
        decisions = sorted(curve)
        points = []
        for d in decisions:
            entry = curve[d]
            pred = float(np.mean(entry["pred"]))
            closed = float(np.mean(entry["closed"])) if entry["closed"] else None
            points.append(
                {
                    "decision": d,
                    "worlds": len(entry["pred"]),
                    "pred": pred,
                    "realised_closed": closed,
                    "realised_truncated": float(np.mean(entry["truncated"])),
                    "gap": None if closed is None else pred - closed,
                }
            )
        stages = {}
        for tag, values in sorted(stage_values.items()):
            worlds = [w for w, _ in values]
            gaps = np.array([g for _, g in values])
            index = {w: i for i, w in enumerate(sorted(set(worlds), key=str))}
            weights = world_balanced([index[w] for w in worlds])
            bias = _wmean(gaps, weights)
            stages[tag] = {
                "rows": len(gaps),
                "bias": bias,
                "abs_bias": None if bias is None else abs(bias),
                "mae": _wmean(np.abs(gaps), weights),
            }
        heads[name] = {
            "curve": points,
            "separation_decision": separation_decision(
                [p["decision"] for p in points],
                [p["gap"] for p in points],
                threshold=threshold,
                persist=persist,
            ),
            "per_world_separation": per_world_sep,
            "stages": stages,
            "mae": stages.get("all", {}).get("mae"),
            "abs_bias": stages.get("all", {}).get("abs_bias"),
            "decisions": len(points),
        }
    return {
        "traces": len(traces),
        "worlds": sorted({str(k[0]) for k in per_world_rows}),
        "critic_digests": critic_digests,
        "units": "reference scales; gap = V - G (closed by V of the final state)",
        "separation": {
            "threshold": float(setting("separation_threshold") if threshold is None else threshold),
            "persist": int(setting("separation_persist") if persist is None else persist),
        },
        "stages": {
            "early/middle/late": "rollout decision thirds",
            "unlock": f"within {setting('stage_window')} decisions of the camp's Farmer unlock",
            "fission": f"daughter and parent-society camps within {setting('stage_window')} "
            "decisions of a daughter's first decision",
        },
        "heads": heads,
    }


# ---------------------------------------------------------------------------
# T4 recorder support (labor_pretrain_eval autonomous --record-critic-trace)
# ---------------------------------------------------------------------------


def critic_values(
    task: Any, costs: Any, tensors: Mapping[str, Any], privileged: Mapping[str, Any] | None
) -> dict[str, np.ndarray]:
    """Decoded raw critic values for live rows (privileged arrays passed separately)."""
    from .labor_pretrain_eval import critic_predictions

    rows = int(tensors["core"].shape[0])
    if task.privileged_layout is not None and privileged is None:
        raise ValueError("the privileged critic trace needs the bridge's privileged view")
    data = dict(tensors)
    if privileged is not None:
        data.update(privileged)
    preds = critic_predictions(task, costs, data, np.arange(rows), batch_size=max(1, rows))
    return {name: preds[name] for name in COMPONENTS}


class CriticTraceRecorder:
    """Builds a :data:`TRACE_SCHEMA` trace decision by decision (off unless requested)."""

    def __init__(self, task: Any, costs: Any, *, critic_digest: str | None = None) -> None:
        self.task, self.costs = task, costs
        self.decisions: list[dict[str, Any]] = []
        self.final: dict[str, Any] | None = None
        self.critic_digest = critic_digest
        self.founded: dict[int, int] = {}

    def _privileged(self, live: Sequence[Mapping[str, Any]]) -> Any:
        if (
            self.task.privileged_layout is None
            and getattr(self.costs, "privileged_layout", None) is None
        ):
            return None
        from .labor_privileged import privileged_row_tensors

        return privileged_row_tensors(live)

    def predict(self, live: Sequence[Mapping[str, Any]], tensors: Mapping[str, Any]) -> dict:
        return critic_values(self.task, self.costs, tensors, self._privileged(live))

    def record(
        self,
        *,
        step: Mapping[str, Any],
        live: Sequence[Mapping[str, Any]],
        tensors: Mapping[str, Any],
        values: Mapping[str, np.ndarray],
        execution: Any,
    ) -> None:
        from .labor_bridge import SPECIALTIES
        from .labor_training import _settlement_guard, _transition_rewards

        turn = int(step["turn"])
        for village in live:
            self.founded.setdefault(int(village["camp"]), turn)
        next_step = execution.step
        camps = tuple(int(v["camp"]) for v in live)
        rewards, terminal, elapsed = _transition_rewards(
            next_step, camps, self.founded, world_index=0, decision_turn=turn
        )
        outcomes = {int(v["camp"]): v for v in next_step["villages"]}
        farmer = SPECIALTIES.index("Farmer")
        mask = tensors.get("labor_specialty_mask")
        entries = []
        for row, village in enumerate(live):
            camp = int(village["camp"])
            attempt = execution.initial_attempts.get(camp)
            entries.append(
                {
                    "camp": camp,
                    "society": int(village.get("society", -1)),
                    "parent_society": int(village.get("parentSociety", 0)),
                    "population": int(village.get("population", 0)),
                    "farmer_unlocked": None if mask is None else bool(mask[row, farmer]),
                    "values": {n: float(values[n][row]) for n in COMPONENTS},
                    "rewards": {
                        "qol_years": float(rewards[camp]["qol_years"]),
                        "population_years": float(rewards[camp]["population_years"]),
                        "feasibility": 0.0 if attempt is None else float(attempt.feasibility_cost),
                        "settlement_guard": float(
                            _settlement_guard(
                                outcomes[camp], self.founded.get(camp, turn), int(next_step["turn"])
                            )
                        ),
                    },
                    "elapsed": float(elapsed[camp]),
                    "terminal": bool(terminal[camp]),
                }
            )
        self.decisions.append({"decision": len(self.decisions), "turn": turn, "camps": entries})

    def finish(
        self, step: Mapping[str, Any], live: Sequence[Mapping[str, Any]], tensors: Any
    ) -> None:
        if not live:
            self.final = {"turn": int(step["turn"]), "camps": []}
            return
        values = self.predict(live, tensors)
        self.final = {
            "turn": int(step["turn"]),
            "camps": [
                {"camp": int(v["camp"]), "values": {n: float(values[n][r]) for n in COMPONENTS}}
                for r, v in enumerate(live)
            ],
        }

    def trace(self) -> dict[str, Any]:
        return {
            "schema": TRACE_SCHEMA,
            "heads": list(COMPONENTS),
            "critic_digest": self.critic_digest,
            "discount_years": DISCOUNT_YEARS,
            "units": "raw decoded critic values and raw rewards",
            "decisions": self.decisions,
            "final": self.final,
        }


# ---------------------------------------------------------------------------
# Receipt section, headline metrics and checkpoint comparison
# ---------------------------------------------------------------------------


def headline_metrics(diagnostics: Mapping[str, Any] | None) -> dict[str, tuple[float, int]]:
    """Scalar T-metrics with their direction (+1 higher is better, -1 lower is better)."""
    out: dict[str, tuple[float, int]] = {}
    if not diagnostics:
        return out

    def put(key: str, value: Any, direction: int) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            out[key] = (float(value), direction)

    t1 = diagnostics.get("t1") or {}
    for target, s in ((t1.get("advantage") or {}).get("targets") or {}).items():
        base = f"t1.a.{target}"
        put(f"{base}.corr_delayed", s["correlation"].get("delayed"), +1)
        put(f"{base}.corr_total", s["correlation"].get("total"), +1)
        put(f"{base}.loading_ratio_error", s["loadings"].get("ratio_error"), -1)
        put(f"{base}.conflict_follows_delayed", s["sign_conflict"].get("follows_delayed"), +1)
        put(
            f"{base}.conflict_dominated_follows_total",
            s["sign_conflict"].get("delayed_dominated_follows_total"),
            +1,
        )
    for head, r in ((t1.get("value") or {}).get("heads") or {}).items():
        for part in (*PARTS, "delayed"):
            put(f"t1.v.{head}.{part}.abs_bias", (r.get(part) or {}).get("abs_bias"), -1)
            put(f"t1.v.{head}.{part}.mae", (r.get(part) or {}).get("mae"), -1)
    t3 = diagnostics.get("t3") or {}
    for head, r in ((t3.get("value") or {}).get("heads") or {}).items():
        for group in GROUPS:
            g = (r.get("groups") or {}).get(group) or {}
            put(f"t3.v.{head}.{group}.nmae", g.get("nmae"), -1)
            put(f"t3.v.{head}.{group}.abs_bias", g.get("abs_bias"), -1)
        put(f"t3.v.{head}.beat_minus_teacher_nmae", r.get("beat_minus_teacher_nmae"), -1)
        put(f"t3.v.{head}.actor_minus_teacher_nmae", r.get("actor_minus_teacher_nmae"), -1)
    for label, report in (t3.get("advantage") or {}).items():
        for group, g in (report.get("groups") or {}).items():
            for key in ("utility:root", "qol_years"):
                put(f"t3.a.{label}.{group}.{key}", (g.get(key) or {}).get("accuracy"), +1)
            put(f"t3.a.{label}.{group}.beat_above_h12", g.get("beat_candidate_above_h12"), +1)
    t4 = diagnostics.get("t4") or {}
    for head, r in (t4.get("heads") or {}).items():
        put(f"t4.{head}.mae", r.get("mae"), -1)
        put(f"t4.{head}.abs_bias", r.get("abs_bias"), -1)
        if r.get("decisions"):
            sep = r.get("separation_decision")
            # Never separating ranks after the last recorded decision.
            put(f"t4.{head}.separation_decision", r["decisions"] if sep is None else sep, +1)
        for stage, s in (r.get("stages") or {}).items():
            put(f"t4.{head}.stage.{stage}.mae", s.get("mae"), -1)
    return out


def aggregate_seeds(per_seed: Mapping[Any, Mapping[str, Any] | None]) -> dict[str, Any]:
    """Seed mean, spread (max - min) and SD of every headline metric."""
    values: dict[str, list[float]] = defaultdict(list)
    directions: dict[str, int] = {}
    for diag in per_seed.values():
        for key, (value, direction) in headline_metrics(diag).items():
            values[key].append(value)
            directions[key] = direction
    return {
        key: {
            "mean": float(np.mean(v)),
            "spread": float(max(v) - min(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "seeds": len(v),
            "direction": directions[key],
        }
        for key, v in sorted(values.items())
    }


def receipt_section(
    evidence: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[str],
    seeds: Sequence[int],
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The receipt's ``trajectory_diagnostics`` section (never part of the verdict)."""
    by_arm: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for record in evidence:
        by_arm[str(record["arm"])][int(record["seed"])] = record
    section: dict[str, Any] = {
        "schema": SCHEMA,
        "gating": False,
        "note": "T1/T3/T4 (section 4a) never gate; a metric worse than its seed spread "
        "against the baseline checkpoint is flagged for review",
        "settings": SETTINGS,
        "arms": {},
    }
    for arm in arms:
        fits = by_arm.get(arm, {})
        per_seed = {seed: (fits.get(seed) or {}).get("trajectory_diagnostics") for seed in seeds}
        section["arms"][arm] = {
            "per_seed": {str(s): d for s, d in per_seed.items()},
            "summary": aggregate_seeds({s: d for s, d in per_seed.items() if d}),
            "missing_seeds": [s for s, d in per_seed.items() if not d],
        }
    if baseline is not None:
        try:
            section["comparison"] = compare(baseline, section)
        except ValueError as error:  # e.g. a baseline receipt that predates section 4a
            section["comparison"] = {"error": repr(error), "flagged_count": 0, "gating": False}
    return section


def _section(document: Mapping[str, Any]) -> dict[str, Any]:
    """A receipt section from a receipt, a section, or one evidence file."""
    if "trajectory_diagnostics" in document and document.get("schema", "").endswith("receipt-v1"):
        return dict(document["trajectory_diagnostics"])
    if document.get("schema") == SCHEMA and "arms" in document:
        return dict(document)
    if "trajectory_diagnostics" in document:  # one fit's evidence
        arm = str(document.get("arm", "fit"))
        return {
            "arms": {
                arm: {
                    "summary": aggregate_seeds(
                        {document.get("seed", 0): document["trajectory_diagnostics"]}
                    )
                }
            }
        }
    raise ValueError("document carries no trajectory diagnostics")


def compare(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    arm: str | None = None,
    baseline_arm: str | None = None,
) -> dict[str, Any]:
    """Flag every T-metric whose candidate mean is worse than baseline by more than the seed spread.

    The spread is the larger of the two checkpoints' seed spreads (max - min). A
    single-seed side has spread zero (``spread_available`` false). Never gates.
    """
    base, cand = _section(baseline), _section(candidate)
    arms = [arm] if arm else sorted(set(base.get("arms", {})) & set(cand.get("arms", {})))
    if arm and baseline_arm:
        pairs = [(baseline_arm, arm)]
    elif not arms and len(base.get("arms", {})) == 1 and len(cand.get("arms", {})) == 1:
        pairs = [(next(iter(base["arms"])), next(iter(cand["arms"])))]
    else:
        pairs = [(a, a) for a in arms]
    out: dict[str, Any] = {"schema": COMPARISON_SCHEMA, "gating": False, "arms": {}}
    flagged_total = 0
    for base_arm, cand_arm in pairs:
        b = base["arms"].get(base_arm, {}).get("summary", {})
        c = cand["arms"].get(cand_arm, {}).get("summary", {})
        rows, flags = {}, []
        for key in sorted(set(b) & set(c)):
            direction = c[key]["direction"]
            change = c[key]["mean"] - b[key]["mean"]
            worsening = -direction * change
            spread = max(b[key]["spread"], c[key]["spread"])
            available = b[key]["seeds"] > 1 or c[key]["seeds"] > 1
            flagged = worsening > spread and worsening > 1e-12
            rows[key] = {
                "baseline": b[key]["mean"],
                "candidate": c[key]["mean"],
                "change": change,
                "worsening": worsening,
                "seed_spread": spread,
                "spread_available": available,
                "flagged": flagged,
            }
            if flagged:
                flags.append(key)
        out["arms"][f"{base_arm}->{cand_arm}" if base_arm != cand_arm else cand_arm] = {
            "metrics": rows,
            "flagged": flags,
            "only_in_baseline": sorted(set(b) - set(c)),
            "only_in_candidate": sorted(set(c) - set(b)),
        }
        flagged_total += len(flags)
    out["flagged_count"] = flagged_total
    return out


def adaptation_series(
    documents: Sequence[tuple[str, Mapping[str, Any]]], *, arm: str | None = None
) -> dict[str, Any]:
    """T3 lag across refits (checkpoints in order): beat/actor minus teacher nMAE per head."""
    lag = float(setting("lag_gap"))
    series: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    labels = []
    for label, document in documents:
        section = _section(document)
        name = arm or next(iter(section["arms"]))
        summary = section["arms"][name]["summary"]
        labels.append(label)
        for head in COMPONENTS:
            for key in ("beat_minus_teacher_nmae", "actor_minus_teacher_nmae"):
                entry = summary.get(f"t3.v.{head}.{key}")
                series[head][key].append(None if entry is None else entry["mean"])
    heads = {}
    for head, entries in series.items():
        beat = entries["beat_minus_teacher_nmae"]
        known = [v for v in beat if v is not None]
        lagging = bool(known) and known[-1] > lag and (len(known) < 2 or known[-1] >= known[0])
        heads[head] = {**{k: v for k, v in entries.items()}, "lagging": lagging}
    return {"refits": labels, "lag_gap": lag, "heads": heads, "gating": False}


# ---------------------------------------------------------------------------
# One fit's diagnostics (labor_r5_acceptance.evaluate_fit)
# ---------------------------------------------------------------------------


def fit_diagnostics(
    *,
    task: Any,
    costs: Any,
    head: Any,
    calibration_dataset: Mapping[str, Any],
    reserved: Any,
    reserved_paths: Sequence[Path | str],
    extra_datasets: Sequence[tuple[str, Mapping[str, Any]]] = (),
    payload_sets: Sequence[tuple[str, Any]] = (),
    traces: Sequence[Mapping[str, Any]] = (),
    critic_digest: str | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    """T1 (A on reserved payloads, V on the calibration trajectories), T3 and T4 for one fit.

    Each part is computed independently; an error is recorded in place (these
    diagnostics never gate and never stop a receipt).
    """
    from .labor_advantage import critic_contexts, predict
    from .labor_pretrain_eval import critic_predictions

    out: dict[str, Any] = {"schema": SCHEMA, "gating": False, "t1": {}, "t3": {}, "t4": None}

    def guarded(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:  # diagnostics never stop the receipt
            return {"error": repr(error)}

    def advantage_prediction(candidates: Any) -> Any:
        tensors, privileged = candidates.root_tensors()
        return predict(head, critic_contexts(task, costs, tensors, privileged), candidates.roots)

    def values(dataset: Mapping[str, Any]) -> dict[str, np.ndarray]:
        n = int(dataset["row_world"].shape[0])
        preds = critic_predictions(task, costs, dataset, np.arange(n), batch_size)
        return {name: preds[name] for name in COMPONENTS}

    reserved_prediction = guarded(advantage_prediction, reserved)
    if isinstance(reserved_prediction, dict) and "error" in reserved_prediction:
        out["t1"]["advantage"] = reserved_prediction
    else:
        table = guarded(load_step_table, reserved_paths)
        out["t1"]["advantage"] = guarded(
            t1_advantage_report,
            reserved.roots,
            reserved_prediction,
            table if isinstance(table, StepTable) else None,
        )
    calibration_values = guarded(values, calibration_dataset)
    if "error" in calibration_values:
        out["t1"]["value"] = calibration_values
    else:
        out["t1"]["value"] = guarded(t1_value_report, calibration_values, calibration_dataset)
    sets = [("calibration", calibration_dataset, calibration_values)]
    for label, dataset in extra_datasets:
        predicted = guarded(values, dataset)
        if "error" not in predicted:
            sets.append((label, dataset, predicted))
    beat = guarded(branch_beat_keys, [d for _, d, _ in sets])
    out["t3"]["value"] = guarded(
        v_adaptation_report,
        [s for s in sets if "error" not in s[2]],
        beat=beat if "error" not in beat else None,
    )
    advantage: dict[str, Any] = {}
    if not (isinstance(reserved_prediction, dict) and "error" in reserved_prediction):
        advantage["reserved"] = guarded(a_adaptation_report, reserved.roots, reserved_prediction)
    for label, candidates in payload_sets:
        prediction = guarded(advantage_prediction, candidates)
        advantage[label] = (
            prediction
            if isinstance(prediction, dict) and "error" in prediction
            else guarded(a_adaptation_report, candidates.roots, prediction)
        )
    out["t3"]["advantage"] = advantage
    if traces:
        report = guarded(t4_report, traces)
        if critic_digest is not None and "critic_digests" in report:
            report["critic_digest_matches"] = report["critic_digests"] == [critic_digest]
        out["t4"] = report
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, default=float), encoding="utf-8")


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cmp_ = sub.add_parser("compare", help="flag T-metrics worse than their seed spread")
    cmp_.add_argument("--baseline", type=Path, required=True, help="receipt or evidence JSON")
    cmp_.add_argument("--candidate", type=Path, required=True, help="receipt or evidence JSON")
    cmp_.add_argument("--arm", default=None)
    cmp_.add_argument("--baseline-arm", default=None)
    cmp_.add_argument("--out", type=Path, default=None)
    series = sub.add_parser("series", help="T3 lag across refits (receipts in refit order)")
    series.add_argument("documents", nargs="+", help="LABEL=receipt.json ...")
    series.add_argument("--arm", default=None)
    series.add_argument("--out", type=Path, default=None)
    t4 = sub.add_parser("t4", help="T4 report from autonomous reports carrying critic traces")
    t4.add_argument("reports", type=Path, nargs="+")
    t4.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.command == "compare":
        result = compare(
            _read(args.baseline),
            _read(args.candidate),
            arm=args.arm,
            baseline_arm=args.baseline_arm,
        )
        for name, entry in result["arms"].items():
            print(f"{name}: {len(entry['flagged'])} flagged of {len(entry['metrics'])}")
            for key in entry["flagged"]:
                row = entry["metrics"][key]
                print(
                    f"  FLAG {key}: {row['baseline']:.4g} -> {row['candidate']:.4g} "
                    f"(worse by {row['worsening']:.4g} > spread {row['seed_spread']:.4g})"
                )
    elif args.command == "series":
        documents = []
        for item in args.documents:
            label, _, path = item.partition("=")
            documents.append((label, _read(Path(path or label))))
        result = adaptation_series(documents, arm=args.arm)
        for head, entry in result["heads"].items():
            gap = entry["beat_minus_teacher_nmae"]
            print(f"{head}: beat-teacher nMAE {gap} lagging={entry['lagging']}")
    else:
        result = t4_report(traces_from_reports(_read(p) for p in args.reports))
        for head, entry in result["heads"].items():
            print(
                f"{head}: separation={entry['separation_decision']} mae={entry['mae']} "
                f"decisions={entry['decisions']}"
            )
    if args.out is not None:
        _write(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
