"""Evaluation of the interim labor pretrain: actor fidelity, critic calibration, closed loop.

Three subcommands, each writing one JSON report (``--out``) and printing a
compact summary:

``actor-fidelity``
    Same-state fidelity of the BC actor on calibration teacher rows. The actor
    is decoded deterministically (masked-softmax means; mode of the
    categorical heads) and compared with the converted heuristic labels, using
    only valid, non-quarantined labels and world-balanced weights recomputed
    per head and per stratum. Strata: camp population bin and turn third.

``critic-calibration``
    The four critics against the discounted returns (tau = 400 years,
    ``labor_bc.discounted_returns``) closed by the dataset's declared tail: the
    randomized teacher tail, or the legacy truncated-zero window in which a
    living cutoff gets no tail; rows with an unavailable tail are excluded, on
    calibration rows, optionally also on the fitting dataset to show
    the generalisation gap. Task heads use the frozen labor reference scales;
    cost heads use scale 1.

``autonomous``
    Closed-loop S1 check on development worlds: the actor controls every
    eligible camp (deterministic and/or stochastic) versus the native
    heuristic controlling everything from the same reset snapshot. Mode
    ``h12`` instead submits the converted heuristic recommendation (exactly as
    ``labor_bc.collect_heuristic_world`` builds it) through the same action-12
    labor runtime, separating interface loss (h12 vs heuristic) from actor
    loss (actor vs h12).

Gate semantics (critic): PASS when every metric gate and the coverage gates
hold; FAIL when any metric gate fails; HOLD when the metrics pass but the
coverage (worlds, informative groups) is insufficient to decide.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPORT_SCHEMA = "labor-pretrain-eval-v1"
POPULATION_BINS: tuple[tuple[str, float, float], ...] = (
    ("<100", 0.0, 100.0),
    ("100-300", 100.0, 300.0),
    ("300-600", 300.0, 600.0),
    (">600", 600.0, math.inf),
)
TURN_THIRDS = ("early-third", "middle-third", "late-third")
FOCUS_ACTIVITIES: tuple[tuple[str, str], ...] = (
    ("Craftsperson", "goods.housing"),
    ("Craftsperson", "goods.light-housing"),
    ("Craftsperson", "goods.housing-material"),
    ("Farmer", "field.operation"),
    ("Farmer", "field.expansion"),
    ("Herder", "herd.management"),
    ("Herder", "herd.expansion"),
)
CRITIC_GATES = {
    "nmae_max": 0.20,
    "abs_bias_max": 0.10,
    "ranking_min": 0.70,
    "worlds_min": 12,
    "informative_groups_min": 30,
    "informative_delta_fraction": 0.05,
    "zero_head_tolerance": 0.05,
}
AUTONOMOUS_GATES = {"aggregate_ratio_min": 0.90, "per_world_ratio_min": 0.85}
GATED_AUTONOMOUS_METRICS = ("welfare", "person_years")
STARVATION_NUTRITION = 0.999
FARMER = "Farmer"


# ---------------------------------------------------------------------------
# Pure metric math (unit tested with synthetic tensors)
# ---------------------------------------------------------------------------


def apportion_largest_remainder_rows(shares: np.ndarray, population: np.ndarray) -> np.ndarray:
    """Row-wise native stable largest-remainder apportionment.

    Identical rule to ``labor_bridge.apportion_largest_remainder``: floor of
    ``share / sum * P``, then the remaining residents go to the largest
    fractional parts, ties broken by the lower ordinal.
    """
    shares = np.asarray(shares, dtype=np.float64)
    population = np.asarray(population).reshape(-1)
    if shares.ndim != 2 or shares.shape[0] != population.shape[0]:
        raise ValueError(
            "apportionment shares must be [rows, specialties] with one population per row"
        )
    out = np.zeros(shares.shape, dtype=np.int64)
    for row in range(shares.shape[0]):
        count = int(round(float(population[row])))
        values = np.clip(shares[row], 0.0, None)
        total = math.fsum(values.tolist())
        if count <= 0 or not total > 0.0:
            continue
        exact = values / total * count
        result = np.floor(exact).astype(np.int64)
        remainder = count - int(result.sum())
        fractional = exact - result
        order = sorted(range(len(result)), key=lambda i: (-fractional[i], i))
        for index in order[:remainder]:
            result[index] += 1
        out[row] = result
    return out


def world_balanced(worlds: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Weights giving every world with a kept row equal total mass (mean one)."""
    worlds = np.asarray(worlds).reshape(-1)
    keep = np.asarray(keep, dtype=bool).reshape(-1)
    weights = np.zeros(len(worlds), dtype=np.float64)
    kept = np.unique(worlds[keep])
    if len(kept) == 0:
        return weights
    total = float(keep.sum())
    for world in kept:
        rows = (worlds == world) & keep
        weights[rows] = total / (len(kept) * float(rows.sum()))
    return weights


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if not total > 0.0:
        return None
    return float((np.asarray(values, dtype=np.float64) * weights).sum() / total)


def normalized_mae(
    pred: np.ndarray, target: np.ndarray, scale: float, weights: np.ndarray | None = None
) -> float | None:
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    weights = np.ones(len(pred)) if weights is None else np.asarray(weights, np.float64)
    value = weighted_mean(np.abs(pred - target), weights)
    return None if value is None else value / float(scale)


def normalized_bias(
    pred: np.ndarray, target: np.ndarray, scale: float, weights: np.ndarray | None = None
) -> float | None:
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    weights = np.ones(len(pred)) if weights is None else np.asarray(weights, np.float64)
    value = weighted_mean(pred - target, weights)
    return None if value is None else value / float(scale)


def ranking_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    worlds: np.ndarray,
    turns: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Informative pairwise ordering accuracy within (world, decision turn) groups.

    A pair is informative when ``|G_i - G_j| > threshold``. It is correct when
    the prediction orders it the same way; a prediction tie counts one half.
    Returns pooled accuracy, a world-balanced (macro) accuracy, and counts.
    """
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    worlds, turns = np.asarray(worlds).reshape(-1), np.asarray(turns).reshape(-1)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, key in enumerate(zip(worlds.tolist(), turns.tolist())):
        groups[(int(key[0]), int(key[1]))].append(i)
    correct = 0.0
    pairs = 0
    informative_groups = 0
    per_world: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for (world, _), rows in groups.items():
        if len(rows) < 2:
            continue
        index = np.asarray(rows)
        g, p = target[index], pred[index]
        dg = g[:, None] - g[None, :]
        dp = p[:, None] - p[None, :]
        upper = np.triu(np.ones((len(rows), len(rows)), dtype=bool), k=1)
        informative = upper & (np.abs(dg) > threshold)
        count = int(informative.sum())
        if count == 0:
            continue
        agree = np.where(dp == 0.0, 0.5, (np.sign(dp) == np.sign(dg)).astype(np.float64))
        hits = float(agree[informative].sum())
        correct += hits
        pairs += count
        informative_groups += 1
        per_world[world][0] += hits
        per_world[world][1] += count
    macro = (
        float(np.mean([hits / count for hits, count in per_world.values()])) if per_world else None
    )
    return {
        "accuracy": correct / pairs if pairs else None,
        "world_balanced_accuracy": macro,
        "informative_pairs": pairs,
        "informative_groups": informative_groups,
        "informative_worlds": len(per_world),
        "groups": len(groups),
    }


def critic_gate(metrics: Mapping[str, Any], *, zero_head: bool) -> str:
    """PASS / HOLD / FAIL for one critic head on one row set."""
    if metrics.get("rows", 0) == 0:
        return "HOLD"
    if zero_head:
        tolerance = CRITIC_GATES["zero_head_tolerance"]
        if metrics["max_abs_pred"] <= tolerance:
            return "PASS"
        return "HOLD" if metrics["mean_abs_pred"] <= tolerance else "FAIL"
    failed = (
        metrics["nmae"] is None
        or metrics["nmae"] > CRITIC_GATES["nmae_max"]
        or abs(metrics["bias"]) > CRITIC_GATES["abs_bias_max"]
        or (
            metrics["ranking"]["accuracy"] is not None
            and metrics["ranking"]["accuracy"] < CRITIC_GATES["ranking_min"]
        )
    )
    if failed:
        return "FAIL"
    if (
        metrics["worlds"] < CRITIC_GATES["worlds_min"]
        or metrics["ranking"]["informative_groups"] < CRITIC_GATES["informative_groups_min"]
        or metrics["ranking"]["accuracy"] is None
    ):
        return "HOLD"
    return "PASS"


def decision_index(worlds: np.ndarray, turns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row position of its decision turn among its world's decisions, and that world's count."""
    worlds, turns = np.asarray(worlds).reshape(-1), np.asarray(turns).reshape(-1)
    index = np.zeros(len(worlds), dtype=np.int64)
    count = np.zeros(len(worlds), dtype=np.int64)
    for world in np.unique(worlds):
        rows = worlds == world
        unique = np.unique(turns[rows])
        index[rows] = np.searchsorted(unique, turns[rows])
        count[rows] = len(unique)
    return index, count


def population_bin(population: np.ndarray) -> np.ndarray:
    population = np.asarray(population, dtype=np.float64)
    labels = np.empty(len(population), dtype=object)
    for name, low, high in POPULATION_BINS:
        labels[(population >= low) & (population < high)] = name
    return labels


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _check_world_policy(worlds: Sequence[int], context: str) -> None:
    from .labor_bc import SEALED_WORLDS

    sealed = sorted(int(w) for w in worlds if int(w) in SEALED_WORLDS)
    if sealed:
        raise ValueError(
            f"{context}: sealed confirmation worlds {sealed[:5]} are never evaluated here"
        )


def _load_models(path: Path) -> tuple[Any, ...]:
    from .labor_training import _models_from_checkpoint

    state = torch.load(path, map_location="cpu", weights_only=True)
    layout, actor, task, costs, normalizer = _models_from_checkpoint(state)
    for module in (actor, task, costs):
        module.eval()
    return state, actor, task, costs, normalizer


def _load_dataset(path: Path, normalizer: Any, *, require_privileged: bool = False) -> dict[str, Any]:
    from .labor_teacher_data import load_labor_dataset
    from .labor_training import _normalization_digest

    dataset = load_labor_dataset(
        path,
        expected_normalizer_digest=_normalization_digest(normalizer),
        require_privileged=require_privileged,
    )
    _check_world_policy(dataset["row_world"].tolist(), f"dataset {path}")
    return dataset


def _input_tensors(dataset: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Observation inputs only; critic-only privileged arrays are never included."""
    from .labor_privileged import is_privileged_key

    return {
        k: v
        for k, v in dataset.items()
        if isinstance(v, torch.Tensor)
        and not is_privileged_key(k)
        and not k.startswith(("target_", "label_", "reward_", "transition_", "row_", "tail_"))
    }


def _batches(tensors: Mapping[str, torch.Tensor], rows: np.ndarray, batch_size: int):
    for start in range(0, len(rows), batch_size):
        index = torch.as_tensor(rows[start : start + batch_size], dtype=torch.long)
        yield index, {k: v[index] for k, v in tensors.items()}


# ---------------------------------------------------------------------------
# 1. Actor fidelity
# ---------------------------------------------------------------------------


def actor_predictions(
    actor: Any, dataset: Mapping[str, Any], batch_size: int = 256
) -> dict[str, np.ndarray]:
    """Deterministic actor decoding on every dataset row."""
    from .labor_model import _masked_softmax, _masks

    tensors = _input_tensors(dataset)
    n = int(dataset["row_world"].shape[0])
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for _, batch in _batches(tensors, np.arange(n), batch_size):
            output = actor(batch)
            spec_mask, act_mask = _masks(batch, len(batch["core"]))
            out["specialty"].append(
                _masked_softmax(output["specialty_logits"], spec_mask).double().numpy()
            )
            out["activity"].append(
                _masked_softmax(output["activity_logits"], act_mask).double().numpy()
            )
            movement = output["movement_union_logits"].masked_fill(
                ~output["movement_union_mask"], -torch.inf
            )
            out["movement_code"].append(output["movement_codes"][movement.argmax(dim=1)].numpy())
            codes = output["movement_codes"]
            out["movement_union_mask"].append(output["movement_union_mask"].numpy())
            out["fission"].append(output["fission"].argmax(dim=1).numpy())
            out["trade_gate"].append(output["trade_gate"].argmax(dim=2).numpy())
    result = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    result["movement_codes"] = codes.numpy()
    return result


def _np(dataset: Mapping[str, Any], key: str) -> np.ndarray:
    return dataset[key].detach().cpu().numpy()


def actor_fidelity_metrics(arrays: Mapping[str, np.ndarray], select: np.ndarray) -> dict[str, Any]:
    """Fidelity metrics for the selected rows (each head's own valid rows, world-balanced)."""
    from .labor_bridge import ACTIVITIES, SPECIALTIES

    worlds = arrays["row_world"]
    keep = select & ~arrays["row_quarantined"]
    spec_valid = keep & arrays["label_specialty_valid"]
    spec_mask = arrays["labor_specialty_mask"]
    act_mask = arrays["labor_activity_mask"]
    target_spec, pred_spec = arrays["target_specialty"], arrays["pred_specialty"]
    report: dict[str, Any] = {
        "rows": int(select.sum()),
        "admitted_rows": int(keep.sum()),
        "specialty_rows": int(spec_valid.sum()),
        "worlds": len(np.unique(worlds[spec_valid])) if spec_valid.any() else 0,
    }

    # Specialty shares and compiled integer residents.
    w = world_balanced(worlds, spec_valid)
    l1_rows = (np.abs(pred_spec - target_spec) * spec_mask).sum(axis=1)
    report["specialty_l1"] = weighted_mean(l1_rows, w)
    counts_target = arrays["target_specialty_counts"]
    counts_pred = arrays["pred_specialty_counts"]
    report["resident_abs_residual_per_camp"] = weighted_mean(
        np.abs(counts_pred - counts_target).sum(axis=1), w
    )
    per_specialty = {}
    for s, name in enumerate(SPECIALTIES):
        legal = spec_valid & spec_mask[:, s]
        if not legal.any():
            continue
        ws = w * legal
        per_specialty[name] = {
            "legal_rows": int(legal.sum()),
            "pred_share": weighted_mean(pred_spec[:, s], ws),
            "target_share": weighted_mean(target_spec[:, s], ws),
            "pred_residents": weighted_mean(counts_pred[:, s], ws),
            "target_residents": weighted_mean(counts_target[:, s], ws),
            "resident_residual": weighted_mean(counts_pred[:, s] - counts_target[:, s], ws),
        }
    report["per_specialty"] = per_specialty

    # Activity within staffed roles, weighted by n_r/P.
    target_act, pred_act = arrays["target_activity"], arrays["pred_activity"]
    role_valid = arrays["label_activity_role_valid"]
    role_weight = (
        w[:, None] * target_spec[:, 1:] * role_valid * spec_valid[:, None] * act_mask.any(axis=2)
    )
    role_l1 = (np.abs(pred_act - target_act) * act_mask).sum(axis=2)
    total = float(role_weight.sum())
    report["activity_l1"] = float((role_l1 * role_weight).sum() / total) if total > 0 else None
    per_role = {}
    for r in range(role_weight.shape[1]):
        weights = role_weight[:, r]
        if not float(weights.sum()) > 0.0:
            continue
        name = SPECIALTIES[r + 1]
        activities = {}
        for a, activity in enumerate(ACTIVITIES):
            legal = act_mask[:, r, a] & (weights > 0)
            if not legal.any():
                continue
            activities[activity] = {
                "pred_share": weighted_mean(pred_act[:, r, a], weights * legal),
                "target_share": weighted_mean(target_act[:, r, a], weights * legal),
            }
        per_role[name] = {
            "staffed_rows": int((weights > 0).sum()),
            "weight": float(weights.sum()),
            "l1": weighted_mean(role_l1[:, r], weights),
            "activities": activities,
        }
    report["per_role_activity"] = per_role
    report["focus_activities"] = {
        f"{role}:{activity}": per_role.get(role, {}).get("activities", {}).get(activity)
        for role, activity in FOCUS_ACTIVITIES
    }

    # Movement: valid label whose target code is legal in the union.
    codes = arrays["movement_codes"]
    target_codes = arrays["target_movement_code"]
    match = codes[None, :] == target_codes[:, None]
    legal_target = (match & arrays["pred_movement_union_mask"]).any(axis=1)
    move_valid = keep & arrays["label_movement_valid"] & legal_target
    report["movement"] = {
        "rows": int(move_valid.sum()),
        "accuracy": weighted_mean(
            arrays["pred_movement_code"] == target_codes, world_balanced(worlds, move_valid)
        ),
        "target_stay_share": weighted_mean(target_codes == -1, world_balanced(worlds, move_valid)),
        "pred_stay_share": weighted_mean(
            arrays["pred_movement_code"] == -1, world_balanced(worlds, move_valid)
        ),
    }
    fission_valid = keep & arrays["label_fission_valid"]
    report["fission"] = {
        "rows": int(fission_valid.sum()),
        "accuracy": weighted_mean(
            arrays["pred_fission"] == arrays["target_fission"],
            world_balanced(worlds, fission_valid),
        ),
    }
    gate_valid = arrays["label_trade_gate_valid"] & keep[:, None]
    gate_rows = gate_valid.any(axis=1)
    hits = ((arrays["pred_trade_gate"] == arrays["target_trade_gate"]) & gate_valid).sum(axis=1)
    per_row = hits / np.maximum(1, gate_valid.sum(axis=1))
    report["trade_gate"] = {
        "rows": int(gate_rows.sum()),
        "gates": int(gate_valid.sum()),
        "accuracy": weighted_mean(per_row, world_balanced(worlds, gate_rows)),
    }
    return report


def _actor_arrays(actor: Any, dataset: Mapping[str, Any], batch_size: int) -> dict[str, np.ndarray]:
    keys = (
        "row_world",
        "row_turn",
        "row_camp",
        "row_quarantined",
        "label_specialty_valid",
        "label_activity_role_valid",
        "label_movement_valid",
        "label_fission_valid",
        "label_trade_gate_valid",
        "labor_specialty_mask",
        "labor_activity_mask",
        "target_specialty",
        "target_activity",
        "target_specialty_counts",
        "target_movement_code",
        "target_fission",
        "target_trade_gate",
    )
    arrays = {key: _np(dataset, key) for key in keys}
    for key in (
        "row_quarantined",
        "label_specialty_valid",
        "label_movement_valid",
        "label_fission_valid",
    ):
        arrays[key] = arrays[key].reshape(-1).astype(bool)
    arrays["target_specialty"] = arrays["target_specialty"].astype(np.float64)
    arrays["target_activity"] = arrays["target_activity"].astype(np.float64)
    arrays["target_movement_code"] = arrays["target_movement_code"].reshape(-1)
    arrays["target_fission"] = arrays["target_fission"].reshape(-1)
    pred = actor_predictions(actor, dataset, batch_size)
    arrays["pred_specialty"] = pred["specialty"]
    arrays["pred_activity"] = pred["activity"]
    arrays["pred_movement_code"] = pred["movement_code"]
    arrays["pred_movement_union_mask"] = pred["movement_union_mask"]
    arrays["movement_codes"] = pred["movement_codes"]
    arrays["pred_fission"] = pred["fission"]
    arrays["pred_trade_gate"] = pred["trade_gate"]
    # P is the label's resident total (exact for valid labels: staffed + residual Gatherers).
    arrays["population"] = arrays["target_specialty_counts"].sum(axis=1)
    arrays["pred_specialty_counts"] = apportion_largest_remainder_rows(
        arrays["pred_specialty"] * arrays["labor_specialty_mask"], arrays["population"]
    ).astype(np.float64)
    return arrays


def run_actor_fidelity(
    checkpoint: Path, dataset_path: Path, batch_size: int = 256
) -> dict[str, Any]:
    _, actor, _, _, normalizer = _load_models(checkpoint)
    dataset = _load_dataset(dataset_path, normalizer)
    arrays = _actor_arrays(actor, dataset, batch_size)
    n = len(arrays["row_world"])
    everything = np.ones(n, dtype=bool)
    bins = population_bin(arrays["population"])
    index, count = decision_index(arrays["row_world"], arrays["row_turn"])
    third = np.minimum(2, (index * 3) // np.maximum(1, count))
    report = {
        "schema": REPORT_SCHEMA,
        "command": "actor-fidelity",
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_path),
        "dataset_digest": dataset["collection"]["dataset_digest"],
        "decoding": "deterministic masked-softmax means; argmax movement/fission/trade gate",
        "weighting": "world-balanced per head over valid, non-quarantined labels",
        "population_source": "label resident total (staffed + residual Gatherers)",
        "overall": actor_fidelity_metrics(arrays, everything),
        "by_population_bin": {
            name: actor_fidelity_metrics(arrays, bins == name)
            for name, _, _ in POPULATION_BINS
            if (bins == name).any()
        },
        "by_turn_third": {
            TURN_THIRDS[t]: actor_fidelity_metrics(arrays, third == t)
            for t in range(3)
            if (third == t).any()
        },
    }
    return report


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def summarize_actor(report: Mapping[str, Any]) -> str:
    o = report["overall"]
    lines = [
        f"actor-fidelity rows={o['rows']} admitted={o['admitted_rows']} worlds={o['worlds']}",
        f"  specialty L1={_fmt(o['specialty_l1'])} activity L1={_fmt(o['activity_l1'])} "
        f"resident |residual|/camp={_fmt(o['resident_abs_residual_per_camp'], 2)}",
        f"  movement acc={_fmt(o['movement']['accuracy'])} (n={o['movement']['rows']}) "
        f"fission acc={_fmt(o['fission']['accuracy'])} trade-gate acc={_fmt(o['trade_gate']['accuracy'])}",
    ]
    for name, s in o["per_specialty"].items():
        lines.append(
            f"  {name:<13} share pred={_fmt(s['pred_share'])} tgt={_fmt(s['target_share'])} "
            f"residents pred={_fmt(s['pred_residents'], 1)} tgt={_fmt(s['target_residents'], 1)}"
        )
    for key, value in o["focus_activities"].items():
        if value is not None:
            lines.append(
                f"  {key:<36} pred={_fmt(value['pred_share'])} tgt={_fmt(value['target_share'])}"
            )
    for group in ("by_population_bin", "by_turn_third"):
        for name, s in report[group].items():
            lines.append(
                f"  [{name}] rows={s['rows']} spec L1={_fmt(s['specialty_l1'])} act L1={_fmt(s['activity_l1'])} "
                f"move={_fmt(s['movement']['accuracy'])} fission={_fmt(s['fission']['accuracy'])} "
                f"gate={_fmt(s['trade_gate']['accuracy'])}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Critic calibration
# ---------------------------------------------------------------------------


def critic_predictions(
    task: Any,
    costs: Any,
    dataset: Mapping[str, Any],
    rows: np.ndarray,
    batch_size: int = 256,
    observed_max: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Decoded predictions; with ``observed_max`` (normalized target max per task
    head) also per-row value-support tail terms keyed ``"<head>/<term>"``."""
    from .labor_model import LABOR_COSTS, LABOR_OBJECTIVES, critic_privileged_batch

    tensors = _input_tensors(dataset)
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for index, batch in _batches(tensors, rows, batch_size):
            # Critic-only privileged batch, separate from the observation batch.
            privileged = critic_privileged_batch(task, dataset, index)
            logits = task(batch, privileged)
            for name in LABOR_OBJECTIVES:
                out[name].append(task.expectation(name, logits[name]).double().numpy())
                if observed_max is not None:
                    terms = task.support_diagnostics(name, logits[name], observed_max[name])
                    for term, value in terms.items():
                        out[f"{name}/{term}"].append(value.numpy())
            for name, value in costs(batch, critic_privileged_batch(costs, dataset, index)).items():
                out[name].append(value.double().numpy())
    names = tuple(LABOR_OBJECTIVES) + tuple(LABOR_COSTS)
    names += tuple(key for key in out if "/" in key)
    return {name: np.concatenate(out[name]) if out[name] else np.zeros(0) for name in names}


def value_support_metrics(
    task: Any,
    preds: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    worlds: np.ndarray,
) -> dict[str, Any]:
    """Brief section 17.2 support report: clipping and decoded upper-bin tails.

    Means are world-balanced like nMAE/bias and in reference-scale units, so
    ``upper_excess`` is directly comparable with the head's bias.
    """
    from .labor_model import LABOR_OBJECTIVE_REFERENCE_SCALES, LABOR_OBJECTIVES

    support = task.support
    weights = world_balanced(worlds, np.ones(len(worlds), dtype=bool))
    report: dict[str, Any] = {
        "support": {"low": support.low, "high": support.high, "bins": support.bins},
        "decoded_max_reference_scales": float(math.expm1(support.high)),
    }
    for name in LABOR_OBJECTIVES:
        normalized = targets[name] / LABOR_OBJECTIVE_REFERENCE_SCALES[name]
        y = np.sign(normalized) * np.log1p(np.abs(normalized))
        head: dict[str, Any] = {
            "observed_normalized_max": float(normalized.max()) if len(y) else None,
            "observed_symlog_min": float(y.min()) if len(y) else None,
            "observed_symlog_max": float(y.max()) if len(y) else None,
            "clip_below_fraction": float((y < support.low).mean()) if len(y) else 0.0,
            "clip_above_fraction": float((y > support.high).mean()) if len(y) else 0.0,
        }
        for term in ("upper_mass", "upper_contribution", "upper_excess", "top_bin_mass"):
            values = preds.get(f"{name}/{term}")
            head[term] = None if values is None or not len(values) else weighted_mean(values, weights)
        excess = preds.get(f"{name}/upper_excess")
        head["upper_excess_max"] = None if excess is None or not len(excess) else float(excess.max())
        report[name] = head
    return report


def critic_head_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    worlds: np.ndarray,
    turns: np.ndarray,
    scale: float,
    *,
    zero_head: bool,
) -> dict[str, Any]:
    weights = world_balanced(worlds, np.ones(len(worlds), dtype=bool))
    metrics: dict[str, Any] = {
        "rows": len(pred),
        "worlds": len(np.unique(worlds)),
        "scale": float(scale),
        "nmae": normalized_mae(pred, target, scale, weights),
        "bias": normalized_bias(pred, target, scale, weights),
        "target_mean": weighted_mean(target, weights),
        "pred_mean": weighted_mean(pred, weights),
        "ranking": ranking_accuracy(
            pred, target, worlds, turns, CRITIC_GATES["informative_delta_fraction"] * scale
        ),
        "zero_target_head": bool(zero_head),
    }
    if zero_head:
        metrics["max_abs_pred"] = float(np.abs(pred).max()) if len(pred) else 0.0
        metrics["mean_abs_pred"] = float(np.abs(pred).mean()) if len(pred) else 0.0
        metrics["tolerance"] = CRITIC_GATES["zero_head_tolerance"]
    if metrics["bias"] is None:
        metrics["bias"] = 0.0
    metrics["gate"] = critic_gate(metrics, zero_head=zero_head)
    return metrics


def critic_dataset_report(
    task: Any, costs: Any, dataset: Mapping[str, Any], batch_size: int = 256
) -> dict[str, Any]:
    from .labor_bc import critic_target_available, discounted_returns, return_tail_method
    from .labor_bridge import SPECIALTIES
    from .labor_model import LABOR_COST_REFERENCE_SCALES, LABOR_OBJECTIVE_REFERENCE_SCALES

    returns = {k: v.double().numpy() for k, v in discounted_returns(dataset).items()}
    quarantined = _np(dataset, "row_quarantined").reshape(-1).astype(bool)
    available = critic_target_available(dataset).numpy().astype(bool)
    keep = ~quarantined & available
    rows = np.flatnonzero(keep)
    worlds = _np(dataset, "row_world").reshape(-1)[rows]
    turns = _np(dataset, "row_turn").reshape(-1)[rows]
    farmer = SPECIALTIES.index(FARMER)
    farmer_unlocked = _np(dataset, "labor_specialty_mask")[rows, farmer].astype(bool)
    index, count = decision_index(worlds, turns)
    early = index < (count + 1) // 2
    observed_max = {
        name: float((returns[name][rows] / LABOR_OBJECTIVE_REFERENCE_SCALES[name]).max())
        if len(rows)
        else 0.0
        for name in LABOR_OBJECTIVE_REFERENCE_SCALES
    }
    preds = critic_predictions(task, costs, dataset, rows, batch_size, observed_max)
    scales = {**LABOR_OBJECTIVE_REFERENCE_SCALES, **LABOR_COST_REFERENCE_SCALES}
    strata = {
        "all": np.ones(len(rows), dtype=bool),
        "early_turns": early,
        "late_turns": ~early,
        "farmer_unlocked": farmer_unlocked,
        "farmer_locked": ~farmer_unlocked,
    }
    heads = {}
    for name, scale in scales.items():
        target = returns[name][rows]
        zero_head = bool(np.all(target == 0.0))
        heads[name] = {
            stratum: critic_head_metrics(
                preds[name][m], target[m], worlds[m], turns[m], scale, zero_head=zero_head
            )
            for stratum, m in strata.items()
            if m.any()
        }
    return {
        "rows": len(rows),
        "excluded_quarantined_rows": int(quarantined.sum()),
        "excluded_tail_unavailable_rows": int((~quarantined & ~available).sum()),
        "return_tail": return_tail_method(dataset),
        "worlds": len(np.unique(worlds)),
        "dataset_digest": dataset["collection"]["dataset_digest"],
        "heads": heads,
        "value_support": value_support_metrics(
            task, preds, {name: returns[name][rows] for name in observed_max}, worlds
        ),
    }


def run_critic_calibration(
    checkpoint: Path, dataset_path: Path, fit_dataset_path: Path | None, batch_size: int = 256
) -> dict[str, Any]:
    from .labor_bc import CRITIC_DISCOUNT_YEARS

    _, _, task, costs, normalizer = _load_models(checkpoint)
    privileged = task.privileged_layout is not None
    calibration_dataset = _load_dataset(dataset_path, normalizer, require_privileged=privileged)
    if privileged:
        fitted = set(task.privileged_normalizer.fit_record.get("worlds", []))
        overlap = sorted(fitted & {int(w) for w in calibration_dataset["row_world"].reshape(-1).tolist()})
        if overlap:
            raise ValueError(
                f"calibration worlds {overlap[:5]} fitted the privileged normalizer; not held out"
            )
    calibration = critic_dataset_report(task, costs, calibration_dataset, batch_size)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "command": "critic-calibration",
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_path),
        "targets": (
            "discounted returns over each (world, camp) trajectory, closed at the collection "
            f"horizon by the dataset's declared tail ({calibration['return_tail']}); "
            "truncated-zero gives a living cutoff no tail"
        ),
        "discount_tau_years": CRITIC_DISCOUNT_YEARS,
        "gates": CRITIC_GATES,
        "gate_semantics": "PASS all gates; FAIL any metric gate; HOLD metrics pass but coverage insufficient",
        "weighting": "nMAE/bias world-balanced; ranking pooled over informative pairs (world-balanced also shown)",
        "calibration": calibration,
        "verdict": {name: strata["all"]["gate"] for name, strata in calibration["heads"].items()},
    }
    if privileged:
        report["privileged_critic"] = {
            "schema_version": int(task.privileged_layout.schema_version),
            "normalizer_digest": task.privileged_normalizer.digest(),
            "normalizer_fit_worlds": list(task.privileged_normalizer.fit_record.get("worlds", [])),
        }
    if fit_dataset_path is not None:
        fitting = critic_dataset_report(
            task,
            costs,
            _load_dataset(fit_dataset_path, normalizer, require_privileged=privileged),
            batch_size,
        )
        report["fit_dataset"] = str(fit_dataset_path)
        report["fitting"] = fitting
        gap = {}
        for name, strata in calibration["heads"].items():
            c, f = strata["all"], fitting["heads"][name]["all"]
            gap[name] = {
                "nmae_cal_minus_fit": None
                if c["nmae"] is None or f["nmae"] is None
                else c["nmae"] - f["nmae"],
                "bias_cal_minus_fit": c["bias"] - f["bias"],
                "ranking_cal_minus_fit": (
                    None
                    if c["ranking"]["accuracy"] is None or f["ranking"]["accuracy"] is None
                    else c["ranking"]["accuracy"] - f["ranking"]["accuracy"]
                ),
            }
        report["generalisation_gap"] = gap
    return report


def summarize_critic(report: Mapping[str, Any]) -> str:
    lines = []
    for label in ("calibration", "fitting"):
        if label not in report:
            continue
        section = report[label]
        lines.append(
            f"critic-calibration [{label}] rows={section['rows']} worlds={section['worlds']}"
        )
        for name, strata in section["heads"].items():
            for stratum, m in strata.items():
                rank = m["ranking"]
                extra = (
                    f" max|pred|={_fmt(m['max_abs_pred'])} mean|pred|={_fmt(m['mean_abs_pred'])}"
                    if m["zero_target_head"]
                    else ""
                )
                lines.append(
                    f"  {name:<17} {stratum:<16} {m['gate']:<4} nMAE={_fmt(m['nmae'])} bias={_fmt(m['bias'])} "
                    f"rank={_fmt(rank['accuracy'])} pairs={rank['informative_pairs']} "
                    f"groups={rank['informative_groups']} worlds={m['worlds']}{extra}"
                )
        support = section.get("value_support")
        if support:
            s = support["support"]
            for name, h in support.items():
                if name in ("support", "decoded_max_reference_scales"):
                    continue
                lines.append(
                    f"  support [{s['low']:g},{s['high']:g}]x{s['bins']} {name:<17} "
                    f"y=[{_fmt(h['observed_symlog_min'])},{_fmt(h['observed_symlog_max'])}] "
                    f"clip<={_fmt(h['clip_below_fraction'], 4)} clip>={_fmt(h['clip_above_fraction'], 4)} "
                    f"mass>max={_fmt(h['upper_mass'], 4)} top-bin={_fmt(h['top_bin_mass'], 5)} "
                    f"decoded>max={_fmt(h['upper_contribution'])} excess={_fmt(h['upper_excess'])}"
                )
    for name, gap in (report.get("generalisation_gap") or {}).items():
        lines.append(
            f"  gap {name:<17} nMAE={_fmt(gap['nmae_cal_minus_fit'])} bias={_fmt(gap['bias_cal_minus_fit'])} "
            f"rank={_fmt(gap['ranking_cal_minus_fit'])}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Autonomous closed loop
# ---------------------------------------------------------------------------


def _live(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        v for v in step["villages"] if v.get("observation") is not None and not v.get("terminated")
    ]


def _deterministic_labor_sample(
    output: Mapping[str, torch.Tensor], tensors: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    from .labor_model import _masked_softmax, _masks

    spec_mask, act_mask = _masks(tensors, output["specialty_logits"].shape[0])

    def latent(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        other = mask.clone()
        other[..., 0] = False
        return torch.where(other, logits - logits[..., :1], torch.zeros_like(logits))

    spec_latent = latent(output["specialty_logits"], spec_mask)
    act_latent = latent(output["activity_logits"], act_mask)
    return {
        "specialty_allocation": _masked_softmax(spec_latent, spec_mask),
        "activity_allocation": _masked_softmax(act_latent, act_mask),
        "specialty_latent": spec_latent,
        "activity_latent": act_latent,
        "specialty_mask": spec_mask.clone(),
        "activity_mask": act_mask.clone(),
    }


def _turn_record(
    step: Mapping[str, Any],
    farmers: int,
    farmer_population: int,
    prior_terminated: set[int],
    last_nutrition: Mapping[int, float],
) -> dict[str, Any]:
    welfare = person_years = 0.0
    housing = clothing = coverage_people = 0.0
    terminations: list[int] = []
    starvation: list[int] = []
    for v in step["villages"]:
        camp = int(v["camp"])
        prior = max(0, int(v["priorPopulation"]))
        if prior > 0:
            welfare += float(v["welfare"])
        if not v["terminated"]:
            person_years += prior * float(v["years"])
            people = max(0, int(v["population"]))
            housing += float(v["housing"]) * people
            clothing += float(v["clothing"]) * people
            coverage_people += people
        elif camp not in prior_terminated:
            terminations.append(camp)
            nutrition = min(float(v.get("nutrition", 1.0)), float(last_nutrition.get(camp, 1.0)))
            if nutrition < STARVATION_NUTRITION:
                starvation.append(camp)
    return {
        "turn": int(step["turn"]),
        "welfare": welfare,
        "person_years": person_years,
        "qol": welfare / person_years if person_years > 0 else 0.0,
        "farmer_share": farmers / farmer_population if farmer_population > 0 else 0.0,
        "housing_coverage": housing / coverage_people if coverage_people > 0 else 0.0,
        "clothing_coverage": clothing / coverage_people if coverage_people > 0 else 0.0,
        "live_settlements": sum(1 for v in step["villages"] if not v["terminated"]),
        "residents": sum(int(v["population"]) for v in step["villages"] if not v["terminated"]),
        "terminations": terminations,
        "starvation_extinctions": starvation,
    }


def _track(step: Mapping[str, Any], terminated: set[int], nutrition: dict[int, float]) -> None:
    for v in step["villages"]:
        if v["terminated"]:
            terminated.add(int(v["camp"]))
        else:
            nutrition[int(v["camp"])] = float(v.get("nutrition", 1.0))


def _farmer_from_diagnostics(
    step: Mapping[str, Any], acted: Mapping[int, int], years: float
) -> int:
    from .labor_bc import _get, _num

    diagnostics = step.get("economicDiagnostics") or {}
    farmers = 0
    for camp in acted:
        diag = diagnostics.get(str(camp))
        cultivation = _get(_get(diag, "domains", default={}), "cultivation")
        assigned = _num(cultivation, "assignedFarmerYears")
        if assigned is not None and assigned > 1e-6 and years > 0:
            farmers += math.ceil(assigned / years - 1e-6)
    return farmers


def _run_heuristic(job: Mapping[str, Any]) -> dict[str, Any]:
    from .bridge import Model9Bridge
    from .labor_bc import LaborResetOptions
    from .labor_bridge import LaborModel9Bridge

    options = LaborResetOptions.from_mapping(job["reset_options"])
    world = int(job["world"])
    with LaborModel9Bridge() as labor_bridge:
        labor_bridge.reset(world, **options.reset_kwargs())
        identity = labor_bridge.last_reset_identity
        snapshot = labor_bridge._bridge.capture()
    turns: list[dict[str, Any]] = []
    with Model9Bridge(economic_diagnostics=True) as bridge:
        step = bridge.reset(
            world, movement=options.movement, withheld=options.withheld, snapshot=snapshot
        )
        terminated: set[int] = set()
        nutrition: dict[int, float] = {}
        _track(step, terminated, nutrition)
        for _ in range(int(job["turns"])):
            live = _live(step)
            if not live:
                break
            acted = {int(v["camp"]): int(v["population"]) for v in live}
            years = float(step.get("years") or 0.0)
            next_step = bridge.step({})
            years = float(next_step.get("years") or years)
            farmers = _farmer_from_diagnostics(next_step, acted, years)
            turns.append(
                _turn_record(next_step, farmers, sum(acted.values()), terminated, nutrition)
            )
            _track(next_step, terminated, nutrition)
            step = next_step
    return {"world": world, "mode": "heuristic", "reset_identity": identity, "turns": turns}


def actor_labor_decision(
    bridge: Any,
    actor: Any,
    normalizer: Any,
    personality: Any,
    live: Sequence[Mapping[str, Any]],
    catalogs: Mapping[int, Any],
    tensors: Mapping[str, torch.Tensor],
    *,
    stochastic: bool,
    generator: torch.Generator,
) -> Any:
    """The actor controls every live camp for one decision through the action-12 runtime.

    Deterministic mode decodes masked-softmax means and categorical modes;
    stochastic mode samples with ``generator``. A rejected initial request is
    retried once with the actor conditioned on its rejected attempt. Shared by
    the autonomous evaluation and the DAgger learner-state collector.
    """
    from .labor_model import (
        deterministic_nonlabor_policy,
        labor_policy_tensors,
        sample_labor_policy,
        sample_nonlabor_policy,
        schema12_initial_action,
        schema12_retry_action,
    )
    from .labor_ppo import RetainedLaborAction, execute_labor_decision
    from .labor_training import _one_row

    village_by_camp = {int(v["camp"]): v for v in live}
    with torch.no_grad():
        output = actor(tensors)
        if stochastic:
            labor = sample_labor_policy(output, tensors, generator=generator)
            nonlabor = sample_nonlabor_policy(output, generator=generator)
        else:
            labor = _deterministic_labor_sample(output, tensors)
            nonlabor = deterministic_nonlabor_policy(output)
    retained = {
        int(village["camp"]): RetainedLaborAction(
            schema12_initial_action(labor, nonlabor, row),
            _one_row(labor, row),
            0.0,
            _one_row(nonlabor, row),
            0.0,
        )
        for row, village in enumerate(live)
    }

    def retry_factory(camp: int, catalog: Any, attempt: Any) -> RetainedLaborAction:
        _, retry_tensors = labor_policy_tensors(
            [village_by_camp[camp]],
            normalizer,
            personality=personality,
            rejected_attempts=[attempt],
        )
        with torch.no_grad():
            retry_output = actor(retry_tensors)
            sample = (
                sample_labor_policy(retry_output, retry_tensors, generator=generator)
                if stochastic
                else _deterministic_labor_sample(retry_output, retry_tensors)
            )
        return RetainedLaborAction(
            schema12_retry_action(sample, 0), _one_row(sample, 0), 0.0, None, None
        )

    return execute_labor_decision(bridge, catalogs, retained, retry_factory)


def _run_actor(job: Mapping[str, Any]) -> dict[str, Any]:
    from .labor_bc import LaborResetOptions
    from .labor_bridge import SPECIALTIES, LaborModel9Bridge
    from .labor_model import labor_checkpoint_personality, labor_policy_tensors
    from .labor_personality import personality_for_world

    torch.set_num_threads(1)
    state, actor, _, _, normalizer = _load_models(Path(job["checkpoint"]))
    world = int(job["world"])
    stochastic = job["mode"] == "stochastic"
    generator = torch.Generator().manual_seed(int(job["policy_seed"]) * 1_000_003 + world)
    personality = personality_for_world(labor_checkpoint_personality(state), world)
    options = LaborResetOptions.from_mapping(job["reset_options"])
    farmer = SPECIALTIES.index(FARMER)
    turns: list[dict[str, Any]] = []
    rejected = retries = 0
    with LaborModel9Bridge() as bridge:
        step = bridge.reset(world, **options.reset_kwargs())
        identity = bridge.last_reset_identity
        terminated: set[int] = set()
        nutrition: dict[int, float] = {}
        _track(step, terminated, nutrition)
        for _ in range(int(job["turns"])):
            live = _live(step)
            if not live:
                break
            contexts, tensors = labor_policy_tensors(live, normalizer, personality=personality)
            catalogs = {context[0].camp_id: context[0] for context in contexts}
            execution = actor_labor_decision(
                bridge,
                actor,
                normalizer,
                personality,
                live,
                catalogs,
                tensors,
                stochastic=stochastic,
                generator=generator,
            )
            next_step = execution.step
            rejected += sum(1 for a in execution.initial_attempts.values() if not a.valid)
            retries += len(execution.retry)
            farmers = sum(
                int(a.accepted_specialty_residents[farmer])
                for a in execution.final_attempts.values()
            )
            population = sum(int(catalogs[c].population) for c in catalogs)
            turns.append(_turn_record(next_step, farmers, population, terminated, nutrition))
            _track(next_step, terminated, nutrition)
            step = next_step
    return {
        "world": world,
        "mode": job["mode"],
        "reset_identity": identity,
        "turns": turns,
        "initial_rejections": rejected,
        "retries": retries,
    }


H12_FALLBACK_HEADS = (
    "specialty",
    "activity_role",
    "movement",
    "fission",
    "trade_gate",
    "over_budget",
    "full_action",
)


def _run_h12(job: Mapping[str, Any]) -> dict[str, Any]:
    """Heuristic recommendation converted to action 12, every eligible camp.

    Mirrors ``labor_bc.collect_heuristic_world``: capture the labor bridge
    snapshot, step a shadow heuristic bridge from it, convert each camp's
    decision with ``convert_heuristic_labor_label`` (whose ``action`` already
    carries the collector's per-head fallbacks), submit it with
    ``_retained_action`` and retry rejections with the all-idle ``_idle_retry``.
    """
    from collections import Counter

    from .bridge import Model9Bridge
    from .labor_bc import (
        LaborResetOptions,
        _idle_retry,
        _retained_action,
        convert_heuristic_labor_label,
    )
    from .labor_bridge import SPECIALTIES, LaborModel9Bridge
    from .labor_model import LaborPersonality, labor_policy_tensors
    from .labor_personality import personality_for_world
    from .labor_ppo import execute_labor_decision

    torch.set_num_threads(1)
    _, _, _, _, normalizer = _load_models(Path(job["checkpoint"]))
    world = int(job["world"])
    personality = personality_for_world(LaborPersonality(), world)
    options = LaborResetOptions.from_mapping(job["reset_options"])
    farmer = SPECIALTIES.index(FARMER)
    turns: list[dict[str, Any]] = []
    rejected = retries = decisions = activity_roles = 0
    fallback: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    with LaborModel9Bridge() as bridge, Model9Bridge(economic_diagnostics=True) as heur:
        step = bridge.reset(world, **options.reset_kwargs())
        identity = bridge.last_reset_identity
        terminated: set[int] = set()
        nutrition: dict[int, float] = {}
        _track(step, terminated, nutrition)
        for _ in range(int(job["turns"])):
            live = _live(step)
            if not live:
                break
            contexts, tensors = labor_policy_tensors(live, normalizer, personality=personality)
            catalogs = {context[0].camp_id: context[0] for context in contexts}
            snap = bridge._bridge.capture()
            heur.reset(world, movement=options.movement, withheld=options.withheld, snapshot=snap)
            h_step = heur.step({})
            h_villages = {int(v["camp"]): v for v in h_step.get("villages", [])}
            econ = h_step.get("economicDiagnostics") or {}
            retained = {}
            for row, (village, context) in enumerate(zip(live, contexts, strict=True)):
                camp = int(village["camp"])
                catalog = context[0]
                label = convert_heuristic_labor_label(
                    catalog,
                    h_villages.get(camp, {}),
                    econ.get(str(camp)),
                    pre_observation=village["observation"],
                )
                decisions += 1
                for reason in label.reasons:
                    reasons[reason.split(":", 1)[0]] += 1
                fallback["specialty"] += int(not label.specialty_valid)
                # Only staffed roles carry an activity label; unstaffed roles are
                # role-idle by construction, not a fallback.
                roles = np.asarray(label.activity_role_valid, dtype=bool).reshape(-1)
                staffed = np.asarray(label.specialty_counts).reshape(-1)[1 : 1 + roles.size] > 0
                activity_roles += int(staffed.sum())
                fallback["activity_role"] += int((staffed & ~roles).sum())
                fallback["over_budget"] += int(bool(label.over_budget))
                fallback["movement"] += int(not label.movement_valid)
                fallback["fission"] += int(not label.fission_valid)
                gates = np.asarray(label.trade_gate_valid, dtype=bool).reshape(-1)
                fallback["trade_gate"] += int(not bool(np.all(gates)))
                fallback["full_action"] += int(not label.full_action_valid)
                retained[camp] = _retained_action(
                    label, catalog, tensors["movement_mask"][row : row + 1]
                )
            execution = execute_labor_decision(bridge, catalogs, retained, _idle_retry)
            next_step = execution.step
            rejected += sum(1 for a in execution.initial_attempts.values() if not a.valid)
            retries += len(execution.retry)
            farmers = sum(
                int(a.accepted_specialty_residents[farmer])
                for a in execution.final_attempts.values()
            )
            population = sum(int(catalogs[c].population) for c in catalogs)
            turns.append(_turn_record(next_step, farmers, population, terminated, nutrition))
            _track(next_step, terminated, nutrition)
            step = next_step
    return {
        "world": world,
        "mode": "h12",
        "reset_identity": identity,
        "turns": turns,
        "initial_rejections": rejected,
        "retries": retries,
        "h12_fallbacks": {
            "decisions": decisions,
            "activity_roles": activity_roles,
            "counts": {k: int(fallback[k]) for k in H12_FALLBACK_HEADS},
            "reasons": dict(sorted(reasons.items())),
        },
    }


def _run_job(job: Mapping[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    if job["mode"] == "heuristic":
        result = _run_heuristic(job)
    elif job["mode"] == "h12":
        result = _run_h12(job)
    else:
        result = _run_actor(job)
    result["wall_seconds"] = time.perf_counter() - started
    print(
        json.dumps(
            {
                "world": result["world"],
                "mode": result["mode"],
                "wall_seconds": round(result["wall_seconds"], 1),
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    return result


def _totals(run: Mapping[str, Any]) -> dict[str, Any]:
    turns = run["turns"]
    welfare = sum(t["welfare"] for t in turns)
    person_years = sum(t["person_years"] for t in turns)
    return {
        "welfare": welfare,
        "person_years": person_years,
        "qol": welfare / person_years if person_years > 0 else 0.0,
        "starvation_extinctions": sum(len(t["starvation_extinctions"]) for t in turns),
        "terminations": sum(len(t["terminations"]) for t in turns),
        "turns": len(turns),
        "final_residents": turns[-1]["residents"] if turns else 0,
        "per_turn": {
            key: [t[key] for t in turns]
            for key in ("farmer_share", "housing_coverage", "clothing_coverage", "qol", "residents")
        },
    }


def _ratio(a: float, b: float) -> float | None:
    return a / b if b > 0 else None


def autonomous_report(results: Sequence[Mapping[str, Any]], modes: Sequence[str]) -> dict[str, Any]:
    by = {(int(r["world"]), r["mode"]): r for r in results}
    worlds = sorted({int(r["world"]) for r in results})
    per_world: dict[str, Any] = {}
    aggregate: dict[str, Any] = {}
    for world in worlds:
        heuristic = _totals(by[(world, "heuristic")])
        entry: dict[str, Any] = {"heuristic": heuristic}
        for mode in modes:
            run = by[(world, mode)]
            actor = _totals(run)
            actor["initial_rejections"] = run.get("initial_rejections", 0)
            actor["same_reset"] = (
                run["reset_identity"] == by[(world, "heuristic")]["reset_identity"]
            )
            actor["ratios"] = {
                m: _ratio(actor[m], heuristic[m]) for m in ("welfare", "person_years", "qol")
            }
            actor["additional_starvation_extinctions"] = (
                actor["starvation_extinctions"] - heuristic["starvation_extinctions"]
            )
            entry[mode] = actor
        per_world[str(world)] = entry
    heur_sum = {
        m: sum(per_world[str(w)]["heuristic"][m] for w in worlds)
        for m in ("welfare", "person_years", "starvation_extinctions")
    }
    for mode in modes:
        sums = {
            m: sum(per_world[str(w)][mode][m] for w in worlds)
            for m in ("welfare", "person_years", "starvation_extinctions")
        }
        ratios = {m: _ratio(sums[m], heur_sum[m]) for m in ("welfare", "person_years")}
        ratios["qol"] = _ratio(
            sums["welfare"] / sums["person_years"] if sums["person_years"] > 0 else 0.0,
            heur_sum["welfare"] / heur_sum["person_years"] if heur_sum["person_years"] > 0 else 0.0,
        )
        checks = {}
        for metric in GATED_AUTONOMOUS_METRICS:
            world_ratios = [per_world[str(w)][mode]["ratios"][metric] for w in worlds]
            checks[f"{metric}_aggregate"] = (
                ratios[metric] is not None
                and ratios[metric] >= AUTONOMOUS_GATES["aggregate_ratio_min"]
            )
            checks[f"{metric}_per_world"] = all(
                r is not None and r >= AUTONOMOUS_GATES["per_world_ratio_min"] for r in world_ratios
            )
            checks[f"{metric}_min_world_ratio"] = min(
                (r for r in world_ratios if r is not None), default=None
            )
        checks["no_additional_starvation_extinction"] = all(
            per_world[str(w)][mode]["additional_starvation_extinctions"] <= 0 for w in worlds
        )
        checks["same_reset_all_worlds"] = all(per_world[str(w)][mode]["same_reset"] for w in worlds)
        passed = all(v for k, v in checks.items() if not k.endswith("_min_world_ratio"))
        aggregate[mode] = {
            "actor": sums,
            "heuristic": heur_sum,
            "ratios": ratios,
            "checks": checks,
            "gate": "PASS" if passed else "FAIL",
        }
    report: dict[str, Any] = {"per_world": per_world, "aggregate": aggregate}
    if "h12" in modes:
        report.update(_h12_extras(by, worlds, modes, per_world))
    return report


def _h12_extras(
    by: Mapping[tuple[int, str], Mapping[str, Any]],
    worlds: Sequence[int],
    modes: Sequence[str],
    per_world: Mapping[str, Any],
) -> dict[str, Any]:
    """Fallback rates of the converted heuristic and actor/h12 ratios."""
    counts: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    decisions = roles = 0
    for world in worlds:
        fb = by[(world, "h12")].get("h12_fallbacks") or {}
        decisions += int(fb.get("decisions", 0))
        roles += int(fb.get("activity_roles", 0))
        for k, v in (fb.get("counts") or {}).items():
            counts[k] += int(v)
        for k, v in (fb.get("reasons") or {}).items():
            reasons[k] += int(v)
    rates = {}
    for k, v in counts.items():
        denominator = roles if k == "activity_role" else decisions
        rates[k] = v / denominator if denominator > 0 else None
    metrics = ("welfare", "person_years")
    h12_sum = {m: sum(per_world[str(w)]["h12"][m] for w in worlds) for m in metrics}
    vs_h12: dict[str, Any] = {}
    for mode in modes:
        if mode == "h12":
            continue
        sums = {m: sum(per_world[str(w)][mode][m] for w in worlds) for m in metrics}
        ratios = {m: _ratio(sums[m], h12_sum[m]) for m in metrics}
        ratios["qol"] = _ratio(
            sums["welfare"] / sums["person_years"] if sums["person_years"] > 0 else 0.0,
            h12_sum["welfare"] / h12_sum["person_years"] if h12_sum["person_years"] > 0 else 0.0,
        )
        vs_h12[mode] = {
            "ratios": ratios,
            "per_world": {
                str(w): {
                    m: _ratio(per_world[str(w)][mode][m], per_world[str(w)]["h12"][m])
                    for m in (*metrics, "qol")
                }
                for w in worlds
            },
        }
    return {
        "h12_fallbacks": {
            "decisions": decisions,
            "activity_roles": roles,
            "counts": dict(counts),
            "rates": rates,
            "reasons": dict(sorted(reasons.items())),
            "note": "rates per camp decision (activity_role per staffed role); the fallbacks "
            "are those built into labor_bc.convert_heuristic_labor_label, as in the BC collector",
        },
        "actor_vs_h12": vs_h12,
    }


def run_autonomous(args: argparse.Namespace) -> dict[str, Any]:
    from .labor_bc import DEVELOPMENT_WORLDS, LaborResetOptions
    from .protocol import seed_list

    worlds = seed_list(args.worlds)
    _check_world_policy(worlds, "autonomous worlds")
    outside = [w for w in worlds if w not in DEVELOPMENT_WORLDS]
    if outside and not args.allow_non_development:
        raise ValueError(
            f"autonomous evaluation uses development worlds 2000-2059; got {outside[:5]}"
        )
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes or any(m not in ("deterministic", "stochastic", "h12") for m in modes):
        raise ValueError("--modes must list deterministic, stochastic and/or h12")
    reset = LaborResetOptions(
        players=args.players,
        movement=not args.no_movement,
        width=args.width,
        height=args.height,
        separation=args.separation,
    )
    base = {
        "checkpoint": str(args.checkpoint),
        "turns": int(args.turns),
        "reset_options": reset.state_dict(),
        "policy_seed": int(args.policy_seed),
    }
    jobs = [dict(base, world=w, mode=m) for w in worlds for m in ("heuristic", *modes)]
    started = time.perf_counter()
    if args.jobs <= 1:
        results = [_run_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
            results = list(pool.map(_run_job, jobs))
    report = autonomous_report(results, modes)
    report.update(
        {
            "schema": REPORT_SCHEMA,
            "command": "autonomous",
            "checkpoint": str(args.checkpoint),
            "worlds": worlds,
            "turns": int(args.turns),
            "modes": modes,
            "reset_options": reset.state_dict(),
            "policy_seed": int(args.policy_seed),
            "gates": dict(AUTONOMOUS_GATES, gated_metrics=list(GATED_AUTONOMOUS_METRICS)),
            "definitions": {
                "welfare": "sum over settlements and turns of native village welfare (prior residents x QoL x years)",
                "person_years": "sum over living settlements of prior residents x years",
                "qol": "welfare / person_years",
                "starvation_extinction": f"newly terminated settlement whose terminal or preceding nutrition < {STARVATION_NUTRITION}",
                "farmer_share_actor": "accepted Farmer residents (native apportionment) / acting residents",
                "farmer_share_heuristic": "ceil(assignedFarmerYears / years) / acting residents (economic diagnostics)",
                "coverage": "resident-weighted housing/clothing coverage of living settlements after the turn",
                "baseline": "native heuristic stepped with empty actions from the actor-labor reset snapshot",
            },
            "wall_seconds": time.perf_counter() - started,
            "runs": results,
        }
    )
    return report


def summarize_autonomous(report: Mapping[str, Any]) -> str:
    lines = [
        f"autonomous worlds={len(report['worlds'])} turns={report['turns']} modes={','.join(report['modes'])}"
    ]
    for mode, agg in report["aggregate"].items():
        r = agg["ratios"]
        lines.append(
            f"  {mode:<13} {agg['gate']} welfare={_fmt(r['welfare'])} person-years={_fmt(r['person_years'])} "
            f"qol={_fmt(r['qol'])} starvation actor/heur={agg['actor']['starvation_extinctions']}/"
            f"{agg['heuristic']['starvation_extinctions']}"
        )
    for world, entry in report["per_world"].items():
        parts = [f"  world {world}:"]
        for mode in report["modes"]:
            rr = entry[mode]["ratios"]
            parts.append(
                f"{mode[:4]} W={_fmt(rr['welfare'])} PY={_fmt(rr['person_years'])} Q={_fmt(rr['qol'])}"
            )
        parts.append(
            "farmer heur/"
            + "/".join(m[:4] for m in report["modes"])
            + "="
            + "/".join(
                _fmt(
                    float(np.mean(entry[k]["per_turn"]["farmer_share"]))
                    if entry[k]["per_turn"]["farmer_share"]
                    else None,
                    2,
                )
                for k in ("heuristic", *report["modes"])
            )
        )
        lines.append(" ".join(parts))
    if "actor_vs_h12" in report:
        for mode, entry in report["actor_vs_h12"].items():
            r = entry["ratios"]
            lines.append(
                f"  {mode}/h12 welfare={_fmt(r['welfare'])} person-years={_fmt(r['person_years'])} "
                f"qol={_fmt(r['qol'])}"
            )
        fb = report["h12_fallbacks"]
        lines.append(
            f"  h12 fallbacks (decisions={fb['decisions']}): "
            + " ".join(f"{k}={_fmt(v)}" for k, v in sorted(fb["rates"].items()))
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=1, sort_keys=True, default=float))
    os.replace(temporary, path)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    actor = commands.add_parser("actor-fidelity")
    actor.add_argument("--checkpoint", type=Path, required=True)
    actor.add_argument("--dataset", type=Path, required=True)
    actor.add_argument("--batch-size", type=int, default=256)
    actor.add_argument("--out", type=Path, required=True)
    critic = commands.add_parser("critic-calibration")
    critic.add_argument("--checkpoint", type=Path, required=True)
    critic.add_argument("--dataset", type=Path, required=True, help="calibration dataset")
    critic.add_argument(
        "--fit-dataset", type=Path, default=None, help="fitting dataset (generalisation gap)"
    )
    critic.add_argument("--batch-size", type=int, default=256)
    critic.add_argument("--out", type=Path, required=True)
    auto = commands.add_parser("autonomous")
    auto.add_argument("--checkpoint", type=Path, required=True)
    auto.add_argument("--worlds", required=True, help="development seeds/ranges, e.g. 2000-2007")
    auto.add_argument("--turns", type=int, default=8)
    auto.add_argument("--modes", default="deterministic,stochastic")
    auto.add_argument("--players", type=int, default=32)
    auto.add_argument("--width", type=int, default=None)
    auto.add_argument("--height", type=int, default=None)
    auto.add_argument("--separation", type=int, default=None)
    auto.add_argument("--no-movement", action="store_true")
    auto.add_argument("--policy-seed", type=int, default=0)
    auto.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    auto.add_argument("--allow-non-development", action="store_true")
    auto.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    torch.set_num_threads(4)
    if args.command == "actor-fidelity":
        report = run_actor_fidelity(args.checkpoint, args.dataset, args.batch_size)
        summary = summarize_actor(report)
    elif args.command == "critic-calibration":
        report = run_critic_calibration(
            args.checkpoint, args.dataset, args.fit_dataset, args.batch_size
        )
        summary = summarize_critic(report)
    else:
        if args.turns < 1 or args.jobs < 1:
            raise ValueError("turns and jobs must be positive")
        report = run_autonomous(args)
        summary = summarize_autonomous(report)
    _write(args.out, report)
    print(summary)
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
