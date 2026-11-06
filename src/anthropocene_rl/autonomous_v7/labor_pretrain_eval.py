"""Evaluation of the interim labor pretrain: actor fidelity, critic calibration, closed loop.

Four subcommands, each writing one JSON report (``--out``) and printing a
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
    QoL gate v2 (28 Oct): ``qol_nstep`` judges the QoL head against the tail-robust
    n-step target (``QOL_GATE_TARGET``) in ``--qol-reference-scale`` units; the Monte Carlo
    metrics stay as non-gating diagnostics. ``--checkpoint`` accepts several critics, scored
    as one averaged critic (per-row mean of the members' value-unit predictions, same
    metrics; ``labor_critic_ensemble``); ``--member-reports`` adds each member's own run.

``autonomous``
    Closed-loop S1 check on development worlds: the actor controls every
    eligible camp (deterministic and/or stochastic) versus the native
    heuristic controlling everything from the same reset snapshot. Mode
    ``h12`` instead submits the converted heuristic recommendation (exactly as
    ``labor_bc.collect_heuristic_world`` builds it) through the same action-12
    labor runtime, separating interface loss (h12 vs heuristic) from actor
    loss (actor vs h12). Two diagnostic modes localise the actor loss:
    ``takeover:K`` lets the deterministic actor decide decisions 0..K-1 and h12
    every later decision of the same episode (``takeover:0`` is h12,
    ``takeover:K`` with K >= turns is deterministic); ``swap:GROUPS`` is h12 with
    the ``+``-joined head groups (labor = specialty+activity, specialty, activity,
    movement, fission, trade) of every action taken from the deterministic actor
    on the same state (``merge_action_payloads``).

``actor-divergence``
    Where the BC actor first leaves its teacher in closed loop. The actor (deterministic
    decode) is the only policy executed; on every actor state H12's converted label is also
    computed (shadow heuristic from a snapshot, never executed) and each head's
    disagreement is recorded per turn: specialty L1 (per-specialty signed difference),
    activity L1 / argmax mismatch per staffed role (per-activity signed difference and
    resident-time shares), movement, fission and trade-gate agreement, with camp context
    (residents, housing/clothing coverage, farmer share). Only valid H12 labels count.
    The report gives per-turn and window (default turns 5-25) pools, the first turn each
    head exceeds 2x its turns 1-5 baseline, and a copy per world.

Gate semantics (critic): PASS when every metric gate and the coverage gates
hold; FAIL when any metric gate fails; HOLD when the metrics pass but the
coverage (worlds, informative groups) is insufficient to decide.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import re
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
    for i, key in enumerate(zip(worlds.tolist(), turns.tolist(), strict=False)):
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


def _load_dataset(
    path: Path | Sequence[Path],
    normalizer: Any,
    *,
    require_privileged: bool = False,
    corrections: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One evaluation dataset, or several batches read as one (``merge_labor_datasets``).

    ``corrections`` (``labor_bc.loader_corrections`` of the evaluated fit) applies the same
    loader corrections the fit trained on; absent or all "none": the data as stored."""
    from .labor_bc import apply_loader_corrections
    from .labor_teacher_data import load_labor_data
    from .labor_training import _normalization_digest

    dataset = load_labor_data(
        path,
        expected_normalizer_digest=_normalization_digest(normalizer),
        require_privileged=require_privileged,
        training=False,  # evaluation reader: development/calibration/held-out-test data
    )
    _check_world_policy(dataset["row_world"].tolist(), f"dataset {dataset_label(path)}")
    if corrections:
        dataset = apply_loader_corrections(dataset, **corrections)
    return dataset


def dataset_label(path: Path | Sequence[Path]) -> str | list[str]:
    """A report's dataset field: the path (one dataset, unchanged) or the list of paths."""
    if isinstance(path, (str, Path)):
        return str(path)
    paths = [str(p) for p in path]
    return paths[0] if len(paths) == 1 else paths


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
        "by_scenario_family": actor_fidelity_by_family(arrays, dataset, normalizer),
    }
    return report


def actor_fidelity_by_family(
    arrays: Mapping[str, np.ndarray], dataset: Mapping[str, Any], normalizer: Any
) -> dict[str, Any]:
    """Report-only actor fidelity per curated scenario family (labor_scenarios rules)."""
    from .labor_scenarios import FAMILY_BITS, row_family_tags

    tags = row_family_tags(dataset, normalizer=normalizer)
    if tags is None:
        return {}
    return {
        family: actor_fidelity_metrics(arrays, (tags & bit) != 0)
        for family, bit in FAMILY_BITS.items()
        if ((tags & bit) != 0).any()
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def summarize_actor(report: Mapping[str, Any]) -> str:
    o = report["overall"]
    lines = [
        f"actor-fidelity rows={o['rows']} admitted={o['admitted_rows']} worlds={o['worlds']}",
        f"  specialty L1={_fmt(o['specialty_l1'])} activity L1={_fmt(o['activity_l1'])} "
        f"resident |residual|/camp={_fmt(o['resident_abs_residual_per_camp'], 2)}",
        f"  movement acc={_fmt(o['movement']['accuracy'])} (n={o['movement']['rows']}) "
        f"fission acc={_fmt(o['fission']['accuracy'])} trade-gate "
        f"acc={_fmt(o['trade_gate']['accuracy'])}",
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
                f"  [{name}] rows={s['rows']} spec L1={_fmt(s['specialty_l1'])} act "
                f"L1={_fmt(s['activity_l1'])} "
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
    head) also per-row value-support tail terms keyed ``"<head>/<term>"``.

    A critic ensemble (:class:`labor_critic_ensemble.EnsembleTask`) is the per-row mean of
    its members' predictions in value units."""
    from .labor_critic_ensemble import EnsembleTask, ensemble_critic_predictions

    if isinstance(task, EnsembleTask):
        return ensemble_critic_predictions(task, costs, dataset, rows, batch_size, observed_max)
    from .labor_device import module_device, to_device
    from .labor_model import LABOR_COSTS, LABOR_OBJECTIVES, critic_privileged_batch

    device = module_device(task)
    tensors = _input_tensors(dataset)
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for index, batch in _batches(tensors, rows, batch_size):
            batch = to_device(batch, device)
            # Critic-only privileged batch, separate from the observation batch.
            privileged = critic_privileged_batch(task, dataset, index, device=device)
            logits = task(batch, privileged)
            for name in LABOR_OBJECTIVES:
                out[name].append(task.expectation(name, logits[name]).double().cpu().numpy())
                if observed_max is not None and not task.is_bounded(name):
                    terms = task.support_diagnostics(name, logits[name], observed_max[name])
                    for term, value in terms.items():
                        out[f"{name}/{term}"].append(value.cpu().numpy())
            cost_priv = critic_privileged_batch(costs, dataset, index, device=device)
            for name, value in costs(batch, cost_priv).items():
                out[name].append(value.double().cpu().numpy())
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
            head[term] = (
                None if values is None or not len(values) else weighted_mean(values, weights)
            )
        excess = preds.get(f"{name}/upper_excess")
        head["upper_excess_max"] = (
            None if excess is None or not len(excess) else float(excess.max())
        )
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


# Advisory tail-weight cut of the QoL bias diagnostic (user decision 27 Oct, non-gating).
QOL_TAIL_WEIGHT_CUT = 0.5


def tail_schedule_factor(record: Mapping[str, Any] | None, weight: float) -> float | None:
    """World-level roulette factor of a frozen stop schedule: what a unit-QoL continuation
    would be estimated at, in units of W.

    The tail estimate is the mean over the schedule's draws of
    sum_{k <= stop} dt gamma^k r_k / rho^k, so a constant QoL of 1 gets
    dt * mean_d sum_{k <= stop_d} (gamma / rho)^k. Divided by W it is 1 in expectation;
    one schedule is shared by every camp of the world, so its realized value scales every
    camp's tail alike. ``None`` without ``stop_indices``.
    """
    from . import contracts

    if not isinstance(record, Mapping) or not record.get("stop_indices"):
        return None
    rho = float(record.get("rho", 0.95))
    ratio = math.exp(-contracts.TURN_YEARS / contracts.HORIZON_YEARS) / rho
    sums = [sum(ratio**k for k in range(int(stop) + 1)) for stop in record["stop_indices"]]
    return contracts.TURN_YEARS * (sum(sums) / len(sums)) / float(weight)


def qol_tail_diagnostics(
    dataset: Mapping[str, Any],
    rows: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    scale: float,
    weight: float,
) -> dict[str, Any]:
    """Per-world tail factor and tail exposure of the QoL calibration targets (advisory).

    Per world: the stop schedule's factor (:func:`tail_schedule_factor`), the share of
    rows whose tail exceeds W, the largest tail / W and the tail's weight in the
    (Monte Carlo) targets, c_i tail_i / G_i (``labor_advantage.qol_tail_weight``). The QoL
    bias (world-balanced, reference-scale units, as the gate) on all rows and on rows
    whose tail weight is at most 0.5. Never gating.
    """
    from .labor_advantage import qol_tail_terms

    rows = np.asarray(rows, dtype=np.int64)
    terms = qol_tail_terms(dataset, 1.0)
    coefficient = terms["coefficient"].numpy()[rows]
    tail = terms["tail"].numpy()[rows]
    has_tail = terms["has_tail"].numpy()[rows]
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    part = coefficient * tail
    tail_weight = np.where(
        has_tail & (target > 0), np.clip(part / np.where(target > 0, target, 1.0), 0.0, 1.0), 0.0
    )
    worlds = _np(dataset, "row_world").reshape(-1)[rows]
    records = (dataset.get("collection") or {}).get("return_tail_by_world") or {}
    per_world: dict[str, Any] = {}
    for world in np.unique(worlds):
        m = worlds == world
        t = tail[m & has_tail]
        per_world[str(int(world))] = {
            "rows": int(m.sum()),
            "schedule_factor": tail_schedule_factor(records.get(str(int(world))), weight),
            "tail_rows": int((m & has_tail).sum()),
            "tail_above_w_share": float((t > weight).sum() / m.sum()),
            "max_tail_over_w": float(t.max() / weight) if t.size else None,
            "tail_weight_mean": float(tail_weight[m].mean()),
            "tail_weight_max": float(tail_weight[m].max()),
            "tail_weight_above_cut_share": float((tail_weight[m] > QOL_TAIL_WEIGHT_CUT).mean()),
        }
    low = tail_weight <= QOL_TAIL_WEIGHT_CUT
    everything = np.ones(len(rows), dtype=bool)
    bias_all = normalized_bias(pred, target, scale, world_balanced(worlds, everything))
    bias_low = (
        normalized_bias(pred[low], target[low], scale, world_balanced(worlds[low], low[low]))
        if low.any()
        else None
    )
    factors = [w["schedule_factor"] for w in per_world.values() if w["schedule_factor"] is not None]
    return {
        "gating": False,
        "weight": float(weight),
        "tail_weight_cut": QOL_TAIL_WEIGHT_CUT,
        "tail_weight_definition": "c_i tail_i / G_i of the Monte Carlo target (lambda = 1)",
        "worlds": per_world,
        "schedule_factor_range": [min(factors), max(factors)] if factors else None,
        "tail_above_w_share": float(((tail > weight) & has_tail).mean()) if len(rows) else 0.0,
        "bias": {
            "units": "QoL reference scales (as the qol:b gate)",
            "all": bias_all,
            "tail_weight_le_cut": bias_low,
            "rows_all": len(rows),
            "rows_tail_weight_le_cut": int(low.sum()),
        },
    }


# --- QoL calibration gate v2: the tail-robust n-step target (user decision 28 Oct) ----------

# The window length of the gate's target; the same k as the acceptance harness's n-step TD
# residual (``labor_r5_acceptance`` PREREGISTERED["td_nstep_k"], checked equal in tests).
QOL_NSTEP_K = 5
# Closure of each row's n-step window (``nstep_targets``).
NSTEP_FULL = 0  # k observed rewards, then gamma^k V(s_{t+k}) (the critic's own bootstrap)
NSTEP_TERMINAL = 1  # the trajectory terminated inside the window: realized rewards, zero tail
NSTEP_LIVING_END_SHORT = 2  # the window reaches the collection window's living end (< k rewards)
NSTEP_LIVING_END_SELF = 3  # the row is the living end itself: no reward precedes a bootstrap
NSTEP_CLOSURES = {
    NSTEP_FULL: "full-window-bootstrap",
    NSTEP_TERMINAL: "terminal-inside-window",
    NSTEP_LIVING_END_SHORT: "living-end-shortened",
    NSTEP_LIVING_END_SELF: "living-end-row",
}
NSTEP_GATED_CLOSURES = (NSTEP_FULL, NSTEP_TERMINAL)
QOL_GATE_TARGET_ID = "nstep-k5-bootstrap-v1"
QOL_GATE_TARGET = {
    "id": QOL_GATE_TARGET_ID,
    "k": QOL_NSTEP_K,
    "definition": "y_t = sum_{j<k} D_j r_{t+j} + D_k V(s_{t+k}), D_j = prod_{m<j} "
    "exp(-elapsed_{t+m} / 400 y), V the evaluated critic's own decoded QoL value; "
    "trajectories keyed (world, camp, row_trajectory), ordered by row_turn",
    "terminal": "a window reaching a terminal row ends there with realized rewards and a zero "
    "tail (exact; gated)",
    "living_end": "a window that would pass the collection window's last row is NOT gated: "
    "no Monte Carlo tail ever enters the gate. The k rows before the living end are "
    "reported (non-gating) closed both by the critic's bootstrap at the last row (rewards "
    "t..last-1 + D V(s_last), fewer than k rewards) and by the dataset's declared tail "
    "(rewards t..last + D tail_last, as the TD residual)",
    "rows": "admitted rows (not quarantined, target available), as the Monte Carlo metrics; "
    "bootstrap values may come from any row of the trajectory",
    "aggregation": "world-balanced per stratum, strata as the Monte Carlo metrics",
    "units": "QoL reference scale of the gate (qol_reference_scale)",
    "monte_carlo": "the randomized-tail Monte Carlo metrics stay in the report, non-gating",
}
# The bias half of the QoL gate (coordinator amendment, 28 Oct): against the n-step target a
# uniform critic offset c shows only as (1 - D_k) c ~ 0.22 c, so |bias| is judged instead
# against the Monte Carlo return on rows whose tail weight in that target is at most the cut.
QOL_BIAS_TAIL_CUT = 0.5
QOL_BIAS_MIN_ROWS = 200  # a stratum with fewer cut rows, or fewer worlds, HOLDs
QOL_BIAS_MIN_WORLDS = 3
QOL_BIAS_TARGET_ID = "mc-tailcut-0.5-v1"
QOL_BIAS_TARGET = {
    "id": QOL_BIAS_TARGET_ID,
    "definition": "world-balanced mean of (V - G) / scale over admitted rows whose tail weight "
    "c_i tail_i / G_i in the Monte Carlo target G (labor_advantage.qol_tail_weight, "
    "lambda = 1) is <= 0.5; rows without a declared tail have weight 0",
    "tail_weight_cut": QOL_BIAS_TAIL_CUT,
    "coverage": f"a stratum with fewer than {QOL_BIAS_MIN_ROWS} such rows or fewer than "
    f"{QOL_BIAS_MIN_WORLDS} worlds HOLDs",
    "units": "QoL reference scale of the gate (qol_reference_scale)",
    "nstep_bias": "the n-step bias stays in each stratum as a non-gating diagnostic",
}
# Non-gating tail attribution of the bias check (user decision 28 Oct): flag when the
# predicted tail-induced bias exceeds this, or when removing it would flip the verdict.
QOL_BIAS_TAIL_ATTRIBUTION_FLAG = 0.05
QOL_BIAS_TAIL_ATTRIBUTION = {
    "gating": False,
    "factor": "per-world roulette schedule factor (tail_schedule_factor, the qol_tail "
    "convention); a world without a schedule record counts as factor 1",
    "predicted_tail_bias": "world-balanced mean over the tail-cut rows of "
    "-tail_weight_i (1 - 1 / f_w) G_i / scale, in reference-scale units with the bias sign "
    "(pred - target): the bias an exact critic shows when the stored tail is f_w x its "
    "expectation; first-order in (f_w - 1) it is the -tail_weight (f_w - 1) G / scale form, "
    "also reported as predicted_tail_bias_first_order",
    "residual_critic_bias": "observed tail-cut bias - predicted_tail_bias",
    "flag": "|predicted_tail_bias| > 0.05, or the |bias| <= 0.10 verdict differs between the "
    "observed and the residual bias",
}
# Evidence field ``qol_gate_target``: one definition per gated metric.
QOL_GATE_TARGETS = {"nmae": QOL_GATE_TARGET, "bias": QOL_BIAS_TARGET}
FROZEN_QOL_REFERENCE_SCALE = 43.152  # contracts.OBJECTIVE_REFERENCE_SCALES["qol_years"]
QOL_REFERENCE_SCALE_SCHEMA = "labor-r5-qol-reference-scale-v1"


def nstep_targets(
    values: Any,
    reward: Any,
    elapsed: Any,
    terminal: Any,
    turns: Any,
    worlds: Any,
    camps: Any,
    trajectories: Any,
    tails: Any = None,
    tail_ok: Any = None,
    *,
    k: int = QOL_NSTEP_K,
    discount_years: float = 400.0,
) -> dict[str, np.ndarray]:
    """Per-row n-step targets along each trajectory (pure; units of ``values``/``reward``).

    ``target`` closes each window with the critic's own bootstrap: k observed rewards then
    D_k V(s_{t+k}) (``NSTEP_FULL``); a terminal row inside the window ends it with a zero
    tail (``NSTEP_TERMINAL``); a window reaching the living end (the trajectory's last,
    non-terminal row) bootstraps from V(s_last) after the rewards before it
    (``NSTEP_LIVING_END_SHORT``, fewer than k rewards); the last row itself has no target
    (``NSTEP_LIVING_END_SELF``, NaN). ``tail_target`` closes the living-end windows instead
    with the stored tail at the last row (rewards through the last row + D tail), where
    ``tail_ok`` says the tail is declared (NaN elsewhere); for full and terminal windows it
    equals ``target``. ``steps`` is the number of rewards in ``target``. Discounts per row
    are exp(-elapsed / discount_years), applied after the row's reward, exactly as
    ``labor_bc.discounted_returns`` and the acceptance TD residual.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    reward = np.asarray(reward, dtype=np.float64).reshape(-1)
    elapsed = np.asarray(elapsed, dtype=np.float64).reshape(-1)
    terminal = np.asarray(terminal).reshape(-1).astype(bool)
    n = len(values)
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i in range(n):
        groups[(int(worlds[i]), int(camps[i]), int(trajectories[i]))].append(i)
    target = np.full(n, np.nan)
    tail_target = np.full(n, np.nan)
    closure = np.zeros(n, dtype=np.int64)
    steps = np.zeros(n, dtype=np.int64)
    position = np.zeros(n, dtype=np.int64)
    remaining = np.zeros(n, dtype=np.int64)
    gamma = np.exp(-elapsed / float(discount_years))
    for indices in groups.values():
        indices.sort(key=lambda i: turns[i])
        size = len(indices)
        for pos, i in enumerate(indices):
            position[i] = pos
            remaining[i] = size - 1 - pos
            total, discount = 0.0, 1.0
            code = NSTEP_FULL
            for j in range(k):
                row = indices[pos + j]
                if pos + j == size - 1 and not terminal[row]:
                    # Living end: bootstrap from V(s_last) after the rewards before it.
                    code = NSTEP_LIVING_END_SHORT if j > 0 else NSTEP_LIVING_END_SELF
                    if j > 0:
                        target[i] = total + discount * values[row]
                        steps[i] = j
                    if tails is not None and (tail_ok is None or bool(tail_ok[row])):
                        tail_target[i] = (
                            total
                            + discount * reward[row]
                            + discount * gamma[row] * float(tails[row])
                        )
                    break
                total += discount * reward[row]
                discount *= gamma[row]
                if terminal[row]:
                    code = NSTEP_TERMINAL
                    target[i] = total
                    tail_target[i] = total
                    steps[i] = j + 1
                    break
            else:
                target[i] = total + discount * values[indices[pos + k]]
                tail_target[i] = target[i]
                steps[i] = k
            closure[i] = code
    return {
        "target": target,
        "tail_target": tail_target,
        "closure": closure,
        "steps": steps,
        "position": position,
        "remaining": remaining,
    }


def frozen_qol_reference() -> dict[str, Any]:
    """The G0-era QoL reference scale: old receipts and runs without ``--qol-reference-scale``."""
    return {
        "value": FROZEN_QOL_REFERENCE_SCALE,
        "source": "frozen-g0",
        "detail": "contracts.OBJECTIVE_REFERENCE_SCALES['qol_years']: std of 25-turn truncated "
        "heuristic QoL returns on Small worlds (G0)",
        "digest": None,
        "path": None,
        "sha256": None,
    }


def _scale_digest(record: Mapping[str, Any]) -> str:
    import hashlib

    body = {key: value for key, value in record.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def load_qol_reference_scale(path: Path | str | None) -> dict[str, Any]:
    """The gate's QoL reference scale record: a verified artifact, or the frozen 43.152."""
    import hashlib

    if path is None:
        return frozen_qol_reference()
    raw = Path(path).read_bytes()
    record = json.loads(raw.decode("utf-8"))
    if record.get("schema") != QOL_REFERENCE_SCALE_SCHEMA:
        raise ValueError(f"{path}: not a QoL reference-scale artifact ({record.get('schema')!r})")
    if record.get("digest") != _scale_digest(record):
        raise ValueError(f"{path}: QoL reference-scale digest does not match its content")
    value = float(record["value"])
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{path}: QoL reference scale must be finite and positive, got {value}")
    return {
        "value": value,
        "source": "artifact",
        "detail": record.get("quantity"),
        "digest": record["digest"],
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "version": record.get("version"),
        "fit_worlds": len(record.get("worlds") or []),
    }


def qol_reference_scale_record(
    dataset: Mapping[str, Any], *, datasets: Sequence[Mapping[str, Any]] = (), version: int = 1
) -> dict[str, Any]:
    """Remeasure the QoL reference scale on (fitting) data: the std of the per-row QoL return.

    The quantity is the one the G0 scale measured, the per-row discounted QoL return, here
    the r5 Monte Carlo return closed by the dataset's declared tail
    (``labor_bc.discounted_returns``) on admitted rows (not quarantined, target available);
    ``value`` is its unweighted population std (numpy ``std``, ddof 0), as
    ``scale_reference``. The Monte Carlo return is chosen over the n-step target because
    it needs no critic: the scale is fixed before any fit, identical for every arm and
    seed, and cannot move with the model it judges. Its spread includes the world-level
    roulette factor, so ``diagnostics`` also reports the world-balanced std, the std of
    the observed (tail-free) part and the tail share; they never replace ``value``.
    """
    from .labor_bc import (
        RETURN_TAIL_RANDOMIZED,
        critic_target_available,
        discounted_returns,
        return_tail_method,
    )

    all_returns = discounted_returns(dataset)
    returns = all_returns["qol_years"].double().numpy()
    quarantined = _np(dataset, "row_quarantined").reshape(-1).astype(bool)
    available = critic_target_available(dataset).numpy().astype(bool)
    keep = ~quarantined & available
    if not keep.any():
        raise ValueError("no admitted rows to measure the QoL reference scale on")
    g = returns[keep]
    worlds = _np(dataset, "row_world").reshape(-1)
    weights = world_balanced(worlds[keep], np.ones(int(keep.sum()), dtype=bool))
    mean_w = float((g * weights).sum() / weights.sum())
    diagnostics: dict[str, Any] = {
        "world_balanced_std": float(np.sqrt((weights * (g - mean_w) ** 2).sum() / weights.sum())),
        "world_balanced_mean": mean_w,
    }
    method = return_tail_method(dataset)
    if method == RETURN_TAIL_RANDOMIZED:
        from .labor_advantage import qol_tail_terms

        terms = qol_tail_terms(dataset, 1.0)
        part = (terms["coefficient"].numpy() * terms["tail"].numpy())[keep]
        part = np.where(terms["has_tail"].numpy()[keep], part, 0.0)
        observed = g - part
        diagnostics.update(
            observed_part_std=float(np.std(observed)),
            tail_part_mean=float(np.mean(part)),
            tail_share_of_mean=float(np.mean(part) / np.mean(g)) if np.mean(g) else None,
        )
    record: dict[str, Any] = {
        "schema": QOL_REFERENCE_SCALE_SCHEMA,
        "version": int(version),
        "head": "qol_years",
        "quantity": "std (ddof 0, unweighted rows) of the per-row discounted QoL return "
        "(tau 400 y) closed by the dataset's declared tail, admitted fitting rows",
        "value": float(np.std(g)),
        "mean": float(np.mean(g)),
        "rows": int(keep.sum()),
        "worlds": sorted({int(w) for w in worlds[keep].tolist()}),
        "return_tail": method,
        "previous": {"value": FROZEN_QOL_REFERENCE_SCALE, "source": "frozen-g0"},
        # Informational only (28 Oct): the r5 equivalent of the frozen population scale;
        # it is never used as a unit.
        "population_years": {
            "informational": True,
            "value": float(np.std(all_returns["population_years"].double().numpy()[keep])),
            "mean": float(np.mean(all_returns["population_years"].double().numpy()[keep])),
            "frozen": POPULATION_REFERENCE_SCALE_PROVENANCE["value"],
            "frozen_provenance": POPULATION_REFERENCE_SCALE_PROVENANCE,
        },
        "diagnostics": diagnostics,
        "datasets": list(datasets),
    }
    return record


def write_qol_reference_scale(record: Mapping[str, Any], out: Path | str) -> dict[str, Any]:
    """Seal (digest over the content without ``digest``) and write the artifact atomically."""
    value = float(record["value"])
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"QoL reference scale must be finite and positive, got {value}")
    sealed = dict(record)
    sealed.pop("digest", None)
    sealed["digest"] = _scale_digest(sealed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(sealed, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(temporary, out)
    return sealed


def _nstep_metrics(
    pred: np.ndarray, target: np.ndarray, worlds: np.ndarray, scale: float
) -> dict[str, Any]:
    weights = world_balanced(worlds, np.ones(len(worlds), dtype=bool))
    return {
        "rows": len(pred),
        "worlds": len(np.unique(worlds)),
        "nmae": normalized_mae(pred, target, scale, weights) if len(pred) else None,
        "bias": normalized_bias(pred, target, scale, weights) if len(pred) else None,
        "target_mean": weighted_mean(target, weights) if len(pred) else None,
        "pred_mean": weighted_mean(pred, weights) if len(pred) else None,
    }


def qol_nstep_calibration(
    dataset: Mapping[str, Any],
    rows: np.ndarray,
    values: np.ndarray,
    monte_carlo: np.ndarray,
    strata: Mapping[str, np.ndarray],
    reference: Mapping[str, Any],
    *,
    k: int = QOL_NSTEP_K,
    world_factors: Mapping[int, float | None] | None = None,
    weight: float | None = None,
    head: str = "qol_years",
    constant: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The QoL calibration gate's metrics against the n-step target (``QOL_GATE_TARGET``).

    ``constant`` ({"value", "source"}): adds ``constant_reference``, the same per-stratum
    n-step nMAE/bias and Monte Carlo nMAE/bias of a predictor that outputs that constant on
    every row (bootstraps included), for reading the critic's numbers against a floor.

    ``head`` selects the reward and tail arrays (``reward_<head>``, ``tail_value__<head>``);
    other heads use it through :func:`population_nstep_diagnostic` (non-gating).

    ``values``: the critic's decoded QoL value on EVERY dataset row (bootstraps may sit on
    any row). ``rows``: the admitted rows; ``strata`` masks over ``rows``;
    ``monte_carlo``: the Monte Carlo return on ``rows``. Per stratum: gated nMAE/bias
    (world-balanced, in the gate's reference scale) on admitted rows whose window is full
    or terminal, and the non-gating Monte Carlo metrics on all of the stratum's rows in
    the same unit. ``living_end`` reports the excluded rows near the collection window end.
    ``bias_tailcut`` per stratum is the gated bias (``QOL_BIAS_TARGET``): Monte Carlo bias
    on rows whose tail weight is <= ``QOL_BIAS_TAIL_CUT``, with its coverage status; the
    stratum's n-step ``bias`` is then a non-gating diagnostic. ``bias_tailcut.attribution``
    (non-gating, ``QOL_BIAS_TAIL_ATTRIBUTION``) splits that bias into the part predicted by
    the worlds' schedule factors (``world_factors``; default from the dataset's
    ``return_tail_by_world`` records and the discount weight ``weight``) and the residual.
    """
    from .labor_bc import (
        CRITIC_DISCOUNT_YEARS,
        RETURN_TAIL_RANDOMIZED,
        TAIL_METHOD_FIELD,
        TAIL_RANDOMIZED,
        TAIL_TERMINAL_EXACT,
        TAIL_VALUE_PREFIX,
        TRAJECTORY_FIELD,
        elapsed_years,
        return_tail_method,
    )

    rows = np.asarray(rows, dtype=np.int64)
    n = int(dataset["row_world"].shape[0])
    worlds_all = _np(dataset, "row_world").reshape(-1)
    randomized = return_tail_method(dataset) == RETURN_TAIL_RANDOMIZED
    codes = _np(dataset, TAIL_METHOD_FIELD).reshape(-1) if randomized else None
    result = nstep_targets(
        values,
        _np(dataset, f"reward_{head}").reshape(-1),
        elapsed_years(dataset).reshape(-1).double().numpy(),
        _np(dataset, "transition_terminal").reshape(-1),
        _np(dataset, "row_turn").reshape(-1),
        worlds_all,
        _np(dataset, "row_camp").reshape(-1),
        _np(dataset, TRAJECTORY_FIELD).reshape(-1)
        if TRAJECTORY_FIELD in dataset
        else np.full(n, -1),
        _np(dataset, TAIL_VALUE_PREFIX + head).reshape(-1) if randomized else None,
        None if codes is None else np.isin(codes, (TAIL_TERMINAL_EXACT, TAIL_RANDOMIZED)),
        k=k,
        discount_years=CRITIC_DISCOUNT_YEARS,
    )
    scale = float(reference["value"])
    pred = np.asarray(values, dtype=np.float64)[rows]
    target = result["target"][rows]
    tail_target = result["tail_target"][rows]
    closure = result["closure"][rows]
    worlds = worlds_all[rows]
    gated = np.isin(closure, NSTEP_GATED_CLOSURES)
    # Tail weight of each admitted row in its Monte Carlo target (as qol_tail_weight).
    from .labor_advantage import qol_tail_terms

    terms = qol_tail_terms(dataset, 1.0)  # discount coefficients are head-independent
    head_tail = (
        _np(dataset, TAIL_VALUE_PREFIX + head).reshape(-1).astype(np.float64)
        if randomized
        else np.zeros(n)
    )
    part = (terms["coefficient"].numpy() * head_tail)[rows]
    has_tail = terms["has_tail"].numpy()[rows]
    mc_rows = np.asarray(monte_carlo, dtype=np.float64)
    tail_weight = np.where(
        has_tail & (mc_rows > 0),
        np.clip(part / np.where(mc_rows > 0, mc_rows, 1.0), 0.0, 1.0),
        0.0,
    )
    low = tail_weight <= QOL_BIAS_TAIL_CUT
    if world_factors is None:
        from .labor_model import labor_qol_discount_weight

        records = (dataset.get("collection") or {}).get("return_tail_by_world") or {}
        w_ = labor_qol_discount_weight() if weight is None else float(weight)
        world_factors = {
            int(w): tail_schedule_factor(records.get(str(int(w))), w_)
            for w in np.unique(worlds_all[rows])
        }
    factors = {int(w): f for w, f in world_factors.items()}
    row_factor = np.array(
        [factors.get(int(w)) if factors.get(int(w)) is not None else 1.0 for w in worlds_all[rows]],
        dtype=np.float64,
    )
    tail_part = tail_weight * mc_rows / scale  # c_i tail_i / scale on rows with a tail
    induced = -tail_part * (1.0 - 1.0 / row_factor)
    first_order = -tail_part * (row_factor - 1.0)
    limit = CRITIC_GATES["abs_bias_max"]

    def attribution(mask: np.ndarray, observed: float | None) -> dict[str, Any]:
        present = sorted({int(w) for w in worlds[mask]})
        known = [factors.get(w) for w in present if factors.get(w) is not None]
        out: dict[str, Any] = {
            "gating": False,
            "worlds": len(present),
            "worlds_without_factor": len(present) - len(known),
            "mean_factor": float(np.mean(known)) if known else None,
            "predicted_tail_bias": None,
            "predicted_tail_bias_first_order": None,
            "residual_critic_bias": None,
            "flag": False,
        }
        if not mask.any() or observed is None:
            return out
        weights_ = world_balanced(worlds[mask], np.ones(int(mask.sum()), dtype=bool))
        predicted = weighted_mean(induced[mask], weights_)
        residual = observed - predicted
        flips = (abs(observed) <= limit) != (abs(residual) <= limit)
        out.update(
            predicted_tail_bias=predicted,
            predicted_tail_bias_first_order=weighted_mean(first_order[mask], weights_),
            residual_critic_bias=residual,
            verdict_flips_without_tail=bool(flips),
            flag=bool(abs(predicted) > QOL_BIAS_TAIL_ATTRIBUTION_FLAG or flips),
        )
        return out

    out_strata: dict[str, Any] = {}
    for name, mask in strata.items():
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            continue
        g = m & gated
        c = m & low
        cut = _nstep_metrics(pred[c], mc_rows[c], worlds[c], scale)
        covered = cut["rows"] >= QOL_BIAS_MIN_ROWS and cut["worlds"] >= QOL_BIAS_MIN_WORLDS
        out_strata[name] = {
            **_nstep_metrics(pred[g], target[g], worlds[g], scale),
            "bias_role": "non-gating diagnostic (the gated bias is bias_tailcut)",
            "bias_tailcut": {
                "gating": True,
                "target": QOL_BIAS_TARGET_ID,
                "bias": cut["bias"],
                "rows": cut["rows"],
                "worlds": cut["worlds"],
                "excluded_tail_weight_rows": int((m & ~low).sum()),
                "coverage": "PASS" if covered else "HOLD",
                "attribution": attribution(c, cut["bias"]),
            },
            "excluded_living_end_rows": int((m & ~gated).sum()),
            "monte_carlo": {
                "gating": False,
                **_nstep_metrics(pred[m], np.asarray(monte_carlo)[m], worlds[m], scale),
            },
        }
    short = closure == NSTEP_LIVING_END_SHORT
    with_tail = (closure != NSTEP_FULL) & (closure != NSTEP_TERMINAL) & np.isfinite(tail_target)
    living_end = {
        "gating": False,
        "rows": int((~gated).sum()),
        "shortened_rows": int(short.sum()),
        "living_end_rows": int((closure == NSTEP_LIVING_END_SELF).sum()),
        "bootstrap_closure": _nstep_metrics(pred[short], target[short], worlds[short], scale),
        "declared_tail_closure": _nstep_metrics(
            pred[with_tail], tail_target[with_tail], worlds[with_tail], scale
        ),
    }
    counts = {NSTEP_CLOSURES[c]: int((closure == c).sum()) for c in NSTEP_CLOSURES}
    constant_reference = None
    if constant is not None:
        c_value = float(constant["value"])
        c_result = nstep_targets(
            np.full(n, c_value),
            _np(dataset, f"reward_{head}").reshape(-1),
            elapsed_years(dataset).reshape(-1).double().numpy(),
            _np(dataset, "transition_terminal").reshape(-1),
            _np(dataset, "row_turn").reshape(-1),
            worlds_all,
            _np(dataset, "row_camp").reshape(-1),
            _np(dataset, TRAJECTORY_FIELD).reshape(-1)
            if TRAJECTORY_FIELD in dataset
            else np.full(n, -1),
            k=k,
            discount_years=CRITIC_DISCOUNT_YEARS,
        )
        c_target = c_result["target"][rows]
        c_gated = np.isin(c_result["closure"][rows], NSTEP_GATED_CLOSURES)
        c_pred = np.full(len(rows), c_value)
        constant_reference = {
            "gating": False,
            "value": c_value,
            "source": constant.get("source"),
            "definition": "a predictor outputting this constant on every row; n-step targets "
            "bootstrap through it",
            "strata": {
                name: {
                    "nstep": _nstep_metrics(
                        c_pred[np.asarray(mask, bool) & c_gated],
                        c_target[np.asarray(mask, bool) & c_gated],
                        worlds[np.asarray(mask, bool) & c_gated],
                        scale,
                    ),
                    "monte_carlo": _nstep_metrics(
                        c_pred[np.asarray(mask, bool)],
                        mc_rows[np.asarray(mask, bool)],
                        worlds[np.asarray(mask, bool)],
                        scale,
                    ),
                }
                for name, mask in strata.items()
                if np.asarray(mask, bool).any()
            },
        }
    return {
        **({} if constant_reference is None else {"constant_reference": constant_reference}),
        "gating": True,
        "target": QOL_GATE_TARGET,
        "targets": {"nmae": QOL_GATE_TARGET_ID, "bias": QOL_BIAS_TARGET_ID},
        "bias_target": QOL_BIAS_TARGET,
        "bias_tail_attribution": QOL_BIAS_TAIL_ATTRIBUTION,
        "world_factors": {str(w): f for w, f in sorted(factors.items())},
        "reference_scale": dict(reference),
        "rows": len(rows),
        "gated_rows": int(gated.sum()),
        "closures": counts,
        "strata": out_strata,
        "living_end": living_end,
    }


# Provenance of the frozen population-years reference scale (item 3 of the 28 Oct
# population diagnostics). Informational; the frozen value stays the model's unit.
POPULATION_REFERENCE_SCALE_PROVENANCE = {
    "value": 85133.98,
    "defined_at": "src/anthropocene_rl/autonomous_v7/contracts.py:28-40 "
    "(OBJECTIVE_REFERENCE_SCALES['population_years']), re-exported as "
    "labor_model.LABOR_OBJECTIVE_REFERENCE_SCALES (labor_model.py:74-77)",
    "measured_by": "scale_reference.py (full_window_std)",
    "data": "native heuristic (no actions), TRAIN seeds 1000-1059, 45-turn rollouts, rows "
    "with at least 20 later turns (the first 25 turns, 37,085 decision rows), Small world, "
    "model9 bridge on native contract 12, simulator f06e3be (21 October 2026)",
    "era": "Round 3 (pre-labor-actor, pre-r5): heuristic discounted (400 y) population-years "
    "return per decision over a full 400-year-equivalent window; mean 172,285 person-years",
    "previous": "first build (25-turn rollouts, all rows, late returns truncated): 50,391",
}


def population_nstep_diagnostic(
    dataset: Mapping[str, Any],
    rows: np.ndarray,
    values: np.ndarray,
    monte_carlo: np.ndarray,
    strata: Mapping[str, np.ndarray],
    scale: float,
    *,
    world_factors: Mapping[int, float | None] | None = None,
    weight: float | None = None,
) -> dict[str, Any]:
    """Population-years against the n-step k = 5 bootstrapped target (NON-GATING).

    The same windows, exclusions, tail-weight cut and tail attribution as the QoL gate
    (:func:`qol_nstep_calibration` with ``head="population_years"``), in the frozen
    population reference scale. Separates roulette-tail target noise (Monte Carlo metrics
    and the predicted tail bias) from critic error (n-step metrics, residual bias).
    """
    report = qol_nstep_calibration(
        dataset,
        rows,
        values,
        monte_carlo,
        strata,
        {"value": float(scale), "source": "frozen-model-scale"},
        world_factors=world_factors,
        weight=weight,
        head="population_years",
    )
    for metrics in report["strata"].values():
        metrics["bias_role"] = "non-gating diagnostic"
        metrics["bias_tailcut"]["gating"] = False
        metrics["bias_tailcut"].pop("target", None)
    for key in ("target", "targets", "bias_target"):
        report.pop(key, None)
    report.update(
        gating=False,
        head="population_years",
        nstep_target=f"nstep-k{QOL_NSTEP_K}-bootstrap (as {QOL_GATE_TARGET_ID})",
        bias_tailcut_rule=f"Monte Carlo bias on rows with tail weight <= {QOL_BIAS_TAIL_CUT}",
        reference_scale_provenance=POPULATION_REFERENCE_SCALE_PROVENANCE,
    )
    return report


# --- Relative-first critic gates (gate version relative-v1, user decision 28 Oct) -------------

RELATIVE_GATE_VERSION = "relative-v1"
RELATIVE_MIN_LAG = 5  # decisions between the two rows of a gate (a) pair
RELATIVE_MARGIN_W = 0.02  # informative pairs: |G_j - G_i| > 0.02 W (QoL-equivalent)
RELATIVE_VARIANTS = ("qol_years", "population_years", "combined")
RELATIVE_DEFINITION = {
    "id": RELATIVE_GATE_VERSION,
    "pairs": "two admitted rows i, j of one trajectory (world, camp, row_trajectory; decision "
    "position = index among the trajectory's rows ordered by row_turn), position_j > "
    "position_i; gate (a) needs position_j - position_i >= 5, gate (b) takes all pairs",
    "target": "realised discounted return G (Monte Carlo, closed by the dataset's declared "
    "tail, as labor_bc.discounted_returns; both rows of a pair share their trajectory's tail)",
    "informative": "|G_j - G_i| > margin; margin 0.02 W for QoL (W the unit-QoL discount "
    "weight); population and combined use the same 0.02 W of QoL expressed in their units "
    "through the PPO combination (population: 0.02 W s_pop / s_qol; combined: w_qol 0.02 W / "
    "s_qol)",
    "score": "1 when sign(V_j - V_i) = sign(G_j - G_i), 0.5 for a tie V_j = V_i, else 0",
    "aggregation": "world-balanced: mean over worlds of each world's mean score over its "
    "informative pairs (pooled mean also reported)",
    "strata": "a pair belongs to the stratum of its earlier row",
    "combined": "V and G of every head combined with labor_ppo.combine_labor_advantages (the "
    "PPO objective: sum_k w_k X_k / s_k - sum_c m_c X_c), balanced personality and unit cost "
    "multipliers unless declared otherwise",
}


def trajectory_positions(
    worlds: Any, camps: Any, trajectories: Any, turns: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row trajectory index (0..T-1, sorted keys) and decision position within it."""
    worlds = np.asarray(worlds).reshape(-1)
    camps = np.asarray(camps).reshape(-1)
    trajectories = np.asarray(trajectories).reshape(-1)
    turns = np.asarray(turns).reshape(-1)
    order = np.lexsort((np.arange(len(worlds)), turns, trajectories, camps, worlds))
    key = np.stack([worlds[order], camps[order], trajectories[order]], axis=1)
    new = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        new[1:] = np.any(key[1:] != key[:-1], axis=1)
    group = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    position_sorted = np.arange(len(order)) - starts[group]
    traj = np.empty(len(order), dtype=np.int64)
    position = np.empty(len(order), dtype=np.int64)
    traj[order] = group
    position[order] = position_sorted
    return traj, position


_TRIU: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _pairs(size: int) -> tuple[np.ndarray, np.ndarray]:
    if size not in _TRIU:
        _TRIU[size] = np.triu_indices(size, 1)
    return _TRIU[size]


def within_trajectory_ordering(
    values: Any,
    targets: Any,
    traj: Any,
    position: Any,
    worlds: Any,
    *,
    margin: float,
    min_lag: int = 1,
    strata: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Within-trajectory ordering agreement of V with the realised return (pure, numpy).

    All arrays are per evaluated row (admitted rows only). ``min_lag`` 1 is every pair;
    ``strata`` masks over the rows (a pair takes its earlier row's stratum). Returns per
    stratum the world-balanced accuracy over informative pairs, the pooled accuracy, the
    informative share and the counts; ``all`` also carries the median per-trajectory Spearman
    rank correlation of V with G (trajectories of >= 10 rows, report-only).
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    traj = np.asarray(traj).reshape(-1)
    position = np.asarray(position).reshape(-1)
    worlds = np.asarray(worlds).reshape(-1)
    strata = {"all": np.ones(len(values), dtype=bool)} if strata is None else dict(strata)
    names = list(strata)
    masks = np.stack([np.asarray(strata[s], dtype=bool).reshape(-1) for s in names])
    world_ids = {int(w): k for k, w in enumerate(np.unique(worlds))}
    score = np.zeros((len(names), len(world_ids)))
    count = np.zeros((len(names), len(world_ids)))
    pairs_total = np.zeros(len(names))
    rho: list[float] = []
    order = np.lexsort((position, traj))
    bounds = np.flatnonzero(np.r_[True, traj[order][1:] != traj[order][:-1], True])
    for a, b in itertools.pairwise(bounds):
        rows = order[a:b]
        size = len(rows)
        if size >= 10:
            rv = np.argsort(np.argsort(values[rows]))
            rt = np.argsort(np.argsort(targets[rows]))
            if rv.std() > 0 and rt.std() > 0:
                rho.append(float(np.corrcoef(rv, rt)[0, 1]))
        if size < 2:
            continue
        i, j = _pairs(size)
        lag = position[rows][j] - position[rows][i]
        keep = lag >= int(min_lag)
        i, j = rows[i[keep]], rows[j[keep]]
        if not len(i):
            continue
        d_target = targets[j] - targets[i]
        d_value = values[j] - values[i]
        informative = np.abs(d_target) > float(margin)
        s = np.where(
            d_value == 0.0, 0.5, (np.sign(d_value) == np.sign(d_target)).astype(np.float64)
        )
        w = world_ids[int(worlds[rows[0]])]
        anchor = masks[:, i]  # (strata, pairs)
        pairs_total += anchor.sum(axis=1)
        sel = anchor & informative[None, :]
        count[:, w] += sel.sum(axis=1)
        score[:, w] += (sel * s[None, :]).sum(axis=1)
    out: dict[str, Any] = {}
    for k, name in enumerate(names):
        present = count[k] > 0
        informative_pairs = int(count[k].sum())
        out[name] = {
            "pairs": int(pairs_total[k]),
            "informative_pairs": informative_pairs,
            "informative_share": None
            if not pairs_total[k]
            else informative_pairs / float(pairs_total[k]),
            "worlds": int(present.sum()),
            "accuracy": None
            if not present.any()
            else float(np.mean(score[k][present] / count[k][present])),
            "accuracy_pooled": None
            if not informative_pairs
            else float(score[k].sum() / informative_pairs),
        }
    if "all" in out:
        out["all"]["median_trajectory_spearman"] = float(np.median(rho)) if rho else None
        out["all"]["trajectories_spearman"] = len(rho)
    return out


def combined_elder_values(
    heads: Mapping[str, Any],
    *,
    personality: Any = None,
    cost_multipliers: Mapping[str, float] | None = None,
) -> np.ndarray:
    """The PPO combination (``labor_ppo.combine_labor_advantages``) of per-head values.

    Linear, so it maps values and returns exactly as PPO maps their advantages. ``heads``
    holds every task objective and cost head (numpy arrays of equal length, float64).
    """
    from .labor_model import LABOR_COSTS, LABOR_OBJECTIVES, LaborPersonality
    from .labor_ppo import combine_labor_advantages

    task = {n: torch.as_tensor(np.asarray(heads[n], dtype=np.float64)) for n in LABOR_OBJECTIVES}
    costs = {n: torch.as_tensor(np.asarray(heads[n], dtype=np.float64)) for n in LABOR_COSTS}
    return (
        combine_labor_advantages(
            task,
            costs,
            personality=LaborPersonality() if personality is None else personality,
            cost_multipliers=dict.fromkeys(LABOR_COSTS, 1.0)
            if cost_multipliers is None
            else dict(cost_multipliers),
        )
        .double()
        .numpy()
    )


def relative_margins(weight: float, personality: Any = None) -> dict[str, float]:
    """0.02 W of QoL in each variant's units (see :data:`RELATIVE_DEFINITION`)."""
    from .labor_model import LABOR_OBJECTIVE_REFERENCE_SCALES, LaborPersonality

    personality = LaborPersonality() if personality is None else personality
    qol = RELATIVE_MARGIN_W * float(weight)
    s_q = LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"]
    s_p = LABOR_OBJECTIVE_REFERENCE_SCALES["population_years"]
    return {
        "qol_years": qol,
        "population_years": qol * s_p / s_q,
        "combined": float(personality.qol_years) * qol / s_q,
    }


def relative_ordering_report(
    dataset: Mapping[str, Any],
    rows: np.ndarray,
    preds: Mapping[str, np.ndarray],
    returns: Mapping[str, np.ndarray],
    strata: Mapping[str, np.ndarray],
    *,
    weight: float,
    personality: Any = None,
    cost_multipliers: Mapping[str, float] | None = None,
    min_lag: int = RELATIVE_MIN_LAG,
) -> dict[str, Any]:
    """Gates (a) lag >= 5 and (b) all-pairs within-trajectory ordering, every variant.

    ``rows``: admitted rows; ``preds`` / ``returns``: per head on ``rows`` (``returns`` may
    also be full-length arrays indexed by ``rows``); ``strata`` masks over ``rows``.
    """
    from .labor_bc import TRAJECTORY_FIELD
    from .labor_model import LABOR_COSTS, LABOR_OBJECTIVES, LaborPersonality

    personality = LaborPersonality() if personality is None else personality
    multipliers = (
        dict.fromkeys(LABOR_COSTS, 1.0) if cost_multipliers is None else dict(cost_multipliers)
    )
    rows = np.asarray(rows, dtype=np.int64)
    n = int(dataset["row_world"].shape[0])
    trajectories = (
        _np(dataset, TRAJECTORY_FIELD).reshape(-1)
        if TRAJECTORY_FIELD in dataset
        else np.full(n, -1)
    )
    traj, position = trajectory_positions(
        _np(dataset, "row_world").reshape(-1),
        _np(dataset, "row_camp").reshape(-1),
        trajectories,
        _np(dataset, "row_turn").reshape(-1),
    )
    worlds = _np(dataset, "row_world").reshape(-1)[rows]

    def on_rows(array: Any) -> np.ndarray:
        array = np.asarray(array, dtype=np.float64).reshape(-1)
        return array[rows] if len(array) == n and len(rows) != n else array

    heads = tuple(LABOR_OBJECTIVES) + tuple(LABOR_COSTS)
    value = {h: np.asarray(preds[h], dtype=np.float64).reshape(-1) for h in heads}
    target = {h: on_rows(returns[h]) for h in heads}
    value["combined"] = combined_elder_values(
        value, personality=personality, cost_multipliers=multipliers
    )
    target["combined"] = combined_elder_values(
        target, personality=personality, cost_multipliers=multipliers
    )
    margins = relative_margins(weight, personality)
    variants: dict[str, Any] = {}
    for name in RELATIVE_VARIANTS:
        common = {
            "traj": traj[rows],
            "position": position[rows],
            "worlds": worlds,
            "strata": strata,
        }
        variants[name] = {
            "margin": margins[name],
            "lag_ge_min": within_trajectory_ordering(
                value[name], target[name], margin=margins[name], min_lag=min_lag, **common
            ),
            "all_pairs": within_trajectory_ordering(
                value[name], target[name], margin=margins[name], min_lag=1, **common
            ),
            "all_pairs_no_margin": {
                "all": within_trajectory_ordering(
                    value[name],
                    target[name],
                    margin=0.0,
                    min_lag=1,
                    **{**common, "strata": {"all": np.ones(len(rows), dtype=bool)}},
                )["all"],
                "gating": False,
            },
        }
    return {
        "gate_version": RELATIVE_GATE_VERSION,
        "definition": RELATIVE_DEFINITION,
        "min_lag": int(min_lag),
        "margin_w": RELATIVE_MARGIN_W,
        "weight": float(weight),
        "combination": {
            "personality": personality.state_dict(),
            "cost_multipliers": multipliers,
            "function": "labor_ppo.combine_labor_advantages",
        },
        "rows": len(rows),
        "variants": variants,
    }


# --- Advantage magnitude on the combined elder reward (non-gating, user 28 Oct) ---------------

ADVANTAGE_MAGNITUDE_KS = (1, 5)
ADVANTAGE_MAGNITUDE_DEFINITION = {
    "gating": False,
    "predicted": "A_hat_k(t) = sum_{i<k} D_i r_{t+i} + D_k V(s_{t+k}) - V(s_t) on the combined "
    "elder reward (rewards, values and returns of every head through "
    "labor_ppo.combine_labor_advantages), the critic's own k-step advantage; windows reaching "
    "a terminal row close exactly; windows reaching the collection window's living end are "
    "excluded (no Monte Carlo tail in A_hat)",
    "realised": "A_MC(t) = G_t - V(s_t), G the combined Monte Carlo return (lambda = 1 GAE). "
    "Given the window's information, E[A_MC | s_t..s_{t+k}, r_t..r_{t+k-1}] = A_hat_k exactly "
    "when V(s_{t+k}) is the true value, so the regression slope of A_MC on A_hat_k is 1 for an "
    "exact critic whatever the return noise; > 1 means A_hat is too small, < 1 too large. "
    "The return-difference form sum D r + D_k G_{t+k} - G_t is identically 0 (the Monte Carlo "
    "recursion) and cannot serve. std(A_hat)/std(A_MC) is below 1 even for an exact critic "
    "(A_MC carries the post-window noise D_k (G_{t+k} - V(s_{t+k})))",
    "aggregation": "world-balanced weights (every world equal mass), per stratum",
    "sign_agreement_world_centred": "the same sign agreement after removing each world's mean "
    "of A_hat and of A_MC: A_MC = G - V shares the critic's own V(s_t), so a critic level "
    "error (or a world's roulette tail factor) shifts every A_MC of that world and the raw "
    "sign agreement then mostly measures that offset (measured: an untrained critic scores "
    "higher than trained ones on the raw form)",
    "ppo_scale": "labor PPO uses the combined advantage raw in the clipped surrogate "
    "(labor_ppo.attempt_policy_terms: -mean(min(rho A, clip(rho) A)) - entropy_weight H); no "
    "per-batch standardisation, so the absolute scale sets the policy step against the "
    "entropy bonus and the gradient clip",
}


def _weighted_moments(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, float | None]:
    total = float(w.sum())
    if not total > 0 or len(x) < 2:
        return {"slope": None, "std_ratio": None, "corr": None}
    mx, my = float((w * x).sum() / total), float((w * y).sum() / total)
    vx = float((w * (x - mx) ** 2).sum() / total)
    vy = float((w * (y - my) ** 2).sum() / total)
    cxy = float((w * (x - mx) * (y - my)).sum() / total)
    return {
        "slope": None if vx <= 0 else cxy / vx,
        "std_ratio": None if vy <= 0 else math.sqrt(vx / vy),
        "corr": None if vx <= 0 or vy <= 0 else cxy / math.sqrt(vx * vy),
    }


def advantage_magnitude_report(
    dataset: Mapping[str, Any],
    rows: np.ndarray,
    values: Mapping[str, np.ndarray],
    returns: Mapping[str, np.ndarray],
    strata: Mapping[str, np.ndarray],
    *,
    ks: Sequence[int] = ADVANTAGE_MAGNITUDE_KS,
    personality: Any = None,
    cost_multipliers: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Direction and order of magnitude of the critic's k-step combined advantage (NON-GATING).

    ``values`` / ``returns``: per head on EVERY dataset row (bootstraps may sit anywhere);
    ``rows``: admitted rows; ``strata``: masks over ``rows``. See
    :data:`ADVANTAGE_MAGNITUDE_DEFINITION`. Per k and stratum: rows, sign agreement of A_hat
    with A_MC (ties 1/2), slope of A_MC on A_hat, std(A_hat) / std(A_MC), their correlation
    (world-balanced) and the pooled slope; the scale block reports A_hat's mean |.| and std in
    combined units (what the raw PPO surrogate sees).
    """
    from .labor_bc import CRITIC_DISCOUNT_YEARS, TRAJECTORY_FIELD, elapsed_years
    from .labor_model import LABOR_COSTS, LABOR_OBJECTIVES

    rows = np.asarray(rows, dtype=np.int64)
    n = int(dataset["row_world"].shape[0])
    heads = (*LABOR_OBJECTIVES, *LABOR_COSTS)
    v = combined_elder_values(
        {h: values[h] for h in heads}, personality=personality, cost_multipliers=cost_multipliers
    )
    g = combined_elder_values(
        {h: returns[h] for h in heads}, personality=personality, cost_multipliers=cost_multipliers
    )
    r = combined_elder_values(
        {h: _np(dataset, f"reward_{h}").reshape(-1).astype(np.float64) for h in heads},
        personality=personality,
        cost_multipliers=cost_multipliers,
    )
    worlds_all = _np(dataset, "row_world").reshape(-1)
    common = (
        _np(dataset, "row_turn").reshape(-1),
        worlds_all,
        _np(dataset, "row_camp").reshape(-1),
        _np(dataset, TRAJECTORY_FIELD).reshape(-1)
        if TRAJECTORY_FIELD in dataset
        else np.full(n, -1),
    )
    elapsed = elapsed_years(dataset).reshape(-1).double().numpy()
    terminal = _np(dataset, "transition_terminal").reshape(-1)
    a_mc_all = g - v
    # Each head's coefficient in the combination (linear: combine a unit vector).
    coefficient = {
        h: float(
            combined_elder_values(
                {x: np.array([1.0 if x == h else 0.0]) for x in heads},
                personality=personality,
                cost_multipliers=cost_multipliers,
            )[0]
        )
        for h in heads
    }
    head_reward = {h: _np(dataset, f"reward_{h}").reshape(-1).astype(np.float64) for h in heads}
    head_value = {h: np.asarray(values[h], dtype=np.float64).reshape(-1) for h in heads}
    head_return = {h: np.asarray(returns[h], dtype=np.float64).reshape(-1) for h in heads}

    def moments(x: np.ndarray, w: np.ndarray) -> dict[str, float | None]:
        mean = weighted_mean(x, w)
        if mean is None:
            return {"mean": None, "std": None}
        return {"mean": mean, "std": float(np.sqrt(weighted_mean((x - mean) ** 2, w)))}

    out: dict[str, Any] = {
        "definition": ADVANTAGE_MAGNITUDE_DEFINITION,
        "headline": "sign_agreement",
        "coefficients": coefficient,
        "ks": {},
    }
    for k in ks:
        result = nstep_targets(
            v, r, elapsed, terminal, *common, k=int(k), discount_years=CRITIC_DISCOUNT_YEARS
        )
        a_hat_all = result["target"] - v
        gated = np.isin(result["closure"][rows], NSTEP_GATED_CLOSURES)
        head_a_hat = {
            h: nstep_targets(
                head_value[h],
                head_reward[h],
                elapsed,
                terminal,
                *common,
                k=int(k),
                discount_years=CRITIC_DISCOUNT_YEARS,
            )["target"]
            - head_value[h]
            for h in heads
        }
        per_stratum: dict[str, Any] = {}
        for name, mask in strata.items():
            m = np.asarray(mask, dtype=bool) & gated
            if not m.any():
                continue
            sel = rows[m]
            a_hat, a_mc = a_hat_all[sel], a_mc_all[sel]
            w = world_balanced(worlds_all[sel], np.ones(len(sel), dtype=bool))
            agree = np.where(
                a_hat == 0.0, 0.5, (np.sign(a_hat) == np.sign(a_mc)).astype(np.float64)
            )
            pooled = _weighted_moments(a_hat, a_mc, np.ones(len(sel)))
            # World-centred sign agreement: the critic's level error on a world (and the
            # world's roulette tail factor) shifts A_MC = G - V by a constant there; removing
            # each world's mean of both sides leaves the within-world direction.
            sw = worlds_all[sel]
            a_hat_c, a_mc_c = a_hat.copy(), a_mc.copy()
            for world in np.unique(sw):
                wm = sw == world
                a_hat_c[wm] -= a_hat[wm].mean()
                a_mc_c[wm] -= a_mc[wm].mean()
            agree_c = np.where(
                a_hat_c == 0.0, 0.5, (np.sign(a_hat_c) == np.sign(a_mc_c)).astype(np.float64)
            )
            per_head: dict[str, Any] = {}
            predicted_spread, realised_spread = {}, {}
            for h in heads:
                ah = head_a_hat[h][sel]
                amc = head_return[h][sel] - head_value[h][sel]
                stats = {
                    "target": moments(head_return[h][sel], w),
                    "value": moments(head_value[h][sel], w),
                    "a_hat": moments(ah, w),
                    "a_mc": moments(amc, w),
                }
                predicted_spread[h] = abs(coefficient[h]) * (stats["a_hat"]["std"] or 0.0)
                realised_spread[h] = abs(coefficient[h]) * (stats["a_mc"]["std"] or 0.0)
                v_std, g_std = stats["value"]["std"], stats["target"]["std"]
                per_head[h] = {
                    "popart": stats,
                    "advantage_to_value_spread": None
                    if not v_std
                    else stats["a_hat"]["std"] / v_std,
                    "realised_advantage_to_return_spread": None
                    if not g_std
                    else stats["a_mc"]["std"] / g_std,
                    "sign_agreement": weighted_mean(
                        np.where(ah == 0.0, 0.5, (np.sign(ah) == np.sign(amc)).astype(np.float64)),
                        w,
                    ),
                }
            total_p = sum(predicted_spread.values())
            total_r = sum(realised_spread.values())
            for h in heads:
                per_head[h]["combined_contribution_std"] = predicted_spread[h]
                per_head[h]["combined_contribution_share"] = (
                    None if not total_p else predicted_spread[h] / total_p
                )
                per_head[h]["realised_contribution_std"] = realised_spread[h]
                per_head[h]["realised_contribution_share"] = (
                    None if not total_r else realised_spread[h] / total_r
                )
            per_stratum[name] = {
                "rows": len(sel),
                "worlds": len(np.unique(worlds_all[sel])),
                "sign_agreement": weighted_mean(agree, w),
                "sign_agreement_world_centred": weighted_mean(agree_c, w),
                "a_mc_mean_over_std": None
                if not moments(a_mc, w)["std"]
                else moments(a_mc, w)["mean"] / moments(a_mc, w)["std"],
                **_weighted_moments(a_hat, a_mc, w),
                "slope_pooled": pooled["slope"],
                "scale": {
                    "a_hat": moments(a_hat, w),
                    "a_mc": moments(a_mc, w),
                    "a_hat_mean_abs": weighted_mean(np.abs(a_hat), w),
                    "units": "combined elder reward (sum_k w_k X_k / s_k - sum_c m_c X_c)",
                },
                "heads": per_head,
            }
        out["ks"][str(int(k))] = per_stratum
    return out


def _head_logits(
    task: Any, dataset: Mapping[str, Any], rows: np.ndarray, head: str, batch_size: int
) -> torch.Tensor:
    from .labor_device import module_device, to_device
    from .labor_model import critic_privileged_batch

    device = module_device(task)
    tensors = _input_tensors(dataset)
    out = []
    with torch.no_grad():
        for index, batch in _batches(tensors, rows, batch_size):
            batch = to_device(batch, device)
            privileged = critic_privileged_batch(task, dataset, index, device=device)
            out.append(task(batch, privileged)[head].double().cpu())
    return torch.cat(out) if out else torch.zeros(0)


def _head_probabilities(
    task: Any, dataset: Mapping[str, Any], rows: np.ndarray, head: str, batch_size: int
) -> np.ndarray:
    """Softmax of a categorical head (rows x bins); a critic ensemble's is the uniform
    mixture of its members' distributions."""
    from .labor_critic_ensemble import EnsembleTask, ensemble_head_probabilities

    if isinstance(task, EnsembleTask):
        return ensemble_head_probabilities(task, dataset, rows, head, batch_size)
    return torch.softmax(_head_logits(task, dataset, rows, head, batch_size), dim=-1).numpy()


def population_tail_explanation(
    task: Any,
    dataset: Mapping[str, Any],
    rows: np.ndarray,
    observed_max: float,
    batch_size: int = 256,
    *,
    top: int = 10,
    alarm: float = 0.01,
) -> dict[str, Any]:
    """Where the population head's mass beyond the observed max sits (NON-GATING).

    ``observed_max`` is the largest normalized target G / s of these rows (as
    ``value_support``; the stability check uses the fitting rows' maximum on its final
    pass). Per support bin above it: decoded value (reference scales and person-years),
    mean probability and mean excess contribution (p (v - max)), so the bins carrying
    ``upper_excess`` are explicit. Per row: how concentrated the excess is (rows above the
    0.01 alarm level, the share of the total from the top 1% / top ``top`` rows, how many
    rows carry 50% / 90% of it) and the ``top`` rows by excess with their targets.
    """
    from .labor_bc import discounted_returns
    from .labor_model import LABOR_OBJECTIVE_REFERENCE_SCALES

    rows = np.asarray(rows, dtype=np.int64)
    scale = LABOR_OBJECTIVE_REFERENCE_SCALES["population_years"]
    report: dict[str, Any] = {
        "gating": False,
        "rows": len(rows),
        "observed_normalized_max": float(observed_max),
        "observed_max_person_years": float(observed_max) * scale,
        "alarm_level": alarm,
    }
    if not len(rows):
        return report
    from .labor_model import symexp

    probabilities = _head_probabilities(task, dataset, rows, "population_years", batch_size)
    values = symexp(task.bins.double()).cpu().numpy()
    upper = values > float(observed_max)
    excess_per_bin = probabilities * np.where(upper, values - float(observed_max), 0.0)
    row_excess = excess_per_bin.sum(axis=1)
    mean_mass = probabilities.mean(axis=0)
    mean_excess = excess_per_bin.mean(axis=0)
    total = float(row_excess.sum())
    bins = []
    for b in np.flatnonzero(upper):
        bins.append(
            {
                "bin": int(b),
                "value_reference_scales": float(values[b]),
                "value_person_years": float(values[b]) * scale,
                "mean_mass": float(mean_mass[b]),
                "mean_excess": float(mean_excess[b]),
                "excess_share": float(mean_excess[b] * len(rows) / total) if total > 0 else 0.0,
            }
        )
    ordered = np.sort(row_excess)[::-1]
    cumulative = np.cumsum(ordered) / total if total > 0 else np.zeros_like(ordered)
    one_percent = max(1, math.ceil(0.01 * len(rows)))
    targets = discounted_returns(dataset)["population_years"].double().numpy()[rows] / scale
    preds = (probabilities * values).sum(axis=1)
    worlds = _np(dataset, "row_world").reshape(-1)[rows]
    camps = _np(dataset, "row_camp").reshape(-1)[rows]
    turns = _np(dataset, "row_turn").reshape(-1)[rows]
    top_rows = []
    for i in np.argsort(-row_excess, kind="stable")[:top]:
        top_rows.append(
            {
                "world": int(worlds[i]),
                "camp": int(camps[i]),
                "turn": int(turns[i]),
                "upper_excess": float(row_excess[i]),
                "upper_mass": float((probabilities[i] * upper).sum()),
                "pred_reference_scales": float(preds[i]),
                "target_reference_scales": float(targets[i]),
            }
        )
    report.update(
        upper_excess_mean=float(row_excess.mean()),
        upper_mass_mean=float((probabilities * upper).sum(axis=1).mean()),
        support_high_reference_scales=float(values[-1]),
        bins_above_observed_max=int(upper.sum()),
        bins=sorted(bins, key=lambda r: -r["mean_excess"])[:top],
        concentration={
            "rows_above_alarm": int((row_excess > alarm).sum()),
            "share_rows_above_alarm": float((row_excess > alarm).mean()),
            "top_1pct_rows": one_percent,
            "top_1pct_share": float(cumulative[one_percent - 1]) if total > 0 else 0.0,
            f"top_{top}_share": float(cumulative[min(top, len(rows)) - 1]) if total > 0 else 0.0,
            "rows_for_50pct": int(np.searchsorted(cumulative, 0.5) + 1) if total > 0 else 0,
            "rows_for_90pct": int(np.searchsorted(cumulative, 0.9) + 1) if total > 0 else 0,
            "row_excess_quantiles": {
                q: float(np.quantile(row_excess, float(q))) for q in ("0.5", "0.9", "0.99")
            },
            "verdict": "concentrated"
            if total > 0 and np.searchsorted(cumulative, 0.5) + 1 <= max(1, 0.05 * len(rows))
            else "widespread",
            "verdict_rule": "concentrated when 5% of rows or fewer carry half the excess",
        },
        top_rows=top_rows,
    )
    return report


def critic_dataset_report(
    task: Any,
    costs: Any,
    dataset: Mapping[str, Any],
    batch_size: int = 256,
    normalizer: Any = None,
    qol_reference: Mapping[str, Any] | None = None,
    constant: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Every head against the Monte Carlo return (``heads``, frozen model reference scales,
    unchanged) and the QoL gate's n-step section (``qol_nstep``, :data:`QOL_GATE_TARGET`)
    in ``qol_reference`` units (default: the frozen 43.152).

    Gate relative-v1 (28 Oct): ``relative`` holds the within-trajectory ordering gates of
    every variant (:func:`relative_ordering_report`); ``constant`` (the fitting-set mean QoL
    return, {"value", "source"}) adds the constant-predictor reference to ``qol_nstep``
    (default: this dataset's own mean, labelled so). A dataset whose tails were normalised
    on load attributes no residual tail bias (factor 1 per world)."""
    from .labor_bc import (
        RETURN_TAIL_RANDOMIZED,
        critic_target_available,
        discounted_returns,
        return_tail_method,
    )
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
    # Report-only curated scenario family strata (labor_scenarios detection rules).
    from .labor_scenarios import FAMILY_BITS, row_family_tags

    by_family: dict[str, Any] = {}
    tags = row_family_tags(dataset, normalizer=normalizer)
    if tags is not None:
        tags = tags[rows]
        for family, bit in FAMILY_BITS.items():
            m = (tags & bit) != 0
            if not m.any():
                continue
            by_family[family] = {
                name: critic_head_metrics(
                    preds[name][m],
                    returns[name][rows][m],
                    worlds[m],
                    turns[m],
                    scale,
                    zero_head=bool(np.all(returns[name][rows] == 0.0)),
                )
                for name, scale in scales.items()
            }
    # QoL gate v2 (user 28 Oct): the n-step target needs V on every row (bootstraps), so the
    # rows outside the admitted set get their own forward pass; the admitted rows' values
    # are the ones above, so the Monte Carlo metrics are unchanged.
    qol_nstep = None
    population_nstep = population_tail = None
    relative = advantage_magnitude = None
    if len(rows):
        from .labor_bc import tail_normalization_mode
        from .labor_model import labor_qol_discount_weight

        qol_head_ = getattr(task, "qol_head", None)
        relative = relative_ordering_report(
            dataset,
            rows,
            preds,
            {
                name: returns[name][rows]
                for name in (*LABOR_OBJECTIVE_REFERENCE_SCALES, *LABOR_COST_REFERENCE_SCALES)
            },
            strata,
            weight=labor_qol_discount_weight() if qol_head_ is None else qol_head_.discount_weight,
        )
        normalized_tails = tail_normalization_mode(dataset) != "none"
        if constant is None:
            constant = {
                "value": float(np.mean(returns["qol_years"][rows])),
                "source": "evaluated dataset's own mean Monte Carlo QoL return (no fitting set)",
            }
        size = int(dataset["row_world"].shape[0])
        values = np.zeros(size, dtype=np.float64)
        values[rows] = preds["qol_years"]
        population = np.zeros(size, dtype=np.float64)
        population[rows] = preds["population_years"]
        others = np.flatnonzero(~keep)
        extra = critic_predictions(task, costs, dataset, others, batch_size) if len(others) else {}
        if len(others):
            values[others] = extra["qol_years"]
            population[others] = extra["population_years"]
        every_head = {}
        for name in (*LABOR_OBJECTIVE_REFERENCE_SCALES, *LABOR_COST_REFERENCE_SCALES):
            full = np.zeros(size, dtype=np.float64)
            full[rows] = preds[name]
            if len(others):
                full[others] = extra[name]
            every_head[name] = full
        advantage_magnitude = advantage_magnitude_report(dataset, rows, every_head, returns, strata)
        qol_nstep = qol_nstep_calibration(
            dataset,
            rows,
            values,
            returns["qol_years"][rows],
            strata,
            frozen_qol_reference() if qol_reference is None else qol_reference,
            weight=None
            if getattr(task, "qol_head", None) is None
            else task.qol_head.discount_weight,
            constant=constant,
            **(
                {"world_factors": {int(w): 1.0 for w in np.unique(worlds)}}
                if normalized_tails
                else {}
            ),
        )
        # Population diagnostics (28 Oct, non-gating): n-step target, tail-cut bias with
        # tail attribution, and where the head's upper-tail excess sits.
        population_nstep = population_nstep_diagnostic(
            dataset,
            rows,
            population,
            returns["population_years"][rows],
            strata,
            LABOR_OBJECTIVE_REFERENCE_SCALES["population_years"],
            world_factors={int(w): f for w, f in qol_nstep["world_factors"].items()},
        )
        population_tail = population_tail_explanation(
            task, dataset, rows, observed_max["population_years"], batch_size
        )
    tail_record: dict[str, Any] = {}
    if return_tail_method(dataset) == RETURN_TAIL_RANDOMIZED and len(rows):
        from .labor_model import labor_qol_discount_weight

        qol_head = getattr(task, "qol_head", None)
        weight = labor_qol_discount_weight() if qol_head is None else qol_head.discount_weight
        tail_record["qol_tail"] = qol_tail_diagnostics(
            dataset,
            rows,
            preds["qol_years"],
            returns["qol_years"][rows],
            LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"],
            weight,
        )
    from .labor_bc import FALSE_TERMINAL_FIELD, TAIL_NORMALIZATION_FIELD

    corrections = {
        key: (dataset.get("collection") or {}).get(key)
        for key in (TAIL_NORMALIZATION_FIELD, FALSE_TERMINAL_FIELD)
        if (dataset.get("collection") or {}).get(key) is not None
    }
    return {
        **tail_record,
        **({} if relative is None else {"relative": relative}),
        **({} if advantage_magnitude is None else {"advantage_magnitude": advantage_magnitude}),
        **({"loader_corrections": corrections} if corrections else {}),
        **({} if qol_nstep is None else {"qol_nstep": qol_nstep}),
        **({} if population_nstep is None else {"population_nstep": population_nstep}),
        **({} if population_tail is None else {"population_tail_explanation": population_tail}),
        "by_scenario_family": by_family,
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


def resolve_loader_corrections(
    state: Mapping[str, Any], override: Mapping[str, str | None] | None = None
) -> tuple[dict[str, str], str]:
    """Active loader corrections of an evaluation: the checkpoint's record, each key replaced
    by a non-None ``override`` value; returns (active corrections, "checkpoint" or
    "explicit-override")."""
    from .labor_bc import loader_corrections

    corrections = dict(loader_corrections(state.get("heuristic_critic_pretraining")))
    explicit = {k: v for k, v in (override or {}).items() if v is not None}
    unknown = sorted(set(explicit) - set(corrections))
    if unknown:
        raise ValueError(f"unknown loader corrections {unknown}")
    corrections.update(explicit)
    active = {k: v for k, v in corrections.items() if v != "none"}
    return active, "explicit-override" if explicit else "checkpoint"


def _checkpoint_paths(checkpoint: Path | str | Sequence[Path | str]) -> list[Path]:
    if isinstance(checkpoint, (str, Path)):
        return [Path(checkpoint)]
    paths = [Path(p) for p in checkpoint]
    if not paths:
        raise ValueError("critic-calibration needs at least one checkpoint")
    return paths


def _generalisation_gap(
    calibration: Mapping[str, Any], fitting: Mapping[str, Any]
) -> dict[str, Any]:
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
    return gap


def run_critic_calibration(
    checkpoint: Path | Sequence[Path],
    dataset_path: Path | Sequence[Path],
    fit_dataset_path: Path | Sequence[Path] | None,
    batch_size: int = 256,
    device: Any = None,
    qol_reference: Mapping[str, Any] | None = None,
    corrections_override: Mapping[str, str | None] | None = None,
    member_reports: bool = False,
) -> dict[str, Any]:
    """``qol_reference`` (:func:`load_qol_reference_scale`) is the QoL gate's unit; absent,
    the frozen 43.152.

    Loader corrections: by default those the checkpoint's critic fit recorded
    (``labor_bc.loader_corrections``). ``corrections_override`` ({"tail_normalize": ...,
    "false_terminal_policy": ...}; a None value keeps the recorded one) applies the given
    corrections to the calibration and fitting datasets whatever the checkpoint recorded,
    for scoring a critic against another fit's targets (apples-to-apples).

    ``checkpoint`` may be several paths: they are scored as ONE critic whose per-row
    predictions are the mean of the members' in value units
    (:mod:`labor_critic_ensemble`), through exactly the single-checkpoint metrics; the
    members must share input kind, normalizers, value support, QoL head and loader
    corrections (otherwise ``ValueError``). The report then carries an ``ensemble`` block
    and, with ``member_reports``, each member's own evaluation of the same loaded data
    (``member_reports``). One path is the historical evaluation, output unchanged."""
    from .labor_bc import CRITIC_DISCOUNT_YEARS

    qol_reference = frozen_qol_reference() if qol_reference is None else dict(qol_reference)

    from .labor_bc import critic_target_available, discounted_returns

    paths = _checkpoint_paths(checkpoint)
    if member_reports and len(paths) < 2:
        raise ValueError("member reports need several checkpoints")
    ensemble = None
    if len(paths) == 1:
        checkpoint = paths[0]
        state, _, task, costs, normalizer = _load_models(checkpoint)
        # The loader corrections the fit trained on (absent: none, the historical evaluation).
        corrections, source = resolve_loader_corrections(state, corrections_override)
        fit_records = [task]
    else:
        from .labor_critic_ensemble import load_critic_ensemble

        ensemble = load_critic_ensemble(paths, corrections_override)
        task, costs, normalizer = ensemble.task, ensemble.costs, ensemble.normalizer
        corrections, source = ensemble.corrections, ensemble.corrections_source
        fit_records = [m.task for m in ensemble.members]
    if device is not None:  # forward passes only; predictions come back to the host
        task.to(device)
        costs.to(device)
    privileged = task.privileged_layout is not None
    calibration_dataset = _load_dataset(
        dataset_path, normalizer, require_privileged=privileged, corrections=corrections
    )
    fit_dataset = None
    constant = None
    if fit_dataset_path is not None:
        fit_dataset = _load_dataset(
            fit_dataset_path, normalizer, require_privileged=privileged, corrections=corrections
        )
        admitted = ~_np(fit_dataset, "row_quarantined").reshape(-1).astype(bool) & (
            critic_target_available(fit_dataset).numpy().astype(bool)
        )
        constant = {
            "value": float(
                discounted_returns(fit_dataset)["qol_years"].double().numpy()[admitted].mean()
            ),
            "source": "fitting-set mean Monte Carlo QoL return (admitted rows, unweighted)",
        }
    if privileged:
        fitted = set()
        for member_task in fit_records:
            fitted |= set(member_task.privileged_normalizer.fit_record.get("worlds", []))
        overlap = sorted(
            fitted & {int(w) for w in calibration_dataset["row_world"].reshape(-1).tolist()}
        )
        if overlap:
            raise ValueError(
                f"calibration worlds {overlap[:5]} fitted the privileged normalizer; not held out"
            )
    calibration = critic_dataset_report(
        task,
        costs,
        calibration_dataset,
        batch_size,
        normalizer=normalizer,
        qol_reference=qol_reference,
        constant=constant,
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "command": "critic-calibration",
        "checkpoint": str(checkpoint) if ensemble is None else [str(p) for p in paths],
        "dataset": dataset_label(dataset_path),
        "targets": (
            "discounted returns over each (world, camp) trajectory, closed at the collection "
            f"horizon by the dataset's declared tail ({calibration['return_tail']}); "
            "truncated-zero gives a living cutoff no tail"
        ),
        "discount_tau_years": CRITIC_DISCOUNT_YEARS,
        "gates": CRITIC_GATES,
        "gate_semantics": "PASS all gates; FAIL any metric gate; HOLD metrics pass but coverage "
        "insufficient",
        "weighting": "nMAE/bias world-balanced; ranking pooled over informative pairs "
        "(world-balanced "
        "also shown)",
        "calibration": calibration,
        "verdict": {name: strata["all"]["gate"] for name, strata in calibration["heads"].items()},
        # QoL gate v2 (user 28 Oct): target definition and unit of the qol_nstep sections.
        "qol_gate_target": QOL_GATE_TARGETS,
        "qol_reference_scale": qol_reference,
    }
    if privileged:
        report["privileged_critic"] = {
            "schema_version": int(task.privileged_layout.schema_version),
            "normalizer_digest": task.privileged_normalizer.digest(),
            "normalizer_fit_worlds": list(task.privileged_normalizer.fit_record.get("worlds", [])),
        }
    if corrections or source != "checkpoint":
        report["loader_corrections"] = corrections
        report["loader_corrections_source"] = source
    if ensemble is not None:
        report["ensemble"] = ensemble.block()
    if fit_dataset_path is not None:
        shared = sorted(
            {int(w) for w in fit_dataset["row_world"].reshape(-1).tolist()}
            & {int(w) for w in calibration_dataset["row_world"].reshape(-1).tolist()}
        )
        if shared:
            raise ValueError(f"calibration and fitting datasets share worlds {shared[:5]}")
        fitting = critic_dataset_report(
            task,
            costs,
            fit_dataset,
            batch_size,
            normalizer=normalizer,
            qol_reference=qol_reference,
            constant=constant,
        )
        report["fit_dataset"] = dataset_label(fit_dataset_path)
        report["fitting"] = fitting
        report["generalisation_gap"] = _generalisation_gap(calibration, fitting)
    if member_reports:
        # Each member alone, over the same loaded datasets (one data load for N+1 scorings).
        members = []
        for m in ensemble.members:
            own: dict[str, Any] = {
                "index": m.index,
                "checkpoint": str(m.path),
                "sha256": m.sha256,
                "calibration": critic_dataset_report(
                    m.task,
                    m.costs,
                    calibration_dataset,
                    batch_size,
                    normalizer=normalizer,
                    qol_reference=qol_reference,
                    constant=constant,
                ),
            }
            own["verdict"] = {
                name: strata["all"]["gate"] for name, strata in own["calibration"]["heads"].items()
            }
            if fit_dataset_path is not None:
                own["fitting"] = critic_dataset_report(
                    m.task,
                    m.costs,
                    fit_dataset,
                    batch_size,
                    normalizer=normalizer,
                    qol_reference=qol_reference,
                    constant=constant,
                )
                own["generalisation_gap"] = _generalisation_gap(own["calibration"], own["fitting"])
            members.append(own)
        report["member_reports"] = members
    return report


def _summarize_member_reports(report: Mapping[str, Any]) -> list[str]:
    lines = []
    for member in report.get("member_reports", []):
        lines.append(f"  member {member['index']} {member['checkpoint']}")
        for name, strata in member["calibration"]["heads"].items():
            m = strata["all"]
            rank = m["ranking"]
            lines.append(
                f"    {name:<17} all {m['gate']:<4} nMAE={_fmt(m['nmae'])} bias={_fmt(m['bias'])} "
                f"rank={_fmt(rank['accuracy'])}"
            )
    return lines


def summarize_critic(report: Mapping[str, Any]) -> str:
    lines = []
    ensemble = report.get("ensemble")
    if ensemble:
        lines.append(
            f"critic-calibration ENSEMBLE of {ensemble['member_count']} checkpoints "
            f"(per-row mean in value units): "
            + ", ".join(f"{m['checkpoint']}@{m['sha256'][:12]}" for m in ensemble["members"])
        )
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
                    f"  {name:<17} {stratum:<16} {m['gate']:<4} nMAE={_fmt(m['nmae'])} "
                    f"bias={_fmt(m['bias'])} "
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
                    f"clip<={_fmt(h['clip_below_fraction'], 4)} "
                    f"clip>={_fmt(h['clip_above_fraction'], 4)} "
                    f"mass>max={_fmt(h['upper_mass'], 4)} top-bin={_fmt(h['top_bin_mass'], 5)} "
                    f"decoded>max={_fmt(h['upper_contribution'])} excess={_fmt(h['upper_excess'])}"
                )
    lines.extend(_summarize_member_reports(report))
    for label in ("calibration", "fitting"):
        nstep = (report.get(label) or {}).get("qol_nstep")
        if not nstep:
            continue
        ref = nstep["reference_scale"]
        lines.append(
            f"  qol gate [{label}] target={nstep['target']['id']} scale={_fmt(ref['value'])} "
            f"({ref['source']}) gated rows={nstep['gated_rows']}/{nstep['rows']} "
            f"closures={nstep['closures']}"
        )
        for stratum, m in nstep["strata"].items():
            mc = m["monte_carlo"]
            cut = m.get("bias_tailcut") or {}
            lines.append(
                f"    {stratum:<16} gate: n-step nMAE={_fmt(m['nmae'])} tail-cut MC "
                f"bias={_fmt(cut.get('bias'))} ({cut.get('rows')} rows, {cut.get('coverage')}) "
                f"| non-gating: n-step bias={_fmt(m['bias'])} MC nMAE={_fmt(mc['nmae'])} "
                f"bias={_fmt(mc['bias'])}"
            )
    for label in ("calibration", "fitting"):
        section = report.get(label) or {}
        pop = section.get("population_nstep")
        if pop:
            for stratum, m in pop["strata"].items():
                a = m["bias_tailcut"]["attribution"]
                lines.append(
                    f"  population (non-gating) [{label}] {stratum:<16} n-step "
                    f"nMAE={_fmt(m['nmae'])} bias={_fmt(m['bias'])} | MC nMAE="
                    f"{_fmt(m['monte_carlo']['nmae'])} | tail-cut bias="
                    f"{_fmt(m['bias_tailcut']['bias'])} factor={_fmt(a['mean_factor'])} "
                    f"tail={_fmt(a['predicted_tail_bias'])} "
                    f"residual={_fmt(a['residual_critic_bias'])}"
                )
        explanation = section.get("population_tail_explanation")
        if explanation and "concentration" in explanation:
            c = explanation["concentration"]
            lines.append(
                f"  population tail [{label}] excess={_fmt(explanation['upper_excess_mean'], 4)} "
                f"max={_fmt(explanation['observed_normalized_max'])} "
                f"bins>max={explanation['bins_above_observed_max']} "
                f"rows>{explanation['alarm_level']}={c['rows_above_alarm']} "
                f"50%-in={c['rows_for_50pct']} rows ({c['verdict']})"
            )
    for label in ("calibration", "fitting"):
        tail = (report.get(label) or {}).get("qol_tail")
        if not tail:
            continue
        b = tail["bias"]
        lines.append(
            f"  qol tail [{label}] (advisory) factor={tail['schedule_factor_range']} "
            f"tail>W={_fmt(tail['tail_above_w_share'], 4)} bias all={_fmt(b['all'])} "
            f"tail-weight<={tail['tail_weight_cut']}={_fmt(b['tail_weight_le_cut'])}"
        )
        for world, w in tail["worlds"].items():
            lines.append(
                f"    world {world} factor={_fmt(w['schedule_factor'])} "
                f"tail>W={_fmt(w['tail_above_w_share'], 4)} "
                f"max tail/W={_fmt(w['max_tail_over_w'])} "
                f"tail weight mean={_fmt(w['tail_weight_mean'])}"
            )
    for name, gap in (report.get("generalisation_gap") or {}).items():
        lines.append(
            f"  gap {name:<17} nMAE={_fmt(gap['nmae_cal_minus_fit'])} "
            f"bias={_fmt(gap['bias_cal_minus_fit'])} "
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


class _PhaseCounts:
    """Decisions, initial rejections and retries of the turns one controller decided."""

    def __init__(self) -> None:
        self.decisions = 0
        self.initial_rejections = 0
        self.retries = 0

    def add(self, execution: Any) -> None:
        self.decisions += 1
        self.initial_rejections += sum(
            1 for a in execution.initial_attempts.values() if not a.valid
        )
        self.retries += len(execution.retry)

    def state_dict(self) -> dict[str, int]:
        return {
            "decisions": self.decisions,
            "initial_rejections": self.initial_rejections,
            "retries": self.retries,
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


class _H12Tally:
    """Per-head fallback counts of the converted heuristic labels (``h12_fallbacks``)."""

    def __init__(self) -> None:
        from collections import Counter

        self.decisions = 0
        self.activity_roles = 0
        self.fallback: Any = Counter()
        self.reasons: Any = Counter()

    def add(self, label: Any) -> None:
        self.decisions += 1
        for reason in label.reasons:
            self.reasons[reason.split(":", 1)[0]] += 1
        self.fallback["specialty"] += int(not label.specialty_valid)
        # Only staffed roles carry an activity label; unstaffed roles are
        # role-idle by construction, not a fallback.
        roles = np.asarray(label.activity_role_valid, dtype=bool).reshape(-1)
        staffed = np.asarray(label.specialty_counts).reshape(-1)[1 : 1 + roles.size] > 0
        self.activity_roles += int(staffed.sum())
        self.fallback["activity_role"] += int((staffed & ~roles).sum())
        self.fallback["over_budget"] += int(bool(label.over_budget))
        self.fallback["movement"] += int(not label.movement_valid)
        self.fallback["fission"] += int(not label.fission_valid)
        gates = np.asarray(label.trade_gate_valid, dtype=bool).reshape(-1)
        self.fallback["trade_gate"] += int(not bool(np.all(gates)))
        self.fallback["full_action"] += int(not label.full_action_valid)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "activity_roles": self.activity_roles,
            "counts": {k: int(self.fallback[k]) for k in H12_FALLBACK_HEADS},
            "reasons": dict(sorted(self.reasons.items())),
        }


def _labor_episode(
    bridge: Any,
    world: int,
    options: Any,
    turns: int,
    decide: Any,
) -> tuple[Any, list[dict[str, Any]], Mapping[str, Any]]:
    """One closed-loop episode on an open labor bridge.

    ``decide(decision, step, live)`` makes decision ``decision`` (0-based) for every
    live camp and returns ``(execution, catalogs)``. Returns the reset identity, the
    per-turn records and the final step.
    """
    from .labor_bridge import SPECIALTIES

    farmer = SPECIALTIES.index(FARMER)
    records: list[dict[str, Any]] = []
    step = bridge.reset(world, **options.reset_kwargs())
    identity = bridge.last_reset_identity
    terminated: set[int] = set()
    nutrition: dict[int, float] = {}
    _track(step, terminated, nutrition)
    for decision in range(int(turns)):
        live = _live(step)
        if not live:
            break
        execution, catalogs = decide(decision, step, live)
        next_step = execution.step
        farmers = sum(
            int(a.accepted_specialty_residents[farmer]) for a in execution.final_attempts.values()
        )
        population = sum(int(catalogs[c].population) for c in catalogs)
        records.append(_turn_record(next_step, farmers, population, terminated, nutrition))
        _track(next_step, terminated, nutrition)
        step = next_step
    return identity, records, step


class _ActorController:
    """The actor decides every live camp (``actor_labor_decision``), one turn at a time."""

    def __init__(
        self,
        actor: Any,
        normalizer: Any,
        personality: Any,
        *,
        stochastic: bool,
        generator: torch.Generator,
        recorder: Any = None,
    ) -> None:
        self.actor = actor
        self.normalizer = normalizer
        self.personality = personality
        self.stochastic = stochastic
        self.generator = generator
        self.recorder = recorder
        self.counts = _PhaseCounts()

    def decide(
        self, bridge: Any, step: Mapping[str, Any], live: Sequence[Mapping[str, Any]]
    ) -> tuple[Any, dict[int, Any]]:
        from .labor_model import labor_policy_tensors

        contexts, tensors = labor_policy_tensors(
            live, self.normalizer, personality=self.personality
        )
        catalogs = {context[0].camp_id: context[0] for context in contexts}
        values = None if self.recorder is None else self.recorder.predict(live, tensors)
        execution = actor_labor_decision(
            bridge,
            self.actor,
            self.normalizer,
            self.personality,
            live,
            catalogs,
            tensors,
            stochastic=self.stochastic,
            generator=self.generator,
        )
        if self.recorder is not None:
            self.recorder.record(
                step=step, live=live, tensors=tensors, values=values, execution=execution
            )
        self.counts.add(execution)
        return execution, catalogs


def _actor_controller(
    job: Mapping[str, Any],
    state: Any,
    actor: Any,
    normalizer: Any,
    world: int,
    *,
    stochastic: bool,
    recorder: Any = None,
) -> _ActorController:
    """The actor modes' controller: generator seeded from policy seed and world, the
    checkpoint's personality resolved for ``world``."""
    from .labor_model import labor_checkpoint_personality
    from .labor_personality import personality_for_world

    generator = torch.Generator().manual_seed(int(job["policy_seed"]) * 1_000_003 + world)
    personality = personality_for_world(labor_checkpoint_personality(state), world)
    return _ActorController(
        actor,
        normalizer,
        personality,
        stochastic=stochastic,
        generator=generator,
        recorder=recorder,
    )


class _H12Controller:
    """The converted heuristic recommendation (h12) decides every live camp.

    Mirrors ``labor_bc.collect_heuristic_world``: capture the labor bridge snapshot,
    step the shadow heuristic bridge ``heur`` from it, convert each camp's decision
    with ``convert_heuristic_labor_label`` (whose ``action`` already carries the
    collector's per-head fallbacks), submit it with ``_retained_action`` and retry
    rejections with the all-idle ``_idle_retry``.
    """

    def __init__(
        self, heur: Any, world: int, options: Any, normalizer: Any, personality: Any
    ) -> None:
        self.heur = heur
        self.world = int(world)
        self.options = options
        self.normalizer = normalizer
        self.personality = personality
        self.counts = _PhaseCounts()
        self.tally = _H12Tally()

    def labels(
        self, bridge: Any, live: Sequence[Mapping[str, Any]], contexts: Sequence[Any]
    ) -> list[Any]:
        """The converted heuristic label of every live camp (``live`` order)."""
        from .labor_bc import convert_heuristic_labor_label

        snap = bridge._bridge.capture()
        self.heur.reset(
            self.world,
            movement=self.options.movement,
            withheld=self.options.withheld,
            snapshot=snap,
        )
        h_step = self.heur.step({})
        h_villages = {int(v["camp"]): v for v in h_step.get("villages", [])}
        econ = h_step.get("economicDiagnostics") or {}
        labels = []
        for village, context in zip(live, contexts, strict=True):
            camp = int(village["camp"])
            label = convert_heuristic_labor_label(
                context[0],
                h_villages.get(camp, {}),
                econ.get(str(camp)),
                pre_observation=village["observation"],
            )
            self.tally.add(label)
            labels.append(label)
        return labels

    def decide(
        self, bridge: Any, step: Mapping[str, Any], live: Sequence[Mapping[str, Any]]
    ) -> tuple[Any, dict[int, Any]]:
        del step
        from .labor_bc import _idle_retry, _retained_action
        from .labor_model import labor_policy_tensors
        from .labor_ppo import execute_labor_decision

        contexts, tensors = labor_policy_tensors(
            live, self.normalizer, personality=self.personality
        )
        catalogs = {context[0].camp_id: context[0] for context in contexts}
        labels = self.labels(bridge, live, contexts)
        retained = {}
        for row, (village, context, label) in enumerate(zip(live, contexts, labels, strict=True)):
            retained[int(village["camp"])] = _retained_action(
                label, context[0], tensors["movement_mask"][row : row + 1]
            )
        execution = execute_labor_decision(bridge, catalogs, retained, _idle_retry)
        self.counts.add(execution)
        return execution, catalogs


def _h12_controller(heur: Any, world: int, options: Any, normalizer: Any) -> _H12Controller:
    from .labor_model import LaborPersonality
    from .labor_personality import personality_for_world

    return _H12Controller(
        heur, world, options, normalizer, personality_for_world(LaborPersonality(), world)
    )


# ---------------------------------------------------------------------------
# Diagnostic modes: takeover:K and swap:GROUPS
# ---------------------------------------------------------------------------

LEGACY_AUTONOMOUS_MODES = ("deterministic", "stochastic", "h12")
TAKEOVER_PREFIX = "takeover:"
SWAP_PREFIX = "swap:"
# Schema-12 payload keys per swappable head group (labor_ppo.LABOR_ACTION_FIELDS
# partitioned; "schemaVersion" is shared and must agree between the two sources).
SWAP_GROUP_KEYS: dict[str, tuple[str, ...]] = {
    "specialty": ("laborSpecialtyAllocation",),
    "activity": ("laborActivityAllocation",),
    "movement": ("movement",),
    "fission": ("fission",),
    "trade": ("tradeValue", "tradeGate", "tradeShare"),
}
SWAP_COMPOSITE_GROUPS: dict[str, tuple[str, ...]] = {"labor": ("specialty", "activity")}
SWAP_SHARED_KEYS = ("schemaVersion",)


def parse_swap_groups(text: str) -> frozenset[str]:
    """``+``-joined head groups -> the atomic groups (``labor`` = specialty+activity).

    Strict: at least one group, no unknown, repeated or overlapping names. An empty
    swap is refused (it is ``h12``)."""
    if not isinstance(text, str) or not text:
        raise ValueError("swap: needs at least one head group (an empty swap is h12)")
    tokens = text.split("+")
    known = (*SWAP_COMPOSITE_GROUPS, *SWAP_GROUP_KEYS)
    atoms: list[str] = []
    for token in tokens:
        if token not in known:
            raise ValueError(f"unknown swap head group {token!r}; expected '+'-joined {known}")
        atoms.extend(SWAP_COMPOSITE_GROUPS.get(token, (token,)))
    if len(set(atoms)) != len(atoms):
        raise ValueError(f"swap head groups {text!r} repeat or overlap")
    return frozenset(atoms)


def parse_autonomous_mode(mode: str) -> tuple[str, Any]:
    """Validate one ``--modes`` entry: ``(kind, parameter)``.

    ``deterministic``/``stochastic``/``h12`` -> ``(mode, None)``; ``takeover:K``
    (K a canonical nonnegative integer) -> ``("takeover", K)``; ``swap:GROUPS``
    -> ``("swap", frozenset of atomic groups)``."""
    if mode in LEGACY_AUTONOMOUS_MODES:
        return mode, None
    if mode.startswith(TAKEOVER_PREFIX):
        text = mode[len(TAKEOVER_PREFIX) :]
        if re.fullmatch(r"0|[1-9][0-9]*", text) is None:
            raise ValueError(f"takeover mode {mode!r} needs a canonical nonnegative integer K")
        return "takeover", int(text)
    if mode.startswith(SWAP_PREFIX):
        return "swap", parse_swap_groups(mode[len(SWAP_PREFIX) :])
    raise ValueError(
        f"unknown autonomous mode {mode!r}: expected deterministic, stochastic, h12, "
        "takeover:K or swap:GROUPS"
    )


def _check_swap_key_partition() -> frozenset[str]:
    from .labor_ppo import LABOR_ACTION_FIELDS

    keys = [key for group in SWAP_GROUP_KEYS.values() for key in group]
    keys.extend(SWAP_SHARED_KEYS)
    if len(keys) != len(set(keys)) or set(keys) != set(LABOR_ACTION_FIELDS):
        raise RuntimeError(
            f"swap head-group keys {sorted(keys)} no longer partition the schema-12 "
            f"action fields {sorted(LABOR_ACTION_FIELDS)}"
        )
    return frozenset(LABOR_ACTION_FIELDS)


def _role_active(row: Sequence[Any]) -> bool:
    """Whether an activity row puts any share on a non-idle activity (index > 0)."""
    return any(float(value) > 0.0 for value in list(row)[1:])


def merge_action_payloads(
    h12: Mapping[str, Any],
    actor: Mapping[str, Any],
    groups: frozenset[str] | set[str],
    *,
    population: int,
) -> tuple[dict[str, Any], list[int]]:
    """H12's schema-12 payload with ``groups`` overwritten from the actor's payload.

    Returns ``(merged, substituted_roles)``. When exactly one of specialty/activity
    is swapped, a role (activity row ``r`` = specialty row ``r + 1``) that the
    specialty source staffs (at least one resident under the native largest-remainder
    apportionment of its specialty shares over ``population``) but whose activity row
    in the activity source is role-idle (no share on any non-idle activity, e.g. h12
    left the role unstaffed) takes its activity row from the specialty source, if that
    row is not itself role-idle. Those roles are returned. Unexpected or missing keys,
    or a disagreeing ``schemaVersion``, raise.
    """
    from .labor_bridge import apportion_largest_remainder

    expected = _check_swap_key_partition()
    for name, payload in (("h12", h12), ("actor", actor)):
        if not isinstance(payload, Mapping) or set(payload) != expected:
            got = set(payload) if isinstance(payload, Mapping) else set()
            raise ValueError(
                f"{name} payload keys differ from the schema-12 fields: unexpected "
                f"{sorted(got - expected)}, missing {sorted(expected - got)}"
            )
    for key in SWAP_SHARED_KEYS:
        if h12[key] != actor[key]:
            raise ValueError(f"h12 and actor payloads disagree on shared key {key!r}")
    unknown = set(groups) - set(SWAP_GROUP_KEYS)
    if unknown:
        raise ValueError(f"unknown swap head groups {sorted(unknown)}")
    merged = copy.deepcopy(dict(h12))
    for group in sorted(groups):
        for key in SWAP_GROUP_KEYS[group]:
            merged[key] = copy.deepcopy(actor[key])
    substituted: list[int] = []
    if ("specialty" in groups) != ("activity" in groups) and int(population) > 0:
        spec_source = actor if "specialty" in groups else h12
        shares = tuple(float(x) for x in spec_source["laborSpecialtyAllocation"])
        staffed = apportion_largest_remainder(shares, int(population))
        rows = [list(row) for row in merged["laborActivityAllocation"]]
        source_rows = spec_source["laborActivityAllocation"]
        for role, row in enumerate(rows):
            if role + 1 >= len(staffed) or staffed[role + 1] <= 0 or _role_active(row):
                continue
            if _role_active(source_rows[role]):
                rows[role] = copy.deepcopy(list(source_rows[role]))
                substituted.append(role)
        if substituted:
            merged["laborActivityAllocation"] = rows
    return merged, substituted


def actor_deterministic_payloads(
    actor: Any, tensors: Mapping[str, torch.Tensor], rows: int
) -> list[dict[str, Any]]:
    """The actor's deterministic schema-12 payload per row, exactly as the initial
    action of ``actor_labor_decision(stochastic=False)``."""
    from .labor_model import deterministic_nonlabor_policy, schema12_initial_action

    with torch.no_grad():
        output = actor(tensors)
        labor = _deterministic_labor_sample(output, tensors)
        nonlabor = deterministic_nonlabor_policy(output)
    return [schema12_initial_action(labor, nonlabor, row) for row in range(rows)]


class _SwapController:
    """h12 decides every camp, but the swapped head groups come from the actor's
    deterministic decision on the same state (``merge_action_payloads``); rejections
    retry with the all-idle ``_idle_retry`` as h12."""

    def __init__(
        self, h12: _H12Controller, actor: _ActorController, groups: frozenset[str]
    ) -> None:
        from collections import Counter

        if not groups:
            raise ValueError("swap needs at least one head group (an empty swap is h12)")
        self.h12 = h12
        self.actor = actor
        self.groups = frozenset(groups)
        self.counts = _PhaseCounts()
        self.substituted_roles = 0
        self.decisions_with_substitution = 0
        self.group_differs: Any = Counter()

    def decide(
        self, bridge: Any, step: Mapping[str, Any], live: Sequence[Mapping[str, Any]]
    ) -> tuple[Any, dict[int, Any]]:
        del step
        from .labor_bc import _idle_retry, _retained_payload_action
        from .labor_model import labor_policy_tensors
        from .labor_ppo import execute_labor_decision

        h = self.h12
        contexts, tensors = labor_policy_tensors(live, h.normalizer, personality=h.personality)
        catalogs = {context[0].camp_id: context[0] for context in contexts}
        labels = h.labels(bridge, live, contexts)
        _, actor_tensors = labor_policy_tensors(
            live, self.actor.normalizer, personality=self.actor.personality
        )
        actor_payloads = actor_deterministic_payloads(self.actor.actor, actor_tensors, len(live))
        retained = {}
        for row, (village, context, label) in enumerate(zip(live, contexts, labels, strict=True)):
            catalog = context[0]
            merged, substituted = merge_action_payloads(
                label.action, actor_payloads[row], self.groups, population=int(catalog.population)
            )
            self.substituted_roles += len(substituted)
            self.decisions_with_substitution += int(bool(substituted))
            for group in self.groups:
                self.group_differs[group] += int(
                    any(label.action[k] != actor_payloads[row][k] for k in SWAP_GROUP_KEYS[group])
                )
            retained[int(village["camp"])] = _retained_payload_action(
                merged, catalog, tensors["movement_mask"][row : row + 1]
            )
        execution = execute_labor_decision(bridge, catalogs, retained, _idle_retry)
        self.counts.add(execution)
        return execution, catalogs

    def state_dict(self) -> dict[str, Any]:
        return {
            "groups": sorted(self.groups),
            "decisions": self.counts.decisions,
            "activity_row_substitutions": self.substituted_roles,
            "decisions_with_activity_row_substitution": self.decisions_with_substitution,
            "decisions_where_swapped_group_differs": {
                g: int(self.group_differs[g]) for g in sorted(self.groups)
            },
            "h12_fallbacks": self.h12.tally.state_dict(),
            "note": "specialty and activity from different sources: a role the specialty "
            "source staffs (>=1 resident, native largest-remainder apportionment over the "
            "catalog population) whose activity row in the activity source is role-idle "
            "takes its activity row from the specialty source (activity_row_substitutions)",
        }


def _run_actor(job: Mapping[str, Any]) -> dict[str, Any]:
    from .labor_bc import LaborResetOptions
    from .labor_bridge import LaborModel9Bridge
    from .labor_model import labor_policy_tensors

    torch.set_num_threads(1)
    state, actor, task, costs, normalizer = _load_models(Path(job["checkpoint"]))
    world = int(job["world"])
    stochastic = job["mode"] == "stochastic"
    recorder = None
    if job.get("record_critic_trace"):
        # T4 (r5 section 4a): per-decision critic predictions and realised rewards.
        # Off by default; read-only (the critic never influences the actor's action).
        from .labor_r5_trajectory import CriticTraceRecorder
        from .labor_value_calibration import labor_critic_digest

        recorder = CriticTraceRecorder(task, costs, critic_digest=labor_critic_digest(task, costs))
    controller = _actor_controller(
        job, state, actor, normalizer, world, stochastic=stochastic, recorder=recorder
    )
    options = LaborResetOptions.from_mapping(job["reset_options"])
    with LaborModel9Bridge() as bridge:
        identity, turns, step = _labor_episode(
            bridge,
            world,
            options,
            int(job["turns"]),
            lambda decision, step, live: controller.decide(bridge, step, live),
        )
        if recorder is not None:
            final_live = _live(step)
            final_tensors = (
                labor_policy_tensors(final_live, normalizer, personality=controller.personality)[1]
                if final_live
                else None
            )
            recorder.finish(step, final_live, final_tensors)
    result = {
        "world": world,
        "mode": job["mode"],
        "reset_identity": identity,
        "turns": turns,
        "initial_rejections": controller.counts.initial_rejections,
        "retries": controller.counts.retries,
    }
    if recorder is not None:
        result["critic_trace"] = recorder.trace()
    return result


def _run_h12(job: Mapping[str, Any]) -> dict[str, Any]:
    """Heuristic recommendation converted to action 12, every eligible camp
    (``_H12Controller``, mirroring ``labor_bc.collect_heuristic_world``)."""
    from .bridge import Model9Bridge
    from .labor_bc import LaborResetOptions
    from .labor_bridge import LaborModel9Bridge

    torch.set_num_threads(1)
    _, _, _, _, normalizer = _load_models(Path(job["checkpoint"]))
    world = int(job["world"])
    options = LaborResetOptions.from_mapping(job["reset_options"])
    with LaborModel9Bridge() as bridge, Model9Bridge(economic_diagnostics=True) as heur:
        controller = _h12_controller(heur, world, options, normalizer)
        identity, turns, _ = _labor_episode(
            bridge,
            world,
            options,
            int(job["turns"]),
            lambda decision, step, live: controller.decide(bridge, step, live),
        )
    return {
        "world": world,
        "mode": "h12",
        "reset_identity": identity,
        "turns": turns,
        "initial_rejections": controller.counts.initial_rejections,
        "retries": controller.counts.retries,
        "h12_fallbacks": controller.tally.state_dict(),
    }


def _run_takeover(job: Mapping[str, Any]) -> dict[str, Any]:
    """``takeover:K``: the actor (deterministic decoding, exactly as mode
    ``deterministic``) decides every live camp for decisions 0..K-1, then h12
    (exactly as mode ``h12``) from decision K on, continuing the same episode.
    ``takeover:0`` is h12 and ``takeover:K`` with K >= turns is deterministic."""
    from .bridge import Model9Bridge
    from .labor_bc import LaborResetOptions
    from .labor_bridge import LaborModel9Bridge

    kind, handover = parse_autonomous_mode(job["mode"])
    if kind != "takeover":
        raise ValueError(f"not a takeover mode: {job['mode']!r}")
    if job.get("record_critic_trace"):
        raise ValueError("takeover modes do not record critic traces")
    torch.set_num_threads(1)
    state, actor, _, _, normalizer = _load_models(Path(job["checkpoint"]))
    world = int(job["world"])
    actor_controller = _actor_controller(job, state, actor, normalizer, world, stochastic=False)
    options = LaborResetOptions.from_mapping(job["reset_options"])
    with LaborModel9Bridge() as bridge, Model9Bridge(economic_diagnostics=True) as heur:
        h12 = _h12_controller(heur, world, options, normalizer)

        def decide(decision: int, step: Mapping[str, Any], live: Any) -> Any:
            controller = actor_controller if decision < handover else h12
            return controller.decide(bridge, step, live)

        identity, turns, _ = _labor_episode(bridge, world, options, int(job["turns"]), decide)
    actor_phase = actor_controller.counts.state_dict()
    h12_phase = dict(h12.counts.state_dict(), h12_fallbacks=h12.tally.state_dict())
    return {
        "world": world,
        "mode": job["mode"],
        "reset_identity": identity,
        "turns": turns,
        "handover_turn": int(handover),
        "initial_rejections": actor_phase["initial_rejections"] + h12_phase["initial_rejections"],
        "retries": actor_phase["retries"] + h12_phase["retries"],
        "actor_phase": actor_phase,
        "h12_phase": h12_phase,
    }


def _run_swap(job: Mapping[str, Any]) -> dict[str, Any]:
    """``swap:GROUPS``: h12 decides every camp each turn, with the named head groups
    of each camp's action taken from the actor's deterministic decision on the same
    state (``merge_action_payloads``)."""
    from .bridge import Model9Bridge
    from .labor_bc import LaborResetOptions
    from .labor_bridge import LaborModel9Bridge

    kind, groups = parse_autonomous_mode(job["mode"])
    if kind != "swap":
        raise ValueError(f"not a swap mode: {job['mode']!r}")
    if job.get("record_critic_trace"):
        raise ValueError("swap modes do not record critic traces")
    torch.set_num_threads(1)
    state, actor, _, _, normalizer = _load_models(Path(job["checkpoint"]))
    world = int(job["world"])
    actor_controller = _actor_controller(job, state, actor, normalizer, world, stochastic=False)
    options = LaborResetOptions.from_mapping(job["reset_options"])
    with LaborModel9Bridge() as bridge, Model9Bridge(economic_diagnostics=True) as heur:
        h12 = _h12_controller(heur, world, options, normalizer)
        swap = _SwapController(h12, actor_controller, groups)
        identity, turns, _ = _labor_episode(
            bridge,
            world,
            options,
            int(job["turns"]),
            lambda decision, step, live: swap.decide(bridge, step, live),
        )
    return {
        "world": world,
        "mode": job["mode"],
        "reset_identity": identity,
        "turns": turns,
        "initial_rejections": swap.counts.initial_rejections,
        "retries": swap.counts.retries,
        "swap": swap.state_dict(),
    }


def _run_job(job: Mapping[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    kind = "heuristic" if job["mode"] == "heuristic" else parse_autonomous_mode(job["mode"])[0]
    if kind == "heuristic":
        result = _run_heuristic(job)
    elif kind == "h12":
        result = _run_h12(job)
    elif kind == "takeover":
        result = _run_takeover(job)
    elif kind == "swap":
        result = _run_swap(job)
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


def validate_autonomous_modes(text: str) -> list[str]:
    """``--modes``: comma-separated deterministic, stochastic, h12, takeover:K and/or
    swap:GROUPS (each validated by ``parse_autonomous_mode``), no repeats. The mode
    strings are kept verbatim."""
    modes = [m.strip() for m in str(text).split(",") if m.strip()]
    if not modes:
        raise ValueError(
            "--modes must list deterministic, stochastic, h12, takeover:K and/or swap:GROUPS"
        )
    for mode in modes:
        parse_autonomous_mode(mode)
    if len(set(modes)) != len(modes):
        raise ValueError(f"--modes repeats a mode: {modes}")
    return modes


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
    modes = validate_autonomous_modes(args.modes)
    if getattr(args, "record_critic_trace", False) and any(
        m not in LEGACY_AUTONOMOUS_MODES for m in modes
    ):
        raise ValueError("--record-critic-trace supports only deterministic, stochastic and h12")
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
        "record_critic_trace": bool(getattr(args, "record_critic_trace", False)),
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
                "welfare": "sum over settlements and turns of native village welfare (prior "
                "residents "
                "x QoL x years)",
                "person_years": "sum over living settlements of prior residents x years",
                "qol": "welfare / person_years",
                "starvation_extinction": "newly terminated settlement whose terminal or preceding "
                "nutrition "
                "< "
                f"{STARVATION_NUTRITION}",
                "farmer_share_actor": "accepted Farmer residents (native apportionment) / acting "
                "residents",
                "farmer_share_heuristic": "ceil(assignedFarmerYears / years) / acting residents "
                "(economic "
                "diagnostics)",
                "coverage": "resident-weighted housing/clothing coverage of living settlements "
                "after "
                "the turn",
                "baseline": "native heuristic stepped with empty actions from the actor-labor "
                "reset "
                "snapshot",
            },
            "wall_seconds": time.perf_counter() - started,
            "runs": results,
        }
    )
    if base["record_critic_trace"]:
        from .labor_r5_trajectory import t4_report, traces_from_reports

        report["record_critic_trace"] = True
        report["trajectory_t4"] = t4_report(traces_from_reports([report]))
    return report


def _mode_tag(mode: str) -> str:
    """Short summary label: the first four letters of a legacy mode, else the full mode."""
    return mode[:4] if mode in LEGACY_AUTONOMOUS_MODES else mode


def summarize_autonomous(report: Mapping[str, Any]) -> str:
    lines = [
        f"autonomous worlds={len(report['worlds'])} turns={report['turns']} "
        f"modes={','.join(report['modes'])}"
    ]
    for mode, agg in report["aggregate"].items():
        r = agg["ratios"]
        lines.append(
            f"  {mode:<13} {agg['gate']} welfare={_fmt(r['welfare'])} "
            f"person-years={_fmt(r['person_years'])} "
            f"qol={_fmt(r['qol'])} starvation actor/heur={agg['actor']['starvation_extinctions']}/"
            f"{agg['heuristic']['starvation_extinctions']}"
        )
    for world, entry in report["per_world"].items():
        parts = [f"  world {world}:"]
        for mode in report["modes"]:
            rr = entry[mode]["ratios"]
            parts.append(
                f"{_mode_tag(mode)} W={_fmt(rr['welfare'])} PY={_fmt(rr['person_years'])} "
                f"Q={_fmt(rr['qol'])}"
            )
        parts.append(
            "farmer heur/"
            + "/".join(_mode_tag(m) for m in report["modes"])
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


# ---------------------------------------------------------------------------
# 4. Actor-vs-H12 divergence (where does the BC actor first leave its teacher?)
# ---------------------------------------------------------------------------

DIVERGENCE_SCHEMA = "labor-actor-divergence-v1"
# Mirrors the scale-out scheduler's panel defaults (ops/r5/scaleout/cpu_sched.py: PLAYERS=16,
# WIDTH=128, HEIGHT=64, SEPARATION=9), which are what the panel actually ran with. The
# ``autonomous`` parser's own defaults (32 players, native map) are not the panel's.
SCALEOUT_RESET_DEFAULTS = {"players": 16, "width": 128, "height": 64, "separation": 9}
RESERVED_HELD_OUT_WORLDS = range(90154, 90160)
DIVERGENCE_BASELINE_TURNS = (1, 5)
DIVERGENCE_WINDOW_TURNS = (5, 25)
DIVERGENCE_EXCEED_FACTOR = 2.0
DIVERGENCE_FLOOR = 0.01
DIVERGENCE_SUSTAIN = 3
ACTIVITY_GROUPS: dict[str, tuple[str, ...]] = {
    "housing": ("goods.housing", "goods.light-housing", "goods.housing-material"),
    "clothing": ("goods.cloth", "goods.animal-fabric"),
    "field": ("field.operation", "field.expansion"),
    "herd": ("herd.management", "herd.expansion"),
    "gathering": ("food.gather",),
    "idle": ("idle",),
}
# head name -> (summary section, metric key) of the per-turn disagreement series.
DIVERGENCE_HEADS: dict[str, tuple[str, str]] = {
    "specialty": ("specialty", "l1_mean"),
    "activity_l1": ("activity", "l1_mean"),
    "activity_mismatch": ("activity", "mismatch_rate"),
    "movement": ("movement", "disagreement"),
    "fission": ("fission", "disagreement"),
    "trade_gate": ("trade_gate", "disagreement"),
}


class TeacherChoice:
    """H12's converted label on one camp state, reduced to arrays and validity flags.

    ``specialty_counts`` has one entry per specialty row (index 0 is Idle, index
    ``r + 1`` is role ``r``); ``activity`` is the (roles x activities) share matrix.
    """

    __slots__ = (
        "activity",
        "fission",
        "fission_valid",
        "movement",
        "movement_valid",
        "population",
        "role_valid",
        "specialty_counts",
        "specialty_valid",
        "trade_gate",
        "trade_gate_valid",
    )

    def __init__(
        self,
        *,
        population: int,
        specialty_counts: Any,
        specialty_valid: bool,
        activity: Any,
        role_valid: Any,
        movement: int,
        movement_valid: bool,
        fission: int,
        fission_valid: bool,
        trade_gate: Any,
        trade_gate_valid: Any,
    ) -> None:
        self.population = int(population)
        self.specialty_counts = np.asarray(specialty_counts, dtype=np.float64).reshape(-1)
        self.specialty_valid = bool(specialty_valid)
        self.activity = np.asarray(activity, dtype=np.float64)
        self.role_valid = np.asarray(role_valid, dtype=bool).reshape(-1)
        self.movement = int(movement)
        self.movement_valid = bool(movement_valid)
        self.fission = int(fission)
        self.fission_valid = bool(fission_valid)
        self.trade_gate = np.asarray(trade_gate, dtype=np.int64).reshape(-1)
        self.trade_gate_valid = np.asarray(trade_gate_valid, dtype=bool).reshape(-1)


class ActorChoice:
    """The actor's executed labor decision and its schema-12 non-labor heads for one camp.

    ``specialty_counts`` are the natively accepted (apportioned) residents; ``activity`` is the
    requested activity matrix of the final (post-retry) attempt, ``None`` when that attempt was
    rejected and the native fallback ran (the labor heads are then not comparable).
    """

    __slots__ = (
        "activity",
        "fission",
        "movement",
        "population",
        "specialty_counts",
        "trade_gate",
    )

    def __init__(
        self,
        *,
        population: int,
        specialty_counts: Any,
        activity: Any,
        movement: int,
        fission: int,
        trade_gate: Any,
    ) -> None:
        self.population = int(population)
        self.specialty_counts = np.asarray(specialty_counts, dtype=np.float64).reshape(-1)
        self.activity = None if activity is None else np.asarray(activity, dtype=np.float64)
        self.movement = int(movement)
        self.fission = int(fission)
        self.trade_gate = np.asarray(trade_gate, dtype=np.int64).reshape(-1)


def teacher_from_label(label: Any, population: int) -> TeacherChoice:
    """Reduce a ``HeuristicLaborLabel`` to a ``TeacherChoice`` (validity flags preserved)."""
    action = label.action
    return TeacherChoice(
        population=population,
        specialty_counts=label.specialty_counts,
        specialty_valid=label.specialty_valid,
        activity=label.activity,
        role_valid=label.activity_role_valid,
        movement=label.movement_code,
        movement_valid=label.movement_valid,
        fission=int(action["fission"]),
        fission_valid=label.fission_valid,
        trade_gate=action["tradeGate"],
        trade_gate_valid=label.trade_gate_valid,
    )


def actor_from_execution(
    payload: Mapping[str, Any], final_attempt: Any, population: int
) -> ActorChoice:
    """The actor's choice: initial non-labor payload heads, final accepted labor allocation."""
    usable = bool(final_attempt.valid) and not bool(final_attempt.fallback)
    return ActorChoice(
        population=population,
        specialty_counts=final_attempt.accepted_specialty_residents,
        activity=final_attempt.requested_activity_allocation if usable else None,
        movement=int(payload["movement"]),
        fission=int(payload["fission"]),
        trade_gate=payload["tradeGate"],
    )


def camp_divergence(teacher: TeacherChoice, actor: ActorChoice) -> dict[str, Any]:
    """Per-head disagreement of one actor decision against H12's label on the same state.

    Pure. A head is ``None`` when the H12 label for it is invalid (or the actor's labor
    request was rejected, for the activity heads); invalid labels never count.

    ``specialty``: L1 between specialty share vectors (counts / P, each side its own count
    vector; both sum to 1) and the signed (actor - teacher) share per specialty row.
    ``roles``: per teacher-staffed, valid role, the L1 between activity share rows, whether the
    argmax activities agree, the teacher's argmax activity, and the signed row difference.
    ``resident_time``: share of the camp's residents-time spent on each activity (idle
    included), actor and teacher, taken only when every staffed teacher role is valid.
    """
    out: dict[str, Any] = {
        "population": teacher.population,
        "specialty": None,
        "roles": None,
        "resident_time": None,
        "movement": None,
        "fission": None,
        "trade_gate": None,
        "labor_rejected": actor.activity is None,
    }
    population = max(1, teacher.population)
    t_counts, a_counts = teacher.specialty_counts, actor.specialty_counts
    t_share = t_counts / population
    a_share = a_counts / max(1, actor.population)
    if teacher.specialty_valid:
        out["specialty"] = {
            "l1": float(np.abs(a_share - t_share).sum()),
            "diff": a_share - t_share,
            "actor_share": a_share,
            "teacher_share": t_share,
        }
    if teacher.specialty_valid and actor.activity is not None:
        roles = teacher.activity.shape[0]
        staffed = t_counts[1 : 1 + roles] > 0
        compared = staffed & teacher.role_valid[:roles]
        if compared.any():
            l1 = np.abs(actor.activity - teacher.activity).sum(axis=1)
            actor_arg = actor.activity.argmax(axis=1)
            teacher_arg = teacher.activity.argmax(axis=1)
            out["roles"] = {
                "compared": compared,
                "l1": np.where(compared, l1, 0.0),
                "match": compared & (actor_arg == teacher_arg),
                "teacher_arg": np.where(compared, teacher_arg, -1),
                "diff": np.where(compared[:, None], actor.activity - teacher.activity, 0.0),
            }
        if bool(np.all(teacher.role_valid[:roles] | ~staffed)):
            out["resident_time"] = {
                "actor": (a_counts[1 : 1 + roles, None] * actor.activity).sum(axis=0) / population,
                "teacher": (t_counts[1 : 1 + roles, None] * teacher.activity).sum(axis=0)
                / population,
            }
    if teacher.movement_valid:
        out["movement"] = bool(actor.movement == teacher.movement)
    if teacher.fission_valid:
        out["fission"] = {
            "agree": bool(actor.fission == teacher.fission),
            "actor_nonzero": bool(actor.fission != 0),
            "teacher_nonzero": bool(teacher.fission != 0),
        }
    groups = min(teacher.trade_gate.size, actor.trade_gate.size)
    if groups:
        valid = teacher.trade_gate_valid[:groups]
        out["trade_gate"] = {
            "valid": valid,
            "agree": valid & (actor.trade_gate[:groups] == teacher.trade_gate[:groups]),
        }
    return out


class DivergenceAccumulator:
    """Mergeable sums of ``camp_divergence`` results (one per turn per world, or pooled)."""

    def __init__(self) -> None:
        self.sums: dict[str, np.ndarray] = {}

    def _add(self, key: str, value: Any) -> None:
        value = np.asarray(value, dtype=np.float64)
        if key in self.sums:
            self.sums[key] = self.sums[key] + value
        else:
            self.sums[key] = value.copy()

    def add(self, result: Mapping[str, Any], context: Mapping[str, float] | None = None) -> None:
        population = float(result["population"])
        self._add("camps", 1.0)
        self._add("labor_rejected", float(bool(result["labor_rejected"])))
        if (spec := result["specialty"]) is not None:
            self._add("spec_n", 1.0)
            self._add("spec_pop", population)
            self._add("spec_l1", spec["l1"])
            self._add("spec_l1_pop", spec["l1"] * population)
            self._add("spec_diff", spec["diff"])
            self._add("spec_actor", spec["actor_share"])
            self._add("spec_teacher", spec["teacher_share"])
        if (roles := result["roles"]) is not None:
            self._add("role_n", roles["compared"].astype(np.float64))
            self._add("role_l1", roles["l1"])
            self._add("role_match", roles["match"].astype(np.float64))
            self._add("role_diff", roles["diff"])
            for activity in range(roles["diff"].shape[1]):
                hit = (roles["teacher_arg"] == activity) & roles["compared"]
                self._add(f"argact_n_{activity}", float(hit.sum()))
                self._add(f"argact_match_{activity}", float((hit & roles["match"]).sum()))
        if (rt := result["resident_time"]) is not None:
            self._add("rt_n", 1.0)
            self._add("rt_actor", rt["actor"])
            self._add("rt_teacher", rt["teacher"])
        if (move := result["movement"]) is not None:
            self._add("move_n", 1.0)
            self._add("move_agree", float(move))
        if (fission := result["fission"]) is not None:
            self._add("fission_n", 1.0)
            self._add("fission_agree", float(fission["agree"]))
            self._add("fission_actor_nz", float(fission["actor_nonzero"]))
            self._add("fission_teacher_nz", float(fission["teacher_nonzero"]))
        if (gate := result["trade_gate"]) is not None:
            self._add("gate_n", gate["valid"].astype(np.float64))
            self._add("gate_agree", gate["agree"].astype(np.float64))
        if context:
            for key, value in context.items():
                self._add(f"ctx_{key}", float(value))

    def merge(self, other: DivergenceAccumulator) -> DivergenceAccumulator:
        merged = DivergenceAccumulator()
        for source in (self, other):
            for key, value in source.sums.items():
                merged._add(key, value)
        return merged

    def get(self, key: str, default: float = 0.0) -> Any:
        return self.sums.get(key, np.asarray(default, dtype=np.float64))


def merge_accumulators(items: Sequence[DivergenceAccumulator]) -> DivergenceAccumulator:
    total = DivergenceAccumulator()
    for item in items:
        total = total.merge(item)
    return total


def _rate(num: Any, den: Any) -> float | None:
    den = float(den)
    return float(num) / den if den > 0 else None


def divergence_summary(acc: DivergenceAccumulator) -> dict[str, Any]:
    """Means/rates from accumulated sums; ``None`` where no valid H12 label contributed."""
    from .labor_bridge import ACTIVITIES, SPECIALTIES

    g = acc.get
    spec_n = float(g("spec_n"))
    specialty: dict[str, Any] = {
        "n": int(spec_n),
        "l1_mean": _rate(g("spec_l1"), spec_n),
        "l1_resident_weighted": _rate(g("spec_l1_pop"), g("spec_pop")),
        "per_specialty": {},
    }
    if spec_n > 0:
        for row, name in enumerate(SPECIALTIES):
            specialty["per_specialty"][name] = {
                "signed_diff": float(g("spec_diff")[row]) / spec_n,
                "actor_share": float(g("spec_actor")[row]) / spec_n,
                "teacher_share": float(g("spec_teacher")[row]) / spec_n,
            }
    role_n = np.atleast_1d(g("role_n"))
    total_roles = float(role_n.sum())
    activity: dict[str, Any] = {
        "n_role_decisions": int(total_roles),
        "l1_mean": _rate(np.atleast_1d(g("role_l1")).sum(), total_roles),
        "match_rate": _rate(np.atleast_1d(g("role_match")).sum(), total_roles),
        "mismatch_rate": None,
        "by_role": {},
        "by_teacher_activity": {},
        "role_activity_signed_diff": {},
        "resident_time": {},
        "resident_time_groups": {},
    }
    if activity["match_rate"] is not None:
        activity["mismatch_rate"] = 1.0 - activity["match_rate"]
    if total_roles > 0:
        role_l1, role_match = np.atleast_1d(g("role_l1")), np.atleast_1d(g("role_match"))
        role_diff = np.atleast_2d(g("role_diff"))
        for role, count in enumerate(role_n):
            if count <= 0:
                continue
            name = SPECIALTIES[role + 1]
            activity["by_role"][name] = {
                "n": int(count),
                "l1_mean": float(role_l1[role] / count),
                "match_rate": float(role_match[role] / count),
            }
            for index, aname in enumerate(ACTIVITIES):
                value = float(role_diff[role, index] / count)
                if abs(value) > 1e-9:
                    activity["role_activity_signed_diff"][f"{name}/{aname}"] = value
        for index, aname in enumerate(ACTIVITIES):
            count = float(g(f"argact_n_{index}"))
            if count > 0:
                activity["by_teacher_activity"][aname] = {
                    "n": int(count),
                    "match_rate": float(g(f"argact_match_{index}")) / count,
                }
    rt_n = float(g("rt_n"))
    if rt_n > 0:
        actor_rt = np.atleast_1d(g("rt_actor")) / rt_n
        teacher_rt = np.atleast_1d(g("rt_teacher")) / rt_n
        activity["resident_time_n"] = int(rt_n)
        for index, aname in enumerate(ACTIVITIES):
            activity["resident_time"][aname] = {
                "actor": float(actor_rt[index]),
                "teacher": float(teacher_rt[index]),
                "signed_diff": float(actor_rt[index] - teacher_rt[index]),
            }
        for group, members in ACTIVITY_GROUPS.items():
            idx = [ACTIVITIES.index(m) for m in members if m in ACTIVITIES]
            activity["resident_time_groups"][group] = {
                "actor": float(actor_rt[idx].sum()),
                "teacher": float(teacher_rt[idx].sum()),
                "signed_diff": float(actor_rt[idx].sum() - teacher_rt[idx].sum()),
            }
    gate_n, gate_agree = np.atleast_1d(g("gate_n")), np.atleast_1d(g("gate_agree"))
    gate_total = float(gate_n.sum())
    move_n, fission_n = float(g("move_n")), float(g("fission_n"))
    movement = {
        "n": int(move_n),
        "agreement": _rate(g("move_agree"), move_n),
        "disagreement": None,
    }
    fission = {
        "n": int(fission_n),
        "agreement": _rate(g("fission_agree"), fission_n),
        "disagreement": None,
        "actor_nonzero_rate": _rate(g("fission_actor_nz"), fission_n),
        "teacher_nonzero_rate": _rate(g("fission_teacher_nz"), fission_n),
    }
    trade = {
        "n": int(gate_total),
        "agreement": _rate(gate_agree.sum(), gate_total),
        "disagreement": None,
        "agreement_by_group": [
            None if n <= 0 else float(a / n) for a, n in zip(gate_agree, gate_n, strict=True)
        ],
    }
    for head in (movement, fission, trade):
        if head["agreement"] is not None:
            head["disagreement"] = 1.0 - head["agreement"]
    camps = float(g("camps"))
    residents = float(g("ctx_residents"))
    context = {
        "camp_decisions": int(camps),
        "labor_rejected_rate": _rate(g("labor_rejected"), camps),
        "residents_per_camp": _rate(residents, camps),
        "housing_coverage": _rate(g("ctx_housing_w"), residents),
        "clothing_coverage": _rate(g("ctx_clothing_w"), residents),
        "farmer_share_actor": _rate(g("ctx_farmer_actor"), g("ctx_farmer_pop")),
        "farmer_share_h12": _rate(g("ctx_farmer_teacher"), g("ctx_farmer_pop")),
        "farmer_share_basis": "residents of camps with a valid H12 specialty label",
    }
    return {
        "context": context,
        "specialty": specialty,
        "activity": activity,
        "movement": movement,
        "fission": fission,
        "trade_gate": trade,
    }


def head_series(per_turn: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[int, float | None]]:
    """The per-turn disagreement of each head, from ``divergence_summary`` records."""
    return {
        head: {turn: summary[section][key] for turn, summary in sorted(per_turn.items())}
        for head, (section, key) in DIVERGENCE_HEADS.items()
    }


def first_exceedance(
    series: Mapping[int, float | None],
    *,
    baseline_turns: tuple[int, int] = DIVERGENCE_BASELINE_TURNS,
    factor: float = DIVERGENCE_EXCEED_FACTOR,
    floor: float = DIVERGENCE_FLOOR,
    sustain: int = DIVERGENCE_SUSTAIN,
) -> dict[str, Any]:
    """First turn after the baseline window whose disagreement exceeds ``factor`` x baseline.

    Baseline is the mean over ``baseline_turns`` (inclusive) of the turns with data; the
    threshold is ``factor * max(baseline, floor)`` so a head that agrees perfectly early does
    not trip on noise. ``first_sustained_turn`` is the first exceeding turn that starts
    ``sustain`` consecutive exceeding turns.
    """
    lo, hi = baseline_turns
    base = [v for t, v in series.items() if lo <= t <= hi and v is not None]
    baseline = float(np.mean(base)) if base else None
    out: dict[str, Any] = {
        "baseline": baseline,
        "threshold": None,
        "first_turn": None,
        "first_sustained_turn": None,
    }
    if baseline is None:
        return out
    threshold = factor * max(baseline, floor)
    out["threshold"] = threshold
    turns = sorted(t for t in series if t > hi)
    above = {t: (series[t] is not None and series[t] > threshold) for t in turns}
    for index, turn in enumerate(turns):
        if above[turn] and out["first_turn"] is None:
            out["first_turn"] = turn
        window = turns[index : index + sustain]
        if (
            out["first_sustained_turn"] is None
            and len(window) == sustain
            and all(above[t] for t in window)
        ):
            out["first_sustained_turn"] = turn
    return out


def _world_block(
    accumulators: Mapping[int, DivergenceAccumulator],
    window: tuple[int, int],
    baseline_turns: tuple[int, int],
) -> dict[str, Any]:
    per_turn = {t: divergence_summary(a) for t, a in sorted(accumulators.items())}
    window_acc = merge_accumulators(
        [a for t, a in accumulators.items() if window[0] <= t <= window[1]]
    )
    baseline_acc = merge_accumulators(
        [a for t, a in accumulators.items() if baseline_turns[0] <= t <= baseline_turns[1]]
    )
    series = head_series(per_turn)
    return {
        "per_turn": {str(t): s for t, s in per_turn.items()},
        "baseline": {"turns": list(baseline_turns), "summary": divergence_summary(baseline_acc)},
        "window": {"turns": list(window), "summary": divergence_summary(window_acc)},
        "heads": {
            head: {
                "series": {str(t): v for t, v in values.items()},
                **first_exceedance(values, baseline_turns=baseline_turns),
            }
            for head, values in series.items()
        },
    }


def divergence_report(
    results: Sequence[Mapping[str, Any]],
    *,
    window: tuple[int, int] = DIVERGENCE_WINDOW_TURNS,
    baseline_turns: tuple[int, int] = DIVERGENCE_BASELINE_TURNS,
) -> dict[str, Any]:
    """Pool per-world results (camp-decision weighted) and keep a copy per world."""
    per_world: dict[str, Any] = {}
    pooled: dict[int, list[DivergenceAccumulator]] = defaultdict(list)
    for result in results:
        accs = result["turn_accumulators"]
        per_world[str(result["world"])] = dict(
            _world_block(accs, window, baseline_turns),
            reset_identity=result.get("reset_identity"),
            actor_turns={str(r["decision"]): r for r in result["actor_turns"]},
            turns_run=len(result["actor_turns"]),
            wall_seconds=result.get("wall_seconds"),
        )
        for turn, acc in accs.items():
            pooled[int(turn)].append(acc)
    merged = {t: merge_accumulators(items) for t, items in pooled.items()}
    report = _world_block(merged, window, baseline_turns)
    # World-balanced window means of the head disagreements.
    balanced: dict[str, Any] = {}
    for head, (section, key) in DIVERGENCE_HEADS.items():
        values = [
            w["window"]["summary"][section][key]
            for w in per_world.values()
            if w["window"]["summary"][section][key] is not None
        ]
        balanced[head] = float(np.mean(values)) if values else None
    report["world_balanced_window_heads"] = balanced
    # Panel-style post-turn context of the actor, averaged over worlds (lines up with the collapse).
    actor_context: dict[str, dict[str, float]] = {}
    for turn in sorted({int(t) for w in per_world.values() for t in w["actor_turns"]}):
        rows = [
            w["actor_turns"][str(turn)] for w in per_world.values() if str(turn) in w["actor_turns"]
        ]
        actor_context[str(turn)] = {
            field: float(np.mean([r[field] for r in rows]))
            for field in (
                "farmer_share",
                "housing_coverage",
                "clothing_coverage",
                "live_settlements",
                "residents",
                "qol",
            )
        }
        actor_context[str(turn)]["worlds"] = len(rows)
    report["actor_turn_world_mean"] = actor_context
    report["per_world"] = per_world
    return report


def _divergence_decision_context(
    village: Mapping[str, Any], teacher: TeacherChoice, actor: ActorChoice, farmer_row: int
) -> dict[str, float]:
    residents = float(max(0, int(village["population"])))
    context = {
        "residents": residents,
        "housing_w": float(village.get("housing", 0.0)) * residents,
        "clothing_w": float(village.get("clothing", 0.0)) * residents,
    }
    if teacher.specialty_valid:
        context["farmer_pop"] = float(teacher.population)
        context["farmer_teacher"] = float(teacher.specialty_counts[farmer_row])
        context["farmer_actor"] = float(actor.specialty_counts[farmer_row])
    return context


def _run_divergence_world(job: Mapping[str, Any]) -> dict[str, Any]:
    """Actor in closed loop; on every actor state also H12's label, never executed."""
    from .bridge import Model9Bridge
    from .labor_bc import LaborResetOptions, convert_heuristic_labor_label
    from .labor_bridge import SPECIALTIES, LaborModel9Bridge
    from .labor_model import labor_checkpoint_personality, labor_policy_tensors
    from .labor_personality import personality_for_world

    torch.set_num_threads(1)
    started = time.perf_counter()
    state, actor_model, _, _, normalizer = _load_models(Path(job["checkpoint"]))
    world = int(job["world"])
    personality = personality_for_world(labor_checkpoint_personality(state), world)
    options = LaborResetOptions.from_mapping(job["reset_options"])
    generator = torch.Generator().manual_seed(int(job.get("policy_seed", 0)) * 1_000_003 + world)
    farmer = SPECIALTIES.index(FARMER)
    accumulators: dict[int, DivergenceAccumulator] = {}
    actor_turns: list[dict[str, Any]] = []
    with LaborModel9Bridge() as bridge, Model9Bridge(economic_diagnostics=True) as heur:
        step = bridge.reset(world, **options.reset_kwargs())
        identity = bridge.last_reset_identity
        terminated: set[int] = set()
        nutrition: dict[int, float] = {}
        _track(step, terminated, nutrition)
        for turn in range(1, int(job["turns"]) + 1):
            live = _live(step)
            if not live:
                break
            contexts, tensors = labor_policy_tensors(live, normalizer, personality=personality)
            catalogs = {context[0].camp_id: context[0] for context in contexts}
            # H12's label on this very state, from a shadow bridge reset to a snapshot of it.
            snapshot = bridge._bridge.capture()
            heur.reset(
                world, movement=options.movement, withheld=options.withheld, snapshot=snapshot
            )
            h_step = heur.step({})
            h_villages = {int(v["camp"]): v for v in h_step.get("villages", [])}
            econ = h_step.get("economicDiagnostics") or {}
            labels = {}
            for village, context in zip(live, contexts, strict=True):
                camp = int(village["camp"])
                labels[camp] = convert_heuristic_labor_label(
                    context[0],
                    h_villages.get(camp, {}),
                    econ.get(str(camp)),
                    pre_observation=village["observation"],
                )
            # Only the actor's decision is executed (deterministic decode).
            execution = actor_labor_decision(
                bridge,
                actor_model,
                normalizer,
                personality,
                live,
                catalogs,
                tensors,
                stochastic=False,
                generator=generator,
            )
            acc = accumulators.setdefault(turn, DivergenceAccumulator())
            for village in live:
                camp = int(village["camp"])
                population = int(catalogs[camp].population)
                teacher = teacher_from_label(labels[camp], population)
                actor_choice = actor_from_execution(
                    execution.initial[camp].payload, execution.final_attempts[camp], population
                )
                acc.add(
                    camp_divergence(teacher, actor_choice),
                    _divergence_decision_context(village, teacher, actor_choice, farmer),
                )
            next_step = execution.step
            farmers = sum(
                int(a.accepted_specialty_residents[farmer])
                for a in execution.final_attempts.values()
            )
            population = sum(int(catalogs[c].population) for c in catalogs)
            record = _turn_record(next_step, farmers, population, terminated, nutrition)
            # ``turn`` is the native clock; ``decision`` is the 1-based index used by the report.
            actor_turns.append(dict(record, decision=turn))
            _track(next_step, terminated, nutrition)
            step = next_step
    result = {
        "world": world,
        "reset_identity": identity,
        "turn_accumulators": accumulators,
        "actor_turns": actor_turns,
        "wall_seconds": time.perf_counter() - started,
    }
    print(
        json.dumps(
            {
                "world": world,
                "mode": "actor-divergence",
                "wall_seconds": round(result["wall_seconds"], 1),
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    return result


def check_divergence_worlds(worlds: Sequence[int], allow_non_development: bool) -> None:
    from .labor_bc import DEVELOPMENT_WORLDS

    _check_world_policy(worlds, "actor-divergence worlds")
    reserved = [w for w in worlds if w in RESERVED_HELD_OUT_WORLDS]
    if reserved:
        raise ValueError(
            f"actor-divergence never uses reserved held-out worlds; got {reserved[:5]}"
        )
    outside = [w for w in worlds if w not in DEVELOPMENT_WORLDS]
    if outside and not allow_non_development:
        raise ValueError(
            f"actor-divergence uses development worlds 2000-2059; got {outside[:5]} "
            "(pass --allow-non-development for other non-sealed, non-reserved worlds)"
        )


def run_divergence(args: argparse.Namespace) -> dict[str, Any]:
    from .labor_bc import LaborResetOptions
    from .protocol import seed_list

    worlds = seed_list(args.worlds)
    check_divergence_worlds(worlds, args.allow_non_development)
    reset = LaborResetOptions(
        players=args.players,
        movement=not args.no_movement,
        withheld=args.withheld,
        width=args.width,
        height=args.height,
        separation=args.separation,
    )
    base = {
        "checkpoint": str(args.checkpoint),
        "turns": int(args.turns),
        "reset_options": reset.state_dict(),
        "policy_seed": 0,
    }
    jobs = [dict(base, world=w) for w in worlds]
    started = time.perf_counter()
    if args.jobs <= 1 or len(jobs) == 1:
        results = [_run_divergence_world(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
            results = list(pool.map(_run_divergence_world, jobs))
    window, baseline = tuple(args.window), tuple(args.baseline_turns)
    report = divergence_report(results, window=window, baseline_turns=baseline)
    report.update(
        {
            "schema": DIVERGENCE_SCHEMA,
            "command": "actor-divergence",
            "checkpoint": str(args.checkpoint),
            "worlds": worlds,
            "turns": int(args.turns),
            "reset_options": reset.state_dict(),
            "mode": "deterministic",
            "exceedance_rule": {
                "baseline_turns": list(baseline),
                "factor": DIVERGENCE_EXCEED_FACTOR,
                "floor": DIVERGENCE_FLOOR,
                "sustain_turns": DIVERGENCE_SUSTAIN,
            },
            "definitions": {
                "turn": "1-based decision index; the actor decides, H12's label on the same state "
                "is computed but never executed",
                "specialty": "L1 between actor (accepted residents / P) and H12 (counts / P) "
                "specialty "
                "share vectors; only camps whose H12 specialty label is valid. "
                "per_specialty.signed_diff = actor - H12 share",
                "activity": "per H12-staffed role with a valid H12 activity label: L1 between "
                "actor "
                "requested and H12 activity share rows, and argmax-activity mismatch; "
                "skipped when the actor's final labor request was rejected "
                "(labor_rejected_rate)",
                "role_activity_signed_diff": "mean (actor - H12) activity share within the role, "
                "Role/activity "
                "keys",
                "resident_time": "fraction of the camp's residents-time on each activity (idle "
                "included), "
                "actor vs H12, camps whose staffed H12 roles are all valid; "
                "resident_time_groups sums housing/clothing/field/herd/gathering/idle "
                "activities",
                "movement/fission/trade_gate": (
                    "disagreement = 1 - agreement of the actor's initial "
                    "schema-12 "
                    "head vs "
                    "the "
                    "H12 label where its *_valid flag holds (trade gate per valid group)"
                ),
                "heads[*].first_turn": "first turn after the baseline window whose pooled "
                "disagreement "
                "exceeds "
                "factor x max(baseline mean over baseline turns, floor); "
                "first_sustained_turn needs sustain consecutive turns",
                "context": "housing/clothing coverage are resident-weighted on the actor's "
                "pre-action "
                "state; actor_turn_world_mean (keyed by decision index, record.turn is the "
                "native clock) is the panel-style post-turn record",
                "pooling": "top-level blocks pool camp decisions across worlds; per_world holds "
                "each "
                "world's own copy; world_balanced_window_heads averages per-world window "
                "values",
            },
            "wall_seconds": time.perf_counter() - started,
        }
    )
    return report


def summarize_divergence(report: Mapping[str, Any]) -> str:
    lines = [
        f"actor-divergence: {len(report['worlds'])} worlds, {report['turns']} turns, "
        f"window {report['window']['turns']}",
        "head                 baseline  threshold  first>2x  sustained  window-mean",
    ]
    for head, record in report["heads"].items():
        section, key = DIVERGENCE_HEADS[head]
        window_value = report["window"]["summary"][section][key]
        lines.append(
            f"{head:<20} {_fmt(record['baseline']):>8}  {_fmt(record['threshold']):>9}  "
            f"{record['first_turn']!s:>8}  {record['first_sustained_turn']!s:>9}  "
            f"{_fmt(window_value):>11}"
        )
    groups = report["window"]["summary"]["activity"].get("resident_time_groups", {})
    for name, group in groups.items():
        lines.append(
            f"resident-time {name:<10} actor {_fmt(group['actor'])} h12 {_fmt(group['teacher'])} "
            f"diff {_fmt(group['signed_diff'])}"
        )
    return "\n".join(lines)


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
    critic.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="critic checkpoint; several are scored as one averaged critic (per-row mean of "
        "the members' predictions in value units; members must share input kind, normalizers, "
        "value support and loader corrections)",
    )
    critic.add_argument(
        "--member-reports",
        action="store_true",
        default=False,
        help="with several checkpoints, also write each member's own evaluation on the same "
        "loaded data (report['member_reports'])",
    )
    critic.add_argument(
        "--tail-normalize",
        choices=("none", "world-factor"),
        default=None,
        help="apply this tail normalisation to --dataset and --fit-dataset whatever the "
        "checkpoint recorded (default: the checkpoint's recorded correction)",
    )
    critic.add_argument(
        "--false-terminal-policy",
        choices=("none", "mask"),
        default=None,
        help="apply this false-terminal policy to --dataset and --fit-dataset whatever the "
        "checkpoint recorded (default: the checkpoint's recorded correction)",
    )
    critic.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        required=True,
        help="calibration dataset(s); several batches are read as one (world order)",
    )
    critic.add_argument(
        "--fit-dataset",
        type=Path,
        nargs="+",
        default=None,
        help="fitting dataset(s) for the generalisation gap; several batches are read as one",
    )
    critic.add_argument(
        "--qol-reference-scale",
        type=Path,
        default=None,
        help="QoL reference-scale artifact (labor_r5_acceptance qol-reference-scale): the unit "
        "of the QoL n-step gate metrics; absent, the frozen G0 scale 43.152",
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
    auto.add_argument(
        "--record-critic-trace",
        action="store_true",
        default=False,
        help="T4: log every critic head's prediction and the realised rewards per actor "
        "decision (runs[*].critic_trace) and add the trajectory_t4 report; off by default",
    )
    auto.add_argument("--out", type=Path, required=True)
    div = commands.add_parser(
        "actor-divergence",
        help="where the BC actor first diverges from H12 in closed loop (per head, per turn)",
    )
    div.add_argument("--checkpoint", type=Path, required=True)
    div.add_argument("--worlds", required=True, help="development seeds/ranges, e.g. 2000-2007")
    div.add_argument("--turns", type=int, default=30)
    div.add_argument("--players", type=int, default=SCALEOUT_RESET_DEFAULTS["players"])
    div.add_argument("--width", type=int, default=SCALEOUT_RESET_DEFAULTS["width"])
    div.add_argument("--height", type=int, default=SCALEOUT_RESET_DEFAULTS["height"])
    div.add_argument("--separation", type=int, default=SCALEOUT_RESET_DEFAULTS["separation"])
    div.add_argument("--withheld", type=int, default=0)
    div.add_argument("--no-movement", action="store_true")
    div.add_argument(
        "--window",
        type=int,
        nargs=2,
        default=list(DIVERGENCE_WINDOW_TURNS),
        metavar=("FIRST", "LAST"),
        help="aggregate turn window (inclusive, 1-based)",
    )
    div.add_argument(
        "--baseline-turns",
        type=int,
        nargs=2,
        default=list(DIVERGENCE_BASELINE_TURNS),
        metavar=("FIRST", "LAST"),
        help="baseline turn window for the 2x first-exceedance rule (inclusive, 1-based)",
    )
    div.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    div.add_argument("--allow-non-development", action="store_true")
    div.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    torch.set_num_threads(4)
    if args.command == "actor-fidelity":
        report = run_actor_fidelity(args.checkpoint, args.dataset, args.batch_size)
        summary = summarize_actor(report)
    elif args.command == "critic-calibration":
        report = run_critic_calibration(
            args.checkpoint,
            args.dataset,
            args.fit_dataset,
            args.batch_size,
            qol_reference=load_qol_reference_scale(args.qol_reference_scale),
            corrections_override={
                "tail_normalize": args.tail_normalize,
                "false_terminal_policy": args.false_terminal_policy,
            },
            member_reports=args.member_reports,
        )
        summary = summarize_critic(report)
    elif args.command == "actor-divergence":
        if args.turns < 1 or args.jobs < 1:
            raise ValueError("turns and jobs must be positive")
        report = run_divergence(args)
        summary = summarize_divergence(report)
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
