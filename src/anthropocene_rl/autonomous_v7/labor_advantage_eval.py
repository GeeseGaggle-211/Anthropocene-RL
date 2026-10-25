"""Preregistered validation of the labor dueling advantage head (intervention B).

Every check is same-root: a village competes only with itself (amendment r5-b).

1. Held-out same-root pairwise ranking on pairs beyond 2 SE, per objective and
   for the personality-weighted sums (each grid personality and the root's own).
2. False predicted improvement: the share of roots whose A-argmax candidate's
   realized return is below H12's by more than the 0.01 normalized margin.
3. Pruning regret on fully evaluated roots: best of all minus best of the top
   four by A (and of the top four plus H12, the set search evaluates).
4. Controls: the all-idle candidate ranks last; the delayed-benefit candidate
   does not rank below H12.
5. Bookkeeping invariance: candidates that differ only in bookkeeping fields
   (data twins and synthetic re-serializations) receive the same A.

Gates: ranking >= 0.70, false improvement <= 0.15. Evaluation is allowed on any
world, including the reserved 90154-90159 (the Q-pi admission set), but never on
a world the head was fitted on.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .labor_advantage import (
    COMPONENTS,
    FALSE_IMPROVEMENT_GATE,
    FALSE_IMPROVEMENT_MARGIN,
    PAIR_SE_MULTIPLIER,
    RANKING_GATE,
    TOP_K,
    CandidateSet,
    LaborAdvantageHead,
    Root,
    _json_digest,
    _value,
    action_features,
    build_pairs,
    critic_contexts,
    load_advantage_head,
    load_candidate_shards,
    predict,
    target_weights,
)
from .labor_bridge import ACTIVITIES, SPECIALTIES, LaborPolicySample
from .labor_privileged import RESERVED_HELDOUT_WORLDS

EVAL_KIND = "labor-advantage-preregistered-validation-v1"
DEFAULT_GRID = ((0.5, 0.5), (0.9, 0.1), (0.1, 0.9))
BOOKKEEPING_TOLERANCE = 1e-4
MIN_PAIRS = 1


def _predicted(
    prediction: Mapping[str, list[np.ndarray]], r: int, weights: Mapping[str, float]
) -> np.ndarray:
    return sum(weights[name] * prediction[name][r] for name in COMPONENTS)


def _bootstrap(per_world: Mapping[int, tuple[float, int]], seed: int = 0) -> list[float] | None:
    worlds = sorted(per_world)
    if len(worlds) < 2:
        return None
    rng = np.random.default_rng([0x4142, seed])
    hits = np.array([per_world[w][0] for w in worlds])
    counts = np.array([per_world[w][1] for w in worlds])
    draws = rng.integers(0, len(worlds), size=(1000, len(worlds)))
    values = hits[draws].sum(axis=1) / np.maximum(1, counts[draws].sum(axis=1))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def pairwise_ranking(
    roots: Sequence[Root],
    prediction: Mapping[str, list[np.ndarray]],
    target: str,
    *,
    horizons: Sequence[int] | None = None,
    se_multiplier: float = PAIR_SE_MULTIPLIER,
    personality: Sequence[float] | str | None = None,
) -> dict[str, Any]:
    """Accuracy of sign(A_i - A_k) against sign(G_i - G_k) on admitted pairs (ties 0.5)."""
    pairs = build_pairs(
        roots, target, horizons=horizons, se_multiplier=se_multiplier, personality=personality
    )
    hits_by_world: dict[int, list[float]] = defaultdict(list)
    by_horizon: dict[int, list[float]] = defaultdict(list)
    for idx in np.flatnonzero(pairs.admitted):
        r = int(pairs.root[idx])
        pers = roots[r].personality if personality == "root" else personality
        predicted = _predicted(prediction, r, target_weights(target, pers))
        diff = float(predicted[pairs.i[idx]] - predicted[pairs.k[idx]])
        hit = 0.5 if diff == 0.0 else float(np.sign(diff) == np.sign(pairs.delta[idx]))
        hits_by_world[roots[r].world].append(hit)
        by_horizon[int(pairs.horizon[idx])].append(hit)
    hits = [h for values in hits_by_world.values() for h in values]
    per_world = {w: (float(sum(v)), len(v)) for w, v in hits_by_world.items()}
    return {
        "accuracy": float(np.mean(hits)) if hits else None,
        "pairs": len(hits),
        "comparable_pairs": len(pairs),
        "reevaluated_pairs": int((pairs.admitted & pairs.reevaluated).sum()),
        "roots": len({int(r) for r in pairs.root[pairs.admitted]}),
        "worlds": sorted(hits_by_world),
        "by_horizon": {
            str(h): {"accuracy": float(np.mean(v)), "pairs": len(v)}
            for h, v in sorted(by_horizon.items())
        },
        "world_bootstrap_95": _bootstrap(per_world),
    }


def _paired_values(
    root: Root, i: int, k: int, horizon: int, weights: Mapping[str, float]
) -> tuple[float, float] | None:
    """Realized values of i and k on a common stream kind (re-evaluated if both have it)."""
    ei, ek = root.candidates[i].evaluations, root.candidates[k].evaluations
    shared = sorted(s for (h, s) in ei if h == horizon and s >= 1 and (horizon, s) in ek)
    if len(shared) >= 2:
        return (
            float(np.mean([_value(ei[(horizon, s)].returns, weights) for s in shared])),
            float(np.mean([_value(ek[(horizon, s)].returns, weights) for s in shared])),
        )
    if (horizon, 0) in ei and (horizon, 0) in ek:
        return _value(ei[(horizon, 0)].returns, weights), _value(ek[(horizon, 0)].returns, weights)
    return None


def false_improvement(
    roots: Sequence[Root],
    prediction: Mapping[str, list[np.ndarray]],
    *,
    personality: Sequence[float] | str = "root",
    margin: float = FALSE_IMPROVEMENT_MARGIN,
) -> dict[str, Any]:
    """Share of roots where the A-argmax candidate realizes below H12 by more than ``margin``.

    Compared at the deepest horizon both were evaluated at, on a common stream
    kind. Roots whose argmax is H12 count as not false.
    """
    false = compared = argmax_h12 = 0
    gaps = []
    for r, root in enumerate(roots):
        pers = root.personality if personality == "root" else personality
        weights = target_weights("utility", pers)
        predicted = _predicted(prediction, r, weights)
        h12 = root.h12
        best = max(range(len(predicted)), key=lambda c: (predicted[c], c == h12, -c))
        if best == h12:
            argmax_h12 += 1
            compared += 1
            continue
        horizons = sorted(
            {h for (h, _) in root.candidates[best].evaluations}
            & {h for (h, _) in root.candidates[h12].evaluations},
            reverse=True,
        )
        values = None
        for horizon in horizons:
            values = _paired_values(root, best, h12, horizon, weights)
            if values is not None:
                break
        if values is None:
            continue
        compared += 1
        gap = values[0] - values[1]
        gaps.append(gap)
        false += int(gap < -margin)
    return {
        "rate": false / compared if compared else None,
        "false": false,
        "roots": compared,
        "argmax_is_h12": argmax_h12,
        "margin": margin,
        "median_gain_vs_h12": float(np.median(gaps)) if gaps else None,
    }


def pruning_regret(
    roots: Sequence[Root],
    prediction: Mapping[str, list[np.ndarray]],
    *,
    personality: Sequence[float] | str = "root",
    top_k: int = TOP_K,
    margin: float = FALSE_IMPROVEMENT_MARGIN,
) -> dict[str, Any]:
    """Best of all minus best of the top-k by A, on fully evaluated roots."""
    regrets, with_h12 = [], []
    for r, root in enumerate(roots):
        if not root.full_evaluation:
            continue
        pers = root.personality if personality == "root" else personality
        weights = target_weights("utility", pers)
        horizons = set.intersection(*[{h for (h, _) in c.evaluations} for c in root.candidates])
        if not horizons:
            continue
        horizon = max(horizons)
        streams = set.intersection(
            *[{s for (h, s) in c.evaluations if h == horizon and s >= 1} for c in root.candidates]
        )
        if len(streams) >= 2:
            realized = [
                float(
                    np.mean([_value(c.evaluations[(horizon, s)].returns, weights) for s in streams])
                )
                for c in root.candidates
            ]
        elif all((horizon, 0) in c.evaluations for c in root.candidates):
            realized = [
                _value(c.evaluations[(horizon, 0)].returns, weights) for c in root.candidates
            ]
        else:
            continue
        predicted = _predicted(prediction, r, weights)
        top = list(np.argsort(-predicted, kind="stable")[:top_k])
        regrets.append(max(realized) - max(realized[c] for c in top))
        with_h12.append(max(realized) - max(realized[c] for c in {*top, root.h12}))

    def stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"roots": 0}
        array = np.asarray(values)
        return {
            "roots": len(values),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.9)),
            "share_above_margin": float((array > margin).mean()),
        }

    return {"top_k": top_k, "top_k_only": stats(regrets), "top_k_plus_h12": stats(with_h12)}


def control_checks(
    roots: Sequence[Root], prediction: Mapping[str, list[np.ndarray]], *, delayed_horizon: int = 3
) -> dict[str, Any]:
    """All-idle must rank strictly last; delayed-benefit must not rank below H12."""
    idle, delayed = [], []
    idle_by_objective: dict[str, list[bool]] = defaultdict(list)
    realized_delayed = []
    for r, root in enumerate(roots):
        weights = target_weights("utility", root.personality)
        predicted = _predicted(prediction, r, weights)
        for c in root.candidates:
            others = [o.index for o in root.candidates if o.index != c.index]
            if c.control == "all-idle":
                idle.append(bool(all(predicted[c.index] < predicted[o] for o in others)))
                for name in ("qol_years", "population_years"):
                    values = prediction[name][r]
                    idle_by_objective[name].append(
                        bool(all(values[c.index] < values[o] for o in others))
                    )
            elif c.control == "delayed-benefit":
                delayed.append(bool(predicted[c.index] >= predicted[root.h12]))
                values = _paired_values(root, c.index, root.h12, delayed_horizon, weights)
                if values is not None:
                    realized_delayed.append(bool(values[0] >= values[1]))

    def rate(values: list[bool]) -> float | None:
        return float(np.mean(values)) if values else None

    return {
        "idle_last": {"rate": rate(idle), "roots": len(idle)},
        "idle_last_by_objective": {k: rate(v) for k, v in idle_by_objective.items()},
        "delayed_not_below_h12": {"rate": rate(delayed), "roots": len(delayed)},
        "delayed_realized_not_below_h12_at_horizon": {
            "horizon": delayed_horizon,
            "rate": rate(realized_delayed),
            "roots": len(realized_delayed),
        },
    }


def _reordered(action: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    """The same action declared under reversed (non-idle) specialty and activity ID orders."""
    specialty_ids = ["idle", *reversed(SPECIALTIES[1:])]
    activity_ids = list(reversed(ACTIVITIES))
    spec = list(action["laborSpecialtyAllocation"])
    acts = [list(row) for row in action["laborActivityAllocation"]]
    new = dict(action)
    new["laborSpecialtyAllocation"] = [spec[SPECIALTIES.index(s)] for s in specialty_ids]
    roles = [s for s in specialty_ids if s != "idle"]
    new["laborActivityAllocation"] = [
        [acts[SPECIALTIES.index(role) - 1][ACTIVITIES.index(a)] for a in activity_ids]
        for role in roles
    ]
    return new, specialty_ids, activity_ids


def bookkeeping_probe(
    head: LaborAdvantageHead,
    contexts: Mapping[str, torch.Tensor],
    roots: Sequence[Root],
    prediction: Mapping[str, list[np.ndarray]],
    *,
    tolerance: float = BOOKKEEPING_TOLERANCE,
) -> dict[str, Any]:
    """Max |delta A| between bookkeeping-only variants (data twins and synthetic ones).

    Synthetic variants of each root's H12 action: canonical re-serialization of
    the labor simplexes, a reversed stable-ID declaration of the same rows, and
    extra non-semantic keys. Raw (uncentred) A is compared for these, so the
    candidate set does not move.
    """
    data_max = 0.0
    data_pairs = 0
    for r, root in enumerate(roots):
        for c in root.candidates:
            if c.twin is not None and c.twin > c.index:
                data_pairs += 1
                for name in COMPONENTS:
                    data_max = max(
                        data_max,
                        abs(float(prediction[name][r][c.index] - prediction[name][r][c.twin])),
                    )
    synthetic_max = 0.0
    for r, root in enumerate(roots):
        action = root.candidates[root.h12].action
        resampled = LaborPolicySample(
            tuple(float(x) for x in action["laborSpecialtyAllocation"]),
            tuple(tuple(float(x) for x in row) for row in action["laborActivityAllocation"]),
        ).payload()
        reserialized = {**action, **resampled}
        reordered, spec_ids, act_ids = _reordered(action)
        noisy = copy.deepcopy(dict(action))
        noisy.update(description="bookkeeping", digest="0" * 64, schemaVersion=12)
        features = np.stack(
            [
                action_features(action),
                action_features(reserialized),
                action_features(reordered, specialty_ids=spec_ids, activity_ids=act_ids),
                action_features(noisy),
            ]
        )
        index = torch.full((len(features),), r, dtype=torch.long)
        with torch.no_grad():
            raw = head(contexts, torch.as_tensor(features, dtype=torch.float32), index)
        for name in COMPONENTS:
            values = raw[name].double()
            synthetic_max = max(synthetic_max, float((values - values[0]).abs().max()))
    worst = max(data_max, synthetic_max)
    return {
        "data_twin_pairs": data_pairs,
        "data_twin_max_abs_delta": data_max,
        "synthetic_max_abs_delta": synthetic_max,
        "tolerance": tolerance,
        "pass": bool(worst <= tolerance),
    }


def _gate(value: float | None, threshold: float, *, above: bool, pairs: int | None = None) -> dict:
    if value is None or (pairs is not None and pairs < MIN_PAIRS):
        return {"value": value, "threshold": threshold, "pass": False, "status": "HOLD"}
    ok = value >= threshold if above else value <= threshold
    return {
        "value": value,
        "threshold": threshold,
        "pass": bool(ok),
        "status": "PASS" if ok else "FAIL",
    }


def evaluate_advantage(
    task_critic: Any,
    cost_critics: Any,
    head: LaborAdvantageHead,
    candidates: CandidateSet,
    *,
    horizons: Mapping[str, Sequence[int] | None],
    fit_worlds: Sequence[int] = (),
    utility_horizons: Sequence[int] | None = None,
    grid: Sequence[Sequence[float]] = DEFAULT_GRID,
    se_multiplier: float = PAIR_SE_MULTIPLIER,
    margin: float = FALSE_IMPROVEMENT_MARGIN,
    top_k: int = TOP_K,
    tolerance: float = BOOKKEEPING_TOLERANCE,
) -> dict[str, Any]:
    """The full preregistered report with gates. Refuses worlds the head was fitted on."""
    worlds = candidates.worlds
    overlap = sorted(set(worlds) & {int(w) for w in fit_worlds})
    if overlap:
        raise ValueError(f"evaluation worlds overlap the advantage fit worlds: {overlap}")
    tensors, privileged = candidates.root_tensors()
    contexts = critic_contexts(task_critic, cost_critics, tensors, privileged)
    roots = candidates.roots
    prediction = predict(head, contexts, roots)
    ranking: dict[str, Any] = {}
    for name in COMPONENTS:
        ranking[name] = pairwise_ranking(
            roots, prediction, name, horizons=horizons[name], se_multiplier=se_multiplier
        )
    ranking["utility:root"] = pairwise_ranking(
        roots,
        prediction,
        "utility",
        horizons=utility_horizons,
        personality="root",
        se_multiplier=se_multiplier,
    )
    for pers in grid:
        ranking[f"utility:{pers[0]:g}/{pers[1]:g}"] = pairwise_ranking(
            roots,
            prediction,
            "utility",
            horizons=utility_horizons,
            personality=pers,
            se_multiplier=se_multiplier,
        )
    false = {"root": false_improvement(roots, prediction, margin=margin)}
    for pers in grid:
        false[f"{pers[0]:g}/{pers[1]:g}"] = false_improvement(
            roots, prediction, personality=pers, margin=margin
        )
    regret = pruning_regret(roots, prediction, top_k=top_k, margin=margin)
    controls = control_checks(roots, prediction)
    bookkeeping = bookkeeping_probe(head, contexts, roots, prediction, tolerance=tolerance)

    gates: dict[str, Any] = {}
    for key, value in ranking.items():
        gates[f"ranking:{key}"] = _gate(
            value["accuracy"], RANKING_GATE, above=True, pairs=value["pairs"]
        )
    gates["false_improvement"] = _gate(false["root"]["rate"], FALSE_IMPROVEMENT_GATE, above=False)
    gates["idle_control"] = _gate(controls["idle_last"]["rate"], 1.0, above=True)
    gates["delayed_benefit"] = _gate(controls["delayed_not_below_h12"]["rate"], 1.0, above=True)
    gates["bookkeeping_invariance"] = {
        "value": max(
            bookkeeping["data_twin_max_abs_delta"], bookkeeping["synthetic_max_abs_delta"]
        ),
        "threshold": tolerance,
        "pass": bookkeeping["pass"],
        "status": "PASS" if bookkeeping["pass"] else "FAIL",
    }
    # Required: QoL (primary, r5-b 3a), population-years, every utility; cost
    # rankings are required only where they have admitted pairs.
    required = ["ranking:qol_years", "ranking:population_years", "ranking:utility:root"]
    required += [f"ranking:utility:{p[0]:g}/{p[1]:g}" for p in grid]
    required += [
        f"ranking:{name}" for name in ("feasibility", "settlement_guard") if ranking[name]["pairs"]
    ]
    required += ["false_improvement", "idle_control", "delayed_benefit", "bookkeeping_invariance"]
    statuses = [gates[key]["status"] for key in required]
    status = "FAIL" if "FAIL" in statuses else "HOLD" if "HOLD" in statuses else "PASS"
    report: dict[str, Any] = {
        "kind": EVAL_KIND,
        "worlds": worlds,
        "roots": len(roots),
        "candidates": sum(len(r.candidates) for r in roots),
        "reserved_worlds_only": bool(worlds) and set(worlds) <= set(RESERVED_HELDOUT_WORLDS),
        "fit_worlds": sorted(int(w) for w in fit_worlds),
        "horizons": {k: None if v is None else list(v) for k, v in horizons.items()},
        "utility_horizons": None if utility_horizons is None else list(utility_horizons),
        "ranking": ranking,
        "false_improvement": false,
        "pruning_regret": regret,
        "controls": controls,
        "bookkeeping": bookkeeping,
        "gates": gates,
        "required_gates": required,
        "status": status,
    }
    report["report_digest"] = _json_digest(report)
    return report


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preregistered advantage-head validation")
    parser.add_argument("--checkpoint", type=Path, required=True, help="frozen critic checkpoint")
    parser.add_argument("--advantage", type=Path, required=True, help="labor_advantage fit output")
    parser.add_argument("--payloads", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--utility-horizons", default=None, help="comma list (default all)")
    parser.add_argument("--margin", type=float, default=FALSE_IMPROVEMENT_MARGIN)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from .labor_advantage import _load_critics

    args = argument_parser().parse_args(argv)
    task, costs = _load_critics(args.checkpoint)
    state = torch.load(args.advantage, map_location="cpu", weights_only=True)
    head, config = load_advantage_head(state, task, costs)
    candidates = load_candidate_shards(args.payloads)
    fit = state.get("fit", {})
    report = evaluate_advantage(
        task,
        costs,
        head,
        candidates,
        horizons=config.horizons,
        fit_worlds=fit.get("fit_worlds", []),
        utility_horizons=None
        if args.utility_horizons is None
        else [int(h) for h in args.utility_horizons.split(",")],
        se_multiplier=config.se_multiplier,
        margin=args.margin,
        top_k=args.top_k,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report["gates"][k] for k in report["required_gates"]}, indent=2))
    print(f"status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
