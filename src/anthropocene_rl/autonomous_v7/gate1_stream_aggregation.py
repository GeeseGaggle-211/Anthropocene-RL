"""Deterministic postprocessing for streamed Gate-1 world artifacts.

This module is deliberately outside the frozen evaluator/source manifest.  It
does not run checkpoints, rank treatments, or issue a gate/readiness decision.
It validates the content-addressed 20-node/64-world collection and reports
paired-world diagnostics while keeping the historical optimizer runs visibly
unpaired.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .gate1_diagnostics import service_outcome_registry
from .provenance import checkpoint_hash, digest

SCHEMA = 1
BOOTSTRAP_SEED = 20_261_023
BOOTSTRAP_REPLICATES = 2_000
REGIMES = ("default", "farming", "fishing", "herding")
PANE_MODES = {
    "fresh-common-producer-deterministic": "deterministic",
    "fresh-common-producer-seeded-stochastic": "seeded-stochastic",
}
FIXED_NODES = {
    "prior-reference",
    "bc-phase",
    "t3-initialization-clone",
    "t2m-guarded-bc-seed6",
    "t3-init-guard-only-intervention",
}
T3_SEEDS = {
    "T3per": tuple(range(70, 75)),
    "T3def": tuple(range(75, 80)),
    "T3lo": tuple(range(80, 85)),
}
EXPECTED_T3 = {
    f"{arm.lower()}-s{seed}": (arm, seed) for arm, seeds in T3_SEEDS.items() for seed in seeds
}
COUNT_FIELDS = {
    "ending_population": "population",
    "extinctions": "extinctions",
    "settlements": "settlements",
    "compiler_interventions": "compiler_interventions",
}
FLOAT_FIELDS = {
    "welfare": "welfare",
    "population_years": "population_years",
    "qol_years": "qol_years",
    "trade_utility": "trade_utility",
}
COMPARISON_METRICS = frozenset(FLOAT_FIELDS) | frozenset(COUNT_FIELDS)


class AggregationError(ValueError):
    """The collection is malformed, forged, or incompatible with this protocol."""


def _fail(message: str) -> None:
    raise AggregationError("Gate 1 streamed aggregation: " + message)


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    if "digest" in value:
        _fail("cannot reseal a document carrying a digest")
    result = dict(value)
    result["digest"] = digest(result)
    return result


def _sealed(value: Any, kind: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("kind") != kind:
        _fail(f"{kind} has the wrong schema or kind")
    body = {key: item for key, item in value.items() if key != "digest"}
    if set(value) != set(body) | {"digest"} or value["digest"] != digest(body):
        _fail(f"{kind} is not digest sealed")
    return value


def _read_json(path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        _fail(f"{kind} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AggregationError(f"cannot read {kind}: {path}") from error
    return _sealed(value, kind)


def _ref_path(ref: Any, base: Path, what: str) -> Path:
    if (
        not isinstance(ref, dict)
        or set(ref) != {"path", "sha256", "size"}
        or not isinstance(ref["path"], str)
        or not ref["path"]
        or not isinstance(ref["sha256"], str)
        or len(ref["sha256"]) != 64
        or not isinstance(ref["size"], int)
        or ref["size"] < 0
    ):
        _fail(f"{what} reference is malformed")
    relative = Path(ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"{what} reference escapes its package")
    root = base.resolve()
    path = (base / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise AggregationError(f"{what} reference escapes its package") from error
    return path


def _read_ref(
    ref: Any, base: Path, kind: str, *, missing_ok: bool = False
) -> tuple[dict | None, Path]:
    path = _ref_path(ref, base, kind)
    if not path.exists() and missing_ok:
        return None, path
    value = _read_json(path, kind)
    if path.stat().st_size != ref["size"] or checkpoint_hash(path) != ref["sha256"]:
        _fail(f"{kind} bytes differ from their reference")
    return value, path


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{what} is not finite numeric evidence")
    return float(value)


def _count(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{what} is not a nonnegative integer count")
    return value


def _sha(value: Any, what: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        _fail(f"{what} is not a SHA-256 digest")
    return value.lower()


def _candidate(candidate_path: Path) -> tuple[dict, dict, dict, dict[str, dict], dict[str, dict]]:
    candidate = _read_json(candidate_path, "gate1_production_freeze_candidate")
    if candidate.get("evaluation_authorized") is not False:
        _fail("candidate must remain non-authorizing input evidence")
    proposals = candidate.get("proposals")
    if not isinstance(proposals, dict):
        _fail("candidate proposal registry is absent")
    dag, _ = _read_ref(
        proposals.get("artifact_dag"), candidate_path.parent, "gate1_artifact_dag_proposal"
    )
    evaluation, _ = _read_ref(
        proposals.get("evaluation"),
        candidate_path.parent,
        "gate1_evaluation_claim_protocol_proposal",
    )
    assert dag is not None and evaluation is not None
    nodes = dag.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 20:
        _fail("candidate does not prescribe exactly 20 nodes")
    node_index = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            _fail("candidate node is malformed")
        if node["id"] in node_index:
            _fail("candidate repeats a node ID")
        node_index[node["id"]] = node
    if set(node_index) != FIXED_NODES | set(EXPECTED_T3):
        _fail("candidate node inventory differs from the reviewed 20-node inventory")
    for node_id, (arm, seed) in EXPECTED_T3.items():
        identity = node_index[node_id].get("run_identity")
        if (
            not isinstance(identity, dict)
            or identity.get("arm_id") != arm
            or identity.get("optimizer_seed") != seed
            or identity.get("pairing_status") != "unpaired"
            or identity.get("pairing_group") is not None
        ):
            _fail("historical T3 optimizer identity is not explicitly unpaired")
    panes = evaluation.get("panes")
    if not isinstance(panes, list) or len(panes) != 2:
        _fail("candidate does not prescribe exactly two fresh panes")
    pane_index = {}
    for pane in panes:
        if (
            not isinstance(pane, dict)
            or pane.get("id") not in PANE_MODES
            or pane.get("kind") != "fresh-common-producer-diagnostic"
            or pane.get("action_mode") != PANE_MODES[pane["id"]]
            or set(pane.get("nodes", [])) != set(node_index)
        ):
            _fail("candidate pane differs from the reviewed fresh-current panes")
        pane_index[pane["id"]] = pane
    if set(pane_index) != set(PANE_MODES):
        _fail("candidate pane inventory is incomplete")
    worlds = evaluation.get("cohort", {}).get("worlds")
    if not isinstance(worlds, list) or len(worlds) != 64:
        _fail("candidate does not prescribe exactly 64 worlds")
    world_index = {}
    for world in worlds:
        if (
            not isinstance(world, dict)
            or not isinstance(world.get("id"), str)
            or not isinstance(world.get("seed"), int)
            or isinstance(world.get("seed"), bool)
            or world.get("regime") not in REGIMES
        ):
            _fail("candidate world is malformed")
        if world["id"] in world_index:
            _fail("candidate repeats a world ID")
        world_index[world["id"]] = world
    seeds = sorted({world["seed"] for world in worlds})
    if len(seeds) != 16 or {(world["seed"], world["regime"]) for world in worlds} != {
        (seed, regime) for seed in seeds for regime in REGIMES
    }:
        _fail("candidate worlds are not a 16-seed by four-regime Cartesian cohort")
    return candidate, dag, evaluation, node_index, world_index


def _metric_record(
    result: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    metrics: dict[str, dict[str, Any]] = {}
    missing = []
    observational_gaps = []
    for output_name, source_name in FLOAT_FIELDS.items():
        if source_name not in result or result[source_name] is None:
            metrics[output_name] = {
                "value": None,
                "unit": None,
                "availability_reason": "mandatory result field absent",
            }
            missing.append({"field": output_name, "reason": "mandatory result field absent"})
        else:
            metrics[output_name] = {
                "value": _number(result[source_name], source_name),
                "unit": None,
                "availability_reason": None,
            }
    for output_name, source_name in COUNT_FIELDS.items():
        if source_name not in result or result[source_name] is None:
            metrics[output_name] = {
                "value": None,
                "unit": "count",
                "availability_reason": "mandatory result field absent",
            }
            missing.append({"field": output_name, "reason": "mandatory result field absent"})
        else:
            metrics[output_name] = {
                "value": _count(result[source_name], source_name),
                "unit": "count",
                "availability_reason": None,
            }
    registry = service_outcome_registry()
    registered = {row["name"]: row for row in registry}
    services = result.get("service_outcomes")
    if not isinstance(services, dict):
        services = {}
    extra = set(services) - set(registered)
    if extra:
        _fail(f"world result contains unregistered service outcomes: {sorted(extra)}")
    for name, field in registered.items():
        key = f"service:{name}"
        measurement = services.get(name)
        if measurement is None:
            reason = "registered service field absent"
            metrics[key] = {"value": None, "unit": field["unit"], "availability_reason": reason}
            missing.append({"field": key, "reason": reason})
            continue
        if not isinstance(measurement, dict) or set(measurement) != {
            "value",
            "availability_reason",
        }:
            _fail(f"service outcome {name} is malformed")
        value, reason = measurement["value"], measurement["availability_reason"]
        if value is None:
            if not isinstance(reason, str) or not reason.strip():
                _fail(f"unavailable service outcome {name} lacks a reason")
            metrics[key] = {"value": None, "unit": field["unit"], "availability_reason": reason}
            observational_gaps.append({"field": key, "reason": reason})
        else:
            if reason is not None:
                _fail(f"available service outcome {name} carries a missingness reason")
            metrics[key] = {
                "value": _number(value, key),
                "unit": field["unit"],
                "availability_reason": None,
            }
    trajectory = result.get("trajectory_digest")
    if trajectory is None:
        missing.append({"field": "trajectory_digest", "reason": "mandatory result field absent"})
    else:
        trajectory = _sha(trajectory, "trajectory_digest")
    summary = result.get("compiler_intervention_summary")
    if summary is not None and not isinstance(summary, dict):
        _fail("compiler intervention summary is malformed")
    return (
        {
            "metrics": metrics,
            "trajectory_digest": trajectory,
            "compiler_intervention_summary": summary,
        },
        missing,
        observational_gaps,
    )


def _load_evaluation(
    path: Path,
    candidate: dict,
    node_index: dict[str, dict],
    pane_ids: set[str],
    world_index: dict[str, dict],
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    report = _read_json(path, "gate1_fresh_common_producer_evaluation")
    node, pane = report.get("node"), report.get("pane")
    if node not in node_index or pane not in pane_ids:
        _fail("evaluation report names an unreviewed node or pane")
    if (
        report.get("candidate_digest") != candidate["digest"]
        or report.get("checkpoint_sha256") != node_index[node]["artifact"]["sha256"]
        or report.get("producer_scope") != "fresh-common-producer-diagnostic"
        or report.get("historical_replay_claim") is not False
        or report.get("selection_allowed") is not False
        or report.get("readiness_claimed") is not False
    ):
        _fail("evaluation report is not bound to the reviewed diagnostic scope")
    refs = report.get("world_artifacts")
    if (
        not isinstance(refs, list)
        or report.get("world_count") != len(refs)
        or len(refs) > len(world_index)
    ):
        _fail("evaluation report world reference count is malformed")
    rows = []
    issues = []
    observational_gaps = []
    seen = set()
    for ref in refs:
        world, world_path = _read_ref(
            ref,
            path.parent,
            "gate1_fresh_common_producer_world",
            missing_ok=True,
        )
        if world is None:
            issues.append(
                {
                    "type": "missing-world-artifact",
                    "node": node,
                    "pane": pane,
                    "path": str(world_path),
                }
            )
            continue
        world_id = world.get("world_id")
        if (
            world.get("candidate_digest") != candidate["digest"]
            or world.get("node") != node
            or world.get("pane") != pane
            or world_id not in world_index
            or world_id in seen
        ):
            _fail("world artifact identity is duplicated or incompatible")
        seen.add(world_id)
        result = world.get("result")
        if not isinstance(result, dict):
            _fail("world artifact result is malformed")
        extracted, missing, gaps = _metric_record(result)
        issues.extend(
            {"type": "missing-value", "node": node, "pane": pane, "world_id": world_id, **item}
            for item in missing
        )
        observational_gaps.extend(
            {
                "type": "declared-unavailable-service",
                "node": node,
                "pane": pane,
                "world_id": world_id,
                **item,
            }
            for item in gaps
        )
        rows.append(
            {
                "node": node,
                "pane": pane,
                "world_id": world_id,
                "seed": world_index[world_id]["seed"],
                "regime": world_index[world_id]["regime"],
                "evaluation_sha256": checkpoint_hash(path),
                "world_artifact_sha256": checkpoint_hash(world_path),
                "mandatory_missing": missing,
                "observational_gaps": gaps,
                **extracted,
            }
        )
    return report, rows, issues, observational_gaps


def _ratio(destination: float | int, source: float | int) -> dict[str, Any]:
    if source == 0:
        return {"status": "zero-denominator", "value": None}
    return {"status": "defined", "value": destination / source}


def _bootstrap(
    values: list[float], identity: dict[str, Any], *, clusters: bool = False
) -> dict[str, Any]:
    if len(values) < 2:
        return {"status": "unavailable", "lower": None, "upper": None}
    substream = BOOTSTRAP_SEED ^ int(digest(identity)[:16], 16)
    rng = np.random.Generator(np.random.PCG64(substream))
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    samples = np.sort(array[indices].mean(axis=1))
    return {
        "status": "defined",
        "lower": samples[49],
        "upper": samples[1949],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "resampling_unit": "world-seed-cluster" if clusters else "paired-world",
    }


def _edge_specs(node_index: dict[str, dict]) -> list[dict[str, Any]]:
    specs = [
        {
            "id": "prior-to-unguarded-init",
            "source": "prior-reference",
            "destination": "t3-initialization-clone",
            "arm": None,
        },
        {
            "id": "init-to-guard-only",
            "source": "t3-initialization-clone",
            "destination": "t3-init-guard-only-intervention",
            "arm": None,
        },
        {
            "id": "init-to-t2m-guarded-bc",
            "source": "t3-initialization-clone",
            "destination": "t2m-guarded-bc-seed6",
            "arm": None,
        },
    ]
    for node_id in sorted(EXPECTED_T3):
        specs.append(
            {
                "id": f"init-to-{node_id}",
                "source": "t3-initialization-clone",
                "destination": node_id,
                "arm": EXPECTED_T3[node_id][0],
                "optimizer_seed": EXPECTED_T3[node_id][1],
                "pairing_status": "unpaired",
            }
        )
    if any(
        spec["source"] not in node_index or spec["destination"] not in node_index for spec in specs
    ):
        _fail("comparison registry references an absent node")
    return specs


def _summarize(
    values: list[dict[str, Any]], identity: dict[str, Any], expected: int
) -> dict[str, Any]:
    differences = [row["difference"] for row in values]
    ratios = [row["ratio"]["value"] for row in values if row["ratio"]["status"] == "defined"]
    complete = len(values) == expected
    return {
        "status": "complete" if complete else "partial",
        "paired_world_count": len(values),
        "expected_paired_world_count": expected,
        "difference": {
            "mean": None if not differences else sum(differences) / len(differences),
            "interval": (
                _bootstrap(differences, {**identity, "statistic": "difference"})
                if complete
                else {"status": "unavailable", "reason": "incomplete cohort"}
            ),
        },
        "ratio": {
            "mean": None if not ratios else sum(ratios) / len(ratios),
            "defined_count": len(ratios),
            "zero_denominator_count": sum(
                row["ratio"]["status"] == "zero-denominator" for row in values
            ),
            "interval": (
                _bootstrap(ratios, {**identity, "statistic": "ratio"})
                if complete and len(ratios) == expected
                else {"status": "unavailable", "reason": "incomplete cohort or zero denominator"}
            ),
        },
    }


def aggregate(candidate_path: Path, evaluation_paths: Iterable[Path]) -> dict[str, Any]:
    """Validate and aggregate a reviewed streamed collection without making a gate decision."""
    candidate_path = candidate_path.resolve()
    candidate, _dag, _evaluation, node_index, world_index = _candidate(candidate_path)
    pane_ids = set(PANE_MODES)
    rows: dict[tuple[str, str, str], dict] = {}
    reports = {}
    issues = []
    observational_gaps = []
    inventory = []
    for original in evaluation_paths:
        path = Path(original).resolve()
        report, loaded, report_issues, report_gaps = _load_evaluation(
            path,
            candidate,
            node_index,
            pane_ids,
            world_index,
        )
        key = (report["node"], report["pane"])
        if key in reports:
            _fail("collection repeats a node/pane evaluation report")
        reports[key] = report
        inventory.append(
            {
                "node": key[0],
                "pane": key[1],
                "path": str(path),
                "sha256": checkpoint_hash(path),
                "size": path.stat().st_size,
            }
        )
        issues.extend(report_issues)
        observational_gaps.extend(report_gaps)
        for row in loaded:
            row_key = (row["node"], row["pane"], row["world_id"])
            if row_key in rows:
                _fail("collection repeats a node/pane/world artifact")
            rows[row_key] = row
    expected_reports = {(node, pane) for node in node_index for pane in pane_ids}
    for node, pane in sorted(expected_reports - set(reports)):
        issues.append({"type": "missing-evaluation", "node": node, "pane": pane})
    for node, pane in sorted(set(reports)):
        observed = {
            world for row_node, row_pane, world in rows if row_node == node and row_pane == pane
        }
        for world_id in sorted(set(world_index) - observed):
            issues.append(
                {"type": "missing-world", "node": node, "pane": pane, "world_id": world_id}
            )

    edge_pairs = []
    per_regime = []
    all_regime = []
    endpoint_index: dict[tuple[str, str, str, str], dict] = {}
    for edge in _edge_specs(node_index):
        for pane in sorted(pane_ids):
            for regime in REGIMES:
                grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for world_id, world in world_index.items():
                    if world["regime"] != regime:
                        continue
                    left = rows.get((edge["source"], pane, world_id))
                    right = rows.get((edge["destination"], pane, world_id))
                    if left is None or right is None:
                        continue
                    for metric in sorted(COMPARISON_METRICS):
                        source = left["metrics"][metric]["value"]
                        destination = right["metrics"].get(metric, {}).get("value")
                        if source is None or destination is None:
                            continue
                        pair = {
                            "edge": edge["id"],
                            "pane": pane,
                            "regime": regime,
                            "world_id": world_id,
                            "world_seed": world["seed"],
                            "metric": metric,
                            "source": source,
                            "destination": destination,
                            "difference": destination - source,
                            "ratio": _ratio(destination, source),
                        }
                        edge_pairs.append(pair)
                        grouped[metric].append(pair)
                for metric in sorted(grouped):
                    identity = {
                        "edge": edge["id"],
                        "pane": pane,
                        "regime": regime,
                        "metric": metric,
                    }
                    summary = {
                        **identity,
                        "source_node": edge["source"],
                        "destination_node": edge["destination"],
                        **_summarize(grouped[metric], identity, 16),
                    }
                    per_regime.append(summary)
                    endpoint_index[(edge["id"], pane, regime, metric)] = summary
            for metric in sorted(
                {
                    row["metric"]
                    for row in edge_pairs
                    if row["edge"] == edge["id"] and row["pane"] == pane
                }
            ):
                by_seed = defaultdict(list)
                for row in edge_pairs:
                    if (
                        row["edge"] == edge["id"]
                        and row["pane"] == pane
                        and row["metric"] == metric
                    ):
                        by_seed[row["world_seed"]].append(row["difference"])
                cluster_values = [
                    sum(values) / len(values)
                    for _seed, values in sorted(by_seed.items())
                    if len(values) == len(REGIMES)
                ]
                identity = {
                    "edge": edge["id"],
                    "pane": pane,
                    "metric": metric,
                    "statistic": "all-regime-clustered-difference",
                }
                all_regime.append(
                    {
                        "edge": edge["id"],
                        "pane": pane,
                        "metric": metric,
                        "world_seed_cluster_count": len(cluster_values),
                        "row_count": sum(len(values) for values in by_seed.values()),
                        "status": "complete" if len(cluster_values) == 16 else "partial",
                        "difference_mean": (
                            None
                            if not cluster_values
                            else sum(cluster_values) / len(cluster_values)
                        ),
                        "interval": (
                            _bootstrap(cluster_values, identity, clusters=True)
                            if len(cluster_values) == 16
                            else {"status": "unavailable", "reason": "incomplete seed clusters"}
                        ),
                        "independent_row_count_claim": 16,
                        "regime_rows_are_independent_claim": False,
                    }
                )

    arm_means = []
    for arm, seeds in T3_SEEDS.items():
        endpoints = [f"{arm.lower()}-s{seed}" for seed in seeds]
        for pane in sorted(pane_ids):
            for regime in REGIMES:
                metric_names = sorted(
                    {
                        key[3]
                        for key in endpoint_index
                        if key[0] in {f"init-to-{node}" for node in endpoints}
                        and key[1] == pane
                        and key[2] == regime
                    }
                )
                for metric in metric_names:
                    summaries = [
                        endpoint_index.get((f"init-to-{node}", pane, regime, metric))
                        for node in endpoints
                    ]
                    values = [
                        row["difference"]["mean"]
                        for row in summaries
                        if row is not None
                        and row["status"] == "complete"
                        and row["difference"]["mean"] is not None
                    ]
                    ratio_values = [
                        row["ratio"]["mean"]
                        for row in summaries
                        if row is not None
                        and row["status"] == "complete"
                        and row["ratio"]["defined_count"] == 16
                        and row["ratio"]["mean"] is not None
                    ]
                    arm_means.append(
                        {
                            "arm_id": arm,
                            "pane": pane,
                            "regime": regime,
                            "metric": metric,
                            "endpoint_nodes": endpoints,
                            "optimizer_seeds": list(seeds),
                            "pairing_status": "unpaired-disjoint",
                            "complete_endpoint_count": len(values),
                            "expected_endpoint_count": 5,
                            "descriptive_mean_of_endpoint_paired_world_means": (
                                None if not values else sum(values) / len(values)
                            ),
                            "complete_ratio_endpoint_count": len(ratio_values),
                            "descriptive_mean_of_endpoint_ratio_means": (
                                None if not ratio_values else sum(ratio_values) / len(ratio_values)
                            ),
                            "optimizer_uncertainty": {
                                "status": "unestimated-unpaired",
                                "interval": None,
                                "reason": (
                                    "world bootstrap does not estimate optimizer uncertainty; "
                                    "historical arm seeds are disjoint and no paired arm contrast is valid"
                                ),
                            },
                        }
                    )

    identity_mismatches = []
    identity_gaps = []
    unmeasured_service_equivalence = []
    identity_observed = 0
    for pane in sorted(pane_ids):
        for world_id in sorted(world_index):
            bc = rows.get(("bc-phase", pane, world_id))
            initialization = rows.get(("t3-initialization-clone", pane, world_id))
            if bc is None or initialization is None:
                continue
            identity_observed += 1
            fields = []
            differing_metrics = []
            for name in sorted(bc["metrics"]):
                source = bc["metrics"][name]
                destination = initialization["metrics"][name]
                if name.startswith("service:") and (
                    source["value"] is None or destination["value"] is None
                ):
                    unmeasured_service_equivalence.append(
                        {
                            "pane": pane,
                            "world_id": world_id,
                            "field": name,
                            "bc_phase": source,
                            "t3_initialization_clone": destination,
                            "claim": "not-claimed",
                        }
                    )
                    continue
                if source != destination:
                    differing_metrics.append(name)
            if differing_metrics:
                fields.append("metrics")
            if bc["trajectory_digest"] != initialization["trajectory_digest"]:
                fields.append("trajectory_digest")
            if fields:
                identity_mismatches.append(
                    {
                        "pane": pane,
                        "world_id": world_id,
                        "differing_fields": fields,
                        "differing_metrics": differing_metrics,
                    }
                )
            required_gaps = []
            if bc["mandatory_missing"] or initialization["mandatory_missing"]:
                required_gaps.append("mandatory_metrics")
            if bc["trajectory_digest"] is None or initialization["trajectory_digest"] is None:
                required_gaps.append("trajectory_digest")
            if required_gaps:
                identity_gaps.append(
                    {"pane": pane, "world_id": world_id, "unverified_fields": required_gaps}
                )

    complete = not issues and set(reports) == expected_reports and len(rows) == 20 * 2 * 64
    source_file = Path(__file__).resolve()
    return _seal(
        {
            "schema": SCHEMA,
            "kind": "gate1_streamed_diagnostic_aggregation",
            "status": (
                "COMPLETE-DIAGNOSTIC-SUMMARY-NOT-A-GATE-DECISION"
                if complete
                else "PARTIAL-NOT-PASS"
            ),
            "gate_decision": "not-evaluated",
            "causal_completeness_claim": False,
            "selection_allowed": False,
            "ranking_allowed": False,
            "promotion_allowed": False,
            "readiness_claimed": False,
            "candidate": {
                "path": str(candidate_path),
                "sha256": checkpoint_hash(candidate_path),
                "size": candidate_path.stat().st_size,
                "digest": candidate["digest"],
            },
            "implementation": {
                "path": str(source_file),
                "sha256": checkpoint_hash(source_file),
                "size": source_file.stat().st_size,
            },
            "input_inventory": sorted(inventory, key=lambda row: (row["node"], row["pane"])),
            "completion": {
                "expected_nodes": 20,
                "expected_panes": 2,
                "expected_worlds_per_node_pane": 64,
                "expected_evaluations": 40,
                "observed_evaluations": len(reports),
                "expected_world_artifacts": 2_560,
                "observed_world_artifacts": len(rows),
                "issues": issues,
            },
            "observational_gaps": observational_gaps,
            "resampling": {
                "bootstrap_seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "algorithm": "numpy.random.PCG64",
                "numpy_version": np.__version__,
                "per_regime_unit": "paired world",
                "all_regime_unit": "world seed cluster containing all four regimes",
                "all_regime_independent_units": 16,
                "optimizer_design": "historical T3 optimizer seeds are disjoint and unpaired",
                "hierarchical_structure": "optimizer runs are nested within arms; arms are unpaired",
                "optimizer_uncertainty": "separately unestimated; never supplied by world bootstrap",
                "between_arm_inference": "not estimated; descriptive arm means only",
            },
            "raw_worlds": sorted(
                rows.values(), key=lambda row: (row["node"], row["pane"], row["world_id"])
            ),
            "raw_pairs": edge_pairs,
            "per_regime_summaries": per_regime,
            "all_regime_cluster_summaries": all_regime,
            "arm_descriptive_means": arm_means,
            "bc_composite_current_identity": {
                "expected": True,
                "observed_world_pairs": identity_observed,
                "expected_world_pairs": 128,
                "status": (
                    "exact-captured-evidence-match"
                    if identity_observed == 128 and not identity_mismatches and not identity_gaps
                    else "mismatch"
                    if identity_mismatches
                    else "unverified-missing"
                    if identity_gaps
                    else "incomplete"
                ),
                "mismatches": identity_mismatches,
                "evidence_gaps": identity_gaps,
                "unmeasured_service_equivalence": unmeasured_service_equivalence,
                "full_physical_equivalence_claim": False,
                "scope": (
                    "fresh-current captured finite mandatory outcomes plus canonical trajectory digest; "
                    "declared-unavailable services and historical BC-time behavior are not claimed equivalent"
                ),
            },
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, action="append", default=[])
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = list(args.evaluation)
    if args.evaluation_root is not None:
        if not args.evaluation_root.is_dir():
            _fail("evaluation root is not a directory")
        paths.extend(sorted(args.evaluation_root.glob("*.json")))
    if args.out.exists():
        _fail("output already exists")
    result = aggregate(args.candidate, paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: result[key] for key in ("status", "gate_decision", "digest")}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
