from __future__ import annotations

import json
from pathlib import Path

import pytest

from anthropocene_rl.autonomous_v7 import gate1_stream_aggregation as stream
from anthropocene_rl.autonomous_v7.gate1_diagnostics import service_outcome_registry
from anthropocene_rl.autonomous_v7.provenance import checkpoint_hash, digest


def _sealed(value: dict) -> dict:
    result = dict(value)
    result["digest"] = digest(result)
    return result


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _ref(path: Path, base: Path) -> dict:
    return {"path": path.relative_to(base).as_posix(),
            "sha256": checkpoint_hash(path), "size": path.stat().st_size}


def _nodes() -> list[dict]:
    fixed = [
        ("prior-reference", "evaluation-reference", None, None, "not-applicable"),
        ("bc-phase", "learned-bc-phase", None, None, "not-applicable"),
        ("t3-initialization-clone", "t3-initialization-clone", None, None,
         "not-applicable"),
        ("t2m-guarded-bc-seed6", "guarded-bc-actor-phase", None, 6,
         "not-applicable"),
        ("t3-init-guard-only-intervention", "prospective-normalizer-intervention",
         None, None, "not-applicable"),
    ]
    rows = []
    for node_id, role, arm, seed, pairing in fixed:
        rows.append({"id": node_id, "role": role,
                     "artifact": {"sha256": digest({"node": node_id})},
                     "run_identity": {"run_id": node_id, "arm_id": arm,
                                      "optimizer_seed": seed, "pairing_status": pairing,
                                      "pairing_group": None}})
    for node_id, (arm, seed) in sorted(stream.EXPECTED_T3.items()):
        rows.append({"id": node_id, "role": "trained-t3",
                     "artifact": {"sha256": digest({"node": node_id})},
                     "run_identity": {"run_id": node_id, "arm_id": arm,
                                      "optimizer_seed": seed, "pairing_status": "unpaired",
                                      "pairing_group": None}})
    return rows


def _result(node: str, pane: str, world: dict, node_order: dict[str, int]) -> dict:
    seed_index = world["seed"] - 50_000
    regime_index = stream.REGIMES.index(world["regime"])
    offset = node_order[node]
    if node == "prior-reference":
        offset = -1
    elif node in {"bc-phase", "t3-initialization-clone"}:
        offset = 0
    base = 10 + seed_index + regime_index + offset
    services = {
        row["name"]: {"value": float(base + index), "availability_reason": None}
        for index, row in enumerate(service_outcome_registry())
    }
    identity_node = "t3-initialization-clone" if node == "bc-phase" else node
    return {
        "welfare": float(base),
        "population_years": float(base * 10),
        "qol_years": float(base * 2),
        "population": max(0, base),
        "trade_utility": 0.0 if node == "prior-reference" else float(base / 2),
        "extinctions": max(0, offset),
        "settlements": 2 + max(0, offset),
        "compiler_interventions": max(0, offset),
        "compiler_intervention_summary": {"total": max(0, offset), "raw": []},
        "service_outcomes": services,
        "trajectory_digest": digest({"node": identity_node, "pane": pane,
                                     "world": world["id"], "base": base}),
    }


def _collection(root: Path) -> tuple[Path, list[Path]]:
    nodes = _nodes()
    node_ids = [node["id"] for node in nodes]
    node_order = {node_id: index for index, node_id in enumerate(node_ids)}
    worlds = [
        {"id": f"s{seed}-{regime}", "seed": seed, "regime": regime}
        for seed in range(50_000, 50_016) for regime in stream.REGIMES
    ]
    panes = [
        {"id": pane, "kind": "fresh-common-producer-diagnostic",
         "action_mode": mode, "nodes": node_ids}
        for pane, mode in stream.PANE_MODES.items()
    ]
    dag_path = root / "artifact-dag.json"
    _write(dag_path, _sealed({"schema": 1, "kind": "gate1_artifact_dag_proposal",
                              "nodes": nodes, "edges": []}))
    evaluation_path = root / "evaluation.json"
    _write(evaluation_path, _sealed({
        "schema": 1, "kind": "gate1_evaluation_claim_protocol_proposal",
        "panes": panes, "cohort": {"worlds": worlds},
    }))
    candidate_path = root / "freeze-candidate.json"
    candidate = _sealed({
        "schema": 1, "kind": "gate1_production_freeze_candidate",
        "evaluation_authorized": False,
        "proposals": {"artifact_dag": _ref(dag_path, root),
                      "evaluation": _ref(evaluation_path, root)},
    })
    _write(candidate_path, candidate)
    evaluation_paths = []
    evaluation_root = root / "evaluations"
    for node in nodes:
        for pane in panes:
            name = f"{node['id']}--{pane['id']}"
            report_path = evaluation_root / f"{name}.json"
            world_refs = []
            for world in worlds:
                world_path = evaluation_root / f"{name}.worlds" / f"{world['id']}.json"
                _write(world_path, _sealed({
                    "schema": 1, "kind": "gate1_fresh_common_producer_world",
                    "candidate_digest": candidate["digest"], "node": node["id"],
                    "pane": pane["id"], "world_id": world["id"],
                    "result": _result(node["id"], pane["id"], world, node_order),
                }))
                world_refs.append(_ref(world_path, evaluation_root))
            _write(report_path, _sealed({
                "schema": 1, "kind": "gate1_fresh_common_producer_evaluation",
                "candidate_digest": candidate["digest"], "node": node["id"],
                "pane": pane["id"], "checkpoint_sha256": node["artifact"]["sha256"],
                "worker_capture_sha256": "1" * 64,
                "producer_scope": "fresh-common-producer-diagnostic",
                "node_evidence_scope": "synthetic-test", "historical_artifact_claim": True,
                "historical_producer_status": "unresolved", "historical_replay_claim": False,
                "selection_allowed": False, "readiness_claimed": False,
                "world_count": len(world_refs), "world_artifacts": world_refs,
                "streaming": {"maximum_in_memory_world_results": 1,
                              "canonical_capture_bound_per_world_bytes": 1024},
            }))
            evaluation_paths.append(report_path)
    return candidate_path, evaluation_paths


@pytest.fixture(scope="module")
def collection(tmp_path_factory):
    return _collection(tmp_path_factory.mktemp("gate1-streams"))


def test_complete_collection_has_reviewed_pairing_and_no_gate_claim(collection):
    candidate, evaluations = collection
    first = stream.aggregate(candidate, evaluations)
    second = stream.aggregate(candidate, evaluations)
    assert first == second
    assert first["status"] == "COMPLETE-DIAGNOSTIC-SUMMARY-NOT-A-GATE-DECISION"
    assert first["gate_decision"] == "not-evaluated"
    assert not first["selection_allowed"] and not first["promotion_allowed"]
    assert first["completion"]["observed_evaluations"] == 40
    assert first["completion"]["observed_world_artifacts"] == 2_560
    assert len(first["raw_worlds"]) == 2_560
    assert first["resampling"]["bootstrap_seed"] == 20_261_023
    assert first["resampling"]["replicates"] == 2_000
    assert first["resampling"]["all_regime_independent_units"] == 16
    assert (first["bc_composite_current_identity"]["status"]
            == "exact-captured-evidence-match")
    welfare = next(row for row in first["per_regime_summaries"]
                   if row["edge"] == "prior-to-unguarded-init"
                   and row["pane"] == "fresh-common-producer-deterministic"
                   and row["regime"] == "default" and row["metric"] == "welfare")
    assert welfare["paired_world_count"] == 16
    assert welfare["difference"]["mean"] == 1.0
    trade = next(row for row in first["per_regime_summaries"]
                 if row["edge"] == "prior-to-unguarded-init"
                 and row["pane"] == "fresh-common-producer-deterministic"
                 and row["regime"] == "default" and row["metric"] == "trade_utility")
    assert trade["ratio"]["zero_denominator_count"] == 16
    clustered = next(row for row in first["all_regime_cluster_summaries"]
                     if row["edge"] == "prior-to-unguarded-init"
                     and row["pane"] == "fresh-common-producer-deterministic"
                     and row["metric"] == "welfare")
    assert (clustered["world_seed_cluster_count"], clustered["row_count"],
            clustered["independent_row_count_claim"]) == (16, 64, 16)
    assert clustered["regime_rows_are_independent_claim"] is False
    arm = next(row for row in first["arm_descriptive_means"]
               if row["arm_id"] == "T3per"
               and row["pane"] == "fresh-common-producer-deterministic"
               and row["regime"] == "default" and row["metric"] == "welfare")
    assert arm["pairing_status"] == "unpaired-disjoint"
    assert arm["complete_ratio_endpoint_count"] == 5
    assert arm["optimizer_uncertainty"]["status"] == "unestimated-unpaired"
    assert arm["optimizer_uncertainty"]["interval"] is None


def test_partial_collection_is_explicitly_not_pass(collection):
    candidate, evaluations = collection
    result = stream.aggregate(candidate, evaluations[:-1])
    assert result["status"] == "PARTIAL-NOT-PASS"
    assert result["gate_decision"] == "not-evaluated"
    assert result["completion"]["observed_evaluations"] == 39
    assert any(issue["type"] == "missing-evaluation"
               for issue in result["completion"]["issues"])
    assert (result["bc_composite_current_identity"]["status"]
            == "exact-captured-evidence-match")


def test_world_hash_tampering_rejects_even_when_original_collection_is_complete(collection):
    candidate, evaluations = collection
    report = json.loads(evaluations[0].read_text(encoding="utf-8"))
    world = evaluations[0].parent / report["world_artifacts"][0]["path"]
    original = world.read_bytes()
    try:
        world.write_bytes(original + b" ")
        with pytest.raises(stream.AggregationError, match="bytes differ"):
            stream.aggregate(candidate, evaluations)
    finally:
        world.write_bytes(original)


def test_explicit_service_unavailability_is_a_visible_observational_gap(collection):
    candidate, evaluations = collection
    report_path = evaluations[0]
    original_report = report_path.read_bytes()
    report = json.loads(original_report)
    world_path = report_path.parent / report["world_artifacts"][0]["path"]
    original_world = world_path.read_bytes()
    try:
        world = json.loads(original_world)
        service = service_outcome_registry()[0]["name"]
        world["result"]["service_outcomes"][service] = {
            "value": None, "availability_reason": "synthetic fixture unavailable",
        }
        world.pop("digest")
        _write(world_path, _sealed(world))
        report["world_artifacts"][0] = _ref(world_path, report_path.parent)
        report.pop("digest")
        _write(report_path, _sealed(report))
        result = stream.aggregate(candidate, evaluations)
        assert result["status"] == "COMPLETE-DIAGNOSTIC-SUMMARY-NOT-A-GATE-DECISION"
        assert result["causal_completeness_claim"] is False
        assert not result["completion"]["issues"]
        assert any(gap.get("field") == f"service:{service}"
                   for gap in result["observational_gaps"])
        assert (result["bc_composite_current_identity"]["status"]
                == "exact-captured-evidence-match")
        raw = next(row for row in result["raw_worlds"]
                   if row["node"] == report["node"] and row["pane"] == report["pane"]
                   and row["world_id"] == world["world_id"])
        assert raw["metrics"][f"service:{service}"]["value"] is None
        assert raw["metrics"][f"service:{service}"]["availability_reason"]
    finally:
        world_path.write_bytes(original_world)
        report_path.write_bytes(original_report)


def _rewrite_first_world(report_path: Path, mutate) -> tuple[bytes, Path, bytes]:
    original_report = report_path.read_bytes()
    report = json.loads(original_report)
    world_path = report_path.parent / report["world_artifacts"][0]["path"]
    original_world = world_path.read_bytes()
    world = json.loads(original_world)
    mutate(world["result"])
    world.pop("digest")
    _write(world_path, _sealed(world))
    report["world_artifacts"][0] = _ref(world_path, report_path.parent)
    report.pop("digest")
    _write(report_path, _sealed(report))
    return original_report, world_path, original_world


def test_bc_identity_requires_trajectory_and_mandatory_values(collection):
    candidate, evaluations = collection
    pane = "fresh-common-producer-deterministic"
    reports = [next(path for path in evaluations if path.stem == f"{node}--{pane}")
               for node in ("bc-phase", "t3-initialization-clone")]
    originals = []
    try:
        for report in reports:
            originals.append((report, *_rewrite_first_world(
                report, lambda result: result.pop("trajectory_digest"),
            )))
        result = stream.aggregate(candidate, evaluations)
        identity = result["bc_composite_current_identity"]
        assert result["status"] == "PARTIAL-NOT-PASS"
        assert identity["status"] == "unverified-missing"
        assert any("trajectory_digest" in row["unverified_fields"]
                   for row in identity["evidence_gaps"])
    finally:
        for report, original_report, world_path, original_world in originals:
            world_path.write_bytes(original_world)
            report.write_bytes(original_report)


def test_bc_identity_separates_declared_unmeasured_service_equivalence(collection):
    candidate, evaluations = collection
    pane = "fresh-common-producer-deterministic"
    reports = [next(path for path in evaluations if path.stem == f"{node}--{pane}")
               for node in ("bc-phase", "t3-initialization-clone")]
    service = service_outcome_registry()[0]["name"]
    originals = []
    try:
        for index, report in enumerate(reports):
            originals.append((report, *_rewrite_first_world(
                report,
                lambda result, index=index: result["service_outcomes"].__setitem__(
                    service, {"value": None,
                              "availability_reason": f"synthetic unavailable side {index}"},
                ),
            )))
        result = stream.aggregate(candidate, evaluations)
        identity = result["bc_composite_current_identity"]
        assert identity["status"] == "exact-captured-evidence-match"
        assert identity["full_physical_equivalence_claim"] is False
        assert any(row["field"] == f"service:{service}"
                   and row["claim"] == "not-claimed"
                   for row in identity["unmeasured_service_equivalence"])
        assert not identity["evidence_gaps"]
    finally:
        for report, original_report, world_path, original_world in originals:
            world_path.write_bytes(original_world)
            report.write_bytes(original_report)


def test_bc_identity_exposes_mismatch_even_when_collection_is_partial(collection):
    candidate, evaluations = collection
    report = next(path for path in evaluations
                  if path.stem == "bc-phase--fresh-common-producer-deterministic")
    original_report, world_path, original_world = _rewrite_first_world(
        report, lambda result: result.__setitem__("welfare", result["welfare"] + 1.0),
    )
    try:
        result = stream.aggregate(candidate, evaluations[:-1])
        assert result["status"] == "PARTIAL-NOT-PASS"
        identity = result["bc_composite_current_identity"]
        assert identity["observed_world_pairs"] == 128
        assert identity["status"] == "mismatch"
        assert any("metrics" in row["differing_fields"] for row in identity["mismatches"])
    finally:
        world_path.write_bytes(original_world)
        report.write_bytes(original_report)
