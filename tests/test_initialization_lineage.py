"""Gate-1 lineage validation is fail-closed and never launches training."""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7 import initialization_lineage as gate1
from anthropocene_rl.autonomous_v7.provenance import ProvenanceError, checkpoint_hash, digest
from anthropocene_rl.autonomous_v7.worker_contract import seal

SEMANTICS = {
    "experts": False, "token": False, "commitments": True, "learned_movement": False,
    "region_movement": False, "learned_fission": False, "learned_trade": False,
    "region_head": "hierarchical", "grouped_region": False, "region_selection": None,
}


def write(path: Path, value, *, seal_digest=False) -> dict:
    if seal_digest:
        value = dict(value); value["digest"] = digest(value)
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": path.name, "sha256": checkpoint_hash(path)}


def write_reviewed_protocol(path: Path, body: dict, reviewer: dict, scope: list[str]) -> dict:
    proposal_kind = body["kind"]
    proposal_digest = digest({key: value for key, value in body.items()
                              if key not in {"review", "digest"}})
    review = write(path.with_name(path.stem + "-review.json"), {
        "schema": 1, "kind": "independent_protocol_review", "reviewer_identity": reviewer,
        "decision": "approved", "proposal_kind": proposal_kind, "proposal_digest": proposal_digest,
        "scope": scope, "findings": ["fixture protocol scope reviewed"],
    }, seal_digest=True)
    body = copy.deepcopy(body); body["review"] = review
    return write(path, body, seal_digest=True)


def rewrite_reviewed_protocol(tmp_path: Path, ref: dict, mutate) -> dict:
    """Mutate a fixture protocol and issue a fresh content-bound independent review."""
    path = tmp_path / ref["path"]
    body = json.loads(path.read_text())
    review = json.loads((tmp_path / body["review"]["path"]).read_text())
    reviewer = review["reviewer_identity"]
    scope = review["scope"]
    body.pop("digest")
    body["review"] = None
    mutate(body)
    ref.clear()
    ref.update(write_reviewed_protocol(path, body, reviewer, scope))
    return body


def rewrite_sealed_ref(tmp_path: Path, ref: dict, mutate) -> dict:
    """Rewrite one digest-sealed fixture artifact and update its content reference."""
    path = tmp_path / ref["path"]
    body = json.loads(path.read_text())
    body.pop("digest", None)
    mutate(body)
    body["digest"] = digest(body)
    path.write_text(json.dumps(body, sort_keys=True))
    ref["sha256"] = checkpoint_hash(path)
    return body


def refresh_gate1_protocols(document: dict, tmp_path: Path, *, inventory: bool = True,
                            evaluation: bool = True) -> None:
    """Issue fresh fixture reviews after an intentional test-surface change."""
    if inventory:
        manifest_edges = {row["id"]: row for row in document["edges"]}

        def update_inventory(body):
            for row in body["edges"]:
                row["equivalence"] = gate1._equivalence_descriptor(
                    manifest_edges[row["id"]].get("equivalence"),
                )

        rewrite_reviewed_protocol(tmp_path, document["lineage_inventory"], update_inventory)
    if evaluation:
        projection = gate1._evaluation_claim_projection(document)
        rewrite_reviewed_protocol(
            tmp_path, document["evaluation_claim_protocol"],
            lambda body: body.update(copy.deepcopy(projection)),
        )


def install_contract8_checkpoint(document: dict, tmp_path: Path, node_id: str,
                                 *, remove_actor_key: str | None = None) -> None:
    """Replace one fixture node with a complete reviewed raw contract-8 execution path."""
    from tests.test_gate1_legacy_contract8 import checkpoint_state
    node = next(row for row in document["nodes"] if row["id"] == node_id)
    state = checkpoint_state()
    if remove_actor_key is not None:
        state["actor"] = dict(state["actor"])
        state["actor"].pop(remove_actor_key)
    state["run_manifest"] = {"worker_contract": torch.load(
        tmp_path / document["nodes"][0]["artifact"]["path"], weights_only=True,
    )["run_manifest"]["worker_contract"]}
    checkpoint = tmp_path / f"{node_id}-contract8.pt"
    torch.save(state, checkpoint)
    node["artifact"] = {"path": checkpoint.name, "sha256": checkpoint_hash(checkpoint)}
    node["raw_schema"], _, _ = gate1.inspect_checkpoint(node["artifact"], tmp_path)
    legacy_semantics = {"experts": False, "token": False, "commitments": False,
                        "learned_movement": False, "region_movement": False,
                        "learned_fission": False, "learned_trade": False,
                        "region_head": "flat", "grouped_region": False,
                        "region_selection": None}
    node["historical_semantics"] = legacy_semantics
    defaults = {"schema": 1, "kind": "checkpoint_default_resolution",
                "checkpoint_sha256": node["artifact"]["sha256"], "implicit_defaults": [],
                "resolved": {key: {"source": "reviewed-contract-8-default", "value": value}
                             for key, value in legacy_semantics.items()}}
    default_ref = write(tmp_path / f"{node_id}-contract8-defaults.json", defaults, seal_digest=True)
    reviewer = write(tmp_path / f"{node_id}-contract8-reviewer.json", {
        "schema": 1, "kind": "independent_reviewer_identity", "provider": "openai",
        "model": "astra-high", "version": "2026-10-22",
    }, seal_digest=True)
    source_path = gate1.REPOSITORY / "src/anthropocene_rl/autonomous_v7/legacy_contract8.py"
    source = {"path": str(source_path.resolve()), "sha256": checkpoint_hash(source_path)}
    protocol = write_reviewed_protocol(tmp_path / f"{node_id}-contract8-adapter.json", {
        "schema": 1, "kind": "gate1_legacy_adapter_protocol", "source_contract": 8,
        "historical_source_commit": "2f4cc895e6bcd897200d340851d69be98fd18e00",
        "source": source,
        "input_mapping": {"core": "raw-checkpoint-core-width-prefix",
                          "sets": ["sites", "fishing", "herds", "commodities", "contacts"],
                          "appended_sets": "excluded-from-contract-8-actor-input"},
        "decoder_defaults": {"commit": 1, "movement": contracts.NATIVE_MOVEMENT,
                             "fission": -1, "trade_posture": "native"},
        "default_resolution": default_ref, "review": None,
    }, reviewer, ["historical-source", "input-mapping", "decoder", "default-resolution"])
    migration = write(tmp_path / f"{node_id}-contract8-migration.json", {
        "schema": 1, "kind": "checkpoint_migration", "status": "reconstructed",
        "historical_conversion": False, "destination_checkpoint_sha256": node["artifact"]["sha256"],
        "adapter": protocol,
    }, seal_digest=True)
    node["loading"] = {"mode": "reviewed-adapter", "implementation": protocol,
                       "default_resolution": default_ref, "migration_record": migration}
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)


def raw_checkpoint(path: Path, worker: dict, value=1.0, contract=contracts.NATIVE_CONTRACT) -> dict:
    from anthropocene_rl.autonomous_v7.model import Actor
    with torch.random.fork_rng():
        torch.manual_seed(1)
        expected = Actor(4, experts=False, token=False, region_head="hierarchical").state_dict()
    pools = {dtype: torch.zeros(1, dtype=dtype) for dtype in {row.dtype for row in expected.values()}}
    actor = {name: (pools[reference.dtype].expand(reference.shape)
                    if reference.ndim else pools[reference.dtype].reshape(()))
             for name, reference in expected.items()}
    actor["strategic.bias"] = torch.full_like(expected["strategic.bias"], value)
    normalizer = {"core_mean": [0.0] * 4, "core_std": [1.0] * 4}
    for name, _, width in contracts.SETS:
        normalizer[name + "_mean"] = [0.0] * width
        normalizer[name + "_std"] = [1.0] * width
    state = {"contract": contract, "core_width": 4, "actor": actor,
             "normalizer": normalizer,
             "run_manifest": {"worker_contract": worker}}
    torch.save(state, path)
    return {"path": path.name, "sha256": checkpoint_hash(path)}


def fixture(tmp_path: Path) -> tuple[dict, dict]:
    worker = seal({"version": 1,
                   "simulator": {"commit": "1" * 40, "buildHash": "2" * 64, "clean": True},
                   "bridge_sha256": "3" * 64,
                   "flags": {"sim_rules": "", "sim_statics": ""},
                   "feature_flags": {"SimulationConfig.ElderRawMaterialsEnabled": False},
                   "configuration": {"sha256": "4" * 64, "players": 8}})
    worker_ref = write(tmp_path / "worker.json", {"schema": 1, "kind": "worker_contract", "contract": worker})
    from anthropocene_rl.autonomous_v7.gate1_diagnostics import service_outcome_registry
    service_ref = write(tmp_path / "services.json", {"schema": 1, "kind": "service_outcome_schema",
                                                      "fields": service_outcome_registry()},
                        seal_digest=True)
    common_ref = write(tmp_path / "evidence.json", {"schema": 1, "kind": "fixture_evidence"})
    reviewer_ref = write(tmp_path / "reviewer.json", {
        "schema": 1, "kind": "independent_reviewer_identity", "provider": "openai",
        "model": "astra-high", "version": "2026-10-22",
    }, seal_digest=True)
    inference_ref = write(tmp_path / "inference.json", {
        "schema": 1, "kind": "checkpoint_inference_implementation", "identity": "fixture-evaluator",
        "commit": "b" * 40, "source": common_ref,
    }, seal_digest=True)
    corpus_inputs = [write(tmp_path / f"equivalence-input-{index}.json", {"row": index})
                     for index in (1, 2)]
    corpus_ref = write(tmp_path / "corpus.json", {
        "schema": 1, "kind": "equivalence_corpus",
        "cases": [{"id": f"case-{index}", "input": ref}
                  for index, ref in enumerate(corpus_inputs, start=1)],
    }, seal_digest=True)
    tool_ref = write(tmp_path / "verification-tool.json", {
        "schema": 1, "kind": "verification_tool_identity", "identity": "pytest+coverage-fixture",
        "commit": "c" * 40, "source": common_ref,
    }, seal_digest=True)
    benchmark_baseline = write(tmp_path / "benchmark-baseline.json", {"scenario": "packet-build"})
    protocol_ref = write_reviewed_protocol(tmp_path / "verification-protocol.json", {
        "schema": 1, "kind": "gate1_verification_protocol",
        "architecture_commit": gate1.ARCHITECTURE_COMMIT,
        "review": None,
        "tests": [{"id": "gate1-focused", "command": ["python", "-m", "pytest",
                                                        "tests/test_initialization_lineage.py"]}],
        "coverage": {"minimum_percent": 90.0,
                     "scope": ["src/anthropocene_rl/autonomous_v7/initialization_lineage.py"],
                     "denominator": "executable-lines", "exclusions": ["pragma:no-cover"], "tool": tool_ref},
        "parity": {"method": "bitwise-diagnostic-off", "scenarios": ["diagnostic-off"]},
        "benchmarks": [{"id": "packet-build", "scenario": "canonical-packet-build",
                        "baseline": benchmark_baseline,
                        "throughput": {"work_unit": "packet", "unit": "packets/second", "minimum": 0.1},
                        "memory": {"method": "process-peak-rss", "unit": "MiB", "maximum": 1024.0}}],
    }, reviewer_ref, ["tests", "coverage", "diagnostic-off-parity", "benchmarks", "budgets"])
    trained_specs = [(arm, seed) for arm, seeds in (("T3per", range(70, 75)),
                                                    ("T3def", range(75, 80)),
                                                    ("T3lo", range(80, 85)))
                     for seed in seeds]
    node_specs = [("prior", "prior-policy", None, None),
                  ("clone", "guarded-clone", None, None)]
    node_specs += [(f"trained-{arm.lower()}-{seed}", "trained-t3", arm, seed)
                   for arm, seed in trained_specs]
    refs = [raw_checkpoint(tmp_path / f"node-{i}.pt", worker) for i in range(len(node_specs))]
    nodes = []
    raw = []
    for i, ref in enumerate(refs):
        inspected, _, _ = gate1.inspect_checkpoint(ref, tmp_path); raw.append(inspected)
        defaults = {"schema": 1, "kind": "checkpoint_default_resolution", "checkpoint_sha256": ref["sha256"],
                    "implicit_defaults": [], "resolved": {key: {"source": "checkpoint-or-manifest", "value": value}
                                                             for key, value in SEMANTICS.items()}}
        default_ref = write(tmp_path / f"defaults-{i}.json", defaults, seal_digest=True)
        node_id, role, arm, seed = node_specs[i]
        run_identity = ({"run_id": ("prior-reference" if node_id == "prior" else "guarded-clone"),
                         "arm_id": None, "optimizer_seed": None,
                         "pairing_status": "not-applicable", "pairing_group": None}
                        if seed is None else
                        {"run_id": f"{arm}-seed-{seed}", "arm_id": arm, "optimizer_seed": seed,
                         "pairing_status": "unpaired", "pairing_group": None})
        nodes.append({"id": node_id, "role": role,
                      "status": "present", "evidence_class": "historical-record", "historical_status": "documented",
                      "run_identity": run_identity, "inference_implementation": inference_ref,
                      "artifact": ref, "raw_schema": copy.deepcopy(inspected),
                      "contracts": {"actor": "actor-v1", "critic": {"status": "not-applicable", "reason": "actor-only"},
                                    "normalizer": "normalizer-v1"},
                      "producing_learner": {"identity": "fixture learner", "commit": "a" * 40},
                      "producer_evidence": common_ref, "run_evidence": common_ref, "source_artifacts": [],
                      "loading": {"mode": "native-strict", "default_resolution": default_ref},
                      "historical_semantics": copy.deepcopy(SEMANTICS)})
    claim = {"kind": "exact_tensor_preservation", "source_schema": raw[0], "destination_schema": raw[1],
             "numeric_runtime": {"framework": "torch", "version": torch.__version__, "device": "cpu",
                                 "dtype": "float32", "determinism": "bitwise tensor compare"},
             "mapped_tensors": [{"source": "actor.strategic.bias",
                                 "destination": "actor.strategic.bias"}],
             "mapped_outputs": [], "appended_outputs": [], "action_decoder": {"kind": "unchanged"},
             "active_heads": ["actor"], "delegated_heads": [], "normalizer_mapping": {"kind": "identity"},
             "tolerances": {"absolute": 0, "relative": 0}, "corpus": corpus_ref}
    trained_ids = [row[0] for row in node_specs[2:]]
    pane = {"id": "historical", "kind": "historical-semantics",
            "nodes": ["prior", "clone", *trained_ids],
            "worlds": [{"family": "evaluation", "seed": 4000, "withheld": 0, "warmup": 0, "horizon": 2,
                        "reset_turn": 1, "snapshot": None}], "action_mode": "deterministic",
            "observation_core_width": 4,
            "capture_bounds": {"max_steps": 2, "max_villages_per_boundary": 64,
                               "max_economic_rows_per_step": 64,
                               "max_canonical_bytes": 4 * 1024 * 1024},
            "metrics": ["welfare", "ending_residents"],
            "background_policy": {**gate1.CURRENT_BACKGROUND_POLICY,
                                  "digest": digest(gate1.CURRENT_BACKGROUND_POLICY)},
            "service_outcome_schema": service_ref,
            "execution": {"tier": "local", "selection_allowed": False, "readiness_claimed": False},
            "personality": [1.0, 0.0, 0.0, 0.0], "evaluation_worker_contract": worker_ref,
            "evaluation_implementation": inference_ref,
            "environment_overrides": {"ANTHROPOCENE_REGION_HEAD": "", "ANTHROPOCENE_SIM_RULES": "",
                                      "ANTHROPOCENE_SIM_STATICS": ""}, "standardized_overrides": {}}
    absence_source = write(tmp_path / "absence-source.json", {"historical": "no conversion"})
    absence_ref = write(tmp_path / "conversion-absence.json", {
        "schema": 1, "kind": "historical_absence_evidence", "slot": "conversion",
        "finding": "not-applicable", "sources": [absence_source],
    }, seal_digest=True)
    inventory = {
        "schema": 1, "kind": "gate1_lineage_inventory_protocol",
        "architecture_commit": gate1.ARCHITECTURE_COMMIT,
        "review": None,
        "slots": [
            {"slot": "prior-reference", "status": "present", "node_ids": ["prior"], "edge_ids": [],
             "reason": None, "historical_support": None},
            {"slot": "conversion", "status": "not-applicable", "node_ids": [], "edge_ids": [],
             "reason": "no historical conversion occurred", "historical_support": absence_ref},
            {"slot": "guarded-clone", "status": "present", "node_ids": ["clone"], "edge_ids": [],
             "reason": None, "historical_support": None},
            {"slot": "trained-t3", "status": "present", "node_ids": trained_ids, "edge_ids": [],
             "reason": None, "historical_support": None},
        ],
        "nodes": [{"id": row["id"], "role": row["role"], "status": row["status"],
                   "historical_panes": ["historical"], "standardized_panes": [],
                   "run_identity": copy.deepcopy(row["run_identity"])} for row in nodes],
        "edges": [
            {"id": "prior-clone", "source": "prior", "destination": "clone",
             "relation": "parameter_parent", "status": "present", "historical_panes": ["historical"],
             "standardized_panes": [], "sequence": "main", "position": 0,
             "equivalence": gate1._equivalence_descriptor(claim)},
        ] + [{"id": f"clone-{node_id}", "source": "clone", "destination": node_id,
              "relation": "ppo_continuation", "status": "present", "historical_panes": ["historical"],
              "standardized_panes": [], "sequence": f"branch-{node_id}", "position": 0,
              "equivalence": None}
             for node_id in trained_ids],
        "required_nodes": ["prior", "clone", *trained_ids],
        "required_edges": ["prior-clone", *[f"clone-{node_id}" for node_id in trained_ids]],
        "sequences": ([{"id": "main", "branch": "shared-base", "edges": ["prior-clone"]}]
                      + [{"id": f"branch-{node_id}", "branch": node_id,
                          "edges": [f"clone-{node_id}"]} for node_id in trained_ids]),
        "comparison_order": ["prior-clone", *[f"clone-{node_id}" for node_id in trained_ids]],
    }
    inventory_ref = write_reviewed_protocol(
        tmp_path / "lineage-inventory.json", inventory, reviewer_ref,
        ["actual-inventory", "slots", "applicability", "chronology", "run-identities"],
    )
    document = {"schema": 1, "kind": "initialization_lineage", "architecture_commit": gate1.ARCHITECTURE_COMMIT,
                "lineage_inventory": inventory_ref,
                "verification_protocol": protocol_ref,
                "required_nodes": ["prior", "clone", *trained_ids],
                "required_edges": ["prior-clone", *[f"clone-{node_id}" for node_id in trained_ids]],
                "nodes": nodes,
                "edges": [{"id": "prior-clone", "status": "present", "source": "prior", "destination": "clone",
                           "relation": "parameter_parent", "evidence": common_ref, "equivalence": claim},
                          *[{"id": f"clone-{node_id}", "status": "present", "source": "clone",
                             "destination": node_id, "relation": "ppo_continuation", "evidence": common_ref}
                            for node_id in trained_ids]],
                "comparison_order": ["prior-clone", *[f"clone-{node_id}" for node_id in trained_ids]],
                "evaluation": {"panes": [pane]}}
    evaluation_protocol = {
        "schema": 1, "kind": "gate1_evaluation_claim_protocol",
        "architecture_commit": gate1.ARCHITECTURE_COMMIT, "review": None,
        **gate1._evaluation_claim_projection(document),
    }
    document["evaluation_claim_protocol"] = write_reviewed_protocol(
        tmp_path / "evaluation-claim-protocol.json", evaluation_protocol, reviewer_ref,
        ["artifacts", "equivalence", "evaluation-cohort", "semantics", "implementations"],
    )
    document["digest"] = digest(document)
    return document, worker


def reseal(document: dict) -> dict:
    document["digest"] = digest({k: v for k, v in document.items() if k != "digest"})
    return document


def trained_ids(document: dict) -> list[str]:
    return [row["id"] for row in document["nodes"] if row["role"] == "trained-t3"]


def implementation_ref(tmp_path: Path, *, mismatched_source: bool = False,
                       mismatched_dirty_state: bool = False) -> dict:
    tree = gate1.implementation_source_tree()
    if mismatched_source:
        tree["source_files"][0]["sha256"] = "0" * 64
    if mismatched_dirty_state:
        tree["dirty_state"] = {"clean": False, "scope": "entire-repository",
                               "porcelain": [" M forged-proof"],
                               "porcelain_sha256": digest([" M forged-proof"])}
    identity = {"schema": 1, "kind": "gate1_implementation_identity",
                "architecture_commit": gate1.ARCHITECTURE_COMMIT, **tree}
    return write(tmp_path / ("implementation-mismatch.json" if mismatched_source else "implementation.json"),
                 identity, seal_digest=True)


def verification_ref(tmp_path: Path, document: dict, *, commit: str | None = None, exit_code: int = 0,
                     observed_coverage: float = 95.0) -> dict:
    protocol = json.loads((tmp_path / document["verification_protocol"]["path"]).read_text())
    output = tmp_path / "test-output.txt"
    output.write_text("focused tests passed\n")
    machine = {"name": "fixture", "os": "test", "cpu": "test-cpu"}
    machine["digest"] = digest(machine)
    test_receipt = write(tmp_path / "test-receipt.json", {
        "schema": 1, "kind": "gate1_test_execution_receipt", "test_id": "gate1-focused",
        "command": protocol["tests"][0]["command"], "exit_code": exit_code,
        "machine_digest": machine["digest"], "started_at": "2026-10-22T00:00:00Z",
        "finished_at": "2026-10-22T00:00:01Z",
        "output": {"path": output.name, "sha256": checkpoint_hash(output)},
    }, seal_digest=True)
    covered = int(observed_coverage)
    coverage_raw = write(tmp_path / "coverage-raw.json", {
        "schema": 1, "kind": "gate1_coverage_raw",
        "tool_sha256": protocol["coverage"]["tool"]["sha256"],
        "scope": protocol["coverage"]["scope"],
        "denominator": protocol["coverage"]["denominator"],
        "exclusions": protocol["coverage"]["exclusions"],
        "files": [{"path": protocol["coverage"]["scope"][0],
                   "covered_lines": list(range(1, covered + 1)),
                   "missing_lines": list(range(covered + 1, 101))}],
    }, seal_digest=True)
    coverage_report = write(tmp_path / "coverage.json", {
        "schema": 1, "kind": "gate1_coverage_report",
        "tool_sha256": protocol["coverage"]["tool"]["sha256"],
        "scope": protocol["coverage"]["scope"], "denominator": protocol["coverage"]["denominator"],
        "exclusions": protocol["coverage"]["exclusions"], "covered": covered, "total": 100,
        "raw_measurements": coverage_raw,
    }, seal_digest=True)
    parity_baseline = write(tmp_path / "parity-run-baseline.json", {"trajectory": "same"})
    parity_off = write(tmp_path / "parity-run-off.json", {"trajectory": "same"})
    parity_report = write(tmp_path / "parity.json", {
        "schema": 1, "kind": "gate1_diagnostic_off_parity", "method": "bitwise-diagnostic-off",
        "scenarios": [{"id": "diagnostic-off", "baseline": parity_baseline,
                       "diagnostic_off": parity_off}],
    }, seal_digest=True)
    benchmark = protocol["benchmarks"][0]
    benchmark_raw = write(tmp_path / "benchmark-raw.json", {
        "schema": 1, "kind": "gate1_benchmark_raw", "benchmark_id": benchmark["id"],
        "scenario": benchmark["scenario"], "machine_digest": machine["digest"],
        "memory_method": benchmark["memory"]["method"],
        "samples": [{"iteration": 0, "elapsed_seconds": 1.0,
                     "work_units": 1, "peak_memory_mib": 128.0}],
    }, seal_digest=True)
    benchmark_receipt = write(tmp_path / "benchmark-receipt.json", {
        "schema": 1, "kind": "gate1_benchmark_receipt", "benchmark_id": benchmark["id"],
        "scenario": benchmark["scenario"], "baseline_sha256": benchmark["baseline"]["sha256"],
        "machine_digest": machine["digest"], "elapsed_seconds": 1.0, "work_units": 1,
        "peak_memory_mib": 128.0, "memory_method": benchmark["memory"]["method"],
        "raw_measurements": benchmark_raw,
    }, seal_digest=True)
    evidence = {
        "schema": 1, "kind": "gate1_verification",
        "implementation_commit": commit or gate1.implementation_source_tree()["commit"],
        "protocol_sha256": document["verification_protocol"]["sha256"],
        "execution": {"tier": "local", "selection_allowed": False, "readiness_claimed": False},
        "machine": machine, "test_receipts": [test_receipt], "coverage": coverage_report,
        "missingness": {"required_missing": 0},
        "parity": parity_report, "benchmarks": [benchmark_receipt],
    }
    return write(tmp_path / "verification.json", evidence, seal_digest=True)


def mandatory_values(*, welfare: float = 1.0, service: float | None = 1.0) -> dict:
    from anthropocene_rl.autonomous_v7.gate1_diagnostics import service_outcome_registry
    return {"welfare": welfare, "population_years": 2.0, "ending_residents": 2,
            "qol_years": 1.5, "trade_utility": 0.0, "extinctions": 0, "settlements": 1,
            "service_outcomes": {field["name"]: {"value": service,
                                                   "availability_reason": (None if service is not None else "not observed")}
                                 for field in service_outcome_registry()},
            "compiler_interventions": 0, "trajectory_digest": "8" * 64}


def policy_observation(core_width: int = 4) -> dict:
    return {
        "core": [0.0] * core_width,
        "corePresent": [True] * core_width,
        "sets": [[0.0] * (rows * width) for _, rows, width in contracts.SETS],
        "rowPresent": [[False] * rows for _, rows, _ in contracts.SETS],
        "movementMask": [True] * contracts.MOVEMENT_SLOTS,
        "regionMovementMask": [True] * contracts.REGION_MOVEMENT_SLOTS,
        "clothingRouteMask": [True, True],
        "shelterRouteMask": [True, True],
        "economyMixMask": [True] * len(contracts.ECONOMY_MIX),
        "commitHoldMask": True,
    }


def policy_action(semantics: dict) -> dict:
    movement = contracts.NATIVE_MOVEMENT
    if semantics["learned_movement"]:
        movement = (contracts.REGION_BASE
                    if semantics["region_movement"] else 0)
    action = {
        "strategic": [0.5] * len(contracts.STRATEGIC),
        "service": [0.5] * len(contracts.SERVICES),
        "clothingRoute": 0,
        "shelterRoute": 0,
        "movement": movement,
        "fission": (0 if semantics["learned_fission"] else -1),
        "commit": (contracts.SPAN_SINGLE if not semantics["commitments"] else 1),
    }
    if semantics["learned_trade"]:
        action.update({"tradeValue": [1.0] * len(contracts.TRADE_GROUPS),
                       "tradeGate": [0] * len(contracts.TRADE_GROUPS),
                       "tradeShare": 0.0})
    return action


def diagnostic_result(world: dict, world_id: str, horizon: int, bounds: dict, values: dict,
                      *, action_mode: str = "deterministic", rng_steps: list[dict] | None = None,
                      semantics: dict | None = None, observation_core_width: int = 4,
                      decoder: str = "native-contract-12") -> dict:
    from anthropocene_rl.autonomous_v7 import gate1_diagnostics as diagnostic
    from tests.test_gate1_diagnostics import decoder_receipt
    complete_values = mandatory_values()
    complete_values.update(copy.deepcopy(values))
    values = complete_values
    semantics = copy.deepcopy(semantics or SEMANTICS)
    reset_fingerprint = digest({"world": world_id, "reset": True})
    warmup_chain = []
    chain_fingerprint = reset_fingerprint
    chain_turn = world["reset_turn"]
    for warmup_index in range(world["warmup"]):
        successor_fingerprint = digest({"world": world_id, "warmup": warmup_index})
        warmup_chain.append({"index": warmup_index, "source_turn": chain_turn,
                             "source_fingerprint": chain_fingerprint,
                             "successor_turn": chain_turn + 1,
                             "successor_fingerprint": successor_fingerprint,
                             "trajectory_id": reset_fingerprint})
        chain_turn += 1
        chain_fingerprint = successor_fingerprint
    opening_fingerprint = chain_fingerprint
    population = int(values["ending_residents"])
    opening_village = {"camp": 1, "society": 1, "parentSociety": 0,
                       "priorPopulation": population, "population": population,
                       "qol": values["qol_years"] / horizon, "years": 1.0,
                       "welfare": 0.0, "tradeUtility": 0.0, "terminated": False,
                       "founded": False, "observation": policy_observation(observation_core_width)}
    opening_village["observation_sha256"] = digest(opening_village["observation"])
    opening = {"turn": chain_turn, "fingerprint": opening_fingerprint,
               "villages": [opening_village]}
    prior = opening
    steps = []
    service = next(iter(values["service_outcomes"].values()))["value"]
    numeric_service = 0.0 if service is None else float(service)
    for index in range(horizon):
        successor_village = copy.deepcopy(opening_village)
        successor_village.update({
            "priorPopulation": population, "welfare": values["welfare"] / horizon,
            "tradeUtility": values["trade_utility"] / horizon,
            "qol": values["qol_years"] / horizon, "years": 1.0,
        })
        successor = {"turn": prior["turn"] + 1,
                     "fingerprint": digest({"world": world_id, "successor": index}),
                     "villages": [successor_village]}
        trade = {"settledTrades": (int(numeric_service) if index == horizon - 1 else 0),
                 "realisedUtility": values["trade_utility"] / horizon}
        for good in diagnostic.TRADE_GOODS:
            trade["imported" + good] = numeric_service / horizon
            trade["exported" + good] = numeric_service / horizon
        unavailable = {}
        cultivation = numeric_service / horizon
        if service is None:
            cultivation = None
            unavailable["domains.cultivation.foodDelivered"] = "not observed"
        economic = {
            "schemaVersion": 1,
            "transitionId": f"{opening_fingerprint}/{prior['turn']}/1",
            "trajectoryId": reset_fingerprint, "decisionTurn": prior["turn"],
            "campId": 1, "societyId": 1, "years": 1.0,
            "domains": {
                "population": {
                    "personYears": values["population_years"] / horizon,
                    "qol": values["qol_years"] / horizon,
                    "openingResidents": population, "openingParties": 0,
                    "openingLinked": population, "reportStartingLinked": population,
                    "reportEndingLinked": population,
                    "afterReintegrationResidents": population,
                    "afterReintegrationParties": 0,
                    "splitOff": 0, "emigrated": 0, "immigrated": 0,
                    "endingResidents": population, "endingParties": 0,
                    "endingLinked": population,
                },
                "cultivation": {"foodDelivered": cultivation, "funnels": []},
                "herding": {"acceptedFood": numeric_service / horizon, "funnels": []},
                "manufacturing": {"funnels": []}, "trade": trade,
            },
            "units": diagnostic.economic_unit_registry(),
            "unavailable": unavailable,
            "herds": [{"herdId": 1, "slaughterFood": numeric_service / horizon}],
            "goods": [{"good": good, "unit": "good-units",
                       "output": numeric_service / horizon} for good in diagnostic.CRAFT_GOODS],
            "materials": [{"material": material, "unit": "material-units",
                           "received": numeric_service / horizon}
                          for material in diagnostic.MATERIALS],
            "requests": [], "trades": [], "transfers": [], "fields": [],
            "conservation": {}, "parentTransitionId": None, "forkId": None, "branchId": None,
        }
        action = policy_action(semantics)
        receipt = decoder_receipt(
            action, opening_village["observation"], semantics,
            mode=action_mode, decoder=decoder,
        )
        rng_receipt = (diagnostic.recompute_rng_receipt(rng_steps[index], 1)
                       if action_mode == "seeded-stochastic" else None)
        steps.append({
            "index": index,
            "transition_id": digest({"index": index, "source_turn": prior["turn"],
                                     "source_fingerprint": prior["fingerprint"],
                                     "successor_turn": successor["turn"],
                                     "successor_fingerprint": successor["fingerprint"]}),
            "source_turn": prior["turn"], "source_fingerprint": prior["fingerprint"],
            "actions": [{"camp": 1, "action_request": action, "decoded_action": action,
                         "decoder_receipt": receipt}],
            "policy_rng_receipt": rng_receipt, "successor": successor,
            "terminal": {"terminated": False, "truncated": index == horizon - 1,
                         "reason": "evaluation-horizon" if index == horizon - 1 else "continuing"},
            "economic": [economic],
        })
        prior = successor
    evidence = diagnostic.trajectory_evidence(
        opening, steps, bounds, world_id=world_id, snapshot_sha256=world["snapshot_sha256"],
        reset_turn=world["reset_turn"], reset_fingerprint=reset_fingerprint,
        warmup_chain=warmup_chain,
    )
    derived = diagnostic.validate_trajectory_evidence(
        evidence, world_id=world_id, snapshot_sha256=world["snapshot_sha256"],
        horizon=horizon, bounds=bounds, action_mode=action_mode,
        rng_steps=rng_steps, service_fields=diagnostic.service_outcome_registry(),
        expected_opening_turn=world["reset_turn"] + world["warmup"],
        expected_reset_turn=world["reset_turn"],
        observation_core_width=observation_core_width, policy_semantics=semantics,
        expected_decoder=decoder,
    )
    return {**derived, "population": derived["ending_residents"],
            "trajectory_evidence": evidence,
            "snapshot_sha256": world["snapshot_sha256"], "action_mode": action_mode}


def evaluation_report(document: dict, tmp_path: Path, node: str, worker: dict) -> dict:
    pane = next(row for row in document["evaluation"]["panes"] if row["id"] == "historical")
    frozen = pane["worlds"][0]
    world_id = gate1._world_id(frozen)
    node_record = next(row for row in document["nodes"] if row["id"] == node)
    decoder = ("pinned-contract-8" if node_record["loading"]["mode"] == "reviewed-adapter"
               else "native-contract-12")

    def runner(*args, **kwargs):
        result = diagnostic_result(
            gate1._frozen_world_record(frozen), world_id, frozen["horizon"],
            pane["capture_bounds"], mandatory_values(), action_mode=pane["action_mode"],
            rng_steps=kwargs.get("policy_rng_steps"), semantics=kwargs["evaluation_semantics"],
            observation_core_width=pane["observation_core_width"],
            decoder=decoder,
        )
        return {**result, "worker_contract": worker, "world_contract": args[14],
                "producer": gate1._producer_identity_for_worker(worker)}

    return gate1.evaluate_node(document, tmp_path, node, "historical", runner)


def manual_report(document: dict, node_id: str, pane_id: str, worker: dict, values: dict,
                  *, base: Path | None = None) -> dict:
    node = next(row for row in document["nodes"] if row["id"] == node_id)
    pane = next(row for row in document["evaluation"]["panes"] if row["id"] == pane_id)
    semantics = dict(node["historical_semantics"])
    if pane["kind"] == "standardized-flags-intervention":
        semantics.update(pane["standardized_overrides"])
    rows = []
    decoder = ("pinned-contract-8" if node["loading"]["mode"] == "reviewed-adapter"
               else "native-contract-12")
    for world in pane["worlds"]:
        rng_steps = None
        if pane["action_mode"] == "seeded-stochastic":
            if base is None:
                raise AssertionError("seeded-stochastic fixture needs its artifact base")
            schedule, _ = gate1._read_json_ref(
                pane["policy_rng_schedule"], base, kind="gate1_policy_rng_schedule",
            )
            from anthropocene_rl.autonomous_v7.gate1_diagnostics import (
                draw_order_from_semantics, validate_rng_schedule,
            )
            rng_steps = validate_rng_schedule(
                schedule, gate1._world_id(world), world["horizon"], node_id=node_id,
                expected_draw_order=draw_order_from_semantics(semantics),
            )
        result = diagnostic_result(
            gate1._frozen_world_record(world), gate1._world_id(world), world["horizon"],
            pane["capture_bounds"], values, action_mode=pane["action_mode"], rng_steps=rng_steps,
            semantics=semantics, observation_core_width=pane["observation_core_width"],
            decoder=decoder,
        )
        row_values = {metric: result[metric] for metric in pane["metrics"]}
        rows.append({"world_id": gate1._world_id(world), "world": gate1._frozen_world_record(world),
                     "metrics": row_values,
                     "compiler_intervention_summary": result["compiler_intervention_summary"],
                     "trajectory_evidence": result["trajectory_evidence"],
                     "policy_rng_receipts": result["policy_rng_receipts"],
                     "evaluation_steps": result["evaluation_steps"],
                     "capture_receipt": result["capture_receipt"],
                     "producer": gate1._producer_identity_for_worker(worker),
                     "worker_contract": worker,
                     "world_contract": gate1._expected_world_descriptor(world, semantics)})
    report = {"schema": 1, "kind": "initialization_evaluation",
              "evidence_class": "checkpoint-evaluation", "manifest_digest": document["digest"],
              "node": node_id, "pane": pane_id, "checkpoint_sha256": node["artifact"]["sha256"],
              "semantics": semantics, "metrics": pane["metrics"], "personality": pane["personality"],
              "background_policy": pane["background_policy"],
              "service_outcome_schema": pane["service_outcome_schema"],
              "inference_implementation": node["inference_implementation"],
              "evaluation_implementation": pane["evaluation_implementation"],
              "loading_protocol": node["loading"].get("implementation"),
              "execution": pane["execution"], "action_mode": pane["action_mode"],
              "policy_rng_schedule": pane.get("policy_rng_schedule"),
              "capture_bounds": pane["capture_bounds"], "rows": rows}
    report["observation_core_width"] = pane["observation_core_width"]
    report["digest"] = digest(report)
    return report


def reseal_trajectory_report(report: dict) -> None:
    """Reseal a deliberately forged imported report without repairing its semantics."""
    from anthropocene_rl.autonomous_v7 import gate1_diagnostics as diagnostic
    for row in report["rows"]:
        evidence = row["trajectory_evidence"]
        evidence["capture_receipt"] = diagnostic._capture_receipt(
            evidence["reset"], evidence["opening"], evidence["steps"],
        )
        evidence["digest"] = digest({key: value for key, value in evidence.items()
                                     if key != "digest"})
        if "trajectory_digest" in row["metrics"]:
            row["metrics"]["trajectory_digest"] = evidence["digest"]
    report["digest"] = digest({key: value for key, value in report.items() if key != "digest"})


def apply_trajectory_attack(report: dict, attack: str) -> None:
    """Apply one Astra-reproduced raw-evidence or asserted-outcome attack."""
    row = report["rows"][0]
    evidence = row["trajectory_evidence"]
    steps = evidence["steps"]
    if attack == "empty":
        steps.clear()
    elif attack == "incomplete":
        steps[0].pop("actions")
    elif attack == "duplicate":
        steps[1]["transition_id"] = steps[0]["transition_id"]
    elif attack == "short":
        del steps[1:]
        steps[0]["terminal"] = {"terminated": False, "truncated": False,
                                "reason": "continuing"}
    elif attack == "missing-terminal-reason":
        steps[-1]["terminal"]["reason"] = ""
    elif attack == "fabricated-headlines":
        for name in ("welfare", "ending_residents", "trade_utility", "extinctions", "settlements"):
            row["metrics"][name] = 999999
    elif attack == "wrong-unit":
        steps[0]["economic"][0]["goods"][0]["unit"] = "kilograms"
    elif attack == "negative-output":
        steps[0]["economic"][0]["goods"][0]["output"] = -5.0
    elif attack == "fractional-trade-count":
        steps[0]["economic"][0]["domains"]["trade"]["settledTrades"] = 0.25
    elif attack == "missing-herds":
        steps[0]["economic"][0].pop("herds")
    elif attack == "successor-prior-suppression":
        steps[0]["successor"]["villages"][0]["priorPopulation"] = 0
        steps[0]["economic"].clear()
    elif attack == "false-terminal":
        del steps[1:]
        steps[0]["terminal"] = {"terminated": True, "truncated": False,
                                "reason": "all-villages-terminated"}
    elif attack == "opening-turn-mismatch":
        evidence["opening"]["turn"] = 17
    elif attack == "null-actions":
        steps[0]["actions"][0]["action_request"] = None
        steps[0]["actions"][0]["decoded_action"] = None
    elif attack == "empty-observation":
        evidence["opening"]["villages"][0]["observation"] = {}
        evidence["opening"]["villages"][0]["observation_sha256"] = digest({})
    elif attack == "wrong-society-id":
        steps[0]["economic"][0]["societyId"] = 999999
    elif attack == "wrong-trade-unit":
        steps[0]["economic"][0]["units"]["domains.trade.realisedUtility"] = "bananas"
    elif attack == "missing-units":
        steps[0]["economic"][0]["units"] = {}
    elif attack == "extra-unit":
        steps[0]["economic"][0]["units"]["domains.trade.unregistered"] = "qol"
    elif attack == "wrong-reset-trajectory-id":
        evidence["reset"]["trajectory_id"] = "f" * 64
    elif attack == "broken-opening-chain":
        evidence["reset"]["warmup_chain"] = [{
            "index": 0, "source_turn": evidence["reset"]["turn"],
            "source_fingerprint": evidence["reset"]["fingerprint"],
            "successor_turn": evidence["reset"]["turn"] + 1,
            "successor_fingerprint": "e" * 64,
            "trajectory_id": evidence["reset"]["trajectory_id"],
        }]
    elif attack in {"invented-founder", "mismatched-founder-witness"}:
        parent = steps[0]["successor"]["villages"][0]
        daughter = copy.deepcopy(parent)
        daughter.update(camp=2, society=2, parentSociety=1, priorPopulation=0,
                        population=(999999 if attack == "invented-founder" else 1),
                        years=0.0, welfare=0.0, tradeUtility=0.0,
                        founded=True)
        steps[0]["successor"]["villages"].append(daughter)
        if attack == "mismatched-founder-witness":
            population = steps[0]["economic"][0]["domains"]["population"]
            population["splitOff"] = 2
            steps[0]["economic"][0]["transfers"].append({
                "kind": "fission", "phase": "close-after-demography",
                "sourceCamp": 1, "destinationCamp": 2, "people": 2, "animals": 0,
            })
    elif attack == "reflag-existing-founder":
        steps[0]["successor"]["villages"][0]["founded"] = True
    elif attack == "illegal-clothing-route":
        observation = evidence["opening"]["villages"][0]["observation"]
        observation["clothingRouteMask"] = [False, True]
        evidence["opening"]["villages"][0]["observation_sha256"] = digest(observation)
    elif attack == "fabricated-trade-boundary":
        steps[0]["successor"]["villages"][0]["tradeUtility"] = 99999.0
    elif attack == "deterministic-unsnapped-levels":
        action = steps[0]["actions"][0]
        action["action_request"]["service"][0] = 0.005
        action["decoded_action"]["service"][0] = 0.005
        action["action_request"]["strategic"][0] = 0.995
        action["decoded_action"]["strategic"][0] = 0.995
    elif attack == "invented-migration-founder":
        parent = steps[0]["successor"]["villages"][0]
        daughter = copy.deepcopy(parent)
        daughter.update(camp=2, society=2, parentSociety=1, priorPopulation=0,
                        population=999999, years=0.0, welfare=0.0,
                        tradeUtility=0.0, founded=True)
        steps[0]["successor"]["villages"].append(daughter)
        steps[0]["economic"][0]["transfers"].append({
            "kind": "migration", "phase": "close-after-demography",
            "sourceCamp": 1, "destinationCamp": 2, "people": 999999,
        })
    elif attack == "migration-copies-deleted":
        population = steps[0]["economic"][0]["domains"]["population"]
        population["emigrated"] = 1
        population["endingResidents"] -= 1
        population["endingLinked"] -= 1
        steps[0]["successor"]["villages"][0]["population"] -= 1
    elif attack == "migration-kind-renamed":
        forged = {"kind": "unreviewed-movement", "phase": "close-after-demography",
                  "sourceCamp": 1, "destinationCamp": 2, "people": 1}
        steps[0]["economic"][0]["transfers"] = [copy.deepcopy(forged),
                                                   copy.deepcopy(forged)]
    elif attack == "migration-population-inflation":
        population = steps[0]["economic"][0]["domains"]["population"]
        population["endingResidents"] += 999999
        steps[0]["successor"]["villages"][0]["population"] += 999999
    else:  # pragma: no cover - fixture authoring guard
        raise AssertionError(f"unknown trajectory attack {attack}")
    reseal_trajectory_report(report)


def complete_packet_inputs(document: dict, worker: dict, tmp_path: Path, prefix: str):
    pane = document["evaluation"]["panes"][0]
    pane["metrics"] = sorted(gate1.MANDATORY_METRICS)
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    reports = [manual_report(document, node, pane["id"], worker, mandatory_values(), base=tmp_path)
               for node in pane["nodes"]]
    evaluation_refs = [write(tmp_path / f"{prefix}-evaluation-{report['node']}.json", report)
                       for report in reports]
    verification = verification_ref(tmp_path, document)
    implementation = implementation_ref(tmp_path)
    manifest_ref = write(tmp_path / f"{prefix}-manifest.json", document)
    return evaluation_refs, verification, implementation, manifest_ref


def install_raw_equivalence(document: dict, tmp_path: Path, *,
                            kind: str = "deterministic_behavioral_equivalence",
                            prefix: str = "raw-equivalence",
                            scopes: list[str] | None = None,
                            omit_rng_case: bool = False,
                            dtype: str = "float32",
                            absolute_paths: bool = False) -> dict:
    """Install complete canonical case/scope/side execution evidence in a fixture."""
    def emit(path: Path, value: dict, *, seal_digest: bool = False) -> dict:
        ref = write(path, value, seal_digest=seal_digest)
        if absolute_paths:
            ref["path"] = str(path.resolve())
        return ref

    edge = document["edges"][0]
    source = next(row for row in document["nodes"] if row["id"] == edge["source"])
    destination = next(row for row in document["nodes"] if row["id"] == edge["destination"])
    claim = copy.deepcopy(edge["equivalence"])
    claim["kind"] = kind
    claim["mapped_tensors"] = []
    claim["mapped_outputs"] = scopes or ["actor-output"]
    claim["tolerances"] = {"absolute": 0.0, "relative": 0.0}
    claim["numeric_runtime"]["dtype"] = dtype
    behavioral = kind in {"deterministic_behavioral_equivalence",
                          "stochastic_behavioral_equivalence"}
    rng_ref = None
    corpus = json.loads((tmp_path / claim["corpus"]["path"]).read_text())
    if kind == "stochastic_behavioral_equivalence":
        rng_ref = emit(tmp_path / f"{prefix}-rng.json", {
            "schema": 1, "kind": "equivalence_rng_schedule",
            "cases": [{"case_id": case["id"], "draws": [1000 + index]}
                      for index, case in enumerate(
                          corpus["cases"][:-1] if omit_rng_case else corpus["cases"]
                      )],
        }, seal_digest=True)
        claim["rng_schedule"] = rng_ref
    executor_source = emit(tmp_path / f"{prefix}-executor-source.json", {"source": prefix})
    executor_ref = emit(tmp_path / f"{prefix}-executor.json", {
        "schema": 1, "kind": "equivalence_executor_implementation",
        "identity": "fixture-equivalence-executor", "commit": "e" * 40,
        "source": executor_source,
    }, seal_digest=True)
    output_contract_ref = emit(tmp_path / f"{prefix}-output-contracts.json", {
        "schema": 1, "kind": "equivalence_output_contracts",
        "contracts": [{
            "case_id": case["id"], "scope": scope,
            "dtype": claim["numeric_runtime"]["dtype"], "shape": [2],
            "components": ["component-0", "component-1"],
            "mask": ({"shape": [2], "components": ["action-0", "action-1"]}
                     if behavioral else None),
            "decoded_action": ({"type": "object", "fields": {
                "index": {"type": "integer", "minimum": 0, "maximum": 1},
            }} if behavioral else None),
        } for case in corpus["cases"] for scope in claim["mapped_outputs"]],
    }, seal_digest=True)
    protocol_ref = emit(tmp_path / f"{prefix}-protocol.json", {
        "schema": 1, "kind": "equivalence_execution_protocol",
        "claim_projection_sha256": gate1._claim_protocol_digest(claim),
        "executor_implementation": executor_ref,
        "source_implementation": source["inference_implementation"],
        "destination_implementation": destination["inference_implementation"],
        "numeric_runtime": claim["numeric_runtime"], "scopes": claim["mapped_outputs"],
        "behavioral": behavioral, "rng_schedule": rng_ref,
        "output_contracts": output_contract_ref,
    }, seal_digest=True)
    claim["execution_protocol"] = protocol_ref
    claim_sha = gate1._claim_digest(claim)
    executions = []
    for case in corpus["cases"]:
        for scope in claim["mapped_outputs"]:
            for side, endpoint in (("source", source), ("destination", destination)):
                execution = {
                    "schema": 1, "kind": "equivalence_execution",
                    "claim_sha256": claim_sha, "protocol_sha256": protocol_ref["sha256"],
                    "case_id": case["id"], "scope": scope, "side": side,
                    "checkpoint_sha256": endpoint["artifact"]["sha256"],
                    "producer_implementation_sha256": endpoint["inference_implementation"]["sha256"],
                    "executor_implementation_sha256": executor_ref["sha256"],
                    "numeric_runtime": claim["numeric_runtime"],
                    "rng_schedule_sha256": (rng_ref["sha256"] if rng_ref else None),
                    "input": case["input"], "output_contract_sha256": output_contract_ref["sha256"],
                }
                execution["context_sha256"] = gate1._equivalence_context(
                    execution, output_contract_ref["sha256"],
                )
                output_ref = emit(tmp_path / f"{prefix}-{case['id']}-{scope}-{side}-output.json", {
                    "schema": 1, "kind": "equivalence_raw_output", "scope": scope,
                    "dtype": claim["numeric_runtime"]["dtype"], "shape": [2],
                    "components": ["component-0", "component-1"], "values": [1.0, 2.0],
                    "mask": ({"shape": [2], "components": ["action-0", "action-1"],
                              "values": [True, False]} if behavioral else None),
                    "decoded_action": ({"index": 0} if behavioral else None),
                    "output_contract_sha256": output_contract_ref["sha256"],
                    "context_sha256": execution["context_sha256"],
                }, seal_digest=True)
                execution["output"] = output_ref
                executions.append(emit(
                    tmp_path / f"{prefix}-{case['id']}-{scope}-{side}-execution.json",
                    execution, seal_digest=True,
                ))
    result_ref = emit(tmp_path / f"{prefix}-result.json", {
        "schema": 1, "kind": "equivalence_execution_result", "claim_sha256": claim_sha,
        "protocol_sha256": protocol_ref["sha256"], "corpus_sha256": claim["corpus"]["sha256"],
        "executions": executions,
    }, seal_digest=True)
    claim["result"] = result_ref
    edge["equivalence"] = claim
    refresh_gate1_protocols(document, tmp_path)
    reseal(document)
    return {"claim": claim, "protocol": protocol_ref, "output_contracts": output_contract_ref,
            "result": result_ref, "executions": executions, "behavioral": behavioral}


def mutate_raw_execution(evidence: dict, tmp_path: Path, index: int, *,
                         output_mutation=None, execution_mutation=None,
                         nonfinite_output: bool = False) -> None:
    """Rewrite one raw output/execution and reseal the containing result."""
    result_ref = evidence["result"]
    result = json.loads((tmp_path / result_ref["path"]).read_text())
    execution_ref = result["executions"][index]
    execution = json.loads((tmp_path / execution_ref["path"]).read_text())
    if output_mutation is not None or nonfinite_output:
        output_ref = execution["output"]
        output_path = tmp_path / output_ref["path"]
        output = json.loads(output_path.read_text())
        output.pop("digest", None)
        if output_mutation is not None:
            output_mutation(output)
            output["digest"] = digest(output)
            output_path.write_text(json.dumps(output, sort_keys=True))
        else:
            output["values"][0] = float("nan")
            output["digest"] = "0" * 64
            output_path.write_text(json.dumps(output, sort_keys=True))
        output_ref["sha256"] = checkpoint_hash(output_path)
        execution["output"] = output_ref
    if execution_mutation is not None:
        execution_mutation(execution)
    execution.pop("digest", None)
    execution["digest"] = digest(execution)
    execution_path = tmp_path / execution_ref["path"]
    execution_path.write_text(json.dumps(execution, sort_keys=True))
    execution_ref["sha256"] = checkpoint_hash(execution_path)
    result["executions"][index] = execution_ref
    result.pop("digest", None)
    result["digest"] = digest(result)
    result_path = tmp_path / result_ref["path"]
    result_path.write_text(json.dumps(result, sort_keys=True))
    result_ref["sha256"] = checkpoint_hash(result_path)


def apply_output_attack(evidence: dict, tmp_path: Path, attack: str) -> None:
    if attack == "output-ref-swap":
        result_ref = evidence["result"]
        result = json.loads((tmp_path / result_ref["path"]).read_text())
        source_ref, destination_ref = result["executions"][:2]
        source = json.loads((tmp_path / source_ref["path"]).read_text())

        def swap(execution):
            execution["output"] = source["output"]

        rewrite_sealed_ref(tmp_path, destination_ref, swap)
        result["executions"][1] = destination_ref
        result.pop("digest", None)
        result["digest"] = digest(result)
        result_path = tmp_path / result_ref["path"]
        result_path.write_text(json.dumps(result, sort_keys=True))
        result_ref["sha256"] = checkpoint_hash(result_path)
        return

    def mutate(output):
        if attack == "equal-overflow":
            output["values"] = [1e308, 1e308]
        elif attack == "dtype":
            output["dtype"] = "float64"
        elif attack == "truncation":
            output["shape"] = [1]
            output["components"] = ["component-0"]
            output["values"] = [1.0]
        elif attack == "wrong-mask":
            output["mask"] = {"shape": [1], "components": ["action-0"], "values": [True]}
        elif attack == "bad-action":
            output["decoded_action"] = {"index": 99}
        elif attack == "component-reorder":
            output["components"] = ["component-1", "component-0"]
        elif attack == "precision":
            output["values"] = [0.1, 2.0]
        elif attack == "schema-ref-swap":
            output["output_contract_sha256"] = "0" * 64
        else:
            raise AssertionError(attack)

    for index in (0, 1):
        mutate_raw_execution(evidence, tmp_path, index, output_mutation=mutate)


def comparator_refs(tmp_path: Path, document: dict, source_worker: dict, live_worker: dict) -> tuple[dict, dict]:
    reviewer = {"path": "reviewer.json", "sha256": checkpoint_hash(tmp_path / "reviewer.json")}
    source_artifact = write(tmp_path / "comparator-source.json", {"implementation": "comparator"})
    implementation = write(tmp_path / "comparator-implementation.json", {
        "schema": 1, "kind": "producer_comparator_implementation", "identity": "fixture-comparator",
        "commit": "d" * 40, "source": source_artifact,
    }, seal_digest=True)
    dimensions = ["reset", "configuration", "action", "observation", "reward", "state_trajectory"]
    inputs = {name: write(tmp_path / f"comparator-input-{name}.json", {"dimension": name})
              for name in dimensions}
    corpus = write(tmp_path / "comparator-corpus.json", {
        "schema": 1, "kind": "producer_comparator_corpus", "cases": [{"id": "case-1", "inputs": inputs}],
    }, seal_digest=True)
    protocol = {"schema": 1, "kind": "producer_comparator_protocol", "dimensions": dimensions,
                "implementation": implementation, "corpus": corpus, "review": None}
    protocol_ref = write_reviewed_protocol(tmp_path / "comparator-protocol.json", protocol, reviewer, dimensions)
    pane = document["evaluation"]["panes"][0]
    source_impl = document["nodes"][0]["inference_implementation"]
    live_impl = pane["evaluation_implementation"]
    executions = []
    for dimension in dimensions:
        output = write(tmp_path / f"comparator-output-{dimension}.json", {"equal-output": dimension})
        for side, worker, producer_impl in (("source", source_worker, source_impl),
                                            ("live", live_worker, live_impl)):
            executions.append(write(tmp_path / f"comparator-{side}-{dimension}.json", {
                "schema": 1, "kind": "producer_comparator_execution",
                "protocol_sha256": protocol_ref["sha256"], "case_id": "case-1", "dimension": dimension,
                "side": side, "worker_digest": worker["digest"],
                "producer_implementation_sha256": producer_impl["sha256"],
                "comparator_implementation_sha256": implementation["sha256"],
                "input": inputs[dimension], "output": output,
            }, seal_digest=True))
    receipt = write(tmp_path / "comparator-receipt.json", {
        "schema": 1, "kind": "producer_comparator_receipt", "protocol_sha256": protocol_ref["sha256"],
        "source_worker_digest": source_worker["digest"], "live_worker_digest": live_worker["digest"],
        "executions": executions,
    }, seal_digest=True)
    return protocol_ref, receipt


def test_strict_manifest_validates_raw_dag_and_computes_tensor_equivalence(tmp_path):
    document, _ = fixture(tmp_path)
    result = gate1.validate_manifest(document, tmp_path)
    assert result["strict_no_upgrade"] is True
    assert result["topological_order"][:2] == ["prior", "clone"]
    assert set(result["topological_order"][2:]) == set(trained_ids(document))
    assert result["equivalence"][0]["status"] == "PASS"
    assert result["holds"] == []


def test_manifest_rejects_subset_raw_schema_declaration(tmp_path):
    document, _ = fixture(tmp_path)
    document["nodes"][0]["raw_schema"] = {"checkpoint_contract": contracts.NATIVE_CONTRACT}
    with pytest.raises(ProvenanceError, match="raw schema differs"):
        gate1.validate_manifest(reseal(document), tmp_path)


def test_optional_not_applicable_conversion_is_explicit_and_nonblocking(tmp_path):
    document, _ = fixture(tmp_path)
    result = gate1.validate_manifest(document, tmp_path)
    assert result["holds"] == []
    conversion = next(row for row in result["lineage_inventory"]["slots"]
                      if row["slot"] == "conversion")
    assert conversion["status"] == "not-applicable"
    assert conversion["historical_support"]["sha256"]
    assert gate1._classify(document, {}, "behavior")["classification"] == "insufficient/incompatible"


def test_manifest_rejects_empty_comparison_order(tmp_path):
    document, _ = fixture(tmp_path)
    document["comparison_order"] = []
    with pytest.raises(ProvenanceError, match="reviewed continuous lineage sequences"):
        gate1.validate_manifest(reseal(document), tmp_path)


def test_manifest_rejects_comparison_order_omitting_required_present_edge(tmp_path):
    document, _ = fixture(tmp_path)
    document["comparison_order"] = ["prior-clone"]
    with pytest.raises(ProvenanceError, match="reviewed continuous lineage sequences"):
        gate1.validate_manifest(reseal(document), tmp_path)


def test_exact_tensor_preservation_compares_literal_float_bits():
    assert not gate1._literal_tensor_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    left = torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32)
    right = torch.tensor([0x7FC00002], dtype=torch.int32).view(torch.float32)
    assert torch.isnan(left).all() and torch.isnan(right).all()
    assert not gate1._literal_tensor_equal(left, right)


@pytest.mark.parametrize("case", ["signed-zero", "nan-payload"])
def test_exact_tensor_claim_rejects_numerically_equal_but_byte_distinct_values(tmp_path, case):
    document, _ = fixture(tmp_path)
    if case == "signed-zero":
        values = (torch.tensor([0.0]), torch.tensor([-0.0]))
    else:
        values = (torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32),
                  torch.tensor([0x7FC00002], dtype=torch.int32).view(torch.float32))
    for node, value in zip(document["nodes"][:2], values, strict=True):
        checkpoint = tmp_path / node["artifact"]["path"]
        state = torch.load(checkpoint, weights_only=True)
        replacement = torch.zeros_like(state["actor"]["strategic.bias"])
        replacement[0] = value[0]
        state["actor"]["strategic.bias"] = replacement
        torch.save(state, checkpoint); node["artifact"]["sha256"] = checkpoint_hash(checkpoint)
        defaults_path = tmp_path / node["loading"]["default_resolution"]["path"]
        defaults = json.loads(defaults_path.read_text())
        defaults["checkpoint_sha256"] = node["artifact"]["sha256"]
        defaults["digest"] = digest({key: item for key, item in defaults.items() if key != "digest"})
        defaults_path.write_text(json.dumps(defaults, sort_keys=True))
        node["loading"]["default_resolution"]["sha256"] = checkpoint_hash(defaults_path)
    refresh_gate1_protocols(document, tmp_path)
    result = gate1.validate_manifest(reseal(document), tmp_path)
    assert result["equivalence"][0]["status"] == "FAIL"


def test_required_missing_node_and_edge_block(tmp_path):
    document, _ = fixture(tmp_path)
    trained_id = trained_ids(document)[0]
    trained = next(row for row in document["nodes"] if row["id"] == trained_id)
    original_identity = copy.deepcopy(trained["run_identity"])
    trained.clear()
    trained.update({"id": trained_id, "role": "trained-t3", "status": "missing",
                    "reason": "checkpoint has not been recovered",
                    "evidence_class": "historical-record",
                    "run_identity": original_identity})
    edge_id = f"clone-{trained_id}"
    edge = next(row for row in document["edges"] if row["id"] == edge_id)
    edge.clear()
    edge.update({"id": edge_id, "status": "missing",
                 "reason": "continuation evidence has not been recovered", "source": "clone",
                 "destination": trained_id, "relation": "ppo_continuation"})
    rewrite_reviewed_protocol(tmp_path, document["lineage_inventory"], lambda body: (
        next(row for row in body["nodes"] if row["id"] == trained_id).update(
            status="missing", historical_panes=[]),
        next(row for row in body["edges"] if row["id"] == edge_id).update(
            status="missing", historical_panes=[]),
        next(row for row in body["slots"] if row["slot"] == "trained-t3").update(
            status="missing", reason="checkpoint has not been recovered",
            historical_support=write(tmp_path / "trained-missing.json", {"missing": "trained"})),
    ))
    document["evaluation"]["panes"][0]["nodes"].remove(trained_id)
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    result = gate1.validate_manifest(reseal(document), tmp_path)
    assert f"node:{trained_id}:missing" in result["holds"]
    assert f"edge:{edge_id}:missing" in result["holds"]


@pytest.mark.parametrize("field", ["learned_movement", "region_movement"])
def test_movement_semantics_are_independently_checked_against_raw_metadata(tmp_path, field):
    document, _ = fixture(tmp_path)
    node = document["nodes"][0]
    checkpoint = tmp_path / node["artifact"]["path"]
    state = torch.load(checkpoint, weights_only=True)
    state[field] = True
    torch.save(state, checkpoint)
    node["artifact"]["sha256"] = checkpoint_hash(checkpoint)
    node["raw_schema"], _, _ = gate1.inspect_checkpoint(node["artifact"], tmp_path)
    ref = node["loading"]["default_resolution"]
    path = tmp_path / ref["path"]
    body = json.loads(path.read_text())
    body["checkpoint_sha256"] = node["artifact"]["sha256"]
    body["digest"] = digest({k: v for k, v in body.items() if k != "digest"})
    path.write_text(json.dumps(body, sort_keys=True))
    ref["sha256"] = checkpoint_hash(path)
    reseal(document)
    with pytest.raises(ProvenanceError, match=f"historical semantic {field} differs"):
        gate1.validate_manifest(document, tmp_path)


@pytest.mark.parametrize("mutation, message", [
    ("cycle", "cycle"), ("future", "unknown/future"), ("implicit", "implicit"),
    ("sealed", "confirmation seed"), ("conflict", "conflicting semantic"),
])
def test_manifest_rejects_cycles_unknown_schemas_implicit_defaults_sealed_seeds_and_conflicts(tmp_path, mutation, message):
    document, _ = fixture(tmp_path)
    if mutation == "cycle":
        document["edges"].append({"id": "back", "status": "present", "source": "clone", "destination": "prior",
                                  "relation": "evaluation_reference", "evidence": document["edges"][0]["evidence"]})
    elif mutation == "future":
        path = tmp_path / document["nodes"][0]["artifact"]["path"]
        state = torch.load(path, weights_only=True); state["contract"] = contracts.NATIVE_CONTRACT + 1; torch.save(state, path)
        document["nodes"][0]["artifact"]["sha256"] = checkpoint_hash(path)
    elif mutation == "implicit":
        ref = document["nodes"][0]["loading"]["default_resolution"]
        path = tmp_path / ref["path"]; body = json.loads(path.read_text()); body["implicit_defaults"] = ["token"]
        body["digest"] = digest({k: v for k, v in body.items() if k != "digest"}); path.write_text(json.dumps(body))
        ref["sha256"] = checkpoint_hash(path)
    elif mutation == "sealed":
        document["evaluation"]["panes"][0]["worlds"][0]["seed"] = 3000
    else:
        # Reuse identical bytes under conflicting actor semantics.
        document["nodes"][1]["artifact"] = copy.deepcopy(document["nodes"][0]["artifact"])
        document["nodes"][1]["raw_schema"] = copy.deepcopy(document["nodes"][0]["raw_schema"])
        document["nodes"][1]["historical_semantics"]["commitments"] = False
        ref = document["nodes"][1]["loading"]["default_resolution"]
        path = tmp_path / ref["path"]; body = json.loads(path.read_text())
        body["checkpoint_sha256"] = document["nodes"][1]["artifact"]["sha256"]
        body["resolved"]["commitments"]["value"] = False
        body["digest"] = digest({k: v for k, v in body.items() if k != "digest"}); path.write_text(json.dumps(body))
        ref["sha256"] = checkpoint_hash(path)
    reseal(document)
    if mutation == "cycle":
        message = "inventory differs"
    with pytest.raises(ProvenanceError, match=message):
        gate1.validate_manifest(document, tmp_path)


def test_reconstructed_migration_cannot_claim_historical_conversion(tmp_path):
    document, _ = fixture(tmp_path)
    adapter = write(tmp_path / "adapter.py", {"adapter": 1})
    defaults = document["nodes"][1]["loading"]["default_resolution"]
    migration = {"schema": 1, "kind": "checkpoint_migration", "status": "reconstructed",
                 "historical_conversion": True, "source_checkpoint_sha256": document["nodes"][0]["artifact"]["sha256"],
                 "destination_checkpoint_sha256": document["nodes"][1]["artifact"]["sha256"],
                 "source_contract": contracts.NATIVE_CONTRACT, "destination_contract": contracts.NATIVE_CONTRACT,
                 "adapter": adapter, "default_resolution": defaults}
    migration_ref = write(tmp_path / "migration.json", migration, seal_digest=True)
    edge = document["edges"][0]; edge["relation"] = "schema_conversion"; edge["migration_record"] = migration_ref
    document["nodes"][1]["historical_status"] = "reconstructed"
    document["nodes"][1]["claims_historical_conversion"] = False
    rewrite_reviewed_protocol(tmp_path, document["lineage_inventory"], lambda body: (
        next(row for row in body["edges"] if row["id"] == "prior-clone").update(
            relation="schema_conversion"),
        next(row for row in body["slots"] if row["slot"] == "conversion").update(
            status="present", edge_ids=["prior-clone"], reason=None, historical_support=None),
    ))
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    with pytest.raises(ProvenanceError, match="cannot claim historical"):
        gate1.validate_manifest(document, tmp_path)


@pytest.mark.parametrize("capability", ["snapshot", "stochastic", "integrated"])
def test_evaluation_executes_bound_gate1_capabilities(tmp_path, capability):
    document, worker = fixture(tmp_path); pane = document["evaluation"]["panes"][0]
    snapshot_bytes = None
    if capability == "snapshot":
        pane["worlds"][0]["snapshot"] = write(tmp_path / "snapshot.bin", {"snapshot": 1})
        pane["worlds"][0]["turn"] = pane["worlds"][0]["reset_turn"]
        snapshot_bytes = (tmp_path / "snapshot.bin").read_bytes()
    elif capability == "stochastic":
        from anthropocene_rl.autonomous_v7.gate1_diagnostics import (
            draw_order_from_semantics, generator_state_sha256,
        )
        pane["action_mode"] = "seeded-stochastic"
        world_id = gate1._world_id(pane["worlds"][0])
        steps = []
        for index in range(pane["worlds"][0]["horizon"]):
            seed = 900 + index
            generator = torch.Generator(device="cpu").manual_seed(seed)
            steps.append({"step": index, "seed": seed,
                          "state_sha256": generator_state_sha256(generator)})
        node_schedules = [{"node_id": node_id,
                           "draw_order": draw_order_from_semantics(SEMANTICS),
                           "steps": copy.deepcopy(steps)} for node_id in pane["nodes"]]
        pane["policy_rng_schedule"] = write(
            tmp_path / "rng.json",
            {"schema": 1, "kind": "gate1_policy_rng_schedule",
             "algorithm": "torch.Generator.cpu-default", "torch_version": torch.__version__,
             "device": "cpu", "worlds": [{"world_id": world_id, "nodes": node_schedules}]},
            seal_digest=True,
        )
    else:
        pane["metrics"].append("population_years")
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    captured = {}

    def runner(*args, **kwargs):
        captured.update(kwargs)
        values = mandatory_values(); values["population_years"] = 3.0
        result = diagnostic_result(
            gate1._frozen_world_record(pane["worlds"][0]), gate1._world_id(pane["worlds"][0]),
            pane["worlds"][0]["horizon"], pane["capture_bounds"], values,
            action_mode=pane["action_mode"], rng_steps=kwargs.get("policy_rng_steps"),
            semantics=kwargs["evaluation_semantics"],
            observation_core_width=pane["observation_core_width"],
        )
        return {**result, "worker_contract": worker, "world_contract": args[14],
                "producer": gate1._producer_identity_for_worker(worker),
                "snapshot_sha256": args[14]["snapshot_sha256"], "action_mode": pane["action_mode"]}

    report = gate1.evaluate_node(document, tmp_path, "prior", "historical", runner)
    assert report["rows"][0]["metrics"]["welfare"] == 1.0
    if capability == "snapshot":
        assert captured["snapshot"] == snapshot_bytes
    elif capability == "stochastic":
        assert [
            {key: row[key] for key in ("step", "seed", "state_sha256")}
            for row in captured["policy_rng_steps"]
        ] == steps
        assert gate1._evaluation_index(document, [report], tmp_path)
    else:
        assert report["rows"][0]["metrics"]["population_years"] == 3.0


def test_evaluation_rejects_incompatible_producer_without_comparator(tmp_path):
    document, _ = fixture(tmp_path)
    other = seal({"version": 1, "producer": "other"})
    ref = write(tmp_path / "other-worker.json", {"schema": 1, "kind": "worker_contract", "contract": other})
    document["evaluation"]["panes"][0]["evaluation_worker_contract"] = ref
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    with pytest.raises(ProvenanceError, match="cannot bind the reported producer"):
        gate1.evaluate_node(document, tmp_path, "prior", "historical", lambda *a, **k: {})


def test_imported_evidence_rechecks_historical_worker_compatibility(tmp_path, monkeypatch):
    document, worker = fixture(tmp_path)
    for name in ("ANTHROPOCENE_REGION_HEAD", "ANTHROPOCENE_SIM_RULES", "ANTHROPOCENE_SIM_STATICS"):
        monkeypatch.delenv(name, raising=False)
    report = evaluation_report(document, tmp_path, "prior", worker)
    live_body = {key: copy.deepcopy(value) for key, value in worker.items() if key != "digest"}
    live_body["bridge_sha256"] = "5" * 64
    live = seal(live_body)
    pane = document["evaluation"]["panes"][0]
    pane["evaluation_worker_contract"] = write(
        tmp_path / "resealed-worker.json", {"schema": 1, "kind": "worker_contract", "contract": live},
    )
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    report["manifest_digest"] = document["digest"]
    report["rows"][0]["worker_contract"] = live
    report["rows"][0]["producer"] = gate1._producer_identity_for_worker(live)
    report["digest"] = digest({key: value for key, value in report.items() if key != "digest"})
    with pytest.raises(ProvenanceError):
        gate1._evaluation_index(document, [report], tmp_path)

    protocol_ref, receipt_ref = comparator_refs(tmp_path, document, worker, live)
    pane["semantic_comparator_protocol"] = protocol_ref
    pane["semantic_comparator"] = receipt_ref
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    report["manifest_digest"] = document["digest"]
    report["digest"] = digest({key: value for key, value in report.items() if key != "digest"})
    assert len(gate1._evaluation_index(document, [report], tmp_path)) == 1


@pytest.mark.parametrize("attack", [
    "empty", "incomplete", "duplicate", "short", "missing-terminal-reason",
    "fabricated-headlines", "wrong-unit", "negative-output",
    "fractional-trade-count", "missing-herds", "successor-prior-suppression",
    "false-terminal", "opening-turn-mismatch", "null-actions", "empty-observation",
    "wrong-society-id", "wrong-trade-unit", "missing-units", "extra-unit",
    "wrong-reset-trajectory-id", "broken-opening-chain", "invented-founder",
    "mismatched-founder-witness", "reflag-existing-founder", "illegal-clothing-route",
    "fabricated-trade-boundary", "deterministic-unsnapped-levels",
    "invented-migration-founder", "migration-copies-deleted",
    "migration-kind-renamed", "migration-population-inflation",
])
def test_imported_evidence_recomputes_complete_typed_trajectory(tmp_path, attack):
    document, worker = fixture(tmp_path)
    pane = document["evaluation"]["panes"][0]
    pane["metrics"] = sorted(gate1.MANDATORY_METRICS)
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    report = manual_report(document, "prior", pane["id"], worker, mandatory_values())
    apply_trajectory_attack(report, attack)
    with pytest.raises(ProvenanceError, match="diagnostic surface|outcomes/receipts"):
        gate1._evaluation_index(document, [report], tmp_path)


def test_imported_evidence_rejects_generic_legacy_loader_blob(tmp_path):
    document, worker = fixture(tmp_path)
    node = document["nodes"][0]
    node["loading"] = {
        "mode": "pinned-historical", "implementation": node["producer_evidence"],
        "default_resolution": node["loading"]["default_resolution"],
    }
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    report = manual_report(
        document, node["id"], document["evaluation"]["panes"][0]["id"],
        worker, mandatory_values(),
    )
    with pytest.raises(ProvenanceError, match="supported reviewed adapter"):
        gate1._evaluation_index(document, [report], tmp_path)


def test_imported_economic_unit_validation_is_observationally_immutable(tmp_path):
    document, worker = fixture(tmp_path)
    pane = document["evaluation"]["panes"][0]
    pane["metrics"] = sorted(gate1.MANDATORY_METRICS)
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    report = manual_report(document, "prior", pane["id"], worker, mandatory_values())
    before = copy.deepcopy(report)
    assert gate1._evaluation_index(document, [report], tmp_path)
    assert report == before


def test_import_and_packet_reject_forced_region_selection_substitution(tmp_path):
    document, worker = fixture(tmp_path)
    prior = next(node for node in document["nodes"] if node["id"] == "prior")
    prior["historical_semantics"].update(
        learned_movement=True, region_movement=True, region_selection="stay",
    )
    rewrite_sealed_ref(
        tmp_path, prior["loading"]["default_resolution"],
        lambda body: body["resolved"].update({
            name: {"source": "checkpoint-or-manifest", "value": value}
            for name, value in prior["historical_semantics"].items()
        }),
    )
    pane = document["evaluation"]["panes"][0]
    pane["metrics"] = sorted(gate1.MANDATORY_METRICS)
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    pristine = manual_report(document, "prior", pane["id"], worker, mandatory_values())
    assert gate1._evaluation_index(document, [pristine], tmp_path)

    forged = copy.deepcopy(pristine)
    action = forged["rows"][0]["trajectory_evidence"]["steps"][0]["actions"][0]
    action["action_request"]["movement"] = contracts.REGION_BASE + 3
    action["decoded_action"]["movement"] = contracts.REGION_BASE + 3
    reseal_trajectory_report(forged)
    with pytest.raises(ProvenanceError, match="diagnostic surface|outcomes/receipts"):
        gate1._evaluation_index(document, [forged], tmp_path)

    reports = [manual_report(document, node, pane["id"], worker, mandatory_values())
               for node in pane["nodes"]]
    reports[0] = forged
    refs = [write(tmp_path / f"forced-region-{report['node']}.json", report)
            for report in reports]
    with pytest.raises(ProvenanceError):
        gate1.build_review_packet(
            document, refs, tmp_path, cli=["packet"],
            verification=verification_ref(tmp_path, document),
            implementation=implementation_ref(tmp_path),
            manifest_ref=write(tmp_path / "forced-region-manifest.json", document),
        )


@pytest.mark.parametrize("attack", [
    "empty", "incomplete", "duplicate", "short", "missing-terminal-reason",
    "fabricated-headlines", "wrong-unit", "negative-output",
    "fractional-trade-count", "missing-herds", "successor-prior-suppression",
    "false-terminal", "opening-turn-mismatch", "null-actions", "empty-observation",
    "wrong-society-id", "wrong-trade-unit", "missing-units", "extra-unit",
    "wrong-reset-trajectory-id", "broken-opening-chain", "invented-founder",
    "mismatched-founder-witness", "reflag-existing-founder", "illegal-clothing-route",
    "fabricated-trade-boundary", "deterministic-unsnapped-levels",
    "invented-migration-founder", "migration-copies-deleted",
    "migration-kind-renamed", "migration-population-inflation", "generic-loader",
])
def test_complete_packet_rejects_forged_diagnostic_evidence(tmp_path, attack):
    document, worker = fixture(tmp_path)
    pane = document["evaluation"]["panes"][0]
    pane["metrics"] = sorted(gate1.MANDATORY_METRICS)
    if attack == "generic-loader":
        node = document["nodes"][0]
        node["loading"] = {
            "mode": "pinned-historical", "implementation": node["producer_evidence"],
            "default_resolution": node["loading"]["default_resolution"],
        }
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    reports = [manual_report(document, node, pane["id"], worker, mandatory_values())
               for node in pane["nodes"]]
    if attack != "generic-loader":
        apply_trajectory_attack(reports[0], attack)
    evaluation_refs = [write(tmp_path / f"{attack}-{report['node']}.json", report)
                       for report in reports]
    with pytest.raises(ProvenanceError):
        gate1.build_review_packet(
            document, evaluation_refs, tmp_path, cli=["packet"],
            verification=verification_ref(tmp_path, document),
            implementation=implementation_ref(tmp_path),
            manifest_ref=write(tmp_path / f"{attack}-manifest.json", document),
        )


@pytest.mark.parametrize("unsupported", ["personality-dict", "native-background"])
def test_evaluation_rejects_unexecutable_frozen_pane_declarations(tmp_path, unsupported):
    document, _ = fixture(tmp_path)
    pane = document["evaluation"]["panes"][0]
    if unsupported == "personality-dict":
        pane["personality"] = {"1": [1.0, 0.0, 0.0, 0.0]}
        message = "supported frozen vector"
    else:
        pane["background_policy"] = {"kind": "native", "digest": digest({"kind": "native"})}
        message = "background policy is unsupported"
    with pytest.raises(ProvenanceError, match=message):
        gate1.validate_manifest(reseal(document), tmp_path)


def test_learned_movement_controls_rollout_while_region_movement_controls_decoder(tmp_path, monkeypatch):
    document, worker = fixture(tmp_path)
    for node in document["nodes"]:
        node["historical_semantics"]["learned_movement"] = True
        ref = node["loading"]["default_resolution"]
        path = tmp_path / ref["path"]
        body = json.loads(path.read_text())
        body["resolved"]["learned_movement"]["value"] = True
        body["digest"] = digest({k: v for k, v in body.items() if k != "digest"})
        path.write_text(json.dumps(body, sort_keys=True))
        ref["sha256"] = checkpoint_hash(path)
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    for name in ("ANTHROPOCENE_REGION_HEAD", "ANTHROPOCENE_SIM_RULES", "ANTHROPOCENE_SIM_STATICS"):
        monkeypatch.delenv(name, raising=False)
    captured = {}

    def runner(*args, **kwargs):
        captured.update({"movement": args[4], "region": args[6], "world": args[14], "kwargs": kwargs})
        result = diagnostic_result(
            gate1._frozen_world_record(document["evaluation"]["panes"][0]["worlds"][0]),
            gate1._world_id(document["evaluation"]["panes"][0]["worlds"][0]), args[1],
            kwargs["capture_bounds"], mandatory_values(), action_mode=kwargs["action_mode"],
            rng_steps=kwargs["policy_rng_steps"],
            semantics=kwargs["evaluation_semantics"],
            observation_core_width=kwargs["observation_core_width"],
        )
        return {**result, "worker_contract": worker,
                "world_contract": args[14], "producer": gate1._producer_identity_for_worker(worker)}

    report = gate1.evaluate_node(document, tmp_path, "prior", "historical", runner)
    assert captured["movement"] is True
    assert captured["region"] is False
    assert captured["world"]["movement"] is True
    assert report["semantics"]["learned_movement"] is True
    assert report["semantics"]["region_movement"] is False


@pytest.mark.parametrize(("mutation", "message"), [
    ("world", "exact frozen request"),
    ("personality", "personality/background scope"),
    ("background", "personality/background scope"),
    ("producer", "exact bound producer"),
])
def test_packet_ingestion_rejects_altered_world_personality_and_background(
        tmp_path, monkeypatch, mutation, message):
    document, worker = fixture(tmp_path)
    for name in ("ANTHROPOCENE_REGION_HEAD", "ANTHROPOCENE_SIM_RULES", "ANTHROPOCENE_SIM_STATICS"):
        monkeypatch.delenv(name, raising=False)
    report = evaluation_report(document, tmp_path, "prior", worker)
    if mutation == "world":
        descriptor = dict(report["rows"][0]["world_contract"])
        descriptor["seed"] = 4001
        descriptor["digest"] = digest({k: v for k, v in descriptor.items() if k != "digest"})
        report["rows"][0]["world_contract"] = descriptor
    elif mutation == "personality":
        report["personality"] = [0.0, 1.0, 0.0, 0.0]
    elif mutation == "background":
        body = {"kind": "none", "actor_scope": "only-one-village"}
        report["background_policy"] = {**body, "digest": digest(body)}
    else:
        report["rows"][0]["producer"]["bridge_sha256"] = "5" * 64
    report["digest"] = digest({k: v for k, v in report.items() if k != "digest"})
    with pytest.raises(ProvenanceError, match=message):
        gate1.build_review_packet(
            document, [report], tmp_path, cli=["packet"],
            verification=verification_ref(tmp_path, document),
            implementation=implementation_ref(tmp_path),
        )


def test_packet_is_deterministic_and_holds_when_required_capabilities_are_missing(tmp_path):
    document, _ = fixture(tmp_path)
    implementation = implementation_ref(tmp_path)
    verification = verification_ref(tmp_path, document)
    a = gate1.build_review_packet(document, [], tmp_path, cli=["packet"], verification=verification,
                                  implementation=implementation)
    b = gate1.build_review_packet(document, [], tmp_path, cli=["packet"], verification=verification,
                                  implementation=implementation)
    assert a == b and a["digest"] == b["digest"] and a["decision"] == "HOLD"
    assert a["behavior"]["classification"] == "insufficient/incompatible"
    out = gate1.write_review_packet(a, tmp_path / "packets")
    assert json.loads((out / "completion.json").read_text())["packet_digest"] == a["digest"]


def test_verification_thresholds_are_manifest_bound_and_raw_reports_are_in_inventory(tmp_path):
    document, _ = fixture(tmp_path)
    implementation = implementation_ref(tmp_path)
    verification = verification_ref(tmp_path, document)
    packet = gate1.build_review_packet(document, [], tmp_path, cli=["packet"],
                                       verification=verification, implementation=implementation)
    verification_body = json.loads((tmp_path / verification["path"]).read_text())
    coverage = json.loads((tmp_path / verification_body["coverage"]["path"]).read_text())
    parity = json.loads((tmp_path / verification_body["parity"]["path"]).read_text())
    benchmark = json.loads((tmp_path / verification_body["benchmarks"][0]["path"]).read_text())
    test_receipt = json.loads((tmp_path / verification_body["test_receipts"][0]["path"]).read_text())
    raw_hashes = {coverage["raw_measurements"]["sha256"], benchmark["raw_measurements"]["sha256"],
                  test_receipt["output"]["sha256"]}
    for row in parity["scenarios"]:
        raw_hashes.update((row["baseline"]["sha256"], row["diagnostic_off"]["sha256"]))
    assert raw_hashes <= set(packet["artifact_hashes"])
    assert verification["sha256"] in packet["artifact_hashes"]
    assert implementation["sha256"] in packet["artifact_hashes"]

    protocol_path = tmp_path / document["verification_protocol"]["path"]
    protocol = json.loads(protocol_path.read_text())
    protocol["coverage"]["minimum_percent"] = 0.1
    protocol["digest"] = digest({key: value for key, value in protocol.items() if key != "digest"})
    protocol_path.write_text(json.dumps(protocol, sort_keys=True))
    with pytest.raises(ProvenanceError, match="hash mismatch"):
        gate1.build_review_packet(document, [], tmp_path, cli=["packet"],
                                  verification=verification, implementation=implementation)


def test_missing_verification_artifact_forces_hold(tmp_path):
    document, _ = fixture(tmp_path)
    packet = gate1.build_review_packet(document, [], tmp_path, cli=["packet"], verification=None,
                                       implementation=implementation_ref(tmp_path))
    assert packet["decision"] == "HOLD"
    assert "Gate 1 verification artifact is missing" in packet["unsupported_claims"]


def test_complete_partial_metric_cohort_cannot_pass_gate1(tmp_path, monkeypatch):
    document, worker = fixture(tmp_path)
    for name in ("ANTHROPOCENE_REGION_HEAD", "ANTHROPOCENE_SIM_RULES", "ANTHROPOCENE_SIM_STATICS"):
        monkeypatch.delenv(name, raising=False)
    reports = [evaluation_report(document, tmp_path, node, worker)
               for node in document["evaluation"]["panes"][0]["nodes"]]
    clean_tree = gate1.implementation_source_tree()
    clean_tree["dirty_state"] = {"clean": True, "scope": "entire-repository", "porcelain": [],
                                 "porcelain_sha256": digest([])}
    monkeypatch.setattr(gate1, "implementation_source_tree", lambda: copy.deepcopy(clean_tree))
    packet = gate1.build_review_packet(
        document, reports, tmp_path, cli=["packet"],
        verification=verification_ref(tmp_path, document),
        implementation=implementation_ref(tmp_path),
    )
    assert packet["missing"] == []
    assert packet["decision"] == "HOLD"
    assert packet["behavior"]["classification"] == "insufficient/incompatible"
    assert packet["outcome"]["classification"] == "insufficient/incompatible"


def test_early_change_cannot_hide_later_partial_metric_boundary(tmp_path, monkeypatch):
    document, worker = fixture(tmp_path)
    target = trained_ids(document)[0]
    target_edge = f"clone-{target}"
    first = document["evaluation"]["panes"][0]
    first["metrics"] = sorted(gate1.MANDATORY_METRICS)
    second = copy.deepcopy(first); second["id"] = "later"; second["nodes"] = ["clone", target]
    second["metrics"] = ["welfare"]
    first["nodes"] = [node for node in first["nodes"] if node != target]
    document["evaluation"]["panes"].append(second)
    rewrite_reviewed_protocol(tmp_path, document["lineage_inventory"], lambda body: (
        next(row for row in body["nodes"] if row["id"] == "prior").update(
            historical_panes=["historical"]),
        next(row for row in body["nodes"] if row["id"] == "clone").update(
            historical_panes=["historical", "later"]),
        next(row for row in body["nodes"] if row["id"] == target).update(
            historical_panes=["later"]),
        next(row for row in body["edges"] if row["id"] == "prior-clone").update(
            historical_panes=["historical"]),
        next(row for row in body["edges"] if row["id"] == target_edge).update(
            historical_panes=["later"]),
    ))
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    reports = [manual_report(document, node, "historical", worker,
                             mandatory_values(welfare=(2.0 if node == "clone" else 1.0)))
               for node in first["nodes"]]
    reports += [manual_report(document, "clone", "later", worker, {"welfare": 2.0}),
                manual_report(document, target, "later", worker, {"welfare": 3.0})]
    clean_tree = gate1.implementation_source_tree()
    clean_tree["dirty_state"] = {"clean": True, "scope": "entire-repository", "porcelain": [],
                                 "porcelain_sha256": digest([])}
    monkeypatch.setattr(gate1, "implementation_source_tree", lambda: copy.deepcopy(clean_tree))
    packet = gate1.build_review_packet(
        document, reports, tmp_path, cli=["packet"], verification=verification_ref(tmp_path, document),
        implementation=implementation_ref(tmp_path),
    )
    assert packet["decision"] == "HOLD"
    assert packet["behavior"]["classification"] == "insufficient/incompatible"
    assert any(row.get("edge") == target_edge and row.get("reason") == "missing mandatory metrics"
               for row in packet["completeness"])


def test_paired_aggregation_is_deterministic_and_marks_zero_denominators(tmp_path):
    document, worker = fixture(tmp_path)
    document["evaluation"]["panes"][0]["metrics"] = sorted(gate1.MANDATORY_METRICS)
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    source = mandatory_values(welfare=0.0, service=0.0)
    destination = mandatory_values(welfare=2.0, service=3.0)
    reports = [manual_report(document, "prior", "historical", worker, source),
               manual_report(document, "clone", "historical", worker, destination)]
    index = gate1._evaluation_index(document, reports, tmp_path)
    first = gate1._paired_aggregation(document, index)
    second = gate1._paired_aggregation(document, index)
    assert first == second and digest(first) == digest(second)
    pair = first["pairs"][0]
    assert pair["derived"]["welfare"]["ratio"] == {"status": "zero-denominator", "value": None}
    assert pair["derived"]["service_outcomes.cultivation.food-delivered"]["ratio"]["status"] == "zero-denominator"
    assert first["resampling"]["unit"] == "paired-world"
    assert first["resampling"]["optimizer_identities_pooled"] is False


def test_explicit_service_missingness_is_valid_evidence_but_holds_packet(tmp_path, monkeypatch):
    document, worker = fixture(tmp_path)
    document["evaluation"]["panes"][0]["metrics"] = sorted(gate1.MANDATORY_METRICS)
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    reports = [manual_report(document, node, "historical", worker, mandatory_values(service=None))
               for node in ("prior", "clone")]
    clean_tree = gate1.implementation_source_tree()
    clean_tree["dirty_state"] = {"clean": True, "scope": "entire-repository", "porcelain": [],
                                 "porcelain_sha256": digest([])}
    monkeypatch.setattr(gate1, "implementation_source_tree", lambda: copy.deepcopy(clean_tree))
    packet = gate1.build_review_packet(
        document, reports, tmp_path, cli=["packet"], verification=verification_ref(tmp_path, document),
        implementation=implementation_ref(tmp_path),
    )
    assert packet["decision"] == "HOLD"
    assert any(row["reason"] == "required service outcomes unavailable" for row in packet["completeness"])


def test_packet_holds_on_source_identity_mismatch_and_rejects_mutated_identity_artifact(tmp_path):
    document, _ = fixture(tmp_path)
    implementation = implementation_ref(tmp_path, mismatched_source=True, mismatched_dirty_state=True)
    verification = verification_ref(tmp_path, document)
    packet = gate1.build_review_packet(
        document, [], tmp_path, cli=["packet"], verification=verification, implementation=implementation,
    )
    assert packet["decision"] == "HOLD"
    assert any("source hash differs from executing source" in reason for reason in packet["unsupported_claims"])
    assert "implementation dirty-state proof differs from executing source tree" in packet["unsupported_claims"]
    path = tmp_path / implementation["path"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ProvenanceError, match="hash mismatch"):
        gate1.build_review_packet(
            document, [], tmp_path, cli=["packet"], verification=verification, implementation=implementation,
        )


def test_strict_evaluate_loader_never_calls_upgrade_or_infers_defaults(tmp_path, monkeypatch):
    from anthropocene_rl.autonomous_v7 import evaluate
    from anthropocene_rl.autonomous_v7.model import Actor, Normalizer
    actor = Actor(4, region_head="hierarchical")
    normalizer = Normalizer(np.zeros(4, np.float32), np.ones(4, np.float32))
    for name, _, width in contracts.SETS:
        normalizer.set_mean[name] = np.zeros(width, np.float32); normalizer.set_std[name] = np.ones(width, np.float32)
    path = tmp_path / "native.pt"
    torch.save({"contract": contracts.NATIVE_CONTRACT, "core_width": 4, "actor": actor.state_dict(),
                "normalizer": normalizer.state_dict()}, path)
    monkeypatch.setattr(evaluate, "upgrade", lambda state: (_ for _ in ()).throw(AssertionError("upgrade called")))
    loaded, _ = evaluate.load(path, strict=True, resolved_semantics=SEMANTICS)
    assert loaded.region_head == "hierarchical"
    with pytest.raises(ValueError, match="complete resolved"):
        evaluate.load(path, strict=True)
    wrong = dict(SEMANTICS, region_head="flat")
    with pytest.raises(ValueError, match="region head differs"):
        evaluate.load(path, strict=True, resolved_semantics=wrong)


def _guarded_native_state() -> dict:
    from anthropocene_rl.autonomous_v7.model import Actor, Normalizer
    actor = Actor(4, region_head="hierarchical")
    normalizer = Normalizer(np.zeros(4, np.float32), np.ones(4, np.float32))
    normalizer.guard = 3.0
    normalizer.dead = {"core": np.asarray([False, True, False, False], dtype=bool)}
    for name, _, width in contracts.SETS:
        normalizer.set_mean[name] = np.zeros(width, np.float32)
        normalizer.set_std[name] = np.ones(width, np.float32)
        normalizer.dead[name] = np.zeros(width, dtype=bool)
    return {"contract": contracts.NATIVE_CONTRACT, "core_width": 4,
            "actor": actor.state_dict(), "normalizer": normalizer.state_dict()}


def test_strict_loader_accepts_exact_guarded_normalizer_state_dict_round_trip():
    from anthropocene_rl.autonomous_v7.evaluate import load_strict_state
    actor, normalizer = load_strict_state(
        _guarded_native_state(), resolved_semantics=SEMANTICS, diagnostic_loading=None,
    )
    assert actor.region_head == "hierarchical"
    assert normalizer.guard == 3.0
    assert normalizer.dead["core"].tolist() == [False, True, False, False]


@pytest.mark.parametrize("attack", [
    "partial", "extra", "dead-dtype", "dead-shape", "guard-negative",
    "guard-nonfinite", "mean-dtype", "mean-nonfinite", "std-zero",
])
def test_strict_loader_rejects_hostile_guarded_normalizer_variants(attack):
    from anthropocene_rl.autonomous_v7.evaluate import load_strict_state
    state = _guarded_native_state()
    normalizer = state["normalizer"]
    if attack == "partial":
        del normalizer["sites_dead"]
    elif attack == "extra":
        normalizer["unknown_dead"] = [False]
    elif attack == "dead-dtype":
        normalizer["core_dead"] = [0, 1, 0, 0]
    elif attack == "dead-shape":
        normalizer["core_dead"] = [False]
    elif attack == "guard-negative":
        normalizer["guard"] = -1.0
    elif attack == "guard-nonfinite":
        normalizer["guard"] = float("nan")
    elif attack == "mean-dtype":
        normalizer["core_mean"][0] = 0
    elif attack == "mean-nonfinite":
        normalizer["core_mean"][0] = float("inf")
    else:
        normalizer["core_std"][0] = 0.0
    with pytest.raises(ValueError, match="normalizer"):
        load_strict_state(state, resolved_semantics=SEMANTICS, diagnostic_loading=None)


def _replace_node_checkpoint(document: dict, tmp_path: Path, node_id: str,
                             state: dict) -> None:
    node = next(row for row in document["nodes"] if row["id"] == node_id)
    checkpoint = tmp_path / node["artifact"]["path"]
    original = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["actor"] = original["actor"]
    for key in ("run_manifest", "experts", "token", "commitments", "learned_movement",
                "region_movement", "learned_fission", "learned_trade"):
        if key in original and key not in state:
            state[key] = original[key]
    torch.save(state, checkpoint)
    node["artifact"]["sha256"] = checkpoint_hash(checkpoint)
    node["raw_schema"], _, _ = gate1.inspect_checkpoint(node["artifact"], tmp_path)
    defaults_path = tmp_path / node["loading"]["default_resolution"]["path"]
    defaults = json.loads(defaults_path.read_text())
    defaults["checkpoint_sha256"] = node["artifact"]["sha256"]
    defaults["digest"] = digest({key: value for key, value in defaults.items() if key != "digest"})
    defaults_path.write_text(json.dumps(defaults, sort_keys=True))
    node["loading"]["default_resolution"]["sha256"] = checkpoint_hash(defaults_path)
    for edge in document["edges"]:
        equivalence = edge.get("equivalence")
        if not isinstance(equivalence, dict):
            continue
        if edge["source"] == node_id:
            equivalence["source_schema"] = copy.deepcopy(node["raw_schema"])
        if edge["destination"] == node_id:
            equivalence["destination_schema"] = copy.deepcopy(node["raw_schema"])
            equivalence["normalizer_mapping"] = {"kind": "explicit-guarded-normalizer"}
    refresh_gate1_protocols(document, tmp_path)
    reseal(document)


def test_guarded_normalizer_is_bound_through_import_packet_and_strict_cas(
        tmp_path, monkeypatch):
    document, worker = fixture(tmp_path)
    _replace_node_checkpoint(document, tmp_path, "clone", _guarded_native_state())
    gate1.validate_manifest(document, tmp_path)
    report = manual_report(document, "clone", "historical", worker, mandatory_values())
    assert gate1._evaluation_index(document, [report], tmp_path)
    clean_tree = gate1.implementation_source_tree()
    clean_tree["dirty_state"] = {"clean": True, "scope": "entire-repository", "porcelain": [],
                                 "porcelain_sha256": digest([])}
    monkeypatch.setattr(gate1, "implementation_source_tree", lambda: copy.deepcopy(clean_tree))
    inputs = complete_packet_inputs(document, worker, tmp_path, "guarded-normalizer")
    packet = gate1.build_review_packet(
        document, inputs[0], tmp_path, cli=["packet"], verification=inputs[1],
        implementation=inputs[2], manifest_ref=inputs[3],
    )
    assert packet["decision"] == "PASS"
    directory = gate1.write_review_packet(packet, tmp_path / "guarded-packet")
    assert gate1.reconstruct_review_packet(directory) == packet
    clone = next(node for node in document["nodes"] if node["id"] == "clone")
    checkpoint_blob = gate1._native_path(directory / "cas" / clone["artifact"]["sha256"])
    checkpoint_blob.write_bytes(checkpoint_blob.read_bytes() + b"corrupt")
    with pytest.raises(ProvenanceError, match="content hash mismatch"):
        gate1.reconstruct_review_packet(directory)


def test_guarded_normalizer_partial_state_rejects_manifest_and_packet(tmp_path):
    document, worker = fixture(tmp_path)
    state = _guarded_native_state()
    del state["normalizer"]["sites_dead"]
    _replace_node_checkpoint(document, tmp_path, "clone", state)
    with pytest.raises(ProvenanceError, match="normalizer keys"):
        gate1.validate_manifest(document, tmp_path)
    inputs = complete_packet_inputs(document, worker, tmp_path, "partial-guard")
    with pytest.raises(ProvenanceError, match="normalizer keys"):
        gate1.build_review_packet(
            document, inputs[0], tmp_path, cli=["packet"], verification=inputs[1],
            implementation=inputs[2], manifest_ref=inputs[3],
        )


@pytest.mark.parametrize("attack", [
    "absent", "raw", "stride", "emitted", "mode", "bool-float",
    "large-float-positive", "large-float-negative", "string", "null", "ragged",
])
def test_import_and_complete_packet_reject_forged_decoder_receipt(tmp_path, attack):
    document, worker = fixture(tmp_path)
    inputs = complete_packet_inputs(document, worker, tmp_path, f"decoder-{attack}")
    report_path = tmp_path / inputs[0][0]["path"]
    report = json.loads(report_path.read_text())
    action_row = report["rows"][0]["trajectory_evidence"]["steps"][0]["actions"][0]
    receipt = action_row["decoder_receipt"]
    if attack == "absent":
        action_row.pop("decoder_receipt")
    elif attack == "raw":
        receipt["raw"]["mix_logits"]["values"][0] += 1.0
        receipt["digest"] = digest({key: value for key, value in receipt.items()
                                    if key != "digest"})
    elif attack == "stride":
        receipt["raw"]["mix_logits"]["stride"] = [0, 1]
        receipt["digest"] = digest({key: value for key, value in receipt.items()
                                    if key != "digest"})
    elif attack == "emitted":
        receipt["emitted_action"]["service"][0] = 0.75
        receipt["digest"] = digest({key: value for key, value in receipt.items()
                                    if key != "digest"})
    elif attack == "mode":
        receipt["mode"] = "seeded-stochastic"
        receipt["digest"] = digest({key: value for key, value in receipt.items()
                                    if key != "digest"})
    else:
        values = receipt["raw"]["strategic_mean"]["values"]
        if attack == "bool-float":
            values[0] = False
        elif attack == "large-float-positive":
            values[0] = 2**60 + 1
        elif attack == "large-float-negative":
            values[0] = -(2**60 + 1)
        elif attack == "string":
            values[0] = "0"
        elif attack == "null":
            values[0] = None
        else:
            values[0] = [0.0]
        receipt["digest"] = digest({key: value for key, value in receipt.items()
                                    if key != "digest"})
    reseal_trajectory_report(report)
    report_path.write_text(json.dumps(report, sort_keys=True))
    inputs[0][0]["sha256"] = checkpoint_hash(report_path)
    with pytest.raises(ProvenanceError, match="decoder|action request|tensor"):
        gate1.build_review_packet(
            document, inputs[0], tmp_path, cli=["packet"], verification=inputs[1],
            implementation=inputs[2], manifest_ref=inputs[3],
        )


def test_stochastic_import_and_complete_packet_reject_bool_in_int_tensor(tmp_path):
    document, worker = fixture(tmp_path)
    pane = document["evaluation"]["panes"][0]
    pane["action_mode"] = "seeded-stochastic"
    world_id = gate1._world_id(pane["worlds"][0])
    nodes = []
    from anthropocene_rl.autonomous_v7.gate1_diagnostics import (
        draw_order_from_semantics, generator_state_sha256,
    )
    for node_id in pane["nodes"]:
        steps = []
        for index in range(pane["worlds"][0]["horizon"]):
            seed = 1100 + index
            generator = torch.Generator(device="cpu").manual_seed(seed)
            steps.append({"step": index, "seed": seed,
                          "state_sha256": generator_state_sha256(generator)})
        nodes.append({"node_id": node_id, "draw_order": draw_order_from_semantics(SEMANTICS),
                      "steps": steps})
    pane["policy_rng_schedule"] = write(
        tmp_path / "hostile-int-rng.json",
        {"schema": 1, "kind": "gate1_policy_rng_schedule",
         "algorithm": "torch.Generator.cpu-default", "torch_version": torch.__version__,
         "device": "cpu", "worlds": [{"world_id": world_id, "nodes": nodes}]},
        seal_digest=True,
    )
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    inputs = complete_packet_inputs(document, worker, tmp_path, "decoder-bool-int")
    report_path = tmp_path / inputs[0][0]["path"]
    report = json.loads(report_path.read_text())
    receipt = report["rows"][0]["trajectory_evidence"]["steps"][0]["actions"][0]["decoder_receipt"]
    receipt["raw"]["region"]["values"] = False
    receipt["digest"] = digest({key: value for key, value in receipt.items()
                                if key != "digest"})
    reseal_trajectory_report(report)
    report_path.write_text(json.dumps(report, sort_keys=True))
    inputs[0][0]["sha256"] = checkpoint_hash(report_path)
    with pytest.raises(ProvenanceError, match="canonical int64"):
        gate1.build_review_packet(
            document, inputs[0], tmp_path, cli=["packet"], verification=inputs[1],
            implementation=inputs[2], manifest_ref=inputs[3],
        )


def test_decoder_receipts_are_preserved_and_replayed_from_strict_packet_cas(
        tmp_path, monkeypatch):
    document, worker = fixture(tmp_path)
    clean_tree = gate1.implementation_source_tree()
    clean_tree["dirty_state"] = {"clean": True, "scope": "entire-repository", "porcelain": [],
                                 "porcelain_sha256": digest([])}
    monkeypatch.setattr(gate1, "implementation_source_tree", lambda: copy.deepcopy(clean_tree))
    inputs = complete_packet_inputs(document, worker, tmp_path, "decoder-cas")
    packet = gate1.build_review_packet(
        document, inputs[0], tmp_path, cli=["packet"], verification=inputs[1],
        implementation=inputs[2], manifest_ref=inputs[3],
    )
    directory = gate1.write_review_packet(packet, tmp_path / "decoder-cas-packet")
    rebuilt = gate1.reconstruct_review_packet(directory)
    assert rebuilt == packet and rebuilt["decision"] == "PASS"
    report_ref = packet["inputs"]["evaluation_reports"][0]["ref"]
    report_blob = gate1._native_path(directory / "cas" / report_ref["sha256"])
    preserved = json.loads(report_blob.read_text())
    receipt = preserved["rows"][0]["trajectory_evidence"]["steps"][0]["actions"][0]["decoder_receipt"]
    assert receipt["digest"] == digest({key: value for key, value in receipt.items()
                                        if key != "digest"})
    receipt["raw"]["strategic_mean"]["values"][0] = False
    receipt["digest"] = digest({key: value for key, value in receipt.items()
                                if key != "digest"})
    reseal_trajectory_report(preserved)
    report_blob.write_text(json.dumps(preserved, sort_keys=True))
    with pytest.raises(ProvenanceError, match="content hash mismatch"):
        gate1.reconstruct_review_packet(directory)


@pytest.mark.parametrize("missing", ["encoder.film_tokens.0.weight", "region_group.weight"])
def test_strict_loader_rejects_tensors_compatibility_hooks_would_synthesize(tmp_path, missing):
    from anthropocene_rl.autonomous_v7 import evaluate
    from anthropocene_rl.autonomous_v7.model import Actor, Normalizer
    actor = Actor(4, token=True, region_head="hierarchical")
    actor_state = actor.state_dict()
    assert missing in actor_state
    del actor_state[missing]
    normalizer = Normalizer(np.zeros(4, np.float32), np.ones(4, np.float32))
    for name, _, width in contracts.SETS:
        normalizer.set_mean[name] = np.zeros(width, np.float32)
        normalizer.set_std[name] = np.ones(width, np.float32)
    path = tmp_path / (missing.replace(".", "-") + ".pt")
    torch.save({"contract": contracts.NATIVE_CONTRACT, "core_width": 4, "token": True,
                "actor": actor_state,
                "normalizer": normalizer.state_dict()}, path)
    with pytest.raises(ValueError, match="exact raw tensor keys"):
        evaluate.load(path, strict=True, resolved_semantics=dict(SEMANTICS, token=True))


def test_tensor_resolver_supports_real_dotted_actor_state_dict_keys_and_rejects_ambiguity():
    from anthropocene_rl.autonomous_v7.model import Actor
    actor_state = Actor(4, region_head="hierarchical").state_dict()
    key = next(name for name in actor_state if name.startswith("encoder.") and name.endswith(".weight"))
    assert gate1._tensor({"actor": actor_state}, "actor." + key) is actor_state[key]
    value = torch.tensor([1.0])
    ambiguous = {"actor": {"encoder.weight": value, "encoder": {"weight": value}}}
    with pytest.raises(ProvenanceError, match="ambiguous"):
        gate1._tensor(ambiguous, "actor.encoder.weight")


@pytest.mark.parametrize("value", [
    {"food": None},
    {},
    {"food": {"value": 1.0, "availability_reason": None},
     "extra": {"value": 1.0, "availability_reason": None}},
])
def test_service_measurements_require_exact_registered_value_missingness_shape(value):
    schema = {"fields": [{"name": "food", "unit": "service-units"}]}
    assert gate1._metric("service_outcomes", value, schema) is False


def test_count_metrics_and_assertion_only_equivalence_are_fail_closed(tmp_path):
    schema = {"fields": [{"name": "food", "unit": "service-units"}]}
    assert gate1._metric("compiler_interventions", -1, schema) is False
    assert gate1._metric("compiler_interventions", 1.5, schema) is False
    document, _ = fixture(tmp_path)
    edge = document["edges"][0]
    claim = edge["equivalence"]
    claim["kind"] = "mapped_distribution_numerical_equivalence"
    claim["mapped_outputs"] = ["actor"]
    claim["mapped_tensors"] = []
    evidence = {"schema": 1, "kind": "equivalence_result",
                "claim_sha256": gate1._claim_digest(claim),
                "corpus_sha256": claim["corpus"]["sha256"],
                "checks": [{"scope": "actor", "count": 1,
                            "max_absolute_error": -1.0, "max_relative_error": 0.0}]}
    claim["result"] = write(tmp_path / "equivalence.json", evidence, seal_digest=True)
    refresh_gate1_protocols(document, tmp_path)
    with pytest.raises(ProvenanceError, match="noncanonical field set"):
        gate1.validate_manifest(reseal(document), tmp_path)


def test_required_trained_node_outside_reviewed_panes_is_rejected(tmp_path):
    document, _ = fixture(tmp_path)
    document["evaluation"]["panes"][0]["nodes"].remove(trained_ids(document)[-1])
    with pytest.raises(ProvenanceError, match="nodes differ from reviewed lineage applicability"):
        gate1.validate_manifest(reseal(document), tmp_path)


def test_reversed_comparison_order_is_rejected(tmp_path):
    document, _ = fixture(tmp_path)
    document["comparison_order"] = list(reversed(document["comparison_order"]))
    with pytest.raises(ProvenanceError, match="reviewed continuous lineage sequences"):
        gate1.validate_manifest(reseal(document), tmp_path)


def test_reviewed_protocol_cannot_reverse_dag_ancestry(tmp_path):
    document, _ = fixture(tmp_path)

    def reverse(body):
        body["sequences"] = list(reversed(body["sequences"]))
        body["comparison_order"] = [edge for sequence in body["sequences"]
                                    for edge in sequence["edges"]]
        document["comparison_order"] = list(body["comparison_order"])

    rewrite_reviewed_protocol(tmp_path, document["lineage_inventory"], reverse)
    with pytest.raises(ProvenanceError, match="reverses DAG ancestry"):
        gate1.validate_manifest(reseal(document), tmp_path)


@pytest.mark.parametrize("value", [
    torch.tensor(0.0), torch.tensor(3, dtype=torch.int64), torch.tensor(True),
])
def test_literal_tensor_bytes_support_scalar_numeric_and_bool(value):
    assert gate1._literal_tensor_equal(value, value.clone())


def test_verification_command_substitution_is_rejected(tmp_path):
    document, _ = fixture(tmp_path)
    verification = verification_ref(tmp_path, document)
    verification_path = tmp_path / verification["path"]
    body = json.loads(verification_path.read_text())
    receipt_ref = body["test_receipts"][0]
    receipt_path = tmp_path / receipt_ref["path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["command"] = ["python", "-c", "pass"]
    receipt["digest"] = digest({key: value for key, value in receipt.items() if key != "digest"})
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    receipt_ref["sha256"] = checkpoint_hash(receipt_path)
    body["digest"] = digest({key: value for key, value in body.items() if key != "digest"})
    verification_path.write_text(json.dumps(body, sort_keys=True))
    verification["sha256"] = checkpoint_hash(verification_path)
    with pytest.raises(ProvenanceError, match="substitutes/omits a prescribed test command"):
        gate1.build_review_packet(document, [], tmp_path, cli=["packet"], verification=verification,
                                  implementation=implementation_ref(tmp_path))


def test_independent_review_cannot_be_reused_after_protocol_mutation(tmp_path):
    document, _ = fixture(tmp_path)
    protocol_ref = document["lineage_inventory"]
    path = tmp_path / protocol_ref["path"]
    body = json.loads(path.read_text())
    body["slots"][0]["reason"] = "forged after approval"
    body["digest"] = digest({key: value for key, value in body.items() if key != "digest"})
    path.write_text(json.dumps(body, sort_keys=True))
    protocol_ref["sha256"] = checkpoint_hash(path)
    with pytest.raises(ProvenanceError, match="review is unbound"):
        gate1.validate_manifest(reseal(document), tmp_path)


@pytest.mark.parametrize("mutation", ["nonexistent", "duplicate", "omit-trained", "relabel-paired"])
def test_lineage_slots_bind_exact_reviewed_nodes_and_historical_t3_runs(tmp_path, mutation):
    document, _ = fixture(tmp_path)

    def mutate(body):
        slot = next(row for row in body["slots"] if row["slot"] == "trained-t3")
        if mutation == "nonexistent":
            slot["node_ids"][0] = "not-a-node"
        elif mutation == "duplicate":
            slot["node_ids"][1] = slot["node_ids"][0]
        elif mutation == "omit-trained":
            slot["node_ids"].pop()
        else:
            node = next(row for row in body["nodes"] if row["id"] == slot["node_ids"][0])
            node["run_identity"]["pairing_status"] = "matched"
            node["run_identity"]["pairing_group"] = "optimizer-seed-70"

    rewrite_reviewed_protocol(tmp_path, document["lineage_inventory"], mutate)
    reseal(document)
    with pytest.raises(ProvenanceError):
        gate1.validate_manifest(document, tmp_path)


def test_missing_historical_conversion_is_explicit_and_blocks(tmp_path):
    document, _ = fixture(tmp_path)
    edge = document["edges"][0]
    edge.clear()
    edge.update({"id": "prior-clone", "status": "missing", "source": "prior",
                 "destination": "clone", "relation": "schema_conversion",
                 "reason": "historical conversion receipt not recovered"})
    support = write(tmp_path / "conversion-missing.json", {"historical": "conversion expected"})
    rewrite_reviewed_protocol(tmp_path, document["lineage_inventory"], lambda body: (
        next(row for row in body["edges"] if row["id"] == "prior-clone").update(
            status="missing", relation="schema_conversion", historical_panes=[]),
        next(row for row in body["slots"] if row["slot"] == "conversion").update(
            status="missing", edge_ids=["prior-clone"], reason="historical artifact missing",
            historical_support=support),
    ))
    refresh_gate1_protocols(document, tmp_path)
    result = gate1.validate_manifest(reseal(document), tmp_path)
    assert "edge:prior-clone:missing" in result["holds"]


def test_packet_reconstructs_from_cas_and_detects_derived_tampering(tmp_path, monkeypatch):
    document, worker = fixture(tmp_path)
    equivalence_evidence = install_raw_equivalence(
        document, tmp_path, absolute_paths=True,
    )
    pane = document["evaluation"]["panes"][0]
    pane["metrics"] = sorted(gate1.MANDATORY_METRICS)
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    reports = [manual_report(document, node, "historical", worker, mandatory_values())
               for node in pane["nodes"]]
    evaluation_refs = [write(tmp_path / f"evaluation-{report['node']}.json", report)
                       for report in reports]
    manifest_ref = write(tmp_path / "manifest.json", document)
    manifest_ref["path"] = str((tmp_path / manifest_ref["path"]).resolve())
    clean_tree = gate1.implementation_source_tree()
    clean_tree["dirty_state"] = {"clean": True, "scope": "entire-repository", "porcelain": [],
                                 "porcelain_sha256": digest([])}
    monkeypatch.setattr(gate1, "implementation_source_tree", lambda: copy.deepcopy(clean_tree))
    implementation = implementation_ref(tmp_path)
    verification = verification_ref(tmp_path, document)
    packet = gate1.build_review_packet(
        document, evaluation_refs, tmp_path, cli=["packet"], verification=verification,
        implementation=implementation, manifest_ref=manifest_ref,
    )
    assert packet["decision"] == "PASS"
    assert implementation["sha256"] in packet["artifact_hashes"]
    assert all(ref["sha256"] in packet["artifact_hashes"] for ref in evaluation_refs)
    assert equivalence_evidence["result"]["sha256"] in packet["artifact_hashes"]
    assert all(ref["sha256"] in packet["artifact_hashes"]
               for ref in equivalence_evidence["executions"])
    assert equivalence_evidence["output_contracts"]["sha256"] in packet["artifact_hashes"]
    first_execution = json.loads(
        (tmp_path / equivalence_evidence["executions"][0]["path"]).read_text(),
    )
    assert Path(manifest_ref["path"]).is_absolute()
    assert Path(equivalence_evidence["protocol"]["path"]).is_absolute()
    assert Path(equivalence_evidence["output_contracts"]["path"]).is_absolute()
    assert Path(first_execution["output"]["path"]).is_absolute()
    assert first_execution["output"]["sha256"] in packet["artifact_hashes"]
    directory = gate1.write_review_packet(packet, tmp_path / "packets")
    completion = json.loads((directory / "completion.json").read_text())
    assert completion["artifact_hashes"] == packet["artifact_hashes"]
    assert not (directory / evaluation_refs[0]["path"]).exists()
    assert gate1._native_path(
        directory / "cas" / first_execution["output"]["sha256"],
    ).is_file()
    assert gate1._native_path(
        directory / "cas" / equivalence_evidence["output_contracts"]["sha256"],
    ).is_file()
    monkeypatch.setattr(gate1, "implementation_source_tree",
                        lambda: (_ for _ in ()).throw(AssertionError("external tree consulted")))
    rebuilt = gate1.reconstruct_review_packet(directory)
    assert rebuilt == packet

    def rejects_blob_change(sha, replacement):
        cas_path = gate1._native_path(directory / "cas" / sha)
        preserved = cas_path.read_bytes()
        cas_path.write_bytes(replacement(preserved))
        with pytest.raises(ProvenanceError, match="CAS artifact content hash mismatch"):
            gate1.reconstruct_review_packet(directory)
        cas_path.write_bytes(preserved)

    output_cas = gate1._native_path(directory / "cas" / first_execution["output"]["sha256"])
    preserved_output = output_cas.read_bytes()
    attacked_output = json.loads(preserved_output)
    attacked_output["values"] = [2**60 + 1, 2**60]
    rejects_blob_change(
        first_execution["output"]["sha256"],
        lambda _: json.dumps(attacked_output, sort_keys=True).encode(),
    )
    rejects_blob_change(
        equivalence_evidence["output_contracts"]["sha256"],
        lambda value: value + b"\ncorrupt output contract",
    )
    rejects_blob_change(
        equivalence_evidence["protocol"]["sha256"],
        lambda value: value + b"\ncorrupt execution protocol",
    )
    rejects_blob_change(
        manifest_ref["sha256"], lambda value: value + b"\ncorrupt manifest",
    )
    attacked_report = copy.deepcopy(reports[0])
    attacked_report["rows"][0]["metrics"]["welfare"] = 999999
    attacked_report["digest"] = digest(
        {key: value for key, value in attacked_report.items() if key != "digest"},
    )
    rejects_blob_change(
        evaluation_refs[0]["sha256"],
        lambda _: json.dumps(attacked_report, sort_keys=True).encode(),
    )
    for attack in ("wrong-reset-trajectory-id", "broken-opening-chain", "invented-founder",
                   "illegal-clothing-route", "missing-units", "fabricated-trade-boundary",
                   "deterministic-unsnapped-levels", "invented-migration-founder",
                   "migration-copies-deleted", "migration-kind-renamed",
                   "migration-population-inflation"):
        attacked_report = copy.deepcopy(reports[0])
        apply_trajectory_attack(attacked_report, attack)
        rejects_blob_change(
            evaluation_refs[0]["sha256"],
            lambda _, value=json.dumps(attacked_report, sort_keys=True).encode(): value,
        )
    protocol_cas = gate1._native_path(
        directory / "cas" / equivalence_evidence["protocol"]["sha256"],
    )
    rejects_blob_change(
        first_execution["output"]["sha256"], lambda _: protocol_cas.read_bytes(),
    )

    output_cas.unlink()
    with pytest.raises(ProvenanceError, match="CAS inventory differs"):
        gate1.reconstruct_review_packet(directory)
    output_cas.write_bytes(preserved_output)

    output_cas.unlink()
    output_cas.mkdir()
    with pytest.raises(ProvenanceError, match="regular non-reparse file"):
        gate1.reconstruct_review_packet(directory)
    output_cas.rmdir()
    output_cas.write_bytes(preserved_output)

    extra_cas = gate1._native_path(directory / "cas" / ("f" * 64))
    extra_cas.write_bytes(b"inconsistent unregistered CAS blob")
    with pytest.raises(ProvenanceError, match="CAS inventory differs"):
        gate1.reconstruct_review_packet(directory)
    extra_cas.unlink()

    report_path = directory / "report.json"
    stored = json.loads(report_path.read_text())
    stored["decision"] = "HOLD"
    report_path.write_text(json.dumps(stored, sort_keys=True))
    completion_path = directory / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["report_sha256"] = checkpoint_hash(report_path)
    completion["digest"] = digest({key: value for key, value in completion.items() if key != "digest"})
    completion_path.write_text(json.dumps(completion, sort_keys=True))
    with pytest.raises(ProvenanceError, match="derived outputs do not reconstruct"):
        gate1.reconstruct_review_packet(directory)


def test_forged_comparator_output_is_recomputed_as_unequal(tmp_path):
    document, source_worker = fixture(tmp_path)
    live_body = {key: copy.deepcopy(value) for key, value in source_worker.items() if key != "digest"}
    live_body["bridge_sha256"] = "5" * 64
    live_worker = seal(live_body)
    protocol_ref, receipt_ref = comparator_refs(tmp_path, document, source_worker, live_worker)
    receipt_path = tmp_path / receipt_ref["path"]
    receipt = json.loads(receipt_path.read_text())
    execution_ref = next(
        ref for ref in receipt["executions"]
        if "comparator-live-reward" in ref["path"]
    )
    execution_path = tmp_path / execution_ref["path"]
    execution = json.loads(execution_path.read_text())
    execution["output"] = write(tmp_path / "forged-live-reward.json", {"different": True})
    execution["digest"] = digest({key: value for key, value in execution.items() if key != "digest"})
    execution_path.write_text(json.dumps(execution, sort_keys=True))
    execution_ref["sha256"] = checkpoint_hash(execution_path)
    receipt["digest"] = digest({key: value for key, value in receipt.items() if key != "digest"})
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    receipt_ref["sha256"] = checkpoint_hash(receipt_path)
    with pytest.raises(ProvenanceError, match="unequal reward output"):
        gate1._semantic_comparator(
            protocol_ref, receipt_ref, tmp_path, source_worker, live_worker,
            document["nodes"][0]["inference_implementation"],
            document["evaluation"]["panes"][0]["evaluation_implementation"],
        )


def test_coverage_scope_substitution_is_rejected(tmp_path):
    document, _ = fixture(tmp_path)
    verification = verification_ref(tmp_path, document)
    verification_path = tmp_path / verification["path"]
    evidence = json.loads(verification_path.read_text())
    coverage_ref = evidence["coverage"]
    coverage_path = tmp_path / coverage_ref["path"]
    coverage = json.loads(coverage_path.read_text())
    coverage["scope"] = ["src/irrelevant.py"]
    coverage["digest"] = digest({key: value for key, value in coverage.items() if key != "digest"})
    coverage_path.write_text(json.dumps(coverage, sort_keys=True))
    coverage_ref["sha256"] = checkpoint_hash(coverage_path)
    evidence["digest"] = digest({key: value for key, value in evidence.items() if key != "digest"})
    verification_path.write_text(json.dumps(evidence, sort_keys=True))
    verification["sha256"] = checkpoint_hash(verification_path)
    with pytest.raises(ProvenanceError, match="substitutes scope/tool/denominator"):
        gate1.build_review_packet(document, [], tmp_path, cli=["packet"], verification=verification,
                                  implementation=implementation_ref(tmp_path))


def test_unapproved_reviewer_identity_cannot_forge_protocol_approval(tmp_path):
    document, _ = fixture(tmp_path)
    protocol_ref = document["lineage_inventory"]
    protocol_path = tmp_path / protocol_ref["path"]
    protocol = json.loads(protocol_path.read_text())
    review_ref = protocol["review"]
    review_path = tmp_path / review_ref["path"]
    review = json.loads(review_path.read_text())
    review["reviewer_identity"] = write(tmp_path / "forged-reviewer.json", {
        "schema": 1, "kind": "independent_reviewer_identity", "provider": "self",
        "model": "self-approved", "version": "today",
    }, seal_digest=True)
    review["digest"] = digest({key: value for key, value in review.items() if key != "digest"})
    review_path.write_text(json.dumps(review, sort_keys=True))
    review_ref["sha256"] = checkpoint_hash(review_path)
    protocol["digest"] = digest({key: value for key, value in protocol.items() if key != "digest"})
    protocol_path.write_text(json.dumps(protocol, sort_keys=True))
    protocol_ref["sha256"] = checkpoint_hash(protocol_path)
    with pytest.raises(ProvenanceError, match="reviewer identity is malformed"):
        gate1.validate_manifest(reseal(document), tmp_path)


def test_single_paired_world_interval_is_explicitly_unavailable():
    assert gate1._paired_interval([1.0], {"edge": "example"}) == {
        "status": "unavailable", "lower": None, "upper": None,
    }


def test_public_packet_builder_rejects_source_tree_injection(tmp_path):
    document, _ = fixture(tmp_path)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        gate1.build_review_packet(
            document, [], tmp_path, cli=["packet"], verification=verification_ref(tmp_path, document),
            implementation=implementation_ref(tmp_path),
            _observed_source_tree={"commit": "0" * 40, "source_files": [],
                                   "dirty_state": {"clean": True}},
        )


def test_raw_equivalence_recomputes_complete_case_scope_side_evidence(tmp_path):
    document, _ = fixture(tmp_path)
    install_raw_equivalence(document, tmp_path)
    result = gate1.validate_manifest(document, tmp_path)
    comparison = result["equivalence"][0]
    assert comparison["status"] == "PASS"
    assert comparison["checks"] == [{
        "scope": "actor-output", "case_count": 2, "value_count": 4,
        "max_absolute_error": 0.0, "max_relative_error": 0.0,
        "masks_equal": True, "decoded_actions_equal": True,
    }]


@pytest.mark.parametrize("mutation", ["omitted-case", "omitted-scope", "omitted-side", "duplicate"])
def test_raw_equivalence_rejects_incomplete_or_duplicate_cross_product(tmp_path, mutation):
    document, _ = fixture(tmp_path)
    evidence = install_raw_equivalence(
        document, tmp_path, scopes=["actor-output", "value-output"],
    )

    def mutate(result):
        refs = result["executions"]
        rows = [(ref, json.loads((tmp_path / ref["path"]).read_text())) for ref in refs]
        if mutation == "omitted-case":
            case = rows[0][1]["case_id"]
            result["executions"] = [ref for ref, row in rows if row["case_id"] != case]
        elif mutation == "omitted-scope":
            result["executions"] = [ref for ref, row in rows if row["scope"] != "value-output"]
        elif mutation == "omitted-side":
            result["executions"] = refs[1:]
        else:
            result["executions"].append(copy.deepcopy(refs[0]))

    rewrite_sealed_ref(tmp_path, evidence["result"], mutate)
    with pytest.raises(ProvenanceError, match="duplicated|omits a case/scope/side"):
        gate1.validate_manifest(reseal(document), tmp_path)


@pytest.mark.parametrize("mutation", ["numeric", "mask", "action"])
def test_raw_equivalence_recomputes_values_masks_and_actions(tmp_path, mutation):
    document, _ = fixture(tmp_path)
    evidence = install_raw_equivalence(document, tmp_path)

    def mutate(output):
        if mutation == "numeric":
            output["values"][1] = 3.0
        elif mutation == "mask":
            output["mask"]["values"] = [False, True]
        else:
            output["decoded_action"] = {"index": 1}

    mutate_raw_execution(evidence, tmp_path, 1, output_mutation=mutate)
    result = gate1.validate_manifest(reseal(document), tmp_path)
    assert result["equivalence"][0]["status"] == "FAIL"


@pytest.mark.parametrize("mutation", ["nonfinite", "malformed-shape", "overflow"])
def test_raw_equivalence_rejects_nonfinite_or_malformed_values(tmp_path, mutation):
    document, _ = fixture(tmp_path)
    evidence = install_raw_equivalence(document, tmp_path)
    if mutation == "nonfinite":
        mutate_raw_execution(evidence, tmp_path, 0, nonfinite_output=True)
    elif mutation == "malformed-shape":
        mutate_raw_execution(
            evidence, tmp_path, 0,
            output_mutation=lambda output: output.update(shape=[3]),
        )
    else:
        mutate_raw_execution(
            evidence, tmp_path, 0,
            output_mutation=lambda output: output.update(values=[1e308, 2.0]),
        )
        mutate_raw_execution(
            evidence, tmp_path, 1,
            output_mutation=lambda output: output.update(values=[-1e308, 2.0]),
        )
    with pytest.raises((ProvenanceError, ValueError), match="non-finite|nonfinite|shape|range"):
        gate1.validate_manifest(reseal(document), tmp_path)


@pytest.mark.parametrize("mutation", ["producer-implementation", "runtime"])
def test_raw_equivalence_rejects_execution_implementation_or_runtime_substitution(tmp_path, mutation):
    document, _ = fixture(tmp_path)
    evidence = install_raw_equivalence(document, tmp_path)

    def alter(execution):
        if mutation == "producer-implementation":
            execution["producer_implementation_sha256"] = "0" * 64
        else:
            execution["numeric_runtime"]["version"] = "forged-runtime"

    mutate_raw_execution(evidence, tmp_path, 0, execution_mutation=alter)
    with pytest.raises(ProvenanceError, match="malformed or unbound"):
        gate1.validate_manifest(reseal(document), tmp_path)


def test_raw_equivalence_rejects_forged_result_summary(tmp_path):
    document, _ = fixture(tmp_path)
    evidence = install_raw_equivalence(document, tmp_path)
    rewrite_sealed_ref(
        tmp_path, evidence["result"],
        lambda result: result.update(checks=[{"count": 2, "max_absolute_error": 0.0}]),
    )
    with pytest.raises(ProvenanceError, match="assertion-only or unbound"):
        gate1.validate_manifest(reseal(document), tmp_path)


def test_stochastic_equivalence_requires_exact_rng_case_schedule(tmp_path):
    document, _ = fixture(tmp_path)
    install_raw_equivalence(
        document, tmp_path, kind="stochastic_behavioral_equivalence", omit_rng_case=True,
    )
    with pytest.raises(ProvenanceError, match="RNG schedule omits/substitutes"):
        gate1.validate_manifest(document, tmp_path)


def test_reviewed_inventory_prevents_removing_failed_expected_equivalence(tmp_path):
    document, _ = fixture(tmp_path)
    clone = document["nodes"][1]
    checkpoint = tmp_path / clone["artifact"]["path"]
    state = torch.load(checkpoint, weights_only=True)
    state["actor"]["strategic.bias"] = torch.full_like(
        state["actor"]["strategic.bias"], 2.0,
    )
    torch.save(state, checkpoint)
    clone["artifact"]["sha256"] = checkpoint_hash(checkpoint)
    defaults_ref = clone["loading"]["default_resolution"]
    rewrite_sealed_ref(
        tmp_path, defaults_ref,
        lambda defaults: defaults.update(checkpoint_sha256=clone["artifact"]["sha256"]),
    )
    document["edges"][0].pop("equivalence")
    reseal(document)
    with pytest.raises(ProvenanceError, match="equivalence differs from reviewed"):
        gate1.validate_manifest(document, tmp_path)
    with pytest.raises(ProvenanceError, match="equivalence differs from reviewed"):
        gate1.build_review_packet(
            document, [], tmp_path, cli=["packet"], verification=None, implementation={},
        )


@pytest.mark.parametrize("mutation", [
    "seed", "horizon", "checkpoint", "claim", "corpus", "pane",
    "producer-evidence", "run-evidence", "source-artifacts", "producing-learner",
    "migration-record",
])
def test_old_evaluation_review_rejects_preregistered_surface_substitution(tmp_path, mutation):
    document, _ = fixture(tmp_path)
    if mutation == "seed":
        document["evaluation"]["panes"][0]["worlds"][0]["seed"] = 4500
    elif mutation == "horizon":
        document["evaluation"]["panes"][0]["worlds"][0]["horizon"] = 100
    elif mutation == "checkpoint":
        node = document["nodes"][0]
        checkpoint = tmp_path / node["artifact"]["path"]
        state = torch.load(checkpoint, weights_only=True)
        state["actor"]["strategic.bias"] = torch.full_like(
            state["actor"]["strategic.bias"], 9.0,
        )
        torch.save(state, checkpoint)
        node["artifact"]["sha256"] = checkpoint_hash(checkpoint)
        rewrite_sealed_ref(
            tmp_path, node["loading"]["default_resolution"],
            lambda defaults: defaults.update(checkpoint_sha256=node["artifact"]["sha256"]),
        )
    elif mutation == "claim":
        document["edges"][0]["equivalence"]["tolerances"]["absolute"] = 1.0
    elif mutation == "corpus":
        document["edges"][0]["equivalence"]["corpus"] = write(
            tmp_path / "substitute-corpus.json", {
                "schema": 1, "kind": "equivalence_corpus", "cases": [],
            }, seal_digest=True,
        )
    elif mutation == "pane":
        document["evaluation"]["panes"][0]["id"] = "substitute-pane"
    elif mutation == "producer-evidence":
        document["nodes"][0]["producer_evidence"] = write(tmp_path / "producer-substitute.json", {"x": 1})
    elif mutation == "run-evidence":
        document["nodes"][0]["run_evidence"] = write(tmp_path / "run-substitute.json", {"x": 1})
    elif mutation == "source-artifacts":
        document["nodes"][0]["source_artifacts"].append(write(tmp_path / "source-substitute.json", {"x": 1}))
    elif mutation == "producing-learner":
        document["nodes"][0]["producing_learner"]["identity"] = "substitute learner"
    else:
        document["edges"][0]["migration_record"] = write(tmp_path / "migration-substitute.json", {"x": 1})
    with pytest.raises(ProvenanceError):
        gate1.validate_manifest(reseal(document), tmp_path)


@pytest.mark.parametrize("attack", [
    "equal-overflow", "dtype", "truncation", "wrong-mask", "bad-action",
    "component-reorder", "precision", "schema-ref-swap", "output-ref-swap",
])
def test_reviewed_output_contract_rejects_post_result_substitution(tmp_path, attack):
    document, _ = fixture(tmp_path)
    evidence = install_raw_equivalence(document, tmp_path)
    if attack == "output-ref-swap":
        mutate_raw_execution(
            evidence, tmp_path, 1,
            output_mutation=lambda output: output.update(values=[1.0, 3.0]),
        )
        assert gate1.validate_manifest(reseal(document), tmp_path)["equivalence"][0]["status"] == "FAIL"
    apply_output_attack(evidence, tmp_path, attack)
    with pytest.raises(ProvenanceError):
        gate1.validate_manifest(reseal(document), tmp_path)


def test_numerical_equivalence_requires_null_behavior_contracts(tmp_path):
    document, _ = fixture(tmp_path)
    install_raw_equivalence(
        document, tmp_path, kind="mapped_distribution_numerical_equivalence",
    )
    result = gate1.validate_manifest(document, tmp_path)
    assert result["equivalence"][0]["status"] == "PASS"
    assert result["equivalence"][0]["checks"][0]["masks_equal"] is None


def test_old_review_rejects_output_contract_mutation(tmp_path):
    document, _ = fixture(tmp_path)
    evidence = install_raw_equivalence(document, tmp_path)
    rewrite_sealed_ref(
        tmp_path, evidence["output_contracts"],
        lambda contracts: contracts["contracts"][0].update(
            components=["renamed-0", "component-1"],
        ),
    )
    rewrite_sealed_ref(
        tmp_path, evidence["protocol"],
        lambda protocol: protocol.update(output_contracts=evidence["output_contracts"]),
    )
    with pytest.raises(ProvenanceError, match="equivalence differs from reviewed|evaluation surface"):
        gate1.validate_manifest(reseal(document), tmp_path)


def test_recursive_decoded_action_contract_is_exact_and_bounded():
    schema = gate1._action_contract({
        "type": "object", "fields": {
            "index": {"type": "integer", "minimum": 0, "maximum": 1},
            "enabled": {"type": "boolean"},
            "mode": {"type": "string", "enum": ["stay", "move"]},
            "weights": {"type": "array", "length": 2,
                        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
        },
    })
    assert gate1._action_matches(
        {"index": 1, "enabled": True, "mode": "move", "weights": [0.25, 0.75]}, schema,
    )
    assert not gate1._action_matches(
        {"index": 99, "enabled": True, "mode": "move", "weights": [0.25, 0.75]}, schema,
    )
    assert not gate1._action_matches(
        {"index": 1, "enabled": True, "mode": "move", "weights": [0.25]}, schema,
    )


@pytest.mark.parametrize("attack", [
    "equal-overflow", "dtype", "truncation", "wrong-mask", "bad-action", "output-ref-swap",
])
def test_full_packet_cannot_pass_post_result_output_attack(tmp_path, attack):
    document, worker = fixture(tmp_path)
    evidence = install_raw_equivalence(document, tmp_path, prefix=f"packet-{attack}")
    pane = document["evaluation"]["panes"][0]
    pane["metrics"] = sorted(gate1.MANDATORY_METRICS)
    refresh_gate1_protocols(document, tmp_path, inventory=False)
    reseal(document)
    reports = [manual_report(document, node, pane["id"], worker, mandatory_values())
               for node in pane["nodes"]]
    evaluation_refs = [write(tmp_path / f"{attack}-evaluation-{report['node']}.json", report)
                       for report in reports]
    verification = verification_ref(tmp_path, document)
    implementation = implementation_ref(tmp_path)
    if attack == "output-ref-swap":
        mutate_raw_execution(
            evidence, tmp_path, 1,
            output_mutation=lambda output: output.update(values=[1.0, 3.0]),
        )
        assert gate1.validate_manifest(reseal(document), tmp_path)["equivalence"][0]["status"] == "FAIL"
    apply_output_attack(evidence, tmp_path, attack)
    manifest_ref = write(tmp_path / f"{attack}-manifest.json", reseal(document))
    with pytest.raises(ProvenanceError):
        gate1.build_review_packet(
            document, evaluation_refs, tmp_path, cli=["packet"], verification=verification,
            implementation=implementation, manifest_ref=manifest_ref,
        )


@pytest.mark.parametrize(("dtype", "value"), [
    ("float32", 2**24 + 1),
    ("float32", -(2**24 + 1)),
    ("float32", 2**60 + 1),
    ("float32", -(2**60 + 1)),
    ("float64", 2**53 + 1),
    ("float64", -(2**53 + 1)),
])
def test_integer_precision_loss_is_rejected_before_numeric_comparison(dtype, value):
    with pytest.raises(ProvenanceError, match="canonically representable"):
        gate1._normalized_numeric(value, dtype)


@pytest.mark.parametrize(("dtype", "value"), [
    ("float32", 0), ("float32", 2**24), ("float32", -(2**24)),
    ("float32", 2**60), ("float32", -(2**60)),
    ("float64", 2**53), ("float64", -(2**53)),
    ("float64", 2**60), ("float64", -(2**60)),
])
def test_exactly_representable_json_integers_are_accepted(dtype, value):
    normalized = gate1._normalized_numeric(value, dtype)
    assert normalized.is_integer()
    assert int(normalized) == value


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -float("inf")])
def test_bool_and_nonfinite_values_remain_rejected(value):
    with pytest.raises(ProvenanceError):
        gate1._normalized_numeric(value, "float64")


@pytest.mark.parametrize(("dtype", "value"), [
    ("float32", 1e-50), ("float32", -1e-50),
    ("float32", 10**400), ("float32", -(10**400)),
    ("float64", 10**400), ("float64", -(10**400)),
])
def test_numeric_underflow_and_overflow_are_rejected(dtype, value):
    with pytest.raises(ProvenanceError):
        gate1._normalized_numeric(value, dtype)


@pytest.mark.parametrize("value", [0.0, -0.0])
def test_signed_zero_is_preserved(value):
    normalized = gate1._normalized_numeric(value, "float32")
    assert math.copysign(1.0, normalized) == math.copysign(1.0, value)


@pytest.mark.parametrize(("dtype", "source", "destination"), [
    ("float32", 2**60 + 1, 2**60),
    ("float32", -(2**24 + 1), -(2**24)),
    ("float64", 2**53 + 1, 2**53),
    ("float64", -(2**53 + 1), -(2**53)),
])
def test_complete_packet_rejects_integer_precision_collision(
        tmp_path, dtype, source, destination):
    document, worker = fixture(tmp_path)
    evidence = install_raw_equivalence(
        document, tmp_path, prefix=f"integer-{dtype}-{abs(source)}", dtype=dtype,
    )
    mutate_raw_execution(
        evidence, tmp_path, 0,
        output_mutation=lambda output: output.update(values=[source, 2]),
    )
    mutate_raw_execution(
        evidence, tmp_path, 1,
        output_mutation=lambda output: output.update(values=[destination, 2]),
    )
    inputs = complete_packet_inputs(document, worker, tmp_path, f"integer-{dtype}")
    with pytest.raises(ProvenanceError, match="canonically representable"):
        gate1.build_review_packet(
            document, inputs[0], tmp_path, cli=["packet"], verification=inputs[1],
            implementation=inputs[2], manifest_ref=inputs[3],
        )


@pytest.mark.parametrize(("dtype", "value"), [
    ("float32", 2**60), ("float64", -(2**53)),
])
def test_complete_packet_and_cas_accept_exact_integer_controls(
        tmp_path, monkeypatch, dtype, value):
    document, worker = fixture(tmp_path)
    evidence = install_raw_equivalence(
        document, tmp_path, prefix=f"integer-control-{dtype}", dtype=dtype,
    )
    for index in (0, 1):
        mutate_raw_execution(
            evidence, tmp_path, index,
            output_mutation=lambda output, value=value: output.update(values=[value, 2]),
        )
    clean_tree = gate1.implementation_source_tree()
    clean_tree["dirty_state"] = {"clean": True, "scope": "entire-repository", "porcelain": [],
                                 "porcelain_sha256": digest([])}
    monkeypatch.setattr(gate1, "implementation_source_tree", lambda: copy.deepcopy(clean_tree))
    inputs = complete_packet_inputs(document, worker, tmp_path, f"integer-control-{dtype}")
    packet = gate1.build_review_packet(
        document, inputs[0], tmp_path, cli=["packet"], verification=inputs[1],
        implementation=inputs[2], manifest_ref=inputs[3],
    )
    assert packet["decision"] == "PASS"
    directory = gate1.write_review_packet(packet, tmp_path / "p")
    assert gate1.reconstruct_review_packet(directory) == packet


def test_contract8_adapter_protocol_binds_historical_source_mapping_and_defaults(tmp_path):
    from tests.test_gate1_legacy_contract8 import checkpoint_state
    reviewer = write(tmp_path / "legacy-reviewer.json", {
        "schema": 1, "kind": "independent_reviewer_identity", "provider": "openai",
        "model": "astra-high", "version": "2026-10-22",
    }, seal_digest=True)
    defaults = write(tmp_path / "legacy-defaults.json", {"defaults": "checkpoint-specific"})
    source_path = gate1.REPOSITORY / "src/anthropocene_rl/autonomous_v7/legacy_contract8.py"
    source = {"path": str(source_path.resolve()), "sha256": checkpoint_hash(source_path)}
    protocol = write_reviewed_protocol(tmp_path / "legacy-adapter.json", {
        "schema": 1, "kind": "gate1_legacy_adapter_protocol", "source_contract": 8,
        "historical_source_commit": "2f4cc895e6bcd897200d340851d69be98fd18e00",
        "source": source,
        "input_mapping": {"core": "raw-checkpoint-core-width-prefix",
                          "sets": ["sites", "fishing", "herds", "commodities", "contacts"],
                          "appended_sets": "excluded-from-contract-8-actor-input"},
        "decoder_defaults": {"commit": 1, "movement": contracts.NATIVE_MOVEMENT,
                             "fission": -1, "trade_posture": "native"},
        "default_resolution": defaults, "review": None,
    }, reviewer, ["historical-source", "input-mapping", "decoder", "default-resolution"])
    semantics = {"experts": False, "token": False, "commitments": False,
                 "learned_movement": False, "region_movement": False,
                 "learned_fission": False, "learned_trade": False,
                 "region_head": "flat", "grouped_region": False, "region_selection": None}
    checkpoint = tmp_path / "legacy-contract8.pt"
    torch.save(checkpoint_state(), checkpoint)
    artifact_ref = {"path": checkpoint.name, "sha256": checkpoint_hash(checkpoint)}
    raw_schema, _, _ = gate1.inspect_checkpoint(artifact_ref, tmp_path)
    node = {"artifact": artifact_ref, "raw_schema": raw_schema,
            "historical_semantics": semantics,
            "loading": {"mode": "reviewed-adapter", "implementation": protocol,
                        "default_resolution": defaults}}
    assert gate1._diagnostic_loading(node, tmp_path)["adapter"] == "pinned-contract-8"
    rewrite_reviewed_protocol(
        tmp_path, protocol,
        lambda body: body["decoder_defaults"].update(commit=2),
    )
    with pytest.raises(ProvenanceError, match="noncanonical"):
        gate1._diagnostic_loading(node, tmp_path)


def _global_rng_state() -> tuple:
    numpy = np.random.get_state()
    return (random.getstate(), (numpy[0], numpy[1].copy(), *numpy[2:]),
            torch.random.get_rng_state().clone())


def _assert_global_rng_state(expected: tuple) -> None:
    observed = _global_rng_state()
    assert observed[0] == expected[0]
    assert observed[1][0] == expected[1][0]
    assert np.array_equal(observed[1][1], expected[1][1])
    assert observed[1][2:] == expected[1][2:]
    assert torch.equal(observed[2], expected[2])


@pytest.mark.parametrize("failing", [False, True])
def test_manifest_import_and_packet_loader_eligibility_are_rng_neutral(
        tmp_path, monkeypatch, failing):
    document, worker = fixture(tmp_path)
    packet_inputs = complete_packet_inputs(document, worker, tmp_path, f"rng-{failing}")
    report = manual_report(document, "prior", "historical", worker, mandatory_values())
    original = gate1._diagnostic_loading_preserved

    def eligibility(*args, **kwargs):
        random.random(); np.random.random(); torch.rand(3)
        if failing:
            raise ProvenanceError("injected loader eligibility failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(gate1, "_diagnostic_loading_preserved", eligibility)
    calls = [
        lambda: gate1.validate_manifest(document, tmp_path),
        lambda: gate1._evaluation_index(document, [report], tmp_path),
        lambda: gate1.build_review_packet(
            document, packet_inputs[0], tmp_path, cli=["packet"],
            verification=packet_inputs[1], implementation=packet_inputs[2],
            manifest_ref=packet_inputs[3],
        ),
    ]
    for call in calls:
        random.seed(123); np.random.seed(456); torch.manual_seed(789)
        before = _global_rng_state()
        if failing:
            with pytest.raises(ProvenanceError, match="injected loader"):
                call()
        else:
            call()
        _assert_global_rng_state(before)


def test_manifest_import_path_rejects_contract8_checkpoint_missing_required_tensor(tmp_path):
    document, _ = fixture(tmp_path)
    target = trained_ids(document)[0]
    install_contract8_checkpoint(
        document, tmp_path, target, remove_actor_key="strategic.bias",
    )
    with pytest.raises(ProvenanceError, match="raw actor keys differ"):
        gate1.validate_manifest(document, tmp_path)


def test_contract8_packet_reconstruction_uses_preserved_adapter_source_only(
        tmp_path, monkeypatch):
    document, worker = fixture(tmp_path)
    target = trained_ids(document)[0]
    install_contract8_checkpoint(document, tmp_path, target)
    clean_tree = gate1.implementation_source_tree()
    clean_tree["dirty_state"] = {"clean": True, "scope": "entire-repository", "porcelain": [],
                                 "porcelain_sha256": digest([])}
    monkeypatch.setattr(gate1, "implementation_source_tree", lambda: copy.deepcopy(clean_tree))
    inputs = complete_packet_inputs(document, worker, tmp_path, "contract8-cas")
    packet = gate1.build_review_packet(
        document, inputs[0], tmp_path, cli=["packet"], verification=inputs[1],
        implementation=inputs[2], manifest_ref=inputs[3],
    )
    directory = gate1.write_review_packet(packet, tmp_path / "p")
    live_source = (gate1.REPOSITORY / "src/anthropocene_rl/autonomous_v7/legacy_contract8.py").resolve()
    original_hash = gate1.checkpoint_hash

    def cas_only_hash(path):
        if Path(path).resolve() == live_source:
            raise AssertionError("strict packet replay consulted the live adapter source")
        return original_hash(path)

    monkeypatch.setattr(gate1, "checkpoint_hash", cas_only_hash)
    assert gate1.reconstruct_review_packet(directory)["decision"] == "PASS"
    source_sha = original_hash(live_source)
    source_cas = gate1._native_path(directory / "cas" / source_sha)
    source_cas.write_bytes(source_cas.read_bytes() + b"\ncorrupt")
    with pytest.raises(ProvenanceError, match="content hash mismatch"):
        gate1.reconstruct_review_packet(directory)
