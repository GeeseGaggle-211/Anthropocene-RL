"""The Gate 1 retrieval preflight is read-only, exact and non-authorizing."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest
import torch

from anthropocene_rl.autonomous_v7 import gate1_preflight as preflight
from anthropocene_rl.autonomous_v7 import initialization_lineage
from anthropocene_rl.autonomous_v7.provenance import ProvenanceError, digest
from tests.test_economic_diagnostics import transition, update_pair, write_artifact


def seal(path: Path, body: dict) -> dict:
    value = copy.deepcopy(body)
    value["digest"] = digest(value)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return preflight.file_reference(path)


def missing(reason: str = "not recovered by read-only retrieval") -> dict:
    return {"status": "missing", "reason": reason}


def node(node_id: str) -> dict:
    expected = preflight.EXPECTED_NODES[node_id]
    return {
        "id": node_id,
        "role": expected["role"],
        "run_identity": {
            "historical_label": expected["historical_label"],
            "producer_run_id": None,
            "arm_id": expected["arm_id"],
            "optimizer_seed": expected["optimizer_seed"],
            "pairing_status": expected["pairing_status"],
            "pairing_group": None,
        },
        "artifacts": {name: missing() for name in preflight.SLOTS},
    }


def inventory(tmp_path: Path, *, conversion: dict | None = None) -> tuple[dict, Path]:
    body = {
        "schema": 1,
        "kind": preflight.KIND,
        "baseline_commit": preflight.BASELINE_COMMIT,
        "execution": preflight.EXECUTION,
        "conversion": conversion or missing("conversion history is not yet recovered"),
        "nodes": [node(node_id) for node_id in preflight.EXPECTED_NODES],
    }
    path = tmp_path / "retrieval-inventory.json"
    seal(path, body)
    return body, path


def reseal(path: Path, body: dict) -> None:
    value = copy.deepcopy(body)
    value["digest"] = digest(value)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def install_endpoint(tmp_path: Path, row: dict, *, value: float = 1.0) -> dict:
    identity = row["run_identity"]
    if identity["producer_run_id"] is None:
        identity["producer_run_id"] = f"run:{identity['historical_label']}"
    state = {
        "contract": preflight.EXPECTED_NODES[row["id"]]["checkpoint_contract"],
        "core_width": 4,
        "seed": identity["optimizer_seed"],
        "actor": {"weight": torch.tensor([value], dtype=torch.float32)},
        "critic": {"weight": torch.tensor([value + 1], dtype=torch.float32)},
        "normalizer": {"core_mean": [0.0] * 4, "core_std": [1.0] * 4},
        "run_manifest": {
            "options": {
                "run_id": identity["producer_run_id"],
                "arm": identity["arm_id"],
                "seed": identity["optimizer_seed"],
            },
            "code": {"commit": "1" * 40, "clean": True},
            "simulator": {"commit": "2" * 40, "buildHash": "3" * 64},
        },
    }
    path = tmp_path / f"{row['id']}.pt"
    torch.save(state, path)
    row["artifacts"]["evaluation-checkpoint"] = {
        "status": "present",
        "source": {"kind": "file", "artifact": preflight.file_reference(path)},
    }
    for slot, key in (
        ("run-manifest", "run_manifest"),
        ("args-config", "run_manifest.options"),
        ("producer-build-provenance", "run_manifest"),
        ("normalizer", "normalizer"),
        ("critic", "critic"),
    ):
        embedded = state
        for part in key.split("."):
            embedded = embedded[part]
        row["artifacts"][slot] = {
            "status": "present",
            "source": preflight.embedded_reference(
                embedded, container_slot="evaluation-checkpoint", key=key
            ),
        }
    if state["contract"] < 12:
        adapter_path = tmp_path / f"{row['id']}-adapter-bundle.json"
        adapter_ref = seal(
            adapter_path,
            {
                "schema": 1,
                "kind": "gate1_retrieved_legacy_adapter_bundle",
                "checkpoint_sha256": row["artifacts"]["evaluation-checkpoint"]["source"][
                    "artifact"
                ]["sha256"],
                "historical_conversion": False,
            },
        )
        row["artifacts"]["adapter-default-migration"] = {
            "status": "present",
            "source": {"kind": "file", "artifact": adapter_ref},
        }
    return state


def install_trace(
    tmp_path: Path,
    row: dict,
    *,
    paired_update: bool = True,
    complete_update_hashes: bool = True,
    update_count: int = 1,
) -> None:
    identity = row["run_identity"]
    checkpoint_path = Path(row["artifacts"]["evaluation-checkpoint"]["source"]["artifact"]["path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    endpoint_actor_sha = preflight._hash_value(checkpoint["actor"])
    evidence = transition(
        seed=identity["optimizer_seed"], arm=identity["arm_id"], world=2000, turn=1
    )
    evidence["runId"] = identity["producer_run_id"]
    evidence["evidenceClass"] = "original_training_trace"
    evidence["policyVersion"] = 0
    evidence["actorSha256"] = "a" * 64
    updates = []
    before_hashes = {
        "actor": evidence["actorSha256"],
        "critic": "b" * 64,
        "cost": "c" * 64,
    }
    if complete_update_hashes:
        before_hashes["optimizer"] = f"{30:064x}"
    for index in range(update_count):
        pair = update_pair([evidence])
        after_actor = endpoint_actor_sha if index == update_count - 1 else f"{index + 10:064x}"
        after_hashes = {
            "actor": after_actor,
            "critic": f"{index + 40:064x}",
            "cost": f"{index + 50:064x}",
        }
        if complete_update_hashes:
            after_hashes["optimizer"] = f"{index + 31:064x}"
        for update in pair:
            update["runId"] = identity["producer_run_id"]
            update["arm"] = identity["arm_id"]
            update["optimizerSeed"] = identity["optimizer_seed"]
            update["evidenceClass"] = "original_training_trace"
            update["minibatch"] = index
            if update["event"].endswith("-start"):
                update["policyVersionBefore"] = index
                update["beforeHashes"] = dict(before_hashes)
            else:
                update["policyVersionAfter"] = index + 1
                update["afterHashes"] = dict(after_hashes)
        updates.extend(pair)
        before_hashes = after_hashes
    if not paired_update:
        updates = updates[:1]
    root = tmp_path / f"{row['id']}-trace"
    write_artifact(root, [evidence], updates=updates)
    transitions_ref = preflight.file_reference(root / "transitions.jsonl")
    updates_ref = preflight.file_reference(root / "updates.jsonl")
    bindings = {
        "checkpoint_sha256": row["artifacts"]["evaluation-checkpoint"]["source"]["artifact"][
            "sha256"
        ],
        "endpoint_actor_sha256": endpoint_actor_sha,
        "normalizer_sha256": row["artifacts"]["normalizer"]["source"]["sha256"],
        "run_manifest_sha256": row["artifacts"]["run-manifest"]["source"]["sha256"],
        "producer_provenance_sha256": row["artifacts"]["producer-build-provenance"]["source"][
            "sha256"
        ],
        "transitions_sha256": transitions_ref["sha256"],
        "updates_sha256": updates_ref["sha256"],
    }
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gate1Bindings"] = bindings
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    row["artifacts"]["transition-update-trace"] = {
        "status": "present",
        "source": {
            "kind": "economic-trace",
            "root": str(root),
            "bindings": bindings,
            "files": {
                "manifest": preflight.file_reference(root / "manifest.json"),
                "done": preflight.file_reference(root / "done.json"),
                "transitions": transitions_ref,
                "updates": updates_ref,
            },
            "snapshots": [],
        },
    }


def rewrite_trace_updates(row: dict, updates: list[dict]) -> None:
    source = row["artifacts"]["transition-update-trace"]["source"]
    transitions_path = Path(source["files"]["transitions"]["path"])
    transitions = [
        json.loads(line)
        for line in transitions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    updates_path = Path(source["files"]["updates"]["path"])
    updates_path.write_text(
        "".join(json.dumps(update, sort_keys=True) + "\n" for update in updates),
        encoding="utf-8",
    )
    source["files"]["updates"] = preflight.file_reference(updates_path)
    source["bindings"]["updates_sha256"] = source["files"]["updates"]["sha256"]
    done_path = Path(source["files"]["done"]["path"])
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["updates"] = {
        "rows": len(updates),
        "sha256": source["files"]["updates"]["sha256"],
    }
    done["merkleRoot"] = preflight.economic_diagnostics.merkle_root([*transitions, *updates])
    done_path.write_text(json.dumps(done, sort_keys=True), encoding="utf-8")
    source["files"]["done"] = preflight.file_reference(done_path)
    manifest_path = Path(source["files"]["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gate1Bindings"] = source["bindings"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    source["files"]["manifest"] = preflight.file_reference(manifest_path)


def install_producer_inventory(tmp_path: Path, row: dict) -> None:
    artifacts: list[dict] = []
    for name, slot in row["artifacts"].items():
        if name == "producer-artifact-inventory" or slot.get("status") != "present":
            continue
        source = slot["source"]
        if source["kind"] == "file":
            artifacts.append(source["artifact"])
        elif source["kind"] == "economic-trace":
            artifacts.extend(source["files"].values())
            artifacts.extend(source["snapshots"])
    ref = seal(
        tmp_path / f"{row['id']}-producer-inventory.json",
        {
            "schema": 1,
            "kind": "gate1_retrieved_artifact_inventory",
            "node_id": row["id"],
            "producer_run_id": row["run_identity"]["producer_run_id"],
            "declared_source_root": f"retrieved://{row['run_identity']['producer_run_id']}",
            "complete": True,
            "artifacts": artifacts,
        },
    )
    row["artifacts"]["producer-artifact-inventory"] = {
        "status": "present",
        "source": {"kind": "file", "artifact": ref, "acquisition_receipt": None},
    }


def selected(report: dict, node_id: str) -> dict:
    return next(row for row in report["nodes"] if row["id"] == node_id)


def test_exact_inventory_and_no_authority_report_are_deterministic(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    first = preflight.run_preflight(path)
    second = preflight.run_preflight(path)
    assert first == second
    assert first["preflight_status"] == "HOLD"
    assert first["ready_for_protocol_freeze"] is False
    assert first["evaluation_authorized"] is False
    assert first["production_gate1"] == "HOLD"
    assert first["inventory_sha256"] in first["artifact_sha256s"]
    assert len(first["nodes"]) == 17
    assert first["digest"] == digest(
        {key: value for key, value in first.items() if key != "digest"}
    )

    body["nodes"].pop()
    reseal(path, body)
    with pytest.raises(ProvenanceError, match="exact 17"):
        preflight.run_preflight(path)


def test_self_asserted_complete_retrieval_remains_hold_not_gate1(tmp_path: Path) -> None:
    history = tmp_path / "conversion-history.txt"
    history.write_text("review source records no conversion\n", encoding="utf-8")
    support = seal(
        tmp_path / "conversion-applicability.json",
        {
            "schema": 1,
            "kind": "gate1_historical_slot_applicability",
            "node_id": "conversion",
            "slot": "conversion",
            "status": "not-applicable",
            "finding": "no conversion is recorded",
            "source_artifacts": [preflight.file_reference(history)],
        },
    )
    body, path = inventory(
        tmp_path,
        conversion={
            "status": "not-applicable",
            "reason": "content-backed history records no conversion",
            "support": support,
        },
    )
    for index, row in enumerate(body["nodes"], 1):
        install_endpoint(tmp_path, row, value=float(index))
        install_producer_inventory(tmp_path, row)
    reseal(path, body)

    report = preflight.run_preflight(path)
    assert report["preflight_status"] == "HOLD"
    assert report["preflight_status_scope"] == "retrieval-inventory-and-answerability-only"
    assert report["ready_for_protocol_freeze"] is False
    assert report["evaluation_authorized"] is False
    assert report["gate1_readiness_claimed"] is False
    assert report["production_gate1"] == "HOLD"
    assert any("separately reviewed provenance" in reason for reason in report["holds"])
    assert all(
        row["answerability"]["strict_evaluator_eligibility"]["status"] == "blocked"
        for row in report["nodes"]
    )


def test_t3_identity_is_exact_disjoint_and_unpaired(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3def-s75")
    row["run_identity"]["pairing_status"] = "matched"
    row["run_identity"]["pairing_group"] = "invented"
    reseal(path, body)
    with pytest.raises(ProvenanceError, match="pairing_status"):
        preflight.run_preflight(path)

    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3lo-s80")
    row["run_identity"]["optimizer_seed"] = 70
    reseal(path, body)
    with pytest.raises(ProvenanceError, match="optimizer_seed"):
        preflight.run_preflight(path)


def test_complete_endpoint_and_trace_classify_exact_historical_answerability(
    tmp_path: Path,
) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    state = install_endpoint(tmp_path, row)
    install_trace(tmp_path, row)
    resume = {
        "actor": state["actor"],
        "critic": state["critic"],
        "optimizers": {"actor": {"state": {0: {"step": 1}}, "param_groups": [{"params": [0]}]}},
        "iteration": 1,
        "python_rng": (3, (1, 2, 3), None),
        "numpy_rng": {"bit_generator": "PCG64", "state": {"state": 1, "inc": 1}},
        "torch_rng": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "simulator_rng": {"seed": 70, "counter": 1},
        "dataloader_state": {"order": [0]},
        "scheduler": {"status": "not-applicable", "reason": "no scheduler configured"},
        "scaler": {"status": "not-applicable", "reason": "CPU execution"},
        "run_manifest": state["run_manifest"],
    }
    resume_path = tmp_path / "resume.pt"
    torch.save(resume, resume_path)
    row["artifacts"]["resume-state"] = {
        "status": "present",
        "source": {"kind": "file", "artifact": preflight.file_reference(resume_path)},
    }
    install_producer_inventory(tmp_path, row)
    reseal(path, body)

    report = preflight.run_preflight(path)
    node_report = selected(report, "t3per-s70")
    answers = node_report["answerability"]
    assert answers["endpoint_artifact_inventory"]["status"] == "blocked"
    assert "separately reviewed provenance" in answers["endpoint_artifact_inventory"]["reasons"][-1]
    assert answers["strict_evaluator_eligibility"]["status"] == "blocked"
    assert answers["critic_state_identity"]["status"] == "answerable"
    assert answers["resume_sections_inventoried"]["status"] == "answerable"
    assert answers["exact_endpoint_resume_eligibility"]["status"] == "blocked"
    assert "restoration contract" in answers["exact_endpoint_resume_eligibility"]["reasons"][0]
    assert answers["historical_optimizer_sequence_reconstruction"]["status"] == "blocked"
    assert answers["transition_trace_structural_consistency"]["status"] == "answerable"
    assert answers["update_trace_structural_consistency"]["status"] == "answerable"
    assert answers["historical_transition_diagnostics"]["status"] == "blocked"
    assert (
        "separately reviewed authority"
        in answers["historical_transition_diagnostics"]["reasons"][0]
    )
    assert answers["historical_optimization_target_diagnostics"]["status"] == "blocked"
    assert answers["historical_optimization_target_diagnostics"]["objective_identities"] == [
        "welfare"
    ]
    assert answers["independent_return_critic_accuracy"]["status"] == "blocked"
    assert answers["historical_update_diagnostics"]["status"] == "blocked"
    assert answers["update_trace_structural_consistency"]["joined_update_count"] == 1
    assert report["preflight_status"] == "HOLD"  # Other historical endpoints remain missing.
    assert report["evaluation_authorized"] is False


def test_missing_resume_and_unpaired_update_do_not_overclaim(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    install_trace(tmp_path, row, paired_update=False)
    install_producer_inventory(tmp_path, row)
    reseal(path, body)

    report = preflight.run_preflight(path)
    answers = selected(report, "t3per-s70")["answerability"]
    assert answers["exact_endpoint_resume_eligibility"]["status"] == "blocked"
    assert answers["transition_trace_structural_consistency"]["status"] == "blocked"
    assert answers["historical_transition_diagnostics"]["status"] == "blocked"
    assert answers["historical_optimization_target_diagnostics"]["status"] == "blocked"
    assert answers["update_trace_structural_consistency"]["status"] == "blocked"
    assert answers["historical_update_diagnostics"]["status"] == "blocked"
    assert "no following completion" in answers["historical_update_diagnostics"]["reasons"][0]


def test_update_answerability_requires_optimizer_pre_and_post_hashes(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    install_trace(tmp_path, row, complete_update_hashes=False)
    install_producer_inventory(tmp_path, row)
    reseal(path, body)

    answers = selected(preflight.run_preflight(path), "t3per-s70")["answerability"]
    assert answers["historical_update_diagnostics"]["status"] == "blocked"
    assert "actor/critic/optimizer" in answers["historical_update_diagnostics"]["reasons"][0]


def test_checkpoint_critic_missingness_conflict_rejects(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    row["artifacts"]["critic"] = missing("claimed lost")
    reseal(path, body)
    with pytest.raises(ProvenanceError, match="has checkpoint critic bytes"):
        preflight.run_preflight(path)


def test_expected_lost_critic_forces_preflight_hold(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    checkpoint_ref = row["artifacts"]["evaluation-checkpoint"]["source"]["artifact"]
    checkpoint_path = Path(checkpoint_ref["path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint.pop("critic")
    torch.save(checkpoint, checkpoint_path)
    checkpoint_ref.update(preflight.file_reference(checkpoint_path))
    row["artifacts"]["critic"] = missing("expected critic bytes were lost")
    install_producer_inventory(tmp_path, row)
    reseal(path, body)

    report = preflight.run_preflight(path)
    answer = selected(report, "t3per-s70")["answerability"]
    assert answer["endpoint_artifact_inventory"]["status"] == "blocked"
    assert (
        "expected critic bytes are missing/lost" in answer["endpoint_artifact_inventory"]["reasons"]
    )
    assert report["preflight_status"] == "HOLD"
    assert report["ready_for_protocol_freeze"] is False


def test_producer_inventory_cannot_omit_a_bound_endpoint_artifact(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    install_producer_inventory(tmp_path, row)
    producer_ref = row["artifacts"]["producer-artifact-inventory"]["source"]["artifact"]
    producer_path = Path(producer_ref["path"])
    record = json.loads(producer_path.read_text(encoding="utf-8"))
    record.pop("digest")
    unrelated = tmp_path / "unrelated.log"
    unrelated.write_text("not the endpoint checkpoint\n", encoding="utf-8")
    record["artifacts"] = [preflight.file_reference(unrelated)]
    producer_ref.update(seal(producer_path, record))
    reseal(path, body)

    with pytest.raises(ProvenanceError, match="does not exactly bind slot"):
        preflight.run_preflight(path)


def test_not_applicable_requires_rehashed_content_support(tmp_path: Path) -> None:
    history = tmp_path / "historical-index.txt"
    history.write_text(
        "no historical conversion entry in reviewed source index\n", encoding="utf-8"
    )
    history_ref = preflight.file_reference(history)
    support = seal(
        tmp_path / "conversion-support.json",
        {
            "schema": 1,
            "kind": "gate1_historical_slot_applicability",
            "node_id": "conversion",
            "slot": "conversion",
            "status": "not-applicable",
            "finding": "the reviewed historical index records no conversion",
            "source_artifacts": [history_ref],
        },
    )
    _, path = inventory(
        tmp_path,
        conversion={
            "status": "not-applicable",
            "reason": "content-backed history records no conversion",
            "support": support,
        },
    )
    report = preflight.run_preflight(path)
    assert report["conversion"]["status"] == "not-applicable"
    assert report["conversion"]["review_status"] == "unverified"
    assert report["ready_for_protocol_freeze"] is False
    assert support["sha256"] in report["artifact_sha256s"]
    assert history_ref["sha256"] in report["artifact_sha256s"]

    history.write_text("mutated after inventory sealing\n", encoding="utf-8")
    with pytest.raises(
        ProvenanceError, match=r"pre-existing-local provenance|size or SHA-256 differs"
    ):
        preflight.run_preflight(path)


def test_trace_identity_and_artifact_hash_conflicts_reject(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    install_trace(tmp_path, row)
    trace_path = Path(
        row["artifacts"]["transition-update-trace"]["source"]["files"]["transitions"]["path"]
    )
    trace_path.write_text(trace_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    reseal(path, body)
    with pytest.raises(
        ProvenanceError, match=r"pre-existing-local provenance|size or SHA-256 differs"
    ):
        preflight.run_preflight(path)


def test_trace_endpoint_binding_cannot_be_resealed_to_another_producer(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    install_trace(tmp_path, row)
    source = row["artifacts"]["transition-update-trace"]["source"]
    source["bindings"]["producer_provenance_sha256"] = "f" * 64
    manifest_path = Path(source["files"]["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gate1Bindings"] = source["bindings"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    source["files"]["manifest"] = preflight.file_reference(manifest_path)
    reseal(path, body)
    with pytest.raises(ProvenanceError, match="endpoint/provenance bindings differ"):
        preflight.run_preflight(path)


def test_exact_value_hash_distinguishes_tuple_from_list() -> None:
    assert preflight._hash_value([1, 2, 3]) != preflight._hash_value((1, 2, 3))
    assert preflight._hash_value({"a": {"b": 1}, "c": 2}) != preflight._hash_value(
        {"a": {"b": 1, "c": 2}}
    )
    assert preflight._hash_value({"a": 1}) != preflight._hash_value({"a": "1"})
    assert preflight._hash_value({"a": True}) != preflight._hash_value({"a": 1})
    with pytest.raises(ProvenanceError, match="non-string key"):
        preflight._hash_value({1: "a"})


@pytest.mark.parametrize(
    "slot",
    ["normalizer", "critic", "args-config", "producer-build-provenance"],
)
def test_sensitive_slots_cannot_alias_actor_bytes(tmp_path: Path, slot: str) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    state = install_endpoint(tmp_path, row)
    row["artifacts"][slot] = {
        "status": "present",
        "source": preflight.embedded_reference(
            state["actor"], container_slot="evaluation-checkpoint", key="actor"
        ),
    }
    reseal(path, body)
    with pytest.raises(ProvenanceError, match="must bind checkpoint section"):
        preflight.run_preflight(path)


def test_cross_slot_file_alias_is_rejected(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    row["artifacts"]["selection-checkpoint"] = copy.deepcopy(
        row["artifacts"]["evaluation-checkpoint"]
    )
    install_producer_inventory(tmp_path, row)
    reseal(path, body)
    with pytest.raises(ProvenanceError, match=r"repeats a file|aliases one file"):
        preflight.run_preflight(path)


def test_incomplete_resume_state_is_never_exactly_answerable(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    state = install_endpoint(tmp_path, row)
    hostile = {
        "actor": state["actor"],
        "critic": state["critic"],
        "optimizers": {"actor": None},
        "iteration": -1,
        "python_rng": (3, (1,), None),
        "numpy_rng": {"bit_generator": "PCG64", "state": {"state": 1}},
        "torch_rng": torch.tensor([1], dtype=torch.uint8),
        "simulator_rng": {"seed": 70},
        "dataloader_state": {"order": [0]},
        "run_manifest": state["run_manifest"],
    }
    resume_path = tmp_path / "hostile-resume.pt"
    torch.save(hostile, resume_path)
    row["artifacts"]["resume-state"] = {
        "status": "present",
        "source": {"kind": "file", "artifact": preflight.file_reference(resume_path)},
    }
    install_producer_inventory(tmp_path, row)
    reseal(path, body)
    answer = selected(preflight.run_preflight(path), "t3per-s70")["answerability"]
    assert answer["resume_sections_inventoried"]["status"] == "blocked"
    assert "exact reconstruction sections" in answer["resume_sections_inventoried"]["reasons"][0]
    assert answer["exact_endpoint_resume_eligibility"]["status"] == "blocked"

    hostile["scheduler"] = {
        "status": "not-applicable",
        "reason": "no scheduler configured",
    }
    hostile["scaler"] = {"status": "not-applicable", "reason": "CPU execution"}
    torch.save(hostile, resume_path)
    row["artifacts"]["resume-state"]["source"]["artifact"] = preflight.file_reference(resume_path)
    install_producer_inventory(tmp_path, row)
    reseal(path, body)
    answer = selected(preflight.run_preflight(path), "t3per-s70")["answerability"]
    assert "structurally incomplete" in answer["resume_sections_inventoried"]["reasons"][0]

    hostile["optimizers"] = {
        "actor": {"state": {0: {"step": 1}}, "param_groups": [{"params": [0]}]}
    }
    torch.save(hostile, resume_path)
    row["artifacts"]["resume-state"]["source"]["artifact"] = preflight.file_reference(resume_path)
    install_producer_inventory(tmp_path, row)
    reseal(path, body)
    answer = selected(preflight.run_preflight(path), "t3per-s70")["answerability"]
    assert "iteration is invalid" in answer["resume_sections_inventoried"]["reasons"][0]


def test_resealed_trace_bindings_cannot_substitute_endpoint_actor(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    install_trace(tmp_path, row)
    checkpoint_ref = row["artifacts"]["evaluation-checkpoint"]["source"]["artifact"]
    checkpoint_path = Path(checkpoint_ref["path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["actor"] = {"weight": torch.tensor([99.0], dtype=torch.float32)}
    torch.save(checkpoint, checkpoint_path)
    checkpoint_ref.update(preflight.file_reference(checkpoint_path))
    trace = row["artifacts"]["transition-update-trace"]["source"]
    trace["bindings"]["checkpoint_sha256"] = checkpoint_ref["sha256"]
    trace["bindings"]["endpoint_actor_sha256"] = preflight._hash_value(checkpoint["actor"])
    manifest_path = Path(trace["files"]["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gate1Bindings"] = trace["bindings"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    trace["files"]["manifest"] = preflight.file_reference(manifest_path)
    install_producer_inventory(tmp_path, row)
    reseal(path, body)

    answers = selected(preflight.run_preflight(path), "t3per-s70")["answerability"]
    assert answers["historical_transition_diagnostics"]["status"] == "blocked"
    assert answers["historical_update_diagnostics"]["status"] == "blocked"
    assert "endpoint actor" in answers["historical_transition_diagnostics"]["reasons"][0]


def test_update_batch_rejects_future_after_policy_transition(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    install_trace(tmp_path, row)
    trace = row["artifacts"]["transition-update-trace"]["source"]
    transitions_path = Path(trace["files"]["transitions"]["path"])
    transition_row = json.loads(transitions_path.read_text(encoding="utf-8").strip())
    transition_row["policyVersion"] = 1
    transition_row["actorSha256"] = trace["bindings"]["endpoint_actor_sha256"]
    transitions_path.write_text(json.dumps(transition_row, sort_keys=True) + "\n", encoding="utf-8")
    trace["files"]["transitions"] = preflight.file_reference(transitions_path)
    trace["bindings"]["transitions_sha256"] = trace["files"]["transitions"]["sha256"]
    done_path = Path(trace["files"]["done"]["path"])
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["transitions"]["sha256"] = trace["files"]["transitions"]["sha256"]
    updates_path = Path(trace["files"]["updates"]["path"])
    update_rows = [
        json.loads(line)
        for line in updates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    done["merkleRoot"] = preflight.economic_diagnostics.merkle_root([transition_row, *update_rows])
    done_path.write_text(json.dumps(done, sort_keys=True), encoding="utf-8")
    trace["files"]["done"] = preflight.file_reference(done_path)
    manifest_path = Path(trace["files"]["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gate1Bindings"] = trace["bindings"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    trace["files"]["manifest"] = preflight.file_reference(manifest_path)
    install_producer_inventory(tmp_path, row)
    reseal(path, body)

    answers = selected(preflight.run_preflight(path), "t3per-s70")["answerability"]
    assert answers["historical_transition_diagnostics"]["status"] == "blocked"
    assert "future policy" in answers["historical_transition_diagnostics"]["reasons"][0]


def test_reused_rollout_may_feed_later_updates_without_historical_provenance(
    tmp_path: Path,
) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    install_trace(tmp_path, row, update_count=2)
    install_producer_inventory(tmp_path, row)
    reseal(path, body)

    answers = selected(preflight.run_preflight(path), "t3per-s70")["answerability"]
    assert answers["transition_trace_structural_consistency"]["status"] == "answerable"
    assert answers["update_trace_structural_consistency"]["status"] == "answerable"
    assert answers["update_trace_structural_consistency"]["joined_update_count"] == 2
    assert answers["historical_transition_diagnostics"]["status"] == "blocked"
    assert answers["historical_update_diagnostics"]["status"] == "blocked"
    assert "separately reviewed authority" in answers["historical_update_diagnostics"]["reasons"][0]


@pytest.mark.parametrize(
    ("attack", "reason"),
    [
        ("disconnected", "not contiguous"),
        ("version-jump", "delta is not zero or one"),
        ("completion-before-start", "completion precedes its start"),
        ("reversed-file", "completion precedes its start"),
    ],
)
def test_update_trace_requires_ordered_contiguous_producer_chain(
    tmp_path: Path, attack: str, reason: str
) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    install_trace(tmp_path, row, update_count=2)
    source = row["artifacts"]["transition-update-trace"]["source"]
    updates_path = Path(source["files"]["updates"]["path"])
    updates = [
        json.loads(line)
        for line in updates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if attack == "disconnected":
        updates[2]["policyVersionBefore"] = 7
        updates[3]["policyVersionAfter"] = 8
    elif attack == "version-jump":
        updates[1]["policyVersionAfter"] = 7
    elif attack == "completion-before-start":
        updates[0], updates[1] = updates[1], updates[0]
    else:
        updates.reverse()
    rewrite_trace_updates(row, updates)
    install_producer_inventory(tmp_path, row)
    reseal(path, body)

    answers = selected(preflight.run_preflight(path), "t3per-s70")["answerability"]
    assert answers["transition_trace_structural_consistency"]["status"] == "blocked"
    assert reason in answers["transition_trace_structural_consistency"]["reasons"][0]
    assert answers["update_trace_structural_consistency"]["status"] == "blocked"


def test_unknown_producer_identity_is_an_explicit_noninvented_hold(tmp_path: Path) -> None:
    body, path = inventory(tmp_path)
    assert all(row["run_identity"]["producer_run_id"] is None for row in body["nodes"])
    report = preflight.run_preflight(path)
    assert report["preflight_status"] == "HOLD"
    assert all(row["run_identity"]["producer_run_id"] is None for row in report["nodes"])
    assert all(
        any(
            "exact producer run identity" in reason
            for reason in row["answerability"]["endpoint_artifact_inventory"]["reasons"]
        )
        for row in report["nodes"]
    )


def test_remote_acquisition_requires_and_inventories_a_receipt(tmp_path: Path) -> None:
    artifact = tmp_path / "downloaded.bin"
    artifact.write_bytes(b"artifact")
    local_ref = preflight.file_reference(artifact)
    missing_observed_time = copy.deepcopy(local_ref)
    missing_observed_time["local_mtime_ns"] = None
    with pytest.raises(ProvenanceError, match="file metadata is malformed"):
        preflight._file_ref(missing_observed_time, tmp_path, "retrieved artifact")
    local_ref["local_mtime_ns"] += 1
    with pytest.raises(ProvenanceError, match="local modification time differs"):
        preflight._file_ref(local_ref, tmp_path, "retrieved artifact")
    missing_local_source_time = preflight.file_reference(artifact)
    missing_local_source_time["acquisition"]["source_mtime"] = None
    missing_local_source_time["acquisition"]["source_mtime_unavailable_reason"] = (
        "local time allegedly unavailable"
    )
    with pytest.raises(ProvenanceError, match="pre-existing-local provenance"):
        preflight._file_ref(missing_local_source_time, tmp_path, "retrieved artifact")
    wrong_local_source_time = preflight.file_reference(artifact)
    wrong_local_source_time["acquisition"]["source_mtime"] = str(
        wrong_local_source_time["local_mtime_ns"] + 1
    )
    with pytest.raises(ProvenanceError, match="pre-existing-local provenance"):
        preflight._file_ref(wrong_local_source_time, tmp_path, "retrieved artifact")
    disguised_remote = {
        "source_uri": "gs://example/artifact",
        "source_generation": None,
        "source_size_bytes": 8,
        "source_mtime": None,
        "source_mtime_unavailable_reason": "remote listing omitted modification time",
        "source_etag": None,
        "method": "pre-existing-local",
        "receipt_sha256": None,
    }
    with pytest.raises(ProvenanceError, match="pre-existing-local provenance"):
        preflight.file_reference(artifact, acquisition=disguised_remote)
    bad = {
        "source_uri": "gs://example/artifact",
        "source_generation": "1",
        "source_size_bytes": 8,
        "source_mtime": "2026-10-23T00:00:00Z",
        "source_mtime_unavailable_reason": None,
        "source_etag": "etag",
        "method": "read-only-download",
        "receipt_sha256": None,
    }
    with pytest.raises(ProvenanceError, match="acquisition provenance"):
        preflight.file_reference(artifact, acquisition=bad)
    unavailable_without_reason = dict(
        bad,
        source_mtime=None,
        source_mtime_unavailable_reason=None,
        receipt_sha256="a" * 64,
    )
    with pytest.raises(ProvenanceError, match="acquisition provenance"):
        preflight.file_reference(artifact, acquisition=unavailable_without_reason)

    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    checkpoint_ref = row["artifacts"]["evaluation-checkpoint"]["source"]["artifact"]
    checkpoint_ref["acquisition"] = {
        "source_uri": "gs://example/endpoint.pt",
        "source_generation": "7",
        "source_size_bytes": checkpoint_ref["size_bytes"],
        "source_mtime": "2026-10-23T00:00:00Z",
        "source_mtime_unavailable_reason": None,
        "source_etag": "etag",
        "method": "read-only-download",
        "receipt_sha256": "a" * 64,
    }
    install_producer_inventory(tmp_path, row)
    reseal(path, body)
    with pytest.raises(ProvenanceError, match="acquisition receipt identity"):
        preflight.run_preflight(path)

    positive = tmp_path / "positive"
    positive.mkdir()
    body, path = inventory(positive)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(positive, row)
    checkpoint_ref = row["artifacts"]["evaluation-checkpoint"]["source"]["artifact"]
    receipt_path = positive / "download-receipt.json"
    receipt_ref = seal(
        receipt_path,
        {
            "schema": 1,
            "kind": "gate1_read_only_acquisition_receipt",
            "status": "complete",
            "source": {
                "uri": "gs://example/endpoint.pt",
                "generation": "1792308001259477",
                "size_bytes": checkpoint_ref["size_bytes"],
                "mtime": "2026-10-18T07:20:01Z",
                "mtime_unavailable_reason": None,
                "etag": "xreqIFyu1BFNM24l8lB8Zg==",
            },
            "destination": {
                "node_id": row["id"],
                "slot": "evaluation-checkpoint",
                "artifact_role": "artifact",
                "path": str(Path(checkpoint_ref["path"]).resolve()),
                "sha256": checkpoint_ref["sha256"],
                "size_bytes": checkpoint_ref["size_bytes"],
            },
            "method": "read-only-download",
            "acquired_at": "2026-10-23T00:00:00Z",
            "tool_identity": {"name": "fixture-read-only-copy", "version": "1"},
        },
    )
    checkpoint_ref["acquisition"] = {
        "source_uri": "gs://example/endpoint.pt",
        "source_generation": "1792308001259477",
        "source_size_bytes": checkpoint_ref["size_bytes"],
        "source_mtime": "2026-10-18T07:20:01Z",
        "source_mtime_unavailable_reason": None,
        "source_etag": "xreqIFyu1BFNM24l8lB8Zg==",
        "method": "read-only-download",
        "receipt_sha256": receipt_ref["sha256"],
    }
    install_producer_inventory(positive, row)
    producer_ref = row["artifacts"]["producer-artifact-inventory"]["source"]["artifact"]
    producer_path = Path(producer_ref["path"])
    record = json.loads(producer_path.read_text(encoding="utf-8"))
    record.pop("digest")
    record["artifacts"].append(receipt_ref)
    producer_ref.update(seal(producer_path, record))
    reseal(path, body)
    report = preflight.run_preflight(path)
    producer = selected(report, "t3per-s70")["artifact_slots"]["producer-artifact-inventory"]
    assert producer["declared_artifact_count"] == 2


def test_original_path_ancestors_reject_reparse_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ancestor = tmp_path / "retrieved"
    ancestor.mkdir()
    artifact = ancestor / "artifact.bin"
    artifact.write_bytes(b"artifact")
    ref = preflight.file_reference(artifact)
    metadata = os.lstat(ancestor)
    identity = (metadata.st_dev, metadata.st_ino)
    original = preflight._reparse

    def marked(candidate: os.stat_result) -> bool:
        return (candidate.st_dev, candidate.st_ino) == identity or original(candidate)

    monkeypatch.setattr(preflight, "_reparse", marked)
    with pytest.raises(ProvenanceError, match="symlink/reparse ancestor"):
        preflight._file_ref(ref, tmp_path, "retrieved artifact")


def test_identical_local_bytes_cannot_substitute_for_remote_slot_receipt(
    tmp_path: Path,
) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    checkpoint_ref = row["artifacts"]["evaluation-checkpoint"]["source"]["artifact"]
    checkpoint_path = Path(checkpoint_ref["path"])
    twin_path = tmp_path / "same-bytes-local.pt"
    twin_path.write_bytes(checkpoint_path.read_bytes())
    twin_ref = preflight.file_reference(twin_path)
    receipt_ref = seal(
        tmp_path / "remote-checkpoint-receipt.json",
        {
            "schema": 1,
            "kind": "gate1_read_only_acquisition_receipt",
            "status": "complete",
            "source": {
                "uri": "gs://example/endpoint.pt",
                "generation": "7",
                "size_bytes": checkpoint_ref["size_bytes"],
                "mtime": "2026-10-23T00:00:00Z",
                "mtime_unavailable_reason": None,
                "etag": "etag",
            },
            "destination": {
                "node_id": row["id"],
                "slot": "evaluation-checkpoint",
                "artifact_role": "artifact",
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint_ref["sha256"],
                "size_bytes": checkpoint_ref["size_bytes"],
            },
            "method": "read-only-download",
            "acquired_at": "2026-10-23T00:00:00Z",
            "tool_identity": {"name": "fixture-read-only-copy", "version": "1"},
        },
    )
    checkpoint_ref["acquisition"] = {
        "source_uri": "gs://example/endpoint.pt",
        "source_generation": "7",
        "source_size_bytes": checkpoint_ref["size_bytes"],
        "source_mtime": "2026-10-23T00:00:00Z",
        "source_mtime_unavailable_reason": None,
        "source_etag": "etag",
        "method": "read-only-download",
        "receipt_sha256": receipt_ref["sha256"],
    }
    install_producer_inventory(tmp_path, row)
    producer_ref = row["artifacts"]["producer-artifact-inventory"]["source"]["artifact"]
    producer_path = Path(producer_ref["path"])
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    producer.pop("digest")
    producer["artifacts"] = [
        twin_ref if ref["path"] == checkpoint_ref["path"] else ref for ref in producer["artifacts"]
    ]
    producer["artifacts"].append(receipt_ref)
    producer_ref.update(seal(producer_path, producer))
    reseal(path, body)

    with pytest.raises(ProvenanceError, match="does not exactly bind slot"):
        preflight.run_preflight(path)


def test_remote_producer_inventory_requires_its_own_valid_receipt(
    tmp_path: Path,
) -> None:
    body, path = inventory(tmp_path)
    row = next(value for value in body["nodes"] if value["id"] == "t3per-s70")
    install_endpoint(tmp_path, row)
    install_producer_inventory(tmp_path, row)
    source = row["artifacts"]["producer-artifact-inventory"]["source"]
    inventory_ref = source["artifact"]
    receipt_path = tmp_path / "producer-inventory-receipt.json"
    invalid_receipt = seal(
        receipt_path,
        {
            "schema": 1,
            "kind": "gate1_read_only_acquisition_receipt",
            "status": "complete",
            "source": {
                "uri": "gs://example/producer-inventory.json",
                "generation": "9",
                "size_bytes": inventory_ref["size_bytes"],
                "mtime": "2026-10-23T00:00:00Z",
                "mtime_unavailable_reason": None,
                "etag": "etag",
            },
            "destination": {
                "node_id": "wrong-node",
                "slot": "producer-artifact-inventory",
                "artifact_role": "artifact",
                "path": str(Path(inventory_ref["path"]).resolve()),
                "sha256": inventory_ref["sha256"],
                "size_bytes": inventory_ref["size_bytes"],
            },
            "method": "read-only-download",
            "acquired_at": "2026-10-23T00:00:00Z",
            "tool_identity": {"name": "fixture-read-only-copy", "version": "1"},
        },
    )
    inventory_ref["acquisition"] = {
        "source_uri": "gs://example/producer-inventory.json",
        "source_generation": "9",
        "source_size_bytes": inventory_ref["size_bytes"],
        "source_mtime": "2026-10-23T00:00:00Z",
        "source_mtime_unavailable_reason": None,
        "source_etag": "etag",
        "method": "read-only-download",
        "receipt_sha256": invalid_receipt["sha256"],
    }
    reseal(path, body)
    with pytest.raises(ProvenanceError, match="lacks its acquisition receipt"):
        preflight.run_preflight(path)

    source["acquisition_receipt"] = invalid_receipt
    reseal(path, body)
    with pytest.raises(ProvenanceError, match="acquisition receipt is invalid"):
        preflight.run_preflight(path)


def test_cli_preflight_never_loads_or_evaluates_a_lineage_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, path = inventory(tmp_path)
    output = tmp_path / "preflight-report.json"
    monkeypatch.setattr(
        initialization_lineage,
        "load_manifest",
        lambda *_args, **_kwargs: pytest.fail("preflight must not load an evaluation manifest"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "initialization_lineage",
            "preflight",
            "--inventory",
            str(path),
            "--out",
            str(output),
        ],
    )
    assert initialization_lineage.main() == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["production_gate1"] == "HOLD"
    assert report["evaluation_authorized"] is False
