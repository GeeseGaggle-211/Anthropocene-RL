from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts, evaluate, initialization_lineage
from anthropocene_rl.autonomous_v7 import gate1_freeze_candidate as freeze
from anthropocene_rl.autonomous_v7.gate1_diagnostics import service_outcome_registry
from anthropocene_rl.autonomous_v7.gate1_freeze_candidate import (
    _corpus,
    _git_bytes,
    _rng_schedule,
    _seed_scan,
    _verify_execution_identity,
    evaluate_reviewed_node,
)
from anthropocene_rl.autonomous_v7.gate1_production_adapter import (
    assemble_bc_phase_state,
    assemble_guard_only_state,
    comparator_runtime_identity,
    guard_only_protocol_fields,
    inactive_region_state,
    load_guarded_bc,
    run_contract8_comparator,
    tensor_state_identity,
)
from anthropocene_rl.autonomous_v7.provenance import (
    ProvenanceError,
    checkpoint_hash,
    digest,
)

SEMANTICS = {
    "experts": False,
    "token": True,
    "commitments": False,
    "learned_movement": False,
    "region_movement": False,
    "learned_fission": False,
    "learned_trade": False,
    "region_head": "flat",
    "grouped_region": False,
    "region_selection": None,
}


def sealed(path: Path, value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("digest", None)
    value["digest"] = digest(value)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": path.name, "sha256": checkpoint_hash(path)}


def native_state() -> dict:
    from anthropocene_rl.autonomous_v7.model import Actor, Normalizer

    actor = Actor(4, token=True, region_head="flat")
    normalizer = Normalizer(np.zeros(4, np.float32), np.ones(4, np.float32))
    for name, _, width in contracts.SETS:
        normalizer.set_mean[name] = np.zeros(width, np.float32)
        normalizer.set_std[name] = np.ones(width, np.float32)
    return {
        "contract": contracts.NATIVE_CONTRACT,
        "core_width": 4,
        "privileged_width": 0,
        "seed": 6,
        "token": True,
        "actor": actor.state_dict(),
        "critic": {},
        "normalizer": normalizer.state_dict(),
    }


def bc_protocol(bc_state: dict, support: dict) -> dict:
    bc_identity = tensor_state_identity(bc_state["actor"])
    support_identity = tensor_state_identity(support["actor"])
    return {
        "schema": 1,
        "kind": "gate1_bc_phase_adapter_protocol",
        "bc_actor_identity": bc_identity,
        "support_actor_identity": support_identity,
        "equivalence_scope": "actor-tensors-only",
        "behavioral_equivalence_claim": False,
        "historical_bc_normalizer_claim": False,
        "support_normalizer_semantics": "unguarded",
        "missing_guarded_clone_claim": False,
        "appended_inactive_region_state": inactive_region_state(
            support["core_width"],
            experts=False,
            token=True,
        )["identity"],
        "appended_region_scope": "inactive-under-learned_movement=false",
        "composite_scope": "current-diagnostic-only",
        "implicit_defaults": [],
    }


def recovered_native_state() -> dict:
    state = native_state()
    for name in ("region_flat", "region_group.weight", "region_group.bias"):
        state["actor"].pop(name)
    return state


def test_bc_phase_adapter_requires_literal_actor_identity_and_no_defaults():
    support = recovered_native_state()
    bc_state = {"actor": copy.deepcopy(support["actor"]), "log": []}
    assembled = assemble_bc_phase_state(bc_state, support, bc_protocol(bc_state, support))
    assert set(assembled["actor"]) == set(bc_state["actor"]) | {
        "region_flat",
        "region_group.weight",
        "region_group.bias",
    }
    assert all(
        torch.equal(assembled["actor"][name], value) for name, value in bc_state["actor"].items()
    )
    assert assembled["normalizer"] == support["normalizer"]
    name = next(
        name
        for name, value in bc_state["actor"].items()
        if value.dtype.is_floating_point and value.numel()
    )
    bc_state["actor"][name] = bc_state["actor"][name].clone()
    bc_state["actor"][name].reshape(-1)[0] += 1
    forged = bc_protocol(bc_state, support)
    with pytest.raises(ProvenanceError, match="not literally identical"):
        assemble_bc_phase_state(bc_state, support, forged)


def test_bc_phase_adapter_rejects_guarded_support_mislabelled_as_t3_initialization():
    support = recovered_native_state()
    bc_state = {"actor": copy.deepcopy(support["actor"]), "log": []}
    protocol = bc_protocol(bc_state, support)
    support["normalizer"]["guard"] = 1.0
    with pytest.raises(ProvenanceError, match="proved unguarded T3 initialization"):
        assemble_bc_phase_state(bc_state, support, protocol)


def test_evaluate_loads_bc_phase_through_bound_composite_adapter_without_upgrade(
    tmp_path, monkeypatch
):
    support = recovered_native_state()
    bc_state = {"actor": copy.deepcopy(support["actor"]), "log": []}
    support_path = tmp_path / "t3-initialization.pt"
    torch.save(support, support_path)
    bc_path = tmp_path / "bc.pt"
    torch.save(bc_state, bc_path)
    protocol = bc_protocol(bc_state, support)
    protocol.update(
        {
            "bc_checkpoint_sha256": checkpoint_hash(bc_path),
            "support_checkpoint_sha256": checkpoint_hash(support_path),
        }
    )
    protocol_ref = sealed(tmp_path / "bc-protocol.json", protocol)
    loading = {
        "mode": "reviewed-adapter",
        "adapter": "t3-initialization-bc-composite",
        "protocol": protocol_ref,
        "support_checkpoint": {"path": support_path.name, "sha256": checkpoint_hash(support_path)},
    }
    monkeypatch.setattr(
        evaluate, "upgrade", lambda value: (_ for _ in ()).throw(AssertionError("upgrade called"))
    )
    actor, normalizer = evaluate.load(
        bc_path, strict=True, resolved_semantics=SEMANTICS, diagnostic_loading=loading
    )
    assert all(
        torch.equal(actor.state_dict()[name], value) for name, value in bc_state["actor"].items()
    )
    assert normalizer.state_dict() == support["normalizer"]
    assert evaluate.decoder_identity(actor) == "native-contract-12"


def guarded_normalizer_from(base: dict) -> dict:
    state = copy.deepcopy(base)
    state["guard"] = 10.0
    for name in [key[:-5] for key in state if key.endswith("_mean")]:
        state[name + "_dead"] = [index == 0 for index in range(len(state[name + "_mean"]))]
    return state


def test_guard_only_intervention_preserves_actor_and_affine_moments(tmp_path):
    base = recovered_native_state()
    guarded = {
        "actor": copy.deepcopy(base["actor"]),
        "normalizer": guarded_normalizer_from(base["normalizer"]),
    }
    identities = guard_only_protocol_fields(base, guarded)
    protocol = {
        "schema": 1,
        "kind": "gate1_guard_only_intervention_protocol",
        "base_checkpoint_sha256": None,
        "guard_source_checkpoint_sha256": None,
        **identities,
        "composition_scope": "prospective-current-diagnostic-only",
        "normalizer_change_only": True,
        "historical_artifact_claim": False,
        "historical_ancestry_claim": False,
        "outcome_selection_used": False,
        "implicit_defaults": [],
    }
    assembled = assemble_guard_only_state(base, guarded, protocol)
    assert tensor_state_identity(assembled["actor"]) == identities["evaluated_actor_identity"]
    assert assembled["normalizer"]["guard"] == 10.0
    for name, value in base["normalizer"].items():
        assert assembled["normalizer"][name] == value

    base_path = tmp_path / "t3-init.pt"
    guard_path = tmp_path / "t2m-s6.bc.pt"
    torch.save(base, base_path)
    torch.save(guarded, guard_path)
    protocol["base_checkpoint_sha256"] = checkpoint_hash(base_path)
    protocol["guard_source_checkpoint_sha256"] = checkpoint_hash(guard_path)
    protocol_ref = sealed(tmp_path / "guard-only.json", protocol)
    actor, normalizer = evaluate.load(
        base_path,
        strict=True,
        resolved_semantics=SEMANTICS,
        diagnostic_loading={
            "mode": "reviewed-adapter",
            "adapter": "t3-init-guard-only-intervention",
            "protocol": protocol_ref,
            "guard_source_checkpoint": {
                "path": guard_path.name,
                "sha256": checkpoint_hash(guard_path),
            },
        },
    )
    assert tensor_state_identity(actor.state_dict()) == identities["evaluated_actor_identity"]
    assert normalizer.guard == 10.0
    assert normalizer.dead["core"].tolist() == guarded["normalizer"]["core_dead"]
    assert evaluate.decoder_identity(actor) == "native-contract-12"

    forged = copy.deepcopy(guarded)
    forged["normalizer"]["core_mean"][0] = 1.0
    with pytest.raises(ProvenanceError, match="not literally identical"):
        guard_only_protocol_fields(base, forged)


def test_guarded_bc_actor_only_loader_is_exact_and_rejects_dummy_critic(tmp_path):
    base = native_state()
    appended_names = {"region_flat", "region_group.weight", "region_group.bias"}
    actor_state = {
        name: value for name, value in base["actor"].items() if name not in appended_names
    }
    guarded = {
        "actor": actor_state,
        "log": [{} for _ in range(12)],
        "normalizer": guarded_normalizer_from(base["normalizer"]),
        "bc_config": {"guard": 10.0, "epochs": 12},
    }
    checkpoint = tmp_path / "guarded.bc.pt"
    torch.save(guarded, checkpoint)
    source = tmp_path / "guarded-historical-source"
    source.mkdir()
    for leaf in ("model.py", "contracts.py"):
        (source / leaf).write_text(leaf, encoding="utf-8")
    from anthropocene_rl.autonomous_v7.model import Actor

    expected = Actor(4, token=True, region_head="flat").state_dict()
    append_identity = tensor_state_identity(
        {
            "region_flat": torch.tensor(True),
            "region_group.weight": torch.zeros_like(expected["region_group.weight"]),
            "region_group.bias": torch.zeros_like(expected["region_group.bias"]),
        }
    )
    protocol_ref = sealed(
        tmp_path / "guarded-protocol.json",
        {
            "schema": 1,
            "kind": "gate1_guarded_bc_adapter_protocol",
            "checkpoint_sha256": checkpoint_hash(checkpoint),
            "core_width": 4,
            "bc_config": guarded["bc_config"],
            "historical_source_root": source.name,
            "historical_sources": {
                leaf: checkpoint_hash(source / leaf) for leaf in ("model.py", "contracts.py")
            },
            "appended_inactive_region_state": append_identity,
            "appended_region_scope": "inactive-under-learned_movement=false",
        },
    )
    actor, normalizer = load_guarded_bc(
        checkpoint,
        {"protocol": protocol_ref},
        base=tmp_path,
    )
    assert normalizer.guard == 10.0
    assert actor.state_dict()["region_flat"].item() is True
    assert evaluate.decoder_identity(actor) == "native-contract-12"
    guarded["critic"] = {}
    torch.save(guarded, checkpoint)
    protocol = json.loads((tmp_path / "guarded-protocol.json").read_text(encoding="utf-8"))
    protocol["checkpoint_sha256"] = checkpoint_hash(checkpoint)
    protocol_ref = sealed(tmp_path / "guarded-protocol.json", protocol)
    with pytest.raises(ProvenanceError, match="actor/log/normalizer/bc_config"):
        load_guarded_bc(checkpoint, {"protocol": protocol_ref}, base=tmp_path)


def test_recovered_contract12_adapter_appends_only_bound_inactive_region_state(tmp_path):
    raw = recovered_native_state()
    checkpoint = tmp_path / "recovered.pt"
    torch.save(raw, checkpoint)
    protocol_ref = sealed(
        tmp_path / "recovered-adapter.json",
        {
            "schema": 1,
            "kind": "gate1_recovered_contract12_adapter_protocol",
            "allowed_checkpoints": {"initialization": checkpoint_hash(checkpoint)},
            "raw_missing_actor_keys": ["region_flat", "region_group.bias", "region_group.weight"],
            "appended_inactive_region_state": {
                "experts-false-token-true": inactive_region_state(
                    raw["core_width"],
                    experts=False,
                    token=True,
                )["identity"],
            },
            "appended_region_scope": "inactive-under-learned_movement=false",
            "historical_worker_contract_claim": False,
            "current_upgrade_allowed": False,
            "implicit_defaults": [],
        },
    )
    actor, normalizer = evaluate.load(
        checkpoint,
        strict=True,
        resolved_semantics=SEMANTICS,
        diagnostic_loading={
            "mode": "reviewed-adapter",
            "adapter": "recovered-contract12-inactive-region",
            "protocol": protocol_ref,
            "node_id": "initialization",
        },
    )
    assert all(torch.equal(actor.state_dict()[name], value) for name, value in raw["actor"].items())
    assert actor.state_dict()["region_flat"].item() is True
    assert normalizer.state_dict() == raw["normalizer"]
    assert evaluate.decoder_identity(actor) == "native-contract-12"
    raw["actor"].pop(next(iter(raw["actor"])))
    torch.save(raw, checkpoint)
    body = json.loads((tmp_path / "recovered-adapter.json").read_text(encoding="utf-8"))
    body["allowed_checkpoints"]["initialization"] = checkpoint_hash(checkpoint)
    protocol_ref = sealed(tmp_path / "recovered-adapter.json", body)
    with pytest.raises(ProvenanceError, match="pre-region source schema"):
        evaluate.load(
            checkpoint,
            strict=True,
            resolved_semantics=SEMANTICS,
            diagnostic_loading={
                "mode": "reviewed-adapter",
                "adapter": "recovered-contract12-inactive-region",
                "protocol": protocol_ref,
                "node_id": "initialization",
            },
        )


def test_seed_scan_fails_closed_on_prior_use_and_known_roles(tmp_path, monkeypatch):
    rl = tmp_path / "rl"
    rl.mkdir()
    (rl / "source.txt").write_text(
        "world seed 49999\nsha256=a709483e202e1f50006bad3b7a626317\npopulation=50010",
        encoding="utf-8",
    )
    monkeypatch.setattr(freeze, "_repo_files", lambda root: ["source.txt"])
    retrieval = tmp_path / "retrieval"
    retrieval.mkdir()
    (retrieval / "record.log").write_text("seed 7", encoding="utf-8")
    package = tmp_path / "package"
    package.mkdir()
    (package / "REPORT.md").write_text("no selected seed", encoding="utf-8")
    inventory = {
        "files": [
            {"path": "record.log", "size": 6, "sha256": checkpoint_hash(retrieval / "record.log")}
        ]
    }
    files, audit = _seed_scan(
        rl, retrieval, package, tmp_path / "out", list(range(50000, 50016)), inventory
    )
    assert files["scanned_file_count"] == 3
    assert {row["seed"] for row in files["decimal_token_candidates"]} == {50010}
    assert audit["status"] == "fresh-in-scanned-scope"
    (rl / "source.txt").write_text("world seed 50007", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="prior-use occurrences"):
        _seed_scan(rl, retrieval, package, tmp_path / "out", list(range(50000, 50016)), inventory)
    with pytest.raises(ProvenanceError, match="known train/dev"):
        _seed_scan(rl, retrieval, package, tmp_path / "out", list(range(3000, 3016)), inventory)


def test_rng_schedule_couples_initial_state_not_post_draw_streams():
    nodes = [
        {"id": "a", "historical_semantics": SEMANTICS},
        {"id": "b", "historical_semantics": {**SEMANTICS, "commitments": True}},
    ]
    worlds = [{"id": "s50000-default", "horizon": 3}]
    schedule = _rng_schedule(nodes, worlds)
    left, right = schedule["worlds"][0]["nodes"]
    assert [row["seed"] for row in left["steps"]] == [row["seed"] for row in right["steps"]]
    assert [row["state_sha256"] for row in left["steps"]] == [
        row["state_sha256"] for row in right["steps"]
    ]
    assert left["draw_order"] != right["draw_order"]
    assert left["post_draw_coupling_claim"] is False
    assert right["post_draw_coupling_claim"] is False


def _contract8_fixture_state() -> dict:
    from anthropocene_rl.autonomous_v7.legacy_contract8 import SETS, Contract8Actor

    actor = Contract8Actor(207)
    normalizer = {"core_mean": [0.0] * 207, "core_std": [1.0] * 207}
    for name, _, width in SETS:
        normalizer[name + "_mean"] = [0.0] * width
        normalizer[name + "_std"] = [1.0] * width
    return {
        "contract": 8,
        "core_width": 207,
        "privileged_width": 0,
        "actor": actor.state_dict(),
        "normalizer": normalizer,
    }


def test_contract8_comparator_uses_frozen_historical_source_not_upgrade(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    historical = tmp_path / "historical-source"
    historical.mkdir()
    commit = "2f4cc895e6bcd897200d340851d69be98fd18e00"
    for leaf in ("model.py", "contracts.py"):
        (historical / leaf).write_bytes(
            _git_bytes(root, commit, f"src/anthropocene_rl/autonomous_v7/{leaf}")
        )
    checkpoint = tmp_path / "prior.pt"
    torch.save(_contract8_fixture_state(), checkpoint)
    corpus_ref = sealed(tmp_path / "corpus.json", _corpus())
    protocol_ref = sealed(
        tmp_path / "comparator.json",
        {
            "schema": 1,
            "kind": "gate1_contract8_comparator_protocol",
            "checkpoint_sha256": checkpoint_hash(checkpoint),
            "corpus_sha256": corpus_ref["sha256"],
            "historical_source_root": historical.name,
            "historical_sources": {
                leaf: checkpoint_hash(historical / leaf) for leaf in ("model.py", "contracts.py")
            },
            "runtime": comparator_runtime_identity(),
            "mapped_outputs": [
                "strategic_mean",
                "strategic_log_std",
                "mix_logits",
                "mix_log_std",
                "service_mean",
                "service_log_std",
                "clothing",
                "shelter",
                "fission",
                "movement",
            ],
            "decoded_actions": [
                "strategic",
                "mix",
                "service",
                "clothing",
                "shelter",
                "fission",
                "movement",
            ],
        },
    )
    monkeypatch.setattr(
        evaluate, "upgrade", lambda value: (_ for _ in ()).throw(AssertionError("upgrade called"))
    )
    result = run_contract8_comparator(checkpoint, corpus_ref, protocol_ref, base=tmp_path)
    assert result["status"] == "PASS"
    (historical / "model.py").write_text("# changed", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="historical source bytes differ"):
        run_contract8_comparator(checkpoint, corpus_ref, protocol_ref, base=tmp_path)


def test_corpus_masks_and_shapes_are_complete_and_deterministic():
    first, second = _corpus(), _corpus()
    assert first == second
    assert len(first["cases"]) == 4
    assert [case["inputs"]["core"]["shape"][0] for case in first["cases"]] == [1, 1, 2, 3]
    for name in ("movement_mask", "clothing_route_mask", "shelter_route_mask", "economy_mix_mask"):
        assert not any(first["cases"][0]["inputs"][name]["values"][0])
        assert all(first["cases"][1]["inputs"][name]["values"][0])


def test_contract8_comparator_rejects_incomplete_source_inventory_and_empty_corpus(tmp_path):
    root = Path(__file__).resolve().parents[1]
    historical = tmp_path / "historical-source"
    historical.mkdir()
    commit = "2f4cc895e6bcd897200d340851d69be98fd18e00"
    for leaf in ("model.py", "contracts.py"):
        (historical / leaf).write_bytes(
            _git_bytes(
                root,
                commit,
                f"src/anthropocene_rl/autonomous_v7/{leaf}",
            )
        )
    checkpoint = tmp_path / "prior.pt"
    torch.save(_contract8_fixture_state(), checkpoint)
    corpus_ref = sealed(tmp_path / "corpus.json", _corpus())
    protocol = {
        "schema": 1,
        "kind": "gate1_contract8_comparator_protocol",
        "checkpoint_sha256": checkpoint_hash(checkpoint),
        "corpus_sha256": corpus_ref["sha256"],
        "historical_source_root": "historical-source",
        "historical_sources": {
            leaf: checkpoint_hash(historical / leaf) for leaf in ("model.py", "contracts.py")
        },
        "runtime": comparator_runtime_identity(),
        "mapped_outputs": [
            "strategic_mean",
            "strategic_log_std",
            "mix_logits",
            "mix_log_std",
            "service_mean",
            "service_log_std",
            "clothing",
            "shelter",
            "fission",
            "movement",
        ],
        "decoded_actions": [
            "strategic",
            "mix",
            "service",
            "clothing",
            "shelter",
            "fission",
            "movement",
        ],
    }
    protocol["historical_sources"].pop("contracts.py")
    protocol_ref = sealed(tmp_path / "protocol.json", protocol)
    with pytest.raises(ProvenanceError, match=r"model.py and contracts.py exactly"):
        run_contract8_comparator(checkpoint, corpus_ref, protocol_ref, base=tmp_path)
    protocol["historical_sources"]["contracts.py"] = checkpoint_hash(historical / "contracts.py")
    empty_ref = sealed(
        tmp_path / "empty.json",
        {
            "schema": 1,
            "kind": "gate1_contract8_corpus",
            "cases": [],
            "purpose": "hostile",
        },
    )
    protocol["corpus_sha256"] = empty_ref["sha256"]
    protocol_ref = sealed(tmp_path / "protocol.json", protocol)
    with pytest.raises(ProvenanceError, match="at least four"):
        run_contract8_comparator(checkpoint, empty_ref, protocol_ref, base=tmp_path)


def test_fresh_common_producer_context_rejects_historical_equivalence_claim(tmp_path):
    node = {"historical_producer_status": "unresolved"}
    pane = {
        "kind": "fresh-common-producer-diagnostic",
        "semantic_comparator_protocol": {"path": "forged", "sha256": "0" * 64},
        "semantic_comparator": None,
    }
    with pytest.raises(ProvenanceError, match="preserve unresolved historical producer"):
        initialization_lineage._fresh_common_producer_scope(node, pane)
    pane["semantic_comparator_protocol"] = None
    initialization_lineage._fresh_common_producer_scope(node, pane)


def test_production_runner_normalizes_refs_and_reaches_bc_strict_load(tmp_path, monkeypatch):
    retrieval = tmp_path / "retrieval"
    retrieval.mkdir()
    freeze_root = tmp_path / "freeze"
    freeze_root.mkdir()
    support = recovered_native_state()
    bc_state = {"actor": copy.deepcopy(support["actor"]), "log": []}
    bc_path = retrieval / "stage1-token-s6.bc.pt"
    support_path = retrieval / "stage1-token-s6.pt"
    torch.save(bc_state, bc_path)
    torch.save(support, support_path)
    protocol = bc_protocol(bc_state, support)
    protocol.update(
        {
            "bc_checkpoint_sha256": checkpoint_hash(bc_path),
            "support_checkpoint_sha256": checkpoint_hash(support_path),
        }
    )
    protocol_ref = sealed(freeze_root / "bc-protocol.json", protocol)
    protocol_ref["size"] = (freeze_root / "bc-protocol.json").stat().st_size
    semantics = copy.deepcopy(SEMANTICS)
    dag_ref = sealed(
        freeze_root / "dag.json",
        {
            "schema": 1,
            "kind": "gate1_artifact_dag_proposal",
            "nodes": [
                {
                    "id": "bc-phase",
                    "historical_semantics": semantics,
                    "artifact": {
                        "root_id": "t3",
                        "path": bc_path.name,
                        "sha256": checkpoint_hash(bc_path),
                        "size": bc_path.stat().st_size,
                    },
                },
                {
                    "id": "t3-initialization-clone",
                    "historical_semantics": semantics,
                    "artifact": {
                        "root_id": "t3",
                        "path": support_path.name,
                        "sha256": checkpoint_hash(support_path),
                        "size": support_path.stat().st_size,
                    },
                },
            ],
        },
    )
    dag_ref["size"] = (freeze_root / "dag.json").stat().st_size
    world = {
        "id": "fresh-world",
        "seed": 50000,
        "withheld": 0,
        "turn": None,
        "reset_turn": None,
        "reset_turn_policy": "bind-native-post-reset-boundary",
        "players": 8,
        "warmup": 0,
        "horizon": 1,
        "cohort_digest": "frozen",
        "snapshot": {"kind": "none", "sha256": None},
    }
    evaluation_ref = sealed(
        freeze_root / "evaluation.json",
        {
            "schema": 1,
            "kind": "gate1_evaluation_claim_protocol_proposal",
            "panes": [
                {
                    "id": "fresh-deterministic",
                    "kind": "fresh-common-producer-diagnostic",
                    "action_mode": "deterministic",
                    "nodes": ["bc-phase"],
                }
            ],
            "cohort": {"worlds": [world]},
            "capture_bounds": {
                "max_steps": 1,
                "max_villages_per_boundary": 8,
                "max_economic_rows_per_step": 8,
                "max_canonical_bytes": 100_000,
            },
            "personality": [0.2] * 5,
            "mandatory_metrics": list(freeze.MANDATORY_METRICS),
        },
    )
    evaluation_ref["size"] = (freeze_root / "evaluation.json").stat().st_size
    candidate = {
        "digest": "candidate",
        "inputs": {"retrieval_root": str(retrieval)},
        "proposals": {
            "artifact_dag": dag_ref,
            "evaluation": evaluation_ref,
            "bc_phase_adapter": protocol_ref,
        },
    }
    worker_path = tmp_path / "worker.json"
    worker_path.write_text("worker", encoding="utf-8")
    monkeypatch.setattr(
        freeze,
        "_load_reviewed_candidate",
        lambda candidate_path, review_path: (candidate, freeze_root),
    )
    monkeypatch.setattr(
        freeze,
        "_worker_capture",
        lambda path, value: {
            "worker_contract": {"synthetic": True},
            "observation_core_width": 4,
        },
    )
    loaded = {}

    def fake_rollout(*args, **kwargs):
        actor, normalizer = evaluate.load(
            Path(args[2]),
            strict=kwargs["strict_checkpoint"],
            resolved_semantics=kwargs["resolved_semantics"],
            diagnostic_loading=kwargs["diagnostic_loading"],
        )
        loaded["actor"] = tensor_state_identity(actor.state_dict())
        loaded["normalizer"] = normalizer.state_dict()
        loaded["world_contract"] = args[14]
        loaded["expected_reset_turn"] = kwargs["expected_reset_turn"]
        loaded["bind_observed_reset_turn"] = kwargs["bind_observed_reset_turn"]
        return {"loaded": True}

    monkeypatch.setattr(evaluate, "rollout", fake_rollout)
    result = evaluate_reviewed_node(
        tmp_path / "candidate.json",
        tmp_path / "review.json",
        worker_path,
        tmp_path / "unused-contract8.json",
        tmp_path / "unused-guarded.json",
        "bc-phase",
        "fresh-deterministic",
        tmp_path / "result.json",
    )
    assert result["producer_scope"] == "fresh-common-producer-diagnostic"
    assert result["historical_producer_status"] == "unresolved"
    assert loaded["actor"] == tensor_state_identity(
        assemble_bc_phase_state(bc_state, support, protocol)["actor"]
    )
    assert loaded["normalizer"] == support["normalizer"]
    assert loaded["world_contract"]["turn"] is None
    assert loaded["expected_reset_turn"] is None
    assert loaded["bind_observed_reset_turn"] is True


@pytest.mark.parametrize("native_turn", [8, 9, 11])
def test_fresh_generated_world_binds_each_native_reset_turn(native_turn):
    from anthropocene_rl.autonomous_v7.worker_contract import world

    descriptor = world(50000, turn=None, players=8, movement=False, head=None)
    observed = evaluate._validated_reset_turn(
        {"turn": native_turn},
        None,
        bind_observed_reset_turn=True,
        world_contract=descriptor,
        snapshot=None,
    )
    assert observed == native_turn


def test_generated_cohort_declares_observed_native_reset_boundary():
    worlds, cohort_digest = freeze._worlds([50000, 50001])
    assert len(worlds) == 2 * len(freeze.MASKS)
    assert {row["cohort_digest"] for row in worlds} == {cohort_digest}
    assert all(row["turn"] is None and row["reset_turn"] is None for row in worlds)
    assert {row["reset_turn_policy"] for row in worlds} == {"bind-native-post-reset-boundary"}


def test_observed_reset_turn_binding_rejects_snapshot_and_materialized_descriptor():
    from anthropocene_rl.autonomous_v7.worker_contract import world

    descriptor = world(50000, turn=9, players=8, movement=False, head=None)
    with pytest.raises(ValueError, match="only valid for a fresh generated world"):
        evaluate._validated_reset_turn(
            {"turn": 9},
            None,
            bind_observed_reset_turn=True,
            world_contract=descriptor,
            snapshot=b"snapshot",
        )


def test_explicit_reset_turn_still_rejects_wrong_native_boundary():
    from anthropocene_rl.autonomous_v7.worker_contract import world

    descriptor = world(50000, turn=None, players=8, movement=False, head=None)
    with pytest.raises(ValueError, match="differs from the reviewed world boundary"):
        evaluate._validated_reset_turn(
            {"turn": 8},
            9,
            bind_observed_reset_turn=False,
            world_contract=descriptor,
            snapshot=None,
        )


@pytest.mark.parametrize(
    ("reviewer_identity", "reviewer_role"),
    [
        ("OpenAI gpt-6-astra/high", "independent-architecture-reviewer"),
        ("user-designated-root-reviewer", "user-designated-root-reviewer"),
    ],
)
def test_review_accepts_exact_authorized_identity_and_role(
    tmp_path, reviewer_identity, reviewer_role
):
    candidate = {"digest": "candidate-digest"}
    review = {
        "schema": 1,
        "kind": "independent_gate1_freeze_review",
        "reviewer_identity": reviewer_identity,
        "reviewer_role": reviewer_role,
        "decision": "approved",
        "candidate_digest": candidate["digest"],
        "production_gate1": "HOLD",
    }
    review["digest"] = digest(review)
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    freeze._review(candidate, path)


@pytest.mark.parametrize(
    ("reviewer_identity", "reviewer_role", "candidate_digest"),
    [
        ("user-designated-root-reviewer", None, "candidate-digest"),
        ("unapproved-reviewer", "user-designated-root-reviewer", "candidate-digest"),
        ("user-designated-root-reviewer", "user-designated-root-reviewer", "stale"),
    ],
)
def test_review_rejects_missing_or_unapproved_role_and_stale_digest(
    tmp_path, reviewer_identity, reviewer_role, candidate_digest
):
    candidate = {"digest": "candidate-digest"}
    review = {
        "schema": 1,
        "kind": "independent_gate1_freeze_review",
        "reviewer_identity": reviewer_identity,
        "decision": "approved",
        "candidate_digest": candidate_digest,
        "production_gate1": "HOLD",
    }
    if reviewer_role is not None:
        review["reviewer_role"] = reviewer_role
    review["digest"] = digest(review)
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="does not approve this exact freeze candidate"):
        freeze._review(candidate, path)


def test_fresh_world_null_metadata_turn_does_not_weaken_post_reset_turn_check():
    from anthropocene_rl.autonomous_v7.worker_contract import world

    descriptor = world(19, turn=None, players=8, movement=False, head=None)
    bounds = {
        "max_steps": 1,
        "max_villages_per_boundary": 4096,
        "max_economic_rows_per_step": 4096,
        "max_canonical_bytes": 256 * 1024 * 1024,
    }
    with pytest.raises(ValueError, match="bridge reset turn differs"):
        evaluate.rollout(
            19,
            1,
            None,
            world_contract=descriptor,
            requested_metrics=["welfare"],
            service_fields=service_outcome_registry(),
            capture_bounds=bounds,
            evaluation_world_id="fresh-reset-turn-negative",
            expected_reset_turn=8,
            observation_core_width=1,
            evaluation_semantics=SEMANTICS,
        )


def test_execution_identity_rehashes_sources_runtime_and_bridge_dependencies(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    freeze_root = tmp_path / "freeze"
    root.mkdir()
    freeze_root.mkdir()
    source_rows = []
    for index, relative in enumerate(freeze.EXECUTION_SOURCES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}", encoding="utf-8")
        source_rows.append(
            {"path": relative, "sha256": checkpoint_hash(path), "size": path.stat().st_size}
        )
    identity_ref = sealed(
        freeze_root / "identity.json",
        {
            "schema": 1,
            "kind": "gate1_freeze_implementation_identity",
            "repository_root": str(root),
            "execution_source_files": source_rows,
        },
    )
    runtime = {"runtime": "frozen"}
    verification_ref = sealed(
        freeze_root / "verification.json",
        {
            "schema": 1,
            "kind": "gate1_verification_protocol_proposal",
            "runtime": runtime,
        },
    )
    dependencies = []
    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    for name in (
        "Anthropocene.Model9Bridge.dll",
        "Anthropocene.Model9Bridge.deps.json",
        "Anthropocene.Model9Bridge.runtimeconfig.json",
    ):
        path = bridge_dir / name
        path.write_text(name, encoding="utf-8")
        dependencies.append(
            {"path": str(path), "sha256": checkpoint_hash(path), "size": path.stat().st_size}
        )
    candidate = {
        "proposals": {"implementation_identity": identity_ref, "verification": verification_ref},
        "inputs": {"current_bridge": dependencies[0], "current_bridge_dependencies": dependencies},
    }
    monkeypatch.setattr(freeze, "_runtime", lambda: runtime)
    monkeypatch.setattr(
        freeze,
        "__file__",
        str(root / "src/anthropocene_rl/autonomous_v7/gate1_freeze_candidate.py"),
    )
    _verify_execution_identity(candidate, freeze_root)
    changed = root / freeze.EXECUTION_SOURCES[0]
    changed.write_text("changed", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="executing source changed"):
        _verify_execution_identity(candidate, freeze_root)
