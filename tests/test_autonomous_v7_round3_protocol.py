"""Adversarial qualification-contract tests; fixture sources are synthetic, never production data."""

import copy
import json
import types
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import (
    teacher_data as td,
    protocol,
    lineage,
    promotion,
    curriculum,
    formal_report,
)
from anthropocene_rl.autonomous_v7 import calibration as cal, provenance as prov, data, ppo
from test_autonomous_v7 import batch


def native_source(seed=1000, withheld=0):
    schema = types.SimpleNamespace(
        build_hash="c" * 32,
        simulator_commit="b" * 40,
        simulator_source_clean=True,
        core_width=188,
        core_features=tuple("f" + str(i) for i in range(188)),
    )
    options = {
        "worlds": [{"seed": seed, "withheld": withheld, "warmup": 0}],
        "players": 8,
        "turns": 2,
        "movement": True,
        "raw_materials": False,
        "opening_limit": 40,
    }
    return td.source_record(options, schema, {"sha256": "d" * 64}, learner="a" * 40)


def synthetic_worker():
    from anthropocene_rl.autonomous_v7.worker_contract import seal

    return seal(
        {
            "version": 1,
            "simulator": native_source()["simulator"],
            "bridge_sha256": "d" * 64,
            "bridge_build_hash": "e" * 32,
            "flags": {"sim_rules": "", "sim_statics": ""},
            "schemas": {
                "contract": 12,
                "action": ppo.contracts.ACTION_SCHEMA,
                "observation": ppo.contracts.OBSERVATION_SCHEMA,
                "environment": ppo.contracts.ENVIRONMENT_SCHEMA,
                "head": 1,
                "envelope": 2,
                "raw": 2,
            },
            "feature_flags": {"SimulationConfig.ElderRawMaterialsEnabled": False},
            "configuration": {"sha256": "f" * 64, "players": 8},
        }
    )


@lru_cache(maxsize=None)
def native_proof(seed=1000):
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="synthetic-teacher-", dir="work"))
    path = publish(root, seed)
    proof = {}
    data.load([path], strict=True, report=proof)
    return proof


def root_state(seed=0):
    protocol_doc = {
        "schema": 1,
        "kind": "round3_preregistration",
        "protocol_digest_version": 1,
        "seed_roles": protocol.ROLES,
        "hypotheses": ["synthetic fixture; not production evidence"],
    }
    protocol_doc["digest"] = protocol.protocol_digest(protocol_doc)
    held = cal.manifest(
        {
            "protocol_digest": protocol_doc["digest"],
            "audit_registry": str(Path("work/synthetic-audits").resolve()),
            "calibration_seeds": list(range(10000, 10100)),
            "training_seeds": [1000],
            "selection_seeds": [2100, *range(2160, 2180)],
        }
    )
    inputs = {"training": native_proof(), "selection": native_proof(2100)}
    config = {"options": {"seed": seed}, "sim_rules": "", "sim_statics": ""}
    run = {
        **config,
        "config_digest": prov.digest(config),
        "code": "a" * 40,
        "simulator": native_source()["simulator"],
        "bridge_sha256": "d" * 64,
        "contract": 12,
        "utility_version": 2,
        "calibration_manifest_digest": held["manifest_digest"],
        "initialization": "random",
        "inputs": {},
        "teacher_inputs": {k: v["digest"] for k, v in inputs.items()},
    }
    run["protocol_digest"] = protocol_doc["digest"]
    run["worker_contract"] = synthetic_worker()
    state = {
        "seed": seed,
        "contract": 12,
        "region_head": "hierarchical",
        "token": True,
        "actor": {"test_weight": torch.tensor([1.0, 2.0])},
        "vector_critic": True,
        "critic": {
            f"heads.{name}.{parameter}": torch.tensor([3.0])
            for name in ppo.contracts.OBJECTIVES
            for parameter in ("weight", "bias")
        },
        "normalizer": {"core_mean": np.array([2.0]), "guard": 10.0},
        "run_manifest": run,
        "calibration_manifest": held,
        "family_lineage": lineage.build(
            held, lineage.families([1000]), lineage.families([2100]), teacher_inputs=inputs
        ),
    }
    plan = {
        "schema": 1,
        "projection_version": 1,
        "kind": "guarded_clone_cohort",
        "protocol_digest": protocol_doc["digest"],
        "cohort_digest": prov.digest(promotion.cohort_config(state)),
    }
    path = Path("work") / ("synthetic-clone-cohort-" + plan["cohort_digest"] + ".json")
    if not path.exists():
        path.write_text(json.dumps(plan, sort_keys=True))
    state["run_manifest"]["clone_protocol"] = {
        "path": str(path.resolve()),
        "sha256": prov.checkpoint_hash(path),
    }
    protocol_doc["clone_protocol_sha256"] = prov.checkpoint_hash(path)
    prereg = path.with_name(path.stem + "-prereg.json")
    prereg_ref = json_ref(prereg, protocol_doc)
    approval = {
        "schema": 1,
        "kind": "protocol_approval",
        "reviewer": "Astra",
        "decision": "APPROVE",
        "protocol_digest": protocol_doc["digest"],
        "preregistration_sha256": prereg_ref["sha256"],
    }
    approval_ref = json_ref(path.with_name(path.stem + "-approval.json"), approval)
    state["run_manifest"]["protocol_evidence"] = {
        "preregistration": prereg_ref,
        "approval": approval_ref,
    }
    return held, promotion.seal(state)


def arrays():
    out = batch(2)
    from anthropocene_rl.autonomous_v7.branch_shards import _dummy_labels

    out.update(_dummy_labels(2))
    out.update(
        camp=np.array([7, 7], np.int32),
        turn=np.array([10, 11], np.int32),
        society=np.array([3, 3]),
        parent_society=np.zeros(2, np.int32),
        prior_population=np.full(2, 100),
        population=np.full(2, 100),
        qol=np.full(2, 0.8),
        nutrition=np.full(2, 1.0),
        years=np.full(2, 20.0),
        welfare=np.full(2, 1600.0),
        terminated=np.array([False, True]),
    )
    return out


def publish(tmp_path, seed=1000, withheld=0):
    source = native_source(seed, withheld)
    meta = {
        "seed": seed,
        "withheld": withheld,
        "warmup": 0,
        "opening_turns": 10,
        "first_turn": 10,
        "players": 8,
        "turns": 2,
        "initial_fingerprint": "e" * 64,
        "configuration_hash": "f" * 64,
        "initial_snapshot_sha256": "9" * 64,
        "source_run": source["source_run"],
    }
    path = tmp_path / ("world-" + td.trajectory(meta))
    record = td.write_shard(path, arrays(), meta, source)
    td.dataset(tmp_path, [record], source)
    return path


def test_teacher_dataset_validates_and_rebases_same_seed_different_masks(tmp_path):
    one, two = publish(tmp_path / "one"), publish(tmp_path / "two", withheld=12)
    declared = {native_source()["source_run"], native_source(withheld=12)["source_run"]}
    report = {}
    rows = data.load([one, two], strict=True, report=report, declared_sources=declared)
    assert len(set(rows["world"].tolist())) == 2 and set(rows["source_seed"]) == {1000}
    assert lineage.teacher_proof(report)
    with pytest.raises(prov.ProvenanceError, match="duplicate teacher"):
        data.load([one, one], strict=True)


@pytest.mark.parametrize(
    "mutation", ["array", "shape", "meta", "unpublished", "source", "dataset", "legacy"]
)
def test_teacher_source_payload_and_publish_last_fail_closed(tmp_path, mutation):
    path = publish(tmp_path)
    if mutation == "array":
        np.save(path / "welfare.npy", np.array([1.0, 2.0]))
    elif mutation == "shape":
        np.save(path / "strategic.npy", np.zeros((2, 4)))
    elif mutation == "meta":
        obj = json.loads((path / "meta.json").read_text())
        obj["seed"] = 1001
        (path / "meta.json").write_text(json.dumps(obj))
    elif mutation == "unpublished":
        (path / "meta.json").rename(path / "meta.json.partial")
    elif mutation == "source":
        obj = json.loads((path / "meta.json").read_text())
        obj["source_record"]["learner_commit"] += "+dirty"
        (path / "meta.json").write_text(json.dumps(obj))
    elif mutation == "dataset":
        (tmp_path / "dataset.json").write_text("{}")
    else:
        (path / "meta.json").write_text(json.dumps({"seed": 1000, "contract": 12}))
    with pytest.raises((prov.ProvenanceError, KeyError, FileNotFoundError)):
        data.load([path], strict=True)


def test_teacher_collision_and_unattributed_root_never_verify(tmp_path):
    path = publish(tmp_path)
    with pytest.raises(prov.ProvenanceError, match="already exists"):
        td.write_shard(path, arrays(), {}, native_source())
    held, state = root_state()
    assert lineage.checkpoint_evidence(state, held)["verified"]
    state["family_lineage"] = lineage.build(
        held, lineage.families([1000]), lineage.families([2100])
    )
    assert not lineage.checkpoint_evidence(state, held)["verified"]


def test_clone_selection_rejects_spoofed_actual_worker_producer(tmp_path, monkeypatch):
    from anthropocene_rl.autonomous_v7 import clone_evaluate, bridge

    held, state = root_state()
    path = tmp_path / "candidate.pt"
    torch.save(state, path)

    class Probe:
        schema = object()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(bridge, "Model9Bridge", Probe)
    monkeypatch.setattr(
        clone_evaluate, "simulator_identity", lambda *a, **k: state["run_manifest"]["simulator"]
    )
    monkeypatch.setattr(clone_evaluate, "binary_identity", lambda: {"sha256": "d" * 64})
    monkeypatch.setenv("ANTHROPOCENE_SIM_RULES", "")
    monkeypatch.setenv("ANTHROPOCENE_SIM_STATICS", "")
    monkeypatch.setattr(
        clone_evaluate,
        "rollout",
        lambda *a, **k: {"welfare": 1.0, "producer": {"bridge_sha256": "f" * 64}},
    )
    with pytest.raises(prov.ProvenanceError, match="worker"):
        clone_evaluate.evaluate(path, held)


@pytest.mark.parametrize("mutation", ["binary", "flags", "raw", "missing"])
def test_registered_search_requires_actual_same_worker_producers(mutation):
    from anthropocene_rl.autonomous_v7.search_teacher import validate_producers

    expected = {
        "simulator": native_source()["simulator"],
        "bridge_sha256": "d" * 64,
        "flags": {"sim_rules": "", "sim_statics": ""},
        "raw_materials": False,
    }
    rows = [{"producer": copy.deepcopy(expected)} for _ in range(2)]
    validate_producers(rows, expected)
    if mutation == "binary":
        rows[1]["producer"]["bridge_sha256"] = "f" * 64
    elif mutation == "flags":
        rows[1]["producer"]["flags"]["sim_rules"] = "changed"
    elif mutation == "raw":
        rows[1]["producer"]["raw_materials"] = True
    else:
        rows[1].clear()
    with pytest.raises(ValueError, match="workers"):
        validate_producers(rows, expected)


def selection_report(state, sha, ratio=1.0):
    run = state["run_manifest"]
    return {
        "schema": 1,
        "kind": "clone_evaluation",
        "checkpoint_sha256": sha,
        "model_semantics": state["model_semantics"],
        "movement": True,
        "region_head": "hierarchical",
        "simulator": run["simulator"],
        "bridge_sha256": run["bridge_sha256"],
        "flags": {k: run[k] for k in ("sim_rules", "sim_statics")},
        "ratio": ratio,
        "worlds": [
            {
                "seed": s,
                "withheld": 0,
                "warmup": 0,
                "snapshot": "",
                "policy_welfare": ratio,
                "heuristic_welfare": 1.0,
            }
            for s in range(2160, 2180)
        ],
    }


def promotion_inputs(tmp_path):
    inputs = []
    for seed in range(10):
        held, state = root_state(seed)
        path = tmp_path / f"clone-{seed}.pt"
        torch.save(state, path)
        report = tmp_path / f"report-{seed}.json"
        report.write_text(json.dumps(selection_report(state, prov.checkpoint_hash(path))))
        inputs.append(
            {
                "checkpoint": {"path": str(path), "sha256": prov.checkpoint_hash(path)},
                "report": {"path": str(report), "sha256": prov.checkpoint_hash(report)},
            }
        )
    return held, inputs


def test_promotion_tie_is_deterministic_preserves_rng_tensors_and_all_losers(tmp_path):
    held, inputs = promotion_inputs(tmp_path)
    rng = torch.get_rng_state().clone()
    result = promotion.promote(list(reversed(inputs)), held)
    assert torch.equal(rng, torch.get_rng_state())
    assert result["promotion"]["selected"] == inputs[0]["checkpoint"]["sha256"]
    assert len(result["family_lineage"]["parents"]) == 10
    assert set(range(2160, 2180)) <= set(result["family_lineage"]["selection"]["seeds"])
    winner = torch.load(inputs[0]["checkpoint"]["path"], weights_only=False)
    assert promotion.semantic_hash(result) == promotion.semantic_hash(winner)
    assert lineage.checkpoint_evidence(result, held)["verified"]
    from anthropocene_rl.autonomous_v7.checkpoint import restore_actor

    with pytest.raises(ValueError, match="non-resumable"):
        restore_actor(None, result, resume=True)


def test_promoted_parent_is_content_backed_and_remains_nonresumable_in_descendants(tmp_path):
    held, inputs = promotion_inputs(tmp_path)
    promoted = promotion.promote(inputs, held)
    path = tmp_path / "promoted.pt"
    torch.save(promoted, path)
    ref = {"path": str(path), "sha256": prov.checkpoint_hash(path)}
    _, child = root_state()
    child["run_manifest"] = copy.deepcopy(child["run_manifest"])
    child["run_manifest"].pop("clone_protocol", None)
    child["run_manifest"].update(initialization="checkpoint", inputs={"init": ref["sha256"]})
    child["family_lineage"] = lineage.build(
        held, lineage.families([1001]), lineage.families([2060]), [(ref["sha256"], promoted, ref)]
    )
    assert child["family_lineage"]["parents"][0]["proof"]["resumable"] is False
    assert lineage.checkpoint_evidence(child, held)["verified"]
    forged = copy.deepcopy(child)
    forged["family_lineage"]["parents"][0]["proof"]["resumable"] = True
    forged["family_lineage"]["digest"] = prov.digest(
        {k: v for k, v in forged["family_lineage"].items() if k != "digest"}
    )
    assert not lineage.checkpoint_evidence(forged, held)["verified"]


def test_teacher_manifest_and_clone_config_cannot_be_self_attested(tmp_path):
    forged = copy.deepcopy(native_proof())
    forged["datasets"][0]["shards"].clear()
    forged["datasets"][0]["dataset_digest"] = prov.digest(
        {k: v for k, v in forged["datasets"][0].items() if k != "dataset_digest"}
    )
    forged["digest"] = prov.digest({k: v for k, v in forged.items() if k != "digest"})
    assert not lineage.teacher_proof(forged)
    held, inputs = promotion_inputs(tmp_path)
    path = Path(inputs[-1]["checkpoint"]["path"])
    state = torch.load(path, weights_only=False)
    state["run_manifest"]["options"]["lr"] = 0.9
    state["run_manifest"]["config_digest"] = prov.digest(
        {"options": state["run_manifest"]["options"], "sim_rules": "", "sim_statics": ""}
    )
    torch.save(state, path)
    inputs[-1]["checkpoint"]["sha256"] = prov.checkpoint_hash(path)
    with pytest.raises(prov.ProvenanceError, match="pre-frozen"):
        promotion.promote(inputs, held)


def test_promoted_init_gets_five_critic_only_iterations_without_fresh_heads():
    assert ppo.warmup_iterations(False, 5, 0, promoted_init=True) == 5
    assert ppo.warmup_iterations(False, 5, 0) == 0


def test_actual_ppo_entrypoint_resumes_effective_promoted_warmup_without_repeating(
    tmp_path, monkeypatch
):
    import sys
    from anthropocene_rl.autonomous_v7 import checkpoint, bridge as bridge_module

    monkeypatch.setenv("ANTHROPOCENE_REGION_HEAD", "hierarchical")
    held, inputs = promotion_inputs(tmp_path)
    promoted = promotion.promote(inputs, held)
    promoted.update(core_width=188, privileged_width=0)
    promotion.seal(promoted)
    init = tmp_path / "init.pt"
    torch.save(promoted, init)

    class Net(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.region_head = "hierarchical"
            self.region_group = torch.nn.Linear(1, 1)

        def set_region_head(self, value):
            self.region_head = value

        def load_state_dict(self, *args, **kwargs):
            return [], []

    class Vector(Net):
        pass

    for name in ("Actor", "Critic", "CostCritic"):
        monkeypatch.setattr(ppo, name, Net)
    monkeypatch.setattr(ppo, "VectorCritic", Vector)
    monkeypatch.setattr(ppo.Normalizer, "from_state", lambda state: None)
    monkeypatch.setattr(checkpoint, "restore_actor", lambda actor, state, *a, **kw: ([], []))

    class Probe:
        schema = types.SimpleNamespace(
            build_hash="c" * 32, simulator_commit="b" * 40, simulator_source_clean=True
        )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(bridge_module, "Model9Bridge", Probe)
    from anthropocene_rl.autonomous_v7 import worker_contract

    monkeypatch.setattr(worker_contract, "observe", lambda bridge: synthetic_worker())
    monkeypatch.setattr(prov, "code_commit", lambda: "a" * 40)
    calls = []

    def iteration(*args, **kwargs):
        calls.append(kwargs["train_actor"])
        return 0.0, {}

    monkeypatch.setattr(ppo, "iteration", iteration)
    local = {
        "schema": 1,
        "execution": {"tier": "local", "purpose": "stability", "arms": {"diagnostic": [0]}},
    }
    local["digest"] = prov.digest(local)
    local_path = tmp_path / "local.json"
    local_path.write_text(json.dumps(local))
    argv = [
        "ppo",
        "--init",
        str(init),
        "--out",
        str(tmp_path / "probe"),
        "--stage",
        "mixed",
        "--iterations",
        "0",
        "--worlds",
        "1",
        "--turns",
        "1",
        "--curriculum",
        "legacy",
        "--vector-critic",
        "--region-head",
        "hierarchical",
        "--nonqualifying",
        "--local-execution-design",
        str(local_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert ppo.main() == 0 and calls == [False] * 5
    saved = torch.load(tmp_path / "probe" / "resume.pt", weights_only=False)
    assert saved["run_manifest"]["effective_iterations"] == {
        "critic_only": 5,
        "actor": 0,
        "total": 5,
    }
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    assert ppo.main() == 0 and len(calls) == 5


@pytest.mark.parametrize(
    "gate_name",
    ["protocol", "source", "clone", "search", "calibration", "movement-integration", "submission"],
)
def test_gate_status_pass_shell_never_qualifies(gate_name, tmp_path):
    from anthropocene_rl.autonomous_v7.gate_validation import gate

    with pytest.raises(prov.ProvenanceError, match="semantic"):
        gate(
            gate_name,
            {"status": "pass", "outputs": [{"path": "arbitrary", "sha256": "f" * 64}]},
            {},
            {},
            {},
            tmp_path,
            {},
        )


def test_live_sweep_producers_are_not_copied_from_spoofed_checkpoint_metadata():
    from anthropocene_rl.autonomous_v7.personality_sweep import validate_producers

    _, state = root_state()
    run = state["run_manifest"]
    producer = {
        "simulator": run["simulator"],
        "bridge_sha256": run["bridge_sha256"],
        "flags": {"sim_rules": "", "sim_statics": ""},
        "raw_materials": False,
    }
    assert validate_producers([{"producer": producer}], state) == producer
    spoofed = copy.deepcopy(state)
    spoofed["run_manifest"]["simulator"]["commit"] = "9" * 40
    with pytest.raises(prov.ProvenanceError, match="live sweep"):
        validate_producers([{"producer": producer}], spoofed)
    with pytest.raises(prov.ProvenanceError):
        validate_producers([{}], state)


def test_calibration_claim_is_protocol_candidate_scoped_not_output_path(tmp_path):
    from anthropocene_rl.autonomous_v7.calibration_audit import claim

    held = {"protocol_digest": "a" * 64, "audit_registry": str((tmp_path / "central").resolve())}
    binding = {
        "protocol_digest": held["protocol_digest"],
        "checkpoint_sha256": "b" * 64,
        "cohort_digest": "c" * 64,
        "payload_inventory_digest": "d" * 64,
    }
    first = claim(held, binding)
    assert Path(first["path"]).parent == Path(held["audit_registry"])
    for repeated in (binding, dict(binding, payload_inventory_digest="e" * 64)):
        with pytest.raises(prov.ProvenanceError, match="already claimed"):
            claim(held, repeated)


def test_calibration_inventory_rejects_subset_and_other_checkpoint_producer(tmp_path):
    from anthropocene_rl.autonomous_v7.calibration_audit import freeze, inventory

    held = cal.manifest(
        {
            "protocol_digest": "a" * 64,
            "calibration_seeds": [10000, 10001],
            "training_seeds": [1000],
            "selection_seeds": [2100],
            "waivers": [],
        }
    )
    cohort = {
        "schema": 1,
        "role": "calibration",
        "worlds": [
            {"seed": s, "withheld": 0, "warmup": 0, "bank_entry": -1, "snapshot_sha256": ""}
            for s in (10000, 10001)
        ],
    }
    cohort["digest"] = prov.digest(cohort)
    roots = []
    for i, seed in enumerate((10000, 10001)):
        directory = tmp_path / str(seed)
        directory.mkdir()
        roots.append(directory)
        np.save(directory / "calibration.npy", np.ones(2, np.int8))
        meta = {
            "seed": seed,
            "withheld": 0,
            "warmup": 0,
            "bank_entry": -1,
            "snapshot_hash": None,
            "trajectory_id": str(i) * 64,
            "heldout_calibration": True,
            "calibration": True,
            "source_record": {
                "source_checkpoint": "b" * 64,
                "search_config": {"worlds_manifest": cohort},
            },
            "calibration_manifest_digest": held["manifest_digest"],
            "data_digest": prov.digest(
                {"calibration.npy": prov.checkpoint_hash(directory / "calibration.npy")}
            ),
        }
        (directory / "meta.json").write_text(json.dumps(meta))
    frozen = freeze(roots, "b" * 64, held, cohort)
    assert inventory(roots, "b" * 64, held, frozen)["cohort_digest"] == cohort["digest"]
    with pytest.raises(prov.ProvenanceError, match="subset"):
        inventory(roots[:1], "b" * 64, held, frozen)
    path = roots[0] / "meta.json"
    meta = json.loads(path.read_text())
    meta["source_record"]["source_checkpoint"] = "c" * 64
    path.write_text(json.dumps(meta))
    with pytest.raises(prov.ProvenanceError, match="producer"):
        inventory(roots, "b" * 64, held, frozen)


def test_queue_receipt_requires_actual_done_status_and_complete_content_marker(tmp_path):
    from anthropocene_rl.autonomous_v7.qualification import queue_receipt

    expected = {"learner_commit": "a" * 40, "simulator": {"commit": "b" * 40}}
    job = {
        "name": "qual",
        "rl_sha": "a" * 40,
        "sim_sha": "b" * 40,
        "phases": [
            {
                "name": "calibration",
                "run": "python -m evidence --seed 10000",
                "done": "calibration.done",
            }
        ],
    }
    job_ref = json_ref(tmp_path / "job.json", job)
    status = {"name": "qual", "phases": [{"name": "calibration", "state": "done", "exit_code": 0}]}
    status_ref = json_ref(tmp_path / "status.json", status)
    output = json_ref(tmp_path / "actual.json", {"actual": "report"})
    done = {
        **expected,
        "kind": "qualification_phase_done",
        "job_sha256": job_ref["sha256"],
        "phase": "calibration",
        "outputs": [output],
    }
    done_ref = json_ref(tmp_path / "calibration.done", done)
    body = {
        "outputs": [output],
        "queue": {"job": job_ref, "status": status_ref, "phase": "calibration", "done": done_ref},
    }
    assert queue_receipt(body, expected, tmp_path) == job_ref["sha256"]
    body["queue"]["done"] = json_ref(tmp_path / "calibration.done", {"status": "pass"})
    with pytest.raises(prov.ProvenanceError, match="done marker"):
        queue_receipt(body, expected, tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "report", "model", "loser", "worlds"])
def test_promotion_validates_every_candidate_and_complete_reports(tmp_path, mutation):
    held, inputs = promotion_inputs(tmp_path)
    if mutation == "missing":
        inputs.pop()
    elif mutation == "duplicate":
        inputs[-1] = inputs[0]
    elif mutation in ("report", "worlds"):
        path = Path(inputs[-1]["report"]["path"])
        obj = json.loads(path.read_text())
        if mutation == "report":
            obj["ratio"] = 1.1
        else:
            obj["worlds"].pop()
        path.write_text(json.dumps(obj))
        inputs[-1]["report"]["sha256"] = prov.checkpoint_hash(path)
    else:
        path = Path(inputs[-1]["checkpoint"]["path"])
        obj = torch.load(path, weights_only=False)
        if mutation == "model":
            obj["actor"]["test_weight"][0] = 99.0
        else:
            obj["family_lineage"]["training"]["seeds"].append(10000)
            obj["family_lineage"]["digest"] = prov.digest(
                {k: v for k, v in obj["family_lineage"].items() if k != "digest"}
            )
        torch.save(obj, path)
        inputs[-1]["checkpoint"]["sha256"] = prov.checkpoint_hash(path)
    with pytest.raises(prov.ProvenanceError):
        promotion.promote(inputs, held)


def test_balanced_sampler_has_mandatory_defaults_and_exact_resume():
    class Bank:
        def sample(self, rng, masks=None):
            return {"seed": 1001, "mask": 0, "id": 7, "snapshot_sha256": "f" * 64}

    stage = ppo.dataclasses.replace(ppo.STAGES["mixed"], region_movement=True)

    def new(rng, bank=None):
        return curriculum.BalancedSampler(
            stage, np.array([1002]), np.array([4001]), rng, [(1201, 4)], bank
        )

    rng = np.random.default_rng(3)
    sampler = new(rng, Bank())
    worlds = sampler(12)
    assert [s["requested"] for s in sampler.last_sources] == list(curriculum.SCHEDULE)
    assert sum(s["realised"] == "default" for s in sampler.last_sources) == 3
    assert [w[1] for w, s in zip(worlds, sampler.last_sources) if s["realised"] == "fixture"] == [
        12,
        10,
    ]
    state, random_state = sampler.state_dict(), copy.deepcopy(rng.bit_generator.state)
    expected = sampler(7)
    other_rng = np.random.default_rng()
    other_rng.bit_generator.state = random_state
    other = new(other_rng, Bank())
    other.load_state_dict(state)
    assert other(7) == expected
    fallback = new(np.random.default_rng(3))
    fallback(12)
    assert sum(row["realised"] == "default" for row in fallback.last_sources) == 6


def test_execution_tier_never_converts_local_probe_into_readiness():
    assert not protocol.execution_design(
        {"tier": "local", "purpose": "stability", "arms": {"probe": [0, 1, 2]}}
    )["readiness_bearing"]
    for design in (
        {"tier": "local", "purpose": "readiness", "arms": {"probe": [0]}},
        {"tier": "local", "purpose": "stability", "arms": {"probe": [0, 1, 2, 3]}},
        {"tier": "vm", "purpose": "readiness", "arms": {"R3def": [170, 171, 172]}},
    ):
        with pytest.raises(prov.ProvenanceError):
            protocol.execution_design(design)
    assert protocol.execution_design(
        {
            "tier": "vm",
            "purpose": "readiness",
            "arms": {arm: list(range(170, 175)) for arm in ("R3def", "R3per", "R3lo")},
        }
    )["readiness_bearing"]


def test_versioned_protocol_projection_and_external_full_byte_approval(tmp_path):
    document = {
        "schema": 1,
        "protocol_digest_version": 1,
        "hypotheses": ["fixed"],
        "threshold": 0.1,
    }
    document["digest"] = protocol.protocol_digest(document)
    original = document["digest"]
    prereg = json_ref(tmp_path / "prereg.json", document)
    approval = {
        "schema": 1,
        "kind": "protocol_approval",
        "reviewer": "Astra",
        "decision": "APPROVE",
        "protocol_digest": original,
        "preregistration_sha256": prereg["sha256"],
    }
    approval_ref = json_ref(tmp_path / "approval.json", approval)
    assert protocol.verify_preregistration(prereg, approval_ref)["digest"] == original
    for changes in (
        {"threshold": 0.2},
        {"hypotheses": ["changed"]},
        {"new_field": "must-participate"},
        {"approval": {"untrusted": True}},
    ):
        assert protocol.protocol_digest(dict(document, **changes)) != original
    document["clone_protocol_sha256"] = "b" * 64
    assert protocol.protocol_digest(document) == original
    prereg = json_ref(tmp_path / "prereg.json", document)
    with pytest.raises(prov.ProvenanceError, match="complete preregistration"):
        protocol.verify_preregistration(prereg, approval_ref)
    approval["preregistration_sha256"] = prereg["sha256"]
    approval_ref = json_ref(tmp_path / "approval.json", approval)
    assert protocol.verify_preregistration(prereg, approval_ref)["digest"] == original
    with pytest.raises(prov.ProvenanceError, match="projection"):
        protocol.protocol_digest(dict(document, protocol_digest_version=2))


def test_registered_ppo_entrypoint_cannot_bypass_execution_design(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["ppo", "--arm", "R3def", "--seed", "170", "--init", "unused.pt", "--out", str(tmp_path)],
    )
    with pytest.raises(prov.ProvenanceError, match="execution-manifest"):
        ppo.main()


def test_frozen_search_cohort_rejects_dynamic_or_unheld_inputs(tmp_path):
    from anthropocene_rl.autonomous_v7.search_teacher import SearchConfig

    held = cal.manifest(
        {
            "calibration_seeds": list(range(10000, 10100)),
            "training_seeds": [1000],
            "selection_seeds": [2100],
            "waivers": [],
        }
    )
    obj = {
        "schema": 1,
        "role": "calibration",
        "personality": "default",
        "max_alternatives": 19,
        "worlds": [
            {"seed": seed, "withheld": 0, "warmup": 0, "bank_entry": -1, "snapshot_sha256": ""}
            for seed in range(10000, 10100)
        ],
    }
    config = SearchConfig(tail=12, calibration_share=1, calibration_min_per_kind=1)
    path = tmp_path / "cohort.json"

    def save():
        obj["digest"] = prov.digest({k: v for k, v in obj.items() if k != "digest"})
        path.write_text(json.dumps(obj))

    save()
    assert len(protocol.search_cohort(path, None, config, held)[1]) == 100
    obj["worlds"][0]["seed"] = 1000
    save()
    with pytest.raises(prov.ProvenanceError, match="held out"):
        protocol.search_cohort(path, None, config, held)
    obj["worlds"][0]["seed"] = 10000
    obj["worlds"].pop()
    save()
    with pytest.raises(prov.ProvenanceError, match="count"):
        protocol.search_cohort(path, None, config, held)


def test_private_bank_images_are_identical_isolated_and_resume_exact(tmp_path):
    from anthropocene_rl.autonomous_v7.daughter_bank import DaughterBank

    source = DaughterBank(tmp_path / "source")
    source.add(b"synthetic allowed snapshot", 1001, 0, 9, 2, 3, 1, True)
    image = tmp_path / "image.json"
    curriculum.freeze_bank(source, image)
    banks = []
    for name in ("arm-a", "arm-b"):
        owner = tmp_path / name
        bank = DaughterBank(owner / "bank")
        curriculum.private_bank(bank, image, owner)
        banks.append(bank)
    assert curriculum.bank_state(banks[0]) == curriculum.bank_state(banks[1])
    saved = curriculum.bank_state(banks[0])
    banks[0].add(b"new daughter", 1001, 0, 10, 4, 5, 2, True)
    assert curriculum.bank_state(banks[0]) != curriculum.bank_state(banks[1])
    curriculum.restore_bank(banks[0], saved)
    assert curriculum.bank_state(banks[0]) == curriculum.bank_state(banks[1])
    with pytest.raises(prov.ProvenanceError, match="private"):
        curriculum.private_bank(banks[0], image, tmp_path / "different")
    with pytest.raises(prov.ProvenanceError, match="resume"):
        curriculum.private_bank(banks[0], image, tmp_path / "arm-a")


@pytest.mark.parametrize("mutation", ["blob", "snapshot", "inventory", "duplicate"])
def test_bank_preflight_reopens_all_payloads_before_private_copy(tmp_path, mutation):
    from anthropocene_rl.autonomous_v7.daughter_bank import DaughterBank

    source = DaughterBank(tmp_path / "source")
    source.add(b"synthetic snapshot", 1001, 0, 9, 2, 3, 1, True)
    image = tmp_path / "image.json"
    obj = curriculum.freeze_bank(source, image)
    assert curriculum.verify_image(image) == obj
    if mutation == "blob":
        Path(next(iter(obj["files"].values()))["path"]).write_bytes(b"mutated")
    elif mutation == "snapshot":
        obj["entries"][0]["snapshot_sha256"] = "f" * 64
    elif mutation == "inventory":
        obj["files"]["extra.snap"] = next(iter(obj["files"].values()))
    else:
        obj["entries"].append(obj["entries"][0])
    obj["digest"] = prov.digest({k: v for k, v in obj.items() if k != "digest"})
    image.write_text(json.dumps(obj))
    with pytest.raises(prov.ProvenanceError):
        curriculum.verify_image(image)


def test_final_preflight_rejects_local_or_unfrozen_design_before_any_bridge_probe(tmp_path):
    from anthropocene_rl.autonomous_v7.qualification import preflight

    spec = {
        "schema": 1,
        "execution": {"tier": "local", "purpose": "stability", "arms": {"probe": [0, 1, 2]}},
    }
    with pytest.raises(prov.ProvenanceError, match="frozen"):
        preflight(spec, tmp_path)
    spec["digest"] = prov.digest(spec)
    with pytest.raises(prov.ProvenanceError, match="VM cohort"):
        preflight(spec, tmp_path)


def launch_fixture(tmp_path):
    import shlex
    from anthropocene_rl.autonomous_v7.checkpoint import arguments

    _, state = root_state()
    state = copy.deepcopy(state)
    state["artifact_kind"] = "promotion"
    init = tmp_path / "init.pt"
    torch.save(state, init)
    expected = {
        "initialization_sha256": prov.checkpoint_hash(init),
        "learner_commit": "a" * 40,
        "simulator": state["run_manifest"]["simulator"],
    }
    image = {"schema": 1, "entries": [], "files": {}}
    image["digest"] = prov.digest(image)
    image_ref = json_ref(tmp_path / "bank-image.json", image)
    job = {
        "name": "synthetic-final",
        "rl_sha": "a" * 40,
        "sim_sha": "b" * 40,
        "threads": 14,
        "phases": [],
    }
    runs = []
    for arm in ("R3def", "R3per", "R3lo"):
        for seed in range(170, 175):
            name = f"{arm}-{seed}"
            argv = [
                "python",
                "-m",
                "anthropocene_rl.autonomous_v7.ppo",
                "--init",
                str(init),
                "--out",
                str(tmp_path / name),
                "--execution-manifest",
                str(tmp_path / "submission.json"),
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--stage",
                "mixed",
                "--iterations",
                "50",
                "--worlds",
                "12",
                "--turns",
                "25",
                "--learned-movement",
                "--learned-fission",
                "--learned-trade",
                "--experts",
                "--vector-critic",
                "--region-head",
                "hierarchical",
                "--cost-bootstrap",
                "--kl-refit",
                "--resume",
                "--curriculum",
                "r3-balanced-v1",
                "--bank-image",
                image_ref["path"],
                "--daughter-bank",
                str(tmp_path / name / "bank"),
                "--eval-seeds",
                "2060-2099",
                "--eval-every",
                "5",
                "--branch-share",
                ".25",
                "--personality",
                "default" if arm == "R3def" else "dirichlet:1",
                "--teacher-kl",
                ".002" if arm == "R3lo" else ".01",
                "--objective-weights",
                ",".join(f"{k}={v}" for k, v in ppo.contracts.OBJECTIVE_WEIGHTS.items()),
                "--search-labels",
                "synthetic-labels.npz",
                "--branch-shards",
                "synthetic-branches",
            ]
            parser = ppo.argument_parser()
            arguments(parser)
            args = parser.parse_args(argv[3:])
            options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            config = {
                "schema": 1,
                "argv": argv,
                "options": options,
                "effective_iterations": {"critic_only": 5, "actor": 50, "total": 55},
                "resolved_hyper": ppo.dataclasses.asdict(ppo.resolve_configuration(args, state)[0]),
                "resolved_stage": ppo.dataclasses.asdict(ppo.resolve_configuration(args, state)[1]),
                "initialization_sha256": expected["initialization_sha256"],
                "bank_image_digest": image["digest"],
            }
            runs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "phase": name,
                    "argv": argv,
                    "resolved_config": json_ref(tmp_path / (name + "-config.json"), config),
                    "bank_image_sha256": image_ref["sha256"],
                    "search_cohort_digest": ("d" if arm == "R3def" else "e") * 64,
                }
            )
            job["phases"].append({"name": name, "run": shlex.join(argv), "done": name + "/done"})
    spec = {"queue_job": json_ref(tmp_path / "queue.json", job), "runs": runs}
    spec["digest"] = prov.digest(spec)
    (tmp_path / "submission.json").write_text(json.dumps(spec))
    return spec, state, expected


def test_launch_gate_parses_actual_queue_argv_and_effective_defaults(tmp_path):
    from anthropocene_rl.autonomous_v7.qualification import validate_runs

    spec, state, expected = launch_fixture(tmp_path)
    assert validate_runs(spec, tmp_path, state, expected)["R3per"] == {"e" * 64}


def test_search_bindings_accept_actual_qualified_labels_and_complete_branch_metadata(tmp_path):
    from anthropocene_rl.autonomous_v7 import branch_shards, search_teacher as st
    from anthropocene_rl.autonomous_v7.qualification import validate_search_bindings
    from test_autonomous_v7_round3_search import fake_tail, village

    held = cal.manifest(
        {"calibration_seeds": [10000], "training_seeds": [1001], "selection_seeds": [2160]}
    )
    evidence = {}
    for role, personality in (
        ("training-default", [1.0, 0.0, 0.0, 0.0]),
        ("training-dirichlet", [0.25, 0.25, 0.25, 0.25]),
    ):
        directory = tmp_path / role
        directory.mkdir()
        config = {
            "calibration_manifest": held,
            "personality": "default" if role == "training-default" else "dirichlet:1",
        }
        source = {
            "source_checkpoint": "d" * 64,
            "code_commit": "a" * 40,
            "provenance_version": 2,
            "contract": ppo.contracts.NATIVE_CONTRACT,
            "utility_version": 2,
            "search_config": config,
            "config_digest": prov.digest(config),
            "simulator": native_source()["simulator"],
        }
        source["source_run"] = prov.digest(source)
        alternatives = [{"kind": st.POLICY, "slot": -1}, {"kind": st.REGION, "slot": 2}]
        results = [
            {
                "returns": np.array([100.0 + 10 * i, 0.0, 0.0, 0.0]),
                "return": 100.0 + 10 * i,
                "bootstrap": 0.0,
                "bootstraps": np.zeros(4),
                "world": 1.0,
                "native_mix": None,
                "tail": fake_tail(i),
            }
            for i in range(2)
        ]
        worlds = [
            {
                "seed": 1001,
                "withheld": 0,
                "warmup": 0,
                "entry": -1,
                "snapshot_hash": None,
                "rows": [
                    {
                        "village": village(),
                        "alternatives": alternatives,
                        "results": results,
                        "turn": 2,
                        "personality": personality,
                    }
                ],
            }
        ]
        labels = directory / "labels.npz"
        np.savez_compressed(labels, **st.label_arrays(worlds, source))
        branches = directory / "branch_shards"
        st.write_branch_shards(worlds, branches, source)
        # These are real writer outputs that also pass strict source/payload loaders, not arbitrary hashed shells.
        ppo.load_search_labels([labels], 0.0, provenance="reject", calibration_manifest=held)
        branch_shards.load([branches], "reject", calibration_manifest=held)
        evidence[role] = {
            "labels": {"path": str(labels), "sha256": prov.checkpoint_hash(labels)},
            "branch_indexes": [
                {"path": str(p / "meta.json"), "sha256": prov.checkpoint_hash(p / "meta.json")}
                for p in branch_shards.directories([branches])
            ],
        }
        assert len(evidence[role]["branch_indexes"]) == 2
    runs = []
    for arm in ("R3def", "R3per", "R3lo"):
        item = evidence["training-default" if arm == "R3def" else "training-dirichlet"]
        runs.append(
            {
                "argv": [
                    "python",
                    "-m",
                    "anthropocene_rl.autonomous_v7.ppo",
                    "--init",
                    "synthetic-init.pt",
                    "--out",
                    str(tmp_path / arm),
                    "--arm",
                    arm,
                    "--search-labels",
                    item["labels"]["path"],
                    "--branch-shards",
                    str(Path(item["branch_indexes"][0]["path"]).parent.parent),
                ]
            }
        )
    validate_search_bindings({"runs": runs}, {"evidence": evidence}, tmp_path)


@pytest.mark.parametrize("mutation", [None, "flat", "horizon", "manifest"])
def test_real_search_summary_normalizes_manifest_before_binding_config(tmp_path, mutation):
    from anthropocene_rl.autonomous_v7 import search_teacher as st
    from anthropocene_rl.autonomous_v7.gate_validation import validate_search_producer

    held, _ = root_state()
    path = tmp_path / "held.json"
    json_ref(path, held)
    cohort = {"schema": 1, "role": "training-default", "worlds": [{"seed": 1001}]}
    config = ppo.dataclasses.asdict(st.SearchConfig(calibration_manifest=str(path)))
    source = {
        "search_config": dict(
            config, calibration_manifest=held, worlds_manifest=cohort, region_head="hierarchical"
        )
    }
    summary = {"config": config}
    if mutation == "flat":
        source["search_config"]["region_head"] = "flat"
    elif mutation == "horizon":
        source["search_config"]["horizon"] = 7
    elif mutation == "manifest":
        changed = dict(held)
        changed["training_seeds"] = [1001]
        changed.pop("manifest_digest")
        json_ref(path, cal.manifest(changed))
    if mutation:
        with pytest.raises(prov.ProvenanceError):
            validate_search_producer(source, summary, cohort, tmp_path)
    else:
        validate_search_producer(source, summary, cohort, tmp_path)
        # Newly published summaries use the canonical embedded form; both vintages resolve identically.
        validate_search_producer(
            source, {"config": dict(config, calibration_manifest=held)}, cohort, tmp_path
        )
        st.validate_registered_head(types.SimpleNamespace(region_head="hierarchical"))
        with pytest.raises(prov.ProvenanceError):
            st.validate_registered_head(types.SimpleNamespace(region_head="flat"))


@pytest.mark.parametrize("mutation", [None, "root", "snapshot", "extra"])
def test_training_search_cohorts_match_all_nonpersonality_declarations(mutation):
    from anthropocene_rl.autonomous_v7.gate_validation import validate_training_cohorts

    a = {
        "schema": 1,
        "role": "training-default",
        "personality": "default",
        "digest": "a" * 64,
        "worlds": [{"seed": 1001, "snapshot_sha256": "b" * 64}],
        "extra_training_roots": [1002],
    }
    b = copy.deepcopy(a)
    b.update(role="training-dirichlet", personality="dirichlet:1", digest="c" * 64)
    if mutation == "root":
        b["worlds"][0]["seed"] = 1003
    elif mutation == "snapshot":
        b["worlds"][0]["snapshot_sha256"] = "d" * 64
    elif mutation == "extra":
        b["extra_training_roots"] = [1004]
    if mutation:
        with pytest.raises(prov.ProvenanceError):
            validate_training_cohorts({"training-default": a, "training-dirichlet": b})
    else:
        validate_training_cohorts({"training-default": a, "training-dirichlet": b})


def test_snapshot_root_is_validated_before_python_reset_or_observation():
    from anthropocene_rl.autonomous_v7.bridge import Model9Bridge

    bridge = Model9Bridge.__new__(Model9Bridge)
    calls = []

    def request(payload):
        calls.append(payload["op"])
        assert payload["op"] == "validate_snapshot", (
            "must not reset/observe an unattributed snapshot"
        )
        return {"sourceSeed": 3000}

    bridge.request = request
    with pytest.raises(prov.ProvenanceError, match="denied"):
        bridge.reset(1001, snapshot=b"synthetic only")
    assert calls == ["validate_snapshot"]
    bridge.request = lambda payload: {"sourceSeed": 1002}
    with pytest.raises(ValueError, match="before reset"):
        bridge.reset(1001, snapshot=b"synthetic only")


def test_native_confirmation_guards_precede_generation_and_environment_reset():
    source = Path("integration/Model9BridgeProgram.cs").read_text()
    reset = source[source.index('case "reset":') : source.index('case "validate_snapshot":')]
    assert reset.index("RequireUnsealedSeed(declaredSeed)") < reset.index(
        "GenerateOpening(declaredSeed"
    )
    assert reset.index("RequireUnsealedSeed(snapshotWorld.Seed)") < reset.index(
        "environment.Reset(snapshot, movement)"
    )
    assert reset.index("snapshotWorld.Seed != declaredSeed") < reset.index(
        "environment.Reset(snapshot, movement)"
    )


def test_raw_on_bank_reset_rejected_before_warmup_or_actor_action(monkeypatch):
    from anthropocene_rl.autonomous_v7 import evaluate

    expected = {
        "simulator": native_source()["simulator"],
        "bridge_sha256": "d" * 64,
        "flags": {"sim_rules": "", "sim_statics": ""},
        "raw_materials": False,
    }
    worker = synthetic_worker()
    actor = types.SimpleNamespace(
        run_producer=expected, worker_contract=worker, calibration_manifest=None
    )

    class Bridge:
        def request(self, payload):
            assert payload["op"] == "reset_metadata"
            return {
                "sourceSeed": 1001,
                "turn": 9,
                "rawMaterialsEnabled": True,
                "featureFlags": worker["feature_flags"],
            }

        def reset(self, seed, **kw):
            assert seed == 1001 and kw["snapshot"] == b"synthetic raw-on bank snapshot"
            return {"turn": 0, "villages": []}

        def step(self, *_):
            pytest.fail("must reject before warm-up or action")

        def close(self):
            pass

    import hashlib

    bank = types.SimpleNamespace(
        get=lambda n: {
            "id": n,
            "turn": 9,
            "snapshot_sha256": hashlib.sha256(b"synthetic raw-on bank snapshot").hexdigest(),
        },
        snapshot=lambda e: b"synthetic raw-on bank snapshot",
    )
    from anthropocene_rl.autonomous_v7 import worker_contract

    monkeypatch.setattr(worker_contract, "observe", lambda bridge: worker)
    monkeypatch.setattr(ppo, "Model9Bridge", Bridge)
    monkeypatch.setattr(
        evaluate, "producer_identity", lambda bridge: dict(expected, raw_materials=True)
    )
    with pytest.raises(prov.ProvenanceError, match="before reset"):
        ppo.collect(
            actor, None, [(1001, 0, 2, 0)], 1, ppo.STAGES["mixed"], torch.Generator(), bank=bank
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "queue",
        "free-config",
        "objective",
        "commitments",
        "selection",
        "warmup",
        "bank",
        "cohort",
        "lr",
    ],
)
def test_launch_refuses_argv_config_cohort_and_default_drift(tmp_path, mutation):
    from anthropocene_rl.autonomous_v7.qualification import validate_runs
    import shlex

    spec, state, expected = launch_fixture(tmp_path)
    run = spec["runs"][-1]
    argv = run["argv"]
    if mutation == "queue":
        obj = json.loads(Path(spec["queue_job"]["path"]).read_text())
        obj["phases"][-1]["run"] += " --lr .9"
        spec["queue_job"] = json_ref(Path(spec["queue_job"]["path"]), obj)
    elif mutation == "free-config":
        config = json.loads(Path(run["resolved_config"]["path"]).read_text())
        config["effective_iterations"]["critic_only"] = 0
        run["resolved_config"] = json_ref(Path(run["resolved_config"]["path"]), config)
    elif mutation == "bank":
        run["bank_image_sha256"] = "f" * 64
    elif mutation == "cohort":
        run["search_cohort_digest"] = "f" * 64
    else:
        if mutation == "objective":
            argv[argv.index("--objective-weights") + 1] = "trade_utility=1"
        elif mutation == "commitments":
            argv.append("--commitments")
        elif mutation == "selection":
            argv[argv.index("--eval-seeds") + 1] = "2000-2059"
        elif mutation == "warmup":
            argv.extend(["--critic-warmup", "0"])
        else:
            argv.extend(["--lr", ".9"])
        job = json.loads(Path(spec["queue_job"]["path"]).read_text())
        job["phases"][-1]["run"] = shlex.join(argv)
        spec["queue_job"] = json_ref(Path(spec["queue_job"]["path"]), job)
    with pytest.raises(prov.ProvenanceError):
        validate_runs(spec, tmp_path, state, expected)


def test_confirmation_denied_in_source_bank_ancestors_commands_and_before_reset():
    for obj in (
        {"seed": 3000},
        {"parents": [{"proof": {"training_seeds": [3059]}}]},
        {"seeds": "2999-3060"},
    ):
        with pytest.raises(prov.ProvenanceError, match="denied"):
            protocol.scan(obj)
    with pytest.raises(prov.ProvenanceError):
        protocol.command(["--seeds", "2999-3060"])
    from anthropocene_rl.autonomous_v7.bridge import Model9Bridge

    bridge = Model9Bridge.__new__(Model9Bridge)
    bridge.request = lambda payload: pytest.fail("must reject before native reset")
    with pytest.raises(prov.ProvenanceError):
        bridge.reset(3000)


def test_inventory_rehashes_content_and_refuses_bare_markers(tmp_path):
    evidence = tmp_path / "scope"
    evidence.mkdir()
    path = evidence / "prior.json"
    path.write_text(json.dumps({"seeds": [1000]}))
    spec = {
        "schema": 1,
        "scope_roots": [str(evidence)],
        "artifacts": [
            {
                "path": str(path),
                "sha256": prov.checkpoint_hash(path),
                "role": "prior_use",
                "kind": "json",
            }
        ],
    }
    result = protocol.inventory(spec)
    report = tmp_path / "inventory.json"
    report.write_text(json.dumps(result))
    assert protocol.verify_inventory(report) == result
    path.write_text(json.dumps({"seeds": [10000]}))
    with pytest.raises(prov.ProvenanceError, match="hash mismatch"):
        protocol.verify_inventory(report)
    marker = tmp_path / "done"
    marker.write_text("pass")
    with pytest.raises((prov.ProvenanceError, json.JSONDecodeError)):
        protocol.receipt({"path": str(marker), "sha256": prov.checkpoint_hash(marker)}, {})


def test_evidence_hashing_detects_same_size_mutation_with_restored_timestamp(tmp_path):
    import os

    path = tmp_path / "evidence.json"
    path.write_text('{"seed":1000}')
    sha = prov.checkpoint_hash(path)
    stat = path.stat()
    path.write_text('{"seed":1001}')
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert prov.checkpoint_hash(path) != sha
    with pytest.raises(prov.ProvenanceError, match="hash mismatch"):
        protocol.artifact({"path": str(path), "sha256": sha})


@pytest.mark.parametrize("n,expected", [(0, True), (1, False), (19, False), (20, True)])
def test_nonmandatory_relative_subset_zero_na_small_fails_populated_gates(n, expected):
    s = ppo.contracts.OBJECTIVE_REFERENCE_SCALES["welfare"]
    good = cal.statistics([s] * 20, [s] * 20, "welfare")
    subset = cal.statistics([0.0] * 20 + [s] * n, [0.0] * 20 + [s] * n, "welfare")
    report = {"kinds": {k: {"welfare": good} for k in ("native", "region", "mix")}}
    report["kinds"]["trade"] = {"welfare": subset}
    assert cal.verdict(report, {"warnings": []})["pass"] is expected


def sweep(seed, qgap=0.1, pgap=10, focal=False):
    from anthropocene_rl.autonomous_v7.personality_sweep import GRID

    result = {"_meta": {"optimizer_seed": seed}}
    for point in ("qol", "population", "welfare"):
        rows = []
        for world in range(2000, 2060):
            from anthropocene_rl.autonomous_v7.worker_contract import world as world_contract

            row = {
                "seed": world,
                "withheld": 0,
                "warmup": 0,
                "world_contract": world_contract(world, head="hierarchical"),
                "personality": GRID[point],
                "qol": 0.5 + (qgap if point == "qol" else 0),
                "population": 100 + (pgap if point == "population" else 0),
                "welfare": 100.0,
                "extinctions": 0,
            }
            if focal:
                row.update(
                    {"focal_" + k: row[k] for k in ("qol", "population", "welfare", "extinctions")},
                    focal_root_lineages=2,
                    focal_population_denominator=row["population"],
                    focal_societies=[1, 2],
                )
            rows.append(row)
        result[point] = {"per_seed": rows}
    return result


def test_formal_paired_statistics_and_focal_evidence_are_not_pooled_proxies():
    reports = [sweep(s, focal=True) for s in range(170, 175)]
    assert formal_report.separation(reports, focal=True)["pass"]
    bad = copy.deepcopy(reports)
    bad[0]["qol"]["per_seed"][0].pop("focal_population_denominator")
    with pytest.raises(prov.ProvenanceError, match="incomplete"):
        formal_report.separation(bad, focal=True)
    values = np.arange(300).reshape(5, 60) + 1.0
    ratio = formal_report.paired_interval(values, values)
    difference = formal_report.paired_interval(values, values, difference=True)
    assert ratio["lo"] == ratio["hi"] == 1 and difference["lo"] == difference["hi"] == 0


def json_ref(path, obj):
    path.write_text(json.dumps(obj))
    return {"path": str(path.resolve()), "sha256": prov.checkpoint_hash(path)}


def formal_fixture(tmp_path, monkeypatch):
    from anthropocene_rl.autonomous_v7.personality_sweep import GRID
    from anthropocene_rl.autonomous_v7 import completion

    launch, _, expected = launch_fixture(tmp_path)
    _, parent = root_state()
    init = Path(launch["runs"][0]["argv"][4])
    init_ref = {"path": str(init), "sha256": prov.checkpoint_hash(init)}
    launch.update(
        schema=1,
        protocol_digest=parent["run_manifest"]["protocol_digest"],
        initialization=init_ref,
        learner_commit="a" * 40,
        simulator=parent["run_manifest"]["simulator"],
        bridge_sha256="d" * 64,
        execution={
            "tier": "vm",
            "purpose": "readiness",
            "arms": {a: list(range(170, 175)) for a in ("R3def", "R3per", "R3lo")},
        },
    )
    gates = {}
    for kind in __import__("anthropocene_rl.autonomous_v7.qualification", fromlist=["GATES"]).GATES:
        body = {"schema": 1, "kind": kind, "status": "pass", "outputs": [init_ref]}
        if kind == "protocol":
            body["evidence"] = {
                "preregistration": parent["run_manifest"]["protocol_evidence"]["preregistration"],
                "review": parent["run_manifest"]["protocol_evidence"]["approval"],
            }
        gates[kind] = json_ref(tmp_path / (kind + "-gate.json"), body)
    launch["gates"] = gates
    launch["digest"] = prov.digest({k: v for k, v in launch.items() if k != "digest"})
    launch_ref = json_ref(tmp_path / "submission.json", launch)
    qualified = {
        "schema": 1,
        "kind": "qualified_submission",
        "status": "pass",
        "submission": launch_ref,
        "submission_digest": launch["digest"],
        "initialization": init_ref,
        "protocol_digest": launch["protocol_digest"],
        "live_probe": {"worker_contract": synthetic_worker()},
        "frozen_runs": {},
        "gates": {
            k: prov.digest(json.loads(Path(ref["path"]).read_text())) for k, ref in gates.items()
        },
    }
    spec = {"arms": {}}
    for arm in ("R3def", "R3per", "R3lo"):
        spec["arms"][arm] = {
            form: {mode: [] for mode in (("all",) if arm == "R3def" else ("all", "focal"))}
            for form in ("best", "latest")
        }
        for seed in range(170, 175):
            directory = tmp_path / f"{arm}-{seed}"
            directory.mkdir()
            _, state = root_state()
            state = copy.deepcopy(state)
            run = state["run_manifest"]
            row = next(r for r in launch["runs"] if r["arm"] == arm and r["seed"] == seed)
            config = json.loads(Path(row["resolved_config"]["path"]).read_text())
            run["options"] = {k: v for k, v in config["options"].items() if k != "resume"}
            run["resolved_hyper"] = config["resolved_hyper"]
            run["resolved_stage"] = config["resolved_stage"]
            run["inputs"] = {
                "init": init_ref["sha256"],
                "execution_manifest": launch_ref["sha256"],
                "bank_image": row["bank_image_sha256"],
            }
            run["effective_iterations"] = {"critic_only": 5, "actor": 50, "total": 55}
            qualified["frozen_runs"][row["phase"]] = {
                "arm": arm,
                "seed": seed,
                "run_manifest": copy.deepcopy(run),
                "run_manifest_digest": prov.digest(run),
            }
            for form, iteration in (("best", 49), ("latest", 54)):
                state["iteration"] = iteration
                torch.save(state, directory / (form + ".pt"))
            rows = [{"iteration": i, "critic_warmup": i < 5} for i in range(55)]
            producer = {
                "simulator": run["simulator"],
                "bridge_sha256": run["bridge_sha256"],
                "flags": {"sim_rules": "", "sim_statics": ""},
                "raw_materials": False,
            }
            rows[49]["eval"] = {
                "eligible": True,
                "selection_score": 1.0,
                "worker_contract": synthetic_worker(),
                "producer": producer,
            }
            provenance = {"worker_contract": synthetic_worker(), "producer": producer}
            rows[49]["eval"]["regimes"] = {"default": provenance}
            if arm != "R3def":
                rows[49]["eval"]["spread"] = {
                    "points": {key: provenance for key in ppo.SELECTION_POINTS}
                }
            (directory / "trajectory.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            done = json_ref(directory / "completion.json", completion.build(directory, run))
            producer = {
                "simulator": run["simulator"],
                "bridge_sha256": run["bridge_sha256"],
                "flags": {"sim_rules": "", "sim_statics": ""},
                "raw_materials": False,
            }
            for form in ("best", "latest"):
                cp = directory / (form + ".pt")
                for mode in spec["arms"][arm][form]:
                    report = sweep(seed, focal=mode == "focal")
                    for point, vector in GRID.items():
                        if point not in report:
                            report[point] = copy.deepcopy(report["welfare"])
                        report[point]["personality"] = vector
                        for row in report[point]["per_seed"]:
                            row["personality"] = vector
                            from anthropocene_rl.autonomous_v7.worker_contract import world

                            row["world_contract"] = world(
                                row["seed"], movement=True, head="hierarchical"
                            )
                    report["_meta"] = {
                        "schema": 1,
                        "arm": arm,
                        "optimizer_seed": seed,
                        "checkpoint_form": form,
                        "checkpoint": str(cp.resolve()),
                        "checkpoint_sha256": prov.checkpoint_hash(cp),
                        "completion": done,
                        "turns": 25,
                        "points": list(GRID),
                        "focal": mode == "focal",
                        "movement": True,
                        "region_head": "hierarchical",
                        "model_semantics": state["model_semantics"],
                        "worker_contract": synthetic_worker(),
                        "producer": producer,
                        "simulator": run["simulator"],
                        "bridge_sha256": run["bridge_sha256"],
                    }
                    spec["arms"][arm][form][mode].append(
                        json_ref(directory / f"{form}-{mode}.json", report)
                    )
    spec["qualification"] = json_ref(tmp_path / "qualified.json", qualified)
    # Synthetic statistical/cohort fixtures are not production-qualified. Isolate only upstream semantic gates;
    # full 15-run configurations, completion artifacts, hashes and membership are checked by real report code.
    from anthropocene_rl.autonomous_v7 import qualification

    monkeypatch.setattr(
        qualification,
        "validate_qualified_submission",
        lambda *a, **kw: {k: v for k, v in qualified.items() if k != "submission"},
    )
    return spec


def test_formal_report_complete_bound_arms_and_low_vs_per_is_report_only(tmp_path, monkeypatch):
    spec = formal_fixture(tmp_path, monkeypatch)
    original = formal_report.default_comparison
    calls = []

    def comparison(a, b):
        calls.append((a[0]["_meta"]["arm"], b[0]["_meta"]["arm"]))
        return {"pass": not (a[0]["_meta"]["arm"] == "R3per" and b[0]["_meta"]["arm"] == "R3lo")}

    monkeypatch.setattr(formal_report, "default_comparison", comparison)
    result = formal_report.evaluate(spec, tmp_path)
    assert result["pass"] and all(not r["pass"] for r in result["lo_vs_per"].values())
    assert calls.count(("R3per", "R3lo")) == 2


@pytest.mark.parametrize(
    "mutation",
    ["alias", "early-latest", "missing-grid", "vector", "row-mask", "turns", "worker-evidence"],
)
def test_formal_reports_refuse_alias_substitution_and_incomplete_grid(
    tmp_path, monkeypatch, mutation
):
    spec = formal_fixture(tmp_path, monkeypatch)
    ref = spec["arms"]["R3per"]["latest"]["all"][0]
    if mutation == "alias":
        spec["arms"]["R3per"]["latest"]["all"][0] = spec["arms"]["R3lo"]["latest"]["all"][0]
    else:
        path = Path(ref["path"])
        report = json.loads(path.read_text())
        if mutation == "early-latest":
            report["_meta"]["checkpoint"] = report["_meta"]["checkpoint"].replace(
                "latest.pt", "best.pt"
            )
            report["_meta"]["checkpoint_sha256"] = prov.checkpoint_hash(
                report["_meta"]["checkpoint"]
            )
        elif mutation == "missing-grid":
            report.pop("trader")
        elif mutation == "vector":
            report["qol"]["personality"] = [1, 0, 0, 0]
        elif mutation == "row-mask":
            report["qol"]["per_seed"][0]["withheld"] = 12
        elif mutation == "worker-evidence":
            report["_meta"]["worker_contract"] = None
        else:
            report["_meta"]["turns"] = 4
        ref.update(json_ref(path, report))
    with pytest.raises(prov.ProvenanceError):
        formal_report.evaluate(spec, tmp_path)


@pytest.mark.parametrize("mutation", ["root", "binary", "bank", "config", "protocol", "code"])
def test_formal_report_rejects_self_consistent_run_outside_one_qualified_cohort(
    tmp_path, monkeypatch, mutation
):
    from anthropocene_rl.autonomous_v7 import completion

    spec = formal_fixture(tmp_path, monkeypatch)
    directory = tmp_path / "R3per-170"
    for form in ("best", "latest"):
        path = directory / (form + ".pt")
        state = torch.load(path, weights_only=False)
        run = state["run_manifest"]
        if mutation == "root":
            run["inputs"]["init"] = "f" * 64
        elif mutation == "binary":
            run["bridge_sha256"] = "f" * 64
        elif mutation == "bank":
            run["inputs"]["bank_image"] = "f" * 64
        elif mutation == "config":
            run["options"]["lr"] = 0.9
        elif mutation == "protocol":
            run["protocol_digest"] = "f" * 64
        else:
            run["code"] = "f" * 40
        torch.save(state, path)
    # Rebuild each run's own completion and sweep hashes so only the global qualification binding rejects it.
    latest = torch.load(directory / "latest.pt", weights_only=False)
    done_ref = json_ref(
        directory / "completion.json", completion.build(directory, latest["run_manifest"])
    )
    for form in ("best", "latest"):
        for mode in ("all", "focal"):
            ref = spec["arms"]["R3per"][form][mode][0]
            path = Path(ref["path"])
            report = json.loads(path.read_text())
            report["_meta"]["checkpoint_sha256"] = prov.checkpoint_hash(directory / (form + ".pt"))
            report["_meta"]["completion"] = done_ref
            ref.update(json_ref(path, report))
    with pytest.raises(prov.ProvenanceError, match="qualified cohort"):
        formal_report.evaluate(spec, tmp_path)


def test_fabricated_immutable_qualified_status_shell_is_not_truth(tmp_path, monkeypatch):
    from anthropocene_rl.autonomous_v7 import qualification

    original = qualification.validate_qualified_submission
    spec = formal_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(qualification, "validate_qualified_submission", original)
    with pytest.raises(prov.ProvenanceError, match="semantic qualification"):
        formal_report.evaluate(spec, tmp_path)
