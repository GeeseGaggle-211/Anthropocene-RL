"""Pre-execution contracts: no reset/generation/warmup/action may precede rejection."""

import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from test_autonomous_v7_round3_protocol import root_state, synthetic_worker

from anthropocene_rl.autonomous_v7 import (
    evaluate,
    ppo,
)
from anthropocene_rl.autonomous_v7 import (
    provenance as prov,
)
from anthropocene_rl.autonomous_v7 import (
    search_teacher as search,
)
from anthropocene_rl.autonomous_v7 import (
    worker_contract as wc,
)


@pytest.fixture
def worker(monkeypatch):
    expected = synthetic_worker()
    calls = []
    schema = types.SimpleNamespace(
        build_hash=expected["simulator"]["buildHash"],
        simulator_commit=expected["simulator"]["commit"],
        simulator_source_clean=True,
        bridge_sha256=expected["bridge_sha256"],
        bridge_build_hash=expected["bridge_build_hash"],
        decision_diagnostics_schema=1,
        diagnostic_envelope_schema=2,
        raw_material_diagnostics_schema=2,
        feature_flags=copy.deepcopy(expected["feature_flags"]),
        producer_flags=expected["flags"],
        configuration_hash=expected["configuration"]["sha256"],
        players=8,
    )
    metadata = {
        "sourceSeed": 1001,
        "turn": 9,
        "rawMaterialsEnabled": False,
        "featureFlags": copy.deepcopy(schema.feature_flags),
        "configurationHash": schema.configuration_hash,
        "players": 8,
    }

    class Bridge:
        def __init__(self):
            self.schema = schema

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def close(self):
            pass

        def request(self, request):
            calls.append(request["op"])
            assert request["op"] == "reset_metadata"
            return {
                **metadata,
                "turn": metadata["turn"] if "snapshot" in request else None,
                "sourceSeed": metadata.get("sourceSeed", request["seed"]),
            }

        def reset(self, *a, **k):
            calls.append("reset")
            pytest.fail("reset executed before producer rejection")

        def step(self, *a, **k):
            calls.append("step")
            pytest.fail("warmup/action executed before rejection")

        def fork(self):
            pytest.fail("branch executed before rejection")

    from anthropocene_rl.autonomous_v7 import teacher_data

    monkeypatch.setattr(
        teacher_data, "binary_identity", lambda: {"sha256": expected["bridge_sha256"]}
    )
    monkeypatch.setenv("ANTHROPOCENE_SIM_RULES", "")
    monkeypatch.setenv("ANTHROPOCENE_SIM_STATICS", "")
    return expected, schema, metadata, calls, Bridge


@pytest.mark.parametrize(
    "mutation",
    [
        "binary",
        "build",
        "envelope",
        "head",
        "raw_schema",
        "feature",
        "snapshot_raw",
        "snapshot_turn",
        "snapshot_root",
        "snapshot_hash",
        "configuration",
        "players",
    ],
)
def test_pre_reset_contract_rejects_every_changed_binding(worker, mutation):
    expected, schema, meta, calls, Bridge = worker
    raw = b"synthetic allowed snapshot"
    descriptor = wc.world(
        1001, snapshot_sha256=hashlib.sha256(raw).hexdigest(), turn=9, head="hierarchical"
    )
    if mutation == "binary":
        schema.bridge_sha256 = "f" * 64
    elif mutation == "build":
        schema.build_hash = "f" * 32
    elif mutation == "envelope":
        schema.diagnostic_envelope_schema = 1
    elif mutation == "head":
        schema.decision_diagnostics_schema = 0
    elif mutation == "raw_schema":
        schema.raw_material_diagnostics_schema = 1
    elif mutation == "feature":
        schema.feature_flags["SimulationConfig.ElderRawMaterialsEnabled"] = True
    elif mutation == "snapshot_raw":
        meta["rawMaterialsEnabled"] = True
    elif mutation == "snapshot_turn":
        meta["turn"] = 10
    elif mutation == "snapshot_root":
        meta["sourceSeed"] = 1002
    elif mutation == "configuration":
        meta["configurationHash"] = "DIFFERENT_NUMERIC_RULES"
    elif mutation == "players":
        meta["players"] = 999
    else:
        raw += b"mutation"
    with pytest.raises(prov.ProvenanceError):
        wc.validate(Bridge(), expected, descriptor, snapshot=raw, head="hierarchical")
    assert "reset" not in calls and "step" not in calls


@pytest.mark.parametrize("path", ["evaluate", "search", "ppo", "daughter", "sweep"])
@pytest.mark.parametrize("mutation", ["raw", "configuration", "players"])
def test_actual_worker_entrypoints_reject_before_native_execution(
    worker, monkeypatch, tmp_path, path, mutation
):
    expected, _schema, metadata, calls, Bridge = worker
    if mutation == "raw":
        metadata["rawMaterialsEnabled"] = True
    elif mutation == "configuration":
        metadata["configurationHash"] = "DIFFERENT_NUMERIC_RULES"
    else:
        metadata["players"] = 999
    actor = types.SimpleNamespace(
        region_head="hierarchical", region_movement=True, worker_contract=expected
    )
    if path == "evaluate":
        monkeypatch.setattr(evaluate, "Model9Bridge", Bridge)
        with pytest.raises(prov.ProvenanceError):
            evaluate.rollout(1001, 2, None, warmup=3, expected_producer=expected)
    elif path == "search":
        monkeypatch.setattr(search, "Model9Bridge", Bridge)
        monkeypatch.setattr(search, "load", lambda path: (actor, None))
        monkeypatch.setattr(search, "load_critic", lambda path: None)
        with pytest.raises(prov.ProvenanceError):
            search.search_world(
                "synthetic.pt", 1001, 0, 3, search.SearchConfig(), expected_producer=expected
            )
    elif path == "ppo":
        monkeypatch.setattr(ppo, "Model9Bridge", Bridge)
        with pytest.raises(prov.ProvenanceError):
            ppo.collect(actor, None, [(1001, 0, 3)], 1, ppo.STAGES["mixed"], torch.Generator())
    elif path == "daughter":
        from anthropocene_rl.autonomous_v7 import daughter_eval

        raw = b"synthetic allowed daughter"
        entry = {
            "id": 0,
            "seed": 1000,
            "mask": 0,
            "society": 3,
            "camp": 7,
            "turn": 9,
            "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        }
        metadata["sourceSeed"] = 1000
        monkeypatch.setattr(
            daughter_eval,
            "DaughterBank",
            lambda bank: types.SimpleNamespace(get=lambda n: entry, snapshot=lambda e: raw),
        )
        monkeypatch.setattr(daughter_eval, "Model9Bridge", Bridge)
        with pytest.raises(prov.ProvenanceError):
            daughter_eval.run_entry("synthetic-bank", 0, None, 2, expected_producer=expected)
    else:
        from anthropocene_rl.autonomous_v7 import personality_sweep

        class Pool:
            def __init__(self, *_args, **_kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def map(self, fn, *args):
                return map(fn, *args)

        _, state = root_state()
        checkpoint = tmp_path / "state.pt"
        torch.save(state, checkpoint)
        monkeypatch.setattr(personality_sweep, "ProcessPoolExecutor", Pool)
        monkeypatch.setattr(evaluate, "Model9Bridge", Bridge)
        monkeypatch.setattr(evaluate, "load", lambda path: (actor, None))
        metadata["sourceSeed"] = 2000
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "personality_sweep",
                "--checkpoint",
                str(checkpoint),
                "--seeds",
                "2000-2000",
                "--points",
                "welfare",
                "--out",
                str(tmp_path / "out.json"),
            ],
        )
        with pytest.raises(prov.ProvenanceError):
            personality_sweep.main()
    assert calls == ["reset_metadata"]


def test_selection_aggregation_refuses_spoofed_worker_before_metrics_or_best(worker, monkeypatch):
    expected, *_ = worker

    class Pool:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def map(self, *a):
            return [{"worker_contract": expected, "producer": {"raw_materials": True}}]

    import concurrent.futures

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", Pool)
    with pytest.raises(prov.ProvenanceError, match="cannot select"):
        ppo.evaluate_closed_loop(None, [(1001, 0)], 25, expected_producer=expected)


def test_pre_reset_valid_contract_installs_exact_frozen_world(worker):
    expected, _, _, calls, Bridge = worker
    raw = b"allowed ordinary snapshot"
    descriptor = wc.world(
        1001, snapshot_sha256=hashlib.sha256(raw).hexdigest(), turn=9, head="hierarchical"
    )
    bridge = Bridge()
    assert wc.validate(bridge, expected, descriptor, snapshot=raw, head="hierarchical") == expected
    assert (
        bridge._expected_worker_contract == expected
        and bridge._expected_world_contract == descriptor
    )
    assert calls == ["reset_metadata"]


def test_collection_proof_rejected_before_flatten_or_optimization(worker, monkeypatch):
    expected, *_ = worker
    actor = types.SimpleNamespace(worker_contract=expected)
    monkeypatch.setattr(
        ppo, "collect", lambda *a, **k: ppo.Rollout(worker_contracts=[], world_contracts=[])
    )
    monkeypatch.setattr(
        ppo,
        "flatten",
        lambda *a: pytest.fail("untrusted collection must not reach optimizer input"),
    )
    hyper = types.SimpleNamespace(
        personality="default", log_std_floor=-2, learner_contract="legacy"
    )
    with pytest.raises(prov.ProvenanceError, match="before optimization"):
        ppo.iteration(
            actor,
            None,
            None,
            None,
            None,
            [1001],
            25,
            types.SimpleNamespace(bridge_movement=False),
            hyper,
            (),
            0,
            None,
            None,
        )


def test_bound_bridge_rechecks_binary_before_step_and_world_before_reset(worker, monkeypatch):
    from anthropocene_rl.autonomous_v7.bridge import Model9Bridge

    expected, schema, _, _, _Bridge = worker
    bridge = Model9Bridge.__new__(Model9Bridge)
    bridge.schema = schema
    bridge._expected_worker_contract = expected
    bridge._expected_world_contract = wc.world(1001)
    monkeypatch.setattr(
        bridge, "request", lambda *a, **k: pytest.fail("native request must not execute")
    )
    with pytest.raises(prov.ProvenanceError, match="world"):
        bridge.reset(1002)
    with pytest.raises(prov.ProvenanceError, match="world"):
        bridge.reset(1001, movement=True)
    with pytest.raises(prov.ProvenanceError, match="world"):
        bridge.reset(1001, size=(20, 20, 3))
    schema.bridge_sha256 = "f" * 64
    with pytest.raises(prov.ProvenanceError):
        bridge.step({})


def test_live_bound_producer_and_immediate_reset_fork_restore_keep_explicit_envelope():
    import os

    from anthropocene_rl.autonomous_v7.bridge import DEFAULT_BRIDGE, Model9Bridge

    if not Path(os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE", DEFAULT_BRIDGE)).exists():
        pytest.skip("bridge not built")
    with Model9Bridge() as bridge:
        expected = wc.observe(bridge)
        wc.validate(bridge, expected, wc.world(1001))
        initial = bridge.reset(1001)
        assert initial["diagnosticSchemaVersion"] == 0 and initial["trajectoryId"] is None
        fork, _ = bridge.fork()
        bridge.step({})
        restored = bridge.restore(fork)
        assert restored["diagnosticSchemaVersion"] == 0 and restored["trajectoryId"] is None
        assert bridge.decision_diagnostics == [] and bridge.raw_material_diagnostics == {}


@pytest.mark.parametrize("kind", ["train", "ppo"])
def test_actual_fit_entrypoint_requires_vm_authority_before_rng_or_fitting(
    tmp_path, monkeypatch, kind
):
    from anthropocene_rl.autonomous_v7 import train

    monkeypatch.setattr(
        torch, "manual_seed", lambda *a: pytest.fail("must reject before RNG/fitting")
    )
    if kind == "train":
        argv = [
            "train",
            "--train",
            "missing",
            "--dev",
            "missing",
            "--out",
            str(tmp_path / "model.pt"),
            "--clone-protocol",
            "missing",
        ]
    else:
        argv = ["ppo", "--init", "missing", "--out", str(tmp_path / "ppo")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(prov.ProvenanceError, match=r"execution|VM"):
        (train.main if kind == "train" else ppo.main)()


def test_nonqualifying_probe_cannot_carry_protocol_or_create_qualified_lineage():
    from anthropocene_rl.autonomous_v7 import lineage, protocol

    args = types.SimpleNamespace(
        nonqualifying=True, arm=None, seed=0, calibration_manifest="held.json"
    )
    with pytest.raises(prov.ProvenanceError):
        protocol.enforce_execution(args)
    held, state = root_state()
    state["nonqualifying"] = True
    assert not lineage.checkpoint_evidence(state, held)["verified"]


@pytest.mark.parametrize("kind", ["train", "ppo"])
def test_local_four_seed_workload_rejected_by_actual_entrypoint(tmp_path, monkeypatch, kind):
    from anthropocene_rl.autonomous_v7 import train

    body = {
        "schema": 1,
        "execution": {
            "tier": "local",
            "purpose": "stability",
            "arms": {"diagnostic": [0, 1, 2, 3]},
        },
    }
    body["digest"] = prov.digest(body)
    path = tmp_path / "local.json"
    path.write_text(json.dumps(body))
    argv = (
        [kind, "--train", "missing", "--dev", "missing"]
        if kind == "train"
        else [kind, "--init", "missing"]
    )
    argv += [
        "--out",
        str(tmp_path / "out"),
        "--nonqualifying",
        "--local-execution-design",
        str(path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(torch, "manual_seed", lambda *a: pytest.fail("must not seed or fit"))
    with pytest.raises(prov.ProvenanceError, match="three"):
        (train.main if kind == "train" else ppo.main)()


@pytest.mark.parametrize("kind", ["promotion", "clone_evaluate"])
def test_clone_selection_publishers_require_actual_vm_authority(tmp_path, monkeypatch, kind):
    from anthropocene_rl.autonomous_v7 import clone_evaluate, promotion

    spec = {
        "kind": "clone_execution",
        "execution": {"tier": "local", "purpose": "stability", "arms": {"clone": list(range(10))}},
    }
    spec["digest"] = prov.digest(spec)
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(spec))
    argv = [
        kind,
        "--inputs" if kind == "promotion" else "--checkpoint",
        "missing",
        "--calibration-manifest",
        "missing",
        "--out",
        str(tmp_path / "out"),
        "--execution-manifest",
        str(path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        promotion, "promote", lambda *a: pytest.fail("local selection must not execute")
    )
    monkeypatch.setattr(
        clone_evaluate, "evaluate", lambda *a: pytest.fail("local selection must not execute")
    )
    with pytest.raises(prov.ProvenanceError, match="VM"):
        (promotion.main if kind == "promotion" else clone_evaluate.main)()


def test_vm_authority_uses_fixed_host_metadata_not_environment_claims(monkeypatch):
    import urllib.request

    from anthropocene_rl.autonomous_v7.protocol import require_vm_authority

    class Response:
        headers = {"Metadata-Flavor": "Google"}  # noqa: RUF012 - class attribute default kept verbatim

        def __init__(self, text):
            self.text = text

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, limit):
            return self.text.encode()

    class Opener:
        def open(self, request, timeout):
            suffix = request.full_url.split("/computeMetadata/v1/")[1]
            assert request.full_url.startswith("http://metadata.google.internal/")
            return Response(
                {
                    "project/project-id": "project",
                    "instance/id": "12345",
                    "instance/zone": "projects/project/zones/zone-a",
                }[suffix]
            )

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Opener())
    expected = {"vm": {"project_id": "project", "instance_id": "12345", "zone": "zone-a"}}
    require_vm_authority(expected)
    with pytest.raises(prov.ProvenanceError):
        require_vm_authority({"vm": dict(expected["vm"], instance_id="other")})
