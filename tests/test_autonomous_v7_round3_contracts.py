"""Regression tests for the reviewed round-3 architecture, without a native simulator."""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from test_autonomous_v7 import batch
from test_autonomous_v7_round3_search import calibration_rows, fake_tail, village

from anthropocene_rl.autonomous_v7 import (
    branch_shards,
    ppo,
)
from anthropocene_rl.autonomous_v7 import calibration as cal
from anthropocene_rl.autonomous_v7 import checkpoint as ck
from anthropocene_rl.autonomous_v7 import diagnostics as diag
from anthropocene_rl.autonomous_v7 import (
    provenance as prov,
)
from anthropocene_rl.autonomous_v7 import (
    search_teacher as st,
)
from anthropocene_rl.autonomous_v7.model import Actor, Normalizer, to_tensors


def legacy_module(file, name):
    result = subprocess.run(
        ["git", "show", "4f2452e^:src/anthropocene_rl/autonomous_v7/" + file + ".py"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        pytest.skip("pre-head Git baseline unavailable")
    module = types.ModuleType("anthropocene_rl.autonomous_v7." + name)
    module.__file__ = str(Path(__file__))
    sys.modules[module.__name__] = module
    exec(compile(result.stdout, "<pre-head baseline>", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("token", [False, True])
def test_flat_actor_initialization_actions_likelihood_rng_and_optimizer_are_exact(token):
    old_model, old_ppo = (
        legacy_module("model", "_baseline_model"),
        legacy_module("ppo", "_baseline_ppo"),
    )
    b = batch()
    b["region_movement_mask"] = np.ones((6, 10), bool)
    norm = Normalizer.fit(b)
    t = to_tensors(b, norm)
    torch.manual_seed(17)
    old = old_model.Actor(188, token=token)
    old_rng = torch.get_rng_state()
    torch.manual_seed(17)
    new = Actor(188, token=token, region_head="flat")
    new_rng = torch.get_rng_state()
    assert torch.equal(old_rng, new_rng)
    for k, v in old.state_dict().items():
        assert torch.equal(v, new.state_dict()[k]), k
    stage = dataclasses.replace(ppo.STAGES["mixed"], movement=True, region_movement=True)
    o, n = old(t), new(t)
    for k in o:
        assert torch.equal(o[k], n[k]), k
    g1, g2 = torch.Generator().manual_seed(13), torch.Generator().manual_seed(13)
    d1, d2 = old_ppo.sample(o, t, stage, g1), ppo.sample(n, t, stage, g2)
    assert all(torch.equal(d1[k], d2[k]) for k in d1)
    assert torch.equal(g1.get_state(), g2.get_state())
    lp1 = old_ppo.log_prob(o, t, d1, stage)[0]
    lp2 = ppo.log_prob(n, t, d2, stage)[0]
    assert torch.equal(lp1, lp2)
    opt1 = torch.optim.Adam(old.parameters(), lr=1e-5)
    old_state = {"actor": old.state_dict()}
    opt2 = torch.optim.Adam(ck.actor_parameters(new, old_state), lr=1e-5)
    opt2.load_state_dict(opt1.state_dict())
    (-lp1.mean()).backward()
    (-lp2.mean()).backward()
    opt1.step()
    opt2.step()
    for k, v in old.state_dict().items():
        assert torch.equal(v, new.state_dict()[k]), k
    with pytest.raises(ValueError, match="frozen resume"):
        ck.restore_actor(new, old_state, "hierarchical", resume=True)


def test_loader_precedence_and_cost_resume_freeze(monkeypatch):
    monkeypatch.delenv("ANTHROPOCENE_REGION_HEAD", raising=False)
    actor = Actor(188, region_head="flat")
    state = {"actor": actor.state_dict()}
    assert ck.resolve_head(state) == "flat"
    assert ck.resolve_head(state, "hierarchical") == "hierarchical"
    assert ck.resolve_head({}, resume=True) == "flat"
    assert ck.resolve_head({}) == "hierarchical"
    assert ck.resume_setting("cost_bootstrap", None, None, True)
    assert not ck.resume_setting("cost_bootstrap", {}, None, True)
    with pytest.raises(ValueError, match="frozen resume"):
        ck.resume_setting("cost_bootstrap", {"cost_bootstrap": False}, True, True)
    parser = argparse.ArgumentParser()
    ck.arguments(parser)
    assert parser.parse_args(["--flat-region-head"]).region_head == "flat"


def test_frozen_run_probes_live_binary_once_without_actor_rng_or_rollout_state(
    tmp_path, monkeypatch
):
    from anthropocene_rl.autonomous_v7 import bridge

    monkeypatch.delenv("ANTHROPOCENE_SIM_COMMIT", raising=False)
    monkeypatch.setattr(prov, "code_commit", lambda: "a" * 40)
    schema = types.SimpleNamespace(
        build_hash="build-one", simulator_commit="b" * 40, simulator_source_clean=True
    )
    calls = []

    class Probe:
        def __init__(self):
            self.schema = schema
            calls.append("schema")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(bridge, "Model9Bridge", Probe)
    from test_autonomous_v7_round3_protocol import synthetic_worker

    from anthropocene_rl.autonomous_v7 import worker_contract

    monkeypatch.setattr(worker_contract, "observe", lambda bridge: synthetic_worker())
    path = tmp_path / "init.pt"
    path.write_bytes(b"frozen init checkpoint")
    args = argparse.Namespace(init=path, region_head=None, cost_bootstrap=None, resume=False)
    actor = Actor(188)
    rng = torch.get_rng_state().clone()
    frozen = ck.run_manifest(args, actor)
    assert calls == ["schema"] and torch.equal(rng, torch.get_rng_state())
    assert frozen["simulator"] == {"buildHash": "build-one", "commit": "b" * 40, "clean": True}
    assert len(frozen["inputs"]["init"]) == 64 and len(frozen["config_digest"]) == 64
    resume = {"run_manifest": frozen, "cost_bootstrap": True}
    assert ck.run_manifest(args, actor, resume) == frozen
    schema.build_hash = "build-two"
    with pytest.raises(ValueError, match="frozen run"):
        ck.run_manifest(args, actor, resume)
    schema.build_hash = "build-one"
    schema.simulator_commit = "c" * 40
    with pytest.raises(ValueError, match="frozen run"):
        ck.run_manifest(args, actor, resume)
    schema.simulator_source_clean = False
    with pytest.raises(prov.ProvenanceError, match="clean simulator"):
        ck.run_manifest(args, actor)


def test_live_simulator_identity_rejects_unverifiable_pin_or_probe_failure(tmp_path, monkeypatch):
    from anthropocene_rl.autonomous_v7 import bridge

    schema = types.SimpleNamespace(
        build_hash="build", simulator_commit="a" * 40, simulator_source_clean=True
    )
    monkeypatch.setenv("ANTHROPOCENE_SIM_COMMIT", "b" * 40)
    with pytest.raises(prov.ProvenanceError, match="pin"):
        prov.simulator_identity(schema, production=True)
    monkeypatch.delenv("ANTHROPOCENE_SIM_COMMIT")
    schema.simulator_commit = "short"
    with pytest.raises(prov.ProvenanceError, match="full clean"):
        prov.simulator_identity(schema, production=True)

    def failed_probe():
        raise RuntimeError("bridge schema probe failed")

    monkeypatch.setattr(bridge, "Model9Bridge", failed_probe)
    with pytest.raises(RuntimeError, match="probe failed"):
        ck.run_manifest(argparse.Namespace(init=tmp_path / "init.pt"), Actor(188))


def diagnostic(turn=4, camp=7, trajectory="main"):
    values = {
        "strategic": [0.8] * 5,
        "service": [0.8] * 6,
        "clothingRoute": 0,
        "shelterRoute": 1,
        "movement": 101,
        "intentTile": 20,
        "fission": 2,
        "commit": 1,
        "tradeValue": [1.0] * 7,
        "tradeGate": [2] * 7,
        "tradeShare": 0.2,
    }
    masks = {k: [True] * len(v) if isinstance(v, list) else True for k, v in values.items()}
    return {
        "schemaVersion": 1,
        "trajectoryId": trajectory,
        "decisionTurn": turn,
        "campId": camp,
        "requested": values,
        "accepted": dict(values, strategic=[0.5] * 5),
        "realised": dict(values, strategic=[0.3] * 5),
        "requestedMask": masks,
        "acceptedMask": masks,
        "realisedMask": masks,
        "activeMask": masks,
    }


def test_diagnostic_alignment_masks_and_magnitudes_are_not_actor_inputs():
    d = diagnostic()
    step = {"villages": [{"camp": 7, "decisionDiagnostics": d}]}
    records = diag.validate(step, 4, [7], required=True, trajectory="main")
    report = diag.aggregate(records)
    assert report["heads"]["strategic[0]"]["requested_to_accepted"][
        "mean_magnitude"
    ] == pytest.approx(0.3)
    assert report["heads"]["strategic[0]"]["accepted_to_realised"][
        "mean_magnitude"
    ] == pytest.approx(0.2)
    for turn, camps, trajectory in [(5, [7], "main"), (4, [8], "main"), (4, [7], "other")]:
        with pytest.raises(ValueError):
            diag.validate(step, turn, camps, required=True, trajectory=trajectory)
    from anthropocene_rl.autonomous_v7.bridge import observation_batch

    v = village()
    before = observation_batch([v])
    v["decisionDiagnostics"] = d
    after = observation_batch([v])
    assert before.keys() == after.keys() and all(
        np.array_equal(before[k], after[k]) for k in before
    )


def test_cost_bootstrap_uses_explicit_shuffled_endpoint_mapping_and_zero_terminal():
    b = batch()
    normalizer = Normalizer.fit(b)

    class Critic:
        def __call__(self, t):
            return torch.tensor([2.0, 7.0, 11.0])

    roll = ppo.Rollout()
    roll.bootstrap_obs = [{k: v[:3] for k, v in b.items()}]
    rows = {"cost": np.zeros(4), "bootstrap_index": np.array([2, -1, 0, 1])}
    assert np.array_equal(
        ppo.cost_bootstrap(Critic(), roll, rows, normalizer, np.array([0, 2, 3])), [11, 0, 2, 7]
    )
    assert not ppo.cost_bootstrap(
        Critic(), ppo.Rollout(), rows, normalizer, np.array([], int)
    ).any()


def test_provenance_failed_git_never_claims_clean_and_dirty_sources_cannot_train(monkeypatch):
    def fail(*a, **k):
        raise subprocess.CalledProcessError(1, a[0])

    monkeypatch.setattr(prov.subprocess, "run", fail)
    assert prov.code_commit() == prov.UNKNOWN
    record = prov.source(None, {"tail": 12}, {"buildHash": "build", "commit": "a" * 40})
    with pytest.raises(prov.ProvenanceError):
        prov.validate_records([record], "reject")
    assert not prov.validate_records([record], "allow")["identity_valid"]
    with pytest.warns(UserWarning):
        prov.check([("a", "b", "c"), ("d", "e", "f")], "flag")


def test_declared_source_mix_stays_strict_about_unknown_or_undeclared_sources():
    sources = [("a", "b", "c" * 64), ("d", "e", "f" * 64)]
    with pytest.raises(prov.ProvenanceError):
        prov.check(sources, "reject")
    assert prov.check(sources, "reject", declared={"c" * 64, "f" * 64})["ok"]
    with pytest.raises(prov.ProvenanceError):
        prov.check(sources, "reject", declared={"c" * 64})
    with pytest.raises(prov.ProvenanceError):
        prov.check([(prov.UNKNOWN, "b", "c" * 64)], "reject", declared={"c" * 64})


def test_mixed_legacy_labels_preserve_new_personality_and_observation_rows(tmp_path):
    b = batch(2)
    common = dict(
        b,
        label_kind=np.full(2, st.REGION),
        label_slot=np.full(2, 4),
        label_mix=np.full((2, 3), np.nan),
        score_policy=np.ones(2),
        score_best=np.full(2, 1.1),
    )
    old = {k: v for k, v in common.items() if k not in ("regions", "regions_present")}
    new = dict(common, personality=np.tile([0.0, 1.0, 0.0, 0.0], (2, 1)))
    np.savez(tmp_path / "old.npz", **old)
    np.savez(tmp_path / "new.npz", **new)
    loaded = ppo.load_search_labels(
        [tmp_path / "old.npz", tmp_path / "new.npz"], 0.004, provenance="allow"
    )
    assert np.array_equal(loaded["personality"][2:], new["personality"])
    assert not loaded["regions_present"][:2].any()
    assert np.array_equal(loaded["regions"][2:], b["regions"])


def test_branch_loader_rebases_roots_and_rejects_duplicate_trajectories(tmp_path):
    for name in ("first", "second"):
        branch_shards.write(
            tmp_path / name, fake_tail(0), {"seed": 1000, "world_id": 123, "kind": st.NATIVE}
        )
    rows, _ = branch_shards.load([tmp_path / "first", tmp_path / "second"], "allow")
    assert len(np.unique(rows["world"])) == 2
    with pytest.raises(prov.ProvenanceError, match="duplicate"):
        branch_shards.load([tmp_path / "first", tmp_path / "first"], "allow")
    assert branch_shards.world_id(1000, 0, 0, 0, 1) != branch_shards.world_id(1000, 0, 64, 0, 1)


def test_same_seed_daughter_starts_write_distinct_attributable_trajectories(tmp_path, monkeypatch):
    monkeypatch.setattr(prov, "code_commit", lambda: "a" * 40)
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"frozen actor")
    source = prov.source(
        checkpoint, {}, {"buildHash": "same-build", "commit": "b" * 40, "clean": True}
    )
    worlds = [
        {
            "seed": 1000,
            "withheld": 0,
            "warmup": 0,
            "entry": entry,
            "snapshot_hash": prov.digest(entry),
            "rows": [
                {
                    "turn": 3,
                    "village": village(),
                    "personality": [1.0, 0, 0, 0],
                    "alternatives": [{"kind": st.NATIVE, "slot": 0}],
                    "results": [{"tail": fake_tail(entry), "calibration": False}],
                }
            ],
        }
        for entry in (1, 2)
    ]
    root = tmp_path / "branches"
    result = st.write_branch_shards(worlds, root, source)
    assert result["branch_shards"] == 2 and len(list(root.iterdir())) == 2
    report = {}
    rows, _ = branch_shards.load([root], "reject", report)
    assert report["identity_valid"] and report["ok"]
    assert set(rows["source_seed"].tolist()) == {1000} and len(np.unique(rows["world"])) == 2
    path = next(root.iterdir()) / "meta.json"
    metadata = json.loads(path.read_text())
    metadata["bank_entry"] = 99
    path.write_text(json.dumps(metadata))
    with pytest.raises(prov.ProvenanceError, match="trajectory metadata"):
        branch_shards.load([root], "reject")


def test_calibration_manifest_family_exclusions_and_waivers():
    m = cal.manifest(
        {"calibration_seeds": [1500], "training_seeds": [1000], "selection_seeds": [2000]}
    )
    assert cal.held_out(m, 1500)
    with pytest.raises(prov.ProvenanceError):
        cal.assert_training(m, [1500])
    with pytest.raises(prov.ProvenanceError):
        cal.manifest(dict(m, training_seeds=[1500]))
    with pytest.raises(prov.ProvenanceError):
        cal.manifest(dict(m, waivers=["native/welfare"]))


def test_training_partition_uses_source_seed_and_snapshot_not_rebased_world_ids():
    m = cal.manifest(
        {
            "calibration_seeds": [1500],
            "calibration_snapshots": ["held"],
            "training_seeds": [1000],
            "selection_seeds": [2000],
        }
    )
    allowed = {
        "world": np.array([1500]),
        "source_seed": np.array([1000]),
        "source_snapshot": np.array([""]),
    }
    cal.assert_training_rows(
        m, allowed
    )  # the sequence number happens to equal a sealed source seed
    with pytest.raises(prov.ProvenanceError):
        cal.assert_training_rows(
            m, dict(allowed, world=np.array([1 << 56]), source_seed=np.array([1500]))
        )
    with pytest.raises(prov.ProvenanceError):
        cal.assert_training_rows(m, dict(allowed, source_snapshot=np.array(["held"])))
    with pytest.raises(prov.ProvenanceError, match="source_seed"):
        cal.assert_training_rows(m, {"world": np.array([1000])})


def test_calibration_short_window_coverage_matches_calibration_and_endpoint_is_required():
    rows = calibration_rows(turns=8)
    assert cal.coverage(rows, warn=False)["branches"]["native"] == 0
    assert cal.calibration({"welfare": np.ones(8)}, rows)["short"] == 1
    rows["endpoint_turn"] = np.full(8, 8)
    with pytest.raises(ValueError, match="endpoint"):
        cal.calibration({"welfare": np.ones(8)}, rows)
    assert cal.coverage(rows, warn=False)["branches"]["native"] == 1
    result = cal.calibration({"welfare": np.ones(8)}, rows, endpoints={(77, 8, "welfare"): 1.0})
    assert result["branches"] == 1


def test_mixed_near_zero_gate_and_mandatory_coverage_cannot_silently_pass():
    scale = ppo.contracts.OBJECTIVE_REFERENCE_SCALES["welfare"]
    bad = cal.statistics([0.2 * scale] + [scale] * 20, [0.0] + [scale] * 20, "welfare")
    assert bad["relative_pass"] and not bad["near_pass"]
    good = cal.statistics([scale] * 20, [scale] * 20, "welfare")
    report = {"kinds": {k: {"welfare": good} for k in ("native", "region", "mix")}}
    assert cal.verdict(report, {"warnings": []})["pass"]
    report["kinds"]["native"]["welfare"] = bad
    assert not cal.verdict(report, {"warnings": []})["pass"]
    assert not cal.verdict({"kinds": {"native": {"welfare": good}}}, {"warnings": []})["pass"]


def test_nonmandatory_populated_relative_subset_is_gated_unless_exactly_waived():
    scale = ppo.contracts.OBJECTIVE_REFERENCE_SCALES["welfare"]
    good = cal.statistics([scale] * 20, [scale] * 20, "welfare")
    trade_scale = ppo.contracts.OBJECTIVE_REFERENCE_SCALES["trade_utility"]
    bad_trade = cal.statistics([trade_scale * 1.2] * 20, [trade_scale] * 20, "trade_utility")
    report = {"kinds": {k: {"welfare": good} for k in ("native", "region", "mix")}}
    report["kinds"]["trade"] = {"trade_utility": bad_trade}
    assert not cal.verdict(report, {"warnings": []})["pass"]
    assert cal.verdict(report, {"warnings": []}, ["trade/trade_utility"])["pass"]
    assert not cal.verdict(report, {"warnings": []}, ["fission/trade_utility"])["pass"]


def test_forced_calibration_catches_the_only_early_rare_alternative():
    alts = [{"kind": st.TRADE}, {"kind": st.POLICY}]
    flags = st.ensure_kind_coverage(alts, [False, False], {}, 1, {st.TRADE: 20}, 1, [1000, 1])
    assert flags == [True, True]


def test_live_diagnostics_preserve_invalid_request_repair_and_fork_alignment():
    from anthropocene_rl.autonomous_v7.bridge import DEFAULT_BRIDGE, Model9Bridge

    if not DEFAULT_BRIDGE.exists():
        pytest.skip("model9 bridge not built")
    with Model9Bridge() as bridge:
        if bridge.schema.decision_diagnostics_schema < 1:
            pytest.skip("historical schema0 bridge")
        step = bridge.reset(1000, movement=True)
        assert all(v.get("decisionDiagnostics") is None for v in step["villages"])
        camp = next(v["camp"] for v in step["villages"] if v.get("observation") is not None)
        action = {
            "strategic": [1.2, -0.2, 0.4, 0.5, 0.5],
            "service": [1.2] * 6,
            "clothingRoute": 99,
            "shelterRoute": 99,
            "movement": 999999,
            "fission": 99,
            "commit": 99,
            "tradeValue": [1.0] * 7,
            "tradeGate": [2] * 7,
            "tradeShare": 0.2,
        }
        after = bridge.step({camp: action})
        d = bridge.decision_diagnostics[0]
        assert d["requested"]["strategic"][0] == 1.2 and d["accepted"]["strategic"][0] <= 1.0
        assert d["campId"] == camp and d["decisionTurn"] == step["turn"]
        fork, saved = bridge.fork()
        main_diagnostics = json.loads(json.dumps(bridge.decision_diagnostics))
        assert (
            next(v for v in saved["villages"] if v["camp"] == camp)["decisionDiagnostics"]
            == next(v for v in after["villages"] if v["camp"] == camp)["decisionDiagnostics"]
        )
        first = bridge.step({camp: action})
        branch_diagnostics = json.loads(json.dumps(bridge.decision_diagnostics))
        bridge.restore(fork)
        assert bridge.decision_diagnostics == main_diagnostics
        assert bridge.decision_diagnostics != branch_diagnostics
        second = bridge.step({camp: action})
        assert first["fingerprint"] == second["fingerprint"]
        bridge.drop(fork)


def search_label_fixture(seed=1000, snapshot="", entry=-1):
    return dict(
        batch(1),
        label_kind=np.array([st.REGION]),
        label_slot=np.array([4]),
        label_mix=np.full((1, 3), np.nan),
        score_policy=np.array([1.0]),
        score_best=np.array([1.1]),
        source_run=np.array(["f" * 64]),
        seed=np.array([seed]),
        withheld=np.array([0]),
        warmup=np.array([0]),
        bank_entry=np.array([entry]),
        snapshot_hash=np.array([snapshot]),
        turn=np.array([4]),
        camp=np.array([7]),
    )


def test_duplicate_search_decisions_fail_before_winner_filtering_and_across_files(tmp_path):
    p = search_label_fixture()
    np.savez(tmp_path / "one.npz", **p)
    np.savez(tmp_path / "other.npz", **dict(p, label_kind=np.array([st.MIX])))
    with pytest.raises(prov.ProvenanceError, match="duplicate search decision"):
        ppo.load_search_labels(
            [tmp_path / "one.npz", tmp_path / "other.npz"], 100.0, provenance="allow"
        )
    doubled = {k: np.concatenate([v, v]) for k, v in p.items()}
    np.savez(tmp_path / "duplicates.npz", **doubled)
    with pytest.raises(prov.ProvenanceError, match="duplicate"):
        ppo.load_search_labels([tmp_path / "duplicates.npz"], 0.004, provenance="allow")


def test_current_manifest_checks_label_and_branch_families_even_if_historical_flags_allow(tmp_path):
    held = cal.manifest(
        {
            "calibration_seeds": [1500],
            "calibration_snapshots": [prov.digest("held-snapshot")],
            "training_seeds": [1000],
            "selection_seeds": [2000],
        }
    )
    for index, p in enumerate(
        (search_label_fixture(1500), search_label_fixture(1000, prov.digest("held-snapshot"), 3))
    ):
        path = tmp_path / f"label{index}.npz"
        np.savez(path, **dict(p, heldout_calibration=np.array([False])))
        with pytest.raises(prov.ProvenanceError, match="held-out"):
            ppo.load_search_labels([path], 0.004, provenance="allow", calibration_manifest=held)
    branch_shards.write(
        tmp_path / "old-branch",
        fake_tail(0),
        {"seed": 1500, "world_id": 123, "kind": st.NATIVE, "heldout_calibration": False},
    )
    with pytest.raises(prov.ProvenanceError, match="held-out"):
        branch_shards.load([tmp_path / "old-branch"], "allow", calibration_manifest=held)
    legacy = search_label_fixture()
    legacy.pop("seed")
    path = tmp_path / "unattributable.npz"
    np.savez(path, **legacy)
    with pytest.raises(prov.ProvenanceError, match="identity"):
        ppo.load_search_labels([path], 0.004, provenance="allow", calibration_manifest=held)


def test_same_seed_distinct_snapshot_decisions_are_not_false_duplicates(tmp_path):
    paths = []
    for entry in (1, 2):
        path = tmp_path / f"bank{entry}.npz"
        np.savez(path, **search_label_fixture(1000, prov.digest(entry), entry))
        paths.append(path)
    rows = ppo.load_search_labels(paths, 0.004, provenance="allow")
    assert len(set(rows["decision_id"].tolist())) == 2


@pytest.mark.parametrize(
    "head,policy", [("flat", "grouped"), ("hierarchical", "argmax"), ("hierarchical", "native")]
)
def test_movement_audit_preserves_loader_head_and_records_diagnostic_selection(
    tmp_path, monkeypatch, head, policy
):
    from anthropocene_rl.autonomous_v7 import movement_audit as audit

    actor = Actor(188, region_head=head)
    monkeypatch.setattr(audit, "load", lambda path: (actor, None))

    class Bridge:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def reset(self, *args, **kwargs):
            return {"villages": []}

        def verify_replay(self):
            return 0

    monkeypatch.setattr(audit, "Model9Bridge", Bridge)
    from anthropocene_rl.autonomous_v7 import worker_contract

    monkeypatch.setattr(
        worker_contract, "validate", lambda *a, **kw: None
    )  # separate pre-execution regressions cover this boundary
    monkeypatch.setattr(audit, "producer_identity", lambda bridge: {"synthetic": True})
    monkeypatch.setattr(audit, "inspect", lambda *args: {"turn": 0, "camps": [], "tiles": []})
    result = audit.audit_rollout(1000, 0, "actor.pt", policy, str(tmp_path))
    assert actor.region_head == head and result["region_head"] == head
    assert result["region_selection"] == ("argmax" if policy == "argmax" else None)


def checkpoint_proof_fixture():
    from test_autonomous_v7_round3_protocol import native_proof

    from anthropocene_rl.autonomous_v7 import lineage

    inputs = {"training": native_proof(1000), "selection": native_proof(2000)}
    held = cal.manifest(
        {"calibration_seeds": [1500], "training_seeds": [1000], "selection_seeds": [2000]}
    )
    config = {"options": {}, "sim_rules": "", "sim_statics": ""}
    state = {
        "contract": ppo.contracts.NATIVE_CONTRACT,
        "calibration_manifest": held,
        "run_manifest": {
            **config,
            "config_digest": prov.digest(config),
            "code": "a" * 40,
            "simulator": {"commit": "b" * 40, "buildHash": "c" * 32, "clean": True},
            "contract": ppo.contracts.NATIVE_CONTRACT,
            "utility_version": 2,
            "bridge_sha256": "d" * 64,
            "teacher_inputs": {k: v["digest"] for k, v in inputs.items()},
            "simulator_compatibility": [],
            "calibration_manifest_digest": held["manifest_digest"],
            "initialization": "random",
            "inputs": {},
        },
        "family_lineage": lineage.build(
            held, lineage.families([1000]), lineage.families([2000]), teacher_inputs=inputs
        ),
    }
    return held, state


def test_calibration_requires_checkpoint_manifest_and_frozen_initialization_lineage(tmp_path):
    import copy

    from anthropocene_rl.autonomous_v7 import lineage

    held, state = checkpoint_proof_fixture()
    assert lineage.checkpoint_evidence(state, held)["verified"]
    for missing in ("calibration_manifest", "family_lineage", "run_manifest"):
        bad = dict(state)
        bad.pop(missing)
        assert not lineage.checkpoint_evidence(bad, held)["verified"]
    bad = copy.deepcopy(state)
    bad["calibration_manifest"] = dict(held, calibration_seeds=[1501])
    assert not lineage.checkpoint_evidence(bad, held)["verified"]
    bad = copy.deepcopy(state)
    bad["run_manifest"]["inputs"]["init"] = "d" * 64
    assert not lineage.checkpoint_evidence(bad, held)["verified"]
    bad = copy.deepcopy(state)
    bad["run_manifest"]["inputs"]["teacher"] = "e" * 64
    assert not lineage.checkpoint_evidence(bad, held)["verified"]
    descendant = copy.deepcopy(state)
    parent_path = tmp_path / "parent.pt"
    torch.save(state, parent_path)
    parent_ref = {"path": str(parent_path), "sha256": prov.checkpoint_hash(parent_path)}
    descendant["run_manifest"]["inputs"]["init"] = parent_ref["sha256"]
    descendant["run_manifest"]["initialization"] = "checkpoint"
    descendant["family_lineage"] = lineage.build(
        held,
        lineage.families([1001]),
        lineage.families([2001]),
        [(parent_ref["sha256"], state, parent_ref)],
    )
    assert lineage.checkpoint_evidence(descendant, held)["verified"]
    bad_parent = copy.deepcopy(state)
    bad_parent["family_lineage"]["training"] = lineage.families([1500])
    bad_parent["family_lineage"]["digest"] = prov.digest(
        {k: v for k, v in bad_parent["family_lineage"].items() if k != "digest"}
    )
    with pytest.raises(prov.ProvenanceError, match="held-out"):
        lineage.build(
            held, lineage.families([1001]), lineage.families([]), [("d" * 64, bad_parent)]
        )
    descendant["family_lineage"]["parents"][0]["proof"] = lineage.proof(bad_parent)
    descendant["family_lineage"]["digest"] = prov.digest(
        {k: v for k, v in descendant["family_lineage"].items() if k != "digest"}
    )
    assert not lineage.checkpoint_evidence(descendant, held)["verified"]


def clean_source(simulator):
    body = {
        "source_checkpoint": "d" * 64,
        "code_commit": "a" * 40,
        "provenance_version": 2,
        "contract": ppo.contracts.NATIVE_CONTRACT,
        "utility_version": 2,
        "search_config": {},
        "config_digest": prov.digest({}),
        "simulator": simulator,
    }
    return dict(body, source_run=prov.digest(body))


def test_offline_source_union_and_live_binary_are_compatible_only_when_equal():
    a = {"commit": "a" * 40, "buildHash": "1" * 32, "clean": True}
    b = {"commit": "b" * 40, "buildHash": "2" * 32, "clean": True}
    assert prov.compatible_sources([clean_source(a), clean_source(a)], a)["compatible"]
    with pytest.raises(prov.ProvenanceError, match="mix incompatible"):
        prov.compatible_sources([clean_source(a), clean_source(b)], a)
    with pytest.raises(prov.ProvenanceError, match="frozen live"):
        prov.compatible_sources([clean_source(a)], b)
    with pytest.raises(prov.ProvenanceError, match="flags differ"):
        prov.compatible_sources([clean_source(a)], a, flags={"sim_rules": "", "sim_statics": ""})
    source = clean_source(a)
    source["search_config"] = {"sim_rules": "", "sim_statics": ""}
    source["config_digest"] = prov.digest(source["search_config"])
    source["source_run"] = prov.digest({k: v for k, v in source.items() if k != "source_run"})
    assert prov.compatible_sources([source], a, flags=source["search_config"])["compatible"]


def test_snapshot_only_schema2_trace_fails_closed(tmp_path):
    from anthropocene_rl.autonomous_v7.semantic_trace import (
        FINGERPRINT_CONTRACT,
        NORMALIZATION,
        verify,
    )

    identity = {"commit": "a" * 40, "buildHash": "1" * 32, "clean": True}
    trace = {
        "schema": 2,
        "kind": "simulator_semantic_trace",
        "simulator": identity,
        "flags": {"sim_rules": "", "sim_statics": ""},
        "seed": 2000,
        "turns": 1,
        "normalization": NORMALIZATION,
        "fingerprint_contract": FINGERPRINT_CONTRACT,
        "frames": [{"turn": 0}, {"turn": 1}],
    }
    with pytest.raises(prov.ProvenanceError, match="replayable complete-save trace"):
        verify(
            trace, tmp_path, identity, trace["flags"], {"seed": 2000, "turns": 1}, "source", set()
        )


def test_restore_retains_exact_public_diagnostics_after_partial_action_submission():
    from anthropocene_rl.autonomous_v7.bridge import Model9Bridge

    bridge = Model9Bridge.__new__(Model9Bridge)
    bridge.schema = types.SimpleNamespace(decision_diagnostics_schema=1)
    main = {
        "turn": 4,
        "villages": [
            {"camp": 1, "decisionDiagnostics": {"campId": 1}},
            {"camp": 2, "decisionDiagnostics": {"campId": 2}},
        ],
    }
    bridge._last_step = main
    bridge._fork_diagnostics = {}
    bridge.decision_diagnostics = [{"campId": 1}]  # only camp 1 was explicitly submitted
    bridge.request = lambda payload: {"fork": 7, "step": main}
    fork, _ = bridge.fork()
    bridge.decision_diagnostics[0]["campId"] = 99  # ensure the saved list is not aliased
    bridge.restore(fork)
    assert bridge.decision_diagnostics == [{"campId": 1}]
    bridge.decision_diagnostics.clear()
    bridge.restore(fork)
    assert bridge.decision_diagnostics == [{"campId": 1}]


def test_teacher_optional_normalization_preserves_newer_conditioning(monkeypatch, tmp_path):
    from anthropocene_rl.autonomous_v7 import data

    old = dict(batch(1), camp=np.array([1]), trade_utility=np.array([0.0]))
    new = dict(old, trade=np.ones_like(old["trade"]), personality=np.array([[2.0, 1.0, 0.0, 0.0]]))
    old.pop("trade")
    old.pop("trade_present")
    old.pop("personality", None)
    for name in ("old", "new"):
        path = tmp_path / name
        path.mkdir()
        (path / "meta.json").write_text("{}")
    monkeypatch.setattr(data, "load_world", lambda path: old if path.name == "old" else new)
    loaded = data.load([tmp_path / "old", tmp_path / "new"])
    assert loaded["trade"].shape[0] == 2 and loaded["trade"].sum() == new["trade"].sum()
    assert any(np.array_equal(p, new["personality"][0]) for p in loaded["personality"])


def test_search_family_report_keeps_filtered_input_families(tmp_path):
    path = tmp_path / "loser.npz"
    np.savez(path, **search_label_fixture())
    report = {}
    assert (
        ppo.load_search_labels(
            [path], 100.0, legacy_margin=100.0, provenance="allow", report=report
        )
        is None
    )
    assert report["families"] == {"seeds": [1000], "snapshots": []}


def test_stage1_saves_frozen_lineage_and_phase_resume_preserves_head(tmp_path, monkeypatch):
    import copy

    from anthropocene_rl.autonomous_v7 import lineage, train

    held, proof = checkpoint_proof_fixture()
    for name in ("train", "dev"):
        (tmp_path / name).mkdir()
    manifest_path = tmp_path / "held.json"
    manifest_path.write_text(json.dumps(held))
    datasets = {
        "train": dict(batch(1), source_seed=np.array([1000]), source_snapshot=np.array([""])),
        "dev": dict(batch(1), source_seed=np.array([2000]), source_snapshot=np.array([""])),
    }
    # main passes a generator; data loading itself is separately covered by native shard tests.
    calls = []

    # Use an explicit small loader rather than train real epochs in this integration-contract test.
    def load_data(paths, **kwargs):
        index = len(calls) % 2
        calls.append(index)
        kwargs["report"].update(
            copy.deepcopy(
                proof["family_lineage"]["teacher_inputs"][("training", "selection")[index]]
            )
        )
        return datasets[("train", "dev")[index]]

    monkeypatch.setattr(train.data, "load", load_data)
    monkeypatch.setattr(train.data, "summary", lambda rows: {})
    monkeypatch.setattr(train, "train_actor", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "train_critic", lambda *args, **kwargs: {})
    frozen_heads = []

    def freeze(args, actor, **kwargs):
        frozen_heads.append(actor.region_head)
        return copy.deepcopy(proof["run_manifest"])

    monkeypatch.setattr(ck, "run_manifest", freeze)
    # Loader/resume fitting and queue authority are mocked; entrypoint boundaries have separate regressions.
    from anthropocene_rl.autonomous_v7 import protocol

    monkeypatch.setattr(
        protocol, "enforce_clone_execution", lambda args: {"selection_allowed": True}
    )
    monkeypatch.setenv("ANTHROPOCENE_REGION_HEAD", "hierarchical")
    argv = [
        "train",
        "--train",
        str(tmp_path / "train"),
        "--dev",
        str(tmp_path / "dev"),
        "--out",
        str(tmp_path / "actor.pt"),
        "--calibration-manifest",
        str(manifest_path),
        "--flat-region-head",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert train.main() == 0
    state = torch.load(tmp_path / "actor.pt", map_location="cpu", weights_only=False)
    assert state["family_lineage"]["training"]["seeds"] == [1000]
    assert state["family_lineage"]["selection"]["seeds"] == [2000]
    assert lineage.checkpoint_evidence(state, held)["verified"]
    # Saved head wins when there is no explicit override; freeze occurs after this resolution.
    monkeypatch.setenv("ANTHROPOCENE_REGION_HEAD", "flat")
    monkeypatch.setattr(sys, "argv", argv[:-1])
    assert train.main() == 0 and frozen_heads == ["flat", "flat"]
    monkeypatch.setattr(sys, "argv", argv[:-1] + ["--region-head", "hierarchical"])
    with pytest.raises(ValueError, match="frozen resume"):
        train.main()


def test_labels_obey_embedded_historical_exclusions_and_bind_row_manifest_digest(tmp_path):
    held, proof = checkpoint_proof_fixture()
    source = clean_source(proof["run_manifest"]["simulator"])
    source["search_config"] = {"calibration_manifest": held}
    source["config_digest"] = prov.digest(source["search_config"])
    source["source_run"] = prov.digest({k: v for k, v in source.items() if k != "source_run"})

    def label(seed, binding):
        return dict(
            search_label_fixture(seed),
            source_run=np.array([source["source_run"]]),
            source_checkpoint=np.array([source["source_checkpoint"]]),
            code_commit=np.array([source["code_commit"]]),
            source_record=np.array([json.dumps(source)]),
            calibration_manifest_digest=np.array([binding]),
            heldout_calibration=np.array([False]),
        )

    path = tmp_path / "labels.npz"
    np.savez(path, **label(1000, "wrong"))
    with pytest.raises(prov.ProvenanceError, match="manifest differs"):
        ppo.load_search_labels([path], 0.004, provenance="reject")
    np.savez(path, **label(1500, held["manifest_digest"]))
    with pytest.raises(prov.ProvenanceError, match="held-out"):
        ppo.load_search_labels([path], 0.004, provenance="reject")
    np.savez(path, **label(1000, held["manifest_digest"]))
    assert ppo.load_search_labels([path], 0.004, provenance="reject") is not None


def physical_step(turn=5, trajectory="main", requested=7.0, accepted=3.0, realised=2.0):
    return {
        "turn": turn,
        "diagnosticSchemaVersion": 2,
        "trajectoryId": trajectory,
        "rawMaterialDiagnostics": {
            "7:3": {
                "schemaVersion": 2,
                "campId": 7,
                "decisionTurn": turn - 1,
                "material": 3,
                "requestId": f"elder-raw:{turn - 1}:7:3",
                "requestedUnits": requested,
                "acceptedUnits": accepted,
                "realisedUnits": realised,
            }
        },
        "villages": [
            {
                "camp": 7,
                "welfare": 17.0,
                "observation": village()["observation"],
                "decisionDiagnostics": diagnostic(turn - 1, 7, trajectory),
            }
        ],
    }


def test_physical_sidecar_units_validate_align_and_survive_step_fork_restore_without_actor_inputs():
    import copy

    from anthropocene_rl.autonomous_v7.bridge import Model9Bridge, observation_batch

    bridge = Model9Bridge.__new__(Model9Bridge)
    bridge.schema = types.SimpleNamespace(
        decision_diagnostics_schema=1, raw_material_diagnostics_schema=2
    )
    bridge._turn = 4
    bridge._trajectory = "main"
    bridge._fork_diagnostics = {}
    main, branch = physical_step(), physical_step(6, "branch", 11.0, 5.0, 1.0)
    bridge.request = lambda payload: {"fork": 4, "step": copy.deepcopy(main)}
    step = bridge.step({7: {}})
    expected = copy.deepcopy(bridge.raw_material_diagnostics)
    assert list(expected.values())[0]["requestedUnits"] == 7.0
    assert list(expected.values())[0]["acceptedUnits"] == 3.0
    assert list(expected.values())[0]["realisedUnits"] == 2.0
    before = observation_batch(step["villages"])
    altered = copy.deepcopy(step)
    altered["rawMaterialDiagnostics"] = physical_step(requested=99.0)["rawMaterialDiagnostics"]
    after = observation_batch(altered["villages"])
    assert all(np.array_equal(before[k], after[k]) for k in before)
    assert step["villages"][0]["welfare"] == altered["villages"][0]["welfare"] == 17.0
    assert not any("diagnostic" in k.lower() or "units" in k.lower() for k in before)
    fork, _ = bridge.fork()
    bridge.request = lambda payload: {
        "step": copy.deepcopy(branch if payload["op"] == "step" else main)
    }
    bridge.step({7: {}})
    assert bridge.raw_material_diagnostics != expected
    restored = bridge.restore(fork)
    assert bridge.raw_material_diagnostics == expected == restored["rawMaterialDiagnostics"]
    bridge._read_physical_diagnostics(
        {
            "turn": 0,
            "villages": [],
            "diagnosticSchemaVersion": 0,
            "trajectoryId": None,
            "rawMaterialDiagnostics": {},
        },
        initial=True,
    )
    assert bridge.raw_material_diagnostics == {} and bridge.diagnostic_schema_version == 0


@pytest.mark.parametrize("mutation", ["key", "turn", "trajectory", "schema", "overgrant", "nan"])
def test_physical_sidecar_rejects_misaligned_or_invalid_receipts(mutation):
    step = physical_step()
    record = step["rawMaterialDiagnostics"]["7:3"]
    if mutation == "key":
        step["rawMaterialDiagnostics"] = {"8:3": record}
    elif mutation == "turn":
        record["decisionTurn"] = 3
    elif mutation == "trajectory":
        step["trajectoryId"] = "other"
    elif mutation == "schema":
        step["diagnosticSchemaVersion"] = 0
    elif mutation == "overgrant":
        record["acceptedUnits"] = 8.0
    else:
        record["realisedUnits"] = float("nan")
    with pytest.raises(ValueError, match="physical"):
        diag.validate_raw_materials(step, 4, required=True)


def test_native_physical_receipts_restore_without_submitted_head_sidecars():
    import copy

    from anthropocene_rl.autonomous_v7.bridge import Model9Bridge

    bridge = Model9Bridge.__new__(Model9Bridge)
    bridge.schema = types.SimpleNamespace(
        decision_diagnostics_schema=1, raw_material_diagnostics_schema=2
    )
    bridge._turn = 4
    bridge._trajectory = None
    bridge._fork_diagnostics = {}
    main, branch = physical_step(), physical_step(6, "branch", 11.0, 5.0, 1.0)
    for step in (main, branch):
        step["villages"][0]["decisionDiagnostics"] = None
    bridge.request = lambda payload: {"fork": 4, "step": copy.deepcopy(main)}
    bridge.step({})
    assert not bridge.decision_diagnostics and bridge.raw_material_diagnostics
    fork, _ = bridge.fork()
    bridge.request = lambda payload: {
        "step": copy.deepcopy(branch if payload["op"] == "step" else main)
    }
    bridge.step({})
    bridge.restore(fork)
    assert bridge.raw_material_diagnostics == main["rawMaterialDiagnostics"]
    assert not bridge.decision_diagnostics and bridge.diagnostic_schema_version == 2
    assert bridge.diagnostic_trajectory_id == "main"


def test_live_snapshot_validation_binds_native_fingerprint_without_resetting_live_world():
    from anthropocene_rl.autonomous_v7.bridge import DEFAULT_BRIDGE, Model9Bridge

    if not DEFAULT_BRIDGE.exists():
        pytest.skip("model9 bridge not built")
    with Model9Bridge() as bridge:
        step = bridge.reset(1001, movement=True)
        for _ in range(2):
            step = bridge.step(
                {}
            )  # include saved turn history/receipts, not only an opening header
        before = bridge.request({"op": "inspect"})["fingerprint"]
        exported = bridge.request({"op": "fingerprint_snapshot"})
        snapshot = exported["snapshot"]
        checked = bridge.request({"op": "validate_fingerprint_snapshot", "snapshot": snapshot})
        assert checked["sourceSeed"] == 1001 and checked["turn"] == step["turn"]
        import base64

        from anthropocene_rl.autonomous_v7.semantic_trace import (
            FINGERPRINT_CONTRACT,
            complete_payload,
        )

        assert exported["fingerprintContract"] == FINGERPRINT_CONTRACT
        assert checked["payloadFingerprint"] == exported["fingerprint"]
        assert (
            complete_payload(base64.b64decode(snapshot))[1]["normalized_payload_sha256"]
            == exported["fingerprint"].lower()
        )
        assert bridge.request({"op": "inspect"})["fingerprint"] == before == exported["fingerprint"]


def test_live_latent_physical_sidecar_authorizes_zero_and_fork_restore_preserves_it(monkeypatch):
    import copy

    from anthropocene_rl.autonomous_v7.bridge import DEFAULT_BRIDGE, Model9Bridge, observation_batch

    if not DEFAULT_BRIDGE.exists():
        pytest.skip("model9 bridge not built")
    monkeypatch.setenv("ANTHROPOCENE_SIM_RULES", "ElderRawMaterialsEnabled=true")
    with Model9Bridge() as bridge:
        if bridge.schema.raw_material_diagnostics_schema < 2:
            pytest.skip("historical physical schema bridge")
        start = bridge.reset(1000, movement=True)
        assert not bridge.raw_material_diagnostics
        actions = {
            v["camp"]: {"service": [0, 0, 0, 0, 0, 1], "movement": -2}
            for v in start["villages"]
            if v.get("observation")
        }
        main = bridge.step(actions)
        receipts = copy.deepcopy(bridge.raw_material_diagnostics)
        # Native fallback owns every residual worker. Positive kernel receipts are not a normal-turn claim.
        assert all(r["acceptedUnits"] == 0 and r["realisedUnits"] == 0 for r in receipts.values())
        assert all(
            r["decisionTurn"] == start["turn"] and r["schemaVersion"] == 2
            for r in receipts.values()
        )
        before = observation_batch([v for v in main["villages"] if v.get("observation")])
        assert not any("diagnostic" in k.lower() or "units" in k.lower() for k in before)
        fork, saved = bridge.fork()
        assert saved["rawMaterialDiagnostics"] == receipts
        bridge.step(actions)
        assert bridge.diagnostic_schema_version == 2
        restored = bridge.restore(fork)
        assert restored["rawMaterialDiagnostics"] == receipts == bridge.raw_material_diagnostics
        assert bridge.diagnostic_trajectory_id == main["trajectoryId"]
        bridge.drop(fork)
