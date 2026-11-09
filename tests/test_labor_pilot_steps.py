"""labor_pilot stage checks used by ops/r5/pilot/ppo_pilot_job.sh (no native bridge).

Synthetic CPU fixtures from the WP-A1 e2e helpers (real collect/train commands, fake bridge
rows, synthetic anchor). No sealed (3000-3059) or reserved (90154-90159) worlds.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_labor_a_ppo_wiring import (
    _default_models,
    _iteration_files,
    _train_args,
    _trainer_state,
)

from anthropocene_rl.autonomous_v7 import labor_pilot as pilot
from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7 import labor_value_calibration as calibration
from anthropocene_rl.autonomous_v7.provenance import checkpoint_hash

RESET = pilot._reset_state(16, 128, 64, 9, True)


@pytest.fixture()
def files(monkeypatch, tmp_path):
    actor, task, costs, normalizer, layout = _default_models()
    state0 = _trainer_state(actor, task, costs, normalizer, layout)
    checkpoint, rollout, anchor, gate = _iteration_files(monkeypatch, tmp_path, state0, "it0")
    return SimpleNamespace(state0=state0, checkpoint=checkpoint, rollout=rollout, anchor=anchor,
                           gate=gate, tmp=tmp_path)


def _explicit(rollout, tmp, reset=RESET):
    """The fixture rollout with recorded explicit training-world options."""
    payload = torch.load(rollout, weights_only=True)
    payload["metadata"]["resetOptions"] = dict(reset)
    payload["metadata"]["resetOptionsSource"] = "explicit"
    out = tmp / "explicit-rollout.pt"
    torch.save(payload, out)
    return out


def test_plan_is_deterministic_disjoint_and_bounded():
    plans = [pilot.pilot_plan(k) for k in range(pilot.PILOT_MAX_ITERATIONS)]
    assert plans[0]["rolloutWorlds"] == [91100, 91101, 91102]
    assert plans[0]["trainAnchorWorlds"] == [91100, 91101, 91102]
    assert plans[0]["gateWorlds"] == [91200, 91201]
    assert plans[2]["rolloutWorlds"] == [91120, 91121, 91122]
    every = [w for p in plans for key in ("rolloutWorlds", "gateWorlds") for w in p[key]]
    assert len(every) == len(set(every))
    assert not {91190, 91290} & set(every)  # the anchor probe's worlds
    assert not set(every) & set(range(91000, 91006))  # D7 / step check
    rollout = {w for p in plans for w in p["rolloutWorlds"]}
    assert not rollout & {w for p in plans for w in p["gateWorlds"]}
    bases = [p[key] for p in plans for key in ("trainAnchorSeedBase", "gateSeedBase")]
    assert len(set(bases)) == len(bases) and min(bases) >= 10_000_000
    assert pilot.pilot_plan(1, train_anchor_worlds=1)["trainAnchorWorlds"] == [91110]
    for bad in (-1, pilot.PILOT_MAX_ITERATIONS, True):
        with pytest.raises(ValueError):
            pilot.pilot_plan(bad)
    with pytest.raises(ValueError, match="train_anchor_worlds"):
        pilot.pilot_plan(0, rollout_worlds=2, train_anchor_worlds=3)


def test_plan_cli(capsys):
    assert pilot._cli(["plan", "--iteration", "3", "--gate-worlds", "1"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["gateWorlds"] == [91230] and plan["iteration"] == 3


def test_preflight_guards(files, tmp_path):
    plan = pilot.pilot_plan(0)
    record = pilot.pilot_preflight(files.checkpoint, plan, expected_iteration=0)
    assert record["checkpointSha256"] == checkpoint_hash(files.checkpoint)
    assert record["pilotBypassUpdates"] == 0
    with pytest.raises(ValueError, match="expected 1"):
        pilot.pilot_preflight(files.checkpoint, plan, expected_iteration=1)
    trained = dict(files.state0, critic_training_world_seeds=[91200])
    path = tmp_path / "trained.pt"
    torch.save(trained, path)
    with pytest.raises(ValueError, match="actor-gate worlds overlap"):
        pilot.pilot_preflight(path, plan, expected_iteration=0)
    torch.save(dict(files.state0, critic_training_world_seeds=[91101]), path)
    with pytest.raises(ValueError, match="already trained the critic"):
        pilot.pilot_preflight(path, plan, expected_iteration=0)
    with pytest.raises(ValueError, match="reserved"):
        pilot.pilot_preflight(files.checkpoint, dict(plan, gateWorlds=[90155]),
                              expected_iteration=0)
    with pytest.raises(Exception, match="denied"):
        pilot.pilot_preflight(files.checkpoint, dict(plan, rolloutWorlds=[3001],
                                                    trainAnchorWorlds=[3001]),
                              expected_iteration=0)


def test_check_rollout(files):
    worlds = [50000, 50001, 50002]
    with pytest.raises(ValueError, match="world options"):  # the fixture is a legacy reset
        pilot.pilot_check_rollout(files.rollout, files.checkpoint, worlds=worlds, turns=2,
                                  reset=RESET)
    explicit = _explicit(files.rollout, files.tmp)
    record = pilot.pilot_check_rollout(explicit, files.checkpoint, worlds=worlds, turns=2,
                                       reset=RESET)
    assert record["parents"] == 5 and record["parentsPerWorld"] == {
        "50000": 2, "50001": 2, "50002": 1}
    with pytest.raises(ValueError, match="worlds/turns"):
        pilot.pilot_check_rollout(explicit, files.checkpoint, worlds=worlds, turns=3, reset=RESET)
    other = files.tmp / "other.pt"
    torch.save(dict(files.state0, optimizer_seed=2), other)
    with pytest.raises(ValueError, match="another checkpoint"):
        pilot.pilot_check_rollout(explicit, other, worlds=worlds, turns=2, reset=RESET)


def _fake_anchor(files, *, worlds, reset=RESET, policy=None, sha=None):
    *_, digest, _, normalization = pilot._load_trainer(files.checkpoint)
    source = {
        "sourceCheckpointSha256": sha or checkpoint_hash(files.checkpoint),
        "sourcePolicyDigest": policy or digest,
        "normalizationBindingDigest": normalization,
        "worldSeeds": list(worlds),
    }
    return SimpleNamespace(
        metadata={"source": source, "resetOptions": dict(reset)},
        independent_unit_ids=np.array([f"world-{w}" for w in worlds]),
        ranking=SimpleNamespace(group_ids=np.array(["g0", "g0", "g1", "g1"])),
        target_masks={"qol_years": np.array([True, False])},
    )


class _Anchor(SimpleNamespace):
    def __len__(self):
        return 2


def test_check_anchor(files, monkeypatch):
    def install(**kw):
        fake = _Anchor(**vars(_fake_anchor(files, **kw)))
        monkeypatch.setattr(calibration, "load_labor_anchor", lambda path, role: fake)

    install(worlds=[91200, 91201])
    record = pilot.pilot_check_anchor(files.anchor, files.checkpoint, role="actor-gate",
                                      worlds=[91200, 91201], reset=RESET)
    assert record["rows"] == 2 and record["rankGroups"] == 2 and record["rankRows"] == 4
    with pytest.raises(ValueError, match="worlds"):
        pilot.pilot_check_anchor(files.anchor, files.checkpoint, role="actor-gate",
                                 worlds=[91200], reset=RESET)
    install(worlds=[91200, 91201], reset=dict(RESET, players=8))
    with pytest.raises(ValueError, match="world options"):
        pilot.pilot_check_anchor(files.anchor, files.checkpoint, role="actor-gate",
                                 worlds=[91200, 91201], reset=RESET)
    install(worlds=[91200, 91201], policy="0" * 64)
    with pytest.raises(ValueError, match="stale"):
        pilot.pilot_check_anchor(files.anchor, files.checkpoint, role="actor-gate",
                                 worlds=[91200, 91201], reset=RESET)


def test_check_calibration_and_gate_summary(files):
    *_, policy, critic, _ = pilot._load_trainer(files.checkpoint)
    report = json.loads(files.gate.read_text(encoding="utf-8"))
    report.update(dataSha256=checkpoint_hash(files.anchor), currentPolicyDigest=policy,
                  sourceCriticDigest=critic, actorGateOpen=False)
    report["heads"]["feasibility"] = {
        "pass": False, "magnitudePass": True, "coveragePass": True,
        "ranking": {"status": "uncertain-target-pairs", "required": True,
                    "informativeIndependentUnits": 0},
    }
    path = files.tmp / "calibration.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    summary = pilot.pilot_check_calibration(path, files.checkpoint, files.anchor)
    assert summary["actorGateOpen"] is False and summary["failingHeads"] == ["feasibility"]
    assert summary["heads"]["feasibility"]["ranking"]["status"] == "uncertain-target-pairs"
    with pytest.raises(ValueError, match="another actor-gate anchor"):
        pilot.pilot_check_calibration(path, files.checkpoint, files.rollout)
    report["sourceCriticDigest"] = "0" * 64
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="another checkpoint"):
        pilot.pilot_check_calibration(path, files.checkpoint, files.anchor)


def test_iteration_summary_after_a_bypass_update(files, capsys):
    report = json.loads(files.gate.read_text(encoding="utf-8"))
    report["actorGateOpen"] = False
    closed = files.tmp / "closed.json"
    closed.write_text(json.dumps(report), encoding="utf-8")
    out = files.tmp / "it1.pt"
    training.main(_train_args(files.checkpoint, files.rollout, files.anchor, closed, out,
                              "--pilot-ungated-actor"))
    seconds = files.tmp / "seconds.json"
    seconds.write_text(json.dumps({"collect": 600, "train": 30}), encoding="utf-8")
    capsys.readouterr()
    assert pilot._cli([
        "summary", "--iteration", "0", "--checkpoint-in", str(files.checkpoint),
        "--checkpoint-out", str(out), "--rollout", str(files.rollout),
        "--train-anchor", str(files.anchor), "--gate-anchor", str(files.anchor),
        "--calibration", str(closed), "--stage-seconds", str(seconds), "--bypass-expected",
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["actorGateMode"] == "pilot-bypass"
    assert summary["actor"]["acceptedSteps"] > 0
    assert summary["checkpointIteration"] == {"in": 0, "out": 1}
    assert summary["gate"]["actorGateOpen"] is False
    assert summary["pilotBypassUpdatesInCheckpoint"] == 1
    assert summary["meanAbsDeltaLogp"]["initialJoint"] > 0
    assert summary["inputVsBehaviourMaxAbsLogp"]["initialJoint"] < 1e-4
    assert len(summary["actor"]["perEpoch"]) >= 1
    assert summary["critic"]["loss"] is not None and summary["stageSeconds"]["train"] == 30
    assert summary["sha256"]["checkpointOut"] == checkpoint_hash(out)
    with pytest.raises(ValueError, match="another rollout"):
        pilot.pilot_summary(iteration=0, checkpoint_in=files.checkpoint, checkpoint_out=out,
                            rollout=files.checkpoint, train_anchor=files.anchor,
                            gate_anchor=None, calibration=None, stage_seconds={},
                            bypass_expected=True)
    plain = files.tmp / "it1-plain.pt"
    training.main(_train_args(files.checkpoint, files.rollout, files.anchor, closed, plain))
    with pytest.raises(ValueError, match="without --pilot-ungated-actor"):
        pilot.pilot_summary(iteration=0, checkpoint_in=files.checkpoint, checkpoint_out=plain,
                            rollout=files.rollout, train_anchor=files.anchor, gate_anchor=None,
                            calibration=None, stage_seconds={}, bypass_expected=True)
    record = pilot.pilot_summary(iteration=0, checkpoint_in=files.checkpoint,
                                 checkpoint_out=plain, rollout=files.rollout,
                                 train_anchor=files.anchor, gate_anchor=None, calibration=None,
                                 stage_seconds={}, bypass_expected=False)
    assert record["actorGateMode"] == "closed-critic-only"
    assert record["meanAbsDeltaLogp"]["initialJoint"] == 0.0
