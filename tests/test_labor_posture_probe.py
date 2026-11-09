"""labor_posture_probe: forced posture arms (never training) and the scope refusals.

Synthetic CPU fixtures: a fake bridge (one opening camp), a fake execution that validates
the submitted schema-12 payload exactly as execute_labor_decision does and returns a scripted
successor with raw trade rows and a fission daughter. No native bridge, no sealed/reserved
worlds.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_labor_a_ppo_wiring import _default_models, _iteration_files, _trainer_state
from test_labor_bridge import _attempt_payload
from test_labor_ppo_perhead import _village_at

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7 import labor_posture_probe as probe
from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7 import labor_value_calibration as calibration
from anthropocene_rl.autonomous_v7.labor_bc import LaborResetOptions
from anthropocene_rl.autonomous_v7.labor_bridge import LaborAttempt
from anthropocene_rl.autonomous_v7.labor_worlds import (
    DIAGNOSTIC_POSTURE_SCOPE,
    refuse_diagnostic_posture,
)

RESET = LaborResetOptions(players=16, width=128, height=64, separation=9)
POTS = contracts.TRADE_GROUPS.index("pots")
SUBMITTED: list[tuple[str, int, dict]] = []


class FakeBridge:
    def reset(self, world, **kw):
        assert kw == RESET.reset_kwargs()
        return {"turn": 0, "villages": [_village_at(7)]}

    def close(self):
        pass


def _successor(turn: int) -> dict:
    village = copy.deepcopy(_village_at(7))
    obs = village["observation"]
    _, rows, width = contracts.SETS[probe.TRADE_SET]
    table = np.zeros((rows, width))
    table[POTS, 7] = 2.5  # exported
    table[POTS, 8] = 0.75  # realised utility
    obs["sets"][probe.TRADE_SET] = table.reshape(-1).tolist()
    present = list(obs["rowPresent"][probe.TRADE_SET])
    present[POTS] = True
    obs["rowPresent"][probe.TRADE_SET] = present
    village["tradeUtility"] = 0.75
    village["population"] = 90
    daughter = {k: v for k, v in _village_at(9).items() if k != "observation"}
    daughter.update(observation=None, parentSociety=70, society=71, population=12)
    return {"turn": turn, "villages": [village, daughter]}


def fake_execute(bridge, catalogs, retained, retry_factory):
    """Validates like execute_labor_decision; the 'compiler' rejects a first proposal whose
    export share is >= 0.2 (typed violation from the bridge fixture) and the retry stays
    invalid, so the probe's feasibility accounting is exercised."""
    initial, final, retry = {}, {}, {}
    for camp, action in retained.items():
        action.validate(catalogs[camp], retry=False)  # the real schema-12 check
        SUBMITTED.append((bridge.__class__.__name__, camp, dict(action.payload)))
        rejected = action.payload["tradeShare"] >= 0.2
        catalog = _village_at(7)["observation"]["laborCatalog"]
        attempt = LaborAttempt.from_payload(
            _attempt_payload(catalog, rejected=rejected), catalogs[camp]
        )
        initial[camp] = attempt
        final[camp] = attempt
        if rejected:
            retry[camp] = retry_factory(camp, catalogs[camp], attempt)
            retry[camp].validate(catalogs[camp], retry=True)
            final[camp] = dataclasses.replace(attempt, attempt=1, remaining_proposals=0)
    return SimpleNamespace(step=_successor(len(SUBMITTED)), retry=retry,
                           initial_attempts=initial, final_attempts=final, feedback={})


@pytest.fixture()
def checkpoint(tmp_path):
    actor, task, costs, normalizer, layout = _default_models()
    path = tmp_path / "start.pt"
    torch.save(_trainer_state(actor, task, costs, normalizer, layout), path)
    SUBMITTED.clear()
    return path


def test_arm_parse_apply_and_validation():
    arm = probe.PostureArm.parse("x:fission=encourage,share=0.2,gate=1,value=0.5/1/1/2/1/1/4")
    assert arm == probe.PostureArm("x", 2, 0.2, (1,) * 7, (0.5, 1.0, 1.0, 2.0, 1.0, 1.0, 4.0))
    assert probe.PostureArm.from_state(arm.state_dict()) == arm
    payload = {"fission": 0, "tradeShare": 1e-4, "tradeGate": [2] * 7, "tradeValue": [1.0] * 7,
               "movement": 3}
    applied = arm.apply(payload)
    assert applied == {"fission": 2, "tradeShare": 0.2, "tradeGate": [1] * 7,
                       "tradeValue": [0.5, 1.0, 1.0, 2.0, 1.0, 1.0, 4.0], "movement": 3}
    assert probe.PostureArm("baseline").apply(payload) == payload
    assert probe.PostureArm("baseline").is_baseline and not arm.is_baseline
    for bad in ("y:fission=3", "y:share=1.5", "y:gate=3", "y:value=9", "y:value=1/1",
                "y:wat=1", "a,b", "y:share"):
        with pytest.raises(ValueError):
            probe.PostureArm.parse(bad)
    assert [a.name for a in probe.DEFAULT_ARMS] == [
        "baseline", "share-0.05", "share-0.2", "share-0.5", "fission-encourage",
        "fission-discourage", "share-0.2-encourage",
    ]


def test_probe_overrides_only_the_posture_and_measures_trade(checkpoint):
    arms = (probe.PostureArm("baseline"),
            probe.PostureArm.parse("forced:fission=2,share=0.2,gate=0,value=2"))
    result = probe.run_posture_probe(
        checkpoint_path=checkpoint, worlds=[50000], turns=2, arms=arms, reset=RESET,
        bridge_factory=FakeBridge, execute=fake_execute,
    )
    assert result["scope"] == DIAGNOSTIC_POSTURE_SCOPE
    base, forced = SUBMITTED[:2], SUBMITTED[2:]
    assert len(base) == len(forced) == 2
    labor = ("laborSpecialtyAllocation", "laborActivityAllocation")
    # first decision: same world state, same per-world sampling stream -> same labor action
    assert all(base[0][2][k] == forced[0][2][k] for k in labor)
    assert base[0][2]["movement"] == forced[0][2]["movement"]
    for _, _, payload in forced:
        assert payload["fission"] == 2 and payload["tradeShare"] == 0.2
        assert payload["tradeGate"] == [0] * 7 and payload["tradeValue"] == [2.0] * 7
    turns = result["results"]["forced"]["50000"]["turns"]
    first, second = turns
    assert first["trade"]["exported"]["pots"] == 2.5
    assert first["trade"]["realisedUtility"]["pots"] == 0.75
    assert first["tradeTotals"]["imported"] == 0.0
    assert first["campsWithTrade"] == 1 and first["villageTradeUtility"] == 0.75
    assert first["fissionDaughters"] == [{"camp": 9, "parentSociety": 70}]
    assert second["fissionDaughters"] == []  # the daughter is no longer new
    assert first["campsBefore"] == 1 and first["populationAfter"] == 90
    assert first["submittedPosture"]["fission"]["encourage"] == 1
    assert first["policyPosture"]["tradeShareMean"] != 0.2
    summary = result["summary"]["forced"]
    assert summary["fissionDaughters"] == 1 and summary["campsWithTrade"] == 2
    assert summary["trade"]["exported"]["pots"] == 5.0
    assert summary["foundersAlive"] == 1 and summary["openingCamps"] == 1
    baseline_first = result["results"]["baseline"]["50000"]["turns"][0]
    assert baseline_first["policyPosture"] == first["policyPosture"]
    assert baseline_first["submittedPosture"] == baseline_first["policyPosture"]
    # feasibility: the forced 0.2 export share is "rejected" by the fake compiler
    catalog = _village_at(7)["observation"]["laborCatalog"]
    rejected_cost = LaborAttempt.from_payload(
        _attempt_payload(catalog, rejected=True),
        probe.labor_policy_tensors([_village_at(7)], _normalizer())[0][0][0],
    ).feasibility_cost
    feas = first["feasibility"]
    assert feas["attemptsKnown"] and feas["rejectedFirstProposals"] == 1
    assert feas["retriesStillInvalid"] == 1 and feas["initialWithViolations"] == 1
    (reason,) = feas["initialViolationReasons"]
    assert reason.endswith("|validation") and feas["retryViolationReasons"] == {reason: 1}
    assert feas["feasibilityInitial"]["sum"] == pytest.approx(rejected_cost)
    assert feas["feasibilityRetry"]["sum"] == pytest.approx(rejected_cost)
    assert feas["settlementGuard"]["sum"] == 0.0 and feas["fallbacks"] == 0
    assert first["retries"] == 1
    base_feas = baseline_first["feasibility"]
    assert base_feas["rejectedFirstProposals"] == 0 and base_feas["initialViolationReasons"] == {}
    assert base_feas["feasibilityRetry"]["sum"] == 0.0
    totals = summary["feasibility"]
    assert totals["rejectedFirstProposals"] == 2 and totals["rejectedShare"] == 1.0
    assert totals["initialViolationReasons"] == {reason: 2}
    assert result["summary"]["baseline"]["feasibility"]["rejectedFirstProposals"] == 0


def _normalizer():
    from anthropocene_rl.autonomous_v7.bridge import observation_batch
    from anthropocene_rl.autonomous_v7.model import Normalizer

    return Normalizer.fit_bounded(observation_batch([_village_at(7)]))


def test_probe_refuses_sealed_and_reserved_worlds(checkpoint):
    for world in (3001, 90155):
        with pytest.raises(Exception, match=r"denied|reserved"):
            probe.run_posture_probe(
                checkpoint_path=checkpoint, worlds=[world], turns=1, reset=RESET,
                bridge_factory=FakeBridge, execute=fake_execute,
            )


def test_probe_output_is_refused_by_training_inputs(checkpoint, tmp_path, monkeypatch):
    result = probe.run_posture_probe(
        checkpoint_path=checkpoint, worlds=[50000], turns=1,
        arms=(probe.PostureArm("baseline"),), reset=RESET,
        bridge_factory=FakeBridge, execute=fake_execute,
    )
    out = tmp_path / "probe.json"
    out.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match=DIAGNOSTIC_POSTURE_SCOPE):
        training._load_json(out)
    # train with the probe file as its calibration report
    state = torch.load(checkpoint, weights_only=True)
    ckpt, rollout, anchor, _ = _iteration_files(monkeypatch, tmp_path, state, "it0")
    with pytest.raises(ValueError, match=DIAGNOSTIC_POSTURE_SCOPE):
        training.main(["train", "--checkpoint", str(ckpt), "--rollout", str(rollout),
                       "--anchor", str(anchor), "--calibration-report", str(out),
                       "--out", str(tmp_path / "x.pt")])
    # a rollout whose metadata carries the scope
    raw = torch.load(rollout, weights_only=True)
    raw["metadata"]["scope"] = DIAGNOSTIC_POSTURE_SCOPE
    tagged = tmp_path / "tagged-rollout.pt"
    torch.save(raw, tagged)
    with pytest.raises(ValueError, match=DIAGNOSTIC_POSTURE_SCOPE):
        training._load_rollout(tagged)
    # a checkpoint carrying the scope (collect, train, the anchor producer and calibration
    # all load through _models_from_checkpoint)
    tagged_state = dict(state, scope=DIAGNOSTIC_POSTURE_SCOPE)
    with pytest.raises(ValueError, match=DIAGNOSTIC_POSTURE_SCOPE):
        training._models_from_checkpoint(tagged_state)
    with pytest.raises(ValueError, match=DIAGNOSTIC_POSTURE_SCOPE):
        training._require_trainer_checkpoint(tagged_state)
    # an anchor whose metadata carries the scope
    anchor_file = tmp_path / "tagged-anchor.npz"
    np.savez(anchor_file, metadata_json=np.asarray(json.dumps(
        {"scope": DIAGNOSTIC_POSTURE_SCOPE, "schemaVersion": 1})))
    with pytest.raises(ValueError, match=DIAGNOSTIC_POSTURE_SCOPE):
        calibration.load_labor_anchor(anchor_file, "actor-gate")
    for record in ({"metadata": {"scope": DIAGNOSTIC_POSTURE_SCOPE}},
                   {"source": {"scope": DIAGNOSTIC_POSTURE_SCOPE}}):
        with pytest.raises(ValueError):
            refuse_diagnostic_posture(record, "x")
    refuse_diagnostic_posture({"scope": "local-diagnostic-rollout-not-readiness"}, "x")


def test_cli_defaults_and_jobs_inline(checkpoint, tmp_path, monkeypatch):
    parser = probe.argument_parser()
    args = parser.parse_args(["--checkpoint", "c.pt", "--out", "o.json"])
    assert args.worlds == "91300,91301" and args.turns == 16 and args.jobs == 1
    assert (args.players, args.width, args.height, args.separation) == (16, 128, 64, 9)
    calls = {}
    original = probe.run_posture_probe

    def run(**kw):
        calls.update(kw)
        return original(**{**kw, "bridge_factory": FakeBridge, "execute": fake_execute,
                           "worlds": [50000], "turns": 1})

    monkeypatch.setattr(probe, "run_posture_probe", run)
    out = tmp_path / "cli.json"
    probe.main(["--checkpoint", str(checkpoint), "--arm", "a", "--arm", "b:share=0.3",
                "--out", str(out)])
    written = json.loads(out.read_text(encoding="utf-8"))
    assert [a["name"] for a in written["arms"]] == ["a", "b"]
    assert written["scope"] == DIAGNOSTIC_POSTURE_SCOPE
    assert calls["reset"] == LaborResetOptions(players=16, width=128, height=64, separation=9)
    units = sorted((tmp_path / "cli.json.units").glob("*.json"))
    assert [u.name for u in units] == ["a-50000.json", "b-50000.json"]
    unit = json.loads(units[1].read_text(encoding="utf-8"))
    assert unit["scope"] == DIAGNOSTIC_POSTURE_SCOPE
    assert unit["unit"] == written["results"]["b"]["50000"]
    with pytest.raises(ValueError, match=DIAGNOSTIC_POSTURE_SCOPE):
        training._load_json(units[0])
