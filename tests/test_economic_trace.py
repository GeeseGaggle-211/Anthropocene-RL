from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest
import torch
from test_autonomous_v7 import batch
from test_economic_diagnostics import transition

from anthropocene_rl.autonomous_v7 import economic_diagnostics, economic_trace, ppo
from anthropocene_rl.autonomous_v7.model import Actor, Critic, Normalizer, to_tensors


def test_writer_proof_covers_interleaved_streams_and_aborts(tmp_path):
    writer = economic_trace.EconomicTraceWriter(
        tmp_path / "complete", {"runId": "run", "arm": "base", "optimizerSeed": 70}
    )
    writer.write("transitions", transition(turn=1))
    writer.write("updates", {"event": "mutation-start"})
    writer.write("transitions", transition(turn=2))
    writer.write("updates", {"event": "mutation-complete"})
    writer.close()
    _, rows, updates = economic_diagnostics.load_artifact(tmp_path / "complete")
    assert len(rows) == len(updates) == 2
    failed = economic_trace.EconomicTraceWriter(tmp_path / "failed", {"runId": "failed"})
    failed.write("transitions", transition())
    with pytest.raises(ValueError, match="nonfinite"):
        failed.write("updates", {"loss": float("nan")})
    failed.abort()
    assert not (tmp_path / "failed" / "done.json").exists()
    with pytest.raises(ValueError, match="complete"):
        economic_diagnostics.load_artifact(tmp_path / "failed")


def step_fixture():
    row = transition()["economicDiagnostics"]
    return {
        "turn": 2,
        "diagnosticSchemaVersion": 3,
        "trajectoryId": row["trajectoryId"],
        "economicDiagnostics": {"7": row},
    }


def test_economic_validation_is_field_specific_and_rejects_bad_identity():
    step = step_fixture()
    step["economicDiagnostics"]["7"]["domains"]["trade"]["realisedUtility"] = -3.0
    step["economicDiagnostics"]["7"]["conservation"]["populationResidual"] = -1.0
    assert economic_trace.validate_economic_step(step, required=True)[7]
    for field, value in (("schemaVersion", 2), ("campId", 8), ("trajectoryId", "foreign")):
        bad = copy.deepcopy(step)
        bad["economicDiagnostics"]["7"][field] = value
        with pytest.raises(ValueError, match="mismatch"):
            economic_trace.validate_economic_step(bad, required=True)
    bad = copy.deepcopy(step)
    bad["economicDiagnostics"]["7"]["domains"]["population"]["endingLinked"] = -1.0
    with pytest.raises(ValueError, match="negative physical"):
        economic_trace.validate_economic_step(bad)


def test_zero_denominator_and_missing_reason_rejected():
    step = step_fixture()
    fields = step["economicDiagnostics"]["7"]["domains"]["cultivation"]
    fields["foodPerFarmerYearDenominator"] = 0.0
    fields["foodPerFarmerYear"] = 1.0
    with pytest.raises(ValueError, match="zero-denominator"):
        economic_trace.validate_economic_step(step)
    fields["foodPerFarmerYear"] = None
    with pytest.raises(ValueError, match="availability"):
        economic_trace.validate_economic_step(step)
    step["economicDiagnostics"]["7"]["unavailable"]["domains.cultivation.foodPerFarmerYear"] = (
        "zero-denominator"
    )
    assert economic_trace.validate_economic_step(step)


def test_diagnostic_tree_cannot_enter_actor_tensor_or_rng_path():
    values = batch(2)
    normalizer = Normalizer.fit(values)
    actor = Actor(values["core"].shape[1])
    tensors = to_tensors(values, normalizer)
    with torch.no_grad():
        before = ppo.sample(
            actor(tensors), tensors, ppo.STAGES["capability"], torch.Generator().manual_seed(2)
        )
    perturbed = dict(values)
    perturbed["economicDiagnostics"] = np.full((2, 200), 999.0, dtype=np.float32)
    tensors_after = to_tensors(perturbed, normalizer)
    assert tensors.keys() == tensors_after.keys()
    assert all(torch.equal(tensors[key], tensors_after[key]) for key in tensors)
    with torch.no_grad():
        after = ppo.sample(
            actor(tensors_after),
            tensors_after,
            ppo.STAGES["capability"],
            torch.Generator().manual_seed(2),
        )
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_actor_policy_activity_distinguishes_native_delegation_and_commitment_holds():
    observed_native = dataclasses.replace(ppo.STAGES["mixed"], observe_movement=True)
    native = ppo.actor_policy_active_mask({"commit": 1}, observed_native)
    assert native["strategic"] == [True] * 5
    assert native["service"] == [True] * 6
    assert native["movement"] is False
    assert native["intentTile"] is False

    learned_region = dataclasses.replace(
        ppo.STAGES["mixed"], movement=True, region_movement=True
    )
    learned = ppo.actor_policy_active_mask({"commit": 1}, learned_region)
    assert learned["movement"] is True
    assert learned["intentTile"] is True

    hold_stage = dataclasses.replace(
        ppo.STAGES["mixed"], commitments=True, fission=True, trade=True
    )
    hold = ppo.actor_policy_active_mask({"commit": 0}, hold_stage)
    assert hold["commit"] is True
    assert hold["service"] == [True] * 6
    assert hold["fission"] is True
    assert hold["strategic"] == [False] * 5
    assert hold["clothingRoute"] is False
    assert hold["tradeValue"] == [False] * len(ppo.contracts.TRADE_GROUPS)


def test_preupdate_trace_and_mutation_metrics_are_joined(tmp_path):
    values = batch(2)
    values.update(
        world=np.zeros(2, dtype=int),
        seed=np.full(2, 2000),
        mask=np.zeros(2, dtype=int),
        camp=np.full(2, 7),
        society=np.full(2, 3),
        turn=np.array([1, 2]),
        years=np.full(2, 20.0),
        population=np.full(2, 100),
        prior_population=np.full(2, 100),
        terminated=np.array([False, True]),
        cost=np.zeros(2),
    )
    normalizer = Normalizer.fit(values)
    actor = Actor(values["core"].shape[1])
    teacher = copy.deepcopy(actor)
    critic = Critic(values["core"].shape[1])
    cost_critic = ppo.CostCritic(values["core"].shape[1])
    tensors = to_tensors(values, normalizer)
    with torch.no_grad():
        draws = ppo.sample(
            actor(tensors), tensors, ppo.STAGES["capability"], torch.Generator().manual_seed(2)
        )
        logp, _ = ppo.log_prob(actor(tensors), tensors, draws, ppo.STAGES["capability"])
    roll = ppo.Rollout()
    roll.worker_contracts = [{}]
    roll.world_contracts = [{}]
    roll.economic_rows = [
        {
            "economicDiagnostics": transition(turn=i + 1)["economicDiagnostics"],
            "actionDiagnostics": transition(turn=i + 1)["actionDiagnostics"],
            "actorPolicyActiveMask": transition(turn=i + 1)["actorPolicyActiveMask"],
            "actorDistribution": {},
            "sampledAction": {},
            "provenance": {},
        }
        for i in range(2)
    ]
    reward = np.array([1.0, 2.0])
    prediction = np.array([5.0, 6.0])
    bootstrap = np.zeros(2)
    advantage, target = ppo.gae(values, reward, prediction, bootstrap)
    cost = (np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    writer = economic_trace.EconomicTraceWriter(
        tmp_path / "run", {"runId": "test", "arm": "base", "optimizerSeed": 70}
    )
    hyper = ppo.Hyper(epochs=1, batch=2)
    hyper.kl_refit = False
    optimizers = tuple(
        torch.optim.Adam(net.parameters(), lr=1e-4) for net in (actor, critic, cost_critic)
    )
    ppo.update(
        actor,
        critic,
        cost_critic,
        teacher,
        normalizer,
        values,
        draws,
        logp.numpy(),
        advantage,
        target / 100,
        np.zeros(2),
        np.zeros(2),
        0.0,
        ppo.STAGES["capability"],
        hyper,
        optimizers,
        np.random.default_rng(1),
        regimes=np.array(["mixed", "mixed"]),
        economic_trace=writer,
        trace_context=(
            roll,
            values,
            draws,
            logp.numpy(),
            {"welfare": (reward, reward, prediction, bootstrap, advantage, target)},
            cost,
        ),
    )
    writer.close()
    _, transitions, updates = economic_diagnostics.load_artifact(tmp_path / "run")
    assert transitions[0]["actionDiagnostics"]["activeMask"]["strategic"][0] is True
    assert transitions[0]["actorPolicyActiveMask"]["strategic"][0] is True
    assert transitions[0]["objectives"]["welfare"]["bootstrap"] == 6.0
    assert transitions[1]["objectives"]["welfare"]["bootstrap"] == 0.0
    assert transitions[0]["policyVersion"] == 0
    mutation = next(row for row in updates if row["event"] == "mutation-complete")
    assert mutation["policyVersionAfter"] == 1
    assert mutation["parameterUpdateNorms"]["actor"] > 0
    assert mutation["gradientMetrics"]["actorGradientNormBeforeClipping"] >= 0
    assert (
        mutation["batchRowDigest"]
        == next(row for row in updates if row["event"] == "mutation-start")["batchRowDigest"]
    )
    assert json.loads((tmp_path / "run" / "done.json").read_text())["transitions"]["rows"] == 2
