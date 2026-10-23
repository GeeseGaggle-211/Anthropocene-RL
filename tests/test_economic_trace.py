from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import random

import numpy as np
import pytest
import torch
from test_autonomous_v7 import batch
from test_economic_diagnostics import transition

from anthropocene_rl.autonomous_v7 import (
    economic_diagnostics,
    economic_trace,
    economic_trace_summary,
    ppo,
)
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
    for field, value in (("schemaVersion", 3), ("campId", 8), ("trajectoryId", "foreign")):
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


def test_policy_input_health_reads_exact_actor_tensor_and_preserves_rng_and_outputs():
    values = copy.deepcopy(batch(2))
    values["core"][:, 47] = np.array([np.expm1(1.252), 0.0], dtype=np.float32)
    values["core"][:, 53] = np.array([5.0, 0.0], dtype=np.float32)
    values["core"][:, 144] = np.array([1.0, 0.0], dtype=np.float32)
    normalizer = Normalizer.fit(values)
    normalizer.core_mean[[47, 53, 144]] = 0.0
    normalizer.core_std[[47, 53, 144]] = np.array([0.002, 0.001, 1.0], np.float32)
    tensors = to_tensors(values, normalizer)
    tensor_before = {name: value.clone() for name, value in tensors.items()}
    batch_before = {name: value.copy() for name, value in values.items()}
    actor = Actor(values["core"].shape[1]).eval()
    with torch.no_grad():
        actor_before = actor(tensors)

    random.seed(712)
    np.random.seed(713)
    torch.manual_seed(714)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    health = economic_trace.capture_policy_input_health(values, tensors, normalizer)

    assert random.getstate() == python_rng
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_rng[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_rng[1])
    assert numpy_after[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert all(np.array_equal(values[name], batch_before[name]) for name in values)
    assert all(torch.equal(tensors[name], tensor_before[name]) for name in tensors)
    with torch.no_grad():
        actor_after = actor(tensors)
    assert all(torch.equal(actor_before[name], actor_after[name]) for name in actor_before)

    row = health[0]
    known = {
        value["index"]: value for value in row["previousDecision"]["coordinates"]
    }
    expected_53 = float(tensors["core"][0, 53])
    assert expected_53 == pytest.approx(1791.759, rel=2e-6)
    assert known[53]["normalized"] == expected_53
    assert known[53]["transformed"] == pytest.approx(float(np.log1p(np.float32(5.0))))
    assert known[53]["normalizerMean"] == 0.0
    assert known[53]["normalizerStd"] == pytest.approx(0.001)
    assert known[47]["normalized"] == float(tensors["core"][0, 47])
    assert row["movement"]["coordinate"]["normalized"] == float(
        tensors["core"][0, 144]
    )
    assert row["source"] == "exact-actor-input-tensors"
    assert len(row["topNormalizedOutliers"]) == economic_trace.POLICY_INPUT_OUTLIER_LIMIT
    assert row["topNormalizedOutliers"][0]["coordinate"] == [53]


def test_policy_input_health_exposes_guard_dead_legacy_and_nonfinite_without_json_nan():
    values = copy.deepcopy(batch(1))
    normalizer = Normalizer.fit(values)
    normalizer.core_mean[:] = 0.0
    normalizer.core_std[:] = 0.001
    normalizer.guard = 10.0
    normalizer.dead = {"core": np.zeros(values["core"].shape[1], dtype=bool)}
    normalizer.dead["core"][47] = True
    values["core"][0, 47] = 10.0
    values["core"][0, 53] = 5.0
    tensors = to_tensors(values, normalizer)
    # Exercise serialization-safe reporting of an actual nonfinite actor input
    # without invoking the actor on that deliberately malformed tensor.
    tensors["activities"][0, 0, 0] = float("inf")
    health = economic_trace.capture_policy_input_health(values, tensors, normalizer)[0]
    coordinates = {
        value["index"]: value for value in health["previousDecision"]["coordinates"]
    }
    assert coordinates[47]["dead"] is True
    assert coordinates[47]["zeroedByDeadMask"] is True
    assert coordinates[47]["normalized"] == 0.0
    assert coordinates[53]["clippedByGuard"] is True
    assert coordinates[53]["normalized"] == 10.0
    assert health["actorFloatInputNonfiniteCount"] == 1
    assert health["normalizedInputNonfiniteCount"] == 1
    encoded = economic_trace.canonical_bytes(health)
    assert b"Infinity" not in encoded and b"NaN" not in encoded

    legacy_values = copy.deepcopy(values)
    legacy_values["core"] = legacy_values["core"][:, :32]
    legacy_values["core_present"] = legacy_values["core_present"][:, :32]
    legacy = Normalizer.fit(legacy_values)
    legacy_health = economic_trace.capture_policy_input_health(
        legacy_values, to_tensors(legacy_values, legacy), legacy
    )[0]
    assert legacy_health["previousDecision"]["status"] == "absent-legacy-core-width"
    assert legacy_health["movement"]["status"] == "absent-legacy-core-width"


def test_policy_input_health_reports_bounded_physical_transform_and_support_mask():
    values = copy.deepcopy(batch(256))
    values["core"][:, 40] = np.arange(256) % 2
    values["core"][:, 41:53] = np.linspace(0, 1, 256)[:, None]
    values["core"][:, 53] = np.arange(256) % 33
    values["core"][:, 54] = np.linspace(0, 20, 256)
    values["core"][:, 55] = 0
    values["core"][:, 144] = 1  # physical movement flag bypasses the fitted dead mask
    values["core_present"][:, 40:56] = True
    values["core_present"][:, 144] = True
    provenance = {
        "schemaVersion": 1,
        "kind": "closed-loop-policy-observations",
        "producerDigest": "b" * 64,
        "transitionIds": [f"t-{index}" for index in range(256)],
        "policyVersions": [0] * 256,
    }
    normalizer = Normalizer.fit_bounded(values, closed_loop_provenance=provenance)
    probe = {key: value[:1] for key, value in values.items()}
    health = economic_trace.capture_policy_input_health(
        probe, to_tensors(probe, normalizer), normalizer,
    )[0]
    coordinates = {row["index"]: row for row in health["previousDecision"]["coordinates"]}
    assert health["normalizationContract"] == "model9-bounded-physical-history-v1"
    assert coordinates[53]["transform"] == "nonnegative-count"
    assert coordinates[53]["transformReference"] == 32.0
    assert coordinates[53]["normalizerMean"] is None
    assert coordinates[53]["affineBeforeGuard"] is None
    assert coordinates[53]["beforeGuard"] == pytest.approx(0.0)
    assert coordinates[53]["supportStatus"] == "supported"
    assert coordinates[55]["supportStatus"] == "masked-unsupported"
    assert coordinates[55]["corePresent"] is False
    assert coordinates[55]["zeroedBySupportMask"] is True
    movement = health["movement"]["coordinate"]
    assert movement["dead"] is True
    assert movement["zeroedByDeadMask"] is False
    assert movement["normalized"] == 1.0


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


@pytest.mark.parametrize("scale_mode", ["reference", "batch"])
def test_personality_objective_trace_exactly_reconstructs_combine_personal(scale_mode):
    advantages = {
        "welfare": np.array([1.0, 2.0, 4.0]),
        "population_years": np.array([100.0, -200.0, 300.0]),
        "trade_utility": np.array([0.0, 100.0, -100.0]),
        "qol_years": np.array([10.0, 20.0, -30.0]),
    }
    personality = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    zeros = np.zeros(3)
    objectives = {
        name: (zeros, zeros, zeros, zeros, advantage, zeros)
        for name, advantage in advantages.items()
    }
    hyper = ppo.Hyper(personality_scales=scale_mode, personality_clip=1.25)

    traced = economic_trace._objective_combination_details(
        objectives, {"personality": personality}, hyper
    )
    expected = ppo.combine_personal(
        advantages, personality, scale_mode, hyper.personality_clip
    )
    contribution_sum = sum(
        component["contribution"] for component in traced["components"].values()
    )

    np.testing.assert_allclose(traced["total"], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(contribution_sum, expected, rtol=0, atol=1e-12)
    assert traced["configuredWeightsApplied"] is False
    assert all(
        component["configuredWeight"] == hyper.objective_weights[name]
        for name, component in traced["components"].items()
    )
    if scale_mode == "reference":
        assert traced["mode"] == "personality-reference-normalized"
        assert traced["batchWelfareSpreadMultiplier"] is None
        assert traced["components"]["trade_utility"]["clipApplied"].tolist() == [
            False,
            True,
            True,
        ]
    else:
        assert traced["mode"] == "personality-batch-normalized-welfare-rescaled"
        assert traced["batchWelfareSpreadMultiplier"] == pytest.approx(
            np.std(advantages["welfare"]) + 1e-9
        )
        assert not any(traced["components"]["trade_utility"]["clipApplied"])


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
    reward = np.array([0.0, 2.0])
    prediction = np.array([5.0, 6.0])
    bootstrap = np.zeros(2)
    advantage, target = ppo.gae(values, reward, prediction, bootstrap)
    cost = (np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    writer = economic_trace.EconomicTraceWriter(
        tmp_path / "run", {"runId": "test", "arm": "base", "optimizerSeed": 70}
    )
    hyper = ppo.Hyper(epochs=1, batch=2, optimizer_audit_rows=2)
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
    normalizer_ref = transitions[0]["normalizerMomentsRef"]
    assert "normalizerMoments" not in transitions[0]
    assert {row["normalizerMomentsRef"] for row in transitions} == {normalizer_ref}
    normalizer_path = tmp_path / "run" / "normalizers" / f"{normalizer_ref}.json"
    assert hashlib.sha256(normalizer_path.read_bytes()).hexdigest() == normalizer_ref
    assert transitions[0]["actionDiagnostics"]["activeMask"]["strategic"][0] is True
    assert transitions[0]["actorPolicyActiveMask"]["strategic"][0] is True
    assert transitions[0]["objectives"]["welfare"]["bootstrap"] == 6.0
    assert transitions[1]["objectives"]["welfare"]["bootstrap"] == 0.0
    assert transitions[0]["objectiveCombinationMode"] == "configured-objective-weights"
    combination = transitions[0]["objectiveCombination"]
    assert combination["configuredWeightsApplied"] is True
    assert combination["preStandardizationObjectiveAdvantage"] == pytest.approx(advantage[0])
    assert combination["preBalanceActorAdvantage"] == pytest.approx(
        combination["objectiveAfterStandardization"]
    )
    objective = transitions[0]["objectives"]["welfare"]
    assert objective["objectiveWeight"] == objective["configuredObjectiveWeight"] == 1.0
    assert objective["configuredObjectiveWeightApplied"] is True
    assert objective["personalityWeight"] is None
    assert objective["weightedAdvantageContribution"] == pytest.approx(advantage[0])
    assert objective["zeroShapedRewardNonzeroBootstrap"] is True
    assert transitions[0]["policyVersion"] == 0
    mutation = next(row for row in updates if row["event"] == "mutation-complete")
    assert mutation["policyVersionAfter"] == 1
    assert mutation["parameterUpdateNorms"]["actor"] > 0
    assert mutation["gradientMetrics"]["actorGradientNormBeforeClipping"] >= 0
    assert (
        mutation["batchRowDigest"]
        == next(row for row in updates if row["event"] == "mutation-start")["batchRowDigest"]
    )
    done = json.loads((tmp_path / "run" / "done.json").read_text())
    assert done["transitions"]["rows"] == 2
    assert done["normalizerMomentHashes"] == [normalizer_ref]
    summary = economic_trace_summary.summarize_trace(tmp_path / "run")
    assert summary["optimizerUpdates"]["alignmentDenominator"] == 2
    context = summary["optimizerUpdates"]["teacherAndGradientContext"][0]
    assert context["lossContributions"]["policyClippedCombined"] is not None
    assert context["actualClip"]["actorStepPerformed"] is True
