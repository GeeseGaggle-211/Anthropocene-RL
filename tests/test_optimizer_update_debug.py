import copy
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_autonomous_v7 import batch

from anthropocene_rl.autonomous_v7 import economic_trace, ppo
from anthropocene_rl.autonomous_v7.model import Actor, Critic, Normalizer, VectorCritic, to_tensors
from anthropocene_rl.autonomous_v7.optimizer_update_debug import (
    MAX_AUDIT_ROWS,
    balanced_advantage_contributions,
    checked_audit_rows,
    preserved_observer_state,
)


class MemoryTrace:
    def __init__(self, rows: int):
        self.policy_version = 0
        self.transition_ids = [f"transition-{row}" for row in range(rows)]
        self.events = []

    def write(self, stream, value):
        self.events.append((stream, value))


def fixture(*, vector: bool, rows: int = 4):
    values = batch(rows)
    values.update(
        camp=np.arange(rows, dtype=np.int32),
        turn=np.ones(rows, dtype=np.int32),
        world=np.zeros(rows, dtype=np.int32),
        years=np.full(rows, 20.0),
        prior_population=np.full(rows, 50),
        population=np.full(rows, 50),
        terminated=np.zeros(rows, dtype=bool),
    )
    normalizer = Normalizer.fit(values)
    torch.manual_seed(7)
    actor = Actor(values["core"].shape[1])
    teacher = copy.deepcopy(actor).eval()
    critic = VectorCritic(values["core"].shape[1]) if vector else Critic(values["core"].shape[1])
    cost = ppo.CostCritic(values["core"].shape[1])
    stage = ppo.STAGES["capability"]
    tensors = to_tensors(values, normalizer)
    with torch.no_grad():
        draws = ppo.sample(actor(tensors), tensors, stage, torch.Generator().manual_seed(11))
        old_logp, _ = ppo.log_prob(actor(tensors), tensors, draws, stage)
    target = ({name: np.linspace(1.0, 2.0, rows) for name in critic.heads}
              if vector else np.linspace(1.0, 2.0, rows))
    return values, normalizer, actor, teacher, critic, cost, stage, draws, old_logp.numpy(), target


def optimizers(actor, critic, cost):
    return tuple(torch.optim.Adam(module.parameters(), lr=1e-4) for module in (actor, critic, cost))


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        assert np.array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            assert_nested_equal(left, right)
    else:
        assert first == second


def run_one(item, *, audit_rows=0, trace=None, train_actor=True):
    values, normalizer, actor, teacher, critic, cost, stage, draws, old_logp, target = item
    rows = len(old_logp)
    options = {}
    if trace is not None:
        economic_rows = [
            {
                "actionDiagnostics": {
                    "activeMask": {
                        "strategic": [False, True, True, False, False],
                        "service": [True, True, True, True, True, False],
                        "tradeValue": [False] * len(ppo.contracts.TRADE_GROUPS),
                        "tradeGate": [False] * len(ppo.contracts.TRADE_GROUPS),
                    }
                }
            }
            for _ in range(rows)
        ]
        options = {
            "economic_trace": trace,
            "trace_context": (SimpleNamespace(economic_rows=economic_rows),),
        }
    rng = np.random.default_rng(91)
    update_optimizers = optimizers(actor, critic, cost)
    stats = ppo.update(
        actor, critic, cost, teacher, normalizer, values, draws, old_logp,
        np.linspace(-1.0, 1.0, rows), target, np.linspace(0.5, -0.5, rows),
        np.zeros(rows), 0.3, stage,
        ppo.Hyper(epochs=1, batch=rows, optimizer_audit_rows=audit_rows),
        update_optimizers, rng, train_actor=train_actor, **options,
    )
    return stats, rng, update_optimizers


@pytest.mark.parametrize("region_movement", [False, True])
def test_per_head_logprob_and_entropy_sum_with_all_optional_heads_and_holds(region_movement):
    values, *_ = fixture(vector=False)
    values["region_movement_mask"] = np.ones(
        (len(values["core"]), ppo.contracts.REGION_MOVEMENT_SLOTS), dtype=bool)
    normalizer = Normalizer.fit(values)
    actor = Actor(values["core"].shape[1])
    stage = dataclasses.replace(
        ppo.STAGES["mixed"], movement=True, region_movement=region_movement,
        commitments=True, fission=True, trade=True)
    tensors = to_tensors(values, normalizer)
    out = ppo.explore(actor(tensors), -2.0)
    draws = ppo.sample(out, tensors, stage, torch.Generator().manual_seed(11))
    draws["span"][0] = ppo.contracts.SPAN_HOLD
    joint, entropy = ppo.log_prob(out, tensors, draws, stage)
    components = ppo.log_prob_components(out, tensors, draws, stage)
    coordinates = ppo.log_prob_coordinate_components(out, tensors, draws, stage)
    logprob_sum = sum(value[0] for value in components.values())
    entropy_sum = sum(value[1] for value in components.values())
    assert torch.allclose(logprob_sum, joint, atol=2e-6, rtol=1e-6)
    assert torch.allclose(entropy_sum, entropy, atol=2e-6, rtol=1e-6)
    for name in ("strategic", "economyMix", "clothingRoute", "shelterRoute",
                 "tradeValue", "tradeShare", "tradeGate"):
        assert components[name][0][0] == 0 and components[name][1][0] == 0
    expected_movement = "regionMovement" if region_movement else "movement"
    assert expected_movement in components
    assert "fission" in components and "commitmentSpan" in components
    for group in ("strategic", "service", "tradeValue", "tradeGate"):
        coordinate_logp = sum(values[0] for values in coordinates[group].values())
        coordinate_entropy = sum(values[1] for values in coordinates[group].values())
        assert torch.allclose(coordinate_logp, components[group][0], atol=2e-6, rtol=1e-6)
        assert torch.allclose(coordinate_entropy, components[group][1], atol=2e-6, rtol=1e-6)
    for group in ("strategic", "tradeValue", "tradeGate"):
        for coordinate_logp, coordinate_entropy, policy_scored in coordinates[group].values():
            assert coordinate_logp[0] == 0 and coordinate_entropy[0] == 0
            assert not bool(policy_scored[0])
    assert all(bool(values[2][0]) for values in coordinates["service"].values())


@pytest.mark.parametrize("vector", [False, True])
def test_audited_and_untraced_update_are_exactly_identical(monkeypatch, vector):
    monkeypatch.setattr(economic_trace, "emit_rollout", lambda *args, **kwargs: None)
    plain = fixture(vector=vector)
    audited = copy.deepcopy(plain)
    torch_state = torch.random.get_rng_state().clone()
    plain_stats, plain_rng, plain_optimizers = run_one(plain)
    plain_torch_after = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(torch_state)
    trace = MemoryTrace(4)
    audited_stats, audited_rng, audited_optimizers = run_one(audited, audit_rows=4, trace=trace)
    audited_torch_after = torch.random.get_rng_state().clone()

    assert_nested_equal(plain_stats, audited_stats)
    for plain_module, audited_module in zip(plain[2:6], audited[2:6], strict=True):
        assert_nested_equal(plain_module.state_dict(), audited_module.state_dict())
    pairs = zip(plain_optimizers, audited_optimizers, strict=True)
    for plain_optimizer, audited_optimizer in pairs:
        assert_nested_equal(plain_optimizer.state_dict(), audited_optimizer.state_dict())
    assert_nested_equal(plain_rng.bit_generator.state, audited_rng.bit_generator.state)
    assert torch.equal(plain_torch_after, audited_torch_after)
    complete = next(value for stream, value in trace.events
                    if stream == "updates" and value["event"] == "mutation-complete")
    debug = complete["optimizerDebug"]
    assert debug["scope"]["sampledRowCount"] == 4
    assert len(debug["sameSampledAction"]["rows"]) == 4
    for row in debug["sameSampledAction"]["rows"]:
        assert sum(head["delta"] for head in row["perHead"].values()) == pytest.approx(
            row["jointDelta"], abs=4e-6)
        for group in ("strategic", "service"):
            aggregate = row["perHead"][group]
            values = row["perCoordinate"][group].values()
            assert sum(value["before"] for value in values) == pytest.approx(
                aggregate["before"], abs=3e-6)
            values = row["perCoordinate"][group].values()
            assert sum(value["after"] for value in values) == pytest.approx(
                aggregate["after"], abs=3e-6)
            values = row["perCoordinate"][group].values()
            assert sum(value["delta"] for value in values) == pytest.approx(
                aggregate["delta"], abs=4e-6)
    assert complete["fixedAuditChanges"] == debug["sameSampledAction"]
    first = debug["sameSampledAction"]["rows"][0]["perCoordinate"]
    retained_field = first["strategic"]["retained-field-operation"]
    materials = first["service"]["materials"]
    assert retained_field["policyScored"] is True
    assert retained_field["nativeRecordedActive"] is False
    assert retained_field["activityRelation"] == "scored-but-native-recorded-inactive"
    assert materials["policyScored"] is True
    assert materials["nativeRecordedActive"] is False
    assert debug["sampledPolicyShift"]["nonnegativeKlEstimateMean"] >= 0
    assert debug["sampledPolicyShift"]["measurementPhase"] == "pre-actor-step"
    assert debug["sampledPolicyShift"]["rowSelection"] == "entire-current-minibatch"
    assert complete["approxKlMeasurementPhase"] == "pre-actor-step"
    post_step = debug["sameSampledAction"]["postStepSampledPolicyShift"]
    deltas = torch.tensor(
        [row["jointDelta"] for row in debug["sameSampledAction"]["rows"]],
        dtype=torch.float64,
    )
    assert post_step["measurementPhase"] == "post-actor-step"
    assert post_step["rowSelection"] == "bounded-fixed-audit-rows"
    assert post_step["rowCount"] == len(deltas)
    assert post_step["signedApproxKlMean"] == pytest.approx(float((-deltas).mean()))
    assert post_step["nonnegativeKlEstimateMean"] == pytest.approx(
        float((torch.expm1(deltas) - deltas).mean())
    )
    ratios = deltas.exp()
    ratio_summary = post_step["sampledProbabilityRatio"]
    assert ratio_summary["count"] == len(deltas)
    assert ratio_summary["min"] == pytest.approx(float(ratios.min()))
    assert ratio_summary["max"] == pytest.approx(float(ratios.max()))
    assert ratio_summary["definition"] == "exp(jointAfter-jointBefore)"
    assert "not exact KL" in post_step["interpretation"]
    strategic = debug["sameSampledAction"]["perCoordinate"]["strategic"]
    assert strategic["retained-field-operation"]["scoredButNativeRecordedInactiveRowCount"] == 4
    for name in ("actor", "critic", "cost"):
        norm = debug["actualClip"][f"{name}PreClipNorm"]
        scale = debug["actualClip"][f"{name}ClipScale"]
        assert scale == pytest.approx(min(1.0, 1.0 / (norm + 1e-6)))
    if vector:
        names = debug["vectorCriticSharedEncoderGradientGeometry"]["components"]
        assert set(names) == set(plain[4].heads)
    else:
        reason = debug["unavailable"]["vectorCriticSharedEncoderGradientGeometry"]
        assert reason.startswith("scalar-critic")


def test_vector_critic_audits_each_head_during_critic_only_warmup(monkeypatch):
    monkeypatch.setattr(economic_trace, "emit_rollout", lambda *args, **kwargs: None)
    item = fixture(vector=True)
    trace = MemoryTrace(4)
    run_one(item, audit_rows=2, trace=trace, train_actor=False)
    complete = next(value for stream, value in trace.events
                    if stream == "updates" and value["event"] == "mutation-complete")
    debug = complete["optimizerDebug"]
    geometry = debug["vectorCriticSharedEncoderGradientGeometry"]
    assert set(geometry["components"]) == set(item[4].heads)
    assert geometry["scope"] == {
        "rowCount": 2, "rows": "first-rows-of-actual-minibatch", "branchRowsIncluded": False,
    }
    assert set(debug["criticHeadDiagnostics"]) == set(item[4].heads)
    assert all(value["rowCount"] == 2 for value in debug["criticHeadDiagnostics"].values())
    assert debug["scope"]["actorAudit"] is False and debug["scope"]["criticAudit"] is True
    assert debug["unavailable"]["sameSampledAction"].startswith("critic-only-warmup")


def test_balance_partition_and_audit_cap_validation():
    objective = np.array([1.0, 3.0, -2.0, 2.0])
    constraint = np.array([-0.5, -1.5, 1.0, 0.0])
    regimes = np.array(["a", "a", "b", "b"])
    combined, _ = ppo.balance(objective + constraint, regimes, 4.0)
    objective_out, constraint_out = balanced_advantage_contributions(
        objective, constraint, combined, regimes)
    assert np.allclose(objective_out + constraint_out, combined)
    assert checked_audit_rows(MAX_AUDIT_ROWS) == MAX_AUDIT_ROWS
    with pytest.raises(ValueError, match="between"):
        checked_audit_rows(MAX_AUDIT_ROWS + 1)


def test_post_step_observer_restores_buffers_modes_gradients_and_rng_on_error():
    module = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 1))
    module.eval()
    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)
    state = copy.deepcopy(module.state_dict())
    gradients = [parameter.grad.clone() for parameter in module.parameters()]
    torch_rng = torch.random.get_rng_state().clone()
    generator = np.random.default_rng(12)
    generator_state = copy.deepcopy(generator.bit_generator.state)

    with (
        pytest.raises(RuntimeError, match="probe failure"),
        preserved_observer_state(module, numpy_generators={"probe": generator}),
    ):
        module.train()
        module(torch.randn(4, 3))
        generator.random()
        raise RuntimeError("probe failure")

    assert_nested_equal(module.state_dict(), state)
    assert module.training is False and module[0].training is False
    for parameter, gradient in zip(module.parameters(), gradients, strict=True):
        assert torch.equal(parameter.grad, gradient)
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert_nested_equal(generator.bit_generator.state, generator_state)
