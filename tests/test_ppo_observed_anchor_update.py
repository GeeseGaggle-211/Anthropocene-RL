import copy
import math

import numpy as np
import torch
from test_optimizer_update_debug import assert_nested_equal, fixture, optimizers

from anthropocene_rl.autonomous_v7 import ppo
from anthropocene_rl.autonomous_v7.value_calibration import (
    GUARD_COST,
    VALUE_HEADS,
    ObservedReturnData,
    behavior_policy_digest,
)


def observed_data(tmp_path, values, *, role: str, unit: str, source_policy: str):
    path = tmp_path / f"{role}-{unit}.npz"
    path.write_bytes(b"synthetic-observed-anchor")
    rows = len(values["core"])
    return ObservedReturnData(
        path=path,
        role=role,
        metadata={"source": {"sourcePolicyDigest": source_policy}},
        inputs={key: np.asarray(value).copy() for key, value in values.items()},
        targets_raw={name: np.zeros(rows, np.float64) for name in VALUE_HEADS},
        target_masks={name: np.ones(rows, bool) for name in VALUE_HEADS},
        target_standard_errors={name: np.zeros(rows, np.float64) for name in VALUE_HEADS},
        target_methods={
            name: np.full(rows, "independent-randomized-tail-path-mean") for name in VALUE_HEADS
        },
        row_ids=np.asarray([f"{unit}-row-{row}" for row in range(rows)]),
        independent_unit_ids=np.full(rows, unit),
        ranking=None,
    )


def rollout_from_fixture(values, draws, old_logp):
    rows = len(old_logp)
    fixed = {key: np.asarray(value).copy() for key, value in values.items()}
    fixed.update(
        seed=np.full(rows, 1000, np.int64),
        mask=np.zeros(rows, np.int64),
        society=np.arange(rows, dtype=np.int64),
        parent_society=np.zeros(rows, np.int64),
        qol=np.full(rows, 0.5, np.float64),
        nutrition=np.ones(rows, np.float64),
        welfare=np.full(rows, 500.0, np.float64),
        terminated=np.ones(rows, bool),
    )
    roll = ppo.Rollout(obs=[fixed], draws=[draws], logp=[old_logp.copy()])
    for key in roll.outcome:
        if key in fixed:
            value = fixed[key]
        elif key == "trade_utility" or key == "cost":
            value = np.zeros(rows, np.float64)
        elif key == "bootstrap_index" or key == "bank":
            value = np.full(rows, -1, np.int64)
        else:
            raise AssertionError(f"unhandled rollout outcome {key}")
        roll.outcome[key] = np.asarray(value).tolist()
    return roll


def zero_module(module):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


def center_vector_critic(critic):
    zero_module(critic)
    target = critic.two_hot(torch.zeros(1))[0]
    with torch.no_grad():
        for head in critic.heads.values():
            head.bias.copy_(torch.log(target.clamp_min(1e-30)))


def test_gate_open_runs_all_minibatches_then_requests_current_policy_refresh(monkeypatch, tmp_path):
    item = fixture(vector=True, rows=7)
    values, normalizer, actor, teacher, critic, cost, stage, draws, old_logp, _ = item
    center_vector_critic(critic)
    zero_module(cost)
    hyper = ppo.Hyper(
        epochs=2,
        batch=3,
        learner_contract="observed-anchor-v1",
        target_kl=1e6,
        post_update_kl_rows=4,
    )
    behavior = ppo.behavior_policy_configuration(stage, hyper, actor)
    source_policy = behavior_policy_digest(actor, behavior)
    train = observed_data(
        tmp_path, values, role="train-anchor", unit="train", source_policy=source_policy
    )
    gate = observed_data(
        tmp_path, values, role="actor-gate", unit="gate", source_policy=source_policy
    )
    monkeypatch.setattr(
        ppo, "collect", lambda *args, **kwargs: rollout_from_fixture(values, draws, old_logp)
    )
    update_optimizers = optimizers(actor, critic, cost)
    _, stats = ppo.iteration(
        actor,
        critic,
        cost,
        teacher,
        normalizer,
        [1000],
        1,
        stage,
        hyper,
        update_optimizers,
        0.0,
        torch.Generator().manual_seed(11),
        np.random.default_rng(91),
        train_actor=True,
        observed_train=train,
        observed_gate=gate,
        enabled_value_heads=("welfare", GUARD_COST),
    )
    assert stats["actor_gate_open"] is True
    audit = stats["post_update_fixed_audit"]
    assert audit["acceptedActorSteps"] == 2 * math.ceil(7 / 3)
    assert audit["rolledBackActorSteps"] == 0
    assert all(epoch["exactlyOnce"] for epoch in stats["epoch_coverage"])
    assert audit["rowIdentities"][0]["rolloutRow"] == 0
    assert audit["rowIdentities"][-1]["rolloutRow"] == 6
    assert behavior_policy_digest(actor, behavior) != source_policy

    latest = tmp_path / "latest.pt"
    latest.write_bytes(b"frozen-current-policy-checkpoint")
    pause = ppo.calibration_pause_after_update(
        actor,
        behavior,
        stats["value_calibration"],
        source_policy,
        latest,
        tmp_path / "calibration-refresh-request.json",
        {"contractDigest": "a" * 64, "stateDigest": "b" * 64, "bindingDigest": "c" * 64},
        ("welfare", GUARD_COST),
    )
    assert pause["status"] == "collection-required-no-readiness-claim"
    assert pause["sourcePolicyDigest"] == behavior_policy_digest(actor, behavior)


def test_stale_gate_runs_one_critic_only_iteration_without_actor_change(monkeypatch, tmp_path):
    item = fixture(vector=True, rows=5)
    values, normalizer, actor, teacher, critic, cost, stage, draws, old_logp, _ = item
    hyper = ppo.Hyper(epochs=2, batch=3, learner_contract="observed-anchor-v1")
    train = observed_data(
        tmp_path, values, role="train-anchor", unit="train", source_policy="f" * 64
    )
    gate = observed_data(tmp_path, values, role="actor-gate", unit="gate", source_policy="f" * 64)
    monkeypatch.setattr(
        ppo, "collect", lambda *args, **kwargs: rollout_from_fixture(values, draws, old_logp)
    )
    before = copy.deepcopy(actor.state_dict())
    _, stats = ppo.iteration(
        actor,
        critic,
        cost,
        teacher,
        normalizer,
        [1000],
        1,
        stage,
        hyper,
        optimizers(actor, critic, cost),
        0.0,
        torch.Generator().manual_seed(11),
        np.random.default_rng(91),
        train_actor=True,
        observed_train=train,
        observed_gate=gate,
        enabled_value_heads=("welfare", GUARD_COST),
    )
    assert stats["actor_gate_requested"] is True
    assert stats["actor_gate_open"] is False
    assert stats["post_update_fixed_audit"]["acceptedActorSteps"] == 0
    assert_nested_equal(actor.state_dict(), before)


def test_post_step_kl_rollback_restores_actor_and_adam_but_keeps_epoch_coverage(tmp_path):
    item = fixture(vector=True, rows=7)
    values, normalizer, actor, teacher, critic, cost, stage, draws, old_logp, target = item
    train = observed_data(
        tmp_path, values, role="train-anchor", unit="train", source_policy="f" * 64
    )
    actor_opt = torch.optim.Adam(actor.parameters(), lr=0.1)
    update_optimizers = (
        actor_opt,
        torch.optim.Adam(critic.parameters(), lr=1e-4),
        torch.optim.Adam(cost.parameters(), lr=1e-4),
    )
    actor_before = copy.deepcopy(actor.state_dict())
    adam_before = copy.deepcopy(actor_opt.state_dict())
    stats = ppo.update(
        actor,
        critic,
        cost,
        teacher,
        normalizer,
        values,
        draws,
        old_logp,
        np.linspace(-1.0, 1.0, 7),
        target,
        np.linspace(0.5, -0.5, 7),
        np.zeros(7),
        0.3,
        stage,
        ppo.Hyper(
            epochs=2,
            batch=3,
            learner_contract="observed-anchor-v1",
            target_kl=1e-30,
            post_update_kl_rows=4,
        ),
        update_optimizers,
        np.random.default_rng(91),
        observed_anchor=train,
    )
    assert stats["post_update_fixed_audit"]["rolledBackActorSteps"] == 1
    assert stats["post_update_fixed_audit"]["acceptedActorSteps"] == 0
    assert all(epoch["exactlyOnce"] for epoch in stats["epoch_coverage"])
    assert_nested_equal(actor.state_dict(), actor_before)
    assert_nested_equal(actor_opt.state_dict(), adam_before)
