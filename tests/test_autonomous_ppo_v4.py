"""Residual value preservation, frozen anchors, populated Adam and H400 mechanics."""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3
from anthropocene_rl.training.autonomous_ppo_v4 import AutonomousTrainerV4
from anthropocene_rl.training.returns import (
    discounted_interval_reward,
    generalized_advantage_estimate,
)


def candidate_config(config, variant="residual_layernorm", horizon=400):
    return dict(
        config,
        version="autonomous-ppo-residual-v4",
        model_version="autonomous-ppo-model-v4",
        critic_variant=variant,
        critic_lr=0.0001,
        horizon_years=horizon,
    )


def rows(model, obs, generator):
    with torch.no_grad():
        action, logp, _, value = model(obs, generator=generator)
    return [
        dict(
            obs={k: v[i : i + 1] for k, v in obs.items()},
            action=action.index(slice(i, i + 1)),
            logp=logp[i : i + 1],
            advantage=float(i % 2 * 2 - 1),
            **{"return": float(value[i] + i % 2 * 2 - 1)},
        )
        for i in range(len(value))
    ]


@pytest.mark.parametrize("variant", ["small_step", "residual_layernorm"])
def test_v4_preserves_values_policy_moments_and_only_updates_live_parameters(tmp_path, variant):
    from test_autonomous_v4 import batch

    torch.set_num_threads(1)
    obs = batch(32)
    config = json.loads(
        Path("configs/training/autonomous-ppo-layernorm-v3-recovery.json").read_text()
    )
    parent = AutonomousTrainerV3(config, 5, {"identity": "parent"}, width=16)
    parent.update(rows(parent.model, obs, parent.sampler))
    old_parameters = dict(parent.model.named_parameters())
    old_moments = {n: copy.deepcopy(parent.optimizer.state[p]) for n, p in old_parameters.items()}
    source = tmp_path / "v3.pt"
    parent.checkpoint(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    trainer = AutonomousTrainerV4(
        candidate_config(config, variant), 9, {"identity": "child"}, width=16
    )
    trainer.warm_start(source, expected_sha256=digest)
    assert torch.equal(parent.model.value(obs), trainer.model.value(obs))
    assert torch.equal(parent.sampler.get_state(), trainer.sampler.get_state())
    assert torch.equal(parent.shuffle.get_state(), trainer.shuffle.get_state())
    assert (trainer.transitions, trainer.updates) == (parent.transitions, parent.updates)
    new_parameters = dict(trainer.model.named_parameters())
    for name, parameter in old_parameters.items():
        assert torch.equal(parameter, new_parameters[name])
        for key, value in old_moments[name].items():
            assert torch.equal(value, trainer.optimizer.state[new_parameters[name]][key])
    a, lp, entropy, value = parent.model(obs, generator=parent.sampler)
    new_a, new_lp, new_entropy, new_value = trainer.model(obs, generator=trainer.sampler)
    assert all(
        torch.equal(getattr(a, name), getattr(new_a, name))
        for name in ("workforce", "activities", "controls")
    )
    assert torch.equal(a.trade, new_a.trade) and torch.equal(a.movement, new_a.movement)
    assert torch.equal(lp, new_lp) and torch.equal(entropy, new_entropy)
    assert torch.equal(value, new_value)
    anchors = {
        n: p.detach().clone() for n, p in trainer.model.named_parameters() if not p.requires_grad
    }
    optimizer_parameters = {p for group in trainer.optimizer.param_groups for p in group["params"]}
    assert all(
        p not in optimizer_parameters for p in trainer.model.parameters() if not p.requires_grad
    )
    trainer.update(rows(trainer.model, obs, trainer.sampler))
    for name, original in anchors.items():
        assert torch.equal(original, new_parameters[name]) and new_parameters[name].grad is None
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    if variant == "residual_layernorm":
        assert len(anchors) == 16
        assert not torch.equal(value, trainer.model.value(obs))


def test_v4_save_resume_and_inference_keep_composite_values_and_next_update(tmp_path):
    from test_autonomous_v4 import batch

    torch.set_num_threads(1)
    config = json.loads(
        Path("configs/training/autonomous-ppo-layernorm-v3-recovery.json").read_text()
    )
    config = candidate_config(config)
    trainer = AutonomousTrainerV4(config, 7, {"test": 1}, width=16)
    obs = batch(32)
    trainer.update(rows(trainer.model, obs, trainer.sampler))
    path = tmp_path / "v4.pt"
    trainer.checkpoint(path)
    restored = AutonomousTrainerV4(config, 19, {"test": 1}, width=16)
    restored.restore(path)
    batch_rows = rows(trainer.model, obs, torch.Generator().manual_seed(911))
    assert trainer.update(batch_rows) == restored.update(batch_rows)
    assert all(
        torch.equal(v, restored.model.state_dict()[k])
        for k, v in trainer.model.state_dict().items()
    )
    from anthropocene_rl.policies.autonomous_inference_v4 import AutonomousElderPolicyV4

    restored.restore(path)
    policy = AutonomousElderPolicyV4(path, expected_provenance={"test": 1}, seed=911)
    assert torch.equal(policy.model.value(obs), restored.model.value(obs))
    assert torch.equal(
        policy.model({k: v[:1] for k, v in obs.items()}, batch_rows[0]["action"])[1],
        restored.model({k: v[:1] for k, v in obs.items()}, batch_rows[0]["action"])[1],
    )


def test_h400_uses_physical_interval_discount_and_truncation_bootstrap():
    for horizon in (100, 400):
        reward = discounted_interval_reward(3, 20, horizon)
        gamma = math.exp(-20 / horizon)
        composed = sum(gamma**i * reward for i in range(25))
        assert composed == pytest.approx(discounted_interval_reward(3, 500, horizon))
        _, returns = generalized_advantage_estimate(
            np.array([reward]),
            np.array([2]),
            np.array([7]),
            np.array([20]),
            np.array([False]),
            np.array([True]),
            horizon_years=horizon,
            lambda_ref=0.95,
        )
        assert returns[0] == pytest.approx(reward + gamma * 7)
    assert math.exp(-20 / 400) == pytest.approx(0.951229424500714)
    assert math.exp(-500 / 400) == pytest.approx(0.2865047968601901)
