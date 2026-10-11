import math

import pytest

pytest.importorskip("torch")

import torch

from anthropocene_rl.policies.elder_mlp import ElderActorCritic
from anthropocene_rl.training.ppo import PPOConfig, PPOTrainer, RolloutBatch
from anthropocene_rl.training.smoke import run_ppo_smoke


def test_ppo_mechanics_smoke_is_finite() -> None:
    metrics = run_ppo_smoke(batch_size=32)
    assert metrics
    assert all(math.isfinite(value) for value in metrics.values())


def test_ppo_keeps_structurally_inactive_continuous_heads_gradient_inert() -> None:
    torch.manual_seed(37)
    count = 32
    observation = torch.randn(count, 512) * 0.1
    observation[:, 80:84] = 0.0
    observation[:, 80] = 1.0  # field retention
    observation[:, 82] = 1.0  # field development
    observation[:, 482:490] = 0.0
    observation[:, 482] = 1.0  # explicit stay
    model = ElderActorCritic()
    with torch.no_grad():
        sampled = model.actor.sample(observation)
        baseline = model.critic(observation)
    advantage = torch.linspace(-1.0, 1.0, count)
    batch = RolloutBatch(
        observations=observation,
        continuous_actions=sampled.continuous.detach(),
        movement_indices=sampled.movement_index.detach(),
        old_log_prob=sampled.log_prob.detach(),
        advantages=advantage,
        returns=(baseline + advantage).detach(),
    )

    metrics = PPOTrainer(
        model,
        PPOConfig(epochs=2, minibatch_size=16, entropy_coefficient=0.0),
    ).update(batch)

    assert metrics["continuous_head_0_grad_max"] > 0.0
    assert metrics["continuous_head_2_grad_max"] > 0.0
    assert metrics["continuous_head_1_grad_max"] == 0.0
    assert metrics["continuous_head_3_grad_max"] == 0.0
