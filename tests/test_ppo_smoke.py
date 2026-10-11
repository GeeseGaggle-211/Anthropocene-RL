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

    trainer = PPOTrainer(
        model,
        PPOConfig(
            epochs=2,
            minibatch_size=16,
            entropy_coefficient=0.0,
            target_kl=1e-12,
        ),
    )
    metrics = trainer.update(batch)

    assert metrics["continuous_head_0_grad_max"] > 0.0
    assert metrics["continuous_head_2_grad_max"] > 0.0
    assert metrics["continuous_head_1_grad_max"] == 0.0
    assert metrics["continuous_head_3_grad_max"] == 0.0
    assert metrics["optimization_epochs_completed"] == 2.0
    assert metrics["optimization_actor_epochs_completed"] == 1.0
    assert metrics["target_kl_early_stop"] == 1.0

    with torch.no_grad():
        evaluated = model.actor.evaluate_actions(
            batch.observations,
            batch.continuous_actions,
            batch.movement_indices,
        )
        log_ratio = evaluated.log_prob - batch.old_log_prob
        ratio = torch.exp(log_ratio)
        expected_kl = ((ratio - 1.0) - log_ratio).mean()
        expected_clip_fraction = (
            torch.abs(ratio - 1.0) > trainer.config.clip_ratio
        ).float().mean()
        expected_value_mse = torch.square(
            model.critic(batch.observations) - batch.returns
        ).mean()

    assert metrics["post_update_approx_kl"] == pytest.approx(float(expected_kl))
    assert metrics["post_update_clip_fraction"] == pytest.approx(
        float(expected_clip_fraction)
    )
    assert metrics["post_update_value_mse"] == pytest.approx(float(expected_value_mse))
