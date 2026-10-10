"""Self-contained optimizer smoke for CI and bring-up."""

from __future__ import annotations

import torch

from anthropocene_rl.policies.elder_mlp import ElderActorCritic
from anthropocene_rl.training.ppo import PPOConfig, PPOTrainer, RolloutBatch


def run_ppo_smoke(seed: int = 17, batch_size: int = 64) -> dict[str, float]:
    torch.manual_seed(seed)
    observation = torch.zeros(batch_size, 512)
    # Mixed-capability state; stay is the only legal movement for this smoke.
    observation[:, 80:84] = 1.0
    observation[:, 482] = 1.0
    model = ElderActorCritic()
    with torch.no_grad():
        sampled = model.actor.sample(observation)
        baseline = model.critic(observation)
    advantages = torch.linspace(-1.0, 1.0, batch_size)
    returns = baseline + advantages
    batch = RolloutBatch(
        observations=observation,
        continuous_actions=sampled.continuous.detach(),
        movement_indices=sampled.movement_index.detach(),
        old_log_prob=sampled.log_prob.detach(),
        advantages=advantages,
        returns=returns.detach(),
    )
    trainer = PPOTrainer(
        model,
        PPOConfig(epochs=2, minibatch_size=min(32, batch_size)),
    )
    return trainer.update(batch)
