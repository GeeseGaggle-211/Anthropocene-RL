"""Minimal PPO optimizer for the first feed-forward Elder campaign."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from anthropocene_rl.policies.elder_mlp import ElderActorCritic


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 64


@dataclass(frozen=True)
class RolloutBatch:
    observations: Tensor
    continuous_actions: Tensor
    movement_indices: Tensor
    old_log_prob: Tensor
    advantages: Tensor
    returns: Tensor

    def validate(self) -> None:
        count = self.observations.shape[0]
        if self.observations.ndim != 2 or self.observations.shape[1] != 512:
            raise ValueError("PPO observations must have shape [N,512]")
        if self.continuous_actions.shape != (count, 4):
            raise ValueError("PPO continuous actions must have shape [N,4]")
        for tensor in (
            self.movement_indices,
            self.old_log_prob,
            self.advantages,
            self.returns,
        ):
            if tensor.shape != (count,):
                raise ValueError("PPO scalar rollout fields must have shape [N]")
        if count == 0:
            raise ValueError("PPO batch cannot be empty")


class PPOTrainer:
    def __init__(
        self,
        model: ElderActorCritic,
        config: PPOConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
        )

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        batch.validate()
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        count = batch.observations.shape[0]
        minibatch = min(self.config.minibatch_size, count)
        metrics: dict[str, float] = {}

        for _ in range(self.config.epochs):
            order = torch.randperm(count, device=batch.observations.device)
            for start in range(0, count, minibatch):
                index = order[start : start + minibatch]
                observation = batch.observations[index]
                continuous = batch.continuous_actions[index]
                movement = batch.movement_indices[index]
                old_log_prob = batch.old_log_prob[index]
                advantage = advantages[index]
                target_return = batch.returns[index]

                evaluated = self.model.actor.evaluate_actions(
                    observation,
                    continuous,
                    movement,
                )
                value = self.model.critic(observation)
                ratio = torch.exp(evaluated.log_prob - old_log_prob)
                unclipped = ratio * advantage
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * torch.square(value - target_return).mean()
                entropy = evaluated.entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_prob - evaluated.log_prob).mean()
                    clip_fraction = (
                        torch.abs(ratio - 1.0) > self.config.clip_ratio
                    ).float().mean()
                metrics = {
                    "loss": float(loss.detach()),
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.detach()),
                    "approx_kl": float(approx_kl.detach()),
                    "clip_fraction": float(clip_fraction.detach()),
                }

        if not metrics or any(not torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise RuntimeError("PPO update produced non-finite diagnostics")
        return metrics
