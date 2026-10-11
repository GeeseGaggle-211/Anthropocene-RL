"""Minimal PPO optimizer for the first feed-forward Elder campaign."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from anthropocene_rl.policies.elder_mlp import ElderActorCritic


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    actor_learning_rate: float | None = 3e-4
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 64
    target_kl: float | None = 0.05


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
        actor_learning_rate = (
            self.config.learning_rate
            if self.config.actor_learning_rate is None
            else self.config.actor_learning_rate
        )
        if actor_learning_rate <= 0 or self.config.learning_rate <= 0:
            raise ValueError("Actor and critic learning rates must be positive")
        self.optimizer = torch.optim.Adam(
            (
                {
                    "params": model.actor.parameters(),
                    "lr": actor_learning_rate,
                },
                {
                    "params": model.critic.parameters(),
                    "lr": self.config.learning_rate,
                },
            ),
            eps=1e-5,
        )

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        batch.validate()
        if self.config.target_kl is not None and self.config.target_kl <= 0:
            raise ValueError("target_kl must be positive or None")
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        count = batch.observations.shape[0]
        minibatch = min(self.config.minibatch_size, count)
        metric_totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        metric_weight = 0
        minibatches_completed = 0
        epochs_completed = 0
        actor_epochs_completed = 0
        early_stop_triggered = False
        policy_frozen = False
        continuous_head_gradient_max = [0.0] * 4
        movement_head_gradient_max = 0.0

        for _ in range(self.config.epochs):
            actor_enabled = not policy_frozen
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
                loss = self.config.value_coefficient * value_loss
                if actor_enabled:
                    loss = (
                        loss
                        + policy_loss
                        - self.config.entropy_coefficient * entropy
                    )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if actor_enabled:
                    mean_weight_grad = self.model.actor.continuous_mean.weight.grad
                    mean_bias_grad = self.model.actor.continuous_mean.bias.grad
                    log_std_grad = self.model.actor.log_std.grad
                    if (
                        mean_weight_grad is None
                        or mean_bias_grad is None
                        or log_std_grad is None
                    ):
                        raise RuntimeError("Continuous actor head gradients are missing")
                    for head in range(4):
                        continuous_head_gradient_max[head] = max(
                            continuous_head_gradient_max[head],
                            float(
                                torch.max(torch.abs(mean_weight_grad[head])).detach()
                            ),
                            float(torch.abs(mean_bias_grad[head]).detach()),
                            float(torch.abs(log_std_grad[head]).detach()),
                        )
                    movement_weight_grad = self.model.actor.movement_logits.weight.grad
                    movement_bias_grad = self.model.actor.movement_logits.bias.grad
                    if movement_weight_grad is None or movement_bias_grad is None:
                        raise RuntimeError("Movement actor head gradients are missing")
                    movement_head_gradient_max = max(
                        movement_head_gradient_max,
                        float(torch.max(torch.abs(movement_weight_grad)).detach()),
                        float(torch.max(torch.abs(movement_bias_grad)).detach()),
                    )
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = evaluated.log_prob - old_log_prob
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        torch.abs(ratio - 1.0) > self.config.clip_ratio
                    ).float().mean()
                minibatch_metrics = {
                    "loss": float(loss.detach()),
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.detach()),
                    "approx_kl": float(approx_kl.detach()),
                    "clip_fraction": float(clip_fraction.detach()),
                }
                weight = int(index.numel())
                for key, value in minibatch_metrics.items():
                    metric_totals[key] += value * weight
                metric_weight += weight
                minibatches_completed += 1

            epochs_completed += 1
            if actor_enabled:
                actor_epochs_completed += 1
            if actor_enabled and self.config.target_kl is not None:
                with torch.no_grad():
                    epoch_evaluated = self.model.actor.evaluate_actions(
                        batch.observations,
                        batch.continuous_actions,
                        batch.movement_indices,
                    )
                    epoch_log_ratio = epoch_evaluated.log_prob - batch.old_log_prob
                    epoch_ratio = torch.exp(epoch_log_ratio)
                    epoch_kl = (
                        (epoch_ratio - 1.0) - epoch_log_ratio
                    ).mean()
                if float(epoch_kl) > self.config.target_kl:
                    early_stop_triggered = True
                    policy_frozen = True

        if metric_weight == 0:
            raise RuntimeError("PPO update produced no minibatches")
        metrics = {
            key: value / metric_weight for key, value in metric_totals.items()
        }
        with torch.no_grad():
            evaluated = self.model.actor.evaluate_actions(
                batch.observations,
                batch.continuous_actions,
                batch.movement_indices,
            )
            value = self.model.critic(batch.observations)
            log_ratio = evaluated.log_prob - batch.old_log_prob
            ratio = torch.exp(log_ratio)
            metrics.update(
                {
                    "post_update_approx_kl": float(
                        (((ratio - 1.0) - log_ratio).mean()).detach()
                    ),
                    "post_update_clip_fraction": float(
                        (
                            torch.abs(ratio - 1.0) > self.config.clip_ratio
                        ).float().mean()
                    ),
                    "post_update_value_mse": float(
                        torch.square(value - batch.returns).mean()
                    ),
                    "post_update_entropy": float(evaluated.entropy.mean()),
                    "optimization_epochs_completed": float(epochs_completed),
                    "optimization_actor_epochs_completed": float(
                        actor_epochs_completed
                    ),
                    "optimization_minibatches_completed": float(
                        minibatches_completed
                    ),
                    "target_kl_early_stop": float(early_stop_triggered),
                }
            )

        finite = all(
            torch.isfinite(torch.tensor(value)) for value in metrics.values()
        )
        if not finite:
            raise RuntimeError("PPO update produced non-finite diagnostics")
        metrics.update(
            {
                f"continuous_head_{head}_grad_max": value
                for head, value in enumerate(continuous_head_gradient_max)
            }
        )
        metrics.update(
            {
                f"continuous_head_{head}_log_std": float(
                    self.model.actor.log_std[head].detach()
                )
                for head in range(4)
            }
        )
        metrics["movement_head_grad_max"] = movement_head_gradient_max
        return metrics
