"""Minimal PPO optimizer for the first feed-forward Elder campaign."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from anthropocene_rl.policies.distributions import structural_head_mask
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
    gradient_clip_mode: Literal["joint", "separate"] = "joint"
    value_target_normalization: bool = False
    value_normalization_epsilon: float = 1e-4


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
        self.value_count = 0.0
        self.value_mean = 0.0
        self.value_m2 = 0.0

    def value(self, observation: Tensor) -> Tensor:
        """Return critic values in the rollout return scale."""

        normalized = self.model.critic(observation)
        if not self.config.value_target_normalization:
            return normalized
        return normalized * self.value_std + self.value_mean

    @property
    def value_std(self) -> float:
        if self.value_count <= 0.0:
            return 1.0
        variance = self.value_m2 / self.value_count
        return max(variance + self.config.value_normalization_epsilon, 0.0) ** 0.5

    def state_dict(self) -> dict[str, object]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "value_normalization": {
                "count": self.value_count,
                "mean": self.value_mean,
                "m2": self.value_m2,
            },
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        optimizer = state.get("optimizer")
        if not isinstance(optimizer, dict):
            raise ValueError("Trainer checkpoint is missing optimizer state")
        self.optimizer.load_state_dict(optimizer)
        normalization = state.get("value_normalization", {})
        if not isinstance(normalization, dict):
            raise ValueError("Trainer checkpoint has invalid value-normalization state")
        self.value_count = float(normalization.get("count", 0.0))
        self.value_mean = float(normalization.get("mean", 0.0))
        self.value_m2 = float(normalization.get("m2", 0.0))

    def _normalize_returns(self, returns: Tensor) -> Tensor:
        if not self.config.value_target_normalization:
            return returns
        return (returns - self.value_mean) / self.value_std

    @torch.no_grad()
    def _update_value_normalization(self, returns: Tensor) -> None:
        if not self.config.value_target_normalization:
            return
        if self.config.value_normalization_epsilon <= 0:
            raise ValueError("value_normalization_epsilon must be positive")
        values = returns.detach().to(dtype=torch.float64)
        batch_count = float(values.numel())
        if batch_count <= 0:
            return
        batch_mean = float(values.mean().cpu())
        batch_m2 = float(torch.sum(torch.square(values - batch_mean)).cpu())
        old_mean = self.value_mean
        old_std = self.value_std
        if self.value_count == 0.0:
            combined_count = batch_count
            combined_mean = batch_mean
            combined_m2 = batch_m2
        else:
            combined_count = self.value_count + batch_count
            delta = batch_mean - self.value_mean
            combined_mean = self.value_mean + delta * batch_count / combined_count
            combined_m2 = (
                self.value_m2
                + batch_m2
                + delta * delta * self.value_count * batch_count / combined_count
            )
        self.value_count = combined_count
        self.value_mean = combined_mean
        self.value_m2 = combined_m2
        new_std = self.value_std
        scale = old_std / new_std
        shift = (old_mean - self.value_mean) / new_std
        head = self.model.critic.value_head
        head.weight.mul_(scale)
        head.bias.mul_(scale).add_(shift)
        self._rescale_optimizer_moments(head.weight, scale)
        self._rescale_optimizer_moments(head.bias, scale)

    def _rescale_optimizer_moments(self, parameter: Tensor, scale: float) -> None:
        state = self.optimizer.state.get(parameter)
        if not state:
            return
        first = state.get("exp_avg")
        second = state.get("exp_avg_sq")
        maximum = state.get("max_exp_avg_sq")
        if isinstance(first, Tensor):
            first.mul_(scale)
        if isinstance(second, Tensor):
            second.mul_(scale * scale)
        if isinstance(maximum, Tensor):
            maximum.mul_(scale * scale)

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        batch.validate()
        if self.config.target_kl is not None and self.config.target_kl <= 0:
            raise ValueError("target_kl must be positive or None")
        if self.config.gradient_clip_mode not in ("joint", "separate"):
            raise ValueError("gradient_clip_mode must be joint or separate")
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        with torch.no_grad():
            pre_update_critic = critic_fit_diagnostics(
                self.value(batch.observations),
                batch.returns,
            )
        self._update_value_normalization(batch.returns)
        normalized_returns = self._normalize_returns(batch.returns)
        actor_before = _parameter_snapshot(self.model.actor)
        actor_backbone_before = _parameter_snapshot(self.model.actor.backbone)
        actor_head_before = _parameter_snapshot(self.model.actor.continuous_mean)
        actor_log_std_before = self.model.actor.log_std.detach().clone()
        critic_before = _parameter_snapshot(self.model.critic)
        critic_backbone_before = _parameter_snapshot(self.model.critic.backbone)
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
        actor_pre_clip_grad_norm_max = 0.0
        critic_pre_clip_grad_norm_max = 0.0
        total_pre_clip_grad_norm_max = 0.0
        paired_actor_grad_norms: list[float] = []
        paired_critic_grad_norms: list[float] = []
        paired_total_grad_norms: list[float] = []
        joint_clip_factors: list[float] = []
        actor_clip_factors: list[float] = []
        critic_clip_factors: list[float] = []

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
                target_return = normalized_returns[index]

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
                actor_pre_clip_grad_norm = 0.0
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
                    actor_pre_clip_grad_norm = _gradient_l2_norm(
                        self.model.actor.parameters()
                    )
                    actor_pre_clip_grad_norm_max = max(
                        actor_pre_clip_grad_norm_max,
                        actor_pre_clip_grad_norm,
                    )
                critic_pre_clip_grad_norm = _gradient_l2_norm(
                    self.model.critic.parameters()
                )
                critic_pre_clip_grad_norm_max = max(
                    critic_pre_clip_grad_norm_max,
                    critic_pre_clip_grad_norm,
                )
                if self.config.gradient_clip_mode == "joint":
                    total_pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    total_pre_clip_grad_norm_value = float(
                        total_pre_clip_grad_norm.detach()
                    )
                else:
                    total_pre_clip_grad_norm_value = _gradient_l2_norm(
                        self.model.parameters()
                    )
                    if actor_enabled:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.actor.parameters(),
                            self.config.max_grad_norm,
                        )
                    torch.nn.utils.clip_grad_norm_(
                        self.model.critic.parameters(),
                        self.config.max_grad_norm,
                    )
                total_pre_clip_grad_norm_max = max(
                    total_pre_clip_grad_norm_max,
                    total_pre_clip_grad_norm_value,
                )
                paired_actor_grad_norms.append(actor_pre_clip_grad_norm)
                paired_critic_grad_norms.append(critic_pre_clip_grad_norm)
                paired_total_grad_norms.append(total_pre_clip_grad_norm_value)
                joint_clip_factor = min(
                    1.0,
                    self.config.max_grad_norm
                    / max(total_pre_clip_grad_norm_value, 1e-12),
                )
                joint_clip_factors.append(joint_clip_factor)
                if self.config.gradient_clip_mode == "joint":
                    actor_clip_factors.append(
                        joint_clip_factor if actor_enabled else 1.0
                    )
                    critic_clip_factors.append(joint_clip_factor)
                else:
                    actor_clip_factors.append(
                        min(
                            1.0,
                            self.config.max_grad_norm
                            / max(actor_pre_clip_grad_norm, 1e-12),
                        )
                        if actor_enabled
                        else 1.0
                    )
                    critic_clip_factors.append(
                        min(
                            1.0,
                            self.config.max_grad_norm
                            / max(critic_pre_clip_grad_norm, 1e-12),
                        )
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
            normalized_value = self.model.critic(batch.observations)
            value = self.value(batch.observations)
            post_update_critic = critic_fit_diagnostics(value, batch.returns)
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
                    "post_update_value_mse": float(torch.square(value - batch.returns).mean()),
                    "post_update_normalized_value_mse": float(
                        torch.square(normalized_value - normalized_returns).mean()
                    ),
                    "value_normalization_count": self.value_count,
                    "value_normalization_mean": self.value_mean,
                    "value_normalization_std": self.value_std,
                    "post_update_entropy": float(evaluated.entropy.mean()),
                    "optimization_epochs_completed": float(epochs_completed),
                    "optimization_actor_epochs_completed": float(
                        actor_epochs_completed
                    ),
                    "optimization_minibatches_completed": float(
                        minibatches_completed
                    ),
                    "target_kl_early_stop": float(early_stop_triggered),
                    "actor_pre_clip_grad_norm_max": actor_pre_clip_grad_norm_max,
                    "critic_pre_clip_grad_norm_max": critic_pre_clip_grad_norm_max,
                    "total_pre_clip_grad_norm_max": total_pre_clip_grad_norm_max,
                    "paired_actor_pre_clip_grad_norm_mean": _mean(
                        paired_actor_grad_norms
                    ),
                    "paired_critic_pre_clip_grad_norm_mean": _mean(
                        paired_critic_grad_norms
                    ),
                    "paired_total_pre_clip_grad_norm_mean": _mean(
                        paired_total_grad_norms
                    ),
                    "joint_clip_factor_mean": _mean(joint_clip_factors),
                    "joint_clip_factor_min": min(joint_clip_factors),
                    "actor_clip_factor_mean": _mean(actor_clip_factors),
                    "actor_clip_factor_min": min(actor_clip_factors),
                    "critic_clip_factor_mean": _mean(critic_clip_factors),
                    "critic_clip_factor_min": min(critic_clip_factors),
                    "actor_parameter_update_l2": _parameter_delta_l2(
                        self.model.actor, actor_before
                    ),
                    "actor_backbone_parameter_update_l2": _parameter_delta_l2(
                        self.model.actor.backbone, actor_backbone_before
                    ),
                    "actor_continuous_head_parameter_update_l2": _parameter_delta_l2(
                        self.model.actor.continuous_mean, actor_head_before
                    ),
                    "actor_log_std_update_l2": float(
                        torch.linalg.vector_norm(
                            self.model.actor.log_std.detach() - actor_log_std_before
                        ).cpu()
                    ),
                    "actor_continuous_head_weight_l2": float(
                        torch.linalg.vector_norm(
                            self.model.actor.continuous_mean.weight.detach()
                        ).cpu()
                    ),
                    "actor_continuous_head_bias_l2": float(
                        torch.linalg.vector_norm(
                            self.model.actor.continuous_mean.bias.detach()
                        ).cpu()
                    ),
                    "critic_parameter_update_l2": _parameter_delta_l2(
                        self.model.critic, critic_before
                    ),
                    "critic_backbone_parameter_update_l2": _parameter_delta_l2(
                        self.model.critic.backbone, critic_backbone_before
                    ),
                }
            )
            metrics.update(
                {
                    f"pre_update_critic_{key}": value
                    for key, value in pre_update_critic.items()
                }
            )
            metrics.update(
                {
                    f"post_update_critic_{key}": value
                    for key, value in post_update_critic.items()
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
        with torch.no_grad():
            actor_mean, _, _ = self.model.actor.parameters_for(batch.observations)
            bounded_mean = torch.sigmoid(2.0 * actor_mean.to(torch.float64))
            active_heads = structural_head_mask(batch.observations)
            for head in range(4):
                active = active_heads[:, head]
                endpoint = (bounded_mean[:, head] <= 0.02) | (
                    bounded_mean[:, head] >= 0.98
                )
                metrics[f"continuous_head_{head}_mean_endpoint_rate"] = float(
                    endpoint[active].to(torch.float32).mean().cpu()
                    if torch.any(active)
                    else 0.0
                )
            metrics["critic_value_head_weight_l2"] = float(
                torch.linalg.vector_norm(self.model.critic.value_head.weight).cpu()
            )
            metrics["critic_value_head_bias_abs"] = float(
                torch.abs(self.model.critic.value_head.bias).max().cpu()
            )
        metrics["movement_head_grad_max"] = movement_head_gradient_max
        return metrics


def critic_fit_diagnostics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    """Describe critic fit on one fixed observation/target cohort."""

    residual = target - prediction
    target_variance = torch.var(target, unbiased=False)
    residual_variance = torch.var(residual, unbiased=False)
    denominator = torch.clamp(target_variance, min=1e-12)
    target_std = torch.sqrt(denominator)
    mse = torch.mean(torch.square(residual))
    bias = torch.mean(prediction - target)
    return {
        "prediction_mean": float(torch.mean(prediction).detach()),
        "prediction_std": float(torch.std(prediction, unbiased=False).detach()),
        "target_mean": float(torch.mean(target).detach()),
        "target_std": float(torch.std(target, unbiased=False).detach()),
        "mse": float(mse.detach()),
        "normalized_mse": float((mse / denominator).detach()),
        "bias": float(bias.detach()),
        "normalized_bias": float((bias / target_std).detach()),
        "explained_variance": float((1.0 - residual_variance / denominator).detach()),
    }


def _gradient_l2_norm(parameters: Iterable[Tensor]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += torch.sum(parameter.grad.detach().double().square()).cpu()
    return float(torch.sqrt(squared))


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def _parameter_delta_l2(
    module: torch.nn.Module,
    before: dict[str, Tensor],
) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for name, parameter in module.named_parameters():
        squared += torch.sum(
            (parameter.detach() - before[name]).double().square()
        ).cpu()
    return float(torch.sqrt(squared))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
