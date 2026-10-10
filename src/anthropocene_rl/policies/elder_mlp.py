"""Feed-forward Elder actor/critic for the first PPO campaign."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from anthropocene_rl.contracts import TENSOR_FEATURE_COUNT
from anthropocene_rl.policies.distributions import (
    MAX_ABS_MEAN,
    MAX_LOG_STD,
    MIN_LOG_STD,
    HybridActionSample,
    evaluate_hybrid,
    sample_hybrid,
)


def _activation(name: str) -> type[nn.Module]:
    if name == "tanh":
        return nn.Tanh
    if name == "silu":
        return nn.SiLU
    raise ValueError(f"Unsupported activation: {name}")


class _Backbone(nn.Module):
    def __init__(self, *, activation: str = "tanh") -> None:
        super().__init__()
        act = _activation(activation)
        self.net = nn.Sequential(
            nn.Linear(TENSOR_FEATURE_COUNT, 256),
            act(),
            nn.Linear(256, 256),
            act(),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)

    def forward(self, observation: Tensor) -> Tensor:
        return self.net(observation)


class ElderActor(nn.Module):
    def __init__(self, *, activation: str = "tanh", initial_log_std: float = -0.5) -> None:
        super().__init__()
        self.backbone = _Backbone(activation=activation)
        self.continuous_mean = nn.Linear(256, 4)
        self.movement_logits = nn.Linear(256, 8)
        self.log_std = nn.Parameter(torch.full((4,), float(initial_log_std)))
        nn.init.orthogonal_(self.continuous_mean.weight, gain=0.01)
        nn.init.zeros_(self.continuous_mean.bias)
        nn.init.orthogonal_(self.movement_logits.weight, gain=0.01)
        nn.init.zeros_(self.movement_logits.bias)

    def parameters_for(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.backbone(observation)
        raw_mean = self.continuous_mean(hidden)
        mean = MAX_ABS_MEAN * torch.tanh(raw_mean / MAX_ABS_MEAN)
        log_std = self.log_std.clamp(MIN_LOG_STD, MAX_LOG_STD)
        movement = self.movement_logits(hidden)
        return mean, log_std, movement

    def sample(self, observation: Tensor, *, deterministic: bool = False) -> HybridActionSample:
        mean, log_std, movement = self.parameters_for(observation)
        return sample_hybrid(
            mean,
            log_std,
            movement,
            observation,
            deterministic=deterministic,
        )

    def evaluate_actions(
        self,
        observation: Tensor,
        continuous: Tensor,
        movement_index: Tensor,
    ) -> HybridActionSample:
        mean, log_std, movement = self.parameters_for(observation)
        return evaluate_hybrid(
            mean,
            log_std,
            movement,
            observation,
            continuous,
            movement_index,
        )


class ElderCritic(nn.Module):
    def __init__(self, *, activation: str = "tanh") -> None:
        super().__init__()
        self.backbone = _Backbone(activation=activation)
        self.value_head = nn.Linear(256, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, observation: Tensor) -> Tensor:
        return self.value_head(self.backbone(observation)).squeeze(-1)


@dataclass(frozen=True)
class ElderActorCriticConfig:
    activation: str = "tanh"
    initial_log_std: float = -0.5


class ElderActorCritic(nn.Module):
    def __init__(self, config: ElderActorCriticConfig | None = None) -> None:
        super().__init__()
        config = config or ElderActorCriticConfig()
        self.actor = ElderActor(
            activation=config.activation,
            initial_log_std=config.initial_log_std,
        )
        self.critic = ElderCritic(activation=config.activation)

    def value(self, observation: Tensor) -> Tensor:
        return self.critic(observation)
