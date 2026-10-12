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
    """Hidden MLP with an explicit, independently configurable normalization ablation."""

    def __init__(self, *, activation: str = "tanh", layer_norm: bool = False) -> None:
        super().__init__()
        self.activation_name = activation
        self.layer_norm = layer_norm
        act = _activation(activation)
        if layer_norm:
            self.net = nn.Sequential(
                nn.Linear(TENSOR_FEATURE_COUNT, 256),
                nn.LayerNorm(256),
                act(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                act(),
            )
            self._diagnostic_layout = ((0, 1, 2), (3, 4, 5))
        else:
            # Keep the legacy module layout and state-dict keys exactly unchanged.
            self.net = nn.Sequential(
                nn.Linear(TENSOR_FEATURE_COUNT, 256),
                act(),
                nn.Linear(256, 256),
                act(),
            )
            self._diagnostic_layout = ((0, None, 1), (2, None, 3))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)

    def forward(self, observation: Tensor) -> Tensor:
        return self.net(observation)

    @torch.no_grad()
    def activation_diagnostics(self, observation: Tensor) -> dict[str, object]:
        hidden = observation.detach()
        layers: list[dict[str, object]] = []
        for layer_index, (linear_offset, norm_offset, activation_offset) in enumerate(
            self._diagnostic_layout, start=1
        ):
            linear = self.net[linear_offset]
            normalization = self.net[norm_offset] if norm_offset is not None else None
            activation = self.net[activation_offset]
            if not isinstance(linear, nn.Linear):  # pragma: no cover - construction invariant
                raise TypeError("Backbone diagnostic expected a linear layer")
            linear_output = linear(hidden)
            preactivation = (
                normalization(linear_output) if normalization is not None else linear_output
            )
            hidden = activation(preactivation)
            layers.append(
                {
                    "layer": layer_index,
                    "linear_output": _tensor_diagnostics(linear_output),
                    "preactivation": _tensor_diagnostics(preactivation),
                    "activation": {
                        **_tensor_diagnostics(hidden),
                        "abs_at_or_above_0_95_rate": float(
                            torch.mean((torch.abs(hidden) >= 0.95).float()).cpu()
                        ),
                        "exact_zero_rate": float(torch.mean((hidden == 0).float()).cpu()),
                        "mean_tanh_derivative": (
                            float(torch.mean(1.0 - hidden.square()).cpu())
                            if self.activation_name == "tanh"
                            else None
                        ),
                    },
                }
            )
        return {
            "activation": self.activation_name,
            "layer_norm": self.layer_norm,
            "layers": layers,
        }


class ElderActor(nn.Module):
    def __init__(
        self,
        *,
        activation: str = "tanh",
        initial_log_std: float = -0.5,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = _Backbone(activation=activation, layer_norm=layer_norm)
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

    def sample(
        self,
        observation: Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> HybridActionSample:
        mean, log_std, movement = self.parameters_for(observation)
        return sample_hybrid(
            mean,
            log_std,
            movement,
            observation,
            deterministic=deterministic,
            generator=generator,
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
    def __init__(self, *, activation: str = "tanh", layer_norm: bool = False) -> None:
        super().__init__()
        self.backbone = _Backbone(activation=activation, layer_norm=layer_norm)
        self.value_head = nn.Linear(256, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, observation: Tensor) -> Tensor:
        return self.value_head(self.backbone(observation)).squeeze(-1)


@dataclass(frozen=True)
class ElderActorCriticConfig:
    activation: str = "tanh"
    initial_log_std: float = -0.5
    actor_layer_norm: bool = False
    critic_layer_norm: bool = False


class ElderActorCritic(nn.Module):
    def __init__(self, config: ElderActorCriticConfig | None = None) -> None:
        super().__init__()
        config = config or ElderActorCriticConfig()
        self.actor = ElderActor(
            activation=config.activation,
            initial_log_std=config.initial_log_std,
            layer_norm=config.actor_layer_norm,
        )
        self.critic = ElderCritic(
            activation=config.activation,
            layer_norm=config.critic_layer_norm,
        )

    def value(self, observation: Tensor) -> Tensor:
        return self.critic(observation)

    @torch.no_grad()
    def hidden_activation_diagnostics(self, observation: Tensor) -> dict[str, object]:
        return {
            "observation_count": int(observation.shape[0]),
            "actor": self.actor.backbone.activation_diagnostics(observation),
            "critic": self.critic.backbone.activation_diagnostics(observation),
        }


def _tensor_diagnostics(tensor: Tensor) -> dict[str, float]:
    values = tensor.detach().float().reshape(-1)
    quantiles = torch.quantile(
        values,
        torch.tensor((0.01, 0.05, 0.50, 0.95, 0.99), device=values.device),
    )
    return {
        "mean": float(torch.mean(values).cpu()),
        "std": float(torch.std(values, unbiased=False).cpu()),
        "p01": float(quantiles[0].cpu()),
        "p05": float(quantiles[1].cpu()),
        "p50": float(quantiles[2].cpu()),
        "p95": float(quantiles[3].cpu()),
        "p99": float(quantiles[4].cpu()),
        "max_abs": float(torch.max(torch.abs(values)).cpu()),
        "abs_at_or_above_2_rate": float(torch.mean((torch.abs(values) >= 2.0).float()).cpu()),
    }
