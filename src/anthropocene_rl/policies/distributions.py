"""Hybrid Elder action distributions with structural masking."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal

from anthropocene_rl.contracts import (
    FIELD_DEVELOPMENT_MASK_OFFSET,
    FIELD_RETENTION_MASK_OFFSET,
    HERD_DEVELOPMENT_MASK_OFFSET,
    HERD_RETENTION_MASK_OFFSET,
    MOVEMENT_CANDIDATE_MASK_OFFSET,
)

MIN_LOG_STD = -5.0
MAX_LOG_STD = 0.5
MAX_ABS_MEAN = 5.0

# Eight-point Gauss-Hermite quadrature gives a deterministic, differentiable
# estimate of the current tanh-Gaussian entropy.  In particular, entropy must
# not be estimated from the old action supplied to PPO evaluation.
_GH_NODES = (
    -2.930637420257244,
    -1.981656756695843,
    -1.157193712446780,
    -0.381186990207322,
    0.381186990207322,
    1.157193712446780,
    1.981656756695843,
    2.930637420257244,
)
_GH_WEIGHTS = (
    0.000199604072211,
    0.017077983007413,
    0.207802325814892,
    0.661147012558241,
    0.661147012558241,
    0.207802325814892,
    0.017077983007413,
    0.000199604072211,
)


@dataclass(frozen=True)
class HybridActionSample:
    continuous: Tensor
    movement_index: Tensor
    log_prob: Tensor
    entropy: Tensor


def structural_head_mask(observation: Tensor) -> Tensor:
    field_retention = observation[..., FIELD_RETENTION_MASK_OFFSET] > 0.5
    herd_retention = observation[..., HERD_RETENTION_MASK_OFFSET] > 0.5
    field_development = observation[..., FIELD_DEVELOPMENT_MASK_OFFSET] > 0.5
    herd_development = observation[..., HERD_DEVELOPMENT_MASK_OFFSET] > 0.5
    development = field_development | herd_development
    split = field_development & herd_development
    return torch.stack((field_retention, herd_retention, development, split), dim=-1)


def movement_mask(observation: Tensor) -> Tensor:
    start = MOVEMENT_CANDIDATE_MASK_OFFSET
    mask = observation[..., start : start + 8] > 0.5
    if torch.any(~mask[..., 0]):
        raise ValueError("Native movement mask must always permit explicit stay")
    return mask


def canonicalize_continuous(observation: Tensor, sampled: Tensor) -> Tensor:
    if sampled.shape[-1] != 4:
        raise ValueError("Expected four Elder continuous actions")
    active = structural_head_mask(observation)
    field_dev = observation[..., FIELD_DEVELOPMENT_MASK_OFFSET] > 0.5
    herd_dev = observation[..., HERD_DEVELOPMENT_MASK_OFFSET] > 0.5
    result = sampled.clone()
    result[..., 0] = torch.where(active[..., 0], result[..., 0], torch.zeros_like(result[..., 0]))
    result[..., 1] = torch.where(active[..., 1], result[..., 1], torch.zeros_like(result[..., 1]))
    result[..., 2] = torch.where(active[..., 2], result[..., 2], torch.zeros_like(result[..., 2]))
    canonical_split = torch.where(
        field_dev & ~herd_dev,
        torch.ones_like(result[..., 3]),
        torch.where(
            ~field_dev & herd_dev,
            torch.zeros_like(result[..., 3]),
            torch.full_like(result[..., 3], 0.5),
        ),
    )
    result[..., 3] = torch.where(active[..., 3], result[..., 3], canonical_split)
    return result


def sample_hybrid(
    mean: Tensor,
    log_std: Tensor,
    movement_logits: Tensor,
    observation: Tensor,
    *,
    deterministic: bool = False,
) -> HybridActionSample:
    _validate_gaussian_parameters(mean, log_std)
    normal = Normal(mean, torch.exp(log_std).expand_as(mean))
    latent = mean if deterministic else normal.rsample()
    # sigmoid(2z) is exactly (tanh(z)+1)/2. Float64 preserves an invertible
    # requested action throughout the supported Gaussian parameter envelope.
    raw_continuous = torch.sigmoid(2.0 * latent.to(torch.float64))
    continuous = canonicalize_continuous(observation, raw_continuous)
    active = structural_head_mask(observation)
    dim_log_prob = _bounded_log_prob(normal, continuous, active)
    active_float = active.to(dim_log_prob.dtype)
    continuous_log_prob = (dim_log_prob * active_float).sum(dim=-1)
    continuous_entropy = (_squashed_entropy(normal) * active_float).sum(dim=-1)

    legal = movement_mask(observation)
    floor = torch.finfo(movement_logits.dtype).min
    masked_logits = movement_logits.masked_fill(~legal, floor)
    categorical = Categorical(logits=masked_logits)
    movement_index = torch.argmax(masked_logits, dim=-1) if deterministic else categorical.sample()
    log_prob = continuous_log_prob + categorical.log_prob(movement_index)
    entropy = continuous_entropy + categorical.entropy()
    return HybridActionSample(continuous, movement_index, log_prob, entropy)


def evaluate_hybrid(
    mean: Tensor,
    log_std: Tensor,
    movement_logits: Tensor,
    observation: Tensor,
    continuous: Tensor,
    movement_index: Tensor,
) -> HybridActionSample:
    active = structural_head_mask(observation)
    _validate_gaussian_parameters(mean, log_std)
    normal = Normal(mean, torch.exp(log_std).expand_as(mean))
    dim_log_prob = _bounded_log_prob(normal, continuous, active)
    active_float = active.to(dim_log_prob.dtype)
    continuous_log_prob = (dim_log_prob * active_float).sum(dim=-1)
    continuous_entropy = (_squashed_entropy(normal) * active_float).sum(dim=-1)

    legal = movement_mask(observation)
    masked_logits = movement_logits.masked_fill(~legal, torch.finfo(movement_logits.dtype).min)
    categorical = Categorical(logits=masked_logits)
    if torch.any(~legal.gather(-1, movement_index.unsqueeze(-1)).squeeze(-1)):
        raise ValueError("Attempted to score an illegal movement action")
    log_prob = continuous_log_prob + categorical.log_prob(movement_index)
    entropy = continuous_entropy + categorical.entropy()
    return HybridActionSample(continuous, movement_index, log_prob, entropy)


def _squashed_log_prob(normal: Normal, latent: Tensor) -> Tensor:
    # a=(tanh(z)+1)/2, therefore da/dz=.5*(1-tanh(z)^2).
    log_one_minus_tanh_sq = 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))
    log_det = math.log(0.5) + log_one_minus_tanh_sq
    return normal.log_prob(latent) - log_det


def _bounded_log_prob(normal: Normal, bounded: Tensor, active: Tensor) -> Tensor:
    bounded64 = bounded.to(torch.float64)
    invalid = active & (
        ~torch.isfinite(bounded64) | (bounded64 <= 0.0) | (bounded64 >= 1.0)
    )
    if torch.any(invalid):
        raise ValueError("Active bounded action must be finite and strictly interior")
    scored = torch.where(active, bounded64, torch.full_like(bounded64, 0.5))
    latent = 0.5 * (torch.log(scored) - torch.log1p(-scored))
    return _squashed_log_prob(normal, latent)


def _squashed_entropy(normal: Normal) -> Tensor:
    nodes = torch.as_tensor(
        _GH_NODES,
        dtype=torch.float64,
        device=normal.loc.device,
    )
    weights = torch.as_tensor(
        _GH_WEIGHTS,
        dtype=torch.float64,
        device=normal.loc.device,
    )
    location = normal.loc.to(torch.float64)
    scale = normal.scale.to(torch.float64)
    latent = location.unsqueeze(-1) + math.sqrt(2.0) * scale.unsqueeze(-1) * nodes
    log_one_minus_tanh_sq = 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))
    log_det = math.log(0.5) + log_one_minus_tanh_sq
    expected_log_det = (log_det * weights).sum(dim=-1) / math.sqrt(math.pi)
    return normal.entropy().to(torch.float64) + expected_log_det


def _validate_gaussian_parameters(mean: Tensor, log_std: Tensor) -> None:
    if torch.any(~torch.isfinite(mean)) or torch.any(torch.abs(mean) > MAX_ABS_MEAN):
        raise ValueError(f"Continuous mean must be finite and within ±{MAX_ABS_MEAN}")
    if torch.any(~torch.isfinite(log_std)) or torch.any(
        (log_std < MIN_LOG_STD) | (log_std > MAX_LOG_STD)
    ):
        raise ValueError(
            "Continuous log_std must be finite and within "
            f"[{MIN_LOG_STD},{MAX_LOG_STD}]"
        )
