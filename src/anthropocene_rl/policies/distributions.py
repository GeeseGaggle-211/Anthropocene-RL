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

_EPS = 1e-6


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
    result[..., 0] = torch.where(
        active[..., 0], result[..., 0], torch.zeros_like(result[..., 0])
    )
    result[..., 1] = torch.where(
        active[..., 1], result[..., 1], torch.zeros_like(result[..., 1])
    )
    result[..., 2] = torch.where(
        active[..., 2], result[..., 2], torch.zeros_like(result[..., 2])
    )
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
    normal = Normal(mean, torch.exp(log_std).expand_as(mean))
    latent = mean if deterministic else normal.rsample()
    raw_continuous = (torch.tanh(latent) + 1.0) * 0.5
    continuous = canonicalize_continuous(observation, raw_continuous)
    active = structural_head_mask(observation)
    dim_log_prob = _squashed_log_prob(normal, latent)
    continuous_log_prob = (
        dim_log_prob * active.to(dim_log_prob.dtype)
    ).sum(dim=-1)
    continuous_entropy = (
        (-dim_log_prob) * active.to(dim_log_prob.dtype)
    ).sum(dim=-1)

    legal = movement_mask(observation)
    floor = torch.finfo(movement_logits.dtype).min
    masked_logits = movement_logits.masked_fill(~legal, floor)
    categorical = Categorical(logits=masked_logits)
    movement_index = (
        torch.argmax(masked_logits, dim=-1)
        if deterministic
        else categorical.sample()
    )
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
    normal = Normal(mean, torch.exp(log_std).expand_as(mean))
    bounded = continuous.clamp(_EPS, 1.0 - _EPS)
    unit = (2.0 * bounded - 1.0).clamp(-1.0 + _EPS, 1.0 - _EPS)
    latent = 0.5 * (torch.log1p(unit) - torch.log1p(-unit))
    dim_log_prob = _squashed_log_prob(normal, latent)
    continuous_log_prob = (dim_log_prob * active.to(dim_log_prob.dtype)).sum(dim=-1)
    continuous_entropy = ((-dim_log_prob) * active.to(dim_log_prob.dtype)).sum(dim=-1)

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
    log_one_minus_tanh_sq = 2.0 * (
        math.log(2.0) - latent - F.softplus(-2.0 * latent)
    )
    log_det = math.log(0.5) + log_one_minus_tanh_sq
    return normal.log_prob(latent) - log_det
