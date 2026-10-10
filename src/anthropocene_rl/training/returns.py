"""Continuous-physical-time reward discount and GAE helpers."""

from __future__ import annotations

import math

import numpy as np


def discounted_interval_reward(welfare_rate: float, elapsed_years: float, horizon_years: float) -> float:
    if elapsed_years < 0:
        raise ValueError("Elapsed physical time cannot be negative")
    if math.isinf(horizon_years):
        return welfare_rate * elapsed_years
    if horizon_years <= 0:
        raise ValueError("Discount horizon must be positive")
    return welfare_rate * horizon_years * (1.0 - math.exp(-elapsed_years / horizon_years))


def physical_gamma(elapsed_years: float, horizon_years: float) -> float:
    if elapsed_years < 0:
        raise ValueError("Elapsed physical time cannot be negative")
    if math.isinf(horizon_years):
        return 1.0
    if horizon_years <= 0:
        raise ValueError("Discount horizon must be positive")
    return math.exp(-elapsed_years / horizon_years)


def physical_lambda(elapsed_years: float, lambda_ref: float, reference_years: float) -> float:
    if not (0.0 <= lambda_ref <= 1.0):
        raise ValueError("lambda_ref must be in [0,1]")
    if elapsed_years < 0 or reference_years <= 0:
        raise ValueError("Invalid physical time for lambda")
    return lambda_ref ** (elapsed_years / reference_years)


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    elapsed_years: np.ndarray,
    terminated: np.ndarray,
    *,
    horizon_years: float,
    lambda_ref: float,
    lambda_reference_years: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = (rewards, values, next_values, elapsed_years, terminated)
    length = len(rewards)
    if any(len(array) != length for array in arrays):
        raise ValueError("GAE arrays must have equal lengths")
    advantages = np.zeros(length, dtype=np.float64)
    carry = 0.0
    for index in range(length - 1, -1, -1):
        gamma = physical_gamma(float(elapsed_years[index]), horizon_years)
        lam = physical_lambda(
            float(elapsed_years[index]),
            lambda_ref,
            lambda_reference_years,
        )
        bootstrap = 0.0 if bool(terminated[index]) else float(next_values[index])
        delta = float(rewards[index]) + gamma * bootstrap - float(values[index])
        carry = delta + gamma * lam * (0.0 if bool(terminated[index]) else carry)
        advantages[index] = carry
    returns = advantages + values.astype(np.float64)
    return advantages, returns
