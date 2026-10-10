import math

import numpy as np

from anthropocene_rl.training.returns import (
    discounted_interval_reward,
    generalized_advantage_estimate,
    physical_gamma,
)


def test_discounted_interval_semigroup() -> None:
    rate = 120.0
    horizon = 400.0
    whole = discounted_interval_reward(rate, 20.0, horizon)
    half = discounted_interval_reward(rate, 10.0, horizon)
    split = half + physical_gamma(10.0, horizon) * half
    assert math.isclose(whole, split, rel_tol=1e-12, abs_tol=1e-12)


def test_terminal_gae_does_not_bootstrap() -> None:
    rewards = np.array([1.0])
    values = np.array([0.25])
    next_values = np.array([100.0])
    elapsed = np.array([20.0])
    terminated = np.array([True])
    advantages, returns = generalized_advantage_estimate(
        rewards,
        values,
        next_values,
        elapsed,
        terminated,
        horizon_years=400.0,
        lambda_ref=0.95,
    )
    assert np.allclose(advantages, [0.75])
    assert np.allclose(returns, [1.0])


def test_truncation_bootstraps_but_cuts_episode_carry() -> None:
    rewards = np.array([0.0, 1.0])
    values = np.array([0.0, 0.0])
    next_values = np.array([0.25, 0.0])
    elapsed = np.array([20.0, 20.0])
    terminated = np.array([False, False])
    truncated = np.array([True, False])
    advantages, _ = generalized_advantage_estimate(
        rewards,
        values,
        next_values,
        elapsed,
        terminated,
        truncated,
        horizon_years=400.0,
        lambda_ref=0.95,
    )
    assert np.allclose(
        advantages,
        [physical_gamma(20.0, 400.0) * 0.25, 1.0],
    )
