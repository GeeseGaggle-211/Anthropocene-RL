"""Opt-in PPO correctness primitives for the reviewed observed-anchor learner.

The historical learner deliberately does not import or call these helpers.  They
are small enough to unit test without a simulator: one permutation covers an
epoch exactly once, and a rejected actor step restores both the module and Adam
state.  The sampled-action KL estimate is diagnostic/step-control evidence, not
a population KL.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


def epoch_minibatches(n: int, batch_size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Yield one shuffled partition of ``range(n)``, including the short final batch."""
    if n <= 0:
        raise ValueError("an epoch needs at least one row")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def epoch_coverage(batches: list[np.ndarray], n: int) -> dict[str, Any]:
    """Describe (and validate) the exact row coverage of one epoch."""
    seen = np.concatenate(batches) if batches else np.empty(0, np.int64)
    counts = np.bincount(seen, minlength=n) if len(seen) else np.zeros(n, np.int64)
    missing = np.flatnonzero(counts == 0)
    repeated = np.flatnonzero(counts > 1)
    return {
        "rowCount": int(n),
        "visitedRows": len(seen),
        "minibatches": len(batches),
        "missingRows": missing.tolist(),
        "repeatedRows": repeated.tolist(),
        "exactlyOnce": bool(len(seen) == n and not len(missing) and not len(repeated)),
    }


def fixed_audit_indices(n: int, requested_rows: int) -> np.ndarray:
    """Deterministically spread a fixed audit across the entire rollout order."""
    if n <= 0 or requested_rows <= 0:
        raise ValueError("fixed audit needs positive row counts")
    count = min(n, requested_rows)
    indices = np.linspace(0, n - 1, count, dtype=np.int64)
    if len(np.unique(indices)) != count:
        raise RuntimeError("fixed audit index construction produced duplicates")
    return indices


@dataclass(frozen=True)
class OptimizerStepSnapshot:
    """Deep copies of all mutable state changed by one optimizer step."""

    module: dict[str, Any]
    optimizer: dict[str, Any]


def capture_step_state(
    module: nn.Module, optimizer: torch.optim.Optimizer
) -> OptimizerStepSnapshot:
    return OptimizerStepSnapshot(
        copy.deepcopy(module.state_dict()), copy.deepcopy(optimizer.state_dict())
    )


def _same_state(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _same_state(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def restore_step_state(
    module: nn.Module, optimizer: torch.optim.Optimizer, snapshot: OptimizerStepSnapshot
) -> None:
    """Restore and verify parameters, buffers, optimizer moments and optimizer step counters."""
    module.load_state_dict(snapshot.module)
    optimizer.load_state_dict(snapshot.optimizer)
    if not _same_state(module.state_dict(), snapshot.module):
        raise RuntimeError("actor parameter/buffer rollback verification failed")
    if not _same_state(optimizer.state_dict(), snapshot.optimizer):
        raise RuntimeError("actor optimizer rollback verification failed")


def sampled_nonnegative_kl(new_logp: torch.Tensor, behaviour_logp: torch.Tensor) -> torch.Tensor:
    """Sample mean of ``exp(log p_new/p_old)-1-log p_new/p_old`` on fixed behaviour actions."""
    if new_logp.shape != behaviour_logp.shape:
        raise ValueError("new and behaviour log probabilities must have the same shape")
    log_ratio = new_logp - behaviour_logp
    estimate = torch.expm1(log_ratio) - log_ratio
    if not bool(torch.isfinite(estimate).all()):
        raise FloatingPointError("non-finite post-update sampled KL estimate")
    return estimate.mean()
