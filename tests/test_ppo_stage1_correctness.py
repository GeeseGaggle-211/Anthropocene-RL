import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_autonomous_v7 import batch

from anthropocene_rl.autonomous_v7 import ppo
from anthropocene_rl.autonomous_v7.model import (
    BOUNDED_FEATURE_CONTRACT,
    LEGACY_FEATURE_CONTRACT,
    Normalizer,
    load_checkpoint_normalizer,
)
from anthropocene_rl.autonomous_v7.ppo_correctness import (
    capture_step_state,
    epoch_coverage,
    epoch_minibatches,
    fixed_audit_indices,
    restore_step_state,
    sampled_nonnegative_kl,
)


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def test_epoch_partition_visits_every_row_once_and_keeps_short_batch():
    batches = list(epoch_minibatches(11, 4, np.random.default_rng(17)))
    assert [len(batch) for batch in batches] == [4, 4, 3]
    assert epoch_coverage(batches, 11) == {
        "rowCount": 11,
        "visitedRows": 11,
        "minibatches": 3,
        "missingRows": [],
        "repeatedRows": [],
        "exactlyOnce": True,
    }


def test_fixed_audit_is_deterministic_and_spans_full_rollout():
    indices = fixed_audit_indices(101, 7)
    assert np.array_equal(indices, fixed_audit_indices(101, 7))
    assert indices[0] == 0 and indices[-1] == 100
    assert len(np.unique(indices)) == 7


def test_rejected_step_restores_parameters_buffers_and_adam_state_exactly():
    torch.manual_seed(4)
    module = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 1)
    )
    optimizer = torch.optim.Adam(module.parameters(), lr=0.02)
    x = torch.randn(8, 3)
    # Populate Adam moments and BatchNorm buffers before the state that must be restored.
    optimizer.zero_grad()
    module(x).square().mean().backward()
    optimizer.step()
    snapshot = capture_step_state(module, optimizer)
    expected_module = copy.deepcopy(module.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad()
    (module(x).mean() * 100).backward()
    optimizer.step()
    assert not _equal(module.state_dict(), expected_module)
    restore_step_state(module, optimizer, snapshot)
    assert _equal(module.state_dict(), expected_module)
    assert _equal(optimizer.state_dict(), expected_optimizer)


def test_nonnegative_sampled_kl_and_nonfinite_rejection():
    old = torch.tensor([-1.0, -2.0])
    new = torch.tensor([-0.5, -2.25])
    expected = (torch.expm1(new - old) - (new - old)).mean()
    assert sampled_nonnegative_kl(new, old) == expected
    assert float(expected) >= 0
    with pytest.raises(FloatingPointError, match="non-finite"):
        sampled_nonnegative_kl(torch.tensor([float("inf")]), torch.zeros(1))


def test_versioned_ppo_learners_reject_cross_loaded_normalizers():
    values = batch(8)
    bounded = Normalizer.fit_bounded(values)
    bounded_checkpoint = {
        "feature_contract": BOUNDED_FEATURE_CONTRACT,
        "normalizer": bounded.state_dict(),
    }
    with pytest.raises(ValueError, match="feature contract"):
        load_checkpoint_normalizer(
            bounded_checkpoint,
            expected_feature_contract=ppo.learner_feature_contract("legacy"),
        )

    legacy_checkpoint = {"normalizer": Normalizer.fit(values).state_dict()}
    with pytest.raises(ValueError, match="feature contract"):
        load_checkpoint_normalizer(
            legacy_checkpoint,
            expected_feature_contract=ppo.learner_feature_contract("observed-anchor-v1"),
        )
    assert ppo.learner_feature_contract("legacy") == LEGACY_FEATURE_CONTRACT


def test_versioned_checkpoint_metadata_preserves_exact_legacy_key_omission():
    assert ppo.versioned_learner_checkpoint_metadata("legacy", None) == {}
    train = SimpleNamespace(metadata={"role": "train-anchor"})
    gate = SimpleNamespace(metadata={"role": "actor-gate"})
    modern = ppo.versioned_learner_checkpoint_metadata(
        "observed-anchor-v1",
        "bounded-v2",
        normalizer={"normalization_contract": {}},
        observed_train=train,
        observed_gate=gate,
    )
    assert modern == {
        "learner_contract": "observed-anchor-v1",
        "feature_contract": "bounded-v2",
        "normalizer": {"normalization_contract": {}},
        "observed_anchor_sources": {
            "train": {"role": "train-anchor"},
            "gate": {"role": "actor-gate"},
        },
    }
