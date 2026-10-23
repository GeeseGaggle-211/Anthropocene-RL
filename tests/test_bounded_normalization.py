"""New-run bounded normalization contract; historical normalizers remain on their exact legacy path."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7.model import (
    BOUNDED_FEATURE_CONTRACT,
    HISTORY_FEATURES,
    Normalizer,
    load_checkpoint_normalizer,
    normalization_corpus_sha256,
    to_tensors,
)


def batch(rows: int = 256, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    width = 188
    values: dict[str, np.ndarray] = {
        "core": rng.normal(size=(rows, width)).astype(np.float32),
        "core_present": np.ones((rows, width), bool),
        "movement_mask": np.ones((rows, contracts.MOVEMENT_SLOTS), bool),
        "clothing_route_mask": np.ones((rows, 2), bool),
        "shelter_route_mask": np.ones((rows, 2), bool),
        "economy_mix_mask": np.ones((rows, len(contracts.ECONOMY_MIX)), bool),
    }
    values["core"][:, 40] = np.arange(rows) % 2
    values["core"][:, 41:53] = rng.random((rows, 12), dtype=np.float32)
    values["core"][:, 53] = np.arange(rows) % 65
    values["core"][:, 54] = np.linspace(0, 40, rows, dtype=np.float32)
    values["core"][:, 55] = 0  # legitimate constant boolean: remains explicitly unsupported/masked
    values["core"][:, 144] = np.arange(rows) % 2
    return values


def provenance(rows: int) -> dict:
    return {
        "schemaVersion": 1,
        "kind": "closed-loop-policy-observations",
        "producerDigest": "a" * 64,
        "transitionIds": [f"transition-{i}" for i in range(rows)],
        "policyVersions": [i // 32 for i in range(rows)],
    }


def test_legacy_normalizer_state_and_tensors_remain_exact() -> None:
    values = batch(32)
    legacy = Normalizer.fit(values, guard=10.0)
    state = legacy.state_dict()
    assert "normalization_contract" not in state
    assert torch.equal(to_tensors(values, legacy)["core"],
                       to_tensors(values, Normalizer.from_state(state))["core"])
    checkpoint = {"normalizer": state}
    assert load_checkpoint_normalizer(checkpoint).normalization_contract is None


def test_initial_bounded_fit_masks_history_but_keeps_physical_movement_flag() -> None:
    values = batch()
    normalizer = Normalizer.fit_bounded(values)
    assert normalizer.guard == 10.0
    assert float(normalizer.core_std.min()) >= 0.1
    assert normalizer.normalization_contract.support_source is None
    assert normalizer.normalization_contract.supported_history_indices == ()
    tensors = to_tensors(values, normalizer)["core"]
    width = values["core"].shape[1]
    assert torch.count_nonzero(tensors[:, 40:56]) == 0
    assert torch.count_nonzero(tensors[:, width + 40:width + 56]) == 0
    assert torch.equal(tensors[:, 144], torch.as_tensor(values["core"][:, 144]))
    assert torch.all(tensors[:, width + 144] == 1)


def test_closed_loop_refit_derives_subset_and_fixed_physical_transforms() -> None:
    values = batch()
    normalizer = Normalizer.fit_bounded(values, closed_loop_provenance=provenance(len(values["core"])))
    supported = set(normalizer.normalization_contract.supported_history_indices)
    assert supported == {feature.index for feature in HISTORY_FEATURES} - {55}
    assert normalizer.normalization_contract.support_source["corpusSha256"] == normalization_corpus_sha256(values)
    normalizer.require_closed_loop_support([40, 41, 53, 54])
    with pytest.raises(ValueError, match=r"\[55\]"):
        normalizer.require_closed_loop_support([55])

    probe = {key: value[:1].copy() for key, value in values.items()}
    probe["core"][0, 41] = 0.25
    probe["core"][0, 53] = 32
    probe["core"][0, 54] = 20
    tensors = to_tensors(probe, normalizer)["core"]
    assert float(tensors[0, 41]) == pytest.approx(0.25)
    assert float(tensors[0, 53]) == pytest.approx(1.0)
    assert float(tensors[0, 54]) == pytest.approx(1.0)
    assert float(tensors[0, 55]) == 0.0


@pytest.mark.parametrize(("index", "value", "message"), [
    (40, 0.5, "exactly 0 or 1"),
    (41, 1.01, r"in \[0, 1\]"),
    (53, 1.5, "integer count"),
    (53, -1, "integer count"),
    (54, -0.01, "nonnegative duration"),
    (144, np.inf, "non-finite"),
])
def test_bounded_physical_domains_fail_closed(index: int, value: float, message: str) -> None:
    values = batch()
    normalizer = Normalizer.fit_bounded(values)
    values["core"][0, index] = value
    with pytest.raises(ValueError, match=message):
        to_tensors(values, normalizer)


def test_support_is_computed_from_rows_not_caller_assertions() -> None:
    values = batch(255)
    normalizer = Normalizer.fit_bounded(values, closed_loop_provenance=provenance(255))
    assert normalizer.normalization_contract.supported_history_indices == ()

    duplicate = provenance(255)
    duplicate["transitionIds"][-1] = duplicate["transitionIds"][0]
    with pytest.raises(ValueError, match="unique"):
        Normalizer.fit_bounded(values, closed_loop_provenance=duplicate)

    changed = {key: value.copy() for key, value in values.items()}
    changed["core"][0, 0] += 1
    assert normalization_corpus_sha256(changed) != normalization_corpus_sha256(values)


def test_contract_round_trip_and_tampering_fail_closed() -> None:
    values = batch()
    normalizer = Normalizer.fit_bounded(values, closed_loop_provenance=provenance(len(values["core"])))
    state = normalizer.state_dict()
    restored = Normalizer.from_state(state)
    assert restored.normalization_contract.state_dict() == normalizer.normalization_contract.state_dict()
    assert torch.equal(to_tensors(values, restored)["core"], to_tensors(values, normalizer)["core"])

    forged = copy.deepcopy(state)
    forged["normalization_contract"]["historySupport"]["55"]["supported"] = True
    with pytest.raises(ValueError, match="not derived"):
        Normalizer.from_state(forged)
    below_floor = copy.deepcopy(state)
    below_floor["core_std"][0] = 0.01
    with pytest.raises(ValueError, match="below the declared floor"):
        Normalizer.from_state(below_floor)


def test_checkpoint_contract_binding_and_resume_mismatch() -> None:
    values = batch()
    normalizer = Normalizer.fit_bounded(values, closed_loop_provenance=provenance(len(values["core"])))
    checkpoint = {"feature_contract": BOUNDED_FEATURE_CONTRACT, "normalizer": normalizer.state_dict()}
    loaded = load_checkpoint_normalizer(
        checkpoint, expected_feature_contract=BOUNDED_FEATURE_CONTRACT, required_history_indices=[40, 53])
    assert loaded.normalization_contract is not None

    for declared in (None, "wrong-contract"):
        bad = dict(checkpoint)
        if declared is None:
            del bad["feature_contract"]
        else:
            bad["feature_contract"] = declared
        with pytest.raises(ValueError, match="feature contract"):
            load_checkpoint_normalizer(bad)
    legacy = {"feature_contract": BOUNDED_FEATURE_CONTRACT, "normalizer": Normalizer.fit(values).state_dict()}
    with pytest.raises(ValueError, match="legacy normalizer"):
        load_checkpoint_normalizer(legacy)


def test_missing_nonfinite_values_are_zeroed_before_fit_and_inference() -> None:
    values = batch()
    values["core_present"][0, 0] = False
    values["core"][0, 0] = np.nan
    normalizer = Normalizer.fit_bounded(values)
    assert np.isfinite(normalizer.core_mean).all()
    assert float(to_tensors(values, normalizer)["core"][0, 0]) == 0.0

    present = batch()
    present["core"][0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite present"):
        Normalizer.fit_bounded(present)
    with pytest.raises(ValueError, match="non-finite present"):
        to_tensors(present, normalizer)


def test_loaded_bounded_state_requires_exact_moment_and_dead_shapes() -> None:
    state = Normalizer.fit_bounded(batch()).state_dict()
    wrong_core = copy.deepcopy(state)
    wrong_core["core_std"].pop()
    with pytest.raises(ValueError, match="core moment shapes"):
        Normalizer.from_state(wrong_core)
    missing_dead = copy.deepcopy(state)
    del missing_dead["sites_dead"]
    with pytest.raises(ValueError, match="dead-mask inventory"):
        Normalizer.from_state(missing_dead)
