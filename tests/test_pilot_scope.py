from __future__ import annotations

import numpy as np
import pytest

from anthropocene_rl.contracts import (
    DOMESTIC_HERD_COUNT_OFFSET,
    FISHING_CAPABILITY_AVAILABLE_OFFSET,
    FISHING_PRESENT_COUNT_OFFSET,
    HERD_DEVELOPMENT_MASK_OFFSET,
    HERD_RETENTION_MASK_OFFSET,
    HUSBANDRY_CAPABILITY_AVAILABLE_OFFSET,
    MOVEMENT_CANDIDATE_MASK_OFFSET,
    MOVEMENT_ENABLED_OFFSET,
    TENSOR_FEATURE_COUNT,
)
from anthropocene_rl.datasets.generate import _candidate_calibration_scope_violations
from anthropocene_rl.observations.schema import (
    EncodedObservation,
    broad_domain_scope_violations,
)
from anthropocene_rl.training.pilot import (
    PilotConfig,
    _apply_group_relative_first_decision,
    _assert_campaign_scope,
    _assert_cultivation_scope,
)


def _observation(*active_offsets: int) -> EncodedObservation:
    values = [0.0] * TENSOR_FEATURE_COUNT
    values[MOVEMENT_CANDIDATE_MASK_OFFSET] = 1.0
    for offset in active_offsets:
        values[offset] = 1.0
    return EncodedObservation(tuple(values), (True,) * TENSOR_FEATURE_COUNT)


def test_cultivation_scope_accepts_excluded_domains_inactive() -> None:
    _assert_cultivation_scope(_observation(), snapshot_id="snapshot", phase="reset")


@pytest.mark.parametrize(
    "offset",
    (
        HUSBANDRY_CAPABILITY_AVAILABLE_OFFSET,
        DOMESTIC_HERD_COUNT_OFFSET,
        HERD_RETENTION_MASK_OFFSET,
        HERD_DEVELOPMENT_MASK_OFFSET,
        FISHING_CAPABILITY_AVAILABLE_OFFSET,
        FISHING_PRESENT_COUNT_OFFSET,
        MOVEMENT_ENABLED_OFFSET,
        MOVEMENT_CANDIDATE_MASK_OFFSET + 1,
    ),
)
def test_cultivation_scope_rejects_excluded_domain_activation(offset: int) -> None:
    with pytest.raises(RuntimeError, match="Cultivation-only scope escaped"):
        _assert_cultivation_scope(_observation(offset), snapshot_id="snapshot", phase="post-step-1")


def test_generation_detects_scope_escape_in_calibration_rollout() -> None:
    observation = _observation(HUSBANDRY_CAPABILITY_AVAILABLE_OFFSET)
    candidate = {
        "calibration_observations": [
            {
                "source": "legal-calibration-step-1",
                "tensor": {
                    "SchemaVersion": 2,
                    "Values": observation.values,
                    "Present": observation.present,
                },
            }
        ]
    }

    assert _candidate_calibration_scope_violations(candidate) == ("husbandry-capability-available",)


def test_broad_domain_scope_allows_movement_and_husbandry_but_not_fishing() -> None:
    permitted = _observation(
        HUSBANDRY_CAPABILITY_AVAILABLE_OFFSET,
        DOMESTIC_HERD_COUNT_OFFSET,
        HERD_RETENTION_MASK_OFFSET,
        HERD_DEVELOPMENT_MASK_OFFSET,
        MOVEMENT_ENABLED_OFFSET,
        MOVEMENT_CANDIDATE_MASK_OFFSET + 1,
    )
    assert broad_domain_scope_violations(permitted) == ()
    assert broad_domain_scope_violations(
        _observation(FISHING_CAPABILITY_AVAILABLE_OFFSET, FISHING_PRESENT_COUNT_OFFSET)
    ) == ("fishing-capability-available", "fishing-ground-present")


def test_broad_campaign_scope_admits_movement_and_husbandry() -> None:
    _assert_campaign_scope(
        _observation(
            HUSBANDRY_CAPABILITY_AVAILABLE_OFFSET,
            DOMESTIC_HERD_COUNT_OFFSET,
            HERD_RETENTION_MASK_OFFSET,
            HERD_DEVELOPMENT_MASK_OFFSET,
            MOVEMENT_ENABLED_OFFSET,
            MOVEMENT_CANDIDATE_MASK_OFFSET + 1,
        ),
        campaign_scope="broad-domain",
        snapshot_id="snapshot",
        phase="post-step-1",
    )


def test_broad_campaign_is_pinned_to_fifty_decisions() -> None:
    with pytest.raises(ValueError, match="50-decision"):
        PilotConfig(campaign_scope="broad-domain", episodes_per_update=26).validate()
    PilotConfig(
        campaign_scope="broad-domain",
        episodes_per_update=26,
        episode_steps=50,
        evaluation_steps=50,
    ).validate()


def test_group_relative_advantage_changes_only_the_first_decision() -> None:
    advantages = [
        np.asarray([1.0, 10.0]),
        np.asarray([3.0, 20.0]),
        np.asarray([5.0, 30.0]),
        np.asarray([7.0, 40.0]),
    ]
    adjusted, original, first = _apply_group_relative_first_decision(advantages, 4)

    assert original == [1.0, 3.0, 5.0, 7.0]
    assert sum(first) == pytest.approx(0.0)
    assert [row[1] for row in adjusted] == [10.0, 20.0, 30.0, 40.0]
    assert [row[0] for row in adjusted] == pytest.approx(
        [-4.0, -4.0 / 3.0, 4.0 / 3.0, 4.0]
    )
