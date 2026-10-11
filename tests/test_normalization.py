import math

import numpy as np
import pytest

from anthropocene_rl.observations.normalization import (
    CAPABILITY_EXPERIENCE_OFFSETS_BY_KEY,
    NormalizationManifest,
    SemanticNormalizerV2,
)
from anthropocene_rl.observations.schema import EncodedObservation


def make_observation(
    *,
    population: float = 100.0,
    horizon: float = 20.0,
    present: bool = True,
) -> EncodedObservation:
    values = [0.0] * 512
    presence = [present] * 512
    values[0] = population
    values[3] = horizon
    values[5] = 0.8
    values[80] = 1.0
    values[82] = 1.0
    values[482] = 1.0
    return EncodedObservation(tuple(values), tuple(presence))


def replace(
    observation: EncodedObservation,
    *,
    values: dict[int, float] | None = None,
    present: dict[int, bool] | None = None,
) -> EncodedObservation:
    raw_values = list(observation.values)
    raw_present = list(observation.present)
    for index, value in (values or {}).items():
        raw_values[index] = value
    for index, value in (present or {}).items():
        raw_present[index] = value
    return EncodedObservation(tuple(raw_values), tuple(raw_present))


def fit(
    observations: list[EncodedObservation],
    *,
    thresholds: dict[str, float] | None = None,
    radius: float = 1.0,
) -> SemanticNormalizerV2:
    return SemanticNormalizerV2.fit(
        observations,
        partition_id="unit-train",
        capability_experience_thresholds=thresholds,
        local_movement_radius=radius,
    )


def test_fit_freeze_roundtrip(tmp_path) -> None:
    first = replace(make_observation(population=100), values={18: 400})
    second = replace(make_observation(population=200), values={18: 900})
    normalizer = fit([first, second])
    transformed = normalizer.transform(
        replace(make_observation(population=150), values={18: 600})
    )
    assert len(transformed) == 512
    assert all(math.isfinite(value) for value in transformed)
    assert transformed[80] == 1.0
    assert transformed[81] == 0.0

    path = tmp_path / "normalizer.json"
    normalizer.manifest.save(path)
    loaded = SemanticNormalizerV2(NormalizationManifest.load(path))
    assert loaded.transform(replace(make_observation(population=150), values={18: 600})) == (
        transformed
    )


def test_food_reserve_is_relative_to_native_decision_horizon() -> None:
    neutral = replace(make_observation(horizon=20), values={10: 20})
    one_physical_year = replace(neutral, values={10: 1})
    zero_reserve = replace(neutral, values={10: 0})
    normalizer = fit([neutral, one_physical_year, zero_reserve])

    neutral_stage2, neutral_present = normalizer.semantic_coordinates(neutral)
    year_stage2, _ = normalizer.semantic_coordinates(one_physical_year)
    zero_stage2, zero_present = normalizer.semantic_coordinates(zero_reserve)

    assert neutral_present[10]
    assert neutral_stage2[10] == pytest.approx(0.0)
    assert normalizer.transform(neutral)[10] == pytest.approx(0.0)
    assert year_stage2[10] == pytest.approx(math.log(1 / 20))
    assert year_stage2[10] < 0.0
    assert zero_present[10]
    assert math.isfinite(zero_stage2[10])
    assert zero_stage2[10] < year_stage2[10]


@pytest.mark.parametrize("missing_index", [3, 10])
def test_food_reserve_dependency_missing_is_zero_and_excluded(missing_index: int) -> None:
    observation = replace(
        make_observation(horizon=20),
        values={10: 40},
        present={missing_index: False},
    )
    normalizer = fit([observation])
    stage2, effective_present = normalizer.semantic_coordinates(observation)

    assert not effective_present[10]
    assert normalizer.transform(observation)[10] == 0.0
    assert stage2[10] <= 0.0
    assert normalizer.manifest.scales[10] == 1.0


def test_capability_experience_uses_semantic_per_capability_thresholds() -> None:
    values = {
        CAPABILITY_EXPERIENCE_OFFSETS_BY_KEY["field-retention"]: 100.0,
        CAPABILITY_EXPERIENCE_OFFSETS_BY_KEY["herd-management"]: 200.0,
        CAPABILITY_EXPERIENCE_OFFSETS_BY_KEY["resident-fishing"]: 50.0,
        CAPABILITY_EXPERIENCE_OFFSETS_BY_KEY["fixed-construction"]: 10_000.0,
        CAPABILITY_EXPERIENCE_OFFSETS_BY_KEY["local-trade"]: 10_000.0,
    }
    observation = replace(make_observation(), values=values)
    thresholds = {
        "resident-fishing": 200.0,
        "field-retention": 100.0,
        "herd-management": 100.0,
    }
    normalizer = fit([observation], thresholds=thresholds)
    stage2, effective_present = normalizer.semantic_coordinates(observation)

    assert stage2[65] == pytest.approx(0.0)
    assert stage2[68] == pytest.approx(math.log(2.0))
    assert stage2[71] == pytest.approx(math.log(0.25))
    assert effective_present[65]
    assert effective_present[68]
    assert effective_present[71]
    assert not effective_present[74]
    assert not effective_present[77]
    assert normalizer.transform(observation)[74] == 0.0
    assert normalizer.transform(observation)[77] == 0.0
    assert normalizer.manifest.capability_experience_thresholds == thresholds


@pytest.mark.parametrize("key", ["fixed-construction", "local-trade", "row-3"])
def test_capability_threshold_rejects_non_semantic_or_unsupported_keys(key: str) -> None:
    with pytest.raises(ValueError):
        fit([make_observation()], thresholds={key: 225.0})


def test_capability_experience_is_omitted_without_its_threshold() -> None:
    observation = replace(make_observation(), values={65: 10_000.0})
    normalizer = fit([observation])
    stage2, effective_present = normalizer.semantic_coordinates(observation)

    assert not effective_present[65]
    assert stage2[65] < 0.0
    assert normalizer.transform(observation)[65] == 0.0
    assert normalizer.manifest.scales[65] == 1.0


def test_signed_features_preserve_sign_through_asinh() -> None:
    signed = {13: -0.5, 15: 2.0, 59: -0.25, 402: -0.75, 403: 1.25, 405: -2.0}
    observation = replace(make_observation(), values=signed)
    normalizer = fit([observation])
    stage2, _ = normalizer.semantic_coordinates(observation)

    for index, raw in signed.items():
        assert stage2[index] == pytest.approx(np.arcsinh(raw))
        assert math.copysign(1.0, normalizer.transform(observation)[index]) == math.copysign(
            1.0, raw
        )


def test_herd_support_ratios_and_zero_capacity_limit_are_finite() -> None:
    ordinary = replace(make_observation(), values={24: 10, 28: 20, 29: 5, 30: 0})
    no_herd_no_capacity = replace(ordinary, values={24: 0, 28: 0, 29: 0, 30: 0})
    normalizer = fit([ordinary, no_herd_no_capacity])
    ordinary_stage2, present = normalizer.semantic_coordinates(ordinary)
    empty_stage2, empty_present = normalizer.semantic_coordinates(no_herd_no_capacity)

    assert ordinary_stage2[28] == pytest.approx(math.log(0.5))
    assert ordinary_stage2[29] == pytest.approx(math.log(2.0))
    assert ordinary_stage2[30] == pytest.approx(-math.log(1e-12))
    assert present[30]
    assert empty_present[28:31] == (True, True, True)
    assert all(math.isfinite(empty_stage2[index]) for index in (28, 29, 30))
    assert all(math.isfinite(normalizer.transform(ordinary)[index]) for index in (28, 29, 30))


def test_missing_herd_capacity_is_not_treated_as_zero_capacity() -> None:
    observation = replace(
        make_observation(), values={24: 10, 28: 0}, present={28: False}
    )
    normalizer = fit([observation])
    _, effective_present = normalizer.semantic_coordinates(observation)

    assert not effective_present[28]
    assert normalizer.transform(observation)[28] == 0.0
    assert normalizer.manifest.scales[28] == 1.0


def test_contact_and_fishing_distance_use_local_movement_radius() -> None:
    observation = replace(make_observation(), values={212: 3.0, 400: 6.0})
    normalizer = fit([observation], radius=3.0)
    stage2, _ = normalizer.semantic_coordinates(observation)

    assert stage2[212] == pytest.approx(np.arcsinh(1.0))
    assert stage2[400] == pytest.approx(np.arcsinh(2.0))
    assert normalizer.manifest.local_movement_radius == 3.0


def test_bounded_values_are_affine_and_binary_masks_stay_binary() -> None:
    observation = replace(
        make_observation(),
        values={5: 0.0, 6: 0.5, 7: 1.0, 80: 1.0, 81: 0.0, 482: 1.0, 483: 0.0},
    )
    transformed = fit([observation]).transform(observation)

    assert transformed[5:8] == pytest.approx((-1.0, 0.0, 1.0))
    assert transformed[80] == 1.0
    assert transformed[81] == 0.0
    assert transformed[482] == 1.0
    assert transformed[483] == 0.0


def test_missing_values_are_excluded_from_scale_fit_and_output_zero() -> None:
    observed = replace(make_observation(), values={18: 2_000.0})
    huge_but_missing = replace(
        make_observation(), values={18: 1e300}, present={18: False}
    )
    baseline = fit([observed])
    with_missing = fit([observed, huge_but_missing])

    assert with_missing.manifest.scales[18] == baseline.manifest.scales[18]
    assert with_missing.transform(huge_but_missing)[18] == 0.0
