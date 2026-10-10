import math

from anthropocene_rl.observations.normalization import (
    NormalizationManifest,
    SemanticNormalizerV2,
)
from anthropocene_rl.observations.schema import EncodedObservation


def make_observation(population: float, maintenance: float) -> EncodedObservation:
    values = [0.0] * 512
    present = [True] * 512
    values[0] = population
    values[3] = 20.0
    values[5] = 0.8
    values[18] = maintenance
    values[80] = 1.0
    values[82] = 1.0
    values[482] = 1.0
    return EncodedObservation(tuple(values), tuple(present))


def test_fit_freeze_roundtrip(tmp_path) -> None:
    normalizer = SemanticNormalizerV2.fit(
        [make_observation(100, 400), make_observation(200, 900)],
        partition_id="unit-train",
    )
    transformed = normalizer.transform(make_observation(150, 600))
    assert len(transformed) == 512
    assert all(math.isfinite(value) for value in transformed)
    assert transformed[80] == 1.0
    assert transformed[81] == 0.0

    path = tmp_path / "normalizer.json"
    normalizer.manifest.save(path)
    loaded = SemanticNormalizerV2(NormalizationManifest.load(path))
    assert loaded.transform(make_observation(150, 600)) == transformed


def test_approved_semantic_references_and_transforms() -> None:
    values = [0.0] * 512
    present = [True] * 512
    values[0] = 100.0
    values[3] = 20.0
    values[5] = 0.25
    values[59] = -0.5
    values[65] = 225.0
    values[400] = 3.0
    values[482] = 1.0
    observation = EncodedObservation(tuple(values), tuple(present))
    normalizer = SemanticNormalizerV2.fit(
        [observation],
        partition_id="semantic-contract",
        capability_experience_threshold=225.0,
        local_movement_radius=3.0,
    )
    transformed = normalizer.transform(observation)

    assert transformed[5] == -0.5
    assert transformed[59] < 0.0
    assert transformed[65] == 0.0
    assert transformed[400] > 0.0
    assert normalizer.manifest.capability_experience_threshold == 225.0
    assert normalizer.manifest.local_movement_radius == 3.0


def test_capability_experience_is_removed_without_a_threshold() -> None:
    observation = make_observation(100, 400)
    values = list(observation.values)
    values[65] = 10_000.0
    observation = EncodedObservation(tuple(values), observation.present)
    normalizer = SemanticNormalizerV2.fit(
        [observation],
        partition_id="no-capability-threshold",
    )
    assert normalizer.transform(observation)[65] == 0.0
