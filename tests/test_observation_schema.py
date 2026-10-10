import pytest

from anthropocene_rl.observations.schema import EncodedObservation, StructuralActionMask


def observation(values: list[float] | None = None) -> EncodedObservation:
    data = [0.0] * 512 if values is None else values
    present = [True] * 512
    data[482] = 1.0
    return EncodedObservation(tuple(data), tuple(present))


def test_tensor_width_validation() -> None:
    observation().validate()
    with pytest.raises(ValueError):
        EncodedObservation((0.0,) * 511, (False,) * 512).validate()


def test_native_tensor_requires_exact_schema_version() -> None:
    payload = {
        "SchemaVersion": 2,
        "Values": [0.0] * 512,
        "Present": [True] * 512,
    }
    assert len(EncodedObservation.from_native(payload).values) == 512
    payload["SchemaVersion"] = 1
    with pytest.raises(ValueError, match="schema version"):
        EncodedObservation.from_native(payload)


def test_structural_mask_canonicalizes_field_only_development() -> None:
    values = [0.0] * 512
    values[80] = 1.0
    values[82] = 1.0
    values[482] = 1.0
    mask = StructuralActionMask.from_observation(observation(values))
    assert mask.field_retention
    assert not mask.herd_retention
    assert mask.field_development
    assert not mask.herd_development
    assert mask.development_active
    assert not mask.development_split_active
    assert mask.canonicalize_continuous((0.4, 0.9, 0.6, 0.0)) == (0.4, 0.0, 0.6, 1.0)
