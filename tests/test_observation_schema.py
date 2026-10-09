import pytest

from anthropocene_rl.observations.schema import EncodedObservation


def test_tensor_width_validation() -> None:
    EncodedObservation((0.0,) * 512, (False,) * 512).validate()
    with pytest.raises(ValueError):
        EncodedObservation((0.0,) * 511, (False,) * 512).validate()
