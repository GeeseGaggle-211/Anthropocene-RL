from __future__ import annotations

from dataclasses import dataclass

from anthropocene_rl.contracts import TENSOR_FEATURE_COUNT


@dataclass(frozen=True)
class EncodedObservation:
    values: tuple[float, ...]
    present: tuple[bool, ...]

    def validate(self) -> None:
        if len(self.values) != TENSOR_FEATURE_COUNT:
            raise ValueError("Unexpected Elder tensor width")
        if len(self.present) != TENSOR_FEATURE_COUNT:
            raise ValueError("Unexpected Elder presence-mask width")
