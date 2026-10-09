"""Learner-visible observation tensor types.

`EncodedObservation` is only the fixed-width neural tensor plus its presence
mask. It is deliberately not the full environment-step envelope: action masks,
agent-local physical delta-t, trusted action metadata, schema/replay identities,
and interventions belong to the adapter/environment context once that contract
is locked.
"""

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
