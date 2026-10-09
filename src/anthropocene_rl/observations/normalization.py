"""Normalization-v1 interface.

The normalization *design* is normative in section 34 of the Anthropocene RL
design document: semantic/reference-relative Stage 1, monotonic Stage 2, tanh
soft saturation, and frozen training-partition calibration.

Implementation remains intentionally incomplete until the native C# observation
plumbing matches those semantics. No running inference normalizer is allowed.
"""

from __future__ import annotations

from typing import Protocol

from .schema import EncodedObservation


class ObservationNormalizer(Protocol):
    version: str

    def transform(self, observation: EncodedObservation) -> tuple[float, ...]: ...
