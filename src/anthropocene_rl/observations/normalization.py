"""Normalization-v1 placeholder.

Implementation is intentionally blocked until the section-34 Yellow approvals
and Red semantic fixes are resolved. No running inference normalizer is allowed.
"""

from __future__ import annotations

from typing import Protocol

from .schema import EncodedObservation


class ObservationNormalizer(Protocol):
    version: str

    def transform(self, observation: EncodedObservation) -> tuple[float, ...]: ...
