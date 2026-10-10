"""Learner-visible schema-v2 tensor and action-mask helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from anthropocene_rl.contracts import (
    ACTION_SCHEMA_VERSION,
    FIELD_DEVELOPMENT_MASK_OFFSET,
    FIELD_RETENTION_MASK_OFFSET,
    HERD_DEVELOPMENT_MASK_OFFSET,
    HERD_RETENTION_MASK_OFFSET,
    MOVEMENT_CANDIDATE_COUNT,
    MOVEMENT_CANDIDATE_MASK_OFFSET,
    TENSOR_FEATURE_COUNT,
)


def _pick(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"Missing required key; tried {names!r}")


@dataclass(frozen=True)
class EncodedObservation:
    values: tuple[float, ...]
    present: tuple[bool, ...]

    @classmethod
    def from_native(cls, payload: Mapping[str, Any]) -> EncodedObservation:
        values = tuple(float(value) for value in _pick(payload, "Values", "values"))
        present = tuple(bool(value) for value in _pick(payload, "Present", "present"))
        result = cls(values=values, present=present)
        result.validate()
        return result

    def validate(self) -> None:
        if len(self.values) != TENSOR_FEATURE_COUNT:
            raise ValueError("Unexpected Elder tensor width")
        if len(self.present) != TENSOR_FEATURE_COUNT:
            raise ValueError("Unexpected Elder presence-mask width")
        if any(not isfinite(value) for value in self.values):
            raise ValueError("Elder tensor contains NaN/Inf")


@dataclass(frozen=True)
class StructuralActionMask:
    field_retention: bool
    herd_retention: bool
    field_development: bool
    herd_development: bool
    movement_candidates: tuple[bool, ...]

    @classmethod
    def from_observation(
        cls, observation: EncodedObservation
    ) -> StructuralActionMask:
        values = observation.values
        movement = tuple(
            values[MOVEMENT_CANDIDATE_MASK_OFFSET + index] > 0.5
            for index in range(MOVEMENT_CANDIDATE_COUNT)
        )
        if not movement[0]:
            raise ValueError("Native movement mask must always allow explicit stay")
        return cls(
            field_retention=values[FIELD_RETENTION_MASK_OFFSET] > 0.5,
            herd_retention=values[HERD_RETENTION_MASK_OFFSET] > 0.5,
            field_development=values[FIELD_DEVELOPMENT_MASK_OFFSET] > 0.5,
            herd_development=values[HERD_DEVELOPMENT_MASK_OFFSET] > 0.5,
            movement_candidates=movement,
        )

    @property
    def development_active(self) -> bool:
        return self.field_development or self.herd_development

    @property
    def development_split_active(self) -> bool:
        return self.field_development and self.herd_development

    def canonicalize_continuous(
        self, values: Sequence[float]
    ) -> tuple[float, float, float, float]:
        if len(values) != 4:
            raise ValueError("Expected four Elder continuous controls")
        field_retention = float(values[0]) if self.field_retention else 0.0
        herd_retention = float(values[1]) if self.herd_retention else 0.0
        development = float(values[2]) if self.development_active else 0.0
        if self.development_split_active:
            split = float(values[3])
        elif self.field_development:
            split = 1.0
        elif self.herd_development:
            split = 0.0
        else:
            split = 0.5
        return field_retention, herd_retention, development, split


@dataclass(frozen=True)
class ElderDirective:
    retained_field_operation: float
    retained_herd_fraction: float
    development_budget: float
    field_development_share: float
    movement_candidate: int = -1

    def to_native(self) -> dict[str, float | int]:
        return {
            "SchemaVersion": ACTION_SCHEMA_VERSION,
            "RetainedFieldOperation": self.retained_field_operation,
            "RetainedHerdFraction": self.retained_herd_fraction,
            "DevelopmentBudget": self.development_budget,
            "FieldDevelopmentShare": self.field_development_share,
            "MovementCandidate": self.movement_candidate,
        }
