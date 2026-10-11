"""Learner-visible schema-v2 tensor and action-mask helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from anthropocene_rl.contracts import (
    ACTION_SCHEMA_VERSION,
    DOMESTIC_HERD_COUNT_OFFSET,
    FIELD_DEVELOPMENT_MASK_OFFSET,
    FIELD_RETENTION_MASK_OFFSET,
    FISHING_CAPABILITY_AVAILABLE_OFFSET,
    FISHING_PRESENT_COUNT_OFFSET,
    HERD_DEVELOPMENT_MASK_OFFSET,
    HERD_RETENTION_MASK_OFFSET,
    HUSBANDRY_CAPABILITY_AVAILABLE_OFFSET,
    MOVEMENT_CANDIDATE_COUNT,
    MOVEMENT_CANDIDATE_MASK_OFFSET,
    MOVEMENT_ENABLED_OFFSET,
    OBSERVATION_SCHEMA_VERSION,
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
        schema_version = int(_pick(payload, "SchemaVersion", "schemaVersion", "schema_version"))
        if schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(f"Unexpected Elder observation schema version: {schema_version}")
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
    def from_observation(cls, observation: EncodedObservation) -> StructuralActionMask:
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

    def canonicalize_continuous(self, values: Sequence[float]) -> tuple[float, float, float, float]:
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


def cultivation_only_scope_violations(observation: EncodedObservation) -> tuple[str, ...]:
    """Return excluded domains that became physically or structurally active."""

    values = observation.values
    mask = StructuralActionMask.from_observation(observation)
    violations: list[str] = []
    if values[HUSBANDRY_CAPABILITY_AVAILABLE_OFFSET] > 0.5:
        violations.append("husbandry-capability-available")
    if values[DOMESTIC_HERD_COUNT_OFFSET] > 1e-9:
        violations.append("domestic-herd-present")
    if mask.herd_retention:
        violations.append("herd-retention-mask-active")
    if mask.herd_development:
        violations.append("herd-development-mask-active")
    if values[FISHING_CAPABILITY_AVAILABLE_OFFSET] > 0.5:
        violations.append("fishing-capability-available")
    if values[FISHING_PRESENT_COUNT_OFFSET] > 1e-9:
        violations.append("fishing-ground-present")
    if values[MOVEMENT_ENABLED_OFFSET] > 0.5 or any(mask.movement_candidates[1:]):
        violations.append("movement-enabled")
    return tuple(violations)


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
