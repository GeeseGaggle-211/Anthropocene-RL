"""Frozen schema-v2 semantic normalization.

The normalizer is fit once from the declared training partition and serialized
with the checkpoint. Inference never updates running statistics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from anthropocene_rl.contracts import (
    OBSERVATION_SCHEMA_VERSION,
    SIMULATOR_COMMIT,
    TENSOR_FEATURE_COUNT,
)
from anthropocene_rl.observations.schema import EncodedObservation

_NORMALIZATION_VERSION = "elder-normalization-v2"
_TARGET_QUANTILE = 0.95
_TARGET_SATURATION = 0.80
_ATANH_TARGET = float(np.arctanh(_TARGET_SATURATION))
_RATIO_FLOOR = 1e-12
_RATIO_CEILING = 1.0 / _RATIO_FLOOR

_BINARY = "binary"
_BOUNDED = "bounded01"
_SIGNED = "signed_asinh"
_POSITIVE = "positive_log1p"
_RATIO = "positive_ratio_log"
_DISTANCE = "distance_asinh"
_CAPABILITY_EXPERIENCE = "capability_experience"

CAPABILITY_EXPERIENCE_OFFSETS_BY_KEY: Mapping[str, int] = {
    "field-retention": 65,
    "herd-management": 68,
    "resident-fishing": 71,
    "fixed-construction": 74,
    "local-trade": 77,
}
CAPABILITIES_WITH_PROGRESSION_THRESHOLDS = frozenset(
    {"field-retention", "herd-management", "resident-fishing"}
)


def _kinds() -> tuple[str, ...]:
    kinds = [_POSITIVE] * TENSOR_FEATURE_COUNT

    binary: set[int] = {
        14,
        40,
        55,
        56,
        57,
        *[64 + 3 * row for row in range(5)],
        *range(80, 84),
        *[128 + 12 * row + col for row in range(7) for col in (0, 1)],
        *[212 + 12 * row + col for row in range(3) for col in (1, 2)],
        *[312 + 8 * row + col for row in range(11) for col in (6, 7)],
        *[400 + 8 * row + col for row in range(8) for col in (6, 7)],
        *[464 + 3 * row + 2 for row in range(5)],
        480,
        *range(482, 490),
    }
    bounded: set[int] = {
        16,
        *range(5, 10),
        17,
        23,
        26,
        27,
        34,
        35,
        36,
        *range(41, 53),
        58,
        60,
        61,
        *[66 + 3 * row for row in range(5)],
        *range(84, 93),
        *range(114, 120),
        *[212 + 12 * row + col for row in range(3) for col in (6, 8, 9)],
        *[248 + 8 * row + col for row in range(8) for col in (1, 2, 6, 7)],
        *[464 + 3 * row for row in range(5)],
        481,
    }
    signed: set[int] = {
        13,
        15,
        59,
        *range(99, 102),
        *range(111, 114),
        *[400 + 8 * row + col for row in range(8) for col in (2, 3, 5)],
    }
    ratios: set[int] = {
        1,
        10,
        *range(19, 23),
        *range(28, 31),
        *range(93, 99),
        *range(102, 111),
        *[128 + 12 * row + col for row in range(7) for col in range(2, 12)],
    }
    distances: set[int] = {
        *[212 + 12 * row for row in range(3)],
        *[400 + 8 * row for row in range(8)],
    }
    capability_experience = {65 + 3 * row for row in range(5)}

    for index in binary:
        kinds[index] = _BINARY
    for index in bounded:
        kinds[index] = _BOUNDED
    for index in signed:
        kinds[index] = _SIGNED
    for index in ratios:
        kinds[index] = _RATIO
    for index in distances:
        kinds[index] = _DISTANCE
    for index in capability_experience:
        kinds[index] = _CAPABILITY_EXPERIENCE
    return tuple(kinds)


KINDS = _kinds()


@dataclass(frozen=True)
class NormalizationManifest:
    version: str
    simulator_commit: str
    observation_schema_version: int
    partition_id: str
    scales: tuple[float, ...]
    kinds: tuple[str, ...]
    capability_experience_thresholds: dict[str, float] = field(default_factory=dict)
    local_movement_radius: float = 1.0
    target_quantile: float = _TARGET_QUANTILE
    target_saturation: float = _TARGET_SATURATION
    partition_manifest_sha256: str | None = None
    configuration_hash: str | None = None
    build_hash: str | None = None
    observation_schema_hash: str | None = None
    fit_observation_count: int | None = None
    fit_source_mix: dict[str, int] = field(default_factory=dict)
    rng_provenance: str | None = None

    def save(self, path: str | Path) -> None:
        self.validate()
        payload = asdict(self)
        payload["scales"] = list(self.scales)
        payload["kinds"] = list(self.kinds)
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> NormalizationManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["scales"] = tuple(float(value) for value in payload["scales"])
        payload["kinds"] = tuple(str(value) for value in payload["kinds"])
        payload["capability_experience_thresholds"] = {
            str(key): float(value)
            for key, value in payload.get("capability_experience_thresholds", {}).items()
        }
        payload["fit_source_mix"] = {
            str(key): int(value) for key, value in payload.get("fit_source_mix", {}).items()
        }
        result = cls(**payload)
        result.validate()
        return result

    def validate(self) -> None:
        if self.version != _NORMALIZATION_VERSION:
            raise ValueError(f"Unsupported normalizer version: {self.version}")
        if self.simulator_commit != SIMULATOR_COMMIT:
            raise ValueError("Normalizer was fit against a different simulator commit")
        if self.observation_schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError("Normalizer observation schema mismatch")
        if len(self.scales) != TENSOR_FEATURE_COUNT or len(self.kinds) != TENSOR_FEATURE_COUNT:
            raise ValueError("Normalizer manifest has the wrong tensor width")
        if any((not np.isfinite(scale)) or scale <= 0 for scale in self.scales):
            raise ValueError("Normalizer scales must be finite and positive")
        unknown = set(self.capability_experience_thresholds) - set(
            CAPABILITY_EXPERIENCE_OFFSETS_BY_KEY
        )
        if unknown:
            raise ValueError(f"Unknown semantic capability threshold keys: {sorted(unknown)}")
        unsupported = set(self.capability_experience_thresholds) - set(
            CAPABILITIES_WITH_PROGRESSION_THRESHOLDS
        )
        if unsupported:
            raise ValueError(
                "Capability rows without a progression threshold must omit Experience: "
                f"{sorted(unsupported)}"
            )
        if any(
            not np.isfinite(threshold) or threshold <= 0
            for threshold in self.capability_experience_thresholds.values()
        ):
            raise ValueError("Capability Experience thresholds must be finite and positive")
        if not np.isfinite(self.local_movement_radius) or self.local_movement_radius <= 0:
            raise ValueError("Local movement radius must be finite and positive")
        if self.fit_observation_count is not None and self.fit_observation_count < 1:
            raise ValueError("Fit observation count must be positive when recorded")
        if any(count < 0 for count in self.fit_source_mix.values()):
            raise ValueError("Fit source counts cannot be negative")


class SemanticNormalizerV2:
    version = _NORMALIZATION_VERSION

    def __init__(self, manifest: NormalizationManifest) -> None:
        manifest.validate()
        self.manifest = manifest
        self._scales = np.asarray(manifest.scales, dtype=np.float64)

    @classmethod
    def fit(
        cls,
        observations: Iterable[EncodedObservation],
        *,
        partition_id: str,
        capability_experience_thresholds: Mapping[str, float] | None = None,
        local_movement_radius: float = 1.0,
        partition_manifest_sha256: str | None = None,
        configuration_hash: str | None = None,
        build_hash: str | None = None,
        observation_schema_hash: str | None = None,
        fit_source_mix: Mapping[str, int] | None = None,
        rng_provenance: str | None = None,
    ) -> SemanticNormalizerV2:
        batch = list(observations)
        if not batch:
            raise ValueError("Cannot fit normalization on an empty partition")
        thresholds = {
            str(key): float(value)
            for key, value in (capability_experience_thresholds or {}).items()
        }
        stage2_rows: list[np.ndarray] = []
        present_rows: list[np.ndarray] = []
        for observation in batch:
            observation.validate()
            stage1, effective_present = _semantic_reference(
                np.asarray(observation.values, dtype=np.float64),
                np.asarray(observation.present, dtype=bool),
                capability_experience_thresholds=thresholds,
                local_movement_radius=local_movement_radius,
            )
            stage2_rows.append(_stage2(stage1))
            present_rows.append(effective_present)
        values = np.stack(stage2_rows)
        present = np.stack(present_rows)

        scales = np.ones(TENSOR_FEATURE_COUNT, dtype=np.float64)
        for index, kind in enumerate(KINDS):
            if kind in (_BINARY, _BOUNDED):
                continue
            observed = np.abs(values[:, index][present[:, index]])
            if observed.size == 0:
                continue
            quantile = float(np.quantile(observed, _TARGET_QUANTILE))
            if quantile > 1e-12:
                scales[index] = max(quantile / _ATANH_TARGET, 1e-6)

        return cls(
            NormalizationManifest(
                version=_NORMALIZATION_VERSION,
                simulator_commit=SIMULATOR_COMMIT,
                observation_schema_version=OBSERVATION_SCHEMA_VERSION,
                partition_id=partition_id,
                scales=tuple(float(value) for value in scales),
                kinds=KINDS,
                capability_experience_thresholds=thresholds,
                local_movement_radius=local_movement_radius,
                partition_manifest_sha256=partition_manifest_sha256,
                configuration_hash=configuration_hash,
                build_hash=build_hash,
                observation_schema_hash=observation_schema_hash,
                fit_observation_count=len(batch),
                fit_source_mix={
                    str(key): int(value) for key, value in (fit_source_mix or {}).items()
                },
                rng_provenance=rng_provenance,
            )
        )

    def transform(self, observation: EncodedObservation) -> tuple[float, ...]:
        observation.validate()
        stage1, effective_present = _semantic_reference(
            np.asarray(observation.values, dtype=np.float64),
            np.asarray(observation.present, dtype=bool),
            capability_experience_thresholds=(
                self.manifest.capability_experience_thresholds
            ),
            local_movement_radius=self.manifest.local_movement_radius,
        )
        stage2 = _stage2(stage1)
        output = np.zeros(TENSOR_FEATURE_COUNT, dtype=np.float64)
        for index, kind in enumerate(self.manifest.kinds):
            if not effective_present[index]:
                continue
            if kind == _BINARY:
                output[index] = 1.0 if stage2[index] > 0.5 else 0.0
            elif kind == _BOUNDED:
                output[index] = float(2.0 * np.clip(stage2[index], 0.0, 1.0) - 1.0)
            else:
                output[index] = float(np.tanh(stage2[index] / self._scales[index]))
        if not np.all(np.isfinite(output)):
            raise ValueError("Normalization produced NaN/Inf")
        return tuple(float(value) for value in output)

    def semantic_coordinates(
        self, observation: EncodedObservation
    ) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        """Return pre-tanh Stage-2 coordinates and their effective presence.

        This is an audit surface. It applies the same dependency-aware missing
        policy used for scale fitting and inference.
        """

        observation.validate()
        stage1, effective_present = _semantic_reference(
            np.asarray(observation.values, dtype=np.float64),
            np.asarray(observation.present, dtype=bool),
            capability_experience_thresholds=(
                self.manifest.capability_experience_thresholds
            ),
            local_movement_radius=self.manifest.local_movement_radius,
        )
        stage2 = _stage2(stage1)
        return (
            tuple(float(value) for value in stage2),
            tuple(bool(value) for value in effective_present),
        )


def _semantic_reference(
    values: np.ndarray,
    present: np.ndarray,
    *,
    capability_experience_thresholds: Mapping[str, float],
    local_movement_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    if values.shape != (TENSOR_FEATURE_COUNT,):
        raise ValueError("Unexpected Elder tensor shape")
    if present.shape != (TENSOR_FEATURE_COUNT,):
        raise ValueError("Unexpected Elder presence shape")
    result = values.copy()
    effective_present = present.copy()
    population = max(float(result[0]), 1.0)
    horizon = max(float(result[3]), 1e-6)
    person_years = population * horizon

    result[1] /= population
    # FoodReserveYears is a physical duration. Its semantic boundary is one
    # *native decision interval*, not one arbitrary physical year. A genuine
    # zero reserve remains present and maps through the explicit ratio floor;
    # a missing/non-positive horizon makes the derived coordinate missing.
    if effective_present[10] and effective_present[3] and values[3] > 0:
        result[10] = max(0.0, float(values[10])) / float(values[3])
    else:
        result[10] = 0.0
        effective_present[10] = False
    result[16] /= 7.0
    result[18] /= person_years

    animals = max(float(values[24]), 0.0)
    result[24] /= population
    for offset in (28, 29, 30):
        capacity = max(float(values[offset]), 0.0)
        if not effective_present[24] or not effective_present[offset]:
            result[offset] = 0.0
            effective_present[offset] = False
        elif animals <= 0.0:
            result[offset] = 0.0
        elif capacity <= 0.0:
            # Positive demand against exactly zero measured support is the
            # mathematical +infinity boundary. Represent the limit explicitly
            # instead of dividing animal count by an arbitrary epsilon.
            result[offset] = _RATIO_CEILING
        else:
            result[offset] = animals / capacity
    result[31] /= person_years
    result[32] /= person_years
    result[33] /= person_years
    result[53] = max(result[53], 0.0)
    result[62] /= population

    for key, offset in CAPABILITY_EXPERIENCE_OFFSETS_BY_KEY.items():
        threshold = capability_experience_thresholds.get(key)
        if threshold is None:
            result[offset] = 0.0
            effective_present[offset] = False
        else:
            result[offset] /= threshold

    for row in range(3):
        base = 212 + 12 * row
        result[base] /= local_movement_radius
        result[base + 4] /= population
        result[base + 5] /= population
        result[base + 10] /= person_years

    for row in range(8):
        result[248 + 8 * row] /= population

    for row in range(11):
        base = 312 + 8 * row
        for col in (0, 1, 2, 3):
            result[base + col] /= population
        for col in (4, 5):
            result[base + col] /= person_years

    for row in range(8):
        result[400 + 8 * row] /= local_movement_radius

    capacities = (7.0, 3.0, 8.0, 11.0, 8.0)
    for row, capacity in enumerate(capacities):
        base = 464 + 3 * row
        result[base] /= capacity
        result[base + 1] /= capacity
    result[481] /= 8.0
    return result, effective_present


def _stage2(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    for index, kind in enumerate(KINDS):
        value = float(result[index])
        if kind in (_BINARY, _BOUNDED):
            continue
        if kind in (_SIGNED, _DISTANCE):
            result[index] = np.arcsinh(value)
        elif kind in (_RATIO, _CAPABILITY_EXPERIENCE):
            result[index] = np.log(max(_RATIO_FLOOR, value))
        else:
            result[index] = np.log1p(max(0.0, value))
    return result
