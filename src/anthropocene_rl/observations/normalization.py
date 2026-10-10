"""Frozen schema-v2 semantic normalization.

The normalizer is fit once from the declared training partition and serialized
with the checkpoint. Inference never updates running statistics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
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

_BINARY = "binary"
_BOUNDED = "bounded01"
_SIGNED = "signed_asinh"
_POSITIVE = "positive_log1p"
_RATIO = "positive_ratio_log"
_DISTANCE = "distance_asinh"
_CAPABILITY_EXPERIENCE = "capability_experience"


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
    capability_experience_threshold: float | None = None
    local_movement_radius: float = 1.0
    target_quantile: float = _TARGET_QUANTILE
    target_saturation: float = _TARGET_SATURATION

    def save(self, path: str | Path) -> None:
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
        if self.capability_experience_threshold is not None and (
            not np.isfinite(self.capability_experience_threshold)
            or self.capability_experience_threshold <= 0
        ):
            raise ValueError("Capability Experience threshold must be positive")
        if not np.isfinite(self.local_movement_radius) or self.local_movement_radius <= 0:
            raise ValueError("Local movement radius must be finite and positive")


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
        capability_experience_threshold: float | None = None,
        local_movement_radius: float = 1.0,
    ) -> SemanticNormalizerV2:
        batch = list(observations)
        if not batch:
            raise ValueError("Cannot fit normalization on an empty partition")
        stage2_rows: list[np.ndarray] = []
        present_rows: list[np.ndarray] = []
        for observation in batch:
            observation.validate()
            stage1 = _semantic_reference(
                np.asarray(observation.values, dtype=np.float64),
                capability_experience_threshold=capability_experience_threshold,
                local_movement_radius=local_movement_radius,
            )
            stage2_rows.append(
                _stage2(
                    stage1,
                    capability_experience_enabled=(capability_experience_threshold is not None),
                )
            )
            present_rows.append(np.asarray(observation.present, dtype=bool))
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
                capability_experience_threshold=capability_experience_threshold,
                local_movement_radius=local_movement_radius,
            )
        )

    def transform(self, observation: EncodedObservation) -> tuple[float, ...]:
        observation.validate()
        stage1 = _semantic_reference(
            np.asarray(observation.values, dtype=np.float64),
            capability_experience_threshold=(self.manifest.capability_experience_threshold),
            local_movement_radius=self.manifest.local_movement_radius,
        )
        stage2 = _stage2(
            stage1,
            capability_experience_enabled=(
                self.manifest.capability_experience_threshold is not None
            ),
        )
        output = np.zeros(TENSOR_FEATURE_COUNT, dtype=np.float64)
        present = np.asarray(observation.present, dtype=bool)
        for index, kind in enumerate(self.manifest.kinds):
            if not present[index]:
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


def _semantic_reference(
    values: np.ndarray,
    *,
    capability_experience_threshold: float | None,
    local_movement_radius: float,
) -> np.ndarray:
    if values.shape != (TENSOR_FEATURE_COUNT,):
        raise ValueError("Unexpected Elder tensor shape")
    result = values.copy()
    population = max(float(result[0]), 1.0)
    horizon = max(float(result[3]), 1e-6)
    person_years = population * horizon

    result[1] /= population
    result[16] /= 7.0
    result[18] /= person_years

    animals = max(float(values[24]), 0.0)
    result[24] /= population
    for offset in (28, 29, 30):
        capacity = max(float(values[offset]), 0.0)
        result[offset] = animals / max(capacity, _RATIO_FLOOR) if animals > 0.0 else 0.0
    result[31] /= person_years
    result[32] /= person_years
    result[33] /= person_years
    result[53] = max(result[53], 0.0)
    result[62] /= population

    for row in range(5):
        offset = 65 + 3 * row
        result[offset] = (
            result[offset] / capability_experience_threshold
            if capability_experience_threshold is not None
            else 0.0
        )

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
    return result


def _stage2(
    values: np.ndarray,
    *,
    capability_experience_enabled: bool,
) -> np.ndarray:
    result = values.copy()
    for index, kind in enumerate(KINDS):
        value = float(result[index])
        if kind in (_BINARY, _BOUNDED):
            continue
        if kind in (_SIGNED, _DISTANCE):
            result[index] = np.arcsinh(value)
        elif kind == _RATIO:
            result[index] = np.log(max(_RATIO_FLOOR, value))
        elif kind == _CAPABILITY_EXPERIENCE:
            result[index] = (
                np.log(max(_RATIO_FLOOR, value)) if capability_experience_enabled else 0.0
            )
        else:
            result[index] = np.log1p(max(0.0, value))
    return result
