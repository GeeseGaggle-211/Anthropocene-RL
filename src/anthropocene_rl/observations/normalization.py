"""Frozen schema-v2 semantic normalization.

The normalizer is fit once from the declared training partition and serialized
with the checkpoint. Inference never updates running statistics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

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

_BINARY = "binary"
_BOUNDED = "bounded01"
_SIGNED = "signed"
_NONNEGATIVE = "nonnegative"


def _kinds() -> tuple[str, ...]:
    kinds = [_NONNEGATIVE] * TENSOR_FEATURE_COUNT

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
    }
    signed: set[int] = {
        13,
        15,
        59,
        *range(99, 102),
        *range(111, 114),
        *[400 + 8 * row + col for row in range(8) for col in (2, 3, 5)],
    }

    for index in binary:
        kinds[index] = _BINARY
    for index in bounded:
        kinds[index] = _BOUNDED
    for index in signed:
        kinds[index] = _SIGNED
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
    target_quantile: float = _TARGET_QUANTILE
    target_saturation: float = _TARGET_SATURATION

    def save(self, path: str | Path) -> None:
        payload = asdict(self)
        payload["scales"] = list(self.scales)
        payload["kinds"] = list(self.kinds)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationManifest":
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
    ) -> "SemanticNormalizerV2":
        batch = list(observations)
        if not batch:
            raise ValueError("Cannot fit normalization on an empty partition")
        stage2_rows: list[np.ndarray] = []
        present_rows: list[np.ndarray] = []
        for observation in batch:
            observation.validate()
            stage1 = _semantic_reference(np.asarray(observation.values, dtype=np.float64))
            stage2_rows.append(_stage2(stage1))
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
            )
        )

    def transform(self, observation: EncodedObservation) -> tuple[float, ...]:
        observation.validate()
        stage1 = _semantic_reference(np.asarray(observation.values, dtype=np.float64))
        stage2 = _stage2(stage1)
        output = np.zeros(TENSOR_FEATURE_COUNT, dtype=np.float64)
        present = np.asarray(observation.present, dtype=bool)
        for index, kind in enumerate(self.manifest.kinds):
            if not present[index]:
                continue
            if kind == _BINARY:
                output[index] = 1.0 if stage2[index] > 0.5 else 0.0
            elif kind == _BOUNDED:
                output[index] = float(np.clip(stage2[index], 0.0, 1.0))
            else:
                output[index] = float(np.tanh(stage2[index] / self._scales[index]))
        if not np.all(np.isfinite(output)):
            raise ValueError("Normalization produced NaN/Inf")
        return tuple(float(value) for value in output)


def _semantic_reference(values: np.ndarray) -> np.ndarray:
    if values.shape != (TENSOR_FEATURE_COUNT,):
        raise ValueError("Unexpected Elder tensor shape")
    result = values.copy()
    population = max(float(result[0]), 1.0)
    horizon = max(float(result[3]), 1e-6)
    person_years = population * horizon

    result[1] /= population
    result[16] /= 7.0
    result[18] /= person_years

    for offset in (24, 28, 29, 30, 32, 33):
        result[offset] /= population
    result[31] /= person_years
    result[53] = max(result[53], 0.0)
    result[62] /= population

    for row in range(3):
        base = 212 + 12 * row
        result[base + 10] /= person_years

    for row in range(8):
        result[248 + 8 * row] /= population

    for row in range(11):
        base = 312 + 8 * row
        for col in (0, 1, 2, 3):
            result[base + col] /= population
        for col in (4, 5):
            result[base + col] /= person_years

    capacities = (7.0, 3.0, 8.0, 11.0, 8.0)
    for row, capacity in enumerate(capacities):
        base = 464 + 3 * row
        result[base] /= capacity
        result[base + 1] /= capacity
    result[481] /= 8.0
    return result


def _stage2(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    for index, kind in enumerate(KINDS):
        value = float(result[index])
        if kind in (_BINARY, _BOUNDED):
            continue
        if kind == _SIGNED:
            result[index] = np.arcsinh(value)
        else:
            result[index] = np.log1p(max(0.0, value))
    return result
