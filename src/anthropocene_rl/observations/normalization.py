"""Frozen schema-v2 semantic normalization.

The normalizer is fit once from the declared training partition and serialized
with the checkpoint. Inference never updates running statistics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from anthropocene_rl.contracts import (
    OBSERVATION_SCHEMA_VERSION,
    SIMULATOR_COMMIT,
    TENSOR_FEATURE_COUNT,
)
from anthropocene_rl.observations.schema import EncodedObservation

NORMALIZATION_VERSION_V2 = "elder-normalization-v2"
NORMALIZATION_VERSION_V3 = "elder-normalization-v3"
SUPPORTED_NORMALIZATION_VERSIONS = frozenset({NORMALIZATION_VERSION_V2, NORMALIZATION_VERSION_V3})
_TARGET_QUANTILE = 0.95
_TARGET_SATURATION = 0.80
_ATANH_TARGET = float(np.arctanh(_TARGET_SATURATION))
_V3_NONZERO_QUANTILE = 0.99
_V3_DENSE_TAIL_QUANTILE = 0.995
_V3_SPARSE_NONZERO_FRACTION = 0.25
_V3_RATIO_SOFTENING_FRACTION = 0.10
_V3_RATIO_SOFTENING_FLOOR = 1e-6
_V3_RATIO_ZERO_TARGET = 0.95
_V3_ATANH_RATIO_ZERO_TARGET = float(np.arctanh(_V3_RATIO_ZERO_TARGET))
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
    scale_fit_modes: tuple[str, ...] = ()
    nonzero_fit_counts: tuple[int, ...] = ()
    ratio_softening: tuple[float, ...] = ()
    fit_weighting: str = "uniform-observations"
    nonzero_target_quantile: float | None = None
    dense_tail_quantile: float | None = None
    sparse_nonzero_fraction: float | None = None
    ratio_zero_target: float | None = None

    def save(self, path: str | Path) -> None:
        self.validate()
        payload = asdict(self)
        payload["scales"] = list(self.scales)
        payload["kinds"] = list(self.kinds)
        payload["scale_fit_modes"] = list(self.scale_fit_modes)
        payload["nonzero_fit_counts"] = list(self.nonzero_fit_counts)
        payload["ratio_softening"] = list(self.ratio_softening)
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> NormalizationManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["scales"] = tuple(float(value) for value in payload["scales"])
        payload["kinds"] = tuple(str(value) for value in payload["kinds"])
        payload["scale_fit_modes"] = tuple(
            str(value) for value in payload.get("scale_fit_modes", ())
        )
        payload["nonzero_fit_counts"] = tuple(
            int(value) for value in payload.get("nonzero_fit_counts", ())
        )
        payload["ratio_softening"] = tuple(
            float(value) for value in payload.get("ratio_softening", ())
        )
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
        if self.version not in SUPPORTED_NORMALIZATION_VERSIONS:
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
        if self.version == NORMALIZATION_VERSION_V2:
            if (
                self.scale_fit_modes
                or self.nonzero_fit_counts
                or self.ratio_softening
                or self.dense_tail_quantile is not None
            ):
                raise ValueError("Normalization-v2 cannot carry v3 fit parameters")
            return
        if len(self.scale_fit_modes) != TENSOR_FEATURE_COUNT:
            raise ValueError("Normalization-v3 requires one scale-fit mode per feature")
        if len(self.nonzero_fit_counts) != TENSOR_FEATURE_COUNT:
            raise ValueError("Normalization-v3 requires one nonzero count per feature")
        if len(self.ratio_softening) != TENSOR_FEATURE_COUNT:
            raise ValueError("Normalization-v3 requires one ratio softening per feature")
        if any(count < 0 for count in self.nonzero_fit_counts):
            raise ValueError("Normalization-v3 nonzero counts cannot be negative")
        if any((not np.isfinite(value)) or value < 0 for value in self.ratio_softening):
            raise ValueError("Normalization-v3 ratio softening must be finite and nonnegative")
        if not self.fit_weighting:
            raise ValueError("Normalization-v3 must declare its fit weighting")
        if self.nonzero_target_quantile != _V3_NONZERO_QUANTILE:
            raise ValueError("Normalization-v3 nonzero quantile contract mismatch")
        if self.dense_tail_quantile != _V3_DENSE_TAIL_QUANTILE:
            raise ValueError("Normalization-v3 dense-tail quantile contract mismatch")
        if self.sparse_nonzero_fraction != _V3_SPARSE_NONZERO_FRACTION:
            raise ValueError("Normalization-v3 sparse-event threshold contract mismatch")
        if self.ratio_zero_target != _V3_RATIO_ZERO_TARGET:
            raise ValueError("Normalization-v3 ratio-zero target contract mismatch")


class SemanticNormalizerV2:
    version = NORMALIZATION_VERSION_V2

    def __init__(self, manifest: NormalizationManifest) -> None:
        manifest.validate()
        if manifest.version != self.version:
            raise ValueError(f"SemanticNormalizerV2 cannot load {manifest.version}")
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
            stage2_rows.append(_stage2_v2(stage1))
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
                version=NORMALIZATION_VERSION_V2,
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
            capability_experience_thresholds=(self.manifest.capability_experience_thresholds),
            local_movement_radius=self.manifest.local_movement_radius,
        )
        stage2 = _stage2_v2(stage1)
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
            capability_experience_thresholds=(self.manifest.capability_experience_thresholds),
            local_movement_radius=self.manifest.local_movement_radius,
        )
        stage2 = _stage2_v2(stage1)
        return (
            tuple(float(value) for value in stage2),
            tuple(bool(value) for value in effective_present),
        )


SPARSE_CAUSAL_SCALE_GROUPS: Mapping[str, tuple[int, ...]] = {
    "memory-realized-migration": (99, 100, 101),
    "memory-realized-trade-utility": (111, 112, 113),
    "commodity-recent-import": tuple(316 + 8 * row for row in range(11)),
    "commodity-recent-export": tuple(317 + 8 * row for row in range(11)),
    "contact-recent-trade-utility": tuple(403 + 8 * row for row in range(8)),
    "contact-recent-trade-flow": tuple(404 + 8 * row for row in range(8)),
    "contact-recent-migration": tuple(405 + 8 * row for row in range(8)),
}

SPARSE_CAUSAL_SUPPORT_GROUPS: Mapping[str, tuple[int, ...]] = {
    "core-realized-migration": (13,),
    "core-realized-trade-utility": (15,),
    **SPARSE_CAUSAL_SCALE_GROUPS,
}


class SemanticNormalizerV3:
    """Zero-aware, family-weightable normalization for sparse causal events."""

    version = NORMALIZATION_VERSION_V3

    def __init__(self, manifest: NormalizationManifest) -> None:
        manifest.validate()
        if manifest.version != self.version:
            raise ValueError(f"SemanticNormalizerV3 cannot load {manifest.version}")
        self.manifest = manifest
        self._scales = np.asarray(manifest.scales, dtype=np.float64)
        self._ratio_softening = np.asarray(manifest.ratio_softening, dtype=np.float64)

    @classmethod
    def fit(
        cls,
        observations: Iterable[EncodedObservation],
        *,
        partition_id: str,
        sample_weights: Sequence[float] | None = None,
        fit_weighting: str = "uniform-observations",
        capability_experience_thresholds: Mapping[str, float] | None = None,
        local_movement_radius: float = 1.0,
        partition_manifest_sha256: str | None = None,
        configuration_hash: str | None = None,
        build_hash: str | None = None,
        observation_schema_hash: str | None = None,
        fit_source_mix: Mapping[str, int] | None = None,
        rng_provenance: str | None = None,
    ) -> SemanticNormalizerV3:
        batch = list(observations)
        if not batch:
            raise ValueError("Cannot fit normalization on an empty partition")
        weights = _validated_weights(sample_weights, len(batch))
        thresholds = {
            str(key): float(value)
            for key, value in (capability_experience_thresholds or {}).items()
        }
        stage1_rows: list[np.ndarray] = []
        present_rows: list[np.ndarray] = []
        for observation in batch:
            observation.validate()
            stage1, effective_present = _semantic_reference(
                np.asarray(observation.values, dtype=np.float64),
                np.asarray(observation.present, dtype=bool),
                capability_experience_thresholds=thresholds,
                local_movement_radius=local_movement_radius,
            )
            stage1_rows.append(stage1)
            present_rows.append(effective_present)
        stage1 = np.stack(stage1_rows)
        present = np.stack(present_rows)
        ratio_softening = _fit_ratio_softening(stage1, present, weights)
        stage2 = np.stack([_stage2_v3(row, ratio_softening) for row in stage1])

        scales = np.ones(TENSOR_FEATURE_COUNT, dtype=np.float64)
        modes = ["fixed"] * TENSOR_FEATURE_COUNT
        nonzero_counts = np.zeros(TENSOR_FEATURE_COUNT, dtype=np.int64)
        for index, kind in enumerate(KINDS):
            if kind in (_BINARY, _BOUNDED):
                continue
            mask = present[:, index]
            scale, mode, nonzero_count = _fit_v3_scale(
                stage1[mask, index],
                stage2[mask, index],
                weights[mask],
                kind=kind,
                ratio_softening=float(ratio_softening[index]),
            )
            scales[index] = scale
            modes[index] = mode
            nonzero_counts[index] = nonzero_count

        for group_name, indices in SPARSE_CAUSAL_SCALE_GROUPS.items():
            columns = np.asarray(indices, dtype=np.int64)
            flattened_presence = present[:, columns].reshape(-1)
            flattened_weights = np.repeat(weights, len(indices))[flattened_presence]
            flattened_stage1 = stage1[:, columns].reshape(-1)[flattened_presence]
            flattened_stage2 = stage2[:, columns].reshape(-1)[flattened_presence]
            group_scale, group_mode, group_nonzero_count = _fit_v3_scale(
                flattened_stage1,
                flattened_stage2,
                flattened_weights,
                kind=KINDS[indices[0]],
                ratio_softening=0.0,
            )
            for index in indices:
                scales[index] = group_scale
                modes[index] = f"shared:{group_name}:{group_mode}"
                nonzero_counts[index] = group_nonzero_count

        return cls(
            NormalizationManifest(
                version=NORMALIZATION_VERSION_V3,
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
                scale_fit_modes=tuple(modes),
                nonzero_fit_counts=tuple(int(value) for value in nonzero_counts),
                ratio_softening=tuple(float(value) for value in ratio_softening),
                fit_weighting=fit_weighting,
                nonzero_target_quantile=_V3_NONZERO_QUANTILE,
                dense_tail_quantile=_V3_DENSE_TAIL_QUANTILE,
                sparse_nonzero_fraction=_V3_SPARSE_NONZERO_FRACTION,
                ratio_zero_target=_V3_RATIO_ZERO_TARGET,
            )
        )

    def transform(self, observation: EncodedObservation) -> tuple[float, ...]:
        observation.validate()
        stage1, effective_present = _semantic_reference(
            np.asarray(observation.values, dtype=np.float64),
            np.asarray(observation.present, dtype=bool),
            capability_experience_thresholds=(self.manifest.capability_experience_thresholds),
            local_movement_radius=self.manifest.local_movement_radius,
        )
        stage2 = _stage2_v3(stage1, self._ratio_softening)
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
        observation.validate()
        stage1, effective_present = _semantic_reference(
            np.asarray(observation.values, dtype=np.float64),
            np.asarray(observation.present, dtype=bool),
            capability_experience_thresholds=(self.manifest.capability_experience_thresholds),
            local_movement_radius=self.manifest.local_movement_radius,
        )
        stage2 = _stage2_v3(stage1, self._ratio_softening)
        return (
            tuple(float(value) for value in stage2),
            tuple(bool(value) for value in effective_present),
        )


SemanticNormalizer = SemanticNormalizerV2 | SemanticNormalizerV3


def semantic_normalizer_from_manifest(manifest: NormalizationManifest) -> SemanticNormalizer:
    if manifest.version == NORMALIZATION_VERSION_V2:
        return SemanticNormalizerV2(manifest)
    if manifest.version == NORMALIZATION_VERSION_V3:
        return SemanticNormalizerV3(manifest)
    raise ValueError(f"Unsupported normalizer version: {manifest.version}")


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


def _stage2_v2(values: np.ndarray) -> np.ndarray:
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


def _stage2_v3(values: np.ndarray, ratio_softening: np.ndarray) -> np.ndarray:
    result = values.copy()
    for index, kind in enumerate(KINDS):
        value = float(result[index])
        if kind in (_BINARY, _BOUNDED):
            continue
        if kind in (_SIGNED, _DISTANCE):
            result[index] = np.arcsinh(value)
        elif kind in (_RATIO, _CAPABILITY_EXPERIENCE):
            epsilon = float(ratio_softening[index])
            result[index] = np.log((max(0.0, value) + epsilon) / (1.0 + epsilon))
        else:
            result[index] = np.log1p(max(0.0, value))
    return result


def _validated_weights(weights: Sequence[float] | None, count: int) -> np.ndarray:
    if weights is None:
        return np.ones(count, dtype=np.float64)
    result = np.asarray(tuple(float(value) for value in weights), dtype=np.float64)
    if result.shape != (count,):
        raise ValueError("Normalization sample weights must match the observation count")
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("Normalization sample weights must be finite and positive")
    return result


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        raise ValueError("Cannot compute a weighted quantile of an empty array")
    if values.shape != weights.shape:
        raise ValueError("Weighted quantile values and weights must have the same shape")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    target = float(quantile) * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, target, side="left")), values.size - 1)
    return float(ordered_values[index])


def _fit_ratio_softening(
    stage1: np.ndarray,
    present: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    result = np.zeros(TENSOR_FEATURE_COUNT, dtype=np.float64)
    for index, kind in enumerate(KINDS):
        if kind not in (_RATIO, _CAPABILITY_EXPERIENCE):
            continue
        mask = present[:, index] & (stage1[:, index] > 0)
        if np.any(mask):
            positive_floor = _weighted_quantile(stage1[mask, index], weights[mask], 0.05)
            result[index] = max(
                _V3_RATIO_SOFTENING_FLOOR,
                _V3_RATIO_SOFTENING_FRACTION * positive_floor,
            )
        else:
            result[index] = _V3_RATIO_SOFTENING_FLOOR
    return result


def _fit_v3_scale(
    stage1: np.ndarray,
    stage2: np.ndarray,
    weights: np.ndarray,
    *,
    kind: str,
    ratio_softening: float,
) -> tuple[float, str, int]:
    if stage2.size == 0:
        return 1.0, "unobserved", 0
    magnitudes = np.abs(stage2)
    base = _weighted_quantile(magnitudes, weights, _TARGET_QUANTILE)
    tail = float(np.quantile(magnitudes, _V3_DENSE_TAIL_QUANTILE, method="higher"))
    nonzero = np.abs(stage1) > 1e-12
    nonzero_count = int(np.count_nonzero(nonzero))
    fit_magnitude = max(base, tail)
    mode = "dense-tail-envelope"
    if kind in (_SIGNED, _POSITIVE) and nonzero_count:
        nonzero_fraction = float(np.sum(weights[nonzero]) / np.sum(weights))
        if nonzero_fraction < _V3_SPARSE_NONZERO_FRACTION:
            fit_magnitude = max(
                fit_magnitude,
                _weighted_quantile(
                    magnitudes[nonzero],
                    weights[nonzero],
                    _V3_NONZERO_QUANTILE,
                ),
            )
            mode = "sparse-nonzero-guard+dense-tail-envelope"
    scale = max(fit_magnitude / _ATANH_TARGET, 1e-6)
    if kind in (_RATIO, _CAPABILITY_EXPERIENCE):
        zero_coordinate = abs(float(np.log(ratio_softening / (1.0 + ratio_softening))))
        scale = max(scale, zero_coordinate / _V3_ATANH_RATIO_ZERO_TARGET)
        mode = "finite-zero-ratio"
    return scale, mode, nonzero_count
