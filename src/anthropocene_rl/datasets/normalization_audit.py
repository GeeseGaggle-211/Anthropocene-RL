"""Provisional TRAIN-only normalization fitting and per-feature diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anthropocene_rl.datasets.models import Partition, SnapshotBankManifest
from anthropocene_rl.observations.features import FEATURE_NAMES
from anthropocene_rl.observations.normalization import (
    KINDS,
    NORMALIZATION_VERSION_V2,
    NORMALIZATION_VERSION_V3,
    SPARSE_CAUSAL_SUPPORT_GROUPS,
    SemanticNormalizer,
    SemanticNormalizerV2,
    SemanticNormalizerV3,
)
from anthropocene_rl.observations.schema import EncodedObservation

_SATURATION_THRESHOLD = 0.95


@dataclass(frozen=True)
class CalibrationObservation:
    snapshot_id: str
    partition: Partition
    source: str
    observation: EncodedObservation
    slices: frozenset[str]


def fit_provisional_and_report(
    samples: list[CalibrationObservation],
    *,
    manifest: SnapshotBankManifest,
    capability_experience_thresholds: dict[str, float],
    local_movement_radius: float,
    normalizer_path: str | Path,
    diagnostics_path: str | Path,
    rng_provenance: str,
) -> tuple[SemanticNormalizerV2, dict[str, Any]]:
    return fit_train_and_report(
        samples,
        manifest=manifest,
        capability_experience_thresholds=capability_experience_thresholds,
        local_movement_radius=local_movement_radius,
        normalizer_path=normalizer_path,
        diagnostics_path=diagnostics_path,
        rng_provenance=rng_provenance,
        partition_suffix="provisional",
        artifact_status="PROVISIONAL_STAGE0_PIPELINE_TEST_NOT_FINAL_CAMPAIGN_NORMALIZER",
    )


def fit_train_and_report(
    samples: list[CalibrationObservation],
    *,
    manifest: SnapshotBankManifest,
    capability_experience_thresholds: dict[str, float],
    local_movement_radius: float,
    normalizer_path: str | Path,
    diagnostics_path: str | Path,
    rng_provenance: str,
    partition_suffix: str,
    artifact_status: str,
    normalization_version: str = NORMALIZATION_VERSION_V2,
) -> tuple[SemanticNormalizer, dict[str, Any]]:
    training = [sample for sample in samples if sample.partition is Partition.TRAIN]
    if not training:
        raise ValueError("Normalization requires TRAIN observations")
    configuration_hashes = {record.configuration_hash for record in manifest.records}
    build_hashes = {record.native_build_hash for record in manifest.records}
    schema_hashes = {record.schema_identity.observation_schema_hash for record in manifest.records}
    if len(configuration_hashes) != 1 or len(build_hashes) != 1 or len(schema_hashes) != 1:
        raise ValueError("Normalization calibration requires one native build/config/schema cohort")
    source_mix = Counter(sample.source for sample in training)
    common_fit = {
        "partition_id": f"{manifest.bank_id}:train:{partition_suffix}",
        "capability_experience_thresholds": capability_experience_thresholds,
        "local_movement_radius": local_movement_radius,
        "partition_manifest_sha256": manifest.sha256(),
        "configuration_hash": next(iter(configuration_hashes)),
        "build_hash": next(iter(build_hashes)),
        "observation_schema_hash": next(iter(schema_hashes)),
        "fit_source_mix": source_mix,
        "rng_provenance": rng_provenance,
    }
    if normalization_version == NORMALIZATION_VERSION_V2:
        normalizer: SemanticNormalizer = SemanticNormalizerV2.fit(
            [sample.observation for sample in training],
            **common_fit,
        )
    elif normalization_version == NORMALIZATION_VERSION_V3:
        family_by_snapshot = {
            record.snapshot_id: record.scenario_family for record in manifest.records
        }
        family_counts = Counter(family_by_snapshot[sample.snapshot_id] for sample in training)
        sample_weights = [
            1.0 / family_counts[family_by_snapshot[sample.snapshot_id]] for sample in training
        ]
        normalizer = SemanticNormalizerV3.fit(
            [sample.observation for sample in training],
            sample_weights=sample_weights,
            fit_weighting="equal-total-weight-per-scenario-family",
            **common_fit,
        )
    else:
        raise ValueError(f"Unsupported normalization version: {normalization_version}")
    normalizer.manifest.save(normalizer_path)
    report = normalization_diagnostics(
        samples,
        normalizer=normalizer,
        manifest=manifest,
        artifact_status=artifact_status,
    )
    target = Path(diagnostics_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if normalization_version == NORMALIZATION_VERSION_V3:
        failures = []
        if not report["sparse_causal_support_pass"]:
            failures.append("sparse causal TRAIN support")
        if not report["saturation_gate_pass"]:
            failures.append("per-feature saturation")
        if failures:
            raise ValueError("Normalization-v3 promotion gate failed: " + ", ".join(failures))
    return normalizer, report


def normalization_diagnostics(
    samples: list[CalibrationObservation],
    *,
    normalizer: SemanticNormalizer,
    manifest: SnapshotBankManifest,
    artifact_status: str = "PROVISIONAL_STAGE0_PIPELINE_TEST_NOT_FINAL_CAMPAIGN_NORMALIZER",
) -> dict[str, Any]:
    coordinates: list[np.ndarray] = []
    effective_presence: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    for sample in samples:
        stage2, present = normalizer.semantic_coordinates(sample.observation)
        coordinates.append(np.asarray(stage2, dtype=np.float64))
        effective_presence.append(np.asarray(present, dtype=bool))
        outputs.append(np.asarray(normalizer.transform(sample.observation), dtype=np.float64))
    coordinate_matrix = np.stack(coordinates)
    presence_matrix = np.stack(effective_presence)
    output_matrix = np.stack(outputs)

    slice_indices: dict[str, list[int]] = {
        "train-fit": [
            index for index, sample in enumerate(samples) if sample.partition is Partition.TRAIN
        ],
        "development": [
            index
            for index, sample in enumerate(samples)
            if sample.partition is Partition.DEVELOPMENT
        ],
        "heldout": [
            index for index, sample in enumerate(samples) if sample.partition is Partition.HELDOUT
        ],
    }
    diagnostic_slices = (
        "low-q",
        "high-q",
        "fishing",
        "fishing-only",
        "no-herd",
        "young-memory",
        "mature-memory",
        "trade-dependent",
        "isolated",
        "immigration",
        "emigration",
        "pre-fission",
        "post-fission",
        "primitive-frontier",
        "plough-frontier",
    )
    for name in diagnostic_slices:
        slice_indices[name] = [
            index for index, sample in enumerate(samples) if name in sample.slices
        ]
    for partition in Partition:
        for name in diagnostic_slices:
            slice_indices[f"{partition.value}:{name}"] = [
                index
                for index, sample in enumerate(samples)
                if sample.partition is partition and name in sample.slices
            ]

    features: list[dict[str, Any]] = []
    modes = normalizer.manifest.scale_fit_modes or ("v2-all-present",) * len(KINDS)
    nonzero_counts = normalizer.manifest.nonzero_fit_counts or (0,) * len(KINDS)
    ratio_softening = normalizer.manifest.ratio_softening or (0.0,) * len(KINDS)
    for feature_index, (name, kind, scale, mode, nonzero_count, softening) in enumerate(
        zip(
            FEATURE_NAMES,
            KINDS,
            normalizer.manifest.scales,
            modes,
            nonzero_counts,
            ratio_softening,
            strict=True,
        )
    ):
        feature: dict[str, Any] = {
            "index": feature_index,
            "name": name,
            "kind": kind,
            "fitted_c_i": scale,
            "scale_fit_mode": mode,
            "train_nonzero_fit_count": nonzero_count,
            "ratio_softening": softening,
            "slices": {},
        }
        for slice_name, indices in slice_indices.items():
            feature["slices"][slice_name] = _feature_stats(
                coordinate_matrix,
                presence_matrix,
                output_matrix,
                feature_index,
                indices,
                saturation_applicable=kind not in {"binary", "bounded01"},
            )
        features.append(feature)
    sparse_support = _sparse_causal_support(
        coordinate_matrix,
        presence_matrix,
        slice_indices["train-fit"],
    )
    saturation_max_by_slice: dict[str, float | None] = {}
    for slice_name, indices in slice_indices.items():
        rates = [
            feature["slices"][slice_name]["saturation_rate"]
            for feature in features
            if feature["slices"][slice_name]["saturation_rate"] is not None
        ]
        saturation_max_by_slice[slice_name] = (
            max(float(rate) for rate in rates) if indices and rates else None
        )
    saturation_gate_pass = all(
        maximum is None or maximum <= 0.10 for maximum in saturation_max_by_slice.values()
    )
    return {
        "status": artifact_status,
        "normalization_version": normalizer.manifest.version,
        "bank_id": manifest.bank_id,
        "partition_manifest_sha256": manifest.sha256(),
        "fit_partition": "train",
        "fit_observation_count": normalizer.manifest.fit_observation_count,
        "fit_source_mix": normalizer.manifest.fit_source_mix,
        "fit_weighting": normalizer.manifest.fit_weighting,
        "heldout_or_development_used_for_fit": False,
        "coordinate_space": ("post-semantic-reference-and-log/asinh, pre-frozen-scale-tanh"),
        "stratum_slice_membership": "authoritative reset measurements only",
        "saturation_absolute_threshold": _SATURATION_THRESHOLD,
        "saturation_definition": (
            "abs(post-tanh value) >= threshold; not applicable to binary/bounded features"
        ),
        "saturation_gate_maximum": 0.10,
        "saturation_max_by_slice": saturation_max_by_slice,
        "saturation_gate_pass": saturation_gate_pass,
        "sample_counts": {name: len(indices) for name, indices in sorted(slice_indices.items())},
        "sparse_causal_support": sparse_support,
        "sparse_causal_support_pass": all(row["sufficient"] for row in sparse_support.values()),
        "features": features,
    }


def _sparse_causal_support(
    coordinates: np.ndarray,
    presence: np.ndarray,
    train_indices: list[int],
) -> dict[str, dict[str, int | bool]]:
    result: dict[str, dict[str, int | bool]] = {}
    for name, feature_indices in SPARSE_CAUSAL_SUPPORT_GROUPS.items():
        columns = np.asarray(feature_indices, dtype=np.int64)
        selected = coordinates[np.ix_(train_indices, columns)].reshape(-1)
        selected_presence = presence[np.ix_(train_indices, columns)].reshape(-1)
        values = selected[selected_presence]
        nonzero = values[np.abs(values) > 1e-12]
        positive = int(np.count_nonzero(nonzero > 0))
        negative = int(np.count_nonzero(nonzero < 0))
        signed_migration = "migration" in name
        sufficient = len(nonzero) >= 32 and (
            not signed_migration or (positive >= 16 and negative >= 16)
        )
        result[name] = {
            "nonzero_count": len(nonzero),
            "positive_count": positive,
            "negative_count": negative,
            "minimum_nonzero_count": 32,
            "minimum_each_sign_when_migration": 16 if signed_migration else 0,
            "sufficient": bool(sufficient),
        }
    return result


def _feature_stats(
    coordinates: np.ndarray,
    presence: np.ndarray,
    outputs: np.ndarray,
    feature_index: int,
    indices: list[int],
    *,
    saturation_applicable: bool,
) -> dict[str, float | int | None]:
    if not indices:
        return {
            "sample_count": 0,
            "presence_rate": None,
            "min": None,
            "p5": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "saturation_rate": None,
        }
    selected_presence = presence[indices, feature_index]
    values = coordinates[indices, feature_index][selected_presence]
    normalized = outputs[indices, feature_index][selected_presence]
    result: dict[str, float | int | None] = {
        "sample_count": len(indices),
        "presence_rate": float(np.mean(selected_presence)),
    }
    if values.size == 0:
        result.update(
            {
                "min": None,
                "p5": None,
                "p50": None,
                "p95": None,
                "p99": None,
                "max": None,
                "saturation_rate": None,
            }
        )
        return result
    quantiles = np.quantile(values, [0.05, 0.5, 0.95, 0.99])
    result.update(
        {
            "min": float(np.min(values)),
            "p5": float(quantiles[0]),
            "p50": float(quantiles[1]),
            "p95": float(quantiles[2]),
            "p99": float(quantiles[3]),
            "max": float(np.max(values)),
            "saturation_rate": (
                float(np.mean(np.abs(normalized) >= _SATURATION_THRESHOLD))
                if saturation_applicable
                else None
            ),
        }
    )
    return result
