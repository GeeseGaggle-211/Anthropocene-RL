"""Deterministic constrained-pairwise plan for the cultivation-only pilot bank."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from anthropocene_rl.datasets.generation_config import SnapshotBankGenerationConfig
from anthropocene_rl.observations.normalization import (
    NORMALIZATION_VERSION_V2,
    NORMALIZATION_VERSION_V3,
)

PILOT_GENERATION_SEED = 5117001
PILOT_V3_REMEDIATION_GENERATION_SEED = 6117001
PILOT_ACCEPTED_FAMILY_COUNTS = {
    "F01": 30,
    "F02": 35,
    "F03": 35,
    "F13": 40,
    "F15": 50,
    "F16": 40,
    "F17": 24,
    "F18": 46,
}
PILOT_ACCEPTED_PARTITION_COUNTS = {
    "F01": {"train": 20, "development": 4, "heldout": 6},
    "F02": {"train": 29, "development": 2, "heldout": 4},
    "F03": {"train": 24, "development": 7, "heldout": 4},
    "F13": {"train": 25, "development": 3, "heldout": 12},
    "F15": {"train": 30, "development": 12, "heldout": 8},
    "F16": {"train": 32, "development": 4, "heldout": 4},
    "F17": {"train": 18, "development": 3, "heldout": 3},
    "F18": {"train": 32, "development": 10, "heldout": 4},
}
PILOT_V3_ACCEPTED_PARTITION_COUNTS = {
    "F01": {"train": 21, "development": 4, "heldout": 5},
    "F02": {"train": 25, "development": 4, "heldout": 6},
    "F03": {"train": 24, "development": 6, "heldout": 5},
    "F13": {"train": 28, "development": 7, "heldout": 5},
    "F15": {"train": 38, "development": 6, "heldout": 6},
    "F16": {"train": 28, "development": 4, "heldout": 8},
    "F17": {"train": 18, "development": 3, "heldout": 3},
    "F18": {"train": 28, "development": 12, "heldout": 6},
}


def build_cultivation_pilot_plan() -> SnapshotBankGenerationConfig:
    return _build_cultivation_pilot_plan(
        generation_seed=PILOT_GENERATION_SEED,
        world_seed_offset=0,
        bank_id="elder-v2-cultivation-pilot",
        bank_version="2.0.0-pilot",
        partition_salt="elder-v2-cultivation-pilot-partitions-v7",
        normalization_version=NORMALIZATION_VERSION_V2,
    )


def build_normalization_v3_remediation_plan() -> SnapshotBankGenerationConfig:
    return _build_cultivation_pilot_plan(
        generation_seed=PILOT_V3_REMEDIATION_GENERATION_SEED,
        world_seed_offset=10_000,
        bank_id="elder-v2-cultivation-pilot-normalization-v3",
        bank_version="3.0.0-remediation",
        partition_salt=("elder-v2-cultivation-pilot-partitions-v8-normalization-v3-12333"),
        normalization_version=NORMALIZATION_VERSION_V3,
        accepted_partition_counts=PILOT_V3_ACCEPTED_PARTITION_COUNTS,
    )


def _build_cultivation_pilot_plan(
    *,
    generation_seed: int,
    world_seed_offset: int,
    bank_id: str,
    bank_version: str,
    partition_salt: str,
    normalization_version: str,
    accepted_partition_counts: dict[str, dict[str, int]] | None = None,
) -> SnapshotBankGenerationConfig:
    trajectories: list[dict[str, Any]] = []
    family_specs: tuple[tuple[str, int, int, dict[str, list[Any]]], ...] = (
        (
            "F01",
            30,
            31001,
            {
                "population": [60, 80, 100, 130, 160],
                "field_q": [0.12, 0.16, 0.20, 0.24],
                "reserve_intervals": [0.25, 1.0, 2.0, 4.0],
                "site_support_quantile": [0.2, 0.5, 0.8, 1.0],
                "tool_abundance": [0.2, 0.5, 1.0, 1.5],
                "amenity_coverage": [0.5, 0.75, 1.0],
            },
        ),
        (
            "F02",
            35,
            31101,
            {
                "population": [100, 150, 210, 280, 360],
                "field_q": [0.35, 0.38, 0.41, 0.44],
                "reserve_intervals": [0.25, 1.0, 2.0, 4.0],
                "site_support_quantile": [0.2, 0.5, 0.8, 1.0],
                "tool_abundance": [0.15, 0.3, 0.6, 1.0],
                "amenity_coverage": [0.5, 0.75, 1.0],
            },
        ),
        (
            "F03",
            35,
            31201,
            {
                "population": [100, 160, 240, 360, 480],
                "field_q": [0.52, 0.56, 0.60, 0.64],
                "reserve_intervals": [0.25, 1.0, 2.0, 4.0],
                "site_support_quantile": [0.2, 0.5, 0.8, 1.0],
                "tool_abundance": [0.1, 0.2, 0.35, 0.5],
                "amenity_coverage": [0.5, 0.75, 1.0],
            },
        ),
        (
            "F13",
            40,
            31301,
            {
                "population": [120, 180, 240, 320, 420],
                "field_q": [0.28, 0.34, 0.40, 0.44],
                "reserve_intervals": [1.0, 2.0, 4.0],
                "site_support_quantile": [0.2, 0.5, 0.8, 1.0],
                "tool_abundance": [0.1, 0.2, 0.35],
                "amenity_coverage": [0.1, 0.2, 0.35, 0.45],
            },
        ),
    )
    for family_index, (family, count, seed_start, factors) in enumerate(family_specs):
        rows = _constrained_pairwise_rows(
            factors,
            count=count,
            seed=generation_seed + family_index,
        )
        for index, row in enumerate(rows):
            label = family.lower()
            requested_tags = ["cultivation-only"]
            classification = "controlled-valid-fixture"
            if family == "F01":
                requested_tags.append("young-memory")
            elif family == "F02":
                requested_tags.append("near-knee-q")
            elif family == "F03":
                requested_tags.append("high-q")
                classification = "synthetic-stress"
            elif family == "F13":
                requested_tags.append("amenity-bottleneck")
            trajectories.append(
                _trajectory(
                    trajectory_id=f"{label}-pairwise-{index + 1:03d}",
                    scenario_family=family,
                    world_seed=seed_start + world_seed_offset + index,
                    trajectory_family="cultivation-pairwise-v1",
                    snapshot_steps=[0],
                    requested_tags=requested_tags,
                    classification=classification,
                    **row,
                )
            )

    open_economy_factors: dict[str, list[Any]] = {
        "population": [90, 130, 180, 250, 340],
        "field_q": [0.24, 0.32, 0.40, 0.48],
        "reserve_intervals": [0.25, 1.0, 2.0, 4.0],
        "site_support_quantile": [0.2, 0.5, 0.8, 1.0],
        "tool_abundance": [0.15, 0.4, 0.8, 1.4],
        "amenity_coverage": [0.3, 0.55, 0.8, 1.0],
    }
    if normalization_version == NORMALIZATION_VERSION_V3:
        # The first independent v3 probe showed low physical-stratum yield for
        # the isolation/import and mild-import/export contrasts.  Oversample
        # candidates before native validation; the accepted quota remains five
        # complete pairs per contrast and 50 F15 snapshots total.
        open_economy_pair_targets = (
            (("isolated", "strong-food-import"), 30),
            (("mild-import", "surplus-export"), 96),
            (("food-shortfall-contact", "material-tool-import"), 50),
            (("immigration-secure", "emigration-distress"), 12),
            (("immigration-labor", "immigration-stress"), 10),
        )
    else:
        open_economy_pair_targets = (
            (("isolated", "strong-food-import"), 15),
            (("mild-import", "surplus-export"), 22),
            (("food-shortfall-contact", "material-tool-import"), 50),
            (("immigration-secure", "emigration-distress"), 12),
            (("immigration-labor", "immigration-stress"), 10),
        )
    open_economy_pairs = [
        pair
        for repeat in range(max(count for _, count in open_economy_pair_targets))
        for pair, count in open_economy_pair_targets
        if repeat < count
    ]
    open_economy_rows = _constrained_pairwise_rows(
        open_economy_factors,
        count=len(open_economy_pairs),
        seed=generation_seed + 15,
    )
    for index, (row, variants) in enumerate(
        zip(open_economy_rows, open_economy_pairs, strict=True)
    ):
        world_seed = 31401 + world_seed_offset + index
        group_id = f"f15-open-economy-pair-{index + 1:03d}"
        for variant in variants:
            required = ["cultivation-only"]
            if variant == "isolated":
                required.append("isolated")
            elif variant in {
                "strong-food-import",
                "mild-import",
                "food-shortfall-contact",
                "material-tool-import",
                "surplus-export",
            }:
                required.append("trade-active")
                if variant in {
                    "strong-food-import",
                    "mild-import",
                    "food-shortfall-contact",
                }:
                    required.append("food-import")
                    if variant == "strong-food-import":
                        required.append("trade-dependent")
                elif variant == "material-tool-import":
                    required.append("material-tool-import")
                elif variant == "surplus-export":
                    required.append("food-export")
            elif variant.startswith("immigration-"):
                required.append("migration-positive")
            elif variant == "emigration-distress":
                required.append("migration-negative")
            trajectories.append(
                _trajectory(
                    trajectory_id=f"{group_id}-{variant}",
                    scenario_family="F15",
                    world_seed=world_seed,
                    trajectory_family="cultivation-open-economy-v1",
                    lineage_id=group_id,
                    counterfactual_group_id=group_id,
                    snapshot_steps=([2] if variant == "material-tool-import" else [1]),
                    requested_tags=required,
                    causal_variant=variant,
                    **row,
                )
            )

    fission_factors: dict[str, list[Any]] = {
        "population": [360, 480, 650, 850, 1100],
        "field_q": [0.28, 0.36, 0.44, 0.52],
        "reserve_intervals": [0.25, 1.0, 2.0],
        "site_support_quantile": [0.2, 0.5, 0.8, 1.0],
        "tool_abundance": [0.2, 0.6, 1.2],
        "amenity_coverage": [0.2, 0.45, 0.75, 1.0],
    }
    fission_candidate_count = 20 if normalization_version == NORMALIZATION_VERSION_V3 else 15
    fission_rows = _constrained_pairwise_rows(
        fission_factors,
        count=fission_candidate_count,
        seed=generation_seed + 16,
    )
    for index, row in enumerate(fission_rows):
        world_seed = 31601 + world_seed_offset + index
        group_id = f"f16-exit-opportunity-pair-{index + 1:03d}"
        for variant in ("poor-exit-opportunity", "strong-exit-opportunity"):
            trajectories.append(
                _trajectory(
                    trajectory_id=f"{group_id}-{variant}",
                    scenario_family="F16",
                    world_seed=world_seed,
                    trajectory_family="cultivation-fission-causal-v1",
                    lineage_id=group_id,
                    counterfactual_group_id=group_id,
                    snapshot_steps=[0, 1],
                    requested_tags=["cultivation-only"],
                    causal_variant=variant,
                    **row,
                )
            )

    memory_factors: dict[str, list[Any]] = {
        "population": [100, 160, 240, 340],
        "field_q": [0.28, 0.34, 0.40, 0.44],
        "reserve_intervals": [2.0, 4.0],
        "site_support_quantile": [0.5, 0.8, 1.0],
        "tool_abundance": [0.75, 1.0, 1.5],
        "amenity_coverage": [0.5, 0.75, 1.0],
    }
    for index, row in enumerate(
        _constrained_pairwise_rows(
            memory_factors,
            count=12,
            seed=generation_seed + 17,
        )
    ):
        trajectories.append(
            _trajectory(
                trajectory_id=f"f17-memory-contrast-{index + 1:03d}",
                scenario_family="F17",
                world_seed=31701 + world_seed_offset + index,
                trajectory_family="cultivation-memory-history-v1",
                snapshot_steps=[0, 3, 9],
                requested_tags=["cultivation-only"],
                **row,
            )
        )

    frontier_factors: dict[str, list[Any]] = {
        "population": [120, 180, 260, 360],
        "reserve_intervals": [1.0, 2.0, 4.0],
        "site_support_quantile": [0.2, 0.5, 0.8, 1.0],
        "tool_abundance": [0.2, 0.5, 1.0, 1.5],
        "amenity_coverage": [0.5, 0.75, 1.0],
        "primitive_q": [0.35, 0.38, 0.41, 0.44],
        "plough_q": [0.52, 0.56, 0.60, 0.64],
    }
    for index, row in enumerate(
        _constrained_pairwise_rows(
            frontier_factors,
            count=23,
            seed=generation_seed + 18,
        )
    ):
        world_seed = 31801 + world_seed_offset + index
        group_id = f"f18-frontier-pair-{index + 1:03d}"
        common = {
            key: value for key, value in row.items() if key not in {"primitive_q", "plough_q"}
        }
        trajectories.extend(
            (
                _trajectory(
                    trajectory_id=f"{group_id}-primitive",
                    scenario_family="F18",
                    world_seed=world_seed,
                    trajectory_family="cultivation-frontier-counterfactual-v1",
                    lineage_id=group_id,
                    counterfactual_group_id=group_id,
                    snapshot_steps=[0],
                    frontier="primitive",
                    field_q=row["primitive_q"],
                    requested_tags=["cultivation-only", "near-knee-q"],
                    **common,
                ),
                _trajectory(
                    trajectory_id=f"{group_id}-plough",
                    scenario_family="F18",
                    world_seed=world_seed,
                    trajectory_family="cultivation-frontier-counterfactual-v1",
                    lineage_id=group_id,
                    counterfactual_group_id=group_id,
                    snapshot_steps=[0],
                    frontier="plough",
                    field_q=row["plough_q"],
                    requested_tags=["cultivation-only", "high-q"],
                    **common,
                ),
            )
        )

    payload: dict[str, Any] = {
        "config_schema_version": "snapshot-bank-generation-v1",
        "bank_kind": "cultivation-pilot",
        "bank_id": bank_id,
        "bank_version": bank_version,
        "native_generator_version": "snapshot-bank-native-v1",
        "generation_seed": generation_seed,
        "world": {
            "width": 48,
            "height": 24,
            "players": 4,
            "start_separation": 6,
            "annual_human_growth_rate": 0.01145,
        },
        "partition": {
            "salt": partition_salt,
            "train_fraction": 0.70,
            "development_fraction": 0.15,
            "heldout_fraction": 0.15,
        },
        "calibration_rollout_steps": (
            5 if normalization_version == NORMALIZATION_VERSION_V3 else 3
        ),
        "verify_deterministic_regeneration": True,
        "local_movement_radius": 1.0,
        "capability_experience_thresholds": {
            "field-retention": 225.0,
            "herd-management": 225.0,
            "resident-fishing": 225.0,
        },
        "trajectories": trajectories,
    }
    if normalization_version != NORMALIZATION_VERSION_V2:
        payload["normalization_version"] = normalization_version
    if accepted_partition_counts is not None:
        payload["pilot_accepted_partition_counts"] = accepted_partition_counts
    return SnapshotBankGenerationConfig.model_validate(payload)


def cultivation_pilot_plan_summary(
    config: SnapshotBankGenerationConfig,
) -> dict[str, Any]:
    resolved, _ = config.resolve_partitions()
    family_snapshots: Counter[str] = Counter()
    family_worlds: dict[str, set[int]] = {}
    partition_worlds: dict[str, set[int]] = {}
    for trajectory in resolved:
        family_snapshots[trajectory.scenario_family] += len(trajectory.snapshot_steps)
        family_worlds.setdefault(trajectory.scenario_family, set()).add(trajectory.world_seed)
        partition_worlds.setdefault(trajectory.partition.value, set()).add(trajectory.world_seed)
    pairwise = {
        family: _requested_pair_coverage(
            [
                trajectory.model_dump(mode="json")
                for trajectory in config.trajectories
                if trajectory.scenario_family == family
            ],
            fields=(
                "population",
                "field_q",
                "reserve_intervals",
                "site_support_quantile",
                "tool_abundance",
                "amenity_coverage",
                "frontier",
                "causal_variant",
            ),
        )
        for family in sorted(family_snapshots)
    }
    return {
        "status": "PLAN_READY_NOT_GENERATED",
        "bank_id": config.bank_id,
        "generator_config_sha256": config.canonical_sha256(),
        "requested_snapshot_count": sum(family_snapshots.values()),
        "target_accepted_snapshot_count": sum(PILOT_ACCEPTED_FAMILY_COUNTS.values()),
        "target_accepted_family_counts": PILOT_ACCEPTED_FAMILY_COUNTS,
        "target_accepted_partition_counts": (
            config.pilot_accepted_partition_counts or PILOT_ACCEPTED_PARTITION_COUNTS
        ),
        "independent_world_count": len({item.world_seed for item in resolved}),
        "family_snapshot_counts": dict(sorted(family_snapshots.items())),
        "family_world_counts": {
            family: len(worlds) for family, worlds in sorted(family_worlds.items())
        },
        "partition_world_counts": {
            partition: len(worlds) for partition, worlds in sorted(partition_worlds.items())
        },
        "requested_pairwise_coverage": pairwise,
        "selection_rule": (
            "deterministic constrained pairwise rows; partitions resolved from complete "
            "world/lineage identity before timestamp selection"
        ),
        "promotion_rule": (
            "requested bands are planning inputs only; freeze the bank only after native "
            "accepted records pass measured physical-strata and leakage audits"
        ),
        "normalization_rule": (
            f"fit final {config.normalization_version} from frozen PILOT TRAIN reset "
            "plus short legal calibration rollouts; DEVELOPMENT and HELDOUT are "
            "transform-only"
        ),
        "live_external_dynamics": ["trade", "immigration", "emigration", "fission"],
        "deferred_capabilities": ["movement", "herding", "fishing"],
    }


def save_cultivation_pilot_plan(
    *,
    config_path: str | Path,
    summary_path: str | Path,
    variant: str = "v2",
) -> dict[str, Any]:
    builders = {
        "v2": build_cultivation_pilot_plan,
        "normalization-v3": build_normalization_v3_remediation_plan,
    }
    try:
        config = builders[variant]()
    except KeyError as error:
        raise ValueError(f"Unsupported cultivation pilot plan variant: {variant}") from error
    summary = cultivation_pilot_plan_summary(config)
    config_payload = config.model_dump(mode="json")
    if "normalization_version" not in config.model_fields_set:
        config_payload.pop("normalization_version", None)
    if "pilot_accepted_partition_counts" not in config.model_fields_set:
        config_payload.pop("pilot_accepted_partition_counts", None)
    _write_atomic(
        Path(config_path),
        yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True),
    )
    _write_atomic(
        Path(summary_path),
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def _trajectory(
    *,
    trajectory_id: str,
    scenario_family: str,
    world_seed: int,
    trajectory_family: str,
    snapshot_steps: list[int],
    requested_tags: list[str],
    population: int,
    field_q: float,
    reserve_intervals: float,
    site_support_quantile: float,
    tool_abundance: float,
    amenity_coverage: float,
    lineage_id: str | None = None,
    counterfactual_group_id: str | None = None,
    classification: str = "controlled-valid-fixture",
    frontier: str = "primitive",
    causal_variant: str = "none",
) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory_id,
        "scenario_family": scenario_family,
        "scenario_version": "cultivation-pilot-v2",
        "world_seed": world_seed,
        "trajectory_family": trajectory_family,
        "lineage_id": lineage_id or trajectory_id,
        "counterfactual_group_id": counterfactual_group_id,
        "classification": classification,
        "snapshot_steps": snapshot_steps,
        "economy": "cultivation",
        "population": population,
        "field_q": field_q,
        "herd_support_ratio": 0.0,
        "reserve_intervals": reserve_intervals,
        "fishing_ground_target": 0,
        "site_support_quantile": site_support_quantile,
        "tool_abundance": tool_abundance,
        "amenity_coverage": amenity_coverage,
        "frontier": frontier,
        "causal_variant": causal_variant,
        "requested_tags": requested_tags,
    }


def _constrained_pairwise_rows(
    factors: dict[str, list[Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    names = tuple(factors)
    universe = {
        (left, _stable_value(left_value), right, _stable_value(right_value))
        for left_index, left in enumerate(names)
        for right in names[left_index + 1 :]
        for left_value, right_value in itertools.product(factors[left], factors[right])
    }
    uncovered = set(universe)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for row_index in range(count):
        candidates = []
        for candidate_index in range(512):
            if candidate_index < len(names):
                candidate = {
                    name: factors[name][
                        (row_index + candidate_index * (factor_index + 1)) % len(factors[name])
                    ]
                    for factor_index, name in enumerate(names)
                }
            else:
                candidate = {name: rng.choice(factors[name]) for name in names}
            fingerprint = _row_fingerprint(candidate)
            if fingerprint in fingerprints:
                continue
            score = len(_row_pairs(candidate) & uncovered)
            candidates.append((score, fingerprint, candidate))
        if not candidates:
            raise RuntimeError("Pairwise planner exhausted unique candidate rows")
        _, fingerprint, chosen = max(candidates, key=lambda item: (item[0], item[1]))
        selected.append(chosen)
        fingerprints.add(fingerprint)
        uncovered.difference_update(_row_pairs(chosen))
    return selected


def _requested_pair_coverage(
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    active = tuple(field for field in fields if len({row[field] for row in rows}) > 1)
    observed = set().union(*(_row_pairs(row, fields=active) for row in rows))
    possible = {
        (left, _stable_value(left_value), right, _stable_value(right_value))
        for left_index, left in enumerate(active)
        for right in active[left_index + 1 :]
        for left_value, right_value in itertools.product(
            sorted({row[left] for row in rows}, key=_stable_value),
            sorted({row[right] for row in rows}, key=_stable_value),
        )
    }
    return {
        "active_factor_count": len(active),
        "observed_pairs": len(observed),
        "possible_pairs": len(possible),
        "fraction": len(observed) / len(possible) if possible else 1.0,
    }


def _row_pairs(
    row: dict[str, Any],
    *,
    fields: tuple[str, ...] | None = None,
) -> set[tuple[str, str, str, str]]:
    names = fields or tuple(row)
    return {
        (left, _stable_value(row[left]), right, _stable_value(row[right]))
        for left_index, left in enumerate(names)
        for right in names[left_index + 1 :]
    }


def _row_fingerprint(row: dict[str, Any]) -> str:
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
