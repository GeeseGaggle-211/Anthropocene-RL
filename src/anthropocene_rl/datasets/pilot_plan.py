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

PILOT_GENERATION_SEED = 5117001


def build_cultivation_pilot_plan() -> SnapshotBankGenerationConfig:
    trajectories: list[dict[str, Any]] = []
    family_specs: tuple[tuple[str, int, int, dict[str, list[Any]]], ...] = (
        (
            "F01",
            40,
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
            40,
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
            40,
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
            48,
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
            seed=PILOT_GENERATION_SEED + family_index,
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
                    world_seed=seed_start + index,
                    trajectory_family="cultivation-pairwise-v1",
                    snapshot_steps=[0],
                    requested_tags=requested_tags,
                    classification=classification,
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
            count=8,
            seed=PILOT_GENERATION_SEED + 17,
        )
    ):
        trajectories.append(
            _trajectory(
                trajectory_id=f"f17-memory-contrast-{index + 1:03d}",
                scenario_family="F17",
                world_seed=31401 + index,
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
            count=24,
            seed=PILOT_GENERATION_SEED + 18,
        )
    ):
        world_seed = 31501 + index
        group_id = f"f18-frontier-pair-{index + 1:03d}"
        common = {
            key: value
            for key, value in row.items()
            if key not in {"primitive_q", "plough_q"}
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

    return SnapshotBankGenerationConfig.model_validate(
        {
            "config_schema_version": "snapshot-bank-generation-v1",
            "bank_kind": "cultivation-pilot",
            "bank_id": "elder-v2-cultivation-pilot",
            "bank_version": "1.0.0-plan",
            "native_generator_version": "snapshot-bank-native-v1",
            "generation_seed": PILOT_GENERATION_SEED,
            "world": {
                "width": 48,
                "height": 24,
                "players": 4,
                "start_separation": 6,
                "annual_human_growth_rate": 0.0,
            },
            "partition": {
                "salt": "elder-v2-cultivation-pilot-partitions-v1",
                "train_fraction": 0.70,
                "development_fraction": 0.15,
                "heldout_fraction": 0.15,
            },
            "calibration_rollout_steps": 3,
            "verify_deterministic_regeneration": True,
            "local_movement_radius": 1.0,
            "capability_experience_thresholds": {
                "field-retention": 225.0,
                "herd-management": 225.0,
                "resident-fishing": 225.0,
            },
            "trajectories": trajectories,
        }
    )


def cultivation_pilot_plan_summary(
    config: SnapshotBankGenerationConfig,
) -> dict[str, Any]:
    resolved, _ = config.resolve_partitions()
    family_snapshots: Counter[str] = Counter()
    family_worlds: dict[str, set[int]] = {}
    partition_worlds: dict[str, set[int]] = {}
    for trajectory in resolved:
        family_snapshots[trajectory.scenario_family] += len(trajectory.snapshot_steps)
        family_worlds.setdefault(trajectory.scenario_family, set()).add(
            trajectory.world_seed
        )
        partition_worlds.setdefault(trajectory.partition.value, set()).add(
            trajectory.world_seed
        )
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
            ),
        )
        for family in sorted(family_snapshots)
    }
    return {
        "status": "PLAN_READY_NOT_GENERATED",
        "bank_id": config.bank_id,
        "generator_config_sha256": config.canonical_sha256(),
        "requested_snapshot_count": sum(family_snapshots.values()),
        "independent_world_count": len({item.world_seed for item in resolved}),
        "family_snapshot_counts": dict(sorted(family_snapshots.items())),
        "family_world_counts": {
            family: len(worlds) for family, worlds in sorted(family_worlds.items())
        },
        "partition_world_counts": {
            partition: len(worlds)
            for partition, worlds in sorted(partition_worlds.items())
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
            "fit final normalization-v2 from frozen PILOT TRAIN reset plus short legal "
            "calibration rollouts; DEVELOPMENT and HELDOUT are transform-only"
        ),
        "deferred_capabilities": ["movement", "herding", "fishing"],
    }


def save_cultivation_pilot_plan(
    *,
    config_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    config = build_cultivation_pilot_plan()
    summary = cultivation_pilot_plan_summary(config)
    config_payload = config.model_dump(mode="json")
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
) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory_id,
        "scenario_family": scenario_family,
        "scenario_version": "cultivation-pilot-v1",
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
                        (row_index + candidate_index * (factor_index + 1))
                        % len(factors[name])
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
