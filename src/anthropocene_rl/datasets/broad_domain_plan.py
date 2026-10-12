"""Deterministic constrained-pairwise plan for the 50-decision Elder cohort."""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from anthropocene_rl.datasets.generation_config import SnapshotBankGenerationConfig
from anthropocene_rl.datasets.pilot_plan import (
    _constrained_pairwise_rows,
    _requested_pair_coverage,
    _write_atomic,
)
from anthropocene_rl.observations.normalization import NORMALIZATION_VERSION_V3

BROAD_DOMAIN_GENERATION_SEED = 7_117_001
BROAD_DOMAIN_SCENARIO_VERSION = "elder-broad-domain-50d-v2"
BROAD_DOMAIN_ACCEPTED_FAMILY_COUNTS = {
    "F01": 30,
    "F02": 35,
    "F03": 35,
    "F04": 40,
    "F05": 40,
    "F06": 45,
    "F07": 45,
    "F13": 35,
    "F14": 50,
    "F15": 50,
    "F16": 40,
    "F17": 24,
    "F18": 30,
}
BROAD_DOMAIN_ACCEPTED_PARTITION_COUNTS = {
    "F01": {"train": 20, "development": 5, "heldout": 5},
    "F02": {"train": 25, "development": 5, "heldout": 5},
    "F03": {"train": 25, "development": 5, "heldout": 5},
    "F04": {"train": 28, "development": 6, "heldout": 6},
    "F05": {"train": 28, "development": 6, "heldout": 6},
    "F06": {"train": 31, "development": 7, "heldout": 7},
    "F07": {"train": 31, "development": 7, "heldout": 7},
    "F13": {"train": 25, "development": 5, "heldout": 5},
    "F14": {"train": 34, "development": 8, "heldout": 8},
    "F15": {"train": 34, "development": 8, "heldout": 8},
    "F16": {"train": 28, "development": 4, "heldout": 8},
    "F17": {"train": 18, "development": 3, "heldout": 3},
    "F18": {"train": 20, "development": 4, "heldout": 6},
}

# The powered proof-of-readiness bank intentionally consumed its HELDOUT
# partition.  Scale-out therefore uses wholly new worlds and a fresh split,
# while keeping the total bank size fixed and reallocating coverage toward the
# three families identified by the campaign audit as evidence-limited.
SCALEOUT_GENERATION_SEED = 7_317_001
SCALEOUT_SCENARIO_VERSION = "elder-broad-domain-50d-scaleout-v1"
SCALEOUT_WORLD_SEED_OFFSET = 200_000
SCALEOUT_SUPPLEMENT_WORLD_SEED_OFFSET = 1_000_000
SCALEOUT_PARTITION_SALT = "elder-v2-broad-domain-50d-scaleout-partitions-v1-12441"
SCALEOUT_ACCEPTED_PARTITION_COUNTS = {
    "F01": {"train": 17, "development": 4, "heldout": 4},
    "F02": {"train": 30, "development": 4, "heldout": 4},
    "F03": {"train": 38, "development": 13, "heldout": 9},
    "F04": {"train": 28, "development": 6, "heldout": 6},
    "F05": {"train": 22, "development": 4, "heldout": 4},
    "F06": {"train": 25, "development": 5, "heldout": 5},
    "F07": {"train": 25, "development": 5, "heldout": 5},
    "F13": {"train": 22, "development": 4, "heldout": 4},
    "F14": {"train": 38, "development": 8, "heldout": 8},
    "F15": {"train": 34, "development": 8, "heldout": 8},
    "F16": {"train": 20, "development": 4, "heldout": 4},
    "F17": {"train": 27, "development": 6, "heldout": 9},
    "F18": {"train": 22, "development": 4, "heldout": 6},
}
SCALEOUT_ACCEPTED_FAMILY_COUNTS = {
    family: sum(partitions.values())
    for family, partitions in SCALEOUT_ACCEPTED_PARTITION_COUNTS.items()
}


@lru_cache(maxsize=1)
def build_elder_broad_domain_plan() -> SnapshotBankGenerationConfig:
    trajectories: list[dict[str, Any]] = []
    base_specs: tuple[tuple[str, int, int, dict[str, list[Any]]], ...] = (
        (
            "F01",
            40,
            51_001,
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
            46,
            52_001,
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
            46,
            53_001,
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
            46,
            63_001,
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
    for family_index, (family, count, seed_start, factors) in enumerate(base_specs):
        rows = _constrained_pairwise_rows(
            factors,
            count=count,
            seed=BROAD_DOMAIN_GENERATION_SEED + family_index,
        )
        for index, row in enumerate(rows):
            requested = ["cultivation-only"]
            if family == "F01":
                requested.append("young-memory")
            elif family == "F02":
                requested.append("near-knee-q")
            elif family == "F03":
                requested.append("high-q")
            elif family == "F13":
                requested.append("amenity-bottleneck")
            trajectories.append(
                _trajectory(
                    trajectory_id=f"{family.lower()}-broad-{index + 1:03d}",
                    scenario_family=family,
                    world_seed=seed_start + index,
                    trajectory_family="elder-broad-cultivation-v1",
                    snapshot_steps=[0],
                    economy="cultivation",
                    herd_support_ratio=0.0,
                    requested_tags=requested,
                    classification=("synthetic-stress" if family == "F03" else None),
                    **row,
                )
            )

    # The first bounded native pass left the F03 held-out capacity one record
    # below its frozen quota under the otherwise feasible partition salt.  Add
    # two independent high-q worlds assigned to that split before native
    # generation so a single physical rejection cannot reopen the shortage.
    # These are construction supplements only; F03 promotion remains capped at
    # the original 35 accepted snapshots.
    for index, (world_seed, row) in enumerate(
        (
            (
                71_001,
                {
                    "population": 240,
                    "field_q": 0.60,
                    "reserve_intervals": 2.0,
                    "site_support_quantile": 0.5,
                    "tool_abundance": 0.2,
                    "amenity_coverage": 0.75,
                },
            ),
            (
                71_005,
                {
                    "population": 360,
                    "field_q": 0.56,
                    "reserve_intervals": 1.0,
                    "site_support_quantile": 0.8,
                    "tool_abundance": 0.35,
                    "amenity_coverage": 1.0,
                },
            ),
        )
    ):
        trajectory_id = f"f03-high-q-maintenance-supplement-{index + 1:03d}"
        trajectories.append(
            _trajectory(
                trajectory_id=trajectory_id,
                scenario_family="F03",
                world_seed=world_seed,
                trajectory_family="elder-broad-cultivation-v1",
                snapshot_steps=[0],
                economy="cultivation",
                herd_support_ratio=0.0,
                requested_tags=["cultivation-only", "high-q"],
                classification="synthetic-stress",
                **row,
            )
        )

    herd_specs: tuple[tuple[str, int, int, dict[str, list[Any]]], ...] = (
        (
            "F04",
            54,
            54_001,
            {
                "population": [60, 100, 160, 240, 360],
                "herd_support_ratio": [0.25, 0.4, 0.55, 0.6],
                "reserve_intervals": [0.25, 1.0, 2.0, 4.0],
                "site_support_quantile": [0.2, 0.5, 0.8, 1.0],
                "tool_abundance": [0.2, 0.5, 1.0, 1.5],
                "amenity_coverage": [0.4, 0.7, 1.0],
            },
        ),
        (
            "F05",
            54,
            55_001,
            {
                "population": [80, 140, 220, 320, 460],
                "herd_support_ratio": [0.85, 1.0, 1.1, 1.3, 1.55, 1.8],
                "reserve_intervals": [0.25, 1.0, 2.0],
                "site_support_quantile": [0.2, 0.5, 0.8, 1.0],
                "tool_abundance": [0.15, 0.4, 0.8, 1.2],
                "amenity_coverage": [0.35, 0.65, 1.0],
            },
        ),
    )
    for family_index, (family, count, seed_start, factors) in enumerate(herd_specs):
        rows = _constrained_pairwise_rows(
            factors,
            count=count,
            seed=BROAD_DOMAIN_GENERATION_SEED + 4 + family_index,
        )
        for index, row in enumerate(rows):
            requested = ["husbandry-only"]
            if row["herd_support_ratio"] >= 1.3:
                requested.append("overstocked-herd")
            trajectories.append(
                _trajectory(
                    trajectory_id=f"{family.lower()}-broad-{index + 1:03d}",
                    scenario_family=family,
                    world_seed=seed_start + index,
                    trajectory_family="elder-broad-husbandry-v1",
                    snapshot_steps=[0],
                    economy="husbandry",
                    field_q=0.0,
                    requested_tags=requested,
                    classification=("synthetic-stress" if family == "F05" else None),
                    **row,
                )
            )

    mixed_specs: tuple[tuple[str, int, int, dict[str, list[Any]]], ...] = (
        (
            "F06",
            60,
            56_001,
            {
                "population": [100, 160, 240, 340, 480],
                "field_q": [0.18, 0.24, 0.30, 0.36],
                "herd_support_ratio": [0.25, 0.4, 0.55, 0.7],
                "reserve_intervals": [0.25, 1.0, 2.0, 4.0],
                "site_support_quantile": [0.35, 0.6, 0.85, 1.0],
                "tool_abundance": [0.25, 0.6, 1.0, 1.5],
                "amenity_coverage": [0.45, 0.7, 1.0],
            },
        ),
        (
            "F07",
            60,
            57_001,
            {
                "population": [140, 220, 320, 460, 620],
                "field_q": [0.35, 0.41, 0.48, 0.56],
                "herd_support_ratio": [0.85, 1.0, 1.2, 1.4, 1.6],
                "reserve_intervals": [0.25, 1.0, 2.0],
                "site_support_quantile": [0.2, 0.45, 0.7, 0.9],
                "tool_abundance": [0.1, 0.25, 0.5, 0.9],
                "amenity_coverage": [0.3, 0.55, 0.8],
            },
        ),
    )
    for family_index, (family, count, seed_start, factors) in enumerate(mixed_specs):
        rows = _constrained_pairwise_rows(
            factors,
            count=count,
            seed=BROAD_DOMAIN_GENERATION_SEED + 6 + family_index,
        )
        for index, row in enumerate(rows):
            trajectories.append(
                _trajectory(
                    trajectory_id=f"{family.lower()}-broad-{index + 1:03d}",
                    scenario_family=family,
                    world_seed=seed_start + index,
                    trajectory_family="elder-broad-mixed-v1",
                    snapshot_steps=[0],
                    economy="mixed",
                    requested_tags=["mixed-cultivation-husbandry"],
                    classification=("synthetic-stress" if family == "F07" else None),
                    **row,
                )
            )

    movement_factors: dict[str, list[Any]] = {
        "population": [100, 180, 280, 420],
        "reserve_intervals": [0.25, 1.0, 2.0, 4.0],
        "site_support_quantile": [0.2, 0.5, 0.8, 1.0],
        "tool_abundance": [0.2, 0.5, 1.0, 1.5],
        "amenity_coverage": [0.4, 0.7, 1.0],
        "herd_support_ratio": [0.0, 0.45],
    }
    for index, row in enumerate(
        _constrained_pairwise_rows(
            movement_factors,
            count=35,
            seed=BROAD_DOMAIN_GENERATION_SEED + 14,
        )
    ):
        world_seed = 64_001 + index
        group_id = f"f14-capital-lock-pair-{index + 1:03d}"
        economy = "mixed" if row["herd_support_ratio"] > 0 else "cultivation"
        tag = "mixed-cultivation-husbandry" if economy == "mixed" else "cultivation-only"
        for variant, field_q in (("low-lock-in", 0.18), ("high-lock-in", 0.58)):
            trajectories.append(
                _trajectory(
                    trajectory_id=f"{group_id}-{variant}",
                    scenario_family="F14",
                    world_seed=world_seed,
                    trajectory_family="elder-broad-movement-lock-in-v1",
                    lineage_id=group_id,
                    counterfactual_group_id=group_id,
                    snapshot_steps=[0],
                    economy=economy,
                    field_q=field_q,
                    requested_tags=[tag],
                    causal_variant=variant,
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
    contrasts = (
        ("isolated", "strong-food-import"),
        ("mild-import", "surplus-export"),
        ("food-shortfall-contact", "material-tool-import"),
        ("immigration-secure", "emigration-distress"),
        ("immigration-labor", "immigration-stress"),
    )
    pairs = [contrast for _ in range(10) for contrast in contrasts]
    rows = _constrained_pairwise_rows(
        open_economy_factors,
        count=len(pairs),
        seed=BROAD_DOMAIN_GENERATION_SEED + 15,
    )
    for index, (row, variants) in enumerate(zip(rows, pairs, strict=True)):
        world_seed = 65_001 + index
        group_id = f"f15-open-economy-pair-{index + 1:03d}"
        for variant in variants:
            requested = ["cultivation-only"]
            if variant == "isolated":
                requested.append("isolated")
            elif variant in {
                "strong-food-import",
                "mild-import",
                "food-shortfall-contact",
                "material-tool-import",
                "surplus-export",
            }:
                requested.append("trade-active")
                if variant in {"strong-food-import", "mild-import", "food-shortfall-contact"}:
                    requested.append("food-import")
                    if variant == "strong-food-import":
                        requested.append("trade-dependent")
                elif variant == "material-tool-import":
                    requested.append("material-tool-import")
                else:
                    requested.append("food-export")
            elif variant.startswith("immigration-"):
                requested.append("migration-positive")
            else:
                requested.append("migration-negative")
            trajectories.append(
                _trajectory(
                    trajectory_id=f"{group_id}-{variant}",
                    scenario_family="F15",
                    world_seed=world_seed,
                    trajectory_family="elder-broad-open-economy-v1",
                    lineage_id=group_id,
                    counterfactual_group_id=group_id,
                    snapshot_steps=([2] if variant == "material-tool-import" else [1]),
                    economy="cultivation",
                    herd_support_ratio=0.0,
                    requested_tags=requested,
                    causal_variant=variant,
                    **row,
                )
            )

    # The first native feasibility pass showed that the two delivered-trade
    # contrasts are substantially harder to realize than the migration and
    # isolation contrasts.  Preserve the orthogonal 50-pair design above, then
    # spend the remaining candidate budget on independent worlds using factor
    # combinations that have already produced both siblings natively.  These
    # remain surplus candidates: promotion still selects exactly five complete
    # pairs per F15 contrast after measured-state validation.
    trade_supplements = (
        (
            "food-tool",
            30,
            ("food-shortfall-contact", "material-tool-import"),
            {
                "population": 340,
                "field_q": 0.32,
                "reserve_intervals": 2.0,
                "site_support_quantile": 0.5,
                "tool_abundance": 1.4,
                "amenity_coverage": 1.0,
            },
        ),
        (
            "import-export",
            14,
            ("mild-import", "surplus-export"),
            {
                "population": 340,
                "field_q": 0.24,
                "reserve_intervals": 2.0,
                "site_support_quantile": 0.5,
                "tool_abundance": 0.8,
                "amenity_coverage": 0.8,
            },
        ),
    )
    supplement_index = 0
    for label, count, variants, row in trade_supplements:
        for local_index in range(count):
            supplement_index += 1
            world_seed = 69_000 + supplement_index
            group_id = f"f15-open-economy-{label}-supplement-{local_index + 1:03d}"
            for variant in variants:
                requested = ["cultivation-only", "trade-active"]
                if variant == "food-shortfall-contact":
                    requested.append("food-import")
                elif variant == "material-tool-import":
                    requested.append("material-tool-import")
                elif variant == "mild-import":
                    requested.append("food-import")
                else:
                    requested.append("food-export")
                trajectories.append(
                    _trajectory(
                        trajectory_id=f"{group_id}-{variant}",
                        scenario_family="F15",
                        world_seed=world_seed,
                        trajectory_family="elder-broad-open-economy-v1",
                        lineage_id=group_id,
                        counterfactual_group_id=group_id,
                        snapshot_steps=(
                            [2] if variant == "material-tool-import" else [1]
                        ),
                        economy="cultivation",
                        herd_support_ratio=0.0,
                        requested_tags=requested,
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
    for index, row in enumerate(
        _constrained_pairwise_rows(
            fission_factors,
            count=14,
            seed=BROAD_DOMAIN_GENERATION_SEED + 16,
        )
    ):
        world_seed = 66_001 + index
        group_id = f"f16-exit-opportunity-pair-{index + 1:03d}"
        for variant in ("poor-exit-opportunity", "strong-exit-opportunity"):
            trajectories.append(
                _trajectory(
                    trajectory_id=f"{group_id}-{variant}",
                    scenario_family="F16",
                    world_seed=world_seed,
                    trajectory_family="elder-broad-fission-v1",
                    lineage_id=group_id,
                    counterfactual_group_id=group_id,
                    snapshot_steps=[0, 1],
                    economy="cultivation",
                    herd_support_ratio=0.0,
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
            seed=BROAD_DOMAIN_GENERATION_SEED + 17,
        )
    ):
        trajectories.append(
            _trajectory(
                trajectory_id=f"f17-memory-{index + 1:03d}",
                scenario_family="F17",
                world_seed=67_001 + index,
                trajectory_family="elder-broad-memory-v1",
                snapshot_steps=[0, 3, 9],
                economy="cultivation",
                herd_support_ratio=0.0,
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
            count=21,
            seed=BROAD_DOMAIN_GENERATION_SEED + 18,
        )
    ):
        world_seed = 68_001 + index
        group_id = f"f18-frontier-pair-{index + 1:03d}"
        common = {
            key: value for key, value in row.items() if key not in {"primitive_q", "plough_q"}
        }
        for variant, frontier, field_q, tag in (
            ("primitive", "primitive", row["primitive_q"], "near-knee-q"),
            ("plough", "plough", row["plough_q"], "high-q"),
        ):
            trajectories.append(
                _trajectory(
                    trajectory_id=f"{group_id}-{variant}",
                    scenario_family="F18",
                    world_seed=world_seed,
                    trajectory_family="elder-broad-frontier-v1",
                    lineage_id=group_id,
                    counterfactual_group_id=group_id,
                    snapshot_steps=[0],
                    economy="cultivation",
                    field_q=field_q,
                    herd_support_ratio=0.0,
                    frontier=frontier,
                    requested_tags=["cultivation-only", tag],
                    **common,
                )
            )

    return SnapshotBankGenerationConfig.model_validate(
        {
            "config_schema_version": "snapshot-bank-generation-v1",
            "bank_kind": "elder-broad-domain",
            "bank_id": "elder-v2-broad-domain-50d-pilot-v2",
            "bank_version": "2.0.0-50d-event-aligned-candidate",
            "native_generator_version": "snapshot-bank-native-v2",
            "generation_seed": BROAD_DOMAIN_GENERATION_SEED,
            "world": {
                "width": 48,
                "height": 24,
                "players": 4,
                "start_separation": 6,
                "annual_human_growth_rate": 0.01145,
            },
            "partition": {
                "salt": "elder-v2-broad-domain-50d-partitions-v2-830",
                "train_fraction": 0.70,
                "development_fraction": 0.15,
                "heldout_fraction": 0.15,
            },
            "calibration_rollout_steps": 5,
            "f15_shared_capture_horizon_steps": 5,
            "verify_deterministic_regeneration": True,
            "local_movement_radius": 1.0,
            "capability_experience_thresholds": {
                "field-retention": 225.0,
                "herd-management": 225.0,
                "resident-fishing": 225.0,
            },
            "normalization_version": NORMALIZATION_VERSION_V3,
            "broad_domain_accepted_partition_counts": (
                BROAD_DOMAIN_ACCEPTED_PARTITION_COUNTS
            ),
            "trajectories": trajectories,
        }
    )


@lru_cache(maxsize=1)
def build_elder_scaleout_curriculum_plan() -> SnapshotBankGenerationConfig:
    """Build the fresh, evidence-weighted scale-out curriculum.

    The construction reuses the already audited factor design, but never its
    worlds or causal identities.  Candidate capacity is moved from families
    with ample proof-of-readiness coverage into F03, F15 and F17 without
    exceeding the native generator's 800-candidate safety bound.
    """

    base = build_elder_broad_domain_plan()
    grouped: dict[tuple[str, str], list[Any]] = {}
    for trajectory in base.trajectories:
        group_id = trajectory.counterfactual_group_id or trajectory.lineage_id
        grouped.setdefault((trajectory.scenario_family, group_id), []).append(trajectory)

    # Free exactly 36 candidate slots while preserving complete groups.
    trim_records = {"F05": 6, "F06": 4, "F07": 4, "F14": 10, "F16": 12}
    dropped: set[tuple[str, str]] = set()
    for family, requested_records in trim_records.items():
        remaining = requested_records
        family_groups = sorted(
            (item for item in grouped.items() if item[0][0] == family),
            key=lambda item: item[0][1],
            reverse=True,
        )
        for key, trajectories in family_groups:
            records = sum(len(trajectory.snapshot_steps) for trajectory in trajectories)
            if records > remaining:
                continue
            dropped.add(key)
            remaining -= records
            if remaining == 0:
                break
        if remaining:
            raise AssertionError(f"Unable to trim {requested_records} records from {family}")

    # The fresh seed cohort needs more isolated/import counterfactuals, while
    # the base plan has a large surplus of the food/tool F15 contrast. Replace
    # twenty complete food/tool pairs without changing the 800-candidate cap.
    dropped.update(
        ("F15", f"f15-open-economy-food-tool-supplement-{index:03d}")
        for index in range(11, 31)
    )

    retained = [
        trajectory
        for trajectory in base.trajectories
        if (
            trajectory.scenario_family,
            trajectory.counterfactual_group_id or trajectory.lineage_id,
        )
        not in dropped
    ]
    f03_sources = sorted(
        (trajectory for trajectory in base.trajectories if trajectory.scenario_family == "F03"),
        key=lambda trajectory: trajectory.trajectory_id,
    )[:12]
    f17_sources = sorted(
        (trajectory for trajectory in base.trajectories if trajectory.scenario_family == "F17"),
        key=lambda trajectory: trajectory.trajectory_id,
    )[:8]
    f15_isolation_groups: dict[str, list[Any]] = {}
    for trajectory in base.trajectories:
        if trajectory.scenario_family != "F15":
            continue
        group_id = trajectory.counterfactual_group_id or trajectory.lineage_id
        f15_isolation_groups.setdefault(group_id, []).append(trajectory)
    f15_isolation_sources = [
        trajectories
        for _, trajectories in sorted(f15_isolation_groups.items())
        if {trajectory.causal_variant for trajectory in trajectories}
        == {"isolated", "strong-food-import"}
    ]
    if len(f15_isolation_sources) != 10:
        raise AssertionError("Expected ten base isolated/import F15 pairs")

    trajectories = [
        _scaleout_trajectory_payload(trajectory)
        for trajectory in retained
    ]
    trajectories.extend(
        _scaleout_trajectory_payload(
            trajectory,
            density_index=index,
        )
        for index, trajectory in enumerate((*f03_sources, *f17_sources), start=1)
    )
    for index, source_group in enumerate(
        (*f15_isolation_sources, *f15_isolation_sources),
        start=101,
    ):
        trajectories.extend(
            _scaleout_trajectory_payload(trajectory, density_index=index)
            for trajectory in source_group
        )
    if sum(len(item["snapshot_steps"]) for item in trajectories) != 800:
        raise AssertionError("Scale-out curriculum must request exactly 800 candidates")

    payload = base.model_dump(mode="python", exclude={"trajectories"})
    payload.update(
        {
            "bank_id": "elder-v2-broad-domain-50d-scaleout-v1",
            "bank_version": "3.0.0-50d-scaleout-candidate",
            "generation_seed": SCALEOUT_GENERATION_SEED,
            "partition": {
                "salt": SCALEOUT_PARTITION_SALT,
                "train_fraction": 0.70,
                "development_fraction": 0.15,
                "heldout_fraction": 0.15,
            },
            "broad_domain_accepted_partition_counts": (
                SCALEOUT_ACCEPTED_PARTITION_COUNTS
            ),
            "trajectories": trajectories,
        }
    )
    return SnapshotBankGenerationConfig.model_validate(payload)


def _scaleout_trajectory_payload(
    trajectory: Any,
    *,
    density_index: int | None = None,
) -> dict[str, Any]:
    payload = trajectory.model_dump(mode="python")
    density_prefix = "" if density_index is None else f"density-{density_index:03d}-"
    identity_prefix = f"scaleout-v1-{density_prefix}"

    def fresh_identity(value: str | None) -> str | None:
        return None if value is None else identity_prefix + value

    payload.update(
        {
            "trajectory_id": fresh_identity(trajectory.trajectory_id),
            "scenario_version": SCALEOUT_SCENARIO_VERSION,
            "world_seed": (
                trajectory.world_seed
                + SCALEOUT_WORLD_SEED_OFFSET
                + (
                    SCALEOUT_SUPPLEMENT_WORLD_SEED_OFFSET
                    if density_index is not None
                    else 0
                )
                + (density_index or 0)
            ),
            "trajectory_family": trajectory.trajectory_family + "-scaleout-v1",
            "lineage_id": fresh_identity(trajectory.lineage_id),
            "counterfactual_group_id": fresh_identity(
                trajectory.counterfactual_group_id
            ),
        }
    )
    return payload


def broad_domain_plan_summary(config: SnapshotBankGenerationConfig) -> dict[str, Any]:
    resolved, _ = config.resolve_partitions()
    family_snapshots: Counter[str] = Counter()
    family_worlds: dict[str, set[int]] = {}
    partition_snapshots: Counter[str] = Counter()
    for trajectory in resolved:
        snapshots = len(trajectory.snapshot_steps)
        family_snapshots[trajectory.scenario_family] += snapshots
        partition_snapshots[trajectory.partition.value] += snapshots
        family_worlds.setdefault(trajectory.scenario_family, set()).add(trajectory.world_seed)
    pairwise = {
        family: _requested_pair_coverage(
            [
                trajectory.model_dump(mode="json")
                for trajectory in config.trajectories
                if trajectory.scenario_family == family
            ],
            fields=(
                "economy",
                "population",
                "field_q",
                "herd_support_ratio",
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
    accepted_partitions = config.broad_domain_accepted_partition_counts
    if accepted_partitions is None:
        raise ValueError("Broad-domain plan summary requires accepted partition counts")
    accepted_families = {
        family: sum(partitions.values())
        for family, partitions in accepted_partitions.items()
    }
    return {
        "status": "PLAN_READY_NOT_GENERATED",
        "bank_id": config.bank_id,
        "generator_config_sha256": config.canonical_sha256(),
        "requested_snapshot_count": sum(family_snapshots.values()),
        "target_accepted_snapshot_count": sum(accepted_families.values()),
        "target_accepted_family_counts": accepted_families,
        "target_accepted_partition_counts": accepted_partitions,
        "requested_family_snapshot_counts": dict(sorted(family_snapshots.items())),
        "requested_partition_snapshot_counts": dict(sorted(partition_snapshots.items())),
        "family_world_counts": {
            family: len(worlds) for family, worlds in sorted(family_worlds.items())
        },
        "requested_pairwise_coverage": pairwise,
        "primary_episode_decisions": 50,
        "nominal_episode_years": 1_000,
        "f15_shared_capture_horizon_decisions": (
            config.f15_shared_capture_horizon_steps
        ),
        "active_policy_domains": ["cultivation", "husbandry", "movement"],
        "live_native_dynamics": ["trade", "immigration", "emigration", "fission"],
        "excluded_policy_domains": ["fishing"],
        "normalization_rule": (
            "freeze the accepted manifest, then fit a fresh normalization-v3 instance "
            "from TRAIN resets plus legal calibration rollouts only"
        ),
    }


def save_elder_broad_domain_plan(
    *,
    config_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    config = build_elder_broad_domain_plan()
    summary = broad_domain_plan_summary(config)
    payload = config.model_dump(mode="json")
    if "pilot_accepted_partition_counts" not in config.model_fields_set:
        payload.pop("pilot_accepted_partition_counts", None)
    _write_atomic(
        Path(config_path),
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    )
    _write_atomic(
        Path(summary_path),
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def save_elder_scaleout_curriculum_plan(
    *,
    config_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    config = build_elder_scaleout_curriculum_plan()
    summary = broad_domain_plan_summary(config)
    payload = config.model_dump(mode="json")
    if "pilot_accepted_partition_counts" not in config.model_fields_set:
        payload.pop("pilot_accepted_partition_counts", None)
    _write_atomic(
        Path(config_path),
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
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
    economy: str,
    requested_tags: list[str],
    population: int,
    field_q: float,
    herd_support_ratio: float,
    reserve_intervals: float,
    site_support_quantile: float,
    tool_abundance: float,
    amenity_coverage: float,
    lineage_id: str | None = None,
    counterfactual_group_id: str | None = None,
    classification: str | None = None,
    frontier: str = "primitive",
    causal_variant: str = "none",
) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory_id,
        "scenario_family": scenario_family,
        "scenario_version": BROAD_DOMAIN_SCENARIO_VERSION,
        "world_seed": world_seed,
        "trajectory_family": trajectory_family,
        "lineage_id": lineage_id or trajectory_id,
        "counterfactual_group_id": counterfactual_group_id,
        "source_controller": "bounded-exploratory-elder-v1",
        "classification": classification or "controlled-valid-fixture",
        "historical_or_consumed_diagnostic": False,
        "snapshot_steps": snapshot_steps,
        "economy": economy,
        "population": population,
        "field_q": field_q,
        "herd_support_ratio": herd_support_ratio,
        "reserve_intervals": reserve_intervals,
        "fishing_ground_target": 0,
        "site_support_quantile": site_support_quantile,
        "tool_abundance": tool_abundance,
        "amenity_coverage": amenity_coverage,
        "frontier": frontier,
        "causal_variant": causal_variant,
        "requested_tags": requested_tags,
    }
