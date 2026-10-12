"""Build the bounded, outcome-blind F15 import/export construction panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from anthropocene_rl.datasets.generation_config import SnapshotBankGenerationConfig
from anthropocene_rl.datasets.partition import DeterministicPartitioner, LineageIdentity

PANEL_SIZE = 24
SEED_SEARCH_START = 72_001
EVENT_ALIGNED_SEED_SEARCH_START = 73_001
TRAJECTORY_FAMILY = "elder-broad-open-economy-v1"
EVENT_ALIGNED_TRAJECTORY_FAMILY = "elder-broad-open-economy-v2"
VARIANTS = ("mild-import", "surplus-export")

# Each row is copied from a native-complete causal pair.  The only deliberate
# change is the declared snapshot timestamp.  Eight independent TRAIN worlds
# per row keep this bounded at the reviewer-approved 24-pair ceiling.
FACTOR_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "source_pair": "f15-open-economy-pair-007",
        "population": 130,
        "field_q": 0.40,
        "reserve_intervals": 0.25,
        "site_support_quantile": 0.20,
        "tool_abundance": 0.15,
        "amenity_coverage": 0.80,
        "snapshot_step": 1,
    },
    {
        "source_pair": "f15-open-economy-pair-017",
        "population": 130,
        "field_q": 0.24,
        "reserve_intervals": 4.00,
        "site_support_quantile": 1.00,
        "tool_abundance": 0.40,
        "amenity_coverage": 0.30,
        "snapshot_step": 2,
    },
    {
        "source_pair": "f15-open-economy-pair-022",
        "population": 180,
        "field_q": 0.48,
        "reserve_intervals": 0.25,
        "site_support_quantile": 0.50,
        "tool_abundance": 0.80,
        "amenity_coverage": 1.00,
        "snapshot_step": 2,
    },
)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _train_seeds(
    config: SnapshotBankGenerationConfig, *, event_aligned: bool
) -> list[int]:
    partitioner = DeterministicPartitioner(
        salt=config.partition.salt,
        train_fraction=config.partition.train_fraction,
        development_fraction=config.partition.development_fraction,
        heldout_fraction=config.partition.heldout_fraction,
    )
    seeds: list[int] = []
    candidate = (
        EVENT_ALIGNED_SEED_SEARCH_START if event_aligned else SEED_SEARCH_START
    )
    prefix = (
        "f15-import-export-event-aligned"
        if event_aligned
        else "f15-import-export-targeted"
    )
    trajectory_family = (
        EVENT_ALIGNED_TRAJECTORY_FAMILY if event_aligned else TRAJECTORY_FAMILY
    )
    while len(seeds) < PANEL_SIZE:
        group_id = f"{prefix}-{len(seeds) + 1:03d}"
        partition = partitioner.assign(
            LineageIdentity(
                candidate,
                trajectory_family,
                group_id,
                group_id,
            )
        )
        if partition.value == "train":
            seeds.append(candidate)
        candidate += 1
    return seeds


def _trajectory(
    *,
    group_id: str,
    world_seed: int,
    variant: str,
    factors: dict[str, Any],
    event_aligned: bool,
) -> dict[str, Any]:
    requested_tags = [
        "cultivation-only",
        "trade-active",
        "food-import" if variant == "mild-import" else "food-export",
    ]
    return {
        "trajectory_id": f"{group_id}-{variant}",
        "scenario_family": "F15",
        "scenario_version": (
            "elder-broad-domain-50d-v2"
            if event_aligned
            else "elder-broad-domain-50d-v1"
        ),
        "world_seed": world_seed,
        "trajectory_family": (
            EVENT_ALIGNED_TRAJECTORY_FAMILY
            if event_aligned
            else TRAJECTORY_FAMILY
        ),
        "lineage_id": group_id,
        "counterfactual_group_id": group_id,
        "source_controller": "bounded-exploratory-elder-v1",
        "classification": "controlled-valid-fixture",
        "historical_or_consumed_diagnostic": False,
        "snapshot_steps": [1 if event_aligned else factors["snapshot_step"]],
        "economy": "cultivation",
        "population": factors["population"],
        "field_q": factors["field_q"],
        "herd_support_ratio": 0.0,
        "reserve_intervals": factors["reserve_intervals"],
        "fishing_ground_target": 0,
        "site_support_quantile": factors["site_support_quantile"],
        "tool_abundance": factors["tool_abundance"],
        "amenity_coverage": factors["amenity_coverage"],
        "frontier": "primitive",
        "causal_variant": variant,
        "requested_tags": requested_tags,
        "partition": "train",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--event-aligned", action="store_true")
    args = parser.parse_args()

    config = SnapshotBankGenerationConfig.load(args.config)
    seeds = _train_seeds(config, event_aligned=args.event_aligned)
    prefix = (
        "f15-import-export-event-aligned"
        if args.event_aligned
        else "f15-import-export-targeted"
    )
    trajectories = []
    groups = []
    per_config = PANEL_SIZE // len(FACTOR_CONFIGS)
    for index, world_seed in enumerate(seeds):
        factors = FACTOR_CONFIGS[index // per_config]
        group_id = f"{prefix}-{index + 1:03d}"
        groups.append(
            {
                "group_id": group_id,
                "world_seed": world_seed,
                "partition": "train",
                "factor_config": index // per_config + 1,
                **factors,
            }
        )
        trajectories.extend(
            _trajectory(
                group_id=group_id,
                world_seed=world_seed,
                variant=variant,
                factors=factors,
                event_aligned=args.event_aligned,
            )
            for variant in VARIANTS
        )

    payload = config.native_payload()
    payload.update(
        {
            "bank_kind": "elder-broad-domain-diagnostic",
            "bank_id": (
                "elder-v2-f15-import-export-event-aligned-panel-v2"
                if args.event_aligned
                else "elder-v2-f15-import-export-targeted-panel-v1"
            ),
            "bank_version": (
                "non-promotable-event-aligned-diagnostic-v2"
                if args.event_aligned
                else "non-promotable-diagnostic-v1"
            ),
            "native_generator_version": (
                "snapshot-bank-native-v2"
                if args.event_aligned
                else payload["native_generator_version"]
            ),
            "f15_shared_capture_horizon_steps": 5 if args.event_aligned else 0,
            "verify_deterministic_regeneration": False,
            "trajectories": trajectories,
        }
    )
    identity_payload = dict(payload)
    identity_payload.pop("generator_config_sha256", None)
    payload["generator_config_sha256"] = _canonical_sha256(identity_payload)
    metadata = {
        "status": (
            "NON_PROMOTABLE_EVENT_ALIGNED_PANEL"
            if args.event_aligned
            else "NON_PROMOTABLE_BOUNDED_TARGETED_PANEL"
        ),
        "source_campaign_config_sha256": config.canonical_sha256(),
        "diagnostic_config_sha256": payload["generator_config_sha256"],
        "selection_rule": (
            "first 24 unused world seeds at or above "
            f"{EVENT_ALIGNED_SEED_SEARCH_START if args.event_aligned else SEED_SEARCH_START} "
            "assigned TRAIN by the frozen pre-outcome partition contract"
        ),
        "outcome_fields_used_for_seed_selection": [],
        "causal_group_count": PANEL_SIZE,
        "requested_record_count": len(trajectories),
        "factor_config_count": len(FACTOR_CONFIGS),
        "shared_capture_rule": (
            "earliest shared decision from 1 through 5 where mild-import has "
            "positive delivered food imports and surplus-export has positive "
            "delivered food exports"
            if args.event_aligned
            else None
        ),
        "success_gate": (
            "at least two complete, unique, scope-valid TRAIN pairs reproduced "
            "deterministically; otherwise NATIVE_SCENARIO_CONSTRUCTION_BLOCKED"
        ),
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in metadata.items() if key != "groups"}))


if __name__ == "__main__":
    main()
