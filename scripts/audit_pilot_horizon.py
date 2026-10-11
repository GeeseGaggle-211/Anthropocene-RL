"""Audit when excluded husbandry/fishing domains emerge in pilot rollouts."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from anthropocene_rl.contracts import (
    DOMESTIC_HERD_COUNT_OFFSET,
    FISHING_CAPABILITY_AVAILABLE_OFFSET,
    FISHING_PRESENT_COUNT_OFFSET,
    HERD_DEVELOPMENT_MASK_OFFSET,
    HERD_RETENTION_MASK_OFFSET,
    HUSBANDRY_CAPABILITY_AVAILABLE_OFFSET,
    MOVEMENT_CANDIDATE_COUNT,
    MOVEMENT_CANDIDATE_MASK_OFFSET,
    MOVEMENT_ENABLED_OFFSET,
)
from anthropocene_rl.training.overfit import _plain_mapping
from anthropocene_rl.training.pilot import (
    CampaignDataset,
    _make_environment,
    _raw_observation,
    _reset_request,
    _validate_campaign_reset,
)


def audit_horizon(
    *,
    manifest: Path,
    blob_root: Path,
    normalizer: Path,
    bridge_dll: Path,
    dotnet: str,
    maximum_steps: int,
    required_safe_steps: int,
) -> dict[str, Any]:
    dataset = CampaignDataset(
        manifest_path=manifest,
        blob_root=blob_root,
        normalizer_path=normalizer,
    )
    records = tuple(sorted(dataset.manifest.records, key=lambda row: row.snapshot_id))
    first = dataset.selection(records[0])
    env = _make_environment(
        bridge_path=bridge_dll.resolve(),
        dotnet=dotnet,
        selection=first,
        episode_limit=maximum_steps,
    )
    rows: list[dict[str, Any]] = []
    try:
        for record in records:
            selection = dataset.selection(record)
            _, info = env.reset(options=_reset_request(selection, maximum_steps))
            _validate_campaign_reset(_plain_mapping(info["native_result"]), selection)
            first_capability: int | None = None
            first_animals: int | None = None
            first_mask: int | None = None
            first_fishing: int | None = None
            first_fishing_ground: int | None = None
            first_movement: int | None = None
            replay_valid = False
            steps = 0
            for step in range(1, maximum_steps + 1):
                _, _, terminated, truncated, info = env.step(
                    {
                        "continuous": [0.5, 0.0, 0.2, 1.0],
                        "movement_index": 0,
                    }
                )
                raw = _raw_observation(info).values
                if raw[HUSBANDRY_CAPABILITY_AVAILABLE_OFFSET] > 0.5 and first_capability is None:
                    first_capability = step
                if raw[DOMESTIC_HERD_COUNT_OFFSET] > 0.5 and first_animals is None:
                    first_animals = step
                if (
                    raw[HERD_RETENTION_MASK_OFFSET] > 0.5 or raw[HERD_DEVELOPMENT_MASK_OFFSET] > 0.5
                ) and first_mask is None:
                    first_mask = step
                if raw[FISHING_CAPABILITY_AVAILABLE_OFFSET] > 0.5 and first_fishing is None:
                    first_fishing = step
                if raw[FISHING_PRESENT_COUNT_OFFSET] > 0.5 and first_fishing_ground is None:
                    first_fishing_ground = step
                if (
                    raw[MOVEMENT_ENABLED_OFFSET] > 0.5
                    or any(
                        raw[MOVEMENT_CANDIDATE_MASK_OFFSET + candidate] > 0.5
                        for candidate in range(1, MOVEMENT_CANDIDATE_COUNT)
                    )
                ) and first_movement is None:
                    first_movement = step
                steps += 1
                if terminated or truncated:
                    break
            replay = env.verify_replay()
            replay_valid = int(replay.get("steps", 0)) == steps and steps > 0
            rows.append(
                {
                    "snapshot_id": record.snapshot_id,
                    "scenario_family": record.scenario_family,
                    "partition": record.partition.value,
                    "world_seed": record.world_seed,
                    "first_husbandry_capability_step": first_capability,
                    "first_domestic_herd_step": first_animals,
                    "first_herd_mask_step": first_mask,
                    "first_fishing_capability_step": first_fishing,
                    "first_fishing_ground_step": first_fishing_ground,
                    "first_movement_enabled_step": first_movement,
                    "replay_valid": replay_valid,
                }
            )
    finally:
        env.close()

    def count_at_or_before(field: str, horizon: int) -> int:
        return sum(row[field] is not None and int(row[field]) <= horizon for row in rows)

    horizons = {
        str(horizon): {
            "husbandry_capability": count_at_or_before("first_husbandry_capability_step", horizon),
            "domestic_herd": count_at_or_before("first_domestic_herd_step", horizon),
            "herd_mask": count_at_or_before("first_herd_mask_step", horizon),
            "fishing_capability": count_at_or_before("first_fishing_capability_step", horizon),
            "fishing_ground": count_at_or_before("first_fishing_ground_step", horizon),
            "movement_enabled": count_at_or_before("first_movement_enabled_step", horizon),
        }
        for horizon in range(1, maximum_steps + 1)
    }
    required = horizons[str(required_safe_steps)]
    violations = [
        row
        for row in rows
        if any(
            row[field] is not None and int(row[field]) <= required_safe_steps
            for field in (
                "first_husbandry_capability_step",
                "first_domestic_herd_step",
                "first_herd_mask_step",
                "first_fishing_capability_step",
                "first_fishing_ground_step",
                "first_movement_enabled_step",
            )
        )
    ]
    return {
        "status": "PASS" if not violations and all(row["replay_valid"] for row in rows) else "FAIL",
        "scope": "fixed-legal-directive-structural-horizon-audit",
        "coverage_note": (
            "This fixed-action audit locates a conservative episode boundary; "
            "the campaign independently fail-fast checks every reset and post-step state."
        ),
        "manifest_sha256": dataset.manifest_sha256,
        "records": len(rows),
        "maximum_steps": maximum_steps,
        "required_safe_steps": required_safe_steps,
        "fixed_requested_action": {
            "retained_field_operation": 0.5,
            "retained_herd_fraction": 0.0,
            "development_budget": 0.2,
            "field_development_share": 1.0,
            "movement_candidate": -1,
        },
        "horizon_counts": horizons,
        "required_horizon_counts": required,
        "first_activation_step_counts": {
            field: dict(
                sorted(
                    Counter(str(row[field]) for row in rows if row[field] is not None).items(),
                    key=lambda item: int(item[0]),
                )
            )
            for field in (
                "first_husbandry_capability_step",
                "first_domestic_herd_step",
                "first_herd_mask_step",
                "first_fishing_capability_step",
                "first_fishing_ground_step",
                "first_movement_enabled_step",
            )
        },
        "required_horizon_violations": violations,
        "replay_failure_snapshot_ids": [
            row["snapshot_id"] for row in rows if not row["replay_valid"]
        ],
        "rows": rows,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/manifests/elder-v2-cultivation-pilot-v1.json"),
    )
    parser.add_argument("--blob-root", type=Path, default=Path("datasets/blobs"))
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=Path("evidence/cultivation-pilot-v1/final-train-normalization.json"),
    )
    parser.add_argument(
        "--bridge-dll",
        type=Path,
        default=Path("integration/bin/Release/net8.0/Anthropocene.RLBridge.dll"),
    )
    parser.add_argument("--dotnet", default="dotnet")
    parser.add_argument("--maximum-steps", type=int, default=8)
    parser.add_argument("--required-safe-steps", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/cultivation-ppo-pilot-v1/structural-horizon-audit.json"),
    )
    args = parser.parse_args()
    result = audit_horizon(
        manifest=args.manifest,
        blob_root=args.blob_root,
        normalizer=args.normalizer,
        bridge_dll=args.bridge_dll,
        dotnet=args.dotnet,
        maximum_steps=args.maximum_steps,
        required_safe_steps=args.required_safe_steps,
    )
    _write_json(args.output, result)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "records", "horizon_counts")},
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
