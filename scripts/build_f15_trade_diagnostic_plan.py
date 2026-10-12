"""Build a non-promotable native diagnostic for the F15 import/export contrast."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from anthropocene_rl.datasets.generation_config import SnapshotBankGenerationConfig

CONTRAST = ("mild-import", "surplus-export")


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--observe-failures", action="store_true")
    parser.add_argument("--snapshot-step", type=int)
    args = parser.parse_args()

    config = SnapshotBankGenerationConfig.load(args.config)
    groups: dict[str, list[Any]] = defaultdict(list)
    for trajectory in config.trajectories:
        group_id = trajectory.counterfactual_group_id or trajectory.lineage_id
        groups[group_id].append(trajectory)
    selected_groups = {
        group_id
        for group_id, trajectories in groups.items()
        if tuple(sorted(item.causal_variant for item in trajectories)) == CONTRAST
    }
    selected = [
        trajectory
        for trajectory in config.native_payload()["trajectories"]
        if (trajectory.get("counterfactual_group_id") or trajectory["lineage_id"])
        in selected_groups
    ]
    if args.snapshot_step is not None:
        if args.snapshot_step < 0:
            raise ValueError("--snapshot-step must be non-negative")
        for trajectory in selected:
            trajectory["snapshot_steps"] = [args.snapshot_step]
    if args.observe_failures:
        # Requested tags are acceptance predicates only; scenario construction
        # depends on the causal variant and declared physical inputs.  Removing
        # the trade predicates in this explicitly non-promotable companion run
        # lets us retain reset/calibration observations for states that the
        # strict diagnostic correctly rejected.
        for trajectory in selected:
            trajectory["requested_tags"] = ["cultivation-only"]
    requested_records = sum(len(item["snapshot_steps"]) for item in selected)
    if len(selected_groups) != 24 or requested_records != 48:
        raise ValueError(
            "The frozen diagnostic must contain exactly 24 complete pairs / 48 records; "
            f"found groups={len(selected_groups)}, records={requested_records}"
        )

    payload = config.native_payload()
    payload.update(
        {
            # This learner-owned native-generator mode preserves the complete
            # broad-domain simulator configuration while permitting a small,
            # explicitly non-promotable construction diagnostic.
            "bank_kind": "elder-broad-domain-diagnostic",
            "bank_id": "elder-v2-f15-import-export-construction-diagnostic-v1",
            "bank_version": "non-promotable-diagnostic-v1",
            "verify_deterministic_regeneration": False,
            "trajectories": selected,
        }
    )
    identity_payload = dict(payload)
    identity_payload.pop("generator_config_sha256", None)
    payload["generator_config_sha256"] = _canonical_sha256(identity_payload)
    metadata = {
        "status": (
            "NON_PROMOTABLE_FAILURE_OBSERVATION_DIAGNOSTIC"
            if args.observe_failures
            else "NON_PROMOTABLE_NATIVE_SCENARIO_CONSTRUCTION_DIAGNOSTIC"
        ),
        "source_campaign_config_sha256": config.canonical_sha256(),
        "diagnostic_config_sha256": payload["generator_config_sha256"],
        "contrast": list(CONTRAST),
        "causal_group_count": len(selected_groups),
        "requested_record_count": requested_records,
        "trade_acceptance_predicates_removed": args.observe_failures,
        "snapshot_step_override": args.snapshot_step,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
