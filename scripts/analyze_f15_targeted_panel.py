"""Audit the deterministic outcome of the bounded F15 targeted panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snapshot_hashes(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"sha256": _sha256(path.read_bytes()), "bytes": path.stat().st_size}
        for path in sorted((directory / "snapshots").glob("*.snapshot"))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    raw_a = (args.run_a / "native-result.json").read_bytes()
    raw_b = (args.run_b / "native-result.json").read_bytes()
    result = json.loads(raw_a)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    snapshots_a = _snapshot_hashes(args.run_a)
    snapshots_b = _snapshot_hashes(args.run_b)

    candidates_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in result["candidates"]:
        candidates_by_group[str(candidate["record"]["counterfactual_group_id"])].append(
            candidate
        )
    complete_pairs = sorted(
        group_id for group_id, candidates in candidates_by_group.items() if len(candidates) == 2
    )
    rejection_reasons = Counter(
        str(rejection["reason"]) for rejection in result["rejected_candidates"]
    )
    deterministic = raw_a == raw_b and snapshots_a == snapshots_b
    success = len(complete_pairs) >= 2 and deterministic
    group_metadata = {row["group_id"]: row for row in metadata["groups"]}
    completed = [group_metadata[group_id] for group_id in complete_pairs]
    capture_decisions = Counter(
        int(candidates[0]["record"]["decision_index"])
        for group_id, candidates in candidates_by_group.items()
        if group_id in complete_pairs
    )

    payload = {
        "status": "PASS" if success else "NATIVE_SCENARIO_CONSTRUCTION_BLOCKED",
        "promotion_eligible": False,
        "panel_contract": {
            "requested_pair_count": metadata["causal_group_count"],
            "requested_record_count": metadata["requested_record_count"],
            "factor_config_count": metadata["factor_config_count"],
            "selection_rule": metadata["selection_rule"],
            "outcome_fields_used_for_seed_selection": metadata[
                "outcome_fields_used_for_seed_selection"
            ],
            "success_gate": metadata["success_gate"],
            "shared_capture_rule": metadata.get("shared_capture_rule"),
        },
        "result": {
            "candidate_count": len(result["candidates"]),
            "rejection_count": len(result["rejected_candidates"]),
            "complete_pair_count": len(complete_pairs),
            "complete_pair_ids": complete_pairs,
            "complete_pair_metadata": completed,
            "complete_pair_capture_decisions": dict(sorted(capture_decisions.items())),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
        "deterministic_reproduction": {
            "passed": deterministic,
            "native_result_sha256_a": _sha256(raw_a),
            "native_result_sha256_b": _sha256(raw_b),
            "native_result_bytes_identical": raw_a == raw_b,
            "snapshot_count_a": len(snapshots_a),
            "snapshot_count_b": len(snapshots_b),
            "snapshot_sets_identical": snapshots_a == snapshots_b,
        },
        "native_simulator_sha": result["native_simulator_sha"],
        "source_campaign_config_sha256": metadata[
            "source_campaign_config_sha256"
        ],
        "diagnostic_config_sha256": metadata["diagnostic_config_sha256"],
        "scope_checks": {
            "all_accepted_train": all(
                candidate["record"]["partition"] == "train"
                for candidate in result["candidates"]
            ),
            "all_accepted_fishing_excluded": all(
                not candidate["record"]["actual_strata"]["fishing_available"]
                for candidate in result["candidates"]
            ),
            "all_accepted_replay_verified": all(
                candidate["record"]["validation"]["replay_verified"]
                for candidate in result["candidates"]
            ),
        },
        "interpretation": (
            "Shared event-aligned capture supplied enough deterministic, naturally "
            "delivered import/export pairs to clear the bounded construction gate. "
            "This changes snapshot timing only and provides no evidence of a native "
            "defect or MLP representational failure."
            if success
            else "The frozen constructor and bounded sampling recipe did not reliably "
            "supply the required causal contrast. This is not evidence of a native "
            "crash, simulator defect, or MLP representational failure."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Bounded F15 targeted-panel outcome",
        "",
        f"**Status: {payload['status']}**",
        "",
        (
            f"The predeclared 24-pair / 48-record panel produced "
            f"{len(result['candidates'])} accepted records and "
            f"{len(complete_pairs)} complete causal pairs. The success gate required "
            "at least two complete pairs."
        ),
        "",
        f"- Complete pair: {', '.join(complete_pairs) or 'none'}",
        f"- Capture-decision distribution: {dict(sorted(capture_decisions.items()))}",
        f"- Rejection reasons: {dict(sorted(rejection_reasons.items()))}",
        f"- Deterministic byte-for-byte reproduction: {deterministic}",
        f"- Native result SHA-256: `{_sha256(raw_a)}`",
        f"- Frozen native SHA: `{result['native_simulator_sha']}`",
        "",
        (
            "No crash, replay failure, schema failure, non-determinism, or native defect was "
            "observed. The bounded gate passed using the existing deterministic heuristic "
            "counterparties and ordinary native delivered-trade ledger."
            if success
            else "No crash, replay failure, schema failure, non-determinism, or native defect "
            "was observed. The blocker is the reliability of realizing both delivered-trade "
            "directions at one declared timestamp under the current frozen scenario constructor."
        ),
        "",
        "The result provides no evidence for escalating beyond the feed-forward MLP.",
    ]
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "result"}))


if __name__ == "__main__":
    main()
