from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from anthropocene_rl.datasets.generation_config import SnapshotBankGenerationConfig
from anthropocene_rl.observations.schema import (
    EncodedObservation,
    broad_domain_scope_violations,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the pre-promotion attrition and identity ledger for a broad bank."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--native-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _group_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record["scenario_family"]),
        str(record.get("counterfactual_group_id") or record["lineage_id"]),
    )


def _scope_violations(candidate: dict[str, Any]) -> tuple[str, ...]:
    violations: set[str] = set()
    for sample in candidate["calibration_observations"]:
        observation = EncodedObservation.from_native(sample["tensor"])
        violations.update(broad_domain_scope_violations(observation))
    return tuple(sorted(violations))


def _duplicates(records: list[dict[str, Any]], key: str) -> dict[str, list[str]]:
    by_value: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_value[str(record[key]).lower()].append(str(record["snapshot_id"]))
    return {
        value: sorted(snapshot_ids)
        for value, snapshot_ids in sorted(by_value.items())
        if len(snapshot_ids) > 1
    }


def analyze(
    config: SnapshotBankGenerationConfig,
    result: dict[str, Any],
) -> dict[str, Any]:
    candidates = list(result["candidates"])
    rejected = list(result["rejected_candidates"])
    requested_count = sum(len(plan.snapshot_steps) for plan in config.trajectories)
    if requested_count != len(candidates) + len(rejected):
        raise ValueError("Native candidate accounting does not close")

    resolved, _ = config.resolve_partitions()
    expected: Counter[tuple[str, str]] = Counter()
    requested_by_family_partition: Counter[tuple[str, str]] = Counter()
    for plan in resolved:
        group = plan.counterfactual_group_id or plan.lineage_id
        expected[(plan.scenario_family, group)] += len(plan.snapshot_steps)
        requested_by_family_partition[(plan.scenario_family, plan.partition.value)] += len(
            plan.snapshot_steps
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    candidate_by_family_partition: Counter[tuple[str, str]] = Counter()
    for candidate in candidates:
        record = candidate["record"]
        grouped[_group_key(record)].append(candidate)
        candidate_by_family_partition[
            (str(record["scenario_family"]), str(record["partition"]))
        ] += 1

    rejection_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in rejected:
        key = (
            str(item["scenario_family"]),
            str(item.get("counterfactual_group_id") or item["lineage_id"]),
        )
        rejection_by_group[key].append(item)

    group_rows: list[dict[str, Any]] = []
    preliminarily_eligible: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key in sorted(expected):
        group_candidates = grouped.get(key, [])
        records = [candidate["record"] for candidate in group_candidates]
        partitions = sorted({str(record["partition"]) for record in records})
        scopes = sorted(
            {
                violation
                for candidate in group_candidates
                for violation in _scope_violations(candidate)
            }
        )
        content_duplicates = _duplicates(records, "content_sha256")
        fingerprint_duplicates = _duplicates(records, "native_fingerprint")
        complete = len(group_candidates) == expected[key]
        within_group_unique = not content_duplicates and not fingerprint_duplicates
        scope_safe = not scopes
        if complete and within_group_unique and scope_safe:
            preliminarily_eligible[key] = group_candidates
        group_rows.append(
            {
                "family": key[0],
                "group_id": key[1],
                "expected_records": expected[key],
                "candidate_records": len(group_candidates),
                "candidate_snapshot_ids": sorted(
                    str(record["snapshot_id"]) for record in records
                ),
                "partitions": partitions,
                "complete": complete,
                "scope_violations": scopes,
                "duplicate_content_within_group": content_duplicates,
                "duplicate_fingerprints_within_group": fingerprint_duplicates,
                "native_rejections": [
                    {
                        "candidate_id": str(item["candidate_id"]),
                        "reason_code": str(item["reason_code"]),
                        "reason": str(item["reason"]),
                    }
                    for item in rejection_by_group.get(key, [])
                ],
            }
        )

    content_owners: dict[str, set[tuple[str, str]]] = defaultdict(set)
    fingerprint_owners: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for key, group_candidates in preliminarily_eligible.items():
        for candidate in group_candidates:
            record = candidate["record"]
            content_owners[str(record["content_sha256"]).lower()].add(key)
            fingerprint_owners[str(record["native_fingerprint"]).lower()].add(key)
    cross_group_content = {
        value: sorted(f"{family}:{group}" for family, group in owners)
        for value, owners in sorted(content_owners.items())
        if len(owners) > 1
    }
    cross_group_fingerprints = {
        value: sorted(f"{family}:{group}" for family, group in owners)
        for value, owners in sorted(fingerprint_owners.items())
        if len(owners) > 1
    }
    cross_conflict_groups = {
        key
        for owners in (*cross_group_content.values(), *cross_group_fingerprints.values())
        for owner in owners
        for key in [(owner.split(":", 1)[0], owner.split(":", 1)[1])]
    }
    eligible = {
        key: group_candidates
        for key, group_candidates in preliminarily_eligible.items()
        if key not in cross_conflict_groups
    }

    row_by_key = {(row["family"], row["group_id"]): row for row in group_rows}
    for key, row in row_by_key.items():
        row["cross_group_identity_conflict"] = key in cross_conflict_groups
        row["eligible"] = key in eligible

    complete_by_family_partition: Counter[tuple[str, str]] = Counter()
    scope_safe_by_family_partition: Counter[tuple[str, str]] = Counter()
    within_unique_by_family_partition: Counter[tuple[str, str]] = Counter()
    eligible_by_family_partition: Counter[tuple[str, str]] = Counter()
    for key, group_candidates in grouped.items():
        records = [candidate["record"] for candidate in group_candidates]
        partition = str(records[0]["partition"])
        count = len(records)
        row = row_by_key[key]
        if row["complete"]:
            complete_by_family_partition[(key[0], partition)] += count
        if row["complete"] and not row["scope_violations"]:
            scope_safe_by_family_partition[(key[0], partition)] += count
        if (
            row["complete"]
            and not row["scope_violations"]
            and not row["duplicate_content_within_group"]
            and not row["duplicate_fingerprints_within_group"]
        ):
            within_unique_by_family_partition[(key[0], partition)] += count
        if key in eligible:
            eligible_by_family_partition[(key[0], partition)] += count

    families = sorted({plan.scenario_family for plan in config.trajectories})
    partitions = ("train", "development", "heldout")
    family_ledger = {
        family: {
            partition: {
                "requested": requested_by_family_partition[(family, partition)],
                "native_candidates": candidate_by_family_partition[(family, partition)],
                "complete_group_records": complete_by_family_partition[(family, partition)],
                "scope_safe_records": scope_safe_by_family_partition[(family, partition)],
                "within_group_unique_records": within_unique_by_family_partition[
                    (family, partition)
                ],
                "fully_eligible_records": eligible_by_family_partition[(family, partition)],
            }
            for partition in partitions
        }
        for family in families
    }

    f15_contrasts: Counter[tuple[str, str]] = Counter()
    for key, group_candidates in eligible.items():
        if key[0] != "F15":
            continue
        variants = tuple(
            sorted(
                str(candidate["record"]["selection_metadata"]["causal_variant"])
                for candidate in group_candidates
            )
        )
        partition = str(group_candidates[0]["record"]["partition"])
        f15_contrasts[(" + ".join(variants), partition)] += 1

    f16_groups = [row for row in group_rows if row["family"] == "F16"]
    return {
        "status": "PREFLIGHT_COMPLETE",
        "generator_config_sha256": config.canonical_sha256(),
        "native_result_identity": {
            key: result.get(key)
            for key in (
                "native_simulator_sha",
                "native_build_hash",
                "configuration_hash",
                "observation_schema_hash",
                "generator_code_sha256",
            )
        },
        "accounting": {
            "requested_records": requested_count,
            "native_candidates": len(candidates),
            "native_rejections": len(rejected),
            "closed": requested_count == len(candidates) + len(rejected),
        },
        "native_rejection_reason_counts": dict(
            sorted(Counter(str(item["reason_code"]) for item in rejected).items())
        ),
        "native_rejection_family_counts": dict(
            sorted(Counter(str(item["scenario_family"]) for item in rejected).items())
        ),
        "family_partition_attrition": family_ledger,
        "identity": {
            "cross_group_duplicate_content": cross_group_content,
            "cross_group_duplicate_fingerprints": cross_group_fingerprints,
            "cross_group_conflict_group_count": len(cross_conflict_groups),
        },
        "f15_complete_eligible_pairs_by_contrast_partition": {
            f"{contrast}|{partition}": count
            for (contrast, partition), count in sorted(f15_contrasts.items())
        },
        "f16_group_diagnosis": f16_groups,
        "groups": group_rows,
    }


def main() -> int:
    args = _parser().parse_args()
    config = SnapshotBankGenerationConfig.load(args.config)
    result = json.loads(args.native_result.read_text(encoding="utf-8"))
    payload = analyze(config, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: payload[key] for key in ("status", "accounting")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
