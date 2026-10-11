"""Human- and machine-readable snapshot-bank coverage summaries."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from anthropocene_rl.datasets.models import SnapshotBankManifest


def coverage_report(manifest: SnapshotBankManifest) -> dict[str, Any]:
    scenario_partition: dict[str, Counter[str]] = defaultdict(Counter)
    masks: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    classifications: Counter[str] = Counter()
    for record in manifest.records:
        scenario_partition[record.scenario_family][record.partition.value] += 1
        mask = record.actual_strata.structural_masks
        signature = "".join(
            "1" if value else "0"
            for value in (
                mask.field_retention,
                mask.herd_retention,
                mask.field_development,
                mask.herd_development,
            )
        )
        masks[signature] += 1
        tags.update(record.actual_strata.tags)
        classifications[record.classification.value] += 1
    return {
        "bank_id": manifest.bank_id,
        "manifest_sha256": manifest.sha256(),
        "snapshot_count": len(manifest.records),
        "partition_counts": dict(
            sorted(Counter(record.partition.value for record in manifest.records).items())
        ),
        "scenario_family_by_partition": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(scenario_partition.items())
        },
        "classification_counts": dict(sorted(classifications.items())),
        "structural_mask_signatures": dict(sorted(masks.items())),
        "measured_tag_counts": dict(sorted(tags.items())),
        "fishing_ground_counts": dict(
            sorted(
                Counter(
                    str(record.actual_strata.fishing_ground_count)
                    for record in manifest.records
                ).items()
            )
        ),
    }


def save_coverage_report(
    manifest: SnapshotBankManifest,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    report = coverage_report(manifest)
    json_target = Path(json_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Snapshot-bank coverage: {manifest.bank_id}",
        "",
        f"Snapshots: **{report['snapshot_count']}**",
        "",
        "## Partitions",
        "",
        "| Partition | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {partition} | {count} |"
        for partition, count in report["partition_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Scenario families",
            "",
            "| Family | Train | Development | Heldout | Total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family, counts in report["scenario_family_by_partition"].items():
        total = sum(counts.values())
        lines.append(
            f"| {family} | {counts.get('train', 0)} | "
            f"{counts.get('development', 0)} | {counts.get('heldout', 0)} | {total} |"
        )
    lines.extend(
        [
            "",
            "## Structural masks",
            "",
            "Signature order: field retention, herd retention, field Development, "
            "herd Development.",
            "",
            "| Signature | Count |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{signature}` | {count} |"
        for signature, count in report["structural_mask_signatures"].items()
    )
    lines.extend(
        [
            "",
            "## Fishing rows",
            "",
            "| Observed grounds | Count |",
            "|---:|---:|",
        ]
    )
    lines.extend(
        f"| {grounds} | {count} |"
        for grounds, count in report["fishing_ground_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Classifications",
            "",
            "| Classification | Count |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| {classification} | {count} |"
        for classification, count in report["classification_counts"].items()
    )
    lines.extend(["", "## Measured strata", "", "| Tag | Count |", "|---|---:|"])
    lines.extend(
        f"| {tag} | {count} |" for tag, count in report["measured_tag_counts"].items()
    )
    markdown_target = Path(markdown_path)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
