"""Human- and machine-readable snapshot-bank coverage summaries."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
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
    measured_rows = [_measured_factor_row(record) for record in manifest.records]
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
        "world_count": len({record.world_seed for record in manifest.records}),
        "lineage_count": len({record.lineage_id for record in manifest.records}),
        "world_counts_by_partition": {
            partition: len(
                {
                    record.world_seed
                    for record in manifest.records
                    if record.partition.value == partition
                }
            )
            for partition in ("train", "development", "heldout")
        },
        "structural_mask_signatures": dict(sorted(masks.items())),
        "measured_tag_counts": dict(sorted(tags.items())),
        "measured_distributions": {
            "field_q": _distribution(
                [record.actual_strata.field_q for record in manifest.records]
            ),
            "population": _distribution(
                [float(record.actual_strata.population) for record in manifest.records]
            ),
            "food_reserve_years": _distribution(
                [record.actual_strata.food_reserve_years for record in manifest.records]
            ),
            "retained_field_capital": _distribution(
                [
                    record.actual_strata.retained_field_capital
                    for record in manifest.records
                ]
            ),
            "maintenance_worker_years": _distribution(
                [
                    record.actual_strata.maintenance_worker_years
                    for record in manifest.records
                ]
            ),
            "delivered_trade_total": _distribution(
                [
                    record.actual_strata.delivered_trade_total
                    for record in manifest.records
                ]
            ),
            "delivered_food_import": _distribution(
                [record.actual_strata.delivered_food_import for record in manifest.records]
            ),
            "delivered_food_export": _distribution(
                [record.actual_strata.delivered_food_export for record in manifest.records]
            ),
            "delivered_material_tool_import": _distribution(
                [
                    record.actual_strata.delivered_material_tool_import
                    for record in manifest.records
                ]
            ),
            "import_dependence": _distribution(
                [record.actual_strata.import_dependence for record in manifest.records]
            ),
            "signed_migration": _distribution(
                [record.actual_strata.signed_migration for record in manifest.records]
            ),
            "fission_pressure": _distribution(
                [record.actual_strata.fission_pressure for record in manifest.records]
            ),
            "fission_exit_opportunity": _distribution(
                [
                    record.actual_strata.fission_exit_opportunity
                    for record in manifest.records
                ]
            ),
            "memory_age_years": _distribution(
                [record.institutional_memory.age_years for record in manifest.records]
            ),
        },
        "measured_factor_counts": {
            factor: dict(sorted(Counter(row[factor] for row in measured_rows).items()))
            for factor in measured_rows[0]
        }
        if measured_rows
        else {},
        "measured_factor_pairwise_coverage": _pairwise_coverage(measured_rows),
        "fishing_ground_counts": dict(
            sorted(
                Counter(
                    str(record.actual_strata.fishing_ground_count)
                    for record in manifest.records
                ).items()
            )
        ),
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p5": None, "p50": None,
                "p95": None, "p99": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p5": quantile(0.05),
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": ordered[-1],
    }


def _measured_factor_row(record: Any) -> dict[str, str]:
    strata = record.actual_strata
    amenity = min(strata.clothing_coverage, strata.housing_coverage)
    tool = min(strata.hand_tools_per_capita, strata.farm_tools_per_capita)
    return {
        "family": record.scenario_family,
        "q_band": "low" if strata.field_q < 0.30 else (
            "moderate" if strata.field_q < 0.50 else "high"
        ),
        "population_band": "small" if strata.population < 150 else (
            "medium" if strata.population < 400 else "large"
        ),
        "reserve_band": "poor" if strata.food_reserve_years < 1 else (
            "moderate" if strata.food_reserve_years < 3 else "strong"
        ),
        "amenity_band": "poor" if amenity < 0.5 else (
            "moderate" if amenity < 0.8 else "strong"
        ),
        "tool_band": "poor" if tool < 0.25 else (
            "moderate" if tool < 0.75 else "strong"
        ),
        "trade_band": "none" if strata.delivered_trade_total <= 1e-9 else (
            "mild" if strata.import_dependence < 0.25 else "strong"
        ),
        "migration_sign": "negative" if strata.signed_migration < 0 else (
            "positive" if strata.signed_migration > 0 else "zero"
        ),
        "fission_state": "recent-split" if strata.fission_recent_split else (
            "candidate" if strata.fission_candidate_present else "none"
        ),
        "memory_maturity": "young" if record.institutional_memory.age_years < 20 else (
            "partial" if record.institutional_memory.age_years < 180 else "mature"
        ),
        "frontier": strata.field_frontier,
    }


def _pairwise_coverage(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"observed_pairs": 0, "possible_pairs": 0, "fraction": 1.0}
    factors = tuple(rows[0])
    levels = {factor: {row[factor] for row in rows} for factor in factors}
    observed = {
        (left, row[left], right, row[right])
        for left, right in combinations(factors, 2)
        for row in rows
    }
    possible = sum(
        len(levels[left]) * len(levels[right])
        for left, right in combinations(factors, 2)
    )
    return {
        "observed_pairs": len(observed),
        "possible_pairs": possible,
        "fraction": len(observed) / possible if possible else 1.0,
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
