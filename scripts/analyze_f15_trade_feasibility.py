"""Explain strict F15 import/export construction outcomes pair by pair."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from anthropocene_rl.observations.features import FEATURE_NAMES

FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
TRACE_FEATURES = (
    "core.population",
    "core.qol",
    "core.food-needs-met",
    "core.food-reserve-years",
    "core.realized-migration-per-capita-year",
    "core.realized-trade-utility-per-capita-year",
    "fields.current-operation",
    "fields.current-intensity",
)


def _trajectory_id(candidate: dict[str, Any]) -> str:
    return str(candidate["record"]["selection_metadata"]["trajectory_id"])


def _contacts(tensor: dict[str, Any]) -> list[dict[str, float | bool]]:
    values = tensor["values"]
    present = tensor["present"]
    rows = []
    for row in range(8):
        offset = 400 + 8 * row
        if not present[offset]:
            continue
        rows.append(
            {
                "distance": values[offset],
                "age_years": values[offset + 1],
                "affinity": values[offset + 2],
                "recent_trade_utility_per_capita_year": values[offset + 3],
                "recent_trade_flow_per_capita_year": values[offset + 4],
                "recent_migration_per_capita_year": values[offset + 5],
                "kin": bool(values[offset + 6]),
                "currently_visible": bool(values[offset + 7]),
            }
        )
    return rows


def _trace(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    trace = []
    for decision, sample in enumerate(candidate["calibration_observations"]):
        tensor = sample["tensor"]
        contacts = _contacts(tensor)
        row = {
            "decision": decision,
            "source": sample["source"],
            "contact_count": len(contacts),
            "visible_contact_count": sum(item["currently_visible"] for item in contacts),
            "contact_trade_flow_sum": sum(
                float(item["recent_trade_flow_per_capita_year"]) for item in contacts
            ),
        }
        row.update(
            {
                feature: tensor["values"][FEATURE_INDEX[feature]]
                for feature in TRACE_FEATURES
            }
        )
        trace.append(row)
    return trace


def _site_metrics(site_evidence: str) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in re.findall(r"([a-z_]+)=([0-9.]+)", site_evidence)
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _outcome(sibling: dict[str, Any]) -> str:
    return "PASS" if sibling["strict_pass"] else str(sibling["failure_mode"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-result", type=Path, required=True)
    parser.add_argument("--observational-result", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    strict = json.loads(args.strict_result.read_text(encoding="utf-8"))
    observational = json.loads(args.observational_result.read_text(encoding="utf-8"))
    strict_candidates = {_trajectory_id(item): item for item in strict["candidates"]}
    strict_rejections = {
        str(item["candidate_id"]): item for item in strict["rejected_candidates"]
    }
    observed = {_trajectory_id(item): item for item in observational["candidates"]}
    if set(observed) != set(strict_candidates) | set(strict_rejections):
        raise ValueError("Strict and observational trajectory identities do not match")

    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_modes: Counter[str] = Counter()
    for trajectory_id, candidate in sorted(observed.items()):
        record = candidate["record"]
        strata = record["actual_strata"]
        metadata = record["selection_metadata"]
        strict_pass = trajectory_id in strict_candidates
        contacts = _contacts(candidate["calibration_observations"][0]["tensor"])
        if strict_pass:
            mode = "strict-pass"
        elif strata["delivered_trade_total"] <= 1e-9:
            mode = (
                "no-observed-contacts"
                if not contacts
                else "contacts-present-but-no-delivery"
            )
        elif metadata["causal_variant"] == "mild-import" and strata[
            "delivered_food_import"
        ] <= 1e-9:
            mode = "delivered-trade-without-food-import"
        elif metadata["causal_variant"] == "surplus-export" and strata[
            "delivered_food_export"
        ] <= 1e-9:
            mode = "delivered-trade-without-food-export"
        else:
            mode = "other-requested-stratum-miss"
        failure_modes[mode] += 1
        pairs[str(record["counterfactual_group_id"])].append(
            {
                "trajectory_id": trajectory_id,
                "variant": metadata["causal_variant"],
                "world_seed": record["world_seed"],
                "partition": record["partition"],
                "strict_pass": strict_pass,
                "strict_rejection": strict_rejections.get(trajectory_id),
                "failure_mode": mode,
                "requested": {
                    "field_q": metadata["requested_field_q"],
                    "reserve_intervals": metadata["requested_reserve_intervals"],
                    "site_support_quantile": metadata["requested_site_support_quantile"],
                    "site_selection": _site_metrics(metadata["site_selection_evidence"]),
                },
                "actual": {
                    "population": strata["population"],
                    "field_q": strata["field_q"],
                    "food_reserve_years": strata["food_reserve_years"],
                    "qol": strata["current_qol"],
                    "delivered_trade_total": strata["delivered_trade_total"],
                    "delivered_import_total": strata["delivered_import_total"],
                    "delivered_export_total": strata["delivered_export_total"],
                    "delivered_food_import": strata["delivered_food_import"],
                    "delivered_food_export": strata["delivered_food_export"],
                    "signed_migration": strata["signed_migration"],
                    "tags": strata["tags"],
                    "contact_count": len(contacts),
                    "visible_contact_count": sum(
                        item["currently_visible"] for item in contacts
                    ),
                    "contacts": contacts,
                },
                "calibration_trace": _trace(candidate),
            }
        )

    pair_rows = []
    for group_id, siblings in sorted(pairs.items()):
        if len(siblings) != 2:
            raise ValueError(f"Incomplete observational pair: {group_id}")
        pair_rows.append(
            {
                "group_id": group_id,
                "world_seed": siblings[0]["world_seed"],
                "partition": siblings[0]["partition"],
                "complete_strict_pair": all(item["strict_pass"] for item in siblings),
                "siblings": sorted(siblings, key=lambda item: item["variant"]),
            }
        )

    complete = [row for row in pair_rows if row["complete_strict_pair"]]
    incomplete = [row for row in pair_rows if not row["complete_strict_pair"]]
    complete_siblings = [item for row in complete for item in row["siblings"]]
    incomplete_siblings = [item for row in incomplete for item in row["siblings"]]
    comparison = {}
    comparison_groups = (
        ("complete_pairs", complete_siblings),
        ("incomplete_pairs", incomplete_siblings),
    )
    for label, rows in comparison_groups:
        flattened = [
            {
                **item["requested"],
                **item["actual"],
                "site_viability": item["requested"]["site_selection"].get("viability"),
                "site_candidates": item["requested"]["site_selection"].get("candidates"),
            }
            for item in rows
        ]
        comparison[label] = {
            key: _mean(flattened, key)
            for key in (
                "field_q",
                "site_support_quantile",
                "site_viability",
                "site_candidates",
                "population",
                "contact_count",
                "visible_contact_count",
                "delivered_trade_total",
            )
        }

    payload = {
        "status": "PASS_DIAGNOSTIC_NOT_PROMOTABLE",
        "strict_candidate_count": len(strict_candidates),
        "strict_rejection_count": len(strict_rejections),
        "pair_count": len(pair_rows),
        "complete_pair_count": len(complete),
        "complete_pair_ids": [row["group_id"] for row in complete],
        "complete_pair_partition_counts": dict(
            sorted(Counter(row["partition"] for row in complete).items())
        ),
        "failure_modes": dict(sorted(failure_modes.items())),
        "aggregate_comparison": comparison,
        "pairs": pair_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# F15 mild-import / surplus-export construction diagnostic",
        "",
        "This is a non-promotable scenario-construction diagnostic. The strict run "
        "kept all campaign gates; the companion observational run removed only "
        "trade acceptance predicates so failed worlds retained their reset and "
        "calibration traces.",
        "",
        f"- Strict candidates: {len(strict_candidates)} / {len(observed)}",
        f"- Complete causal pairs: {len(complete)} / {len(pair_rows)}",
        "- Complete-pair partitions: "
        f"{dict(sorted(Counter(row['partition'] for row in complete).items()))}",
        f"- Complete pair IDs: {', '.join(row['group_id'] for row in complete)}",
        f"- Failure modes: {dict(sorted(failure_modes.items()))}",
        "",
        "## Pair outcomes",
        "",
        "| Group | Split | Mild import | Surplus export | Complete |",
        "|---|---|---:|---:|---:|",
    ]
    for row in pair_rows:
        siblings = {item["variant"]: item for item in row["siblings"]}
        lines.append(
            f"| {row['group_id']} | {row['partition']} | "
            f"{_outcome(siblings['mild-import'])} | "
            f"{_outcome(siblings['surplus-export'])} | "
            f"{'yes' if row['complete_strict_pair'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate comparison",
            "",
            "```json",
            json.dumps(comparison, indent=2, sort_keys=True),
            "```",
            "",
            "Full per-sibling physical measurements, contacts, rejection evidence, "
            "and six-observation calibration traces are in the JSON report.",
        ]
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
