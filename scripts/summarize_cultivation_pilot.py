"""Reduce full cultivation-pilot traces to compact, reviewable Git evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

EVALUATION_METRICS = (
    "discounted_scaled_return",
    "raw_prior_cohort_welfare",
    "final_qol",
    "final_population",
    "final_food",
    "final_water",
    "final_clothing",
    "final_housing",
    "final_field_q",
    "final_retained_field_capital",
    "final_field_operation",
    "final_maintenance_worker_years",
    "final_food_reserve_years",
    "final_hand_tools_per_capita",
    "final_farm_tools_per_capita",
    "final_materials_per_capita",
    "delivered_trade",
    "delivered_food_imports",
    "signed_migration",
    "fission_events",
    "requested_field_operation",
    "accepted_field_operation",
    "realized_field_operation",
    "requested_development_budget",
    "accepted_development_budget",
    "realized_development_budget",
    "intervention_transition_rate",
    "intervention_mean_count",
    "requested_to_accepted_mean_l2",
    "accepted_to_realized_mean_l2",
    "action_saturation_rate",
)

BEHAVIOR_SLICES = (
    "low-q",
    "high-q",
    "high-maintenance",
    "trade-dependent",
    "isolated",
    "immigration",
    "emigration",
    "fission-opportunity",
    "young-memory",
    "mature-memory",
    "primitive-frontier",
    "plough-frontier",
)

NORMALIZATION_SLICES = (
    "train-fit",
    "development",
    "heldout",
    "low-q",
    "high-q",
    "trade-dependent",
    "isolated",
    "immigration",
    "emigration",
    "pre-fission",
    "post-fission",
    "young-memory",
    "mature-memory",
    "plough-frontier",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _edge_mean(rows: list[dict[str, Any]], key: str, *, first: bool) -> float:
    count = max(1, math.floor(len(rows) * 0.2))
    selected = rows[:count] if first else rows[-count:]
    return sum(float(row[key]) for row in selected) / len(selected)


def _evaluation_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "episodes": summary["episodes"],
        "survival_rate": summary["survival_rate"],
        "contract_failure_count": summary["contract_failure_count"],
        **{
            name: {
                statistic: summary[name][statistic]
                for statistic in ("min", "mean", "p50", "p95", "max")
            }
            for name in EVALUATION_METRICS
        },
    }


def _behavior_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in BEHAVIOR_SLICES:
        row = evaluation["behavior_slices"].get(name)
        if row is None:
            continue
        result[name] = {
            "episodes": row["episodes"],
            "discounted_scaled_return_mean": row["discounted_scaled_return"]["mean"],
            "requested_field_operation_mean": row["requested_field_operation"]["mean"],
            "requested_development_budget_mean": row["requested_development_budget"]["mean"],
            "final_food_reserve_years_mean": row["final_food_reserve_years"]["mean"],
            "delivered_trade_mean": row["delivered_trade"]["mean"],
            "signed_migration_mean": row["signed_migration"]["mean"],
            "fission_events_mean": row["fission_events"]["mean"],
            "intervention_transition_rate_mean": row["intervention_transition_rate"]["mean"],
            "action_saturation_rate_mean": row["action_saturation_rate"]["mean"],
        }
    return result


def _slim_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "update": row["update"],
            "discounted_scaled_return_mean": row["discounted_scaled_return_mean"],
            "requested_field_operation": row["actions"]["requested"]["field_operation"],
            "realized_field_operation": row["actions"]["realized"]["field_operation"],
            "requested_development_budget": row["actions"]["requested"]["development_budget"],
            "realized_development_budget": row["actions"]["realized"]["development_budget"],
            "intervention_transition_rate": row["intervention_transition_rate"],
            "intervention_mean_count": row["intervention_mean_count"],
            "requested_to_accepted_mean_l2": row["requested_to_accepted_mean_l2"],
            "normalizer_saturation_rate": row["normalizer_saturation_rate"],
            "post_update_approx_kl": row["ppo"]["post_update_approx_kl"],
            "post_update_clip_fraction": row["ppo"]["post_update_clip_fraction"],
            "post_update_value_mse": row["ppo"]["post_update_value_mse"],
            "target_kl_early_stop": bool(row["ppo"]["target_kl_early_stop"]),
        }
        for row in rows
    ]


def _seed_summary(seed: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    curve = seed["training"]["learning_curve"]
    selected = next(row for row in curve if row["update"] == seed["selected_update"])
    early_rate = _edge_mean(curve, "intervention_transition_rate", first=True)
    late_rate = _edge_mean(curve, "intervention_transition_rate", first=False)
    early_count = _edge_mean(curve, "intervention_mean_count", first=True)
    late_count = _edge_mean(curve, "intervention_mean_count", first=False)
    early_l2 = _edge_mean(curve, "requested_to_accepted_mean_l2", first=True)
    late_l2 = _edge_mean(curve, "requested_to_accepted_mean_l2", first=False)
    intervention_checks = {
        "rate_not_escalated": late_rate <= early_rate + 0.25,
        "count_not_escalated": late_count <= early_count + 0.25,
        "distance_not_escalated": late_l2 <= early_l2 + 0.05,
    }
    values = [float(row["ppo"]["post_update_value_mse"]) for row in curve]
    sorted_values = sorted(values)
    evaluations: dict[str, Any] = {}
    for split in ("train_probe", "development", "heldout"):
        evaluation = seed["evaluation"][split]
        evaluations[split] = {
            "summary": _evaluation_summary(evaluation["summary"]),
            "self_sufficiency": evaluation["self_sufficiency"],
            "behavior_slices": _behavior_summary(evaluation),
        }
    return {
        "seed": seed["seed"],
        "status": seed["status"],
        "selected_update": seed["selected_update"],
        "gate_failures": [name for name, passed in seed["gates"].items() if not passed],
        "gates": seed["gates"],
        "training": {
            "transitions": seed["training"]["transitions"],
            "visitation": seed["training"]["visitation"],
            "contract_totals": seed["training"]["contract_totals"],
            "inactive_head_gradient_max": seed["training"]["inactive_head_gradient_max"],
            "per_head_gradient_max": seed["training"]["per_head_gradient_max"],
            "movement_head_gradient_max": seed["training"]["movement_head_gradient_max"],
            "maximum_post_update_approx_kl": seed["training"]["maximum_post_update_approx_kl"],
            "mean_post_update_clip_fraction": seed["training"]["mean_post_update_clip_fraction"],
            "target_kl_early_stop_count": sum(
                bool(row["ppo"]["target_kl_early_stop"]) for row in curve
            ),
            "critic_post_update_mse": {
                "selected": selected["ppo"]["post_update_value_mse"],
                "mean": sum(values) / len(values),
                "median": sorted_values[len(sorted_values) // 2],
                "min": min(values),
                "max": max(values),
            },
            "intervention_trend": {
                "early_transition_rate": early_rate,
                "late_transition_rate": late_rate,
                "early_count_per_transition": early_count,
                "late_count_per_transition": late_count,
                "early_requested_to_accepted_l2": early_l2,
                "late_requested_to_accepted_l2": late_l2,
                "checks": intervention_checks,
                "not_pathological": all(intervention_checks.values()),
            },
            "timing": seed["training"]["timing"],
            "learning_curve": _slim_curve(curve),
        },
        "evaluation": evaluations,
        "baseline_comparison": seed["baseline_comparison"],
        "checkpoint": seed["checkpoint"],
        "raw_seed_file": {
            "path": str(raw_path),
            "sha256": _sha256(raw_path),
            "bytes": raw_path.stat().st_size,
            "tracked_by_git": False,
        },
    }


def _normalization_summary(payload: dict[str, Any]) -> dict[str, Any]:
    slices: dict[str, Any] = {}
    for slice_name in NORMALIZATION_SLICES:
        rows: list[dict[str, Any]] = []
        for feature in payload["features"]:
            values = feature["slices"].get(slice_name)
            if values is None or values.get("saturation_rate") is None:
                continue
            rate = float(values["saturation_rate"])
            rows.append(
                {
                    "feature": feature["name"],
                    "rate": rate,
                    "presence_rate": values["presence_rate"],
                    "p99": values["p99"],
                    "max": values["max"],
                    "fitted_c_i": feature["fitted_c_i"],
                }
            )
        saturated = [row for row in rows if row["rate"] > 0]
        top = sorted(saturated, key=lambda row: row["rate"], reverse=True)[:5]
        slices[slice_name] = {
            "sample_count": payload["sample_counts"].get(slice_name),
            "applicable_feature_count": len(rows),
            "features_with_saturation": len(saturated),
            "maximum_saturation_rate": top[0]["rate"] if top else 0.0,
            "top_saturated_features": top,
        }
    sparse_event_blocker = (
        slices["trade-dependent"]["maximum_saturation_rate"] >= 0.95
        and max(
            slices["immigration"]["maximum_saturation_rate"],
            slices["emigration"]["maximum_saturation_rate"],
        )
        >= 0.50
    )
    return {
        "status": payload["status"],
        "fit_partition": payload["fit_partition"],
        "fit_observation_count": payload["fit_observation_count"],
        "fit_source_mix": payload["fit_source_mix"],
        "heldout_or_development_used_for_fit": payload["heldout_or_development_used_for_fit"],
        "partition_manifest_sha256": payload["partition_manifest_sha256"],
        "saturation_definition": payload["saturation_definition"],
        "slices": slices,
        "sparse_event_scale_saturation_detected": sparse_event_blocker,
    }


def summarize(
    campaign_path: Path,
    normalizer_diagnostics_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    campaign = _read(campaign_path)
    normalizer = _read(normalizer_diagnostics_path)
    seed_reports = [
        _seed_summary(
            seed,
            campaign_path.parent / f"seed-{seed['seed']}.json",
        )
        for seed in campaign["seeds"]
    ]
    normalization_summary = _normalization_summary(normalizer)
    failed_seed_gates = {
        str(seed["seed"]): seed["gate_failures"] for seed in seed_reports if seed["gate_failures"]
    }
    pathological_interventions = [
        seed["seed"]
        for seed in seed_reports
        if not seed["training"]["intervention_trend"]["not_pathological"]
    ]
    trajectory = {
        key: value
        for key, value in campaign["cultivation_intensity_trajectory"].items()
        if key != "updates"
    }
    result = {
        "status": campaign["status"],
        "recommendation": "NO_GO_NEXT_CURRICULUM",
        "raw_campaign": {
            "path": str(campaign_path),
            "file_sha256": _sha256(campaign_path),
            "internal_evidence_sha256": campaign["evidence_sha256"],
            "bytes": campaign_path.stat().st_size,
            "tracked_by_git": False,
        },
        "identities": campaign["identities"],
        "configuration": campaign["configuration"],
        "bank": campaign["bank"],
        "promotion_gates": campaign["promotion_gates"],
        "posthoc_assessment": {
            "failed_seed_gates": failed_seed_gates,
            "pathological_intervention_trend_seeds": pathological_interventions,
            "sparse_event_normalization_blocker": normalization_summary[
                "sparse_event_scale_saturation_detected"
            ],
            "failure_attribution": [
                "optimizer-instability-seed-1901-single-kl-breach",
                "seed-instability-and-option-a-underperformance-seed-2101",
                "scenario-coverage-or-transform-semantics-for-sparse-trade-migration-features",
            ],
            "feed_forward_representational_failure_demonstrated": False,
            "architecture_escalation_justified": False,
            "heldout_must_not_be_reused_for_posthoc_tuning": True,
        },
        "cultivation_intensity_trajectory": trajectory,
        "option_a_baseline": {
            split: _evaluation_summary(campaign["option_a_baseline"][split])
            for split in ("train_probe", "development", "heldout")
        },
        "normalization": normalization_summary,
        "seeds": seed_reports,
    }
    result["summary_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    _write(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("evidence/cultivation-ppo-pilot-v1/campaign-summary.json"),
    )
    parser.add_argument(
        "--normalizer-diagnostics",
        type=Path,
        default=Path("evidence/cultivation-pilot-v1/final-normalization-diagnostics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/cultivation-ppo-pilot-v1/summary.json"),
    )
    args = parser.parse_args()
    result = summarize(args.campaign, args.normalizer_diagnostics, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "recommendation": result["recommendation"],
                "summary_sha256": result["summary_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
