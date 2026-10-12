"""Predeclared constant-action headroom audit for the cultivation pilot.

The audit performs no optimizer updates.  It evaluates a fixed 5x5 grid and
Option A on every TRAIN and DEVELOPMENT reset through a common five-decision
ceiling.  Three-decision physical return is the primary endpoint; the
five-decision endpoint is descriptive and cannot authorize PPO by itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from anthropocene_rl.datasets.blobs import sha256_file
from anthropocene_rl.datasets.models import Partition, SnapshotRecord
from anthropocene_rl.training.five_step_credit_audit import (
    AuditIntegrityError,
    _digest,
    _validate_scope_and_observation,
    _validated_step,
)
from anthropocene_rl.training.overfit import (
    _mapping,
    _pick,
    _plain_mapping,
    _write_json,
)
from anthropocene_rl.training.pilot import (
    CampaignDataset,
    CampaignSelection,
    _make_environment,
    _reset_request,
    _validate_campaign_reset,
)
from anthropocene_rl.training.returns import discounted_interval_reward, physical_gamma

GRID_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
GRID_ACTIONS = tuple((field, development) for field in GRID_VALUES for development in GRID_VALUES)
AUDIT_STEPS = 5
PRIMARY_STEPS = 3
HORIZON_YEARS = 400.0
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_261_012
OVERALL_EFFECT_FLOOR = 0.015
FAMILY_EFFECT_FLOOR = 0.02
OVERALL_CLUSTER_MINIMUM = 20
FAMILY_DEVELOPMENT_CLUSTER_MINIMUM = 5
FAMILY_BONFERRONI_TESTS = 7
NONQUALIFYING_FAMILY = "F15"
TIE_RELATIVE_TOLERANCE = 1e-12

ProgressCallback = Callable[[dict[str, Any]], None]


def run_constant_grid_headroom_audit(
    *,
    evidence_directory: str | Path,
    bridge_dll: str | Path,
    dotnet: str,
    manifest_path: str | Path,
    blob_root: str | Path,
    normalizer_path: str | Path,
    snapshot_store_config: str | Path,
    dataset_manifest: str | Path | None,
    dataset_store_config: str | Path,
    repository_root: str | Path,
    selected_policy_evidence: tuple[str | Path, ...] = (),
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the frozen TRAIN+DEVELOPMENT constant-grid headroom audit."""

    evidence_dir = Path(evidence_directory)
    bridge_path = Path(bridge_dll).resolve()
    dataset = CampaignDataset(
        manifest_path=manifest_path,
        blob_root=blob_root,
        normalizer_path=normalizer_path,
        snapshot_store_config=snapshot_store_config,
        dataset_manifest=dataset_manifest,
        dataset_store_config=dataset_store_config,
    )
    train = dataset.by_partition[Partition.TRAIN]
    development = dataset.by_partition[Partition.DEVELOPMENT]
    if len(train) != 210 or len(development) != 46:
        raise ValueError(
            "Expected the frozen 210 TRAIN / 46 DEVELOPMENT cohort, found "
            f"{len(train)} / {len(development)}"
        )
    identities = _audit_identities(
        repository_root=Path(repository_root),
        bridge_path=bridge_path,
        dataset=dataset,
        normalizer_path=Path(normalizer_path),
        selected_policy_evidence=tuple(Path(path) for path in selected_policy_evidence),
    )
    design = _design_contract()
    _write_json(evidence_dir / "design.json", design)

    raw_files: dict[str, str] = {}
    partition_rows: dict[Partition, list[dict[str, Any]]] = {}
    try:
        for partition, records in (
            (Partition.TRAIN, train),
            (Partition.DEVELOPMENT, development),
        ):
            rows = _run_partition(
                partition=partition,
                records=records,
                dataset=dataset,
                bridge_path=bridge_path,
                dotnet=dotnet,
                progress=progress,
            )
            partition_rows[partition] = rows
            raw_path = evidence_dir / f"{partition.value}-episodes.json"
            _write_json(raw_path, {"partition": partition.value, "episodes": rows})
            raw_files[raw_path.name] = sha256_file(raw_path)[0]
    except AuditIntegrityError as error:
        failure = {
            "status": "STOP_INTEGRITY_FAILURE",
            "scope": "constant-grid-headroom-audit",
            "identities": identities,
            "design": design,
            "failure": error.payload,
            "partitions_completed": [partition.value for partition in partition_rows],
            "heldout_read": False,
        }
        failure["evidence_sha256"] = _digest(failure)
        _write_json(evidence_dir / "failure.json", failure)
        _write_json(evidence_dir / "summary.json", failure)
        return failure

    analyses = {
        partition.value: _analyze_partition(partition_rows[partition])
        for partition in (Partition.TRAIN, Partition.DEVELOPMENT)
    }
    global_action = _select_global_action(
        train_rows=partition_rows[Partition.TRAIN],
        development_rows=partition_rows[Partition.DEVELOPMENT],
    )
    policy_comparisons = _compare_selected_policies(
        analyses[Partition.DEVELOPMENT.value],
        tuple(Path(path) for path in selected_policy_evidence),
    )
    assessment = _assess_headroom(analyses)
    design_path = evidence_dir / "design.json"
    raw_files[design_path.name] = sha256_file(design_path)[0]
    result = {
        "status": assessment["status"],
        "scope": "train-development-constant-grid-headroom-audit",
        "identities": identities,
        "design": design,
        "partitions": analyses,
        "train_selected_global_constant_action": global_action,
        "selected_policy_comparisons": policy_comparisons,
        "assessment": assessment,
        "raw_file_sha256": raw_files,
        "heldout_read": False,
        "optimizer_updates": 0,
        "f15_trade_contrast_status": "UNEVALUABLE",
    }
    result["evidence_sha256"] = _digest(result)
    _write_json(evidence_dir / "summary.json", result)
    return result


def _design_contract() -> dict[str, Any]:
    return {
        "frozen_before_development_outcomes": True,
        "partitions": [Partition.TRAIN.value, Partition.DEVELOPMENT.value],
        "heldout_read": False,
        "train_snapshots": 210,
        "development_snapshots": 46,
        "grid": {
            "field_operation": list(GRID_VALUES),
            "development_budget": list(GRID_VALUES),
            "constant_across_episode": True,
            "field_development_share": 1.0,
            "inactive_herd_retention": 0.0,
            "requested_cells": len(GRID_ACTIONS),
        },
        "baselines": ["option-a-through-shared-compiler"],
        "episode_ceiling": AUDIT_STEPS,
        "primary_endpoint_decisions": PRIMARY_STEPS,
        "secondary_endpoint_decisions": AUDIT_STEPS,
        "horizon_years": HORIZON_YEARS,
        "episode_budget": 6_656,
        "forward_transition_budget": 33_280,
        "optimizer_updates": 0,
        "relative_improvement": "(candidate-option_a)/max(abs(option_a),1 raw reward unit)",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "connected scenario cluster",
            "dependency_keys": [
                "world_seed",
                "trajectory_family+lineage_id",
                "counterfactual_group_id",
            ],
            "overall_one_sided_confidence": 0.95,
            "family_one_sided_confidence": 1.0 - 0.05 / FAMILY_BONFERRONI_TESTS,
        },
        "gates": {
            "overall_effect_floor": OVERALL_EFFECT_FLOOR,
            "family_effect_floor": FAMILY_EFFECT_FLOOR,
            "minimum_overall_clusters_per_partition": OVERALL_CLUSTER_MINIMUM,
            "minimum_development_clusters_per_qualifying_family": (
                FAMILY_DEVELOPMENT_CLUSTER_MINIMUM
            ),
            "qualifying_families_required": 2,
            "excluded_qualifying_family": NONQUALIFYING_FAMILY,
            "go": (
                "complete three-step integrity; overall means >=1.5pp on TRAIN and DEV "
                "with positive one-sided 95% lower bounds; >=2 non-F15 families with "
                "means >=2pp on both partitions, positive Bonferroni DEV lower bound, "
                "and >=5 independent DEV clusters"
            ),
            "no_go": (
                "adequate overall cluster coverage and one-sided 95% upper bound below "
                "1.5pp on TRAIN or DEV"
            ),
            "otherwise": "INCONCLUSIVE",
        },
        "prefix_equivalence": {
            "method": "extract first three physical transitions from five-step trajectories",
            "basis": (
                "At the frozen native SHA, EpisodeDecisionLimit is consulted only after "
                "TurnResolver.Resolve, reward construction, and episode-step increment."
            ),
            "three_step_truncation_flag_excluded_from_physical_prefix": True,
        },
        "interpretation_limits": {
            "best_grid": "per-snapshot hindsight within the constant-action class",
            "not_feedback_policy_upper_bound": True,
            "five_step_cannot_authorize_phase_0b": True,
            "negative_grid_does_not_prove_task_impossibility": True,
            "development_is_diagnostic_not_pristine": True,
            "f15_cannot_qualify_family_gate": True,
        },
    }


def _run_partition(
    *,
    partition: Partition,
    records: tuple[SnapshotRecord, ...],
    dataset: CampaignDataset,
    bridge_path: Path,
    dotnet: str,
    progress: ProgressCallback | None,
) -> list[dict[str, Any]]:
    env = _make_environment(
        bridge_path=bridge_path,
        dotnet=dotnet,
        selection=dataset.selection(records[0]),
        episode_limit=AUDIT_STEPS,
    )
    rows: list[dict[str, Any]] = []
    try:
        for index, record in enumerate(records, start=1):
            selection = dataset.selection(record)
            for grid_index, (field, development) in enumerate(GRID_ACTIONS):
                rows.append(
                    _run_episode(
                        env=env,
                        selection=selection,
                        policy="constant-grid",
                        grid_index=grid_index,
                        field=field,
                        development=development,
                    )
                )
            rows.append(
                _run_episode(
                    env=env,
                    selection=selection,
                    policy="option-a",
                    grid_index=None,
                    field=None,
                    development=None,
                )
            )
            if index % 8 == 0 or index == len(records):
                _notify(
                    progress,
                    phase="constant-grid",
                    partition=partition.value,
                    completed_snapshots=index,
                    total_snapshots=len(records),
                    completed_episodes=index * (len(GRID_ACTIONS) + 1),
                )
    finally:
        env.close()
    expected = len(records) * (len(GRID_ACTIONS) + 1)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} {partition.value} episodes, found {len(rows)}")
    return rows


def _run_episode(
    *,
    env: Any,
    selection: CampaignSelection,
    policy: str,
    grid_index: int | None,
    field: float | None,
    development: float | None,
) -> dict[str, Any]:
    observation, info = env.reset(options=_reset_request(selection, AUDIT_STEPS))
    _validate_campaign_reset(_plain_mapping(info["native_result"]), selection)
    trajectory: list[dict[str, Any]] = []
    context = {
        "policy": policy,
        "grid_index": grid_index,
        "field_operation": field,
        "development_budget": development,
    }
    _validate_scope_and_observation(
        selection=selection,
        phase="reset",
        observation=observation,
        info=info,
        prefix=trajectory,
        context=context,
    )
    reset_state = _mapping(_plain_mapping(info["native_result"]), "StateDiagnostics")
    prior_population = int(_pick(reset_state, "Population"))
    for decision in range(AUDIT_STEPS):
        if policy == "constant-grid":
            assert field is not None and development is not None
            continuous = np.asarray((field, 0.0, development, 1.0), dtype=np.float64)
            movement = 0
            next_observation, _, terminated, truncated, next_info = env.step(
                {"continuous": continuous, "movement_index": movement}
            )
        else:
            next_observation, _, terminated, truncated, next_info = env.option_a_step()
            native = _plain_mapping(next_info["native_result"])
            requested = _mapping(native, "Requested")
            continuous = np.asarray(
                (
                    float(_pick(requested, "RetainedFieldOperation")),
                    float(_pick(requested, "RetainedHerdFraction")),
                    float(_pick(requested, "DevelopmentBudget")),
                    float(_pick(requested, "FieldDevelopmentShare")),
                ),
                dtype=np.float64,
            )
            movement = int(_pick(requested, "MovementCandidate")) + 1
        result = _plain_mapping(next_info["native_result"])
        step = _validated_step(
            selection=selection,
            phase=f"{policy}-post-step-{decision + 1}",
            observation=next_observation,
            info=next_info,
            result=result,
            continuous=continuous,
            movement=movement,
            prior_population=prior_population,
            prefix=trajectory,
            context={**context, "decision": decision + 1},
            model=None,
            terminated=terminated,
            truncated=truncated,
        )
        step["state_fingerprint"] = str(_pick(result, "StateFingerprint"))
        trajectory.append(step)
        state = _mapping(result, "StateDiagnostics")
        if bool(_pick(state, "Active")):
            prior_population = int(_pick(state, "Population"))
        observation = next_observation
        info = next_info
        if truncated and not terminated and decision + 1 != AUDIT_STEPS:
            _raise_episode_failure(
                "Episode truncated before the frozen five-decision ceiling",
                selection,
                context,
                trajectory,
            )
        if terminated or truncated:
            break
    replay = env.verify_replay()
    if not bool(replay.get("ok", True)) or int(replay.get("steps", 0)) != len(trajectory):
        _raise_episode_failure(
            "Native replay failed during constant-grid audit",
            selection,
            context,
            trajectory,
            extra={"native_replay": replay},
        )
    if not trajectory:
        _raise_episode_failure(
            "Constant-grid audit produced an empty trajectory",
            selection,
            context,
            trajectory,
        )
    returns = _trajectory_returns(trajectory)
    return {
        "snapshot_id": selection.record.snapshot_id,
        "partition": selection.record.partition.value,
        "family": selection.record.scenario_family,
        "world_seed": selection.record.world_seed,
        "trajectory_family": selection.record.trajectory_family,
        "lineage_id": selection.record.lineage_id,
        "counterfactual_group_id": selection.record.counterfactual_group_id,
        "policy": policy,
        "grid_index": grid_index,
        "constant_requested_action": (
            {"field_operation": field, "development_budget": development}
            if policy == "constant-grid"
            else None
        ),
        "steps": len(trajectory),
        "terminated": bool(trajectory[-1]["terminated"]),
        "truncated": bool(trajectory[-1]["truncated"]),
        "returns": returns,
        "first_three_prefix_sha256": _prefix_digest(trajectory[:PRIMARY_STEPS]),
        "native_replay": replay,
        "trajectory": trajectory,
    }


def _trajectory_returns(trajectory: list[dict[str, Any]]) -> dict[str, float]:
    discount = 1.0
    through_three = 0.0
    through_five = 0.0
    raw_three = 0.0
    raw_five = 0.0
    for index, step in enumerate(trajectory):
        interval = discounted_interval_reward(
            step["reward"]["welfare_rate"],
            step["elapsed_years"],
            HORIZON_YEARS,
        )
        through_five += discount * interval
        raw_five += step["reward"]["undiscounted_interval_welfare"]
        if index < PRIMARY_STEPS:
            through_three += discount * interval
            raw_three += step["reward"]["undiscounted_interval_welfare"]
        discount *= physical_gamma(step["elapsed_years"], HORIZON_YEARS)
    return {
        "physical_discounted_through_3": through_three,
        "physical_discounted_through_5": through_five,
        "raw_welfare_through_3": raw_three,
        "raw_welfare_through_5": raw_five,
    }


def _prefix_digest(trajectory: list[dict[str, Any]]) -> str:
    payload = [
        {
            "decision": step["decision"],
            "requested": step["requested"],
            "accepted": step["accepted"],
            "realized": step["realized"],
            "elapsed_years": step["elapsed_years"],
            "reward": step["reward"],
            "state_fingerprint": step["state_fingerprint"],
            "terminated": step["terminated"],
        }
        for step in trajectory
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _analyze_partition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_snapshot[row["snapshot_id"]].append(row)
    records: list[dict[str, Any]] = []
    for snapshot_id in sorted(by_snapshot):
        episodes = by_snapshot[snapshot_id]
        option = [episode for episode in episodes if episode["policy"] == "option-a"]
        grid = [episode for episode in episodes if episode["policy"] == "constant-grid"]
        if len(option) != 1 or len(grid) != len(GRID_ACTIONS):
            raise ValueError(f"Incomplete grid for {snapshot_id}")
        record = {
            "snapshot_id": snapshot_id,
            "partition": episodes[0]["partition"],
            "family": episodes[0]["family"],
            "world_seed": episodes[0]["world_seed"],
            "trajectory_family": episodes[0]["trajectory_family"],
            "lineage_id": episodes[0]["lineage_id"],
            "counterfactual_group_id": episodes[0]["counterfactual_group_id"],
            "option_a": {
                "physical_discounted_through_3": option[0]["returns"][
                    "physical_discounted_through_3"
                ],
                "physical_discounted_through_5": option[0]["returns"][
                    "physical_discounted_through_5"
                ],
            },
        }
        for horizon in (PRIMARY_STEPS, AUDIT_STEPS):
            key = f"physical_discounted_through_{horizon}"
            ordered = sorted(grid, key=lambda episode: int(episode["grid_index"]))
            values = np.asarray([episode["returns"][key] for episode in ordered])
            best_value = float(np.max(values))
            tolerance = TIE_RELATIVE_TOLERANCE * max(1.0, abs(best_value))
            tied = [
                episode
                for episode in ordered
                if abs(float(episode["returns"][key]) - best_value) <= tolerance
            ]
            selected = tied[0]
            option_value = float(option[0]["returns"][key])
            relative = _relative_improvement(best_value, option_value)
            portfolio = max(best_value, option_value)
            record[f"through_{horizon}"] = {
                "best_grid_return": best_value,
                "option_a_return": option_value,
                "absolute_improvement": best_value - option_value,
                "relative_improvement": relative,
                "hindsight_portfolio_return": portfolio,
                "hindsight_portfolio_label": (
                    "per-snapshot hindsight diagnostic, not a deployable policy"
                ),
                "selected_grid_index": int(selected["grid_index"]),
                "selected_action": selected["constant_requested_action"],
                "tie_count": len(tied),
                "tied_grid_indices": [int(episode["grid_index"]) for episode in tied],
                "distinct_accepted_action_trajectories": _distinct_accepted_count(
                    ordered, horizon
                ),
            }
        records.append(record)

    cluster_ids = _scenario_cluster_ids(records)
    for record in records:
        record["scenario_cluster_id"] = cluster_ids[record["snapshot_id"]]
    overall = {
        f"through_{horizon}": _summarize_headroom(records, horizon)
        for horizon in (PRIMARY_STEPS, AUDIT_STEPS)
    }
    families = {
        family: {
            f"through_{horizon}": _summarize_headroom(
                [record for record in records if record["family"] == family], horizon
            )
            for horizon in (PRIMARY_STEPS, AUDIT_STEPS)
        }
        for family in sorted({record["family"] for record in records})
    }
    behavior = _compiler_summary(rows)
    return {
        "partition": rows[0]["partition"],
        "snapshots": len(records),
        "episodes": len(rows),
        "maximum_forward_transitions": len(rows) * AUDIT_STEPS,
        "scenario_clusters": len(set(cluster_ids.values())),
        "integrity": {
            "status": "PASS",
            "complete_three_step_matrix": True,
            "complete_five_step_matrix": True,
            "scope_failures": 0,
            "replay_failures": 0,
            "request_mismatches": 0,
            "non_finite_values": 0,
            "reward_denominator_violations": 0,
        },
        "overall": overall,
        "families": families,
        "requested_accepted_realized": behavior,
        "snapshot_results": records,
    }


def _summarize_headroom(records: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    if not records:
        return {"snapshots": 0, "scenario_clusters": 0}
    values = np.asarray(
        [record[f"through_{horizon}"]["relative_improvement"] for record in records],
        dtype=np.float64,
    )
    absolute = np.asarray(
        [record[f"through_{horizon}"]["absolute_improvement"] for record in records],
        dtype=np.float64,
    )
    clusters: dict[str, list[float]] = defaultdict(list)
    for record, value in zip(records, values, strict=True):
        clusters[record["scenario_cluster_id"]].append(float(value))
    bootstrap = _cluster_bootstrap(
        clusters,
        replicates=BOOTSTRAP_REPLICATES,
        seed=_stable_seed(
            f"{BOOTSTRAP_SEED}:{records[0].get('partition', '')}:"
            f"{records[0]['family']}:{horizon}:{len(records)}"
        ),
    )
    return {
        "snapshots": len(records),
        "scenario_clusters": len(clusters),
        "relative_improvement_mean": float(np.mean(values)),
        "relative_improvement_p50": float(np.quantile(values, 0.5)),
        "relative_improvement_p05": float(np.quantile(values, 0.05)),
        "relative_improvement_p95": float(np.quantile(values, 0.95)),
        "absolute_improvement_mean": float(np.mean(absolute)),
        "fraction_positive": float(np.mean(values > 0.0)),
        "one_sided_95_lower": float(np.quantile(bootstrap, 0.05)),
        "one_sided_95_upper": float(np.quantile(bootstrap, 0.95)),
        "one_sided_bonferroni_family_lower": float(
            np.quantile(bootstrap, 0.05 / FAMILY_BONFERRONI_TESTS)
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _cluster_bootstrap(
    clusters: dict[str, list[float]],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    keys = sorted(clusters)
    if not keys:
        raise ValueError("Cluster bootstrap requires at least one scenario cluster")
    sums = np.asarray([sum(clusters[key]) for key in keys], dtype=np.float64)
    counts = np.asarray([len(clusters[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(replicates, len(keys)))
    return np.sum(sums[draws], axis=1) / np.sum(counts[draws], axis=1)


def _scenario_cluster_ids(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    rows = list(records)
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(rows):
        keys = [
            ("world", str(record["world_seed"])),
            (
                "lineage",
                f"{record['trajectory_family']}::{record['lineage_id']}",
            ),
        ]
        if record.get("counterfactual_group_id") is not None:
            keys.append(("counterfactual", str(record["counterfactual_group_id"])))
        for key in keys:
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index

    members: dict[int, list[str]] = defaultdict(list)
    for index, record in enumerate(rows):
        members[find(index)].append(str(record["snapshot_id"]))
    labels = {
        root: "cluster-"
        + hashlib.sha256("\n".join(sorted(snapshot_ids)).encode()).hexdigest()[:16]
        for root, snapshot_ids in members.items()
    }
    return {
        str(record["snapshot_id"]): labels[find(index)]
        for index, record in enumerate(rows)
    }


def _select_global_action(
    *,
    train_rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    train_grid = [row for row in train_rows if row["policy"] == "constant-grid"]
    means: list[tuple[int, float]] = []
    for grid_index in range(len(GRID_ACTIONS)):
        values = [
            row["returns"]["physical_discounted_through_3"]
            for row in train_grid
            if row["grid_index"] == grid_index
        ]
        if len(values) != 210:
            raise ValueError(f"Global action {grid_index} does not cover all TRAIN snapshots")
        means.append((grid_index, float(np.mean(values))))
    best_mean = max(value for _, value in means)
    tolerance = TIE_RELATIVE_TOLERANCE * max(1.0, abs(best_mean))
    selected_index = min(index for index, value in means if abs(value - best_mean) <= tolerance)
    field, development = GRID_ACTIONS[selected_index]
    return {
        "selection_partition": Partition.TRAIN.value,
        "selection_endpoint": "physical_discounted_through_3",
        "grid_index": selected_index,
        "action": {"field_operation": field, "development_budget": development},
        "train": _global_action_partition_summary(train_rows, selected_index),
        "development": _global_action_partition_summary(development_rows, selected_index),
        "development_used_for_selection": False,
    }


def _global_action_partition_summary(
    rows: list[dict[str, Any]], grid_index: int
) -> dict[str, Any]:
    grid_by_id = {
        row["snapshot_id"]: row
        for row in rows
        if row["policy"] == "constant-grid" and row["grid_index"] == grid_index
    }
    option_by_id = {
        row["snapshot_id"]: row for row in rows if row["policy"] == "option-a"
    }
    relative = []
    absolute = []
    for snapshot_id in sorted(option_by_id):
        candidate = grid_by_id[snapshot_id]["returns"]["physical_discounted_through_3"]
        option = option_by_id[snapshot_id]["returns"]["physical_discounted_through_3"]
        absolute.append(candidate - option)
        relative.append(_relative_improvement(candidate, option))
    return {
        "snapshots": len(relative),
        "mean_relative_improvement_vs_option_a": float(np.mean(relative)),
        "mean_absolute_improvement_vs_option_a": float(np.mean(absolute)),
        "fraction_at_or_above_option_a": float(np.mean(np.asarray(relative) >= -1e-12)),
    }


def _compare_selected_policies(
    development_analysis: dict[str, Any], evidence_paths: tuple[Path, ...]
) -> list[dict[str, Any]]:
    if not evidence_paths:
        return []
    best_by_id = {
        record["snapshot_id"]: record["through_3"]["best_grid_return"]
        for record in development_analysis["snapshot_results"]
    }
    comparisons: list[dict[str, Any]] = []
    for path in evidence_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes = payload["evaluation"]["development"]["episodes"]
        if len(episodes) != len(best_by_id) or {row["snapshot_id"] for row in episodes} != set(
            best_by_id
        ):
            raise ValueError(f"Selected-policy evidence {path} is not the frozen DEV panel")
        if any(int(row["steps"]) != PRIMARY_STEPS for row in episodes):
            raise ValueError(f"Selected-policy evidence {path} is not three-decision evidence")
        relative = []
        absolute = []
        for row in episodes:
            best = best_by_id[row["snapshot_id"]]
            policy = float(row["discounted_return"])
            absolute.append(best - policy)
            relative.append(_relative_improvement(best, policy))
        comparisons.append(
            {
                "seed": int(payload["seed"]),
                "selected_update": int(payload["selected_update"]),
                "evidence_path": str(path),
                "evidence_sha256": sha256_file(path)[0],
                "mean_best_grid_minus_policy": float(np.mean(absolute)),
                "mean_relative_best_grid_minus_policy": float(np.mean(relative)),
                "fraction_best_grid_at_or_above_policy": float(
                    np.mean(np.asarray(relative) >= -1e-12)
                ),
            }
        )
    return comparisons


def _assess_headroom(analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    train = analyses[Partition.TRAIN.value]
    development = analyses[Partition.DEVELOPMENT.value]
    train_overall = train["overall"]["through_3"]
    development_overall = development["overall"]["through_3"]
    integrity = all(
        analysis["integrity"]["status"] == "PASS"
        and analysis["integrity"]["complete_three_step_matrix"]
        for analysis in (train, development)
    )
    overall_go = bool(
        train_overall["relative_improvement_mean"] >= OVERALL_EFFECT_FLOOR
        and development_overall["relative_improvement_mean"] >= OVERALL_EFFECT_FLOOR
        and train_overall["one_sided_95_lower"] > 0.0
        and development_overall["one_sided_95_lower"] > 0.0
    )
    qualifying: list[str] = []
    family_disposition: dict[str, dict[str, Any]] = {}
    for family in sorted(train["families"]):
        train_family = train["families"][family]["through_3"]
        development_family = development["families"][family]["through_3"]
        eligible = bool(
            family != NONQUALIFYING_FAMILY
            and development_family["scenario_clusters"]
            >= FAMILY_DEVELOPMENT_CLUSTER_MINIMUM
        )
        passes = bool(
            eligible
            and train_family["relative_improvement_mean"] >= FAMILY_EFFECT_FLOOR
            and development_family["relative_improvement_mean"] >= FAMILY_EFFECT_FLOOR
            and development_family["one_sided_bonferroni_family_lower"] > 0.0
        )
        if passes:
            qualifying.append(family)
        family_disposition[family] = {
            "eligible_to_qualify": eligible,
            "qualifies": passes,
            "development_clusters": development_family["scenario_clusters"],
            "f15_trade_contrast_status": (
                "UNEVALUABLE" if family == NONQUALIFYING_FAMILY else None
            ),
        }
    go = bool(integrity and overall_go and len(qualifying) >= 2)
    coverage_adequate = bool(
        train_overall["scenario_clusters"] >= OVERALL_CLUSTER_MINIMUM
        and development_overall["scenario_clusters"] >= OVERALL_CLUSTER_MINIMUM
    )
    no_go = bool(
        integrity
        and coverage_adequate
        and (
            train_overall["one_sided_95_upper"] < OVERALL_EFFECT_FLOOR
            or development_overall["one_sided_95_upper"] < OVERALL_EFFECT_FLOOR
        )
    )
    if not integrity:
        status = "STOP_INTEGRITY_FAILURE"
    elif go:
        status = "GO_PROPOSE_PHASE_0B"
    elif no_go:
        status = "NO_GO_CONSTANT_GRID_ROUTE"
    else:
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "three_step_primary_integrity_gate": integrity,
        "overall_go_gate": overall_go,
        "overall_cluster_coverage_adequate": coverage_adequate,
        "qualifying_families": qualifying,
        "qualifying_family_count": len(qualifying),
        "family_disposition": family_disposition,
        "phase_0b_proposal_justified": status == "GO_PROPOSE_PHASE_0B",
        "five_step_can_change_status": False,
        "architecture_change_justified": False,
        "normalizer_change_justified": False,
        "activation_change_justified": False,
        "longer_ppo_justified": False,
        "heldout_remains_sealed": True,
    }


def _compiler_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grid = [row for row in rows if row["policy"] == "constant-grid"]
    requested_accepted: list[float] = []
    accepted_realized: list[float] = []
    interventions = 0
    transitions = 0
    distinct_three: list[int] = []
    distinct_five: list[int] = []
    ties_three: list[int] = []
    ties_five: list[int] = []
    by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in grid:
        by_snapshot[row["snapshot_id"]].append(row)
        for step in row["trajectory"]:
            requested_accepted.append(_action_distance(step["requested"], step["accepted"]))
            accepted_realized.append(_action_distance(step["accepted"], step["realized"]))
            interventions += len(step["interventions"])
            transitions += 1
    for episodes in by_snapshot.values():
        distinct_three.append(_distinct_accepted_count(episodes, PRIMARY_STEPS))
        distinct_five.append(_distinct_accepted_count(episodes, AUDIT_STEPS))
        for horizon, target in ((PRIMARY_STEPS, ties_three), (AUDIT_STEPS, ties_five)):
            key = f"physical_discounted_through_{horizon}"
            best = max(float(row["returns"][key]) for row in episodes)
            tolerance = TIE_RELATIVE_TOLERANCE * max(1.0, abs(best))
            target.append(
                sum(abs(float(row["returns"][key]) - best) <= tolerance for row in episodes)
            )
    return {
        "grid_transitions": transitions,
        "requested_to_accepted_l2": _distribution(requested_accepted),
        "accepted_to_realized_l2": _distribution(accepted_realized),
        "compiler_interventions": interventions,
        "interventions_per_transition": interventions / transitions if transitions else None,
        "distinct_accepted_trajectories_through_3": _distribution(distinct_three),
        "distinct_accepted_trajectories_through_5": _distribution(distinct_five),
        "best_grid_tie_count_through_3": _distribution(ties_three),
        "best_grid_tie_count_through_5": _distribution(ties_five),
    }


def _distinct_accepted_count(episodes: list[dict[str, Any]], horizon: int) -> int:
    signatures = {
        json.dumps(
            [step["accepted"] for step in episode["trajectory"][:horizon]],
            sort_keys=True,
            separators=(",", ":"),
        )
        for episode in episodes
    }
    return len(signatures)


def _action_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = (
        "field_operation",
        "herd_retention",
        "development_budget",
        "field_development_share",
    )
    return math.sqrt(sum((float(left[key]) - float(right[key])) ** 2 for key in keys))


def _relative_improvement(candidate: float, reference: float) -> float:
    return (candidate - reference) / max(abs(reference), 1.0)


def _distribution(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _audit_identities(
    *,
    repository_root: Path,
    bridge_path: Path,
    dataset: CampaignDataset,
    normalizer_path: Path,
    selected_policy_evidence: tuple[Path, ...],
) -> dict[str, Any]:
    learner_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    tracked_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    )
    if tracked_dirty:
        raise ValueError("Headroom audit requires a clean tracked learner source tree")
    return {
        "audit_learner_commit": learner_commit,
        "native_simulator_commit": dataset.manifest.native_simulator_sha,
        "snapshot_manifest_sha256": dataset.manifest_sha256,
        "normalizer_partition_id": dataset.normalizer.manifest.partition_id,
        "normalizer_sha256": sha256_file(normalizer_path)[0],
        "bridge_sha256": sha256_file(bridge_path)[0],
        "selected_policy_evidence_sha256": {
            str(path): sha256_file(path)[0] for path in selected_policy_evidence
        },
    }


def _raise_episode_failure(
    message: str,
    selection: CampaignSelection,
    context: dict[str, Any],
    trajectory: list[dict[str, Any]],
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    raise AuditIntegrityError(
        message,
        {
            "snapshot_id": selection.record.snapshot_id,
            "partition": selection.record.partition.value,
            "family": selection.record.scenario_family,
            "context": context,
            "trajectory": trajectory,
            **(extra or {}),
        },
    )


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**63 - 1)


def _notify(progress: ProgressCallback | None, **payload: Any) -> None:
    if progress is not None:
        progress(payload)
