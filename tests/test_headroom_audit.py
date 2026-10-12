from __future__ import annotations

import numpy as np

from anthropocene_rl.training.headroom_audit import (
    GRID_ACTIONS,
    _assess_headroom,
    _cluster_bootstrap,
    _relative_improvement,
    _scenario_cluster_ids,
)


def test_grid_is_fixed_lexicographic_five_by_five() -> None:
    assert len(GRID_ACTIONS) == 25
    assert GRID_ACTIONS[0] == (0.0, 0.0)
    assert GRID_ACTIONS[-1] == (1.0, 1.0)
    assert len(set(GRID_ACTIONS)) == 25


def test_scenario_clusters_join_shared_dependencies() -> None:
    rows = [
        {
            "snapshot_id": "a",
            "world_seed": 1,
            "trajectory_family": "t",
            "lineage_id": "l1",
            "counterfactual_group_id": None,
        },
        {
            "snapshot_id": "b",
            "world_seed": 2,
            "trajectory_family": "t",
            "lineage_id": "l1",
            "counterfactual_group_id": None,
        },
        {
            "snapshot_id": "c",
            "world_seed": 3,
            "trajectory_family": "t",
            "lineage_id": "l3",
            "counterfactual_group_id": "pair",
        },
        {
            "snapshot_id": "d",
            "world_seed": 4,
            "trajectory_family": "t",
            "lineage_id": "l4",
            "counterfactual_group_id": "pair",
        },
    ]
    clusters = _scenario_cluster_ids(rows)
    assert clusters["a"] == clusters["b"]
    assert clusters["c"] == clusters["d"]
    assert clusters["a"] != clusters["c"]


def test_cluster_bootstrap_is_deterministic_and_cluster_aware() -> None:
    clusters = {"a": [0.0, 0.0], "b": [1.0]}
    first = _cluster_bootstrap(clusters, replicates=100, seed=7)
    second = _cluster_bootstrap(clusters, replicates=100, seed=7)
    assert np.array_equal(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_relative_improvement_uses_one_raw_unit_floor() -> None:
    assert _relative_improvement(0.5, 0.0) == 0.5
    assert _relative_improvement(110.0, 100.0) == 0.1


def _analysis(mean: float, lower: float, upper: float) -> dict[str, object]:
    family = {
        "through_3": {
            "relative_improvement_mean": mean,
            "one_sided_bonferroni_family_lower": lower,
            "scenario_clusters": 6,
        }
    }
    return {
        "integrity": {"status": "PASS", "complete_three_step_matrix": True},
        "overall": {
            "through_3": {
                "relative_improvement_mean": mean,
                "one_sided_95_lower": lower,
                "one_sided_95_upper": upper,
                "scenario_clusters": 30,
            }
        },
        "families": {
            "F01": family,
            "F02": family,
            "F15": family,
        },
    }


def test_go_gate_requires_two_non_f15_qualifying_families() -> None:
    analyses = {
        "train": _analysis(0.03, 0.01, 0.05),
        "development": _analysis(0.03, 0.01, 0.05),
    }
    result = _assess_headroom(analyses)
    assert result["status"] == "GO_PROPOSE_PHASE_0B"
    assert result["qualifying_families"] == ["F01", "F02"]


def test_no_go_requires_upper_bound_below_effect_floor() -> None:
    analyses = {
        "train": _analysis(0.005, -0.01, 0.014),
        "development": _analysis(0.005, -0.01, 0.02),
    }
    result = _assess_headroom(analyses)
    assert result["status"] == "NO_GO_CONSTANT_GRID_ROUTE"
