"""Synthetic-tensor checks of the labor pretrain evaluation metric math."""

from __future__ import annotations

import numpy as np
import pytest

from anthropocene_rl.autonomous_v7 import labor_pretrain_eval as ev
from anthropocene_rl.autonomous_v7.labor_bridge import (
    ACTIVITY_ROWS,
    ROLE_ROWS,
    SPECIALTY_ROWS,
    apportion_largest_remainder,
)


def test_apportionment_matches_native_rule() -> None:
    rng = np.random.default_rng(3)
    shares = rng.dirichlet(np.ones(SPECIALTY_ROWS), size=200)
    shares[:, 3] = 0.0
    shares /= shares.sum(axis=1, keepdims=True)
    population = rng.integers(1, 900, size=200)
    got = ev.apportion_largest_remainder_rows(shares, population)
    for row in range(200):
        assert tuple(got[row]) == apportion_largest_remainder(
            tuple(shares[row]), int(population[row])
        )
        assert got[row].sum() == population[row]
    # Ties go to the lower ordinal.
    assert ev.apportion_largest_remainder_rows(np.array([[0.5, 0.5]]), np.array([3])).tolist() == [
        [2, 1]
    ]


def test_nmae_and_bias() -> None:
    pred, target = np.array([1.0, 3.0]), np.array([2.0, 2.0])
    assert ev.normalized_mae(pred, target, 2.0) == pytest.approx(0.5)
    assert ev.normalized_bias(pred, target, 2.0) == pytest.approx(0.0)
    assert ev.normalized_bias(pred + 1.0, target, 4.0) == pytest.approx(0.25)
    assert ev.normalized_mae(pred, target, 1.0, np.array([3.0, 1.0])) == pytest.approx(1.0)


def test_world_balanced_weights() -> None:
    weights = ev.world_balanced(np.array([1, 1, 1, 2]), np.array([True, True, True, True]))
    assert weights[:3].sum() == pytest.approx(weights[3])
    assert weights.mean() == pytest.approx(1.0)


def test_ranking_accuracy_counts_only_informative_pairs_within_groups() -> None:
    target = np.array([0.0, 1.0, 2.0, 2.01, 5.0, 0.0])
    pred = np.array([0.0, 1.0, 3.0, 2.5, 0.0, 9.0])
    worlds = np.array([1, 1, 1, 1, 2, 2])
    turns = np.array([0, 0, 0, 0, 0, 0])
    result = ev.ranking_accuracy(pred, target, worlds, turns, threshold=0.05)
    # World 1: informative pairs (0,1),(0,2),(0,3),(1,2),(1,3) all correct; (2,3) |dG|=0.01 excluded.
    # World 2: one pair, wrong order.
    assert result["informative_pairs"] == 6
    assert result["accuracy"] == pytest.approx(5 / 6)
    assert result["world_balanced_accuracy"] == pytest.approx(0.5)
    assert result["informative_groups"] == 2 and result["informative_worlds"] == 2
    # Different turns never pair; prediction ties score one half.
    split = ev.ranking_accuracy(
        np.array([1.0, 1.0]), np.array([0.0, 1.0]), np.array([1, 1]), np.array([0, 1]), 0.05
    )
    assert split["informative_pairs"] == 0 and split["accuracy"] is None
    tie = ev.ranking_accuracy(
        np.array([1.0, 1.0]), np.array([0.0, 1.0]), np.array([1, 1]), np.array([0, 0]), 0.05
    )
    assert tie["accuracy"] == pytest.approx(0.5)


def _head(nmae: float, bias: float, rank: float | None, worlds: int, groups: int) -> dict:
    return {
        "rows": 10,
        "nmae": nmae,
        "bias": bias,
        "worlds": worlds,
        "ranking": {"accuracy": rank, "informative_groups": groups},
    }


def test_critic_gate_semantics() -> None:
    assert ev.critic_gate(_head(0.1, 0.05, 0.8, 20, 40), zero_head=False) == "PASS"
    assert ev.critic_gate(_head(0.3, 0.05, 0.8, 20, 40), zero_head=False) == "FAIL"
    assert ev.critic_gate(_head(0.1, -0.2, 0.8, 20, 40), zero_head=False) == "FAIL"
    assert ev.critic_gate(_head(0.1, 0.0, 0.6, 20, 40), zero_head=False) == "FAIL"
    assert ev.critic_gate(_head(0.1, 0.0, 0.8, 5, 40), zero_head=False) == "HOLD"
    assert ev.critic_gate(_head(0.1, 0.0, 0.8, 20, 10), zero_head=False) == "HOLD"
    zero = {"rows": 5, "max_abs_pred": 0.01, "mean_abs_pred": 0.005}
    assert ev.critic_gate(zero, zero_head=True) == "PASS"
    assert ev.critic_gate(dict(zero, max_abs_pred=0.2), zero_head=True) == "HOLD"
    assert ev.critic_gate(dict(zero, max_abs_pred=0.2, mean_abs_pred=0.1), zero_head=True) == "FAIL"


def test_strata_helpers() -> None:
    index, count = ev.decision_index(np.array([1, 1, 1, 2]), np.array([10, 12, 11, 5]))
    assert index.tolist() == [0, 2, 1, 0] and count.tolist() == [3, 3, 3, 1]
    assert ev.population_bin(np.array([50, 100, 450, 600])).tolist() == [
        "<100",
        "100-300",
        "300-600",
        ">600",
    ]


def _synthetic_actor_arrays(n: int = 6) -> dict:
    rng = np.random.default_rng(0)
    spec_mask = np.ones((n, SPECIALTY_ROWS), dtype=bool)
    act_mask = np.zeros((n, ROLE_ROWS, ACTIVITY_ROWS), dtype=bool)
    act_mask[:, :, :3] = True
    target_spec = rng.dirichlet(np.ones(SPECIALTY_ROWS), size=n)
    target_act = np.zeros((n, ROLE_ROWS, ACTIVITY_ROWS))
    target_act[:, :, :3] = rng.dirichlet(np.ones(3), size=(n, ROLE_ROWS))
    population = np.full(n, 100)
    counts = ev.apportion_largest_remainder_rows(target_spec, population).astype(float)
    codes = np.array([-1, 0, 1, 2])
    return {
        "row_world": np.array([1, 1, 1, 2, 2, 2]),
        "row_turn": np.zeros(n, dtype=int),
        "row_quarantined": np.zeros(n, dtype=bool),
        "label_specialty_valid": np.ones(n, dtype=bool),
        "label_activity_role_valid": np.ones((n, ROLE_ROWS), dtype=bool),
        "label_movement_valid": np.ones(n, dtype=bool),
        "label_fission_valid": np.ones(n, dtype=bool),
        "label_trade_gate_valid": np.ones((n, 7), dtype=bool),
        "labor_specialty_mask": spec_mask,
        "labor_activity_mask": act_mask,
        "target_specialty": target_spec,
        "target_activity": target_act,
        "target_specialty_counts": counts,
        "target_movement_code": np.array([-1, 0, 1, -1, 2, -1]),
        "target_fission": np.zeros(n, dtype=int),
        "target_trade_gate": np.ones((n, 7), dtype=int),
        "pred_specialty": target_spec.copy(),
        "pred_activity": target_act.copy(),
        "pred_specialty_counts": counts.copy(),
        "pred_movement_code": np.array([-1, 0, 1, -1, 2, -1]),
        "pred_movement_union_mask": np.ones((n, 4), dtype=bool),
        "movement_codes": codes,
        "pred_fission": np.zeros(n, dtype=int),
        "pred_trade_gate": np.ones((n, 7), dtype=int),
        "population": population,
    }


def test_actor_fidelity_perfect_and_perturbed() -> None:
    arrays = _synthetic_actor_arrays()
    report = ev.actor_fidelity_metrics(arrays, np.ones(6, dtype=bool))
    assert report["specialty_l1"] == pytest.approx(0.0)
    assert report["activity_l1"] == pytest.approx(0.0)
    assert report["movement"]["accuracy"] == pytest.approx(1.0)
    assert report["trade_gate"]["accuracy"] == pytest.approx(1.0)
    farmer = report["per_specialty"]["Farmer"]
    assert farmer["pred_share"] == pytest.approx(farmer["target_share"])
    # One wrong movement in world 2 (3 rows) and quarantine of a world-1 row.
    arrays["pred_movement_code"][4] = -1
    arrays["row_quarantined"][0] = True
    report = ev.actor_fidelity_metrics(arrays, np.ones(6, dtype=bool))
    assert report["admitted_rows"] == 5
    assert report["movement"]["accuracy"] == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))


def test_autonomous_report_ratios_and_gates() -> None:
    def run(world, mode, welfare, py, starve=0):
        turn = {
            "welfare": welfare,
            "person_years": py,
            "qol": welfare / py,
            "farmer_share": 0.1,
            "housing_coverage": 1.0,
            "clothing_coverage": 1.0,
            "residents": 10,
            "terminations": [1] * starve,
            "starvation_extinctions": [1] * starve,
        }
        return {"world": world, "mode": mode, "turns": [turn], "reset_identity": {"f": world}}

    results = [
        run(1, "heuristic", 100.0, 200.0),
        run(1, "deterministic", 95.0, 200.0),
        run(2, "heuristic", 100.0, 200.0),
        run(2, "deterministic", 80.0, 200.0, starve=1),
    ]
    report = ev.autonomous_report(results, ["deterministic"])
    agg = report["aggregate"]["deterministic"]
    assert agg["ratios"]["welfare"] == pytest.approx(0.875)
    assert not agg["checks"]["welfare_aggregate"]
    assert not agg["checks"]["welfare_per_world"]
    assert not agg["checks"]["no_additional_starvation_extinction"]
    assert agg["checks"]["person_years_aggregate"] and agg["checks"]["same_reset_all_worlds"]
    assert agg["gate"] == "FAIL"
    assert report["per_world"]["1"]["deterministic"]["ratios"]["welfare"] == pytest.approx(0.95)
