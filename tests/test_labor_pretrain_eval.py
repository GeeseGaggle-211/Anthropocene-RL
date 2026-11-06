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


def test_autonomous_report_h12_extras() -> None:
    def run(world, mode, welfare, py, **extra):
        turn = {
            "welfare": welfare,
            "person_years": py,
            "qol": welfare / py,
            "farmer_share": 0.1,
            "housing_coverage": 1.0,
            "clothing_coverage": 1.0,
            "residents": 10,
            "terminations": [],
            "starvation_extinctions": [],
        }
        return dict(
            {"world": world, "mode": mode, "turns": [turn], "reset_identity": {"f": world}}, **extra
        )

    fallbacks = {
        "decisions": 4,
        "activity_roles": 52,
        "counts": {"specialty": 1, "activity_role": 13, "movement": 2},
        "reasons": {"movement": 2},
    }
    base = [run(1, "heuristic", 100.0, 200.0), run(1, "deterministic", 60.0, 200.0)]
    plain = ev.autonomous_report(base, ["deterministic"])
    assert "actor_vs_h12" not in plain and "h12_fallbacks" not in plain
    results = [*base, run(1, "h12", 80.0, 200.0, h12_fallbacks=fallbacks)]
    report = ev.autonomous_report(results, ["h12", "deterministic"])
    assert report["aggregate"]["deterministic"] == plain["aggregate"]["deterministic"]
    assert report["aggregate"]["h12"]["ratios"]["welfare"] == pytest.approx(0.8)
    vs = report["actor_vs_h12"]["deterministic"]
    assert vs["ratios"]["welfare"] == pytest.approx(0.75)
    assert vs["per_world"]["1"]["person_years"] == pytest.approx(1.0)
    rates = report["h12_fallbacks"]["rates"]
    assert rates["specialty"] == pytest.approx(0.25)
    assert rates["activity_role"] == pytest.approx(0.25)
    report.update({"worlds": [1], "turns": 1, "modes": ["h12", "deterministic"]})
    assert "deterministic/h12 welfare=0.750" in ev.summarize_autonomous(report)


# ---------------------------------------------------------------------------
# actor-divergence metrics
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from anthropocene_rl.autonomous_v7.labor_bridge import ACTIVITIES, SPECIALTIES  # noqa: E402

_FARMER = SPECIALTIES.index("Farmer")
_CRAFT = SPECIALTIES.index("Craftsperson")
_A = {name: i for i, name in enumerate(ACTIVITIES)}


def _activity(rows: dict[int, dict[str, float]]) -> np.ndarray:
    """(roles x activities) share matrix; roles not listed are idle."""
    matrix = np.zeros((ROLE_ROWS, ACTIVITY_ROWS))
    matrix[:, 0] = 1.0
    for role, shares in rows.items():
        matrix[role] = 0.0
        for name, value in shares.items():
            matrix[role, _A[name]] = value
        matrix[role, 0] = 1.0 - sum(shares.values())
    return matrix


def _counts(**by_name: int) -> np.ndarray:
    counts = np.zeros(SPECIALTY_ROWS)
    for name, value in by_name.items():
        counts[SPECIALTIES.index(name)] = value
    return counts


def _teacher(**overrides):
    base = dict(
        population=10,
        specialty_counts=_counts(Gatherer=4, Farmer=4, Craftsperson=2),
        specialty_valid=True,
        activity=_activity(
            {
                _FARMER - 1: {"field.operation": 0.5, "field.expansion": 0.5},
                _CRAFT - 1: {"goods.housing": 1.0},
            }
        ),
        role_valid=np.ones(ROLE_ROWS, dtype=bool),
        movement=3,
        movement_valid=True,
        fission=1,
        fission_valid=True,
        trade_gate=[0, 1, 2],
        trade_gate_valid=[True, True, False],
    )
    base.update(overrides)
    return ev.TeacherChoice(**base)


def _actor(**overrides):
    base = dict(
        population=10,
        specialty_counts=_counts(Gatherer=4, Farmer=4, Craftsperson=2),
        activity=_activity(
            {
                _FARMER - 1: {"field.operation": 0.5, "field.expansion": 0.5},
                _CRAFT - 1: {"goods.housing": 1.0},
            }
        ),
        movement=3,
        fission=1,
        trade_gate=[0, 1, 0],
    )
    base.update(overrides)
    return ev.ActorChoice(**base)


def test_divergence_identical_choice_has_zero_disagreement() -> None:
    result = ev.camp_divergence(_teacher(), _actor())
    assert result["specialty"]["l1"] == pytest.approx(0.0)
    assert result["roles"]["compared"].sum() == 3  # Gatherer, Farmer, Craftsperson staffed
    assert result["roles"]["l1"].sum() == pytest.approx(0.0)
    assert result["roles"]["match"].sum() == 3
    assert result["movement"] is True
    assert result["fission"]["agree"] is True
    # The third trade group's teacher label is invalid: it never counts.
    assert result["trade_gate"]["valid"].tolist() == [True, True, False]
    assert result["trade_gate"]["agree"].tolist() == [True, True, False]


def test_divergence_specialty_l1_and_signed_difference() -> None:
    # The actor moves two Farmers into Gatherer: farmer share 0.4 -> 0.2, gatherer 0.4 -> 0.6.
    actor = _actor(specialty_counts=_counts(Gatherer=6, Farmer=2, Craftsperson=2))
    spec = ev.camp_divergence(_teacher(), actor)["specialty"]
    assert spec["l1"] == pytest.approx(0.4)
    assert spec["diff"][_FARMER] == pytest.approx(-0.2)
    assert spec["diff"][SPECIALTIES.index("Gatherer")] == pytest.approx(0.2)
    assert spec["diff"].sum() == pytest.approx(0.0)


def test_divergence_activity_role_l1_mismatch_and_resident_time() -> None:
    # The actor spends the Farmers' time on operation only and the Craftspersons on clothing.
    actor = _actor(
        activity=_activity(
            {
                _FARMER - 1: {"field.operation": 1.0},
                _CRAFT - 1: {"goods.cloth": 1.0},
            }
        )
    )
    result = ev.camp_divergence(_teacher(), actor)
    roles = result["roles"]
    assert roles["l1"][_FARMER - 1] == pytest.approx(1.0)
    assert roles["l1"][_CRAFT - 1] == pytest.approx(2.0)
    # Farmer argmax: tie 0.5/0.5 vs 1.0 operation -> operation (first index) on both sides.
    assert bool(roles["match"][_FARMER - 1]) is True
    assert bool(roles["match"][_CRAFT - 1]) is False
    assert roles["teacher_arg"][_CRAFT - 1] == _A["goods.housing"]
    assert roles["diff"][_CRAFT - 1, _A["goods.housing"]] == pytest.approx(-1.0)
    assert roles["diff"][_CRAFT - 1, _A["goods.cloth"]] == pytest.approx(1.0)
    rt = result["resident_time"]
    assert rt["actor"][_A["goods.cloth"]] - rt["teacher"][_A["goods.cloth"]] == pytest.approx(0.2)
    assert rt["actor"][_A["goods.housing"]] - rt["teacher"][_A["goods.housing"]] == pytest.approx(
        -0.2
    )
    assert rt["actor"].sum() == pytest.approx(rt["teacher"].sum())  # both account for all residents


def test_divergence_invalid_labels_and_rejected_actor_never_count() -> None:
    teacher = _teacher(
        specialty_valid=False,
        movement_valid=False,
        fission_valid=False,
        trade_gate_valid=[False, False, False],
    )
    result = ev.camp_divergence(teacher, _actor(movement=9, fission=0, trade_gate=[2, 2, 2]))
    assert result["specialty"] is None and result["roles"] is None
    assert result["movement"] is None and result["fission"] is None
    assert result["trade_gate"]["valid"].sum() == 0
    # A role whose activity label is invalid is excluded, the valid one still counts.
    role_valid = np.ones(ROLE_ROWS, dtype=bool)
    role_valid[_CRAFT - 1] = False
    partial = ev.camp_divergence(_teacher(role_valid=role_valid), _actor())
    assert partial["roles"]["compared"].sum() == 2
    assert partial["resident_time"] is None  # a staffed role is invalid
    # A rejected (fallback) actor request leaves the activity heads out, specialty stays.
    rejected = ev.camp_divergence(_teacher(), _actor(activity=None))
    assert rejected["labor_rejected"] is True
    assert rejected["roles"] is None and rejected["specialty"] is not None


def test_divergence_nonlabor_heads() -> None:
    result = ev.camp_divergence(_teacher(), _actor(movement=4, fission=0, trade_gate=[1, 1, 1]))
    assert result["movement"] is False
    assert result["fission"] == {"agree": False, "actor_nonzero": False, "teacher_nonzero": True}
    assert result["trade_gate"]["agree"].tolist() == [False, True, False]


def test_divergence_accumulator_summary_and_merge() -> None:
    far = ev.camp_divergence(
        _teacher(), _actor(specialty_counts=_counts(Gatherer=6, Farmer=2, Craftsperson=2))
    )
    same = ev.camp_divergence(_teacher(), _actor())
    a, b = ev.DivergenceAccumulator(), ev.DivergenceAccumulator()
    a.add(
        far,
        {
            "residents": 10.0,
            "housing_w": 5.0,
            "clothing_w": 2.0,
            "farmer_pop": 10.0,
            "farmer_teacher": 4.0,
            "farmer_actor": 2.0,
        },
    )
    b.add(
        same,
        {
            "residents": 10.0,
            "housing_w": 9.0,
            "clothing_w": 8.0,
            "farmer_pop": 10.0,
            "farmer_teacher": 4.0,
            "farmer_actor": 4.0,
        },
    )
    summary = ev.divergence_summary(ev.merge_accumulators([a, b]))
    assert summary["specialty"]["n"] == 2
    assert summary["specialty"]["l1_mean"] == pytest.approx(0.2)
    assert summary["specialty"]["per_specialty"]["Farmer"]["signed_diff"] == pytest.approx(-0.1)
    assert summary["activity"]["mismatch_rate"] == pytest.approx(0.0)
    assert summary["movement"]["disagreement"] == pytest.approx(0.0)
    assert summary["trade_gate"]["n"] == 4  # two valid groups x two camps
    ctx = summary["context"]
    assert ctx["housing_coverage"] == pytest.approx(0.7)
    assert ctx["clothing_coverage"] == pytest.approx(0.5)
    assert ctx["farmer_share_actor"] == pytest.approx(0.3)
    assert ctx["farmer_share_h12"] == pytest.approx(0.4)
    empty = ev.divergence_summary(ev.DivergenceAccumulator())
    assert empty["specialty"]["l1_mean"] is None and empty["movement"]["disagreement"] is None


def test_first_exceedance_uses_baseline_floor_and_sustain() -> None:
    series = {1: 0.10, 2: 0.10, 3: 0.12, 4: 0.10, 5: 0.10, 6: 0.15, 7: 0.30, 8: 0.35, 9: 0.40}
    result = ev.first_exceedance(series)
    assert result["baseline"] == pytest.approx(0.104)
    assert result["threshold"] == pytest.approx(0.208)
    assert result["first_turn"] == 7
    assert result["first_sustained_turn"] == 7
    # A one-turn spike is the first exceedance but is not sustained.
    spike = ev.first_exceedance({1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.5, 7: 0.1, 8: 0.1})
    assert spike["first_turn"] == 6 and spike["first_sustained_turn"] is None
    # A perfectly agreeing head is floored: noise below 2 x floor does not trip it.
    quiet = ev.first_exceedance({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.015, 7: 0.03})
    assert quiet["threshold"] == pytest.approx(0.02)
    assert quiet["first_turn"] == 7
    assert ev.first_exceedance({6: 0.5})["first_turn"] is None  # no baseline data


def test_teacher_and_actor_adapters() -> None:
    label = SimpleNamespace(
        action={"fission": 2, "tradeGate": [1, 0]},
        specialty_counts=_counts(Gatherer=10),
        specialty_valid=True,
        activity=_activity({}),
        activity_role_valid=np.ones(ROLE_ROWS, dtype=bool),
        movement_code=-1,
        movement_valid=True,
        fission_valid=True,
        trade_gate_valid=np.array([True, False]),
    )
    teacher = ev.teacher_from_label(label, 10)
    assert teacher.fission == 2 and teacher.trade_gate.tolist() == [1, 0]
    assert teacher.trade_gate_valid.tolist() == [True, False]
    attempt = SimpleNamespace(
        valid=True,
        fallback=False,
        accepted_specialty_residents=tuple(int(v) for v in _counts(Gatherer=10)),
        requested_activity_allocation=tuple(map(tuple, _activity({}))),
    )
    payload = {"movement": -1, "fission": 2, "tradeGate": [1, 0]}
    actor = ev.actor_from_execution(payload, attempt, 10)
    assert actor.activity is not None and actor.fission == 2
    fallback = SimpleNamespace(**dict(vars(attempt), fallback=True))
    assert ev.actor_from_execution(payload, fallback, 10).activity is None
    assert ev.camp_divergence(teacher, actor)["specialty"]["l1"] == pytest.approx(0.0)


def test_divergence_report_pools_worlds_and_keeps_per_world_copies() -> None:
    def world(seed: int, drifted_farmers: int) -> dict:
        accs = {}
        for turn in range(1, 9):
            acc = ev.DivergenceAccumulator()
            farmers = drifted_farmers if turn >= 6 else 4
            result = ev.camp_divergence(
                _teacher(),
                _actor(
                    specialty_counts=_counts(
                        Gatherer=10 - farmers - 2, Farmer=farmers, Craftsperson=2
                    )
                ),
            )
            acc.add(result, {"residents": 10.0, "housing_w": 5.0, "clothing_w": 5.0})
            accs[turn] = acc
        turns = [
            {
                "turn": t + 9,
                "decision": t,
                "farmer_share": 0.2,
                "housing_coverage": 0.5,
                "clothing_coverage": 0.5,
                "live_settlements": 3,
                "residents": 30,
                "qol": 0.1,
            }
            for t in range(1, 9)
        ]
        return {"world": seed, "turn_accumulators": accs, "actor_turns": turns}

    report = ev.divergence_report([world(2000, 0), world(2001, 4)])
    assert set(report["per_world"]) == {"2000", "2001"}
    # World 2000 drifts from turn 6; world 2001 never does; the pooled series sits in between.
    assert report["per_world"]["2000"]["heads"]["specialty"]["first_turn"] == 6
    assert report["per_world"]["2001"]["heads"]["specialty"]["first_turn"] is None
    assert report["heads"]["specialty"]["series"]["6"] == pytest.approx(
        0.5 * report["per_world"]["2000"]["heads"]["specialty"]["series"]["6"]
    )
    assert report["window"]["turns"] == [5, 25]
    assert report["actor_turn_world_mean"]["3"]["worlds"] == 2
    assert "specialty" in ev.summarize_divergence(dict(report, worlds=[2000, 2001], turns=8))


def test_divergence_world_policy_and_defaults() -> None:
    assert ev.SCALEOUT_RESET_DEFAULTS == {
        "players": 16,
        "width": 128,
        "height": 64,
        "separation": 9,
    }
    ev.check_divergence_worlds([2000, 2001], False)
    for bad in ([3000], [90154], [90159]):
        with pytest.raises(ValueError):
            ev.check_divergence_worlds(bad, True)
    with pytest.raises(ValueError):
        ev.check_divergence_worlds([4000], False)
    ev.check_divergence_worlds([4000], True)
    args = ev.argument_parser().parse_args(
        ["actor-divergence", "--checkpoint", "x.pt", "--worlds", "2000", "--out", "o.json"]
    )
    assert (args.players, args.width, args.height, args.separation) == (16, 128, 64, 9)
    assert args.turns == 30 and args.withheld == 0 and args.window == [5, 25]
