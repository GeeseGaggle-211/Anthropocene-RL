"""Gate relative-v1 and the r5 critic level / loader fixes (user and reviewer, 28 Oct).

1. Relative-first gates: (a) within-trajectory ordering at lag >= 5 with a 0.02 W margin,
   (b) all-pairs ordering, (c) the TD(0) residual, per head and for the combined elder
   reward PPO consumes; absolute QoL calibration demoted to diagnostics printed next to a
   constant-predictor reference. Evidence without ``gate_version`` keeps gate v2.
2. QoL head bias init at logit(mean p*) and step-0 TD(lambda) targets from the Monte Carlo
   returns (``--qol-head-bias-init mean-target``, ``--td-step0-targets mc``).
3. Per-world tail normalisation (``--tail-normalize world-factor``).
4. ``check_support`` tolerance; 5. ``--campaign-r5`` encoder default (test_labor_qol_encoder).
6. False-terminal correction (``--false-terminal-policy mask``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from test_labor_qol_encoder import _dataset as _fit_dataset
from test_labor_qol_encoder import _pair
from test_labor_r5_acceptance import _evidence, _receipt

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7 import labor_advantage as la
from anthropocene_rl.autonomous_v7 import labor_bc
from anthropocene_rl.autonomous_v7 import labor_pretrain_eval as pe
from anthropocene_rl.autonomous_v7 import labor_r5_acceptance as ra
from anthropocene_rl.autonomous_v7 import labor_training as lt
from anthropocene_rl.autonomous_v7.labor_bc import (
    TAIL_HEADS,
    TAIL_NONE_TERMINAL,
    TAIL_RANDOMIZED,
    TAIL_TERMINAL_EXACT,
    correct_false_terminals,
    discounted_returns,
    normalize_world_tails,
    pretrain_critic_on_dataset,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    LABOR_COSTS,
    LABOR_OBJECTIVE_REFERENCE_SCALES,
    LABOR_OBJECTIVES,
    LaborPersonality,
    LaborQoLHead,
    LaborTaskCritic,
    LaborValueSupport,
    labor_qol_discount_weight,
)
from anthropocene_rl.autonomous_v7.labor_ppo import combine_labor_advantages

GAMMA = math.exp(-contracts.TURN_YEARS / contracts.HORIZON_YEARS)
W = labor_qol_discount_weight()


# --- synthetic trajectories ---------------------------------------------------------------------


def _trajectories(spec, *, rewards=None):
    """``spec``: (world, camp, turns, terminal_turns, tail_code, tail) per trajectory."""
    worlds, camps, turns, terminal, codes, tails = [], [], [], [], [], []
    for world, camp, turn_list, terminal_turns, code, tail in spec:
        for turn in turn_list:
            worlds.append(world)
            camps.append(camp)
            turns.append(turn)
            terminal.append(turn in terminal_turns)
            codes.append(code)
            tails.append(float(tail))
    n = len(worlds)
    reward = torch.full((n,), 10.0, dtype=torch.float64) if rewards is None else rewards
    data = {
        "row_world": torch.tensor(worlds, dtype=torch.long),
        "row_camp": torch.tensor(camps, dtype=torch.long),
        "row_turn": torch.tensor(turns, dtype=torch.long),
        "row_quarantined": torch.zeros(n, dtype=torch.bool),
        "transition_elapsed_years": torch.full((n,), contracts.TURN_YEARS, dtype=torch.float64),
        "transition_terminal": torch.tensor(terminal, dtype=torch.bool),
        "reward_qol_years": reward,
        "reward_population_years": 3.0 * reward,
        "reward_feasibility": torch.zeros(n, dtype=torch.float64),
        "reward_settlement_guard": torch.zeros(n, dtype=torch.float64),
        "tail_method": torch.tensor(codes, dtype=torch.long),
        "tail_cutoff_turn": torch.zeros(n, dtype=torch.long),
    }
    for head in TAIL_HEADS:
        scale = {"qol_years": 1.0, "population_years": 3.0}.get(head, 0.0)
        data["tail_value__" + head] = torch.tensor(tails, dtype=torch.float64) * scale
        data["tail_standard_error__" + head] = torch.full((n,), 2.0, dtype=torch.float64)
    data["collection"] = {}
    return data


def _ordering(values, targets, worlds=None, traj=None, position=None, **kwargs):
    n = len(values)
    return pe.within_trajectory_ordering(
        np.asarray(values, float),
        np.asarray(targets, float),
        np.zeros(n, int) if traj is None else traj,
        np.arange(n) if position is None else position,
        np.zeros(n, int) if worlds is None else worlds,
        **kwargs,
    )["all"]


# --- 1. gate metrics ----------------------------------------------------------------------------


def test_perfect_reversed_and_constant_critics() -> None:
    rng = np.random.default_rng(0)
    target = np.cumsum(rng.normal(0, 30.0, 40))
    margin = 0.02 * W
    perfect = _ordering(target, target, margin=margin, min_lag=5)
    reversed_ = _ordering(-target, target, margin=margin, min_lag=5)
    constant = _ordering(np.full(40, 7.0), target, margin=margin, min_lag=5)
    assert perfect["accuracy"] == 1.0 and reversed_["accuracy"] == 0.0
    assert constant["accuracy"] == 0.5  # ties score one half
    # Pair counts: lag >= 5 among 40 rows; informative ones beyond the margin.
    assert perfect["pairs"] == sum(40 - lag for lag in range(5, 40))
    d = target[None, :] - target[:, None]
    lag = np.arange(40)[None, :] - np.arange(40)[:, None]
    assert perfect["informative_pairs"] == int(((lag >= 5) & (np.abs(d) > margin)).sum())
    every = _ordering(target, target, margin=margin, min_lag=1)
    assert every["pairs"] == 40 * 39 // 2
    assert every["median_trajectory_spearman"] == pytest.approx(1.0)


def test_ordering_is_world_balanced_and_strata_take_the_earlier_row() -> None:
    target = np.arange(20, dtype=float) * 20.0  # every pair informative
    values = np.concatenate([target[:10], -target[10:]])  # world 0 right, world 1 reversed
    worlds = np.array([0] * 10 + [1] * 10)
    traj = worlds.copy()
    position = np.concatenate([np.arange(10), np.arange(10)])
    report = pe.within_trajectory_ordering(
        values, target, traj, position, worlds, margin=0.0, min_lag=1,
        strata={"all": np.ones(20, bool), "first": position < 2},
    )
    assert report["all"]["accuracy"] == pytest.approx(0.5)
    assert report["all"]["worlds"] == 2
    # Pairs anchored at positions 0 and 1: 9 + 8 per world.
    assert report["first"]["pairs"] == 2 * (9 + 8)
    # Unbalanced pools: world 1 with more rows dominates the pooled but not the balanced mean.
    worlds2 = np.array([0] * 5 + [1] * 15)
    traj2 = worlds2.copy()
    pos2 = np.concatenate([np.arange(5), np.arange(15)])
    values2 = np.concatenate([target[:5], -target[5:]])
    r2 = pe.within_trajectory_ordering(values2, target, traj2, pos2, worlds2, margin=0.0)["all"]
    assert r2["accuracy"] == pytest.approx(0.5)
    assert r2["accuracy_pooled"] == pytest.approx(10 / (10 + 105))


def test_trajectory_positions_follow_turn_order_per_trajectory() -> None:
    traj, position = pe.trajectory_positions(
        [1, 1, 2, 1], [5, 5, 5, 5], [-1, -1, -1, -1], [3, 1, 0, 2]
    )
    assert traj.tolist() == [0, 0, 1, 0]
    assert position.tolist() == [2, 0, 0, 1]


def _all_heads(n, base):
    rng = np.random.default_rng(3)
    return {
        "qol_years": base,
        "population_years": 50.0 * base + rng.normal(0, 1, n),
        "feasibility": rng.uniform(0, 0.1, n),
        "settlement_guard": rng.uniform(0, 0.1, n),
    }


def test_combined_variant_is_the_ppo_combination() -> None:
    n = 12
    heads = _all_heads(n, np.linspace(0, 300, n))
    combined = pe.combined_elder_values(heads)
    expected = combine_labor_advantages(
        {k: torch.as_tensor(heads[k]) for k in LABOR_OBJECTIVES},
        {k: torch.as_tensor(heads[k]) for k in LABOR_COSTS},
        personality=LaborPersonality(),
        cost_multipliers=dict.fromkeys(LABOR_COSTS, 1.0),
    ).numpy()
    assert np.array_equal(combined, expected)
    s_q = LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"]
    s_p = LABOR_OBJECTIVE_REFERENCE_SCALES["population_years"]
    manual = (
        0.5 * heads["qol_years"] / s_q
        + 0.5 * heads["population_years"] / s_p
        - heads["feasibility"]
        - heads["settlement_guard"]
    )
    assert np.allclose(combined, manual)
    margins = pe.relative_margins(W)
    assert margins["qol_years"] == pytest.approx(0.02 * W)
    assert margins["combined"] == pytest.approx(0.5 * 0.02 * W / s_q)
    assert margins["population_years"] == pytest.approx(0.02 * W * s_p / s_q)


def test_relative_report_on_a_dataset_and_combined_ordering() -> None:
    data = _trajectories(
        [(90100 + w, c, list(range(12)), (), TAIL_RANDOMIZED, 200.0) for w in range(3)
         for c in (1, 2)],
        rewards=torch.tensor(
            np.random.default_rng(1).uniform(0, 20, 72), dtype=torch.float64
        ),
    )
    # Population-years on its own scale (the 0.02 W QoL-equivalent margin is ~16,000).
    data["reward_population_years"] = 5000.0 * data["reward_qol_years"]
    data["tail_value__population_years"] = 5000.0 * data["tail_value__qol_years"]
    returns = {k: v.double().numpy() for k, v in discounted_returns(data).items()}
    rows = np.arange(72)
    strata = {"all": np.ones(72, bool)}
    perfect = pe.relative_ordering_report(data, rows, returns, returns, strata, weight=W)
    flipped = {k: -v for k, v in returns.items()}
    worst = pe.relative_ordering_report(data, rows, flipped, returns, strata, weight=W)
    for name in pe.RELATIVE_VARIANTS:
        assert perfect["variants"][name]["lag_ge_min"]["all"]["accuracy"] == 1.0
        assert worst["variants"][name]["all_pairs"]["all"]["accuracy"] == 0.0
    assert perfect["combination"]["function"] == "labor_ppo.combine_labor_advantages"
    # A critic right on QoL but reversed on population: per-head variants split, and the
    # combined variant follows the PPO weights: 0.5 x 5000 G / s_pop (0.029 G) outweighs
    # 0.5 G / s_qol (0.012 G), so the combination tracks population and is reversed.
    mixed = dict(returns, population_years=-returns["population_years"])
    report = pe.relative_ordering_report(data, rows, mixed, returns, strata, weight=W)
    v = report["variants"]
    assert v["qol_years"]["all_pairs"]["all"]["accuracy"] == 1.0
    assert v["population_years"]["all_pairs"]["all"]["accuracy"] == 0.0
    assert v["combined"]["all_pairs"]["all"]["accuracy"] == 0.0
    qol_heavy = pe.relative_ordering_report(
        data, rows, mixed, returns, strata, weight=W,
        personality=LaborPersonality(qol_years=0.9, population_years=0.1),
    )
    assert qol_heavy["variants"]["combined"]["all_pairs"]["all"]["accuracy"] == 1.0


def test_constant_predictor_reference_matches_a_constant_critic() -> None:
    data = _trajectories([(90100 + w, 1, list(range(15)), (), TAIL_RANDOMIZED, 300.0)
                          for w in range(3)])
    n = int(data["row_world"].shape[0])
    rows = np.arange(n)
    mc = discounted_returns(data)["qol_years"].double().numpy()
    strata = {"all": np.ones(n, bool)}
    c = 250.0
    with_ref = pe.qol_nstep_calibration(
        data, rows, np.full(n, 123.0), mc, strata, pe.frozen_qol_reference(),
        constant={"value": c, "source": "test"},
    )
    as_critic = pe.qol_nstep_calibration(
        data, rows, np.full(n, c), mc, strata, pe.frozen_qol_reference()
    )
    ref = with_ref["constant_reference"]["strata"]["all"]
    assert ref["nstep"]["nmae"] == pytest.approx(as_critic["strata"]["all"]["nmae"])
    assert ref["nstep"]["bias"] == pytest.approx(as_critic["strata"]["all"]["bias"])
    assert ref["monte_carlo"]["nmae"] == pytest.approx(
        as_critic["strata"]["all"]["monte_carlo"]["nmae"]
    )
    # Without a constant the report is the historical one.
    plain = pe.qol_nstep_calibration(
        data, rows, np.full(n, 123.0), mc, strata, pe.frozen_qol_reference()
    )
    assert "constant_reference" not in plain


def test_td0_residual_is_zero_for_the_exact_critic_and_excludes_the_living_end() -> None:
    data = _trajectories([(90100 + w, 1, list(range(10)), (), TAIL_RANDOMIZED, 0.0)
                          for w in range(3)])
    n = int(data["row_world"].shape[0])
    v_true = np.full(n, 10.0 / (1.0 - GAMMA))
    keep = np.ones(n, bool)
    exact = ra.td0_residual_report(v_true, data, keep, 43.152, constant=100.0)
    assert exact["bias"] == pytest.approx(0.0, abs=1e-9)
    assert exact["mae"] == pytest.approx(0.0, abs=1e-9)
    assert exact["rows"] == n - 3 and exact["excluded_living_end_rows"] == 3
    offset = ra.td0_residual_report(v_true + 43.152, data, keep, 43.152)
    assert offset["bias"] == pytest.approx(1.0 - GAMMA)  # a uniform offset c shows as (1-g) c
    ref = exact["constant_reference"]
    assert ref["bias"] == pytest.approx((100.0 * (1 - GAMMA) - 10.0) / 43.152)


# --- 1b. acceptance: relative-v1 verdicts; old evidence keeps gate v2 ------------------------------


def _relative(acc_a=0.75, acc_b=0.70, pairs=5000, worlds=8):
    def section(acc):
        return {
            stratum: {"accuracy": acc, "informative_pairs": pairs, "worlds": worlds}
            for stratum in ra.STRATA
        }

    return {
        "min_lag": 5,
        "variants": {
            v: {"margin": 1.0, "lag_ge_min": section(acc_a), "all_pairs": section(acc_b),
                "all_pairs_no_margin": {"all": {"accuracy": 0.6}}}
            for v in ra.RELATIVE_VARIANTS
        },
    }


def _relative_evidence(acc_a=0.75, acc_b=0.70, td0=(0.005, 0.09), **kwargs):
    record = _evidence(**kwargs)
    record["gate_version"] = ra.GATE_RELATIVE_V1
    record["calibration"]["calibration"]["relative"] = _relative(acc_a, acc_b)
    record["td_residual"] = dict(record["td_residual"], td0={"bias": td0[0], "mae": td0[1]})
    return record


def test_old_evidence_keeps_gate_v2_verdicts() -> None:
    old = ra.fit_checks(_evidence())
    assert old["gate_version"] == ra.GATE_V2 and "diagnostics" not in old
    ids = {c["id"] for c in old["primary"]}
    assert "qol:b:nmae:all" in ids and not any(i.startswith("rel:") for i in ids)
    bad_nmae = _evidence()
    bad_nmae["calibration"]["calibration"]["heads"]["qol_years"]["all"]["nmae"] = 0.5
    assert ra.fit_checks(bad_nmae)["primary_status"] == "FAIL"


def test_relative_v1_demotes_absolute_qol_checks_and_gates_relative_ones() -> None:
    good = ra.fit_checks(_relative_evidence())
    assert good["gate_version"] == ra.GATE_RELATIVE_V1
    assert good["primary_status"] == "PASS", [c for c in good["primary"] if c["status"] != "PASS"]
    ids = {c["id"] for c in good["primary"]}
    # Only the combined elder reward gates (a)/(b); the per-head variants are diagnostics.
    assert {"rel:a:lag-ordering:combined", "rel:b:all-pairs-ordering:combined"} <= ids
    assert not any(i.endswith((":qol_years", ":population_years")) for i in ids if
                   i.startswith("rel:"))
    heads = {d["id"]: d for d in good["diagnostics"]["relative_heads"]}
    assert set(heads) == {
        f"rel:{g}:{v}" for g in ("a:lag-ordering", "b:all-pairs-ordering")
        for v in ("qol_years", "population_years")
    }
    assert all(d["gating"] is False for d in heads.values())
    assert {"rel:c:td0-abs-bias", "rel:c:td0-mae", "qol:a:same-village-ranking",
            "stability:critic:td_divergence", "a:false_improvement"} <= ids
    assert not any(i.startswith(("qol:b:nmae", "qol:b:abs-bias", "qol:c:")) for i in ids)
    assert "gap:qol_years:nmae" not in ids
    demoted = {d["id"] for d in good["diagnostics"]["demoted_gate_v2_checks"]}
    assert "qol:b:nmae:all" in demoted and "gap:qol_years:nmae" in demoted
    # An absolute QoL failure no longer fails a relative-v1 fit; it is reported.
    bad_abs = _relative_evidence()
    bad_abs["calibration"]["calibration"]["heads"]["qol_years"]["all"]["nmae"] = 0.5
    checks = ra.fit_checks(bad_abs)
    assert checks["primary_status"] == "PASS"
    row = [d for d in checks["diagnostics"]["demoted_gate_v2_checks"]
           if d["id"] == "qol:b:nmae:all"][0]
    assert row["would_have_been"] == "FAIL"
    # Relative failures fail; thin coverage holds; missing relative evidence holds.
    assert ra.fit_checks(_relative_evidence(acc_a=0.69))["primary_status"] == "FAIL"
    # A failing per-head variant alone does not fail the fit.
    head_only = _relative_evidence()
    head_only["calibration"]["calibration"]["relative"]["variants"]["qol_years"][
        "lag_ge_min"]["all"]["accuracy"] = 0.5
    checks = ra.fit_checks(head_only)
    assert checks["primary_status"] == "PASS"
    assert {d["id"]: d for d in checks["diagnostics"]["relative_heads"]}[
        "rel:a:lag-ordering:qol_years"]["would_have_been"] == "FAIL"
    assert ra.fit_checks(_relative_evidence(td0=(0.005, 0.5)))["primary_status"] == "FAIL"
    thin = _relative_evidence()
    thin["calibration"]["calibration"]["relative"] = _relative(pairs=50)
    assert ra.fit_checks(thin)["primary_status"] == "HOLD"
    missing = _relative_evidence()
    del missing["calibration"]["calibration"]["relative"]
    assert ra.fit_checks(missing)["primary_status"] == "HOLD"


def test_relative_thresholds_are_preregistered_and_marked() -> None:
    p = ra.PREREGISTERED
    assert p["rel_lag_ordering_min"]["value"] == 0.70
    assert p["rel_lag_ordering_min"]["status"].startswith("approved")
    assert p["rel_lag_min"]["value"] == pe.RELATIVE_MIN_LAG == 5
    assert p["rel_margin_w"]["value"] == pe.RELATIVE_MARGIN_W == 0.02
    assert pe.RELATIVE_VARIANTS == ra.RELATIVE_VARIANTS
    assert tuple(p["rel_variants"]["value"]) == ra.RELATIVE_GATED_VARIANTS == ("combined",)
    assert p["rel_all_pairs_ordering_min"]["value"] == 0.65
    assert p["rel_all_pairs_ordering_min"]["status"].startswith("approved")
    for name in ("rel_td0_abs_bias_max", "rel_td0_mae_max", "adv_sign_agreement_min",
                 "adv_slope_range", "adv_std_ratio_range"):
        assert p[name]["status"].startswith("proposed")
    assert p["a_ranking_min"]["value"] == 0.70  # (d) unchanged


def test_receipt_holds_on_mixed_gate_versions() -> None:
    def mutate(records):
        for record in records:
            record["gate_version"] = ra.GATE_RELATIVE_V1
            record["calibration"]["calibration"]["relative"] = _relative()
            record["td_residual"]["td0"] = {"bias": 0.0, "mae": 0.09}

    from test_labor_r5_acceptance import _evidence_set

    receipt = _receipt(_evidence_set(mutate=mutate))
    assert receipt["gate_versions"] == [ra.GATE_RELATIVE_V1]
    assert receipt["machine_verdict"] == "PASS", receipt["unresolved_findings"]
    mixed = _evidence_set(mutate=mutate)
    mixed[0].pop("gate_version")
    receipt = _receipt(mixed)
    assert receipt["machine_verdict"] == "HOLD"
    assert any(f.get("check") == "gate-version-consistent" for f in receipt["unresolved_findings"])


# --- 1c. advantage magnitude on the combined reward (non-gating) ---------------------------------


def _level_world(trajectories=300, length=20, seed=0):
    """Camps with a hidden QoL level m plus i.i.d. noise, ending terminal (exact returns).
    V*(t) = m sum_{j < remaining} gamma^j; population and cost rewards are zero."""
    rng = np.random.default_rng(seed)
    levels = rng.uniform(2.0, 18.0, trajectories)
    spec, rewards, truth = [], [], []
    for index, m in enumerate(levels):
        spec.append((90100 + index % 6, index, list(range(length)), (length - 1,),
                     TAIL_NONE_TERMINAL, 0.0))
        rewards.extend(m + rng.normal(0.0, 1.0, length))
        truth.extend(m * sum(GAMMA**j for j in range(length - p)) for p in range(length))
    data = _trajectories(spec, rewards=torch.tensor(rewards, dtype=torch.float64))
    data["reward_population_years"] = torch.zeros_like(data["reward_qol_years"])
    for head in TAIL_HEADS:
        data["tail_value__" + head] = torch.zeros_like(data["tail_value__" + head])
    return data, np.asarray(truth)


def _magnitude(data, qol_values):
    n = len(qol_values)
    returns = {k: v.double().numpy() for k, v in discounted_returns(data).items()}
    values = {"qol_years": qol_values, "population_years": np.zeros(n),
              "feasibility": np.zeros(n), "settlement_guard": np.zeros(n)}
    turns = data["row_turn"].numpy()
    strata = {"all": np.ones(n, bool), "early_turns": turns < 10, "late_turns": turns >= 10}
    return pe.advantage_magnitude_report(data, np.arange(n), values, returns, strata)


def test_advantage_magnitude_perfect_scaled_and_constant_critics() -> None:
    data, truth = _level_world()
    perfect = _magnitude(data, truth)
    scaled = _magnitude(data, 10.0 * truth)
    constant = _magnitude(data, np.full(len(truth), truth.mean()))
    lo, hi = ra.threshold("adv_slope_range")
    for k in ("1", "5"):
        p, x, c = (r["ks"][k]["all"] for r in (perfect, scaled, constant))
        assert p["slope"] == pytest.approx(1.0, abs=0.1)  # E[A_MC | window] = A_hat exactly
        assert lo <= p["slope"] <= hi
        # x10 values: the window's value terms dominate A_hat; A_MC shares -V(s_t), so the
        # slope leaves the bound while the sign agreement is trivially high (both sides are
        # about -9 V): sign agreement alone cannot see a scale error.
        assert x["slope"] > hi
        assert x["sign_agreement"] > 0.95
        # Constant: A_hat misses the level signal: lower sign agreement, slope off 1.
        assert c["sign_agreement"] < p["sign_agreement"]
        assert abs(c["slope"] - 1.0) > abs(p["slope"] - 1.0)
        assert set(perfect["ks"][k]) == {"all", "early_turns", "late_turns"}
    # A uniform V offset: the raw sign agreement is trivially 1 (A_hat and A_MC both carry
    # -offset); the world-centred form removes it and falls to chance.
    offset = _magnitude(data, truth + 300.0)
    for k in ("1", "5"):
        o = offset["ks"][k]["all"]
        assert o["sign_agreement"] > 0.99
        assert 0.4 < o["sign_agreement_world_centred"] < 0.6
    # Documented limitation: A_MC = G - V shares the critic's V, so a constant critic (whose
    # A_hat and A_MC both carry the hidden level) still agrees on sign at least as often as
    # the exact critic once centred. The sign check is not a ground-truth advantage test.
    assert (constant["ks"]["5"]["all"]["sign_agreement_world_centred"]
            >= perfect["ks"]["5"]["all"]["sign_agreement_world_centred"])
    # Only QoL moves: it carries the whole combined spread, predicted and realised.
    heads = perfect["ks"]["5"]["all"]["heads"]
    assert heads["qol_years"]["combined_contribution_share"] == pytest.approx(1.0)
    assert heads["qol_years"]["realised_contribution_share"] == pytest.approx(1.0)
    s_q = LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"]
    assert perfect["coefficients"]["qol_years"] == pytest.approx(0.5 / s_q)
    assert perfect["coefficients"]["feasibility"] == pytest.approx(-1.0)
    popart = heads["qol_years"]["popart"]
    assert popart["target"]["std"] > 0 and popart["a_hat"]["std"] > 0
    assert heads["qol_years"]["advantage_to_value_spread"] == pytest.approx(
        popart["a_hat"]["std"] / popart["value"]["std"]
    )
    # The advisory reads the combined k-step numbers; the scaled critic is flagged.
    def advisory(report):
        return ra.advantage_magnitude_advisory(
            {"calibration": {"advantage_magnitude": report}}
        )

    assert advisory(scaled)["status"] == "FLAG"
    assert advisory(perfect)["rows"][0]["slope"] == pytest.approx(
        perfect["ks"]["1"]["all"]["slope"]
    )
    assert advisory(perfect)["gating"] is False


# --- 2. QoL head bias init and step-0 targets ---------------------------------------------------


def _fit(task, costs, data, **kwargs):
    return pretrain_critic_on_dataset(
        task_critic=task, cost_critics=costs, dataset=data, epochs=1, batch_size=2,
        rng=np.random.default_rng(7), stability=None, **kwargs,
    )


def _state(task, costs):
    return {**{f"t.{k}": v for k, v in task.state_dict().items()},
            **{f"c.{k}": v for k, v in costs.state_dict().items()}}


def test_flags_off_is_the_historical_fit() -> None:
    a_task, a_costs = _pair("shared", seed=0, target="td-lambda")
    a = _fit(a_task, a_costs, _fit_dataset())
    b_task, b_costs = _pair("shared", seed=0, target="td-lambda")
    b = _fit(b_task, b_costs, _fit_dataset(), qol_head_bias_init=None, td_step0_targets=None)
    sa, sb = _state(a_task, a_costs), _state(b_task, b_costs)
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    for key in ("qol_head_bias_init", "td_step0_targets", "tail_normalization",
                "false_terminal_correction"):
        assert key not in a and key not in b
    assert a["final_loss"] == b["final_loss"]
    data = _fit_dataset()
    assert normalize_world_tails(data, "none") is data
    assert correct_false_terminals(data, "none") is data
    assert labor_bc.apply_loader_corrections(data) is data


def test_bias_init_sets_logit_of_the_mean_target_fraction() -> None:
    task, costs = _pair("shared", seed=0, target="td-lambda")
    data = _fit_dataset()
    summary = _fit(task, costs, data, qol_head_bias_init="mean-target", max_steps=0)
    record = summary["qol_head_bias_init"]
    returns = discounted_returns(data)["qol_years"].double()
    p = (returns / (task.qol_head.discount_weight * task.qol_head.scale)).clamp(0, 1)
    # Two worlds of three rows: world-balanced sampling equals the plain mean here.
    assert record["mean_target_fraction"] == pytest.approx(float(p.mean()))
    expected = math.log(record["mean_target_fraction"] / (1 - record["mean_target_fraction"]))
    assert float(task.heads["qol_years"].bias.detach()) == pytest.approx(expected, rel=1e-6)
    assert record["bias"] == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError, match="logistic"):
        silu, silu_costs = _pair("shared", seed=0, output="silu-clamp")
        _fit(silu, silu_costs, _fit_dataset(), qol_head_bias_init="mean-target")


def test_step0_targets_come_from_monte_carlo(monkeypatch) -> None:
    calls = []
    original = la.qol_td_lambda_targets

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(la, "qol_td_lambda_targets", spy)
    task, costs = _pair("shared", seed=0, target="td-lambda")
    _fit(task, costs, _fit_dataset(), max_steps=0)
    assert len(calls) == 1  # historical step-0 bootstrap through the untrained head
    calls.clear()
    task, costs = _pair("shared", seed=0, target="td-lambda")
    summary = _fit(task, costs, _fit_dataset(), max_steps=0, td_step0_targets="mc")
    assert calls == []  # Monte Carlo targets at step 0
    assert summary["td_step0_targets"]["mode"] == "mc"
    # Later refreshes are unchanged: they run on schedule.
    task, costs = _pair("shared", seed=0, target="td-lambda")
    _fit(task, costs, _fit_dataset(), td_step0_targets="mc", td_refresh_steps=1)
    assert len(calls) == 2  # three steps per epoch: refreshes at steps 1 and 2
    with pytest.raises(ValueError, match="TD"):
        mc_task, mc_costs = _pair("shared", seed=0)
        _fit(mc_task, mc_costs, _fit_dataset(), td_step0_targets="mc")


def test_level_fixes_resume_exactly(tmp_path) -> None:
    def run(path, max_steps=None):
        task, costs = _pair("shared", seed=0, target="td-lambda")
        summary = pretrain_critic_on_dataset(
            task_critic=task, cost_critics=costs, dataset=_fit_dataset(), epochs=2,
            batch_size=2, rng=np.random.default_rng(7), stability=None,
            qol_head_bias_init="mean-target", td_step0_targets="mc", td_refresh_steps=2,
            resume_path=None if path is None else str(path), resume_every=1,
            max_steps=max_steps,
        )
        return _state(task, costs), summary

    full, summary = run(None)
    resume = tmp_path / "resume.pt"
    run(resume, max_steps=1)
    resumed, summary_r = run(resume)
    assert all(torch.equal(full[k], resumed[k]) for k in full)
    assert summary_r["qol_head_bias_init"] == summary["qol_head_bias_init"]


def _cli(*extra):
    return lt.argument_parser().parse_args(
        ["pretrain-critic", "--checkpoint", "c.pt", "--out", "o.pt", *extra]
    )


def test_cli_defaults_on_only_under_campaign_r5() -> None:
    td_head = LaborQoLHead(target="td-lambda")
    state = {"labor_model_contract": None}

    def kwargs(args, head=td_head, trained=False):
        import unittest.mock as mock

        st = dict(state, **({"heuristic_critic_pretraining": {}} if trained else {}))
        with mock.patch.object(lt, "labor_checkpoint_qol_head", lambda s: head):
            return lt._critic_level_kwargs(args, st)

    assert kwargs(_cli()) == {}
    assert lt._false_terminal_policy(_cli()) == "none"
    assert _cli().tail_normalize == "none"
    assert kwargs(_cli("--campaign-r5")) == {
        "qol_head_bias_init": "mean-target", "td_step0_targets": "mc",
    }
    assert lt._false_terminal_policy(_cli("--campaign-r5")) == "mask"
    assert kwargs(_cli("--campaign-r5", "--qol-head-bias-init", "none",
                       "--td-step0-targets", "head")) == {}
    explicit = _cli("--campaign-r5", "--false-terminal-policy", "none")
    assert lt._false_terminal_policy(explicit) == "none"
    assert kwargs(_cli("--campaign-r5"), trained=True) == {}  # warm start: head trained
    with pytest.raises(ValueError, match="untrained"):
        kwargs(_cli("--qol-head-bias-init", "mean-target"), trained=True)
    assert kwargs(_cli("--campaign-r5"), head=LaborQoLHead()) == {
        "qol_head_bias_init": "mean-target"
    }  # Monte Carlo head: no step-0 rule
    assert kwargs(_cli("--campaign-r5"), head=None) == {}


# --- 3. per-world tail normalisation -------------------------------------------------------------


def _with_schedules(data, stops):
    data["collection"] = {
        "return_tail": labor_bc.RETURN_TAIL_RANDOMIZED,
        "return_tail_by_world": {str(w): {"rho": 0.95, "stop_indices": s} for w, s in stops.items()},
    }
    return data


def test_tail_normalisation_divides_randomized_tails_by_the_world_factor() -> None:
    data = _with_schedules(
        _trajectories([
            (90100, 1, list(range(5)), (), TAIL_RANDOMIZED, 300.0),
            (90100, 2, list(range(5)), (), TAIL_TERMINAL_EXACT, 50.0),
            (90101, 1, list(range(5)), (), TAIL_RANDOMIZED, 300.0),
            (90101, 2, list(range(5)), (4,), TAIL_NONE_TERMINAL, 0.0),
        ]),
        {90100: [3, 40, 7], 90101: [60, 2]},
    )
    out = normalize_world_tails(data, "world-factor")
    f = {w: pe.tail_schedule_factor(data["collection"]["return_tail_by_world"][str(w)], W)
         for w in (90100, 90101)}
    record = out["collection"]["tail_normalization"]
    assert record["factors"] == {str(w): pytest.approx(v) for w, v in f.items()}
    assert record["rows_normalized"] == 10
    for head, scale in (("qol_years", 1.0), ("population_years", 3.0)):
        tail = out["tail_value__" + head]
        assert torch.allclose(tail[:5], torch.full((5,), 300.0 * scale / f[90100]).double())
        assert torch.equal(tail[5:10], data["tail_value__" + head][5:10])  # exact: unchanged
        assert torch.allclose(tail[10:15], torch.full((5,), 300.0 * scale / f[90101]).double())
        se = out["tail_standard_error__" + head]
        assert float(se[0]) == pytest.approx(2.0 / f[90100])
    # Cost heads and the input dataset are untouched; the result is recorded and refused twice.
    assert torch.equal(out["tail_value__feasibility"], data["tail_value__feasibility"])
    assert "tail_normalization" not in data["collection"]
    assert labor_bc.tail_normalization_mode(out) == "world-factor"
    with pytest.raises(ValueError, match="already"):
        normalize_world_tails(out, "world-factor")
    missing = _with_schedules(_trajectories([(90102, 1, [0, 1], (), TAIL_RANDOMIZED, 9.0)]), {})
    with pytest.raises(ValueError, match="schedule"):
        normalize_world_tails(missing, "world-factor")
    # The fit records it and refuses a validation set normalised differently.
    loss_returns = discounted_returns(out)["qol_years"]
    assert float(loss_returns[4]) == pytest.approx(10.0 + GAMMA * 300.0 / f[90100], rel=1e-6)


# --- 4. check_support tolerance -----------------------------------------------------------------


def test_check_support_tolerates_gpu_roundoff_only() -> None:
    from test_labor_privileged_model import _layout

    task = LaborTaskCritic(_layout(False), width=16, token_width=16,
                           support=LaborValueSupport(-0.5, 2.5, 255))
    state = task.state_dict()
    near = dict(state, bins=state["bins"] + 2.4e-7)
    task.load_state_dict(near, strict=True)
    task.check_support()  # loads
    far = dict(state, bins=state["bins"] + 1e-4)
    task.load_state_dict(far, strict=True)
    with pytest.raises(ValueError, match="bins differ"):
        task.check_support()


# --- 6. false terminals -----------------------------------------------------------------------------


def test_false_terminal_with_a_gap_masks_the_rows_through_it() -> None:
    data = _trajectories([
        (90104, 24, [7, 8, 9, 11, 12, 13], (9,), TAIL_RANDOMIZED, 100.0),
        (90104, 25, [7, 8, 9], (9,), TAIL_NONE_TERMINAL, 0.0),  # a true terminal
    ])
    before = discounted_returns(data)["qol_years"].clone()
    out = correct_false_terminals(data, "mask")
    record = out["collection"]["false_terminal_correction"]
    assert record["false_terminal_rows"] == 1 and record["value_masked_rows"] == 3
    assert record["events"][0]["action"] == "masked-through-row"
    assert not bool(out["transition_terminal"][2]) and bool(out["transition_terminal"][8])
    available = labor_bc.critic_target_available(out)
    assert available.tolist() == [False, False, False, True, True, True, True, True, True]
    after = discounted_returns(out)["qol_years"]
    assert torch.equal(after[3:], before[3:])  # later rows and other camps unchanged
    assert labor_bc.critic_target_available(data).all()  # the input is untouched


def test_false_terminal_without_a_gap_is_rechained_exactly() -> None:
    data = _trajectories([(90104, 24, [7, 8, 9, 10, 11], (9,), TAIL_RANDOMIZED, 100.0)])
    out = correct_false_terminals(data, "mask")
    record = out["collection"]["false_terminal_correction"]
    assert record["rechained_rows"] == 1 and record["value_masked_rows"] == 0
    assert labor_bc.VALUE_MASK_FIELD not in out
    clean = _trajectories([(90104, 24, [7, 8, 9, 10, 11], (), TAIL_RANDOMIZED, 100.0)])
    assert torch.equal(discounted_returns(out)["qol_years"], discounted_returns(clean)["qol_years"])


def test_false_terminal_policy_is_recorded_by_the_fit() -> None:
    data = _fit_dataset()
    # Camp 1 is marked terminal at turn 0 and reappears at turn 2 (turn 1 missing).
    data["row_turn"] = torch.tensor([0, 2, 3, 0, 1, 2])
    data["transition_terminal"] = torch.tensor([True, False, True, False, False, True])
    corrected = correct_false_terminals(data, "mask")
    task, costs = _pair("shared", seed=0)
    summary = _fit(task, costs, corrected, max_steps=0)
    assert summary["false_terminal_correction"]["value_masked_rows"] == 1
    assert summary["rows"] == 5


B01 = os.environ.get("ANTHROPOCENE_R5_B01_DIR")


@pytest.mark.skipif(not B01 or not Path(B01).is_dir(), reason="needs the local fit8-b01 copy")
def test_b01_false_terminal_row_is_masked() -> None:
    from anthropocene_rl.autonomous_v7.labor_teacher_data import load_labor_datasets

    data = load_labor_datasets(
        [Path(B01)], expected_normalizer_digest=None, require_privileged=False,
        drop_reserved=False, merge_teacher=True,
    )
    out = correct_false_terminals(data, "mask")
    record = out["collection"]["false_terminal_correction"]
    assert record["false_terminal_rows"] == 1
    event = record["events"][0]
    assert (event["world"], event["camp"], event["turn"], event["next_turn"]) == (90104, 24, 9, 11)
    world = data["row_world"].reshape(-1)
    camp = data["row_camp"].reshape(-1)
    turn = data["row_turn"].reshape(-1)
    mine = (world == 90104) & (camp == 24)
    expected = mine & (turn <= 9)
    assert record["value_masked_rows"] == int(expected.sum())
    available = labor_bc.critic_target_available(out)
    assert not available[expected].any()
    before = discounted_returns(data)
    after = discounted_returns(out)
    later = ~expected
    for head in LABOR_OBJECTIVES:
        assert torch.equal(after[head][later], before[head][later])


def test_explicit_loader_corrections_override_the_checkpoint_record() -> None:
    trained = {"heuristic_critic_pretraining": {}}
    assert pe.resolve_loader_corrections(trained) == ({}, "checkpoint")
    active, source = pe.resolve_loader_corrections(
        trained, {"tail_normalize": "world-factor", "false_terminal_policy": "mask"}
    )
    assert active == {"tail_normalize": "world-factor", "false_terminal_policy": "mask"}
    assert source == "explicit-override"
    recorded = {"heuristic_critic_pretraining": {"tail_normalization": {"mode": "world-factor"}}}
    assert pe.resolve_loader_corrections(recorded, {"tail_normalize": None})[0] == {
        "tail_normalize": "world-factor"
    }
    assert pe.resolve_loader_corrections(recorded, {"tail_normalize": "none"}) == (
        {}, "explicit-override"
    )
    args = pe.argument_parser().parse_args(
        ["critic-calibration", "--checkpoint", "c.pt", "--dataset", "d", "--out", "o",
         "--tail-normalize", "world-factor", "--false-terminal-policy", "mask"]
    )
    assert (args.tail_normalize, args.false_terminal_policy) == ("world-factor", "mask")
    evaluate = ra.argument_parser().parse_args(
        ["evaluate", "--arm", "a", "--seed", "0", "--checkpoint", "c", "--advantage", "a",
         "--calibration-data", "x", "--reserved-payloads", "p", "--out", "o",
         "--tail-normalize", "world-factor"]
    )
    assert evaluate.tail_normalize == "world-factor" and evaluate.false_terminal_policy is None
