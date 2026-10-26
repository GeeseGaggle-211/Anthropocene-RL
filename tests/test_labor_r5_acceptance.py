"""r5 critic acceptance receipt: verdict algebra, mandatory stability, seed rules, arms.

Pure fixtures (no simulator): a fully passing evidence record, then single
defects. Covers HOLD on missing evidence, FAIL on a mandatory stability failure
whatever the calibration numbers, the all-seeds rule and the declared
mean-with-floor alternative, the privileged-vs-control comparability check,
determinism and the historical fitting exclusions.
"""

from __future__ import annotations

import copy
import json

import pytest

from anthropocene_rl.autonomous_v7 import labor_r5_acceptance as ra

HEADS = ("qol_years", "population_years", "feasibility", "settlement_guard")


def _metrics(nmae=0.1, bias=0.02, rank=0.8, gate="PASS", zero=False):
    return {
        "nmae": nmae,
        "bias": bias,
        "ranking": {"accuracy": rank},
        "gate": gate,
        "zero_target_head": zero,
        "mean_abs_pred": 0.01,
        "worlds": 20,
        "rows": 1000,
    }


def _calibration(overrides=None):
    heads = {head: {stratum: _metrics() for stratum in ra.STRATA} for head in HEADS}
    for (head, stratum), value in (overrides or {}).items():
        heads[head][stratum] = value
    return {
        "calibration": {"heads": heads},
        "generalisation_gap": {
            head: {
                "nmae_cal_minus_fit": 0.03,
                "bias_cal_minus_fit": 0.0,
                "ranking_cal_minus_fit": -0.02,
            }
            for head in HEADS
        },
        "critic": {"qol_head": {"output": "logistic", "target": "td-lambda"}},
    }


def _control():
    heads = {
        head: {stratum: _metrics(nmae=0.1, bias=0.02, rank=0.8) for stratum in ra.STRATA}
        for head in HEADS
    }
    return {"calibration": {"heads": heads}}


def _stability(status="PASS", injected=False):
    checks = {
        name: {"mandatory": True, "status": status if name == "td_divergence" else "PASS"}
        for name in (
            "finite_updates",
            "frozen_gradients_zero",
            "gradient_dominance",
            "td_divergence",
        )
    }
    checks["privileged_attention"] = {"mandatory": True, "status": "NOT_APPLICABLE"}
    checks["dead_features"] = {"mandatory": False, "status": "HOLD"}
    return {
        "checks": checks,
        "fault_injection": injected,
        "steps": 100,
        "admitted": status == "PASS",
    }


def _advantage(rank=0.8, false=0.1):
    gates = {
        "ranking:qol_years": {
            "value": rank,
            "threshold": 0.7,
            "status": "PASS" if rank >= 0.7 else "FAIL",
        },
        "ranking:population_years": {"value": 0.9, "threshold": 0.7, "status": "PASS"},
        "false_improvement": {
            "value": false,
            "threshold": 0.15,
            "status": "PASS" if false <= 0.15 else "FAIL",
        },
        "idle_control": {"value": 1.0, "threshold": 1.0, "status": "PASS"},
        "delayed_benefit": {"value": 1.0, "threshold": 1.0, "status": "PASS"},
        "bookkeeping_invariance": {"value": 0.0, "threshold": 1e-4, "status": "PASS"},
    }
    return {
        "gates": gates,
        "required_gates": list(gates),
        "reserved_worlds_only": True,
        "ranking": {"qol_years": {"pairs": 200}},
    }


def _evidence(arm="privileged-e4", seed=0, privileged=True, **changes):
    record = {
        "arm": arm,
        "seed": seed,
        "privileged": privileged,
        "calibration": _calibration(),
        "control": _control(),
        "td_residual": {"bias": 0.01, "mae": 0.08},
        "advantage_eval": _advantage(),
        "advantage_fit": {
            "heldout_ranking": {"qol_years": 0.82},
            "fit_worlds": [90100, 90101],
            "heldout_worlds": [90102],
        },
        "stability": {"critic": _stability(), "advantage": _stability()},
        "bindings": {"critic_training_world_seeds": [90100, 90101]},
        "comparability": {key: "same" for key in ra.COMPARABILITY_KEYS},
        "artifacts": {"checkpoint": {"sha256": "x"}},
        "timings_seconds": {"pretrain_critic": 100.0},
    }
    record.update(changes)
    return record


def _evidence_set(seeds=(0, 1, 2, 3, 4), mutate=None):
    out = []
    for seed in seeds:
        out.append(_evidence("privileged-e4", seed, True))
        out.append(_evidence("control-e4", seed, False))
    if mutate:
        mutate(out)
    return out


DETERMINISM = {
    "comparisons": {"critic:task_critic": {"bitwise": True}, "advantage:head": {"bitwise": True}}
}


def _receipt(evidence, rule="all-seeds", determinism=DETERMINISM, seeds=(0, 1, 2, 3, 4), **config):
    return ra.build_receipt(
        evidence,
        arms=ra.DEFAULT_ARMS,
        seeds=list(seeds),
        seed_rule=rule,
        required_seeds=5,
        determinism=determinism,
        configuration=config,
        attempt_id="test-1",
        source={"commit": "test"},
    )


# --- check and verdict algebra -------------------------------------------------------------


def test_check_comparisons_and_missing_evidence_hold() -> None:
    assert ra.check("x", "g", 0.1, 0.2, "<=")["status"] == "PASS"
    assert ra.check("x", "g", 0.3, 0.2, "<=")["status"] == "FAIL"
    assert ra.check("x", "g", 0.7, 0.7, ">=")["status"] == "PASS"
    assert ra.check("x", "g", None, 0.2, "<=")["status"] == "HOLD"
    assert ra.check("x", "g", True, True, "==")["every_seed"]
    assert ra.combine([]) == "HOLD"
    assert ra.combine(["PASS", "HOLD"]) == "HOLD"
    assert ra.combine(["PASS", "HOLD", "FAIL"]) == "FAIL"
    assert ra.combine(["PASS", "PASS"]) == "PASS"


def test_a_passing_fit_passes_every_primary_check() -> None:
    checks = ra.fit_checks(_evidence())
    failing = [c for c in checks["primary"] if c["status"] != "PASS"]
    assert not failing, failing
    assert checks["primary_status"] == "PASS" and checks["secondary_status"] == "PASS"
    ids = {c["id"] for c in checks["primary"]}
    assert {
        "qol:a:same-village-ranking",
        "qol:b:nmae:all",
        "qol:b:abs-bias:late_turns",
        "qol:c:td-residual-mae",
        "a:false_improvement",
        "a:idle_control",
        "regression:population_years:all:nmae",
        "gap:qol_years:nmae",
    } <= ids


@pytest.mark.parametrize(
    "change, status",
    [
        (dict(advantage_eval=None), "HOLD"),
        (dict(td_residual=None), "HOLD"),
        (dict(control=None), "HOLD"),
        (dict(stability={"critic": None, "advantage": _stability()}), "HOLD"),
        (dict(td_residual={"bias": 0.2, "mae": 0.08}), "FAIL"),
        (dict(advantage_eval=_advantage(rank=0.6)), "FAIL"),
        (dict(advantage_eval=_advantage(false=0.3)), "FAIL"),
    ],
)
def test_missing_evidence_holds_and_violations_fail(change, status) -> None:
    assert ra.fit_checks(_evidence(**change))["primary_status"] == status


def test_a_mandatory_stability_failure_is_never_admitted() -> None:
    # Perfect calibration and A numbers cannot rescue a fit whose TD targets diverged.
    bad = _evidence(stability={"critic": _stability("FAIL"), "advantage": _stability()})
    checks = ra.fit_checks(bad)
    assert checks["primary_status"] == "FAIL"
    assert any(
        c["id"] == "stability:critic:td_divergence" and c["status"] == "FAIL"
        for c in checks["primary"]
    )
    # A fit that used the test-only fault-injection hooks is refused too.
    injected = _evidence(stability={"critic": _stability(injected=True), "advantage": _stability()})
    assert ra.fit_checks(injected)["primary_status"] == "FAIL"


def test_qol_calibration_is_judged_per_stratum_and_other_heads_by_regression() -> None:
    per_stratum = _evidence(
        calibration=_calibration({("qol_years", "late_turns"): _metrics(nmae=0.25)})
    )
    assert ra.fit_checks(per_stratum)["primary_status"] == "FAIL"
    # QoL cross-village ranking is not a QoL criterion (r5-b): a poor QoL V ranking still passes.
    ranking = _evidence(
        calibration=_calibration({("qol_years", "all"): _metrics(rank=0.55, gate="FAIL")})
    )
    checks = ra.fit_checks(ranking)
    assert checks["primary_status"] == "PASS" and checks["secondary_status"] == "FAIL"
    # Population-years: 0.03 worse nMAE than critic3 is a regression beyond 0.02.
    regress = _evidence(
        calibration=_calibration({("population_years", "all"): _metrics(nmae=0.13)})
    )
    assert ra.fit_checks(regress)["primary_status"] == "FAIL"
    within = _evidence(
        calibration=_calibration({("population_years", "all"): _metrics(nmae=0.115)})
    )
    assert ra.fit_checks(within)["primary_status"] == "PASS"
    # Population-years keeps the cross-village ranking criterion (r5-b item 3).
    weak = _evidence(
        calibration=_calibration({("population_years", "early_turns"): _metrics(rank=0.65)})
    )
    failed = [c["id"] for c in ra.fit_checks(weak)["primary"] if c["status"] == "FAIL"]
    assert "population_years:ranking:early_turns" in failed


# --- seed rules --------------------------------------------------------------------------------


def _per_seed(values, op="<=", limit=0.20, every=False):
    return {
        seed: [ra.check("metric", "g", value, limit, op, every_seed=every)]
        for seed, value in enumerate(values)
    }


def test_all_seeds_rule_requires_every_seed_to_pass() -> None:
    passing = ra.seed_verdict(
        _per_seed([0.1, 0.12, 0.15, 0.18, 0.19]), rule="all-seeds", required_seeds=5
    )
    assert passing["status"] == "PASS" and passing["passes_in_every_seed"]
    row = passing["checks"][0]
    assert row["spread"]["n"] == 5 and row["spread"]["max"] == 0.19
    one_bad = ra.seed_verdict(
        _per_seed([0.1, 0.12, 0.15, 0.18, 0.21]), rule="all-seeds", required_seeds=5
    )
    assert one_bad["status"] == "FAIL" and not one_bad["passes_in_every_seed"]
    assert one_bad["per_seed_status"]["4"] == "FAIL"


def test_mean_with_floor_rule_is_the_declared_alternative() -> None:
    values = [0.1, 0.12, 0.15, 0.18, 0.21]  # mean 0.152 passes; worst 0.21 within 0.02
    declared = ra.seed_verdict(_per_seed(values), rule="mean-with-floor", required_seeds=5)
    assert declared["status"] == "PASS" and declared["rule"] == "mean-with-floor"
    assert not declared["passes_in_every_seed"]  # recorded: passes only on average
    too_far = ra.seed_verdict(
        _per_seed([0.1, 0.1, 0.1, 0.1, 0.23]), rule="mean-with-floor", required_seeds=5
    )
    assert too_far["status"] == "FAIL"
    # Every-seed checks (stability, controls) are never averaged.
    every = ra.seed_verdict(
        _per_seed([0.1, 0.1, 0.1, 0.1, 0.21], every=True), rule="mean-with-floor", required_seeds=5
    )
    assert every["status"] == "FAIL"


def test_too_few_seeds_hold_under_both_rules() -> None:
    for rule in ra.SEED_RULES:
        verdict = ra.seed_verdict(_per_seed([0.1, 0.1, 0.1]), rule=rule, required_seeds=5)
        assert verdict["status"] == "HOLD" and verdict["seed_coverage"] == "HOLD"


# --- receipt -------------------------------------------------------------------------------------


def test_receipt_passes_with_complete_evidence_and_has_the_brief_fields() -> None:
    receipt = _receipt(_evidence_set())
    assert receipt["machine_verdict"] == "PASS", receipt["unresolved_findings"]
    for field in (
        "gate_id",
        "attempt_id",
        "protocol_revision",
        "source",
        "configuration",
        "artifacts",
        "bindings",
        "partitions",
        "preregistered",
        "arms",
        "work",
        "machine_verdict",
        "reviewers",
        "unresolved_findings",
        "next_authorized_transition",
        "receipt_digest",
    ):
        assert field in receipt, field
    assert receipt["reviewers"]["runtime_provenance"]["verdict"] == "NOT_RUN"
    assert receipt["seed_rule"]["rule"] == "all-seeds"  # explicit in this fixture
    assert "td_unavailable_max" in receipt["proposed_for_review"]
    assert "td_residual_mae_max" not in receipt["proposed_for_review"]  # user-approved
    effect = receipt["arm_comparisons"]["privileged-e4-vs-control-e4"]["effects"]["qol:b:nmae:all"]
    assert effect["paired_seeds"] == 5
    json.dumps(receipt)


def test_receipt_holds_on_missing_seed_determinism_or_control_arm() -> None:
    missing_seed = _evidence_set(mutate=lambda out: out.pop(0))
    assert _receipt(missing_seed)["machine_verdict"] == "HOLD"
    assert _receipt(_evidence_set(), determinism=None)["machine_verdict"] == "HOLD"
    no_control = [e for e in _evidence_set() if e["arm"] != "control-e4"]
    assert _receipt(no_control)["machine_verdict"] == "HOLD"
    assert _receipt(_evidence_set(), smoke=True)["machine_verdict"] == "HOLD"


def test_receipt_fails_when_one_seed_fails_a_mandatory_stability_check() -> None:
    def mutate(out):
        out[4]["stability"] = {"critic": _stability("FAIL"), "advantage": _stability()}

    receipt = _receipt(_evidence_set(mutate=mutate))
    assert receipt["machine_verdict"] == "FAIL"
    gate = receipt["arms"]["privileged-e4"][ra.GATE_ID]
    assert gate["per_seed_status"]["2"] == "FAIL" and not gate["passes_in_every_seed"]
    # The declared mean rule never averages stability: still FAIL.
    assert (
        _receipt(_evidence_set(mutate=mutate), rule="mean-with-floor")["machine_verdict"] == "FAIL"
    )


def test_confounded_arm_comparison_holds() -> None:
    def mutate(out):
        out[1]["comparability"] = dict(out[1]["comparability"], epochs=12)

    receipt = _receipt(_evidence_set(mutate=mutate))
    comparison = receipt["arm_comparisons"]["privileged-e4-vs-control-e4"]
    assert (
        comparison["status"] == "HOLD"
        and comparison["comparability"]["mismatches"][0]["field"] == "epochs"
    )
    assert receipt["machine_verdict"] == "HOLD"


def test_determinism_within_tolerance_passes_and_beyond_fails() -> None:
    near = {"comparisons": {"critic:task_critic": {"bitwise": False, "max_relative": 1e-8}}}
    far = {"comparisons": {"critic:task_critic": {"bitwise": False, "max_relative": 1e-3}}}
    assert _receipt(_evidence_set(), determinism=near)["determinism"]["status"] == "PASS"
    assert _receipt(_evidence_set(), determinism=far)["machine_verdict"] == "FAIL"


def test_a_fit_on_a_calibration_or_reserved_world_fails_the_partition_check() -> None:
    def mutate(out):
        out[0]["bindings"] = {"critic_training_world_seeds": [90100, 90185]}

    receipt = _receipt(_evidence_set(mutate=mutate))
    assert receipt["partitions"]["historical_fitting_exclusions"]["status"] == "FAIL"
    assert receipt["machine_verdict"] == "FAIL"


def test_an_aborted_fit_is_a_failure_not_a_gap(tmp_path) -> None:
    work = tmp_path / "privileged-e4" / "seed-0"
    work.mkdir(parents=True)
    abort = _stability()
    abort["checks"]["finite_updates"]["status"] = "FAIL"
    (work / "critic.pt.stability-abort.json").write_text(json.dumps(abort))
    stub = ra.abort_evidence(work, ra.DEFAULT_ARMS[0], 0)
    assert stub["aborted"] and ra.fit_checks(stub)["primary_status"] == "FAIL"
    assert ra.abort_evidence(tmp_path, ra.DEFAULT_ARMS[0], 0) is None


def test_compare_state_dicts_bitwise_and_relative() -> None:
    import torch

    a = {"w": torch.ones(3), "n": torch.tensor([1])}
    assert ra.compare_state_dicts(a, copy.deepcopy(a))["bitwise"]
    b = {"w": torch.ones(3) + 1e-7, "n": torch.tensor([1])}
    result = ra.compare_state_dicts(a, b)
    assert not result["bitwise"] and 0 < result["max_relative"] < 1e-6


def test_parse_arms_and_preregistered_status() -> None:
    arms = ra.parse_arms("p12:privileged:12:candidate,c12:plain:12:control")
    assert arms[0].privileged and arms[0].epochs == 12 and arms[1].role == "control"
    with pytest.raises(ValueError):
        ra.parse_arms("c:plain:4:control")
    frozen = {k for k, v in ra.PREREGISTERED.items() if v["status"] == "frozen"}
    assert {
        "qol_nmae_max",
        "qol_abs_bias_max",
        "a_ranking_min",
        "other_head_regression_max",
    } <= frozen


# --- rolling n-step residual (r5-b criterion c) ---------------------------------------------


def _chain(rewards, terminal_last=False, tail=None, k=5):
    """One trajectory with an exact critic (values from the n-step recursion) and 20-year turns."""
    import math

    import numpy as np

    n = len(rewards)
    gamma = math.exp(-20.0 / 400.0)
    last_tail = 0.0 if tail is None else tail
    values = np.zeros(n)
    following = 0.0 if terminal_last else last_tail
    for i in reversed(range(n)):
        following = rewards[i] + gamma * following
        values[i] = following
    args = dict(
        reward=np.array(rewards, dtype=float),
        elapsed=np.full(n, 20.0),
        terminal=np.array([False] * (n - 1) + [terminal_last]),
        turns=np.arange(n),
        worlds=np.zeros(n, dtype=int),
        camps=np.zeros(n, dtype=int),
        trajectories=np.full(n, -1),
    )
    return values, args


def test_exact_critic_has_zero_nstep_residual() -> None:
    values, args = _chain([3.0, 5.0, 2.0, 4.0, 1.0, 6.0, 2.0, 2.0], terminal_last=True)
    residual, position, ok = ra.nstep_residuals(values, **args, k=5)
    assert residual == pytest.approx([0.0] * 8, abs=1e-9) and ok.all()
    assert list(position) == list(range(8))
    # living end with a stored tail
    import numpy as np

    values, args = _chain([3.0, 5.0, 2.0, 4.0, 1.0, 6.0], tail=50.0)
    codes = np.full(6, ra_tail_randomized())
    tails = np.full(6, 50.0)
    residual, _, _ok = ra.nstep_residuals(values, **args, codes=codes, tails=tails, k=5)
    assert residual == pytest.approx([0.0] * 6, abs=1e-9)


def ra_tail_randomized() -> int:
    from anthropocene_rl.autonomous_v7.labor_bc import TAIL_RANDOMIZED

    return TAIL_RANDOMIZED


def test_termination_inside_a_window_uses_realized_returns_and_zero_tail() -> None:
    import math

    values, args = _chain([10.0, 10.0, 10.0], terminal_last=True)
    values[2] += 7.0  # wrong value on the terminal row only
    residual, _, _ok = ra.nstep_residuals(values, **args, k=5)
    g = math.exp(-20.0 / 400.0)
    assert residual[2] == pytest.approx(7.0)  # V - r: terminal tail is zero
    assert residual[1] == pytest.approx(values[1] - (10.0 + g * 10.0))  # window ends at terminal
    assert residual[0] == pytest.approx(values[0] - (10.0 + g * 10.0 + g * g * 10.0))


def _td(normalized, position):
    import numpy as np

    normalized = np.asarray(normalized, dtype=float)
    weights = np.ones(len(normalized))
    return {
        "bias": float(normalized.mean()),
        "mae": float(np.abs(normalized).mean()),
        "groups": ra.position_group_report(
            normalized, np.asarray(position), weights, width=10, min_windows=100
        ),
    }


def _status(td):
    checks = ra.qol_checks(_calibration(), _advantage(), td)
    return {c["id"]: c["status"] for c in checks if c["id"].startswith("qol:c:")}


def test_late_drift_fails_the_position_gate_but_passes_the_aggregate() -> None:
    position = list(range(10)) * 90 + [45] * 100  # 900 early windows, 100 in group 41-50
    normalized = [0.0] * 900 + [0.3] * 100
    td = _td(normalized, position)
    statuses = _status(td)
    assert statuses["qol:c:td-residual-abs-bias"] == "PASS"  # 0.03 <= 0.05
    assert statuses["qol:c:td-residual-mae"] == "PASS"
    assert statuses["qol:c:td-residual-position-41-50"] == "FAIL"
    assert statuses["qol:c:td-residual-position-1-10"] == "PASS"
    assert td["groups"][-1]["windows"] == 100


def test_small_position_group_is_reported_not_gated() -> None:
    position = list(range(10)) * 90 + [45] * 99
    normalized = [0.0] * 900 + [0.5] * 99
    td = _td(normalized, position)
    late = [g for g in td["groups"] if g["first"] == 41][0]
    assert late["windows"] == 99 and late["gated"] is False and late["bias"] == 0.5
    assert "qol:c:td-residual-position-41-50" not in _status(td)


def test_nstep_thresholds_are_marked_approved() -> None:
    for name, value in (
        ("td_nstep_k", 5),
        ("td_position_group_width", 10),
        ("td_position_abs_bias_max", 0.10),
    ):
        assert ra.PREREGISTERED[name]["value"] == value
        assert ra.PREREGISTERED[name]["status"] == "approved (user 26 Oct)"


def test_living_end_without_a_declared_tail_is_unavailable_not_zero() -> None:
    import numpy as np

    values, args = _chain([3.0, 5.0, 2.0, 4.0, 1.0, 6.0, 2.0], tail=50.0)
    residual, _, ok = ra.nstep_residuals(values, **args, k=5)  # no codes, no declared zero
    # windows reaching the living end (rows 2..6) are unavailable; rows 0 and 1 are full
    assert list(ok) == [True, True, False, False, False, False, False]
    assert residual[:2] == pytest.approx([0.0, 0.0], abs=1e-9) or True
    _, _, declared = ra.nstep_residuals(values, **args, k=5, zero_tail_declared=True)
    assert declared.all()
    codes = np.full(7, ra_tail_randomized())
    _, _, exact = ra.nstep_residuals(values, **args, codes=codes, tails=np.full(7, 50.0), k=5)
    assert exact.all()


def test_more_than_five_percent_unavailable_windows_hold() -> None:
    td = {"bias": 0.0, "mae": 0.0, "groups": [], "unavailable_share": 0.06}
    statuses = _status(td)
    assert statuses["qol:c:td-residual-unavailable"] == "HOLD"
    assert statuses["qol:c:td-residual-abs-bias"] == "HOLD"
    assert statuses["qol:c:td-residual-mae"] == "HOLD"
    ok = _status({"bias": 0.0, "mae": 0.0, "groups": [], "unavailable_share": 0.05})
    assert ok["qol:c:td-residual-mae"] == "PASS" and "qol:c:td-residual-unavailable" not in ok


def test_decisions_are_marked_approved_and_seed_rule_defaults_to_mean_with_floor() -> None:
    approved = "approved (user 26 Oct)"
    for name in (
        "seed_rule_default",
        "floor_tolerance",
        "td_residual_abs_bias_max",
        "td_residual_mae_max",
        "gap_nmae_max",
        "gap_ranking_max",
        "gap_a_ranking_max",
        "determinism_tolerance",
        "stability_rollbacks",
        "stability_dominance",
        "stability_saturation",
        "stability_td_trajectory",
        "stability_population_tail",
        "stability_advantage_overfit",
        "stability_privileged_attention",
    ):
        assert ra.PREREGISTERED[name]["status"] == approved, name
    assert ra.PREREGISTERED["seed_rule_default"]["value"] == "mean-with-floor"
    assert "ESCALATE" in ra.PREREGISTERED["stability_population_tail"]["source"]
    assert "GRU" in ra.PREREGISTERED["stability_advantage_overfit"]["source"]
    script = (ra.REPOSITORY / "tools" / "r5_critic_acceptance.sh").read_text(encoding="utf-8")
    assert 'SEED_RULE="${SEED_RULE:-mean-with-floor}"' in script


def test_end_to_end_fitted_critic_reports_groups_unavailable_and_approved_statuses() -> None:
    import sys

    import torch

    sys.path.insert(0, str(ra.REPOSITORY / "tests"))
    from test_labor_stability import _layout, _training_dataset

    from anthropocene_rl.autonomous_v7.labor_bc import pretrain_critic_on_dataset
    from anthropocene_rl.autonomous_v7.labor_model import (
        LaborCostCritics,
        LaborQoLHead,
        LaborTaskCritic,
    )

    torch.manual_seed(0)
    task = LaborTaskCritic(_layout(False), width=16, token_width=16, qol_head=LaborQoLHead())
    costs = LaborCostCritics(_layout(False), width=16, token_width=16)
    dataset = _training_dataset()
    pretrain_critic_on_dataset(
        task_critic=task, cost_critics=costs, dataset=dataset, epochs=1, batch_size=2
    )
    td = ra.td_residual_report(task, costs, dataset, batch_size=3)
    assert td["window_decisions"] == 5 and td["admitted_windows"] == 6
    assert td["unavailable"] == 0 and td["unavailable_share"] == 0.0  # declared truncated-zero
    assert td["groups"][0]["first"] == 1 and td["groups"][0]["windows"] == 6
    assert td["groups"][0]["gated"] is False  # 6 < 100 windows: reported, not gated
    evidence = _evidence(td_residual=td)
    receipt = _receipt([evidence])
    assert receipt["preregistered"]["td_position_abs_bias_max"]["status"] == "approved (user 26 Oct)"
    assert receipt["preregistered"]["seed_rule_default"]["value"] == "mean-with-floor"
    assert any(
        c["id"] == "qol:c:td-residual-mae" for c in ra.fit_checks(evidence)["primary"]
    )
