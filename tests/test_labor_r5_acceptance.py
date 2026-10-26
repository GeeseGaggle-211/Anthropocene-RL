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
    assert receipt["seed_rule"]["rule"] == "all-seeds"
    assert "td_residual_mae_max" in receipt["proposed_for_review"]
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
