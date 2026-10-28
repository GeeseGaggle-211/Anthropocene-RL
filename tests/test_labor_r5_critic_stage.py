"""``labor_r5_acceptance run --stage critic``: V fits and their evaluation without payloads.

Pure orchestration fixtures (no simulator, no real fit): the stage refuses incomplete
inputs, skips the A fit, evaluates with ``--stage critic`` into ``evidence-critic.json``,
probes critic determinism only, writes ``critic-receipt.json`` with a V-only summary
whose A-dependent checks are deferred (never PASS), and a later ``--stage all`` in the
same output continues at the A fit without refitting the critic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_r5_acceptance as ra
from tests.test_labor_r5_acceptance import _evidence, _stability

CRITIC_DETERMINISM = {
    "comparisons": {
        "critic:task_critic": {"bitwise": True},
        "critic:cost_critics": {"bitwise": True},
    }
}


def _args(tmp_path: Path, *extra: str):
    for name in ("fit-a", "fit-b", "cal"):
        (tmp_path / name).mkdir(exist_ok=True)
    return ra.argument_parser().parse_args(
        [
            "run",
            "--normalizer-checkpoint", str(tmp_path / "n.pt"),
            "--privileged-normalizer", str(tmp_path / "p.pt"),
            "--fit-data", str(tmp_path / "fit-a"), str(tmp_path / "fit-b"),
            "--calibration-data", str(tmp_path / "cal"),
            "--attempt-id", "critic-stage-test",
            "--out", str(tmp_path / "out"),
            *extra,
        ]
    )


def _critic_evidence(arm: str, seed: int, privileged: bool) -> dict:
    record = _evidence(arm, seed, privileged)
    record.update(
        stage="critic",
        advantage_eval=None,
        advantage_fit=None,
        stability={"critic": _stability(), "advantage": None},
    )
    return record


# --- argument validation -------------------------------------------------------------------


def test_stage_all_still_requires_both_payload_sets(tmp_path) -> None:
    args = _args(tmp_path)
    assert args.stage == "all" and args.fit_payloads is None
    with pytest.raises(ValueError, match="--stage all needs --fit-payloads"):
        ra.require_run_inputs(args)
    only_fit = _args(tmp_path, "--fit-payloads", str(tmp_path / "fp"))
    with pytest.raises(ValueError, match="--stage all needs"):
        ra.require_run_inputs(only_fit)
    ra.require_run_inputs(_args(tmp_path, "--stage", "critic"))  # no payloads needed


def test_gap_fit_data_must_be_a_subset_of_the_fitting_data(tmp_path) -> None:
    args = _args(tmp_path, "--stage", "critic", "--gap-fit-data", str(tmp_path / "fit-a"))
    ra.require_run_inputs(args)
    assert ra._evaluation_fit_data(args) == [tmp_path / "fit-a"]
    assert ra._evaluation_fit_data(_args(tmp_path, "--stage", "critic")) == [
        tmp_path / "fit-a",
        tmp_path / "fit-b",
    ]
    bad = _args(tmp_path, "--stage", "critic", "--gap-fit-data", str(tmp_path / "cal"))
    with pytest.raises(ValueError, match="--gap-fit-data must list --fit-data"):
        ra.require_run_inputs(bad)


def test_evaluate_refuses_mixed_stage_inputs(tmp_path) -> None:
    parser = ra.argument_parser()
    base = [
        "evaluate", "--arm", "a", "--seed", "0", "--checkpoint", "c.pt",
        "--calibration-data", "cal", "--out", str(tmp_path / "e.json"),
    ]
    with pytest.raises(ValueError, match="needs --advantage and --reserved-payloads"):
        ra.command_evaluate(parser.parse_args(base))
    with pytest.raises(ValueError, match="V critic only"):
        ra.command_evaluate(parser.parse_args([*base, "--stage", "critic", "--advantage", "a.pt"]))
    with pytest.raises(ValueError, match="go together"):
        ra.evaluate_fit(
            arm="a", seed=0, checkpoint=Path("c.pt"), advantage=None,
            calibration_data=Path("cal"), fit_data=None, reserved_payloads=[Path("r")],
        )


# --- run_fit -------------------------------------------------------------------------------


def _record_commands(monkeypatch) -> list[list[str]]:
    """Replace subprocess execution: record each command and create its output file."""
    commands: list[list[str]] = []

    def run(command, log, threads):
        command = [str(c) for c in command]
        commands.append(command)
        out = Path(command[command.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")

    monkeypatch.setattr(ra, "_run", run)
    monkeypatch.setattr(ra, "_stamp_fit_settings", lambda path, settings: None)
    return commands


def _module(command: list[str]) -> str:
    return command[command.index("-m") + 1].rsplit(".", 1)[-1] + ":" + command[
        command.index("-m") + 2
    ]


def test_critic_stage_skips_the_a_fit_and_evaluates_v_only(tmp_path, monkeypatch) -> None:
    commands = _record_commands(monkeypatch)
    args = _args(tmp_path, "--stage", "critic", "--gap-fit-data", str(tmp_path / "fit-a"))
    arm = ra.DEFAULT_ARMS[0]
    path = ra.run_fit(args, arm, 0)
    assert path.name == "evidence-critic.json" and path.exists()
    assert [_module(c) for c in commands] == [
        "labor_training:initialize",
        "labor_training:pretrain-critic",
        "labor_r5_acceptance:evaluate",
    ]
    evaluate = commands[-1]
    assert evaluate[evaluate.index("--stage") + 1] == "critic"
    assert "--advantage" not in evaluate and "--reserved-payloads" not in evaluate
    fit = evaluate[evaluate.index("--fit-data") + 1 : evaluate.index("--timings")]
    assert fit == [str(tmp_path / "fit-a")]  # the gap subset, not every fitting batch
    # The V fit is the same command as in a full run (validation on CAL_DATA, early stop).
    critic = commands[1]
    assert "--stability-early-stop" in critic and "--validation-dataset" in critic
    assert critic.count("--dataset") == 2

    # A later full run in the same output reuses the critic and continues at the A fit.
    commands.clear()
    full = _args(
        tmp_path,
        "--fit-payloads", str(tmp_path / "fp"),
        "--reserved-payloads", str(tmp_path / "rp"),
    )
    path = ra.run_fit(full, arm, 0)
    assert path.name == "evidence.json"
    assert [_module(c) for c in commands] == [
        "labor_advantage:fit",
        "labor_r5_acceptance:evaluate",
    ]
    evaluate = commands[-1]
    assert evaluate[evaluate.index("--advantage") + 1].endswith("advantage.pt")
    assert "--stage" not in evaluate
    assert (tmp_path / "out" / arm.name / "seed-0" / "evidence-critic.json").exists()


def test_critic_stage_determinism_probe_has_no_a_probe(tmp_path, monkeypatch) -> None:
    def run(command, log, threads):
        command = [str(c) for c in command]
        assert "labor_advantage" not in " ".join(command)
        out = Path(command[command.index("--out") + 1])
        torch.save({"task_critic": {"w": torch.ones(2)}, "cost_critics": {"w": torch.zeros(2)}}, out)

    monkeypatch.setattr(ra, "_run", run)
    args = _args(tmp_path, "--stage", "critic")
    path = ra.run_determinism(args, ra.DEFAULT_ARMS[0], 0)
    assert path.name == "determinism-critic.json"
    report = json.loads(path.read_text())
    assert report["stage"] == "critic" and report["advantage_epochs"] is None
    assert sorted(report["comparisons"]) == ["critic:cost_critics", "critic:task_critic"]
    assert all(c["bitwise"] for c in report["comparisons"].values())


# --- command_run and the critic-stage receipt ---------------------------------------------------


def _fake_critic_fits(monkeypatch, *, fail_candidate_seed0: bool = False):
    calls = []

    def run_fit(args, arm, seed):
        assert ra._stage(args) == "critic"
        calls.append((arm.name, seed))
        work = Path(args.out) / arm.name / f"seed-{seed}"
        work.mkdir(parents=True, exist_ok=True)
        record = _critic_evidence(arm.name, seed, arm.privileged)
        if fail_candidate_seed0 and arm.role == "candidate" and seed == 0:
            record["stability"]["critic"] = _stability("FAIL")
        path = work / "evidence-critic.json"
        path.write_text(json.dumps(record))
        return path

    monkeypatch.setattr(ra, "run_fit", run_fit)
    monkeypatch.setattr(ra, "source_record", lambda *a, **k: {"commit": "test"})
    return calls


def test_critic_stage_run_writes_a_v_only_receipt(tmp_path, monkeypatch) -> None:
    calls = _fake_critic_fits(monkeypatch)
    args = _args(tmp_path, "--stage", "critic", "--determinism-steps", "0", "--jobs", "1")
    assert ra.command_run(args) == 0
    assert len(calls) == 10 and set(calls[:2]) == {("privileged-e4", 0), ("control-e4", 0)}
    out = tmp_path / "out"
    assert not (out / "gate-receipt.json").exists()
    receipt = json.loads((out / "critic-receipt.json").read_text())
    assert receipt["configuration"]["stage"] == "critic"
    assert receipt["configuration"]["fit_payloads"] == []
    # The machine verdict cannot pass without the A evidence ...
    assert receipt["machine_verdict"] == "HOLD"
    stage = receipt["critic_stage"]
    assert stage["final"] is False and stage["candidate_arm"] == "privileged-e4"
    for name in ("privileged-e4", "control-e4"):
        arm = stage["arms"][name]
        # ... while every V check passes and every A check is listed as deferred.
        assert arm["status"] == "PASS", arm["not_passing"]
        assert "qol:a:same-village-ranking" in arm["deferred"]
        assert "a:report" in arm["deferred"] and "stability:advantage" in arm["deferred"]
        assert all(ra.critic_stage_deferred(i) for i in arm["deferred"])
        assert set(arm["per_seed_status"].values()) == {"PASS"}
    # No determinism probe ran here, so the stage summary HOLDs on it.
    assert stage["determinism"]["status"] == "HOLD" and stage["status"] == "HOLD"
    digest = ra.json_digest({k: v for k, v in receipt.items() if k != "receipt_digest"})
    assert receipt["receipt_digest"] == digest


def test_critic_stage_summary_passes_with_critic_determinism() -> None:
    evidence = []
    for seed in range(5):
        evidence.append(_critic_evidence("privileged-e4", seed, True))
        evidence.append(_critic_evidence("control-e4", seed, False))
    receipt = ra.build_receipt(
        evidence,
        arms=ra.DEFAULT_ARMS,
        seeds=list(range(5)),
        seed_rule="mean-with-floor",
        required_seeds=5,
        determinism=CRITIC_DETERMINISM,
        configuration={"stage": "critic"},
        attempt_id="t",
        source={"commit": "test"},
    )
    assert receipt["machine_verdict"] == "HOLD"  # A has no evidence
    stage = ra.critic_stage_summary(receipt)
    assert stage["status"] == "PASS" and stage["determinism"]["status"] == "PASS"
    # A V failure in one seed is not hidden by the deferral.
    evidence[0]["stability"]["critic"] = _stability("FAIL")
    failed = ra.build_receipt(
        evidence,
        arms=ra.DEFAULT_ARMS,
        seeds=list(range(5)),
        seed_rule="mean-with-floor",
        required_seeds=5,
        determinism=CRITIC_DETERMINISM,
        configuration={"stage": "critic"},
        attempt_id="t",
        source={"commit": "test"},
    )
    stage = ra.critic_stage_summary(failed)
    assert stage["status"] == "FAIL" and stage["arms"]["privileged-e4"]["status"] == "FAIL"
    assert stage["arms"]["control-e4"]["status"] == "PASS"


def test_critic_stage_seed_gate_reads_the_critic_evidence(tmp_path, monkeypatch) -> None:
    calls = _fake_critic_fits(monkeypatch, fail_candidate_seed0=True)
    args = _args(tmp_path, "--stage", "critic", "--determinism-steps", "0", "--jobs", "1")
    ra.command_run(args)
    assert sorted(calls) == [("control-e4", 0), ("privileged-e4", 0)]
    gate = json.loads((tmp_path / "out" / "seed-gate.json").read_text())
    assert gate["tripped"] and gate["reason"] == "stability-fail"
    assert gate["evidence"]["path"].endswith("evidence-critic.json")
    receipt = json.loads((tmp_path / "out" / "critic-receipt.json").read_text())
    assert receipt["machine_verdict"] == "FAIL" and receipt["critic_stage"]["status"] == "FAIL"


# --- tools/r5_critic_acceptance.sh ---------------------------------------------------------


@pytest.mark.skipif(__import__("shutil").which("bash") is None, reason="bash is not available")
def test_driver_script_runs_the_critic_stage_without_payloads(tmp_path) -> None:
    import os
    import subprocess

    stub = tmp_path / "python-stub.sh"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\n" "$@" > "$ARGS_OUT"\n', newline="\n")
    stub.chmod(0o755)
    script = ra.REPOSITORY / "tools" / "r5_critic_acceptance.sh"
    env = {
        **os.environ,
        "PYTHON": stub.as_posix(),
        "ARGS_OUT": (tmp_path / "args.txt").as_posix(),
        "NORMALIZER": "n.pt", "PRIV_NORMALIZER": "p.pt",
        "FIT_DATA": "fa:fb", "CAL_DATA": "c1 c2", "OUT": (tmp_path / "out").as_posix(),
        "ATTEMPT_ID": "t", "STAGE": "critic", "GAP_FIT_DATA": "fa", "LR": "1e-4",
        "CRITIC_ARGS": "--qol-loss-weight 30 --tail-normalize world-factor",
        "JOBS": "1", "THREADS": "1",
    }
    env.pop("FIT_PAYLOADS", None)
    env.pop("RESERVED_PAYLOADS", None)
    result = subprocess.run(
        ["bash", script.as_posix()], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args[:3] == ["-m", "anthropocene_rl.autonomous_v7.labor_r5_acceptance", "run"]
    assert args[args.index("--stage") + 1] == "critic"
    assert "--fit-payloads" not in args and "--reserved-payloads" not in args
    assert args[args.index("--gap-fit-data") + 1] == "fa"
    assert args[args.index("--lr") + 1] == "1e-4"
    # The critic arguments stay one argument for run --critic-args.
    assert args[args.index("--critic-args") + 1] == (
        "--qol-loss-weight 30 --tail-normalize world-factor"
    )
    fit = args[args.index("--fit-data") + 1 : args.index("--calibration-data")]
    assert fit == ["fa", "fb"]
    # STAGE=all still refuses to start without payloads.
    env["STAGE"] = "all"
    result = subprocess.run(
        ["bash", script.as_posix()], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0 and "FIT_PAYLOADS" in result.stderr


# --- population tail: advisory for r5 (user 28 Oct) ---------------------------------------------


def _with_tail_fail(record: dict) -> dict:
    for label in ("critic", "advantage"):
        summary = record["stability"].get(label)
        if summary is not None:
            summary["checks"]["population_tail"] = {
                "mandatory": True, "status": "FAIL", "upper_excess": 0.032,
            }
    return record


def test_population_tail_is_advisory_by_default_and_gates_on_request() -> None:
    assert ra.PREREGISTERED["population_tail_policy_r5"] == {
        **ra.PREREGISTERED["population_tail_policy_r5"],
        "value": "advisory",
        "status": "approved (user 28 Oct)",
    }
    record = _with_tail_fail(_critic_evidence("privileged-e4", 0, True))
    gated = ra.fit_checks(record)  # library default: the 26 Oct gating rule
    assert "stability:critic:population_tail" in {r["id"] for r in gated["primary"]}
    assert gated["primary_status"] == "FAIL"
    advisory = ra.fit_checks(record, population_tail="advisory")
    ids = {r["id"] for r in advisory["primary"]}
    assert not any("population_tail" in i for i in ids)
    row = next(a for a in advisory["advisory"] if a["check"] == "advisory:population-tail")
    assert row["status"] == "FLAG" and row["escalate"] and row["failed"] == ["critic"]
    assert row["gating"] is False and row["fits"]["critic"]["upper_excess"] == 0.032
    # Every other stability check still gates.
    record["stability"]["critic"]["checks"]["td_divergence"]["status"] = "FAIL"
    still = ra.fit_checks(record, population_tail="advisory")
    assert "stability:critic:td_divergence" in {
        r["id"] for r in still["primary"] if r["status"] == "FAIL"
    }
    with pytest.raises(ValueError, match="population-tail policy"):
        ra.fit_checks(record, population_tail="ignore")


def test_run_defaults_to_the_advisory_tail(tmp_path) -> None:
    args = _args(tmp_path, "--stage", "critic")
    assert args.population_tail == "advisory"
    assert _args(tmp_path, "--population-tail", "gate").population_tail == "gate"


@pytest.mark.parametrize("policy", ["advisory", "gate"])
def test_population_tail_trips_the_seed_gate_only_when_gated(tmp_path, monkeypatch, policy):
    calls = []

    def run_fit(args, arm, seed):
        calls.append((arm.name, seed))
        work = Path(args.out) / arm.name / f"seed-{seed}"
        work.mkdir(parents=True, exist_ok=True)
        path = work / "evidence-critic.json"
        path.write_text(json.dumps(_with_tail_fail(_critic_evidence(arm.name, seed, arm.privileged))))
        return path

    monkeypatch.setattr(ra, "run_fit", run_fit)
    monkeypatch.setattr(ra, "source_record", lambda *a, **k: {"commit": "test"})
    args = _args(tmp_path, "--stage", "critic", "--determinism-steps", "0", "--jobs", "1",
                 "--population-tail", policy)
    ra.command_run(args)
    gate = json.loads((tmp_path / "out" / "seed-gate.json").read_text())
    receipt = json.loads((tmp_path / "out" / "critic-receipt.json").read_text())
    assert receipt["population_tail_policy"] == policy
    assert receipt["configuration"]["population_tail"] == policy
    if policy == "advisory":
        assert not gate["tripped"] and len(calls) == 10
        assert receipt["critic_stage"]["arms"]["privileged-e4"]["status"] == "PASS"
        flags = receipt["arms"]["privileged-e4"]["advisory"]["0"]
        assert any(a["check"] == "advisory:population-tail" and a["status"] == "FLAG" for a in flags)
    else:
        assert gate["tripped"] and gate["reason"] == "stability-fail" and len(calls) == 2
        assert receipt["machine_verdict"] == "FAIL"


# --- other-head regression against the run's own control arm (user 28 Oct) --------------------


def _no_report(record: dict) -> dict:
    record.pop("control", None)
    return record


def _receipt_for(evidence):
    return ra.build_receipt(
        evidence, arms=ra.DEFAULT_ARMS, seeds=list(range(5)), seed_rule="mean-with-floor",
        required_seeds=5, determinism=CRITIC_DETERMINISM, configuration={"stage": "critic"},
        attempt_id="t", source={"commit": "test"},
    )


def _regression(receipt, arm="privileged-e4"):
    return {
        r["id"]: r for r in receipt["arms"][arm][ra.GATE_ID]["checks"]
        if r["id"].startswith("regression:")
    }


def test_regression_reference_is_the_seed_matched_control_arm() -> None:
    from tests.test_labor_r5_acceptance import _metrics

    evidence = []
    for seed in range(5):
        evidence.append(_no_report(_critic_evidence("privileged-e4", seed, True)))
        evidence.append(_no_report(_critic_evidence("control-e4", seed, False)))
    receipt = _receipt_for(evidence)
    rows = _regression(receipt)
    assert "regression:control" not in rows and rows  # no HOLD for a missing report
    assert all(r["status"] == "PASS" for r in rows.values())
    reference = receipt["regression_reference"]
    assert reference["privileged-e4/seed-3"] == {"kind": "control-arm", "arm": "control-e4", "seed": 3}
    assert reference["control-e4/seed-3"]["kind"] == "self (reference arm)"
    assert ra.critic_stage_summary(receipt)["status"] == "PASS"

    # A candidate whose settlement-guard calibration is worse than its seed-matched control
    # arm by more than 0.02 fails the regression check (seed rule over the 5 seeds).
    for record in evidence:
        if record["arm"] == "privileged-e4":
            record["calibration"]["calibration"]["heads"]["settlement_guard"]["all"] = _metrics(
                nmae=0.2
            )
    worse = _regression(_receipt_for(evidence))
    assert worse["regression:settlement_guard:all:nmae"]["status"] == "FAIL"
    assert worse["regression:settlement_guard:all:nmae"]["spread"]["mean"] == pytest.approx(0.1)


def test_a_supplied_control_report_still_wins_and_a_missing_seed_holds() -> None:
    evidence = [_critic_evidence("privileged-e4", 0, True), _critic_evidence("control-e4", 0, False)]
    receipt = ra.build_receipt(
        evidence, arms=ra.DEFAULT_ARMS, seeds=[0], seed_rule="mean-with-floor",
        required_seeds=1, determinism=CRITIC_DETERMINISM, configuration={},
        attempt_id="t", source={"commit": "test"},
    )
    assert receipt["regression_reference"]["privileged-e4/seed-0"] == {"kind": "control-report"}
    lonely = [_no_report(_critic_evidence("privileged-e4", 1, True)),
              _no_report(_critic_evidence("control-e4", 2, False))]
    receipt = ra.build_receipt(
        lonely, arms=ra.DEFAULT_ARMS, seeds=[1, 2], seed_rule="mean-with-floor",
        required_seeds=1, determinism=CRITIC_DETERMINISM, configuration={},
        attempt_id="t", source={"commit": "test"},
    )
    assert receipt["regression_reference"]["privileged-e4/seed-1"] is None
    rows = {r["id"]: r for r in receipt["arms"]["privileged-e4"][ra.GATE_ID]["checks"]}
    assert rows["regression:control"]["status"] == "HOLD"
