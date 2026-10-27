"""Live stability early stop, V-critic held-out validation and the r5 seed gate.

* Each early-stop path (gradient dominance, TD divergence, more than 5 rollbacks) is
  triggered synthetically: the fit raises StabilityAbort at once, FAILs the same check id
  the end-of-run judgement uses and records the step; without the option the same fit
  runs on and FAILs that check at the end.
* With no trip the option changes nothing (bitwise parameters, same checks).
* Validation on/off gives bitwise-identical fitted parameters and RNG states on CPU.
* The seed gate stops the remaining seeds after a failing candidate seed 0.
"""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import pytest
import torch
from test_labor_privileged_model import _layout
from test_labor_qol_head import _training_dataset
from torch import nn

from anthropocene_rl.autonomous_v7 import labor_r5_acceptance as ra
from anthropocene_rl.autonomous_v7.labor_bc import pretrain_critic_on_dataset
from anthropocene_rl.autonomous_v7.labor_model import (
    LaborCostCritics,
    LaborQoLHead,
    LaborTaskCritic,
)
from anthropocene_rl.autonomous_v7.labor_stability import (
    StabilityAbort,
    StabilityConfig,
    StabilityMonitor,
    ValidationMonitor,
    stability_config_from_args,
)

# --- shared fixtures -------------------------------------------------------------------------


def _dominance_setup(config):
    torch.manual_seed(0)
    shared = nn.Linear(4, 4)
    head_a, head_b = nn.Linear(4, 1), nn.Linear(4, 1)
    params = [*shared.parameters(), *head_a.parameters(), *head_b.parameters()]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    monitor = StabilityMonitor(
        config,
        fit="toy",
        optimizer=optimizer,
        parameters=params,
        blocks={
            "shared": list(shared.parameters()),
            "head:a": list(head_a.parameters()),
            "head:b": list(head_b.parameters()),
        },
        shared_heads={"shared": (list(shared.parameters()), ("a", "b"))},
    )
    x = torch.randn(32, 4)

    def step():
        optimizer.zero_grad()
        hidden = shared(x)
        loss_a = (head_a(hidden) - 1e3).square().mean()
        loss_b = (head_b(hidden.detach()) - 5.0).square().mean().detach() + 0.0 * head_b.bias.sum()
        return monitor.step(
            loss_a + loss_b, max_grad_norm=1.0, head_losses={"a": loss_a, "b": loss_b}
        )

    return monitor, step, params


def _toy_monitor(config):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    monitor = StabilityMonitor(
        config,
        fit="toy",
        optimizer=optimizer,
        parameters=list(model.parameters()),
        blocks={"first": list(model[0].parameters()), "second": list(model[2].parameters())},
    )
    generator = torch.Generator().manual_seed(1)
    x, y = torch.randn(16, 4, generator=generator), torch.randn(16, 1, generator=generator)

    def step(scale: float = 1.0, diagnostics=None):
        optimizer.zero_grad()
        return monitor.step(
            (model(x) - y).square().mean() * scale, max_grad_norm=1.0, diagnostics=diagnostics
        )

    return monitor, step, model


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- 1. early stop on each would-FAIL condition --------------------------------------------


def test_dominance_alarm_stops_the_fit_under_the_end_of_run_check_id(tmp_path) -> None:
    log = tmp_path / "s.jsonl"
    _, step, params = _dominance_setup(
        StabilityConfig(log_every=1, early_stop=True, jsonl=str(log))
    )
    with pytest.raises(StabilityAbort, match="early stop at step 16: gradient_dominance") as caught:
        for _ in range(40):
            step()
    summary = caught.value.summary
    assert summary["steps"] == 16  # the 16-point window fills at step 16 and alarms at once
    assert summary["early_stop"]["check"] == "gradient_dominance"
    assert summary["early_stop"]["step"] == 16 and summary["early_stop"]["stopped_early"]
    assert summary["checks"]["gradient_dominance"]["status"] == "FAIL"
    assert "gradient_dominance" in summary["mandatory_failed"] and not summary["admitted"]
    stopped = [p.detach().clone() for p in params]
    kinds = [line["kind"] for line in _lines(log)]
    assert kinds[-3:] == ["alarm", "step", "abort"]  # the step is committed, then stopped
    # Without the option the same fit runs on and the end-of-run check FAILs the same id,
    # from the same alarm at the same step, and the parameters at that step agree.
    reference, reference_step, reference_params = _dominance_setup(StabilityConfig(log_every=1))
    for _ in range(16):
        reference_step()
    assert all(torch.equal(a, b) for a, b in zip(stopped, reference_params, strict=True))
    for _ in range(4):
        reference_step()
    end = reference.finish()
    assert end["checks"]["gradient_dominance"]["status"] == "FAIL"
    assert end["alarms"][0]["step"] == 16 and "early_stop" not in end


def _td_monitor(config):
    monitor, _, _ = _toy_monitor(config)
    return monitor


def _td_feed(monitor, increments, weight=100.0):
    current = torch.full((10,), 40.0)
    monitor.td_refresh(0, current - 30.0, current, weight, growth_eligible=False, error=0.0)
    for epoch, step in enumerate(increments, start=1):
        previous, current = current, current + step * (1 if epoch % 2 else -1)
        monitor.td_refresh(epoch, previous, current, weight, error=0.0)


def test_td_growth_alarm_stops_the_fit() -> None:
    growing = [1.0, 1.0, 1.0, 2.0, 4.0, 8.0, 8.0]  # rises > 10% on refreshes 4, 5, 6
    with pytest.raises(StabilityAbort, match="td_divergence") as caught:
        _td_feed(_td_monitor(StabilityConfig(log_every=1, early_stop=True)), growing)
    summary = caught.value.summary
    assert summary["early_stop"]["check"] == "td_divergence"
    assert "drift-rose" in summary["early_stop"]["reason"]
    assert summary["checks"]["td_divergence"]["status"] == "FAIL"
    refreshes = summary["checks"]["td_divergence"]["refreshes"]
    assert refreshes[-1]["epoch"] == 6  # stopped at the alarm, refresh 7 never ran
    # The library default judges the same alarm at the end of the run.
    plain = _td_monitor(StabilityConfig(log_every=1))
    _td_feed(plain, growing)
    end = plain.finish()
    assert end["checks"]["td_divergence"]["status"] == "FAIL"
    assert end["checks"]["td_divergence"]["alarms"][0]["epoch"] == 6


def test_td_target_outside_the_bound_stops_the_fit_immediately() -> None:
    monitor = _td_monitor(StabilityConfig(log_every=1, early_stop=True))
    current = torch.full((10,), 40.0)
    with pytest.raises(StabilityAbort, match="targets-outside") as caught:
        monitor.td_refresh(0, None, current, 100.0, growth_eligible=False)
        monitor.td_refresh(1, current, current + 100.0, 100.0)  # 140 > W = 100
    assert caught.value.summary["early_stop"]["check"] == "td_divergence"
    assert caught.value.summary["checks"]["td_divergence"]["status"] == "FAIL"


def test_more_than_five_rollbacks_stop_the_fit_as_finite_updates() -> None:
    _, step, _ = _toy_monitor(StabilityConfig(log_every=1, early_stop=True))
    with pytest.raises(StabilityAbort, match="finite_updates") as caught:
        for index in range(20):  # alternate: never K=3 consecutive failures
            step(float("nan") if index % 2 == 0 else 1.0)
    summary = caught.value.summary
    assert summary["rollbacks"] == 6 and summary["steps"] == 11
    assert summary["early_stop"]["check"] == "finite_updates"
    assert summary["checks"]["finite_updates"]["status"] == "FAIL"
    assert not summary["checks"]["finite_updates"]["aborted"]  # not the K-consecutive abort
    plain, plain_step, _ = _toy_monitor(StabilityConfig(log_every=1))
    for index in range(20):
        plain_step(float("nan") if index % 2 == 0 else 1.0)
    assert plain.summary()["checks"]["finite_updates"]["status"] == "FAIL"
    # Five rollbacks are admitted (the rule is "more than 5").
    five, five_step, _ = _toy_monitor(StabilityConfig(log_every=1, early_stop=True))
    for index in range(10):
        five_step(float("nan") if index % 2 == 0 else 1.0)
    assert five.rollbacks == 5 and five.summary()["checks"]["finite_updates"]["status"] == "PASS"


def test_population_tail_raises_a_live_alarm_but_never_stops(tmp_path) -> None:
    log = tmp_path / "s.jsonl"
    monitor, step, _ = _toy_monitor(StabilityConfig(log_every=1, early_stop=True, jsonl=str(log)))
    tails = [0.0, 0.5, 0.5, 0.0, 0.5]  # upper excess; entering the alarm twice

    for value in tails:
        tail = {"upper_excess": value, "top_bin_mass": 0.0, "rows": 4}
        assert step(diagnostics=lambda tail=tail: {"population_tail": tail})
    summary = monitor.finish()
    advisories = summary["advisories"]
    assert [a["step"] for a in advisories] == [2, 5]
    assert all(a["check"] == "population_tail" and not a["gating"] for a in advisories)
    assert "early_stop" not in summary
    alarms = [line for line in _lines(log) if line["kind"] == "alarm"]
    assert len(alarms) == 2 and summary["alarms"] == []  # never a mandatory alarm


def test_without_a_trip_early_stop_changes_nothing() -> None:
    def fit(config):
        torch.manual_seed(0)
        task = LaborTaskCritic(
            _layout(False), width=16, token_width=16, qol_head=LaborQoLHead(target="td-lambda")
        )
        costs = LaborCostCritics(_layout(False), width=16, token_width=16)
        summary = pretrain_critic_on_dataset(
            task_critic=task,
            cost_critics=costs,
            dataset=_training_dataset(),
            epochs=2,
            batch_size=4,
            stability=config,
        )
        state = {**task.state_dict(), **{f"c.{k}": v for k, v in costs.state_dict().items()}}
        return state, summary["stability"]

    plain, plain_summary = fit(StabilityConfig(log_every=1))
    early, early_summary = fit(StabilityConfig(log_every=1, early_stop=True))
    assert all(torch.equal(plain[k], early[k]) for k in plain)
    assert "early_stop" not in early_summary
    # The only additions: the config flag and the live, non-gating population-tail alarm
    # (this untrained fixture's distributional head sits in its tail).
    assert {a["check"] for a in early_summary.get("advisories", [])} <= {"population_tail"}
    assert early_summary["config"].pop("early_stop") is True
    assert "early_stop" not in plain_summary["config"] and "advisories" not in plain_summary

    def timeless(checks):
        checks = copy.deepcopy(checks)
        for refresh in checks["td_divergence"].get("refreshes", []):
            refresh.pop("refresh_seconds", None)
        return checks

    assert timeless(plain_summary["checks"]) == timeless(early_summary["checks"])
    for key in ("config", "alarms", "events", "steps", "rollbacks", "lr_final", "admitted"):
        assert plain_summary[key] == early_summary[key]


def test_cli_flag_builds_an_explicit_early_stop_config() -> None:
    args = argparse.Namespace(stability_early_stop=True, stability_log=None, stability_every=None)
    config = stability_config_from_args(args)
    assert isinstance(config, StabilityConfig) and config.early_stop
    assert config.state_dict()["early_stop"] is True
    assert stability_config_from_args(argparse.Namespace()) == "auto"
    assert "early_stop" not in StabilityConfig().state_dict()
    with pytest.raises(ValueError, match="needs the stability monitor"):
        stability_config_from_args(
            argparse.Namespace(no_stability=True, stability_early_stop=True)
        )


def test_acceptance_check_rows_carry_the_early_stop_step(tmp_path) -> None:
    _, step, _ = _toy_monitor(StabilityConfig(log_every=1, early_stop=True))
    with pytest.raises(StabilityAbort) as caught:
        for index in range(20):
            step(float("nan") if index % 2 == 0 else 1.0)
    work = tmp_path / "privileged-e4" / "seed-0"
    work.mkdir(parents=True)
    (work / "critic.pt.stability-abort.json").write_text(json.dumps(caught.value.summary))
    stub = ra.abort_evidence(work, ra.DEFAULT_ARMS[0], 0)
    rows = {r["id"]: r for r in ra.stability_checks("critic", stub["stability"]["critic"])}
    row = rows["stability:critic:finite_updates"]
    assert row["status"] == "FAIL" and row["stopped_early_at_step"] == 11
    assert ra.fit_checks(stub)["primary_status"] == "FAIL"
    assert all("stopped_early_at_step" not in r for k, r in rows.items() if k != row["id"])


# --- 2. held-out validation ----------------------------------------------------------------


def _validation_dataset(worlds=(90180, 90181)):
    data = _training_dataset()
    data["row_world"] = torch.tensor([worlds[0]] * 3 + [worlds[1]] * 3)
    generator = torch.Generator().manual_seed(5)
    for key in ("reward_qol_years", "reward_population_years"):
        data[key] = data[key] * (1.0 + torch.rand(6, generator=generator, dtype=torch.float64))
    return data


def _critic_fit(target, *, stability, validation, epochs=3, rng_seed=7, dataset=None, **kwargs):
    torch.manual_seed(0)
    task = LaborTaskCritic(
        _layout(False), width=16, token_width=16, qol_head=LaborQoLHead(target=target)
    )
    costs = LaborCostCritics(_layout(False), width=16, token_width=16)
    generator = np.random.default_rng(rng_seed)
    summary = pretrain_critic_on_dataset(
        task_critic=task,
        cost_critics=costs,
        dataset=_training_dataset() if dataset is None else dataset,
        epochs=epochs,
        batch_size=2,
        rng=generator,
        stability=stability,
        validation_dataset=validation,
        **kwargs,
    )
    state = {**task.state_dict(), **{f"c.{k}": v for k, v in costs.state_dict().items()}}
    return state, summary, generator.bit_generator.state, torch.get_rng_state()


@pytest.mark.parametrize("target", ["monte-carlo", "td-lambda"])
@pytest.mark.parametrize("monitored", [False, True])
def test_validation_on_or_off_gives_bitwise_identical_fits(target, monitored, tmp_path) -> None:
    def config(name):
        return StabilityConfig(log_every=1, jsonl=str(tmp_path / name)) if monitored else None

    off, off_summary, off_rng, off_torch = _critic_fit(
        target, stability=config("off.jsonl"), validation=None
    )
    on, on_summary, on_rng, on_torch = _critic_fit(
        target, stability=config("on.jsonl"), validation=_validation_dataset()
    )
    assert all(torch.equal(off[k], on[k]) for k in off)
    assert off_rng == on_rng and torch.equal(off_torch, on_torch)  # no RNG consumed
    for key in ("final_loss", "final_task_loss", "final_cost_loss"):
        assert off_summary[key] == on_summary[key]
    report = on_summary["validation"]
    assert report["gating"] is False and report["worlds"] == [90180, 90181]
    assert [e["epoch"] for e in report["epochs"]] == [1, 2, 3]
    for record in report["epochs"]:
        assert set(record["heads"]) == {
            "qol_years", "population_years", "feasibility", "settlement_guard"
        }
        assert record["train_steps"] == 3 and record["train_loss"] is not None
        assert record["qol_target"] == target
        assert ("qol_years_monte_carlo" in record) == (target == "td-lambda")
    assert "validation" not in off_summary
    if monitored:
        assert on_summary["stability"]["validation"]["epochs"] == report["epochs"]
        assert "validation" not in off_summary["stability"]
        kinds = [line["kind"] for line in _lines(tmp_path / "on.jsonl")]
        assert kinds.count("validation") == 3
        # Validation never alters the gating checks (timings aside).
        def timeless(checks):
            checks = copy.deepcopy(checks)
            for refresh in checks["td_divergence"].get("refreshes", []):
                refresh.pop("refresh_seconds", None)
            return checks

        assert timeless(on_summary["stability"]["checks"]) == timeless(
            off_summary["stability"]["checks"]
        )


def test_validation_losses_match_the_training_loss_definition() -> None:
    # The training rows used as their own "validation" set (worlds relabelled) reproduce
    # the fit's final unweighted loss: same components, same row mean.
    data = _training_dataset()
    same = copy.deepcopy(data)
    same["row_world"] = torch.tensor([90180] * 3 + [90181] * 3)
    _, summary, _, _ = _critic_fit("monte-carlo", stability=None, validation=same, dataset=data)
    last = summary["validation"]["epochs"][-1]
    assert last["total"] == pytest.approx(summary["final_loss"], rel=1e-6)
    assert last["task"] == pytest.approx(summary["final_task_loss"], rel=1e-6)


def test_validation_refuses_training_reserved_and_sealed_worlds() -> None:
    for worlds, match in (
        ((90100, 90181), "overlap the training worlds"),
        ((90154, 90181), "reserved held-out-test or sealed"),
        ((3000, 90181), "reserved held-out-test or sealed"),
    ):
        with pytest.raises(ValueError, match=match):
            _critic_fit("monte-carlo", stability=None, validation=_validation_dataset(worlds))


def test_the_training_loader_refuses_calibration_worlds() -> None:
    torch.manual_seed(0)
    task = LaborTaskCritic(_layout(False), width=16, token_width=16)
    costs = LaborCostCritics(_layout(False), width=16, token_width=16)
    with pytest.raises(ValueError, match="calibration"):
        pretrain_critic_on_dataset(
            task_critic=task, cost_critics=costs, dataset=_validation_dataset(), epochs=1
        )
    args = argparse.Namespace(fit_data=["a/fit", "b/cal"], calibration_data=["b/cal"])
    with pytest.raises(ValueError, match="calibration data listed as fitting data"):
        ra.require_calibration_not_trained(args)


def test_validation_divergence_detector() -> None:
    def run(train, total):
        monitor = ValidationMonitor()
        alarms = [
            monitor.update({"epoch": i + 1, "train_loss": t, "total": v})
            for i, (t, v) in enumerate(zip(train, total, strict=True))
        ]
        return monitor, alarms

    monitor, alarms = run([4.0, 3.0, 2.0, 1.0], [1.0, 0.9, 1.0, 1.1])
    assert alarms[:3] == [None] * 3 and alarms[3]["check"] == "validation_divergence"
    assert alarms[3]["epoch"] == 4 and not alarms[3]["gating"]
    assert monitor.report()["status"] == "FLAG"
    _, alarms = run([4.0, 3.0, 2.0, 1.0], [1.0, 1.1, 1.0, 1.1])  # one rise at a time
    assert alarms == [None] * 4
    _, alarms = run([1.0, 2.0, 3.0], [1.0, 1.1, 1.2])  # training also rising
    assert alarms == [None] * 3


def test_validation_divergence_is_a_non_gating_advisory_in_the_receipt(tmp_path) -> None:
    monitor, step, _ = _toy_monitor(StabilityConfig(log_every=1, jsonl=str(tmp_path / "v.jsonl")))
    step()
    for epoch, (train, total) in enumerate([(3.0, 1.0), (2.0, 1.1), (1.0, 1.2)], start=1):
        monitor.record_validation({"epoch": epoch, "train_loss": train, "total": total})
    summary = monitor.finish()
    assert summary["advisories"][0]["check"] == "validation_divergence"
    assert summary["validation"]["status"] == "FLAG"
    assert summary["checks"]["finite_updates"]["status"] == "PASS"
    assert "validation_divergence" not in summary["mandatory_failed"]
    alarm = [line for line in _lines(tmp_path / "v.jsonl") if line["kind"] == "alarm"]
    assert alarm and alarm[0]["check"] == "validation_divergence"
    from test_labor_r5_acceptance import _evidence

    evidence = _evidence()
    plain = ra.fit_checks(evidence)
    evidence["stability"]["critic"]["validation"] = summary["validation"]
    flagged = ra.fit_checks(evidence)
    assert flagged["primary"] == plain["primary"]
    assert flagged["primary_status"] == plain["primary_status"]
    advisory = flagged["advisory"][-1]
    assert advisory["check"] == "advisory:critic-validation-divergence"
    assert advisory["status"] == "FLAG" and advisory["gating"] is False
    assert len(plain["advisory"]) == len(flagged["advisory"]) - 1


def test_validation_history_survives_an_exact_resume(tmp_path) -> None:
    data = _training_dataset()
    data["collection"] = dict(data["collection"], dataset_digest="fixture")

    def fit(**kwargs):
        torch.manual_seed(0)
        task = LaborTaskCritic(
            _layout(False), width=16, token_width=16, qol_head=LaborQoLHead(target="td-lambda")
        )
        costs = LaborCostCritics(_layout(False), width=16, token_width=16)
        summary = pretrain_critic_on_dataset(
            task_critic=task,
            cost_critics=costs,
            dataset=data,
            epochs=3,
            batch_size=2,
            rng=np.random.default_rng(7),
            validation_dataset=_validation_dataset(),
            **kwargs,
        )
        return task.state_dict(), summary

    reference, reference_summary = fit()
    path = str(tmp_path / "resume.pt")
    fit(max_steps=5, resume_path=path, resume_every=2)  # interrupted inside epoch 2
    resumed, resumed_summary = fit(resume_path=path, resume_every=2)
    assert all(torch.equal(reference[k], v) for k, v in resumed.items())

    def strip(epochs):
        return [{k: v for k, v in e.items() if k != "t"} for e in epochs]

    assert strip(resumed_summary["validation"]["epochs"]) == strip(
        reference_summary["validation"]["epochs"]
    )


# --- 3. seed gate ----------------------------------------------------------------------------


def _run_args(tmp_path, *extra):
    return ra.argument_parser().parse_args(
        [
            "run",
            "--normalizer-checkpoint", str(tmp_path / "n.pt"),
            "--privileged-normalizer", str(tmp_path / "p.pt"),
            "--fit-data", str(tmp_path / "fit"),
            "--calibration-data", str(tmp_path / "cal"),
            "--fit-payloads", str(tmp_path / "fp"),
            "--reserved-payloads", str(tmp_path / "rp"),
            "--attempt-id", "gate-test",
            "--determinism-steps", "0",
            "--jobs", "4",
            "--out", str(tmp_path / "out"),
            *extra,
        ]
    )


def _fake_fits(monkeypatch, *, fail_candidate_seed0: str | None):
    from test_labor_r5_acceptance import _evidence, _stability

    calls = []

    def run_fit(args, arm, seed):
        calls.append((arm.name, seed))
        assert args.stability_early_stop and args.critic_validation  # run defaults
        work = ra.Path(args.out) / arm.name / f"seed-{seed}"
        work.mkdir(parents=True, exist_ok=True)
        if arm.role == "candidate" and seed == 0 and fail_candidate_seed0 == "abort":
            summary = _stability("FAIL")
            summary["early_stop"] = {"check": "td_divergence", "step": 120, "reason": "x"}
            (work / "critic.pt.stability-abort.json").write_text(json.dumps(summary))
            raise RuntimeError("command failed (1)")
        evidence = _evidence(arm.name, seed, arm.privileged)
        if arm.role == "candidate" and seed == 0 and fail_candidate_seed0 == "fail":
            evidence["stability"]["advantage"] = _stability("FAIL")
        path = work / "evidence.json"
        path.write_text(json.dumps(evidence))
        return path

    receipts = []
    monkeypatch.setattr(ra, "run_fit", run_fit)
    monkeypatch.setattr(ra, "command_receipt", lambda args: receipts.append(args) or 0)
    return calls, receipts


@pytest.mark.parametrize("failure", ["abort", "fail"])
def test_a_failing_candidate_seed0_stops_the_remaining_seeds(tmp_path, monkeypatch, failure):
    real_receipt = ra.command_receipt
    calls, receipts = _fake_fits(monkeypatch, fail_candidate_seed0=failure)
    assert ra.command_run(_run_args(tmp_path)) == 0
    assert sorted(calls) == [("control-e4", 0), ("privileged-e4", 0)]
    gate = json.loads((tmp_path / "out" / "seed-gate.json").read_text())
    assert gate["tripped"] and gate["arm"] == "privileged-e4" and gate["seed"] == 0
    assert gate["reason"] == ("stability-abort" if failure == "abort" else "stability-fail")
    if failure == "abort":
        assert gate["evidence"]["critic"]["early_stop"]["check"] == "td_divergence"
    else:
        assert gate["evidence"]["failed"] == ["stability:advantage:td_divergence"]
    assert len(gate["skipped"]) == 8 and "privileged-e4/seed-4" in gate["skipped"]
    configuration = receipts[0].configuration
    assert configuration["seed_gate"] == gate
    assert configuration["stability_early_stop"] and configuration["critic_validation"]
    assert {p.parent.name for p in receipts[0].evidence} == {"seed-0"}
    # The receipt says so and FAILs.
    monkeypatch.setattr(ra, "source_record", lambda *a, **k: {"commit": "test"})
    real_receipt(receipts[0])
    receipt = json.loads((tmp_path / "out" / "gate-receipt.json").read_text())
    assert receipt["seed_gate"]["tripped"] and receipt["machine_verdict"] == "FAIL"
    finding = next(f for f in receipt["unresolved_findings"] if f.get("check") == "seed-gate")
    assert finding["status"] == "FAIL" and len(finding["skipped"]) == 8
    digest = ra.json_digest({k: v for k, v in receipt.items() if k != "receipt_digest"})
    assert receipt["receipt_digest"] == digest


def test_a_passing_candidate_seed0_launches_the_remaining_seeds(tmp_path, monkeypatch) -> None:
    calls, receipts = _fake_fits(monkeypatch, fail_candidate_seed0=None)
    ra.command_run(_run_args(tmp_path))
    assert set(calls[:2]) == {("control-e4", 0), ("privileged-e4", 0)}
    assert len(calls) == 10
    gate = receipts[0].configuration["seed_gate"]
    assert gate["enabled"] and not gate["tripped"] and "skipped" not in gate
    assert len(receipts[0].evidence) == 10


def test_no_seed_gate_restores_the_single_launch(tmp_path, monkeypatch) -> None:
    calls, receipts = _fake_fits(monkeypatch, fail_candidate_seed0="abort")
    ra.command_run(_run_args(tmp_path, "--no-seed-gate"))
    assert len(calls) == 10
    assert receipts[0].configuration["seed_gate"] == {"enabled": False, "reason": "--no-seed-gate"}
    assert not (tmp_path / "out" / "seed-gate.json").exists()
    # The aborted candidate seed 0 is still a FAIL stub, as before.
    names = {p.name for p in receipts[0].evidence}
    assert "evidence-aborted.json" in names


def test_run_passes_the_monitoring_options_to_the_fits(tmp_path) -> None:
    args = _run_args(tmp_path)
    assert ra._early_stop_flag(args) == ["--stability-early-stop"]
    assert ra._validation_flag(args) == ["--validation-dataset", tmp_path / "cal"]
    off = _run_args(tmp_path, "--no-stability-early-stop", "--no-critic-validation")
    assert ra._early_stop_flag(off) == [] and ra._validation_flag(off) == []
