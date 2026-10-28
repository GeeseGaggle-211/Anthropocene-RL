"""r5 critic gate v2 (user decision 28 Oct).

1. The QoL calibration gate measures against the tail-robust n-step target (k = 5
   observed rewards + gamma^k x the critic's own bootstrap); the randomized-tail Monte
   Carlo metrics stay as non-gating diagnostics.
2. The QoL reference scale is remeasured on the fitting batches into a versioned, sealed
   artifact; evaluate and the receipt record which scale and digest were used.
3. The TD trajectory rule v2: warm-up by value, PASS declining or stationary below a
   floor on the ERROR series (drift advisory), divergence relative to the stationary
   median. Old monitor states and receipts keep their v1 verdicts.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pytest
import torch
from test_labor_early_stop import _td_monitor
from test_labor_r5_acceptance import _calibration, _evidence, _receipt

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7 import labor_pretrain_eval as pe
from anthropocene_rl.autonomous_v7 import labor_r5_acceptance as ra
from anthropocene_rl.autonomous_v7.labor_bc import (
    TAIL_HEADS,
    TAIL_NONE_TERMINAL,
    TAIL_RANDOMIZED,
    discounted_returns,
)
from anthropocene_rl.autonomous_v7.labor_stability import (
    TD_RULE_V1,
    TD_RULE_V2,
    StabilityAbort,
    StabilityConfig,
    StabilityThresholds,
    TDDriftMonitor,
    rejudge_td_refreshes,
    td_divergence_v2,
    td_warmup_by_value,
    trajectory_report_v2,
)

GAMMA = math.exp(-contracts.TURN_YEARS / contracts.HORIZON_YEARS)
REWARD = 10.0
V_TRUE = REWARD / (1.0 - GAMMA)  # constant QoL reward forever


# --- synthetic trajectories --------------------------------------------------------------------


def _dataset(trajectories):
    """``trajectories``: (world, camp, length, terminal_last, tail) per (world, camp)."""
    worlds, camps, turns, terminal, codes, tails = [], [], [], [], [], []
    for world, camp, length, terminal_last, tail in trajectories:
        for turn in range(length):
            worlds.append(world)
            camps.append(camp)
            turns.append(turn)
            last = turn == length - 1
            terminal.append(bool(terminal_last and last))
            codes.append(TAIL_NONE_TERMINAL if terminal_last else TAIL_RANDOMIZED)
            tails.append(0.0 if terminal_last else float(tail))
    n = len(worlds)
    data = {
        "row_world": torch.tensor(worlds, dtype=torch.long),
        "row_camp": torch.tensor(camps, dtype=torch.long),
        "row_turn": torch.tensor(turns, dtype=torch.long),
        "row_quarantined": torch.zeros(n, dtype=torch.bool),
        "transition_elapsed_years": torch.full((n,), contracts.TURN_YEARS, dtype=torch.float64),
        "transition_terminal": torch.tensor(terminal, dtype=torch.bool),
        "reward_qol_years": torch.full((n,), REWARD, dtype=torch.float64),
        "reward_population_years": torch.zeros(n, dtype=torch.float64),
        "reward_feasibility": torch.zeros(n, dtype=torch.float64),
        "reward_settlement_guard": torch.zeros(n, dtype=torch.float64),
        "tail_method": torch.tensor(codes, dtype=torch.long),
        "tail_cutoff_turn": torch.zeros(n, dtype=torch.long),
    }
    for head in TAIL_HEADS:
        data["tail_value__" + head] = torch.tensor(
            tails if head == "qol_years" else [0.0] * n, dtype=torch.float64
        )
        data["tail_standard_error__" + head] = torch.zeros(n, dtype=torch.float64)
    return data


def _two_worlds(factor_a, factor_b, length=12):
    """Two living worlds whose roulette schedules scale the true tail by a world factor."""
    return _dataset(
        [
            (90100, 1, length, False, factor_a * V_TRUE),
            (90100, 2, length, False, factor_a * V_TRUE),
            (90101, 1, length, False, factor_b * V_TRUE),
        ]
    )


def _gate(data, values=None):
    n = int(data["row_world"].shape[0])
    values = np.full(n, V_TRUE) if values is None else values
    rows = np.arange(n)
    mc = discounted_returns(data)["qol_years"].double().numpy()
    return pe.qol_nstep_calibration(
        data, rows, values, mc, {"all": np.ones(n, dtype=bool)}, pe.frozen_qol_reference()
    )


# --- 1. the n-step target -----------------------------------------------------------------------


def test_nstep_gate_is_insensitive_to_the_world_tail_factor_and_mc_is_not() -> None:
    fair = _gate(_two_worlds(1.0, 1.0))
    skewed = _gate(_two_worlds(1.67, 0.61))  # the pilot's range of world factors
    for report in (fair, skewed):
        assert report["strata"]["all"]["nmae"] == pytest.approx(0.0, abs=1e-9)
        assert report["strata"]["all"]["bias"] == pytest.approx(0.0, abs=1e-9)
    # The exact critic is exact against the Monte Carlo return only when the tail is fair.
    assert fair["strata"]["all"]["monte_carlo"]["nmae"] == pytest.approx(0.0, abs=1e-6)
    mc = skewed["strata"]["all"]["monte_carlo"]
    assert mc["gating"] is False and mc["nmae"] > 0.20  # unreachable under the MC target
    # A biased critic is seen by the gate whatever the tail factor.
    n = int(_two_worlds(1.0, 1.0)["row_world"].shape[0])
    biased = np.full(n, 1.2 * V_TRUE)
    a = _gate(_two_worlds(1.0, 1.0), biased)["strata"]["all"]
    b = _gate(_two_worlds(1.67, 0.61), biased)["strata"]["all"]
    assert a["nmae"] > 0.0 and a["nmae"] == pytest.approx(b["nmae"], rel=1e-12)
    assert a["bias"] == pytest.approx((1 - GAMMA**5) * 0.2 * V_TRUE / 43.152, rel=1e-9)


def test_living_end_rows_are_reported_not_gated_and_terminal_windows_are_exact() -> None:
    data = _dataset([(90100, 1, 12, False, 1.67 * V_TRUE), (90100, 2, 8, True, 0.0)])
    report = _gate(data)
    closures = report["closures"]
    # Living trajectory: 7 full windows, 4 shortened, the last row itself.
    # Terminal trajectory: 3 full windows, 5 ending at the terminal row (exact).
    assert closures == {
        "full-window-bootstrap": 7 + 3,
        "terminal-inside-window": 5,
        "living-end-shortened": 4,
        "living-end-row": 1,
    }
    assert report["gated_rows"] == 15 and report["strata"]["all"]["rows"] == 15
    assert report["strata"]["all"]["excluded_living_end_rows"] == 5
    living = report["living_end"]
    assert living["gating"] is False and living["shortened_rows"] == 4
    assert living["bootstrap_closure"]["nmae"] == pytest.approx(0.0, abs=1e-9)
    assert living["declared_tail_closure"]["rows"] == 5  # closed by the 1.67 V tail
    assert living["declared_tail_closure"]["bias"] < 0.0
    targets = pe.nstep_targets(
        np.full(20, V_TRUE),
        np.full(20, REWARD),
        np.full(20, contracts.TURN_YEARS),
        data["transition_terminal"].numpy(),
        data["row_turn"].numpy(),
        data["row_world"].numpy(),
        data["row_camp"].numpy(),
        np.full(20, -1),
    )
    terminal_rows = np.flatnonzero(targets["closure"] == pe.NSTEP_TERMINAL)
    expected = [REWARD * sum(GAMMA**j for j in range(8 - p)) for p in range(3, 8)]
    assert targets["target"][terminal_rows] == pytest.approx(expected)
    assert math.isnan(targets["target"][11])  # the living end has no target


def test_full_windows_match_the_acceptance_nstep_residual() -> None:
    data = _two_worlds(1.3, 0.7)
    n = int(data["row_world"].shape[0])
    values = np.linspace(0.5, 1.5, n) * V_TRUE
    args = (
        values,
        data["reward_qol_years"].numpy(),
        data["transition_elapsed_years"].numpy(),
        data["transition_terminal"].numpy(),
        data["row_turn"].numpy(),
        data["row_world"].numpy(),
        data["row_camp"].numpy(),
        np.full(n, -1),
    )
    targets = pe.nstep_targets(*args, data["tail_value__qol_years"].numpy())
    residual, _, ok = ra.nstep_residuals(
        *args, data["tail_method"].numpy(), data["tail_value__qol_years"].numpy(), k=5
    )
    assert ok.all()
    # Full windows: identical; living-end windows: the declared-tail closure is the residual's.
    assert values - targets["tail_target"] == pytest.approx(residual, rel=1e-12, abs=1e-9)
    full = targets["closure"] == pe.NSTEP_FULL
    assert np.allclose(targets["target"][full], targets["tail_target"][full])
    assert ra.threshold("td_nstep_k") == pe.QOL_NSTEP_K
    assert ra.PREREGISTERED["qol_calibration_target"]["value"] == pe.QOL_GATE_TARGET_ID


def test_ranking_and_mc_heads_stay_unchanged_in_the_qol_checks() -> None:
    # Old evidence (no n-step section) is judged exactly as before, on the MC heads.
    old = ra.qol_checks(_calibration(), None, None)
    rows = {c["id"]: c for c in old}
    assert rows["qol:b:nmae:all"]["value"] == 0.1 and rows["qol:b:nmae:all"]["detail"] is None
    # New evidence: the gate reads the n-step section; MC rides along as a non-gating detail.
    calibration = _calibration()
    nstep = _gate(_two_worlds(1.67, 0.61), np.full(36, 1.1 * V_TRUE))
    nstep["strata"] = {s: nstep["strata"]["all"] for s in ra.STRATA}
    calibration["calibration"]["qol_nstep"] = nstep
    new = {c["id"]: c for c in ra.qol_checks(calibration, None, None)}
    check = new["qol:b:nmae:all"]
    assert check["value"] == nstep["strata"]["all"]["nmae"] != 0.1
    assert check["detail"]["target"] == pe.QOL_GATE_TARGET_ID
    assert check["detail"]["scale_source"] == "frozen-g0"
    assert check["detail"]["monte_carlo_non_gating"]["gating"] is False
    assert new["qol:b:abs-bias:all"]["value"] == pytest.approx(abs(nstep["strata"]["all"]["bias"]))


# --- 2. the reference-scale artifact --------------------------------------------------------------


def test_reference_scale_artifact_digest_and_loading(tmp_path) -> None:
    data = _two_worlds(1.67, 0.61)
    record = pe.qol_reference_scale_record(data, datasets=[{"paths": ["x"]}])
    returns = discounted_returns(data)["qol_years"].double().numpy()
    assert record["value"] == pytest.approx(float(np.std(returns)))
    assert record["schema"] == pe.QOL_REFERENCE_SCALE_SCHEMA and record["version"] == 1
    assert record["worlds"] == [90100, 90101] and record["previous"]["value"] == 43.152
    assert "observed_part_std" in record["diagnostics"]
    path = tmp_path / "scale.json"
    sealed = pe.write_qol_reference_scale(record, path)
    loaded = pe.load_qol_reference_scale(path)
    assert loaded["value"] == sealed["value"] and loaded["digest"] == sealed["digest"]
    assert loaded["source"] == "artifact" and loaded["sha256"] == ra.file_sha256(path)
    # Tampering breaks the digest.
    tampered = json.loads(path.read_text())
    tampered["value"] = 1.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="digest"):
        pe.load_qol_reference_scale(bad)
    # Absent: the frozen G0 scale.
    frozen = pe.load_qol_reference_scale(None)
    assert frozen["value"] == 43.152 and frozen["source"] == "frozen-g0"
    assert frozen["value"] == contracts.OBJECTIVE_REFERENCE_SCALES["qol_years"]


def test_reference_scale_command_uses_fitting_worlds_only(tmp_path, monkeypatch) -> None:
    from anthropocene_rl.autonomous_v7 import labor_teacher_data

    flat = _two_worlds(1.0, 1.0)  # every return equal: a zero scale is refused
    flat["collection"] = {"dataset_digest": "abc", "return_tail": None}
    monkeypatch.setattr(labor_teacher_data, "load_labor_data", lambda *a, **k: flat)
    out = tmp_path / "scale.json"
    args = argparse.Namespace(fit_data=[tmp_path / "b01"], version=2, out=out)
    with pytest.raises(ValueError, match="finite and positive"):
        ra.command_qol_reference_scale(args)
    data = _two_worlds(1.67, 0.61)
    data["collection"] = {"dataset_digest": "abc", "return_tail": None}
    monkeypatch.setattr(labor_teacher_data, "load_labor_data", lambda *a, **k: data)
    assert ra.command_qol_reference_scale(args) == 0
    loaded = pe.load_qol_reference_scale(out)
    sealed = json.loads(out.read_text())
    assert sealed["version"] == 2 and sealed["datasets"][0]["dataset_digest"] == "abc"
    assert "commit" in sealed["source"] and loaded["digest"] == sealed["digest"]
    held_out = _dataset([(90180, 1, 6, False, V_TRUE)])
    monkeypatch.setattr(labor_teacher_data, "load_labor_data", lambda *a, **k: held_out)
    with pytest.raises(ValueError, match="fitting worlds only"):
        ra.command_qol_reference_scale(args)


def test_td_residual_and_receipt_use_the_artifact_scale(tmp_path, monkeypatch) -> None:
    data = _two_worlds(1.0, 1.0)
    data["collection"] = {"return_tail": None}
    n = int(data["row_world"].shape[0])
    monkeypatch.setattr(
        pe, "critic_predictions", lambda *a, **k: {"qol_years": np.full(n, 1.2 * V_TRUE)}
    )
    reference = {"value": 10.0, "source": "artifact", "digest": "d" * 64}
    base = ra.td_residual_report(None, None, data)
    scaled = ra.td_residual_report(None, None, data, qol_reference=reference)
    assert base["scale"] == 43.152 and base["scale_source"] == "frozen-g0"
    assert scaled["scale"] == 10.0 and scaled["scale_digest"] == "d" * 64
    assert scaled["mae"] == pytest.approx(base["mae"] * 43.152 / 10.0)
    # The receipt records each fit's target and scale; one unit across fits PASSes.
    stamped = {
        "value": 61.0,
        "source": "artifact",
        "digest": "e" * 64,
        "path": "s.json",
        "sha256": "f" * 64,
    }
    evidence = [
        _evidence(seed=s, qol_reference_scale=stamped, qol_gate_target=pe.QOL_GATE_TARGET)
        for s in range(5)
    ] + [
        _evidence(
            "control-e4", s, False, qol_reference_scale=stamped, qol_gate_target=pe.QOL_GATE_TARGET
        )
        for s in range(5)
    ]
    receipt = _receipt(evidence)
    gate = receipt["qol_gate"]
    assert gate["status"] == "PASS" and gate["distinct"] == [
        {"value": 61.0, "digest": "e" * 64, "source": "artifact"}
    ]
    assert gate["fits"]["privileged-e4/seed-0"]["target"] == pe.QOL_GATE_TARGET_ID
    assert gate["definitions"] == [pe.QOL_GATE_TARGET]
    assert receipt["verdict_basis"]["qol_reference_scale"] == "PASS"
    # Mixed units cannot share a seed rule: the receipt HOLDs.
    evidence[0] = _evidence(seed=0)
    mixed = _receipt(evidence)
    assert mixed["qol_gate"]["status"] == "HOLD" and mixed["machine_verdict"] != "PASS"


def test_old_receipt_evidence_keeps_its_verdict() -> None:
    evidence = [_evidence(seed=s) for s in range(5)] + [
        _evidence("control-e4", s, False) for s in range(5)
    ]
    receipt = _receipt(evidence)
    assert receipt["qol_gate"]["status"] == "PASS"
    assert receipt["qol_gate"]["fits"]["privileged-e4/seed-0"]["scale"]["source"] == "frozen-g0"
    assert receipt["machine_verdict"] == "PASS"  # as test_labor_r5_acceptance's passing set


# --- 3. TD trajectory rule v2 -------------------------------------------------------------------


def _v2(series, floor=0.05):
    return trajectory_report_v2(
        series, warmup_factor=1.5, window=6, spread=0.5, floor=floor, ratio=0.5
    )


def _noise(level, count, amplitude=0.1):
    return [level * (1 + amplitude * math.sin(1.7 * i)) for i in range(count)]


def test_warmup_by_value() -> None:
    assert td_warmup_by_value([0.3, 0.1, 0.02, 0.021, 0.019, 0.02], 1.5)[0] == 2
    assert td_warmup_by_value([0.02, 0.3, 0.02], 1.5)[0] == 0  # only leading values drop
    assert td_warmup_by_value([], 1.5) == (0, None)


def test_stationary_noise_at_two_percent_passes() -> None:
    series = [0.2, 0.1, *_noise(0.02, 12)]
    report = _v2(series)
    assert report["warmup_dropped"] == 2 and report["status"] == "stationary-below-floor"
    assert report["pass"] and not report["declining"]
    # The pilot out-e5 error series (fractions of W), which v1 failed.
    pilot = [
        0.04198,
        0.03748,
        0.03916,
        0.03457,
        0.03198,
        0.02919,
        0.02791,
        0.02624,
        0.02559,
        0.02570,
        0.02442,
        0.02416,
        0.02354,
        0.02290,
        0.02327,
    ]
    assert _v2(pilot)["pass"]
    v1 = TDDriftMonitor(1.0, growth=3, floor=1e-3, tolerance=1e-6)
    assert v1.rule == TD_RULE_V1  # direct construction keeps v1


def test_stationary_above_the_floor_or_wide_fails_and_declining_passes() -> None:
    assert _v2(_noise(0.08, 10))["status"] == "not-declining-or-stationary"  # above 0.05 W
    assert not _v2([0.02, 0.04] * 5)["pass"]  # too wide: max - min > 0.5 x median
    declining = _v2([0.4 * 0.8**i for i in range(10)])
    assert declining["status"] == "declining" and declining["pass"]
    # A declining series above the floor passes on the declining branch alone.
    assert _v2([0.4 * 0.8**i for i in range(10)], floor=0.001)["pass"]


def test_too_few_usable_refreshes_hold() -> None:
    report = _v2([0.3, 0.2, 0.02, 0.02, 0.02])
    assert report["status"] == "insufficient" and report["pass"]


def test_divergence_needs_rises_above_the_stationary_median() -> None:
    base = _noise(0.02, 8, amplitude=0.02)
    assert (
        td_divergence_v2(
            [*base, 0.03, 0.04, 0.05], warmup_factor=1.5, growth=3, increase=0.10, median_factor=1.5
        )
        is not None
    )
    # Three >10% rises that stay inside the noise band are not divergence (v1 alarmed).
    quiet = [*base, 0.017, 0.0189, 0.021, 0.0233]
    assert (
        td_divergence_v2(quiet, warmup_factor=1.5, growth=3, increase=0.10, median_factor=1.5)
        is None
    )


def _feed(monitor, errors, drifts=None, weight=100.0):
    """One refresh per value: error and drift in fractions of W."""
    current = torch.full((10,), 40.0)
    monitor.td_refresh(0, current - 30.0, current, weight, growth_eligible=False, error=0.0)
    drifts = drifts or [0.01] * len(errors)
    for epoch, (error, drift) in enumerate(zip(errors, drifts, strict=True), start=1):
        previous = current
        current = current + drift * weight * (1 if epoch % 2 else -1)
        monitor.td_refresh(epoch, previous, current, weight, error=error * weight)


def test_genuine_error_growth_alarms_and_stops_the_fit() -> None:
    errors = [*_noise(0.02, 8, amplitude=0.02), 0.03, 0.04, 0.05, 0.06]
    with pytest.raises(StabilityAbort, match="stationary-median") as caught:
        _feed(_td_monitor(StabilityConfig(log_every=1, early_stop=True)), errors)
    summary = caught.value.summary
    assert summary["early_stop"]["check"] == "td_divergence"
    td = summary["checks"]["td_divergence"]
    assert td["status"] == "FAIL" and td["td_rule"] == TD_RULE_V2
    assert td["refreshes"][-1]["epoch"] == 11  # stopped at the third rise
    # Without early stop the same alarm FAILs at the end of the run.
    plain = _td_monitor(StabilityConfig(log_every=1))
    _feed(plain, errors)
    assert plain.finish()["checks"]["td_divergence"]["status"] == "FAIL"


def test_drift_is_advisory_and_stationary_error_passes() -> None:
    errors = [0.1, 0.06, *_noise(0.02, 10)]
    drifts = [0.01] * 6 + [0.02, 0.04, 0.08, 0.16, 0.30, 0.31]  # drift grows and is wide
    monitor = _td_monitor(StabilityConfig(log_every=1, early_stop=True))
    _feed(monitor, errors, drifts)  # no StabilityAbort
    summary = monitor.finish()
    td = summary["checks"]["td_divergence"]
    assert td["status"] == "PASS" and td["gated_series"] == "error"
    assert not td["trajectory"]["drift"]["pass"] and td["trajectory"]["drift"]["gating"] is False
    reasons = [a["reason"] for a in td["advisories"]]
    assert any(r.startswith("drift-rose") for r in reasons)
    assert "drift-trajectory-not-declining-or-stationary" in reasons
    assert any(a.get("gating") is False for a in summary["advisories"])  # live advisory


def test_short_v2_series_holds() -> None:
    monitor = _td_monitor(StabilityConfig(log_every=1))
    _feed(monitor, [0.02] * 4)
    td = monitor.finish()["checks"]["td_divergence"]
    assert td["status"] == "HOLD" and td["insufficient"] == ["error"]


def test_old_monitor_states_keep_the_v1_verdict(tmp_path) -> None:
    # A fit that v1 FAILs (flat noise above 1e-3 is not diminishing) and v2 PASSes.
    errors = _noise(0.02, 10)
    v1_config = StabilityConfig(log_every=1, thresholds=StabilityThresholds(td_rule=TD_RULE_V1))
    old = _td_monitor(v1_config)
    _feed(old, errors)
    state = old.state_dict()
    del state["td"]["rule"]  # a resume state written before 28 Oct
    del state["td"]["advisories"]
    restored = _td_monitor(StabilityConfig(log_every=1))  # the new default thresholds
    restored.load_state_dict(state)
    td = restored.finish()["checks"]["td_divergence"]
    assert td["status"] == "FAIL" and "td_rule" not in td
    new = _td_monitor(StabilityConfig(log_every=1))
    _feed(new, errors)
    state = new.state_dict()
    assert state["td"]["rule"] == TD_RULE_V2
    again = _td_monitor(StabilityConfig(log_every=1))
    again.load_state_dict(state)
    assert again.finish()["checks"]["td_divergence"]["status"] == "PASS"
    # An old stability receipt's recorded status is what the acceptance check reads.
    evidence = _evidence(seed=0)
    evidence["stability"]["critic"]["checks"]["td_divergence"]["status"] = "FAIL"
    rows = {c["id"]: c for c in ra.fit_checks(evidence)["primary"]}
    assert rows["stability:critic:td_divergence"]["status"] == "FAIL"


def test_rejudge_replays_a_log_and_reproduces_the_recorded_v1_verdict(tmp_path) -> None:
    log = tmp_path / "stability.jsonl"
    config = StabilityConfig(
        log_every=1, jsonl=str(log), thresholds=StabilityThresholds(td_rule=TD_RULE_V1)
    )
    monitor = _td_monitor(config)
    _feed(monitor, [0.1, 0.06, *_noise(0.02, 10)])
    recorded = monitor.finish()["checks"]["td_divergence"]["status"]
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    refreshes = [r for r in lines if r["kind"] == "td_refresh"]
    assert rejudge_td_refreshes(refreshes, rule=TD_RULE_V1)["status"] == recorded == "FAIL"
    v2 = rejudge_td_refreshes(refreshes)
    assert v2["status"] == "PASS" and v2["live_alarms"] == []
    out = tmp_path / "rejudge.json"
    args = argparse.Namespace(stability_log=[log], rule=TD_RULE_V2, out=out)
    assert ra.command_td_rejudge(args) == 0
    result = json.loads(out.read_text())[str(log)]
    assert result["status"] == "PASS" and result["recorded_status"] == "FAIL"


def test_v2_thresholds_are_preregistered() -> None:
    th = StabilityThresholds()
    assert th.td_rule == TD_RULE_V2 == ra.threshold("td_rule")
    assert (
        (th.td_warmup_median_factor, th.td_stationary_refreshes, th.td_stationary_spread)
        == (
            ra.threshold("td_warmup_median_factor"),
            ra.threshold("td_stationary_refreshes"),
            ra.threshold("td_stationary_spread"),
        )
        == (1.5, 6, 0.5)
    )
    assert th.td_stationary_error_floor == ra.threshold("td_stationary_error_floor") == 0.05
    assert th.td_stationary_drift_floor == ra.threshold("td_stationary_drift_floor") == 0.05
    assert th.td_divergence_median_factor == ra.threshold("td_divergence_median_factor") == 1.5
    assert "recalibrate" in ra.PREREGISTERED["td_stationary_error_floor"]["status"]
    with pytest.raises(ValueError, match="unknown TD trajectory rule"):
        TDDriftMonitor(1.0, growth=3, floor=1e-3, tolerance=1e-6, rule="v3")


def test_critic_dataset_report_carries_the_nstep_gate_section() -> None:
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
    dataset.setdefault("collection", {}).setdefault("dataset_digest", "test")
    reference = {"value": 10.0, "source": "artifact", "digest": "a" * 64}
    report = pe.critic_dataset_report(task, costs, dataset, batch_size=3, qol_reference=reference)
    nstep = report["qol_nstep"]
    assert nstep["target"]["id"] == pe.QOL_GATE_TARGET_ID and nstep["reference_scale"] == reference
    assert set(nstep["strata"]) <= set(report["heads"]["qol_years"])
    assert nstep["rows"] == report["rows"]
    # The Monte Carlo heads are unchanged by the new section (frozen model scale).
    assert report["heads"]["qol_years"]["all"]["scale"] == 43.152
    mc = nstep["strata"]["all"]["monte_carlo"]
    assert mc["nmae"] == pytest.approx(report["heads"]["qol_years"]["all"]["nmae"] * 43.152 / 10.0)
