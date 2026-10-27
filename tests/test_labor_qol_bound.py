"""Per-row admissible TD bound, the 2W QoL head and the tail-factor diagnostics (27 Oct).

1. The flat [0, W] TD target check fired on correct randomized-roulette overshoot (the
   tail weights gamma / rho exceed one, so single tail estimates exceed W). It is
   replaced by the per-row admissible bound U_i plus the reward / discount / head-range
   checks that catch the real bugs.
2. The bounded QoL head range widens from (0, W) to (0, 2W) under ``--campaign-r5``; an
   absent ``qolHeadScale`` is the historical head, bit for bit.
3. Per-world tail factor and the tail-weight-cut QoL bias are advisory diagnostics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from test_labor_early_stop import _td_monitor
from test_labor_model import _labor_tensors, _layout
from test_labor_qol_head import _training_dataset
from test_labor_return_tail import _dataset as _tail_dataset

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7.labor_advantage import (
    qol_saturation_report,
    qol_tail_terms,
    qol_tail_weight,
    qol_td_admissibility,
    qol_td_lambda_targets,
)
from anthropocene_rl.autonomous_v7.labor_bc import (
    RETURN_TAIL_RANDOMIZED,
    TAIL_HEADS,
    TAIL_NONE_TERMINAL,
    TAIL_RANDOMIZED,
    discounted_returns,
    pretrain_critic_on_dataset,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    LABOR_CAMPAIGN_R5_QOL_HEAD_SCALE,
    LABOR_QOL_HEAD_SCALE_FIELD,
    LaborCostCritics,
    LaborModelContract,
    LaborQoLHead,
    LaborTaskCritic,
    labor_qol_discount_weight,
)
from anthropocene_rl.autonomous_v7.labor_pretrain_eval import (
    qol_tail_diagnostics,
    tail_schedule_factor,
)
from anthropocene_rl.autonomous_v7.labor_r5_acceptance import fit_checks, qol_tail_advisory
from anthropocene_rl.autonomous_v7.labor_stability import (
    TD_REASON_ADMISSIBLE,
    TD_REASON_DISCOUNT,
    TD_REASON_FLAT_BOUND,
    TD_REASON_HEAD_RANGE,
    TD_REASON_REWARD,
    StabilityAbort,
    StabilityConfig,
    TDDriftMonitor,
    saturation_components,
)

W = labor_qol_discount_weight()
GAMMA = math.exp(-contracts.TURN_YEARS / contracts.HORIZON_YEARS)


def _trajectory(rewards, tail, *, terminal_last=False, lam_rows=None):
    """One (world, camp) trajectory of len(rewards) rows closed by ``tail``."""
    n = len(rewards)
    codes = [TAIL_NONE_TERMINAL if terminal_last else TAIL_RANDOMIZED] * n
    data = _tail_dataset(
        codes, [0.0 if terminal_last else tail] * n, [False] * (n - 1) + [terminal_last],
        camps=(4,) * n, turns=tuple(range(n)),
    )
    data["reward_qol_years"] = torch.tensor(rewards, dtype=torch.float64)
    return data


def _upper_reference(rewards, tail, values_max, lam):
    """Independent forward statement of U_i (closed form for a single trajectory)."""
    n = len(rewards)
    out = [0.0] * n
    u_next = v_next = max(tail, values_max)
    for i in reversed(range(n)):
        out[i] = rewards[i] + GAMMA * ((1 - lam) * v_next + lam * u_next)
        u_next, v_next = out[i], values_max
    return out


# --- 1. admissible bound ------------------------------------------------------------------


@pytest.mark.parametrize("lam", [0.0, 0.8, 1.0])
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_upper_bound_recursion_on_synthetic_trajectories(lam: float, scale: float) -> None:
    rewards = [20.0, 5.0, 12.0, 20.0]
    tail = 1.54 * W  # roulette overshoot, as fit8-b04 world 90129
    data = _trajectory(rewards, tail)
    head_max = scale * W
    values = torch.tensor([0.3 * W, 0.9 * W, head_max, 0.1 * W])
    result = qol_td_admissibility(
        data, values, lam, head_max=head_max, weight=W, discount_years=contracts.HORIZON_YEARS
    )
    expected = _upper_reference(rewards, tail, head_max, lam)
    assert torch.allclose(result["upper"], torch.tensor(expected, dtype=torch.float64))
    targets = qol_td_lambda_targets(data, values, lam).double()
    assert bool((targets <= result["upper"] * (1 + 1e-6)).all())
    assert (result["reward_violations"], result["discount_violations"]) == (0, 0)
    assert result["head_violations"] == 0
    diag = result["diagnostics"]
    assert diag["tail_above_w_rows"] == 4 and diag["tail_above_w_share"] == 1.0
    assert math.isclose(diag["max_tail_over_w"], 1.54)
    assert diag["gating"] is False
    # The last row bounds its tail term by max(tail, head maximum).
    assert math.isclose(float(result["upper"][-1]), 20.0 + GAMMA * max(tail, head_max))


def test_terminal_rows_bound_by_their_reward_and_rows_select() -> None:
    data = _trajectory([4.0, 6.0, 8.0], 0.0, terminal_last=True)
    values = torch.tensor([0.5 * W, 0.5 * W, 0.5 * W])
    result = qol_td_admissibility(
        data, values, 0.8, head_max=W, weight=W, discount_years=contracts.HORIZON_YEARS,
        rows=torch.tensor([1, 2]),
    )
    assert result["upper"].shape == (2,)
    assert float(result["upper"][1]) == 8.0  # terminal: U = r
    assert math.isclose(float(result["upper"][0]), 6.0 + GAMMA * (0.2 * W + 0.8 * 8.0))
    assert result["diagnostics"]["tail_rows"] == 0


def _monitor_update(data, values, lam=0.8, head_max=W, discount_years=None, targets=None):
    discount_years = contracts.HORIZON_YEARS if discount_years is None else discount_years
    adm = qol_td_admissibility(
        data, values, lam, head_max=head_max, weight=W, discount_years=discount_years
    )
    targets = qol_td_lambda_targets(data, values, lam) if targets is None else targets
    monitor = TDDriftMonitor(W, growth=3, floor=1e-3, tolerance=1e-6)
    return monitor.update(0, None, targets, growth_eligible=False, admissibility=adm), monitor


def test_roulette_overshoot_does_not_alarm_but_the_flat_bound_did() -> None:
    data = _trajectory([20.0, 20.0, 20.0], 1.54 * W)
    values = torch.full((3,), 0.95 * W)
    targets = qol_td_lambda_targets(data, values, 0.8)
    assert float(targets.max()) > W  # correct estimator noise above W
    alarm, monitor = _monitor_update(data, values)
    assert alarm is None
    record = monitor.refreshes[-1]
    assert record["bound"] == "admissible" and record["out_of_bounds"] == 0
    assert record["diagnostics"]["tail_above_w_share"] == 1.0
    # The historical flat check (no admissibility) fires on the same targets.
    flat = TDDriftMonitor(W, growth=3, floor=1e-3, tolerance=1e-6)
    assert flat.update(0, None, targets)["reason"] == TD_REASON_FLAT_BOUND


def test_reward_above_dt_alarms() -> None:
    data = _trajectory([21.0, 5.0, 5.0], 0.5 * W)  # r = 21 > dt = 20
    alarm, monitor = _monitor_update(data, torch.full((3,), 0.5 * W))
    assert alarm["check"] == "td_divergence" and alarm["reason"] == TD_REASON_REWARD
    assert monitor.refreshes[-1]["reward_violations"] == 1
    negative = _trajectory([-0.5, 5.0, 5.0], 0.5 * W)
    assert _monitor_update(negative, torch.full((3,), 0.5 * W))[0]["reason"] == TD_REASON_REWARD


def test_wrong_gamma_alarms() -> None:
    data = _trajectory([5.0, 5.0, 5.0], 0.5 * W)
    alarm, _ = _monitor_update(data, torch.full((3,), 0.5 * W), discount_years=300.0)
    assert alarm["reason"] == TD_REASON_DISCOUNT
    bad = _trajectory([5.0, 5.0, 5.0], 0.5 * W)
    bad["transition_elapsed_years"] = torch.tensor([20.0, float("nan"), 20.0], dtype=torch.float64)
    adm = qol_td_admissibility(
        bad, torch.full((3,), 0.5 * W), 0.8, head_max=W, weight=W,
        discount_years=contracts.HORIZON_YEARS,
    )
    assert adm["discount_violations"] >= 1


def test_decoded_head_outside_range_alarms() -> None:
    data = _trajectory([5.0, 5.0, 5.0], 0.5 * W)
    alarm, _ = _monitor_update(data, torch.tensor([0.5 * W, 1.2 * W, 0.5 * W]))
    assert alarm["reason"] == TD_REASON_HEAD_RANGE
    # The same value is inside a 2W head's range.
    wide = _monitor_update(data, torch.tensor([0.5 * W, 1.2 * W, 0.5 * W]), head_max=2 * W)
    assert wide[0] is None


def test_target_above_its_admissible_bound_alarms() -> None:
    data = _trajectory([5.0, 5.0, 5.0], 0.5 * W)
    values = torch.full((3,), 0.5 * W)
    targets = qol_td_lambda_targets(data, values, 0.8).clone()
    targets[0] = 1.0 * W + 30.0  # above U_0 (< W here), e.g. a corrupted refresh
    alarm, _ = _monitor_update(data, values, targets=targets)
    assert alarm["reason"] == TD_REASON_ADMISSIBLE and alarm["out_of_bounds"] == 1
    targets[0] = -1.0
    assert _monitor_update(data, values, targets=targets)[0]["reason"] == TD_REASON_ADMISSIBLE


def test_early_stop_on_the_new_and_the_old_reason() -> None:
    data = _trajectory([21.0, 5.0, 5.0], 0.5 * W)
    values = torch.full((3,), 0.5 * W)
    adm = qol_td_admissibility(
        data, values, 0.8, head_max=W, weight=W, discount_years=contracts.HORIZON_YEARS
    )
    monitor = _td_monitor(StabilityConfig(log_every=1, early_stop=True))
    with pytest.raises(StabilityAbort, match="rewards-outside") as caught:
        monitor.td_refresh(
            0, None, qol_td_lambda_targets(data, values, 0.8), W, growth_eligible=False,
            admissibility=adm,
        )
    assert caught.value.summary["early_stop"]["check"] == "td_divergence"
    assert caught.value.summary["checks"]["td_divergence"]["status"] == "FAIL"
    # Callers without the bound keep the flat check under its historical id.
    old = _td_monitor(StabilityConfig(log_every=1, early_stop=True))
    with pytest.raises(StabilityAbort, match=r"targets-outside-\[0,W\]"):
        old.td_refresh(0, None, torch.full((4,), 1.5 * W), W, growth_eligible=False)
    # A resume state carrying an old-id alarm still FAILs the check (old receipts).
    plain = _td_monitor(StabilityConfig(log_every=1))
    plain.td_refresh(0, None, torch.full((4,), 1.5 * W), W, growth_eligible=False)
    restored = _td_monitor(StabilityConfig(log_every=1))
    restored.load_state_dict(plain.state_dict())
    td = restored.finish()["checks"]["td_divergence"]
    assert td["status"] == "FAIL" and td["alarms"][0]["reason"] == TD_REASON_FLAT_BOUND


# --- end-to-end: a TD fit on roulette overshoot ------------------------------------------


def _randomized_dataset(tail: float) -> dict:
    data = _training_dataset()
    data["reward_qol_years"] = torch.full((6,), 18.0, dtype=torch.float64)
    data["transition_terminal"] = torch.zeros(6, dtype=torch.bool)
    data["tail_method"] = torch.full((6,), TAIL_RANDOMIZED, dtype=torch.long)
    data["tail_cutoff_turn"] = torch.full((6,), 3, dtype=torch.long)
    for head in TAIL_HEADS:
        value = tail if head == "qol_years" else 0.0
        data["tail_value__" + head] = torch.full((6,), value, dtype=torch.float64)
        data["tail_standard_error__" + head] = torch.zeros(6, dtype=torch.float64)
    data["collection"] = {"return_tail": RETURN_TAIL_RANDOMIZED}
    return data


def _td_fit(scale: float, tail: float, **kwargs):
    torch.manual_seed(0)
    head = LaborQoLHead(target="td-lambda", scale=scale)
    task = LaborTaskCritic(_layout(), width=16, token_width=16, qol_head=head)
    costs = LaborCostCritics(_layout(), width=16, token_width=16)
    summary = pretrain_critic_on_dataset(
        task_critic=task, cost_critics=costs, dataset=_randomized_dataset(tail), epochs=2,
        batch_size=4, stability=StabilityConfig(log_every=1, early_stop=True), **kwargs
    )
    return summary, task


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_step0_refresh_on_overshooting_tails_no_longer_alarms(scale: float) -> None:
    summary, _ = _td_fit(scale, 1.5 * W)  # early stop on: an alarm would raise
    refreshes = summary["stability"]["checks"]["td_divergence"]["refreshes"]
    first = refreshes[0]
    assert first["epoch"] == 0 and first["target_max"] > W
    assert first["bound"] == "admissible" and first["out_of_bounds"] == 0
    assert all(r["reward_violations"] == r["discount_violations"] == 0 for r in refreshes)
    assert all(r["head_violations"] == 0 for r in refreshes)
    assert first["diagnostics"]["tail_above_w_share"] == 1.0
    clipped = [r["diagnostics"]["clipped_targets"] for r in refreshes]
    if scale == 2.0:
        assert clipped == [0] * len(clipped) and summary["qol_clipped_targets"] == 0
    else:
        assert clipped[0] > 0 and "qol_clipped_targets" not in summary


# --- 2. head scale ------------------------------------------------------------------------


def test_scale_one_is_the_historical_head_bit_for_bit() -> None:
    head = LaborQoLHead()
    state = head.state_dict()
    assert LABOR_QOL_HEAD_SCALE_FIELD not in state and head.scale == 1.0
    assert LaborQoLHead(scale=1.0) == head and LaborQoLHead.from_state(state) == head
    critic = LaborTaskCritic(_layout(), width=24, token_width=16, qol_head=head)
    z = torch.tensor([[-9.0], [-0.3], [0.0], [2.5], [11.0]])
    assert torch.equal(critic.bounded_mean(z), torch.sigmoid(z.squeeze(-1)))
    target = torch.tensor([0.0, 0.2 * W, 0.5 * W, W, 1.6 * W])
    assert torch.equal(
        critic.value_loss("qol_years", z, target),
        torch.nn.functional.binary_cross_entropy_with_logits(
            z.squeeze(-1), (target / W).clamp(0.0, 1.0), reduction="none"
        ),
    )
    silu = LaborTaskCritic(
        _layout(), width=24, token_width=16, qol_head=LaborQoLHead("silu-clamp")
    )
    assert torch.equal(
        silu.bounded_mean(z), torch.nn.functional.silu(z.squeeze(-1)).clamp(0.0, 1.0)
    )


def test_old_checkpoint_contract_without_the_field_loads_as_scale_one() -> None:
    contract = LaborModelContract(qol_head=LaborQoLHead(target="td-lambda"))
    state = contract.state_dict()
    assert LABOR_QOL_HEAD_SCALE_FIELD not in state["criticQoLHead"]
    loaded = LaborModelContract.from_state(state)
    assert loaded.qol_head.scale == 1.0 and loaded.state_dict() == state
    wide = LaborModelContract(qol_head=LaborQoLHead(target="td-lambda", scale=2.0))
    wide_state = wide.state_dict()
    assert wide_state["criticQoLHead"][LABOR_QOL_HEAD_SCALE_FIELD] == 2.0
    assert LaborModelContract.from_state(wide_state) == wide
    assert wide != contract  # a scale mismatch is a different contract (load rejects it)
    with pytest.raises(ValueError, match="canonical"):  # explicit 1.0 is never written
        LaborQoLHead.from_state(dict(state["criticQoLHead"], qolHeadScale=1.0))
    with pytest.raises(ValueError):
        LaborQoLHead(scale=0.5)


def test_wide_checkpoint_round_trip_and_mismatch_rejection() -> None:
    from test_labor_qol_head import _models, _normalizer, _outputs

    import anthropocene_rl.autonomous_v7.labor_model as current

    layout = current.LaborEncoderLayout(*_layout().state_dict().values())
    head = LaborQoLHead(scale=2.0)
    models = _models(current, qol_head=head)
    payload = current.labor_checkpoint_payload(*models, _normalizer(), layout)
    assert current.labor_checkpoint_qol_head(payload) == head
    restored = _models(current, seed=8, qol_head=head)
    current.load_labor_checkpoint_models(payload, *restored, expected_layout=layout)
    tensors = _labor_tensors()
    left, right = _outputs(*models, tensors), _outputs(*restored, tensors)
    for key in left:
        assert torch.equal(left[key], right[key]), key
    with pytest.raises(ValueError, match="different QoL head"):
        current.load_labor_checkpoint_models(
            payload, *_models(current, qol_head=LaborQoLHead()), expected_layout=layout
        )


@pytest.mark.parametrize("output", ["logistic", "silu-clamp"])
def test_two_w_head_forward_loss_and_decode(output: str) -> None:
    head = LaborQoLHead(output, scale=2.0)
    assert math.isclose(head.head_max, 2.0 * W)
    critic = LaborTaskCritic(_layout(), width=24, token_width=16, qol_head=head)
    out = critic(_labor_tensors())["qol_years"]
    value = critic.expectation("qol_years", out)
    assert ((value >= 0) & (value <= 2.0 * W)).all()
    z = torch.tensor([[-8.0], [0.0], [0.5], [12.0]], requires_grad=True)
    m = critic.bounded_mean(z)
    if output == "logistic":
        assert torch.allclose(m, 2.0 * torch.sigmoid(z.squeeze(-1)))
        assert math.isclose(float(critic.expectation("qol_years", z.detach())[1]), W, rel_tol=1e-6)
    else:
        assert torch.allclose(m, 2.0 * torch.nn.functional.silu(z.squeeze(-1)).clamp(0, 1))
        assert float(m[3].detach()) == 2.0
    target = torch.tensor([0.0, W, 1.54 * W, 2.5 * W])
    p = torch.tensor([0.0, 0.5, 0.77, 1.0])  # normalized by 2W; 2.5W hits the guard clamp
    loss = critic.value_loss("qol_years", z, target)
    if output == "logistic":
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            z.squeeze(-1), p, reduction="none"
        )
    else:
        expected = (torch.nn.functional.silu(z.squeeze(-1)) - p).square()
    assert torch.allclose(loss, expected)
    loss.sum().backward()
    assert z.grad.abs().sum() > 0
    assert critic.qol_clip_count(target) == 1
    assert critic.qol_clip_count(target[:3]) == 0
    assert LaborTaskCritic(
        _layout(), width=24, token_width=16, qol_head=LaborQoLHead(output)
    ).qol_clip_count(target) == 2


def test_parse_and_campaign_r5_initialization_scale() -> None:
    import argparse

    from anthropocene_rl.autonomous_v7.labor_training import _initialize_qol_head

    assert LaborQoLHead.parse("logistic,td-lambda,0.8").scale == 1.0
    assert LaborQoLHead.parse("logistic,td-lambda,0.8", 2.0).scale == 2.0
    with pytest.raises(ValueError, match="bounded QoL head"):
        LaborQoLHead.parse(None, 2.0)

    def ns(**kw):
        return argparse.Namespace(**{"qol_head": "logistic,td-lambda,0.8", **kw})

    assert _initialize_qol_head(ns(campaign_r5=True)).scale == LABOR_CAMPAIGN_R5_QOL_HEAD_SCALE
    assert LABOR_CAMPAIGN_R5_QOL_HEAD_SCALE == 2.0
    assert _initialize_qol_head(ns(campaign_r5=False)).scale == 1.0
    assert _initialize_qol_head(ns(campaign_r5=True, qol_head_scale=1.0)).scale == 1.0
    assert _initialize_qol_head(ns(qol_head=None, campaign_r5=True)) is None


def test_saturation_stratum_keeps_its_physical_meaning() -> None:
    # Targets are passed as G / W whatever the head scale: G < 0.05 W is expected-low.
    z = torch.tensor([-7.0, -7.0, -7.0, 0.0])
    fraction = torch.tensor([0.04, 0.06, 0.03, 0.5])
    early = torch.zeros(4, dtype=torch.bool)
    one = saturation_components(z, fraction, early)
    two = saturation_components(z, fraction, early, head_scale=2.0)
    for key in ("expected_low_share", "rest_unexpected_share", "alarm", "low_share"):
        assert one[key] == two[key], key
    assert one["expected_low_share"] == 0.5 and one["rest_unexpected_share"] == 0.25
    assert "head_scale" not in one
    assert two["head_scale"] == 2.0 and math.isclose(two["expected_low_head_fraction"], 0.025)
    # The report of a 2W critic carries the scale through.
    critic = LaborTaskCritic(
        _layout(), width=24, token_width=16, qol_head=LaborQoLHead(scale=2.0)
    )
    report = qol_saturation_report(
        critic, z.reshape(-1, 1), target_fraction=fraction, early=early
    )
    assert report["components"]["head_scale"] == 2.0
    assert report["components"]["expected_low_share"] == 0.5


# --- 3. tail-factor diagnostics -----------------------------------------------------------


def test_schedule_factor_and_tail_weight() -> None:
    record = {"rho": 0.95, "stop_indices": [0, 3, 10]}
    ratio = GAMMA / 0.95
    expected = 20.0 * np.mean([sum(ratio**k for k in range(s + 1)) for s in (0, 3, 10)]) / W
    assert math.isclose(tail_schedule_factor(record, W), expected)
    assert tail_schedule_factor({"rho": 0.95}, W) is None
    data = _trajectory([10.0, 10.0], 2.0 * W)
    terms = qol_tail_terms(data, 1.0)
    expected = torch.tensor([GAMMA**2, GAMMA], dtype=torch.float64)
    assert torch.allclose(terms["coefficient"], expected)
    mc = discounted_returns(data)["qol_years"].double()
    weight = qol_tail_weight(data, mc, 1.0)
    assert torch.allclose(weight, terms["coefficient"] * 2.0 * W / mc)
    assert bool((weight > 0.5).all())
    lam = qol_tail_terms(data, 0.8)["coefficient"]
    assert math.isclose(float(lam[0]), 0.8 * GAMMA * GAMMA)


def _two_world_dataset():
    """World 1: short trajectory dominated by its tail; world 2: no tail (terminal)."""
    a = _trajectory([10.0, 10.0], 1.6 * W)
    b = _trajectory([15.0, 15.0, 15.0], 0.0, terminal_last=True)
    data = {k: torch.cat([a[k], b[k]]) for k in a}
    data["row_world"] = torch.tensor([90100, 90100, 90101, 90101, 90101])
    data["collection"] = {
        "return_tail": RETURN_TAIL_RANDOMIZED,
        "return_tail_by_world": {
            "90100": {"rho": 0.95, "stop_indices": [61, 54, 30]},
            "90101": {"rho": 0.95, "stop_indices": [0, 1, 2]},
        },
    }
    return data


def test_qol_tail_diagnostics_per_world_and_cut_bias() -> None:
    data = _two_world_dataset()
    target = discounted_returns(data)["qol_years"].double().numpy()
    pred = target + np.array([40.0, 40.0, 4.0, 4.0, 4.0])
    report = qol_tail_diagnostics(data, np.arange(5), pred, target, 43.152, W)
    assert report["gating"] is False and set(report["worlds"]) == {"90100", "90101"}
    first, second = report["worlds"]["90100"], report["worlds"]["90101"]
    assert first["schedule_factor"] > 1.0 > second["schedule_factor"]
    assert first["tail_above_w_share"] == 1.0 and second["tail_above_w_share"] == 0.0
    assert first["tail_weight_mean"] > 0.5 and second["tail_weight_mean"] == 0.0
    assert math.isclose(first["max_tail_over_w"], 1.6)
    bias = report["bias"]
    # World-balanced: (40 + 4) / 2 on all rows; the tail-dominated world drops out at the cut.
    assert math.isclose(bias["all"], 22.0 / 43.152)
    assert math.isclose(bias["tail_weight_le_cut"], 4.0 / 43.152)
    assert bias["rows_tail_weight_le_cut"] == 3


def test_receipt_advisory_reports_and_never_flags() -> None:
    data = _two_world_dataset()
    target = discounted_returns(data)["qol_years"].double().numpy()
    tail = qol_tail_diagnostics(data, np.arange(5), target, target, 43.152, W)
    evidence = {"calibration": {"calibration": {"qol_tail": tail, "heads": {}}}}
    advisory = qol_tail_advisory(evidence)
    assert advisory["status"] == "REPORT" and advisory["gating"] is False
    assert advisory["calibration"]["worlds"]["90100"]["schedule_factor"] > 1.0
    assert qol_tail_advisory({})["status"] == "NOT-APPLICABLE"
    checks = fit_checks(evidence)
    names = [a["check"] for a in checks["advisory"]]
    assert names[0] == "advisory:food-trade-vs-all" and "advisory:qol-tail-factor" in names
    assert all("qol-tail" not in str(c) for c in (*checks["primary"], *checks["secondary"]))
