"""r5 critic loss balancing (user-approved 29 Oct): GradNorm task-head weights, per-encoder
gradient clipping and the gating task_progress_stall check that replaces gradient_dominance
under gradnorm."""

from __future__ import annotations

import itertools
import json
import math

import pytest
import torch
from test_labor_critic_balance import _equal, _fit, _lines, _starved_dataset
from torch import nn

from anthropocene_rl.autonomous_v7 import labor_training as lt
from anthropocene_rl.autonomous_v7.labor_bc import (
    GradNormBalancing,
    GradNormState,
    gradnorm_shared_parameters,
)
from anthropocene_rl.autonomous_v7.labor_r5_acceptance import PREREGISTERED, stability_checks
from anthropocene_rl.autonomous_v7.labor_stability import (
    EARLY_STOP_CHECKS,
    DominanceDetector,
    StabilityAbort,
    StabilityConfig,
    StabilityMonitor,
    clip_grad_groups,
)

HEADS = ("qol_years", "population_years")


# --- 1. GradNorm arithmetic ----------------------------------------------------------------


def test_gradnorm_weights_move_a_two_head_imbalance_toward_balance() -> None:
    # Head a's raw gradient on W is 100x head b's at equal training progress: GradNorm must
    # move the weights until the weighted norms w_i g_i meet (w_a / w_b -> 1 / 100).
    state = GradNormState(GradNormBalancing(lr=0.05), ("a", "b"), {"a": 1.0, "b": 1.0})
    raw, losses = {"a": 100.0, "b": 1.0}, {"a": 1.0, "b": 1.0}
    first = state.proposal(raw, losses)
    assert first["weighted_grad_norm"] == {"a": 100.0, "b": 1.0}
    assert first["target_grad_norm"] == {"a": 50.5, "b": 50.5}  # equal rates: the mean
    assert first["weight_grad"] == {"a": 100.0, "b": -1.0}  # sign(G - target) g
    for _ in range(300):
        weights = state.commit(state.proposal(raw, losses))
        assert sum(weights.values()) == pytest.approx(2.0, abs=1e-12)
    ratio = (weights["a"] * 100.0) / (weights["b"] * 1.0)
    assert 0.5 < ratio < 2.0
    assert weights["b"] > 1.9 and weights["a"] < 0.05


def test_gradnorm_raises_the_weight_of_the_slower_head() -> None:
    state = GradNormState(GradNormBalancing(), ("a", "b"), {"a": 1.0, "b": 1.0})
    state.commit(state.proposal({"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 2.0}))  # L(0) = 2
    proposal = state.proposal({"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 1.0})
    # a has not trained (ratio 1), b halved (0.5): rates 4/3 and 2/3, alpha 1.5.
    assert proposal["inverse_rate"]["a"] == pytest.approx(4.0 / 3.0)
    assert proposal["target_grad_norm"]["a"] > proposal["target_grad_norm"]["b"]
    for _ in range(50):
        weights = state.commit(state.proposal({"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 1.0}))
    assert weights["a"] > weights["b"]


def test_gradnorm_initial_ratio_floor_and_state_roundtrip() -> None:
    initial = {"qol_years": 30.0, "population_years": 1.0}
    state = GradNormState(GradNormBalancing(), HEADS, initial)
    assert state.values() == pytest.approx({"qol_years": 60 / 31, "population_years": 2 / 31})
    for _ in range(200):  # push b towards zero: the floor keeps it positive
        state.commit(state.proposal({"qol_years": 1e-6, "population_years": 1e3},
                                    {"qol_years": 1.0, "population_years": 1.0}))
    assert min(state.values().values()) >= 1e-3 * 0.98  # floor, then renormalised
    copy_ = GradNormState(GradNormBalancing(), HEADS, {"qol_years": 1.0, "population_years": 1.0})
    copy_.load_state_dict(state.state_dict())
    step = (
        {"qol_years": 0.3, "population_years": 0.7},
        {"qol_years": 0.9, "population_years": 0.4},
    )
    assert state.commit(state.proposal(*step)) == copy_.commit(copy_.proposal(*step))
    with pytest.raises(ValueError):
        GradNormBalancing(lr=0.0)


# --- 2. per-encoder clip -------------------------------------------------------------------


def test_per_encoder_clip_bounds_each_group_and_reports_the_global_norm() -> None:
    a, b, c = (nn.Parameter(torch.zeros(3)) for _ in range(3))
    a.grad = torch.tensor([30.0, 40.0, 0.0])  # norm 50 -> clipped to 1
    b.grad = torch.tensor([0.0, 0.0, 120.0])  # same group as a: joint norm 130
    c.grad = torch.tensor([0.3, 0.4, 0.0])  # norm 0.5: under the limit, untouched
    total, norms = clip_grad_groups({"task": [a, b], "cost": [c]}, 1.0)
    assert norms["task"] == pytest.approx(130.0) and norms["cost"] == pytest.approx(0.5)
    assert total == pytest.approx(math.sqrt(130.0**2 + 0.5**2))
    joint = torch.cat([a.grad, b.grad]).norm()
    assert float(joint) == pytest.approx(1.0, rel=1e-5)
    assert torch.allclose(a.grad, torch.tensor([30.0, 40.0, 0.0]) / (130.0 + 1e-6))
    assert torch.equal(c.grad, torch.tensor([0.3, 0.4, 0.0]))
    # A global clip would have scaled the small group down too.
    c.grad = torch.tensor([0.3, 0.4, 0.0])
    a.grad, b.grad = torch.tensor([30.0, 40.0, 0.0]), torch.tensor([0.0, 0.0, 120.0])
    nn.utils.clip_grad_norm_([a, b, c], 1.0)
    assert float(c.grad.norm()) < 0.01


# --- 3. task_progress_stall ----------------------------------------------------------------


def _stall_detector():
    return DominanceDetector(
        "task_encoder", HEADS, window=8, share=0.9, persistence=0.75, min_progress=0.01,
        stall=True,
    )


def test_task_progress_stall_fires_on_a_flat_head_and_not_on_progress() -> None:
    flat = _stall_detector()
    alarms = []
    for step in range(8):
        flat.update(step, {"qol_years": 1.0, "population_years": 1.0},
                    {"qol_years": 2.0 * 0.9**step, "population_years": 3.0})
        alarms.append(flat.progress_stall(step))
    assert alarms[:7] == [None] * 7
    assert alarms[7]["check"] == "task_progress_stall"
    assert list(alarms[7]["stalled_progress"]) == ["population_years"]
    assert not flat.alarms  # balanced norms: no dominance at all, yet the stall fires
    moving = _stall_detector()
    for step in range(12):
        moving.update(step, {"qol_years": 1.0, "population_years": 1.0},
                      {"qol_years": 2.0 * 0.9**step, "population_years": 3.0 * 0.95**step})
        assert moving.progress_stall(step) is None
    off = DominanceDetector("x", HEADS, window=8, share=0.9, persistence=0.75, min_progress=0.01)
    for step in range(8):
        off.update(step, {"qol_years": 1.0, "population_years": 1.0},
                   {"qol_years": 1.0, "population_years": 1.0})
    assert off.progress_stall(7) is None and "stall_alarms" not in off.report()


def _shared_monitor(config, loss_balancing):
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
        loss_balancing=loss_balancing,
    )
    return monitor, optimizer, shared, head_a, head_b


def _run_stalled(monitor, optimizer, shared, head_a, head_b, steps=20):
    x = torch.randn(32, 4)
    for _ in range(steps):  # the toy of test_labor_stability's dominance wiring test
        optimizer.zero_grad()
        hidden = shared(x)
        loss_a = (head_a(hidden) - 1e3).square().mean()
        loss_b = (head_b(hidden.detach()) - 5.0).square().mean().detach() + 0.0 * head_b.bias.sum()
        monitor.step(loss_a + loss_b, max_grad_norm=1.0, head_losses={"a": loss_a, "b": loss_b})


def test_under_gradnorm_dominance_is_advisory_and_the_stall_gates() -> None:
    monitor, *toy = _shared_monitor(StabilityConfig(log_every=1), "gradnorm")
    _run_stalled(monitor, *toy)
    summary = monitor.finish()
    dominance = summary["checks"]["gradient_dominance"]
    assert dominance["status"] == "FAIL" and not dominance["mandatory"] and dominance["advisory"]
    stall = summary["checks"]["task_progress_stall"]
    assert stall["status"] == "FAIL" and stall["mandatory"]
    assert summary["mandatory_failed"] == ["task_progress_stall"]
    assert all(a["check"] == "task_progress_stall" for a in summary["alarms"])
    assert any(a["check"] == "gradient_dominance" for a in summary["advisories"])
    assert summary["loss_balancing"] == "gradnorm"
    # The acceptance seed gate / receipt rows gate the stall and skip the advisory check.
    rows = {r["id"]: r for r in stability_checks("critic", summary)}
    assert rows["stability:critic:task_progress_stall"]["status"] == "FAIL"
    assert "stability:critic:gradient_dominance" not in rows
    assert "task_progress_stall" in EARLY_STOP_CHECKS
    assert PREREGISTERED["stability_task_progress_stall"]["status"] == "approved (user 29 Oct)"


def test_the_stall_stops_a_gradnorm_fit_early() -> None:
    monitor, *toy = _shared_monitor(StabilityConfig(log_every=1, early_stop=True), "gradnorm")
    with pytest.raises(StabilityAbort) as error:
        _run_stalled(monitor, *toy)
    assert error.value.summary["early_stop"]["check"] == "task_progress_stall"
    assert error.value.summary["early_stop"]["step"] == 16  # first full 16-point window


def test_without_gradnorm_the_receipt_is_unchanged() -> None:
    monitor, *toy = _shared_monitor(StabilityConfig(log_every=1), None)
    _run_stalled(monitor, *toy)
    summary = monitor.finish()
    assert "task_progress_stall" not in summary["checks"]
    assert summary["checks"]["gradient_dominance"]["mandatory"]
    assert "advisory" not in summary["checks"]["gradient_dominance"]
    assert "loss_balancing" not in summary and "grad_clip" not in summary
    assert "stall_alarms" not in summary["checks"]["gradient_dominance"]["detectors"]["shared"]


# --- 4. the critic fit ---------------------------------------------------------------------


def _gn_fit(tmp_path, name, **kwargs):
    return _fit(
        dataset=_starved_dataset(),
        epochs=4,
        stability=StabilityConfig(log_every=1, jsonl=str(tmp_path / name)),
        **kwargs,
    )


GN = {"loss_balancing": GradNormBalancing(), "task_loss_weights": {"qol_years": 30.0}}


def test_flags_off_is_the_historical_fit_bitwise(tmp_path) -> None:
    for monitored in (False, True):
        config = StabilityConfig(log_every=1) if monitored else None
        weights = {"task_loss_weights": {"qol_years": 30.0}}
        plain, plain_summary, plain_rng = _fit(stability=config, **weights)
        explicit, explicit_summary, explicit_rng = _fit(
            stability=config, loss_balancing=None, grad_clip="global", **weights
        )
        assert _equal(plain, explicit) and plain_rng == explicit_rng
        assert plain_summary == explicit_summary or monitored  # monitored: timings differ
        assert plain_summary["final_loss"] == explicit_summary["final_loss"]
        assert "grad_clip" not in plain_summary
        assert plain_summary["task_loss_weights"]["mode"] == "fixed"
    # A resume state of the historical fit stays valid (fingerprint unchanged).
    path = str(tmp_path / "plain.pt")
    _fit(max_steps=3, resume_path=path, resume_every=1, **weights)
    _fit(resume_path=path, resume_every=1, loss_balancing=None, grad_clip="global", **weights)


def test_gradnorm_fit_logs_weights_every_step_and_balances(tmp_path) -> None:
    state, summary, rng = _gn_fit(tmp_path, "gn.jsonl", **GN)
    again, _, again_rng = _gn_fit(tmp_path, "gn2.jsonl", **GN)
    assert _equal(state, again) and rng == again_rng
    _, _, fixed_rng = _gn_fit(tmp_path, "fixed.jsonl", task_loss_weights={"qol_years": 30.0})
    assert rng == fixed_rng  # the trainer generator is not consumed
    record = summary["task_loss_weights"]
    assert record["mode"] == "gradnorm" and record["initial_ratio"] == {"qol_years": 30.0}
    assert set(record["initial_losses"]) == set(HEADS)
    assert sum(record["final"].values()) == pytest.approx(2.0)
    steps = [s for s in _lines(tmp_path / "gn.jsonl") if s["kind"] == "step"]
    assert len(steps) == len(record["history"]) == summary["stability"]["successful_steps"]
    assert steps[0]["task_loss_weights"] == pytest.approx(
        {"qol_years": 60 / 31, "population_years": 2 / 31}
    )
    for before, after in itertools.pairwise(steps):
        assert after["task_loss_weights"] == before["gradnorm"]["weights_next"]
        assert before["gradnorm"]["weights"] == before["task_loss_weights"]

    def imbalance(entry):
        g = entry["weighted_grad_norm"]
        return abs(math.log(g["qol_years"] / g["population_years"]))

    history = record["history"]
    assert sum(map(imbalance, history[-8:])) / 8 < sum(map(imbalance, history[:4])) / 4
    checks = summary["stability"]["checks"]
    assert checks["task_progress_stall"]["mandatory"]
    assert not checks["gradient_dominance"]["mandatory"]


def test_gradnorm_and_per_encoder_clip_resume_bit_exactly(tmp_path) -> None:
    options = {**GN, "grad_clip": "per-encoder"}
    reference, reference_summary, _ = _gn_fit(tmp_path, "r.jsonl", **options)
    path = str(tmp_path / "resume.pt")
    _gn_fit(tmp_path, "i.jsonl", max_steps=7, resume_path=path, resume_every=2, **options)
    payload = torch.load(path, weights_only=True)
    assert set(payload["gradnorm"]) == {"weights", "optimizer", "initial_losses"}
    resumed, resumed_summary, _ = _gn_fit(
        tmp_path, "s.jsonl", resume_path=path, resume_every=2, **options
    )
    assert _equal(reference, resumed)
    assert resumed_summary["task_loss_weights"] == reference_summary["task_loss_weights"]
    assert resumed_summary["grad_clip"]["mode"] == "per-encoder"
    # The fingerprint separates gradnorm settings and the clip mode.
    for other in (
        {**GN, "loss_balancing": GradNormBalancing(alpha=1.0), "grad_clip": "per-encoder"},
        GN,
        {"task_loss_weights": {"qol_years": 30.0}, "grad_clip": "per-encoder"},
    ):
        with pytest.raises(ValueError, match="different critic fit"):
            _gn_fit(tmp_path, "x.jsonl", resume_path=path, **other)


def test_per_encoder_clip_fit_records_group_norms(tmp_path) -> None:
    _, summary, _ = _gn_fit(tmp_path, "c.jsonl", grad_clip="per-encoder")
    steps = [s for s in _lines(tmp_path / "c.jsonl") if s["kind"] == "step"]
    groups = {"task", "cost:feasibility", "cost:settlement_guard"}
    assert set(steps[0]["grad_norm_group_preclip"]) == groups
    for s in steps:
        norms = s["grad_norm_group_preclip"]
        assert s["grad_norm_global"] == pytest.approx(math.sqrt(sum(v * v for v in norms.values())))
    assert summary["stability"]["grad_clip"] == "per-encoder"
    assert set(summary["grad_clip"]["groups"]) == groups


def test_gradnorm_unmonitored_fit_and_validation() -> None:
    _, summary, _ = _fit(dataset=_starved_dataset(), **GN)
    assert summary["task_loss_weights"]["mode"] == "gradnorm"
    assert summary["task_loss_weights"]["final"] != pytest.approx(
        {"qol_years": 60 / 31, "population_years": 2 / 31}
    )
    from test_labor_critic_balance import _models

    task, _ = _models()
    assert gradnorm_shared_parameters(task)[0] is task.encoder.trunk[3].weight
    with pytest.raises(ValueError, match="exclude each other"):
        from anthropocene_rl.autonomous_v7.labor_bc import AutoQoLLossWeight

        _fit(loss_balancing=GradNormBalancing(), qol_loss_weight_auto=AutoQoLLossWeight(),
             stability=StabilityConfig())
    with pytest.raises(ValueError, match="grad_clip"):
        _fit(grad_clip="per-head")


def test_cli_passes_only_non_default_balance_options() -> None:
    def parse(*extra):
        return lt.argument_parser().parse_args(
            ["pretrain-critic", "--checkpoint", "c.pt", "--dataset", "d", "--out", "o.pt", *extra]
        )

    assert lt._balance_kwargs(parse()) == {}
    kwargs = lt._balance_kwargs(
        parse("--loss-balancing", "gradnorm", "--gradnorm-alpha", "1.0", "--gradnorm-lr",
              "0.01", "--grad-clip", "per-encoder", "--qol-loss-weight", "30")
    )
    assert kwargs == {
        "loss_balancing": GradNormBalancing(alpha=1.0, lr=0.01),
        "grad_clip": "per-encoder",
    }
    with pytest.raises(ValueError, match="exclude each other"):
        lt._balance_kwargs(parse("--loss-balancing", "gradnorm", "--qol-loss-weight", "auto"))
    assert json.dumps(GradNormBalancing().state_dict())


def test_the_pilot_critic_args_parse_as_pretrain_critic_arguments() -> None:
    import shlex

    pilot = (
        "--qol-loss-weight 30 --qol-head-bias-init mean-target --td-step0-targets mc "
        "--tail-normalize world-factor --false-terminal-policy mask "
        "--loss-balancing gradnorm --grad-clip per-encoder"
    )
    args = lt.argument_parser().parse_args(
        ["pretrain-critic", "--checkpoint", "c.pt", "--dataset", "d", "--out", "o.pt",
         *shlex.split(pilot)]
    )
    assert lt._task_weight_kwargs(args) == {"task_loss_weights": {"qol_years": 30.0}}
    assert lt._balance_kwargs(args) == {
        "loss_balancing": GradNormBalancing(), "grad_clip": "per-encoder"
    }


def test_task_progress_stall_exempts_a_converged_head() -> None:
    # gradnorm pilot 29 Oct: QoL excess BCE fell 96% then sat flat; that is convergence.
    detector = _stall_detector()
    for step in range(12):
        qol = 0.03 * 0.5**step if step < 4 else 0.03 * 0.5**4
        detector.update(step, {"qol_years": 1.0, "population_years": 1.0},
                        {"qol_years": qol, "population_years": 3.0 * 0.9**step})
        assert detector.progress_stall(step) is None
    assert "qol_years" in detector.stall_converged
