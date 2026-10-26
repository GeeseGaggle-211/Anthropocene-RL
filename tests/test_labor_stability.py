"""Enforced r5 stability instrumentation (labor_stability) and its fit wiring.

Required fixtures (r5 acceptance harness, section D): an injected NaN triggers
parameter+Adam rollback and LR halving; K consecutive failures abort; the
dominance detector, the saturation alarm and the TD divergence alarm each fire
on a constructed case and stay quiet on its control; the actor receives
exactly zero gradient during critic and advantage fits. A monitored fit without
failures is bitwise identical to an unmonitored one.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
from test_labor_model import _labor_tensors
from test_labor_privileged_bridge import _raw
from test_labor_privileged_model import _critics, _layout
from test_labor_qol_head import _training_dataset
from torch import nn

from anthropocene_rl.autonomous_v7 import labor_advantage as la
from anthropocene_rl.autonomous_v7.labor_bc import pretrain_critic_on_dataset
from anthropocene_rl.autonomous_v7.labor_model import (
    LaborActor,
    LaborCostCritics,
    LaborQoLHead,
    LaborTaskCritic,
    critic_privileged_batch,
)
from anthropocene_rl.autonomous_v7.labor_stability import (
    DominanceDetector,
    StabilityAbort,
    StabilityConfig,
    StabilityMonitor,
    StabilityThresholds,
    TDDriftMonitor,
    dead_feature_counts,
    overfit_alarm,
    privileged_attention_mass,
    saturation_alarm,
)


def _toy(seed: int = 0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    return model, optimizer


def _batches(count: int = 8):
    generator = torch.Generator().manual_seed(1)
    return [
        (torch.randn(16, 4, generator=generator), torch.randn(16, 1, generator=generator))
        for _ in range(count)
    ]


def _monitor(model, optimizer, config=None, **kwargs):
    return StabilityMonitor(
        config or StabilityConfig(log_every=1),
        fit="toy",
        optimizer=optimizer,
        parameters=list(model.parameters()),
        blocks={"first": list(model[0].parameters()), "second": list(model[2].parameters())},
        **kwargs,
    )


def _plain_steps(model, optimizer, batches):
    for x, y in batches:
        optimizer.zero_grad()
        loss = (model(x) - y).square().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
        optimizer.step()


def _state(model, optimizer):
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def _equal_states(a, b) -> bool:
    (pa, oa), (pb, ob) = a, b
    if any(not torch.equal(pa[k], pb[k]) for k in pa):
        return False
    for key, state in oa["state"].items():
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and not torch.equal(value, ob["state"][key][name]):
                return False
    return True


# --- 1. finite checks, rollback, LR halving, abort ------------------------------------------


@pytest.mark.parametrize("hook", ["inject_nonfinite_param_steps", "inject_nonfinite_grad_steps"])
def test_injected_nan_rolls_back_parameters_and_adam_and_halves_the_lr(hook: str) -> None:
    batches = _batches(4)
    reference_model, reference_opt = _toy()
    _plain_steps(reference_model, reference_opt, batches[:2])
    after_two = _state(reference_model, reference_opt)

    model, optimizer = _toy()
    monitor = _monitor(model, optimizer, StabilityConfig(log_every=1, **{hook: (3,)}))
    results = []
    for x, y in batches[:3]:
        optimizer.zero_grad()
        results.append(monitor.step((model(x) - y).square().mean(), max_grad_norm=1.0))
    assert results == [True, True, False]
    # Rolled back to the last good state: parameters and Adam moments of step 2.
    assert _equal_states(_state(model, optimizer), after_two)
    assert optimizer.state_dict()["state"][0]["step"] == after_two[1]["state"][0]["step"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-3)
    event = monitor.events[-1]
    assert event["kind"] == "rollback" and event["step"] == 3 and event["injected"]
    assert event["reason"] in ("nonfinite-parameter", "nonfinite-gradient")
    assert event["lr_after"] == [pytest.approx(5e-3)]
    # Training continues at the halved rate; the receipt records the event and refuses
    # admission because a fault was injected.
    x, y = batches[3]
    optimizer.zero_grad()
    assert monitor.step((model(x) - y).square().mean(), max_grad_norm=1.0)
    summary = monitor.finish()
    assert summary["rollbacks"] == 1 and summary["checks"]["finite_updates"]["status"] == "PASS"
    assert summary["fault_injection"] and not summary["admitted"]


def test_k_consecutive_failures_abort_with_the_receipt() -> None:
    model, optimizer = _toy()
    monitor = _monitor(
        model,
        optimizer,
        StabilityConfig(log_every=1, thresholds=StabilityThresholds(max_consecutive_failures=3)),
    )
    x, y = _batches(1)[0]
    with pytest.raises(StabilityAbort, match="3 consecutive") as caught:
        for _ in range(5):
            optimizer.zero_grad()
            monitor.step((model(x) - y).square().mean() * float("nan"), max_grad_norm=1.0)
    summary = caught.value.summary
    assert summary["rollbacks"] == 3
    assert summary["checks"]["finite_updates"]["status"] == "FAIL"
    assert "finite_updates" in summary["mandatory_failed"] and not summary["admitted"]
    assert summary["lr_final"] == [pytest.approx(1e-2 * 0.5**3)]


def test_too_many_total_rollbacks_fail_the_fit() -> None:
    model, optimizer = _toy()
    config = StabilityConfig(
        log_every=1,
        thresholds=StabilityThresholds(max_total_rollbacks=1),
    )
    monitor = _monitor(model, optimizer, config)
    for index, (x, y) in enumerate(_batches(4)):
        optimizer.zero_grad()
        scale = float("nan") if index in (0, 2) else 1.0
        monitor.step((model(x) - y).square().mean() * scale, max_grad_norm=1.0)
    assert monitor.summary()["checks"]["finite_updates"]["status"] == "FAIL"


def test_monitored_steps_are_bitwise_identical_to_the_plain_loop(tmp_path) -> None:
    batches = _batches(6)
    plain_model, plain_opt = _toy()
    _plain_steps(plain_model, plain_opt, batches)
    model, optimizer = _toy()
    monitor = _monitor(
        model, optimizer, StabilityConfig(log_every=2, jsonl=str(tmp_path / "s.jsonl"))
    )
    for x, y in batches:
        optimizer.zero_grad()
        assert monitor.step((model(x) - y).square().mean(), max_grad_norm=1.0)
    assert _equal_states(_state(model, optimizer), _state(plain_model, plain_opt))
    monitor.finish()
    lines = [json.loads(line) for line in (tmp_path / "s.jsonl").read_text().splitlines()]
    kinds = [line["kind"] for line in lines]
    assert kinds[0] == "start" and kinds[-1] == "summary" and kinds.count("step") == 3
    step = next(line for line in lines if line["kind"] == "step")
    for key in ("grad_norm_preclip", "grad_norm_global", "clip_multiplier", "param_step_norm"):
        assert key in step
    assert set(step["grad_norm_preclip"]) == {"first", "second"}
    assert 0.0 < step["clip_multiplier"] <= 1.0


# --- 2. gradient dominance -------------------------------------------------------------------


def _detector(**kwargs):
    defaults = dict(window=8, share=0.9, persistence=0.75, min_progress=0.01)
    defaults.update(kwargs)
    return DominanceDetector("task_encoder", ("qol_years", "population_years"), **defaults)


def test_dominance_detector_fires_on_persistent_dominance_without_progress() -> None:
    detector = _detector()
    alarms = [
        detector.update(
            step,
            {"qol_years": 19.0, "population_years": 1.0},
            {"qol_years": 1.0 / (step + 1), "population_years": 2.0},
        )
        for step in range(8)
    ]
    assert alarms[:7] == [None] * 7
    assert alarms[7]["dominant"] == "qol_years"
    assert "population_years" in alarms[7]["stalled_progress"]


def test_dominance_with_progress_or_without_persistence_is_not_an_alarm() -> None:
    progressing = _detector()
    for step in range(8):
        progressing.update(
            step,
            {"qol_years": 19.0, "population_years": 1.0},
            {"qol_years": 1.0, "population_years": 2.0 * 0.9**step},
        )
    assert progressing.dominant_windows == 1 and not progressing.alarms
    balanced = _detector()
    for step in range(8):
        balanced.update(
            step,
            {"qol_years": 1.0, "population_years": 1.0},
            {"qol_years": 1.0, "population_years": 2.0},
        )
    assert balanced.dominant_windows == 0 and not balanced.alarms
    intermittent = _detector()
    for step in range(8):
        norms = {"qol_years": 19.0 if step % 2 else 1.0, "population_years": 1.0}
        intermittent.update(step, norms, {"qol_years": 1.0, "population_years": 2.0})
    assert not intermittent.alarms


def test_monitor_wires_dominance_on_a_shared_block_into_a_mandatory_failure() -> None:
    torch.manual_seed(0)
    shared = nn.Linear(4, 4)
    head_a, head_b = nn.Linear(4, 1), nn.Linear(4, 1)
    params = [*shared.parameters(), *head_a.parameters(), *head_b.parameters()]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    monitor = StabilityMonitor(
        StabilityConfig(log_every=1),
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
    for _ in range(20):  # exceeds the 16-point dominance window (user decision, 26 Oct)
        optimizer.zero_grad()
        hidden = shared(x)
        # Large-but-finite residual on head a; head b reads the shared block detached, so
        # its gradient on the shared block vanishes and it makes no progress there.
        loss_a = (head_a(hidden) - 1e3).square().mean()
        loss_b = (head_b(hidden.detach()) - 5.0).square().mean().detach() + 0.0 * head_b.bias.sum()
        monitor.step(loss_a + loss_b, max_grad_norm=1.0, head_losses={"a": loss_a, "b": loss_b})
    summary = monitor.finish()
    check = summary["checks"]["gradient_dominance"]
    assert check["status"] == "FAIL" and check["mandatory"]
    assert summary["alarms"][0]["dominant"] == "a"
    assert "gradient_dominance" in summary["mandatory_failed"]


# --- 3. bounded QoL saturation -------------------------------------------------------------


def test_saturation_alarm_threshold() -> None:
    assert saturation_alarm(0.2, 0.05) and not saturation_alarm(0.01, 0.05)
    assert not saturation_alarm(None, 0.05)


def test_saturated_bounded_head_fails_the_mandatory_check() -> None:
    torch.manual_seed(0)
    task = LaborTaskCritic(_layout(False), width=16, token_width=16, qol_head=LaborQoLHead())
    costs = LaborCostCritics(_layout(False), width=16, token_width=16)
    with torch.no_grad():
        task.heads["qol_years"].bias.fill_(50.0)
    summary = pretrain_critic_on_dataset(
        task_critic=task,
        cost_critics=costs,
        dataset=_training_dataset(),
        epochs=1,
        batch_size=4,
        lr=1e-9,
    )
    check = summary["stability"]["checks"]["qol_saturation"]
    assert check["status"] == "FAIL" and check["final_share"] == 1.0
    assert not summary["stability"]["admitted"]


# --- 4. TD(lambda) drift --------------------------------------------------------------------


def test_td_divergence_fires_when_drift_grows_over_three_refreshes() -> None:
    weight = 100.0
    monitor = TDDriftMonitor(weight, growth=3, floor=1e-3, tolerance=1e-6)
    base = torch.full((10,), 40.0)
    assert monitor.update(0, base - 30.0, base, growth_eligible=False) is None  # vs Monte Carlo
    current = base
    alarm = None
    for epoch, step in enumerate((1.0, 2.0, 4.0), start=1):
        previous, current = current, current + step
        alarm = monitor.update(epoch, previous, current)
    assert alarm is not None and alarm["reason"] == "drift-grew-over-3-refreshes"
    assert alarm["drifts"] == pytest.approx([0.01, 0.02, 0.04])


def test_td_shrinking_drift_is_quiet_and_out_of_range_targets_alarm() -> None:
    monitor = TDDriftMonitor(100.0, growth=3, floor=1e-3, tolerance=1e-6)
    current = torch.full((10,), 40.0)
    for epoch, step in enumerate((4.0, 2.0, 1.0, 0.5), start=1):
        previous, current = current, current + step
        assert monitor.update(epoch, previous, current) is None
    tiny = TDDriftMonitor(100.0, growth=3, floor=1e-3, tolerance=1e-6)
    current = torch.full((10,), 40.0)
    for epoch, step in enumerate((1e-4, 2e-4, 4e-4), start=1):  # growing but below the floor
        previous, current = current, current + step
        assert tiny.update(epoch, previous, current) is None
    bounds = TDDriftMonitor(100.0, growth=3, floor=1e-3, tolerance=1e-6)
    alarm = bounds.update(0, None, torch.tensor([10.0, 100.5]))
    assert alarm["reason"] == "targets-outside-[0,W]" and alarm["out_of_bounds"] == 1
    assert bounds.update(1, None, torch.tensor([0.0, 100.0])) is None


def test_td_lambda_fit_records_refreshes() -> None:
    torch.manual_seed(0)
    task = LaborTaskCritic(
        _layout(False), width=16, token_width=16, qol_head=LaborQoLHead(target="td-lambda")
    )
    costs = LaborCostCritics(_layout(False), width=16, token_width=16)
    summary = pretrain_critic_on_dataset(
        task_critic=task, cost_critics=costs, dataset=_training_dataset(), epochs=3, batch_size=4
    )
    td = summary["stability"]["checks"]["td_divergence"]
    assert td["status"] in ("PASS", "FAIL") and len(td["refreshes"]) == 3
    assert td["refreshes"][0]["growth_eligible"] is False
    assert all(r["out_of_bounds"] == 0 for r in td["refreshes"])


# --- 6. advantage overfit ----------------------------------------------------------------------


def test_overfit_alarm_on_gap_or_falling_heldout() -> None:
    assert overfit_alarm([0.7, 0.8, 0.95], [0.7, 0.72, 0.74], gap=0.10, drop=0.03)
    assert overfit_alarm([0.7, 0.75, 0.76], [0.72, 0.76, 0.70], gap=0.10, drop=0.03)
    assert overfit_alarm([0.7, 0.75, 0.78], [0.70, 0.73, 0.74], gap=0.10, drop=0.03) is None
    assert overfit_alarm([None], [None], gap=0.1, drop=0.03) is None


# --- 7. privileged attention and dead features ------------------------------------------------


def test_privileged_attention_mass_and_dead_features() -> None:
    task, _ = _critics(_layout())
    tensors = _labor_tensors(batch=3)
    privileged = critic_privileged_batch(task, _raw(3))
    report = privileged_attention_mass(task.encoder, tensors, privileged)
    rows = sum(f.rows for f in task.privileged_layout.families)  # caps plus overflow rows
    assert report["privileged_rows"] == rows and len(report["layers"]) == 2
    for layer in report["layers"]:
        assert 0.0 < layer["village_mass"] < 1.0 and 0.0 < layer["uniform_share"] < 1.0
    assert report["max_village_ratio"] > 0.0
    # The probe leaves the encoder's outputs and mode untouched.
    with torch.no_grad():
        before = task(tensors, privileged)
        privileged_attention_mass(task.encoder, tensors, privileged)
        after = task(tensors, privileged)
    assert all(torch.equal(before[k], after[k]) for k in before)
    dead = dead_feature_counts(task.encoder, tensors, privileged, std=1e-6)
    assert dead["trunk_units"] == 16 and set(dead["privileged"]) == {
        f.name for f in task.privileged_layout.families
    }
    plain, _ = _critics(_layout(False))
    assert privileged_attention_mass(plain.encoder, tensors, None) is None


# --- actor gradient exactly zero ---------------------------------------------------------------


def test_actor_gradient_is_exactly_zero_during_the_critic_fit() -> None:
    torch.manual_seed(0)
    actor = LaborActor(_layout(False), width=16, token_width=16)
    task = LaborTaskCritic(_layout(False), width=16, token_width=16, qol_head=LaborQoLHead())
    costs = LaborCostCritics(_layout(False), width=16, token_width=16)
    before = copy.deepcopy(actor.state_dict())
    summary = pretrain_critic_on_dataset(
        task_critic=task,
        cost_critics=costs,
        dataset=_training_dataset(),
        epochs=2,
        batch_size=4,
        frozen_actor=actor,
        stability=StabilityConfig(log_every=1),
    )
    check = summary["stability"]["checks"]["frozen_gradients_zero"]
    assert (
        check["status"] == "PASS" and check["modules"] == ["actor"] and check["digests_unchanged"]
    )
    assert all(p.grad is None or not bool((p.grad != 0).any()) for p in actor.parameters())
    assert all(torch.equal(before[k], v) for k, v in actor.state_dict().items())
    step = summary["stability"]["checks"]
    assert step["finite_updates"]["status"] == "PASS"


def test_actor_gradient_is_exactly_zero_during_the_advantage_fit() -> None:
    task, costs = _critics(_layout(False))
    torch.manual_seed(0)
    actor = LaborActor(_layout(False), width=16, token_width=16)
    shard = la.synthetic_candidate_shard(
        worlds=tuple(range(90100, 90106)), roots_per_world=3, candidates=5, seed=1
    )
    config = la.AdvantageFitConfig(epochs=3, batch_roots=4, lr=2e-3, seed=0, hidden=32)
    _, summary = la.fit_advantage_head(
        task,
        costs,
        la.parse_candidate_shard(shard),
        config,
        frozen_actor=actor,
        stability=StabilityConfig(log_every=1),
    )
    checks = summary["stability"]["checks"]
    assert checks["frozen_gradients_zero"]["status"] == "PASS"
    assert checks["frozen_gradients_zero"]["modules"] == ["actor", "cost_critics", "task_critic"]
    assert all(p.grad is None for p in actor.parameters())
    epochs = checks["advantage_overfit"]["epochs"]
    assert len(epochs) == 3 and all("heldout" in e and "train" in e for e in epochs)


def test_a_frozen_module_that_receives_gradient_fails_the_check() -> None:
    model, optimizer = _toy()
    leak = nn.Linear(4, 1)
    monitor = _monitor(model, optimizer, frozen={"actor": leak})
    x, y = _batches(1)[0]
    optimizer.zero_grad()
    monitor.step((model(x) - y).square().mean() + leak(x).mean(), max_grad_norm=1.0)
    summary = monitor.finish()
    assert summary["checks"]["frozen_gradients_zero"]["status"] == "FAIL"
    assert "frozen_gradients_zero" in summary["mandatory_failed"]


def test_monitored_critic_fit_is_bitwise_identical_to_the_unmonitored_fit() -> None:
    def fit(stability):
        torch.manual_seed(0)
        task = LaborTaskCritic(_layout(False), width=16, token_width=16, qol_head=LaborQoLHead())
        costs = LaborCostCritics(_layout(False), width=16, token_width=16)
        pretrain_critic_on_dataset(
            task_critic=task,
            cost_critics=costs,
            dataset=_training_dataset(),
            epochs=2,
            batch_size=4,
            stability=stability,
        )
        return {**task.state_dict(), **{f"c.{k}": v for k, v in costs.state_dict().items()}}

    plain, monitored = fit(None), fit(StabilityConfig(log_every=1))
    assert all(torch.equal(plain[k], monitored[k]) for k in plain)


def test_exact_resume_matches_the_uninterrupted_fit(tmp_path) -> None:
    import numpy as np

    def models():
        torch.manual_seed(0)
        return (
            LaborTaskCritic(
                _layout(False), width=16, token_width=16, qol_head=LaborQoLHead(target="td-lambda")
            ),
            LaborCostCritics(_layout(False), width=16, token_width=16),
        )

    data = _training_dataset()
    data["collection"] = dict(data["collection"], dataset_digest="fixture")
    task, costs = models()
    pretrain_critic_on_dataset(
        task_critic=task,
        cost_critics=costs,
        dataset=data,
        epochs=3,
        batch_size=2,
        rng=np.random.default_rng(7),
    )
    reference = task.state_dict()
    path = str(tmp_path / "resume.pt")
    task_a, costs_a = models()
    rng = np.random.default_rng(7)
    pretrain_critic_on_dataset(
        task_critic=task_a,
        cost_critics=costs_a,
        dataset=data,
        epochs=3,
        batch_size=2,
        rng=rng,
        max_steps=4,
        resume_path=path,
        resume_every=2,
    )
    task_b, costs_b = models()  # a fresh process: models and RNG restart
    summary = pretrain_critic_on_dataset(
        task_critic=task_b,
        cost_critics=costs_b,
        dataset=data,
        epochs=3,
        batch_size=2,
        rng=np.random.default_rng(7),
        resume_path=path,
        resume_every=2,
    )
    assert all(torch.equal(reference[k], v) for k, v in task_b.state_dict().items())
    assert summary["stability"]["steps"] == 9
    other = dict(data, collection=dict(data["collection"], dataset_digest="other"))
    with pytest.raises(ValueError, match="different critic fit"):
        pretrain_critic_on_dataset(
            task_critic=task_b,
            cost_critics=costs_b,
            dataset=other,
            epochs=3,
            batch_size=2,
            resume_path=path,
        )


def test_td_lambda_targets_keep_continuations_as_their_own_trajectories() -> None:
    # Regression: TD(lambda) grouped by (world, camp) only, chaining two learner-state
    # continuations of the same camp; it must use discounted_returns' trajectory keys.
    from test_labor_dagger_continuation import _rows

    from anthropocene_rl.autonomous_v7.labor_bc import CONTINUATION_ROW_SOURCE, discounted_returns

    data = _rows([4, 4], trajectory=[0, 3], source=CONTINUATION_ROW_SOURCE)
    data["tail_value__qol_years"] = torch.tensor([10.0, 20.0], dtype=torch.float64)
    mc = discounted_returns(data)["qol_years"]
    values = torch.tensor([100.0, 200.0])
    assert torch.allclose(
        la.qol_td_lambda_targets(data, values, 1.0).double(), mc.double(), atol=1e-5
    )
    assert torch.allclose(
        la.qol_td_lambda_targets(data, values, 0.0).double(), mc.double(), atol=1e-5
    )
