"""r5 critic balance: per-head task loss weights, the auto QoL weight, excess BCE, best-val.

Every option defaults to the historical fit (bitwise; the defaults were also compared
against the pre-change source 66bc618 on the fixtures below).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from test_labor_early_stop import _validation_dataset
from test_labor_privileged_model import _labor_tensors, _layout
from test_labor_qol_head import _training_dataset

from anthropocene_rl.autonomous_v7 import labor_training as lt
from anthropocene_rl.autonomous_v7.labor_bc import (
    AutoQoLLossWeight,
    pretrain_critic_on_dataset,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    LaborCostCritics,
    LaborQoLHead,
    LaborTaskCritic,
)
from anthropocene_rl.autonomous_v7.labor_ppo import labor_task_value_losses
from anthropocene_rl.autonomous_v7.labor_stability import (
    StabilityConfig,
    stability_config_from_args,
)


def _models(target="monte-carlo"):
    torch.manual_seed(0)
    task = LaborTaskCritic(
        _layout(False), width=16, token_width=16, qol_head=LaborQoLHead(target=target)
    )
    costs = LaborCostCritics(_layout(False), width=16, token_width=16)
    return task, costs


def _starved_dataset():
    """Population rewards spread over two orders of magnitude (seeded features)."""
    torch.manual_seed(11)
    data = _training_dataset()
    data["reward_population_years"] = torch.tensor(
        [1.0, 40.0, 3.0, 25.0, 0.5, 60.0], dtype=torch.float64
    )
    data["collection"] = dict(data["collection"], dataset_digest="balance-fixture")
    return data


def _fit(*, target="monte-carlo", dataset=None, epochs=3, stability=None, **kwargs):
    task, costs = _models(target)
    generator = np.random.default_rng(7)
    summary = pretrain_critic_on_dataset(
        task_critic=task,
        cost_critics=costs,
        dataset=_training_dataset() if dataset is None else dataset,
        epochs=epochs,
        batch_size=2,
        rng=generator,
        stability=stability,
        **kwargs,
    )
    state = {**task.state_dict(), **{f"c.{k}": v for k, v in costs.state_dict().items()}}
    return state, summary, generator.bit_generator.state


def _equal(a, b) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- 1. defaults are the historical fit ------------------------------------------------------


@pytest.mark.parametrize("target", ["monte-carlo", "td-lambda"])
@pytest.mark.parametrize("monitored", [False, True])
def test_unit_weights_are_the_unweighted_fit_bitwise(target, monitored) -> None:
    config = StabilityConfig(log_every=1) if monitored else None
    plain, plain_summary, plain_rng = _fit(target=target, stability=config)
    ones, ones_summary, ones_rng = _fit(
        target=target,
        stability=config,
        task_loss_weights={"qol_years": 1.0, "population_years": 1.0},
    )
    assert _equal(plain, ones) and plain_rng == ones_rng
    assert plain_summary["final_loss"] == ones_summary["final_loss"]
    assert "task_loss_weights" not in plain_summary and "task_loss_weights" not in ones_summary


def test_cli_defaults_pass_no_balance_options() -> None:
    args = lt.argument_parser().parse_args(
        ["pretrain-critic", "--checkpoint", "c.pt", "--dataset", "d", "--out", "o.pt"]
    )
    assert lt._task_weight_kwargs(args) == {}
    assert stability_config_from_args(args) == "auto"
    assert args.select_best_validation is False
    ones = lt.argument_parser().parse_args(
        ["pretrain-critic", "--checkpoint", "c.pt", "--dataset", "d", "--out", "o.pt",
         "--task-loss-weights", "qol_years=1,population_years=1"]
    )
    assert lt._task_weight_kwargs(ones) == {}


def test_cli_parses_weights_and_auto() -> None:
    def kwargs(*extra):
        return lt._task_weight_kwargs(
            lt.argument_parser().parse_args(
                ["pretrain-critic", "--checkpoint", "c", "--dataset", "d", "--out", "o", *extra]
            )
        )

    assert kwargs("--task-loss-weights", "qol_years=30") == {
        "task_loss_weights": {"qol_years": 30.0}
    }
    assert kwargs("--qol-loss-weight", "30") == {"task_loss_weights": {"qol_years": 30.0}}
    auto = kwargs("--qol-loss-weight", "auto", "--qol-weight-target-share", "0.4")
    assert auto["qol_loss_weight_auto"] == AutoQoLLossWeight(target_share=0.4, ema=0.3)
    with pytest.raises(ValueError, match="either|--task-loss-weights or"):
        kwargs("--task-loss-weights", "qol_years=2", "--qol-loss-weight", "auto")
    with pytest.raises(ValueError, match="head=weight"):
        kwargs("--task-loss-weights", "feasibility=2")
    with pytest.raises(ValueError, match="positive"):
        kwargs("--task-loss-weights", "qol_years=0")
    config = stability_config_from_args(
        lt.argument_parser().parse_args(
            ["pretrain-critic", "--checkpoint", "c", "--dataset", "d", "--out", "o",
             "--dominance-excess-loss"]
        )
    )
    assert config.dominance_excess_loss and config.state_dict()["dominance_excess_loss"]
    assert "dominance_excess_loss" not in StabilityConfig().state_dict()


# --- 2. weights scale the right head ---------------------------------------------------------


def test_weights_scale_the_named_heads_loss_and_gradient() -> None:
    task, _ = _models()
    tensors = _labor_tensors(batch=6)
    targets = {
        "qol_years": torch.linspace(5.0, 300.0, 6),
        "population_years": torch.linspace(1.0, 50.0, 6),
    }
    base, heads, _ = labor_task_value_losses(task, tensors, targets)
    weighted, w_heads, _ = labor_task_value_losses(
        task, tensors, targets, weights={"qol_years": 30.0, "population_years": 2.0}
    )
    expected = (30.0 * heads["qol_years"].mean() + 2.0 * heads["population_years"].mean()) / 2
    assert float(weighted.detach()) == pytest.approx(float(expected.detach()), rel=1e-6)
    assert all(torch.equal(heads[k], w_heads[k]) for k in heads)  # per-head rows unweighted

    def head_grads(weights):
        task.zero_grad()
        loss = labor_task_value_losses(task, tensors, targets, weights=weights)[0]
        loss.backward()
        return {
            name: torch.cat([p.grad.reshape(-1) for p in task.heads[name].parameters()])
            for name in ("qol_years", "population_years")
        }

    unit = head_grads({"qol_years": 1.0, "population_years": 1.0})
    scaled = head_grads({"qol_years": 7.0, "population_years": 1.0})
    assert torch.allclose(scaled["qol_years"], 7.0 * unit["qol_years"], rtol=1e-5, atol=1e-9)
    assert torch.equal(scaled["population_years"], unit["population_years"])
    with pytest.raises(ValueError, match="exactly"):
        labor_task_value_losses(task, tensors, targets, weights={"qol_years": 2.0})


def test_a_fixed_weight_changes_the_fit_and_is_logged(tmp_path) -> None:
    log = tmp_path / "w.jsonl"
    plain, _, plain_rng = _fit(stability=StabilityConfig(log_every=1))
    heavy, summary, heavy_rng = _fit(
        stability=StabilityConfig(log_every=1, jsonl=str(log)),
        task_loss_weights={"qol_years": 30.0},
    )
    assert not _equal(plain, heavy)
    assert plain_rng == heavy_rng  # sampling untouched
    assert summary["task_loss_weights"]["mode"] == "fixed"
    assert summary["task_loss_weights"]["final"] == {"qol_years": 30.0, "population_years": 1.0}
    steps = [line for line in _lines(log) if line["kind"] == "step"]
    assert steps and all(
        s["task_loss_weights"] == {"qol_years": 30.0, "population_years": 1.0} for s in steps
    )
    for record in steps:  # effective norm = weight x unweighted norm
        raw = record["shared_head_grad_norm_unweighted"]["task_encoder"]
        effective = record["shared_head_grad_norm"]["task_encoder"]
        assert effective["qol_years"] == pytest.approx(30.0 * raw["qol_years"])
        assert effective["population_years"] == raw["population_years"]


# --- 3. auto QoL weight ----------------------------------------------------------------------


def test_auto_controller_arithmetic() -> None:
    auto = AutoQoLLossWeight(target_share=0.3, ema=1.0)
    # q = 1, p = 99 at w = 1: share 1%; w* = 0.3 / 0.7 * 99 = 42.43, applied at ema 1.
    update = auto.update(1.0, 1.0, 99.0, 1.0)
    assert update["qol_share_effective"] == pytest.approx(0.01)
    assert update["qol_weight_next"] == pytest.approx(0.3 / 0.7 * 99.0)
    assert auto.update(1.0, 0.0, 1.0, 1.0)["qol_weight_next"] == pytest.approx(200.0)  # q = 0
    assert auto.update(1.0, 10.0, 1.0, 1.0)["qol_weight_next"] == 1.0  # already above: min
    assert auto.update(1.0, 1e-9, 1.0, 1.0)["qol_weight_target"] == 200.0  # clamp
    assert auto.update(150.0, 1e-9, 1.0, 1.0)["qol_weight_next"] <= 200.0
    half = AutoQoLLossWeight(target_share=0.3, ema=0.5).update(1.0, 1.0, 99.0, 1.0)
    assert half["qol_weight_next"] == pytest.approx(math.sqrt(0.3 / 0.7 * 99.0))  # log EMA
    with pytest.raises(ValueError):
        AutoQoLLossWeight(target_share=1.0)


def _auto_fit(tmp_path, name, **kwargs):
    return _fit(
        dataset=_starved_dataset(),
        epochs=4,
        stability=StabilityConfig(log_every=1, jsonl=str(tmp_path / name)),
        **kwargs,
    )


def test_auto_weight_is_deterministic_and_leaves_the_rng_alone(tmp_path) -> None:
    auto = AutoQoLLossWeight(ema=0.5)
    first, first_summary, first_rng = _auto_fit(tmp_path, "a.jsonl", qol_loss_weight_auto=auto)
    second, second_summary, second_rng = _auto_fit(
        tmp_path, "b.jsonl", qol_loss_weight_auto=auto
    )
    assert _equal(first, second) and first_rng == second_rng
    assert first_summary["task_loss_weights"] == second_summary["task_loss_weights"]
    _, _, fixed_rng = _auto_fit(tmp_path, "c.jsonl")
    assert first_rng == fixed_rng  # the trainer generator is not consumed
    assert first_summary["task_loss_weights"]["mode"] == "auto-qol"
    history = first_summary["task_loss_weights"]["history"]
    assert len(history) == first_summary["stability"]["successful_steps"]  # log_every 1
    steps = [s for s in _lines(tmp_path / "a.jsonl") if s["kind"] == "step"]
    assert [s["qol_loss_weight_auto"] for s in steps] == history
    # The weight a step used is the previous step's update (it applies from the next step).
    for before, after in zip(steps, steps[1:], strict=False):
        assert after["task_loss_weights"]["qol_years"] == before["qol_loss_weight_auto"][
            "qol_weight_next"
        ]


def test_auto_weight_raises_the_qol_share(tmp_path) -> None:
    def shares(name):
        steps = [s for s in _lines(tmp_path / name) if s["kind"] == "step"]
        values = []
        for s in steps:
            norms = s["shared_head_grad_norm"]["task_encoder"]
            values.append(norms["qol_years"] / (norms["qol_years"] + norms["population_years"]))
        return values

    # The fixture's weight-1 QoL share is about 0.45, so the test targets 0.8 (the pilot
    # targets 0.3 from about 0.01-0.03); the controller must move the share toward it.
    _auto_fit(tmp_path, "fixed.jsonl")
    _, summary, _ = _auto_fit(
        tmp_path,
        "auto.jsonl",
        qol_loss_weight_auto=AutoQoLLossWeight(target_share=0.8, ema=0.5),
    )
    fixed, auto = shares("fixed.jsonl"), shares("auto.jsonl")
    assert np.mean(fixed[-6:]) < 0.6
    assert summary["task_loss_weights"]["final"]["qol_years"] > 2.0
    assert np.mean(auto[-6:]) > np.mean(fixed[-6:]) + 0.15
    assert np.mean(auto[-6:]) > 0.65


def test_auto_weight_survives_an_exact_resume(tmp_path) -> None:
    auto = AutoQoLLossWeight(ema=0.5)
    reference, reference_summary, _ = _auto_fit(tmp_path, "r.jsonl", qol_loss_weight_auto=auto)
    path = str(tmp_path / "resume.pt")
    _auto_fit(tmp_path, "i.jsonl", qol_loss_weight_auto=auto, max_steps=7, resume_path=path,
              resume_every=2)
    resumed, resumed_summary, _ = _auto_fit(
        tmp_path, "s.jsonl", qol_loss_weight_auto=auto, resume_path=path, resume_every=2
    )
    assert _equal(reference, resumed)
    assert resumed_summary["task_loss_weights"] == reference_summary["task_loss_weights"]
    # A resume file of a different weighting is refused.
    with pytest.raises(ValueError, match="different critic fit"):
        _auto_fit(tmp_path, "x.jsonl", resume_path=path)


def test_auto_weight_needs_the_monitor() -> None:
    with pytest.raises(ValueError, match="stability monitor"):
        _fit(qol_loss_weight_auto=AutoQoLLossWeight())


# --- 4. readable QoL loss --------------------------------------------------------------------


def test_excess_bce_is_the_kl_to_the_soft_target() -> None:
    task, _ = _models()
    weight = task.qol_head.discount_weight * task.qol_head.scale
    target = torch.tensor([0.0, 0.1, 0.33, 0.5, 0.9, 1.0], dtype=torch.float64) * weight
    logits = torch.tensor([[-2.0], [0.3], [1.0], [-0.5], [2.0], [4.0]])
    excess = task.qol_excess_bce(logits, target)
    p = target / weight
    q = torch.sigmoid(logits.squeeze(-1).double())
    kl = torch.special.xlogy(p, p / q) + torch.special.xlogy(1 - p, (1 - p) / (1 - q))
    assert torch.allclose(excess, kl, atol=1e-6)
    bce = task.value_loss("qol_years", logits, target).double()
    entropy = -(torch.special.xlogy(p, p) + torch.special.xlogy(1 - p, 1 - p))
    assert torch.allclose(excess, bce - entropy, atol=1e-6)
    inner = p[1:-1]
    exact = torch.logit(inner).float().reshape(-1, 1)  # q = p*: zero excess
    assert task.qol_excess_bce(exact, target[1:-1]).abs().max() < 1e-6
    plain = LaborTaskCritic(_layout(False), width=16, token_width=16)
    assert plain.qol_excess_bce(logits, target) is None


def test_excess_bce_is_logged_in_step_validation_and_final_records(tmp_path) -> None:
    log = tmp_path / "e.jsonl"
    _, summary, _ = _fit(
        stability=StabilityConfig(log_every=1, jsonl=str(log)),
        validation_dataset=_validation_dataset(),
    )
    assert summary["final_qol_excess_bce"] >= -1e-6
    assert summary["initial_qol_excess_bce"] >= -1e-6
    steps = [s for s in _lines(log) if s["kind"] == "step"]
    assert all("qol_excess_bce" in s for s in steps)
    for record in summary["validation"]["epochs"]:
        assert record["qol_excess_bce"] >= -1e-6
        assert "qol_excess_bce" not in record["heads"]  # never part of the total
        heads = record["heads"]
        assert record["task"] == pytest.approx(
            (heads["qol_years"] + heads["population_years"]) / 2
        )


def test_dominance_excess_progress_is_opt_in(tmp_path) -> None:
    plain_log, excess_log = tmp_path / "p.jsonl", tmp_path / "x.jsonl"
    plain, plain_summary, _ = _fit(stability=StabilityConfig(log_every=1, jsonl=str(plain_log)))
    excess, excess_summary, _ = _fit(
        stability=StabilityConfig(
            log_every=1, jsonl=str(excess_log), dominance_excess_loss=True
        )
    )
    assert _equal(plain, excess)  # measurement only
    assert all("dominance_progress_losses" not in s for s in _lines(plain_log))
    steps = [s for s in _lines(excess_log) if s["kind"] == "step"]
    for record in steps:
        assert record["dominance_progress_losses"]["qol_years"] == record["qol_excess_bce"]
        assert (
            record["dominance_progress_losses"]["population_years"]
            == record["head_losses"]["population_years"]
        )
    assert plain_summary["stability"]["config"].get("dominance_excess_loss") is None
    assert excess_summary["stability"]["config"]["dominance_excess_loss"] is True


# --- 5. best validation ----------------------------------------------------------------------


def test_best_validation_is_the_lowest_total_epoch_and_leaves_the_fit_alone() -> None:
    sink: dict = {}
    without, without_summary, without_rng = _fit(epochs=4)
    with_val, summary, rng = _fit(
        epochs=4, validation_dataset=_validation_dataset(), best_validation_sink=sink
    )
    assert _equal(without, with_val) and without_rng == rng
    assert without_summary["final_loss"] == summary["final_loss"]
    totals = [e["total"] for e in summary["validation"]["epochs"]]
    best = summary["validation"]["best"]
    assert best["epoch"] == 1 + int(np.argmin(totals)) and best["total"] == min(totals)
    assert sink["epoch"] == best["epoch"]
    # The stored weights are the fit's own weights at that epoch.
    at_best, _, _ = _fit(epochs=best["epoch"])
    stored = {**sink["task"], **{f"c.{k}": v for k, v in sink["costs"].items()}}
    assert _equal(stored, at_best)
    for head, entry in best["per_head"].items():
        losses = [e["heads"][head] for e in summary["validation"]["epochs"]]
        assert entry["epoch"] == 1 + int(np.argmin(losses))


def test_best_validation_survives_a_resume(tmp_path) -> None:
    data = _training_dataset()
    data["collection"] = dict(data["collection"], dataset_digest="fixture")
    sink_ref: dict = {}
    _fit(dataset=data, epochs=4, validation_dataset=_validation_dataset(),
         best_validation_sink=sink_ref)
    path = str(tmp_path / "resume.pt")
    _fit(dataset=data, epochs=4, validation_dataset=_validation_dataset(), max_steps=10,
         resume_path=path, resume_every=2)
    sink: dict = {}
    _, summary, _ = _fit(dataset=data, epochs=4, validation_dataset=_validation_dataset(),
                         resume_path=path, resume_every=2, best_validation_sink=sink)
    assert sink["epoch"] == sink_ref["epoch"]
    assert _equal(sink["task"], sink_ref["task"]) and _equal(sink["costs"], sink_ref["costs"])


@pytest.mark.parametrize("select", [False, True])
def test_checkpoint_files_with_best_validation(tmp_path, select) -> None:
    live = torch.nn.Parameter(torch.tensor([3.0]))  # final weights; state aliases storage
    out = tmp_path / "critic.pt"
    trainer_state = {
        "critic": {"w": live.data},
        "extra": "kept",
        "heuristic_critic_pretraining": {"final_loss": 1.0},
    }

    def rebuild():
        with torch.no_grad():
            live.copy_(torch.tensor([1.0]))  # load the best weights in place
        return {"critic": {"w": live.data}}

    lt._save_critic_checkpoints(
        out,
        trainer_state,
        {"epoch": 2, "step": 6, "total": 0.5},
        select=select,
        rebuild_with_best=rebuild,
    )
    best = torch.load(tmp_path / "critic.pt.best-val.pt", weights_only=True)
    main = torch.load(out, weights_only=True)
    assert float(best["critic"]["w"]) == 1.0 and best["extra"] == "kept"
    assert best["heuristic_critic_pretraining"]["checkpoint_weights"] == "best-validation-epoch-2"
    record = main["heuristic_critic_pretraining"]["best_validation"]
    assert record["epoch"] == 2 and record["validation_total"] == 0.5
    if select:
        final = torch.load(tmp_path / "critic.pt.final.pt", weights_only=True)
        assert float(main["critic"]["w"]) == 1.0 and float(final["critic"]["w"]) == 3.0
        assert record["selected"] == "best-validation"
    else:
        assert float(main["critic"]["w"]) == 3.0
        assert not (tmp_path / "critic.pt.final.pt").exists()
        assert record["selected"] == "final"
    plain = tmp_path / "plain.pt"
    lt._save_critic_checkpoints(
        plain, {"critic": {"w": torch.tensor([3.0])}}, None, select=False,
        rebuild_with_best=None,
    )
    assert plain.exists() and not (tmp_path / "plain.pt.best-val.pt").exists()
