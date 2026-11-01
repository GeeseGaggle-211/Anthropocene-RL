"""Opt-in actor BC GradNorm (pretrain-bc --loss-balancing gradnorm)."""

from __future__ import annotations

import json
import math

import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_training as lt
from anthropocene_rl.autonomous_v7.labor_actor_gradnorm import (
    ACTOR_BC_TASKS,
    ACTOR_ROLE_TASKS,
    ActorGradNorm,
    ActorGradNormState,
    role_resident_scales,
)
from anthropocene_rl.autonomous_v7.labor_bc import pretrain_labor_actor_bc
from anthropocene_rl.autonomous_v7.labor_stability import StabilityConfig
from tests.test_labor_actor_task_gradients import synthetic_actor_dataset

SCALES = {t: s for t, s in zip(ACTOR_ROLE_TASKS, [0.5, 0.4, 0.01, 0.03] + [0.005] * 9)}


def _fit(loss_balancing=None, stability=None, epochs=2):
    actor, data = synthetic_actor_dataset(rows=32, seed=5)
    summary = pretrain_labor_actor_bc(
        actor=actor,
        dataset=data,
        epochs=epochs,
        batch_size=8,
        lr=1e-3,
        rng=__import__("numpy").random.default_rng(0),
        **({} if loss_balancing is None else {"loss_balancing": loss_balancing}),
        **({} if stability is None else {"stability": stability}),
    )
    return actor, summary


def test_disabled_path_is_the_historical_fit() -> None:
    default, summary = _fit()
    explicit, _ = _fit(loss_balancing=None)
    assert "loss_balancing" not in summary
    for a, b in zip(default.state_dict().values(), explicit.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    args = lt.argument_parser().parse_args(
        ["pretrain-bc", "--checkpoint", "c.pt", "--dataset", "d", "--out", "o.pt"]
    )
    assert args.loss_balancing == "none" and lt._bc_loss_balancing_kwargs(args) == {}


def test_share_init_is_the_current_loss_and_inits_keep_the_scale() -> None:
    share = ActorGradNormState(ActorGradNorm(init="share"), SCALES)
    assert share.values() == {t: 1.0 for t in ACTOR_BC_TASKS}
    for init in ("uniform", "sqrt"):
        state = ActorGradNormState(ActorGradNorm(init=init, max_weight=1000.0), SCALES)
        assert state.scale_sum() == pytest.approx(state.target_scale, rel=1e-9)
        for task in ("specialty", "movement", "fission", "trade_gate"):
            assert state.values()[task] == 1.0
    uniform = ActorGradNormState(ActorGradNorm(init="uniform", max_weight=1000.0), SCALES)
    masses = [uniform.values()[t] * SCALES[t] for t in ACTOR_ROLE_TASKS]
    assert max(masses) == pytest.approx(min(masses))  # one equal expected weight per role
    clamped = ActorGradNormState(ActorGradNorm(init="uniform", max_weight=20.0), SCALES)
    assert max(clamped.values().values()) <= 20.0 + 1e-9


def test_update_renormalises_and_holds_understaffed_roles() -> None:
    config = ActorGradNorm(min_rows=4, lr=0.5, max_weight=1000.0)
    state = ActorGradNormState(config, SCALES)
    norms = {t: 1.0 + i for i, t in enumerate(ACTOR_BC_TASKS)}
    losses = {t: 2.0 for t in ACTOR_BC_TASKS}
    staffed = {t: 10 for t in ACTOR_ROLE_TASKS}
    staffed["activity:Potter"] = 3  # below min_rows: held
    before = state.values()["activity:Potter"]
    entry = state.update(norms, losses, staffed)
    assert "activity:Potter" not in entry["updated"]
    assert state.values()["activity:Potter"] == before
    assert state.ema_norm["activity:Potter"] is None
    assert state.task_holds["activity:Potter"] == 1
    assert state.scale_sum() == pytest.approx(state.target_scale, rel=1e-9)
    assert entry["scale_sum"] == pytest.approx(state.target_scale, rel=1e-9)
    weights = state.values()
    assert all(config.min_weight <= w <= config.max_weight for w in weights.values())
    snapshot = state.snapshot()
    state.update(norms, losses, staffed)
    state.restore(snapshot)
    assert state.values() == weights and state.updates == 1


def test_alpha_zero_equalises_the_weighted_ema_norms() -> None:
    config = ActorGradNorm(alpha=0.0, lr=0.5, ema=0.5, init="share", min_weight=1e-3,
                           max_weight=1e3)
    state = ActorGradNormState(config, SCALES)
    raw = {t: 0.01 * (1 + 3 * i) for i, t in enumerate(ACTOR_BC_TASKS)}
    staffed = {t: 100 for t in ACTOR_ROLE_TASKS}
    for _ in range(200):
        state.update(raw, {t: 1.0 for t in ACTOR_BC_TASKS}, staffed)
    weighted = [state.values()[t] * state.ema_norm[t] for t in ACTOR_BC_TASKS]
    assert max(weighted) / min(weighted) == pytest.approx(1.0, abs=1e-6)
    assert state.scale_sum() == pytest.approx(state.target_scale, rel=1e-9)


def test_alpha_favours_slower_tasks() -> None:
    config = ActorGradNorm(alpha=1.0, lr=1.0, ema=1.0, init="share", max_weight=1e3)
    state = ActorGradNormState(config, SCALES)
    staffed = {t: 100 for t in ACTOR_ROLE_TASKS}
    raw = {t: 1.0 for t in ACTOR_BC_TASKS}
    state.update(raw, {t: 1.0 for t in ACTOR_BC_TASKS}, staffed)  # sets L(0)
    losses = {t: (1.0 if t == "movement" else 0.5) for t in ACTOR_BC_TASKS}
    state.update(raw, losses, staffed)
    weights = state.values()
    assert weights["movement"] > weights["specialty"]


def test_role_scales_are_mean_resident_shares() -> None:
    target = torch.zeros(4, 14)
    target[:, 1] = torch.tensor([0.5, 0.7, 0.9, 0.3])
    target[:, 7] = torch.tensor([0.0, 0.2, 0.0, 0.0])
    scales = role_resident_scales(target, torch.ones(4), torch.tensor([True, True, True, False]))
    assert scales["activity:Gatherer"] == pytest.approx(0.7)
    assert scales["activity:Potter"] == pytest.approx(0.2 / 3)


def test_gradnorm_fit_logs_weights_and_judges_dominance(tmp_path) -> None:
    config = ActorGradNorm(min_rows=2)
    jsonl = tmp_path / "stability.jsonl"
    stability = StabilityConfig(jsonl=str(jsonl), log_every=2)
    _, summary = _fit(loss_balancing=config, stability=stability)
    record = summary["loss_balancing"]
    assert record["mode"] == "gradnorm" and record["tasks"] == list(ACTOR_BC_TASKS)
    assert record["updates"] == 8
    assert record["scale_sum"] == pytest.approx(record["scale_target"], rel=1e-6)
    assert record["task_holds"]["activity:ClayMiner"] == 8  # never staffed
    assert record["final_weights"]["activity:ClayMiner"] == record["initial_weights"][
        "activity:ClayMiner"
    ]
    steps = [json.loads(line) for line in jsonl.read_text().splitlines()]
    logged = [r for r in steps if r.get("kind") == "step" and "actor_gradnorm" in r]
    assert logged and set(logged[0]["actor_task_weights"]) == set(ACTOR_BC_TASKS)
    assert set(logged[0]["shared_head_grad_norm"]["actor_tasks"]) == set(ACTOR_BC_TASKS)
    checks = summary["stability"]["checks"]
    assert checks["gradient_dominance"]["status"] in ("PASS", "FAIL")


def test_checkpoint_round_trip_records_the_config(tmp_path) -> None:
    config = ActorGradNorm(
        alpha=0.5, init="sqrt", min_rows=3, max_weight=50.0, shared="last-layer", every=2
    )
    _, summary = _fit(loss_balancing=config, epochs=1)
    path = tmp_path / "actor.pt"
    lt._save_checkpoint(path, {"heuristic_bc_pretraining": summary})
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    record = loaded["heuristic_bc_pretraining"]["loss_balancing"]
    for key, value in config.state_dict().items():
        assert record[key] == value
    assert all(math.isfinite(w) for w in record["final_weights"].values())


def test_cli_builds_the_controller() -> None:
    args = lt.argument_parser().parse_args(
        ["pretrain-bc", "--checkpoint", "c.pt", "--dataset", "d", "--out", "o.pt",
         "--loss-balancing", "gradnorm", "--gradnorm-alpha", "0.5", "--gradnorm-init", "sqrt",
         "--gradnorm-min-rows", "8", "--gradnorm-max-weight", "30", "--gradnorm-every", "2",
         "--gradnorm-shared", "last-layer"]
    )
    config = lt._bc_loss_balancing_kwargs(args)["loss_balancing"]
    assert config == ActorGradNorm(
        alpha=0.5, init="sqrt", min_rows=8, max_weight=30.0, every=2, shared="last-layer"
    )
    with pytest.raises(ValueError):
        ActorGradNorm(shared="heads")
    with pytest.raises(ValueError):
        ActorGradNorm(init="flat")
    with pytest.raises(ValueError):
        ActorGradNorm(min_weight=2.0)
