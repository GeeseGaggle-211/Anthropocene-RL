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
    config = ActorGradNorm(
        alpha=0.0, lr=0.5, ema=0.5, init="share", min_weight=1e-3, max_weight=1e3, hold_below=0.0
    )
    state = ActorGradNormState(config, SCALES)
    raw = {t: 0.01 * (1 + 3 * i) for i, t in enumerate(ACTOR_BC_TASKS)}
    staffed = {t: 100 for t in ACTOR_ROLE_TASKS}
    for _ in range(200):
        state.update(raw, {t: 1.0 for t in ACTOR_BC_TASKS}, staffed)
    weighted = [state.values()[t] * state.ema_norm[t] for t in ACTOR_BC_TASKS]
    assert max(weighted) / min(weighted) == pytest.approx(1.0, abs=1e-6)
    assert state.scale_sum() == pytest.approx(state.target_scale, rel=1e-9)


def test_negligible_gradient_tasks_are_held_not_inflated() -> None:
    config = ActorGradNorm(lr=1.0, ema=1.0, init="share", max_weight=1e3)
    state = ActorGradNormState(config, SCALES)
    raw = {t: 0.05 * SCALES.get(t, 1.0) for t in ACTOR_BC_TASKS}  # equal intensity g/s
    raw["fission"] = raw["trade_gate"] = 1e-6  # a solved, constant-label head
    staffed = {t: 100 for t in ACTOR_ROLE_TASKS}
    for _ in range(20):
        entry = state.update(raw, {t: 1.0 for t in ACTOR_BC_TASKS}, staffed)
    assert set(entry["negligible"]) == {"fission", "trade_gate"}
    assert state.values()["fission"] == 1.0 and state.values()["trade_gate"] == 1.0
    assert state.task_negligible["fission"] == 20
    assert state.scale_sum() == pytest.approx(state.target_scale, rel=1e-9)
    assert min(state.values().values()) > config.min_weight  # nothing pinned to the floor
    unheld = ActorGradNormState(
        ActorGradNorm(lr=1.0, ema=1.0, init="share", hold_below=0.0), SCALES
    )
    for _ in range(20):
        unheld.update(raw, {t: 1.0 for t in ACTOR_BC_TASKS}, staffed)
    assert unheld.values()["fission"] > 2.0  # the failure the hold prevents


def test_alpha_favours_slower_tasks() -> None:
    config = ActorGradNorm(alpha=1.0, lr=1.0, ema=1.0, init="share", max_weight=1e3, hold_below=0.0)
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
    assert (
        record["final_weights"]["activity:ClayMiner"]
        == record["initial_weights"]["activity:ClayMiner"]
    )
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
        [
            "pretrain-bc",
            "--checkpoint",
            "c.pt",
            "--dataset",
            "d",
            "--out",
            "o.pt",
            "--loss-balancing",
            "gradnorm",
            "--gradnorm-alpha",
            "0.5",
            "--gradnorm-init",
            "sqrt",
            "--gradnorm-min-rows",
            "8",
            "--gradnorm-max-weight",
            "30",
            "--gradnorm-every",
            "2",
            "--gradnorm-shared",
            "last-layer",
        ]
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


# ---------------------------------------------------------------------------
# Two-level (grouped) balancing
# ---------------------------------------------------------------------------

from anthropocene_rl.autonomous_v7.labor_actor_gradnorm import (  # noqa: E402
    ACTOR_GROUPS,
    ActorGroupedGradNorm,
    ActorGroupedGradNormState,
)

STAFFED = {t: 100 for t in ACTOR_ROLE_TASKS}


def _role_sum(state: ActorGroupedGradNormState) -> float:
    return sum(state.role_weights()[t] * SCALES[t] for t in ACTOR_ROLE_TASKS)


def test_grouped_initial_weights_and_invariants() -> None:
    state = ActorGroupedGradNormState(ActorGroupedGradNorm(inner_max_weight=1000.0), SCALES)
    weights = state.values()
    assert set(weights) == set(ACTOR_BC_TASKS)
    for task in ("specialty", "movement", "fission", "trade_gate"):
        assert weights[task] == 1.0
    assert state.group_weights() == {g: 1.0 for g in ACTOR_GROUPS}
    assert _role_sum(state) == pytest.approx(sum(SCALES.values()), rel=1e-9)
    masses = [state.role_weights()[t] * SCALES[t] for t in ACTOR_ROLE_TASKS]
    assert max(masses) == pytest.approx(min(masses))  # uniform per role inside activity
    for t in ACTOR_ROLE_TASKS:  # effective role weight = W_activity * v_r
        assert weights[t] == pytest.approx(state.role_weights()[t])
    share = ActorGroupedGradNormState(ActorGroupedGradNorm(init="share"), SCALES)
    assert share.values() == {t: 1.0 for t in ACTOR_BC_TASKS}  # the historical loss


def test_grouped_top_equalises_groups_and_renormalises() -> None:
    config = ActorGroupedGradNorm(lr=0.5, ema=0.5, min_weight=0.01, max_weight=3.0)
    state = ActorGroupedGradNormState(config, SCALES)
    norms = {"specialty": 0.8, "movement": 2.0, "activity": 0.4}
    for _ in range(200):
        state.update(norms, {g: 1.0 for g in ACTOR_GROUPS}, None, None, STAFFED)
    top = state.group_weights()
    assert sum(top.values()) == pytest.approx(3.0, rel=1e-9)
    weighted = [top[g] * norms[g] for g in ACTOR_GROUPS]
    assert max(weighted) / min(weighted) == pytest.approx(1.0, abs=1e-6)
    assert state.values()["fission"] == 1.0 and state.values()["trade_gate"] == 1.0
    roles = state.values()
    for t in ACTOR_ROLE_TASKS:
        assert roles[t] == pytest.approx(top["activity"] * state.role_weights()[t])


def test_grouped_group_scale_is_independent_of_inner_weights() -> None:
    config = ActorGroupedGradNorm(inner="gradnorm", lr=1.0, ema=1.0, inner_max_weight=1000.0)
    state = ActorGroupedGradNormState(config, SCALES)
    role_norms = {t: 0.01 * (1 + i) for i, t in enumerate(ACTOR_ROLE_TASKS)}
    groups = {g: 1.0 for g in ACTOR_GROUPS}
    state.update(groups, groups, role_norms, {t: 1.0 for t in ACTOR_ROLE_TASKS}, STAFFED)
    before_top = state.group_weights()
    before_inner = state.role_weights()
    shifted = dict(role_norms, **{"activity:Gatherer": 5.0})
    state.update(groups, groups, shifted, {t: 1.0 for t in ACTOR_ROLE_TASKS}, STAFFED)
    assert state.role_weights() != before_inner  # the roles moved...
    assert state.group_weights() == pytest.approx(before_top)  # ...the groups did not
    assert _role_sum(state) == pytest.approx(sum(SCALES.values()), rel=1e-9)


def test_grouped_inner_holds_and_inner_none_is_fixed() -> None:
    fixed = ActorGroupedGradNormState(ActorGroupedGradNorm(inner="none"), SCALES)
    initial = fixed.role_weights()
    role_norms = {t: 0.01 * (1 + i) for i, t in enumerate(ACTOR_ROLE_TASKS)}
    groups = {g: 1.0 for g in ACTOR_GROUPS}
    fixed.update(groups, groups, role_norms, {t: 1.0 for t in ACTOR_ROLE_TASKS}, STAFFED)
    assert fixed.role_weights() == initial
    config = ActorGroupedGradNorm(inner="gradnorm", min_rows=4, inner_max_weight=1000.0)
    state = ActorGroupedGradNormState(config, SCALES)
    potter = state.role_weights()["activity:Potter"]
    staffed = dict(STAFFED, **{"activity:Potter": 2})
    norms = {t: 0.05 * SCALES[t] for t in ACTOR_ROLE_TASKS}
    norms["activity:Fisher"] = 1e-9  # negligible: held
    entry = state.update(groups, groups, norms, {t: 1.0 for t in ACTOR_ROLE_TASKS}, staffed)
    assert state.role_weights()["activity:Potter"] == potter
    assert state.inner.task_holds["activity:Potter"] == 1
    assert entry["inner"]["negligible"] == ["activity:Fisher"]
    snapshot = state.snapshot()
    state.update(groups, groups, norms, {t: 1.0 for t in ACTOR_ROLE_TASKS}, staffed)
    state.restore(snapshot)
    assert state.inner.task_holds["activity:Potter"] == 1 and state.updates == 1


def test_grouped_fit_logs_groups_roles_and_judges_dominance(tmp_path) -> None:
    jsonl = tmp_path / "stability.jsonl"
    config = ActorGroupedGradNorm(inner="gradnorm", min_rows=2)
    _, summary = _fit(
        loss_balancing=config, stability=StabilityConfig(jsonl=str(jsonl), log_every=2)
    )
    record = summary["loss_balancing"]
    assert record["mode"] == "grouped" and record["groups"] == list(ACTOR_GROUPS)
    assert record["updates"] == 8
    assert sum(record["final_group_weights"].values()) == pytest.approx(3.0, rel=1e-6)
    steps = [json.loads(line) for line in jsonl.read_text().splitlines()]
    logged = [r for r in steps if r.get("kind") == "step" and "actor_group_weights" in r]
    assert logged
    assert set(logged[0]["shared_head_grad_norm"]["actor_groups"]) == set(ACTOR_GROUPS)
    assert "activity:Gatherer" in logged[0]["actor_roles"]["raw_grad_norm"]
    assert "top" in logged[0]["actor_gradnorm"]
    assert summary["stability"]["checks"]["gradient_dominance"]["status"] in ("PASS", "FAIL")


def test_grouped_checkpoint_round_trip_and_cli(tmp_path) -> None:
    config = ActorGroupedGradNorm(
        alpha=0.5, init="sqrt", inner="gradnorm", inner_max_weight=20.0, every=2
    )
    _, summary = _fit(loss_balancing=config, epochs=1)
    path = tmp_path / "actor.pt"
    lt._save_checkpoint(path, {"heuristic_bc_pretraining": summary})
    record = torch.load(path, map_location="cpu", weights_only=True)["heuristic_bc_pretraining"][
        "loss_balancing"
    ]
    for key, value in config.state_dict().items():
        assert record[key] == value
    args = lt.argument_parser().parse_args(
        [
            "pretrain-bc",
            "--checkpoint",
            "c.pt",
            "--dataset",
            "d",
            "--out",
            "o.pt",
            "--loss-balancing",
            "grouped",
            "--gradnorm-inner",
            "gradnorm",
            "--gradnorm-inner-max-weight",
            "20",
            "--gradnorm-every",
            "4",
        ]
    )
    built = lt._bc_loss_balancing_kwargs(args)["loss_balancing"]
    assert built == ActorGroupedGradNorm(inner="gradnorm", inner_max_weight=20.0, every=4)
    bad = lt.argument_parser().parse_args(
        [
            "pretrain-bc",
            "--checkpoint",
            "c.pt",
            "--dataset",
            "d",
            "--out",
            "o.pt",
            "--loss-balancing",
            "gradnorm",
            "--gradnorm-inner",
            "gradnorm",
        ]
    )
    with pytest.raises(ValueError, match="grouped only"):
        lt._bc_loss_balancing_kwargs(bad)
    with pytest.raises(ValueError):
        ActorGroupedGradNorm(inner="flat")
