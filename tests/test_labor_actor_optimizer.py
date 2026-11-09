"""The labor actor's two-group Adam (encoder / heads), lr-as-configuration, warm-up and
PPO ratio-overflow rollback (``labor_actor_optim``, ``update_labor_policy``).

Synthetic CPU fixtures only (no native DLL, no sealed or reserved worlds).
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import math
import subprocess
import sys

import numpy as np
import pytest
import torch
from test_labor_a_ppo_wiring import (  # noqa: F401  (e2e is a fixture)
    REPO,
    _default_models,
    _same,
    _small,
    _train_args,
    e2e,
)
from test_labor_ppo import _DynamicActor, _gate, _NonlaborPolicy, _policy_batch
from test_labor_ppo_perhead import _gate as _actor_gate
from test_labor_ppo_perhead import _heads_scorer, _joint_scorer
from torch import nn

from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7.labor_actor_optim import (
    ACTOR_PARAM_GROUPS,
    ActorLrConfig,
    ActorLrSchedule,
    actor_group_counts,
    actor_lr_config,
    actor_optimizer_state_by_name,
    actor_parameter_names,
    adam_steps_taken,
    apply_actor_lrs,
    build_actor_optimizer,
    load_actor_optimizer_state,
    rebuild_actor_optimizer,
)
from anthropocene_rl.autonomous_v7.labor_model import LaborPersonality
from anthropocene_rl.autonomous_v7.labor_ppo import LaborRatioOverflow, update_labor_policy
from anthropocene_rl.autonomous_v7.value_calibration import behavior_policy_digest

BASELINE = "f948d4a"  # the commit before per-group actor learning rates


def _at_commit(name: str):
    """Import ``name`` as it was at ``BASELINE`` under a private module name."""
    modname = f"anthropocene_rl.autonomous_v7._f948d4a_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    if name != "labor_ppo":
        _at_commit("labor_ppo")
    try:
        source = subprocess.run(
            ["git", "show", f"{BASELINE}:src/anthropocene_rl/autonomous_v7/{name}.py"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("baseline source is not available from git")
    source = source.replace("from .labor_ppo import", "from ._f948d4a_labor_ppo import")
    spec = importlib.util.spec_from_loader(modname, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "anthropocene_rl.autonomous_v7"
    sys.modules[modname] = module
    exec(compile(source, f"<{BASELINE}:{name}.py>", "exec"), module.__dict__)
    return module


def _state_by_name(params: dict[str, nn.Parameter], optimizer) -> dict[str, dict]:
    by_param = {p: n for n, p in params.items()}
    return {by_param[p]: s for p, s in optimizer.state.items()}


class _HeadsFirst(nn.Module):
    """Encoder parameters NOT first in named_parameters(): a legacy one-group index is
    then not the two-group index, so a remap by position-in-the-flat-list would misalign."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Linear(3, 2)
        self.encoder = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 3))
        self.omega = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.omega(h) + self.alpha(h).sum(dim=1, keepdim=True)


def _steps(module: nn.Module, optimizer, count: int, seed: int = 0) -> None:
    generator = torch.Generator().manual_seed(seed)
    for _ in range(count):
        x = torch.randn(5, 4, generator=generator)
        optimizer.zero_grad()
        module(x).square().mean().backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


def test_groups_split_the_real_actor_by_encoder_prefix_in_named_order():
    actor = _default_models()[0]
    names = [name for name, _ in actor.named_parameters()]
    groups = actor_parameter_names(actor)
    assert list(groups) == list(ACTOR_PARAM_GROUPS)
    assert groups["encoder"] and groups["heads"]
    assert all(n.startswith("encoder.") for n in groups["encoder"])
    assert not any(n.startswith("encoder.") for n in groups["heads"])
    assert sorted(groups["encoder"] + groups["heads"]) == sorted(names)
    for group in groups.values():  # named_parameters() order within each group
        assert group == [n for n in names if n in set(group)]
    counts = actor_group_counts(actor)
    assert counts["encoder"]["parameters"] + counts["heads"]["parameters"] == sum(
        p.numel() for p in actor.parameters()
    )
    optimizer = build_actor_optimizer(actor, ActorLrConfig(1e-7, 3e-5))
    assert [g["name"] for g in optimizer.param_groups] == ["encoder", "heads"]
    assert [g["lr"] for g in optimizer.param_groups] == [1e-7, 3e-5]
    assert [g["param_names"] for g in optimizer.param_groups] == [
        groups["encoder"], groups["heads"]
    ]


def test_cli_flags_default_to_actor_lr_and_no_warmup():
    parser = training.argument_parser()
    base = ["train", "--checkpoint", "c", "--rollout", "r", "--anchor", "a", "--out", "o"]
    args = parser.parse_args([*base, "--actor-lr", "2e-5"])
    assert (args.actor_encoder_lr, args.actor_head_lr, args.actor_lr_warmup_steps) == (
        None, None, 0
    )
    assert actor_lr_config(args) == ActorLrConfig(2e-5, 2e-5, 0)
    args = parser.parse_args([*base, "--actor-encoder-lr", "0", "--actor-head-lr", "3e-5",
                              "--actor-lr-warmup-steps", "12"])
    assert actor_lr_config(args) == ActorLrConfig(0.0, 3e-5, 12)
    init = parser.parse_args(["initialize", "--normalizer-checkpoint", "n", "--world-seed", "1",
                              "--optimizer-seed", "1", "--out", "o", "--actor-head-lr", "1e-5"])
    assert actor_lr_config(init) == ActorLrConfig(3e-4, 1e-5, 0)
    with pytest.raises(ValueError):
        ActorLrConfig(0.0, 0.0)
    with pytest.raises(ValueError):
        ActorLrConfig(-1e-5, 1e-5)
    with pytest.raises(ValueError):
        ActorLrConfig(1e-5, 1e-5, -1)


# ---------------------------------------------------------------------------
# Defaults are bitwise the single-group optimizer (f948d4a)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("per_head", [False, True])
def test_default_update_is_bitwise_the_single_group_update(monkeypatch, per_head):
    old_ppo = _at_commit("labor_ppo")
    models, _, batch, _ = _small(monkeypatch)
    results = []
    for new in (False, True):
        actor = copy.deepcopy(models[0])
        params = dict(actor.named_parameters())
        if new:
            config = ActorLrConfig(1e-5, 1e-5)
            optimizer = build_actor_optimizer(actor, config)
            update, extra = update_labor_policy, {"lr_schedule": ActorLrSchedule(config)}
            candidate = batch
        else:
            optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
            update, extra = old_ppo.update_labor_policy, {}
            candidate = old_ppo.LaborAttemptPolicyBatch(
                **{f.name: getattr(batch, f.name)
                   for f in dataclasses.fields(old_ppo.LaborAttemptPolicyBatch)}
            )
        stats = update(
            policy_module=actor, labor_actor=actor, optimizer=optimizer, batch=candidate,
            nonlabor_scorer=(_heads_scorer if per_head else _joint_scorer)(actor),
            rng=np.random.default_rng(3), epochs=2, batch_size=3, clip=0.2,
            entropy_weight=0.01, max_grad_norm=1.0, target_kl=10.0, audit_rows=4,
            actor_gate=_actor_gate(batch),
            behavior_configuration=training.labor_behavior_configuration(LaborPersonality()),
            **({"per_head_surrogate": True} if per_head else {}), **extra,
        )
        results.append((stats, actor.state_dict(), _state_by_name(params, optimizer)))
    (stats_old, actor_old, opt_old), (stats_new, actor_new, opt_new) = results
    record = stats_new.pop("actorOptimizer")
    assert stats_old == stats_new and stats_new["acceptedSteps"] == 6
    assert record["lrFactorPerAttemptedStep"] == [1.0] * 6 and record["rollbacks"] == 0
    for name, value in actor_old.items():
        assert torch.equal(value, actor_new[name]), name
    assert set(opt_old) == set(opt_new)
    for name, state in opt_old.items():
        for key, value in state.items():
            assert torch.equal(value, opt_new[name][key]), (name, key)


def test_default_train_is_bitwise_the_baseline_train(e2e, monkeypatch):  # noqa: F811
    """Same seed and fixture: f948d4a's train vs this train with the new flags at default
    (--actor-lr equal to the checkpoint's saved lr, as the fixture saves 1e-6)."""
    old = _at_commit("labor_training")
    checkpoint, rollout, anchor, gate = e2e["files0"]
    monkeypatch.setattr(old, "load_labor_anchor", training.load_labor_anchor)
    outputs = {}
    for tag, main in (("old", old.main), ("new", training.main)):
        out = e2e["tmp"] / f"{tag}.pt"
        main(_train_args(checkpoint, rollout, anchor, gate, out))
        outputs[tag] = torch.load(out, weights_only=True)
    old_state, new_state = outputs["old"], outputs["new"]
    assert old_state["last_update"]["actor"]["acceptedSteps"] > 0
    _, actor, *_ = training._models_from_checkpoint(new_state)
    # Everything but the optimizer layout and the added records is bitwise equal.
    for key in set(old_state) | set(new_state):
        if key not in ("actor_optimizer", "last_update"):
            _same(old_state[key], new_state[key], key)
    old_update, new_update = dict(old_state["last_update"]), dict(new_state["last_update"])
    record = new_update.pop("actorOptimizer")
    new_actor_stats = dict(new_update.pop("actor"))
    new_actor_stats.pop("actorOptimizer")
    old_actor_stats = old_update.pop("actor")
    _same(old_update, new_update, "last_update")
    _same(old_actor_stats, new_actor_stats, "last_update.actor")
    assert record["encoderLr"] == record["headLr"] == 1e-6 and record["warmupSteps"] == 0
    assert record["load"]["savedLayout"] == "legacy-single-group"
    # Per-parameter Adam state (moments and steps) is bitwise equal, by parameter name.
    old_by_name = actor_optimizer_state_by_name(actor, old_state["actor_optimizer"])
    new_by_name = actor_optimizer_state_by_name(actor, new_state["actor_optimizer"])
    assert set(old_by_name) == set(new_by_name) and old_by_name
    for name, state in old_by_name.items():
        _same(state, new_by_name[name], name)
    assert [g["name"] for g in new_state["actor_optimizer"]["param_groups"]] == [
        "encoder", "heads"
    ]
    old_group = {k: v for k, v in old_state["actor_optimizer"]["param_groups"][0].items()
                 if k != "params"}
    for group in new_state["actor_optimizer"]["param_groups"]:
        assert {k: v for k, v in group.items()
                if k not in ("params", "param_names", "name")} == old_group


# ---------------------------------------------------------------------------
# Checkpoint compatibility: legacy one-group <-> two-group
# ---------------------------------------------------------------------------


def test_legacy_one_group_state_remaps_by_parameter_and_continues_bitwise():
    torch.manual_seed(0)
    module = _HeadsFirst()
    names = [n for n, _ in module.named_parameters()]
    assert not names[0].startswith("encoder.")  # the remap is not the identity
    legacy = torch.optim.Adam(module.parameters(), lr=1e-2)
    _steps(module, legacy, 3)
    saved = copy.deepcopy(legacy.state_dict())
    reference = copy.deepcopy(module)
    reference_opt = torch.optim.Adam(reference.parameters(), lr=1e-2)
    reference_opt.load_state_dict(copy.deepcopy(saved))

    config = ActorLrConfig(1e-2, 1e-2)
    optimizer = build_actor_optimizer(module, config)
    record = load_actor_optimizer_state(optimizer, module, saved, config)
    assert record["savedLayout"] == "legacy-single-group"
    assert record["adamStepsTaken"] == 3
    params = dict(module.named_parameters())
    legacy_by_name = {names[i]: s for i, s in saved["state"].items()}
    loaded = _state_by_name(params, optimizer)
    assert set(loaded) == set(names)
    for name in names:  # the moments landed on the same tensors
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(loaded[name][key], legacy_by_name[name][key]), (name, key)
        assert loaded[name]["exp_avg"].shape == params[name].shape
    # Continuing either optimizer gives bitwise the same parameters and moments.
    _steps(module, optimizer, 2, seed=1)
    _steps(reference, reference_opt, 2, seed=1)
    for (name, a), (_, b) in zip(module.named_parameters(), reference.named_parameters(),
                                 strict=True):
        assert torch.equal(a, b), name
    ref_state = _state_by_name(dict(reference.named_parameters()), reference_opt)
    for name, state in _state_by_name(params, optimizer).items():
        for key, value in state.items():
            assert torch.equal(value, ref_state[name][key]), (name, key)


def test_two_group_state_round_trips_and_back_to_names():
    torch.manual_seed(1)
    module = _HeadsFirst()
    config = ActorLrConfig(1e-3, 1e-2)
    optimizer = build_actor_optimizer(module, config)
    _steps(module, optimizer, 2)
    saved = copy.deepcopy(optimizer.state_dict())
    assert [g["name"] for g in saved["param_groups"]] == ["encoder", "heads"]
    other = copy.deepcopy(module)
    again = build_actor_optimizer(other, ActorLrConfig(5e-4, 5e-4))
    record = load_actor_optimizer_state(again, other, saved, ActorLrConfig(5e-4, 5e-4))
    assert record["savedLrs"] == {"encoder": 1e-3, "heads": 1e-2}
    assert record["appliedLrs"] == {"encoder": 5e-4, "heads": 5e-4}
    a = _state_by_name(dict(module.named_parameters()), optimizer)
    b = _state_by_name(dict(other.named_parameters()), again)
    assert set(a) == set(b)
    for name in a:
        for key in a[name]:
            assert torch.equal(a[name][key], b[name][key]), (name, key)
    # Pass-through rebuild keeps the saved rates.
    rebuilt = rebuild_actor_optimizer(other, saved)
    assert [g["lr"] for g in rebuilt.param_groups] == [1e-3, 1e-2]
    assert actor_optimizer_state_by_name(other, saved).keys() == a.keys()


def test_mismatched_states_are_refused():
    torch.manual_seed(2)
    module = _HeadsFirst()
    config = ActorLrConfig(1e-3, 1e-3)
    optimizer = build_actor_optimizer(module, config)
    # Legacy state of a different module (one parameter fewer).
    smaller = nn.Sequential(nn.Linear(4, 3))
    legacy_small = torch.optim.Adam(smaller.parameters()).state_dict()
    with pytest.raises(ValueError, match="legacy actor optimizer state has 2 parameters"):
        load_actor_optimizer_state(optimizer, module, legacy_small, config)
    # Legacy state whose moments have the wrong shapes (same parameter count).
    twin = _HeadsFirst()
    twin.alpha = nn.Linear(3, 5)
    twin_opt = torch.optim.Adam(twin.parameters())
    _steps(twin, twin_opt, 1)
    with pytest.raises(ValueError, match="has shape"):
        load_actor_optimizer_state(optimizer, module, twin_opt.state_dict(), config)
    # Two-group state with a group of another size.
    good = copy.deepcopy(build_actor_optimizer(module, config).state_dict())
    bad = copy.deepcopy(good)
    bad["param_groups"][0]["params"] = bad["param_groups"][0]["params"][:-1]
    bad["param_groups"][0]["param_names"] = bad["param_groups"][0]["param_names"][:-1]
    with pytest.raises(ValueError, match="size differs"):
        load_actor_optimizer_state(optimizer, module, bad, config)
    renamed = copy.deepcopy(good)
    renamed["param_groups"][1]["name"] = "trunk"
    with pytest.raises(ValueError, match="named"):
        load_actor_optimizer_state(optimizer, module, renamed, config)
    three = copy.deepcopy(good)
    three["param_groups"].append(copy.deepcopy(three["param_groups"][1]))
    with pytest.raises(ValueError, match="3 parameter groups"):
        load_actor_optimizer_state(optimizer, module, three, config)
    with pytest.raises(ValueError, match="not built for this actor"):
        load_actor_optimizer_state(optimizer, twin, good, config)


def test_real_actor_legacy_checkpoint_state_loads_into_two_groups():
    """The real LaborActor layout, as checkpoint-start.pt carries it (one group)."""
    actor = _default_models()[0]
    legacy = torch.optim.Adam(actor.parameters(), lr=3e-4)
    for p in actor.parameters():
        p.grad = torch.randn_like(p)
    legacy.step()
    saved = copy.deepcopy(legacy.state_dict())
    config = ActorLrConfig(0.0, 3e-5)
    optimizer = build_actor_optimizer(actor, config)
    record = load_actor_optimizer_state(optimizer, actor, saved, config)
    assert record["appliedLrs"] == {"encoder": 0.0, "heads": 3e-5}
    names = [n for n, _ in actor.named_parameters()]
    loaded = _state_by_name(dict(actor.named_parameters()), optimizer)
    for index, name in enumerate(names):
        for key, value in saved["state"][index].items():
            assert torch.equal(loaded[name][key], value), (name, key)


# ---------------------------------------------------------------------------
# Learning rate is configuration; warm-up from the Adam state; rollback
# ---------------------------------------------------------------------------


def test_lr_on_load_is_the_configuration_not_the_saved_value():
    module = _HeadsFirst()
    legacy = torch.optim.Adam(module.parameters(), lr=1e-3)
    _steps(module, legacy, 1)
    config = ActorLrConfig(1e-6, 2e-5)
    optimizer = build_actor_optimizer(module, config)
    record = load_actor_optimizer_state(optimizer, module, legacy.state_dict(), config)
    assert [g["lr"] for g in optimizer.param_groups] == [1e-6, 2e-5]
    assert record["savedLrs"] == {"legacy-single-group": 1e-3}
    assert record["lrSource"].startswith("configuration")
    # Without a configuration (pass-through) the saved rate is kept.
    kept = build_actor_optimizer(module, config)
    load_actor_optimizer_state(kept, module, legacy.state_dict())
    assert [g["lr"] for g in kept.param_groups] == [1e-3, 1e-3]


def test_train_reapplies_the_configured_lr_over_the_checkpoint_lr(e2e):  # noqa: F811
    """The fixture checkpoint saves lr 1e-6; --actor-lr 1e-5 must win (it used not to)."""
    checkpoint, rollout, anchor, gate = e2e["files0"]
    out = e2e["tmp"] / "lr.pt"
    args = _train_args(checkpoint, rollout, anchor, gate, out)
    args[args.index("--actor-lr") + 1] = "1e-5"
    training.main([*args, "--actor-encoder-lr", "0", "--actor-lr-warmup-steps", "3"])
    state = torch.load(out, weights_only=True)
    record = state["last_update"]["actorOptimizer"]
    assert (record["encoderLr"], record["headLr"], record["warmupSteps"]) == (0.0, 1e-5, 3)
    assert record["load"]["savedLrs"] == {"legacy-single-group": 1e-6}
    stats = state["last_update"]["actor"]
    steps = stats["attemptedSteps"]
    factors = stats["actorOptimizer"]["lrFactorPerAttemptedStep"]
    assert factors == [min(1.0, (n + 1) / 3) for n in range(steps)]
    groups = state["actor_optimizer"]["param_groups"]
    assert [g["lr"] for g in groups] == [0.0, 1e-5]  # past the warm-up: factor 1
    # The encoder was frozen; the heads moved.
    before = e2e["state0"]["actor"]
    assert all(torch.equal(v, state["actor"][k]) for k, v in before.items()
               if k.startswith("encoder.") and v.is_floating_point())
    assert any(not torch.equal(v, state["actor"][k]) for k, v in before.items()
               if not k.startswith("encoder."))


def test_warmup_factor_counts_adam_steps_and_survives_resume():
    module = _HeadsFirst()
    config = ActorLrConfig(1e-3, 1e-2, warmup_steps=3)
    optimizer = build_actor_optimizer(module, config)
    schedule = ActorLrSchedule(config)
    seen = []
    for _ in range(5):
        seen.append(schedule(optimizer))
        assert [g["lr"] for g in optimizer.param_groups] == [
            config.encoder_lr * seen[-1], config.head_lr * seen[-1]
        ]
        _steps(module, optimizer, 1)
    assert seen == [1 / 3, 2 / 3, 1.0, 1.0, 1.0]
    # Resume mid warm-up from a saved state: the factor continues from the Adam step.
    fresh = _HeadsFirst()
    first = build_actor_optimizer(fresh, config)
    _steps(fresh, first, 1)
    resumed = build_actor_optimizer(fresh, config)
    record = load_actor_optimizer_state(resumed, fresh, first.state_dict(), config)
    assert adam_steps_taken(resumed) == 1 and record["lrFactor"] == 2 / 3
    assert [g["lr"] for g in resumed.param_groups] == [1e-3 * (2 / 3), 1e-2 * (2 / 3)]
    assert apply_actor_lrs(build_actor_optimizer(_HeadsFirst(), config), config) == 1 / 3


def _toy_update(policy, actor, nonlabor, batch, optimizer, *, target_kl, **kwargs):
    behavior = {"laborRegistryDigest": batch.registry_digest, "logStdFloor": -2.0}
    batch = dataclasses.replace(
        batch, behavior_policy_digest=behavior_policy_digest(policy, behavior)
    )
    return update_labor_policy(
        policy_module=policy, labor_actor=actor, optimizer=optimizer, batch=batch,
        nonlabor_scorer=nonlabor.score, rng=np.random.default_rng(5),
        epochs=kwargs.pop("epochs", 3), batch_size=kwargs.pop("batch_size", 2), clip=0.2,
        entropy_weight=kwargs.pop("entropy_weight", 0.0), max_grad_norm=1.0,
        target_kl=target_kl, audit_rows=kwargs.pop("audit_rows", 4),
        actor_gate=_gate(policy, behavior), behavior_configuration=behavior, **kwargs,
    )


def test_rollback_restores_the_lr_with_the_adam_state_and_encoder_lr_zero_freezes():
    actor, nonlabor = _DynamicActor(), _NonlaborPolicy()
    # "encoder.mean" is the encoder group; "labor.shift" a head.
    policy = nn.ModuleDict({"labor": actor, "encoder": nonlabor})
    batch = _policy_batch(actor, nonlabor)
    config = ActorLrConfig(0.0, 0.1, warmup_steps=4)
    optimizer = build_actor_optimizer(policy, config)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    stats = _toy_update(policy, actor, nonlabor, batch, optimizer, target_kl=1e-12,
                        lr_schedule=ActorLrSchedule(config))
    assert stats["actorStepRolledBack"] and stats["attemptedSteps"] == 1
    assert stats["actorOptimizer"]["rollbacks"] == 1
    assert stats["actorOptimizer"]["lrFactorPerAttemptedStep"] == [0.25]
    assert stats["actorOptimizer"]["nextLrFactor"] == 0.25
    _same(optimizer.state_dict(), before_optimizer, "optimizer")  # lr and Adam state
    assert adam_steps_taken(optimizer) == 0

    # Accepted steps with the encoder group at lr 0: it never moves, its moments do.
    stats = _toy_update(policy, actor, nonlabor, batch, optimizer, target_kl=1e9,
                        lr_schedule=ActorLrSchedule(config))
    assert stats["acceptedSteps"] == stats["attemptedSteps"] == 6
    assert stats["actorOptimizer"]["lrFactorPerAttemptedStep"] == [0.25, 0.5, 0.75, 1, 1, 1]
    assert nonlabor.mean.item() == 0.0 and actor.shift.item() != 0.0
    assert optimizer.state[nonlabor.mean]["exp_avg"].item() != 0.0
    assert int(optimizer.state[nonlabor.mean]["step"]) == 6


# ---------------------------------------------------------------------------
# Ratio overflow is a KL violation (rollback), not an exception
# ---------------------------------------------------------------------------


class _Runaway(nn.Module):
    """Non-labor head whose entropy bonus drives one Adam step to raise the log-prob of
    the rows in ``rows`` by ``jump`` nats (exp overflows float32 above ~88.7)."""

    def __init__(self, jump: float = 100.0, rows: torch.Tensor | None = None) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.jump = jump
        self.rows = rows

    def score(self, _tensors, sample):
        z = sample["z"]
        base = -0.5 * z.square() - 0.5 * math.log(2 * math.pi)  # == _NonlaborPolicy at 0
        lift = -self.jump * self.w
        if self.rows is not None:
            lift = lift * torch.isin(z, self.rows).to(z.dtype)
        entropy = torch.full_like(z, 0.5 * math.log(2 * math.pi * math.e)) - 10.0 * self.w
        return base + lift, entropy


def _zero_advantage_batch(actor):
    batch = _policy_batch(actor, _NonlaborPolicy())
    return dataclasses.replace(
        batch,
        initial_advantage=torch.zeros_like(batch.initial_advantage),
        retry_advantage=torch.zeros_like(batch.retry_advantage),
    )


def test_ratio_overflow_on_the_post_step_audit_rolls_back():
    actor, runaway = _DynamicActor(), _Runaway()
    policy = nn.ModuleDict({"labor": actor, "nonlabor": runaway})
    batch = _zero_advantage_batch(actor)
    optimizer = torch.optim.Adam(runaway.parameters(), lr=1.0)  # only w steps
    before = copy.deepcopy(policy.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    stats = _toy_update(policy, actor, runaway, batch, optimizer, target_kl=10.0,
                        entropy_weight=1.0)
    assert stats["actorStepRolledBack"] and stats["acceptedSteps"] == 0
    assert stats["sampledNonnegativeKL"] == [math.inf]
    assert stats["ratioOverflow"]["where"] == "post-step-audit"
    _same(policy.state_dict(), before, "policy")
    _same(optimizer.state_dict(), before_optimizer, "optimizer")


def test_ratio_overflow_on_a_later_minibatch_stops_without_a_step():
    actor = _DynamicActor()
    # Only the parent with z = 0.4 (index 2) runs away; the audit sees parent 0 only.
    runaway = _Runaway(rows=torch.tensor([0.4]))
    policy = nn.ModuleDict({"labor": actor, "nonlabor": runaway})
    batch = _zero_advantage_batch(actor)
    optimizer = torch.optim.Adam(runaway.parameters(), lr=1.0)  # only w steps
    stats = _toy_update(policy, actor, runaway, batch, optimizer, target_kl=10.0,
                        entropy_weight=1.0, audit_rows=1, epochs=2, batch_size=2)
    overflow = stats["ratioOverflow"]
    assert overflow["where"] == "training-minibatch-before-step"
    assert stats["actorStepRolledBack"]
    assert stats["attemptedSteps"] == stats["acceptedSteps"] == overflow["afterAttemptedSteps"]
    assert stats["attemptedSteps"] >= 1 and runaway.w.item() < 0


def test_ratio_overflow_before_any_step_still_raises():
    actor, runaway = _DynamicActor(), _Runaway()
    policy = nn.ModuleDict({"labor": actor, "nonlabor": runaway})
    batch = _zero_advantage_batch(actor)
    # Behaviour log-probs 100 nats below the current ones: inconsistent data.
    batch = dataclasses.replace(
        batch, behaviour_initial_nonlabor_logp=batch.behaviour_initial_nonlabor_logp - 100.0
    )
    optimizer = torch.optim.Adam(runaway.parameters(), lr=1.0)  # only w steps
    with pytest.raises(LaborRatioOverflow, match="non-finite labor PPO ratio"):
        _toy_update(policy, actor, runaway, batch, optimizer, target_kl=10.0)
    assert issubclass(LaborRatioOverflow, FloatingPointError)
