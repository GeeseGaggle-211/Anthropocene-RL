import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from test_autonomous_readiness_v7 import batch, config

from anthropocene_rl.autonomous_v5.causal import protected_state
from anthropocene_rl.autonomous_v6.objective_collection import assign_gae, context, targets
from anthropocene_rl.autonomous_v6.objective_rollout_audit import save_rollout_audit
from anthropocene_rl.autonomous_v6.reward_objectives import (
    EpisodeObjectiveState,
    Objective,
    crowding_transferred_headcount,
)
from anthropocene_rl.training.autonomous_ppo_v7 import AutonomousTrainerV7
from anthropocene_rl.training.autonomous_ppo_v8 import AutonomousTrainerV8


def cfg(kind):
    return dict(
        config(),
        model_version="autonomous-ppo-model-v8",
        horizon_years=400,
        training_objective=Objective(kind).wire(),
        critic_objective_loss="raw-objective-units-fixed-parent-scale",
    )


def receipt(start, end, elapsed, terminal=False, qol=0.6):
    return dict(
        population_start=start,
        population_end=end,
        elapsed=elapsed,
        qol=qol,
        reward=start * qol * elapsed,
        terminal=terminal,
    )


def test_training_objectives_keep_true_focal_welfare_and_reset_normalization():
    r = receipt(50, 40, 20)
    before = copy.deepcopy(r)
    original, midpoint, alpha = [
        Objective(k).interval(r, 100) for k in ["original", "midpoint", "superlinear"]
    ]
    assert r == before
    assert (
        original["original_welfare"]
        == midpoint["original_welfare"]
        == alpha["original_welfare"]
        == 600
    )
    assert midpoint["raw_training_welfare"] == 540
    assert alpha["raw_training_welfare"] == pytest.approx(100 * 0.5**1.5 * 0.6 * 20)
    for p0 in [10, 40, 100, 400]:
        objective = Objective("superlinear")
        actual = objective.interval(r, p0)
        reference = objective.interval(r, objective.reference_population)
        assert actual["reward"] == pytest.approx(actual["value_factor"] * reference["reward"])
    with pytest.raises(ValueError):
        Objective("original").interval(dict(r, reward=800), 100)
    for bad in [0, -1, float("nan")]:
        with pytest.raises(ValueError):
            EpisodeObjectiveState(bad, Objective("superlinear"))


@pytest.mark.parametrize("terminal", [False, True])
def test_bounded_potential_telescopes_and_gae_matches_with_physical_time_and_tail(terminal):
    population, elapsed = [100, 130, 40, 25], [3, 20, 55]
    plain, shaped = [], []
    base = Objective("original")
    potential = Objective("potential", potential_coefficient=0.25)
    values = [22.0, 31.0, 8.0, 4.0]
    if terminal:
        population[-1] = 0
        values[-1] = 0
    for i in range(3):
        r = receipt(population[i], population[i + 1], elapsed[i], terminal and i == 2)
        common = dict(elapsed=elapsed[i], terminal=r["terminal"], truncated=not terminal and i == 2)
        plain.append(
            dict(
                common,
                value=values[i],
                next_value=values[i + 1],
                reward=base.interval(r, 100)["reward"],
            )
        )
        shaped.append(
            dict(
                common,
                value=values[i] - potential.potential(population[i]),
                next_value=values[i + 1] - potential.potential(population[i + 1], r["terminal"]),
                reward=potential.interval(r, 100)["reward"],
            )
        )
    conf = dict(horizon_years=400, lambda_ref=0.95)
    assign_gae(plain, conf)
    assign_gae(shaped, conf)
    np.testing.assert_allclose(
        [r["advantage"] for r in plain], [r["advantage"] for r in shaped], atol=1e-12
    )
    discount, delta = 1.0, 0.0
    for a, b in zip(plain, shaped, strict=True):
        delta += discount * (b["reward"] - a["reward"])
        discount *= math.exp(-a["elapsed"] / 400)
    expected = -potential.potential(population[0]) + discount * potential.potential(
        population[-1], terminal
    )
    assert delta == pytest.approx(expected)
    tail = values[-1]
    corrected_tail = tail - potential.potential(population[-1], terminal)
    left, right = targets(plain, tail, 400), targets(shaped, corrected_tail, 400)
    np.testing.assert_allclose(
        right["target"],
        left["target"] - np.array([potential.potential(p) for p in population[:-1]]),
        atol=1e-12,
    )
    assert potential.potential(1e12) == 25


def test_mid_episode_context_preserves_original_p0_after_outflows_and_restore():
    state = EpisodeObjectiveState(100, Objective("superlinear"))
    restored = EpisodeObjectiveState.restore(state.wire())
    x = context(batch(), restored, 25)
    assert (x["InitialPopulation"] == 100).all() and (x["OwnPopulation"] == 25).all()
    assert (x["ValueFactor"] == 1).all()
    assert restored.wire() == state.wire()
    with pytest.raises(ValueError):
        EpisodeObjectiveState.restore(dict(state.wire(), initial_population=0))


def test_crowding_credit_basis_excludes_other_fission_migration_and_daughter_welfare():
    r = dict(
        demographics=dict(
            SchemaVersion=1,
            FissionHeadcount=19,
            FissionByTrigger={"Crowding": 3, "Distress": 7, "ExitOpportunity": 5, "LegacyQoL": 4},
            Emigrants=9,
            StarvationDeaths=8,
            DaughterWelfare=1e9,
        )
    )
    assert crowding_transferred_headcount(r) == 3
    r["demographics"]["FissionHeadcount"] = 20
    with pytest.raises(ValueError):
        crowding_transferred_headcount(r)


def test_objective_fork_preserves_actor_adam_rng_values_and_exact_continuation(tmp_path):
    torch.set_num_threads(1)
    parent = AutonomousTrainerV7(dict(config(), horizon_years=400), 47, {}, width=16)
    sum(p.square().mean() for p in parent.model.parameters() if p.requires_grad).backward()
    parent.optimizer.step()
    parent.optimizer.zero_grad(set_to_none=True)
    parent.model.fit_value_scale(torch.tensor([10.0, 20.0, 70.0]))
    parent.episodes, parent.transitions, parent.updates = 4, 87, 1
    path = tmp_path / "parent.pt"
    parent.checkpoint(path)
    new = AutonomousTrainerV8(cfg("original"), 13, {}, width=16)
    new.import_v7(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert protected_state(parent) == protected_state(new)
    x, adapted = batch(), context(batch(), EpisodeObjectiveState(100, Objective("original")), 50)
    with torch.no_grad():
        a, lp, _, value = parent.model(x, generator=torch.Generator().manual_seed(19))
        b, blp, _, bvalue = new.model(adapted, generator=torch.Generator().manual_seed(19))
    assert torch.equal(lp, blp) and torch.equal(value, bvalue)
    assert torch.equal(a.trade, b.trade) and torch.equal(a.workforce, b.workforce)
    with pytest.raises(ValueError):
        new.model.value(x)
    new.config = dict(new.config, epochs=1, minibatch=4)
    rows = [
        dict(
            obs={k: v[i : i + 1] for k, v in adapted.items()},
            action=a.index(torch.tensor([i])),
            logp=lp[i : i + 1],
            advantage=float(i - 3),
            **{"return": float(value[i]) + i},
        )
        for i in range(8)
    ]
    report = new.update(rows)
    assert any(r["actor_grad"] > 0 and r["critic_grad"] > 0 for r in report)
    path = tmp_path / "v8.pt"
    new.checkpoint(path)
    copy_trainer = AutonomousTrainerV8(new.config, 97, {}, width=16)
    copy_trainer.restore(path)
    assert new.update(rows) == copy_trainer.update(rows)
    assert all(
        torch.equal(v, copy_trainer.model.state_dict()[k])
        for k, v in new.model.state_dict().items()
    )


def test_fixed_rollout_fills_early_terminals_before_equal_ppo_boundary(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from run_autonomous_reward_gate_v11 import fixed_rollout

    class Collector:
        def __init__(self):
            self.attempts = []

        def rollout(self, trainer, bank, pool, seed, count):
            self.attempts.append(count)
            actual = min(3, count)
            return list(range(actual)), [dict(steps=actual)]

    collector = Collector()
    rows, episodes = fixed_rollout(collector, None, None, None, 42, 8)
    assert len(rows) == 8 and sum(e["steps"] for e in episodes) == 8
    assert collector.attempts == [8, 5, 2]


def test_factor_aware_intercept_and_audit_keep_objective_context(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from run_autonomous_reward_gate_v11 import fit_objective_intercept

    trainer = AutonomousTrainerV8(cfg("superlinear"), 91, {}, width=16)
    state = EpisodeObjectiveState(400, Objective("superlinear"))
    x = context(batch(), state, 50)
    trainer.model.fit_value_scale(torch.tensor([10.0, 20.0, 60.0]))
    before = protected_state(trainer)
    with torch.no_grad():
        y = trainer.model.value(x) + 7
    result = fit_objective_intercept(trainer, (x, y))
    assert protected_state(trainer) == before
    assert result["raw_mean_applied_delta"] == pytest.approx(7, abs=1e-5)
    with torch.no_grad():
        action, lp, _, values = trainer.model(x, generator=trainer.sampler)
    rows = [
        dict(
            obs={k: v[i : i + 1] for k, v in x.items()},
            action=action.index(torch.tensor([i])),
            logp=lp[i : i + 1],
            value=float(values[i]),
            next_value=float(values[i]),
            elapsed=20,
            reward=0.4,
            terminal=False,
            truncated=i == 7,
            advantage=float(i - 3),
            **{"return": float(values[i]) + i},
        )
        for i in range(8)
    ]
    path = tmp_path / "audit.pt"
    save_rollout_audit(trainer, rows, path, rollout_provenance={"test": "objective-context"})
    audit = torch.load(path, weights_only=False)
    assert audit["version"] == "model8-native6-objective-rollout-audit-v1"
    assert audit["config"]["training_objective"]["kind"] == "superlinear"
    assert all(float(r["obs"]["InitialPopulation"][0]) == 400 for r in audit["rows"])
    assert all(float(r["obs"]["ValueFactor"][0]) == 0.5 for r in audit["rows"])
