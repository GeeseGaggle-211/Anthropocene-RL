"""Randomized teacher tail at the collection horizon (intervention brief 2026-10-26, item 1b)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts, labor_bc
from anthropocene_rl.autonomous_v7.gate34_continuation_collector import (
    bounded_geometric_tail_assumptions,
    estimate_randomized_tail,
)
from anthropocene_rl.autonomous_v7.labor_bc import (
    RETURN_TAIL_RANDOMIZED,
    RETURN_TAIL_TRUNCATED_ZERO,
    TAIL_HEADS,
    TAIL_NONE_TERMINAL,
    TAIL_RANDOMIZED,
    TAIL_TERMINAL_EXACT,
    TAIL_UNAVAILABLE,
    RandomizedTailSettings,
    WorldCollection,
    critic_target_available,
    discounted_returns,
    return_tail_method,
)

GAMMA = math.exp(-contracts.TURN_YEARS / contracts.HORIZON_YEARS)


def _dataset(codes, tails, terminal, camps=(4, 4, 8), turns=(0, 1, 0)):
    n = len(camps)
    data = {
        "row_world": torch.full((n,), 5000, dtype=torch.long),
        "row_camp": torch.tensor(camps, dtype=torch.long),
        "row_turn": torch.tensor(turns, dtype=torch.long),
        "row_quarantined": torch.zeros(n, dtype=torch.bool),
        "transition_elapsed_years": torch.full((n,), contracts.TURN_YEARS, dtype=torch.float64),
        "transition_terminal": torch.tensor(terminal, dtype=torch.bool),
        "reward_qol_years": torch.tensor([1.0, 2.0, 3.0][:n], dtype=torch.float64),
        "reward_population_years": torch.zeros(n, dtype=torch.float64),
        "reward_feasibility": torch.zeros(n, dtype=torch.float64),
        "reward_settlement_guard": torch.zeros(n, dtype=torch.float64),
        "tail_method": torch.tensor(codes, dtype=torch.long),
        "tail_cutoff_turn": torch.full((n,), 2, dtype=torch.long),
    }
    for head in TAIL_HEADS:
        data["tail_value__" + head] = torch.tensor(
            tails if head == "qol_years" else [0.0] * n, dtype=torch.float64
        )
        data["tail_standard_error__" + head] = torch.zeros(n, dtype=torch.float64)
    return data


def test_living_cutoff_starts_from_its_typed_tail_and_terminal_does_not() -> None:
    data = _dataset(
        [TAIL_RANDOMIZED, TAIL_RANDOMIZED, TAIL_NONE_TERMINAL],
        [10.0, 10.0, 0.0],
        [False, False, True],
    )
    assert return_tail_method(data) == RETURN_TAIL_RANDOMIZED
    returns = discounted_returns(data)["qol_years"].double()
    assert math.isclose(float(returns[1]), 2.0 + GAMMA * 10.0, rel_tol=1e-6)
    assert math.isclose(float(returns[0]), 1.0 + GAMMA * (2.0 + GAMMA * 10.0), rel_tol=1e-6)
    assert math.isclose(float(returns[2]), 3.0, rel_tol=1e-6)
    # Without tail arrays the same rows are the legacy zero-tail window, named explicitly.
    legacy = {k: v for k, v in data.items() if not k.startswith("tail_")}
    assert return_tail_method(legacy) == RETURN_TAIL_TRUNCATED_ZERO
    assert math.isclose(float(discounted_returns(legacy)["qol_years"][1]), 2.0, rel_tol=1e-6)


def test_living_cutoff_is_never_silently_zero() -> None:
    living_without_tail = _dataset(
        [TAIL_NONE_TERMINAL, TAIL_NONE_TERMINAL, TAIL_NONE_TERMINAL],
        [0.0, 0.0, 0.0],
        [False, False, True],
    )
    with pytest.raises(ValueError, match="living cutoff has no tail"):
        discounted_returns(living_without_tail)
    terminal_with_tail = _dataset(
        [TAIL_RANDOMIZED, TAIL_RANDOMIZED, TAIL_RANDOMIZED], [1.0, 1.0, 1.0], [False, False, True]
    )
    with pytest.raises(ValueError, match="terminated trajectory carries a tail"):
        discounted_returns(terminal_with_tail)
    differing = _dataset(
        [TAIL_RANDOMIZED, TAIL_RANDOMIZED, TAIL_NONE_TERMINAL],
        [1.0, 2.0, 0.0],
        [False, False, True],
    )
    with pytest.raises(ValueError, match="differs within trajectory"):
        discounted_returns(differing)
    declared_without_arrays = {
        k: v for k, v in living_without_tail.items() if not k.startswith("tail_")
    }
    declared_without_arrays["collection"] = {"return_tail": RETURN_TAIL_RANDOMIZED}
    with pytest.raises(ValueError, match="carries no tail arrays"):
        discounted_returns(declared_without_arrays)
    mislabelled = dict(living_without_tail, collection={"return_tail": RETURN_TAIL_TRUNCATED_ZERO})
    with pytest.raises(ValueError, match="carries tail arrays"):
        return_tail_method(mislabelled)


def test_unavailable_tail_rows_are_excluded_and_required_tail_is_enforced() -> None:
    data = _dataset(
        [TAIL_UNAVAILABLE, TAIL_UNAVAILABLE, TAIL_NONE_TERMINAL],
        [0.0, 0.0, 0.0],
        [False, False, True],
    )
    assert critic_target_available(data).tolist() == [False, False, True]
    legacy = {k: v for k, v in data.items() if not k.startswith("tail_")}
    assert critic_target_available(legacy).tolist() == [True, True, True]
    with pytest.raises(ValueError, match="requires randomized-tail targets"):
        labor_bc.pretrain_critic_on_dataset(
            task_critic=None,
            cost_critics=None,
            dataset=legacy,  # type: ignore[arg-type]
            epochs=0,
            require_randomized_tail=True,
        )


def test_settings_are_the_preregistered_defaults_and_round_trip() -> None:
    settings = RandomizedTailSettings()
    assert (settings.draws, settings.rho, settings.cap) == (16, 0.95, 200)
    assert RandomizedTailSettings.from_state(settings.state_dict()) == settings
    assert RandomizedTailSettings.from_state(None) is None
    schedule = settings.schedule(5000)
    assert schedule.rng_seed == settings.seed_base + 5000
    assert schedule.draw_count == 16
    assert schedule.stop_indices == settings.schedule(5000).stop_indices  # frozen, not redrawn
    with pytest.raises(Exception, match="denied"):
        RandomizedTailSettings(seed_base=0).schedule(3000)
    with pytest.raises(ValueError):
        RandomizedTailSettings(draws=1)


def _row(camp: int, turn: int, terminal: bool) -> dict[str, torch.Tensor]:
    return {
        "row_camp": torch.tensor([camp]),
        "row_turn": torch.tensor([turn]),
        "transition_terminal": torch.tensor([terminal]),
    }


def _reward(camp: int, term: int) -> dict[str, float]:
    return {
        "qol_years": 1.0 + 0.01 * camp + 0.1 * term,
        "population_years": 50.0 - term,
        "feasibility": 0.0,
        "settlement_guard": 1.0 if term % 3 == 0 else 0.0,
    }


def test_continuation_closes_each_cutoff_with_the_anchor_estimator(monkeypatch) -> None:
    """Living (A), terminating (B), vanishing (C) and early-leaving (D) cutoffs."""
    settings = RandomizedTailSettings(draws=4, rho=0.95, cap=200, seed_base=9_100_000)
    schedule = settings.schedule(5000)
    required = schedule.required_terms
    calls = {"n": 0}

    def fake_transition(**kwargs):
        term = calls["n"]
        calls["n"] += 1
        decided = [1, 2] + ([3] if term == 0 else [])
        terminal = {1: False, 2: term == 2, 3: False}
        return SimpleNamespace(
            contexts=[(SimpleNamespace(camp_id=camp),) for camp in decided],
            elapsed=dict.fromkeys(decided, contracts.TURN_YEARS),
            terminal=terminal,
            head_rewards=lambda camp, founded, term=term: _reward(camp, term),
            next_step={"turn": 100 + term + 1},
        )

    monkeypatch.setattr(labor_bc, "_teacher_transition", fake_transition)
    result = WorldCollection(world_seed=5000)
    result.rows = [
        _row(1, 24, False),
        _row(2, 24, False),
        _row(3, 24, False),
        _row(4, 20, False),
        _row(5, 24, True),
    ]
    entries = labor_bc._collect_teacher_tail(
        result=result,
        tail=settings,
        labor_bridge=None,
        heur_bridge=None,
        step={"turn": 25},
        options=None,
        normalizer=None,
        personality=None,
        founded={},
        window_turns=25,
    )
    assert calls["n"] == required and result.tail_decisions == required
    assert result.tail_native_resolves == 2 * required
    assert set(entries) == {1, 2, 3, 4}  # camp 5 terminated inside the window
    assert entries[1]["method"] == TAIL_RANDOMIZED
    assert entries[2]["method"] == TAIL_TERMINAL_EXACT
    assert entries[3]["method"] == TAIL_UNAVAILABLE
    assert entries[4]["method"] == TAIL_UNAVAILABLE
    assert entries[4]["reason"] == "camp-left-live-set-before-collection-horizon"
    for head in TAIL_HEADS:
        discounted = [GAMMA**t * _reward(1, t)[head] for t in range(required)]
        if head == "feasibility":
            bound = 1.0
        else:
            bound = 1e12
        expected = estimate_randomized_tail(
            discounted,
            schedule,
            bounded_geometric_tail_assumptions(
                per_turn_absolute_bound=bound,
                discount=GAMMA,
                survival_probability=schedule.survival_probability,
                provenance_id="test",
            ),
        )
        assert math.isclose(
            entries[1]["value"][head], expected.estimate, rel_tol=1e-12, abs_tol=1e-12
        )
        assert math.isclose(
            entries[1]["standard_error"][head],
            expected.standard_error,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        exact = sum(GAMMA**t * _reward(2, t)[head] for t in range(3))
        assert math.isclose(entries[2]["value"][head], exact, rel_tol=1e-12, abs_tol=1e-12)
        assert entries[2]["standard_error"][head] == 0.0
    record = result.tail_record
    assert record["method"] == RETURN_TAIL_RANDOMIZED
    assert record["prefix_terms"] == 25 and record["required_terms"] == required
    assert record["critic_bootstrap_used"] is False and record["dropped_draws"] == 0
    assert record["method_counts"] == {
        "terminal-exact-in-continuation": 1,
        "independent-randomized-tail": 1,
        "unavailable": 2,
    }
    # Broadcast onto rows, then targets start from the tail.
    labor_bc._attach_tail_rows(result.rows, entries)
    assert int(result.rows[0]["tail_method"]) == TAIL_RANDOMIZED
    assert int(result.rows[4]["tail_method"]) == TAIL_NONE_TERMINAL
    assert float(result.rows[0]["tail_value__qol_years"]) == entries[1]["value"]["qol_years"]


def test_hard_cap_schedule_marks_cutoffs_unavailable_without_continuation(monkeypatch) -> None:
    settings = RandomizedTailSettings(draws=16, rho=0.95, cap=1)
    assert settings.schedule(5000).exceeds_hard_cap

    def never(**kwargs):
        raise AssertionError("an unexecutable schedule must not run a continuation")

    monkeypatch.setattr(labor_bc, "_teacher_transition", never)
    result = WorldCollection(world_seed=5000)
    result.rows = [_row(1, 24, False)]
    entries = labor_bc._collect_teacher_tail(
        result=result,
        tail=settings,
        labor_bridge=None,
        heur_bridge=None,
        step={"turn": 25},
        options=None,
        normalizer=None,
        personality=None,
        founded={},
        window_turns=25,
    )
    assert entries[1]["method"] == TAIL_UNAVAILABLE
    assert entries[1]["reason"] == "unresolved-hard-cap"
    assert result.tail_record["exceeds_hard_cap"] is True


# Live native (local bridge) -----------------------------------------------------


@pytest.fixture(scope="module")
def normalizer_path(tmp_path_factory):
    from anthropocene_rl.autonomous_v7.labor_model import BOUNDED_FEATURE_CONTRACT
    from tests.test_labor_bc import _fit_test_normalizer

    path = tmp_path_factory.mktemp("tailnorm") / "norm.pt"
    torch.save(
        {
            "feature_contract": BOUNDED_FEATURE_CONTRACT,
            "normalizer": _fit_test_normalizer().state_dict(),
        },
        path,
    )
    return path


def test_live_teacher_shards_carry_typed_tails_and_feed_pretrain_critic(
    tmp_path, normalizer_path
) -> None:
    from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
    from anthropocene_rl.autonomous_v7.labor_training import main as labor_training_main

    # Small declared settings keep the continuation to two decisions for world 5000.
    tail_args = ["--tail-draws", "2", "--tail-rho", "0.91"]
    assert RandomizedTailSettings(draws=2, rho=0.91).schedule(5000).required_terms == 2
    common = [
        "--normalizer-checkpoint",
        str(normalizer_path),
        "--worlds",
        "5000",
        "--turns",
        "2",
        "--players",
        "8",
        "--jobs",
        "1",
        "--allow-dirty",
    ]
    tailed, legacy = tmp_path / "tailed", tmp_path / "legacy"
    assert ltd.main([*common, *tail_args, "--out", str(tailed)]) == 0
    assert ltd.main([*common, "--return-tail", "truncated-zero", "--out", str(legacy)]) == 0
    with_tail = ltd.load_labor_dataset(tailed)
    without = ltd.load_labor_dataset(legacy)
    assert with_tail["collection"]["return_tail"] == RETURN_TAIL_RANDOMIZED
    assert without["collection"]["return_tail"] == RETURN_TAIL_TRUNCATED_ZERO
    assert not any(k.startswith("tail_") for k in without)
    # The observed window is unchanged by the continuation.
    for key, value in without.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, with_tail[key]), key
    record = with_tail["collection"]["return_tail_by_world"]["5000"]
    assert record["draws"] == 2 and record["prefix_terms"] == 2
    assert record["continuation_native_resolves"] == 2 * record["continuation_decisions"]
    codes = with_tail["tail_method"]
    living = ~with_tail["transition_terminal"]
    last = with_tail["row_turn"] == with_tail["row_turn"].max()
    assert bool((codes[living & last] != TAIL_NONE_TERMINAL).all())
    assert bool((codes[living & last] != TAIL_UNAVAILABLE).all())
    tailed_returns = discounted_returns(with_tail)
    window_returns = discounted_returns(without)
    assert not torch.equal(tailed_returns["population_years"], window_returns["population_years"])

    init = tmp_path / "init.pt"
    labor_training_main(
        [
            "initialize",
            "--normalizer-checkpoint",
            str(normalizer_path),
            "--world-seed",
            "5000",
            "--optimizer-seed",
            "1",
            "--campaign-r5",
            "--out",
            str(init),
        ]
    )
    critic = tmp_path / "critic.pt"
    labor_training_main(
        [
            "pretrain-critic",
            "--checkpoint",
            str(init),
            "--dataset",
            str(tailed),
            "--require-randomized-tail",
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--out",
            str(critic),
        ]
    )
    summary = torch.load(critic, map_location="cpu", weights_only=True)[
        "heuristic_critic_pretraining"
    ]
    assert summary["return_tail"] == RETURN_TAIL_RANDOMIZED
    assert summary["return_tail_settings"]["draws"] == 2
    assert summary["discount_years"] == 400.0
    with pytest.raises(ValueError, match="requires randomized-tail targets"):
        labor_training_main(
            [
                "pretrain-critic",
                "--checkpoint",
                str(init),
                "--dataset",
                str(legacy),
                "--require-randomized-tail",
                "--epochs",
                "1",
                "--out",
                str(tmp_path / "refused.pt"),
            ]
        )
