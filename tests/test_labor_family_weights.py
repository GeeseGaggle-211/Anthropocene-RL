"""Family weight decay schedule, the food-trade guard family and its advisory flag."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from test_labor_privileged_model import _layout
from test_labor_qol_head import _training_dataset
from test_labor_scenarios import FARMER, _synthetic_arrays

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7 import labor_r5_acceptance as acc
from anthropocene_rl.autonomous_v7 import labor_scenarios as ls
from anthropocene_rl.autonomous_v7.labor_bc import pretrain_critic_on_dataset
from anthropocene_rl.autonomous_v7.labor_model import (
    LaborCostCritics,
    LaborQoLHead,
    LaborTaskCritic,
)
from anthropocene_rl.autonomous_v7.labor_stability import StabilityConfig

# Decay schedule ------------------------------------------------------------------------------


def test_decay_curves_and_multipliers() -> None:
    assert ls.decay_factor("none", 5.0, 4.0) == 1.0
    assert [ls.decay_factor("linear", t, 4.0) for t in (0, 1, 2, 4, 9)] == [1, 0.75, 0.5, 0, 0]
    cosine = [ls.decay_factor("cosine", t, 4.0) for t in (0, 2, 4, 6)]
    assert cosine[0] == 1.0 and cosine[1] == pytest.approx(0.5) and cosine[2:] == [0.0, 0.0]
    assert ls.decay_factor("cosine", 1.0, 4.0) == pytest.approx(0.5 * (1 + math.cos(math.pi / 4)))
    schedule = ls.FamilyWeightSchedule({"first-use-farming": 10.0, "mixed": 5.0}, "linear", 4.0)
    assert schedule.multipliers(0.0) == {"first-use-farming": 10.0, "mixed": 5.0}
    assert schedule.multipliers(2.0) == {"first-use-farming": 5.5, "mixed": 3.0}
    assert schedule.multipliers(4.0) == {"first-use-farming": 1.0, "mixed": 1.0}
    with pytest.raises(ValueError):
        ls.decay_factor("step", 0.0, 1.0)


def test_schedule_continues_across_rounds_and_restarts_only_on_request() -> None:
    weights = {"mixed": 5.0}
    first = ls.continued_schedule(weights, None, None, 4, None)
    assert first.decay == "cosine" and first.decay_epochs == 4 and first.epoch_offset == 0
    record = {**first.state_dict(), "epochs_end": 4.0}
    second = ls.continued_schedule(weights, None, None, 2, record)
    assert second.epoch_offset == 4.0 and second.decay_epochs == 4.0  # stays decayed
    assert second.factor(0.0) == 0.0
    restarted = ls.continued_schedule(weights, None, None, 2, record, restart=True)
    assert restarted.epoch_offset == 0.0 and restarted.decay_epochs == 2
    changed = ls.continued_schedule({"mixed": 3.0}, None, None, 2, record)
    assert changed.epoch_offset == 0.0  # different weights: a new schedule
    assert ls.continued_schedule(None, None, None, 2, record) is None


def test_closed_form_weighter_matches_explicit_renormalization() -> None:
    rng = np.random.default_rng(0)
    base = rng.random(40) + 0.1
    tags = rng.integers(0, 1 << len(ls.FAMILIES), size=40)
    sources = rng.integers(0, 2, size=40)
    worlds = 90100 + rng.integers(0, 3, size=40)
    weights = ls.PROPOSED_FAMILY_WEIGHTS
    weighter = ls.FamilyRowWeighter(base, tags, sources, worlds, weights)
    for s in (1.0, 0.6, 0.0):
        decayed = {k: 1.0 + (w - 1.0) * s for k, w in weights.items()}
        expected = ls.apply_family_weights(base, tags, sources, worlds, decayed)
        assert np.allclose(weighter.weights(s), expected)
        assert np.allclose(weighter.rows([3, 7, 11], s), expected[[3, 7, 11]])
    assert np.allclose(weighter.weights(0.0), base)


def _models():
    torch.manual_seed(0)
    return (
        LaborTaskCritic(
            _layout(False), width=16, token_width=16, qol_head=LaborQoLHead(target="td-lambda")
        ),
        LaborCostCritics(_layout(False), width=16, token_width=16),
    )


def _weighted_dataset():
    data = _training_dataset()
    data["collection"] = dict(
        data["collection"],
        dataset_digest="fixture",
        family_weighting={"weights": {"mixed": 9.0}, "status": "proposed"},
    )
    data["row_family_tags"] = torch.tensor([ls.FAMILY_BITS["mixed"], 0, 0, 0, 0, 0])
    return data


def test_decayed_critic_fit_resumes_exactly_and_logs_multipliers(tmp_path) -> None:
    data = _weighted_dataset()
    schedule = ls.FamilyWeightSchedule({"mixed": 9.0}, "cosine", 3.0)
    log = tmp_path / "stability.jsonl"
    task, costs = _models()
    summary = pretrain_critic_on_dataset(
        task_critic=task, cost_critics=costs, dataset=data, epochs=3, batch_size=2,
        rng=np.random.default_rng(7), family_schedule=schedule,
        stability=StabilityConfig(log_every=1, jsonl=str(log)),
    )
    reference = task.state_dict()
    record = summary["family_weighting"]["schedule"]
    assert record["epochs_end"] == pytest.approx(3.0)
    assert record["final_multipliers"]["mixed"] == pytest.approx(1.0)
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    logged = [r["family_weight_multipliers"]["mixed"] for r in lines if "family_weight_factor" in r]
    assert logged[0] == pytest.approx(9.0) and logged == sorted(logged, reverse=True)
    assert logged[-1] < 2.0
    path = str(tmp_path / "resume.pt")
    task_a, costs_a = _models()
    pretrain_critic_on_dataset(
        task_critic=task_a, cost_critics=costs_a, dataset=data, epochs=3, batch_size=2,
        rng=np.random.default_rng(7), family_schedule=schedule, max_steps=4,
        resume_path=path, resume_every=2, stability=StabilityConfig(log_every=1),
    )
    task_b, costs_b = _models()
    pretrain_critic_on_dataset(
        task_critic=task_b, cost_critics=costs_b, dataset=data, epochs=3, batch_size=2,
        rng=np.random.default_rng(7), family_schedule=schedule, resume_path=path,
        resume_every=2, stability=StabilityConfig(log_every=1),
    )
    assert all(torch.equal(reference[k], v) for k, v in task_b.state_dict().items())
    # A resume state of another schedule is refused.
    with pytest.raises(ValueError, match="different critic fit"):
        pretrain_critic_on_dataset(
            task_critic=task_b, cost_critics=costs_b, dataset=data, epochs=3, batch_size=2,
            family_schedule=ls.FamilyWeightSchedule({"mixed": 9.0}, "linear", 3.0),
            resume_path=path, stability=StabilityConfig(log_every=1),
        )
    # Without decay the fixed-weight path is unchanged by the schedule plumbing.
    task_c, costs_c = _models()
    task_d, costs_d = _models()
    for t, c, sched in ((task_c, costs_c, None),
                        (task_d, costs_d, ls.FamilyWeightSchedule({"mixed": 9.0}, "none", 3.0))):
        pretrain_critic_on_dataset(
            task_critic=t, cost_critics=c, dataset=data, epochs=3, batch_size=2,
            rng=np.random.default_rng(7), family_schedule=sched,
        )
    assert all(torch.equal(v, task_d.state_dict()[k]) for k, v in task_c.state_dict().items())


# Food-trade guard ------------------------------------------------------------------------


def _with_trade(arrays, imported, meals):
    n = len(arrays["row_world"])
    names = [name for name, _, _ in contracts.SETS]
    _, rows, width = contracts.SETS[names.index("trade")]
    food = contracts.TRADE_GROUPS.index("food")
    raw = np.zeros((n, rows, width))
    raw[:, food, 6] = imported
    economy = ls._feature_columns()["meals"]
    arrays["privileged_economy"][:, 0, economy] = meals
    arrays["trade_raw"] = raw.reshape(n, -1)
    present = np.zeros((n, rows), dtype=bool)
    present[:, food] = True
    arrays["trade_present"] = present
    return arrays


class _Normalizer:
    def __init__(self, dead=False):
        width = 10
        self.set_mean = {"trade": np.full(width, 0.5, np.float32)}
        self.set_std = {"trade": np.full(width, 2.0, np.float32)}
        self.dead = {"trade": np.full(width, dead)}


def test_food_trade_tagging_from_raw_and_normalized_trade_rows() -> None:
    rows = [(90100, 8, c, FARMER, 300.0, 1.0, 1.0, 1.0, 1) for c in (1, 2, 3)]
    arrays = _with_trade(_synthetic_arrays(rows), np.asarray([30.0, 10.0, 0.0]), 100.0)
    tags = [set(ls.family_names(int(t))) for t in ls.row_family_tags(arrays)]
    assert ["food-trade" in t for t in tags] == [True, False, False]  # share 0.3, 0.1, 0
    # Stored rows: the bounded-normalized trade set is inverted with the dataset normalizer.
    norm = _Normalizer()
    raw = arrays.pop("trade_raw")
    y = np.sign(raw) * np.log1p(np.abs(raw))
    arrays["trade"] = (y - 0.5) / 2.0
    tags = [set(ls.family_names(int(t))) for t in ls.row_family_tags(arrays, normalizer=norm)]
    assert ["food-trade" in t for t in tags] == [True, False, False]
    assert ls.food_trade_tagging(arrays, norm).startswith("available")
    dead = _Normalizer(dead=True)
    assert ls.food_trade_tagging(arrays, dead).startswith("unavailable")
    assert not any("food-trade" in ls.family_names(int(t))
                   for t in ls.row_family_tags(arrays, normalizer=dead))
    assert not any("food-trade" in ls.family_names(int(t)) for t in ls.row_family_tags(arrays))
    # A guard family: never up-weighted.
    with pytest.raises(ValueError, match="never up-weighted"):
        ls.parse_family_weights("default,food-trade=2")
    assert "food-trade" not in ls.PROPOSED_FAMILY_WEIGHTS


def _evidence(ft_nmae, ft_rank, a_ft, weights=True):
    heads = {"qol_years": {"all": {"nmae": 0.10, "ranking": {"accuracy": 0.80}}}}
    family = {"food-trade": {"qol_years": {"nmae": ft_nmae, "ranking": {"accuracy": ft_rank}}}}
    return {
        "family_weighting": {"weights": {"mixed": 5.0}} if weights else None,
        "calibration": {"calibration": {"heads": heads, "by_scenario_family": family}},
        "advantage_eval": {
            "ranking": {"qol_years": {"accuracy": 0.75}},
            "by_scenario_family": {"food-trade": {"ranking": {"qol_years": {"accuracy": a_ft}}}},
        },
    }


def test_food_trade_advisory_flags_without_gating() -> None:
    ok = acc.food_trade_advisory(_evidence(0.14, 0.77, 0.72))
    assert ok["status"] == "OK" and ok["gating"] is False
    worse_v = acc.food_trade_advisory(_evidence(0.16, 0.80, 0.75))
    assert worse_v["status"] == "FLAG"
    assert [c["check"] for c in worse_v["comparisons"] if c["flag"]] == ["V:qol_years:nmae"]
    worse_a = acc.food_trade_advisory(_evidence(0.10, 0.80, 0.69))
    assert [c["check"] for c in worse_a["comparisons"] if c["flag"]] == ["A:qol_years:ranking"]
    assert acc.food_trade_advisory(_evidence(0.5, 0.1, 0.1, weights=False))["status"] == (
        "NOT-APPLICABLE"
    )
    # Advisory never enters the primary/secondary gate statuses.
    checks = acc.fit_checks(_evidence(0.5, 0.1, 0.1))
    assert checks["advisory"][0]["status"] == "FLAG"
    assert all(not c["id"].startswith("advisory") for c in checks["primary"])
    summary = acc.scenario_family_summary(_evidence(0.2, 0.7, 0.6))
    assert summary["food-trade"]["V"]["qol_years"]["nmae"] == 0.2
    assert summary["food-trade"]["A"]["qol_years"] == 0.6
