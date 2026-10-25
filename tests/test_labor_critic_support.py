"""Versioned labor task-critic value support (brief section 17.2)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
from test_labor_model import _layout

from anthropocene_rl.autonomous_v7.labor_critic_support import (
    convert_task_critic_support,
    recommend_support,
    target_audit,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    LABOR_OBJECTIVE_REFERENCE_SCALES,
    LABOR_OBJECTIVES,
    LaborModelContract,
    LaborTaskCritic,
    LaborValueSupport,
    labor_checkpoint_value_support,
)
from anthropocene_rl.autonomous_v7.model import BIN_HIGH, BIN_LOW, BINS, symlog


def test_default_support_is_bitwise_identical_and_not_serialized() -> None:
    assert LaborValueSupport() == LaborValueSupport(BIN_LOW, BIN_HIGH, BINS)
    assert LaborValueSupport().is_default
    assert "criticValueSupport" not in LaborModelContract().state_dict()
    layout = _layout()
    torch.manual_seed(7)
    implicit = LaborTaskCritic(layout, width=16, token_width=16)
    torch.manual_seed(7)
    explicit = LaborTaskCritic(layout, width=16, token_width=16, support=LaborValueSupport())
    a, b = implicit.state_dict(), explicit.state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert torch.equal(implicit.bins, torch.linspace(BIN_LOW, BIN_HIGH, BINS))
    # Historical two-hot arithmetic, reproduced inline.
    values = torch.tensor([0.0, 0.3, -1.0, 20.0, 1e6, -1e6])
    scale = LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"]
    y = symlog(values / scale).clamp(BIN_LOW, BIN_HIGH)
    step = (BIN_HIGH - BIN_LOW) / (BINS - 1)
    position = (y - BIN_LOW) / step
    low = position.floor().long().clamp(0, BINS - 2)
    upper = (position - low.float()).clamp(0, 1)
    expected = torch.zeros(len(values), BINS)
    expected.scatter_(1, low.unsqueeze(1), (1 - upper).unsqueeze(1))
    expected.scatter_add_(1, (low + 1).unsqueeze(1), upper.unsqueeze(1))
    assert torch.equal(explicit.two_hot("qol_years", values), expected)


def test_support_contract_round_trip_and_rejection() -> None:
    narrow = LaborValueSupport(-1.0, 4.0, 255)
    state = LaborModelContract(value_support=narrow).state_dict()
    assert state["criticValueSupport"]["version"] == "labor-task-symlog-two-hot-support-v2"
    assert LaborModelContract.from_state(state).value_support == narrow
    default_declared = copy.deepcopy(state)
    default_declared["criticValueSupport"].update(low=BIN_LOW, high=BIN_HIGH, bins=BINS)
    with pytest.raises(ValueError, match="implicit"):
        LaborModelContract.from_state(default_declared)
    bad = copy.deepcopy(state)
    bad["criticValueSupport"]["version"] = "v9"
    with pytest.raises(ValueError):
        LaborModelContract.from_state(bad)
    for args in ((0.0, 4.0, 255), (-1.0, 4.0, 2), (-1.0, math.inf, 255)):
        with pytest.raises(ValueError):
            LaborValueSupport(*args)
    assert LaborValueSupport.parse("-1,4,101") == LaborValueSupport(-1.0, 4.0, 101)
    assert LaborValueSupport.parse(None).is_default


def _decoded_error(support: LaborValueSupport, value: float, stray: float) -> float:
    critic = LaborTaskCritic(_layout(), width=8, token_width=8, support=support)
    target = critic.two_hot("qol_years", torch.tensor([value]))[0].double()
    target = target * (1.0 - stray)
    target[-1] += stray
    logits = target.clamp_min(1e-300).log().float().unsqueeze(0)
    decoded = float(critic.expectation_normalized(logits)[0])
    return abs(decoded - value / LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"])


def test_narrow_support_reduces_decoded_error_from_top_bin_mass() -> None:
    value = 1.0 * LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"]  # normalized target 1
    wide = _decoded_error(LaborValueSupport(), value, 1e-4)
    narrow = _decoded_error(LaborValueSupport(-1.0, 4.0, 255), value, 1e-4)
    assert wide == pytest.approx(1e-4 * (math.expm1(9.0) - 1.0), rel=0.02)  # ~0.81 (brief 17.2)
    assert narrow == pytest.approx(1e-4 * (math.expm1(4.0) - 1.0), rel=0.05)  # ~0.0054
    assert narrow < wide / 100.0
    critic = LaborTaskCritic(_layout(), width=8, token_width=8)
    logits = torch.full((1, BINS), -30.0)
    logits[0, -1] = 0.0
    terms = critic.support_diagnostics("qol_years", logits, 1.0)
    assert float(terms["top_bin_mass"][0]) == pytest.approx(1.0)
    assert float(terms["upper_excess"][0]) == pytest.approx(math.expm1(9.0) - 1.0, rel=1e-6)


def _fixture():
    from test_autonomous_v7_round3_search import observation as legacy_observation
    from test_labor_bridge import _catalog_payload

    from anthropocene_rl.autonomous_v7.bridge import observation_batch
    from anthropocene_rl.autonomous_v7.labor_bridge import (
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
        LABOR_SPECIALTY_FEATURE_WIDTH,
    )
    from anthropocene_rl.autonomous_v7.labor_model import LaborEncoderLayout, labor_policy_tensors
    from anthropocene_rl.autonomous_v7.model import Normalizer

    raw = legacy_observation(seed=92)
    raw["core"][40:56] = [0.0] * 16
    raw["core"][144] = 0.0
    raw.update(laborCatalog=_catalog_payload(), laborPrevious=None)
    village = {"observation": raw}
    normalizer = Normalizer.fit_bounded(observation_batch([village]))
    _, tensors = labor_policy_tensors([village], normalizer)
    layout = LaborEncoderLayout(
        tensors["core"].shape[1],
        LABOR_SPECIALTY_FEATURE_WIDTH,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
    )
    return normalizer, layout, tensors


def _trainer_state(support: LaborValueSupport | None = None) -> tuple[dict, dict]:
    from anthropocene_rl.autonomous_v7.labor_model import (
        LaborActor,
        LaborCostCritics,
        LaborPersonality,
    )
    from anthropocene_rl.autonomous_v7.labor_training import _trainer_checkpoint

    normalizer, layout, tensors = _fixture()
    torch.manual_seed(3)
    actor = LaborActor(layout)
    task = LaborTaskCritic(layout, support=support)
    costs = LaborCostCritics(layout)
    state = _trainer_checkpoint(
        actor,
        task,
        costs,
        normalizer,
        layout,
        torch.optim.Adam(actor.parameters(), lr=3e-4),
        torch.optim.Adam(list(task.parameters()) + list(costs.parameters()), lr=2e-4),
        personality=LaborPersonality(),
        optimizer_seed=11,
        iteration=4,
        sampling_generator=torch.Generator().manual_seed(11),
        shuffle_rng=np.random.default_rng(11),
        critic_training_world_seeds=(90100,),
        critic_training_independent_unit_ids=(),
    )
    state["heuristic_critic_pretraining"] = {"epochs": 2}
    return state, tensors


def test_default_checkpoint_loads_bitwise() -> None:
    from anthropocene_rl.autonomous_v7.labor_training import _models_from_checkpoint

    state, _ = _trainer_state()
    assert "criticValueSupport" not in state["labor_model_contract"]
    assert labor_checkpoint_value_support(state).is_default
    _, _, task, _, _ = _models_from_checkpoint(state)
    assert task.support.is_default
    for key, value in task.state_dict().items():
        assert torch.equal(value, state["task_critic"][key])


def test_narrow_checkpoint_declares_and_requires_its_support() -> None:
    from anthropocene_rl.autonomous_v7.labor_model import (
        LaborActor,
        LaborCostCritics,
        load_labor_checkpoint_models,
    )
    from anthropocene_rl.autonomous_v7.labor_training import _models_from_checkpoint

    narrow = LaborValueSupport(-1.0, 4.0, 101)
    state, tensors = _trainer_state(narrow)
    assert state["labor_model_contract"]["criticValueSupport"]["bins"] == 101
    layout, _, task, _, _ = _models_from_checkpoint(state)
    assert task.support == narrow
    task.eval()
    assert task(tensors)["qol_years"].shape[-1] == 101
    with pytest.raises(ValueError, match="value support"):
        load_labor_checkpoint_models(
            state,
            LaborActor(layout),
            LaborTaskCritic(layout),
            LaborCostCritics(layout),
            expected_layout=layout,
        )
    same_count = LaborValueSupport(-1.0, 5.0, 101)
    tampered = copy.deepcopy(state)
    tampered["labor_model_contract"]["criticValueSupport"]["high"] = 5.0
    with pytest.raises(ValueError, match="bins differ"):
        load_labor_checkpoint_models(
            tampered,
            LaborActor(layout),
            LaborTaskCritic(layout, support=same_count),
            LaborCostCritics(layout),
            expected_layout=layout,
        )


def test_converter_reinitializes_only_task_heads() -> None:
    from anthropocene_rl.autonomous_v7.labor_training import _models_from_checkpoint

    state, tensors = _trainer_state()
    narrow = LaborValueSupport(-1.0, 4.0, 255)
    converted = convert_task_critic_support(state, narrow, seed=5)
    assert labor_checkpoint_value_support(converted) == narrow
    _, _, task, _, _ = _models_from_checkpoint(converted)
    for key, value in state["task_critic"].items():
        if key.startswith("encoder."):
            assert torch.equal(task.state_dict()[key], value), key
    for name in LABOR_OBJECTIVES:
        assert not torch.equal(
            task.state_dict()[f"heads.{name}.weight"], state["task_critic"][f"heads.{name}.weight"]
        )
    assert torch.equal(task.bins, narrow.bin_values())
    for section in ("actor", "cost_critics"):
        for key, value in state[section].items():
            assert torch.equal(converted[section][key], value), (section, key)
    assert converted["normalizer"].keys() == state["normalizer"].keys()
    for key in ("iteration", "optimizer_seed", "critic_training_world_seeds", "shuffle_rng_state"):
        assert converted[key] == state[key]
    assert torch.equal(converted["sampling_generator_state"], state["sampling_generator_state"])
    assert converted["critic_optimizer"]["state"] == {}
    assert converted["critic_optimizer"]["param_groups"][0]["lr"] == 2e-4
    assert converted["actor_optimizer"]["param_groups"][0]["lr"] == 3e-4
    assert converted["heuristic_critic_pretraining"] == {"epochs": 2}
    record = converted["critic_value_support_conversions"][-1]
    assert record["from"]["default"] and record["to"]["high"] == 4.0
    again = convert_task_critic_support(state, narrow, seed=5)
    assert all(torch.equal(again["task_critic"][k], v) for k, v in converted["task_critic"].items())
    with pytest.raises(ValueError, match="already"):
        convert_task_critic_support(converted, narrow)
    task.eval()
    with torch.no_grad():
        assert torch.isfinite(task.expectation("qol_years", task(tensors)["qol_years"])).all()


def test_target_audit_and_recommendation() -> None:
    q = LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"]
    p = LABOR_OBJECTIVE_REFERENCE_SCALES["population_years"]
    n = 4
    dataset = {
        "row_world": torch.tensor([1, 1, 2, 2]),
        "row_camp": torch.tensor([0, 0, 0, 0]),
        "row_turn": torch.tensor([0, 1, 0, 1]),
        "row_quarantined": torch.zeros(n, dtype=torch.bool),
        "transition_elapsed_years": torch.zeros(n),
        "transition_terminal": torch.zeros(n, dtype=torch.bool),
        "reward_qol_years": torch.tensor([q, q, 0.0, 2 * q]),
        "reward_population_years": torch.tensor([p, 0.0, 0.0, 0.0]),
        "reward_feasibility": torch.zeros(n),
        "reward_settlement_guard": torch.zeros(n),
    }
    report = target_audit(dataset, candidates=((-2.0, 9.0), (-0.5, 1.0)))
    head = report["task_heads"]["qol_years"]
    # Zero elapsed days -> gamma 1: returns are suffix sums 2q, q, 2q, 2q.
    assert head["normalized"]["max"] == pytest.approx(2.0)
    assert head["symlog_normalized"]["max"] == pytest.approx(math.log(3.0))
    assert head["clipping"]["-0.5,1"]["above_fraction"] == pytest.approx(0.75)
    assert head["clipping"]["-2,9"]["above_fraction"] == 0.0
    rec = recommend_support(report["task_heads"], margin=0.5)
    assert rec["high"] == 2.0 and rec["low"] == -0.5 and rec["bins"] == 255


def test_campaign_r5_flag_declares_named_support_and_default_is_unchanged() -> None:
    import argparse

    from anthropocene_rl.autonomous_v7.labor_model import LABOR_CAMPAIGN_R5_VALUE_SUPPORT
    from anthropocene_rl.autonomous_v7.labor_training import (
        _initialize_value_support,
        argument_parser,
    )

    assert LABOR_CAMPAIGN_R5_VALUE_SUPPORT == LaborValueSupport(-0.5, 2.5, 255)
    assert not LABOR_CAMPAIGN_R5_VALUE_SUPPORT.is_default
    base = ["initialize", "--normalizer-checkpoint", "n.pt", "--world-seed", "5000",
            "--optimizer-seed", "1", "--out", "o.pt"]
    default = argument_parser().parse_args(base)
    assert default.campaign_r5 is False
    assert _initialize_value_support(default).is_default
    campaign = argument_parser().parse_args([*base, "--campaign-r5"])
    assert _initialize_value_support(campaign) == LABOR_CAMPAIGN_R5_VALUE_SUPPORT
    agreeing = argument_parser().parse_args(
        [*base, "--campaign-r5", "--critic-value-support=-0.5,2.5,255"]
    )
    assert _initialize_value_support(agreeing) == LABOR_CAMPAIGN_R5_VALUE_SUPPORT
    contradicting = argparse.Namespace(campaign_r5=True, critic_value_support="-1,4,255")
    with pytest.raises(ValueError, match="contradicts"):
        _initialize_value_support(contradicting)
    state = LaborModelContract(value_support=LABOR_CAMPAIGN_R5_VALUE_SUPPORT).state_dict()
    assert state["criticValueSupport"]["low"] == -0.5
    assert state["criticValueSupport"]["high"] == 2.5
    assert LaborModelContract.from_state(state).value_support == LABOR_CAMPAIGN_R5_VALUE_SUPPORT
