"""Bounded QoL value head (G0 amendment r5-b item 4) and old-checkpoint bit identity.

The head predicts discounted-mean QoL m in [0, 1] (G = m * W), replaces the
symlog two-hot for QoL only, is a versioned contract field, and leaves every
checkpoint without that field byte-for-byte and output-for-output unchanged.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_labor_model import _labor_tensors, _layout
from test_labor_return_tail import _dataset as _tail_dataset

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7.labor_advantage import (
    qol_saturation_report,
    qol_td_lambda_targets,
)
from anthropocene_rl.autonomous_v7.labor_bc import (
    TAIL_NONE_TERMINAL,
    TAIL_RANDOMIZED,
    discounted_returns,
    pretrain_critic_on_dataset,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    LABOR_QOL_HEAD_VERSION,
    LaborActor,
    LaborCostCritics,
    LaborModelContract,
    LaborQoLHead,
    LaborTaskCritic,
    labor_checkpoint_payload,
    labor_checkpoint_qol_head,
    labor_qol_discount_weight,
    load_labor_checkpoint_models,
)
from anthropocene_rl.autonomous_v7.labor_ppo import labor_task_value_loss
from anthropocene_rl.autonomous_v7.model import Normalizer

BASE_COMMIT = "6622109"  # feature/r5-privileged-critic, before the bounded head
REPOSITORY = Path(__file__).resolve().parents[1]
W = labor_qol_discount_weight()


def _base_module():
    try:
        source = subprocess.run(
            ["git", "show", f"{BASE_COMMIT}:src/anthropocene_rl/autonomous_v7/labor_model.py"],
            cwd=REPOSITORY,
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("base labor_model source is not available in this checkout")
    import tempfile

    path = Path(tempfile.mkdtemp()) / "labor_model_r5a.py"
    path.write_text(source, encoding="utf-8")
    name = "anthropocene_rl.autonomous_v7._labor_model_r5a"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _models(module, seed: int = 5, **task_kwargs):
    torch.manual_seed(seed)
    layout = module.LaborEncoderLayout(*_layout().state_dict().values())
    return (
        module.LaborActor(layout, width=24, token_width=16),
        module.LaborTaskCritic(layout, width=24, token_width=16, **task_kwargs),
        module.LaborCostCritics(layout, width=20, token_width=16),
    )


def _outputs(actor, task, costs, tensors):
    with torch.no_grad():
        out = {f"actor.{k}": v for k, v in actor(tensors).items()}
        out.update({f"task.{k}": v for k, v in task(tensors).items()})
        out.update({f"cost.{k}": v for k, v in costs(tensors).items()})
        out.update({f"value.{k}": task.expectation(k, v) for k, v in task(tensors).items()})
    return out


def _normalizer():
    from test_labor_privileged_model import _village

    from anthropocene_rl.autonomous_v7.bridge import observation_batch

    return Normalizer.fit_bounded(observation_batch([_village(1)]))


def test_default_models_are_bitwise_the_base_commit_models() -> None:
    import anthropocene_rl.autonomous_v7.labor_model as current

    old = _base_module()
    old_models, new_models = _models(old), _models(current)
    for old_module, new_module in zip(old_models, new_models, strict=True):
        assert old_module.state_dict().keys() == new_module.state_dict().keys()
        for name, value in old_module.state_dict().items():
            assert torch.equal(new_module.state_dict()[name], value), name
    tensors = _labor_tensors()
    left, right = _outputs(*old_models, tensors), _outputs(*new_models, tensors)
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_old_checkpoint_contract_and_payload_are_unchanged_and_load() -> None:
    import anthropocene_rl.autonomous_v7.labor_model as current

    old = _base_module()
    normalizer = _normalizer()
    layout = current.LaborEncoderLayout(*_layout().state_dict().values())
    old_layout = old.LaborEncoderLayout(*_layout().state_dict().values())
    old_payload = old.labor_checkpoint_payload(*_models(old), normalizer, old_layout)
    new_payload = labor_checkpoint_payload(*_models(current), normalizer, layout)
    assert old_payload["labor_model_contract"] == new_payload["labor_model_contract"]
    assert "criticQoLHead" not in new_payload["labor_model_contract"]
    assert labor_checkpoint_qol_head(old_payload) is None
    for key in ("actor", "task_critic", "cost_critics"):
        assert old_payload[key].keys() == new_payload[key].keys()
        for name, value in old_payload[key].items():
            assert torch.equal(new_payload[key][name], value)
    # An old checkpoint loads into the current code and decodes identically.
    restored = _models(current, seed=77)
    load_labor_checkpoint_models(old_payload, *restored, expected_layout=layout)
    tensors = _labor_tensors()
    left, right = _outputs(*_models(old), tensors), _outputs(*restored, tensors)
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_bounded_head_contract_round_trips_and_is_strict() -> None:
    head = LaborQoLHead()
    state = head.state_dict()
    assert state["version"] == LABOR_QOL_HEAD_VERSION
    assert (state["output"], state["target"], state["tdLambda"]) == ("logistic", "monte-carlo", 0.8)
    assert LaborQoLHead.from_state(state) == head
    assert math.isclose(W, 20.0 / (1.0 - math.exp(-20.0 / 400.0)))
    assert math.isclose(W, 410.08, abs_tol=0.01)
    assert LaborQoLHead.parse(None) is None and LaborQoLHead.parse("two-hot") is None
    assert LaborQoLHead.parse("silu-clamp") == LaborQoLHead("silu-clamp")
    parsed = LaborQoLHead.parse("logistic,td-lambda,0.7")
    assert parsed == LaborQoLHead("logistic", "td-lambda", 0.7)
    with pytest.raises(ValueError):
        LaborQoLHead("tanh")
    with pytest.raises(ValueError):
        LaborQoLHead(td_lambda=1.5)
    tampered = dict(state, saturationThreshold=5.0)
    with pytest.raises(ValueError, match="canonical"):
        LaborQoLHead.from_state(tampered)
    contract = LaborModelContract(qol_head=head)
    assert contract.state_dict()["criticQoLHead"] == state
    assert LaborModelContract.from_state(contract.state_dict()) == contract


def test_bounded_checkpoint_round_trip_and_mismatch_rejection() -> None:
    import anthropocene_rl.autonomous_v7.labor_model as current

    layout = current.LaborEncoderLayout(*_layout().state_dict().values())
    normalizer = _normalizer()
    head = LaborQoLHead("silu-clamp")
    models = _models(current, qol_head=head)
    payload = labor_checkpoint_payload(*models, normalizer, layout)
    assert labor_checkpoint_qol_head(payload) == head
    assert payload["task_critic"]["heads.qol_years.weight"].shape == (1, 24)
    assert payload["task_critic"]["heads.population_years.weight"].shape == (255, 24)
    restored = _models(current, seed=8, qol_head=head)
    load_labor_checkpoint_models(payload, *restored, expected_layout=layout)
    tensors = _labor_tensors()
    left, right = _outputs(*models, tensors), _outputs(*restored, tensors)
    for key in left:
        assert torch.equal(left[key], right[key]), key
    with pytest.raises(ValueError, match="different QoL head"):
        load_labor_checkpoint_models(payload, *_models(current), expected_layout=layout)
    with pytest.raises(ValueError, match="different QoL head"):
        load_labor_checkpoint_models(
            payload, *_models(current, qol_head=LaborQoLHead()), expected_layout=layout
        )


def test_bounded_head_keeps_every_other_initial_parameter() -> None:
    torch.manual_seed(4)
    plain = LaborTaskCritic(_layout(), width=24, token_width=16)
    after_plain = torch.rand(3)
    torch.manual_seed(4)
    bounded = LaborTaskCritic(_layout(), width=24, token_width=16, qol_head=LaborQoLHead())
    after_bounded = torch.rand(3)
    plain_state, bounded_state = plain.state_dict(), bounded.state_dict()
    for name, value in plain_state.items():
        if not name.startswith("heads.qol_years"):
            assert torch.equal(bounded_state[name], value), name
    # Later modules (cost critics, the actor) see the same global RNG stream too.
    assert torch.equal(after_plain, after_bounded)


def test_bounded_decode_losses_and_population_head_are_distributional() -> None:
    critic = LaborTaskCritic(_layout(), width=24, token_width=16, qol_head=LaborQoLHead())
    tensors = _labor_tensors()
    out = critic(tensors)
    assert out["qol_years"].shape == (3, 1) and out["population_years"].shape == (3, 255)
    value = critic.expectation("qol_years", out["qol_years"])
    assert ((value >= 0) & (value <= W)).all()
    assert critic.is_bounded("qol_years") and not critic.is_bounded("population_years")
    target = torch.tensor([0.0, 0.5 * W, 2.0 * W])
    loss = critic.value_loss("qol_years", out["qol_years"], target)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        out["qol_years"].squeeze(-1), torch.tensor([0.0, 0.5, 1.0]), reduction="none"
    )
    assert torch.allclose(loss, expected)
    with pytest.raises(ValueError, match="bounded QoL head"):
        critic.two_hot("qol_years", target)
    with pytest.raises(ValueError, match="bounded QoL head"):
        critic.support_diagnostics("qol_years", out["qol_years"], 1.0)
    pop = torch.tensor([10.0, 20.0, 30.0])
    assert torch.equal(
        critic.value_loss("population_years", out["population_years"], pop),
        -(critic.two_hot("population_years", pop) * torch.log_softmax(out["population_years"], -1))
        .sum(-1),
    )
    total = labor_task_value_loss(critic, tensors, {"qol_years": target, "population_years": pop})
    total.backward()
    assert critic.heads["qol_years"].weight.grad.abs().sum() > 0


def test_silu_clamp_fallback_decodes_in_unit_interval_and_keeps_gradients() -> None:
    critic = LaborTaskCritic(
        _layout(), width=24, token_width=16, qol_head=LaborQoLHead("silu-clamp")
    )
    z = torch.tensor([[-8.0], [0.5], [9.0]], requires_grad=True)
    m = critic.bounded_mean(z)
    assert m[0] >= 0 and m[2] == 1.0
    assert torch.isclose(m[1], torch.nn.functional.silu(torch.tensor(0.5)))
    loss = critic.value_loss("qol_years", z, torch.tensor([0.0, 0.3 * W, 1.0 * W])).sum()
    loss.backward()
    assert z.grad[2].abs() > 0  # the unclamped SiLU keeps a gradient past m = 1
    logistic = LaborTaskCritic(_layout(), width=24, token_width=16, qol_head=LaborQoLHead())
    assert logistic.qol_saturation(torch.tensor([[-7.0], [0.0], [6.5], [1.0]])) == 0.5
    report = qol_saturation_report(logistic, torch.tensor([[-7.0], [0.0], [6.5], [1.0]]))
    assert report["share_abs_z_gt_6"] == 0.5 and report["rows"] == 4


def _training_dataset(rows: int = 6) -> dict:
    tensors = _labor_tensors(batch=rows)
    dataset = dict(tensors)
    dataset.update(
        row_world=torch.tensor([90100] * 3 + [90101] * 3),
        row_camp=torch.tensor([1, 1, 1, 2, 2, 2]),
        row_turn=torch.tensor([0, 1, 2, 0, 1, 2]),
        row_quarantined=torch.zeros(rows, dtype=torch.bool),
        reward_qol_years=torch.full((rows,), 12.0, dtype=torch.float64),
        reward_population_years=torch.ones(rows, dtype=torch.float64),
        reward_feasibility=torch.zeros(rows, dtype=torch.float64),
        reward_settlement_guard=torch.zeros(rows, dtype=torch.float64),
        transition_elapsed_years=torch.full((rows,), 20.0, dtype=torch.float64),
        transition_terminal=torch.tensor([False, False, True] * 2),
    )
    dataset["collection"] = {"return_tail": "truncated-zero-at-collection-horizon"}
    return dataset


@pytest.mark.parametrize("target", ["monte-carlo", "td-lambda"])
def test_fit_reports_the_saturation_monitor(target: str) -> None:
    torch.manual_seed(0)
    task = LaborTaskCritic(
        _layout(), width=16, token_width=16, qol_head=LaborQoLHead(target=target)
    )
    costs = LaborCostCritics(_layout(), width=16, token_width=16)
    summary = pretrain_critic_on_dataset(
        task_critic=task, cost_critics=costs, dataset=_training_dataset(), epochs=2, batch_size=4
    )
    assert summary["qol_head"]["target"] == target
    for key in ("qol_saturation_initial", "qol_saturation_final"):
        assert 0.0 <= summary[key]["share_abs_z_gt_6"] <= 1.0
        assert summary[key]["rows"] == 6
    plain = LaborTaskCritic(_layout(), width=16, token_width=16)
    plain_summary = pretrain_critic_on_dataset(
        task_critic=plain, cost_critics=costs, dataset=_training_dataset(), epochs=1, batch_size=4
    )
    assert "qol_saturation_final" not in plain_summary


def test_td_lambda_targets_reduce_to_monte_carlo_and_bootstrap_through_the_head() -> None:
    data = _tail_dataset(
        [TAIL_RANDOMIZED, TAIL_RANDOMIZED, TAIL_NONE_TERMINAL], [10.0, 10.0, 0.0],
        [False, False, True],
    )
    values = torch.tensor([100.0, 200.0, 300.0])
    mc = discounted_returns(data)["qol_years"]
    assert torch.allclose(qol_td_lambda_targets(data, values, 1.0), mc)
    gamma = math.exp(-contracts.TURN_YEARS / contracts.HORIZON_YEARS)
    one_step = qol_td_lambda_targets(data, values, 0.0).double()
    # Row 0 bootstraps the head's value of row 1; row 1 closes on its tail; row 2 terminates.
    assert math.isclose(float(one_step[0]), 1.0 + gamma * 200.0, rel_tol=1e-6)
    assert math.isclose(float(one_step[1]), 2.0 + gamma * 10.0, rel_tol=1e-6)
    assert math.isclose(float(one_step[2]), 3.0, rel_tol=1e-6)
    mixed = qol_td_lambda_targets(data, values, 0.8).double()
    assert math.isclose(
        float(mixed[0]), 1.0 + gamma * (0.2 * 200.0 + 0.8 * float(mixed[1])), rel_tol=1e-6
    )


def test_film_conditions_the_bounded_head() -> None:
    critic = LaborTaskCritic(_layout(), width=24, token_width=16, qol_head=LaborQoLHead())
    tilted = dict(_labor_tensors(batch=1), labor_personality=torch.tensor([[0.9, 0.1]]))
    identity = critic(tilted)["qol_years"]  # zero-initialised FiLM is the identity
    state = {k: v.clone() for k, v in critic.state_dict().items() if ".film_" not in k}
    for film in critic.encoder._films():
        torch.nn.init.normal_(film.weight, std=0.3)
    assert not torch.equal(critic(tilted)["qol_years"], identity)
    # A pre-FiLM bounded state still loads with identity FiLM.
    fresh = LaborTaskCritic(_layout(), width=24, token_width=16, qol_head=LaborQoLHead())
    for film in fresh.encoder._films():
        torch.nn.init.normal_(film.weight, std=0.3)
    fresh.load_state_dict(state, strict=True)
    assert torch.equal(fresh(tilted)["qol_years"], identity)


def test_actor_is_untouched_by_the_bounded_head() -> None:
    torch.manual_seed(2)
    actor = LaborActor(_layout(), width=24, token_width=16)
    assert not any("qol" in name for name, _ in actor.named_parameters())
