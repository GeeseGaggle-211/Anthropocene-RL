import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from anthropocene_rl.autonomous_v7.optimizer_direction_audit import (
    ATTRIBUTION_SCOPE,
    DirectionProbe,
    capture_component_gradients,
    component_gradient_geometry,
    parameter_overlap_report,
    score_direction_probes,
)


class ParameterBag(nn.Module):
    def __init__(self, parameter: nn.Parameter) -> None:
        super().__init__()
        self.parameter = parameter


def assert_nested_equal(first, second) -> None:
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        assert np.array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            assert_nested_equal(left, right)
    else:
        assert first == second


def test_parameter_overlap_audits_actor_value_and_cost_pairs_separately() -> None:
    shared = nn.Parameter(torch.tensor([1.0]))
    actor = ParameterBag(shared)
    value = ParameterBag(shared)
    cost = ParameterBag(nn.Parameter(torch.tensor([2.0])))
    report = parameter_overlap_report(actor, value, cost)
    assert report["actorValueZeroOverlap"] is False
    assert report["actorCostZeroOverlap"] is True
    assert report["valueCostZeroOverlap"] is True
    overlap = report["pairs"]["actor__valueCritic"]["overlaps"][0]
    assert overlap["sameObject"] is True
    assert overlap["sameStorage"] is True


def test_component_gradient_capture_restores_grad_rng_parameter_and_optimizer_state() -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    explicit_torch = torch.Generator().manual_seed(14)
    explicit_numpy = np.random.default_rng(15)
    parameter = nn.Parameter(torch.tensor([2.0]))
    module = ParameterBag(parameter)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
    optimizer.zero_grad()
    parameter.square().sum().backward()
    optimizer.step()
    parameter.grad = torch.tensor([7.0])

    parameter_before = parameter.detach().clone()
    gradient_before = parameter.grad.detach().clone()
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    python_before = copy.deepcopy(random.getstate())
    numpy_before = copy.deepcopy(np.random.get_state())
    torch_before = torch.random.get_rng_state().clone()
    explicit_torch_before = explicit_torch.get_state().clone()
    explicit_numpy_before = copy.deepcopy(explicit_numpy.bit_generator.state)
    draws = {}

    def component(name: str, sign: float):
        def loss() -> torch.Tensor:
            draws[name] = (
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
                float(torch.rand((), generator=explicit_torch)),
                float(explicit_numpy.random()),
            )
            optimizer.param_groups[0]["lr"] = 9.0
            optimizer.state[parameter]["step"].add_(100)
            with torch.no_grad():
                parameter.add_(3.0)
                parameter.grad.fill_(99.0)
            return sign * parameter.square().sum()

        return loss

    captured = capture_component_gradients(
        {"positive": component("positive", 1.0), "negative": component("negative", -1.0)},
        module,
        optimizers=[optimizer],
        torch_generators={"audit": explicit_torch},
        numpy_generators={"audit": explicit_numpy},
    )
    assert draws["positive"] == draws["negative"]
    assert torch.equal(parameter, parameter_before)
    assert torch.equal(parameter.grad, gradient_before)
    assert_nested_equal(optimizer.state_dict(), optimizer_before)
    assert random.getstate() == python_before
    assert_nested_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert torch.equal(explicit_torch.get_state(), explicit_torch_before)
    assert_nested_equal(explicit_numpy.bit_generator.state, explicit_numpy_before)

    geometry = component_gradient_geometry(captured)
    assert len(geometry) == 1
    assert geometry[0]["cosine"] == pytest.approx(-1.0)
    assert geometry[0]["attributionScope"] == ATTRIBUTION_SCOPE


def test_component_gradient_failure_still_restores_state() -> None:
    parameter = nn.Parameter(torch.tensor([2.0]))
    module = ParameterBag(parameter)
    parameter.grad = torch.tensor([3.0])
    before = parameter.detach().clone()
    before_grad = parameter.grad.detach().clone()
    rng_before = torch.random.get_rng_state().clone()

    def invalid() -> torch.Tensor:
        with torch.no_grad():
            parameter.add_(10.0)
            parameter.grad.fill_(20.0)
        torch.rand(4)
        return parameter.sum() * torch.tensor(float("nan"))

    with pytest.raises(ValueError, match="nonfinite"):
        capture_component_gradients({"invalid": invalid}, module)
    assert torch.equal(parameter, before)
    assert torch.equal(parameter.grad, before_grad)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_component_gradient_failure_restores_buffers_and_mixed_training_modes() -> None:
    module = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 1))
    module.train()
    module[1].eval()
    state_before = copy.deepcopy(module.state_dict())
    modes_before = [submodule.training for submodule in module.modules()]

    def invalid() -> torch.Tensor:
        result = module(torch.tensor([[1.0, 2.0], [3.0, 5.0]]))
        module.eval()
        return result.sum() * torch.tensor(float("nan"))

    with pytest.raises(ValueError, match="nonfinite"):
        capture_component_gradients({"invalid": invalid}, module)
    assert_nested_equal(module.state_dict(), state_before)
    assert [submodule.training for submodule in module.modules()] == modes_before


def test_component_gradient_rejects_nonfinite_derivative_and_restores_state() -> None:
    module = ParameterBag(nn.Parameter(torch.tensor([0.0])))
    module.register_buffer("audit_counter", torch.tensor([4.0]))
    module.parameter.grad = torch.tensor([7.0])
    module.train()
    state_before = copy.deepcopy(module.state_dict())
    gradient_before = module.parameter.grad.detach().clone()
    mode_before = module.training
    rng_before = torch.random.get_rng_state().clone()

    def singular() -> torch.Tensor:
        with torch.no_grad():
            module.audit_counter.add_(1.0)
            module.parameter.grad.fill_(99.0)
        module.eval()
        torch.rand(4)
        # The value is finite at zero, while its derivative is positive infinity.
        return torch.sqrt(module.parameter).sum()

    with pytest.raises(ValueError, match="nonfinite gradient"):
        capture_component_gradients({"singular": singular}, module)
    assert_nested_equal(module.state_dict(), state_before)
    assert torch.equal(module.parameter.grad, gradient_before)
    assert module.training is mode_before
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_direction_scoring_keeps_ties_ambiguity_and_unsupported_heads_explicit() -> None:
    probes = [
        DirectionProbe("state-1", "trade", "good", "beneficial", "log-density", -2.0, -1.0),
        DirectionProbe("state-2", "trade", "bad", "harmful", "log-density", -2.0, -1.0),
        DirectionProbe("state-3", "trade", "tie", "tie", "log-density", -2.0, -2.0),
        DirectionProbe(
            "state-4", "trade", "uncertain", "ambiguous", "log-density", -2.0, -3.0
        ),
        DirectionProbe(
            "state-5",
            "trade",
            "masked",
            "beneficial",
            "log-density",
            -2.0,
            -1.0,
            active=False,
            supported=False,
            unsupported_reason="commitment mask inactive",
        ),
    ]
    score = score_direction_probes(probes)[0]
    assert score["independentUnitCount"] == 5
    assert score["supportedDirectionProbeCount"] == 2
    assert score["correctDirectionProbeCount"] == 1
    assert score["probeWeightedDirectionAccuracy"] == 0.5
    assert score["equalIndependentUnitDirectionAccuracy"] == 0.5
    assert score["tieEffectProbeCount"] == 1
    assert score["ambiguousEffectProbeCount"] == 1
    assert score["inactiveProbeCount"] == 1
    assert score["unsupportedReasons"] == ["commitment mask inactive"]
    assert "not additive attribution" in score["attributionScope"]


def test_direction_scoring_separates_probe_weighting_from_equal_unit_weighting() -> None:
    probes = [
        DirectionProbe("many", "trade", "a", "beneficial", "log-density", -2.0, -1.0),
        DirectionProbe("many", "trade", "b", "beneficial", "log-density", -3.0, -2.0),
        DirectionProbe("few", "trade", "a", "harmful", "log-density", -2.0, -1.0),
    ]
    score = score_direction_probes(probes)[0]
    assert score["probeWeightedCorrectNumerator"] == 2
    assert score["probeWeightedSupportedDenominator"] == 3
    assert score["probeWeightedDirectionAccuracy"] == pytest.approx(2 / 3)
    assert score["equalIndependentUnitScoreSum"] == 1.0
    assert score["equalIndependentUnitDenominator"] == 2
    assert score["equalIndependentUnitDirectionAccuracy"] == 0.5
    assert score["equalIndependentUnitExcludedCount"] == 0
