"""Reusable, non-mutating Gate 4 optimizer-direction audit primitives.

This module deliberately does not run a PPO update.  It audits parameter ownership,
captures component-gradient geometry on caller-selected parameters, and scores already
bound before/after probability probes.  Component gradients are descriptive geometry;
they are never added up and presented as attribution of a clipped nonlinear Adam step.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import isfinite, sqrt
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

ATTRIBUTION_SCOPE = (
    "component gradients describe local geometry only; they are not additive attribution "
    "of clipping or the nonlinear Adam update"
)

NamedParameters = Sequence[tuple[str, nn.Parameter]]


def _named_parameters(value: nn.Module | NamedParameters) -> list[tuple[str, nn.Parameter]]:
    parameters = list(value.named_parameters()) if isinstance(value, nn.Module) else list(value)
    if not parameters:
        raise ValueError("parameter set must not be empty")
    names = [name for name, _ in parameters]
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique within a component")
    if len({id(parameter) for _, parameter in parameters}) != len(parameters):
        raise ValueError("one parameter object has multiple names within a component")
    return parameters


def _storage_key(parameter: nn.Parameter) -> tuple[str, int, int]:
    storage = parameter.untyped_storage()
    return str(parameter.device), int(storage.data_ptr()), int(storage.nbytes())


def parameter_overlap_report(
    actor: nn.Module | NamedParameters,
    value_critic: nn.Module | NamedParameters,
    cost_critic: nn.Module | NamedParameters,
) -> dict[str, object]:
    """Report actor/value/cost ownership overlap as three separate pair audits."""
    components = {
        "actor": _named_parameters(actor),
        "valueCritic": _named_parameters(value_critic),
        "costCritic": _named_parameters(cost_critic),
    }
    pairs: dict[str, dict[str, object]] = {}
    for first_name, second_name in (
        ("actor", "valueCritic"),
        ("actor", "costCritic"),
        ("valueCritic", "costCritic"),
    ):
        overlaps = []
        for first_parameter_name, first in components[first_name]:
            for second_parameter_name, second in components[second_name]:
                same_object = first is second
                same_storage = _storage_key(first) == _storage_key(second)
                if same_object or same_storage:
                    overlaps.append(
                        {
                            "firstParameter": first_parameter_name,
                            "secondParameter": second_parameter_name,
                            "sameObject": same_object,
                            "sameStorage": same_storage,
                        }
                    )
        key = f"{first_name}__{second_name}"
        pairs[key] = {
            "first": first_name,
            "second": second_name,
            "firstParameterCount": len(components[first_name]),
            "secondParameterCount": len(components[second_name]),
            "overlapCount": len(overlaps),
            "zeroOverlap": not overlaps,
            "overlaps": overlaps,
        }
    return {
        "pairs": pairs,
        "actorValueZeroOverlap": pairs["actor__valueCritic"]["zeroOverlap"],
        "actorCostZeroOverlap": pairs["actor__costCritic"]["zeroOverlap"],
        "valueCostZeroOverlap": pairs["valueCritic__costCritic"]["zeroOverlap"],
        "allZeroOverlap": all(bool(pair["zeroOverlap"]) for pair in pairs.values()),
    }


@dataclass(frozen=True)
class ComponentGradient:
    name: str
    loss: float
    norm: float
    gradients: Mapping[str, Tensor]


@dataclass
class _RngSnapshot:
    python: Any
    numpy_global: Any
    torch_cpu: Tensor
    torch_cuda: list[Tensor]
    torch_generators: dict[str, Tensor]
    numpy_generators: dict[str, dict[str, Any]]


def _capture_rng(
    torch_generators: Mapping[str, torch.Generator],
    numpy_generators: Mapping[str, np.random.Generator],
) -> _RngSnapshot:
    return _RngSnapshot(
        python=copy.deepcopy(random.getstate()),
        numpy_global=copy.deepcopy(np.random.get_state()),
        torch_cpu=torch.random.get_rng_state().clone(),
        torch_cuda=[state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
        torch_generators={
            name: generator.get_state().clone() for name, generator in torch_generators.items()
        },
        numpy_generators={
            name: copy.deepcopy(dict(generator.bit_generator.state))
            for name, generator in numpy_generators.items()
        },
    )


def _restore_rng(
    state: _RngSnapshot,
    torch_generators: Mapping[str, torch.Generator],
    numpy_generators: Mapping[str, np.random.Generator],
) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy_global)
    torch.random.set_rng_state(state.torch_cpu)
    if state.torch_cuda:
        torch.cuda.set_rng_state_all(state.torch_cuda)
    for name, torch_generator in torch_generators.items():
        torch_generator.set_state(state.torch_generators[name])
    for name, numpy_generator in numpy_generators.items():
        numpy_generator.bit_generator.state = copy.deepcopy(state.numpy_generators[name])


def _restore_parameters(
    parameters: NamedParameters,
    values: Mapping[str, Tensor],
    gradients: Mapping[str, Tensor | None],
) -> None:
    with torch.no_grad():
        for name, parameter in parameters:
            parameter.copy_(values[name])
            saved_gradient = gradients[name]
            parameter.grad = None if saved_gradient is None else saved_gradient.clone()


def _restore_modules(
    states: Sequence[tuple[nn.Module, Mapping[str, Any]]],
    training_modes: Sequence[tuple[nn.Module, bool]],
) -> None:
    for module, state in states:
        module.load_state_dict(copy.deepcopy(state), strict=True)
    # Assign directly so a parent's recursive train() call cannot flatten mixed child modes.
    for module, training in training_modes:
        module.training = training


def capture_component_gradients(
    components: Mapping[str, Callable[[], Tensor]],
    parameters: nn.Module | NamedParameters,
    *,
    optimizers: Sequence[torch.optim.Optimizer] = (),
    state_modules: Sequence[nn.Module] = (),
    torch_generators: Mapping[str, torch.Generator] | None = None,
    numpy_generators: Mapping[str, np.random.Generator] | None = None,
) -> dict[str, ComponentGradient]:
    """Capture component gradients from an identical state and restore all audited state.

    Each callable must build and return one finite scalar loss without calling ``backward``.
    Parameters, pre-existing ``.grad`` values, Python/NumPy/Torch RNG state, explicitly
    supplied generator state, optimizer state, and buffers/training modes of a supplied module
    are restored even when a component fails. ``state_modules`` extends that protection when
    ``parameters`` is a selected named-parameter subset rather than the owning module.
    """
    if not components:
        raise ValueError("at least one gradient component is required")
    named = _named_parameters(parameters)
    if any(not parameter.requires_grad for _, parameter in named):
        raise ValueError("all selected parameters must require gradients")
    if len(set(components)) != len(components):
        raise ValueError("gradient component names must be unique")
    torch_generators = dict(torch_generators or {})
    numpy_generators = dict(numpy_generators or {})
    audit_modules = ([parameters] if isinstance(parameters, nn.Module) else []) + list(
        state_modules
    )
    if len({id(module) for module in audit_modules}) != len(audit_modules):
        raise ValueError("state_modules repeats a module")
    module_states = [
        (module, copy.deepcopy(module.state_dict())) for module in audit_modules
    ]
    training_modes_by_id: dict[int, tuple[nn.Module, bool]] = {}
    for module in audit_modules:
        for submodule in module.modules():
            training_modes_by_id.setdefault(id(submodule), (submodule, submodule.training))
    training_modes = list(training_modes_by_id.values())
    parameter_values = {name: parameter.detach().clone() for name, parameter in named}
    parameter_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in named
    }
    optimizer_states = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
    original_rng = _capture_rng(torch_generators, numpy_generators)
    common_rng = copy.deepcopy(original_rng)
    output: dict[str, ComponentGradient] = {}
    try:
        for component_name in sorted(components):
            _restore_modules(module_states, training_modes)
            _restore_parameters(named, parameter_values, parameter_gradients)
            for optimizer, state in zip(optimizers, optimizer_states, strict=True):
                optimizer.load_state_dict(copy.deepcopy(state))
            _restore_rng(common_rng, torch_generators, numpy_generators)
            loss = components[component_name]()
            if not isinstance(loss, Tensor) or loss.numel() != 1:
                raise ValueError(
                    f"gradient component {component_name} must return one tensor scalar"
                )
            if not bool(torch.isfinite(loss.detach())):
                raise ValueError(f"gradient component {component_name} returned a nonfinite loss")
            gradients = torch.autograd.grad(
                loss,
                [parameter for _, parameter in named],
                allow_unused=True,
                create_graph=False,
                retain_graph=False,
            )
            for (name, _), gradient in zip(named, gradients, strict=True):
                if gradient is not None and not bool(torch.isfinite(gradient.detach()).all()):
                    raise ValueError(
                        f"gradient component {component_name} returned a nonfinite "
                        f"gradient for parameter {name}"
                    )
            by_parameter = {
                name: (
                    torch.zeros_like(parameter, memory_format=torch.preserve_format).cpu()
                    if gradient is None
                    else gradient.detach().cpu().clone()
                )
                for (name, parameter), gradient in zip(named, gradients, strict=True)
            }
            squared_norm = sum(
                float(torch.sum(gradient.to(torch.float64) ** 2))
                for gradient in by_parameter.values()
            )
            output[component_name] = ComponentGradient(
                name=component_name,
                loss=float(loss.detach()),
                norm=sqrt(squared_norm),
                gradients=by_parameter,
            )
    finally:
        _restore_modules(module_states, training_modes)
        _restore_parameters(named, parameter_values, parameter_gradients)
        for optimizer, state in zip(optimizers, optimizer_states, strict=True):
            optimizer.load_state_dict(copy.deepcopy(state))
        _restore_rng(original_rng, torch_generators, numpy_generators)
    return output


def component_gradient_geometry(
    gradients: Mapping[str, ComponentGradient],
) -> list[dict[str, object]]:
    """Return pairwise gradient norms/cosines without claiming optimizer attribution."""
    if not gradients:
        return []
    parameter_names = None
    for component in gradients.values():
        names = tuple(sorted(component.gradients))
        if parameter_names is None:
            parameter_names = names
        elif names != parameter_names:
            raise ValueError("gradient components do not cover the same parameter set")
    output = []
    for first_name, second_name in combinations(sorted(gradients), 2):
        first, second = gradients[first_name], gradients[second_name]
        dot = sum(
            float(
                torch.sum(
                    first.gradients[name].to(torch.float64)
                    * second.gradients[name].to(torch.float64)
                )
            )
            for name in parameter_names or ()
        )
        cosine = dot / (first.norm * second.norm) if first.norm and second.norm else None
        output.append(
            {
                "firstComponent": first_name,
                "secondComponent": second_name,
                "firstNorm": first.norm,
                "secondNorm": second.norm,
                "dotProduct": dot,
                "cosine": cosine,
                "attributionScope": ATTRIBUTION_SCOPE,
            }
        )
    return output


EffectLabel = Literal["beneficial", "harmful", "tie", "ambiguous"]
ProbabilityMeasure = Literal["log-density", "log-neighborhood-probability"]


@dataclass(frozen=True)
class DirectionProbe:
    """A before/after probability probe with an independently frozen effect label."""

    independent_unit_id: str
    head: str
    alternative_id: str
    effect_label: EffectLabel
    measure: ProbabilityMeasure
    before_log_measure: float
    after_log_measure: float
    active: bool = True
    supported: bool = True
    unsupported_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.independent_unit_id or not self.head or not self.alternative_id:
            raise ValueError("direction probe identities must be nonempty")
        if self.effect_label not in {"beneficial", "harmful", "tie", "ambiguous"}:
            raise ValueError("unknown effect label")
        if self.measure not in {"log-density", "log-neighborhood-probability"}:
            raise ValueError("unknown probability measure")
        for name in ("before_log_measure", "after_log_measure"):
            if not isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not self.active and self.supported:
            raise ValueError("inactive head cannot be a supported direction probe")
        if not self.supported and not self.unsupported_reason:
            raise ValueError("unsupported direction probe requires a reason")

    @property
    def log_change(self) -> float:
        return float(self.after_log_measure - self.before_log_measure)


def score_direction_probes(
    probes: Sequence[DirectionProbe], *, change_tolerance: float = 0.0
) -> list[dict[str, object]]:
    """Score supported probability movement with explicit tie/ambiguity exclusions."""
    change_tolerance = float(change_tolerance)
    if not isfinite(change_tolerance) or change_tolerance < 0:
        raise ValueError("change_tolerance must be finite and nonnegative")
    seen: set[tuple[str, str, str]] = set()
    groups: dict[tuple[str, str], list[DirectionProbe]] = {}
    for probe in probes:
        identity = (probe.independent_unit_id, probe.head, probe.alternative_id)
        if identity in seen:
            raise ValueError("duplicate direction probe independent unit/head/alternative")
        seen.add(identity)
        groups.setdefault((probe.head, probe.measure), []).append(probe)
    output = []
    for (head, measure), records in sorted(groups.items()):
        supported = [
            row
            for row in records
            if row.active and row.supported and row.effect_label in {"beneficial", "harmful"}
        ]
        correct = 0
        flat = 0
        supported_by_unit: dict[str, list[int]] = {}
        changes_by_unit: dict[str, list[float]] = {}
        for row in supported:
            change = row.log_change
            flat += int(abs(change) <= change_tolerance)
            is_correct = int(
                (row.effect_label == "beneficial" and change > change_tolerance)
                or (row.effect_label == "harmful" and change < -change_tolerance)
            )
            correct += is_correct
            supported_by_unit.setdefault(row.independent_unit_id, []).append(is_correct)
            changes_by_unit.setdefault(row.independent_unit_id, []).append(change)
        all_units = {row.independent_unit_id for row in records}
        unit_scores = [
            float(np.mean(supported_by_unit[unit])) for unit in sorted(supported_by_unit)
        ]
        unit_changes = [
            float(np.mean(changes_by_unit[unit])) for unit in sorted(changes_by_unit)
        ]
        output.append(
            {
                "head": head,
                "measure": measure,
                "independentUnitCount": len(all_units),
                "probeCount": len(records),
                "supportedDirectionProbeCount": len(supported),
                "correctDirectionProbeCount": correct,
                "probeWeightedCorrectNumerator": correct,
                "probeWeightedSupportedDenominator": len(supported),
                "probeWeightedDirectionAccuracy": (
                    correct / len(supported) if supported else None
                ),
                "equalIndependentUnitScoreSum": (
                    float(sum(unit_scores)) if unit_scores else None
                ),
                "equalIndependentUnitDenominator": len(unit_scores),
                "equalIndependentUnitDirectionAccuracy": (
                    float(np.mean(unit_scores)) if unit_scores else None
                ),
                "equalIndependentUnitExcludedCount": len(all_units) - len(unit_scores),
                "flatChangeProbeCount": flat,
                "tieEffectProbeCount": sum(row.effect_label == "tie" for row in records),
                "ambiguousEffectProbeCount": sum(
                    row.effect_label == "ambiguous" for row in records
                ),
                "inactiveProbeCount": sum(not row.active for row in records),
                "unsupportedProbeCount": sum(
                    row.active and not row.supported for row in records
                ),
                "unsupportedReasons": sorted(
                    {row.unsupported_reason for row in records if row.unsupported_reason}
                ),
                "probeWeightedMeanLogChange": (
                    float(np.mean([row.log_change for row in supported])) if supported else None
                ),
                "equalIndependentUnitMeanLogChange": (
                    float(np.mean(unit_changes)) if unit_changes else None
                ),
                "attributionScope": ATTRIBUTION_SCOPE,
            }
        )
    return output
