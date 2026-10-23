"""Small, state-preserving measurements for opt-in PPO update traces.

The records produced here describe the sampled minibatch and local gradient
geometry.  They do not decompose Adam's nonlinear update and they do not attach
an independently established benefit label to an advantage or action.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .optimizer_direction_audit import (
    ATTRIBUTION_SCOPE,
    ComponentGradient,
    component_gradient_geometry,
)

MAX_AUDIT_ROWS = 64


def checked_audit_rows(value: int) -> int:
    """Validate the deliberately small first-minibatch audit cap."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("optimizer audit rows must be an integer")
    if not 0 <= value <= MAX_AUDIT_ROWS:
        raise ValueError(f"optimizer audit rows must be between 0 and {MAX_AUDIT_ROWS}")
    return value


def balanced_advantage_contributions(
    objective: np.ndarray,
    constraint: np.ndarray,
    combined: np.ndarray,
    regimes: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition the already-balanced advantage using its exact centering/scale.

    ``objective + constraint`` is the unbalanced combined advantage and
    ``combined`` is the result returned by :func:`ppo.balance`.  Applying the
    same centering and denominator to both inputs preserves their sum.  A
    single-row regime is zeroed exactly as ``balance`` zeroes it.
    """
    objective = np.asarray(objective, dtype=np.float64)
    constraint = np.asarray(constraint, dtype=np.float64)
    combined = np.asarray(combined, dtype=np.float64)
    if objective.shape != constraint.shape or objective.shape != combined.shape:
        raise ValueError("advantage contribution arrays must have the same shape")

    objective_out = np.empty_like(objective)
    constraint_out = np.empty_like(constraint)
    if regimes is None or len(set(np.asarray(regimes).tolist())) <= 1:
        total = objective + constraint
        denominator = max(1e-8, float(total.std()))
        objective_out[:] = (objective - objective.mean()) / denominator
        constraint_out[:] = (constraint - constraint.mean()) / denominator
    else:
        regimes = np.asarray(regimes)
        if regimes.shape != objective.shape:
            raise ValueError("regimes must have one entry per advantage row")
        for name in sorted(set(regimes.tolist())):
            rows = regimes == name
            if int(rows.sum()) <= 1:
                objective_out[rows] = 0.0
                constraint_out[rows] = 0.0
                continue
            total = objective[rows] + constraint[rows]
            denominator = max(1e-8, float(total.std()))
            objective_out[rows] = (objective[rows] - objective[rows].mean()) / denominator
            constraint_out[rows] = (constraint[rows] - constraint[rows].mean()) / denominator

    # Training arrays may originate as float32. Validate the accounting identity
    # without pretending those inputs carry float64 precision.
    if not np.allclose(objective_out + constraint_out, combined, rtol=1e-6, atol=1e-7):
        raise ValueError(
            "balanced objective and constraint contributions do not reconstruct combined advantage"
        )
    return objective_out, constraint_out


def frozen_branch_policy_terms(
    ratio: Tensor,
    objective_advantage: Tensor,
    constraint_advantage: Tensor,
    weight: Tensor,
    clip: float,
    *,
    use_unclipped: Tensor | None = None,
) -> tuple[dict[str, Tensor], Tensor]:
    """Split PPO policy loss while holding its actual combined clip branch fixed.

    The returned component losses add to the policy expression for the selected
    branch.  This is a local loss/gradient probe, not additive Adam attribution.
    """
    combined = objective_advantage + constraint_advantage
    clipped_ratio = ratio.clamp(1 - clip, 1 + clip)
    if use_unclipped is None:
        use_unclipped = (ratio * combined <= clipped_ratio * combined).detach()
    if use_unclipped.shape != ratio.shape:
        raise ValueError("frozen PPO branch mask has the wrong shape")
    selected_ratio = torch.where(use_unclipped, ratio, clipped_ratio)
    denominator = weight.sum()
    terms = {
        "objectivePolicy": -(weight * selected_ratio * objective_advantage).sum() / denominator,
        "constraintPolicy": -(weight * selected_ratio * constraint_advantage).sum() / denominator,
    }
    return terms, use_unclipped


def geometry_record(captured: Mapping[str, ComponentGradient]) -> dict[str, Any]:
    """Compact component losses/norms and pairwise cosines; never raw gradients."""
    return {
        "components": {
            name: {"loss": float(component.loss), "norm": float(component.norm)}
            for name, component in sorted(captured.items())
        },
        "pairs": component_gradient_geometry(captured),
        "attributionScope": ATTRIBUTION_SCOPE,
    }


def critic_distribution_record(logits: Tensor, target: Tensor, bins: Tensor) -> dict[str, Any]:
    """Target-support and predicted-distribution diagnostics in the critic's own space."""
    if logits.ndim != 2 or target.ndim != 1 or len(logits) != len(target):
        raise ValueError("critic logits/targets have incompatible shapes")
    if bins.ndim != 1 or logits.shape[1] != len(bins):
        raise ValueError("critic bins do not match logits")
    with torch.no_grad():
        target = target.to(dtype=logits.dtype, device=logits.device)
        bins = bins.to(dtype=logits.dtype, device=logits.device)
        transformed = torch.sign(target) * torch.log1p(torch.abs(target))
        probability = torch.softmax(logits, dim=-1)
        log_probability = torch.log_softmax(logits, dim=-1)
        raw_bins = torch.sign(bins) * torch.expm1(torch.abs(bins))
        prediction = (probability * raw_bins).sum(dim=-1)
        entropy = -(probability * log_probability).sum(dim=-1)

        def number(value: Tensor) -> float:
            return float(value.detach().to(torch.float64).cpu())

        below = transformed < bins[0]
        above = transformed > bins[-1]
        return {
            "rowCount": len(target),
            "targetRawMin": number(target.min()),
            "targetRawMax": number(target.max()),
            "targetSymlogMin": number(transformed.min()),
            "targetSymlogMax": number(transformed.max()),
            "supportSymlogLow": number(bins[0]),
            "supportSymlogHigh": number(bins[-1]),
            "targetBelowSupportFraction": number(below.float().mean()),
            "targetAboveSupportFraction": number(above.float().mean()),
            "targetOutsideSupportFraction": number((below | above).float().mean()),
            "predictedRawMean": number(prediction.mean()),
            "lowerEndpointProbabilityMean": number(probability[:, 0].mean()),
            "upperEndpointProbabilityMean": number(probability[:, -1].mean()),
            "categoricalEntropyMean": number(entropy.mean()),
            "interpretation": (
                "internal lambda-return target support/fit diagnostic; "
                "not independent full-return calibration"
            ),
        }


def same_sampled_action_record(
    transition_ids: Sequence[str],
    before_joint: Tensor,
    after_joint: Tensor,
    before_entropy: Tensor,
    after_entropy: Tensor,
    before_heads: Mapping[str, tuple[Tensor, Tensor]],
    after_heads: Mapping[str, tuple[Tensor, Tensor]],
    before_coordinates: Mapping[
        str, Mapping[str, tuple[Tensor, Tensor, Tensor]]
    ] | None = None,
    after_coordinates: Mapping[
        str, Mapping[str, tuple[Tensor, Tensor, Tensor]]
    ] | None = None,
    native_recorded_activity: Mapping[
        str, Mapping[str, Sequence[bool | None]]
    ] | None = None,
) -> dict[str, Any]:
    """Serialize bounded same-state/same-action log-probability changes."""
    if set(before_heads) != set(after_heads):
        raise ValueError("before/after policy heads differ")
    count = len(transition_ids)
    tensors = (before_joint, after_joint, before_entropy, after_entropy)
    if any(tensor.ndim != 1 or len(tensor) != count for tensor in tensors):
        raise ValueError("joint policy measurements must have one value per transition")
    for name in before_heads:
        values = (*before_heads[name], *after_heads[name])
        if any(value.ndim != 1 or len(value) != count for value in values):
            raise ValueError(f"policy head {name} must have one value per transition")
    before_coordinates = dict(before_coordinates or {})
    after_coordinates = dict(after_coordinates or {})
    native_recorded_activity = dict(native_recorded_activity or {})
    if before_coordinates.keys() != after_coordinates.keys():
        raise ValueError("before/after policy coordinate groups differ")

    def mean(value: Tensor) -> float:
        return float(value.detach().to(torch.float64).mean().cpu())

    joint_delta = after_joint - before_joint
    entropy_delta = after_entropy - before_entropy
    joint_delta_float64 = joint_delta.detach().to(torch.float64)
    probability_ratio = torch.exp(joint_delta_float64)
    nonnegative_kl = torch.expm1(joint_delta_float64) - joint_delta_float64
    per_head = {}
    for name in sorted(before_heads):
        before_lp, before_ent = before_heads[name]
        after_lp, after_ent = after_heads[name]
        per_head[name] = {
            "beforeMean": mean(before_lp),
            "afterMean": mean(after_lp),
            "meanDelta": mean(after_lp - before_lp),
            "beforeEntropyMean": mean(before_ent),
            "afterEntropyMean": mean(after_ent),
            "meanEntropyDelta": mean(after_ent - before_ent),
        }

    per_coordinate = {}
    coordinate_values = {}
    for group in before_coordinates:
        if before_coordinates[group].keys() != after_coordinates[group].keys():
            raise ValueError(f"before/after policy coordinates differ for {group}")
        per_coordinate[group] = {}
        coordinate_values[group] = {}
        for index, name in enumerate(before_coordinates[group]):
            before_lp, before_ent, before_active = before_coordinates[group][name]
            after_lp, after_ent, after_active = after_coordinates[group][name]
            values = (before_lp, before_ent, before_active, after_lp, after_ent, after_active)
            if any(value.ndim != 1 or len(value) != count for value in values):
                raise ValueError(
                    f"policy coordinate {group}.{name} must have one value per transition"
                )
            if not torch.equal(before_active.bool(), after_active.bool()):
                raise ValueError(f"policy coordinate activity changed for {group}.{name}")
            native = list(native_recorded_activity.get(group, {}).get(name, [None] * count))
            if len(native) != count or any(value not in (True, False, None) for value in native):
                raise ValueError(f"native recorded activity is malformed for {group}.{name}")
            active = [bool(value) for value in before_active.detach().cpu()]
            per_coordinate[group][name] = {
                "nativeContractIndex": index,
                "beforeMean": mean(before_lp),
                "afterMean": mean(after_lp),
                "meanDelta": mean(after_lp - before_lp),
                "beforeEntropyMean": mean(before_ent),
                "afterEntropyMean": mean(after_ent),
                "meanEntropyDelta": mean(after_ent - before_ent),
                "policyScoredRowCount": sum(active),
                "policyNotScoredRowCount": count - sum(active),
                "nativeRecordedActiveRowCount": sum(value is True for value in native),
                "nativeRecordedInactiveRowCount": sum(value is False for value in native),
                "nativeRecordedActivityUnavailableRowCount": sum(value is None for value in native),
                "scoredButNativeRecordedInactiveRowCount": sum(
                    scored and value is False for scored, value in zip(active, native, strict=True)
                ),
            }
            coordinate_values[group][name] = (
                before_lp, before_ent, after_lp, after_ent, active, native, index)

    rows = []
    for row, transition_id in enumerate(transition_ids):
        heads = {
            name: {
                "before": float(before_heads[name][0][row].detach().cpu()),
                "after": float(after_heads[name][0][row].detach().cpu()),
                "delta": float(
                    (after_heads[name][0][row] - before_heads[name][0][row])
                    .detach()
                    .cpu()
                ),
            }
            for name in sorted(before_heads)
        }
        coordinates = {}
        for group in coordinate_values:
            coordinates[group] = {}
            for name, values in coordinate_values[group].items():
                before_lp, before_ent, after_lp, after_ent, active, native, index = values
                relation = (
                    "not-policy-scored"
                    if not active[row]
                    else "native-recorded-activity-unavailable"
                    if native[row] is None
                    else "scored-and-native-recorded-active"
                    if native[row]
                    else "scored-but-native-recorded-inactive"
                )
                coordinates[group][name] = {
                    "nativeContractIndex": index,
                    "before": float(before_lp[row].detach().cpu()),
                    "after": float(after_lp[row].detach().cpu()),
                    "delta": float((after_lp[row] - before_lp[row]).detach().cpu()),
                    "beforeEntropy": float(before_ent[row].detach().cpu()),
                    "afterEntropy": float(after_ent[row].detach().cpu()),
                    "entropyDelta": float((after_ent[row] - before_ent[row]).detach().cpu()),
                    "policyScored": active[row],
                    "nativeRecordedActive": native[row],
                    "activityRelation": relation,
                }
        rows.append(
            {
                "transitionId": transition_id,
                "jointBefore": float(before_joint[row].detach().cpu()),
                "jointAfter": float(after_joint[row].detach().cpu()),
                "jointDelta": float(joint_delta[row].detach().cpu()),
                "perHead": heads,
                "perCoordinate": coordinates,
            }
        )
    return {
        "postStepSampledPolicyShift": {
            "measurementPhase": "post-actor-step",
            "rowSelection": "bounded-fixed-audit-rows",
            "rowCount": count,
            "signedApproxKlMean": mean(-joint_delta),
            "nonnegativeKlEstimateMean": mean(nonnegative_kl),
            "sampledProbabilityRatio": {
                "definition": "exp(jointAfter-jointBefore)",
                "count": count,
                "min": float(probability_ratio.detach().min().cpu()),
                "max": float(probability_ratio.detach().max().cpu()),
            },
            "interpretation": (
                "post-step shift on the bounded fixed audit rows and their sampled actions; "
                "not exact KL, not a full-rollout estimate, and not used for early stopping"
            ),
        },
        "joint": {
            "beforeMean": mean(before_joint),
            "afterMean": mean(after_joint),
            "meanDelta": mean(joint_delta),
            "beforeEntropyMean": mean(before_entropy),
            "afterEntropyMean": mean(after_entropy),
            "meanEntropyDelta": mean(entropy_delta),
        },
        "perHead": per_head,
        "perCoordinate": per_coordinate,
        "rows": rows,
        "nativeRecordedActivitySemantics": (
            "actionDiagnostics.activeMask is descriptive native execution context: "
            "strategic entries record opening capability and service entries record a "
            "positive post-request allocation; it is not an exogenous feasibility mask "
            "or a recommendation to remove policy likelihood terms"
        ),
        "interpretation": (
            "same-state/same-sampled-action policy density change; advantage sign is "
            "descriptive training context, "
            "not an independently established benefit label"
        ),
    }


@contextmanager
def preserved_observer_state(
    module: nn.Module,
    *,
    numpy_generators: Mapping[str, np.random.Generator] | None = None,
):
    """Preserve RNG and complete module state around a read-only forward.

    The parameter snapshot is taken when this context is entered, which is after
    the real optimizer step for the post-update observation. Restoring it therefore
    preserves that update while rejecting side effects of the diagnostic forward.
    """
    numpy_generators = dict(numpy_generators or {})
    parameters = {
        name: parameter.detach().clone() for name, parameter in module.named_parameters()
    }
    buffers = {name: buffer.detach().clone() for name, buffer in module.named_buffers()}
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
    }
    modes = [(submodule, submodule.training) for submodule in module.modules()]
    python_rng = copy.deepcopy(random.getstate())
    numpy_rng = copy.deepcopy(np.random.get_state())
    torch_rng = torch.random.get_rng_state().clone()
    cuda_rng = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    explicit_numpy = {
        name: copy.deepcopy(dict(generator.bit_generator.state))
        for name, generator in numpy_generators.items()
    }
    try:
        yield
    finally:
        structure_error = None
        with torch.no_grad():
            current_parameters = dict(module.named_parameters())
            current_buffers = dict(module.named_buffers())
            if current_parameters.keys() != parameters.keys():
                structure_error = "observer forward changed module parameter structure"
            elif current_buffers.keys() != buffers.keys():
                structure_error = "observer forward changed module buffer structure"
            else:
                for name, value in parameters.items():
                    current_parameters[name].copy_(value)
                for name, value in buffers.items():
                    current_buffers[name].copy_(value)
        if structure_error is None:
            for name, parameter in module.named_parameters():
                gradient = gradients[name]
                parameter.grad = None if gradient is None else gradient.clone()
        for submodule, training in modes:
            submodule.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.set_rng_state(torch_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        for name, generator in numpy_generators.items():
            generator.bit_generator.state = copy.deepcopy(explicit_numpy[name])
        if structure_error is not None:
            raise RuntimeError(structure_error)
