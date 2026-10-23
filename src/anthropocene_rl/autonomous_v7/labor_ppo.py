"""Attempt-aware PPO primitives for the opt-in specialty-labor learner.

This module deliberately does not alter the historical Model 9 PPO entry
point.  A labor decision has one initial policy edge and, only after a typed
feasibility rejection, one zero-time retry edge.  The initial edge owns the
single non-labor sample.  The retry edge resamples and scores labor only.

The functions below operate on retained policy samples and explicit GAE
values.  They do not infer rewards from bootstrap values and do not treat
scarcity, low utilisation, or low production as feasibility failures.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from . import contracts
from .labor_bridge import (
    ACTION_SCHEMA,
    LaborAttempt,
    LaborCatalog,
    LaborFeedback,
    LaborPolicySample,
)
from .labor_model import (
    LABOR_COSTS,
    LABOR_OBJECTIVES,
    LaborPersonality,
    labor_entropy,
    labor_log_prob,
)
from .ppo_correctness import (
    capture_step_state,
    epoch_coverage,
    epoch_minibatches,
    fixed_audit_indices,
    restore_step_state,
)

LABOR_PPO_CONTRACT = "model9-specialty-attempt-ppo-v1"
LABOR_GATE_HEADS = (*LABOR_OBJECTIVES, *LABOR_COSTS)
LABOR_ACTION_FIELDS = frozenset(
    {
        "schemaVersion",
        "laborSpecialtyAllocation",
        "laborActivityAllocation",
        "movement",
        "fission",
        "tradeValue",
        "tradeGate",
        "tradeShare",
    }
)
LABOR_RETRY_FIELDS = frozenset({"laborSpecialtyAllocation", "laborActivityAllocation"})


def _vector(value: torch.Tensor, length: int, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.shape != (length,)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite floating vector of length {length}")
    return value


def _boolean_vector(value: torch.Tensor, length: int, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.shape != (length,):
        raise ValueError(f"{label} must be a boolean vector of length {length}")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_schema12_action(
    action: Mapping[str, Any], catalog: LaborCatalog, *, retry: bool
) -> LaborPolicySample:
    """Validate the exact reviewed action surface without inventing retired heads.

    Initial actions retain only movement/region, fission, and trade alongside
    labor.  Strategic, mix, service, routes, and commitment fields are not
    accepted.  Retry payloads contain exactly the two labor arrays; the native
    bridge copies all retained non-labor fields from the initial action.
    """

    expected = LABOR_RETRY_FIELDS if retry else LABOR_ACTION_FIELDS
    if not isinstance(action, Mapping) or set(action) != expected:
        raise ValueError(
            f"{'retry' if retry else 'initial'} labor action fields must be exactly "
            f"{sorted(expected)}"
        )
    if not retry:
        if action["schemaVersion"] != ACTION_SCHEMA:
            raise ValueError("labor action schema version differs from the producer contract")
        trade_values = action["tradeValue"]
        trade_gates = action["tradeGate"]
        if (
            not isinstance(trade_values, Sequence)
            or isinstance(trade_values, (str, bytes))
            or len(trade_values) != len(contracts.TRADE_GROUPS)
            or any(not math.isfinite(_finite_number(value, "tradeValue")) for value in trade_values)
        ):
            raise ValueError("tradeValue has the wrong width or a non-finite value")
        if (
            not isinstance(trade_gates, Sequence)
            or isinstance(trade_gates, (str, bytes))
            or len(trade_gates) != len(contracts.TRADE_GROUPS)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in trade_gates)
        ):
            raise ValueError("tradeGate has the wrong width or a non-integer value")
        share = _finite_number(action["tradeShare"], "tradeShare")
        if not 0.0 <= share <= 1.0:
            raise ValueError("tradeShare must be in [0, 1]")
        for name in ("movement", "fission"):
            value = action[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a scalar integer ordinal")
    return LaborPolicySample.from_payload(
        {name: action[name] for name in LABOR_RETRY_FIELDS}, catalog.masks
    )


class LaborRuntime(Protocol):
    """Typed bridge surface required by the bounded two-attempt collector."""

    def validate_labor(
        self, actions: Mapping[int, Mapping[str, Any]], *, attempt: int
    ) -> Mapping[int, LaborAttempt]: ...

    def step_labor(
        self,
        initial_actions: Mapping[int, Mapping[str, Any]],
        retry_actions: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[int, LaborFeedback]]: ...


@dataclass(frozen=True)
class RetainedLaborAction:
    """One submitted action plus the exact behavior samples used to score it."""

    payload: Mapping[str, Any]
    labor_sample: Mapping[str, torch.Tensor]
    behavior_labor_logp: float
    nonlabor_sample: Mapping[str, torch.Tensor] | None
    behavior_nonlabor_logp: float | None

    def validate(self, catalog: LaborCatalog, *, retry: bool) -> LaborPolicySample:
        decoded = validate_schema12_action(self.payload, catalog, retry=retry)
        if not math.isfinite(float(self.behavior_labor_logp)):
            raise ValueError("behavior labor log probability must be finite")
        required = {
            "specialty_allocation",
            "activity_allocation",
            "specialty_latent",
            "activity_latent",
            "specialty_mask",
            "activity_mask",
        }
        if set(self.labor_sample) != required:
            raise ValueError("retained labor sample fields are incomplete or unexpected")
        specialty = self.labor_sample["specialty_allocation"]
        activity = self.labor_sample["activity_allocation"]
        if (
            not isinstance(specialty, torch.Tensor)
            or not isinstance(activity, torch.Tensor)
            or specialty.shape != (1, len(decoded.specialty_allocation))
            or activity.shape
            != (
                1,
                len(decoded.activity_allocation),
                len(decoded.activity_allocation[0]),
            )
        ):
            raise ValueError("serialized labor action differs from its retained policy sample")
        # The wire contract canonicalizes one coordinate in every simplex so
        # float32 softmax roundoff cannot make an otherwise valid action miss
        # the strict native sum.  Compare the submitted payload with the
        # retained sample through that *same* declared canonicalizer.  The
        # retained latents and original float32 allocation remain untouched
        # and continue to define the behavior-policy likelihood.
        retained_wire = LaborPolicySample(
            tuple(float(value) for value in specialty[0].detach().cpu()),
            tuple(
                tuple(float(value) for value in row.detach().cpu())
                for row in activity[0]
            ),
        ).payload()
        retained_decoded = LaborPolicySample.from_payload(retained_wire, catalog.masks)
        if decoded != retained_decoded:
            raise ValueError("serialized labor action differs from its retained policy sample")
        if retry:
            if self.nonlabor_sample is not None or self.behavior_nonlabor_logp is not None:
                raise ValueError("retry action must not retain or score a non-labor sample")
        else:
            if not self.nonlabor_sample or self.behavior_nonlabor_logp is None:
                raise ValueError(
                    "initial action must retain its one non-labor sample and log probability"
                )
            if not math.isfinite(float(self.behavior_nonlabor_logp)):
                raise ValueError("behavior non-labor log probability must be finite")
        return decoded


RetryFactory = Callable[[int, LaborCatalog, LaborAttempt], RetainedLaborAction]


@dataclass(frozen=True)
class LaborDecisionExecution:
    """Typed result of one world-mutating decision after pure validation calls."""

    step: Mapping[str, Any]
    initial: Mapping[int, RetainedLaborAction]
    retry: Mapping[int, RetainedLaborAction]
    initial_attempts: Mapping[int, LaborAttempt]
    final_attempts: Mapping[int, LaborAttempt]
    feedback: Mapping[int, LaborFeedback]


def execute_labor_decision(
    runtime: LaborRuntime,
    catalogs: Mapping[int, LaborCatalog],
    initial: Mapping[int, RetainedLaborAction],
    retry_factory: RetryFactory,
) -> LaborDecisionExecution:
    """Execute exactly one bounded labor decision without hidden resampling.

    ``validate_labor`` is pure.  It is called first for the complete initial
    schema-12 action map and then, if needed, for labor-only retry payloads of
    rejected camps.  ``step_labor`` is the sole mutating call.  Scarcity and
    low production are never interpreted here as feasibility failures; only
    the producer's typed ``LaborAttempt.valid`` controls the retry.
    """

    if not catalogs or set(initial) != set(catalogs):
        raise ValueError("initial labor actions must cover exactly the current camps")
    for camp, action in initial.items():
        if catalog := catalogs.get(camp):
            if catalog.camp_id != camp:
                raise ValueError("labor catalog map key differs from its typed camp id")
            action.validate(catalog, retry=False)
        else:  # pragma: no cover - guarded by exact key equality
            raise ValueError("initial labor action lacks a catalog")
    initial_payloads = {camp: dict(action.payload) for camp, action in initial.items()}
    initial_attempts = dict(runtime.validate_labor(initial_payloads, attempt=0))
    if set(initial_attempts) != set(catalogs):
        raise ValueError(
            "initial labor validation did not return every submitted camp exactly once"
        )
    rejected = {}
    for camp, attempt in initial_attempts.items():
        if (
            not isinstance(attempt, LaborAttempt)
            or attempt.camp_id != camp
            or attempt.attempt != 0
            or attempt.catalog_digest != catalogs[camp].digest
        ):
            raise ValueError("initial validation returned a mismatched typed attempt")
        if not attempt.valid:
            if attempt.remaining_proposals != 1:
                raise ValueError("rejected first attempt must expose exactly one retry")
            rejected[camp] = retry_factory(camp, catalogs[camp], attempt)

    for camp, action in rejected.items():
        action.validate(catalogs[camp], retry=True)
    retry_payloads = {camp: dict(action.payload) for camp, action in rejected.items()}
    retry_attempts = (
        dict(runtime.validate_labor(retry_payloads, attempt=1)) if retry_payloads else {}
    )
    if set(retry_attempts) != set(rejected):
        raise ValueError("retry validation did not return exactly the rejected camps")
    for camp, attempt in retry_attempts.items():
        if (
            not isinstance(attempt, LaborAttempt)
            or attempt.camp_id != camp
            or attempt.attempt != 1
            or attempt.remaining_proposals != 0
            or attempt.catalog_digest != catalogs[camp].digest
        ):
            raise ValueError("retry validation returned a mismatched typed attempt")

    step, feedback = runtime.step_labor(
        initial_payloads, retry_payloads if retry_payloads else None
    )
    feedback = dict(feedback)
    if set(feedback) != set(catalogs):
        raise ValueError("labor execution feedback did not cover every submitted camp")
    final_attempts = {camp: retry_attempts.get(camp, initial_attempts[camp]) for camp in catalogs}
    for camp, row in feedback.items():
        if (
            not isinstance(row, LaborFeedback)
            or row.camp_id != camp
            or row.initial_request != initial_attempts[camp]
            or row.final_sampled_request != final_attempts[camp]
            or row.execution_comparator_attempt != final_attempts[camp].attempt
        ):
            raise ValueError("post-execution labor feedback differs from submitted attempts")
    return LaborDecisionExecution(
        step,
        dict(initial),
        rejected,
        initial_attempts,
        final_attempts,
        feedback,
    )


@dataclass(frozen=True)
class AttemptGAE:
    """GAE values for one initial edge and the optional retry edge."""

    initial_advantage: torch.Tensor
    initial_target: torch.Tensor
    retry_advantage: torch.Tensor
    retry_target: torch.Tensor


def two_attempt_gae(
    *,
    initial_value: torch.Tensor,
    retry_value: torch.Tensor,
    initial_reward: torch.Tensor,
    execution_reward: torch.Tensor,
    retry_mask: torch.Tensor,
    next_value: torch.Tensor,
    next_advantage: torch.Tensor,
    outer_discount: torch.Tensor,
    outer_continuation: torch.Tensor,
    lam: float,
) -> AttemptGAE:
    """Fold a bounded retry into GAE without duplicating the physical reward.

    For an accepted first proposal, ``initial_reward`` is the physical reward
    and ``execution_reward`` is ignored.  For a rejected first proposal,
    ``initial_reward`` is the zero-time edge reward (zero for task objectives,
    the typed rejection cost for the feasibility critic), and
    ``execution_reward`` belongs to the retry/execution edge.

    The internal edge has discount=1, continuation=1, and GAE lambda=1.  This
    is intentional: the original initial non-labor sample remains on that edge
    and receives the eventual outcome through the internal continuation.  A
    retry never acquires a second non-labor likelihood term.
    """

    n = len(initial_value)
    values = {
        "initial_value": initial_value,
        "retry_value": retry_value,
        "initial_reward": initial_reward,
        "execution_reward": execution_reward,
        "next_value": next_value,
        "next_advantage": next_advantage,
        "outer_discount": outer_discount,
        "outer_continuation": outer_continuation,
    }
    for name, value in values.items():
        _vector(value, n, name)
    _boolean_vector(retry_mask, n, "retry_mask")
    if not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be finite and in [0, 1]")
    if bool((outer_discount < 0).any()) or bool((outer_discount > 1).any()):
        raise ValueError("outer_discount must be in [0, 1]")
    if bool((outer_continuation < 0).any()) or bool((outer_continuation > 1).any()):
        raise ValueError("outer_continuation must be in [0, 1]")

    future = outer_discount * outer_continuation
    retry_delta = execution_reward + future * next_value - retry_value
    retry_advantage = retry_delta + future * lam * next_advantage
    direct_delta = initial_reward + future * next_value - initial_value
    direct_advantage = direct_delta + future * lam * next_advantage

    # On the zero-time edge gamma=lambda=continuation=1.  Keeping lambda=1
    # here makes the initial target independent of the arbitrary retry-state
    # critic decomposition when the execution return is fixed.
    internal_delta = initial_reward + retry_value - initial_value
    initial_advantage = torch.where(retry_mask, internal_delta + retry_advantage, direct_advantage)
    initial_target = initial_value + initial_advantage
    retry_target = retry_value + retry_advantage
    zeros = torch.zeros_like(retry_advantage)
    return AttemptGAE(
        initial_advantage,
        initial_target,
        torch.where(retry_mask, retry_advantage, zeros),
        torch.where(retry_mask, retry_target, zeros),
    )


def combine_labor_advantages(
    task: Mapping[str, torch.Tensor],
    costs: Mapping[str, torch.Tensor],
    *,
    personality: LaborPersonality,
    cost_multipliers: Mapping[str, float],
) -> torch.Tensor:
    """Combine task and cost advantages in their declared fixed units.

    Positive cost advantages are undesirable and therefore enter with a
    negative sign.  This is policy-gradient accounting, not a claim that a
    positive task GAE was independently beneficial.
    """

    if set(task) != set(LABOR_OBJECTIVES):
        raise ValueError(f"task advantages must be exactly {list(LABOR_OBJECTIVES)}")
    if set(costs) != set(LABOR_COSTS) or set(cost_multipliers) != set(LABOR_COSTS):
        raise ValueError(f"cost advantages/multipliers must be exactly {list(LABOR_COSTS)}")
    combined = personality.combine(task)
    for name in LABOR_COSTS:
        value = _vector(costs[name], len(combined), f"{name} advantage")
        multiplier = cost_multipliers[name]
        if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
            raise ValueError(f"{name} multiplier must be numeric")
        multiplier = float(multiplier)
        if not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError(f"{name} multiplier must be finite and nonnegative")
        combined = combined - multiplier * value
    return combined


def _clipped_surrogate(ratio: torch.Tensor, advantage: torch.Tensor, clip: float) -> torch.Tensor:
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * advantage
    return torch.minimum(unclipped, clipped)


@dataclass(frozen=True)
class AttemptPolicyTerms:
    """Loss and transparent edge accounting for one parent minibatch."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    initial_ratio: torch.Tensor
    retry_ratio: torch.Tensor
    parent_surrogate: torch.Tensor
    initial_joint_logp: torch.Tensor
    retry_labor_logp: torch.Tensor


def attempt_policy_terms(
    *,
    current_initial_labor_logp: torch.Tensor,
    behaviour_initial_labor_logp: torch.Tensor,
    current_initial_nonlabor_logp: torch.Tensor,
    behaviour_initial_nonlabor_logp: torch.Tensor,
    initial_advantage: torch.Tensor,
    retry_parent: torch.Tensor,
    current_retry_labor_logp: torch.Tensor,
    behaviour_retry_labor_logp: torch.Tensor,
    retry_advantage: torch.Tensor,
    initial_labor_entropy: torch.Tensor,
    initial_nonlabor_entropy: torch.Tensor,
    retry_labor_entropy: torch.Tensor,
    clip: float,
    entropy_weight: float,
) -> AttemptPolicyTerms:
    """Build the actual two-edge PPO objective with a parent denominator.

    The initial edge contains labor plus the one retained non-labor sample.
    ``retry_parent`` maps each retry row to a distinct initial parent; retry
    rows contain labor likelihood only.  Surrogates and entropies are summed
    within parent and then averaged over initial parents, so retried parents do
    not turn into extra pseudo-independent rows.
    """

    n = len(current_initial_labor_logp)
    initial_values = {
        "behaviour_initial_labor_logp": behaviour_initial_labor_logp,
        "current_initial_nonlabor_logp": current_initial_nonlabor_logp,
        "behaviour_initial_nonlabor_logp": behaviour_initial_nonlabor_logp,
        "initial_advantage": initial_advantage,
        "initial_labor_entropy": initial_labor_entropy,
        "initial_nonlabor_entropy": initial_nonlabor_entropy,
    }
    _vector(current_initial_labor_logp, n, "current_initial_labor_logp")
    for name, value in initial_values.items():
        _vector(value, n, name)
    if not math.isfinite(clip) or not 0.0 < clip < 1.0:
        raise ValueError("clip must be finite and in (0, 1)")
    if not math.isfinite(entropy_weight) or entropy_weight < 0:
        raise ValueError("entropy_weight must be finite and nonnegative")

    if (
        not isinstance(retry_parent, torch.Tensor)
        or retry_parent.dtype != torch.long
        or retry_parent.ndim != 1
    ):
        raise ValueError("retry_parent must be a one-dimensional int64 tensor")
    m = len(retry_parent)
    for name, value in {
        "current_retry_labor_logp": current_retry_labor_logp,
        "behaviour_retry_labor_logp": behaviour_retry_labor_logp,
        "retry_advantage": retry_advantage,
        "retry_labor_entropy": retry_labor_entropy,
    }.items():
        _vector(value, m, name)
    if m and (
        bool((retry_parent < 0).any())
        or bool((retry_parent >= n).any())
        or retry_parent.unique().numel() != m
    ):
        raise ValueError("retry_parent must contain distinct valid parent indices")

    initial_joint = current_initial_labor_logp + current_initial_nonlabor_logp
    behaviour_initial_joint = behaviour_initial_labor_logp + behaviour_initial_nonlabor_logp
    initial_ratio = torch.exp(initial_joint - behaviour_initial_joint)
    retry_ratio = torch.exp(current_retry_labor_logp - behaviour_retry_labor_logp)
    if not bool(torch.isfinite(initial_ratio).all()) or not bool(torch.isfinite(retry_ratio).all()):
        raise FloatingPointError("non-finite labor PPO ratio")

    parent_surrogate = _clipped_surrogate(initial_ratio, initial_advantage, clip)
    parent_entropy = initial_labor_entropy + initial_nonlabor_entropy
    if m:
        retry_surrogate = _clipped_surrogate(retry_ratio, retry_advantage, clip)
        parent_surrogate = parent_surrogate.index_add(0, retry_parent, retry_surrogate)
        parent_entropy = parent_entropy.index_add(0, retry_parent, retry_labor_entropy)
    policy_loss = -parent_surrogate.mean()
    entropy = parent_entropy.mean()
    return AttemptPolicyTerms(
        policy_loss - entropy_weight * entropy,
        policy_loss,
        entropy,
        initial_ratio,
        retry_ratio,
        parent_surrogate,
        initial_joint,
        current_retry_labor_logp,
    )


def score_retained_labor_samples(
    actor: torch.nn.Module,
    initial_tensors: Mapping[str, torch.Tensor],
    initial_sample: Mapping[str, torch.Tensor],
    retry_tensors: Mapping[str, torch.Tensor],
    retry_sample: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Score retained initial/retry labor samples without assuming registry dimensions."""

    initial_output = actor(initial_tensors)
    retry_output = actor(retry_tensors)
    initial_logp = labor_log_prob(initial_output, initial_sample, initial_tensors)
    retry_logp = labor_log_prob(retry_output, retry_sample, retry_tensors)
    initial_entropy = labor_entropy(initial_output, initial_tensors)["joint"]
    retry_entropy = labor_entropy(retry_output, retry_tensors)["joint"]
    return initial_logp, retry_logp, initial_entropy, retry_entropy


def score_labor_sample(
    actor: torch.nn.Module,
    tensors: Mapping[str, torch.Tensor],
    sample: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return joint retained-sample log probability and latent entropy."""

    output = actor(tensors)
    return labor_log_prob(output, sample, tensors)["joint"], labor_entropy(output, tensors)["joint"]


def _rows(value: Mapping[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
    result = {}
    for name, tensor in value.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
            raise ValueError(f"{name} must be a row-aligned tensor")
        result[name] = tensor.index_select(0, index.to(tensor.device))
    return result


NonlaborScorer = Callable[
    [Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]],
    tuple[torch.Tensor, torch.Tensor],
]


@dataclass(frozen=True)
class LaborActorGate:
    """Strict current-policy gate binding for one frozen-behavior update.

    A separate labor module does not bypass observed-return calibration.  The
    gate report must cover both task heads and both cost heads and must bind the
    complete sampled policy (labor and enabled non-labor modules plus behavior
    configuration).  It remains valid through all minibatches of this one PPO
    update and becomes stale before the next rollout if any actor step lands.
    """

    source_policy_digest: str
    source_critic_digest: str
    normalization_binding_digest: str
    data_sha256: str

    @classmethod
    def from_report(cls, report: Mapping[str, object]) -> LaborActorGate:
        if (
            report.get("kind") != "model9-labor-value-calibration-report-v1"
            or report.get("localReadinessClaim") is not False
        ):
            raise ValueError("labor actor gate requires the independent-return calibration report")
        if report.get("dataRole") != "actor-gate":
            raise ValueError("labor actor gate data role must be actor-gate")
        if (
            report.get("actorGateOpen") is not True
            or report.get("policyCurrent") is not True
            or report.get("normalizationCurrent") is not True
            or report.get("registryCurrent") is not True
        ):
            raise ValueError("labor actor gate is closed or stale")
        enabled = report.get("enabledHeads")
        if not isinstance(enabled, list) or set(enabled) != set(LABOR_GATE_HEADS):
            raise ValueError(f"labor actor gate must cover exactly {list(LABOR_GATE_HEADS)}")
        heads = report.get("heads")
        if not isinstance(heads, Mapping) or any(
            not isinstance(heads.get(name), Mapping) or heads[name].get("pass") is not True
            for name in LABOR_GATE_HEADS
        ):
            raise ValueError("labor actor gate has an uncalibrated required head")
        source = report.get("sourcePolicyDigest")
        source_critic = report.get("sourceCriticDigest")
        normalization = report.get("normalizationBindingDigest")
        data_sha = report.get("dataSha256")
        for value, label in (
            (source, "sourcePolicyDigest"),
            (source_critic, "sourceCriticDigest"),
            (normalization, "normalizationBindingDigest"),
            (data_sha, "dataSha256"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"labor actor gate {label} must be a lowercase SHA-256")
        return cls(source, source_critic, normalization, data_sha)

    def require_current(
        self,
        policy_module: torch.nn.Module,
        behavior_configuration: Mapping[str, object],
        registry_digest: str,
        rollout_critic_digest: str,
        normalization_binding_digest: str,
    ) -> str:
        from .value_calibration import behavior_policy_digest

        if behavior_configuration.get("laborRegistryDigest") != registry_digest:
            raise ValueError("labor gate behavior configuration has another registry")
        current = behavior_policy_digest(policy_module, behavior_configuration)
        if current != self.source_policy_digest:
            raise ValueError("labor actor gate is stale for the current sampled policy")
        if rollout_critic_digest != self.source_critic_digest:
            raise ValueError(
                "labor actor gate was evaluated against other rollout-advantage critic weights"
            )
        if normalization_binding_digest != self.normalization_binding_digest:
            raise ValueError(
                "labor actor gate was evaluated under another input normalization binding"
            )
        return current


@dataclass(frozen=True)
class LaborAttemptPolicyBatch:
    """Retained behavior samples for a bounded two-attempt PPO update.

    ``initial_nonlabor_sample`` exists only for initial rows.  Retry rows have
    no corresponding field, making duplicate non-labor likelihood structurally
    unavailable to the update routine.
    """

    initial_tensors: Mapping[str, torch.Tensor]
    initial_labor_sample: Mapping[str, torch.Tensor]
    initial_nonlabor_sample: Mapping[str, torch.Tensor]
    behaviour_initial_labor_logp: torch.Tensor
    behaviour_initial_nonlabor_logp: torch.Tensor
    initial_advantage: torch.Tensor
    retry_parent: torch.Tensor
    retry_tensors: Mapping[str, torch.Tensor]
    retry_labor_sample: Mapping[str, torch.Tensor]
    behaviour_retry_labor_logp: torch.Tensor
    retry_advantage: torch.Tensor
    registry_digest: str
    behavior_policy_digest: str
    advantage_critic_digest: str
    normalization_binding_digest: str

    def __post_init__(self) -> None:
        n = len(self.behaviour_initial_labor_logp)
        _vector(self.behaviour_initial_labor_logp, n, "behaviour_initial_labor_logp")
        _vector(self.behaviour_initial_nonlabor_logp, n, "behaviour_initial_nonlabor_logp")
        _vector(self.initial_advantage, n, "initial_advantage")
        for label, value in (
            ("registry_digest", self.registry_digest),
            ("behavior_policy_digest", self.behavior_policy_digest),
            ("advantage_critic_digest", self.advantage_critic_digest),
            ("normalization_binding_digest", self.normalization_binding_digest),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{label} must be a SHA-256 digest")
            if any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{label} must be lowercase hexadecimal")
        for label, values in (
            ("initial_tensors", self.initial_tensors),
            ("initial_labor_sample", self.initial_labor_sample),
            ("initial_nonlabor_sample", self.initial_nonlabor_sample),
        ):
            if not values or any(
                not isinstance(value, torch.Tensor) or value.shape[0] != n or value.requires_grad
                for value in values.values()
            ):
                raise ValueError(f"{label} must contain {n} row-aligned tensors")
        if (
            not isinstance(self.retry_parent, torch.Tensor)
            or self.retry_parent.dtype != torch.long
            or self.retry_parent.ndim != 1
        ):
            raise ValueError("retry_parent must be a one-dimensional int64 tensor")
        m = len(self.retry_parent)
        _vector(self.behaviour_retry_labor_logp, m, "behaviour_retry_labor_logp")
        _vector(self.retry_advantage, m, "retry_advantage")
        if m and (
            bool((self.retry_parent < 0).any())
            or bool((self.retry_parent >= n).any())
            or self.retry_parent.unique().numel() != m
        ):
            raise ValueError("retry_parent must contain distinct valid parent indices")
        for label, values in (
            ("retry_tensors", self.retry_tensors),
            ("retry_labor_sample", self.retry_labor_sample),
        ):
            if not values or any(
                not isinstance(value, torch.Tensor) or value.shape[0] != m or value.requires_grad
                for value in values.values()
            ):
                raise ValueError(f"{label} must contain {m} row-aligned tensors")

    @property
    def parent_count(self) -> int:
        return len(self.initial_advantage)


def _retry_rows_for_parents(
    retry_parent: torch.Tensor, parent_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return retry rows and their local parent indices for a parent minibatch."""

    lookup = {int(parent): local for local, parent in enumerate(parent_index.tolist())}
    retry_rows = [row for row, parent in enumerate(retry_parent.tolist()) if parent in lookup]
    local_parent = [lookup[int(retry_parent[row])] for row in retry_rows]
    return torch.tensor(retry_rows, dtype=torch.long), torch.tensor(local_parent, dtype=torch.long)


def labor_policy_minibatch_terms(
    actor: torch.nn.Module,
    batch: LaborAttemptPolicyBatch,
    parent_index: torch.Tensor,
    nonlabor_scorer: NonlaborScorer,
    *,
    clip: float,
    entropy_weight: float,
) -> AttemptPolicyTerms:
    """Recompute one parent minibatch under the current policy."""

    if (
        not isinstance(parent_index, torch.Tensor)
        or parent_index.dtype != torch.long
        or parent_index.ndim != 1
        or not len(parent_index)
        or parent_index.unique().numel() != len(parent_index)
        or bool((parent_index < 0).any())
        or bool((parent_index >= batch.parent_count).any())
    ):
        raise ValueError("parent_index must contain distinct valid initial rows")
    initial_tensors = _rows(batch.initial_tensors, parent_index)
    initial_labor_sample = _rows(batch.initial_labor_sample, parent_index)
    initial_nonlabor_sample = _rows(batch.initial_nonlabor_sample, parent_index)
    initial_labor_logp, initial_labor_entropy = score_labor_sample(
        actor, initial_tensors, initial_labor_sample
    )
    initial_nonlabor_logp, initial_nonlabor_entropy = nonlabor_scorer(
        initial_tensors, initial_nonlabor_sample
    )
    retry_rows, local_parent = _retry_rows_for_parents(batch.retry_parent, parent_index)
    if len(retry_rows):
        retry_tensors = _rows(batch.retry_tensors, retry_rows)
        retry_sample = _rows(batch.retry_labor_sample, retry_rows)
        retry_logp, retry_entropy = score_labor_sample(actor, retry_tensors, retry_sample)
    else:
        retry_logp = batch.behaviour_retry_labor_logp.new_empty(0)
        retry_entropy = retry_logp.clone()
    return attempt_policy_terms(
        current_initial_labor_logp=initial_labor_logp,
        behaviour_initial_labor_logp=batch.behaviour_initial_labor_logp.index_select(
            0, parent_index
        ),
        current_initial_nonlabor_logp=initial_nonlabor_logp,
        behaviour_initial_nonlabor_logp=batch.behaviour_initial_nonlabor_logp.index_select(
            0, parent_index
        ),
        initial_advantage=batch.initial_advantage.index_select(0, parent_index),
        retry_parent=local_parent,
        current_retry_labor_logp=retry_logp,
        behaviour_retry_labor_logp=batch.behaviour_retry_labor_logp.index_select(0, retry_rows),
        retry_advantage=batch.retry_advantage.index_select(0, retry_rows),
        initial_labor_entropy=initial_labor_entropy,
        initial_nonlabor_entropy=initial_nonlabor_entropy,
        retry_labor_entropy=retry_entropy,
        clip=clip,
        entropy_weight=entropy_weight,
    )


def _audit_nonnegative_kl(terms: AttemptPolicyTerms) -> torch.Tensor:
    initial_log_ratio = torch.log(terms.initial_ratio)
    parent = torch.expm1(initial_log_ratio) - initial_log_ratio
    if len(terms.retry_ratio):
        retry_log_ratio = torch.log(terms.retry_ratio)
        retry = torch.expm1(retry_log_ratio) - retry_log_ratio
        # This audit receives already-local retry parent mappings only inside
        # AttemptPolicyTerms' parent surrogate.  The mean of edge estimates is
        # diagnostic and conservative for rollback; it is not a population KL.
        return torch.cat([parent, retry]).mean()
    return parent.mean()


def update_labor_policy(
    *,
    policy_module: torch.nn.Module,
    labor_actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: LaborAttemptPolicyBatch,
    nonlabor_scorer: NonlaborScorer,
    rng: np.random.Generator,
    epochs: int,
    batch_size: int,
    clip: float,
    entropy_weight: float,
    max_grad_norm: float,
    target_kl: float,
    audit_rows: int,
    actor_gate: LaborActorGate,
    behavior_configuration: Mapping[str, object],
) -> dict[str, object]:
    """Run one frozen-behavior PPO update with transactional KL rollback.

    Each epoch uses one permutation of parents exactly once.  A rejected
    minibatch step restores both policy tensors/buffers and Adam state.  The
    fixed audit is deterministically spread across the full parent rollout;
    its sampled-action estimate is step control, not full-policy KL evidence.
    """

    gate_policy_digest = actor_gate.require_current(
        policy_module,
        behavior_configuration,
        batch.registry_digest,
        batch.advantage_critic_digest,
        batch.normalization_binding_digest,
    )
    if gate_policy_digest != batch.behavior_policy_digest:
        raise ValueError("labor PPO batch was sampled by another behavior policy")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for value, label in (
        (max_grad_norm, "max_grad_norm"),
        (target_kl, "target_kl"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and positive")
    audit = torch.as_tensor(fixed_audit_indices(batch.parent_count, audit_rows), dtype=torch.long)
    coverage = []
    losses = []
    kl_values = []
    rolled_back = False
    attempted_steps = 0
    accepted_steps = 0
    for _ in range(epochs):
        partitions = list(epoch_minibatches(batch.parent_count, batch_size, rng))
        processed: list[np.ndarray] = []
        accepted: list[np.ndarray] = []
        for raw_index in partitions:
            processed.append(raw_index)
            parent_index = torch.as_tensor(raw_index, dtype=torch.long)
            snapshot = capture_step_state(policy_module, optimizer)
            terms = labor_policy_minibatch_terms(
                labor_actor,
                batch,
                parent_index,
                nonlabor_scorer,
                clip=clip,
                entropy_weight=entropy_weight,
            )
            optimizer.zero_grad()
            terms.loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_module.parameters(), max_grad_norm)
            optimizer.step()
            attempted_steps += 1
            with torch.no_grad():
                audit_terms = labor_policy_minibatch_terms(
                    labor_actor,
                    batch,
                    audit,
                    nonlabor_scorer,
                    clip=clip,
                    entropy_weight=entropy_weight,
                )
                kl = float(_audit_nonnegative_kl(audit_terms))
            if not math.isfinite(kl) or kl > target_kl:
                restore_step_state(policy_module, optimizer, snapshot)
                rolled_back = True
                kl_values.append(kl)
                break
            accepted_steps += 1
            accepted.append(raw_index)
            kl_values.append(kl)
            losses.append(float(terms.loss.detach()))
        coverage.append(
            {
                "planned": epoch_coverage(partitions, batch.parent_count),
                "processed": epoch_coverage(processed, batch.parent_count),
                "accepted": epoch_coverage(accepted, batch.parent_count),
            }
        )
        if rolled_back:
            break
    return {
        "contract": LABOR_PPO_CONTRACT,
        "registryDigest": batch.registry_digest,
        "parentRows": batch.parent_count,
        "retryRows": len(batch.retry_parent),
        "attemptedSteps": attempted_steps,
        "acceptedSteps": accepted_steps,
        "actorStepRolledBack": rolled_back,
        "losses": losses,
        "fixedAuditRows": audit.tolist(),
        "sampledNonnegativeKL": kl_values,
        "epochCoverage": coverage,
        "actorGate": {
            "sourcePolicyDigest": gate_policy_digest,
            "dataSha256": actor_gate.data_sha256,
            "validFor": "this-frozen-behavior-update-only",
            "nextRolloutStatus": "stale-refresh-required" if accepted_steps else "unchanged",
        },
        "scope": "frozen-behavior-update; sampled-action KL is not full-policy KL",
    }


def labor_task_value_loss(
    critic: torch.nn.Module,
    tensors: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Mean two-hot cross entropy across the two declared task heads."""

    if set(targets) != set(LABOR_OBJECTIVES):
        raise ValueError(f"task targets must be exactly {list(LABOR_OBJECTIVES)}")
    output = critic(tensors)
    if set(output) != set(LABOR_OBJECTIVES):
        raise ValueError("labor task critic heads differ from its contract")
    losses = []
    for name in LABOR_OBJECTIVES:
        value = targets[name]
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise ValueError(f"{name} target must be a vector")
        distribution = critic.two_hot(name, value)
        losses.append(-(distribution * torch.log_softmax(output[name], dim=-1)).sum(dim=-1))
    return torch.stack(losses, dim=0).mean()


def labor_cost_value_loss(
    critics: torch.nn.Module,
    tensors: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Mean squared error across distinct feasibility and guard critics."""

    if set(targets) != set(LABOR_COSTS):
        raise ValueError(f"cost targets must be exactly {list(LABOR_COSTS)}")
    output = critics(tensors)
    if set(output) != set(LABOR_COSTS):
        raise ValueError("labor cost critic heads differ from its contract")
    losses = []
    for name in LABOR_COSTS:
        target = _vector(targets[name], len(output[name]), f"{name} target")
        losses.append((output[name] - target).square())
    return torch.stack(losses, dim=0).mean()
