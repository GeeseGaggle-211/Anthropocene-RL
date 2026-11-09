from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from torch import nn

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7.labor_bridge import (
    ACTIVITIES,
    GOODS_ROWS,
    RESOURCE_ROWS,
    SPECIALTIES,
    LaborAttempt,
    LaborCatalog,
    LaborFeedback,
    LaborMasks,
    LaborPolicySample,
    LaborViolation,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    LABOR_COSTS,
    LABOR_OBJECTIVES,
    LaborPersonality,
    labor_log_prob,
    sample_labor_policy,
)
from anthropocene_rl.autonomous_v7.labor_ppo import (
    LABOR_GATE_HEADS,
    LABOR_RETRY_FIELDS,
    LaborActorGate,
    LaborAttemptPolicyBatch,
    RetainedLaborAction,
    attempt_policy_terms,
    combine_labor_advantages,
    execute_labor_decision,
    score_retained_labor_samples,
    two_attempt_gae,
    update_labor_policy,
    validate_schema12_action,
)
from anthropocene_rl.autonomous_v7.value_calibration import behavior_policy_digest


def test_two_attempt_gae_uses_unit_internal_edge_and_one_physical_reward() -> None:
    common = {
        "initial_value": torch.tensor([10.0, 10.0]),
        "initial_reward": torch.tensor([0.0, 4.0]),
        "execution_reward": torch.tensor([4.0, 999.0]),
        "retry_mask": torch.tensor([True, False]),
        "next_value": torch.tensor([2.0, 2.0]),
        "next_advantage": torch.tensor([3.0, 3.0]),
        "outer_discount": torch.tensor([0.5, 0.5]),
        "outer_continuation": torch.ones(2),
        "lam": 0.9,
    }
    first = two_attempt_gae(retry_value=torch.tensor([7.0, 0.0]), **common)
    shifted = two_attempt_gae(retry_value=torch.tensor([9.0, 0.0]), **common)

    assert first.retry_advantage.tolist() == pytest.approx([-0.65, 0.0])
    assert first.initial_advantage.tolist() == pytest.approx([-3.65, -3.65])
    # The retry-state value only partitions the same return across two TD
    # errors; unit discount/lambda makes the initial return invariant to it.
    assert shifted.initial_advantage.tolist() == pytest.approx(first.initial_advantage.tolist())
    # The unused execution reward on an accepted initial edge is ignored.
    assert first.initial_target[1].item() == pytest.approx(6.35)


def test_attempt_policy_terms_scores_nonlabor_once_and_uses_parent_denominator() -> None:
    zero_initial = torch.zeros(2)
    zero_retry = torch.zeros(1)
    terms = attempt_policy_terms(
        current_initial_labor_logp=zero_initial,
        behaviour_initial_labor_logp=zero_initial,
        current_initial_nonlabor_logp=zero_initial,
        behaviour_initial_nonlabor_logp=zero_initial,
        initial_advantage=torch.tensor([2.0, 3.0]),
        retry_parent=torch.tensor([0]),
        current_retry_labor_logp=zero_retry,
        behaviour_retry_labor_logp=zero_retry,
        retry_advantage=torch.tensor([5.0]),
        initial_labor_entropy=zero_initial,
        initial_nonlabor_entropy=zero_initial,
        retry_labor_entropy=zero_retry,
        clip=0.2,
        entropy_weight=0.0,
    )
    assert terms.parent_surrogate.tolist() == pytest.approx([7.0, 3.0])
    assert terms.policy_loss.item() == pytest.approx(-5.0)

    # Changing the retained initial non-labor likelihood changes the initial
    # ratio.  There is intentionally no retry non-labor argument to duplicate
    # or move this score onto the resampled edge.
    changed = attempt_policy_terms(
        current_initial_labor_logp=zero_initial,
        behaviour_initial_labor_logp=zero_initial,
        current_initial_nonlabor_logp=torch.tensor([math.log(1.1), 0.0]),
        behaviour_initial_nonlabor_logp=zero_initial,
        initial_advantage=torch.tensor([2.0, 3.0]),
        retry_parent=torch.tensor([0]),
        current_retry_labor_logp=zero_retry,
        behaviour_retry_labor_logp=zero_retry,
        retry_advantage=torch.tensor([5.0]),
        initial_labor_entropy=zero_initial,
        initial_nonlabor_entropy=zero_initial,
        retry_labor_entropy=zero_retry,
        clip=0.2,
        entropy_weight=0.0,
    )
    assert changed.initial_ratio.tolist() == pytest.approx([1.1, 1.0])
    assert changed.retry_ratio.tolist() == pytest.approx([1.0])
    assert changed.parent_surrogate.tolist() == pytest.approx([7.2, 3.0])


def test_audit_kl_is_finite_when_the_ratio_underflows() -> None:
    """D7 real-rollout regression: an executed action whose current log-prob fell
    by thousands of nats (a tight ALR Gaussian after one Adam step) has ratio
    exp(x) == 0 in float32. The audit must use x itself, not log(0) = -inf."""

    from anthropocene_rl.autonomous_v7.labor_ppo import _audit_nonnegative_kl

    def terms(initial_delta: list[float], retry_delta: list[float]):
        initial = torch.tensor(initial_delta)
        retry = torch.tensor(retry_delta)
        zero_initial, zero_retry = torch.zeros(len(initial)), torch.zeros(len(retry))
        return attempt_policy_terms(
            current_initial_labor_logp=initial,
            behaviour_initial_labor_logp=zero_initial,
            current_initial_nonlabor_logp=zero_initial,
            behaviour_initial_nonlabor_logp=zero_initial,
            initial_advantage=torch.tensor([1.0, -1.0, 0.5]),
            retry_parent=torch.arange(len(retry)),
            current_retry_labor_logp=retry,
            behaviour_retry_labor_logp=zero_retry,
            retry_advantage=torch.ones(len(retry)),
            initial_labor_entropy=zero_initial,
            initial_nonlabor_entropy=zero_initial,
            retry_labor_entropy=zero_retry,
            clip=0.2,
            entropy_weight=0.0,
        )

    # Before any step the policy is its own behaviour: exactly zero.
    assert _audit_nonnegative_kl(terms([0.0, 0.0, 0.0], [0.0])).item() == 0.0
    # The real-rollout pattern: ratios underflow to 0 on the initial and retry edges.
    underflow = terms([-5635.6, -0.25, 0.1], [-300.0])
    assert underflow.initial_ratio[0].item() == 0.0 and underflow.retry_ratio[0].item() == 0.0
    kl = _audit_nonnegative_kl(underflow)
    expected = [math.expm1(x) - x for x in (-5635.6, -0.25, 0.1, -300.0)]
    assert math.isfinite(kl.item())
    assert kl.item() == pytest.approx(sum(expected) / 4, rel=1e-5)
    # The surrogate still sees the exponentiated (underflowed) ratio.
    assert math.isfinite(underflow.policy_loss.item())


def test_attempt_policy_terms_rejects_retry_pseudoreplication() -> None:
    initial = torch.zeros(2)
    retry = torch.zeros(2)
    with pytest.raises(ValueError, match="distinct"):
        attempt_policy_terms(
            current_initial_labor_logp=initial,
            behaviour_initial_labor_logp=initial,
            current_initial_nonlabor_logp=initial,
            behaviour_initial_nonlabor_logp=initial,
            initial_advantage=initial,
            retry_parent=torch.tensor([0, 0]),
            current_retry_labor_logp=retry,
            behaviour_retry_labor_logp=retry,
            retry_advantage=retry,
            initial_labor_entropy=initial,
            initial_nonlabor_entropy=initial,
            retry_labor_entropy=retry,
            clip=0.2,
            entropy_weight=0.0,
        )


def test_combined_advantage_uses_fixed_task_scales_and_negative_costs() -> None:
    task = {name: torch.ones(2) * (index + 1) for index, name in enumerate(LABOR_OBJECTIVES)}
    costs = {name: torch.ones(2) * (index + 3) for index, name in enumerate(LABOR_COSTS)}
    combined = combine_labor_advantages(
        task,
        costs,
        personality=LaborPersonality(),
        cost_multipliers={"feasibility": 0.25, "settlement_guard": 0.5},
    )
    expected_task = LaborPersonality().combine(task)
    assert torch.allclose(
        combined, expected_task - 0.25 * costs["feasibility"] - 0.5 * costs["settlement_guard"]
    )


class _DynamicActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        specialty = tensors["labor_specialty_mask"]
        activity = tensors["labor_activity_mask"]
        specialty_coordinate = torch.arange(
            specialty.shape[1], dtype=torch.float32, device=specialty.device
        )
        activity_coordinate = torch.arange(
            activity.shape[2], dtype=torch.float32, device=activity.device
        )
        return {
            "specialty_logits": torch.zeros_like(specialty, dtype=torch.float32)
            + self.shift * specialty_coordinate,
            "specialty_log_std": torch.zeros_like(specialty, dtype=torch.float32),
            "activity_logits": torch.zeros_like(activity, dtype=torch.float32)
            + self.shift * activity_coordinate,
            "activity_log_std": torch.zeros_like(activity, dtype=torch.float32),
        }


def _dynamic_tensors(batch: int, specialties: int, activities: int) -> dict[str, torch.Tensor]:
    specialty = torch.ones(batch, specialties, dtype=torch.bool)
    activity = torch.ones(batch, specialties - 1, activities, dtype=torch.bool)
    # Exercise deterministic idle-only rows without assuming 13x21.
    activity[:, -1, 1:] = False
    return {
        "labor_specialty_mask": specialty,
        "labor_activity_mask": activity,
        "labor_specialty_present": torch.ones_like(specialty),
        "labor_activity_present": torch.ones(
            batch, (specialties - 1) * activities, dtype=torch.bool
        ),
    }


def test_retained_labor_scoring_derives_dimensions_from_runtime_masks() -> None:
    actor = _DynamicActor()
    initial_tensors = _dynamic_tensors(3, specialties=5, activities=4)
    retry_tensors = _dynamic_tensors(1, specialties=5, activities=4)
    initial_output = actor(initial_tensors)
    retry_output = actor(retry_tensors)
    initial_sample = sample_labor_policy(
        initial_output, initial_tensors, generator=torch.Generator().manual_seed(11)
    )
    retry_sample = sample_labor_policy(
        retry_output, retry_tensors, generator=torch.Generator().manual_seed(12)
    )
    scored = score_retained_labor_samples(
        actor, initial_tensors, initial_sample, retry_tensors, retry_sample
    )
    initial_logp, retry_logp, initial_entropy, retry_entropy = scored
    assert initial_logp["joint"].shape == (3,)
    assert retry_logp["joint"].shape == (1,)
    assert initial_entropy.shape == (3,)
    assert retry_entropy.shape == (1,)
    assert torch.allclose(
        initial_logp["joint"],
        labor_log_prob(initial_output, initial_sample, initial_tensors)["joint"],
    )
    # Last role is idle-only and therefore adds exactly zero to the activity
    # density regardless of how many registry coordinates precede it.
    assert initial_sample["activity_latent"][:, -1].count_nonzero().item() == 0


class _NonlaborPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(0.0))

    def score(
        self, _tensors: dict[str, torch.Tensor], sample: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = sample["z"]
        logp = -0.5 * (z - self.mean).square() - 0.5 * math.log(2 * math.pi)
        return logp, torch.full_like(logp, 0.5 * math.log(2 * math.pi * math.e))


def _policy_batch(actor: _DynamicActor, nonlabor: _NonlaborPolicy) -> LaborAttemptPolicyBatch:
    initial_tensors = _dynamic_tensors(4, specialties=5, activities=4)
    retry_tensors = _dynamic_tensors(2, specialties=5, activities=4)
    nonlabor_sample = {"z": torch.tensor([0.2, -0.3, 0.4, -0.1])}
    with torch.no_grad():
        initial_sample = sample_labor_policy(
            actor(initial_tensors), initial_tensors, generator=torch.Generator().manual_seed(21)
        )
        retry_sample = sample_labor_policy(
            actor(retry_tensors), retry_tensors, generator=torch.Generator().manual_seed(22)
        )
        initial_logp = labor_log_prob(actor(initial_tensors), initial_sample, initial_tensors)[
            "joint"
        ]
        retry_logp = labor_log_prob(actor(retry_tensors), retry_sample, retry_tensors)["joint"]
        nonlabor_logp = nonlabor.score(initial_tensors, nonlabor_sample)[0]
    return LaborAttemptPolicyBatch(
        initial_tensors=initial_tensors,
        initial_labor_sample=initial_sample,
        initial_nonlabor_sample=nonlabor_sample,
        behaviour_initial_labor_logp=initial_logp,
        behaviour_initial_nonlabor_logp=nonlabor_logp,
        initial_advantage=torch.tensor([1.0, -0.5, 0.7, 1.2]),
        retry_parent=torch.tensor([1, 3]),
        retry_tensors=retry_tensors,
        retry_labor_sample=retry_sample,
        behaviour_retry_labor_logp=retry_logp,
        retry_advantage=torch.tensor([-0.8, 0.9]),
        registry_digest="a" * 64,
        behavior_policy_digest="e" * 64,
        advantage_critic_digest="c" * 64,
        normalization_binding_digest="d" * 64,
    )


def _gate(policy: nn.Module, behavior: dict[str, object]) -> LaborActorGate:
    policy_sha = behavior_policy_digest(policy, behavior)
    return LaborActorGate.from_report(
        {
            "kind": "model9-labor-value-calibration-report-v1",
            "localReadinessClaim": False,
            "dataRole": "actor-gate",
            "actorGateOpen": True,
            "policyCurrent": True,
            "normalizationCurrent": True,
            "registryCurrent": True,
            "enabledHeads": list(LABOR_GATE_HEADS),
            "heads": {name: {"pass": True} for name in LABOR_GATE_HEADS},
            "sourcePolicyDigest": policy_sha,
            "sourceCriticDigest": "c" * 64,
            "normalizationBindingDigest": "d" * 64,
            "dataSha256": "b" * 64,
        }
    )


def test_labor_actor_gate_requires_all_four_heads_and_current_complete_policy() -> None:
    actor = _DynamicActor()
    nonlabor = _NonlaborPolicy()
    policy = nn.ModuleDict({"labor": actor, "nonlabor": nonlabor})
    behavior = {"laborRegistryDigest": "a" * 64, "logStdFloor": -2.0}
    report = {
        "kind": "model9-labor-value-calibration-report-v1",
        "localReadinessClaim": False,
        "dataRole": "actor-gate",
        "actorGateOpen": True,
        "policyCurrent": True,
        "normalizationCurrent": True,
        "registryCurrent": True,
        "enabledHeads": list(LABOR_GATE_HEADS),
        "heads": {name: {"pass": True} for name in LABOR_GATE_HEADS},
        "sourcePolicyDigest": behavior_policy_digest(policy, behavior),
        "sourceCriticDigest": "c" * 64,
        "normalizationBindingDigest": "d" * 64,
        "dataSha256": "b" * 64,
    }
    missing = dict(report)
    missing["enabledHeads"] = list(LABOR_GATE_HEADS[:-1])
    with pytest.raises(ValueError, match="cover exactly"):
        LaborActorGate.from_report(missing)
    gate = LaborActorGate.from_report(report)
    with torch.no_grad():
        nonlabor.mean.add_(0.1)
    with pytest.raises(ValueError, match="stale"):
        gate.require_current(policy, behavior, "a" * 64, "c" * 64, "d" * 64)


def test_labor_update_covers_each_parent_once_per_epoch_and_updates_both_policies() -> None:
    actor = _DynamicActor()
    nonlabor = _NonlaborPolicy()
    policy = nn.ModuleDict({"labor": actor, "nonlabor": nonlabor})
    batch = _policy_batch(actor, nonlabor)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    behavior = {"laborRegistryDigest": batch.registry_digest, "logStdFloor": -2.0}
    batch = dataclasses.replace(
        batch, behavior_policy_digest=behavior_policy_digest(policy, behavior)
    )
    stats = update_labor_policy(
        policy_module=policy,
        labor_actor=actor,
        optimizer=optimizer,
        batch=batch,
        nonlabor_scorer=nonlabor.score,
        rng=__import__("numpy").random.default_rng(4),
        epochs=2,
        batch_size=3,
        clip=0.2,
        entropy_weight=0.0,
        max_grad_norm=1.0,
        target_kl=10.0,
        audit_rows=4,
        actor_gate=_gate(policy, behavior),
        behavior_configuration=behavior,
    )
    assert stats["attemptedSteps"] == 4
    assert stats["acceptedSteps"] == 4
    assert not stats["actorStepRolledBack"]
    assert all(row["accepted"]["exactlyOnce"] for row in stats["epochCoverage"])
    assert actor.shift.item() != 0.0
    assert nonlabor.mean.item() != 0.0


def test_labor_update_rolls_back_policy_and_adam_together_on_fixed_audit_kl() -> None:
    actor = _DynamicActor()
    nonlabor = _NonlaborPolicy()
    policy = nn.ModuleDict({"labor": actor, "nonlabor": nonlabor})
    batch = _policy_batch(actor, nonlabor)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
    behavior = {"laborRegistryDigest": batch.registry_digest, "logStdFloor": -2.0}
    batch = dataclasses.replace(
        batch, behavior_policy_digest=behavior_policy_digest(policy, behavior)
    )
    before_policy = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    before_optimizer = optimizer.state_dict()
    stats = update_labor_policy(
        policy_module=policy,
        labor_actor=actor,
        optimizer=optimizer,
        batch=batch,
        nonlabor_scorer=nonlabor.score,
        rng=__import__("numpy").random.default_rng(5),
        epochs=3,
        batch_size=2,
        clip=0.2,
        entropy_weight=0.0,
        max_grad_norm=1.0,
        target_kl=1e-12,
        audit_rows=4,
        actor_gate=_gate(policy, behavior),
        behavior_configuration=behavior,
    )
    assert stats["actorStepRolledBack"]
    assert stats["attemptedSteps"] == 1
    assert stats["acceptedSteps"] == 0
    assert stats["epochCoverage"][0]["planned"]["exactlyOnce"]
    assert not stats["epochCoverage"][0]["processed"]["exactlyOnce"]
    assert not stats["epochCoverage"][0]["accepted"]["exactlyOnce"]
    assert all(
        torch.equal(policy.state_dict()[name], value) for name, value in before_policy.items()
    )
    assert optimizer.state_dict() == before_optimizer


def _catalog(camp: int) -> LaborCatalog:
    specialty = [True] * len(SPECIALTIES)
    activity = [[True] + [False] * (len(ACTIVITIES) - 1) for _ in SPECIALTIES[1:]]
    return LaborCatalog(
        camp_id=camp,
        turn=9,
        population=10,
        turn_years=20.0,
        masks=LaborMasks.from_values(specialty, activity),
        specialty_rows=(),
        activity_rows=(),
        resources=(),
        digest=("c" if camp == 11 else "d") * 64,
    )


def _labor_payload(catalog: LaborCatalog) -> dict[str, object]:
    specialty = [0.0] * len(catalog.masks.specialty)
    specialty[0] = 1.0
    activity = []
    for row in catalog.masks.activity:
        allocation = [0.0] * len(row)
        allocation[0] = 1.0
        activity.append(allocation)
    return {
        "laborSpecialtyAllocation": specialty,
        "laborActivityAllocation": activity,
    }


def _retained(catalog: LaborCatalog, *, retry: bool) -> RetainedLaborAction:
    payload = _labor_payload(catalog)
    if not retry:
        payload = {
            "schemaVersion": 12,
            **payload,
            "movement": 0,
            "fission": 0,
            "tradeValue": [1.0] * len(contracts.TRADE_GROUPS),
            "tradeGate": [0] * len(contracts.TRADE_GROUPS),
            "tradeShare": 0.5,
        }
    specialty = torch.tensor(payload["laborSpecialtyAllocation"]).unsqueeze(0)
    activity = torch.tensor(payload["laborActivityAllocation"]).unsqueeze(0)
    draw = {
        "specialty_allocation": specialty,
        "activity_allocation": activity,
        "specialty_latent": torch.zeros_like(specialty),
        "activity_latent": torch.zeros_like(activity),
        "specialty_mask": torch.tensor(catalog.masks.specialty).unsqueeze(0),
        "activity_mask": torch.tensor(catalog.masks.activity).unsqueeze(0),
    }
    return RetainedLaborAction(
        payload=payload,
        labor_sample=draw,
        behavior_labor_logp=-1.0,
        nonlabor_sample=None if retry else {"movement": torch.tensor([0])},
        behavior_nonlabor_logp=None if retry else -0.25,
    )


def _attempt(catalog: LaborCatalog, attempt: int, *, valid: bool) -> LaborAttempt:
    action = _labor_payload(catalog)
    activities = tuple(tuple(row) for row in action["laborActivityAllocation"])
    zero_work = tuple(tuple(0.0 for _ in ACTIVITIES) for _ in SPECIALTIES[1:])
    violation = LaborViolation(
        1,
        "over-capacity",
        "known",
        "worker-year",
        "planning",
        (0,),
        (),
        1.0,
        0.0,
        1.0,
        True,
    )
    return LaborAttempt(
        camp_id=catalog.camp_id,
        attempt=attempt,
        remaining_proposals=1 - attempt,
        valid=valid,
        fallback=bool(attempt == 1 and not valid),
        requested_specialty_allocation=tuple(action["laborSpecialtyAllocation"]),
        requested_specialty_residents=tuple([10.0] + [0.0] * (len(SPECIALTIES) - 1)),
        accepted_specialty_residents=tuple([10] + [0] * (len(SPECIALTIES) - 1)),
        requested_activity_allocation=activities,
        requested_activity_worker_years=zero_work,
        planned_activity_worker_years=zero_work,
        violations=() if valid else (violation,),
        catalog_digest=catalog.digest,
        feasibility_cost=0.0 if valid else 1.0,
    )


def _feedback(catalog: LaborCatalog, initial: LaborAttempt, final: LaborAttempt) -> LaborFeedback:
    empty_work = tuple(tuple(0.0 for _ in ACTIVITIES) for _ in SPECIALTIES[1:])
    unknown_work = tuple(tuple(None for _ in ACTIVITIES) for _ in SPECIALTIES[1:])
    false_work = tuple(tuple(False for _ in ACTIVITIES) for _ in SPECIALTIES[1:])
    reasons = tuple(tuple("unavailable" for _ in ACTIVITIES) for _ in SPECIALTIES[1:])
    reason_codes = tuple(tuple(15 for _ in ACTIVITIES) for _ in SPECIALTIES[1:])
    return LaborFeedback(
        camp_id=catalog.camp_id,
        decision_turn=catalog.turn,
        execution_comparator_attempt=final.attempt,
        parent_decision_id=f"decision-{catalog.camp_id}",
        founding_parent_decision_id=None,
        decision_catalog=catalog,
        initial_request=initial,
        final_sampled_request=final,
        committed_activity_worker_years=empty_work,
        actual_scheduler_consumed_worker_years=unknown_work,
        actual_worker_years_known=false_work,
        actual_worker_years_reasons=reasons,
        actual_worker_years_reason_codes=reason_codes,
        planned_goods_output=(None,) * GOODS_ROWS,
        realized_goods_output=(None,) * GOODS_ROWS,
        planned_goods_output_known=(False,) * GOODS_ROWS,
        planned_goods_output_reasons=("unavailable",) * GOODS_ROWS,
        planned_goods_output_reason_codes=(16,) * GOODS_ROWS,
        goods_output_known=(False,) * GOODS_ROWS,
        goods_output_reasons=("unavailable",) * GOODS_ROWS,
        goods_output_reason_codes=(17,) * GOODS_ROWS,
        realized_resource_receipts=(None,) * RESOURCE_ROWS,
        resource_receipt_known=(False,) * RESOURCE_ROWS,
        resource_receipt_reasons=("unavailable",) * RESOURCE_ROWS,
        resource_receipt_reason_codes=(18,) * RESOURCE_ROWS,
        fallback=final.fallback,
        fallback_reason="fallback" if final.fallback else "",
        fallback_reason_code=14 if final.fallback else 0,
    )


class _Runtime:
    def __init__(self, catalogs: dict[int, LaborCatalog]) -> None:
        self.initial = {
            11: _attempt(catalogs[11], 0, valid=False),
            12: _attempt(catalogs[12], 0, valid=True),
        }
        self.retry = {11: _attempt(catalogs[11], 1, valid=True)}
        self.calls: list[tuple[str, object]] = []

    def validate_labor(self, actions: dict[int, dict], *, attempt: int) -> dict[int, LaborAttempt]:
        self.calls.append(("validate", (attempt, actions)))
        return self.initial if attempt == 0 else self.retry

    def step_labor(
        self, initial_actions: dict[int, dict], retry_actions: dict[int, dict] | None = None
    ) -> tuple[dict, dict[int, LaborFeedback]]:
        self.calls.append(("step", (initial_actions, retry_actions)))
        return {"turn": 10}, {
            11: _feedback(_catalog(11), self.initial[11], self.retry[11]),
            12: _feedback(_catalog(12), self.initial[12], self.initial[12]),
        }


def test_typed_collector_retries_only_rejected_labor_and_mutates_world_once() -> None:
    catalogs = {11: _catalog(11), 12: _catalog(12)}
    runtime = _Runtime(catalogs)
    initial = {camp: _retained(catalog, retry=False) for camp, catalog in catalogs.items()}
    execution = execute_labor_decision(
        runtime,
        catalogs,
        initial,
        lambda _camp, catalog, _attempt: _retained(catalog, retry=True),
    )
    assert [name for name, _ in runtime.calls] == ["validate", "validate", "step"]
    retry_payload = runtime.calls[1][1][1][11]
    assert set(retry_payload) == LABOR_RETRY_FIELDS
    assert set(execution.retry) == {11}
    assert execution.final_attempts[12] is execution.initial_attempts[12]
    assert execution.feedback[11].execution_comparator_attempt == 1


def test_schema12_action_rejects_retired_heads_and_retry_nonlabor() -> None:
    catalog = _catalog(11)
    initial = _retained(catalog, retry=False)
    validate_schema12_action(initial.payload, catalog, retry=False)
    retired = dict(initial.payload)
    retired["service"] = [0.0]
    with pytest.raises(ValueError, match="fields must be exactly"):
        validate_schema12_action(retired, catalog, retry=False)
    retry = _retained(catalog, retry=True)
    invalid = RetainedLaborAction(
        retry.payload, retry.labor_sample, -1.0, {"movement": torch.tensor([0])}, -0.2
    )
    with pytest.raises(ValueError, match="must not retain"):
        invalid.validate(catalog, retry=True)


def test_retained_sample_uses_wire_simplex_canonicalization_without_mutating_draw() -> None:
    catalog = _catalog(11)
    specialty = torch.zeros(1, len(SPECIALTIES), dtype=torch.float32)
    specialty[0, 0] = 0.5
    specialty[0, 1] = 0.4999998
    activity = torch.zeros(1, len(SPECIALTIES) - 1, len(ACTIVITIES), dtype=torch.float32)
    activity[:, :, 0] = 1.0
    # The serializer repairs only the wire simplex's reference coordinate.
    # PPO must retain the original allocation/latents that generated log p.
    labor_wire = LaborPolicySample(
        tuple(float(value) for value in specialty[0]),
        tuple(tuple(float(value) for value in row) for row in activity[0]),
    ).payload()
    payload = {
        "schemaVersion": 12,
        **labor_wire,
        "movement": 0,
        "fission": 0,
        "tradeValue": [1.0] * len(contracts.TRADE_GROUPS),
        "tradeGate": [0] * len(contracts.TRADE_GROUPS),
        "tradeShare": 0.5,
    }
    original = specialty.clone()
    retained = RetainedLaborAction(
        payload,
        {
            "specialty_allocation": specialty,
            "activity_allocation": activity,
            "specialty_latent": torch.full_like(specialty, 0.25),
            "activity_latent": torch.full_like(activity, -0.25),
            "specialty_mask": torch.tensor(catalog.masks.specialty).unsqueeze(0),
            "activity_mask": torch.tensor(catalog.masks.activity).unsqueeze(0),
        },
        -1.0,
        {"movement": torch.tensor([0])},
        -0.25,
    )
    decoded = retained.validate(catalog, retry=False)
    assert decoded.specialty_allocation[1] == pytest.approx(0.5)
    assert torch.equal(retained.labor_sample["specialty_allocation"], original)
    assert retained.labor_sample["specialty_latent"][0, 0].item() == pytest.approx(0.25)
