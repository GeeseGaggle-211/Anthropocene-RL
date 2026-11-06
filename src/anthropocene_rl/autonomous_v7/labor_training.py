"""Runnable collection and one-update training for the opt-in labor learner.

This is deliberately a small vertical slice over the reviewed typed boundaries:

* :class:`LaborModel9Bridge` is the only native labor execution path;
* initial attempts score labor plus the retained non-labor posture;
* rejected attempts receive one labor-only retry at zero simulated time;
* physical task reward is recorded once, on the execution edge;
* all four critics receive frozen-rollout TD targets plus an independently
  observed-return anchor loss; and
* actor updates require a current four-head calibration report.

The commands are executable once a contract-13 bridge is supplied, but this
module never launches a workload merely by being imported.  Local collection
is capped at three world seeds and is diagnostic only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contracts, data
from .bridge import Model9Bridge, observation_batch
from .labor_bc import extract_heuristic_labor_action
from .labor_bridge import (
    LABOR_ACTIVITY_FEATURE_WIDTH,
    LABOR_REGISTRY_DIGEST,
    LABOR_RESOURCE_FEATURE_WIDTH,
    LABOR_SPECIALTY_FEATURE_WIDTH,
    LaborAttempt,
    LaborCatalog,
    LaborModel9Bridge,
    labor_observation_fields,
)
from .labor_model import (
    LABOR_CAMPAIGN_R5_QOL_ENCODER,
    LABOR_CAMPAIGN_R5_QOL_HEAD_SCALE,
    LABOR_CAMPAIGN_R5_VALUE_SUPPORT,
    LABOR_COSTS,
    LABOR_OBJECTIVES,
    LaborActor,
    LaborCostCritics,
    LaborEncoderLayout,
    LaborPersonality,
    LaborQoLHead,
    LaborTaskCritic,
    LaborValueSupport,
    Normalizer,
    bind_privileged_normalizer,
    labor_checkpoint_payload,
    labor_checkpoint_personality,
    labor_checkpoint_qol_encoder,
    labor_checkpoint_qol_head,
    labor_checkpoint_value_support,
    labor_log_prob,
    labor_policy_tensors,
    labor_qol_encoder_seed,
    load_labor_checkpoint_models,
    nonlabor_entropy,
    nonlabor_log_prob,
    sample_labor_policy,
    sample_nonlabor_policy,
    schema12_initial_action,
    schema12_retry_action,
    set_labor_film_bound,
)
from .labor_personality import (
    LaborPersonalityAssignment,
    LaborPersonalityConditioning,
    WorldPersonalities,
    personality_for_world,
)
from .labor_ppo import (
    LaborActorGate,
    LaborAttemptPolicyBatch,
    RetainedLaborAction,
    combine_labor_advantages,
    execute_labor_decision,
    labor_cost_value_loss,
    labor_task_value_loss,
    two_attempt_gae,
    update_labor_policy,
)
from .labor_stability import add_stability_arguments
from .labor_value_calibration import (
    labor_anchor_loss,
    labor_critic_digest,
    load_labor_anchor,
)
from .labor_worlds import require_training_worlds
from .model import BOUNDED_FEATURE_CONTRACT, load_checkpoint_normalizer
from .ppo import SMALL_SETTLEMENT, YOUNG_DAUGHTER_TURNS
from .protocol import scan
from .protocol import seeds as checked_seeds
from .provenance import checkpoint_hash, digest
from .value_calibration import behavior_policy_digest

LABOR_TRAINING_CONTRACT = "model9-specialty-attempt-training-v1"
LABOR_ROLLOUT_KIND = "model9-specialty-attempt-rollout-v1"
LOCAL_WORLD_LIMIT = 3


def _cpu_rows(values: Mapping[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 1:
            raise ValueError(f"{name} is not a row-aligned tensor")
        if name == "movement_codes" and value.ndim == 1:
            result[name] = value.detach().cpu().clone()
            continue
        result[name] = value.index_select(0, rows.to(value.device)).detach().cpu().clone()
    return result


def _one_row(values: Mapping[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    return _cpu_rows(values, torch.tensor([row], dtype=torch.long))


def _concat(rows: Sequence[Mapping[str, torch.Tensor]], label: str) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError(f"{label} cannot be empty")
    names = set(rows[0])
    if any(set(row) != names for row in rows):
        raise ValueError(f"{label} tensor fields differ between rows")
    result: dict[str, torch.Tensor] = {}
    for name in sorted(names):
        if name == "movement_codes" and rows[0][name].ndim == 1:
            result[name] = rows[0][name].clone()
        else:
            result[name] = torch.cat([row[name] for row in rows], dim=0)
    if any(value.requires_grad for value in result.values()):
        raise ValueError(f"{label} unexpectedly retained an autograd graph")
    return result


def _empty_like_rows(example: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value[:0].detach().cpu().clone() for name, value in example.items()}


def _normalization_digest(normalizer: Any) -> str:
    return digest({"semantics": "labor-normalizer-binding-v1", "state": normalizer.state_dict()})


def labor_behavior_configuration(
    personality: LaborPersonality | LaborPersonalityAssignment,
) -> dict[str, object]:
    """Complete behavior signature for sampled schema-12 policy data.

    A fixed personality keeps its historical signature. A declared
    distribution adds its conditioning record (and the balanced reference).
    """

    if isinstance(personality, LaborPersonalityAssignment):
        configuration = labor_behavior_configuration(LaborPersonality())
        configuration["personalityConditioning"] = personality.conditioning.state_dict()
        configuration["personalityIteration"] = int(personality.iteration)
        return configuration
    return {
        "contract": LABOR_TRAINING_CONTRACT,
        "laborRegistryDigest": LABOR_REGISTRY_DIGEST,
        "actionSchema": 12,
        "initialHeads": ["labor", "movement", "fission", "trade"],
        "retryHeads": ["labor"],
        "laborSampler": "gauge-fixed-additive-log-ratio-gaussian-v1",
        "nonlaborSampler": "categorical-and-gaussian-latent-v1",
        "personality": personality.state_dict(),
        "physicalTaskRewardEdges": 1,
        "internalRetryDiscount": 1.0,
        "internalRetryTraceContinuation": 1.0,
    }


def _task_values(
    critic: LaborTaskCritic, tensors: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        logits = critic(tensors)
        return {
            name: critic.expectation(name, logits[name]).detach().cpu() for name in LABOR_OBJECTIVES
        }


def _cost_values(
    critics: LaborCostCritics, tensors: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return {name: value.detach().cpu() for name, value in critics(tensors).items()}


def _score_nonlabor(
    actor: LaborActor,
    tensors: Mapping[str, torch.Tensor],
    sample: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    output = actor(tensors)
    return (
        nonlabor_log_prob(output, sample)["joint"],
        nonlabor_entropy(output)["joint"],
    )


@dataclass
class LaborRolloutRow:
    world_seed: int
    camp_id: int
    turn: int
    years: float
    terminated: bool
    initial_tensors: dict[str, torch.Tensor]
    initial_labor_sample: dict[str, torch.Tensor]
    initial_nonlabor_sample: dict[str, torch.Tensor]
    behavior_initial_labor_logp: float
    behavior_initial_nonlabor_logp: float
    initial_task_values: dict[str, float]
    initial_cost_values: dict[str, float]
    retry_tensors: dict[str, torch.Tensor] | None
    retry_labor_sample: dict[str, torch.Tensor] | None
    behavior_retry_labor_logp: float | None
    retry_task_values: dict[str, float] | None
    retry_cost_values: dict[str, float] | None
    physical_task_rewards: dict[str, float]
    feasibility_initial: float
    feasibility_retry: float
    settlement_guard: float


def _transition_rewards(
    step: Mapping[str, Any],
    acted_camps: Sequence[int],
    founded_turns: dict[int, int],
    *,
    world_index: int,
    decision_turn: int,
) -> tuple[dict[int, dict[str, float]], dict[int, bool], dict[int, float]]:
    """Derive complete-transition task/guard rows before selecting acted camps."""

    villages = step.get("villages")
    if not isinstance(villages, list) or not villages:
        raise ValueError("labor successor step contains no village outcomes")
    outcome: dict[str, list[Any]] = defaultdict(list)
    camp_to_row: dict[int, int] = {}
    for village in villages:
        if not isinstance(village, Mapping):
            raise ValueError("labor successor village is malformed")
        camp = int(village["camp"])
        if camp in camp_to_row:
            raise ValueError("labor successor repeats a camp")
        founded_turns.setdefault(camp, int(step["turn"]))
        camp_to_row[camp] = len(outcome["camp"])
        outcome["world"].append(world_index)
        outcome["camp"].append(camp)
        outcome["turn"].append(decision_turn)
        outcome["society"].append(int(village["society"]))
        outcome["parent_society"].append(int(village["parentSociety"]))
        outcome["founded_turn"].append(int(founded_turns[camp]))
        outcome["prior_population"].append(int(village["priorPopulation"]))
        outcome["population"].append(int(village["population"]))
        outcome["qol"].append(float(village["qol"]))
        outcome["years"].append(float(village["years"]))
        outcome["welfare"].append(float(village["welfare"]))
        outcome["terminated"].append(bool(village["terminated"]))
        outcome["trade_utility"].append(float(village.get("tradeUtility", 0.0)))
    arrays = {name: np.asarray(values) for name, values in outcome.items()}
    components = data.reward_components(arrays)
    rewards: dict[int, dict[str, float]] = {}
    terminated: dict[int, bool] = {}
    years: dict[int, float] = {}
    for camp in acted_camps:
        if camp not in camp_to_row:
            raise ValueError(f"acted labor camp {camp} is absent from the physical outcome")
        row = camp_to_row[camp]
        rewards[camp] = {
            "qol_years": float(components["qol_years"][row]),
            "population_years": float(components["population_years"][row]),
        }
        terminated[camp] = bool(arrays["terminated"][row])
        years[camp] = float(arrays["years"][row])
    return rewards, terminated, years


def _settlement_guard(village: Mapping[str, Any], founded_turn: int, turn: int) -> float:
    young = int(village["parentSociety"]) > 0 and turn - founded_turn <= YOUNG_DAUGHTER_TURNS
    return float(
        not bool(village["terminated"])
        and int(village["population"]) < SMALL_SETTLEMENT
        and not young
    )


def _execute_single_step_detailed(
    *,
    bridge: LaborModel9Bridge,
    step: Mapping[str, Any],
    actor: LaborActor,
    task_critic: LaborTaskCritic | None = None,
    cost_critics: LaborCostCritics | None = None,
    normalizer: Any,
    personality: LaborPersonality | WorldPersonalities,
    generator: torch.Generator,
    world_seed: int,
    world_index: int,
    founded_turns: dict[int, int],
) -> tuple[Mapping[str, Any], list[LaborRolloutRow], dict[int, bool]]:
    live = [
        village
        for village in step["villages"]
        if village.get("observation") is not None and not village.get("terminated")
    ]
    if not live:
        return step, [], {}
    contexts, tensors = labor_policy_tensors(live, normalizer, personality=personality)
    catalogs = {context[0].camp_id: context[0] for context in contexts}
    if len(catalogs) != len(live):
        raise ValueError("labor rollout has duplicate live camp identities")

    with torch.no_grad():
        output = actor(tensors)
        initial_labor = sample_labor_policy(output, tensors, generator=generator)
        initial_nonlabor = sample_nonlabor_policy(output, generator=generator)
        labor_lp = labor_log_prob(output, initial_labor, tensors)["joint"].detach().cpu()
        nonlabor_lp = nonlabor_log_prob(output, initial_nonlabor)["joint"].detach().cpu()
        if task_critic is not None:
            initial_task = _task_values(task_critic, tensors)
        else:
            initial_task = {name: torch.zeros(len(live)) for name in LABOR_OBJECTIVES}
        if cost_critics is not None:
            initial_cost = _cost_values(cost_critics, tensors)
        else:
            initial_cost = {name: torch.zeros(len(live)) for name in LABOR_COSTS}

    retained: dict[int, RetainedLaborAction] = {}
    village_by_camp: dict[int, Mapping[str, Any]] = {}
    row_by_camp: dict[int, int] = {}
    for row, (village, context) in enumerate(zip(live, contexts, strict=True)):
        camp = int(village["camp"])
        if context[0].camp_id != camp:
            raise ValueError("labor observation/catalog camp mismatch")
        village_by_camp[camp] = village
        row_by_camp[camp] = row
        retained[camp] = RetainedLaborAction(
            schema12_initial_action(initial_labor, initial_nonlabor, row),
            _one_row(initial_labor, row),
            float(labor_lp[row]),
            _one_row(initial_nonlabor, row),
            float(nonlabor_lp[row]),
        )

    retry_rows: dict[
        int,
        tuple[
            dict[str, torch.Tensor],
            dict[str, torch.Tensor],
            float,
            dict[str, float],
            dict[str, float],
        ],
    ] = {}

    def retry_factory(
        camp: int, catalog: LaborCatalog, attempt: LaborAttempt
    ) -> RetainedLaborAction:
        retry_contexts, retry_tensors = labor_policy_tensors(
            [village_by_camp[camp]],
            normalizer,
            personality=personality,
            rejected_attempts=[attempt],
        )
        if retry_contexts[0][0].digest != catalog.digest or retry_contexts[0][2] != attempt:
            raise ValueError("labor retry tensor context differs from typed rejection")
        with torch.no_grad():
            retry_output = actor(retry_tensors)
            retry_sample = sample_labor_policy(retry_output, retry_tensors, generator=generator)
            retry_logp = float(
                labor_log_prob(retry_output, retry_sample, retry_tensors)["joint"].detach().cpu()[0]
            )
            if task_critic is not None:
                task_values = {
                    name: float(value[0])
                    for name, value in _task_values(task_critic, retry_tensors).items()
                }
            else:
                task_values = {name: 0.0 for name in LABOR_OBJECTIVES}
            if cost_critics is not None:
                cost_values = {
                    name: float(value[0])
                    for name, value in _cost_values(cost_critics, retry_tensors).items()
                }
            else:
                cost_values = {name: 0.0 for name in LABOR_COSTS}
        retry_rows[camp] = (
            _one_row(retry_tensors, 0),
            _one_row(retry_sample, 0),
            retry_logp,
            task_values,
            cost_values,
        )
        return RetainedLaborAction(
            schema12_retry_action(retry_sample, 0),
            _one_row(retry_sample, 0),
            retry_logp,
            None,
            None,
        )

    execution = execute_labor_decision(bridge, catalogs, retained, retry_factory)
    next_step = execution.step
    rewards, terminal, elapsed = _transition_rewards(
        next_step,
        tuple(catalogs),
        founded_turns,
        world_index=world_index,
        decision_turn=int(step["turn"]),
    )
    outcomes = {int(village["camp"]): village for village in next_step["villages"]}
    rows: list[LaborRolloutRow] = []
    # Each row records its own personality in initial_tensors["labor_personality"].
    for camp in catalogs:
        row = row_by_camp[camp]
        retry = retry_rows.get(camp)
        outcome = outcomes[camp]
        rows.append(
            LaborRolloutRow(
                world_seed=world_seed,
                camp_id=camp,
                turn=int(step["turn"]),
                years=elapsed[camp],
                terminated=terminal[camp],
                initial_tensors=_one_row(tensors, row),
                initial_labor_sample=_one_row(initial_labor, row),
                initial_nonlabor_sample=_one_row(initial_nonlabor, row),
                behavior_initial_labor_logp=float(labor_lp[row]),
                behavior_initial_nonlabor_logp=float(nonlabor_lp[row]),
                initial_task_values={
                    name: float(value[row]) for name, value in initial_task.items()
                },
                initial_cost_values={
                    name: float(value[row]) for name, value in initial_cost.items()
                },
                retry_tensors=None if retry is None else retry[0],
                retry_labor_sample=None if retry is None else retry[1],
                behavior_retry_labor_logp=None if retry is None else retry[2],
                retry_task_values=None if retry is None else retry[3],
                retry_cost_values=None if retry is None else retry[4],
                physical_task_rewards=rewards[camp],
                feasibility_initial=float(execution.initial_attempts[camp].feasibility_cost),
                feasibility_retry=(
                    0.0 if retry is None else float(execution.final_attempts[camp].feasibility_cost)
                ),
                settlement_guard=_settlement_guard(
                    outcome, founded_turns[camp], int(next_step["turn"])
                ),
            )
        )
    return next_step, rows, terminal


def _sample_world_step(
    *,
    bridge: LaborModel9Bridge,
    step: Mapping[str, Any],
    actor: LaborActor,
    task_critic: LaborTaskCritic | None = None,
    cost_critics: LaborCostCritics | None = None,
    normalizer: Any,
    personality: LaborPersonality | WorldPersonalities,
    generator: torch.Generator,
    world_seed: int,
    world_index: int,
    founded_turns: dict[int, int],
    max_self_destruct_rewinds: int = 0,
) -> tuple[Mapping[str, Any], list[LaborRolloutRow]]:
    if max_self_destruct_rewinds <= 0:
        next_step, rows, _ = _execute_single_step_detailed(
            bridge=bridge,
            step=step,
            actor=actor,
            task_critic=task_critic,
            cost_critics=cost_critics,
            normalizer=normalizer,
            personality=personality,
            generator=generator,
            world_seed=world_seed,
            world_index=world_index,
            founded_turns=founded_turns,
        )
        return next_step, rows

    fork, _ = bridge.fork()
    accumulated_rows: list[LaborRolloutRow] = []
    current_step = step
    saved_founded = dict(founded_turns)

    for rewind_idx in range(1 + max_self_destruct_rewinds):
        next_step, attempt_rows, terminal = _execute_single_step_detailed(
            bridge=bridge,
            step=current_step,
            actor=actor,
            task_critic=task_critic,
            cost_critics=cost_critics,
            normalizer=normalizer,
            personality=personality,
            generator=generator,
            world_seed=world_seed,
            world_index=world_index,
            founded_turns=founded_turns,
        )
        if not attempt_rows:
            bridge.drop(fork)
            return next_step, accumulated_rows

        is_self_destruct = any(terminal.values()) or any(
            r.settlement_guard > 0.0 for r in attempt_rows
        )
        if not is_self_destruct or rewind_idx == max_self_destruct_rewinds:
            bridge.drop(fork)
            accumulated_rows.extend(attempt_rows)
            return next_step, accumulated_rows
        else:
            for r in attempt_rows:
                if terminal.get(r.camp_id, False) or r.settlement_guard > 0.0:
                    turn_num = int(current_step["turn"])
                    synth_camp = -(r.camp_id * 1_000_000 + turn_num * 100 + rewind_idx + 1)
                    failed_row = copy.copy(r)
                    failed_row.camp_id = synth_camp
                    failed_row.terminated = True
                    accumulated_rows.append(failed_row)
            founded_turns.clear()
            founded_turns.update(saved_founded)
            current_step = bridge.restore(fork)

    return current_step, accumulated_rows


def _bootstrap_values(
    step: Mapping[str, Any],
    normalizer: Any,
    task_critic: LaborTaskCritic,
    cost_critics: LaborCostCritics,
    personality: LaborPersonality | WorldPersonalities,
) -> dict[int, dict[str, float]]:
    live = [
        village
        for village in step["villages"]
        if village.get("observation") is not None and not village.get("terminated")
    ]
    if not live:
        return {}
    _, tensors = labor_policy_tensors(live, normalizer, personality=personality)
    task = _task_values(task_critic, tensors)
    costs = _cost_values(cost_critics, tensors)
    return {
        int(village["camp"]): {
            **{name: float(task[name][row]) for name in LABOR_OBJECTIVES},
            **{name: float(costs[name][row]) for name in LABOR_COSTS},
        }
        for row, village in enumerate(live)
    }


def collect_labor_rollout(
    *,
    actor: LaborActor,
    task_critic: LaborTaskCritic,
    cost_critics: LaborCostCritics,
    normalizer: Any,
    world_seeds: Sequence[int],
    turns: int,
    generator: torch.Generator,
    personality: LaborPersonality | LaborPersonalityAssignment | None = None,
    bridge_factory: Any = LaborModel9Bridge,
    economic_trace: Any | None = None,
    max_self_destruct_rewinds: int = 0,
    allow_vm_scale: bool = False,
) -> tuple[list[LaborRolloutRow], dict[tuple[int, int], dict[str, float]]]:
    """Collect diagnostic local worlds without performing any optimizer step."""

    worlds = checked_seeds(list(world_seeds), "labor local rollout worlds")
    if not allow_vm_scale and (not worlds or len(worlds) > LOCAL_WORLD_LIMIT):
        raise ValueError(f"local labor collection requires 1..{LOCAL_WORLD_LIMIT} world seeds")
    if not worlds:
        raise ValueError("world seeds cannot be empty")
    if isinstance(turns, bool) or not isinstance(turns, int) or turns <= 0:
        raise ValueError("labor rollout turns must be a positive integer")
    declared = LaborPersonality() if personality is None else personality
    collected: list[LaborRolloutRow] = []
    bootstrap: dict[tuple[int, int], dict[str, float]] = {}
    for world_index, seed in enumerate(worlds):
        bridge = (
            bridge_factory(economic_diagnostics=True)
            if economic_trace is not None
            else bridge_factory()
        )
        try:
            step = bridge.reset(seed)
            founded = {int(row["camp"]): int(step["turn"]) for row in step["villages"]}
            world_personality = personality_for_world(declared, seed)
            for _ in range(turns):
                step, rows = _sample_world_step(
                    bridge=bridge,
                    step=step,
                    actor=actor,
                    task_critic=task_critic,
                    cost_critics=cost_critics,
                    normalizer=normalizer,
                    personality=world_personality,
                    generator=generator,
                    world_seed=seed,
                    world_index=world_index,
                    founded_turns=founded,
                    max_self_destruct_rewinds=max_self_destruct_rewinds,
                )
                if (
                    economic_trace is not None
                    and getattr(bridge, "_bridge", None)
                    and getattr(bridge._bridge, "economic_diagnostics", None)
                ):
                    for camp_id, diag in bridge._bridge.economic_diagnostics.items():
                        trans_id = (
                            diag.get("transitionId") or f"{seed}:{camp_id}:{int(step['turn']) - 1}"
                        )
                        if trans_id not in economic_trace.transition_ids:
                            economic_trace.transition_ids.append(trans_id)
                        economic_trace.write(
                            "transitions",
                            {
                                "campId": int(camp_id),
                                "turn": int(step["turn"]) - 1,
                                "worldSeed": seed,
                                "transitionId": trans_id,
                                "economic": diag,
                            },
                        )
                collected.extend(rows)
                if not rows:
                    break
            for camp, values in _bootstrap_values(
                step, normalizer, task_critic, cost_critics, world_personality
            ).items():
                bootstrap[(seed, camp)] = values
        finally:
            bridge.close()
    if economic_trace is not None:
        economic_trace.close()
    if not collected:
        raise ValueError("labor rollout collected no policy decisions")
    return collected, bootstrap


def build_labor_training_batch(
    rows: Sequence[LaborRolloutRow],
    bootstrap: Mapping[tuple[int, int], Mapping[str, float]],
    *,
    personality: LaborPersonality,
    cost_multipliers: Mapping[str, float],
    behavior_policy_sha256: str,
    critic_sha256: str,
    normalization_sha256: str,
    lam: float = 0.95,
) -> tuple[
    LaborAttemptPolicyBatch,
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, torch.Tensor]],
]:
    """Freeze four-head GAE/targets while preserving the parent decision denominator."""

    if not rows:
        raise ValueError("labor training batch requires rollout rows")
    by_sequence: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_sequence[(row.world_seed, row.camp_id)].append(index)
    for indices in by_sequence.values():
        indices.sort(key=lambda index: rows[index].turn)
        turns = [rows[index].turn for index in indices]
        if len(turns) != len(set(turns)) or turns != sorted(turns):
            raise ValueError("labor camp sequence has duplicate or unordered decisions")

    initial_advantages = {
        name: torch.zeros(len(rows)) for name in (*LABOR_OBJECTIVES, *LABOR_COSTS)
    }
    initial_targets = {name: torch.zeros(len(rows)) for name in LABOR_OBJECTIVES}
    initial_cost_targets = {name: torch.zeros(len(rows)) for name in LABOR_COSTS}
    retry_advantages: dict[str, list[float]] = {
        name: [] for name in (*LABOR_OBJECTIVES, *LABOR_COSTS)
    }
    retry_task_targets: dict[str, list[float]] = {name: [] for name in LABOR_OBJECTIVES}
    retry_cost_targets: dict[str, list[float]] = {name: [] for name in LABOR_COSTS}
    retry_parent: list[int] = []

    # Retry arrays are emitted in parent order, so append after all per-head
    # values for one parent have been computed.
    per_retry: dict[int, tuple[dict[str, float], dict[str, float]]] = {}
    for key, indices in by_sequence.items():
        next_advantage = {name: 0.0 for name in (*LABOR_OBJECTIVES, *LABOR_COSTS)}
        last = rows[indices[-1]]
        if last.terminated:
            next_value = {name: 0.0 for name in (*LABOR_OBJECTIVES, *LABOR_COSTS)}
        else:
            tail = bootstrap.get(key)
            if not isinstance(tail, Mapping) or set(tail) != set((*LABOR_OBJECTIVES, *LABOR_COSTS)):
                raise ValueError(
                    "living labor sequence requires a complete four-head successor bootstrap"
                )
            next_value = {name: float(tail[name]) for name in (*LABOR_OBJECTIVES, *LABOR_COSTS)}
            if any(not math.isfinite(value) for value in next_value.values()):
                raise ValueError("living labor successor bootstrap must be finite")
        for index in reversed(indices):
            row = rows[index]
            retry = row.retry_tensors is not None
            retry_result_adv: dict[str, float] = {}
            retry_result_target: dict[str, float] = {}
            for name in (*LABOR_OBJECTIVES, *LABOR_COSTS):
                if name in LABOR_OBJECTIVES:
                    physical = row.physical_task_rewards[name]
                    direct = 0.0 if retry else physical
                    execution = physical
                    initial_value = row.initial_task_values[name]
                    retry_value = 0.0 if not retry else row.retry_task_values[name]  # type: ignore[index]
                elif name == "feasibility":
                    direct = row.feasibility_initial
                    execution = row.feasibility_retry
                    initial_value = row.initial_cost_values[name]
                    retry_value = 0.0 if not retry else row.retry_cost_values[name]  # type: ignore[index]
                else:
                    direct = 0.0 if retry else row.settlement_guard
                    execution = row.settlement_guard
                    initial_value = row.initial_cost_values[name]
                    retry_value = 0.0 if not retry else row.retry_cost_values[name]  # type: ignore[index]
                result = two_attempt_gae(
                    initial_value=torch.tensor([initial_value]),
                    retry_value=torch.tensor([retry_value]),
                    initial_reward=torch.tensor([direct]),
                    execution_reward=torch.tensor([execution]),
                    retry_mask=torch.tensor([retry]),
                    next_value=torch.tensor([next_value[name]]),
                    next_advantage=torch.tensor([next_advantage[name]]),
                    outer_discount=torch.tensor([math.exp(-row.years / contracts.HORIZON_YEARS)]),
                    outer_continuation=torch.tensor([0.0 if row.terminated else 1.0]),
                    lam=lam,
                )
                initial_advantages[name][index] = result.initial_advantage[0]
                if name in LABOR_OBJECTIVES:
                    initial_targets[name][index] = result.initial_target[0]
                else:
                    initial_cost_targets[name][index] = result.initial_target[0]
                retry_result_adv[name] = float(result.retry_advantage[0])
                retry_result_target[name] = float(result.retry_target[0])
                next_advantage[name] = float(result.initial_advantage[0])
                next_value[name] = initial_value
            if retry:
                per_retry[index] = (retry_result_adv, retry_result_target)

    retry_tensor_rows: list[dict[str, torch.Tensor]] = []
    retry_sample_rows: list[dict[str, torch.Tensor]] = []
    retry_logp: list[float] = []
    for index, row in enumerate(rows):
        if row.retry_tensors is None:
            continue
        assert row.retry_labor_sample is not None and row.behavior_retry_labor_logp is not None
        retry_parent.append(index)
        retry_tensor_rows.append(row.retry_tensors)
        retry_sample_rows.append(row.retry_labor_sample)
        retry_logp.append(row.behavior_retry_labor_logp)
        advantages, targets = per_retry[index]
        for name in (*LABOR_OBJECTIVES, *LABOR_COSTS):
            retry_advantages[name].append(advantages[name])
            if name in LABOR_OBJECTIVES:
                retry_task_targets[name].append(targets[name])
            else:
                retry_cost_targets[name].append(targets[name])

    # Each row's combined advantage uses that row's own recorded personality
    # (A = wq Aq/sq + wp Ap/sp - costs); a retry uses its parent's row.
    initial_weights = _row_personality_weights(rows)
    initial_combined = combine_labor_advantages(
        {name: initial_advantages[name] for name in LABOR_OBJECTIVES},
        {name: initial_advantages[name] for name in LABOR_COSTS},
        personality=personality,
        cost_multipliers=cost_multipliers,
        row_weights=initial_weights,
    )
    retry_combined = combine_labor_advantages(
        {name: torch.tensor(retry_advantages[name]) for name in LABOR_OBJECTIVES},
        {name: torch.tensor(retry_advantages[name]) for name in LABOR_COSTS},
        personality=personality,
        cost_multipliers=cost_multipliers,
        row_weights=initial_weights.index_select(0, torch.tensor(retry_parent, dtype=torch.long)),
    )
    initial_tensor_rows = [row.initial_tensors for row in rows]
    initial_labor_rows = [row.initial_labor_sample for row in rows]
    initial_nonlabor_rows = [row.initial_nonlabor_sample for row in rows]
    if retry_tensor_rows:
        retry_tensors = _concat(retry_tensor_rows, "retry tensors")
        retry_samples = _concat(retry_sample_rows, "retry labor samples")
    else:
        retry_tensors = _empty_like_rows(initial_tensor_rows[0])
        retry_samples = _empty_like_rows(initial_labor_rows[0])
    batch = LaborAttemptPolicyBatch(
        initial_tensors=_concat(initial_tensor_rows, "initial tensors"),
        initial_labor_sample=_concat(initial_labor_rows, "initial labor samples"),
        initial_nonlabor_sample=_concat(initial_nonlabor_rows, "initial nonlabor samples"),
        behaviour_initial_labor_logp=torch.tensor(
            [row.behavior_initial_labor_logp for row in rows]
        ),
        behaviour_initial_nonlabor_logp=torch.tensor(
            [row.behavior_initial_nonlabor_logp for row in rows]
        ),
        initial_advantage=initial_combined,
        retry_parent=torch.tensor(retry_parent, dtype=torch.long),
        retry_tensors=retry_tensors,
        retry_labor_sample=retry_samples,
        behaviour_retry_labor_logp=torch.tensor(retry_logp),
        retry_advantage=retry_combined,
        registry_digest=LABOR_REGISTRY_DIGEST,
        behavior_policy_digest=behavior_policy_sha256,
        advantage_critic_digest=critic_sha256,
        normalization_binding_digest=normalization_sha256,
    )
    task_targets = {
        "initial": dict(initial_targets),
        "retry": {name: torch.tensor(retry_task_targets[name]) for name in LABOR_OBJECTIVES},
    }
    cost_targets = {
        "initial": dict(initial_cost_targets),
        "retry": {name: torch.tensor(retry_cost_targets[name]) for name in LABOR_COSTS},
    }
    return batch, task_targets, cost_targets


def _row_personality_weights(rows: Sequence[LaborRolloutRow]) -> torch.Tensor:
    weights = []
    for row in rows:
        value = row.initial_tensors.get("labor_personality")
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (1, len(LABOR_OBJECTIVES))
            or not value.is_floating_point()
        ):
            raise ValueError("labor rollout row lacks its recorded personality")
        if row.retry_tensors is not None and not torch.equal(
            row.retry_tensors.get("labor_personality", value), value
        ):
            raise ValueError("labor retry personality differs from its parent decision")
        weights.append(value[0])
    return torch.stack(weights)


def _state_dict_rollout(
    batch: LaborAttemptPolicyBatch,
    task_targets: Mapping[str, Mapping[str, torch.Tensor]],
    cost_targets: Mapping[str, Mapping[str, torch.Tensor]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": LABOR_ROLLOUT_KIND,
        "metadata": dict(metadata),
        "batch": {name: getattr(batch, name) for name in batch.__dataclass_fields__},
        "task_targets": dict(task_targets),
        "cost_targets": dict(cost_targets),
    }


def _load_rollout(
    path: Path,
) -> tuple[
    LaborAttemptPolicyBatch,
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, torch.Tensor]],
    dict[str, Any],
]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("kind") != LABOR_ROLLOUT_KIND:
        raise ValueError("file is not a labor attempt rollout")
    metadata = state.get("metadata")
    batch_state = state.get("batch")
    if not isinstance(metadata, dict) or not isinstance(batch_state, dict):
        raise ValueError("labor rollout metadata/batch is missing")
    batch = LaborAttemptPolicyBatch(**batch_state)
    task = state.get("task_targets")
    costs = state.get("cost_targets")
    if (
        not isinstance(task, dict)
        or set(task) != {"initial", "retry"}
        or any(not isinstance(value, dict) for value in task.values())
        or not isinstance(costs, dict)
        or set(costs) != {"initial", "retry"}
        or any(not isinstance(value, dict) for value in costs.values())
    ):
        raise ValueError("labor rollout critic targets are missing")
    return batch, task, costs, metadata


def _models_from_checkpoint(
    state: Mapping[str, Any],
) -> tuple[LaborEncoderLayout, LaborActor, LaborTaskCritic, LaborCostCritics, Any]:
    layout = LaborEncoderLayout.from_state(state.get("labor_encoder_layout"))
    actor = LaborActor(layout)
    task = LaborTaskCritic(
        layout,
        support=labor_checkpoint_value_support(state),
        qol_head=labor_checkpoint_qol_head(state),
        qol_encoder=labor_checkpoint_qol_encoder(state),
    )
    costs = LaborCostCritics(layout)
    # Also applies the contract's optional FiLM bound to all four encoders.
    _, normalizer = load_labor_checkpoint_models(state, actor, task, costs, expected_layout=layout)
    return layout, actor, task, costs, normalizer


def _trainer_checkpoint(
    actor: LaborActor,
    task: LaborTaskCritic,
    costs: LaborCostCritics,
    normalizer: Any,
    layout: LaborEncoderLayout,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    personality: LaborPersonality | LaborPersonalityAssignment,
    optimizer_seed: int,
    iteration: int,
    sampling_generator: torch.Generator,
    shuffle_rng: np.random.Generator,
    critic_training_world_seeds: Sequence[int],
    critic_training_independent_unit_ids: Sequence[str],
) -> dict[str, Any]:
    state = labor_checkpoint_payload(
        actor, task, costs, normalizer, layout, personality=personality
    )
    state.update(
        {
            "labor_training_contract": LABOR_TRAINING_CONTRACT,
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "optimizer_seed": int(optimizer_seed),
            "iteration": int(iteration),
            "sampling_generator_state": sampling_generator.get_state(),
            "shuffle_rng_state": copy.deepcopy(shuffle_rng.bit_generator.state),
            "rollout_order": "world-seed-then-turn-then-native-camp-order-v1",
            "critic_training_world_seeds": sorted(
                set(int(value) for value in critic_training_world_seeds)
            ),
            "critic_training_independent_unit_ids": sorted(
                set(str(value) for value in critic_training_independent_unit_ids)
            ),
        }
    )
    return state


def _require_trainer_checkpoint(state: Mapping[str, Any]) -> None:
    required = {
        "labor_training_contract",
        "actor_optimizer",
        "critic_optimizer",
        "optimizer_seed",
        "iteration",
        "sampling_generator_state",
        "shuffle_rng_state",
        "rollout_order",
        "critic_training_world_seeds",
        "critic_training_independent_unit_ids",
    }
    if not isinstance(state, Mapping) or not required.issubset(state):
        raise ValueError("labor trainer checkpoint lacks optimizer/RNG/order metadata")
    if state["labor_training_contract"] != LABOR_TRAINING_CONTRACT:
        raise ValueError("labor trainer checkpoint contract differs")
    if state["rollout_order"] != "world-seed-then-turn-then-native-camp-order-v1":
        raise ValueError("labor trainer checkpoint rollout order differs")
    training_worlds = state["critic_training_world_seeds"]
    training_units = state["critic_training_independent_unit_ids"]
    if (
        not isinstance(training_worlds, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in training_worlds)
        or training_worlds != sorted(set(training_worlds))
        or not isinstance(training_units, list)
        or any(not isinstance(value, str) or not value for value in training_units)
        or training_units != sorted(set(training_units))
    ):
        raise ValueError("labor trainer critic-data history is malformed")
    scan(
        {"criticTrainingWorldSeeds": training_worlds},
        "labor trainer critic-data history",
    )


def _parse_worlds(text: str, *, allow_vm_scale: bool = False) -> list[int]:
    values = [int(value) for value in text.split(",") if value]
    scan({"worldSeeds": values}, "labor local CLI seeds")
    checked = checked_seeds(values, "labor local CLI seeds")
    if not allow_vm_scale and (not checked or len(checked) > LOCAL_WORLD_LIMIT):
        raise ValueError(f"local labor CLI requires 1..{LOCAL_WORLD_LIMIT} seeds")
    if not checked:
        raise ValueError("world seeds cannot be empty")
    return checked


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _save_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, path)


def _critic_td_losses(
    task: LaborTaskCritic,
    costs: LaborCostCritics,
    batch: LaborAttemptPolicyBatch,
    task_targets: Mapping[str, Mapping[str, torch.Tensor]],
    cost_targets: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average physical and retry edges over the physical-parent denominator."""

    task_loss = labor_task_value_loss(task, batch.initial_tensors, task_targets["initial"])
    cost_loss = labor_cost_value_loss(costs, batch.initial_tensors, cost_targets["initial"])
    retry_count = len(batch.retry_parent)
    if retry_count:
        retry_weight = retry_count / batch.parent_count
        task_loss = task_loss + retry_weight * labor_task_value_loss(
            task, batch.retry_tensors, task_targets["retry"]
        )
        cost_loss = cost_loss + retry_weight * labor_cost_value_loss(
            costs, batch.retry_tensors, cost_targets["retry"]
        )
    return task_loss, cost_loss


def _gate_data_identity(report: Mapping[str, Any]) -> tuple[set[int], set[str]]:
    expected_heads = set((*LABOR_OBJECTIVES, *LABOR_COSTS))
    enabled = report.get("enabledHeads")
    heads = report.get("heads")
    if (
        report.get("kind") != "model9-labor-value-calibration-report-v1"
        or report.get("dataRole") != "actor-gate"
        or report.get("localReadinessClaim") is not False
        or report.get("actorGateOpen") not in (True, False)
        or not isinstance(enabled, list)
        or set(enabled) != expected_heads
        or not isinstance(heads, Mapping)
        or set(heads) != expected_heads
        or any(not isinstance(heads[name], Mapping) for name in expected_heads)
    ):
        raise ValueError("labor calibration report structure is invalid")
    for key in (
        "sourcePolicyDigest",
        "sourceCriticDigest",
        "normalizationBindingDigest",
        "dataSha256",
    ):
        value = report.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"labor calibration report {key} is not a SHA-256")
    worlds = report.get("worldSeeds")
    units = report.get("independentUnitIds")
    if (
        not isinstance(worlds, list)
        or not worlds
        or any(isinstance(value, bool) or not isinstance(value, int) for value in worlds)
        or len(worlds) != len(set(worlds))
        or not isinstance(units, list)
        or not units
        or any(not isinstance(value, str) or not value for value in units)
        or len(units) != len(set(units))
    ):
        raise ValueError("labor calibration report lacks unique world/unit identities")
    checked = checked_seeds(worlds, "labor calibration report worlds")
    return set(checked), set(units)


def _validate_training_data_separation(
    anchor: Any,
    batch: LaborAttemptPolicyBatch,
    rollout_metadata: Mapping[str, Any],
    checkpoint_state: Mapping[str, Any],
    gate_report: Mapping[str, Any] | None,
) -> tuple[set[int], set[str]]:
    source = anchor.metadata["source"]
    if source["sourcePolicyDigest"] != batch.behavior_policy_digest:
        raise ValueError("labor train anchor is off-policy for this frozen behavior rollout")
    rollout_worlds = set(
        checked_seeds(rollout_metadata.get("worldSeeds", []), "labor rollout training worlds")
    )
    if not rollout_worlds:
        raise ValueError("labor rollout has no bound training worlds")
    train_anchor_worlds = set(source["worldSeeds"])
    train_anchor_units = set(anchor.independent_unit_ids.tolist())
    prior_training_worlds = set(checkpoint_state["critic_training_world_seeds"])
    prior_training_units = set(checkpoint_state["critic_training_independent_unit_ids"])
    current_training_worlds = prior_training_worlds | rollout_worlds | train_anchor_worlds
    current_training_units = prior_training_units | train_anchor_units
    if gate_report is not None:
        gate_worlds, gate_units = _gate_data_identity(gate_report)
        if gate_worlds & current_training_worlds or gate_units & current_training_units:
            raise ValueError(
                "labor actor-gate worlds/units overlap rollout or critic-training data"
            )
    return current_training_worlds, current_training_units


def _labor_reset_options(args: argparse.Namespace) -> Any:
    """Manifest-driven reset options for heuristic BC/critic collection (movement on)."""
    from .labor_bc import LaborResetOptions

    return LaborResetOptions(
        players=getattr(args, "players", 8),
        movement=not getattr(args, "no_movement", False),
        width=getattr(args, "width", None),
        height=getattr(args, "height", None),
        separation=getattr(args, "separation", None),
    )


def _load_labor_teacher_dataset(
    args: argparse.Namespace, normalizer: Any, *, require_privileged: bool = False
) -> dict[str, Any]:
    """One critic dataset, or several.

    Several heuristic-teacher datasets (the r5 collection batches) are read as ONE
    teacher dataset (``labor_teacher_data.merge_labor_datasets``): shards in world order,
    equal mass per world, a world in two batches refused. One dataset is read exactly as
    before. Learner-state continuations (``--learner-state-continuations``) and curated
    scenarios (``--curated-scenarios``) then join that teacher dataset as their own
    sources (equal source mass); any other source kind is refused by
    ``require_critic_teacher_dataset``.
    """
    from .labor_teacher_data import load_labor_datasets

    directories = args.dataset if isinstance(args.dataset, list) else [args.dataset]
    return load_labor_datasets(
        [Path(d) for d in directories],
        expected_normalizer_digest=_normalization_digest(normalizer),
        require_privileged=require_privileged,
        drop_reserved=bool(getattr(args, "drop_reserved_worlds", False)),
        family_weights=_family_weights(args),
        family_normalizer=normalizer,
        merge_teacher=True,
    )


def _family_weights(args: argparse.Namespace) -> dict[str, float] | None:
    from .labor_scenarios import parse_family_weights

    return parse_family_weights(getattr(args, "family_weights", None))


FAMILY_SCHEDULES_KEY = "family_weight_schedules"


def _family_schedule(
    args: argparse.Namespace, state: Mapping[str, Any], role: str, epochs: int
) -> Any:
    """The fit's family weight schedule, continued from the checkpoint's ``role`` record."""
    from .labor_scenarios import continued_schedule

    weights = _family_weights(args)
    if not weights:
        if getattr(args, "family_weight_decay", None) is not None:
            raise ValueError("--family-weight-decay needs --family-weights")
        return None
    previous = (state.get(FAMILY_SCHEDULES_KEY) or {}).get(role)
    return continued_schedule(
        weights,
        getattr(args, "family_weight_decay", None),
        getattr(args, "family_weight_decay_epochs", None),
        int(epochs),
        previous,
        restart=bool(getattr(args, "family_weight_restart", False)),
    )


def _carry_family_schedules(
    trainer_state: dict[str, Any], state: Mapping[str, Any], role: str, record: Any
) -> None:
    """Keep every role's schedule record; update ``role`` with this fit's end state."""
    schedules = dict(state.get(FAMILY_SCHEDULES_KEY) or {})
    if record is not None:
        schedules[role] = dict(record)
    if schedules:
        trainer_state[FAMILY_SCHEDULES_KEY] = schedules


def _load_privileged_normalizer(path: Path | None) -> Any:
    """A frozen privileged normalizer from ``fit-privileged-normalizer`` (or None)."""
    if path is None:
        return None
    from .labor_privileged import PRIVILEGED_NORMALIZER_KINDS, LaborPrivilegedNormalizer

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or state.get("kind") not in PRIVILEGED_NORMALIZER_KINDS:
        raise ValueError(f"{path} is not a fitted labor privileged normalizer")
    return LaborPrivilegedNormalizer.from_state(state["privileged_normalizer"])


def _bc_dataset_directories(args: argparse.Namespace) -> list[Path]:
    """--dataset (repeatable) plus --dataset-list entries, in order; empty when neither."""
    directories = [Path(d) for d in (getattr(args, "dataset", None) or [])]
    listing = getattr(args, "dataset_list", None)
    if listing is not None:
        for line in Path(listing).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                directories.append(Path(line))
    return directories


def _initialize_value_support(args: argparse.Namespace) -> LaborValueSupport:
    """Task-critic support for a new checkpoint.

    ``--campaign-r5`` declares :data:`LABOR_CAMPAIGN_R5_VALUE_SUPPORT`
    ([-0.5, 2.5] by 255). An explicit ``--critic-value-support`` must then
    agree with it. Without either flag the implicit historical default is used,
    keeping default checkpoints bit-identical.
    """
    explicit = getattr(args, "critic_value_support", None)
    if not getattr(args, "campaign_r5", False):
        return LaborValueSupport.parse(explicit)
    declared = LABOR_CAMPAIGN_R5_VALUE_SUPPORT
    if explicit is not None and LaborValueSupport.parse(explicit) != declared:
        raise ValueError(
            "--campaign-r5 declares value support -0.5,2.5,255; "
            f"--critic-value-support={explicit} contradicts it"
        )
    return LABOR_CAMPAIGN_R5_VALUE_SUPPORT


def _initialize_qol_head(args: argparse.Namespace) -> LaborQoLHead | None:
    """Bounded QoL head for a new checkpoint.

    ``--qol-head-scale`` sets the head range in units of W. Without it,
    ``--campaign-r5`` declares :data:`LABOR_CAMPAIGN_R5_QOL_HEAD_SCALE` (2.0, user
    decision 27 Oct) for a bounded head, and every other initialization keeps the
    historical 1.0 (no serialized field, bit-identical checkpoints).
    """
    text = getattr(args, "qol_head", None)
    scale = getattr(args, "qol_head_scale", None)
    head = LaborQoLHead.parse(text, None if scale is None else float(scale))
    if head is not None and scale is None and getattr(args, "campaign_r5", False):
        head = head.with_scale(LABOR_CAMPAIGN_R5_QOL_HEAD_SCALE)
    return head


def _initialize_qol_encoder(args: argparse.Namespace) -> str:
    """Task-critic QoL encoder for a new checkpoint.

    An explicit ``--qol-encoder shared|separate`` wins; otherwise ``--campaign-r5``
    declares :data:`LABOR_CAMPAIGN_R5_QOL_ENCODER` ("shared" again since 28 Oct, after
    the Fable audit; the r5 baseline is the shared encoder with ``pretrain-critic
    --qol-loss-weight 30``) and every other initialization keeps the historical shared
    encoder (no serialized field, bit-identical checkpoints).
    """
    explicit = getattr(args, "qol_encoder", None)
    if explicit is not None:
        return str(explicit)
    return LABOR_CAMPAIGN_R5_QOL_ENCODER if getattr(args, "campaign_r5", False) else "shared"


def _command_initialize(args: argparse.Namespace) -> None:
    source = torch.load(args.normalizer_checkpoint, map_location="cpu", weights_only=True)
    normalizer = load_checkpoint_normalizer(
        source, expected_feature_contract=BOUNDED_FEATURE_CONTRACT
    )
    privileged_normalizer = _load_privileged_normalizer(
        getattr(args, "privileged_normalizer", None)
    )
    world = checked_seeds([args.world_seed], "labor initialize world")[0]
    with LaborModel9Bridge() as bridge:
        step = bridge.reset(world)
        live = [row for row in step["villages"] if row.get("observation") is not None]
        if not live:
            raise ValueError("labor initialization world has no live policy rows")
        _, tensors = labor_policy_tensors(live, normalizer)
        if privileged_normalizer is not None:
            from .labor_privileged import privileged_raw_batch

            producer = getattr(getattr(bridge, "_bridge", bridge), "privileged_layout", None)
            if producer != privileged_normalizer.layout:
                raise ValueError("bridge privileged layout differs from the privileged normalizer")
            privileged_raw_batch(live, privileged_normalizer.layout)  # producer emits it now
    layout = LaborEncoderLayout(
        int(tensors["core"].shape[1]),
        LABOR_SPECIALTY_FEATURE_WIDTH,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
        privileged=None if privileged_normalizer is None else privileged_normalizer.layout,
    )
    torch.manual_seed(args.optimizer_seed)
    actor = LaborActor(layout)
    task = LaborTaskCritic(
        layout,
        support=_initialize_value_support(args),
        qol_head=_initialize_qol_head(args),
        qol_encoder=_initialize_qol_encoder(args),
        qol_encoder_seed=labor_qol_encoder_seed(args.optimizer_seed),
    )
    costs = LaborCostCritics(layout)
    bind_privileged_normalizer(privileged_normalizer, task, costs)
    conditioning = LaborPersonalityConditioning(
        getattr(args, "personality", "balanced"),
        key=(
            args.optimizer_seed
            if getattr(args, "personality_key", None) is None
            else args.personality_key
        ),
        daughter_drift=getattr(args, "personality_drift", 200.0),
        film_bound=getattr(args, "film_bound", None),
    )
    set_labor_film_bound(conditioning.film_bound, actor, task, costs)
    run_personality: LaborPersonality | LaborPersonalityAssignment = (
        LaborPersonality() if conditioning.alpha is None else conditioning.assignment(0)
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(
        list(task.parameters()) + list(costs.parameters()), lr=args.critic_lr
    )
    sampling = torch.Generator().manual_seed(args.optimizer_seed)
    shuffle = np.random.default_rng(args.optimizer_seed)
    bc_summary = None
    if getattr(args, "pretrain_bc_turns", 0) > 0:
        from .labor_bc import collect_heuristic_bc_dataset, pretrain_labor_actor_bc

        ds = collect_heuristic_bc_dataset(
            normalizer=normalizer,
            world_seeds=[world],
            turns=args.pretrain_bc_turns,
            personality=run_personality,
            allow_vm_scale=False,
        )
        bc_summary = pretrain_labor_actor_bc(
            actor=actor,
            dataset=ds,
            lr=args.pretrain_bc_lr,
            epochs=args.pretrain_bc_epochs,
            target_log_std=args.pretrain_bc_target_log_std,
            rng=shuffle,
            inactive_role_weight=getattr(args, "pretrain_bc_inactive_role_weight", 0.0),
            label_smoothing=getattr(args, "pretrain_bc_label_smoothing", 1e-6),
            inverse_freq_power=getattr(args, "pretrain_bc_inverse_freq_power", 0.5),
        )
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    trainer_state = _trainer_checkpoint(
        actor,
        task,
        costs,
        normalizer,
        layout,
        actor_optimizer,
        critic_optimizer,
        personality=run_personality,
        optimizer_seed=args.optimizer_seed,
        iteration=0,
        sampling_generator=sampling,
        shuffle_rng=shuffle,
        critic_training_world_seeds=(),
        critic_training_independent_unit_ids=(),
    )
    if bc_summary is not None:
        trainer_state["heuristic_bc_pretraining"] = bc_summary
    _save_checkpoint(args.out, trainer_state)


def _command_pretrain_bc(args: argparse.Namespace) -> None:
    from .labor_bc import collect_heuristic_bc_dataset, pretrain_labor_actor_bc
    from .labor_device import device_record, setup

    device = setup(getattr(args, "device", "cpu"))
    if device.type != "cpu" and not _bc_dataset_directories(args):
        raise ValueError(
            "--worlds collects from the game and runs on CPU; use --dataset for --device"
        )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _require_trainer_checkpoint(state)
    layout, actor, task, costs, normalizer = _models_from_checkpoint(state)
    personality = labor_checkpoint_personality(state)

    critic_optimizer = torch.optim.Adam(list(task.parameters()) + list(costs.parameters()), lr=3e-4)
    critic_optimizer.load_state_dict(state["critic_optimizer"])

    sampling = torch.Generator()
    sampling.set_state(state["sampling_generator_state"])
    shuffle = np.random.default_rng(state["optimizer_seed"])
    shuffle.bit_generator.state = copy.deepcopy(state["shuffle_rng_state"])

    dataset_dirs = _bc_dataset_directories(args)
    if bool(dataset_dirs) == (getattr(args, "worlds", None) is not None):
        raise ValueError("pretrain-bc requires exactly one of --dataset/--dataset-list or --worlds")
    worlds = (
        []
        if dataset_dirs
        else require_training_worlds(
            _parse_worlds(args.worlds, allow_vm_scale=getattr(args, "allow_vm_scale", False)),
            "labor pretrain-bc",
        )
    )

    writer = None
    if getattr(args, "economic_trace", None) is not None:
        from .economic_trace import EconomicTraceWriter

        writer = EconomicTraceWriter(
            args.economic_trace,
            {
                "sourceCheckpoint": str(args.checkpoint),
                "scope": "labor-pretrain-bc-economic-trace",
            },
        )

    if dataset_dirs:
        from .labor_teacher_data import load_labor_datasets

        dataset = load_labor_datasets(
            dataset_dirs,
            expected_normalizer_digest=_normalization_digest(normalizer),
            drop_reserved=bool(getattr(args, "drop_reserved_worlds", False)),
            family_weights=_family_weights(args),
            family_normalizer=normalizer,
        )
    else:
        dataset = collect_heuristic_bc_dataset(
            normalizer=normalizer,
            world_seeds=worlds,
            turns=args.turns,
            personality=personality,
            economic_trace=writer,
            allow_vm_scale=getattr(args, "allow_vm_scale", False),
            reset_options=_labor_reset_options(args),
        )
    actor.to(device)  # data stays on the host; minibatches move per step
    bc_summary = pretrain_labor_actor_bc(
        actor=actor,
        dataset=dataset,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_log_std=args.target_log_std,
        max_grad_norm=args.max_grad_norm,
        rng=shuffle,
        inactive_role_weight=getattr(args, "inactive_role_weight", 0.0),
        label_smoothing=getattr(args, "label_smoothing", 1e-6),
        inverse_freq_power=getattr(args, "inverse_freq_power", 0.5),
        **_bc_lr_schedule_kwargs(args),
        **_bc_loss_balancing_kwargs(args),
        **_bc_stability_kwargs(args, task, costs),
        **(
            {"family_schedule": schedule}
            if (schedule := _family_schedule(args, state, "actor", args.epochs)) is not None
            else {}
        ),
    )
    if getattr(args, "economic_trace", None) is not None:
        bc_summary["economic_trace_dir"] = str(args.economic_trace)
    if (dataset.get("collection") or {}).get("family_weighting") is not None:
        bc_summary["family_weighting"] = dataset["collection"]["family_weighting"]
    actor.to("cpu")  # device-agnostic checkpoint
    if device.type != "cpu":
        bc_summary["device"] = device_record(device)

    actor_lr = float(state.get("actor_optimizer", {}).get("param_groups", [{}])[0].get("lr", 3e-4))
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)

    trainer_state = _trainer_checkpoint(
        actor,
        task,
        costs,
        normalizer,
        layout,
        actor_optimizer,
        critic_optimizer,
        personality=personality,
        optimizer_seed=int(state["optimizer_seed"]),
        iteration=int(state["iteration"]),
        sampling_generator=sampling,
        shuffle_rng=shuffle,
        critic_training_world_seeds=state["critic_training_world_seeds"],
        critic_training_independent_unit_ids=state["critic_training_independent_unit_ids"],
    )
    trainer_state["heuristic_bc_pretraining"] = bc_summary
    _carry_family_schedules(trainer_state, state, "actor", bc_summary.get("family_weight_schedule"))
    _save_checkpoint(args.out, trainer_state)


def _bc_lr_schedule_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """``--lr-schedule``: absent or constant keeps the historical fixed --lr (no kwarg)."""
    kind = getattr(args, "lr_schedule", None) or "constant"
    if kind == "constant":
        return {}
    schedule: dict[str, Any] = {
        "kind": kind,
        "warmup_fraction": float(args.lr_warmup_fraction),
        "final_fraction": float(args.lr_final_fraction),
        "start_step": int(args.lr_schedule_start_step),
    }
    if getattr(args, "lr_schedule_total_steps", None) is not None:
        schedule["total_steps"] = int(args.lr_schedule_total_steps)
    return {"lr_schedule": schedule}


def _bc_loss_balancing_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """``pretrain-bc --loss-balancing gradnorm|grouped``: actor BC task balancing (absent or
    none: no kwarg, the historical loss bit for bit). ``--gradnorm-min-weight`` /
    ``--gradnorm-max-weight`` default to each mode's own clamp (flat: per task; grouped: per
    group)."""
    mode = getattr(args, "loss_balancing", None) or "none"
    if mode == "none":
        return {}
    if mode not in ("gradnorm", "grouped"):
        raise ValueError("--loss-balancing must be none, gradnorm or grouped")
    from .labor_actor_gradnorm import ActorGradNorm, ActorGroupedGradNorm

    cls = ActorGradNorm if mode == "gradnorm" else ActorGroupedGradNorm
    common: dict[str, Any] = {
        "alpha": float(args.gradnorm_alpha),
        "ema": float(args.gradnorm_ema),
        "lr": float(args.gradnorm_lr),
        "init": str(args.gradnorm_init),
        "min_rows": int(args.gradnorm_min_rows),
        "every": int(args.gradnorm_every),
        "shared": str(args.gradnorm_shared),
        "hold_below": float(args.gradnorm_hold_below),
    }
    for name in ("min_weight", "max_weight"):
        value = getattr(args, f"gradnorm_{name}", None)
        if value is not None:
            common[name] = float(value)
    if mode == "grouped":
        common["inner"] = str(getattr(args, "gradnorm_inner", "none"))
        for name in ("inner_min_weight", "inner_max_weight"):
            value = getattr(args, f"gradnorm_{name}", None)
            if value is not None:
                common[name] = float(value)
    elif getattr(args, "gradnorm_inner", "none") != "none":
        raise ValueError("--gradnorm-inner applies to --loss-balancing grouped only")
    return {"loss_balancing": cls(**common)}


def _bc_stability_kwargs(args: argparse.Namespace, task: Any, costs: Any) -> dict[str, Any]:
    """Stability monitor for BC refits: on for r5 checkpoints (or an explicit log)."""
    from .labor_stability import is_r5_critic, stability_config_from_args

    config = stability_config_from_args(args)
    if config is None:
        return {}
    if config == "auto":
        if not is_r5_critic(task):
            return {}
        from .labor_stability import StabilityConfig

        config = StabilityConfig()
    return {"stability": config, "frozen_critics": {"task_critic": task, "cost_critics": costs}}


def _critic_stability_kwargs(args: argparse.Namespace, actor: Any) -> dict[str, Any]:
    """Stability, frozen-actor check and exact resume for pretrain-critic (dataset path)."""
    from .labor_stability import stability_config_from_args

    kwargs: dict[str, Any] = {
        "stability": stability_config_from_args(args),
        "frozen_actor": actor,
    }
    if getattr(args, "resume_state", None) is not None:
        kwargs["resume_path"] = str(args.resume_state)
        kwargs["resume_every"] = int(getattr(args, "resume_every", 100))
    if getattr(args, "max_steps", None) is not None:
        kwargs["max_steps"] = int(args.max_steps)
    if getattr(args, "progress", False):
        kwargs["progress"] = True
    return kwargs


def _load_validation_dataset(args: argparse.Namespace, normalizer: Any, task: Any) -> Any:
    """``--validation-dataset``: held-out rows for monitoring only, read by the evaluation
    loader (never the training loader) and never a training --dataset directory."""
    paths = getattr(args, "validation_dataset", None)
    if not paths:
        return None
    training = {Path(p).resolve() for p in (args.dataset or [])}
    shared = sorted(str(p) for p in paths if Path(p).resolve() in training)
    if shared:
        raise ValueError(f"--validation-dataset repeats training --dataset directories {shared}")
    from .labor_pretrain_eval import _load_dataset

    return _tail_normalize(
        args,
        _load_dataset(
            [Path(p) for p in paths],
            normalizer,
            require_privileged=task.privileged_layout is not None,
        ),
    )


def _write_abort_receipt(out: Path, error: Any) -> None:
    import json

    path = out.with_suffix(out.suffix + ".stability-abort.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(error.summary, indent=2, sort_keys=True), encoding="utf-8")


def _cost_weight_kwargs(args: argparse.Namespace) -> dict[str, float]:
    """Pass a non-default cost-loss weight only, keeping default runs byte-identical."""
    weight = float(getattr(args, "cost_loss_weight", 1.0))
    return {} if weight == 1.0 else {"cost_loss_weight": weight}


def _task_weight_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """``--task-loss-weights`` / ``--qol-loss-weight``: per-head task loss weights.

    Only non-default settings are passed, keeping default runs byte-identical.
    """
    from .labor_bc import AutoQoLLossWeight

    weights: dict[str, float] = {}
    text = getattr(args, "task_loss_weights", None)
    if text:
        for item in str(text).split(","):
            name, sep, value = item.strip().partition("=")
            if not sep or name.strip() not in LABOR_OBJECTIVES:
                raise ValueError(
                    f"--task-loss-weights takes head=weight for heads {list(LABOR_OBJECTIVES)}"
                )
            if name.strip() in weights:
                raise ValueError(f"--task-loss-weights repeats {name.strip()}")
            weights[name.strip()] = float(value)
    qol = getattr(args, "qol_loss_weight", None)
    kwargs: dict[str, Any] = {}
    if qol is not None:
        if "qol_years" in weights:
            raise ValueError("give the QoL weight in --task-loss-weights or --qol-loss-weight")
        if str(qol).strip().lower() == "auto":
            kwargs["qol_loss_weight_auto"] = AutoQoLLossWeight(
                target_share=float(getattr(args, "qol_weight_target_share", 0.3)),
                ema=float(getattr(args, "qol_weight_ema", 0.3)),
            )
        else:
            weights["qol_years"] = float(qol)
    for name, value in weights.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"task loss weight {name} must be finite and positive")
    if any(value != 1.0 for value in weights.values()) or (
        weights and "qol_loss_weight_auto" in kwargs
    ):
        kwargs["task_loss_weights"] = weights
    return kwargs


def _balance_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """``--loss-balancing gradnorm`` / ``--grad-clip per-encoder`` (user-approved 29 Oct).

    Only non-default settings are passed, keeping default runs byte-identical."""
    from .labor_bc import GradNormBalancing

    kwargs: dict[str, Any] = {}
    mode = getattr(args, "loss_balancing", None) or "none"
    if mode == "gradnorm":
        if str(getattr(args, "qol_loss_weight", None) or "").strip().lower() == "auto":
            raise ValueError(
                "--loss-balancing gradnorm and --qol-loss-weight auto exclude each other"
            )
        kwargs["loss_balancing"] = GradNormBalancing(
            alpha=float(getattr(args, "gradnorm_alpha", 1.5)),
            lr=float(getattr(args, "gradnorm_lr", 0.025)),
        )
    elif mode != "none":
        raise ValueError("--loss-balancing must be none or gradnorm")
    clip = getattr(args, "grad_clip", None) or "global"
    if clip != "global":
        kwargs["grad_clip"] = clip
    return kwargs


def _critic_level_kwargs(args: argparse.Namespace, state: Mapping[str, Any]) -> dict[str, Any]:
    """``--qol-head-bias-init`` / ``--td-step0-targets`` (Fable audit section 2, 28 Oct).

    An explicit value wins ("none" / "head" turn a fix off). Unset, ``--campaign-r5`` turns
    both on when they apply (a logistic bounded QoL head for the bias, TD(lambda) targets for
    the step-0 rule) and the head is untrained (no earlier critic pretraining in the input
    checkpoint); without ``--campaign-r5`` both stay off. Only active fixes are passed, so
    every default run keeps its historical arguments and bytes.
    """
    campaign = bool(getattr(args, "campaign_r5", False))
    head = labor_checkpoint_qol_head(state)
    trained = state.get("heuristic_critic_pretraining") is not None
    kwargs: dict[str, Any] = {}
    bias = getattr(args, "qol_head_bias_init", None)
    if bias is None:
        bias = (
            "mean-target"
            if campaign and head is not None and head.output == "logistic" and not trained
            else "none"
        )
    elif bias == "mean-target" and trained:
        raise ValueError(
            "--qol-head-bias-init mean-target initialises an untrained QoL head; this "
            "checkpoint already carries critic pretraining"
        )
    if bias == "mean-target":
        kwargs["qol_head_bias_init"] = "mean-target"
    step0 = getattr(args, "td_step0_targets", None)
    if step0 is None:
        step0 = (
            "mc"
            if campaign and head is not None and head.target == "td-lambda" and not trained
            else "head"
        )
    if step0 == "mc":
        kwargs["td_step0_targets"] = "mc"
    return kwargs


def _false_terminal_policy(args: argparse.Namespace) -> str:
    """``--false-terminal-policy``: explicit value, else mask under ``--campaign-r5``."""
    policy = getattr(args, "false_terminal_policy", None)
    if policy is None:
        policy = "mask" if getattr(args, "campaign_r5", False) else "none"
    return str(policy)


def _tail_normalize(args: argparse.Namespace, dataset: Any) -> Any:
    """Loader corrections of critic data: ``--false-terminal-policy`` (labor_bc
    ``correct_false_terminals``), then ``--tail-normalize world-factor`` (divide stored
    randomized tails by f_w). Both off returns the loaded dataset itself."""
    from .labor_bc import apply_loader_corrections

    return apply_loader_corrections(
        dataset,
        false_terminal_policy=_false_terminal_policy(args),
        tail_normalize=getattr(args, "tail_normalize", None) or "none",
    )


def _pretrain_critic_tail(args: argparse.Namespace) -> Any:
    """Randomized-tail settings for in-process (--worlds) critic collection, or None."""
    from .labor_bc import RandomizedTailSettings

    if getattr(args, "return_tail", "truncated-zero") == "randomized":
        return RandomizedTailSettings()
    if getattr(args, "require_randomized_tail", False):
        raise ValueError("--require-randomized-tail needs --return-tail randomized with --worlds")
    return None


def _command_pretrain_critic(args: argparse.Namespace) -> None:
    from .labor_bc import pretrain_critic_on_heuristic_next_states
    from .labor_device import device_record, setup

    device = setup(getattr(args, "device", "cpu"))
    if device.type != "cpu" and getattr(args, "dataset", None) is None:
        raise ValueError(
            "--worlds collects from the game and runs on CPU; use --dataset for --device"
        )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _require_trainer_checkpoint(state)
    layout, actor, task, costs, normalizer = _models_from_checkpoint(state)
    personality = labor_checkpoint_personality(state)

    sampling = torch.Generator()
    sampling.set_state(state["sampling_generator_state"])
    shuffle = np.random.default_rng(state["optimizer_seed"])
    shuffle.bit_generator.state = copy.deepcopy(state["shuffle_rng_state"])

    dataset_dir = getattr(args, "dataset", None)
    if (dataset_dir is None) == (getattr(args, "worlds", None) is None):
        raise ValueError("pretrain-critic requires exactly one of --dataset or --worlds")
    if dataset_dir is not None:
        from .labor_bc import pretrain_critic_on_dataset, require_critic_teacher_dataset

        teacher = _tail_normalize(
            args,
            _load_labor_teacher_dataset(
                args, normalizer, require_privileged=task.privileged_layout is not None
            ),
        )
        continuations = bool(getattr(args, "learner_state_continuations", False))
        scenarios = bool(getattr(args, "curated_scenarios", False))
        require_critic_teacher_dataset(
            teacher,
            allow_learner_state_continuation=continuations,
            allow_curated_scenarios=scenarios,
        )
        worlds = list(teacher["collection"]["world_seeds"])
        validation = _load_validation_dataset(args, normalizer, task)
        if getattr(args, "select_best_validation", False) and validation is None:
            raise ValueError("--select-best-validation needs --validation-dataset")
        best_sink: dict[str, Any] | None = {} if validation is not None else None
        from .labor_stability import StabilityAbort

        # Critics move to the device; the frozen actor stays on CPU (not in the graph).
        task.to(device)
        costs.to(device)
        try:
            summary = pretrain_critic_on_dataset(
                task_critic=task,
                cost_critics=costs,
                dataset=teacher,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_grad_norm=args.max_grad_norm,
                rng=shuffle,
                require_randomized_tail=bool(getattr(args, "require_randomized_tail", False)),
                **_cost_weight_kwargs(args),
                **({"allow_learner_state_continuation": True} if continuations else {}),
                **({"allow_curated_scenarios": True} if scenarios else {}),
                **(
                    {"family_schedule": schedule}
                    if (schedule := _family_schedule(args, state, "critic", args.epochs))
                    is not None
                    else {}
                ),
                **_critic_stability_kwargs(args, actor),
                **_task_weight_kwargs(args),
                **_balance_kwargs(args),
                **_critic_level_kwargs(args, state),
                **({} if validation is None else {"validation_dataset": validation}),
                **({} if best_sink is None else {"best_validation_sink": best_sink}),
            )
        except StabilityAbort as error:
            _write_abort_receipt(args.out, error)
            raise
        if validation is not None:
            summary["validation"]["datasets"] = [str(p) for p in args.validation_dataset]
            summary["validation"]["dataset_digest"] = validation["collection"].get("dataset_digest")
        summary["dataset_digest"] = teacher["collection"]["dataset_digest"]
        if teacher["collection"].get("family_weighting") is not None:
            summary.setdefault("family_weighting", {})["loader_shares"] = teacher["collection"][
                "family_weighting"
            ]["shares"]
        task.to("cpu")  # device-agnostic checkpoint
        costs.to("cpu")
        if device.type != "cpu":
            summary["device"] = device_record(device)
    else:
        best_sink = None
        if _cost_weight_kwargs(args):
            raise ValueError("--cost-loss-weight is supported only with --dataset")
        if (
            _task_weight_kwargs(args)
            or _balance_kwargs(args)
            or getattr(args, "dominance_excess_loss", False)
        ):
            raise ValueError(
                "--task-loss-weights / --qol-loss-weight / --loss-balancing / --grad-clip / "
                "--dominance-excess-loss are supported only with --dataset"
            )
        if getattr(args, "select_best_validation", False):
            raise ValueError("--select-best-validation needs --validation-dataset")
        if getattr(args, "validation_dataset", None):
            raise ValueError("--validation-dataset is supported only with --dataset")
        if (
            _critic_level_kwargs(args, state)
            or (getattr(args, "tail_normalize", None) or "none") != "none"
            or _false_terminal_policy(args) != "none"
        ):
            raise ValueError(
                "--qol-head-bias-init / --td-step0-targets / --tail-normalize / "
                "--false-terminal-policy (and their --campaign-r5 defaults) are supported only "
                "with --dataset"
            )
        worlds = require_training_worlds(
            _parse_worlds(args.worlds, allow_vm_scale=getattr(args, "allow_vm_scale", False)),
            "labor pretrain-critic",
        )
        summary = pretrain_critic_on_heuristic_next_states(
            task_critic=task,
            cost_critics=costs,
            normalizer=normalizer,
            world_seeds=worlds,
            turns=args.turns,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_grad_norm=args.max_grad_norm,
            personality=personality,
            rng=shuffle,
            allow_vm_scale=getattr(args, "allow_vm_scale", False),
            reset_options=_labor_reset_options(args),
            tail=_pretrain_critic_tail(args),
        )

    actor_lr = float(state.get("actor_optimizer", {}).get("param_groups", [{}])[0].get("lr", 3e-4))
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    actor_optimizer.load_state_dict(state["actor_optimizer"])

    critic_lr = float(
        state.get("critic_optimizer", {}).get("param_groups", [{}])[0].get("lr", 3e-4)
    )
    critic_optimizer = torch.optim.Adam(
        list(task.parameters()) + list(costs.parameters()), lr=critic_lr
    )

    updated_critic_worlds = sorted(set(state["critic_training_world_seeds"]) | set(worlds))
    trainer_state = _trainer_checkpoint(
        actor,
        task,
        costs,
        normalizer,
        layout,
        actor_optimizer,
        critic_optimizer,
        personality=personality,
        optimizer_seed=int(state["optimizer_seed"]),
        iteration=int(state["iteration"]),
        sampling_generator=sampling,
        shuffle_rng=shuffle,
        critic_training_world_seeds=updated_critic_worlds,
        critic_training_independent_unit_ids=state["critic_training_independent_unit_ids"],
    )
    trainer_state["heuristic_critic_pretraining"] = summary
    _carry_family_schedules(
        trainer_state,
        state,
        "critic",
        (summary.get("family_weighting") or {}).get("schedule"),
    )

    def rebuild_with_best() -> dict[str, Any]:
        task.load_state_dict(best_sink["task"])
        costs.load_state_dict(best_sink["costs"])
        return _trainer_checkpoint(
            actor,
            task,
            costs,
            normalizer,
            layout,
            actor_optimizer,
            critic_optimizer,
            personality=personality,
            optimizer_seed=int(state["optimizer_seed"]),
            iteration=int(state["iteration"]),
            sampling_generator=sampling,
            shuffle_rng=shuffle,
            critic_training_world_seeds=updated_critic_worlds,
            critic_training_independent_unit_ids=state["critic_training_independent_unit_ids"],
        )

    _save_critic_checkpoints(
        Path(args.out),
        trainer_state,
        best_sink,
        select=bool(getattr(args, "select_best_validation", False)),
        rebuild_with_best=rebuild_with_best,
    )


def _save_critic_checkpoints(
    out: Path,
    trainer_state: dict[str, Any],
    best_sink: Mapping[str, Any] | None,
    *,
    select: bool,
    rebuild_with_best: Any,
) -> None:
    """Write pretrain-critic's checkpoint(s).

    Without a best-validation record (no --validation-dataset): ``out`` only, as before.
    With one: ``<out>.best-val.pt`` holds the same trainer checkpoint with the critic
    weights of the epoch of lowest validation total; ``out`` holds the final weights, or,
    with ``select`` (--select-best-validation), the best ones while the final weights go to
    ``<out>.final.pt``. The training trajectory and the final weights are unchanged.
    """
    if not best_sink:
        _save_checkpoint(out, trainer_state)
        return
    summary = dict(trainer_state["heuristic_critic_pretraining"])
    best_path = out.with_name(out.name + ".best-val.pt")
    final_path = out.with_name(out.name + ".final.pt")
    record = {
        "epoch": int(best_sink["epoch"]),
        "step": int(best_sink["step"]),
        "validation_total": float(best_sink["total"]),
        "best_validation_checkpoint": str(best_path),
        "selected": "best-validation" if select else "final",
        **({"final_checkpoint": str(final_path)} if select else {}),
    }
    summary["best_validation"] = record
    final_state = dict(
        trainer_state, heuristic_critic_pretraining=dict(summary, checkpoint_weights="final")
    )
    # Written before the best weights are loaded: the payload's state-dict tensors share
    # storage with the live parameters.
    _save_checkpoint(final_path if select else out, final_state)
    best_state = rebuild_with_best()
    for key, value in trainer_state.items():
        best_state.setdefault(key, value)
    best_state["heuristic_critic_pretraining"] = dict(
        summary, checkpoint_weights=f"best-validation-epoch-{record['epoch']}"
    )
    _save_checkpoint(best_path, best_state)
    if select:
        _save_checkpoint(out, best_state)


def _command_collect(args: argparse.Namespace) -> None:
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _require_trainer_checkpoint(state)
    _, actor, task, costs, normalizer = _models_from_checkpoint(state)
    personality = labor_checkpoint_personality(state)
    sampling = torch.Generator()
    sampling.set_state(state["sampling_generator_state"])
    before_rng = sampling.get_state().clone()
    worlds = require_training_worlds(
        _parse_worlds(args.worlds, allow_vm_scale=getattr(args, "allow_vm_scale", False)),
        "labor PPO rollout collection",
    )
    writer = None
    if getattr(args, "economic_trace", None) is not None:
        from .economic_trace import EconomicTraceWriter

        writer = EconomicTraceWriter(
            args.economic_trace,
            {
                "sourceCheckpoint": str(args.checkpoint),
                "scope": "labor-rollout-economic-trace",
            },
        )
    rows, bootstrap = collect_labor_rollout(
        actor=actor,
        task_critic=task,
        cost_critics=costs,
        normalizer=normalizer,
        world_seeds=worlds,
        turns=args.turns,
        generator=sampling,
        personality=personality,
        economic_trace=writer,
        max_self_destruct_rewinds=getattr(args, "max_self_destruct_rewinds", 0),
        allow_vm_scale=getattr(args, "allow_vm_scale", False),
    )
    behavior = labor_behavior_configuration(personality)
    policy_sha = behavior_policy_digest(actor, behavior)
    critic_sha = labor_critic_digest(task, costs)
    norm_sha = _normalization_digest(normalizer)
    batch, task_targets, cost_targets = build_labor_training_batch(
        rows,
        bootstrap,
        personality=personality,
        cost_multipliers={
            "feasibility": args.feasibility_multiplier,
            "settlement_guard": args.guard_multiplier,
        },
        behavior_policy_sha256=policy_sha,
        critic_sha256=critic_sha,
        normalization_sha256=norm_sha,
        lam=args.lam,
    )
    metadata = {
        "schemaVersion": 1,
        "sourceCheckpointSha256": checkpoint_hash(args.checkpoint),
        "worldSeeds": worlds,
        "turns": args.turns,
        "behaviorConfiguration": behavior,
        "behaviorPolicyDigest": policy_sha,
        "advantageCriticDigest": critic_sha,
        "normalizationBindingDigest": norm_sha,
        "samplingGeneratorStateBefore": before_rng,
        "samplingGeneratorStateAfter": sampling.get_state(),
        "scope": "local-diagnostic-rollout-not-readiness",
    }
    if getattr(args, "economic_trace", None) is not None:
        metadata["economicTraceDir"] = str(args.economic_trace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_state_dict_rollout(batch, task_targets, cost_targets, metadata), args.out)


def _command_train(args: argparse.Namespace) -> None:
    if getattr(args, "critic_epochs", 1) < 1:
        raise ValueError("critic_epochs must be >= 1")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _require_trainer_checkpoint(state)
    layout, actor, task, costs, normalizer = _models_from_checkpoint(state)
    personality = labor_checkpoint_personality(state)
    batch, task_targets, cost_targets, rollout_metadata = _load_rollout(args.rollout)
    behavior = labor_behavior_configuration(personality)
    if rollout_metadata.get("sourceCheckpointSha256") != checkpoint_hash(args.checkpoint):
        raise ValueError("labor rollout belongs to another checkpoint")
    if batch.behavior_policy_digest != behavior_policy_digest(actor, behavior):
        raise ValueError("labor rollout behavior policy is stale")
    if batch.advantage_critic_digest != labor_critic_digest(task, costs):
        raise ValueError("labor rollout advantages came from other critic weights")
    if batch.normalization_binding_digest != _normalization_digest(normalizer):
        raise ValueError("labor rollout normalization binding is stale")

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(
        list(task.parameters()) + list(costs.parameters()), lr=args.critic_lr
    )
    actor_optimizer.load_state_dict(state["actor_optimizer"])
    critic_optimizer.load_state_dict(state["critic_optimizer"])
    anchor = load_labor_anchor(args.anchor, "train-anchor")
    if (
        anchor.metadata["source"]["normalizationBindingDigest"]
        != batch.normalization_binding_digest
    ):
        raise ValueError("labor train anchor uses another normalizer")
    gate: LaborActorGate | None = None
    gate_report = None
    if args.calibration_report is not None:
        gate_report = _load_json(args.calibration_report)
        _gate_data_identity(gate_report)
        if gate_report["actorGateOpen"]:
            gate = LaborActorGate.from_report(gate_report)
    current_training_worlds, current_training_units = _validate_training_data_separation(
        anchor, batch, rollout_metadata, state, gate_report
    )
    if gate is not None:
        gate.require_current(
            actor,
            behavior,
            LABOR_REGISTRY_DIGEST,
            batch.advantage_critic_digest,
            batch.normalization_binding_digest,
        )
    anchor_rows = np.arange(len(anchor), dtype=np.int64)
    critic_epochs = getattr(args, "critic_epochs", 1)
    initial_critic_loss: float | None = None
    critic_loss = torch.tensor(0.0)
    independent_heads: dict[str, torch.Tensor] = {}
    for _ in range(critic_epochs):
        critic_optimizer.zero_grad()
        td_task, td_cost = _critic_td_losses(task, costs, batch, task_targets, cost_targets)
        independent, independent_heads = labor_anchor_loss(task, costs, anchor, anchor_rows)
        critic_loss = td_task + td_cost + args.anchor_weight * independent
        if initial_critic_loss is None:
            initial_critic_loss = float(critic_loss.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(task.parameters()) + list(costs.parameters()), args.max_grad_norm
        )
        critic_optimizer.step()

    actor_stats: dict[str, Any]
    if args.calibration_report is None:
        actor_stats = {"actorGateOpen": False, "reason": "calibration-report-not-supplied"}
    elif gate is None:
        actor_stats = {
            "actorGateOpen": False,
            "reason": "calibration-report-closed",
            "calibrationReportSha256": checkpoint_hash(args.calibration_report),
        }
    else:
        shuffle = np.random.default_rng()
        shuffle.bit_generator.state = copy.deepcopy(state["shuffle_rng_state"])
        actor_stats = update_labor_policy(
            policy_module=actor,
            labor_actor=actor,
            optimizer=actor_optimizer,
            batch=batch,
            nonlabor_scorer=lambda tensors, sample: _score_nonlabor(actor, tensors, sample),
            rng=shuffle,
            epochs=args.epochs,
            batch_size=args.batch_size,
            clip=args.clip,
            entropy_weight=args.entropy_weight,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            audit_rows=args.audit_rows,
            actor_gate=gate,
            behavior_configuration=behavior,
        )
        state["shuffle_rng_state"] = copy.deepcopy(shuffle.bit_generator.state)

    sampling = torch.Generator()
    sampling.set_state(rollout_metadata["samplingGeneratorStateAfter"])
    shuffle_state = state["shuffle_rng_state"]
    shuffle_for_save = np.random.default_rng()
    shuffle_for_save.bit_generator.state = copy.deepcopy(shuffle_state)
    next_state = _trainer_checkpoint(
        actor,
        task,
        costs,
        normalizer,
        layout,
        actor_optimizer,
        critic_optimizer,
        personality=personality,
        optimizer_seed=int(state["optimizer_seed"]),
        iteration=int(state["iteration"]) + 1,
        sampling_generator=sampling,
        shuffle_rng=shuffle_for_save,
        critic_training_world_seeds=sorted(current_training_worlds),
        critic_training_independent_unit_ids=sorted(current_training_units),
    )
    next_state["last_update"] = {
        "rolloutSha256": checkpoint_hash(args.rollout),
        "anchorSha256": checkpoint_hash(args.anchor),
        "criticEpochs": int(critic_epochs),
        "initialCriticLoss": float(initial_critic_loss),
        "criticLoss": float(critic_loss.detach()),
        "independentHeadLosses": {
            name: float(value.detach()) for name, value in independent_heads.items()
        },
        "actor": actor_stats,
        "scope": "local-update-validation-not-readiness",
    }
    _save_checkpoint(args.out, next_state)


def _command_fit_privileged_normalizer(args: argparse.Namespace) -> None:
    """Fit the frozen per-family privileged moments on fitting-world teacher shards."""
    from .labor_bc import CRITIC_TEACHER_SOURCE_KIND
    from .labor_privileged import (
        PRIVILEGED_NORMALIZER_KIND,
        LaborPrivilegedNormalizer,
        dead_mask_audit,
    )
    from .labor_teacher_data import load_labor_dataset

    excluded: set[int] = set()
    for directory in args.calibration_dataset or []:
        manifest = json.loads((Path(directory) / "dataset.json").read_text())
        excluded |= {int(world) for world in manifest["shards"]}
    parts = []
    for directory in args.dataset:
        part = load_labor_dataset(Path(directory), require_privileged=True)
        if part["collection"]["source_kind"] != CRITIC_TEACHER_SOURCE_KIND:
            raise ValueError("privileged moments are fitted on heuristic-teacher shards only")
        parts.append(part)
    from .labor_privileged import shard_keys

    arrays = {key: torch.cat([part[key] for part in parts], dim=0) for key in shard_keys()}
    worlds = sorted({int(w) for part in parts for w in part["row_world"].reshape(-1).tolist()})
    normalizer = LaborPrivilegedNormalizer.fit(
        arrays,
        worlds=worlds,
        excluded_worlds=excluded,
        fit_record={
            "dataset_digests": [part["collection"]["dataset_digest"] for part in parts],
            "excluded_calibration_worlds": sorted(excluded),
            "source_kind": CRITIC_TEACHER_SOURCE_KIND,
        },
    )
    report = {
        "normalizer_dead_mask": normalizer.dead_mask_report(),
        "data_audit": dead_mask_audit(arrays),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(
        {
            "kind": PRIVILEGED_NORMALIZER_KIND,
            "privileged_normalizer": normalizer.state_dict(),
            "dead_mask_report": report["normalizer_dead_mask"],
        },
        temporary,
    )
    os.replace(temporary, out_path)
    if getattr(args, "report", None) is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    summary = report["normalizer_dead_mask"]
    print(
        json.dumps(
            {
                "out": str(out_path),
                "digest": normalizer.digest(),
                "worlds": worlds,
                "rows": normalizer.fit_record["rows"],
                "dead_total": summary["dead_total"],
                "fields_total": summary["fields_total"],
                "dead_by_family": {
                    name: family["dead_fields"] for name, family in summary["families"].items()
                },
            },
            sort_keys=True,
        )
    )


def _command_privileged_audit(args: argparse.Namespace) -> None:
    """Data-only dead-mask audit of stored privileged arrays (no fit-world policy)."""
    from .labor_privileged import dead_mask_audit, shard_keys
    from .labor_teacher_data import load_labor_dataset

    parts = [
        load_labor_dataset(Path(d), require_privileged=True, training=False) for d in args.dataset
    ]
    arrays = {key: torch.cat([part[key] for part in parts], dim=0) for key in shard_keys()}
    report = dead_mask_audit(arrays)
    report["worlds"] = sorted({int(w) for p in parts for w in p["row_world"].reshape(-1).tolist()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({name: f["dead_fields"] for name, f in report["families"].items()}))


def _fit_normalizer_world(
    seed: int, turns: int, options_state: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """One world's teacher-driven observations for fit-normalizer (picklable, for --jobs)."""
    from .labor_bc import LaborResetOptions

    options = LaborResetOptions.from_mapping(dict(options_state))
    all_live: list[Mapping[str, Any]] = []
    with (
        LaborModel9Bridge() as labor_bridge,
        Model9Bridge(economic_diagnostics=True) as heur_bridge,
    ):
        step = labor_bridge.reset(seed, **options.reset_kwargs())
        for turn in range(turns):
            live = [
                v
                for v in step.get("villages", [])
                if v.get("observation") is not None and not v.get("terminated")
            ]
            if not live:
                break
            all_live.extend(live)
            if turn == turns - 1:
                break

            snap = labor_bridge._bridge.capture()
            heur_bridge.reset(
                seed, movement=options.movement, withheld=options.withheld, snapshot=snap
            )
            h_step = heur_bridge.step({})
            h_villages = {int(v["camp"]): v for v in h_step.get("villages", [])}
            econ_by_camp = h_step.get("economicDiagnostics") or {}

            actions: dict[int, Mapping[str, Any]] = {}
            for v in live:
                camp = int(v["camp"])
                cat, _ = labor_observation_fields(v["observation"])
                h_v = h_villages.get(camp, {})
                econ = econ_by_camp.get(str(camp))
                _, _, action = extract_heuristic_labor_action(cat, h_v, econ)
                actions[camp] = action

            step, _ = labor_bridge.step_labor(actions)
    return all_live


def _command_fit_normalizer(args: argparse.Namespace) -> None:
    from .labor_worlds import require_training_worlds

    worlds = _parse_worlds(args.worlds, allow_vm_scale=args.allow_vm_scale)
    # Fitting partition only (r5 manifest section 10): never calibration (90180-90199),
    # development (2000-2059), reserved held-out (90154-90159) or sealed (3000-3059) worlds.
    worlds = require_training_worlds(worlds, "fit-normalizer")
    refused = [w for w in worlds if 90180 <= w <= 90199 or 2000 <= w <= 2059]
    if refused:
        raise ValueError(f"fit-normalizer refuses calibration/development worlds {refused}")
    turns = int(args.turns)
    if turns < 1:
        raise ValueError("turns must be >= 1")
    options = _labor_reset_options(args)

    jobs = max(1, int(getattr(args, "jobs", 1) or 1))
    options_state = options.state_dict()
    if jobs == 1:
        per_world = [_fit_normalizer_world(int(seed), turns, options_state) for seed in worlds]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=min(jobs, len(worlds))) as pool:
            # map preserves world order, so the pooled rows are identical to the serial path.
            per_world = list(
                pool.map(
                    _fit_normalizer_world,
                    [int(w) for w in worlds],
                    [turns] * len(worlds),
                    [options_state] * len(worlds),
                )
            )
    all_live: list[Mapping[str, Any]] = [row for rows in per_world for row in rows]

    if not all_live:
        raise ValueError("no live villages observed to fit normalizer")

    batch = observation_batch(all_live)
    normalizer = Normalizer.fit_bounded(batch)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(
        {
            "feature_contract": BOUNDED_FEATURE_CONTRACT,
            "normalizer": normalizer.state_dict(),
            "source": {
                "fit_command": "labor_training fit-normalizer",
                "partition": "fitting",
                "world_seeds": [int(w) for w in worlds],
                "turns": turns,
                "rows": len(all_live),
                "reset_options": options.state_dict(),
                "learner_commit": _fit_normalizer_commit(),
                "bridge_sha256": _fit_normalizer_bridge_sha(),
            },
        },
        temporary,
    )
    os.replace(temporary, out_path)


def _fit_normalizer_commit() -> str:
    try:
        from .provenance import code_commit

        return str(code_commit())
    except Exception:  # provenance is best-effort for a diagnostic record
        return "unknown"


def _fit_normalizer_bridge_sha() -> str:
    from .bridge import DEFAULT_BRIDGE

    path = Path(os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE", str(DEFAULT_BRIDGE)))
    return checkpoint_hash(path) if path.exists() else "missing"


def argument_parser() -> argparse.ArgumentParser:
    from .labor_device import add_device_argument

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--normalizer-checkpoint", type=Path, required=True)
    initialize.add_argument("--world-seed", type=int, required=True)
    initialize.add_argument("--optimizer-seed", type=int, required=True)
    initialize.add_argument("--actor-lr", type=float, default=3e-4)
    initialize.add_argument("--critic-lr", type=float, default=3e-4)
    initialize.add_argument("--pretrain-bc-turns", type=int, default=0)
    initialize.add_argument("--pretrain-bc-epochs", type=int, default=25)
    initialize.add_argument("--pretrain-bc-lr", type=float, default=1e-3)
    initialize.add_argument("--pretrain-bc-target-log-std", type=float, default=-2.0)
    initialize.add_argument("--pretrain-bc-inactive-role-weight", type=float, default=0.0)
    initialize.add_argument("--pretrain-bc-label-smoothing", type=float, default=1e-6)
    initialize.add_argument("--pretrain-bc-inverse-freq-power", type=float, default=0.5)
    initialize.add_argument(
        "--personality",
        default="balanced",
        help="labor personality distribution: 'balanced' or 'dirichlet:ALPHA' "
        "(per founding lineage, keyed; docs/LABOR_PERSONALITY.md)",
    )
    initialize.add_argument(
        "--personality-key",
        type=int,
        default=None,
        help="keyed personality stream (default: the optimizer seed)",
    )
    initialize.add_argument(
        "--personality-drift",
        type=float,
        default=200.0,
        help="fission-daughter Dirichlet drift concentration (0 copies the parent)",
    )
    initialize.add_argument(
        "--film-bound",
        type=float,
        default=None,
        help="optional tanh bound on FiLM gamma/beta (off by default)",
    )
    initialize.add_argument(
        "--critic-value-support",
        default=None,
        metavar="LOW,HIGH,BINS",
        help="versioned task-critic symlog clamp and bin count (default -2,9,255)",
    )
    initialize.add_argument(
        "--qol-head",
        default=None,
        metavar="two-hot|OUTPUT[,TARGET[,LAMBDA]]",
        help="bounded QoL value head (amendment r5-b item 4): OUTPUT logistic or silu-clamp, "
        "TARGET monte-carlo (default) or td-lambda, LAMBDA default 0.8; omitted or two-hot "
        "keeps the historical symlog two-hot QoL head",
    )
    initialize.add_argument(
        "--qol-head-scale",
        type=float,
        default=None,
        help="bounded QoL head range in units of W (V = scale W sigmoid(z)); default 1.0, "
        "or 2.0 under --campaign-r5; recorded in the contract as qolHeadScale when not 1",
    )
    initialize.add_argument(
        "--campaign-r5",
        action="store_true",
        default=False,
        help="declare the campaign-r5 task-critic value support -0.5,2.5,255 "
        "(LABOR_CAMPAIGN_R5_VALUE_SUPPORT) and, with a bounded --qol-head, the QoL head "
        "scale 2.0 (LABOR_CAMPAIGN_R5_QOL_HEAD_SCALE) and the shared QoL encoder "
        "(LABOR_CAMPAIGN_R5_QOL_ENCODER; --qol-encoder separate overrides). r5 baseline: "
        "shared encoder + pretrain-critic --qol-loss-weight 30",
    )
    initialize.add_argument(
        "--qol-encoder",
        choices=("shared", "separate"),
        default=None,
        help="task-critic QoL encoder: 'shared' (historical: QoL and population-years read one "
        "encoder) or 'separate' (QoL head reads its own full copy of the task encoder, "
        "privileged path included; contract field qolEncoder). Default shared, also under "
        "--campaign-r5 (r5 baseline: shared + --qol-loss-weight 30)",
    )
    initialize.add_argument(
        "--privileged-normalizer",
        type=Path,
        default=None,
        help="fit-privileged-normalizer output: build the three critics with the critic-only "
        "privileged block (schema 1); the actor never receives it",
    )
    initialize.add_argument("--out", type=Path, required=True)
    initialize.set_defaults(handler=_command_initialize)

    pretrain_bc = commands.add_parser("pretrain-bc")
    pretrain_bc.add_argument("--checkpoint", type=Path, required=True)
    pretrain_bc.add_argument(
        "--worlds", default=None, help="comma-separated local diagnostic seeds"
    )
    pretrain_bc.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=None,
        help=(
            "published labor dataset directory (alternative to --worlds); repeat to aggregate "
            "the teacher data with learner-state (DAgger) rounds at equal source mass"
        ),
    )
    pretrain_bc.add_argument(
        "--dataset-list",
        type=Path,
        default=None,
        help="text file with one labor dataset directory per line (added after --dataset)",
    )
    pretrain_bc.add_argument("--players", type=int, default=8)
    pretrain_bc.add_argument("--width", type=int, default=None)
    pretrain_bc.add_argument("--height", type=int, default=None)
    pretrain_bc.add_argument("--separation", type=int, default=None)
    pretrain_bc.add_argument(
        "--no-movement",
        action="store_true",
        default=False,
        help="named mechanical fixture only; movement is on for real data",
    )
    pretrain_bc.add_argument("--turns", type=int, default=25)
    pretrain_bc.add_argument("--epochs", type=int, default=25)
    pretrain_bc.add_argument("--batch-size", type=int, default=64)
    pretrain_bc.add_argument("--lr", type=float, default=1e-3)
    pretrain_bc.add_argument(
        "--lr-schedule",
        choices=("constant", "warmup-cosine"),
        default="constant",
        help="warmup-cosine (brief section 5.1): --lr is the peak; linear warm-up over "
        "--lr-warmup-fraction of the program, cosine decay to --lr-final-fraction of the peak. "
        "constant (default) keeps the historical fixed --lr",
    )
    pretrain_bc.add_argument("--lr-warmup-fraction", type=float, default=0.1)
    pretrain_bc.add_argument("--lr-final-fraction", type=float, default=0.1)
    pretrain_bc.add_argument(
        "--lr-schedule-total-steps",
        type=int,
        default=None,
        help="optimizer steps of the whole schedule (default: this fit's epochs x steps/epoch)",
    )
    pretrain_bc.add_argument(
        "--lr-schedule-start-step",
        type=int,
        default=0,
        help="global step this fit starts at (a resumed fit or a later round continues it)",
    )
    pretrain_bc.add_argument("--target-log-std", type=float, default=-2.0)
    pretrain_bc.add_argument("--max-grad-norm", type=float, default=1.0)
    pretrain_bc.add_argument("--inactive-role-weight", type=float, default=0.0)
    pretrain_bc.add_argument("--label-smoothing", type=float, default=1e-6)
    pretrain_bc.add_argument("--inverse-freq-power", type=float, default=0.5)
    pretrain_bc.add_argument(
        "--loss-balancing",
        choices=("none", "gradnorm", "grouped"),
        default="none",
        help="gradnorm: flat GradNorm over the actor BC tasks (specialty, the 13 role activity "
        "CEs, movement, fission, trade gate) on the shared actor encoder; grouped: GradNorm over "
        "the specialty/movement/activity groups with role-balanced weights inside activity "
        "(labor_actor_gradnorm.ActorGroupedGradNorm); weights logged in the stability JSONL. "
        "none (default) keeps the historical loss",
    )
    pretrain_bc.add_argument(
        "--gradnorm-inner",
        choices=("none", "gradnorm"),
        default="none",
        help="grouped only: none keeps the role weights inside activity fixed at "
        "--gradnorm-init; gradnorm equalises them too",
    )
    pretrain_bc.add_argument("--gradnorm-inner-min-weight", type=float, default=None)
    pretrain_bc.add_argument(
        "--gradnorm-inner-max-weight",
        type=float,
        default=None,
        help="grouped only: cap of a role weight inside activity (default: ActorGroupedGradNorm)",
    )
    pretrain_bc.add_argument(
        "--gradnorm-alpha",
        type=float,
        default=0.0,
        help="GradNorm asymmetry alpha (0: pure gradient-norm equalisation)",
    )
    pretrain_bc.add_argument(
        "--gradnorm-lr",
        type=float,
        default=0.25,
        help="log-space step of each weight toward its ideal per update, in (0, 1] (0.25)",
    )
    pretrain_bc.add_argument(
        "--gradnorm-ema",
        type=float,
        default=0.1,
        help="EMA rate of the task gradient norms and losses, in (0, 1] (0.1)",
    )
    pretrain_bc.add_argument(
        "--gradnorm-init",
        choices=("uniform", "sqrt", "share"),
        default="uniform",
        help="initial role weights: uniform per role, sqrt of the resident share, or share "
        "(the current loss); normalised to the current loss scale",
    )
    pretrain_bc.add_argument(
        "--gradnorm-min-rows",
        type=int,
        default=4,
        help="a role's weight is updated only when the batch staffs it in this many rows (4)",
    )
    pretrain_bc.add_argument(
        "--gradnorm-min-weight",
        type=float,
        default=None,
        help="weight clamp floor (default: flat 0.01 per task, grouped 0.1 per group)",
    )
    pretrain_bc.add_argument(
        "--gradnorm-max-weight",
        type=float,
        default=None,
        help="weight clamp cap (default: flat 100 per task, grouped 3 per group)",
    )
    pretrain_bc.add_argument(
        "--gradnorm-every",
        type=int,
        default=1,
        help="controller update every N optimizer steps (1)",
    )
    pretrain_bc.add_argument(
        "--gradnorm-shared",
        choices=("trunk", "last-layer"),
        default="trunk",
        help="parameters the task gradient norms are measured on: the whole actor encoder "
        "(default) or its last global attention layer + output norm (cheaper, biased against "
        "the role activity tasks)",
    )
    pretrain_bc.add_argument(
        "--gradnorm-hold-below",
        type=float,
        default=0.05,
        help="hold a task whose resident-scale-normalised gradient norm is below this fraction "
        "of the median (solved/constant heads such as fission and trade gate); 0 disables",
    )
    pretrain_bc.add_argument("--allow-vm-scale", action="store_true", default=False)
    pretrain_bc.add_argument(
        "--drop-reserved-worlds",
        action="store_true",
        default=False,
        help="drop shards of the reserved held-out worlds 90154-90159 instead of refusing the "
        "dataset (recorded in the summary)",
    )
    pretrain_bc.add_argument(
        "--family-weight-decay",
        choices=("none", "linear", "cosine"),
        default=None,
        help="decay of each family up-weight toward 1 as 1 + (w - 1) s(t) (default cosine when "
        "--family-weights is set); keyed on the global optimizer step and continued from the "
        "input checkpoint's schedule across rounds",
    )
    pretrain_bc.add_argument(
        "--family-weight-decay-epochs",
        type=float,
        default=None,
        help="epochs over which s goes from 1 to 0 (default: the continued schedule's, else this "
        "fit's --epochs)",
    )
    pretrain_bc.add_argument(
        "--family-weight-restart",
        action="store_true",
        default=False,
        help="start the family weight schedule at full weight instead of continuing it",
    )
    pretrain_bc.add_argument(
        "--family-weights",
        default=None,
        help="up-weight scenario-family rows of --dataset (labor_scenarios): 'default' (the "
        "proposed weights), 'family=w,...' or 'default,family=w'; each row takes its max family "
        "weight, then every (source, world) keeps its mass",
    )
    pretrain_bc.add_argument(
        "--economic-trace",
        type=Path,
        default=None,
        help="opt-in directory to stream prospective economic diagnostics trace",
    )
    pretrain_bc.add_argument("--out", type=Path, required=True)
    add_device_argument(pretrain_bc)
    add_stability_arguments(pretrain_bc)
    pretrain_bc.set_defaults(handler=_command_pretrain_bc)

    pretrain_critic = commands.add_parser("pretrain-critic")
    pretrain_critic.add_argument("--checkpoint", type=Path, required=True)
    pretrain_critic.add_argument(
        "--worlds", default=None, help="comma-separated local diagnostic seeds"
    )
    pretrain_critic.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=None,
        help="published labor_teacher_data directory (alternative to --worlds); repeat only with "
        "--learner-state-continuations",
    )
    pretrain_critic.add_argument(
        "--learner-state-continuations",
        action="store_true",
        default=False,
        help="admit learner-state teacher continuations (labor_dagger --teacher-continuation, "
        "row_source 3) as V^rho targets, alone or aggregated with teacher trajectories",
    )
    pretrain_critic.add_argument(
        "--curated-scenarios",
        action="store_true",
        default=False,
        help="admit curated Neolithic scenario trajectories (labor_scenarios, row_source 4) as "
        "V targets, alone or aggregated; sampling then gives each source equal mass",
    )
    pretrain_critic.add_argument("--players", type=int, default=8)
    pretrain_critic.add_argument("--width", type=int, default=None)
    pretrain_critic.add_argument("--height", type=int, default=None)
    pretrain_critic.add_argument("--separation", type=int, default=None)
    pretrain_critic.add_argument(
        "--no-movement",
        action="store_true",
        default=False,
        help="named mechanical fixture only; movement is on for real data",
    )
    pretrain_critic.add_argument("--turns", type=int, default=25)
    pretrain_critic.add_argument("--epochs", type=int, default=50)
    pretrain_critic.add_argument("--batch-size", type=int, default=64)
    pretrain_critic.add_argument("--lr", type=float, default=1e-3)
    pretrain_critic.add_argument("--max-grad-norm", type=float, default=1.0)
    pretrain_critic.add_argument("--allow-vm-scale", action="store_true", default=False)
    pretrain_critic.add_argument(
        "--drop-reserved-worlds",
        action="store_true",
        default=False,
        help="drop shards of the reserved held-out worlds 90154-90159 instead of refusing the "
        "dataset (recorded in the summary)",
    )
    pretrain_critic.add_argument(
        "--family-weight-decay",
        choices=("none", "linear", "cosine"),
        default=None,
        help="decay of each family up-weight toward 1 as 1 + (w - 1) s(t) (default cosine when "
        "--family-weights is set); keyed on the global optimizer step and continued from the "
        "input checkpoint's schedule across rounds",
    )
    pretrain_critic.add_argument(
        "--family-weight-decay-epochs",
        type=float,
        default=None,
        help="epochs over which s goes from 1 to 0 (default: the continued schedule's, else this "
        "fit's --epochs)",
    )
    pretrain_critic.add_argument(
        "--family-weight-restart",
        action="store_true",
        default=False,
        help="start the family weight schedule at full weight instead of continuing it",
    )
    pretrain_critic.add_argument(
        "--family-weights",
        default=None,
        help="up-weight scenario-family rows of --dataset (labor_scenarios): 'default' (the "
        "proposed weights), 'family=w,...' or 'default,family=w'; each row takes its max family "
        "weight, then every (source, world) keeps its mass",
    )
    pretrain_critic.add_argument(
        "--return-tail",
        choices=("truncated-zero", "randomized"),
        default="truncated-zero",
        help="--worlds only: close living cutoffs with the preregistered randomized teacher tail "
        "(the --dataset path uses the tail its shards declare)",
    )
    pretrain_critic.add_argument(
        "--require-randomized-tail",
        action="store_true",
        default=False,
        help="refuse targets whose living cutoffs are not closed by the randomized tail",
    )
    pretrain_critic.add_argument(
        "--cost-loss-weight",
        type=float,
        default=1.0,
        help="multiplier on the summed cost-critic MSE (default 1.0, unchanged)",
    )
    pretrain_critic.add_argument(
        "--task-loss-weights",
        default=None,
        help="per-head task loss weights 'qol_years=W1,population_years=W2' (default 1.0 each, "
        "unchanged); they scale each head's term of the optimizer objective only",
    )
    pretrain_critic.add_argument(
        "--qol-loss-weight",
        default=None,
        help="QoL task loss weight: a number, or 'auto' (GradNorm-lite: every --stability-every "
        "steps set the weight so QoL's share of the task-encoder gradient targets "
        "--qol-weight-target-share, log-space EMA, clamped to [1, 200]; needs the monitor)",
    )
    pretrain_critic.add_argument("--qol-weight-target-share", type=float, default=0.3)
    pretrain_critic.add_argument("--qol-weight-ema", type=float, default=0.3)
    pretrain_critic.add_argument(
        "--loss-balancing",
        choices=("none", "gradnorm"),
        default="none",
        help="gradnorm: GradNorm (Chen et al. 2018) learns the task-head weights on the shared "
        "task encoder's last layer, starting from the --qol-loss-weight ratio renormalised to "
        "sum to the head count; target mean grad norm x (relative inverse training rate)^alpha; "
        "under the stability monitor gradient_dominance becomes advisory and "
        "task_progress_stall gates (user-approved 29 Oct). Default none (unchanged)",
    )
    pretrain_critic.add_argument(
        "--gradnorm-alpha", type=float, default=1.5, help="GradNorm asymmetry alpha (1.5)"
    )
    pretrain_critic.add_argument(
        "--gradnorm-lr", type=float, default=0.025, help="Adam rate of the GradNorm weights (0.025)"
    )
    pretrain_critic.add_argument(
        "--grad-clip",
        choices=("global", "per-encoder"),
        default="global",
        help="per-encoder: clip the task critic (task encoder + heads) and each cost critic "
        "(encoder + head) to --max-grad-norm separately instead of one global clip. "
        "Default global (unchanged)",
    )
    pretrain_critic.add_argument(
        "--dominance-excess-loss",
        action="store_true",
        default=False,
        help="the task-encoder dominance detector judges QoL progress on its excess BCE "
        "(BCE - H(p*)) instead of the raw BCE (off by default: changes an approved check's "
        "measurement; recorded in the stability config)",
    )
    pretrain_critic.add_argument(
        "--campaign-r5",
        action="store_true",
        default=False,
        help="r5 campaign fit: turns on --qol-head-bias-init mean-target and --td-step0-targets "
        "mc (each only when it applies and the QoL head is untrained) and "
        "--false-terminal-policy mask, unless given explicitly",
    )
    pretrain_critic.add_argument(
        "--false-terminal-policy",
        choices=("none", "mask"),
        default=None,
        help="mask: a terminal row whose (world, camp, trajectory) has later rows (the old "
        "simulator's Population <= 0 while hunters are away) is non-terminal; with the next "
        "turn present its return is rebuilt through it, otherwise it and the earlier rows of its "
        "trajectory are dropped from value targets; counts recorded in the summary. Applies to "
        "--dataset and --validation-dataset. Default none, mask under --campaign-r5",
    )
    pretrain_critic.add_argument(
        "--qol-head-bias-init",
        choices=("none", "mean-target"),
        default=None,
        help="mean-target: set the logistic QoL head's output bias to logit(mean p*), p* = "
        "Monte Carlo target / (scale W), mean under the fit's sampling probabilities, before "
        "the first step (fresh fits; recorded). Default none, mean-target under --campaign-r5",
    )
    pretrain_critic.add_argument(
        "--td-step0-targets",
        choices=("head", "mc"),
        default=None,
        help="TD(lambda) QoL targets before the first scheduled refresh: head (historical "
        "step-0 bootstrap through the untrained head) or mc (the Monte Carlo returns). "
        "Default head, mc under --campaign-r5",
    )
    pretrain_critic.add_argument(
        "--tail-normalize",
        choices=("none", "world-factor"),
        default="none",
        help="world-factor: on load, divide each stored randomized tail (both task heads, "
        "values and standard errors) by its world's roulette schedule factor "
        "(labor_pretrain_eval.tail_schedule_factor); applies to --dataset and "
        "--validation-dataset; the factors are recorded in the summary. Default none",
    )
    pretrain_critic.add_argument(
        "--select-best-validation",
        action="store_true",
        default=False,
        help="with --validation-dataset: --out holds the critic of the epoch of lowest "
        "validation total (also <out>.best-val.pt); the final weights go to <out>.final.pt",
    )
    pretrain_critic.add_argument("--out", type=Path, required=True)
    add_device_argument(pretrain_critic)
    add_stability_arguments(pretrain_critic)
    pretrain_critic.add_argument(
        "--resume-state",
        type=Path,
        default=None,
        help="exact mid-fit resume file (written every --resume-every steps; resumed if present)",
    )
    pretrain_critic.add_argument("--resume-every", type=int, default=100)
    pretrain_critic.add_argument(
        "--max-steps", type=int, default=None, help="stop after N optimizer steps (probes)"
    )
    pretrain_critic.add_argument("--progress", action="store_true", default=False)
    pretrain_critic.add_argument(
        "--validation-dataset",
        type=Path,
        nargs="+",
        default=None,
        help="held-out dataset dir(s) (e.g. the calibration batches 90180-90199) evaluated at "
        "every epoch end for monitoring only (kind=validation records and the non-gating "
        "validation_divergence alarm); never training data, the fit is bitwise unchanged",
    )
    pretrain_critic.set_defaults(handler=_command_pretrain_critic)

    collect = commands.add_parser("collect")
    collect.add_argument("--checkpoint", type=Path, required=True)
    collect.add_argument("--worlds", required=True, help="comma-separated local diagnostic seeds")
    collect.add_argument("--turns", type=int, default=25)
    collect.add_argument("--lam", type=float, default=0.95)
    collect.add_argument("--feasibility-multiplier", type=float, default=1.0)
    collect.add_argument("--guard-multiplier", type=float, default=1.0)
    collect.add_argument("--max-self-destruct-rewinds", type=int, default=0)
    collect.add_argument("--allow-vm-scale", action="store_true", default=False)
    collect.add_argument(
        "--economic-trace",
        type=Path,
        default=None,
        help="opt-in directory to stream prospective economic diagnostics trace",
    )
    collect.add_argument("--out", type=Path, required=True)
    collect.set_defaults(handler=_command_collect)

    train = commands.add_parser("train")
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--rollout", type=Path, required=True)
    train.add_argument("--anchor", type=Path, required=True)
    train.add_argument("--calibration-report", type=Path)
    train.add_argument("--anchor-weight", type=float, default=1.0)
    train.add_argument("--actor-lr", type=float, default=3e-4)
    train.add_argument("--critic-lr", type=float, default=3e-4)
    train.add_argument("--critic-epochs", type=int, default=1)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--clip", type=float, default=0.2)
    train.add_argument("--entropy-weight", type=float, default=0.0)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--target-kl", type=float, default=0.02)
    train.add_argument("--audit-rows", type=int, default=64)
    train.add_argument("--out", type=Path, required=True)
    train.set_defaults(handler=_command_train)

    fit_normalizer = commands.add_parser("fit-normalizer")
    fit_normalizer.add_argument(
        "--worlds", required=True, help="comma-separated local diagnostic seeds"
    )
    fit_normalizer.add_argument("--turns", type=int, default=25)
    fit_normalizer.add_argument("--players", type=int, default=8)
    fit_normalizer.add_argument("--width", type=int, default=None)
    fit_normalizer.add_argument("--height", type=int, default=None)
    fit_normalizer.add_argument("--separation", type=int, default=None)
    fit_normalizer.add_argument(
        "--no-movement", action="store_true", help="named mechanical fixture only"
    )
    fit_normalizer.add_argument(
        "--jobs", type=int, default=1, help="parallel worlds (rows pooled in world order)"
    )
    fit_normalizer.add_argument("--allow-vm-scale", action="store_true", default=False)
    fit_normalizer.add_argument("--out", type=Path, required=True)
    fit_normalizer.set_defaults(handler=_command_fit_normalizer)

    fit_privileged = commands.add_parser(
        "fit-privileged-normalizer",
        help="fit frozen per-family privileged moments on fitting-world teacher shards",
    )
    fit_privileged.add_argument(
        "--dataset",
        type=Path,
        action="append",
        required=True,
        help="published privileged teacher dataset on fitting worlds (repeatable)",
    )
    fit_privileged.add_argument(
        "--calibration-dataset",
        type=Path,
        action="append",
        default=None,
        help="calibration dataset whose worlds must not be fitted (repeatable)",
    )
    fit_privileged.add_argument("--report", type=Path, default=None, help="dead-mask report JSON")
    fit_privileged.add_argument("--out", type=Path, required=True)
    fit_privileged.set_defaults(handler=_command_fit_privileged_normalizer)

    privileged_audit = commands.add_parser(
        "privileged-dead-mask-audit", help="report constant/dead named privileged fields in data"
    )
    privileged_audit.add_argument("--dataset", type=Path, action="append", required=True)
    privileged_audit.add_argument("--out", type=Path, required=True)
    privileged_audit.set_defaults(handler=_command_privileged_audit)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    torch.set_num_threads(4)
    args = argument_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
