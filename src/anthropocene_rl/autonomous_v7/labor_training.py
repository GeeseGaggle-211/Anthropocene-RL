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
    LABOR_COSTS,
    LABOR_OBJECTIVES,
    LaborActor,
    LaborCostCritics,
    LaborEncoderLayout,
    LaborPersonality,
    LaborTaskCritic,
    Normalizer,
    labor_checkpoint_payload,
    labor_log_prob,
    labor_policy_tensors,
    load_labor_checkpoint_models,
    nonlabor_entropy,
    nonlabor_log_prob,
    sample_labor_policy,
    sample_nonlabor_policy,
    schema12_initial_action,
    schema12_retry_action,
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
from .labor_value_calibration import (
    labor_anchor_loss,
    labor_critic_digest,
    load_labor_anchor,
)
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


def labor_behavior_configuration(personality: LaborPersonality) -> dict[str, object]:
    """Complete behavior signature for sampled schema-12 policy data."""

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
            name: critic.expectation(name, logits[name]).detach().cpu()
            for name in LABOR_OBJECTIVES
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
    personality: LaborPersonality,
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

    retry_rows: dict[int, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float,
                                dict[str, float], dict[str, float]]] = {}

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
    personality: LaborPersonality,
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
    step: Mapping[str, Any], normalizer: Any, task_critic: LaborTaskCritic,
    cost_critics: LaborCostCritics, personality: LaborPersonality,
) -> dict[int, dict[str, float]]:
    live = [
        village for village in step["villages"]
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
    personality: LaborPersonality | None = None,
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
            for _ in range(turns):
                step, rows = _sample_world_step(
                    bridge=bridge,
                    step=step,
                    actor=actor,
                    task_critic=task_critic,
                    cost_critics=cost_critics,
                    normalizer=normalizer,
                    personality=declared,
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
                            diag.get("transitionId")
                            or f"{seed}:{camp_id}:{int(step['turn'])-1}"
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
                step, normalizer, task_critic, cost_critics, declared
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
            if not isinstance(tail, Mapping) or set(tail) != set(
                (*LABOR_OBJECTIVES, *LABOR_COSTS)
            ):
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
                    outer_discount=torch.tensor([
                        math.exp(-row.years / contracts.HORIZON_YEARS)
                    ]),
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

    initial_combined = combine_labor_advantages(
        {name: initial_advantages[name] for name in LABOR_OBJECTIVES},
        {name: initial_advantages[name] for name in LABOR_COSTS},
        personality=personality,
        cost_multipliers=cost_multipliers,
    )
    retry_combined = combine_labor_advantages(
        {name: torch.tensor(retry_advantages[name]) for name in LABOR_OBJECTIVES},
        {name: torch.tensor(retry_advantages[name]) for name in LABOR_COSTS},
        personality=personality,
        cost_multipliers=cost_multipliers,
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
        "retry": {
            name: torch.tensor(retry_task_targets[name]) for name in LABOR_OBJECTIVES
        },
    }
    cost_targets = {
        "initial": dict(initial_cost_targets),
        "retry": {
            name: torch.tensor(retry_cost_targets[name]) for name in LABOR_COSTS
        },
    }
    return batch, task_targets, cost_targets


def _state_dict_rollout(
    batch: LaborAttemptPolicyBatch,
    task_targets: Mapping[str, Mapping[str, torch.Tensor]],
    cost_targets: Mapping[str, Mapping[str, torch.Tensor]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": LABOR_ROLLOUT_KIND,
        "metadata": dict(metadata),
        "batch": {
            name: getattr(batch, name)
            for name in batch.__dataclass_fields__
        },
        "task_targets": dict(task_targets),
        "cost_targets": dict(cost_targets),
    }


def _load_rollout(path: Path) -> tuple[
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


def _models_from_checkpoint(state: Mapping[str, Any]) -> tuple[
    LaborEncoderLayout, LaborActor, LaborTaskCritic, LaborCostCritics, Any
]:
    layout = LaborEncoderLayout.from_state(state.get("labor_encoder_layout"))
    actor = LaborActor(layout)
    task = LaborTaskCritic(layout)
    costs = LaborCostCritics(layout)
    _, normalizer = load_labor_checkpoint_models(
        state, actor, task, costs, expected_layout=layout
    )
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
    personality: LaborPersonality,
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
        checked_seeds(
            rollout_metadata.get("worldSeeds", []), "labor rollout training worlds"
        )
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


def _command_initialize(args: argparse.Namespace) -> None:
    source = torch.load(args.normalizer_checkpoint, map_location="cpu", weights_only=True)
    normalizer = load_checkpoint_normalizer(
        source, expected_feature_contract=BOUNDED_FEATURE_CONTRACT
    )
    world = checked_seeds([args.world_seed], "labor initialize world")[0]
    with LaborModel9Bridge() as bridge:
        step = bridge.reset(world)
        live = [row for row in step["villages"] if row.get("observation") is not None]
        if not live:
            raise ValueError("labor initialization world has no live policy rows")
        _, tensors = labor_policy_tensors(live, normalizer)
    layout = LaborEncoderLayout(
        int(tensors["core"].shape[1]),
        LABOR_SPECIALTY_FEATURE_WIDTH,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
    )
    torch.manual_seed(args.optimizer_seed)
    actor, task, costs = LaborActor(layout), LaborTaskCritic(layout), LaborCostCritics(layout)
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
            personality=LaborPersonality(),
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
        personality=LaborPersonality(),
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

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _require_trainer_checkpoint(state)
    layout, actor, task, costs, normalizer = _models_from_checkpoint(state)
    personality = LaborPersonality.from_mapping(
        state["labor_model_contract"]["personality"]
    )

    critic_optimizer = torch.optim.Adam(
        list(task.parameters()) + list(costs.parameters()), lr=3e-4
    )
    critic_optimizer.load_state_dict(state["critic_optimizer"])

    sampling = torch.Generator()
    sampling.set_state(state["sampling_generator_state"])
    shuffle = np.random.default_rng(state["optimizer_seed"])
    shuffle.bit_generator.state = copy.deepcopy(state["shuffle_rng_state"])

    worlds = _parse_worlds(
        args.worlds, allow_vm_scale=getattr(args, "allow_vm_scale", False)
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

    dataset = collect_heuristic_bc_dataset(
        normalizer=normalizer,
        world_seeds=worlds,
        turns=args.turns,
        personality=personality,
        economic_trace=writer,
        allow_vm_scale=getattr(args, "allow_vm_scale", False),
    )
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
    )
    if getattr(args, "economic_trace", None) is not None:
        bc_summary["economic_trace_dir"] = str(args.economic_trace)

    actor_lr = float(
        state.get("actor_optimizer", {}).get("param_groups", [{}])[0].get("lr", 3e-4)
    )
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
    _save_checkpoint(args.out, trainer_state)


def _command_pretrain_critic(args: argparse.Namespace) -> None:
    from .labor_bc import pretrain_critic_on_heuristic_next_states

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _require_trainer_checkpoint(state)
    layout, actor, task, costs, normalizer = _models_from_checkpoint(state)
    personality = LaborPersonality.from_mapping(
        state["labor_model_contract"]["personality"]
    )

    sampling = torch.Generator()
    sampling.set_state(state["sampling_generator_state"])
    shuffle = np.random.default_rng(state["optimizer_seed"])
    shuffle.bit_generator.state = copy.deepcopy(state["shuffle_rng_state"])

    worlds = _parse_worlds(
        args.worlds, allow_vm_scale=getattr(args, "allow_vm_scale", False)
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
    )

    actor_lr = float(
        state.get("actor_optimizer", {}).get("param_groups", [{}])[0].get("lr", 3e-4)
    )
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
    _save_checkpoint(args.out, trainer_state)


def _command_collect(args: argparse.Namespace) -> None:
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _require_trainer_checkpoint(state)
    _, actor, task, costs, normalizer = _models_from_checkpoint(state)
    personality = LaborPersonality.from_mapping(state["labor_model_contract"]["personality"])
    sampling = torch.Generator()
    sampling.set_state(state["sampling_generator_state"])
    before_rng = sampling.get_state().clone()
    worlds = _parse_worlds(
        args.worlds, allow_vm_scale=getattr(args, "allow_vm_scale", False)
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
        cost_multipliers={"feasibility": args.feasibility_multiplier,
                          "settlement_guard": args.guard_multiplier},
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
    personality = LaborPersonality.from_mapping(state["labor_model_contract"]["personality"])
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


def _command_fit_normalizer(args: argparse.Namespace) -> None:
    worlds = _parse_worlds(args.worlds, allow_vm_scale=args.allow_vm_scale)
    turns = int(args.turns)
    if turns < 1:
        raise ValueError("turns must be >= 1")

    all_live: list[Mapping[str, Any]] = []
    for seed in worlds:
        with (
            LaborModel9Bridge() as labor_bridge,
            Model9Bridge(economic_diagnostics=True) as heur_bridge,
        ):
            step = labor_bridge.reset(seed)
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
                heur_bridge.reset(seed, snapshot=snap)
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
        },
        temporary,
    )
    os.replace(temporary, out_path)


def argument_parser() -> argparse.ArgumentParser:
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
    initialize.add_argument("--out", type=Path, required=True)
    initialize.set_defaults(handler=_command_initialize)

    pretrain_bc = commands.add_parser("pretrain-bc")
    pretrain_bc.add_argument("--checkpoint", type=Path, required=True)
    pretrain_bc.add_argument(
        "--worlds", required=True, help="comma-separated local diagnostic seeds"
    )
    pretrain_bc.add_argument("--turns", type=int, default=25)
    pretrain_bc.add_argument("--epochs", type=int, default=25)
    pretrain_bc.add_argument("--batch-size", type=int, default=64)
    pretrain_bc.add_argument("--lr", type=float, default=1e-3)
    pretrain_bc.add_argument("--target-log-std", type=float, default=-2.0)
    pretrain_bc.add_argument("--max-grad-norm", type=float, default=1.0)
    pretrain_bc.add_argument("--inactive-role-weight", type=float, default=0.0)
    pretrain_bc.add_argument("--label-smoothing", type=float, default=1e-6)
    pretrain_bc.add_argument("--inverse-freq-power", type=float, default=0.5)
    pretrain_bc.add_argument("--allow-vm-scale", action="store_true", default=False)
    pretrain_bc.add_argument(
        "--economic-trace",
        type=Path,
        default=None,
        help="opt-in directory to stream prospective economic diagnostics trace",
    )
    pretrain_bc.add_argument("--out", type=Path, required=True)
    pretrain_bc.set_defaults(handler=_command_pretrain_bc)

    pretrain_critic = commands.add_parser("pretrain-critic")
    pretrain_critic.add_argument("--checkpoint", type=Path, required=True)
    pretrain_critic.add_argument(
        "--worlds", required=True, help="comma-separated local diagnostic seeds"
    )
    pretrain_critic.add_argument("--turns", type=int, default=25)
    pretrain_critic.add_argument("--epochs", type=int, default=50)
    pretrain_critic.add_argument("--batch-size", type=int, default=64)
    pretrain_critic.add_argument("--lr", type=float, default=1e-3)
    pretrain_critic.add_argument("--max-grad-norm", type=float, default=1.0)
    pretrain_critic.add_argument("--allow-vm-scale", action="store_true", default=False)
    pretrain_critic.add_argument("--out", type=Path, required=True)
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
    fit_normalizer.add_argument("--allow-vm-scale", action="store_true", default=False)
    fit_normalizer.add_argument("--out", type=Path, required=True)
    fit_normalizer.set_defaults(handler=_command_fit_normalizer)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    torch.set_num_threads(4)
    args = argument_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
