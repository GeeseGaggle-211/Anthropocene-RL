"""Native fork producer for ``observed-anchor-v1`` value targets.

This is a narrow refresh adapter, not a readiness or selection runner.  It reuses the
reviewed Gate3/4 in-process fork, sampled-policy, shaping, and frozen geometric-tail
implementation.  Worlds are the independent units; camps, regimes, frozen states,
action paths, alternatives, and stopping draws remain nested observations.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from . import contracts, ppo
from .bridge import DEFAULT_BRIDGE, Model9Bridge
from .gate34_checkpoint_collector import (
    FROZEN_PROTOCOL,
    BranchResult,
    EndpointReceipt,
    EndpointRuntime,
    PersonalityState,
    _collect_frozen_state,
    _derived_seed,
    _discounted_terms,
    _exact_terminal_tail,
    _live_villages,
    _outcome_rows,
    _tail_assumptions,
    _zero_tail,
    sample_training_actions,
)
from .gate34_continuation_collector import (
    CampReward,
    RandomizedTailEstimate,
    aggregate_camp_prefix,
    estimate_randomized_tail,
    raw_camp_reward_absolute_bound,
    shaped_nonnegative_component_absolute_bound,
    shaped_welfare_reward_absolute_bound,
)
from .model import (
    BOUNDED_FEATURE_CONTRACT,
    Actor,
    VectorCritic,
    load_checkpoint_normalizer,
)
from .protocol import seed_list
from .provenance import ProvenanceError, checkpoint_hash, digest
from .value_calibration import (
    ESTIMATOR_SOURCE_KIND,
    GUARD_COST,
    LEARNER_CONTRACT,
    RANK_SEMANTICS,
    REWARD_SEMANTICS,
    TERMINATION_SEMANTICS,
    VALUE_HEADS,
    behavior_policy_digest,
    build_randomized_estimator_dataset,
    normalization_binding,
    policy_digest,
)

Json = dict[str, Any]


@dataclass(frozen=True)
class FrozenStateCapture:
    world_seed: int
    optimizer_seed: int
    regime_id: str
    decision_step: int
    turn: int
    parent_digest: str
    opening_camps: tuple[int, ...]
    founded_turns: Mapping[int, int]
    parent_inputs: Mapping[str, NDArray[Any]]
    paths: tuple[BranchResult, ...]


def _input_digest(values: Mapping[str, NDArray[Any]]) -> str:
    return digest(
        {
            key: {
                "dtype": str(np.asarray(value).dtype),
                "shape": list(np.asarray(value).shape),
                "values": np.asarray(value).tolist(),
            }
            for key, value in sorted(values.items())
        }
    )


def _row(values: Mapping[str, NDArray[Any]], index: int) -> dict[str, NDArray[Any]]:
    return {key: np.asarray(value[index]).copy() for key, value in values.items()}


def _guard_rows(
    path: BranchResult,
    opening_camps: Sequence[int],
    founded_turns: Mapping[int, int],
) -> list[CampReward]:
    step_by_camp = {int(camp): 0 for camp in opening_camps}
    result: list[CampReward] = []
    for row in path.anchor_branch_rows:
        camp = int(row["camp"])
        if camp not in step_by_camp:
            continue
        founded = founded_turns.get(camp)
        if founded is None:
            raise ValueError(f"guard target lacks founding turn for opening camp {camp}")
        young = (
            int(row["parent_society"]) > 0
            and int(row["turn"]) + 1 - int(founded) <= ppo.YOUNG_DAUGHTER_TURNS
        )
        cost = float(
            not bool(row["terminated"])
            and int(row["population"]) < ppo.SMALL_SETTLEMENT
            and not young
        )
        result.append(
            CampReward(
                camp_id=camp,
                step_index=step_by_camp[camp],
                reward=cost,
                years=float(row["years"]),
                terminated=bool(row["terminated"]),
            )
        )
        step_by_camp[camp] += 1
    return result


def _camp_path_return(
    capture: FrozenStateCapture,
    path: BranchResult,
    head: str,
    camp: int,
    runtime: EndpointRuntime,
) -> tuple[float, RandomizedTailEstimate, float, bool, float]:
    if head == GUARD_COST:
        rows = [
            row
            for row in _guard_rows(path, capture.opening_camps, capture.founded_turns)
            if row.camp_id == camp
        ]
        prefix_rows = [row for row in rows if row.step_index < FROZEN_PROTOCOL.prefix_terms]
        prefix = aggregate_camp_prefix(prefix_rows, [camp], FROZEN_PROTOCOL.prefix_terms)
        if prefix.terminal:
            tail = _zero_tail(path.schedule)
        elif rows[-1].terminated:
            tail = _exact_terminal_tail(
                rows, [camp], start=FROZEN_PROTOCOL.prefix_terms, schedule=path.schedule
            )
        elif path.schedule.exceeds_hard_cap:
            tail = RandomizedTailEstimate("unresolved-hard-cap", path.schedule.schedule_id)
        else:
            terms = _discounted_terms(
                rows,
                [camp],
                start=FROZEN_PROTOCOL.prefix_terms,
                required=path.schedule.required_terms,
            )
            tail = estimate_randomized_tail(
                terms,
                path.schedule,
                _tail_assumptions(runtime, cast(Any, GUARD_COST), 1, path.schedule),
            )
        first = rows[0]
        return prefix.prefix_return, tail, first.reward, first.terminated, first.years
    objective = cast(Any, head)
    if not any(int(row["camp"]) == camp for row in path.anchor_branch_rows):
        raise ValueError(f"branch omitted opening camp {camp}")
    rewards = path.camp_prefix_returns.get(objective)
    tails = path.camp_tail_estimates.get(objective)
    if rewards is None or tails is None or camp not in rewards or camp not in tails:
        raise ValueError(f"branch lacks {head} target for camp {camp}")
    one_step = (path.anchor_one_step_rewards or {}).get(objective, {}).get(camp)
    if one_step is None:
        raise ValueError(f"branch lacks one {head} immediate reward for camp {camp}")
    first = one_step
    return float(rewards[camp]), tails[camp], first.reward, first.terminated, first.years


def _tail_exact(tail: RandomizedTailEstimate) -> bool:
    return bool(
        tail.identified
        and tail.standard_error_scope is not None
        and tail.standard_error_scope.startswith("exact")
    )


def _draw_vector(tail: RandomizedTailEstimate, draw_count: int) -> NDArray[np.float64]:
    if tail.identified:
        values = np.asarray(tail.draw_estimates, dtype=np.float64)
        if values.shape != (draw_count,) or not np.isfinite(values).all():
            raise ValueError("identified tail does not retain every frozen stopping draw")
        return values
    return np.zeros(draw_count, dtype=np.float64)


def _stack_rows(rows: list[Mapping[str, NDArray[Any]]]) -> dict[str, NDArray[Any]]:
    if not rows:
        raise ValueError("native anchor collection produced no model-input rows")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("collected model-input schemas differ across rows")
    return {key: np.stack([np.asarray(row[key]) for row in rows]) for key in sorted(keys)}


def build_native_source_arrays(
    captures: Sequence[FrozenStateCapture],
    runtime: EndpointRuntime,
    source_metadata: Mapping[str, Any],
) -> dict[str, NDArray[Any]]:
    """Convert exact in-memory fork captures to the validated source-NPZ schema."""

    if not captures:
        raise ValueError("native source requires at least one frozen state")
    worlds = {capture.world_seed for capture in captures}
    if worlds != set(source_metadata.get("worldSeeds", [])):
        raise ValueError("captured worlds differ from frozen source metadata")
    parent_rows: list[dict[str, NDArray[Any]]] = []
    row_ids: list[str] = []
    unit_ids: list[str] = []
    survived: list[bool] = []
    prefixes = {head: [] for head in VALUE_HEADS}
    statuses = {head: [] for head in VALUE_HEADS}
    exact = {head: [] for head in VALUE_HEADS}
    draws = {head: [] for head in VALUE_HEADS}
    schedule_ids: list[list[str]] = []

    rank_inputs: list[dict[str, NDArray[Any]]] = []
    rank_groups: list[str] = []
    rank_parents: list[str] = []
    rank_successors: list[str] = []
    rank_units: list[str] = []
    rank_discount: list[float] = []
    rank_terminal: list[bool] = []
    rank_immediate = {head: [] for head in VALUE_HEADS}
    rank_realized = {head: [] for head in VALUE_HEADS}
    rank_error = {head: [] for head in VALUE_HEADS}
    rank_mask = {head: [] for head in VALUE_HEADS}

    for capture in captures:
        baseline = sorted(
            (path for path in capture.paths if path.alternative_id == "sampled-policy"),
            key=lambda path: path.path_replicate,
        )
        if len(baseline) != FROZEN_PROTOCOL.path_replicates:
            raise ValueError("frozen state lacks four baseline sampled-policy paths")
        by_pair = {(path.path_replicate, path.alternative_id): path for path in capture.paths}
        if len(by_pair) != FROZEN_PROTOCOL.path_replicates * len(FROZEN_PROTOCOL.alternatives):
            raise ValueError("frozen state lacks its complete paired path set")
        unit = f"world-{capture.world_seed}"
        for row_index, camp in enumerate(capture.opening_camps):
            parent_rows.append(_row(capture.parent_inputs, row_index))
            row_ids.append(
                f"{unit}/{capture.regime_id}/decision-{capture.decision_step}/camp-{camp}"
            )
            unit_ids.append(unit)
            schedule_ids.append([path.schedule.schedule_id for path in baseline])
            camp_prefix_terminal = []
            for head in VALUE_HEADS:
                head_prefixes: list[float] = []
                head_statuses: list[str] = []
                head_exact: list[bool] = []
                head_draws: list[NDArray[np.float64]] = []
                for path in baseline:
                    prefix, tail, _, _, _ = _camp_path_return(capture, path, head, camp, runtime)
                    head_prefixes.append(prefix)
                    head_statuses.append(tail.status)
                    head_exact.append(_tail_exact(tail))
                    head_draws.append(_draw_vector(tail, FROZEN_PROTOCOL.tail_draws))
                prefixes[head].append(head_prefixes)
                statuses[head].append(head_statuses)
                exact[head].append(head_exact)
                draws[head].append(head_draws)
                if head == "welfare":
                    camp_prefix_terminal = head_exact
            survived.append(not all(camp_prefix_terminal))

            for path_replicate in range(FROZEN_PROTOCOL.path_replicates):
                for alternative in FROZEN_PROTOCOL.alternatives:
                    path = by_pair[(path_replicate, alternative)]
                    successor = (path.anchor_successor_inputs or {}).get(camp)
                    terminal_first = None
                    first_years = None
                    values: dict[str, tuple[float, RandomizedTailEstimate, float, bool, float]] = {}
                    for head in VALUE_HEADS:
                        values[head] = _camp_path_return(capture, path, head, camp, runtime)
                        terminal_first = (
                            values[head][3] if terminal_first is None else terminal_first
                        )
                        first_years = values[head][4] if first_years is None else first_years
                        if values[head][3] != terminal_first or values[head][4] != first_years:
                            raise ValueError("objective rows disagree on transition identity")
                    rank_row = (
                        _row(capture.parent_inputs, row_index)
                        if successor is None
                        else {key: np.asarray(value).copy() for key, value in successor.items()}
                    )
                    rank_inputs.append(rank_row)
                    rank_groups.append(
                        f"{unit}/{capture.regime_id}/decision-{capture.decision_step}/"
                        f"path-{path_replicate}/camp-{camp}"
                    )
                    rank_parents.append(capture.parent_digest)
                    rank_successors.append(
                        digest(
                            {
                                "successorWorld": path.anchor_successor_world_digest,
                                "camp": camp,
                                "modelInput": _input_digest(rank_row),
                            }
                        )
                    )
                    rank_units.append(unit)
                    assert terminal_first is not None and first_years is not None
                    rank_terminal.append(bool(terminal_first))
                    rank_discount.append(exp(-float(first_years) / contracts.HORIZON_YEARS))
                    for head, (prefix, tail, immediate, _, _) in values.items():
                        available = tail.identified
                        rank_immediate[head].append(immediate)
                        rank_realized[head].append(
                            prefix + cast(float, tail.estimate) if available else 0.0
                        )
                        rank_error[head].append(
                            cast(float, tail.standard_error) if available else 0.0
                        )
                        rank_mask[head].append(available)

    arrays: dict[str, NDArray[Any]] = {
        "metadata_json": np.asarray(
            json.dumps(dict(source_metadata), sort_keys=True, separators=(",", ":"))
        ),
        "row_id": np.asarray(row_ids),
        "independent_unit_id": np.asarray(unit_ids),
        "survived_fixed_prefix": np.asarray(survived, dtype=bool),
        "path_schedule_id": np.asarray(schedule_ids),
        **_stack_rows(parent_rows),
    }
    keep_rank: list[int] = []
    for group in sorted(set(rank_groups)):
        members = [index for index, value in enumerate(rank_groups) if value == group]
        if len({rank_successors[index] for index in members}) > 1:
            keep_rank.extend(members)
    if keep_rank:
        arrays.update(
            {
                "rank_group_id": np.asarray([rank_groups[index] for index in keep_rank]),
                "rank_parent_digest": np.asarray([rank_parents[index] for index in keep_rank]),
                "rank_successor_digest": np.asarray(
                    [rank_successors[index] for index in keep_rank]
                ),
                "rank_independent_unit_id": np.asarray([rank_units[index] for index in keep_rank]),
                "rank_discount": np.asarray(
                    [rank_discount[index] for index in keep_rank], dtype=np.float64
                ),
                "rank_terminal": np.asarray(
                    [rank_terminal[index] for index in keep_rank], dtype=bool
                ),
                **{
                    f"rank_input__{key}": value
                    for key, value in _stack_rows(
                        [rank_inputs[index] for index in keep_rank]
                    ).items()
                },
            }
        )
    for head in VALUE_HEADS:
        arrays[f"path_prefix__{head}"] = np.asarray(prefixes[head], dtype=np.float64)
        arrays[f"path_tail_status__{head}"] = np.asarray(statuses[head])
        arrays[f"path_tail_exact_terminal__{head}"] = np.asarray(exact[head], dtype=bool)
        arrays[f"path_tail_draws__{head}"] = np.asarray(draws[head], dtype=np.float64)
        if keep_rank:
            arrays[f"rank_immediate__{head}"] = np.asarray(
                [rank_immediate[head][index] for index in keep_rank], dtype=np.float64
            )
            arrays[f"rank_realized__{head}"] = np.asarray(
                [rank_realized[head][index] for index in keep_rank], dtype=np.float64
            )
            arrays[f"rank_standard_error__{head}"] = np.asarray(
                [rank_error[head][index] for index in keep_rank], dtype=np.float64
            )
            arrays[f"rank_mask__{head}"] = np.asarray(
                [rank_mask[head][index] for index in keep_rank], dtype=bool
            )
    return arrays


def _tail_bounds(runtime: EndpointRuntime) -> dict[str, Json]:
    result: dict[str, Json] = {}
    for head in VALUE_HEADS:
        if head == "trade_utility":
            continue
        bound = raw_camp_reward_absolute_bound(cast(Any, head), 1)
        if runtime.reward_semantics == "history-conditioned-shaped":
            if head == "welfare":
                bound = shaped_welfare_reward_absolute_bound(
                    bound,
                    raw_camp_reward_absolute_bound("welfare", 1),
                    mix_change_cost=runtime.mix_change_cost,
                    break_cost=runtime.break_cost,
                    commitments_active=runtime.stage.commitments,
                )
            elif head != GUARD_COST and runtime.personality_spec != "default":
                bound = shaped_nonnegative_component_absolute_bound(
                    bound,
                    mix_change_cost=runtime.mix_change_cost,
                    break_cost=runtime.break_cost,
                    commitments_active=runtime.stage.commitments,
                )
        result[head] = {
            "perTurnAbsoluteBound": bound,
            "provenanceId": f"signed-int32-population+qol-unit-bound/{head}",
        }
    return result


def _strict_current_runtime(
    checkpoint: Path, refresh: Mapping[str, Any]
) -> tuple[EndpointRuntime, Json]:
    if checkpoint_hash(checkpoint) != refresh.get("frozenCheckpointSha256"):
        raise ProvenanceError("refresh request checkpoint hash differs from executing bytes")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ProvenanceError("current-policy checkpoint must contain a state object")
    required = {
        "actor",
        "critic",
        "cost_critic",
        "normalizer",
        "core_width",
        "privileged_width",
        "experts",
        "token",
        "vector_critic",
        "region_head",
        "stage",
        "commitments",
        "region_movement",
        "observe_movement",
        "learned_fission",
        "learned_trade",
        "hyper",
        "feature_contract",
        "learner_contract",
        "run_manifest",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ProvenanceError(f"modern checkpoint lacks exact runtime fields: {missing}")
    if (
        state["learner_contract"] != LEARNER_CONTRACT
        or state["feature_contract"] != BOUNDED_FEATURE_CONTRACT
        or not bool(state["vector_critic"])
    ):
        raise ProvenanceError("refresh requires a bounded observed-anchor vector checkpoint")
    core = int(state["core_width"])
    privileged = int(state["privileged_width"])
    actor = Actor(
        core,
        experts=bool(state["experts"]),
        token=bool(state["token"]),
        region_head=str(state["region_head"]),
    )
    actor.load_state_dict(state["actor"], strict=True)
    critic = VectorCritic(core, privileged_width=privileged, token=bool(state["token"]))
    critic.load_state_dict(state["critic"], strict=True)
    cost_critic = ppo.CostCritic(core, token=bool(state["token"]))
    cost_critic.load_state_dict(state["cost_critic"], strict=True)
    actor.eval()
    critic.eval()
    cost_critic.eval()
    normalizer = load_checkpoint_normalizer(
        state, expected_feature_contract=BOUNDED_FEATURE_CONTRACT
    )
    stage_name = str(state["stage"])
    if stage_name not in ppo.STAGES:
        raise ProvenanceError("checkpoint stage is unknown")
    stage = dataclasses.replace(
        ppo.STAGES[stage_name],
        commitments=bool(state["commitments"]),
        region_movement=bool(state["region_movement"]),
        observe_movement=bool(state["observe_movement"]),
        fission=bool(state["learned_fission"]),
        trade=bool(state["learned_trade"]),
    )
    hyper = state["hyper"]
    if not isinstance(hyper, dict):
        raise ProvenanceError("checkpoint hyperparameters are not explicit")
    policy_hyper = SimpleNamespace(
        log_std_floor=float(hyper["log_std_floor"]),
        personality=str(hyper["personality"]),
        personality_drift=float(hyper["personality_drift"]),
        personality_scales=str(state["personality_scales"]),
        personality_clip=float(state["personality_clip"]),
    )
    configuration = ppo.behavior_policy_configuration(stage, policy_hyper, actor)
    if behavior_policy_digest(actor, configuration) != refresh.get("sourcePolicyDigest"):
        raise ProvenanceError("refresh request differs from the effective sampled policy")
    normalizer_state = cast(Json, state["normalizer"])
    binding = normalization_binding(
        normalizer_state, cast(Json, normalizer_state["normalization_contract"])
    )
    if binding != refresh.get("normalizationBinding"):
        raise ProvenanceError("refresh request differs from checkpoint input normalization")
    options = cast(Json, state["run_manifest"]).get("options")
    if not isinstance(options, dict) or isinstance(options.get("seed"), bool):
        raise ProvenanceError("checkpoint run manifest lacks its optimizer seed")
    optimizer_seed = int(options["seed"])
    receipt = EndpointReceipt(
        node_id=f"current-{checkpoint_hash(checkpoint)[:16]}",
        checkpoint=checkpoint.resolve(),
        checkpoint_sha256=checkpoint_hash(checkpoint),
        artifact_metadata={},
        resolved_semantics={},
        loading={},
        critic_typed_content_sha256=policy_digest(critic),
        arm_id=None,
        optimizer_seed=optimizer_seed,
    )
    runtime = EndpointRuntime(
        receipt=receipt,
        actor=actor,
        critic=critic,
        normalizer=normalizer,
        stage=stage,
        vector_critic=True,
        objectives=cast(Any, tuple(contracts.OBJECTIVES)),
        personality_spec=policy_hyper.personality,
        personality_drift=policy_hyper.personality_drift,
        log_std_floor=policy_hyper.log_std_floor,
        mix_change_cost=float(hyper["mix_change_cost"]),
        break_cost=float(hyper["break_cost"]),
        reward_semantics="history-conditioned-shaped",
        return_contract_id="observed-anchor-v1-native-sampled-policy",
    )
    return runtime, {
        "state": state,
        "costCritic": cost_critic,
        "configuration": configuration,
        "binding": binding,
        "optimizerSeed": optimizer_seed,
    }


def collect_source(
    checkpoint: Path,
    refresh_request: Path,
    source_output: Path,
    anchor_output: Path,
    role: str,
    world_seeds: Sequence[int],
) -> dict[str, Any]:
    """Run the bounded current-policy native collection and emit source+anchor NPZs."""

    if source_output.exists() or anchor_output.exists():
        raise FileExistsError("source and anchor outputs must both be new")
    refresh = json.loads(refresh_request.read_text(encoding="utf-8"))
    if refresh.get("kind") != "observed-anchor-refresh-request-v1":
        raise ProvenanceError("input is not an observed-anchor refresh request")
    checked_worlds = tuple(seed_list(",".join(str(value) for value in world_seeds)))
    if not checked_worlds or len(checked_worlds) != len(set(checked_worlds)):
        raise ProvenanceError("world seeds must be explicit, nonempty, and unique")
    torch.set_num_threads(4)
    runtime, loaded = _strict_current_runtime(checkpoint, refresh)
    captures: list[FrozenStateCapture] = []
    resets: list[Json] = []
    simulator: Json | None = None
    continuation_by_world: dict[int, set[int]] = {world: set() for world in checked_worlds}
    parent_by_world: dict[int, list[str]] = {world: [] for world in checked_worlds}
    branch_by_world: dict[int, list[str]] = {world: [] for world in checked_worlds}
    schedules: dict[str, Json] = {}

    for world_seed in checked_worlds:
        for regime_id, withheld in FROZEN_PROTOCOL.regimes:
            bridge = Model9Bridge()
            try:
                bridge_hash = getattr(bridge.schema, "bridge_sha256", None)
                manifest_hash = cast(Json, loaded["state"])["run_manifest"].get("bridge_sha256")
                if bridge_hash != manifest_hash or checkpoint_hash(DEFAULT_BRIDGE) != bridge_hash:
                    raise ProvenanceError("live bridge differs from the frozen training runtime")
                current = bridge.reset(
                    world_seed,
                    movement=runtime.stage.bridge_movement,
                    withheld=withheld,
                )
                if simulator is None:
                    simulator = {
                        "bridgeSha256": bridge_hash,
                        "simulatorCommit": getattr(bridge.schema, "simulator_commit", None),
                    }
                resets.append(
                    {
                        "worldSeed": world_seed,
                        "regime": regime_id,
                        "postResetTurn": int(current["turn"]),
                        "postResetFingerprint": str(current["fingerprint"]),
                        "resetIdentity": copy.deepcopy(bridge.last_reset_identity),
                    }
                )
                main_action_seed = _derived_seed(
                    "gate34-main-action-v1", runtime.receipt.node_id, world_seed, regime_id
                )
                main_personality_seed = _derived_seed(
                    "gate34-main-personality-v1", runtime.receipt.node_id, world_seed, regime_id
                )
                generator = torch.Generator(device="cpu").manual_seed(main_action_seed)
                personalities = PersonalityState(
                    runtime.personality_spec,
                    runtime.personality_drift,
                    main_personality_seed,
                )
                previous_mixes: dict[int, tuple[float, ...]] = {}
                history: list[Json] = []
                founded = {
                    int(village["camp"]): int(current["turn"])
                    for village in _live_villages(current)
                }
                decision_count = 0
                while decision_count <= max(FROZEN_PROTOCOL.decision_steps):
                    if decision_count in FROZEN_PROTOCOL.decision_steps:
                        parent_digest = digest(
                            {
                                "world": world_seed,
                                "regime": regime_id,
                                "decisionStep": decision_count,
                                "turn": int(current["turn"]),
                                "fingerprint": str(current["fingerprint"]),
                            }
                        )

                        def capture(
                            camps: Sequence[int],
                            inputs: Mapping[str, NDArray[Any]],
                            paths: Sequence[BranchResult],
                            *,
                            _turn: int = int(current["turn"]),
                            _decision: int = decision_count,
                            _parent: str = parent_digest,
                            _world: int = world_seed,
                            _regime: str = regime_id,
                            _founded: Mapping[int, int] = dict(founded),
                        ) -> None:
                            item = FrozenStateCapture(
                                _world,
                                int(loaded["optimizerSeed"]),
                                _regime,
                                _decision,
                                _turn,
                                _parent,
                                tuple(int(camp) for camp in camps),
                                _founded,
                                inputs,
                                tuple(paths),
                            )
                            captures.append(item)
                            parent_by_world[_world].append(_parent)
                            for path in paths:
                                branch = f"{_parent}/{path.path_replicate}/{path.alternative_id}"
                                branch_by_world[_world].append(branch)
                                assert path.schedule.rng_seed is not None
                                continuation_by_world[_world].add(path.schedule.rng_seed)
                                schedules.setdefault(
                                    path.schedule.schedule_id,
                                    {
                                        "scheduleId": path.schedule.schedule_id,
                                        "survivalProbability": path.schedule.survival_probability,
                                        "stopIndices": list(path.schedule.stop_indices),
                                        "hardCapTerms": path.schedule.hard_cap_terms,
                                        "rngSeed": path.schedule.rng_seed,
                                        "generator": path.schedule.generator,
                                        "frozenBeforeCollection": True,
                                    },
                                )

                        _collect_frozen_state(
                            bridge=bridge,
                            current=current,
                            runtime=runtime,
                            protocol=FROZEN_PROTOCOL,
                            world_seed=world_seed,
                            regime_id=regime_id,
                            decision_step=decision_count,
                            history=history,
                            main_personalities=personalities,
                            previous_mixes=previous_mixes,
                            capture_economic=False,
                            anchor_capture=capture,
                        )
                    if decision_count == max(FROZEN_PROTOCOL.decision_steps):
                        break
                    sampled = sample_training_actions(runtime, current, generator, personalities)
                    if not sampled.actions:
                        raise ValueError("main path terminated before a prescribed frozen state")
                    after = bridge.step(sampled.actions)
                    for village in _live_villages(after):
                        founded.setdefault(int(village["camp"]), int(after["turn"]))
                    history.extend(
                        _outcome_rows(
                            world_seed=world_seed,
                            before=current,
                            after=after,
                            sampled=sampled,
                            previous_mixes=previous_mixes,
                            runtime=runtime,
                        )
                    )
                    previous_mixes.update(sampled.mixes)
                    current = after
                    decision_count += 1
            finally:
                bridge.close()
    assert simulator is not None
    all_continuation = sorted(set().union(*continuation_by_world.values()))
    source_meta: Json = {
        "schemaVersion": 1,
        "kind": ESTIMATOR_SOURCE_KIND,
        "nativeContract": contracts.NATIVE_CONTRACT,
        "rewardSemantics": REWARD_SEMANTICS,
        "terminationSemantics": TERMINATION_SEMANTICS,
        "discountSemantics": "exp(-years/horizonYears)",
        "turnYears": contracts.TURN_YEARS,
        "horizonYears": contracts.HORIZON_YEARS,
        "policyDigestSemantics": "effective-sampled-policy-digest-v1",
        "tailSemantics": "fixed-prefix-plus-geometric-russian-roulette",
        "truncated": False,
        "criticBootstrapUsed": False,
        "continuationPane": "sampled-training-policy",
        "discountedTailTerms": "discounted-from-original-frozen-state",
        "sourceCheckpointSha256": checkpoint_hash(checkpoint),
        "sourceActorTensorDigest": policy_digest(runtime.actor),
        "sourcePolicyConfiguration": loaded["configuration"],
        "sourcePolicyDigest": behavior_policy_digest(runtime.actor, loaded["configuration"]),
        "normalizationBinding": loaded["binding"],
        "simulator": simulator,
        "worldSeeds": list(checked_worlds),
        "optimizerSeeds": [int(loaded["optimizerSeed"])],
        "continuationSeeds": all_continuation,
        "independentUnits": [
            {
                "unitId": f"world-{world}",
                "worldSeed": world,
                "optimizerSeed": int(loaded["optimizerSeed"]),
                "continuationSeeds": sorted(continuation_by_world[world]),
                "parentIdentities": sorted(parent_by_world[world]),
                "branchIdentities": sorted(branch_by_world[world]),
            }
            for world in checked_worlds
        ],
        "schedules": [schedules[key] for key in sorted(schedules)],
        "tailAbsoluteBounds": _tail_bounds(runtime),
        "rankWitnessSemantics": RANK_SEMANTICS,
        "resetReceipts": resets,
        "scope": "training-debug-only-no-local-readiness-claim",
    }
    arrays = build_native_source_arrays(captures, runtime, source_meta)
    source_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(source_output, **arrays)
    built = build_randomized_estimator_dataset(source_output, anchor_output, role)
    return {
        "sourcePath": str(source_output.resolve()),
        "sourceSha256": checkpoint_hash(source_output),
        "anchor": built,
        "worlds": list(checked_worlds),
        "independentUnit": "world",
        "readinessClaimed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect critic-independent native continuation anchors"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--refresh-request", type=Path, required=True)
    parser.add_argument("--source-out", type=Path, required=True)
    parser.add_argument("--anchor-out", type=Path, required=True)
    parser.add_argument("--role", choices=("train-anchor", "actor-gate"), required=True)
    parser.add_argument("--world-seeds", required=True)
    args = parser.parse_args(argv)
    result = collect_source(
        args.checkpoint,
        args.refresh_request,
        args.source_out,
        args.anchor_out,
        args.role,
        seed_list(args.world_seeds),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
