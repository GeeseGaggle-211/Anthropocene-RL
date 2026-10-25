"""G4 learner-state recovery (DAgger) collection for the labor actor (brief 6.1, 6.4 S2).

For each fitting world, one teacher trajectory (the h12 path: the converted
native heuristic recommendation submitted through the action-12 runtime, exactly
as ``labor_bc.collect_heuristic_world`` drives it) runs from the reset. At each
keyed start decision the labor bridge is forked and the frozen actor controls
every eligible camp for ``depth`` decisions. Before each actor decision executes,
the complete teacher is queried on that exact state with the collector's
capture/reset shadow step and label conversion, and the teacher's full action
map is validated (purely) on that state; a rejected teacher recommendation is
quarantined exactly as in the collector. Each row stores the actor-visited
inputs, the teacher's full-action label with per-head validity, the collector's
identity fields plus ``row_source`` (1 = dagger), ``row_round``, ``row_depth``,
``row_start`` (start decision index), ``row_start_index`` and
``row_branch_step``. After the branch the fork is restored (with the lineage
personality table) and the teacher continues to the next start, so every
branch is the state an independent teacher-prefix run would reach.

Rewards are the learner branch's transition rewards in the collector's fields,
truncated at the branch end. The shards are therefore ``actor_only``:
``labor_training pretrain-critic`` refuses them; ``pretrain-bc`` aggregates
them with the teacher data (``--dataset A --dataset B ...``).

V^rho targets at learner-visited states (amendment r5-b, item 6). With
``--teacher-continuation branch-end`` (or ``all-visited``) the collector also
forks every learner-visited state -- the state after the branch's last actor
decision, or after each actor decision -- runs one teacher (H12) continuation
from it for every live camp and closes it with the preregistered randomized
tail (``labor_bc.RandomizedTailSettings``, the r5 method of
``labor_teacher_data --return-tail randomized``), then restores the fork, so
the actor rows are unchanged. Each continuation is written in the teacher
collector's row format with the same typed ``tail_*`` fields, plus the branch
identity, ``row_source`` 3 and a per-continuation ``row_trajectory``, into a
separate dataset (``--continuation-out``) of source kind
``learner_state_teacher_continuation``. ``pretrain-critic`` accepts it (alone
or with teacher shards) only with ``--learner-state-continuations``; actor BC
refuses it. Its observation and privileged arrays are those of the
learner-visited state.

Shards and the manifest follow ``labor_teacher_data`` (immutable shards, meta
last, dataset.json only after every shard). ``--resume`` continues an
interrupted output with the same source record: verified shards are kept,
unpublished shard directories are discarded and recollected.

Example::

    python -m anthropocene_rl.autonomous_v7.labor_dagger collect \\
        --checkpoint pretrained.pt --worlds 90100-90159 --round 1 --depth 1 \\
        --starts 3 --horizon 25 --players 16 --width 128 --height 64 \\
        --separation 9 --jobs 30 --out dagger/round-1/data
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import labor_bc
from .labor_bc import (
    RETURN_TAIL_RANDOMIZED,
    TRAJECTORY_FIELD,
    HeuristicLaborLabel,
    LaborResetOptions,
    RandomizedTailSettings,
    WorldCollection,
    _concat,
    _idle_retry,
    _label_tensors,
    _one_row,
    _retained_action,
    shadow_teacher_labels,
    validate_fitting_worlds,
    world_balanced_weights,
)
from .labor_ppo import execute_labor_decision
from .labor_privileged import privileged_row_tensors
from .labor_teacher_data import (
    publish_dataset,
    source_record,
    verify_dataset,
    verify_shard,
    write_shard,
)
from .labor_worlds import require_training_worlds
from .protocol import seed_list
from .provenance import ProvenanceError, checkpoint_hash, code_commit

DAGGER_SOURCE_KIND = "learner_state_dagger"
DAGGER_ROW_SOURCE = 1
DAGGER_CONTRACT = "model9-labor-dagger-learner-state-v1"
DEFAULT_START_KEY = 4243
CONTINUATION_SOURCE_KIND = labor_bc.CONTINUATION_SOURCE_KIND
CONTINUATION_ROW_SOURCE = labor_bc.CONTINUATION_ROW_SOURCE
CONTINUATION_CONTRACT = "model9-labor-dagger-teacher-continuation-v1"
CONTINUATION_STATES = ("off", "branch-end", "all-visited")


def draw_start_turns(
    key: int, world: int, round_index: int, count: int, horizon: int, depth: int
) -> list[int]:
    """Keyed start decisions, one per equal stratum of [0, horizon - depth].

    Start index k is drawn from its own stream keyed by (key, world, round, k),
    so adding starts never changes earlier ones' strata draws within a stratum
    layout, and the draw is independent of process order.
    """
    if count < 1 or depth < 1 or horizon < 1:
        raise ValueError("starts, depth and horizon must be positive")
    span = max(1, horizon - depth + 1)
    starts = []
    for index in range(count):
        low = (index * span) // count
        high = max(low + 1, ((index + 1) * span) // count)
        rng = np.random.default_rng([int(key), int(world), int(round_index), int(index)])
        starts.append(int(rng.integers(low, high)))
    return starts


def _stream_seed(*parts: int) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def _live(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        v for v in step["villages"] if v.get("observation") is not None and not v.get("terminated")
    ]


def _teacher_step(
    labor_bridge: Any,
    heur_bridge: Any,
    world: int,
    options: LaborResetOptions,
    live: Sequence[Mapping[str, Any]],
    contexts: Sequence[Any],
    tensors: Mapping[str, torch.Tensor],
) -> Mapping[str, Any]:
    """One teacher (h12) decision for every live camp; returns the successor step."""
    catalogs = {context[0].camp_id: context[0] for context in contexts}
    _, labels = shadow_teacher_labels(labor_bridge, heur_bridge, world, options, live, contexts)
    retained = {
        int(village["camp"]): _retained_action(
            label, context[0], tensors["movement_mask"][row : row + 1]
        )
        for row, (village, context, label) in enumerate(zip(live, contexts, labels, strict=True))
    }
    return execute_labor_decision(labor_bridge, catalogs, retained, _idle_retry).step


def teacher_continuation(
    *,
    labor_bridge: Any,
    heur_bridge: Any,
    step: Mapping[str, Any],
    world: int,
    options: LaborResetOptions,
    normalizer: Any,
    personality: Any,
    founded: Mapping[int, int],
    tail: RandomizedTailSettings,
    identity: Mapping[str, int],
) -> WorldCollection | None:
    """One teacher continuation from ``step`` with randomized-tail rows; the bridge is restored.

    Rows are the teacher collector's (labels, rewards and typed tail of the
    first teacher decision from this state, for every live camp), tagged with
    ``identity`` (branch fields and ``row_trajectory``) and ``row_source`` 3.
    ``personality`` is the world resolver; its lineage table is restored too.
    """
    if not _live(step):
        return None
    started = time.perf_counter()
    turn = int(step["turn"])
    fork, _ = labor_bridge.fork()
    saved_table = copy.deepcopy(personality.table) if hasattr(personality, "table") else None
    try:
        temp = WorldCollection(world_seed=int(world))
        transition = labor_bc._teacher_transition(
            labor_bridge=labor_bridge,
            heur_bridge=heur_bridge,
            step=step,
            world_seed=int(world),
            options=options,
            normalizer=normalizer,
            personality=personality,
            founded=founded,
        )
        if transition is None:
            return None
        temp.native_resolves += 2
        temp.decisions += 1
        labor_bc.record_teacher_transition(temp, transition, int(world), founded)
        entries = labor_bc._collect_teacher_tail(
            result=temp,
            tail=tail,
            labor_bridge=labor_bridge,
            heur_bridge=heur_bridge,
            step=transition.next_step,
            options=options,
            normalizer=normalizer,
            personality=personality,
            founded=founded,
            window_turns=1,
        )
        labor_bc._attach_tail_rows(temp.rows, entries)
    finally:
        restored = labor_bridge.restore(fork)
        labor_bridge.drop(fork)
        if saved_table is not None:
            personality.table = saved_table
    if int(restored["turn"]) != turn:
        raise ValueError("labor bridge restore after a teacher continuation changed the turn")
    for row in temp.rows:
        row["row_source"] = torch.tensor([CONTINUATION_ROW_SOURCE], dtype=torch.long)
        for key, value in identity.items():
            row[key] = torch.tensor([int(value)], dtype=torch.long)
    temp.window_seconds = time.perf_counter() - started
    return temp


def collect_learner_world(
    *,
    actor: Any,
    normalizer: Any,
    personality: Any,
    world_seed: int,
    round_index: int,
    depth: int,
    starts: Sequence[int],
    reset_options: LaborResetOptions | None = None,
    stochastic: bool = False,
    policy_seed: int = 0,
    labor_bridge_factory: Any = None,
    heuristic_bridge_factory: Any = None,
    progress: Any | None = None,
    continuation: RandomizedTailSettings | None = None,
    continuation_states: str = "branch-end",
    continuations: list[WorldCollection] | None = None,
) -> WorldCollection:
    """Teacher prefix, forked actor branches of ``depth`` decisions, teacher labels on every branch state.

    ``personality`` is the checkpoint's declared personality (fixed or a
    distribution assignment); it conditions the actor and is recorded in the
    row inputs exactly as the collector does. The teacher label itself is
    personality-agnostic.

    With ``continuation`` every learner-visited state (``branch-end``: after
    the branch's last actor decision; ``all-visited``: after each) also gets
    one teacher continuation (:func:`teacher_continuation`), appended to
    ``continuations``; the actor rows are unchanged by it.
    """
    from .bridge import Model9Bridge
    from .labor_bridge import LaborModel9Bridge
    from .labor_model import labor_policy_tensors
    from .labor_personality import personality_for_world
    from .labor_pretrain_eval import actor_labor_decision
    from .labor_training import _settlement_guard, _transition_rewards

    require_training_worlds([world_seed], "labor dagger collection")
    options = reset_options or LaborResetOptions()
    labor_factory = labor_bridge_factory or LaborModel9Bridge
    heur_factory = heuristic_bridge_factory or (lambda: Model9Bridge(economic_diagnostics=True))
    world = int(world_seed)
    if continuation is not None and (
        continuation_states not in CONTINUATION_STATES[1:] or continuations is None
    ):
        raise ValueError("teacher continuations need a state rule and an output list")
    world_personality = personality_for_world(personality, world)
    schedule = sorted((int(s), index) for index, s in enumerate(starts))
    result = WorldCollection(world_seed=world)
    with labor_factory() as labor_bridge, heur_factory() as heur_bridge:
        step = labor_bridge.reset(world, **options.reset_kwargs())
        identity = getattr(labor_bridge, "last_reset_identity", None)
        result.reset_identity = dict(identity) if isinstance(identity, Mapping) else {}
        founded = {int(row["camp"]): int(step["turn"]) for row in step["villages"]}
        decision = 0
        for start, start_index in schedule:
            # Teacher prefix up to this start decision.
            while decision < start:
                live = _live(step)
                if not live:
                    break
                contexts, tensors = labor_policy_tensors(
                    live, normalizer, personality=world_personality
                )
                step = _teacher_step(labor_bridge, heur_bridge, world, options, live, contexts, tensors)
                for village in step["villages"]:
                    founded.setdefault(int(village["camp"]), int(step["turn"]))
                result.native_resolves += 2
                result.counters["teacher_prefix_decisions"] += 1
                decision += 1
            if decision < start or not _live(step):
                result.counters["starts_after_extinction"] += 1
                break

            fork, _ = labor_bridge.fork()
            fork_turn = int(step["turn"])
            saved_table = (
                copy.deepcopy(world_personality.table) if hasattr(world_personality, "table") else None
            )
            branch_founded = dict(founded)
            generator = torch.Generator().manual_seed(
                _stream_seed(policy_seed, world, round_index, start_index)
            )
            branch = step

            def continue_teacher(
                state: Mapping[str, Any],
                actor_steps: int,
                *,
                branch_founded: Mapping[int, int] = branch_founded,
                start: int = start,
                start_index: int = start_index,
            ) -> None:
                collected = teacher_continuation(
                    labor_bridge=labor_bridge,
                    heur_bridge=heur_bridge,
                    step=state,
                    world=world,
                    options=options,
                    normalizer=normalizer,
                    personality=world_personality,
                    founded=branch_founded,
                    tail=continuation,
                    identity={
                        "row_round": round_index,
                        "row_depth": depth,
                        "row_start": start,
                        "row_start_index": start_index,
                        "row_branch_step": actor_steps,
                        TRAJECTORY_FIELD: start_index * (depth + 1) + actor_steps,
                    },
                )
                if collected is not None:
                    continuations.append(collected)
                    result.counters["teacher_continuations"] += 1

            actor_steps = 0
            try:
                for branch_step in range(depth):
                    live = _live(branch)
                    if not live:
                        break
                    if (
                        continuation is not None
                        and continuation_states == "all-visited"
                        and branch_step > 0
                    ):
                        continue_teacher(branch, branch_step)
                    contexts, tensors = labor_policy_tensors(
                        live, normalizer, personality=world_personality
                    )
                    privileged = privileged_row_tensors(live)
                    catalogs = {context[0].camp_id: context[0] for context in contexts}
                    decision_turn = int(branch["turn"])

                    # Teacher query on this exact learner-visited state.
                    _, labels = shadow_teacher_labels(
                        labor_bridge, heur_bridge, world, options, live, contexts
                    )
                    result.native_resolves += 1
                    teacher_actions = {}
                    for row, (village, context, label) in enumerate(
                        zip(live, contexts, labels, strict=True)
                    ):
                        action = _retained_action(
                            label, context[0], tensors["movement_mask"][row : row + 1]
                        )
                        action.validate(context[0], retry=False)
                        teacher_actions[int(village["camp"])] = dict(action.payload)
                        for reason in label.reasons:
                            result.reasons[reason.split(":", 1)[0]] += 1
                    teacher_attempts = labor_bridge.validate_labor(teacher_actions, attempt=0)

                    execution = actor_labor_decision(
                        labor_bridge,
                        actor,
                        normalizer,
                        world_personality,
                        live,
                        catalogs,
                        tensors,
                        stochastic=stochastic,
                        generator=generator,
                    )
                    next_step = execution.step
                    result.native_resolves += 1
                    result.decisions += 1
                    result.counters["actor_initial_rejected"] += sum(
                        1 for a in execution.initial_attempts.values() if not a.valid
                    )
                    result.counters["actor_retries"] += len(execution.retry)
                    rewards, terminal, elapsed = _transition_rewards(
                        next_step,
                        tuple(catalogs),
                        branch_founded,
                        world_index=0,
                        decision_turn=decision_turn,
                    )
                    outcomes = {int(v["camp"]): v for v in next_step["villages"]}
                    for row, (village, label) in enumerate(zip(live, labels, strict=True)):
                        camp = int(village["camp"])
                        attempt = teacher_attempts[camp]
                        accepted = bool(attempt.valid)
                        _record_row(
                            result,
                            tensors,
                            row,
                            label,
                            accepted=accepted,
                            violations=attempt.violations,
                            world=world,
                            camp=camp,
                            decision_turn=decision_turn,
                            round_index=round_index,
                            depth=depth,
                            start=start,
                            start_index=start_index,
                            branch_step=branch_step,
                            reward=rewards[camp],
                            feasibility=float(attempt.feasibility_cost),
                            guard=float(
                                _settlement_guard(
                                    outcomes[camp],
                                    branch_founded.get(camp, decision_turn),
                                    int(next_step["turn"]),
                                )
                            ),
                            elapsed=float(elapsed[camp]),
                            terminal=bool(terminal[camp]),
                            privileged=privileged,
                        )
                    branch = next_step
                    actor_steps = branch_step + 1
                    if progress is not None:
                        progress(result)
                if continuation is not None and actor_steps > 0:
                    continue_teacher(branch, actor_steps)
            finally:
                step = labor_bridge.restore(fork)
                labor_bridge.drop(fork)
                if saved_table is not None:
                    world_personality.table = saved_table
            if int(step["turn"]) != fork_turn:
                raise ValueError("labor bridge restore returned a different decision turn")
            result.counters["branches"] += 1
        result.surviving_camps = len(_live(step))
    return result


def _record_row(
    result: WorldCollection,
    tensors: Mapping[str, torch.Tensor],
    row: int,
    label: HeuristicLaborLabel,
    *,
    accepted: bool,
    violations: Sequence[Any],
    world: int,
    camp: int,
    decision_turn: int,
    round_index: int,
    depth: int,
    start: int,
    start_index: int,
    branch_step: int,
    reward: Mapping[str, float],
    feasibility: float,
    guard: float,
    elapsed: float,
    terminal: bool,
    privileged: Mapping[str, torch.Tensor] | None = None,
) -> None:
    result.counters["total_decisions"] += 1
    result.counters["initial_accepted" if accepted else "initial_rejected"] += 1
    if not accepted:
        result.counters["quarantined"] += 1
        result.quarantine.append(
            {
                "world": world,
                "camp": camp,
                "turn": decision_turn,
                "attempt": 0,
                "start_index": start_index,
                "violations": [
                    {"reasonCode": v.reason_code, "reason": v.reason} for v in violations
                ],
                "labelReasons": list(label.reasons),
            }
        )
    if label.specialty_valid:
        result.counters["labor_label_valid"] += 1
    if label.full_action_valid:
        result.counters["full_action_label_valid"] += 1
    row_tensors = _one_row(tensors, row)
    if privileged is not None:
        # Critic-only raw arrays of the learner-visited state (same boundary as the observation).
        row_tensors.update(_one_row(privileged, row))
    row_tensors.update(_label_tensors(label))

    def long(value: int) -> torch.Tensor:
        return torch.tensor([int(value)], dtype=torch.long)

    row_tensors.update(
        {
            "row_world": long(world),
            "row_camp": long(camp),
            "row_turn": long(decision_turn),
            "row_attempt": long(0),
            "row_quarantined": torch.tensor([not accepted], dtype=torch.bool),
            "row_source": long(DAGGER_ROW_SOURCE),
            "row_round": long(round_index),
            "row_depth": long(depth),
            "row_start": long(start),
            "row_start_index": long(start_index),
            "row_branch_step": long(branch_step),
            "reward_qol_years": torch.tensor([reward["qol_years"]], dtype=torch.float64),
            "reward_population_years": torch.tensor(
                [reward["population_years"]], dtype=torch.float64
            ),
            "reward_feasibility": torch.tensor([feasibility], dtype=torch.float64),
            "reward_settlement_guard": torch.tensor([guard], dtype=torch.float64),
            "transition_elapsed_years": torch.tensor([elapsed], dtype=torch.float64),
            "transition_terminal": torch.tensor([terminal], dtype=torch.bool),
        }
    )
    result.rows.append(row_tensors)


# ---------------------------------------------------------------------------
# Shards and CLI
# ---------------------------------------------------------------------------


def _load_actor(checkpoint: Path) -> tuple[Any, Any, Any]:
    from .labor_model import labor_checkpoint_personality
    from .labor_training import _models_from_checkpoint

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _, actor, _, _, normalizer = _models_from_checkpoint(state)
    actor.eval()
    return actor, normalizer, labor_checkpoint_personality(state)


def _collect_shard(job: Mapping[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    actor, normalizer, personality = _load_actor(Path(job["checkpoint"]))
    options = LaborResetOptions.from_mapping(job["reset_options"])
    world = int(job["world_seed"])
    started = time.perf_counter()

    def progress(state: WorldCollection) -> None:
        print(
            json.dumps(
                {
                    "world": world,
                    "decisions": state.decisions,
                    "rows": len(state.rows),
                    "branches": int(state.counters.get("branches", 0)),
                    "native_resolves": state.native_resolves,
                    "wall_seconds": round(time.perf_counter() - started, 2),
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    tail = RandomizedTailSettings.from_state(job.get("continuation_tail"))
    continuations: list[WorldCollection] = []
    collection = collect_learner_world(
        actor=actor,
        normalizer=normalizer,
        personality=personality,
        world_seed=world,
        round_index=int(job["round"]),
        depth=int(job["depth"]),
        starts=list(job["starts"]),
        reset_options=options,
        stochastic=job["actor_mode"] == "stochastic",
        policy_seed=int(job["policy_seed"]),
        progress=progress,
        continuation=tail,
        continuation_states=job.get("continuation_states", "branch-end"),
        continuations=continuations,
    )
    seconds = time.perf_counter() - started
    continuation_entry = None
    if tail is not None:
        continuation_entry = _write_continuation_shard(job, collection, continuations, tail)
    if not collection.rows:
        raise ValueError(f"world {world} produced no learner-state rows")
    tensors = _concat(collection.rows, f"labor dagger world {world}")
    keep = ~tensors["row_quarantined"].reshape(-1)
    tensors["row_weight"] = world_balanced_weights(tensors["row_world"], keep)
    arrays = {key: value.detach().cpu().numpy() for key, value in tensors.items()}
    meta = {
        "source_run": job["source_run"],
        "contract": DAGGER_CONTRACT,
        "world_seed": world,
        "round": int(job["round"]),
        "depth": int(job["depth"]),
        "starts": [int(s) for s in job["starts"]],
        "actor_mode": job["actor_mode"],
        "turns": int(job["horizon"]),
        "reset_options": options.state_dict(),
        "reset_identity": collection.reset_identity,
        "normalizer_digest": job["normalizer_digest"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "decisions": int(collection.decisions),
        "native_resolves": int(collection.native_resolves),
        "wall_seconds": float(seconds),
        "counters": dict(collection.counters),
        "label_unavailable_reasons": dict(sorted(collection.reasons.items())),
        "quarantine": collection.quarantine,
        "surviving_camps": int(collection.surviving_camps),
        "return_tail": "learner-branch-truncated-actor-only",
        "partition": "fitting",
    }
    if continuation_entry is not None:
        meta["teacher_continuation"] = {
            "data_digest": continuation_entry["data_digest"],
            "meta_sha256": continuation_entry["meta_sha256"],
            "continuations": continuation_entry["continuations"],
        }
    directory = Path(job["out"]) / f"world-{world}"
    record = write_shard(directory, arrays, meta)
    entry = {
        "path": directory.name,
        "world_seed": world,
        "meta_sha256": checkpoint_hash(directory / "meta.json"),
        "data_digest": record["data_digest"],
        "rows": record["rows"],
        "decisions": record["decisions"],
        "native_resolves": record["native_resolves"],
        "wall_seconds": record["wall_seconds"],
    }
    if continuation_entry is not None:
        entry["continuation"] = continuation_entry
    return entry


def _write_continuation_shard(
    job: Mapping[str, Any],
    collection: WorldCollection,
    continuations: Sequence[WorldCollection],
    tail: RandomizedTailSettings,
) -> dict[str, Any]:
    """The world's teacher-continuation shard (V^rho targets at learner-visited states)."""
    world = int(job["world_seed"])
    rows = [row for c in continuations for row in c.rows]
    if not rows:
        raise ValueError(f"world {world} produced no teacher-continuation rows")
    tensors = _concat(rows, f"labor dagger teacher continuations world {world}")
    keep = ~tensors["row_quarantined"].reshape(-1)
    tensors["row_weight"] = world_balanced_weights(tensors["row_world"], keep)
    arrays = {key: value.detach().cpu().numpy() for key, value in tensors.items()}
    counters: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    method_counts: dict[str, int] = defaultdict(int)
    for c in continuations:
        for key, value in c.counters.items():
            counters[key] += int(value)
        for key, value in c.reasons.items():
            reasons[key] += int(value)
        for name, value in (c.tail_record or {}).get("method_counts", {}).items():
            method_counts[name] += int(value)
    per = [
        {
            "row_start_index": int(c.rows[0]["row_start_index"].reshape(-1)[0]),
            "row_branch_step": int(c.rows[0]["row_branch_step"].reshape(-1)[0]),
            "row_trajectory": int(c.rows[0][TRAJECTORY_FIELD].reshape(-1)[0]),
            "turn": int(c.rows[0]["row_turn"].reshape(-1)[0]),
            "camps": len(c.rows),
            "first_step_native_resolves": int(c.native_resolves),
            "tail_decisions": int(c.tail_decisions),
            "tail_native_resolves": int(c.tail_native_resolves),
            "wall_seconds": float(c.window_seconds),
            "method_counts": dict((c.tail_record or {}).get("method_counts", {})),
        }
        for c in continuations
    ]
    first = continuations[0].tail_record or {}
    tail_record = {
        **tail.state_dict(),
        "prefix_terms": 1,
        "prefix_semantics": (
            "each row is the first teacher decision from a learner-visited state; the tail is "
            "valued at the state after it, from one teacher continuation"
        ),
        "schedule_id": first.get("schedule_id"),
        "schedule_seed": first.get("schedule_seed"),
        "required_terms": first.get("required_terms"),
        "stop_indices": first.get("stop_indices"),
        "exceeds_hard_cap": first.get("exceeds_hard_cap"),
        "critic_bootstrap_used": False,
        "schedule_scope": "one frozen stopping schedule per world, shared by its continuations",
        "continuations": len(continuations),
        "method_counts": dict(sorted(method_counts.items())),
    }
    native_resolves = sum(c.native_resolves + c.tail_native_resolves for c in continuations)
    seconds = sum(c.window_seconds for c in continuations)
    meta = {
        "source_run": job["continuation_source_run"],
        "contract": CONTINUATION_CONTRACT,
        "dagger_source_run": job["source_run"],
        "world_seed": world,
        "round": int(job["round"]),
        "depth": int(job["depth"]),
        "starts": [int(s) for s in job["starts"]],
        "actor_mode": job["actor_mode"],
        "continuation_states": job.get("continuation_states", "branch-end"),
        "turns": int(job["horizon"]),
        "reset_options": dict(job["reset_options"]),
        "reset_identity": collection.reset_identity,
        "normalizer_digest": job["normalizer_digest"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "decisions": int(sum(c.decisions for c in continuations)),
        "native_resolves": int(native_resolves),
        "wall_seconds": float(seconds),
        "counters": dict(counters),
        "label_unavailable_reasons": dict(sorted(reasons.items())),
        "quarantine": [q for c in continuations for q in c.quarantine],
        "surviving_camps": int(collection.surviving_camps),
        "return_tail": RETURN_TAIL_RANDOMIZED,
        "return_tail_record": tail_record,
        "continuation_costs": per,
        "partition": "fitting",
    }
    directory = Path(job["continuation_out"]) / f"world-{world}"
    record = write_shard(directory, arrays, meta)
    return {
        "path": directory.name,
        "world_seed": world,
        "meta_sha256": checkpoint_hash(directory / "meta.json"),
        "data_digest": record["data_digest"],
        "rows": record["rows"],
        "decisions": record["decisions"],
        "native_resolves": record["native_resolves"],
        "wall_seconds": record["wall_seconds"],
        "continuations": len(continuations),
    }


def _existing_shard(
    out: Path,
    world: int,
    source_run: str,
    continuation_out: Path | None = None,
    continuation_source_run: str | None = None,
) -> dict[str, Any] | None:
    directory = out / f"world-{world}"
    side = None if continuation_out is None else continuation_out / f"world-{world}"
    if not directory.exists():
        if side is not None and side.exists():
            shutil.rmtree(side)  # continuation of an unpublished learner shard
        return None
    try:
        meta = verify_shard(directory)
    except (ProvenanceError, OSError, ValueError, KeyError):
        meta = None
    side_meta = None
    if meta is not None and side is not None:
        try:
            side_meta = verify_shard(side)
        except (ProvenanceError, OSError, ValueError, KeyError):
            side_meta = None
        bound = meta.get("teacher_continuation") or {}
        if (
            side_meta is None
            or side_meta.get("source_run") != continuation_source_run
            or bound.get("meta_sha256") != checkpoint_hash(side / "meta.json")
        ):
            meta = None
    if meta is None or meta.get("source_run") != source_run:
        shutil.rmtree(directory)  # unpublished/partial shard from an interrupted run
        if side is not None and side.exists():
            shutil.rmtree(side)
        return None
    entry = {
        "path": directory.name,
        "world_seed": world,
        "meta_sha256": checkpoint_hash(directory / "meta.json"),
        "data_digest": meta["data_digest"],
        "rows": meta["rows"],
        "decisions": meta["decisions"],
        "native_resolves": meta["native_resolves"],
        "wall_seconds": meta["wall_seconds"],
    }
    if side_meta is not None:
        entry["continuation"] = {
            "path": side.name,
            "world_seed": world,
            "meta_sha256": checkpoint_hash(side / "meta.json"),
            "data_digest": side_meta["data_digest"],
            "rows": side_meta["rows"],
            "decisions": side_meta["decisions"],
            "native_resolves": side_meta["native_resolves"],
            "wall_seconds": side_meta["wall_seconds"],
            "continuations": len(side_meta.get("continuation_costs", [])),
        }
    return entry


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="collect one learner-state round")
    collect.add_argument("--checkpoint", type=Path, required=True, help="frozen actor checkpoint")
    collect.add_argument("--worlds", required=True, help="fitting seeds/ranges, e.g. 90100-90159")
    collect.add_argument("--round", type=int, required=True, help="aggregation round (>= 1)")
    collect.add_argument("--depth", type=int, required=True, help="actor decisions per branch")
    collect.add_argument("--starts", type=int, default=3, help="branch starts per world")
    collect.add_argument("--horizon", type=int, default=25, help="teacher episode decisions")
    collect.add_argument("--start-key", type=int, default=DEFAULT_START_KEY)
    collect.add_argument("--stochastic", action="store_true", help="sample the actor (default: means)")
    collect.add_argument("--policy-seed", type=int, default=0)
    collect.add_argument("--players", type=int, default=16)
    collect.add_argument("--width", type=int, default=None)
    collect.add_argument("--height", type=int, default=None)
    collect.add_argument("--separation", type=int, default=None)
    collect.add_argument("--no-movement", action="store_true", help="named mechanical fixture only")
    collect.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    collect.add_argument("--resume", action="store_true", help="continue an interrupted output")
    collect.add_argument(
        "--teacher-continuation",
        choices=CONTINUATION_STATES,
        default="off",
        help="V^rho targets: one teacher (H12) continuation from each learner-visited state "
        "(branch-end: after the last actor decision; all-visited: after every actor decision)",
    )
    collect.add_argument(
        "--continuation-out",
        type=Path,
        default=None,
        help="separate dataset for the teacher continuations (source kind "
        "learner_state_teacher_continuation)",
    )
    collect.add_argument(
        "--return-tail",
        choices=("randomized",),
        default="randomized",
        help="continuation tail method: the preregistered randomized teacher tail (r5)",
    )
    tail_defaults = RandomizedTailSettings()
    collect.add_argument("--tail-draws", type=int, default=tail_defaults.draws)
    collect.add_argument("--tail-rho", type=float, default=tail_defaults.rho)
    collect.add_argument("--tail-cap", type=int, default=tail_defaults.cap)
    collect.add_argument(
        "--tail-schedule-seed-base", type=int, default=tail_defaults.seed_base,
        help="each world's frozen stopping schedule uses seed base + world seed",
    )
    collect.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development only: permit a dirty learner/simulator identity (non-production)",
    )
    collect.add_argument("--out", type=Path, required=True)
    return parser


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    from .bridge import DEFAULT_BRIDGE
    from .labor_bridge import LaborModel9Bridge
    from .labor_model import labor_personality_conditioning
    from .labor_teacher_data import normalizer_digest

    worlds = validate_fitting_worlds(seed_list(args.worlds), "labor dagger fitting worlds")
    if args.round < 1 or args.depth < 1 or args.starts < 1 or args.jobs < 1:
        raise ValueError("round, depth, starts and jobs must be positive")
    if args.depth > args.horizon:
        raise ValueError("depth cannot exceed the horizon")
    continuation = args.teacher_continuation != "off"
    if continuation and args.continuation_out is None:
        raise ValueError("--teacher-continuation needs --continuation-out")
    if not continuation and args.continuation_out is not None:
        raise ValueError("--continuation-out needs --teacher-continuation")
    if continuation and Path(args.continuation_out).resolve() == Path(args.out).resolve():
        raise ValueError("--continuation-out must differ from --out")
    tail = (
        RandomizedTailSettings(
            args.tail_draws, args.tail_rho, args.tail_cap, args.tail_schedule_seed_base
        )
        if continuation
        else None
    )
    if tail is not None:
        for world in worlds:
            tail.schedule_seed(world)  # refuse a denied schedule seed before any collection
    if (args.out / "dataset.json").is_file():
        if continuation and not (Path(args.continuation_out) / "dataset.json").is_file():
            raise ProvenanceError(
                "labor dagger output is published without its teacher-continuation dataset"
            )
        print(json.dumps({"dataset": str(args.out), "status": "already-published"}))
        return verify_dataset(args.out)
    if args.out.exists() and not args.resume:
        raise ProvenanceError("labor dagger output exists; pass --resume or use a new directory")
    if continuation and Path(args.continuation_out).exists() and not args.resume:
        raise ProvenanceError(
            "labor dagger continuation output exists; pass --resume or use a new directory"
        )
    reset = LaborResetOptions(
        players=args.players,
        movement=not args.no_movement,
        width=args.width,
        height=args.height,
        separation=args.separation,
    )
    actor, normalizer, personality = _load_actor(args.checkpoint)
    del actor
    reference, conditioning = labor_personality_conditioning(personality)
    actor_mode = "stochastic" if args.stochastic else "deterministic"
    starts = {
        world: draw_start_turns(args.start_key, world, args.round, args.starts, args.horizon, args.depth)
        for world in worlds
    }
    bridge_path = Path(os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE", str(DEFAULT_BRIDGE)))
    bridge_sha = checkpoint_hash(bridge_path)
    if not re.fullmatch(r"[0-9a-f]{64}", bridge_sha):
        raise ProvenanceError("production bridge binary missing")
    checkpoint_sha = checkpoint_hash(args.checkpoint)
    options = {
        "contract": DAGGER_CONTRACT,
        "worlds": worlds,
        "turns": int(args.horizon),
        "round": int(args.round),
        "depth": int(args.depth),
        "starts_per_world": int(args.starts),
        "start_key": int(args.start_key),
        "starts": {str(w): s for w, s in starts.items()},
        "actor_mode": actor_mode,
        "policy_seed": int(args.policy_seed),
        "checkpoint_sha256": checkpoint_sha,
        "reset_options": reset.state_dict(),
        "personality": reference.state_dict(),
        "personality_conditioning": conditioning.state_dict(),
        "personality_iteration": int(getattr(personality, "iteration", 0)),
        "normalizer_digest": normalizer_digest(normalizer),
        "teacher": "h12-converted-native-heuristic-shadow-step",
        "actor_only": True,
        "partition": "fitting",
    }
    if continuation:
        # Recorded in the learner source too: the continuation fork/restore is part of the run.
        options["teacher_continuation"] = {
            "states": args.teacher_continuation,
            "return_tail": tail.state_dict(),
        }
    with LaborModel9Bridge() as probe:
        source = source_record(
            options,
            probe.schema,
            bridge_sha,
            production=not args.allow_dirty,
            source_kind=DAGGER_SOURCE_KIND,
        )
    continuation_source = None
    if continuation:
        with LaborModel9Bridge() as probe:
            continuation_source = source_record(
                {
                    **options,
                    "contract": CONTINUATION_CONTRACT,
                    "dagger_source_run": source["source_run"],
                    "actor_only": False,
                    "v_target": "teacher-rho-continuation-from-learner-visited-states",
                    "return_tail": tail.state_dict(),
                },
                probe.schema,
                bridge_sha,
                production=not args.allow_dirty,
                source_kind=CONTINUATION_SOURCE_KIND,
            )
    outputs = [(args.out, source)]
    if continuation:
        outputs.append((Path(args.continuation_out), continuation_source))
    expected = {f"world-{world}" for world in worlds}
    for out, record in outputs:
        out.mkdir(parents=True, exist_ok=True)
        source_path = out / "source.json"
        if source_path.is_file():
            previous = json.loads(source_path.read_text())
            if previous.get("source_run") != record["source_run"]:
                raise ProvenanceError(
                    "labor dagger resume: source record differs (checkpoint, code, bridge or "
                    "options)"
                )
        else:
            tmp = out / "source.json.tmp"
            tmp.write_text(json.dumps(record, sort_keys=True, allow_nan=False))
            os.replace(tmp, source_path)
        undeclared = sorted(p.name for p in out.iterdir() if p.is_dir() and p.name not in expected)
        if undeclared:
            raise ProvenanceError(f"labor dagger output holds undeclared directories: {undeclared}")
    done = {}
    for world in worlds:
        entry = _existing_shard(
            args.out,
            world,
            source["source_run"],
            Path(args.continuation_out) if continuation else None,
            continuation_source["source_run"] if continuation else None,
        )
        if entry is not None:
            done[world] = entry
    jobs = [
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "normalizer_digest": options["normalizer_digest"],
            "reset_options": reset.state_dict(),
            "world_seed": world,
            "round": int(args.round),
            "depth": int(args.depth),
            "starts": starts[world],
            "horizon": int(args.horizon),
            "actor_mode": actor_mode,
            "policy_seed": int(args.policy_seed),
            "source_run": source["source_run"],
            "out": str(args.out),
            "continuation_tail": None if tail is None else tail.state_dict(),
            "continuation_states": args.teacher_continuation,
            "continuation_out": str(args.continuation_out) if continuation else None,
            "continuation_source_run": (
                continuation_source["source_run"] if continuation else None
            ),
        }
        for world in worlds
        if world not in done
    ]
    print(
        json.dumps({"dagger_round": args.round, "depth": args.depth, "worlds": len(worlds),
                    "resumed_shards": len(done), "jobs": len(jobs)}),
        file=sys.stderr,
        flush=True,
    )
    started = time.perf_counter()
    if args.jobs == 1 or len(jobs) <= 1:
        shards = [_collect_shard(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
            shards = list(pool.map(_collect_shard, jobs))
    if checkpoint_hash(bridge_path) != bridge_sha or code_commit() != source["learner_commit"]:
        raise ProvenanceError("labor dagger producer changed during collection; not published")
    all_shards = list(done.values()) + shards
    if continuation:
        publish_dataset(
            Path(args.continuation_out),
            continuation_source,
            [s["continuation"] for s in all_shards],
        )
    manifest = publish_dataset(
        args.out, source, [{k: v for k, v in s.items() if k != "continuation"} for s in all_shards]
    )
    elapsed = time.perf_counter() - started
    rows = sum(s["rows"] for s in shards)
    process_seconds = sum(s["wall_seconds"] for s in shards)
    summary = {
        "dataset": str(args.out),
        "dataset_digest": manifest["dataset_digest"],
        "round": int(args.round),
        "depth": int(args.depth),
        "worlds": len(all_shards),
        "rows": int(sum(s["rows"] for s in all_shards)),
        "rows_this_run": int(rows),
        "native_resolves_this_run": int(sum(s["native_resolves"] for s in shards)),
        "wall_seconds": elapsed,
        "rows_per_minute": rows / max(elapsed, 1e-9) * 60.0,
        "rows_per_process_minute": rows / max(process_seconds, 1e-9) * 60.0,
    }
    if continuation:
        parts = [s["continuation"] for s in shards]
        count = sum(int(p["continuations"]) for p in parts)
        summary["teacher_continuation"] = {
            "dataset": str(args.continuation_out),
            "states": args.teacher_continuation,
            "continuations_this_run": count,
            "rows_this_run": int(sum(p["rows"] for p in parts)),
            "native_resolves_this_run": int(sum(p["native_resolves"] for p in parts)),
            "wall_seconds_this_run": float(sum(p["wall_seconds"] for p in parts)),
            "native_resolves_per_continuation": (
                sum(p["native_resolves"] for p in parts) / count if count else None
            ),
            "wall_seconds_per_continuation": (
                sum(p["wall_seconds"] for p in parts) / count if count else None
            ),
            "learner_wall_seconds_this_run": float(
                process_seconds - sum(p["wall_seconds"] for p in parts)
            ),
        }
    print(json.dumps(summary, sort_keys=True))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.command == "collect":
        run_collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
