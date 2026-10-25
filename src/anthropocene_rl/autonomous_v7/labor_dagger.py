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

from .labor_bc import (
    HeuristicLaborLabel,
    LaborResetOptions,
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
from .protocol import seed_list
from .provenance import ProvenanceError, checkpoint_hash, code_commit
from .labor_teacher_data import (
    publish_dataset,
    source_record,
    verify_dataset,
    verify_shard,
    write_shard,
)

DAGGER_SOURCE_KIND = "learner_state_dagger"
DAGGER_ROW_SOURCE = 1
DAGGER_CONTRACT = "model9-labor-dagger-learner-state-v1"
DEFAULT_START_KEY = 4243


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
) -> WorldCollection:
    """Teacher prefix, forked actor branches of ``depth`` decisions, teacher labels on every branch state.

    ``personality`` is the checkpoint's declared personality (fixed or a
    distribution assignment); it conditions the actor and is recorded in the
    row inputs exactly as the collector does. The teacher label itself is
    personality-agnostic.
    """
    from .bridge import Model9Bridge
    from .labor_bridge import LaborModel9Bridge
    from .labor_model import labor_policy_tensors
    from .labor_personality import personality_for_world
    from .labor_pretrain_eval import actor_labor_decision
    from .labor_training import _settlement_guard, _transition_rewards

    options = reset_options or LaborResetOptions()
    labor_factory = labor_bridge_factory or LaborModel9Bridge
    heur_factory = heuristic_bridge_factory or (lambda: Model9Bridge(economic_diagnostics=True))
    world = int(world_seed)
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
            try:
                for branch_step in range(depth):
                    live = _live(branch)
                    if not live:
                        break
                    contexts, tensors = labor_policy_tensors(
                        live, normalizer, personality=world_personality
                    )
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
                        )
                    branch = next_step
                    if progress is not None:
                        progress(result)
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
    )
    seconds = time.perf_counter() - started
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
    }
    directory = Path(job["out"]) / f"world-{world}"
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
    }


def _existing_shard(out: Path, world: int, source_run: str) -> dict[str, Any] | None:
    directory = out / f"world-{world}"
    if not directory.exists():
        return None
    try:
        meta = verify_shard(directory)
    except (ProvenanceError, OSError, ValueError, KeyError):
        meta = None
    if meta is None or meta.get("source_run") != source_run:
        shutil.rmtree(directory)  # unpublished/partial shard from an interrupted run
        return None
    return {
        "path": directory.name,
        "world_seed": world,
        "meta_sha256": checkpoint_hash(directory / "meta.json"),
        "data_digest": meta["data_digest"],
        "rows": meta["rows"],
        "decisions": meta["decisions"],
        "native_resolves": meta["native_resolves"],
        "wall_seconds": meta["wall_seconds"],
    }


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
    if (args.out / "dataset.json").is_file():
        print(json.dumps({"dataset": str(args.out), "status": "already-published"}))
        return verify_dataset(args.out)
    if args.out.exists() and not args.resume:
        raise ProvenanceError("labor dagger output exists; pass --resume or use a new directory")
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
    with LaborModel9Bridge() as probe:
        source = source_record(
            options,
            probe.schema,
            bridge_sha,
            production=not args.allow_dirty,
            source_kind=DAGGER_SOURCE_KIND,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    source_path = args.out / "source.json"
    if source_path.is_file():
        previous = json.loads(source_path.read_text())
        if previous.get("source_run") != source["source_run"]:
            raise ProvenanceError(
                "labor dagger resume: source record differs (checkpoint, code, bridge or options)"
            )
    else:
        tmp = args.out / "source.json.tmp"
        tmp.write_text(json.dumps(source, sort_keys=True, allow_nan=False))
        os.replace(tmp, source_path)
    expected = {f"world-{world}" for world in worlds}
    undeclared = sorted(p.name for p in args.out.iterdir() if p.is_dir() and p.name not in expected)
    if undeclared:
        raise ProvenanceError(f"labor dagger output holds undeclared directories: {undeclared}")
    done = {}
    for world in worlds:
        entry = _existing_shard(args.out, world, source["source_run"])
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
    manifest = publish_dataset(args.out, source, all_shards)
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
    print(json.dumps(summary, sort_keys=True))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.command == "collect":
        run_collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
