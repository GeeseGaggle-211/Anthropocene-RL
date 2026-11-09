"""Native fork producer for independent labor value anchors.

The command in this module is intentionally narrow: at the reset decision of
each explicitly supplied local world it collects four sampled-policy paths and
one paired alternative per path.  Each path uses a pre-result geometric
Russian-roulette schedule, no critic bootstrap, and an in-process native fork.
The sampled-policy arm supplies V(s) anchors; paired distinct successors supply
``r + gamma V(s')`` ranking witnesses.  Camps and paths remain correlated rows
inside one world unit.

Importing this module never starts the simulator.  Local execution is capped
at three worlds and is diagnostic, not readiness-bearing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contracts
from .gate34_continuation_collector import (
    FrozenGeometricSchedule,
    bounded_geometric_tail_assumptions,
    estimate_randomized_tail,
    freeze_geometric_schedule,
    raw_camp_reward_absolute_bound,
)
from .labor_bridge import LABOR_REGISTRY_DIGEST, LaborModel9Bridge, LaborProducerContract
from .labor_model import (
    LABOR_COST_REFERENCE_SCALES,
    LABOR_MODEL_INPUT_CONTRACT,
    LABOR_OBJECTIVE_REFERENCE_SCALES,
    LABOR_OBJECTIVES,
    LaborPersonality,
    labor_checkpoint_personality,
    labor_policy_tensors,
)
from .labor_personality import WorldPersonalities, personality_for_world
from .labor_training import (
    LOCAL_WORLD_LIMIT,
    _models_from_checkpoint,
    _normalization_digest,
    _rollout_privileged,
    _sample_world_step,
    labor_behavior_configuration,
)
from .labor_value_calibration import (
    LABOR_ANCHOR_KIND,
    LABOR_RANK_SEMANTICS,
    LABOR_VALUE_HEADS,
)
from .protocol import scan
from .protocol import seeds as checked_seeds
from .provenance import checkpoint_hash, digest
from .value_calibration import behavior_policy_digest

LABOR_ANCHOR_PRODUCER = "native-labor-randomized-continuation-source-v1"
DEFAULT_PREFIX_TERMS = 25
DEFAULT_TAIL_DRAWS = 16
DEFAULT_TAIL_RHO = 0.95
DEFAULT_TAIL_CAP = 200
DEFAULT_PATHS = 4
PROGRESS_EVERY_TERMS = 5


@dataclass(frozen=True)
class PathHeadReturn:
    value: float | None
    standard_error: float | None
    method: str
    immediate: float
    terminal_first: bool

    @property
    def available(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class LaborContinuationPath:
    world_seed: int
    path_index: int
    alternative: str
    schedule: FrozenGeometricSchedule
    successor_inputs: Mapping[int, Mapping[str, np.ndarray]]
    successor_digests: Mapping[int, str]
    rewards: Mapping[int, Mapping[str, tuple[float, ...]]]
    years: Mapping[int, tuple[float, ...]]
    terminated: Mapping[int, tuple[bool, ...]]
    successor_privileged: Mapping[int, Mapping[str, np.ndarray]] | None = None


def _tensor_row(tensors: Mapping[str, torch.Tensor], row: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or not 0 <= row < len(value):
            raise ValueError("labor anchor input tensors are not row aligned")
        result[name] = value[row].detach().cpu().numpy().copy()
    return result


def _stack(rows: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("labor anchor cannot stack zero input rows")
    fields = set(rows[0])
    if any(set(row) != fields for row in rows):
        raise ValueError("labor anchor input schemas differ")
    return {name: np.stack([np.asarray(row[name]) for row in rows]) for name in sorted(fields)}


def _input_digest(row: Mapping[str, np.ndarray]) -> str:
    return digest(
        {
            name: {
                "dtype": str(np.asarray(value).dtype),
                "shape": list(np.asarray(value).shape),
                "bytesSha256": __import__("hashlib").sha256(
                    np.ascontiguousarray(value).tobytes()
                ).hexdigest(),
            }
            for name, value in sorted(row.items())
        }
    )


def _head_reward(row: Any, head: str) -> float:
    if head in LABOR_OBJECTIVES:
        return float(row.physical_task_rewards[head])
    if head == "feasibility":
        return float(row.feasibility_initial + row.feasibility_retry)
    if head == "settlement_guard":
        return float(row.settlement_guard)
    raise ValueError(f"unknown labor value head {head!r}")


def summarize_labor_path(
    rewards: Sequence[float],
    years: Sequence[float],
    terminated: Sequence[bool],
    schedule: FrozenGeometricSchedule,
    *,
    head: str,
    prefix_terms: int = DEFAULT_PREFIX_TERMS,
    provenance_id: str,
) -> PathHeadReturn:
    """Close one camp path exactly at termination or with the frozen tail draw."""

    if head not in LABOR_VALUE_HEADS:
        raise ValueError(f"unknown labor head {head!r}")
    if not rewards or len(rewards) != len(years) or len(rewards) != len(terminated):
        raise ValueError("labor continuation rewards/years/termination are misaligned")
    if any(not math.isfinite(float(value)) for value in (*rewards, *years)):
        raise ValueError("labor continuation path contains non-finite values")
    if any(float(value) != contracts.TURN_YEARS for value in years):
        raise ValueError("labor randomized-tail proof requires the fixed native turn duration")
    ended = [index for index, value in enumerate(terminated) if value]
    if ended and (len(ended) != 1 or ended[0] != len(rewards) - 1):
        raise ValueError("labor continuation contains outcomes after camp termination")
    discount = 1.0
    discounted: list[float] = []
    for reward, elapsed in zip(rewards, years, strict=True):
        discounted.append(discount * float(reward))
        discount *= math.exp(-float(elapsed) / contracts.HORIZON_YEARS)
    immediate = float(rewards[0])
    if ended:
        return PathHeadReturn(
            value=float(sum(discounted)),
            standard_error=0.0,
            method="terminal-exact",
            immediate=immediate,
            terminal_first=bool(terminated[0]),
        )
    if len(rewards) < prefix_terms + schedule.required_terms:
        raise ValueError("living labor path does not cover its frozen stopping schedule")
    prefix = float(sum(discounted[:prefix_terms]))
    producer = LaborProducerContract().state_dict()
    if head == "qol_years":
        per_turn = raw_camp_reward_absolute_bound("qol_years", 1)
    elif head == "population_years":
        per_turn = raw_camp_reward_absolute_bound("population_years", 1)
    elif head == "feasibility":
        per_turn = float(producer["decisionFeasibilityCostMaximum"])
    else:
        per_turn = float(producer["settlementGuardCostMaximum"])
    assumptions = bounded_geometric_tail_assumptions(
        per_turn_absolute_bound=per_turn,
        discount=math.exp(-contracts.TURN_YEARS / contracts.HORIZON_YEARS),
        survival_probability=schedule.survival_probability,
        provenance_id=provenance_id,
    )
    tail = estimate_randomized_tail(discounted[prefix_terms:], schedule, assumptions)
    if not tail.identified:
        return PathHeadReturn(None, None, "unavailable", immediate, False)
    assert tail.estimate is not None and tail.standard_error is not None
    return PathHeadReturn(
        value=prefix + tail.estimate,
        standard_error=tail.standard_error,
        method="independent-randomized-tail-path-mean",
        immediate=immediate,
        terminal_first=False,
    )


def _combine_paths(paths: Sequence[PathHeadReturn]) -> tuple[float, float, str, bool]:
    if not paths or any(not path.available for path in paths):
        return 0.0, 0.0, "unavailable", False
    values = np.asarray([path.value for path in paths], dtype=np.float64)
    within = np.asarray([path.standard_error for path in paths], dtype=np.float64)
    between = float(values.var(ddof=1) / len(values)) if len(values) > 1 else 0.0
    within_mean = float(np.sum(within**2) / len(values) ** 2)
    method = (
        "terminal-exact"
        if all(path.method == "terminal-exact" for path in paths)
        else "independent-randomized-tail-path-mean"
    )
    return float(values.mean()), math.sqrt(between + within_mean), method, True


def _successor_inputs(
    step: Mapping[str, Any], normalizer: Any, personality: LaborPersonality | WorldPersonalities
) -> dict[int, dict[str, np.ndarray]]:
    live = [
        village for village in step["villages"]
        if village.get("observation") is not None and not village.get("terminated")
    ]
    if not live:
        return {}
    contexts, tensors = labor_policy_tensors(live, normalizer, personality=personality)
    return {
        context[0].camp_id: _tensor_row(tensors, row)
        for row, context in enumerate(contexts)
    }


def _privileged_inputs(step, task, costs):
    live = [
        v for v in step["villages"]
        if v.get("observation") is not None and not v.get("terminated")
    ]
    if not live:
        return {}
    raw = _rollout_privileged(live, task, costs)
    return {} if raw is None else {int(v["camp"]): _tensor_row(raw, i) for i, v in enumerate(live)}


def _run_continuation(
    *,
    bridge: LaborModel9Bridge,
    step: Mapping[str, Any],
    actor: Any,
    task_critic: Any = None,
    cost_critics: Any = None,
    normalizer: Any,
    personality: LaborPersonality | WorldPersonalities,
    initial_seed: int,
    future_seed: int,
    world_seed: int,
    world_index: int,
    opening_camps: Sequence[int],
    schedule: FrozenGeometricSchedule,
    path_index: int,
    alternative: str,
    prefix_terms: int,
    progress_every: int = 0,
) -> LaborContinuationPath:
    clock = time.perf_counter()
    initial_generator = torch.Generator().manual_seed(initial_seed)
    future_generator = torch.Generator().manual_seed(future_seed)
    founded = {int(row["camp"]): int(step["turn"]) for row in step["villages"]}
    rewards: dict[int, dict[str, list[float]]] = {
        camp: {head: [] for head in LABOR_VALUE_HEADS} for camp in opening_camps
    }
    years: dict[int, list[float]] = {camp: [] for camp in opening_camps}
    terminal: dict[int, list[bool]] = {camp: [] for camp in opening_camps}
    maximum_terms = prefix_terms + schedule.required_terms
    successor: dict[int, dict[str, np.ndarray]] = {}
    successor_digests: dict[int, str] = {}
    successor_privileged = {}
    for term in range(maximum_terms):
        step, rows = _sample_world_step(
            bridge=bridge,
            step=step,
            actor=actor,
            task_critic=None,
            cost_critics=None,
            normalizer=normalizer,
            personality=personality,
            generator=initial_generator if term == 0 else future_generator,
            world_seed=world_seed,
            world_index=world_index,
            founded_turns=founded,
        )
        by_camp = {row.camp_id: row for row in rows}
        for camp in opening_camps:
            if terminal[camp] and terminal[camp][-1]:
                continue
            row = by_camp.get(camp)
            if row is None:
                raise ValueError("opening labor camp vanished without a terminal outcome")
            for head in LABOR_VALUE_HEADS:
                rewards[camp][head].append(_head_reward(row, head))
            years[camp].append(row.years)
            terminal[camp].append(row.terminated)
        if term == 0:
            successor = _successor_inputs(step, normalizer, personality)
            successor_privileged = _privileged_inputs(step, task_critic, cost_critics)
            for camp in opening_camps:
                row = successor.get(camp)
                successor_digests[camp] = digest(
                    {
                        "worldFingerprint": str(step.get("fingerprint", "terminal")),
                        "camp": camp,
                        "input": None if row is None else _input_digest(row),
                    }
                )
        if progress_every and (term + 1) % progress_every == 0:
            # wall-clock telemetry on stderr only; never in the payload
            print(
                json.dumps(
                    {
                        "anchorTerm": {
                            "world": world_seed,
                            "path": path_index,
                            "alternative": alternative,
                            "terms": term + 1,
                            "maximumTerms": maximum_terms,
                            "seconds": round(time.perf_counter() - clock, 3),
                        }
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        if all(terminal[camp] and terminal[camp][-1] for camp in opening_camps):
            break
    return LaborContinuationPath(
        world_seed,
        path_index,
        alternative,
        schedule,
        successor,
        successor_digests,
        {
            camp: {head: tuple(values) for head, values in heads.items()}
            for camp, heads in rewards.items()
        },
        {camp: tuple(values) for camp, values in years.items()},
        {camp: tuple(values) for camp, values in terminal.items()},
        successor_privileged,
    )


def _schedule_seed(base: int, world_index: int, path_index: int) -> int:
    return int(base + world_index * 10_000 + path_index * 10)


def _path_rng_seeds(base: int, world_index: int, path_index: int) -> tuple[int, ...]:
    """Every actual stopping/action RNG identity used by one paired path."""

    schedule = _schedule_seed(base, world_index, path_index)
    return (schedule, schedule * 100, schedule * 100 + 1, schedule * 100 + 99)


@dataclass(frozen=True)
class _WorldParent:
    """The reset decision of one world: the parent rows every path starts from."""

    runtime_identity: Mapping[str, Any]
    world_personality: Any
    opening: tuple[int, ...]
    parent_by_camp: Mapping[int, Mapping[str, np.ndarray]]
    privileged_by_camp: Mapping[int, Mapping[str, np.ndarray]]
    parent_digest: str


def _world_parent(bridge, world, reset, normalizer, personality, task, costs) -> _WorldParent:
    parent_step = bridge.reset(world, **reset.reset_kwargs())
    identity = bridge.runtime_identity()
    live = [
        village for village in parent_step["villages"]
        if village.get("observation") is not None and not village.get("terminated")
    ]
    world_personality = personality_for_world(personality, world)
    contexts, parent_tensors = labor_policy_tensors(
        live, normalizer, personality=world_personality
    )
    opening = [context[0].camp_id for context in contexts]
    if len(opening) != len(set(opening)):
        raise ValueError("labor anchor parent repeats a camp")
    privileged_by_camp = _privileged_inputs(parent_step, task, costs)
    parent_by_camp = {
        camp: _tensor_row(parent_tensors, row) for row, camp in enumerate(opening)
    }
    parent_digest = digest(
        {
            "fingerprint": parent_step.get("fingerprint"),
            "inputs": {camp: _input_digest(row) for camp, row in parent_by_camp.items()},
        }
    )
    return _WorldParent(
        identity, world_personality, tuple(opening), parent_by_camp, privileged_by_camp,
        parent_digest,
    )


def _path_pair(
    bridge, fork, *, actor, task, costs, normalizer, parent: _WorldParent, world: int,
    world_index: int, schedule: FrozenGeometricSchedule, path_index: int, prefix_terms: int,
    progress: bool = False,
) -> list[LaborContinuationPath]:
    """One path's sampled-policy and paired-sample continuations, each from the fork."""

    continuation_seed = schedule.rng_seed
    assert continuation_seed is not None
    paths = []
    for alternative_index, alternative in enumerate(("sampled-policy", "paired-sample")):
        clock = time.perf_counter()
        restored = bridge.restore(fork)
        telemetry = {"progress_every": PROGRESS_EVERY_TERMS} if progress else {}
        path = _run_continuation(
            **telemetry,
            bridge=bridge,
            step=restored,
            actor=actor,
            task_critic=task,
            cost_critics=costs,
            normalizer=normalizer,
            personality=parent.world_personality,
            initial_seed=continuation_seed * 100 + alternative_index,
            future_seed=continuation_seed * 100 + 99,
            world_seed=world,
            world_index=world_index,
            opening_camps=list(parent.opening),
            schedule=schedule,
            path_index=path_index,
            alternative=alternative,
            prefix_terms=prefix_terms,
        )
        paths.append(path)
        if progress:  # wall-clock telemetry on stderr only; never in the payload
            terms = max((len(values) for values in path.years.values()), default=0)
            print(
                json.dumps(
                    {
                        "anchorProgress": {
                            "world": world,
                            "path": path_index,
                            "alternative": alternative,
                            "terms": terms,
                            "maximumTerms": prefix_terms + schedule.required_terms,
                            "seconds": round(time.perf_counter() - clock, 3),
                        }
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    return paths


def _worker_init(threads: int | None) -> None:
    """Pool worker setup: die with the parent (Linux) and pin the torch thread count."""

    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal

            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PDEATHSIG
        except Exception:  # best effort: the job script's cap still bounds the run
            pass
    if threads is not None:
        torch.set_num_threads(int(threads))


def _anchor_unit(job: Mapping[str, Any]) -> dict[str, Any]:
    """One (world, path) unit in a pool worker: fresh reset, fork, both continuations.

    Loads the exact checkpoint by path (sha-checked) and refuses another sampled policy;
    the parent compares every unit's reset decision for its world before assembly.
    """

    clock = time.perf_counter()
    path = Path(job["checkpointPath"])
    if checkpoint_hash(path) != job["checkpointSha256"]:
        raise ValueError("labor anchor worker: checkpoint changed during collection")
    state = torch.load(path, map_location="cpu", weights_only=True)
    _, actor, task, costs, normalizer = _models_from_checkpoint(state)
    personality = labor_checkpoint_personality(state)
    behavior = labor_behavior_configuration(personality)
    if behavior_policy_digest(actor, behavior) != job["policyDigest"]:
        raise ValueError("labor anchor worker loaded another sampled policy")
    bridge = job["bridgeFactory"]()
    try:
        parent = _world_parent(
            bridge, job["world"], job["reset"], normalizer, personality, task, costs
        )
        fork, _ = bridge.fork()
        try:
            paths = _path_pair(
                bridge, fork, actor=actor, task=task, costs=costs, normalizer=normalizer,
                parent=parent, world=job["world"], world_index=job["worldIndex"],
                schedule=job["schedule"], path_index=job["pathIndex"],
                prefix_terms=job["prefixTerms"], progress=job["progress"],
            )
        finally:
            bridge.restore(fork)
            bridge.drop(fork)
    finally:
        bridge.close()
    return {
        "world": job["world"],
        "pathIndex": job["pathIndex"],
        "parent": parent,
        "paths": paths,
        "seconds": time.perf_counter() - clock,
    }


def _default_executor(max_workers: int, threads: int | None) -> Any:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
        initargs=(threads,),
    )


def _parallel_units(
    *, checkpoint_path, worlds, schedules, path_count, prefix_terms, reset, bridge_factory,
    jobs, threads, progress, policy_digest, executor_factory,
) -> dict[int, list[dict[str, Any]]]:
    sha = checkpoint_hash(checkpoint_path)
    jobs_list = [
        {
            "checkpointPath": str(checkpoint_path),
            "checkpointSha256": sha,
            "policyDigest": policy_digest,
            "bridgeFactory": bridge_factory,
            "world": world,
            "worldIndex": world_index,
            "pathIndex": path_index,
            "schedule": schedules[(world, path_index)],
            "prefixTerms": prefix_terms,
            "reset": reset,
            "progress": progress,
        }
        for world_index, world in enumerate(worlds)
        for path_index in range(path_count)
    ]
    factory = executor_factory if executor_factory is not None else _default_executor
    executor = factory(min(jobs, len(jobs_list)), threads)
    try:
        futures = [executor.submit(_anchor_unit, job) for job in jobs_list]
        results = [future.result() for future in futures]  # canonical (world, path) order
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    by_world: dict[int, list[dict[str, Any]]] = {world: [] for world in worlds}
    for result in results:
        by_world[result["world"]].append(result)
    return by_world


def _same_privileged(left: Mapping[int, Any], right: Mapping[int, Any]) -> bool:
    if set(left) != set(right):
        return False
    for camp, rows in left.items():
        if set(rows) != set(right[camp]) or any(
            not np.array_equal(value, right[camp][key]) for key, value in rows.items()
        ):
            return False
    return True


def _merge_world_units(
    world: int, units: Sequence[Mapping[str, Any]], path_count: int
) -> tuple[_WorldParent, Mapping[str, Any], list[LaborContinuationPath]]:
    """Refuse a world whose units did not start from the identical reset decision."""

    if [unit["pathIndex"] for unit in units] != list(range(path_count)):
        raise ValueError(f"labor anchor world {world} lacks a path unit")
    first = units[0]["parent"]
    for unit in units[1:]:
        other = unit["parent"]
        if (
            other.parent_digest != first.parent_digest
            or other.opening != first.opening
            or dict(other.runtime_identity) != dict(first.runtime_identity)
            or not _same_privileged(first.privileged_by_camp, other.privileged_by_camp)
        ):
            raise ValueError(
                f"labor anchor world {world}: parallel units saw different reset decisions "
                "(native reset is not reproducible across processes); use --jobs 1"
            )
    paths = [path for unit in units for path in unit["paths"]]
    return first, first.runtime_identity, paths


def collect_labor_anchor_arrays(
    *,
    checkpoint_path: Path,
    state: Mapping[str, Any],
    world_seeds: Sequence[int],
    optimizer_seed: int,
    continuation_seed_base: int,
    role: str,
    prefix_terms: int = DEFAULT_PREFIX_TERMS,
    tail_draws: int = DEFAULT_TAIL_DRAWS,
    tail_rho: float = DEFAULT_TAIL_RHO,
    tail_cap: int = DEFAULT_TAIL_CAP,
    path_count: int = DEFAULT_PATHS,
    bridge_factory: Any = LaborModel9Bridge,
    allow_vm_scale: bool = False,
    reset_options: Any = None,
    jobs: int = 1,
    threads: int | None = None,
    progress: bool = False,
    executor_factory: Any = None,
) -> dict[str, np.ndarray]:
    """Collect a strict NPZ payload without using any critic value as a target.

    ``reset_options`` (a :class:`labor_bc.LaborResetOptions`) is the manifest's
    native reset contract; its default enables movement.

    ``jobs`` (default 1: one bridge, worlds and paths in sequence) > 1 runs every
    (world, path) unit in its own spawned process with its own bridge: a fresh reset of
    the world, a fork, then the same two continuations from the restored fork. Every
    seed, schedule and row order is the sequential one; each world's units must report
    the identical reset decision (fingerprint, inputs, privileged rows, runtime identity)
    or the collection is refused. ``threads`` pins each worker's torch threads.
    ``progress`` prints per-continuation wall-clock lines on stderr (never in the payload).
    """
    from .labor_bc import LaborResetOptions

    reset = reset_options if reset_options is not None else LaborResetOptions()
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("labor anchor jobs must be a positive integer")

    worlds = checked_seeds(list(world_seeds), "labor anchor worlds")
    continuations = checked_seeds(
        [
            seed
            for world_index in range(len(worlds))
            for path in range(path_count)
            for seed in _path_rng_seeds(continuation_seed_base, world_index, path)
        ],
        "labor anchor continuation seeds",
    )
    scan(
        {
            "worldSeeds": worlds,
            "optimizerSeed": optimizer_seed,
            "continuationSeeds": continuations,
        },
        "labor anchor seeds",
    )
    if not allow_vm_scale and (not worlds or len(worlds) > LOCAL_WORLD_LIMIT):
        raise ValueError(f"local labor anchor collection requires 1..{LOCAL_WORLD_LIMIT} worlds")
    if not worlds:
        raise ValueError("world seeds cannot be empty")
    if role not in {"train-anchor", "actor-gate"}:
        raise ValueError("labor anchor role must be train-anchor or actor-gate")
    if path_count < 2:
        raise ValueError("labor anchor policy expectation requires at least two paths")
    layout, actor, task, costs, normalizer = _models_from_checkpoint(state)
    del layout
    personality = labor_checkpoint_personality(state)
    behavior = labor_behavior_configuration(personality)
    actor_tensor_digest = __import__(
        "anthropocene_rl.autonomous_v7.economic_trace", fromlist=["state_digest"]
    ).state_digest(actor)
    policy_digest = behavior_policy_digest(actor, behavior)
    schedules: dict[tuple[int, int], FrozenGeometricSchedule] = {}
    for world_index, world in enumerate(worlds):
        for path_index in range(path_count):
            seed = _schedule_seed(continuation_seed_base, world_index, path_index)
            schedule = freeze_geometric_schedule(
                schedule_id=f"world-{world}/path-{path_index}",
                rng_seed=seed,
                draw_count=tail_draws,
                survival_probability=tail_rho,
                hard_cap_terms=tail_cap,
            )
            if schedule.exceeds_hard_cap:
                raise ValueError(
                    f"frozen labor schedule {schedule.schedule_id} exceeds its hard cap; "
                    "the schedule was retained and must not be redrawn"
                )
            schedules[(world, path_index)] = schedule
    if progress:  # the frozen work plan, for wall-clock extrapolation (stderr only)
        print(
            json.dumps(
                {
                    "anchorPlan": {
                        "role": role,
                        "worlds": list(worlds),
                        "paths": path_count,
                        "continuationsPerPath": 2,
                        "jobs": jobs,
                        "threads": threads,
                        "maximumTerms": {
                            f"{world}/{path}": prefix_terms + schedule.required_terms
                            for (world, path), schedule in sorted(schedules.items())
                        },
                    }
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    parent_privileged = []
    rank_privileged = []
    parent_rows: list[dict[str, np.ndarray]] = []
    row_ids: list[str] = []
    unit_ids: list[str] = []
    targets = {head: [] for head in LABOR_VALUE_HEADS}
    errors = {head: [] for head in LABOR_VALUE_HEADS}
    methods = {head: [] for head in LABOR_VALUE_HEADS}
    masks = {head: [] for head in LABOR_VALUE_HEADS}
    rank_inputs: list[dict[str, np.ndarray]] = []
    rank_groups: list[str] = []
    rank_parents: list[str] = []
    rank_successors: list[str] = []
    rank_units: list[str] = []
    rank_discount: list[float] = []
    rank_terminal: list[bool] = []
    rank_immediate = {head: [] for head in LABOR_VALUE_HEADS}
    rank_realized = {head: [] for head in LABOR_VALUE_HEADS}
    rank_errors = {head: [] for head in LABOR_VALUE_HEADS}
    rank_masks = {head: [] for head in LABOR_VALUE_HEADS}
    runtime_identities: dict[str, Mapping[str, Any]] = {}

    def assemble(world: int, parent: _WorldParent, paths: Sequence[LaborContinuationPath]) -> None:
        opening = parent.opening
        parent_by_camp = parent.parent_by_camp
        privileged_by_camp = parent.privileged_by_camp
        parent_digest = parent.parent_digest
        unit = f"world-{world}"
        by_path = {(path.path_index, path.alternative): path for path in paths}
        baseline = [by_path[(index, "sampled-policy")] for index in range(path_count)]
        for camp in opening:
            parent_rows.append(parent_by_camp[camp])
            if privileged_by_camp:
                parent_privileged.append(privileged_by_camp[camp])
            row_ids.append(f"{unit}/decision-0/camp-{camp}")
            unit_ids.append(unit)
            for head in LABOR_VALUE_HEADS:
                summaries = [
                    summarize_labor_path(
                        path.rewards[camp][head],
                        path.years[camp],
                        path.terminated[camp],
                        path.schedule,
                        head=head,
                        prefix_terms=prefix_terms,
                        provenance_id=f"{unit}/path-{path.path_index}/{head}/camp-{camp}",
                    )
                    for path in baseline
                ]
                value, standard_error, method, available = _combine_paths(summaries)
                targets[head].append(value)
                errors[head].append(standard_error)
                methods[head].append(method)
                masks[head].append(available)

            for path_index in range(path_count):
                pair = [
                    by_path[(path_index, "sampled-policy")],
                    by_path[(path_index, "paired-sample")],
                ]
                group = f"{unit}/path-{path_index}/camp-{camp}"
                for path in pair:
                    successor = path.successor_inputs.get(camp, parent_by_camp[camp])
                    rank_inputs.append(dict(successor))
                    if privileged_by_camp:
                        # Terminal rows use a shape-valid placeholder; their bootstrap is masked.
                        rank_privileged.append(
                            (path.successor_privileged or {}).get(camp, privileged_by_camp[camp])
                        )
                    rank_groups.append(group)
                    rank_parents.append(parent_digest)
                    rank_successors.append(path.successor_digests[camp])
                    rank_units.append(unit)
                    first_years = path.years[camp][0]
                    rank_discount.append(
                        math.exp(-first_years / contracts.HORIZON_YEARS)
                    )
                    rank_terminal.append(bool(path.terminated[camp][0]))
                    for head in LABOR_VALUE_HEADS:
                        summary = summarize_labor_path(
                            path.rewards[camp][head],
                            path.years[camp],
                            path.terminated[camp],
                            path.schedule,
                            head=head,
                            prefix_terms=prefix_terms,
                            provenance_id=f"{group}/{path.alternative}/{head}",
                        )
                        rank_immediate[head].append(summary.immediate)
                        rank_realized[head].append(0.0 if summary.value is None else summary.value)
                        rank_errors[head].append(
                            0.0 if summary.standard_error is None else summary.standard_error
                        )
                        rank_masks[head].append(summary.available)


    if jobs == 1:
        for world_index, world in enumerate(worlds):
            bridge = bridge_factory()
            try:
                parent = _world_parent(bridge, world, reset, normalizer, personality, task, costs)
                runtime_identities[str(world)] = parent.runtime_identity
                fork, _ = bridge.fork()
                paths: list[LaborContinuationPath] = []
                try:
                    for path_index in range(path_count):
                        paths.extend(
                            _path_pair(
                                bridge, fork, actor=actor, task=task, costs=costs,
                                normalizer=normalizer, parent=parent, world=world,
                                world_index=world_index, schedule=schedules[(world, path_index)],
                                path_index=path_index, prefix_terms=prefix_terms,
                                progress=progress,
                            )
                        )
                finally:
                    bridge.restore(fork)
                    bridge.drop(fork)
            finally:
                bridge.close()
            assemble(world, parent, paths)
    else:
        units = _parallel_units(
            checkpoint_path=checkpoint_path,
            worlds=worlds,
            schedules=schedules,
            path_count=path_count,
            prefix_terms=prefix_terms,
            reset=reset,
            bridge_factory=bridge_factory,
            jobs=jobs,
            threads=threads,
            progress=progress,
            policy_digest=policy_digest,
            executor_factory=executor_factory,
        )
        for world in worlds:
            parent, identity, paths = _merge_world_units(world, units[world], path_count)
            runtime_identities[str(world)] = identity
            assemble(world, parent, paths)

    schedule_digest = digest(
        {
            f"{world}/{path}": {
                "id": schedule.schedule_id,
                "seed": schedule.rng_seed,
                "stops": list(schedule.stop_indices),
            }
            for (world, path), schedule in sorted(schedules.items())
        }
    )
    source = {
        "schemaVersion": 1,
        "kind": LABOR_ANCHOR_PRODUCER,
        "modelInputContract": LABOR_MODEL_INPUT_CONTRACT,
        "laborRegistryDigest": LABOR_REGISTRY_DIGEST,
        "criticBootstrapUsed": False,
        "truncatedTailAsZero": False,
        "returnEstimator": "fixed-prefix-plus-geometric-russian-roulette",
        "independentUnit": "world",
        "policyDigestSemantics": "effective-sampled-policy-digest-v1",
        "rewardSemantics": "labor-qol-population-and-two-cost-returns-v1",
        "terminationSemantics": "camp-terminal-after-recorded-physical-transition",
        "physicalDiscountSemantics": "exp(-years/horizonYears)",
        "horizonYears": contracts.HORIZON_YEARS,
        "internalRetryDiscount": 1.0,
        "internalRetryTraceContinuation": 1.0,
        "physicalTaskRewardEdges": 1,
        "continuationPane": "sampled-training-policy",
        "continuationSeedRoles": [
            "geometric-stopping-PCG64",
            "sampled-policy-initial-action-torch",
            "paired-sample-initial-action-torch",
            "paired-future-actions-torch",
        ],
        "sourceCheckpointSha256": checkpoint_hash(checkpoint_path),
        "nativeRuntimeIdentities": runtime_identities,
        "sourceActorTensorDigest": actor_tensor_digest,
        "sourcePolicyConfiguration": behavior,
        "sourcePolicyDigest": policy_digest,
        "normalizationBindingDigest": _normalization_digest(normalizer),
        "objectiveReferenceScales": dict(LABOR_OBJECTIVE_REFERENCE_SCALES),
        "costReferenceScales": dict(LABOR_COST_REFERENCE_SCALES),
        "randomizedTail": {
            "prefixTerms": prefix_terms,
            "drawCount": tail_draws,
            "survivalProbability": tail_rho,
            "hardCapTerms": tail_cap,
            "scheduleDigest": schedule_digest,
            "absoluteIntegrabilityEstablished": True,
            "finiteVarianceEstablished": True,
            "droppedDraws": 0,
        },
        "worldSeeds": worlds,
        "optimizerSeeds": [int(optimizer_seed)],
        "continuationSeeds": continuations,
        "independentUnits": [
            {
                "unitId": f"world-{world}",
                "worldSeed": world,
                "optimizerSeed": int(optimizer_seed),
                "continuationSeeds": [
                    seed
                    for path in range(path_count)
                    for seed in _path_rng_seeds(continuation_seed_base, index, path)
                ],
            }
            for index, world in enumerate(worlds)
        ],
    }
    if task.privileged_layout is not None:
        source["privilegedNormalizerDigest"] = task.privileged_normalizer.digest()
    metadata = {
        "schemaVersion": 1,
        "kind": LABOR_ANCHOR_KIND,
        "role": role,
        "valueHeads": list(LABOR_VALUE_HEADS),
        "rankSemantics": LABOR_RANK_SEMANTICS,
        "source": source,
        "scope": "local-independent-return-diagnostic-not-readiness",
        "resetOptions": reset.state_dict(),
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
        "row_id": np.asarray(row_ids),
        "independent_unit_id": np.asarray(unit_ids),
        **{f"input__{name}": value for name, value in _stack(parent_rows).items()},
    }
    if parent_privileged:
        arrays.update(_stack(parent_privileged))
    # A ranking group is usable only if the paired native successors differ.
    keep: list[int] = []
    for group in sorted(set(rank_groups)):
        members = [index for index, value in enumerate(rank_groups) if value == group]
        if len({rank_successors[index] for index in members}) > 1:
            keep.extend(members)
    if keep:
        arrays.update(
            {
                "rank_group_id": np.asarray([rank_groups[index] for index in keep]),
                "rank_parent_digest": np.asarray([rank_parents[index] for index in keep]),
                "rank_successor_digest": np.asarray([rank_successors[index] for index in keep]),
                "rank_independent_unit_id": np.asarray([rank_units[index] for index in keep]),
                "rank_discount": np.asarray([rank_discount[index] for index in keep]),
                "rank_terminal": np.asarray([rank_terminal[index] for index in keep]),
                **{
                    f"rank_input__{name}": value
                    for name, value in _stack([rank_inputs[index] for index in keep]).items()
                },
            }
        )
        if rank_privileged:
            arrays.update({
                "rank_" + key: value
                for key, value in _stack([rank_privileged[i] for i in keep]).items()
            })
    else:
        metadata["rankSemantics"] = "absent"
        arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
    for head in LABOR_VALUE_HEADS:
        arrays[f"target__{head}"] = np.asarray(targets[head], dtype=np.float64)
        arrays[f"target_mask__{head}"] = np.asarray(masks[head], dtype=bool)
        arrays[f"target_standard_error__{head}"] = np.asarray(errors[head], dtype=np.float64)
        arrays[f"target_method__{head}"] = np.asarray(methods[head])
        if keep:
            arrays[f"rank_immediate__{head}"] = np.asarray(
                [rank_immediate[head][index] for index in keep], dtype=np.float64
            )
            arrays[f"rank_realized__{head}"] = np.asarray(
                [rank_realized[head][index] for index in keep], dtype=np.float64
            )
            arrays[f"rank_standard_error__{head}"] = np.asarray(
                [rank_errors[head][index] for index in keep], dtype=np.float64
            )
            arrays[f"rank_mask__{head}"] = np.asarray(
                [rank_masks[head][index] for index in keep], dtype=bool
            )
    return arrays


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--worlds", required=True, help="comma-separated local diagnostic seeds")
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--continuation-seed-base", type=int, required=True)
    parser.add_argument("--role", choices=("train-anchor", "actor-gate"), required=True)
    parser.add_argument("--prefix-terms", type=int, default=DEFAULT_PREFIX_TERMS)
    parser.add_argument("--tail-draws", type=int, default=DEFAULT_TAIL_DRAWS)
    parser.add_argument("--tail-rho", type=float, default=DEFAULT_TAIL_RHO)
    parser.add_argument("--tail-cap", type=int, default=DEFAULT_TAIL_CAP)
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--allow-vm-scale", action="store_true", default=False)
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--separation", type=int, default=None)
    parser.add_argument("--no-movement", action="store_true", default=False)
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="parallel (world, path) units, one spawned process + bridge each "
        "(default 1: sequential, one bridge)",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="torch threads in this process and in every --jobs worker (default 4)",
    )
    parser.add_argument(
        "--progress", action="store_true", default=False,
        help="per-continuation wall-clock lines on stderr (telemetry only)",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from .labor_bc import LaborResetOptions

    args = argument_parser().parse_args(argv)
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    torch.set_num_threads(args.threads)
    worlds = [int(value) for value in args.worlds.split(",") if value]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    arrays = collect_labor_anchor_arrays(
        checkpoint_path=args.checkpoint,
        state=state,
        world_seeds=worlds,
        optimizer_seed=args.optimizer_seed,
        continuation_seed_base=args.continuation_seed_base,
        role=args.role,
        prefix_terms=args.prefix_terms,
        tail_draws=args.tail_draws,
        tail_rho=args.tail_rho,
        tail_cap=args.tail_cap,
        path_count=args.paths,
        allow_vm_scale=args.allow_vm_scale,
        reset_options=LaborResetOptions(
            players=args.players,
            movement=not args.no_movement,
            width=args.width,
            height=args.height,
            separation=args.separation,
        ),
        jobs=args.jobs,
        threads=args.threads,
        progress=args.progress,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)


if __name__ == "__main__":
    main()
