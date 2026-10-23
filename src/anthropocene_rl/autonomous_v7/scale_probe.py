"""Probe model9 bridge throughput with every control head kept native.

The legacy mode times world generation, native turns, JSON round trips,
batching, and a freshly initialized token-actor forward pass. The paired mode
measures opt-in economic diagnostics against the same native background.

    python -m anthropocene_rl.autonomous_v7.scale_probe \
      --configs small:8,standard:32,standard:80 --turns 10 --out probe.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bridge import Model9Bridge, dotnet_command, observation_batch
from .model import Actor, Normalizer, to_tensors

# (width, height, start separation). Small is the training world; Standard is the game default.
SIZES = {"small": (96, 48, 9), "standard": (1024, 512, 12)}

PAIRED_WARMUP_REPETITIONS = 2
PAIRED_MEASURED_REPETITIONS = 7
PAIRED_RSS_LIMIT_MIB = 256.0
DIAGNOSTIC_PROFILES = (
    {
        "id": "diagnostic-on-current",
        "seeds": (50000,),
        "players": 8,
        "expectedResetTurn": 9,
        "worldWarmupSteps": 0,
        "horizon": 25,
        "workers": 1,
        "metric": "step-time-ratio",
        "direction": "<=",
        "limit": 1.15,
    },
    {
        "id": "diagnostic-on-mature",
        "seeds": (50001,),
        "players": 8,
        "expectedResetTurn": 9,
        "worldWarmupSteps": 25,
        "horizon": 25,
        "workers": 1,
        "metric": "step-time-ratio",
        "direction": "<=",
        "limit": 1.20,
    },
    {
        "id": "eight-worker-end-to-end",
        "seeds": tuple(range(50000, 50008)),
        "players": 8,
        "expectedResetTurn": 9,
        "worldWarmupSteps": 0,
        "horizon": 25,
        "workers": 8,
        "metric": "throughput-ratio",
        "direction": ">=",
        "limit": 0.80,
    },
)


def peak_memory_mib(pid: int) -> float:
    """Peak working set of a process in MiB (Windows; NaN elsewhere)."""
    if sys.platform != "win32":
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            pass
        return float("nan")

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    handle = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    ctypes.windll.kernel32.CloseHandle(handle)
    return counters.PeakWorkingSetSize / 2**20 if ok else float("nan")


def probe(size: str, players: int, seed: int, turns: int) -> dict:
    torch.set_num_threads(1)
    with Model9Bridge() as bridge:
        clock = time.perf_counter()
        step = bridge.reset(seed, players=players, size=SIZES[size])
        generation = time.perf_counter() - clock
        actor = normalizer = None
        rows = []
        for turn in range(turns):
            live = [
                v
                for v in step["villages"]
                if v.get("observation") is not None and not v["terminated"]
            ]
            forward = batching = 0.0
            if live:
                clock = time.perf_counter()
                batch = observation_batch(live)
                batching = time.perf_counter() - clock
                if actor is None:
                    normalizer = Normalizer.fit(batch)
                    torch.manual_seed(0)
                    actor = Actor(bridge.schema.core_width, token=True)
                    actor.eval()
                clock = time.perf_counter()
                with torch.no_grad():
                    actor(to_tensors(batch, normalizer))
                forward = time.perf_counter() - clock
            clock = time.perf_counter()
            step = bridge.step({})  # every village native
            round_trip = time.perf_counter() - clock
            native = float(step.get("nativeMs", 0.0)) / 1000.0
            rows.append(
                {
                    "turn": turn,
                    "villages": len(live),
                    "native_s": native,
                    "bridge_s": round_trip,
                    "overhead_s": max(0.0, round_trip - native),
                    "batch_s": batching,
                    "forward_s": forward,
                    "population": int(sum(v["population"] for v in step["villages"])),
                }
            )
            print(json.dumps({"size": size, "players": players, **rows[-1]}), flush=True)
            if not step["villages"]:
                break
        memory = peak_memory_mib(bridge._process.pid)
    villages = [r["villages"] for r in rows]
    per_turn = [r["bridge_s"] + r["batch_s"] + r["forward_s"] for r in rows]
    return {
        "size": size,
        "players": players,
        "seed": seed,
        "turns": len(rows),
        "generation_s": generation,
        "villages_mean": float(np.mean(villages)),
        "villages_max": int(max(villages)),
        "turn_s_mean": float(np.mean(per_turn)),
        "native_s_mean": float(np.mean([r["native_s"] for r in rows])),
        "overhead_s_mean": float(np.mean([r["overhead_s"] for r in rows])),
        "forward_s_mean": float(np.mean([r["forward_s"] for r in rows])),
        "ms_per_village_decision": 1000.0 * sum(per_turn) / max(1, sum(villages)),
        "bridge_peak_mib": memory,
        "rows": rows,
    }


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _boundary_trace(step: dict[str, Any]) -> dict[str, Any]:
    villages = sorted(step["villages"], key=lambda row: int(row["camp"]))
    observations = [
        {"camp": int(village["camp"]), "observation": village.get("observation")}
        for village in villages
    ]
    rewards = []
    for village in villages:
        reward = {"camp": int(village["camp"])}
        for field in ("reward", "welfare", "tradeUtility"):
            if field in village:
                reward[field] = village[field]
        rewards.append(reward)
    return {
        "fingerprint": str(step["fingerprint"]),
        "observationsSha256": _sha256_json(observations),
        "rewardsSha256": _sha256_json(rewards),
    }


def _trace_receipt(
    boundaries: Sequence[dict[str, Any]], actions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    receipt = {
        "fingerprints": [row["fingerprint"] for row in boundaries],
        "observationsSha256": [row["observationsSha256"] for row in boundaries],
        "rewardsSha256": [row["rewardsSha256"] for row in boundaries],
        "actionsSha256": [_sha256_json(row) for row in actions],
    }
    receipt["combinedSha256"] = _sha256_json(receipt)
    return receipt


def _native_world(
    *,
    seed: int,
    players: int,
    expected_reset_turn: int,
    world_warmup_steps: int,
    horizon: int,
    economic_diagnostics: bool,
    command: Sequence[str] | None,
) -> dict[str, Any]:
    """Run one frozen native-background world and retain raw timing/parity receipts."""

    process_start = time.perf_counter()
    with Model9Bridge(command=command, economic_diagnostics=economic_diagnostics) as bridge:
        reset_start = time.perf_counter()
        step = bridge.reset(seed, players=players)
        reset_seconds = time.perf_counter() - reset_start
        if int(step["turn"]) != expected_reset_turn:
            raise ValueError(
                f"seed {seed} reset turn {step['turn']} differs from {expected_reset_turn}"
            )
        boundaries = [_boundary_trace(step)]
        action_trace: list[dict[str, Any]] = []
        warmup_seconds: list[float] = []
        measured_seconds: list[float] = []
        terminal_during_warmup = False
        for index in range(world_warmup_steps):
            actions: dict[int, dict[str, Any]] = {}
            action_trace.append({"phase": "world-warmup", "index": index, "actions": actions})
            clock = time.perf_counter()
            step = bridge.step(actions)
            warmup_seconds.append(time.perf_counter() - clock)
            boundaries.append(_boundary_trace(step))
            if not step["villages"]:
                terminal_during_warmup = True
                break
        if not terminal_during_warmup:
            for index in range(horizon):
                actions = {}
                action_trace.append({"phase": "measured", "index": index, "actions": actions})
                clock = time.perf_counter()
                step = bridge.step(actions)
                measured_seconds.append(time.perf_counter() - clock)
                boundaries.append(_boundary_trace(step))
                if not step["villages"]:
                    break
        peak_mib = peak_memory_mib(bridge._process.pid)
    end_to_end_seconds = time.perf_counter() - process_start
    rss_mib = None if math.isnan(peak_mib) else float(peak_mib)
    return {
        "seed": seed,
        "economicDiagnostics": economic_diagnostics,
        "background": "native-heuristic-all-camps",
        "players": players,
        "resetTurn": expected_reset_turn,
        "worldWarmupStepsRequested": world_warmup_steps,
        "worldWarmupStepsCompleted": len(warmup_seconds),
        "horizonRequested": horizon,
        "measuredStepsCompleted": len(measured_seconds),
        "resetSeconds": reset_seconds,
        "worldWarmupStepSeconds": warmup_seconds,
        "measuredStepSeconds": measured_seconds,
        "measuredStepSecondsMedian": (
            statistics.median(measured_seconds) if measured_seconds else None
        ),
        "measuredStepSecondsTotal": sum(measured_seconds),
        "endToEndSeconds": end_to_end_seconds,
        "workerPeakRssMiB": rss_mib,
        "workerPeakRssStatus": "available" if rss_mib is not None else "unavailable",
        "trace": _trace_receipt(boundaries, action_trace),
    }


WorldRunner = Callable[..., dict[str, Any]]


def _execution_summary(
    profile: dict[str, Any], workers: Sequence[dict[str, Any]], wall_seconds: float
) -> dict[str, Any]:
    ordered = sorted(workers, key=lambda row: int(row["seed"]))
    rss_by_worker = [
        {
            "seed": int(row["seed"]),
            "peakRssMiB": row["workerPeakRssMiB"],
            "status": row["workerPeakRssStatus"],
        }
        for row in ordered
    ]
    rss_available = all(row["peakRssMiB"] is not None for row in rss_by_worker)
    rss_sum = sum(float(row["peakRssMiB"]) for row in rss_by_worker) if rss_available else None
    step_seconds = [float(seconds) for row in ordered for seconds in row["measuredStepSeconds"]]
    completed_steps = sum(int(row["measuredStepsCompleted"]) for row in ordered)
    return {
        "workers": len(ordered),
        "completedWorlds": len(ordered),
        "completedSteps": completed_steps,
        "wallSeconds": wall_seconds,
        "worldsPerSecond": len(ordered) / wall_seconds if wall_seconds > 0 else None,
        "stepsPerSecond": completed_steps / wall_seconds if wall_seconds > 0 else None,
        "measuredStepSecondsMedian": (statistics.median(step_seconds) if step_seconds else None),
        "workerPeakRss": rss_by_worker,
        "workerPeakRssSumMiB": rss_sum,
        "workerPeakRssSumStatus": "available" if rss_available else "unavailable",
        "traceBySeed": {str(row["seed"]): row["trace"] for row in ordered},
        "rawWorkers": ordered,
        "profileId": profile["id"],
    }


def _execute_profile(
    profile: dict[str, Any],
    *,
    economic_diagnostics: bool,
    command: Sequence[str] | None,
    world_runner: WorldRunner = _native_world,
) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in profile["seeds"])
    if len(seeds) != int(profile["workers"]):
        raise ValueError("diagnostic profile seed count differs from worker count")

    def run(seed: int) -> dict[str, Any]:
        return world_runner(
            seed=seed,
            players=int(profile["players"]),
            expected_reset_turn=int(profile["expectedResetTurn"]),
            world_warmup_steps=int(profile["worldWarmupSteps"]),
            horizon=int(profile["horizon"]),
            economic_diagnostics=economic_diagnostics,
            command=command,
        )

    clock = time.perf_counter()
    if len(seeds) == 1:
        workers = [run(seeds[0])]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as executor:
            workers = list(executor.map(run, seeds))
    wall_seconds = time.perf_counter() - clock
    return _execution_summary(profile, workers, wall_seconds)


def _parity(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    baseline_traces = baseline["traceBySeed"]
    treatment_traces = treatment["traceBySeed"]
    same_seeds = set(baseline_traces) == set(treatment_traces)
    categories = {
        "fingerprintsExact": "fingerprints",
        "observationsExact": "observationsSha256",
        "rewardsExact": "rewardsSha256",
        "actionTraceExact": "actionsSha256",
    }
    result: dict[str, Any] = {"seedSetExact": same_seeds}
    for output_name, trace_name in categories.items():
        result[output_name] = same_seeds and all(
            baseline_traces[seed][trace_name] == treatment_traces[seed][trace_name]
            for seed in baseline_traces
        )
    result["exact"] = all(result.values())
    return result


def _ratio(
    profile: dict[str, Any], baseline: dict[str, Any], treatment: dict[str, Any]
) -> float | None:
    if profile["metric"] == "step-time-ratio":
        baseline_value = baseline["measuredStepSecondsMedian"]
        treatment_value = treatment["measuredStepSecondsMedian"]
        if baseline_value is None or treatment_value is None or baseline_value <= 0:
            return None
        return float(treatment_value) / float(baseline_value)
    if profile["metric"] == "throughput-ratio":
        baseline_value = baseline["worldsPerSecond"]
        treatment_value = treatment["worldsPerSecond"]
        if baseline_value is None or treatment_value is None or baseline_value <= 0:
            return None
        return float(treatment_value) / float(baseline_value)
    raise ValueError(f"unsupported diagnostic profile metric {profile['metric']}")


def _threshold_met(value: float | None, direction: str, limit: float) -> bool | None:
    if value is None:
        return None
    if direction == "<=":
        return value <= limit
    if direction == ">=":
        return value >= limit
    raise ValueError(f"unsupported threshold direction {direction}")


def run_paired_profile(
    profile: dict[str, Any],
    *,
    baseline: dict[str, Any],
    treatment: dict[str, Any],
    execute: Callable[..., dict[str, Any]] = _execute_profile,
) -> dict[str, Any]:
    """Run the frozen alternating-pair protocol and evaluate only paired evidence."""

    warmup_pairs: list[dict[str, Any]] = []
    measured_pairs: list[dict[str, Any]] = []

    def one_pair(index: int, *, measured: bool) -> dict[str, Any]:
        baseline_first = index % 2 == 0
        order = (
            (("baseline", baseline), ("treatment", treatment))
            if baseline_first
            else (("treatment", treatment), ("baseline", baseline))
        )
        results: dict[str, Any] = {}
        for name, variant in order:
            results[name] = execute(
                profile,
                economic_diagnostics=bool(variant["economicDiagnostics"]),
                command=variant.get("command"),
            )
        baseline_result = results["baseline"]
        treatment_result = results["treatment"]
        baseline_rss = baseline_result["workerPeakRssSumMiB"]
        treatment_rss = treatment_result["workerPeakRssSumMiB"]
        rss_delta = (
            None
            if baseline_rss is None or treatment_rss is None
            else float(treatment_rss) - float(baseline_rss)
        )
        return {
            "index": index,
            "measured": measured,
            "executionOrder": [name for name, _ in order],
            "baseline": baseline_result,
            "treatment": treatment_result,
            "ratio": _ratio(profile, baseline_result, treatment_result),
            "parity": _parity(baseline_result, treatment_result),
            "pairedIncrementalPeakWorkerRssMiB": rss_delta,
            "rssStatus": "available" if rss_delta is not None else "unavailable",
        }

    for index in range(PAIRED_WARMUP_REPETITIONS):
        warmup_pairs.append(one_pair(index, measured=False))
    for index in range(PAIRED_MEASURED_REPETITIONS):
        measured_pairs.append(one_pair(index, measured=True))

    ratios = [row["ratio"] for row in measured_pairs]
    median_ratio = (
        statistics.median(float(value) for value in ratios if value is not None)
        if ratios and all(value is not None for value in ratios)
        else None
    )
    rss_values = [row["pairedIncrementalPeakWorkerRssMiB"] for row in measured_pairs]
    maximum_rss_delta = (
        max(float(value) for value in rss_values if value is not None)
        if rss_values and all(value is not None for value in rss_values)
        else None
    )
    ratio_met = _threshold_met(median_ratio, str(profile["direction"]), float(profile["limit"]))
    rss_met = None if maximum_rss_delta is None else maximum_rss_delta <= PAIRED_RSS_LIMIT_MIB
    parity_exact = all(row["parity"]["exact"] for row in [*warmup_pairs, *measured_pairs])
    if not parity_exact or ratio_met is False or rss_met is False:
        acceptance_status = "threshold-not-met"
    elif ratio_met is None or rss_met is None:
        acceptance_status = "unavailable"
    else:
        acceptance_status = "thresholds-met"
    return {
        "id": profile["id"],
        "fixture": {
            "seeds": list(profile["seeds"]),
            "players": profile["players"],
            "expectedResetTurn": profile["expectedResetTurn"],
            "worldWarmupSteps": profile["worldWarmupSteps"],
            "horizon": profile["horizon"],
            "workers": profile["workers"],
            "background": "native-heuristic-all-camps",
        },
        "metric": profile["metric"],
        "ratioDefinition": (
            "treatment median measured step time / baseline median measured step time"
            if profile["metric"] == "step-time-ratio"
            else "treatment same-world-set throughput / baseline same-world-set throughput"
        ),
        "rule": {"direction": profile["direction"], "limit": profile["limit"]},
        "warmupPairs": warmup_pairs,
        "measuredPairs": measured_pairs,
        "medianOfPerPairRatios": median_ratio,
        "ratioThresholdMet": ratio_met,
        "parityExact": parity_exact,
        "maximumPairedIncrementalPeakWorkerRssMiB": maximum_rss_delta,
        "rssLimitMiB": PAIRED_RSS_LIMIT_MIB,
        "rssThresholdMet": rss_met,
        "acceptanceStatus": acceptance_status,
    }


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _producer_identity(command: Sequence[str] | None) -> dict[str, Any]:
    actual = list(command or dotnet_command())
    bridge = _file_identity(Path(actual[-1]))
    directory = Path(bridge["path"]).parent
    dependencies = []
    for suffix in (".deps.json", ".runtimeconfig.json"):
        candidate = directory / f"{Path(bridge['path']).stem}{suffix}"
        if candidate.is_file():
            dependencies.append(_file_identity(candidate))
    return {"command": actual, "bridge": bridge, "dependencies": dependencies}


def paired_diagnostics_probe(
    *,
    current_command: Sequence[str] | None = None,
    old_baseline_command: Sequence[str] | None = None,
    execute: Callable[..., dict[str, Any]] = _execute_profile,
) -> dict[str, Any]:
    baseline = {
        "id": "diagnostics-off",
        "economicDiagnostics": False,
        "command": current_command,
    }
    treatment = {
        "id": "diagnostics-on",
        "economicDiagnostics": True,
        "command": current_command,
    }
    profiles = [
        run_paired_profile(dict(profile), baseline=baseline, treatment=treatment, execute=execute)
        for profile in DIAGNOSTIC_PROFILES
    ]
    if old_baseline_command is None:
        old_baseline: dict[str, Any] = {
            "status": "unavailable",
            "reason": "no reviewed pre-change bridge binary was supplied",
            "requiredRatio": 0.98,
            "measuredValue": None,
        }
    else:
        old_profile = {
            **dict(DIAGNOSTIC_PROFILES[0]),
            "id": "diagnostic-off-current-vs-old-baseline",
            "metric": "throughput-ratio",
            "direction": ">=",
            "limit": 0.98,
        }
        old_baseline = {
            "status": "measured",
            "producer": _producer_identity(old_baseline_command),
            "profile": run_paired_profile(
                old_profile,
                baseline={
                    "id": "old-binary-diagnostics-off",
                    "economicDiagnostics": False,
                    "command": old_baseline_command,
                },
                treatment=baseline,
                execute=execute,
            ),
        }
    return {
        "schemaVersion": 1,
        "kind": "model9-paired-economic-diagnostics-scale-probe",
        "interpretation": (
            "relative diagnostic overhead and exact replay parity only; "
            "not a training, readiness, or production-outcome claim"
        ),
        "protocol": {
            "warmupPairs": PAIRED_WARMUP_REPETITIONS,
            "measuredPairs": PAIRED_MEASURED_REPETITIONS,
            "measuredPairOrder": [
                "baseline-treatment" if index % 2 == 0 else "treatment-baseline"
                for index in range(PAIRED_MEASURED_REPETITIONS)
            ],
            "aggregation": "median-of-per-pair-ratios",
            "rssAggregation": "maximum-of-paired-incremental-peak-worker-rss",
            "missingRss": "unavailable-not-zero",
            "eightWorkerComparison": (
                "same-eight-world diagnostics-on throughput divided by "
                "same-eight-world diagnostics-off throughput"
            ),
        },
        "currentProducer": _producer_identity(current_command),
        "harness": _file_identity(Path(__file__)),
        "profiles": profiles,
        "oldBaseline": old_baseline,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", default="small:8,standard:32,standard:80", help="size:players[,size:players]"
    )
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument(
        "--paired-diagnostics",
        action="store_true",
        help="run the frozen paired diagnostic on/off performance and parity protocol",
    )
    parser.add_argument(
        "--current-bridge",
        type=Path,
        help="explicit current bridge DLL; otherwise use ANTHROPOCENE_MODEL9_BRIDGE/default",
    )
    parser.add_argument(
        "--old-baseline-bridge",
        type=Path,
        help="reviewed pre-change bridge DLL; omission is reported as unavailable",
    )
    parser.add_argument("--out", type=Path, required=True)
    from .checkpoint import parse

    args = parse(parser)
    if args.paired_diagnostics:
        dotnet = dotnet_command()[0]
        current_command = (
            [dotnet, str(args.current_bridge.resolve())]
            if args.current_bridge is not None
            else None
        )
        old_baseline_command = (
            [dotnet, str(args.old_baseline_bridge.resolve())]
            if args.old_baseline_bridge is not None
            else None
        )
        result = paired_diagnostics_probe(
            current_command=current_command,
            old_baseline_command=old_baseline_command,
        )
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(
            json.dumps(
                {
                    "kind": result["kind"],
                    "profiles": [
                        {
                            "id": row["id"],
                            "acceptanceStatus": row["acceptanceStatus"],
                            "medianOfPerPairRatios": row["medianOfPerPairRatios"],
                        }
                        for row in result["profiles"]
                    ],
                    "oldBaselineStatus": result["oldBaseline"]["status"],
                }
            ),
            flush=True,
        )
        return 0
    report = []
    for item in args.configs.split(","):
        size, players = item.split(":")
        result = probe(size, int(players), args.seed, args.turns)
        report.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
        args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
