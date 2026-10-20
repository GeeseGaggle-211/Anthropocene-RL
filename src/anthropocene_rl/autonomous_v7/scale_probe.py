"""Scale probe: model9 bridge throughput at larger worlds and tribe counts, with every head native (the heuristic).

For each configuration it times world generation through the opening, then each 20-year turn: the native simulator
step, the bridge round trip (JSON), batching and a token-actor forward pass. It reports villages per turn and the
bridge's peak memory. The actor is freshly initialised: its forward cost does not depend on its weights.

    python -m anthropocene_rl.autonomous_v7.scale_probe --configs small:8,standard:32,standard:80 --turns 10 --out probe.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .bridge import Model9Bridge, observation_batch
from .model import Actor, Normalizer, to_tensors

# (width, height, start separation): Small is SimulationConfig.Small (training worlds); Standard is the game default.
SIZES = {"small": (96, 48, 9), "standard": (1024, 512, 12)}


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
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    handle = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    counters = Counters(); counters.cb = ctypes.sizeof(Counters)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    ctypes.windll.kernel32.CloseHandle(handle)
    return counters.PeakWorkingSetSize / 2 ** 20 if ok else float("nan")


def probe(size: str, players: int, seed: int, turns: int) -> dict:
    torch.set_num_threads(1)
    with Model9Bridge() as bridge:
        clock = time.perf_counter()
        step = bridge.reset(seed, players=players, size=SIZES[size])
        generation = time.perf_counter() - clock
        actor = normalizer = None
        rows = []
        for turn in range(turns):
            live = [v for v in step["villages"] if v.get("observation") is not None and not v["terminated"]]
            forward = batching = 0.0
            if live:
                clock = time.perf_counter()
                batch = observation_batch(live)
                batching = time.perf_counter() - clock
                if actor is None:
                    normalizer = Normalizer.fit(batch)
                    torch.manual_seed(0); actor = Actor(bridge.schema.core_width, token=True); actor.eval()
                clock = time.perf_counter()
                with torch.no_grad():
                    actor(to_tensors(batch, normalizer))
                forward = time.perf_counter() - clock
            clock = time.perf_counter()
            step = bridge.step({})  # every village native
            round_trip = time.perf_counter() - clock
            native = float(step.get("nativeMs", 0.0)) / 1000.0
            rows.append({"turn": turn, "villages": len(live), "native_s": native, "bridge_s": round_trip,
                         "overhead_s": max(0.0, round_trip - native), "batch_s": batching, "forward_s": forward,
                         "population": int(sum(v["population"] for v in step["villages"]))})
            print(json.dumps({"size": size, "players": players, **rows[-1]}), flush=True)
            if not step["villages"]:
                break
        memory = peak_memory_mib(bridge._process.pid)
    villages = [r["villages"] for r in rows]
    per_turn = [r["bridge_s"] + r["batch_s"] + r["forward_s"] for r in rows]
    return {"size": size, "players": players, "seed": seed, "turns": len(rows), "generation_s": generation,
            "villages_mean": float(np.mean(villages)), "villages_max": int(max(villages)),
            "turn_s_mean": float(np.mean(per_turn)), "native_s_mean": float(np.mean([r["native_s"] for r in rows])),
            "overhead_s_mean": float(np.mean([r["overhead_s"] for r in rows])),
            "forward_s_mean": float(np.mean([r["forward_s"] for r in rows])),
            "ms_per_village_decision": 1000.0 * sum(per_turn) / max(1, sum(villages)),
            "bridge_peak_mib": memory, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="small:8,standard:32,standard:80", help="size:players[,size:players]")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
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
