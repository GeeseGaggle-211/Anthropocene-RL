"""Daughter-start evaluation (docs/MODEL9_PERSONALITY.md, daughter bank): how a checkpoint treats young daughters.

From each held-out daughter-bank snapshot (``daughter_bank.held_out``: the held-out world-seed family, never drawn
for training; the partition is checked for shared snapshots) the world runs
``turns`` turns twice: every village on the checkpoint's deterministic policy, and every village on the native
heuristic. Both start from the same bytes, so each entry is a paired comparison. Reported per entry and in aggregate:
the daughter society's survival (any of its settlements alive at the end), its population growth (final residents over
the daughter camp's residents at the snapshot) and its population-weighted final QoL.

    python -m anthropocene_rl.autonomous_v7.daughter_eval --checkpoint <pt> --bank <dir> --turns 5 --out <json>
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bridge import Model9Bridge
from .daughter_bank import DaughterBank, held_out
from .evaluate import load, policy_actions


def daughter_outcome(villages: list[dict[str, Any]], society: int) -> dict[str, float]:
    own = [v for v in villages if int(v["society"]) == society and not v["terminated"] and v["population"] > 0]
    people = sum(v["population"] for v in own)
    return {"alive": float(bool(own)), "population": float(people),
            "qol": float(sum(v["qol"] * v["population"] for v in own) / max(1, people)), "settlements": float(len(own))}


def run_entry(bank: str, entry_id: int, checkpoint: str | None, turns: int, movement: bool | None = None,
              personality: list[float] | None = None) -> dict[str, Any]:
    """One start, played by ``checkpoint`` (None: the heuristic) for ``turns`` turns."""
    torch.set_num_threads(1)
    store = DaughterBank(bank); entry = store.get(entry_id)
    if not held_out(entry):
        raise ValueError(f"daughter_eval uses only the held-out family; entry {entry_id} (seed {entry['seed']}) trains")
    move = bool(entry.get("movement", False)) if movement is None else movement
    actor = normalizer = None
    if checkpoint is not None:
        actor, normalizer = load(Path(checkpoint))
    society, camp = int(entry["society"]), int(entry["camp"])
    with Model9Bridge() as bridge:
        step = bridge.reset(int(entry["seed"]), withheld=int(entry["mask"]), movement=move, snapshot=store.snapshot(entry))
        start = next((v for v in step["villages"] if int(v["camp"]) == camp), None)
        initial = float(start["population"]) if start is not None else 0.0
        for _ in range(turns):
            actions = (policy_actions(actor, normalizer, step["villages"], move and getattr(actor, "region_movement", False),
                                      personality=personality) if actor is not None else {})
            step = bridge.step(actions)
            if not any(not v["terminated"] for v in step["villages"]):
                break
    out = daughter_outcome(step["villages"], society)
    out.update({"id": int(entry_id), "initial_population": initial, "growth": out["population"] / max(1.0, initial)})
    return out


def paired(policy: list[dict[str, Any]], heuristic: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired outcomes: means for each side and the mean paired difference (policy minus heuristic)."""
    report: dict[str, Any] = {"entries": len(policy)}
    for key in ("alive", "growth", "qol", "population"):
        p = np.asarray([r[key] for r in policy]); h = np.asarray([r[key] for r in heuristic])
        report[key] = {"policy": float(p.mean()) if len(p) else 0.0, "heuristic": float(h.mean()) if len(h) else 0.0,
                       "difference": float((p - h).mean()) if len(p) else 0.0,
                       "difference_se": float((p - h).std(ddof=1) / np.sqrt(len(p))) if len(p) > 1 else 0.0}
    # QoL among daughters alive on both sides (a dead daughter's QoL is 0 and would mix survival into QoL).
    both = [(a["qol"], b["qol"]) for a, b in zip(policy, heuristic) if a["alive"] and b["alive"]]
    report["qol_both_alive"] = {"entries": len(both), "difference": float(np.mean([a - b for a, b in both])) if both else 0.0}
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="evaluate at most this many held-out entries (0: all)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--personality", default=None, help="w,w,w,w for every village (personality checkpoints)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    entries = DaughterBank(args.bank).evaluation_entries()  # held-out family only (asserts the partition)
    if args.limit > 0:
        entries = entries[:args.limit]
    ids = [int(e["id"]) for e in entries]
    p = [float(x) for x in args.personality.split(",")] if args.personality else None
    n = len(ids); clock = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, n))) as pool:
        policy = list(pool.map(run_entry, [str(args.bank)] * n, ids, [args.checkpoint] * n, [args.turns] * n, [None] * n, [p] * n))
        heuristic = list(pool.map(run_entry, [str(args.bank)] * n, ids, [None] * n, [args.turns] * n))
    report = {"checkpoint": args.checkpoint, "bank": str(args.bank), "turns": args.turns, **paired(policy, heuristic),
              "per_entry": [{"id": i, "policy": a, "heuristic": b} for i, a, b in zip(ids, policy, heuristic)],
              "seconds": time.time() - clock}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_entry"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
