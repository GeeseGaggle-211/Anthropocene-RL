"""Reference scales for Elder personalities (docs/MODEL9_PERSONALITY.md, stable scales).

A personality weights objective advantages after dividing each by a fixed reference spread s_k_ref: the standard
deviation of objective k's discounted return per decision under the native heuristic. This tool measures them: it plays
the heuristic (no actions submitted) on TRAIN worlds through the bridge, records one row per village decision with the
fields ``data.reward_components`` and ``data.discount`` read, and computes each (world, camp) sequence's discounted
return per objective with the physical per-turn discount exp(-years / HORIZON_YEARS) that ``ppo.gae`` uses. Returns are
truncated at the rollout end (no critic: the heuristic has none), so the stored scales use only full-window rows. The
measured values are stored in
``contracts.OBJECTIVE_REFERENCE_SCALES``.

    python -m anthropocene_rl.autonomous_v7.scale_reference --seeds 1000-1059 --turns 45 --full-window 20 --out <json>

The stored values are the ``full_window_std`` of that run: only rows with at least 20 later turns (the first 25 turns),
so no return is cut short by the rollout end. The first measurement (25 turns, all rows) truncated late returns.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from . import contracts, data
from .bridge import Model9Bridge

ROW_KEYS = (
    "world",
    "camp",
    "turn",
    "society",
    "parent_society",
    "prior_population",
    "population",
    "qol",
    "years",
    "terminated",
    "trade_utility",
)


def heuristic_rows(seed: int, turns: int, withheld: int = 0) -> dict[str, list]:
    """One heuristic rollout: a row per village decision (a live village with an observation before the step), with
    the outcome the step reported for it (the layout ``ppo.collect`` records)."""
    rows: dict[str, list] = {k: [] for k in ROW_KEYS}
    with Model9Bridge() as bridge:
        step = bridge.reset(seed, withheld=withheld)
        for _ in range(turns):
            deciding = {
                int(v["camp"])
                for v in step["villages"]
                if v.get("observation") is not None and not v.get("terminated")
            }
            if not deciding:
                break
            turn = step["turn"]
            step = bridge.step({})
            for v in step["villages"]:
                if int(v["camp"]) not in deciding:
                    continue
                rows["world"].append(seed)
                rows["camp"].append(int(v["camp"]))
                rows["turn"].append(int(turn))
                rows["society"].append(int(v["society"]))
                rows["parent_society"].append(int(v["parentSociety"]))
                rows["prior_population"].append(float(v["priorPopulation"]))
                rows["population"].append(float(v["population"]))
                rows["qol"].append(float(v["qol"]))
                rows["years"].append(float(v["years"]))
                rows["terminated"].append(bool(v["terminated"]))
                rows["trade_utility"].append(float(v.get("tradeUtility", 0.0)))
    return rows


def discounted_returns(batch: dict[str, np.ndarray], reward: np.ndarray) -> np.ndarray:
    """G_i = r_i + gamma_i G_{i+1} per (world, camp) sequence; zero after termination and at the rollout end."""
    out = np.zeros(len(reward), dtype=np.float64)
    for rows in data.sequences(batch):
        g = 0.0
        for i in rows[::-1]:
            g = reward[i] + (
                0.0 if batch["terminated"][i] else float(data.discount(batch["years"][i])) * g
            )
            out[i] = g
    return out


def later_turns(batch: dict[str, np.ndarray]) -> np.ndarray:
    """Decision turns left in each row's world rollout after the row (the world's last decision turn minus the row's).
    A row with ``n`` later turns has a return covering ``n + 1`` turns unless its village ended first, and then the
    return is complete anyway."""
    world = np.asarray(batch["world"])
    turn = np.asarray(batch["turn"])
    last = {w: int(turn[world == w].max()) for w in np.unique(world)}
    return np.asarray([last[w] - t for w, t in zip(world, turn, strict=False)])


def reference_scales(batch: dict[str, np.ndarray], full_window: int = 0) -> dict[str, Any]:
    """Each objective's return std and mean over every decision row; ``full_window`` > 0 also reports the rows with at
    least that many later turns in their world's rollout (``later_turns``): with 20, every such return covers a full
    400-year-equivalent window (the stored reference scales, round 3 revision)."""
    components = data.reward_components(batch)
    report: dict[str, Any] = {"rows": len(batch["camp"]), "objectives": {}}
    remaining = later_turns(batch)
    for name in contracts.OBJECTIVES:
        g = discounted_returns(batch, components[name])
        entry = {"std": float(np.std(g)), "mean": float(np.mean(g))}
        if full_window > 0 and (remaining >= full_window).any():
            sub = g[remaining >= full_window]
            entry.update(
                full_window_std=float(np.std(sub)),
                full_window_mean=float(np.mean(sub)),
                full_window_rows=len(sub),
            )
        report["objectives"][name] = entry
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1000-1059", help="TRAIN seed range")
    parser.add_argument("--turns", type=int, default=45)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--full-window",
        type=int,
        default=20,
        help="also report rows with at least this many later turns (20 turns = 400 years); with "
        "--turns 45 these are the first 25 turns, the stored reference scales",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a, b = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(a, b + 1))
    clock = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        worlds = list(pool.map(lambda s: heuristic_rows(s, args.turns), seeds))
    batch = {k: np.concatenate([np.asarray(w[k]) for w in worlds]) for k in ROW_KEYS}
    report = reference_scales(batch, args.full_window)
    report.update(
        {
            "seeds": args.seeds,
            "turns": args.turns,
            "horizon_years": contracts.HORIZON_YEARS,
            "worlds": len(seeds),
            "seconds": time.time() - clock,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
