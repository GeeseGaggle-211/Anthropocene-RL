"""Personality sweep (docs/MODEL9_PERSONALITY.md): one preference-conditioned checkpoint evaluated at a grid of Elder
personalities on the same worlds, reporting each point's population, QoL, welfare and trade, so the frontier can be
checked for monotonicity (more QoL weight never lowers QoL).

    python -m anthropocene_rl.autonomous_v7.personality_sweep --checkpoint <pt> --seeds 2000-2019 --out <json>
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import contracts
from .evaluate import rollout

# welfare, population_years, trade_utility, qol_years
GRID = {
    "welfare": [1, 0, 0, 0],
    "population": [0, 1, 0, 0],
    "qol": [0, 0, 0, 1],
    "balanced": [0.25, 0.25, 0.25, 0.25],
    "trader": [0.5, 0, 0.5, 0],
    "qol-leaning": [0.5, 0, 0, 0.5],
    "pop-leaning": [0.5, 0.5, 0, 0],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", default="2000-2019")
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--movement", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert len(next(iter(GRID.values()))) == contracts.PERSONALITY_WIDTH
    a, b = (int(x) for x in args.seeds.split("-")); seeds = list(range(a, b + 1))
    report = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for name, p in GRID.items():
            n = len(seeds)
            rows = list(pool.map(rollout, seeds, [args.turns] * n, [args.checkpoint] * n, [0] * n, [args.movement] * n,
                                 [None] * n, [True if args.movement else None] * n, [0] * n, [None] * n, [None] * n, [p] * n))
            population = sum(r["population"] for r in rows)
            report[name] = {"personality": p, "welfare": sum(r["welfare"] for r in rows), "population": population,
                            "qol": sum(r["qol"] * r["population"] for r in rows) / max(1, population),
                            "trade_utility": sum(r["trade_utility"] for r in rows), "extinctions": sum(r["extinctions"] for r in rows)}
            print(json.dumps({name: report[name]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
