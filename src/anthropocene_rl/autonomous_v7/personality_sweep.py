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
    parser.add_argument("--seeds", default="2000-2059")
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--movement", action="store_true")
    parser.add_argument("--grouped-region", action="store_true", help="hierarchical region choice (model.grouped_region_slot)")
    parser.add_argument("--focal", action="store_true",
                        help="only alternate founding lineages take the sweep point; the rest draw Dirichlet(1) as in "
                             "training, and the focal lineages' outcomes are reported")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert len(next(iter(GRID.values()))) == contracts.PERSONALITY_WIDTH
    a, b = (int(x) for x in args.seeds.split("-")); seeds = list(range(a, b + 1))
    report = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for name, p in GRID.items():
            n = len(seeds)
            rows = list(pool.map(rollout, seeds, [args.turns] * n, [args.checkpoint] * n, [0] * n, [args.movement] * n,
                                 [None] * n, [True if args.movement else None] * n, [0] * n, [None] * n, [None] * n, [p] * n,
                                 [args.grouped_region] * n, [args.focal] * n))
            population = sum(r["population"] for r in rows)
            settlements = sum(r["settlements"] for r in rows)
            report[name] = {"personality": p, "welfare": sum(r["welfare"] for r in rows), "population": population,
                            "qol": sum(r["qol"] * r["population"] for r in rows) / max(1, population),
                            "specialist_share": sum(r.get("specialist_share", 0.0) * r["population"] for r in rows) / max(1, population),
                            "trade_utility": sum(r["trade_utility"] for r in rows), "extinctions": sum(r["extinctions"] for r in rows),
                            # A degenerate split shows as many settlements and a high below-30 share.
                            "settlements": settlements, "below_30_share": sum(r["below_30"] for r in rows) / max(1, settlements),
                            "per_seed": [{"seed": s, "population": r["population"], "qol": r["qol"],
                                           "specialist_share": r.get("specialist_share", 0.0)} for s, r in zip(seeds, rows)]}
            if args.focal:
                focal = sum(r["focal_population"] for r in rows)
                report[name].update({"focal_population": focal,
                                     "focal_qol": sum(r["focal_qol"] * r["focal_population"] for r in rows) / max(1, focal),
                                     "focal_settlements": sum(r["focal_settlements"] for r in rows),
                                     "focal_below_30": sum(r["focal_below_30"] for r in rows),
                                     "focal_specialist_share": sum(r.get("focal_specialist_share", 0.0) * r["focal_population"]
                                                                   for r in rows) / max(1, focal)})
            print(json.dumps({name: report[name]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
