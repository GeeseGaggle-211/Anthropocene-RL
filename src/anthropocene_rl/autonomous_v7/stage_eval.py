"""Stage-gate evaluation for M7: closed-loop deterministic rollouts per world category.

    python -m anthropocene_rl.autonomous_v7.stage_eval --init <stage1.pt> --candidates a.pt,b.pt --out <json>

Categories are ordinary DEVELOPMENT worlds and the three capability fixtures (same seeds, withheld-technology
masks), and optionally grassland-rich openings. Each candidate is compared with the init policy and the native
teacher on the same worlds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import contracts
from .ppo import evaluate_closed_loop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--candidates", required=True, help="comma-separated checkpoints")
    parser.add_argument("--default-seeds", default="2000-2019")
    parser.add_argument("--fixture-seeds", default="2000-2009")
    parser.add_argument("--grassland-seeds", type=Path, default=None, help="file of grassland seeds; uses the last 10")
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--movement", action="store_true", help="policies choose movement (stage 4); teacher stays native")
    parser.add_argument("--regime-tolerance", type=float, default=0.03,
                        help="a candidate fails if any regime's welfare falls more than this share below the init")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a, b = (int(x) for x in args.default_seeds.split("-")); fa, fb = (int(x) for x in args.fixture_seeds.split("-"))
    categories = {"default": [(s, 0) for s in range(a, b + 1)]}
    for name, mask in contracts.CAPABILITY_MASKS.items():
        categories[name] = [(s, mask) for s in range(fa, fb + 1)]
    if args.grassland_seeds is not None:
        seeds = [int(x) for x in args.grassland_seeds.read_text().split(",") if x.strip()][-10:]
        categories["grassland"] = [(s, 0) for s in seeds]
    report = {}
    for name, worlds in categories.items():
        teacher = evaluate_closed_loop(None, worlds, args.turns, args.workers)
        init = evaluate_closed_loop(args.init, worlds, args.turns, args.workers, movement=args.movement)
        row = {"teacher": teacher, "init": init, "candidates": {}}
        for candidate in args.candidates.split(","):
            ev = evaluate_closed_loop(Path(candidate), worlds, args.turns, args.workers, movement=args.movement)
            ev["ratio_vs_init"] = ev["welfare"] / max(1e-9, init["welfare"])
            ev["ratio_vs_teacher"] = ev["welfare"] / max(1e-9, teacher["welfare"])
            row["candidates"][candidate] = ev
        row["init"]["ratio_vs_teacher"] = init["welfare"] / max(1e-9, teacher["welfare"])
        report[name] = row
        print(json.dumps({name: {"init_vs_teacher": round(row["init"]["ratio_vs_teacher"], 4),
                                 **{Path(k).parent.name + "/" + Path(k).stem: {"vs_init": round(v["ratio_vs_init"], 4),
                                    "vs_teacher": round(v["ratio_vs_teacher"], 4), "qol": round(v["qol"], 3),
                                    "below_30": v["below_30"], "extinctions": v["extinctions"]}
                                    for k, v in row["candidates"].items()}}}), flush=True)
    # Gate: default welfare at least the init's, no regime more than the tolerance below the init, and no more
    # extinctions than the init in any regime.
    gate = {}
    for candidate in args.candidates.split(","):
        failures = []
        for name, row in report.items():
            ev = row["candidates"][candidate]
            floor = 1.0 if name == "default" else 1.0 - args.regime_tolerance
            if ev["ratio_vs_init"] < floor:
                failures.append(f"{name} {ev['ratio_vs_init']:.3f} < {floor:.2f}")
            if ev["extinctions"] > row["init"]["extinctions"]:
                failures.append(f"{name} extinctions {ev['extinctions']:.0f} > {row['init']['extinctions']:.0f}")
        gate[candidate] = {"pass": not failures, "failures": failures}
        print(json.dumps({"gate": Path(candidate).parent.name + "/" + Path(candidate).stem, "pass": not failures,
                          "failures": failures}), flush=True)
    report["gate"] = gate
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
