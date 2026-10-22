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


def seed_list(text: str) -> list[int]:
    """"2000-2019,2026-2059" -> the seeds in order."""
    seeds: list[int] = []
    for part in text.split(","):
        a, _, b = part.strip().partition("-")
        seeds += list(range(int(a), int(b or a) + 1))
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--candidates", required=True, help="comma-separated checkpoints")
    parser.add_argument("--default-seeds", default="2000-2019", help="comma-separated seeds or ranges")
    parser.add_argument("--fixture-seeds", default="2000-2009", help="comma-separated seeds or ranges")
    parser.add_argument("--grassland-seeds", type=Path, default=None, help="file of grassland seeds; uses the last --grassland-count")
    parser.add_argument("--grassland-count", type=int, default=10)
    parser.add_argument("--valley", type=Path, default=None,
                        help="contract-10 valley fixtures: lines 'seed,warmup' (the heuristic plays warmup turns first)")
    parser.add_argument("--valley-strict", type=Path, default=None, help="strict valley fixtures (competition and a valley)")
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--movement", action="store_true", help="policies choose movement (stage 4); teacher stays native")
    parser.add_argument("--region-movement", action="store_true",
                        help="with --movement: every policy, the init included, moves over contract-10 region rows")
    parser.add_argument("--grouped-region", action="store_true",
                        help="candidates choose their region slot hierarchically (native / stay / go first, then where)")
    parser.add_argument("--regime-tolerance", type=float, default=0.03,
                        help="a candidate fails if any regime's welfare falls more than this share below the init")
    parser.add_argument("--out", type=Path, required=True)
    from .checkpoint import parse
    args = parse(parser)
    categories = {"default": [(s, 0) for s in seed_list(args.default_seeds)]}
    for name, mask in contracts.CAPABILITY_MASKS.items():
        categories[name] = [(s, mask) for s in seed_list(args.fixture_seeds)]
    if args.grassland_seeds is not None:
        seeds = [int(x) for x in args.grassland_seeds.read_text().split(",") if x.strip()][-args.grassland_count:]
        categories["grassland"] = [(s, 0) for s in seeds]
    for name, path in (("valley", args.valley), ("valley-strict", args.valley_strict)):
        if path is not None:
            pairs = [line.split(",") for line in path.read_text().split() if line.strip()]
            categories[name] = [(int(s), 0, int(w)) for s, w in pairs]
    from .worker_contract import from_state
    import torch
    expected_producer = from_state(torch.load(args.init, map_location="cpu", weights_only=False))
    report = {}
    for name, worlds in categories.items():
        teacher = evaluate_closed_loop(None, worlds, args.turns, args.workers, expected_producer=expected_producer)
        region = True if args.region_movement else None
        init = evaluate_closed_loop(args.init, worlds, args.turns, args.workers, movement=args.movement, region=region,
                                    expected_producer=expected_producer)
        row = {"teacher": teacher, "init": init, "candidates": {}}
        for candidate in args.candidates.split(","):
            ev = evaluate_closed_loop(Path(candidate), worlds, args.turns, args.workers, movement=args.movement, region=region,
                                      grouped=args.grouped_region, expected_producer=expected_producer)
            ev["ratio_vs_init"] = ev["welfare"] / max(1e-9, init["welfare"])
            ev["ratio_vs_teacher"] = ev["welfare"] / max(1e-9, teacher["welfare"])
            row["candidates"][candidate] = ev
        row["init"]["ratio_vs_teacher"] = init["welfare"] / max(1e-9, teacher["welfare"])
        report[name] = row
        print(json.dumps({name: {"init_vs_teacher": round(row["init"]["ratio_vs_teacher"], 4),
                                 "specialist_share": {"teacher": round(teacher["specialist_share"], 4),
                                                      "init": round(init["specialist_share"], 4)},
                                 **{Path(k).parent.name + "/" + Path(k).stem: {"vs_init": round(v["ratio_vs_init"], 4),
                                    "vs_teacher": round(v["ratio_vs_teacher"], 4), "qol": round(v["qol"], 3),
                                    "specialist_share": round(v["specialist_share"], 4),
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
