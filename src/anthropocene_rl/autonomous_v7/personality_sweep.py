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
    parser.add_argument(
        "--grouped-region",
        action="store_true",
        help="hierarchical region choice (model.grouped_region_slot)",
    )
    parser.add_argument(
        "--focal",
        action="store_true",
        help="only alternate founding lineages take the sweep point; the rest draw Dirichlet(1) as in "
        "training, and the focal lineages' outcomes are reported",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--points", default=None, help="explicit comma-separated grid points; default all seven"
    )
    parser.add_argument("--completion", type=Path, default=None)
    parser.add_argument("--checkpoint-form", choices=("best", "latest"), default=None)
    from .checkpoint import parse, resolve_head

    args = parse(parser)
    import torch
    from .provenance import checkpoint_hash, ProvenanceError

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    initial_sha = checkpoint_hash(args.checkpoint)
    actual_head = resolve_head(saved, args.region_head)
    from .promotion import semantic_hash

    actual_semantics = semantic_hash(saved)
    from .worker_contract import from_state, world

    expected_worker = from_state(
        saved,
        required=bool(
            args.completion or saved.get("run_manifest", {}).get("options", {}).get("arm")
        ),
    )
    completion_ref = None
    if args.completion:
        from .completion import validate

        completion_ref = {
            "path": str(args.completion.resolve()),
            "sha256": checkpoint_hash(args.completion),
        }
        validate(completion_ref, args.checkpoint, args.checkpoint_form)
    if args.points is not None and (
        not args.points or any(p not in GRID for p in args.points.split(","))
    ):
        raise ProvenanceError("unknown or empty personality sweep grid")
    assert len(next(iter(GRID.values()))) == contracts.PERSONALITY_WIDTH
    a, b = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(a, b + 1))
    report = {}
    producer = None
    with ProcessPoolExecutor(args.workers) as pool:
        for name, p in GRID.items():
            if args.points is not None and name not in args.points.split(","):
                continue
            n = len(seeds)
            descriptors = [world(seed, movement=args.movement, head=actual_head) for seed in seeds]
            rows = list(
                pool.map(
                    rollout,
                    seeds,
                    [args.turns] * n,
                    [args.checkpoint] * n,
                    [0] * n,
                    [args.movement] * n,
                    [None] * n,
                    [True if args.movement else None] * n,
                    [0] * n,
                    [None] * n,
                    [None] * n,
                    [p] * n,
                    [args.grouped_region] * n,
                    [args.focal] * n,
                    [expected_worker] * n,
                    descriptors,
                )
            )
            if len(rows) != n or any(
                row.get("world_contract") != descriptor
                for row, descriptor in zip(rows, descriptors)
            ):
                raise ProvenanceError(
                    "sweep worker world differs from frozen pre-execution contract"
                )
            if expected_worker is not None and any(
                row.get("worker_contract") != expected_worker for row in rows
            ):
                raise ProvenanceError("sweep worker did not honor frozen pre-execution contract")
            producer = validate_producers(rows, saved, producer)
            if any(row.get("personality") != p for row in rows):
                raise ProvenanceError("sweep workers evaluated a different personality vector")
            population = sum(r["population"] for r in rows)
            settlements = sum(r["settlements"] for r in rows)
            report[name] = {
                "personality": p,
                "welfare": sum(r["welfare"] for r in rows),
                "population": population,
                "qol": sum(r["qol"] * r["population"] for r in rows) / max(1, population),
                "specialist_share": sum(
                    r.get("specialist_share", 0.0) * r["population"] for r in rows
                )
                / max(1, population),
                "trade_utility": sum(r["trade_utility"] for r in rows),
                "extinctions": sum(r["extinctions"] for r in rows),
                # A degenerate split shows as many settlements and a high below-30 share.
                "settlements": settlements,
                "below_30_share": sum(r["below_30"] for r in rows) / max(1, settlements),
                "per_seed": [
                    {
                        k: r[k]
                        for k in (
                            "seed",
                            "withheld",
                            "warmup",
                            "personality",
                            "population",
                            "qol",
                            "welfare",
                            "extinctions",
                            "settlements",
                            "below_30",
                            "specialist_share",
                            "world_contract",
                        )
                    }
                    | {k: v for k, v in r.items() if k.startswith("focal_")}
                    for r in rows
                ],
            }
            if args.focal:
                focal = sum(r["focal_population"] for r in rows)
                report[name].update(
                    {
                        "focal_population": focal,
                        "focal_qol": sum(r["focal_qol"] * r["focal_population"] for r in rows)
                        / max(1, focal),
                        "focal_settlements": sum(r["focal_settlements"] for r in rows),
                        "focal_below_30": sum(r["focal_below_30"] for r in rows),
                        "focal_specialist_share": sum(
                            r.get("focal_specialist_share", 0.0) * r["focal_population"]
                            for r in rows
                        )
                        / max(1, focal),
                    }
                )
            print(json.dumps({name: report[name]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_hash(args.checkpoint) != initial_sha:
        raise ProvenanceError("checkpoint changed during sweep")
    report["_meta"] = {
        "schema": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": initial_sha,
        "checkpoint_form": args.checkpoint_form,
        "completion": completion_ref,
        "turns": args.turns,
        "points": list(report),
        "model_semantics": actual_semantics,
        "simulator": producer["simulator"],
        "bridge_sha256": producer["bridge_sha256"],
        "producer": producer,
        "arm": saved.get("run_manifest", {}).get("options", {}).get("arm"),
        "worker_contract": expected_worker,
        "optimizer_seed": saved.get("run_manifest", {})
        .get("options", {})
        .get("seed", saved.get("seed")),
        "focal": args.focal,
        "movement": args.movement,
        "region_head": actual_head,
    }
    args.out.write_text(json.dumps(report, indent=2))
    return 0


def validate_producers(rows, checkpoint, previous=None):
    from .provenance import ProvenanceError

    frozen = checkpoint.get("run_manifest", {})
    producers = [row.get("producer") for row in rows]
    if not producers or any(not isinstance(p, dict) for p in producers):
        raise ProvenanceError("sweep workers did not supply live producer identities")
    expected = {
        "simulator": frozen.get("simulator"),
        "bridge_sha256": frozen.get("bridge_sha256"),
        "flags": {k: frozen.get(k) for k in ("sim_rules", "sim_statics")},
        "raw_materials": False,
    }
    if any(p != expected or (previous is not None and p != previous) for p in producers):
        raise ProvenanceError(
            "live sweep worker producer differs across workers/checkpoint/configuration"
        )
    return producers[0]


if __name__ == "__main__":
    raise SystemExit(main())
