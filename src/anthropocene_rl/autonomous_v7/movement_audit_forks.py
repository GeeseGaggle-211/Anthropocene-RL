"""Paired fork test of learned movement (docs/MODEL9_MOVEMENT_AUDIT.md).

    python -m anthropocene_rl.autonomous_v7.movement_audit_forks --checkpoint s55.pt --seeds 2000-2019 --points 4,8,12,16 --horizon 4 --out forks.json

The checkpoint plays with the grouped region choice. At each fork point the world forks into two exactly equal branches
played for ``horizon`` turns: MOVE (the policy's own actions) and STAY (identical actions except the movement code,
which is the current-tile row, so no camp relocates). Reported per pair: welfare, expectation-neutral welfare (QoL with
the settled expectation E = 1, from the read-only inspect op), residents, settlements, gross births and deaths, fissions,
physical moves, and food produced by source.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

from . import contracts
from .bridge import Model9Bridge
from .evaluate import load, policy_actions
from .stage_eval import seed_list

STAY = contracts.REGION_BASE  # region row 0: the current tile


def branch(bridge: Model9Bridge, actor, normalizer, step: dict[str, Any], horizon: int, stay: bool) -> dict[str, Any]:
    totals = {"welfare": 0.0, "welfare_e1": 0.0, "births": 0, "deaths": 0, "fissions": 0, "moves": 0,
              "land": 0.0, "fish": 0.0, "herd": 0.0, "produced": 0.0, "clothing_food": 0.0}
    pre = bridge.request({"op": "inspect"})["inspect"]
    for _ in range(horizon):
        actions = policy_actions(actor, normalizer, step["villages"], True)
        if stay:
            for a in actions.values():
                a["movement"] = STAY
        step = bridge.step(actions)
        post = bridge.request({"op": "inspect"})["inspect"]
        camps = {c["camp"]: c for c in post["camps"]}
        before = {c["camp"]: c["tile"] for c in pre["camps"]}
        for v in step["villages"]:
            if v["priorPopulation"] <= 0:
                continue
            totals["welfare"] += v["welfare"]
            c = camps.get(v["camp"])
            totals["welfare_e1"] += v["welfare"] * (c["gearSettled"] / c["gear"] if c and c["gear"] > 1e-9 else 1.0)
        for c in post["camps"]:
            h = c.get("harvest")
            if h:
                totals["births"] += h["births"]; totals["deaths"] += h["deaths"]
                totals["land"] += h["terrestrial"]; totals["fish"] += h["marine"] + h["lake"] + h["fish"]
                totals["herd"] += h["animal"] + h["hunt"]; totals["produced"] += h["produced"]
            totals["moves"] += int(c["camp"] in before and before[c["camp"]] != c["tile"])
        totals["fissions"] += len(post.get("fissions") or [])
        pre = post
        if not any(not v["terminated"] for v in step["villages"]):
            break
    final = pre["camps"]
    people = sum(c["population"] for c in final)
    totals.update(residents=people, settlements=len(final),
                  qol=sum(c["qol"] * c["population"] for c in final) / max(1, people),
                  qol_e1=sum(c["qol"] * c["gearSettled"] / max(1e-9, c["gear"]) * c["population"] for c in final) / max(1, people),
                  cloth=sum(c["cloth"] * c["population"] for c in final) / max(1, people),
                  house=sum(c["house"] * c["population"] for c in final) / max(1, people),
                  expectation=sum(c["expectation"] * c["population"] for c in final) / max(1, people))
    return totals


def fork_seed(seed: int, checkpoint: str, points: list[int], horizon: int) -> list[dict[str, Any]]:
    torch.set_num_threads(1)
    actor, normalizer = load(Path(checkpoint))
    actor.region_movement = True; actor.grouped_region = True
    out = []
    with Model9Bridge() as bridge:
        from .worker_contract import validate, world
        validate(bridge, getattr(actor, "worker_contract", None), world(seed, movement=True, head=actor.region_head), head=actor.region_head)
        step = bridge.reset(seed, movement=True)
        from .evaluate import producer_identity
        from .provenance import checkpoint_hash
        producer = producer_identity(bridge); checkpoint_sha = checkpoint_hash(checkpoint)
        turn = 0
        for point in sorted(points):
            while turn < point:
                step = bridge.step(policy_actions(actor, normalizer, step["villages"], True)); turn += 1
            fork, step = bridge.fork()
            move = branch(bridge, actor, normalizer, step, horizon, stay=False)
            step = bridge.restore(fork)
            stay = branch(bridge, actor, normalizer, step, horizon, stay=True)
            step = bridge.restore(fork)
            bridge.drop(fork)
            out.append({"seed": seed, "point": point, "horizon": horizon, "move": move, "stay": stay,
                        "checkpoint_sha256": checkpoint_sha, "producer": producer})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", default="2000-2019")
    parser.add_argument("--points", default="4,8,12,16")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    from .checkpoint import parse
    args = parse(parser)
    seeds = seed_list(args.seeds); points = [int(p) for p in args.points.split(",")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pairs = [p for rows in pool.map(fork_seed, seeds, [args.checkpoint] * len(seeds), [points] * len(seeds),
                                       [args.horizon] * len(seeds)) for p in rows]
    keys = list(pairs[0]["move"])
    summary = {k: {"move": sum(p["move"][k] for p in pairs) / len(pairs), "stay": sum(p["stay"][k] for p in pairs) / len(pairs)}
               for k in keys}
    ratios = sorted(p["move"]["welfare"] / max(1e-9, p["stay"]["welfare"]) for p in pairs)
    summary["welfare_ratio_total"] = sum(p["move"]["welfare"] for p in pairs) / sum(p["stay"]["welfare"] for p in pairs)
    summary["welfare_e1_ratio_total"] = sum(p["move"]["welfare_e1"] for p in pairs) / sum(p["stay"]["welfare_e1"] for p in pairs)
    summary["welfare_ratio_median"] = ratios[len(ratios) // 2]
    summary["move_wins"] = sum(r > 1 for r in ratios) / len(ratios)
    args.out.write_text(json.dumps({"pairs": pairs, "summary": summary}, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
