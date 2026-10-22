"""Movement audit (docs/MODEL9_MOVEMENT_AUDIT.md): paired closed-loop rollouts that record, every turn, the read-only
bridge ``inspect`` view of every camp (tile, lifestyle, capital, food by source, footprint stock) next to the policy's
movement directive.

    python -m anthropocene_rl.autonomous_v7.movement_audit --policy grouped --checkpoint s55.pt --seeds 2000-2019,2026-2059 --out dir

Policies: ``grouped`` (learned region movement, hierarchical choice), ``argmax`` (learned region movement, plain
argmax), ``native`` (the checkpoint's levers with the heuristic's movement), ``stay`` (learned levers, region slot
forced to the current tile: never moves). Rollouts match evaluate.rollout exactly;
``inspect`` asserts the world fingerprint is unchanged, and the post-step fingerprints are recorded for a paired check.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

from . import contracts
from .bridge import Model9Bridge
from .evaluate import load, policy_actions, producer_identity
from .stage_eval import seed_list


def inspect(bridge: Model9Bridge, tiles: list[int]) -> dict[str, Any]:
    return bridge.request({"op": "inspect", "tiles": sorted(set(tiles))})["inspect"]


def audit_rollout(seed: int, turns: int, checkpoint: str, policy: str, out: str, tag: str = "", withheld: int = 0) -> dict[str, Any]:
    torch.set_num_threads(1)
    from .provenance import checkpoint_hash
    checkpoint_sha = checkpoint_hash(checkpoint)
    actor, normalizer = load(Path(checkpoint))
    movement = policy in ("grouped", "argmax", "stay")
    actor.region_movement = movement
    actor.grouped_region = policy == "grouped"
    # Selection counterfactuals preserve the saved/explicit head and its authoritative joint distribution.
    actor.region_selection = policy if policy in ("argmax", "stay") else None
    abandoned: list[int] = []
    rows: list[dict[str, Any]] = []
    with Model9Bridge() as bridge:
        from .worker_contract import validate, world
        validate(bridge, getattr(actor, "worker_contract", None), world(seed, withheld, movement=movement, head=actor.region_head), head=actor.region_head)
        step = bridge.reset(seed, withheld=withheld, movement=movement)
        producer = producer_identity(bridge)
        pre = inspect(bridge, abandoned)
        for _ in range(turns):
            actions = policy_actions(actor, normalizer, step["villages"], movement)
            step = bridge.step(actions)
            post = inspect(bridge, abandoned)
            before = {c["camp"]: c for c in pre["camps"]}
            after = {c["camp"]: c for c in post["camps"]}
            for camp, c in after.items():
                if camp in before and before[camp]["tile"] != c["tile"]:
                    abandoned.append(before[camp]["tile"])
            rows.append({
                "turn": pre["turn"], "pre": pre, "post_tiles": post["tiles"],
                "directives": {str(k): v.get("movement", contracts.NATIVE_MOVEMENT) for k, v in actions.items()},
                "villages": [{k: v[k] for k in ("camp", "society", "parentSociety", "priorPopulation", "population", "qol",
                                                 "welfare", "terminated", "founded")} for v in step["villages"]],
                "fingerprint": step["fingerprint"],
            })
            pre = post
            if not any(not v["terminated"] for v in step["villages"]):
                break
        rows.append({"turn": pre["turn"], "pre": pre, "final": True})
        replay = bridge.verify_replay()
    path = Path(out) / f"{tag or policy}-{seed}.json"
    path.write_text(json.dumps({"seed": seed, "policy": policy, "region_head": actor.region_head,
                               "checkpoint_sha256": checkpoint_sha, "producer": producer,
                               "region_selection": actor.region_selection, "turns": turns, "replay": replay, "rows": rows}))
    return {"seed": seed, "policy": tag or policy, "region_head": actor.region_head,
            "region_selection": actor.region_selection, "replay": replay, "path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy", choices=("grouped", "argmax", "native", "stay"), required=True)
    parser.add_argument("--seeds", default="2000-2019,2026-2059")
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", default="", help="file prefix (default: the policy)")
    parser.add_argument("--fixture", default="", help="capability fixture name (contracts.CAPABILITY_MASKS), e.g. fishing")
    from .checkpoint import parse
    args = parse(parser)
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = seed_list(args.seeds)
    clock = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(audit_rollout, seeds, [args.turns] * len(seeds), [args.checkpoint] * len(seeds),
                          [args.policy] * len(seeds), [str(args.out)] * len(seeds), [args.tag] * len(seeds),
                          [contracts.CAPABILITY_MASKS[args.fixture] if args.fixture else 0] * len(seeds)):
            print(json.dumps(r), flush=True)
    print(json.dumps({"seconds": time.time() - clock}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
