"""Closed-loop evaluation of a model9 checkpoint against the native heuristic (stage-1 gate).

    python -m anthropocene_rl.autonomous_v7.evaluate --checkpoint <pt> --seeds 2000-2019 --turns 25 --out <json>

Each held-out world runs twice from the same opening: the native heuristic for every village (teacher) and the
cloned policy for every village. Stages 1-3 keep movement and fission native. Welfare is realized population x
QoL x years summed over every settlement (lineage welfare summed over lineages), zero after extinction.
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

from . import contracts
from .bridge import Model9Bridge, observation_batch
from .model import Actor, Normalizer, act_deterministic, region_code, to_tensors
from .upgrade import upgrade


def load(checkpoint: Path) -> tuple[Actor, Normalizer]:
    state = upgrade(torch.load(checkpoint, map_location="cpu", weights_only=False))
    actor = Actor(int(state["core_width"]), experts=bool(state.get("experts", False)))
    actor.load_state_dict(state["actor"]); actor.eval()
    # Contract 9: a policy trained without commitments decides every turn at evaluation too, whatever span head it
    # inherited. Checkpoints saved before the flag existed keep the span argmax, so recorded results reproduce.
    actor.commitments = bool(state.get("commitments", True))
    # Contract 10: a policy trained with learned movement moves over region rows; others keep the contract-8 pointer.
    actor.region_movement = bool(state.get("region_movement", False))
    return actor, Normalizer.from_state(state["normalizer"])


def policy_actions(actor: Actor, normalizer: Normalizer, villages: list[dict[str, Any]],
                   movement: bool = False, mixes: dict[int, np.ndarray] | None = None) -> dict[int, dict[str, Any]]:
    """Deterministic actions; movement stays native unless ``movement`` (curriculum stage 4). ``mixes`` receives each
    village's economy mix."""
    live = [v for v in villages if v.get("observation") is not None and not v.get("terminated")]
    if not live:
        return {}
    region = movement and getattr(actor, "region_movement", False)
    with torch.no_grad():
        mode = act_deterministic(actor(to_tensors(observation_batch(live), normalizer)), region=region)
    if not getattr(actor, "commitments", True):
        mode["span"] = np.full(len(live), contracts.SPAN_SINGLE)
    actions = {}
    if mixes is not None:
        for i, v in enumerate(live):
            mixes[int(v["camp"])] = np.asarray(mode["mix"][i], dtype=np.float64)
    for i, v in enumerate(live):
        actions[int(v["camp"])] = {
            "strategic": mode["strategic"][i].tolist(), "service": mode["service"][i].tolist(),
            "clothingRoute": int(mode["clothing"][i]), "shelterRoute": int(mode["shelter"][i]),
            "movement": (region_code(int(mode["region"][i])) if region else int(mode["movement"][i])) if movement
                        else contracts.NATIVE_MOVEMENT, "fission": -1,
            "commit": contracts.SPANS[int(mode["span"][i])],
        }
    return actions


def rollout(seed: int, turns: int, checkpoint: str | None, withheld: int = 0, movement: bool = False,
            commitments: bool | None = None, region: bool | None = None) -> dict[str, Any]:
    """``commitments`` and ``region`` (contract-10 learned movement) override the checkpoint's own settings."""
    torch.set_num_threads(1)
    actor = normalizer = None
    if checkpoint is not None:
        actor, normalizer = load(Path(checkpoint))
        if commitments is not None:
            actor.commitments = commitments
        if region is not None:
            actor.region_movement = region
    welfare = 0.0; camp_turns = hungry = interventions = decisions = 0
    previous: dict[int, np.ndarray] = {}; changes: list[float] = []
    spans: dict[int, int] = {}
    moves = 0
    with Model9Bridge() as bridge:
        step = bridge.reset(seed, withheld=withheld, movement=movement)
        for _ in range(turns):
            mixes: dict[int, np.ndarray] = {}
            actions = policy_actions(actor, normalizer, step["villages"], movement, mixes) if actor is not None else {}
            for action in actions.values():
                spans[action.get("commit", 1)] = spans.get(action.get("commit", 1), 0) + 1
                code = action.get("movement", contracts.NATIVE_MOVEMENT)
                moves += int(code == contracts.KEEP_INTENT or code > contracts.REGION_BASE)
            for camp, mix in mixes.items():
                if camp in previous:
                    changes.append(0.5 * float(np.abs(mix - previous[camp]).sum()))
                previous[camp] = mix
            decisions += len(actions)
            step = bridge.step(actions)
            for v in step["villages"]:
                if v["priorPopulation"] <= 0:
                    continue
                welfare += float(v["welfare"]); camp_turns += 1
                hungry += int(v["nutrition"] < 0.8 and not v["terminated"])
                interventions += int(v["interventions"])
            if not any(not v["terminated"] for v in step["villages"]):
                break
        replay = bridge.verify_replay()
        final = [v for v in step["villages"] if not v["terminated"] and v["population"] > 0]
    population = sum(v["population"] for v in final)
    weight = max(1, population)
    return {
        "seed": seed, "withheld": withheld, "policy": checkpoint is not None, "welfare": welfare, "population": population,
        "settlements": len(final), "villages_600": sum(1 for v in final if v["population"] >= 600),
        "nutrition": sum(v["nutrition"] * v["population"] for v in final) / weight,
        "clothing": sum(v["clothing"] * v["population"] for v in final) / weight,
        "housing": sum(v["housing"] * v["population"] for v in final) / weight,
        "qol": sum(v["qol"] * v["population"] for v in final) / weight,
        "hungry_share": hungry / max(1, camp_turns),
        "extinctions": sum(1 for v in step["villages"] if v["terminated"]),
        "below_30": sum(1 for v in final if v["population"] < 30),
        "interventions_per_decision": interventions / max(1, decisions), "replay_steps": replay,
        # Turn-to-turn total-variation change of each village's economy mix, and the share of abrupt switches.
        "mix_change": float(np.mean(changes)) if changes else 0.0,
        "mix_switches": float(np.mean([c > 0.5 for c in changes])) if changes else 0.0,
        # Contract 9: share of decisions holding (0), deciding one turn (1) or committing (2-6).
        "hold_share": spans.get(0, 0) / max(1, sum(spans.values())),
        "commit_share": sum(v for k, v in spans.items() if k >= 2) / max(1, sum(spans.values())),
        # Contract 10: share of decisions that chose a region row other than the current tile, or kept an intent.
        "region_move_share": moves / max(1, sum(spans.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", default="2000-2019")
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a, b = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(a, b + 1))
    clock = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        teacher = list(pool.map(rollout, seeds, [args.turns] * len(seeds), [None] * len(seeds)))
        policy = list(pool.map(rollout, seeds, [args.turns] * len(seeds), [str(args.checkpoint)] * len(seeds)))
    ratio = [p["welfare"] / max(1e-9, t["welfare"]) for p, t in zip(policy, teacher)]

    def mean(rows: list[dict[str, Any]], key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    keys = ["welfare", "population", "settlements", "villages_600", "nutrition", "clothing", "housing", "qol",
            "hungry_share", "extinctions", "below_30", "interventions_per_decision"]
    report = {
        "seeds": args.seeds, "turns": args.turns, "checkpoint": str(args.checkpoint),
        "welfare_ratio_total": sum(p["welfare"] for p in policy) / max(1e-9, sum(t["welfare"] for t in teacher)),
        "welfare_ratio_median": float(np.median(ratio)), "welfare_ratio_min": float(np.min(ratio)),
        "teacher": {k: mean(teacher, k) for k in keys}, "policy": {k: mean(policy, k) for k in keys},
        "per_seed": [{"seed": t["seed"], "teacher": t["welfare"], "policy": p["welfare"], "ratio": r}
                     for t, p, r in zip(teacher, policy, ratio)],
        "seconds": time.time() - clock,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("welfare_ratio_total", "welfare_ratio_median", "welfare_ratio_min", "teacher", "policy", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
