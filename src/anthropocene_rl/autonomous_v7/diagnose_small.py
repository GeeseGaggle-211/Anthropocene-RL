"""Trace settlements that end below 30 residents under the cloned policy (and the teacher) on held-out worlds.

For each final small settlement: founding turn (age at the end), lineage, population / nutrition / QoL path and the
service levels the policy requested. Distinguishes young daughters (exempt from the guard), starving camps and
camps that stay small while fully fed (possible throttling).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .bridge import Model9Bridge
from .evaluate import load, policy_actions


def trace(seed: int, turns: int, checkpoint: Path | None) -> list[dict]:
    actor = normalizer = None
    if checkpoint is not None:
        actor, normalizer = load(checkpoint)
    history: dict[int, dict] = {}
    with Model9Bridge() as bridge:
        step = bridge.reset(seed)
        start = step["turn"]
        for v in step["villages"]:
            history.setdefault(
                v["camp"], {"founded": start, "parent": v["parentSociety"], "path": []}
            )
        for _ in range(turns):
            actions = (
                policy_actions(actor, normalizer, step["villages"]) if actor is not None else {}
            )
            step = bridge.step(actions)
            for v in step["villages"]:
                h = history.setdefault(
                    v["camp"], {"founded": step["turn"], "parent": v["parentSociety"], "path": []}
                )
                a = actions.get(v["camp"])
                h["path"].append(
                    {
                        "turn": step["turn"],
                        "pop": v["population"],
                        "F": round(v["nutrition"], 3),
                        "Q": round(v["qol"], 3),
                        "C": round(v["clothing"], 3),
                        "H": round(v["housing"], 3),
                        "service": None if a is None else [round(x, 2) for x in a["service"]],
                        "strategic": None if a is None else [round(x, 2) for x in a["strategic"]],
                        "terminated": v["terminated"],
                    }
                )
    end = step["turn"]
    out = []
    for camp, h in history.items():
        last = h["path"][-1] if h["path"] else None
        if last is None or last["terminated"] or last["pop"] >= 30:
            continue
        pops = [p["pop"] for p in h["path"]]
        out.append(
            {
                "seed": seed,
                "camp": camp,
                "age_turns": end - h["founded"],
                "daughter": h["parent"] > 0,
                "pop_path": pops,
                "peak": max(pops),
                "F_mean": float(np.mean([p["F"] for p in h["path"]])),
                "Q_last": last["Q"],
                "C_last": last["C"],
                "H_last": last["H"],
                "last_service": last["service"],
                "last_strategic": last["strategic"],
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", default="2000-2019")
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    from .checkpoint import parse

    args = parse(parser)
    torch.set_num_threads(2)
    a, b = (int(x) for x in args.seeds.split("-"))
    report = {"policy": [], "teacher": []}
    for seed in range(a, b + 1):
        report["policy"] += trace(seed, args.turns, args.checkpoint)
        report["teacher"] += trace(seed, args.turns, None)
    args.out.write_text(json.dumps(report, indent=1))
    for key in ("policy", "teacher"):
        rows = report[key]
        young = sum(1 for r in rows if r["daughter"] and r["age_turns"] <= 3)
        fed = sum(1 for r in rows if r["F_mean"] >= 0.95)
        print(
            json.dumps(
                {
                    key: {
                        "below30": len(rows),
                        "young_daughters": young,
                        "fed": fed,
                        "examples": rows[:6],
                    }
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
