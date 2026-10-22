"""Train/evaluation input-shift check (docs/MODEL9_TOKENS.md, "Evaluating with movement on").

Compares the inputs of teacher shards (TRAIN) with the inputs a closed-loop evaluation feeds the policy, per bridge
movement setting. For every core column and set feature it reports the largest normalized deviation (TRAIN moments,
no guard) and flags the shift that collapsed the token clones: an input constant in TRAIN that takes another value at
evaluation (``dead_shifted``), or any input beyond ``--threshold`` standard deviations (``far``). The named core
columns (``--columns``, default the movement inputs v2-480 to v2-489) are listed with their raw ranges.

    python -m anthropocene_rl.autonomous_v7.input_shift --train <teacher>/train --seeds 2000-2002 --turns 10 \
        [--checkpoint <pt>] --out shift.json

Without ``--checkpoint`` the heuristic plays (the teacher's own states); with one, the policy plays as in
evaluate.rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contracts, data
from .bridge import Model9Bridge, observation_batch
from .evaluate import load, policy_actions
from .model import DEAD_STD

MOVEMENT_COLUMNS = [f"v2-{i}" for i in range(480, 490)]


def _symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def moments(train: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Per input group (core and each set): symlog mean and std over present entries, raw min and max."""
    out = {}
    for name, x, present in _groups(train):
        w = present.astype(np.float64)
        n = np.maximum(1.0, w.sum(axis=0))
        s = _symlog(x)
        mean = (s * w).sum(axis=0) / n
        std = np.sqrt((((s - mean) ** 2) * w).sum(axis=0) / n)
        lo = np.where(present, x, np.inf).min(axis=0)
        hi = np.where(present, x, -np.inf).max(axis=0)
        out[name] = (mean, std, lo, hi)
    return out


def _groups(batch: dict[str, np.ndarray]):
    yield "core", batch["core"].astype(np.float64), batch["core_present"].astype(bool)
    for name, _, width in contracts.SETS:
        if name not in batch:
            continue
        x = batch[name].reshape(-1, width).astype(np.float64)
        present = np.repeat(batch[name + "_present"].reshape(-1, 1), width, axis=1)
        yield name, x, present


def shift(train_moments: dict, evaluated: dict[str, np.ndarray], threshold: float = 10.0) -> dict[str, list[dict]]:
    """Inputs whose evaluation values leave the TRAIN distribution: per group, the columns that are constant in TRAIN
    yet differ at evaluation (dead_shifted) or exceed ``threshold`` normalized deviations (far)."""
    report: dict[str, list[dict]] = {}
    for name, x, present in _groups(evaluated):
        if name not in train_moments or not present.any():
            continue
        mean, std, lo, hi = train_moments[name]
        z = (_symlog(x) - mean) / np.maximum(std, 1e-3)
        z = np.where(present, np.abs(z), 0.0)
        rows = []
        for j in range(x.shape[1]):
            if not present[:, j].any():
                continue
            values = x[present[:, j], j]
            dead = bool(std[j] < DEAD_STD)
            dead_shifted = dead and bool(np.any(np.abs(_symlog(values) - mean[j]) > DEAD_STD))
            far = float(z[:, j].max())
            if dead_shifted or far > threshold:
                rows.append({"column": j, "dead_in_train": dead, "dead_shifted": dead_shifted, "max_abs_z": far,
                             "share_far": float((z[present[:, j], j] > threshold).mean()),
                             "train_range": [float(lo[j]), float(hi[j])],
                             "eval_range": [float(values.min()), float(values.max())]})
        if rows:
            report[name] = rows
    return report


def named_columns(train_moments: dict, evaluated: dict[str, np.ndarray], names: list[str],
                  features: list[str]) -> list[dict[str, Any]]:
    mean, std, lo, hi = train_moments["core"]
    out = []
    for feature in names:
        j = features.index(feature)
        present = evaluated["core_present"][:, j].astype(bool)
        values = evaluated["core"][present, j].astype(np.float64)
        z = np.abs((_symlog(values) - mean[j]) / max(float(std[j]), 1e-3)) if len(values) else np.zeros(0)
        out.append({"feature": feature, "column": j, "train_range": [float(lo[j]), float(hi[j])],
                    "train_symlog_std": float(std[j]), "dead_in_train": bool(std[j] < DEAD_STD),
                    "eval_range": [float(values.min()), float(values.max())] if len(values) else None,
                    "eval_mean": float(values.mean()) if len(values) else None,
                    "max_abs_z": float(z.max()) if len(z) else 0.0})
    return out


def collect(seeds: list[int], turns: int, movement: bool, checkpoint: str | None) -> tuple[dict[str, np.ndarray], dict]:
    """Every live village observation of closed-loop rollouts, and how often the policy's movement was legal / used."""
    torch.set_num_threads(1)
    actor = normalizer = None
    if checkpoint is not None:
        actor, normalizer = load(Path(checkpoint))
    batches = []
    decisions = legal = moves = 0
    with Model9Bridge() as bridge:
        features = list(bridge.request({"op": "schema"})["coreFeatures"])
        for seed in seeds:
            step = bridge.reset(seed, movement=movement)
            for _ in range(turns):
                live = [v for v in step["villages"] if v.get("observation") is not None and not v.get("terminated")]
                if live:
                    batches.append(observation_batch(live))
                actions = policy_actions(actor, normalizer, step["villages"], movement) if actor is not None else {}
                for v in live:
                    if int(v["camp"]) in actions:
                        decisions += 1
                        legal += int(any(v["observation"]["movementMask"][1:]))
                        moves += int(0 <= actions[int(v["camp"])]["movement"] < contracts.REGION_BASE)
                step = bridge.step(actions)
                if not any(not v["terminated"] for v in step["villages"]):
                    break
    merged = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
    return merged, {"features": features, "decisions": decisions, "movement_legal_share": legal / max(1, decisions),
                    "site_move_share": moves / max(1, decisions)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True, help="teacher shard directory (world-* subdirectories, any depth)")
    parser.add_argument("--checkpoint", default=None, help="the policy plays (default: the heuristic)")
    parser.add_argument("--seeds", default="2000-2002")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--columns", default=",".join(MOVEMENT_COLUMNS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a, _, b = args.seeds.partition("-")
    seeds = list(range(int(a), int(b or a) + 1))
    worlds = sorted(p.parent for p in args.train.rglob("meta.json"))
    train = data.load(worlds)
    train_moments = moments(train)
    report: dict[str, Any] = {"train_worlds": len(worlds), "train_rows": int(len(train["core"])), "seeds": seeds,
                              "turns": args.turns, "checkpoint": args.checkpoint}
    for movement in (True, False):
        evaluated, stats = collect(seeds, args.turns, movement, args.checkpoint)
        features = stats.pop("features")
        key = "movement_on" if movement else "movement_off"
        report[key] = {**stats, "rows": int(len(evaluated["core"])),
                       "named": named_columns(train_moments, evaluated, args.columns.split(","), features),
                       "shifted": shift(train_moments, evaluated, args.threshold)}
        for row in report[key]["shifted"].get("core", []):
            row["feature"] = features[row["column"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    for key in ("movement_on", "movement_off"):
        r = report[key]
        print(json.dumps({key: {"rows": r["rows"], "movement_legal_share": round(r["movement_legal_share"], 4),
                                "site_move_share": round(r["site_move_share"], 4),
                                "dead_shifted": {g: [x.get("feature", x["column"]) for x in rows if x["dead_shifted"]]
                                                 for g, rows in r["shifted"].items()},
                                "far": {g: len([x for x in rows if not x["dead_shifted"]]) for g, rows in r["shifted"].items()}}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
