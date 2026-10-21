"""Critic bias on search branch states (docs/MODEL9_SEARCH_TEACHER.md, *Validation*).

    python -m anthropocene_rl.autonomous_v7.branch_bias --checkpoint <pt> --branch-shards <dir>[,<dir>] [--out bias.json]

On held-out branch shards (search_teacher ``<out>/branch_shards`` from worlds the critic never trained on), per
alternative kind and objective head, as relative bias (prediction - realised) / |realised|, mean and median:

* ``calibration`` (the gate): calibration branches (12-turn tails). Per branch, the critic's implied 8-turn lineage
  value V(s_0) - Gamma_8 V(s_8) against the realised discounted 8-turn lineage return, with no bootstrap in the
  target. ``pass``: within 10% (mean and median, welfare head) for native, region and mix.
* ``self_consistency`` (diagnostic): every branch; the critic's value against the tail return plus its own
  discounted bootstrap after the last tail turn (lambda = 1). With a 4-turn tail most of that target is the
  bootstrap, so it measures consistency, not calibration. ``entry`` is one number per branch (lineage sums at the
  first tail turn), ``rows`` every tail row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import branch_shards, data
from .search_teacher import KINDS, MIX, NATIVE, REGION, load_critic
from .upgrade import upgrade
from .model import Normalizer

TARGET = 0.10
GATED = (NATIVE, REGION, MIX)
CALIBRATION_TURNS = 8


def relative(pred: np.ndarray, real: np.ndarray) -> np.ndarray:
    return (pred - real) / np.maximum(1e-9, np.abs(real))


def realised(critic, rows: dict[str, np.ndarray], boot, normalizer) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Predicted raw values per head and the realised tail return plus bootstrap (lambda = 1)."""
    value, bootstrap = branch_shards.values(critic, rows, boot, normalizer)
    components = data.reward_components(rows)
    real = {name: data.lambda_returns(rows, components[name], value[name], 1.0, bootstrap[name]) for name in value}
    return value, real


def report(pred: dict[str, np.ndarray], real: dict[str, np.ndarray], rows: dict[str, np.ndarray]) -> dict:
    kinds = rows["alt_kind"].astype(int)
    world, turn = rows["world"], rows["turn"]
    # Entry rows: each branch world's first tail turn (all lineage settlements deciding then).
    first = {}
    for w, t in zip(world.tolist(), turn.tolist()):
        first[w] = min(first.get(w, t), t)
    entry = np.asarray([first[w] == t for w, t in zip(world.tolist(), turn.tolist())])
    out: dict = {"rows": int(len(kinds)), "branches": int(len(first)), "target": TARGET, "kinds": {}}
    for k in sorted(set(kinds.tolist())):
        sel = kinds == k
        by_head = {}
        for name in pred:
            r = relative(pred[name][sel], real[name][sel])
            worlds = sorted(set(world[sel & entry].tolist()))
            e = np.asarray([relative(np.asarray([pred[name][(world == w) & entry].sum()]),
                                     np.asarray([real[name][(world == w) & entry].sum()]))[0] for w in worlds])
            by_head[name] = {"entry_mean": float(np.mean(e)) if len(e) else None,
                             "entry_median": float(np.median(e)) if len(e) else None,
                             "rows_mean": float(np.mean(r)), "rows_median": float(np.median(r)),
                             "branches": len(worlds), "rows": int(sel.sum())}
        out["kinds"][KINDS[k] if 0 <= k < len(KINDS) else str(k)] = by_head
    return out


def calibration(pred: dict[str, np.ndarray], rows: dict[str, np.ndarray], turns: int = CALIBRATION_TURNS) -> dict:
    """Calibration branches (rows with ``calibration`` 1): per branch, the critic's implied ``turns``-turn lineage
    value, V(s_0) - Gamma_turns V(s_turns) (the lineage's summed value at the first tail turn minus its discounted
    summed value ``turns`` turns later), against the realised discounted lineage return over those turns, with no
    bootstrap. A lineage that dies sooner has V(s_turns) = 0 and its whole realised return; a branch whose tail
    stopped short with the lineage alive is skipped (``short``)."""
    cal = np.asarray(rows.get("calibration", np.zeros(len(rows["camp"]))), dtype=bool)
    out: dict = {"turns": turns, "target": TARGET, "branches": 0, "short": 0, "kinds": {}}
    if not cal.any():
        return out
    components = data.reward_components(rows)
    world, turn = rows["world"], rows["turn"]
    found: dict[int, dict[str, list[float]]] = {}
    for w in sorted(set(world[cal].tolist())):
        m = world == w
        ts = sorted(set(turn[m].tolist()))
        if len(ts) > turns:
            window, end = ts[:turns], ts[turns]
        elif rows["terminated"][m & (turn == ts[-1])].all():
            window, end = ts, None
        else:
            out["short"] += 1
            continue
        kind = int(rows["alt_kind"][m][0])
        biases = found.setdefault(kind, {name: [] for name in pred})
        for name in pred:
            gamma, real = 1.0, 0.0
            for t in window:
                at = m & (turn == t)
                real += gamma * float(components[name][at].sum())
                gamma *= float(data.discount(rows["years"][at][:1])[0])
            v0 = float(pred[name][m & (turn == ts[0])].sum())
            v_end = float(pred[name][m & (turn == end)].sum()) if end is not None else 0.0
            biases[name].append(float(relative(np.asarray([v0 - gamma * v_end]), np.asarray([real]))[0]))
        out["branches"] += 1
    for kind, by_head in sorted(found.items()):
        out["kinds"][KINDS[kind] if 0 <= kind < len(KINDS) else str(kind)] = {
            name: {"mean": float(np.mean(b)), "median": float(np.median(b)), "branches": len(b)} for name, b in by_head.items()}
    return out


def evaluate(pred: dict[str, np.ndarray], real: dict[str, np.ndarray], rows: dict[str, np.ndarray]) -> dict:
    """Both reports. ``pass`` reads the calibration branches only: welfare mean and median within TARGET for native,
    region and mix (each present); the self-consistency report is a diagnostic."""
    cal = calibration(pred, rows)
    gated = [cal["kinds"][KINDS[k]]["welfare"] for k in GATED if KINDS[k] in cal["kinds"]]
    passed = bool(gated) and all(abs(h["mean"]) <= TARGET and abs(h["median"]) <= TARGET for h in gated)
    return {"pass": passed, "calibration": cal, "self_consistency": report(pred, real, rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--branch-shards", required=True, help="comma-separated held-out branch-shard directories")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = upgrade(torch.load(args.checkpoint, map_location="cpu", weights_only=False))
    normalizer = Normalizer.from_state(state["normalizer"])
    critic = load_critic(args.checkpoint)
    rows, boot = branch_shards.load([Path(p) for p in args.branch_shards.split(",")])
    pred, real = realised(critic, rows, boot, normalizer)
    result = {"checkpoint": str(args.checkpoint), **evaluate(pred, real, rows)}
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
