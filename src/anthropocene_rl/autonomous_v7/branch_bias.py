"""Critic bias on search branch states (docs/MODEL9_SEARCH_TEACHER.md, *Validation*).

    python -m anthropocene_rl.autonomous_v7.branch_bias --checkpoint <pt> --branch-shards <dir>[,<dir>] [--out bias.json]

On held-out branch shards (search_teacher ``<out>/branch_shards`` from worlds the critic never trained on), compares
the critic's predicted value with the realised return: the tail's discounted lineage rewards per objective plus the
critic's discounted bootstrap after the last tail turn (a lambda = 1 return). Reported per alternative kind and per
objective head as relative bias (prediction - realised) / |realised|, mean and median, at two levels:

* ``entry``: one number per branch, the lineage's summed value at the first tail turn against its summed realised
  return. This is the state the search scores at the end of a horizon, so it is the bias the search would inherit;
* ``rows``: every tail row.

The spec's target is an entry bias within 10% for native, region and mix branches (``pass``).
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
    gated = [out["kinds"][KINDS[k]] for k in GATED if KINDS[k] in out["kinds"]]
    out["pass"] = bool(gated) and all(h["welfare"]["entry_median"] is not None and abs(h["welfare"]["entry_median"]) <= TARGET
                                      and abs(h["welfare"]["entry_mean"]) <= TARGET for h in gated)
    return out


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
    result = {"checkpoint": str(args.checkpoint), **report(pred, real, rows)}
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
