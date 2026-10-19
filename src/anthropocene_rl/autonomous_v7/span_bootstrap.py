"""Bootstrap the contract-9 span head from the heuristic's own persistence.

    python -m anthropocene_rl.autonomous_v7.span_bootstrap --init <stage1.pt> --train <train dir> --dev <dev dir> --out <pt>

The init checkpoint is upgraded to contract 9; the teacher shards are extended with span labels, hold masks and the
commitment features the simulator would show (``span_labels.extend``). The whole actor is then fine-tuned with the
stage-1 behaviour-cloning loss plus a span cross-entropy, so the existing heads keep their teacher fit while the
encoder learns to read the commitment block. The critic is carried over unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import contracts, data, span_labels
from .model import Actor, Normalizer, to_tensors
from .train import bc_loss, label_tensors, subset
from .upgrade import upgrade


def loss_with_span(actor: Actor, t: dict[str, torch.Tensor], labels: dict[str, torch.Tensor], span: torch.Tensor,
                   weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    total, parts = bc_loss(actor, t, labels)
    span_loss = nn.functional.cross_entropy(actor(t)["span"], span)
    parts["span"] = float(span_loss.detach())
    return total + weight * span_loss, parts


def span_metrics(actor: Actor, t: dict[str, torch.Tensor], span: np.ndarray) -> dict[str, float]:
    with torch.no_grad():
        predicted = actor(t)["span"].argmax(dim=1).numpy()
    out = {"accuracy": float((predicted == span).mean())}
    for i, s in enumerate(contracts.SPANS):
        rows = span == i
        if rows.any():
            out[f"recall_{s}"] = float((predicted[rows] == i).mean())
        out[f"predicted_{s}"] = float((predicted == i).mean())
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--span-weight", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed); torch.set_num_threads(6)
    state = upgrade(torch.load(args.init, map_location="cpu", weights_only=False))
    normalizer = Normalizer.from_state(state["normalizer"])
    actor = Actor(int(state["core_width"])); actor.load_state_dict(state["actor"])
    load = lambda root: span_labels.extend(data.load(p for p in sorted(root.iterdir()) if p.is_dir()), args.tolerance)
    train, dev = load(args.train), load(args.dev)
    dev_t, dev_l, dev_span = to_tensors(dev, normalizer), label_tensors(dev), dev["span_label"]
    log = []
    with torch.no_grad():
        _, before = bc_loss(actor, dev_t, dev_l)
    entry = {"epoch": -1, "dev": before, "span": span_metrics(actor, dev_t, dev_span)}
    log.append(entry); print(json.dumps(entry), flush=True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    n = len(train["camp"])
    for epoch in range(args.epochs):
        actor.train()
        order = np.random.default_rng(args.seed + epoch).permutation(n)
        for start in range(0, n, args.batch):
            index = order[start:start + args.batch]
            b = subset(train, index)
            loss, _ = loss_with_span(actor, to_tensors(b, normalizer), label_tensors(b),
                                     torch.as_tensor(b["span_label"]), args.span_weight)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 1.0); optimizer.step()
        actor.eval()
        with torch.no_grad():
            _, parts = loss_with_span(actor, dev_t, dev_l, torch.as_tensor(dev_span), args.span_weight)
        entry = {"epoch": epoch, "dev": parts, "span": span_metrics(actor, dev_t, dev_span)}
        log.append(entry); print(json.dumps(entry), flush=True)
    state = dict(state)
    state["actor"] = actor.state_dict(); state["teacher_actor"] = actor.state_dict(); state["commitments"] = True
    state["span_bootstrap"] = {"tolerance": args.tolerance, "epochs": args.epochs, "lr": args.lr, "log": log}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    print(json.dumps({"saved": str(args.out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
