"""Stage 1 of the model9 curriculum: behaviour cloning of the native heuristic and critic pretraining.

    python -m anthropocene_rl.autonomous_v7.train --train <shards> --dev <shards> --out <checkpoint.pt>

Behaviour cloning maximizes the teacher's action likelihood on every head (posture/mixed routes included here for
initialisation; PPO later carries no teacher anchor on them). The factored critic predicts u = G / P from TD(lambda)
targets, refitted for a few rounds with its own bootstrap. No actor update depends on the critic in this stage.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import contracts, data
from .model import Actor, Critic, Normalizer, logistic_normal_nll, mix_soft_cross_entropy, mix_target, to_tensors


def subset(batch: dict[str, np.ndarray], index: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[index] for key, value in batch.items()}


def bc_loss(actor: Actor, t: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    out = actor(t)
    parts: dict[str, torch.Tensor] = {}
    u = contracts.UNIT_STRATEGIC
    sm = labels["strategic_mask"][:, :u].float()
    parts["strategic"] = (logistic_normal_nll(out["strategic_mean"], out["strategic_log_std"], labels["strategic"][:, :u]) * sm).sum() / sm.sum().clamp(min=1)
    # Economy mix: the teacher's split is a label wherever its field or fishing share is (development > 0).
    mix_loss, carried = mix_soft_cross_entropy(out["mix_logits"], mix_target(labels["strategic"]), t["economy_mix_mask"])
    mm = (labels["strategic_mask"][:, u:].any(dim=1) & carried).float()
    parts["mix"] = (mix_loss * mm).sum() / mm.sum().clamp(min=1)
    vm = labels["service_mask"].float()
    parts["service"] = (logistic_normal_nll(out["service_mean"], out["service_log_std"], labels["service_levels"]) * vm).sum() / vm.sum().clamp(min=1)
    ce = nn.functional.cross_entropy
    for head, label, mask, legal in (("clothing", "clothing_route", "clothing_route_label_mask", "clothing_route_mask"),
                                     ("shelter", "shelter_route", "shelter_route_label_mask", "shelter_route_mask"),
                                     ("movement", "movement", "movement_label_mask", "movement_mask")):
        target = labels[label].long().clamp(min=0)
        # A label is used only when the observation's own mask allows the teacher's class.
        allowed = t[legal].gather(1, target.clamp(max=t[legal].shape[1] - 1).unsqueeze(1)).squeeze(1)
        m = labels[mask].float() * allowed.float()
        loss = ce(out[head], target, reduction="none")
        loss = torch.where(m > 0, loss, torch.zeros_like(loss))
        parts[head] = loss.sum() / m.sum().clamp(min=1)
    parts["fission"] = ce(out["fission"], labels["fission_policy"].long().clamp(0, contracts.FISSION_POLICIES - 1))
    total = sum(parts.values())
    return total, {k: float(v.detach()) for k, v in parts.items()}


def label_tensors(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    out = {}
    for key in data.LABEL_KEYS:
        value = batch[key]
        out[key] = torch.as_tensor(value.astype(np.float32) if value.dtype != np.bool_ else value)
    return out


def train_actor(actor: Actor, train: dict[str, np.ndarray], dev: dict[str, np.ndarray], normalizer: Normalizer,
                epochs: int, batch_size: int, lr: float, log: list) -> None:
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    n = len(train["camp"])
    dev_t, dev_l = to_tensors(dev, normalizer), label_tensors(dev)
    for epoch in range(epochs):
        actor.train()
        order = np.random.default_rng(epoch).permutation(n)
        totals: dict[str, float] = {}
        for start in range(0, n, batch_size):
            index = order[start:start + batch_size]
            b = subset(train, index)
            loss, parts = bc_loss(actor, to_tensors(b, normalizer), label_tensors(b))
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v * len(index)
        actor.eval()
        with torch.no_grad():
            _, dev_parts = bc_loss(actor, dev_t, dev_l)
        entry = {"epoch": epoch, "train": {k: v / n for k, v in totals.items()}, "dev": dev_parts}
        log.append(entry)
        print(json.dumps({"bc": entry}), flush=True)


def critic_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    var = float(np.var(target))
    err = pred - target
    return {
        "explained_variance": 1.0 - float(np.var(err)) / max(var, 1e-12),
        "mse_over_variance": float(np.mean(err ** 2)) / max(var, 1e-12),
        "bias": float(np.mean(err)),
        "bias_over_std": float(np.mean(err)) / max(math.sqrt(var), 1e-12),
        "target_mean": float(np.mean(target)),
        "target_std": math.sqrt(var),
    }


def predict(critic: Critic, batch: dict[str, np.ndarray], normalizer: Normalizer, batch_size: int = 4096) -> np.ndarray:
    critic.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(batch["camp"]), batch_size):
            b = subset(batch, np.arange(start, min(start + batch_size, len(batch["camp"]))))
            out.append(critic.expectation(critic(to_tensors(b, normalizer))).numpy())
    return np.concatenate(out).astype(np.float64)


def train_critic(critic: Critic, train: dict[str, np.ndarray], dev: dict[str, np.ndarray], normalizer: Normalizer,
                 rounds: int, epochs: int, batch_size: int, lr: float, log: list, warm: bool = False) -> dict[str, float]:
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    steps_per_epoch = math.ceil(len(train["camp"]) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, rounds * epochs * steps_per_epoch), eta_min=lr * 0.05)
    reward_train, reward_dev = data.lineage_rewards(train), data.lineage_rewards(dev)
    value_train = value_dev = None
    if warm:
        # A resumed critic bootstraps its first targets from its own predictions rather than restarting from MC.
        value_train = predict(critic, train, normalizer) * np.maximum(1.0, train["prior_population"])
        value_dev = predict(critic, dev, normalizer) * np.maximum(1.0, dev["prior_population"])
    metrics: dict[str, float] = {}
    for round_ in range(rounds):
        u_train = data.scale_free(data.lambda_returns(train, reward_train, value_train), train)
        u_dev = data.scale_free(data.lambda_returns(dev, reward_dev, value_dev), dev)
        target = torch.as_tensor(u_train, dtype=torch.float32)
        n = len(u_train)
        for epoch in range(epochs):
            critic.train()
            order = np.random.default_rng(1000 * round_ + epoch).permutation(n)
            for start in range(0, n, batch_size):
                index = order[start:start + batch_size]
                logits = critic(to_tensors(subset(train, index), normalizer))
                loss = -(critic.two_hot(target[index]) * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step(); scheduler.step()
        p_train, p_dev = predict(critic, train, normalizer), predict(critic, dev, normalizer)
        value_train = p_train * np.maximum(1.0, train["prior_population"])
        value_dev = p_dev * np.maximum(1.0, dev["prior_population"])
        metrics = {"round": round_, "train": critic_metrics(p_train, u_train), "dev": critic_metrics(p_dev, u_dev)}
        log.append({"critic": metrics})
        print(json.dumps({"critic": metrics}), flush=True)
    # Final world-disjoint metrics against targets bootstrapped by the trained critic.
    u_dev = data.scale_free(data.lambda_returns(dev, reward_dev, value_dev), dev)
    final = critic_metrics(predict(critic, dev, normalizer), u_dev)
    log.append({"critic_final_dev": final})
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bc-epochs", type=int, default=12)
    parser.add_argument("--critic-rounds", type=int, default=3)
    parser.add_argument("--critic-epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--resume", type=Path, default=None, help="continue from a checkpoint (actor and critic)")
    parser.add_argument("--critic-lr", type=float, default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(args.threads)
    clock = time.time()
    train = data.load(p for p in args.train.iterdir() if p.is_dir())
    dev = data.load(p for p in args.dev.iterdir() if p.is_dir())
    print(json.dumps({"train": data.summary(train), "dev": data.summary(dev)}), flush=True)
    normalizer = Normalizer.fit(train)
    core_width = train["core"].shape[1]
    privileged_width = train["privileged"].shape[1] if "privileged" in train else 0
    actor, critic = Actor(core_width), Critic(core_width, privileged_width=privileged_width)
    log: list = []
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        actor.load_state_dict(state["actor"]); critic.load_state_dict(state["critic"])
        normalizer = Normalizer.from_state(state["normalizer"]); log = list(state.get("log", []))
    train_actor(actor, train, dev, normalizer, args.bc_epochs, args.batch, args.lr, log)
    final = train_critic(critic, train, dev, normalizer, args.critic_rounds, args.critic_epochs, args.batch,
                         args.critic_lr or args.lr, log, warm=args.resume is not None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "normalizer": normalizer.state_dict(),
                "core_width": core_width, "privileged_width": privileged_width, "contract": contracts.NATIVE_CONTRACT, "seed": args.seed, "log": log,
                "critic_dev": final}, args.out)
    print(json.dumps({"saved": str(args.out), "seconds": time.time() - clock, "critic_dev": final}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
