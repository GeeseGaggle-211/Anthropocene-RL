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
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import contracts, data
from .model import (BOUNDED_FEATURE_CONTRACT, LEGACY_FEATURE_CONTRACT, Actor, Critic, Normalizer, VectorCritic,
                    act_deterministic, load_checkpoint_normalizer, logistic_normal_nll, mix_soft_cross_entropy,
                    mix_target, normalizer_feature_contract, to_tensors)


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


def mode_agreement(out: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> dict[str, float]:
    """Fast fidelity proxy: the deterministic (closed-loop evaluation) action against the teacher's, per head: mean
    absolute error of each service level and of the unit strategic controls (after the 0/1 snapping), and agreement
    of the routes. The logistic-normal NLL weights boundary labels by 1 / std^2, so it can move by nats while the mode
    the evaluation plays is unchanged, or the reverse."""
    mode = act_deterministic(out)
    result: dict[str, float] = {}
    vm = labels["service_mask"].numpy().astype(bool)
    err = np.abs(mode["service"] - labels["service_levels"].numpy())
    for j, name in enumerate(contracts.SERVICES):
        if vm[:, j].any():
            result["service_" + name] = float(err[vm[:, j], j].mean())
    u = contracts.UNIT_STRATEGIC
    sm = labels["strategic_mask"].numpy().astype(bool)[:, :u]
    serr = np.abs(mode["strategic"][:, :u] - labels["strategic"].numpy()[:, :u])
    if sm.any():
        result["strategic"] = float(serr[sm].mean())
    for head, label, mask in (("clothing", "clothing_route", "clothing_route_label_mask"),
                              ("shelter", "shelter_route", "shelter_route_label_mask")):
        m = labels[mask].numpy().astype(bool)
        if m.any():
            result[head + "_route"] = float((mode[head][m] == labels[label].numpy()[m]).mean())
    return result


def lr_factor(step: int, total: int, warmup: float, final: float | None) -> float:
    """Learning-rate multiplier: linear warm-up over the first ``warmup`` share of steps, then cosine decay to
    ``final`` of the peak at the last step; ``final`` None keeps the peak after warm-up (the original flat rate)."""
    warm = int(round(warmup * total))
    if step < warm:
        return (step + 1) / warm
    if final is None:
        return 1.0
    progress = (step - warm) / max(1, total - 1 - warm)
    return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_actor(actor: Actor, train: dict[str, np.ndarray], dev: dict[str, np.ndarray], normalizer: Normalizer,
                epochs: int, batch_size: int, lr: float, log: list, warmup: float = 0.0, final_lr: float | None = None,
                encoder_lr_scale: float = 1.0, grad_clip: float = 1.0) -> None:
    """Behaviour cloning. The defaults are the original optimiser (Adam at a flat ``lr``, gradient clip 1.0); ``main``
    turns the schedule on for ``--token`` (docs/MODEL9_TOKENS.md, "Cloning stability"). ``encoder_lr_scale`` sets the
    encoder's peak rate relative to the heads' rate."""
    if encoder_lr_scale == 1.0:
        groups = [{"params": list(actor.parameters()), "peak": lr}]  # one group: the original optimiser layout
    else:
        encoder = list(actor.encoder.parameters())
        ids = {id(p) for p in encoder}
        groups = [{"params": [p for p in actor.parameters() if id(p) not in ids], "peak": lr},
                  {"params": encoder, "peak": lr * encoder_lr_scale}]
    for group in groups:
        group["lr"] = group["peak"]
    optimizer = torch.optim.Adam(groups, lr=lr)
    n = len(train["camp"])
    total = epochs * math.ceil(n / batch_size)
    step = 0
    dev_t, dev_l = to_tensors(dev, normalizer), label_tensors(dev)
    for epoch in range(epochs):
        actor.train()
        order = np.random.default_rng(epoch).permutation(n)
        totals: dict[str, float] = {}
        for start in range(0, n, batch_size):
            index = order[start:start + batch_size]
            b = subset(train, index)
            factor = lr_factor(step, total, warmup, final_lr)
            for group in optimizer.param_groups:
                group["lr"] = group["peak"] * factor
            loss, parts = bc_loss(actor, to_tensors(b, normalizer), label_tensors(b))
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
            optimizer.step(); step += 1
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v * len(index)
        actor.eval()
        with torch.no_grad():
            _, dev_parts = bc_loss(actor, dev_t, dev_l)
            dev_mode = mode_agreement(actor(dev_t), dev_l)
        entry = {"epoch": epoch, "train": {k: v / n for k, v in totals.items()}, "dev": dev_parts, "dev_mode": dev_mode,
                 "lr": lr * lr_factor(step - 1, total, warmup, final_lr)}
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


def critic_inputs(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The arrays a critic reads (observations, privileged block, personality)."""
    keep = set(data.OBSERVATION_KEYS) | {"region_movement_mask", "commit_hold_mask", "privileged", "personality"}
    return {k: v for k, v in batch.items() if k in keep}


def head_values(critic, batch: dict[str, np.ndarray], normalizer: Normalizer, batch_size: int = 4096) -> dict[str, np.ndarray]:
    """Per-head expectations in the critic's own units (a scalar critic as ``welfare``)."""
    from .branch_shards import _head_values
    critic.eval()
    return _head_values(critic, batch, normalizer, batch_size)


def train_critic_heads(critic, train: dict[str, np.ndarray], dev: dict[str, np.ndarray], normalizer: Normalizer,
                       rounds: int, epochs: int, batch_size: int, lr: float, log: list, warm: bool = False,
                       branch: tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None] | None = None) -> dict[str, Any]:
    """Critic pretraining for ``--vector-critic`` and/or ``--branch-shards``: a VectorCritic learns one head per
    contracts.OBJECTIVES from data.reward_components (each in ppo.head_scale units: qol_years raw, the others per
    resident); a scalar Critic learns welfare. Branch rows (critic only) join the training rows with their own
    lambda-returns, bootstrapped by the critic after each branch's last tail turn. Rounds refit the targets with the
    critic's own values, as ``train_critic``."""
    from . import branch_shards
    from .ppo import head_scale
    vector = isinstance(critic, VectorCritic)
    names = list(critic.heads) if vector else ["welfare"]
    sources = [("train", train)] + ([("branch", branch[0])] if branch is not None else [])
    components = {key: data.reward_components(rows) for key, rows in sources}
    components["dev"] = data.reward_components(dev)
    boot = branch[1] if branch is not None else None
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    n_total = sum(len(rows["camp"]) for _, rows in sources)
    steps_per_epoch = math.ceil(n_total / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, rounds * epochs * steps_per_epoch), eta_min=lr * 0.05)

    def raw(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        u = head_values(critic, rows, normalizer)
        return {name: u[name] * head_scale(name, rows["prior_population"]) for name in names}

    def targets(rows: dict[str, np.ndarray], key: str, value: dict[str, np.ndarray] | None,
                bootstrap: dict[str, np.ndarray] | None) -> dict[str, np.ndarray]:
        return {name: data.lambda_returns(rows, components[key][name], None if value is None else value[name],
                                          bootstrap=None if bootstrap is None else bootstrap[name])
                / head_scale(name, rows["prior_population"]) for name in names}

    def branch_bootstrap() -> dict[str, np.ndarray] | None:
        if branch is None:
            return None
        return branch_shards.values(critic, branch[0], boot, normalizer)[1]

    value = {key: raw(rows) for key, rows in sources} if warm else {key: None for key, _ in sources}
    value_dev = raw(dev) if warm else None
    inputs = critic_inputs(train)
    if branch is not None:
        b = critic_inputs(branch[0])
        if "personality" in b and "personality" not in inputs:  # teacher rows act at the default personality
            from .ppo import default_personality
            inputs["personality"] = np.tile(default_personality(), (len(train["camp"]), 1))
        keys = [k for k in inputs if k in b]
        inputs = {k: np.concatenate([inputs[k], b[k]]) for k in keys}
    metrics: dict[str, Any] = {}
    for round_ in range(rounds):
        boot_value = branch_bootstrap() if warm or round_ > 0 else None
        u = {key: targets(rows, key, value[key], boot_value if key == "branch" else None) for key, rows in sources}
        u_dev = targets(dev, "dev", value_dev, None)
        target = {name: torch.as_tensor(np.concatenate([u[key][name] for key, _ in sources]), dtype=torch.float32) for name in names}
        n = len(target[names[0]])
        for epoch in range(epochs):
            critic.train()
            order = np.random.default_rng(1000 * round_ + epoch).permutation(n)
            for start in range(0, n, batch_size):
                index = order[start:start + batch_size]
                logits = critic(to_tensors(subset(inputs, index), normalizer))
                heads = logits if vector else {"welfare": logits}
                loss = sum(-(critic.two_hot(target[name][index]) * torch.log_softmax(heads[name], dim=-1)).sum(dim=-1).mean()
                           for name in names)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step(); scheduler.step()
        value = {key: raw(rows) for key, rows in sources}
        value_dev = raw(dev)
        p_dev = head_values(critic, dev, normalizer)
        metrics = {"round": round_, "dev": {name: critic_metrics(p_dev[name], u_dev[name]) for name in names}}
        if branch is not None:
            p_branch = head_values(critic, branch[0], normalizer)
            metrics["branch"] = {name: critic_metrics(p_branch[name], u["branch"][name]) for name in names}
        log.append({"critic": metrics})
        print(json.dumps({"critic": metrics}), flush=True)
    u_dev = targets(dev, "dev", value_dev, None)
    p_dev = head_values(critic, dev, normalizer)
    final = {name: critic_metrics(p_dev[name], u_dev[name]) for name in names}
    log.append({"critic_final_dev": final})
    return final["welfare"] if not vector else final


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
    parser.add_argument("--token", action="store_true", help="contract 12 token model (docs/MODEL9_TOKENS.md)")
    parser.add_argument("--vector-critic", action="store_true",
                        help="pretrain a VectorCritic: one head per contracts.OBJECTIVES (saved vector_critic: True)")
    # Cloning stability (docs/MODEL9_TOKENS.md): each defaults on for --token and to the original behaviour otherwise.
    parser.add_argument("--lr-warmup", type=float, default=None, help="share of BC steps in linear warm-up (token 0.05, else 0)")
    parser.add_argument("--lr-final", type=float, default=None,
                        help="cosine-decay BC to this share of the peak lr (token 0.05); negative keeps the flat lr (else)")
    parser.add_argument("--encoder-lr-scale", type=float, default=1.0, help="encoder peak lr relative to the heads (BC only)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="BC gradient-norm clip")
    parser.add_argument("--input-guard", type=float, default=None,
                        help="normalizer guard: drop inputs constant in TRAIN, clip the rest to +-this (token 10, else 0 = off)")
    parser.add_argument("--normalization-contract", choices=("legacy", "bounded-v2"), default="legacy",
                        help="new-run bounded physical/history contract; legacy preserves historical checkpoints")
    parser.add_argument("--provenance", choices=("reject", "flag", "allow"), default="reject",
                        help="branch shards that mix sources or hold shards of unknown source: reject, flag or allow")
    parser.add_argument("--branch-shards", default=None,
                        help="comma-separated search branch-shard directories added to critic pretraining only")
    parser.add_argument("--calibration-manifest", type=Path, default=None)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--prior-use-inventory", type=Path, default=None)
    parser.add_argument("--clone-protocol", type=Path, default=None, help="pre-frozen shared guarded-clone cohort manifest")
    parser.add_argument("--preregistration", type=Path, default=None)
    parser.add_argument("--protocol-approval", type=Path, default=None)
    parser.add_argument("--execution-manifest", type=Path, default=None)
    parser.add_argument("--nonqualifying", action="store_true", help="local stability only; emitted weights cannot qualify or select")
    parser.add_argument("--local-execution-design", type=Path, default=None)
    from .checkpoint import parse, restore_actor
    args = parse(parser)
    from .protocol import enforce_clone_execution
    enforce_clone_execution(args)
    if args.provenance != "reject" and args.branch_shards:
        raise ValueError("flag/allow provenance is for analysis and migrations; training requires reject")
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(args.threads)
    clock = time.time()
    from .calibration import manifest, assert_training_rows
    held_manifest = manifest(args.calibration_manifest) if args.calibration_manifest else None
    from .provenance import declared_sources
    declared = declared_sources(args.source_manifest)
    train_report, dev_report = {}, {}
    train = data.load((p for p in args.train.iterdir() if p.is_dir()), strict=True, report=train_report,
                      declared_sources=declared, calibration_manifest=held_manifest)
    dev = data.load((p for p in args.dev.iterdir() if p.is_dir()), strict=True, report=dev_report,
                    declared_sources=declared, calibration_manifest=held_manifest)
    if held_manifest:
        assert_training_rows(held_manifest, train)
        assert_training_rows(held_manifest, dev)
    print(json.dumps({"train": data.summary(train), "dev": data.summary(dev)}), flush=True)
    warmup = args.lr_warmup if args.lr_warmup is not None else (0.05 if args.token else 0.0)
    final_lr = args.lr_final if args.lr_final is not None else (0.05 if args.token else -1.0)
    final_lr = None if final_lr < 0 else final_lr
    guard = args.input_guard if args.input_guard is not None else (10.0 if args.token else 0.0)
    bc_config = {"lr": args.lr, "warmup": warmup, "final_lr": final_lr, "encoder_lr_scale": args.encoder_lr_scale,
                 "grad_clip": args.grad_clip, "input_guard": guard or None}
    if args.normalization_contract == "bounded-v2":
        if args.input_guard not in (None, 10.0):
            raise ValueError("bounded-v2 fixes the input guard at 10; do not override --input-guard")
        guard = 10.0
        bc_config["input_guard"] = guard
        bc_config["normalization_contract"] = "bounded-v2"
    core_width = train["core"].shape[1]
    privileged_width = train["privileged"].shape[1] if "privileged" in train else 0
    # Round 3: the hierarchical region head by default (model.Actor; saved in the actor state); --region-head flat keeps
    # the earlier flat head. Teacher shards carry no region label, so cloning never trains the region head either way.
    actor = Actor(core_width, token=args.token, region_head=args.region_head)
    critic = (VectorCritic if args.vector_critic else Critic)(core_width, privileged_width=privileged_width, token=args.token)
    branch = None
    branch_report: dict = {}
    if args.branch_shards:
        from . import branch_shards
        branch = branch_shards.load([Path(p) for p in args.branch_shards.split(",")], args.provenance, branch_report,
                                    declared_sources=declared, calibration_manifest=held_manifest)
        print(json.dumps({"branch_rows": int(len(branch[0]["camp"])), "branch_provenance": branch_report}), flush=True)
    from .checkpoint import run_manifest, resolve_head
    from .provenance import checkpoint_hash, compatible_sources
    from .lineage import build as build_lineage, rows_families, merge, source_families
    phase = args.out.with_suffix(".bc.pt")
    load_from = args.resume if args.resume is not None else phase if phase.exists() else None
    loading = torch.load(load_from, map_location="cpu", weights_only=False) if load_from else None
    if loading is not None:
        actor.set_region_head(resolve_head(loading, args.region_head, resume=True))
    frozen_run = run_manifest(args, actor, require_init=False)
    frozen_run["teacher_inputs"] = {"training": train_report["digest"], "selection": dev_report["digest"]}
    if args.clone_protocol:
        from .promotion import validate_cohort
        if args.preregistration is None or args.protocol_approval is None:
            raise ValueError("guarded clones require full --preregistration and external --protocol-approval")
        frozen_run["protocol_digest"] = held_manifest["protocol_digest"]
        frozen_run["protocol_evidence"] = {"preregistration": {"path": str(args.preregistration.resolve()), "sha256": checkpoint_hash(args.preregistration)},
            "approval": {"path": str(args.protocol_approval.resolve()), "sha256": checkpoint_hash(args.protocol_approval)}}
        frozen_run["clone_protocol"] = {"path": str(args.clone_protocol.resolve()), "sha256": checkpoint_hash(args.clone_protocol)}
        validate_cohort({"run_manifest": frozen_run, "bc_config": bc_config, "token": args.token,
                         "vector_critic": args.vector_critic, "region_head": actor.region_head, "calibration_manifest": held_manifest})
    elif held_manifest and frozen_run.get("options"):
        raise ValueError("qualification clones require a pre-frozen --clone-protocol")
    offline_compatibility = (compatible_sources(train_report["records"] + dev_report["records"] + branch_report.get("records", []), frozen_run["simulator"],
                                                frozen_run["simulator_compatibility"],
                                                {k: frozen_run[k] for k in ("sim_rules", "sim_statics")}) if frozen_run else None)
    training_families = merge(rows_families(train), source_families(branch_report)) if held_manifest else {}
    selection_families = rows_families(dev) if held_manifest else {}
    from .teacher_data import validate_source
    for record in train_report["records"] + dev_report["records"]:
        validate_source(record, live=frozen_run["simulator"], binary_sha256=frozen_run["bridge_sha256"])
    normalizer = (Normalizer.fit_bounded(train) if args.normalization_contract == "bounded-v2"
                  else Normalizer.fit(train, guard=guard or None))
    print(json.dumps({"bc_config": bc_config, "dropped_inputs": normalizer.dead_names()}), flush=True)
    teacher_inputs = {"training": train_report, "selection": dev_report}
    lineage = None
    log: list = []
    if args.resume is not None:
        state = loading
        restore_actor(actor, state, args.region_head, resume=True); critic.load_state_dict(state["critic"])
        expected_contract = (BOUNDED_FEATURE_CONTRACT if args.normalization_contract == "bounded-v2"
                             else LEGACY_FEATURE_CONTRACT)
        normalizer = load_checkpoint_normalizer(state, expected_feature_contract=expected_contract)
        log = list(state.get("log", []))
        if frozen_run:
            frozen_run["inputs"]["init"] = checkpoint_hash(args.resume)
            frozen_run["initialization"] = "checkpoint"
        lineage = build_lineage(held_manifest, training_families, selection_families,
                                [(checkpoint_hash(args.resume), state, {"path": str(args.resume.resolve()), "sha256": checkpoint_hash(args.resume)})], teacher_inputs=teacher_inputs)
    # Phase checkpoint: cloning is the long phase, so it is saved on its own (atomically) and a rerun with the same
    # --out skips it. A preempted VM then loses at most the critic pretraining.
    if args.resume is None and phase.exists():
        saved = loading
        restore_actor(actor, saved, args.region_head, resume=True); log = list(saved["log"])
        # The actor was cloned against this normalizer (phase files written before the guard carry none: refit).
        if "normalizer" in saved:
            expected_contract = (BOUNDED_FEATURE_CONTRACT if args.normalization_contract == "bounded-v2"
                                 else LEGACY_FEATURE_CONTRACT)
            normalizer = load_checkpoint_normalizer(saved, expected_feature_contract=expected_contract)
        bc_config = saved.get("bc_config", {"phase": "unrecorded"})
        if frozen_run and saved.get("run_manifest") != frozen_run:
            raise ValueError("BC phase lacks matching frozen run provenance; use a new output")
        lineage = saved.get("family_lineage")
        print(json.dumps({"resumed_bc": str(phase), "bc_config": bc_config}), flush=True)
    else:
        if lineage is None:
            lineage = build_lineage(held_manifest, training_families, selection_families, teacher_inputs=teacher_inputs)
        train_actor(actor, train, dev, normalizer, args.bc_epochs, args.batch, args.lr, log, warmup=warmup,
                    final_lr=final_lr, encoder_lr_scale=args.encoder_lr_scale, grad_clip=args.grad_clip)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        from .promotion import seal
        feature_contract = normalizer_feature_contract(normalizer)
        torch.save(seal({"actor": actor.state_dict(), "log": log, "normalizer": normalizer.state_dict(), "bc_config": bc_config,
                    "nonqualifying": args.nonqualifying,
                    "contract": contracts.NATIVE_CONTRACT, "run_manifest": frozen_run,
                    "calibration_manifest": held_manifest, "family_lineage": lineage,
                    **({"feature_contract": feature_contract} if feature_contract is not None else {})}),
                   phase.with_suffix(".tmp"))
        os.replace(phase.with_suffix(".tmp"), phase)
    if args.vector_critic or branch is not None:
        final = train_critic_heads(critic, train, dev, normalizer, args.critic_rounds, args.critic_epochs, args.batch,
                                   args.critic_lr or args.lr, log, warm=args.resume is not None, branch=branch)
    else:
        final = train_critic(critic, train, dev, normalizer, args.critic_rounds, args.critic_epochs, args.batch,
                             args.critic_lr or args.lr, log, warm=args.resume is not None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    extra = {"vector_critic": True} if args.vector_critic else {}
    feature_contract = normalizer_feature_contract(normalizer)
    from .promotion import seal
    torch.save(seal({"actor": actor.state_dict(), "critic": critic.state_dict(), "normalizer": normalizer.state_dict(),
                "core_width": core_width, "privileged_width": privileged_width, "contract": contracts.NATIVE_CONTRACT, "token": args.token,
                "seed": args.seed, "log": log, "bc_config": bc_config, "region_head": actor.region_head,
                "nonqualifying": args.nonqualifying,
                "calibration_manifest": held_manifest,
                "run_manifest": frozen_run, "family_lineage": lineage, "offline_compatibility": offline_compatibility,
                "branch_provenance": branch_report, "declared_source_runs": sorted(declared) if declared is not None else None,
                "critic_dev": final, **({"feature_contract": feature_contract} if feature_contract is not None else {}),
                **extra}), args.out)
    print(json.dumps({"saved": str(args.out), "seconds": time.time() - clock, "critic_dev": final}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
