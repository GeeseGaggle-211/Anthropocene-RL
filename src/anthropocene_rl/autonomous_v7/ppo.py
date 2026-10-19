"""Stages 2-4 of the model9 curriculum (M7): PPO from the stage-1 checkpoint.

    python -m anthropocene_rl.autonomous_v7.ppo --init <stage1.pt> --stage capability --seeds 1000-1599 --out <dir>

Every village of a world acts with the same shared policy. Actions are sampled from the actor's own heads and their
latent draws are stored, so a minibatch recomputes exact log-probabilities:

* unit strategic controls and service levels: logistic-normal (Gaussian in logit space);
* economy mix (cultivation, herding, fishing): additive logistic-normal on the available coordinates;
* clothing and shelter routes, and (stage 4) movement: masked categorical;
* fission stays native until stages 5-6 and movement until stage 4.

Rewards are the stage-1 lineage rewards (decision D4) and returns use the physical discount exp(-years / 400) with
GAE(lambda). The critic keeps its value equation (u = G / P, two-hot) and its critic-only privileged inputs. A
frozen copy of the stage-1 actor anchors the strategic and service heads with a KL penalty; posture (economy mix)
and routes carry no teacher anchor. The guard on settlements below 30 residents (young daughters exempt) is a
constraint: a separate cost critic and a Lagrange multiplier, not a reward bonus, so being small is never rewarded or
punished beyond the guard.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import contracts, data
from .bridge import Model9Bridge, observation_batch
from .model import MASKED, Actor, Critic, Normalizer, VectorCritic, to_tensors

SMALL_SETTLEMENT = 30
YOUNG_DAUGHTER_TURNS = 3


@dataclass(frozen=True)
class Stage:
    name: str
    movement: bool          # the policy chooses movement (else native)
    routes: bool            # the policy chooses clothing/shelter routes (else native)
    description: str
    # Teacher KL on retained field/herd and development levels (service levels always). Freeing them in stage 3 drove the
    # development budget up ~30% and closed-loop welfare down 12% within 20 iterations, so every stage keeps the
    # anchor; the economy mix (posture) and routes stay unanchored.
    anchor_strategic: bool = True
    # Teacher KL on the clothing and shelter route categoricals (stage-3 regression isolation).
    anchor_routes: bool = False
    # Contract 9: the policy chooses a capital commitment span (--commitments); otherwise every turn decides for
    # itself (span 1), exactly as contract 8.
    commitments: bool = False


STAGES = {
    "capability": Stage("capability", movement=False, routes=False,
                        description="stage 2: single-capability worlds; economy levels and mix, native routes and movement"),
    "mixed": Stage("mixed", movement=False, routes=True, anchor_strategic=True,
                   description="stage 3: mixed-use worlds; routes and posture, no teacher anchor on either"),
    # Stage-3 regression isolation: the mixed-use curriculum with native routes (route heads untrained), and with
    # learned routes anchored to the teacher.
    "mixed-native-routes": Stage("mixed-native-routes", movement=False, routes=False,
                                 description="stage 3 curriculum, native clothing/shelter routes"),
    "mixed-anchored-routes": Stage("mixed-anchored-routes", movement=False, routes=True, anchor_routes=True,
                                   description="stage 3 curriculum, learned routes with a teacher KL anchor"),
    "movement": Stage("movement", movement=True, routes=True, anchor_strategic=True,
                      description="stage 4: the policy moves settlements over value-ordered sites"),
}


# --------------------------------------------------------------------------------------------- distributions
def _normal_logp(z: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    return -0.5 * ((z - mean) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)


def _mix_reference(mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    return (torch.arange(mask.shape[1], device=mask.device) * m).argmax(dim=1, keepdim=True)


def _mix_parts(out: dict[str, torch.Tensor], mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Means of the free log-ratios, their mask, and the reference coordinate."""
    ref = _mix_reference(mask)
    mu = out["mix_logits"] - out["mix_logits"].gather(1, ref)
    free = mask.float().clone(); free.scatter_(1, ref, 0.0)
    return mu, free, ref


def explore(out: dict[str, torch.Tensor], floor: float) -> dict[str, torch.Tensor]:
    """Behaviour cloning drives the logistic-normal scales toward the clamp (the teacher is near-deterministic).
    PPO samples, recomputes and anchors with the same floored scales, so ratios stay bounded."""
    out = dict(out)
    for key in ("strategic_log_std", "service_log_std", "mix_log_std"):
        out[key] = out[key].clamp(min=floor)
    return out


def mix_from_ratios(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = torch.where(mask, z, torch.full_like(z, MASKED))
    return torch.softmax(logits, dim=1) * mask.float()


def sample(out: dict[str, torch.Tensor], t: dict[str, torch.Tensor], stage: Stage,
           generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Draw one action per row; returns the latent draws needed to recompute log-probabilities."""
    def gauss(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        return mean + log_std.exp() * torch.randn(mean.shape, generator=generator, device=mean.device)

    def categorical(logits: torch.Tensor) -> torch.Tensor:
        return torch.multinomial(torch.softmax(logits, dim=1), 1, generator=generator).squeeze(1)

    mask = t["economy_mix_mask"]
    mu, free, _ = _mix_parts(out, mask)
    mix_z = torch.where(free > 0, gauss(mu, out["mix_log_std"]), torch.zeros_like(mu))
    draw = {
        "strategic_z": gauss(out["strategic_mean"], out["strategic_log_std"]),
        "service_z": gauss(out["service_mean"], out["service_log_std"]),
        "mix_z": mix_z,
        "clothing": categorical(out["clothing"]) if stage.routes else torch.full((len(mu),), -1, dtype=torch.long),
        "shelter": categorical(out["shelter"]) if stage.routes else torch.full((len(mu),), -1, dtype=torch.long),
        "movement": categorical(out["movement"]) if stage.movement else torch.full((len(mu),), -1, dtype=torch.long),
        "span": categorical(out["span"]) if stage.commitments else torch.full((len(mu),), contracts.SPAN_SINGLE, dtype=torch.long),
    }
    return draw


def log_prob(out: dict[str, torch.Tensor], t: dict[str, torch.Tensor], draw: dict[str, torch.Tensor],
             stage: Stage) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint log-probability of the policy-chosen parts of an action, and the joint entropy of those heads.

    Contract 9: on a hold row (span 0) the capital controls (strategic levels, economy mix and routes) are the
    committed ones, so they are not part of this turn's action: their draws contribute no log-probability or entropy.
    Service levels and movement are per-turn in every row; the span head counts whenever the stage learns it."""
    half_log_2pie = 0.5 * math.log(2 * math.pi * math.e)
    mu, free, _ = _mix_parts(out, t["economy_mix_mask"])
    capital_lp = _normal_logp(draw["strategic_z"], out["strategic_mean"], out["strategic_log_std"]).sum(dim=1)
    capital_lp = capital_lp + (_normal_logp(draw["mix_z"], mu, out["mix_log_std"]) * free).sum(dim=1)
    capital_ent = (out["strategic_log_std"] + half_log_2pie).sum(dim=1) + ((out["mix_log_std"] + half_log_2pie) * free).sum(dim=1)
    lp = _normal_logp(draw["service_z"], out["service_mean"], out["service_log_std"]).sum(dim=1)
    entropy = (out["service_log_std"] + half_log_2pie).sum(dim=1)

    def categorical(head: str) -> tuple[torch.Tensor, torch.Tensor]:
        logq = torch.log_softmax(out[head], dim=1)
        return logq.gather(1, draw[head].clamp(min=0).unsqueeze(1)).squeeze(1), -(logq.exp() * logq).sum(dim=1)

    if stage.routes:
        for head in ("clothing", "shelter"):
            a, b = categorical(head); capital_lp = capital_lp + a; capital_ent = capital_ent + b
    if stage.movement:
        a, b = categorical("movement"); lp = lp + a; entropy = entropy + b
    decided = torch.ones_like(lp)
    if stage.commitments:
        a, b = categorical("span"); lp = lp + a; entropy = entropy + b
        decided = (draw["span"] != contracts.SPAN_HOLD).float()
    return lp + decided * capital_lp, entropy + decided * capital_ent


def teacher_kl(out: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor], strategic: bool = True,
               routes: bool = False) -> torch.Tensor:
    """KL(policy || teacher) on the anchored heads: service levels, and (stage 2 only) the unit strategic controls.
    The economy mix and routes are never anchored; from stage 3 the retained field/herd levels are free too, so
    pastoral and mixed postures can depart from the field-leaning teacher."""
    def gauss_kl(m1, s1, m2, s2):
        return (s2 - s1) + (torch.exp(2 * s1) + (m1 - m2) ** 2) / (2 * torch.exp(2 * s2)) - 0.5

    kl = gauss_kl(out["service_mean"], out["service_log_std"], teacher["service_mean"], teacher["service_log_std"]).sum(dim=1)
    if strategic:
        kl = kl + gauss_kl(out["strategic_mean"], out["strategic_log_std"], teacher["strategic_mean"], teacher["strategic_log_std"]).sum(dim=1)
    if routes:
        for head in ("clothing", "shelter"):
            logp, logq = torch.log_softmax(out[head], dim=1), torch.log_softmax(teacher[head], dim=1)
            kl = kl + (logp.exp() * (logp - logq)).sum(dim=1)
    return kl


def to_bridge_actions(draw: dict[str, torch.Tensor], t: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    strategic = torch.sigmoid(draw["strategic_z"]).numpy()
    service = torch.sigmoid(draw["service_z"]).numpy()
    mix = mix_from_ratios(draw["mix_z"], t["economy_mix_mask"]).numpy()
    rest = mix[:, 0] + mix[:, 1]
    field_share = np.where(rest > 1e-9, mix[:, 0] / np.maximum(rest, 1e-12), 0.5)
    actions = []
    for i in range(len(strategic)):
        movement = int(draw["movement"][i])
        actions.append({
            "strategic": [*strategic[i].tolist(), float(field_share[i]), float(mix[i, 2])],
            "service": service[i].tolist(),
            "clothingRoute": int(draw["clothing"][i]), "shelterRoute": int(draw["shelter"][i]),
            "movement": movement - 1 if movement >= 0 else contracts.NATIVE_MOVEMENT,
            "fission": -1,
            "commit": contracts.SPANS[int(draw["span"][i])],
        })
    return actions


# --------------------------------------------------------------------------------------------- rollouts
@dataclass
class Rollout:
    """One decision per row, in the shard layout data.py reads, plus the sampled draw and behaviour log-prob."""
    obs: list[dict[str, np.ndarray]] = field(default_factory=list)
    draws: list[dict[str, torch.Tensor]] = field(default_factory=list)
    logp: list[np.ndarray] = field(default_factory=list)
    outcome: dict[str, list] = field(default_factory=lambda: {k: [] for k in data.OUTCOME_KEYS + ["world", "seed", "mask", "cost", "bootstrap_index"]})
    bootstrap_obs: list[dict[str, np.ndarray]] = field(default_factory=list)


def village_batch(villages: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Observation arrays plus the critic-only privileged block (never an actor input)."""
    batch = observation_batch(villages)
    if villages and villages[0].get("privileged") is not None:
        batch["privileged"] = np.asarray([v["privileged"] for v in villages], dtype=np.float32)
    return batch


def _cost(v: dict[str, Any], founded_turn: int, turn: int) -> float:
    young = v["parentSociety"] > 0 and turn - founded_turn <= YOUNG_DAUGHTER_TURNS
    return float(not v["terminated"] and v["population"] < SMALL_SETTLEMENT and not young)


def collect(actor: Actor, normalizer: Normalizer, seeds: list, turns: int, stage: Stage,
            generator: torch.Generator, workers: int = 6, floor: float = -2.0) -> Rollout:
    """Run every world for ``turns`` steps with the sampled shared policy (worlds step in lockstep). A world is a
    seed or a (seed, withheld-technology mask) pair."""
    worlds = [(s, 0) if isinstance(s, int) else (int(s[0]), int(s[1])) for s in seeds]
    seeds = [s for s, _ in worlds]
    bridges = [Model9Bridge() for _ in seeds]
    roll = Rollout()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            steps = list(pool.map(lambda pair: pair[0].reset(pair[1][0], movement=stage.movement, withheld=pair[1][1]),
                                  zip(bridges, worlds)))
            founded = [{v["camp"]: s["turn"] for v in s["villages"]} for s in steps]
            for _ in range(turns):
                live = [(w, v) for w, s in enumerate(steps) for v in s["villages"]
                        if v.get("observation") is not None and not v.get("terminated")]
                if not live:
                    break
                batch = village_batch([v for _, v in live])
                t = to_tensors(batch, normalizer)
                with torch.no_grad():
                    out = explore(actor(t), floor)
                    draw = sample(out, t, stage, generator)
                    lp, _ = log_prob(out, t, draw, stage)
                actions = to_bridge_actions(draw, t)
                per_world: list[dict[int, dict]] = [{} for _ in seeds]
                for (w, v), a in zip(live, actions):
                    per_world[w][int(v["camp"])] = a
                turn_before = [s["turn"] for s in steps]
                steps = list(pool.map(lambda pair: pair[0].step(pair[1]), zip(bridges, per_world)))
                row_of = {}
                for k, (w, v) in enumerate(live):
                    row_of[(w, int(v["camp"]))] = k
                offset = sum(len(o["core"]) for o in roll.obs)
                kept = []
                for w, s in enumerate(steps):
                    for v in s["villages"]:
                        founded[w].setdefault(v["camp"], s["turn"])
                        k = row_of.get((w, int(v["camp"])))
                        if k is None:
                            continue
                        kept.append(k)
                        o = roll.outcome
                        # "world" is this iteration's world index: a seed drawn twice (say as a default world and a
                        # fixture) must not merge sequences.
                        o["world"].append(w); o["seed"].append(seeds[w]); o["mask"].append(worlds[w][1])
                        o["camp"].append(v["camp"]); o["turn"].append(turn_before[w])
                        o["society"].append(v["society"]); o["parent_society"].append(v["parentSociety"])
                        o["prior_population"].append(v["priorPopulation"]); o["population"].append(v["population"])
                        o["qol"].append(v["qol"]); o["nutrition"].append(v["nutrition"]); o["years"].append(v["years"])
                        o["welfare"].append(v["welfare"]); o["terminated"].append(v["terminated"])
                        o["cost"].append(_cost(v, founded[w][v["camp"]], s["turn"]))
                        o["bootstrap_index"].append(-1)
                order = np.asarray(kept, dtype=np.int64)
                roll.obs.append({key: value[order] for key, value in batch.items()})
                roll.draws.append({key: value[order] for key, value in draw.items()})
                roll.logp.append(lp.numpy()[order])
            # Bootstrap observations for sequences truncated at the rollout horizon.
            tail = [(w, v) for w, s in enumerate(steps) for v in s["villages"]
                    if v.get("observation") is not None and not v.get("terminated")]
            if tail:
                roll.bootstrap_obs.append(village_batch([v for _, v in tail]))
                index = {(w, int(v["camp"])): i for i, (w, v) in enumerate(tail)}
                o = roll.outcome
                last: dict[tuple[int, int], int] = {}
                for row, key in enumerate(zip(o["world"], o["camp"])):
                    last[key] = row
                for key, row in last.items():
                    if key in index and not o["terminated"][row]:
                        o["bootstrap_index"][row] = index[key]
    finally:
        for b in bridges:
            b.close()
    return roll


def flatten(roll: Rollout) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor], np.ndarray]:
    obs = {key: np.concatenate([o[key] for o in roll.obs]) for key in roll.obs[0]}
    draws = {key: torch.cat([d[key] for d in roll.draws]) for key in roll.draws[0]}
    outcome = {key: np.asarray(value) for key, value in roll.outcome.items()}
    obs.update(outcome)
    return obs, draws, np.concatenate(roll.logp)


# --------------------------------------------------------------------------------------------- advantages
def values(critic: Critic, batch: dict[str, np.ndarray], normalizer: Normalizer) -> np.ndarray:
    """Raw-unit state values V = u x P, P the residents at the decision."""
    with torch.no_grad():
        u = critic.expectation(critic(to_tensors(batch, normalizer))).numpy().astype(np.float64)
    return u * np.maximum(1.0, batch["prior_population"].astype(np.float64))


def vector_values(critic: VectorCritic, batch: dict[str, np.ndarray], normalizer: Normalizer) -> dict[str, np.ndarray]:
    """Raw-unit state values per objective component."""
    with torch.no_grad():
        logits = critic(to_tensors(batch, normalizer))
    scale = np.maximum(1.0, batch["prior_population"].astype(np.float64))
    return {name: critic.expectation(value).numpy().astype(np.float64) * scale for name, value in logits.items()}


def combine(advantages: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """The policy's advantage: the weighted sum of component advantages (Part 4 of the commitments design)."""
    return sum(float(weights.get(name, 0.0)) * value for name, value in advantages.items())


def gae(batch: dict[str, np.ndarray], reward: np.ndarray, value: np.ndarray, bootstrap: np.ndarray,
        lam: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """GAE(lambda) per (world, camp) sequence with the physical per-turn discount. ``bootstrap`` is the value of
    the observation after each row's decision when a sequence is truncated (0 elsewhere)."""
    advantage = np.zeros(len(reward)); target = np.zeros(len(reward))
    for rows in data.sequences(batch):
        a_next = 0.0
        for position in range(len(rows) - 1, -1, -1):
            i = rows[position]
            gamma = float(data.discount(batch["years"][i]))
            if batch["terminated"][i]:
                v_next, a_next = 0.0, 0.0
            elif position == len(rows) - 1:
                v_next, a_next = bootstrap[i], 0.0
            else:
                v_next = value[rows[position + 1]]
            delta = reward[i] + gamma * v_next - value[i]
            advantage[i] = delta + gamma * lam * a_next
            target[i] = advantage[i] + value[i]
            a_next = advantage[i]
    return advantage, target


class CostCritic(nn.Module):
    """Discounted guard cost (settlement-turns below 30 residents) per decision; plain regression."""

    def __init__(self, core_width: int, width: int = 128) -> None:
        super().__init__()
        from .model import Encoder
        self.encoder = Encoder(core_width, width)
        self.head = nn.Linear(width, 1)

    def forward(self, t: dict[str, torch.Tensor]) -> torch.Tensor:
        h, _ = self.encoder(t)
        return self.head(h).squeeze(1)


# --------------------------------------------------------------------------------------------- update
@dataclass
class Hyper:
    clip: float = 0.2
    epochs: int = 4
    batch: int = 1024
    lr: float = 1e-5               # measured: approximate KL 0.004-0.015 per update on the stage-1 checkpoint
    critic_lr: float = 1e-4
    entropy: float = 1e-3
    teacher_kl: float = 0.05
    cost_budget: float = 0.01      # tolerated share of guarded settlement-turns below 30
    lagrange_lr: float = 0.5
    target_kl: float = 0.03        # early stop on approximate KL to the behaviour policy
    log_std_floor: float = -2.0    # exploration floor on logit-space scales (std 0.135)
    balance_regimes: bool = True   # standardize advantages per world regime and give each regime equal weight
    regime_weight_cap: float = 4.0 # per-decision regime weights are clipped to [1 / cap, cap]
    mix_change_cost: float = 0.02  # reward cost per unit of turn-to-turn total-variation change in the economy mix,
                                   # as a share of the village's welfare that turn (a slight switching penalty)
    break_cost: float = 0.01       # contract 9: reward cost of a voluntary break (re-deciding while a hold was legal),
                                   # as a share of the village's welfare that turn
    objective_weights: dict = field(default_factory=lambda: dict(contracts.OBJECTIVE_WEIGHTS))


def update(actor: Actor, critic: Critic, cost_critic: CostCritic, teacher: Actor, normalizer: Normalizer,
           batch: dict[str, np.ndarray], draws: dict[str, torch.Tensor], old_logp: np.ndarray,
           advantage: np.ndarray, target_u: np.ndarray, cost_adv: np.ndarray, cost_target: np.ndarray,
           lagrange: float, stage: Stage, hyper: Hyper, optimizers: tuple, rng: np.random.Generator,
           regimes: np.ndarray | None = None) -> dict[str, float]:
    actor_opt, critic_opt, cost_opt = optimizers
    n = len(advantage)
    combined = advantage / max(1e-8, advantage.std()) - lagrange * cost_adv / max(1e-8, cost_adv.std() + 1e-8)
    combined, weight = balance(combined, regimes if hyper.balance_regimes else None, hyper.regime_weight_cap)
    stats: dict[str, float] = {}
    for epoch in range(hyper.epochs):
        approx = []
        for start in range(0, n, hyper.batch):
            index = rng.permutation(n)[: hyper.batch] if n > hyper.batch else np.arange(n)
            sub = {k: v[index] for k, v in batch.items()}
            t = to_tensors(sub, normalizer)
            out = explore(actor(t), hyper.log_std_floor)
            d = {k: v[torch.as_tensor(index)] for k, v in draws.items()}
            lp, ent = log_prob(out, t, d, stage)
            old = torch.as_tensor(old_logp[index], dtype=torch.float32)
            ratio = torch.exp(lp - old)
            adv = torch.as_tensor(combined[index], dtype=torch.float32)
            w = torch.as_tensor(weight[index], dtype=torch.float32)
            policy = -(w * torch.min(ratio * adv, ratio.clamp(1 - hyper.clip, 1 + hyper.clip) * adv)).sum() / w.sum()
            with torch.no_grad():
                anchor = explore(teacher(t), hyper.log_std_floor)
            kl = teacher_kl(out, anchor, stage.anchor_strategic, stage.anchor_routes).mean()
            loss = policy - hyper.entropy * ent.mean() + hyper.teacher_kl * kl
            actor_opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 1.0); actor_opt.step()
            logits = critic(t)
            if isinstance(target_u, dict):
                value_loss = sum(-(critic.two_hot(torch.as_tensor(target_u[name][index], dtype=torch.float32)) *
                                   torch.log_softmax(logits[name], dim=-1)).sum(-1).mean() for name in target_u)
            else:
                value_loss = -(critic.two_hot(torch.as_tensor(target_u[index], dtype=torch.float32)) * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
            critic_opt.zero_grad(); value_loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(), 1.0); critic_opt.step()
            cost_loss = ((cost_critic(t) - torch.as_tensor(cost_target[index], dtype=torch.float32)) ** 2).mean()
            cost_opt.zero_grad(); cost_loss.backward(); nn.utils.clip_grad_norm_(cost_critic.parameters(), 1.0); cost_opt.step()
            approx.append(float((old - lp).mean().detach()))
            stats = {"policy": float(policy.detach()), "teacher_kl": float(kl.detach()), "entropy": float(ent.mean().detach()),
                     "value": float(value_loss.detach()), "cost_value": float(cost_loss.detach())}
            if n <= hyper.batch:
                break
        stats["approx_kl"] = float(np.mean(approx)); stats["epochs"] = epoch + 1
        if stats["approx_kl"] > hyper.target_kl:
            break
    return stats


def iteration(actor: Actor, critic: Critic, cost_critic: CostCritic, teacher: Actor, normalizer: Normalizer,
              seeds: list[int], turns: int, stage: Stage, hyper: Hyper, optimizers: tuple, lagrange: float,
              generator: torch.Generator, rng: np.random.Generator,
              grassland: frozenset = frozenset()) -> tuple[float, dict[str, float]]:
    roll = collect(actor, normalizer, seeds, turns, stage, generator, floor=hyper.log_std_floor)
    batch, draws, old_logp = flatten(roll)
    reward = data.lineage_rewards(batch)
    change = mix_changes(batch, draws)
    reward = reward - hyper.mix_change_cost * change * np.maximum(0.0, batch["welfare"].astype(np.float64))
    regimes = np.asarray([regime(int(s), int(m), grassland) for s, m in zip(batch["seed"], batch["mask"])])
    span = draws["span"].numpy()
    hold_legal = batch.get("commit_hold_mask", np.zeros(len(reward), bool)).astype(bool)
    voluntary_break = hold_legal & (span != contracts.SPAN_HOLD) if stage.commitments else np.zeros(len(reward), bool)
    reward = reward - hyper.break_cost * voluntary_break * np.maximum(0.0, batch["welfare"].astype(np.float64))
    rows = np.flatnonzero(batch["bootstrap_index"] >= 0)
    if isinstance(critic, VectorCritic):
        # Each objective component: its own reward, value, bootstrap and GAE; the policy follows their weighted sum.
        # The costs above are in welfare units and stay on the welfare component.
        components = data.reward_components(batch); components["welfare"] = reward
        value_by = vector_values(critic, batch, normalizer)
        tails = {}
        if roll.bootstrap_obs:
            with torch.no_grad():
                logits = critic(to_tensors(roll.bootstrap_obs[0], normalizer))
            tails = {name: critic.expectation(v).numpy() for name, v in logits.items()}
        advantages, target_by = {}, {}
        for name in critic.heads:
            boot = np.zeros(len(reward))
            if name in tails:
                boot[rows] = tails[name][batch["bootstrap_index"][rows]] * np.maximum(1.0, batch["population"][rows])
            advantages[name], target_by[name] = gae(batch, components[name], value_by[name], boot)
        advantage = combine(advantages, hyper.objective_weights)
        target = {name: data.scale_free(value, batch) for name, value in target_by.items()}
    else:
        value = values(critic, batch, normalizer)
        bootstrap = np.zeros(len(reward))
        if roll.bootstrap_obs:
            with torch.no_grad():
                u_tail = critic.expectation(critic(to_tensors(roll.bootstrap_obs[0], normalizer))).numpy()
            # The next observation's value per resident times the residents it was taken at (this row's end population).
            bootstrap[rows] = u_tail[batch["bootstrap_index"][rows]] * np.maximum(1.0, batch["population"][rows])
        advantage, target = gae(batch, reward, value, bootstrap)
        target = data.scale_free(target, batch)
    with torch.no_grad():
        cost_value = cost_critic(to_tensors(batch, normalizer)).numpy().astype(np.float64)
    cost_adv, cost_target = gae(batch, batch["cost"].astype(np.float64), cost_value, np.zeros(len(reward)))
    stats = update(actor, critic, cost_critic, teacher, normalizer, batch, draws, old_logp, advantage,
                   target, cost_adv, cost_target, lagrange, stage, hyper, optimizers, rng, regimes)
    mean_cost = float(batch["cost"].mean())
    lagrange = max(0.0, lagrange + hyper.lagrange_lr * (mean_cost - hyper.cost_budget))
    stats.update({"decisions": float(len(reward)), "welfare": float(batch["welfare"].sum()),
                  "mean_cost": mean_cost, "lagrange": lagrange, "mean_qol": float(batch["qol"].mean()),
                  "mix_change": float(change.mean()),
                  "span_shares": {str(s): float((span == i).mean()) for i, s in enumerate(contracts.SPANS)},
                  "hold_legal": float(hold_legal.mean()), "voluntary_breaks": float(voluntary_break.mean()),
                  "regime_decisions": {r: int((regimes == r).sum()) for r in REGIMES if (regimes == r).any()}})
    return lagrange, stats


REGIMES = ("default", "grassland", "herding", "farming", "fishing")


def regime(seed: int, mask: int, grassland: frozenset) -> str:
    """The world regime of a decision: a capability fixture by its withheld mask, else a grassland-rich opening or an
    ordinary world. These are the stage-gate categories."""
    for name, m in contracts.CAPABILITY_MASKS.items():
        if mask == m:
            return name
    return "grassland" if seed in grassland else "default"


def balance(combined: np.ndarray, regimes: np.ndarray | None, cap: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-regime advantage balancing: standardize within each regime and weight every present regime equally (each
    decision's weight n / (K n_r), clipped to [1 / cap, cap] and renormalized to mean 1). Without regimes, the
    previous global standardization with unit weights."""
    if regimes is None or len(set(regimes.tolist())) <= 1:
        out = (combined - combined.mean()) / max(1e-8, combined.std())
        return out, np.ones(len(combined))
    out = np.empty_like(combined); weight = np.empty(len(combined))
    names = sorted(set(regimes.tolist())); n = len(combined)
    for r in names:
        rows = regimes == r
        x = combined[rows]
        out[rows] = (x - x.mean()) / max(1e-8, x.std()) if rows.sum() > 1 else 0.0
        weight[rows] = np.clip(n / (len(names) * rows.sum()), 1.0 / cap, cap)
    return out, weight / weight.mean()


def mix_changes(batch: dict[str, np.ndarray], draws: dict[str, torch.Tensor]) -> np.ndarray:
    """Total-variation distance between each decision's sampled economy mix and the same village's previous one
    (0 at a village's first decision)."""
    mix = mix_from_ratios(draws["mix_z"], torch.as_tensor(batch["economy_mix_mask"], dtype=torch.bool)).numpy()
    change = np.zeros(len(mix))
    for rows in data.sequences(batch):
        for a, b in zip(rows[:-1], rows[1:]):
            change[b] = 0.5 * float(np.abs(mix[b] - mix[a]).sum())
    return change


def world_sampler(stage: Stage, train: np.ndarray, grassland: np.ndarray, rng: np.random.Generator):
    """Curriculum worlds as (seed, withheld-technology mask). Stage 2 draws the three single-capability fixtures
    equally; stages 3-4 draw half ordinary TRAIN worlds, a quarter grassland-rich openings (where herding competes
    with farming) and a quarter rehearsed capability fixtures."""
    masks = list(contracts.CAPABILITY_MASKS.values())

    def draw(n: int) -> list[tuple[int, int]]:
        worlds = []
        for i in range(n):
            if stage.name == "capability":
                worlds.append((int(rng.choice(train)), int(masks[i % len(masks)])))
            elif i % 4 == 1 and len(grassland) > 0:
                worlds.append((int(rng.choice(grassland)), 0))
            elif i % 4 == 3:
                # Rehearsal: a quarter of stage 3-4 worlds are the stage-2 capability fixtures, so single-capability
                # skills (the fishing regression seen after stage 2) are not forgotten.
                worlds.append((int(rng.choice(train)), int(masks[(i // 4) % len(masks)])))
            else:
                worlds.append((int(rng.choice(train)), 0))
        return worlds
    return draw


def evaluate_closed_loop(checkpoint: Path | None, worlds: list[tuple[int, int]], turns: int, workers: int = 6,
                         movement: bool = False, commitments: bool | None = None) -> dict[str, float]:
    """Deterministic closed-loop rollouts (evaluate.rollout) summed over held-out worlds. ``commitments`` overrides
    the checkpoint's own setting (the baseline of a single-turn run evaluates its init single-turn)."""
    from concurrent.futures import ProcessPoolExecutor
    from .evaluate import rollout
    with ProcessPoolExecutor(max_workers=min(workers, len(worlds))) as pool:
        rows = list(pool.map(rollout, [w for w, _ in worlds], [turns] * len(worlds),
                             [None if checkpoint is None else str(checkpoint)] * len(worlds), [m for _, m in worlds],
                             [movement] * len(worlds), [commitments] * len(worlds)))
    population = sum(r["population"] for r in rows)
    return {"welfare": float(sum(r["welfare"] for r in rows)), "population": float(population),
            "qol": float(sum(r["qol"] * r["population"] for r in rows) / max(1, population)),
            "below_30": float(np.mean([r["below_30"] for r in rows])), "extinctions": float(sum(r["extinctions"] for r in rows)),
            "settlements": float(np.mean([r["settlements"] for r in rows])),
            "interventions_per_decision": float(np.mean([r["interventions_per_decision"] for r in rows])),
            "mix_change": float(np.mean([r.get("mix_change", 0.0) for r in rows])),
            "mix_switches": float(np.mean([r.get("mix_switches", 0.0) for r in rows])),
            "hold_share": float(np.mean([r.get("hold_share", 0.0) for r in rows])),
            "commit_share": float(np.mean([r.get("commit_share", 0.0) for r in rows]))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, required=True, help="stage-1 or earlier-stage checkpoint")
    parser.add_argument("--teacher", type=Path, default=None, help="anchor actor (default: the stage-1 actor in --init)")
    parser.add_argument("--stage", choices=sorted(STAGES), default="capability")
    parser.add_argument("--seeds", default="1000-1599", help="TRAIN seed range")
    parser.add_argument("--grassland-seeds", type=Path, default=None, help="comma-separated grassland-rich seeds")
    parser.add_argument("--worlds", type=int, default=12, help="worlds per iteration")
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-seeds", default="2020-2025",
                        help="DEVELOPMENT selection seeds for periodic evaluation (disjoint from the stage gate's 2000-2019)")
    parser.add_argument("--no-balance", action="store_true", help="global advantage standardization (pre-pilot)")
    parser.add_argument("--mix-change-cost", type=float, default=None)
    parser.add_argument("--commitments", action="store_true", help="contract 9: learn the capital commitment span")
    parser.add_argument("--vector-critic", action="store_true",
                        help="one value head per objective component; the advantage is their weighted sum")
    parser.add_argument("--objective-weights", default=None, help="name=weight[,name=weight] over contracts.OBJECTIVES")
    parser.add_argument("--break-cost", type=float, default=None)
    parser.add_argument("--experts", action="store_true",
                        help="add capability-conditioned expert adapters (warm start: identical to --init)")
    parser.add_argument("--eval-turns", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--log-std-floor", type=float, default=None)
    parser.add_argument("--lagrange-lr", type=float, default=None)
    parser.add_argument("--cost-budget", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed); torch.set_num_threads(4)
    from .upgrade import upgrade
    state = upgrade(torch.load(args.init, map_location="cpu", weights_only=False))
    core_width = int(state["core_width"]); privileged = int(state.get("privileged_width", 0))
    normalizer = Normalizer.from_state(state["normalizer"])
    experts = bool(state.get("experts", False)) or args.experts
    actor, teacher, critic = Actor(core_width, experts=experts), Actor(core_width), Critic(core_width, privileged_width=privileged)
    missing, unexpected = actor.load_state_dict(state["actor"], strict=False)
    if unexpected or any(not k.startswith("experts.") for k in missing):
        raise ValueError(f"checkpoint does not match the actor: missing {missing}, unexpected {unexpected}")
    if args.vector_critic or state.get("vector_critic"):
        if state.get("vector_critic"):
            vector = VectorCritic(core_width, privileged_width=privileged); vector.load_state_dict(state["critic"]); critic = vector
        else:
            critic.load_state_dict(state["critic"]); critic = VectorCritic.from_scalar(critic, core_width, privileged)
    else:
        critic.load_state_dict(state["critic"])
    anchor = upgrade(torch.load(args.teacher, map_location="cpu", weights_only=False)) if args.teacher else state
    teacher.load_state_dict(anchor.get("teacher_actor", anchor["actor"])); teacher.eval()
    cost_critic = CostCritic(core_width)
    if "cost_critic" in state:
        cost_critic.load_state_dict(state["cost_critic"])
    hyper = Hyper()
    if args.lr is not None:
        hyper.lr = args.lr
    if args.lagrange_lr is not None:
        hyper.lagrange_lr = args.lagrange_lr
    if args.cost_budget is not None:
        hyper.cost_budget = args.cost_budget
    if args.log_std_floor is not None:
        hyper.log_std_floor = args.log_std_floor
    if args.no_balance:
        hyper.balance_regimes = False
    if args.mix_change_cost is not None:
        hyper.mix_change_cost = args.mix_change_cost
    optimizers = (torch.optim.Adam(actor.parameters(), lr=hyper.lr), torch.optim.Adam(critic.parameters(), lr=hyper.critic_lr),
                  torch.optim.Adam(cost_critic.parameters(), lr=hyper.critic_lr))
    a, b = (int(x) for x in args.seeds.split("-"))
    train = np.arange(a, b + 1)
    grassland = np.array([int(x) for x in args.grassland_seeds.read_text().split(",") if x.strip()]) \
        if args.grassland_seeds else np.array([], dtype=int)
    rng = np.random.default_rng(args.seed); generator = torch.Generator().manual_seed(args.seed)
    stage = STAGES[args.stage]; lagrange = float(state.get("lagrange", 0.0))
    if args.commitments:
        import dataclasses
        stage = dataclasses.replace(stage, commitments=True)
    if args.break_cost is not None:
        hyper.break_cost = args.break_cost
    if args.objective_weights:
        for item in args.objective_weights.split(","):
            name, weight = item.split("=")
            if name not in contracts.OBJECTIVES:
                raise ValueError(f"unknown objective {name}")
            hyper.objective_weights[name] = float(weight)
    draw = world_sampler(stage, train, grassland, rng)
    ea, eb = (int(x) for x in args.eval_seeds.split("-"))
    # Selection: default worlds plus two of each capability fixture in every stage; best.pt maximizes the geometric
    # mean of the per-regime welfare ratios, so no regime can be traded away for the majority.
    eval_worlds = [(s, 0) for s in range(ea, eb + 1)]
    eval_worlds += [(s, m) for s in range(ea, min(eb, ea + 1) + 1) for m in contracts.CAPABILITY_MASKS.values()]
    grassland_set = frozenset(int(x) for x in grassland)
    args.out.mkdir(parents=True, exist_ok=True)
    trajectory = args.out / "trajectory.jsonl"
    log: list = []
    baseline = None
    best = -np.inf

    def save(path: Path) -> None:
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "cost_critic": cost_critic.state_dict(),
                    "teacher_actor": teacher.state_dict(), "normalizer": state["normalizer"], "core_width": core_width,
                    "privileged_width": privileged, "contract": contracts.NATIVE_CONTRACT, "stage": stage.name,
                    "commitments": stage.commitments, "experts": experts, "vector_critic": isinstance(critic, VectorCritic),
                    "objective_weights": hyper.objective_weights,
                    "lagrange": lagrange, "log": log}, path)

    def evaluate(checkpoint: Path) -> dict:
        groups: dict[str, list] = {}
        for seed, mask in eval_worlds:
            groups.setdefault(regime(seed, mask, frozenset()), []).append((seed, mask))
        result = {name: evaluate_closed_loop(checkpoint, worlds, args.eval_turns, movement=stage.movement,
                                             commitments=stage.commitments)
                  for name, worlds in groups.items()}
        total = {key: float(sum(r[key] for r in result.values())) for key in ("welfare", "population", "extinctions")}
        total["below_30"] = float(np.mean([r["below_30"] for r in result.values()]))
        total["qol"] = float(sum(r["qol"] * r["population"] for r in result.values()) / max(1.0, total["population"]))
        total["regimes"] = result
        return total

    if args.eval_every > 0:
        baseline = evaluate(args.init)
        with trajectory.open("a") as f:
            f.write(json.dumps({"baseline": baseline, "eval_worlds": eval_worlds}) + "\n")
        print(json.dumps({"baseline": baseline}), flush=True)
    for it in range(args.iterations):
        clock = time.time()
        worlds = draw(args.worlds)
        lagrange, stats = iteration(actor, critic, cost_critic, teacher, normalizer, worlds, args.turns, stage, hyper,
                                    optimizers, lagrange, generator, rng, grassland_set)
        stats.update({"iteration": it, "seconds": time.time() - clock, "worlds": worlds})
        save(args.out / "latest.pt")
        if args.eval_every > 0 and (it + 1) % args.eval_every == 0:
            ev = evaluate(args.out / "latest.pt")
            ev["welfare_ratio_vs_init"] = ev["welfare"] / max(1e-9, baseline["welfare"])
            ratios = {name: r["welfare"] / max(1e-9, baseline["regimes"][name]["welfare"]) for name, r in ev["regimes"].items()}
            ev["regime_ratios"] = ratios
            ev["selection_score"] = float(np.exp(np.mean(np.log(np.maximum(1e-9, list(ratios.values()))))))
            stats["eval"] = ev
            if ev["selection_score"] > best and ev["extinctions"] <= baseline["extinctions"]:
                best = ev["selection_score"]; save(args.out / "best.pt")
        log.append(stats)
        with trajectory.open("a") as f:
            f.write(json.dumps(stats) + "\n")
        print(json.dumps({k: v for k, v in stats.items() if k != "worlds"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
