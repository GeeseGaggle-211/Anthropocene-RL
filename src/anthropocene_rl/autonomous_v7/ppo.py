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
from .model import MASKED, Actor, Critic, Normalizer, to_tensors

SMALL_SETTLEMENT = 30
YOUNG_DAUGHTER_TURNS = 3


@dataclass(frozen=True)
class Stage:
    name: str
    movement: bool          # the policy chooses movement (else native)
    routes: bool            # the policy chooses clothing/shelter routes (else native)
    description: str


STAGES = {
    "capability": Stage("capability", movement=False, routes=False,
                        description="stage 2: single-capability worlds; economy levels and mix, native routes and movement"),
    "mixed": Stage("mixed", movement=False, routes=True,
                   description="stage 3: mixed-use worlds; routes and posture, no teacher anchor on either"),
    "movement": Stage("movement", movement=True, routes=True,
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
    }
    return draw


def log_prob(out: dict[str, torch.Tensor], t: dict[str, torch.Tensor], draw: dict[str, torch.Tensor],
             stage: Stage) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint log-probability of the policy-chosen parts of an action, and the joint entropy of those heads."""
    lp = _normal_logp(draw["strategic_z"], out["strategic_mean"], out["strategic_log_std"]).sum(dim=1)
    lp = lp + _normal_logp(draw["service_z"], out["service_mean"], out["service_log_std"]).sum(dim=1)
    mu, free, _ = _mix_parts(out, t["economy_mix_mask"])
    lp = lp + (_normal_logp(draw["mix_z"], mu, out["mix_log_std"]) * free).sum(dim=1)
    entropy = (out["strategic_log_std"] + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=1)
    entropy = entropy + (out["service_log_std"] + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=1)
    entropy = entropy + ((out["mix_log_std"] + 0.5 * math.log(2 * math.pi * math.e)) * free).sum(dim=1)
    heads = [("clothing", stage.routes), ("shelter", stage.routes), ("movement", stage.movement)]
    for head, active in heads:
        if not active:
            continue
        logq = torch.log_softmax(out[head], dim=1)
        lp = lp + logq.gather(1, draw[head].clamp(min=0).unsqueeze(1)).squeeze(1)
        entropy = entropy - (logq.exp() * logq).sum(dim=1)
    return lp, entropy


def teacher_kl(out: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor]) -> torch.Tensor:
    """KL(policy || teacher) on the anchored heads only: unit strategic controls and service levels."""
    def gauss_kl(m1, s1, m2, s2):
        return (s2 - s1) + (torch.exp(2 * s1) + (m1 - m2) ** 2) / (2 * torch.exp(2 * s2)) - 0.5

    kl = gauss_kl(out["strategic_mean"], out["strategic_log_std"], teacher["strategic_mean"], teacher["strategic_log_std"]).sum(dim=1)
    return kl + gauss_kl(out["service_mean"], out["service_log_std"], teacher["service_mean"], teacher["service_log_std"]).sum(dim=1)


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
        })
    return actions


# --------------------------------------------------------------------------------------------- rollouts
@dataclass
class Rollout:
    """One decision per row, in the shard layout data.py reads, plus the sampled draw and behaviour log-prob."""
    obs: list[dict[str, np.ndarray]] = field(default_factory=list)
    draws: list[dict[str, torch.Tensor]] = field(default_factory=list)
    logp: list[np.ndarray] = field(default_factory=list)
    outcome: dict[str, list] = field(default_factory=lambda: {k: [] for k in data.OUTCOME_KEYS + ["world", "cost", "bootstrap_index"]})
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


def collect(actor: Actor, normalizer: Normalizer, seeds: list[int], turns: int, stage: Stage,
            generator: torch.Generator, workers: int = 6) -> Rollout:
    """Run every seed's world for ``turns`` steps with the sampled shared policy (worlds step in lockstep)."""
    bridges = [Model9Bridge() for _ in seeds]
    roll = Rollout()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            steps = list(pool.map(lambda pair: pair[0].reset(pair[1], movement=stage.movement), zip(bridges, seeds)))
            founded = [{v["camp"]: s["turn"] for v in s["villages"]} for s in steps]
            for _ in range(turns):
                live = [(w, v) for w, s in enumerate(steps) for v in s["villages"]
                        if v.get("observation") is not None and not v.get("terminated")]
                if not live:
                    break
                batch = village_batch([v for _, v in live])
                t = to_tensors(batch, normalizer)
                with torch.no_grad():
                    out = actor(t)
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
                        o["world"].append(seeds[w]); o["camp"].append(v["camp"]); o["turn"].append(turn_before[w])
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
                index = {(seeds[w], int(v["camp"])): i for i, (w, v) in enumerate(tail)}
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
    lr: float = 1e-4
    critic_lr: float = 1e-4
    entropy: float = 1e-3
    teacher_kl: float = 0.05
    cost_budget: float = 0.01      # tolerated share of guarded settlement-turns below 30
    lagrange_lr: float = 0.5
    target_kl: float = 0.03        # early stop on approximate KL to the behaviour policy


def update(actor: Actor, critic: Critic, cost_critic: CostCritic, teacher: Actor, normalizer: Normalizer,
           batch: dict[str, np.ndarray], draws: dict[str, torch.Tensor], old_logp: np.ndarray,
           advantage: np.ndarray, target_u: np.ndarray, cost_adv: np.ndarray, cost_target: np.ndarray,
           lagrange: float, stage: Stage, hyper: Hyper, optimizers: tuple, rng: np.random.Generator) -> dict[str, float]:
    actor_opt, critic_opt, cost_opt = optimizers
    n = len(advantage)
    combined = advantage / max(1e-8, advantage.std()) - lagrange * cost_adv / max(1e-8, cost_adv.std() + 1e-8)
    combined = (combined - combined.mean()) / max(1e-8, combined.std())
    stats: dict[str, float] = {}
    for epoch in range(hyper.epochs):
        approx = []
        for start in range(0, n, hyper.batch):
            index = rng.permutation(n)[: hyper.batch] if n > hyper.batch else np.arange(n)
            sub = {k: v[index] for k, v in batch.items()}
            t = to_tensors(sub, normalizer)
            out = actor(t)
            d = {k: v[torch.as_tensor(index)] for k, v in draws.items()}
            lp, ent = log_prob(out, t, d, stage)
            old = torch.as_tensor(old_logp[index], dtype=torch.float32)
            ratio = torch.exp(lp - old)
            adv = torch.as_tensor(combined[index], dtype=torch.float32)
            policy = -torch.min(ratio * adv, ratio.clamp(1 - hyper.clip, 1 + hyper.clip) * adv).mean()
            with torch.no_grad():
                anchor = teacher(t)
            kl = teacher_kl(out, anchor).mean()
            loss = policy - hyper.entropy * ent.mean() + hyper.teacher_kl * kl
            actor_opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 1.0); actor_opt.step()
            logits = critic(t)
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
              generator: torch.Generator, rng: np.random.Generator) -> tuple[float, dict[str, float]]:
    roll = collect(actor, normalizer, seeds, turns, stage, generator)
    batch, draws, old_logp = flatten(roll)
    reward = data.lineage_rewards(batch)
    value = values(critic, batch, normalizer)
    bootstrap = np.zeros(len(reward))
    if roll.bootstrap_obs:
        with torch.no_grad():
            u_tail = critic.expectation(critic(to_tensors(roll.bootstrap_obs[0], normalizer))).numpy()
        rows = np.flatnonzero(batch["bootstrap_index"] >= 0)
        # The next observation's value per resident times the residents it was taken at (this row's end population).
        bootstrap[rows] = u_tail[batch["bootstrap_index"][rows]] * np.maximum(1.0, batch["population"][rows])
    advantage, target = gae(batch, reward, value, bootstrap)
    with torch.no_grad():
        cost_value = cost_critic(to_tensors(batch, normalizer)).numpy().astype(np.float64)
    cost_adv, cost_target = gae(batch, batch["cost"].astype(np.float64), cost_value, np.zeros(len(reward)))
    stats = update(actor, critic, cost_critic, teacher, normalizer, batch, draws, old_logp, advantage,
                   data.scale_free(target, batch), cost_adv, cost_target, lagrange, stage, hyper, optimizers, rng)
    mean_cost = float(batch["cost"].mean())
    lagrange = max(0.0, lagrange + hyper.lagrange_lr * (mean_cost - hyper.cost_budget))
    stats.update({"decisions": float(len(reward)), "welfare": float(batch["welfare"].sum()),
                  "mean_cost": mean_cost, "lagrange": lagrange, "mean_qol": float(batch["qol"].mean())})
    return lagrange, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, required=True, help="stage-1 checkpoint (actor, critic, normalizer)")
    parser.add_argument("--stage", choices=sorted(STAGES), default="capability")
    parser.add_argument("--seeds", default="1000-1599")
    parser.add_argument("--worlds", type=int, default=12, help="worlds per iteration")
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed); torch.set_num_threads(4)
    state = torch.load(args.init, map_location="cpu", weights_only=False)
    core_width = int(state["core_width"]); privileged = int(state.get("privileged_width", 0))
    normalizer = Normalizer.from_state(state["normalizer"])
    actor, teacher, critic = Actor(core_width), Actor(core_width), Critic(core_width, privileged_width=privileged)
    actor.load_state_dict(state["actor"]); teacher.load_state_dict(state["actor"]); teacher.eval()
    critic.load_state_dict(state["critic"])
    cost_critic = CostCritic(core_width)
    hyper = Hyper()
    optimizers = (torch.optim.Adam(actor.parameters(), lr=hyper.lr), torch.optim.Adam(critic.parameters(), lr=hyper.critic_lr),
                  torch.optim.Adam(cost_critic.parameters(), lr=hyper.critic_lr))
    a, b = (int(x) for x in args.seeds.split("-"))
    pool = np.arange(a, b + 1)
    rng = np.random.default_rng(args.seed); generator = torch.Generator().manual_seed(args.seed)
    stage = STAGES[args.stage]; lagrange = 0.0
    args.out.mkdir(parents=True, exist_ok=True)
    log = []
    for it in range(args.iterations):
        clock = time.time()
        seeds = [int(s) for s in rng.choice(pool, size=min(args.worlds, len(pool)), replace=False)]
        lagrange, stats = iteration(actor, critic, cost_critic, teacher, normalizer, seeds, args.turns, stage, hyper,
                                    optimizers, lagrange, generator, rng)
        stats.update({"iteration": it, "seconds": time.time() - clock})
        log.append(stats); print(json.dumps(stats), flush=True)
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "cost_critic": cost_critic.state_dict(),
                    "normalizer": state["normalizer"], "core_width": core_width, "privileged_width": privileged,
                    "contract": contracts.NATIVE_CONTRACT, "stage": stage.name, "lagrange": lagrange, "log": log},
                   args.out / "latest.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
