"""Bounded CPU PPO with physical-time returns and complete-episode checkpoints."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch

from anthropocene_rl.autonomous_v4.contracts import observation
from anthropocene_rl.autonomous_v4.policy import Action, AutonomousPPO, inputs, to_native
from anthropocene_rl.training.returns import (
    discounted_interval_reward,
    generalized_advantage_estimate,
)


def market_peer(obs):
    """Fixed public market rule, using only own observation; never sees focal action."""
    workforce = np.zeros(13)
    workforce[0] = 0.6 if obs["WorkforceMask"][1] else 1.0
    workforce[1] = 0.4 if obs["WorkforceMask"][1] else 0.0
    activities = obs["ActivityMask"].astype(float)
    activities /= np.maximum(1, activities.sum(-1, keepdims=True))
    activities[0] = 0
    activities[0, 0] = 1
    action = dict(workforce=workforce, activities=activities, controls=np.zeros(3), movement=0)
    contacts = np.flatnonzero(obs["CounterpartyMask"])
    # Offer food for hand tools at a publicly fixed 1:1 price.
    if len(contacts) and obs["GoodMask"][3] and obs["GoodMask"][6] and obs["Stocks"][3] >= 1e-6:
        quantity = max(1e-6, min(1e6, obs["Stocks"][3] * 0.5))
        action["quote"] = dict(
            Counterparty=int(contacts[0]),
            Give=3,
            Want=6,
            GiveQuantity=quantity,
            WantQuantity=quantity,
        )
    return action


class Trainer:
    def __init__(self, config, seed, provenance, width=256):
        torch.manual_seed(seed)
        self.model = AutonomousPPO(width)
        self.config, self.provenance = config, provenance
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": [*self.model.actor.parameters(), *self.model.heads.parameters()],
                    "lr": config["actor_lr"],
                },
                {"params": self.model.critic.parameters(), "lr": config["critic_lr"]},
            ],
            eps=1e-5,
        )
        self.sampler = torch.Generator().manual_seed(seed + 1000)
        self.shuffle = torch.Generator().manual_seed(seed + 2000)
        self.episodes = self.transitions = self.updates = 0

    def checkpoint(self, path):
        # Only called after complete episodes/update; native world continuation is unnecessary.
        torch.save(
            dict(
                version="autonomous-ppo-checkpoint-v1",
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                sampler=self.sampler.get_state(),
                shuffle=self.shuffle.get_state(),
                episodes=self.episodes,
                transitions=self.transitions,
                updates=self.updates,
                config=self.config,
                provenance=self.provenance,
                width=self.model.width,
            ),
            path,
        )

    def restore(self, path):
        state = torch.load(path, weights_only=False)
        if (
            state["version"] != "autonomous-ppo-checkpoint-v1"
            or state["config"] != self.config
            or state["provenance"] != self.provenance
        ):
            raise ValueError("Checkpoint provenance/configuration mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.sampler.set_state(state["sampler"])
        self.shuffle.set_state(state["shuffle"])
        self.episodes, self.transitions, self.updates = (
            state["episodes"],
            state["transitions"],
            state["updates"],
        )

    def update(self, rows):
        obs = {k: torch.cat([r["obs"][k] for r in rows]) for k in rows[0]["obs"]}
        actions = Action.cat([r["action"] for r in rows])
        old = torch.cat([r["logp"] for r in rows])
        returns = torch.tensor([r["return"] for r in rows], dtype=torch.float32)
        advantage = torch.tensor([r["advantage"] for r in rows], dtype=torch.float32)
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
        metrics = []
        actor_parameters = [*self.model.actor.parameters(), *self.model.heads.parameters()]
        critic_parameters = list(self.model.critic.parameters())
        actor_stopped = False
        for _ in range(self.config["epochs"]):
            for ix in torch.randperm(len(rows), generator=self.shuffle).split(
                self.config["minibatch"]
            ):
                _, lp, entropy, value = self.model(
                    {k: v[ix] for k, v in obs.items()}, actions.index(ix)
                )
                log_ratio = lp - old[ix]
                ratio = log_ratio.exp()
                kl = ((ratio - 1) - log_ratio).mean()
                actor_stopped |= float(kl.detach()) > self.config["target_kl"]
                actor_loss = (
                    -torch.minimum(
                        ratio * advantage[ix], ratio.clamp(0.8, 1.2) * advantage[ix]
                    ).mean()
                    - self.config["entropy_coef"] * entropy.mean()
                )
                critic_loss = 0.5 * (value - returns[ix]).square().mean()
                loss = critic_loss + (actor_loss * 0 if actor_stopped else actor_loss)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(
                    actor_parameters, 0.5, error_if_nonfinite=True
                )
                critic_norm = torch.nn.utils.clip_grad_norm_(
                    critic_parameters, 0.5, error_if_nonfinite=True
                )
                # Adam momentum must also stop when the actor KL guard fires.
                if actor_stopped:
                    for p in actor_parameters:
                        p.grad = None
                head_grad = {
                    k: sum(float(p.grad.abs().sum()) for p in h.parameters() if p.grad is not None)
                    for k, h in self.model.heads.items()
                }
                self.optimizer.step()
                metrics.append(
                    dict(
                        loss=float(loss.detach()),
                        kl=float(kl.detach()),
                        actor_grad=float(actor_norm),
                        critic_grad=float(critic_norm),
                        clip_fraction=float(((ratio - 1).abs() > 0.2).float().mean()),
                        head_grad=head_grad,
                        actor_stopped=actor_stopped,
                    )
                )
        self.updates += 1
        self.transitions += len(rows)
        return metrics


def episode(
    model, transport, bank, entry, generator, config, *, collect=False, verify=False, scripted=False
):
    snapshot = (bank / entry["snapshot"]).read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != entry["sha256"]:
        raise ValueError("Snapshot checksum mismatch")
    ids = [entry["focal"]] + ([entry["peer"]] if entry["peer"] else [])
    response = transport.reset_world(snapshot, ids, config["episode_limit"])
    current = {int(k): observation(v) for k, v in response["observations"].items()}
    focal = entry["focal"]
    rows, welfare, elapsed = [], 0.0, 0.0
    events = dict(
        movement=0,
        trades=0,
        fissions=0,
        migrants=0,
        quotes=0,
        focal_trades=0,
        focal_fissions=0,
        focal_migrants=0,
    )
    populations, food, native_seconds = [], [], []
    for _ in range(config["episode_limit"]):
        obs = inputs([current[focal]])
        with torch.no_grad():
            action, logp, _, value = model(obs, generator=generator)
        native = (
            to_native(action.wire_action(), current[focal])
            if not scripted
            else market_peer(current[focal])
        )
        submitted = {focal: native}
        for peer in ids[1:]:
            if peer in current:
                submitted[peer] = market_peer(current[peer])
        response = transport.step_world(submitted)
        result, telemetry = response["result"], response["result"]["Telemetry"]
        current = {int(k): observation(v) for k, v in result["Observations"].items()}
        terminal, truncated = focal not in current, result["Truncated"]
        years, reward = result["ElapsedYears"], result["Rewards"][str(focal)]
        welfare += reward
        elapsed += years
        with torch.no_grad():
            next_value = 0.0 if terminal else float(model.value(inputs([current[focal]])))
        if collect:
            rows.append(
                dict(
                    obs=obs,
                    action=action,
                    logp=logp.detach(),
                    value=float(value),
                    next_value=next_value,
                    elapsed=years,
                    terminal=terminal,
                    truncated=truncated,
                    reward=discounted_interval_reward(
                        reward / years / config["reward_scale"], years, config["horizon_years"]
                    ),
                )
            )
        events["movement"] += telemetry["Moved"][str(focal)]
        events["trades"] += telemetry["Trades"]
        events["fissions"] += telemetry["Fissions"]
        events["migrants"] += telemetry["Migrants"]
        events["quotes"] += int(native.get("quote") is not None)
        events["focal_trades"] += telemetry["SettledTrades"][str(focal)]
        events["focal_fissions"] += telemetry["DaughterSettlements"][str(focal)]
        events["focal_migrants"] += (
            telemetry["Immigrated"][str(focal)] + telemetry["Emigrated"][str(focal)]
        )
        populations.append(telemetry["Population"][str(focal)])
        food.append(telemetry["FoodMet"][str(focal)])
        native_seconds.append(response["nativeStepSeconds"])
        if terminal or truncated:
            break
    if verify:
        transport.verify_replay()
    if collect:
        adv, returns = generalized_advantage_estimate(
            *(
                np.asarray([r[k] for r in rows])
                for k in ["reward", "value", "next_value", "elapsed", "terminal", "truncated"]
            ),
            horizon_years=config["horizon_years"],
            lambda_ref=config["lambda_ref"],
        )
        for r, a, ret in zip(rows, adv, returns, strict=True):
            r.update(advantage=float(a), **{"return": float(ret)})
    return rows, dict(
        id=entry["id"],
        family=entry["family"],
        welfare=welfare / entry["population"],
        survived=not terminal,
        final_population=populations[-1],
        initial_population=entry["population"],
        food_met=float(np.mean(food)),
        elapsed_years=elapsed,
        events=events,
        native_seconds=sum(native_seconds),
        fingerprint=result["Fingerprint"],
    )


def rollout(trainer, transport, bank, entries, *, verify=False):
    rows, summaries = [], []
    for _ in range(trainer.config["episodes_per_rollout"]):
        entry = entries[trainer.episodes % len(entries)]
        batch, summary = episode(
            trainer.model,
            transport,
            bank,
            entry,
            trainer.sampler,
            trainer.config,
            collect=True,
            verify=verify,
        )
        trainer.episodes += 1
        rows.extend(batch)
        summaries.append(summary)
    return rows, summaries


def clone_state(model):
    return copy.deepcopy(model.state_dict())


def state_hash(model):
    digest = hashlib.sha256()
    for k, v in model.state_dict().items():
        digest.update(k.encode())
        digest.update(v.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
