"""Versioned autonomous PPO remedy; original v4 trainer/checkpoints stay frozen."""

import copy
import hashlib

import torch

from anthropocene_rl.autonomous_v4.policy import Action
from anthropocene_rl.policies.autonomous_ppo_v3 import AutonomousPPOV3


class AutonomousTrainerV3:
    def __init__(self, config, seed, provenance, width=256):
        torch.manual_seed(seed)
        self.model = AutonomousPPOV3(width, uniform_mix=config["uniform_mix"])
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
                version="autonomous-ppo-checkpoint-v3",
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
            state["version"] != "autonomous-ppo-checkpoint-v3"
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
                regularizer = self.model.regularization({k: v[ix] for k, v in obs.items()}).mean()
                actor_loss = actor_loss + self.config["legal_reverse_kl_coef"] * regularizer
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
                        legal_reverse_kl=float(regularizer.detach()),
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

    def warm_start(self, path, *, expected_sha256):
        """Behavior-changing conversion; preserve old Adam state by name, not position."""
        from anthropocene_rl.autonomous_v4.training import Trainer as OldTrainer
        from anthropocene_rl.training.autonomous_ppo_v2 import AutonomousTrainerV2

        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Parent checkpoint digest mismatch")
        state = torch.load(path, weights_only=False)
        classes = {
            "autonomous-ppo-checkpoint-v1": OldTrainer,
            "autonomous-ppo-checkpoint-v2": AutonomousTrainerV2,
        }
        if state["version"] not in classes:
            raise ValueError("LayerNorm conversion requires a v1 or v2 checkpoint")
        old = classes[state["version"]](state["config"], 0, state["provenance"], state["width"])
        old.restore(path)
        missing, unexpected = self.model.load_state_dict(state["model"], strict=False)
        expected = {f"actor.norm{i}.{k}" for i in range(2) for k in ["weight", "bias"]}
        if set(missing) != expected or unexpected:
            raise ValueError("Unexpected architecture conversion")
        new_parameters = dict(self.model.named_parameters())
        self.optimizer.state.clear()
        for name, parameter in old.model.named_parameters():
            target = new_parameters[name]
            if target.shape != parameter.shape:
                raise ValueError(f"Shape changed for {name}")
            if parameter in old.optimizer.state:
                self.optimizer.state[target] = copy.deepcopy(old.optimizer.state[parameter])
                for key, value in old.optimizer.state[parameter].items():
                    actual = self.optimizer.state[target][key]
                    assert (
                        torch.equal(value, actual)
                        if isinstance(value, torch.Tensor)
                        else value == actual
                    )
        # Preserve optimizer hyperparameters other than the explicitly versioned learning rates.
        for new, previous in zip(
            self.optimizer.param_groups, old.optimizer.param_groups, strict=True
        ):
            new.update(
                {k: copy.deepcopy(v) for k, v in previous.items() if k not in {"params", "lr"}}
            )
        self.sampler.set_state(state["sampler"])
        self.shuffle.set_state(state["shuffle"])
        self.episodes, self.transitions, self.updates = (
            state["episodes"],
            state["transitions"],
            state["updates"],
        )
