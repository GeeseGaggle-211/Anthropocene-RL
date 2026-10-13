"""Explicit v3 fork with immutable value anchors, preserved Adam and physical H."""

import copy
import hashlib

import torch

from anthropocene_rl.policies.autonomous_ppo_v4 import AutonomousPPOV4
from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3


class AutonomousTrainerV4(AutonomousTrainerV3):
    def __init__(self, config, seed, provenance, width=256):
        torch.manual_seed(seed)
        self.model = AutonomousPPOV4(
            width, uniform_mix=config["uniform_mix"], critic_variant=config["critic_variant"]
        )
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
        torch.save(
            dict(
                version="autonomous-ppo-checkpoint-v4",
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
            state["version"] != "autonomous-ppo-checkpoint-v4"
            or state["config"] != self.config
            or state["provenance"] != self.provenance
        ):
            raise ValueError("V4 checkpoint configuration/provenance mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.sampler.set_state(state["sampler"])
        self.shuffle.set_state(state["shuffle"])
        self.episodes, self.transitions, self.updates = (
            state["episodes"],
            state["transitions"],
            state["updates"],
        )

    def warm_start(self, path, *, expected_sha256):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("V4 parent checkpoint digest mismatch")
        state = torch.load(path, weights_only=False)
        if state["version"] != "autonomous-ppo-checkpoint-v3":
            raise ValueError("V4 conversion requires frozen v3 parent")
        allowed = {"version", "model_version", "critic_variant", "critic_lr", "horizon_years"}
        changed = {
            k
            for k in set(state["config"]) | set(self.config)
            if state["config"].get(k) != self.config.get(k)
        }
        if changed - allowed or self.config["horizon_years"] not in (100, 400):
            raise ValueError("Unregistered v4 configuration change")
        if self.config["critic_lr"] != 0.0001:
            raise ValueError("V4 candidate critic LR must be .0001")
        old = AutonomousTrainerV3(state["config"], 0, state["provenance"], state["width"])
        old.restore(path)
        self.model.import_v3(old.model)
        parameters = dict(self.model.named_parameters())
        self.optimizer.state.clear()
        for name, parameter in old.model.named_parameters():
            if parameter in old.optimizer.state:
                target = parameters[name]
                self.optimizer.state[target] = copy.deepcopy(old.optimizer.state[parameter])
                for key, value in old.optimizer.state[parameter].items():
                    actual = self.optimizer.state[target][key]
                    assert (
                        torch.equal(value, actual)
                        if isinstance(value, torch.Tensor)
                        else value == actual
                    )
        for current, previous in zip(
            self.optimizer.param_groups, old.optimizer.param_groups, strict=True
        ):
            current.update(
                {k: copy.deepcopy(v) for k, v in previous.items() if k not in {"params", "lr"}}
            )
        self.sampler.set_state(state["sampler"])
        self.shuffle.set_state(state["shuffle"])
        self.episodes, self.transitions, self.updates = (
            state["episodes"],
            state["transitions"],
            state["updates"],
        )
