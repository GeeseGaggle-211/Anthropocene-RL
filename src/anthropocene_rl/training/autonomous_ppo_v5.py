"""Explicit memory-input fork; preserve old Adam/RNG/counters without refitting."""

import copy
import hashlib

import torch

from anthropocene_rl.autonomous_v5.contracts import FEATURE_WIDTH, HISTORY_NAMES
from anthropocene_rl.policies.autonomous_ppo_v5 import AutonomousPPOV5
from anthropocene_rl.training.autonomous_ppo_v4 import AutonomousTrainerV4


class AutonomousTrainerV5(AutonomousTrainerV4):
    def __init__(self, config, seed, provenance, width=256):
        if (
            config.get("model_version") != "autonomous-ppo-model-v5"
            or config.get("observation_version") != 5
        ):
            raise ValueError("V5 requires the institutional-history observation contract")
        torch.manual_seed(seed)
        self.model = AutonomousPPOV5(
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
                version="autonomous-ppo-checkpoint-v5",
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
                feature_width=FEATURE_WIDTH,
                history_names=HISTORY_NAMES,
            ),
            path,
        )

    @staticmethod
    def validate_checkpoint(state, config, provenance):
        if (
            state["version"] != "autonomous-ppo-checkpoint-v5"
            or state["config"] != config
            or state["provenance"] != provenance
            or state["feature_width"] != FEATURE_WIDTH
            or state["history_names"] != HISTORY_NAMES
        ):
            raise ValueError("V5 checkpoint configuration/provenance/history layout mismatch")

    def restore(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.validate_checkpoint(state, self.config, self.provenance)
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
            raise ValueError("V5 parent checkpoint digest mismatch")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["version"] != "autonomous-ppo-checkpoint-v4":
            raise ValueError("V5 fork requires a frozen model4 parent")
        allowed = {"version", "model_version", "observation_version"}
        changed = {
            k
            for k in set(state["config"]) | set(self.config)
            if state["config"].get(k) != self.config.get(k)
        }
        if changed - allowed:
            raise ValueError("History conversion cannot also change critic/horizon/optimization")
        old = AutonomousTrainerV4(state["config"], 0, state["provenance"], state["width"])
        old.restore(path)
        self.model.import_v4(old.model)
        parameters = dict(self.model.named_parameters())
        self.optimizer.state.clear()
        for name, parameter in old.model.named_parameters():
            if parameter in old.optimizer.state:
                self.optimizer.state[parameters[name]] = copy.deepcopy(
                    old.optimizer.state[parameter]
                )
        for new, previous in zip(
            self.optimizer.param_groups, old.optimizer.param_groups, strict=True
        ):
            new.update({k: copy.deepcopy(v) for k, v in previous.items() if k != "params"})
        self.sampler.set_state(state["sampler"])
        self.shuffle.set_state(state["shuffle"])
        self.episodes, self.transitions, self.updates = (
            state["episodes"],
            state["transitions"],
            state["updates"],
        )
