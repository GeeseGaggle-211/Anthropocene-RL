"""Explicit new physical-action contract; inherited optimizer mapping stays exact."""

import torch

from anthropocene_rl.autonomous_v6.contracts import CONTROL_TRANSFORM, FEATURE_WIDTH, HISTORY_NAMES
from anthropocene_rl.policies.autonomous_ppo_v6 import AutonomousPPOV6
from anthropocene_rl.training.autonomous_ppo_v5 import AutonomousTrainerV5


class AutonomousTrainerV6(AutonomousTrainerV5):
    @staticmethod
    def validate_config(config):
        if (
            config.get("model_version") != "autonomous-ppo-model-v6"
            or config.get("observation_version") != 6
            or config.get("feature_mode", "full") not in ("full", "receipt_none")
        ):
            raise ValueError("Model6 configuration/feature contract mismatch")

    def __init__(self, config, seed, provenance, width=256):
        self.validate_config(config)
        if (
            config.get("model_version") != "autonomous-ppo-model-v6"
            or config.get("observation_version") != 6
        ):
            raise ValueError("Model6 contract required")
        torch.manual_seed(seed)
        self.model = AutonomousPPOV6(
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

    def warm_start(self, path, *, expected_sha256):
        original = self.config
        self.config = dict(original, model_version="autonomous-ppo-model-v5", observation_version=5)
        try:
            super().warm_start(path, expected_sha256=expected_sha256)
        finally:
            self.config = original

    def checkpoint(self, path):
        torch.save(
            dict(
                version="autonomous-ppo-checkpoint-v6",
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
                control_transform=CONTROL_TRANSFORM,
            ),
            path,
        )

    @staticmethod
    def validate_checkpoint(state, config, provenance):
        AutonomousTrainerV6.validate_config(config)
        if (
            state["version"] != "autonomous-ppo-checkpoint-v6"
            or state["config"] != config
            or state["provenance"] != provenance
            or state["feature_width"] != FEATURE_WIDTH
            or state["history_names"] != HISTORY_NAMES
            or state["control_transform"] != CONTROL_TRANSFORM
        ):
            raise ValueError("Model6 checkpoint authority/history mismatch")
