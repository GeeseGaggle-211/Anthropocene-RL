"""Model7 explicit floor/normalizer provenance; raw-unit GAE and actor PPO unchanged."""

import time

import torch

from anthropocene_rl.autonomous_v6.contracts import CONTROL_TRANSFORM, FEATURE_WIDTH, HISTORY_NAMES
from anthropocene_rl.policies.autonomous_ppo_v7 import AutonomousPPOV7
from anthropocene_rl.training.autonomous_ppo_v6 import AutonomousTrainerV6


class AutonomousTrainerV7(AutonomousTrainerV6):
    @staticmethod
    def validate_config(config):
        if (
            config.get("model_version") != "autonomous-ppo-model-v7"
            or config.get("observation_version") != 6
            or config.get("feature_mode", "full") not in ("full", "receipt_none")
            or config.get("value_normalization") != "fixed-train-affine"
            or not 0.05 <= config.get("concentration_floor", 0.5) <= 0.5
        ):
            raise ValueError("Model7 floor/value/schema configuration mismatch")

    def __init__(self, config, seed, provenance, width=256):
        self.validate_config(config)
        torch.manual_seed(seed)
        self.model = AutonomousPPOV7(
            width,
            uniform_mix=config["uniform_mix"],
            critic_variant=config["critic_variant"],
            concentration_floor=config.get("concentration_floor", 0.5),
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
        self.deadline = None

    def critic_loss(self, value, target):
        return 0.5 * ((value - target) / self.model.value_scale).square().mean()

    def check_update_deadline(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("PPO update exceeded the shared deadline")

    def warm_start(self, *args, **kwargs):
        raise ValueError("Model7 currently requires an independent initialization or exact restore")

    def checkpoint(self, path):
        torch.save(
            dict(
                version="autonomous-ppo-checkpoint-v7",
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
                beta_floor=0.5,
                concentration_floor=self.model.concentration_floor,
                current_uniform_mix=self.model.uniform_mix,
            ),
            path,
        )

    @staticmethod
    def validate_checkpoint(state, config, provenance):
        AutonomousTrainerV7.validate_config(config)
        if (
            state["version"] != "autonomous-ppo-checkpoint-v7"
            or state["config"] != config
            or state["provenance"] != provenance
            or state["feature_width"] != FEATURE_WIDTH
            or state["history_names"] != HISTORY_NAMES
            or state["control_transform"] != CONTROL_TRANSFORM
            or state["beta_floor"] != 0.5
            or state["concentration_floor"] != config.get("concentration_floor", 0.5)
            or not 0 < state["current_uniform_mix"] < 1
            or float(state["model"]["value_scale"]) < 1
            or not torch.isfinite(state["model"]["value_scale"])
            or not torch.isfinite(state["model"]["value_offset"])
        ):
            raise ValueError("Model7 checkpoint normalization/floor/provenance mismatch")

    def restore(self, path):
        super().restore(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.uniform_mix = state["current_uniform_mix"]

    def set_rollout_mix(self):
        """Change exploration only between rollouts, never while scoring PPO rows."""
        schedule = self.config.get("uniform_mix_schedule", [])
        previous = -1
        mixture = self.config["uniform_mix"]
        for boundary, value in schedule:
            if not isinstance(boundary, int) or boundary <= previous or not 0 < value < 1:
                raise ValueError("Invalid uniform-mixture schedule")
            previous = boundary
            if boundary <= self.transitions:
                mixture = value
        self.model.uniform_mix = mixture
