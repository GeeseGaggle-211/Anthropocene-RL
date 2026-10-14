"""Explicit objective-semantic fork preserving all model7 parameters and Adam."""

import copy
import hashlib

import torch

from anthropocene_rl.autonomous_v6.contracts import CONTROL_TRANSFORM, FEATURE_WIDTH, HISTORY_NAMES
from anthropocene_rl.autonomous_v6.reward_objectives import Objective
from anthropocene_rl.policies.autonomous_ppo_v8 import VALUE_CONTEXT, AutonomousPPOV8
from anthropocene_rl.training.autonomous_ppo_v7 import AutonomousTrainerV7


class AutonomousTrainerV8(AutonomousTrainerV7):
    @staticmethod
    def validate_config(config):
        if config.get("model_version") != "autonomous-ppo-model-v8":
            raise ValueError("Model8 objective contract required")
        base = dict(config, model_version="autonomous-ppo-model-v7")
        AutonomousTrainerV7.validate_config(base)
        objective = Objective.from_wire(config["training_objective"])
        if (
            objective.horizon != config["horizon_years"]
            or objective.reward_scale != config["reward_scale"]
        ):
            raise ValueError("Objective physical units differ from PPO")
        if config.get("critic_objective_loss") != "raw-objective-units-fixed-parent-scale":
            raise ValueError("Unspecified objective critic loss")

    def __init__(self, config, seed, provenance, width=256):
        self.validate_config(config)
        torch.manual_seed(seed)
        self.model = AutonomousPPOV8(
            width,
            uniform_mix=config["uniform_mix"],
            critic_variant=config["critic_variant"],
            concentration_floor=config.get("concentration_floor", 0.5),
        )
        self.config, self.provenance = config, provenance
        self.optimizer = torch.optim.Adam(
            [
                dict(
                    params=[*self.model.actor.parameters(), *self.model.heads.parameters()],
                    lr=config["actor_lr"],
                ),
                dict(params=self.model.critic.parameters(), lr=config["critic_lr"]),
            ],
            eps=1e-5,
        )
        self.sampler = torch.Generator().manual_seed(seed + 1000)
        self.shuffle = torch.Generator().manual_seed(seed + 2000)
        self.episodes = self.transitions = self.updates = 0
        self.deadline = None

    def import_v7(self, path, *, expected_sha256):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Changed objective parent")
        state = torch.load(path, map_location="cpu", weights_only=False)
        old = AutonomousTrainerV7(state["config"], 0, state["provenance"], state["width"])
        old.restore(path)
        changed = {
            k
            for k in set(state["config"]) | set(self.config)
            if state["config"].get(k) != self.config.get(k)
        }
        if changed != {"model_version", "training_objective", "critic_objective_loss"}:
            raise ValueError("Objective fork changed another training mechanism")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.model.uniform_mix = state["current_uniform_mix"]
        self.sampler.set_state(state["sampler"])
        self.shuffle.set_state(state["shuffle"])
        self.episodes, self.transitions, self.updates = (
            state["episodes"],
            state["transitions"],
            state["updates"],
        )

    def checkpoint(self, path):
        torch.save(
            dict(
                version="autonomous-ppo-checkpoint-v8",
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
                value_context=VALUE_CONTEXT,
            ),
            path,
        )

    @staticmethod
    def validate_checkpoint(state, config, provenance):
        AutonomousTrainerV8.validate_config(config)
        if (
            state["version"] != "autonomous-ppo-checkpoint-v8"
            or state["config"] != config
            or state["provenance"] != provenance
            or state["value_context"] != VALUE_CONTEXT
        ):
            raise ValueError("Model8 objective/context provenance differs")
        old = dict(
            state,
            version="autonomous-ppo-checkpoint-v7",
            config=dict(config, model_version="autonomous-ppo-model-v7"),
        )
        AutonomousTrainerV7.validate_checkpoint(old, old["config"], provenance)
