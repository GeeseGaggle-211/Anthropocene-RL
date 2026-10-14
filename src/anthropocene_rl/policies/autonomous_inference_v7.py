"""Exact model7 checkpoint loading; native6 observation/action authority unchanged."""

import torch

from anthropocene_rl.policies.autonomous_inference_v6 import AutonomousElderPolicyV6
from anthropocene_rl.policies.autonomous_ppo_v7 import AutonomousPPOV7
from anthropocene_rl.training.autonomous_ppo_v7 import AutonomousTrainerV7


class AutonomousElderPolicyV7(AutonomousElderPolicyV6):
    def __init__(self, checkpoint, *, expected_provenance, seed):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        AutonomousTrainerV7.validate_checkpoint(state, state["config"], expected_provenance)
        self.model = AutonomousPPOV7(
            state["width"],
            uniform_mix=state["current_uniform_mix"],
            critic_variant=state["config"]["critic_variant"],
            concentration_floor=state["concentration_floor"],
        )
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.feature_mode = state["config"].get("feature_mode", "full")
        self.generator = torch.Generator().manual_seed(seed)
