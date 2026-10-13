"""V4 inference preserves the knowledge-isolated, sorted WEGO action interface."""

import torch

from anthropocene_rl.autonomous_v4.inference import AutonomousElderPolicy
from anthropocene_rl.policies.autonomous_ppo_v4 import AutonomousPPOV4


class AutonomousElderPolicyV4(AutonomousElderPolicy):
    def __init__(self, checkpoint, *, expected_provenance, seed):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            state["version"] != "autonomous-ppo-checkpoint-v4"
            or state["provenance"] != expected_provenance
            or state["config"]["model_version"] != "autonomous-ppo-model-v4"
        ):
            raise ValueError("V4 inference checkpoint provenance/model mismatch")
        self.model = AutonomousPPOV4(
            state["width"],
            uniform_mix=state["config"]["uniform_mix"],
            critic_variant=state["config"]["critic_variant"],
        )
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.generator = torch.Generator().manual_seed(seed)
