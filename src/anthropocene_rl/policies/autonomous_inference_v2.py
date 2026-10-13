"""Explicit v2 loader; inherited isolated WEGO adapter uses the mixed policy."""

import torch

from anthropocene_rl.autonomous_v4.inference import AutonomousElderPolicy
from anthropocene_rl.policies.autonomous_ppo_v2 import AutonomousPPOV2


class AutonomousElderPolicyV2(AutonomousElderPolicy):
    def __init__(self, checkpoint, *, expected_provenance, seed):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            state["version"] != "autonomous-ppo-checkpoint-v2"
            or state["provenance"] != expected_provenance
            or state["config"]["model_version"] != "autonomous-ppo-model-v2"
        ):
            raise ValueError("V2 policy checkpoint provenance/model mismatch")
        self.model = AutonomousPPOV2(state["width"], uniform_mix=state["config"]["uniform_mix"])
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.generator = torch.Generator().manual_seed(seed)
