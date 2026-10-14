"""Sorted WEGO inference with separate village-local history and no hidden state."""

import torch

from anthropocene_rl.autonomous_v5.contracts import observation
from anthropocene_rl.autonomous_v5.policy import inputs_with_mode, to_native
from anthropocene_rl.policies.autonomous_ppo_v5 import AutonomousPPOV5
from anthropocene_rl.training.autonomous_ppo_v5 import AutonomousTrainerV5


class AutonomousElderPolicyV5:
    def __init__(self, checkpoint, *, expected_provenance, seed):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            state["config"].get("model_version") != "autonomous-ppo-model-v5"
            or state["config"].get("observation_version") != 5
        ):
            raise ValueError("V5 inference model/observation contract mismatch")
        AutonomousTrainerV5.validate_checkpoint(state, state["config"], expected_provenance)
        self.model = AutonomousPPOV5(
            state["width"],
            uniform_mix=state["config"]["uniform_mix"],
            critic_variant=state["config"]["critic_variant"],
        )
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.feature_mode = state["config"].get("feature_mode", "full")
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def act_many(self, wire_observations, *, deterministic=False):
        ids = sorted(wire_observations)
        if not ids:
            return {}
        obs = [observation(wire_observations[i]) for i in ids]
        actions, _, _, _ = self.model(
            inputs_with_mode(obs, self.feature_mode),
            generator=self.generator,
            deterministic=deterministic,
        )
        return {camp: to_native(actions.wire_action(i), obs[i]) for i, camp in enumerate(ids)}

    def act(self, wire_observation, *, deterministic=False):
        return self.act_many({0: wire_observation}, deterministic=deterministic)[0]
