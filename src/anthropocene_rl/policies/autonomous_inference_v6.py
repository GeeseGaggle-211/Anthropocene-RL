"""Knowledge-isolated native6 WEGO inference with deterministic control mapping."""

import torch

from anthropocene_rl.autonomous_v6.contracts import observation
from anthropocene_rl.autonomous_v6.policy import inputs_with_mode, to_native
from anthropocene_rl.policies.autonomous_ppo_v6 import AutonomousPPOV6
from anthropocene_rl.training.autonomous_ppo_v6 import AutonomousTrainerV6


class AutonomousElderPolicyV6:
    def __init__(self, checkpoint, *, expected_provenance, seed):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        AutonomousTrainerV6.validate_checkpoint(state, state["config"], expected_provenance)
        self.model = AutonomousPPOV6(
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
        action, _, _, _ = self.model(
            inputs_with_mode(obs, self.feature_mode),
            generator=self.generator,
            deterministic=deterministic,
        )
        return {camp: to_native(action.wire_action(i), obs[i]) for i, camp in enumerate(ids)}

    def act(self, wire_observation, *, deterministic=False):
        return self.act_many({0: wire_observation}, deterministic=deterministic)[0]
