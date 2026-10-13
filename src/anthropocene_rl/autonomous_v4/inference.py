"""Version-checked parent-Elder plug-in; actor sees only its own native observation."""

from __future__ import annotations

import torch

from anthropocene_rl.autonomous_v4.contracts import observation
from anthropocene_rl.autonomous_v4.policy import AutonomousPPO, inputs, to_native


class AutonomousElderPolicy:
    def __init__(self, checkpoint, *, expected_provenance: dict, seed: int):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            state["version"] != "autonomous-ppo-checkpoint-v1"
            or state["provenance"] != expected_provenance
        ):
            raise ValueError("Policy checkpoint provenance mismatch")
        self.model = AutonomousPPO(state["width"])
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def act_many(self, wire_observations: dict[int, dict], *, deterministic=False):
        """Canonical batching for a shared WEGO boundary; no cross-agent features."""
        ids = sorted(wire_observations)
        if not ids:
            return {}
        observations = [observation(wire_observations[i]) for i in ids]
        actions, _, _, _ = self.model(
            inputs(observations), generator=self.generator, deterministic=deterministic
        )
        return {
            camp: to_native(actions.wire_action(i), observations[i]) for i, camp in enumerate(ids)
        }

    def act(self, wire_observation: dict, *, deterministic=False):
        return self.act_many({0: wire_observation}, deterministic=deterministic)[0]
