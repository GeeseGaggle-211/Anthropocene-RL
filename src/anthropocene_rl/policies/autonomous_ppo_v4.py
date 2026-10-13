"""Versioned smaller-step and exactly value-preserving residual-LayerNorm critics."""

import copy
from collections import OrderedDict

from torch import nn

from anthropocene_rl.policies.autonomous_ppo_v3 import AutonomousPPOV3

CRITIC_VARIANTS = {"small_step", "residual_layernorm"}


class AutonomousPPOV4(AutonomousPPOV3):
    def __init__(self, width=256, *, uniform_mix=0.05, critic_variant="small_step"):
        if critic_variant not in CRITIC_VARIANTS:
            raise ValueError("Unknown v4 critic variant")
        super().__init__(width, uniform_mix=uniform_mix)
        self.critic_variant = critic_variant
        if critic_variant == "residual_layernorm":
            self.reference_critic = copy.deepcopy(self.critic).requires_grad_(False)
            trunk = self.critic[0]
            self.critic[0] = nn.Sequential(
                OrderedDict(
                    [
                        ("0", trunk[0]),
                        ("norm0", nn.LayerNorm(width, eps=1e-5)),
                        ("1", trunk[1]),
                        ("2", trunk[2]),
                        ("norm1", nn.LayerNorm(width, eps=1e-5)),
                        ("3", trunk[3]),
                    ]
                )
            )
            self.initial_normalized_critic = copy.deepcopy(self.critic).requires_grad_(False)

    def import_v3(self, parent):
        previous = parent.state_dict()
        expected_missing = set(self.state_dict()) - set(previous)
        missing, unexpected = self.load_state_dict(previous, strict=False)
        if set(missing) != expected_missing or unexpected:
            raise ValueError("Unexpected v3-to-v4 parameter mapping")
        if self.critic_variant == "residual_layernorm":
            self.reference_critic.load_state_dict(parent.critic.state_dict())
            self.initial_normalized_critic.load_state_dict(self.critic.state_dict())
        for name, value in previous.items():
            if not value.equal(self.state_dict()[name]):
                raise ValueError(f"Changed inherited parameter: {name}")

    def value(self, obs):
        current = super().value(obs)
        if self.critic_variant == "small_step":
            return current
        reference = self.reference_critic(obs["x"]).squeeze(-1)
        initial = self.initial_normalized_critic(obs["x"]).squeeze(-1)
        # Subtraction must precede addition to preserve every parent float at entry.
        # Frozen parameters still allow the correct derivative with respect to input.
        return reference + (current - initial)
