"""Preserve the old linear expression exactly; learn history through a separate term."""

import torch
from torch import nn
from torch.nn import functional as F

from anthropocene_rl.autonomous_v5.contracts import BASE_FEATURE_WIDTH, HISTORY_WIDTH
from anthropocene_rl.policies.autonomous_ppo_v4 import AutonomousPPOV4


class HistoryLinear(nn.Linear):
    def __init__(self, original):
        super().__init__(
            original.in_features, original.out_features, bias=original.bias is not None
        )
        self.weight = original.weight
        self.bias = original.bias
        self.history_weight = nn.Parameter(torch.zeros(original.out_features, HISTORY_WIDTH))

    def forward(self, x):
        base = F.linear(x[..., :BASE_FEATURE_WIDTH].contiguous(), self.weight, self.bias)
        return base + F.linear(x[..., BASE_FEATURE_WIDTH:], self.history_weight)


class AutonomousPPOV5(AutonomousPPOV4):
    def __init__(self, width=256, *, uniform_mix=0.05, critic_variant="small_step"):
        super().__init__(width, uniform_mix=uniform_mix, critic_variant=critic_variant)
        self.actor[0] = HistoryLinear(self.actor[0])
        self.critic[0][0] = HistoryLinear(self.critic[0][0])
        if critic_variant == "residual_layernorm":
            self.reference_critic[0][0] = HistoryLinear(self.reference_critic[0][0])
            self.initial_normalized_critic[0][0] = HistoryLinear(
                self.initial_normalized_critic[0][0]
            )
            self.reference_critic.requires_grad_(False)
            self.initial_normalized_critic.requires_grad_(False)

    def import_v4(self, parent):
        missing, unexpected = self.load_state_dict(parent.state_dict(), strict=False)
        expected = {n for n in self.state_dict() if n.endswith("history_weight")}
        if set(missing) != expected or unexpected:
            raise ValueError("Unexpected v4-to-v5 parameter map")
        for name in expected:
            self.state_dict()[name].zero_()
        for name, value in parent.state_dict().items():
            if not torch.equal(value, self.state_dict()[name]):
                raise ValueError(f"Changed inherited parameter {name}")
