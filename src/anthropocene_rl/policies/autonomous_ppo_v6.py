"""One extra history discriminator; distributions still score latent requests."""

import torch
from torch import nn

from anthropocene_rl.policies.autonomous_ppo_v5 import AutonomousPPOV5


class AutonomousPPOV6(AutonomousPPOV5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = [self.actor[0], self.critic[0][0]]
        if self.critic_variant == "residual_layernorm":
            layers.extend([self.reference_critic[0][0], self.initial_normalized_critic[0][0]])
        for layer in layers:
            layer.history_weight = nn.Parameter(torch.zeros(layer.out_features, 301))
        if self.critic_variant == "residual_layernorm":
            self.reference_critic.requires_grad_(False)
            self.initial_normalized_critic.requires_grad_(False)
