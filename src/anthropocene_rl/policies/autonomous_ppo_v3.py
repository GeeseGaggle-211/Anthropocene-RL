"""Actor-only LayerNorm ablation; normalization precedes each actor tanh."""

from collections import OrderedDict

from torch import nn

from anthropocene_rl.policies.autonomous_ppo_v2 import AutonomousPPOV2


class AutonomousPPOV3(AutonomousPPOV2):
    def __init__(self, width=256, *, uniform_mix=0.05):
        super().__init__(width, uniform_mix=uniform_mix)
        # Keep every original parameter name for auditable optimizer-state conversion.
        self.actor = nn.Sequential(
            OrderedDict(
                [
                    ("0", self.actor[0]),
                    ("norm0", nn.LayerNorm(width)),
                    ("1", self.actor[1]),
                    ("2", self.actor[2]),
                    ("norm1", nn.LayerNorm(width)),
                    ("3", self.actor[3]),
                ]
            )
        )
