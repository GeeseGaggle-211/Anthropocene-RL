"""Versioned simplex floors and fixed TRAIN-only value-output normalization."""

import math

import torch

from anthropocene_rl.policies.autonomous_ppo_v6 import AutonomousPPOV6


class AutonomousPPOV7(AutonomousPPOV6):
    def __init__(self, *args, concentration_floor=0.5, **kwargs):
        if not math.isfinite(concentration_floor) or not 0.05 <= concentration_floor <= 0.5:
            raise ValueError("Workforce/activity concentration floor must be in [.05,.5]")
        super().__init__(*args, **kwargs)
        self.concentration_floor = float(concentration_floor)
        # Beta controls retain .5: only their first coordinate is stored, and
        # lower alpha can round it to 1 before reconstructing the complement.
        self.register_buffer("value_offset", torch.tensor(0.0))
        self.register_buffer("value_scale", torch.tensor(1.0))

    def concentrations(self, raw):
        base = torch.nn.functional.softplus(raw.clamp(-12, 12))
        return torch.cat([base[:, :208] + self.concentration_floor, base[:, 208:] + 0.5], -1)

    def value(self, obs):
        return self.value_offset + self.value_scale * super().value(obs)

    @torch.no_grad()
    def fit_value_scale(self, training_targets):
        if training_targets.ndim != 1 or len(training_targets) < 2:
            raise ValueError("TRAIN value scaling needs a nondegenerate target vector")
        if not torch.isfinite(training_targets).all():
            raise ValueError("Nonfinite TRAIN value targets")
        self.value_offset.copy_(training_targets.mean())
        self.value_scale.copy_(training_targets.std(unbiased=False).clamp_min(1.0))
        return dict(
            offset=float(self.value_offset),
            scale=float(self.value_scale),
            semantics="fixed TRAIN fitting-world affine output, frozen throughout PPO",
        )
