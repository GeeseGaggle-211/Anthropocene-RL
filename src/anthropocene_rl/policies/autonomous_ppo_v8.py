"""Unchanged model7 actor; explicit learner-objective value coordinates."""

import torch

from anthropocene_rl.policies.autonomous_ppo_v7 import AutonomousPPOV7

VALUE_CONTEXT = ("InitialPopulation", "ValueFactor", "ValuePotential", "OwnPopulation")


class AutonomousPPOV8(AutonomousPPOV7):
    def value(self, obs):
        if any(k not in obs for k in VALUE_CONTEXT):
            raise ValueError("Model8 needs persisted own objective-episode context")
        if any(
            obs[k].shape != (len(obs["x"]),) or not torch.isfinite(obs[k]).all()
            for k in VALUE_CONTEXT
        ):
            raise ValueError("Invalid objective-context shape or amount")
        if (obs["InitialPopulation"] <= 0).any() or (obs["OwnPopulation"] < 0).any():
            raise ValueError("Invalid own initial/current population")
        factor, potential = obs["ValueFactor"], obs["ValuePotential"]
        if (
            not torch.isfinite(factor).all()
            or not torch.isfinite(potential).all()
            or (factor <= 0).any()
            or (potential < 0).any()
        ):
            raise ValueError("Invalid objective value coordinates")
        return factor * super().value(obs) - potential
