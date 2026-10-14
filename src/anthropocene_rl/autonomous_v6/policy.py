"""PPO scores latent controls; wire conversion applies the physical map once."""

import numpy as np
import torch

from anthropocene_rl.autonomous_v4.policy import to_native as old_to_native
from anthropocene_rl.autonomous_v5.policy import inputs as old_inputs
from anthropocene_rl.autonomous_v6.contracts import features


def physical_controls(latent):
    x = np.asarray(latent, dtype=np.float64)
    if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError("Invalid latent controls")
    result = np.clip((x - 0.05) / 0.90, 0, 1)
    return np.where(x <= 0.05, 0.0, np.where(x >= 0.95, 1.0, result))


def to_native(action, obs):
    return old_to_native(dict(action, controls=physical_controls(action["controls"])), obs)


def inputs_with_mode(observations, mode="full"):
    if mode not in ("full", "receipt_none"):
        raise ValueError("Unknown receipt mode")
    prefix = old_inputs([dict(o, History=o["History"][:300]) for o in observations])
    prefix["x"] = torch.tensor(np.stack([features(o) for o in observations]))
    if mode == "receipt_none":
        prefix["x"][:, 761:] = 0
    return prefix
