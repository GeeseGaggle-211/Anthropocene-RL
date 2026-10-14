"""Preserve old coordinates; append the semantic version of the last request."""

import numpy as np

from anthropocene_rl.autonomous_v4.contracts import action_wire as old_action_wire
from anthropocene_rl.autonomous_v5.contracts import HISTORY_NAMES as OLD_NAMES
from anthropocene_rl.autonomous_v5.contracts import features as old_features
from anthropocene_rl.autonomous_v5.contracts import observation as old_observation

ACTION_VERSION = 4
OBSERVATION_VERSION = ENVIRONMENT_VERSION = REPLAY_VERSION = 6
HISTORY_NAMES = (*OLD_NAMES, "previous_action_version")
HISTORY_WIDTH = 301
FEATURE_WIDTH = 1062
CONTROL_TRANSFORM = "clip((latent-.05)/.90,0,1)"


def observation(wire):
    if wire.get("SchemaVersion") != 6:
        raise ValueError("Post-growth observation version mismatch")
    h = np.asarray(wire["History"], dtype=np.float64)
    if h.shape != (301,) or not np.isfinite(h).all() or h[-1] not in (0, 3, 4):
        raise ValueError("Invalid versioned history")
    if (h[0] == 0) != (h[-1] == 0):
        raise ValueError("Absent decision/version disagree")
    result = old_observation(dict(wire, SchemaVersion=5, History=h[:300]))
    result["History"] = h
    return result


def features(obs):
    prefix = old_features(dict(obs, History=obs["History"][:300]))
    return np.concatenate((prefix, np.array([obs["History"][-1] / 4], dtype=np.float32)))


def action_wire(action):
    return dict(old_action_wire(action), SchemaVersion=4)
