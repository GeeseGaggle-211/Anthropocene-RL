"""Versioned native action/observation identifiers and strict wire validation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

ACTION_VERSION = 3
OBSERVATION_VERSION = ENVIRONMENT_VERSION = REPLAY_VERSION = 4
ROLES = 13
ACTIVITIES = 15
CONTACTS = 8
GOODS = 11
CONTEXT_WIDTH = 512
FEATURE_WIDTH = CONTEXT_WIDTH + ROLES + ROLES * ACTIVITIES + 3 + CONTACTS + 2 * GOODS + 8


def observation(wire: dict[str, Any]) -> dict[str, NDArray[Any]]:
    if wire.get("SchemaVersion") != OBSERVATION_VERSION:
        raise ValueError("Autonomous observation version mismatch")
    shapes = {
        "Context": (512,),
        "WorkforceMask": (13,),
        "ActivityMask": (13, 15),
        "ControlMask": (3,),
        "CounterpartyMask": (8,),
        "GoodMask": (11,),
        "Stocks": (11,),
        "MovementMask": (8,),
    }
    result = {}
    for key, shape in shapes.items():
        data = np.asarray(wire[key], dtype=np.float64 if key in ("Context", "Stocks") else np.bool_)
        if data.shape != shape or not np.all(np.isfinite(data)):
            raise ValueError(f"Invalid {key}")
        result[key] = data
    return result


def features(obs: dict[str, NDArray[Any]]) -> NDArray[np.float32]:
    # Fixed stateless feature compression; no historical normalizer is overwritten.
    context = np.sign(obs["Context"]) * np.log1p(np.abs(obs["Context"]))
    return np.concatenate(
        [
            context,
            obs["WorkforceMask"],
            obs["ActivityMask"].ravel(),
            obs["ControlMask"],
            obs["CounterpartyMask"],
            obs["GoodMask"],
            np.log1p(obs["Stocks"]),
            obs["MovementMask"],
        ]
    ).astype(np.float32)


def action_wire(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "SchemaVersion": ACTION_VERSION,
        "Workforce": np.asarray(action["workforce"], dtype=np.float64).tolist(),
        "Activities": np.asarray(action["activities"], dtype=np.float64).tolist(),
        "FieldExpansionFraction": float(action["controls"][0]),
        "RetainedHerdFraction": float(action["controls"][1]),
        "Breeding": float(action["controls"][2]),
        "Quote": action.get("quote"),
        "Movement": int(action.get("movement", 0)),
    }
