"""Same requested action distributions, with versioned institutional-history inputs."""

import numpy as np
import torch

from anthropocene_rl.autonomous_v4.policy import Action as Action
from anthropocene_rl.autonomous_v4.policy import to_native as to_native
from anthropocene_rl.autonomous_v5.contracts import features


def inputs(observations):
    return dict(
        x=torch.tensor(np.stack([features(o) for o in observations])),
        **{
            k: torch.tensor(np.stack([o[k] for o in observations]))
            for k in (
                "WorkforceMask",
                "ActivityMask",
                "ControlMask",
                "MovementMask",
                "CounterpartyMask",
                "GoodMask",
                "Stocks",
            )
        },
    )


def inputs_with_mode(observations, mode="full"):
    if mode not in ("full", "receipt_none"):
        raise ValueError("Unknown receipt-history feature mode")
    result = inputs(observations)
    if mode == "receipt_none":
        result["x"][:, 761:] = 0
    return result
