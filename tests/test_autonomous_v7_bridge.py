"""Native contract-7 bridge tests (skipped unless the model9 bridge is built)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7.bridge import DEFAULT_BRIDGE, Model9Bridge, observation_batch

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE", str(DEFAULT_BRIDGE))),
    reason="model9 bridge not built",
)


def test_reset_step_and_replay_through_the_native_contract() -> None:
    with Model9Bridge() as bridge:
        assert bridge.schema.core_width == len(bridge.schema.core_features)
        step = bridge.reset(17)
        villages = [v for v in step["villages"] if v["observation"] is not None]
        assert villages, "the opening boundary has resident-economy villages"
        obs = observation_batch(villages)
        assert obs["core"].shape == (len(villages), bridge.schema.core_width)
        for name, rows, width in contracts.SETS:
            assert obs[name].shape == (len(villages), rows, width)
        teacher = bridge.step({})
        labelled = [v for v in teacher["villages"] if v["teacher"] is not None]
        assert labelled and all(len(v["teacher"]["service"]) == len(contracts.SERVICES) for v in labelled)
        assert all(np.isfinite(v["welfare"]) for v in teacher["villages"])
        # The teacher's own labels as actions: the contract must accept them and keep replay exact.
        actions = {v["camp"]: {"strategic": v["teacher"]["strategic"], "service": v["teacher"]["service"],
                               "clothingRoute": v["teacher"]["clothingRoute"], "shelterRoute": v["teacher"]["shelterRoute"],
                               "movement": contracts.NATIVE_MOVEMENT, "fission": -1} for v in labelled}
        bridge.step(actions)
        assert bridge.verify_replay() == 2
