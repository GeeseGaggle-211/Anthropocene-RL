"""The audit-only bridge ``inspect`` op is read-only: interleaving it with steps leaves every post-step fingerprint and
the replay unchanged (docs/MODEL9_MOVEMENT_AUDIT.md). Needs the built model9 bridge (ANTHROPOCENE_MODEL9_BRIDGE)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from anthropocene_rl.autonomous_v7.bridge import DEFAULT_BRIDGE, Model9Bridge

BRIDGE = Path(os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE", str(DEFAULT_BRIDGE)))


def _run(seed: int, turns: int, probe: bool) -> tuple[list[str], int]:
    prints = []
    with Model9Bridge() as bridge:
        step = bridge.reset(seed, movement=True)
        prints.append(step["fingerprint"])
        for _ in range(turns):
            if probe:
                # The op itself refuses to return if the fingerprint moved; a second call must agree too. (The reset
                # step's reported fingerprint predates Observe initializing Elder memory, so it is not compared.)
                response = bridge.request({"op": "inspect", "tiles": [0, 1, 2]})
                assert bridge.request({"op": "inspect"})["fingerprint"] == response["fingerprint"]
                assert response["inspect"]["camps"], "inspect returned no camps"
            step = bridge.step({})
            prints.append(step["fingerprint"])
        return prints, bridge.verify_replay()


@pytest.mark.skipif(not BRIDGE.exists(), reason="model9 bridge not built")
def test_inspect_leaves_fingerprints_unchanged() -> None:
    plain, plain_replay = _run(2000, 3, probe=False)
    probed, probed_replay = _run(2000, 3, probe=True)
    assert plain[1:] == probed[1:]
    assert plain_replay == probed_replay == 3
