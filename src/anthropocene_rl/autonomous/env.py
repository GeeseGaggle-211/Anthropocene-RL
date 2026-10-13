"""Native WEGO client and single-Elder Gymnasium view with scripted peers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from anthropocene_rl.autonomous.contracts import (
    ACTION_VERSION,
    ENVIRONMENT_VERSION,
    FEATURE_WIDTH,
    OBSERVATION_VERSION,
    REPLAY_VERSION,
    action_wire,
    features,
    observation,
)
from anthropocene_rl.env.transport import JsonLineProcessTransport


class AutonomousTransport(JsonLineProcessTransport):
    def schema(self) -> dict[str, Any]:
        return self._request({"op": "autonomous_schema"})

    def reset_world(self, snapshot: bytes, controlled: list[int], limit: int) -> dict[str, Any]:
        return self._request(
            {
                "op": "autonomous_reset",
                "snapshot": snapshot,
                "controlled": controlled,
                "decisionLimit": limit,
            }
        )

    def step_world(self, actions: Mapping[int, dict[str, Any]]) -> dict[str, Any]:
        return self._request(
            {
                "op": "autonomous_step",
                "actions": {str(k): action_wire(v) for k, v in actions.items()},
            }
        )

    def verify_replay(self) -> dict[str, Any]:
        return self._request({"op": "autonomous_verify_replay"})

    def capture(self) -> dict[str, Any]:
        return self._request({"op": "autonomous_capture"})


class AutonomousElderEnv(gym.Env):
    """Gym focal agent; peer callbacks see only their own committed observation.

    Quotes are accepted by step_world; this initial learned action space fixes
    quote=None. Scripted peers may quote. Learned trade heads remain gated.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        transport: AutonomousTransport,
        snapshot: bytes,
        focal: int,
        peers: Mapping[int, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
        decision_limit: int = 32,
    ):
        self.transport, self.snapshot, self.focal = transport, snapshot, focal
        self.peers = dict(peers or {})
        self.limit = decision_limit
        schema = transport.schema()
        if tuple(
            schema[k]
            for k in ("actionVersion", "observationVersion", "environmentVersion", "replayVersion")
        ) != (ACTION_VERSION, OBSERVATION_VERSION, ENVIRONMENT_VERSION, REPLAY_VERSION):
            raise ValueError("Autonomous schema mismatch")
        self.action_space = spaces.Dict(
            {
                "workforce": spaces.Box(0, 1, (13,), np.float64),
                "activities": spaces.Box(0, 1, (13, 15), np.float64),
                "controls": spaces.Box(0, 1, (3,), np.float64),
            }
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, (FEATURE_WIDTH,), np.float32)
        self.current: dict[int, dict[str, Any]] = {}
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        response = self.transport.reset_world(self.snapshot, [self.focal, *self.peers], self.limit)
        self.current = {int(k): observation(v) for k, v in response["observations"].items()}
        self.done = False
        return features(self.current[self.focal]), {
            "native": self.current[self.focal],
            "fingerprint": response["fingerprint"],
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset required")
        actions = {self.focal: action}
        for camp, policy in self.peers.items():
            if camp in self.current:
                actions[camp] = policy({k: v.copy() for k, v in self.current[camp].items()})
        response = self.transport.step_world(actions)
        result = response["result"]
        self.current = {int(k): observation(v) for k, v in result["Observations"].items()}
        terminated = self.focal not in self.current
        truncated = result["Truncated"] and not terminated
        self.done = terminated or truncated
        native = self.current.get(self.focal)
        obs = np.zeros(FEATURE_WIDTH, np.float32) if native is None else features(native)
        return (
            obs,
            result["Rewards"][str(self.focal)],
            terminated,
            truncated,
            {
                "native": native,
                "elapsed_years": result["ElapsedYears"],
                "staffing": result["Staffing"][str(self.focal)],
                "fingerprint": result["Fingerprint"],
                "native_step_seconds": response["nativeStepSeconds"],
            },
        )

    def close(self):
        self.transport.close()
