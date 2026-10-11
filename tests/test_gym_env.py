from typing import Any

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from anthropocene_rl.env.gym_env import AnthropoceneElderEnv


def _tensor() -> dict[str, Any]:
    values = [0.0] * 512
    values[80] = 1.0
    values[82] = 1.0
    values[482] = 1.0
    return {
        "SchemaVersion": 2,
        "Values": values,
        "Present": [True] * 512,
    }


class _TerminalTransport:
    closed = False

    def schema(self) -> dict[str, Any]:
        return {
            "ok": True,
            "observationSchemaVersion": 2,
            "actionSchemaVersion": 1,
            "environmentSchemaVersion": 2,
            "criticSidecarSchemaVersion": 1,
            "tensorFeatureCount": 512,
            "simulatorVersion": "test-native-v2",
        }

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "result": {
                "ObservationSchemaVersion": 2,
                "ActionSchemaVersion": 1,
                "EnvironmentSchemaVersion": 2,
                "SimulatorVersion": "test-native-v2",
            },
            "tensor": _tensor(),
        }

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "result": {
                "ScalarReward": 12.5,
                "Terminated": True,
                "Truncated": False,
            },
            "tensor": None,
        }

    def verify_replay(self) -> dict[str, Any]:
        return {"ok": True, "steps": 1, "finalStateFingerprint": "abc"}

    def close(self) -> None:
        self.closed = True


def test_terminal_step_preserves_reward_when_native_has_no_observation() -> None:
    env = AnthropoceneElderEnv(_TerminalTransport(), {})
    first, _ = env.reset()
    final, reward, terminated, truncated, info = env.step(
        {"continuous": np.full(4, 0.5), "movement_index": 0}
    )
    assert np.array_equal(final, first)
    assert reward == 12.5
    assert terminated
    assert not truncated
    assert info["terminal_observation_missing"] is True
    assert env.verify_replay()["steps"] == 1
    with pytest.raises(RuntimeError, match="reset"):
        env.step({"continuous": np.full(4, 0.5), "movement_index": 0})
