"""Gymnasium facade over the pinned native C# Elder environment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from anthropocene_rl.contracts import (
    ACTION_SCHEMA_VERSION,
    CRITIC_SIDECAR_SCHEMA_VERSION,
    ENVIRONMENT_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    TENSOR_FEATURE_COUNT,
)
from anthropocene_rl.env.transport import ElderTransport
from anthropocene_rl.observations.normalization import SemanticNormalizerV2
from anthropocene_rl.observations.schema import (
    ElderDirective,
    EncodedObservation,
    StructuralActionMask,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency boundary
    gym = None
    spaces = None


_BaseEnv = gym.Env if gym is not None else object


class AnthropoceneElderEnv(_BaseEnv):
    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        transport: ElderTransport,
        reset_request: Mapping[str, Any],
        *,
        normalizer: SemanticNormalizerV2 | None = None,
    ) -> None:
        if gym is None or spaces is None:
            raise ImportError("Install anthropocene-rl[adapter] to use Gymnasium")
        super().__init__()
        self.transport = transport
        self.reset_request = dict(reset_request)
        self.normalizer = normalizer
        try:
            self._bridge_schema = transport.schema()
            _validate_bridge_schema(self._bridge_schema)
        except Exception:
            transport.close()
            raise
        self.observation_space = spaces.Box(
            low=-1.0 if normalizer is not None else -np.inf,
            high=1.0 if normalizer is not None else np.inf,
            shape=(512,),
            dtype=np.float32,
        )
        self.action_space = spaces.Dict(
            {
                "continuous": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
                "movement_index": spaces.Discrete(8),
            }
        )
        self._raw: EncodedObservation | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        payload = dict(self.reset_request)
        if seed is not None:
            payload["Seed"] = int(seed)
        if options:
            payload.update(options)
        response = self.transport.reset(payload)
        result = _mapping(response, "result", "Result")
        _validate_reset_contract(result, self._bridge_schema)
        raw = _encoded(response)
        self._raw = raw
        return self._observation(raw), _info(response, raw)

    def step(
        self,
        action: Mapping[str, Any],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._raw is None:
            raise RuntimeError("reset() must be called before step()")
        continuous = np.asarray(action["continuous"], dtype=np.float64)
        if continuous.shape != (4,) or np.any(~np.isfinite(continuous)):
            raise ValueError("continuous action must be four finite values")
        if np.any((continuous < 0.0) | (continuous > 1.0)):
            raise ValueError("continuous action must already be bounded to [0,1]")

        mask = StructuralActionMask.from_observation(self._raw)
        bounded = mask.canonicalize_continuous(continuous)
        movement_index = int(action.get("movement_index", 0))
        if movement_index < 0 or movement_index >= len(mask.movement_candidates):
            raise ValueError("movement_index is outside the native categorical layout")
        if not mask.movement_candidates[movement_index]:
            raise ValueError("movement_index is masked illegal")
        directive = ElderDirective(
            retained_field_operation=bounded[0],
            retained_herd_fraction=bounded[1],
            development_budget=bounded[2],
            field_development_share=bounded[3],
            movement_candidate=movement_index - 1,
        )
        previous_raw = self._raw
        response = self.transport.step(directive.to_native())
        result = _mapping(response, "result", "Result")
        reward = float(_value(result, "ScalarReward", "scalarReward"))
        terminated = bool(_value(result, "Terminated", "terminated"))
        truncated = bool(_value(result, "Truncated", "truncated"))
        tensor = response.get("tensor", response.get("Tensor"))
        terminal_observation_missing = tensor is None
        if terminal_observation_missing:
            if not terminated:
                raise ValueError("Native bridge omitted a non-terminal observation")
            raw = previous_raw
        else:
            if not isinstance(tensor, Mapping):
                raise TypeError("Expected native tensor mapping")
            raw = EncodedObservation.from_native(tensor)

        info = _info(response, raw)
        if terminal_observation_missing:
            info["terminal_observation_missing"] = True
        self._raw = None if terminated or truncated else raw
        return self._observation(raw), reward, terminated, truncated, info

    def close(self) -> None:
        self.transport.close()

    def verify_replay(self) -> dict[str, Any]:
        return self.transport.verify_replay()

    def _observation(self, raw: EncodedObservation) -> np.ndarray:
        values = raw.values if self.normalizer is None else self.normalizer.transform(raw)
        return np.asarray(values, dtype=np.float32)


def _encoded(response: Mapping[str, Any]) -> EncodedObservation:
    tensor = _mapping(response, "tensor", "Tensor")
    return EncodedObservation.from_native(tensor)


def _info(response: Mapping[str, Any], raw: EncodedObservation) -> dict[str, Any]:
    result = response.get("result", response.get("Result"))
    return {
        "raw_observation": raw,
        "action_mask": StructuralActionMask.from_observation(raw),
        "native_result": result,
    }


def _mapping(mapping: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    value = _value(mapping, *names)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping for {names!r}")
    return value


def _value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"Missing required native field; tried {names!r}")


def _validate_bridge_schema(schema: Mapping[str, Any]) -> None:
    expected = {
        "observationSchemaVersion": OBSERVATION_SCHEMA_VERSION,
        "actionSchemaVersion": ACTION_SCHEMA_VERSION,
        "environmentSchemaVersion": ENVIRONMENT_SCHEMA_VERSION,
        "criticSidecarSchemaVersion": CRITIC_SIDECAR_SCHEMA_VERSION,
        "tensorFeatureCount": TENSOR_FEATURE_COUNT,
    }
    mismatches = {
        key: (schema.get(key), value) for key, value in expected.items() if schema.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Native bridge schema mismatch: {mismatches}")
    simulator_version = schema.get("simulatorVersion")
    if not isinstance(simulator_version, str) or not simulator_version:
        raise ValueError("Native bridge omitted simulatorVersion")


def _validate_reset_contract(
    result: Mapping[str, Any],
    bridge_schema: Mapping[str, Any],
) -> None:
    expected = (
        (("ObservationSchemaVersion", "observationSchemaVersion"), OBSERVATION_SCHEMA_VERSION),
        (("ActionSchemaVersion", "actionSchemaVersion"), ACTION_SCHEMA_VERSION),
        (("EnvironmentSchemaVersion", "environmentSchemaVersion"), ENVIRONMENT_SCHEMA_VERSION),
    )
    mismatches: dict[str, tuple[Any, int]] = {}
    for names, wanted in expected:
        actual = _value(result, *names)
        if actual != wanted:
            mismatches[names[0]] = (actual, wanted)
    if mismatches:
        raise ValueError(f"Native reset schema mismatch: {mismatches}")
    simulator_version = _value(result, "SimulatorVersion", "simulatorVersion")
    if simulator_version != bridge_schema["simulatorVersion"]:
        raise ValueError("Native reset simulator version changed after handshake")
