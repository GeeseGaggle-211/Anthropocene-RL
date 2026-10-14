"""Explicit native-v5 transport and Gym adapter; v4 defaults retain their contract."""

from anthropocene_rl.autonomous_v4.env import AutonomousElderEnv as BaseEnv
from anthropocene_rl.autonomous_v4.env import AutonomousTransport as BaseTransport
from anthropocene_rl.autonomous_v5.contracts import (
    FEATURE_WIDTH,
    HISTORY_NAMES,
    features,
    observation,
)


class AutonomousTransport(BaseTransport):
    def _request(self, payload):
        if payload["op"].startswith("autonomous_"):
            payload = dict(payload, op=payload["op"].replace("autonomous_", "autonomous5_", 1))
        return super()._request(payload)


class AutonomousElderEnv(BaseEnv):
    schema_versions = (3, 5, 5, 5)
    feature_width = FEATURE_WIDTH
    decode_observation = staticmethod(observation)
    encode_features = staticmethod(features)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        schema = self.transport.schema()
        if (
            schema["featureWidth"] != FEATURE_WIDTH
            or tuple(schema["historyFeatures"]) != HISTORY_NAMES
            or schema["saveEnvelope"] != "ANTHROPOCENE-12"
        ):
            raise ValueError("Native history layout/save contract mismatch")
