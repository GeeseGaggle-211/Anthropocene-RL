"""Native6 adapter; all prior native5 defaults remain explicit and unchanged."""

from anthropocene_rl.autonomous_v4.env import AutonomousTransport as BaseTransport
from anthropocene_rl.autonomous_v5.env import AutonomousElderEnv as BaseEnv
from anthropocene_rl.autonomous_v6.contracts import action_wire, features, observation


class AutonomousTransport(BaseTransport):
    def _request(self, payload):
        if payload["op"].startswith("autonomous_"):
            payload = dict(payload, op=payload["op"].replace("autonomous_", "autonomous6_", 1))
        return super()._request(payload)

    def step_world(self, actions):
        return self._request(
            dict(op="autonomous_step", actions={str(k): action_wire(v) for k, v in actions.items()})
        )


class AutonomousElderEnv(BaseEnv):
    schema_versions = (4, 6, 6, 6)
    feature_width = 1062
    decode_observation = staticmethod(observation)
    encode_features = staticmethod(features)

    def __init__(self, *args, **kwargs):
        # Bypass native5-only schema guard, retain the common Gym contract.
        from anthropocene_rl.autonomous_v4.env import AutonomousElderEnv as CommonEnv

        CommonEnv.__init__(self, *args, **kwargs)
        schema = self.transport.schema()
        if (
            schema["historyWidth"] != 301
            or schema["featureWidth"] != 1062
            or schema["saveEnvelope"] != "ANTHROPOCENE-13"
        ):
            raise ValueError("Native6 history contract mismatch")

    def step(self, action):
        # Common Gym conversion delegates to v4. Map only the latent controls here.
        from anthropocene_rl.autonomous_v6.policy import physical_controls

        return super().step(dict(action, controls=physical_controls(action["controls"])))
