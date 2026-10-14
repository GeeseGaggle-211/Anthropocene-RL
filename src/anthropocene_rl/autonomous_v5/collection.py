"""Same bounded canonical collector; explicit native-v5 decoder/transport injection."""

from anthropocene_rl.autonomous_v4.collection import Collector as BaseCollector
from anthropocene_rl.autonomous_v4.collection import DeadlineTransport
from anthropocene_rl.autonomous_v4.training import episode as base_episode
from anthropocene_rl.autonomous_v5.contracts import observation
from anthropocene_rl.autonomous_v5.env import AutonomousTransport
from anthropocene_rl.autonomous_v5.policy import inputs


def episode(*args, **kwargs):
    config = kwargs.get("config", args[5] if len(args) > 5 else {})
    # Native capability acquisition is preserved. A core-only experiment must
    # stop if progression reveals an unvalidated face, before evaluating the MLP.
    require_core = config.get("require_core_action_faces", True)
    if not isinstance(require_core, bool):
        raise ValueError("require_core_action_faces must be an explicit boolean")
    encoder = core_inputs if require_core else inputs
    return base_episode(*args, **kwargs, decoder=observation, tensor_inputs=encoder)


def core_inputs(observations):
    for obs in observations:
        if obs["WorkforceMask"][[2, 3]].any() or obs["ControlMask"][[1, 2]].any():
            raise ValueError("Core curriculum reached an unvalidated husbandry/fishing action face")
    return inputs(observations)


class DeadlineHistoryTransport(AutonomousTransport, DeadlineTransport):
    """Prefix conversion then inherited IPC timer; no bypass of bounded close."""


class Collector(BaseCollector):
    transport_class = DeadlineHistoryTransport
    episode_runner = staticmethod(episode)
