"""Sorted owner-isolated WEGO inference with explicit objective episode sidecars."""

import hashlib
from pathlib import Path

import torch

from anthropocene_rl.autonomous_v6.contracts import observation
from anthropocene_rl.autonomous_v6.objective_collection import context
from anthropocene_rl.autonomous_v6.policy import inputs_with_mode, to_native
from anthropocene_rl.autonomous_v6.reward_objectives import EpisodeObjectiveState, Objective
from anthropocene_rl.policies.autonomous_ppo_v8 import AutonomousPPOV8
from anthropocene_rl.training.autonomous_ppo_v8 import AutonomousTrainerV8


def own_population(obs):
    population = float(obs["Context"][0])
    if not 0 < population <= 2**24 or population != int(population):
        raise ValueError("Own current population is outside exact native observation domain")
    return population


def canonical_observations(wire_observations):
    result = {}
    for camp, wire in wire_observations.items():
        if isinstance(camp, bool) or not isinstance(camp, (int, str)):
            raise ValueError("Integer village identity required")
        identity = int(camp)
        if identity < 0 or identity in result:
            raise ValueError("Duplicate or negative village identity")
        result[identity] = wire
    return result


class AutonomousElderPolicyV8:
    def __init__(self, checkpoint, *, expected_provenance, seed):
        path = Path(checkpoint)
        state = torch.load(path, map_location="cpu", weights_only=False)
        AutonomousTrainerV8.validate_checkpoint(state, state["config"], expected_provenance)
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.model = AutonomousPPOV8(
            state["width"],
            uniform_mix=state["current_uniform_mix"],
            critic_variant=state["config"]["critic_variant"],
            concentration_floor=state["concentration_floor"],
        )
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.feature_mode = state["config"].get("feature_mode", "full")
        self.objective = Objective.from_wire(state["config"]["training_objective"])
        self.generator = torch.Generator().manual_seed(seed)
        self.episode_states = {}

    def start_episode(self, wire_observations):
        """Call at an actual new episode; restoring a native save uses restore_sidecar."""
        wire_observations = canonical_observations(wire_observations)
        self.episode_states = {
            camp: EpisodeObjectiveState(own_population(observation(wire)), self.objective)
            for camp, wire in wire_observations.items()
        }

    @torch.no_grad()
    def act_many(self, wire_observations, *, deterministic=False):
        wire_observations = canonical_observations(wire_observations)
        ids = sorted(wire_observations)
        if not ids:
            return {}
        if any(camp not in self.episode_states for camp in ids):
            raise ValueError("Start an objective episode or restore its sidecar first")
        observations = [observation(wire_observations[camp]) for camp in ids]
        rows = [
            context(
                inputs_with_mode([obs], self.feature_mode),
                self.episode_states[camp],
                own_population(obs),
            )
            for camp, obs in zip(ids, observations, strict=True)
        ]
        inputs = {k: torch.cat([row[k] for row in rows]) for k in rows[0]}
        action, _, _, _ = self.model(inputs, generator=self.generator, deterministic=deterministic)
        return {
            camp: to_native(action.wire_action(i), observations[i]) for i, camp in enumerate(ids)
        }

    def act(self, wire_observation, *, deterministic=False):
        return self.act_many({0: wire_observation}, deterministic=deterministic)[0]

    def capture_sidecar(self):
        return dict(
            version="elder-model8-inference-sidecar-v1",
            checkpoint_sha256=self.checkpoint_sha256,
            objective=self.objective.wire(),
            feature_mode=self.feature_mode,
            episode_states={str(camp): s.wire() for camp, s in self.episode_states.items()},
            sampler=self.generator.get_state().clone(),
        )

    def restore_sidecar(self, sidecar):
        if (
            sidecar.get("version") != "elder-model8-inference-sidecar-v1"
            or sidecar.get("checkpoint_sha256") != self.checkpoint_sha256
            or sidecar.get("objective") != self.objective.wire()
            or sidecar.get("feature_mode") != self.feature_mode
        ):
            raise ValueError("Inference sidecar model/objective/provenance differs")
        states = {
            camp: EpisodeObjectiveState.restore(s)
            for camp, s in canonical_observations(sidecar["episode_states"]).items()
        }
        if any(s.objective != self.objective for s in states.values()):
            raise ValueError("Village objective differs from checkpoint")
        self.generator.set_state(sidecar["sampler"])
        self.episode_states = states
