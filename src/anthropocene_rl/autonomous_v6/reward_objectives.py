"""Learner-only objective v1; native focal welfare and physics stay authoritative."""

import math
from dataclasses import asdict, dataclass

from anthropocene_rl.training.returns import discounted_interval_reward

OBJECTIVES = {"original", "midpoint", "superlinear", "potential"}
OBJECTIVE_VERSION = "focal-training-objective-v1"


@dataclass(frozen=True)
class Objective:
    kind: str
    horizon: float = 400.0
    reward_scale: float = 120.0
    alpha: float = 1.5
    reference_population: float = 100.0
    potential_coefficient: float = 1.0
    potential_population_cap: float = 100.0

    def __post_init__(self):
        if self.kind not in OBJECTIVES:
            raise ValueError("Unknown learner objective")
        positive = [
            self.horizon,
            self.reward_scale,
            self.reference_population,
            self.potential_population_cap,
        ]
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Invalid objective units or bounds")
        if not 1.5 <= self.alpha <= 2 or not math.isfinite(self.potential_coefficient):
            raise ValueError("Invalid population exponent/potential")
        if self.potential_coefficient < 0:
            raise ValueError("Negative demographic potential")

    def wire(self):
        return dict(version=OBJECTIVE_VERSION, **asdict(self))

    @classmethod
    def from_wire(cls, wire):
        if wire.get("version") != OBJECTIVE_VERSION:
            raise ValueError("Unversioned learner objective")
        return cls(**{k: v for k, v in wire.items() if k != "version"})

    def value_factor(self, initial_population):
        if not math.isfinite(initial_population) or initial_population <= 0:
            raise ValueError("Positive immutable reset population required")
        if self.kind == "superlinear":
            return (self.reference_population / initial_population) ** (self.alpha - 1)
        return 1.0

    def potential(self, population, terminal=False):
        if not math.isfinite(population) or population < 0:
            raise ValueError("Invalid own population")
        if self.kind != "potential" or terminal:
            return 0.0
        return self.potential_coefficient * min(population, self.potential_population_cap)

    def interval(self, receipt, initial_population):
        """Already discounted learner units; never replace the native reward field."""
        factor = self.value_factor(initial_population)
        start, end, qol, elapsed = [
            receipt[k] for k in ["population_start", "population_end", "qol", "elapsed"]
        ]
        if (
            any(not math.isfinite(x) for x in [start, end, qol, elapsed])
            or min(start, end) < 0
            or not 0 <= qol <= 1
            or elapsed <= 0
        ):
            raise ValueError("Invalid closed owner interval")
        true_reward = start * qol * elapsed
        if not math.isclose(receipt["reward"], true_reward, rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError("Native focal welfare identity differs")
        population = (start + end) / 2 if self.kind == "midpoint" else start
        if self.kind == "superlinear":
            population = initial_population * (start / initial_population) ** self.alpha
        raw = population * qol * elapsed
        reward = discounted_interval_reward(
            raw / elapsed / self.reward_scale, elapsed, self.horizon
        )
        shaping = math.exp(-elapsed / self.horizon) * self.potential(
            end, receipt["terminal"]
        ) - self.potential(start)
        return dict(
            reward=reward + shaping,
            raw_training_welfare=raw,
            shaping=shaping,
            value_factor=factor,
            original_welfare=true_reward,
        )


@dataclass(frozen=True)
class EpisodeObjectiveState:
    """Own reset metadata survives fission, migration and mid-episode restoration."""

    initial_population: float
    objective: Objective

    def __post_init__(self):
        self.objective.value_factor(self.initial_population)

    def wire(self):
        return dict(
            version="focal-objective-episode-v1",
            initial_population=self.initial_population,
            objective=self.objective.wire(),
        )

    @classmethod
    def restore(cls, wire):
        if wire.get("version") != "focal-objective-episode-v1":
            raise ValueError("Objective episode context missing")
        return cls(wire["initial_population"], Objective.from_wire(wire["objective"]))


def crowding_transferred_headcount(receipt):
    """Optional credit basis only: no daughter future welfare or other outflow credit."""
    d = receipt["demographics"]
    if d["SchemaVersion"] != 1 or sum(d["FissionByTrigger"].values()) != d["FissionHeadcount"]:
        raise ValueError("Unreconciled fission diagnostic")
    return d["FissionByTrigger"].get("Crowding", 0)
