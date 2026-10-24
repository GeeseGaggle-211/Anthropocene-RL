"""Manifest-declared labor personality distributions (docs/LABOR_PERSONALITY.md).

A labor run declares one personality *distribution* at ``labor_training
initialize`` (``--personality balanced`` or ``--personality dirichlet:ALPHA``).
It is stored in the checkpoint's labor model contract, so every later phase
(BC collection, critic pretraining, anchor/value collection, PPO rollout and
evaluation) re-creates exactly the same assignment from the checkpoint.

Assignment rules:

* ``balanced``: every village is (0.5, 0.5). No drift. The FiLM condition is
  exactly zero, so the model computes bitwise what it computed before FiLM.
* ``dirichlet:ALPHA``: each *founding lineage* of a world draws
  ``(q, p) ~ Dirichlet(ALPHA, ALPHA)`` once, from a keyed stream
  ``(key, iteration, world_seed, society, FOUNDER_TAG)``. A fission daughter
  inherits its parent's personality with a small Dirichlet drift,
  ``Dirichlet(concentration * parent)`` (concentration 200 by default), keyed
  ``(key, iteration, world_seed, daughter_society, DRIFT_TAG)``.

Because every draw is keyed rather than taken from a shared generator, the
assignment does not depend on village order, rewinds, retries or which phase
asks first. ``iteration`` is the checkpoint iteration, so successive PPO
iterations on the same world seed see fresh lineages, while the rollout and the
train-anchor of one iteration share them.

Teacher-label caveat: the heuristic BC teacher is personality-agnostic. BC rows
record the sampled personality, but their labels carry no personality signal,
so BC teaches the actor to ignore personality. Personality-conditioned labels
need search scored per personality (a later G3 item); do not fake them.

This module deliberately imports nothing from :mod:`labor_model`; it returns
plain ``(qol_years, population_years)`` float pairs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

BALANCED = "balanced"
DEFAULT_DAUGHTER_DRIFT = 200.0
DEFAULT_PAIR = (0.5, 0.5)
FOUNDER_TAG = 0x4C50  # "LP": founder draw
DRIFT_TAG = 0x4C44  # "LD": daughter drift draw
_DRIFT_FLOOR = 1e-4


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def parse_personality_distribution(spec: str) -> tuple[str, float | None]:
    """Validate ``balanced`` or ``dirichlet:ALPHA``; returns (canonical spec, alpha)."""

    if not isinstance(spec, str):
        raise ValueError("labor personality distribution must be a string")
    if spec == BALANCED:
        return BALANCED, None
    kind, sep, argument = spec.partition(":")
    if kind == "dirichlet" and sep:
        try:
            alpha = float(argument)
        except ValueError as error:
            raise ValueError(f"invalid dirichlet alpha in {spec!r}") from error
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("dirichlet alpha must be finite and positive")
        return f"dirichlet:{alpha!r}", alpha
    raise ValueError(
        f"labor personality distribution must be 'balanced' or 'dirichlet:alpha', not {spec!r}"
    )


def _pair(values: np.ndarray) -> tuple[float, float]:
    q = min(1.0, max(0.0, float(values[0])))
    return (q, 1.0 - q)


@dataclass(frozen=True)
class LaborPersonalityConditioning:
    """The declared distribution, its key, the daughter drift and the FiLM bound."""

    distribution: str = BALANCED
    key: int = 0
    daughter_drift: float = DEFAULT_DAUGHTER_DRIFT
    film_bound: float | None = None

    def __post_init__(self) -> None:
        canonical, _ = parse_personality_distribution(self.distribution)
        object.__setattr__(self, "distribution", canonical)
        object.__setattr__(self, "key", _nonnegative_int(self.key, "labor personality key"))
        drift = float(self.daughter_drift)
        if not math.isfinite(drift) or drift < 0:
            raise ValueError("labor personality daughter drift must be finite and nonnegative")
        object.__setattr__(self, "daughter_drift", drift)
        if self.film_bound is not None:
            bound = float(self.film_bound)
            if not math.isfinite(bound) or bound <= 0:
                raise ValueError("labor FiLM bound must be finite and positive")
            object.__setattr__(self, "film_bound", bound)

    @property
    def alpha(self) -> float | None:
        return parse_personality_distribution(self.distribution)[1]

    @property
    def is_default(self) -> bool:
        """Balanced with no FiLM bound: checkpoints keep the pre-FiLM contract bytes."""

        return self.distribution == BALANCED and self.film_bound is None

    def state_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution,
            "key": self.key,
            "daughterDrift": self.daughter_drift,
            "filmBound": self.film_bound,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any] | None) -> LaborPersonalityConditioning:
        if state is None:
            return cls()
        if not isinstance(state, Mapping) or set(state) != {
            "distribution",
            "key",
            "daughterDrift",
            "filmBound",
        }:
            raise ValueError("labor personality conditioning fields are malformed")
        return cls(state["distribution"], state["key"], state["daughterDrift"], state["filmBound"])

    def assignment(self, iteration: int = 0) -> LaborPersonalityAssignment:
        return LaborPersonalityAssignment(self, _nonnegative_int(iteration, "labor iteration"))


@dataclass
class WorldPersonalities:
    """The lineage table of one world under one assignment (resolves village rows)."""

    conditioning: LaborPersonalityConditioning
    iteration: int
    world_seed: int
    table: dict[int, tuple[float, float]] = field(default_factory=dict)

    def _rng(self, society: int, tag: int) -> np.random.Generator:
        return np.random.default_rng(
            [self.conditioning.key, self.iteration, self.world_seed, society, tag]
        )

    def founder(self, society: int) -> tuple[float, float]:
        alpha = self.conditioning.alpha
        if alpha is None:
            return DEFAULT_PAIR
        return _pair(self._rng(society, FOUNDER_TAG).dirichlet([alpha, alpha]))

    def daughter(self, society: int, parent: tuple[float, float]) -> tuple[float, float]:
        if self.conditioning.alpha is None:
            return DEFAULT_PAIR
        concentration = self.conditioning.daughter_drift
        if concentration <= 0:
            return parent
        weights = np.maximum(np.asarray(parent, dtype=np.float64), _DRIFT_FLOOR) * concentration
        return _pair(self._rng(society, DRIFT_TAG).dirichlet(weights))

    def resolve(self, villages: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
        """One (qol, population) pair per village row, order-independent.

        A society already in the table keeps its value. A new society whose
        parent is known (or resolved in this same batch) inherits with drift;
        any other new society is a founder.
        """

        if self.conditioning.alpha is None:  # balanced: no lineage identity is read
            return [DEFAULT_PAIR for _ in villages]
        pending: dict[int, int] = {}
        for village in villages:
            society = _nonnegative_int(int(village["society"]), "village society")
            parent = int(village.get("parentSociety", 0) or 0)
            if society not in self.table:
                pending[society] = parent
        while pending:
            progressed = False
            for society, parent in sorted(pending.items()):
                if parent != society and parent in self.table:
                    self.table[society] = self.daughter(society, self.table[parent])
                    del pending[society]
                    progressed = True
            if progressed:
                continue
            # No pending society has a known parent: roots (parent unknown and not
            # itself pending) found new lineages; a parent cycle breaks at its least id.
            roots = [s for s, p in sorted(pending.items()) if p == s or p not in pending]
            for society in roots or [min(pending)]:
                self.table[society] = self.founder(society)
                del pending[society]
        return [self.table[int(village["society"])] for village in villages]


@dataclass
class LaborPersonalityAssignment:
    """A run's conditioning bound to one checkpoint iteration; caches per-world tables."""

    conditioning: LaborPersonalityConditioning
    iteration: int = 0
    worlds: dict[int, WorldPersonalities] = field(default_factory=dict)

    def world(self, world_seed: int) -> WorldPersonalities:
        seed = _nonnegative_int(world_seed, "world seed")
        if seed not in self.worlds:
            self.worlds[seed] = WorldPersonalities(self.conditioning, self.iteration, seed)
        return self.worlds[seed]


def personality_for_world(personality: Any, world_seed: int) -> Any:
    """Collector hook: an assignment's per-world resolver, else ``personality`` unchanged."""

    if isinstance(personality, LaborPersonalityAssignment):
        return personality.world(world_seed)
    return personality
