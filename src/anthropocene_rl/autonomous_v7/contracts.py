"""Native Elder contract 9 (model9): the learner-side registry.

The native side is authoritative (``ElderModel9Contract`` in the simulator). ``verify_schema`` checks a bridge's
schema response against this registry so a learner never trains against a mismatched layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

NATIVE_CONTRACT = 9
ACTION_SCHEMA = 9
OBSERVATION_SCHEMA = 9
ENVIRONMENT_SCHEMA = 9
# Contract 9 capital commitments: the span head's categories (0 holds the active commitment, 1 decides this turn
# only, 2-6 commit the capital controls for that many turns) and the commitment block appended to the core.
SPANS: tuple[int, ...] = (0, 1, 2, 3, 4, 6)
SPAN_HOLD, SPAN_SINGLE = 0, 1
COMMITMENT_FEATURES = 11

SETS: tuple[tuple[str, int, int], ...] = (
    ("sites", 7, 14),
    ("fishing", 3, 11),
    ("herds", 8, 8),
    ("commodities", 11, 8),
    ("contacts", 8, 9),
)
SERVICES: tuple[str, ...] = ("food", "clothing", "shelter", "tools", "storage", "materials")
STRATEGIC: tuple[str, ...] = (
    "retained-field-operation",
    "retained-herd-fraction",
    "development-budget",
    "field-development-share",
    "fishing-development-share",
)
# The first three strategic controls are unit-interval levels; the last two are the economy mix, which the actor
# emits as one masked softmax over (cultivation, herding, fishing) = ((1 - f) s, (1 - f)(1 - s), f).
UNIT_STRATEGIC = 3
ECONOMY_MIX: tuple[str, ...] = ("cultivation", "herding", "fishing")
MOVEMENT_SLOTS = 8  # stay + seven site rows
# Technology bits (Anthropocene.Simulation.Technology) for SimulationConfig.TrainingWithheldTechnologies.
HUSBANDRY, CULTIVATION, WATERCRAFT = 1 << 1, 1 << 2, 1 << 3
# Capability fixtures: the mask withholds every technology but the named one.
CAPABILITY_MASKS = {"herding": CULTIVATION | WATERCRAFT, "farming": HUSBANDRY | WATERCRAFT, "fishing": HUSBANDRY | CULTIVATION}
NATIVE_MOVEMENT = -2
FISSION_POLICIES = 3  # Neutral, Discourage, Encourage

# Physical discounting: one decision per 20-year turn, value horizon of 400 years.
TURN_YEARS = 20.0
HORIZON_YEARS = 400.0
# Lineage sharing window (decision D4): daughters founded within this many turns share the parent's QoL.
LINEAGE_TURNS = 3


@dataclass(frozen=True)
class Schema:
    core_width: int
    core_features: tuple[str, ...]
    build_hash: str


def verify_schema(response: Mapping[str, Any]) -> Schema:
    expected = {
        "nativeContract": NATIVE_CONTRACT,
        "actionSchema": ACTION_SCHEMA,
        "observationSchema": OBSERVATION_SCHEMA,
        "environmentSchema": ENVIRONMENT_SCHEMA,
        "movementSlots": MOVEMENT_SLOTS,
        "nativeMovement": NATIVE_MOVEMENT,
    }
    for key, value in expected.items():
        if response.get(key) != value:
            raise ValueError(f"model9 schema mismatch for {key}: {response.get(key)!r} != {value!r}")
    sets = tuple((s["name"], s["rows"], s["width"]) for s in response["sets"])
    if sets != SETS:
        raise ValueError(f"model9 entity-set layout mismatch: {sets} != {SETS}")
    if tuple(response["services"]) != SERVICES or tuple(response["strategic"]) != STRATEGIC:
        raise ValueError("model9 action registry mismatch")
    features = tuple(response["coreFeatures"])
    if len(features) != response["coreWidth"]:
        raise ValueError("model9 core feature registry is inconsistent")
    return Schema(core_width=int(response["coreWidth"]), core_features=features, build_hash=str(response["buildHash"]))
