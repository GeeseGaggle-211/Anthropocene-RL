"""Native Elder contract 10 (model9): the learner-side registry.

The native side is authoritative (``ElderModel9Contract`` in the simulator). ``verify_schema`` checks a bridge's
schema response against this registry so a learner never trains against a mismatched layout.
"""  # noqa: E501 - docstring/prose line kept verbatim

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

NATIVE_CONTRACT = 12
ACTION_SCHEMA = 11
OBSERVATION_SCHEMA = 12
ENVIRONMENT_SCHEMA = 12
# Contract 9 capital commitments: the span head's categories (0 holds the active commitment, 1
# decides this turn
# only, 2-6 commit the capital controls for that many turns) and the commitment block appended to
# the core.
SPANS: tuple[int, ...] = (0, 1, 2, 3, 4, 6)
SPAN_HOLD, SPAN_SINGLE = 0, 1
COMMITMENT_FEATURES = 11
# Objective components (Docs/ElderCapitalCommitmentsDesign.md, Part 4). Lineage welfare is the
# interim reward; each
# component gets its own value head and the advantage is a weighted sum. The default weights
# reproduce welfare.
OBJECTIVES: tuple[str, ...] = ("welfare", "population_years", "trade_utility", "qol_years")
OBJECTIVE_WEIGHTS: dict[str, float] = {
    "welfare": 1.0,
    "population_years": 0.0,
    "trade_utility": 0.0,
    "qol_years": 0.0,
}
# Elder personality (docs/MODEL9_PERSONALITY.md): a weighting over OBJECTIVES, an input to the token
# model's village
# token and critics. The default personality is OBJECTIVE_WEIGHTS (welfare only), which reproduces
# non-personality runs.
PERSONALITY_WIDTH = len(OBJECTIVES)
# Round 3 stable personality scales (ppo.combine_personal, --personality-scales reference): each
# objective's reference
# spread s_k_ref, the standard deviation of its discounted (HORIZON_YEARS) return per decision under
# the native
# heuristic. Measured by scale_reference.py: heuristic (no actions), TRAIN seeds 1000-1059, 45-turn
# rollouts, keeping
# only rows with at least 20 later turns (the first 25 turns, 37,085 decision rows), so every return
# covers a full
# 400-year-equivalent window; Small world, model9 bridge on native contract 12 (simulator f06e3be,
# 21 October 2026).
# The first build's values (25-turn rollouts, all rows, late returns truncated) were welfare 37864,
# population_years
# 50391, trade_utility 17.55, qol_years 64.0. Re-measure when the reward components or the
# simulator's economy change.
OBJECTIVE_REFERENCE_SCALES: dict[str, float] = {
    "welfare": 59318.99,  # person-QoL-years (mean return 136573)
    "population_years": 85133.98,  # person-years (mean 172285)
    # person-QoL-years of realised trade gain (mean 6.35; the heuristic trades little)
    "trade_utility": 26.684,
    "qol_years": 43.152,  # QoL-years (mean 259.0)
}
# Contract 11: the trade_utility weight a --learned-trade run uses unless --objective-weights sets
# it. Trade gains already
# reach welfare through the goods; this small extra credit speeds learning for economies whose
# output they cannot use.
TRADE_UTILITY_WEIGHT = 0.1

SETS: tuple[tuple[str, int, int], ...] = (
    ("sites", 7, 14),
    ("fishing", 3, 11),
    ("herds", 8, 8),
    ("commodities", 11, 8),
    ("contacts", 8, 9),
    # Contract 10: regional movement candidates (simulator ElderMovement); row 0 is the current
    # tile.
    ("regions", 8, 24),
    # Contract 11: one row per trade group (simulator ElderTrade).
    ("trade", 7, 10),
    # Contract 12: activity tokens in ECONOMY_MIX order (cultivation, herding, fishing); later kinds
    # append rows.
    ("activities", 3, 12),
)
# Contract 11 trade posture per group: relative value (log-scale Gaussian, clamped to
# TRADE_VALUE_RANGE), offer gate
# (0 withhold, 1 offer stock, 2 offer and make to order), and one export labour share in [0, 1].
TRADE_GROUPS: tuple[str, ...] = (
    "clothing",
    "housing",
    "tents",
    "pots",
    "tools",
    "materials",
    "food",
)
TRADE_VALUE_RANGE = (0.25, 4.0)
TRADE_GATES = 3
TRADE_MAKE = 2
# Contract 10 movement head: native (the heuristic's orders), keep the destination intent, then
# region rows 0-7.
# Bridge codes: native -2, keep -3, region r is REGION_BASE + r (row 0 is stay).
REGION_ROWS = 8
REGION_MOVEMENT_SLOTS = 2 + REGION_ROWS
REGION_NATIVE, REGION_KEEP = 0, 1
KEEP_INTENT, REGION_BASE = -3, 100
SERVICES: tuple[str, ...] = ("food", "clothing", "shelter", "tools", "storage", "materials")
STRATEGIC: tuple[str, ...] = (
    "retained-field-operation",
    "retained-herd-fraction",
    "development-budget",
    "field-development-share",
    "fishing-development-share",
)
# The first three strategic controls are unit-interval levels; the last two are the economy mix,
# which the actor
# emits as one masked softmax over (cultivation, herding, fishing) = ((1 - f) s, (1 - f)(1 - s), f).
UNIT_STRATEGIC = 3
ECONOMY_MIX: tuple[str, ...] = ("cultivation", "herding", "fishing")
MOVEMENT_SLOTS = 8  # stay + seven site rows
# Technology bits (Anthropocene.Simulation.Technology) for
# SimulationConfig.TrainingWithheldTechnologies.
HUSBANDRY, CULTIVATION, WATERCRAFT = 1 << 1, 1 << 2, 1 << 3
# Capability fixtures: the mask withholds every technology but the named one.
CAPABILITY_MASKS = {
    "herding": CULTIVATION | WATERCRAFT,
    "farming": HUSBANDRY | WATERCRAFT,
    "fishing": HUSBANDRY | CULTIVATION,
}
NATIVE_MOVEMENT = -2
FISSION_POLICIES = 3  # Neutral, Discourage, Encourage

# Physical discounting: one decision per 20-year turn, value horizon of 400 years.
TURN_YEARS = 20.0
HORIZON_YEARS = 400.0
# Lineage sharing window (decision D4): daughters founded within this many turns share the parent's
# QoL.
LINEAGE_TURNS = 3


@dataclass(frozen=True)
class Schema:
    core_width: int
    core_features: tuple[str, ...]
    build_hash: str
    decision_diagnostics_schema: int = 0
    simulator_commit: str = "unknown"
    simulator_source_clean: bool = False
    raw_material_diagnostics_schema: int = 0
    diagnostic_envelope_schema: int = 0
    economic_diagnostics_schema: int = 0
    bridge_build_hash: str = "unknown"
    bridge_sha256: str = "unknown"
    feature_flags: dict | None = None
    producer_flags: dict | None = None
    configuration_hash: str = "unknown"
    players: int = 0


def verify_schema(response: Mapping[str, Any]) -> Schema:
    expected = {
        "nativeContract": NATIVE_CONTRACT,
        "actionSchema": ACTION_SCHEMA,
        "observationSchema": OBSERVATION_SCHEMA,
        "environmentSchema": ENVIRONMENT_SCHEMA,
        "movementSlots": MOVEMENT_SLOTS,
        "nativeMovement": NATIVE_MOVEMENT,
        "regionMovementSlots": REGION_MOVEMENT_SLOTS,
        "keepIntent": KEEP_INTENT,
        "regionBase": REGION_BASE,
    }
    for key, value in expected.items():
        if response.get(key) != value:
            raise ValueError(
                f"model9 schema mismatch for {key}: {response.get(key)!r} != {value!r}"
            )
    sets = tuple((s["name"], s["rows"], s["width"]) for s in response["sets"])
    if sets != SETS:
        raise ValueError(f"model9 entity-set layout mismatch: {sets} != {SETS}")
    if tuple(response["services"]) != SERVICES or tuple(response["strategic"]) != STRATEGIC:
        raise ValueError("model9 action registry mismatch")
    features = tuple(response["coreFeatures"])
    if len(features) != response["coreWidth"]:
        raise ValueError("model9 core feature registry is inconsistent")
    return Schema(
        core_width=int(response["coreWidth"]),
        core_features=features,
        build_hash=str(response["buildHash"]),
        decision_diagnostics_schema=int(response.get("decisionDiagnosticsSchema", 0)),
        simulator_commit=str(response.get("simulatorCommit", "unknown")),
        simulator_source_clean=bool(response.get("simulatorSourceClean", False)),
        raw_material_diagnostics_schema=int(response.get("rawMaterialDiagnosticsSchema", 0)),
        diagnostic_envelope_schema=int(response.get("diagnosticEnvelopeSchema", 0)),
        economic_diagnostics_schema=int(response.get("economicDiagnosticsSchema", 0)),
        bridge_build_hash=str(response.get("bridgeBuildHash", "unknown")),
        bridge_sha256=str(response.get("bridgeSha256", "unknown")),
        feature_flags=response.get("featureFlags"),
        producer_flags=response.get("producerFlags"),
        configuration_hash=str(response.get("configurationHash", "unknown")),
        players=int(response.get("players", 0)),
    )
