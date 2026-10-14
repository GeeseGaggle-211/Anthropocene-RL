"""Frozen history layout; no obsolete supervisory action semantics are reused."""

import numpy as np

from anthropocene_rl.autonomous_v4.contracts import (
    FEATURE_WIDTH as BASE_FEATURE_WIDTH,
)
from anthropocene_rl.autonomous_v4.contracts import (
    action_wire as action_wire,
)
from anthropocene_rl.autonomous_v4.contracts import (
    features as base_features,
)
from anthropocene_rl.autonomous_v4.contracts import (
    observation as base_observation,
)

ACTION_VERSION = 3
OBSERVATION_VERSION = ENVIRONMENT_VERSION = REPLAY_VERSION = 5
GROUPS = (
    ("decision_present", 1),
    ("decision_age_years", 1),
    ("decision_elapsed_years", 1),
    ("requested_workforce", 13),
    ("requested_activities", 195),
    ("requested_controls", 3),
    ("requested_movement", 8),
    ("quote_present", 1),
    ("quoted_partner_ordinal", 8),
    ("quoted_give", 11),
    ("quoted_want", 11),
    ("quoted_quantities", 2),
    ("compiled_staffing", 13),
    ("production_output", 11),
    ("material_extraction", 3),
    ("population_start", 1),
    ("population_end", 1),
    ("qol", 1),
    ("food_met", 1),
    ("water_met", 1),
    ("moved", 1),
    ("settled_trade_count", 1),
    ("settled_give", 1),
    ("settled_want", 1),
    ("immigrated", 1),
    ("emigrated", 1),
    ("daughters", 1),
    ("opening_animals", 1),
    ("ending_animals", 1),
    ("opening_field_capital", 1),
    ("ending_field_capital", 1),
    ("institution_present", 1),
    ("institution_age_years", 1),
)
HISTORY_NAMES = tuple(
    name if size == 1 else f"{name}:{i}" for name, size in GROUPS for i in range(size)
)
HISTORY_WIDTH = len(HISTORY_NAMES)
FEATURE_WIDTH = BASE_FEATURE_WIDTH + HISTORY_WIDTH
SLICES = {}
offset = 0
for _name, _size in GROUPS:
    SLICES[_name] = slice(offset, offset + _size)
    offset += _size
RAW_GROUPS = {
    "decision_present",
    "requested_workforce",
    "requested_activities",
    "requested_controls",
    "requested_movement",
    "quote_present",
    "quoted_partner_ordinal",
    "quoted_give",
    "quoted_want",
    "qol",
    "food_met",
    "water_met",
    "moved",
    "institution_present",
}
LOG_MASK = np.array([name not in RAW_GROUPS for name, size in GROUPS for _ in range(size)])


def validate_history(data):
    h = np.asarray(data, dtype=np.float64)
    if h.shape != (HISTORY_WIDTH,) or not np.isfinite(h).all() or (h < 0).any():
        raise ValueError("Invalid autonomous history shape/amounts")
    for name in RAW_GROUPS:
        if (h[SLICES[name]] > 1).any():
            raise ValueError(f"Invalid bounded history group {name}")
    if h[0] not in (0, 1) or h[-2] != 1:
        raise ValueError("Invalid institution/decision presence")
    if not h[0]:
        if h[:-2].any():
            raise ValueError("Absent decision contains history")
        return h
    if h[1] + 1e-8 < h[2] or h[2] <= 0 or h[-1] + 1e-8 < h[1]:
        raise ValueError("Invalid history chronology")
    if not np.isclose(h[SLICES["requested_workforce"]].sum(), 1, atol=1e-8, rtol=0):
        raise ValueError("Invalid previous workforce simplex")
    rows = h[SLICES["requested_activities"]].reshape(13, 15).sum(-1)
    if not np.all(np.isclose(rows, 0, atol=1e-8, rtol=0) | np.isclose(rows, 1, atol=1e-8, rtol=0)):
        raise ValueError("Invalid previous activity simplexes")
    for name in ("requested_movement", "quoted_partner_ordinal", "quoted_give", "quoted_want"):
        v = h[SLICES[name]]
        expected = 1 if name == "requested_movement" else h[SLICES["quote_present"]][0]
        if not np.all((v == 0) | (v == 1)) or v.sum() != expected:
            raise ValueError(f"Invalid previous one-hot {name}")
    q = h[SLICES["quoted_quantities"]]
    quoted = h[SLICES["quote_present"]][0]
    if (
        quoted not in (0, 1)
        or (quoted and ((q < 1e-6) | (q > 1e6)).any())
        or (not quoted and q.any())
    ):
        raise ValueError("Invalid previous quote quantities")
    if quoted and np.argmax(h[SLICES["quoted_give"]]) == np.argmax(h[SLICES["quoted_want"]]):
        raise ValueError("Previous quote gives and wants the same good")
    staffing = h[SLICES["compiled_staffing"]]
    if (
        not np.all(staffing == np.floor(staffing))
        or staffing.sum() != h[SLICES["population_start"]][0]
    ):
        raise ValueError("Invalid previous compiled staffing")
    return h


def observation(wire):
    if wire.get("SchemaVersion") != 5:
        raise ValueError("Autonomous history observation version mismatch")
    for key, shape in {
        "WorkforceMask": (13,),
        "ActivityMask": (13, 15),
        "ControlMask": (3,),
        "MovementMask": (8,),
        "CounterpartyMask": (8,),
        "GoodMask": (11,),
    }.items():
        raw = np.asarray(wire[key], dtype=object)
        if raw.shape != shape or any(not isinstance(x, (bool, np.bool_)) for x in raw.ravel()):
            raise ValueError(f"Invalid boolean mask {key}")
    result = base_observation(dict(wire, SchemaVersion=4))
    if (result["Stocks"] < 0).any():
        raise ValueError("Invalid own stock")
    if result["Context"][40:56].any() or result["Context"][114:117].any():
        raise ValueError("Obsolete supervisory receipt/development fields in v5")
    result["History"] = validate_history(wire["History"])
    return result


def features(obs):
    history = obs["History"].copy()
    history[LOG_MASK] = np.log1p(history[LOG_MASK])
    return np.concatenate((base_features(obs), history.astype(np.float32)))
