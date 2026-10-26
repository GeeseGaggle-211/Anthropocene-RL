"""Curated Neolithic scenario bank for the labor actor and critics (r5 curriculum extension).

Random fitting worlds under-sample the states where the action space grows: the
first decisions after a Farmer, Herder or Fisher head unmasks, under-housed or
under-clothed camps, fresh fission daughters, trade-eligible camps and very
small or very large camps. This module fills those gaps; it is **coverage-driven
gap filling, not a parallel bank** (manifest section 11):

1. ``coverage`` audits existing ``labor_teacher_data`` banks (the 50-decision
   fitting and calibration banks) with the same detection rules the harvest
   uses, and reports rows, roots and worlds per family against the support
   minima (default: >= 32 fitting roots across >= 4 worlds and >= 16
   calibration roots across >= 4 worlds per family), plus each food head's
   first active use.
2. ``collect`` harvests roots only for families below quota in that audit
   (``--coverage``): it drives each world under the converted teacher (H12)
   exactly as ``labor_teacher_data`` does, tags every camp at every decision
   boundary, keeps a keyed bottom-k quota of root candidates per family and
   world (rare families are kept whole, common ones capped) and holds a native
   fork of every boundary that still owns a selected root. States already
   recorded by an audited bank (``--exclude-dataset``: same world, reset options,
   turn and camp) are never roots, so a bank with the same reset options only
   contributes roots past its window. From each kept root the teacher is
   recorded for ``--scenario-decisions`` decisions (default 25) for the root
   camp and closed by the preregistered randomized tail.

Families (``FAMILIES``; a row may carry several, every tag is recorded):

* ``farming-unlocked`` / ``herding-unlocked`` / ``fishing-unlocked``: the first
  ``head_window`` (2) decisions at which the Farmer / Herder / Fisher specialty
  is legal (``labor_specialty_mask``), counted from an unmask observed inside the
  trajectory (masked at the camp's previous decision) or, for a camp present at
  the episode's first decision, from that decision (the head unmasked during the
  autonomous opening, before any recorded decision). A fission daughter born with
  the head legal opens no window (it is the ``fission-daughter`` family).
* ``mixed``: the first ``head_window`` decisions at which two or more of those
  three food heads are legal (same unmask rule on the count).
* ``small-camp`` / ``large-camp``: residents (privileged economy ``residents``,
  else the catalog population) below 100 / above 600.
* ``under-housed`` / ``under-clothed``: privileged economy ``housing-per-capita``
  / ``cloth-per-capita`` (the simulator's ``HousingPerCapita`` /
  ``ClothPerCapita`` coverage) below 0.3.
* ``fission-daughter``: a camp absent at the episode's first decision, in its
  first ``daughter_window`` (2) decisions (founded at most one decision ago).
* ``trade-eligible``: privileged economy ``trade-partners`` >= 1 (the remembered
  partners ``Trade.Plan`` would pair this camp with this turn).

Every family requires the Neolithic era (privileged unlocks ``era-neolithic``)
unless ``--any-era`` (diagnostic only). A **root** is a run entry: the decision
at which a camp enters a family (a maximal run of consecutive tagged decisions
of one camp counts once).

Scenario rows use the teacher shard format (privileged arrays, typed ``tail_*``
fields) plus ``row_source`` 4, source kind :data:`SOURCE_KIND`,
``row_scenario_family`` (bitmask of every family of the root),
``row_scenario_tags`` (the row's own families), ``row_root_turn``,
``row_start`` (root decision index), ``row_start_index`` (root index),
``row_branch_step`` (scenario step) and a per-root ``row_trajectory``
(:data:`TRAJECTORY_BASE` + root index). Shard metas list their roots; the
dataset-level ``roots.json`` makes them available to ``labor_search --source
scenario --scenario-roots``.

Partitions: ``--partition fitting`` (90100-90153, 90160-90165; training),
``calibration`` (90180-90199; the calibration scenario set, refused by every
training loader, read by ``critic-calibration`` and ``labor_r5_acceptance
--scenario-calibration-data`` for per-family calibration) and ``held-out-test``
(the reserved 90154-90159, only with ``--held-out-test``; for search roots and
per-family A ranking). Development 2000-2059 and sealed 3000-3059 are refused.

Example::

    python -m anthropocene_rl.autonomous_v7.labor_scenarios coverage \\
        --dataset fit50 --dataset cal50 --out coverage.json
    python -m anthropocene_rl.autonomous_v7.labor_scenarios collect \\
        --normalizer-checkpoint norm.pt --partition fitting --worlds 90100-90153,90160-90165 \\
        --coverage coverage.json --exclude-dataset fit50 --decisions 100 \\
        --players 16 --width 128 --height 64 --separation 9 --jobs 30 --out scen-fit
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_KIND = "curated_neolithic_scenario"
ROW_SOURCE = 4
CONTRACT = "model9-labor-curated-neolithic-scenarios-v1"
COVERAGE_SCHEMA = "labor-scenario-coverage-v1"
ROOTS_SCHEMA = "labor-scenario-roots-v1"
ROOTS_FILE = "roots.json"
# Scenario trajectories are their own (world, camp) trajectories; the offset keeps
# their ids disjoint from learner-state continuation ids in aggregated data.
TRAJECTORY_BASE = 1_000_000
PRIORITY_TAG = 0x5343454E  # "SCEN": root priority stream

HEAD_FAMILIES = (
    ("farming-unlocked", "Farmer"),
    ("herding-unlocked", "Herder"),
    ("fishing-unlocked", "Fisher"),
)
# First active use: the first decision at which the camp's valid, accepted teacher
# label staffs the head (target_specialty > 0), and the next decision.
FIRST_USE_FAMILIES = (
    ("first-use-farming", "Farmer"),
    ("first-use-herding", "Herder"),
    ("first-use-fishing", "Fisher"),
)
# Bit order is append-only: stored row_scenario_* masks keep their meaning.
FAMILIES = (
    "farming-unlocked",
    "herding-unlocked",
    "fishing-unlocked",
    "mixed",
    "small-camp",
    "large-camp",
    "under-housed",
    "under-clothed",
    "fission-daughter",
    "trade-eligible",
    "first-use-farming",
    "first-use-herding",
    "first-use-fishing",
    "food-trade",
)
FAMILY_BITS = {name: 1 << index for index, name in enumerate(FAMILIES)}
LABEL_FAMILIES = tuple(name for name, _ in FIRST_USE_FAMILIES)
HYSTERESIS_FAMILIES = ("under-housed", "under-clothed")
# Food-trade guard (user condition, 26 Oct): villages fed substantially by trade imports.
# Never up-weighted; reported per family so up-weighting food production cannot silently
# penalise them.
NEVER_UPWEIGHTED = ("food-trade",)

FITTING_WORLDS = tuple(range(90100, 90154)) + tuple(range(90160, 90166))
CALIBRATION_WORLDS = tuple(range(90180, 90200))
PARTITIONS = ("fitting", "calibration", "held-out-test")
# Support minima (manifest section 11): (roots, worlds) per family.
DEFAULT_MINIMA = {"fitting": (32, 4), "calibration": (16, 4), "held-out-test": (16, 4)}


def family_mask(names: Iterable[str]) -> int:
    mask = 0
    for name in names:
        if name not in FAMILY_BITS:
            raise ValueError(f"unknown scenario family {name!r}")
        mask |= FAMILY_BITS[name]
    return mask


def family_names(mask: int) -> list[str]:
    return [name for name in FAMILIES if int(mask) & FAMILY_BITS[name]]


def world_partition(world: int) -> str:
    """Partition of a world seed under the r5 freeze (manifest section 10)."""
    from .labor_bc import DEVELOPMENT_WORLDS, SEALED_WORLDS
    from .labor_worlds import RESERVED_HELDOUT_WORLDS

    world = int(world)
    if world in FITTING_WORLDS:
        return "fitting"
    if world in CALIBRATION_WORLDS:
        return "calibration"
    if world in RESERVED_HELDOUT_WORLDS:
        return "held-out-test"
    if world in DEVELOPMENT_WORLDS:
        return "development"
    if world in SEALED_WORLDS:
        return "sealed"
    return "other"


def validate_scenario_worlds(
    worlds: Sequence[int], partition: str, *, held_out_test: bool = False
) -> list[int]:
    """Worlds of exactly one partition; reserved worlds only with ``held_out_test``."""
    from .labor_bc import validate_fitting_worlds

    if partition not in PARTITIONS:
        raise ValueError(f"scenario partition must be one of {PARTITIONS}")
    if (partition == "held-out-test") != bool(held_out_test):
        raise ValueError(
            "the reserved worlds 90154-90159 are harvested only with --partition held-out-test "
            "and --held-out-test together"
        )
    context = f"labor scenario {partition} worlds"
    worlds = validate_fitting_worlds(worlds, context, held_out_test=held_out_test)
    outside = [w for w in worlds if world_partition(w) != partition]
    if outside:
        raise ValueError(f"{context}: worlds {outside[:5]} are not {partition} worlds")
    return worlds


# ---------------------------------------------------------------------------
# Detection rules and per-camp state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionRules:
    head_window: int = 2
    daughter_window: int = 2
    small_below: float = 100.0
    large_above: float = 600.0
    housing_below: float = 0.3
    clothing_below: float = 0.3
    trade_partners_min: float = 1.0
    require_neolithic: bool = True
    first_use_window: int = 2
    # Hysteresis (under-housed / under-clothed): enter below *_below, exit above
    # *_exit; a re-entry opens a new root only after rearm_decisions decisions above exit.
    housing_exit: float = 0.4
    clothing_exit: float = 0.4
    rearm_decisions: int = 2
    # food-trade: last turn's imported food (observation trade row "food", column
    # "imported") over the camp's meal need per turn (privileged economy meals-per-turn).
    food_import_share: float = 0.2

    def __post_init__(self) -> None:
        if (
            int(self.head_window) < 1
            or int(self.daughter_window) < 1
            or int(self.first_use_window) < 1
        ):
            raise ValueError("scenario windows must be positive")
        if not self.small_below < self.large_above:
            raise ValueError("small-camp bound must be below the large-camp bound")
        if self.housing_exit < self.housing_below or self.clothing_exit < self.clothing_below:
            raise ValueError("hysteresis exit thresholds must not be below the entry thresholds")
        if int(self.rearm_decisions) < 0:
            raise ValueError("rearm decisions must be nonnegative")

    def state_dict(self) -> dict[str, Any]:
        return {
            "head_window": int(self.head_window),
            "daughter_window": int(self.daughter_window),
            "small_below": float(self.small_below),
            "large_above": float(self.large_above),
            "housing_below": float(self.housing_below),
            "clothing_below": float(self.clothing_below),
            "trade_partners_min": float(self.trade_partners_min),
            "require_neolithic": bool(self.require_neolithic),
            "first_use_window": int(self.first_use_window),
            "housing_exit": float(self.housing_exit),
            "clothing_exit": float(self.clothing_exit),
            "rearm_decisions": int(self.rearm_decisions),
            "food_import_share": float(self.food_import_share),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any] | None) -> DetectionRules:
        if not state:
            return cls()
        return cls(**{k: state[k] for k in cls().state_dict() if k in state})


@dataclass(frozen=True)
class CampState:
    """The detection inputs of one camp at one decision boundary (None: not observed)."""

    camp: int
    turn: int
    food_heads: tuple[bool, bool, bool]
    residents: float | None = None
    neolithic: bool | None = None
    housing: float | None = None
    clothing: float | None = None
    trade_partners: float | None = None
    # Food heads the camp's valid, accepted teacher label staffs (None: no valid label).
    staffed: tuple[bool, bool, bool] | None = None
    # Imported food last turn over meals per turn (None: not observable).
    food_import_share: float | None = None


def _feature_columns() -> dict[str, int]:
    from .labor_privileged import LABOR_PRIVILEGED_LAYOUT_V1 as layout

    economy = layout.family("economy").features
    unlocks = layout.family("unlocks").features
    return {
        "residents": economy.index("residents"),
        "housing": economy.index("housing-per-capita"),
        "clothing": economy.index("cloth-per-capita"),
        "trade_partners": economy.index("trade-partners"),
        "meals": economy.index("meals-per-turn"),
        "neolithic": unlocks.index("era-neolithic"),
    }


def _head_columns() -> tuple[int, int, int]:
    from .labor_bridge import SPECIALTIES

    return tuple(SPECIALTIES.index(head) for _, head in HEAD_FAMILIES)  # type: ignore[return-value]


def _trade_columns() -> tuple[int, int, int]:
    from . import contracts

    names = [name for name, _, _ in contracts.SETS]
    # Trade feature 6 is "imported" (simulator ElderModel9.TradeFeatureNames).
    return names.index("trade"), contracts.TRADE_GROUPS.index("food"), 6


def _food_imported(arrays: Mapping[str, np.ndarray], normalizer: Any) -> np.ndarray | None:
    """Raw food imported last turn per row, or None when it cannot be recovered.

    Live rows carry the raw observation trade set (``trade_raw``). Stored rows carry the
    bounded-normalized ``trade`` set, inverted with the dataset normalizer
    (``y = z std + mean``, ``x = sign(y) expm1|y|``); a dead (zeroed) column is unknown.
    """
    from . import contracts

    set_index, food_row, imported = _trade_columns()
    _, rows, width = contracts.SETS[set_index]
    present = arrays.get("trade_present")
    if "trade_raw" in arrays:
        table = np.asarray(arrays["trade_raw"], dtype=np.float64).reshape(-1, rows, width)
        value = table[:, food_row, imported]
    elif "trade" in arrays and normalizer is not None:
        table = np.asarray(arrays["trade"], dtype=np.float64).reshape(-1, rows, width)
        dead = (getattr(normalizer, "dead", {}) or {}).get("trade")
        if dead is not None and bool(np.asarray(dead).reshape(-1)[imported]):
            return None
        mean = float(np.asarray(normalizer.set_mean["trade"]).reshape(-1)[imported])
        std = float(np.asarray(normalizer.set_std["trade"]).reshape(-1)[imported])
        y = table[:, food_row, imported] * std + mean
        value = np.sign(y) * np.expm1(np.abs(y))
    else:
        return None
    if present is not None:
        live = np.asarray(present, dtype=bool).reshape(len(value), -1)[:, food_row]
        value = np.where(live, value, 0.0)
    return np.maximum(value, 0.0)


def food_trade_tagging(dataset: Mapping[str, Any], normalizer: Any) -> str:
    """Whether the food-trade guard is observable for stored rows, and why not."""
    if normalizer is None:
        return "unavailable: no normalizer to recover imported food"
    if "trade" not in dataset:
        return "unavailable: rows carry no trade set"
    dead = (getattr(normalizer, "dead", {}) or {}).get("trade")
    if dead is not None and bool(np.asarray(dead).reshape(-1)[_trade_columns()[2]]):
        return (
            "unavailable: the normalizer zeroed the trade 'imported' column (constant where it "
            "was fitted), so stored rows lost imported food"
        )
    return "available: imported food (trade row food) over meals-per-turn"


def states_from_arrays(
    arrays: Mapping[str, Any],
    rows: Sequence[int] | np.ndarray,
    population: Any = None,
    normalizer: Any = None,
) -> list[CampState]:
    """Camp states of stored rows (teacher shard format); privileged arrays are optional.

    ``normalizer`` (the dataset's bounded normalizer) recovers raw imported food from the
    normalized ``trade`` set for the food-trade family; without it food-trade is unknown.
    """
    selected = np.asarray(rows, dtype=np.int64).reshape(-1)
    # Index the requested rows first: callers pass whole-bank arrays per decision.
    np_ = {k: np.asarray(v)[selected] for k, v in arrays.items() if k in _ARRAY_KEYS}
    if population is not None:
        population = np.asarray(population).reshape(-1)[selected]
    rows = np.arange(len(selected))
    heads = np_["labor_specialty_mask"].astype(bool)[:, list(_head_columns())]
    columns = _feature_columns()
    food_imported = _food_imported(np_, normalizer)
    staffing = labels_valid = quarantined = None
    if "target_specialty" in np_:
        staffing = np_["target_specialty"][:, list(_head_columns())]
        labels_valid = np_.get("label_specialty_valid")
        quarantined = np_.get("row_quarantined")
    economy = unlocks = None
    if "privileged_economy" in np_ and "privileged_unlocks" in np_:
        raw_economy, raw_unlocks = np_["privileged_economy"], np_["privileged_unlocks"]
        economy = raw_economy.reshape(len(heads), -1, raw_economy.shape[-1])
        unlocks = raw_unlocks.reshape(len(heads), -1, raw_unlocks.shape[-1])
        economy_present = np_.get("privileged_economy_present")
        unlocks_present = np_.get("privileged_unlocks_present")
    out = []
    for r in rows.tolist():
        residents = housing = clothing = trade = None
        neolithic = None
        food_share = None
        if economy is not None and (
            economy_present is None or bool(economy_present[r].reshape(-1)[0])
        ):
            row = economy[r, 0]
            residents = float(row[columns["residents"]])
            housing = float(row[columns["housing"]])
            clothing = float(row[columns["clothing"]])
            trade = float(row[columns["trade_partners"]])
            meals = float(row[columns["meals"]])
            if food_imported is not None and meals > 0.0:
                food_share = float(food_imported[r]) / meals
        if unlocks is not None and (
            unlocks_present is None or bool(unlocks_present[r].reshape(-1)[0])
        ):
            neolithic = bool(unlocks[r, 0, columns["neolithic"]] > 0.5)
        if residents is None and population is not None:
            residents = float(np.asarray(population).reshape(-1)[r])
        staffed = None
        if staffing is not None:
            valid = labels_valid is None or bool(labels_valid.reshape(-1)[r])
            accepted = quarantined is None or not bool(quarantined.reshape(-1)[r])
            if valid and accepted:
                staffed = tuple(bool(x > 0.0) for x in staffing[r])
        out.append(
            CampState(
                camp=int(np_["row_camp"].reshape(-1)[r]),
                turn=int(np_["row_turn"].reshape(-1)[r]),
                food_heads=tuple(bool(x) for x in heads[r]),  # type: ignore[arg-type]
                residents=residents,
                neolithic=neolithic,
                housing=housing,
                clothing=clothing,
                trade_partners=trade,
                staffed=staffed,  # type: ignore[arg-type]
                food_import_share=food_share,
            )
        )
    return out


_ARRAY_KEYS = {
    "trade",
    "trade_present",
    "trade_raw",
    "target_specialty",
    "label_specialty_valid",
    "row_quarantined",
    "labor_specialty_mask",
    "privileged_economy",
    "privileged_economy_present",
    "privileged_unlocks",
    "privileged_unlocks_present",
    "row_camp",
    "row_turn",
}


def states_from_live(
    live: Sequence[Mapping[str, Any]],
    labels: Sequence[Any] | None = None,
    accepted: Sequence[bool] | None = None,
) -> list[CampState]:
    """Camp states of live bridge village rows at a decision boundary (pre-action).

    ``labels`` (converted teacher labels aligned with ``live``) and ``accepted``
    (initial recommendation valid) supply the first-active-use inputs.
    """
    from .labor_bridge import labor_observation_fields
    from .labor_privileged import privileged_raw_batch

    catalogs = [labor_observation_fields(v["observation"])[0] for v in live]
    arrays: dict[str, Any] = {
        "labor_specialty_mask": np.asarray([c.masks.specialty for c in catalogs], dtype=bool),
        "row_camp": np.asarray([int(v["camp"]) for v in live], dtype=np.int64),
        "row_turn": np.asarray([int(c.turn) for c in catalogs], dtype=np.int64),
    }
    if all(isinstance(v, Mapping) and v.get("privilegedView") is not None for v in live):
        arrays.update(privileged_raw_batch(live))
    set_index, _, _ = _trade_columns()
    raw_trade, raw_present = [], []
    for village in live:
        observation = village["observation"]
        sets, present = observation.get("sets"), observation.get("rowPresent")
        if not sets or not present or len(sets) <= set_index:
            raw_trade = []
            break
        raw_trade.append(np.asarray(sets[set_index], dtype=np.float64).reshape(-1))
        raw_present.append(np.asarray(present[set_index], dtype=bool).reshape(-1))
    if raw_trade:
        arrays["trade_raw"] = np.stack(raw_trade)
        arrays["trade_present"] = np.stack(raw_present)
    population = np.asarray([c.population for c in catalogs], dtype=np.float64)
    if labels is not None:
        arrays["target_specialty"] = np.asarray([lab.specialty for lab in labels], dtype=np.float64)
        arrays["label_specialty_valid"] = np.asarray(
            [bool(lab.specialty_valid) for lab in labels], dtype=bool
        )
        if accepted is not None:
            arrays["row_quarantined"] = ~np.asarray(accepted, dtype=bool)
    return states_from_arrays(arrays, np.arange(len(live)), population)


def states_from_transition(transition: Any) -> list[CampState]:
    """Pre-action camp states of one teacher decision with its converted labels."""
    attempts = transition.execution.initial_attempts
    return states_from_live(
        transition.live,
        transition.labels,
        [bool(attempts[int(v["camp"])].valid) for v in transition.live],
    )


# ---------------------------------------------------------------------------
# Family tracker (shared by the coverage audit and the harvest)
# ---------------------------------------------------------------------------


@dataclass
class _CampHistory:
    reset_camp: bool
    observations: int = 0
    previous_heads: tuple[bool, ...] | None = None
    head_counter: list[int | None] = field(default_factory=lambda: [None, None, None])
    head_kind: list[str | None] = field(default_factory=lambda: [None, None, None])
    mixed_counter: int | None = None
    mixed_kind: str | None = None
    previous_tags: frozenset[str] = frozenset()
    used: list[bool] = field(default_factory=lambda: [False, False, False])
    use_counter: list[int | None] = field(default_factory=lambda: [None, None, None])
    # Hysteresis per family: [low, armed, decisions above exit].
    hysteresis: dict[str, list[Any]] = field(
        default_factory=lambda: {name: [False, True, 0] for name in HYSTERESIS_FAMILIES}
    )


@dataclass(frozen=True)
class Observation:
    tags: frozenset[str]
    entries: frozenset[str]
    # Entry subtype per entered family: "reset"/"in-trajectory" (head and mixed
    # windows), "initial"/"relapse" (under-housed/under-clothed).
    subtypes: Mapping[str, str]


class FamilyTracker:
    """Per-trajectory family detection; ``observe`` is called once per decision boundary."""

    def __init__(self, rules: DetectionRules | None = None) -> None:
        self.rules = rules or DetectionRules()
        self.first_decision: int | None = None
        self.camps: dict[int, _CampHistory] = {}

    def observe(self, decision: int, states: Sequence[CampState]) -> dict[int, Observation]:
        rules = self.rules
        if self.first_decision is None:
            self.first_decision = int(decision)
        out: dict[int, Observation] = {}
        for state in states:
            history = self.camps.get(state.camp)
            if history is None:
                history = _CampHistory(reset_camp=int(decision) == self.first_decision)
                self.camps[state.camp] = history
            first = history.observations == 0
            heads = tuple(bool(x) for x in state.food_heads)
            tags: set[str] = set()
            kinds: dict[str, str] = {}
            for index, (name, _) in enumerate(HEAD_FAMILIES):
                if not heads[index]:
                    history.head_counter[index] = None
                    continue
                newly = history.previous_heads is not None and not history.previous_heads[index]
                if newly or (first and history.reset_camp):
                    history.head_counter[index] = 0
                    history.head_kind[index] = "reset" if first else "in-trajectory"
                counter = history.head_counter[index]
                if counter is not None:
                    if counter < rules.head_window:
                        tags.add(name)
                        kinds[name] = str(history.head_kind[index])
                    history.head_counter[index] = counter + 1
            count = sum(heads)
            if count < 2:
                history.mixed_counter = None
            else:
                previous = None if history.previous_heads is None else sum(history.previous_heads)
                if (previous is not None and previous < 2) or (first and history.reset_camp):
                    history.mixed_counter = 0
                    history.mixed_kind = "reset" if first else "in-trajectory"
                if history.mixed_counter is not None:
                    if history.mixed_counter < rules.head_window:
                        tags.add("mixed")
                        kinds["mixed"] = str(history.mixed_kind)
                    history.mixed_counter += 1
            if state.residents is not None:
                if state.residents < rules.small_below:
                    tags.add("small-camp")
                if state.residents > rules.large_above:
                    tags.add("large-camp")
            hysteresis_entries: set[str] = set()
            for name, value, below, exit_ in (
                ("under-housed", state.housing, rules.housing_below, rules.housing_exit),
                ("under-clothed", state.clothing, rules.clothing_below, rules.clothing_exit),
            ):
                memory = history.hysteresis[name]
                if value is None:
                    continue
                if memory[0]:
                    if value > exit_:
                        memory[0], memory[2] = False, 1
                elif value < below:
                    memory[0] = True
                    if memory[1] or memory[2] >= rules.rearm_decisions:
                        hysteresis_entries.add(name)
                        memory[1] = False
                    memory[2] = 0
                elif value > exit_:
                    memory[2] += 1
                else:
                    memory[2] = 0
                if memory[0]:
                    tags.add(name)
                    kinds[name] = "initial" if first else "relapse"
            for index, (name, _) in enumerate(FIRST_USE_FAMILIES):
                counter = history.use_counter[index]
                staffs = state.staffed is not None and state.staffed[index]
                if counter is None and staffs and not history.used[index]:
                    history.used[index] = True
                    counter = 0
                if counter is not None:
                    if counter < rules.first_use_window:
                        tags.add(name)
                        history.use_counter[index] = counter + 1
                    else:
                        history.use_counter[index] = None
            if not history.reset_camp and history.observations < rules.daughter_window:
                tags.add("fission-daughter")
            trade = state.trade_partners
            if trade is not None and trade >= rules.trade_partners_min:
                tags.add("trade-eligible")
            share = state.food_import_share
            if share is not None and share > rules.food_import_share:
                tags.add("food-trade")
            if rules.require_neolithic and state.neolithic is False:
                tags = set()
                hysteresis_entries = set()
            frozen = frozenset(tags)
            entries = (frozen - history.previous_tags - set(HYSTERESIS_FAMILIES)) | (
                hysteresis_entries & frozen
            )
            out[state.camp] = Observation(
                frozen, entries, {name: kinds[name] for name in entries if name in kinds}
            )
            history.previous_tags = frozen
            history.previous_heads = heads
            history.observations += 1
        return out


# ---------------------------------------------------------------------------
# Coverage audit
# ---------------------------------------------------------------------------

_COVERAGE_KEYS = _ARRAY_KEYS | {
    "row_world",
    "row_source",
    "row_quarantined",
    "row_trajectory",
    "row_start_index",
    "row_scenario_tags",
    "row_scenario_family",
    "target_specialty",
    "label_specialty_valid",
}


def _load_shard_arrays(shard: Path, meta: Mapping[str, Any]) -> dict[str, np.ndarray]:
    names = {name[: -len(".npy")] for name in meta["payload"]}
    return {
        key: np.load(shard / f"{key}.npy", allow_pickle=False)
        for key in sorted(names & _COVERAGE_KEYS)
    }


@dataclass
class _FamilyCount:
    rows: int = 0
    roots: int = 0
    root_worlds: set[int] = field(default_factory=set)
    row_worlds: set[int] = field(default_factory=set)
    subtypes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sources: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class _Coverage:
    families: dict[str, dict[str, _FamilyCount]] = field(
        default_factory=lambda: defaultdict(lambda: {f: _FamilyCount() for f in FAMILIES})
    )
    rows: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    worlds: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    first_use: dict[str, dict[str, Any]] = field(
        default_factory=lambda: defaultdict(
            lambda: {head: {"camps": 0, "tags": defaultdict(int)} for _, head in HEAD_FAMILIES}
        )
    )
    datasets: list[dict[str, Any]] = field(default_factory=list)


def _first_active_use(
    cov: _Coverage, partition: str, arrays: Mapping[str, np.ndarray], order: np.ndarray,
    tags: Mapping[int, frozenset[str]], seen: set[tuple[int, int, int, str]], trajectory: int,
) -> None:
    """Per (trajectory, head): the first valid teacher label staffing that head."""
    if "target_specialty" not in arrays:
        return
    target = arrays["target_specialty"]
    valid = arrays.get("label_specialty_valid")
    quarantined = arrays.get("row_quarantined")
    columns = dict(zip((h for _, h in HEAD_FAMILIES), _head_columns(), strict=True))
    for r in order.tolist():
        if quarantined is not None and bool(quarantined.reshape(-1)[r]):
            continue
        if valid is not None and not bool(valid.reshape(-1)[r]):
            continue
        world = int(arrays["row_world"].reshape(-1)[r])
        camp = int(arrays["row_camp"].reshape(-1)[r])
        for head, column in columns.items():
            key = (world, camp, trajectory, head)
            if key in seen or float(target[r, column]) <= 0.0:
                continue
            seen.add(key)
            entry = cov.first_use[partition][head]
            entry["camps"] += 1
            for name in tags.get(r, frozenset()):
                entry["tags"][name] += 1


def _audit_shard(
    cov: _Coverage,
    arrays: Mapping[str, np.ndarray],
    rules: DetectionRules,
    label: str,
    normalizer: Any = None,
) -> None:
    worlds = arrays["row_world"].reshape(-1)
    n = len(worlds)
    sources = arrays.get("row_source")
    sources = np.zeros(n, dtype=np.int64) if sources is None else sources.reshape(-1)
    trajectories = arrays.get("row_trajectory")
    trajectories = (
        np.full(n, -1, dtype=np.int64) if trajectories is None else trajectories.reshape(-1)
    )
    turns = arrays["row_turn"].reshape(-1)
    for world in np.unique(worlds).tolist():
        partition = world_partition(world)
        cov.worlds[partition].add(int(world))
        in_world = worlds == world
        cov.rows[partition] += int(in_world.sum())
        for source in np.unique(sources[in_world]).tolist():
            selected = in_world & (sources == source)
            if int(source) == ROW_SOURCE and "row_scenario_tags" in arrays:
                _audit_scenario_rows(cov, arrays, np.flatnonzero(selected), partition, label)
                continue
            # Teacher-format trajectories: (world, trajectory) in turn order, tracked per camp.
            first_use_seen: set[tuple[int, int, int, str]] = set()
            for trajectory in np.unique(trajectories[selected]).tolist():
                rows = np.flatnonzero(selected & (trajectories == trajectory))
                rows = rows[np.lexsort((arrays["row_camp"].reshape(-1)[rows], turns[rows]))]
                tracker = FamilyTracker(rules)
                tags_by_row: dict[int, frozenset[str]] = {}
                unique_turns = np.unique(turns[rows])
                for decision, turn in enumerate(unique_turns.tolist()):
                    at = rows[turns[rows] == turn]
                    states = states_from_arrays(arrays, at, normalizer=normalizer)
                    observed = tracker.observe(decision, states)
                    for r, state in zip(at.tolist(), states, strict=True):
                        obs = observed[state.camp]
                        tags_by_row[r] = obs.tags
                        for name in obs.tags:
                            count = cov.families[partition][name]
                            count.rows += 1
                            count.row_worlds.add(int(world))
                        for name in obs.entries:
                            count = cov.families[partition][name]
                            count.roots += 1
                            count.root_worlds.add(int(world))
                            count.sources[label] += 1
                            if name in obs.subtypes:
                                count.subtypes[obs.subtypes[name]] += 1
                _first_active_use(
                    cov, partition, arrays, rows, tags_by_row, first_use_seen, int(trajectory)
                )


def _audit_scenario_rows(
    cov: _Coverage, arrays: Mapping[str, np.ndarray], rows: np.ndarray, partition: str, label: str
) -> None:
    """Scenario rows carry their tags; a root counts once per family of its root mask."""
    tags = arrays["row_scenario_tags"].reshape(-1)
    roots_mask = arrays["row_scenario_family"].reshape(-1)
    starts = arrays["row_start_index"].reshape(-1)
    worlds = arrays["row_world"].reshape(-1)
    seen: set[tuple[int, int]] = set()
    for r in rows.tolist():
        world = int(worlds[r])
        for name in family_names(int(tags[r])):
            count = cov.families[partition][name]
            count.rows += 1
            count.row_worlds.add(world)
        key = (world, int(starts[r]))
        if key in seen:
            continue
        seen.add(key)
        for name in family_names(int(roots_mask[r])):
            count = cov.families[partition][name]
            count.roots += 1
            count.root_worlds.add(world)
            count.sources[label] += 1


def row_family_tags(
    dataset: Mapping[str, Any],
    rules: DetectionRules | None = None,
    tracked_sources: Sequence[int] = (0,),
    normalizer: Any = None,
) -> np.ndarray | None:
    """Per-row family bitmask of a loaded labor dataset (torch or numpy arrays).

    Curated scenario rows use their stored ``row_scenario_tags``; teacher
    trajectories (``tracked_sources``, default ``row_source`` 0: chronological
    (world, trajectory) episodes) are tracked in turn order with the harvest's
    rules. Rows of other sources (DAgger branches, search labels, continuations)
    are untagged (0). ``None`` when the dataset lacks the specialty mask.
    """
    rules = rules or DetectionRules()

    def array(key: str) -> np.ndarray | None:
        value = dataset.get(key)
        if value is None:
            return None
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    arrays = {key: array(key) for key in _COVERAGE_KEYS}
    arrays = {k: v for k, v in arrays.items() if v is not None}
    if "labor_specialty_mask" not in arrays or "row_world" not in arrays:
        return None
    n = len(arrays["row_world"].reshape(-1))
    out = np.zeros(n, dtype=np.int64)
    worlds = arrays["row_world"].reshape(-1)
    sources = arrays.get("row_source", np.zeros(n, dtype=np.int64)).reshape(-1)
    trajectories = arrays.get("row_trajectory", np.full(n, -1, dtype=np.int64)).reshape(-1)
    turns = arrays["row_turn"].reshape(-1)
    camps = arrays["row_camp"].reshape(-1)
    scenario = sources == ROW_SOURCE
    if scenario.any() and "row_scenario_tags" in arrays:
        out[scenario] = arrays["row_scenario_tags"].reshape(-1)[scenario]
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    tracked = np.isin(sources, np.asarray(list(tracked_sources), dtype=sources.dtype))
    for r in np.flatnonzero(~scenario & tracked).tolist():
        groups[(int(worlds[r]), int(sources[r]), int(trajectories[r]))].append(r)
    for rows in groups.values():
        rows_np = np.asarray(rows, dtype=np.int64)
        rows_np = rows_np[np.lexsort((camps[rows_np], turns[rows_np]))]
        tracker = FamilyTracker(rules)
        for decision, turn in enumerate(np.unique(turns[rows_np]).tolist()):
            at = rows_np[turns[rows_np] == turn]
            states = states_from_arrays(arrays, at, normalizer=normalizer)
            observed = tracker.observe(decision, states)
            for r, state in zip(at.tolist(), states, strict=True):
                out[r] = family_mask(observed[state.camp].tags)
    return out


AUDITED_SOURCE_KINDS = ("native_heuristic_labor_teacher", SOURCE_KIND)
# Families decidable from one decision boundary alone (no trajectory history).
STATE_FAMILIES = (
    "small-camp",
    "large-camp",
    "under-housed",
    "under-clothed",
    "trade-eligible",
    "food-trade",
)


def root_state_families(candidates: Any, normalizer: Any = None) -> list[tuple[str, ...]]:
    """State-only families of each search root (``labor_advantage.CandidateSet``).

    Roots carry one boundary (actor tensors and raw privileged arrays), so only
    :data:`STATE_FAMILIES` apply (housing and clothing at the entry threshold).
    """
    tensors = {
        k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
        for k, v in dict(candidates.tensors).items()
    }
    for k, v in dict(candidates.privileged or {}).items():
        tensors[k] = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    if "labor_specialty_mask" not in tensors:
        return [() for _ in candidates.roots]
    out = []
    for root, row in zip(candidates.roots, candidates.rows, strict=True):
        arrays = {k: v[row : row + 1] for k, v in tensors.items() if k in _ARRAY_KEYS}
        arrays["row_camp"] = np.asarray([root.camp])
        arrays["row_turn"] = np.asarray([root.turn])
        state = states_from_arrays(arrays, [0], normalizer=normalizer)[0]
        tags = FamilyTracker(DetectionRules()).observe(0, [state])[state.camp].tags
        out.append(tuple(name for name in STATE_FAMILIES if name in tags))
    return out


# ---------------------------------------------------------------------------
# Family row up-weighting in existing banks (no new collection)
# ---------------------------------------------------------------------------

# Defaults (co-architect proposal, approved by the user on 26 Oct with decay required).
PROPOSED_FAMILY_WEIGHTS = {
    "first-use-farming": 10.0,
    "first-use-herding": 10.0,
    "first-use-fishing": 10.0,
    "farming-unlocked": 5.0,
    "herding-unlocked": 5.0,
    "fishing-unlocked": 5.0,
    "mixed": 5.0,
    "fission-daughter": 2.0,
    "under-housed": 3.0,
    "under-clothed": 3.0,
}
FAMILY_WEIGHTS_STATUS = "approved-with-decay"
FAMILY_WEIGHTING_RULE = (
    "row multiplier = max weight over the row families (1 untagged); then each "
    "(row_source, world) group is rescaled to its unweighted mass, preserving world "
    "balance and source mass"
)


def parse_family_weights(text: str | None) -> dict[str, float] | None:
    """``default`` (the proposed weights), ``family=w,...``, or ``default,family=w`` overrides."""
    if text is None or not str(text).strip():
        return None
    weights: dict[str, float] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if item == "default":
            weights.update(PROPOSED_FAMILY_WEIGHTS)
            continue
        name, sep, value = item.partition("=")
        if not sep or name not in FAMILY_BITS:
            raise ValueError(f"family weight {item!r}: expected family=w with a known family")
        if name in NEVER_UPWEIGHTED:
            raise ValueError(f"family {name!r} is a guard family and is never up-weighted")
        w = float(value)
        if not math.isfinite(w) or w <= 0.0:
            raise ValueError(f"family weight {item!r} must be positive and finite")
        weights[name] = w
    return {name: weights[name] for name in FAMILIES if name in weights}


DECAYS = ("none", "linear", "cosine")


def decay_factor(kind: str, progress_epochs: float, decay_epochs: float) -> float:
    """s(t) in [0, 1]: 1 at the start, 0 after ``decay_epochs`` epochs (``none``: always 1)."""
    if kind not in DECAYS:
        raise ValueError(f"family weight decay must be one of {DECAYS}")
    if kind == "none":
        return 1.0
    if decay_epochs <= 0:
        raise ValueError("family weight decay epochs must be positive")
    x = min(1.0, max(0.0, float(progress_epochs) / float(decay_epochs)))
    if kind == "linear":
        return 1.0 - x
    return 0.5 * (1.0 + math.cos(math.pi * x))


@dataclass(frozen=True)
class FamilyWeightSchedule:
    """Each up-weight w acts as 1 + (w - 1) s(t), keyed on epochs of optimizer steps.

    ``epoch_offset`` continues a schedule across fits (DAgger and refit rounds read it
    from the input checkpoint), so a later round never restarts at full weight.
    """

    weights: Mapping[str, float]
    decay: str = "cosine"
    decay_epochs: float = 1.0
    epoch_offset: float = 0.0

    def __post_init__(self) -> None:
        decay_factor(self.decay, 0.0, max(1e-9, float(self.decay_epochs)))
        if float(self.epoch_offset) < 0:
            raise ValueError("family weight schedule offset must be nonnegative")

    def factor(self, epochs_into_fit: float) -> float:
        progress = float(self.epoch_offset) + float(epochs_into_fit)
        return decay_factor(self.decay, progress, float(self.decay_epochs))

    def multipliers(self, epochs_into_fit: float) -> dict[str, float]:
        s = self.factor(epochs_into_fit)
        return {name: 1.0 + (float(w) - 1.0) * s for name, w in self.weights.items()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "decay": self.decay,
            "decay_epochs": float(self.decay_epochs),
            "epoch_offset": float(self.epoch_offset),
            "status": FAMILY_WEIGHTS_STATUS,
        }


def continued_schedule(
    weights: Mapping[str, float] | None,
    decay: str | None,
    decay_epochs: float | None,
    fit_epochs: int,
    previous: Mapping[str, Any] | None,
    *,
    restart: bool = False,
) -> FamilyWeightSchedule | None:
    """The fit's schedule: continued from ``previous`` (same weights and decay) unless restarted.

    Default decay is cosine; default decay epochs are the previous schedule's, else this
    fit's epoch count.
    """
    if not weights:
        return None
    decay = decay or "cosine"
    same = (
        previous is not None
        and not restart
        and dict(previous.get("weights", {})) == dict(weights)
        and previous.get("decay") == decay
    )
    if decay_epochs is None:
        decay_epochs = float(previous["decay_epochs"]) if same else float(fit_epochs)
    offset = float(previous.get("epochs_end", 0.0)) if same else 0.0
    return FamilyWeightSchedule(dict(weights), decay, float(decay_epochs), offset)


class FamilyRowWeighter:
    """Closed-form decayed row weights: base (1 + (m - 1) s), each (source, world) at its mass.

    With m the max family weight of a row, a group of base mass B and multiplied mass M
    has rescale B / (B + s (M - B)), so no per-step pass over the groups is needed.
    """

    def __init__(
        self, base: Any, tags: Any, sources: Any, worlds: Any, weights: Mapping[str, float]
    ) -> None:
        self.base = np.asarray(base, dtype=np.float64).reshape(-1)
        self.m = family_multipliers(tags, weights)
        sources = np.asarray(sources).reshape(-1)
        worlds = np.asarray(worlds).reshape(-1)
        keys = np.stack([sources.astype(np.int64), worlds.astype(np.int64)], axis=1)
        _, self.group = np.unique(keys, axis=0, return_inverse=True)
        self.group = self.group.reshape(-1)
        groups = int(self.group.max()) + 1 if len(self.group) else 0
        self.mass = np.bincount(self.group, weights=self.base, minlength=groups)
        self.multiplied = np.bincount(self.group, weights=self.base * self.m, minlength=groups)

    def _scale(self, s: float) -> np.ndarray:
        denominator = self.mass + s * (self.multiplied - self.mass)
        out = np.zeros_like(self.mass)
        return np.divide(self.mass, denominator, out=out, where=denominator > 0)

    def weights(self, s: float) -> np.ndarray:
        return self.base * (1.0 + (self.m - 1.0) * s) * self._scale(s)[self.group]

    def rows(self, index: Any, s: float) -> np.ndarray:
        index = np.asarray(index, dtype=np.int64).reshape(-1)
        scale = self._scale(s)
        return self.base[index] * (1.0 + (self.m[index] - 1.0) * s) * scale[self.group[index]]


def family_multipliers(tags: np.ndarray, weights: Mapping[str, float]) -> np.ndarray:
    """Per-row multiplier: the max weight over the row families, 1 when untagged."""
    tags = np.asarray(tags, dtype=np.int64).reshape(-1)
    out = np.ones(len(tags), dtype=np.float64)
    tagged = np.zeros(len(tags), dtype=bool)
    best = np.zeros(len(tags), dtype=np.float64)
    for name, w in weights.items():
        has = (tags & FAMILY_BITS[name]) != 0
        best = np.where(has, np.maximum(best, float(w)), best)
        tagged |= has
    out[tagged] = best[tagged]
    return out


def apply_family_weights(
    base: Any, tags: Any, sources: Any, worlds: Any, weights: Mapping[str, float]
) -> np.ndarray:
    """Multiply ``base`` row weights by family multipliers, rescaling each (source, world)."""
    base = np.asarray(base, dtype=np.float64).reshape(-1)
    sources = np.asarray(sources).reshape(-1)
    worlds = np.asarray(worlds).reshape(-1)
    raw = base * family_multipliers(tags, weights)
    out = np.zeros_like(base)
    for key in {(int(s), int(w)) for s, w in zip(sources, worlds, strict=True)}:
        group = (sources == key[0]) & (worlds == key[1])
        mass, weighted = base[group].sum(), raw[group].sum()
        if weighted > 0.0:
            out[group] = raw[group] * (mass / weighted)
    return out


def family_share_report(
    before: Any, after: Any, tags: Any, keep: Any | None = None
) -> dict[str, Any]:
    """Row share and weight share of each family before and after reweighting."""
    before = np.asarray(before, dtype=np.float64).reshape(-1)
    after = np.asarray(after, dtype=np.float64).reshape(-1)
    tags = np.asarray(tags, dtype=np.int64).reshape(-1)
    keep = np.ones(len(tags), dtype=bool) if keep is None else np.asarray(keep).reshape(-1)
    total_b, total_a, rows = before[keep].sum(), after[keep].sum(), int(keep.sum())
    out: dict[str, Any] = {}
    for name in [*FAMILIES, "untagged"]:
        has = (tags == 0) if name == "untagged" else ((tags & FAMILY_BITS[name]) != 0)
        has = has & keep
        out[name] = {
            "rows": int(has.sum()),
            "row_share": float(has.sum() / rows) if rows else 0.0,
            "weight_share_before": float(before[has].sum() / total_b) if total_b else 0.0,
            "weight_share_after": float(after[has].sum() / total_a) if total_a else 0.0,
        }
    return out


def reweight_dataset(
    dataset: dict[str, Any],
    weights: Mapping[str, float],
    rules: DetectionRules | None = None,
    normalizer: Any = None,
) -> dict[str, Any]:
    """Up-weight family rows of a loaded labor dataset in place (tags computed at load time).

    Stores ``row_family_tags`` (never a model input), replaces ``row_weight`` and
    records the weights, their status and the before/after shares in
    ``collection["family_weighting"]``.
    """
    import torch

    tags = row_family_tags(dataset, rules, normalizer=normalizer)
    if tags is None:
        raise ValueError("family weighting needs labor_specialty_mask rows")
    n = len(tags)
    base = dataset["row_weight"].reshape(-1).double().numpy()
    sources = (
        dataset["row_source"].reshape(-1).numpy()
        if "row_source" in dataset
        else np.zeros(n, dtype=np.int64)
    )
    worlds = dataset["row_world"].reshape(-1).numpy()
    new = apply_family_weights(base, tags, sources, worlds, weights)
    keep = ~dataset["row_quarantined"].reshape(-1).numpy().astype(bool)
    dataset["row_family_tags"] = torch.as_tensor(tags, dtype=torch.long)
    # The unweighted loader weights stay available for decayed schedules.
    dataset["row_weight_unweighted"] = dataset["row_weight"].clone()
    dataset["row_weight"] = torch.as_tensor(new, dtype=torch.float32)
    record = {
        "weights": dict(weights),
        "status": FAMILY_WEIGHTS_STATUS,
        "rule": FAMILY_WEIGHTING_RULE,
        "rules": (rules or DetectionRules()).state_dict(),
        "tagged_sources": "teacher trajectories (row_source 0) by the tracker; curated scenario "
        "rows by their stored tags; other sources untagged",
        "shares": family_share_report(base, new, tags, keep),
        "food_trade_tagging": food_trade_tagging(dataset, normalizer),
    }
    collection = dataset.setdefault("collection", {})
    collection["family_weighting"] = record
    return record


def coverage_audit(
    directories: Sequence[Path],
    *,
    rules: DetectionRules | None = None,
    minima: Mapping[str, tuple[int, int]] | None = None,
    shards: Sequence[Path] = (),
    normalizer: Any = None,
) -> dict[str, Any]:
    """Rows, roots and worlds per family and partition across published labor datasets.

    ``normalizer`` (the banks' bounded normalizer) enables the food-trade family.

    ``shards`` adds individually verified, unpublished teacher shard directories
    (diagnostic: an interrupted collection whose dataset.json was never written).
    """
    from .labor_teacher_data import verify_dataset, verify_shard

    rules = rules or DetectionRules()
    minima = dict(DEFAULT_MINIMA if minima is None else minima)
    cov = _Coverage()
    for directory in directories:
        directory = Path(directory)
        manifest = verify_dataset(directory)
        kind = manifest["source_record"].get("source_kind")
        if kind not in AUDITED_SOURCE_KINDS:
            raise ValueError(
                f"coverage audits teacher banks and scenario sets only, not {kind!r} "
                f"({directory}): other sources are not chronological episode trajectories"
            )
        label = f"{kind}:{manifest['dataset_digest'][:12]}"
        for entry in manifest["shards"].values():
            shard = directory / entry["path"]
            meta = json.loads((shard / "meta.json").read_text())
            _audit_shard(cov, _load_shard_arrays(shard, meta), rules, label, normalizer)
        cov.datasets.append(
            {
                "path": str(directory),
                "source_kind": kind,
                "dataset_digest": manifest["dataset_digest"],
                "label": label,
                "worlds": sorted(int(w) for w in manifest["shards"]),
                "turns": manifest["source_record"].get("options", {}).get("turns"),
                "reset_options": manifest["source_record"].get("options", {}).get("reset_options"),
            }
        )
    for shard in shards:
        shard = Path(shard)
        meta = verify_shard(shard)
        label = f"unpublished-shard:{meta['data_digest'][:12]}"
        _audit_shard(cov, _load_shard_arrays(shard, meta), rules, label, normalizer)
        cov.datasets.append(
            {"path": str(shard), "source_kind": "unpublished-teacher-shard", "label": label,
             "worlds": [int(meta["world_seed"])], "turns": meta.get("turns"),
             "reset_options": meta.get("reset_options")}
        )
    partitions: dict[str, Any] = {}
    for partition in sorted(set(cov.rows) | set(cov.families)):
        min_roots, min_worlds = minima.get(partition, (0, 0))
        families = {}
        for name in FAMILIES:
            count = cov.families[partition][name]
            deficit_roots = max(0, int(min_roots) - count.roots)
            deficit_worlds = max(0, int(min_worlds) - len(count.root_worlds))
            families[name] = {
                "rows": count.rows,
                "roots": count.roots,
                "worlds": len(count.root_worlds),
                "row_worlds": len(count.row_worlds),
                "min_roots": int(min_roots),
                "min_worlds": int(min_worlds),
                "deficit_roots": deficit_roots,
                "deficit_worlds": deficit_worlds,
                "met": deficit_roots == 0 and deficit_worlds == 0,
                "subtypes": dict(sorted(count.subtypes.items())),
                "roots_by_dataset": dict(sorted(count.sources.items())),
                "root_world_list": sorted(count.root_worlds),
            }
        first_use = {}
        for head, entry in cov.first_use[partition].items():
            camps = int(entry["camps"])
            family = next(name for name, h in HEAD_FAMILIES if h == head)
            first_use[head] = {
                "first_active_use_camps": camps,
                "share_in_unlock_window": (entry["tags"][family] / camps) if camps else None,
                "share_by_family": {
                    name: entry["tags"][name] / camps for name in FAMILIES if camps
                },
            }
        partitions[partition] = {
            "rows": int(cov.rows.get(partition, 0)),
            "worlds": sorted(cov.worlds.get(partition, set())),
            "families": families,
            "below_quota": [name for name in FAMILIES if not families[name]["met"]],
            "first_active_use": first_use,
        }
    return {
        "schema": COVERAGE_SCHEMA,
        "rules": rules.state_dict(),
        "minima": {k: list(v) for k, v in sorted(minima.items())},
        "root_definition": (
            "run entry: the decision at which a camp enters a family (a maximal run of "
            "consecutive tagged decisions of one camp's trajectory counts once); scenario rows "
            "count each root once per family of its root mask"
        ),
        "datasets": cov.datasets,
        "partitions": partitions,
    }


def summarize_coverage(report: Mapping[str, Any]) -> str:
    lines = []
    for partition, section in report["partitions"].items():
        lines.append(
            f"[{partition}] rows={section['rows']} worlds={len(section['worlds'])} "
            f"below quota: {', '.join(section['below_quota']) or 'none'}"
        )
        for name, f in section["families"].items():
            sub = ", ".join(f"{k}={v}" for k, v in f["subtypes"].items())
            lines.append(
                f"  {name:18s} rows={f['rows']:6d} roots={f['roots']:5d} worlds={f['worlds']:3d} "
                f"(min {f['min_roots']}/{f['min_worlds']}) {'MET' if f['met'] else 'GAP'}"
                + (f" [{sub}]" if sub else "")
            )
        for head, f in section["first_active_use"].items():
            share = f["share_in_unlock_window"]
            lines.append(
                f"  first active use {head:7s} camps={f['first_active_use_camps']:4d} "
                f"in unlock window={'n/a' if share is None else f'{share:.2f}'}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quotas
# ---------------------------------------------------------------------------


def quota_caps(
    coverage: Mapping[str, Any] | None,
    partition: str,
    worlds: int,
    *,
    families: Sequence[str] | None = None,
    per_world: int | None = None,
    oversample: float = 1.5,
    max_per_world: int = 8,
) -> dict[str, int]:
    """Per-family, per-world root caps.

    With a coverage audit: zero for a family that meets its minima, else enough to
    close the root deficit with ``oversample`` slack spread over the harvest worlds
    (at least one, at most ``max_per_world``). ``per_world`` overrides every
    requested family's cap (pure scenario mode without an audit).
    """
    chosen = list(FAMILIES if families is None else families)
    for name in chosen:
        family_mask([name])
    if per_world is not None:
        if per_world < 0:
            raise ValueError("per-world quota must be nonnegative")
        return {name: int(per_world) for name in chosen}
    if coverage is None:
        raise ValueError("quotas need a coverage audit (--coverage) or --quota-per-world")
    section = coverage["partitions"].get(partition)
    caps = {}
    for name in chosen:
        f = None if section is None else section["families"].get(name)
        if f is not None and f["met"]:
            caps[name] = 0
            continue
        minima = coverage["minima"].get(partition, DEFAULT_MINIMA[partition])
        deficit = int(minima[0]) if f is None else int(f["deficit_roots"])
        cap = math.ceil(oversample * max(deficit, 1) / max(1, int(worlds)))
        caps[name] = int(min(max_per_world, max(1, cap)))
    return caps


def root_priority(key: int, world: int, family: str, decision: int, camp: int) -> float:
    """Keyed uniform priority of one candidate; the quota keeps the lowest ``cap`` per family."""
    rng = np.random.default_rng(
        [PRIORITY_TAG, int(key), int(world), FAMILIES.index(family), int(decision), int(camp)]
    )
    return float(rng.random())


class QuotaReservoir:
    """Keyed bottom-k selection per family; the result is independent of offer order."""

    def __init__(self, caps: Mapping[str, int], key: int, world: int) -> None:
        self.caps = {k: int(v) for k, v in caps.items() if int(v) > 0}
        self.key, self.world = int(key), int(world)
        self.selected: dict[str, list[tuple[float, int, int]]] = {f: [] for f in self.caps}
        self.offered: dict[str, int] = defaultdict(int)

    def offer(self, family: str, decision: int, camp: int) -> tuple[bool, tuple[int, int] | None]:
        """Returns (accepted, evicted (decision, camp) or None)."""
        if family not in self.caps:
            return False, None
        self.offered[family] += 1
        priority = root_priority(self.key, self.world, family, decision, camp)
        held = self.selected[family]
        item = (priority, int(decision), int(camp))
        if len(held) < self.caps[family]:
            held.append(item)
            return True, None
        worst = max(held)
        if item < worst:
            held.remove(worst)
            held.append(item)
            return True, (worst[1], worst[2])
        return False, None

    def holders(self) -> dict[tuple[int, int], dict[str, float]]:
        """(decision, camp) -> {family: priority} of every selected root."""
        out: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
        for family, held in self.selected.items():
            for priority, decision, camp in held:
                out[(decision, camp)][family] = priority
        return dict(out)

    def decisions(self) -> set[int]:
        return {decision for held in self.selected.values() for _, decision, _ in held}


def select_roots(
    candidates: Iterable[tuple[int, int, Iterable[str]]], caps: Mapping[str, int], key: int,
    world: int,
) -> dict[tuple[int, int], dict[str, float]]:
    """Offline selection over (decision, camp, entered families) candidates (tests, reports)."""
    reservoir = QuotaReservoir(caps, key, world)
    for decision, camp, families in candidates:
        for name in families:
            reservoir.offer(name, decision, camp)
    return reservoir.holders()


# ---------------------------------------------------------------------------
# Harvest and scenario collection (one world per process)
# ---------------------------------------------------------------------------


def _live(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        v for v in step["villages"] if v.get("observation") is not None and not v.get("terminated")
    ]


@dataclass
class _Fork:
    fork: int
    decision: int
    turn: int
    founded: dict[int, int]
    tracker: FamilyTracker
    observations: dict[int, Observation]
    table: Any


def _tensor(value: int) -> Any:
    import torch

    return torch.tensor([int(value)], dtype=torch.long)


def collect_scenario_world(
    job: Mapping[str, Any],
    *,
    labor_bridge_factory: Any = None,
    heuristic_bridge_factory: Any = None,
    normalizer: Any = None,
) -> dict[str, Any]:
    """Harvest one world's roots and record the teacher scenario from each kept root."""
    import torch

    from . import labor_bc
    from .labor_bc import LaborResetOptions, RandomizedTailSettings, WorldCollection
    from .labor_model import LaborPersonality
    from .labor_personality import LaborPersonalityConditioning, personality_for_world

    torch.set_num_threads(1)
    world = int(job["world_seed"])
    options = LaborResetOptions.from_mapping(job["reset_options"])
    rules = DetectionRules.from_state(job.get("rules"))
    tail = RandomizedTailSettings.from_state(job.get("return_tail"))
    if normalizer is None:
        from .labor_teacher_data import load_normalizer

        normalizer = load_normalizer(Path(job["normalizer_checkpoint"]))
    personality: Any = LaborPersonality.from_mapping(job["personality"])
    conditioning = LaborPersonalityConditioning.from_state(job.get("personality_conditioning"))
    if conditioning.alpha is not None:
        personality = personality_for_world(conditioning.assignment(0), world)
    if labor_bridge_factory is None:
        from .labor_bridge import LaborModel9Bridge

        labor_bridge_factory = LaborModel9Bridge
    if heuristic_bridge_factory is None:
        from .bridge import Model9Bridge

        heuristic_bridge_factory = lambda: Model9Bridge(economic_diagnostics=True)  # noqa: E731
    excluded = {(int(t), int(c)) for t, c in job.get("excluded", [])}
    caps = dict(job["caps"])
    decisions = int(job["decisions"])
    harvest_start = int(job.get("harvest_start", 0))
    scenario_decisions = int(job["scenario_decisions"])
    reservoir = QuotaReservoir(caps, int(job["root_key"]), world)
    stats: dict[str, Any] = {
        "candidates": defaultdict(int),
        "excluded_candidates": 0,
        "before_harvest_start": 0,
        "forks_taken": 0,
        "forks_held_max": 0,
    }
    started = time.perf_counter()
    forks: dict[int, _Fork] = {}
    rows: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    harvest = {"decisions": 0, "native_resolves": 0}
    with labor_bridge_factory() as labor_bridge, heuristic_bridge_factory() as heur_bridge:
        step = labor_bridge.reset(world, **options.reset_kwargs())
        identity = getattr(labor_bridge, "last_reset_identity", None)
        reset_identity = dict(identity) if isinstance(identity, Mapping) else {}
        founded = {int(row["camp"]): int(step["turn"]) for row in step["villages"]}
        tracker = FamilyTracker(rules)

        def release_unreferenced() -> None:
            keep = reservoir.decisions()
            for decision in [d for d in forks if d not in keep]:
                labor_bridge.drop(forks.pop(decision).fork)

        for decision in range(decisions):
            live = _live(step)
            if not live:
                break
            turn = int(step["turn"])
            # First-active-use tags need this decision's teacher labels, so the boundary
            # is forked speculatively before the transition and kept only if a root here
            # enters a quota.
            speculative = labor_bridge.fork()[0] if reservoir.caps else None
            snapshot = (dict(founded), copy.deepcopy(getattr(personality, "table", None)))
            transition = labor_bc._teacher_transition(
                labor_bridge=labor_bridge,
                heur_bridge=heur_bridge,
                step=step,
                world_seed=world,
                options=options,
                normalizer=normalizer,
                personality=personality,
                founded=founded,
            )
            if transition is None:
                if speculative is not None:
                    labor_bridge.drop(speculative)
                break
            harvest["decisions"] += 1
            harvest["native_resolves"] += 2
            observed = tracker.observe(decision, states_from_transition(transition))
            accepted = False
            for camp, obs in sorted(observed.items()):
                if not obs.entries:
                    continue
                for name in sorted(obs.entries):
                    stats["candidates"][name] += 1
                if decision < harvest_start:
                    stats["before_harvest_start"] += 1
                    continue
                if (turn, camp) in excluded:
                    stats["excluded_candidates"] += 1
                    continue
                for name in sorted(obs.entries):
                    ok, _ = reservoir.offer(name, decision, camp)
                    accepted = accepted or ok
            if accepted:
                stats["forks_taken"] += 1
                forks[decision] = _Fork(
                    speculative,
                    decision,
                    turn,
                    snapshot[0],
                    copy.deepcopy(tracker),
                    dict(observed),
                    snapshot[1],
                )
            elif speculative is not None:
                labor_bridge.drop(speculative)
            release_unreferenced()
            stats["forks_held_max"] = max(stats["forks_held_max"], len(forks))
            step = transition.next_step
        harvest["wall_seconds"] = time.perf_counter() - started
        holders = reservoir.holders()
        root_index = 0
        for decision in sorted(forks):
            saved = forks[decision]
            camps = sorted(c for (d, c) in holders if d == decision)
            group_started = time.perf_counter()
            step = labor_bridge.restore(saved.fork)
            if int(step["turn"]) != saved.turn:
                raise ValueError("scenario fork restore returned a different decision turn")
            if saved.table is not None:
                personality.table = copy.deepcopy(saved.table)
            group_founded = dict(saved.founded)
            group_tracker = copy.deepcopy(saved.tracker)
            temp = WorldCollection(world_seed=world)
            row_tags: list[int] = []
            alive = set(camps)
            observations = saved.observations
            for index in range(scenario_decisions):
                transition = labor_bc._teacher_transition(
                    labor_bridge=labor_bridge,
                    heur_bridge=heur_bridge,
                    step=step,
                    world_seed=world,
                    options=options,
                    normalizer=normalizer,
                    personality=personality,
                    founded=group_founded,
                )
                if transition is None:
                    break
                if index > 0:
                    observations = group_tracker.observe(
                        decision + index, states_from_transition(transition)
                    )
                temp.native_resolves += 2
                temp.decisions += 1
                recorded = alive & {context[0].camp_id for context in transition.contexts}
                before = len(temp.rows)
                labor_bc.record_teacher_transition(
                    temp, transition, world, group_founded, camps=recorded
                )
                for row in temp.rows[before:]:
                    camp = int(row["row_camp"].reshape(-1)[0])
                    obs = observations.get(camp)
                    row_tags.append(family_mask(obs.tags) if obs is not None else 0)
                    row["row_branch_step"] = _tensor(index)
                    if bool(row["transition_terminal"].reshape(-1)[0]):
                        alive.discard(camp)
                alive &= recorded
                step = transition.next_step
                if not alive:
                    break
            window_seconds = time.perf_counter() - group_started
            entries = None
            if tail is not None and temp.rows:
                entries = labor_bc._collect_teacher_tail(
                    result=temp,
                    tail=tail,
                    labor_bridge=labor_bridge,
                    heur_bridge=heur_bridge,
                    step=step,
                    options=options,
                    normalizer=normalizer,
                    personality=personality,
                    founded=group_founded,
                    window_turns=scenario_decisions,
                )
                labor_bc._attach_tail_rows(temp.rows, entries)
            labor_bridge.drop(saved.fork)
            index_of = {}
            for camp in camps:
                index_of[camp] = root_index
                obs = saved.observations[camp]
                selected = holders[(decision, camp)]
                camp_rows = [r for r in temp.rows if int(r["row_camp"].reshape(-1)[0]) == camp]
                roots.append(
                    {
                        "root_id": f"{world}/{saved.turn}/{camp}",
                        "root_index": root_index,
                        "world": world,
                        "decision_index": int(decision),
                        "turn": int(saved.turn),
                        "camp": int(camp),
                        "families": sorted(obs.tags, key=FAMILIES.index),
                        "family_mask": family_mask(obs.tags),
                        "entered": sorted(obs.entries, key=FAMILIES.index),
                        "subtypes": dict(obs.subtypes),
                        "selected_for": {
                            k: selected[k] for k in sorted(selected, key=FAMILIES.index)
                        },
                        "founded_turn": int(saved.founded.get(camp, saved.turn)),
                        "rows": len(camp_rows),
                        "terminated": any(
                            bool(r["transition_terminal"].reshape(-1)[0]) for r in camp_rows
                        ),
                        "tail": None
                        if entries is None or camp not in entries
                        else labor_bc.TAIL_METHOD_NAMES[int(entries[camp]["method"])],
                    }
                )
                root_index += 1
            for row, tags in zip(temp.rows, row_tags, strict=True):
                camp = int(row["row_camp"].reshape(-1)[0])
                index = index_of[camp]
                row["row_source"] = _tensor(ROW_SOURCE)
                row["row_start"] = _tensor(decision)
                row["row_start_index"] = _tensor(index)
                row["row_root_turn"] = _tensor(saved.turn)
                row[labor_bc.TRAJECTORY_FIELD] = _tensor(TRAJECTORY_BASE + index)
                row["row_scenario_family"] = _tensor(roots[index]["family_mask"])
                row["row_scenario_tags"] = _tensor(tags)
                rows.append(row)
            groups.append(
                {
                    "decision_index": int(decision),
                    "turn": int(saved.turn),
                    "roots": len(camps),
                    "rows": len(temp.rows),
                    "scenario_decisions": int(temp.decisions),
                    "scenario_native_resolves": int(temp.native_resolves),
                    "scenario_wall_seconds": float(window_seconds),
                    "tail_decisions": int(temp.tail_decisions),
                    "tail_native_resolves": int(temp.tail_native_resolves),
                    "tail_wall_seconds": float(temp.tail_seconds),
                    "counters": dict(temp.counters),
                    "reasons": dict(temp.reasons),
                    "quarantine": list(temp.quarantine),
                    "tail_record": temp.tail_record,
                }
            )
    stats["candidates"] = dict(sorted(stats["candidates"].items()))
    stats["offered"] = dict(sorted(reservoir.offered.items()))
    stats["selected"] = {name: len(held) for name, held in reservoir.selected.items()}
    stats["root_tags"] = {
        name: sum(1 for root in roots if name in root["families"]) for name in FAMILIES
    }
    return {
        "world": world,
        "rows": rows,
        "roots": roots,
        "groups": groups,
        "harvest": harvest,
        "stats": stats,
        "reset_identity": reset_identity,
        "wall_seconds": time.perf_counter() - started,
    }


def _combined_tail_record(
    groups: Sequence[Mapping[str, Any]], tail: Any, scenario_decisions: int
) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    first: Mapping[str, Any] = {}
    for g in groups:
        record = g.get("tail_record") or {}
        first = first or record
        for name, value in record.get("method_counts", {}).items():
            counts[name] += int(value)
    return {
        **tail.state_dict(),
        "prefix_terms": int(scenario_decisions),
        "prefix_semantics": (
            "each root's scenario window (up to scenario_decisions teacher decisions from the "
            "root); the tail is valued at the state after the window, from one teacher "
            "continuation per root group"
        ),
        "schedule_id": first.get("schedule_id"),
        "schedule_seed": first.get("schedule_seed"),
        "required_terms": first.get("required_terms"),
        "stop_indices": first.get("stop_indices"),
        "exceeds_hard_cap": first.get("exceeds_hard_cap"),
        "critic_bootstrap_used": False,
        "schedule_scope": "one frozen stopping schedule per world, shared by its root groups",
        "groups": [
            {
                "decision_index": g["decision_index"],
                "turn": g["turn"],
                "cutoffs": (g.get("tail_record") or {}).get("cutoffs", {}),
            }
            for g in groups
        ],
        "method_counts": dict(sorted(counts.items())),
    }


def write_scenario_shard(job: Mapping[str, Any], **factories: Any) -> dict[str, Any]:
    """Collect one world and publish its shard (or an empty marker without roots)."""
    from .labor_bc import (
        RETURN_TAIL_RANDOMIZED,
        RETURN_TAIL_TRUNCATED_ZERO,
        RandomizedTailSettings,
        _concat,
        world_balanced_weights,
    )
    from .labor_teacher_data import write_shard
    from .provenance import checkpoint_hash

    result = collect_scenario_world(job, **factories)
    world = int(result["world"])
    out = Path(job["out"])
    tail = RandomizedTailSettings.from_state(job.get("return_tail"))
    groups = result["groups"]
    cost = {
        "harvest": result["harvest"],
        "groups": [
            {k: v for k, v in g.items() if k not in ("tail_record", "quarantine", "reasons")}
            for g in groups
        ],
        "wall_seconds": float(result["wall_seconds"]),
    }
    summary = {
        "world_seed": world,
        "roots": len(result["roots"]),
        "rows": len(result["rows"]),
        "stats": result["stats"],
        "cost": cost,
    }
    if not result["rows"]:
        marker = out / f"world-{world}.empty.json"
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"source_run": job["source_run"], "world_seed": world, **summary}, sort_keys=True
            )
        )
        os.replace(tmp, marker)
        summary["path"] = None
        return summary
    tensors = _concat(result["rows"], f"labor scenario world {world}")
    keep = ~tensors["row_quarantined"].reshape(-1)
    tensors["row_weight"] = world_balanced_weights(tensors["row_world"], keep)
    arrays = {key: value.detach().cpu().numpy() for key, value in tensors.items()}
    counters: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    for g in groups:
        for key, value in g["counters"].items():
            counters[key] += int(value)
        for key, value in g["reasons"].items():
            reasons[key] += int(value)
    native = result["harvest"]["native_resolves"] + sum(
        g["scenario_native_resolves"] + g["tail_native_resolves"] for g in groups
    )
    meta = {
        "source_run": job["source_run"],
        "contract": CONTRACT,
        "world_seed": world,
        "turns": int(job["scenario_decisions"]),
        "harvest_decisions": int(job["decisions"]),
        "harvest_start": int(job.get("harvest_start", 0)),
        "reset_options": dict(job["reset_options"]),
        "reset_identity": result["reset_identity"],
        "normalizer_digest": job["normalizer_digest"],
        "personality": dict(job["personality"]),
        "rules": dict(job["rules"]),
        "caps": dict(job["caps"]),
        "root_key": int(job["root_key"]),
        "decisions": int(sum(g["scenario_decisions"] for g in groups)),
        "native_resolves": int(native),
        "wall_seconds": float(result["wall_seconds"]),
        "counters": dict(counters),
        "label_unavailable_reasons": dict(sorted(reasons.items())),
        "quarantine": [q for g in groups for q in g["quarantine"]],
        "surviving_camps": 0,
        "return_tail": RETURN_TAIL_TRUNCATED_ZERO if tail is None else RETURN_TAIL_RANDOMIZED,
        "partition": job["partition"],
        "roots": result["roots"],
        "harvest_stats": result["stats"],
        "cost": cost,
    }
    if tail is not None:
        meta["return_tail_record"] = _combined_tail_record(
            groups, tail, int(job["scenario_decisions"])
        )
    directory = out / f"world-{world}"
    staging = out / f".staging-world-{world}"
    if staging.exists():
        shutil.rmtree(staging)
    record = write_shard(staging, arrays, meta)
    os.replace(staging, directory)
    summary.update(
        {
            "path": directory.name,
            "meta_sha256": checkpoint_hash(directory / "meta.json"),
            "data_digest": record["data_digest"],
            "decisions": record["decisions"],
            "native_resolves": record["native_resolves"],
            "wall_seconds": record["wall_seconds"],
        }
    )
    return summary


def _existing(out: Path, world: int, source_run: str) -> dict[str, Any] | None:
    """A verified shard or empty marker of this source run, else None (partials removed)."""
    from .labor_teacher_data import verify_shard
    from .provenance import ProvenanceError, checkpoint_hash

    directory = out / f"world-{world}"
    marker = out / f"world-{world}.empty.json"
    if marker.is_file():
        record = json.loads(marker.read_text())
        if record.get("source_run") == source_run:
            return {**record, "path": None}
        marker.unlink()
    if not directory.exists():
        return None
    try:
        meta = verify_shard(directory)
    except (ProvenanceError, OSError, ValueError, KeyError):
        meta = None
    if meta is None or meta.get("source_run") != source_run:
        shutil.rmtree(directory)
        return None
    return {
        "world_seed": world,
        "path": directory.name,
        "meta_sha256": checkpoint_hash(directory / "meta.json"),
        "data_digest": meta["data_digest"],
        "rows": meta["rows"],
        "roots": len(meta["roots"]),
        "decisions": meta["decisions"],
        "native_resolves": meta["native_resolves"],
        "wall_seconds": meta["wall_seconds"],
        "stats": meta.get("harvest_stats"),
        "cost": meta.get("cost"),
    }


def excluded_states(
    directories: Sequence[Path], reset_options: Mapping[str, Any]
) -> dict[int, list[tuple[int, int]]]:
    """(turn, camp) per world already recorded by a bank with the same reset options."""
    from .labor_teacher_data import verify_dataset

    out: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for directory in directories:
        directory = Path(directory)
        manifest = verify_dataset(directory)
        for world, entry in manifest["shards"].items():
            shard = directory / entry["path"]
            meta = json.loads((shard / "meta.json").read_text())
            if dict(meta.get("reset_options") or {}) != dict(reset_options):
                continue
            turns = np.load(shard / "row_turn.npy").reshape(-1).tolist()
            camps = np.load(shard / "row_camp.npy").reshape(-1).tolist()
            out[int(world)].update(zip(map(int, turns), map(int, camps), strict=True))
    return {w: sorted(v) for w, v in out.items()}


def publish_roots(
    out: Path, source: Mapping[str, Any], shards: Sequence[Mapping[str, Any]]
) -> dict:
    """The dataset-level root list (search ``--source scenario --scenario-roots``)."""
    from .provenance import digest

    roots = []
    for shard in sorted(shards, key=lambda s: s["world_seed"]):
        if not shard.get("path"):
            continue
        meta = json.loads((out / shard["path"] / "meta.json").read_text())
        roots.extend(meta["roots"])
    options = source["options"]
    body = {
        "schema": ROOTS_SCHEMA,
        "source_run": source["source_run"],
        "partition": options["partition"],
        "reset_options": options["reset_options"],
        "rules": options["rules"],
        "root_driver": "h12",
        "reproduce": (
            "reset each world with reset_options and drive the converted H12 teacher "
            "(labor_bc._teacher_transition) for decision_index decisions"
        ),
        "roots": roots,
    }
    body["roots_digest"] = digest(body)
    tmp = out / (ROOTS_FILE + ".tmp")
    tmp.write_text(json.dumps(body, sort_keys=True, allow_nan=False))
    os.replace(tmp, out / ROOTS_FILE)
    return body


def load_scenario_roots(path: Path) -> dict[str, Any]:
    """A published ``roots.json`` (or its dataset directory), digest-checked."""
    from .provenance import ProvenanceError, digest

    path = Path(path)
    if path.is_dir():
        path = path / ROOTS_FILE
    body = json.loads(path.read_text())
    if body.get("schema") != ROOTS_SCHEMA:
        raise ProvenanceError(f"{path}: not a scenario roots file")
    if body.get("roots_digest") != digest({k: v for k, v in body.items() if k != "roots_digest"}):
        raise ProvenanceError(f"{path}: scenario roots digest mismatch")
    return body


# ---------------------------------------------------------------------------
# Cost report
# ---------------------------------------------------------------------------


def cost_report(
    shards: Sequence[Mapping[str, Any]], *, roots_per_family: int = 40, families: int | None = None
) -> dict[str, Any]:
    """Measured cost per root and the projection for ``roots_per_family`` per family."""
    harvest_s = harvest_r = scen_s = scen_r = tail_s = tail_r = 0.0
    roots = groups = worlds = 0
    family_roots: dict[str, int] = defaultdict(int)
    for shard in shards:
        cost = shard.get("cost") or {}
        harvest = cost.get("harvest") or {}
        worlds += 1
        harvest_s += float(harvest.get("wall_seconds", 0.0))
        harvest_r += float(harvest.get("native_resolves", 0.0))
        for g in cost.get("groups", []):
            groups += 1
            roots += int(g["roots"])
            scen_s += float(g["scenario_wall_seconds"])
            scen_r += float(g["scenario_native_resolves"])
            tail_s += float(g["tail_wall_seconds"])
            tail_r += float(g["tail_native_resolves"])
        for name, count in ((shard.get("stats") or {}).get("selected", {}) or {}).items():
            family_roots[name] += int(count)
    per_root_s = (scen_s + tail_s) / roots if roots else None
    per_root_r = (scen_r + tail_r) / roots if roots else None
    per_world_harvest_s = harvest_s / worlds if worlds else None
    per_resolve = (harvest_s + scen_s + tail_s) / max(1.0, harvest_r + scen_r + tail_r)
    count = len(FAMILIES) if families is None else int(families)
    projected_roots = count * int(roots_per_family)
    return {
        "worlds": worlds,
        "roots": roots,
        "root_groups": groups,
        "roots_per_group": roots / groups if groups else None,
        "harvest_wall_seconds_per_world": per_world_harvest_s,
        "harvest_native_resolves": harvest_r,
        "scenario_plus_tail_wall_seconds_per_root": per_root_s,
        "scenario_plus_tail_native_resolves_per_root": per_root_r,
        "wall_seconds_per_native_resolve": per_resolve,
        "projection": {
            "roots_per_family": int(roots_per_family),
            "families": count,
            "roots": projected_roots,
            "scenario_plus_tail_cpu_hours": None
            if per_root_s is None
            else projected_roots * per_root_s / 3600.0,
            "note": (
                "single-process CPU hours, excluding the harvest drive (one per world, "
                "amortized over that world's roots) and root sharing within a group"
            ),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_minima(text: str | None) -> dict[str, tuple[int, int]]:
    minima = dict(DEFAULT_MINIMA)
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        roots, _, worlds = value.partition("/")
        if name not in PARTITIONS:
            raise ValueError(f"unknown partition {name!r} in --minima")
        minima[name] = (int(roots), int(worlds or 0))
    return minima


def argument_parser() -> argparse.ArgumentParser:
    from .labor_bc import RandomizedTailSettings

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def rule_args(p: argparse.ArgumentParser) -> None:
        d = DetectionRules()
        p.add_argument("--head-window", type=int, default=d.head_window)
        p.add_argument("--daughter-window", type=int, default=d.daughter_window)
        p.add_argument("--small-below", type=float, default=d.small_below)
        p.add_argument("--large-above", type=float, default=d.large_above)
        p.add_argument("--housing-below", type=float, default=d.housing_below)
        p.add_argument("--clothing-below", type=float, default=d.clothing_below)
        p.add_argument("--trade-partners-min", type=float, default=d.trade_partners_min)
        p.add_argument(
            "--any-era",
            action="store_true",
            help="diagnostic only: tag Paleolithic states too (families normally require the "
            "Neolithic era)",
        )

    coverage = commands.add_parser("coverage", help="family coverage of published labor datasets")
    coverage.add_argument("--dataset", type=Path, action="append", default=[])
    coverage.add_argument(
        "--shard",
        type=Path,
        action="append",
        default=[],
        help="diagnostic: an unpublished, individually verified teacher shard directory",
    )
    coverage.add_argument(
        "--minima",
        default=None,
        help="support minima per partition, e.g. 'fitting=32/4,calibration=16/4'",
    )
    rule_args(coverage)
    coverage.add_argument(
        "--normalizer-checkpoint",
        type=Path,
        default=None,
        help="the banks' bounded normalizer: enables the food-trade family (imported food)",
    )
    coverage.add_argument("--out", type=Path, default=None)

    collect = commands.add_parser("collect", help="harvest gap roots and record their scenarios")
    collect.add_argument("--normalizer-checkpoint", type=Path, required=True)
    collect.add_argument("--partition", choices=PARTITIONS, required=True)
    collect.add_argument("--worlds", required=True)
    collect.add_argument(
        "--coverage", type=Path, default=None, help="coverage audit JSON (fills only its gaps)"
    )
    collect.add_argument(
        "--families", default=None, help="comma-separated subset of families to harvest"
    )
    collect.add_argument(
        "--quota-per-world",
        type=int,
        default=None,
        help="fixed root cap per family and world (without --coverage)",
    )
    collect.add_argument("--oversample", type=float, default=1.5)
    collect.add_argument("--max-per-world", type=int, default=8)
    collect.add_argument(
        "--exclude-dataset",
        type=Path,
        action="append",
        default=[],
        help="banks whose recorded states are never roots (same world/reset/turn/camp)",
    )
    collect.add_argument("--decisions", type=int, default=50, help="harvest drive decisions N")
    collect.add_argument("--harvest-start", type=int, default=0)
    collect.add_argument("--scenario-decisions", type=int, default=25, help="K per root")
    collect.add_argument("--root-key", type=int, default=0)
    collect.add_argument("--players", type=int, default=8)
    collect.add_argument("--width", type=int, default=None)
    collect.add_argument("--height", type=int, default=None)
    collect.add_argument("--separation", type=int, default=None)
    collect.add_argument("--qol-weight", type=float, default=0.5)
    collect.add_argument("--population-weight", type=float, default=0.5)
    collect.add_argument("--personality", default="balanced")
    collect.add_argument("--personality-key", type=int, default=0)
    collect.add_argument("--personality-drift", type=float, default=200.0)
    collect.add_argument(
        "--return-tail", choices=("randomized", "truncated-zero"), default="randomized"
    )
    defaults = RandomizedTailSettings()
    collect.add_argument("--tail-draws", type=int, default=defaults.draws)
    collect.add_argument("--tail-rho", type=float, default=defaults.rho)
    collect.add_argument("--tail-cap", type=int, default=defaults.cap)
    collect.add_argument("--tail-schedule-seed-base", type=int, default=defaults.seed_base)
    rule_args(collect)
    collect.add_argument("--held-out-test", action="store_true")
    collect.add_argument("--jobs", type=int, default=1)
    collect.add_argument("--resume", action="store_true")
    collect.add_argument("--allow-dirty", action="store_true")
    collect.add_argument("--out", type=Path, required=True)

    shares = commands.add_parser(
        "weights", help="family row and weight shares before/after --family-weights"
    )
    shares.add_argument("--dataset", type=Path, action="append", default=[])
    shares.add_argument("--shard", type=Path, action="append", default=[])
    shares.add_argument("--family-weights", default="default")
    shares.add_argument("--normalizer-checkpoint", type=Path, default=None)
    shares.add_argument("--out", type=Path, default=None)

    cost = commands.add_parser("cost", help="measured cost per root and projection")
    cost.add_argument("--dataset", type=Path, required=True)
    cost.add_argument("--roots-per-family", type=int, default=40)
    cost.add_argument("--families", type=int, default=None)
    return parser


def _rules(args: argparse.Namespace) -> DetectionRules:
    return DetectionRules(
        head_window=args.head_window,
        daughter_window=args.daughter_window,
        small_below=args.small_below,
        large_above=args.large_above,
        housing_below=args.housing_below,
        clothing_below=args.clothing_below,
        trade_partners_min=args.trade_partners_min,
        require_neolithic=not args.any_era,
    )


def _optional_normalizer(args: argparse.Namespace) -> Any:
    path = getattr(args, "normalizer_checkpoint", None)
    if path is None:
        return None
    from .labor_teacher_data import load_normalizer

    return load_normalizer(Path(path))


def run_coverage(args: argparse.Namespace) -> dict[str, Any]:
    if not args.dataset and not args.shard:
        raise ValueError("coverage needs at least one --dataset or --shard")
    report = coverage_audit(
        args.dataset,
        rules=_rules(args),
        minima=_parse_minima(args.minima),
        shards=args.shard,
        normalizer=_optional_normalizer(args),
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, sort_keys=True, indent=1))
    print(summarize_coverage(report))
    return report


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    from .bridge import DEFAULT_BRIDGE
    from .labor_bc import LaborResetOptions, RandomizedTailSettings
    from .labor_bridge import LaborModel9Bridge
    from .labor_model import LaborPersonality
    from .labor_personality import LaborPersonalityConditioning
    from .labor_teacher_data import (
        load_normalizer,
        normalizer_digest,
        publish_dataset,
        source_record,
        verify_dataset,
    )
    from .protocol import seed_list
    from .provenance import ProvenanceError, checkpoint_hash, code_commit

    worlds = validate_scenario_worlds(
        seed_list(args.worlds), args.partition, held_out_test=args.held_out_test
    )
    if args.decisions < 1 or args.scenario_decisions < 1 or args.jobs < 1:
        raise ValueError("decisions, scenario decisions and jobs must be positive")
    if args.harvest_start < 0 or args.harvest_start >= args.decisions:
        raise ValueError("harvest start must lie inside the harvest drive")
    rules = _rules(args)
    families = None if args.families is None else [f.strip() for f in args.families.split(",")]
    coverage = None if args.coverage is None else json.loads(args.coverage.read_text())
    if coverage is not None and coverage.get("schema") != COVERAGE_SCHEMA:
        raise ValueError("--coverage is not a scenario coverage audit")
    if coverage is not None and coverage.get("rules") != rules.state_dict():
        raise ValueError("--coverage was audited with different detection rules")
    caps = quota_caps(
        coverage,
        args.partition,
        len(worlds),
        families=families,
        per_world=args.quota_per_world,
        oversample=args.oversample,
        max_per_world=args.max_per_world,
    )
    reset = LaborResetOptions(
        players=args.players,
        movement=True,
        width=args.width,
        height=args.height,
        separation=args.separation,
    )
    tail = (
        RandomizedTailSettings(
            args.tail_draws, args.tail_rho, args.tail_cap, args.tail_schedule_seed_base
        )
        if args.return_tail == "randomized"
        else None
    )
    if tail is not None:
        for world in worlds:
            tail.schedule_seed(world)
    if (args.out / "dataset.json").is_file() or (args.out / "summary.json").is_file():
        print(json.dumps({"dataset": str(args.out), "status": "already-published"}))
        summary_path = args.out / "summary.json"
        return json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if args.out.exists() and not args.resume:
        raise ProvenanceError("labor scenario output exists; pass --resume or use a new directory")
    personality = LaborPersonality(
        qol_years=args.qol_weight, population_years=args.population_weight
    ).state_dict()
    conditioning = LaborPersonalityConditioning(
        args.personality, args.personality_key, args.personality_drift
    )
    normalizer = load_normalizer(args.normalizer_checkpoint)
    bridge_path = Path(os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE", str(DEFAULT_BRIDGE)))
    bridge_sha = checkpoint_hash(bridge_path)
    if not re.fullmatch(r"[0-9a-f]{64}", bridge_sha):
        raise ProvenanceError("production bridge binary missing")
    excluded = excluded_states(args.exclude_dataset, reset.state_dict())
    exclude_digests = [verify_dataset(d)["dataset_digest"] for d in args.exclude_dataset]
    options = {
        "contract": CONTRACT,
        "worlds": worlds,
        "turns": int(args.scenario_decisions),
        "decisions": int(args.decisions),
        "harvest_start": int(args.harvest_start),
        "scenario_decisions": int(args.scenario_decisions),
        "reset_options": reset.state_dict(),
        "personality": personality,
        "personality_conditioning": conditioning.state_dict(),
        "normalizer_digest": normalizer_digest(normalizer),
        "normalizer_checkpoint_sha256": checkpoint_hash(args.normalizer_checkpoint),
        "rules": rules.state_dict(),
        "caps": caps,
        "root_key": int(args.root_key),
        "coverage_sha256": None if args.coverage is None else checkpoint_hash(args.coverage),
        "excluded_dataset_digests": exclude_digests,
        "teacher": "h12-converted-native-heuristic-shadow-step",
        "partition": args.partition,
        "return_tail": {"method": "truncated-zero-at-collection-horizon"}
        if tail is None
        else tail.state_dict(),
    }
    if args.held_out_test:
        options["held_out_test"] = True
    with LaborModel9Bridge() as probe:
        source = source_record(
            options, probe.schema, bridge_sha, production=not args.allow_dirty,
            source_kind=SOURCE_KIND,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    source_path = args.out / "source.json"
    if source_path.is_file():
        previous = json.loads(source_path.read_text())
        if previous.get("source_run") != source["source_run"]:
            raise ProvenanceError("labor scenario resume: source record differs")
    else:
        tmp = args.out / "source.json.tmp"
        tmp.write_text(json.dumps(source, sort_keys=True, allow_nan=False))
        os.replace(tmp, source_path)
    done = {}
    for world in worlds:
        entry = _existing(args.out, world, source["source_run"])
        if entry is not None:
            done[world] = entry
    jobs = [
        {
            "normalizer_checkpoint": str(args.normalizer_checkpoint),
            "normalizer_digest": options["normalizer_digest"],
            "reset_options": reset.state_dict(),
            "personality": personality,
            "personality_conditioning": conditioning.state_dict(),
            "world_seed": world,
            "decisions": int(args.decisions),
            "harvest_start": int(args.harvest_start),
            "scenario_decisions": int(args.scenario_decisions),
            "rules": rules.state_dict(),
            "caps": caps,
            "root_key": int(args.root_key),
            "excluded": excluded.get(world, []),
            "return_tail": None if tail is None else tail.state_dict(),
            "source_run": source["source_run"],
            "partition": args.partition,
            "out": str(args.out),
        }
        for world in worlds
        if world not in done
    ]
    print(
        json.dumps(
            {"partition": args.partition, "worlds": len(worlds), "resumed": len(done),
             "jobs": len(jobs), "caps": caps}
        ),
        file=sys.stderr,
        flush=True,
    )
    started = time.perf_counter()
    if args.jobs == 1 or len(jobs) <= 1:
        shards = [write_scenario_shard(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
            shards = list(pool.map(write_scenario_shard, jobs))
    if checkpoint_hash(bridge_path) != bridge_sha or code_commit() != source["learner_commit"]:
        raise ProvenanceError("labor scenario producer changed during collection; not published")
    all_shards = sorted(list(done.values()) + shards, key=lambda s: s["world_seed"])
    published = [s for s in all_shards if s.get("path")]
    manifest = None
    if published:
        manifest = publish_dataset(
            args.out,
            source,
            [
                {k: s[k] for k in ("path", "world_seed", "meta_sha256", "data_digest", "rows")}
                for s in published
            ],
        )
        publish_roots(args.out, source, published)
    summary = {
        "dataset": str(args.out),
        "dataset_digest": None if manifest is None else manifest["dataset_digest"],
        "partition": args.partition,
        "caps": caps,
        "worlds": len(all_shards),
        "worlds_with_roots": len(published),
        "roots": int(sum(int(s.get("roots", 0)) for s in all_shards)),
        "rows": int(sum(int(s.get("rows", 0)) for s in all_shards)),
        "wall_seconds_this_run": time.perf_counter() - started,
        "cost": cost_report(all_shards),
    }
    tmp = args.out / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, sort_keys=True, indent=1))
    os.replace(tmp, args.out / "summary.json")
    print(json.dumps(summary, sort_keys=True))
    return summary


def run_weights(args: argparse.Namespace) -> dict[str, Any]:
    """Shares per family before/after reweighting (datasets via the training loader view)."""
    import torch

    from .labor_bc import world_balanced_weights
    from .labor_teacher_data import load_labor_datasets, verify_shard

    weights = parse_family_weights(args.family_weights)
    if not weights:
        raise ValueError("--family-weights is empty")
    if args.dataset:
        dataset = load_labor_datasets(args.dataset, training=False)
    else:
        parts = []
        for shard in args.shard:
            meta = verify_shard(Path(shard))
            arrays = _load_shard_arrays(Path(shard), meta)
            parts.append({k: torch.as_tensor(v) for k, v in arrays.items()})
        if not parts:
            raise ValueError("weights needs --dataset or --shard")
        keys = set.intersection(*(set(p) for p in parts))
        dataset = {k: torch.cat([p[k] for p in parts]) for k in keys}
        keep = ~dataset["row_quarantined"].reshape(-1).bool()
        dataset["row_weight"] = world_balanced_weights(dataset["row_world"], keep)
    record = reweight_dataset(dataset, weights, normalizer=_optional_normalizer(args))
    report = {
        "weights": record["weights"],
        "status": record["status"],
        "rule": record["rule"],
        "rows": int(dataset["row_world"].shape[0]),
        "worlds": sorted({int(w) for w in dataset["row_world"].reshape(-1).tolist()}),
        "shares": record["shares"],
    }
    lines = [f"{'family':18s} {'rows':>6s} {'row%':>7s} {'before%':>8s} {'after%':>8s}"]
    for name, s in record["shares"].items():
        lines.append(
            f"{name:18s} {s['rows']:6d} {100 * s['row_share']:7.2f} "
            f"{100 * s['weight_share_before']:8.2f} {100 * s['weight_share_after']:8.2f}"
        )
    print("\n".join(lines))
    if args.out is not None:
        args.out.write_text(json.dumps(report, sort_keys=True, indent=1))
    return report


def run_cost(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.dataset)
    shards = []
    for path in sorted(out.glob("world-*")):
        if path.is_dir():
            meta = json.loads((path / "meta.json").read_text())
            shards.append({"cost": meta.get("cost"), "stats": meta.get("harvest_stats")})
        elif path.name.endswith(".empty.json"):
            record = json.loads(path.read_text())
            shards.append({"cost": record.get("cost"), "stats": record.get("stats")})
    report = cost_report(shards, roots_per_family=args.roots_per_family, families=args.families)
    print(json.dumps(report, sort_keys=True, indent=1))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.command == "coverage":
        run_coverage(args)
    elif args.command == "collect":
        run_collect(args)
    elif args.command == "cost":
        run_cost(args)
    elif args.command == "weights":
        run_weights(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
