"""G3 native search teacher for the action-12 specialty-labor actor.

The search improves on the converted heuristic (H12) labels so the pretrained
actor and critics do not inherit heuristic bias. It never enumerates the
action space. For one focal camp at one protected root it builds at most M
complete, coherent, natively validated schema-12 actions (the H12 action, the
actor's deterministic and sampled actions, structured single-block
perturbations of H12 and a few a-priori joint perturbations), then evaluates
them by successive halving over the horizon with paired continuations:

* restore the root, execute the candidate for the focal camp while every other
  camp takes its root H12 action, then continue every camp under H12;
* score the focal camp's discounted four-component return (QoL-years,
  population-years, feasibility cost, settlement guard), credited exactly as
  ``labor_training._transition_rewards``/``_settlement_guard`` and
  ``labor_bc.discounted_returns`` do (tau = 400 years, zero after the focal
  camp terminates);
* store each candidate's return vector once and score it under several
  personalities (U_p = w_q G_q/s_q + w_p G_p/s_p - eta_f C_f - eta_s C_s with
  the frozen labor reference scales). Halving keeps candidates by their best
  rank over personalities, so no personality's best is pruned early;
* per personality, the best-found candidate becomes the label only if it beats
  H12 on fresh, independent continuation streams by the declared margin plus
  ``se_multiplier`` paired standard errors. Otherwise the label stays H12.

Continuation streams. The native simulator and the converted H12 controller
are deterministic, so stream 0 (selection) is the deterministic H12
continuation and every candidate at a root shares it exactly. Independent
re-evaluation needs stochastic streams; streams >= 1 therefore use the declared
``h12-dither`` controller: H12 with a keyed, natively validated transfer of a
few residents between Gatherer and one staffed role for each camp with
probability ``dither_probability``. A stream's draws are keyed by (world,
root, focal, stream, step, camp), so the winner and H12 see common random
numbers. The focal camp's first action is never dithered.

T2 plans (``--plan-candidates``, diagnostic): per root, keyed multi-decision
courses held by the focal camp for ``--plan-steps`` decisions, each with a paired
first-step control (same first step, then H12). They count toward M, never enter
halving or the labels, and are exported as plan rows of the candidate payload.

Successor rows (``--successor-rows``, payload export v2): every candidate
branch also records the focal camp's critic-input rows (actor-visible
observation row plus the raw privileged row, the root-row format) at the state
after decision 1 and at the last declared horizon, per stream, with the focal
camp's status at every boundary, so ``V(s'_a)`` is computable per candidate
(``labor_sa_advantage``). Reading the successor state never changes a search
decision or a return; without the flag the export is the v1 payload unchanged.

``--resume`` continues a preempted output: verified world shards are kept,
partial worlds are redone into fresh shards, a changed source record is refused,
and dataset.json is published only once every world is complete.

Selected labels are "best found within the search specification", never
"optimal". Forced-first-action finite-window returns live in the candidate
table; they are not policy V returns.

Example::

    python -m anthropocene_rl.autonomous_v7.labor_search \\
        --normalizer-checkpoint ckpt.pt --actor-checkpoint ckpt.pt \\
        --worlds 90100-90159 --roots-per-world 8 --jobs 30 --out datasets/labor-search-r0
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import orjson
import torch

from . import contracts
from . import labor_search_payload as payload_module
from .labor_bc import (
    LABOR_LABEL_CONTRACT,
    HeuristicLaborLabel,
    LaborResetOptions,
    _concat,
    _idle_retry,
    _label_tensors,
    _movement_legal,
    _one_row,
    _retained_action,
    convert_heuristic_labor_label,
    world_balanced_weights,
)
from .labor_bridge import (
    ACTIVITIES,
    SPECIALTIES,
    LaborCatalog,
    LaborPolicySample,
    apportion_largest_remainder,
)
from .labor_model import (
    LABOR_COST_REFERENCE_SCALES,
    LABOR_OBJECTIVE_REFERENCE_SCALES,
    LaborPersonality,
)
from .labor_privileged import privileged_row_tensors
from .labor_worlds import partition_for

SEARCH_CONTRACT = "model9-labor-search-teacher-v1"
SOURCE_KIND = "native_search_labor_teacher"
SCHEMA = 1
COMPONENTS = ("qol_years", "population_years", "feasibility", "settlement_guard")
FAMILIES = (
    "h12",
    "actor-deterministic",
    "actor-sample",
    "transfer",
    "activity",
    "movement",
    "fission",
    "trade",
    "joint",
    # Evaluator controls (intervention B gates), only on the keyed control roots.
    "control-idle",
    "control-delayed",
    "control-twin",
    # T2 multi-decision plans (diagnostic): the plan and its paired first-step control.
    "plan",
    "plan-control",
)
PLAN_FAMILY = "plan"
PLAN_CONTROL_FAMILY = "plan-control"
PLAN_ROLES = ("plan", "first-step-control")
# Plan templates: a coherent course held for P decisions by the focal camp.
PLAN_TEMPLATES = ("farm-expand", "shelter-and-clothe", "start-herding")
PLAN_TAG = 0x5350  # "SP": plan template draw
DISCOURAGE_FISSION = 1  # fission policy codes: 0 Neutral, 1 Discourage, 2 Encourage
CONTROL_FAMILIES = {
    "control-idle": "all-idle",
    "control-delayed": "delayed-benefit",
    "control-twin": None,  # bookkeeping twin of H12 (bookkeeping_twin = 0)
}
FAMILY_CODES = {name: index for index, name in enumerate(FAMILIES)}
PERTURBATION_FAMILIES = ("transfer", "activity", "movement", "fission", "trade", "joint")
ROW_SOURCE_CODES = {"heuristic": 0, "search": 2}  # 1 is learner-state DAgger
# "scenario": H12-driven trajectories searched at curated scenario roots
# (labor_scenarios roots.json); their payload roots keep source "h12" (whose state)
# and carry scenario_family / scenario_root_id.
SOURCES = ("h12", "actor", "scenario")
FITTING_WORLDS = range(90100, 90160)

# Keyed-stream tags (numpy SeedSequence entropy words).
ROOT_TAG = 0x5352  # "SR": root turn draw
FOCAL_TAG = 0x5346  # "SF": focal order
CAND_TAG = 0x5343  # "SC": candidate family shuffles
ACTOR_TAG = 0x5341  # "SA": actor samples
DITHER_TAG = 0x5344  # "SD": continuation dither
CONTROL_TAG = 0x5343 + 0x100  # control-root draw
PLAN_ROOT_TAG = 0x5350 + 0x100  # plan-root draw (--plan-fraction)
REEVAL_SCOPES = ("final-survivors", "winners")

TRADE_SET = [name for name, _, _ in contracts.SETS].index("trade")
TRADE_HELD = 0
TRADE_ACTIVITY = (6, 7, 8)  # imported, exported, realised utility

# Key activity for a role that becomes staffed by a transfer.
DEFAULT_ACTIVITY = {
    "Gatherer": "food.gather",
    "Farmer": "field.operation",
    "Herder": "herd.management",
    "Fisher": "fishing.harvest",
    "Craftsperson": "goods.housing",
    "Builder": "goods.housing",
    "Weaver": "goods.cloth",
    "Potter": "goods.pots",
    "Toolmaker": "goods.tools",
    "Carpenter": "goods.housing-material",
    "Woodcutter": "stock.wood",
    "Stonecutter": "stock.stone",
    "ClayMiner": "stock.clay",
}
TRANSFER_PAIRS = (
    ("Gatherer", "Farmer"),
    ("Farmer", "Gatherer"),
    ("Gatherer", "Craftsperson"),
    ("Craftsperson", "Gatherer"),
    ("Gatherer", "Builder"),
    ("Gatherer", "Weaver"),
)
GATHERER = SPECIALTIES.index("Gatherer")
_REFERENCE_SCALES = np.asarray(
    [
        LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"],
        LABOR_OBJECTIVE_REFERENCE_SCALES["population_years"],
        LABOR_COST_REFERENCE_SCALES["feasibility"],
        LABOR_COST_REFERENCE_SCALES["settlement_guard"],
    ],
    dtype=np.float64,
)
IDENTITY_KEYS = ("nativems", "trajectoryid", "decisionid")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _parse_keep(token: str) -> tuple[str, float]:
    token = str(token).strip().lower()
    if token == "all":
        return ("all", 1.0)
    if "/" in token:
        a, b = token.split("/", 1)
        value = float(a) / float(b)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"keep fraction {token} must be in (0, 1]")
        return ("fraction", value)
    count = int(token)
    if count < 1:
        raise ValueError("keep counts must be positive")
    return ("count", float(count))


@dataclass(frozen=True)
class SearchConfig:
    """Declared search specification (recorded in every manifest)."""

    candidates: int = 16
    actor_samples: int = 4
    horizons: tuple[int, ...] = (1, 2, 3, 5)
    keep: tuple[str, ...] = ("all", "1/2", "1/4", "2")
    personalities: tuple[tuple[float, float], ...] = ((0.5, 0.5), (0.9, 0.1), (0.1, 0.9))
    root_personality: bool = True
    margin: float = 0.005
    se_multiplier: float = 2.0
    reeval_streams: int = 3
    dither_probability: float = 0.5
    dither_fraction: float = 0.05
    max_resolves: int = 400
    focal_per_root: int = 1
    movement_candidates: int = 3
    validation_cap: int = 64
    feasibility_multiplier: float = 1.0
    guard_multiplier: float = 1.0
    discount_years: float = contracts.HORIZON_YEARS
    # Re-evaluation scope: every candidate surviving to the final horizon (payload v1,
    # so every same-root pair has paired streams), or the legacy H12-plus-contenders.
    reeval_scope: str = "final-survivors"
    # A-ranker (intervention B; off unless a ranker checkpoint is supplied).
    ranker: bool = False
    ranker_top: int = 4
    ranker_full_fraction: float = 0.2
    # Keyed share of roots that add the evaluator controls (all-idle, delayed-benefit
    # Farmer adoption, bookkeeping twin of H12), carried through halving like H12.
    control_fraction: float = 0.0
    # T2 plan candidates (diagnostic): plans per root, each a course held for
    # ``plan_steps`` decisions plus its paired first-step control. Both rows count
    # toward ``candidates`` (M); they never enter halving or the label decision.
    plan_candidates: int = 0
    plan_steps: int = 3
    # Keyed share of roots that carry the T2 plans (1.0: every root, the historical
    # behaviour). Single-step candidates stay ``single_budget`` on every root.
    plan_fraction: float = 1.0
    # Export each candidate's successor critic-input rows (payload export v2).
    successor_rows: bool = False

    def __post_init__(self) -> None:
        if self.candidates < 2:
            raise ValueError("search needs at least two candidates (H12 and one alternative)")
        if not self.horizons or any(int(h) < 1 for h in self.horizons):
            raise ValueError("horizons must be positive")
        if list(self.horizons) != sorted(set(int(h) for h in self.horizons)):
            raise ValueError("horizons must be strictly increasing")
        if len(self.keep) != len(self.horizons):
            raise ValueError("keep needs one entry per horizon round")
        for token in self.keep:
            _parse_keep(token)
        if self.reeval_streams < 2:
            raise ValueError(
                "independent re-evaluation needs at least two streams for a standard error"
            )
        for pair in self.personalities:
            LaborPersonality(*pair)
        if not 0.0 <= self.dither_probability <= 1.0 or not 0.0 < self.dither_fraction <= 1.0:
            raise ValueError("dither probability/fraction out of range")
        if self.reeval_scope not in REEVAL_SCOPES:
            raise ValueError(f"reeval scope must be one of {REEVAL_SCOPES}")
        if not 0.0 <= self.control_fraction <= 1.0:
            raise ValueError("control fraction must be in [0, 1]")
        if self.ranker_top < 1 or not 0.0 <= self.ranker_full_fraction <= 1.0:
            raise ValueError("ranker top must be positive and its full fraction in [0, 1]")
        if not 0.0 <= self.plan_fraction <= 1.0:
            raise ValueError("plan fraction must be in [0, 1]")
        if self.plan_candidates < 0 or self.plan_steps < 1:
            raise ValueError("plan candidates must be non-negative and plan steps positive")
        if self.plan_candidates and self.candidates - 2 * self.plan_candidates < 2:
            raise ValueError(
                "plans count toward the candidate budget: M - 2 x plans must leave H12 and one "
                "single-step alternative"
            )

    @property
    def successor_decisions(self) -> tuple[int, ...]:
        """Decision boundaries whose successor rows are exported: 1 and the last horizon."""
        return tuple(sorted({1, int(self.horizons[-1])}))

    @property
    def single_budget(self) -> int:
        """Single-step candidates allowed once the plan and control rows are reserved."""
        return self.candidates - 2 * self.plan_candidates

    def keep_counts(self, n: int) -> list[int]:
        """Survivors per round, non-increasing and at least one."""
        out: list[int] = []
        previous = n
        for kind, value in (_parse_keep(t) for t in self.keep):
            if kind == "all":
                count = previous
            elif kind == "fraction":
                count = math.ceil(n * value)
            else:
                count = int(value)
            count = max(1, min(previous, count))
            out.append(count)
            previous = count
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "actor_samples": self.actor_samples,
            "horizons": list(self.horizons),
            "keep": list(self.keep),
            "personalities": [list(p) for p in self.personalities],
            "root_personality": self.root_personality,
            "margin": self.margin,
            "se_multiplier": self.se_multiplier,
            "reeval_streams": self.reeval_streams,
            "dither_probability": self.dither_probability,
            "dither_fraction": self.dither_fraction,
            "max_resolves": self.max_resolves,
            "focal_per_root": self.focal_per_root,
            "movement_candidates": self.movement_candidates,
            "validation_cap": self.validation_cap,
            "feasibility_multiplier": self.feasibility_multiplier,
            "guard_multiplier": self.guard_multiplier,
            "discount_years": self.discount_years,
            "reeval_scope": self.reeval_scope,
            "ranker": self.ranker,
            "ranker_top": self.ranker_top,
            "ranker_full_fraction": self.ranker_full_fraction,
            "control_fraction": self.control_fraction,
            "plan_candidates": self.plan_candidates,
            "plan_steps": self.plan_steps,
            # Recorded only when it narrows the plan roots, so every historical source
            # record (and its --resume identity) is unchanged.
            **({} if self.plan_fraction == 1.0 else {"plan_fraction": self.plan_fraction}),
            # Likewise recorded only when on (v1 source records and payloads unchanged).
            **({"successor_rows": True} if self.successor_rows else {}),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SearchConfig:
        data = dict(value)
        data["horizons"] = tuple(int(h) for h in data["horizons"])
        data["keep"] = tuple(str(k) for k in data["keep"])
        data["personalities"] = tuple(tuple(float(x) for x in p) for p in data["personalities"])
        return cls(**data)


# ---------------------------------------------------------------------------
# Personality scoring
# ---------------------------------------------------------------------------


def utility(returns: np.ndarray, personality: Sequence[float], config: SearchConfig) -> np.ndarray:
    """U_p per row of ``returns`` (columns in COMPONENTS order)."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1, len(COMPONENTS))
    w_q, w_p = float(personality[0]), float(personality[1])
    return (
        w_q * r[:, 0] / LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"]
        + w_p * r[:, 1] / LABOR_OBJECTIVE_REFERENCE_SCALES["population_years"]
        - config.feasibility_multiplier * r[:, 2] / LABOR_COST_REFERENCE_SCALES["feasibility"]
        - config.guard_multiplier * r[:, 3] / LABOR_COST_REFERENCE_SCALES["settlement_guard"]
    )


def utility_matrix(
    returns: np.ndarray, personalities: Sequence[Sequence[float]], config: SearchConfig
) -> np.ndarray:
    """[candidates, personalities] utilities from one stored return matrix."""
    return np.stack([utility(returns, p, config) for p in personalities], axis=1)


def halving_order(
    u: np.ndarray, ids: Sequence[int], tiebreak: Mapping[int, int] | None = None
) -> list[int]:
    """Candidate ids ordered by their best rank over personalities.

    A candidate's rank under a personality is the number of candidates with a
    strictly higher utility (equal utilities share a rank). The key is
    (best rank, sum of ranks, tiebreak), so the top-k keeps every personality's
    best before anyone's second best. ``tiebreak`` is a keyed order; without
    it ties fall back to candidate id. Exact ties are common at short horizons
    (many single-block changes do not move the first transition), so a keyed
    tie-break avoids systematically favouring early candidate families.
    """
    ids = list(ids)
    if not ids:
        return []
    order = tiebreak if tiebreak is not None else {c: c for c in ids}
    sub = np.asarray(u, dtype=np.float64)[ids]
    ranks = (sub[None, :, :] > sub[:, None, :]).sum(axis=1)
    key = {
        ids[i]: (int(ranks[i].min()), int(ranks[i].sum()), int(order[ids[i]]), ids[i])
        for i in range(len(ids))
    }
    return sorted(ids, key=lambda c: key[c])


def select_survivors(
    u: np.ndarray,
    ids: Sequence[int],
    keep: int,
    protected: int,
    tiebreak: Mapping[int, int] | None = None,
) -> list[int]:
    """The top ``keep`` by best rank over personalities, plus the protected reference."""
    ranked = [c for c in halving_order(u, ids, tiebreak) if c != protected]
    survivors = ranked[:keep]
    if protected in ids and protected not in survivors:
        survivors = [*survivors, protected]
    return sorted(survivors)


def best_for_personality(u_column: np.ndarray, ids: Sequence[int], reference: int) -> int:
    """Argmax utility; ties prefer the reference, then the lower id."""
    return max(ids, key=lambda c: (float(u_column[c]), c == reference, -c))


def paired_acceptance(
    winner_u: Sequence[float], reference_u: Sequence[float], margin: float, se_multiplier: float
) -> dict[str, float | bool]:
    """Winner beats the reference on paired streams by margin*|U_ref| + k*SE."""
    w = np.asarray(winner_u, dtype=np.float64)
    r = np.asarray(reference_u, dtype=np.float64)
    if w.shape != r.shape or w.size < 2:
        raise ValueError("paired acceptance needs >= 2 matched streams")
    d = w - r
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size))
    threshold = margin * abs(float(r.mean())) + se_multiplier * se
    return {
        "mean_gain": mean,
        "se": se,
        "threshold": threshold,
        "reference_mean": float(r.mean()),
        "accepted": bool(mean > threshold),
    }


# ---------------------------------------------------------------------------
# Candidate construction (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One complete schema-12 action for the focal camp."""

    payload: dict[str, Any]
    family: str
    description: str
    chosen_heads: frozenset[str]

    def digest(self) -> str:
        return payload_digest(self.payload)


def payload_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(orjson.dumps(dict(payload), option=orjson.OPT_SORT_KEYS)).hexdigest()


def labor_state(payload: Mapping[str, Any], catalog: LaborCatalog) -> tuple[np.ndarray, np.ndarray]:
    """(integer residents per specialty, activity matrix) the native apportionment yields."""
    sample = LaborPolicySample.from_payload(
        {
            "laborSpecialtyAllocation": payload["laborSpecialtyAllocation"],
            "laborActivityAllocation": payload["laborActivityAllocation"],
        },
        catalog.masks,
    )
    counts = np.asarray(
        apportion_largest_remainder(sample.specialty_allocation, int(catalog.population)),
        dtype=np.int64,
    )
    return counts, np.asarray(sample.activity_allocation, dtype=np.float64)


def _activity_index(catalog: LaborCatalog, role: int, activity_id: str) -> int | None:
    index = ACTIVITIES.index(activity_id) if activity_id in ACTIVITIES else None
    if index is None or not catalog.masks.activity[role][index]:
        return None
    return index


def _default_activity(catalog: LaborCatalog, role: int) -> int | None:
    name = SPECIALTIES[role + 1]
    preferred = DEFAULT_ACTIVITY.get(name)
    if preferred is not None:
        index = _activity_index(catalog, role, preferred)
        if index is not None:
            return index
    legal = [i for i, ok in enumerate(catalog.masks.activity[role]) if ok and i > 0]
    return legal[0] if legal else None


def compose_payload(
    base: Mapping[str, Any],
    catalog: LaborCatalog,
    *,
    counts: np.ndarray | None = None,
    activity: np.ndarray | None = None,
    movement: int | None = None,
    fission: int | None = None,
    trade_gate: Sequence[int] | None = None,
) -> dict[str, Any]:
    """A complete schema-12 payload: ``base`` with replaced blocks, canonically serialized."""
    base_counts, base_activity = labor_state(base, catalog)
    counts = base_counts if counts is None else np.asarray(counts, dtype=np.int64)
    activity = base_activity if activity is None else np.asarray(activity, dtype=np.float64)
    population = int(catalog.population)
    if population <= 0 or int(counts.sum()) != population or np.any(counts < 0):
        raise ValueError("composed staffing must conserve residents")
    spec = tuple(float(c) / population for c in counts)
    sample = LaborPolicySample(spec, tuple(tuple(float(x) for x in row) for row in activity))
    payload = {
        "schemaVersion": 12,
        "movement": int(base["movement"]) if movement is None else int(movement),
        "fission": int(base["fission"]) if fission is None else int(fission),
        "tradeValue": [float(x) for x in base["tradeValue"]],
        "tradeGate": [int(x) for x in (base["tradeGate"] if trade_gate is None else trade_gate)],
        "tradeShare": float(base["tradeShare"]),
        **sample.payload(),
    }
    # Round trip through the strict parser: masks, simplexes, exact residents.
    check_counts, _ = labor_state(payload, catalog)
    if not np.array_equal(check_counts, counts):
        raise ValueError("composed staffing does not apportion to the declared residents")
    return payload


def conserves_labor(payload: Mapping[str, Any], catalog: LaborCatalog) -> bool:
    """Residents sum to P and every role's worker-time simplex sums to one."""
    counts, activity = labor_state(payload, catalog)
    rows_ok = bool(np.allclose(activity.sum(axis=1), 1.0, atol=1e-8))
    return int(counts.sum()) == int(catalog.population) and rows_ok


def transfer(
    counts: np.ndarray,
    activity: np.ndarray,
    catalog: LaborCatalog,
    source: int,
    target: int,
    amount: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Move ``amount`` residents from specialty ``source`` to ``target`` (specialty rows)."""
    mask = catalog.masks.specialty
    if source == target or source < 1 or target < 1 or amount < 1:
        return None
    if not mask[source] or not mask[target] or int(counts[source]) < amount:
        return None
    new_counts = counts.copy()
    new_activity = activity.copy()
    new_counts[source] -= amount
    new_counts[target] += amount
    target_role = target - 1
    if int(counts[target]) == 0 or new_activity[target_role, 0] >= 1.0 - 1e-12:
        index = _default_activity(catalog, target_role)
        if index is None:
            return None
        new_activity[target_role] = 0.0
        new_activity[target_role, index] = 1.0
    if int(new_counts[source]) == 0:
        new_activity[source - 1] = 0.0
        new_activity[source - 1, 0] = 1.0
    return new_counts, new_activity


def control_candidates(h12: Mapping[str, Any], catalog: LaborCatalog) -> list[Candidate]:
    """Evaluator controls built from H12 (each natively validated later like any candidate).

    * ``control-idle``: every resident idle (known harmful; must rank last);
    * ``control-delayed``: Farmer adoption, 10% of residents Gatherer -> Farmer, only
      when H12 staffs no Farmer and Farmer is legal (a suitable site);
    * ``control-twin``: H12 with the activity simplex of one *unstaffed* role moved to
      another legal activity -- no resident works it, so the physics is H12's and only
      bookkeeping differs; failing that (unstaffed roles admit only idle), H12 with
      specialty shares shifted by a quarter resident, which apportion to the same
      integer residents.
    """
    counts, activity = labor_state(h12, catalog)
    population = int(catalog.population)
    out: list[Candidate] = []
    if population <= 0:
        return out
    idle_counts = np.zeros_like(counts)
    idle_counts[0] = population
    idle_activity = np.zeros_like(activity)
    idle_activity[:, 0] = 1.0
    with contextlib.suppress(ValueError):
        out.append(
            Candidate(
                compose_payload(h12, catalog, counts=idle_counts, activity=idle_activity),
                "control-idle",
                "all residents idle",
                frozenset({"labor"}),
            )
        )
    farmer = SPECIALTIES.index("Farmer")
    if int(counts[farmer]) == 0 and catalog.masks.specialty[farmer]:
        amount = min(int(counts[GATHERER]), max(1, round(0.1 * population)))
        moved = transfer(counts, activity, catalog, GATHERER, farmer, amount)
        if moved is not None:
            with contextlib.suppress(ValueError):
                out.append(
                    Candidate(
                        compose_payload(h12, catalog, counts=moved[0], activity=moved[1]),
                        "control-delayed",
                        f"Farmer adoption: {amount} Gatherer->Farmer",
                        frozenset({"labor"}),
                    )
                )
    for role in range(activity.shape[0]):
        if int(counts[role + 1]) > 0:
            continue
        legal = [i for i, ok in enumerate(catalog.masks.activity[role]) if ok]
        current = int(np.argmax(activity[role]))
        others = [i for i in legal if i != current]
        if not others:
            continue
        twin_activity = activity.copy()
        twin_activity[role] = 0.0
        twin_activity[role, others[0]] = 1.0
        try:
            payload = compose_payload(h12, catalog, counts=counts, activity=twin_activity)
        except ValueError:
            continue
        if payload_digest(payload) != payload_digest(h12):
            out.append(
                Candidate(
                    payload,
                    "control-twin",
                    f"bookkeeping twin of H12 (unstaffed {SPECIALTIES[role + 1]} activity)",
                    frozenset(),
                )
            )
            break
    if not any(c.family == "control-twin" for c in out):
        # Fallback twin: shift a quarter resident of specialty share between two
        # specialties; the native largest-remainder apportionment yields H12's residents.
        staffed = [i for i in range(len(counts)) if counts[i] > 0 and catalog.masks.specialty[i]]
        if len(staffed) >= 2:
            a, b = staffed[0], staffed[-1]
            spec = counts.astype(np.float64) / population
            delta = 0.25 / population
            spec[a] += delta
            spec[b] -= delta
            try:
                sample = LaborPolicySample(
                    tuple(float(x) for x in spec),
                    tuple(tuple(float(x) for x in r) for r in activity),
                )
                twin = dict(h12, **sample.payload())
                same = np.array_equal(labor_state(twin, catalog)[0], counts)
            except ValueError:
                same = False
            if same and payload_digest(twin) != payload_digest(h12):
                out.append(
                    Candidate(
                        twin,
                        "control-twin",
                        "bookkeeping twin of H12 (shares apportioning to H12's residents)",
                        frozenset(),
                    )
                )
    return out


def is_control_root(key: Sequence[int], fraction: float) -> bool:
    return bool(np.random.default_rng([CONTROL_TAG, *[int(k) for k in key]]).random() < fraction)


def is_plan_root(key: Sequence[int], fraction: float) -> bool:
    """Keyed T2 plan-root draw (independent of the control draw); 1.0 is every root."""
    if fraction >= 1.0:
        return True
    return bool(np.random.default_rng([PLAN_ROOT_TAG, *[int(k) for k in key]]).random() < fraction)


def transfer_amounts(population: int) -> list[int]:
    return sorted({1, 2, max(1, round(0.1 * population))})


def trade_opportunities(observation: Mapping[str, Any] | None) -> list[int]:
    """Groups whose trade row is present with stock or recent trade activity."""
    if not isinstance(observation, Mapping):
        return []
    sets, present = observation.get("sets"), observation.get("rowPresent")
    if not sets or not present or len(sets) <= TRADE_SET or len(present) <= TRADE_SET:
        return []
    _, rows, width = contracts.SETS[TRADE_SET]
    table = np.asarray(sets[TRADE_SET], dtype=np.float64).reshape(rows, width)
    live = np.asarray(present[TRADE_SET], dtype=bool)[:rows]
    out = []
    for group in range(min(rows, len(contracts.TRADE_GROUPS))):
        if not live[group]:
            continue
        activity = np.abs(table[group, list(TRADE_ACTIVITY)]).sum()
        if table[group, TRADE_HELD] > 0 or activity > 0:
            out.append(group)
    return out


def legal_region_moves(observation: Mapping[str, Any] | None) -> list[int]:
    """Region movement codes (REGION_BASE + r, r >= 1) legal in the pre-action mask."""
    if not isinstance(observation, Mapping):
        return []
    codes = [contracts.REGION_BASE + r for r in range(1, contracts.REGION_ROWS)]
    return [code for code in codes if _movement_legal(code, observation)]


def _keyed_shuffle(items: list[Any], key: Sequence[int]) -> list[Any]:
    rng = np.random.default_rng(list(key))
    return [items[i] for i in rng.permutation(len(items))]


def structured_candidates(
    base: Mapping[str, Any],
    catalog: LaborCatalog,
    observation: Mapping[str, Any] | None,
    *,
    key: Sequence[int],
    movement_candidates: int = 3,
) -> dict[str, list[Candidate]]:
    """Single-block H12 perturbations by family, plus a-priori joint ones.

    Each family list is keyed-shuffled. Every labor perturbation conserves
    residents and worker-time; nonlabor perturbations keep labor exactly.
    Joint candidates combine structurally chosen blocks fixed before any
    evaluation, never evaluated bests from different branches.
    """
    counts, activity = labor_state(base, catalog)
    population = int(catalog.population)
    out: dict[str, list[Candidate]] = {name: [] for name in PERTURBATION_FAMILIES}
    row = {name: i for i, name in enumerate(SPECIALTIES)}

    def add(family: str, description: str, heads: set[str], **blocks: Any) -> Candidate | None:
        try:
            payload = compose_payload(base, catalog, **blocks)
        except ValueError:
            return None
        cand = Candidate(payload, family, description, frozenset(heads))
        out[family].append(cand)
        return cand

    # a. Specialty transfers.
    pairs = list(TRANSFER_PAIRS)
    staffed = sorted(
        (i for i in range(1, len(SPECIALTIES)) if counts[i] > 0), key=lambda i: (-counts[i], i)
    )
    if len(staffed) >= 2:
        a, b = SPECIALTIES[staffed[0]], SPECIALTIES[staffed[1]]
        for pair in ((a, b), (b, a)):
            if pair not in pairs:
                pairs.append(pair)
    transfers: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for src, dst in pairs:
        for amount in transfer_amounts(population):
            moved = transfer(counts, activity, catalog, row[src], row[dst], amount)
            if moved is None:
                continue
            transfers[(src, dst, amount)] = moved
            add(
                "transfer",
                f"transfer {amount} {src}->{dst}",
                {"labor"},
                counts=moved[0],
                activity=moved[1],
            )

    # b. Within-role activity shifts for staffed roles.
    for spec in range(1, len(SPECIALTIES)):
        if counts[spec] <= 0:
            continue
        role = spec - 1
        legal = [i for i, ok in enumerate(catalog.masks.activity[role]) if ok and i > 0]
        current = activity[role]
        if not legal:
            continue
        main = max(range(len(current)), key=lambda i: (current[i], -i))
        if current[0] > 0.05 and main != 0:
            shifted = activity.copy()
            shifted[role, main] += shifted[role, 0]
            shifted[role, 0] = 0.0
            add(
                "activity",
                f"{SPECIALTIES[spec]} idle->{ACTIVITIES[main]}",
                {"labor"},
                activity=shifted,
            )
        for target in legal:
            if target == main:
                continue
            delta = min(0.25, float(current[main]))
            if delta <= 1e-9:
                continue
            shifted = activity.copy()
            shifted[role, main] -= delta
            shifted[role, target] += delta
            add(
                "activity",
                f"{SPECIALTIES[spec]} {ACTIVITIES[main]}->{ACTIVITIES[target]} {delta:.2f}",
                {"labor"},
                activity=shifted,
            )

    # c. Movement: stay versus the top legal destinations (keyed choice).
    base_move = int(base["movement"])
    moves: list[int] = []
    if base_move != -1:
        moves.append(-1)
    region = [code for code in legal_region_moves(observation) if code != base_move]
    moves += _keyed_shuffle(region, [*key, 1])[: max(0, movement_candidates)]
    for code in moves:
        add("movement", f"movement {code}", {"movement"}, movement=code)

    # d. Fission: every other legal policy.
    for code in range(contracts.FISSION_POLICIES):
        if code != int(base["fission"]):
            add("fission", f"fission {code}", {"fission"}, fission=code)

    # e. Trade gates on/off for groups with an executable opportunity.
    gates = [int(g) for g in base["tradeGate"]]
    opportunities = trade_opportunities(observation)
    for group in opportunities:
        flipped = list(gates)
        flipped[group] = 0 if gates[group] > 0 else contracts.TRADE_MAKE
        add(
            "trade",
            f"trade {contracts.TRADE_GROUPS[group]} gate {gates[group]}->{flipped[group]}",
            {"trade_gate"},
            trade_gate=flipped,
        )

    # f. A-priori joint perturbations.
    tenth = max(1, round(0.1 * population))
    farm = transfers.get(("Gatherer", "Farmer", tenth))
    craft = transfers.get(("Gatherer", "Craftsperson", tenth))
    if moves and farm is not None:
        add(
            "joint",
            f"movement {moves[0]} + transfer {tenth} Gatherer->Farmer",
            {"movement", "labor"},
            movement=moves[0],
            counts=farm[0],
            activity=farm[1],
        )
    if opportunities and craft is not None:
        flipped = list(gates)
        group = opportunities[0]
        flipped[group] = 0 if gates[group] > 0 else contracts.TRADE_MAKE
        add(
            "joint",
            f"transfer {tenth} Gatherer->Craftsperson + trade {contracts.TRADE_GROUPS[group]} gate",
            {"labor", "trade_gate"},
            counts=craft[0],
            activity=craft[1],
            trade_gate=flipped,
        )
    if len(staffed) >= 2 and staffed[0] == GATHERER:
        partner = SPECIALTIES[staffed[1]]
        reserve = transfers.get(("Gatherer", partner, tenth))
        alt_fission = 2 if int(base["fission"]) != 2 else 1
        if reserve is not None:
            add(
                "joint",
                f"fission {alt_fission} + transfer {tenth} Gatherer->{partner}",
                {"fission", "labor"},
                fission=alt_fission,
                counts=reserve[0],
                activity=reserve[1],
            )

    for family in PERTURBATION_FAMILIES:
        out[family] = _keyed_shuffle(out[family], [*key, 2, FAMILY_CODES[family]])
    return out


def round_robin(families: Mapping[str, list[Candidate]]) -> list[Candidate]:
    """Interleave families in declared order so a budget covers every family."""
    queues = {name: list(families.get(name, [])) for name in PERTURBATION_FAMILIES}
    out: list[Candidate] = []
    while any(queues.values()):
        for name in PERTURBATION_FAMILIES:
            if queues[name]:
                out.append(queues[name].pop(0))
    return out


# ---------------------------------------------------------------------------
# T2 plan candidates (pure construction)
# ---------------------------------------------------------------------------
#
# A plan is a coherent course held for P decisions by the focal camp. Step 0 is
# built from the root H12 action; step k >= 1 is rebuilt from H12's action at the
# state the plan reached (so it stays complete and conserving there), then
# natively validated; an unconstructible or rejected step falls back to H12 and
# is recorded. After P steps H12 continues. Templates:
#
# * ``farm-expand``: +10% residents Gatherer -> Farmer; later steps hold at least
#   the adopted Farmers and shift half of Farmer worker-time to field.expansion;
# * ``shelter-and-clothe``: the death rate depends on clothing and shelter, and
#   settlements that max out food stall without them; +10% residents Gatherer ->
#   Craftsperson/Weaver split across housing, housing material and cloth (Weaver
#   on cloth when that head is legal, else Craftsperson cloth); later steps hold
#   those workers on the same three goods;
# * ``start-herding``: +10% residents Gatherer -> Herder; later steps hold at least
#   the adopted Herders and shift half of Herder worker-time to herd.expansion.
#
# The paired first-step control executes the same step 0, then H12 from decision 2.

PLAN_SPECIALTY = {
    "farm-expand": "Farmer",
    "shelter-and-clothe": "Craftsperson",
    "start-herding": "Herder",
}
PLAN_EXPANSION = {"farm-expand": "field.expansion", "start-herding": "herd.expansion"}


@dataclass(frozen=True)
class PlanSpec:
    """One plan at one root: its template, length and validated-at-root first step."""

    plan_id: str
    template: str
    steps: int
    amount: int
    anchor: int  # residents of the plan specialty after step 0 (held by later steps)
    first: Candidate
    anchor2: int = 0  # shelter-and-clothe only: Weaver residents after step 0


def ensure_share(row: np.ndarray, index: int, share: float) -> np.ndarray:
    """A worker-time row with at least ``share`` on ``index``, the rest rescaled."""
    row = np.asarray(row, dtype=np.float64).copy()
    if row[index] >= share - 1e-12:
        return row
    rest = 1.0 - float(row[index])
    if rest <= 1e-12:
        return row
    row *= (1.0 - share) / rest
    row[index] = share
    return row


def _shelter_plan(
    catalog: LaborCatalog, amount: int
) -> tuple[int, int, dict[int, float], dict[int, float]] | None:
    """(Craftsperson adds, Weaver adds, Craftsperson mix, Weaver mix) for shelter-and-clothe.

    Housing, housing material and cloth each get a third of the moved workers.
    Weaver takes the cloth third when it is a legal head with a legal cloth
    activity; otherwise Craftsperson works cloth too. None when housing is illegal.
    """
    craft = SPECIALTIES.index("Craftsperson")
    weaver = SPECIALTIES.index("Weaver")
    housing = _activity_index(catalog, craft - 1, "goods.housing")
    if housing is None or not catalog.masks.specialty[craft]:
        return None
    material = _activity_index(catalog, craft - 1, "goods.housing-material")
    cloth_craft = _activity_index(catalog, craft - 1, "goods.cloth")
    cloth_weaver = (
        _activity_index(catalog, weaver - 1, "goods.cloth")
        if catalog.masks.specialty[weaver]
        else None
    )
    amount = int(amount)
    if cloth_weaver is not None and amount >= 2:
        n_weave = max(1, round(amount / 3))
        n_craft = amount - n_weave
        mix_craft = {housing: 0.5} if material is None else {housing: 0.5, material: 0.5}
        return n_craft, n_weave, mix_craft, {cloth_weaver: 1.0}
    acts = [i for i in (housing, material, cloth_craft) if i is not None]
    return amount, 0, {i: 1.0 / len(acts) for i in acts}, {}


def _blend_row(row: np.ndarray, mix: Mapping[int, float], share: float) -> np.ndarray:
    """``row`` with ``share`` of its worker-time replaced by ``mix`` (a simplex)."""
    out = np.asarray(row, dtype=np.float64).copy() * (1.0 - share)
    for index, weight in mix.items():
        out[index] += share * weight
    return out


def _shelter_step(
    step: int,
    base: Mapping[str, Any],
    catalog: LaborCatalog,
    counts: np.ndarray,
    activity: np.ndarray,
    amount: int,
    anchor: int,
    anchor2: int,
) -> tuple[dict[str, Any], str] | None:
    plan = _shelter_plan(catalog, amount)
    if plan is None:
        return None
    n_craft, n_weave, mix_craft, mix_weave = plan
    craft = SPECIALTIES.index("Craftsperson")
    weaver = SPECIALTIES.index("Weaver")
    new_counts, new_activity = counts.copy(), activity.copy()
    notes = []
    if step == 0:
        for target, n, mix in ((craft, n_craft, mix_craft), (weaver, n_weave, mix_weave)):
            if n <= 0:
                continue
            old = int(new_counts[target])
            previous = new_activity[target - 1].copy() if old > 0 else np.zeros(activity.shape[1])
            moved = transfer(new_counts, new_activity, catalog, GATHERER, target, n)
            if moved is None:
                return None
            new_counts, new_activity = moved
            row = _blend_row(np.zeros(activity.shape[1]), mix, 1.0)
            new_activity[target - 1] = (old * previous + n * row) / (old + n)
        names = "Craftsperson" if n_weave == 0 else "Craftsperson/Weaver"
        description = (
            f"shelter-and-clothe step 0: transfer {amount} Gatherer->{names} "
            "(housing, housing material, cloth)"
        )
    else:
        for target, floor in ((craft, anchor), (weaver, anchor2)):
            shortfall = min(max(0, int(floor) - int(new_counts[target])), int(new_counts[GATHERER]))
            if shortfall > 0:
                moved = transfer(new_counts, new_activity, catalog, GATHERER, target, shortfall)
                if moved is None:
                    return None
                new_counts, new_activity = moved
                notes.append(f"hold {SPECIALTIES[target]} (+{shortfall} from Gatherer)")
        for target, n, mix in ((craft, n_craft, mix_craft), (weaver, n_weave, mix_weave)):
            if n <= 0:
                continue
            if int(new_counts[target]) <= 0:
                return None
            share = min(1.0, n / int(new_counts[target]))
            new_activity[target - 1] = _blend_row(new_activity[target - 1], mix, share)
            notes.append(f"{SPECIALTIES[target]} on {'/'.join(ACTIVITIES[i] for i in mix)}")
        description = f"shelter-and-clothe step {step}: {', '.join(notes)}"
    payload = compose_payload(base, catalog, counts=new_counts, activity=new_activity)
    return payload, description


def plan_step_payload(
    template: str,
    step: int,
    base: Mapping[str, Any],
    catalog: LaborCatalog,
    *,
    amount: int,
    anchor: int = 0,
    anchor2: int = 0,
) -> tuple[dict[str, Any], str] | None:
    """Plan ``template``'s focal action at plan step ``step`` built from H12's ``base``.

    Returns (complete schema-12 payload, description), or None when the step is not
    constructible at this state (the caller then executes H12 and records why).
    """
    if template not in PLAN_TEMPLATES:
        raise ValueError(f"unknown plan template {template!r}")
    counts, activity = labor_state(base, catalog)
    if int(catalog.population) <= 0:
        return None
    spec = SPECIALTIES.index(PLAN_SPECIALTY[template])
    role = spec - 1
    try:
        if template == "shelter-and-clothe":
            return _shelter_step(step, base, catalog, counts, activity, int(amount), anchor, anchor2)
        if step == 0:
            moved = transfer(counts, activity, catalog, GATHERER, spec, int(amount))
            if moved is None:
                return None
            new_counts, new_activity = moved
            description = f"{template} step 0: transfer {amount} Gatherer->{SPECIALTIES[spec]}"
            payload = compose_payload(base, catalog, counts=new_counts, activity=new_activity)
            return payload, description
        new_counts, new_activity = counts.copy(), activity.copy()
        notes = []
        shortfall = min(max(0, int(anchor) - int(counts[spec])), int(counts[GATHERER]))
        if shortfall > 0:
            moved = transfer(counts, activity, catalog, GATHERER, spec, shortfall)
            if moved is None:
                return None
            new_counts, new_activity = moved
            notes.append(f"hold {SPECIALTIES[spec]} (+{shortfall} from Gatherer)")
        if int(new_counts[spec]) <= 0:
            return None
        expansion = _activity_index(catalog, role, PLAN_EXPANSION[template])
        if expansion is None:
            return None
        new_activity[role] = ensure_share(new_activity[role], expansion, 0.5)
        notes.append(f"{SPECIALTIES[spec]} half time to {PLAN_EXPANSION[template]}")
        payload = compose_payload(base, catalog, counts=new_counts, activity=new_activity)
        return payload, f"{template} step {step}: {', '.join(notes)}"
    except ValueError:
        return None


def plan_specs(
    base: Mapping[str, Any],
    catalog: LaborCatalog,
    *,
    count: int,
    steps: int,
    key: Sequence[int],
    unavailable: list[str] | None = None,
) -> list[PlanSpec]:
    """Up to ``count`` constructible plans in a keyed template order.

    Templates that cannot be built at this root (e.g. Farmer or Herder illegal at
    the site, too few Gatherers) are appended to ``unavailable`` when given.
    """
    if count <= 0:
        return []
    population = int(catalog.population)
    amount = max(1, round(0.1 * population))
    out: list[PlanSpec] = []
    for template in _keyed_shuffle(list(PLAN_TEMPLATES), [PLAN_TAG, *key]):
        if len(out) >= count:
            break
        built = plan_step_payload(template, 0, base, catalog, amount=amount)
        if built is None or payload_digest(built[0]) == payload_digest(base):
            if unavailable is not None:
                unavailable.append(template)
            continue
        payload, description = built
        spec = SPECIALTIES.index(PLAN_SPECIALTY[template])
        placed = labor_state(payload, catalog)[0]
        anchor = int(placed[spec])
        anchor2 = int(placed[SPECIALTIES.index("Weaver")]) if template == "shelter-and-clothe" else 0
        out.append(
            PlanSpec(
                plan_id=f"{template}/P{steps}",
                template=template,
                steps=int(steps),
                amount=amount,
                anchor=anchor,
                anchor2=anchor2,
                first=Candidate(payload, PLAN_FAMILY, description, frozenset({"labor"})),
            )
        )
    return out


def dither_payload(
    payload: Mapping[str, Any],
    catalog: LaborCatalog,
    draws: Sequence[float],
    *,
    probability: float,
    fraction: float,
) -> dict[str, Any] | None:
    """The declared h12-dither: maybe move a few residents Gatherer <-> one staffed role.

    ``draws`` are three uniforms (apply, direction, role) so streams use a fixed
    number of draws per camp (common random numbers across branches).
    """
    if draws[0] >= probability or int(catalog.population) <= 0:
        return None
    counts, activity = labor_state(payload, catalog)
    others = [i for i in range(2, len(SPECIALTIES)) if counts[i] > 0 and catalog.masks.specialty[i]]
    if counts[GATHERER] <= 0 or not others:
        return None
    role = others[min(len(others) - 1, int(draws[2] * len(others)))]
    amount = max(1, round(fraction * int(catalog.population)))
    if draws[1] < 0.5:
        moved = transfer(
            counts, activity, catalog, GATHERER, role, min(amount, int(counts[GATHERER]))
        )
    else:
        moved = transfer(counts, activity, catalog, role, GATHERER, min(amount, int(counts[role])))
    if moved is None:
        return None
    try:
        return compose_payload(payload, catalog, counts=moved[0], activity=moved[1])
    except ValueError:
        return None


def candidate_label(
    candidate: Candidate,
    catalog: LaborCatalog,
    h12: HeuristicLaborLabel,
    pre_observation: Mapping[str, Any] | None,
) -> HeuristicLaborLabel:
    """The collector-format label of a validated search candidate.

    Specialty = natively apportioned residents / P, activity = the submitted
    worker-time simplex, planned worker-years = residents x share x dt. A head
    is valid when this candidate's source chose it (H12-valid heads, actor
    heads, perturbed heads); heads inherited from an H12 drive default stay
    invalid, exactly as in the H12 label.
    """
    if candidate.family == "h12":
        return h12
    counts, activity = labor_state(candidate.payload, catalog)
    population = int(catalog.population)
    years = float(catalog.turn_years)
    heads = candidate.chosen_heads
    labor_chosen = "labor" in heads
    labor_valid = bool(labor_chosen or h12.specialty_valid) and population > 0
    roles = activity.shape[0]
    staffed = counts[1 : 1 + roles] > 0
    role_valid = staffed & labor_valid
    planned = counts[1 : 1 + roles, None].astype(np.float64) * years * activity
    payload = candidate.payload
    movement = int(payload["movement"])
    movement_valid = ("movement" in heads or h12.movement_valid) and _movement_legal(
        movement, pre_observation
    )
    fission_valid = bool("fission" in heads or h12.fission_valid)
    gate_valid = (
        np.ones(len(contracts.TRADE_GROUPS), dtype=bool)
        if "trade_gate" in heads
        else np.asarray(h12.trade_gate_valid, dtype=bool).copy()
    )
    gates = np.asarray(payload["tradeGate"], dtype=np.int64)
    value_active = (gates > 0) & gate_valid
    if "trade_gate" not in heads and "trade_value" not in heads:
        value_active = value_active & np.asarray(h12.trade_value_active, dtype=bool)
    share_valid = bool("trade_share" in heads or h12.trade_share_active or "trade_gate" in heads)
    return HeuristicLaborLabel(
        specialty=counts.astype(np.float64) / max(1, population),
        activity=activity.copy(),
        action=dict(payload),
        specialty_counts=counts.copy(),
        planned_worker_years=planned,
        specialty_valid=labor_valid,
        activity_role_valid=role_valid,
        movement_code=movement if movement_valid else -1,
        movement_valid=bool(movement_valid),
        fission_valid=fission_valid,
        trade_gate_valid=gate_valid,
        trade_value_active=value_active,
        trade_share_active=bool(share_valid and value_active.any()),
        over_budget=False,
        reasons=(),
    )


# ---------------------------------------------------------------------------
# Native evaluation
# ---------------------------------------------------------------------------


def physical_digest(step: Mapping[str, Any]) -> str:
    """Digest of a successor step without wall-time and branch-identity tags."""

    def strip(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                k: strip(v)
                for k, v in value.items()
                if not any(tag in str(k).lower() for tag in IDENTITY_KEYS)
            }
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return hashlib.sha256(orjson.dumps(strip(step), option=orjson.OPT_SORT_KEYS)).hexdigest()


def _live(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        v for v in step["villages"] if v.get("observation") is not None and not v.get("terminated")
    ]


def _retained(payload: Mapping[str, Any], catalog: LaborCatalog) -> Any:
    return _retained_action(
        SimpleNamespace(action=dict(payload)), catalog, torch.zeros((1, 1), dtype=torch.bool)
    )


@dataclass
class Work:
    labor_steps: int = 0
    shadow_steps: int = 0
    validations: int = 0
    forks: int = 0
    restores: int = 0

    @property
    def resolves(self) -> int:
        return self.labor_steps + self.shadow_steps

    def state_dict(self) -> dict[str, int]:
        return {
            "labor_steps": self.labor_steps,
            "shadow_steps": self.shadow_steps,
            "native_resolves": self.resolves,
            "validations": self.validations,
            "forks": self.forks,
            "restores": self.restores,
        }


@dataclass
class Branch:
    """One candidate continuation on one stream, extendable by horizon."""

    candidate: int
    stream: int
    fork: int | None = None
    horizon: int = 0
    returns: np.ndarray = field(default_factory=lambda: np.zeros(len(COMPONENTS)))
    lineage: np.ndarray = field(default_factory=lambda: np.zeros(2))
    discount: float = 1.0
    founded: dict[int, int] = field(default_factory=dict)
    societies: set[int] = field(default_factory=set)
    alive: bool = True
    first: dict[str, Any] | None = None
    resolves: int = 0
    steps: list[list[float]] = field(default_factory=list)
    years: list[float] = field(default_factory=list)
    terminated: bool = False
    # Successor rows (--successor-rows): set on first-action candidate branches only.
    record: bool = False
    boundaries: list[dict[str, Any]] = field(default_factory=list)


class NativeSearch:
    """Shadow-H12 continuation and branch evaluation on one labor bridge."""

    def __init__(
        self,
        bridge: Any,
        heuristic: Any,
        *,
        world: int,
        options: LaborResetOptions,
        config: SearchConfig,
    ) -> None:
        self.bridge = bridge
        self.heuristic = heuristic
        self.world = int(world)
        self.options = options
        self.config = config
        self.work = Work()
        # ``successor_view(village) -> {"tensors", "privileged"}`` (--successor-rows):
        # the focal camp's critic-input row at a successor state. Read-only.
        self.successor_view: Any | None = None
        self.successor_seconds = 0.0
        # Paired-branch runner (labor_branches): export a row at EVERY decision, not only
        # at config.successor_decisions. Off for the search itself.
        self.successor_every_decision = False

    def record_boundary(
        self, branch: Branch, step: Mapping[str, Any], focal: int, *, terminal: bool, live: bool
    ) -> None:
        """Record the focal camp's status (and, at exported decisions, its critic-input
        row) at the boundary ``branch.horizon`` just reached. Reads ``step`` only."""
        started = time.perf_counter()
        decision = int(branch.horizon)
        status = "terminated" if terminal else ("live" if live else "inactive")
        entry: dict[str, Any] = {
            "decision": decision,
            "turn": int(step["turn"]),
            "status": status,
            "row": None,
        }
        exported = self.successor_every_decision or decision in self.config.successor_decisions
        if status == "live" and exported:
            village = next(v for v in _live(step) if int(v["camp"]) == focal)
            entry["row"] = self.successor_view(village)
        branch.boundaries.append(entry)
        self.successor_seconds += time.perf_counter() - started

    # -- primitives -------------------------------------------------------
    def catalogs(self) -> dict[int, LaborCatalog]:
        return dict(self.bridge.labor_runtime.catalogs)

    def shadow_h12(
        self, step: Mapping[str, Any], catalogs: Mapping[int, LaborCatalog]
    ) -> dict[int, HeuristicLaborLabel]:
        """Converted H12 labels for every live camp (one heuristic resolve)."""
        snapshot = self.bridge._bridge.capture()
        self.heuristic.reset(
            self.world,
            movement=self.options.movement,
            withheld=self.options.withheld,
            snapshot=snapshot,
        )
        h_step = self.heuristic.step({})
        self.work.shadow_steps += 1
        h_villages = {int(v["camp"]): v for v in h_step.get("villages", [])}
        econ = h_step.get("economicDiagnostics") or {}
        observations = {int(v["camp"]): v["observation"] for v in _live(step)}
        return {
            camp: convert_heuristic_labor_label(
                catalogs[camp],
                h_villages.get(camp, {}),
                econ.get(str(camp)),
                pre_observation=observations.get(camp),
            )
            for camp in sorted(catalogs)
        }

    def validate(self, camp: int, payload: Mapping[str, Any]) -> Any:
        self.work.validations += 1
        return self.bridge.validate_labor({camp: dict(payload)}, attempt=0)[camp]

    def execute(
        self, catalogs: Mapping[int, LaborCatalog], payloads: Mapping[int, Mapping[str, Any]]
    ) -> Any:
        from .labor_ppo import execute_labor_decision

        retained = {camp: _retained(payloads[camp], catalogs[camp]) for camp in sorted(catalogs)}
        execution = execute_labor_decision(self.bridge, dict(catalogs), retained, _idle_retry)
        self.work.labor_steps += 1
        return execution

    def fork(self) -> int:
        fork, _ = self.bridge.fork()
        self.work.forks += 1
        return fork

    def restore(self, fork: int) -> Mapping[str, Any]:
        self.work.restores += 1
        return self.bridge.restore(fork)

    def drop(self, fork: int | None) -> None:
        if fork is not None:
            self.bridge.drop(fork)

    # -- dither -------------------------------------------------------------
    def dithered(
        self,
        payloads: Mapping[int, Mapping[str, Any]],
        catalogs: Mapping[int, LaborCatalog],
        *,
        key: Sequence[int],
        exempt: int | None,
    ) -> dict[int, dict[str, Any]]:
        out = {camp: dict(p) for camp, p in payloads.items()}
        changed: dict[int, dict[str, Any]] = {}
        for camp in sorted(payloads):
            draws = np.random.default_rng([DITHER_TAG, *key, camp]).random(3)
            if camp == exempt:
                continue
            new = dither_payload(
                payloads[camp],
                catalogs[camp],
                draws,
                probability=self.config.dither_probability,
                fraction=self.config.dither_fraction,
            )
            if new is not None:
                changed[camp] = new
        if changed:
            self.work.validations += 1
            attempts = self.bridge.validate_labor(changed, attempt=0)
            for camp, payload in changed.items():
                if attempts[camp].valid:
                    out[camp] = payload
        return out

    # -- branches -----------------------------------------------------------
    def extend(
        self,
        branch: Branch,
        target: int,
        *,
        root_fork: int,
        focal: int,
        candidate: Mapping[str, Any],
        root_h12: Mapping[int, HeuristicLaborLabel],
        root_key: Sequence[int],
        keep_fork: bool,
        focal_policy: Any | None = None,
    ) -> None:
        """Advance ``branch`` to ``target`` decisions (or focal termination).

        ``focal_policy(decision, label, catalog, step)`` (plans) may replace the
        focal camp's H12 action at decisions >= 1; it returns the payload to execute
        or None for H12. A replaced focal action is exempt from the dither, as the
        candidate's first action is.
        """
        from .labor_training import _settlement_guard, _transition_rewards

        if branch.horizon >= target or not branch.alive:
            if not keep_fork:
                self.drop(branch.fork)
                branch.fork = None
            return
        step = self.restore(branch.fork if branch.fork is not None else root_fork)
        catalogs = self.catalogs()
        before = self.work.resolves
        while branch.horizon < target and branch.alive:
            exempt = focal if branch.horizon == 0 else None
            if branch.horizon == 0:
                payloads = {c: root_h12[c].action for c in catalogs}
                payloads[focal] = dict(candidate)
            else:
                labels = self.shadow_h12(step, catalogs)
                payloads = {c: labels[c].action for c in catalogs}
                if focal_policy is not None and focal in labels:
                    chosen = focal_policy(branch.horizon, labels[focal], catalogs[focal], step)
                    if chosen is not None:
                        payloads[focal] = dict(chosen)
                        exempt = focal
            if branch.stream > 0:
                payloads = self.dithered(
                    payloads,
                    catalogs,
                    key=[*root_key, branch.stream, branch.horizon],
                    exempt=exempt,
                )
            decision_turn = int(step["turn"])
            execution = self.execute(catalogs, payloads)
            next_step = execution.step
            rewards, terminal, years = _transition_rewards(
                next_step, (focal,), branch.founded, world_index=0, decision_turn=decision_turn
            )
            outcome = {int(v["camp"]): v for v in next_step["villages"]}
            guard = _settlement_guard(
                outcome[focal], branch.founded.get(focal, decision_turn), int(next_step["turn"])
            )
            attempt = execution.initial_attempts[focal]
            r = np.asarray(
                [
                    rewards[focal]["qol_years"],
                    rewards[focal]["population_years"],
                    float(attempt.feasibility_cost),
                    float(guard),
                ],
                dtype=np.float64,
            )
            branch.returns = branch.returns + branch.discount * r
            branch.lineage = branch.lineage + branch.discount * _lineage_components(
                next_step, branch.societies
            )
            branch.steps.append([float(x) for x in r])
            branch.years.append(float(years[focal]))
            if terminal[focal]:
                branch.terminated = True
            if branch.first is None:
                branch.first = {
                    "rewards": r.copy(),
                    "elapsed": float(years[focal]),
                    "terminal": bool(terminal[focal]),
                    "attempt_valid": bool(attempt.valid),
                }
            branch.discount *= math.exp(-float(years[focal]) / self.config.discount_years)
            branch.horizon += 1
            step = next_step
            catalogs = self.catalogs() if _live(step) else {}
            if terminal[focal] or focal not in catalogs:
                branch.alive = False
            if branch.record and self.successor_view is not None:
                self.record_boundary(
                    branch,
                    step,
                    focal,
                    terminal=bool(terminal[focal]),
                    live=focal in catalogs and any(int(v["camp"]) == focal for v in _live(step)),
                )
        branch.resolves += self.work.resolves - before
        old = branch.fork
        branch.fork = self.fork() if keep_fork and branch.alive else None
        self.drop(old)


def _lineage_components(step: Mapping[str, Any], societies: set[int]) -> np.ndarray:
    """Diagnostic lineage (population-years, population-weighted QoL-years); joins daughters."""
    for v in step["villages"]:
        if int(v.get("parentSociety", 0) or 0) in societies:
            societies.add(int(v["society"]))
    people = weighted = pop_years = 0.0
    years = 0.0
    for v in step["villages"]:
        if int(v["society"]) not in societies or v["terminated"] or int(v["priorPopulation"]) <= 0:
            continue
        p = float(v["priorPopulation"])
        dt = float(v.get("years", 0.0))
        pop_years += p * dt
        people += p
        weighted += p * float(v["qol"])
        years = dt
    return np.asarray([pop_years, weighted / people * years if people > 0 else 0.0])


# ---------------------------------------------------------------------------
# Root search
# ---------------------------------------------------------------------------


@dataclass
class RootResult:
    rows: list[dict[str, torch.Tensor]]
    table: dict[str, Any]
    # Candidate payload root (labor_search_payload); absent for skipped roots.
    payload_root: dict[str, Any] | None = None


def _personality_grid(
    config: SearchConfig, root_pair: tuple[float, float] | None
) -> list[dict[str, Any]]:
    names = {(0.5, 0.5): "balanced"}
    grid: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for pair in config.personalities:
        key = (round(float(pair[0]), 12), round(float(pair[1]), 12))
        if key in seen:
            continue
        seen.add(key)
        grid.append(
            {
                "name": names.get(key, f"({pair[0]:g},{pair[1]:g})"),
                "vector": [float(pair[0]), float(pair[1])],
            }
        )
    if config.root_personality and root_pair is not None:
        key = (round(float(root_pair[0]), 12), round(float(root_pair[1]), 12))
        if key in seen:
            for row in grid:
                if (round(row["vector"][0], 12), round(row["vector"][1], 12)) == key:
                    row["root"] = True
        else:
            grid.append(
                {"name": "root", "vector": [float(root_pair[0]), float(root_pair[1])], "root": True}
            )
    return grid


def _actor_candidates(
    actor: Any,
    tensors: Mapping[str, torch.Tensor],
    row: int,
    personality: tuple[float, float],
    samples: int,
    key: Sequence[int],
) -> list[Candidate]:
    from .labor_model import (
        deterministic_nonlabor_policy,
        sample_labor_policy,
        sample_nonlabor_policy,
        schema12_initial_action,
    )
    from .labor_pretrain_eval import _deterministic_labor_sample

    one = _one_row(tensors, row)
    one["labor_personality"] = torch.tensor([list(personality)], dtype=torch.float32)
    heads = frozenset({"labor", "movement", "fission", "trade_gate", "trade_value", "trade_share"})
    out: list[Candidate] = []
    seed = int(np.random.default_rng([ACTOR_TAG, *key]).integers(0, 2**62))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        output = actor(one)
        labor = _deterministic_labor_sample(output, one)
        nonlabor = deterministic_nonlabor_policy(output)
        out.append(
            Candidate(
                schema12_initial_action(labor, nonlabor, 0),
                "actor-deterministic",
                "actor mode",
                heads,
            )
        )
        for index in range(samples):
            labor = sample_labor_policy(output, one, generator=generator)
            nonlabor = sample_nonlabor_policy(output, generator=generator)
            out.append(
                Candidate(
                    schema12_initial_action(labor, nonlabor, 0),
                    "actor-sample",
                    f"actor sample {index}",
                    heads,
                )
            )
    return out


def _step_entry(
    step: int, payload: Mapping[str, Any], catalog: LaborCatalog, description: str, **extra: Any
) -> dict[str, Any]:
    counts = labor_state(payload, catalog)[0]
    return {
        "step": int(step),
        "action": dict(payload),
        "digest": payload_digest(payload),
        "description": description,
        "conserves": bool(conserves_labor(payload, catalog)),
        "specialty_counts": counts.tolist(),
        "population": int(catalog.population),
        **extra,
    }


def evaluate_plan(
    native: NativeSearch,
    spec: PlanSpec,
    *,
    horizon: int,
    streams: Sequence[int],
    root_fork: int,
    focal: int,
    root_h12: Mapping[int, HeuristicLaborLabel],
    root_key: Sequence[int],
    founded: Mapping[int, int],
    society: int,
    rows: tuple[int, int],
    catalog: LaborCatalog,
) -> dict[str, Any]:
    """Run one plan and its paired first-step control on each stream.

    ``catalog`` is the focal camp's root catalog (the bridge may sit at any
    branch state when this is called).

    Both share the root fork, the first step (executed once per stream, then
    forked) and the stream's dither draws for every other camp. The plan holds
    its course for ``spec.steps`` decisions (each step rebuilt from that state's
    H12 action, natively validated, H12 on failure), then H12 continues; the
    control continues under H12 from decision 2.
    """
    first_entry = _step_entry(
        0,
        spec.first.payload,
        catalog,
        spec.first.description,
        valid=True,
        fallback=None,
        same_as_h12=payload_digest(spec.first.payload) == payload_digest(root_h12[focal].action),
    )
    out: dict[int, dict[str, Any]] = {}
    for stream in streams:
        records: list[dict[str, Any]] = [first_entry]

        def policy(
            decision: int,
            label: HeuristicLaborLabel,
            catalog: LaborCatalog,
            _step: Mapping[str, Any],
            records: list[dict[str, Any]] = records,
        ) -> dict[str, Any] | None:
            if decision >= spec.steps:
                return None
            same = {"same_as_h12": False}
            built = plan_step_payload(
                spec.template,
                decision,
                label.action,
                catalog,
                amount=spec.amount,
                anchor=spec.anchor,
                anchor2=spec.anchor2,
            )
            if built is None:
                records.append(
                    _step_entry(
                        decision,
                        label.action,
                        catalog,
                        "H12 (plan step not constructible)",
                        valid=False,
                        fallback="h12:not-constructible",
                        same_as_h12=True,
                    )
                )
                return None
            payload, description = built
            attempt = native.validate(focal, payload)
            if not attempt.valid:
                reason = ";".join(v.reason for v in attempt.violations) or "native-rejected"
                records.append(
                    _step_entry(
                        decision,
                        label.action,
                        catalog,
                        f"H12 (plan step rejected: {reason})",
                        valid=False,
                        fallback=f"h12:{reason}",
                        same_as_h12=True,
                    )
                )
                return None
            same["same_as_h12"] = payload_digest(payload) == payload_digest(label.action)
            records.append(
                _step_entry(
                    decision, payload, catalog, description, valid=True, fallback=None, **same
                )
            )
            return payload

        plan = Branch(rows[0], stream, founded=dict(founded), societies={int(society)})
        common = {
            "root_fork": root_fork,
            "focal": focal,
            "candidate": spec.first.payload,
            "root_h12": root_h12,
            "root_key": root_key,
        }
        native.extend(plan, 1, keep_fork=horizon > 1, **common)
        control = copy.deepcopy(plan)
        control.candidate = rows[1]
        control.fork = native.fork() if plan.fork is not None else None
        native.extend(plan, horizon, keep_fork=False, focal_policy=policy, **common)
        native.extend(control, horizon, keep_fork=False, **common)
        out[int(stream)] = {"plan": plan, "control": control, "steps": records}
    return {"spec": spec, "horizon": int(horizon), "streams": out}


def _branch_info(branch: Branch) -> dict[str, Any]:
    return {
        "returns": branch.returns.copy(),
        "horizon": branch.horizon,
        "terminated": branch.terminated,
        "years": list(branch.years),
    }


def plan_payload_rows(
    spec: PlanSpec,
    result: Mapping[str, Any],
    catalog: LaborCatalog,
    horizons: Sequence[int],
    horizon: int,
    config: SearchConfig,
) -> list[dict[str, Any]]:
    """The plan row and its first-step-control row for the candidate payload.

    Both carry the shared first step as ``action`` (the actor-visible first
    decision) and the standard ``evaluations`` (stream 0 at every declared
    horizon up to ``horizon``, streams >= 1 at ``horizon``), plus ``plan_id``,
    ``plan_role``, ``plan_steps`` (the executed focal action per step on stream
    0; the control's later steps are H12) and per-step returns per stream.
    """
    streams = result["streams"]
    counts = labor_state(spec.first.payload, catalog)[0]
    rows = []
    for role, family in zip(PLAN_ROLES, (PLAN_FAMILY, PLAN_CONTROL_FAMILY), strict=True):
        key = "plan" if role == "plan" else "control"
        zero = streams[0][key]
        evaluations = payload_module.selection_evaluations(
            zero.steps,
            zero.years,
            horizons,
            terminated=zero.terminated,
            discount_years=config.discount_years,
        )
        evaluations += payload_module.reevaluation_records(
            {s: _branch_info(streams[s][key]) for s in streams if s >= 1}, horizon
        )
        steps = streams[0]["steps"] if role == "plan" else streams[0]["steps"][:1]
        rows.append(
            {
                "index": int(zero.candidate),
                "family": family,
                "is_h12": False,
                "digest": spec.first.digest(),
                "action": dict(spec.first.payload),
                "control": None,
                "bookkeeping_twin": None,
                "evaluations": evaluations,
                "description": (
                    spec.first.description
                    if role == "plan"
                    else f"first-step control of {spec.plan_id}: step 0 then H12"
                ),
                "chosen_heads": sorted(spec.first.chosen_heads),
                "attempt_valid": True,
                "natively_evaluated": True,
                "compiled": payload_module.compiled_action(
                    spec.first.payload, counts, float(catalog.turn_years)
                ),
                "horizon_reached": int(zero.horizon),
                "terminated": bool(zero.terminated),
                "last_round": -1,
                "step_rewards": [list(map(float, r)) for r in zero.steps],
                "step_years": [float(y) for y in zero.years],
                "predicted_advantage": None,
                # T2 plan fields (ignored by first-action readers).
                "plan_id": spec.plan_id,
                "plan_role": role,
                "plan_template": spec.template,
                "plan_length": int(spec.steps),
                "plan_steps": steps,
                "plan_continuation": "h12",
                "plan_step_digests": {
                    str(s): [e["digest"] for e in streams[s]["steps"]]
                    if role == "plan"
                    else [streams[s]["steps"][0]["digest"]]
                    for s in streams
                },
                "plan_fallbacks": {
                    str(s): sum(1 for e in streams[s]["steps"] if e["fallback"])
                    if role == "plan"
                    else 0
                    for s in streams
                },
                "step_returns": {
                    str(s): [list(map(float, r)) for r in streams[s][key].steps] for s in streams
                },
                "step_years_by_stream": {
                    str(s): [float(y) for y in streams[s][key].years] for s in streams
                },
            }
        )
    return rows


def plan_table(spec: PlanSpec, result: Mapping[str, Any]) -> dict[str, Any]:
    """Candidate-table summary of one plan and its control (returns per stream)."""
    streams = result["streams"]
    return {
        "plan_id": spec.plan_id,
        "template": spec.template,
        "steps": int(spec.steps),
        "amount": int(spec.amount),
        "horizon": int(result["horizon"]),
        "first_digest": spec.first.digest(),
        "plan_returns": {str(s): streams[s]["plan"].returns.tolist() for s in streams},
        "control_returns": {str(s): streams[s]["control"].returns.tolist() for s in streams},
        "plan_steps": [{k: v for k, v in e.items() if k != "action"} for e in streams[0]["steps"]]
        if 0 in streams
        else [],
        "fallbacks": {
            str(s): sum(1 for e in streams[s]["steps"] if e["fallback"]) for s in streams
        },
    }


def search_root(
    native: NativeSearch,
    *,
    step: Mapping[str, Any],
    root_h12: Mapping[int, HeuristicLaborLabel],
    focal: int,
    founded: Mapping[int, int],
    tensors: Mapping[str, torch.Tensor],
    row: int,
    root_pair: tuple[float, float] | None,
    actor: Any | None,
    root_fork: int,
    key: Sequence[int],
    root_source: str = "h12",
    root_index: int = 0,
    ranker: Any | None = None,
    root_extra: Mapping[str, Any] | None = None,
) -> RootResult | None:
    """Search one focal camp at one root. The bridge ends at an arbitrary branch.

    Besides the label rows and the candidate table, the result carries the
    versioned candidate payload (``labor_search_payload``): one root row and
    one row per accepted candidate, shard-local root index ``root_index``.
    With ``ranker`` (and ``config.ranker``) every candidate is scored by the
    advantage head first; outside the keyed full-evaluation fraction only the
    top ``ranker_top`` plus H12 are natively evaluated.
    """
    config = native.config
    started = time.perf_counter()
    work0 = copy.copy(native.work)
    catalogs = native.catalogs()
    catalog = catalogs[focal]
    village = next(v for v in _live(step) if int(v["camp"]) == focal)
    observation = village["observation"]
    h12 = root_h12[focal]
    grid = _personality_grid(config, root_pair)
    personalities = [g["vector"] for g in grid]

    # 1. Candidates: H12, actor, structured perturbations; native validation in order.
    proposals: list[Candidate] = [
        Candidate(dict(h12.action), "h12", "converted heuristic", frozenset())
    ]
    control_root = is_control_root(key, config.control_fraction)
    if control_root:
        proposals += control_candidates(h12.action, catalog)
    if actor is not None:
        proposals += _actor_candidates(
            actor, tensors, row, root_pair or (0.5, 0.5), config.actor_samples, [*key, 0]
        )
    proposals += round_robin(
        structured_candidates(
            h12.action,
            catalog,
            observation,
            key=[CAND_TAG, *key],
            movement_candidates=config.movement_candidates,
        )
    )
    accepted: list[Candidate] = []
    attempts: list[Any] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    validations = 0
    for cand in proposals:
        if len(accepted) >= config.single_budget or validations >= config.validation_cap:
            break
        digest = cand.digest()
        if digest in seen:
            continue
        seen.add(digest)
        reason = None
        if not _movement_legal(int(cand.payload["movement"]), observation):
            reason = "movement-illegal-in-pre-action-mask"
        elif not 0 <= int(cand.payload["fission"]) < contracts.FISSION_POLICIES:
            reason = "fission-out-of-range"
        attempt = None
        if reason is None:
            validations += 1
            attempt = native.validate(focal, cand.payload)
            if not attempt.valid:
                reason = ";".join(v.reason for v in attempt.violations) or "native-rejected"
        if reason is not None:
            rejected.append(
                {"family": cand.family, "description": cand.description, "reason": reason}
            )
            if cand.family == "h12":
                return RootResult(
                    [], {"skipped": "h12-rejected", "focal": focal, "rejected": rejected}
                )
            continue
        accepted.append(cand)
        attempts.append(attempt)
    n = len(accepted)
    if n < 2:
        return RootResult(
            [], {"skipped": "fewer-than-two-valid-candidates", "focal": focal, "rejected": rejected}
        )

    # 1a. T2 plans: keyed templates, first step natively validated at the root. Plan
    # and control rows are reserved from M and never enter halving or the labels.
    # Every constructible template in keyed order; the first N natively valid are kept.
    plans: list[PlanSpec] = []
    plan_unavailable: list[str] = []
    for spec in plan_specs(
        h12.action,
        catalog,
        count=len(PLAN_TEMPLATES)
        if config.plan_candidates and is_plan_root(key, config.plan_fraction)
        else 0,
        steps=config.plan_steps,
        key=key,
        unavailable=plan_unavailable,
    ):
        if len(plans) >= config.plan_candidates:
            break
        attempt = native.validate(focal, spec.first.payload)
        if attempt.valid:
            plans.append(spec)
        else:
            rejected.append(
                {
                    "family": PLAN_FAMILY,
                    "description": spec.first.description,
                    "reason": ";".join(v.reason for v in attempt.violations) or "native-rejected",
                }
            )

    controls = {i for i, c in enumerate(accepted) if c.family in CONTROL_FAMILIES}
    # Root of the candidate payload: actor-visible observation row (root personality)
    # and the raw critic-only privileged arrays of the same boundary, kept apart.
    root_vector = [float(x) for x in (root_pair if root_pair is not None else (0.5, 0.5))]
    root_tensors = _one_row(tensors, row)
    root_tensors["labor_personality"] = torch.tensor([root_vector], dtype=torch.float32)
    privileged = privileged_row_tensors([village])
    root_privileged = None if privileged is None else _one_row(privileged, 0)
    tie_perm = np.random.default_rng([CAND_TAG, *key, 3]).permutation(n)
    tiebreak = {i: int(tie_perm[i]) for i in range(n)}

    # 1b. Optional A-ranker (labor_advantage.AdvantageRanker): score every candidate,
    # natively evaluate the top-k plus H12 except on its keyed full-evaluation roots.
    evaluated = list(range(n))
    ranker_mode = payload_module.RANKER_OFF
    predicted: np.ndarray | None = None
    if ranker is not None and config.ranker:
        plan = ranker.plan(
            root_tensors,
            [c.payload for c in accepted],
            h12_index=0,
            personalities=personalities,
            privileged_raw=root_privileged,
            root_key=key,
        )
        predicted = np.asarray(
            [[float(plan.advantage[c][i]) for c in COMPONENTS] for i in range(n)],
            dtype=np.float64,
        )
        evaluated = sorted({0, *(int(i) for i in plan.evaluate), *controls})
        ranker_mode = (
            payload_module.RANKER_FULL if plan.full_evaluation else payload_module.RANKER_PRUNED
        )

    # 2. Successive halving over horizon on the deterministic selection stream.
    record = bool(config.successor_rows)
    branches = {
        i: Branch(i, 0, founded=dict(founded), societies={int(village["society"])}, record=record)
        for i in range(n)
    }
    keep = config.keep_counts(len(evaluated))
    active = list(evaluated)
    rounds: list[dict[str, Any]] = []
    survived = {i: (0 if i in evaluated else -1) for i in range(n)}
    final_ids: list[int] = []
    final_h = 0
    search_resolves = 0
    truncated = False
    for r, horizon in enumerate(config.horizons):
        if r > 0:
            u_prev = utility_matrix(
                np.stack([branches[i].returns for i in range(n)]), personalities, config
            )
            survivors = select_survivors(u_prev, active, keep[r], protected=0, tiebreak=tiebreak)
            # Evaluator controls are carried to the final horizon like H12.
            survivors = sorted({*survivors, *(i for i in active if i in controls)})
            for i in active:
                if i not in survivors:
                    native.drop(branches[i].fork)
                    branches[i].fork = None
            active = survivors
        needed = sum(
            2 * (horizon - branches[i].horizon) - (1 if branches[i].horizon == 0 else 0)
            for i in active
            if branches[i].alive
        )
        if r > 0 and search_resolves + needed > config.max_resolves:
            truncated = True
            break
        last = r == len(config.horizons) - 1
        for i in active:
            before = native.work.resolves
            native.extend(
                branches[i],
                horizon,
                root_fork=root_fork,
                focal=focal,
                candidate=accepted[i].payload,
                root_h12=root_h12,
                root_key=key,
                keep_fork=not last,
            )
            search_resolves += native.work.resolves - before
            survived[i] = r
        u = utility_matrix(np.stack([branches[i].returns for i in range(n)]), personalities, config)
        rounds.append(
            {
                "round": r,
                "horizon": horizon,
                "candidates": list(active),
                "resolves_so_far": search_resolves,
                "utility": {str(i): u[i].tolist() for i in active},
            }
        )
        final_ids, final_h = list(active), horizon
    for b in branches.values():
        native.drop(b.fork)
        b.fork = None

    u_final = utility_matrix(
        np.stack([branches[i].returns for i in range(n)]), personalities, config
    )

    # 3. Per-personality best found; independent paired re-evaluation. Payload v1
    # re-evaluates every final-horizon survivor (H12 is always one), so every
    # same-root pair has paired streams; the label test still uses only H12 and
    # the contenders beyond the selection margin.
    winners = {k: best_for_personality(u_final[:, k], final_ids, 0) for k in range(len(grid))}
    contenders = sorted(
        {
            w
            for k, w in winners.items()
            if w != 0 and u_final[w, k] - u_final[0, k] > config.margin * abs(u_final[0, k])
        }
    )
    if config.reeval_scope == "final-survivors":
        reeval_ids = sorted({0, *final_ids})
    else:
        reeval_ids = [0, *contenders] if contenders else []
    reeval: dict[int, dict[int, np.ndarray]] = {}
    reeval_info: dict[int, dict[int, dict[str, Any]]] = {}
    reeval_resolves = 0
    for cand in reeval_ids:
        reeval[cand] = {}
        reeval_info[cand] = {}
        for stream in range(1, config.reeval_streams + 1):
            branch = Branch(
                cand,
                stream,
                founded=dict(founded),
                societies={int(village["society"])},
                record=record,
            )
            before = native.work.resolves
            native.extend(
                branch,
                final_h,
                root_fork=root_fork,
                focal=focal,
                candidate=accepted[cand].payload,
                root_h12=root_h12,
                root_key=key,
                keep_fork=False,
            )
            reeval_resolves += native.work.resolves - before
            reeval[cand][stream] = branch.returns.copy()
            reeval_info[cand][stream] = {
                "returns": branch.returns.copy(),
                "horizon": branch.horizon,
                "terminated": branch.terminated,
                "years": list(branch.years),
                **(
                    {"steps": [list(r) for r in branch.steps], "boundaries": branch.boundaries}
                    if record
                    else {}
                ),
            }

    # 3b. T2 plans and their paired first-step controls on every stream (0 and the
    # re-evaluation streams, common random numbers with H12), to max(final, P + 1):
    # a decision's reward is credited on its own transition from the prior state, so
    # the last plan step needs one H12 decision after it to show in the returns.
    plan_started = time.perf_counter()
    plan_before = native.work.resolves
    plan_horizon = max(int(final_h), int(config.plan_steps) + 1)
    plan_results = [
        evaluate_plan(
            native,
            spec,
            horizon=plan_horizon,
            streams=range(config.reeval_streams + 1),
            root_fork=root_fork,
            focal=focal,
            root_h12=root_h12,
            root_key=key,
            founded=founded,
            society=int(village["society"]),
            rows=(n + 2 * j, n + 2 * j + 1),
            catalog=catalog,
        )
        for j, spec in enumerate(plans)
    ]
    plan_resolves = native.work.resolves - plan_before
    plan_seconds = time.perf_counter() - plan_started

    decisions = []
    for k, entry in enumerate(grid):
        w = winners[k]
        decision: dict[str, Any] = {
            "personality": entry,
            "best_found": w,
            "best_found_family": accepted[w].family,
            "selection_gain": float(u_final[w, k] - u_final[0, k]),
            "label": 0,
            "reason": "h12-best" if w == 0 else "below-selection-margin",
        }
        if w in contenders and w in reeval:
            streams = sorted(reeval[w])
            wu = [float(utility(reeval[w][s], entry["vector"], config)[0]) for s in streams]
            hu = [float(utility(reeval[0][s], entry["vector"], config)[0]) for s in streams]
            test = paired_acceptance(wu, hu, config.margin, config.se_multiplier)
            decision["reeval"] = test
            if test["accepted"]:
                decision["label"], decision["reason"] = w, "search-beat-h12"
            else:
                decision["reason"] = "reeval-not-significant"
        decisions.append(decision)

    # 4. Rows: one per personality, collector format plus search fields.
    base_row = _one_row(tensors, row)
    if root_privileged is not None:
        base_row.update(root_privileged)
    rows: list[dict[str, torch.Tensor]] = []
    for k, decision in enumerate(decisions):
        index = decision["label"]
        cand = accepted[index]
        label = candidate_label(cand, catalog, h12, observation)
        first = branches[index].first or {}
        values = dict(base_row)
        values["labor_personality"] = torch.tensor([grid[k]["vector"]], dtype=torch.float32)
        values.update(_label_tensors(label))
        reward = first.get("rewards", np.zeros(len(COMPONENTS)))
        reeval_test = decision.get("reeval") or {}
        values.update(
            {
                "row_world": torch.tensor([native.world], dtype=torch.long),
                "row_camp": torch.tensor([focal], dtype=torch.long),
                "row_turn": torch.tensor([int(step["turn"])], dtype=torch.long),
                "row_attempt": torch.tensor([0], dtype=torch.long),
                "row_quarantined": torch.tensor(
                    [not bool(attempts[index].valid)], dtype=torch.bool
                ),
                "reward_qol_years": torch.tensor([float(reward[0])], dtype=torch.float64),
                "reward_population_years": torch.tensor([float(reward[1])], dtype=torch.float64),
                "reward_feasibility": torch.tensor([float(reward[2])], dtype=torch.float64),
                "reward_settlement_guard": torch.tensor([float(reward[3])], dtype=torch.float64),
                "transition_elapsed_years": torch.tensor(
                    [float(first.get("elapsed", 0.0))], dtype=torch.float64
                ),
                "transition_terminal": torch.tensor(
                    [bool(first.get("terminal", False))], dtype=torch.bool
                ),
                "row_source": torch.tensor([ROW_SOURCE_CODES["search"]], dtype=torch.long),
                "row_label_variant": torch.tensor([k], dtype=torch.long),
                "search_label_personality": torch.tensor([grid[k]["vector"]], dtype=torch.float32),
                "search_label_is_h12": torch.tensor([index == 0], dtype=torch.bool),
                "search_label_family": torch.tensor([FAMILY_CODES[cand.family]], dtype=torch.long),
                "search_best_family": torch.tensor(
                    [FAMILY_CODES[decision["best_found_family"]]], dtype=torch.long
                ),
                "search_selection_gain": torch.tensor(
                    [decision["selection_gain"]], dtype=torch.float64
                ),
                "search_reeval_gain": torch.tensor(
                    [float(reeval_test.get("mean_gain", float("nan")))], dtype=torch.float64
                ),
                "search_reeval_se": torch.tensor(
                    [float(reeval_test.get("se", float("nan")))], dtype=torch.float64
                ),
                "search_horizon": torch.tensor([final_h], dtype=torch.long),
                "search_candidates": torch.tensor([n], dtype=torch.long),
            }
        )
        rows.append(values)

    # 5. Candidate payload (labor-advantage-candidates-v1 via labor_search_payload):
    # every accepted candidate's complete action and every native evaluation.
    horizons = [int(h) for h in config.horizons]
    payload_candidates = []
    store = payload_module.SuccessorRows(root_vector) if record else None
    for i in range(n):
        branch = branches[i]
        counts = labor_state(accepted[i].payload, catalog)[0]
        evaluations = (
            payload_module.selection_evaluations(
                branch.steps,
                branch.years,
                horizons,
                terminated=branch.terminated,
                discount_years=config.discount_years,
            )
            if i in evaluated
            else []
        )
        evaluations += payload_module.reevaluation_records(reeval_info.get(i, {}), final_h)
        payload_candidates.append(
            {
                "index": i,
                "family": accepted[i].family,
                "is_h12": i == 0,
                "digest": accepted[i].digest(),
                "action": dict(accepted[i].payload),
                "control": CONTROL_FAMILIES.get(accepted[i].family)
                or ("all-idle" if int(counts[0]) == int(catalog.population) else None),
                "bookkeeping_twin": 0 if accepted[i].family == "control-twin" else None,
                "evaluations": evaluations,
                "description": accepted[i].description,
                "chosen_heads": sorted(accepted[i].chosen_heads),
                "attempt_valid": bool(attempts[i].valid),
                "natively_evaluated": i in evaluated,
                "compiled": payload_module.compiled_action(
                    accepted[i].payload, counts, float(catalog.turn_years)
                ),
                "horizon_reached": int(branch.horizon),
                "terminated": bool(branch.terminated),
                "last_round": int(survived[i]),
                "step_rewards": [list(map(float, r)) for r in branch.steps],
                "step_years": [float(y) for y in branch.years],
                "predicted_advantage": None if predicted is None else predicted[i].tolist(),
            }
        )
        if store is not None:
            streams_out = []
            if i in evaluated:
                streams_out.append(
                    payload_module.successor_stream(
                        0, branch.steps, branch.years, branch.terminated, branch.boundaries, store
                    )
                )
            for s, info in sorted(reeval_info.get(i, {}).items()):
                streams_out.append(
                    payload_module.successor_stream(
                        s, info["steps"], info["years"], info["terminated"], info["boundaries"],
                        store,
                    )
                )
            payload_candidates[-1]["successor_streams"] = streams_out
    plan_horizons = sorted({*(h for h in horizons if h <= plan_horizon), plan_horizon})
    for spec, result in zip(plans, plan_results, strict=True):
        payload_candidates.extend(
            plan_payload_rows(spec, result, catalog, plan_horizons, plan_horizon, config)
        )
    root_record = {
        "root_id": f"{native.world}/{int(step['turn'])}/{focal}/{root_source}",
        "world": native.world,
        "turn": int(step["turn"]),
        "camp": int(focal),
        "society": int(village["society"]),
        "source": root_source,
        "tensor_row": int(root_index),
        "personality": root_vector,
        "full_evaluation": ranker_mode != payload_module.RANKER_PRUNED,
        "pair_sd": None,
        "candidates": payload_candidates,
        "final_horizon": int(final_h),
        "truncated": bool(truncated),
        "horizons": horizons,
        "reeval_streams": int(config.reeval_streams),
        "ranker_mode": int(ranker_mode),
        "control_root": bool(control_root),
        "population": int(catalog.population),
        "turn_years": float(catalog.turn_years),
        # T2 plans: rows with plan_role are diagnostic (not first-action candidates).
        "plan_candidates": len(plans),
        "plan_steps": int(config.plan_steps),
        "plan_horizon": int(plan_horizon) if plans else None,
        "plan_unavailable": list(plan_unavailable),
        **dict(root_extra or {}),
    }
    if store is not None:
        root_record["successor_decisions"] = list(config.successor_decisions)
    root_payload = {
        "record": root_record,
        "tensors": root_tensors,
        "privileged": root_privileged,
    }
    if store is not None:
        root_payload["successors"] = store.rows

    work = native.work
    table = {
        "world": native.world,
        "turn": int(step["turn"]),
        "focal": focal,
        "society": int(village["society"]),
        "population": int(catalog.population),
        "personalities": grid,
        "candidates": [
            {
                "index": i,
                "family": accepted[i].family,
                "description": accepted[i].description,
                "digest": accepted[i].digest(),
                "counts": labor_state(accepted[i].payload, catalog)[0].tolist(),
                "movement": int(accepted[i].payload["movement"]),
                "fission": int(accepted[i].payload["fission"]),
                "trade_gate": [int(g) for g in accepted[i].payload["tradeGate"]],
                "horizon": branches[i].horizon,
                "alive": branches[i].alive,
                "last_round": survived[i],
                "returns": dict(zip(COMPONENTS, branches[i].returns.tolist(), strict=True)),
                "lineage_population_years": float(branches[i].lineage[0]),
                "lineage_qol_years": float(branches[i].lineage[1]),
                "step_rewards": branches[i].steps,
                "utility": u_final[i].tolist(),
                "resolves": branches[i].resolves,
                "reeval_returns": {
                    str(s): v.tolist() for s, v in sorted(reeval.get(i, {}).items())
                },
                "step_years": list(branches[i].years),
                "terminated": branches[i].terminated,
                "natively_evaluated": i in evaluated,
                "predicted_advantage": None if predicted is None else predicted[i].tolist(),
                "payload": dict(accepted[i].payload),
            }
            for i in range(n)
        ],
        "root_index": int(root_index),
        "root_source": root_source,
        "ranker_mode": ranker_mode,
        "reeval_candidates": list(reeval_ids),
        "rejected": rejected,
        "rounds": rounds,
        "final_horizon": final_h,
        "truncated_by_resolve_cap": truncated,
        "decisions": decisions,
        "plan_unavailable": list(plan_unavailable),
        "plans": [
            plan_table(spec, result) for spec, result in zip(plans, plan_results, strict=True)
        ],
        "work": {
            "search_resolves": search_resolves,
            "reeval_resolves": reeval_resolves,
            "plan_resolves": plan_resolves,
            "plan_wall_seconds": plan_seconds,
            "native_resolves": work.resolves - work0.resolves,
            "labor_steps": work.labor_steps - work0.labor_steps,
            "shadow_steps": work.shadow_steps - work0.shadow_steps,
            "validations": work.validations - work0.validations,
            "forks": work.forks - work0.forks,
            "restores": work.restores - work0.restores,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    return RootResult(rows, table, root_payload)


# ---------------------------------------------------------------------------
# World driver
# ---------------------------------------------------------------------------


def root_turns(world: int, key: int, turns: int, roots: int, first: int = 0) -> list[int]:
    """Keyed draw of decision indices in [first, turns) used as roots."""
    pool = list(range(max(0, first), max(0, turns)))
    rng = np.random.default_rng([ROOT_TAG, int(key), int(world)])
    return sorted(int(pool[i]) for i in rng.permutation(len(pool))[: min(roots, len(pool))])


def focal_order(camps: Sequence[int], searched: Mapping[int, int], key: Sequence[int]) -> list[int]:
    """Least-searched camps first (cycling), keyed tie order."""
    camps = sorted(int(c) for c in camps)
    rng = np.random.default_rng([FOCAL_TAG, *key])
    tie = {c: int(i) for c, i in zip(camps, rng.permutation(len(camps)), strict=True)}
    return sorted(camps, key=lambda c: (searched.get(c, 0), tie[c]))


def verify_exact_restore(
    native: NativeSearch, payloads: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    """Repeat one decision from a fork twice; the physical successors must match."""
    fork = native.fork()
    try:
        digests = []
        for _ in range(2):
            native.restore(fork)
            execution = native.execute(native.catalogs(), payloads)
            digests.append(physical_digest(execution.step))
        native.restore(fork)
    finally:
        native.drop(fork)
    if digests[0] != digests[1]:
        raise RuntimeError("search fork/restore is not exact: repeated successors differ")
    return {"equal": True, "digest": digests[0]}


def successor_row_view(normalizer: Any) -> Any:
    """The focal camp's critic-input row at a successor state, in the root-row format:
    ``labor_policy_tensors`` (bounded normalizer; rows are built independently, so a
    one-village batch equals that village's row of the full batch) and the raw
    privileged arrays (``privileged_row_tensors``)."""
    from .labor_model import labor_policy_tensors

    def view(village: Mapping[str, Any]) -> dict[str, Any]:
        _, tensors = labor_policy_tensors([village], normalizer)
        privileged = privileged_row_tensors([village])
        return {
            "tensors": _one_row(tensors, 0),
            "privileged": None if privileged is None else _one_row(privileged, 0),
        }

    return view


def search_world(job: Mapping[str, Any]) -> dict[str, Any]:
    """Drive one fitting world, searching its keyed roots. Returns rows and tables."""
    from .bridge import Model9Bridge
    from .labor_bridge import LaborModel9Bridge
    from .labor_model import (
        deterministic_nonlabor_policy,
        labor_policy_tensors,
        schema12_initial_action,
    )
    from .labor_personality import LaborPersonalityConditioning, personality_for_world
    from .labor_pretrain_eval import _deterministic_labor_sample
    from .labor_teacher_data import load_normalizer

    torch.set_num_threads(1)
    world = int(job["world"])
    config = SearchConfig.from_mapping(job["config"])
    options = LaborResetOptions.from_mapping(job["reset_options"])
    normalizer = load_normalizer(Path(job["normalizer_checkpoint"]))
    actor = None
    if job.get("actor_checkpoint"):
        from .labor_pretrain_eval import _load_models

        _, actor, _, _, _ = _load_models(Path(job["actor_checkpoint"]))
    conditioning = LaborPersonalityConditioning.from_state(job.get("personality_conditioning"))
    resolver = (
        personality_for_world(conditioning.assignment(0), world) if conditioning.alpha else None
    )
    source = job["source"]
    if source == "actor" and actor is None:
        raise ValueError("actor-visited roots need --actor-checkpoint")
    ranker = None
    if job.get("ranker_checkpoint"):
        if not config.ranker:
            raise ValueError("a ranker checkpoint needs the declared ranker configuration")
        ranker = payload_module.load_ranker(
            Path(job["ranker_checkpoint"]),
            Path(job["ranker_critic_checkpoint"]),
            top_k=config.ranker_top,
            full_fraction=config.ranker_full_fraction,
            key=int(job["key"]),
        )
    elif config.ranker:
        raise ValueError("the declared ranker configuration needs a ranker checkpoint")
    turns = int(job["turns"])
    scenario_roots: dict[int, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    if source == "scenario":
        # Curated roots: their decision indices on the H12 trajectory, their camps as focals.
        for root in job.get("scenario_roots") or []:
            scenario_roots[int(root["decision_index"])][int(root["camp"])] = root
        chosen = set(scenario_roots)
    else:
        chosen = set(
            root_turns(
                world, int(job["key"]), turns, int(job["roots"]), int(job.get("first_turn", 0))
            )
        )
    rows: list[dict[str, torch.Tensor]] = []
    tables: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    searched: dict[int, int] = defaultdict(int)
    restore_checks: list[dict[str, Any]] = []
    payload_roots: list[dict[str, Any]] = []
    started = time.perf_counter()
    # Decision-boundary checkpoints (labor_search_resume): a resumed world continues from
    # its newest checkpoint and loses at most the in-flight decision and its root.
    from . import labor_search_resume as lsr

    work_dir = job.get("checkpoint_dir")
    checkpoint_identity = {
        "schema": lsr.SEARCH_CHECKPOINT_SCHEMA,
        "source_run": job.get("source_run"),
        "world": world,
    }
    resume_point = None
    if work_dir is not None:
        if job.get("resume"):
            resume_point = lsr.load_search_resume_point(
                Path(work_dir), checkpoint_identity, log=_log_json
            )
        elif Path(work_dir).exists():
            import shutil

            shutil.rmtree(work_dir)
    with LaborModel9Bridge() as bridge, Model9Bridge(economic_diagnostics=True) as heuristic:
        native = NativeSearch(bridge, heuristic, world=world, options=options, config=config)
        if config.successor_rows:
            native.successor_view = successor_row_view(normalizer)
        first_index = 0
        if resume_point is not None:
            step = resume_point.restore(bridge, world, options.reset_kwargs())
            state = resume_point.loop_state()
            first_index = resume_point.boundary
            identity = state["reset_identity"]
            continuation = state["continuation"]
            founded = state["founded"]
            skipped = state["skipped"]
            searched = state["searched"]
            restore_checks = state["restore_checks"]
            main_resolves = state["main_resolves"]
            native.work = Work(
                **{k: int(v) for k, v in state["work"].items() if k != "native_resolves"}
            )
            rows = list(resume_point.rows)
            tables = list(resume_point.tables)
            payload_roots = list(resume_point.payload_roots)
            _log_json({"world": world, "search_resumed_from_checkpoint": first_index,
                       "roots_kept": len(payload_roots)})
        else:
            step = bridge.reset(world, **options.reset_kwargs())
            identity = bridge.last_reset_identity
            founded = {int(v["camp"]): int(step["turn"]) for v in step["villages"]}
            main_resolves = 0
            continuation = {
                "startTurn": int(step["turn"]),
                "trajectoryId": str(identity.get("fingerprint")),
            }
        checkpointer = (
            None
            if work_dir is None
            else lsr.SearchCheckpointer(
                Path(work_dir),
                checkpoint_identity,
                keep=int(job.get("checkpoint_keep", 2)),
                resume=resume_point,
                log=_log_json,
            )
        )
        for index in range(first_index, max(chosen) + 1 if chosen else 0):
            live = _live(step)
            if not live:
                break
            catalogs = native.catalogs()
            root_h12 = (
                native.shadow_h12(step, catalogs)
                if (source in ("h12", "scenario") or index in chosen)
                else {}
            )
            if source in ("h12", "scenario"):
                main_payloads = {c: root_h12[c].action for c in catalogs}
            else:
                _, drive = labor_policy_tensors(live, normalizer, personality=resolver)
                with torch.no_grad():
                    output = actor(drive)
                    labor = _deterministic_labor_sample(output, drive)
                    nonlabor = deterministic_nonlabor_policy(output)
                main_payloads = {
                    int(v["camp"]): schema12_initial_action(labor, nonlabor, r)
                    for r, v in enumerate(live)
                }
            if index in chosen:
                if bool(job.get("verify_restore", True)) and not restore_checks:
                    restore_checks.append(verify_exact_restore(native, main_payloads))
                _, tensors = labor_policy_tensors(live, normalizer)
                pairs = resolver.resolve(live) if resolver is not None else [None] * len(live)
                row_of = {int(v["camp"]): r for r, v in enumerate(live)}
                pair_of = {int(v["camp"]): pairs[r] for r, v in enumerate(live)}
                eligible = [c for c in catalogs if catalogs[c].population > 0]
                focal_limit = config.focal_per_root
                at_root = scenario_roots.get(index, {})
                if source == "scenario":
                    turns_here = {int(r["turn"]) for r in at_root.values()}
                    if turns_here != {int(step["turn"])}:
                        raise ValueError(
                            f"world {world} decision {index}: scenario roots name turns "
                            f"{sorted(turns_here)}, the H12 replay is at turn {int(step['turn'])}"
                        )
                    eligible = [c for c in eligible if c in at_root]
                    focal_limit = len(at_root)
                order = focal_order(eligible, searched, [world, int(job["key"]), int(step["turn"])])
                root_fork = native.fork()
                done = 0
                for focal in order:
                    if done >= focal_limit:
                        break
                    result = search_root(
                        native,
                        step=step,
                        root_h12=root_h12,
                        focal=focal,
                        founded=founded,
                        tensors=tensors,
                        row=row_of[focal],
                        root_pair=pair_of[focal],
                        actor=actor,
                        root_fork=root_fork,
                        key=[world, int(job["key"]), int(step["turn"]), focal],
                        root_source="h12" if source == "scenario" else source,
                        root_index=len(payload_roots),
                        ranker=ranker,
                        root_extra=(
                            {
                                # Distinct from an h12-source root of the same state.
                                "root_id": f"{world}/{int(step['turn'])}/{focal}/scenario",
                                "scenario_root_id": str(at_root[focal]["root_id"]),
                                "scenario_family": list(at_root[focal]["families"]),
                                "root_kind": "curated_neolithic_scenario",
                            }
                            if source == "scenario"
                            else None
                        ),
                    )
                    searched[focal] += 1
                    if result is None or not result.rows:
                        skipped[(result.table.get("skipped") if result else "none")] += 1
                        if result is not None:
                            tables.append(result.table)
                        continue
                    rows.extend(result.rows)
                    tables.append(result.table)
                    if result.payload_root is not None:
                        payload_roots.append(result.payload_root)
                    done += 1
                    print(
                        json.dumps(
                            {
                                "world": world,
                                "turn": int(step["turn"]),
                                "focal": focal,
                                "work": result.table["work"],
                                "wall_seconds": round(result.table["wall_seconds"], 1),
                                "labels": [d["reason"] for d in result.table["decisions"]],
                                **(
                                    {"successor_seconds_total": round(native.successor_seconds, 2)}
                                    if config.successor_rows
                                    else {}
                                ),
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                native.restore(root_fork)
                native.drop(root_fork)
            from .labor_training import _transition_rewards

            if checkpointer is not None:
                # The boundary this decision starts from (the root fork was restored above).
                boundary_snapshot = bridge._bridge.capture()
                boundary_transient = bridge.capture_transient()
            execution = native.execute(native.catalogs(), main_payloads)
            _transition_rewards(
                execution.step,
                tuple(catalogs),
                founded,
                world_index=0,
                decision_turn=int(step["turn"]),
            )
            # The root shadow step is shared with the search; the restore check is not counted.
            main_resolves += (1 if root_h12 else 0) + 1
            step = execution.step
            if checkpointer is not None:
                checkpointer.write(
                    boundary=index + 1,
                    snapshot=boundary_snapshot,
                    transient=boundary_transient,
                    execution=execution,
                    next_step=step,
                    rows=rows,
                    tables=tables,
                    payload_roots=payload_roots,
                    founded=founded,
                    skipped=skipped,
                    searched=searched,
                    restore_checks=restore_checks,
                    main_resolves=main_resolves,
                    work=native.work.state_dict(),
                    reset_identity=identity,
                    continuation=continuation,
                )
    wall = (
        checkpointer.wall_seconds() if checkpointer is not None else time.perf_counter() - started
    )
    return {
        "world": world,
        "rows": rows,
        "tables": tables,
        "skipped": dict(skipped),
        "reset_identity": identity,
        "root_turn_indices": sorted(chosen),
        "restore_checks": restore_checks,
        "work": native.work.state_dict(),
        "main_path_resolves": main_resolves,
        "wall_seconds": wall,
        "payload_roots": payload_roots,
        "checkpoints": None if checkpointer is None else checkpointer.summary(),
    }


def _log_json(record: Mapping[str, Any]) -> None:
    print(json.dumps(dict(record), sort_keys=True, default=str), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Shards, manifest and report
# ---------------------------------------------------------------------------


def write_search_shard(job: Mapping[str, Any]) -> dict[str, Any]:
    from .labor_search_resume import ContinuationMismatch
    from .labor_teacher_data import write_shard
    from .provenance import ProvenanceError, checkpoint_hash

    fallback = None
    try:
        result = search_world(job)
    except ContinuationMismatch as error:
        # The checkpoint cannot reproduce its boundary: redo the world from its reset (the
        # uninterrupted run by definition), logged and recorded in the shard meta.
        import shutil

        fallback = str(error)
        _log_json({"world": int(job["world"]), "resume_refused": fallback,
                   "action": "searching the world from its reset"})
        if job.get("checkpoint_dir"):
            shutil.rmtree(job["checkpoint_dir"], ignore_errors=True)
        result = search_world(dict(job, resume=False))
    world = result["world"]
    directory = Path(job["out"]) / f"world-{world}"
    summary = {
        "world_seed": world,
        "rows": len(result["rows"]),
        "tables": result["tables"],
        "skipped": result["skipped"],
        "work": result["work"],
        "main_path_resolves": result["main_path_resolves"],
        "wall_seconds": result["wall_seconds"],
        "restore_checks": result["restore_checks"],
        "search_checkpoints": result.get("checkpoints"),
        "search_resume_fallback": fallback,
    }
    if not result["rows"]:
        # No label rows (every root skipped): a completion marker lets --resume skip
        # the world; it is a file, so the published dataset holds no shard for it.
        summary["path"] = None
        marker = empty_marker(Path(job["out"]), world)
        temporary = marker.with_name(marker.name + ".tmp")
        temporary.write_text(
            json.dumps({"source_run": job["source_run"], "summary": summary}, allow_nan=True),
            encoding="utf-8",
        )
        os.replace(temporary, marker)
        _drop_checkpoints(job)
        return summary
    tensors = _concat(result["rows"], f"labor search world {world}")
    keep = ~tensors["row_quarantined"].reshape(-1)
    tensors["row_weight"] = world_balanced_weights(tensors["row_world"], keep)
    arrays = {k: v.detach().cpu().numpy() for k, v in tensors.items()}
    table_text = "".join(
        json.dumps(t, sort_keys=True, allow_nan=True) + "\n" for t in result["tables"]
    )
    labels: Counter[str] = Counter()
    for t in result["tables"]:
        for d in t.get("decisions", []):
            labels[d["reason"]] += 1
    meta = {
        "source_run": job["source_run"],
        "world_seed": world,
        "row_source": "search",
        "row_source_codes": ROW_SOURCE_CODES,
        "family_codes": FAMILY_CODES,
        "search_contract": SEARCH_CONTRACT,
        "turns": int(job["turns"]),
        "reset_options": job["reset_options"],
        "reset_identity": result["reset_identity"],
        "normalizer_digest": job["normalizer_digest"],
        "root_turn_indices": result["root_turn_indices"],
        "counters": {
            "total_decisions": len(result["rows"]),
            **{f"label_{k}": v for k, v in labels.items()},
        },
        "label_unavailable_reasons": {},
        "quarantine": [],
        "skipped": result["skipped"],
        "work": result["work"],
        "main_path_resolves": result["main_path_resolves"],
        "wall_seconds": float(result["wall_seconds"]),
        "restore_checks": result["restore_checks"],
        "search_checkpoints": result.get("checkpoints"),
        "search_resume_fallback": fallback,
        "candidates_sha256": hashlib.sha256(table_text.encode("utf-8")).hexdigest(),
        "return_tail": (
            "search rows are labels, not V targets; reward_* is the label's first transition"
        ),
        "partition": job.get("partition", "fitting"),
    }
    # Candidate payload (labor-advantage-candidates-v1): saved first so the label
    # shard's meta can bind its hash, then moved into the staged shard.
    config = SearchConfig.from_mapping(job["config"])
    shard = payload_module.candidate_shard(
        result["payload_roots"],
        producer={
            "exporter": "labor_search",
            "search_contract": SEARCH_CONTRACT,
            "source_run": job["source_run"],
            "world": int(world),
            "partition": job.get("partition", "fitting"),
            "root_source": job["source"],
            "search": config.state_dict(),
            "reference_scales": dict(zip(COMPONENTS, _REFERENCE_SCALES.tolist(), strict=True)),
            "family_codes": FAMILY_CODES,
            "movement_union_codes": list(payload_module.MOVEMENT_UNION_CODES),
            "ranker_checkpoint_sha256": job.get("ranker_checkpoint_sha256"),
        },
        discount_years=config.discount_years,
    )
    payload_staging = directory.with_name(directory.name + ".payload-staging")
    if payload_staging.exists():
        raise ProvenanceError(f"{payload_staging} exists; no overwrite")
    meta["candidate_payload"] = payload_module.write_candidate_shard(
        payload_staging / payload_module.PAYLOAD_FILE, shard
    )
    # Stage the shard (arrays, meta last, then the bound candidate table) and
    # rename it into place; dataset.json is published only after every shard.
    staging = directory.with_name(directory.name + ".staging")
    record = write_shard(staging, arrays, meta)
    (staging / "candidates.jsonl").write_text(table_text, encoding="utf-8")
    os.replace(payload_staging / payload_module.PAYLOAD_FILE, staging / payload_module.PAYLOAD_FILE)
    payload_staging.rmdir()
    os.replace(staging, directory)
    _drop_checkpoints(job)  # the published shard supersedes its checkpoints
    summary.update(
        {
            "path": directory.name,
            "meta_sha256": checkpoint_hash(directory / "meta.json"),
            "data_digest": record["data_digest"],
        }
    )
    return summary


# ---------------------------------------------------------------------------
# Preemption-safe resume
# ---------------------------------------------------------------------------
#
# A world is complete when its shard directory ``world-<seed>`` exists (it is
# renamed into place only after arrays, meta, candidate table and payload are
# written) and verifies against the run's source record, or, for a world with no
# label rows, when its ``world-<seed>.empty.json`` marker names the run. Staging
# directories are partial worlds: --resume discards them and redoes the world
# into a fresh shard. source.json pins the run; a resume whose source record
# differs (learner code, bridge, checkpoints or any option) is refused.

SOURCE_FILE = "source.json"
_PARTIAL_SUFFIXES = (".staging", ".payload-staging")


def _drop_checkpoints(job: Mapping[str, Any]) -> None:
    import shutil

    directory = job.get("checkpoint_dir")
    if directory:
        shutil.rmtree(directory, ignore_errors=True)
        parent = Path(directory).parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()


def empty_marker(out: Path, world: int) -> Path:
    return Path(out) / f"world-{int(world)}.empty.json"


def _shard_summary(directory: Path, world: int, source_run: str) -> dict[str, Any] | None:
    """The write_search_shard summary of a verified complete shard, else None."""
    from .labor_teacher_data import verify_shard
    from .provenance import ProvenanceError, checkpoint_hash

    try:
        meta = verify_shard(directory)
        if meta.get("source_run") != source_run or int(meta["world_seed"]) != int(world):
            return None
        text = (directory / "candidates.jsonl").read_text(encoding="utf-8")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != meta.get("candidates_sha256"):
            return None
        payload_module.verify_candidate_payload(directory, meta)
        tables = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (ProvenanceError, OSError, ValueError, KeyError, TypeError):
        return None
    return {
        "world_seed": int(world),
        "rows": int(meta["rows"]),
        "tables": tables,
        "skipped": meta.get("skipped", {}),
        "work": meta.get("work", {}),
        "main_path_resolves": meta.get("main_path_resolves", 0),
        "wall_seconds": float(meta.get("wall_seconds", 0.0)),
        "restore_checks": meta.get("restore_checks", []),
        "path": directory.name,
        "meta_sha256": checkpoint_hash(directory / "meta.json"),
        "data_digest": meta["data_digest"],
    }


def existing_search_world(out: Path, world: int, source_run: str) -> dict[str, Any] | None:
    """A completed world's summary (kept), or None after discarding partial state."""
    import shutil

    out = Path(out)
    directory = out / f"world-{int(world)}"
    for suffix in _PARTIAL_SUFFIXES:
        partial = directory.with_name(directory.name + suffix)
        if partial.exists():
            shutil.rmtree(partial)  # interrupted mid-write: redo into a fresh shard
    marker = empty_marker(out, world)
    temporary = marker.with_name(marker.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    from .labor_search_resume import search_work_dir

    if directory.exists():
        summary = _shard_summary(directory, world, source_run)
        if summary is None:
            shutil.rmtree(directory)  # unverifiable or another run's shard
        else:
            shutil.rmtree(search_work_dir(out, world), ignore_errors=True)
        if marker.exists():
            marker.unlink()
        return summary
    if marker.is_file():
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            summary = data["summary"]
            if data.get("source_run") == source_run and int(summary["world_seed"]) == int(world):
                shutil.rmtree(search_work_dir(out, world), ignore_errors=True)
                return summary
        except (OSError, ValueError, KeyError, TypeError):
            pass
        marker.unlink()
    return None


def prepare_search_output(
    out: Path, source: Mapping[str, Any], worlds: Sequence[int], *, resume: bool
) -> dict[int, dict[str, Any]]:
    """Create (or, with ``resume``, reopen) the output; returns completed worlds' summaries."""
    from .provenance import ProvenanceError

    out = Path(out)
    if out.exists() and not resume:
        raise ProvenanceError(
            "labor search output exists; pass --resume or regenerate into a new directory"
        )
    out.mkdir(parents=True, exist_ok=True)
    source_path = out / SOURCE_FILE
    if source_path.is_file():
        previous = json.loads(source_path.read_text(encoding="utf-8"))
        if previous.get("source_run") != source["source_run"]:
            raise ProvenanceError(
                "labor search resume: source record differs (learner code, bridge, checkpoint "
                "or options); regenerate into a new directory"
            )
    else:
        if any(p.name != SOURCE_FILE + ".tmp" for p in out.iterdir()):
            raise ProvenanceError("labor search resume: output has no source record")
        temporary = out / (SOURCE_FILE + ".tmp")
        temporary.write_text(json.dumps(dict(source), sort_keys=True, allow_nan=False))
        os.replace(temporary, source_path)
    from .labor_search_resume import WORK_DIR

    declared = {f"world-{int(w)}{suffix}" for w in worlds for suffix in ("", *_PARTIAL_SUFFIXES)}
    declared.add(WORK_DIR)  # decision-boundary checkpoints of unfinished worlds
    undeclared = sorted(p.name for p in out.iterdir() if p.is_dir() and p.name not in declared)
    if undeclared:
        raise ProvenanceError(f"labor search output holds undeclared directories: {undeclared}")
    done = {}
    for world in worlds:
        summary = existing_search_world(out, int(world), source["source_run"])
        if summary is not None:
            done[int(world)] = summary
    return done


def run_search_jobs(
    out: Path,
    source: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    *,
    workers: int = 1,
    resume: bool = False,
    before_publish: Any | None = None,
) -> dict[str, Any]:
    """Search every world not already complete, then report and publish dataset.json.

    dataset.json is written only after every declared world is complete (this run
    or a verified earlier one); an exception anywhere leaves the output resumable.
    """
    from .labor_teacher_data import publish_dataset

    out = Path(out)
    worlds = [int(job["world"]) for job in jobs]
    done = prepare_search_output(out, source, worlds, resume=resume)
    pending = [job for job in jobs if int(job["world"]) not in done]
    print(
        json.dumps({"worlds": len(worlds), "resumed_worlds": len(done), "jobs": len(pending)}),
        file=sys.stderr,
        flush=True,
    )
    started = time.perf_counter()
    if workers <= 1 or len(pending) <= 1:
        fresh = [write_search_shard(job) for job in pending]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            fresh = list(pool.map(write_search_shard, pending))
    if before_publish is not None:
        before_publish()
    completed = {**done, **{int(s["world_seed"]): s for s in fresh}}
    missing = sorted(set(worlds) - set(completed))
    if missing:
        raise RuntimeError(f"labor search worlds incomplete: {missing[:5]}; not published")
    summaries = [completed[w] for w in worlds]
    report = search_report(summaries)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["jobs"] = int(workers)
    report["source_run"] = source["source_run"]
    report["resumed_worlds"] = sorted(done)
    report["worlds_this_run"] = sorted(int(s["world_seed"]) for s in fresh)
    temporary = out / "search_report.json.tmp"
    temporary.write_text(json.dumps(report, sort_keys=True, indent=1))
    os.replace(temporary, out / "search_report.json")
    from .labor_search_resume import WORK_DIR

    work_root = out / WORK_DIR
    if work_root.is_dir() and not any(work_root.iterdir()):
        work_root.rmdir()
    shards = [s for s in summaries if s.get("path")]
    if shards:
        publish_dataset(
            out,
            source,
            [
                {k: s[k] for k in ("path", "world_seed", "meta_sha256", "data_digest", "rows")}
                for s in shards
            ],
        )
    return report


def search_source_record(
    options: Mapping[str, Any], schema: Any, bridge_sha256: str, *, production: bool
) -> dict:
    from .protocol import scan
    from .provenance import ProvenanceError, code_commit, digest, simulator_identity

    scan(dict(options))
    learner = code_commit()
    if production and not re.fullmatch(r"[0-9a-f]{40}", learner):
        raise ProvenanceError("labor search generation requires a clean learner commit")
    body = {
        "schema": SCHEMA,
        "source_kind": SOURCE_KIND,
        "search_contract": SEARCH_CONTRACT,
        "label_contract": LABOR_LABEL_CONTRACT,
        "learner_commit": learner,
        "simulator": simulator_identity(schema, production=production),
        "bridge_sha256": bridge_sha256,
        "options": dict(options),
        "production": bool(production),
        # Simulator rule flags alter the physics, so a resume must refuse a change.
        "flags": {
            "sim_rules": os.environ.get("ANTHROPOCENE_SIM_RULES", ""),
            "sim_statics": os.environ.get("ANTHROPOCENE_SIM_STATICS", ""),
        },
    }
    return dict(body, source_run=digest(body))


def search_report(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Win fractions per personality, mean gains, winning families and work per root."""
    tables = [t for s in summaries for t in s["tables"] if "decisions" in t]
    per: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "roots": 0,
            "search_beat_h12": 0,
            "gain_u": [],
            "gain_components": [],
            "reeval_components": [],
            "families": Counter(),
            "best_found_families": Counter(),
            "reasons": Counter(),
        }
    )
    resolves, seconds, search_res, reeval_res = [], [], [], []
    plan_res, plan_seconds = [], []
    for t in tables:
        resolves.append(t["work"]["native_resolves"])
        search_res.append(t["work"]["search_resolves"])
        reeval_res.append(t["work"]["reeval_resolves"])
        plan_res.append(t["work"].get("plan_resolves", 0))
        plan_seconds.append(t["work"].get("plan_wall_seconds", 0.0))
        seconds.append(t["wall_seconds"])
        h12 = t["candidates"][0]
        for d in t["decisions"]:
            name = d["personality"]["name"]
            row = per[name]
            row["roots"] += 1
            row["reasons"][d["reason"]] += 1
            row["best_found_families"][d["best_found_family"]] += 1
            if d["label"] != 0:
                win = t["candidates"][d["label"]]
                row["search_beat_h12"] += 1
                row["families"][win["family"]] += 1
                row["gain_u"].append(d["reeval"]["mean_gain"])
                row["gain_components"].append(
                    [win["returns"][c] - h12["returns"][c] for c in COMPONENTS]
                )
                streams = sorted(win["reeval_returns"])
                row["reeval_components"].append(
                    np.mean(
                        [
                            np.subtract(win["reeval_returns"][s], h12["reeval_returns"][s])
                            for s in streams
                        ],
                        axis=0,
                    ).tolist()
                )
    out_per = {}
    for name, row in per.items():
        comps = np.asarray(row["gain_components"], dtype=np.float64).reshape(-1, len(COMPONENTS))
        recomps = np.asarray(row["reeval_components"], dtype=np.float64).reshape(
            -1, len(COMPONENTS)
        )
        out_per[name] = {
            "roots": row["roots"],
            "search_beat_h12": row["search_beat_h12"],
            "win_fraction": row["search_beat_h12"] / max(1, row["roots"]),
            "mean_reeval_gain_u_among_wins": float(np.mean(row["gain_u"]))
            if row["gain_u"]
            else None,
            "mean_gain_u_all_roots": float(np.sum(row["gain_u"]) / max(1, row["roots"])),
            "mean_selection_component_gain_among_wins": (
                dict(zip(COMPONENTS, comps.mean(axis=0).tolist(), strict=True))
                if len(comps)
                else None
            ),
            "mean_reeval_component_gain_among_wins": (
                dict(zip(COMPONENTS, recomps.mean(axis=0).tolist(), strict=True))
                if len(recomps)
                else None
            ),
            "winning_families": dict(row["families"]),
            "best_found_families": dict(row["best_found_families"]),
            "reasons": dict(row["reasons"]),
        }
    return {
        "roots": len(tables),
        "skipped_roots": sum(sum(s["skipped"].values()) for s in summaries),
        "per_personality": out_per,
        "native_resolves_per_root": _stats(resolves),
        "search_resolves_per_root": _stats(search_res),
        "reeval_resolves_per_root": _stats(reeval_res),
        "plan_resolves_per_root": _stats(plan_res),
        "plan_wall_seconds_per_root": _stats(plan_seconds),
        "wall_seconds_per_root": _stats(seconds),
        "main_path_resolves": sum(int(s["main_path_resolves"]) for s in summaries),
        "world_wall_seconds": _stats([s["wall_seconds"] for s in summaries]),
        "restore_checks": [c for s in summaries for c in s["restore_checks"]],
    }


def _stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
        "total": float(a.sum()),
    }


def _parse_personalities(text: str) -> tuple[tuple[float, float], ...]:
    out = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        if item == "balanced":
            out.append((0.5, 0.5))
            continue
        q, p = (float(x) for x in item.split(","))
        out.append((q, p))
    return tuple(out)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--normalizer-checkpoint",
        type=Path,
        required=True,
        help="bounded normalizer file or labor trainer checkpoint (row inputs)",
    )
    parser.add_argument(
        "--actor-checkpoint",
        type=Path,
        default=None,
        help="labor trainer checkpoint: adds actor deterministic/sampled candidates",
    )
    parser.add_argument("--worlds", required=True, help="fitting seeds only (90100-90159)")
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="h12",
        help="root trajectory driver: H12 teacher or the actor (deterministic)",
    )
    parser.add_argument(
        "--scenario-roots",
        type=Path,
        default=None,
        help="labor_scenarios roots.json (or its dataset directory); required with "
        "--source scenario, whose roots replace the keyed root draw",
    )
    parser.add_argument(
        "--turns", type=int, default=16, help="trajectory decisions roots are drawn from"
    )
    parser.add_argument("--first-turn", type=int, default=0)
    parser.add_argument("--roots-per-world", type=int, default=8)
    parser.add_argument("--root-key", type=int, default=0)
    parser.add_argument("--focal-per-root", type=int, default=1)
    parser.add_argument(
        "--candidates", type=int, default=16, help="M: complete candidates per focal camp"
    )
    parser.add_argument("--actor-samples", type=int, default=4)
    parser.add_argument("--horizons", default="1,2,3,5")
    parser.add_argument(
        "--keep", default="all,1/2,1/4,2", help="survivors per round: all, a/b or a count"
    )
    parser.add_argument("--personalities", default="balanced;0.9,0.1;0.1,0.9")
    parser.add_argument("--no-root-personality", action="store_true")
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--se-multiplier", type=float, default=2.0)
    parser.add_argument("--reeval-streams", type=int, default=3)
    parser.add_argument("--dither-probability", type=float, default=0.5)
    parser.add_argument("--dither-fraction", type=float, default=0.05)
    parser.add_argument(
        "--max-resolves", type=int, default=400, help="search resolve cap per focal root"
    )
    parser.add_argument("--movement-candidates", type=int, default=3)
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--separation", type=int, default=None)
    parser.add_argument("--personality", default="balanced", help="root personality distribution")
    parser.add_argument("--personality-key", type=int, default=0)
    parser.add_argument("--personality-drift", type=float, default=200.0)
    parser.add_argument(
        "--reeval-scope",
        choices=REEVAL_SCOPES,
        default="final-survivors",
        help="re-evaluate every final-horizon survivor (payload default) or only H12 and the "
        "contenders (legacy)",
    )
    parser.add_argument(
        "--advantage-ranker",
        type=Path,
        default=None,
        help="intervention B advantage-head artifact (labor_advantage fit); off by default. "
        "Scores all candidates and natively evaluates the top --ranker-top plus H12",
    )
    parser.add_argument(
        "--ranker-critic-checkpoint",
        type=Path,
        default=None,
        help="the frozen labor critic checkpoint the advantage head was fitted on",
    )
    parser.add_argument("--ranker-top", type=int, default=4)
    parser.add_argument(
        "--ranker-full-fraction",
        type=float,
        default=0.2,
        help="keyed share of roots still fully evaluated with the ranker on (regret measurement)",
    )
    parser.add_argument(
        "--control-fraction",
        type=float,
        default=0.25,
        help="keyed share of roots that add the evaluator controls (all-idle, delayed-benefit "
        "Farmer adoption, bookkeeping twin of H12), carried to the final horizon",
    )
    parser.add_argument(
        "--plan-candidates",
        type=int,
        default=2,
        help="T2 plans per root (diagnostic): each adds a plan row and its paired first-step "
        "control row, both counted in --candidates; 0 disables",
    )
    parser.add_argument(
        "--plan-fraction",
        type=float,
        default=1.0,
        help="keyed share of roots that carry the T2 plans (default 1.0: every root); the "
        "single-step budget M - 2N is the same on every root",
    )
    parser.add_argument(
        "--plan-steps",
        type=int,
        default=3,
        help="P: decisions a plan holds its course before H12 continues",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted output with the same source record: verified world "
        "shards are kept, partial worlds are redone into fresh shards",
    )
    parser.add_argument(
        "--checkpoint-keep",
        type=int,
        default=2,
        help="decision-boundary checkpoints kept per unfinished world (labor_search_resume)",
    )
    parser.add_argument(
        "--successor-rows",
        action="store_true",
        help="also export every candidate's successor critic-input rows (after decision 1 and "
        "at the last horizon, per stream) and focal status (candidate payload export v2); "
        "search decisions and returns are unchanged",
    )
    parser.add_argument("--no-verify-restore", action="store_true")
    parser.add_argument(
        "--held-out-test",
        action="store_true",
        help="search the reserved held-out worlds 90154-90159 only; shards are tagged held-out "
        "test and every training loader refuses them",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development only: dirty learner identity (recorded as non-production)",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def validate_search_worlds(worlds: Sequence[int], *, held_out_test: bool = False) -> list[int]:
    """Fitting worlds without the reserved 90154-90159, or (held-out test) only those."""
    from .labor_bc import validate_fitting_worlds

    worlds = validate_fitting_worlds(
        worlds, "labor search fitting worlds", held_out_test=held_out_test
    )
    outside = [w for w in worlds if w not in FITTING_WORLDS]
    if outside:
        raise ValueError(
            f"labor search labels use fitting worlds 90100-90159 only, got {outside[:5]}"
        )
    return worlds


def main(argv: Sequence[str] | None = None) -> int:
    from .bridge import DEFAULT_BRIDGE
    from .labor_bridge import LaborModel9Bridge
    from .labor_personality import LaborPersonalityConditioning
    from .labor_teacher_data import load_normalizer, normalizer_digest
    from .protocol import seed_list
    from .provenance import ProvenanceError, checkpoint_hash, code_commit

    args = argument_parser().parse_args(argv)
    worlds = validate_search_worlds(seed_list(args.worlds), held_out_test=args.held_out_test)
    scenario = None
    if (args.source == "scenario") != (args.scenario_roots is not None):
        raise ValueError("--source scenario and --scenario-roots go together")
    if args.scenario_roots is not None:
        from .labor_scenarios import load_scenario_roots

        scenario = load_scenario_roots(args.scenario_roots)
        expected = "held-out-test" if args.held_out_test else "fitting"
        if scenario["partition"] != expected:
            raise ValueError(
                f"scenario roots of partition {scenario['partition']!r} cannot be searched as "
                f"{expected} (calibration roots are V calibration only)"
            )
    if (args.advantage_ranker is None) != (args.ranker_critic_checkpoint is None):
        raise ValueError("--advantage-ranker and --ranker-critic-checkpoint go together")
    if (args.out / "dataset.json").is_file():
        from .labor_teacher_data import verify_dataset

        verify_dataset(args.out)
        print(json.dumps({"dataset": str(args.out), "status": "already-published"}))
        return 0
    if args.out.exists() and not args.resume:
        raise ProvenanceError(
            "labor search output exists; pass --resume or regenerate into a new directory"
        )
    config = SearchConfig(
        candidates=args.candidates,
        actor_samples=args.actor_samples if args.actor_checkpoint else 0,
        horizons=tuple(int(x) for x in args.horizons.split(",")),
        keep=tuple(x.strip() for x in args.keep.split(",")),
        personalities=_parse_personalities(args.personalities),
        root_personality=not args.no_root_personality,
        margin=args.margin,
        se_multiplier=args.se_multiplier,
        reeval_streams=args.reeval_streams,
        dither_probability=args.dither_probability,
        dither_fraction=args.dither_fraction,
        max_resolves=args.max_resolves,
        focal_per_root=args.focal_per_root,
        movement_candidates=args.movement_candidates,
        reeval_scope=args.reeval_scope,
        ranker=args.advantage_ranker is not None,
        ranker_top=args.ranker_top,
        ranker_full_fraction=args.ranker_full_fraction,
        control_fraction=args.control_fraction,
        plan_candidates=args.plan_candidates,
        plan_steps=args.plan_steps,
        plan_fraction=args.plan_fraction,
        successor_rows=bool(args.successor_rows),
    )
    reset = LaborResetOptions(
        players=args.players,
        movement=True,
        width=args.width,
        height=args.height,
        separation=args.separation,
    )
    scenario_by_world: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if scenario is not None:
        if dict(scenario["reset_options"]) != reset.state_dict():
            raise ValueError(
                "scenario roots were harvested with different reset options "
                f"{scenario['reset_options']}; pass the same --players/--width/--height/"
                "--separation"
            )
        for root in scenario["roots"]:
            scenario_by_world[int(root["world"])].append(root)
        worlds = [w for w in worlds if w in scenario_by_world]
        if not worlds:
            raise ValueError("no scenario roots in the requested worlds")
    conditioning = LaborPersonalityConditioning(
        args.personality, args.personality_key, args.personality_drift
    )
    normalizer = load_normalizer(args.normalizer_checkpoint)
    norm_digest = normalizer_digest(normalizer)
    if (
        args.actor_checkpoint is not None
        and normalizer_digest(load_normalizer(args.actor_checkpoint)) != norm_digest
    ):
        raise ValueError("actor checkpoint normalizer differs from --normalizer-checkpoint")
    bridge_path = Path(os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE", str(DEFAULT_BRIDGE)))
    bridge_sha = checkpoint_hash(bridge_path)
    options = {
        "worlds": worlds,
        "turns": int(args.turns),
        "first_turn": int(args.first_turn),
        "roots_per_world": int(args.roots_per_world),
        "root_key": int(args.root_key),
        "source": args.source,
        "reset_options": reset.state_dict(),
        "personality_conditioning": conditioning.state_dict(),
        "normalizer_digest": norm_digest,
        "normalizer_checkpoint_sha256": checkpoint_hash(args.normalizer_checkpoint),
        "actor_checkpoint_sha256": checkpoint_hash(args.actor_checkpoint)
        if args.actor_checkpoint
        else None,
        "search": config.state_dict(),
        "candidate_payload_schema": payload_module.CANDIDATE_PAYLOAD_SCHEMA,
        "candidate_export_schema": payload_module.export_schema(config.successor_rows),
        "advantage_ranker_sha256": checkpoint_hash(args.advantage_ranker)
        if args.advantage_ranker
        else None,
        "ranker_critic_checkpoint_sha256": checkpoint_hash(args.ranker_critic_checkpoint)
        if args.ranker_critic_checkpoint
        else None,
        "continuation": "h12 (stream 0, deterministic); h12-dither (re-evaluation streams)",
        "partition": partition_for(args.held_out_test),
    }
    if scenario is not None:
        options["scenario_roots_digest"] = scenario["roots_digest"]
        options["scenario_source_run"] = scenario["source_run"]
    with LaborModel9Bridge() as probe:
        source = search_source_record(
            options, probe.schema, bridge_sha, production=not args.allow_dirty
        )
    jobs = [
        {
            "world": world,
            "key": int(args.root_key),
            "turns": (
                max(int(r["decision_index"]) for r in scenario_by_world[world]) + 1
                if scenario is not None
                else int(args.turns)
            ),
            "scenario_roots": scenario_by_world.get(world, []),
            "first_turn": int(args.first_turn),
            "roots": int(args.roots_per_world),
            "source": args.source,
            "config": config.state_dict(),
            "reset_options": reset.state_dict(),
            "normalizer_checkpoint": str(args.normalizer_checkpoint),
            "normalizer_digest": norm_digest,
            "actor_checkpoint": str(args.actor_checkpoint) if args.actor_checkpoint else None,
            "personality_conditioning": conditioning.state_dict(),
            "verify_restore": not args.no_verify_restore,
            "source_run": source["source_run"],
            "out": str(args.out),
            "partition": options["partition"],
            "ranker_checkpoint": str(args.advantage_ranker) if args.advantage_ranker else None,
            "ranker_critic_checkpoint": (
                str(args.ranker_critic_checkpoint) if args.ranker_critic_checkpoint else None
            ),
            "ranker_checkpoint_sha256": options["advantage_ranker_sha256"],
            # Execution only (not in the source record): where this world checkpoints and
            # whether an existing checkpoint is resumed.
            "checkpoint_dir": str(
                Path(args.out) / "_work" / f"world-{int(world)}"
            ),
            "resume": bool(args.resume),
            "checkpoint_keep": int(args.checkpoint_keep),
        }
        for world in worlds
    ]

    def unchanged() -> None:
        if checkpoint_hash(bridge_path) != bridge_sha or code_commit() != source["learner_commit"]:
            raise ProvenanceError(
                "labor search producer changed during generation; dataset not published"
            )

    report = run_search_jobs(
        args.out, source, jobs, workers=int(args.jobs), resume=args.resume, before_publish=unchanged
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "roots",
                    "per_personality",
                    "native_resolves_per_root",
                    "wall_seconds_per_root",
                    "elapsed_seconds",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
