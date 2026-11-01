"""Live heuristic labor behavioral cloning (BC) pretraining for Model 9 actor.

Reconstructs native heuristic specialty and activity simplexes from shadow
simulation steps with economic diagnostics, and trains the LaborActor model
via masked supervised behavioral cloning before RL rollouts.

G1 corrections (brief v1.9 sections 4.2, 5.1 and 5.4):

* label conversion preserves unused role time as explicit idle, never divides
  work by the extracted work total, and never turns missing, unknown or
  over-budget work into residual Gatherers; such rows are typed unavailable;
* Herder staffing comes from the native assigned residents and planned
  worker-years, never from tool-covered service;
* every head carries an explicit label validity, resolved against the
  original pre-action catalog; invalid movement and inactive/default trade
  labels contribute no loss;
* rows whose initial recommendation the native validator rejected are
  quarantined with their violation evidence, never trained as demonstrations;
* allocation cross-entropies use a stable masked log-softmax;
* collection is manifest driven (movement on, explicit worlds, configurable
  players/size) and every row carries world/camp/turn/attempt identity with a
  world-balanced weight.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import contracts
from .bridge import Model9Bridge
from .labor_bridge import (
    LaborAttempt,
    LaborCatalog,
    LaborModel9Bridge,
    LaborPolicySample,
    SIMPLEX_TOLERANCE,
    _simplex,
)
from .labor_model import (
    LABOR_COSTS,
    LABOR_OBJECTIVE_REFERENCE_SCALES,
    LABOR_OBJECTIVES,
    LaborActor,
    LaborCostCritics,
    LaborPersonality,
    LaborTaskCritic,
    _masked_softmax,
    _masks,
    critic_privileged_batch,
    labor_policy_tensors,
)
from .labor_device import cpu_copy, module_device, to_device
from .labor_personality import personality_for_world
from .labor_privileged import actor_inputs, privileged_row_tensors, require_critic_training_worlds
from .labor_worlds import (
    require_held_out_test_worlds,
    require_training_record,
    require_training_worlds,
)
from .labor_ppo import (
    labor_cost_value_losses,
    labor_task_value_losses,
    RetainedLaborAction,
    execute_labor_decision,
    labor_cost_value_loss,
    labor_task_value_loss,
)
from .protocol import scan
from .protocol import seeds as checked_seeds

LABOR_BC_CONTRACT = "model9-heuristic-labor-bc-v2"
LABOR_LABEL_CONTRACT = "model9-heuristic-labor-label-v2"
LABOR_CRITIC_CONTRACT = "model9-heuristic-critic-pretraining-v2"
LABEL_TOLERANCE = 1e-6
DEVELOPMENT_WORLDS = range(2000, 2060)
SEALED_WORLDS = range(3000, 3060)
# Discount time constant in physical YEARS (equal to contracts.HORIZON_YEARS):
# gamma = exp(-elapsed_years / 400). It was historically misnamed "days"; the
# value and the elapsed field it divides were always years (20 per turn).
CRITIC_DISCOUNT_YEARS = 400.0
CRITIC_DISCOUNT_DAYS = CRITIC_DISCOUNT_YEARS  # deprecated alias; the unit is years
# Per-row elapsed physical years of the recorded transition (contracts.TURN_YEARS
# per native turn). Shards written before 2026-10-26 store the same values under
# the misnamed legacy key; loaders rename it without conversion.
ELAPSED_YEARS_FIELD = "transition_elapsed_years"
LEGACY_ELAPSED_DAYS_FIELD = "transition_elapsed_days"


def _get(mapping: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    """Case-tolerant field lookup for bridge and diagnostic mappings."""
    if not isinstance(mapping, Mapping):
        return default
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
        for alt in (k.lower(), k[:1].upper() + k[1:], k[:1].lower() + k[1:]):
            if alt in mapping and mapping[alt] is not None:
                return mapping[alt]
    return default


def _num(mapping: Mapping[str, Any] | None, key: str) -> float | None:
    """An explicit finite numeric field, or None when absent/null.

    Unlike ``_get(...) or fallback`` this keeps a recorded zero distinct from a
    missing value, so a legitimate zero is never replaced by another stage's
    grant or coverage.
    """
    if not isinstance(mapping, Mapping):
        return None
    for alt in (key, key[:1].upper() + key[1:], key[:1].lower() + key[1:]):
        if alt in mapping:
            value = mapping[alt]
            if value is None or isinstance(value, bool):
                return None
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) else None
    return None


def _one_row(values: Mapping[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    return {
        key: tensor[row : row + 1].detach().cpu()
        for key, tensor in values.items()
        if isinstance(tensor, torch.Tensor)
    }


def _concat(rows: Sequence[Mapping[str, torch.Tensor]], label: str) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError(f"{label} has no rows to concatenate")
    keys = set(rows[0])
    for item in rows[1:]:
        if set(item) != keys:
            raise ValueError(f"{label} row keys differ")
    return {key: torch.cat([row[key] for row in rows], dim=0) for key in sorted(keys)}


# ---------------------------------------------------------------------------
# Manifest-driven reset options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaborResetOptions:
    """Native reset options shared by labor BC, critic and anchor collection.

    Movement is on by default: a movement-disabled reset is a named
    mechanical fixture only. ``players`` sets the generated opening's player
    count; ``width``/``height``/``separation`` optionally enlarge the map
    (absent keeps the native Small training world).
    """

    players: int = 8
    movement: bool = True
    withheld: int = 0
    width: int | None = None
    height: int | None = None
    separation: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.players, bool) or int(self.players) < 1:
            raise ValueError("reset players must be a positive integer")
        if not isinstance(self.movement, bool):
            raise ValueError("reset movement must be boolean")
        for name in ("width", "height", "separation"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or int(value) < 1):
                raise ValueError(f"reset {name} must be positive when present")

    @property
    def size(self) -> tuple[int, int, int] | None:
        if self.width is None and self.height is None and self.separation is None:
            return None
        return (int(self.width or 0), int(self.height or 0), int(self.separation or 0))

    def reset_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "players": int(self.players),
            "movement": bool(self.movement),
            "withheld": int(self.withheld),
        }
        if self.size is not None:
            kwargs["size"] = self.size
        return kwargs

    def state_dict(self) -> dict[str, Any]:
        return {
            "players": int(self.players),
            "movement": bool(self.movement),
            "withheld": int(self.withheld),
            "width": self.width,
            "height": self.height,
            "separation": self.separation,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> LaborResetOptions:
        if value is None:
            return cls()
        return cls(**{key: value[key] for key in cls().state_dict() if key in value})


def validate_fitting_worlds(
    world_seeds: Sequence[int], context: str, *, held_out_test: bool = False
) -> list[int]:
    """Fitting worlds exclude sealed confirmation, development-only and reserved seeds.

    The reserved held-out worlds 90154-90159 (``labor_worlds``) are refused
    unless ``held_out_test`` is set, in which case *only* reserved worlds are
    accepted and the caller must tag its outputs as held-out test data.
    """
    worlds = checked_seeds(list(world_seeds), context)
    scan({"worldSeeds": worlds}, context)
    if len(set(worlds)) != len(worlds):
        raise ValueError(f"{context} repeats a world seed")
    for seed in worlds:
        if seed in SEALED_WORLDS:
            raise ValueError(f"{context}: sealed confirmation world {seed} is not fitting data")
        if seed in DEVELOPMENT_WORLDS:
            raise ValueError(f"{context}: development world {seed} is development-only")
    if held_out_test:
        return require_held_out_test_worlds(worlds, context)
    return require_training_worlds(worlds, context)


# ---------------------------------------------------------------------------
# Native heuristic -> action-12 label conversion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeuristicLaborLabel:
    """One converted heuristic recommendation with per-head validity.

    ``specialty``/``activity`` are the faithful label simplexes (role idle
    retains unused worker-time). ``action`` is always an executable schema-12
    payload; when a label is unavailable it is only a drive action used to
    advance the world and must not be trained as a demonstration.
    """

    specialty: np.ndarray
    activity: np.ndarray
    action: dict[str, Any]
    specialty_counts: np.ndarray
    planned_worker_years: np.ndarray
    specialty_valid: bool
    activity_role_valid: np.ndarray
    movement_code: int
    movement_valid: bool
    fission_valid: bool
    trade_gate_valid: np.ndarray
    trade_value_active: np.ndarray
    trade_share_active: bool
    over_budget: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def labor_valid(self) -> bool:
        return bool(self.specialty_valid)

    @property
    def full_action_valid(self) -> bool:
        return bool(
            self.specialty_valid
            and self.movement_valid
            and self.fission_valid
            and bool(np.all(self.trade_gate_valid))
        )


def _movement_legal(raw: int, observation: Mapping[str, Any] | None) -> bool:
    """Resolve a movement code against the original pre-action masks."""
    if raw == -1:
        return True
    if not isinstance(observation, Mapping):
        return False
    m_mask = observation.get("movementMask") or []
    r_mask = observation.get("regionMovementMask") or []
    if 0 <= raw < contracts.MOVEMENT_SLOTS - 1:
        return raw + 1 < len(m_mask) and bool(m_mask[raw + 1])
    if raw == contracts.KEEP_INTENT:
        return len(r_mask) > 1 and bool(r_mask[1])
    if raw >= contracts.REGION_BASE:
        index = 2 + raw - contracts.REGION_BASE
        return index < len(r_mask) and bool(r_mask[index])
    return False  # includes the forbidden native sentinel -2


def convert_heuristic_labor_label(
    catalog: LaborCatalog,
    village_heur: Mapping[str, Any],
    econ_diag: Mapping[str, Any] | None,
    *,
    pre_observation: Mapping[str, Any] | None = None,
) -> HeuristicLaborLabel:
    """Convert one native heuristic decision into an action-12 label.

    Specialty shares are integer staffed residents over P. For a staffed role
    the activity share of each activity is its worker-years over
    ``n_r * dt`` and role idle is the nonnegative residual. Unstaffed roles
    emit canonical role idle and are excluded from the activity loss. Native
    residual fallback gathering stays Gatherer work; missing, unmapped or
    over-budget work makes the labor label unavailable instead. So does a role
    whose work is within the staffing tolerance but whose shares overrun the
    simplex tolerance (``activity-simplex-tolerance-overrun``): its row cannot
    be a simplex, and is typed unavailable rather than repaired.
    """

    reasons: list[str] = []
    population = int(catalog.population)
    years = float(catalog.turn_years)
    if population <= 0:
        reasons.append("zero-population")
    if not years > 0.0:
        reasons.append("nonpositive-turn-years")
        years = 1.0
    spec_mask = np.asarray(catalog.masks.specialty, dtype=bool)
    act_mask = np.asarray(catalog.masks.activity, dtype=bool)
    specialties = len(catalog.specialty_rows)
    roles, activities = act_mask.shape
    spec_row = {row["id"]: int(row["row"]) for row in catalog.specialty_rows}

    def role_of(name: str) -> int | None:
        row = spec_row.get(name)
        return None if row is None or row <= 0 else row - 1

    def legal(role: int | None) -> bool:
        return role is not None and bool(spec_mask[role + 1])

    def role_activity(role: int | None, activity_id: str) -> int | None:
        """The legal activity row of ``role`` with the stable registry ID."""
        if role is None:
            return None
        for row in catalog.activity_rows:
            if int(row["specialtyRow"]) == role and row.get("activityId") == activity_id:
                index = int(row["activityRow"])
                return index if act_mask[role, index] else None
        return None

    counts = np.zeros(specialties, dtype=np.int64)
    work = np.zeros((roles, activities), dtype=np.float64)

    def place(role: int, activity: int | None, worker_years: float, name: str) -> None:
        if not worker_years > LABEL_TOLERANCE:
            return
        if activity is None:
            reasons.append(f"{name}-activity-unavailable")
            return
        work[role, activity] += worker_years

    if not isinstance(econ_diag, Mapping) or not econ_diag:
        reasons.append("economic-diagnostics-unavailable")
        econ_diag = {}
    domains = _get(econ_diag, "domains", default={})
    domains = domains if isinstance(domains, Mapping) else {}
    unavailable = _get(econ_diag, "unavailable", default={})
    unavailable = unavailable if isinstance(unavailable, Mapping) else {}

    # 1. Farmer: assigned farmer worker-years split into operation and expansion.
    farmer = role_of("Farmer")
    cultivation = domains.get("cultivation")
    assigned_farmer = _num(cultivation, "assignedFarmerYears")
    if assigned_farmer is None:
        reason = unavailable.get("domains.cultivation.assignedFarmerYears")
        if legal(farmer) and reason != "no-active-field-work":
            reasons.append("farmer-work-unavailable")
        assigned_farmer = 0.0
    if assigned_farmer > LABEL_TOLERANCE:
        if not legal(farmer):
            reasons.append("farmer-work-on-illegal-specialty")
        else:
            expansion = _num(cultivation, "expansionWorkerYears")
            if expansion is None:
                for funnel in _get(cultivation, "funnels", default=[]) or []:
                    if _get(funnel, "metric") == "fieldDevelopment":
                        people = _num(funnel, "executedWork")
                        if people is None:
                            people = _num(funnel, "compilerGrant")
                        if people is not None:
                            expansion = people * years
                        break
            if expansion is None:
                reasons.append("field-expansion-work-unavailable")
                expansion = 0.0
            if expansion > assigned_farmer + LABEL_TOLERANCE:
                reasons.append("field-expansion-exceeds-assigned-farmer-work")
            assert farmer is not None
            counts[farmer + 1] = math.ceil(assigned_farmer / years - LABEL_TOLERANCE)
            place(
                farmer,
                role_activity(farmer, "field.operation"),
                max(0.0, assigned_farmer - expansion),
                "field-operation",
            )
            place(farmer, role_activity(farmer, "field.expansion"), expansion, "field-expansion")

    # 2. Herder: the native assigned residents and their planned worker-years.
    herder = role_of("Herder")
    herding = domains.get("herding")
    assigned_herders = _num(herding, "assignedHerderResidents")
    if legal(herder):
        management = _num(herding, "managementHerderWorkerYears")
        development = _num(herding, "developmentHerderWorkerYears")
        if assigned_herders is None or management is None or development is None:
            reasons.append("herder-staffing-unavailable")
        else:
            whole = int(round(assigned_herders))
            if whole < 0 or abs(whole - assigned_herders) > LABEL_TOLERANCE:
                reasons.append("herder-staffing-not-whole-residents")
            assert herder is not None
            counts[herder + 1] = max(0, whole)
            place(herder, role_activity(herder, "herd.management"), management, "herd-management")
            place(herder, role_activity(herder, "herd.expansion"), development, "herd-expansion")
    elif assigned_herders is not None and assigned_herders > LABEL_TOLERANCE:
        reasons.append("herder-staffing-on-illegal-specialty")

    # 3. Fisher: harvest and development staffing.
    fisher = role_of("Fisher")
    if legal(fisher):
        assert fisher is not None
        harvest_people: int | None = None
        development_people: int | None = None
        observation = village_heur.get("observation")
        if isinstance(observation, Mapping):
            pop_after = max(1, int(village_heur.get("population", catalog.population)))
            core = observation.get("core", []) or []
            present = observation.get("corePresent", []) or []

            def staffed(index: int) -> int | None:
                if len(core) > index and (len(present) <= index or present[index]):
                    return max(0, round(float(core[index]) * pop_after))
                return None

            harvest_people, development_people = staffed(202), staffed(206)
        else:
            requests = _get(econ_diag, "requests")
            if isinstance(requests, list):
                harvest_people = max(
                    0,
                    round(
                        sum(
                            float(_num(r, "compilerGrant") or 0.0)
                            for r in requests
                            if _get(r, "role") == "Fisher"
                        )
                    ),
                )
                development_people = 0
        if harvest_people is None or development_people is None:
            reasons.append("fisher-staffing-unavailable")
        else:
            counts[fisher + 1] = harvest_people + development_people
            place(fisher, role_activity(fisher, "fishing.harvest"), harvest_people * years, "fishing-harvest")
            place(
                fisher,
                role_activity(fisher, "fishing.development"),
                development_people * years,
                "fishing-development",
            )

    # 4. Craft goods and 5. raw materials: worker-years mapped to specialist roles.
    specialist_work = np.zeros((roles, activities), dtype=np.float64)
    gatherer = role_of("Gatherer")
    gatherer_material = np.zeros(activities, dtype=np.float64)

    def normalized(text: Any) -> str:
        return str(text or "").lower().replace("-", "").replace("_", "").replace(" ", "")

    for item in _get(econ_diag, "goods", default=[]) or []:
        name = _get(item, "good")
        if not name or name == "none":
            continue
        matching = [
            row
            for row in catalog.activity_rows
            if normalized(row.get("outputGood")) == normalized(name)
            and spec_mask[int(row["specialtyRow"]) + 1]
            and act_mask[int(row["specialtyRow"]), int(row["activityRow"])]
        ]
        target = None
        if matching:
            specialist = [r for r in matching if r.get("specialistOnly")]
            crafted = [r for r in matching if r.get("specialtyId") != "Craftsperson"]
            target = (specialist or crafted or matching)[0]
        worker_years = _num(item, "workerYears")
        if worker_years is None or worker_years <= LABEL_TOLERANCE:
            ceiling = _num(item, "plannedOutputCeiling")
            if ceiling is not None and ceiling > LABEL_TOLERANCE:
                features = target.get("features", []) if target is not None else []
                rate = float(features[13]) if len(features) > 13 else 0.0
                if target is None or not rate > 1e-6:
                    reasons.append(f"craft-work-unknown:{name}")
                    continue
                worker_years = ceiling / rate
            elif worker_years is None:
                reasons.append(f"craft-work-unavailable:{name}")
                continue
            else:
                continue
        if target is None:
            reasons.append(f"craft-work-unmapped:{name}")
            continue
        specialist_work[int(target["specialtyRow"]), int(target["activityRow"])] += worker_years

    material_flags = {"Stone": "outputsStone", "Clay": "outputsClay", "Wood": "outputsWood"}
    for item in _get(econ_diag, "materials", default=[]) or []:
        name = _get(item, "material")
        worker_years = _num(item, "workerYears")
        if not name:
            continue
        if worker_years is None:
            reasons.append(f"material-work-unavailable:{name}")
            continue
        if worker_years <= LABEL_TOLERANCE:
            continue
        flag = material_flags.get(str(name))
        if flag is None:
            reasons.append(f"material-work-unmapped:{name}")
            continue
        matching = [
            row
            for row in catalog.activity_rows
            if row.get(flag)
            and spec_mask[int(row["specialtyRow"]) + 1]
            and act_mask[int(row["specialtyRow"]), int(row["activityRow"])]
        ]
        if not matching:
            reasons.append(f"material-work-unmapped:{name}")
            continue
        specialist = [r for r in matching if r.get("specialistOnly")]
        non_gatherer = [r for r in matching if int(r["specialtyRow"]) != gatherer]
        target = (specialist or non_gatherer or matching)[0]
        role, activity = int(target["specialtyRow"]), int(target["activityRow"])
        if role == gatherer:
            gatherer_material[activity] += worker_years
        else:
            specialist_work[role, activity] += worker_years

    for role in range(roles):
        total = float(specialist_work[role].sum())
        if total > LABEL_TOLERANCE:
            counts[role + 1] += math.ceil(total / years - LABEL_TOLERANCE)
            work[role] += specialist_work[role]

    # 6. Population budget: residual residents are native fallback Gatherers.
    idle_spec = spec_row.get("idle", 0)
    non_gatherers = int(counts.sum())
    over_budget = non_gatherers > max(0, population)
    if over_budget:
        reasons.append("over-budget-staffing")
    residual = max(0, population - non_gatherers)
    if legal(gatherer):
        assert gatherer is not None
        counts[gatherer + 1] = residual
        material = float(gatherer_material.sum())
        capacity = residual * years
        if material > capacity + LABEL_TOLERANCE:
            reasons.append("gatherer-material-over-budget")
            over_budget = True
        work[gatherer] += gatherer_material
        food = role_activity(gatherer, "food.gather")
        place(gatherer, food, max(0.0, capacity - material), "gathering")
    else:
        if gatherer_material.sum() > LABEL_TOLERANCE:
            reasons.append("gatherer-material-on-illegal-specialty")
        counts[idle_spec] += residual
    counts[~spec_mask] = 0

    # 7. Activity shares over available worker-time (role idle keeps the residual).
    label_activity = np.zeros((roles, activities), dtype=np.float64)
    drive_activity = np.zeros((roles, activities), dtype=np.float64)
    role_valid = np.zeros(roles, dtype=bool)
    for role in range(roles):
        staffed = int(counts[role + 1]) if spec_mask[role + 1] else 0
        if staffed <= 0:
            if work[role].sum() > LABEL_TOLERANCE:
                reasons.append(f"work-without-staff:role{role}")
            label_activity[role, 0] = 1.0
            drive_activity[role, 0] = 1.0
            continue
        vec = work[role].copy()
        if np.any(vec[~act_mask[role]] > LABEL_TOLERANCE):
            reasons.append(f"work-on-illegal-activity:role{role}")
        vec[~act_mask[role]] = 0.0
        shares = vec / (staffed * years)
        used = float(shares.sum())
        if used > 1.0 + LABEL_TOLERANCE:
            # Over-budget work is flagged, never rescaled into a training target.
            reasons.append(f"activity-over-budget:role{role}")
            over_budget = True
            shares = shares / used
        else:
            filled = shares.copy()
            filled[0] = max(0.0, 1.0 - float(filled[1:].sum()))
            if (
                used > 1.0
                and act_mask[role].any()
                and not math.isclose(
                    math.fsum(np.clip(filled, 0.0, 1.0)), 1.0, rel_tol=0.0, abs_tol=SIMPLEX_TOLERANCE
                )
            ):
                # Work inside the staffing tolerance (LABEL_TOLERANCE) but past the
                # simplex tolerance (SIMPLEX_TOLERANCE): the residual idle share cannot
                # absorb the overrun, so the converted row is not a simplex. Typed
                # exactly like over-budget work -- the labor label is unavailable and
                # only the executable drive is rescaled -- never repaired into a target.
                reasons.append(f"activity-simplex-tolerance-overrun:role{role}")
                over_budget = True
                shares = shares / used
            else:
                shares = filled
                role_valid[role] = True
        label_activity[role] = shares
        drive_activity[role] = shares

    def to_simplex(matrix: np.ndarray) -> list[tuple[float, ...]]:
        return [
            _simplex(tuple(np.clip(matrix[r], 0.0, 1.0)), tuple(act_mask[r]), f"activity[{r}]")
            for r in range(roles)
        ]

    # 8. Specialty shares n/P (label) and an executable drive allocation.
    spec_label = np.zeros(specialties, dtype=np.float64)
    if population > 0 and not over_budget:
        spec_label = counts / float(population)
    drive_counts = counts.astype(np.float64).copy()
    if over_budget or population <= 0:
        drive_counts = _drive_counts(counts, spec_mask, population, idle_spec, gatherer)
    drive_spec = drive_counts / max(1.0, drive_counts.sum())
    if population > 0 and not over_budget:
        spec_label_tuple = _simplex(tuple(spec_label), tuple(spec_mask), "specialty")
    else:
        spec_label_tuple = _simplex(tuple(drive_spec), tuple(spec_mask), "specialty")
    drive_spec_tuple = _simplex(tuple(drive_spec), tuple(spec_mask), "specialty")
    label_activity_list = to_simplex(label_activity)
    drive_activity_list = to_simplex(drive_activity)

    # 9. Nonlabor heads against the original pre-action catalog.
    teacher = village_heur.get("teacher") if isinstance(village_heur, Mapping) else None
    teacher = teacher if isinstance(teacher, Mapping) else None
    if teacher is None:
        reasons.append("teacher-unavailable")
    observation = pre_observation if pre_observation is not None else (
        village_heur.get("observation") if isinstance(village_heur, Mapping) else None
    )
    raw_movement = teacher.get("movement") if teacher is not None else None
    movement_valid = (
        teacher is not None
        and isinstance(raw_movement, int)
        and not isinstance(raw_movement, bool)
        and teacher.get("movementLabel", True) is True
        and _movement_legal(int(raw_movement), observation)
    )
    if teacher is not None and not movement_valid:
        reasons.append(f"movement-label-unavailable:{raw_movement}")
    movement = int(raw_movement) if movement_valid else -1

    raw_fission = teacher.get("fission") if teacher is not None else None
    fission_valid = (
        isinstance(raw_fission, int) and not isinstance(raw_fission, bool) and 0 <= raw_fission <= 2
    )
    fission = int(raw_fission) if fission_valid else 0
    if teacher is not None and not fission_valid:
        reasons.append("fission-label-unavailable")

    tv_raw = teacher.get("tradeValue") if teacher is not None else None
    tg_raw = teacher.get("tradeGate") if teacher is not None else None
    ts_raw = teacher.get("tradeShare") if teacher is not None else None
    trade_labelled = (
        teacher is not None
        and teacher.get("tradeLabel", True) is True
        and isinstance(tv_raw, Sequence)
        and len(tv_raw) == len(contracts.TRADE_GROUPS)
        and isinstance(tg_raw, Sequence)
        and len(tg_raw) == len(contracts.TRADE_GROUPS)
        and all(isinstance(x, (int, float)) and 0 <= int(x) <= 2 and int(x) == x for x in tg_raw)
        and all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in tv_raw)
    )
    groups = len(contracts.TRADE_GROUPS)
    if trade_labelled:
        trade_value = [
            float(np.clip(float(x), contracts.TRADE_VALUE_RANGE[0], contracts.TRADE_VALUE_RANGE[1]))
            for x in tv_raw
        ]
        trade_gate = [int(x) for x in tg_raw]
        gate_valid = np.ones(groups, dtype=bool)
        value_active = np.asarray([gate > 0 for gate in trade_gate], dtype=bool)
    else:
        if teacher is not None:
            reasons.append("trade-label-unavailable")
        trade_value = [1.0] * groups
        trade_gate = [1] * groups
        gate_valid = np.zeros(groups, dtype=bool)
        value_active = np.zeros(groups, dtype=bool)
    share_valid = (
        trade_labelled
        and isinstance(ts_raw, (int, float))
        and not isinstance(ts_raw, bool)
        and math.isfinite(float(ts_raw))
    )
    trade_share = float(np.clip(float(ts_raw) if share_valid else 0.5, 1e-4, 1.0 - 1e-4))
    share_active = bool(share_valid and value_active.any())

    labor_valid = not reasons_block_labor(reasons) and population > 0 and not over_budget
    drive_sample = LaborPolicySample(drive_spec_tuple, tuple(drive_activity_list))
    action = {
        "schemaVersion": 12,
        "movement": movement,
        "fission": fission,
        "tradeValue": trade_value,
        "tradeGate": trade_gate,
        "tradeShare": trade_share,
        **drive_sample.payload(),
    }
    return HeuristicLaborLabel(
        specialty=np.asarray(spec_label_tuple, dtype=np.float64),
        activity=np.asarray(label_activity_list, dtype=np.float64),
        action=action,
        specialty_counts=counts.copy(),
        planned_worker_years=work.copy(),
        specialty_valid=bool(labor_valid),
        activity_role_valid=role_valid & bool(labor_valid),
        movement_code=movement,
        movement_valid=bool(movement_valid),
        fission_valid=bool(fission_valid),
        trade_gate_valid=gate_valid,
        trade_value_active=value_active & gate_valid,
        trade_share_active=share_active,
        over_budget=bool(over_budget),
        reasons=tuple(reasons),
    )


_NONLABOR_REASONS = ("movement-label-unavailable", "fission-label-unavailable", "trade-label-unavailable")


def reasons_block_labor(reasons: Sequence[str]) -> bool:
    """Whether any recorded reason makes the labor (specialty/activity) label unavailable."""
    return any(not reason.startswith(_NONLABOR_REASONS) for reason in reasons)


def _drive_counts(
    counts: np.ndarray,
    spec_mask: np.ndarray,
    population: int,
    idle_spec: int,
    gatherer: int | None,
) -> np.ndarray:
    """Executable drive staffing for an over-budget label (never a training target)."""
    result = counts.astype(np.float64).copy()
    population = max(1, population)
    if gatherer is not None:
        result[gatherer + 1] = 0.0
    result[idle_spec] = 0.0
    total = result.sum()
    if total > population:
        result = np.floor(result * (population / total))
    rest = population - result.sum()
    if gatherer is not None and spec_mask[gatherer + 1]:
        result[gatherer + 1] += rest
    else:
        result[idle_spec] += rest
    result[~spec_mask] = 0.0
    return result


def extract_heuristic_labor_action(
    catalog: LaborCatalog,
    village_heur: Mapping[str, Any],
    econ_diag: Mapping[str, Any] | None,
    *,
    pre_observation: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compatibility wrapper: (specialty label, activity label, executable action).

    Callers that train on the result must use :func:`convert_heuristic_labor_label`
    and honour its validity flags; this wrapper exists for world driving only.
    """
    label = convert_heuristic_labor_label(
        catalog, village_heur, econ_diag, pre_observation=pre_observation
    )
    return label.specialty, label.activity, label.action


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _retained_action(
    label: HeuristicLaborLabel,
    catalog: LaborCatalog,
    movement_mask: torch.Tensor,
) -> RetainedLaborAction:
    action = label.action
    tv = [math.log(float(np.clip(float(x), 0.25, 4.0))) for x in action["tradeValue"]]
    ts = float(np.clip(float(action["tradeShare"]), 1e-4, 1.0 - 1e-4))
    executed = LaborPolicySample.from_payload(
        {
            "laborSpecialtyAllocation": action["laborSpecialtyAllocation"],
            "laborActivityAllocation": action["laborActivityAllocation"],
        },
        catalog.masks,
    )
    return RetainedLaborAction(
        payload=action,
        labor_sample={
            "specialty_allocation": torch.as_tensor(
                executed.specialty_allocation, dtype=torch.float64
            ).unsqueeze(0),
            "activity_allocation": torch.as_tensor(
                executed.activity_allocation, dtype=torch.float64
            ).unsqueeze(0),
            "specialty_latent": torch.zeros((1, 14), dtype=torch.float32),
            "activity_latent": torch.zeros((1, 13, 22), dtype=torch.float32),
            "specialty_mask": torch.tensor([catalog.masks.specialty], dtype=torch.bool),
            "activity_mask": torch.tensor([catalog.masks.activity], dtype=torch.bool),
        },
        behavior_labor_logp=0.0,
        nonlabor_sample={
            "movement_index": torch.tensor([0], dtype=torch.long),
            "movement": torch.tensor([int(action["movement"])], dtype=torch.long),
            "fission": torch.tensor([int(action["fission"])], dtype=torch.long),
            "trade_value_latent": torch.tensor(tv, dtype=torch.float32).unsqueeze(0),
            "trade_gate": torch.tensor(
                [int(x) for x in action["tradeGate"]], dtype=torch.long
            ).unsqueeze(0),
            "trade_share_latent": torch.tensor(
                [math.log(ts / (1.0 - ts))], dtype=torch.float32
            ),
            "movement_mask": movement_mask,
            "movement_codes": torch.tensor([-1], dtype=torch.long),
        },
        behavior_nonlabor_logp=0.0,
    )


def _idle_retry(camp: int, catalog: LaborCatalog, attempt: LaborAttempt) -> RetainedLaborAction:
    del camp, attempt
    idle_spec = np.zeros(14, dtype=np.float64)
    idle_spec[0] = 1.0
    idle_act = np.zeros((13, 22), dtype=np.float64)
    idle_act[:, 0] = 1.0
    safe_sample = LaborPolicySample(tuple(idle_spec), tuple(tuple(r) for r in idle_act))
    return RetainedLaborAction(
        payload=safe_sample.payload(),
        labor_sample={
            "specialty_allocation": torch.as_tensor(idle_spec, dtype=torch.float64).unsqueeze(0),
            "activity_allocation": torch.as_tensor(idle_act, dtype=torch.float64).unsqueeze(0),
            "specialty_latent": torch.zeros((1, 14), dtype=torch.float32),
            "activity_latent": torch.zeros((1, 13, 22), dtype=torch.float32),
            "specialty_mask": torch.tensor([catalog.masks.specialty], dtype=torch.bool),
            "activity_mask": torch.tensor([catalog.masks.activity], dtype=torch.bool),
        },
        behavior_labor_logp=0.0,
        nonlabor_sample=None,
        behavior_nonlabor_logp=None,
    )


def _label_tensors(label: HeuristicLaborLabel) -> dict[str, torch.Tensor]:
    tv = [math.log(float(np.clip(float(x), 0.25, 4.0))) for x in label.action["tradeValue"]]
    ts = float(np.clip(float(label.action["tradeShare"]), 1e-4, 1.0 - 1e-4))
    return {
        "target_specialty": torch.tensor(label.specialty, dtype=torch.float32).unsqueeze(0),
        "target_activity": torch.tensor(label.activity, dtype=torch.float32).unsqueeze(0),
        "target_specialty_counts": torch.tensor(
            label.specialty_counts, dtype=torch.float32
        ).unsqueeze(0),
        "target_planned_worker_years": torch.tensor(
            label.planned_worker_years, dtype=torch.float32
        ).unsqueeze(0),
        "target_movement_code": torch.tensor([int(label.movement_code)], dtype=torch.long),
        "target_fission": torch.tensor([int(label.action["fission"])], dtype=torch.long),
        "target_trade_value_latent": torch.tensor(tv, dtype=torch.float32).unsqueeze(0),
        "target_trade_gate": torch.tensor(
            [int(x) for x in label.action["tradeGate"]], dtype=torch.long
        ).unsqueeze(0),
        "target_trade_share_latent": torch.tensor(
            [math.log(ts / (1.0 - ts))], dtype=torch.float32
        ),
        "label_specialty_valid": torch.tensor([label.specialty_valid], dtype=torch.bool),
        "label_activity_role_valid": torch.tensor(
            label.activity_role_valid, dtype=torch.bool
        ).unsqueeze(0),
        "label_movement_valid": torch.tensor([label.movement_valid], dtype=torch.bool),
        "label_fission_valid": torch.tensor([label.fission_valid], dtype=torch.bool),
        "label_trade_gate_valid": torch.tensor(
            label.trade_gate_valid, dtype=torch.bool
        ).unsqueeze(0),
        "label_trade_value_active": torch.tensor(
            label.trade_value_active, dtype=torch.bool
        ).unsqueeze(0),
        "label_trade_share_active": torch.tensor([label.trade_share_active], dtype=torch.bool),
    }


def shadow_teacher_labels(
    labor_bridge: Any,
    heur_bridge: Any,
    world_seed: int,
    options: LaborResetOptions,
    live: Sequence[Mapping[str, Any]],
    contexts: Sequence[Any],
    snapshot_sink: list[str] | None = None,
) -> tuple[Mapping[str, Any], list[HeuristicLaborLabel]]:
    """The teacher's converted full-action label for every live camp at this exact state.

    Captures the actor-labor bridge boundary, resets the shadow heuristic bridge
    from that snapshot with the same reset rules, takes one native heuristic step
    and converts each camp's recommendation against its pre-action catalog. The
    labor bridge itself is not advanced (one native resolve on the shadow bridge).
    Shared by the teacher collector, the h12 evaluation path and the DAgger
    learner-state collector. ``snapshot_sink`` receives the captured boundary
    snapshot (collection checkpoints reuse it; no extra capture).
    """
    snap = labor_bridge._bridge.capture()
    if snapshot_sink is not None:
        snapshot_sink.append(snap)
    heur_bridge.reset(
        int(world_seed), movement=options.movement, withheld=options.withheld, snapshot=snap
    )
    h_step = heur_bridge.step({})
    h_villages = {int(v["camp"]): v for v in h_step.get("villages", [])}
    econ_by_camp = h_step.get("economicDiagnostics") or {}
    labels = [
        convert_heuristic_labor_label(
            context[0],
            h_villages.get(int(village["camp"]), {}),
            econ_by_camp.get(str(int(village["camp"]))),
            pre_observation=village["observation"],
        )
        for village, context in zip(live, contexts, strict=True)
    ]
    return h_step, labels


# ---------------------------------------------------------------------------
# Return tail at the collection horizon
# ---------------------------------------------------------------------------

# Declared critic-target tail methods. The legacy method zero-fills every
# living cutoff; it is accepted only when a dataset names it explicitly.
RETURN_TAIL_TRUNCATED_ZERO = "truncated-zero-at-collection-horizon"
RETURN_TAIL_RANDOMIZED = "teacher-continuation-randomized-tail-v1"
RETURN_TAIL_ESTIMATOR = (
    "labor_anchor_producer.summarize_labor_path with prefix_terms=0 on one teacher "
    "continuation from the collection-horizon state (gate34 geometric Russian roulette)"
)
# Per-row typed tail target, constant within each (world, camp) trajectory and
# valued at that trajectory's cutoff state (the state after its last row).
TAIL_METHOD_FIELD = "tail_method"
TAIL_CUTOFF_TURN_FIELD = "tail_cutoff_turn"
TAIL_VALUE_PREFIX = "tail_value__"
TAIL_STANDARD_ERROR_PREFIX = "tail_standard_error__"
TAIL_HEADS = ("qol_years", "population_years", "feasibility", "settlement_guard")
# Optional per-row trajectory id (>= 0) separating several trajectories of one
# (world, camp): learner-state teacher continuations (labor_dagger).
TRAJECTORY_FIELD = "row_trajectory"
# Codes stored in ``tail_method``.
TAIL_NONE_TERMINAL = 0  # the trajectory terminated inside the collection window
TAIL_TERMINAL_EXACT = 1  # living cutoff; the continuation reached a true terminal
TAIL_RANDOMIZED = 2  # living cutoff; randomized-tail estimate
TAIL_UNAVAILABLE = 3  # living cutoff without an identified tail; excluded from fitting
TAIL_METHOD_NAMES = {
    TAIL_NONE_TERMINAL: "terminal-in-collection-window",
    TAIL_TERMINAL_EXACT: "terminal-exact-in-continuation",
    TAIL_RANDOMIZED: "independent-randomized-tail",
    TAIL_UNAVAILABLE: "unavailable",
}


@dataclass(frozen=True)
class RandomizedTailSettings:
    """Preregistered randomized-tail settings for teacher critic targets.

    The anchors' fixed 25-term observed prefix is the collection window itself
    (``turns``, default 25): every row observes its remaining window exactly and
    the tail starts at the collection-horizon state. One deterministic teacher
    continuation from that state is run for the frozen schedule's
    ``required_terms`` (the largest of ``draws`` geometric stops, ``P(N >= t) =
    rho**t``) and closed exactly at termination or by the Russian-roulette
    estimator. The schedule is frozen per world from ``seed_base + world_seed``
    before any continuation result exists and is never redrawn.
    """

    draws: int = 16
    rho: float = 0.95
    cap: int = 200
    seed_base: int = 9_100_000

    def __post_init__(self) -> None:
        if isinstance(self.draws, bool) or int(self.draws) < 2:
            raise ValueError("randomized tail requires at least two draws")
        if not 0.0 < float(self.rho) < 1.0:
            raise ValueError("randomized tail rho must be in (0, 1)")
        if isinstance(self.cap, bool) or int(self.cap) < 1:
            raise ValueError("randomized tail cap must be a positive integer")
        if isinstance(self.seed_base, bool) or int(self.seed_base) < 0:
            raise ValueError("randomized tail seed base must be a nonnegative integer")

    def schedule_seed(self, world_seed: int) -> int:
        return checked_seeds([int(self.seed_base) + int(world_seed)], "labor tail schedule")[0]

    def schedule(self, world_seed: int) -> Any:
        from .gate34_continuation_collector import freeze_geometric_schedule

        return freeze_geometric_schedule(
            schedule_id=f"labor-teacher-tail/world-{int(world_seed)}",
            rng_seed=self.schedule_seed(world_seed),
            draw_count=int(self.draws),
            survival_probability=float(self.rho),
            hard_cap_terms=int(self.cap),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": RETURN_TAIL_RANDOMIZED,
            "estimator": RETURN_TAIL_ESTIMATOR,
            "draws": int(self.draws),
            "rho": float(self.rho),
            "cap": int(self.cap),
            "schedule_seed_base": int(self.seed_base),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any] | None) -> RandomizedTailSettings | None:
        if state is None:
            return None
        if state.get("method") != RETURN_TAIL_RANDOMIZED:
            raise ValueError("unsupported return tail settings")
        return cls(
            int(state["draws"]), float(state["rho"]), int(state["cap"]),
            int(state["schedule_seed_base"]),
        )


@dataclass
class WorldCollection:
    """All rows of one collected world, including quarantined ones."""

    world_seed: int
    rows: list[dict[str, torch.Tensor]] = field(default_factory=list)
    quarantine: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surviving_camps: int = 0
    reset_identity: dict[str, Any] = field(default_factory=dict)
    decisions: int = 0
    native_resolves: int = 0
    # Randomized tail (absent for the legacy truncated-zero method).
    tail_record: dict[str, Any] | None = None
    tail_decisions: int = 0
    tail_native_resolves: int = 0
    tail_seconds: float = 0.0
    window_seconds: float = 0.0


@dataclass
class _TeacherTransition:
    live: list[Mapping[str, Any]]
    contexts: Any
    tensors: Mapping[str, torch.Tensor]
    labels: list[HeuristicLaborLabel]
    h_step: Mapping[str, Any]
    execution: Any
    next_step: Mapping[str, Any]
    decision_turn: int
    rewards: Mapping[int, Mapping[str, float]]
    terminal: Mapping[int, bool]
    elapsed: Mapping[int, float]
    outcomes: Mapping[int, Mapping[str, Any]]
    # Critic-only raw privileged arrays aligned with ``live`` (None: pre-privileged producer).
    privileged: Mapping[str, torch.Tensor] | None = None
    # The captured pre-decision boundary snapshot (base64) and the live-only state the
    # snapshot does not hold (``capture_transient``), for collection checkpoints.
    snapshot: str | None = None
    transient: Mapping[str, Any] | None = None

    def replay_payloads(self) -> dict[str, dict[str, Any]]:
        """The exact ``step_labor`` initial/retry payloads this decision executed."""
        return {
            "initial": {str(c): dict(a.payload) for c, a in sorted(self.execution.initial.items())},
            "retry": {str(c): dict(a.payload) for c, a in sorted(self.execution.retry.items())},
        }

    def head_rewards(self, camp: int, founded: Mapping[int, int]) -> dict[str, float]:
        """The four critic rewards of ``camp``'s transition, exactly as rows record them."""
        from .labor_training import _settlement_guard

        guard = _settlement_guard(
            self.outcomes[camp], founded.get(camp, self.decision_turn), int(self.next_step["turn"])
        )
        return {
            "qol_years": float(self.rewards[camp]["qol_years"]),
            "population_years": float(self.rewards[camp]["population_years"]),
            "feasibility": float(self.execution.initial_attempts[camp].feasibility_cost),
            "settlement_guard": float(guard),
        }


def _teacher_transition(
    *,
    labor_bridge: Any,
    heur_bridge: Any,
    step: Mapping[str, Any],
    world_seed: int,
    options: LaborResetOptions,
    normalizer: Any,
    personality: LaborPersonality,
    founded: Mapping[int, int],
) -> _TeacherTransition | None:
    """One converted-teacher decision for every live camp (two native resolves)."""
    from .labor_training import _transition_rewards

    live = [
        v for v in step["villages"] if v.get("observation") is not None and not v.get("terminated")
    ]
    if not live:
        return None
    contexts, tensors = labor_policy_tensors(
        live, normalizer, personality=personality_for_world(personality, int(world_seed))
    )
    catalogs = {context[0].camp_id: context[0] for context in contexts}
    decision_turn = int(step["turn"])
    # Shadow heuristic step from the captured boundary, same reset rules.
    snapshots: list[str] = []
    h_step, labels = shadow_teacher_labels(
        labor_bridge, heur_bridge, int(world_seed), options, live, contexts, snapshots
    )
    # Same boundary as the snapshot: the shadow step never advances the labor bridge.
    capture_transient = getattr(labor_bridge, "capture_transient", None)
    transient = capture_transient() if callable(capture_transient) else None
    retained: dict[int, RetainedLaborAction] = {}
    for row, (village, context, label) in enumerate(zip(live, contexts, labels, strict=True)):
        retained[int(village["camp"])] = _retained_action(
            label, context[0], tensors["movement_mask"][row : row + 1]
        )
    execution = execute_labor_decision(labor_bridge, catalogs, retained, _idle_retry)
    next_step = execution.step
    rewards, terminal, elapsed = _transition_rewards(
        next_step, tuple(catalogs), founded, world_index=0, decision_turn=decision_turn
    )
    return _TeacherTransition(
        live=live,
        contexts=contexts,
        tensors=tensors,
        labels=labels,
        h_step=h_step,
        execution=execution,
        next_step=next_step,
        decision_turn=decision_turn,
        rewards=rewards,
        terminal=terminal,
        elapsed=elapsed,
        outcomes={int(v["camp"]): v for v in next_step["villages"]},
        privileged=privileged_row_tensors(live),
        snapshot=snapshots[0] if snapshots else None,
        transient=transient,
    )


def _living_cutoffs(rows: Sequence[Mapping[str, torch.Tensor]]) -> dict[int, int]:
    """Camp -> turn of its last recorded row, for trajectories that did not terminate."""
    last: dict[int, tuple[int, bool]] = {}
    for row in rows:
        camp = int(row["row_camp"].reshape(-1)[0])
        turn = int(row["row_turn"].reshape(-1)[0])
        terminal = bool(row["transition_terminal"].reshape(-1)[0])
        if camp not in last or turn > last[camp][0]:
            last[camp] = (turn, terminal)
    return {camp: turn for camp, (turn, terminal) in sorted(last.items()) if not terminal}


@dataclass
class TeacherTailState:
    """Everything the teacher tail loop carries between decisions (JSON-serializable).

    ``done`` counts completed continuation decisions; a collection checkpoint
    taken after one of them resumes the loop exactly (``collect_heuristic_world``
    ``checkpoint``/``resume``). Dict insertion order is part of the state.
    """

    final_turn: int
    tracked: list[int]
    active: list[int]
    reasons: dict[int, str]
    path_rewards: dict[int, dict[str, list[float]]]
    path_years: dict[int, list[float]]
    path_terminal: dict[int, list[bool]]
    done: int = 0
    seconds: float = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "final_turn": int(self.final_turn),
            "tracked": [int(c) for c in self.tracked],
            "active": [int(c) for c in self.active],
            # JSON objects keep insertion order; lists of pairs make that explicit.
            "reasons": [[int(c), str(r)] for c, r in self.reasons.items()],
            "path_rewards": [
                [int(c), {h: [float(v) for v in values] for h, values in heads.items()}]
                for c, heads in self.path_rewards.items()
            ],
            "path_years": [[int(c), [float(v) for v in vs]] for c, vs in self.path_years.items()],
            "path_terminal": [[int(c), [bool(v) for v in vs]] for c, vs in self.path_terminal.items()],
            "done": int(self.done),
            "seconds": float(self.seconds),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> TeacherTailState:
        return cls(
            final_turn=int(state["final_turn"]),
            tracked=[int(c) for c in state["tracked"]],
            active=[int(c) for c in state["active"]],
            reasons={int(c): str(r) for c, r in state["reasons"]},
            path_rewards={
                int(c): {h: [float(v) for v in values] for h, values in heads.items()}
                for c, heads in state["path_rewards"]
            },
            path_years={int(c): [float(v) for v in vs] for c, vs in state["path_years"]},
            path_terminal={int(c): [bool(v) for v in vs] for c, vs in state["path_terminal"]},
            done=int(state["done"]),
            seconds=float(state["seconds"]),
        )


def _teacher_tail_state(
    result: WorldCollection, step: Mapping[str, Any], schedule: Any
) -> TeacherTailState:
    """The tail loop's initial state at the collection-horizon step."""
    cutoffs = _living_cutoffs(result.rows)
    reasons: dict[int, str] = {}
    # A living camp that did not decide at the final window step has no
    # continuation from its own cutoff state.
    last_decision = max((int(r["row_turn"].reshape(-1)[0]) for r in result.rows), default=-1)
    tracked = []
    for camp, turn in cutoffs.items():
        if turn != last_decision:
            reasons[camp] = "camp-left-live-set-before-collection-horizon"
        else:
            tracked.append(camp)
    if schedule.exceeds_hard_cap:
        for camp in tracked:
            reasons[camp] = "unresolved-hard-cap"
        tracked = []
    return TeacherTailState(
        final_turn=int(step["turn"]),
        tracked=tracked,
        active=list(tracked),
        reasons=reasons,
        path_rewards={camp: {head: [] for head in TAIL_HEADS} for camp in tracked},
        path_years={camp: [] for camp in tracked},
        path_terminal={camp: [] for camp in tracked},
    )


def _collect_teacher_tail(
    *,
    result: WorldCollection,
    tail: RandomizedTailSettings,
    labor_bridge: Any,
    heur_bridge: Any,
    step: Mapping[str, Any],
    options: LaborResetOptions,
    normalizer: Any,
    personality: LaborPersonality,
    founded: Mapping[int, int],
    window_turns: int,
    checkpoint: Any | None = None,
    resume: TeacherTailState | None = None,
) -> dict[int, dict[str, Any]]:
    """Continue the teacher from the collection-horizon state and close every living cutoff.

    Returns camp -> typed tail entry (method code, per-head value and standard
    error valued at the cutoff state). The continuation is one deterministic
    teacher path shared by every camp of the world; daughters founded during it
    are controlled but not credited to the cutoff camps.

    ``checkpoint.tail(result, transition, state)`` (optional) runs after every
    continuation decision; ``resume`` continues from such a state, with
    ``step`` the restored successor of its last decision.
    """
    from .labor_anchor_producer import summarize_labor_path

    world = int(result.world_seed)
    schedule = tail.schedule(world)
    state = _teacher_tail_state(result, step, schedule) if resume is None else resume
    prior_seconds = state.seconds
    final_turn = state.final_turn
    entries: dict[int, dict[str, Any]] = {}
    reasons = state.reasons
    tracked = state.tracked
    path_rewards = state.path_rewards
    path_years = state.path_years
    path_terminal = state.path_terminal
    active = state.active
    started = __import__("time").perf_counter()
    for _ in range(state.done, schedule.required_terms if tracked else 0):
        if not active:  # reachable only on resume after the decision that emptied it
            break
        transition = _teacher_transition(
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
            break
        result.tail_native_resolves += 2
        result.tail_decisions += 1
        decided = {context[0].camp_id for context in transition.contexts}
        for camp in list(active):
            if camp not in decided:
                reasons[camp] = "camp-left-live-set-without-terminal-in-continuation"
                active.remove(camp)
                continue
            for head, value in transition.head_rewards(camp, founded).items():
                path_rewards[camp][head].append(value)
            path_years[camp].append(float(transition.elapsed[camp]))
            path_terminal[camp].append(bool(transition.terminal[camp]))
            if transition.terminal[camp]:
                active.remove(camp)
        step = transition.next_step
        state.done += 1
        if checkpoint is not None:
            state.seconds = prior_seconds + __import__("time").perf_counter() - started
            checkpoint.tail(result, transition, state)
        if not active:
            break
    result.tail_seconds = prior_seconds + __import__("time").perf_counter() - started
    for camp in tracked:
        if camp in reasons:
            continue
        if not path_terminal[camp] or (
            not path_terminal[camp][-1] and len(path_terminal[camp]) < schedule.required_terms
        ):
            reasons[camp] = "continuation-ended-before-schedule-without-terminal"
            continue
        values: dict[str, float] = {}
        errors: dict[str, float] = {}
        methods = set()
        for head in TAIL_HEADS:
            summary = summarize_labor_path(
                path_rewards[camp][head],
                path_years[camp],
                path_terminal[camp],
                schedule,
                head=head,
                prefix_terms=0,
                provenance_id=f"world-{world}/camp-{camp}/teacher-tail/{head}",
            )
            if not summary.available:
                break
            values[head] = float(summary.value)
            errors[head] = float(summary.standard_error)
            methods.add(summary.method)
        if len(values) != len(TAIL_HEADS):
            reasons[camp] = "randomized-tail-not-identified"
            continue
        code = TAIL_TERMINAL_EXACT if methods == {"terminal-exact"} else TAIL_RANDOMIZED
        entries[camp] = {
            "method": code,
            "cutoff_turn": final_turn,
            "value": values,
            "standard_error": errors,
            "continuation_terms": len(path_terminal[camp]),
        }
    for camp, reason in reasons.items():
        entries[camp] = {
            "method": TAIL_UNAVAILABLE,
            "cutoff_turn": final_turn,
            "value": dict.fromkeys(TAIL_HEADS, 0.0),
            "standard_error": dict.fromkeys(TAIL_HEADS, 0.0),
            "continuation_terms": len(path_terminal.get(camp, [])),
            "reason": reason,
        }
    result.tail_record = {
        **tail.state_dict(),
        "prefix_terms": int(window_turns),
        "prefix_semantics": (
            "the observed collection window; a row at window index i adds turns-i exact "
            "terms before the tail, which is valued at the collection-horizon state"
        ),
        "schedule_id": schedule.schedule_id,
        "schedule_seed": int(schedule.rng_seed),
        "stop_indices": list(schedule.stop_indices),
        "required_terms": int(schedule.required_terms),
        "exceeds_hard_cap": bool(schedule.exceeds_hard_cap),
        "dropped_draws": 0,
        "critic_bootstrap_used": False,
        "standard_error_scope": (
            "stopping-randomization only: the teacher and simulator are deterministic, so "
            "one continuation is the whole teacher path; excludes world uncertainty"
        ),
        "continuation_decisions": int(result.tail_decisions),
        "continuation_native_resolves": int(result.tail_native_resolves),
        "continuation_seconds": float(result.tail_seconds),
        "cutoffs": {
            str(camp): {
                "method": TAIL_METHOD_NAMES[entry["method"]],
                "cutoff_turn": int(entry["cutoff_turn"]),
                "continuation_terms": int(entry["continuation_terms"]),
                "value": dict(entry["value"]),
                "standard_error": dict(entry["standard_error"]),
                **({"reason": entry["reason"]} if "reason" in entry else {}),
            }
            for camp, entry in sorted(entries.items())
        },
        "method_counts": {
            name: sum(1 for e in entries.values() if e["method"] == code)
            for code, name in TAIL_METHOD_NAMES.items()
            if code != TAIL_NONE_TERMINAL
        },
    }
    return entries


def _attach_tail_rows(
    rows: Sequence[dict[str, torch.Tensor]], entries: Mapping[int, Mapping[str, Any]]
) -> None:
    """Broadcast each trajectory's typed tail target onto all of its rows."""
    for row in rows:
        camp = int(row["row_camp"].reshape(-1)[0])
        entry = entries.get(camp)
        if entry is None:
            code, cutoff = TAIL_NONE_TERMINAL, -1
            values = errors = dict.fromkeys(TAIL_HEADS, 0.0)
        else:
            code, cutoff = int(entry["method"]), int(entry["cutoff_turn"])
            values, errors = entry["value"], entry["standard_error"]
        row[TAIL_METHOD_FIELD] = torch.tensor([code], dtype=torch.long)
        row[TAIL_CUTOFF_TURN_FIELD] = torch.tensor([cutoff], dtype=torch.long)
        for head in TAIL_HEADS:
            row[TAIL_VALUE_PREFIX + head] = torch.tensor([float(values[head])], dtype=torch.float64)
            row[TAIL_STANDARD_ERROR_PREFIX + head] = torch.tensor(
                [float(errors[head])], dtype=torch.float64
            )


def record_teacher_transition(
    result: WorldCollection,
    transition: _TeacherTransition,
    world_seed: int,
    founded: Mapping[int, int],
    camps: set[int] | None = None,
) -> None:
    """Append one converted-teacher decision's rows (every live camp) to ``result``.

    The collector's row format: pre-action inputs, the critic-only privileged
    arrays of the same boundary, the converted label, identity and the
    transition rewards; rejected initial recommendations are quarantined.
    ``camps`` restricts the recorded rows (and their counters) to those camps
    (curated scenario roots, ``labor_scenarios``); every camp still acted.
    """
    world_seed = int(world_seed)
    decision_turn = transition.decision_turn
    execution = transition.execution
    for row, (village, label) in enumerate(
        zip(transition.live, transition.labels, strict=True)
    ):
        camp = int(village["camp"])
        if camps is not None and camp not in camps:
            continue
        for reason in label.reasons:
            result.reasons[reason.split(":", 1)[0]] += 1
        attempt = execution.initial_attempts[camp]
        accepted = bool(attempt.valid)
        result.counters["total_decisions"] += 1
        result.counters["initial_accepted" if accepted else "initial_rejected"] += 1
        if not accepted:
            result.counters["quarantined"] += 1
            result.quarantine.append(
                {
                    "world": world_seed,
                    "camp": camp,
                    "turn": decision_turn,
                    "attempt": 0,
                    "violations": [
                        {"reasonCode": v.reason_code, "reason": v.reason}
                        for v in attempt.violations
                    ],
                    "labelReasons": list(label.reasons),
                }
            )
        if label.specialty_valid:
            result.counters["labor_label_valid"] += 1
        if label.full_action_valid:
            result.counters["full_action_label_valid"] += 1
        row_tensors = _one_row(transition.tensors, row)
        if transition.privileged is not None:
            # Raw critic-only arrays from the same decision boundary as the
            # observation (row schema 1); never part of the actor tensors.
            row_tensors.update(_one_row(transition.privileged, row))
        row_tensors.update(_label_tensors(label))
        heads = transition.head_rewards(camp, founded)
        row_tensors.update(
            {
                "row_world": torch.tensor([world_seed], dtype=torch.long),
                "row_camp": torch.tensor([camp], dtype=torch.long),
                "row_turn": torch.tensor([decision_turn], dtype=torch.long),
                "row_attempt": torch.tensor([0], dtype=torch.long),
                "row_quarantined": torch.tensor([not accepted], dtype=torch.bool),
                "reward_qol_years": torch.tensor(
                    [transition.rewards[camp]["qol_years"]], dtype=torch.float64
                ),
                "reward_population_years": torch.tensor(
                    [transition.rewards[camp]["population_years"]], dtype=torch.float64
                ),
                "reward_feasibility": torch.tensor(
                    [heads["feasibility"]], dtype=torch.float64
                ),
                "reward_settlement_guard": torch.tensor(
                    [heads["settlement_guard"]], dtype=torch.float64
                ),
                ELAPSED_YEARS_FIELD: torch.tensor(
                    [float(transition.elapsed[camp])], dtype=torch.float64
                ),
                "transition_terminal": torch.tensor(
                    [bool(transition.terminal[camp])], dtype=torch.bool
                ),
            }
        )
        result.rows.append(row_tensors)


def collect_heuristic_world(
    *,
    normalizer: Any,
    world_seed: int,
    turns: int,
    reset_options: LaborResetOptions | None = None,
    personality: LaborPersonality | None = None,
    labor_bridge_factory: Any = LaborModel9Bridge,
    heuristic_bridge_factory: Any = None,
    economic_trace: Any | None = None,
    progress: Any | None = None,
    tail: RandomizedTailSettings | None = None,
    checkpoint: Any | None = None,
    resume: Any | None = None,
) -> WorldCollection:
    """Advance one world under converted heuristic actions and record every row.

    Each row records pre-action inputs, the converted label with per-head
    validity, identity (world, camp, turn, attempt=0), the transition rewards
    used for critic returns, and whether the initial recommendation was
    accepted. Rejected initial recommendations are quarantined: the all-idle
    retry that executes instead never becomes their label.

    With ``tail`` the teacher continues past the window (no rows recorded) and
    every row carries its trajectory's typed tail target (``tail_*`` arrays);
    without it the legacy truncated-zero window is collected unchanged.

    Decision-boundary checkpointing (``labor_collect_resume``): after every
    window decision ``checkpoint.window(result, transition, founded, done)``
    and after every tail decision ``checkpoint.tail(result, transition,
    tail_state)`` run; ``resume`` (a ``labor_collect_resume.ResumePoint``)
    restores the result, the labor bridge and the loop position of such a
    checkpoint, so the remaining decisions and the output are those of the
    uninterrupted run. Neither changes anything when absent.
    """

    import time

    options = reset_options or LaborResetOptions()
    declared_personality = LaborPersonality() if personality is None else personality
    heur_factory = heuristic_bridge_factory or (lambda: Model9Bridge(economic_diagnostics=True))

    result = WorldCollection(world_seed=int(world_seed))
    started = time.perf_counter()
    with labor_bridge_factory() as labor_bridge, heur_factory() as heur_bridge:
        if resume is None:
            step = labor_bridge.reset(int(world_seed), **options.reset_kwargs())
            identity = getattr(labor_bridge, "last_reset_identity", None)
            result.reset_identity = dict(identity) if isinstance(identity, Mapping) else {}
            founded = {int(row["camp"]): int(step["turn"]) for row in step["villages"]}
            window_done, tail_state, prior_window_seconds = 0, None, 0.0
            if checkpoint is not None:
                checkpoint.begin(result, step)
        else:
            step = resume.restore(labor_bridge, result)
            founded = dict(resume.founded)
            window_done, tail_state = int(resume.window_done), resume.tail_state
            prior_window_seconds = float(result.window_seconds)
        for index in range(window_done, turns if tail_state is None else window_done):
            transition = _teacher_transition(
                labor_bridge=labor_bridge,
                heur_bridge=heur_bridge,
                step=step,
                world_seed=int(world_seed),
                options=options,
                normalizer=normalizer,
                personality=declared_personality,
                founded=founded,
            )
            if transition is None:
                break
            result.native_resolves += 2
            result.decisions += 1
            decision_turn = transition.decision_turn
            if economic_trace is not None and getattr(heur_bridge, "economic_diagnostics", None):
                for camp_id, diag in heur_bridge.economic_diagnostics.items():
                    trans_id = diag.get("transitionId") or f"{world_seed}:{camp_id}:{decision_turn}"
                    if trans_id not in economic_trace.transition_ids:
                        economic_trace.transition_ids.append(trans_id)
                    economic_trace.write(
                        "transitions",
                        {
                            "campId": int(camp_id),
                            "turn": decision_turn,
                            "worldSeed": int(world_seed),
                            "transitionId": trans_id,
                            "economic": diag,
                        },
                    )
            record_teacher_transition(result, transition, int(world_seed), founded)
            step = transition.next_step
            if checkpoint is not None:
                result.window_seconds = prior_window_seconds + time.perf_counter() - started
                checkpoint.window(result, transition, founded, index + 1)
            if progress is not None:
                progress(result)
        if tail_state is None:
            result.window_seconds = prior_window_seconds + time.perf_counter() - started
            result.surviving_camps = len(
                [v for v in step.get("villages", []) if not v.get("terminated")]
            )
        if tail is not None:
            entries = _collect_teacher_tail(
                result=result,
                tail=tail,
                labor_bridge=labor_bridge,
                heur_bridge=heur_bridge,
                step=step,
                options=options,
                normalizer=normalizer,
                personality=declared_personality,
                founded=founded,
                window_turns=int(turns),
                checkpoint=checkpoint,
                resume=tail_state,
            )
            _attach_tail_rows(result.rows, entries)
            if progress is not None:
                progress(result)
    return result


def world_balanced_weights(worlds: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
    """Per-row weights giving every world equal total mass (mean weight one)."""
    worlds = worlds.reshape(-1)
    keep = torch.ones_like(worlds, dtype=torch.bool) if keep is None else keep.reshape(-1)
    weights = torch.zeros(len(worlds), dtype=torch.float32)
    kept_worlds = torch.unique(worlds[keep])
    if len(kept_worlds) == 0:
        return weights
    total_rows = int(keep.sum())
    for world in kept_worlds.tolist():
        rows = (worlds == world) & keep
        weights[rows] = total_rows / (len(kept_worlds) * float(rows.sum()))
    return weights


def _refuse_critic_only_rows(dataset: Mapping[str, Any], context: str) -> None:
    """Learner-state teacher continuations are V targets only; actor labels stay unchanged."""
    collection = dataset.get("collection") if isinstance(dataset, Mapping) else None
    kinds = set()
    if isinstance(collection, Mapping):
        kinds.add(collection.get("source_kind"))
        kinds |= {c.get("source_kind") for c in collection.get("component_collections", [])}
    sources = dataset.get("row_source")
    if CONTINUATION_SOURCE_KIND in kinds or (
        isinstance(sources, torch.Tensor)
        and bool((sources.reshape(-1) == CONTINUATION_ROW_SOURCE).any())
    ):
        raise ValueError(
            f"{context} refuses learner-state teacher continuation rows: they are critic "
            "(V^rho) targets, not actor labels"
        )


def migrate_legacy_elapsed_field(dataset: dict[str, Any]) -> dict[str, Any]:
    """Rename a legacy ``transition_elapsed_days`` array to ``transition_elapsed_years``.

    The legacy array already holds years (the bridge's per-village ``years``,
    contracts.TURN_YEARS per turn), so the values are kept unchanged. A dataset
    carrying both names must agree exactly. Mutates and returns ``dataset``.
    """
    if LEGACY_ELAPSED_DAYS_FIELD not in dataset:
        return dataset
    legacy = dataset.pop(LEGACY_ELAPSED_DAYS_FIELD)
    current = dataset.get(ELAPSED_YEARS_FIELD)
    if current is not None and not torch.equal(torch.as_tensor(current), torch.as_tensor(legacy)):
        raise ValueError(
            f"dataset carries both {ELAPSED_YEARS_FIELD} and the legacy "
            f"{LEGACY_ELAPSED_DAYS_FIELD} with different values"
        )
    dataset[ELAPSED_YEARS_FIELD] = legacy if current is None else current
    return dataset


def elapsed_years(dataset: Mapping[str, Any]) -> torch.Tensor:
    """Per-row elapsed physical years, reading the legacy misnamed key if needed."""
    if ELAPSED_YEARS_FIELD in dataset:
        return dataset[ELAPSED_YEARS_FIELD]
    if LEGACY_ELAPSED_DAYS_FIELD in dataset:
        return dataset[LEGACY_ELAPSED_DAYS_FIELD]
    raise KeyError(ELAPSED_YEARS_FIELD)


def return_tail_method(dataset: Mapping[str, Any]) -> str:
    """The declared tail method of a critic dataset's targets (never inferred as zero silently).

    A dataset carrying ``tail_*`` arrays uses the randomized teacher tail. One
    without them uses the legacy truncated-zero window, which must be what its
    collection metadata declares (absent metadata: an in-memory legacy fixture).
    """
    collection = dataset.get("collection") if isinstance(dataset, Mapping) else None
    declared = collection.get("return_tail") if isinstance(collection, Mapping) else None
    if TAIL_METHOD_FIELD in dataset:
        if declared not in (None, RETURN_TAIL_RANDOMIZED):
            raise ValueError(f"dataset carries tail arrays but declares return tail {declared!r}")
        return RETURN_TAIL_RANDOMIZED
    if declared == RETURN_TAIL_RANDOMIZED:
        raise ValueError("dataset declares a randomized return tail but carries no tail arrays")
    if declared not in (None, RETURN_TAIL_TRUNCATED_ZERO):
        raise ValueError(f"unsupported critic return tail {declared!r}")
    return RETURN_TAIL_TRUNCATED_ZERO


def critic_target_available(dataset: Mapping[str, Any]) -> torch.Tensor:
    """Rows whose critic target is identified (a living cutoff without a tail is not)."""
    n = int(dataset["row_world"].shape[0])
    if return_tail_method(dataset) != RETURN_TAIL_RANDOMIZED:
        available = torch.ones(n, dtype=torch.bool)
    else:
        available = dataset[TAIL_METHOD_FIELD].reshape(n) != TAIL_UNAVAILABLE
    if VALUE_MASK_FIELD in dataset:  # only present after a loader correction masked rows
        available = available & ~dataset[VALUE_MASK_FIELD].reshape(n).bool()
    return available


# False-terminal correction (user-approved item 6, 28 Oct). The collecting simulator set
# Terminated = Population <= 0 while hunters were away (ElderModel9.cs:1075), so a camp can be
# marked terminal and reappear later (fit8-b01 row 17948: world 90104, camp 24, turn 9, back at
# turn 11). Policy "mask": a terminal row with later rows in its own (world, camp,
# row_trajectory) trajectory is NOT terminal. If its next row is the next turn, the return is
# rebuilt exactly through the continuing rows (rechained). If a turn is missing (the camp had no
# row while empty), the missing turn's rewards and the true discount are not in the data, so the
# return through it is not constructible: the false-terminal row and every earlier row of its
# trajectory (whose returns pass through it) are dropped from value targets (``row_value_masked``;
# BC labels and the rows themselves are kept). Later rows' returns never cross it and are
# unchanged. "none" (default) returns the dataset object itself (bitwise unchanged).
FALSE_TERMINAL_FIELD = "false_terminal_correction"
FALSE_TERMINAL_POLICIES = ("none", "mask")
VALUE_MASK_FIELD = "row_value_masked"


def false_terminal_policy_of(dataset: Mapping[str, Any]) -> str:
    """The false-terminal correction already applied to a loaded dataset ("none" if absent)."""
    collection = dataset.get("collection") if isinstance(dataset, Mapping) else None
    record = collection.get(FALSE_TERMINAL_FIELD) if isinstance(collection, Mapping) else None
    return "none" if record is None else str(record["policy"])


def correct_false_terminals(dataset: Mapping[str, Any], policy: str | None) -> Mapping[str, Any]:
    """Apply ``--false-terminal-policy`` (see :data:`FALSE_TERMINAL_FIELD`); counts recorded."""
    if policy is None or policy == "none":
        return dataset
    if policy not in FALSE_TERMINAL_POLICIES:
        raise ValueError(f"false-terminal policy must be one of {FALSE_TERMINAL_POLICIES}")
    if false_terminal_policy_of(dataset) != "none":
        raise ValueError("dataset false terminals are already corrected")
    n = int(dataset["row_world"].shape[0])
    original = dataset["transition_terminal"]
    terminal = original.reshape(n).clone()
    worlds = dataset["row_world"].reshape(n).tolist()
    camps = dataset["row_camp"].reshape(n).tolist()
    turns = dataset["row_turn"].reshape(n).tolist()
    trajectories = (
        dataset[TRAJECTORY_FIELD].reshape(n).tolist() if TRAJECTORY_FIELD in dataset else [-1] * n
    )
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i in range(n):
        groups[(int(worlds[i]), int(camps[i]), int(trajectories[i]))].append(i)
    masked = torch.zeros(n, dtype=torch.bool)
    events: list[dict[str, Any]] = []
    for key in sorted(groups):
        indices = sorted(groups[key], key=lambda i: turns[i])
        for pos, i in enumerate(indices[:-1]):
            if not bool(terminal[i]):
                continue
            terminal[i] = False
            following = indices[pos + 1]
            rechained = int(turns[following]) - int(turns[i]) == 1
            if not rechained:
                masked[indices[: pos + 1]] = True
            events.append(
                {
                    "row": int(i),
                    "world": key[0],
                    "camp": key[1],
                    "trajectory": key[2],
                    "turn": int(turns[i]),
                    "next_turn": int(turns[following]),
                    "later_rows": len(indices) - pos - 1,
                    "action": "rechained" if rechained else "masked-through-row",
                    "masked_rows": 0 if rechained else pos + 1,
                }
            )
    out = dict(dataset)
    out["transition_terminal"] = terminal.to(original.dtype).reshape(original.shape)
    if bool(masked.any()):
        out[VALUE_MASK_FIELD] = masked
    collection = dict(dataset.get("collection") or {})
    collection[FALSE_TERMINAL_FIELD] = {
        "policy": policy,
        "false_terminal_rows": len(events),
        "rechained_rows": sum(1 for e in events if e["action"] == "rechained"),
        "value_masked_rows": int(masked.sum()),
        "events": events,
        "rule": "a terminal row with later rows in its (world, camp, row_trajectory) trajectory "
        "is non-terminal; next row at turn + 1: return rebuilt through it; otherwise that row "
        "and every earlier row of the trajectory are dropped from value targets",
    }
    out["collection"] = collection
    return out


def loader_corrections(summary: Mapping[str, Any] | None) -> dict[str, str]:
    """The loader corrections a critic fit recorded (absent: none), for evaluation loads."""
    summary = summary or {}
    return {
        "false_terminal_policy": str(
            (summary.get(FALSE_TERMINAL_FIELD) or {}).get("policy", "none")
        ),
        "tail_normalize": str((summary.get(TAIL_NORMALIZATION_FIELD) or {}).get("mode", "none")),
    }


def apply_loader_corrections(
    dataset: Mapping[str, Any],
    *,
    false_terminal_policy: str | None = None,
    tail_normalize: str | None = None,
) -> Mapping[str, Any]:
    """False-terminal correction, then tail normalisation; both off returns ``dataset``."""
    return normalize_world_tails(
        correct_false_terminals(dataset, false_terminal_policy), tail_normalize
    )


# Per-world tail normalisation (reviewer tweak 28 Oct, Fable audit section 3). Every camp of a
# world shares one frozen roulette stop schedule, so a constant-QoL continuation is estimated at
# f_w x its value (``labor_pretrain_eval.tail_schedule_factor``, E f = 1; b01 factors 0.6-1.7 and
# corr(f_w, world-mean target) 0.979). ``--tail-normalize world-factor`` divides every stored
# randomized tail (and its standard error) of the two task heads by f_w on load: exact for a
# constant continuation, approximate otherwise. Terminal-exact tails carry no roulette and are
# left alone. ``none`` (the default) returns the dataset object itself (bitwise unchanged).
TAIL_NORMALIZATION_FIELD = "tail_normalization"
TAIL_NORMALIZATION_MODES = ("none", "world-factor")
TAIL_NORMALIZED_HEADS = ("qol_years", "population_years")


def tail_normalization_mode(dataset: Mapping[str, Any]) -> str:
    """The normalisation already applied to a loaded dataset's tails ("none" if absent)."""
    collection = dataset.get("collection") if isinstance(dataset, Mapping) else None
    record = collection.get(TAIL_NORMALIZATION_FIELD) if isinstance(collection, Mapping) else None
    return "none" if record is None else str(record["mode"])


def normalize_world_tails(
    dataset: Mapping[str, Any], mode: str | None, *, weight: float | None = None
) -> Mapping[str, Any]:
    """Divide each stored randomized tail by its world's deterministic schedule factor.

    ``mode`` None or "none": ``dataset`` itself. "world-factor": a shallow copy whose
    ``tail_value__<head>`` and ``tail_standard_error__<head>`` arrays of
    :data:`TAIL_NORMALIZED_HEADS` are divided by f_w on rows with ``TAIL_RANDOMIZED`` (the
    convention of ``labor_pretrain_eval.qol_tail_diagnostics``: f_w =
    ``tail_schedule_factor(return_tail_by_world[w], W)``, W the unit-QoL discount weight).
    The factors are recorded in ``collection["tail_normalization"]``; the dataset digest is
    the raw data's and is unchanged. A randomized world without a schedule record, an
    already-normalised dataset and a dataset without randomized tails are refused.
    """
    if mode is None or mode == "none":
        return dataset
    if mode not in TAIL_NORMALIZATION_MODES:
        raise ValueError(f"tail normalisation must be one of {TAIL_NORMALIZATION_MODES}")
    if return_tail_method(dataset) != RETURN_TAIL_RANDOMIZED:
        raise ValueError("--tail-normalize world-factor needs randomized-tail targets")
    if tail_normalization_mode(dataset) != "none":
        raise ValueError("dataset tails are already normalised")
    from .labor_model import labor_qol_discount_weight
    from .labor_pretrain_eval import tail_schedule_factor

    w = labor_qol_discount_weight() if weight is None else float(weight)
    collection = dict(dataset.get("collection") or {})
    records = collection.get("return_tail_by_world") or {}
    n = int(dataset["row_world"].shape[0])
    worlds = dataset["row_world"].reshape(n)
    randomized = dataset[TAIL_METHOD_FIELD].reshape(n) == TAIL_RANDOMIZED
    factors: dict[int, float] = {}
    for world in sorted({int(x) for x in worlds[randomized].tolist()}):
        factor = tail_schedule_factor(records.get(str(world)), w)
        if factor is None or not math.isfinite(factor) or factor <= 0.0:
            raise ValueError(f"world {world}: no usable roulette schedule record to normalise by")
        factors[world] = float(factor)
    row_factor = torch.ones(n, dtype=torch.float64)
    for world, factor in factors.items():
        row_factor[randomized & (worlds == world)] = factor
    out = dict(dataset)
    for head in TAIL_NORMALIZED_HEADS:
        for prefix in (TAIL_VALUE_PREFIX, TAIL_STANDARD_ERROR_PREFIX):
            key = prefix + head
            if key not in dataset:
                continue
            original = dataset[key]
            out[key] = (original.reshape(n).double() / row_factor).to(original.dtype).reshape(
                original.shape
            )
    collection[TAIL_NORMALIZATION_FIELD] = {
        "mode": "world-factor",
        "heads": list(TAIL_NORMALIZED_HEADS),
        "factors": {str(world): factor for world, factor in factors.items()},
        "mean_factor": float(sum(factors.values()) / len(factors)) if factors else None,
        "rows_normalized": int(randomized.sum()),
        "weight": w,
        "convention": "tail / f_w with f_w = labor_pretrain_eval.tail_schedule_factor("
        "return_tail_by_world[w], W), randomized-tail rows only (terminal-exact unchanged); "
        "values and standard errors of the task heads",
    }
    out["collection"] = collection
    return out


def discounted_returns(dataset: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per (world, camp) trajectory discounted returns, closed by the declared tail.

    With the randomized teacher tail, each trajectory's running return starts
    at its typed tail value (valued at the state after its last row); a living
    cutoff must carry a tail entry and a terminated one must not. Unavailable
    tails return finite placeholders that :func:`critic_target_available`
    excludes. The legacy method starts every trajectory at zero.
    """
    names = {
        "qol_years": "reward_qol_years",
        "population_years": "reward_population_years",
        "feasibility": "reward_feasibility",
        "settlement_guard": "reward_settlement_guard",
    }
    randomized = return_tail_method(dataset) == RETURN_TAIL_RANDOMIZED
    n = int(dataset["row_world"].shape[0])
    returns = {name: torch.zeros(n, dtype=torch.float64) for name in names}
    # Teacher trajectories are (world, camp); rows carrying ``row_trajectory`` >= 0
    # (learner-state teacher continuations) are their own trajectories within it.
    trajectories = (
        dataset[TRAJECTORY_FIELD].reshape(-1).tolist()
        if TRAJECTORY_FIELD in dataset
        else [-1] * n
    )
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, (world, camp, trajectory) in enumerate(
        zip(dataset["row_world"].tolist(), dataset["row_camp"].tolist(), trajectories, strict=True)
    ):
        groups[(int(world), int(camp), int(trajectory))].append(index)
    turns = dataset["row_turn"].tolist()
    elapsed = elapsed_years(dataset).reshape(-1).tolist()
    terminal = dataset["transition_terminal"].reshape(-1).tolist()
    if randomized:
        codes = dataset[TAIL_METHOD_FIELD].reshape(-1).tolist()
        tail_values = {
            name: dataset[TAIL_VALUE_PREFIX + name].reshape(-1).double().tolist() for name in names
        }
    for (world, camp, _), indices in groups.items():
        indices.sort(key=lambda i: turns[i])
        running = dict.fromkeys(names, 0.0)
        if randomized:
            group_codes = {codes[i] for i in indices}
            if len(group_codes) != 1:
                raise ValueError(
                    f"world {world} camp {camp}: tail method differs within trajectory"
                )
            code = group_codes.pop()
            if any(len({tail_values[name][i] for i in indices}) != 1 for name in names):
                raise ValueError(f"world {world} camp {camp}: tail value differs within trajectory")
            last = indices[-1]
            if terminal[last] and code != TAIL_NONE_TERMINAL:
                raise ValueError(f"world {world} camp {camp}: terminated trajectory carries a tail")
            if not terminal[last] and code == TAIL_NONE_TERMINAL:
                raise ValueError(
                    f"world {world} camp {camp}: living cutoff has no tail entry (would be zero)"
                )
            if code in (TAIL_TERMINAL_EXACT, TAIL_RANDOMIZED):
                running = {name: tail_values[name][last] for name in names}
        for i in reversed(indices):
            gamma = math.exp(-float(elapsed[i]) / CRITIC_DISCOUNT_YEARS)
            if terminal[i]:
                running = dict.fromkeys(names, 0.0)
            for name, key in names.items():
                running[name] = float(dataset[key].reshape(-1)[i]) + gamma * running[name]
                returns[name][i] = running[name]
    return {name: value.to(torch.float32) for name, value in returns.items()}


def _finish_dataset(collections: Sequence[WorldCollection], turns: int) -> dict[str, Any]:
    rows = [row for c in collections for row in c.rows]
    if not rows:
        raise ValueError("heuristic BC collection produced zero training rows")
    dataset: dict[str, Any] = _concat(rows, "heuristic BC dataset")
    keep = ~dataset["row_quarantined"].reshape(-1)
    dataset["row_weight"] = world_balanced_weights(dataset["row_world"], keep)
    counters: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    for c in collections:
        for key, value in c.counters.items():
            counters[key] += int(value)
        for key, value in c.reasons.items():
            reasons[key] += int(value)
    valid = dataset["label_specialty_valid"] & keep
    mean_spec = (
        dataset["target_specialty"][valid].double().mean(dim=0).tolist() if bool(valid.any()) else []
    )
    dataset["collection"] = {
        "label_contract": LABOR_LABEL_CONTRACT,
        "total_decisions": int(counters["total_decisions"]),
        "initial_accepted": int(counters["initial_accepted"]),
        "initial_rejected": int(counters["initial_rejected"]),
        "quarantined_rows": int(counters["quarantined"]),
        "labor_label_valid_rows": int(counters["labor_label_valid"]),
        "full_action_label_valid_rows": int(counters["full_action_label_valid"]),
        "label_unavailable_reasons": dict(sorted(reasons.items())),
        "quarantine": [q for c in collections for q in c.quarantine],
        "surviving_camps_by_world": {str(c.world_seed): int(c.surviving_camps) for c in collections},
        "rows_by_world": {
            str(c.world_seed): len(c.rows) for c in collections
        },
        "decisions_by_world": {str(c.world_seed): int(c.decisions) for c in collections},
        "native_resolves": int(sum(c.native_resolves for c in collections)),
        "mean_specialty_target": mean_spec,
        "world_seeds": [int(c.world_seed) for c in collections],
        "turns": int(turns),
        "sampling": "world-balanced-row-weights",
    }
    tails = [c.tail_record for c in collections]
    if any(t is not None for t in tails):
        if any(t is None for t in tails):
            raise ValueError("heuristic collection mixes randomized-tail and truncated worlds")
        dataset["collection"]["return_tail"] = RETURN_TAIL_RANDOMIZED
        dataset["collection"]["return_tail_by_world"] = {
            str(c.world_seed): c.tail_record for c in collections
        }
    else:
        dataset["collection"]["return_tail"] = RETURN_TAIL_TRUNCATED_ZERO
    return dataset


def collect_heuristic_bc_dataset(
    *,
    normalizer: Any,
    world_seeds: Sequence[int],
    turns: int,
    personality: LaborPersonality | None = None,
    labor_bridge_factory: Any = LaborModel9Bridge,
    heuristic_bridge_factory: Any = None,
    economic_trace: Any | None = None,
    allow_vm_scale: bool = False,
    reset_options: LaborResetOptions | None = None,
    tail: RandomizedTailSettings | None = None,
) -> dict[str, Any]:
    """Collect closed-loop heuristic labor trajectories for behavioral cloning.

    ``tail`` adds the randomized teacher tail to every living cutoff (critic targets).
    """
    worlds = checked_seeds(list(world_seeds), "labor heuristic bc worlds")
    scan({"worldSeeds": worlds}, "labor heuristic bc worlds")
    if not allow_vm_scale and len(worlds) > 3:
        raise ValueError("local labor heuristic bc requires 1..3 world seeds")
    if not worlds:
        raise ValueError("labor heuristic bc requires at least one world seed")
    if turns < 1:
        raise ValueError("turns must be >= 1")
    collections = [
        collect_heuristic_world(
            normalizer=normalizer,
            world_seed=seed,
            turns=turns,
            reset_options=reset_options,
            personality=personality,
            labor_bridge_factory=labor_bridge_factory,
            heuristic_bridge_factory=heuristic_bridge_factory,
            economic_trace=economic_trace,
            tail=tail,
        )
        for seed in worlds
    ]
    if economic_trace is not None:
        economic_trace.close()
    dataset = _finish_dataset(collections, turns)
    dataset["collection"]["reset_options"] = (reset_options or LaborResetOptions()).state_dict()
    return dataset


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def masked_allocation_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """-sum_i target_i log softmax_mask(logits)_i over legal coordinates only.

    Uses a stable masked log-softmax (no probability clamp), so a
    positive-target coordinate keeps its corrective gradient however small its
    probability. Rows with no legal coordinate return zero.
    """
    if logits.shape != target.shape or logits.shape != mask.shape:
        raise ValueError("allocation logits, targets and masks must share a shape")
    has_legal = mask.any(dim=-1, keepdim=True)
    safe_mask = mask | ~has_legal
    log_probs = torch.log_softmax(logits.masked_fill(~safe_mask, -torch.inf), dim=-1)
    log_probs = log_probs.masked_fill(~mask, 0.0)
    return -(target.masked_fill(~mask, 0.0) * log_probs).sum(dim=-1)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp(min=1e-12)


def _batch_loss_and_l1(
    actor: LaborActor,
    batch_tensors: Mapping[str, torch.Tensor],
    *,
    target_log_std: float,
    inactive_role_weight: float = 0.0,
    label_smoothing: float = 1e-6,
    spec_weights: torch.Tensor | None = None,
    act_weights: torch.Tensor | None = None,
    task_terms: dict[str, Any] | None = None,
    task_weights: Mapping[str, float] | None = None,
    role_weighting: str = "share",
) -> tuple[torch.Tensor, float, float, dict[str, float]]:
    """The actor BC loss of one batch, its specialty/activity L1 and loss components.

    Opt-in task decomposition (``labor_actor_gradnorm``; all three default to the historical
    loss, bit for bit): ``task_terms`` (a dict) is filled with the per-task loss tensors of
    :data:`~.labor_actor_gradnorm.ACTOR_BC_TASKS` on the same graph (``"loss"``), each role's
    staffed-row count (``"staffed_rows"``) and its composition-normalised mean CE
    (``"mean_ce"``). The 13 role terms ``activity:<role>`` share the activity denominator, so
    they sum to the activity loss. ``task_weights`` rebuilds the total as the weighted sum of
    those task terms (plus the unweighted gauge anchors, log-std, trade value and trade share
    terms). ``role_weighting`` changes only the reported role terms (``uniform``: each staffed
    role of a row gets an equal share of the row's activity weight); the total is unchanged.
    """
    if role_weighting not in ("share", "uniform"):
        raise ValueError("role_weighting must be 'share' or 'uniform'")
    if role_weighting != "share" and task_terms is None:
        raise ValueError("role_weighting other than 'share' only reports task terms")
    if role_weighting != "share" and task_weights is not None:
        raise ValueError("task_weights use the training ('share') role weighting")
    B = len(batch_tensors["target_specialty"])
    # Stored rows may carry critic-only privileged arrays; the actor never sees them.
    output = actor(actor_inputs(batch_tensors))
    spec_mask, act_mask = _masks(batch_tensors, B)
    device = output["specialty_logits"].device

    def flag(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = batch_tensors.get(name)
        if value is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        return value.to(device).reshape(shape).bool()

    row_weight = batch_tensors.get("row_weight")
    row_weight = (
        torch.ones(B, dtype=torch.float32, device=device)
        if row_weight is None
        else row_weight.to(device).reshape(B).float()
    )
    if "row_quarantined" in batch_tensors:
        row_weight = row_weight * (~flag("row_quarantined", (B,))).float()
    spec_valid = flag("label_specialty_valid", (B,))
    role_valid = flag("label_activity_role_valid", (B, act_mask.shape[1]))

    # 1. Specialty loss (stable masked log-softmax).
    pred_spec = _masked_softmax(output["specialty_logits"], spec_mask)
    uniform_spec = spec_mask.float() / spec_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
    target_spec = batch_tensors["target_specialty"]
    smoothed_target_spec = (1.0 - label_smoothing) * target_spec + label_smoothing * uniform_spec
    spec_ce = masked_allocation_cross_entropy(
        output["specialty_logits"], smoothed_target_spec, spec_mask
    )
    spec_row_w = row_weight * spec_valid.float()
    if spec_weights is not None:
        sw = spec_weights.to(device)
        spec_row_w = spec_row_w * (target_spec * sw).sum(dim=-1).detach()
    spec_loss = _weighted_mean(spec_ce, spec_row_w)
    gauge_anchor_spec = 0.01 * (output["specialty_logits"][:, 0] ** 2).mean()
    total_spec_loss = spec_loss + gauge_anchor_spec
    spec_l1_rows = (torch.abs(pred_spec - target_spec) * spec_mask.float()).sum(dim=-1)
    spec_l1 = float(_weighted_mean(spec_l1_rows, row_weight * spec_valid.float()).item())

    # 2. Activity loss: sum_r (n_r/P) CE_r per row, reduced over weighted rows.
    pred_act = _masked_softmax(output["activity_logits"], act_mask)
    uniform_act = act_mask.float() / act_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
    target_act = batch_tensors["target_activity"]
    smoothed_target_act = (1.0 - label_smoothing) * target_act + label_smoothing * uniform_act
    act_role_ce = masked_allocation_cross_entropy(
        output["activity_logits"], smoothed_target_act, act_mask
    )  # (B, 13)
    legal_roles = act_mask.any(dim=-1)
    role_act_w = (
        (target_act * act_weights.to(device)).sum(dim=-1).detach()
        if act_weights is not None
        else torch.ones_like(act_role_ce)
    )
    base_role_weights = (
        (target_spec[:, 1:] + inactive_role_weight)
        * legal_roles.float()
        * role_valid.float()
        * spec_valid.float().unsqueeze(-1)
    )
    act_row = (act_role_ce * base_role_weights * role_act_w).sum(dim=-1)
    act_row_w = row_weight * (base_role_weights.sum(dim=-1) > 0).float()
    act_loss = _weighted_mean(act_row, act_row_w)
    gauge_anchor_act = 0.01 * (output["activity_logits"][..., 0] ** 2).mean()
    total_act_loss = act_loss + gauge_anchor_act
    act_role_l1 = (torch.abs(pred_act - target_act) * act_mask.float()).sum(dim=-1)
    act_l1 = float(
        (
            (act_role_l1 * base_role_weights * row_weight.unsqueeze(-1)).sum()
            / (base_role_weights * row_weight.unsqueeze(-1)).sum().clamp(min=1e-6)
        ).item()
    )

    # 3. Log std regularization (unchanged prototype objective).
    target_std_val = float(target_log_std)
    spec_std_loss = ((output["specialty_log_std"] - target_std_val) ** 2)[spec_mask].mean()
    act_std_loss = ((output["activity_log_std"] - target_std_val) ** 2)[act_mask].mean()
    std_reg_loss = spec_std_loss + act_std_loss

    # 4. Movement: unmatched or illegal targets are invalid, never index zero.
    codes = output["movement_codes"]
    target_codes = batch_tensors["target_movement_code"].to(device).reshape(B)
    union_mask = output["movement_union_mask"]
    matches = codes.unsqueeze(0) == target_codes.unsqueeze(1)  # (B, K)
    matched = matches.any(dim=1)
    target_idx = matches.float().argmax(dim=1)
    legal_target = union_mask.gather(1, target_idx.unsqueeze(1)).squeeze(1) & matched
    move_valid = flag("label_movement_valid", (B,)) & legal_target
    masked_movement_logits = output["movement_union_logits"].masked_fill(~union_mask, -1e9)
    move_ce = F.cross_entropy(masked_movement_logits, target_idx, reduction="none")
    move_w = row_weight * move_valid.float()
    move_loss = _weighted_mean(move_ce, move_w)

    # 5. Fission loss.
    fission_ce = F.cross_entropy(
        output["fission"], batch_tensors["target_fission"].to(device).reshape(B), reduction="none"
    )
    fission_w = row_weight * flag("label_fission_valid", (B,)).float()
    fission_loss = _weighted_mean(fission_ce, fission_w)

    # 6. Trade gates: every valid gate (including inactive gates), averaged per row.
    groups = len(contracts.TRADE_GROUPS)
    gate_logits = output["trade_gate"].view(B * groups, contracts.TRADE_GATES)
    gate_targets = batch_tensors["target_trade_gate"].to(device).view(-1)
    gate_ce = F.cross_entropy(gate_logits, gate_targets, reduction="none").view(B, groups)
    gate_valid = flag("label_trade_gate_valid", (B, groups)).float()
    gate_row = (gate_ce * gate_valid).sum(dim=-1) / gate_valid.sum(dim=-1).clamp(min=1.0)
    gate_w = row_weight * (gate_valid.sum(dim=-1) > 0).float()
    trade_gate_loss = _weighted_mean(gate_row, gate_w)

    # 7. Trade values: valid active groups only.
    value_active = flag("label_trade_value_active", (B, groups)).float()
    tv_err = (output["trade_value_mean"] - batch_tensors["target_trade_value_latent"].to(device)) ** 2
    tv_row = (tv_err * value_active).sum(dim=-1) / value_active.sum(dim=-1).clamp(min=1.0)
    tv_w = row_weight * (value_active.sum(dim=-1) > 0).float()
    tv_mean_loss = _weighted_mean(tv_row, tv_w)
    tv_std_loss = ((output["trade_value_log_std"] - (-1.5)) ** 2).mean()
    trade_value_loss = tv_mean_loss + tv_std_loss

    # 8. Trade share: only when at least one valid gate is active.
    share_active = flag("label_trade_share_active", (B,)).float()
    ts_err = (
        output["trade_share_mean"].reshape(B)
        - batch_tensors["target_trade_share_latent"].to(device).reshape(B)
    ) ** 2
    ts_mean_loss = _weighted_mean(ts_err, row_weight * share_active)
    ts_std_loss = ((output["trade_share_log_std"] - (-1.5)) ** 2).mean()
    trade_share_loss = ts_mean_loss + ts_std_loss

    if task_terms is None and task_weights is None:
        total_loss = (
            total_spec_loss
            + total_act_loss
            + std_reg_loss
            + move_loss
            + fission_loss
            + trade_gate_loss
            + trade_value_loss
            + trade_share_loss
        )
    else:
        from .labor_actor_gradnorm import ACTOR_BC_TASKS, ACTOR_ROLES, role_task

        if role_weighting == "uniform":
            staffed = (base_role_weights > 0).to(base_role_weights.dtype)
            per_row = staffed.sum(dim=-1, keepdim=True).clamp(min=1.0)
            role_base = staffed / per_row * base_role_weights.sum(dim=-1, keepdim=True)
        else:
            role_base = base_role_weights
        denominator = act_row_w.sum().clamp(min=1e-12)
        role_mass = role_base * role_act_w * row_weight.unsqueeze(-1)
        role_losses = (act_role_ce * role_mass).sum(dim=0) / denominator  # (13,)
        losses: dict[str, torch.Tensor] = {"specialty": spec_loss}
        for index, role in enumerate(ACTOR_ROLES):
            losses[role_task(role)] = role_losses[index]
        losses.update(movement=move_loss, fission=fission_loss, trade_gate=trade_gate_loss)
        if task_terms is not None:
            with torch.no_grad():
                staffed_rows = ((role_base > 0) & (row_weight.unsqueeze(-1) > 0)).sum(dim=0)
                mass = role_mass.sum(dim=0)
                mean_ce = (act_role_ce * role_mass).sum(dim=0) / mass.clamp(min=1e-12)
            task_terms["loss"] = losses
            task_terms["staffed_rows"] = {
                role_task(r): int(staffed_rows[i]) for i, r in enumerate(ACTOR_ROLES)
            }
            task_terms["mean_ce"] = {
                role_task(r): float(mean_ce[i]) for i, r in enumerate(ACTOR_ROLES)
            }
        if task_weights is None:
            total_loss = (
                total_spec_loss
                + total_act_loss
                + std_reg_loss
                + move_loss
                + fission_loss
                + trade_gate_loss
                + trade_value_loss
                + trade_share_loss
            )
        else:
            missing = set(ACTOR_BC_TASKS) - set(task_weights)
            if missing:
                raise ValueError(f"actor task weights miss {sorted(missing)}")
            weighted = sum(float(task_weights[name]) * losses[name] for name in ACTOR_BC_TASKS)
            total_loss = (
                weighted
                + gauge_anchor_spec
                + gauge_anchor_act
                + std_reg_loss
                + trade_value_loss
                + trade_share_loss
            )
    components = {
        "specialty": float(spec_loss.detach()),
        "activity": float(act_loss.detach()),
        "std_regularization": float(std_reg_loss.detach()),
        "movement": float(move_loss.detach()),
        "fission": float(fission_loss.detach()),
        "trade_gate": float(trade_gate_loss.detach()),
        "trade_value": float(tv_mean_loss.detach()),
        "trade_share": float(ts_mean_loss.detach()),
        "count_specialty": float((spec_row_w > 0).sum()),
        "count_activity_rows": float((act_row_w > 0).sum()),
        "count_movement": float((move_w > 0).sum()),
        "count_fission": float((fission_w > 0).sum()),
        "count_trade_gate_rows": float((gate_w > 0).sum()),
        "count_trade_value_rows": float((tv_w > 0).sum()),
        "count_trade_share": float(((row_weight * share_active) > 0).sum()),
        "weight_specialty": float(spec_row_w.sum()),
    }
    return total_loss, spec_l1, act_l1, components


def _evaluate_dataset(
    actor: LaborActor,
    dataset: Mapping[str, Any],
    *,
    batch_size: int,
    target_log_std: float,
    inactive_role_weight: float = 0.0,
    label_smoothing: float = 1e-6,
    spec_weights: torch.Tensor | None = None,
    act_weights: torch.Tensor | None = None,
) -> tuple[float, float, float, dict[str, float]]:
    actor.eval()
    device = module_device(actor)
    tensor_keys = [k for k, v in dataset.items() if isinstance(v, torch.Tensor)]
    N = dataset["target_specialty"].shape[0]
    total_loss_sum = 0.0
    spec_l1_sum = 0.0
    act_l1_sum = 0.0
    total_count = 0
    component_sums: dict[str, float] = defaultdict(float)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            batch = to_device({k: dataset[k][start:end] for k in tensor_keys}, device)
            count = end - start
            loss, spec_l1, act_l1, components = _batch_loss_and_l1(
                actor,
                batch,
                target_log_std=target_log_std,
                inactive_role_weight=inactive_role_weight,
                label_smoothing=label_smoothing,
                spec_weights=spec_weights,
                act_weights=act_weights,
            )
            total_loss_sum += float(loss.item()) * count
            spec_l1_sum += spec_l1 * count
            act_l1_sum += act_l1 * count
            total_count += count
            for key, value in components.items():
                component_sums[key] += value * (1 if key.startswith(("count_", "weight_")) else count)

    denominator = max(1, total_count)
    components = {
        key: (value if key.startswith(("count_", "weight_")) else value / denominator)
        for key, value in component_sums.items()
    }
    return (
        total_loss_sum / denominator,
        spec_l1_sum / denominator,
        act_l1_sum / denominator,
        components,
    )


def _actor_gradnorm_update(
    gradnorm: Any,
    terms: Mapping[str, Any],
    params: Sequence[nn.Parameter],
    step: int,
) -> dict[str, Any]:
    """One actor GradNorm controller update from the step's task terms (before backward).

    Raw task gradient norms on ``params`` (``torch.autograd.grad``: no ``.grad``
    accumulation, the graph is kept for the step's backward); a role task staffed in fewer
    than ``min_rows`` rows of the batch is not measured (held). A role's progress loss is its
    composition-normalised mean CE, every other task's its loss."""
    from .labor_actor_gradnorm import ACTOR_ROLE_TASKS

    config = gradnorm.config
    raw: dict[str, float] = {}
    losses: dict[str, float] = {}
    for name in gradnorm.tasks:
        value = terms["loss"][name]
        if name in ACTOR_ROLE_TASKS:
            if terms["staffed_rows"][name] < config.min_rows:
                continue
            losses[name] = float(terms["mean_ce"][name])
        else:
            losses[name] = float(value.detach())
        if not value.requires_grad:
            continue
        grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        raw[name] = math.sqrt(
            sum(float(g.detach().double().pow(2).sum()) for g in grads if g is not None)
        )
    entry = gradnorm.update(raw, losses, terms["staffed_rows"])
    return {"update_step": int(step), **entry}


LR_SCHEDULES = ("constant", "warmup-cosine")


def lr_schedule_factor(
    step: int, total_steps: int, warmup_fraction: float = 0.1, final_fraction: float = 0.1
) -> float:
    """Brief section 5.1 actor schedule: linear warm-up over the first ``warmup_fraction`` of
    ``total_steps`` (to the peak at the last warm-up step), then cosine decay to
    ``final_fraction`` of the peak at ``total_steps``; constant at that floor afterwards.

    ``step`` is the 0-based global optimizer step of the whole program (rounds and resumed
    fits continue it); the factor multiplies the peak learning rate."""
    total = max(1, int(total_steps))
    warmup = max(1, math.ceil(float(warmup_fraction) * total)) if warmup_fraction > 0 else 0
    if step < warmup:
        return float(step + 1) / float(warmup)
    progress = min(1.0, float(step - warmup) / float(max(1, total - warmup)))
    return float(final_fraction) + (1.0 - float(final_fraction)) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def pretrain_labor_actor_bc(
    *,
    actor: LaborActor,
    dataset: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    lr: float = 1e-3,
    epochs: int = 25,
    batch_size: int = 64,
    target_log_std: float = -2.0,
    max_grad_norm: float = 1.0,
    rng: np.random.Generator | None = None,
    inactive_role_weight: float = 0.0,
    label_smoothing: float = 1e-6,
    inverse_freq_power: float = 0.5,
    stability: Any = None,
    frozen_critics: Mapping[str, nn.Module] | None = None,
    family_schedule: Any = None,
    lr_schedule: Mapping[str, Any] | None = None,
    loss_balancing: Any = None,
) -> dict[str, Any]:
    """Supervised behavioral cloning on LaborActor using heuristic dataset.

    ``loss_balancing`` (None: the historical loss, bit for bit; or a
    :class:`~.labor_actor_gradnorm.ActorGradNorm`, ``pretrain-bc --loss-balancing gradnorm``)
    weights the BC tasks (specialty, the 13 role activity CEs, movement, fission, trade gate)
    by GradNorm on the shared encoder (or its last layer); the weights, gradient norms and
    losses join the stability log (``actor_gradnorm``) and the summary, and the monitor then
    judges ``gradient_dominance`` over the tasks on those parameters (block ``actor_tasks``).

    ``lr_schedule`` (None or kind ``constant``: the historical fixed ``lr``, bit for bit)
    ``{"kind": "warmup-cosine", "total_steps": T, "warmup_fraction": 0.1,
    "final_fraction": 0.1, "start_step": s}`` sets every parameter group's rate to
    ``lr x lr_schedule_factor(s + step, T, ...)`` before each optimizer step, ``lr`` being
    the peak (brief section 5.1: peak 3e-5, 10% warm-up, cosine to 10%). ``total_steps``
    defaults to this fit's ``epochs x ceil(rows / batch_size)``; ``start_step`` continues
    an earlier part of the program (a resumed fit or the next round).

    ``family_schedule`` (``labor_scenarios.FamilyWeightSchedule``, with a dataset
    reweighted by ``--family-weights``) decays each family up-weight toward 1 as
    1 + (w - 1) s(t), t the global optimizer step in epochs plus the schedule offset;
    the current multipliers are logged in the stability JSONL. Decay ``none`` keeps
    the loader's fixed weights.

    Rows from the reserved held-out worlds 90154-90159 and held-out-test data
    are refused (``labor_worlds``), exactly as for critic training.

    ``stability`` (a :class:`~.labor_stability.StabilityConfig`; the
    ``pretrain-bc`` CLI turns it on for r5 checkpoints) wraps every step in the
    enforced stability monitor; ``frozen_critics`` must then receive exactly
    zero gradient (no cross-network gradient, r5-b item 6).
    """
    if not isinstance(dataset, Mapping) or "target_specialty" not in dataset:
        raise ValueError("dataset must contain target_specialty and input tensors")
    N = int(dataset["target_specialty"].shape[0])
    if N == 0:
        raise ValueError("cannot pretrain on empty dataset")
    if "row_world" in dataset:
        require_training_worlds(dataset["row_world"].reshape(-1).tolist(), "labor actor BC")
    require_training_record(dataset.get("collection"), "labor actor BC")
    _refuse_critic_only_rows(dataset, "labor actor BC")

    opt = optimizer if optimizer is not None else torch.optim.Adam(actor.parameters(), lr=lr)
    device = module_device(actor)
    generator = rng if rng is not None else np.random.default_rng(0)
    tensor_keys = [k for k, v in dataset.items() if isinstance(v, torch.Tensor)]

    spec_weights: torch.Tensor | None = None
    act_weights: torch.Tensor | None = None
    if inverse_freq_power > 0.0:
        valid = dataset.get("label_specialty_valid")
        keep = (
            torch.ones(N, dtype=torch.bool)
            if valid is None
            else valid.reshape(N).bool()
        )
        if "row_quarantined" in dataset:
            keep = keep & ~dataset["row_quarantined"].reshape(N).bool()
        if not bool(keep.any()):
            keep = torch.ones(N, dtype=torch.bool)
        mean_spec = dataset["target_specialty"][keep].mean(dim=0)
        spec_weights = (mean_spec + 0.02).pow(-inverse_freq_power)
        spec_weights = spec_weights / spec_weights.mean().clamp(min=1e-8)

        mean_act = dataset["target_activity"][keep].mean(dim=0)
        act_weights = (mean_act + 0.02).pow(-inverse_freq_power)
        act_weights = act_weights / act_weights.mean().clamp(min=1e-8)
        # Computed on CPU (identical values on every device), then moved once.
        spec_weights = to_device(spec_weights, device)
        act_weights = to_device(act_weights, device)

    gradnorm = None
    gradnorm_params: list[nn.Parameter] = []
    gradnorm_history: list[dict[str, Any]] = []
    if loss_balancing is not None:
        from .labor_actor_gradnorm import (
            ActorGradNorm,
            ActorGradNormState,
            gradnorm_shared_parameters_for,
            role_resident_scales,
        )

        if not isinstance(loss_balancing, ActorGradNorm):
            raise ValueError("loss_balancing must be an ActorGradNorm or None")
        scale_keep = torch.ones(N, dtype=torch.bool)
        if "label_specialty_valid" in dataset:
            scale_keep = scale_keep & dataset["label_specialty_valid"].reshape(N).bool()
        if "row_quarantined" in dataset:
            scale_keep = scale_keep & ~dataset["row_quarantined"].reshape(N).bool()
        gradnorm = ActorGradNormState(
            loss_balancing,
            role_resident_scales(
                dataset["target_specialty"], dataset.get("row_weight"), scale_keep
            ),
        )
        gradnorm_params = gradnorm_shared_parameters_for(actor, loss_balancing.shared)

    evaluation = {
        "batch_size": batch_size,
        "target_log_std": target_log_std,
        "inactive_role_weight": inactive_role_weight,
        "label_smoothing": label_smoothing,
        "spec_weights": spec_weights,
        "act_weights": act_weights,
    }
    init_loss, spec_l1_before, act_l1_before, components_before = _evaluate_dataset(
        actor, dataset, **evaluation
    )

    from .labor_stability import StabilityMonitor, resolve_stability

    stability_config = resolve_stability(stability, r5=True)
    monitor = None
    if stability_config is not None:
        if optimizer is not None and {id(p) for g in opt.param_groups for p in g["params"]} != {
            id(p) for p in actor.parameters()
        }:
            raise ValueError("monitored BC needs an optimizer over exactly the actor parameters")
        blocks = {"actor_encoder": list(actor.encoder.parameters())}
        for name, module in actor.named_children():
            if name != "encoder":
                params = list(module.parameters())
                if params:
                    blocks[f"actor_head:{name}"] = params
        monitor = StabilityMonitor(
            stability_config,
            fit="pretrain-bc",
            optimizer=opt,
            parameters=list(actor.parameters()),
            blocks=blocks,
            frozen=dict(frozen_critics or {}),
            **(
                {}
                if gradnorm is None
                else {"shared_heads": {"actor_tasks": (gradnorm_params, gradnorm.tasks)}}
            ),
        )

    weighter = None
    steps_per_epoch = max(1, -(-N // batch_size))
    global_step = 0
    if family_schedule is not None and family_schedule.decay != "none":
        from .labor_scenarios import FamilyRowWeighter

        if "row_weight_unweighted" not in dataset or "row_family_tags" not in dataset:
            raise ValueError("a decaying family schedule needs a --family-weights dataset")
        weighter = FamilyRowWeighter(
            dataset["row_weight_unweighted"].reshape(-1).numpy(),
            dataset["row_family_tags"].reshape(-1).numpy(),
            dataset["row_source"].reshape(-1).numpy()
            if "row_source" in dataset
            else np.zeros(N, dtype=np.int64),
            dataset["row_world"].reshape(-1).numpy(),
            family_schedule.weights,
        )

    schedule = dict(lr_schedule or {"kind": "constant"})
    kind = str(schedule.get("kind", "constant"))
    if kind not in LR_SCHEDULES:
        raise ValueError(f"lr schedule must be one of {LR_SCHEDULES}")
    if kind != "constant":
        schedule.setdefault("total_steps", int(epochs) * steps_per_epoch)
        schedule.setdefault("warmup_fraction", 0.1)
        schedule.setdefault("final_fraction", 0.1)
        schedule.setdefault("start_step", 0)
        schedule["peak_lr"] = float(lr)
        if int(schedule["total_steps"]) < 1 or int(schedule["start_step"]) < 0:
            raise ValueError("lr schedule needs total_steps >= 1 and start_step >= 0")
        if not 0.0 <= float(schedule["warmup_fraction"]) < 1.0 or not (
            0.0 <= float(schedule["final_fraction"]) <= 1.0
        ):
            raise ValueError("lr schedule fractions out of range")
    applied_lrs: list[float] = []

    def apply_lr() -> None:
        if kind == "constant":
            return
        rate = float(lr) * lr_schedule_factor(
            int(schedule["start_step"]) + global_step,
            int(schedule["total_steps"]),
            float(schedule["warmup_fraction"]),
            float(schedule["final_fraction"]),
        )
        for group in opt.param_groups:
            group["lr"] = rate
        if not applied_lrs or len(applied_lrs) < 2:
            applied_lrs.append(rate)
        else:
            applied_lrs[1] = rate

    def schedule_diagnostics() -> dict[str, Any]:
        progress = global_step / steps_per_epoch
        return {
            "family_weight_progress_epochs": float(family_schedule.epoch_offset) + progress,
            "family_weight_factor": family_schedule.factor(progress),
            "family_weight_multipliers": family_schedule.multipliers(progress),
        }

    actor.train()
    for _ in range(epochs):
        indices = generator.permutation(N)
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            batch_idx = torch.as_tensor(indices[start:end], dtype=torch.long)
            cpu_batch = {k: dataset[k][batch_idx] for k in tensor_keys}
            if weighter is not None:
                s = family_schedule.factor(global_step / steps_per_epoch)
                cpu_batch["row_weight"] = torch.as_tensor(
                    weighter.rows(batch_idx.numpy(), s), dtype=torch.float32
                )
            batch = to_device(cpu_batch, device)

            terms: dict[str, Any] | None = None
            weights_used: dict[str, float] | None = None
            if gradnorm is not None:
                terms = {}
                weights_used = gradnorm.values()
            loss, _, _, _ = _batch_loss_and_l1(
                actor,
                batch,
                target_log_std=target_log_std,
                inactive_role_weight=inactive_role_weight,
                label_smoothing=label_smoothing,
                spec_weights=spec_weights,
                act_weights=act_weights,
                **({} if terms is None else {"task_terms": terms, "task_weights": weights_used}),
            )
            gradnorm_entry: dict[str, Any] | None = None
            gradnorm_saved: dict[str, Any] | None = None
            if gradnorm is not None and global_step % loss_balancing.every == 0:
                gradnorm_saved = gradnorm.snapshot()
                gradnorm_entry = _actor_gradnorm_update(
                    gradnorm, terms, gradnorm_params, global_step + 1
                )
                if not gradnorm_history or (global_step + 1) % 500 == 0:
                    gradnorm_history.append(
                        {"step": global_step + 1, "weights": gradnorm_entry["weights_next"]}
                    )
            apply_lr()
            global_step += 1
            if monitor is not None:
                opt.zero_grad()
                gradnorm_kwargs: dict[str, Any] = {}
                if gradnorm is not None:

                    def gradnorm_log(
                        effective: Mapping[str, Mapping[str, float]],
                        raw: Mapping[str, Mapping[str, float]],
                        entry: Mapping[str, Any] | None = gradnorm_entry,
                        used: Mapping[str, float] | None = weights_used,
                    ) -> dict[str, Any]:
                        """Log the task weights this step used and the controller update."""
                        fields: dict[str, Any] = {"actor_task_weights": dict(used or {})}
                        if entry is not None:
                            fields["actor_gradnorm"] = {**entry, "step": int(monitor.step_index)}
                        return fields

                    gradnorm_kwargs = {
                        "head_losses": terms["loss"],
                        "head_grad_scales": weights_used,
                        "log_hook": gradnorm_log,
                    }
                accepted = monitor.step(
                    loss,
                    max_grad_norm=max_grad_norm,
                    diagnostics=schedule_diagnostics if family_schedule is not None else None,
                    **gradnorm_kwargs,
                )
                if not accepted and gradnorm_saved is not None:
                    gradnorm.restore(gradnorm_saved)  # rolled back: weights unchanged
                continue
            if not torch.isfinite(loss):
                raise ValueError("labor BC produced a nonfinite loss")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            opt.step()

    final_loss, spec_l1_after, act_l1_after, components_after = _evaluate_dataset(
        actor, dataset, **evaluation
    )

    return {
        "contract": LABOR_BC_CONTRACT,
        "epochs": int(epochs),
        "rows": int(N),
        "initial_loss": float(init_loss),
        "final_loss": float(final_loss),
        "specialty_l1_before": float(spec_l1_before),
        "specialty_l1_after": float(spec_l1_after),
        "activity_l1_before": float(act_l1_before),
        "activity_l1_after": float(act_l1_after),
        "components_before": components_before,
        "components_after": components_after,
        "inactive_role_weight": float(inactive_role_weight),
        "label_smoothing": float(label_smoothing),
        "inverse_freq_power": float(inverse_freq_power),
        **(
            {}
            if kind == "constant"
            else {
                "lr_schedule": {
                    **schedule,
                    "steps_this_fit": int(global_step),
                    "end_step": int(schedule["start_step"]) + int(global_step),
                    "first_lr": applied_lrs[0] if applied_lrs else None,
                    "last_lr": applied_lrs[-1] if applied_lrs else None,
                }
            }
        ),
        "collection": copy.deepcopy(dataset.get("collection", {})),
        **(
            {}
            if gradnorm is None
            else {"loss_balancing": {**gradnorm.summary(), "history": gradnorm_history}}
        ),
        **({} if monitor is None else {"stability": monitor.finish()}),
        **(
            {}
            if family_schedule is None
            else {
                "family_weight_schedule": {
                    **family_schedule.state_dict(),
                    "epochs_end": float(family_schedule.epoch_offset)
                    + global_step / steps_per_epoch,
                    "final_multipliers": family_schedule.multipliers(
                        global_step / steps_per_epoch
                    ),
                }
            }
        ),
    }


# ---------------------------------------------------------------------------
# Critic pretraining on heuristic returns
# ---------------------------------------------------------------------------


# The only shard kind whose rewards are teacher-policy returns along one
# chronological (world, camp) trajectory. Equal to labor_teacher_data.SOURCE_KIND.
CRITIC_TEACHER_SOURCE_KIND = "native_heuristic_labor_teacher"
# Teacher (H12) continuations from learner-visited (DAgger) states, closed by the
# randomized tail: V^rho targets at learner states (labor_dagger, row_source 3).
CONTINUATION_SOURCE_KIND = "learner_state_teacher_continuation"
CONTINUATION_ROW_SOURCE = 3
# Curated Neolithic scenario trajectories (labor_scenarios): H12 teacher rows from
# keyed family roots, closed by the randomized tail (row_source 4).
SCENARIO_SOURCE_KIND = "curated_neolithic_scenario"
SCENARIO_ROW_SOURCE = 4
_CRITIC_REFUSALS = {
    "native_search_labor_teacher": (
        "search shards (row_source 2) are action labels, one per root and personality; their "
        "reward_* is only the label's first transition, not a teacher-policy return"
    ),
    "learner_state_dagger": (
        "actor-only (learner-state DAgger) data: its rewards follow the learner's short "
        "branches, not teacher-policy returns"
    ),
    "aggregated": (
        "aggregated multi-source data: critic targets need one teacher trajectory source"
    ),
    CONTINUATION_SOURCE_KIND: (
        "learner-state teacher continuations are V^rho targets only with "
        "--learner-state-continuations"
    ),
    SCENARIO_SOURCE_KIND: (
        "curated Neolithic scenario trajectories are V targets only with --curated-scenarios"
    ),
}


def require_critic_teacher_dataset(
    dataset: Mapping[str, Any],
    *,
    allow_learner_state_continuation: bool = False,
    allow_curated_scenarios: bool = False,
) -> None:
    """Refuse any dataset that is not teacher-policy returns.

    Search shards (``row_source == 2``), learner-state DAgger/actor-only shards
    and multi-source aggregates are rejected by source kind, by the
    ``actor_only`` flag and by any nonzero ``row_source`` row. A dataset
    without a source kind (in-process heuristic collection) is the teacher.

    ``allow_learner_state_continuation`` additionally admits teacher (H12)
    continuations from learner-visited states (``row_source`` 3, source kind
    :data:`CONTINUATION_SOURCE_KIND`), alone or aggregated with teacher
    trajectories, provided their randomized-tail arrays survived aggregation
    (a continuation row without its tail would be a one-step truncated return).

    ``allow_curated_scenarios`` likewise admits curated Neolithic scenario
    trajectories (``row_source`` 4, :data:`SCENARIO_SOURCE_KIND`), alone or
    aggregated with the sources above, under the same tail requirement.
    """
    collection = dataset.get("collection", {}) if isinstance(dataset, Mapping) else {}
    kind = collection.get("source_kind", CRITIC_TEACHER_SOURCE_KIND)
    if collection.get("actor_only"):
        raise ValueError(
            "pretrain-critic refuses actor-only (learner-state DAgger) data: its rewards "
            "follow the learner's short branches, not teacher-policy returns"
        )
    admitted = {CRITIC_TEACHER_SOURCE_KIND}
    codes_allowed = {0}
    if allow_learner_state_continuation:
        admitted.add(CONTINUATION_SOURCE_KIND)
        codes_allowed.add(CONTINUATION_ROW_SOURCE)
    if allow_curated_scenarios:
        admitted.add(SCENARIO_SOURCE_KIND)
        codes_allowed.add(SCENARIO_ROW_SOURCE)
    if (allow_learner_state_continuation or allow_curated_scenarios) and kind == "aggregated":
        components = {
            c.get("source_kind") for c in collection.get("component_collections", [])
        }
        if not components or not components <= admitted:
            raise ValueError(
                "pretrain-critic refuses aggregated data with components "
                f"{sorted(map(str, components - admitted))}: only heuristic-teacher "
                "trajectories, learner-state teacher continuations (--learner-state-"
                "continuations) and curated scenarios (--curated-scenarios) carry V targets"
            )
        kind = CRITIC_TEACHER_SOURCE_KIND
    if kind not in admitted:
        reason = _CRITIC_REFUSALS.get(kind, f"unknown source kind {kind!r}")
        raise ValueError(f"pretrain-critic refuses {kind} data: {reason}")
    sources = dataset.get("row_source") if isinstance(dataset, Mapping) else None
    if isinstance(sources, torch.Tensor) and not set(sources.reshape(-1).tolist()) <= codes_allowed:
        codes = sorted({int(v) for v in sources.reshape(-1).tolist()} - codes_allowed)
        raise ValueError(
            f"pretrain-critic refuses rows with row_source {codes}: only heuristic-teacher "
            "rows (row_source 0) carry critic targets; search rows are 2, DAgger rows 1, "
            "teacher continuations 3 (with --learner-state-continuations), curated scenarios 4 "
            "(with --curated-scenarios)"
        )
    if (
        isinstance(sources, torch.Tensor)
        and bool((sources.reshape(-1) == CONTINUATION_ROW_SOURCE).any())
        and (TAIL_METHOD_FIELD not in dataset or TRAJECTORY_FIELD not in dataset)
    ):
        raise ValueError(
            "learner-state teacher continuation rows lost their randomized tail or trajectory "
            "ids (aggregated with truncated-zero teacher data?); refusing one-step returns"
        )
    if isinstance(sources, torch.Tensor) and bool(
        (sources.reshape(-1) == SCENARIO_ROW_SOURCE).any()
    ):
        # Scenario windows may be declared truncated-zero (legacy fixture); a randomized
        # scenario set whose tails were dropped by aggregation is refused.
        declared = (
            {c.get("return_tail") for c in collection.get("component_collections", [])}
            if collection.get("source_kind") == "aggregated"
            else {collection.get("return_tail")}
        )
        if TRAJECTORY_FIELD not in dataset or (
            TAIL_METHOD_FIELD not in dataset and declared != {RETURN_TAIL_TRUNCATED_ZERO}
        ):
            raise ValueError(
                "curated scenario rows lost their randomized tail or trajectory ids "
                "(aggregated with truncated-zero teacher data?); refusing truncated returns"
            )


def _tail_settings_summary(collection: Mapping[str, Any]) -> dict[str, Any]:
    """Settings shared by every world's tail record (per-camp entries omitted)."""
    records = collection.get("return_tail_by_world") or {}
    keys = ("method", "estimator", "draws", "rho", "cap", "schedule_seed_base", "prefix_terms")
    shared = {tuple((k, json_safe(r.get(k))) for k in keys) for r in records.values()}
    if len(shared) > 1:
        raise ValueError("randomized-tail settings differ across worlds")
    summary: dict[str, Any] = dict(next(iter(shared))) if shared else {}
    counts: dict[str, int] = defaultdict(int)
    for record in records.values():
        for name, value in record.get("method_counts", {}).items():
            counts[name] += int(value)
    summary["method_counts"] = dict(sorted(counts.items()))
    return summary


def json_safe(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _qol_saturation_summary(
    task_critic: LaborTaskCritic,
    logits: torch.Tensor,
    *,
    target_fraction: torch.Tensor | None = None,
    early: torch.Tensor | None = None,
    thresholds: Any = None,
) -> dict[str, Any]:
    from .labor_advantage import qol_saturation_report

    return qol_saturation_report(
        task_critic, logits, target_fraction=target_fraction, early=early, thresholds=thresholds
    )


def _critic_resume_fingerprint(
    dataset: Mapping[str, Any],
    epochs: int,
    batch_size: int,
    lr: float,
    max_grad_norm: float,
    cost_loss_weight: float,
    rows: int,
) -> str:
    """What a resume state must match: data, schedule and admitted row count."""
    import hashlib
    import json

    collection = dataset.get("collection", {}) if isinstance(dataset, Mapping) else {}
    body = {
        "dataset_digest": collection.get("dataset_digest"),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "max_grad_norm": float(max_grad_norm),
        "cost_loss_weight": float(cost_loss_weight),
        "rows": int(rows),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def _probe_rows(kept: torch.Tensor, count: int) -> torch.Tensor:
    """Evenly spaced admitted rows (deterministic, no RNG)."""
    if len(kept) <= count:
        return kept
    positions = torch.linspace(0, len(kept) - 1, count).round().long()
    return kept[positions]


def _critic_stability_finish(
    monitor: Any,
    task_critic: LaborTaskCritic,
    cost_critics: LaborCostCritics,
    dataset: Mapping[str, Any],
    kept: torch.Tensor,
    tensor_keys: Sequence[str],
    saturation_final: Mapping[str, Any] | None,
    observed_pop_max: float,
) -> dict[str, Any]:
    """Final saturation, decoded population tail, attention mass and dead features."""
    from .labor_stability import (
        dead_feature_counts,
        population_tail_report,
        privileged_attention_mass,
    )

    if saturation_final is not None:
        monitor.record_final("qol_saturation", saturation_final["share_abs_z_gt_6"])
        if saturation_final.get("components") is not None:
            monitor.record_final("qol_saturation_detail", saturation_final["components"])
    tail_rows = _probe_rows(kept, 4096)
    device = module_device(task_critic)
    if not task_critic.is_bounded("population_years"):
        parts = []
        with torch.no_grad():
            for start in range(0, len(tail_rows), 512):
                index = tail_rows[start : start + 512]
                batch = to_device({k: dataset[k][index] for k in tensor_keys}, device)
                p = critic_privileged_batch(task_critic, dataset, index, device=device)
                parts.append(task_critic(batch, p)["population_years"])
        monitor.record_final(
            "population_tail",
            population_tail_report(task_critic, torch.cat(parts), observed_pop_max),
        )
    probe = _probe_rows(kept, monitor.config.probe_rows)
    batch = to_device({k: dataset[k][probe] for k in tensor_keys}, device)
    task_priv = critic_privileged_batch(task_critic, dataset, probe, device=device)
    cost_priv = critic_privileged_batch(cost_critics, dataset, probe, device=device)
    encoders = {name: (encoder, task_priv) for name, encoder in task_critic.task_encoders().items()}
    for name in LABOR_COSTS:
        encoders[f"cost:{name}"] = (cost_critics.encoders[name], cost_priv)
    monitor.record_probe(
        "privileged_attention",
        {
            name: privileged_attention_mass(encoder, batch, priv)
            for name, (encoder, priv) in encoders.items()
        },
    )
    dead_rows = _probe_rows(kept, max(monitor.config.probe_rows, 256))
    dead_batch = to_device({k: dataset[k][dead_rows] for k in tensor_keys}, device)
    dead_task = critic_privileged_batch(task_critic, dataset, dead_rows, device=device)
    dead_cost = critic_privileged_batch(cost_critics, dataset, dead_rows, device=device)
    monitor.record_probe(
        "dead_features",
        {
            name: dead_feature_counts(
                encoder,
                dead_batch,
                dead_cost if name.startswith("cost:") else dead_task,
                std=monitor.t.dead_unit_std,
            )
            for name, (encoder, _) in encoders.items()
        },
    )
    return monitor.finish()


def _stochastic_modules(*modules: nn.Module) -> list[str]:
    """Submodules whose train-mode forward is stochastic or updates state (refused for
    validation, which must not consume the RNG or change the fit)."""
    found = []
    for root in modules:
        for name, module in root.named_modules():
            dropout = (isinstance(module, nn.Dropout) and float(module.p) > 0.0) or (
                isinstance(module, nn.MultiheadAttention) and float(module.dropout) > 0.0
            )
            if dropout or isinstance(module, nn.modules.batchnorm._BatchNorm):
                found.append(name)
    return found


def _critic_validation_set(
    dataset: Mapping[str, Any],
    *,
    training_worlds: set[int],
    tensor_keys: Sequence[str],
    tail_method: str,
) -> dict[str, Any]:
    """Held-out validation rows of a critic fit (monitoring only, never training data).

    The same admission rule and return definition as the training rows; its worlds must be
    disjoint from the training worlds and may not be reserved held-out-test or sealed
    worlds (calibration worlds 90180-90199 are the intended source)."""
    from .labor_privileged import RESERVED_HELDOUT_WORLDS

    if return_tail_method(dataset) != tail_method:
        raise ValueError(
            f"validation dataset declares return tail {return_tail_method(dataset)!r}; the fit "
            f"uses {tail_method!r}"
        )
    missing = [k for k in tensor_keys if k not in dataset]
    if missing:
        raise ValueError(f"validation dataset lacks critic inputs {missing[:5]}")
    n = int(dataset["row_world"].shape[0])
    keep = ~dataset["row_quarantined"].reshape(n).bool() & critic_target_available(dataset)
    rows = torch.nonzero(keep).reshape(-1)
    if not len(rows):
        raise ValueError("validation dataset has no admitted rows")
    worlds = {int(w) for w in dataset["row_world"].reshape(-1)[rows].tolist()}
    overlap = sorted(worlds & set(training_worlds))
    if overlap:
        raise ValueError(f"validation worlds overlap the training worlds {overlap[:5]}")
    reserved = {int(w) for w in RESERVED_HELDOUT_WORLDS}
    banned = sorted(w for w in worlds if w in SEALED_WORLDS or w in reserved)
    if banned:
        raise ValueError(
            f"validation refuses reserved held-out-test or sealed worlds {banned[:5]}"
        )
    returns = discounted_returns(dataset)
    return {
        "dataset": dataset,
        "rows": rows,
        "worlds": sorted(worlds),
        "task_targets": {name: returns[name][rows] for name in LABOR_OBJECTIVES},
        "cost_targets": {name: returns[name][rows] for name in LABOR_COSTS},
    }


@dataclass(frozen=True)
class AutoQoLLossWeight:
    """``--qol-loss-weight auto``: a GradNorm-lite controller of the QoL task-loss weight.

    Every stability log step (``--stability-every``) the monitor measures each task head's
    unweighted gradient norm on the shared task encoder (``shared_head_grad_norm``); with
    raw norms q (QoL) and p (population-years, weight w_p) the effective QoL share at weight
    w is w q / (w q + w_p p), the dominance detector's share definition. The weight that
    puts the share exactly at ``target_share`` is w* = s / (1 - s) * w_p p / q (w* = max if
    q = 0). The applied weight follows a log-space EMA,
    ln w <- (1 - ema) ln w + ema ln clamp(w*, min, max), clamped to [min, max], and takes
    effect from the next optimizer step. Pure float arithmetic on monitor readings: no RNG,
    no effect on the step that was measured, persisted in the exact resume state.
    """

    target_share: float = 0.3
    ema: float = 0.3
    minimum: float = 1.0
    maximum: float = 200.0
    initial: float = 1.0

    def __post_init__(self) -> None:
        for name in ("target_share", "ema", "minimum", "maximum", "initial"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"auto QoL weight {name} must be finite")
        if not 0.0 < self.target_share < 1.0:
            raise ValueError("auto QoL weight target share must be in (0, 1)")
        if not 0.0 < self.ema <= 1.0:
            raise ValueError("auto QoL weight EMA must be in (0, 1]")
        if not 0.0 < self.minimum <= self.maximum:
            raise ValueError("auto QoL weight needs 0 < minimum <= maximum")
        if not self.minimum <= self.initial <= self.maximum:
            raise ValueError("auto QoL weight initial value must lie in [minimum, maximum]")

    def state_dict(self) -> dict[str, float]:
        return {
            "target_share": float(self.target_share),
            "ema": float(self.ema),
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
            "initial": float(self.initial),
        }

    def clamp(self, value: float) -> float:
        return min(self.maximum, max(self.minimum, float(value)))

    def update(
        self, weight: float, qol_norm: float, population_norm: float, population_weight: float
    ) -> dict[str, float]:
        """One controller update from raw (unweighted) encoder gradient norms."""
        q = max(0.0, float(qol_norm))
        p = max(0.0, float(population_weight) * float(population_norm))
        share = (weight * q) / (weight * q + p) if weight * q + p > 0.0 else 0.0
        ideal = (
            self.maximum
            if q <= 0.0
            else self.target_share / (1.0 - self.target_share) * p / q
        )
        target = self.clamp(ideal) if ideal > 0.0 else self.minimum
        updated = self.clamp(
            math.exp((1.0 - self.ema) * math.log(weight) + self.ema * math.log(target))
        )
        return {
            "qol_share_effective": share,
            "qol_weight_ideal": float(ideal),
            "qol_weight_target": target,
            "qol_weight_next": updated,
        }


GRAD_CLIP_MODES = ("global", "per-encoder")


@dataclass(frozen=True)
class GradNormBalancing:
    """``--loss-balancing gradnorm``: GradNorm (Chen et al. 2018) over the task heads.

    Learnable per-head weights w_i (renormalised to sum to the head count K after every
    update) on the shared task encoder's last layer W (:func:`gradnorm_shared_parameters`).
    Each optimizer step measures the unweighted head gradient norms g_i = ||d L_i / d W||
    and G_i = w_i g_i; with loss ratios Lt_i = L_i(t) / L_i(0) (L_i(0) the first step's batch
    losses) and relative inverse training rates r_i = Lt_i / mean(Lt), the target is
    mean(G) r_i^alpha, held constant; the weight loss sum_i |G_i - target_i| has gradient
    sign(G_i - target_i) g_i in w_i, applied by the weights' own Adam (``lr``), then
    w <- max(w, floor), w <- K w / sum(w). The weights a step used are those of the previous
    update; a rolled-back step leaves the weights and their optimizer unchanged. The progress
    losses (raw head means, or the QoL excess BCE under ``--dominance-excess-loss``) are the
    same series the stability detectors read. No RNG; all weight arithmetic is float64 on CPU.
    """

    alpha: float = 1.5
    lr: float = 0.025
    floor: float = 1e-3

    def __post_init__(self) -> None:
        for name in ("alpha", "lr", "floor"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"gradnorm {name} must be finite")
        if self.alpha < 0.0:
            raise ValueError("gradnorm alpha must be non-negative")
        if self.lr <= 0.0:
            raise ValueError("gradnorm lr must be positive")
        if not 0.0 < self.floor < 1.0:
            raise ValueError("gradnorm weight floor must be in (0, 1)")

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": "gradnorm",
            "alpha": float(self.alpha),
            "lr": float(self.lr),
            "floor": float(self.floor),
        }


def gradnorm_shared_parameters(task_critic: nn.Module) -> list[nn.Parameter]:
    """GradNorm's W: the weight of the last ``nn.Linear`` of the shared task-encoder trunk."""
    trunk = getattr(task_critic.encoder, "trunk", None)
    linears = [] if trunk is None else [m for m in trunk.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("gradnorm needs a task encoder with a linear trunk")
    return [linears[-1].weight]


class GradNormState:
    """Runtime GradNorm weights, their optimizer and initial losses (exact resume state)."""

    def __init__(
        self, config: GradNormBalancing, heads: Sequence[str], initial: Mapping[str, float]
    ) -> None:
        self.config = config
        self.heads = tuple(heads)
        values = torch.tensor([float(initial[h]) for h in self.heads], dtype=torch.float64)
        values = values * (len(self.heads) / values.sum())
        self.weights = nn.Parameter(values)
        self.optimizer = torch.optim.Adam([self.weights], lr=float(config.lr))
        self.initial_losses: dict[str, float] | None = None

    def values(self) -> dict[str, float]:
        values = self.weights.detach().tolist()
        return {h: float(v) for h, v in zip(self.heads, values, strict=True)}

    def proposal(
        self, raw_norms: Mapping[str, float], losses: Mapping[str, float]
    ) -> dict[str, Any]:
        """The update's quantities at the current weights (pure)."""
        initial = self.initial_losses or {h: float(losses[h]) for h in self.heads}
        w = self.weights.detach().tolist()
        g = [max(0.0, float(raw_norms[h])) for h in self.heads]
        big_g = [wi * gi for wi, gi in zip(w, g, strict=True)]
        mean_g = sum(big_g) / len(big_g)
        ratio = [
            float(losses[h]) / initial[h] if initial[h] > 0.0 else 1.0 for h in self.heads
        ]
        mean_ratio = sum(ratio) / len(ratio)
        rate = [r / mean_ratio if mean_ratio > 0.0 else 1.0 for r in ratio]
        target = [mean_g * max(0.0, ri) ** float(self.config.alpha) for ri in rate]
        grad = [
            (0.0 if gi_big == ti else math.copysign(1.0, gi_big - ti)) * gi
            for gi_big, ti, gi in zip(big_g, target, g, strict=True)
        ]
        named = lambda values: dict(zip(self.heads, values, strict=True))  # noqa: E731
        return {
            "weights": named(w),
            "raw_grad_norm": named(g),
            "weighted_grad_norm": named(big_g),
            "mean_grad_norm": mean_g,
            "loss": {h: float(losses[h]) for h in self.heads},
            "initial_loss": dict(initial),
            "loss_ratio": named(ratio),
            "inverse_rate": named(rate),
            "target_grad_norm": named(target),
            "weight_grad": named(grad),
            "gradnorm_loss": sum(abs(a - b) for a, b in zip(big_g, target, strict=True)),
        }

    def snapshot(self) -> dict[str, Any]:
        return self.state_dict()

    def commit(self, proposal: Mapping[str, Any]) -> dict[str, float]:
        """Apply one weight update; returns the renormalised weights for the next step."""
        if self.initial_losses is None:
            self.initial_losses = dict(proposal["initial_loss"])
        self.optimizer.zero_grad(set_to_none=True)
        self.weights.grad = torch.tensor(
            [proposal["weight_grad"][h] for h in self.heads], dtype=torch.float64
        )
        self.optimizer.step()
        with torch.no_grad():
            self.weights.clamp_(min=float(self.config.floor))
            self.weights.mul_(len(self.heads) / self.weights.sum())
        self.optimizer.zero_grad(set_to_none=True)
        return self.values()

    def state_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.detach().clone(),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "initial_losses": None if self.initial_losses is None else dict(self.initial_losses),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        with torch.no_grad():
            self.weights.copy_(state["weights"])
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        initial = state["initial_losses"]
        self.initial_losses = None if initial is None else {k: float(v) for k, v in initial.items()}


def pretrain_critic_on_dataset(
    *,
    task_critic: LaborTaskCritic,
    cost_critics: LaborCostCritics,
    dataset: Mapping[str, Any],
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_grad_norm: float = 1.0,
    rng: np.random.Generator | None = None,
    cost_loss_weight: float = 1.0,
    require_randomized_tail: bool = False,
    allow_learner_state_continuation: bool = False,
    allow_curated_scenarios: bool = False,
    family_schedule: Any = None,
    stability: Any = "auto",
    frozen_actor: nn.Module | None = None,
    max_steps: int | None = None,
    resume_path: str | None = None,
    resume_every: int = 100,
    progress: bool = False,
    td_refreshes_per_epoch: int = 4,
    td_refresh_steps: int | None = None,
    validation_dataset: Mapping[str, Any] | None = None,
    task_loss_weights: Mapping[str, float] | None = None,
    qol_loss_weight_auto: AutoQoLLossWeight | None = None,
    best_validation_sink: dict[str, Any] | None = None,
    qol_head_bias_init: str | None = None,
    td_step0_targets: str | None = None,
    loss_balancing: GradNormBalancing | None = None,
    grad_clip: str = "global",
) -> dict[str, Any]:
    """Fit task/cost critics on heuristic returns closed by the declared tail, world-balanced.

    Loss balancing and clipping (user-approved 29 Oct; both default off, the historical fit
    bitwise): ``loss_balancing`` (:class:`GradNormBalancing`) learns the task-head weights by
    GradNorm on the shared task encoder's last layer, starting from the ratio of
    ``task_loss_weights`` (``--qol-loss-weight``) renormalised to sum to the head count; its
    weights, initial losses and optimizer state are in the resume payload and fingerprint,
    and under the stability monitor ``gradient_dominance`` is advisory and
    ``task_progress_stall`` gates. ``grad_clip="per-encoder"`` clips the task critic (task
    encoder and heads) and each cost critic (encoder and head) to ``max_grad_norm``
    separately instead of one global clip.

    QoL level fixes (reviewer tweak 28 Oct, Fable audit section 2; both default off, which is
    the historical fit bitwise): ``qol_head_bias_init="mean-target"`` sets the logistic QoL
    head's output bias to logit(mean p*) before the first step, p* = clamp(G / (scale W), 0,
    1) the Monte Carlo target fraction and the mean taken under the fit's own minibatch
    sampling probabilities (deterministic; recorded with the unweighted mean). It applies at
    the start of a fresh fit only (a resume restores the parameters).
    ``td_step0_targets="mc"`` keeps the Monte Carlo returns as the TD(lambda) QoL targets
    until the first scheduled refresh instead of bootstrapping them through the untrained
    head at step 0; later refreshes are unchanged, and the refresh that first replaces the
    Monte Carlo targets is, like the historical step-0 refresh, not growth-eligible.
    A dataset normalised by :func:`normalize_world_tails` is fitted as given; its record is
    copied into the summary (``tail_normalization``) and the resume fingerprint.

    Task-head balance (r5 critic balance; every default keeps the historical fit bitwise):
    ``task_loss_weights`` multiplies each task head's row loss in the training objective
    (missing heads 1.0; all-1.0 is the unweighted historical mean). ``qol_loss_weight_auto``
    (:class:`AutoQoLLossWeight`, needs the stability monitor) adapts the QoL weight from the
    monitor's shared-encoder gradient shares. Weights act on the optimizer objective only:
    every reported loss (step head losses, validation, initial/final) stays unweighted, and
    the weight in force is logged on every step record (``task_loss_weights``).
    ``qol_excess_bce`` (BCE - H(p*), the logistic QoL head's readable loss) is logged in
    step, validation and final records. With ``validation_dataset``, the epoch of lowest
    validation ``total`` is reported under ``validation.best`` and, if
    ``best_validation_sink`` is a dict, its critic states are written into it
    (``task``/``costs``/``epoch``); tracking reads parameters only (trajectory unchanged).

    ``validation_dataset`` (monitoring only, never training data): at each completed epoch
    the same loss components are computed on its admitted rows (no gradient, fixed row
    order, torch RNG forked and restored, trainer generator untouched) and logged as
    ``kind=validation``; the non-gating ``validation_divergence`` alarm fires when the
    validation total rises on 2 consecutive epochs while the epoch training loss (mean of
    its accepted minibatch losses) falls. The fit's arithmetic is unchanged, so the fitted
    parameters are bitwise identical with or without it (tested on CPU).

    TD(lambda) QoL targets (bounded head with ``target="td-lambda"``) refresh every
    ``td_refresh_steps`` optimizer steps, or, if that is None, ``td_refreshes_per_epoch``
    times per epoch (default 4, about 100 steps at 400 steps per epoch); the refresh
    cadence is a pure function of the global step, so exact resume is preserved.

    Returns are computed over every row of each (world, camp) trajectory;
    rows whose initial recommendation was rejected (an idle retry executed)
    are excluded from fitting because their continuation is not the
    heuristic's. Minibatches are drawn with world-balanced probabilities.

    ``stability`` (``labor_stability``): ``"auto"`` (default) enables the
    enforced stability monitor for r5 critics (privileged block or bounded QoL
    head); a :class:`~.labor_stability.StabilityConfig` enables it explicitly,
    ``None`` disables it. A monitored fit without failures is bitwise identical
    to an unmonitored one; its receipt is returned under ``"stability"``.
    ``frozen_actor`` is checked to receive exactly zero gradient. ``max_steps``
    stops after that many optimizer steps (determinism probes). ``resume_path``
    writes an exact mid-fit resume state every ``resume_every`` steps
    (atomic ``.tmp`` then replace, the ``tools/g0launch.py`` partial pattern)
    and, if the file exists at start, continues from it bit-for-bit.
    """
    if not math.isfinite(cost_loss_weight) or cost_loss_weight < 0.0:
        raise ValueError("cost_loss_weight must be finite and non-negative")
    if qol_head_bias_init not in (None, "mean-target"):
        raise ValueError("qol_head_bias_init must be None or 'mean-target'")
    if td_step0_targets not in (None, "mc"):
        raise ValueError("td_step0_targets must be None or 'mc'")
    weight_values = {name: 1.0 for name in LABOR_OBJECTIVES}
    if task_loss_weights is not None:
        unknown = sorted(set(task_loss_weights) - set(LABOR_OBJECTIVES))
        if unknown:
            raise ValueError(f"task loss weights name unknown heads {unknown}")
        for name, value in task_loss_weights.items():
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"task loss weight {name} must be finite and positive")
            weight_values[name] = value
    separate_qol = bool(getattr(task_critic, "separate_qol_encoder", False))
    if qol_loss_weight_auto is not None and separate_qol:
        # The controller targets QoL's share of the *shared* task-encoder gradient; with a
        # separate QoL encoder no parameter is shared between the task heads.
        raise ValueError(
            "--qol-loss-weight auto balances QoL's share of a shared task encoder; this "
            "critic has a separate QoL encoder (use a fixed --qol-loss-weight or none)"
        )
    if qol_loss_weight_auto is not None:
        if task_loss_weights is not None and "qol_years" in task_loss_weights:
            raise ValueError("give the QoL weight either explicitly or as auto, not both")
        weight_values["qol_years"] = float(qol_loss_weight_auto.initial)
    # All-1.0 fixed weights are the historical unweighted objective (bitwise default path).
    if grad_clip not in GRAD_CLIP_MODES:
        raise ValueError(f"grad_clip must be one of {GRAD_CLIP_MODES}")
    if loss_balancing is not None:
        if not isinstance(loss_balancing, GradNormBalancing):
            raise ValueError("loss_balancing must be a GradNormBalancing or None")
        if qol_loss_weight_auto is not None:
            raise ValueError(
                "--loss-balancing gradnorm and --qol-loss-weight auto exclude each other"
            )
        if separate_qol:
            raise ValueError(
                "--loss-balancing gradnorm balances the heads on a shared task encoder; this "
                "critic has a separate QoL encoder"
            )
    weighted = (
        qol_loss_weight_auto is not None
        or loss_balancing is not None
        or any(v != 1.0 for v in weight_values.values())
    )
    weight_history: list[dict[str, Any]] = []
    require_critic_teacher_dataset(
        dataset,
        allow_learner_state_continuation=allow_learner_state_continuation,
        allow_curated_scenarios=allow_curated_scenarios,
    )
    require_training_record(dataset.get("collection"), "heuristic critic pretraining")
    if require_randomized_tail and return_tail_method(dataset) != RETURN_TAIL_RANDOMIZED:
        raise ValueError(
            "pretrain-critic requires randomized-tail targets; this dataset declares "
            f"{return_tail_method(dataset)!r}"
        )
    tail_method = return_tail_method(dataset)
    returns = discounted_returns(dataset)
    n = int(dataset["row_world"].shape[0])
    available = critic_target_available(dataset)
    quarantined = dataset["row_quarantined"].reshape(n).bool()
    keep = ~quarantined & available
    if not bool(keep.any()):
        raise ValueError("heuristic critic dataset has no admitted rows")
    kept = torch.nonzero(keep).reshape(-1)
    require_critic_training_worlds(
        dataset["row_world"].reshape(-1)[kept].tolist(), "heuristic critic pretraining"
    )
    tensor_keys = [
        k
        for k, v in actor_inputs(dataset).items()
        if isinstance(v, torch.Tensor)
        and not k.startswith(("target_", "label_", "reward_", "transition_", "row_", "tail_"))
    ]
    tensors = {k: dataset[k][kept] for k in tensor_keys}
    # Critic-only privileged batch: its own dictionary, never merged into ``tensors``.
    # Data without privileged arrays is rejected for a privileged critic, never zero-filled.
    privileged = critic_privileged_batch(task_critic, dataset, kept)
    cost_privileged = critic_privileged_batch(cost_critics, dataset, kept)
    if (privileged is None) != (cost_privileged is None):
        raise ValueError("task and cost critics disagree on the privileged block")

    def privileged_rows(index: torch.Tensor) -> dict[str, torch.Tensor] | None:
        return None if privileged is None else {k: v[index] for k, v in privileged.items()}
    task_targets = {name: returns[name][kept] for name in LABOR_OBJECTIVES}
    cost_targets = {name: returns[name][kept] for name in LABOR_COSTS}
    scenario_rows = "row_source" in dataset and bool(
        (dataset["row_source"].reshape(-1)[kept] == SCENARIO_ROW_SOURCE).any()
    )
    if scenario_rows:
        # Curated scenarios are their own source: equal source mass, world-balanced within
        # each source (labor_teacher_data.source_world_balanced_weights). Without scenario
        # rows the sampling is unchanged (world-balanced).
        from .labor_teacher_data import source_world_balanced_weights

        weights = source_world_balanced_weights(
            dataset["row_source"].reshape(-1)[kept], dataset["row_world"].reshape(-1)[kept]
        ).double().numpy()
    else:
        weights = world_balanced_weights(dataset["row_world"][kept]).double().numpy()
    family_weighting = (dataset.get("collection") or {}).get("family_weighting")
    critic_family_shares = None
    critic_weighter = None
    if family_weighting is not None and "row_family_tags" in dataset:
        # Family up-weighting (labor_scenarios): the same multipliers on the critic sampling
        # weights of the admitted rows, rescaled per (source, world).
        from .labor_scenarios import apply_family_weights, family_share_report

        tags = dataset["row_family_tags"].reshape(-1)[kept].numpy()
        sources = (
            dataset["row_source"].reshape(-1)[kept].numpy()
            if "row_source" in dataset
            else np.zeros(len(kept), dtype=np.int64)
        )
        base = weights
        weights = apply_family_weights(
            base, tags, sources, dataset["row_world"].reshape(-1)[kept].numpy(),
            family_weighting["weights"],
        )
        critic_family_shares = family_share_report(base, weights, tags)
        if family_schedule is not None and family_schedule.decay != "none":
            from .labor_scenarios import FamilyRowWeighter

            # Decayed schedule: per-step probabilities from the closed-form weighter.
            critic_weighter = FamilyRowWeighter(
                base, tags, sources, dataset["row_world"].reshape(-1)[kept].numpy(),
                family_schedule.weights,
            )
    elif family_schedule is not None:
        raise ValueError("a family weight schedule needs a --family-weights dataset")
    probabilities = weights / weights.sum()
    N = len(kept)

    critic_params = list(task_critic.parameters()) + list(cost_critics.parameters())
    optimizer = torch.optim.Adam(critic_params, lr=lr)
    gradnorm = None
    gradnorm_params: list[nn.Parameter] = []
    if loss_balancing is not None:
        gradnorm = GradNormState(loss_balancing, LABOR_OBJECTIVES, weight_values)
        weight_values.update(gradnorm.values())
        gradnorm_params = gradnorm_shared_parameters(task_critic)
    clip_groups = None
    if grad_clip == "per-encoder":
        # The task critic (encoder(s) and heads) and each cost critic (its own encoder and
        # head) are clipped on their own: no group's norm can shrink another's update.
        clip_groups = {"task": list(task_critic.parameters())}
        for name in LABOR_COSTS:
            clip_groups[f"cost:{name}"] = list(cost_critics.encoders[name].parameters()) + list(
                cost_critics.heads[name].parameters()
            )
        grouped = [id(p) for params in clip_groups.values() for p in params]
        if set(grouped) != {id(p) for p in critic_params} or len(grouped) != len(critic_params):
            raise ValueError("per-encoder clip groups must partition the critic parameters")
    # Modules decide the device (labor_device); data stays in host memory and each
    # chunk/minibatch moves to the device. On CPU every move is a no-op (bitwise path).
    device = module_device(task_critic)
    if module_device(cost_critics) != device:
        raise ValueError("task and cost critics are on different devices")

    excess_sum: list[float | None] = [None]  # QoL excess BCE of the last losses() pass

    def qol_excess(
        output: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | None:
        """Per-row BCE - H(p*) of a logistic bounded QoL head, else None (no gradient)."""
        excess_of = getattr(task_critic, "qol_excess_bce", None)
        if excess_of is None or getattr(task_critic, "qol_head", None) is None:
            return None
        return excess_of(output["qol_years"], targets["qol_years"])

    def losses(chunk: int = 512) -> tuple[float, float]:
        # Chunked: one forward pass over every row needed ~100 GB of activations at 96k rows.
        # labor_task_value_losses()[0] is labor_task_value_loss (same operations); its head
        # outputs also give the QoL excess BCE without a second forward pass.
        task_sum = cost_sum = 0.0
        excess: float | None = None
        with torch.no_grad():
            for start in range(0, N, chunk):
                index = torch.arange(start, min(N, start + chunk))
                t = to_device({k: v[index] for k, v in tensors.items()}, device)
                count = len(index)
                p = to_device(privileged_rows(index), device)
                chunk_targets = to_device({k: v[index] for k, v in task_targets.items()}, device)
                chunk_loss, _, chunk_output = labor_task_value_losses(
                    task_critic, t, chunk_targets, p
                )
                task_sum += count * float(chunk_loss)
                rows_excess = qol_excess(chunk_output, chunk_targets)
                if rows_excess is not None:
                    excess = (excess or 0.0) + float(rows_excess.sum())
                cost_sum += count * float(
                    labor_cost_value_loss(
                        cost_critics,
                        t,
                        to_device({k: v[index] for k, v in cost_targets.items()}, device),
                        p,
                    )
                )
        excess_sum[0] = None if excess is None else excess / N
        return task_sum / N, cost_sum / N

    # Bounded QoL head (amendment r5-b item 4): saturation monitor every fit,
    # and optional TD(lambda) targets bootstrapped through the head.
    qol_head = getattr(task_critic, "qol_head", None)

    def qol_logits(rows: torch.Tensor, chunk: int = 512) -> torch.Tensor:
        parts = []
        with torch.no_grad():
            for start in range(0, len(rows), chunk):
                index = rows[start : start + chunk]
                batch = to_device({k: dataset[k][index] for k in tensor_keys}, device)
                p = critic_privileged_batch(task_critic, dataset, index, device=device)
                # Host copies: TD targets and saturation statistics stay on CPU.
                parts.append(task_critic(batch, p)["qol_years"].cpu())
        return torch.cat(parts)

    from .labor_stability import (
        StabilityMonitor,
        is_r5_critic,
        population_tail_report,
        resolve_stability,
    )

    config = resolve_stability(stability, r5=is_r5_critic(task_critic))
    monitor = None
    if config is not None:
        blocks = {"task_encoder": list(task_critic.encoder.parameters())}
        if separate_qol:
            blocks["qol_encoder"] = list(task_critic.qol_encoder.parameters())
        for name in LABOR_OBJECTIVES:
            blocks[f"task_head:{name}"] = list(task_critic.heads[name].parameters())
        for name in LABOR_COSTS:
            blocks[f"cost_encoder:{name}"] = list(cost_critics.encoders[name].parameters())
            blocks[f"cost_head:{name}"] = list(cost_critics.heads[name].parameters())
        task_blocks = [
            "task_encoder",
            *(["qol_encoder"] if separate_qol else []),
            *(f"task_head:{n}" for n in LABOR_OBJECTIVES),
        ]
        # Shared-head dominance needs two heads on one block. With a separate QoL encoder
        # the population encoder serves population-years alone (and the QoL encoder QoL
        # alone), so no shared-trunk detector exists; each encoder's gradient norm is still
        # logged per step (grad_norm_preclip task_encoder / qol_encoder) and the task
        # critic as a whole stays a contender of the global-clip detector.
        shared = (
            {}
            if separate_qol
            else {"task_encoder": (blocks["task_encoder"], tuple(LABOR_OBJECTIVES))}
        )
        monitor = StabilityMonitor(
            config,
            fit="pretrain-critic",
            optimizer=optimizer,
            parameters=critic_params,
            blocks=blocks,
            frozen={} if frozen_actor is None else {"actor": frozen_actor},
            shared_heads=shared,
            contender_blocks={
                "global_clip": {
                    "task": task_blocks,
                    **{
                        name: [f"cost_encoder:{name}", f"cost_head:{name}"]
                        for name in LABOR_COSTS
                    },
                }
            },
            applicable={
                "qol_saturation": qol_head is not None,
                "population_tail": not task_critic.is_bounded("population_years"),
                "privileged_attention": task_critic.privileged_layout is not None,
                "dead_features": True,
            },
            **({} if loss_balancing is None else {"loss_balancing": "gradnorm"}),
            **({} if clip_groups is None else {"clip_groups": clip_groups}),
        )
    if qol_loss_weight_auto is not None and monitor is None:
        raise ValueError("--qol-loss-weight auto reads the stability monitor; enable it")
    excess_progress = monitor is not None and monitor.config.dominance_excess_loss
    if excess_progress and (qol_head is None or qol_head.output != "logistic"):
        raise ValueError("--dominance-excess-loss needs a logistic bounded QoL head")
    if excess_progress and separate_qol:
        import warnings

        warnings.warn(
            "--dominance-excess-loss has no effect with a separate QoL encoder: no shared-"
            "trunk dominance detector judges QoL progress (qol_excess_bce is still logged)",
            stacklevel=2,
        )
    observed_pop_max = float(
        (
            task_targets["population_years"]
            / LABOR_OBJECTIVE_REFERENCE_SCALES["population_years"]
        ).max()
    )

    sat_early = sat_thresholds = None
    if qol_head is not None:
        from .labor_advantage import early_decision_mask
        from .labor_stability import StabilityThresholds

        sat_thresholds = config.thresholds if config is not None else StabilityThresholds()
        sat_early = early_decision_mask(dataset, sat_thresholds.saturation_early_decisions)[kept]

    def saturation_summary() -> dict[str, Any]:
        fraction = task_targets["qol_years"].reshape(-1).double() / float(
            qol_head.discount_weight
        )
        return _qol_saturation_summary(
            task_critic,
            qol_logits(kept),
            target_fraction=fraction,
            early=sat_early,
            thresholds=sat_thresholds,
        )

    steps_per_epoch = max(1, -(-N // batch_size))
    if td_refresh_steps is not None:
        td_interval = int(td_refresh_steps)
    else:
        td_interval = max(1, -(-steps_per_epoch // max(1, int(td_refreshes_per_epoch))))
    if td_interval < 1:
        raise ValueError("TD refresh cadence must be at least one optimizer step")

    # True while the QoL targets are still the Monte Carlo returns (td_step0_targets="mc"
    # before the first refresh); the refresh that replaces them is not growth-eligible.
    mc_pending = [False]

    def refresh_td_targets(epoch: int) -> None:
        """``epoch`` is the optimizer step at which the refresh happens (0: start)."""
        from .labor_advantage import qol_td_admissibility, qol_td_lambda_targets

        refresh_started = time.time()
        previous = task_targets["qol_years"]
        every = torch.arange(n)
        values = task_critic.expectation("qol_years", qol_logits(every))
        task_targets["qol_years"] = qol_td_lambda_targets(dataset, values, qol_head.td_lambda)[
            kept
        ]
        replaced_mc = mc_pending[0]
        mc_pending[0] = False
        if monitor is not None:
            # The first refresh replaces the Monte Carlo return; its change is
            # recorded but does not count toward the growth alarm.
            error = float(
                (values[kept].double() - task_targets["qol_years"].double()).abs().mean()
            )
            # Per-row admissible bound and the reward / discount / head-range checks
            # (user decision 27 Oct; replaces the flat [0, W] bound).
            admissibility = qol_td_admissibility(
                dataset,
                values,
                qol_head.td_lambda,
                head_max=qol_head.head_max,
                weight=qol_head.discount_weight,
                discount_years=qol_head.discount_years,
                rows=kept,
                tolerance=monitor.t.td_bound_tolerance,
            )
            admissibility["diagnostics"]["clipped_targets"] = task_critic.qol_clip_count(
                task_targets["qol_years"]
            )
            monitor.td_refresh(
                epoch,
                previous,
                task_targets["qol_years"],
                qol_head.discount_weight,
                growth_eligible=epoch > 0 and not replaced_mc,
                error=error,
                seconds=time.time() - refresh_started,
                admissibility=admissibility,
            )

    td_lambda = qol_head is not None and qol_head.target == "td-lambda"

    # Held-out validation (monitoring only): its rows never reach the minibatch sampler,
    # which draws only from ``kept`` of the training dataset above.
    validation = None
    validation_history = None  # used only when the stability monitor is off
    if validation_dataset is not None:
        stochastic = _stochastic_modules(task_critic, cost_critics)
        if stochastic:
            raise ValueError(
                f"validation needs a deterministic critic forward; stochastic modules {stochastic}"
            )
        if false_terminal_policy_of(validation_dataset) != false_terminal_policy_of(dataset):
            raise ValueError("validation and training datasets differ in false-terminal policy")
        if tail_normalization_mode(validation_dataset) != tail_normalization_mode(dataset):
            raise ValueError(
                "validation and training datasets differ in tail normalisation "
                f"({tail_normalization_mode(validation_dataset)!r} vs "
                f"{tail_normalization_mode(dataset)!r})"
            )
        validation = _critic_validation_set(
            validation_dataset,
            training_worlds={int(w) for w in dataset["row_world"].reshape(-1)[kept].tolist()},
            tensor_keys=tensor_keys,
            tail_method=tail_method,
        )
        if monitor is None:
            from .labor_stability import ValidationMonitor

            validation_history = ValidationMonitor()
    epoch_loss_sum: torch.Tensor | None = None
    epoch_loss_steps = 0
    best_validation: dict[str, Any] = {
        "epoch": None, "step": None, "total": None, "task_state": None, "cost_state": None,
        "per_head": {},
    }

    def validation_logits(rows: torch.Tensor, chunk: int = 512) -> torch.Tensor:
        data = validation["dataset"]
        parts = []
        for start in range(0, len(rows), chunk):
            index = rows[start : start + chunk]
            batch = to_device({k: data[k][index] for k in tensor_keys}, device)
            p = critic_privileged_batch(task_critic, data, index, device=device)
            parts.append(task_critic(batch, p)["qol_years"].cpu())
        return torch.cat(parts)

    def evaluate_validation(epoch: int, chunk: int = 512) -> None:
        """Validation losses after a completed epoch; reads parameters, writes nothing to the
        fit. Same per-head losses as training (TD(lambda) QoL targets bootstrapped through the
        current head on the validation trajectories, as training refreshes its own)."""
        from .labor_ppo import labor_cost_value_losses as cost_losses
        from .labor_ppo import labor_task_value_losses as task_losses

        data, rows = validation["dataset"], validation["rows"]
        count = len(rows)
        sums: dict[str, float] = defaultdict(float)
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            task_targets_v = dict(validation["task_targets"])
            if td_lambda:
                from .labor_advantage import qol_td_lambda_targets

                every = torch.arange(int(data["row_world"].shape[0]))
                values = task_critic.expectation("qol_years", validation_logits(every))
                task_targets_v["qol_years"] = qol_td_lambda_targets(
                    data, values, qol_head.td_lambda
                )[rows]
            for start in range(0, count, chunk):
                index = rows[start : start + chunk]
                local = torch.arange(start, min(count, start + chunk))
                t = to_device({k: data[k][index] for k in tensor_keys}, device)
                p = critic_privileged_batch(task_critic, data, index, device=device)
                chunk_targets = to_device({k: v[local] for k, v in task_targets_v.items()}, device)
                _, heads, output = task_losses(task_critic, t, chunk_targets, p)
                rows_excess = qol_excess(output, chunk_targets)
                if rows_excess is not None:
                    sums["qol_excess_bce"] += float(rows_excess.sum())
                _, c_heads = cost_losses(
                    cost_critics,
                    t,
                    to_device({k: v[local] for k, v in validation["cost_targets"].items()}, device),
                    p,
                )
                for name, value in {**heads, **c_heads}.items():
                    sums[name] += float(value.double().sum())
                if td_lambda:
                    monte_carlo = task_critic.value_loss(
                        "qol_years",
                        output["qol_years"],
                        to_device(validation["task_targets"]["qol_years"][local], device),
                    )
                    sums["qol_years_monte_carlo"] += float(monte_carlo.double().sum())
        heads_mean = {name: value / count for name, value in sums.items()}
        task_v = sum(heads_mean[name] for name in LABOR_OBJECTIVES) / len(LABOR_OBJECTIVES)
        cost_v = sum(heads_mean[name] for name in LABOR_COSTS) / len(LABOR_COSTS)
        record = {
            "epoch": int(epoch),
            "step": int(steps),
            "rows": int(count),
            "worlds": len(validation["worlds"]),
            "train_loss": None
            if not epoch_loss_steps
            else float(epoch_loss_sum) / epoch_loss_steps,
            "train_steps": int(epoch_loss_steps),
            "total": task_v + cost_loss_weight * cost_v,
            "task": task_v,
            "cost": cost_v,
            "heads": {
                k: v
                for k, v in heads_mean.items()
                if k not in ("qol_years_monte_carlo", "qol_excess_bce")
            },
            "qol_target": "td-lambda" if td_lambda else "monte-carlo",
            **(
                {"qol_years_monte_carlo": heads_mean["qol_years_monte_carlo"]}
                if td_lambda
                else {}
            ),
            **(
                {"qol_excess_bce": heads_mean["qol_excess_bce"]}
                if "qol_excess_bce" in heads_mean
                else {}
            ),
        }
        if monitor is not None:
            monitor.record_validation(record)
        else:
            validation_history.update(record)
        # Best-validation tracking (lowest total; earliest epoch on ties): reads parameters.
        for head, value in record["heads"].items():
            seen = best_validation["per_head"].get(head)
            if seen is None or value < seen["loss"]:
                best_validation["per_head"][head] = {"epoch": int(epoch), "loss": float(value)}
        if best_validation["epoch"] is None or record["total"] < best_validation["total"]:
            best_validation.update(
                epoch=int(epoch),
                step=int(steps),
                total=float(record["total"]),
                # Cloned: cpu_copy returns CPU tensors as they are (they would alias the
                # live parameters).
                task_state={k: v.detach().cpu().clone() for k, v in task_critic.state_dict().items()},
                cost_state={k: v.detach().cpu().clone() for k, v in cost_critics.state_dict().items()},
            )

    generator = rng if rng is not None else np.random.default_rng(0)
    fingerprint = _critic_resume_fingerprint(
        dataset, epochs, batch_size, lr, max_grad_norm, cost_loss_weight, N
    )
    if family_schedule is not None:
        import hashlib
        import json

        fingerprint = hashlib.sha256(
            json.dumps(
                {"base": fingerprint, "family_schedule": family_schedule.state_dict()},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    if weighted:
        import hashlib
        import json

        # Only a weighted fit extends the fingerprint (default resume files stay valid).
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "base": fingerprint,
                    "task_loss_weights": weight_values,
                    "qol_loss_weight_auto": None
                    if qol_loss_weight_auto is None
                    else qol_loss_weight_auto.state_dict(),
                    # Only a gradnorm fit adds the key (fixed/auto fingerprints unchanged).
                    **(
                        {}
                        if loss_balancing is None
                        else {"loss_balancing": loss_balancing.state_dict()}
                    ),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    tail_normalization = (dataset.get("collection") or {}).get(TAIL_NORMALIZATION_FIELD)
    false_terminals = (dataset.get("collection") or {}).get(FALSE_TERMINAL_FIELD)
    level_fix = {
        key: value
        for key, value in (
            ("qol_head_bias_init", qol_head_bias_init),
            ("td_step0_targets", td_step0_targets),
            ("tail_normalization", tail_normalization),
            ("false_terminal_correction", false_terminals),
            ("grad_clip", None if grad_clip == "global" else grad_clip),
        )
        if value is not None
    }
    if level_fix:
        import hashlib
        import json

        # Only a fit using a level fix or normalised tails extends the fingerprint.
        fingerprint = hashlib.sha256(
            json.dumps({"base": fingerprint, **level_fix}, sort_keys=True, default=str).encode(
                "utf-8"
            )
        ).hexdigest()
    bias_init_record: dict[str, Any] | None = None
    if qol_head_bias_init is not None and (qol_head is None or qol_head.output != "logistic"):
        raise ValueError("--qol-head-bias-init mean-target needs a logistic bounded QoL head")
    if td_step0_targets is not None and not (
        qol_head is not None and qol_head.target == "td-lambda"
    ):
        raise ValueError("--td-step0-targets mc needs a bounded QoL head with TD(lambda) targets")

    def schedule_progress() -> float:
        return steps / steps_per_epoch
    resume = None
    if resume_path is not None and os.path.exists(resume_path):
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume.get("fingerprint") != fingerprint:
            raise ValueError(f"resume state {resume_path} belongs to a different critic fit")
    if resume is None:
        if qol_head_bias_init == "mean-target":
            fraction = task_critic.qol_target_fraction(task_targets["qol_years"].reshape(-1))
            weighted_mean = float(
                (torch.as_tensor(probabilities, dtype=torch.float64) * fraction.cpu()).sum()
            )
            plain_mean = float(fraction.mean())
            p_bar = min(max(weighted_mean, 1e-6), 1.0 - 1e-6)
            logit = math.log(p_bar / (1.0 - p_bar))
            bias = task_critic.heads["qol_years"].bias
            previous_bias = float(bias.detach().reshape(-1)[0])
            with torch.no_grad():
                bias.fill_(logit)
            bias_init_record = {
                "mode": "mean-target",
                "mean_target_fraction": weighted_mean,
                "mean_target_fraction_unweighted": plain_mean,
                "weighting": "the fit's minibatch sampling probabilities (world-balanced, "
                "family/source weights included)",
                "target": "Monte Carlo return / (head scale x W), clamped to [0, 1]",
                "bias": float(bias.detach().reshape(-1)[0]),
                "logit": logit,
                "previous_bias": previous_bias,
            }
        if td_lambda and td_step0_targets == "mc":
            mc_pending[0] = True  # Monte Carlo targets until the first scheduled refresh
        elif td_lambda:
            refresh_td_targets(0)
        saturation_initial = (
            None
            if qol_head is None
            else saturation_summary()
        )
        init_task, init_cost = losses()
        init_excess = excess_sum[0]
        first_epoch, first_start, epoch_state, steps = 0, 0, None, 0
    else:
        task_critic.load_state_dict(resume["task"])
        cost_critics.load_state_dict(resume["costs"])
        optimizer.load_state_dict(resume["optimizer"])
        if td_lambda:
            task_targets["qol_years"] = resume["td_targets"]
        if td_step0_targets is not None:
            mc_pending[0] = bool(resume["td_step0_mc_pending"])
        if qol_head_bias_init is not None:
            bias_init_record = resume["qol_head_bias_init"]
        saturation_initial = resume["saturation_initial"]
        init_task, init_cost = float(resume["init_task"]), float(resume["init_cost"])
        init_excess = resume.get("init_qol_excess_bce")
        if weighted:
            saved_weights = resume.get("task_loss_weights")
            if saved_weights is None:
                raise ValueError("resume state has no task loss weight state")
            weight_values.update({k: float(v) for k, v in saved_weights["values"].items()})
            weight_history[:] = copy.deepcopy(list(saved_weights["history"]))
        if gradnorm is not None:
            if resume.get("gradnorm") is None:
                raise ValueError("resume state has no gradnorm state")
            gradnorm.load_state_dict(resume["gradnorm"])
        if resume.get("best_validation") is not None:
            best_validation.update(copy.deepcopy(resume["best_validation"]))
        first_epoch, first_start = int(resume["epoch"]), int(resume["start"])
        epoch_state, steps = resume["generator_state"], int(resume["steps"])
        if monitor is not None:
            if resume.get("monitor") is None:
                raise ValueError("resume state has no stability monitor history")
            monitor.load_state_dict(resume["monitor"])
        if validation is not None and resume.get("validation_epoch") is not None:
            loss_sum, epoch_loss_steps = resume["validation_epoch"]
            epoch_loss_sum = None if loss_sum is None else torch.tensor(
                float(loss_sum), dtype=torch.float64, device=device
            )
            if validation_history is not None and resume.get("validation_history"):
                validation_history.load_state_dict(resume["validation_history"])
    started = time.time()

    def save_resume(epoch: int, start: int, state: Any) -> None:
        payload = {
            "fingerprint": fingerprint,
            # Device-agnostic: always saved on CPU (labor_device.cpu_copy).
            "task": cpu_copy(task_critic.state_dict()),
            "costs": cpu_copy(cost_critics.state_dict()),
            "optimizer": cpu_copy(optimizer.state_dict()),
            "td_targets": task_targets["qol_years"] if td_lambda else None,
            "saturation_initial": saturation_initial,
            "init_task": init_task,
            "init_cost": init_cost,
            "epoch": epoch,
            "start": start,
            "steps": steps,
            "generator_state": state,
            "monitor": None if monitor is None else monitor.state_dict(),
            "init_qol_excess_bce": init_excess,
        }
        if weighted:
            payload["task_loss_weights"] = {
                "values": dict(weight_values),
                "history": copy.deepcopy(weight_history),
            }
        if gradnorm is not None:
            payload["gradnorm"] = gradnorm.state_dict()
        if td_step0_targets is not None:
            payload["td_step0_mc_pending"] = bool(mc_pending[0])
        if qol_head_bias_init is not None:
            payload["qol_head_bias_init"] = bias_init_record
        if validation is not None:
            payload["best_validation"] = best_validation
            # The partial epoch's training-loss accumulator and (monitor off) the history.
            payload["validation_epoch"] = (
                None if epoch_loss_sum is None else float(epoch_loss_sum),
                int(epoch_loss_steps),
            )
            payload["validation_history"] = (
                None if validation_history is None else validation_history.state_dict()
            )
        temporary = resume_path + ".tmp"
        torch.save(payload, temporary)
        os.replace(temporary, resume_path)

    stop = False
    for epoch in range(first_epoch, epochs):
        resumed_epoch = resume is not None and epoch == first_epoch
        if resumed_epoch:
            generator.bit_generator.state = copy.deepcopy(epoch_state)
        state_at_epoch = copy.deepcopy(generator.bit_generator.state)
        if validation is not None and not resumed_epoch:
            epoch_loss_sum, epoch_loss_steps = None, 0
        if critic_weighter is None:
            draws = generator.choice(N, size=N, replace=True, p=probabilities)
        else:
            # One uniform per draw from the epoch-start state; each step maps its slice
            # through that step's decayed CDF (exact on resume).
            uniforms = generator.random(N)
        for start in range(first_start if resumed_epoch else 0, N, batch_size):
            if max_steps is not None and steps >= max_steps:
                stop = True
                break
            if td_lambda and steps > 0 and steps % td_interval == 0:
                refresh_td_targets(steps)
            if critic_weighter is None:
                batch_draws = draws[start : start + batch_size]
            else:
                factor = family_schedule.factor(schedule_progress())
                cdf = np.cumsum(critic_weighter.weights(factor))
                batch_draws = np.minimum(
                    np.searchsorted(cdf, uniforms[start : start + batch_size] * cdf[-1], "right"),
                    N - 1,
                )
            batch_idx = torch.as_tensor(batch_draws, dtype=torch.long)
            b_tensors = to_device({k: v[batch_idx] for k, v in tensors.items()}, device)
            b_privileged = to_device(privileged_rows(batch_idx), device)
            optimizer.zero_grad()
            b_task_targets = to_device({k: v[batch_idx] for k, v in task_targets.items()}, device)
            step_weights = dict(weight_values) if weighted else None
            task_loss, task_heads, task_output = labor_task_value_losses(
                task_critic,
                b_tensors,
                b_task_targets,
                b_privileged,
                **({} if step_weights is None else {"weights": step_weights}),
            )
            cost_loss, cost_heads = labor_cost_value_losses(
                cost_critics,
                b_tensors,
                to_device({k: v[batch_idx] for k, v in cost_targets.items()}, device),
                b_privileged,
            )
            loss = task_loss + cost_loss_weight * cost_loss
            accepted = True
            gradnorm_entry: dict[str, Any] | None = None
            gradnorm_saved: dict[str, Any] | None = None
            if gradnorm is not None:
                # Unweighted head gradients on W (no .grad accumulation; the graph is kept for
                # the step's backward), committed before the step and undone on a rollback.
                raw = {}
                for name in LABOR_OBJECTIVES:
                    grads = torch.autograd.grad(
                        task_heads[name].mean(), gradnorm_params, retain_graph=True,
                        allow_unused=True,
                    )
                    raw[name] = math.sqrt(
                        sum(float(g.detach().double().pow(2).sum()) for g in grads if g is not None)
                    )
                rate_losses = {
                    name: float(task_heads[name].detach().mean()) for name in LABOR_OBJECTIVES
                }
                if excess_progress:
                    excess_rows = qol_excess(task_output, b_task_targets)
                    if excess_rows is not None:
                        rate_losses["qol_years"] = float(excess_rows.detach().mean())
                proposal = gradnorm.proposal(raw, rate_losses)
                if all(math.isfinite(v) for v in (*raw.values(), *rate_losses.values())):
                    gradnorm_saved = gradnorm.snapshot()
                    next_weights = gradnorm.commit(proposal)
                    gradnorm_entry = {
                        "step": int(steps) + 1, **proposal, "weights_next": next_weights
                    }
            if monitor is None:
                if not torch.isfinite(loss):
                    raise ValueError("heuristic critic pretraining produced a nonfinite loss")
                loss.backward()
                if clip_groups is None:
                    nn.utils.clip_grad_norm_(critic_params, max_grad_norm)
                else:
                    from .labor_stability import clip_grad_groups

                    clip_grad_groups(clip_groups, max_grad_norm)
                optimizer.step()
            else:

                def batch_excess(
                    output: Mapping[str, torch.Tensor] = task_output,
                    targets: Mapping[str, torch.Tensor] = b_task_targets,
                ) -> float | None:
                    rows_excess = qol_excess(output, targets)
                    return None if rows_excess is None else float(rows_excess.mean())

                def diagnostics(output: Mapping[str, torch.Tensor] = task_output) -> dict[str, Any]:
                    with torch.no_grad():
                        extra: dict[str, Any] = {}
                        if qol_head is not None:
                            extra["qol_saturation"] = task_critic.qol_saturation(
                                output["qol_years"]
                            )
                        excess_value = batch_excess()
                        if excess_value is not None:
                            extra["qol_excess_bce"] = excess_value
                        if not task_critic.is_bounded("population_years"):
                            extra["population_tail"] = population_tail_report(
                                task_critic,
                                output["population_years"].detach(),
                                observed_pop_max,
                            )
                        if family_schedule is not None:
                            progress = schedule_progress()
                            extra["family_weight_progress_epochs"] = (
                                float(family_schedule.epoch_offset) + progress
                            )
                            extra["family_weight_factor"] = family_schedule.factor(progress)
                            extra["family_weight_multipliers"] = family_schedule.multipliers(
                                progress
                            )
                        return extra

                head_losses = {name: value.mean() for name, value in task_heads.items()}
                head_losses["task"] = task_loss
                head_losses.update({name: value.mean() for name, value in cost_heads.items()})
                def weight_hook(
                    effective: Mapping[str, Mapping[str, float]],
                    raw: Mapping[str, Mapping[str, float]],
                    used: Mapping[str, float] | None = step_weights,
                    gradnorm_step: Mapping[str, Any] | None = gradnorm_entry,
                ) -> dict[str, Any]:
                    """Log the weights this step used (and the gradnorm update); update the
                    auto QoL weight."""
                    fields: dict[str, Any] = {"task_loss_weights": dict(used)}
                    if gradnorm_step is not None:
                        entry = {**gradnorm_step, "step": int(monitor.step_index)}
                        weight_history.append(entry)
                        fields["gradnorm"] = entry
                    if qol_loss_weight_auto is not None:
                        norms = raw["task_encoder"]
                        update = qol_loss_weight_auto.update(
                            used["qol_years"],
                            norms["qol_years"],
                            norms["population_years"],
                            used["population_years"],
                        )
                        weight_values["qol_years"] = update["qol_weight_next"]
                        entry = {"step": int(monitor.step_index), **update}
                        weight_history.append(entry)
                        fields["qol_loss_weight_auto"] = entry
                    return fields

                accepted = monitor.step(
                    loss,
                    max_grad_norm=max_grad_norm,
                    head_losses=head_losses,
                    diagnostics=diagnostics,
                    **(
                        {}
                        if step_weights is None
                        else {"head_grad_scales": step_weights, "log_hook": weight_hook}
                    ),
                    **(
                        {
                            "progress_losses": lambda: {"qol_years": batch_excess()},
                        }
                        if excess_progress
                        else {}
                    ),
                )
            if gradnorm is not None:
                if accepted and gradnorm_saved is not None:
                    weight_values.update(gradnorm.values())
                elif gradnorm_saved is not None:
                    gradnorm.load_state_dict(gradnorm_saved)  # rolled back: weights unchanged
            if validation is not None and accepted:
                # Read-only accumulation on the device (no host sync per step).
                value = loss.detach().double()
                epoch_loss_sum = value if epoch_loss_sum is None else epoch_loss_sum + value
                epoch_loss_steps += 1
            steps += 1
            if progress and steps % 25 == 0:
                elapsed = time.time() - started
                print(
                    f"[progress] steps={steps} epoch={epoch} elapsed_s={elapsed:.0f}",
                    file=sys.stderr,
                    flush=True,
                )
            due = resume_path is not None and steps % max(1, resume_every) == 0
            if due and start + batch_size < N:
                save_resume(epoch, start + batch_size, state_at_epoch)
        if stop:
            break
        if validation is not None:
            evaluate_validation(epoch + 1)
    final_task, final_cost = losses()
    final_excess = excess_sum[0]
    collection = dataset.get("collection", {}) if isinstance(dataset, Mapping) else {}
    qol_record = (
        {}
        if qol_head is None
        else {
            "qol_head": qol_head.state_dict(),
            "qol_saturation_initial": saturation_initial,
            "qol_saturation_final": saturation_summary(),
            # Targets the loss's guard clamp at scale * W moves (expected zero at scale 2).
            # Recorded for a widened head only, so historical summaries are unchanged.
            **(
                {"qol_clipped_targets": task_critic.qol_clip_count(task_targets["qol_years"])}
                if qol_head.scale != 1.0
                else {}
            ),
        }
    )
    stability_record: dict[str, Any] = {}
    if monitor is not None:
        stability_record["stability"] = _critic_stability_finish(
            monitor,
            task_critic,
            cost_critics,
            dataset,
            kept,
            tensor_keys,
            qol_record.get("qol_saturation_final"),
            observed_pop_max,
        )
    validation_record: dict[str, Any] = {}
    if validation is not None:
        history = monitor.validation if monitor is not None else validation_history
        validation_record["validation"] = {
            "gating": False,
            "worlds": validation["worlds"],
            "rows": len(validation["rows"]),
            **(
                {"status": "NO-COMPLETED-EPOCH", "epochs": [], "alarms": []}
                if history is None
                else history.report()
            ),
            "best": {
                "criterion": "lowest validation total (earliest epoch on ties)",
                "epoch": best_validation["epoch"],
                "step": best_validation["step"],
                "total": best_validation["total"],
                "per_head": copy.deepcopy(best_validation["per_head"]),
            },
        }
        if best_validation_sink is not None and best_validation["epoch"] is not None:
            best_validation_sink.update(
                epoch=best_validation["epoch"],
                step=best_validation["step"],
                total=best_validation["total"],
                task=best_validation["task_state"],
                costs=best_validation["cost_state"],
            )
    weight_record: dict[str, Any] = {}
    if weighted:
        weight_record["task_loss_weights"] = {
            "mode": "gradnorm"
            if gradnorm is not None
            else "auto-qol"
            if qol_loss_weight_auto is not None
            else "fixed",
            "final": dict(weight_values),
            **(
                {}
                if gradnorm is None
                else {
                    "gradnorm": loss_balancing.state_dict(),
                    "initial_ratio": {
                        k: float(v) for k, v in (task_loss_weights or {}).items()
                    },
                    "shared_layer": "task_encoder.trunk last Linear weight",
                    "initial_losses": None
                    if gradnorm.initial_losses is None
                    else dict(gradnorm.initial_losses),
                    "history": copy.deepcopy(weight_history),
                }
            ),
            **(
                {}
                if qol_loss_weight_auto is None
                else {
                    "auto": qol_loss_weight_auto.state_dict(),
                    "history": copy.deepcopy(weight_history),
                }
            ),
            "note": "weights act on the optimizer objective; reported losses are unweighted",
        }
    level_record: dict[str, Any] = {}
    if clip_groups is not None:
        level_record["grad_clip"] = {
            "mode": grad_clip,
            "max_grad_norm": float(max_grad_norm),
            "groups": {
                name: sum(p.numel() for p in params) for name, params in clip_groups.items()
            },
        }
    if bias_init_record is not None:
        level_record["qol_head_bias_init"] = bias_init_record
    if td_step0_targets is not None:
        level_record["td_step0_targets"] = {
            "mode": td_step0_targets,
            "definition": "Monte Carlo returns are the QoL targets until the first scheduled "
            "TD(lambda) refresh (no step-0 bootstrap through the untrained head)",
            "first_refresh_step": int(td_interval),
        }
    if tail_normalization is not None:
        level_record["tail_normalization"] = copy.deepcopy(tail_normalization)
    if false_terminals is not None:
        level_record[FALSE_TERMINAL_FIELD] = copy.deepcopy(false_terminals)
    excess_record: dict[str, Any] = {}
    if final_excess is not None:
        excess_record = {
            "initial_qol_excess_bce": init_excess,
            "final_qol_excess_bce": final_excess,
        }
    return {
        **stability_record,
        **qol_record,
        **validation_record,
        **weight_record,
        **excess_record,
        **level_record,
        # Recorded for a separate QoL encoder only (historical summaries unchanged).
        **(
            {
                "qol_encoder": {
                    "mode": "separate",
                    "dominance": "no shared task-encoder detector (one head per encoder); "
                    "task critic remains a global-clip contender",
                    "dominance_excess_loss_effective": False,
                }
            }
            if separate_qol
            else {}
        ),
        "contract": LABOR_CRITIC_CONTRACT,
        "epochs": int(epochs),
        "rows": int(N),
        "excluded_quarantined_rows": int(quarantined.sum()),
        "excluded_tail_unavailable_rows": int((~quarantined & ~available).sum()),
        "initial_loss": init_task + init_cost,
        "final_loss": final_task + final_cost,
        "initial_task_loss": init_task,
        "final_task_loss": final_task,
        "initial_cost_loss": init_cost,
        "final_cost_loss": final_cost,
        "world_seeds": list(collection.get("world_seeds", [])),
        "turns": int(collection.get("turns", 0)),
        "return_tail": tail_method,
        **(
            {"return_tail_settings": _tail_settings_summary(collection)}
            if tail_method == RETURN_TAIL_RANDOMIZED
            else {}
        ),
        "sampling": (
            "equal-source-mass-then-world-balanced-with-replacement"
            if scenario_rows
            else "world-balanced-with-replacement"
        ),
        "discount_years": CRITIC_DISCOUNT_YEARS,
        **(
            {
                "family_weighting": {
                    "weights": dict(family_weighting["weights"]),
                    "status": family_weighting["status"],
                    "critic_shares": critic_family_shares,
                    **(
                        {}
                        if family_schedule is None
                        else {
                            "schedule": {
                                **family_schedule.state_dict(),
                                "epochs_end": float(family_schedule.epoch_offset)
                                + steps / steps_per_epoch,
                                "final_multipliers": family_schedule.multipliers(
                                    steps / steps_per_epoch
                                ),
                            }
                        }
                    ),
                }
            }
            if critic_family_shares is not None
            else {}
        ),
        **(
            {
                "curated_scenario_rows": int(
                    (dataset["row_source"].reshape(-1)[kept] == SCENARIO_ROW_SOURCE).sum()
                )
            }
            if scenario_rows
            else {}
        ),
        **(
            {
                "learner_state_continuation_rows": int(
                    (dataset["row_source"].reshape(-1)[kept] == CONTINUATION_ROW_SOURCE).sum()
                )
            }
            if allow_learner_state_continuation and "row_source" in dataset
            else {}
        ),
        **(
            {}
            if privileged is None
            else {
                "privileged_schema_version": int(task_critic.privileged_layout.schema_version),
                "privileged_normalizer_digest": task_critic.privileged_normalizer.digest(),
            }
        ),
        # Reported losses above are unweighted; only a non-default weight is recorded.
        **({} if cost_loss_weight == 1.0 else {"cost_loss_weight": float(cost_loss_weight)}),
    }


def pretrain_critic_on_heuristic_next_states(
    *,
    task_critic: LaborTaskCritic,
    cost_critics: LaborCostCritics,
    normalizer: Any,
    world_seeds: Sequence[int],
    turns: int = 25,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_grad_norm: float = 1.0,
    personality: LaborPersonality | None = None,
    rng: np.random.Generator | None = None,
    labor_bridge_factory: Any = LaborModel9Bridge,
    heuristic_bridge_factory: Any = None,
    allow_vm_scale: bool = False,
    reset_options: LaborResetOptions | None = None,
    tail: RandomizedTailSettings | None = None,
) -> dict[str, Any]:
    """Collect heuristic trajectories and train value critics on their returns.

    ``tail`` closes living cutoffs with the randomized teacher tail; without it
    the legacy truncated-zero window is used and named in the summary.
    """
    worlds = checked_seeds(list(world_seeds), "labor heuristic critic worlds")
    scan({"worldSeeds": worlds}, "labor heuristic critic worlds")
    if not allow_vm_scale and len(worlds) > 3:
        raise ValueError("local labor heuristic critic pretraining requires 1..3 world seeds")
    if not worlds:
        raise ValueError("world seeds cannot be empty")
    if turns < 1:
        raise ValueError("turns must be >= 1")
    dataset = collect_heuristic_bc_dataset(
        normalizer=normalizer,
        world_seeds=worlds,
        turns=turns,
        personality=personality,
        labor_bridge_factory=labor_bridge_factory,
        heuristic_bridge_factory=heuristic_bridge_factory,
        allow_vm_scale=allow_vm_scale,
        reset_options=reset_options,
        tail=tail,
    )
    summary = pretrain_critic_on_dataset(
        task_critic=task_critic,
        cost_critics=cost_critics,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_grad_norm=max_grad_norm,
        rng=rng,
    )
    summary["reset_options"] = (reset_options or LaborResetOptions()).state_dict()
    return summary
