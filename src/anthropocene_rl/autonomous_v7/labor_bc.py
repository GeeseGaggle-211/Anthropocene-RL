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
    _simplex,
)
from .labor_model import (
    LABOR_COSTS,
    LABOR_OBJECTIVES,
    LaborActor,
    LaborCostCritics,
    LaborPersonality,
    LaborTaskCritic,
    _masked_softmax,
    _masks,
    labor_policy_tensors,
)
from .labor_personality import personality_for_world
from .labor_ppo import (
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


def validate_fitting_worlds(world_seeds: Sequence[int], context: str) -> list[int]:
    """Fitting worlds exclude sealed confirmation and development-only seeds."""
    worlds = checked_seeds(list(world_seeds), context)
    scan({"worldSeeds": worlds}, context)
    if len(set(worlds)) != len(worlds):
        raise ValueError(f"{context} repeats a world seed")
    for seed in worlds:
        if seed in SEALED_WORLDS:
            raise ValueError(f"{context}: sealed confirmation world {seed} is not fitting data")
        if seed in DEVELOPMENT_WORLDS:
            raise ValueError(f"{context}: development world {seed} is development-only")
    return worlds


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
    over-budget work makes the labor label unavailable instead.
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
            shares[0] = max(0.0, 1.0 - float(shares[1:].sum()))
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
) -> tuple[Mapping[str, Any], list[HeuristicLaborLabel]]:
    """The teacher's converted full-action label for every live camp at this exact state.

    Captures the actor-labor bridge boundary, resets the shadow heuristic bridge
    from that snapshot with the same reset rules, takes one native heuristic step
    and converts each camp's recommendation against its pre-action catalog. The
    labor bridge itself is not advanced (one native resolve on the shadow bridge).
    Shared by the teacher collector, the h12 evaluation path and the DAgger
    learner-state collector.
    """
    snap = labor_bridge._bridge.capture()
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
) -> WorldCollection:
    """Advance one world under converted heuristic actions and record every row.

    Each row records pre-action inputs, the converted label with per-head
    validity, identity (world, camp, turn, attempt=0), the transition rewards
    used for critic returns, and whether the initial recommendation was
    accepted. Rejected initial recommendations are quarantined: the all-idle
    retry that executes instead never becomes their label.
    """

    options = reset_options or LaborResetOptions()
    declared_personality = LaborPersonality() if personality is None else personality
    heur_factory = heuristic_bridge_factory or (lambda: Model9Bridge(economic_diagnostics=True))
    from .labor_training import _settlement_guard, _transition_rewards

    result = WorldCollection(world_seed=int(world_seed))
    with labor_bridge_factory() as labor_bridge, heur_factory() as heur_bridge:
        step = labor_bridge.reset(int(world_seed), **options.reset_kwargs())
        identity = getattr(labor_bridge, "last_reset_identity", None)
        result.reset_identity = dict(identity) if isinstance(identity, Mapping) else {}
        founded = {int(row["camp"]): int(step["turn"]) for row in step["villages"]}
        for _ in range(turns):
            live = [
                v
                for v in step["villages"]
                if v.get("observation") is not None and not v.get("terminated")
            ]
            if not live:
                break
            contexts, tensors = labor_policy_tensors(
                live, normalizer, personality=personality_for_world(declared_personality, int(world_seed))
            )
            catalogs = {context[0].camp_id: context[0] for context in contexts}
            decision_turn = int(step["turn"])

            # Shadow heuristic step from the captured boundary, same reset rules.
            h_step, labels = shadow_teacher_labels(
                labor_bridge, heur_bridge, int(world_seed), options, live, contexts
            )
            result.native_resolves += 1
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

            retained: dict[int, RetainedLaborAction] = {}
            pending: dict[int, tuple[int, HeuristicLaborLabel]] = {}
            for row, (village, context, label) in enumerate(zip(live, contexts, labels, strict=True)):
                camp = int(village["camp"])
                catalog = context[0]
                for reason in label.reasons:
                    result.reasons[reason.split(":", 1)[0]] += 1
                pending[camp] = (row, label)
                retained[camp] = _retained_action(
                    label, catalog, tensors["movement_mask"][row : row + 1]
                )

            execution = execute_labor_decision(labor_bridge, catalogs, retained, _idle_retry)
            next_step = execution.step
            result.native_resolves += 1
            result.decisions += 1
            rewards, terminal, elapsed = _transition_rewards(
                next_step, tuple(catalogs), founded, world_index=0, decision_turn=decision_turn
            )
            outcomes = {int(v["camp"]): v for v in next_step["villages"]}
            for camp, (row, label) in pending.items():
                attempt = execution.initial_attempts[camp]
                accepted = bool(attempt.valid)
                result.counters["total_decisions"] += 1
                result.counters["initial_accepted" if accepted else "initial_rejected"] += 1
                if not accepted:
                    result.counters["quarantined"] += 1
                    result.quarantine.append(
                        {
                            "world": int(world_seed),
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
                row_tensors = _one_row(tensors, row)
                row_tensors.update(_label_tensors(label))
                guard = _settlement_guard(
                    outcomes[camp], founded.get(camp, decision_turn), int(next_step["turn"])
                )
                row_tensors.update(
                    {
                        "row_world": torch.tensor([int(world_seed)], dtype=torch.long),
                        "row_camp": torch.tensor([camp], dtype=torch.long),
                        "row_turn": torch.tensor([decision_turn], dtype=torch.long),
                        "row_attempt": torch.tensor([0], dtype=torch.long),
                        "row_quarantined": torch.tensor([not accepted], dtype=torch.bool),
                        "reward_qol_years": torch.tensor(
                            [rewards[camp]["qol_years"]], dtype=torch.float64
                        ),
                        "reward_population_years": torch.tensor(
                            [rewards[camp]["population_years"]], dtype=torch.float64
                        ),
                        "reward_feasibility": torch.tensor(
                            [float(attempt.feasibility_cost)], dtype=torch.float64
                        ),
                        "reward_settlement_guard": torch.tensor([float(guard)], dtype=torch.float64),
                        ELAPSED_YEARS_FIELD: torch.tensor(
                            [float(elapsed[camp])], dtype=torch.float64
                        ),
                        "transition_terminal": torch.tensor([bool(terminal[camp])], dtype=torch.bool),
                    }
                )
                result.rows.append(row_tensors)
            step = next_step
            if progress is not None:
                progress(result)
        result.surviving_camps = len([v for v in step.get("villages", []) if not v.get("terminated")])
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


def discounted_returns(dataset: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per (world, camp) trajectory discounted returns (truncated, no bootstrap)."""
    names = {
        "qol_years": "reward_qol_years",
        "population_years": "reward_population_years",
        "feasibility": "reward_feasibility",
        "settlement_guard": "reward_settlement_guard",
    }
    n = int(dataset["row_world"].shape[0])
    returns = {name: torch.zeros(n, dtype=torch.float64) for name in names}
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (world, camp) in enumerate(
        zip(dataset["row_world"].tolist(), dataset["row_camp"].tolist(), strict=True)
    ):
        groups[(int(world), int(camp))].append(index)
    turns = dataset["row_turn"].tolist()
    elapsed = elapsed_years(dataset).reshape(-1).tolist()
    terminal = dataset["transition_terminal"].reshape(-1).tolist()
    for indices in groups.values():
        indices.sort(key=lambda i: turns[i])
        running = dict.fromkeys(names, 0.0)
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
) -> dict[str, Any]:
    """Collect closed-loop heuristic labor trajectories for behavioral cloning."""
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
) -> tuple[torch.Tensor, float, float, dict[str, float]]:
    B = len(batch_tensors["target_specialty"])
    output = actor(batch_tensors)
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
            batch = {k: dataset[k][start:end] for k in tensor_keys}
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
) -> dict[str, Any]:
    """Supervised behavioral cloning on LaborActor using heuristic dataset."""
    if not isinstance(dataset, Mapping) or "target_specialty" not in dataset:
        raise ValueError("dataset must contain target_specialty and input tensors")
    N = int(dataset["target_specialty"].shape[0])
    if N == 0:
        raise ValueError("cannot pretrain on empty dataset")

    opt = optimizer if optimizer is not None else torch.optim.Adam(actor.parameters(), lr=lr)
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

    actor.train()
    for _ in range(epochs):
        indices = generator.permutation(N)
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            batch_idx = torch.as_tensor(indices[start:end], dtype=torch.long)
            batch = {k: dataset[k][batch_idx] for k in tensor_keys}

            loss, _, _, _ = _batch_loss_and_l1(
                actor,
                batch,
                target_log_std=target_log_std,
                inactive_role_weight=inactive_role_weight,
                label_smoothing=label_smoothing,
                spec_weights=spec_weights,
                act_weights=act_weights,
            )
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
        "collection": copy.deepcopy(dataset.get("collection", {})),
    }


# ---------------------------------------------------------------------------
# Critic pretraining on heuristic returns
# ---------------------------------------------------------------------------


# The only shard kind whose rewards are teacher-policy returns along one
# chronological (world, camp) trajectory. Equal to labor_teacher_data.SOURCE_KIND.
CRITIC_TEACHER_SOURCE_KIND = "native_heuristic_labor_teacher"
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
}


def require_critic_teacher_dataset(dataset: Mapping[str, Any]) -> None:
    """Refuse any dataset that is not chronological heuristic-teacher returns.

    Search shards (``row_source == 2``), learner-state DAgger/actor-only shards
    and multi-source aggregates are rejected by source kind, by the
    ``actor_only`` flag and by any nonzero ``row_source`` row. A dataset
    without a source kind (in-process heuristic collection) is the teacher.
    """
    collection = dataset.get("collection", {}) if isinstance(dataset, Mapping) else {}
    kind = collection.get("source_kind", CRITIC_TEACHER_SOURCE_KIND)
    if collection.get("actor_only"):
        raise ValueError(
            "pretrain-critic refuses actor-only (learner-state DAgger) data: its rewards "
            "follow the learner's short branches, not teacher-policy returns"
        )
    if kind != CRITIC_TEACHER_SOURCE_KIND:
        reason = _CRITIC_REFUSALS.get(kind, f"unknown source kind {kind!r}")
        raise ValueError(f"pretrain-critic refuses {kind} data: {reason}")
    sources = dataset.get("row_source") if isinstance(dataset, Mapping) else None
    if isinstance(sources, torch.Tensor) and bool((sources.reshape(-1) != 0).any()):
        codes = sorted({int(v) for v in sources.reshape(-1).tolist()} - {0})
        raise ValueError(
            f"pretrain-critic refuses rows with row_source {codes}: only heuristic-teacher "
            "rows (row_source 0) carry critic targets; search rows are 2, DAgger rows 1"
        )


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
) -> dict[str, Any]:
    """Fit task/cost critics on truncated heuristic returns, world-balanced.

    Returns are computed over every row of each (world, camp) trajectory;
    rows whose initial recommendation was rejected (an idle retry executed)
    are excluded from fitting because their continuation is not the
    heuristic's. Minibatches are drawn with world-balanced probabilities.
    """
    if not math.isfinite(cost_loss_weight) or cost_loss_weight < 0.0:
        raise ValueError("cost_loss_weight must be finite and non-negative")
    require_critic_teacher_dataset(dataset)
    returns = discounted_returns(dataset)
    n = int(dataset["row_world"].shape[0])
    keep = ~dataset["row_quarantined"].reshape(n).bool()
    if not bool(keep.any()):
        raise ValueError("heuristic critic dataset has no admitted rows")
    kept = torch.nonzero(keep).reshape(-1)
    tensor_keys = [
        k
        for k, v in dataset.items()
        if isinstance(v, torch.Tensor) and not k.startswith(("target_", "label_", "reward_", "transition_", "row_"))
    ]
    tensors = {k: dataset[k][kept] for k in tensor_keys}
    task_targets = {name: returns[name][kept] for name in LABOR_OBJECTIVES}
    cost_targets = {name: returns[name][kept] for name in LABOR_COSTS}
    weights = world_balanced_weights(dataset["row_world"][kept]).double().numpy()
    probabilities = weights / weights.sum()
    N = len(kept)

    critic_params = list(task_critic.parameters()) + list(cost_critics.parameters())
    optimizer = torch.optim.Adam(critic_params, lr=lr)

    def losses(chunk: int = 512) -> tuple[float, float]:
        # Chunked: one forward pass over every row needed ~100 GB of activations at 96k rows.
        task_sum = cost_sum = 0.0
        with torch.no_grad():
            for start in range(0, N, chunk):
                index = torch.arange(start, min(N, start + chunk))
                t = {k: v[index] for k, v in tensors.items()}
                count = len(index)
                task_sum += count * float(
                    labor_task_value_loss(task_critic, t, {k: v[index] for k, v in task_targets.items()})
                )
                cost_sum += count * float(
                    labor_cost_value_loss(cost_critics, t, {k: v[index] for k, v in cost_targets.items()})
                )
        return task_sum / N, cost_sum / N

    init_task, init_cost = losses()
    generator = rng if rng is not None else np.random.default_rng(0)
    for _ in range(epochs):
        draws = generator.choice(N, size=N, replace=True, p=probabilities)
        for start in range(0, N, batch_size):
            batch_idx = torch.as_tensor(draws[start : start + batch_size], dtype=torch.long)
            b_tensors = {k: v[batch_idx] for k, v in tensors.items()}
            optimizer.zero_grad()
            loss = labor_task_value_loss(
                task_critic, b_tensors, {k: v[batch_idx] for k, v in task_targets.items()}
            ) + cost_loss_weight * labor_cost_value_loss(
                cost_critics, b_tensors, {k: v[batch_idx] for k, v in cost_targets.items()}
            )
            if not torch.isfinite(loss):
                raise ValueError("heuristic critic pretraining produced a nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(critic_params, max_grad_norm)
            optimizer.step()
    final_task, final_cost = losses()
    collection = dataset.get("collection", {}) if isinstance(dataset, Mapping) else {}
    return {
        "contract": LABOR_CRITIC_CONTRACT,
        "epochs": int(epochs),
        "rows": int(N),
        "excluded_quarantined_rows": int(n - N),
        "initial_loss": init_task + init_cost,
        "final_loss": final_task + final_cost,
        "initial_task_loss": init_task,
        "final_task_loss": final_task,
        "initial_cost_loss": init_cost,
        "final_cost_loss": final_cost,
        "world_seeds": list(collection.get("world_seeds", [])),
        "turns": int(collection.get("turns", 0)),
        "return_tail": "truncated-zero-at-collection-horizon",
        "sampling": "world-balanced-with-replacement",
        "discount_years": CRITIC_DISCOUNT_YEARS,
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
) -> dict[str, Any]:
    """Collect heuristic trajectories and train value critics on their returns."""
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
