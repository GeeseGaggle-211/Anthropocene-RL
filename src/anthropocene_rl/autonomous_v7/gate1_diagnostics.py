"""Fail-closed, reporting-only helpers for the Gate-1 diagnostic evaluator.

The functions in this module consume bridge responses. They never provide an
actor input, change an action, or mutate simulator state. The same validators
are used after a live rollout and when artifact-backed reports are ingested.
"""
from __future__ import annotations

import copy
import json
import math
import random
import struct
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch

from . import contracts
from .provenance import digest

CRAFT_GOODS = (
    "HandTools", "Pots", "Cloth", "FarmTools", "Housing", "Tools",
    "AnimalFabric", "LightHousing", "HerdingTools", "HousingMaterial", "FishingTools",
)
MATERIALS = ("Stone", "Clay", "Wood")
TRADE_GOODS = (
    "Cloth", "AnimalFabric", "HousingMaterial", "Food", "LightHousing", "Pots",
    "HandTools", "FarmTools", "Stone", "Clay", "Wood",
)

CAPTURE_BOUND_FIELDS = {
    "max_steps", "max_villages_per_boundary", "max_economic_rows_per_step",
    "max_canonical_bytes",
}
HARD_CAPTURE_LIMITS = {
    "max_steps": 10_000,
    "max_villages_per_boundary": 4_096,
    "max_economic_rows_per_step": 4_096,
    "max_canonical_bytes": 256 * 1024 * 1024,
}
ECONOMIC_ROW_FIELDS = {
    "schemaVersion", "transitionId", "trajectoryId", "decisionTurn", "campId",
    "societyId", "years", "domains", "units", "unavailable", "goods", "materials",
    "herds", "requests", "trades", "transfers", "fields", "conservation",
    "parentTransitionId", "forkId", "branchId",
}
GATE2_ADDITIVE_ECONOMIC_ROW_FIELDS = {
    "controls", "conservationRows", "physicalEvents",
}
ECONOMIC_DOMAINS = {"population", "cultivation", "herding", "manufacturing", "trade"}
BOUNDARY_VILLAGE_FIELDS = {
    "camp", "society", "parentSociety", "priorPopulation", "population", "qol", "years",
    "welfare", "tradeUtility", "terminated", "founded", "observation", "observation_sha256",
}
SUPPORTED_TRANSFER_KINDS = {
    "fission", "founding", "migration", "fission-inventory", "fission-herd",
    "dispatch", "reintegration", "hunter-logistics",
}


def economic_unit_registry() -> dict[str, str]:
    """Exact native units for every scalar consumed by Gate-1 aggregation."""
    units = {
        "years": "years",
        "domains.population.personYears": "person-years",
        "domains.population.qol": "qol",
        "domains.population.endingResidents": "people",
        "domains.cultivation.foodDelivered": "food",
        "domains.herding.acceptedFood": "food",
        "herds.*.slaughterFood": "food",
        "domains.trade.settledTrades": "transactions",
        "domains.trade.realisedUtility": "person-qol-years",
    }
    for good in TRADE_GOODS:
        units[f"domains.trade.imported{good}"] = f"{good}-units"
        units[f"domains.trade.exported{good}"] = f"{good}-units"
    return units


_NATIVE_UNIT_NAMES = {
    "years", "people", "person-years", "qol", "fraction", "food", "animals",
    "worker-years", "field-capital", "food/worker-year", "planner-native",
    "transactions", "person-qol-years", "good-units", "material-units",
} | {f"{good}-units" for good in TRADE_GOODS}


def _unit_path_exists(row: dict, path: str) -> bool:
    if path == "years":
        return "years" in row
    if path == "herds.*.slaughterFood":
        return (isinstance(row.get("herds"), list)
                and all(isinstance(item, dict) and "slaughterFood" in item
                        for item in row["herds"]))
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def _validate_declared_economic_units(row: dict, *, require_gate1: bool) -> None:
    units = row.get("units")
    if not isinstance(units, dict) or not units:
        raise ValueError("economic diagnostic row has no unit registry")
    expected = economic_unit_registry()
    for path, unit in units.items():
        if not isinstance(path, str) or not isinstance(unit, str) or not unit:
            raise ValueError("economic diagnostic unit declaration is malformed")
        if path in expected:
            if unit != expected[path]:
                raise ValueError("economic source unit differs from the authoritative native unit")
        elif unit not in _NATIVE_UNIT_NAMES or not path.startswith("domains."):
            raise ValueError("economic diagnostic row has an unknown/conflicting native unit")
        if not _unit_path_exists(row, path):
            raise ValueError("economic unit declaration names an absent field")
    if require_gate1 and (set(expected) - set(units)):
        raise ValueError("economic diagnostic row omits a required Gate-1 unit declaration")


def complete_economic_units(row: dict) -> None:
    """Enrich one live, unsealed bridge row before capture; imported rows never call this."""
    units = row.get("units")
    if units is None:
        units = {}
        row["units"] = units
    if not isinstance(units, dict):
        raise ValueError("economic diagnostic row has no unit registry")
    if units:
        _validate_declared_economic_units(row, require_gate1=False)
    units.update(economic_unit_registry())


def service_outcome_registry() -> list[dict[str, str]]:
    """The complete 41-field Gate-1 service surface in stable native-unit order."""
    rows = [
        {"name": "cultivation.food-delivered", "unit": "food",
         "source": "domains.cultivation.foodDelivered"},
        {"name": "herding.accepted-food", "unit": "food",
         "source": "domains.herding.acceptedFood"},
        {"name": "herding.slaughter-food", "unit": "food",
         "source": "herds.*.slaughterFood"},
    ]
    rows.extend({"name": f"manufacturing.{good}.output", "unit": "good-units",
                 "source": f"goods.{good}.output"} for good in CRAFT_GOODS)
    rows.extend({"name": f"materials.{material}.received", "unit": "material-units",
                 "source": f"materials.{material}.received"} for material in MATERIALS)
    rows.extend([
        {"name": "trading.settled-trades", "unit": "transactions",
         "source": "domains.trade.settledTrades"},
        {"name": "trading.realised-utility", "unit": "person-qol-years",
         "source": "domains.trade.realisedUtility"},
    ])
    for good in TRADE_GOODS:
        rows.extend((
            {"name": f"trading.{good}.imported", "unit": f"{good}-units",
             "source": f"domains.trade.imported{good}"},
            {"name": f"trading.{good}.exported", "unit": f"{good}-units",
             "source": f"domains.trade.exported{good}"},
        ))
    if len(rows) != 41:
        raise AssertionError("Gate-1 service registry cardinality changed")
    return rows


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what} is not a finite numeric value")
    return float(value)


def _nonnegative(value: Any, what: str) -> float:
    result = _finite(value, what)
    if result < 0:
        raise ValueError(f"{what} is negative")
    return result


def _count(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{what} is not a nonnegative integer count")
    return value


def _sha(value: Any, what: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)):
        raise ValueError(f"{what} is not a SHA-256 digest")
    return value


def _walk_finite(value: Any, path: str = "economic") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _walk_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_finite(child, f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
        raise ValueError(f"{path} is nonfinite")


def validate_capture_bounds(bounds: Any, horizon: int) -> dict[str, int]:
    if not isinstance(bounds, dict) or set(bounds) != CAPTURE_BOUND_FIELDS:
        raise ValueError("trajectory capture bounds are absent or malformed")
    for name, ceiling in HARD_CAPTURE_LIMITS.items():
        value = bounds.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > ceiling:
            raise ValueError(f"trajectory capture bound {name} is invalid")
    if bounds["max_steps"] < horizon:
        raise ValueError("trajectory step bound does not cover the frozen evaluation horizon")
    return dict(bounds)


def _identity_value(rows: list[dict], collection: str, identity: str,
                    field: str) -> tuple[float | None, str | None]:
    matches = [row for row in rows if row.get(collection[:-1]) == identity]
    if len(matches) != 1:
        return None, f"expected-one-{collection[:-1]}-{identity}-row"
    value = matches[0].get(field)
    if value is None:
        return None, f"{collection}.{identity}.{field}-unavailable"
    return _finite(value, f"{collection}.{identity}.{field}"), None


def _service_value(row: dict, source: str) -> tuple[float | None, str | None]:
    parts = source.split(".")
    if parts[0] == "herds" and parts[1] == "*":
        values = []
        for item in row["herds"]:
            value = item.get(parts[2])
            if value is None:
                return None, f"herds.*.{parts[2]}-unavailable"
            values.append(_finite(value, source))
        return sum(values), None
    if parts[0] in {"goods", "materials"}:
        return _identity_value(row[parts[0]], parts[0], parts[1], parts[2])
    value: Any = row
    traversed = []
    for part in parts:
        traversed.append(part)
        if not isinstance(value, dict) or part not in value or value[part] is None:
            reason = row["unavailable"].get(".".join(traversed))
            return None, str(reason or f"{'.'.join(traversed)}-unavailable")
        value = value[part]
    return _finite(value, source), None


def _exact_identities(rows: Any, identity: str, expected: set[str], what: str) -> list[dict]:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"economic {what} collection is malformed")
    identities = [row.get(identity) for row in rows]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError(f"economic {what} identity coverage is incomplete or duplicated")
    return rows


def _unique_optional_collection(row: dict, collection: str, identity: str) -> None:
    values = row[collection]
    if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
        raise ValueError(f"economic {collection} collection is malformed")
    identities = [item.get(identity) for item in values]
    if any(value is None for value in identities) or len(identities) != len(set(identities)):
        raise ValueError(f"economic {collection} identities are missing or duplicated")


def _unique_trade_collection(row: dict, schema: int) -> None:
    """Validate the identity surface for the schema-specific trade envelope."""
    values = row["trades"]
    if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
        raise ValueError("economic trades collection is malformed")
    if schema == 1:
        identities = [item.get("transactionId") for item in values]
    else:
        identity_by_type = {
            "eligibility": "eventId",
            "requisition": "requestId",
            "transaction": "transactionId",
        }
        identities = []
        for item in values:
            record_type = item.get("recordType")
            identity_name = identity_by_type.get(record_type)
            identity = item.get(identity_name) if identity_name is not None else None
            if (identity_name is None or not isinstance(identity, str)
                    or not identity.strip()):
                raise ValueError("economic schema-2 trade identity is missing or malformed")
            identities.append((record_type, identity))
    if any(value is None for value in identities) or len(identities) != len(set(identities)):
        raise ValueError("economic trades identities are missing or duplicated")


def validate_economic_rows(rows: Any, *, trajectory_id: str, decision_turn: int,
                           source_villages: list[dict], successor_villages: list[dict]) -> list[dict]:
    """Validate the one authoritative raw economic envelope used by Gate 1."""
    if not isinstance(rows, list):
        raise ValueError("economic transition rows are not a list")
    source_by_camp = {int(village["camp"]): village for village in source_villages}
    expected = {camp for camp, village in source_by_camp.items()
                if not village["terminated"] and village["population"] > 0
                and village["observation"] is not None}
    camps: set[int] = set()
    transitions: set[str] = set()
    by_camp = {int(village["camp"]): village for village in successor_villages}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("economic row does not match the frozen envelope schema")
        schema = row.get("schemaVersion")
        expected_fields = (ECONOMIC_ROW_FIELDS if schema == 1 else
                           ECONOMIC_ROW_FIELDS | GATE2_ADDITIVE_ECONOMIC_ROW_FIELDS
                           if schema == 2 else None)
        if expected_fields is None or set(row) != expected_fields:
            raise ValueError("economic row does not match the frozen envelope schema")
        _walk_finite(row)
        if row["trajectoryId"] != trajectory_id:
            raise ValueError("economic row schema/trajectory identity differs from the opening boundary")
        if row["decisionTurn"] != decision_turn:
            raise ValueError("economic row decision turn differs from its trajectory step")
        camp = row["campId"]
        if isinstance(camp, bool) or not isinstance(camp, int) or camp not in expected or camp in camps:
            raise ValueError("economic camp coverage is missing, duplicated, or unexpected")
        identity = row["transitionId"]
        if not isinstance(identity, str) or not identity or identity in transitions:
            raise ValueError("economic transition identity is missing or duplicated")
        camps.add(camp); transitions.add(identity)
        source = source_by_camp[camp]
        if (isinstance(row["societyId"], bool) or not isinstance(row["societyId"], int)
                or row["societyId"] != source["society"]):
            raise ValueError("economic society identity differs from its decision-bearing source")
        if not isinstance(row["domains"], dict) or set(row["domains"]) != ECONOMIC_DOMAINS:
            raise ValueError("economic domain coverage differs from the frozen schema")
        if not isinstance(row["units"], dict) or not isinstance(row["unavailable"], dict):
            raise ValueError("economic units/missingness declarations are absent")
        if any(not isinstance(key, str) or not isinstance(reason, str) or not reason.strip()
               for key, reason in row["unavailable"].items()):
            raise ValueError("economic missingness reason is malformed")
        years = _finite(row["years"], "economic years")
        if years <= 0:
            raise ValueError("economic duration must be positive")
        population = row["domains"]["population"]
        cultivation = row["domains"]["cultivation"]
        herding = row["domains"]["herding"]
        trade = row["domains"]["trade"]
        if not all(isinstance(block, dict) for block in (population, cultivation, herding, trade)):
            raise ValueError("economic service domains are malformed")
        _validate_declared_economic_units(row, require_gate1=True)
        if any(row["units"].get(path) != unit
               for path, unit in economic_unit_registry().items()):
            raise ValueError("economic source unit differs from the authoritative native unit")
        _nonnegative(population.get("personYears"), "population.personYears")
        qol = _nonnegative(population.get("qol"), "population.qol")
        ending = _count(population.get("endingResidents"), "population.endingResidents")
        village = by_camp[camp]
        if (village["society"] != source["society"]
                or village["priorPopulation"] != source["population"]
                or ending != village["population"] or qol != float(village["qol"])
                or years != float(village["years"])):
            raise ValueError("economic identity/population/QoL/duration conflicts with source/successor state")
        for path, value in (("cultivation.foodDelivered", cultivation.get("foodDelivered")),
                            ("herding.acceptedFood", herding.get("acceptedFood"))):
            if value is not None:
                _nonnegative(value, path)
            elif f"domains.{path}" not in row["unavailable"]:
                raise ValueError(f"{path} is missing without an availability reason")
        goods = _exact_identities(row["goods"], "good", set(CRAFT_GOODS), "goods")
        for item in goods:
            if item.get("unit") != "good-units":
                raise ValueError("manufacturing output has the wrong native unit")
            _nonnegative(item.get("output"), f"goods.{item['good']}.output")
        materials = _exact_identities(row["materials"], "material", set(MATERIALS), "materials")
        for item in materials:
            if item.get("unit") != "material-units":
                raise ValueError("material receipt has the wrong native unit")
            _nonnegative(item.get("received"), f"materials.{item['material']}.received")
        if not isinstance(row["herds"], list) or any(not isinstance(item, dict) for item in row["herds"]):
            raise ValueError("economic herds collection is malformed")
        herd_ids = [item.get("herdId") for item in row["herds"]]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in herd_ids):
            raise ValueError("economic herd identity is malformed")
        if len(herd_ids) != len(set(herd_ids)):
            raise ValueError("economic herd identities are duplicated")
        for item in row["herds"]:
            _nonnegative(item.get("slaughterFood"), f"herds.{item['herdId']}.slaughterFood")
        _count(trade.get("settledTrades"), "trade.settledTrades")
        _finite(trade.get("realisedUtility"), "trade.realisedUtility")
        for good in TRADE_GOODS:
            _nonnegative(trade.get("imported" + good), f"trade.imported{good}")
            _nonnegative(trade.get("exported" + good), f"trade.exported{good}")
        _unique_optional_collection(row, "fields", "fieldId")
        _unique_optional_collection(row, "requests", "requestId")
        _unique_trade_collection(row, schema)
        if not isinstance(row["transfers"], list) or any(not isinstance(item, dict) for item in row["transfers"]):
            raise ValueError("economic transfers collection is malformed")
    if camps != expected:
        raise ValueError("economic rows do not exactly cover transition-bearing camps")
    return rows


def integrate_economic_rows(rows: list[dict]) -> dict[str, float]:
    """Integrate validated simulator diagnostics without reconstructing state."""
    population_years = 0.0
    qol_years = 0.0
    for row in rows:
        years = _finite(row.get("years"), "economic years")
        if years < 0:
            raise ValueError("economic years are negative")
        population = row.get("domains", {}).get("population", {})
        units = row.get("units", {})
        if units.get("domains.population.personYears") != "person-years":
            raise ValueError("population.personYears lacks its authoritative unit")
        if units.get("domains.population.qol") != "qol":
            raise ValueError("population.qol lacks its authoritative unit")
        person_years = _finite(population.get("personYears"), "population.personYears")
        qol = _finite(population.get("qol"), "population.qol")
        if person_years < 0 or qol < 0:
            raise ValueError("integrated population/QoL inputs are negative")
        population_years += person_years
        qol_years += qol * years
    return {"population_years": population_years, "qol_years": qol_years}


def service_outcomes(rows: list[dict], fields: list[dict]) -> dict[str, dict[str, Any]]:
    output = {}
    for field in fields:
        total = 0.0
        reasons = []
        for row in rows:
            value, reason = _service_value(row, field["source"])
            if reason is not None:
                reasons.append(reason)
            else:
                total += float(value)
        output[field["name"]] = ({"value": None, "availability_reason": "; ".join(sorted(set(reasons)))}
                                  if reasons else {"value": total, "availability_reason": None})
    return output


def compiler_interventions(rows: list[dict]) -> tuple[int, dict]:
    """Count named compiler constraints while retaining every contributing row."""
    raw = []
    for row in rows:
        identity = {key: row.get(key) for key in ("transitionId", "decisionTurn", "campId")}
        domains = row["domains"]
        for domain in ("cultivation", "herding"):
            funnels = domains[domain].get("funnels")
            if not isinstance(funnels, list):
                raise ValueError(f"{domain} compiler funnels are unavailable")
            for funnel in funnels:
                reasons = funnel.get("constraintReasons")
                if not isinstance(reasons, list):
                    raise ValueError("compiler constraint reasons are malformed")
                for reason in reasons:
                    if not isinstance(reason, str) or not reason:
                        raise ValueError("compiler constraint reason is malformed")
                    raw.append({**identity, "domain": domain, "reason": reason,
                                "source": f"funnel:{funnel.get('metric')}"})
        for request in row["requests"]:
            reason = request.get("unmetReason")
            if reason:
                domain = "materials" if request.get("rawMaterial") is not None else "manufacturing"
                raw.append({**identity, "domain": domain, "reason": str(reason),
                            "source": f"request:{request.get('requestId')}"})
    raw.sort(key=lambda item: (str(item["transitionId"]), int(item["decisionTurn"]),
                               int(item["campId"]), item["domain"], item["reason"], item["source"]))
    counts: dict[tuple[str, str], int] = {}
    for item in raw:
        key = (item["domain"], item["reason"])
        counts[key] = counts.get(key, 0) + 1
    summary = {"schema": 1, "kind": "gate1_compiler_intervention_summary", "total": len(raw),
               "by_domain_reason": [{"domain": domain, "reason": reason, "count": count}
                                    for (domain, reason), count in sorted(counts.items())],
               "raw_rows": raw}
    summary["digest"] = digest(summary)
    return len(raw), summary


_UNORDERED_IDENTITIES = {
    "goods": "good", "materials": "material", "herds": "herdId", "fields": "fieldId",
    "requests": "ordinal", "trades": "transactionId", "transfers": None,
    "villages": "camp", "actions": "camp", "economic": "campId",
}


def _semantic_identity(item: dict[str, Any], identity: str) -> Any:
    if identity != "transactionId" or "recordType" not in item:
        return item.get(identity)
    identity_by_type = {
        "eligibility": "eventId",
        "requisition": "requestId",
        "transaction": "transactionId",
    }
    record_type = item.get("recordType")
    name = identity_by_type.get(record_type)
    return (record_type, item.get(name) if name is not None else None)


def _canonical_semantic(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {name: _canonical_semantic(value[name], name) for name in sorted(value)}
    if isinstance(value, list):
        values = [_canonical_semantic(item) for item in value]
        if key in _UNORDERED_IDENTITIES:
            identity = _UNORDERED_IDENTITIES[key]
            values.sort(key=(lambda item: (str(_semantic_identity(item, identity)), digest(item))) if identity
                        else lambda item: digest(item))
        return values
    if isinstance(value, np.generic):
        return value.item()
    return value


def capture_component_bytes(value: Any) -> int:
    """Return the exact retained canonical JSON size without retaining a second representation."""
    return len(json.dumps(_canonical_semantic(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8"))


def capture_envelope_bytes(reset_bytes: int, opening_bytes: int,
                           step_bytes: list[int]) -> int:
    """Canonical retained bytes for reset/warm-up, opening, and transitions.

    The keys are in the same sorted order used by :func:`_canonical_semantic`.
    Counting the reset record is important: its warm-up chain is retained raw
    evidence, not free metadata outside the reviewed capture budget.
    """
    return (len(b'{"opening":') + opening_bytes + len(b',"reset":') + reset_bytes
            + len(b',"steps":[') + sum(step_bytes) + max(0, len(step_bytes) - 1)
            + len(b']}'))


def boundary(step: dict) -> dict:
    """Capture only the semantic reset/successor boundary required by Gate 1."""
    villages = []
    for village in sorted(step["villages"], key=lambda row: int(row["camp"])):
        captured = {key: copy.deepcopy(village.get(key))
                    for key in BOUNDARY_VILLAGE_FIELDS - {"observation_sha256"}}
        captured["observation_sha256"] = (None if captured["observation"] is None
                                           else digest(captured["observation"]))
        villages.append(captured)
    return {"turn": int(step["turn"]), "fingerprint": step["fingerprint"], "villages": villages}


def _validate_observation(value: Any, expected_core_width: int, what: str) -> None:
    required = {"core", "corePresent", "sets", "rowPresent", "movementMask",
                "regionMovementMask", "clothingRouteMask", "shelterRouteMask",
                "economyMixMask", "commitHoldMask"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{what} observation schema differs from native contract 12")
    core, present = value["core"], value["corePresent"]
    if (not isinstance(core, list) or len(core) != expected_core_width
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) for item in core)
            or not isinstance(present, list) or len(present) != expected_core_width
            or any(type(item) is not bool for item in present)):
        raise ValueError(f"{what} core observation layout is malformed")
    sets, row_present = value["sets"], value["rowPresent"]
    if (not isinstance(sets, list) or not isinstance(row_present, list)
            or len(sets) != len(contracts.SETS) or len(row_present) != len(contracts.SETS)):
        raise ValueError(f"{what} entity-set observation coverage is malformed")
    for index, (name, rows, width) in enumerate(contracts.SETS):
        values, masks = sets[index], row_present[index]
        if (not isinstance(values, list) or len(values) != rows * width
                or any(isinstance(item, bool) or not isinstance(item, (int, float))
                       or not math.isfinite(item) for item in values)
                or not isinstance(masks, list) or len(masks) != rows
                or any(type(item) is not bool for item in masks)):
            raise ValueError(f"{what} {name} observation layout is malformed")
    masks = {
        "movementMask": contracts.MOVEMENT_SLOTS,
        "regionMovementMask": contracts.REGION_MOVEMENT_SLOTS,
        "clothingRouteMask": 2, "shelterRouteMask": 2,
        "economyMixMask": len(contracts.ECONOMY_MIX),
    }
    for name, width in masks.items():
        if (not isinstance(value[name], list) or len(value[name]) != width
                or any(type(item) is not bool for item in value[name])):
            raise ValueError(f"{what} {name} observation mask is malformed")
    if type(value["commitHoldMask"]) is not bool:
        raise ValueError(f"{what} commitment observation mask is malformed")


_OBSERVATION_MASKS = (
    "movementMask", "regionMovementMask", "clothingRouteMask",
    "shelterRouteMask", "economyMixMask", "commitHoldMask",
)


def _tensor_row(value: torch.Tensor, row: int) -> dict[str, Any]:
    """Canonical JSON representation of one CPU decoder-input row."""
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.device.type != "cpu" or value.dtype not in {
                torch.float32, torch.float64, torch.int64, torch.bool,
            } or value.ndim < 1 or not 0 <= row < len(value)
            or value.storage_offset() != 0
            or any(stride <= 0 for stride in value.stride())):
        raise ValueError("decoder receipt tensor has an unsupported dtype/device/layout/row")
    selected = value[row].detach().contiguous()
    if selected.dtype in {torch.float32, torch.float64} and not torch.isfinite(selected).all():
        raise ValueError("decoder receipt tensor contains a nonfinite floating value")
    names = {torch.float32: "float32", torch.float64: "float64",
             torch.int64: "int64", torch.bool: "bool"}
    return {"dtype": names[selected.dtype], "shape": list(selected.shape),
            "stride": list(value.stride()), "storage_offset": 0,
            "values": selected.tolist()}


def _canonical_tensor_leaf(value: Any, dtype: str, what: str) -> None:
    """Validate one original JSON scalar without coercing or inferring its type."""
    if dtype == "bool":
        if type(value) is not bool:
            raise ValueError(f"{what} tensor is not canonical bool")
        return
    if dtype == "int64":
        if type(value) is not int or not -(2 ** 63) <= value < 2 ** 63:
            raise ValueError(f"{what} tensor is not canonical int64")
        return
    if dtype not in {"float32", "float64"} or type(value) not in {int, float}:
        raise ValueError(f"{what} tensor is not a canonical numeric scalar")
    if type(value) is int:
        precision, max_exponent = ((24, 127) if dtype == "float32" else (53, 1023))
        magnitude = abs(value)
        if magnitude:
            bits = magnitude.bit_length()
            discarded = max(0, bits - precision)
            if bits - 1 > max_exponent or magnitude % (1 << discarded):
                raise ValueError(f"{what} tensor integer is not exactly representable as {dtype}")
        try:
            normalized = struct.unpack(">f" if dtype == "float32" else ">d",
                                       struct.pack(">f" if dtype == "float32" else ">d",
                                                   float(value)))[0]
        except (OverflowError, struct.error):
            raise ValueError(f"{what} tensor integer exceeds the {dtype} range") from None
        if not math.isfinite(normalized) or not normalized.is_integer() or int(normalized) != value:
            raise ValueError(f"{what} tensor integer is not exactly representable as {dtype}")
        return
    if not math.isfinite(value):
        raise ValueError(f"{what} tensor is not finite canonical {dtype}")
    try:
        normalized = struct.unpack(">f" if dtype == "float32" else ">d",
                                   struct.pack(">f" if dtype == "float32" else ">d", value))[0]
    except (OverflowError, struct.error):
        raise ValueError(f"{what} tensor exceeds the {dtype} range") from None
    if (not math.isfinite(normalized) or normalized != value
            or normalized == 0.0
            and math.copysign(1.0, normalized) != math.copysign(1.0, value)):
        raise ValueError(f"{what} tensor is not finite canonical {dtype}")


def _validated_tensor_tree(value: Any, dtype: str, what: str) -> tuple[int, ...]:
    """Return the exact rectangular JSON shape after validating every original leaf."""
    if isinstance(value, list):
        children = [_validated_tensor_tree(child, dtype, what) for child in value]
        if children and any(child != children[0] for child in children[1:]):
            raise ValueError(f"{what} tensor values are ragged")
        return (len(value), *(children[0] if children else ()))
    _canonical_tensor_leaf(value, dtype, what)
    return ()


def _tensor_from_row(value: Any, *, dtype: str, shape: tuple[int, ...],
                     batch_rows: int, row: int, what: str) -> torch.Tensor:
    full_shape = (batch_rows, *shape)
    if (not isinstance(value, dict)
            or set(value) != {"dtype", "shape", "stride", "storage_offset", "values"}
            or value.get("dtype") != dtype):
        raise ValueError(f"{what} tensor declaration differs from the decoder contract")
    observed_shape = _validated_tensor_tree(value["values"], dtype, what)
    if observed_shape != shape:
        raise ValueError(f"{what} tensor values differ from the declared shape")
    if (value.get("shape") != list(shape)
            or value.get("storage_offset") != 0
            or not isinstance(value.get("stride"), list)
            or len(value["stride"]) != len(full_shape)
            or any(type(stride) is not int or stride <= 0 for stride in value["stride"])):
        raise ValueError(f"{what} tensor declaration differs from the decoder contract")
    torch_dtype = {
        "float32": torch.float32, "float64": torch.float64,
        "int64": torch.int64, "bool": torch.bool,
    }[dtype]
    dimensions = sorted((stride, size) for stride, size in zip(
        value["stride"], full_shape, strict=True,
    ) if size > 1)
    storage_span = 1
    for stride, size in dimensions:
        if stride < storage_span:
            raise ValueError(f"{what} tensor stride overlaps logical values")
        storage_span += (size - 1) * stride
    if storage_span > 4 * max(1, math.prod(full_shape)):
        raise ValueError(f"{what} tensor stride exceeds the bounded decoder layout")
    try:
        restored = torch.empty_strided(full_shape, tuple(value["stride"]), dtype=torch_dtype)
    except RuntimeError as exc:
        raise ValueError(f"{what} tensor stride is unsupported") from exc
    restored.zero_()
    restored[row] = torch.tensor(value["values"], dtype=torch_dtype).reshape(shape)
    return restored


def _deterministic_shapes(decoder: str, semantics: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    common = {
        "strategic_mean": ("float32", (contracts.UNIT_STRATEGIC,)),
        "mix_logits": ("float32", (len(contracts.ECONOMY_MIX),)),
        "service_mean": ("float32", (len(contracts.SERVICES),)),
        "clothing": ("float32", (2,)), "shelter": ("float32", (2,)),
        "fission": ("float32", (contracts.FISSION_POLICIES,)),
        "movement": ("float32", (contracts.MOVEMENT_SLOTS,)),
    }
    if decoder == "pinned-contract-8":
        return common
    if decoder != "native-contract-12":
        raise ValueError("decoder receipt names an unsupported implementation")
    common.update({
        "span": ("float32", (len(contracts.SPANS),)),
        "trade_value_mean": ("float32", (len(contracts.TRADE_GROUPS),)),
        "trade_gate": ("float32", (len(contracts.TRADE_GROUPS), contracts.TRADE_GATES)),
        "trade_share_mean": ("float32", ()),
    })
    if semantics["learned_movement"] and semantics["region_movement"]:
        common.update({
            "region": ("float32", (contracts.REGION_MOVEMENT_SLOTS,)),
            "region_logp": ("float32", (contracts.REGION_MOVEMENT_SLOTS,)),
            "mix_offset": ("float32", (contracts.REGION_MOVEMENT_SLOTS,
                                         len(contracts.ECONOMY_MIX))),
        })
        if semantics["region_head"] == "hierarchical":
            common["region_group"] = ("float32", (3,))
    return common


def _stochastic_shapes(semantics: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    shapes = {
        "region": ("int64", ()),
        "strategic_z": ("float32", (contracts.UNIT_STRATEGIC,)),
        "service_z": ("float32", (len(contracts.SERVICES),)),
        "mix_z": ("float32", (len(contracts.ECONOMY_MIX),)),
        "clothing": ("int64", ()), "shelter": ("int64", ()),
        "movement": ("int64", ()), "span": ("int64", ()),
        "fission": ("int64", ()),
    }
    if semantics["learned_trade"]:
        shapes.update({
            "trade_value_z": ("float32", (len(contracts.TRADE_GROUPS),)),
            "trade_gate": ("int64", (len(contracts.TRADE_GROUPS),)),
            "trade_share_z": ("float32", ()),
        })
    return shapes


def action_decoder_receipt(*, mode: str, decoder: str, semantics: dict,
                           observation: dict, emitted_action: dict, camp: int,
                           raw: dict[str, torch.Tensor], row: int,
                           economy_mix_mask: torch.Tensor | None = None) -> dict:
    """Seal the exact float32 inputs consumed by the production action decoder."""
    shapes = (_deterministic_shapes(decoder, semantics) if mode == "deterministic"
              else _stochastic_shapes(semantics) if mode == "seeded-stochastic" else None)
    if shapes is None or set(raw) < set(shapes):
        raise ValueError("decoder receipt lacks an exact active decoder input")
    batch_rows = len(raw[next(iter(shapes))])
    if any(len(raw[name]) != batch_rows for name in shapes):
        raise ValueError("decoder receipt inputs do not share an exact batch shape")
    encoded = {name: _tensor_row(raw[name], row) for name in sorted(shapes)}
    payload: dict[str, Any] = {
        "schema": 2, "kind": "gate1_action_decoder_receipt", "mode": mode,
        "decoder": decoder, "semantics": copy.deepcopy(semantics),
        "batch_rows": batch_rows, "row": row, "camp": camp,
        "observation_sha256": digest(observation),
        "observation_masks": {name: copy.deepcopy(observation[name])
                              for name in _OBSERVATION_MASKS},
        "raw": encoded, "emitted_action": copy.deepcopy(emitted_action),
    }
    if mode == "seeded-stochastic":
        if economy_mix_mask is None:
            raise ValueError("stochastic decoder receipt lacks its exact economy mask tensor")
        payload["economy_mix_mask"] = _tensor_row(economy_mix_mask, row)
    else:
        payload["economy_mix_mask"] = None
    payload["digest"] = digest(payload)
    return payload


def _validate_decoder_receipt(receipt: Any, action: dict, semantics: dict,
                              observation: dict, action_mode: str,
                              expected_decoder: str, expected_camp: int,
                              what: str) -> np.ndarray:
    fields = {"schema", "kind", "mode", "decoder", "semantics", "batch_rows", "row", "camp",
              "observation_sha256", "observation_masks", "raw", "emitted_action",
              "economy_mix_mask", "digest"}
    if (not isinstance(receipt, dict) or set(receipt) != fields
            or receipt.get("schema") != 2
            or receipt.get("kind") != "gate1_action_decoder_receipt"
            or receipt.get("mode") != action_mode or receipt.get("decoder") != expected_decoder
            or receipt.get("semantics") != semantics
            or type(receipt.get("batch_rows")) is not int or receipt["batch_rows"] <= 0
            or type(receipt.get("row")) is not int
            or not 0 <= receipt["row"] < receipt["batch_rows"]
            or receipt.get("camp") != expected_camp
            or receipt.get("observation_sha256") != digest(observation)
            or receipt.get("observation_masks") != {
                name: observation[name] for name in _OBSERVATION_MASKS
            }
            or digest(receipt.get("emitted_action")) != digest(action)
            or receipt.get("digest") != digest({key: value for key, value in receipt.items()
                                                 if key != "digest"})):
        raise ValueError(f"{what} decoder receipt is malformed or not bound to its action/input")
    shapes = (_deterministic_shapes(expected_decoder, semantics)
              if action_mode == "deterministic" else _stochastic_shapes(semantics))
    raw = receipt["raw"]
    if not isinstance(raw, dict) or set(raw) != set(shapes):
        raise ValueError(f"{what} decoder receipt input set differs from its active heads")
    decoded_raw = {
        name: _tensor_from_row(raw[name], dtype=dtype, shape=shape,
                               batch_rows=receipt["batch_rows"], row=receipt["row"],
                               what=f"{what} {name}")
        for name, (dtype, shape) in shapes.items()
    }
    if action_mode == "deterministic":
        if receipt["economy_mix_mask"] is not None:
            raise ValueError(f"{what} deterministic receipt carries a stochastic mask tensor")
        if expected_decoder == "pinned-contract-8":
            from .legacy_contract8 import deterministic_mode
            decoded_mode = deterministic_mode(decoded_raw)
        else:
            from .model import act_deterministic
            decoded_mode = act_deterministic(
                decoded_raw,
                region=semantics["learned_movement"] and semantics["region_movement"],
                grouped=semantics["grouped_region"], selection=semantics["region_selection"],
            )
        from .model import bridge_actions_from_mode
        replayed = bridge_actions_from_mode(
            decoded_mode, movement=semantics["learned_movement"],
            region_movement=semantics["region_movement"],
            commitments=semantics["commitments"],
            learned_fission=semantics["learned_fission"],
            learned_trade=semantics["learned_trade"],
        )
        decoded_mix = decoded_mode["mix"][receipt["row"]]
    else:
        mask = _tensor_from_row(
            receipt["economy_mix_mask"], dtype="bool",
            shape=(len(contracts.ECONOMY_MIX),), batch_rows=receipt["batch_rows"],
            row=receipt["row"], what=f"{what} economy mix mask",
        )
        if mask[receipt["row"]].tolist() != observation["economyMixMask"]:
            raise ValueError(f"{what} stochastic decoder mask differs from the source observation")
        from .ppo import mix_from_ratios, to_bridge_actions
        replayed = to_bridge_actions(decoded_raw, {"economy_mix_mask": mask})
        decoded_mix = mix_from_ratios(decoded_raw["mix_z"], mask)[receipt["row"]].numpy()
    if digest(replayed[receipt["row"]]) != digest(action):
        raise ValueError(f"{what} emitted action differs from exact forward decoder replay")
    return decoded_mix


def _validate_action(value: Any, semantics: dict, what: str,
                     observation: dict | None = None, *,
                     action_mode: str = "deterministic",
                     decoder_receipt: dict | None = None,
                     expected_decoder: str = "native-contract-12",
                     expected_camp: int = 1) -> None:
    required = {"strategic", "service", "clothingRoute", "shelterRoute",
                "movement", "fission", "commit"}
    if semantics["learned_trade"]:
        required |= {"tradeValue", "tradeGate", "tradeShare"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{what} action schema differs from its active checkpoint heads")
    for name, width in (("strategic", len(contracts.STRATEGIC)),
                        ("service", len(contracts.SERVICES))):
        vector = value[name]
        if (not isinstance(vector, list) or len(vector) != width
                or any(isinstance(item, bool) or not isinstance(item, (int, float))
                       or not math.isfinite(item) or not 0 <= item <= 1 for item in vector)):
            raise ValueError(f"{what} {name} action is malformed")
    if type(value["clothingRoute"]) is not int or value["clothingRoute"] not in (0, 1):
        raise ValueError(f"{what} clothing route is malformed")
    if type(value["shelterRoute"]) is not int or value["shelterRoute"] not in (0, 1):
        raise ValueError(f"{what} shelter route is malformed")
    movement = value["movement"]
    if type(movement) is not int:
        raise ValueError(f"{what} movement action is malformed")
    if not semantics["learned_movement"] and movement != contracts.NATIVE_MOVEMENT:
        raise ValueError(f"{what} unexpectedly claims learned movement")
    if semantics["learned_movement"] and semantics["region_movement"]:
        if movement not in {contracts.NATIVE_MOVEMENT, contracts.KEEP_INTENT,
                            *range(contracts.REGION_BASE,
                                   contracts.REGION_BASE + contracts.REGION_ROWS)}:
            raise ValueError(f"{what} regional movement action is malformed")
    elif semantics["learned_movement"] and not -1 <= movement < contracts.MOVEMENT_SLOTS - 1:
        raise ValueError(f"{what} site movement action is malformed")
    fission = value["fission"]
    if (type(fission) is not int
            or (fission not in range(contracts.FISSION_POLICIES)
                if semantics["learned_fission"] else fission != -1)):
        raise ValueError(f"{what} fission action is malformed")
    commit = value["commit"]
    if (type(commit) is not int
            or (commit not in contracts.SPANS if semantics["commitments"]
                else commit != contracts.SPAN_SINGLE)):
        raise ValueError(f"{what} commitment action is malformed")
    if semantics["learned_trade"]:
        values, gates, share = value["tradeValue"], value["tradeGate"], value["tradeShare"]
        lo, hi = contracts.TRADE_VALUE_RANGE
        if (not isinstance(values, list) or len(values) != len(contracts.TRADE_GROUPS)
                or any(isinstance(item, bool) or not isinstance(item, (int, float))
                       or not math.isfinite(item) or not lo <= item <= hi for item in values)
                or not isinstance(gates, list) or len(gates) != len(contracts.TRADE_GROUPS)
                or any(type(item) is not int or item not in range(contracts.TRADE_GATES)
                       for item in gates)
                or isinstance(share, bool) or not isinstance(share, (int, float))
                or not math.isfinite(share) or not 0 <= share <= 1):
            raise ValueError(f"{what} trade action is malformed")
    if action_mode not in {"deterministic", "seeded-stochastic"}:
        raise ValueError(f"{what} has an unsupported action mode")
    if observation is None:
        raise ValueError(f"{what} lacks the source observation required by its decoder receipt")
    decoded_mix = _validate_decoder_receipt(
        decoder_receipt, value, semantics, observation, action_mode, expected_decoder,
        expected_camp, what,
    )
    from .model import validate_deterministic_level_image
    if action_mode == "deterministic":
        # Validate the imported representation in the exact float32 image used
        # by ``act_deterministic``; Python-double reconstruction is not faithful
        # around the .01/.99 thresholds.
        try:
            validate_deterministic_level_image(
                [*value["strategic"][:-2], *value["service"]],
            )
        except ValueError as exc:
            raise ValueError(f"{what} is not in the deterministic level decoder image: {exc}") from exc
        if semantics["learned_trade"] and 0.0 < value["tradeShare"] < 0.01:
            raise ValueError(f"{what} is not in the deterministic trade-share decoder image")
        if (semantics["learned_movement"] and semantics["region_movement"]
                and semantics.get("region_selection") == "stay"
                and movement != contracts.REGION_BASE):
            raise ValueError(f"{what} substitutes the forced deterministic region selection")
    if not observation["clothingRouteMask"][value["clothingRoute"]]:
        raise ValueError(f"{what} clothing route is illegal under its source observation")
    if not observation["shelterRouteMask"][value["shelterRoute"]]:
        raise ValueError(f"{what} shelter route is illegal under its source observation")
    if any(float(amount) > 0.0 and not legal
           for amount, legal in zip(decoded_mix, observation["economyMixMask"], strict=True)):
        raise ValueError(f"{what} economy mix uses an unavailable source activity")
    if semantics["learned_movement"]:
        if semantics["region_movement"]:
            if movement == contracts.NATIVE_MOVEMENT:
                movement_slot = contracts.REGION_NATIVE
            elif movement == contracts.KEEP_INTENT:
                movement_slot = contracts.REGION_KEEP
            else:
                movement_slot = movement - contracts.REGION_BASE + 2
            if not observation["regionMovementMask"][movement_slot]:
                raise ValueError(f"{what} regional movement is illegal under its source observation")
        else:
            movement_slot = movement + 1
            if not observation["movementMask"][movement_slot]:
                raise ValueError(f"{what} site movement is illegal under its source observation")
    if semantics["commitments"] and commit == contracts.SPAN_HOLD and not observation["commitHoldMask"]:
        raise ValueError(f"{what} commitment hold is illegal under its source observation")


def _validate_boundary(value: Any, what: str, bounds: dict, expected_core_width: int) -> dict:
    if (not isinstance(value, dict) or set(value) != {"turn", "fingerprint", "villages"}
            or isinstance(value.get("turn"), bool) or not isinstance(value.get("turn"), int)
            or value["turn"] < 0):
        raise ValueError(f"{what} boundary is malformed")
    _sha(value["fingerprint"], f"{what} fingerprint")
    villages = value["villages"]
    if (not isinstance(villages, list) or not villages
            or len(villages) > bounds["max_villages_per_boundary"]):
        raise ValueError(f"{what} village boundary is absent or exceeds its reviewed bound")
    camps = []
    for village in villages:
        if not isinstance(village, dict) or set(village) != BOUNDARY_VILLAGE_FIELDS:
            raise ValueError(f"{what} village boundary has an incomplete schema")
        camp = village["camp"]
        if isinstance(camp, bool) or not isinstance(camp, int) or camp < 0:
            raise ValueError(f"{what} village camp identity is malformed")
        camps.append(camp)
        for key in ("society", "parentSociety", "priorPopulation", "population"):
            _count(village[key], f"{what}.{camp}.{key}")
        for key in ("qol", "years", "welfare", "tradeUtility"):
            _finite(village[key], f"{what}.{camp}.{key}")
        if village["years"] < 0 or village["qol"] < 0:
            raise ValueError(f"{what} village has an invalid physical boundary")
        if type(village["terminated"]) is not bool or type(village["founded"]) is not bool:
            raise ValueError(f"{what} village flags are malformed")
        observation = village["observation"]
        if village["terminated"]:
            if village["population"] != 0 or observation is not None or village["observation_sha256"] is not None:
                raise ValueError(f"{what} terminated village retains population or a live observation")
        else:
            if village["population"] <= 0:
                raise ValueError(f"{what} nonterminal village has no live population")
            _validate_observation(observation, expected_core_width, f"{what}.{camp}")
            if village["observation_sha256"] != digest(observation):
                raise ValueError(f"{what} policy observation fingerprint is absent or inconsistent")
    if len(camps) != len(set(camps)) or camps != sorted(camps):
        raise ValueError(f"{what} village identities are duplicated or unstably ordered")
    return value


def draw_order_from_semantics(semantics: dict) -> list[str]:
    order = (["region"] if semantics["learned_movement"] and semantics["region_movement"] else [])
    order += ["mix", "strategic", "service", "clothing", "shelter"]
    if semantics["learned_movement"] and not semantics["region_movement"]:
        order.append("movement")
    if semantics["commitments"]:
        order.append("commitment-span")
    if semantics["learned_fission"]:
        order.append("fission")
    if semantics["learned_trade"]:
        order.extend(("trade-value", "trade-gate", "trade-share"))
    return order


def _advance_generator(generator: torch.Generator, rows: int, draw_order: list[str]) -> None:
    normal_widths = {"mix": len(contracts.ECONOMY_MIX), "strategic": contracts.UNIT_STRATEGIC,
                     "service": len(contracts.SERVICES), "trade-value": len(contracts.TRADE_GROUPS),
                     "trade-share": 1}
    categorical = {"region": contracts.REGION_MOVEMENT_SLOTS, "clothing": 2, "shelter": 2,
                   "movement": contracts.MOVEMENT_SLOTS, "commitment-span": len(contracts.SPANS),
                   "fission": contracts.FISSION_POLICIES}
    for name in draw_order:
        if name in normal_widths:
            torch.randn((rows, normal_widths[name]), generator=generator)
        elif name == "trade-gate":
            torch.multinomial(torch.ones((rows * len(contracts.TRADE_GROUPS), contracts.TRADE_GATES)),
                              1, generator=generator)
        elif name in categorical:
            torch.multinomial(torch.ones((rows, categorical[name])), 1, generator=generator)
        else:
            raise ValueError(f"unknown policy RNG draw {name}")


def recompute_rng_receipt(schedule: dict, rows: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(schedule["seed"])
    before = generator_state_sha256(generator)
    if before != schedule["state_sha256"]:
        raise ValueError("policy RNG initial state differs from frozen schedule")
    _advance_generator(generator, rows, schedule["draw_order"])
    return {"step": schedule["step"], "seed": schedule["seed"],
            "draw_order": schedule["draw_order"], "state_before_sha256": before,
            "state_after_sha256": generator_state_sha256(generator), "rows": rows}


def _capture_receipt(reset: dict, opening: dict, steps: list[dict]) -> dict:
    reset_bytes = capture_component_bytes(reset)
    opening_bytes = capture_component_bytes(opening)
    step_bytes = [capture_component_bytes(step) for step in steps]
    return {"steps": len(steps),
            "warmup_steps": len(reset["warmup_chain"]),
            "max_villages_observed": max([len(opening["villages"])]
                                          + [len(step["successor"]["villages"]) for step in steps]),
            "max_economic_rows_observed": max([0] + [len(step["economic"]) for step in steps]),
            "canonical_bytes": capture_envelope_bytes(reset_bytes, opening_bytes, step_bytes)}


def trajectory_evidence(opening: dict, steps: list[dict], bounds: dict, *, world_id: str,
                        snapshot_sha256: str, reset_turn: int | None = None,
                        reset_fingerprint: str | None = None,
                        warmup_chain: list[dict] | None = None) -> dict:
    """Build the complete canonical Gate-1 trajectory artifact."""
    reset_turn = opening["turn"] if reset_turn is None else reset_turn
    reset_fingerprint = opening["fingerprint"] if reset_fingerprint is None else reset_fingerprint
    reset = {"world_id": world_id, "snapshot_sha256": snapshot_sha256,
             "turn": reset_turn, "fingerprint": reset_fingerprint,
             "trajectory_id": reset_fingerprint,
             "warmup_chain": copy.deepcopy(warmup_chain or [])}
    surface = {"schema": 2, "kind": "gate1_canonical_trajectory",
               "ordering": "step-order; boundary/action/economic identities canonicalized by registered key",
               "reset": reset,
               "bounds": copy.deepcopy(bounds), "opening": _canonical_semantic(opening),
               "steps": _canonical_semantic(steps)}
    surface["capture_receipt"] = _capture_receipt(
        surface["reset"], surface["opening"], surface["steps"],
    )
    surface["digest"] = digest(surface)
    return surface


def trajectory_digest(opening: dict, steps: list[dict], bounds: dict, *, world_id: str,
                      snapshot_sha256: str) -> str:
    return trajectory_evidence(opening, steps, bounds, world_id=world_id,
                               snapshot_sha256=snapshot_sha256)["digest"]


def validate_trajectory_evidence(evidence: Any, *, world_id: str, snapshot_sha256: str,
                                 horizon: int, bounds: dict, action_mode: str,
                                 rng_steps: list[dict] | None,
                                 service_fields: list[dict], expected_opening_turn: int,
                                 observation_core_width: int, policy_semantics: dict,
                                 expected_reset_turn: int | None = None,
                                 expected_decoder: str = "native-contract-12") -> dict[str, Any]:
    """Validate continuity and recompute every mandatory Gate-1 outcome."""
    required = {"schema", "kind", "ordering", "reset", "bounds", "opening", "steps",
                "capture_receipt", "digest"}
    if (not isinstance(evidence, dict) or set(evidence) != required
            or evidence.get("schema") != 2 or evidence.get("kind") != "gate1_canonical_trajectory"
            or evidence.get("digest") != digest({k: v for k, v in evidence.items() if k != "digest"})):
        raise ValueError("trajectory evidence is malformed or not content-addressed")
    reset = evidence["reset"]
    reset_fields = {"world_id", "snapshot_sha256", "turn", "fingerprint",
                    "trajectory_id", "warmup_chain"}
    if (not isinstance(reset, dict) or set(reset) != reset_fields
            or reset["world_id"] != world_id or reset["snapshot_sha256"] != snapshot_sha256):
        raise ValueError("trajectory opening reset/snapshot identity differs from the frozen world")
    expected_reset_turn = expected_opening_turn if expected_reset_turn is None else expected_reset_turn
    if (type(reset["turn"]) is not int or reset["turn"] != expected_reset_turn
            or reset["turn"] > expected_opening_turn):
        raise ValueError("trajectory reset turn differs from the frozen world")
    _sha(reset["fingerprint"], "trajectory reset fingerprint")
    if reset["trajectory_id"] != reset["fingerprint"]:
        raise ValueError("trajectory reset identity is not bound to the reset fingerprint")
    warmup = reset["warmup_chain"]
    if (not isinstance(warmup, list)
            or len(warmup) != expected_opening_turn - expected_reset_turn):
        raise ValueError("trajectory warm-up chain does not cover reset to evaluation opening")
    chain_turn, chain_fingerprint = reset["turn"], reset["fingerprint"]
    warmup_fields = {"index", "source_turn", "source_fingerprint", "successor_turn",
                     "successor_fingerprint", "trajectory_id"}
    for index, link in enumerate(warmup):
        if (not isinstance(link, dict) or set(link) != warmup_fields or link["index"] != index
                or link["source_turn"] != chain_turn
                or link["source_fingerprint"] != chain_fingerprint
                or link["successor_turn"] != chain_turn + 1
                or link["trajectory_id"] != reset["trajectory_id"]):
            raise ValueError("trajectory warm-up chain is malformed or discontinuous")
        _sha(link["successor_fingerprint"], "trajectory warm-up successor fingerprint")
        if link["successor_fingerprint"] == chain_fingerprint:
            raise ValueError("trajectory warm-up transition did not advance state")
        chain_turn, chain_fingerprint = link["successor_turn"], link["successor_fingerprint"]
    expected_bounds = validate_capture_bounds(bounds, horizon)
    if evidence["bounds"] != expected_bounds:
        raise ValueError("trajectory evidence substitutes unreviewed capture bounds")
    opening = _validate_boundary(
        evidence["opening"], "opening", expected_bounds, observation_core_width,
    )
    if opening["turn"] != expected_opening_turn:
        raise ValueError("trajectory opening turn differs from reviewed reset plus warm-up semantics")
    if opening["turn"] != chain_turn or opening["fingerprint"] != chain_fingerprint:
        raise ValueError("trajectory evaluation opening is not the end of its warm-up chain")
    steps = evidence["steps"]
    if not isinstance(steps, list) or not steps or len(steps) > horizon:
        raise ValueError("trajectory has no transition or exceeds the frozen horizon")
    if len(steps) > expected_bounds["max_steps"]:
        raise ValueError("trajectory exceeds its reviewed step bound")
    if evidence["capture_receipt"] != _capture_receipt(reset, opening, steps):
        raise ValueError("trajectory capture receipt does not derive from retained evidence")
    receipt = evidence["capture_receipt"]
    if (receipt["max_villages_observed"] > expected_bounds["max_villages_per_boundary"]
            or receipt["max_economic_rows_observed"] > expected_bounds["max_economic_rows_per_step"]
            or receipt["canonical_bytes"] > expected_bounds["max_canonical_bytes"]):
        raise ValueError("trajectory retained evidence exceeds a reviewed capture bound")
    prior = opening
    transition_ids: set[str] = set()
    economic_ids: set[str] = set()
    camp_turns: set[tuple[int, int]] = set()
    economic_rows: list[dict] = []
    welfare = 0.0
    trade_utility = 0.0
    receipts = []
    for index, step in enumerate(steps):
        required_step = {"index", "transition_id", "source_turn", "source_fingerprint", "actions",
                         "policy_rng_receipt", "successor", "terminal", "economic"}
        if not isinstance(step, dict) or set(step) != required_step or step["index"] != index:
            raise ValueError("trajectory step schema/index is incomplete or noncontiguous")
        if step["source_turn"] != prior["turn"] or step["source_fingerprint"] != prior["fingerprint"]:
            raise ValueError("trajectory successor-to-next-source continuity is broken")
        successor = _validate_boundary(
            step["successor"], f"step-{index}-successor", expected_bounds,
            observation_core_width,
        )
        if successor["turn"] != prior["turn"] + 1 or successor["fingerprint"] == prior["fingerprint"]:
            raise ValueError("trajectory transition did not advance exactly one turn")
        expected_transition = digest({"index": index, "source_turn": prior["turn"],
                                      "source_fingerprint": prior["fingerprint"],
                                      "successor_turn": successor["turn"],
                                      "successor_fingerprint": successor["fingerprint"]})
        if step["transition_id"] != expected_transition or step["transition_id"] in transition_ids:
            raise ValueError("trajectory transition identity is missing, duplicated, or unbound")
        transition_ids.add(step["transition_id"])
        actions = step["actions"]
        if not isinstance(actions, list):
            raise ValueError("trajectory action rows are absent")
        action_camps = []
        for action in actions:
            if (not isinstance(action, dict)
                    or set(action) != {"camp", "action_request", "decoded_action",
                                           "decoder_receipt"}
                    or type(action.get("camp")) is not int or action["camp"] < 0
                    or not isinstance(action.get("decoder_receipt"), dict)
                    or action["action_request"] != action["decoded_action"]):
                raise ValueError("trajectory action request/decoded action is malformed or inconsistent")
            action_camps.append(action["camp"])
            _walk_finite(action["decoded_action"], "decoded action")
        if action_camps != sorted(action_camps) or len(action_camps) != len(set(action_camps)):
            raise ValueError("trajectory action camp identities are duplicated or unstably ordered")
        if (sorted(action["decoder_receipt"].get("row", -1) for action in actions)
                != list(range(len(actions)))
                or any(action["decoder_receipt"].get("batch_rows") != len(actions)
                       for action in actions)):
            raise ValueError("trajectory decoder receipts do not cover the exact policy batch")
        active = sorted(int(village["camp"]) for village in prior["villages"]
                        if not village["terminated"] and village["observation"] is not None)
        if action_camps != active:
            raise ValueError("trajectory actions do not exactly cover the observations that produced them")
        source_by_camp = {int(village["camp"]): village for village in prior["villages"]}
        for action in actions:
            _validate_action(
                action["action_request"], policy_semantics,
                f"trajectory step {index} camp {action['camp']}",
                source_by_camp[action["camp"]]["observation"],
                action_mode=action_mode,
                decoder_receipt=action["decoder_receipt"],
                expected_decoder=expected_decoder,
                expected_camp=action["camp"],
            )
        if len(step["economic"]) > expected_bounds["max_economic_rows_per_step"]:
            raise ValueError("trajectory economic rows exceed the reviewed bound")
        successor_by_camp = {int(village["camp"]): village for village in successor["villages"]}
        for camp, source in source_by_camp.items():
            if camp not in successor_by_camp:
                raise ValueError("source camp disappears without an explicit successor witness")
            target = successor_by_camp[camp]
            if target["priorPopulation"] != source["population"] or target["society"] != source["society"]:
                raise ValueError("successor identity/prior population differs from its source boundary")
            if source["terminated"] and not target["terminated"]:
                raise ValueError("terminated source camp reappears as a live successor")
            if target["founded"]:
                raise ValueError("an existing camp is reflagged as newly founded")
        for camp, target in successor_by_camp.items():
            if camp in source_by_camp:
                continue
            if (not target["founded"] or target["priorPopulation"] != 0
                    or target["years"] != 0
                    or target["parentSociety"] not in {source_by_camp[item]["society"] for item in active}):
                raise ValueError("new successor identity lacks an explicit founding lineage witness")
        validated = validate_economic_rows(step["economic"], trajectory_id=reset["trajectory_id"],
                                           decision_turn=prior["turn"], source_villages=prior["villages"],
                                           successor_villages=successor["villages"])
        typed_witnesses = []
        migration_copies: dict[tuple[Any, ...], list[tuple[int, dict]]] = {}
        for row in validated:
            for transfer in row["transfers"]:
                if not isinstance(transfer, dict):
                    raise ValueError("economic transfer witness is malformed")
                kind = transfer.get("kind")
                if kind not in SUPPORTED_TRANSFER_KINDS:
                    raise ValueError(f"economic transfer kind {kind!r} is unsupported")
                if kind not in {"fission", "founding", "migration"}:
                    continue
                source_camp, destination_camp, people = (
                    transfer.get("sourceCamp"), transfer.get("destinationCamp"),
                    transfer.get("people"),
                )
                if (type(source_camp) is not int or type(destination_camp) is not int
                        or type(people) is not int or people <= 0
                        or source_camp == destination_camp
                        or transfer.get("phase") != "close-after-demography"):
                    raise ValueError("typed founding/fission/migration witness is malformed")
                if kind == "migration":
                    if row["campId"] not in {source_camp, destination_camp}:
                        raise ValueError("migration witness is not attached to one of its endpoints")
                    stable = tuple((name, transfer[name]) for name in
                                   ("migrationId", "eventId", "turn") if name in transfer)
                    event = (source_camp, destination_camp, people, transfer["phase"], stable)
                    migration_copies.setdefault(event, []).append((row["campId"], transfer))
                    continue
                if source_camp != row["campId"]:
                    raise ValueError("typed founding/fission witness is not attached to its source")
                typed_witnesses.append({"kind": kind, "source": source_camp,
                                        "destination": destination_camp, "people": people})
        migration_witnesses = []
        validated_camps = {row["campId"] for row in validated}
        for event, copies in migration_copies.items():
            source_camp, destination_camp, people, phase, _stable = event
            if (source_camp not in source_by_camp or destination_camp not in source_by_camp
                    or source_camp not in successor_by_camp
                    or destination_camp not in successor_by_camp
                    or source_camp not in validated_camps or destination_camp not in validated_camps
                    or sorted(camp for camp, _transfer in copies)
                    != sorted([source_camp, destination_camp])
                    or len(copies) != 2 or copies[0][1] != copies[1][1]):
                raise ValueError("migration witness lacks exact reciprocal endpoint evidence")
            migration_witnesses.append({"kind": "migration", "source": source_camp,
                                        "destination": destination_camp, "people": people,
                                        "phase": phase})
        new_camps = set(successor_by_camp) - set(source_by_camp)
        for camp in new_camps:
            target = successor_by_camp[camp]
            matches = [witness for witness in typed_witnesses if witness["destination"] == camp]
            if len(matches) != 1:
                raise ValueError("new successor identity lacks exactly one typed transfer witness")
            witness = matches[0]
            source = source_by_camp.get(witness["source"])
            if (source is None or source["society"] != target["parentSociety"]
                    or witness["people"] != target["population"]
                    or witness["kind"] not in {"fission", "founding"}):
                raise ValueError("typed founding/fission/migration witness conflicts with successor identity")
        if any(witness["destination"] not in new_camps for witness in typed_witnesses
               if witness["kind"] in {"fission", "founding"}):
            raise ValueError("typed founding/fission witness does not identify a new successor")
        for row in validated:
            witnessed = sum(witness["people"] for witness in typed_witnesses
                            if witness["source"] == row["campId"]
                            and witness["kind"] in {"fission", "founding"})
            split_off = row["domains"]["population"].get("splitOff", 0)
            if (type(split_off) is not int or split_off < 0 or split_off != witnessed):
                raise ValueError("typed founding/fission population differs from the raw split-off receipt")
        for row in validated:
            population = row["domains"]["population"]
            outgoing = sum(witness["people"] for witness in migration_witnesses
                           if witness["source"] == row["campId"])
            incoming = sum(witness["people"] for witness in migration_witnesses
                           if witness["destination"] == row["campId"])
            # These equations are unconditional.  In particular, a report may
            # not turn a non-zero migration receipt into "no migration" by
            # deleting or renaming both reciprocal endpoint witnesses.
            emigrated = _count(population.get("emigrated"), "population.emigrated")
            immigrated = _count(population.get("immigrated"), "population.immigrated")
            opening_residents = _count(
                population.get("openingResidents"), "population.openingResidents",
            )
            opening_parties = _count(
                population.get("openingParties"), "population.openingParties",
            )
            opening_linked = _count(
                population.get("openingLinked"), "population.openingLinked",
            )
            report_starting = _count(
                population.get("reportStartingLinked"), "population.reportStartingLinked",
            )
            linked = _count(
                population.get("reportEndingLinked"), "population.reportEndingLinked",
            )
            ending = _count(population.get("endingResidents"), "population.endingResidents")
            ending_parties = _count(
                population.get("endingParties"), "population.endingParties",
            )
            ending_linked = _count(
                population.get("endingLinked"), "population.endingLinked",
            )
            after_residents = _count(
                population.get("afterReintegrationResidents"),
                "population.afterReintegrationResidents",
            )
            after_parties = _count(
                population.get("afterReintegrationParties"),
                "population.afterReintegrationParties",
            )
            split_off = _count(population.get("splitOff"), "population.splitOff")
            expected_closing = linked - split_off - emigrated + immigrated
            if (emigrated != outgoing or immigrated != incoming
                    or opening_residents != source_by_camp[row["campId"]]["population"]
                    or opening_linked != opening_residents + opening_parties
                    or report_starting != opening_linked
                    or linked != after_residents + after_parties
                    or expected_closing < 0
                    or ending_linked != expected_closing
                    or ending_linked != ending + ending_parties):
                raise ValueError(
                    "migration/fission endpoint population receipts do not reconcile "
                    f"for camp {row['campId']}: opening={opening_residents}+{opening_parties}, "
                    f"source={source_by_camp[row['campId']]['population']}, "
                    f"ending={ending}+{ending_parties}, linked={ending_linked}/{linked}, "
                    f"split={split_off}, emigrated={emigrated}/{outgoing}, "
                    f"immigrated={immigrated}/{incoming}"
                )
        for row in validated:
            identity = row["transitionId"]
            camp_turn = (row["decisionTurn"], row["campId"])
            if identity in economic_ids or camp_turn in camp_turns:
                raise ValueError("trajectory repeats an economic transition/camp-turn identity")
            economic_ids.add(identity); camp_turns.add(camp_turn)
        economic_rows.extend(validated)
        terminal = step["terminal"]
        if (not isinstance(terminal, dict)
                or set(terminal) != {"terminated", "truncated", "reason"}
                or type(terminal["terminated"]) is not bool or type(terminal["truncated"]) is not bool
                or not isinstance(terminal["reason"], str) or not terminal["reason"]
                or terminal["terminated"] and terminal["truncated"]):
            raise ValueError("trajectory termination declaration is malformed")
        all_terminal = not any(not village["terminated"] for village in successor["villages"])
        if all_terminal:
            expected_terminal = {"terminated": True, "truncated": False,
                                 "reason": "all-villages-terminated"}
        elif index == horizon - 1:
            expected_terminal = {"terminated": False, "truncated": True,
                                 "reason": "evaluation-horizon"}
        else:
            expected_terminal = {"terminated": False, "truncated": False, "reason": "continuing"}
        if terminal != expected_terminal:
            raise ValueError("trajectory termination reason conflicts with state/horizon")
        if (terminal["terminated"] or terminal["truncated"]) and index != len(steps) - 1:
            raise ValueError("trajectory continues after its terminal/truncation boundary")
        contribution_rows = [village for village in successor["villages"]
                             if village["priorPopulation"] > 0]
        welfare += sum(float(village["welfare"]) for village in contribution_rows)
        rows_by_camp = {row["campId"]: row for row in validated}
        for village in contribution_rows:
            raw_trade = _finite(
                rows_by_camp[village["camp"]]["domains"]["trade"]["realisedUtility"],
                "trade.realisedUtility",
            )
            if float(village["tradeUtility"]) != raw_trade:
                raise ValueError("successor trade utility conflicts with its authoritative raw transition")
            trade_utility += raw_trade
        rng_receipt = step["policy_rng_receipt"]
        if action_mode == "seeded-stochastic":
            if rng_steps is None or index >= len(rng_steps):
                raise ValueError("trajectory stochastic step lacks its frozen schedule")
            expected_receipt = recompute_rng_receipt(rng_steps[index], len(actions))
            if rng_receipt != expected_receipt:
                raise ValueError("trajectory policy RNG receipt does not derive from actual action rows")
            receipts.append(expected_receipt)
        elif rng_receipt is not None:
            raise ValueError("deterministic trajectory carries stochastic RNG evidence")
        prior = successor
    last_terminal = steps[-1]["terminal"]
    if len(steps) < horizon and not last_terminal["terminated"]:
        raise ValueError("short trajectory lacks an evidenced terminal boundary")
    if len(steps) == horizon and not (last_terminal["terminated"] or last_terminal["truncated"]):
        raise ValueError("horizon-complete trajectory lacks its truncation/termination boundary")
    final = [village for village in prior["villages"]
             if not village["terminated"] and village["population"] > 0]
    integrated = integrate_economic_rows(economic_rows)
    services = service_outcomes(economic_rows, service_fields)
    compiler_total, compiler_summary = compiler_interventions(economic_rows)
    return {"welfare": welfare, "population_years": integrated["population_years"],
            "ending_residents": sum(village["population"] for village in final),
            "qol_years": integrated["qol_years"], "trade_utility": trade_utility,
            "extinctions": sum(1 for village in prior["villages"] if village["terminated"]),
            "settlements": len(final), "service_outcomes": services,
            "compiler_interventions": compiler_total, "compiler_intervention_summary": compiler_summary,
            "trajectory_digest": evidence["digest"], "policy_rng_receipts": receipts,
            "evaluation_steps": len(steps), "capture_receipt": receipt,
            "world_id": world_id, "snapshot_sha256": snapshot_sha256}


def generator_state_sha256(generator: torch.Generator) -> str:
    return digest({"bytes": bytes(generator.get_state().contiguous().view(torch.uint8).tolist()).hex()})


def validate_rng_schedule(schedule: dict, world_id: str, horizon: int, *, node_id: str,
                          expected_draw_order: list[str]) -> list[dict]:
    required = {"schema", "kind", "algorithm", "torch_version", "device", "worlds", "digest"}
    if (set(schedule) != required or schedule.get("schema") != 1
            or schedule.get("kind") != "gate1_policy_rng_schedule"
            or schedule.get("algorithm") != "torch.Generator.cpu-default"
            or schedule.get("torch_version") != torch.__version__ or schedule.get("device") != "cpu"
            or schedule.get("digest") != digest({k: v for k, v in schedule.items() if k != "digest"})):
        raise ValueError("policy RNG schedule is malformed or runtime-incompatible")
    matches = [row for row in schedule["worlds"] if row.get("world_id") == world_id]
    if len(matches) != 1 or set(matches[0]) != {"world_id", "nodes"}:
        raise ValueError("policy RNG schedule does not uniquely bind the world")
    node_matches = [row for row in matches[0]["nodes"] if row.get("node_id") == node_id]
    if len(node_matches) != 1 or set(node_matches[0]) != {"node_id", "draw_order", "steps"}:
        raise ValueError("policy RNG schedule does not uniquely bind the checkpoint node")
    execution = node_matches[0]
    if execution["draw_order"] != expected_draw_order or not execution["draw_order"]:
        raise ValueError("policy RNG schedule differs from the node's active policy heads")
    steps = execution["steps"]
    if not isinstance(steps, list) or len(steps) != horizon:
        raise ValueError("policy RNG schedule does not cover the exact horizon")
    output = []
    for index, row in enumerate(steps):
        if (not isinstance(row, dict) or set(row) != {"step", "seed", "state_sha256"}
                or row["step"] != index or isinstance(row["seed"], bool) or not isinstance(row["seed"], int)
                or row["seed"] < 0):
            raise ValueError("policy RNG schedule has a malformed step")
        generator = torch.Generator(device="cpu").manual_seed(row["seed"])
        if generator_state_sha256(generator) != row["state_sha256"]:
            raise ValueError("policy RNG schedule state does not match its seed/runtime")
        output.append({**row, "draw_order": list(execution["draw_order"])})
    return output


@contextmanager
def preserve_global_rng() -> Iterator[None]:
    """Make diagnostic inference observationally neutral to process RNG streams."""
    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    torch_state = torch.random.get_rng_state().clone()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
