"""Strict Gate 2 causal-accounting schema and completeness checks.

This module validates reporting evidence only.  It is deliberately independent of
policy, reward, and simulator execution code so imported traces and live captures
cross the same fail-closed boundary.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

SCHEMA_VERSION = 2
TOLERANCE = 1e-8

CRAFT_GOODS = (
    "HandTools",
    "Pots",
    "Cloth",
    "FarmTools",
    "Housing",
    "Tools",
    "AnimalFabric",
    "LightHousing",
    "HerdingTools",
    "HousingMaterial",
    "FishingTools",
)
MATERIALS = ("Stone", "Clay", "Wood")
TRADE_GOODS = (
    "Cloth",
    "AnimalFabric",
    "HousingMaterial",
    "Food",
    "LightHousing",
    "Pots",
    "HandTools",
    "FarmTools",
    "Stone",
    "Clay",
    "Wood",
)
TRADE_PARTS = {
    "Food": ("PlantFood", "AnimalFood", "AquaticFood"),
    "LightHousing": ("LightHousing",),
    **{name: (name,) for name in TRADE_GOODS if name not in {"Food", "LightHousing"}},
}
CONTROL_PHASES = (
    "nativeDemand",
    "actorCappedRequest",
    "compilerGrant",
    "executedWorkerOrInputClaim",
    "committedPhysicalReceipt",
    "inventoryChange",
    "consumptionOrFinalService",
    "endingPopulation",
    "endingQol",
)
CONSERVATION_TERMS = (
    "opening",
    "production",
    "tradeIn",
    "transferIn",
    "consumption",
    "tradeOut",
    "transferOut",
    "discard",
    "closing",
)
PHYSICAL_EVENT_CATEGORIES = {
    "opening-stock-boundary": ("boundary",),
    "ending-stock-boundary": ("boundary",),
    "production-receipt": ("in",),
    "material-extraction-receipt": ("in",),
    "intermediate-recipe-consumption": ("out",),
    "final-service-fulfilment": ("out",),
    "wear-depreciation": ("out",),
    "provisional-service-claim": ("claim",),
    "opening-stock-service-claim": ("claim",),
    "export-reservation": ("claim",),
    "storage-blocked-potential": ("potential",),
    "population-birth": ("in",),
    "population-death": ("out",),
    "food-production-receipt": ("in",),
    "human-food-consumption": ("out",),
    "manufacturing-food-consumption": ("out",),
    "trade-in": ("in",),
    "trade-out": ("out",),
    "transfer-in": ("in",),
    "transfer-out": ("out",),
    "discard-loss": ("out",),
    "field-capital-addition": ("in",),
    "field-capital-loss": ("out",),
    "herd-birth": ("in",),
    "herd-death-or-slaughter": ("out",),
}
CONTROL_UNITS = {
    "operation-fraction",
    "animal-fraction",
    "worker-headcount",
    "worker-years",
    "good-units",
    "material-units",
    "trade-good-units",
    "field-capital",
    "animals",
    "food",
    "utility",
    "people",
    "qol",
}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _rows(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{label} must be a list of objects")
    return value


def _number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if nonnegative and number < 0:
        raise ValueError(f"{label} must be nonnegative")
    return number


def _integer(value: Any, label: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _exact_keys(row: dict[str, Any], expected: set[str], label: str) -> None:
    if set(row) != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - set(row))}, "
            f"extra={sorted(set(row) - expected)}"
        )


def _unique(rows: list[dict[str, Any]], field: str, label: str) -> None:
    identities = [_text(row.get(field), f"{label}.{field}") for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate {label} identity")


def _validate_controls(sidecar: dict[str, Any]) -> None:
    controls = _rows(sidecar.get("controls"), "controls")
    _unique(controls, "controlId", "control")
    for index, control in enumerate(controls):
        label = f"controls[{index}]"
        _exact_keys(
            control,
            {
                "controlId",
                "domain",
                "identity",
                "nativeUnit",
                "actorHead",
                "phases",
                "phaseUnits",
                "unavailable",
                "metadata",
                "classification",
                "classificationReason",
            },
            label,
        )
        for name in ("domain", "identity", "nativeUnit", "actorHead"):
            _text(control[name], f"{label}.{name}")
        if control["nativeUnit"] not in CONTROL_UNITS:
            raise ValueError(f"{label}.nativeUnit is not registered")
        classification = control["classification"]
        if classification not in {"active", "inactive", "unsupported"}:
            raise ValueError(f"{label}.classification is invalid")
        if classification == "active":
            if control["classificationReason"] is not None:
                raise ValueError(f"{label} active control has a classification reason")
        else:
            _text(control["classificationReason"], f"{label}.classificationReason")
        phases = _object(control["phases"], f"{label}.phases")
        units = _object(control["phaseUnits"], f"{label}.phaseUnits")
        unavailable = _object(control["unavailable"], f"{label}.unavailable")
        _object(control["metadata"], f"{label}.metadata")
        if set(phases) != set(CONTROL_PHASES) or set(units) != set(CONTROL_PHASES):
            raise ValueError(f"{label} phase order/schema differs")
        for phase in CONTROL_PHASES:
            unit = _text(units[phase], f"{label}.phaseUnits.{phase}")
            if unit not in CONTROL_UNITS:
                raise ValueError(f"{label}.phaseUnits.{phase} is not registered")
            value = phases[phase]
            if value is None:
                _text(unavailable.get(phase), f"{label}.unavailable.{phase}")
            else:
                _number(
                    value,
                    f"{label}.phases.{phase}",
                    nonnegative=phase != "inventoryChange" and unit != "utility",
                )
                if phase in unavailable:
                    raise ValueError(f"{label}.{phase} is both available and unavailable")
        if set(unavailable) != {name for name, value in phases.items() if value is None}:
            raise ValueError(f"{label} unavailable map does not exactly cover null phases")
        if units["endingPopulation"] != "people" or units["endingQol"] != "qol":
            raise ValueError(f"{label} outcome phase units differ")
        if any(units[name] != control["nativeUnit"] for name in CONTROL_PHASES[:3]):
            raise ValueError(f"{label} native/request/grant units differ")


def _validate_physical_rows(sidecar: dict[str, Any]) -> None:
    goods = _rows(sidecar.get("goods"), "goods")
    if tuple(row.get("good") for row in goods) != CRAFT_GOODS:
        raise ValueError("goods must exactly cover the stable CraftGood order")
    good_numeric = {
        "output",
        "openingStock",
        "endingStock",
        "afterOpeningTradeStock",
        "beforeFinalTradeStock",
        "afterFinalTradeStock",
        "intermediateConsumption",
        "serviceConsumptionOrDepreciation",
        "actualTradeIn",
        "actualTradeOut",
        "transferIn",
        "transferOut",
        "workerIntervals",
        "workerYears",
        "equipmentCoveredWorkerYears",
        "stone",
        "clay",
        "wood",
        "plantFood",
        "animalFood",
        "animalFabric",
        "cloth",
        "housingMaterial",
        "handToolsConsumed",
        "toolsConsumed",
        "discardOrDestruction",
    }
    for index, row in enumerate(goods):
        _exact_keys(
            row,
            {"good", "unit", "stockKind", "plannedOutputCeiling", *good_numeric},
            f"goods[{index}]",
        )
        if row.get("unit") != "good-units":
            raise ValueError(f"goods[{index}] unit differs")
        if row.get("stockKind") not in {
            "camp-inventory",
            "fixed-housing-tranche-capacity",
        }:
            raise ValueError(f"goods[{index}] stock kind differs")
        for name in good_numeric:
            _number(row.get(name), f"goods[{index}].{name}", nonnegative=True)
        if row["plannedOutputCeiling"] is not None:
            _number(
                row["plannedOutputCeiling"],
                f"goods[{index}].plannedOutputCeiling",
                nonnegative=True,
            )
        else:
            _text(
                sidecar.get("unavailable", {}).get(f"goods[{index}].plannedOutputCeiling"),
                f"goods[{index}].plannedOutputCeiling unavailable reason",
            )

    materials = _rows(sidecar.get("materials"), "materials")
    if tuple(row.get("material") for row in materials) != MATERIALS:
        raise ValueError("materials must exactly cover Stone/Clay/Wood in stable order")
    for index, row in enumerate(materials):
        _exact_keys(
            row,
            {
                "material",
                "unit",
                "received",
                "openingStock",
                "endingStock",
                "recipeConsumption",
                "actualTradeIn",
                "actualTradeOut",
                "transferIn",
                "transferOut",
                "workerIntervals",
                "workerYears",
                "requested",
                "granted",
                "phaseBRequested",
                "phaseBGranted",
                "phaseBReceived",
            },
            f"materials[{index}]",
        )
        if row.get("unit") != "material-units":
            raise ValueError(f"materials[{index}] unit differs")
        for name in (
            "received",
            "openingStock",
            "endingStock",
            "recipeConsumption",
            "actualTradeIn",
            "actualTradeOut",
            "transferIn",
            "transferOut",
            "workerIntervals",
            "workerYears",
        ):
            _number(row.get(name), f"materials[{index}].{name}", nonnegative=True)
        for name in ("requested", "granted", "phaseBRequested", "phaseBGranted", "phaseBReceived"):
            if row[name] is not None:
                _number(row[name], f"materials[{index}].{name}", nonnegative=True)
            else:
                _text(
                    sidecar.get("unavailable", {}).get(f"materials[{index}].{name}"),
                    f"materials[{index}].{name} unavailable reason",
                )

    fields = _rows(sidecar.get("fields"), "fields")
    field_ids = [_integer(row.get("fieldId"), "field.fieldId", nonnegative=True) for row in fields]
    if len(field_ids) != len(set(field_ids)):
        raise ValueError("duplicate field identity")
    for index, row in enumerate(fields):
        _exact_keys(
            row,
            {
                "fieldId",
                "openingCapital",
                "meanCapital",
                "decayedCapital",
                "capitalAdded",
                "capitalLost",
                "transferIn",
                "transferOut",
                "endingCapital",
                "residual",
            },
            f"fields[{index}]",
        )
        _integer(row["fieldId"], f"fields[{index}].fieldId", nonnegative=True)
        values = {
            name: _number(row[name], f"fields[{index}].{name}", nonnegative=True)
            for name in (
                "openingCapital",
                "meanCapital",
                "decayedCapital",
                "capitalAdded",
                "capitalLost",
                "transferIn",
                "transferOut",
                "endingCapital",
            )
        }
        residual = _number(row["residual"], f"fields[{index}].residual")
        if not math.isclose(
            residual,
            values["openingCapital"]
            + values["capitalAdded"]
            - values["capitalLost"]
            + values["transferIn"]
            - values["transferOut"]
            - values["endingCapital"],
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ):
            raise ValueError(f"fields[{index}] residual is not derived")

    herds = _rows(sidecar.get("herds"), "herds")
    herd_ids = [_integer(row.get("herdId"), "herd.herdId", nonnegative=True) for row in herds]
    if len(herd_ids) != len(set(herd_ids)):
        raise ValueError("duplicate herd identity")
    herd_fields = {
        "herdId",
        "openingAnimals",
        "startingAnimals",
        "survivingAnimals",
        "newbornAnimals",
        "starvationDeaths",
        "slaughtered",
        "foodNutrition",
        "waterNutrition",
        "nutrition",
        "managementFulfilment",
        "potentialFood",
        "acceptedFood",
        "slaughterFood",
        "coveredHerderYears",
        "herdingToolsConsumed",
        "ownershipInflow",
        "ownershipOutflow",
        "committedEndingAnimals",
        "reportAvailable",
        "unavailableReason",
        "residual",
    }
    for index, row in enumerate(herds):
        _exact_keys(row, herd_fields, f"herds[{index}]")
        _integer(row["herdId"], f"herds[{index}].herdId", nonnegative=True)
        if not isinstance(row["reportAvailable"], bool):
            raise ValueError(f"herds[{index}].reportAvailable must be boolean")
        if row["reportAvailable"]:
            if row["unavailableReason"] is not None:
                raise ValueError(f"herds[{index}] available report has unavailable reason")
        else:
            _text(row["unavailableReason"], f"herds[{index}].unavailableReason")
        nullable_report = {"foodNutrition", "waterNutrition", "nutrition", "managementFulfilment"}
        for name in herd_fields - {"herdId", "reportAvailable", "unavailableReason", "residual"}:
            if name in nullable_report and row[name] is None:
                if row["reportAvailable"]:
                    raise ValueError(f"herds[{index}].{name} missing from available report")
                continue
            _number(row[name], f"herds[{index}].{name}", nonnegative=True)
        residual = _number(row["residual"], f"herds[{index}].residual")
        expected_residual = (
            float(row["openingAnimals"])
            + float(row["newbornAnimals"])
            + float(row["ownershipInflow"])
            - float(row["starvationDeaths"])
            - float(row["slaughtered"])
            - float(row["ownershipOutflow"])
            - float(row["committedEndingAnimals"])
        )
        if not math.isclose(residual, expected_residual, rel_tol=0.0, abs_tol=TOLERANCE):
            raise ValueError(f"herds[{index}] residual is not derived")

    events = _rows(sidecar.get("physicalEvents"), "physicalEvents")
    _unique(events, "eventId", "physical event")
    allowed_items = {
        *CRAFT_GOODS,
        *MATERIALS,
        "PlantFood",
        "AnimalFood",
        "AquaticFood",
        "linked-population",
        "linked-food-reserves",
    }
    authoritative_request_ids = {
        _text(request.get("requestId"), "request.requestId")
        for request in _rows(sidecar.get("requests"), "requests")
    }
    for index, event in enumerate(events):
        label = f"physicalEvents[{index}]"
        required = {
            "eventId",
            "campId",
            "category",
            "item",
            "unit",
            "quantity",
            "direction",
        }
        optional = {"recipe", "receiptId", "phase", "sourceRequestId"}
        if not required <= set(event) or set(event) - required - optional:
            raise ValueError(f"{label} fields differ")
        if _integer(event["campId"], f"{label}.campId", nonnegative=True) != sidecar.get("campId"):
            raise ValueError(f"{label} belongs to another camp")
        if not str(event["eventId"]).startswith(f"{sidecar.get('trajectoryId')}/"):
            raise ValueError(f"{label} is not bound to the trajectory")
        category = _text(event["category"], f"{label}.category")
        if category not in PHYSICAL_EVENT_CATEGORIES:
            raise ValueError(f"{label} category is not registered")
        if event["direction"] not in PHYSICAL_EVENT_CATEGORIES[category]:
            raise ValueError(f"{label} direction differs from category")
        item = _text(event["item"], f"{label}.item")
        if category == "final-service-fulfilment":
            if event.get("phase") not in {
                "work-before-production",
                "needs-after-production",
            }:
                raise ValueError(f"{label} final service phase is not registered")
        elif category == "human-food-consumption" and item in {
            "PlantFood",
            "AnimalFood",
            "AquaticFood",
        }:
            if event.get("phase") != "needs-after-production":
                raise ValueError(f"{label} typed food-consumption phase differs")
        elif "phase" in event:
            raise ValueError(f"{label} phase is not valid for category")
        if "sourceRequestId" in event:
            if category not in {"production-receipt", "intermediate-recipe-consumption"}:
                raise ValueError(f"{label} source request is not valid for category")
            if "receiptId" not in event:
                raise ValueError(f"{label} request-bound mutation lacks receipt identity")
            if event["sourceRequestId"] not in authoritative_request_ids:
                raise ValueError(f"{label} source request is not authoritative")
        if item not in allowed_items and not item.startswith(("field:", "herd:")):
            raise ValueError(f"{label} item is not registered")
        expected_unit = (
            "material-units"
            if item in MATERIALS
            else "food"
            if item in {"PlantFood", "AnimalFood", "AquaticFood", "linked-food-reserves"}
            else "people"
            if item == "linked-population"
            else "field-capital"
            if item.startswith("field:")
            else "animals"
            if item.startswith("herd:")
            else "good-units"
        )
        if event["unit"] != expected_unit:
            raise ValueError(f"{label} unit differs from item registry")
        _number(event["quantity"], f"{label}.quantity", nonnegative=True)
        for optional_name in optional & set(event):
            _text(event[optional_name], f"{label}.{optional_name}")

    def total(category: str, item: str) -> float:
        return sum(
            float(event["quantity"])
            for event in events
            if event["category"] == category and event["item"] == item
        )

    for row in goods:
        name = str(row["good"])
        expectations = {
            "opening-stock-boundary": float(row["openingStock"]),
            "ending-stock-boundary": float(row["endingStock"]),
            "production-receipt": float(row["output"]),
            "intermediate-recipe-consumption": float(row["intermediateConsumption"]),
        }
        service = total("final-service-fulfilment", name) + total("wear-depreciation", name)
        if not math.isclose(
            service,
            float(row["serviceConsumptionOrDepreciation"]),
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ):
            raise ValueError(f"physical service events do not reconcile for {name}")
        for category, expected in expectations.items():
            matches = [
                event for event in events if event["category"] == category and event["item"] == name
            ]
            if category.endswith("boundary") and len(matches) != 1:
                raise ValueError(f"physical {category} must exactly cover {name}")
            if not math.isclose(
                sum(float(event["quantity"]) for event in matches),
                expected,
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ):
                raise ValueError(f"physical {category} does not reconcile for {name}")

    for row in materials:
        name = str(row["material"])
        for category, expected in (
            ("opening-stock-boundary", float(row["openingStock"])),
            ("ending-stock-boundary", float(row["endingStock"])),
            ("material-extraction-receipt", float(row["received"])),
            ("intermediate-recipe-consumption", float(row["recipeConsumption"])),
        ):
            matches = [
                event for event in events if event["category"] == category and event["item"] == name
            ]
            if category.endswith("boundary") and len(matches) != 1:
                raise ValueError(f"physical {category} must exactly cover {name}")
            if not math.isclose(
                sum(float(event["quantity"]) for event in matches),
                expected,
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ):
                raise ValueError(f"physical {category} does not reconcile for {name}")

    def require_total(category: str, item: str, expected: float, *, boundary: bool = False) -> None:
        matches = [
            event for event in events if event["category"] == category and event["item"] == item
        ]
        if boundary and len(matches) != 1:
            raise ValueError(f"physical {category} must exactly cover {item}")
        if not math.isclose(
            sum(float(event["quantity"]) for event in matches),
            float(expected),
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ):
            raise ValueError(f"physical {category} does not reconcile for {item}")

    ledgers = {
        (row["domain"], row["item"]): row
        for row in _rows(sidecar.get("conservationRows"), "conservationRows")
    }
    population = ledgers.get(("population", "linked-population"))
    food = ledgers.get(("food", "linked-food-reserves"))
    if population is not None:
        require_total(
            "opening-stock-boundary", "linked-population", population["opening"], boundary=True
        )
        require_total(
            "ending-stock-boundary", "linked-population", population["closing"], boundary=True
        )
        require_total("population-birth", "linked-population", population["production"])
        require_total("population-death", "linked-population", population["consumption"])
        require_total("transfer-in", "linked-population", population["transferIn"])
        require_total("transfer-out", "linked-population", population["transferOut"])
    if food is not None:
        require_total(
            "opening-stock-boundary", "linked-food-reserves", food["opening"], boundary=True
        )
        require_total(
            "ending-stock-boundary", "linked-food-reserves", food["closing"], boundary=True
        )
        require_total("food-production-receipt", "linked-food-reserves", food["production"])
        require_total(
            "human-food-consumption",
            "linked-food-reserves",
            float(sidecar["domains"]["population"]["humanFoodConsumption"]),
        )
        require_total(
            "manufacturing-food-consumption",
            "linked-food-reserves",
            float(sidecar["domains"]["population"]["manufacturingFood"]),
        )
        require_total("trade-in", "linked-food-reserves", food["tradeIn"])
        require_total("trade-out", "linked-food-reserves", food["tradeOut"])
        require_total("transfer-in", "linked-food-reserves", food["transferIn"])
        require_total("transfer-out", "linked-food-reserves", food["transferOut"])
        require_total("discard-loss", "linked-food-reserves", food["discard"])
    for row in goods:
        name = str(row["good"])
        require_total("trade-in", name, row["actualTradeIn"])
        require_total("trade-out", name, row["actualTradeOut"])
        require_total("transfer-in", name, row["transferIn"])
        require_total("transfer-out", name, row["transferOut"])
        require_total("discard-loss", name, row["discardOrDestruction"])
    for row in materials:
        name = str(row["material"])
        require_total("trade-in", name, row["actualTradeIn"])
        require_total("trade-out", name, row["actualTradeOut"])
        require_total("transfer-in", name, row["transferIn"])
        require_total("transfer-out", name, row["transferOut"])
    for row in fields:
        name = f"field:{row['fieldId']}"
        require_total("opening-stock-boundary", name, row["openingCapital"], boundary=True)
        require_total("ending-stock-boundary", name, row["endingCapital"], boundary=True)
        require_total("field-capital-addition", name, row["capitalAdded"])
        require_total("field-capital-loss", name, row["capitalLost"])
        require_total("transfer-in", name, row["transferIn"])
        require_total("transfer-out", name, row["transferOut"])
    for row in herds:
        name = f"herd:{row['herdId']}"
        require_total("opening-stock-boundary", name, row["openingAnimals"], boundary=True)
        require_total("ending-stock-boundary", name, row["committedEndingAnimals"], boundary=True)
        require_total("herd-birth", name, row["newbornAnimals"])
        require_total(
            "herd-death-or-slaughter",
            name,
            float(row["starvationDeaths"]) + float(row["slaughtered"]),
        )
        require_total("transfer-in", name, row["ownershipInflow"])
        require_total("transfer-out", name, row["ownershipOutflow"])


def _validate_conservation(sidecar: dict[str, Any]) -> None:
    rows = _rows(sidecar.get("conservationRows"), "conservationRows")
    identities: list[tuple[str, str]] = []
    expected_keys = {
        "domain",
        "item",
        "unit",
        *CONSERVATION_TERMS,
        "residual",
        "complete",
        "unavailableReason",
    }
    for index, row in enumerate(rows):
        label = f"conservationRows[{index}]"
        _exact_keys(row, expected_keys, label)
        identity = (_text(row["domain"], f"{label}.domain"), _text(row["item"], f"{label}.item"))
        identities.append(identity)
        unit = _text(row["unit"], f"{label}.unit")
        expected_unit = {
            "population": "people",
            "food": "food",
            "material": "material-units",
            "craft-good": "good-units",
            "field": "field-capital",
            "herd": "animals",
        }.get(identity[0])
        if expected_unit is None or unit != expected_unit:
            raise ValueError(f"{label}.unit differs from the domain registry")
        if not isinstance(row["complete"], bool):
            raise ValueError(f"{label}.complete must be boolean")
        if row["complete"]:
            if row["unavailableReason"] is not None:
                raise ValueError(f"{label} complete row has unavailable reason")
            values = {
                name: _number(row[name], f"{label}.{name}", nonnegative=True)
                for name in CONSERVATION_TERMS
            }
            residual = _number(row["residual"], f"{label}.residual")
            recomputed = (
                values["opening"]
                + values["production"]
                + values["tradeIn"]
                + values["transferIn"]
                - values["consumption"]
                - values["tradeOut"]
                - values["transferOut"]
                - values["discard"]
                - values["closing"]
            )
            if not math.isclose(residual, recomputed, rel_tol=0.0, abs_tol=TOLERANCE):
                raise ValueError(f"{label} residual is not derived from its terms")
        else:
            _text(row["unavailableReason"], f"{label}.unavailableReason")
            if row["residual"] is not None:
                raise ValueError(f"{label} incomplete row must not assert a residual")
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate conservation identity")

    domains = _object(sidecar.get("domains"), "domains")
    cultivation = _object(domains.get("cultivation"), "domains.cultivation")
    herding = _object(domains.get("herding"), "domains.herding")

    def identity_list(value: Any, label: str) -> set[int]:
        if not isinstance(value, list):
            raise ValueError(f"{label} must be a list")
        result = {_integer(item, label, nonnegative=True) for item in value}
        if len(result) != len(value):
            raise ValueError(f"{label} contains duplicate identities")
        return result

    opening_fields = identity_list(
        cultivation.get("openingFieldIds"), "cultivation.openingFieldIds"
    )
    closing_fields = identity_list(
        cultivation.get("closingFieldIds"), "cultivation.closingFieldIds"
    )
    authoritative_fields = identity_list(
        cultivation.get("accountedFieldIds"), "cultivation.accountedFieldIds"
    )
    if not (opening_fields | closing_fields) <= authoritative_fields:
        raise ValueError("accounted field identities omit an opening/closing identity")
    authoritative_herds = identity_list(
        herding.get("openingHerdIds"), "herding.openingHerdIds"
    ) | identity_list(herding.get("closingHerdIds"), "herding.closingHerdIds")
    emitted_fields = {int(row["fieldId"]) for row in sidecar["fields"]}
    emitted_herds = {int(row["herdId"]) for row in sidecar["herds"]}
    if emitted_fields != authoritative_fields:
        raise ValueError("field rows do not exactly cover opening/closing identities")
    if emitted_herds != authoritative_herds:
        raise ValueError("herd rows do not exactly cover opening/closing identities")
    required = {
        ("population", "linked-population"),
        ("food", "linked-food-reserves"),
        *(("material", name) for name in MATERIALS),
        *(("craft-good", name) for name in CRAFT_GOODS),
        *(("field", str(identity)) for identity in authoritative_fields),
        *(("herd", str(identity)) for identity in authoritative_herds),
    }
    missing = sorted(required - set(identities))
    if missing:
        raise ValueError(f"mandatory conservation identities missing: {missing}")

    by_identity = {(row["domain"], row["item"]): row for row in rows}
    for good in sidecar["goods"]:
        name = str(good["good"])
        ledger = by_identity[("craft-good", name)]
        expected = {
            "opening": good["openingStock"],
            "production": good["output"],
            "tradeIn": good["actualTradeIn"],
            "transferIn": good["transferIn"],
            "consumption": good["intermediateConsumption"]
            + good["serviceConsumptionOrDepreciation"],
            "tradeOut": good["actualTradeOut"],
            "transferOut": good["transferOut"],
            "discard": good["discardOrDestruction"],
            "closing": good["endingStock"],
        }
        if any(
            not math.isclose(float(ledger[field]), float(value), rel_tol=0.0, abs_tol=TOLERANCE)
            for field, value in expected.items()
        ):
            raise ValueError(f"craft-good conservation terms differ from physical rows for {name}")
    for material in sidecar["materials"]:
        name = str(material["material"])
        ledger = by_identity[("material", name)]
        expected = {
            "opening": material["openingStock"],
            "production": material["received"],
            "tradeIn": material["actualTradeIn"],
            "transferIn": material["transferIn"],
            "consumption": material["recipeConsumption"],
            "tradeOut": material["actualTradeOut"],
            "transferOut": material["transferOut"],
            "discard": 0.0,
            "closing": material["endingStock"],
        }
        if any(
            not math.isclose(float(ledger[field]), float(value), rel_tol=0.0, abs_tol=TOLERANCE)
            for field, value in expected.items()
        ):
            raise ValueError(f"material conservation terms differ from physical rows for {name}")
    for field in sidecar["fields"]:
        name = str(field["fieldId"])
        ledger = by_identity[("field", name)]
        expected = {
            "opening": field["openingCapital"],
            "production": field["capitalAdded"],
            "transferIn": field["transferIn"],
            "consumption": field["capitalLost"],
            "transferOut": field["transferOut"],
            "closing": field["endingCapital"],
        }
        if any(
            not math.isclose(float(ledger[term]), float(value), rel_tol=0.0, abs_tol=TOLERANCE)
            for term, value in expected.items()
        ):
            raise ValueError(f"field conservation terms differ from physical row for {name}")
    for herd in sidecar["herds"]:
        name = str(herd["herdId"])
        ledger = by_identity[("herd", name)]
        expected = {
            "opening": herd["openingAnimals"],
            "production": herd["newbornAnimals"],
            "transferIn": herd["ownershipInflow"],
            "consumption": herd["starvationDeaths"] + herd["slaughtered"],
            "transferOut": herd["ownershipOutflow"],
            "closing": herd["committedEndingAnimals"],
        }
        if any(
            not math.isclose(float(ledger[term]), float(value), rel_tol=0.0, abs_tol=TOLERANCE)
            for term, value in expected.items()
        ):
            raise ValueError(f"herd conservation terms differ from physical row for {name}")

    population_domain = _object(domains.get("population"), "domains.population")
    population_ledger = by_identity[("population", "linked-population")]
    aggregate_conservation = _object(sidecar.get("conservation"), "conservation")
    reported_gross_residual = _number(
        aggregate_conservation.get("grossPopulationResidual"),
        "conservation.grossPopulationResidual",
    )
    independently_derived_gross_residual = (
        _number(
            population_domain.get("reportEndingLinked"),
            "population.reportEndingLinked",
            nonnegative=True,
        )
        - _number(
            population_domain.get("reportStartingLinked"),
            "population.reportStartingLinked",
            nonnegative=True,
        )
        - _number(population_domain.get("grossBirths"), "population.grossBirths", nonnegative=True)
        + _number(population_domain.get("grossDeaths"), "population.grossDeaths", nonnegative=True)
    )
    if not math.isclose(
        reported_gross_residual,
        independently_derived_gross_residual,
        rel_tol=0.0,
        abs_tol=TOLERANCE,
    ):
        raise ValueError("gross population residual is not independently derived")
    population_expected = {
        "opening": population_domain.get("openingLinked"),
        "production": population_domain.get("grossBirths"),
        "tradeIn": 0.0,
        "transferIn": population_domain.get("immigrated"),
        "consumption": population_domain.get("grossDeaths"),
        "tradeOut": 0.0,
        "transferOut": _number(
            population_domain.get("emigrated"), "population.emigrated", nonnegative=True
        )
        + _number(population_domain.get("splitOff"), "population.splitOff", nonnegative=True),
        "discard": 0.0,
        "closing": population_domain.get("endingLinked"),
    }
    if any(
        not math.isclose(
            float(population_ledger[term]),
            _number(value, f"population.{term}", nonnegative=True),
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        )
        for term, value in population_expected.items()
    ):
        raise ValueError("population conservation is not cross-bound to demographic witnesses")

    food_ledger = by_identity[("food", "linked-food-reserves")]
    food_expected = {
        "opening": population_domain.get("decisionOpeningFood"),
        "production": population_domain.get("foodProductionReceipt"),
        "tradeIn": population_domain.get("foodTradeIn"),
        "transferIn": population_domain.get("foodTransferIn"),
        "consumption": _number(
            population_domain.get("humanFoodConsumption"),
            "population.humanFoodConsumption",
            nonnegative=True,
        )
        + _number(
            population_domain.get("manufacturingFood"),
            "population.manufacturingFood",
            nonnegative=True,
        ),
        "tradeOut": population_domain.get("foodTradeOut"),
        "transferOut": population_domain.get("foodTransferOut"),
        "discard": population_domain.get("foodDiscard"),
        "closing": population_domain.get("committedEndingFood"),
    }
    if any(
        not math.isclose(
            float(food_ledger[term]),
            _number(value, f"population.{term}", nonnegative=True),
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        )
        for term, value in food_expected.items()
    ):
        raise ValueError("food conservation is not cross-bound to independent mutation witnesses")


def _validate_trade(sidecar: dict[str, Any]) -> None:
    rows = _rows(sidecar.get("trades"), "trades")
    kinds = {"eligibility", "requisition", "transaction"}
    if any(row.get("recordType") not in kinds for row in rows):
        raise ValueError("unknown trade lifecycle record type")
    eligibility = [row for row in rows if row["recordType"] == "eligibility"]
    if sum(row.get("campId") == sidecar.get("campId") for row in eligibility) != 1:
        raise ValueError("trade lifecycle must contain the source camp eligibility row")
    if len({row.get("campId") for row in eligibility}) != len(eligibility):
        raise ValueError("duplicate trade eligibility endpoint")
    for row in eligibility:
        expected = {
            "recordType",
            "eventId",
            "campId",
            "societyId",
            "eligible",
            "gatingReason",
            "partnerSearchCount",
        }
        if set(row) not in (expected, expected | {"partnerSearchReason"}):
            raise ValueError("trade eligibility fields differ")
        event_id = _text(row.get("eventId"), "trade eligibility eventId")
        if not event_id.startswith(f"{sidecar.get('trajectoryId')}/"):
            raise ValueError("trade eligibility is not bound to the trajectory")
        _integer(row.get("campId"), "trade eligibility campId", nonnegative=True)
        _integer(row.get("societyId"), "trade eligibility societyId", nonnegative=True)
        if not isinstance(row.get("eligible"), bool):
            raise ValueError("trade eligibility flag must be boolean")
        _integer(row.get("partnerSearchCount"), "trade partnerSearchCount", nonnegative=True)
        reason = row.get("gatingReason")
        if row["eligible"] and reason is not None:
            raise ValueError("eligible camp cannot have a gating reason")
        if not row["eligible"]:
            _text(reason, "trade gatingReason")

    requests = [row for row in rows if row["recordType"] == "requisition"]
    _unique(requests, "requestId", "trade requisition")
    request_by_id = {row["requestId"]: row for row in requests}
    for row in requests:
        _exact_keys(
            row,
            {
                "recordType",
                "requestId",
                "requester",
                "responder",
                "distance",
                "affinity",
                "wantedGoods",
                "offeredGoods",
                "outcome",
                "rejectionReason",
            },
            "trade requisition",
        )
        if not str(row["requestId"]).startswith(f"{sidecar.get('trajectoryId')}/"):
            raise ValueError("trade requisition is not bound to the trajectory")
        for name in ("requester", "responder", "distance"):
            _integer(row.get(name), f"trade requisition {name}", nonnegative=True)
        _number(row.get("affinity"), "trade requisition affinity", nonnegative=True)
        for name in ("wantedGoods", "offeredGoods"):
            values = row.get(name)
            if (
                not isinstance(values, list)
                or not values
                or any(value not in TRADE_GOODS for value in values)
            ):
                raise ValueError(f"trade requisition {name} differs from stable goods")
        if row.get("outcome") not in {"honoured", "countered", "ignored"}:
            raise ValueError("trade requisition outcome invalid")
        if row["outcome"] == "ignored":
            _text(row.get("rejectionReason"), "trade rejectionReason")
        elif row.get("rejectionReason") is not None:
            raise ValueError("accepted trade requisition has rejection reason")

    transactions = [row for row in rows if row["recordType"] == "transaction"]
    _unique(transactions, "transactionId", "trade transaction")
    used_requests: list[str] = []
    physical_in: Counter[str] = Counter()
    physical_out: Counter[str] = Counter()
    for row in transactions:
        _exact_keys(
            row,
            {
                "recordType",
                "requestId",
                "transactionId",
                "requester",
                "responder",
                "distance",
                "handshake",
                "ratio",
                "requesterPlannedGain",
                "responderPlannedGain",
                "settlementState",
                "deliveryFraction",
                "requesterImmediateUtility",
                "responderImmediateUtility",
                "requesterImmediateUtilityUnavailableReason",
                "responderImmediateUtilityUnavailableReason",
                "delayedUtility",
                "delayedUtilityUnavailableReason",
                "legs",
            },
            "trade transaction",
        )
        request_id = _text(row.get("requestId"), "trade transaction requestId")
        if request_id not in request_by_id:
            raise ValueError("trade transaction is not joined to its requisition")
        used_requests.append(request_id)
        if not str(row.get("transactionId", "")).startswith(f"{sidecar.get('trajectoryId')}/"):
            raise ValueError("trade transaction is not bound to the trajectory")
        request = request_by_id[request_id]
        requester = _integer(row.get("requester"), "trade transaction requester", nonnegative=True)
        responder = _integer(row.get("responder"), "trade transaction responder", nonnegative=True)
        if requester == responder or (requester, responder) != (
            request["requester"],
            request["responder"],
        ):
            raise ValueError("trade transaction endpoints differ from requisition")
        if row.get("handshake") != request["outcome"]:
            raise ValueError("trade handshake differs from requisition outcome")
        if row.get("settlementState") not in {
            "delivered",
            "partial",
            "undelivered",
            "party-vanished",
        }:
            raise ValueError("trade settlement state invalid")
        for name in ("distance",):
            _integer(row.get(name), f"trade transaction {name}", nonnegative=True)
        for name in ("ratio", "requesterPlannedGain", "responderPlannedGain", "deliveryFraction"):
            _number(row.get(name), f"trade transaction {name}")
        for name in ("requester", "responder"):
            utility_name = f"{name}ImmediateUtility"
            reason_name = f"{utility_name}UnavailableReason"
            if row[utility_name] is None:
                _text(row.get(reason_name), f"trade transaction {reason_name}")
            else:
                _number(row[utility_name], f"trade transaction {utility_name}")
                if row.get(reason_name) is not None:
                    raise ValueError("available immediate utility has an unavailable reason")
        if row.get("delayedUtility") is not None:
            raise ValueError("current-transition delayed trade utility must be unavailable")
        _text(row.get("delayedUtilityUnavailableReason"), "trade delayed utility reason")
        legs = _rows(row.get("legs"), "trade transaction legs")
        if len(legs) != 2:
            raise ValueError("bilateral trade must contain exactly two legs")
        directions = Counter()
        leg_ids = set()
        for leg in legs:
            _exact_keys(
                leg,
                {
                    "legId",
                    "sender",
                    "receiver",
                    "good",
                    "unit",
                    "requestedQuantity",
                    "agreedQuantity",
                    "stockQuantity",
                    "madeToOrderQuantity",
                    "escrowedQuantity",
                    "producedMadeToOrderQuantity",
                    "supplyAvailableQuantity",
                    "bilateralDeliverableQuantity",
                    "deliveredQuantity",
                    "acceptedQuantity",
                    "senderDebitQuantity",
                    "receiverCreditQuantity",
                    "deliveryFraction",
                    "producerDeliveryFraction",
                    "storageBlockedQuantity",
                    "counterpartyShortfallQuantity",
                    "refundedOrReturnedQuantity",
                    "destroyedQuantity",
                    "deliveryPhase",
                    "state",
                    "parts",
                },
                "trade leg",
            )
            leg_id = _text(leg.get("legId"), "trade legId")
            if not leg_id.startswith(f"{row['transactionId']}/"):
                raise ValueError("trade leg is not bound to its transaction")
            if leg_id in leg_ids:
                raise ValueError("duplicate trade leg identity")
            leg_ids.add(leg_id)
            sender = _integer(leg.get("sender"), "trade leg sender", nonnegative=True)
            receiver = _integer(leg.get("receiver"), "trade leg receiver", nonnegative=True)
            if sender == receiver or {sender, receiver} != {requester, responder}:
                raise ValueError("trade leg endpoints are not reciprocal transaction endpoints")
            directions[(sender, receiver)] += 1
            good = leg.get("good")
            if good not in TRADE_GOODS or leg.get("unit") != "trade-good-units":
                raise ValueError("trade leg good/unit differs from registry")
            if leg.get("deliveryPhase") not in {"opening", "after-production"}:
                raise ValueError("trade leg delivery phase invalid")
            if leg.get("state") not in {"delivered", "partial", "undelivered", "party-vanished"}:
                raise ValueError("trade leg state invalid")
            amounts = {
                name: _number(leg.get(name), f"trade leg {name}", nonnegative=True)
                for name in (
                    "requestedQuantity",
                    "agreedQuantity",
                    "stockQuantity",
                    "madeToOrderQuantity",
                    "escrowedQuantity",
                    "producedMadeToOrderQuantity",
                    "supplyAvailableQuantity",
                    "bilateralDeliverableQuantity",
                    "deliveredQuantity",
                    "acceptedQuantity",
                    "senderDebitQuantity",
                    "receiverCreditQuantity",
                    "deliveryFraction",
                    "producerDeliveryFraction",
                    "storageBlockedQuantity",
                    "counterpartyShortfallQuantity",
                    "refundedOrReturnedQuantity",
                    "destroyedQuantity",
                )
            }
            if not math.isclose(
                amounts["stockQuantity"] + amounts["madeToOrderQuantity"],
                amounts["agreedQuantity"],
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ):
                raise ValueError("trade leg stock/made split does not equal agreement")
            if not math.isclose(
                amounts["deliveredQuantity"],
                amounts["acceptedQuantity"],
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ):
                raise ValueError("trade leg delivery differs from accepted credit")
            if not math.isclose(
                amounts["deliveredQuantity"],
                amounts["senderDebitQuantity"],
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ) or not math.isclose(
                amounts["deliveredQuantity"],
                amounts["receiverCreditQuantity"],
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ):
                raise ValueError("trade leg delivery differs from actual debit/credit receipts")
            expected_fraction = (
                amounts["deliveredQuantity"] / amounts["agreedQuantity"]
                if amounts["agreedQuantity"] > TOLERANCE
                else 0.0
            )
            if not math.isclose(
                amounts["deliveryFraction"], expected_fraction, rel_tol=0.0, abs_tol=TOLERANCE
            ):
                raise ValueError("trade leg delivery fraction is not derived")
            expected_producer_fraction = (
                min(1.0, amounts["producedMadeToOrderQuantity"] / amounts["madeToOrderQuantity"])
                if amounts["madeToOrderQuantity"] > TOLERANCE
                else 1.0
            )
            if not math.isclose(
                amounts["producerDeliveryFraction"],
                expected_producer_fraction,
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ):
                raise ValueError("trade leg producer delivery fraction is not derived")
            if not math.isclose(
                amounts["deliveredQuantity"] + amounts["storageBlockedQuantity"],
                amounts["bilateralDeliverableQuantity"],
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ):
                raise ValueError("trade leg delivery/storage closure differs")
            if amounts["refundedOrReturnedQuantity"] > amounts["escrowedQuantity"] + TOLERANCE:
                raise ValueError("trade leg refund exceeds escrow")
            if not math.isclose(
                amounts["supplyAvailableQuantity"],
                amounts["escrowedQuantity"] + amounts["producedMadeToOrderQuantity"],
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ):
                raise ValueError("trade leg supply does not reconcile to escrow and production")
            if not math.isclose(
                amounts["counterpartyShortfallQuantity"],
                amounts["supplyAvailableQuantity"] - amounts["bilateralDeliverableQuantity"],
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ):
                raise ValueError("trade leg counterparty shortfall does not reconcile")
            if not math.isclose(
                amounts["refundedOrReturnedQuantity"],
                amounts["escrowedQuantity"]
                - min(amounts["escrowedQuantity"], amounts["deliveredQuantity"]),
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ):
                raise ValueError("trade leg refund/return does not reconcile")
            if amounts["destroyedQuantity"] != 0:
                raise ValueError("trade leg reports unsupported destruction")
            expected_state = (
                "delivered"
                if amounts["deliveredQuantity"] >= amounts["agreedQuantity"] - TOLERANCE
                else "partial"
                if amounts["deliveredQuantity"] > TOLERANCE
                else "undelivered"
            )
            if row["settlementState"] == "party-vanished":
                expected_state = "party-vanished"
            if leg["state"] != expected_state:
                raise ValueError("trade leg state differs from delivered quantity")
            parts = _rows(leg.get("parts"), "trade leg parts")
            if tuple(part.get("item") for part in parts) != TRADE_PARTS[good]:
                raise ValueError("trade leg parts differ from stable good registry")
            for part in parts:
                _exact_keys(
                    part,
                    {
                        "item",
                        "stockQuantity",
                        "madeToOrderQuantity",
                        "deliveredQuantity",
                    },
                    "trade part",
                )
                for name in ("stockQuantity", "madeToOrderQuantity", "deliveredQuantity"):
                    _number(part.get(name), f"trade part {name}", nonnegative=True)
                endpoint = (
                    physical_in
                    if receiver == sidecar.get("campId")
                    else physical_out
                    if sender == sidecar.get("campId")
                    else None
                )
                if endpoint is not None:
                    endpoint[str(part["item"])] += float(part["deliveredQuantity"])
            for name in ("stockQuantity", "madeToOrderQuantity", "deliveredQuantity"):
                if not math.isclose(
                    sum(float(part[name]) for part in parts),
                    amounts[name],
                    rel_tol=0.0,
                    abs_tol=TOLERANCE,
                ):
                    raise ValueError(f"trade leg part {name} does not reconcile")
        if set(directions.values()) != {1} or set(directions) != {
            (requester, responder),
            (responder, requester),
        }:
            raise ValueError("trade transaction legs are not exactly reciprocal")
        if any(leg["state"] != row["settlementState"] for leg in legs):
            raise ValueError("trade transaction settlement differs from leg state")
        if any(
            not math.isclose(
                float(leg["deliveryFraction"]),
                float(row["deliveryFraction"]),
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            )
            for leg in legs
        ):
            raise ValueError("bilateral trade legs have unequal delivery fractions")
        eligibility_by_camp = {entry["campId"]: entry for entry in eligibility}
        if requester not in eligibility_by_camp or responder not in eligibility_by_camp:
            raise ValueError("trade lifecycle is missing reciprocal endpoint eligibility")
        if (
            not eligibility_by_camp[requester]["eligible"]
            or not eligibility_by_camp[responder]["eligible"]
        ):
            raise ValueError("settled trade has an ineligible endpoint")
        wanted_leg = next(leg for leg in legs if leg["receiver"] == requester)
        offered_leg = next(leg for leg in legs if leg["receiver"] == responder)
        if (
            wanted_leg["good"] not in request["wantedGoods"]
            or offered_leg["good"] not in request["offeredGoods"]
        ):
            raise ValueError("trade transaction goods differ from bound requisition goods")
        if row["settlementState"] == "party-vanished" and (
            row["requesterImmediateUtility"] is not None
            or row["responderImmediateUtility"] is not None
        ):
            raise ValueError("party-vanished trade cannot assert terminal utility")
    if len(used_requests) != len(set(used_requests)):
        raise ValueError("one trade requisition produced multiple transactions")
    for request in requests:
        if (request["outcome"] in {"honoured", "countered"}) != (
            request["requestId"] in used_requests
        ):
            raise ValueError("trade requisition/transaction lifecycle is incomplete")
    for row in [*sidecar["goods"], *sidecar["materials"]]:
        item = str(row.get("good", row.get("material")))
        if not math.isclose(
            float(row["actualTradeIn"]),
            physical_in[item],
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ) or not math.isclose(
            float(row["actualTradeOut"]),
            physical_out[item],
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ):
            raise ValueError(f"trade lifecycle physical parts do not reconcile for {item}")
    food_ledger = next(
        row
        for row in sidecar["conservationRows"]
        if row["domain"] == "food" and row["item"] == "linked-food-reserves"
    )
    food_items = ("PlantFood", "AnimalFood", "AquaticFood")
    if not math.isclose(
        float(food_ledger["tradeIn"]),
        sum(physical_in[item] for item in food_items),
        rel_tol=0.0,
        abs_tol=TOLERANCE,
    ) or not math.isclose(
        float(food_ledger["tradeOut"]),
        sum(physical_out[item] for item in food_items),
        rel_tol=0.0,
        abs_tol=TOLERANCE,
    ):
        raise ValueError("trade lifecycle food parts do not reconcile")


def _validate_required_coverage(sidecar: dict[str, Any]) -> None:
    domains = _object(sidecar.get("domains"), "domains")
    controls = _rows(sidecar.get("controls"), "controls")
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for control in controls:
        by_domain.setdefault(str(control["domain"]), []).append(control)
    for name in ("cultivation", "herding", "manufacturing", "trade"):
        _object(domains.get(name), f"domains.{name}")
    cultivation = domains["cultivation"]
    herding = domains["herding"]
    manufacturing = domains["manufacturing"]
    trade = domains["trade"]
    population = _object(domains.get("population"), "domains.population")
    for name, domain in (("cultivation", cultivation), ("herding", herding)):
        if not isinstance(domain.get("capable"), bool):
            raise ValueError(f"domains.{name}.capable must be boolean")
    if not isinstance(manufacturing.get("active"), bool):
        raise ValueError("domains.manufacturing.active must be boolean")
    if not isinstance(trade.get("eligible"), bool):
        raise ValueError("domains.trade.eligible must be boolean")

    requests = _rows(sidecar.get("requests"), "requests")
    request_ids = [_text(row.get("requestId"), "request.requestId") for row in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("duplicate manufacturing request identity")
    transactions = [row for row in sidecar["trades"] if row.get("recordType") == "transaction"]
    any_authority = (
        cultivation["capable"]
        or herding["capable"]
        or manufacturing["active"]
        or trade["eligible"]
        or bool(requests)
        or bool(transactions)
    )
    if any_authority and not controls:
        raise ValueError("authoritative active domains require causal controls")

    expected_strategic = {
        "cultivation": {"retainedFieldOperation", "fieldDevelopment"},
        "herding": {"retainedHerdFraction", "herdDevelopment"},
    }
    for domain_name, identities in expected_strategic.items():
        domain = domains[domain_name]
        emitted = {row["identity"] for row in by_domain.get(domain_name, [])}
        if domain["capable"] and not identities <= emitted:
            raise ValueError(f"active {domain_name} capability lacks required causal controls")
        if not domain["capable"] and any(
            row["classification"] == "active" for row in by_domain.get(domain_name, [])
        ):
            raise ValueError(f"inactive {domain_name} capability has active controls")

    manufacturing_controls = {row["identity"]: row for row in by_domain.get("manufacturing", [])}
    if set(request_ids) - set(manufacturing_controls):
        raise ValueError("manufacturing requests lack causal control rows")
    for request in requests:
        control = manufacturing_controls[request["requestId"]]
        phases = control["phases"]
        for phase, request_field in (
            ("nativeDemand", "nativeDemand"),
            ("actorCappedRequest", "actorCappedRequest"),
            ("compilerGrant", "compilerGrant"),
            ("executedWorkerOrInputClaim", "executedWorkerYears"),
            ("committedPhysicalReceipt", "committedPhysicalReceipt"),
            ("inventoryChange", "inventoryChange"),
            ("consumptionOrFinalService", "consumptionOrFinalService"),
        ):
            expected = request.get(request_field)
            if (expected is None) != (phases[phase] is None) or (
                expected is not None
                and not math.isclose(
                    float(expected), float(phases[phase]), rel_tol=0.0, abs_tol=TOLERANCE
                )
            ):
                raise ValueError("manufacturing control differs from its bound request")
        if phases["committedPhysicalReceipt"] is not None:
            if request.get("good") is not None:
                bound_receipt = sum(
                    float(event["quantity"])
                    for event in sidecar["physicalEvents"]
                    if event["category"] == "production-receipt"
                    and event["item"] == request["good"]
                    and event.get("sourceRequestId") == request["requestId"]
                )
            elif request.get("rawMaterial") is not None:
                bound_receipt = sum(
                    float(event["quantity"])
                    for event in sidecar["physicalEvents"]
                    if event["category"] == "material-extraction-receipt"
                    and event["item"] == request["rawMaterial"]
                    and event.get("receiptId") == request["requestId"]
                )
            else:
                raise ValueError("physical receipt is available without a physical request kind")
            if not math.isclose(
                bound_receipt,
                float(phases["committedPhysicalReceipt"]),
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ):
                raise ValueError("manufacturing receipt is not bound to its source request")

    own_eligibility = next(
        row
        for row in sidecar["trades"]
        if row.get("recordType") == "eligibility" and row.get("campId") == sidecar.get("campId")
    )
    if bool(own_eligibility["eligible"]) != bool(trade["eligible"]):
        raise ValueError("trade domain eligibility differs from authoritative lifecycle")
    trade_controls = {row["identity"]: row for row in by_domain.get("trade", [])}
    camp_id = sidecar.get("campId")
    for transaction in transactions:
        endpoint_utility = transaction[
            "requesterImmediateUtility"
            if transaction["requester"] == camp_id
            else "responderImmediateUtility"
        ]
        for leg in transaction["legs"]:
            if camp_id not in {leg["sender"], leg["receiver"]}:
                continue
            control = trade_controls.get(leg["legId"])
            if control is None:
                raise ValueError("trade endpoint leg lacks a causal control row")
            phases = control["phases"]
            expected_inventory = (
                float(leg["receiverCreditQuantity"])
                if leg["receiver"] == camp_id
                else -float(leg["senderDebitQuantity"])
            )
            expected = {
                "nativeDemand": leg["requestedQuantity"],
                "actorCappedRequest": leg["agreedQuantity"],
                "compilerGrant": leg["agreedQuantity"],
                "executedWorkerOrInputClaim": leg["supplyAvailableQuantity"],
                "committedPhysicalReceipt": leg["deliveredQuantity"],
                "inventoryChange": expected_inventory,
                "consumptionOrFinalService": endpoint_utility,
            }
            for phase, value in expected.items():
                if (value is None) != (phases[phase] is None) or (
                    value is not None
                    and not math.isclose(
                        float(value), float(phases[phase]), rel_tol=0.0, abs_tol=TOLERANCE
                    )
                ):
                    raise ValueError("trade control is not cross-bound to lifecycle receipts")

    ending_population = _number(
        population.get("endingLinked"), "population.endingLinked", nonnegative=True
    )
    ending_qol = _number(population.get("qol"), "population.qol", nonnegative=True)
    for control in controls:
        for phase, expected in (("endingPopulation", ending_population), ("endingQol", ending_qol)):
            if control["phases"][phase] is not None and not math.isclose(
                float(control["phases"][phase]), expected, rel_tol=0.0, abs_tol=TOLERANCE
            ):
                raise ValueError(f"control {phase} differs from population outcome")


def validate_economic_accounting(sidecar: dict[str, Any]) -> None:
    """Reject malformed or incomplete-schema v2 accounting evidence.

    A correctly derived non-zero residual remains readable evidence; it is a Gate 2
    completeness HOLD and is reported by :func:`accounting_completeness`.
    """

    _object(sidecar, "economic sidecar")
    if sidecar.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unsupported Gate 2 economic accounting schema")
    _validate_controls(sidecar)
    _validate_physical_rows(sidecar)
    _validate_conservation(sidecar)
    _validate_trade(sidecar)
    _validate_required_coverage(sidecar)


def accounting_completeness(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic fail-closed Gate 2 completeness classification."""

    validate_economic_accounting(sidecar)
    reasons: list[str] = []
    gross_population_residual = float(sidecar["conservation"]["grossPopulationResidual"])
    if abs(gross_population_residual) > TOLERANCE:
        reasons.append(f"nonzero gross population residual: {gross_population_residual}")
    for row in sidecar["conservationRows"]:
        identity = f"{row['domain']}:{row['item']}"
        if row["complete"] is not True:
            reasons.append(f"incomplete conservation row {identity}: {row['unavailableReason']}")
        elif abs(float(row["residual"])) > TOLERANCE:
            reasons.append(f"nonzero conservation residual {identity}: {row['residual']}")
    domains = sidecar["domains"]
    authoritative_domain_active = {
        "cultivation": bool(domains["cultivation"]["capable"]),
        "herding": bool(domains["herding"]["capable"]),
        "manufacturing": bool(domains["manufacturing"]["active"]),
        "trade": bool(domains["trade"]["eligible"]),
    }
    excluded_controls: list[dict[str, str]] = []
    complete_evidence_controls = 0
    evidence_control_denominator = 0
    for control in sidecar["controls"]:
        classification = control["classification"]
        domain_active = authoritative_domain_active.get(str(control["domain"]), True)
        if classification == "active":
            evidence_control_denominator += 1
            if not control["unavailable"]:
                complete_evidence_controls += 1
            for phase, reason in control["unavailable"].items():
                reasons.append(f"control {control['controlId']} missing {phase}: {reason}")
            continue
        if domain_active:
            evidence_control_denominator += 1
            reasons.append(
                f"active-capability control {control['controlId']} is {classification}: "
                f"{control['classificationReason']}"
            )
            continue
        # An explicitly classified control outside the decision-opening
        # capability set is coverage evidence, not a failed causal measurement.
        # Keep it in the total count and expose why it was excluded from the
        # active-evidence denominator without turning its inapplicable phases
        # into completeness failures.
        excluded_controls.append(
            {
                "controlId": str(control["controlId"]),
                "classification": str(classification),
                "reason": str(control["classificationReason"]),
            }
        )
    for transaction in (row for row in sidecar["trades"] if row.get("recordType") == "transaction"):
        if transaction.get("delayedUtility") is None:
            reasons.append(
                "trade transaction "
                f"{transaction['transactionId']} has no matched-intervention delayed causal effect"
            )
    active_terminal_missing = any(
        control["classification"] == "active"
        and control["phases"]["consumptionOrFinalService"] is None
        for control in sidecar["controls"]
    )
    unsupported_active_capability = any(
        control["classification"] != "active"
        and authoritative_domain_active.get(str(control["domain"]), True)
        for control in sidecar["controls"]
    )
    return {
        "status": "PASS-ACCOUNTING-ONLY" if not reasons else "HOLD",
        "complete": not reasons,
        "reasons": sorted(reasons),
        "controlCount": len(sidecar["controls"]),
        "activeControlCount": sum(
            control["classification"] == "active" for control in sidecar["controls"]
        ),
        "inactiveControlCount": sum(
            control["classification"] == "inactive" for control in sidecar["controls"]
        ),
        "unsupportedControlCount": sum(
            control["classification"] == "unsupported" for control in sidecar["controls"]
        ),
        "controlEvidenceDenominator": evidence_control_denominator,
        "completeEvidenceControlCount": complete_evidence_controls,
        "excludedControlCount": len(excluded_controls),
        "excludedControls": sorted(excluded_controls, key=lambda row: row["controlId"]),
        "conservationCount": len(sidecar["conservationRows"]),
        "tradeTransactionCount": sum(
            row.get("recordType") == "transaction" for row in sidecar["trades"]
        ),
        "terminalLinkComplete": not active_terminal_missing
        and not unsupported_active_capability
        and not any(
            row.get("recordType") == "transaction" and row.get("delayedUtility") is None
            for row in sidecar["trades"]
        ),
    }
