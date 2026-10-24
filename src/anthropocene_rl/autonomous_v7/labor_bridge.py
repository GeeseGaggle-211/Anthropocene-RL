"""Typed learner boundary for the opt-in specialty-labor contract.

The native bridge has not activated contract 13 yet.  These helpers validate
the reviewed ordered registries and the labor portion of action 12 without
changing or accepting action 11.  Observation/retry containers are added only
when the native producer emits their final field registries.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

NATIVE_CONTRACT = 13
ACTION_SCHEMA = 12
OBSERVATION_SCHEMA = 13
ENVIRONMENT_SCHEMA = 13
REPLAY_SCHEMA = 13
LABOR_REGISTRY_SCHEMA = 1

SPECIALTIES = (
    "idle",
    "Gatherer",
    "Farmer",
    "Herder",
    "Fisher",
    "Craftsperson",
    "Toolmaker",
    "Potter",
    "Weaver",
    "Carpenter",
    "Builder",
    "Woodcutter",
    "Stonecutter",
    "ClayMiner",
)
ACTIVITIES = (
    "idle",
    "food.gather",
    "stock.stone",
    "stock.clay",
    "stock.wood",
    "field.operation",
    "field.expansion",
    "herd.management",
    "herd.expansion",
    "fishing.harvest",
    "fishing.development",
    "goods.hand-tools",
    "goods.pots",
    "goods.cloth",
    "goods.farm-tools",
    "goods.housing",
    "goods.tools",
    "goods.animal-fabric",
    "goods.light-housing",
    "goods.herding-tools",
    "goods.housing-material",
    "goods.fishing-tools",
)
SPECIALTY_ROWS = len(SPECIALTIES)
ROLE_ROWS = SPECIALTY_ROWS - 1
ACTIVITY_ROWS = len(ACTIVITIES)
SIMPLEX_TOLERANCE = 1e-8
RESOURCE_IDS = (
    "resident-work",
    "stone",
    "clay",
    "wood",
    "plant-food",
    "animal-food",
    "animal-fabric",
    "housing-material",
    "cloth",
    "hand-tools",
    "tools",
    "farm-tools",
    "herding-tools",
    "fishing-tools",
    "terrestrial-food",
    "aquatic-food",
    "pasture-food",
    "freshwater",
    "cultivable-capital",
    "food-room",
)
RESOURCE_ROWS = len(RESOURCE_IDS)
GOODS_ROWS = 11
REASON_IDS = (
    "none",
    "catalog-unavailable",
    "unsupported-action-schema",
    "specialty-row-count",
    "specialty-nonfinite",
    "specialty-negative",
    "specialty-masked-nonzero",
    "specialty-sum",
    "activity-specialty-row-count",
    "activity-row-count",
    "activity-nonfinite",
    "activity-negative",
    "activity-masked-nonzero",
    "activity-sum",
    "fallback-after-retry",
    "actual-work-unavailable",
    "planned-output-unavailable",
    "realized-output-unavailable",
    "resource-receipt-unavailable",
    "camp-vanished",
)
REASON_REGISTRY_VERSION = 1
SPECIALTY_FEATURES = (
    "present",
    "known",
    "actorControllable",
    "wholeResidents",
    "isGlobalIdle",
    "canGatherFood",
    "canExtractRaw",
    "canCultivate",
    "canHerd",
    "canFish",
    "canCraft",
    "requiredTechnologySatisfied",
)
ACTIVITY_FEATURES = (
    "present",
    "known",
    "actorControllable",
    "fractionalWorkerTime",
    "nonlinear",
    "visibleCompetitionPossible",
    "phaseCode",
    "unitCode",
    "outputGoodCodePlus1",
    "scheduleOrdinalPlus1",
    "maxWorkerYearsKnown",
    "maxWorkerYears",
    "outputCapacityKnown",
    "capacityPerWorkerYearLower",
    "capacityPerWorkerYearUpper",
    "stonePerOutput",
    "clayPerOutput",
    "woodPerOutput",
    "plantFoodPerOutput",
    "animalFoodPerOutput",
    "animalFabricPerOutput",
    "housingMaterialPerOutput",
    "clothPerOutput",
    "constructionMaterialPerOutput",
    "toolServicePerOutput",
    "specialistOnly",
    "isIdle",
    "isFoodGather",
    "isRawExtract",
    "isField",
    "isHerd",
    "isFishing",
    "isCraft",
    "outputUnitCode",
    "isOperationOrHarvest",
    "isDevelopmentOrExpansion",
    "outputsStone",
    "outputsClay",
    "outputsWood",
    "goodHandTools",
    "goodPots",
    "goodCloth",
    "goodFarmTools",
    "goodHousing",
    "goodTools",
    "goodAnimalFabric",
    "goodLightHousing",
    "goodHerdingTools",
    "goodHousingMaterial",
    "goodFishingTools",
)
RESOURCE_FEATURES = (
    "present",
    "known",
    "phaseCode",
    "unitCode",
    "opening",
    "locallyAccessible",
    "fixedReserved",
    "competitionPossible",
)
SPECIALTY_HISTORY_WIDTH = 11
ACTIVITY_HISTORY_WIDTH = 18
RESOURCE_HISTORY_WIDTH = 4
SPECIALTY_RETRY_WIDTH = 9
ACTIVITY_RETRY_WIDTH = 10
LABOR_SPECIALTY_FEATURE_WIDTH = (
    len(SPECIALTY_FEATURES) + SPECIALTY_HISTORY_WIDTH + SPECIALTY_RETRY_WIDTH
)
LABOR_ACTIVITY_FEATURE_WIDTH = (
    len(ACTIVITY_FEATURES) + ACTIVITY_HISTORY_WIDTH + ACTIVITY_RETRY_WIDTH
)
LABOR_RESOURCE_FEATURE_WIDTH = len(RESOURCE_FEATURES) + RESOURCE_ROWS + RESOURCE_HISTORY_WIDTH


def labor_registry_digest(
    specialties: Sequence[str] = SPECIALTIES, activities: Sequence[str] = ACTIVITIES
) -> str:
    canonical = (
        f"1|12|{len(ACTIVITY_FEATURES)}|" + "|".join(specialties) + "|" + "|".join(activities)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


LABOR_REGISTRY_DIGEST = labor_registry_digest()
REASON_REGISTRY_DIGEST = hashlib.sha256(
    ("1|0|0|" + "|".join(REASON_IDS)).encode("utf-8")
).hexdigest()


def _dotnet_roundtrip(value: float) -> str:
    """Canonical finite-double spelling used by native ``ToString("R")``."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("catalog digest values must be finite")
    if number == 0.0:
        return "0"
    text = repr(number)
    if text.endswith(".0") and "e" not in text:
        text = text[:-2]
    if "e" in text:
        mantissa, exponent = text.split("e")
        sign = "+" if int(exponent) >= 0 else "-"
        text = f"{mantissa}E{sign}{abs(int(exponent)):02d}"
    return text


def labor_catalog_digest(payload: Mapping[str, Any]) -> str:
    """Recompute the producer's content digest over ordered numeric rows."""
    parts = [
        str(payload["schemaVersion"]),
        str(payload["registryVersion"]),
        str(payload["registryDigest"]),
        str(payload["reasonRegistryDigest"]),
        str(payload["campId"]),
        str(payload["turn"]),
        str(payload["population"]),
        _dotnet_roundtrip(_number(payload["turnYears"], "labor catalog turnYears")),
    ]
    text = "|".join(parts)
    for row in payload["specialtyRows"]:
        text += "|" + str(row["id"])
        text += "".join(
            "," + _dotnet_roundtrip(_number(item, "specialty digest feature"))
            for item in row["features"]
        )
    for row in payload["activityRows"]:
        text += "|" + str(row["specialtyId"]) + ":" + str(row["activityId"])
        text += "".join(
            "," + _dotnet_roundtrip(_number(item, "activity digest feature"))
            for item in row["features"]
        )
    for row in payload["resources"]:
        text += "|" + str(row["resourceId"])
        text += "".join(
            "," + _dotnet_roundtrip(_number(item, "resource digest feature"))
            for item in row["features"]
        )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reason_code(value: Any, text: str, label: str, *, available: bool | None = None) -> int:
    code = _integer(value, label)
    if code >= len(REASON_IDS):
        raise ValueError(f"{label} is outside the reviewed registry")
    if available is True and (code != 0 or text):
        raise ValueError(f"{label} must be none for an available value")
    if available is False and (code == 0 or not text):
        raise ValueError(f"{label} must explain an unavailable value")
    if text and text != REASON_IDS[code]:
        raise ValueError(f"{label} text/code disagree")
    return code


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _text(value: Any, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _digest(value: Any, label: str) -> str:
    text = _text(value, label)
    assert text is not None
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return text


def _fields(value: Any, names: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != names:
        raise ValueError(f"{label} fields must be exactly {sorted(names)}")
    return value


def _number_vector(
    value: Any, length: int, label: str, *, nonnegative: bool = False, nullable: bool = False
) -> tuple[float | None, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{label} must have length {length}")
    result: list[float | None] = []
    for index, item in enumerate(value):
        if nullable and item is None:
            result.append(None)
            continue
        number = _number(item, f"{label}[{index}]")
        if nonnegative and number < 0:
            raise ValueError(f"{label}[{index}] must be nonnegative")
        result.append(number)
    return tuple(result)


def _number_matrix(
    value: Any,
    rows: int,
    columns: int,
    label: str,
    *,
    nonnegative: bool = False,
    nullable: bool = False,
) -> tuple[tuple[float | None, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != rows:
        raise ValueError(f"{label} must have {rows} rows")
    return tuple(
        _number_vector(
            row, columns, f"{label}[{index}]", nonnegative=nonnegative, nullable=nullable
        )
        for index, row in enumerate(value)
    )


def _bool_matrix(value: Any, rows: int, columns: int, label: str) -> tuple[tuple[bool, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != rows:
        raise ValueError(f"{label} must have {rows} rows")
    return tuple(_bool_vector(row, columns, f"{label}[{index}]") for index, row in enumerate(value))


def _integer_vector(value: Any, length: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{label} must have length {length}")
    return tuple(_integer(item, f"{label}[{index}]") for index, item in enumerate(value))


def _integer_matrix(value: Any, rows: int, columns: int, label: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != rows:
        raise ValueError(f"{label} must have {rows} rows")
    return tuple(
        _integer_vector(row, columns, f"{label}[{index}]") for index, row in enumerate(value)
    )


def _text_matrix(value: Any, rows: int, columns: int, label: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != rows:
        raise ValueError(f"{label} must have {rows} rows")
    result = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != columns:
            raise ValueError(f"{label}[{row_index}] must have length {columns}")
        if any(not isinstance(item, str) for item in row):
            raise ValueError(f"{label}[{row_index}] values must be strings")
        result.append(tuple(row))
    return tuple(result)


def _bool_vector(value: Any, length: int, label: str) -> tuple[bool, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{label} must have length {length}")
    if any(not isinstance(item, bool) for item in value):
        raise ValueError(f"{label} values must be booleans")
    return tuple(value)


def _simplex(value: Any, mask: tuple[bool, ...], label: str) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != len(mask)
    ):
        raise ValueError(f"{label} must have length {len(mask)}")
    result = tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))
    if any(item < 0.0 or item > 1.0 for item in result):
        raise ValueError(f"{label} values must be in [0, 1]")
    if any(not legal and item != 0.0 for item, legal in zip(result, mask, strict=True)):
        raise ValueError(f"{label} assigns mass to a masked coordinate")
    expected = 1.0 if any(mask) else 0.0
    if not math.isclose(math.fsum(result), expected, rel_tol=0.0, abs_tol=SIMPLEX_TOLERANCE):
        raise ValueError(f"{label} must sum to {expected}")
    if expected == 0.0:
        return result
    # Canonicalize at ingestion as well as serialization. This preserves the
    # emitted coordinates while making a parse/serialize/parse cycle exact,
    # even when float32 softmax coordinates add to 1 only within tolerance.
    reference = max(
        index
        for index, (item, legal) in enumerate(zip(result, mask, strict=True))
        if legal and item > 0.0
    )
    canonical = list(result)
    canonical[reference] = 1.0 - math.fsum(
        item for index, item in enumerate(canonical) if index != reference
    )
    if canonical[reference] < 0.0:
        reference = max(
            (
                index
                for index, (item, legal) in enumerate(zip(result, mask, strict=True))
                if legal and item > 0.0
            ),
            key=lambda idx: result[idx],
        )
        canonical = list(result)
        canonical[reference] = 1.0 - math.fsum(
            item for index, item in enumerate(canonical) if index != reference
        )
    if canonical[reference] < 0.0:
        raise ValueError(f"{label} cannot be canonicalized as a simplex")
    return tuple(canonical)


@dataclass(frozen=True)
class LaborProducerContract:
    """Exact registry/version record required from a contract-13 producer."""

    def state_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": LABOR_REGISTRY_SCHEMA,
            "nativeContract": NATIVE_CONTRACT,
            "actionSchema": ACTION_SCHEMA,
            "observationSchema": OBSERVATION_SCHEMA,
            "environmentSchema": ENVIRONMENT_SCHEMA,
            "replaySchema": REPLAY_SCHEMA,
            "featureFlag": "actorLaborRetryV1",
            "specialties": list(SPECIALTIES),
            "activities": list(ACTIVITIES),
            "specialtyRows": SPECIALTY_ROWS,
            "roleRows": ROLE_ROWS,
            "activityRows": ACTIVITY_ROWS,
            "resourceRows": RESOURCE_ROWS,
            "resources": list(RESOURCE_IDS),
            "registryDigest": LABOR_REGISTRY_DIGEST,
            "reasonRegistryVersion": REASON_REGISTRY_VERSION,
            "reasonIds": list(REASON_IDS),
            "reasonRegistryDigest": REASON_REGISTRY_DIGEST,
            "populationAuthority": "full-decision-boundary-controllable-population",
            "apportionment": "largest-remainder-stable-ordinal",
            "activitySemantics": "staffing-work-share-not-output-promise",
            "maxProposals": 2,
            "attemptFeasibilityCostMaximum": 319,
            "decisionFeasibilityCostMaximum": 638,
            "fallbackSurcharge": 0,
            "settlementGuardCostMaximum": 1,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> LaborProducerContract:
        expected = cls().state_dict()
        if not isinstance(state, Mapping) or set(state) != set(expected):
            raise ValueError(f"labor producer contract fields must be exactly {sorted(expected)}")
        for key, value in expected.items():
            if state[key] != value:
                raise ValueError(
                    f"labor producer contract {key} differs from the reviewed registry"
                )
        return cls()


@dataclass(frozen=True)
class LaborMasks:
    specialty: tuple[bool, ...]
    activity: tuple[tuple[bool, ...], ...]

    @classmethod
    def from_values(cls, specialty: Any, activity: Any) -> LaborMasks:
        specialty_mask = _bool_vector(specialty, SPECIALTY_ROWS, "specialty mask")
        if not specialty_mask[0]:
            raise ValueError("idle specialty must always be legal")
        if (
            not isinstance(activity, Sequence)
            or isinstance(activity, (str, bytes))
            or len(activity) != ROLE_ROWS
        ):
            raise ValueError(f"activity mask must have {ROLE_ROWS} role rows")
        activity_mask = tuple(
            _bool_vector(row, ACTIVITY_ROWS, f"activity mask[{index}]")
            for index, row in enumerate(activity)
        )
        # The native v1 registry emits every physical role row, including an
        # unavailable role, and always preserves its explicit idle activity.
        # Model-level padding for a future registry is separate from this wire
        # validator and is handled by ``labor_model._masks``.
        for role, row in enumerate(activity_mask, start=1):
            if not row[0]:
                raise ValueError("idle activity must be legal for every physical role")
            if not specialty_mask[role] and any(row[1:]):
                raise ValueError("masked specialty cannot advertise non-idle activities")
        return cls(specialty_mask, activity_mask)


@dataclass(frozen=True)
class LaborPolicySample:
    specialty_allocation: tuple[float, ...]
    activity_allocation: tuple[tuple[float, ...], ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], masks: LaborMasks) -> LaborPolicySample:
        fields = {"laborSpecialtyAllocation", "laborActivityAllocation"}
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise ValueError(f"labor action fields must be exactly {sorted(fields)}")
        specialty = _simplex(
            payload["laborSpecialtyAllocation"], masks.specialty, "laborSpecialtyAllocation"
        )
        raw_activity = payload["laborActivityAllocation"]
        if (
            not isinstance(raw_activity, Sequence)
            or isinstance(raw_activity, (str, bytes))
            or len(raw_activity) != ROLE_ROWS
        ):
            raise ValueError(f"laborActivityAllocation must have {ROLE_ROWS} role rows")
        activity = tuple(
            _simplex(row, masks.activity[index], f"laborActivityAllocation[{index}]")
            for index, row in enumerate(raw_activity)
        )
        return cls(specialty, activity)

    def payload(self) -> dict[str, Any]:
        def canonical(values: tuple[float, ...]) -> list[float]:
            result = list(values)
            nonzero = [index for index, item in enumerate(result) if item > 0]
            if not nonzero:
                raise ValueError("cannot serialize an empty labor simplex")
            reference = nonzero[-1]
            ref_val = 1.0 - math.fsum(
                item for index, item in enumerate(result) if index != reference
            )
            if ref_val < 0.0:
                reference = max(nonzero, key=lambda idx: result[idx])
                ref_val = 1.0 - math.fsum(
                    item for index, item in enumerate(result) if index != reference
                )
            result[reference] = ref_val
            if result[reference] < 0 or not math.isclose(
                math.fsum(result), 1.0, rel_tol=0.0, abs_tol=SIMPLEX_TOLERANCE
            ):
                raise ValueError("labor simplex cannot be serialized canonically")
            return result

        return {
            "laborSpecialtyAllocation": canonical(self.specialty_allocation),
            "laborActivityAllocation": [canonical(row) for row in self.activity_allocation],
        }


@dataclass(frozen=True)
class LaborCatalog:
    camp_id: int
    turn: int
    population: int
    turn_years: float
    masks: LaborMasks
    specialty_rows: tuple[dict[str, Any], ...]
    activity_rows: tuple[dict[str, Any], ...]
    resources: tuple[dict[str, Any], ...]
    digest: str
    # Identity/count/order are frozen. Numeric row semantics are deliberately
    # unavailable to the learner until their producer registry is finalized.
    semantic_rows_supported: bool = True

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LaborCatalog:
        names = {
            "schemaVersion",
            "registryVersion",
            "registryDigest",
            "reasonRegistryVersion",
            "reasonIds",
            "reasonRegistryDigest",
            "campId",
            "turn",
            "population",
            "turnYears",
            "specialtyIds",
            "activityIds",
            "specialtyMask",
            "activityMask",
            "specialtyRows",
            "activityRows",
            "resources",
            "digest",
        }
        value = _fields(payload, names, "labor catalog")
        if value["schemaVersion"] != 1:
            raise ValueError("unsupported labor catalog schema")
        if value["specialtyIds"] != list(SPECIALTIES) or value["activityIds"] != list(ACTIVITIES):
            raise ValueError("labor catalog registries differ from its producer contract")
        if value["registryVersion"] != LABOR_REGISTRY_SCHEMA:
            raise ValueError("unsupported labor registry version")
        registry_digest = _digest(value["registryDigest"], "labor registry digest")
        if registry_digest != LABOR_REGISTRY_DIGEST:
            raise ValueError("labor catalog registry digest differs from the learner registry")
        if (
            value["reasonRegistryVersion"] != REASON_REGISTRY_VERSION
            or value["reasonIds"] != list(REASON_IDS)
            or _digest(value["reasonRegistryDigest"], "reason registry digest")
            != REASON_REGISTRY_DIGEST
        ):
            raise ValueError("labor catalog reason registry differs from the learner registry")
        masks = LaborMasks.from_values(value["specialtyMask"], value["activityMask"])
        specialty_raw = value["specialtyRows"]
        if not isinstance(specialty_raw, list) or len(specialty_raw) != SPECIALTY_ROWS:
            raise ValueError("labor catalog specialty row inventory is incomplete")
        specialty_rows = []
        specialty_fields = {
            "row",
            "id",
            "requiredTechnology",
            "unavailableReason",
            "present",
            "known",
            "actorControllable",
            "wholeResidents",
            "features",
        }
        for index, row in enumerate(specialty_raw):
            row = _fields(row, specialty_fields, f"labor specialty row {index}")
            if row["row"] != index or row["id"] != SPECIALTIES[index]:
                raise ValueError("labor catalog specialty row identity/order differs")
            for key in ("requiredTechnology", "unavailableReason"):
                if row[key] is not None and not isinstance(row[key], str):
                    raise ValueError(f"labor specialty row {key} is malformed")
            for key in ("present", "known", "actorControllable", "wholeResidents"):
                if not isinstance(row[key], bool):
                    raise ValueError(f"labor specialty row {key} must be boolean")
            features = _number_vector(
                row["features"], len(SPECIALTY_FEATURES), f"labor specialty row {index} features"
            )
            if any(item not in (0.0, 1.0) for item in features):
                raise ValueError("labor specialty features must be typed booleans")
            expected_flags = tuple(
                1.0 if row[key] else 0.0
                for key in ("present", "known", "actorControllable", "wholeResidents")
            )
            if features[:4] != expected_flags:
                raise ValueError("labor specialty features disagree with their typed fields")
            specialty_rows.append(copy.deepcopy(dict(row)))
            if masks.specialty[index] != row["actorControllable"]:
                raise ValueError("labor specialty action mask disagrees with its row")
        rows = value["activityRows"]
        if not isinstance(rows, list) or len(rows) != ROLE_ROWS * ACTIVITY_ROWS:
            raise ValueError("labor catalog activity row inventory is incomplete")
        activity_rows = []
        for flat, row in enumerate(rows):
            role, activity = divmod(flat, ACTIVITY_ROWS)
            activity_fields = {
                "row",
                "specialtyRow",
                "activityRow",
                "specialtyId",
                "activityId",
                "phase",
                "unit",
                "outputGood",
                "outputUnit",
                "unavailableReason",
                "present",
                "known",
                "actorControllable",
                "fractionalWorkerTime",
                "nonlinear",
                "visibleCompetitionPossible",
                "maxWorkerYearsKnown",
                "outputCapacityKnown",
                "specialistOnly",
                "isOperationOrHarvest",
                "isDevelopmentOrExpansion",
                "outputsStone",
                "outputsClay",
                "outputsWood",
                "recipeScheduleOrdinal",
                "maxWorkerYears",
                "capacityPerWorkerYearLower",
                "capacityPerWorkerYearUpper",
                "stonePerOutput",
                "clayPerOutput",
                "woodPerOutput",
                "plantFoodPerOutput",
                "animalFoodPerOutput",
                "animalFabricPerOutput",
                "housingMaterialPerOutput",
                "clothPerOutput",
                "constructionMaterialPerOutput",
                "toolServicePerOutput",
                "features",
            }
            row = _fields(row, activity_fields, f"labor activity row {flat}")
            expected = {
                "row": flat,
                "specialtyRow": role,
                "activityRow": activity,
                "specialtyId": SPECIALTIES[role + 1],
                "activityId": ACTIVITIES[activity],
            }
            if any(row.get(key) != item for key, item in expected.items()):
                raise ValueError("labor catalog activity row identity/order differs")
            for key in ("phase", "unit", "outputGood", "outputUnit", "unavailableReason"):
                if row[key] is not None and not isinstance(row[key], str):
                    raise ValueError(f"labor activity row {key} is malformed")
            flag_names = (
                "present",
                "known",
                "actorControllable",
                "fractionalWorkerTime",
                "nonlinear",
                "visibleCompetitionPossible",
                "maxWorkerYearsKnown",
                "outputCapacityKnown",
                "specialistOnly",
                "isOperationOrHarvest",
                "isDevelopmentOrExpansion",
                "outputsStone",
                "outputsClay",
                "outputsWood",
            )
            if any(not isinstance(row[key], bool) for key in flag_names):
                raise ValueError("labor activity row flags must be boolean")
            schedule_ordinal = _number(row["recipeScheduleOrdinal"], "recipe schedule ordinal")
            if schedule_ordinal < -1:
                raise ValueError("recipe schedule ordinal must be at least -1")
            for key in (
                "maxWorkerYears",
                "capacityPerWorkerYearLower",
                "capacityPerWorkerYearUpper",
                "stonePerOutput",
                "clayPerOutput",
                "woodPerOutput",
                "plantFoodPerOutput",
                "animalFoodPerOutput",
                "animalFabricPerOutput",
                "housingMaterialPerOutput",
                "clothPerOutput",
                "constructionMaterialPerOutput",
                "toolServicePerOutput",
            ):
                if _number(row[key], f"labor activity row {key}") < 0:
                    raise ValueError(f"labor activity row {key} must be nonnegative")
            features = _number_vector(
                row["features"], len(ACTIVITY_FEATURES), f"labor activity row {flat} features"
            )
            expected_flags = tuple(1.0 if row[key] else 0.0 for key in flag_names)
            boolean_indices = (*range(6), 10, 12, *range(25, 33), *range(34, 50))
            if any(features[index] not in (0.0, 1.0) for index in boolean_indices):
                raise ValueError("labor activity boolean features must be exactly zero or one")
            numeric_fields = (
                "maxWorkerYears",
                "capacityPerWorkerYearLower",
                "capacityPerWorkerYearUpper",
                "stonePerOutput",
                "clayPerOutput",
                "woodPerOutput",
                "plantFoodPerOutput",
                "animalFoodPerOutput",
                "animalFabricPerOutput",
                "housingMaterialPerOutput",
                "clothPerOutput",
                "constructionMaterialPerOutput",
                "toolServicePerOutput",
            )
            expected_numeric = tuple(float(row[key]) for key in numeric_fields)
            if (
                features[:6] != expected_flags[:6]
                or features[9] != schedule_ordinal + 1.0
                or features[10] != (1.0 if row["maxWorkerYearsKnown"] else 0.0)
                or features[12] != (1.0 if row["outputCapacityKnown"] else 0.0)
                or features[11:12] + features[13:25] != expected_numeric
                or features[25] != (1.0 if row["specialistOnly"] else 0.0)
                or features[34:39] != expected_flags[9:14]
            ):
                raise ValueError("labor activity features disagree with their typed fields")
            if features[13] > features[14]:
                raise ValueError("labor activity capacity bounds are reversed")
            if not row["maxWorkerYearsKnown"] and row["maxWorkerYears"] != 0:
                raise ValueError("unknown labor maximum must not carry a value")
            if not row["outputCapacityKnown"] and (
                row["capacityPerWorkerYearLower"] != 0 or row["capacityPerWorkerYearUpper"] != 0
            ):
                raise ValueError("unknown labor output capacity must not carry values")
            activity_rows.append(copy.deepcopy(dict(row)))
            if masks.activity[role][activity] != row["actorControllable"]:
                raise ValueError("labor activity action mask disagrees with its row")
        resources = value["resources"]
        if not isinstance(resources, list) or len(resources) != RESOURCE_ROWS:
            raise ValueError("labor catalog resource inventory is incomplete")
        resource_rows = []
        for index, row in enumerate(resources):
            row = _fields(
                row,
                {
                    "row",
                    "resourceId",
                    "unit",
                    "phase",
                    "reason",
                    "present",
                    "known",
                    "competitionPossible",
                    "opening",
                    "locallyAccessible",
                    "fixedReserved",
                    "features",
                },
                f"labor resource row {index}",
            )
            if row["row"] != index or row["resourceId"] != RESOURCE_IDS[index]:
                raise ValueError("labor catalog resource identity/order differs")
            unit = _text(row["unit"], f"labor resource row {index} unit")
            phase = _text(row["phase"], f"labor resource row {index} phase")
            if row["reason"] is not None and not isinstance(row["reason"], str):
                raise ValueError("labor resource reason is malformed")
            if any(
                not isinstance(row[key], bool)
                for key in ("present", "known", "competitionPossible")
            ):
                raise ValueError("labor resource typed flags must be booleans")
            quantities = tuple(
                _number(row[key], f"labor resource row {index} {key}")
                for key in ("opening", "locallyAccessible", "fixedReserved")
            )
            if any(item < 0 for item in quantities):
                raise ValueError("labor resource quantities must be nonnegative")
            features = _number_vector(
                row["features"], len(RESOURCE_FEATURES), f"labor resource row {index} features"
            )
            if (
                features[0] not in (0.0, 1.0)
                or features[1] not in (0.0, 1.0)
                or features[7] not in (0.0, 1.0)
                or any(item < 0 for item in features[4:7])
            ):
                raise ValueError("labor resource features have invalid flags/quantities")
            phase_code = {"opening": 1.0, "work-before-production": 2.0, "production": 3.0}.get(
                phase, 0.0
            )
            unit_code = {
                "worker-years": 1.0,
                "good-units": 2.0,
                "material-units": 3.0,
                "animal-heads": 4.0,
            }.get(unit, 0.0)
            expected_features = (
                float(row["present"]),
                float(row["known"]),
                phase_code,
                unit_code,
                *quantities,
                float(row["competitionPossible"]),
            )
            if features != expected_features:
                raise ValueError("labor resource features disagree with their typed fields")
            if not row["known"] and (any(quantities) or not row["reason"]):
                raise ValueError("unknown labor resource needs a reason and zero quantities")
            if row["known"] and row["reason"]:
                raise ValueError("known labor resource cannot carry an unavailable reason")
            resource_rows.append(copy.deepcopy(dict(row)))
        years = _number(value["turnYears"], "labor catalog turnYears")
        if years <= 0:
            raise ValueError("labor catalog turnYears must be positive")
        digest = _digest(value["digest"], "catalog digest")
        if labor_catalog_digest(value) != digest:
            raise ValueError("labor catalog content digest does not match its rows")
        return cls(
            _integer(value["campId"], "labor catalog campId"),
            _integer(value["turn"], "labor catalog turn"),
            _integer(value["population"], "labor catalog population"),
            years,
            masks,
            tuple(specialty_rows),
            tuple(activity_rows),
            tuple(resource_rows),
            digest,
        )


@dataclass(frozen=True)
class LaborViolation:
    reason_code: int
    reason: str
    knowledge: str
    unit: str
    phase: str
    specialty_rows: tuple[int, ...]
    activity_rows: tuple[int, ...]
    requested: float | None
    available: float | None
    severity: float
    avoidable: bool | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LaborViolation:
        names = {
            "reasonCode",
            "reason",
            "knowledge",
            "unit",
            "phase",
            "specialtyRows",
            "activityRows",
            "requested",
            "requestedKnown",
            "available",
            "availableKnown",
            "severity",
            "avoidable",
        }
        value = _fields(payload, names, "labor violation")
        specialties = value["specialtyRows"]
        activities = value["activityRows"]
        if (
            not isinstance(specialties, list)
            or any(
                _integer(item, "violation specialty row") >= SPECIALTY_ROWS for item in specialties
            )
            or len(set(specialties)) != len(specialties)
        ):
            raise ValueError("labor violation specialty rows are invalid")
        if (
            not isinstance(activities, list)
            or any(
                _integer(item, "violation activity row") >= ROLE_ROWS * ACTIVITY_ROWS
                for item in activities
            )
            or len(set(activities)) != len(activities)
        ):
            raise ValueError("labor violation activity rows are invalid")
        avoidable = value["avoidable"]
        if avoidable is not None and not isinstance(avoidable, bool):
            raise ValueError("labor violation avoidable must be boolean or null")
        if not isinstance(value["requestedKnown"], bool) or not isinstance(
            value["availableKnown"], bool
        ):
            raise ValueError("labor violation known flags must be booleans")
        requested_value = _number(value["requested"], "violation requested placeholder")
        available_value = _number(value["available"], "violation available placeholder")
        severity = _number(value["severity"], "labor violation severity")
        if severity < 0 or severity > 1:
            raise ValueError("labor violation severity must be in [0, 1]")
        reason = _text(value["reason"], "violation reason")
        return cls(
            _reason_code(value["reasonCode"], reason, "violation reason code"),
            reason,
            _text(value["knowledge"], "violation knowledge"),
            _text(value["unit"], "violation unit"),
            _text(value["phase"], "violation phase"),
            tuple(specialties),
            tuple(activities),
            requested_value if value["requestedKnown"] else None,
            available_value if value["availableKnown"] else None,
            severity,
            avoidable,
        )


@dataclass(frozen=True)
class LaborAttempt:
    camp_id: int
    attempt: int
    remaining_proposals: int
    valid: bool
    fallback: bool
    requested_specialty_allocation: tuple[float, ...]
    requested_specialty_residents: tuple[float | None, ...]
    accepted_specialty_residents: tuple[int, ...]
    requested_activity_allocation: tuple[tuple[float, ...], ...]
    requested_activity_worker_years: tuple[tuple[float | None, ...], ...]
    planned_activity_worker_years: tuple[tuple[float | None, ...], ...]
    violations: tuple[LaborViolation, ...]
    catalog_digest: str
    feasibility_cost: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], catalog: LaborCatalog) -> LaborAttempt:
        names = {
            "schemaVersion",
            "campId",
            "attempt",
            "remainingProposals",
            "valid",
            "fallback",
            "requestedSpecialtyAllocation",
            "requestedSpecialtyResidents",
            "acceptedSpecialtyResidents",
            "requestedActivityAllocation",
            "requestedActivityWorkerYears",
            "plannedActivityWorkerYears",
            "violations",
            "catalogDigest",
            "feasibilityCost",
        }
        value = _fields(payload, names, "labor attempt")
        if value["schemaVersion"] != 1 or value["campId"] != catalog.camp_id:
            raise ValueError("labor attempt schema/camp differs from catalog")
        attempt = _integer(value["attempt"], "labor attempt index")
        remaining = _integer(value["remainingProposals"], "remaining proposals")
        if attempt > 1 or remaining > 1 or attempt + remaining > 1:
            raise ValueError("labor attempt exceeds the two-proposal protocol")
        if not isinstance(value["valid"], bool) or not isinstance(value["fallback"], bool):
            raise ValueError("labor attempt valid/fallback fields must be booleans")
        if value["valid"]:
            specialty = _simplex(
                value["requestedSpecialtyAllocation"],
                catalog.masks.specialty,
                "requestedSpecialtyAllocation",
            )
        else:
            # Rejected previews are evidence of the exact malformed request.
            # Preserve finite coordinates so typed violations can explain it;
            # never rewrite an invalid request into a policy simplex.
            specialty = tuple(
                _number_vector(
                    value["requestedSpecialtyAllocation"],
                    SPECIALTY_ROWS,
                    "requestedSpecialtyAllocation",
                )
            )
        requested_residents = _number_vector(
            value["requestedSpecialtyResidents"],
            SPECIALTY_ROWS,
            "requestedSpecialtyResidents",
            nonnegative=value["valid"],
        )
        if value["valid"] and any(
            not math.isclose(requested, share * catalog.population, rel_tol=0.0, abs_tol=1e-7)
            for requested, share in zip(requested_residents, specialty, strict=True)
        ):
            raise ValueError(
                "requested specialty residents differ from the sampled population shares"
            )
        accepted_raw = value["acceptedSpecialtyResidents"]
        if not isinstance(accepted_raw, list) or len(accepted_raw) != SPECIALTY_ROWS:
            raise ValueError("acceptedSpecialtyResidents has the wrong width")
        accepted = tuple(_integer(item, "accepted specialty residents") for item in accepted_raw)
        if value["valid"] and sum(accepted) != catalog.population:
            raise ValueError(
                "accepted specialties do not allocate the full decision-boundary population"
            )
        if value["valid"] and accepted != apportion_largest_remainder(
            specialty, catalog.population
        ):
            raise ValueError(
                "accepted specialties differ from stable largest-remainder apportionment"
            )
        raw_activity = value["requestedActivityAllocation"]
        if not isinstance(raw_activity, list) or len(raw_activity) != ROLE_ROWS:
            raise ValueError("requestedActivityAllocation has the wrong row count")
        if value["valid"]:
            activities = tuple(
                _simplex(
                    row, catalog.masks.activity[index], f"requestedActivityAllocation[{index}]"
                )
                for index, row in enumerate(raw_activity)
            )
        else:
            activities = tuple(
                tuple(_number_vector(row, ACTIVITY_ROWS, f"requestedActivityAllocation[{index}]"))
                for index, row in enumerate(raw_activity)
            )
        requested_work = _number_matrix(
            value["requestedActivityWorkerYears"],
            ROLE_ROWS,
            ACTIVITY_ROWS,
            "requestedActivityWorkerYears",
            nonnegative=value["valid"],
        )
        planned_work = _number_matrix(
            value["plannedActivityWorkerYears"],
            ROLE_ROWS,
            ACTIVITY_ROWS,
            "plannedActivityWorkerYears",
            nonnegative=value["valid"],
        )
        if value["valid"]:
            for role in range(ROLE_ROWS):
                requested_capacity = requested_residents[role + 1] * catalog.turn_years
                planned_capacity = accepted[role + 1] * catalog.turn_years
                if not math.isclose(
                    sum(item for item in requested_work[role] if item is not None),
                    requested_capacity,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise ValueError(
                        "requested activity worker-years differ from fractional specialty request"
                    )
                expected_rows = tuple(requested_capacity * share for share in activities[role])
                if any(
                    not math.isclose(requested, expected_row, rel_tol=0.0, abs_tol=1e-7)
                    for requested, expected_row in zip(
                        requested_work[role], expected_rows, strict=True
                    )
                ):
                    raise ValueError(
                        "requested activity worker-years differ from sampled activity shares"
                    )
                planned_rows = tuple(planned_capacity * share for share in activities[role])
                if any(
                    not math.isclose(planned, expected_row, rel_tol=0.0, abs_tol=1e-7)
                    for planned, expected_row in zip(planned_work[role], planned_rows, strict=True)
                ):
                    raise ValueError(
                        "planned activity worker-years differ from accepted specialty residents"
                    )
        raw_violations = value["violations"]
        if not isinstance(raw_violations, list):
            raise ValueError("labor attempt violations must be a list")
        violations = tuple(LaborViolation.from_payload(row) for row in raw_violations)
        if not value["valid"] and not violations:
            raise ValueError("invalid labor attempt requires a typed violation")
        cost = _number(value["feasibilityCost"], "labor attempt feasibilityCost")
        if cost < 0 or (value["valid"] and cost != 0):
            raise ValueError("labor attempt feasibility cost is inconsistent")
        catalog_digest = _digest(value["catalogDigest"], "attempt catalog digest")
        if catalog_digest != catalog.digest:
            raise ValueError("labor attempt is bound to another catalog")
        return cls(
            catalog.camp_id,
            attempt,
            remaining,
            value["valid"],
            value["fallback"],
            specialty,
            requested_residents,
            accepted,
            activities,
            requested_work,
            planned_work,
            violations,
            catalog_digest,
            cost,
        )


def _known_nullable(
    values: tuple[Any, ...],
    known: tuple[bool, ...],
    reasons: tuple[str, ...],
    reason_codes: tuple[int, ...],
    label: str,
) -> None:
    for index, (value, available, reason, code) in enumerate(
        zip(values, known, reasons, reason_codes, strict=True)
    ):
        if available:
            if value is None or reason:
                raise ValueError(f"{label}[{index}] known value/reason is inconsistent")
        elif value is not None or not reason:
            raise ValueError(f"{label}[{index}] unavailable value needs a nonempty reason")
        _reason_code(code, reason, f"{label}[{index}] reason code", available=available)


@dataclass(frozen=True)
class LaborFeedback:
    camp_id: int
    decision_turn: int
    execution_comparator_attempt: int
    parent_decision_id: str
    founding_parent_decision_id: str | None
    decision_catalog: LaborCatalog
    initial_request: LaborAttempt
    final_sampled_request: LaborAttempt
    committed_activity_worker_years: tuple[tuple[float | None, ...], ...]
    actual_scheduler_consumed_worker_years: tuple[tuple[float | None, ...], ...]
    actual_worker_years_known: tuple[tuple[bool, ...], ...]
    actual_worker_years_reasons: tuple[tuple[str, ...], ...]
    actual_worker_years_reason_codes: tuple[tuple[int, ...], ...]
    planned_goods_output: tuple[float | None, ...]
    realized_goods_output: tuple[float | None, ...]
    planned_goods_output_known: tuple[bool, ...]
    planned_goods_output_reasons: tuple[str, ...]
    planned_goods_output_reason_codes: tuple[int, ...]
    goods_output_known: tuple[bool, ...]
    goods_output_reasons: tuple[str, ...]
    goods_output_reason_codes: tuple[int, ...]
    realized_resource_receipts: tuple[float | None, ...]
    resource_receipt_known: tuple[bool, ...]
    resource_receipt_reasons: tuple[str, ...]
    resource_receipt_reason_codes: tuple[int, ...]
    fallback: bool
    fallback_reason: str
    fallback_reason_code: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], catalog: LaborCatalog) -> LaborFeedback:
        names = {
            "schemaVersion",
            "campId",
            "decisionTurn",
            "executionComparatorAttempt",
            "parentDecisionId",
            "foundingParentDecisionId",
            "decisionCatalog",
            "initialRequest",
            "finalSampledRequest",
            "committedActivityWorkerYears",
            "actualSchedulerConsumedWorkerYears",
            "actualWorkerYearsKnown",
            "actualWorkerYearsReasons",
            "actualWorkerYearsReasonCodes",
            "plannedGoodsOutput",
            "plannedGoodsOutputKnown",
            "plannedGoodsOutputReasons",
            "plannedGoodsOutputReasonCodes",
            "realizedGoodsOutput",
            "goodsOutputKnown",
            "goodsOutputReasons",
            "goodsOutputReasonCodes",
            "realizedResourceReceipts",
            "resourceReceiptKnown",
            "resourceReceiptReasons",
            "resourceReceiptReasonCodes",
            "fallback",
            "fallbackReason",
            "fallbackReasonCode",
        }
        value = _fields(payload, names, "labor feedback")
        if value["schemaVersion"] != 1 or value["campId"] != catalog.camp_id:
            raise ValueError("labor feedback schema/camp differs from catalog")
        decision_catalog = LaborCatalog.from_payload(value["decisionCatalog"])
        if decision_catalog.camp_id != catalog.camp_id:
            raise ValueError("labor feedback decision/current catalog camps differ")
        initial = LaborAttempt.from_payload(value["initialRequest"], decision_catalog)
        final = LaborAttempt.from_payload(value["finalSampledRequest"], decision_catalog)
        comparator = _integer(value["executionComparatorAttempt"], "execution comparator attempt")
        if comparator != final.attempt or initial.attempt != 0 or final.attempt not in (0, 1):
            raise ValueError(
                "labor feedback execution comparator does not bind the final sampled request"
            )
        committed = _number_matrix(
            value["committedActivityWorkerYears"],
            ROLE_ROWS,
            ACTIVITY_ROWS,
            "committedActivityWorkerYears",
            nonnegative=True,
        )
        actual = _number_matrix(
            value["actualSchedulerConsumedWorkerYears"],
            ROLE_ROWS,
            ACTIVITY_ROWS,
            "actualSchedulerConsumedWorkerYears",
            nonnegative=True,
            nullable=True,
        )
        known = _bool_matrix(
            value["actualWorkerYearsKnown"], ROLE_ROWS, ACTIVITY_ROWS, "actualWorkerYearsKnown"
        )
        reasons = _text_matrix(
            value["actualWorkerYearsReasons"], ROLE_ROWS, ACTIVITY_ROWS, "actualWorkerYearsReasons"
        )
        reason_codes = _integer_matrix(
            value["actualWorkerYearsReasonCodes"],
            ROLE_ROWS,
            ACTIVITY_ROWS,
            "actualWorkerYearsReasonCodes",
        )
        for role in range(ROLE_ROWS):
            _known_nullable(
                actual[role],
                known[role],
                reasons[role],
                reason_codes[role],
                f"actualSchedulerConsumedWorkerYears[{role}]",
            )
        planned_goods = _number_vector(
            value["plannedGoodsOutput"],
            GOODS_ROWS,
            "plannedGoodsOutput",
            nonnegative=True,
            nullable=True,
        )
        realized_goods = _number_vector(
            value["realizedGoodsOutput"],
            GOODS_ROWS,
            "realizedGoodsOutput",
            nonnegative=True,
            nullable=True,
        )
        planned_known = _bool_vector(
            value["plannedGoodsOutputKnown"], GOODS_ROWS, "plannedGoodsOutputKnown"
        )
        planned_reasons_raw = value["plannedGoodsOutputReasons"]
        if (
            not isinstance(planned_reasons_raw, list)
            or len(planned_reasons_raw) != GOODS_ROWS
            or any(not isinstance(item, str) for item in planned_reasons_raw)
        ):
            raise ValueError("plannedGoodsOutputReasons is malformed")
        planned_reasons = tuple(planned_reasons_raw)
        planned_reason_codes = _integer_vector(
            value["plannedGoodsOutputReasonCodes"], GOODS_ROWS, "plannedGoodsOutputReasonCodes"
        )
        goods_known = _bool_vector(value["goodsOutputKnown"], GOODS_ROWS, "goodsOutputKnown")
        goods_reasons_raw = value["goodsOutputReasons"]
        if (
            not isinstance(goods_reasons_raw, list)
            or len(goods_reasons_raw) != GOODS_ROWS
            or any(not isinstance(item, str) for item in goods_reasons_raw)
        ):
            raise ValueError("goodsOutputReasons is malformed")
        goods_reasons = tuple(goods_reasons_raw)
        goods_reason_codes = _integer_vector(
            value["goodsOutputReasonCodes"], GOODS_ROWS, "goodsOutputReasonCodes"
        )
        _known_nullable(
            planned_goods,
            planned_known,
            planned_reasons,
            planned_reason_codes,
            "plannedGoodsOutput",
        )
        _known_nullable(
            realized_goods, goods_known, goods_reasons, goods_reason_codes, "realizedGoodsOutput"
        )
        resource_receipts = _number_vector(
            value["realizedResourceReceipts"],
            RESOURCE_ROWS,
            "realizedResourceReceipts",
            nonnegative=True,
            nullable=True,
        )
        resource_known = _bool_vector(
            value["resourceReceiptKnown"], RESOURCE_ROWS, "resourceReceiptKnown"
        )
        resource_reasons_raw = value["resourceReceiptReasons"]
        if (
            not isinstance(resource_reasons_raw, list)
            or len(resource_reasons_raw) != RESOURCE_ROWS
            or any(not isinstance(item, str) for item in resource_reasons_raw)
        ):
            raise ValueError("resourceReceiptReasons is malformed")
        resource_reasons = tuple(resource_reasons_raw)
        resource_reason_codes = _integer_vector(
            value["resourceReceiptReasonCodes"], RESOURCE_ROWS, "resourceReceiptReasonCodes"
        )
        _known_nullable(
            resource_receipts,
            resource_known,
            resource_reasons,
            resource_reason_codes,
            "realizedResourceReceipts",
        )
        if not isinstance(value["fallback"], bool) or not isinstance(value["fallbackReason"], str):
            raise ValueError("labor feedback fallback fields are malformed")
        if value["fallback"] != final.fallback or value["fallback"] != bool(
            value["fallbackReason"]
        ):
            raise ValueError("labor feedback fallback reason/attempt are inconsistent")
        fallback_code = _reason_code(
            value["fallbackReasonCode"],
            value["fallbackReason"],
            "fallback reason code",
            available=not value["fallback"],
        )
        return cls(
            catalog.camp_id,
            _integer(value["decisionTurn"], "labor decision turn"),
            comparator,
            _text(value["parentDecisionId"], "parent decision id"),
            _text(value["foundingParentDecisionId"], "founding parent decision id", optional=True),
            decision_catalog,
            initial,
            final,
            committed,
            actual,
            known,
            reasons,
            reason_codes,
            planned_goods,
            realized_goods,
            planned_known,
            planned_reasons,
            planned_reason_codes,
            goods_known,
            goods_reasons,
            goods_reason_codes,
            resource_receipts,
            resource_known,
            resource_reasons,
            resource_reason_codes,
            value["fallback"],
            value["fallbackReason"],
            fallback_code,
        )


def labor_observation_fields(
    observation: Mapping[str, Any],
) -> tuple[LaborCatalog, LaborFeedback | None]:
    """Parse only the two contract-13 additions; legacy observation fields remain elsewhere."""
    if "laborCatalog" not in observation or "laborPrevious" not in observation:
        raise ValueError("contract-13 observation lacks labor catalog/previous feedback fields")
    catalog = LaborCatalog.from_payload(observation["laborCatalog"])
    previous_raw = observation["laborPrevious"]
    previous = None if previous_raw is None else LaborFeedback.from_payload(previous_raw, catalog)
    return catalog, previous


def labor_retry_field(response: Mapping[str, Any], catalog: LaborCatalog) -> LaborAttempt | None:
    """Parse zero-time rejection feedback without accepting execution feedback."""
    if "laborRetry" not in response:
        raise ValueError("contract-13 response lacks laborRetry")
    raw = response["laborRetry"]
    if raw is None:
        return None
    if any(
        key in raw
        for key in (
            "committedActivityWorkerYears",
            "actualSchedulerConsumedWorkerYears",
            "realizedGoodsOutput",
            "executionComparatorAttempt",
        )
    ):
        raise ValueError("retry feedback leaks post-execution fields")
    attempt = LaborAttempt.from_payload(raw, catalog)
    if attempt.valid:
        raise ValueError("laborRetry may only carry a rejected attempt")
    return attempt


def labor_observation_tensors(
    observation: Mapping[str, Any],
    retry_response: Mapping[str, Any] | None = None,
) -> tuple[LaborCatalog, LaborFeedback | None, LaborAttempt | None, dict[str, np.ndarray]]:
    """Strict raw-wire to bounded labor-token boundary for one decision.

    ``laborPrevious`` is historical post-execution feedback. ``laborRetry`` is
    a separate zero-time rejected attempt. Keeping these separate prevents a
    retry from seeing final receipts that do not exist until after world
    advancement.
    """
    catalog, previous = labor_observation_fields(observation)
    rejected = None
    if retry_response is not None:
        rejected = labor_retry_field(retry_response, catalog)
        if rejected is None:
            raise ValueError("retry response does not contain a rejected labor attempt")
    tensors = labor_catalog_tensors(catalog, previous, rejected)
    return catalog, previous, rejected, tensors


def labor_observation_batch(
    observations: Sequence[Mapping[str, Any]],
    retry_responses: Sequence[Mapping[str, Any] | None] | None = None,
    *,
    rejected_attempts: Sequence[LaborAttempt | None] | None = None,
) -> tuple[
    tuple[LaborCatalog, LaborFeedback | None, LaborAttempt | None], ..., dict[str, np.ndarray]
]:
    """Stack strict labor token rows without coercing registry dimensions."""
    if (
        not isinstance(observations, Sequence)
        or isinstance(observations, (str, bytes))
        or not observations
    ):
        raise ValueError("labor observation batch must be a nonempty sequence")
    if retry_responses is not None and rejected_attempts is not None:
        raise ValueError("raw retry responses and typed rejected attempts are mutually exclusive")
    if rejected_attempts is not None:
        if len(rejected_attempts) != len(observations):
            raise ValueError("typed labor retry batch differs from observation batch")
        parsed = []
        for observation, rejected in zip(observations, rejected_attempts, strict=True):
            catalog, previous = labor_observation_fields(observation)
            if rejected is not None and not isinstance(rejected, LaborAttempt):
                raise ValueError("typed labor retry values must be LaborAttempt or None")
            tensors = labor_catalog_tensors(catalog, previous, rejected)
            parsed.append((catalog, previous, rejected, tensors))
    else:
        if retry_responses is None:
            retries: Sequence[Mapping[str, Any] | None] = (None,) * len(observations)
        else:
            if len(retry_responses) != len(observations):
                raise ValueError("labor retry batch differs from observation batch")
            retries = retry_responses
        parsed = [
            labor_observation_tensors(observation, retry)
            for observation, retry in zip(observations, retries, strict=True)
        ]
    names = set(parsed[0][3])
    if any(set(row[3]) != names for row in parsed):
        raise ValueError("labor tensor fields differ across the batch")
    batch = {name: np.stack([row[3][name] for row in parsed], axis=0) for name in sorted(names)}
    contexts = tuple((catalog, previous, rejected) for catalog, previous, rejected, _ in parsed)
    return contexts, batch


def _bounded_catalog_features(catalog: LaborCatalog) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    specialty = np.asarray([row["features"] for row in catalog.specialty_rows], dtype=np.float64)
    activity = np.asarray([row["features"] for row in catalog.activity_rows], dtype=np.float64)
    resources = np.asarray([row["features"] for row in catalog.resources], dtype=np.float64)
    # Specialty features are all typed flags.
    specialty = np.clip(specialty, 0.0, 1.0)
    # Categorical activity fields have reviewed finite registries; physical
    # rates/quantities use a fixed symlog guard rather than fitted tiny stds.
    activity_index = {name: index for index, name in enumerate(ACTIVITY_FEATURES)}
    activity[:, activity_index["phaseCode"]] /= 2.0
    activity[:, activity_index["unitCode"]] /= 1.0
    activity[:, activity_index["outputGoodCodePlus1"]] /= float(GOODS_ROWS)
    activity[:, activity_index["scheduleOrdinalPlus1"]] /= float(GOODS_ROWS)
    physical = [
        activity_index[name]
        for name in (
            "maxWorkerYears",
            "capacityPerWorkerYearLower",
            "capacityPerWorkerYearUpper",
            "stonePerOutput",
            "clayPerOutput",
            "woodPerOutput",
            "plantFoodPerOutput",
            "animalFoodPerOutput",
            "animalFabricPerOutput",
            "housingMaterialPerOutput",
            "clothPerOutput",
            "constructionMaterialPerOutput",
            "toolServicePerOutput",
        )
    ]
    activity[:, physical] = np.sign(activity[:, physical]) * np.log1p(np.abs(activity[:, physical]))
    activity[:, physical] = np.clip(activity[:, physical], -10.0, 10.0)
    activity[:, activity_index["outputUnitCode"]] /= 4.0
    resource_index = {name: index for index, name in enumerate(RESOURCE_FEATURES)}
    resources[:, resource_index["phaseCode"]] /= 2.0
    resources[:, resource_index["unitCode"]] /= 4.0
    resource_physical = [
        resource_index[name] for name in ("opening", "locallyAccessible", "fixedReserved")
    ]
    resources[:, resource_physical] = np.sign(resources[:, resource_physical]) * np.log1p(
        np.abs(resources[:, resource_physical])
    )
    resources[:, resource_physical] = np.clip(resources[:, resource_physical], -10.0, 10.0)
    # Stable resource identities are otherwise physically aliased (for
    # example opening stone/clay/wood can carry equal numeric features).
    # Specialty identity is instead expressed by capabilities plus its owned
    # activity tokens; activity identity is in the producer's width-50
    # semantic feature contract. Neither uses a flat learned row ordinal.
    resource_identity = np.eye(RESOURCE_ROWS, dtype=np.float64)
    return (
        specialty.astype(np.float32),
        activity.astype(np.float32),
        np.concatenate([resources, resource_identity], axis=1).astype(np.float32),
    )


def labor_catalog_tensors(
    catalog: LaborCatalog,
    previous: LaborFeedback | None = None,
    rejected_attempt: LaborAttempt | None = None,
) -> dict[str, np.ndarray]:
    """Create bounded, identity-aligned labor tokens for the new model.

    Text reasons remain in parsed evidence. The reviewed numeric reason-code
    registry is admitted to bounded token features; free text is never hashed
    into a model input.
    """
    specialty, activity, resources = _bounded_catalog_features(catalog)
    specialty_history = np.zeros((SPECIALTY_ROWS, SPECIALTY_HISTORY_WIDTH), np.float32)
    activity_history = np.zeros((ROLE_ROWS * ACTIVITY_ROWS, ACTIVITY_HISTORY_WIDTH), np.float32)
    resource_history = np.zeros((RESOURCE_ROWS, RESOURCE_HISTORY_WIDTH), np.float32)
    if previous is not None:
        prior = previous.decision_catalog
        if prior.camp_id != catalog.camp_id:
            raise ValueError("previous labor feedback belongs to another camp")
        if tuple(row["id"] for row in prior.specialty_rows) != SPECIALTIES:
            raise ValueError("previous labor specialty registry differs")
        capacity = max(1e-12, prior.population * prior.turn_years)
        specialty_history[:, 0] = 1.0
        specialty_history[:, 1] = previous.final_sampled_request.requested_specialty_allocation
        specialty_history[:, 2] = np.asarray(
            previous.final_sampled_request.requested_specialty_residents, np.float32
        ) / max(1, prior.population)
        specialty_history[:, 3] = np.asarray(
            previous.final_sampled_request.accepted_specialty_residents, np.float32
        ) / max(1, prior.population)
        for role in range(ROLE_ROWS):
            start = role * ACTIVITY_ROWS
            stop = start + ACTIVITY_ROWS
            requested = np.asarray(
                previous.final_sampled_request.requested_activity_worker_years[role], np.float32
            )
            planned = np.asarray(
                previous.final_sampled_request.planned_activity_worker_years[role], np.float32
            )
            committed = np.asarray(previous.committed_activity_worker_years[role], np.float32)
            actual_known = np.asarray(previous.actual_worker_years_known[role], bool)
            actual = np.asarray(
                [
                    0.0 if value is None else value
                    for value in previous.actual_scheduler_consumed_worker_years[role]
                ],
                np.float32,
            )
            specialty_history[role + 1, 4] = planned.sum() / capacity
            specialty_history[role + 1, 5] = committed.sum() / capacity
            specialty_history[role + 1, 6] = float(actual_known.mean())
            specialty_history[role + 1, 7] = actual.sum() / capacity
            specialty_history[role + 1, 8] = float(previous.fallback)
            specialty_history[role + 1, 9] = float(previous.execution_comparator_attempt)
            specialty_history[role + 1, 10] = previous.fallback_reason_code / (len(REASON_IDS) - 1)
            requested_allocation = np.asarray(
                previous.final_sampled_request.requested_activity_allocation[role], np.float32
            )
            block = activity_history[start:stop]
            block[:, 0] = 1.0
            block[:, 1] = requested_allocation
            block[:, 2] = requested / capacity
            block[:, 3] = planned / capacity
            block[:, 4] = committed / capacity
            block[:, 5] = actual / capacity
            block[:, 6] = actual_known.astype(np.float32)
            # Deltas follow the declared chronology: accepted/planned minus
            # requested, committed minus accepted, actual minus committed.
            block[:, 7] = (planned - requested) / capacity
            block[:, 8] = (committed - planned) / capacity
            block[:, 9] = np.where(actual_known, actual - committed, 0.0) / capacity
            # Goods attach to the corresponding stable craft activity for every
            # specialty token; known zero stays distinct from unavailable.
            for activity_index in range(11, ACTIVITY_ROWS):
                good = activity_index - 11
                block[activity_index, 10] = (
                    0.0
                    if previous.planned_goods_output[good] is None
                    else np.float32(
                        np.clip(np.log1p(previous.planned_goods_output[good]), 0.0, 10.0)
                    )
                )
                block[activity_index, 11] = float(previous.planned_goods_output_known[good])
                block[activity_index, 12] = (
                    0.0
                    if previous.realized_goods_output[good] is None
                    else np.float32(
                        np.clip(np.log1p(previous.realized_goods_output[good]), 0.0, 10.0)
                    )
                )
                block[activity_index, 13] = float(previous.goods_output_known[good])
                if previous.planned_goods_output_known[good] and previous.goods_output_known[good]:
                    block[activity_index, 14] = np.clip(
                        np.sign(
                            previous.realized_goods_output[good]
                            - previous.planned_goods_output[good]
                        )
                        * np.log1p(
                            abs(
                                previous.realized_goods_output[good]
                                - previous.planned_goods_output[good]
                            )
                        ),
                        -10.0,
                        10.0,
                    )
                block[activity_index, 16] = previous.planned_goods_output_reason_codes[good] / (
                    len(REASON_IDS) - 1
                )
                block[activity_index, 17] = previous.goods_output_reason_codes[good] / (
                    len(REASON_IDS) - 1
                )
            block[:, 15] = np.asarray(
                previous.actual_worker_years_reason_codes[role], np.float32
            ) / (len(REASON_IDS) - 1)
        specialty_history[0, 8] = float(previous.fallback)
        specialty_history[0, 9] = float(previous.execution_comparator_attempt)
        specialty_history[0, 10] = previous.fallback_reason_code / (len(REASON_IDS) - 1)
        resource_history[:, 0] = 1.0
        resource_history[:, 1] = np.asarray(
            [
                0.0 if value is None else np.clip(np.log1p(value), 0.0, 10.0)
                for value in previous.realized_resource_receipts
            ],
            np.float32,
        )
        resource_history[:, 2] = np.asarray(previous.resource_receipt_known, np.float32)
        resource_history[:, 3] = np.asarray(previous.resource_receipt_reason_codes, np.float32) / (
            len(REASON_IDS) - 1
        )
    specialty_retry = np.zeros((SPECIALTY_ROWS, SPECIALTY_RETRY_WIDTH), np.float32)
    activity_retry = np.zeros((ROLE_ROWS * ACTIVITY_ROWS, ACTIVITY_RETRY_WIDTH), np.float32)
    if rejected_attempt is not None:
        if (
            rejected_attempt.camp_id != catalog.camp_id
            or rejected_attempt.catalog_digest != catalog.digest
            or rejected_attempt.valid
            or rejected_attempt.attempt != 0
            or rejected_attempt.remaining_proposals != 1
        ):
            raise ValueError(
                "immediate retry context must be the rejected first attempt for this catalog"
            )
        population = max(1, catalog.population)
        capacity = max(1e-12, catalog.population * catalog.turn_years)
        specialty_retry[:, 0] = 1.0
        specialty_retry[:, 1] = rejected_attempt.requested_specialty_allocation
        specialty_retry[:, 2] = (
            np.asarray(rejected_attempt.requested_specialty_residents, np.float32) / population
        )
        specialty_retry[:, 3] = (
            np.asarray(rejected_attempt.accepted_specialty_residents, np.float32) / population
        )
        for violation in rejected_attempt.violations:
            for row in violation.specialty_rows:
                specialty_retry[row, 4] = min(1.0, specialty_retry[row, 4] + 1.0)
                specialty_retry[row, 5] = max(specialty_retry[row, 5], violation.severity)
                specialty_retry[row, 6] = float(violation.avoidable is not None)
                specialty_retry[row, 7] = float(violation.avoidable is True)
                specialty_retry[row, 8] = violation.reason_code / (len(REASON_IDS) - 1)
            for flat in violation.activity_rows:
                activity_retry[flat, 5] = min(1.0, activity_retry[flat, 5] + 1.0)
                activity_retry[flat, 6] = max(activity_retry[flat, 6], violation.severity)
                activity_retry[flat, 7] = float(violation.avoidable is not None)
                activity_retry[flat, 8] = float(violation.avoidable is True)
                activity_retry[flat, 9] = violation.reason_code / (len(REASON_IDS) - 1)
        for role in range(ROLE_ROWS):
            start = role * ACTIVITY_ROWS
            stop = start + ACTIVITY_ROWS
            block = activity_retry[start:stop]
            requested = np.asarray(
                rejected_attempt.requested_activity_worker_years[role], np.float32
            )
            planned = np.asarray(rejected_attempt.planned_activity_worker_years[role], np.float32)
            block[:, 0] = 1.0
            block[:, 1] = rejected_attempt.requested_activity_allocation[role]
            block[:, 2] = requested / capacity
            block[:, 3] = planned / capacity
            block[:, 4] = (planned - requested) / capacity
    return {
        "labor_specialty_features": np.concatenate(
            [specialty, specialty_history, specialty_retry], axis=1
        ),
        "labor_specialty_present": np.ones(SPECIALTY_ROWS, np.bool_),
        "labor_activity_features": np.concatenate(
            [activity, activity_history, activity_retry], axis=1
        ),
        "labor_activity_present": np.ones(ROLE_ROWS * ACTIVITY_ROWS, np.bool_),
        "labor_resource_features": np.concatenate([resources, resource_history], axis=1),
        "labor_resource_present": np.ones(RESOURCE_ROWS, np.bool_),
        "labor_specialty_mask": np.asarray(catalog.masks.specialty, np.bool_),
        "labor_activity_mask": np.asarray(catalog.masks.activity, np.bool_),
    }


def apportion_largest_remainder(shares: Sequence[float], population: int) -> tuple[int, ...]:
    """Native-compatible stable largest-remainder apportionment."""
    count = _integer(population, "population")
    mask = tuple(True for _ in shares)
    simplex = _simplex(shares, mask, "apportionment shares")
    total = math.fsum(simplex)
    exact = tuple(item / total * count for item in simplex)
    result = [math.floor(item) for item in exact]
    remainder = count - sum(result)
    order = sorted(range(len(result)), key=lambda index: (-(exact[index] - result[index]), index))
    for index in order[:remainder]:
        result[index] += 1
    return tuple(result)


def _camp_payloads(value: Any, label: str) -> dict[int, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a camp-keyed object")
    result: dict[int, Mapping[str, Any]] = {}
    for raw_camp, payload in value.items():
        if (
            not isinstance(raw_camp, str)
            or not raw_camp.isdigit()
            or str(int(raw_camp)) != raw_camp
        ):
            raise ValueError(f"{label} camp keys must be canonical nonnegative integers")
        camp = int(raw_camp)
        if camp in result or not isinstance(payload, Mapping):
            raise ValueError(f"{label} contains a duplicate or malformed camp row")
        result[camp] = payload
    return result


class LaborBridgeRuntime:
    """Typed adapter for the native contract-13 validation/step operations.

    ``bridge`` is the existing JSON-lines client and must expose ``request``.
    This adapter owns only contract-13 parsing and catalog identity. It never
    routes labor actions through the legacy ``step`` operation.
    """

    def __init__(self, bridge: Any, catalogs: Mapping[int, LaborCatalog]) -> None:
        if not callable(getattr(bridge, "request", None)):
            raise ValueError("labor runtime requires the JSON-lines bridge request boundary")
        self.bridge = bridge
        self.catalogs = self._catalog_map(catalogs)

    @staticmethod
    def _catalog_map(catalogs: Mapping[int, LaborCatalog]) -> dict[int, LaborCatalog]:
        if not isinstance(catalogs, Mapping) or not catalogs:
            raise ValueError("labor runtime requires a nonempty camp catalog map")
        result: dict[int, LaborCatalog] = {}
        for camp, catalog in catalogs.items():
            if (
                isinstance(camp, bool)
                or not isinstance(camp, int)
                or camp < 0
                or not isinstance(catalog, LaborCatalog)
                or catalog.camp_id != camp
            ):
                raise ValueError("labor runtime catalog keys differ from typed camp identities")
            result[camp] = catalog
        return result

    def _digests(self, camps: Sequence[int] | None = None) -> dict[str, str]:
        selected = sorted(self.catalogs) if camps is None else sorted(camps)
        return {str(camp): self.catalogs[camp].digest for camp in selected}

    def validate_labor(
        self, actions: Mapping[int, Mapping[str, Any]], *, attempt: int
    ) -> dict[int, LaborAttempt]:
        if isinstance(attempt, bool) or attempt not in (0, 1):
            raise ValueError("labor validation attempt must be zero or one")
        if (
            not isinstance(actions, Mapping)
            or not actions
            or any(
                isinstance(camp, bool)
                or camp not in self.catalogs
                or not isinstance(action, Mapping)
                for camp, action in actions.items()
            )
        ):
            raise ValueError("labor validation actions are not bound to current catalogs")
        response = self.bridge.request(
            {
                "op": "validate_labor",
                "attempt": attempt,
                "actions": {str(camp): dict(action) for camp, action in sorted(actions.items())},
                "catalogDigests": self._digests(tuple(actions)),
            }
        )
        response = _fields(response, {"ok", "op", "attempts"}, "validate_labor response")
        if response["ok"] is not True or response["op"] != "validate_labor":
            raise ValueError("native labor validation returned the wrong operation")
        raw = _camp_payloads(response["attempts"], "validate_labor attempts")
        if set(raw) != set(actions):
            raise ValueError("native labor validation returned a different camp set")
        parsed = {
            camp: LaborAttempt.from_payload(payload, self.catalogs[camp])
            for camp, payload in raw.items()
        }
        if any(row.attempt != attempt for row in parsed.values()):
            raise ValueError("native labor validation returned the wrong attempt index")
        return parsed

    def step_labor(
        self,
        initial_actions: Mapping[int, Mapping[str, Any]],
        retry_actions: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> tuple[Mapping[str, Any], dict[int, LaborFeedback]]:
        if not isinstance(initial_actions, Mapping) or set(initial_actions) != set(self.catalogs):
            raise ValueError("initial labor actions must cover exactly the current catalog camps")
        retries = {} if retry_actions is None else retry_actions
        if (
            not isinstance(retries, Mapping)
            or not set(retries).issubset(self.catalogs)
            or any(not isinstance(row, Mapping) for row in retries.values())
        ):
            raise ValueError("labor retry actions are malformed or belong to another camp")
        response = self.bridge.request(
            {
                "op": "step_labor",
                "actions": {
                    str(camp): dict(action) for camp, action in sorted(initial_actions.items())
                },
                "retryActions": {
                    str(camp): dict(action) for camp, action in sorted(retries.items())
                },
                "catalogDigests": self._digests(),
            }
        )
        response = _fields(response, {"ok", "op", "step", "laborFeedback"}, "step_labor response")
        if response["ok"] is not True or response["op"] != "step_labor":
            raise ValueError("native labor step returned the wrong operation")
        step = response["step"]
        if not isinstance(step, Mapping) or not isinstance(step.get("villages"), list):
            raise ValueError("native labor step lacks an ordinary encoded step")
        raw_feedback = _camp_payloads(response["laborFeedback"], "labor feedback")
        if not set(initial_actions).issubset(raw_feedback):
            raise ValueError("native labor step feedback differs from the submitted camp set")
        raw_feedback = {camp: raw_feedback[camp] for camp in initial_actions}
        feedback = {
            camp: LaborFeedback.from_payload(payload, self.catalogs[camp])
            for camp, payload in raw_feedback.items()
        }
        successor: dict[int, LaborCatalog] = {}
        for village in step["villages"]:
            if not isinstance(village, Mapping):
                raise ValueError("native labor successor village is malformed")
            if village.get("terminated") is True and village.get("observation") is None:
                continue
            if village.get("terminated") is True or not isinstance(
                village.get("observation"), Mapping
            ):
                raise ValueError("native labor successor village is malformed")
            catalog, _ = labor_observation_fields(village["observation"])
            if catalog.camp_id in successor:
                raise ValueError("native labor successor repeats a camp catalog")
            successor[catalog.camp_id] = catalog
        self.catalogs = successor
        # Keep the existing bridge client's physical/economic sidecars and
        # alignment cursor current when this adapter wraps Model9Bridge.
        if callable(getattr(self.bridge, "_read_physical_diagnostics", None)):
            from .diagnostics import validate

            self.bridge.decision_diagnostics = validate(
                step,
                self.bridge._turn,
                initial_actions,
                required=self.bridge.schema.decision_diagnostics_schema >= 1,
                trajectory=self.bridge._trajectory,
            )
            if self.bridge.decision_diagnostics:
                self.bridge._trajectory = self.bridge.decision_diagnostics[0]["trajectoryId"]
            self.bridge._read_physical_diagnostics(step)
            self.bridge._turn = int(step["turn"])
            self.bridge._last_step = step
        return step, feedback


class LaborModel9Bridge:
    """Opt-in process client for the actor-labor sidecar contract.

    The native process deliberately retains its legacy top-level schema for
    compatibility. This wrapper composes the existing client, validates the
    exact ``actorLabor`` schema sidecar, and injects ``actorLabor:true`` only
    into its own resets. Ordinary :class:`bridge.Model9Bridge` is unchanged.
    """

    def __init__(
        self, command: Sequence[str] | None = None, *, economic_diagnostics: bool = False
    ) -> None:
        from .bridge import Model9Bridge

        self._bridge = Model9Bridge(command, economic_diagnostics=economic_diagnostics)
        schema = self._bridge.request({"op": "schema"})
        if not isinstance(schema, Mapping) or "actorLabor" not in schema:
            self._bridge.close()
            raise ValueError("native bridge does not advertise the actor-labor sidecar")
        self._producer_contract = LaborProducerContract.from_state(schema["actorLabor"])
        self.labor_runtime: LaborBridgeRuntime | None = None
        self._labor_fork_catalogs: dict[int, dict[int, LaborCatalog]] = {}

    @property
    def schema(self) -> Any:
        return self._bridge.schema

    @property
    def last_reset_identity(self) -> dict[str, Any]:
        """Return a defensive copy of the native reset identity."""
        identity = getattr(self._bridge, "last_reset_identity", None)
        if not isinstance(identity, Mapping):
            raise ValueError("actor-labor bridge has no completed reset identity")
        return copy.deepcopy(dict(identity))

    def runtime_identity(self) -> dict[str, Any]:
        """Return the reset and reviewed labor-producer identities read-only."""
        return {
            "resetIdentity": self.last_reset_identity,
            "actorLabor": self._producer_contract.state_dict(),
        }

    def reset(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        original_request = self._bridge.request

        def actor_labor_request(payload: Mapping[str, Any]) -> dict[str, Any]:
            request = dict(payload)
            if request.get("op") == "reset":
                if "actorLabor" in request:
                    raise ValueError("actorLabor reset flag is owned by LaborModel9Bridge")
                request["actorLabor"] = True
            response = original_request(request)
            if request.get("op") == "reset" and response.get("actorLaborEnabled") is not True:
                raise ValueError("native reset did not enable the actor-labor contract")
            return response

        self._bridge.request = actor_labor_request
        try:
            step = self._bridge.reset(*args, **kwargs)
        finally:
            self._bridge.request = original_request
        catalogs: dict[int, LaborCatalog] = {}
        for village in step.get("villages", []):
            if not isinstance(village, Mapping) or not isinstance(
                village.get("observation"), Mapping
            ):
                raise ValueError("actor-labor reset returned a malformed village")
            catalog, previous = labor_observation_fields(village["observation"])
            if previous is not None:
                raise ValueError("actor-labor reset must not carry post-execution feedback")
            if catalog.camp_id in catalogs:
                raise ValueError("actor-labor reset repeats a camp catalog")
            catalogs[catalog.camp_id] = catalog
        if not catalogs:
            raise ValueError("actor-labor reset returned no decision catalogs")
        self.labor_runtime = LaborBridgeRuntime(self._bridge, catalogs)
        self._labor_fork_catalogs.clear()
        return step

    def validate_labor(
        self, actions: Mapping[int, Mapping[str, Any]], *, attempt: int
    ) -> dict[int, LaborAttempt]:
        if self.labor_runtime is None:
            raise ValueError("actor-labor bridge must reset before validation")
        return self.labor_runtime.validate_labor(actions, attempt=attempt)

    def step_labor(
        self,
        initial_actions: Mapping[int, Mapping[str, Any]],
        retry_actions: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> tuple[Mapping[str, Any], dict[int, LaborFeedback]]:
        if self.labor_runtime is None:
            raise ValueError("actor-labor bridge must reset before stepping")
        return self.labor_runtime.step_labor(initial_actions, retry_actions)

    @staticmethod
    def _step_catalogs(step: Mapping[str, Any]) -> dict[int, LaborCatalog]:
        if not isinstance(step, Mapping) or not isinstance(step.get("villages"), list):
            raise ValueError("actor-labor fork/restore returned a malformed step")
        catalogs: dict[int, LaborCatalog] = {}
        for village in step["villages"]:
            if not isinstance(village, Mapping):
                raise ValueError("actor-labor fork/restore returned a malformed village")
            if village.get("terminated") is True and village.get("observation") is None:
                continue
            if village.get("terminated") is True or not isinstance(
                village.get("observation"), Mapping
            ):
                raise ValueError("actor-labor fork/restore returned a malformed village")
            catalog, _ = labor_observation_fields(village["observation"])
            if catalog.camp_id in catalogs:
                raise ValueError("actor-labor fork/restore repeats a camp catalog")
            catalogs[catalog.camp_id] = catalog
        return catalogs

    def fork(self) -> tuple[int, Mapping[str, Any]]:
        if self.labor_runtime is None:
            raise ValueError("actor-labor bridge must reset before forking")
        fork, step = self._bridge.fork()
        observed = self._step_catalogs(step)
        current = self.labor_runtime.catalogs
        if {camp: row.digest for camp, row in observed.items()} != {
            camp: row.digest for camp, row in current.items()
        }:
            raise ValueError("actor-labor fork changed the current catalog identities")
        self._labor_fork_catalogs[fork] = dict(current)
        return fork, step

    def restore(self, fork: int) -> Mapping[str, Any]:
        if fork not in self._labor_fork_catalogs:
            raise ValueError("actor-labor restore refers to an unknown fork")
        step = self._bridge.restore(fork)
        observed = self._step_catalogs(step)
        expected = self._labor_fork_catalogs[fork]
        if {camp: row.digest for camp, row in observed.items()} != {
            camp: row.digest for camp, row in expected.items()
        }:
            raise ValueError("actor-labor restore changed the saved catalog identities")
        self.labor_runtime = LaborBridgeRuntime(self._bridge, observed)
        return step

    def drop(self, fork: int) -> None:
        if fork not in self._labor_fork_catalogs:
            raise ValueError("actor-labor drop refers to an unknown fork")
        self._bridge.drop(fork)
        self._labor_fork_catalogs.pop(fork)

    def close(self) -> None:
        self._bridge.close()

    def __enter__(self) -> LaborModel9Bridge:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
