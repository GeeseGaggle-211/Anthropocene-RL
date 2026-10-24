from __future__ import annotations

import copy

import numpy as np
import pytest

from anthropocene_rl.autonomous_v7.labor_bridge import (
    ACTIVITIES,
    ACTIVITY_FEATURES,
    ACTIVITY_ROWS,
    GOODS_ROWS,
    LABOR_ACTIVITY_FEATURE_WIDTH,
    LABOR_REGISTRY_DIGEST,
    LABOR_RESOURCE_FEATURE_WIDTH,
    LABOR_SPECIALTY_FEATURE_WIDTH,
    REASON_IDS,
    REASON_REGISTRY_DIGEST,
    RESOURCE_IDS,
    RESOURCE_ROWS,
    ROLE_ROWS,
    SPECIALTIES,
    SPECIALTY_ROWS,
    LaborAttempt,
    LaborBridgeRuntime,
    LaborCatalog,
    LaborFeedback,
    LaborMasks,
    LaborModel9Bridge,
    LaborPolicySample,
    LaborProducerContract,
    apportion_largest_remainder,
    labor_catalog_digest,
    labor_catalog_tensors,
    labor_observation_batch,
    labor_observation_tensors,
)


def _masks() -> LaborMasks:
    specialty = [True] * SPECIALTY_ROWS
    activity = [[True] + [False] * (ACTIVITY_ROWS - 1) for _ in range(ROLE_ROWS)]
    activity[0][1:5] = [True] * 4
    activity[1][5:7] = [True] * 2
    activity[2][7:9] = [True] * 2
    activity[3][9:11] = [True] * 2
    for role in range(4, 10):
        activity[role][11:] = [True] * 11
    activity[10][4] = True
    activity[11][2] = True
    activity[12][3] = True
    return LaborMasks.from_values(specialty, activity)


def _payload(masks: LaborMasks) -> dict:
    specialty = [0.0] * SPECIALTY_ROWS
    specialty[0] = 0.25
    specialty[1] = 0.75
    activities = []
    for row in masks.activity:
        values = [0.0] * ACTIVITY_ROWS
        legal = [index for index, enabled in enumerate(row) if enabled]
        for index in legal:
            values[index] = 1.0 / len(legal)
        activities.append(values)
    return {"laborSpecialtyAllocation": specialty, "laborActivityAllocation": activities}


def _catalog_payload() -> dict:
    masks = _masks()
    specialty_rows = []
    for index, name in enumerate(SPECIALTIES):
        capabilities = [0.0] * 8
        if index == 0:
            capabilities[0] = 1.0
        elif index == 1:
            capabilities[1] = 1.0
        elif index in (11, 12, 13):
            capabilities[2] = 1.0
        elif index == 2:
            capabilities[3] = 1.0
        elif index == 3:
            capabilities[4] = 1.0
        elif index == 4:
            capabilities[5] = 1.0
        elif 5 <= index <= 10:
            capabilities[6] = 1.0
        capabilities[7] = 1.0
        specialty_rows.append(
            {
                "row": index,
                "id": name,
                "requiredTechnology": None,
                "unavailableReason": None,
                "present": True,
                "known": True,
                "actorControllable": masks.specialty[index],
                "wholeResidents": True,
                "features": [1.0, 1.0, float(masks.specialty[index]), 1.0, *capabilities],
            }
        )
    activity_rows = []
    for role in range(ROLE_ROWS):
        for activity, activity_id in enumerate(ACTIVITIES):
            flat = role * ACTIVITY_ROWS + activity
            actor = masks.activity[role][activity]
            is_idle = activity == 0
            is_food = activity == 1
            is_raw = 2 <= activity <= 4
            is_field = activity in (5, 6)
            is_herd = activity in (7, 8)
            is_fishing = activity in (9, 10)
            is_craft = activity >= 11
            operation = activity in (1, 2, 3, 4, 5, 7, 9) or is_craft
            development = activity in (6, 8, 10)
            good = activity - 11 if is_craft else None
            features = [
                1.0,
                1.0,
                float(actor),
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0 if good is None else float(good + 1),
                0.0,
                1.0,
                20.0,
                float(not is_idle),
                float(not is_idle),
                float(not is_idle),
                *([0.0] * 10),
                0.0,
                float(is_idle),
                float(is_food),
                float(is_raw),
                float(is_field),
                float(is_herd),
                float(is_fishing),
                float(is_craft),
                float(1 if is_craft else 0),
                float(operation),
                float(development),
                float(activity == 2),
                float(activity == 3),
                float(activity == 4),
                *[float(good == item) for item in range(GOODS_ROWS)],
            ]
            assert len(features) == len(ACTIVITY_FEATURES)
            activity_rows.append(
                {
                    "row": flat,
                    "specialtyRow": role,
                    "activityRow": activity,
                    "specialtyId": SPECIALTIES[role + 1],
                    "activityId": activity_id,
                    "phase": "work-before-production",
                    "unit": "worker-years",
                    "outputGood": None if good is None else activity_id.removeprefix("goods."),
                    "outputUnit": "good-units" if is_craft else None,
                    "unavailableReason": None,
                    "present": True,
                    "known": True,
                    "actorControllable": actor,
                    "fractionalWorkerTime": True,
                    "nonlinear": False,
                    "visibleCompetitionPossible": False,
                    "maxWorkerYearsKnown": True,
                    "outputCapacityKnown": not is_idle,
                    "specialistOnly": False,
                    "isOperationOrHarvest": operation,
                    "isDevelopmentOrExpansion": development,
                    "outputsStone": activity == 2,
                    "outputsClay": activity == 3,
                    "outputsWood": activity == 4,
                    "recipeScheduleOrdinal": -1,
                    "maxWorkerYears": 20.0,
                    "capacityPerWorkerYearLower": float(not is_idle),
                    "capacityPerWorkerYearUpper": float(not is_idle),
                    "stonePerOutput": 0.0,
                    "clayPerOutput": 0.0,
                    "woodPerOutput": 0.0,
                    "plantFoodPerOutput": 0.0,
                    "animalFoodPerOutput": 0.0,
                    "animalFabricPerOutput": 0.0,
                    "housingMaterialPerOutput": 0.0,
                    "clothPerOutput": 0.0,
                    "constructionMaterialPerOutput": 0.0,
                    "toolServicePerOutput": 0.0,
                    "features": features,
                }
            )
    resources = [
        {
            "row": index,
            "resourceId": name,
            "unit": "worker-years",
            "phase": "opening",
            "reason": None,
            "present": True,
            "known": True,
            "competitionPossible": False,
            "opening": 0.0,
            "locallyAccessible": 0.0,
            "fixedReserved": 0.0,
            "features": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        }
        for index, name in enumerate(RESOURCE_IDS)
    ]
    payload = {
        "schemaVersion": 1,
        "registryVersion": 1,
        "registryDigest": LABOR_REGISTRY_DIGEST,
        "reasonRegistryVersion": 1,
        "reasonIds": list(REASON_IDS),
        "reasonRegistryDigest": REASON_REGISTRY_DIGEST,
        "campId": 7,
        "turn": 3,
        "population": 10,
        "turnYears": 2.0,
        "specialtyIds": list(SPECIALTIES),
        "activityIds": list(ACTIVITIES),
        "specialtyMask": list(masks.specialty),
        "activityMask": [list(row) for row in masks.activity],
        "specialtyRows": specialty_rows,
        "activityRows": activity_rows,
        "resources": resources,
        "digest": "0" * 64,
    }
    payload["digest"] = labor_catalog_digest(payload)
    return payload


def _attempt_payload(catalog: dict, *, rejected: bool = False) -> dict:
    specialty = [0.0] * SPECIALTY_ROWS
    specialty[1] = 1.0
    residents = [0.0] * SPECIALTY_ROWS
    residents[1] = 10.0
    accepted = [0] * SPECIALTY_ROWS
    accepted[1] = 10
    activity, requested, planned = [], [], []
    for role in range(ROLE_ROWS):
        shares = [0.0] * ACTIVITY_ROWS
        shares[1 if role == 0 else 0] = 1.0
        work = [0.0] * ACTIVITY_ROWS
        work[1 if role == 0 else 0] = 20.0 if role == 0 else 0.0
        activity.append(shares)
        requested.append(work)
        planned.append(list(work))
    violations = (
        []
        if not rejected
        else [
            {
                "reasonCode": 3,
                "reason": REASON_IDS[3],
                "knowledge": "opening-known",
                "unit": "residents",
                "phase": "validation",
                "specialtyRows": [1],
                "activityRows": [1],
                "requested": 11.0,
                "requestedKnown": True,
                "available": 10.0,
                "availableKnown": True,
                "severity": 0.5,
                "avoidable": True,
            }
        ]
    )
    return {
        "schemaVersion": 1,
        "campId": 7,
        "attempt": 0,
        "remainingProposals": 1 if rejected else 0,
        "valid": not rejected,
        "fallback": False,
        "requestedSpecialtyAllocation": specialty,
        "requestedSpecialtyResidents": residents,
        "acceptedSpecialtyResidents": accepted,
        "requestedActivityAllocation": activity,
        "requestedActivityWorkerYears": requested,
        "plannedActivityWorkerYears": planned,
        "violations": violations,
        "catalogDigest": catalog["digest"],
        "feasibilityCost": 0.5 if rejected else 0.0,
    }


def _feedback_payload(catalog: dict) -> dict:
    attempt = _attempt_payload(catalog)
    work = attempt["plannedActivityWorkerYears"]
    actual = copy.deepcopy(work)
    known = [[True] * ACTIVITY_ROWS for _ in range(ROLE_ROWS)]
    reasons = [[""] * ACTIVITY_ROWS for _ in range(ROLE_ROWS)]
    codes = [[0] * ACTIVITY_ROWS for _ in range(ROLE_ROWS)]
    return {
        "schemaVersion": 1,
        "campId": 7,
        "decisionTurn": 3,
        "executionComparatorAttempt": 0,
        "parentDecisionId": "decision-7-3",
        "foundingParentDecisionId": None,
        "decisionCatalog": copy.deepcopy(catalog),
        "initialRequest": copy.deepcopy(attempt),
        "finalSampledRequest": copy.deepcopy(attempt),
        "committedActivityWorkerYears": work,
        "actualSchedulerConsumedWorkerYears": actual,
        "actualWorkerYearsKnown": known,
        "actualWorkerYearsReasons": reasons,
        "actualWorkerYearsReasonCodes": codes,
        "plannedGoodsOutput": [0.0] * GOODS_ROWS,
        "plannedGoodsOutputKnown": [True] * GOODS_ROWS,
        "plannedGoodsOutputReasons": [""] * GOODS_ROWS,
        "plannedGoodsOutputReasonCodes": [0] * GOODS_ROWS,
        "realizedGoodsOutput": [0.0] * GOODS_ROWS,
        "goodsOutputKnown": [True] * GOODS_ROWS,
        "goodsOutputReasons": [""] * GOODS_ROWS,
        "goodsOutputReasonCodes": [0] * GOODS_ROWS,
        "realizedResourceReceipts": [0.0] * RESOURCE_ROWS,
        "resourceReceiptKnown": [True] * RESOURCE_ROWS,
        "resourceReceiptReasons": [""] * RESOURCE_ROWS,
        "resourceReceiptReasonCodes": [0] * RESOURCE_ROWS,
        "fallback": False,
        "fallbackReason": "",
        "fallbackReasonCode": 0,
    }


def test_reviewed_specialty_and_activity_registries_are_stable() -> None:
    assert SPECIALTIES == (
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
    assert ACTIVITIES[:10] == (
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
    )
    assert ACTIVITIES[10] == "fishing.development"
    assert len(ACTIVITIES) == 22
    assert (
        LABOR_REGISTRY_DIGEST == "83012ab8f9bc043c9ce261484c1ec8b63d825715d451a8afd4499783c0b351c7"
    )
    assert (
        REASON_REGISTRY_DIGEST == "d9f4fda6aed5a6599b599023c80ecf5507a1109b55a8945e8c92cccace0e1cb4"
    )
    LaborProducerContract.from_state(LaborProducerContract().state_dict())


@pytest.mark.parametrize(
    "key,value",
    [
        ("nativeContract", 12),
        ("specialties", ["idle"]),
        ("populationAuthority", "native-plan-residual"),
        ("activitySemantics", "output-promise"),
    ],
)
def test_producer_contract_rejects_version_registry_and_semantic_drift(key: str, value) -> None:
    state = LaborProducerContract().state_dict()
    state[key] = value
    with pytest.raises(ValueError):
        LaborProducerContract.from_state(state)


def test_action_round_trip_and_largest_remainder_are_exact() -> None:
    masks = _masks()
    sample = LaborPolicySample.from_payload(_payload(masks), masks)
    assert LaborPolicySample.from_payload(sample.payload(), masks) == sample
    # Stable ordinal breaks equal remainders; no prior/native assignment is reserved.
    assert apportion_largest_remainder([0.25, 0.25, 0.25, 0.25], 3) == (1, 1, 1, 0)
    assert apportion_largest_remainder([0.250000001, 0.25, 0.25, 0.25], 3) == (1, 1, 1, 0)
    assert sum(apportion_largest_remainder(sample.specialty_allocation, 17)) == 17


def test_float32_softmax_payload_is_canonical_for_native_tolerance() -> None:
    import torch

    masks = _masks()
    for seed in range(50):
        generator = torch.Generator().manual_seed(seed)
        specialty = torch.softmax(torch.randn(SPECIALTY_ROWS, generator=generator), dim=0).tolist()
        activity = []
        for mask in masks.activity:
            logits = torch.randn(ACTIVITY_ROWS, generator=generator).masked_fill(
                ~torch.tensor(mask), -torch.inf
            )
            activity.append(tuple(torch.softmax(logits, dim=0).tolist()))
        sample = LaborPolicySample(tuple(specialty), tuple(activity))
        parsed = LaborPolicySample.from_payload(sample.payload(), masks)
        assert sum(parsed.specialty_allocation) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["laborSpecialtyAllocation"].__setitem__(0, True),
        lambda p: p["laborSpecialtyAllocation"].__setitem__(0, float("nan")),
        lambda p: p["laborSpecialtyAllocation"].__setitem__(2, 0.1),
        lambda p: p["laborActivityAllocation"][0].__setitem__(20, 0.1),
        lambda p: p["laborActivityAllocation"].pop(),
        lambda p: p.update(extra=[]),
    ],
)
def test_action_rejects_coercion_mask_and_shape_attacks(mutation) -> None:
    masks = _masks()
    payload = copy.deepcopy(_payload(masks))
    mutation(payload)
    with pytest.raises(ValueError):
        LaborPolicySample.from_payload(payload, masks)


def test_rejected_attempt_preserves_invalid_finite_request_without_rewriting_it() -> None:
    raw = _catalog_payload()
    catalog = LaborCatalog.from_payload(raw)
    rejected = _attempt_payload(raw, rejected=True)
    rejected["requestedSpecialtyAllocation"][1] = 0.5
    rejected["requestedSpecialtyResidents"][1] = 5.0
    parsed = LaborAttempt.from_payload(rejected, catalog)
    assert not parsed.valid and sum(parsed.requested_specialty_allocation) == 0.5
    accepted = copy.deepcopy(rejected)
    accepted["valid"] = True
    accepted["remainingProposals"] = 0
    accepted["violations"] = []
    accepted["feasibilityCost"] = 0.0
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        LaborAttempt.from_payload(accepted, catalog)


def test_violation_known_flags_preserve_unavailable_semantics() -> None:
    raw = _catalog_payload()
    catalog = LaborCatalog.from_payload(raw)
    attempt = _attempt_payload(raw, rejected=True)
    violation = attempt["violations"][0]
    violation.update(requested=0.0, requestedKnown=False, available=0.0, availableKnown=False)
    parsed = LaborAttempt.from_payload(attempt, catalog)
    assert parsed.violations[0].requested is None
    assert parsed.violations[0].available is None

    violation["requestedKnown"] = 0
    with pytest.raises(ValueError, match="known flags"):
        LaborAttempt.from_payload(attempt, catalog)


def test_valid_attempt_separates_fractional_requested_from_integer_planned_work() -> None:
    raw = _catalog_payload()
    raw["population"] = 3
    raw["digest"] = labor_catalog_digest(raw)
    catalog = LaborCatalog.from_payload(raw)
    attempt = _attempt_payload(raw)
    attempt["requestedSpecialtyAllocation"] = [0.0] * SPECIALTY_ROWS
    attempt["requestedSpecialtyAllocation"][0] = 0.7
    attempt["requestedSpecialtyAllocation"][1] = 0.3
    attempt["requestedSpecialtyResidents"] = [0.0] * SPECIALTY_ROWS
    attempt["requestedSpecialtyResidents"][0] = 2.1
    attempt["requestedSpecialtyResidents"][1] = 0.9
    attempt["acceptedSpecialtyResidents"] = [0] * SPECIALTY_ROWS
    attempt["acceptedSpecialtyResidents"][0] = 2
    attempt["acceptedSpecialtyResidents"][1] = 1
    attempt["requestedActivityWorkerYears"][0] = [0.0] * ACTIVITY_ROWS
    attempt["requestedActivityWorkerYears"][0][1] = 1.8
    attempt["plannedActivityWorkerYears"][0] = [0.0] * ACTIVITY_ROWS
    attempt["plannedActivityWorkerYears"][0][1] = 2.0
    attempt["catalogDigest"] = raw["digest"]
    parsed = LaborAttempt.from_payload(attempt, catalog)
    assert parsed.requested_activity_worker_years[0][1] == 1.8
    assert parsed.planned_activity_worker_years[0][1] == 2.0


def test_masks_require_idle_and_do_not_advertise_masked_specialties() -> None:
    specialty = [True] * SPECIALTY_ROWS
    activity = [[True] + [False] * (ACTIVITY_ROWS - 1) for _ in range(ROLE_ROWS)]
    specialty[3] = False
    activity[2][7] = True
    with pytest.raises(ValueError, match="masked specialty"):
        LaborMasks.from_values(specialty, activity)
    activity[2][7] = False
    LaborMasks.from_values(specialty, activity)
    activity[2][0] = False
    with pytest.raises(ValueError, match="idle activity"):
        LaborMasks.from_values(specialty, activity)


def test_catalog_feedback_and_retry_have_distinct_bounded_tensor_surfaces() -> None:
    raw = _catalog_payload()
    catalog = LaborCatalog.from_payload(raw)
    assert catalog.digest == labor_catalog_digest(raw)
    empty = labor_catalog_tensors(catalog)
    assert empty["labor_specialty_features"].shape == (
        SPECIALTY_ROWS,
        LABOR_SPECIALTY_FEATURE_WIDTH,
    )
    assert empty["labor_activity_features"].shape == (
        ROLE_ROWS * ACTIVITY_ROWS,
        LABOR_ACTIVITY_FEATURE_WIDTH,
    )
    assert empty["labor_resource_features"].shape == (RESOURCE_ROWS, LABOR_RESOURCE_FEATURE_WIDTH)
    feedback = LaborFeedback.from_payload(_feedback_payload(raw), catalog)
    history = labor_catalog_tensors(catalog, feedback)
    rejected = LaborAttempt.from_payload(_attempt_payload(raw, rejected=True), catalog)
    retry = labor_catalog_tensors(catalog, feedback, rejected)
    assert np.count_nonzero(empty["labor_specialty_features"] - history["labor_specialty_features"])
    assert np.count_nonzero(history["labor_activity_features"] - retry["labor_activity_features"])
    # Retry context cannot contain successor goods/resource receipts.
    retry_only = labor_catalog_tensors(catalog, None, rejected)
    assert retry_only["labor_resource_features"][:, -4:].sum() == 0.0
    assert np.isfinite(retry["labor_activity_features"]).all()


def test_activity_feature_normalization_uses_named_fields_without_mutating_flags() -> None:
    catalog = LaborCatalog.from_payload(_catalog_payload())
    features = labor_catalog_tensors(catalog)["labor_activity_features"]
    index = {name: offset for offset, name in enumerate(ACTIVITY_FEATURES)}
    raw_wood = 4
    craft = 11
    assert features[raw_wood, index["outputsWood"]] == 1.0
    assert features[raw_wood, index["outputUnitCode"]] == 0.0
    assert features[craft, index["isCraft"]] == 1.0
    assert features[craft, index["outputUnitCode"]] == pytest.approx(0.25)


def test_fractional_recipe_schedule_ordinal_matches_native_catalog() -> None:
    payload = _catalog_payload()
    row = payload["activityRows"][4 * ACTIVITY_ROWS + 20]
    assert row["activityId"] == "goods.housing-material"
    row["recipeScheduleOrdinal"] = 3.5
    row["features"][9] = 4.5
    payload["digest"] = labor_catalog_digest(payload)
    catalog = LaborCatalog.from_payload(payload)
    assert catalog.activity_rows[4 * ACTIVITY_ROWS + 20]["recipeScheduleOrdinal"] == 3.5
    changed = copy.deepcopy(payload)
    changed["activityRows"][4 * ACTIVITY_ROWS + 20]["recipeScheduleOrdinal"] = -1.5
    changed["activityRows"][4 * ACTIVITY_ROWS + 20]["features"][9] = -0.5
    changed["digest"] = labor_catalog_digest(changed)
    with pytest.raises(ValueError, match="at least -1"):
        LaborCatalog.from_payload(changed)


def test_known_zero_and_unavailable_feedback_remain_distinct() -> None:
    raw = _catalog_payload()
    catalog = LaborCatalog.from_payload(raw)
    zero_payload = _feedback_payload(raw)
    zero = LaborFeedback.from_payload(zero_payload, catalog)
    unavailable_payload = copy.deepcopy(zero_payload)
    unavailable_payload["realizedGoodsOutput"][0] = None
    unavailable_payload["goodsOutputKnown"][0] = False
    unavailable_payload["goodsOutputReasons"][0] = "realized-output-unavailable"
    unavailable_payload["goodsOutputReasonCodes"][0] = 17
    unavailable = LaborFeedback.from_payload(unavailable_payload, catalog)
    zero_features = labor_catalog_tensors(catalog, zero)["labor_activity_features"]
    unavailable_features = labor_catalog_tensors(catalog, unavailable)["labor_activity_features"]
    hand_tools = 11
    assert zero_features[hand_tools, -15] == 1.0
    assert unavailable_features[hand_tools, -15] == 0.0
    forged = copy.deepcopy(unavailable_payload)
    forged["goodsOutputReasonCodes"][0] = 0
    with pytest.raises(ValueError, match="reason code"):
        LaborFeedback.from_payload(forged, catalog)


def test_raw_observation_retry_path_rejects_postexecution_leak_and_batches_exactly() -> None:
    catalog = _catalog_payload()
    observation = {"laborCatalog": catalog, "laborPrevious": None}
    retry = {"laborRetry": _attempt_payload(catalog, rejected=True)}
    parsed_catalog, previous, rejected, _tensors = labor_observation_tensors(observation, retry)
    assert previous is None and rejected is not None and not rejected.valid
    assert parsed_catalog.digest == catalog["digest"]
    contexts, batch = labor_observation_batch([observation, observation], [retry, None])
    assert len(contexts) == 2
    assert batch["labor_activity_features"].shape[0] == 2
    typed = LaborAttempt.from_payload(retry["laborRetry"], parsed_catalog)
    typed_contexts, typed_batch = labor_observation_batch(
        [observation, observation], rejected_attempts=[typed, None]
    )
    assert typed_contexts[0][2] is typed
    assert np.array_equal(typed_batch["labor_activity_features"], batch["labor_activity_features"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        labor_observation_batch([observation], [retry], rejected_attempts=[typed])
    leaked = copy.deepcopy(retry)
    leaked["laborRetry"]["realizedGoodsOutput"] = [0.0] * GOODS_ROWS
    with pytest.raises(ValueError, match="post-execution"):
        labor_observation_tensors(observation, leaked)


def test_catalog_content_and_registry_mutations_are_rejected() -> None:
    payload = _catalog_payload()
    for mutation in (
        lambda value: value["activityRows"][5]["features"].__setitem__(34, 0.0),
        lambda value: value["resources"][1]["features"].__setitem__(4, 2.0),
        lambda value: value.update(reasonRegistryDigest="0" * 64),
        lambda value: value["activityRows"].reverse(),
    ):
        changed = copy.deepcopy(payload)
        mutation(changed)
        with pytest.raises(ValueError):
            LaborCatalog.from_payload(changed)


def test_typed_runtime_adapter_binds_native_operations_catalogs_and_feedback() -> None:
    catalog_payload = _catalog_payload()
    catalog = LaborCatalog.from_payload(catalog_payload)
    feedback_payload = _feedback_payload(catalog_payload)

    class FakeBridge:
        def __init__(self) -> None:
            self.requests = []

        def request(self, payload):
            self.requests.append(copy.deepcopy(payload))
            if payload["op"] == "validate_labor":
                return {
                    "ok": True,
                    "op": "validate_labor",
                    "attempts": {"7": _attempt_payload(catalog_payload, rejected=True)},
                }
            return {
                "ok": True,
                "op": "step_labor",
                "step": {
                    "turn": 4,
                    "villages": [
                        {
                            "observation": {
                                "laborCatalog": catalog_payload,
                                "laborPrevious": feedback_payload,
                            }
                        }
                    ],
                },
                "laborFeedback": {"7": feedback_payload},
            }

    bridge = FakeBridge()
    runtime = LaborBridgeRuntime(bridge, {7: catalog})
    labor = _payload(catalog.masks)
    initial = {
        "schemaVersion": 12,
        "movement": -1,
        "fission": 0,
        "tradeValue": [1.0] * 7,
        "tradeGate": [0] * 7,
        "tradeShare": 0.0,
        **labor,
    }
    attempts = runtime.validate_labor({7: initial}, attempt=0)
    assert not attempts[7].valid
    assert bridge.requests[0] == {
        "op": "validate_labor",
        "attempt": 0,
        "actions": {"7": initial},
        "catalogDigests": {"7": catalog.digest},
    }
    step, feedback = runtime.step_labor({7: initial}, {7: labor})
    assert step["turn"] == 4 and feedback[7].decision_catalog.digest == catalog.digest
    assert bridge.requests[1]["retryActions"] == {"7": labor}
    forged = FakeBridge()
    bad_runtime = LaborBridgeRuntime(forged, {7: catalog})
    forged.request = lambda _payload: {
        "ok": True,
        "op": "validate_labor",
        "attempts": {"07": _attempt_payload(catalog_payload, rejected=True)},
    }
    with pytest.raises(ValueError, match="canonical"):
        bad_runtime.validate_labor({7: initial}, attempt=0)


def test_opt_in_process_wrapper_validates_sidecar_and_owns_reset_flag(monkeypatch) -> None:
    from anthropocene_rl.autonomous_v7 import bridge as bridge_module

    catalog_payload = _catalog_payload()

    class FakeModel9Bridge:
        def __init__(self, _command=None, *, economic_diagnostics=False) -> None:
            self.schema = {"legacy": True}
            self.requests = []
            self.closed = False
            self.dropped = []
            self.last_reset_identity = None

        def request(self, payload):
            self.requests.append(copy.deepcopy(payload))
            if payload["op"] == "schema":
                return {"ok": True, "actorLabor": LaborProducerContract().state_dict()}
            if payload["op"] == "reset":
                self.last_reset_identity = {
                    "configurationHash": "c" * 64,
                    "fingerprint": "f" * 64,
                    "buildHash": "b" * 64,
                    "rawMaterialsEnabled": False,
                    "sourceSeed": 1,
                }
                return {
                    "ok": True,
                    "actorLaborEnabled": True,
                    "step": {
                        "turn": 3,
                        "villages": [
                            {
                                "observation": {
                                    "laborCatalog": catalog_payload,
                                    "laborPrevious": None,
                                }
                            }
                        ],
                    },
                }
            raise AssertionError(payload)

        def reset(self, *_args, **_kwargs):
            return self.request({"op": "reset"})["step"]

        def close(self):
            self.closed = True

        def fork(self):
            return 5, {
                "turn": 3,
                "villages": [
                    {"observation": {"laborCatalog": catalog_payload, "laborPrevious": None}}
                ],
            }

        def restore(self, fork):
            assert fork == 5
            return {
                "turn": 3,
                "villages": [
                    {"observation": {"laborCatalog": catalog_payload, "laborPrevious": None}}
                ],
            }

        def drop(self, fork):
            self.dropped.append(fork)

    monkeypatch.setattr(bridge_module, "Model9Bridge", FakeModel9Bridge)
    client = LaborModel9Bridge()
    step = client.reset(seed=1)
    assert step["turn"] == 3
    assert client._bridge.requests[-1] == {"op": "reset", "actorLabor": True}
    assert client.labor_runtime is not None and set(client.labor_runtime.catalogs) == {7}
    assert client.last_reset_identity["sourceSeed"] == 1
    runtime_identity = client.runtime_identity()
    assert runtime_identity["resetIdentity"] == client.last_reset_identity
    assert runtime_identity["actorLabor"] == LaborProducerContract().state_dict()
    runtime_identity["resetIdentity"]["sourceSeed"] = 99
    assert client.last_reset_identity["sourceSeed"] == 1
    fork, _ = client.fork()
    client.labor_runtime.catalogs = {}
    client.restore(fork)
    assert set(client.labor_runtime.catalogs) == {7}
    client.drop(fork)
    assert client._bridge.dropped == [5]
    client.close()
    assert client._bridge.closed


def test_step_labor_and_step_catalogs_handle_terminated_villages() -> None:
    catalog_payload = _catalog_payload()
    catalog = LaborCatalog.from_payload(catalog_payload)
    feedback_payload = _feedback_payload(catalog_payload)

    class FakeBridge:
        def __init__(self, step_payload) -> None:
            self.step_payload = step_payload

        def request(self, payload):
            if payload["op"] == "step_labor":
                return {
                    "ok": True,
                    "op": "step_labor",
                    "step": self.step_payload,
                    "laborFeedback": {"7": feedback_payload},
                }
            raise AssertionError(payload)

    living_village = {
        "camp": 7,
        "terminated": False,
        "observation": {
            "laborCatalog": catalog_payload,
            "laborPrevious": feedback_payload,
        },
    }
    terminated_village = {
        "camp": 16,
        "terminated": True,
        "observation": None,
    }
    bridge = FakeBridge({
        "turn": 5,
        "villages": [living_village, terminated_village],
    })
    runtime = LaborBridgeRuntime(bridge, {7: catalog})
    initial = {
        "schemaVersion": 12,
        "movement": -1,
        "fission": 0,
        "tradeValue": [1.0] * 7,
        "tradeGate": [0] * 7,
        "tradeShare": 0.0,
        **_payload(catalog.masks),
    }
    step, _feedback = runtime.step_labor({7: initial})
    assert step["turn"] == 5
    assert set(runtime.catalogs) == {7}
    assert 16 not in runtime.catalogs

    bad_bridge1 = FakeBridge({
        "turn": 5,
        "villages": [living_village, {"camp": 16, "terminated": False, "observation": None}],
    })
    bad_runtime1 = LaborBridgeRuntime(bad_bridge1, {7: catalog})
    with pytest.raises(ValueError, match="malformed"):
        bad_runtime1.step_labor({7: initial})

    bad_bridge2 = FakeBridge({
        "turn": 5,
        "villages": [
            living_village,
            {
                "camp": 16,
                "terminated": True,
                "observation": {"laborCatalog": catalog_payload, "laborPrevious": None},
            },
        ],
    })
    bad_runtime2 = LaborBridgeRuntime(bad_bridge2, {7: catalog})
    with pytest.raises(ValueError, match="malformed"):
        bad_runtime2.step_labor({7: initial})

    step_dict = {"turn": 5, "villages": [living_village, terminated_village]}
    catalogs = LaborModel9Bridge._step_catalogs(step_dict)
    assert set(catalogs) == {7}

    with pytest.raises(ValueError, match="malformed"):
        LaborModel9Bridge._step_catalogs({
            "turn": 5,
            "villages": [
                living_village,
                {"camp": 16, "terminated": False, "observation": None},
            ],
        })

    with pytest.raises(ValueError, match="malformed"):
        LaborModel9Bridge._step_catalogs({
            "turn": 5,
            "villages": [
                living_village,
                {
                    "camp": 16,
                    "terminated": True,
                    "observation": {"laborCatalog": catalog_payload, "laborPrevious": None},
                },
            ],
        })


def test_step_labor_handles_extraneous_feedback_for_unsubmitted_camps() -> None:
    catalog_payload = _catalog_payload()
    catalog = LaborCatalog.from_payload(catalog_payload)
    feedback_payload = _feedback_payload(catalog_payload)

    class FakeBridge:
        def request(self, payload):
            if payload["op"] == "step_labor":
                return {
                    "ok": True,
                    "op": "step_labor",
                    "step": {
                        "turn": 5,
                        "villages": [
                            {
                                "camp": 7,
                                "terminated": False,
                                "observation": {
                                    "laborCatalog": catalog_payload,
                                    "laborPrevious": feedback_payload,
                                },
                            }
                        ],
                    },
                    "laborFeedback": {
                        "7": feedback_payload,
                        "11": feedback_payload,
                    },
                }
            raise AssertionError(payload["op"])

    runtime = LaborBridgeRuntime(FakeBridge(), {7: catalog})
    initial = {
        "movementCandidate": -1,
        "fissionPolicy": -1,
        **_payload(catalog.masks),
    }
    step, feedback = runtime.step_labor({7: initial})
    assert step["turn"] == 5
    assert set(feedback) == {7}


