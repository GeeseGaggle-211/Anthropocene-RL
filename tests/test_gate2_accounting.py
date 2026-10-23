from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest
from test_economic_diagnostics import transition, write_artifact

from anthropocene_rl.autonomous_v7 import economic_diagnostics
from anthropocene_rl.autonomous_v7.economic_accounting import (
    CRAFT_GOODS,
    MATERIALS,
    accounting_completeness,
    validate_economic_accounting,
)


def _conservation(domain: str, item: str, unit: str, *, zero: bool = False) -> dict:
    row = {
        "domain": domain,
        "item": item,
        "unit": unit,
        "opening": 1.0,
        "production": 1.0,
        "tradeIn": 0.0,
        "transferIn": 0.0,
        "consumption": 1.0,
        "tradeOut": 0.0,
        "transferOut": 0.0,
        "discard": 0.0,
        "closing": 1.0,
        "residual": 0.0,
        "complete": True,
        "unavailableReason": None,
    }
    if zero:
        for name in (
            "opening",
            "production",
            "tradeIn",
            "transferIn",
            "consumption",
            "tradeOut",
            "transferOut",
            "discard",
            "closing",
            "residual",
        ):
            row[name] = 0.0
    return row


def _good(name: str) -> dict:
    row = {
        "good": name,
        "unit": "good-units",
        "stockKind": "camp-inventory",
        "plannedOutputCeiling": 1.0,
    }
    for field in (
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
    ):
        row[field] = 0.0
    return row


def _material(name: str) -> dict:
    return {
        "material": name,
        "unit": "material-units",
        "received": 0.0,
        "openingStock": 0.0,
        "endingStock": 0.0,
        "recipeConsumption": 0.0,
        "actualTradeIn": 0.0,
        "actualTradeOut": 0.0,
        "transferIn": 0.0,
        "transferOut": 0.0,
        "workerIntervals": 0.0,
        "workerYears": 0.0,
        "requested": 0.0,
        "granted": 0.0,
        "phaseBRequested": 0.0,
        "phaseBGranted": 0.0,
        "phaseBReceived": 0.0,
    }


def _control(identity: str = "retainedFieldOperation") -> dict:
    phases = {
        "nativeDemand": 1.0,
        "actorCappedRequest": 0.8,
        "compilerGrant": 0.7,
        "executedWorkerOrInputClaim": 0.7,
        "committedPhysicalReceipt": 0.7,
        "inventoryChange": 0.7,
        "consumptionOrFinalService": 0.7,
        "endingPopulation": 1,
        "endingQol": 0.7,
    }
    return {
        "controlId": f"trajectory/base/1/7/control/{identity}",
        "domain": "cultivation",
        "identity": identity,
        "nativeUnit": "operation-fraction",
        "actorHead": "strategic[0]",
        "phases": phases,
        "phaseUnits": {
            name: "people"
            if name == "endingPopulation"
            else "qol"
            if name == "endingQol"
            else "operation-fraction"
            for name in phases
        },
        "unavailable": {},
        "metadata": {},
        "classification": "inactive",
        "classificationReason": "fixture-capability-inactive",
    }


def sidecar() -> dict:
    row = transition()["economicDiagnostics"]
    row = {
        **row,
        "schemaVersion": 2,
        "domains": {
            "population": {
                "openingLinked": 1.0,
                "endingLinked": 1.0,
                "reportStartingLinked": 1.0,
                "reportEndingLinked": 1.0,
                "grossBirths": 1.0,
                "grossDeaths": 1.0,
                "immigrated": 0.0,
                "emigrated": 0.0,
                "splitOff": 0.0,
                "decisionOpeningFood": 1.0,
                "committedEndingFood": 1.0,
                "foodProductionReceipt": 1.0,
                "humanFoodConsumption": 1.0,
                "manufacturingFood": 0.0,
                "foodTradeIn": 0.0,
                "foodTradeOut": 0.0,
                "foodTransferIn": 0.0,
                "foodTransferOut": 0.0,
                "foodDiscard": 0.0,
                "openingPlantFood": 0.0,
                "openingAnimalFood": 0.0,
                "openingAquaticFood": 0.0,
                "endingPlantFood": 0.0,
                "endingAnimalFood": 0.0,
                "endingAquaticFood": 0.0,
                "qol": 0.7,
            },
            "cultivation": {
                "capable": False,
                "openingCapital": 0.0,
                "openingFieldIds": [],
                "closingFieldIds": [],
                "accountedFieldIds": [],
            },
            "herding": {
                "capable": False,
                "openingHerdIds": [],
                "closingHerdIds": [],
            },
            "manufacturing": {"active": False},
            "trade": {"eligible": True},
        },
        "conservation": {"grossPopulationResidual": 0.0},
        "goods": [_good(name) for name in CRAFT_GOODS],
        "materials": [_material(name) for name in MATERIALS],
        "fields": [],
        "herds": [],
        "requests": [],
        "transfers": [],
        "controls": [_control()],
        "conservationRows": [
            _conservation("population", "linked-population", "people"),
            _conservation("food", "linked-food-reserves", "food"),
            *[_conservation("material", name, "material-units", zero=True) for name in MATERIALS],
            *[_conservation("craft-good", name, "good-units", zero=True) for name in CRAFT_GOODS],
        ],
        "physicalEvents": [
            {
                "eventId": f"{row['trajectoryId']}/boundary/{category}/{name}",
                "campId": row["campId"],
                "category": category,
                "item": name,
                "unit": "material-units" if name in MATERIALS else "good-units",
                "quantity": 0.0,
                "direction": "boundary",
            }
            for name in (*CRAFT_GOODS, *MATERIALS)
            for category in ("opening-stock-boundary", "ending-stock-boundary")
        ]
        + [
            {
                "eventId": f"{row['trajectoryId']}/population-food/{category}/{item}",
                "campId": row["campId"],
                "category": category,
                "item": item,
                "unit": unit,
                "quantity": quantity,
                "direction": direction,
            }
            for category, item, unit, quantity, direction in (
                ("opening-stock-boundary", "linked-population", "people", 1.0, "boundary"),
                ("ending-stock-boundary", "linked-population", "people", 1.0, "boundary"),
                ("population-birth", "linked-population", "people", 1.0, "in"),
                ("population-death", "linked-population", "people", 1.0, "out"),
                ("opening-stock-boundary", "linked-food-reserves", "food", 1.0, "boundary"),
                ("ending-stock-boundary", "linked-food-reserves", "food", 1.0, "boundary"),
                ("food-production-receipt", "linked-food-reserves", "food", 1.0, "in"),
                ("human-food-consumption", "linked-food-reserves", "food", 1.0, "out"),
            )
        ],
        "trades": [
            {
                "recordType": "eligibility",
                "eventId": f"{row['trajectoryId']}/eligibility/7",
                "campId": 7,
                "societyId": 3,
                "eligible": True,
                "gatingReason": None,
                "partnerSearchCount": 0,
                "partnerSearchReason": "no-eligible-remembered-partner-in-range",
            }
        ],
    }
    return row


def transition_v2(*, turn: int = 1) -> dict:
    row = transition(turn=turn)
    economic = sidecar()
    economic["transitionId"] = row["transitionId"]
    economic["trajectoryId"] = row["trajectoryId"]
    economic["decisionTurn"] = row["decisionTurn"]
    for control in economic["controls"]:
        control["controlId"] = f"{row['transitionId']}/control/{control['identity']}"
    row["economicDiagnostics"] = economic
    row["actorDistribution"] = {"strategic": {"mean": [0.8] * 5}}
    return row


def reciprocal_endpoint(row: dict) -> dict:
    """Return the responder's independently validated sidecar for one bilateral fixture."""

    result = copy.deepcopy(row)
    old_transition = str(result["transitionId"])
    result["transitionId"] = old_transition.rsplit(":", 1)[0] + ":8"
    result["campId"] = 8
    result["societyId"] = 4
    result["actionDiagnostics"]["campId"] = 8
    economic = result["economicDiagnostics"]
    economic["transitionId"] = result["transitionId"]
    economic["campId"] = 8
    economic["societyId"] = 4
    for control in economic["controls"]:
        control["controlId"] = str(control["controlId"]).replace(
            old_transition, result["transitionId"], 1
        )
    goods = {item["good"]: item for item in economic["goods"]}
    ledgers = {(item["domain"], item["item"]): item for item in economic["conservationRows"]}
    for good in goods.values():
        good.update(openingStock=0.0, endingStock=0.0, actualTradeIn=0.0, actualTradeOut=0.0)
        ledger = ledgers[("craft-good", good["good"])]
        ledger.update(opening=0.0, closing=0.0, tradeIn=0.0, tradeOut=0.0, residual=0.0)
    goods["Cloth"].update(openingStock=1.5, actualTradeOut=1.5)
    ledgers[("craft-good", "Cloth")].update(opening=1.5, tradeOut=1.5)
    goods["Pots"].update(actualTradeIn=2.25, endingStock=2.25)
    ledgers[("craft-good", "Pots")].update(tradeIn=2.25, closing=2.25)
    for event in economic["physicalEvents"]:
        event["campId"] = 8
        event["eventId"] = f"{event['eventId']}/endpoint-8"
        if event["item"] in CRAFT_GOODS and event["category"] in {
            "opening-stock-boundary",
            "ending-stock-boundary",
            "trade-in",
            "trade-out",
        }:
            event["quantity"] = 0.0
        if event["item"] == "Cloth" and event["category"] == "opening-stock-boundary":
            event["quantity"] = 1.5
        elif event["category"] == "trade-out":
            event.update(item="Cloth", quantity=1.5)
        elif event["item"] == "Pots" and event["category"] == "ending-stock-boundary":
            event["quantity"] = 2.25
        elif event["category"] == "trade-in":
            event.update(item="Pots", quantity=2.25)
    transaction = next(item for item in economic["trades"] if item["recordType"] == "transaction")
    for control in [item for item in economic["controls"] if item["domain"] == "trade"]:
        leg = next(item for item in transaction["legs"] if item["legId"] == control["identity"])
        control["phases"]["inventoryChange"] = (
            leg["receiverCreditQuantity"] if leg["receiver"] == 8 else -leg["senderDebitQuantity"]
        )
    return result


def _leg(identity: str, sender: int, receiver: int, quantity: float, good: str) -> dict:
    delivered = quantity * 0.75
    return {
        "legId": identity,
        "sender": sender,
        "receiver": receiver,
        "good": good,
        "unit": "trade-good-units",
        "requestedQuantity": quantity,
        "agreedQuantity": quantity,
        "stockQuantity": quantity,
        "madeToOrderQuantity": 0.0,
        "escrowedQuantity": quantity,
        "producedMadeToOrderQuantity": 0.0,
        "supplyAvailableQuantity": quantity,
        "bilateralDeliverableQuantity": quantity,
        "deliveredQuantity": delivered,
        "acceptedQuantity": delivered,
        "senderDebitQuantity": delivered,
        "receiverCreditQuantity": delivered,
        "deliveryFraction": 0.75,
        "producerDeliveryFraction": 1.0,
        "storageBlockedQuantity": quantity - delivered,
        "counterpartyShortfallQuantity": 0.0,
        "refundedOrReturnedQuantity": quantity - delivered,
        "destroyedQuantity": 0.0,
        "deliveryPhase": "opening",
        "state": "partial",
        "parts": [
            {
                "item": good,
                "stockQuantity": quantity,
                "madeToOrderQuantity": 0.0,
                "deliveredQuantity": delivered,
            }
        ],
    }


def with_trade(row: dict) -> dict:
    row = copy.deepcopy(row)
    request_id = f"{row['trajectoryId']}/turn/1/request/0"
    transaction_id = f"{row['trajectoryId']}/turn/1/trade/0"
    row["trades"].extend(
        [
            {
                "recordType": "requisition",
                "requestId": request_id,
                "requester": 7,
                "responder": 8,
                "distance": 1,
                "affinity": 0.5,
                "wantedGoods": ["Cloth"],
                "offeredGoods": ["Pots"],
                "outcome": "honoured",
                "rejectionReason": None,
            },
            {
                "recordType": "transaction",
                "requestId": request_id,
                "transactionId": transaction_id,
                "requester": 7,
                "responder": 8,
                "distance": 1,
                "handshake": "honoured",
                "ratio": 1.0,
                "requesterPlannedGain": 1.0,
                "responderPlannedGain": 1.0,
                "settlementState": "partial",
                "deliveryFraction": 0.75,
                "requesterImmediateUtility": 0.5,
                "responderImmediateUtility": 0.5,
                "requesterImmediateUtilityUnavailableReason": None,
                "responderImmediateUtilityUnavailableReason": None,
                "delayedUtility": None,
                "delayedUtilityUnavailableReason": (
                    "future-outcome-linkage-not-yet-observed-at-current-transition"
                ),
                "legs": [
                    _leg(f"{transaction_id}/want", 8, 7, 2.0, "Cloth"),
                    _leg(f"{transaction_id}/give", 7, 8, 3.0, "Pots"),
                ],
            },
            {
                "recordType": "eligibility",
                "eventId": f"{row['trajectoryId']}/eligibility/8",
                "campId": 8,
                "societyId": 4,
                "eligible": True,
                "gatingReason": None,
                "partnerSearchCount": 1,
            },
        ]
    )
    goods = {item["good"]: item for item in row["goods"]}
    ledgers = {(item["domain"], item["item"]): item for item in row["conservationRows"]}
    goods["Cloth"]["actualTradeIn"] = 1.5
    goods["Cloth"]["endingStock"] = 1.5
    ledgers[("craft-good", "Cloth")]["tradeIn"] = 1.5
    ledgers[("craft-good", "Cloth")]["closing"] = 1.5
    goods["Pots"]["openingStock"] = 2.25
    goods["Pots"]["actualTradeOut"] = 2.25
    ledgers[("craft-good", "Pots")]["opening"] = 2.25
    ledgers[("craft-good", "Pots")]["tradeOut"] = 2.25
    for event in row["physicalEvents"]:
        if event["item"] == "Cloth" and event["category"] == "ending-stock-boundary":
            event["quantity"] = 1.5
        if event["item"] == "Pots" and event["category"] == "opening-stock-boundary":
            event["quantity"] = 2.25
    row["physicalEvents"].extend(
        [
            {
                "eventId": f"{row['trajectoryId']}/trade/{direction}/{item}",
                "campId": row["campId"],
                "category": f"trade-{direction}",
                "item": item,
                "unit": "good-units",
                "quantity": quantity,
                "direction": "in" if direction == "in" else "out",
            }
            for direction, item, quantity in (("in", "Cloth", 1.5), ("out", "Pots", 2.25))
        ]
    )
    transaction = row["trades"][2]
    for leg in transaction["legs"]:
        incoming = leg["receiver"] == row["campId"]
        phases = {
            "nativeDemand": leg["requestedQuantity"],
            "actorCappedRequest": leg["agreedQuantity"],
            "compilerGrant": leg["agreedQuantity"],
            "executedWorkerOrInputClaim": leg["supplyAvailableQuantity"],
            "committedPhysicalReceipt": leg["deliveredQuantity"],
            "inventoryChange": leg["receiverCreditQuantity"]
            if incoming
            else -leg["senderDebitQuantity"],
            "consumptionOrFinalService": 0.5,
            "endingPopulation": 1.0,
            "endingQol": 0.7,
        }
        row["controls"].append(
            {
                "controlId": (
                    f"{row['transitionId']}/control/trade/{leg['legId'].rsplit('/', 1)[-1]}"
                ),
                "domain": "trade",
                "identity": leg["legId"],
                "nativeUnit": "trade-good-units",
                "actorHead": "tradeValue+tradeGate+tradeShare",
                "phases": phases,
                "phaseUnits": {
                    name: "people"
                    if name == "endingPopulation"
                    else "qol"
                    if name == "endingQol"
                    else "utility"
                    if name == "consumptionOrFinalService"
                    else "trade-good-units"
                    for name in phases
                },
                "unavailable": {},
                "metadata": {"transactionId": transaction_id, "good": leg["good"]},
                "classification": "active",
                "classificationReason": None,
            }
        )
    return row


def _set_good_flow(
    row: dict,
    good: str,
    *,
    opening: float,
    consumption: float,
    closing: float,
) -> None:
    physical = {item["good"]: item for item in row["goods"]}[good]
    physical["openingStock"] = opening
    physical["serviceConsumptionOrDepreciation"] = consumption
    physical["endingStock"] = closing
    ledger = next(
        item
        for item in row["conservationRows"]
        if (item["domain"], item["item"]) == ("craft-good", good)
    )
    ledger["opening"] = opening
    ledger["consumption"] = consumption
    ledger["closing"] = closing
    ledger["residual"] = (
        opening
        + physical["output"]
        + physical["actualTradeIn"]
        + physical["transferIn"]
        - consumption
        - physical["actualTradeOut"]
        - physical["transferOut"]
        - closing
    )
    for event in row["physicalEvents"]:
        if event["item"] != good:
            continue
        if event["category"] == "opening-stock-boundary":
            event["quantity"] = opening
        elif event["category"] == "ending-stock-boundary":
            event["quantity"] = closing
    if consumption:
        row["physicalEvents"].append(
            {
                "eventId": f"{row['trajectoryId']}/service/{row['decisionTurn']}/{good}",
                "campId": row["campId"],
                "category": "final-service-fulfilment",
                "item": good,
                "unit": "good-units",
                "quantity": consumption,
                "direction": "out",
                "phase": "needs-after-production",
            }
        )


def test_complete_schema_and_bilateral_lifecycle_are_accepted() -> None:
    value = with_trade(sidecar())
    validate_economic_accounting(value)
    result = accounting_completeness(value)
    assert result["status"] == "HOLD"
    assert result["tradeTransactionCount"] == 1
    assert result["terminalLinkComplete"] is False
    assert "matched-intervention delayed causal effect" in " ".join(result["reasons"])


def _active_cultivation_coverage() -> dict:
    value = sidecar()
    value["domains"]["cultivation"]["capable"] = True
    retained = value["controls"][0]
    retained.update(classification="active", classificationReason=None)
    development = copy.deepcopy(retained)
    development["identity"] = "fieldDevelopment"
    development["controlId"] = f"{value['transitionId']}/control/fieldDevelopment"
    value["controls"].append(development)
    return value


def test_honest_inactive_control_is_coverage_not_a_completeness_failure() -> None:
    result = accounting_completeness(sidecar())
    assert result["status"] == "PASS-ACCOUNTING-ONLY"
    assert result["controlCount"] == 1
    assert result["activeControlCount"] == 0
    assert result["inactiveControlCount"] == 1
    assert result["controlEvidenceDenominator"] == 0
    assert result["completeEvidenceControlCount"] == 0
    assert result["excludedControlCount"] == 1
    assert result["excludedControls"] == [
        {
            "controlId": sidecar()["controls"][0]["controlId"],
            "classification": "inactive",
            "reason": "fixture-capability-inactive",
        }
    ]


def test_active_control_with_unavailable_phase_remains_a_hold() -> None:
    value = _active_cultivation_coverage()
    value["controls"][0]["phases"]["consumptionOrFinalService"] = None
    value["controls"][0]["unavailable"] = {
        "consumptionOrFinalService": "no-authoritative-terminal-link"
    }
    result = accounting_completeness(value)
    assert result["status"] == "HOLD"
    assert result["controlEvidenceDenominator"] == 2
    assert result["completeEvidenceControlCount"] == 1
    assert result["terminalLinkComplete"] is False
    assert "missing consumptionOrFinalService" in " ".join(result["reasons"])


def test_unsupported_control_for_active_capability_remains_a_hold() -> None:
    value = _active_cultivation_coverage()
    value["controls"][0].update(
        classification="unsupported",
        classificationReason="no-bound-elder-decision",
    )
    result = accounting_completeness(value)
    assert result["status"] == "HOLD"
    assert result["unsupportedControlCount"] == 1
    assert result["controlEvidenceDenominator"] == 2
    assert result["excludedControlCount"] == 0
    assert "active-capability control" in " ".join(result["reasons"])


def test_signed_negative_trade_utility_is_valid_for_utility_phase_only() -> None:
    value = with_trade(sidecar())
    transaction = next(row for row in value["trades"] if row["recordType"] == "transaction")
    transaction["requesterImmediateUtility"] = -0.5
    for control in value["controls"]:
        if control["domain"] == "trade":
            control["phases"]["consumptionOrFinalService"] = -0.5
    validate_economic_accounting(value)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row["conservationRows"].pop(), "mandatory conservation"),
        (lambda row: row["goods"][0].__setitem__("unit", "mixed-service"), "unit differs"),
        (lambda row: row["controls"][0]["phases"].__setitem__("compilerGrant", -1), "nonnegative"),
        (lambda row: row["trades"][2]["legs"].pop(), "exactly two legs"),
        (
            lambda row: row["trades"][2]["legs"][0].__setitem__("acceptedQuantity", 9),
            "accepted credit",
        ),
        (
            lambda row: row["trades"][2]["legs"][0]["parts"][0].__setitem__("item", "Pots"),
            "stable good registry",
        ),
        (
            lambda row: row["trades"][2]["legs"][0].__setitem__("supplyAvailableQuantity", 9),
            "supply does not reconcile",
        ),
        (
            lambda row: row["trades"][2]["legs"][0].__setitem__("refundedOrReturnedQuantity", 0),
            "refund/return does not reconcile",
        ),
        (lambda row: row["trades"][2].__setitem__("assertedComplete", True), "fields differ"),
        (lambda row: row["physicalEvents"].pop(0), "must exactly cover"),
    ],
)
def test_hostile_schema_conservation_and_lifecycle_fail_closed(mutator, message: str) -> None:
    value = with_trade(sidecar())
    mutator(value)
    with pytest.raises(ValueError, match=message):
        validate_economic_accounting(value)


def test_nonzero_but_honestly_derived_residual_is_a_hold() -> None:
    value = sidecar()
    ledger = value["conservationRows"][0]
    ledger["closing"] = 0.5
    ledger["residual"] = 0.5
    value["domains"]["population"]["endingLinked"] = 0.5
    value["controls"][0]["phases"]["endingPopulation"] = 0.5
    next(
        event
        for event in value["physicalEvents"]
        if event["category"] == "ending-stock-boundary" and event["item"] == "linked-population"
    )["quantity"] = 0.5
    result = accounting_completeness(value)
    assert result["status"] == "HOLD"
    assert any("nonzero conservation residual" in reason for reason in result["reasons"])


def test_forged_residual_is_rejected_instead_of_masking_terms() -> None:
    value = sidecar()
    value["conservationRows"][0]["closing"] = 0.5
    with pytest.raises(ValueError, match=r"not derived|does not reconcile"):
        validate_economic_accounting(value)


def test_demographic_endpoint_change_cannot_be_hidden_as_integer_carry() -> None:
    value = sidecar()
    value["domains"]["population"]["reportEndingLinked"] = 2.0
    with pytest.raises(ValueError, match="gross population residual is not independently derived"):
        validate_economic_accounting(value)

    value["conservation"]["grossPopulationResidual"] = 1.0
    result = accounting_completeness(value)
    assert result["status"] == "HOLD"
    assert any("nonzero gross population residual" in reason for reason in result["reasons"])


def test_stored_production_keeps_receipt_and_work_separate_from_terminal_service() -> None:
    value = sidecar()
    value["domains"]["manufacturing"]["active"] = True
    request_id = f"{value['transitionId']}/request/0"
    value["requests"] = [
        {
            "requestId": request_id,
            "good": "Cloth",
            "rawMaterial": None,
            "nativeDemand": 2.0,
            "actorCappedRequest": 2.0,
            "compilerGrant": 2.0,
            "executedWorkerYears": 1.0,
            "committedPhysicalReceipt": 2.0,
            "inventoryChange": None,
            "consumptionOrFinalService": 0.0,
        }
    ]
    control = _control(request_id)
    control.update(
        controlId=f"{value['transitionId']}/control/request/0",
        domain="manufacturing",
        identity=request_id,
        nativeUnit="good-units",
        actorHead="service[1]",
        classification="active",
        classificationReason=None,
    )
    control["phases"].update(
        nativeDemand=2.0,
        actorCappedRequest=2.0,
        compilerGrant=2.0,
        executedWorkerOrInputClaim=1.0,
        committedPhysicalReceipt=2.0,
        inventoryChange=None,
        consumptionOrFinalService=0.0,
    )
    control["phaseUnits"].update(
        {
            phase: "good-units"
            for phase in (
                "nativeDemand",
                "actorCappedRequest",
                "compilerGrant",
                "committedPhysicalReceipt",
                "inventoryChange",
                "consumptionOrFinalService",
            )
        },
        executedWorkerOrInputClaim="worker-years",
    )
    control["unavailable"] = {
        "inventoryChange": "request-level-net-stock-mutation-is-not-authoritatively-attributed"
    }
    value["controls"].append(control)

    cloth = next(row for row in value["goods"] if row["good"] == "Cloth")
    cloth.update(
        output=2.0,
        endingStock=2.0,
        beforeFinalTradeStock=2.0,
        afterFinalTradeStock=2.0,
        workerIntervals=1.0,
        workerYears=1.0,
    )
    ledger = next(
        row
        for row in value["conservationRows"]
        if row["domain"] == "craft-good" and row["item"] == "Cloth"
    )
    ledger.update(production=2.0, closing=2.0, residual=0.0)
    next(
        event
        for event in value["physicalEvents"]
        if event["category"] == "ending-stock-boundary" and event["item"] == "Cloth"
    )["quantity"] = 2.0
    value["physicalEvents"].extend(
        [
            {
                "eventId": f"{value['trajectoryId']}/production/Cloth",
                "campId": value["campId"],
                "category": "production-receipt",
                "item": "Cloth",
                "unit": "good-units",
                "quantity": 2.0,
                "direction": "in",
                "receiptId": f"{value['transitionId']}/receipt/0",
                "sourceRequestId": request_id,
            },
            {
                "eventId": f"{value['trajectoryId']}/service-final/Cloth",
                "campId": value["campId"],
                "category": "final-service-fulfilment",
                "item": "Cloth",
                "unit": "good-units",
                "quantity": 0.0,
                "direction": "out",
                "phase": "needs-after-production",
            },
        ]
    )

    validate_economic_accounting(value)
    assert control["phases"]["committedPhysicalReceipt"] == 2.0
    assert control["phases"]["executedWorkerOrInputClaim"] == 1.0
    assert control["phases"]["consumptionOrFinalService"] == 0.0

    forged = copy.deepcopy(value)
    production = next(
        event
        for event in forged["physicalEvents"]
        if event["category"] == "production-receipt" and event["item"] == "Cloth"
    )
    production["sourceRequestId"] = f"{value['transitionId']}/request/forged"
    with pytest.raises(ValueError, match="source request is not authoritative"):
        validate_economic_accounting(forged)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row.__setitem__("controls", []), "require causal controls"),
        (
            lambda row: row["controls"][0]["phaseUnits"].__setitem__(
                "executedWorkerOrInputClaim", "bananas"
            ),
            "not registered",
        ),
        (
            lambda row: (
                row["conservationRows"][0].__setitem__("opening", 2.0),
                row["conservationRows"][0].__setitem__("closing", 2.0),
                next(
                    event
                    for event in row["physicalEvents"]
                    if event["category"] == "opening-stock-boundary"
                    and event["item"] == "linked-population"
                ).__setitem__("quantity", 2.0),
                next(
                    event
                    for event in row["physicalEvents"]
                    if event["category"] == "ending-stock-boundary"
                    and event["item"] == "linked-population"
                ).__setitem__("quantity", 2.0),
            ),
            "not cross-bound",
        ),
        (
            lambda row: row["domains"]["cultivation"].update(
                capable=True,
                openingCapital=2.0,
                openingFieldIds=[44],
                closingFieldIds=[44],
                accountedFieldIds=[44],
            ),
            "field rows do not exactly cover",
        ),
    ],
)
def test_authoritative_coverage_and_units_fail_closed(mutator, message: str) -> None:
    value = sidecar()
    mutator(value)
    with pytest.raises(ValueError, match=message):
        validate_economic_accounting(value)


def test_missing_mutation_reason_cannot_hide_inventory_consumption() -> None:
    value = sidecar()
    _set_good_flow(value, "Cloth", opening=1.0, consumption=1.0, closing=0.0)
    value["physicalEvents"] = [
        event
        for event in value["physicalEvents"]
        if event["category"] != "final-service-fulfilment"
    ]
    with pytest.raises(ValueError, match="service events do not reconcile"):
        validate_economic_accounting(value)


def test_final_service_requires_an_authoritative_phase() -> None:
    value = sidecar()
    _set_good_flow(value, "Cloth", opening=1.0, consumption=1.0, closing=0.0)
    service = next(
        event
        for event in value["physicalEvents"]
        if event["category"] == "final-service-fulfilment"
    )
    service["phase"] = "after-everything"
    with pytest.raises(ValueError, match="final service phase is not registered"):
        validate_economic_accounting(value)


def test_typed_food_consumption_requires_the_physical_needs_phase() -> None:
    value = sidecar()
    for index, item in enumerate(("PlantFood", "AnimalFood", "AquaticFood")):
        value["physicalEvents"].append(
            {
                "eventId": f"{value['trajectoryId']}/typed-food/{index}",
                "campId": value["campId"],
                "category": "human-food-consumption",
                "item": item,
                "unit": "food",
                "quantity": 0.25,
                "direction": "out",
                "phase": "needs-after-production",
            }
        )
    validate_economic_accounting(value)

    missing = copy.deepcopy(value)
    del missing["physicalEvents"][-1]["phase"]
    with pytest.raises(ValueError, match="typed food-consumption phase differs"):
        validate_economic_accounting(missing)

    wrong = copy.deepcopy(value)
    wrong["physicalEvents"][-1]["phase"] = "work-before-production"
    with pytest.raises(ValueError, match="typed food-consumption phase differs"):
        validate_economic_accounting(wrong)


def test_aggregate_food_consumption_cannot_claim_a_component_phase() -> None:
    value = sidecar()
    aggregate = next(
        event
        for event in value["physicalEvents"]
        if event["category"] == "human-food-consumption"
        and event["item"] == "linked-food-reserves"
    )
    aggregate["phase"] = "needs-after-production"
    with pytest.raises(ValueError, match="phase is not valid for category"):
        validate_economic_accounting(value)


def test_trade_parts_must_reconcile_to_physical_inventory_and_trajectory() -> None:
    mismatched = with_trade(sidecar())
    mismatched["goods"][2]["actualTradeIn"] = 0.0
    with pytest.raises(
        ValueError, match=r"conservation terms differ|physical parts|trade-in does not reconcile"
    ):
        validate_economic_accounting(mismatched)
    unbound = with_trade(sidecar())
    unbound["trades"][2]["transactionId"] = "other-trajectory/trade/0"
    for leg in unbound["trades"][2]["legs"]:
        leg["legId"] = f"{unbound['trades'][2]['transactionId']}/{leg['legId'].rsplit('/', 1)[-1]}"
    with pytest.raises(ValueError, match="not bound to the trajectory"):
        validate_economic_accounting(unbound)


def test_bilateral_trade_rejects_goods_eligibility_fraction_and_endpoint_forgery() -> None:
    mismatched_goods = with_trade(sidecar())
    mismatched_goods["trades"][1]["wantedGoods"] = ["Pots"]
    with pytest.raises(ValueError, match="bound requisition goods"):
        validate_economic_accounting(mismatched_goods)

    false_eligibility = with_trade(sidecar())
    false_eligibility["trades"][3]["eligible"] = False
    false_eligibility["trades"][3]["gatingReason"] = "forged-ineligible"
    with pytest.raises(ValueError, match="ineligible endpoint"):
        validate_economic_accounting(false_eligibility)

    unequal = with_trade(sidecar())
    unequal["trades"][2]["deliveryFraction"] = 0.5
    with pytest.raises(ValueError, match="unequal delivery fractions"):
        validate_economic_accounting(unequal)

    one_endpoint = with_trade(sidecar())
    one_endpoint["trades"].pop(3)
    with pytest.raises(ValueError, match="missing reciprocal endpoint eligibility"):
        validate_economic_accounting(one_endpoint)


def test_partial_made_to_order_and_party_vanished_receipts_are_explicit() -> None:
    partial = with_trade(sidecar())
    leg = partial["trades"][2]["legs"][0]
    leg.update(
        stockQuantity=1.0,
        madeToOrderQuantity=1.0,
        escrowedQuantity=1.0,
        producedMadeToOrderQuantity=0.5,
        supplyAvailableQuantity=1.5,
        bilateralDeliverableQuantity=1.5,
        storageBlockedQuantity=0.0,
        refundedOrReturnedQuantity=0.0,
        producerDeliveryFraction=0.5,
    )
    leg["parts"][0].update(stockQuantity=1.0, madeToOrderQuantity=1.0)
    control = next(item for item in partial["controls"] if item["identity"] == leg["legId"])
    control["phases"]["executedWorkerOrInputClaim"] = 1.5
    validate_economic_accounting(partial)

    vanished = with_trade(sidecar())
    transaction = vanished["trades"][2]
    transaction.update(
        settlementState="party-vanished",
        deliveryFraction=0.0,
        requesterImmediateUtility=None,
        responderImmediateUtility=None,
        requesterImmediateUtilityUnavailableReason="party-vanished-before-terminal-utility",
        responderImmediateUtilityUnavailableReason="party-vanished-before-terminal-utility",
    )
    for leg in transaction["legs"]:
        leg.update(
            supplyAvailableQuantity=leg["escrowedQuantity"],
            bilateralDeliverableQuantity=0.0,
            deliveredQuantity=0.0,
            acceptedQuantity=0.0,
            senderDebitQuantity=0.0,
            receiverCreditQuantity=0.0,
            deliveryFraction=0.0,
            storageBlockedQuantity=0.0,
            counterpartyShortfallQuantity=leg["escrowedQuantity"],
            refundedOrReturnedQuantity=leg["escrowedQuantity"],
            state="party-vanished",
        )
        leg["parts"][0]["deliveredQuantity"] = 0.0
    goods = {item["good"]: item for item in vanished["goods"]}
    ledgers = {(item["domain"], item["item"]): item for item in vanished["conservationRows"]}
    goods["Cloth"].update(actualTradeIn=0.0, endingStock=0.0)
    ledgers[("craft-good", "Cloth")].update(tradeIn=0.0, closing=0.0)
    goods["Pots"].update(openingStock=0.0, actualTradeOut=0.0)
    ledgers[("craft-good", "Pots")].update(opening=0.0, tradeOut=0.0)
    for event in vanished["physicalEvents"]:
        if event["item"] in {"Cloth", "Pots"}:
            event["quantity"] = 0.0
    for control in [item for item in vanished["controls"] if item["domain"] == "trade"]:
        bound = next(leg for leg in transaction["legs"] if leg["legId"] == control["identity"])
        control["phases"].update(
            executedWorkerOrInputClaim=bound["supplyAvailableQuantity"],
            committedPhysicalReceipt=0.0,
            inventoryChange=0.0,
            consumptionOrFinalService=None,
        )
        control["unavailable"] = {
            "consumptionOrFinalService": "party-vanished-before-terminal-utility"
        }
    validate_economic_accounting(vanished)


def test_field_and_herd_receipts_are_typed_and_bound_to_conservation() -> None:
    value = sidecar()
    value["fields"] = [
        {
            "fieldId": 4,
            "openingCapital": 2.0,
            "meanCapital": 2.25,
            "decayedCapital": 1.5,
            "capitalAdded": 1.0,
            "capitalLost": 0.5,
            "transferIn": 0.0,
            "transferOut": 0.0,
            "endingCapital": 2.5,
            "residual": 0.0,
        }
    ]
    field = _conservation("field", "4", "field-capital", zero=True)
    field.update(opening=2.0, production=1.0, consumption=0.5, closing=2.5)
    value["herds"] = [
        {
            "herdId": 9,
            "openingAnimals": 10,
            "startingAnimals": 10,
            "survivingAnimals": 9,
            "newbornAnimals": 2,
            "starvationDeaths": 1,
            "slaughtered": 1,
            "foodNutrition": 1.0,
            "waterNutrition": 1.0,
            "nutrition": 1.0,
            "managementFulfilment": 1.0,
            "potentialFood": 3.0,
            "acceptedFood": 2.0,
            "slaughterFood": 1.0,
            "coveredHerderYears": 4.0,
            "herdingToolsConsumed": 0.5,
            "ownershipInflow": 0.0,
            "ownershipOutflow": 0.0,
            "committedEndingAnimals": 10,
            "reportAvailable": True,
            "unavailableReason": None,
            "residual": 0.0,
        }
    ]
    herd = _conservation("herd", "9", "animals", zero=True)
    herd.update(opening=10.0, production=2.0, consumption=2.0, closing=10.0)
    value["conservationRows"].extend([field, herd])
    value["domains"]["cultivation"]["openingFieldIds"] = [4]
    value["domains"]["cultivation"]["closingFieldIds"] = [4]
    value["domains"]["cultivation"]["accountedFieldIds"] = [4]
    value["domains"]["herding"]["openingHerdIds"] = [9]
    value["domains"]["herding"]["closingHerdIds"] = [9]
    value["physicalEvents"].extend(
        {
            "eventId": f"{value['trajectoryId']}/identity/{item}/{category}",
            "campId": value["campId"],
            "category": category,
            "item": item,
            "unit": unit,
            "quantity": quantity,
            "direction": direction,
        }
        for item, unit, entries in (
            (
                "field:4",
                "field-capital",
                (
                    ("opening-stock-boundary", 2.0, "boundary"),
                    ("ending-stock-boundary", 2.5, "boundary"),
                    ("field-capital-addition", 1.0, "in"),
                    ("field-capital-loss", 0.5, "out"),
                ),
            ),
            (
                "herd:9",
                "animals",
                (
                    ("opening-stock-boundary", 10.0, "boundary"),
                    ("ending-stock-boundary", 10.0, "boundary"),
                    ("herd-birth", 2.0, "in"),
                    ("herd-death-or-slaughter", 2.0, "out"),
                ),
            ),
        )
        for category, quantity, direction in entries
    )
    validate_economic_accounting(value)
    bad_field = copy.deepcopy(value)
    bad_field["fields"][0]["capitalAdded"] = 3.0
    with pytest.raises(ValueError, match="residual is not derived"):
        validate_economic_accounting(bad_field)
    bad_herd = copy.deepcopy(value)
    bad_herd["herds"][0]["committedEndingAnimals"] = 999
    with pytest.raises(
        ValueError,
        match=(
            r"herds\[0\] residual is not derived|herd conservation terms differ|"
            r"does not reconcile for herd"
        ),
    ):
        validate_economic_accounting(bad_herd)


def test_artifact_aggregation_emits_deterministic_gate2_tables(tmp_path: Path) -> None:
    rows = [transition_v2(turn=1), transition_v2(turn=2)]
    first, second = tmp_path / "first", tmp_path / "second"
    write_artifact(first, rows)
    write_artifact(second, list(reversed(rows)))
    first_out, second_out = tmp_path / "first-out", tmp_path / "second-out"
    first_report = economic_diagnostics.aggregate_trace_directories([first], first_out)
    second_report = economic_diagnostics.aggregate_trace_directories([second], second_out)
    assert first_report["gate2Accounting"]["status"] == "HOLD"
    assert second_report["gate2Accounting"]["status"] == "HOLD"
    assert "no bilateral transaction evidence" in " ".join(
        first_report["gate2Accounting"]["reasons"]
    )
    for name in (
        "control_matrix.csv",
        "conservation_terms.csv",
        "physical_events.csv",
        "trade_lifecycle.csv",
        "trade_provenance_lots.csv",
        "trade_provenance_allocations.csv",
        "trade_downstream_windows.csv",
    ):
        assert (first_out / name).read_bytes() == (second_out / name).read_bytes()


def test_full_artifact_rejects_incomplete_subset_and_trade_forgery(tmp_path: Path) -> None:
    for index, mutate in enumerate(
        (
            lambda row: row["economicDiagnostics"]["conservationRows"].pop(),
            lambda row: row["economicDiagnostics"]["trades"].append(
                {"recordType": "transaction", "transactionId": "unjoined"}
            ),
        )
    ):
        row = transition_v2()
        mutate(row)
        artifact = tmp_path / f"bad-{index}"
        write_artifact(artifact, [row])
        with pytest.raises(ValueError):
            economic_diagnostics.aggregate_trace_directories([artifact], tmp_path / f"out-{index}")


def test_trade_lot_bookkeeping_is_bounded_deterministic_and_not_causal(
    tmp_path: Path,
) -> None:
    first = transition_v2(turn=1)
    first["economicDiagnostics"] = with_trade(first["economicDiagnostics"])
    counterpart = reciprocal_endpoint(first)
    _set_good_flow(first["economicDiagnostics"], "Cloth", opening=0.0, consumption=1.0, closing=0.5)
    second = transition_v2(turn=2)
    _set_good_flow(
        second["economicDiagnostics"], "Cloth", opening=0.5, consumption=0.5, closing=0.0
    )
    artifact = tmp_path / "artifact"
    write_artifact(artifact, [first, counterpart, second])
    output = tmp_path / "out"
    report = economic_diagnostics.aggregate_trace_directories([artifact], output)
    assert report["gate2Accounting"]["status"] == "HOLD"
    assert report["counts"]["tradeProvenanceLots"] == 2
    with (output / "trade_provenance_lots.csv").open(newline="", encoding="utf-8") as handle:
        lots = list(csv.DictReader(handle))
    with (output / "trade_provenance_allocations.csv").open(newline="", encoding="utf-8") as handle:
        allocations = list(csv.DictReader(handle))
    with (output / "trade_downstream_windows.csv").open(newline="", encoding="utf-8") as handle:
        windows = list(csv.DictReader(handle))
    cloth_lot = next(row for row in lots if row["item"] == "Cloth")
    cloth_allocations = [row for row in allocations if row["item"] == "Cloth"]
    assert cloth_lot["horizonStatus"] == "fully-consumed"
    assert float(cloth_lot["receivedQuantity"]) == 1.5
    assert float(cloth_lot["consumedQuantity"]) == 1.5
    assert float(cloth_lot["storageBlockedLegQuantity"]) == 0.5
    assert float(cloth_lot["refundedOrReturnedLegQuantity"]) == 0.5
    assert sum(float(row["allocatedQuantity"]) for row in cloth_allocations) == 1.5
    assert [float(row["allocatedQuantity"]) for row in cloth_allocations] == [1.0, 0.5]
    assert {row["observedNotCausal"] for row in windows} == {"True"}
    assert all(row["causalUnavailableReason"] for row in windows)
    reordered = copy.deepcopy([second, counterpart, first])
    reordered[2]["economicDiagnostics"]["trades"].reverse()
    reordered[2]["economicDiagnostics"]["physicalEvents"].reverse()
    alternate, alternate_output = tmp_path / "alternate", tmp_path / "alternate-out"
    write_artifact(alternate, reordered)
    economic_diagnostics.aggregate_trace_directories([alternate], alternate_output)
    for name in (
        "trade_provenance_lots.csv",
        "trade_provenance_allocations.csv",
        "trade_downstream_windows.csv",
    ):
        assert (output / name).read_bytes() == (alternate_output / name).read_bytes()


def test_unconsumed_import_expires_at_evaluation_horizon_without_double_counting(
    tmp_path: Path,
) -> None:
    row = transition_v2()
    row["economicDiagnostics"] = with_trade(row["economicDiagnostics"])
    counterpart = reciprocal_endpoint(row)
    artifact = tmp_path / "artifact"
    write_artifact(artifact, [row, counterpart])
    output = tmp_path / "out"
    economic_diagnostics.aggregate_trace_directories([artifact], output)
    with (output / "trade_provenance_lots.csv").open(newline="", encoding="utf-8") as handle:
        lots = list(csv.DictReader(handle))
    with (output / "trade_provenance_allocations.csv").open(newline="", encoding="utf-8") as handle:
        allocations = list(csv.DictReader(handle))
    cloth_lot = next(item for item in lots if item["item"] == "Cloth")
    assert cloth_lot["horizonStatus"] == "unconsumed-at-evaluation-horizon"
    assert float(cloth_lot["remainingQuantity"]) == 1.5
    assert allocations == []


def test_imported_lot_turn_gap_terminates_as_unresolved() -> None:
    first = transition_v2(turn=1)
    first["economicDiagnostics"] = with_trade(first["economicDiagnostics"])
    counterpart = reciprocal_endpoint(first)
    ninth = transition_v2(turn=9)
    _set_good_flow(ninth["economicDiagnostics"], "Cloth", opening=1.5, consumption=0.0, closing=1.5)
    lots, allocations, windows = economic_diagnostics._gate2_trade_provenance(
        [first, counterpart, ninth]
    )
    cloth = next(item for item in lots if item["item"] == "Cloth")
    assert cloth["horizonStatus"] == "unresolved-continuity"
    assert "non-adjacent-successor:1->9" in cloth["continuityUnavailableReason"]
    assert not [item for item in allocations if item["item"] == "Cloth"]
    assert 9 not in {int(item["observationTurn"]) for item in windows if item["item"] == "Cloth"}


def test_exported_or_transferred_import_is_not_reported_as_consumed() -> None:
    first = transition_v2(turn=1)
    first["economicDiagnostics"] = with_trade(first["economicDiagnostics"])
    counterpart = reciprocal_endpoint(first)
    first["economicDiagnostics"]["physicalEvents"].append(
        {
            "eventId": f"{first['transitionId']}/transfer/Cloth",
            "campId": first["campId"],
            "category": "transfer-out",
            "item": "Cloth",
            "unit": "good-units",
            "quantity": 1.5,
            "direction": "out",
        }
    )
    lots, allocations, _ = economic_diagnostics._gate2_trade_provenance([first, counterpart])
    cloth = next(item for item in lots if item["item"] == "Cloth")
    assert cloth["consumedQuantity"] == 0.0
    assert cloth["exportedQuantity"] == 0.0
    assert cloth["transferredQuantity"] == 1.5
    assert cloth["lostQuantity"] == 0.0
    assert cloth["totalRetiredQuantity"] == 1.5
    assert cloth["horizonStatus"] == "fully-retired-by-export-transfer-or-loss"
    assert {row["consumptionCategory"] for row in allocations if row["item"] == "Cloth"} == {
        "transfer-out"
    }


def test_after_production_tool_import_cannot_fund_work_time_service() -> None:
    row = transition_v2(turn=1)
    row["economicDiagnostics"] = with_trade(row["economicDiagnostics"])
    economic = row["economicDiagnostics"]
    transaction = next(item for item in economic["trades"] if item["recordType"] == "transaction")
    request = next(item for item in economic["trades"] if item["recordType"] == "requisition")
    request["wantedGoods"] = ["FarmTools"]
    incoming = next(item for item in transaction["legs"] if item["receiver"] == row["campId"])
    incoming["good"] = "FarmTools"
    incoming["deliveryPhase"] = "after-production"
    incoming["parts"][0]["item"] = "FarmTools"
    next(item for item in economic["goods"] if item["good"] == "FarmTools")["endingStock"] = 0.5
    economic["physicalEvents"].append(
        {
            "eventId": f"{row['trajectoryId']}/service/FarmTools",
            "campId": row["campId"],
            "category": "final-service-fulfilment",
            "item": "FarmTools",
            "unit": "good-units",
            "quantity": 1.0,
            "direction": "out",
            "phase": "work-before-production",
        }
    )

    lots, allocations, _ = economic_diagnostics._gate2_trade_provenance([row])
    tool_lot = next(item for item in lots if item["item"] == "FarmTools")
    assert not [item for item in allocations if item["item"] == "FarmTools"]
    assert tool_lot["horizonStatus"] == "unresolved-continuity"
    assert (
        tool_lot["continuityUnavailableReason"]
        == "retained-import-lots-exceed-authoritative-closing-stock"
    )


def test_after_production_import_cannot_fund_preproduction_wear() -> None:
    row = transition_v2(turn=1)
    row["economicDiagnostics"] = with_trade(row["economicDiagnostics"])
    economic = row["economicDiagnostics"]
    transaction = next(item for item in economic["trades"] if item["recordType"] == "transaction")
    incoming = next(item for item in transaction["legs"] if item["receiver"] == row["campId"])
    incoming["deliveryPhase"] = "after-production"
    next(item for item in economic["goods"] if item["good"] == "Cloth")["endingStock"] = 0.5
    economic["physicalEvents"].append(
        {
            "eventId": f"{row['trajectoryId']}/wear/Cloth",
            "campId": row["campId"],
            "category": "wear-depreciation",
            "item": "Cloth",
            "unit": "good-units",
            "quantity": 1.0,
            "direction": "out",
        }
    )

    lots, allocations, _ = economic_diagnostics._gate2_trade_provenance([row])
    cloth_lot = next(item for item in lots if item["item"] == "Cloth")
    assert not [item for item in allocations if item["item"] == "Cloth"]
    assert cloth_lot["horizonStatus"] == "unresolved-continuity"
    assert (
        cloth_lot["continuityUnavailableReason"]
        == "retained-import-lots-exceed-authoritative-closing-stock"
    )


def test_food_import_is_not_falsely_bound_to_aggregate_food_consumption() -> None:
    row = transition_v2(turn=1)
    row["economicDiagnostics"] = with_trade(row["economicDiagnostics"])
    economic = row["economicDiagnostics"]
    transaction = next(item for item in economic["trades"] if item["recordType"] == "transaction")
    request = next(item for item in economic["trades"] if item["recordType"] == "requisition")
    request["wantedGoods"] = ["Food"]
    incoming = next(item for item in transaction["legs"] if item["receiver"] == row["campId"])
    incoming["good"] = "Food"
    incoming["parts"][0]["item"] = "PlantFood"
    economic["domains"]["population"]["endingPlantFood"] = 1.5

    lots, allocations, windows = economic_diagnostics._gate2_trade_provenance([row])
    food_lot = next(item for item in lots if item["item"] == "PlantFood")
    for field in (
        "consumedQuantity",
        "remainingQuantity",
        "exportedQuantity",
        "transferredQuantity",
        "lostQuantity",
        "totalRetiredQuantity",
    ):
        assert food_lot[field] is None
    assert food_lot["consumptionAttributionStatus"] == "unresolved"
    assert food_lot["retirementAttributionStatus"] == "unresolved"
    assert food_lot["retirementAttributionUnavailableReason"]
    assert food_lot["horizonStatus"] == "unresolved-component-consumption"
    assert food_lot["horizonUnavailableReason"]
    assert not [item for item in allocations if item["item"] == "PlantFood"]
    food_windows = [item for item in windows if item["item"] == "PlantFood"]
    assert {item["physicalAttributionStatus"] for item in food_windows} == {"unresolved"}
    assert all(item["physicalAttributionUnavailableReason"] for item in food_windows)


def test_component_food_replacement_then_later_export_stays_unresolved() -> None:
    first = transition_v2(turn=1)
    first["economicDiagnostics"] = with_trade(first["economicDiagnostics"])
    first_economic = first["economicDiagnostics"]
    first_transaction = next(
        item for item in first_economic["trades"] if item["recordType"] == "transaction"
    )
    first_request = next(
        item for item in first_economic["trades"] if item["recordType"] == "requisition"
    )
    first_request["wantedGoods"] = ["Food"]
    incoming = next(
        item for item in first_transaction["legs"] if item["receiver"] == first["campId"]
    )
    incoming["good"] = "Food"
    incoming["parts"][0]["item"] = "PlantFood"
    first_economic["domains"]["population"]["endingPlantFood"] = 1.5
    replacement = next(
        event
        for event in first_economic["physicalEvents"]
        if event["category"] == "food-production-receipt"
        and event["item"] == "linked-food-reserves"
    )
    consumption = next(
        event
        for event in first_economic["physicalEvents"]
        if event["category"] == "human-food-consumption"
        and event["item"] == "linked-food-reserves"
    )
    replacement["quantity"] = 1.5
    consumption["quantity"] = 1.5

    second = transition_v2(turn=2)
    second_economic = second["economicDiagnostics"]
    second_economic["domains"]["population"]["openingPlantFood"] = 1.5
    second_economic["domains"]["population"]["endingPlantFood"] = 0.0
    second_economic["trades"].append(
        {
            "recordType": "transaction",
            "transactionId": f"{second['trajectoryId']}/turn/2/trade/export-food",
            "legs": [
                {
                    "legId": f"{second['trajectoryId']}/turn/2/trade/export-food/give",
                    "sender": second["campId"],
                    "receiver": 8,
                    "deliveryPhase": "opening",
                    "parts": [{"item": "PlantFood", "deliveredQuantity": 1.5}],
                }
            ],
        }
    )

    lots, allocations, _ = economic_diagnostics._gate2_trade_provenance([first, second])
    food_lot = next(item for item in lots if item["item"] == "PlantFood")
    assert food_lot["receivedQuantity"] == 1.5
    assert food_lot["exportedQuantity"] is None
    assert food_lot["totalRetiredQuantity"] is None
    assert food_lot["remainingQuantity"] is None
    assert food_lot["horizonStatus"] == "unresolved-component-consumption"
    assert food_lot["retirementAttributionUnavailableReason"]
    assert replacement["quantity"] == consumption["quantity"] == 1.5
    assert not [item for item in allocations if item["item"] == "PlantFood"]


def test_world_trade_closure_requires_both_endpoint_sidecars(tmp_path: Path) -> None:
    row = transition_v2()
    row["economicDiagnostics"] = with_trade(row["economicDiagnostics"])
    artifact = tmp_path / "one-endpoint"
    write_artifact(artifact, [row])
    with pytest.raises(ValueError, match="missing a reciprocal endpoint copy"):
        economic_diagnostics.aggregate_trace_directories([artifact], tmp_path / "out")
