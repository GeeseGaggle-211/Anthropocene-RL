"""Compact, actionable summary for one completed ``--economic-trace`` artifact.

This is descriptive training telemetry.  It neither ranks treatments nor treats
bootstrap/GAE target error as independent critic calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import economic_diagnostics
from .economic_accounting import CONTROL_PHASES, accounting_completeness
from .economic_accounting import SUPPORTED_SCHEMA_VERSIONS as ACCOUNTING_SCHEMAS
from .economic_trace import POLICY_INPUT_OUTLIER_LIMIT, canonical_bytes

SUMMARY_SCHEMA_VERSION = 1
_HEAD = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9]*)(?:\[(?P<index>\d+)\])?$")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _stats(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [number for value in values if (number := _number(value)) is not None]
    return {
        "count": len(numbers),
        "sum": math.fsum(numbers) if numbers else None,
        "mean": math.fsum(numbers) / len(numbers) if numbers else None,
        "absoluteMean": math.fsum(abs(number) for number in numbers) / len(numbers)
        if numbers
        else None,
        "minimum": min(numbers) if numbers else None,
        "maximum": max(numbers) if numbers else None,
    }


def _optional_stats(values: Iterable[Any]) -> dict[str, Any]:
    result = _stats(values)
    result["unavailableReason"] = None if result["count"] else "not-emitted-by-trace"
    return result


def _optional_boolean_stats(values: Iterable[Any]) -> dict[str, Any]:
    items = list(values)
    invalid = [value for value in items if value is not None and not isinstance(value, bool)]
    if invalid:
        raise ValueError("objective boolean diagnostic is not boolean/null")
    observed = [value for value in items if isinstance(value, bool)]
    true_count = sum(observed)
    return {
        "observedCount": len(observed),
        "unavailableCount": len(items) - len(observed),
        "trueCount": true_count if observed else None,
        "falseCount": len(observed) - true_count if observed else None,
        "trueFraction": true_count / len(observed) if observed else None,
        "unavailableReason": None if observed else "not-emitted-by-trace",
    }


def _paired_ratio(
    records: list[tuple[dict, dict]],
    source: str,
    destination: str,
    numerator_unit: str | None,
    denominator_unit: str | None,
) -> dict[str, Any]:
    pairs = [
        (float(source_value), float(destination_value))
        for _, control in records
        if (source_value := _number(control["phases"].get(source))) is not None
        and (destination_value := _number(control["phases"].get(destination))) is not None
    ]
    bottom = math.fsum(value[0] for value in pairs) if pairs else None
    top = math.fsum(value[1] for value in pairs) if pairs else None
    reason = None
    value = None
    if numerator_unit is None or denominator_unit is None:
        reason = "mixed-or-missing-unit"
    elif numerator_unit != denominator_unit:
        reason = "unit-mismatch"
    elif top is None or bottom is None:
        reason = "phase-value-unavailable"
    elif bottom == 0:
        reason = "zero-denominator"
    else:
        value = top / bottom
    return {
        "value": value,
        "numeratorSum": top,
        "denominatorSum": bottom,
        "pairedValueCount": len(pairs),
        "numeratorUnit": numerator_unit,
        "denominatorUnit": denominator_unit,
        "unavailableReason": reason,
    }


def _reason_values(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "reason" in str(key).lower() and isinstance(child, str) and child:
                yield path, child
            else:
                yield from _reason_values(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _reason_values(child, f"{prefix}[{index}]")


def _head_request(transition: dict[str, Any], actor_head: str) -> float | None:
    match = _HEAD.fullmatch(actor_head)
    if match is None:
        return None
    requested = transition.get("actionDiagnostics", {}).get("requested", {})
    raw = requested.get(match.group("name"))
    if not isinstance(raw, list):
        return None
    index = int(match.group("index") or 0)
    return _number(raw[index]) if index < len(raw) else None


def _control_semantic_key(control: dict[str, Any]) -> dict[str, Any]:
    domain = str(control["domain"])
    metadata = control.get("metadata", {})
    if domain == "manufacturing":
        return {
            "domain": domain,
            "actorHead": control["actorHead"],
            "service": metadata.get("service"),
            "serviceClass": metadata.get("serviceClass"),
            "good": metadata.get("good"),
            "rawMaterial": metadata.get("rawMaterial"),
            "role": metadata.get("role"),
            "nativeUnit": control["nativeUnit"],
        }
    if domain == "trade":
        key = {
            "domain": domain,
            "good": metadata.get("good"),
            "direction": metadata.get("direction"),
            "nativeUnit": control["nativeUnit"],
        }
        if key["good"] is None or key["direction"] is None:
            key["unresolvedRawIdentity"] = control["identity"]
        return key
    return {
        "domain": domain,
        "identity": control["identity"],
        "actorHead": control["actorHead"],
        "nativeUnit": control["nativeUnit"],
    }


def _economic_activity(control: dict[str, Any]) -> tuple[str, str]:
    """Add a readable activity without changing the native evidence domain."""
    domain = str(control["domain"])
    if domain in {"cultivation", "herding", "trade"}:
        return domain, f"native-domain:{domain}"
    metadata = control.get("metadata", {})
    role = str(metadata.get("role") or "").lower()
    role_activity = {
        "farmer": "cultivation",
        "herder": "herding",
        "fisher": "fishing",
        "gatherer": "gathering",
    }
    if role in role_activity:
        return role_activity[role], f"metadata.role:{metadata['role']}"
    service = str(metadata.get("service") or "").lower()
    for activity, markers in (
        ("cultivation", ("field", "farm")),
        ("herding", ("herd",)),
        ("fishing", ("fish",)),
        ("gathering", ("gather", "food reserve", "food growth")),
    ):
        if any(marker in service for marker in markers):
            return activity, f"metadata.service:{metadata.get('service')}"
    return domain, f"native-domain:{domain}; no more specific reviewed role/service mapping"


def _request_unmet_reason(transition: dict[str, Any], control: dict[str, Any]) -> str | None:
    for request in transition["economicDiagnostics"].get("requests", []):
        if request.get("requestId") == control.get("identity"):
            reason = request.get("unmetReason")
            return reason if isinstance(reason, str) and reason else None
    return None


def _request_grant_finding(
    semantic: dict[str, Any], records: list[tuple[dict, dict]]
) -> dict[str, Any]:
    activity, basis = _economic_activity(records[0][1])
    pairs = []
    for transition, control in records:
        request = _number(control["phases"].get("actorCappedRequest"))
        grant = _number(control["phases"].get("compilerGrant"))
        if request is not None and grant is not None:
            pairs.append((transition, control, request, grant))
    positive = [row for row in pairs if row[2] > 0]
    shortfall = [row for row in positive if row[3] < row[2] - 1e-12]
    zero_grant = [row for row in positive if abs(row[3]) <= 1e-12]
    partial_grant = [row for row in shortfall if row[3] > 1e-12]
    active = [row for row in pairs if row[1].get("classification") == "active"]
    active_positive = [row for row in active if row[2] > 0]
    active_shortfall = [row for row in active_positive if row[3] < row[2] - 1e-12]
    active_zero_grant = [row for row in active_positive if abs(row[3]) <= 1e-12]
    inactive_positive = [
        row for row in pairs if row[1].get("classification") == "inactive" and row[2] > 0
    ]
    inactive_denied = [row for row in inactive_positive if row[3] < row[2] - 1e-12]
    request_sum = math.fsum(row[2] for row in positive)
    grant_sum = math.fsum(row[3] for row in positive)
    active_request_sum = math.fsum(row[2] for row in active_positive)
    active_grant_sum = math.fsum(row[3] for row in active_positive)
    overall_reasons = Counter(
        reason
        for transition, control, _, _ in shortfall
        if (reason := _request_unmet_reason(transition, control)) is not None
    )
    reasons = Counter(
        reason
        for transition, control, _, _ in active_shortfall
        if (reason := _request_unmet_reason(transition, control)) is not None
    )
    missing_physical = {
        phase: sum(control["phases"].get(phase) is None for _, control, _, _ in shortfall)
        for phase in (
            "executedWorkerOrInputClaim",
            "committedPhysicalReceipt",
            "inventoryChange",
            "consumptionOrFinalService",
        )
    }
    active_missing_physical = {
        phase: sum(
            control["phases"].get(phase) is None for _, control, _, _ in active_shortfall
        )
        for phase in (
            "executedWorkerOrInputClaim",
            "committedPhysicalReceipt",
            "inventoryChange",
            "consumptionOrFinalService",
        )
    }
    request_units = {
        control.get("phaseUnits", {}).get("actorCappedRequest") for _, control in records
    }
    grant_units = {control.get("phaseUnits", {}).get("compilerGrant") for _, control in records}
    units_compatible = (
        len(request_units) == 1
        and len(grant_units) == 1
        and None not in request_units
        and request_units == grant_units
    )
    request_unit = next(iter(request_units)) if len(request_units) == 1 else None
    grant_unit = next(iter(grant_units)) if len(grant_units) == 1 else None

    def ratio(top: float, bottom: float) -> tuple[float | None, str | None]:
        if not units_compatible:
            return None, "mixed-missing-or-incompatible-units"
        if bottom <= 0:
            return None, "zero-or-missing-positive-request"
        return top / bottom, None

    def outcome_context(rows: list[tuple[dict, dict, float, float]]) -> dict[str, Any]:
        by_transition: dict[str, tuple[Any, Any]] = {}
        for transition, control, _, _ in rows:
            identity = str(transition["transitionId"])
            values = (
                control["phases"].get("endingPopulation"),
                control["phases"].get("endingQol"),
            )
            if identity in by_transition and by_transition[identity] != values:
                raise ValueError("control rows disagree on ending outcome context")
            by_transition[identity] = values
        return {
            "requestRowCount": len(rows),
            "transitionCount": len(by_transition),
            "endingPopulation": _stats(value[0] for value in by_transition.values()),
            "endingQol": _stats(value[1] for value in by_transition.values()),
            "interpretation": (
                "transition-deduplicated same-row context only; no causal relationship between "
                "request/grant shortfall and population or QoL is claimed"
            ),
        }

    overall_ratio, overall_ratio_reason = ratio(grant_sum, request_sum)
    active_ratio, active_ratio_reason = ratio(active_grant_sum, active_request_sum)
    display = str(
        semantic.get("identity")
        or semantic.get("service")
        or semantic.get("good")
        or semantic.get("role")
        or "unresolved"
    )
    qualifier = semantic.get("good") or semantic.get("role")
    if qualifier and str(qualifier).lower() not in display.lower():
        display += f" [{qualifier}]"
    return {
        "economicActivity": activity,
        "economicActivityBasis": basis,
        "nativeDomain": semantic["domain"],
        "semanticIdentity": display,
        "semanticKey": semantic,
        "nativeUnit": semantic["nativeUnit"],
        "actorHead": semantic.get("actorHead"),
        "pairedRequestGrantCount": len(pairs),
        "positiveRequestCount": len(positive),
        "requestShortfallCount": len(shortfall),
        "zeroGrantCount": len(zero_grant),
        "partialGrantCount": len(partial_grant),
        "activePairedRequestGrantCount": len(active),
        "activePositiveRequestCount": len(active_positive),
        "activeRequestShortfallCount": len(active_shortfall),
        "activeZeroGrantCount": len(active_zero_grant),
        "inactivePositiveRequestCount": len(inactive_positive),
        "inactiveDeniedRequestCount": len(inactive_denied),
        "requestSum": request_sum,
        "grantSum": grant_sum,
        "activeRequestSum": active_request_sum,
        "activeGrantSum": active_grant_sum,
        "requestUnit": request_unit,
        "grantUnit": grant_unit,
        "requestGrantUnitsCompatible": units_compatible,
        "grantToRequestRatio": overall_ratio,
        "grantToRequestRatioUnavailableReason": overall_ratio_reason,
        "activeGrantToRequestRatio": active_ratio,
        "activeGrantToRequestRatioUnavailableReason": active_ratio_reason,
        "compilerReasonCounts": dict(sorted(reasons.items())),
        "overallCompilerReasonCounts": dict(sorted(overall_reasons.items())),
        "missingPhysicalAttributionCountsAmongShortfalls": missing_physical,
        "missingPhysicalAttributionCountsAmongActiveShortfalls": active_missing_physical,
        "outcomeContextAmongShortfalls": outcome_context(shortfall),
        "outcomeContextAmongActiveShortfalls": outcome_context(active_shortfall),
        "exampleTransitionIds": sorted(
            {str(transition["transitionId"]) for transition, _, _, _ in active_shortfall}
        )[:5],
        "overallExampleTransitionIds": sorted(
            {str(transition["transitionId"]) for transition, _, _, _ in shortfall}
        )[:5],
    }


def _completeness_reason_key(reason: Any) -> str:
    text = str(reason)
    if text.startswith("control ") and " missing " in text:
        return "control-missing:" + text.split(" missing ", 1)[1]
    if text.startswith("trade transaction ") and " has no matched-intervention" in text:
        return "trade-transaction:matched-delayed-causal-effect-unavailable"
    return text


def _control_summary(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    semantic_keys: dict[str, dict[str, Any]] = {}
    reason_counts: Counter[str] = Counter()
    schema2_rows = 0
    complete_rows = 0
    completeness_reasons: Counter[str] = Counter()
    for transition in transitions:
        economic = transition["economicDiagnostics"]
        if economic.get("schemaVersion") not in ACCOUNTING_SCHEMAS:
            continue
        schema2_rows += 1
        completeness = accounting_completeness(economic)
        if completeness["complete"]:
            complete_rows += 1
        for reason in completeness["reasons"]:
            completeness_reasons[_completeness_reason_key(reason)] += 1
        manufacturing = economic.get("domains", {}).get("manufacturing", {})
        for funnel in manufacturing.get("funnels", []):
            for reason in funnel.get("constraintReasons", []):
                if isinstance(reason, str) and reason:
                    reason_counts[f"domain-funnel:{reason}"] += 1
        for control in economic["controls"]:
            semantic = _control_semantic_key(control)
            key = canonical_bytes(semantic).decode("utf-8")
            semantic_keys[key] = semantic
            groups[key].append((transition, control))
            if control["classificationReason"]:
                reason_counts[f"classification:{control['classificationReason']}"] += 1
            for phase, reason in control["unavailable"].items():
                reason_counts[f"unavailable:{phase}:{reason}"] += 1
            for path, reason in _reason_values(control.get("metadata", {})):
                reason_counts[f"metadata:{path}:{reason}"] += 1
            if reason := _request_unmet_reason(transition, control):
                reason_counts[f"compiler-request:{reason}"] += 1

    rows = []
    request_grant_findings = []
    chain = (
        ("nativeDemand", "actorCappedRequest"),
        ("actorCappedRequest", "compilerGrant"),
        ("compilerGrant", "committedPhysicalReceipt"),
        ("committedPhysicalReceipt", "inventoryChange"),
        ("inventoryChange", "consumptionOrFinalService"),
    )
    for key in sorted(groups):
        semantic = semantic_keys[key]
        domain = str(semantic["domain"])
        identity = str(
            semantic.get("identity")
            or semantic.get("service")
            or semantic.get("good")
            or semantic.get("rawMaterial")
            or semantic.get("role")
            or "unresolved"
        )
        unit = str(semantic["nativeUnit"])
        actor_head = str(semantic.get("actorHead", "unregistered"))
        records = groups[key]
        activity, activity_basis = _economic_activity(records[0][1])
        phase_units = {}
        for phase in CONTROL_PHASES:
            observed = sorted({str(control["phaseUnits"][phase]) for _, control in records})
            phase_units[phase] = observed[0] if len(observed) == 1 else None
        phase_stats = {
            phase: _stats(control["phases"].get(phase) for _, control in records)
            for phase in CONTROL_PHASES
        }
        control_reasons: Counter[str] = Counter()
        for transition, control in records:
            if control["classificationReason"]:
                control_reasons[f"classification:{control['classificationReason']}"] += 1
            for phase, reason in control["unavailable"].items():
                control_reasons[f"unavailable:{phase}:{reason}"] += 1
            for path, reason in _reason_values(control.get("metadata", {})):
                control_reasons[f"metadata:{path}:{reason}"] += 1
            if reason := _request_unmet_reason(transition, control):
                control_reasons[f"compiler-request:{reason}"] += 1
        rows.append(
            {
                "domain": domain,
                "identity": identity,
                "semanticKey": semantic,
                "economicActivity": activity,
                "economicActivityBasis": activity_basis,
                "nativeUnit": unit,
                "actorHead": actor_head,
                "controlRowCount": len(records),
                "transitionCount": len({row["transitionId"] for row, _ in records}),
                "rawIdentityCount": len({control["identity"] for _, control in records}),
                "rawIdentityExamples": sorted({control["identity"] for _, control in records})[:5],
                "classificationCounts": dict(
                    sorted(Counter(control["classification"] for _, control in records).items())
                ),
                "actorRequested": _stats(
                    _head_request(transition, actor_head) for transition, _ in records
                ),
                "phases": phase_stats,
                "phaseUnits": phase_units,
                "phaseRatios": {
                    f"{source}_to_{destination}": _paired_ratio(
                        records,
                        source,
                        destination,
                        phase_units[destination],
                        phase_units[source],
                    )
                    for source, destination in chain
                },
                "unavailablePhaseCounts": dict(
                    sorted(
                        Counter(
                            phase
                            for _, control in records
                            for phase in control["unavailable"]
                        ).items()
                    )
                ),
                "reasonCounts": dict(sorted(control_reasons.items())),
            }
        )
        request_grant_findings.append(_request_grant_finding(semantic, records))
    return {
        "schema2TransitionCount": schema2_rows,
        "completeAccountingTransitionCount": complete_rows,
        "accountingTransitionDenominator": schema2_rows,
        "accountingCompletenessReasons": dict(sorted(completeness_reasons.items())),
        "reasonCounts": dict(sorted(reason_counts.items())),
        "controls": rows,
        "requestGrantFindings": request_grant_findings,
        "unavailableReason": None if schema2_rows else "no-schema-2-economic-accounting-rows",
    }


def _trade_summary(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    domain_fields = (
        "requisitions",
        "honoured",
        "countered",
        "ignored",
        "settledTrades",
        "refund",
        "storageRejection",
    )
    domain_values: dict[str, list[Any]] = defaultdict(list)
    endpoint_types: Counter[str] = Counter()
    endpoint_outcomes: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    unique: dict[tuple[str, str], dict] = {}
    for transition in transitions:
        economic = transition["economicDiagnostics"]
        trade = economic.get("domains", {}).get("trade", {})
        for field in domain_fields:
            domain_values[field].append(trade.get(field))
        for record in economic.get("trades", []):
            kind = str(record.get("recordType", "unknown"))
            endpoint_types[kind] += 1
            if isinstance(record.get("outcome"), str):
                endpoint_outcomes[str(record["outcome"])] += 1
            for field in ("gatingReason", "partnerSearchReason", "rejectionReason"):
                reason = record.get(field)
                if isinstance(reason, str) and reason:
                    reason_counts[f"{field}:{reason}"] += 1
            identity_field = {
                "eligibility": "eventId",
                "requisition": "requestId",
                "transaction": "transactionId",
            }.get(kind)
            identity = record.get(identity_field) if identity_field else None
            if isinstance(identity, str):
                key = (
                    str(transition.get("runId")),
                    str(transition.get("worldSeed")),
                    str(transition.get("trajectoryId")),
                    kind,
                    identity,
                )
                if key in unique and canonical_bytes(unique[key]) != canonical_bytes(record):
                    raise ValueError("trade lifecycle copies differ for one stable identity")
                unique.setdefault(key, record)
    unique_types = Counter(key[3] for key in unique)
    unique_outcomes = Counter(
        str(record["outcome"])
        for record in unique.values()
        if isinstance(record.get("outcome"), str)
    )
    refunded, storage_blocked = [], []
    for key, record in unique.items():
        kind = key[3]
        if kind != "transaction":
            continue
        for leg in record.get("legs", []):
            refunded.append(leg.get("refundedOrReturnedQuantity"))
            storage_blocked.append(leg.get("storageBlockedQuantity"))
    return {
        "domainEndpointCounts": {field: _stats(values) for field, values in domain_values.items()},
        "endpointRecordTypeCounts": dict(sorted(endpoint_types.items())),
        "uniqueLifecycleRecordTypeCounts": dict(sorted(unique_types.items())),
        "endpointRequisitionOutcomeCounts": dict(sorted(endpoint_outcomes.items())),
        "uniqueRequisitionOutcomeCounts": dict(sorted(unique_outcomes.items())),
        "reasonCounts": dict(sorted(reason_counts.items())),
        "uniqueTransactionRefundOrReturn": _stats(refunded),
        "uniqueTransactionStorageRejection": _stats(storage_blocked),
        "countingCaveat": (
            "domain counters and endpoint records are camp-endpoint observations; unique lifecycle "
            "counts de-duplicate only stable typed IDs within this trace"
        ),
    }


def _physical_event_summary(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exact event semantics without inferring absent-event coverage."""
    typed_items = {"PlantFood", "AnimalFood", "AquaticFood"}
    groups: dict[tuple[str, str, str, str, str | None], list[tuple[str, dict]]] = (
        defaultdict(list)
    )
    schema2_transitions = 0
    typed_transition_ids: set[str] = set()
    typed_event_count = 0
    for transition in transitions:
        economic = transition["economicDiagnostics"]
        if economic.get("schemaVersion") not in ACCOUNTING_SCHEMAS:
            continue
        schema2_transitions += 1
        transition_id = str(transition["transitionId"])
        for event in economic.get("physicalEvents", []):
            key = (
                str(event["category"]),
                str(event["item"]),
                str(event["unit"]),
                str(event["direction"]),
                str(event["phase"]) if event.get("phase") is not None else None,
            )
            groups[key].append((transition_id, event))
            if key[0] == "human-food-consumption" and key[1] in typed_items:
                typed_event_count += 1
                typed_transition_ids.add(transition_id)

    rows = []
    for key in sorted(groups, key=lambda value: tuple(item or "" for item in value)):
        category, item, unit, direction, phase = key
        records = groups[key]
        event_class = (
            "typed-camp-food-component"
            if category == "human-food-consumption" and item in typed_items
            else "linked-food-aggregate"
            if category == "human-food-consumption" and item == "linked-food-reserves"
            else "other-physical-event"
        )
        rows.append(
            {
                "category": category,
                "item": item,
                "unit": unit,
                "direction": direction,
                "phase": phase,
                "eventClass": event_class,
                "eventCount": len(records),
                "transitionCount": len({transition_id for transition_id, _ in records}),
                "quantity": _stats(event["quantity"] for _, event in records),
            }
        )

    missing_typed = schema2_transitions - len(typed_transition_ids)
    return {
        "grouping": ["category", "item", "unit", "direction", "phase"],
        "schema2TransitionDenominator": schema2_transitions,
        "eventCount": sum(len(records) for records in groups.values()),
        "groupCount": len(rows),
        "groups": rows,
        "typedCampFoodConsumption": {
            "registeredItems": sorted(typed_items),
            "positiveEventCount": typed_event_count,
            "transitionsWithObservedPositiveTypedEvent": len(typed_transition_ids),
            "transitionsWithoutObservedPositiveTypedEvent": missing_typed,
            "transitionDenominator": schema2_transitions,
            "coverage": None,
            "coverageUnavailableReason": (
                "physical-event rows contain positive mutations only and have no per-transition "
                "capability marker; absence can mean zero consumption or unsupported older capture"
            ),
        },
        "linkedFoodAggregateKeptSeparate": True,
        "sourceAttribution": None,
        "sourceAttributionUnavailableReason": (
            "physical mutation rows do not bind typed consumption to source requests or receipts"
        ),
    }


def _outcome_summary(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    populations, qols, population_years, qol_years, trade = [], [], [], [], []
    rewards: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    final_advantages = []
    for row in transitions:
        economic = row["economicDiagnostics"]
        population = economic.get("domains", {}).get("population", {})
        pop = population.get("endingLinked")
        qol = population.get("qol")
        years = row.get("years")
        populations.append(pop)
        qols.append(qol)
        population_years.append(population.get("personYears"))
        if _number(qol) is not None and _number(years) is not None:
            qol_years.append(float(qol) * float(years))
        trade.append(economic.get("domains", {}).get("trade", {}).get("realisedUtility"))
        final_advantages.append(row.get("finalPolicyAdvantage"))
        for name, objective in row.get("objectives", {}).items():
            if not isinstance(objective, dict):
                continue
            for field in (
                "reward",
                "shapedReward",
                "prediction",
                "target",
                "tdResidual",
                "advantage",
                "weightedAdvantageContribution",
                "preClipEffectiveCombinationWeight",
                "effectiveCombinationWeight",
            ):
                rewards[str(name)][field].append(objective.get(field))
            rewards[str(name)]["zeroShapedRewardNonzeroBootstrap"].append(
                objective.get("zeroShapedRewardNonzeroBootstrap")
            )

    objective_rows = {}
    for name in sorted(rewards):
        values = rewards[name]
        advantages = [value for value in values["advantage"] if _number(value) is not None]
        legacy_fields = {
            "reward",
            "shapedReward",
            "prediction",
            "target",
            "tdResidual",
            "advantage",
        }
        objective_rows[name] = {
            **{
                field: _stats(items) if field in legacy_fields else _optional_stats(items)
                for field, items in sorted(values.items())
                if field != "zeroShapedRewardNonzeroBootstrap"
            },
            "zeroShapedRewardNonzeroBootstrap": _optional_boolean_stats(
                values["zeroShapedRewardNonzeroBootstrap"]
            ),
            "advantageSignCounts": {
                "negative": sum(float(value) < 0 for value in advantages),
                "zero": sum(float(value) == 0 for value in advantages),
                "positive": sum(float(value) > 0 for value in advantages),
                "denominator": len(advantages),
            },
        }
    weighted_qol = None
    weighted_qol_denominator = 0.0
    weighted_qol_numerator = 0.0
    for population, qol in zip(populations, qols, strict=True):
        if (pop := _number(population)) is not None and (quality := _number(qol)) is not None:
            weighted_qol_denominator += pop
            weighted_qol_numerator += pop * quality
    if weighted_qol_denominator > 0:
        weighted_qol = weighted_qol_numerator / weighted_qol_denominator
    return {
        "transitionCount": len(transitions),
        "endingPopulation": _stats(populations),
        "endingQol": _stats(qols),
        "populationWeightedEndingQol": {
            "value": weighted_qol,
            "populationDenominator": weighted_qol_denominator,
            "unavailableReason": None if weighted_qol is not None else "zero-or-missing-population",
        },
        "populationYears": _stats(population_years),
        "qolYears": _stats(qol_years),
        "tradeUtility": _stats(trade),
        "tradeLifecycle": _trade_summary(transitions),
        "objectives": objective_rows,
        "finalPolicyAdvantage": _stats(final_advantages),
        "criticInterpretation": "internal-bootstrap-or-GAE-target-error-only",
        "independentCriticCalibration": None,
        "independentCriticCalibrationUnavailableReason": (
            "economic trace has no independently generated continuation-return target"
        ),
    }


def _debug_rows(debug: Any, batch_ids: list[str]) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(debug, dict):
        return [], "optimizerDebug-not-captured"
    if debug.get("schemaVersion") != 1 or debug.get("enabled") is not True:
        raise ValueError("unsupported optimizerDebug schema")
    for group in ("scope", "lossContributions", "actualClip"):
        if not isinstance(debug.get(group), dict):
            raise ValueError(f"optimizerDebug missing {group}")
    for group in ("lossContributions", "actualClip"):
        for name, value in debug[group].items():
            if group == "actualClip" and name in ("actorStepPerformed", "actorStepRolledBack"):
                if not isinstance(value, bool):
                    raise ValueError(f"optimizerDebug actualClip.{name} is not boolean")
                continue
            if value is not None and _number(value) is None:
                raise ValueError(f"optimizerDebug {group}.{name} is not finite numeric/null")
    shift = debug.get("sampledPolicyShift")
    if shift is not None:
        if not isinstance(shift, dict):
            raise ValueError("optimizerDebug sampledPolicyShift is malformed")
        row_count = shift.get("rowCount")
        numeric_fields = (
            "nonnegativeKlEstimateMean",
            "logRatioMean",
            "logRatioMin",
            "logRatioP10",
            "logRatioMedian",
            "logRatioP90",
            "logRatioMax",
        )
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count != len(batch_ids)
            or any(_number(shift.get(field)) is None for field in numeric_fields)
            or not isinstance(shift.get("interpretation"), str)
            or not shift["interpretation"]
        ):
            raise ValueError("optimizerDebug sampledPolicyShift is malformed")
        if float(shift["nonnegativeKlEstimateMean"]) < 0:
            raise ValueError("optimizerDebug sampledPolicyShift KL estimate is negative")
        ordered = [
            float(shift[field])
            for field in (
                "logRatioMin",
                "logRatioP10",
                "logRatioMedian",
                "logRatioP90",
                "logRatioMax",
            )
        ]
        if (
            ordered != sorted(ordered)
            or not ordered[0] <= float(shift["logRatioMean"]) <= ordered[-1]
        ):
            raise ValueError("optimizerDebug sampledPolicyShift order is inconsistent")
        if shift.get("measurementPhase", "pre-actor-step") != "pre-actor-step":
            raise ValueError("optimizerDebug sampledPolicyShift phase is not pre-actor-step")
    same = debug.get("sameSampledAction")
    if same is None:
        return [], "per-transition-action-probability-audit-not-selected"
    if not isinstance(same, dict) or not isinstance(same.get("rows"), list):
        raise ValueError("optimizerDebug sameSampledAction rows absent")
    rows = same["rows"]
    if len(rows) > 64:
        raise ValueError("optimizerDebug action-probability audit exceeds 64 rows")
    expected = debug["scope"].get("auditTransitionIds")
    identities = [row.get("transitionId") for row in rows]
    if (
        not all(isinstance(identity, str) for identity in identities)
        or len(identities) != len(set(identities))
        or any(identity not in batch_ids for identity in identities)
        or not isinstance(expected, list)
        or expected != identities
        or debug["scope"].get("sampledRowCount") != len(rows)
    ):
        raise ValueError("optimizerDebug action audit identity/count differs")
    for row in rows:
        for field in ("jointBefore", "jointAfter", "jointDelta"):
            if _number(row.get(field)) is None:
                raise ValueError(f"optimizerDebug action audit missing {field}")
        if not math.isclose(
            float(row["jointAfter"]) - float(row["jointBefore"]),
            float(row["jointDelta"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("optimizerDebug joint action delta differs")
        if not isinstance(row.get("perHead"), dict):
            raise ValueError("optimizerDebug per-head action audit absent")
        for values in row["perHead"].values():
            if not isinstance(values, dict) or any(
                _number(values.get(field)) is None for field in ("before", "after", "delta")
            ):
                raise ValueError("optimizerDebug per-head action audit malformed")
            if not math.isclose(
                float(values["after"]) - float(values["before"]),
                float(values["delta"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError("optimizerDebug per-head action delta differs")
        coordinates = row.get("perCoordinate", {})
        if not isinstance(coordinates, dict):
            raise ValueError("optimizerDebug per-coordinate action audit malformed")
        for group, values in coordinates.items():
            if not isinstance(group, str) or not isinstance(values, dict):
                raise ValueError("optimizerDebug per-coordinate group malformed")
            for label, coordinate in values.items():
                if not isinstance(label, str) or not isinstance(coordinate, dict):
                    raise ValueError("optimizerDebug per-coordinate row malformed")
                if any(
                    _number(coordinate.get(field)) is None
                    for field in (
                        "before",
                        "after",
                        "delta",
                        "beforeEntropy",
                        "afterEntropy",
                        "entropyDelta",
                    )
                ) or not isinstance(coordinate.get("policyScored"), bool) or coordinate.get(
                    "nativeRecordedActive"
                ) not in (True, False, None):
                    raise ValueError("optimizerDebug per-coordinate values malformed")
                relation = coordinate.get("activityRelation")
                expected_relation = (
                    "not-policy-scored"
                    if not coordinate["policyScored"]
                    else "native-recorded-activity-unavailable"
                    if coordinate["nativeRecordedActive"] is None
                    else "scored-and-native-recorded-active"
                    if coordinate["nativeRecordedActive"]
                    else "scored-but-native-recorded-inactive"
                )
                if relation != expected_relation:
                    raise ValueError("optimizerDebug per-coordinate activity relation absent")
                if not math.isclose(
                    float(coordinate["after"]) - float(coordinate["before"]),
                    float(coordinate["delta"]),
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    raise ValueError("optimizerDebug per-coordinate action delta differs")
    post_step = same.get("postStepSampledPolicyShift")
    if post_step is not None:
        if not isinstance(post_step, dict):
            raise ValueError("optimizerDebug post-step sampled policy shift is malformed")
        ratio = post_step.get("sampledProbabilityRatio")
        if (
            not rows
            or post_step.get("measurementPhase") != "post-actor-step"
            or post_step.get("rowSelection") != "bounded-fixed-audit-rows"
            or post_step.get("rowCount") != len(rows)
            or not isinstance(post_step.get("interpretation"), str)
            or not post_step["interpretation"]
            or not isinstance(ratio, dict)
            or ratio.get("definition") != "exp(jointAfter-jointBefore)"
            or ratio.get("count") != len(rows)
            or any(
                _number(value) is None
                for value in (
                    post_step.get("signedApproxKlMean"),
                    post_step.get("nonnegativeKlEstimateMean"),
                    ratio.get("min"),
                    ratio.get("max"),
                )
            )
        ):
            raise ValueError("optimizerDebug post-step sampled policy shift is malformed")
        deltas = [float(row["jointDelta"]) for row in rows]
        expected_signed = math.fsum(-delta for delta in deltas) / len(deltas)
        expected_nonnegative = math.fsum(
            math.expm1(delta) - delta for delta in deltas
        ) / len(deltas)
        expected_ratios = [math.exp(delta) for delta in deltas]
        if (
            float(post_step["nonnegativeKlEstimateMean"]) < 0
            or not math.isclose(
                float(post_step["signedApproxKlMean"]),
                expected_signed,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(post_step["nonnegativeKlEstimateMean"]),
                expected_nonnegative,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(ratio["min"]), min(expected_ratios), rel_tol=1e-6, abs_tol=1e-6
            )
            or not math.isclose(
                float(ratio["max"]), max(expected_ratios), rel_tol=1e-6, abs_tol=1e-6
            )
        ):
            raise ValueError(
                "optimizerDebug post-step sampled policy shift differs from audit rows"
            )
    return rows, None


def _update_coverage(
    transitions: list[dict[str, Any]], batches: list[tuple[tuple[Any, ...], list[str]]]
) -> list[dict[str, Any]]:
    """Describe actual minibatch draws without assuming one visit per epoch."""
    grouped: dict[tuple[Any, ...], list[list[str]]] = defaultdict(list)
    for identity, batch_ids in batches:
        grouped[identity].append(batch_ids)
    rows = []
    for identity in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        run_id, arm, optimizer_seed, rollout, epoch = identity
        rollout_ids = {
            str(row["transitionId"])
            for row in transitions
            if row.get("runId") == run_id
            and row.get("arm") == arm
            and row.get("optimizerSeed") == optimizer_seed
            and row.get("rollout") == rollout
        }
        draws = [item for batch in grouped[identity] for item in batch]
        visited = set(draws)
        repeated = len(draws) - len(visited)
        not_visited = sorted(rollout_ids - visited)
        repeated_ids = sorted(identity for identity, count in Counter(draws).items() if count > 1)
        rows.append(
            {
                "runId": run_id,
                "arm": arm,
                "optimizerSeed": optimizer_seed,
                "rollout": rollout,
                "epoch": epoch,
                "rolloutTransitionCount": len(rollout_ids),
                "minibatchCount": len(grouped[identity]),
                "batchDrawCount": len(draws),
                "uniqueTransitionCount": len(visited),
                "repeatedDrawCount": repeated,
                "notVisitedTransitionCount": len(not_visited),
                "visitedFraction": len(visited) / len(rollout_ids) if rollout_ids else None,
                "completeSingleVisit": (
                    bool(rollout_ids) and not repeated and visited == rollout_ids
                ),
                "exampleRepeatedTransitionIds": repeated_ids[:5],
                "exampleNotVisitedTransitionIds": not_visited[:5],
            }
        )
    return rows


def _update_summary(
    transitions: list[dict[str, Any]], updates: list[dict[str, Any]]
) -> dict[str, Any]:
    joined = economic_diagnostics.optimizer_update_rows(transitions, updates)
    transition_by_id = {row["transitionId"]: row for row in transitions}
    completions = {
        (
            row["runId"],
            row["arm"],
            row["optimizerSeed"],
            row["rollout"],
            row["epoch"],
            row.get("minibatch", 0),
            row["batchRowDigest"],
        ): row
        for row in updates
        if row.get("event") == "mutation-complete"
    }
    alignment = Counter()
    audited = 0
    per_head: dict[str, Counter] = defaultdict(Counter)
    per_coordinate: dict[str, Counter] = defaultdict(Counter)
    contexts = []
    unavailable = Counter()
    coverage_batches: list[tuple[tuple[Any, ...], list[str]]] = []
    policy_shifts: list[dict[str, Any]] = []
    post_step_policy_shifts: list[dict[str, Any]] = []
    for update in joined:
        if update["updateKind"] != "mutation":
            continue
        key = (
            update["runId"],
            update["arm"],
            update["optimizerSeed"],
            update["rollout"],
            update["epoch"],
            update["minibatch"],
            update["batchRowDigest"],
        )
        complete = completions[key]
        batch_ids = json.loads(update["batchTransitionIds"])
        coverage_batches.append(
            (
                (
                    update["runId"],
                    update["arm"],
                    update["optimizerSeed"],
                    update["rollout"],
                    update["epoch"],
                ),
                batch_ids,
            )
        )
        rows, reason = _debug_rows(complete.get("optimizerDebug"), batch_ids)
        debug = complete.get("optimizerDebug")
        if isinstance(debug, dict):
            shift = debug.get("sampledPolicyShift")
            if isinstance(shift, dict):
                policy_shifts.append(shift)
            else:
                unavailable["sampled-policy-shift-not-captured"] += 1
            same = debug.get("sameSampledAction")
            post_step_shift = (
                same.get("postStepSampledPolicyShift") if isinstance(same, dict) else None
            )
            if isinstance(post_step_shift, dict):
                post_step_policy_shifts.append(post_step_shift)
            else:
                unavailable["post-step-sampled-policy-shift-not-captured"] += 1
            contexts.append(
                {
                    "runId": update["runId"],
                    "rollout": update["rollout"],
                    "epoch": update["epoch"],
                    "minibatch": update["minibatch"],
                    "batchTransitionCount": update["batchTransitionCount"],
                    "lossContributions": debug.get("lossContributions"),
                    "actualClip": debug.get("actualClip"),
                    "actorGradientGeometry": debug.get("actorGradientGeometry"),
                    "vectorCriticSharedEncoderGradientGeometry": debug.get(
                        "vectorCriticSharedEncoderGradientGeometry"
                    ),
                    "criticHeadDiagnostics": debug.get("criticHeadDiagnostics"),
                    "sampledPolicyShift": shift,
                    "postStepSampledPolicyShift": post_step_shift,
                    "signedApproxKl": update.get("approxKl"),
                    "attributionScope": debug.get("attributionScope"),
                }
            )
        if reason is not None:
            unavailable[reason] += 1
            continue
        for action in rows:
            transition = transition_by_id[action["transitionId"]]
            advantage = _number(transition.get("finalPolicyAdvantage"))
            if advantage is None:
                unavailable["finalPolicyAdvantage-missing"] += 1
                continue
            audited += 1
            delta = float(action["jointDelta"])
            if delta == 0:
                alignment["unchangedSampledActionLogProbability"] += 1
            elif advantage < 0 and delta > 0:
                alignment["negativeAdvantageReinforcement"] += 1
            elif advantage > 0 and delta < 0:
                alignment["positiveAdvantageSuppression"] += 1
            elif advantage < 0:
                alignment["negativeAdvantageAligned"] += 1
            elif advantage > 0:
                alignment["positiveAdvantageAligned"] += 1
            else:
                alignment["zeroAdvantage"] += 1
            for head, values in action["perHead"].items():
                head_delta = float(values["delta"])
                per_head[str(head)]["denominator"] += 1
                if head_delta == 0:
                    per_head[str(head)]["unchangedSampledActionLogProbability"] += 1
                elif advantage < 0 and head_delta > 0:
                    per_head[str(head)]["negativeAdvantageReinforcement"] += 1
                if advantage > 0 and head_delta < 0:
                    per_head[str(head)]["positiveAdvantageSuppression"] += 1
            for group, coordinates in action.get("perCoordinate", {}).items():
                for label, values in coordinates.items():
                    key = f"{group}.{label}"
                    coordinate = per_coordinate[key]
                    coordinate["denominator"] += 1
                    coordinate["policyScored" if values["policyScored"] else "policyNotScored"] += 1
                    coordinate[
                        "nativeRecordedActivityUnavailable"
                        if values["nativeRecordedActive"] is None
                        else "nativeRecordedActive"
                        if values["nativeRecordedActive"]
                        else "nativeRecordedInactive"
                    ] += 1
                    coordinate[f"activityRelation:{values['activityRelation']}"] += 1
                    delta = float(values["delta"])
                    if not values["policyScored"]:
                        continue
                    coordinate["signAlignmentDenominator"] += 1
                    if delta == 0:
                        coordinate["unchangedSampledActionLogProbability"] += 1
                    elif advantage < 0 and delta > 0:
                        coordinate["negativeAdvantageReinforcement"] += 1
                    elif advantage > 0 and delta < 0:
                        coordinate["positiveAdvantageSuppression"] += 1
    shift_rows = sum(int(row["rowCount"]) for row in policy_shifts)
    sampled_policy_shift = {
        "measurementPhase": "pre-actor-step",
        "capturedMinibatchCount": len(policy_shifts),
        "rowCount": shift_rows,
        "nonnegativeKlEstimateMean": (
            math.fsum(
                float(row["nonnegativeKlEstimateMean"]) * int(row["rowCount"])
                for row in policy_shifts
            )
            / shift_rows
            if shift_rows
            else None
        ),
        "logRatioMean": (
            math.fsum(float(row["logRatioMean"]) * int(row["rowCount"]) for row in policy_shifts)
            / shift_rows
            if shift_rows
            else None
        ),
        "interpretation": (
            "sampled-action policy shift; nonnegative KL estimate is separate from the retained "
            "signed approximate-KL early-stop diagnostic"
        ),
    }
    post_step_rows = sum(int(row["rowCount"]) for row in post_step_policy_shifts)
    post_step_ratio_count = sum(
        int(row["sampledProbabilityRatio"]["count"])
        for row in post_step_policy_shifts
    )
    post_step_sampled_policy_shift = (
        {
            "measurementPhase": "post-actor-step",
            "rowSelection": "bounded-fixed-audit-rows",
            "capturedMinibatchCount": len(post_step_policy_shifts),
            "rowCount": post_step_rows,
            "signedApproxKlMean": (
                math.fsum(
                    float(row["signedApproxKlMean"]) * int(row["rowCount"])
                    for row in post_step_policy_shifts
                )
                / post_step_rows
            ),
            "nonnegativeKlEstimateMean": (
                math.fsum(
                    float(row["nonnegativeKlEstimateMean"]) * int(row["rowCount"])
                    for row in post_step_policy_shifts
                )
                / post_step_rows
            ),
            "sampledProbabilityRatio": {
                "definition": "exp(jointAfter-jointBefore)",
                "count": post_step_ratio_count,
                "min": min(
                    float(row["sampledProbabilityRatio"]["min"])
                    for row in post_step_policy_shifts
                ),
                "max": max(
                    float(row["sampledProbabilityRatio"]["max"])
                    for row in post_step_policy_shifts
                ),
            },
            "interpretation": (
                "post-step shift on bounded fixed audit rows and their sampled actions; "
                "not exact KL, not a full-rollout estimate, and not used for early stopping"
            ),
        }
        if post_step_rows
        else None
    )
    return {
        "joinedMutationCount": sum(row["updateKind"] == "mutation" for row in joined),
        "measurement": "same-state same-sampled-action log-probability delta",
        "auditedTransitionCount": audited,
        "alignmentDenominator": audited,
        "descriptiveSignAlignment": {
            name: alignment[name]
            for name in (
                "negativeAdvantageReinforcement",
                "positiveAdvantageSuppression",
                "negativeAdvantageAligned",
                "positiveAdvantageAligned",
                "zeroAdvantage",
                "unchangedSampledActionLogProbability",
            )
        },
        "perHeadDescriptiveSignAlignment": {
            head: dict(sorted(values.items())) for head, values in sorted(per_head.items())
        },
        "perCoordinateDescriptiveSignAlignment": {
            coordinate: dict(sorted(values.items()))
            for coordinate, values in sorted(per_coordinate.items())
        },
        "nativeRecordedActiveMaskSemantics": {
            "strategic": "opening capability",
            "service": (
                "post-request ServiceRequested > 1e-9; action-dependent, not exogenous feasibility"
            ),
            "routes": "both native routes recorded feasible",
            "trade": "economic eligibility",
            "materialsService": "disabled when the raw-material feature flag is off",
            "automaticPolicyMaskingRecommendation": None,
            "interpretation": (
                "descriptive native activity only; native-recorded-inactive does not establish "
                "that a coordinate is always ineffective"
            ),
        },
        "updateCoverage": {
            "unit": "transition-identity draws per rollout and epoch",
            "rows": _update_coverage(transitions, coverage_batches),
            "interpretation": (
                "descriptive random-minibatch coverage; repeats and unvisited rows are exposed "
                "rather than treated as a full epoch traversal"
            ),
        },
        "sampledPolicyShift": sampled_policy_shift,
        "postStepSampledPolicyShift": post_step_sampled_policy_shift,
        "postStepSampledPolicyShiftUnavailableReason": (
            None
            if post_step_sampled_policy_shift is not None
            else "not-emitted-by-trace"
        ),
        "teacherAndGradientContext": contexts,
        "unknownCounts": dict(sorted(unavailable.items())),
        "positiveAdvantageBenefit": None,
        "positiveAdvantageBenefitUnavailableReason": (
            "probability alignment is not an independently measured downstream benefit"
        ),
        "causalAttribution": None,
        "causalAttributionUnavailableReason": (
            "one optimizer step combines policy, teacher, entropy and optional search losses; "
            "reported component geometry is pre-Adam descriptive context"
        ),
    }


def _policy_input_health_summary(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    missing = Counter()
    absent_legacy = Counter()
    group_totals: dict[str, dict[str, Any]] = {}
    known: dict[int, dict[str, Any]] = {}
    top: list[dict[str, Any]] = []
    rows_with = 0
    actor_nonfinite = 0
    normalized_nonfinite = 0
    maximum = None

    def nonnegative_count(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"policy input health {field} is invalid")
        return value

    def observe_coordinate(transition_id: str, value: dict[str, Any]) -> None:
        index = value.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("policy input health coordinate index is invalid")
        entry = known.setdefault(
            index,
            {
                "index": index,
                "label": value.get("label", f"core[{index}]"),
                "observationCount": 0,
                "nonfiniteNormalizedCount": 0,
                "maximumAbsoluteNormalized": None,
                "maximumRecord": None,
            },
        )
        if entry["label"] != value.get("label", entry["label"]):
            raise ValueError("policy input health coordinate label changed")
        entry["observationCount"] += 1
        normalized = _number(value.get("normalized"))
        if normalized is None:
            entry["nonfiniteNormalizedCount"] += 1
            return
        magnitude = abs(normalized)
        if (
            entry["maximumAbsoluteNormalized"] is None
            or magnitude > entry["maximumAbsoluteNormalized"]
        ):
            entry["maximumAbsoluteNormalized"] = magnitude
            entry["maximumRecord"] = {
                "transitionId": transition_id,
                "raw": value.get("raw"),
                "transformed": value.get("transformed"),
                "normalizerMean": value.get("normalizerMean"),
                "normalizerStd": value.get("normalizerStd"),
                "affineBeforeGuard": value.get("affineBeforeGuard"),
                "normalized": normalized,
                "corePresent": value.get("corePresent"),
                "guardEnabled": value.get("guardEnabled"),
                "guardLimit": value.get("guardLimit"),
                "dead": value.get("dead"),
                "clippedByGuard": value.get("clippedByGuard"),
                "zeroedByDeadMask": value.get("zeroedByDeadMask"),
                "zeroedByPresenceMask": value.get("zeroedByPresenceMask"),
            }

    for transition in transitions:
        transition_id = str(transition["transitionId"])
        health = transition.get("policyInputHealth")
        if health is None:
            reason = transition.get("unavailable", {}).get(
                "policyInputHealth", "not-emitted-by-legacy-economic-trace"
            )
            missing[str(reason)] += 1
            continue
        if (
            not isinstance(health, dict)
            or health.get("schemaVersion") != 1
            or health.get("source") != "exact-actor-input-tensors"
        ):
            raise ValueError("policy input health schema/source is invalid")
        rows_with += 1
        actor_nonfinite += nonnegative_count(
            health.get("actorFloatInputNonfiniteCount"), "actor nonfinite count"
        )
        normalized_nonfinite += nonnegative_count(
            health.get("normalizedInputNonfiniteCount"), "normalized nonfinite count"
        )
        row_maximum = _number(health.get("maximumAbsoluteNormalized"))
        if row_maximum is not None:
            if row_maximum < 0:
                raise ValueError("policy input health maximum is negative")
            maximum = row_maximum if maximum is None else max(maximum, row_maximum)

        groups = health.get("groups")
        if not isinstance(groups, list):
            raise ValueError("policy input health groups are absent")
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("group"), str):
                raise ValueError("policy input health group is invalid")
            name = group["group"]
            target = group_totals.setdefault(
                name,
                {
                    "group": name,
                    "semantics": group.get("semantics"),
                    "rowCount": 0,
                    "scalarCount": 0,
                    "finiteCount": 0,
                    "nonfiniteCount": 0,
                    "maximumAbsolute": None,
                },
            )
            if target["semantics"] != group.get("semantics"):
                raise ValueError("policy input health group semantics changed")
            target["rowCount"] += 1
            for field in ("scalarCount", "finiteCount", "nonfiniteCount"):
                target[field] += nonnegative_count(group.get(field), f"group {field}")
            group_maximum = _number(group.get("maximumAbsolute"))
            if group_maximum is not None:
                if group_maximum < 0:
                    raise ValueError("policy input health group maximum is negative")
                target["maximumAbsolute"] = (
                    group_maximum
                    if target["maximumAbsolute"] is None
                    else max(target["maximumAbsolute"], group_maximum)
                )

        for scope in ("previousDecision", "movement"):
            block = health.get(scope)
            if not isinstance(block, dict):
                raise ValueError(f"policy input health {scope} block is absent")
            status = block.get("status")
            if status != "complete":
                absent_legacy[f"{scope}:{status}"] += 1
            coordinates = (
                block.get("coordinates", [])
                if scope == "previousDecision"
                else [block.get("coordinate")] if block.get("coordinate") is not None else []
            )
            if not isinstance(coordinates, list) or any(
                not isinstance(value, dict) for value in coordinates
            ):
                raise ValueError(f"policy input health {scope} coordinates are invalid")
            for value in coordinates:
                observe_coordinate(transition_id, value)

        outliers = health.get("topNormalizedOutliers")
        if not isinstance(outliers, list) or len(outliers) > POLICY_INPUT_OUTLIER_LIMIT:
            raise ValueError("policy input health outlier bound/layout is invalid")
        for value in outliers:
            if not isinstance(value, dict):
                raise ValueError("policy input health outlier is invalid")
            magnitude = _number(value.get("absoluteNormalized"))
            normalized = _number(value.get("normalized"))
            if magnitude is None or normalized is None or magnitude != abs(normalized):
                raise ValueError("policy input health outlier magnitude is invalid")
            top.append({"transitionId": transition_id, **value})

    top.sort(
        key=lambda item: (
            -item["absoluteNormalized"],
            item["transitionId"],
            item.get("group", ""),
            item.get("coordinate", []),
        )
    )
    return {
        "transitionDenominator": len(transitions),
        "rowsWithTelemetry": rows_with,
        "rowsMissingTelemetry": len(transitions) - rows_with,
        "missingTelemetryReasons": dict(sorted(missing.items())),
        "actorFloatInputNonfiniteCount": actor_nonfinite,
        "normalizedInputNonfiniteCount": normalized_nonfinite,
        "maximumAbsoluteNormalized": maximum,
        "groups": [group_totals[name] for name in sorted(group_totals)],
        "knownCoreCoordinates": [known[index] for index in sorted(known)],
        "topNormalizedOutliers": top[:POLICY_INPUT_OUTLIER_LIMIT],
        "absentLegacyFields": dict(sorted(absent_legacy.items())),
        "interpretation": (
            "descriptive input-health telemetry from the exact actor tensor; large values are "
            "not by themselves causal evidence of an outcome or update"
        ),
    }


def _text(summary: dict[str, Any]) -> str:
    outcomes = summary["outcomes"]
    accounting = summary["economicControlFlow"]
    physical = summary["physicalEvents"]
    updates = summary["optimizerUpdates"]
    policy_inputs = summary["policyInputHealth"]
    signs = updates["descriptiveSignAlignment"]
    funnels = sorted(
        (
            finding
            for finding in accounting["requestGrantFindings"]
            if finding["activeGrantToRequestRatio"] is not None
            and finding["activePositiveRequestCount"] > 0
        ),
        key=lambda finding: (
            finding["activeGrantToRequestRatio"],
            finding["economicActivity"],
            finding["semanticIdentity"],
        ),
    )
    funnel_text = ", ".join(
        f"{finding['economicActivity']}/{finding['semanticIdentity']} "
        f"active grant/request={finding['activeGrantToRequestRatio']:.3g} "
        f"(positive request n={finding['activePositiveRequestCount']})"
        for finding in funnels[:3]
    ) or "no active positive-demand request/grant ratios available"
    objectives = outcomes["objectives"]
    critic_text = ", ".join(
        f"{name} mean|TD residual|={values['tdResidual']['absoluteMean']!r}, "
        f"adv(-/+)={values['advantageSignCounts']['negative']}/"
        f"{values['advantageSignCounts']['positive']}"
        for name, values in sorted(objectives.items())
    ) or "no objective diagnostics"
    trade_types = outcomes["tradeLifecycle"]["uniqueLifecycleRecordTypeCounts"]
    failures = sorted(
        (
            finding
            for finding in accounting["requestGrantFindings"]
            if finding["activeRequestShortfallCount"]
        ),
        key=lambda finding: (
            -finding["activeRequestShortfallCount"],
            finding["economicActivity"],
            finding["semanticIdentity"],
        ),
    )

    def failure_text(finding: dict[str, Any]) -> str:
        reasons = sorted(
            finding["compilerReasonCounts"].items(), key=lambda item: (-item[1], item[0])
        )[:2]
        reason = "/".join(f"{name}:{count}" for name, count in reasons) or "reason-unrecorded"
        example = finding["exampleTransitionIds"][0] if finding["exampleTransitionIds"] else "none"
        return (
            f"{finding['economicActivity']}/{finding['semanticIdentity']} "
            f"active shortfall={finding['activeRequestShortfallCount']}/"
            f"{finding['activePositiveRequestCount']} positive requests, "
            f"active zero-grant={finding['activeZeroGrantCount']}, "
            f"inactive denied={finding['inactiveDeniedRequestCount']}, "
            f"active grant/request={finding['activeGrantToRequestRatio']!r}, reasons={reason}, "
            f"example={example}"
        )
    operational_reasons = {
        reason: count
        for reason, count in accounting["reasonCounts"].items()
        if reason.startswith(("compiler-request:", "domain-funnel:"))
    }
    attribution_reasons = {
        reason: count
        for reason, count in accounting["reasonCounts"].items()
        if reason not in operational_reasons
    }

    def top_reason_text(reasons: dict[str, int]) -> str:
        top = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:3]
        return ", ".join(f"{reason} ({count})" for reason, count in top) or "none"

    coverage_text = "; ".join(
        f"rollout {row['rollout']} epoch {row['epoch']}: "
        f"{row['uniqueTransitionCount']}/{row['rolloutTransitionCount']} unique, "
        f"{row['batchDrawCount']} draws, {row['repeatedDrawCount']} repeats, "
        f"{row['notVisitedTransitionCount']} unvisited"
        for row in updates["updateCoverage"]["rows"]
    ) or "no mutation minibatches"
    policy_shift = updates["sampledPolicyShift"]
    policy_shift_text = (
        f"sampled nonnegative KL estimate={policy_shift['nonnegativeKlEstimateMean']!r} "
        f"over {policy_shift['rowCount']} draws"
        if policy_shift["nonnegativeKlEstimateMean"] is not None
        else "sampled nonnegative KL estimate unavailable"
    )
    post_step_shift = updates["postStepSampledPolicyShift"]
    post_step_shift_text = (
        "post-step bounded-audit sampled shift: "
        f"signed mean={post_step_shift['signedApproxKlMean']!r}, "
        f"nonnegative estimate={post_step_shift['nonnegativeKlEstimateMean']!r} "
        f"over {post_step_shift['rowCount']} fixed rows (not exact KL)"
        if post_step_shift is not None
        else "post-step bounded-audit sampled shift unavailable"
    )
    input_outliers = ", ".join(
        f"{row['group']}[{','.join(str(index) for index in row['coordinate'])}]="
        f"{row['normalized']:.6g} at {row['transitionId']}"
        for row in policy_inputs["topNormalizedOutliers"][:3]
    ) or "none"
    legacy_input_gaps = ", ".join(
        f"{name} ({count})" for name, count in policy_inputs["absentLegacyFields"].items()
    ) or "none"
    return "\n".join(
        (
            f"Trace: {outcomes['transitionCount']} transitions; "
            f"{updates['joinedMutationCount']} joined optimizer mutations.",
            f"End-to-end source attribution: {accounting['completeAccountingTransitionCount']}/"
            f"{accounting['accountingTransitionDenominator']} schema-2 transitions complete "
            f"(incomplete attribution does not mean source logs are absent); "
            f"{len(accounting['controls'])} distinct control identities.",
            f"Request/grant funnels: {funnel_text}. Operational reasons: "
            f"{top_reason_text(operational_reasons)}. Attribution gaps: "
            f"{top_reason_text(attribution_reasons)}.",
            "Recorded request/grant findings: "
            + ("; ".join(failure_text(finding) for finding in failures[:5]) or "none")
            + ". Population/QoL values are same-row context, not causal effects.",
            f"Trade lifecycle unique IDs: {trade_types or 'none'}; endpoint counters remain "
            "endpoint-scoped.",
            f"Physical events: {physical['eventCount']} rows in {physical['groupCount']} exact "
            f"category/item/unit/direction/phase groups. Typed camp-food positive rows: "
            f"{physical['typedCampFoodConsumption']['positiveEventCount']} across "
            f"{physical['typedCampFoodConsumption']['transitionsWithObservedPositiveTypedEvent']}/"
            f"{physical['typedCampFoodConsumption']['transitionDenominator']} transitions; "
            "zero versus unsupported absence and source attribution remain unresolved.",
            f"Critic/update context: {critic_text}; "
            f"sampled-action log-probability audit n={updates['alignmentDenominator']}, "
            f"negative-advantage reinforcement={signs['negativeAdvantageReinforcement']}, "
            f"positive-advantage suppression={signs['positiveAdvantageSuppression']}.",
            f"Update coverage: {coverage_text}; pre-step {policy_shift_text}; "
            f"{post_step_shift_text}. Signed approximate-KL remains a separate pre-step "
            "early-stop diagnostic.",
            f"Policy-input health: {policy_inputs['rowsWithTelemetry']}/"
            f"{policy_inputs['transitionDenominator']} rows captured from exact actor tensors; "
            f"max |normalized|={policy_inputs['maximumAbsoluteNormalized']!r}; "
            f"actor-float nonfinite count={policy_inputs['actorFloatInputNonfiniteCount']}; "
            f"top outliers: {input_outliers}; legacy-field gaps: {legacy_input_gaps}.",
            "Critic TD/advantage statistics are internal bootstrap/GAE target diagnostics; "
            "independent critic calibration is unavailable.",
            "Optimizer-loss attribution remains null without a matched intervention; gradient "
            "geometry is descriptive pre-Adam context.",
        )
    )


def summarize_trace(path: str | Path) -> dict[str, Any]:
    manifest, transitions, updates = economic_diagnostics.load_artifact(path)
    transition_ids = set()
    for row in transitions:
        economic_diagnostics.validate_transition(row)
        identity = str(row["transitionId"])
        if identity in transition_ids:
            raise ValueError(f"duplicate economic transition: {identity}")
        transition_ids.add(identity)
    for row in updates:
        economic_diagnostics.validate_update(row, transition_ids)
    result = {
        "schemaVersion": SUMMARY_SCHEMA_VERSION,
        "kind": "economic-training-trace-summary",
        "scope": "descriptive prospective training telemetry; no arm ranking or readiness claim",
        "source": {
            "artifactPath": manifest["artifactPath"],
            "manifestSha256": manifest["manifestSha256"],
            "doneSha256": manifest["doneSha256"],
            "runId": manifest.get("runId"),
            "arm": manifest.get("arm"),
            "optimizerSeed": manifest.get("optimizerSeed"),
        },
        "denominators": {
            "transitions": len(transitions),
            "updateEvents": len(updates),
            "optimizerMutations": sum(row.get("event") == "mutation-complete" for row in updates),
        },
        "economicControlFlow": _control_summary(transitions),
        "physicalEvents": _physical_event_summary(transitions),
        "outcomes": _outcome_summary(transitions),
        "optimizerUpdates": _update_summary(transitions, updates),
        "policyInputHealth": _policy_input_health_summary(transitions),
    }
    result["text"] = _text(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")
    summary = summarize_trace(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(summary) + b"\n")
    print(summary["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
