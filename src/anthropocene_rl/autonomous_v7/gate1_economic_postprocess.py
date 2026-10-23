"""Descriptive economic decomposition for sealed Gate-1 streamed worlds.

This module consumes checkpoint-evaluation evidence only.  It does not execute a
policy, reconstruct a simulator transition, or assign causal credit.  Every rate
is an exposure rate with its numerator and denominator; the base-world seed is
the inferential unit for any later analysis.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .economic_accounting import CONTROL_PHASES, validate_economic_accounting
from .provenance import digest

WORLD_KIND = "gate1_fresh_common_producer_world"
EVALUATION_KIND = "gate1_fresh_common_producer_evaluation"
OUTPUT_KIND = "gate1_economic_failure_decomposition"
REGIMES = {0: "default", 10: "farming", 6: "fishing", 12: "herding"}
LINKS = (
    ("nativeDemandToActorCappedRequest", "nativeDemand", "actorCappedRequest"),
    ("actorCappedRequestToCompilerGrant", "actorCappedRequest", "compilerGrant"),
    ("compilerGrantToExecutedClaim", "compilerGrant", "executedWorkerOrInputClaim"),
    ("compilerGrantToPhysicalReceipt", "compilerGrant", "committedPhysicalReceipt"),
    ("physicalReceiptToInventoryChange", "committedPhysicalReceipt", "inventoryChange"),
    (
        "physicalReceiptToConsumptionOrFinalService",
        "committedPhysicalReceipt",
        "consumptionOrFinalService",
    ),
)
FUNNEL_PHASES = (
    "nativeDemand",
    "actorCappedRequest",
    "compilerGrant",
    "executedWork",
    "physicalReceipt",
    "finalService",
)

_WORLD_FIELDS = {
    "schema",
    "kind",
    "candidate_digest",
    "node",
    "pane",
    "world_id",
    "result",
    "digest",
}
_EVALUATION_FIELDS = {
    "schema",
    "kind",
    "candidate_digest",
    "node",
    "pane",
    "checkpoint_sha256",
    "worker_capture_sha256",
    "producer_scope",
    "node_evidence_scope",
    "historical_artifact_claim",
    "historical_producer_status",
    "historical_replay_claim",
    "selection_allowed",
    "readiness_claimed",
    "world_count",
    "world_artifacts",
    "streaming",
    "digest",
}
_TRAJECTORY_FIELDS = {
    "schema",
    "kind",
    "ordering",
    "reset",
    "bounds",
    "opening",
    "steps",
    "capture_receipt",
    "digest",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _sealed(value: Any, kind: str, fields: set[str]) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value.get("schema") != 1
        or value.get("kind") != kind
        or value.get("digest")
        != digest({key: item for key, item in value.items() if key != "digest"})
    ):
        raise ValueError(f"{kind} is malformed or not content-addressed")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _contract(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("digest") != digest(
        {key: item for key, item in value.items() if key != "digest"}
    ):
        raise ValueError(f"{what} is missing or mutated")
    return value


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what} is not finite numeric data")
    return float(value)


def _nonnegative_integer(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{what} is not a nonnegative integer")
    return value


def _rate(numerator: int | float, denominator: int | float, reason: str) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
        "undefinedReason": reason if denominator == 0 else None,
    }


def _reason_counter(counter: Counter[str]) -> list[dict[str, Any]]:
    return [{"reason": reason, "count": count} for reason, count in sorted(counter.items())]


def _sum_by_unit(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    quantities: defaultdict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for event in events:
        unit = str(event["unit"])
        quantities[unit] += _number(event["quantity"], "physical event quantity")
        counts[unit] += 1
    return [
        {"unit": unit, "quantity": quantities[unit], "eventCount": counts[unit]}
        for unit in sorted(quantities)
    ]


def _group_identity(control: dict[str, Any]) -> str:
    domain = control["domain"]
    metadata = control["metadata"]
    if domain == "manufacturing":
        fields = ("serviceClass", "service", "good", "rawMaterial", "role")
        return "request:" + "|".join(f"{field}={metadata.get(field)}" for field in fields)
    if domain == "trade":
        return "leg:" + "|".join(
            f"{field}={metadata.get(field)}" for field in ("direction", "good")
        )
    return str(control["identity"])


def _control_bucket(dimensions: dict[str, Any]) -> dict[str, Any]:
    return {
        "dimensions": dimensions,
        "worldSeeds": set(),
        "worldIds": set(),
        "campTurns": set(),
        "controlIds": set(),
        "controlRows": 0,
        "classificationCounts": Counter(),
        "classificationReasons": Counter(),
        "activeDemandAvailable": 0,
        "activePositiveDemand": 0,
        "activeRequestAvailable": 0,
        "activePositiveRequest": 0,
        "positiveRequestWithGrant": 0,
        "constrainedPositiveRequest": 0,
        "phases": {
            phase: {
                "available": defaultdict(lambda: {"count": 0, "sum": 0.0}),
                "unavailableCount": 0,
                "unavailableReasons": Counter(),
            }
            for phase in CONTROL_PHASES
        },
        "links": {
            name: {
                "fromPhase": source,
                "toPhase": target,
                "available": defaultdict(
                    lambda: {"rowCount": 0, "numerator": 0.0, "denominator": 0.0}
                ),
                "excluded": Counter(),
            }
            for name, source, target in LINKS
        },
    }


def _add_control(
    bucket: dict[str, Any],
    control: dict[str, Any],
    *,
    world_seed: int,
    world_id: str,
    turn: int,
    camp: int,
) -> None:
    bucket["worldSeeds"].add(world_seed)
    bucket["worldIds"].add(world_id)
    bucket["campTurns"].add((world_id, turn, camp))
    bucket["controlIds"].add(control["controlId"])
    bucket["controlRows"] += 1
    classification = control["classification"]
    bucket["classificationCounts"][classification] += 1
    if control.get("classificationReason"):
        bucket["classificationReasons"][str(control["classificationReason"])] += 1

    phases = control["phases"]
    units = control["phaseUnits"]
    unavailable = control["unavailable"]
    for phase in CONTROL_PHASES:
        value = phases[phase]
        phase_state = bucket["phases"][phase]
        if value is None:
            phase_state["unavailableCount"] += 1
            phase_state["unavailableReasons"][str(unavailable[phase])] += 1
        else:
            unit = str(units[phase])
            current = phase_state["available"][unit]
            current["count"] += 1
            current["sum"] += _number(value, f"control phase {phase}")

    if classification == "active":
        demand = phases["nativeDemand"]
        request = phases["actorCappedRequest"]
        grant = phases["compilerGrant"]
        if demand is not None:
            bucket["activeDemandAvailable"] += 1
            bucket["activePositiveDemand"] += int(float(demand) > 0)
        if request is not None:
            bucket["activeRequestAvailable"] += 1
            bucket["activePositiveRequest"] += int(float(request) > 0)
        if request is not None and float(request) > 0 and grant is not None:
            bucket["positiveRequestWithGrant"] += 1
            bucket["constrainedPositiveRequest"] += int(float(grant) + 1e-12 < float(request))

        for name, source, target in LINKS:
            state = bucket["links"][name]
            source_value, target_value = phases[source], phases[target]
            if source_value is None:
                state["excluded"][f"source-unavailable:{unavailable[source]}"] += 1
            elif target_value is None:
                state["excluded"][f"target-unavailable:{unavailable[target]}"] += 1
            elif units[source] != units[target]:
                state["excluded"][f"unit-mismatch:{units[source]}->{units[target]}"] += 1
            elif float(source_value) <= 0:
                state["excluded"]["zero-or-negative-denominator"] += 1
            else:
                current = state["available"][str(units[source])]
                current["rowCount"] += 1
                current["numerator"] += float(target_value)
                current["denominator"] += float(source_value)


def _finalize_control(bucket: dict[str, Any]) -> dict[str, Any]:
    phases = {}
    for phase, state in bucket["phases"].items():
        phases[phase] = {
            "availableByUnit": [
                {
                    "unit": unit,
                    "count": values["count"],
                    "sum": values["sum"],
                    "mean": values["sum"] / values["count"],
                }
                for unit, values in sorted(state["available"].items())
            ],
            "unavailableCount": state["unavailableCount"],
            "unavailableReasons": _reason_counter(state["unavailableReasons"]),
        }
    links = {}
    for name, state in bucket["links"].items():
        by_unit = [
            {
                "unit": unit,
                **values,
                "ratioOfSums": values["numerator"] / values["denominator"],
            }
            for unit, values in sorted(state["available"].items())
        ]
        links[name] = {
            "fromPhase": state["fromPhase"],
            "toPhase": state["toPhase"],
            "comparableByUnit": by_unit,
            "excludedRows": sum(state["excluded"].values()),
            "excludedReasons": _reason_counter(state["excluded"]),
            "value": by_unit[0]["ratioOfSums"] if len(by_unit) == 1 else None,
            "undefinedReason": (
                None
                if len(by_unit) == 1
                else "no-unit-compatible-positive-denominator-rows"
                if not by_unit
                else "multiple-comparable-units"
            ),
        }
    return {
        **bucket["dimensions"],
        "baseWorldSeeds": sorted(bucket["worldSeeds"]),
        "baseWorldCount": len(bucket["worldSeeds"]),
        "worldIds": sorted(bucket["worldIds"]),
        "campTurnCount": len(bucket["campTurns"]),
        "controlRowCount": bucket["controlRows"],
        "controlIds": sorted(bucket["controlIds"]),
        "classificationCounts": dict(sorted(bucket["classificationCounts"].items())),
        "classificationReasons": _reason_counter(bucket["classificationReasons"]),
        "rates": {
            "positiveNativeDemand": _rate(
                bucket["activePositiveDemand"],
                bucket["activeDemandAvailable"],
                "no-active-controls-with-available-native-demand",
            ),
            "positiveActorCappedRequest": _rate(
                bucket["activePositiveRequest"],
                bucket["activeRequestAvailable"],
                "no-active-controls-with-available-actor-capped-request",
            ),
            "compilerConstraintAmongPositiveRequests": _rate(
                bucket["constrainedPositiveRequest"],
                bucket["positiveRequestWithGrant"],
                "no-positive-requests-with-available-grant",
            ),
        },
        "phases": phases,
        "unitSafeLinks": links,
        "outcomeInterpretation": "ending population and QoL are descriptive context only",
    }


def _compiler_rows(
    context: dict[str, Any], row: dict[str, Any], unresolved: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    domain_active = {
        "cultivation": bool(row["domains"]["cultivation"].get("capable")),
        "herding": bool(row["domains"]["herding"].get("capable")),
        "manufacturing": bool(row["domains"]["manufacturing"].get("active")),
        "trade": bool(row["domains"]["trade"].get("eligible")),
    }
    for domain in ("cultivation", "herding", "manufacturing", "trade"):
        funnels = row["domains"][domain].get("funnels")
        if funnels is None:
            if domain_active[domain]:
                unresolved.append(
                    {
                        **context,
                        "campId": row["campId"],
                        "sourceType": "compiler-funnel",
                        "sourceId": domain,
                        "path": f"domains.{domain}.funnels",
                        "reason": "active-domain-compiler-funnels-unavailable",
                    }
                )
            continue
        if not isinstance(funnels, list) or any(not isinstance(funnel, dict) for funnel in funnels):
            raise ValueError(f"{domain} compiler funnels are malformed")
        for ordinal, funnel in enumerate(funnels):
            if not isinstance(funnel.get("metric"), str) or not isinstance(funnel.get("unit"), str):
                raise ValueError(f"{domain} compiler funnel identity/unit is malformed")
            reasons = funnel.get("constraintReasons")
            if not isinstance(reasons, list) or any(
                not isinstance(reason, str) or not reason for reason in reasons
            ):
                raise ValueError(f"{domain} compiler constraint reasons are malformed")
            prefix = f"domains.{domain}.funnels[{ordinal}]."
            availability = {
                phase: row["unavailable"].get(prefix + phase)
                for phase in FUNNEL_PHASES
                if funnel.get(phase) is None
            }
            output.append(
                {
                    **context,
                    "campId": row["campId"],
                    "societyId": row["societyId"],
                    "domain": domain,
                    "ordinal": ordinal,
                    "metric": funnel["metric"],
                    "unit": funnel["unit"],
                    "active": bool(funnel.get("active")),
                    "phases": {phase: funnel.get(phase) for phase in FUNNEL_PHASES},
                    "constraintReasons": sorted(reasons),
                    "unavailable": availability,
                }
            )
            for phase, reason in availability.items():
                unresolved.append(
                    {
                        **context,
                        "campId": row["campId"],
                        "sourceType": "compiler-funnel",
                        "sourceId": f"{domain}:{funnel['metric']}:{ordinal}",
                        "path": phase,
                        "reason": reason or "compiler-funnel-phase-unavailable",
                    }
                )
    return output


def _compiler_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    dimension_fields = (
        "candidateDigest",
        "node",
        "pane",
        "withheld",
        "regime",
        "decisionTurn",
        "domain",
        "metric",
        "unit",
    )
    for row in rows:
        dimensions = {field: row[field] for field in dimension_fields}
        key = tuple(dimensions.values())
        group = groups.setdefault(
            key,
            {
                "dimensions": dimensions,
                "worldSeeds": set(),
                "worldIds": set(),
                "campTurns": set(),
                "rows": 0,
                "active": 0,
                "requestAvailable": 0,
                "positiveRequests": 0,
                "positiveRequestWithGrant": 0,
                "constrained": 0,
                "actorGrantComparableRows": 0,
                "actorGrantNumerator": 0.0,
                "actorGrantDenominator": 0.0,
                "reasons": Counter(),
                "phases": {
                    phase: {"count": 0, "sum": 0.0, "unavailable": 0} for phase in FUNNEL_PHASES
                },
            },
        )
        group["worldSeeds"].add(row["baseWorldSeed"])
        group["worldIds"].add(row["worldId"])
        group["campTurns"].add((row["worldId"], row["decisionTurn"], row["campId"]))
        group["rows"] += 1
        group["active"] += int(row["active"])
        group["reasons"].update(row["constraintReasons"])
        for phase, value in row["phases"].items():
            if value is None:
                group["phases"][phase]["unavailable"] += 1
            else:
                group["phases"][phase]["count"] += 1
                group["phases"][phase]["sum"] += _number(value, f"compiler funnel {phase}")
        request = row["phases"]["actorCappedRequest"]
        grant = row["phases"]["compilerGrant"]
        if row["active"] and request is not None:
            group["requestAvailable"] += 1
            group["positiveRequests"] += int(float(request) > 0)
        if row["active"] and request is not None and float(request) > 0 and grant is not None:
            group["positiveRequestWithGrant"] += 1
            group["constrained"] += int(
                float(grant) + 1e-12 < float(request) or bool(row["constraintReasons"])
            )
            group["actorGrantComparableRows"] += 1
            group["actorGrantNumerator"] += float(grant)
            group["actorGrantDenominator"] += float(request)
    output = []
    for group in groups.values():
        phases = {
            phase: {
                **values,
                "mean": (None if values["count"] == 0 else values["sum"] / values["count"]),
                "undefinedReason": ("no-available-values" if values["count"] == 0 else None),
            }
            for phase, values in group["phases"].items()
        }
        actor_sum = group["actorGrantDenominator"]
        grant_sum = group["actorGrantNumerator"]
        comparable = group["actorGrantComparableRows"] > 0 and actor_sum > 0
        output.append(
            {
                **group["dimensions"],
                "baseWorldSeeds": sorted(group["worldSeeds"]),
                "baseWorldCount": len(group["worldSeeds"]),
                "worldIds": sorted(group["worldIds"]),
                "campTurnCount": len(group["campTurns"]),
                "funnelRowCount": group["rows"],
                "activeCount": group["active"],
                "rates": {
                    "positiveActorCappedRequest": _rate(
                        group["positiveRequests"],
                        group["requestAvailable"],
                        "no-active-funnels-with-available-request",
                    ),
                    "constraintAmongPositiveRequests": _rate(
                        group["constrained"],
                        group["positiveRequestWithGrant"],
                        "no-positive-requests-with-available-grant",
                    ),
                },
                "actorCappedRequestToCompilerGrant": {
                    "comparableRowCount": group["actorGrantComparableRows"],
                    "numerator": grant_sum if comparable else None,
                    "denominator": actor_sum if comparable else None,
                    "value": grant_sum / actor_sum if comparable else None,
                    "undefinedReason": (
                        None if comparable else "unavailable-or-zero-unit-compatible-denominator"
                    ),
                },
                "constraintReasons": _reason_counter(group["reasons"]),
                "phases": phases,
            }
        )
    return sorted(
        output,
        key=lambda item: tuple(str(item[field]) for field in dimension_fields),
    )


def _regime(world_id: str, withheld: int) -> str:
    label = REGIMES.get(withheld, f"withheld-{withheld}")
    if withheld in REGIMES and not world_id.endswith(f"-{label}"):
        raise ValueError("world ID regime suffix differs from the sealed world descriptor")
    return label


def _validate_trajectory(world: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = world["result"]
    if not isinstance(result, dict):
        raise ValueError("world result is malformed")
    descriptor = _contract(result.get("world_contract"), "sealed world descriptor")
    for field in ("seed", "withheld", "turn", "players"):
        if field not in descriptor:
            raise ValueError(f"sealed world descriptor lacks {field}")
    seed = _nonnegative_integer(descriptor["seed"], "world seed")
    withheld = _nonnegative_integer(descriptor["withheld"], "world withheld mask")
    if result.get("seed") != seed or result.get("withheld") != withheld:
        raise ValueError("world result differs from its frozen descriptor")

    evidence = result.get("trajectory_evidence")
    if (
        not isinstance(evidence, dict)
        or set(evidence) != _TRAJECTORY_FIELDS
        or evidence.get("schema") != 2
        or evidence.get("kind") != "gate1_canonical_trajectory"
        or evidence.get("digest")
        != digest({key: item for key, item in evidence.items() if key != "digest"})
    ):
        raise ValueError("canonical trajectory is malformed or not content-addressed")
    if result.get("trajectory_digest") != evidence["digest"]:
        raise ValueError("world result trajectory digest differs from canonical evidence")
    reset = evidence["reset"]
    if (
        not isinstance(reset, dict)
        or reset.get("world_id") != world["world_id"]
        or reset.get("snapshot_sha256") != descriptor.get("snapshot_sha256")
        or reset.get("turn") != descriptor.get("turn")
        or not _is_sha256(reset.get("fingerprint"))
        or reset.get("trajectory_id") != reset.get("fingerprint")
    ):
        raise ValueError("canonical trajectory reset differs from the frozen world descriptor")
    warmup = reset.get("warmup_chain")
    if not isinstance(warmup, list):
        raise ValueError("canonical trajectory warm-up chain is malformed")
    turn, fingerprint = reset["turn"], reset["fingerprint"]
    for index, link in enumerate(warmup):
        if (
            not isinstance(link, dict)
            or link.get("index") != index
            or link.get("source_turn") != turn
            or link.get("source_fingerprint") != fingerprint
            or link.get("successor_turn") != turn + 1
            or link.get("trajectory_id") != reset["trajectory_id"]
            or not _is_sha256(link.get("successor_fingerprint"))
        ):
            raise ValueError("canonical trajectory warm-up chain is discontinuous")
        turn, fingerprint = link["successor_turn"], link["successor_fingerprint"]
    opening = evidence["opening"]
    if (
        not isinstance(opening, dict)
        or opening.get("turn") != turn
        or opening.get("fingerprint") != fingerprint
        or not isinstance(opening.get("villages"), list)
    ):
        raise ValueError("canonical trajectory opening boundary is discontinuous")

    current = opening
    rows: list[dict[str, Any]] = []
    steps = evidence["steps"]
    if not isinstance(steps, list):
        raise ValueError("canonical trajectory steps are malformed")
    for index, step in enumerate(steps):
        if (
            not isinstance(step, dict)
            or step.get("index") != index
            or step.get("source_turn") != current["turn"]
            or step.get("source_fingerprint") != current["fingerprint"]
        ):
            raise ValueError("canonical trajectory step is discontinuous")
        expected_transition = digest(
            {
                "index": index,
                "source_turn": step["source_turn"],
                "source_fingerprint": step["source_fingerprint"],
                "successor_turn": step["successor"]["turn"],
                "successor_fingerprint": step["successor"]["fingerprint"],
            }
        )
        if step.get("transition_id") != expected_transition:
            raise ValueError("canonical trajectory step identity is not content-bound")
        actions = step.get("actions")
        economics = step.get("economic")
        if not isinstance(actions, list) or not isinstance(economics, list):
            raise ValueError("canonical trajectory actions/economic rows are malformed")
        action_camps = [action.get("camp") for action in actions if isinstance(action, dict)]
        economic_camps = [row.get("campId") for row in economics if isinstance(row, dict)]
        if (
            len(action_camps) != len(actions)
            or len(set(action_camps)) != len(action_camps)
            or sorted(action_camps) != sorted(economic_camps)
        ):
            raise ValueError("canonical trajectory economic rows do not cover action camps")
        source_villages = {
            village["camp"]: village
            for village in current["villages"]
            if isinstance(village, dict) and "camp" in village
        }
        for row in economics:
            if (
                row.get("schemaVersion") != 2
                or row.get("trajectoryId") != reset["trajectory_id"]
                or row.get("decisionTurn") != step["source_turn"]
                or row.get("campId") not in source_villages
                or row.get("societyId") != source_villages[row["campId"]].get("society")
            ):
                raise ValueError("economic row identity differs from the source boundary")
            validate_economic_accounting(row)
            rows.append(
                {
                    "stepIndex": index,
                    "sourceTurn": step["source_turn"],
                    "sourceFingerprint": step["source_fingerprint"],
                    "economic": row,
                }
            )
        successor = step.get("successor")
        if (
            not isinstance(successor, dict)
            or successor.get("turn") != step["source_turn"] + 1
            or not _is_sha256(successor.get("fingerprint"))
            or not isinstance(successor.get("villages"), list)
        ):
            raise ValueError("canonical trajectory successor boundary is malformed")
        current = successor
    receipt = evidence["capture_receipt"]
    if (
        not isinstance(receipt, dict)
        or receipt.get("steps") != len(steps)
        or receipt.get("warmup_steps") != len(warmup)
        or result.get("evaluation_steps") != len(steps)
        or result.get("capture_receipt") != receipt
    ):
        raise ValueError("canonical trajectory capture receipt differs from retained evidence")
    return {
        "seed": seed,
        "withheld": withheld,
        "regime": _regime(world["world_id"], withheld),
        "trajectoryId": reset["trajectory_id"],
        "trajectoryDigest": evidence["digest"],
    }, rows


def _inventory_source(row: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any] | None:
    domain, identity = ledger["domain"], str(ledger["item"])
    if domain == "craft-good":
        return next((item for item in row["goods"] if item["good"] == identity), None)
    if domain == "material":
        return next((item for item in row["materials"] if item["material"] == identity), None)
    if domain == "herd":
        return next((item for item in row["herds"] if str(item["herdId"]) == identity), None)
    if domain == "field":
        return next((item for item in row["fields"] if str(item["fieldId"]) == identity), None)
    if domain in {"population", "food"}:
        return row["domains"]["population"]
    return None


def _event_item(ledger: dict[str, Any]) -> str:
    if ledger["domain"] in {"herd", "field"}:
        return f"{ledger['domain']}:{ledger['item']}"
    return str(ledger["item"])


def _inventory_rows(context: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for ledger in row["conservationRows"]:
        item = _event_item(ledger)
        events = [event for event in row["physicalEvents"] if event["item"] == item]
        output.append(
            {
                **context,
                "campId": row["campId"],
                "societyId": row["societyId"],
                "domain": ledger["domain"],
                "item": ledger["item"],
                "unit": ledger["unit"],
                "conservation": copy.deepcopy(ledger),
                "stockOrIdentityRow": copy.deepcopy(_inventory_source(row, ledger)),
                "physicalEvents": copy.deepcopy(events),
                "physicalEventTotalsByUnit": _sum_by_unit(events),
            }
        )
    return output


def _manufacturing_rows(
    context: dict[str, Any], row: dict[str, Any], unresolved: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    controls = {
        control["identity"]: control
        for control in row["controls"]
        if control["domain"] == "manufacturing"
    }
    output = []
    for request in row["requests"]:
        request_id = request["requestId"]
        control = controls.get(request_id)
        if control is None:
            raise ValueError("manufacturing request lacks its cross-bound control")
        source_events = [
            event for event in row["physicalEvents"] if event.get("sourceRequestId") == request_id
        ]
        receipt_events = [
            event for event in row["physicalEvents"] if event.get("receiptId") == request_id
        ]
        production_events = [
            event for event in source_events if event["category"] == "production-receipt"
        ]
        recipe_events = [
            event
            for event in source_events
            if event["category"] == "intermediate-recipe-consumption"
        ]
        output.append(
            {
                **context,
                "campId": row["campId"],
                "societyId": row["societyId"],
                "request": copy.deepcopy(request),
                "control": copy.deepcopy(control),
                "requestBoundProductionReceipts": _sum_by_unit(production_events),
                "requestBoundRecipeDebits": _sum_by_unit(recipe_events),
                "receiptClaims": copy.deepcopy(receipt_events),
                "unmetReason": request.get("unmetReason") or None,
                "inventoryAttributionAvailable": control["phases"]["inventoryChange"] is not None,
            }
        )
        for field in (
            "executedWorkerYears",
            "committedPhysicalReceipt",
            "inventoryChange",
            "consumptionOrFinalService",
        ):
            if request.get(field) is None:
                phase = {
                    "executedWorkerYears": "executedWorkerOrInputClaim",
                    "committedPhysicalReceipt": "committedPhysicalReceipt",
                    "inventoryChange": "inventoryChange",
                    "consumptionOrFinalService": "consumptionOrFinalService",
                }[field]
                unresolved.append(
                    {
                        **context,
                        "campId": row["campId"],
                        "sourceType": "manufacturing-request",
                        "sourceId": request_id,
                        "path": field,
                        "reason": control["unavailable"].get(
                            phase,
                            row["unavailable"].get(f"requests[{request.get('ordinal')}].{field}"),
                        )
                        or "authoritative-request-attribution-unavailable",
                    }
                )
    return output


def _manufacturing_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dimension_fields = (
        "candidateDigest",
        "node",
        "pane",
        "withheld",
        "regime",
        "decisionTurn",
    )
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        request = row["request"]
        request_dimensions = {
            "serviceClass": request.get("serviceClass"),
            "service": request.get("service"),
            "good": request.get("good"),
            "rawMaterial": request.get("rawMaterial"),
            "role": request.get("role"),
            "nativeUnit": request.get("nativeUnit"),
        }
        dimensions = {
            **{field: row[field] for field in dimension_fields},
            **request_dimensions,
        }
        key = tuple(dimensions.values())
        group = groups.setdefault(
            key,
            {
                "dimensions": dimensions,
                "worldSeeds": set(),
                "worldIds": set(),
                "campTurns": set(),
                "requestRows": 0,
                "positiveRequestRows": 0,
                "requestsWithGrant": 0,
                "constrainedRows": 0,
                "unmetRows": 0,
                "unmetReasons": Counter(),
            },
        )
        group["worldSeeds"].add(row["baseWorldSeed"])
        group["worldIds"].add(row["worldId"])
        group["campTurns"].add((row["worldId"], row["decisionTurn"], row["campId"]))
        group["requestRows"] += 1
        requested = request.get("actorCappedRequest")
        grant = request.get("compilerGrant")
        if requested is not None and float(requested) > 0:
            group["positiveRequestRows"] += 1
            if grant is not None:
                group["requestsWithGrant"] += 1
                group["constrainedRows"] += int(float(grant) + 1e-12 < float(requested))
        reason = request.get("unmetReason")
        if reason:
            group["unmetRows"] += 1
            group["unmetReasons"][str(reason)] += 1
    output = []
    for group in groups.values():
        output.append(
            {
                **group["dimensions"],
                "baseWorldSeeds": sorted(group["worldSeeds"]),
                "baseWorldCount": len(group["worldSeeds"]),
                "worldIds": sorted(group["worldIds"]),
                "campTurnCount": len(group["campTurns"]),
                "requestRowCount": group["requestRows"],
                "rates": {
                    "positiveRequest": _rate(
                        group["positiveRequestRows"],
                        group["requestRows"],
                        "no-manufacturing-requests",
                    ),
                    "compilerConstraintAmongPositiveRequests": _rate(
                        group["constrainedRows"],
                        group["requestsWithGrant"],
                        "no-positive-requests-with-available-grant",
                    ),
                    "categoricalUnmet": _rate(
                        group["unmetRows"],
                        group["requestRows"],
                        "no-manufacturing-requests",
                    ),
                },
                "unmetReasons": _reason_counter(group["unmetReasons"]),
            }
        )
    sort_fields = (
        *dimension_fields,
        "serviceClass",
        "service",
        "good",
        "rawMaterial",
        "role",
        "nativeUnit",
    )
    return sorted(output, key=lambda item: tuple(str(item[field]) for field in sort_fields))


def _trade_identity(record: dict[str, Any]) -> tuple[str, str]:
    field = {
        "eligibility": "eventId",
        "requisition": "requestId",
        "transaction": "transactionId",
    }.get(record.get("recordType"))
    if field is None or not isinstance(record.get(field), str) or not record[field]:
        raise ValueError("trade lifecycle record lacks a typed identity")
    return str(record["recordType"]), str(record[field])


def _trade_rows(
    context: dict[str, Any], economic_rows: list[dict[str, Any]], unresolved: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    observed: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    for row in economic_rows:
        camp = row["campId"]
        for record in row["trades"]:
            identity = _trade_identity(record)
            if identity in records and _canonical(records[identity]) != _canonical(record):
                raise ValueError("bilateral trade endpoint copies disagree")
            records[identity] = copy.deepcopy(record)
            observed[identity].add(camp)

    eligibility = {
        record["campId"]: record for (kind, _), record in records.items() if kind == "eligibility"
    }
    requisitions = {
        identity: record for (kind, identity), record in records.items() if kind == "requisition"
    }
    transactions = {
        record["requestId"]: record
        for (kind, _), record in records.items()
        if kind == "transaction"
    }
    lifecycle_rows = []
    participant_rows = []
    proposal_outcomes: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    settlement_states: Counter[str] = Counter()
    refund_by_good: defaultdict[str, float] = defaultdict(float)
    blocked_by_good: defaultdict[str, float] = defaultdict(float)
    shortfall_by_good: defaultdict[str, float] = defaultdict(float)
    delivered_by_good: defaultdict[str, float] = defaultdict(float)
    for request_id, request in sorted(requisitions.items()):
        proposal_outcomes[str(request["outcome"])] += 1
        if request.get("rejectionReason"):
            rejection_reasons[str(request["rejectionReason"])] += 1
        transaction = transactions.get(request_id)
        if transaction is not None:
            settlement_states[str(transaction["settlementState"])] += 1
            for leg in transaction["legs"]:
                good = str(leg["good"])
                refund_by_good[good] += float(leg["refundedOrReturnedQuantity"])
                blocked_by_good[good] += float(leg["storageBlockedQuantity"])
                shortfall_by_good[good] += float(leg["counterpartyShortfallQuantity"])
                delivered_by_good[good] += float(leg["deliveredQuantity"])
            if transaction.get("delayedUtility") is None:
                unresolved.append(
                    {
                        **context,
                        "sourceType": "trade-transaction",
                        "sourceId": transaction["transactionId"],
                        "path": "delayedUtility",
                        "reason": transaction.get("delayedUtilityUnavailableReason")
                        or "delayed-utility-unavailable",
                    }
                )
        lifecycle_rows.append(
            {
                **context,
                "requestId": request_id,
                "requesterEligibility": copy.deepcopy(eligibility.get(request["requester"])),
                "responderEligibility": copy.deepcopy(eligibility.get(request["responder"])),
                "requisition": copy.deepcopy(request),
                "transaction": copy.deepcopy(transaction),
                "observedInCamps": sorted(observed[("requisition", request_id)]),
                "transactionObservedInCamps": (
                    []
                    if transaction is None
                    else sorted(observed[("transaction", transaction["transactionId"])])
                ),
                "endpointCoverageComplete": (
                    transaction is None
                    or set(observed[("transaction", transaction["transactionId"])])
                    == {request["requester"], request["responder"]}
                ),
            }
        )
        if transaction is None:
            continue
        transaction_camps = set(observed[("transaction", transaction["transactionId"])])
        expected_camps = {request["requester"], request["responder"]}
        if transaction_camps != expected_camps:
            unresolved.append(
                {
                    **context,
                    "sourceType": "trade-transaction",
                    "sourceId": transaction["transactionId"],
                    "path": "endpointCoverage",
                    "reason": (
                        "missing-reciprocal-endpoint-copies:"
                        + ",".join(str(camp) for camp in sorted(expected_camps - transaction_camps))
                    ),
                }
            )
        for camp, role, utility_field, reason_field in (
            (
                transaction["requester"],
                "requester",
                "requesterImmediateUtility",
                "requesterImmediateUtilityUnavailableReason",
            ),
            (
                transaction["responder"],
                "responder",
                "responderImmediateUtility",
                "responderImmediateUtilityUnavailableReason",
            ),
        ):
            flows = []
            for leg in transaction["legs"]:
                if leg["sender"] == camp:
                    quantity = -float(leg["senderDebitQuantity"])
                    flows.append(
                        {
                            "legId": leg["legId"],
                            "good": leg["good"],
                            "unit": leg["unit"],
                            "direction": "debit",
                            "signedQuantity": quantity,
                        }
                    )
                if leg["receiver"] == camp:
                    quantity = float(leg["receiverCreditQuantity"])
                    flows.append(
                        {
                            "legId": leg["legId"],
                            "good": leg["good"],
                            "unit": leg["unit"],
                            "direction": "credit",
                            "signedQuantity": quantity,
                        }
                    )
            participant_rows.append(
                {
                    **context,
                    "transactionId": transaction["transactionId"],
                    "campId": camp,
                    "role": role,
                    "goodsFlows": flows,
                    "immediateUtility": transaction[utility_field],
                    "immediateUtilityUnavailableReason": transaction[reason_field],
                    "delayedUtility": transaction["delayedUtility"],
                    "delayedUtilityUnavailableReason": transaction[
                        "delayedUtilityUnavailableReason"
                    ],
                }
            )
    summary = {
        **context,
        "eligibilityCampCount": len(eligibility),
        "eligibleCampCount": sum(int(record["eligible"]) for record in eligibility.values()),
        "partnerSearchCount": sum(record["partnerSearchCount"] for record in eligibility.values()),
        "requisitionCount": len(requisitions),
        "proposalOutcomes": dict(sorted(proposal_outcomes.items())),
        "rejectionReasons": _reason_counter(rejection_reasons),
        "transactionCount": len(transactions),
        "settlementStates": dict(sorted(settlement_states.items())),
        "deliveredByGood": dict(sorted(delivered_by_good.items())),
        "refundedOrReturnedByGood": dict(sorted(refund_by_good.items())),
        "storageBlockedByGood": dict(sorted(blocked_by_good.items())),
        "counterpartyShortfallByGood": dict(sorted(shortfall_by_good.items())),
    }
    return lifecycle_rows, participant_rows, summary


def _manifest_worlds(path: Path, manifest: dict[str, Any]) -> Iterable[tuple[Path, dict[str, Any]]]:
    if (
        not _is_sha256(manifest["candidate_digest"])
        or not _is_sha256(manifest["checkpoint_sha256"])
        or not _is_sha256(manifest["worker_capture_sha256"])
        or not isinstance(manifest["node"], str)
        or not manifest["node"]
        or not isinstance(manifest["pane"], str)
        or not manifest["pane"]
        or isinstance(manifest["world_count"], bool)
        or not isinstance(manifest["world_count"], int)
        or manifest["world_count"] < 0
        or not isinstance(manifest["world_artifacts"], list)
    ):
        raise ValueError("evaluation manifest identity/count fields are malformed")
    if manifest["world_count"] != len(manifest["world_artifacts"]):
        raise ValueError("evaluation manifest world count differs from its references")
    for reference in manifest["world_artifacts"]:
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "size"}:
            raise ValueError("evaluation manifest has a noncanonical world reference")
        if (
            not isinstance(reference["path"], str)
            or not _is_sha256(reference["sha256"])
            or isinstance(reference["size"], bool)
            or not isinstance(reference["size"], int)
            or reference["size"] < 0
        ):
            raise ValueError("evaluation manifest world reference identity is malformed")
        world_path = (path.parent / reference["path"]).resolve()
        if (
            not world_path.is_file()
            or world_path.stat().st_size != reference["size"]
            or _file_sha256(world_path) != reference["sha256"]
        ):
            raise ValueError("evaluation manifest world reference is missing or changed")
        yield world_path, _read_json(world_path)


def _world_context(
    world: dict[str, Any], identity: dict[str, Any], source_sha: str
) -> dict[str, Any]:
    return {
        "candidateDigest": world["candidate_digest"],
        "node": world["node"],
        "pane": world["pane"],
        "worldId": world["world_id"],
        "worldArtifactSha256": source_sha,
        "baseWorldSeed": identity["seed"],
        "withheld": identity["withheld"],
        "regime": identity["regime"],
        "trajectoryId": identity["trajectoryId"],
        "trajectoryDigest": identity["trajectoryDigest"],
    }


def postprocess(inputs: Iterable[Path | str], output: Path | str | None = None) -> dict[str, Any]:
    """Aggregate sealed streamed-world artifacts without executing model code."""

    paths = sorted({Path(path).resolve() for path in inputs}, key=lambda path: path.as_posix())
    if not paths:
        raise ValueError("at least one Gate-1 evaluation or world artifact is required")
    sources = []
    worlds: list[tuple[Path, dict[str, Any], dict[str, Any] | None]] = []
    for path in paths:
        raw = _read_json(path)
        kind = raw.get("kind")
        if kind == EVALUATION_KIND:
            manifest = _sealed(raw, EVALUATION_KIND, _EVALUATION_FIELDS)
            sources.append(
                {
                    "path": path.as_posix(),
                    "sha256": _file_sha256(path),
                    "kind": kind,
                    "digest": manifest["digest"],
                }
            )
            for world_path, world in _manifest_worlds(path, manifest):
                worlds.append((world_path, world, manifest))
        elif kind == WORLD_KIND:
            sources.append(
                {
                    "path": path.as_posix(),
                    "sha256": _file_sha256(path),
                    "kind": kind,
                    "digest": raw.get("digest"),
                }
            )
            worlds.append((path, raw, None))
        else:
            raise ValueError(f"unsupported Gate-1 input kind: {kind}")

    control_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    summaries: dict[tuple[Any, ...], dict[str, Any]] = {}
    compiler_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    manufacturing_rows: list[dict[str, Any]] = []
    trade_lifecycles: list[dict[str, Any]] = []
    trade_participants: list[dict[str, Any]] = []
    trade_groups: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    world_bindings = []
    seen_worlds: set[tuple[str, str, str, str]] = set()
    for world_path, raw_world, manifest in sorted(worlds, key=lambda item: item[0].as_posix()):
        world = _sealed(raw_world, WORLD_KIND, _WORLD_FIELDS)
        for field in ("candidate_digest", "node", "pane", "world_id"):
            if not isinstance(world[field], str) or not world[field]:
                raise ValueError(f"world artifact has invalid {field}")
        if not _is_sha256(world["candidate_digest"]):
            raise ValueError("world artifact candidate digest is invalid")
        if manifest is not None and any(
            world[field] != manifest[field] for field in ("candidate_digest", "node", "pane")
        ):
            raise ValueError("world artifact differs from its evaluation manifest binding")
        world_key = tuple(
            world[field] for field in ("candidate_digest", "node", "pane", "world_id")
        )
        if world_key in seen_worlds:
            raise ValueError("duplicate Gate-1 streamed world artifact")
        seen_worlds.add(world_key)
        source_sha = _file_sha256(world_path)
        identity, trajectory_rows = _validate_trajectory(world)
        base = _world_context(world, identity, source_sha)
        base.update(
            {
                "evaluationDigest": None if manifest is None else manifest["digest"],
                "checkpointSha256": (None if manifest is None else manifest["checkpoint_sha256"]),
                "workerCaptureSha256": (
                    None if manifest is None else manifest["worker_capture_sha256"]
                ),
            }
        )
        world_bindings.append({**base, "worldDigest": world["digest"]})

        by_step: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in trajectory_rows:
            by_step[item["stepIndex"]].append(item["economic"])
            row = item["economic"]
            context = {
                **base,
                "stepIndex": item["stepIndex"],
                "decisionTurn": item["sourceTurn"],
                "sourceFingerprint": item["sourceFingerprint"],
            }
            compiler_rows.extend(_compiler_rows(context, row, unresolved))
            inventory_rows.extend(_inventory_rows(context, row))
            manufacturing_rows.extend(_manufacturing_rows(context, row, unresolved))
            for path_name, reason in sorted(row["unavailable"].items()):
                unresolved.append(
                    {
                        **context,
                        "campId": row["campId"],
                        "sourceType": "economic-sidecar",
                        "sourceId": row["transitionId"],
                        "path": path_name,
                        "reason": reason,
                    }
                )
            for ledger in row["conservationRows"]:
                if not ledger["complete"]:
                    unresolved.append(
                        {
                            **context,
                            "campId": row["campId"],
                            "sourceType": "conservation-row",
                            "sourceId": f"{ledger['domain']}:{ledger['item']}",
                            "path": "residual",
                            "reason": ledger["unavailableReason"],
                        }
                    )
            for control in row["controls"]:
                group_identity = _group_identity(control)
                dimensions = {
                    "candidateDigest": base["candidateDigest"],
                    "node": base["node"],
                    "pane": base["pane"],
                    "withheld": base["withheld"],
                    "regime": base["regime"],
                    "decisionTurn": item["sourceTurn"],
                    "domain": control["domain"],
                    "control": group_identity,
                    "nativeUnit": control["nativeUnit"],
                }
                key = tuple(dimensions.values())
                bucket = control_groups.setdefault(key, _control_bucket(dimensions))
                _add_control(
                    bucket,
                    control,
                    world_seed=identity["seed"],
                    world_id=world["world_id"],
                    turn=item["sourceTurn"],
                    camp=row["campId"],
                )
                summary_dimensions = {
                    key: value for key, value in dimensions.items() if key != "decisionTurn"
                }
                summary_key = tuple(summary_dimensions.values())
                summary = summaries.setdefault(summary_key, _control_bucket(summary_dimensions))
                _add_control(
                    summary,
                    control,
                    world_seed=identity["seed"],
                    world_id=world["world_id"],
                    turn=item["sourceTurn"],
                    camp=row["campId"],
                )
                for phase, reason in control["unavailable"].items():
                    unresolved.append(
                        {
                            **context,
                            "campId": row["campId"],
                            "sourceType": "control",
                            "sourceId": control["controlId"],
                            "path": phase,
                            "reason": reason,
                            "classification": control["classification"],
                        }
                    )

        for step_index, economic_rows in sorted(by_step.items()):
            item = trajectory_rows[
                next(
                    i for i, value in enumerate(trajectory_rows) if value["stepIndex"] == step_index
                )
            ]
            context = {
                **base,
                "stepIndex": step_index,
                "decisionTurn": item["sourceTurn"],
                "sourceFingerprint": item["sourceFingerprint"],
            }
            lifecycle, participants, summary = _trade_rows(context, economic_rows, unresolved)
            trade_lifecycles.extend(lifecycle)
            trade_participants.extend(participants)
            trade_groups.append(summary)

    output_body = {
        "schema": 1,
        "kind": OUTPUT_KIND,
        "evidenceClass": "checkpoint-evaluation",
        "claimScope": "descriptive-economic-failure-decomposition-only",
        "causalCreditClaimed": False,
        "inferentialUnit": "base-world-seed",
        "intervalPolicy": (
            "not-computed; any later interval must cluster the four regimes by base-world seed"
        ),
        "sources": sorted(sources, key=lambda item: (item["kind"], item["sha256"], item["path"])),
        "worlds": sorted(
            world_bindings,
            key=lambda item: (item["candidateDigest"], item["node"], item["pane"], item["worldId"]),
        ),
        "controlGroups": sorted(
            (_finalize_control(bucket) for bucket in control_groups.values()),
            key=lambda item: tuple(
                str(item[field])
                for field in (
                    "candidateDigest",
                    "node",
                    "pane",
                    "withheld",
                    "decisionTurn",
                    "domain",
                    "control",
                    "nativeUnit",
                )
            ),
        ),
        "controlSummary": sorted(
            (_finalize_control(bucket) for bucket in summaries.values()),
            key=lambda item: tuple(
                str(item[field])
                for field in (
                    "candidateDigest",
                    "node",
                    "pane",
                    "withheld",
                    "domain",
                    "control",
                    "nativeUnit",
                )
            ),
        ),
        "compilerRows": sorted(
            compiler_rows,
            key=lambda item: (
                item["candidateDigest"],
                item["node"],
                item["pane"],
                item["worldId"],
                item["decisionTurn"],
                item["campId"],
                item["domain"],
                item["ordinal"],
            ),
        ),
        "compilerGroups": _compiler_groups(compiler_rows),
        "inventoryRows": sorted(
            inventory_rows,
            key=lambda item: tuple(
                str(item[field])
                for field in (
                    "candidateDigest",
                    "node",
                    "pane",
                    "worldId",
                    "decisionTurn",
                    "campId",
                    "domain",
                    "item",
                )
            ),
        ),
        "manufacturingRows": sorted(
            manufacturing_rows,
            key=lambda item: (
                item["candidateDigest"],
                item["node"],
                item["pane"],
                item["worldId"],
                item["decisionTurn"],
                item["campId"],
                item["request"]["ordinal"],
            ),
        ),
        "manufacturingGroups": _manufacturing_groups(manufacturing_rows),
        "tradeLifecycles": sorted(
            trade_lifecycles,
            key=lambda item: (
                item["candidateDigest"],
                item["node"],
                item["pane"],
                item["worldId"],
                item["decisionTurn"],
                item["requestId"],
            ),
        ),
        "tradeParticipants": sorted(
            trade_participants,
            key=lambda item: (
                item["candidateDigest"],
                item["node"],
                item["pane"],
                item["worldId"],
                item["decisionTurn"],
                item["transactionId"],
                item["campId"],
            ),
        ),
        "tradeGroups": sorted(
            trade_groups,
            key=lambda item: (
                item["candidateDigest"],
                item["node"],
                item["pane"],
                item["worldId"],
                item["decisionTurn"],
            ),
        ),
        "unresolved": sorted(
            unresolved,
            key=lambda item: tuple(
                str(item.get(field, ""))
                for field in (
                    "candidateDigest",
                    "node",
                    "pane",
                    "worldId",
                    "decisionTurn",
                    "campId",
                    "sourceType",
                    "sourceId",
                    "path",
                    "reason",
                )
            ),
        ),
        "limitations": [
            "cultivation and herding production are not attributed to terminal human consumption",
            "linked mixed food cannot be partitioned to component-food import lots",
            (
                "request inventory is unavailable unless an authoritative "
                "request-bound mutation exists"
            ),
            "shared-recipe request work or receipt may be unavailable",
            "ratios never cross units and zero denominators remain undefined",
            "ending population and QoL are observational context, not per-control causal credit",
            "delayed trade utility lacks a future transaction-bound causal receipt",
            (
                "inactive controls are exposure classifications; unsupported "
                "active-capability rows remain unresolved"
            ),
        ],
    }
    result = {**output_body, "digest": digest(output_body)}
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            json.dumps(
                result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
            + b"\n"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate sealed Gate-1 streamed worlds into descriptive economic failure chains"
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    postprocess(args.inputs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
