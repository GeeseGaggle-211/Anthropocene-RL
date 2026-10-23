from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_gate2_accounting import _control, reciprocal_endpoint, sidecar, transition_v2, with_trade

from anthropocene_rl.autonomous_v7.gate1_economic_postprocess import postprocess
from anthropocene_rl.autonomous_v7.provenance import digest


def _seal(value: dict) -> dict:
    return {**value, "digest": digest(value)}


def _replace(value, old: str, new: str):
    if isinstance(value, dict):
        return {key: _replace(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace(item, old, new) for item in value]
    if isinstance(value, str):
        return value.replace(old, new)
    return value


def _bound_sidecar(*, quantity: float = 2.0) -> dict:
    value = sidecar()
    old_trajectory = value["trajectoryId"]
    old_transition = value["transitionId"]
    trajectory = hashlib.sha256(b"gate1-postprocess-trajectory").hexdigest()
    transition = f"{trajectory}/turn/1/camp/7"
    value = _replace(value, old_transition, transition)
    value = _replace(value, old_trajectory, trajectory)
    value["trajectoryId"] = trajectory
    value["transitionId"] = transition
    value["decisionTurn"] = 1

    request_id = f"{transition}/request/0"
    value["domains"]["manufacturing"]["active"] = True
    value["domains"]["manufacturing"]["funnels"] = [
        {
            "metric": "clothing",
            "unit": "selected-route-clothing-units",
            "nativeDemand": quantity,
            "actorCappedRequest": quantity,
            "compilerGrant": quantity,
            "executedWork": None,
            "physicalReceipt": None,
            "finalService": None,
            "active": True,
            "constraintReasons": ["storage"],
        }
    ]
    for phase in ("executedWork", "physicalReceipt", "finalService"):
        value["unavailable"][f"domains.manufacturing.funnels[0].{phase}"] = (
            "compiler-control-unit-is-not-a-physical-good-or-work-unit"
        )
    value["requests"] = [
        {
            "requestId": request_id,
            "ordinal": 0,
            "service": 1,
            "serviceClass": "Clothing",
            "good": "Cloth",
            "role": "Tailor",
            "rawMaterial": None,
            "nativeDemand": quantity,
            "actorCappedRequest": quantity,
            "compilerGrant": quantity,
            "unmetReason": "",
            "serviceEquivalentPerUnit": 1.0,
            "nativeUnit": "good-units",
            "reservedFabric": 0.0,
            "reservedHousingMaterial": 0.0,
            "reservedCloth": 0.0,
            "workerYearsEquivalent": quantity / 2,
            "executedWorkerYears": quantity / 2,
            "committedPhysicalReceipt": quantity,
            "inventoryChange": None,
            "consumptionOrFinalService": 0.0,
            "openingStockDraw": 0.0,
            "importedStockDraw": 0.0,
            "transferInStockDraw": 0.0,
            "productionServiceDraw": 0.0,
            "productionReceiptAllocation": quantity,
        }
    ]
    control = _control(request_id)
    control.update(
        controlId=f"{transition}/control/request/0",
        domain="manufacturing",
        identity=request_id,
        nativeUnit="good-units",
        actorHead="service[1]",
        classification="active",
        classificationReason=None,
        metadata={
            "service": 1,
            "serviceClass": "Clothing",
            "good": "Cloth",
            "rawMaterial": None,
            "role": "Tailor",
        },
    )
    control["phases"].update(
        nativeDemand=quantity,
        actorCappedRequest=quantity,
        compilerGrant=quantity,
        executedWorkerOrInputClaim=quantity / 2,
        committedPhysicalReceipt=quantity,
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
        output=quantity,
        endingStock=quantity,
        beforeFinalTradeStock=quantity,
        afterFinalTradeStock=quantity,
        workerIntervals=quantity / 2,
        workerYears=quantity / 2,
    )
    ledger = next(
        row
        for row in value["conservationRows"]
        if row["domain"] == "craft-good" and row["item"] == "Cloth"
    )
    ledger.update(production=quantity, closing=quantity, residual=0.0)
    next(
        event
        for event in value["physicalEvents"]
        if event["category"] == "ending-stock-boundary" and event["item"] == "Cloth"
    )["quantity"] = quantity
    value["physicalEvents"].extend(
        [
            {
                "eventId": f"{trajectory}/production/Cloth",
                "campId": 7,
                "category": "production-receipt",
                "item": "Cloth",
                "unit": "good-units",
                "quantity": quantity,
                "direction": "in",
                "receiptId": f"{transition}/receipt/0",
                "sourceRequestId": request_id,
            },
            {
                "eventId": f"{trajectory}/service-final/Cloth",
                "campId": 7,
                "category": "final-service-fulfilment",
                "item": "Cloth",
                "unit": "good-units",
                "quantity": 0.0,
                "direction": "out",
                "phase": "needs-after-production",
            },
        ]
    )
    return value


def _boundary(turn: int, fingerprint: str, camps: list[tuple[int, int]]) -> dict:
    return {
        "turn": turn,
        "fingerprint": fingerprint,
        "villages": [{"camp": camp, "society": society} for camp, society in camps],
    }


def _world(
    economics: list[dict],
    *,
    world_id: str = "s2000-default",
    seed: int = 2000,
    withheld: int = 0,
    node: str = "node-a",
    pane: str = "fresh-common-producer-diagnostic",
) -> dict:
    trajectory = economics[0]["trajectoryId"]
    opening = _boundary(1, trajectory, [(row["campId"], row["societyId"]) for row in economics])
    successor_fingerprint = hashlib.sha256(b"gate1-postprocess-successor").hexdigest()
    successor = _boundary(
        2, successor_fingerprint, [(row["campId"], row["societyId"]) for row in economics]
    )
    transition_id = digest(
        {
            "index": 0,
            "source_turn": 1,
            "source_fingerprint": trajectory,
            "successor_turn": 2,
            "successor_fingerprint": successor_fingerprint,
        }
    )
    receipt = {
        "steps": 1,
        "warmup_steps": 0,
        "max_villages_observed": len(economics),
        "max_economic_rows_observed": len(economics),
        "canonical_bytes": 1,
    }
    evidence_body = {
        "schema": 2,
        "kind": "gate1_canonical_trajectory",
        "ordering": (
            "step-order; boundary/action/economic identities canonicalized by registered key"
        ),
        "reset": {
            "world_id": world_id,
            "snapshot_sha256": "b" * 64,
            "turn": 1,
            "fingerprint": trajectory,
            "trajectory_id": trajectory,
            "warmup_chain": [],
        },
        "bounds": {"fixture": 1},
        "opening": opening,
        "steps": [
            {
                "index": 0,
                "transition_id": transition_id,
                "source_turn": 1,
                "source_fingerprint": trajectory,
                "actions": [
                    {
                        "camp": row["campId"],
                        "action_request": {},
                        "decoded_action": {},
                        "decoder_receipt": {},
                    }
                    for row in economics
                ],
                "policy_rng_receipt": None,
                "successor": successor,
                "terminal": {
                    "terminated": False,
                    "truncated": True,
                    "reason": "evaluation-horizon",
                },
                "economic": economics,
            }
        ],
        "capture_receipt": receipt,
    }
    evidence = {**evidence_body, "digest": digest(evidence_body)}
    descriptor = _seal(
        {
            "seed": seed,
            "withheld": withheld,
            "snapshot_sha256": "b" * 64,
            "turn": 1,
            "players": 8,
            "movement": False,
            "region_head": None,
            "cohort_digest": "c" * 64,
        }
    )
    result = {
        "seed": seed,
        "withheld": withheld,
        "world_contract": descriptor,
        "trajectory_evidence": evidence,
        "trajectory_digest": evidence["digest"],
        "evaluation_steps": 1,
        "capture_receipt": receipt,
    }
    return _seal(
        {
            "schema": 1,
            "kind": "gate1_fresh_common_producer_world",
            "candidate_digest": "a" * 64,
            "node": node,
            "pane": pane,
            "world_id": world_id,
            "result": result,
        }
    )


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _manifest(path: Path, world_path: Path, world: dict) -> Path:
    raw = world_path.read_bytes()
    body = {
        "schema": 1,
        "kind": "gate1_fresh_common_producer_evaluation",
        "candidate_digest": world["candidate_digest"],
        "node": world["node"],
        "pane": world["pane"],
        "checkpoint_sha256": "d" * 64,
        "worker_capture_sha256": "e" * 64,
        "producer_scope": "fresh-common-producer-diagnostic",
        "node_evidence_scope": "recovered-artifact",
        "historical_artifact_claim": True,
        "historical_producer_status": "unresolved",
        "historical_replay_claim": False,
        "selection_allowed": False,
        "readiness_claimed": False,
        "world_count": 1,
        "world_artifacts": [
            {
                "path": world_path.relative_to(path.parent).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        ],
        "streaming": {
            "maximum_in_memory_world_results": 1,
            "canonical_capture_bound_per_world_bytes": 1000,
        },
    }
    return _write(path, _seal(body))


def test_postprocess_reports_denominators_unit_mismatch_inventory_and_unresolved(
    tmp_path: Path,
) -> None:
    world_path = _write(tmp_path / "world.json", _world([_bound_sidecar()]))
    output_path = tmp_path / "decomposition.json"
    result = postprocess([world_path], output_path)

    assert output_path.is_file()
    assert result["digest"] == digest(
        {key: value for key, value in result.items() if key != "digest"}
    )
    assert result["inferentialUnit"] == "base-world-seed"
    manufacturing = next(row for row in result["controlGroups"] if row["domain"] == "manufacturing")
    assert manufacturing["baseWorldCount"] == 1
    assert manufacturing["campTurnCount"] == 1
    assert manufacturing["rates"]["positiveNativeDemand"] == {
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
        "undefinedReason": None,
    }
    assert manufacturing["unitSafeLinks"]["compilerGrantToPhysicalReceipt"]["value"] == 1.0
    work_link = manufacturing["unitSafeLinks"]["compilerGrantToExecutedClaim"]
    assert work_link["value"] is None
    assert work_link["excludedReasons"][0]["reason"] == "unit-mismatch:good-units->worker-years"
    request = result["manufacturingRows"][0]
    assert request["requestBoundProductionReceipts"] == [
        {"unit": "good-units", "quantity": 2.0, "eventCount": 1}
    ]
    assert request["inventoryAttributionAvailable"] is False
    compiler = next(
        row
        for row in result["compilerGroups"]
        if row["domain"] == "manufacturing" and row["metric"] == "clothing"
    )
    assert compiler["rates"]["constraintAmongPositiveRequests"] == {
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
        "undefinedReason": None,
    }
    assert compiler["constraintReasons"] == [{"reason": "storage", "count": 1}]
    assert any(
        row["sourceType"] == "manufacturing-request" and row["path"] == "inventoryChange"
        for row in result["unresolved"]
    )
    cloth = next(
        row
        for row in result["inventoryRows"]
        if row["domain"] == "craft-good" and row["item"] == "Cloth"
    )
    assert cloth["conservation"]["production"] == 2.0
    assert cloth["conservation"]["closing"] == 2.0


def test_zero_demand_has_explicit_zero_denominator(tmp_path: Path) -> None:
    world_path = _write(tmp_path / "world.json", _world([_bound_sidecar(quantity=0.0)]))
    result = postprocess([world_path])
    manufacturing = next(row for row in result["controlGroups"] if row["domain"] == "manufacturing")
    assert manufacturing["rates"]["positiveNativeDemand"]["denominator"] == 1
    assert manufacturing["rates"]["positiveNativeDemand"]["numerator"] == 0
    link = manufacturing["unitSafeLinks"]["compilerGrantToPhysicalReceipt"]
    assert link["value"] is None
    assert link["undefinedReason"] == "no-unit-compatible-positive-denominator-rows"
    assert link["excludedReasons"] == [{"reason": "zero-or-negative-denominator", "count": 1}]


def test_manifest_binds_candidate_node_pane_world_hash_and_descriptor_regime(
    tmp_path: Path,
) -> None:
    world = _world([_bound_sidecar()])
    world_path = _write(tmp_path / "worlds" / "s2000-default.json", world)
    manifest_path = _manifest(tmp_path / "evaluation.json", world_path, world)
    result = postprocess([manifest_path])
    assert result["worlds"][0]["candidateDigest"] == "a" * 64
    assert result["worlds"][0]["regime"] == "default"

    bad = copy.deepcopy(world)
    bad["world_id"] = "s2000-herding"
    bad_result = bad["result"]
    bad_evidence = bad_result["trajectory_evidence"]
    bad_evidence["reset"]["world_id"] = bad["world_id"]
    bad_evidence["digest"] = digest(
        {key: value for key, value in bad_evidence.items() if key != "digest"}
    )
    bad_result["trajectory_digest"] = bad_evidence["digest"]
    bad["digest"] = digest({key: value for key, value in bad.items() if key != "digest"})
    bad_path = _write(tmp_path / "bad-world.json", bad)
    with pytest.raises(ValueError, match="regime suffix"):
        postprocess([bad_path])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["world_artifacts"][0]["sha256"] = "f" * 64
    manifest["digest"] = digest({key: value for key, value in manifest.items() if key != "digest"})
    changed_manifest = _write(tmp_path / "changed-evaluation.json", manifest)
    with pytest.raises(ValueError, match="missing or changed"):
        postprocess([changed_manifest])


def _trade_economics() -> list[dict]:
    first = transition_v2()
    old_trajectory = first["trajectoryId"]
    trajectory = hashlib.sha256(b"gate1-postprocess-trajectory").hexdigest()
    first = _replace(first, old_trajectory, trajectory)
    first["trajectoryId"] = trajectory
    first["economicDiagnostics"]["trajectoryId"] = trajectory
    first["economicDiagnostics"] = with_trade(first["economicDiagnostics"])
    second = reciprocal_endpoint(first)
    return [first["economicDiagnostics"], second["economicDiagnostics"]]


def test_bilateral_dedup_preserves_participant_signed_flows_and_delayed_gap(
    tmp_path: Path,
) -> None:
    economics = _trade_economics()
    world_path = _write(tmp_path / "trade-world.json", _world(economics))
    result = postprocess([world_path])
    assert len(result["tradeLifecycles"]) == 1
    assert len(result["tradeParticipants"]) == 2
    participants = {row["campId"]: row for row in result["tradeParticipants"]}
    assert sorted(flow["signedQuantity"] for flow in participants[7]["goodsFlows"]) == [
        -2.25,
        1.5,
    ]
    assert sorted(flow["signedQuantity"] for flow in participants[8]["goodsFlows"]) == [
        -1.5,
        2.25,
    ]
    assert participants[7]["immediateUtility"] == 0.5
    assert participants[8]["immediateUtility"] == 0.5
    assert result["tradeGroups"][0]["transactionCount"] == 1
    assert result["tradeGroups"][0]["refundedOrReturnedByGood"] == {
        "Cloth": 0.5,
        "Pots": 0.75,
    }
    assert (
        sum(
            row["sourceType"] == "trade-transaction" and row["path"] == "delayedUtility"
            for row in result["unresolved"]
        )
        == 1
    )


def test_conflicting_bilateral_endpoint_copy_is_rejected(tmp_path: Path) -> None:
    economics = _trade_economics()
    transaction = next(row for row in economics[1]["trades"] if row["recordType"] == "transaction")
    transaction["requesterPlannedGain"] = 9.0
    world_path = _write(tmp_path / "forged-trade-world.json", _world(economics))
    with pytest.raises(ValueError, match="endpoint copies disagree"):
        postprocess([world_path])


def test_one_trade_endpoint_is_retained_but_explicitly_unresolved(tmp_path: Path) -> None:
    economics = _trade_economics()[:1]
    world_path = _write(tmp_path / "one-endpoint-world.json", _world(economics))
    result = postprocess([world_path])
    lifecycle = result["tradeLifecycles"][0]
    assert lifecycle["endpointCoverageComplete"] is False
    assert lifecycle["transactionObservedInCamps"] == [7]
    assert any(
        row["sourceType"] == "trade-transaction"
        and row["path"] == "endpointCoverage"
        and "missing-reciprocal-endpoint-copies:8" in row["reason"]
        for row in result["unresolved"]
    )


def test_world_and_trajectory_digests_fail_closed(tmp_path: Path) -> None:
    world = _world([_bound_sidecar()])
    world["result"]["trajectory_evidence"]["steps"][0]["economic"][0]["years"] = 99.0
    world_path = _write(tmp_path / "forged-world.json", world)
    with pytest.raises(ValueError, match="not content-addressed"):
        postprocess([world_path])
