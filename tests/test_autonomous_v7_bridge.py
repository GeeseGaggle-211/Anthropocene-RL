"""Native contract-7 bridge tests (skipped unless the model9 bridge is built)."""

from __future__ import annotations

import base64
import copy
import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts, evaluate
from anthropocene_rl.autonomous_v7.bridge import DEFAULT_BRIDGE, Model9Bridge, observation_batch
from anthropocene_rl.autonomous_v7.gate1_diagnostics import (
    compiler_interventions,
    integrate_economic_rows,
    service_outcome_registry,
    service_outcomes,
    validate_economic_rows,
)

GATE1_SEMANTICS = {
    "experts": False,
    "token": False,
    "commitments": True,
    "learned_movement": False,
    "region_movement": False,
    "learned_fission": False,
    "learned_trade": False,
    "region_head": "hierarchical",
    "grouped_region": False,
    "region_selection": None,
}


def _native_gate1_checkpoint(tmp_path, core_width: int):
    from anthropocene_rl.autonomous_v7.model import Actor, Normalizer

    actor = Actor(core_width, region_head="hierarchical")
    normalizer = Normalizer(np.zeros(core_width, np.float32), np.ones(core_width, np.float32))
    for name, _, width in contracts.SETS:
        normalizer.set_mean[name] = np.zeros(width, np.float32)
        normalizer.set_std[name] = np.ones(width, np.float32)
    path = tmp_path / "native-gate1.pt"
    torch.save(
        {
            "contract": contracts.NATIVE_CONTRACT,
            "core_width": core_width,
            "actor": actor.state_dict(),
            "normalizer": normalizer.state_dict(),
        },
        path,
    )
    return path


def _teacher_actions(villages, *, prefer_teacher: bool = True, legalize_decoder: bool = False):
    actions = {}
    for village in villages:
        if village.get("observation") is None or village.get("terminated"):
            continue
        teacher = village.get("teacher") if prefer_teacher else None
        if teacher is None:
            economy = village["observation"]["economyMixMask"]
            if economy[0]:
                field, fishing = 1.0, 0.0
            elif economy[1]:
                field, fishing = 0.0, 0.0
            else:
                field, fishing = 0.5, 1.0
            strategic = [0.5] * (len(contracts.STRATEGIC) - 2) + [field, fishing]
            service = [0.5] * len(contracts.SERVICES)
            clothing = village["observation"]["clothingRouteMask"].index(True)
            shelter = village["observation"]["shelterRouteMask"].index(True)
        else:
            strategic, service = list(teacher["strategic"]), list(teacher["service"])
            clothing, shelter = teacher["clothingRoute"], teacher["shelterRoute"]
            if legalize_decoder:
                for vector, width in ((strategic, len(strategic) - 2), (service, len(service))):
                    for index in range(width):
                        value = np.float32(vector[index])
                        vector[index] = float(
                            0.0
                            if 0.0 < value < np.float32(0.01)
                            else 1.0
                            if np.float32(0.99) < value < 1.0
                            else value
                        )
                field, fishing = strategic[-2:]
                field, fishing = np.float32(field), np.float32(fishing)
                rest = np.float32(1.0) - fishing
                mix = np.asarray(
                    [[rest * field, rest * (np.float32(1.0) - field), fishing]], dtype=np.float32
                )
                mix[mix < np.float32(0.01)] = np.float32(0.0)
                mix /= np.maximum(mix.sum(axis=1, keepdims=True), np.float32(1e-12))
                from anthropocene_rl.autonomous_v7.model import mix_shares

                share, fishing = mix_shares(mix)
                strategic[-2:] = [float(share[0]), float(fishing[0])]
                if any(
                    value > 0 and not allowed
                    for value, allowed in zip(
                        mix[0], village["observation"]["economyMixMask"], strict=True
                    )
                ):
                    economy = village["observation"]["economyMixMask"]
                    field, fishing = (
                        (1.0, 0.0) if economy[0] else (0.0, 0.0) if economy[1] else (0.0, 1.0)
                    )
                    if fishing == 1.0:
                        field = 0.5
                    strategic[-2:] = [field, fishing]
                if not village["observation"]["clothingRouteMask"][clothing]:
                    clothing = village["observation"]["clothingRouteMask"].index(True)
                if not village["observation"]["shelterRouteMask"][shelter]:
                    shelter = village["observation"]["shelterRouteMask"].index(True)
        actions[int(village["camp"])] = {
            "strategic": strategic,
            "service": service,
            "clothingRoute": clothing,
            "shelterRoute": shelter,
            "movement": contracts.NATIVE_MOVEMENT,
            "fission": -1,
            "commit": contracts.SPAN_SINGLE,
        }
    return actions


pytestmark = pytest.mark.skipif(
    not os.path.exists(os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE", str(DEFAULT_BRIDGE))),
    reason="model9 bridge not built",
)


def test_reset_step_and_replay_through_the_native_contract() -> None:
    with Model9Bridge() as bridge:
        assert bridge.schema.core_width == len(bridge.schema.core_features)
        step = bridge.reset(17)
        villages = [v for v in step["villages"] if v["observation"] is not None]
        assert villages, "the opening boundary has resident-economy villages"
        obs = observation_batch(villages)
        assert obs["core"].shape == (len(villages), bridge.schema.core_width)
        for name, rows, width in contracts.SETS:
            assert obs[name].shape == (len(villages), rows, width)
        teacher = bridge.step({})
        labelled = [v for v in teacher["villages"] if v["teacher"] is not None]
        assert labelled and all(
            len(v["teacher"]["service"]) == len(contracts.SERVICES) for v in labelled
        )
        assert all(np.isfinite(v["welfare"]) for v in teacher["villages"])
        # The teacher's own labels as actions: the contract must accept them and keep replay exact.
        actions = {
            v["camp"]: {
                "strategic": v["teacher"]["strategic"],
                "service": v["teacher"]["service"],
                "clothingRoute": v["teacher"]["clothingRoute"],
                "shelterRoute": v["teacher"]["shelterRoute"],
                "movement": contracts.NATIVE_MOVEMENT,
                "fission": -1,
            }
            for v in labelled
        }
        bridge.step(actions)
        assert bridge.verify_replay() == 2


def test_economic_diagnostics_are_bitwise_reporting_only() -> None:
    with Model9Bridge() as ordinary, Model9Bridge(economic_diagnostics=True) as diagnostic:
        left = ordinary.reset(19)
        right = diagnostic.reset(19)
        source_villages = copy.deepcopy(right["villages"])
        trajectory_id, decision_turn = right["fingerprint"], right["turn"]
        assert left["fingerprint"] == right["fingerprint"]
        assert left["villages"] == right["villages"]
        assert ordinary.economic_diagnostics == diagnostic.economic_diagnostics == {}
        left = ordinary.step({})
        right = diagnostic.step({})
        assert left["fingerprint"] == right["fingerprint"]
        assert left["villages"] == right["villages"]
        assert ordinary.verify_replay() == diagnostic.verify_replay() == 1
        economic = [
            diagnostic.economic_diagnostics[camp]
            for camp in sorted(diagnostic.economic_diagnostics)
        ]
        assert (
            validate_economic_rows(
                economic,
                trajectory_id=trajectory_id,
                decision_turn=decision_turn,
                source_villages=source_villages,
                successor_villages=right["villages"],
            )
            == economic
        )
        integrated = integrate_economic_rows(economic)
        assert integrated["population_years"] >= 0 and integrated["qol_years"] >= 0
        services = service_outcomes(economic, service_outcome_registry())
        assert services and all(value["value"] is not None for value in services.values())
        count, summary = compiler_interventions(economic)
        assert count == summary["total"] == len(summary["raw_rows"])


def test_ordinary_rollout_does_not_capture_gate1_evidence(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("diagnostic-off rollout deep-copied retained Gate-1 evidence")

    monkeypatch.setattr(evaluate, "copy", SimpleNamespace(deepcopy=forbidden))
    result = evaluate.rollout(29, 1, None, requested_metrics=[])
    assert "trajectory_evidence" not in result
    assert "capture_receipt" not in result
    assert result["evaluation_steps"] == 0


def test_diagnostic_rollout_rejects_capture_before_exceeding_reviewed_bytes() -> None:
    bounds = {
        "max_steps": 1,
        "max_villages_per_boundary": 4096,
        "max_economic_rows_per_step": 4096,
        "max_canonical_bytes": 1,
    }
    with pytest.raises(ValueError, match="canonical-byte capture bound"):
        evaluate.rollout(
            31,
            1,
            None,
            requested_metrics=["welfare"],
            service_fields=service_outcome_registry(),
            capture_bounds=bounds,
            evaluation_world_id="bounded-live-rollout",
            expected_reset_turn=8,
            observation_core_width=1,
            evaluation_semantics={
                "experts": False,
                "token": False,
                "commitments": True,
                "learned_movement": False,
                "region_movement": False,
                "learned_fission": False,
                "learned_trade": False,
                "region_head": "hierarchical",
                "grouped_region": False,
                "region_selection": None,
            },
        )


def test_snapshot_reset_resumes_exactly_once_with_empty_reset_sidecar() -> None:
    with Model9Bridge(economic_diagnostics=True) as source:
        source.reset(23)
        captured = source.step({})
        snapshot = source.capture()
        expected = source.step({})
    with Model9Bridge(economic_diagnostics=True) as restored:
        opening = restored.reset(23, snapshot=snapshot)
        assert opening["turn"] == captured["turn"]
        assert opening["fingerprint"] == captured["fingerprint"]
        assert restored.economic_diagnostics == {}
        actual = restored.step({})
        assert actual["turn"] == expected["turn"]
        assert actual["fingerprint"] == expected["fingerprint"]

        def semantic(villages):
            return [
                {
                    key: village[key]
                    for key in (
                        "camp",
                        "society",
                        "population",
                        "qol",
                        "nutrition",
                        "clothing",
                        "housing",
                        "years",
                        "welfare",
                        "terminated",
                        "observation",
                    )
                }
                for village in villages
            ]

        assert semantic(actual["villages"]) == semantic(expected["villages"])


@pytest.mark.parametrize("warmup,expect_founding", [(0, False), (4, True)])
def test_real_native_gate1_rollout_accepts_reset_warmup_and_founding_boundaries(
    tmp_path, warmup, expect_founding
) -> None:
    from anthropocene_rl.autonomous_v7.worker_contract import world

    with Model9Bridge() as probe:
        core_width = probe.schema.core_width
    checkpoint = _native_gate1_checkpoint(tmp_path, core_width)
    bounds = {
        "max_steps": 1,
        "max_villages_per_boundary": 4096,
        "max_economic_rows_per_step": 4096,
        "max_canonical_bytes": 256 * 1024 * 1024,
    }
    descriptor = world(19, turn=None, players=8, movement=False, head="hierarchical")
    result = evaluate.rollout(
        19,
        1,
        str(checkpoint),
        warmup=warmup,
        strict_checkpoint=True,
        resolved_semantics=GATE1_SEMANTICS,
        evaluation_semantics=GATE1_SEMANTICS,
        requested_metrics=["welfare"],
        service_fields=service_outcome_registry(),
        capture_bounds=bounds,
        evaluation_world_id=f"native-seed19-warmup-{warmup}",
        expected_reset_turn=9,
        observation_core_width=core_width,
        world_contract=descriptor,
    )
    evidence = result["trajectory_evidence"]
    assert evidence["opening"]["turn"] == 9 + warmup
    assert len(evidence["reset"]["warmup_chain"]) == warmup
    founded = [
        village for village in evidence["steps"][0]["successor"]["villages"] if village["founded"]
    ]
    assert bool(founded) is expect_founding
    if founded:
        assert all(village["years"] == 0 for village in founded)
        witnessed = {
            transfer["destinationCamp"]
            for row in evidence["steps"][0]["economic"]
            for transfer in row["transfers"]
            if transfer.get("kind") == "fission"
        }
        assert {village["camp"] for village in founded} == witnessed


def test_real_native_gate1_rollout_accepts_reciprocal_migration_endpoints(
    tmp_path, monkeypatch
) -> None:
    from anthropocene_rl.autonomous_v7.worker_contract import world

    with Model9Bridge() as source:
        step = source.reset(19)
        step = source.step({})
        step = source.step(_teacher_actions(step["villages"]))
        assert step["turn"] == 11
        snapshot = base64.b64decode(source.capture())
        core_width = source.schema.core_width
    checkpoint = _native_gate1_checkpoint(tmp_path, core_width)

    def teacher_policy(_actor, _normalizer, villages, *_args, **kwargs):
        from tests.test_gate1_diagnostics import forward_action_and_receipt

        requested = _teacher_actions(villages, legalize_decoder=True)
        actions = {}
        receipts = kwargs["decoder_receipts"]
        live = [village for village in villages if int(village["camp"]) in requested]
        for row, village in enumerate(live):
            camp = int(village["camp"])
            actions[camp], receipts[camp] = forward_action_and_receipt(
                requested[camp],
                village["observation"],
                kwargs["decoder_semantics"],
                camp=camp,
                batch_rows=len(live),
                row=row,
            )
        return actions

    monkeypatch.setattr(evaluate, "policy_actions", teacher_policy)

    snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
    descriptor = world(
        19, snapshot_sha256=snapshot_sha256, turn=11, players=8, movement=False, head="hierarchical"
    )
    bounds = {
        "max_steps": 2,
        "max_villages_per_boundary": 4096,
        "max_economic_rows_per_step": 4096,
        "max_canonical_bytes": 256 * 1024 * 1024,
    }
    result = evaluate.rollout(
        19,
        2,
        str(checkpoint),
        warmup=1,
        snapshot=snapshot,
        strict_checkpoint=True,
        resolved_semantics=GATE1_SEMANTICS,
        evaluation_semantics=GATE1_SEMANTICS,
        requested_metrics=["welfare"],
        service_fields=service_outcome_registry(),
        capture_bounds=bounds,
        evaluation_world_id="native-seed19-migration",
        expected_reset_turn=11,
        observation_core_width=core_width,
        world_contract=descriptor,
    )
    migrations = [
        (row["campId"], transfer)
        for step in result["trajectory_evidence"]["steps"]
        for row in step["economic"]
        for transfer in row["transfers"]
        if transfer.get("kind") == "migration"
    ]
    assert migrations
    for transfer in {tuple(sorted(item.items())) for _camp, item in migrations}:
        decoded = dict(transfer)
        endpoints = sorted(camp for camp, item in migrations if item == decoded)
        assert endpoints == sorted([decoded["sourceCamp"], decoded["destinationCamp"]])
