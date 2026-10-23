from __future__ import annotations

import base64
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from anthropocene_rl.autonomous_v7 import economic_trace_training_smoke as smoke
from anthropocene_rl.autonomous_v7 import optimizer_exact_replay as exact_replay
from anthropocene_rl.autonomous_v7.optimizer_exact_replay import (
    RETAINED_TURNS,
    SmokeAuditRetention,
    decoded_unit_neighborhood_probability,
    replay_retained_snapshots,
    write_retention_package,
)


class _FakeBridge:
    def __init__(self, *, drift_turn: int | None = None) -> None:
        self.normalized = False
        self.drift_turn = drift_turn
        self.current = self._step(9, initial=True)
        self.closed = False
        self.forks: dict[int, dict] = {}

    def _step(self, turn: int, *, initial: bool = False, restored: bool = False) -> dict:
        trajectory = None if initial else ("branch-lineage" if self.normalized else "root-lineage")
        population = turn + (1 if self.drift_turn == turn else 0)
        village = {
            "camp": 11,
            "society": 27,
            "parentSociety": 0,
            "player": 0,
            "priorPopulation": turn - 1,
            "population": population,
            "qol": 0.5,
            "nutrition": 1.0,
            "years": 20.0,
            "welfare": float(population * 10),
            "terminated": False,
            "tradeUtility": 0.0,
            "observation": {"core": [float(turn)]},
            "decisionDiagnostics": None
            if initial
            else {
                "trajectoryId": trajectory,
                "decisionTurn": turn - 1,
                "campId": 11,
            },
        }
        economic = (
            {}
            if initial
            else {
                "11": {
                    "trajectoryId": trajectory,
                    "transitionId": f"{trajectory}/{turn - 1}/11",
                    "sourceTransitionId": f"{trajectory}/{turn - 1}/11",
                    "parentTransitionId": None if not self.normalized else "fork-parent",
                    "forkId": None if not self.normalized else "fork-1",
                    "branchId": None if not self.normalized else "branch-1",
                    "eventId": f"{trajectory}/event/11",
                    "quantity": population,
                }
            }
        )
        return {
            "turn": turn,
            "years": 20.0,
            "fingerprint": f"{'canonical' if self.normalized else 'root'}-{turn}",
            "nativeMs": turn * (2 if self.normalized else 1),
            "diagnosticSchemaVersion": 0 if initial else 3,
            "trajectoryId": trajectory,
            "rawMaterialDiagnostics": {},
            "villages": [village],
            "restoredObservation": restored,
            "economicDiagnostics": economic,
        }

    def reset(self, _seed: int, *, snapshot: bytes | None = None, **_kwargs) -> dict:
        turn = int(snapshot.decode()) if snapshot is not None else 9
        self.normalized = snapshot is not None
        self.current = self._step(turn, initial=turn == 9 and snapshot is None)
        return copy.deepcopy(self.current)

    def capture(self) -> bytes:
        return base64.b64encode(str(self.current["turn"]).encode())

    def step(self, actions: dict) -> dict:
        turn = int(self.current["turn"]) + 1
        assert len(actions) == 1
        self.current = self._step(turn)
        return copy.deepcopy(self.current)

    def fork(self) -> tuple[int, dict]:
        handle = len(self.forks) + 1
        self.normalized = True
        self.current = self._step(
            int(self.current["turn"]),
            initial=int(self.current["turn"]) == 9,
            restored=True,
        )
        self.forks[handle] = copy.deepcopy(self.current)
        return handle, copy.deepcopy(self.current)

    def restore(self, handle: int) -> dict:
        self.current = copy.deepcopy(self.forks[handle])
        return copy.deepcopy(self.current)

    def drop(self, handle: int) -> None:
        del self.forks[handle]

    def close(self) -> None:
        self.closed = True


def _fake_ppo() -> SimpleNamespace:
    def to_tensors(batch: dict, *_args, **_kwargs) -> dict:
        return {"core": torch.as_tensor(batch["core"], dtype=torch.float32)}

    def explore(output: dict, *_args, **_kwargs) -> dict:
        return dict(output)

    def to_bridge_actions(draw: dict, _tensors: dict) -> list[dict]:
        return copy.deepcopy(draw["decoded"])

    return SimpleNamespace(
        Model9Bridge=lambda *_args, **_kwargs: _FakeBridge(),
        to_tensors=to_tensors,
        explore=explore,
        to_bridge_actions=to_bridge_actions,
    )


def _retained_payload() -> tuple[dict, _FakeBridge]:
    ppo = _fake_ppo()
    actor = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(4)
    rng = np.random.default_rng(5)
    original_functions = (
        ppo.Model9Bridge,
        ppo.to_tensors,
        ppo.explore,
        ppo.to_bridge_actions,
        optimizer.step,
    )
    capture = SmokeAuditRetention(ppo, actor, optimizer, generator, rng)
    with capture:
        bridge = ppo.Model9Bridge()
        bridge.reset(50000)
        for turn in RETAINED_TURNS:
            batch = {"core": np.asarray([[float(turn)]], dtype=np.float32)}
            tensors = ppo.to_tensors(batch, None)
            ppo.explore({"strategic_mean": tensors["core"]}, -2.0)
            decoded = [{"strategic": [0.0, 0.0, turn / 100.0]}]
            actions = ppo.to_bridge_actions({"decoded": decoded}, tensors)
            bridge.step({11: actions[0]})
        loss = actor(torch.ones(1, 1)).sum()
        loss.backward()
        optimizer.step()
    assert ppo.Model9Bridge is original_functions[0]
    assert ppo.to_tensors is original_functions[1]
    assert ppo.explore is original_functions[2]
    assert ppo.to_bridge_actions is original_functions[3]
    assert optimizer.step == original_functions[4]
    rollout = SimpleNamespace(
        obs=[{"core": np.ones((1, 1), dtype=np.float32)}],
        draws=[{"z": torch.ones(1)}],
        logp=[np.ones(1, dtype=np.float32)],
        outcome={"camp": [11]},
    )
    payload = capture.payload(
        normalizer_state={"core_mean": [0.0], "core_std": [1.0]},
        rollout=rollout,
        stage=SimpleNamespace(name="mixed"),
        log_std_floor=-2.0,
    )
    return payload, bridge._bridge


def test_retention_is_bounded_and_binds_fresh_adam_first_step() -> None:
    torch.manual_seed(2)
    payload, _ = _retained_payload()

    assert [row["preTurn"] for row in payload["records"]] == list(RETAINED_TURNS)
    assert all(len(row["completeActionMap"]) == 1 for row in payload["records"])
    assert all("rawPolicyBatch" in row for row in payload["records"])
    first = payload["firstActorStep"]
    assert first["adamBefore"]["state"] == {}
    assert first["adamAfter"]["state"]
    assert first["traceActorHashBefore"] != first["traceActorHashAfter"]
    assert payload["neighborhoodPlan"]["densityKeptSeparate"] is True
    assert payload["neighborhoodPlan"]["finitePrecisionExecutionExact"] is False
    assert payload["resetReconstruction"]["arguments"] == {
        "seed": 50000,
        "players": 8,
        "movement": False,
        "withheld": 0,
        "size": None,
        "snapshotProvided": False,
    }


def test_retained_package_verifies_observed_prefix_content(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _ = _retained_payload()
    root = tmp_path / "retention"
    manifest = write_retention_package(
        root, payload, {"status": "observed-reference-match"}
    )

    monkeypatch.setattr(
        exact_replay,
        "_verify_retained_policy_context",
        lambda _current, record, _state: {
            "campIdsInPolicyRowOrder": record["campIdsInPolicyRowOrder"],
            "rawPolicyBatchDigest": record["rawPolicyBatchDigest"],
            "packedPolicyInputDigest": record["packedPolicyInputDigest"],
        },
    )
    result = replay_retained_snapshots(
        root,
        _FakeBridge,
        world_seed=50000,
        movement=False,
        withheld=0,
    )
    assert result["status"] == "observed-step-policy-reward-prefix-verified"
    assert [row["turn"] for row in result["records"]] == list(RETAINED_TURNS)
    assert all(
        row["observedDecisionPolicyRewardMatch"]
        and row["forkPolicyContextMatch"]
        and row["fullWorldEquivalenceProven"] is False
        for row in result["records"]
    )
    first_mapping = result["records"][0]["forkIdentity"]["branchMetadata"]
    assert first_mapping["recorded"]["fingerprint"] == "root-9"
    assert first_mapping["replayed"]["fingerprint"] == "canonical-9"
    assert len(manifest["snapshots"]) == 6


def test_retained_replay_rejects_nonmetadata_physical_drift(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _ = _retained_payload()
    root = tmp_path / "retention"
    write_retention_package(root, payload, {"status": "reference-bound"})
    monkeypatch.setattr(
        exact_replay,
        "_verify_retained_policy_context",
        lambda _current, record, _state: {
            "campIdsInPolicyRowOrder": record["campIdsInPolicyRowOrder"],
            "rawPolicyBatchDigest": record["rawPolicyBatchDigest"],
            "packedPolicyInputDigest": record["packedPolicyInputDigest"],
        },
    )

    with pytest.raises(ValueError, match="observed step/policy/reward content differs"):
        replay_retained_snapshots(
            root,
            lambda: _FakeBridge(drift_turn=11),
            world_seed=50000,
            movement=False,
            withheld=0,
        )


@pytest.mark.parametrize("drift", ["unrelated-string", "nested-fingerprint"])
def test_branch_normalization_does_not_hide_unrelated_nested_drift(drift: str) -> None:
    recorded_bridge = _FakeBridge()
    recorded_bridge.reset(50000)
    recorded = recorded_bridge.step({11: {}})
    replayed_bridge = _FakeBridge()
    replayed_bridge.reset(50000)
    replayed_bridge.normalized = True
    replayed = replayed_bridge._step(10)
    recorded_row = recorded["economicDiagnostics"]["11"]
    replayed_row = replayed["economicDiagnostics"]["11"]
    if drift == "unrelated-string":
        recorded_row["note"] = "private-root-lineage-marker"
        replayed_row["note"] = "private-branch-lineage-marker"
    else:
        recorded_row["fingerprint"] = "nested-root-fingerprint"
        replayed_row["fingerprint"] = "nested-branch-fingerprint"

    with pytest.raises(ValueError, match="observed step/policy/reward content differs"):
        exact_replay._compare_step_identity(
            exact_replay._step_identity_bundle(recorded), replayed, drift
        )


def test_decoded_unit_neighborhood_uses_analytic_sigmoid_tail_bounds() -> None:
    mean = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    log_std = torch.tensor([0.0, -9.0, 0.0], dtype=torch.float64)
    center = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    probability = decoded_unit_neighborhood_probability(
        mean, log_std, center, radius=0.01, log_std_floor=-2.0
    )

    bound = torch.logit(torch.tensor(0.51, dtype=torch.float64))
    expected_middle = 2.0 * torch.special.ndtr(bound) - 1.0
    endpoint = torch.special.ndtr(torch.logit(torch.tensor(0.01, dtype=torch.float64)))
    assert probability[0] == pytest.approx(float(expected_middle))
    assert probability[1] > probability[0]
    assert probability[2] == pytest.approx(float(endpoint))
    assert torch.equal(
        probability,
        decoded_unit_neighborhood_probability(
            mean, log_std, center, radius=0.01, log_std_floor=-2.0
        ),
    )


def test_reference_binding_requires_rollout_batch_and_first_actor_hash_match(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference"
    current = tmp_path / "current-trace"
    reference.mkdir()
    current.mkdir()
    rollout = {
        "digest": "rollout",
        "initialObservationDigest": "obs",
        "actionDigest": "actions",
        "rewardDigest": "rewards",
        "rewardSum": 1.0,
        "decisions": 16,
    }
    validation = {
        "checkpoint": {"sha256": "c" * 64},
        "optimizerSeed": 4242,
        "worldSeed": 50000,
        "turns": 6,
        "parity": {"rollout": rollout},
    }
    (reference / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
    audit = {
        "transitionIds": ["all"],
        "stableTransitionIds": ["all"],
        "batchTransitionIds": ["batch"],
        "batchRowDigest": "batch-digest",
        "stableBatchTransitionIds": ["batch"],
        "stableBatchRowDigest": "stable-batch-digest",
        "auditTransitionIds": [f"audit-{index}" for index in range(16)],
        "stableAuditTransitionIds": [f"audit-{index}" for index in range(16)],
        "actorHashBefore": "before",
        "actorHashAfter": "after",
    }
    monkeypatch.setattr(smoke, "_first_actor_audit", lambda _path: copy.deepcopy(audit))
    retention = {
        "firstActorStep": {
            "traceActorHashBefore": "before",
            "traceActorHashAfter": "after",
            "weightsBefore": {"w": torch.tensor([0.0])},
            "weightsAfter": {"w": torch.tensor([1.0])},
            "adamBefore": {"state": {}, "param_groups": []},
            "adamAfter": {"state": {0: {}}, "param_groups": []},
            "rngBefore": {"torch": torch.tensor([1])},
            "rngAfter": {"torch": torch.tensor([1])},
        }
    }
    binding = smoke._reference_replay_binding(
        reference,
        current,
        {"rollout": copy.deepcopy(rollout)},
        retention,
        checkpoint_sha256="c" * 64,
        optimizer_seed=4242,
        world_seed=50000,
        turns=6,
    )
    assert binding["status"].startswith("observed-smoke-match")
    assert binding["fullWorldEquivalenceProven"] is False
    assert binding["optimizerDirectionReplayComplete"] is False
    assert binding["freshAdamProspective"]["historicalOptimizerRecovered"] is False
    assert binding["firstActorStep"]["auditTransitionIds"] == audit[
        "auditTransitionIds"
    ]

    bad = copy.deepcopy(rollout)
    bad["rewardDigest"] = "different"
    with pytest.raises(ValueError, match="rollout differs"):
        smoke._reference_replay_binding(
            reference,
            current,
            {"rollout": bad},
            retention,
            checkpoint_sha256="c" * 64,
            optimizer_seed=4242,
            world_seed=50000,
            turns=6,
        )


def test_retained_policy_rows_bind_exact_audit_transition_ids() -> None:
    retention = {
        "tracePostResetWorldSnapshotSha256": "s" * 64,
        "records": [
            {"preTurn": 9, "campIdsInPolicyRowOrder": [11, 12]},
            {"preTurn": 10, "campIdsInPolicyRowOrder": [11]},
        ]
    }
    transitions = [
        {"decisionTurn": 9, "campId": 11, "transitionId": "run/0/0/a",
         "provenance": {"resetSnapshotHash": "s" * 64}},
        {"decisionTurn": 9, "campId": 12, "transitionId": "run/0/0/b",
         "provenance": {"resetSnapshotHash": "s" * 64}},
        {"decisionTurn": 10, "campId": 11, "transitionId": "run/0/0/c",
         "provenance": {"resetSnapshotHash": "s" * 64}},
    ]
    audit = {
        "batchTransitionIds": ["run/0/0/c", "run/0/0/a"],
        "auditTransitionIds": ["run/0/0/c", "run/0/0/a"],
    }

    result = smoke._bind_retained_trace_rows(retention, transitions, audit)

    assert retention["records"][0]["transitionIdsInPolicyRowOrder"] == [
        "run/0/0/a",
        "run/0/0/b",
    ]
    assert retention["auditRowLocations"]["run/0/0/c"] == {
        "recordIndex": 1,
        "policyRowIndex": 0,
        "decisionTurn": 10,
        "campId": 11,
    }
    assert result == {
        "retainedPolicyRowCount": 3,
        "retainedTransitionIdDigest": smoke._digest(
            ["run/0/0/a", "run/0/0/b", "run/0/0/c"]
        ),
        "auditRowCount": 2,
        "allAuditRowsBoundToRetainedInputs": True,
        "tracePostResetWorldSnapshotSha256": "s" * 64,
        "postResetWorldCaptureMatchesTrace": True,
        "snapshotIsCompleteBridgeContext": False,
    }

    with pytest.raises(ValueError, match="absent"):
        smoke._bind_retained_trace_rows(
            {
                "tracePostResetWorldSnapshotSha256": "s" * 64,
                "records": copy.deepcopy(retention["records"]),
            },
            transitions,
            {
                "batchTransitionIds": ["missing"],
                "auditTransitionIds": ["missing"],
            },
        )
