import hashlib
import json
from pathlib import Path

import pytest

from anthropocene_rl.autonomous_v7.gate34_checkpoint_collector import (
    BranchResult,
    paired_path_diagnostic,
)
from anthropocene_rl.autonomous_v7.gate34_continuation_collector import (
    FrozenGeometricSchedule,
    RandomizedTailEstimate,
)
from anthropocene_rl.autonomous_v7.gate34_single_state_debug import (
    EXPECTED_RESET_TURN,
    WORLD_SEED,
    _trace_reset_binding,
    _worker_command,
)
from anthropocene_rl.autonomous_v7.provenance import ProvenanceError


def _tail(
    schedule: FrozenGeometricSchedule,
    draws: tuple[float, ...],
) -> RandomizedTailEstimate:
    import numpy as np

    values = np.asarray(draws, dtype=np.float64)
    return RandomizedTailEstimate(
        status="identified",
        schedule_id=schedule.schedule_id,
        estimate=float(values.mean()),
        standard_error=float(values.std(ddof=1) / np.sqrt(len(values))),
        draw_estimates=draws,
        standard_error_scope="synthetic paired stopping randomization",
        stopping_randomization_contribution_to_standard_error=float(
            values.std(ddof=1) / np.sqrt(len(values))
        ),
    )


def _branch(
    alternative: str,
    path: int,
    schedule: FrozenGeometricSchedule,
    *,
    prefix: float,
    prediction: float,
    tail: RandomizedTailEstimate,
    terminal: bool = False,
) -> BranchResult:
    return BranchResult(
        alternative_id=alternative,
        path_replicate=path,
        schedule=schedule,
        prefix_returns={"welfare": prefix},
        tail_estimates={"welfare": tail},
        camp_prefix_returns={"welfare": {1: prefix}},
        camp_tail_estimates={"welfare": {1: tail}},
        ranking_predictions={"welfare": prediction},
        action_receipts=(),
        collected_steps=2,
        opening_camp_count=1,
        terminal=terminal,
        economic=None,
    )


def test_paired_path_diagnostic_preserves_paired_draws_ties_and_uncertainty() -> None:
    schedule = FrozenGeometricSchedule("paired", 0.95, (0, 0), 10)
    baseline = _branch(
        "sampled-policy",
        0,
        schedule,
        prefix=4.0,
        prediction=5.0,
        tail=_tail(schedule, (0.0, 0.0)),
    )
    positive = _branch(
        "development-budget-native-v1",
        0,
        schedule,
        prefix=6.0,
        prediction=7.0,
        tail=_tail(schedule, (2.0, 2.0)),
    )
    result = paired_path_diagnostic(positive, baseline, "welfare")
    assert result["predictedDifference"] == 2.0
    assert result["realisedFullReturnDifference"] == 4.0
    assert result["realisedRelation"] == "positive"
    assert result["predictionTargetRelationAgreement"] is True
    assert "not an independent world" in result["nestedUnit"]

    tied = _branch(
        "development-budget-native-v1",
        0,
        schedule,
        prefix=4.0,
        prediction=5.0,
        tail=_tail(schedule, (0.0, 0.0)),
    )
    tie_result = paired_path_diagnostic(tied, baseline, "welfare")
    assert tie_result["predictedRelation"] == "predicted-tie"
    assert tie_result["realisedRelation"] == "estimated-within-tolerance"
    assert tie_result["predictionTargetRelationAgreement"] is None

    terminal_baseline = _branch(
        "sampled-policy",
        0,
        schedule,
        prefix=4.0,
        prediction=5.0,
        tail=_tail(schedule, (0.0, 0.0)),
        terminal=True,
    )
    terminal_tie = _branch(
        "development-budget-native-v1",
        0,
        schedule,
        prefix=4.0,
        prediction=5.0,
        tail=_tail(schedule, (0.0, 0.0)),
        terminal=True,
    )
    exact = paired_path_diagnostic(terminal_tie, terminal_baseline, "welfare")
    assert exact["realisedRelation"] == "exact-terminal-tie"
    assert exact["predictionTargetRelationAgreement"] is True

    uncertain = _branch(
        "development-budget-native-v1",
        0,
        schedule,
        prefix=4.0,
        prediction=7.0,
        tail=_tail(schedule, (-1.0, 1.0)),
    )
    uncertain_result = paired_path_diagnostic(uncertain, baseline, "welfare")
    assert uncertain_result["realisedRelation"] == "uncertain"
    assert uncertain_result["predictionTargetRelationAgreement"] is None


def test_paired_path_diagnostic_withholds_unresolved_tail() -> None:
    schedule = FrozenGeometricSchedule("trade", 0.95, (0, 0), 10)
    unresolved = RandomizedTailEstimate(
        status="unresolved-absolute-integrability", schedule_id=schedule.schedule_id
    )
    baseline = _branch(
        "sampled-policy",
        0,
        schedule,
        prefix=1.0,
        prediction=1.0,
        tail=unresolved,
    )
    intervention = _branch(
        "development-budget-native-v1",
        0,
        schedule,
        prefix=2.0,
        prediction=3.0,
        tail=unresolved,
    )
    result = paired_path_diagnostic(intervention, baseline, "welfare")
    assert result["realisedFullReturnDifference"] is None
    assert result["realisedDifferenceInterval"] is None
    assert result["realisedRelation"] == "unresolved"
    assert result["predictionTargetRelationAgreement"] is None


def _write_trace_fixture(root: Path, *, pre_fingerprint: str = "reset-fingerprint") -> None:
    trace = root / "trace"
    snapshots = trace / "snapshots"
    normalizers = trace / "normalizers"
    snapshots.mkdir(parents=True)
    normalizers.mkdir()
    snapshot = b"snapshot"
    snapshot_hash = hashlib.sha256(snapshot).hexdigest()
    (snapshots / f"{snapshot_hash}.anthro").write_bytes(snapshot)
    normalizer = b"{}\n"
    normalizer_hash = hashlib.sha256(normalizer).hexdigest()
    (normalizers / f"{normalizer_hash}.json").write_bytes(normalizer)
    (root / "validation.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "worldSeed": WORLD_SEED,
                "checkpoint": {"sha256": "c" * 64},
            }
        ),
        encoding="utf-8",
    )
    (trace / "manifest.json").write_text(
        json.dumps({"nonqualifying": True}), encoding="utf-8"
    )
    identity = {
        "configurationHash": "configuration",
        "fingerprint": "reset-fingerprint",
        "buildHash": "build",
        "rawMaterialsEnabled": False,
        "sourceSeed": WORLD_SEED,
        "preFingerprint": pre_fingerprint,
        "bridgeSha256": "b" * 64,
        "resetSnapshotHash": snapshot_hash,
    }
    rows = [
        {
            "decisionTurn": EXPECTED_RESET_TURN,
            "worldSeed": WORLD_SEED,
            "policyVersion": 0,
            "normalizerMomentsRef": normalizer_hash,
            "trajectoryId": "trajectory-is-not-the-reset-fingerprint",
            "provenance": identity,
        },
        {
            "decisionTurn": EXPECTED_RESET_TURN + 1,
            "worldSeed": WORLD_SEED,
            "policyVersion": 0,
            "normalizerMomentsRef": normalizer_hash,
            "trajectoryId": "later",
            "provenance": identity,
        },
    ]
    (trace / "transitions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_trace_binding_uses_native_reset_fingerprint_not_trajectory(tmp_path: Path) -> None:
    _write_trace_fixture(tmp_path)
    result = _trace_reset_binding(tmp_path, "c" * 64, "b" * 64)
    assert result["firstDecisionRowCount"] == 1
    assert result["nativeResetIdentity"]["fingerprint"] == "reset-fingerprint"
    assert result["trajectoryIdsObservedButNotUsedAsFingerprint"] == [
        "trajectory-is-not-the-reset-fingerprint"
    ]


def test_trace_binding_rejects_reset_pre_fingerprint_mismatch(tmp_path: Path) -> None:
    _write_trace_fixture(tmp_path, pre_fingerprint="trajectory-not-reset")
    with pytest.raises(ProvenanceError, match="reset fingerprint and pre-fingerprint"):
        _trace_reset_binding(tmp_path, "c" * 64, "b" * 64)


def test_worker_command_uses_canonical_importable_module(tmp_path: Path) -> None:
    command = _worker_command(tmp_path / "plan.json")
    assert command[1:3] == [
        "-m",
        "anthropocene_rl.autonomous_v7.gate34_single_state_debug",
    ]
    assert "__main__" not in command
