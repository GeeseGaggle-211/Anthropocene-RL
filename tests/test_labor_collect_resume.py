"""Checkpoint mechanics of teacher collection (labor_collect_resume), without the bridge.

The bitwise kill-and-resume gate is tests/test_labor_collect_resume_live.py.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.labor_bc import TeacherTailState, WorldCollection
from anthropocene_rl.autonomous_v7.labor_collect_resume import (
    CheckpointIdentityError,
    ContinuationMismatch,
    WorldCheckpointer,
    load_resume_point,
    step_digest,
)

IDENTITY = {"source_run": "s", "world_seed": 90100}


def _transition(turn: int, snapshot: str = "c25hcA=="):
    action = SimpleNamespace(payload={"schemaVersion": 12, "laborSpecialtyAllocation": [0.1, 0.9]})
    return SimpleNamespace(
        snapshot=snapshot,
        transient={"schema": 1, "cultivationOperatingFraction": [[5, 4604930618986332160]]},
        next_step={"turn": turn + 1, "villages": [], "nativeMs": 3.5},
        replay_payloads=lambda: {"initial": {"7": dict(action.payload)}, "retry": {}},
    )


def _result() -> WorldCollection:
    result = WorldCollection(world_seed=90100)
    result.reset_identity = {"fingerprint": "ROOT", "configurationHash": "c"}
    return result


def _row(turn: int) -> dict[str, torch.Tensor]:
    return {"row_turn": torch.tensor([turn]), "x": torch.tensor([[0.25, 1e-30]], dtype=torch.float32)}


def _window(checkpointer, result, decisions):
    for done in range(1, decisions + 1):
        result.rows.append(_row(done))
        result.decisions += 1
        result.counters["total_decisions"] += 1
        checkpointer.window(result, _transition(10 + done), {7: 10}, done)


def test_step_digest_ignores_only_top_level_timings() -> None:
    a = {"turn": 3, "nativeMs": 1.0, "villages": [{"x": 1.5}]}
    assert step_digest(a) == step_digest(dict(a, nativeMs=99.0))
    assert step_digest(a) != step_digest(dict(a, villages=[{"x": 1.25}]))


def test_tail_state_round_trips_with_order() -> None:
    state = TeacherTailState(
        final_turn=12, tracked=[9, 3], active=[3], reasons={9: "b", 3: "a"},
        path_rewards={9: {"qol_years": [0.1, 1 / 3]}, 3: {"qol_years": [2.0]}},
        path_years={9: [20.0], 3: [20.0]}, path_terminal={9: [True], 3: [False]}, done=1, seconds=2.5,
    )
    restored = TeacherTailState.from_state(json.loads(json.dumps(state.state_dict())))
    assert restored == state
    assert list(restored.reasons) == [9, 3] and list(restored.path_rewards) == [9, 3]


def test_checkpoints_are_atomic_keep_the_last_two_and_append_rows(tmp_path: Path) -> None:
    work = tmp_path / "world"
    result = _result()
    checkpointer = WorldCheckpointer(work, IDENTITY, keep=2)
    checkpointer.begin(result, {"turn": 10, "villages": [{"camp": 7}]})
    _window(checkpointer, result, 3)
    assert sorted(p.name for p in work.glob("ckpt-*")) == ["ckpt-000002", "ckpt-000003"]
    assert sorted(p.name for p in (work / "rows").iterdir()) == [
        "part-000001.pt", "part-000002.pt", "part-000003.pt"
    ]
    point = load_resume_point(work, IDENTITY, {"players": 4})
    assert point.boundary == 3 and point.window_done == 3 and point.founded == {7: 10}
    assert [int(r["row_turn"]) for r in point.rows] == [1, 2, 3]
    assert torch.equal(point.rows[0]["x"], _row(1)["x"])
    state = point.state
    assert state["continuation"] == {"startTurn": 10, "trajectoryId": "ROOT"}
    assert state["replay"]["payloads"]["initial"]["7"]["laborSpecialtyAllocation"] == [0.1, 0.9]
    assert state["replay"]["step_digest"] == step_digest({"turn": 14, "villages": []})
    # A tail checkpoint carries the tail loop state; rows are not rewritten.
    tail = TeacherTailState(13, [7], [7], {}, {7: {"qol_years": [1.0]}}, {7: [20.0]}, {7: [False]}, 1)
    result.tail_decisions = 1
    WorldCheckpointer(work, IDENTITY, resume=point).tail(result, _transition(13), tail)
    newest = load_resume_point(work, IDENTITY, {})
    assert newest.boundary == 4 and newest.tail_state == tail and newest.state["sessions"] == 2


def test_resume_refuses_another_identity_and_falls_back_from_a_torn_checkpoint(tmp_path: Path) -> None:
    work = tmp_path / "world"
    result = _result()
    checkpointer = WorldCheckpointer(work, IDENTITY)
    checkpointer.begin(result, {"turn": 10, "villages": [{"camp": 7}]})
    _window(checkpointer, result, 2)
    with pytest.raises(CheckpointIdentityError, match="source_run"):
        load_resume_point(work, dict(IDENTITY, source_run="other"), {})
    (work / "ckpt-000002" / "snapshot.b64").write_bytes(b"torn")
    warnings = []
    point = load_resume_point(work, IDENTITY, {}, log=warnings.append)
    assert point.boundary == 1 and "snapshot hash mismatch" in warnings[0]["checkpoint_warning"]
    # Resuming from boundary 1 drops the orphaned later row part.
    WorldCheckpointer(work, IDENTITY, resume=point)
    assert sorted(p.name for p in (work / "rows").iterdir()) == ["part-000001.pt"]
    (work / "rows" / "part-000001.pt").write_bytes(b"x")
    assert load_resume_point(work, IDENTITY, {}, log=warnings.append) is None


def test_restore_verifies_the_replayed_boundary(tmp_path: Path) -> None:
    work = tmp_path / "world"
    result = _result()
    checkpointer = WorldCheckpointer(work, IDENTITY)
    checkpointer.begin(result, {"turn": 10, "villages": [{"camp": 7}]})
    _window(checkpointer, result, 1)
    point = load_resume_point(work, IDENTITY, {"players": 4, "movement": True})
    calls = []

    class Bridge:
        def __init__(self, successor):
            self.successor = successor

        def reset(self, seed, **kwargs):
            calls.append(("reset", seed, kwargs))

        def step_labor(self, initial, retry):
            calls.append(("step", initial, retry))
            return self.successor, {}

    restored = _result()
    step = point.restore(Bridge({"turn": 12, "villages": [], "nativeMs": 9.0}), restored)
    assert step["turn"] == 12 and restored.decisions == 1 and restored.counters["total_decisions"] == 1
    assert calls[0] == ("reset", 90100, {"snapshot": base64.b64encode(b"snap").decode(),
                                          "continuation": {"startTurn": 10, "trajectoryId": "ROOT", "transient": {
                                              "schema": 1, "cultivationOperatingFraction": [[5, 4604930618986332160]]}},
                                          "players": 4, "movement": True})
    assert calls[1][1] == {7: {"schemaVersion": 12, "laborSpecialtyAllocation": [0.1, 0.9]}}
    assert calls[1][2] is None
    with pytest.raises(ContinuationMismatch):
        point.restore(Bridge({"turn": 12, "villages": [{"camp": 1}]}), _result())


def test_resume_preflight_names_the_differing_identity_fields(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    source = {"source_run": "a", "options": {"turns": 50, "return_tail": {"rho": 0.95}}}
    ltd._write_run_record(out, source)
    changed = {"source_run": "b", "options": {"turns": 50, "return_tail": {"rho": 0.9}}}
    with pytest.raises(ltd.ProvenanceError, match=r"options\.return_tail\.rho"):
        ltd._resume_preflight(out, changed, [90100])
    (out / "stray").mkdir()
    with pytest.raises(ltd.ProvenanceError, match="undeclared"):
        ltd._resume_preflight(out, source, [90100])
    (out / "stray").rmdir()
    (out / "world-90100").mkdir()  # unpublished shard directory
    assert ltd._resume_preflight(out, source, [90100]) is None
    assert ltd._published_shards(out, source, [90100]) == {}
    assert not (out / "world-90100").exists()
    with pytest.raises(ltd.ProvenanceError, match="no collection.json"):
        ltd._resume_preflight(tmp_path, source, [90100])
