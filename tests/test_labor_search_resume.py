"""Root-level (decision-boundary) resume of the labor search teacher, without the bridge.

A deterministic fake world (snapshot = the turn, a step digest that depends on the turn,
the executed payload and the restored transient state) drives the real ``search_world``
loop; ``search_root`` is a keyed stand-in that produces rows, tables, payload roots and
work. The world is killed (an exception standing for SIGKILL) at chosen points -- inside
a root search, after the root search before the main decision, right after a checkpoint,
twice in a row, with a torn newest checkpoint -- and rerun with ``resume``. The resumed
output must equal the uninterrupted run's exactly (rows bitwise, tables, payload roots,
skip reasons, search counts, work counters, main-path resolves, reset identity), and only
the in-flight root may be searched again. The bitwise gate on the real simulator is
tests/test_labor_search_resume_live.py.
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from anthropocene_rl.autonomous_v7 import bridge as bridge_module
from anthropocene_rl.autonomous_v7 import labor_bridge
from anthropocene_rl.autonomous_v7 import labor_model
from anthropocene_rl.autonomous_v7 import labor_search as ls
from anthropocene_rl.autonomous_v7 import labor_search_resume as lsr
from anthropocene_rl.autonomous_v7 import labor_teacher_data
from anthropocene_rl.autonomous_v7 import labor_training
from anthropocene_rl.autonomous_v7.labor_bc import LaborResetOptions

CAMPS = (3, 7, 11)


class Killed(RuntimeError):
    """Stands for a hard kill of the search process."""


class FakeWorld:
    def __init__(self) -> None:
        self.turn = 0
        self.transient = 0
        self.history: list[int] = []

    def step(self) -> dict:
        return {
            "turn": self.turn,
            "villages": [
                {"camp": c, "observation": {"t": self.turn}, "terminated": False} for c in CAMPS
            ],
            "nativeMs": 1.0,  # wall clock: ignored by the digest
            "h": self.history[-1] if self.history else 0,
        }


class FakeLaborBridge:
    world: FakeWorld

    def __init__(self, *_, **__) -> None:
        self.world = FakeWorld()
        self.last_reset_identity = {"fingerprint": "F", "configurationHash": "c"}
        self._bridge = SimpleNamespace(capture=self._capture)
        self.resets: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _capture(self) -> str:
        return base64.b64encode(json.dumps([self.world.turn, self.world.history]).encode()).decode()

    def capture_transient(self) -> dict:
        return {"transient": self.world.transient}

    def reset(self, world, *, snapshot=None, continuation=None, **kwargs):
        self.resets.append({"snapshot": snapshot is not None, "continuation": continuation})
        if snapshot is None:
            self.world = FakeWorld()
        else:
            turn, history = json.loads(base64.b64decode(snapshot))
            self.world = FakeWorld()
            self.world.turn, self.world.history = turn, list(history)
            # The live-only state comes back only through the continuation.
            self.world.transient = (continuation or {}).get("transient", {}).get("transient", -99)
        return self.world.step()

    def step_labor(self, initial, retry=None):
        mix = sum(int(p["a"]) * c for c, p in sorted(initial.items()))
        self.world.history.append(mix + 1000 * self.world.transient)
        self.world.turn += 1
        self.world.transient = self.world.turn % 3  # live-only state a snapshot lacks
        return self.world.step(), {}


class FakeHeuristic:
    def __init__(self, *_, **__) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install(monkeypatch, *, kill_root=None, kill_execute=None, calls=None):
    """Fakes for every native primitive search_world uses. ``kill_*`` raise Killed at
    the n-th root search / main execution (counted per process run)."""
    counters = {"root": 0, "execute": 0}
    calls = [] if calls is None else calls

    monkeypatch.setattr(labor_bridge, "LaborModel9Bridge", FakeLaborBridge)
    monkeypatch.setattr(bridge_module, "Model9Bridge", FakeHeuristic)
    monkeypatch.setattr(labor_teacher_data, "load_normalizer", lambda path: None)
    monkeypatch.setattr(
        labor_model,
        "labor_policy_tensors",
        lambda live, normalizer, personality=None: (
            None,
            {
                "core": torch.tensor(
                    [[float(v["observation"]["t"]), float(v["camp"])] for v in live]
                )
            },
        ),
    )
    monkeypatch.setattr(
        ls.NativeSearch, "catalogs", lambda self: {c: SimpleNamespace(population=10) for c in CAMPS}
    )

    def shadow(self, step, catalogs):
        self.work.shadow_steps += 1
        return {c: SimpleNamespace(action={"a": int(step["turn"]) + c}) for c in catalogs}

    monkeypatch.setattr(ls.NativeSearch, "shadow_h12", shadow)
    monkeypatch.setattr(
        ls.NativeSearch,
        "fork",
        lambda self: (self.work.__setattr__("forks", self.work.forks + 1), 1)[1],
    )
    monkeypatch.setattr(ls.NativeSearch, "restore", lambda self, fork: None)
    monkeypatch.setattr(ls.NativeSearch, "drop", lambda self, fork: None)
    monkeypatch.setattr(
        ls, "verify_exact_restore", lambda native, payloads: {"equal": True, "digest": "d"}
    )

    def execute(self, catalogs, payloads):
        counters["execute"] += 1
        if kill_execute is not None and counters["execute"] == kill_execute:
            raise Killed("killed before the main decision executed")
        self.work.labor_steps += 1
        step, _ = self.bridge.step_labor({c: dict(p) for c, p in payloads.items()})
        return SimpleNamespace(
            step=step,
            initial={c: SimpleNamespace(payload=dict(p)) for c, p in payloads.items()},
            retry={},
        )

    monkeypatch.setattr(ls.NativeSearch, "execute", execute)
    monkeypatch.setattr(
        labor_training,
        "_transition_rewards",
        lambda step, acted, founded, **kw: founded.setdefault(99, int(step["turn"])),
    )

    def search_root(native, *, step, focal, key, root_index, **kw):
        counters["root"] += 1
        calls.append((int(step["turn"]), focal))
        if kill_root is not None and counters["root"] == kill_root:
            raise Killed("killed inside a root search")
        native.work.labor_steps += 5
        native.work.validations += 2
        turn = int(step["turn"])
        value = torch.tensor([[turn * 0.1 + focal / 7, 1e-30]], dtype=torch.float32)
        if focal == 11 and turn % 2:
            return ls.RootResult([], {"skipped": "fewer-than-two-valid-candidates", "turn": turn})
        return ls.RootResult(
            [{"row_turn": torch.tensor([turn]), "x": value}],
            {
                "turn": turn,
                "focal": focal,
                "work": native.work.state_dict(),
                "wall_seconds": 0.0,
                "decisions": [{"reason": "search-beat-h12", "label": 1}],
            },
            {
                "record": {"root_id": f"{key[0]}/{turn}/{focal}/h12", "root_index": root_index},
                "tensors": {"core": value.clone()},
            },
        )

    monkeypatch.setattr(ls, "search_root", search_root)
    return counters


def _job(tmp_path: Path, *, resume: bool, roots: int = 4, turns: int = 9) -> dict:
    return {
        "world": 90100,
        "key": 0,
        "turns": turns,
        "first_turn": 0,
        "roots": roots,
        "source": "h12",
        "config": ls.SearchConfig(focal_per_root=2).state_dict(),
        "reset_options": LaborResetOptions().state_dict(),
        "normalizer_checkpoint": "n.pt",
        "actor_checkpoint": None,
        "personality_conditioning": None,
        "verify_restore": True,
        "source_run": "RUN",
        "out": str(tmp_path),
        "checkpoint_dir": str(tmp_path / "_work" / "world-90100"),
        "resume": resume,
        "checkpoint_keep": 2,
    }


def _same(a: dict, b: dict) -> None:
    assert len(a["rows"]) == len(b["rows"]) > 0
    for x, y in zip(a["rows"], b["rows"], strict=True):
        assert x.keys() == y.keys()
        for k in x:
            assert x[k].dtype == y[k].dtype and torch.equal(x[k], y[k])
            assert x[k].numpy().tobytes() == y[k].numpy().tobytes()
    for key in (
        "tables",
        "skipped",
        "reset_identity",
        "root_turn_indices",
        "restore_checks",
        "work",
        "main_path_resolves",
    ):
        assert a[key] == b[key], key
    assert len(a["payload_roots"]) == len(b["payload_roots"])
    for x, y in zip(a["payload_roots"], b["payload_roots"], strict=True):
        assert x["record"] == y["record"] and torch.equal(
            x["tensors"]["core"], y["tensors"]["core"]
        )


def test_replay_digest_ignores_process_local_branch_identity_only() -> None:
    recorded = {
        "turn": 4,
        "decisionId": "interrupted-process/37",
        "villages": [{"camp": 3, "observation": {"food": 1.25, "trajectoryId": "old"}}],
        "nativeMs": 3.0,
    }
    replayed = {
        "turn": 4,
        "decisionId": "resumed-process/1",
        "villages": [{"camp": 3, "observation": {"food": 1.25, "trajectoryId": "new"}}],
        "nativeMs": 99.0,
    }
    assert lsr.replay_step_digest(recorded) == lsr.replay_step_digest(replayed)
    replayed["villages"][0]["observation"]["food"] = 1.5
    assert lsr.replay_step_digest(recorded) != lsr.replay_step_digest(replayed)


def _reference(tmp_path, monkeypatch) -> dict:
    _install(monkeypatch)
    return ls.search_world(_job(tmp_path / "ref", resume=False))


@pytest.mark.parametrize(
    "kill",
    [
        {"kill_root": 1},  # the first root
        {"kill_root": 4},  # a later root, after several checkpoints
        {"kill_execute": 3},  # after a root search, before its main decision executed
        {"kill_execute": 7},  # late on the main path
    ],
)
def test_killed_world_resumes_to_the_uninterrupted_result(tmp_path, monkeypatch, kill) -> None:
    reference = _reference(tmp_path, monkeypatch)
    out = tmp_path / "run"
    calls_first: list = []
    _install(monkeypatch, calls=calls_first, **kill)
    with pytest.raises(Killed):
        ls.search_world(_job(out, resume=False))
    ckpts = sorted(p.name for p in (out / "_work" / "world-90100").glob("ckpt-*"))
    calls_resumed: list = []
    _install(monkeypatch, calls=calls_resumed)
    resumed = ls.search_world(_job(out, resume=True))
    _same(resumed, reference)
    # Only the in-flight decision's roots (focal_per_root=2) are searched again.
    redone = len(calls_first) + len(calls_resumed) - len(_reference_calls(tmp_path, monkeypatch))
    assert 0 <= redone <= 2
    if ckpts:
        assert resumed["checkpoints"]["sessions"] == 2
        assert calls_resumed[0][0] >= int(ckpts[-1][len("ckpt-") :]) - 1


def _reference_calls(tmp_path, monkeypatch) -> list:
    calls: list = []
    _install(monkeypatch, calls=calls)
    ls.search_world(_job(tmp_path / "ref-calls", resume=False))
    return calls


def test_twice_killed_and_torn_newest_checkpoint_still_resume_exactly(
    tmp_path, monkeypatch
) -> None:
    reference = _reference(tmp_path, monkeypatch)
    out = tmp_path / "run"
    _install(monkeypatch, kill_execute=5)
    with pytest.raises(Killed):
        ls.search_world(_job(out, resume=False))
    _install(monkeypatch, kill_root=2)
    with pytest.raises(Killed):
        ls.search_world(_job(out, resume=True))
    work = out / "_work" / "world-90100"
    newest = sorted(work.glob("ckpt-*"))[-1]
    (newest / "state.pt").write_bytes(b"torn")  # torn write: fall back to its predecessor
    _install(monkeypatch)
    resumed = ls.search_world(_job(out, resume=True))
    _same(resumed, reference)
    assert resumed["checkpoints"]["sessions"] == 3


def test_a_mismatching_replay_is_refused_and_the_world_is_redone(tmp_path, monkeypatch) -> None:
    reference = _reference(tmp_path, monkeypatch)
    out = tmp_path / "run"
    _install(monkeypatch, kill_execute=4)
    with pytest.raises(Killed):
        ls.search_world(_job(out, resume=False))
    # Corrupt the recorded transient: the replayed successor then differs.
    newest = sorted((out / "_work" / "world-90100").glob("ckpt-*"))[-1]
    state = torch.load(newest / "state.pt", weights_only=False)
    state["replay"]["transient"] = {"transient": 7}
    torch.save(state, newest / "state.pt")
    _install(monkeypatch)
    with pytest.raises(lsr.ContinuationMismatch):
        ls.search_world(_job(out, resume=True))
    # write_search_shard falls back to the reset (the uninterrupted run by definition).
    monkeypatch.setattr(ls, "_concat", lambda rows, context: (_ for _ in ()).throw(Killed("stop")))
    with pytest.raises(Killed):
        ls.write_search_shard(_job(out, resume=True))
    newest_after = sorted((out / "_work" / "world-90100").glob("ckpt-*"))[-1]
    assert torch.load(newest_after / "state.pt", weights_only=False)["sessions"] == 1
    redone = ls.search_world(_job(out, resume=False))
    _same(redone, reference)


def test_checkpoint_identity_is_enforced_and_keep_is_bounded(tmp_path, monkeypatch) -> None:
    out = tmp_path / "run"
    _install(monkeypatch, kill_execute=6)
    with pytest.raises(Killed):
        ls.search_world(_job(out, resume=False))
    work = out / "_work" / "world-90100"
    assert len(list(work.glob("ckpt-*"))) == 2
    job = dict(_job(out, resume=True), source_run="OTHER")
    _install(monkeypatch)
    with pytest.raises(lsr.CheckpointIdentityError, match="source_run"):
        ls.search_world(job)
    # Without resume an existing checkpoint is discarded and the world starts at its reset.
    fresh = ls.search_world(_job(out, resume=False))
    assert fresh["checkpoints"]["sessions"] == 1


def test_prepare_admits_the_work_dir_and_completion_drops_it(tmp_path) -> None:
    source = {"source_run": "RUN"}
    out = tmp_path / "out"
    ls.prepare_search_output(out, source, [90100], resume=False)
    (out / "_work" / "world-90100" / "ckpt-000003").mkdir(parents=True)
    assert ls.prepare_search_output(out, source, [90100], resume=True) == {}
    marker = ls.empty_marker(out, 90100)
    marker.write_text(json.dumps({"source_run": "RUN", "summary": {"world_seed": 90100}}))
    done = ls.prepare_search_output(out, source, [90100], resume=True)
    assert 90100 in done and not (out / "_work" / "world-90100").exists()
    shutil.rmtree(out)


def test_checkpointing_does_not_change_the_result(tmp_path, monkeypatch) -> None:
    reference = _reference(tmp_path, monkeypatch)
    _install(monkeypatch)
    plain = ls.search_world(dict(_job(tmp_path / "plain", resume=False), checkpoint_dir=None))
    _same(plain, reference)
    assert plain["checkpoints"] is None and reference["checkpoints"]["checkpoints"] > 0
