"""Kill-and-resume bitwise identity of teacher collection (labor_collect_resume).

A collector subprocess is killed hard (SIGKILL; TerminateProcess on Windows) at
chosen points -- right after a window checkpoint, at the window/tail boundary,
inside the tail, in the middle of a decision (after its shadow-heuristic resolve,
before the labor resolve), twice in a row, with a torn newest checkpoint and in
the second world of a two-world batch -- then rerun with ``--resume``. Every
resumed shard must equal the uninterrupted run's shard bitwise (every array's
dtype, shape and bytes, and ``data_digest``), and the run must have resumed from
a checkpoint (no fallback recollection).

Needs the model9 bridge built from this checkout (reset ``continuation``):
ANTHROPOCENE_MODEL9_BRIDGE. Small fixture: 4 players, 3-decision window and a
3-4 decision randomized tail (2 draws, rho 0.95, cap 8, seed base 9100567).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

BRIDGE = os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE")
REPOSITORY = Path(__file__).resolve().parents[1]
WORLDS = (90100, 90101)
ARGS = [
    "--turns",
    "3",
    "--players",
    "4",
    "--tail-draws",
    "2",
    "--tail-rho",
    "0.95",
    "--tail-cap",
    "8",
    "--tail-schedule-seed-base",
    "9100567",
    "--jobs",
    "1",
    "--allow-dirty",
]

pytestmark = pytest.mark.skipif(
    not BRIDGE or not Path(BRIDGE).is_file(), reason="native model9 bridge unavailable"
)

# Runs labor_teacher_data.main in-process (jobs 1) with an optional hard-kill hook.
RUNNER = r"""
import json, os, signal, sys
kill = json.loads(sys.argv[1])
from anthropocene_rl.autonomous_v7 import labor_bc, labor_collect_resume, labor_teacher_data

def die():
    sys.stderr.flush()
    os.kill(os.getpid(), getattr(signal, "SIGKILL", signal.SIGTERM))

if kill and "after_checkpoint" in kill:
    world, boundary = kill["after_checkpoint"]
    original = labor_collect_resume.WorldCheckpointer._write
    def write(self, result, *args, **kwargs):
        original(self, result, *args, **kwargs)
        if int(result.world_seed) == world and result.decisions + result.tail_decisions == boundary:
            die()
    labor_collect_resume.WorldCheckpointer._write = write
if kill and "after_shadow" in kill:
    world, call = kill["after_shadow"]
    original_shadow = labor_bc.shadow_teacher_labels
    calls = {"n": 0}
    def shadow(labor_bridge, heur_bridge, world_seed, *args, **kwargs):
        out = original_shadow(labor_bridge, heur_bridge, world_seed, *args, **kwargs)
        if int(world_seed) == world:
            calls["n"] += 1
            if calls["n"] == call:
                die()
        return out
    labor_bc.shadow_teacher_labels = shadow
raise SystemExit(labor_teacher_data.main(sys.argv[2:]))
"""


@pytest.fixture(scope="module")
def normalizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from anthropocene_rl.autonomous_v7.bridge import observation_batch
    from anthropocene_rl.autonomous_v7.labor_bridge import LaborModel9Bridge
    from anthropocene_rl.autonomous_v7.labor_model import BOUNDED_FEATURE_CONTRACT, Normalizer

    with LaborModel9Bridge() as bridge:
        if int(bridge.schema_response.get("resumeContinuation") or 0) < 3:
            pytest.skip("model9 bridge predates reset continuation; rebuild integration/")
        step = bridge.reset(50000)
        live = [
            v
            for v in step["villages"]
            if v.get("observation") is not None and not v.get("terminated")
        ]
        norm = Normalizer.fit_bounded(observation_batch(live))
    path = tmp_path_factory.mktemp("norm") / "normalizer.pt"
    torch.save(
        {"feature_contract": BOUNDED_FEATURE_CONTRACT, "normalizer": norm.state_dict()}, path
    )
    return path


def _run(out: Path, normalizer: Path, worlds=WORLDS[:1], *, kill=None, resume=False, extra=()):
    env = dict(os.environ, PYTHONPATH=str(REPOSITORY / "src"))
    argv = [
        sys.executable,
        "-c",
        RUNNER,
        json.dumps(kill),
        "--normalizer-checkpoint",
        str(normalizer),
        "--worlds",
        ",".join(str(w) for w in worlds),
        *ARGS,
        *extra,
        "--out",
        str(out),
    ]
    if resume:
        argv.append("--resume")
    return subprocess.run(
        argv, env=env, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL
    )


def _arrays(shard: Path) -> dict[str, tuple[str, tuple, bytes]]:
    out = {}
    for path in sorted(shard.glob("*.npy")):
        array = np.load(path, allow_pickle=False)
        out[path.name] = (array.dtype.str, array.shape, array.tobytes())
    return out


def _tail_record(meta: dict) -> dict:
    record = dict(meta.get("return_tail_record") or {})
    record.pop("continuation_seconds", None)
    return record


def _assert_identical(reference: Path, resumed: Path) -> None:
    ref_meta = json.loads((reference / "meta.json").read_text())
    meta = json.loads((resumed / "meta.json").read_text())
    assert meta["data_digest"] == ref_meta["data_digest"]
    assert meta["payload"] == ref_meta["payload"]
    assert _arrays(resumed) == _arrays(reference)
    for key in (
        "rows",
        "decisions",
        "native_resolves",
        "tail_decisions",
        "tail_native_resolves",
        "counters",
        "label_unavailable_reasons",
        "quarantine",
        "surviving_camps",
    ):
        assert meta.get(key) == ref_meta.get(key), key
    assert _tail_record(meta) == _tail_record(ref_meta)
    assert "collection_resume_fallback" not in meta


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory, normalizer: Path) -> Path:
    out = tmp_path_factory.mktemp("ref") / "ref"
    result = _run(out, normalizer, WORLDS)
    assert result.returncode == 0, result.stderr[-4000:]
    for world in WORLDS:
        meta = json.loads((out / f"world-{world}" / "meta.json").read_text())
        assert meta["decisions"] == 3 and meta["tail_decisions"] >= 2, meta
        assert meta["collection_checkpoints"]["sessions"] == 1
    assert not (out / "_work").exists()
    return out


def _killed_then_resumed(
    tmp_path: Path, normalizer: Path, kills, worlds=WORLDS[:1], before_resume=None
):
    out = tmp_path / "out"
    for kill in kills:
        killed = _run(out, normalizer, worlds, kill=kill, resume=out.exists())
        assert killed.returncode != 0, killed.stderr[-2000:]
        assert not (out / "dataset.json").exists()
    if before_resume is not None:
        before_resume(out)
    resumed = _run(out, normalizer, worlds, resume=True)
    assert resumed.returncode == 0, resumed.stderr[-4000:]
    assert "resumed_from_checkpoint" in resumed.stderr
    assert (out / "dataset.json").is_file() and not (out / "_work").exists()
    from anthropocene_rl.autonomous_v7.labor_teacher_data import verify_dataset

    verify_dataset(out)
    return out, resumed


@pytest.mark.parametrize(
    "kill",
    [
        {"after_checkpoint": [90100, 1]},  # inside the window
        {"after_checkpoint": [90100, 3]},  # window/tail boundary
        {"after_checkpoint": [90100, 5]},  # inside the tail
        {"after_shadow": [90100, 2]},  # mid-decision in the window (after the shadow resolve)
        {"after_shadow": [90100, 5]},  # mid-decision in the tail
    ],
    ids=["window-k1", "window-end-k3", "tail-k5", "window-mid-decision", "tail-mid-decision"],
)
def test_kill_and_resume_is_bitwise_identical(tmp_path, normalizer, reference, kill) -> None:
    out, _ = _killed_then_resumed(tmp_path, normalizer, [kill])
    _assert_identical(reference / "world-90100", out / "world-90100")
    meta = json.loads((out / "world-90100" / "meta.json").read_text())
    assert meta["collection_checkpoints"]["sessions"] == 2


def test_two_kills_then_resume(tmp_path, normalizer, reference) -> None:
    # Session 2 resumes at boundary 2 and is killed again inside decision 4.
    out, _ = _killed_then_resumed(
        tmp_path, normalizer, [{"after_checkpoint": [90100, 2]}, {"after_shadow": [90100, 2]}]
    )
    _assert_identical(reference / "world-90100", out / "world-90100")
    meta = json.loads((out / "world-90100" / "meta.json").read_text())
    assert meta["collection_checkpoints"]["sessions"] == 3


def test_torn_newest_checkpoint_falls_back_to_its_predecessor(
    tmp_path, normalizer, reference
) -> None:
    def tear(out: Path) -> None:
        work = out / "_work" / "world-90100"
        names = sorted(p.name for p in work.glob("ckpt-*"))
        assert names == ["ckpt-000004", "ckpt-000005"], names  # keep = 2
        (work / "ckpt-000005" / "snapshot.b64").write_bytes(b"torn")

    out, resumed = _killed_then_resumed(
        tmp_path, normalizer, [{"after_checkpoint": [90100, 5]}], before_resume=tear
    )
    assert "snapshot hash mismatch" in resumed.stderr
    assert '"resumed_from_checkpoint": 4' in resumed.stderr
    _assert_identical(reference / "world-90100", out / "world-90100")


def test_published_worlds_are_kept_and_the_rest_resume(tmp_path, normalizer, reference) -> None:
    published: dict[str, bytes] = {}

    def remember(out: Path) -> None:
        published["meta"] = (out / "world-90100" / "meta.json").read_bytes()

    out, _resumed = _killed_then_resumed(
        tmp_path,
        normalizer,
        [{"after_checkpoint": [90101, 4]}],
        worlds=WORLDS,
        before_resume=remember,
    )
    assert (out / "world-90100" / "meta.json").read_bytes() == published[
        "meta"
    ]  # skipped, untouched
    for world in WORLDS:
        _assert_identical(reference / f"world-{world}", out / f"world-{world}")
    manifest = json.loads((out / "dataset.json").read_text())
    reference_manifest = json.loads((reference / "dataset.json").read_text())
    assert {w: s["data_digest"] for w, s in manifest["shards"].items()} == {
        w: s["data_digest"] for w, s in reference_manifest["shards"].items()
    }


def test_resume_refuses_a_different_run_identity(tmp_path, normalizer, reference) -> None:
    out = tmp_path / "out"
    killed = _run(out, normalizer, kill={"after_checkpoint": [90100, 1]})
    assert killed.returncode != 0
    changed = _run(out, normalizer, resume=True, extra=["--tail-rho", "0.96"])
    assert changed.returncode != 0
    assert "run identity differs" in changed.stderr and "tail_rho" not in changed.stderr
    assert "options.return_tail.rho" in changed.stderr
    no_flag = _run(out, normalizer)
    assert no_flag.returncode != 0 and "--resume" in no_flag.stderr
    shutil.rmtree(out)
