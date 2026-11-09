"""Kill-and-resume bitwise identity of the labor search teacher (labor_search_resume).

A ``labor_search`` subprocess is killed hard (SIGKILL; TerminateProcess on Windows) right
after a decision-boundary checkpoint, inside a root search, twice in a row and with a torn
newest checkpoint, then rerun with ``--resume``. The resumed world's shard must equal the
uninterrupted run's: every array (dtype, shape, bytes), ``data_digest``, the candidate
table and the candidate payload (``advantage-candidates.pt``) up to wall-clock fields; and
the run must have resumed from a checkpoint (no fallback). Same pattern as
tests/test_labor_collect_resume_live.py.

Needs the model9 bridge with reset ``continuation`` (resumeContinuation >= 3):
ANTHROPOCENE_MODEL9_BRIDGE. Small fixture: 4 players, roots at 3 of 5 decisions, M=6.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

BRIDGE = os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE")
REPOSITORY = Path(__file__).resolve().parents[1]
WORLD = 90100
ARGS = [
    "--worlds",
    str(WORLD),
    "--source",
    "h12",
    "--turns",
    "5",
    "--roots-per-world",
    "3",
    "--candidates",
    "6",
    "--horizons",
    "1,2",
    "--keep",
    "all,2",
    "--reeval-streams",
    "2",
    "--personalities",
    "balanced",
    "--control-fraction",
    "0.5",
    "--plan-candidates",
    "1",
    "--plan-steps",
    "2",
    "--plan-fraction",
    "0.5",
    "--players",
    "4",
    "--jobs",
    "1",
    "--allow-dirty",
    # Extra search flags for a variant selftest, e.g. "--successor-rows" (export v2).
    *os.environ.get("ANTHROPOCENE_SEARCH_SELFTEST_EXTRA", "").split(),
]

pytestmark = pytest.mark.skipif(
    not BRIDGE or not Path(BRIDGE).is_file(), reason="native model9 bridge unavailable"
)

RUNNER = r"""
import json, os, signal, sys
kill = json.loads(sys.argv[1])
from anthropocene_rl.autonomous_v7 import labor_search, labor_search_resume

def die():
    sys.stderr.flush()
    os.kill(os.getpid(), getattr(signal, "SIGKILL", signal.SIGTERM))

if kill and "after_checkpoint" in kill:
    boundary = int(kill["after_checkpoint"])
    original = labor_search_resume.SearchCheckpointer.write
    def write(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if int(kwargs["boundary"]) == boundary:
            die()
    labor_search_resume.SearchCheckpointer.write = write
if kill and "in_root" in kill:
    call = int(kill["in_root"])
    original_root = labor_search.search_root
    calls = {"n": 0}
    def search_root(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == call:
            die()
        return original_root(*args, **kwargs)
    labor_search.search_root = search_root
raise SystemExit(labor_search.main(sys.argv[2:]))
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


def _run(out: Path, normalizer: Path, *, kill=None, resume=False):
    env = dict(os.environ, PYTHONPATH=str(REPOSITORY / "src"))
    argv = [
        sys.executable,
        "-c",
        RUNNER,
        json.dumps(kill),
        "--normalizer-checkpoint",
        str(normalizer),
        *ARGS,
        "--out",
        str(out),
    ]
    if resume:
        argv.append("--resume")
    return subprocess.run(
        argv, env=env, capture_output=True, text=True, timeout=3600, stdin=subprocess.DEVNULL
    )


def _strip(value):
    """Drop wall-clock fields (anything named *seconds* or *Ms)."""
    if isinstance(value, dict):
        return {
            k: _strip(v)
            for k, v in value.items()
            if "seconds" not in str(k) and not str(k).endswith("Ms")
        }
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _equal(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=False))
    return a == b


def _tables(directory: Path) -> list:
    lines = (directory / "candidates.jsonl").read_text().splitlines()
    return [_strip(json.loads(line)) for line in lines if line]


def _payload(directory: Path):
    return _strip(torch.load(directory / "advantage-candidates.pt", weights_only=False))


def _assert_identical(reference: Path, resumed: Path) -> None:
    ref_meta = json.loads((reference / "meta.json").read_text())
    meta = json.loads((resumed / "meta.json").read_text())
    assert meta["data_digest"] == ref_meta["data_digest"]
    for path in sorted(reference.glob("*.npy")):
        x, y = np.load(path), np.load(resumed / path.name)
        assert (x.dtype.str, x.shape, x.tobytes()) == (y.dtype.str, y.shape, y.tobytes()), path.name
    for key in (
        "rows",
        "counters",
        "skipped",
        "work",
        "main_path_resolves",
        "root_turn_indices",
        "reset_identity",
        "restore_checks",
    ):
        assert meta.get(key) == ref_meta.get(key), key
    assert _tables(resumed) == _tables(reference)
    assert _equal(_payload(resumed), _payload(reference))
    assert meta.get("search_resume_fallback") is None
    assert meta["search_checkpoints"]["sessions"] >= 2


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory, normalizer: Path) -> Path:
    out = tmp_path_factory.mktemp("ref") / "ref"
    result = _run(out, normalizer)
    assert result.returncode == 0, result.stderr[-4000:]
    meta = json.loads((out / f"world-{WORLD}" / "meta.json").read_text())
    assert meta["search_checkpoints"]["sessions"] == 1 and meta["rows"] > 0
    assert not (out / "_work").exists()
    return out


def _refusals(stderr: str) -> list[str]:
    """Every refused checkpoint with its full reason (never truncated in the failure)."""
    out = []
    for line in stderr.splitlines():
        if '"resume_refused"' in line or "CheckpointIdentityError" in line:
            try:
                record = json.loads(line)
                out.append(f"world {record.get('world')}: {record.get('resume_refused')}")
            except ValueError:
                out.append(line)
    return out


def _killed_then_resumed(tmp_path: Path, normalizer: Path, kills, before_resume=None) -> Path:
    out = tmp_path / "out"
    for kill in kills:
        killed = _run(out, normalizer, kill=kill, resume=out.exists())
        assert killed.returncode != 0, killed.stderr[-2000:]
        assert not (out / "dataset.json").exists()
    if before_resume is not None:
        before_resume(out)
    resumed = _run(out, normalizer, resume=True)
    refused = _refusals(resumed.stderr)
    assert resumed.returncode == 0, "\n".join([*refused, resumed.stderr[-4000:]])
    assert not refused, "resume refused:\n" + "\n".join(refused)
    assert "search_resumed_from_checkpoint" in resumed.stderr, resumed.stderr[-4000:]
    assert (out / "dataset.json").is_file() and not (out / "_work").exists()
    return out


@pytest.mark.parametrize(
    "kill",
    [{"after_checkpoint": 1}, {"after_checkpoint": 3}, {"in_root": 2}, {"in_root": 3}],
    ids=["ckpt-1", "ckpt-3", "mid-root-2", "mid-root-3"],
)
def test_kill_and_resume_is_bitwise_identical(tmp_path, normalizer, reference, kill) -> None:
    out = _killed_then_resumed(tmp_path, normalizer, [kill])
    _assert_identical(reference / f"world-{WORLD}", out / f"world-{WORLD}")


def test_two_kills_and_a_torn_checkpoint(tmp_path, normalizer, reference) -> None:
    def tear(out: Path) -> None:
        newest = sorted((out / "_work" / f"world-{WORLD}").glob("ckpt-*"))[-1]
        (newest / "state.pt").write_bytes(b"torn")

    out = _killed_then_resumed(
        tmp_path, normalizer, [{"after_checkpoint": 2}, {"in_root": 2}], before_resume=tear
    )
    _assert_identical(reference / f"world-{WORLD}", out / f"world-{WORLD}")
