"""Decision-boundary checkpoints and exact resume for the labor search teacher.

``labor_search`` drives one world's main (H12 or actor) trajectory and searches its
keyed roots on the way. Before this module a preempted world restarted from its reset,
losing every root already searched. Now every world is checkpointed at each main-path
decision boundary, after the decision (and the roots searched at it) completed; a
``--resume`` run continues from the newest valid checkpoint, so a stop loses at most
the in-flight decision and its root. The method is the collector's
(``labor_collect_resume``), applied to the search main path:

* ``snapshot.b64`` -- the simulator snapshot of boundary ``b - 1``, captured right
  before decision ``b - 1`` executed (after the root search restored its root fork),
  plus the live-only state a snapshot lacks (``capture_transient``), restored by the
  resume reset.
* ``state.pt`` -- the exact executed payloads of decision ``b - 1`` (initial and
  retry), the digest of its successor step, the episode continuation identity, and
  every loop field that affects later output: founded turns, per-camp search counts
  (focal order), skip reasons, the exact-restore check, main-path resolves, the work
  counters and the original reset identity.
* ``parts/part-<b>.pt`` -- the label rows, candidate tables and payload roots produced
  by decision ``b - 1`` (appended; never rewritten).

Resume resets from the boundary ``b - 1`` snapshot with the interrupted episode's start
turn and trajectory id, replays decision ``b - 1`` (one native resolve, not counted in
the work counters, which are restored) and requires the successor step to match the
recorded digest (``ContinuationMismatch`` otherwise; the caller then redoes the world
from its reset, which is the uninterrupted run by definition). Search randomness is
keyed by (world, key, turn, focal, ...), the teacher and simulator are deterministic,
so the resumed world's rows, tables and payloads equal the uninterrupted run's; only
wall-clock fields differ.

Writes are atomic (``ckpt-<b>.tmp`` then rename); the last ``keep`` checkpoints are kept.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .labor_collect_resume import (
    CheckpointIdentityError,
    ContinuationMismatch,
    _canonical_payloads,
    _fsync_dir,
)

SEARCH_CHECKPOINT_SCHEMA = 1
WORK_DIR = "_work"
PREFIX = "ckpt-"

__all__ = [
    "WORK_DIR",
    "CheckpointIdentityError",
    "ContinuationMismatch",
    "SearchCheckpointer",
    "SearchResumePoint",
    "load_search_resume_point",
    "search_work_dir",
]


def search_work_dir(out: Path, world: int) -> Path:
    return Path(out) / WORK_DIR / f"world-{int(world)}"


def _bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replay_step_digest(step: Mapping[str, Any]) -> str:
    """Digest replayed search state without process-local branch identities.

    Search explores native forks before executing the main-path decision.  A fresh
    bridge process recreates the same physical successor on resume, but its
    ``decisionId`` values are process-local and therefore need not equal the ids from
    the interrupted process.  This is the same physical-state contract used by the
    search fork/restore verifier.
    """
    from .labor_search import physical_digest

    return physical_digest(step)


def _write(path: Path, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _boundary(path: Path) -> int | None:
    name = path.name
    if not path.is_dir() or not name.startswith(PREFIX) or name.endswith(".tmp"):
        return None
    try:
        return int(name[len(PREFIX) :])
    except ValueError:
        return None


@dataclass
class SearchResumePoint:
    """A validated checkpoint: the loop state and the parts produced so far."""

    directory: Path
    state: dict[str, Any]
    snapshot: str
    rows: list[Any]
    tables: list[Any]
    payload_roots: list[Any]

    @property
    def boundary(self) -> int:
        return int(self.state["boundary"])

    def restore(
        self, bridge: Any, world: int, reset_kwargs: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Reset from the boundary ``b - 1`` snapshot, replay decision ``b - 1``, verify it."""
        replay = self.state["replay"]
        continuation = dict(self.state["continuation"])
        if replay.get("transient") is not None:
            continuation["transient"] = replay["transient"]
        bridge.reset(int(world), snapshot=self.snapshot, continuation=continuation, **reset_kwargs)
        initial = {int(c): dict(p) for c, p in replay["payloads"]["initial"].items()}
        retry = {int(c): dict(p) for c, p in replay["payloads"]["retry"].items()}
        step, _ = bridge.step_labor(initial, retry or None)
        actual = replay_step_digest(step)
        if actual != replay["step_digest"]:
            raise ContinuationMismatch(
                f"search world {world} checkpoint {self.boundary}: replayed boundary digest "
                f"{actual} != recorded {replay['step_digest']}"
            )
        return step

    def loop_state(self) -> dict[str, Any]:
        s = self.state
        return {
            "founded": {int(c): int(t) for c, t in s["founded"]},
            "skipped": Counter({str(k): int(v) for k, v in s["skipped"]}),
            "searched": defaultdict(int, {int(c): int(n) for c, n in s["searched"]}),
            "restore_checks": list(s["restore_checks"]),
            "main_resolves": int(s["main_resolves"]),
            "work": dict(s["work"]),
            "reset_identity": dict(s["reset_identity"]),
            "continuation": dict(s["continuation"]),
        }


def load_search_resume_point(
    directory: Path, identity: Mapping[str, Any], *, log: Any = None
) -> SearchResumePoint | None:
    """The newest valid checkpoint of one search world, or None.

    Another run's checkpoint is refused (``CheckpointIdentityError``); a torn or corrupt
    one is reported and its predecessor tried."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = sorted(
        ((b, p) for p in directory.iterdir() if (b := _boundary(p)) is not None), reverse=True
    )
    for boundary, path in candidates:
        try:
            state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        except Exception as error:  # torn write
            _warn(log, directory, boundary, f"unreadable state: {error!r}")
            continue
        if state.get("schema") != SEARCH_CHECKPOINT_SCHEMA:
            raise CheckpointIdentityError(
                f"{path}: search checkpoint schema {state.get('schema')!r} != "
                f"{SEARCH_CHECKPOINT_SCHEMA}"
            )
        if state.get("identity") != dict(identity):
            saved = state.get("identity") or {}
            differing = sorted(
                k for k in set(saved) | set(identity) if saved.get(k) != identity.get(k)
            )
            raise CheckpointIdentityError(
                f"{path}: checkpoint belongs to another search run; differing fields {differing}"
            )
        if int(state.get("boundary", -1)) != boundary:
            _warn(log, directory, boundary, "boundary field differs from the directory name")
            continue
        try:
            snapshot = (path / "snapshot.b64").read_bytes()
        except OSError as error:
            _warn(log, directory, boundary, f"unreadable snapshot: {error}")
            continue
        if _sha(snapshot) != state["replay"]["snapshot_sha256"]:
            _warn(log, directory, boundary, "snapshot hash mismatch")
            continue
        rows: list[Any] = []
        tables: list[Any] = []
        roots: list[Any] = []
        valid = True
        for part in state["parts"]:
            part_path = directory / "parts" / part["name"]
            try:
                data = part_path.read_bytes()
            except OSError:
                data = b""
            if _sha(data) != part["sha256"]:
                _warn(log, directory, boundary, f"part {part['name']} missing or corrupt")
                valid = False
                break
            loaded = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
            rows.extend(loaded["rows"])
            tables.extend(loaded["tables"])
            roots.extend(loaded["payload_roots"])
        if not valid:
            continue
        return SearchResumePoint(path, state, snapshot.decode("ascii"), rows, tables, roots)
    return None


def _warn(log: Any, directory: Path, boundary: int, message: str) -> None:
    if log is not None:
        log({"search_checkpoint": str(directory), "boundary": boundary, "skipped": message})


class SearchCheckpointer:
    """Writes one search world's decision-boundary checkpoints."""

    def __init__(
        self,
        directory: Path,
        identity: Mapping[str, Any],
        *,
        keep: int = 2,
        resume: SearchResumePoint | None = None,
        log: Any = None,
    ) -> None:
        if keep < 1:
            raise ValueError("checkpoint keep must be positive")
        self.directory = Path(directory)
        self.identity = dict(identity)
        self.keep = int(keep)
        self.log = log
        self.session_started = time.perf_counter()
        (self.directory / "parts").mkdir(parents=True, exist_ok=True)
        for stale in self.directory.glob(PREFIX + "*.tmp"):
            shutil.rmtree(stale, ignore_errors=True)
        if resume is None:
            self.parts: list[dict[str, Any]] = []
            self.counts = (0, 0, 0)
            self.prior_wall = 0.0
            self.sessions = 1
            self.written = 0
        else:
            state = resume.state
            self.parts = [dict(p) for p in state["parts"]]
            self.counts = (len(resume.rows), len(resume.tables), len(resume.payload_roots))
            self.prior_wall = float(state["wall_seconds"])
            self.sessions = int(state["sessions"]) + 1
            self.written = int(state.get("written", 0))
            keep_names = {p["name"] for p in self.parts}
            for part in (self.directory / "parts").iterdir():
                if part.name not in keep_names:
                    part.unlink()

    def wall_seconds(self) -> float:
        return self.prior_wall + time.perf_counter() - self.session_started

    def write(
        self,
        *,
        boundary: int,
        snapshot: str,
        transient: Mapping[str, Any] | None,
        execution: Any,
        next_step: Mapping[str, Any],
        rows: list[Any],
        tables: list[Any],
        payload_roots: list[Any],
        founded: Mapping[int, int],
        skipped: Mapping[str, int],
        searched: Mapping[int, int],
        restore_checks: list[Any],
        main_resolves: int,
        work: Mapping[str, int],
        reset_identity: Mapping[str, Any],
        continuation: Mapping[str, Any],
    ) -> None:
        started = time.perf_counter()
        n_rows, n_tables, n_roots = self.counts
        part = {
            "rows": rows[n_rows:],
            "tables": tables[n_tables:],
            "payload_roots": payload_roots[n_roots:],
        }
        name = f"part-{boundary:06d}.pt"
        data = _bytes(part)
        tmp = self.directory / "parts" / (name + ".tmp")
        _write(tmp, data)
        os.replace(tmp, self.directory / "parts" / name)
        self.parts = [p for p in self.parts if p["name"] != name] + [
            {"name": name, "sha256": _sha(data)}
        ]
        self.counts = (len(rows), len(tables), len(payload_roots))
        snapshot_bytes = snapshot.encode("ascii")
        payloads = _canonical_payloads(
            {
                "initial": {str(c): dict(a.payload) for c, a in sorted(execution.initial.items())},
                "retry": {str(c): dict(a.payload) for c, a in sorted(execution.retry.items())},
            }
        )
        self.written += 1
        state = {
            "schema": SEARCH_CHECKPOINT_SCHEMA,
            "identity": self.identity,
            "boundary": int(boundary),
            "founded": [[int(c), int(t)] for c, t in founded.items()],
            "skipped": [[str(k), int(v)] for k, v in skipped.items()],
            "searched": [[int(c), int(n)] for c, n in searched.items()],
            "restore_checks": list(restore_checks),
            "main_resolves": int(main_resolves),
            "work": dict(work),
            "reset_identity": dict(reset_identity),
            "continuation": dict(continuation),
            "parts": self.parts,
            "replay": {
                "snapshot_sha256": _sha(snapshot_bytes),
                "transient": None if transient is None else dict(transient),
                "payloads": payloads,
                "step_digest": replay_step_digest(next_step),
                "turn": int(next_step["turn"]),
            },
            "wall_seconds": self.wall_seconds(),
            "sessions": self.sessions,
            "written": self.written,
        }
        final = self.directory / f"{PREFIX}{boundary:06d}"
        tmp_dir = final.with_name(final.name + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        _write(tmp_dir / "snapshot.b64", snapshot_bytes)
        _write(tmp_dir / "state.pt", _bytes(state))
        _fsync_dir(tmp_dir)
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp_dir, final)
        _fsync_dir(self.directory)
        for _, old in sorted(
            (b, p) for p in self.directory.iterdir() if (b := _boundary(p)) is not None
        )[: -self.keep]:
            shutil.rmtree(old, ignore_errors=True)
        if self.log is not None:
            self.log(
                {
                    "search_checkpoint": int(boundary),
                    "bytes": len(snapshot_bytes) + len(data),
                    "seconds": round(time.perf_counter() - started, 4),
                }
            )

    def summary(self) -> dict[str, Any]:
        return {"checkpoints": int(self.written), "sessions": int(self.sessions)}


def decode_snapshot(snapshot: str) -> bytes:  # pragma: no cover - debugging helper
    return base64.b64decode(snapshot)
