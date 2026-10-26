"""Decision-boundary checkpoints and exact resume for heuristic teacher collection.

Spot collectors are preempted every few hours while one world takes longer, so
``labor_teacher_data`` checkpoints every world at every decision boundary of the
window and of the randomized return tail and resumes from the newest valid
checkpoint (``--resume``). Turn boundaries are authoritative: a checkpoint is
taken only after a decision fully executed; a kill anywhere inside a decision
(including the shadow-heuristic step) resumes from the preceding boundary and
redoes that decision.

What a checkpoint at boundary ``b`` (``b`` completed decisions, window plus
tail) holds:

* ``snapshot.b64`` -- the simulator snapshot of boundary ``b - 1``. It is the
  snapshot the collector already captured for that decision's shadow-heuristic
  step, so checkpointing adds no native capture; plus the live-only state a
  snapshot lacks (``capture_transient``: every ``[NonSerialized]`` scalar/array
  field of camps, their ``EconomicLaborPlan`` and hunters,
  ``Tile.CultivationOperatingFraction`` and the live ``FixedHousing`` order),
  restored by the resume reset. Measured: the actor-labor step leaves
  ``EconomicLaborPlan.ActorActivityAuthority``/``Actor*Workers`` set past the
  boundary and the next turn's heuristic settlement valuation reads them, so a
  snapshot alone diverges (move-candidate scores off by 0.05).
* ``state.json`` -- the exact ``step_labor`` initial/retry payloads of decision
  ``b - 1``, the digest of its successor step (the boundary-``b`` response),
  the episode continuation identity (start turn, trajectory id), every
  collector field that affects later output (counters, label reasons,
  quarantine, founded turns, reset identity, surviving camps, the tail loop
  state: tracked/active camps, per-head rewards, years, terminal flags, reasons
  and the iteration count) and the run identity.
* Rows are appended incrementally: one ``rows/part-<b>.pt`` per window
  decision (``torch.save`` of that decision's row tensors), referenced from
  ``state.json`` with its SHA-256; nothing is rewritten.

Why not the snapshot of boundary ``b`` alone: a ``SaveCodec`` snapshot restores
the world but not the environment's per-village context (previous decision,
post-execution labor feedback, ...), which the next observation reports. The
resume therefore resets from the boundary ``b - 1`` snapshot with the
interrupted episode's start turn and trajectory id (``reset`` ``continuation``)
and replays the recorded decision ``b - 1`` (one native resolve); that rebuilds
the context exactly, and the successor step must match the recorded digest
(byte-identical response except wall-clock ``*Ms`` timings) or the resume is
refused (``ContinuationMismatch``; the caller then recollects the world from
its reset, which is the uninterrupted run by definition).

Randomness is keyed, not stored: the tail stopping schedule is frozen from
``seed_base + world_seed`` and personality draws are keyed by world; the
teacher and simulator are deterministic. Snapshot bytes are not stable across
processes (the generation report stores wall-clock stage timings, which
``SaveCodec.Fingerprint`` excludes); they are stable within one episode
because a reset carries them forward, and no output depends on them.

Writes are atomic: ``ckpt-<b>.tmp`` then rename; the last ``keep`` (default 2)
checkpoints are kept, so a torn newest checkpoint falls back to its
predecessor.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import torch

CHECKPOINT_SCHEMA = 1
WORK_DIR = "_work"
RUN_RECORD = "collection.json"
CHECKPOINT_PREFIX = "ckpt-"


class ContinuationMismatch(RuntimeError):
    """The replayed boundary differs from the recorded one; this checkpoint cannot resume."""


class CheckpointIdentityError(RuntimeError):
    """A checkpoint belongs to another run (arguments, code, normalizer or producer)."""


def step_digest(step: Mapping[str, Any]) -> str:
    """SHA-256 of a decision step response without its wall-clock timings (top-level ``*Ms``)."""
    body = {k: v for k, v in step.items() if not str(k).endswith("Ms")}
    return hashlib.sha256(
        orjson.dumps(body, option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY)
    ).hexdigest()


def _canonical_payloads(payloads: Mapping[str, Any]) -> dict[str, Any]:
    """The payloads exactly as the bridge client serializes them, parsed back."""
    return orjson.loads(orjson.dumps(payloads, option=orjson.OPT_SERIALIZE_NUMPY))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_file(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    _write_bytes(tmp, data)
    os.replace(tmp, path)


def _fsync_dir(path: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _boundary_of(path: Path) -> int | None:
    name = path.name
    if not path.is_dir() or not name.startswith(CHECKPOINT_PREFIX) or name.endswith(".tmp"):
        return None
    try:
        return int(name[len(CHECKPOINT_PREFIX):])
    except ValueError:
        return None


def _result_state(result: Any) -> dict[str, Any]:
    return {
        "decisions": int(result.decisions),
        "native_resolves": int(result.native_resolves),
        "counters": [[k, int(v)] for k, v in result.counters.items()],
        "reasons": [[k, int(v)] for k, v in result.reasons.items()],
        "quarantine": list(result.quarantine),
        "surviving_camps": int(result.surviving_camps),
        "reset_identity": dict(result.reset_identity),
        "tail_decisions": int(result.tail_decisions),
        "tail_native_resolves": int(result.tail_native_resolves),
        "window_seconds": float(result.window_seconds),
        "tail_seconds": float(result.tail_seconds),
    }


def _restore_result(result: Any, state: Mapping[str, Any]) -> None:
    result.decisions = int(state["decisions"])
    result.native_resolves = int(state["native_resolves"])
    result.counters = defaultdict(int, {k: int(v) for k, v in state["counters"]})
    result.reasons = defaultdict(int, {k: int(v) for k, v in state["reasons"]})
    result.quarantine = list(state["quarantine"])
    result.surviving_camps = int(state["surviving_camps"])
    result.reset_identity = dict(state["reset_identity"])
    result.tail_decisions = int(state["tail_decisions"])
    result.tail_native_resolves = int(state["tail_native_resolves"])
    result.window_seconds = float(state["window_seconds"])
    result.tail_seconds = float(state["tail_seconds"])


@dataclass
class ResumePoint:
    """A validated checkpoint; ``restore`` rebuilds the bridge and the collector result."""

    directory: Path
    state: dict[str, Any]
    snapshot: str
    rows: list[dict[str, torch.Tensor]]
    reset_kwargs: dict[str, Any]

    @property
    def boundary(self) -> int:
        return int(self.state["boundary"])

    @property
    def founded(self) -> dict[int, int]:
        return {int(c): int(t) for c, t in self.state["founded"]}

    @property
    def window_done(self) -> int:
        return int(self.state["window_done"])

    @property
    def tail_state(self) -> Any:
        if self.state["tail_state"] is None:
            return None
        from .labor_bc import TeacherTailState

        return TeacherTailState.from_state(self.state["tail_state"])

    def restore(self, labor_bridge: Any, result: Any) -> Mapping[str, Any]:
        """Reset from the boundary ``b - 1`` snapshot, replay decision ``b - 1`` and verify it."""
        _restore_result(result, self.state["result"])
        result.rows = list(self.rows)
        replay = self.state["replay"]
        continuation = dict(self.state["continuation"])
        if replay.get("transient") is not None:
            continuation["transient"] = replay["transient"]
        labor_bridge.reset(
            int(self.state["world_seed"]),
            snapshot=self.snapshot,
            continuation=continuation,
            **self.reset_kwargs,
        )
        initial = {int(c): dict(p) for c, p in replay["payloads"]["initial"].items()}
        retry = {int(c): dict(p) for c, p in replay["payloads"]["retry"].items()}
        step, _ = labor_bridge.step_labor(initial, retry or None)
        actual = step_digest(step)
        if actual != replay["step_digest"]:
            raise ContinuationMismatch(
                f"world {self.state['world_seed']} checkpoint {self.boundary}: replayed boundary digest "
                f"{actual} != recorded {replay['step_digest']}"
            )
        return step


class WorldCheckpointer:
    """Writes one world's decision-boundary checkpoints (``collect_heuristic_world`` hook)."""

    def __init__(
        self,
        directory: Path,
        identity: Mapping[str, Any],
        *,
        keep: int = 2,
        resume: ResumePoint | None = None,
        log: Any = None,
    ) -> None:
        if keep < 1:
            raise ValueError("checkpoint keep must be positive")
        self.directory = Path(directory)
        self.identity = dict(identity)
        self.keep = int(keep)
        self.log = log
        self.session_started = time.perf_counter()
        (self.directory / "rows").mkdir(parents=True, exist_ok=True)
        for stale in self.directory.glob(CHECKPOINT_PREFIX + "*.tmp"):
            shutil.rmtree(stale, ignore_errors=True)
        if resume is None:
            self.parts: list[dict[str, Any]] = []
            self.rows_recorded = 0
            self.founded: dict[int, int] = {}
            self.continuation: dict[str, Any] | None = None
            self.window_done = 0
            self.prior_wall = 0.0
            self.sessions = 1
            self.overhead = {"count": 0, "seconds": 0.0, "bytes": 0, "max_seconds": 0.0, "max_bytes": 0}
        else:
            state = resume.state
            self.parts = [dict(p) for p in state["parts"]]
            self.rows_recorded = int(state["rows"])
            self.founded = resume.founded
            self.continuation = dict(state["continuation"])
            self.window_done = resume.window_done
            self.prior_wall = float(state["wall_seconds"])
            self.sessions = int(state["sessions"]) + 1
            self.overhead = dict(state["overhead"])
            # Parts beyond the resumed boundary belong to a lost or torn later checkpoint.
            keep_names = {p["name"] for p in self.parts}
            for part in (self.directory / "rows").iterdir():
                if part.name not in keep_names:
                    part.unlink()

    def wall_seconds(self) -> float:
        return self.prior_wall + time.perf_counter() - self.session_started

    # -- collect_heuristic_world hooks -------------------------------------------------

    def begin(self, result: Any, step: Mapping[str, Any]) -> None:
        trajectory = result.reset_identity.get("fingerprint")
        if not isinstance(trajectory, str) or not trajectory:
            raise ValueError("collection checkpoints need the reset fingerprint (episode trajectory id)")
        self.continuation = {"startTurn": int(step["turn"]), "trajectoryId": trajectory}
        self.founded = {int(row["camp"]): int(step["turn"]) for row in step["villages"]}

    def window(self, result: Any, transition: Any, founded: Mapping[int, int], done: int) -> None:
        started = time.perf_counter()
        self.founded = {int(c): int(t) for c, t in founded.items()}
        new_rows = result.rows[self.rows_recorded:]
        name = f"part-{done:06d}.pt"
        path = self.directory / "rows" / name
        buffer = _torch_bytes(new_rows)
        _atomic_file(path, buffer)
        self.parts.append({"name": name, "sha256": _sha256_bytes(buffer), "rows": len(new_rows)})
        self.rows_recorded = len(result.rows)
        self.window_done = int(done)
        self._write(result, transition, None, started, len(buffer))

    def tail(self, result: Any, transition: Any, state: Any) -> None:
        self._write(result, transition, state.state_dict(), time.perf_counter(), 0)

    # -- writing ---------------------------------------------------------------------

    def _write(
        self, result: Any, transition: Any, tail_state: Any, started: float, part_bytes: int
    ) -> None:
        if self.continuation is None:
            raise ValueError("checkpoint written before begin()")
        if transition.snapshot is None:
            raise ValueError("teacher transition lacks its boundary snapshot")
        boundary = int(result.decisions) + int(result.tail_decisions)
        snapshot = transition.snapshot.encode("ascii")
        payloads = _canonical_payloads(transition.replay_payloads())
        overhead = dict(self.overhead)
        state = {
            "schema": CHECKPOINT_SCHEMA,
            "identity": self.identity,
            "world_seed": int(result.world_seed),
            "boundary": boundary,
            "window_done": int(self.window_done),
            "tail_state": tail_state,
            "founded": [[int(c), int(t)] for c, t in self.founded.items()],
            "continuation": self.continuation,
            "result": _result_state(result),
            "parts": self.parts,
            "rows": int(self.rows_recorded),
            "replay": {
                "snapshot_sha256": _sha256_bytes(snapshot),
                "transient": None if transition.transient is None else dict(transition.transient),
                "payloads": payloads,
                "step_digest": step_digest(transition.next_step),
                "turn": int(transition.next_step["turn"]),
            },
            "wall_seconds": self.wall_seconds(),
            "sessions": int(self.sessions),
            "overhead": overhead,
        }
        final = self.directory / f"{CHECKPOINT_PREFIX}{boundary:06d}"
        tmp = final.with_name(final.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        _write_bytes(tmp / "snapshot.b64", snapshot)
        body = json.dumps(state, sort_keys=True, allow_nan=False).encode()
        _write_bytes(tmp / "state.json", body)
        _fsync_dir(tmp)
        if final.exists():  # a resumed run rewrites the boundary it resumed from
            shutil.rmtree(final)
        os.replace(tmp, final)
        _fsync_dir(self.directory)
        for old in sorted(
            (b, p) for p in self.directory.iterdir() if (b := _boundary_of(p)) is not None
        )[: -self.keep]:
            shutil.rmtree(old[1], ignore_errors=True)
        seconds = time.perf_counter() - started
        size = len(snapshot) + len(body) + int(part_bytes)
        self.overhead = {
            "count": int(overhead["count"]) + 1,
            "seconds": float(overhead["seconds"]) + seconds,
            "bytes": int(overhead["bytes"]) + size,
            "max_seconds": max(float(overhead["max_seconds"]), seconds),
            "max_bytes": max(int(overhead["max_bytes"]), size),
        }
        if self.log is not None:
            self.log(
                {
                    "world": int(result.world_seed),
                    "checkpoint": boundary,
                    "checkpoint_bytes": size,
                    "checkpoint_seconds": round(seconds, 4),
                }
            )

    def summary(self) -> dict[str, Any]:
        count = max(int(self.overhead["count"]), 1)
        return {
            "checkpoints": int(self.overhead["count"]),
            "sessions": int(self.sessions),
            "mean_seconds": float(self.overhead["seconds"]) / count,
            "max_seconds": float(self.overhead["max_seconds"]),
            "mean_bytes": float(self.overhead["bytes"]) / count,
            "max_bytes": int(self.overhead["max_bytes"]),
        }


def _torch_bytes(value: Any) -> bytes:
    import io

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def load_resume_point(
    directory: Path,
    identity: Mapping[str, Any],
    reset_kwargs: Mapping[str, Any],
    *,
    log: Any = None,
) -> ResumePoint | None:
    """The newest valid checkpoint of one world, or None.

    A checkpoint of another run identity is refused (``CheckpointIdentityError``),
    never skipped; a torn or corrupt one is reported and its predecessor tried.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = sorted(
        ((b, p) for p in directory.iterdir() if (b := _boundary_of(p)) is not None), reverse=True
    )
    for boundary, path in candidates:
        try:
            state = json.loads((path / "state.json").read_text())
        except (OSError, ValueError) as error:
            _warn(log, directory, boundary, f"unreadable state: {error}")
            continue
        if state.get("schema") != CHECKPOINT_SCHEMA:
            raise CheckpointIdentityError(
                f"{path}: checkpoint schema {state.get('schema')!r} != {CHECKPOINT_SCHEMA}"
            )
        if state.get("identity") != dict(identity):
            saved = state.get("identity") or {}
            differing = sorted(
                key for key in set(saved) | set(identity) if saved.get(key) != identity.get(key)
            )
            raise CheckpointIdentityError(
                f"{path}: checkpoint belongs to another collection run; differing identity fields: "
                f"{differing}. Resume with the original arguments, code and normalizer, or collect "
                "into a new --out"
            )
        if int(state.get("boundary", -1)) != boundary:
            _warn(log, directory, boundary, "boundary field differs from the directory name")
            continue
        try:
            snapshot_bytes = (path / "snapshot.b64").read_bytes()
        except OSError as error:
            _warn(log, directory, boundary, f"unreadable snapshot: {error}")
            continue
        if _sha256_bytes(snapshot_bytes) != state["replay"]["snapshot_sha256"]:
            _warn(log, directory, boundary, "snapshot hash mismatch")
            continue
        rows: list[dict[str, torch.Tensor]] = []
        valid = True
        for part in state["parts"]:
            part_path = directory / "rows" / part["name"]
            if not part_path.is_file() or _sha256_file(part_path) != part["sha256"]:
                _warn(log, directory, boundary, f"row part {part['name']} missing or corrupt")
                valid = False
                break
            loaded = torch.load(part_path, map_location="cpu", weights_only=True)
            if len(loaded) != int(part["rows"]):
                valid = False
                break
            rows.extend(loaded)
        if not valid or len(rows) != int(state["rows"]):
            continue
        base64.b64decode(snapshot_bytes, validate=True)
        return ResumePoint(
            directory=directory,
            state=state,
            snapshot=snapshot_bytes.decode("ascii"),
            rows=rows,
            reset_kwargs=dict(reset_kwargs),
        )
    if candidates:
        _warn(log, directory, -1, "no valid checkpoint; the world restarts from its reset")
    return None


def _warn(log: Any, directory: Path, boundary: int, message: str) -> None:
    record = {"checkpoint_warning": message, "directory": str(directory), "boundary": boundary}
    if log is not None:
        log(record)
    else:
        print(json.dumps(record), file=sys.stderr, flush=True)
