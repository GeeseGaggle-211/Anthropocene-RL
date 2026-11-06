"""Fail-closed orchestration for the reviewed Gate-1 execution argv.

The dispatcher never constructs or edits scientific commands.  It executes the
exact argv sealed in a VM-materialized freeze candidate, without a shell, and
records operational progress and resource use.  It neither reads outcomes for
decision-making nor grants a Gate/readiness/selection decision.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gate1_freeze_candidate as freeze
from .provenance import checkpoint_hash, digest

SCHEMA = 1
MAX_POLL_SECONDS = 2.0
EXPECTED_WORLD_COUNT = 64
EXPECTED_EVALUATION_COUNT = 40
MAX_REVIEWED_CONCURRENCY = 8


class DispatchError(RuntimeError):
    """The reviewed command set or its operational evidence is unsafe."""


def _fail(message: str) -> None:
    raise DispatchError("Gate 1 execution dispatcher: " + message)


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    if "digest" in body:
        _fail("cannot reseal a document carrying a digest")
    result = dict(body)
    result["digest"] = digest(result)
    return result


def _write_new(path: Path, body: dict[str, Any]) -> None:
    if path.exists():
        _fail(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sealed_json(path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        _fail(f"{kind} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DispatchError(f"cannot parse {kind}: {path}") from error
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("kind") != kind:
        _fail(f"{kind} has the wrong schema or kind")
    body = {key: item for key, item in value.items() if key != "digest"}
    if set(value) != set(body) | {"digest"} or value["digest"] != digest(body):
        _fail(f"{kind} is not digest sealed")
    return value


def _file_ref(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        _fail(f"evidence is not a regular file: {path}")
    return {"path": str(path), "sha256": checkpoint_hash(path), "size": path.stat().st_size}


def _same_ref(path: Path, ref: Any, what: str) -> None:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256", "size"}:
        _fail(f"{what} reference is malformed")
    observed = _file_ref(path)
    if observed != ref:
        _fail(f"{what} changed after it was recorded")


def _command_out(argv: list[str]) -> Path | None:
    positions = [index for index, value in enumerate(argv) if value == "--out"]
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        _fail("reviewed argv has a malformed --out")
    return Path(argv[positions[0] + 1])


def _check_argv(argv: Any, *, expected_subcommand: str | None = None) -> list[str]:
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        _fail("reviewed command argv is malformed")
    if not Path(argv[0]).is_absolute():
        _fail("reviewed command interpreter is not absolute")
    if expected_subcommand is not None and expected_subcommand not in argv:
        _fail(f"reviewed command is not {expected_subcommand}")
    return list(argv)


@dataclass(frozen=True)
class Context:
    materialized_root: Path
    rl_root: Path
    state_dir: Path
    candidate_path: Path
    review_path: Path
    candidate: dict[str, Any]
    review: dict[str, Any]
    commands_path: Path
    commands: dict[str, Any]
    command_set: dict[str, Any]
    world_ids: frozenset[str]
    capture_bound_per_world: int


def _context(
    materialized_root: Path, rl_root: Path, state_dir: Path, review: Path | None
) -> Context:
    root = materialized_root.resolve()
    source_root = rl_root.resolve()
    if not root.is_dir() or root.is_symlink():
        _fail("materialized root is absent or unsafe")
    if not (source_root / "src/anthropocene_rl/autonomous_v7").is_dir():
        _fail("RL source root is malformed")
    module_path = Path(freeze.__file__).resolve()
    try:
        module_path.relative_to((source_root / "src").resolve())
    except ValueError as error:
        raise DispatchError("dispatcher imported Gate-1 code from another source root") from error
    candidate_path = root / "freeze-candidate.json"
    candidate = freeze._validate_seal(
        json.loads(candidate_path.read_text(encoding="utf-8")),
        "gate1_production_freeze_candidate",
    )
    if candidate.get("execution_tier") != "vm":
        _fail("dispatcher requires a VM-materialized candidate")
    freeze.validate_candidate(candidate_path)
    review_path = (review or root / "astra-review.json").resolve()
    freeze._review(candidate, review_path)
    review_body = _sealed_json(review_path, "independent_gate1_freeze_review")
    command_ref = candidate["proposals"]["execution_commands"]
    commands_path = (root / command_ref["path"]).resolve()
    try:
        commands_path.relative_to(root)
    except ValueError as error:
        raise DispatchError("execution command artifact escapes the materialized root") from error
    commands = freeze._read_ref(command_ref, root, kind="gate1_post_review_execution_commands")
    if (
        commands_path.stat().st_size != command_ref["size"]
        or checkpoint_hash(commands_path) != command_ref["sha256"]
    ):
        _fail("execution command artifact differs from the candidate reference")
    if Path(commands.get("review_path", "")).resolve() != review_path:
        _fail("reviewed command artifact names another review path")
    command_set = commands.get("commands")
    if not isinstance(command_set, dict):
        _fail("execution command set is absent")
    evaluations = command_set.get("checkpoint_evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != EXPECTED_EVALUATION_COUNT:
        _fail("execution command set does not contain exactly 40 node/pane jobs")
    evaluation = freeze._read_ref(
        candidate["proposals"]["evaluation"],
        root,
        kind="gate1_evaluation_claim_protocol_proposal",
    )
    worlds = evaluation.get("cohort", {}).get("worlds")
    bounds = evaluation.get("capture_bounds")
    if (
        not isinstance(worlds, list)
        or len(worlds) != EXPECTED_WORLD_COUNT
        or len({row.get("id") for row in worlds if isinstance(row, dict)}) != EXPECTED_WORLD_COUNT
        or not isinstance(bounds, dict)
        or type(bounds.get("max_canonical_bytes")) is not int
        or bounds["max_canonical_bytes"] <= 0
    ):
        _fail("reviewed world inventory or capture bound is malformed")
    return Context(
        root,
        source_root,
        state_dir.resolve(),
        candidate_path,
        review_path,
        candidate,
        review_body,
        commands_path,
        commands,
        command_set,
        frozenset(row["id"] for row in worlds),
        bounds["max_canonical_bytes"],
    )


def _base_env(context: Context) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str((context.rl_root / "src").resolve())
    return environment


def _state_path(context: Context, name: str) -> Path:
    return context.state_dir / name


def _command_receipt(
    argv: list[str], started_ns: int, ended_ns: int, exit_code: int, log_path: Path
) -> dict[str, Any]:
    return {
        "argv": argv,
        "argv_digest": digest({"argv": argv}),
        "started_ns": started_ns,
        "ended_ns": ended_ns,
        "exit_code": exit_code,
        "log": _file_ref(log_path),
    }


def _run_sync(context: Context, argv: list[str], log_path: Path) -> dict[str, Any]:
    _check_argv(argv)
    if log_path.exists():
        _fail(f"command log already exists: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time_ns()
    with log_path.open("xb") as log:
        result = subprocess.run(
            argv,
            cwd=context.rl_root,
            env=_base_env(context),
            shell=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    ended = time.time_ns()
    receipt = _command_receipt(argv, started, ended, result.returncode, log_path)
    if result.returncode != 0:
        _fail(f"reviewed command failed ({result.returncode}); log: {log_path}")
    return receipt


def _node_index(context: Context) -> dict[str, dict[str, Any]]:
    dag = freeze._read_ref(
        context.candidate["proposals"]["artifact_dag"],
        context.materialized_root,
        kind="gate1_artifact_dag_proposal",
    )
    return {row["id"]: row for row in dag["nodes"]}


def _validate_preflight_outputs(context: Context) -> dict[str, dict[str, Any]]:
    nodes = _node_index(context)
    proposals = context.candidate["proposals"]
    result: dict[str, dict[str, Any]] = {}
    specifications = (
        (
            "contract8_comparator",
            "gate1_contract8_comparator_result",
            "prior-reference",
            "contract8_comparator",
            "contract8_corpus",
        ),
        (
            "guarded_bc_comparator",
            "gate1_guarded_bc_comparator_result",
            "t2m-guarded-bc-seed6",
            "guarded_bc_adapter",
            "guarded_bc_corpus",
        ),
    )
    for command_key, kind, node_id, protocol_key, corpus_key in specifications:
        argv = _check_argv(context.command_set[command_key])
        path = _command_out(argv)
        if path is None:
            _fail(f"{command_key} lacks its reviewed output")
        body = _sealed_json(path, kind)
        if (
            body.get("status") != "PASS"
            or body.get("checkpoint_sha256") != nodes[node_id]["artifact"]["sha256"]
            or body.get("protocol_sha256") != proposals[protocol_key]["sha256"]
            or body.get("corpus_sha256") != proposals[corpus_key]["sha256"]
        ):
            _fail(f"{command_key} output is bound to another checkpoint/protocol/corpus")
        result[command_key] = {"body_digest": body["digest"], **_file_ref(path)}
    worker_argv = _check_argv(context.command_set["capture_current_worker"])
    worker_path = _command_out(worker_argv)
    if worker_path is None:
        _fail("worker-capture command lacks its reviewed output")
    worker = freeze._worker_capture(worker_path, context.candidate)
    result["capture_current_worker"] = {"body_digest": worker["digest"], **_file_ref(worker_path)}
    return result


def _preflight_body(
    context: Context,
    outputs: dict[str, dict[str, Any]],
    *,
    adopted: bool,
    receipts: list[dict[str, Any]],
    context_note: str | None,
) -> dict[str, Any]:
    return _seal(
        {
            "schema": SCHEMA,
            "kind": "gate1_dispatch_preflight_receipt",
            "status": "PASS-OPERATIONAL-PREFLIGHT-ONLY",
            "candidate_digest": context.candidate["digest"],
            "candidate": _file_ref(context.candidate_path),
            "review_digest": context.review["digest"],
            "review": _file_ref(context.review_path),
            "execution_commands_digest": context.commands["digest"],
            "execution_commands": _file_ref(context.commands_path),
            "source_root": str(context.rl_root),
            "adopted_existing_outputs": adopted,
            "context_note": context_note,
            "commands_executed_by_dispatcher": receipts,
            "validated_outputs": outputs,
            "authority": {
                "checkpoint_evaluation": False,
                "selection": False,
                "readiness": False,
                "production_gate1": "not-decided",
            },
        }
    )


def preflight(context: Context, *, adopt_existing: bool, context_note: str | None) -> Path:
    if context.state_dir.exists():
        _fail("preflight state directory already exists")
    context.state_dir.mkdir(parents=True)
    receipts = []
    validate_argv = _check_argv(
        context.command_set["validate_candidate"], expected_subcommand="validate"
    )
    receipts.append(_run_sync(context, validate_argv, _state_path(context, "logs/validate.log")))
    if not adopt_existing:
        for key in ("contract8_comparator", "guarded_bc_comparator", "capture_current_worker"):
            argv = _check_argv(context.command_set[key])
            target = _command_out(argv)
            if target is None or target.exists():
                _fail(f"{key} output is absent from argv or already exists")
        for key in ("contract8_comparator", "guarded_bc_comparator", "capture_current_worker"):
            receipts.append(
                _run_sync(
                    context,
                    _check_argv(context.command_set[key]),
                    _state_path(context, f"logs/{key}.log"),
                )
            )
    outputs = _validate_preflight_outputs(context)
    body = _preflight_body(
        context,
        outputs,
        adopted=adopt_existing,
        receipts=receipts,
        context_note=context_note,
    )
    path = _state_path(context, "preflight.json")
    _write_new(path, body)
    return path


def _validate_preflight(context: Context) -> dict[str, Any]:
    body = _sealed_json(_state_path(context, "preflight.json"), "gate1_dispatch_preflight_receipt")
    if (
        body.get("status") != "PASS-OPERATIONAL-PREFLIGHT-ONLY"
        or body.get("candidate_digest") != context.candidate["digest"]
        or body.get("review_digest") != context.review["digest"]
        or body.get("execution_commands_digest") != context.commands["digest"]
    ):
        _fail("preflight receipt is bound to another reviewed command set")
    _same_ref(context.candidate_path, body.get("candidate"), "preflight candidate")
    _same_ref(context.review_path, body.get("review"), "preflight review")
    _same_ref(context.commands_path, body.get("execution_commands"), "preflight commands")
    observed = _validate_preflight_outputs(context)
    if observed != body.get("validated_outputs"):
        _fail("preflight output bytes changed after validation")
    return body


def _proc_status_rss(pid: int) -> int:
    path = Path("/proc") / str(pid) / "status"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return 0
    for line in lines:
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                return int(fields[1]) * 1024
    return 0


def _proc_children(pid: int) -> list[int]:
    path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    try:
        return [int(value) for value in path.read_text(encoding="ascii").split()]
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return []


def _process_tree_rss(pid: int) -> int:
    if not Path("/proc/self/status").is_file():
        _fail("resource-bearing dispatch requires Linux /proc")
    seen: set[int] = set()
    stack = [pid]
    total = 0
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        total += _proc_status_rss(current)
        stack.extend(_proc_children(current))
    return total


def _directory_bytes(paths: list[Path]) -> int:
    total = 0
    for root in paths:
        if root.is_file() and not root.is_symlink():
            total += root.stat().st_size
        elif root.is_dir() and not root.is_symlink():
            total += sum(
                path.stat().st_size
                for path in root.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
    return total


@dataclass
class WorldProgress:
    candidate_digest: str
    node: str
    pane: str
    expected_ids: frozenset[str]
    valid: dict[Path, tuple[int, int]] = field(default_factory=dict)

    def scan(self, world_root: Path) -> int:
        if not world_root.exists():
            return 0
        if not world_root.is_dir() or world_root.is_symlink():
            _fail("world-artifact path is not a safe directory")
        for path in world_root.glob("*.json"):
            stat = path.stat()
            marker = (stat.st_size, stat.st_mtime_ns)
            if self.valid.get(path) == marker:
                continue
            try:
                body = _sealed_json(path, "gate1_fresh_common_producer_world")
            except DispatchError:
                continue  # a direct writer may still be completing this file
            if (
                body.get("candidate_digest") != self.candidate_digest
                or body.get("node") != self.node
                or body.get("pane") != self.pane
                or body.get("world_id") not in self.expected_ids
            ):
                _fail("streamed world artifact is bound to another reviewed execution")
            if any(other != path and other.name == path.name for other in self.valid):
                _fail("streamed world artifact identity is duplicated")
            self.valid[path] = marker
        return len(self.valid)


@dataclass
class Active:
    index: int
    node: str
    pane: str
    argv: list[str]
    out: Path
    world_root: Path
    process: subprocess.Popen[bytes]
    log_path: Path
    log_handle: Any
    started_ns: int
    progress: WorldProgress
    peak_rss_bytes: int = 0
    minimum_disk_free_bytes: int | None = None


def _evaluation_spec(context: Context, index: int) -> tuple[dict[str, Any], list[str], Path, Path]:
    row = context.command_set["checkpoint_evaluations"][index]
    if (
        not isinstance(row, dict)
        or set(row) != {"node", "pane", "out", "command"}
        or not isinstance(row["node"], str)
        or not isinstance(row["pane"], str)
    ):
        _fail("reviewed checkpoint-evaluation row is malformed")
    argv = _check_argv(row["command"], expected_subcommand="evaluate-reviewed")
    out = _command_out(argv)
    if out is None or str(out) != row["out"]:
        _fail("reviewed evaluation row and argv disagree on output")
    world_root = out.parent / f"{out.stem}.worlds"
    return row, argv, out, world_root


def _ensure_new_evaluation_targets(context: Context, indices: list[int]) -> None:
    seen: set[Path] = set()
    for index in indices:
        _row, _argv, out, world_root = _evaluation_spec(context, index)
        if out in seen or world_root in seen:
            _fail("reviewed evaluation commands repeat an output")
        seen.update((out, world_root))
        if out.exists() or world_root.exists():
            _fail(f"refusing to overwrite evaluation output: {out}")


def _start_evaluation(context: Context, index: int, log_path: Path) -> Active:
    row, argv, out, world_root = _evaluation_spec(context, index)
    if log_path.exists():
        _fail(f"evaluation log already exists: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("xb")
    try:
        process = subprocess.Popen(
            argv,
            cwd=context.rl_root,
            env=_base_env(context),
            shell=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log.close()
        raise
    return Active(
        index=index,
        node=row["node"],
        pane=row["pane"],
        argv=argv,
        out=out,
        world_root=world_root,
        process=process,
        log_path=log_path,
        log_handle=log,
        started_ns=time.time_ns(),
        progress=WorldProgress(
            context.candidate["digest"], row["node"], row["pane"], context.world_ids
        ),
    )


def _terminate(active: list[Active]) -> None:
    for job in active:
        if job.process.poll() is not None:
            continue
        try:
            if os.name == "posix":
                os.killpg(job.process.pid, signal.SIGTERM)
            else:
                job.process.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5.0
    for job in active:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            job.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(job.process.pid, signal.SIGKILL)
                else:
                    job.process.kill()
            except ProcessLookupError:
                pass


def _validate_completed_evaluation(context: Context, job: Active) -> dict[str, Any]:
    body = _sealed_json(job.out, "gate1_fresh_common_producer_evaluation")
    if (
        body.get("candidate_digest") != context.candidate["digest"]
        or body.get("node") != job.node
        or body.get("pane") != job.pane
        or body.get("world_count") != EXPECTED_WORLD_COUNT
        or body.get("producer_scope") != "fresh-common-producer-diagnostic"
        or body.get("historical_replay_claim") is not False
        or body.get("selection_allowed") is not False
        or body.get("readiness_claimed") is not False
    ):
        _fail("completed evaluation summary is not the reviewed diagnostic output")
    refs = body.get("world_artifacts")
    if not isinstance(refs, list) or len(refs) != EXPECTED_WORLD_COUNT:
        _fail("completed evaluation lacks exactly 64 world references")
    seen: set[str] = set()
    for ref in refs:
        if (
            not isinstance(ref, dict)
            or set(ref) != {"path", "sha256", "size"}
            or not isinstance(ref["path"], str)
        ):
            _fail("completed evaluation contains a malformed world reference")
        path = (job.out.parent / ref["path"]).resolve()
        try:
            path.relative_to(job.out.parent.resolve())
        except ValueError as error:
            raise DispatchError(
                "completed evaluation world reference escapes output root"
            ) from error
        world = _sealed_json(path, "gate1_fresh_common_producer_world")
        if (
            path.stat().st_size != ref["size"]
            or checkpoint_hash(path) != ref["sha256"]
            or world.get("candidate_digest") != context.candidate["digest"]
            or world.get("node") != job.node
            or world.get("pane") != job.pane
            or world.get("world_id") not in context.world_ids
            or world["world_id"] in seen
        ):
            _fail("completed evaluation contains an incompatible world artifact")
        seen.add(world["world_id"])
    if seen != set(context.world_ids):
        _fail("completed evaluation world coverage differs from the frozen cohort")
    return body


def _event(log, event: dict[str, Any]) -> None:
    log.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
    log.flush()


def _monitor_jobs(
    context: Context,
    active: list[Active],
    *,
    poll_seconds: float,
    reserve_bytes: int,
    event_log,
    fail_on_low_disk: bool = True,
) -> list[dict[str, Any]]:
    if not 0 < poll_seconds <= MAX_POLL_SECONDS:
        _fail("poll interval must be in (0, 2] seconds")
    completed = []
    while active:
        disk_free = shutil.disk_usage(context.materialized_root).free
        aggregate_rss = 0
        for job in list(active):
            rss = _process_tree_rss(job.process.pid)
            aggregate_rss += rss
            job.peak_rss_bytes = max(job.peak_rss_bytes, rss)
            job.minimum_disk_free_bytes = (
                disk_free
                if job.minimum_disk_free_bytes is None
                else min(job.minimum_disk_free_bytes, disk_free)
            )
            valid_worlds = job.progress.scan(job.world_root)
            bytes_written = _directory_bytes([job.world_root, job.out])
            _event(
                event_log,
                {
                    "at_ns": time.time_ns(),
                    "type": "progress",
                    "index": job.index,
                    "node": job.node,
                    "pane": job.pane,
                    "valid_worlds": valid_worlds,
                    "artifact_bytes": bytes_written,
                    "process_tree_rss_bytes": rss,
                    "aggregate_active_rss_bytes": aggregate_rss,
                    "disk_free_bytes": disk_free,
                },
            )
            returncode = job.process.poll()
            if returncode is None:
                continue
            job.log_handle.close()
            if returncode != 0:
                _terminate(active)
                _fail(f"reviewed evaluation {job.node}/{job.pane} failed ({returncode})")
            body = _validate_completed_evaluation(context, job)
            completed.append(
                {
                    "index": job.index,
                    "node": job.node,
                    "pane": job.pane,
                    "argv": job.argv,
                    "argv_digest": digest({"argv": job.argv}),
                    "started_ns": job.started_ns,
                    "ended_ns": time.time_ns(),
                    "exit_code": 0,
                    "peak_process_tree_rss_bytes": job.peak_rss_bytes,
                    "minimum_disk_free_bytes": job.minimum_disk_free_bytes,
                    "artifact_bytes": _directory_bytes([job.world_root, job.out]),
                    "valid_world_count": len(job.progress.valid),
                    "evaluation_digest": body["digest"],
                    "evaluation": _file_ref(job.out),
                    "log": _file_ref(job.log_path),
                }
            )
            active.remove(job)
        if fail_on_low_disk and disk_free < reserve_bytes:
            _terminate(active)
            _fail("disk free space fell below the approved reserve")
        if active:
            time.sleep(poll_seconds)
    return completed


def calibrate(context: Context, *, reserve_bytes: int, poll_seconds: float) -> Path:
    preflight_body = _validate_preflight(context)
    target = _state_path(context, "calibration.json")
    event_path = _state_path(context, "calibration-progress.jsonl")
    if target.exists() or event_path.exists():
        _fail("calibration evidence already exists")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        _fail("disk reserve must be a nonnegative integer byte count")
    _ensure_new_evaluation_targets(context, [0])
    hard_bound = context.capture_bound_per_world * EXPECTED_WORLD_COUNT
    free_before = shutil.disk_usage(context.materialized_root).free
    if free_before < hard_bound + reserve_bytes:
        _fail("disk free space cannot cover the reviewed 64-world capture bound plus reserve")
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("x", encoding="utf-8") as events:
        job = _start_evaluation(context, 0, _state_path(context, "logs/calibration.log"))
        try:
            receipts = _monitor_jobs(
                context,
                [job],
                poll_seconds=poll_seconds,
                reserve_bytes=reserve_bytes,
                event_log=events,
            )
        except BaseException:
            _terminate([job])
            if not job.log_handle.closed:
                job.log_handle.close()
            raise
    if len(receipts) != 1 or receipts[0]["valid_world_count"] != EXPECTED_WORLD_COUNT:
        _fail("calibration did not produce exactly 64 validated world artifacts")
    body = _seal(
        {
            "schema": SCHEMA,
            "kind": "gate1_dispatch_calibration_receipt",
            "status": "COMPLETE-OPERATIONAL-CALIBRATION-NOT-OUTCOME-REVIEW",
            "candidate_digest": context.candidate["digest"],
            "review_digest": context.review["digest"],
            "execution_commands_digest": context.commands["digest"],
            "preflight_digest": preflight_body["digest"],
            "first_declared_evaluation_index": 0,
            "world_count": EXPECTED_WORLD_COUNT,
            "capture_upper_bound_bytes": hard_bound,
            "disk_reserve_bytes": reserve_bytes,
            "disk_free_before_bytes": free_before,
            "poll_seconds": poll_seconds,
            "progress_log": _file_ref(event_path),
            "evaluation": receipts[0],
            "outcome_values_inspected_for_action": False,
            "authority": {"full_cohort": False, "selection": False, "readiness": False},
        }
    )
    _write_new(target, body)
    return target


def _validate_calibration(context: Context) -> dict[str, Any]:
    body = _sealed_json(
        _state_path(context, "calibration.json"), "gate1_dispatch_calibration_receipt"
    )
    preflight_body = _validate_preflight(context)
    if (
        body.get("status") != "COMPLETE-OPERATIONAL-CALIBRATION-NOT-OUTCOME-REVIEW"
        or body.get("candidate_digest") != context.candidate["digest"]
        or body.get("review_digest") != context.review["digest"]
        or body.get("execution_commands_digest") != context.commands["digest"]
        or body.get("preflight_digest") != preflight_body["digest"]
        or body.get("first_declared_evaluation_index") != 0
        or body.get("world_count") != EXPECTED_WORLD_COUNT
        or body.get("outcome_values_inspected_for_action") is not False
    ):
        _fail("calibration receipt is incomplete or bound to another collection")
    progress = body.get("progress_log")
    if not isinstance(progress, dict):
        _fail("calibration receipt lacks its progress log")
    _same_ref(Path(progress["path"]), progress, "calibration progress log")
    evaluation = body.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("index") != 0:
        _fail("calibration receipt does not describe the first reviewed evaluation")
    _same_ref(
        Path(evaluation["evaluation"]["path"]), evaluation["evaluation"], "calibration evaluation"
    )
    return body


def _approval(path: Path, context: Context, calibration_body: dict[str, Any]) -> dict[str, Any]:
    body = _sealed_json(path.resolve(), "gate1_dispatch_full_cohort_approval")
    exact = {
        "schema",
        "kind",
        "candidate_digest",
        "candidate_sha256",
        "review_digest",
        "execution_commands_digest",
        "calibration_digest",
        "decision",
        "reviewer_identity",
        "scope",
        "max_concurrency",
        "minimum_disk_reserve_bytes",
        "storage_safety_factor",
        "outcome_based_decision",
        "digest",
    }
    if set(body) != exact:
        _fail("full-cohort approval field set is not exact")
    concurrency = body.get("max_concurrency")
    factor = body.get("storage_safety_factor")
    reserve = body.get("minimum_disk_reserve_bytes")
    if (
        body.get("candidate_digest") != context.candidate["digest"]
        or body.get("candidate_sha256") != checkpoint_hash(context.candidate_path)
        or body.get("review_digest") != context.review["digest"]
        or body.get("execution_commands_digest") != context.commands["digest"]
        or body.get("calibration_digest") != calibration_body["digest"]
        or body.get("decision") != "approved"
        or body.get("reviewer_identity") != "root-orchestrator"
        or body.get("scope") != "remaining-39-reviewed-node-pane-evaluations"
        or body.get("outcome_based_decision") is not False
        or type(concurrency) is not int
        or not 1 <= concurrency <= MAX_REVIEWED_CONCURRENCY
        or type(reserve) is not int
        or reserve < 0
        or isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not 1 <= float(factor) <= 100
    ):
        _fail("full-cohort approval is invalid or bound to another calibration")
    return body


def full(context: Context, *, approval_path: Path, poll_seconds: float) -> Path:
    calibration_body = _validate_calibration(context)
    approval = _approval(approval_path, context, calibration_body)
    target = _state_path(context, "full-cohort.json")
    event_path = _state_path(context, "full-cohort-progress.jsonl")
    if target.exists() or event_path.exists():
        _fail("full-cohort evidence already exists")
    indices = list(range(1, EXPECTED_EVALUATION_COUNT))
    _ensure_new_evaluation_targets(context, indices)
    calibration_bytes = calibration_body["evaluation"]["artifact_bytes"]
    if type(calibration_bytes) is not int or calibration_bytes <= 0:
        _fail("calibration lacks a positive observed artifact byte count")
    projected = int(calibration_bytes * len(indices) * float(approval["storage_safety_factor"]))
    free_before = shutil.disk_usage(context.materialized_root).free
    reserve = approval["minimum_disk_reserve_bytes"]
    if free_before < projected + reserve:
        _fail("disk free space cannot cover approved calibrated projection plus reserve")
    pending = list(indices)
    active: list[Active] = []
    completed: list[dict[str, Any]] = []
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("x", encoding="utf-8") as events:
        try:
            while pending or active:
                while pending and len(active) < approval["max_concurrency"]:
                    index = pending.pop(0)
                    active.append(
                        _start_evaluation(
                            context, index, _state_path(context, f"logs/evaluation-{index:02d}.log")
                        )
                    )
                # Monitor until at least one active job completes, then refill.
                while active:
                    before = len(active)
                    receipts = _monitor_jobs(
                        context,
                        active,
                        poll_seconds=poll_seconds,
                        reserve_bytes=reserve,
                        event_log=events,
                    )
                    # _monitor_jobs drains the supplied list; this branch normally
                    # runs one concurrency wave.  It remains fail-fast.
                    completed.extend(receipts)
                    if len(active) < before:
                        break
                # Because the monitor drains the wave, launch the next wave.
        except BaseException:
            _terminate(active)
            for job in active:
                if not job.log_handle.closed:
                    job.log_handle.close()
            raise
    if len(completed) != len(indices):
        _fail("full-cohort dispatcher did not complete all remaining reviewed jobs")
    actual_bytes = sum(row["artifact_bytes"] for row in completed)
    body = _seal(
        {
            "schema": SCHEMA,
            "kind": "gate1_dispatch_full_cohort_receipt",
            "status": "COMPLETE-OPERATIONAL-COLLECTION-NOT-A-GATE-DECISION",
            "candidate_digest": context.candidate["digest"],
            "review_digest": context.review["digest"],
            "execution_commands_digest": context.commands["digest"],
            "calibration_digest": calibration_body["digest"],
            "approval_digest": approval["digest"],
            "approval": _file_ref(approval_path.resolve()),
            "concurrency": approval["max_concurrency"],
            "storage_safety_factor": approval["storage_safety_factor"],
            "projected_remaining_artifact_bytes": projected,
            "actual_remaining_artifact_bytes": actual_bytes,
            "minimum_disk_reserve_bytes": reserve,
            "disk_free_before_bytes": free_before,
            "poll_seconds": poll_seconds,
            "progress_log": _file_ref(event_path),
            "evaluations": sorted(completed, key=lambda row: row["index"]),
            "outcome_values_inspected_for_action": False,
            "authority": {
                "selection": False,
                "promotion": False,
                "readiness": False,
                "production_gate1": "not-decided",
            },
        }
    )
    _write_new(target, body)
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("preflight", "calibrate", "full"):
        stage = subparsers.add_parser(name)
        stage.add_argument("--materialized-root", type=Path, required=True)
        stage.add_argument("--rl-root", type=Path, required=True)
        stage.add_argument("--state-dir", type=Path, required=True)
        stage.add_argument("--review", type=Path)
        if name in {"calibrate", "full"}:
            stage.add_argument("--poll-seconds", type=float, default=1.0)
    preflight_parser = subparsers.choices["preflight"]
    preflight_parser.add_argument("--adopt-existing", action="store_true")
    preflight_parser.add_argument("--context-note")
    calibration_parser = subparsers.choices["calibrate"]
    calibration_parser.add_argument("--minimum-disk-reserve-bytes", type=int, required=True)
    full_parser = subparsers.choices["full"]
    full_parser.add_argument("--approval", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = _context(
        args.materialized_root,
        args.rl_root,
        args.state_dir,
        args.review,
    )
    if args.stage == "preflight":
        path = preflight(
            context,
            adopt_existing=args.adopt_existing,
            context_note=args.context_note,
        )
    elif args.stage == "calibrate":
        path = calibrate(
            context,
            reserve_bytes=args.minimum_disk_reserve_bytes,
            poll_seconds=args.poll_seconds,
        )
    else:
        path = full(context, approval_path=args.approval, poll_seconds=args.poll_seconds)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
