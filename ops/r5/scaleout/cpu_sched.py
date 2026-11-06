"""r5 scale-out CPU VM scheduler: search chunks with full payloads plus the actor panel.

Run by cpu_queue.sh (which owns the lock, the upload of the final state and the shutdown).
Stdlib only. Every task is a separate process whose completion is a file on disk, so the
scheduler is restartable: rerun it after a VM stop and finished tasks are skipped, while
interrupted ones are relaunched (labor_search --resume keeps every verified world shard;
a panel world reruns from its reset).

Tasks
  search chunk  one labor_search invocation for ONE world and one disjoint decision range
                [first, turns) with R roots (--jobs 1, one slot). Done when its output
                holds search_report.json (written even when a world yields no label rows).
                Disjoint ranges keep root ids unique across chunks (the A loader refuses a
                repeated world/turn/camp/source root id) and stratify roots over the window.
  panel world   one labor_pretrain_eval autonomous run for one development world, every
                mode (heuristic, h12, deterministic, stochastic), PANEL_JOBS slots. Gated on
                the actor checkpoint (fetched from GCS when the GPU pilot publishes it).

Admission: a task starts when its slots fit under MAX_SLOTS and MemAvailable >= MIN_FREE_GB,
at most one launch every LAUNCH_GAP_S seconds (memory ramps after a launch). Priority:
reserved-world chunks, then panel worlds (when the actor exists), then fitting chunks in
passes (each pass gives every world one stratum, rotated), then actor-root chunks.
A failed task is retried once (MAX_ATTEMPTS=2), then recorded as failed; the rest go on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def splits(text: str) -> list[tuple[int, int, int]]:
    """'0:13:4 13:25:4' -> [(first, turns, roots), ...] (disjoint, increasing)."""
    out = [tuple(int(x) for x in part.split(":")) for part in text.split()]
    for (a, b, r), nxt in zip(out, out[1:] + [None], strict=False):  # noqa: RUF005 - concatenation kept (operand type not guaranteed list)
        if not (0 <= a < b) or r < 1 or (nxt is not None and nxt[0] < b):
            raise ValueError(f"bad split list {text!r}: ranges must be disjoint and increasing")
        if r > b - a:
            raise ValueError(f"split {a}:{b} cannot hold {r} distinct root decisions")
    return out  # type: ignore[return-value]


@dataclass
class Task:
    key: str
    kind: str  # search | panel
    slots: int
    priority: tuple
    command: list[str]
    out: Path
    done_file: Path
    log: Path
    upload: str | None = None
    needs_actor: bool = False
    attempts: int = 0
    failed: bool = False
    proc: subprocess.Popen | None = None
    started: float = 0.0
    durations: list[float] = field(default_factory=list)
    env: dict = field(default_factory=dict)  # per-task environment overrides (SUCC_TASKS)

    def done(self) -> bool:
        return self.done_file.is_file()


class Scheduler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.work = Path(args.work)
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / "logs").mkdir(exist_ok=True)
        self.log_path = self.work / "sched.log"
        self.status_path = self.work / "status.json"
        self.durations_path = self.work / "durations.jsonl"
        self.tasks: list[Task] = []
        self.uploads: list[subprocess.Popen] = []
        self.last_launch = 0.0
        self.last_actor_try = 0.0
        self.idle_since: float | None = None
        self.stopping = False
        self.markers: set[str] = set()

    # -- logging ---------------------------------------------------------------------------
    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [sched] {message}"
        print(line, file=sys.stderr, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    # -- task construction -----------------------------------------------------------------
    def search_command(
        self, out: Path, world: int, first: int, turns: int, roots: int, held_out: bool, source: str
    ) -> list[str]:
        a = self.args
        command = [
            a.python,
            "-m",
            "anthropocene_rl.autonomous_v7.labor_search",
            "--normalizer-checkpoint",
            a.normalizer,
            "--worlds",
            str(world),
            "--source",
            source,
            "--turns",
            str(turns),
            "--first-turn",
            str(first),
            "--roots-per-world",
            str(roots),
            "--root-key",
            "0",
            "--candidates",
            str(a.candidates),
            "--horizons",
            a.horizons,
            "--keep",
            a.keep,
            "--reeval-streams",
            str(a.reeval_streams),
            "--control-fraction",
            str(a.control_fraction),
            "--plan-candidates",
            str(a.plan_candidates),
            "--plan-steps",
            str(a.plan_steps),
            "--plan-fraction",
            str(a.plan_fraction),
            "--personality",
            a.personality,
            "--personality-key",
            str(a.personality_key),
            "--players",
            str(a.players),
            "--width",
            str(a.width),
            "--height",
            str(a.height),
            "--separation",
            str(a.separation),
            "--jobs",
            "1",
            "--resume",
            "--out",
            str(out),
        ]
        if source == "actor":
            command += ["--actor-checkpoint", a.actor, "--actor-samples", str(a.actor_samples)]
        if held_out:
            command += ["--held-out-test"]
        command += a.search_extra.split()
        return command

    def build(self) -> None:
        a = self.args
        fit_worlds = seeds(a.fit_worlds)
        reserved_worlds = seeds(a.reserved_worlds)
        search_root = self.work / "search"
        stages = [("reserved-h12", reserved_worlds, splits(a.reserved_splits), True, "h12", 0)]
        stages.append(("fit-h12", fit_worlds, splits(a.fit_splits), False, "h12", 2))
        if a.actor_roots:
            stages.append(
                (
                    "reserved-actor",
                    reserved_worlds,
                    splits(a.actor_reserved_splits),
                    True,
                    "actor",
                    3,
                )
            )
            stages.append(("fit-actor", fit_worlds, splits(a.actor_fit_splits), False, "actor", 3))
        for stage, worlds, parts, held_out, source, rank in stages:
            for w_index, world in enumerate(worlds):
                for s_index, (first, turns, roots) in enumerate(parts):
                    rel = f"{stage}/t{first:02d}-{turns:02d}/w{world}"
                    out = search_root / rel
                    passes = len(parts)
                    self.tasks.append(
                        Task(
                            key=f"search:{rel}",
                            kind="search",
                            slots=1,
                            # Each pass gives every world one stratum (rotated by world index).
                            priority=(rank, (s_index + w_index) % passes, w_index, s_index),
                            command=self.search_command(
                                out, world, first, turns, roots, held_out, source
                            ),
                            out=out,
                            done_file=out / "search_report.json",
                            log=self.work
                            / "logs"
                            / f"search-{stage}-t{first:02d}-{turns:02d}-w{world}.log",
                            upload=f"{a.gcs}/search/{rel}",
                            needs_actor=source == "actor",
                        )
                    )
        if a.panel:
            panel_dir = self.work / "panel"
            for w_index, world in enumerate(seeds(a.panel_worlds)):
                if world in range(3000, 3060):
                    raise ValueError("sealed worlds 3000-3059 are never evaluated")
                out = panel_dir / f"world-{world}.json"
                self.tasks.append(
                    Task(
                        key=f"panel:{world}",
                        kind="panel",
                        slots=int(a.panel_jobs),
                        priority=(1, 0, w_index, 0),
                        command=[
                            a.python,
                            "-m",
                            "anthropocene_rl.autonomous_v7.labor_pretrain_eval",
                            "autonomous",
                            "--checkpoint",
                            a.actor,
                            "--worlds",
                            str(world),
                            "--turns",
                            str(a.panel_turns),
                            "--modes",
                            a.panel_modes,
                            "--players",
                            str(a.players),
                            "--width",
                            str(a.width),
                            "--height",
                            str(a.height),
                            "--separation",
                            str(a.separation),
                            "--jobs",
                            str(a.panel_jobs),
                            "--out",
                            str(out),
                        ],
                        out=out,
                        done_file=out,
                        log=self.work / "logs" / f"panel-{world}.log",
                        upload=None,
                        needs_actor=True,
                    )
                )
        self.apply_successor_tasks()
        self.tasks.sort(key=lambda t: t.priority)
        self.log(
            f"built {len(self.tasks)} tasks: "
            + ", ".join(
                f"{k}={sum(1 for t in self.tasks if t.key.startswith(k))}"
                for k in (
                    "search:reserved-h12",
                    "search:fit-h12",
                    "search:reserved-actor",
                    "search:fit-actor",
                    "panel",
                )
            )
        )

    def apply_successor_tasks(self) -> None:
        """Tasks named in WORK/SUCC_TASKS export successor rows (payload v2) using the code
        whose src is in WORK/SUCC_PYTHONPATH (4 Nov 2026, approved option A)."""
        listing = self.work / "SUCC_TASKS"
        if not listing.is_file():
            return
        keys = {line.strip() for line in listing.read_text().splitlines() if line.strip()}
        src_dir = (self.work / "SUCC_PYTHONPATH").read_text().strip()
        search_py = Path(src_dir) / "anthropocene_rl" / "autonomous_v7" / "labor_search.py"
        if "--successor-rows" not in search_py.read_text():
            raise SystemExit(f"{search_py} lacks --successor-rows")
        hit = 0
        for task in self.tasks:
            if task.key in keys:
                task.command = task.command + ["--successor-rows"]  # noqa: RUF005 - concatenation kept (operand type not guaranteed list)
                task.env = {"PYTHONPATH": src_dir}
                hit += 1
        self.log(f"successor rows on {hit}/{len(keys)} listed tasks, PYTHONPATH={src_dir}")

    # -- actor -----------------------------------------------------------------------------
    def actor_ready(self) -> bool:
        actor = Path(self.args.actor)
        if actor.is_file():
            return True
        if not self.args.actor_gcs or time.time() - self.last_actor_try < 600:
            return False
        self.last_actor_try = time.time()
        try:
            done = subprocess.run(
                ["gcloud", "storage", "ls", f"{self.args.actor_gcs}/DONE"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=120,
            )
            if done.returncode:
                return False
            actor.parent.mkdir(parents=True, exist_ok=True)
            for name in (actor.name, actor.name + ".sha256"):
                fetched = subprocess.run(
                    [
                        "gcloud",
                        "storage",
                        "cp",
                        f"{self.args.actor_gcs}/{name}",
                        str(actor.parent / (name + ".download")),
                    ],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=900,
                )
                if fetched.returncode:
                    self.log(f"actor fetch failed: {name}")
                    return False
        except subprocess.TimeoutExpired:
            self.log("actor fetch timed out; retrying later")
            return False
        expected = (actor.parent / (actor.name + ".sha256.download")).read_text().split()[0]
        digest = hashlib.sha256(
            (actor.parent / (actor.name + ".download")).read_bytes()
        ).hexdigest()
        if digest != expected:
            self.log(f"actor sha256 mismatch {digest[:16]} != {expected[:16]}; not used")
            return False
        os.replace(actor.parent / (actor.name + ".download"), actor)
        os.replace(actor.parent / (actor.name + ".sha256.download"), Path(str(actor) + ".sha256"))
        self.log(f"actor fetched {actor} sha256 {digest[:16]}")
        return True

    # -- process management ----------------------------------------------------------------
    def used_slots(self) -> int:
        return sum(t.slots for t in self.tasks if t.proc is not None)

    def mem_available_gb(self) -> float:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
        return 0.0

    def launch(self, task: Task) -> None:
        task.attempts += 1
        task.log.parent.mkdir(parents=True, exist_ok=True)
        handle = open(task.log, "a", encoding="utf-8")  # noqa: SIM115 - file handle lifetime kept as written
        handle.write(
            f"\n# attempt {task.attempts} {time.strftime('%FT%TZ', time.gmtime())}\n$ "
            + " ".join(task.command)
            + "\n"
        )
        handle.flush()
        child_env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        child_env.update(task.env)
        task.proc = subprocess.Popen(
            task.command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=self.args.code,
            env=child_env,
            start_new_session=True,
        )
        handle.close()
        task.started = time.time()
        self.last_launch = task.started
        self.log(
            f"launch {task.key} attempt {task.attempts} pid {task.proc.pid} "
            f"slots {self.used_slots()}/{self.args.max_slots} mem {self.mem_available_gb():.0f} GB"
        )

    def reap(self) -> None:
        for task in self.tasks:
            if task.proc is None:
                continue
            rc = task.proc.poll()
            if rc is None:
                continue
            seconds = time.time() - task.started
            task.proc = None
            if rc == 0 and task.done():
                task.durations.append(seconds)
                with open(self.durations_path, "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"task": task.key, "seconds": seconds, "attempt": task.attempts})
                        + "\n"
                    )
                self.log(f"done {task.key} in {seconds / 3600:.2f} h")
                if task.upload:
                    self.upload(task.out, task.upload)
            else:
                self.log(f"exit {task.key} rc={rc} after {seconds / 3600:.2f} h (see {task.log})")
                if task.attempts >= self.args.max_attempts:
                    task.failed = True
                    (self.work / "FAILED").mkdir(exist_ok=True)
                    (
                        self.work / "FAILED" / task.key.replace("/", "_").replace(":", "_")
                    ).write_text(f"rc={rc}\n")
                    self.log(f"FAILED {task.key} after {task.attempts} attempts")

    def upload(self, source: Path, dest: str) -> None:
        self.uploads = [p for p in self.uploads if p.poll() is None]
        self.uploads.append(
            subprocess.Popen(
                [
                    "timeout",
                    "3600",
                    "gcloud",
                    "storage",
                    "rsync",
                    "-r",
                    "--no-user-output-enabled",
                    str(source),
                    dest,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )

    def terminate(self) -> None:
        for task in self.tasks:
            if task.proc is not None:
                try:  # noqa: SIM105 - explicit try/except kept verbatim
                    os.killpg(task.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    # -- milestones and status -------------------------------------------------------------
    def milestone(self, name: str, condition: bool) -> None:
        if condition and name not in self.markers:
            self.markers.add(name)
            path = self.work / name
            path.write_text(time.strftime("%FT%TZ\n", time.gmtime()))
            self.log(f"milestone {name}")
            self.upload_file(path, f"{self.args.gcs}/{name}")
            if name == "PANEL_DONE" and self.args.panel_report:
                self.panel_report()

    def upload_file(self, path: Path, dest: str) -> None:
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", "--no-user-output-enabled", str(path), dest],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            self.log(f"upload timed out: {path}")

    def panel_report(self) -> None:
        out = self.work / "panel-report.json"
        try:
            result = subprocess.run(
                [
                    self.args.python,
                    self.args.panel_report,
                    "--panel",
                    str(self.work / "panel"),
                    "--modes",
                    self.args.panel_modes,
                    "--out",
                    str(out),
                ],
                cwd=self.args.code,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            self.log("panel report timed out")
            return
        (self.work / "panel-report.txt").write_text(result.stdout + result.stderr)
        self.log(f"panel report rc={result.returncode}: {result.stdout.strip().splitlines()[-1:]}")
        self.upload(self.work / "panel", f"{self.args.gcs}/panel")
        for name in ("panel-report.json", "panel-report.txt"):
            if (self.work / name).is_file():
                self.upload_file(self.work / name, f"{self.args.gcs}/{name}")

    @staticmethod
    def group(task: Task) -> str:
        return task.key.split("/")[0] if task.kind == "search" else task.kind

    def write_status(self) -> None:
        groups: dict[str, dict[str, int]] = {}
        seconds: dict[str, list[float]] = {}
        for task in self.tasks:
            group = self.group(task)
            g = groups.setdefault(group, {"done": 0, "running": 0, "failed": 0, "pending": 0})
            if task.done():
                g["done"] += 1
            elif task.proc is not None:
                g["running"] += 1
            elif task.failed:
                g["failed"] += 1
            else:
                g["pending"] += 1
            seconds.setdefault(group, []).extend(task.durations)
        mean = {k: (sum(v) / len(v) / 3600 if v else None) for k, v in seconds.items()}
        remaining = sum(
            (mean.get(self.group(t)) or 0.0) * t.slots
            for t in self.tasks
            if not t.done() and not t.failed
        )
        status = {
            "utc": time.strftime("%FT%TZ", time.gmtime()),
            "groups": groups,
            "mean_hours_per_task": mean,
            "slots": {"used": self.used_slots(), "max": self.args.max_slots},
            "mem_available_gb": round(self.mem_available_gb(), 1),
            "projected_remaining_wall_hours": None
            if not any(mean.values())
            else round(remaining / max(1, self.args.max_slots), 1),
            "markers": sorted(self.markers),
        }
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=1, sort_keys=True))
        os.replace(tmp, self.status_path)

    # -- main loop -------------------------------------------------------------------------
    def run(self) -> int:
        self.build()
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stopping", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stopping", True))
        last_status = 0.0
        while not self.stopping:
            self.reap()
            actor = self.actor_ready() if any(t.needs_actor for t in self.tasks) else False
            pending = [t for t in self.tasks if not t.done() and not t.failed and t.proc is None]
            runnable = [t for t in pending if actor or not t.needs_actor]
            for task in runnable:
                if time.time() - self.last_launch < self.args.launch_gap_s:
                    break
                if self.used_slots() + task.slots > self.args.max_slots:
                    if task.slots > 1:
                        continue  # a wide panel task waits; narrow search chunks may fit
                    break
                if self.mem_available_gb() < self.args.min_free_gb:
                    break
                self.launch(task)
            search = [t for t in self.tasks if t.kind == "search"]
            reserved = [t for t in search if t.key.startswith("search:reserved-h12/")]
            fit = [t for t in search if t.key.startswith("search:fit-h12/")]
            # Phase 1: the reserved worlds plus the first half of the fitting passes (8 of 16
            # roots per world with the default splits): enough for a first A fit and gate (d).
            half = max(1, len(splits(self.args.fit_splits)) // 2)
            first_pass = [t for t in fit if t.priority[1] < half]
            self.milestone("RESERVED_H12_DONE", bool(reserved) and all(t.done() for t in reserved))
            self.milestone(
                "PHASE1_DONE",
                bool(reserved) and all(t.done() for t in reserved + first_pass),
            )
            self.milestone("SEARCH_H12_DONE", bool(fit) and all(t.done() for t in reserved + fit))
            panel = [t for t in self.tasks if t.kind == "panel"]
            self.milestone("PANEL_DONE", bool(panel) and all(t.done() for t in panel))
            running = [t for t in self.tasks if t.proc is not None]
            if not running and not runnable:
                waiting = [t for t in pending if t.needs_actor and not actor]
                if not waiting:
                    break
                self.idle_since = self.idle_since or time.time()
                if time.time() - self.idle_since > self.args.actor_wait_h * 3600:
                    self.log(
                        f"actor not available after {self.args.actor_wait_h} h idle; "
                        f"{len(waiting)} actor-gated task(s) not run"
                    )
                    break
            else:
                self.idle_since = None
            if time.time() - last_status > 120:
                self.write_status()
                last_status = time.time()
            time.sleep(self.args.poll_s)
        if self.stopping:
            self.log("stop requested: terminating running tasks (they resume on the next run)")
            self.terminate()
        for upload in self.uploads:
            try:
                upload.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                upload.kill()
        self.write_status()
        failed = [t.key for t in self.tasks if t.failed]
        left = [t.key for t in self.tasks if not t.done() and not t.failed]
        self.log(f"exit: failed={len(failed)} not_done={len(left)}")
        return 1 if failed or self.stopping else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--work", default=env("WORK", str(Path.home() / "r5" / "scaleout")))
    p.add_argument("--code", default=env("CODE", ""))
    p.add_argument("--python", default=env("PY", sys.executable))
    p.add_argument("--normalizer", default=env("NORMALIZER", ""))
    p.add_argument(
        "--gcs",
        default=env("SCALEOUT_GCS", "gs://anthropocene-rl-datasets-508912/model9/r5/scaleout"),
    )
    p.add_argument("--fit-worlds", default=env("FIT_WORLDS", "90100-90153,90160-90165"))
    p.add_argument("--reserved-worlds", default=env("RESERVED_WORLDS", "90154-90159"))
    p.add_argument("--fit-splits", default=env("FIT_SPLITS", "0:25:8 25:50:8"))
    p.add_argument(
        "--reserved-splits", default=env("RESERVED_SPLITS", "0:13:8 13:25:8 25:38:8 38:50:8")
    )
    p.add_argument("--candidates", type=int, default=int(env("CANDIDATES", "14")))
    p.add_argument("--horizons", default=env("HORIZONS", "1,3"))
    p.add_argument("--keep", default=env("KEEP", "all,3"))
    p.add_argument("--reeval-streams", type=int, default=int(env("REEVAL_STREAMS", "2")))
    p.add_argument("--control-fraction", type=float, default=float(env("CONTROL_FRACTION", "0.25")))
    p.add_argument("--plan-candidates", type=int, default=int(env("PLAN_CANDIDATES", "2")))
    p.add_argument("--plan-steps", type=int, default=int(env("PLAN_STEPS", "3")))
    p.add_argument("--plan-fraction", type=float, default=float(env("PLAN_FRACTION", "0.25")))
    p.add_argument("--personality", default=env("PERSONALITY", "dirichlet:1.0"))
    p.add_argument("--personality-key", type=int, default=int(env("PERSONALITY_KEY", "4242")))
    p.add_argument("--players", type=int, default=int(env("PLAYERS", "16")))
    p.add_argument("--width", type=int, default=int(env("WIDTH", "128")))
    p.add_argument("--height", type=int, default=int(env("HEIGHT", "64")))
    p.add_argument("--separation", type=int, default=int(env("SEPARATION", "9")))
    p.add_argument("--search-extra", default=env("SEARCH_EXTRA", ""))
    p.add_argument(
        "--actor",
        default=env("ACTOR", str(Path.home() / "r5" / "scaleout" / "actor" / "actor-bc-pilot.pt")),
    )
    p.add_argument("--actor-gcs", default=env("ACTOR_GCS", ""))
    p.add_argument("--actor-wait-h", type=float, default=float(env("ACTOR_WAIT_H", "6")))
    p.add_argument("--actor-roots", type=int, default=int(env("ACTOR_ROOTS", "0")))
    p.add_argument("--actor-samples", type=int, default=int(env("ACTOR_SAMPLES", "4")))
    p.add_argument("--actor-fit-splits", default=env("ACTOR_FIT_SPLITS", "0:25:2 25:50:2"))
    p.add_argument(
        "--actor-reserved-splits", default=env("ACTOR_RESERVED_SPLITS", "0:25:4 25:50:4")
    )
    p.add_argument("--panel", type=int, default=int(env("PANEL", "1")))
    p.add_argument("--panel-worlds", default=env("PANEL_WORLDS", "2000-2023"))
    p.add_argument("--panel-turns", type=int, default=int(env("PANEL_TURNS", "50")))
    p.add_argument("--panel-modes", default=env("PANEL_MODES", "h12,deterministic,stochastic"))
    p.add_argument("--panel-jobs", type=int, default=int(env("PANEL_JOBS", "4")))
    p.add_argument("--panel-report", default=env("PANEL_REPORT", ""))
    p.add_argument("--max-slots", type=int, default=int(env("MAX_SLOTS", "16")))
    p.add_argument("--min-free-gb", type=float, default=float(env("MIN_FREE_GB", "20")))
    p.add_argument("--launch-gap-s", type=float, default=float(env("LAUNCH_GAP_S", "30")))
    p.add_argument("--max-attempts", type=int, default=int(env("MAX_ATTEMPTS", "2")))
    p.add_argument("--poll-s", type=float, default=float(env("POLL_S", "20")))
    p.add_argument("--dry-run", action="store_true", help="print the task list and exit")
    args = p.parse_args()
    if not args.code or not args.normalizer:
        p.error("CODE and NORMALIZER are required")
    scheduler = Scheduler(args)
    if args.dry_run:
        scheduler.build()
        for task in scheduler.tasks:
            print(task.priority, task.key, "slots", task.slots, "done" if task.done() else "")
        print(" ".join(scheduler.tasks[0].command))
        return 0
    return scheduler.run()


if __name__ == "__main__":
    raise SystemExit(main())
