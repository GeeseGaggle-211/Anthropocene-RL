#!/usr/bin/env python3
"""VM job-queue runner and shared helpers (Python 3 standard library only).

Layout under HOME (override with ELDERQ_HOME):
  queue/pending/<name>.yaml|json   submitted, not yet started
  queue/jobs/<name>/job.json       normalized job, status.json, logs/<phase>.log
  queue/cancel/<name>              cancel request marker
  queue/config.json                optional {"cap": 56}, re-read every tick
  code/<name>/{rl,sim}             immutable checkout (git/checkout.sh)
  results/<name>/                  phase outputs and done markers
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CAP = 56
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TERMINAL = ("done", "failed", "cancelled")


class JobError(Exception):
    def __init__(self, problems):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


def now_iso(t=None):
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- minimal YAML
_KEY = re.compile(r"^([A-Za-z0-9_.-]+):(?:\s+(.*))?$")


def _is_item(txt):
    return txt == "-" or txt.startswith("- ")


def _scalar(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        if s[0] == '"':
            try:
                return json.loads(s)
            except ValueError:
                pass
        return s[1:-1]
    return s


_QUOTED = re.compile(r'''^("(?:[^"\\]|\\.)*"|'[^']*')\s*(?:#.*)?$''')


def _strip_comment(val):
    """Drop a trailing ' # comment' (never applied to `run`, whose commands may contain '#')."""
    m = _QUOTED.match(val)
    if m:
        return m.group(1)
    return re.split(r"\s+#", val, maxsplit=1)[0].strip()


class _Parser:
    """Subset: comment lines, `key: scalar`, nested mappings, lists of scalars or mappings, `|` blocks."""

    def __init__(self, text):
        self.lines = []
        for raw in text.replace("\t", "    ").splitlines():
            self.lines.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))
        self.i = 0

    def skip(self):
        while self.i < len(self.lines) and (not self.lines[self.i][1] or self.lines[self.i][1].startswith("#")):
            self.i += 1
        return self.i < len(self.lines)

    def node(self, min_indent=0):
        if not self.skip():
            return None
        ind, txt = self.lines[self.i]
        if ind < min_indent:
            return None
        return self.seq(ind) if _is_item(txt) else self.mapping(ind)

    def block_scalar(self, parent_indent):
        out, base = [], None
        while self.i < len(self.lines):
            ind, txt = self.lines[self.i]
            if txt and ind <= parent_indent:
                break
            if txt and base is None:
                base = ind
            out.append(" " * max(0, ind - (base or 0)) + txt if txt else "")
            self.i += 1
        return "\n".join(out).strip("\n")

    def mapping(self, ind):
        result = {}
        while self.skip():
            i, txt = self.lines[self.i]
            if i != ind or _is_item(txt):
                if i > ind:
                    raise JobError("bad indentation near: %s" % txt)
                break
            m = _KEY.match(txt)
            if not m:
                raise JobError("cannot parse line: %s" % txt)
            key, val = m.group(1), (m.group(2) or "").strip()
            if key in result:
                raise JobError("duplicate key: %s" % key)
            self.i += 1
            if val in ("|", ">"):
                result[key] = self.block_scalar(ind)
            elif val:
                result[key] = _scalar(val if key == "run" else _strip_comment(val))
            else:
                result[key] = None
                if self.skip():
                    ni, nt = self.lines[self.i]
                    if ni > ind or (ni == ind and _is_item(nt)):
                        result[key] = self.node(ind)
        return result

    def seq(self, ind):
        result = []
        while self.skip():
            i, txt = self.lines[self.i]
            if i != ind or not _is_item(txt):
                break
            rest = txt[1:].strip()
            if not rest:
                self.i += 1
                result.append(self.node(ind + 1))
            elif _KEY.match(rest):
                offset = len(txt) - len(rest)
                self.lines[self.i] = (ind + offset, rest)
                result.append(self.mapping(ind + offset))
            else:
                self.i += 1
                result.append(_scalar(rest))
        return result


def parse_yaml(text):
    p = _Parser(text)
    value = p.node(0)
    if p.skip():
        raise JobError("could not parse near: %s" % p.lines[p.i][1])
    return value if value is not None else {}


# ---------------------------------------------------------------- validation
def _int(v, field, problems, minimum):
    try:
        n = int(str(v).strip())
    except (TypeError, ValueError):
        problems.append("%s must be an integer (got %r)" % (field, v))
        return None
    if n < minimum:
        problems.append("%s must be >= %d" % (field, minimum))
    return n


def validate_job(data):
    """Return the normalized job dict or raise JobError listing every problem."""
    if not isinstance(data, dict):
        raise JobError("job file must be a mapping with keys name, rl_sha, sim_sha, threads, priority, phases")
    problems = []
    for k in data:
        if k not in ("name", "rl_sha", "sim_sha", "threads", "priority", "phases"):
            problems.append("unknown field: %s" % k)
    name = str(data.get("name") or "")
    if not NAME_RE.match(name):
        problems.append("name must be 1-64 chars of letters, digits, '_', '-', '.' and start with a letter or digit")
    for f in ("rl_sha", "sim_sha"):
        v = str(data.get(f) or "")
        if not SHA_RE.match(v):
            problems.append("%s must be a full 40-character lowercase hex SHA (got %r)" % (f, v))
    threads = None
    if data.get("threads") is None:
        problems.append("threads is required (integer thread budget)")
    else:
        threads = _int(data["threads"], "threads", problems, 1)
    priority = 0
    if data.get("priority") is not None:
        priority = _int(data["priority"], "priority", problems, -1000000) or 0
    phases = data.get("phases")
    norm = []
    if not isinstance(phases, list) or not phases:
        problems.append("phases must be a non-empty list of {name, run, done}")
    else:
        seen = set()
        for n, ph in enumerate(phases, 1):
            if not isinstance(ph, dict):
                problems.append("phase %d must be a mapping" % n)
                continue
            for k in ph:
                if k not in ("name", "run", "done"):
                    problems.append("phase %d: unknown field %s" % (n, k))
            pn = str(ph.get("name") or "")
            if not NAME_RE.match(pn):
                problems.append("phase %d: name must be letters, digits, '_', '-', '.'" % n)
            elif pn in seen:
                problems.append("phase %d: duplicate phase name %s" % (n, pn))
            elif pn == "checkout":
                problems.append("phase %d: 'checkout' is reserved" % n)
            seen.add(pn)
            run = ph.get("run")
            if not run or not str(run).strip():
                problems.append("phase %s: run command is required" % (pn or n))
            done = str(ph.get("done") or "")
            if not done:
                problems.append("phase %s: done marker is required" % (pn or n))
            elif done.startswith("/") or re.match(r"^[A-Za-z]:", done) or ".." in Path(done).parts:
                problems.append("phase %s: done must be a relative path inside the results dir" % (pn or n))
            norm.append({"name": pn, "run": str(run or ""), "done": done})
    if problems:
        raise JobError(problems)
    return {"name": name, "rl_sha": str(data["rl_sha"]), "sim_sha": str(data["sim_sha"]),
            "threads": threads, "priority": priority, "phases": norm}


def load_job_file(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except ValueError as e:
            raise JobError("invalid JSON: %s" % e)
    else:
        data = parse_yaml(text)
    return validate_job(data)


# ---------------------------------------------------------------- paths / state
def home_dir(home=None):
    return Path(home or os.environ.get("ELDERQ_HOME") or Path.home())


class Paths:
    def __init__(self, home=None):
        self.home = home_dir(home)
        self.queue = self.home / "queue"
        self.pending = self.queue / "pending"
        self.jobs = self.queue / "jobs"
        self.cancel = self.queue / "cancel"
        self.code = self.home / "code"
        self.results = self.home / "results"

    def job_dir(self, name):
        return self.jobs / name

    def ensure(self):
        for d in (self.pending, self.jobs, self.cancel, self.code, self.results):
            d.mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for attempt in range(50):  # Windows refuses to replace a file a reader has open; POSIX never loops
        try:
            os.replace(str(tmp), str(path))
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.01)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def pending_files(paths):
    if not paths.pending.is_dir():
        return []
    return sorted(p for p in paths.pending.iterdir() if p.is_file() and p.suffix.lower() in (".yaml", ".yml", ".json"))


def initial_status(job, submitted_ts, state, **extra):
    st = {"name": job["name"], "state": state, "threads": job["threads"], "priority": job["priority"],
          "rl_sha": job["rl_sha"], "sim_sha": job["sim_sha"], "submitted": now_iso(submitted_ts),
          "started": None, "ended": None, "provenance": None, "error": None,
          "phases": [{"name": p["name"], "state": "pending", "started": None, "ended": None,
                      "exit_code": None, "attempts": 0} for p in job["phases"]]}
    st.update(extra)
    return st


def list_jobs(paths):
    """Status dicts for every known job (pending ones synthesized)."""
    out = {}
    if paths.jobs.is_dir():
        for d in sorted(paths.jobs.iterdir()):
            st = read_json(d / "status.json")
            if st:
                out[st["name"]] = st
    for f in pending_files(paths):
        try:
            job = load_job_file(f)
        except (JobError, OSError) as e:
            out.setdefault(f.stem, {"name": f.stem, "state": "invalid", "error": str(e), "phases": [],
                                    "threads": None, "priority": None, "submitted": ""})
            continue
        if job["name"] not in out:
            out[job["name"]] = initial_status(job, f.stat().st_mtime, "pending")
    return sorted(out.values(), key=lambda s: (s.get("submitted") or "", s["name"]))


def cancel_job(paths, name):
    """Cancel a job. Returns a short message; raises JobError for unknown jobs."""
    paths.ensure()
    for f in pending_files(paths):
        try:
            job = load_job_file(f)
        except (JobError, OSError):
            job = None
        if f.stem == name or (job and job["name"] == name):
            if job:
                st = initial_status(job, f.stat().st_mtime, "cancelled", ended=now_iso())
                write_json(paths.job_dir(job["name"]) / "job.json", job)
                write_json(paths.job_dir(job["name"]) / "status.json", st)
            f.unlink()
            return "cancelled pending job %s" % name
    st = read_json(paths.job_dir(name) / "status.json")
    if not st:
        raise JobError("unknown job: %s" % name)
    if st["state"] in TERMINAL:
        return "job %s already %s" % (name, st["state"])
    (paths.cancel / name).write_text(now_iso(), encoding="utf-8")
    return "cancel requested for %s (the runner stops it within a few seconds)" % name


# ---------------------------------------------------------------- execution
def default_checkout(paths):
    script = paths.home / "git" / "checkout.sh"

    def run(rl_sha, sim_sha, name, logfile):
        with open(str(logfile), "ab") as lf:
            return subprocess.call(["bash", str(script), rl_sha, sim_sha, name], stdout=lf, stderr=subprocess.STDOUT)
    return run


class System:
    """Linux process identity helpers (injectable). Returns None where /proc is unavailable."""

    def boot_id(self):
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            return None

    def start_time(self, pid):
        """Field 22 of /proc/<pid>/stat (clock ticks since boot); None if the process is gone."""
        try:
            stat = Path("/proc/%d/stat" % pid).read_text()
            fields = stat[stat.rindex(")") + 2:].split()  # after 'pid (comm)'; state is field 3
            if fields[0] == "Z":
                return None
            return int(fields[19])
        except (OSError, ValueError, IndexError):
            return None

    def kill_group(self, pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except (OSError, AttributeError):
            return
        for _ in range(200):
            if self.start_time(pid) is None:
                return
            time.sleep(0.1)
        try:
            os.killpg(pid, signal.SIGKILL)
        except (OSError, AttributeError):
            pass


def default_execute(run, cwd, env, logfile, should_cancel, on_start=None):
    """Run a shell command in its own process group; return its exit code (-15 if cancelled).
    on_start(pgid) is called right after launch so the runner can record the process identity."""
    with open(str(logfile), "ab") as lf:
        proc = subprocess.Popen(["bash", "-c", run], cwd=str(cwd), env=env, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=(os.name != "nt"))
        if on_start:
            on_start(proc.pid)
        while True:
            try:
                return proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                if should_cancel():
                    try:
                        if os.name != "nt":
                            os.killpg(proc.pid, signal.SIGTERM)
                        else:
                            proc.terminate()
                    except OSError:
                        pass
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    return -15


class Runner:
    """Schedules jobs under a thread cap. `checkout` and `execute` are injectable for tests.

    checkout(rl_sha, sim_sha, name, logfile) -> exit code (must create ~/code/<name>/PROVENANCE)
    execute(run, cwd, env, logfile, should_cancel, on_start) -> exit code; on_start(pgid) records the process
    system: boot_id(), start_time(pid), kill_group(pid); a live phase of the same boot is adopted, not relaunched
    """

    def __init__(self, home=None, cap=None, checkout=None, execute=None, poll=5.0, log=None, system=None):
        self.system = system or System()
        self.paths = Paths(home)
        self.paths.ensure()
        self.default_cap = cap if cap is not None else int(os.environ.get("ELDERQ_CAP", DEFAULT_CAP))
        self.checkout = checkout or default_checkout(self.paths)
        self.execute = execute or default_execute
        self.poll = poll
        self.log = log or (lambda m: print("%s %s" % (now_iso(), m), flush=True))
        self.active = {}  # name -> (thread, threads)
        self.lock = threading.Lock()

    @staticmethod
    def cap_of(paths, default=None):
        default = default if default is not None else int(os.environ.get("ELDERQ_CAP", DEFAULT_CAP))
        cfg = read_json(paths.queue / "config.json") or {}
        try:
            return int(cfg.get("cap", default))
        except (TypeError, ValueError):
            return default

    def cap(self):
        return self.cap_of(self.paths, self.default_cap)

    def _quarantine(self, f, why):
        self.log("%s: %s; moved to pending/invalid" % (f.name, why))
        bad = self.paths.pending / "invalid"
        bad.mkdir(exist_ok=True)
        os.replace(str(f), str(bad / f.name))

    def candidates(self):
        """Startable jobs sorted by priority: resumable ('running' after a restart) plus pending files."""
        found = {}
        if self.paths.jobs.is_dir():
            for d in sorted(self.paths.jobs.iterdir()):
                st, job = read_json(d / "status.json"), read_json(d / "job.json")
                if st and job and st["state"] in ("running", "queued") and st["name"] not in self.active:
                    found[st["name"]] = (job, st["submitted"], None)
        for f in pending_files(self.paths):
            try:
                job = load_job_file(f)
            except (JobError, OSError) as e:
                self._quarantine(f, "invalid (%s)" % e)
                continue
            if job["name"] in found or (self.paths.job_dir(job["name"]) / "job.json").exists():
                self._quarantine(f, "duplicates an existing job name")
                continue
            found[job["name"]] = (job, now_iso(f.stat().st_mtime), f)
        return sorted(found.values(), key=lambda x: (-x[0]["priority"], x[1], x[0]["name"]))

    def tick(self):
        with self.lock:
            for name in [n for n, (t, _) in self.active.items() if not t.is_alive()]:
                del self.active[name]
            cap = self.cap()
            used = sum(th for _, th in self.active.values())
            for job, submitted, src in self.candidates():
                name = job["name"]
                if src is None and (self.paths.cancel / name).exists():
                    st = read_json(self.paths.job_dir(name) / "status.json")
                    st["state"], st["ended"] = "cancelled", now_iso()
                    write_json(self.paths.job_dir(name) / "status.json", st)
                    (self.paths.cancel / name).unlink()
                    continue
                if job["threads"] > cap:
                    self._fail_early(job, submitted, src, "thread budget %d exceeds runner cap %d" % (job["threads"], cap))
                    continue
                if used + job["threads"] > cap:
                    break  # strict priority order: smaller jobs never starve a larger, higher-priority one
                if src is not None:
                    write_json(self.paths.job_dir(name) / "job.json", job)
                    write_json(self.paths.job_dir(name) / "status.json",
                               initial_status(job, time.time(), "queued", submitted=submitted))
                    src.unlink()
                t = threading.Thread(target=self.run_job, args=(job,), name="job-" + name, daemon=True)
                self.active[name] = (t, job["threads"])
                used += job["threads"]
                self.log("start %s (threads %d, used %d/%d)" % (name, job["threads"], used, cap))
                t.start()

    def _fail_early(self, job, submitted, src, msg):
        d = self.paths.job_dir(job["name"])
        write_json(d / "job.json", job)
        write_json(d / "status.json", initial_status(job, time.time(), "failed", submitted=submitted,
                                                     error=msg, ended=now_iso()))
        if src is not None:
            src.unlink()
        self.log("failed %s: %s" % (job["name"], msg))

    def drain(self, timeout=600):
        """Run until nothing is active and nothing more can start (`once` mode and tests)."""
        end = time.time() + timeout
        while time.time() < end:
            self.tick()
            with self.lock:
                alive = any(t.is_alive() for t, _ in self.active.values())
            if not alive:
                self.tick()
                with self.lock:
                    if not any(t.is_alive() for t, _ in self.active.values()):
                        return True
            time.sleep(min(self.poll, 0.02))
        return False

    def serve(self):
        self.log("runner started, cap %d" % self.cap())
        while True:
            try:
                self.tick()
            except Exception as e:  # never let one bad file stop the runner
                self.log("tick error: %r" % (e,))
            time.sleep(self.poll)

    def _adoptable(self, ps):
        """True if this phase's recorded process is still alive in the same boot (runner-only restart)."""
        pid, st, boot = ps.get("pid"), ps.get("start_time"), ps.get("boot_id")
        if ps.get("state") != "running" or not pid or st is None or not boot:
            return False
        return boot == self.system.boot_id() and self.system.start_time(pid) == st

    def _wait_adopted(self, ps, cancelled):
        """Poll an adopted process group until it exits; kill its group on cancel. Returns -15 or None."""
        pid, st = ps["pid"], ps["start_time"]
        while self.system.start_time(pid) == st:
            if cancelled():
                self.system.kill_group(pid)
                return -15
            time.sleep(min(self.poll, 1.0))
        return None

    def run_job(self, job):
        name = job["name"]
        d = self.paths.job_dir(name)
        logs = d / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        stpath = d / "status.json"
        st = read_json(stpath)
        res = self.paths.results / name
        res.mkdir(parents=True, exist_ok=True)
        code = self.paths.code / name
        flag = self.paths.cancel / name

        def save():
            write_json(stpath, st)

        def cancelled():
            return flag.exists()

        def finish(state, error=None):
            st["state"], st["ended"] = state, now_iso()
            if error:
                st["error"] = error
            save()
            if flag.exists():
                flag.unlink()
            self.log("%s %s%s" % (state, name, (": " + error) if error else ""))

        try:
            st["state"] = "running"
            st["started"] = st.get("started") or now_iso()
            st["error"] = None
            save()
            if not (code / "PROVENANCE").exists():
                rc = self.checkout(job["rl_sha"], job["sim_sha"], name, logs / "checkout.log")
                if rc != 0 or not (code / "PROVENANCE").exists():
                    return finish("failed", "checkout failed (exit %s); see: elderq logs %s checkout" % (rc, name))
            try:
                st["provenance"] = (code / "PROVENANCE").read_text(encoding="utf-8", errors="replace")
            except OSError:
                st["provenance"] = None
            save()
            env = dict(os.environ)
            env.update({"OMP_NUM_THREADS": str(job["threads"]), "RESULTS": str(res), "CODE": str(code), "JOB": name})
            for spec, ps in zip(job["phases"], st["phases"]):
                if cancelled():
                    return finish("cancelled")
                if (res / spec["done"]).exists():
                    if ps["state"] != "done":
                        ps["state"] = "skipped"
                    save()
                    continue
                logfile = logs / (spec["name"] + ".log")
                if self._adoptable(ps):
                    self.log("adopting running phase %s of %s (pid %s)" % (spec["name"], name, ps["pid"]))
                    rc = self._wait_adopted(ps, cancelled)
                    if rc is None:  # exit code of an adopted process is unknowable; the done marker decides
                        rc = 0 if (res / spec["done"]).exists() else "unknown (adopted; no done marker)"
                else:
                    ps.update(state="running", started=now_iso(), ended=None, exit_code=None,
                              attempts=ps.get("attempts", 0) + 1, pid=None, start_time=None, boot_id=None)
                    save()
                    with open(str(logfile), "ab") as lf:
                        lf.write(("=== %s phase %s attempt %d\n" % (now_iso(), spec["name"], ps["attempts"])).encode())

                    def started(pid, ps=ps):
                        ps.update(pid=pid, start_time=self.system.start_time(pid), boot_id=self.system.boot_id())
                        save()
                    rc = self.execute(spec["run"], code / "rl", env, logfile, cancelled, started)
                ps["ended"], ps["exit_code"] = now_iso(), rc
                if cancelled():
                    ps["state"] = "cancelled"
                    return finish("cancelled")
                if rc != 0:
                    ps["state"] = "failed"
                    return finish("failed", "phase %s exited with code %s" % (spec["name"], rc))
                if not (res / spec["done"]).exists():
                    ps["state"] = "failed"
                    return finish("failed", "phase %s exited 0 but did not create done marker %s"
                                  % (spec["name"], spec["done"]))
                ps["state"] = "done"
                save()
            finish("done")
        except Exception as e:  # other jobs keep running whatever happens here
            finish("failed", "runner error: %r" % (e,))


def main(argv=None):
    ap = argparse.ArgumentParser(description="elderq job runner")
    ap.add_argument("mode", choices=["run", "once"], help="run = serve forever; once = drain the queue and exit")
    ap.add_argument("--home")
    ap.add_argument("--cap", type=int, help="default thread cap %d (queue/config.json overrides)" % DEFAULT_CAP)
    ap.add_argument("--poll", type=float, default=5.0)
    a = ap.parse_args(argv)
    r = Runner(a.home, a.cap, poll=a.poll)
    if a.mode == "once":
        return 0 if r.drain(timeout=10 ** 9) else 1
    r.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
