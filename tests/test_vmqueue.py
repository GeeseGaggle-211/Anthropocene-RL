"""Tests for tools/vmqueue (elderq + runner) using a temporary home and fake checkout/execute commands."""  # noqa: E501 - docstring/prose line kept verbatim

import importlib.machinery
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

VMQ = Path(__file__).resolve().parents[1] / "tools" / "vmqueue"
sys.path.insert(0, str(VMQ))
import runner as rq  # noqa: E402

SHA1, SHA2 = "a" * 40, "b" * 40


def load_elderq():
    loader = importlib.machinery.SourceFileLoader("elderq_cli", str(VMQ / "elderq"))
    spec = importlib.util.spec_from_loader("elderq_cli", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_imports_resolve_to_this_worktree():
    assert Path(rq.__file__).resolve().parent == VMQ.resolve()


YAML = """\
# comment
name: demo
rl_sha: %s
sim_sha: "%s"
threads: 8
priority: 2
phases:
  - name: eval
    run: python -m x --out $RESULTS/eval.json   # not a comment
    done: eval.json
  - name: cmp
    run: |
      echo one
      echo two
    done: cmp/done.json
""" % (SHA1, SHA2)


def good(name="j", threads=4, priority=0, phases=None):
    return {
        "name": name,
        "rl_sha": SHA1,
        "sim_sha": SHA2,
        "threads": threads,
        "priority": priority,
        "phases": phases or [{"name": "p", "run": "go", "done": "p.done"}],
    }


# ------------------------------------------------------------ validation
def test_parse_and_validate_yaml():
    job = rq.validate_job(rq.parse_yaml(YAML))
    assert job["name"] == "demo" and job["threads"] == 8 and job["priority"] == 2
    assert job["sim_sha"] == SHA2
    assert job["phases"][0]["run"].endswith("# not a comment")
    assert job["phases"][1]["run"] == "echo one\necho two"
    assert job["phases"][1]["done"] == "cmp/done.json"


def test_json_job_file(tmp_path):
    f = tmp_path / "a.json"
    f.write_text(json.dumps(good("viajson")))
    assert rq.load_job_file(f)["name"] == "viajson"


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda d: d.update(rl_sha="abc123"), "rl_sha"),
        (lambda d: d.update(sim_sha="G" * 40), "sim_sha"),
        (lambda d: d.update(name="bad name!"), "name"),
        (lambda d: d.update(threads="many"), "threads"),
        (lambda d: d.update(threads=0), "threads"),
        (lambda d: d.pop("threads"), "threads"),
        (lambda d: d.update(phases=[]), "phases"),
        (lambda d: d.update(phases=[{"name": "p", "run": "x"}]), "done marker"),
        (lambda d: d.update(phases=[{"name": "p", "run": "x", "done": "../out"}]), "relative"),
        (lambda d: d.update(phases=[{"name": "p", "run": "x", "done": "/abs"}]), "relative"),
        (
            lambda d: d.update(
                phases=[
                    {"name": "p", "run": "x", "done": "a"},
                    {"name": "p", "run": "y", "done": "b"},
                ]
            ),
            "duplicate",
        ),
        (lambda d: d.update(extra=1), "unknown field"),
    ],
)
def test_validation_rejects(mutate, fragment):
    d = good()
    mutate(d)
    with pytest.raises(rq.JobError) as e:
        rq.validate_job(d)
    assert fragment in str(e.value)


def test_submit_validates_and_rejects_duplicates(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ELDERQ_HOME", str(tmp_path))
    cli = load_elderq()
    f = tmp_path / "job.yaml"
    f.write_text(YAML)
    assert cli.main(["submit", str(f)]) == 0
    assert (tmp_path / "queue" / "pending" / "demo.yaml").exists()
    assert cli.main(["submit", str(f)]) == 1  # name already taken
    bad = tmp_path / "bad.yaml"
    bad.write_text(YAML.replace(SHA1, "short"))
    assert cli.main(["submit", str(bad)]) == 2
    assert "rl_sha" in capsys.readouterr().err
    assert not (tmp_path / "queue" / "pending" / "bad.yaml").exists()


def test_help_states_rules(capsys):
    cli = load_elderq()
    assert cli.main(["help"]) == 0
    text = capsys.readouterr().out
    for needle in (
        "pushed commits",
        "Never edit a running job",
        "idempotent",
        "submit",
        "status",
        "logs",
        "cancel",
        "results",
    ):
        assert needle.lower() in text.lower()


# ------------------------------------------------------------ runner harness
class Fake:
    """Checkout creates PROVENANCE; execute interprets 'touch <rel>', 'fail', 'hold', 'nomark' commands."""  # noqa: E501 - docstring/prose line kept verbatim

    def __init__(self, home):
        self.home = Path(home)
        self.runs = []
        self.checkouts = []
        self.running = 0
        self.peak = 0
        self.names_running = set()
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.order = []

    def checkout(self, rl, sim, name, logfile):
        self.checkouts.append(name)
        d = self.home / "code" / name
        (d / "rl").mkdir(parents=True, exist_ok=True)
        (d / "PROVENANCE").write_text("rl=%s\nsim=%s\n" % (rl, sim))
        return 0

    def execute(self, run, cwd, env, logfile, should_cancel, on_start=None):
        with self.lock:
            self.runs.append((env["JOB"], run, env["OMP_NUM_THREADS"], env["RESULTS"], env["CODE"]))
            self.running += int(env["OMP_NUM_THREADS"])
            self.peak = max(self.peak, self.running)
            self.order.append(env["JOB"])
        try:
            if run.startswith("hold"):
                while not self.release.is_set():
                    if should_cancel():
                        return -15
                    time.sleep(0.01)
            if run.startswith("fail"):
                return 3
            if run.startswith("touch "):
                target = Path(env["RESULTS"]) / run.split()[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("ok")
            return 0
        finally:
            with self.lock:
                self.running -= int(env["OMP_NUM_THREADS"])


def make(tmp_path, cap=56, fake=None):
    fake = fake or Fake(tmp_path)
    r = rq.Runner(
        str(tmp_path),
        cap=cap,
        checkout=fake.checkout,
        execute=fake.execute,
        poll=0.01,
        log=lambda m: None,
    )
    return r, fake


def submit(tmp_path, job):
    p = tmp_path / "queue" / "pending"
    p.mkdir(parents=True, exist_ok=True)
    (p / (job["name"] + ".json")).write_text(json.dumps(job))


def status(tmp_path, name):
    return json.loads((tmp_path / "queue" / "jobs" / name / "status.json").read_text())


def phases(*names, run="touch {n}.done"):
    return [{"name": n, "run": run.format(n=n), "done": n + ".done"} for n in names]


# ------------------------------------------------------------ tests
def test_runs_phases_and_writes_status(tmp_path):
    r, fake = make(tmp_path)
    submit(tmp_path, good("a", threads=6, phases=phases("one", "two")))
    assert r.drain(10)
    st = status(tmp_path, "a")
    assert st["state"] == "done"
    assert [p["state"] for p in st["phases"]] == ["done", "done"]
    assert all(p["exit_code"] == 0 and p["started"] and p["ended"] for p in st["phases"])
    assert "rl=" + SHA1 in st["provenance"]
    assert st["started"] and st["ended"] and st["threads"] == 6
    job, run, omp, results, code = fake.runs[0]
    assert (
        omp == "6"
        and results == str(tmp_path / "results" / "a")
        and code == str(tmp_path / "code" / "a")
    )
    assert (tmp_path / "queue" / "jobs" / "a" / "logs" / "one.log").exists()
    assert [x[1] for x in fake.runs] == ["touch one.done", "touch two.done"]
    assert not (tmp_path / "queue" / "pending" / "a.json").exists()


def test_priority_order_and_thread_cap(tmp_path):
    r, fake = make(tmp_path, cap=10)
    for name, th, pr in [("low", 5, 0), ("high", 6, 9), ("mid", 5, 5)]:
        submit(
            tmp_path,
            good(
                name,
                threads=th,
                priority=pr,
                phases=[{"name": "p", "run": "hold", "done": "p.done"}],
            ),
        )
    r.tick()
    time.sleep(0.2)
    # strict priority: high (6) starts; mid (5) does not fit with it (11 > 10) and low must not jump
    # ahead.
    assert set(r.active) == {"high"}
    assert status(tmp_path, "high")["state"] == "running"
    assert not (tmp_path / "queue" / "jobs" / "mid").exists()
    fake.release.set()
    # let held phases finish without a marker -> failures are fine; we only check ordering and the
    # cap
    assert r.drain(10)
    assert fake.order[0] == "high" and sorted(fake.order[1:]) == [
        "low",
        "mid",
    ]  # 5+5 fit together afterwards
    assert fake.peak <= 10


def test_cap_never_exceeded_with_many_jobs(tmp_path):
    r, fake = make(tmp_path, cap=12)
    for i in range(6):
        submit(tmp_path, good("j%d" % i, threads=5, phases=phases("p")))
    assert r.drain(20)
    assert fake.peak <= 12
    assert all(status(tmp_path, "j%d" % i)["state"] == "done" for i in range(6))


def test_cap_from_config_file(tmp_path):
    r, _ = make(tmp_path, cap=56)
    assert r.cap() == 56
    (tmp_path / "queue" / "config.json").write_text('{"cap": 3}')
    assert r.cap() == 3
    submit(tmp_path, good("toobig", threads=4))
    r.drain(5)
    st = status(tmp_path, "toobig")
    assert st["state"] == "failed" and "exceeds" in st["error"]


def test_skips_phases_with_done_markers(tmp_path):
    r, fake = make(tmp_path)
    res = tmp_path / "results" / "s"
    res.mkdir(parents=True)
    (res / "one.done").write_text("x")
    submit(tmp_path, good("s", phases=phases("one", "two")))
    assert r.drain(10)
    st = status(tmp_path, "s")
    assert [p["state"] for p in st["phases"]] == ["skipped", "done"]
    assert [x[1] for x in fake.runs] == ["touch two.done"]


def test_resume_after_simulated_restart(tmp_path):
    # First runner: phase one done, phase two "interrupted" (process dies mid-run, status left as
    # running).
    fake = Fake(tmp_path)
    r1, _ = make(tmp_path, fake=fake)
    submit(tmp_path, good("res", phases=phases("one", "two", "three")))
    orig = fake.execute

    def crash_on_two(run, cwd, env, logfile, should_cancel, on_start=None):
        if run == "touch two.done":
            raise SystemExit  # simulates the VM dying: thread ends without finishing status
        return orig(run, cwd, env, logfile, should_cancel, on_start)

    fake.execute = crash_on_two
    r1.execute = crash_on_two
    r1.tick()
    for t, _ in list(r1.active.values()):
        t.join(5)
    st = status(tmp_path, "res")
    assert st["state"] == "running"  # stale: nobody is running it
    assert [p["state"] for p in st["phases"]] == ["done", "running", "pending"]
    # New runner process after the restart.
    fake2 = Fake(tmp_path)
    r2, _ = make(tmp_path, fake=fake2)
    assert r2.drain(10)
    st = status(tmp_path, "res")
    assert st["state"] == "done"
    assert [x[1] for x in fake2.runs] == ["touch two.done", "touch three.done"]  # one not repeated
    assert fake2.checkouts == []  # existing checkout reused
    assert st["phases"][1]["attempts"] == 2


def test_failure_stops_job_but_others_continue(tmp_path):
    r, fake = make(tmp_path)
    submit(
        tmp_path,
        good(
            "bad",
            priority=5,
            phases=[
                {"name": "boom", "run": "fail", "done": "x"},
                {"name": "never", "run": "touch n", "done": "n"},
            ],
        ),
    )
    submit(tmp_path, good("fine", phases=phases("p")))
    assert r.drain(10)
    bad, fine = status(tmp_path, "bad"), status(tmp_path, "fine")
    assert bad["state"] == "failed" and bad["phases"][0]["exit_code"] == 3
    assert bad["phases"][0]["state"] == "failed" and bad["phases"][1]["state"] == "pending"
    assert "boom" in bad["error"]
    assert fine["state"] == "done"
    assert all(x[1] != "touch n" for x in fake.runs)


def test_missing_done_marker_is_failure(tmp_path):
    r, _ = make(tmp_path)
    submit(tmp_path, good("nm", phases=[{"name": "p", "run": "true", "done": "never.json"}]))
    r.drain(5)
    st = status(tmp_path, "nm")
    assert (
        st["state"] == "failed"
        and "done marker" in st["error"]
        and st["phases"][0]["exit_code"] == 0
    )


def test_checkout_failure(tmp_path):
    fake = Fake(tmp_path)
    r, _ = make(tmp_path, fake=fake)
    r.checkout = lambda *a: 1
    submit(tmp_path, good("co"))
    r.drain(5)
    assert status(tmp_path, "co")["state"] == "failed"
    assert "checkout" in status(tmp_path, "co")["error"]


def test_cancel_pending_and_running(tmp_path):
    r, fake = make(tmp_path, cap=4)
    submit(
        tmp_path, good("run", threads=4, phases=[{"name": "p", "run": "hold", "done": "p.done"}])
    )
    submit(tmp_path, good("wait", threads=4, phases=phases("p")))
    r.tick()
    time.sleep(0.2)
    paths = rq.Paths(str(tmp_path))
    assert "cancelled pending" in rq.cancel_job(paths, "wait")
    assert status(tmp_path, "wait")["state"] == "cancelled"
    assert "requested" in rq.cancel_job(paths, "run")
    assert r.drain(10)
    assert status(tmp_path, "run")["state"] == "cancelled"
    assert not any(x[0] == "wait" for x in fake.runs)


def test_status_json_and_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ELDERQ_HOME", str(tmp_path))
    r, _ = make(tmp_path)
    submit(tmp_path, good("done1", phases=phases("p")))
    submit(tmp_path, good("later", threads=60))  # exceeds the cap of 56 -> failed
    r.drain(5)
    submit(tmp_path, good("queued1"))  # stays pending (runner not ticked)
    cli = load_elderq()
    assert cli.main(["status", "--json"]) == 0
    rows = {s["name"]: s for s in json.loads(capsys.readouterr().out)}
    assert (
        rows["done1"]["state"] == "done"
        and rows["later"]["state"] == "failed"
        and rows["queued1"]["state"] == "pending"
    )
    for key in ("phases", "state", "started", "ended", "provenance", "threads", "priority"):
        assert key in rows["done1"]
    assert cli.main(["status", "done1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["phases"][0]["exit_code"] == 0
    assert cli.main(["status"]) == 0
    assert "done1" in capsys.readouterr().out
    assert cli.main(["logs", "done1", "p", "--tail", "5"]) == 0
    assert "phase p attempt 1" in capsys.readouterr().out
    assert cli.main(["results", "done1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["files"][0]["path"] == "p.done"
    assert cli.main(["status", "nope"]) == 1


def test_real_shell_execution(tmp_path):
    import os
    import shutil

    if os.name == "nt" or not shutil.which("bash"):
        pytest.skip("needs a POSIX bash")
    logf = tmp_path / "l.log"
    rc = rq.default_execute(
        'echo "$JOB:$OMP_NUM_THREADS" && exit 7',
        tmp_path,
        dict(os.environ, JOB="j", OMP_NUM_THREADS="4"),
        logf,
        lambda: False,
    )
    assert rc == 7 and "j:4" in logf.read_text()


def test_inline_comments_and_docs_example():
    data = rq.parse_yaml('a: 5   # c\nb: "x # y"  # c\nrun: echo # keep\n')
    assert data == {"a": "5", "b": "x # y", "run": "echo # keep"}
    doc = (Path(__file__).resolve().parents[1] / "docs" / "VM_QUEUE.md").read_text(encoding="utf-8")
    example = doc.split("```yaml\n")[1].split("```")[0]
    job = rq.validate_job(rq.parse_yaml(example))
    assert [p["name"] for p in job["phases"]] == ["eval", "compare-cap", "compare-all"]
    assert job["threads"] == 35 and "wait" in job["phases"][0]["run"]
    assert "elderq help" in doc and "pushed commits" in doc


class FakeSystem:
    def __init__(self, boot="boot-1"):
        self.boot, self.alive, self.killed = boot, {}, []

    def boot_id(self):
        return self.boot

    def start_time(self, pid):
        return self.alive.get(pid)

    def kill_group(self, pid):
        self.killed.append(pid)
        self.alive.pop(pid, None)


def stale_running_job(tmp_path, name, boot="boot-1", pid=4242, start=999):
    """A job whose runner died while phase 'one' was running (process identity recorded)."""
    r, _ = make(tmp_path)
    submit(tmp_path, good(name, phases=phases("one", "two")))
    r.drain(5)
    d = tmp_path / "queue" / "jobs" / name
    for f in (tmp_path / "results" / name).glob("*"):
        f.unlink()
    st = json.loads((d / "status.json").read_text())
    st.update(state="running", ended=None)
    st["phases"][0].update(state="running", pid=pid, start_time=start, boot_id=boot, attempts=1)
    st["phases"][1].update(state="pending")
    (d / "status.json").write_text(json.dumps(st))


def resume(tmp_path, system):
    fake = Fake(tmp_path)
    r = rq.Runner(
        str(tmp_path),
        cap=56,
        checkout=fake.checkout,
        execute=fake.execute,
        poll=0.01,
        log=lambda m: None,
        system=system,
    )
    return r, fake


def test_live_phase_is_adopted_not_relaunched(tmp_path):
    stale_running_job(tmp_path, "ad")
    sysm = FakeSystem()
    sysm.alive[4242] = 999
    r, fake = resume(tmp_path, sysm)
    r.tick()
    time.sleep(0.2)
    assert fake.runs == [] and status(tmp_path, "ad")["state"] == "running"
    (tmp_path / "results" / "ad" / "one.done").write_text(
        "x"
    )  # the adopted process finishes its work
    sysm.alive.pop(4242)
    assert r.drain(10)
    st = status(tmp_path, "ad")
    assert st["state"] == "done" and st["phases"][0]["state"] == "done"
    assert [x[1] for x in fake.runs] == ["touch two.done"]


def test_adopted_exit_without_marker_fails(tmp_path):
    stale_running_job(tmp_path, "ad2")
    sysm = FakeSystem()
    r, fake = resume(tmp_path, sysm)
    sysm.alive[4242] = 999
    r.tick()
    time.sleep(0.1)
    sysm.alive.pop(4242)
    assert r.drain(10)
    assert status(tmp_path, "ad2")["state"] == "failed" and fake.runs == []


def test_reboot_relaunches(tmp_path):
    stale_running_job(tmp_path, "rb", boot="old-boot")
    sysm = FakeSystem("new-boot")
    sysm.alive[4242] = 999  # pid reused by something else after the reboot
    r, fake = resume(tmp_path, sysm)
    assert r.drain(10)
    assert [x[1] for x in fake.runs] == ["touch one.done", "touch two.done"]
    assert status(tmp_path, "rb")["state"] == "done"


def test_dead_pid_and_reused_pid_relaunch(tmp_path):
    stale_running_job(tmp_path, "dp")
    r, fake = resume(tmp_path, FakeSystem())  # pid not alive
    assert r.drain(10)
    assert [x[1] for x in fake.runs][0] == "touch one.done"
    stale_running_job(tmp_path, "rp")
    sysm = FakeSystem()
    sysm.alive[4242] = 1000  # same pid, different start time -> a different process
    r, fake = resume(tmp_path, sysm)
    assert r.drain(10)
    assert [x[1] for x in fake.runs][0] == "touch one.done"


def test_cancel_kills_adopted_group(tmp_path):
    stale_running_job(tmp_path, "ca")
    sysm = FakeSystem()
    sysm.alive[4242] = 999
    r, fake = resume(tmp_path, sysm)
    r.tick()
    time.sleep(0.1)
    rq.cancel_job(rq.Paths(str(tmp_path)), "ca")
    assert r.drain(10)
    assert sysm.killed == [4242] and fake.runs == []
    assert status(tmp_path, "ca")["state"] == "cancelled"


def test_start_time_parsing():
    import os

    if not os.path.exists("/proc/self/stat"):
        pytest.skip("needs /proc")
    s = rq.System()
    assert isinstance(s.start_time(os.getpid()), int) and s.boot_id()
