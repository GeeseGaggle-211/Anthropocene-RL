"""Bounded, discarded retained-PPO copies selected by technical v5 criteria."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from preflight_autonomous_v4_residual import sha
from probe_autonomous_state_baseline import TASKS
from run_autonomous_critic_horizon_v5 import health_extremes

CORE = ["workforce", "movement", "enabled", "control_0"]


def diagnostic_jobs(campaign):
    selected = {}

    def include(frame, tasks, reason):
        pin = frame["checkpoint"]
        job = selected.setdefault(pin["path"], dict(checkpoint=pin, tasks=[], reasons=[]))
        job["tasks"] = [t for t in TASKS if t in set(job["tasks"]) | set(tasks)]
        job["reasons"].append(reason)

    for cell in campaign["cells"]:
        if not cell["health"]:
            raise ValueError("Stopped candidate lacks a saved boundary")
        frame = cell["health"][-1]
        if cell["status"] == "completed":
            include(frame, CORE, f"final:{cell['id']}")
        else:
            failed = Path(frame["checkpoint"]["path"]).parent / f"{cell['id']}-failed.pt"
            if not failed.exists():
                raise ValueError("Actual stopped-state checkpoint missing")
            include(
                dict(checkpoint=dict(path=str(failed), sha256=sha(failed))),
                CORE,
                f"actual_stopped_state:{cell['id']}",
            )
    extremes = health_extremes(campaign["cells"])
    if extremes != campaign["selected_extremes"]:
        raise ValueError("Technical checkpoint selection changed")
    for pick in extremes:
        tasks = TASKS if pick["reason"] == "categorical_confidence" else CORE
        include(pick, tasks, f"{pick['variant']}:{pick['reason']}")
    if sum(len(j["tasks"]) for j in selected.values()) > 66:
        raise ValueError("Frozen retained-task cap exceeded")
    return list(selected.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign_path = args.run / "summary.json"
    campaign = json.loads(campaign_path.read_text())
    cfg = campaign["config"]
    assert campaign["version"] == "autonomous-critic-horizon-gate-v5"
    assert cfg["maximum_diagnostic_action_draws"] == 135168
    assert cfg["actor_diagnostic_seconds_per_checkpoint"] == 900
    assert cfg["actor_diagnostic_maximum_seconds"] == 4200
    probe = cfg["inputs"]["probe"]
    assert sha(probe["path"]) == probe["sha256"]
    jobs = diagnostic_jobs(campaign)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + 4200
    report = dict(
        version="autonomous-horizon-diagnostics-v5",
        campaign_sha256=sha(campaign_path),
        config_sha256=campaign["config_sha256"],
        wrapper_sha256=sha(__file__),
        probe_script_sha256=sha("scripts/probe_autonomous_state_baseline_v4.py"),
        baseline_helpers_sha256=sha("scripts/probe_autonomous_state_baseline.py"),
        probe_sha256=probe["sha256"],
        planned_tasks=sum(len(j["tasks"]) for j in jobs),
        maximum_reserved_draws=sum(len(j["tasks"]) for j in jobs) * 64 * 32,
        checkpoints=[],
        complete=False,
        passed=False,
    )

    def save():
        report["wall_seconds"] = time.monotonic() - started
        report["reserved_draws"] = sum(j["reserved_draws"] for j in report["checkpoints"])
        report["observed_draws"] = sum(j["observed_draws"] for j in report["checkpoints"])
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    for index, job in enumerate(jobs):
        pin = job["checkpoint"]
        assert sha(pin["path"]) == pin["sha256"]
        output = args.output / f"checkpoint-{index}.json"
        item = dict(
            **job,
            output=str(output),
            status="running",
            complete=False,
            passed=False,
            reserved_draws=len(job["tasks"]) * 64 * 32,
            observed_draws=0,
        )
        report["checkpoints"].append(item)
        save()  # Charge the full job before dispatch, including timeout/interruption.
        remaining = min(900, deadline - time.monotonic())
        if remaining <= 0:
            item["status"] = "campaign_timeout"
            save()
            break
        command = [
            sys.executable,
            "scripts/probe_autonomous_state_baseline_v4.py",
            "--checkpoint",
            pin["path"],
            "--probe",
            probe["path"],
            "--output",
            str(output),
            "--maximum-seconds",
            str(remaining),
            "--tasks",
            *job["tasks"],
        ]
        with output.with_suffix(".log").open("w") as log:
            try:
                process = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining
                )
                item["status"] = "completed" if process.returncode == 0 else "error"
            except subprocess.TimeoutExpired:
                item["status"] = "timeout"
        assert sha(pin["path"]) == pin["sha256"]
        try:
            data = json.loads(output.read_text()) if output.exists() else {}
        except json.JSONDecodeError:
            data = {}
            item["status"] = "malformed_partial"
        item["complete"] = (
            item["status"] == "completed"
            and "passed" in data
            and not data.get("partial", True)
            and set(data["tasks"]) == set(job["tasks"])
        )
        if item["complete"]:
            assert data["checkpoint_sha256"] == pin["sha256"]
            assert data["probe_sha256"] == probe["sha256"]
            item["passed"] = data["passed"]
        item["observed_draws"] = sum(r["draws"] for r in data.get("tasks", {}).values())
        item["original_unchanged"] = True
        assert item["observed_draws"] <= item["reserved_draws"]
        save()
        print(json.dumps(item), flush=True)
    report["complete"] = (
        campaign.get("complete", False)
        and len(report["checkpoints"]) == len(jobs)
        and all(j["complete"] for j in report["checkpoints"])
    )
    report["passed"] = report["complete"] and all(j["passed"] for j in report["checkpoints"])
    save()


if __name__ == "__main__":
    main()
