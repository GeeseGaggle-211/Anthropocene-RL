"""Execute the preregistered checkpoint union without extending its diagnostic budget."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from anthropocene_rl.autonomous_v4.continuation import select_checkpoints


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_complete(campaign):
    cfg = campaign["config"]
    results = campaign["seeds"]
    if cfg["seeds"] != [29, 43] or sorted(r["seed"] for r in results) != [29, 43]:
        raise ValueError("Both unique configured seeds must reach terminal summaries")
    allowed = {
        "completed",
        "recovery_probe_pause",
        "optimizer_stall_pause",
        "resource_growth_pause",
    }
    for result in results:
        amount = result["additional_transitions"]
        if (
            result["status"] not in allowed
            or not 0 <= amount <= cfg["additional_transitions_per_seed"]
        ):
            raise ValueError("Invalid terminal seed status or transition count")
        if result["status"] == "completed" and amount != cfg["additional_transitions_per_seed"]:
            raise ValueError("Completed seed did not reach its frozen budget")
    if sum(r["additional_transitions"] for r in results) != campaign["additional_transitions"]:
        raise ValueError("Terminal campaign transition accounting mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = json.loads((args.run / "summary.json").read_text())
    validate_complete(campaign)
    if (args.run / "failure.json").exists():
        raise ValueError("Native campaign has an unresolved failure")
    cfg = campaign["config"]
    assert cfg["version"] == "autonomous-collapse-continuation-v4"
    assert sha(args.probe) == cfg["probe_sha256"]
    jobs = []
    for result in campaign["seeds"]:
        seed = result["seed"]
        frames = json.loads((args.run / f"seed-{seed}-health.json").read_text())
        selected = select_checkpoints(frames)
        assert selected == result["diagnostic_checkpoints"]
        for pick in selected:
            boundary = pick["transitions"]
            checkpoint = args.run / f"seed-{seed}-step-{boundary}.pt"
            jobs.append(dict(seed=seed, boundary=boundary, path=checkpoint, selection=pick))
    assert 0 < len(jobs) <= 6
    assert len(jobs) * 13 * 64 * 32 <= cfg["maximum_diagnostic_draws"] == 159744
    assert cfg["diagnostic_seconds_per_checkpoint"] == 900
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for job in jobs:
        output = args.output / f"seed-{job['seed']}-step-{job['boundary']}.json"
        digest = sha(job["path"])
        command = [
            sys.executable,
            "scripts/probe_autonomous_state_baseline.py",
            "--checkpoint",
            str(job["path"]),
            "--probe",
            str(args.probe),
            "--variant",
            "v3",
            "--output",
            str(output),
        ]
        with output.with_suffix(".log").open("w") as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                status = "completed" if process.returncode == 0 else "error"
            except subprocess.TimeoutExpired:
                status = "timeout"
        assert sha(job["path"]) == digest
        try:
            result = json.loads(output.read_text()) if output.exists() else {}
        except json.JSONDecodeError:
            result = {}
            status = "timeout_malformed_partial" if status == "timeout" else "malformed_output"
        # A complete result omits partial, so recognize only its explicit aggregate verdict.
        complete = status == "completed" and "passed" in result and not result.get("partial", False)
        if complete:
            assert result["checkpoint_sha256"] == digest
            assert result["probe_sha256"] == cfg["probe_sha256"]
            assert len(result["tasks"]) == 13
        reports.append(
            dict(
                seed=job["seed"],
                boundary=job["boundary"],
                selection=job["selection"],
                checkpoint_sha256=digest,
                status=status,
                complete=complete,
                passed=complete and result["passed"],
                observed_draws=sum(t["draws"] for t in result.get("tasks", {}).values()),
                reserved_draws=13 * 64 * 32,
                original_unchanged=True,
            )
        )
        report = dict(
            version="autonomous-continuation-diagnostics-v4",
            wrapper_sha256=sha(Path(__file__)),
            probe_script_sha256=sha(Path("scripts/probe_autonomous_state_baseline.py")),
            campaign_sha256=sha(args.run / "summary.json"),
            probe_sha256=sha(args.probe),
            checkpoints=reports,
            reserved_draws=len(jobs) * 13 * 64 * 32,
            observed_draws=sum(r["observed_draws"] for r in reports),
            complete=len(reports) == len(jobs) and all(r["complete"] for r in reports),
            passed=len(reports) == len(jobs) and all(r["passed"] for r in reports),
        )
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(reports[-1]), flush=True)


if __name__ == "__main__":
    main()
