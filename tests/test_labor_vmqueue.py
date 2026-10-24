"""Tests for tools/vmqueue/labor_3x5_manifest.py and Spot-preemption resumption."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

VMQ = Path(__file__).resolve().parents[1] / "tools" / "vmqueue"
if str(VMQ) not in sys.path:
    sys.path.insert(0, str(VMQ))

import labor_3x5_manifest as lm  # noqa: E402
import runner as rq  # noqa: E402


def test_manifest_3x5_generation_count_and_naming() -> None:
    manifests = lm.build_all_3x5_manifests(iterations=2)
    assert len(manifests) == 15

    for arm in lm.ARM_NAMES:
        for seed in lm.PAIRED_OPTIMIZER_SEEDS:
            name = lm.job_name_for(arm, seed)
            assert name in manifests
            job = manifests[name]
            assert job["name"] == name
            assert rq.NAME_RE.match(name)
            assert job["threads"] == 18
            assert rq.SHA_RE.match(job["rl_sha"])
            assert rq.SHA_RE.match(job["sim_sha"])
            # Phase 00 + 0a + 0b + 2 iterations * 5 sub-stages = 13 phases
            assert len(job["phases"]) == 13


def test_manifest_validation_under_elderq_runner() -> None:
    manifests = lm.build_all_3x5_manifests(iterations=3)
    for name, job in manifests.items():
        validated = rq.validate_job(job)
        assert validated["name"] == name
        assert len(validated["phases"]) == 18
        for phase in validated["phases"]:
            assert rq.NAME_RE.match(phase["name"])
            assert "anthropocene_rl.autonomous_v7" in phase["run"]
            assert not phase["done"].startswith("/")
            assert not phase["done"].startswith("\\")
            assert ".." not in Path(phase["done"]).parts


def test_yaml_roundtrip_with_elderq_parser() -> None:
    job = lm.build_job_dict("labor-arm1-cold-rewind", 170, iterations=2)
    yaml_text = lm.dump_job_yaml(job)

    parsed = rq.parse_yaml(yaml_text)
    validated = rq.validate_job(parsed)

    assert validated["name"] == job["name"]
    assert validated["rl_sha"] == job["rl_sha"]
    assert validated["sim_sha"] == job["sim_sha"]
    assert validated["threads"] == job["threads"]
    assert validated["priority"] == job["priority"]
    assert len(validated["phases"]) == len(job["phases"])

    for orig, roundtrip in zip(job["phases"], validated["phases"], strict=True):
        assert orig["name"] == roundtrip["name"]
        assert orig["run"] == roundtrip["run"]
        assert orig["done"] == roundtrip["done"]


def test_json_roundtrip() -> None:
    job = lm.build_job_dict("labor-arm2-bc-baseline", 171, iterations=2)
    json_text = lm.dump_job_json(job)

    parsed = json.loads(json_text)
    validated = rq.validate_job(parsed)
    assert validated == job


def test_arm_specific_command_specifications() -> None:
    arm1 = lm.build_job_dict("labor-arm1-cold-rewind", 170, iterations=1)
    arm2 = lm.build_job_dict("labor-arm2-bc-baseline", 170, iterations=1)
    arm3 = lm.build_job_dict("labor-arm3-bc-token-unsmoothed", 170, iterations=1)

    # Phase 00 normalizer
    p00_arm1 = next(p for p in arm1["phases"] if p["name"] == "phase00-normalizer")
    assert "fit-normalizer" in p00_arm1["run"]
    assert "--allow-vm-scale" in p00_arm1["run"]
    assert p00_arm1["done"] == "bounded-normalizer.pt"

    # Phase 0a initialization uses $RESULTS/bounded-normalizer.pt
    p0a_arm1 = next(p for p in arm1["phases"] if p["name"] == "phase0a-init")
    assert "--normalizer-checkpoint $RESULTS/bounded-normalizer.pt" in p0a_arm1["run"]

    # Phase 0b differences
    p0b_arm1 = next(p for p in arm1["phases"] if p["name"] == "phase0b-pretrain-critic")
    assert "pretrain-critic" in p0b_arm1["run"]
    assert "--epochs 50" in p0b_arm1["run"]
    assert "--allow-vm-scale" in p0b_arm1["run"]
    assert p0b_arm1["done"] == "checkpoint-000.pt"

    p0b_arm2 = next(p for p in arm2["phases"] if p["name"] == "phase0b-pretrain-bc")
    assert "pretrain-bc" in p0b_arm2["run"]
    assert "--epochs 30" in p0b_arm2["run"]
    assert "--inactive-role-weight 0.05" in p0b_arm2["run"]
    assert "--label-smoothing 1e-4" in p0b_arm2["run"]
    assert "--inverse-freq-power 0.0" in p0b_arm2["run"]
    assert "--allow-vm-scale" in p0b_arm2["run"]
    assert p0b_arm2["done"] == "checkpoint-000.pt"

    p0b_arm3 = next(p for p in arm3["phases"] if p["name"] == "phase0b-pretrain-bc")
    assert "pretrain-bc" in p0b_arm3["run"]
    assert "--epochs 100" in p0b_arm3["run"]
    assert "--inactive-role-weight 0.0" in p0b_arm3["run"]
    assert "--label-smoothing 1e-6" in p0b_arm3["run"]
    assert "--inverse-freq-power 0.5" in p0b_arm3["run"]
    assert "--allow-vm-scale" in p0b_arm3["run"]
    assert p0b_arm3["done"] == "checkpoint-000.pt"

    # Iteration collect differences
    c_arm1 = next(p for p in arm1["phases"] if p["name"] == "iter000-collect")
    assert "--max-self-destruct-rewinds 3" in c_arm1["run"]
    assert "--allow-vm-scale" in c_arm1["run"]
    assert c_arm1["done"] == "rollout-000.pt"

    c_arm2 = next(p for p in arm2["phases"] if p["name"] == "iter000-collect")
    assert "--max-self-destruct-rewinds" not in c_arm2["run"]
    assert "--allow-vm-scale" in c_arm2["run"]
    assert c_arm2["done"] == "rollout-000.pt"

    c_arm3 = next(p for p in arm3["phases"] if p["name"] == "iter000-collect")
    assert "--max-self-destruct-rewinds" not in c_arm3["run"]
    assert "--allow-vm-scale" in c_arm3["run"]
    assert c_arm3["done"] == "rollout-000.pt"

    # Train sub-stage critic-epochs 50
    for arm_dict in (arm1, arm2, arm3):
        tr = next(p for p in arm_dict["phases"] if p["name"] == "iter000-train")
        assert "--critic-epochs 50" in tr["run"]
        assert tr["done"] == "checkpoint-001.pt"


def test_confirmation_seed_denial() -> None:
    with pytest.raises(ValueError, match="sealed confirmation range 3000-3059"):
        lm.check_not_confirmation_seed(3005)

    with pytest.raises(ValueError, match="sealed confirmation range 3000-3059"):
        lm.build_job_dict("labor-arm1-cold-rewind", 3050)


def test_write_manifests_to_disk(tmp_path: Path) -> None:
    manifests = lm.build_all_3x5_manifests(iterations=1)
    written = lm.write_manifests(tmp_path, manifests, output_format="yaml")

    assert len(written) == 16  # 15 yaml + index.json
    index_file = tmp_path / "index.json"
    assert index_file.exists()
    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    assert len(index_data) == 15

    for item in index_data:
        file_path = tmp_path / item["file"]
        assert file_path.exists()
        loaded = rq.load_job_file(file_path)
        assert loaded["name"] == item["name"]


class ExecutionHarness:
    """Simulates phase execution by creating the phase's expected done file."""

    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.executed_phases: list[tuple[str, str]] = []
        self.lock = threading.Lock()
        self.crash_phase_name: str | None = None

    def checkout(self, rl: str, sim: str, name: str, logfile: Path) -> int:
        d = self.home / "code" / name
        (d / "rl").mkdir(parents=True, exist_ok=True)
        (d / "PROVENANCE").write_text(f"rl={rl}\nsim={sim}\n")
        return 0

    def execute(
        self,
        run: str,
        cwd: Path,
        env: dict[str, str],
        logfile: Path,
        should_cancel: Any,
        on_start: Any = None,
    ) -> int:
        job_name = env["JOB"]
        results_dir = Path(env["RESULTS"])
        with self.lock:
            # Extract out path from the run command
            parts = run.split()
            out_idx = parts.index("--out")
            out_rel = Path(parts[out_idx + 1].replace("$RESULTS/", "").replace("${RESULTS}/", ""))
            out_file = results_dir / out_rel
            out_file.parent.mkdir(parents=True, exist_ok=True)

            self.executed_phases.append((job_name, run))

            if self.crash_phase_name and self.crash_phase_name in run:
                # Simulate VM dying / thread termination without writing done file
                raise SystemExit

            out_file.write_text("simulated_stage_output")
            return 0


def test_spot_preemption_resumption_skips_completed_stages(tmp_path: Path) -> None:
    """Verify that after mid-run Spot preemption, completed sub-stages are skipped."""
    harness1 = ExecutionHarness(tmp_path)
    job = lm.build_job_dict("labor-arm1-cold-rewind", 170, iterations=1)

    # We will simulate a Spot preemption crash during iter000-train-anchor
    harness1.crash_phase_name = "train-anchor"

    r1 = rq.Runner(
        str(tmp_path),
        cap=56,
        checkout=harness1.checkout,
        execute=harness1.execute,
        poll=0.01,
        log=lambda m: None,
    )

    pending_dir = tmp_path / "queue" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    job_file = pending_dir / f"{job['name']}.yaml"
    job_file.write_text(lm.dump_job_yaml(job), encoding="utf-8")

    # Start runner 1
    r1.tick()
    for t, _ in list(r1.active.values()):
        t.join(5)

    status_file = tmp_path / "queue" / "jobs" / job["name"] / "status.json"
    st1 = json.loads(status_file.read_text(encoding="utf-8"))
    assert st1["state"] == "running"  # Interrupted mid-run by crash

    phases_st1 = {p["name"]: p["state"] for p in st1["phases"]}
    assert phases_st1["phase00-normalizer"] == "done"
    assert phases_st1["phase0a-init"] == "done"
    assert phases_st1["phase0b-pretrain-critic"] == "done"
    assert phases_st1["iter000-collect"] == "done"
    assert phases_st1["iter000-train-anchor"] == "running"  # Crashed here

    results_dir = tmp_path / "results" / job["name"]
    assert (results_dir / "bounded-normalizer.pt").exists()
    assert (results_dir / "checkpoint-000-raw.pt").exists()
    assert (results_dir / "checkpoint-000.pt").exists()
    assert (results_dir / "rollout-000.pt").exists()
    assert not (results_dir / "train-anchor-000.npz").exists()

    # Second runner starts on reboot
    harness2 = ExecutionHarness(tmp_path)  # No crash this time
    r2 = rq.Runner(
        str(tmp_path),
        cap=56,
        checkout=harness2.checkout,
        execute=harness2.execute,
        poll=0.01,
        log=lambda m: None,
    )

    assert r2.drain(10)

    st2 = json.loads(status_file.read_text(encoding="utf-8"))
    assert st2["state"] == "done"

    phases_st2 = {p["name"]: p["state"] for p in st2["phases"]}
    # All stages are done; completed stages in pass 1 stayed done/skipped and were not re-run
    assert phases_st2["phase00-normalizer"] in ("done", "skipped")
    assert phases_st2["phase0a-init"] in ("done", "skipped")
    assert phases_st2["phase0b-pretrain-critic"] in ("done", "skipped")
    assert phases_st2["iter000-collect"] in ("done", "skipped")
    assert phases_st2["iter000-train-anchor"] == "done"
    assert phases_st2["iter000-actor-gate"] == "done"
    assert phases_st2["iter000-calibration"] == "done"
    assert phases_st2["iter000-train"] == "done"

    # Verify that harness2 only executed the uncompleted phases!
    executed_in_pass2 = [run for _, run in harness2.executed_phases]
    assert len(executed_in_pass2) == 4
    assert any("train-anchor" in run for run in executed_in_pass2)
    assert any("actor-gate" in run for run in executed_in_pass2)
    assert any("labor_value_calibration" in run for run in executed_in_pass2)
    assert any("labor_training train" in run for run in executed_in_pass2)

    # And Phase 00, 0a, 0b, and collect were NEVER executed in pass 2
    assert not any("fit-normalizer" in run for run in executed_in_pass2)
    assert not any("initialize" in run for run in executed_in_pass2)
    assert not any("pretrain-critic" in run for run in executed_in_pass2)
    assert not any("labor_training collect" in run for run in executed_in_pass2)
