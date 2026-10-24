#!/usr/bin/env python3
"""Builds elderq job manifests for the 3-arm x 5-paired-seed actor-labor experiment.

Target workload: anthropocene-trainer-60 (60-vCPU Spot VM).
Arms:
  1. labor-arm1-cold-rewind:
     - Phase 0a (init): labor_training initialize
     - Phase 0b (pretrain-critic): labor_training pretrain-critic (--epochs 50)
     - Iterations: collect (--max-self-destruct-rewinds 3), train-anchor,
       actor-gate, calibration, train (--critic-epochs 50)
  2. labor-arm2-bc-baseline:
     - Phase 0a (init): labor_training initialize
     - Phase 0b (pretrain-bc): labor_training pretrain-bc (--epochs 30,
       --inactive-role-weight 0.05, --label-smoothing 1e-4, --inverse-freq-power 0.0)
     - Iterations: collect, train-anchor, actor-gate, calibration, train (--critic-epochs 50)
  3. labor-arm3-bc-token-unsmoothed:
     - Phase 0a (init): labor_training initialize
     - Phase 0b (pretrain-bc): labor_training pretrain-bc (--epochs 100,
       --inactive-role-weight 0.0, --label-smoothing 1e-6, --inverse-freq-power 0.5)
     - Iterations: collect, train-anchor, actor-gate, calibration, train (--critic-epochs 50)

Paired optimizer seeds: 170, 171, 172, 173, 174 (15 runs total).
Every sub-stage writes its own discrete done marker in $RESULTS so elderq (runner.py)
resumes interrupted jobs after Spot preemption without losing completed sub-stages.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Add tools/vmqueue to sys.path for runner validation if imported locally
VMQ_DIR = Path(__file__).resolve().parent
if str(VMQ_DIR) not in sys.path:
    sys.path.insert(0, str(VMQ_DIR))

try:
    import runner as rq
except ImportError:
    rq = None  # type: ignore[assignment]

ARM_NAMES: tuple[str, ...] = (
    "labor-arm1-cold-rewind",
    "labor-arm2-bc-baseline",
    "labor-arm3-bc-token-unsmoothed",
)

PAIRED_OPTIMIZER_SEEDS: tuple[int, ...] = (170, 171, 172, 173, 174)

CONFIRMATION_SEEDS: range = range(3000, 3060)

DEFAULT_RL_SHA = "36e8b66263cbce0897baa966013c3b6ad06a7d8f"
DEFAULT_SIM_SHA = "4c52da7ce8e0b3e6dc90aa9053d327a228e67e45"
DEFAULT_NORMALIZER = "$RESULTS/bounded-normalizer.pt"
DEFAULT_PYTHON_BIN = "~/m9-bench/venv/bin/python"
DEFAULT_ENV_PREFIX = (
    "PYTHONPATH=$CODE/rl/src "
    "ANTHROPOCENE_DOTNET=${ANTHROPOCENE_DOTNET:-/usr/bin/dotnet} "
)
DEFAULT_WORLD_SEED0 = 50000
DEFAULT_TRAIN_WORLDS = "50000"
DEFAULT_TRAIN_ANCHOR_WORLDS = "50000"
DEFAULT_GATE_WORLDS = "50001,50002"
DEFAULT_THREADS = 18
DEFAULT_PRIORITY = 0
DEFAULT_ITERATIONS = 5
DEFAULT_PERSONALITY = "balanced"


def check_not_confirmation_seed(seed: int, context: str = "seed") -> int:
    if seed in CONFIRMATION_SEEDS:
        raise ValueError(
            f"Seed {seed} in {context} falls within sealed confirmation range 3000-3059"
        )
    return seed


def check_seed_string(seed_str: str, context: str = "seed list") -> str:
    for part in seed_str.split(","):
        part = part.strip()
        if part:
            check_not_confirmation_seed(int(part), context)
    return seed_str


def personality_flags(personality: str, film_bound: float | None) -> str:
    """initialize flags for the declared labor personality distribution (docs/LABOR_PERSONALITY.md).

    Only ``initialize`` takes them: the distribution is stored in the checkpoint
    contract and every later phase reads it from there. The balanced default
    emits nothing, so existing manifests are byte-identical.
    """
    if personality == DEFAULT_PERSONALITY:
        flags = ""
    elif re.fullmatch(r"dirichlet:[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?", personality) and float(
        personality.partition(":")[2]
    ) > 0:
        flags = f"--personality {personality} "
    else:
        raise ValueError(
            f"labor personality must be 'balanced' or 'dirichlet:ALPHA', not {personality!r}"
        )
    if film_bound is not None:
        if not film_bound > 0:
            raise ValueError("FiLM bound must be positive")
        flags += f"--film-bound {float(film_bound)!r} "
    return flags


def build_phase(name: str, run_cmd: str, done_marker: str) -> dict[str, str]:
    done_clean = str(done_marker).strip()
    is_invalid = (
        done_clean.startswith("/")
        or bool(re.match(r"^[A-Za-z]:", done_clean))
        or ".." in Path(done_clean).parts
    )
    if is_invalid:
        raise ValueError(f"Done marker must be a relative path inside $RESULTS: {done_marker}")
    return {
        "name": name,
        "run": run_cmd.strip(),
        "done": done_clean,
    }


def build_arm_phases(
    arm: str,
    optimizer_seed: int,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    normalizer_checkpoint: str = DEFAULT_NORMALIZER,
    world_seed0: int = DEFAULT_WORLD_SEED0,
    train_worlds: str = DEFAULT_TRAIN_WORLDS,
    train_anchor_worlds: str = DEFAULT_TRAIN_ANCHOR_WORLDS,
    gate_worlds: str = DEFAULT_GATE_WORLDS,
    python_bin: str = DEFAULT_PYTHON_BIN,
    env_prefix: str = DEFAULT_ENV_PREFIX,
    personality: str = DEFAULT_PERSONALITY,
    film_bound: float | None = None,
) -> list[dict[str, str]]:
    """Build the ordered list of idempotent phases for an arm and optimizer seed."""
    if arm not in ARM_NAMES:
        raise ValueError(f"Unknown arm: {arm!r}. Must be one of {ARM_NAMES}")

    check_not_confirmation_seed(optimizer_seed, "optimizer seed")
    check_not_confirmation_seed(world_seed0, "world seed 0")
    check_seed_string(train_worlds, "train worlds")
    check_seed_string(train_anchor_worlds, "train anchor worlds")
    check_seed_string(gate_worlds, "gate worlds")

    phases: list[dict[str, str]] = []

    # Phase 00: Fit bounded normalizer (if using default $RESULTS/bounded-normalizer.pt)
    if normalizer_checkpoint == "$RESULTS/bounded-normalizer.pt":
        norm_cmd = (
            f"{env_prefix}{python_bin} -m "
            "anthropocene_rl.autonomous_v7.labor_training fit-normalizer "
            f"--worlds {train_worlds} "
            f"--turns 25 "
            f"--allow-vm-scale "
            f"--out $RESULTS/bounded-normalizer.pt"
        )
        phases.append(build_phase("phase00-normalizer", norm_cmd, "bounded-normalizer.pt"))

    # Phase 0a: Initialize
    init_cmd = (
        f"{env_prefix}{python_bin} -m anthropocene_rl.autonomous_v7.labor_training initialize "
        f"--normalizer-checkpoint {normalizer_checkpoint} "
        f"--world-seed {world_seed0} "
        f"--optimizer-seed {optimizer_seed} "
        f"{personality_flags(personality, film_bound)}"
        f"--out $RESULTS/checkpoint-000-raw.pt"
    )
    phases.append(build_phase("phase0a-init", init_cmd, "checkpoint-000-raw.pt"))

    # Phase 0b: Arm-specific pretraining
    if arm == "labor-arm1-cold-rewind":
        phase0b_cmd = (
            f"{env_prefix}{python_bin} -m "
            "anthropocene_rl.autonomous_v7.labor_training pretrain-critic "
            f"--checkpoint $RESULTS/checkpoint-000-raw.pt "
            f"--worlds {train_worlds} "
            f"--turns 25 "
            f"--epochs 50 "
            f"--allow-vm-scale "
            f"--out $RESULTS/checkpoint-000.pt"
        )
        phases.append(build_phase("phase0b-pretrain-critic", phase0b_cmd, "checkpoint-000.pt"))
    elif arm == "labor-arm2-bc-baseline":
        phase0b_cmd = (
            f"{env_prefix}{python_bin} -m anthropocene_rl.autonomous_v7.labor_training pretrain-bc "
            f"--checkpoint $RESULTS/checkpoint-000-raw.pt "
            f"--worlds {train_worlds} "
            f"--turns 25 "
            f"--epochs 30 "
            f"--inactive-role-weight 0.05 "
            f"--label-smoothing 1e-4 "
            f"--inverse-freq-power 0.0 "
            f"--allow-vm-scale "
            f"--out $RESULTS/checkpoint-000.pt"
        )
        phases.append(build_phase("phase0b-pretrain-bc", phase0b_cmd, "checkpoint-000.pt"))
    elif arm == "labor-arm3-bc-token-unsmoothed":
        phase0b_cmd = (
            f"{env_prefix}{python_bin} -m anthropocene_rl.autonomous_v7.labor_training pretrain-bc "
            f"--checkpoint $RESULTS/checkpoint-000-raw.pt "
            f"--worlds {train_worlds} "
            f"--turns 25 "
            f"--epochs 100 "
            f"--inactive-role-weight 0.0 "
            f"--label-smoothing 1e-6 "
            f"--inverse-freq-power 0.5 "
            f"--allow-vm-scale "
            f"--out $RESULTS/checkpoint-000.pt"
        )
        phases.append(build_phase("phase0b-pretrain-bc", phase0b_cmd, "checkpoint-000.pt"))

    # Iterations 0 .. iterations - 1
    for i in range(iterations):
        in_ckpt = f"checkpoint-{i:03d}.pt"
        out_rollout = f"rollout-{i:03d}.pt"
        out_anchor = f"train-anchor-{i:03d}.npz"
        out_gate = f"actor-gate-{i:03d}.npz"
        out_cal = f"calibration-{i:03d}.json"
        out_next_ckpt = f"checkpoint-{i+1:03d}.pt"

        # Deterministic seed bases disjoint between train anchor and gate
        continuation_seed_base = 60000 + (optimizer_seed % 1000) * 100 + i
        gate_continuation_seed_base = 70000 + (optimizer_seed % 1000) * 100 + i

        # Sub-stage 1: collect
        if arm == "labor-arm1-cold-rewind":
            collect_cmd = (
                f"{env_prefix}{python_bin} -m anthropocene_rl.autonomous_v7.labor_training collect "
                f"--checkpoint $RESULTS/{in_ckpt} "
                f"--worlds {train_worlds} "
                f"--turns 25 "
                f"--max-self-destruct-rewinds 3 "
                f"--allow-vm-scale "
                f"--out $RESULTS/{out_rollout}"
            )
        else:
            collect_cmd = (
                f"{env_prefix}{python_bin} -m anthropocene_rl.autonomous_v7.labor_training collect "
                f"--checkpoint $RESULTS/{in_ckpt} "
                f"--worlds {train_worlds} "
                f"--turns 25 "
                f"--allow-vm-scale "
                f"--out $RESULTS/{out_rollout}"
            )
        phases.append(build_phase(f"iter{i:03d}-collect", collect_cmd, out_rollout))

        # Sub-stage 2: train-anchor
        anchor_cmd = (
            f"{env_prefix}{python_bin} -m anthropocene_rl.autonomous_v7.labor_anchor_producer "
            f"--checkpoint $RESULTS/{in_ckpt} "
            f"--worlds {train_anchor_worlds} "
            f"--optimizer-seed {optimizer_seed} "
            f"--continuation-seed-base {continuation_seed_base} "
            f"--role train-anchor "
            f"--allow-vm-scale "
            f"--out $RESULTS/{out_anchor}"
        )
        phases.append(build_phase(f"iter{i:03d}-train-anchor", anchor_cmd, out_anchor))

        # Sub-stage 3: actor-gate
        gate_cmd = (
            f"{env_prefix}{python_bin} -m anthropocene_rl.autonomous_v7.labor_anchor_producer "
            f"--checkpoint $RESULTS/{in_ckpt} "
            f"--worlds {gate_worlds} "
            f"--optimizer-seed {optimizer_seed} "
            f"--continuation-seed-base {gate_continuation_seed_base} "
            f"--role actor-gate "
            f"--allow-vm-scale "
            f"--out $RESULTS/{out_gate}"
        )
        phases.append(build_phase(f"iter{i:03d}-actor-gate", gate_cmd, out_gate))

        # Sub-stage 4: calibration
        cal_cmd = (
            f"{env_prefix}{python_bin} -m anthropocene_rl.autonomous_v7.labor_value_calibration "
            f"--checkpoint $RESULTS/{in_ckpt} "
            f"--anchor $RESULTS/{out_gate} "
            f"--out $RESULTS/{out_cal}"
        )
        phases.append(build_phase(f"iter{i:03d}-calibration", cal_cmd, out_cal))

        # Sub-stage 5: train
        train_cmd = (
            f"{env_prefix}{python_bin} -m anthropocene_rl.autonomous_v7.labor_training train "
            f"--checkpoint $RESULTS/{in_ckpt} "
            f"--rollout $RESULTS/{out_rollout} "
            f"--anchor $RESULTS/{out_anchor} "
            f"--calibration-report $RESULTS/{out_cal} "
            f"--critic-epochs 50 "
            f"--out $RESULTS/{out_next_ckpt}"
        )
        phases.append(build_phase(f"iter{i:03d}-train", train_cmd, out_next_ckpt))

    return phases


def job_name_for(arm: str, optimizer_seed: int) -> str:
    return f"{arm}-s{optimizer_seed}"


def build_job_dict(
    arm: str,
    optimizer_seed: int,
    *,
    rl_sha: str = DEFAULT_RL_SHA,
    sim_sha: str = DEFAULT_SIM_SHA,
    threads: int = DEFAULT_THREADS,
    priority: int = DEFAULT_PRIORITY,
    iterations: int = DEFAULT_ITERATIONS,
    normalizer_checkpoint: str = DEFAULT_NORMALIZER,
    world_seed0: int = DEFAULT_WORLD_SEED0,
    train_worlds: str = DEFAULT_TRAIN_WORLDS,
    train_anchor_worlds: str = DEFAULT_TRAIN_ANCHOR_WORLDS,
    gate_worlds: str = DEFAULT_GATE_WORLDS,
    python_bin: str = DEFAULT_PYTHON_BIN,
    env_prefix: str = DEFAULT_ENV_PREFIX,
    personality: str = DEFAULT_PERSONALITY,
    film_bound: float | None = None,
) -> dict[str, Any]:
    name = job_name_for(arm, optimizer_seed)
    phases = build_arm_phases(
        arm=arm,
        optimizer_seed=optimizer_seed,
        iterations=iterations,
        normalizer_checkpoint=normalizer_checkpoint,
        world_seed0=world_seed0,
        train_worlds=train_worlds,
        train_anchor_worlds=train_anchor_worlds,
        gate_worlds=gate_worlds,
        python_bin=python_bin,
        env_prefix=env_prefix,
        personality=personality,
        film_bound=film_bound,
    )
    job: dict[str, Any] = {
        "name": name,
        "rl_sha": str(rl_sha).strip(),
        "sim_sha": str(sim_sha).strip(),
        "threads": int(threads),
        "priority": int(priority),
        "phases": phases,
    }
    if rq is not None:
        rq.validate_job(job)
    return job


def dump_job_yaml(job: Mapping[str, Any]) -> str:
    """Format job dictionary into clean, runner.parse_yaml-compatible YAML."""
    lines: list[str] = [
        f"name: {job['name']}",
        f"rl_sha: {job['rl_sha']}",
        f"sim_sha: {job['sim_sha']}",
        f"threads: {job['threads']}",
        f"priority: {job['priority']}",
        "phases:",
    ]
    for ph in job["phases"]:
        lines.append(f"  - name: {ph['name']}")
        run_cmd = ph["run"]
        if "\n" in run_cmd:
            lines.append("    run: |")
            for subline in run_cmd.splitlines():
                lines.append(f"      {subline}")
        else:
            lines.append(f"    run: {run_cmd}")
        lines.append(f"    done: {ph['done']}")
    return "\n".join(lines) + "\n"


def dump_job_json(job: Mapping[str, Any]) -> str:
    return json.dumps(job, indent=2) + "\n"


def build_all_3x5_manifests(
    *,
    arms: Sequence[str] = ARM_NAMES,
    optimizer_seeds: Sequence[int] = PAIRED_OPTIMIZER_SEEDS,
    rl_sha: str = DEFAULT_RL_SHA,
    sim_sha: str = DEFAULT_SIM_SHA,
    threads: int = DEFAULT_THREADS,
    priority: int = DEFAULT_PRIORITY,
    iterations: int = DEFAULT_ITERATIONS,
    normalizer_checkpoint: str = DEFAULT_NORMALIZER,
    world_seed0: int = DEFAULT_WORLD_SEED0,
    train_worlds: str = DEFAULT_TRAIN_WORLDS,
    train_anchor_worlds: str = DEFAULT_TRAIN_ANCHOR_WORLDS,
    gate_worlds: str = DEFAULT_GATE_WORLDS,
    python_bin: str = DEFAULT_PYTHON_BIN,
    env_prefix: str = DEFAULT_ENV_PREFIX,
    personality: str = DEFAULT_PERSONALITY,
    film_bound: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate all 3-arm x 5-seed (15 runs) job specifications."""
    manifests: dict[str, dict[str, Any]] = {}
    for arm in arms:
        for seed in optimizer_seeds:
            job = build_job_dict(
                arm=arm,
                optimizer_seed=seed,
                rl_sha=rl_sha,
                sim_sha=sim_sha,
                threads=threads,
                priority=priority,
                iterations=iterations,
                normalizer_checkpoint=normalizer_checkpoint,
                world_seed0=world_seed0,
                train_worlds=train_worlds,
                train_anchor_worlds=train_anchor_worlds,
                gate_worlds=gate_worlds,
                python_bin=python_bin,
                env_prefix=env_prefix,
                personality=personality,
                film_bound=film_bound,
            )
            manifests[job["name"]] = job
    return manifests


def write_manifests(
    out_dir: Path,
    manifests: Mapping[str, Mapping[str, Any]],
    output_format: str = "yaml",
) -> list[Path]:
    """Write job specifications to out_dir and return list of created file paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    ext = ".json" if output_format.lower() == "json" else ".yaml"

    summary: list[dict[str, Any]] = []

    for name, job in manifests.items():
        dest = out_dir / f"{name}{ext}"
        content = dump_job_json(job) if ext == ".json" else dump_job_yaml(job)
        dest.write_text(content, encoding="utf-8")
        created.append(dest)
        summary.append({
            "name": name,
            "file": dest.name,
            "phases_count": len(job["phases"]),
            "threads": job["threads"],
            "priority": job["priority"],
        })

    summary_file = out_dir / "index.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    created.append(summary_file)
    return created


def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build elderq job manifests for 3-arm x 5-seed actor-labor experiment"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("queue/pending"),
        help="Directory where generated job files will be placed (default: queue/pending)",
    )
    p.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Manifest file format (default: yaml)",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of training iterations (default: {DEFAULT_ITERATIONS})",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Thread budget per run (default: {DEFAULT_THREADS})",
    )
    p.add_argument(
        "--priority",
        type=int,
        default=DEFAULT_PRIORITY,
        help=f"Base priority for jobs (default: {DEFAULT_PRIORITY})",
    )
    p.add_argument(
        "--rl-sha",
        default=DEFAULT_RL_SHA,
        help=f"40-hex Git commit SHA for RL (default: {DEFAULT_RL_SHA})",
    )
    p.add_argument(
        "--sim-sha",
        default=DEFAULT_SIM_SHA,
        help=f"40-hex Git commit SHA for Simulator (default: {DEFAULT_SIM_SHA})",
    )
    p.add_argument(
        "--normalizer-checkpoint",
        default=DEFAULT_NORMALIZER,
        help="Path to normalizer checkpoint",
    )
    p.add_argument(
        "--world-seed0",
        type=int,
        default=DEFAULT_WORLD_SEED0,
        help=f"World seed for initialization (default: {DEFAULT_WORLD_SEED0})",
    )
    p.add_argument(
        "--train-worlds",
        default=DEFAULT_TRAIN_WORLDS,
        help=f"Comma-separated training worlds (default: {DEFAULT_TRAIN_WORLDS})",
    )
    p.add_argument(
        "--train-anchor-worlds",
        default=DEFAULT_TRAIN_ANCHOR_WORLDS,
        help=f"Comma-separated train-anchor worlds (default: {DEFAULT_TRAIN_ANCHOR_WORLDS})",
    )
    p.add_argument(
        "--gate-worlds",
        default=DEFAULT_GATE_WORLDS,
        help=f"Comma-separated actor-gate worlds (default: {DEFAULT_GATE_WORLDS})",
    )
    p.add_argument(
        "--seeds",
        default=",".join(str(s) for s in PAIRED_OPTIMIZER_SEEDS),
        help="Comma-separated optimizer seeds (default: 170,171,172,173,174)",
    )
    p.add_argument(
        "--arms",
        default=",".join(ARM_NAMES),
        help=f"Comma-separated arm names (default: {','.join(ARM_NAMES)})",
    )
    p.add_argument(
        "--python-bin",
        default=DEFAULT_PYTHON_BIN,
        help=f"Python executable to invoke in phase commands (default: {DEFAULT_PYTHON_BIN})",
    )
    p.add_argument(
        "--env-prefix",
        default=DEFAULT_ENV_PREFIX,
        help="Environment variables prefix for phase commands",
    )
    p.add_argument(
        "--personality",
        default=DEFAULT_PERSONALITY,
        help="labor personality distribution passed to initialize: 'balanced' or 'dirichlet:ALPHA'",
    )
    p.add_argument(
        "--film-bound",
        type=float,
        default=None,
        help="optional tanh bound on labor FiLM gamma/beta (off by default)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary and first job manifest to stdout without writing files",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)

    opt_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    manifests = build_all_3x5_manifests(
        arms=arms,
        optimizer_seeds=opt_seeds,
        rl_sha=args.rl_sha,
        sim_sha=args.sim_sha,
        threads=args.threads,
        priority=args.priority,
        iterations=args.iterations,
        normalizer_checkpoint=args.normalizer_checkpoint,
        world_seed0=args.world_seed0,
        train_worlds=args.train_worlds,
        train_anchor_worlds=args.train_anchor_worlds,
        gate_worlds=args.gate_worlds,
        python_bin=args.python_bin,
        env_prefix=args.env_prefix,
        personality=args.personality,
        film_bound=args.film_bound,
    )

    if args.dry_run:
        print(f"Generated {len(manifests)} manifests for arms={arms} seeds={opt_seeds}:")
        first_name = next(iter(manifests))
        first_job = manifests[first_name]
        print(f"\n--- Sample Job: {first_name} ({len(first_job['phases'])} phases) ---")
        if args.format == "json":
            print(dump_job_json(first_job))
        else:
            print(dump_job_yaml(first_job))
        return 0

    written = write_manifests(args.out_dir, manifests, output_format=args.format)
    print(f"Successfully wrote {len(written)} files (including index.json) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
