from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Final

from .config import load_yaml
from .contracts import (
    ACTION_SCHEMA_VERSION,
    CRITIC_SIDECAR_SCHEMA_VERSION,
    ENVIRONMENT_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    SIMULATOR_COMMIT,
    TENSOR_FEATURE_COUNT,
)

_MANIFEST_PATH: Final = Path("configs/simulator/elder-v2.yaml")


def _check_manifest() -> int:
    manifest = load_yaml(_MANIFEST_PATH)
    expected = {
        "commit": SIMULATOR_COMMIT,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "critic_sidecar_schema_version": CRITIC_SIDECAR_SCHEMA_VERSION,
        "tensor_feature_count": TENSOR_FEATURE_COUNT,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Simulator manifest mismatch: {mismatches}")
    print(json.dumps(expected, sort_keys=True))
    return 0


def _run_script(path: str, extra: list[str]) -> int:
    completed = subprocess.run([sys.executable, path, *extra], check=False)
    return completed.returncode


def _ppo_smoke() -> int:
    from anthropocene_rl.training.smoke import run_ppo_smoke

    print(json.dumps(run_ppo_smoke(), sort_keys=True))
    return 0


def _adapter_schema_smoke(dotnet: str) -> int:
    from anthropocene_rl.env.transport import JsonLineProcessTransport

    dll = Path("integration/bin/Release/net8.0/Anthropocene.RLBridge.dll")
    if not dll.exists():
        raise FileNotFoundError(
            "RL bridge is not built; run scripts/bootstrap_simulator.py first"
        )
    with JsonLineProcessTransport([dotnet, str(dll)]) as transport:
        schema = transport.schema()
    expected = {
        "observationSchemaVersion": OBSERVATION_SCHEMA_VERSION,
        "actionSchemaVersion": ACTION_SCHEMA_VERSION,
        "environmentSchemaVersion": ENVIRONMENT_SCHEMA_VERSION,
        "criticSidecarSchemaVersion": CRITIC_SIDECAR_SCHEMA_VERSION,
        "tensorFeatureCount": TENSOR_FEATURE_COUNT,
    }
    mismatches = {
        key: (schema.get(key), value)
        for key, value in expected.items()
        if schema.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Bridge schema mismatch: {mismatches}")
    print(json.dumps(expected, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="anthropocene-rl")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-manifest")
    check_sim = sub.add_parser("check-simulator")
    check_sim.add_argument("--root")
    check_sim.add_argument("--dotnet", default="dotnet")
    adapter = sub.add_parser("adapter-smoke")
    adapter.add_argument("--dotnet", default="dotnet")
    sub.add_parser("ppo-smoke")
    generate_bank = sub.add_parser("generate-snapshot-bank")
    generate_bank.add_argument("--config", type=Path, required=True)
    generate_bank.add_argument("--manifest", type=Path, required=True)
    generate_bank.add_argument("--blob-root", type=Path, default=Path("datasets/blobs"))
    generate_bank.add_argument("--evidence-dir", type=Path, required=True)
    generate_bank.add_argument("--simulator-root", type=Path)
    generate_bank.add_argument("--dotnet", default="dotnet")
    audit_bank = sub.add_parser("audit-snapshot-bank")
    audit_bank.add_argument("--manifest", type=Path, required=True)
    audit_bank.add_argument("--blob-root", type=Path)
    audit_bank.add_argument("--report", type=Path)
    audit_bank.add_argument("--coverage-json", type=Path)
    audit_bank.add_argument("--coverage-markdown", type=Path)
    audit_bank.add_argument("--no-stage0-coverage", action="store_true")
    audit_bank.add_argument("--skip-blob-verification", action="store_true")
    overfit = sub.add_parser("overfit-one-snapshot")
    overfit.add_argument("--manifest", type=Path, default=Path(
        "datasets/manifests/elder-v2-stage0-v1.json"
    ))
    overfit.add_argument("--blob-root", type=Path, default=Path("datasets/blobs"))
    overfit.add_argument("--normalizer", type=Path, default=Path(
        "evidence/stage0-snapshot-bank/provisional-train-normalization.json"
    ))
    overfit.add_argument("--bridge-dll", type=Path, default=Path(
        "integration/bin/Release/net8.0/Anthropocene.RLBridge.dll"
    ))
    overfit.add_argument("--dotnet", default="dotnet")
    overfit.add_argument("--snapshot-id")
    overfit.add_argument("--checkpoint", type=Path, default=Path(
        "checkpoints/stage0-one-snapshot.pt"
    ))
    overfit.add_argument("--diagnostics", type=Path, default=Path(
        "artifacts/stage0-one-snapshot-overfit.json"
    ))
    overfit.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    overfit.add_argument("--seed", type=int, default=1701)
    overfit.add_argument("--updates", type=int, default=12)
    overfit.add_argument("--episodes-per-update", type=int, default=4)
    overfit.add_argument("--episode-steps", type=int, default=8)
    overfit.add_argument("--horizon-years", type=float, default=400.0)
    overfit.add_argument("--lambda-ref", type=float, default=0.95)
    overfit.add_argument("--lambda-reference-years", type=float, default=20.0)
    overfit.add_argument("--reward-scale", type=float, default=0.001)
    overfit.add_argument("--learning-rate", type=float, default=1e-3)
    overfit.add_argument("--actor-learning-rate", type=float, default=3e-4)
    overfit.add_argument("--ppo-epochs", type=int, default=8)
    overfit.add_argument("--minibatch-size", type=int, default=64)
    overfit.add_argument("--target-kl", type=float, default=0.05)
    readiness = sub.add_parser("cultivation-readiness")
    readiness.add_argument("--manifest", type=Path, default=Path(
        "datasets/manifests/elder-v2-stage0-v1.json"
    ))
    readiness.add_argument("--blob-root", type=Path, default=Path("datasets/blobs"))
    readiness.add_argument("--normalizer", type=Path, default=Path(
        "evidence/stage0-snapshot-bank/provisional-train-normalization.json"
    ))
    readiness.add_argument("--bridge-dll", type=Path, default=Path(
        "integration/bin/Release/net8.0/Anthropocene.RLBridge.dll"
    ))
    readiness.add_argument("--dotnet", default="dotnet")
    readiness.add_argument("--snapshot-ids")
    readiness.add_argument("--seeds", default="1701,1901,2101")
    readiness.add_argument("--run-directory", type=Path, default=Path(
        "artifacts/cultivation-readiness-runs"
    ))
    readiness.add_argument("--checkpoint-directory", type=Path, default=Path(
        "checkpoints/cultivation-readiness"
    ))
    readiness.add_argument("--diagnostics", type=Path, default=Path(
        "artifacts/cultivation-readiness.json"
    ))
    readiness.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    readiness.add_argument("--updates", type=int, default=24)
    readiness.add_argument("--episodes-per-update", type=int, default=4)
    readiness.add_argument("--episode-steps", type=int, default=8)
    readiness.add_argument("--horizon-years", type=float, default=400.0)
    readiness.add_argument("--lambda-ref", type=float, default=0.95)
    readiness.add_argument("--lambda-reference-years", type=float, default=20.0)
    readiness.add_argument("--reward-scale", type=float, default=0.001)
    readiness.add_argument("--learning-rate", type=float, default=1e-3)
    readiness.add_argument("--actor-learning-rate", type=float, default=3e-4)
    readiness.add_argument("--ppo-epochs", type=int, default=8)
    readiness.add_argument("--minibatch-size", type=int, default=64)
    readiness.add_argument("--target-kl", type=float, default=0.05)
    throughput = sub.add_parser("benchmark-native-throughput")
    throughput.add_argument("--manifest", type=Path, default=Path(
        "datasets/manifests/elder-v2-stage0-v1.json"
    ))
    throughput.add_argument("--blob-root", type=Path, default=Path("datasets/blobs"))
    throughput.add_argument("--normalizer", type=Path, default=Path(
        "evidence/stage0-snapshot-bank/provisional-train-normalization.json"
    ))
    throughput.add_argument("--bridge-dll", type=Path, default=Path(
        "integration/bin/Release/net8.0/Anthropocene.RLBridge.dll"
    ))
    throughput.add_argument("--dotnet", default="dotnet")
    throughput.add_argument("--snapshot-id")
    throughput.add_argument("--checkpoint", type=Path)
    throughput.add_argument("--workers", default="1,2,4,8")
    throughput.add_argument("--transitions-per-environment", type=int, default=8)
    throughput.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    throughput.add_argument("--seed", type=int, default=1701)
    throughput.add_argument("--diagnostics", type=Path, default=Path(
        "artifacts/stage0-native-throughput.json"
    ))
    args = parser.parse_args()

    if args.command == "check-manifest":
        return _check_manifest()
    if args.command == "check-simulator":
        extra = ["--dotnet", args.dotnet]
        root = args.root or os.environ.get("ANTHROPOCENE_SIM_ROOT")
        if root:
            extra.extend(["--root", root])
        return _run_script("scripts/check_native_contract.py", extra)
    if args.command == "adapter-smoke":
        return _adapter_schema_smoke(args.dotnet)
    if args.command == "ppo-smoke":
        return _ppo_smoke()
    if args.command == "generate-snapshot-bank":
        from anthropocene_rl.datasets.generate import GenerationPaths, generate_snapshot_bank

        simulator_root = args.simulator_root or Path(
            os.environ.get("ANTHROPOCENE_SIM_ROOT", ".simulator/Anthropocene")
        )
        summary = generate_snapshot_bank(
            config_path=args.config,
            paths=GenerationPaths(
                manifest=args.manifest,
                blob_root=args.blob_root,
                evidence_directory=args.evidence_dir,
            ),
            simulator_root=simulator_root,
            dotnet=args.dotnet,
        )
        print(json.dumps(asdict(summary), sort_keys=True))
        return 0
    if args.command == "audit-snapshot-bank":
        from anthropocene_rl.datasets.generate import audit_manifest_file

        report = audit_manifest_file(
            manifest_path=args.manifest,
            blob_root=args.blob_root,
            report_path=args.report,
            coverage_json_path=args.coverage_json,
            coverage_markdown_path=args.coverage_markdown,
            require_stage0_coverage=not args.no_stage0_coverage,
            verify_blobs=not args.skip_blob_verification,
        )
        print(json.dumps(report.model_dump(mode="json"), sort_keys=True))
        return 0 if report.passed else 1
    if args.command == "overfit-one-snapshot":
        from anthropocene_rl.training.overfit import OverfitConfig, run_one_snapshot_overfit
        from anthropocene_rl.training.ppo import PPOConfig

        diagnostics = run_one_snapshot_overfit(
            bridge_dll=args.bridge_dll,
            dotnet=args.dotnet,
            manifest_path=args.manifest,
            blob_root=args.blob_root,
            normalizer_path=args.normalizer,
            snapshot_id=args.snapshot_id,
            checkpoint_path=args.checkpoint,
            diagnostics_path=args.diagnostics,
            config=OverfitConfig(
                seed=args.seed,
                device=args.device,
                updates=args.updates,
                episodes_per_update=args.episodes_per_update,
                episode_steps=args.episode_steps,
                horizon_years=args.horizon_years,
                lambda_ref=args.lambda_ref,
                lambda_reference_years=args.lambda_reference_years,
                reward_scale=args.reward_scale,
            ),
            ppo_config=PPOConfig(
                learning_rate=args.learning_rate,
                actor_learning_rate=args.actor_learning_rate,
                entropy_coefficient=0.0,
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                target_kl=args.target_kl,
            ),
        )
        print(json.dumps(diagnostics, sort_keys=True))
        return 0 if diagnostics["status"] == "PASS" else 1
    if args.command == "benchmark-native-throughput":
        from anthropocene_rl.training.overfit import benchmark_native_throughput

        worker_counts = tuple(int(value) for value in args.workers.split(","))
        diagnostics = benchmark_native_throughput(
            bridge_dll=args.bridge_dll,
            dotnet=args.dotnet,
            manifest_path=args.manifest,
            blob_root=args.blob_root,
            normalizer_path=args.normalizer,
            snapshot_id=args.snapshot_id,
            checkpoint_path=args.checkpoint,
            worker_counts=worker_counts,
            transitions_per_environment=args.transitions_per_environment,
            seed=args.seed,
            device_name=args.device,
            diagnostics_path=args.diagnostics,
        )
        print(json.dumps(diagnostics, sort_keys=True))
        return 0 if diagnostics["status"] == "PASS" else 1
    if args.command == "cultivation-readiness":
        from anthropocene_rl.training.overfit import (
            OverfitConfig,
            run_cultivation_readiness_matrix,
        )
        from anthropocene_rl.training.ppo import PPOConfig

        seeds = tuple(int(value) for value in args.seeds.split(","))
        snapshot_ids = (
            tuple(value for value in args.snapshot_ids.split(",") if value)
            if args.snapshot_ids
            else None
        )
        diagnostics = run_cultivation_readiness_matrix(
            bridge_dll=args.bridge_dll,
            dotnet=args.dotnet,
            manifest_path=args.manifest,
            blob_root=args.blob_root,
            normalizer_path=args.normalizer,
            snapshot_ids=snapshot_ids,
            seeds=seeds,
            run_directory=args.run_directory,
            checkpoint_directory=args.checkpoint_directory,
            diagnostics_path=args.diagnostics,
            config=OverfitConfig(
                seed=seeds[0],
                device=args.device,
                updates=args.updates,
                episodes_per_update=args.episodes_per_update,
                episode_steps=args.episode_steps,
                horizon_years=args.horizon_years,
                lambda_ref=args.lambda_ref,
                lambda_reference_years=args.lambda_reference_years,
                reward_scale=args.reward_scale,
            ),
            ppo_config=PPOConfig(
                learning_rate=args.learning_rate,
                actor_learning_rate=args.actor_learning_rate,
                entropy_coefficient=0.0,
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                target_kl=args.target_kl,
            ),
        )
        print(json.dumps(diagnostics, sort_keys=True))
        return 0 if diagnostics["status"] == "PASS" else 1
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
