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
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
