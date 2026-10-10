from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
