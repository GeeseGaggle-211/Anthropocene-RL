#!/usr/bin/env python3
"""Build the pinned native Elder tool and verify schema/version identity."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from anthropocene_rl.contracts import (
    ACTION_SCHEMA_VERSION,
    CRITIC_SIDECAR_SCHEMA_VERSION,
    ENVIRONMENT_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    SIMULATOR_COMMIT,
    TENSOR_FEATURE_COUNT,
)


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--dotnet", default="dotnet")
    args = parser.parse_args()
    root = args.root or (
        Path(os.environ["ANTHROPOCENE_SIM_ROOT"])
        if "ANTHROPOCENE_SIM_ROOT" in os.environ
        else Path(".simulator/Anthropocene")
    )
    root = root.resolve()
    head = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    if head != SIMULATOR_COMMIT:
        raise SystemExit(f"Simulator pin mismatch: {head} != {SIMULATOR_COMMIT}")

    project = root / "Tools/ElderRL/Anthropocene.ElderRL.csproj"
    _run([args.dotnet, "build", str(project), "-c", "Release", "-v:q"])
    dll = root / "Tools/ElderRL/bin/Release/net8.0/Anthropocene.ElderRL.dll"
    schema = json.loads(_run([args.dotnet, str(dll), "schema"]))
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
        raise SystemExit(f"Native schema mismatch: {mismatches}")
    print(json.dumps(expected, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
