from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from .config import load_yaml
from .contracts import (
    ACTION_SCHEMA_VERSION,
    ENVIRONMENT_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    SIMULATOR_COMMIT,
    TENSOR_FEATURE_COUNT,
)

_MANIFEST_PATH: Final = Path("configs/simulator/elder-v1.yaml")
_COMMANDS: Final = (
    "check-manifest",
    "check-simulator",
    "adapter-smoke",
    "train",
    "evaluate",
    "export",
)


def _check_manifest() -> int:
    manifest = load_yaml(_MANIFEST_PATH)
    expected = {
        "commit": SIMULATOR_COMMIT,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "tensor_feature_count": TENSOR_FEATURE_COUNT,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        formatted = ", ".join(
            f"{key}: manifest={actual!r}, expected={wanted!r}"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(f"Simulator manifest mismatch: {formatted}")

    print(
        "manifest OK:",
        SIMULATOR_COMMIT,
        OBSERVATION_SCHEMA_VERSION,
        ACTION_SCHEMA_VERSION,
        ENVIRONMENT_SCHEMA_VERSION,
        TENSOR_FEATURE_COUNT,
    )
    return 0


def _not_implemented(command: str) -> int:
    raise NotImplementedError(
        f"{command} is a repository skeleton command; implement after section 34 is locked"
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="anthropocene-rl")
    parser.add_argument("command", choices=_COMMANDS)
    args = parser.parse_args()
    if args.command == "check-manifest":
        return _check_manifest()
    return _not_implemented(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
