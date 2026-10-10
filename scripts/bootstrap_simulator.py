#!/usr/bin/env python3
"""Prepare the exact native simulator checkout and build the RL stdio bridge."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from anthropocene_rl.contracts import SIMULATOR_COMMIT, SIMULATOR_REPOSITORY


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def _head(root: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--clone-dir", type=Path, default=Path(".simulator/Anthropocene"))
    parser.add_argument("--dotnet", default="dotnet")
    args = parser.parse_args()

    explicit = args.root or (
        Path(os.environ["ANTHROPOCENE_SIM_ROOT"])
        if "ANTHROPOCENE_SIM_ROOT" in os.environ
        else None
    )
    if explicit is not None:
        root = explicit.resolve()
        if _head(root) != SIMULATOR_COMMIT:
            raise SystemExit(
                f"ANTHROPOCENE_SIM_ROOT must be exactly {SIMULATOR_COMMIT}; got {_head(root)}"
            )
    else:
        root = args.clone_dir.resolve()
        if not root.exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", f"https://github.com/{SIMULATOR_REPOSITORY}.git", str(root)])
        if _head(root) != SIMULATOR_COMMIT:
            _run(["git", "fetch", "origin", SIMULATOR_COMMIT], cwd=root)
            _run(["git", "checkout", "--detach", SIMULATOR_COMMIT], cwd=root)
        if _head(root) != SIMULATOR_COMMIT:
            raise SystemExit("Failed to checkout the frozen native simulator commit")

    project = Path("integration/Anthropocene.RLBridge.csproj").resolve()
    _run(
        [
            args.dotnet,
            "build",
            str(project),
            "-c",
            "Release",
            f"-p:AnthropoceneSimRoot={root}",
        ]
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
