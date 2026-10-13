"""Verify the independently pinned autonomous simulator source and shared contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "Assets/_Anthropocene/Simulation").glob("*.cs")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / "configs/simulator/elder-autonomous-v3.json").read_text())
    actual = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != manifest["simulator_commit"]:
        raise SystemExit("Autonomous simulator commit mismatch")
    if source_hash(args.root) != manifest["simulation_source_sha256"]:
        raise SystemExit("Autonomous simulator source hash mismatch")
    shared = (repo / "docs/AUTONOMOUS_ELDER_ECONOMY_V1.md").read_bytes()
    if shared != (args.root / "Docs/AutonomousElderEconomy-v1.md").read_bytes():
        raise SystemExit("Shared implementation contract differs")
    if hashlib.sha256(shared).hexdigest() != manifest["contract_sha256"]:
        raise SystemExit("Contract digest mismatch")
    print(
        json.dumps(
            {
                "verified": True,
                "simulator_commit": actual,
                "source_sha256": manifest["simulation_source_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
