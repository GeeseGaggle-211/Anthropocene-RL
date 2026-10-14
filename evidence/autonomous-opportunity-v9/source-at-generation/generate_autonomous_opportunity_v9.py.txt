"""Freeze and charge native-only curriculum generation; no neural operations."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.output.exists() or args.bank.exists():
        raise FileExistsError("Use fresh output and bank directories")
    for pin in cfg["inputs"].values():
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError(f"Changed pinned input: {pin['path']}")
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    if cfg["caps"] != dict(
        mechanical=3, mechanical_replay=3, generation=1200, generation_replay=1200
    ):
        raise ValueError("Frozen generation envelope changed")
    args.output.mkdir(parents=True)
    ledger = BudgetLedger(
        args.output / "budget.json",
        cfg["caps"],
        prior_resolves=prior["total_resolves"],
        prior_wall_seconds=prior["cumulative_wall_seconds"],
        deadline=time.monotonic() + 600,
        native_version=6,
    )
    result = dict(version=cfg["version"], status="RUNNING", config_sha256=sha(args.config))

    def run(name, argv):
        try:
            completed = subprocess.run(
                [cfg["dotnet"], *argv],
                text=True,
                capture_output=True,
                timeout=ledger.deadline - time.monotonic(),
            )
        except subprocess.TimeoutExpired as error:
            for suffix, value in [("stdout", error.stdout), ("stderr", error.stderr)]:
                text = value.decode(errors="replace") if isinstance(value, bytes) else value
                (args.output / f"{name}.{suffix}").write_text(text or "")
            raise
        (args.output / f"{name}.stdout").write_text(completed.stdout)
        (args.output / f"{name}.stderr").write_text(completed.stderr)
        completed.check_returncode()
        return completed.stdout

    try:
        ledger.reserve("mechanical", 3)
        ledger.reserve("mechanical_replay", 3)
        proof = json.loads(
            run(
                "mechanical",
                [cfg["inputs"]["native_tool"]["path"], "--autonomous-opportunity-only"],
            )
            .strip()
            .splitlines()[-1]
        )
        if (
            proof["result"] != "PASS"
            or proof["validationResolves"] != 3
            or proof["replayResolves"] != 3
        ):
            raise ValueError("Known-address physical proof failed")
        result["mechanical"] = proof
        # Native progress.json independently charges actual calls before each step.
        # Conservatively reserve full invocations globally even for a partial failure.
        ledger.reserve("generation", 1200)
        ledger.reserve("generation_replay", 1200)
        run(
            "generation",
            [cfg["inputs"]["generator"]["path"], cfg["inputs"]["plan"]["path"], str(args.bank)],
        )
        native = json.loads((args.bank / "manifest.json").read_text())
        if (
            native["status"] not in ("PASS", "PASS_WITH_GAPS")
            or native["generation_resolves"] > 1200
            or native["replay_resolves"] != native["generation_resolves"]
        ):
            raise ValueError("Native generation/replay envelope failed")
        for entry in native["entries"]:
            if sha(args.bank / entry["snapshot"]) != entry["sha256"]:
                raise ValueError("Generated snapshot hash mismatch")
        result.update(
            status="PASS",
            manifest_sha256=sha(args.bank / "manifest.json"),
            native={k: v for k, v in native.items() if k not in ("entries",)},
            snapshots=len(native["entries"]),
            neural_calls=0,
            optimizer_steps=0,
        )
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "native"}))


if __name__ == "__main__":
    main()
