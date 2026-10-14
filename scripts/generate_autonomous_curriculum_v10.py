"""Fresh native-only source bank, with a separate cumulative V10 reservation."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger


def main():
    import hashlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.bank.exists():
        raise FileExistsError("Never overwrite a source bank or run")
    cfg = json.loads(args.config.read_text())
    for pin in cfg["inputs"].values():
        if hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest() != pin["sha256"]:
            raise ValueError("Changed prospective pin: " + pin["path"])
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    if cfg["caps"] != dict(generation=1200, generation_replay=1200):
        raise ValueError("Frozen generation caps")
    args.output.mkdir(parents=True)
    ledger = BudgetLedger(
        args.output / "budget.json",
        cfg["caps"],
        prior_resolves=prior["total_resolves"],
        prior_wall_seconds=prior["cumulative_wall_seconds"],
        deadline=time.monotonic() + cfg["maximum_additional_wall_seconds"],
        native_version=6,
    )
    result = dict(version=cfg["version"], status="RUNNING", neural_calls=0, optimizer_steps=0)
    try:
        ledger.reserve("generation", 1200)
        ledger.reserve("generation_replay", 1200)
        try:
            completed = subprocess.run(
                [
                    cfg["dotnet"],
                    cfg["inputs"]["generator"]["path"],
                    cfg["inputs"]["plan"]["path"],
                    str(args.bank),
                ],
                text=True,
                capture_output=True,
                timeout=ledger.deadline - time.monotonic(),
            )
        except subprocess.TimeoutExpired as error:
            for name, value in [("stdout", error.stdout), ("stderr", error.stderr)]:
                (args.output / name).write_text(
                    value.decode(errors="replace") if isinstance(value, bytes) else value or ""
                )
            raise
        (args.output / "stdout").write_text(completed.stdout)
        (args.output / "stderr").write_text(completed.stderr)
        completed.check_returncode()
        manifest = json.loads((args.bank / "manifest.json").read_text())
        if (
            manifest["status"] not in ("PASS", "PASS_WITH_GAPS")
            or manifest["generation_resolves"] > 1200
            or manifest["replay_resolves"] != manifest["generation_resolves"]
        ):
            raise ValueError("Native generation/replay failed")
        for entry in manifest["entries"]:
            if (
                hashlib.sha256((args.bank / entry["snapshot"]).read_bytes()).hexdigest()
                != entry["sha256"]
            ):
                raise ValueError("Source snapshot changed")
        result.update(
            status="PASS",
            manifest_sha256=hashlib.sha256((args.bank / "manifest.json").read_bytes()).hexdigest(),
            snapshots=len(manifest["entries"]),
            source_native_version=manifest["version"],
            actual_generation_resolves=manifest["generation_resolves"],
            actual_replay_resolves=manifest["replay_resolves"],
            unavailable=manifest["unavailable"],
        )
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
