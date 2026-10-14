"""V13 execution probe; retained V11 actors/critics are not promoted for training."""

import argparse
import json
import time
from pathlib import Path

import torch
from run_autonomous_model8_plug_v11 import plug, sha

from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.output.exists() or cfg["caps"] != dict(plug=8, plug_replay=8):
        raise ValueError("Fresh output and fixed16-resolve probe required")
    for pin in cfg["inputs"].values():
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError("Changed probe pin: " + pin["path"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    ledger = BudgetLedger(
        args.output / "budget.json",
        cfg["caps"],
        prior_resolves=cfg["prior_resolves"],
        prior_wall_seconds=cfg["prior_active_seconds"],
        deadline=time.monotonic() + 60,
        native_version=6,
    )
    result = dict(
        version=cfg["version"],
        status="RUNNING",
        probe_only=True,
        new_profile_training_ready=False,
        config_sha256=sha(args.config),
    )
    try:
        result["plug"] = plug(cfg, ledger, args.output)
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(status=result["status"], budget=result["budget"])))


if __name__ == "__main__":
    main()
