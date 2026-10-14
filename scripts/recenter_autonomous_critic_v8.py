"""One pinned TRAIN intercept fit for each failedH400; no native/PPO work."""

import argparse
import copy
import json
import signal
import time
from pathlib import Path

import torch
from run_autonomous_causal_gate_v7 import load_calibrated, sha

from anthropocene_rl.autonomous_v5.causal import critic_ready
from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger
from anthropocene_rl.autonomous_v6.calibration import fit_intercept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(args.config.read_text())
    for pin in cfg["inputs"].values():
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError("Changed intercept-stage input")
    if (
        cfg["maximum_closed_form_fits"] != 2
        or cfg["additional_native_resolves"] != 0
        or cfg["additional_optimizer_steps"] != 0
    ):
        raise ValueError("Intercept-stage allowance changed")
    original = json.loads(Path(cfg["inputs"]["original_summary"]["path"]).read_text())
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    if original["status"] != "NO-GO_CRITIC":
        raise ValueError("Expected preserved failed critic stage")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + 1800 - prior["cumulative_wall_seconds"]
    ledger = BudgetLedger(
        args.output / "budget.json",
        {},
        prior_resolves=prior["total_resolves"],
        prior_wall_seconds=prior["cumulative_wall_seconds"],
        deadline=deadline,
        native_version=6,
    )
    result = copy.deepcopy(original)
    result.update(
        version="autonomous-causal-intercept-v8",
        config=cfg,
        config_sha256=sha(args.config),
        status="RUNNING",
        passed=False,
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = torch.load(
        cfg["inputs"]["native_targets_h400"]["path"], map_location="cpu", weights_only=False
    )
    fit, y = data["fit"], data["targets"]
    if not fit.any() or not (~fit).any():
        raise ValueError("Empty protected target fold")

    def expired(*_):
        raise TimeoutError("Cumulative causal deadline")

    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(deadline - time.monotonic())))
    try:
        for c in result["calibrations"]:
            if c["report"]["passed"]:
                if c["horizon"] != 100:
                    raise ValueError("Unexpected failed-model selection")
                continue
            if c["horizon"] != 400:
                raise ValueError("Only failedH400 models may change")
            mode = c["feature_mode"]
            trainer = load_calibrated(cfg["inputs"][f"failed_h400_{mode}"])
            x = {k: v.clone() for k, v in data["obs"].items()}
            if mode == "receipt_none":
                x["x"][:, 761:] = 0
            report = fit_intercept(trainer, ({k: v[fit] for k, v in x.items()}, y[fit]))
            with torch.no_grad():
                errors = trainer.model.value({k: v[~fit] for k, v in x.items()}) - y[~fit]
                mse, bias = float(errors.square().mean()), float(errors.mean())
            r = c["report"]
            metrics = dict(
                mse=mse,
                normalized_mse=mse / max(r["constant_mse"], 1),
                normalized_bias=bias / max(float(y[~fit].std(unbiased=False)), 2),
            )
            r.update(
                analytic_intercept=report,
                final=metrics,
                passed=critic_ready(metrics, r["entry"], r["constant_mse"]),
            )
            checkpoint = args.output / f"calibrated-h400-{mode}.pt"
            trainer.provenance = dict(
                trainer.provenance,
                intercept_config_sha256=result["config_sha256"],
                intercept_native_target_sha256=cfg["inputs"]["native_targets_h400"]["sha256"],
            )
            trainer.checkpoint(checkpoint)
            c.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
        result["passed"] = all(c["report"]["passed"] for c in result["calibrations"])
        result["status"] = "READY" if result["passed"] else "NO-GO_CRITIC"
    except BaseException as error:
        result.update(status="FAILED", error=str(error))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        result["wall_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        ledger.save()
    print(json.dumps({k: result[k] for k in ["status", "passed", "wall_seconds"]}))


if __name__ == "__main__":
    main()
