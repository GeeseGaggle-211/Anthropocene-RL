"""Prospective uniform entry-readiness rule; no refit, target refresh or native work."""

import argparse
import json
import time
from pathlib import Path

import torch
from run_autonomous_learning_v10 import restore, sha
from run_autonomous_reward_gate_v11 import fork, strata

from anthropocene_rl.autonomous_v5.causal import protected_state
from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger
from anthropocene_rl.autonomous_v6.readiness import metrics, ready
from anthropocene_rl.autonomous_v6.reward_objectives import Objective


def exact(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(exact(left[k], right[k]) for k in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(exact(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


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
            raise ValueError("Changed frozen pin: " + pin["path"])
    source = json.loads(Path(cfg["inputs"]["source_calibration"]["path"]).read_text())
    if source["status"] != "REVIEW_REQUIRED_CRITIC" or len(source["calibrations"]) != 9:
        raise ValueError("Expected preserved nine-arm calibration result")
    if not all(ready(c["entry"], c["entry"]["mse"]) for c in source["calibrations"]):
        raise ValueError("Entry-failing arm requires a separate repair rule")
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    envelope = json.loads(Path(cfg["inputs"]["envelope"]["path"]).read_text())
    remaining = envelope["total_active_second_ceiling"] - prior["cumulative_wall_seconds"]
    if remaining <= 0 or prior["total_resolves"] > envelope["total_resolve_ceiling"]:
        raise ValueError("Prospective envelope exhausted")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    ledger = BudgetLedger(
        args.output / "budget.json",
        {},
        prior_resolves=prior["total_resolves"],
        prior_wall_seconds=prior["cumulative_wall_seconds"],
        deadline=time.monotonic() + min(remaining, 60),
        native_version=6,
    )
    result = dict(
        version=cfg["version"], status="RUNNING", config_sha256=sha(args.config), calibrations=[]
    )

    def save():
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        for c in source["calibrations"]:
            if time.monotonic() >= ledger.deadline:
                raise TimeoutError("Entry restoration deadline")
            seed, kind = c["seed"], c["kind"]
            pin = cfg["inputs"][f"parent_seed{seed}"]
            parent = restore(pin)
            trainer = fork(parent, pin, Objective(kind), source["config_sha256"])
            if not exact(parent.model.state_dict(), trainer.model.state_dict()):
                raise ValueError("Entry model differs from exact original parent")
            if not exact(parent.optimizer.state_dict(), trainer.optimizer.state_dict()):
                raise ValueError("Entry populated optimizer differs")
            if protected_state(parent) != protected_state(trainer):
                raise ValueError("Entry actor/Adam/RNG/counters differ")
            cache = torch.load(c["objective_cache"], map_location="cpu", weights_only=False)
            offset, maximum_difference = 0, 0.0
            with torch.no_grad():
                for e in cache["episodes"]:
                    rows = cache["rows"][offset : offset + e["steps"]]
                    offset += e["steps"]
                    x = {k: torch.cat([r["obs"][k] for r in rows]) for k in rows[0]["obs"]}
                    values = trainer.model.value(x)
                    expected = torch.tensor([r["value"] for r in rows], dtype=values.dtype)
                    difference = float((values - expected).abs().max())
                    maximum_difference = max(maximum_difference, difference)
                    if not torch.equal(values, expected):
                        raise ValueError("Restored entry values differ from original frozen cache")
            value = metrics(trainer, cache["fitting"], cache["validation"])
            if not ready(value, c["entry"]["mse"]):
                raise ValueError("Restored unchanged entry does not pass strict gate")
            trainer.provenance = dict(
                trainer.provenance,
                objective_entry_restoration_config_sha256=sha(args.config),
                unchanged_objective_target_sha256=c["objective_cache_sha256"],
            )
            path = args.output / f"entry-ready-{kind}-seed{seed}.pt"
            trainer.checkpoint(path)
            result["calibrations"].append(
                dict(
                    seed=seed,
                    kind=kind,
                    passed=True,
                    mode="entry-ready-restored-no-fit",
                    entry=c["entry"],
                    final=value,
                    maximum_cached_value_difference=maximum_difference,
                    full_model_and_optimizer_exact=True,
                    protected_actor_rng_counters_exact=True,
                    objective_cache=c["objective_cache"],
                    objective_cache_sha256=c["objective_cache_sha256"],
                    strata=strata(trainer, cache["validation"]),
                    checkpoint=str(path),
                    checkpoint_sha256=sha(path),
                    checkpoint_counters=dict(
                        episodes=trainer.episodes,
                        transitions=trainer.transitions,
                        updates=trainer.updates,
                    ),
                    fitted_attempt_passed=c["passed"],
                    fitted_attempt_checkpoint=c["checkpoint"],
                    fitted_attempt_checkpoint_sha256=c["checkpoint_sha256"],
                )
            )
            save()
        result["status"] = "READY"
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print(json.dumps({k: result[k] for k in ["status", "budget"]}))


if __name__ == "__main__":
    main()
