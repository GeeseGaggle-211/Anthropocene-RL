"""No-update, three-family, training-world-disjoint current-policy value audit."""

import argparse
import json
import time
from pathlib import Path

import torch
from run_autonomous_causal_gate_v7 import combine_inputs, summary_without_tensors
from run_autonomous_learning_v10 import load_bank, restore, sha
from train_autonomous_opportunity_v9 import OpportunityCollector, pair_hash

from anthropocene_rl.autonomous_v4.collection import episode_seed
from anthropocene_rl.autonomous_v5.causal import n_step_targets
from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger


def score(prediction, target):
    error = prediction - target
    variance = float(target.var(unbiased=False))
    if variance <= 1e-8 or not torch.isfinite(error).all():
        raise ValueError("Nonfinite or degenerate validation target")
    return dict(
        rows=len(target),
        mse=float(error.square().mean()),
        return_variance=variance,
        mse_over_return_variance=float(error.square().mean()) / variance,
        explained_variance=1 - float(error.var(unbiased=False)) / variance,
        mean_value_bias=float(error.mean()),
    )


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
            raise ValueError("Changed frozen input: " + pin["path"])
    if cfg["caps"] != dict(audit=225, audit_replay=225) or cfg["families"] != [
        "survival",
        "trade",
        "husbandry",
    ]:
        raise ValueError("Undeclared audit scope")
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    remaining = 1812.8917767360253 - prior["cumulative_wall_seconds"]
    if prior["total_resolves"] + 450 > 51951 or remaining <= 0:
        raise ValueError("V10 cumulative envelope exhausted")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    ledger = BudgetLedger(
        args.output / "budget.json",
        cfg["caps"],
        prior_resolves=prior["total_resolves"],
        prior_wall_seconds=prior["cumulative_wall_seconds"],
        deadline=time.monotonic() + min(remaining, cfg["max_active_seconds"]),
        native_version=6,
    )
    bank, entries = load_bank(cfg["inputs"]["bank"])
    gate = json.loads(Path(cfg["inputs"]["parent_gate"]["path"]).read_text())
    if gate["status"] != "STAGED_REVIEW_REQUIRED":
        raise ValueError("Wrong retained parent stage")
    result = dict(
        version=cfg["version"], status="RUNNING", config_sha256=sha(args.config), cells=[]
    )

    def save():
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        with OpportunityCollector(
            [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]],
            ledger=ledger,
            workers=4,
            deadline=ledger.deadline,
        ) as collector:
            ledger.phase = "audit"
            for seed in [101, 211, 307]:
                pin = cfg["inputs"][f"parent_seed{seed}"]
                trainer = restore(pin)
                parent = next(c for c in gate["cells"] if c["seed"] == seed)
                if (
                    parent["status"] != "COMPLETE"
                    or parent["durable_transitions"] != 4096
                    or parent["checkpoint_sha256"] != pin["sha256"]
                    or parent["checkpoint_counters"]
                    != dict(
                        episodes=trainer.episodes,
                        transitions=trainer.transitions,
                        updates=trainer.updates,
                    )
                ):
                    raise ValueError("Wrong retained checkpoint/counters")
                ids = cfg["audit_entries"]
                if len(ids) != 3 or [entries[i]["family"] for i in ids] != cfg["families"]:
                    raise ValueError("Wrong audit families")
                if any(entries[i]["world_seed"] not in [886005, 886006] for i in ids):
                    raise ValueError("Validation-world leakage")
                protected = pair_hash(trainer)
                jobs = [
                    dict(entry=entries[i], sampler_seed=episode_seed(seed + 310000, j), limit=25)
                    for j, i in enumerate(ids)
                ]
                rows, episodes = collector.collect(
                    trainer.model,
                    bank,
                    jobs,
                    trainer.config,
                    collect=True,
                    verify=True,
                )
                targets = {k: [] for k in ["target", "zero_tail", "doubled_tail", "tail_weight"]}
                for e in episodes:
                    trace = e["trace"]
                    y = n_step_targets(
                        [r["reward"] for r in trace],
                        [r["elapsed"] for r in trace],
                        [r["terminal"] for r in trace],
                        e["tail_value"],
                        400,
                        120,
                    )
                    for key in targets:
                        targets[key].extend(y[key])
                with torch.no_grad():
                    values = trainer.model.value(combine_inputs(rows)).cpu()
                cache = args.output / f"targets-seed{seed}.pt"
                torch.save(
                    dict(
                        rows=rows,
                        targets=targets,
                        values=values,
                        episodes=[summary_without_tensors(e) for e in episodes],
                    ),
                    cache,
                )
                if pair_hash(trainer) != protected:
                    raise ValueError("Read-only audit changed retained state")
                result["cells"].append(
                    dict(
                        seed=seed,
                        protected_before=protected,
                        protected_after=pair_hash(trainer),
                        metrics={
                            k: score(values, torch.tensor(targets[k], dtype=values.dtype))
                            for k in ["target", "zero_tail", "doubled_tail"]
                        },
                        mean_tail_weight=float(torch.tensor(targets["tail_weight"]).mean()),
                        terminated=sum(e["terminal"] for e in episodes),
                        truncated=sum(not e["terminal"] for e in episodes),
                        cache=str(cache),
                        cache_sha256=sha(cache),
                        episodes=[summary_without_tensors(e) for e in episodes],
                    )
                )
                save()
        result["status"] = "REVIEW_REQUIRED"
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print(json.dumps({k: result[k] for k in ["status", "budget"]}))


if __name__ == "__main__":
    main()
