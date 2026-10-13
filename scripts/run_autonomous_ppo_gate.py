#!/usr/bin/env python3
"""Run only the frozen bounded CPU pilot, never the proposed canary."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from anthropocene_rl.autonomous_v4.env import AutonomousTransport
from anthropocene_rl.autonomous_v4.training import Trainer, episode, rollout, state_hash


def summarize(episodes):
    families = {}
    for family in sorted({e["family"] for e in episodes}):
        rows = [e for e in episodes if e["family"] == family]
        families[family] = dict(
            welfare=float(np.mean([r["welfare"] for r in rows])),
            survival=float(np.mean([r["survived"] for r in rows])),
            population_retention=float(
                np.mean([r["final_population"] / r["initial_population"] for r in rows])
            ),
            food_met=float(np.mean([r["food_met"] for r in rows])),
            events={k: sum(r["events"][k] for r in rows) for k in rows[0]["events"]},
        )
    return dict(
        welfare=float(np.mean([r["welfare"] for r in families.values()])),
        survival=float(np.mean([r["survival"] for r in families.values()])),
        families=families,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/autonomous-ppo-gate-v1.json")
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dotnet", default="/workspace/Anthropocene/.tools/dotnet/dotnet")
    parser.add_argument("--sim-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["transitions_per_seed"] > 2048 or len(config["initialization_seeds"]) > 3:
        raise ValueError("This command is capped at the pilot budget")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    bank_manifest = (args.bank / "manifest.json").read_bytes()
    entries = json.loads(bank_manifest)["entries"]
    train = [e for e in entries if e["partition"] == "train"]
    development = [e for e in entries if e["partition"] == "development"]
    if {e["world_seed"] for e in train} & {e["world_seed"] for e in development}:
        raise ValueError("World seed leakage")
    source = hashlib.sha256()
    for p in sorted((args.sim_root / "Assets/_Anthropocene/Simulation").glob("*.cs")):
        source.update(p.name.encode() + b"\0" + p.read_bytes())
    provenance = dict(
        bank_sha256=hashlib.sha256(bank_manifest).hexdigest(),
        bridge_sha256=hashlib.sha256(args.bridge.read_bytes()).hexdigest(),
        simulation_source_sha256=source.hexdigest(),
        config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        torch=torch.__version__,
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    reports = []
    started = time.monotonic()
    with AutonomousTransport([args.dotnet, str(args.bridge.resolve())]) as transport:
        schema = transport.schema()
        if [
            schema[k]
            for k in ["actionVersion", "observationVersion", "environmentVersion", "replayVersion"]
        ] != [3, 4, 4, 4]:
            raise ValueError("Native version mismatch")
        for seed in config["initialization_seeds"]:
            trainer = Trainer(config, seed, provenance)
            initial_hash = state_hash(trainer.model)

            def evaluate(trainer=trainer):
                result = []
                for e in development:
                    for rng in config["evaluation_rng_seeds"]:
                        _, s = episode(
                            trainer.model,
                            transport,
                            args.bank,
                            e,
                            torch.Generator().manual_seed(rng),
                            config,
                            verify=True,
                        )
                        result.append(dict(s, sampler_seed=rng))
                return result

            before = evaluate()
            print(
                json.dumps(dict(seed=seed, phase="untrained", result=summarize(before))), flush=True
            )
            training, updates = [], []
            continuation = False
            while trainer.transitions < config["transitions_per_seed"]:
                # Full episodes provide reset-safe checkpoints. The frozen bank survives
                # eight decisions; reject an overshoot instead of silently changing budget.
                if trainer.updates == 0:
                    trainer.checkpoint(args.output / f"seed-{seed}-initial.pt")
                rows, summaries = rollout(
                    trainer, transport, args.bank, train, verify=trainer.updates == 0
                )
                if trainer.transitions + len(rows) > config["transitions_per_seed"]:
                    raise ValueError("Budget overshoot")
                metrics = trainer.update(rows)
                if trainer.updates == 1:
                    restored = Trainer(config, seed, provenance)
                    restored.restore(args.output / f"seed-{seed}-initial.pt")
                    repeated, repeated_summaries = rollout(restored, transport, args.bank, train)
                    restored.update(repeated)
                    continuation = [r["fingerprint"] for r in summaries] == [
                        r["fingerprint"] for r in repeated_summaries
                    ]
                    continuation &= state_hash(trainer.model) == state_hash(restored.model)
                    continuation &= torch.equal(
                        trainer.sampler.get_state(), restored.sampler.get_state()
                    )
                    continuation &= torch.equal(
                        trainer.shuffle.get_state(), restored.shuffle.get_state()
                    )
                    if not continuation:
                        raise ValueError("Exact native continuation failed")
                training.extend(summaries)
                updates.extend(metrics)
                print(
                    json.dumps(
                        dict(
                            seed=seed,
                            transitions=trainer.transitions,
                            kl=max(m["kl"] for m in metrics),
                            recent=summarize(summaries),
                        )
                    ),
                    flush=True,
                )
            trainer.checkpoint(args.output / f"seed-{seed}-final.pt")
            after = evaluate()
            report = dict(
                seed=seed,
                transitions=trainer.transitions,
                updates=trainer.updates,
                before=summarize(before),
                after=summarize(after),
                training=summarize(training),
                initial_model_sha256=initial_hash,
                final_model_sha256=state_hash(trainer.model),
                checkpoint_continuation=continuation,
                head_gradient_max={
                    k: max(m["head_grad"][k] for m in updates) for k in updates[0]["head_grad"]
                },
                max_kl=max(m["kl"] for m in updates),
                actor_updates=sum(not m["actor_stopped"] for m in updates),
            )
            reports.append(report)
            (args.output / f"seed-{seed}-episodes.json").write_text(
                json.dumps(dict(before=before, after=after, training=training), indent=2)
            )
            (args.output / f"seed-{seed}-updates.json").write_text(json.dumps(updates, indent=2))
            (args.output / f"seed-{seed}-summary.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(dict(seed=seed, phase="final", result=report)), flush=True)
    passed = (
        all(
            r["after"]["survival"] >= 0.75 and r["after"]["welfare"] >= 0.8 * r["before"]["welfare"]
            for r in reports
        )
        and sum(r["after"]["welfare"] > r["before"]["welfare"] for r in reports) >= 2
        and sum(r["after"]["welfare"] for r in reports)
        > sum(r["before"]["welfare"] for r in reports)
    )
    summary = dict(
        protocol=config,
        provenance=provenance,
        seeds=reports,
        numerical_welfare_gate=passed,
        wall_seconds=time.monotonic() - started,
        independent_review="pending",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(dict(numerical_welfare_gate=passed, wall_seconds=summary["wall_seconds"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
