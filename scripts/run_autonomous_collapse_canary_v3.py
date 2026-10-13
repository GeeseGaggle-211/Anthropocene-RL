"""Hard-capped CPU canary: economic outcomes are metrics; recoverability is a gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from run_autonomous_ppo_gate import summarize

from anthropocene_rl.autonomous_v4.collection import Collector
from anthropocene_rl.autonomous_v4.diagnostics import full_recovery_probe, health, optimizer_stall
from anthropocene_rl.autonomous_v4.training import state_hash
from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3 as Trainer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rss_bytes(pids):
    total = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
        except FileNotFoundError:
            pass
    return total


def parameter_vector(parameters):
    return torch.cat([p.detach().flatten() for p in parameters]).clone()


def measured_update(trainer, rows):
    actor = [*trainer.model.actor.parameters(), *trainer.model.heads.parameters()]
    critic = list(trainer.model.critic.parameters())
    before_a, before_c = parameter_vector(actor), parameter_vector(critic)
    observed = {k: torch.cat([r["obs"][k] for r in rows]) for k in rows[0]["obs"]}
    with torch.no_grad():
        residual_mse = float(
            (trainer.model.value(observed) - torch.tensor([r["return"] for r in rows]))
            .square()
            .mean()
        )
    gradients = dict(actor=[], critic=[])
    hooks = [
        trainer.model.actor[0].weight.register_hook(
            lambda g: gradients["actor"].append(float(g.norm()))
        ),
        trainer.model.critic[0][0].weight.register_hook(
            lambda g: gradients["critic"].append(float(g.norm()))
        ),
    ]
    try:
        metrics = trainer.update(rows)
    finally:
        for hook in hooks:
            hook.remove()
    step_a = float((parameter_vector(actor) - before_a).norm())
    step_c = float((parameter_vector(critic) - before_c).norm())
    return dict(
        critic_residual_mse=residual_mse,
        actor_step_l2=step_a,
        critic_step_l2=step_c,
        actor_relative_step=step_a / max(1, float(before_a.norm())),
        critic_relative_step=step_c / max(1, float(before_c.norm())),
        actor_first_layer_grad=max(gradients["actor"]),
        critic_first_layer_grad=max(gradients["critic"]),
        advantage_std=float(np.std([r["advantage"] for r in rows])),
        return_std=float(np.std([r["return"] for r in rows])),
        actor_updates=sum(not r["actor_stopped"] for r in metrics),
        minibatches=len(metrics),
        maximum_kl=max(r["kl"] for r in metrics),
        clip_fraction=float(np.mean([r["clip_fraction"] for r in metrics])),
        head_gradient_max={
            k: max(r["head_grad"][k] for r in metrics) for k in metrics[0]["head_grad"]
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training/autonomous-collapse-canary-v3.json")
    )
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if not (
        cfg["additional_transitions_per_seed"] == 4096
        and 0 < cfg["maximum_wall_seconds"] <= 1800
        and cfg["seeds"] == [29, 43]
        and cfg["maximum_total_transitions"] == 8192
        and cfg["additional_transitions_per_seed"] * len(cfg["seeds"])
        <= cfg["maximum_total_transitions"]
        and cfg["workers"] <= 4
        and cfg["torch_threads"] == 1
        and cfg["evaluate_every_transitions"] == 4096
    ):
        raise ValueError("Canary envelope exceeded")
    if sha(args.parent_run / "summary.json") != cfg["previous_campaign_sha256"]:
        raise ValueError("Previous campaign digest mismatch")
    prior = json.loads((args.parent_run / "summary.json").read_text())
    if prior["additional_transitions"] + cfg["maximum_total_transitions"] > 98304:
        raise ValueError("Cumulative authorized native budget exceeded")
    prior_by_seed = {r["seed"]: r["additional_transitions"] for r in prior["seeds"]}
    if any(prior_by_seed[s] + cfg["additional_transitions_per_seed"] > 32768 for s in cfg["seeds"]):
        raise ValueError("Cumulative per-seed budget exceeded")
    for seed in cfg["seeds"]:
        parent = args.parent_run / f"seed-{seed}-step-{prior_by_seed[seed]}.pt"
        if sha(parent) != cfg["parent_checkpoint_sha256"][str(seed)]:
            raise ValueError("Preregistered parent checkpoint mismatch")
    if sha(args.probe) != cfg["probe_sha256"]:
        raise ValueError("Preregistered probe mismatch")
    if (
        sha(args.bridge) != cfg["bridge_sha256"]
        or sha(args.bank / "manifest.json") != cfg["bank_sha256"]
    ):
        raise ValueError("Preregistered bridge/bank mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pilot = json.loads((args.pilot / "summary.json").read_text())
    protocol_path = Path("configs/training/autonomous-ppo-layernorm-v3-recovery.json")
    if sha(protocol_path) != cfg["model_config_sha256"]:
        raise ValueError("Preregistered model configuration mismatch")
    protocol = json.loads(protocol_path.read_text())
    original_provenance = pilot["provenance"]
    if (
        sha(args.bridge) != original_provenance["bridge_sha256"]
        or sha(args.bank / "manifest.json") != original_provenance["bank_sha256"]
    ):
        raise ValueError("Frozen native bridge/bank mismatch")
    probe_manifest = json.loads(args.probe.with_suffix(".json").read_text())
    if sha(args.probe) != probe_manifest["sha256"]:
        raise ValueError("Probe checksum mismatch")
    obs = torch.load(args.probe, weights_only=True)
    entries = json.loads((args.bank / "manifest.json").read_text())["entries"]
    train = [e for e in entries if e["partition"] == "train"]
    dev = [e for e in entries if e["partition"] == "development"]
    if {e["world_seed"] for e in train} & {e["world_seed"] for e in dev}:
        raise ValueError("Partition leakage")
    source_files = [
        *Path("src/anthropocene_rl/autonomous_v4").glob("*.py"),
        *Path("src/anthropocene_rl/policies").glob("autonomous_*.py"),
        *Path("src/anthropocene_rl/training").glob("autonomous_*.py"),
        Path("src/anthropocene_rl/autonomous/policy.py"),
        Path("src/anthropocene_rl/training/returns.py"),
        protocol_path,
        Path(__file__),
        Path("scripts/run_autonomous_ppo_gate.py"),
    ]
    provenance = dict(
        pilot=original_provenance,
        previous_campaign_sha256=sha(args.parent_run / "summary.json"),
        parent_checkpoint_sha256=cfg["parent_checkpoint_sha256"],
        protocol_sha256=sha(args.config),
        probe_sha256=sha(args.probe),
        learner_files={str(p): sha(p) for p in sorted(source_files)},
        collector="episode-keyed-v1",
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    started = time.monotonic()
    deadline = started + cfg["maximum_wall_seconds"]

    def expired(*_):
        raise TimeoutError("Hard campaign wall limit reached")

    prior_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(cfg["maximum_wall_seconds"])
    results = []
    total = 0
    try:
        with Collector(
            ["/workspace/Anthropocene/.tools/dotnet/dotnet", str(args.bridge.resolve())],
            workers=cfg["workers"],
            deadline=deadline,
            request_timeout=cfg["native_request_timeout_seconds"],
        ) as collector:
            for seed in cfg["seeds"]:
                trainer = Trainer(protocol, seed, original_provenance)
                parent = args.parent_run / f"seed-{seed}-step-{prior_by_seed[seed]}.pt"
                trainer.warm_start(
                    parent, expected_sha256=cfg["parent_checkpoint_sha256"][str(seed)]
                )
                trainer.provenance = dict(
                    provenance, parent_checkpoint_sha256=sha(parent), training_seed=seed
                )
                opening = trainer.transitions
                sequence = []
                training = []
                updates = []
                timings = []
                peak_rss = 0
                baseline_rss = None

                def evaluate(seed=seed, trainer=trainer, sequence=sequence, opening=opening):
                    stats = health(trainer.model, obs)
                    probe = full_recovery_probe(
                        trainer.model,
                        obs,
                        steps=cfg["recovery_probe_steps"],
                        lr=cfg["recovery_probe_lr"],
                        critic_lr=cfg["critic_probe_lr"],
                    )
                    if not probe["passed"]:
                        repeat = full_recovery_probe(
                            trainer.model,
                            obs,
                            steps=cfg["recovery_probe_steps"],
                            lr=cfg["recovery_probe_lr"],
                            critic_lr=cfg["critic_probe_lr"],
                        )
                        if probe != repeat:
                            raise ValueError("Recovery probe is not reproducible")
                    jobs = [
                        dict(entry=e, sampler_seed=rng)
                        for e in dev
                        for rng in protocol["evaluation_rng_seeds"]
                    ]
                    _, episodes = collector.collect(
                        trainer.model, args.bank, jobs, protocol, collect=False, verify=True
                    )
                    frame = dict(
                        seed=seed,
                        additional_transitions=trainer.transitions - opening,
                        health=stats,
                        recovery=probe,
                        evaluation=summarize(episodes),
                        model_sha256=state_hash(trainer.model),
                        elapsed_seconds=time.monotonic() - started,
                    )
                    sequence.append(frame)
                    boundary = trainer.transitions - opening
                    trainer.checkpoint(args.output / f"seed-{seed}-step-{boundary}.pt")
                    (args.output / f"seed-{seed}-step-{boundary}-evaluation.json").write_text(
                        json.dumps(episodes, indent=2)
                    )
                    (args.output / f"seed-{seed}-health.json").write_text(
                        json.dumps(sequence, indent=2)
                    )
                    print(
                        json.dumps(
                            dict(
                                seed=seed,
                                transitions=boundary,
                                recovery=probe["passed"],
                                welfare=frame["evaluation"]["welfare"],
                                survival=frame["evaluation"]["survival"],
                                actor_saturation=[
                                    r["saturated_fraction"] for r in stats["actor_layers"]
                                ],
                                critic_saturation=[
                                    r["saturated_fraction"] for r in stats["critic_layers"]
                                ],
                                elapsed_seconds=frame["elapsed_seconds"],
                            )
                        ),
                        flush=True,
                    )
                    return probe["passed"]

                alive = evaluate()
                status = "completed"
                while (
                    alive and trainer.transitions - opening < cfg["additional_transitions_per_seed"]
                ):
                    progress = trainer.transitions - opening
                    target = min(
                        cfg["additional_transitions_per_seed"], ((progress // 4096) + 1) * 4096
                    )
                    begin = time.monotonic()
                    rows, summaries = collector.rollout(
                        trainer, args.bank, train, seed, target - progress, verify=not updates
                    )
                    if total + len(rows) > cfg["maximum_total_transitions"]:
                        raise ValueError("Total sample budget exceeded")
                    metric = measured_update(trainer, rows)
                    total += len(rows)
                    metric["additional_transitions"] = trainer.transitions - opening
                    metric["rollout_and_update_seconds"] = time.monotonic() - begin
                    metric["native_seconds"] = sum(s["native_seconds"] for s in summaries)
                    metric["rss_bytes"] = rss_bytes(
                        [os.getpid(), *[t._process.pid for t in collector.transports]]
                    )
                    peak_rss = max(peak_rss, metric["rss_bytes"])
                    updates.append(metric)
                    with (args.output / f"seed-{seed}-updates.jsonl").open("a") as stream:
                        stream.write(json.dumps(metric) + "\n")
                    with (args.output / f"seed-{seed}-training.jsonl").open("a") as stream:
                        for summary in summaries:
                            stream.write(json.dumps(summary) + "\n")
                    training.extend(summaries)
                    timings.append(metric["rollout_and_update_seconds"] / len(rows))
                    if len(updates) == 8:
                        baseline_rss = metric["rss_bytes"]
                    if len(updates) >= 24 and (
                        metric["rss_bytes"] > 2 * baseline_rss
                        or np.median(timings[-8:]) > 2 * np.median(timings[:8])
                    ):
                        status = "resource_growth_pause"
                        alive = False
                    # A real optimizer stall cannot be excused by a fresh-Adam recovery probe.
                    if optimizer_stall(updates):
                        status = "optimizer_stall_pause"
                        alive = False
                    progress = trainer.transitions - opening
                    if progress == target or not alive:
                        recovery_ok = evaluate()
                        if not recovery_ok:
                            status = "recovery_probe_pause"
                            alive = False
                        (args.output / f"seed-{seed}-updates.json").write_text(
                            json.dumps(updates, indent=2)
                        )
                        (args.output / f"seed-{seed}-training.json").write_text(
                            json.dumps(training, indent=2)
                        )
                if not sequence[-1]["recovery"]["passed"]:
                    status = "recovery_probe_pause"
                results.append(
                    dict(
                        seed=seed,
                        status=status,
                        additional_transitions=trainer.transitions - opening,
                        health_evaluations=len(sequence),
                        initial=sequence[0]["evaluation"],
                        final=sequence[-1]["evaluation"],
                        final_recovery=sequence[-1]["recovery"],
                        training=summarize(training) if training else {},
                        peak_rss_bytes=peak_rss,
                        mean_step_seconds=float(np.mean(timings)) if timings else None,
                    )
                )
                (args.output / "summary.json").write_text(
                    json.dumps(
                        dict(
                            config=cfg,
                            provenance=provenance,
                            seeds=results,
                            additional_transitions=total,
                            wall_seconds=time.monotonic() - started,
                        ),
                        indent=2,
                    )
                )
    except BaseException as error:
        (args.output / "failure.json").write_text(
            json.dumps(
                dict(
                    type=type(error).__name__,
                    message=str(error),
                    additional_transitions=total,
                    wall_seconds=time.monotonic() - started,
                ),
                indent=2,
            )
        )
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior_handler)
    print(
        json.dumps(
            dict(
                completed=True,
                additional_transitions=total,
                wall_seconds=time.monotonic() - started,
                statuses=[r["status"] for r in results],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
