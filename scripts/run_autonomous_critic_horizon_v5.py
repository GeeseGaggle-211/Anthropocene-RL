"""Fresh, hard-capped native 2x2 critic-architecture/physical-horizon canary."""

import argparse
import copy
import json
import os
import signal
import time
from pathlib import Path

import torch
from preflight_autonomous_v4_residual import composite_health, fork_v4, retained_critic_test, sha
from run_autonomous_continuation_v4 import measured_update, rss_bytes
from run_autonomous_ppo_gate import summarize

from anthropocene_rl.autonomous_v4.collection import Collector
from anthropocene_rl.autonomous_v4.diagnostics import (
    full_recovery_probe,
    optimizer_stall,
    probe_face,
)
from anthropocene_rl.autonomous_v4.training import state_hash


def validate_gate(cfg, previous):
    if (
        cfg["variants"] != ["small_step", "residual_layernorm"]
        or cfg["horizons"] != [100, 400]
        or cfg["seeds"] != [29, 43]
    ):
        raise ValueError("Frozen factorial changed")
    requested = (
        len(cfg["variants"])
        * len(cfg["horizons"])
        * len(cfg["seeds"])
        * cfg["transitions_per_cell"]
    )
    if (
        requested != cfg["maximum_training_transitions"]
        or requested != 32768
        or cfg["transitions_per_cell"] != 4096
    ):
        raise ValueError("V5 native allowance mismatch")
    if (
        previous["cumulative_transitions"] != cfg["prior_additional_transitions"]
        or cfg["prior_additional_transitions"] != 98304
    ):
        raise ValueError("Historical native use mismatch")
    if (
        requested + previous["cumulative_transitions"] != cfg["maximum_cumulative_additional"]
        or cfg["maximum_cumulative_additional"] != 131072
    ):
        raise ValueError("V5 cumulative cap mismatch")
    if sorted(r["seed"] for r in previous["seeds"]) != [29, 43] or any(
        r["status"] != "completed" for r in previous["seeds"]
    ):
        raise ValueError("Incomplete v4 native parent campaign")
    if cfg["maximum_wall_seconds"] != 2400 or cfg["workers"] != 4 or cfg["torch_threads"] != 1:
        raise ValueError("Frozen resource envelope changed")
    return requested


def critic_probe_subset(obs):
    selected, labels, _, _ = probe_face(obs)
    indices = torch.cat(
        [
            torch.linspace(0, len(labels) // 2 - 1, 16).round().long(),
            torch.linspace(len(labels) // 2, len(labels) - 1, 16).round().long(),
        ]
    )
    return {k: v[indices] for k, v in selected.items()}, labels[indices]


def critic_return_diagnostics(prediction, returns, residual_mse):
    error = prediction - returns
    variance = float(returns.var(unbiased=False))
    denominator = max(1e-12, variance)
    return dict(
        critic_bias=float(error.mean()),
        explained_variance=1 - float(error.var(unbiased=False)) / denominator,
        return_variance=variance,
        near_constant_returns=variance < 1e-12,
        variance_denominator=denominator,
        return_mean=float(returns.mean()),
        return_std=float(returns.std(unbiased=False)),
        critic_mse_over_return_variance=residual_mse / denominator,
    )


def health_extremes(cells):
    """Per architecture, select technical extremes, never economic score."""
    result = []
    for variant in ("small_step", "residual_layernorm"):
        candidates = [
            (cell, frame)
            for cell in cells
            if cell["variant"] == variant
            for frame in cell["health"]
        ]
        if not candidates:
            continue
        for kind in ("categorical_confidence", "actor_layer_saturation"):

            def score(pair, kind=kind):
                stats = pair[1]["health"]
                if kind == "actor_layer_saturation":
                    return max(layer["saturated_fraction"] for layer in stats["actor_layers"])
                return max(
                    (
                        head["maximum_probability"]
                        for head in stats["categorical"].values()
                        if head["eligible"] > 0
                    ),
                    default=-1.0,
                )

            # Stable candidate order gives cell order/earliest boundary tie-breaking.
            cell, frame = max(candidates, key=score)
            result.append(
                dict(
                    variant=variant,
                    reason=kind,
                    cell=cell["id"],
                    transitions=frame["transitions"],
                    checkpoint=frame["checkpoint"],
                    score=score((cell, frame)),
                )
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    for name, pin in cfg["inputs"].items():
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError(f"Pinned input changed: {name}")
    previous = json.loads(Path(cfg["inputs"]["prior_campaign"]["path"]).read_text())
    validate_gate(cfg, previous)
    preflight = json.loads(Path(cfg["inputs"]["preflight"]["path"]).read_text())
    if not preflight["complete"] or not preflight["passed"]:
        raise ValueError("Residual preflight did not pass")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    bank = Path(cfg["inputs"]["bank"]["path"]).parent
    entries = json.loads((bank / "manifest.json").read_text())["entries"]
    train = [e for e in entries if e["partition"] == "train"]
    dev = [e for e in entries if e["partition"] == "development"]
    assert {e["world_seed"] for e in train}.isdisjoint({e["world_seed"] for e in dev})
    obs = torch.load(cfg["inputs"]["probe"]["path"], weights_only=True)
    critic_obs, labels = critic_probe_subset(obs)
    started = time.monotonic()
    deadline = started + 2400
    result = dict(
        version="autonomous-critic-horizon-gate-v5",
        config=cfg,
        config_sha256=sha(args.config),
        cells=[],
        training_transitions=0,
        reserved_inflight_transitions=0,
        evaluation_steps=0,
        replay_steps=0,
        reserved_evaluation_resolves=0,
        prior_additional_transitions=98304,
        cumulative_additional_transitions=98304,
        status="running",
    )

    def save():
        result["selected_extremes"] = health_extremes(result["cells"])
        result["wall_seconds"] = time.monotonic() - started
        result["budget_consumed_transitions"] = (
            result["training_transitions"] + result["reserved_inflight_transitions"]
        )
        result["cumulative_additional_transitions"] = 98304 + result["budget_consumed_transitions"]
        result["native_resolves_accounted"] = (
            result["budget_consumed_transitions"]
            + result["evaluation_steps"]
            + result["replay_steps"]
            + result["reserved_evaluation_resolves"]
        )
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    def expired(*_):
        raise TimeoutError("V5 native campaign deadline")

    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(2400)
    try:
        with Collector(
            [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]],
            workers=4,
            deadline=deadline,
            request_timeout=60,
        ) as collector:
            for variant in cfg["variants"]:
                for horizon in cfg["horizons"]:
                    for seed in cfg["seeds"]:
                        parent = Path(cfg["inputs"][f"parent_{seed}"]["path"])
                        cell_id = f"{variant}-h{horizon}-seed{seed}"
                        provenance = dict(
                            gate_config_sha256=sha(args.config),
                            parent_checkpoint_sha256=sha(parent),
                            bank_sha256=cfg["inputs"]["bank"]["sha256"],
                            bridge_sha256=cfg["inputs"]["bridge"]["sha256"],
                            prior_additional_transitions=98304,
                            native_gate_allowance=32768,
                            critic_variant=variant,
                            horizon_years=horizon,
                            training_seed=seed,
                        )
                        trainer = fork_v4(parent, variant, horizon, provenance)
                        opening = trainer.transitions
                        anchors = {
                            n: p.detach().clone()
                            for n, p in trainer.model.named_parameters()
                            if not p.requires_grad
                        }
                        cell = dict(
                            id=cell_id,
                            variant=variant,
                            horizon=horizon,
                            seed=seed,
                            status="running",
                            transitions=0,
                            health=[],
                            updates=[],
                            episodes=[],
                            parent_sha256=sha(parent),
                        )
                        result["cells"].append(cell)
                        baseline_rss = None
                        step_timings = []

                        def evaluate(
                            trainer=trainer, opening=opening, cell_id=cell_id, cell=cell, seed=seed
                        ):
                            count = trainer.transitions - opening
                            checkpoint = args.output / f"{cell_id}-step-{count}.pt"
                            if checkpoint.exists():
                                raise ValueError("Refuse checkpoint overwrite")
                            trainer.checkpoint(checkpoint)
                            stats = composite_health(trainer.model, obs)
                            fresh = full_recovery_probe(
                                trainer.model, obs, steps=16, lr=0.001, critic_lr=0.0001
                            )
                            if not fresh["passed"]:
                                assert fresh == full_recovery_probe(
                                    trainer.model, obs, steps=16, lr=0.001, critic_lr=0.0001
                                )
                            retained = retained_critic_test(
                                copy.deepcopy(trainer), critic_obs, labels
                            )
                            if not retained["passed"]:
                                repeated = retained_critic_test(
                                    copy.deepcopy(trainer), critic_obs, labels
                                )
                                assert repeated == retained
                            jobs = [
                                dict(entry=e, sampler_seed=rng)
                                for e in dev
                                for rng in trainer.config["evaluation_rng_seeds"]
                            ]
                            result["reserved_evaluation_resolves"] = (
                                2 * len(jobs) * trainer.config["episode_limit"]
                            )
                            save()
                            _, episodes = collector.collect(
                                trainer.model,
                                bank,
                                jobs,
                                trainer.config,
                                collect=False,
                                verify=True,
                            )
                            observed_steps = sum(round(e["elapsed_years"] / 20) for e in episodes)
                            result["evaluation_steps"] += observed_steps
                            result["replay_steps"] += observed_steps
                            result["reserved_evaluation_resolves"] = 0
                            if (
                                result["evaluation_steps"] > cfg["maximum_evaluation_steps"]
                                or result["replay_steps"] > cfg["maximum_replay_steps"]
                            ):
                                raise ValueError("V5 read-only native allowance exceeded")
                            (args.output / f"{cell_id}-step-{count}-evaluation.json").write_text(
                                json.dumps(episodes, indent=2) + "\n"
                            )
                            if count == 0:
                                original = json.loads(
                                    Path(
                                        cfg["inputs"][f"parent_evaluation_{seed}"]["path"]
                                    ).read_text()
                                )
                                assert [(r["fingerprint"], r["welfare"]) for r in episodes] == [
                                    (r["fingerprint"], r["welfare"]) for r in original
                                ]
                            frame = dict(
                                transitions=count,
                                checkpoint=dict(path=str(checkpoint), sha256=sha(checkpoint)),
                                model_sha256=state_hash(trainer.model),
                                health=stats,
                                fresh_recovery=fresh,
                                retained_critic=retained,
                                evaluation=summarize(episodes),
                                passed=fresh["passed"] and retained["passed"],
                            )
                            cell["health"].append(frame)
                            cell["transitions"] = count
                            save()
                            print(
                                json.dumps(
                                    dict(
                                        cell=cell_id,
                                        transitions=count,
                                        passed=frame["passed"],
                                        welfare=frame["evaluation"]["welfare"],
                                        retained_critic_mse=retained["final_mse"],
                                    )
                                ),
                                flush=True,
                            )
                            if not frame["passed"]:
                                cell["status"] = "adaptation_failed"
                                raise RuntimeError(
                                    f"Reproducible recovery failure: {cell_id}/{count}"
                                )

                        evaluate()
                        while trainer.transitions - opening < 4096:
                            batch_start = time.perf_counter()
                            next_boundary = 2048 if trainer.transitions - opening < 2048 else 4096
                            remaining = next_boundary - (trainer.transitions - opening)
                            result["reserved_inflight_transitions"] = min(
                                remaining,
                                trainer.config["episode_limit"]
                                * trainer.config["episodes_per_rollout"],
                            )
                            save()
                            rows, episodes = collector.rollout(
                                trainer, bank, train, seed, remaining
                            )
                            used = len(rows)
                            result["training_transitions"] += used
                            result["reserved_inflight_transitions"] = 0
                            if result["training_transitions"] > 32768:
                                raise ValueError("V5 native budget exhausted")
                            cell["episodes"].extend(episodes)
                            returns = torch.tensor([r["return"] for r in rows])
                            with torch.no_grad():
                                observed = {
                                    k: torch.cat([r["obs"][k] for r in rows])
                                    for k in rows[0]["obs"]
                                }
                                prediction = trainer.model.value(observed)
                            metrics = measured_update(trainer, rows)
                            metrics.update(
                                critic_return_diagnostics(
                                    prediction, returns, metrics["critic_residual_mse"]
                                )
                            )
                            params = dict(trainer.model.named_parameters())
                            if not all(
                                torch.equal(value, params[name]) and params[name].grad is None
                                for name, value in anchors.items()
                            ):
                                raise ValueError("Frozen anchor changed")
                            cell["updates"].append(
                                dict(transitions=trainer.transitions - opening, **metrics)
                            )
                            cell["transitions"] = trainer.transitions - opening
                            elapsed = time.perf_counter() - batch_start
                            step_timings.append(elapsed / used)
                            rss = rss_bytes(
                                [
                                    os.getpid(),
                                    *(t._process.pid for t in collector.transports),
                                ]
                            )
                            cell["peak_rss_bytes"] = max(cell.get("peak_rss_bytes", 0), rss)
                            baseline_rss = rss if baseline_rss is None else baseline_rss
                            if rss > cfg["maximum_rss_bytes"] or (
                                len(cell["updates"]) > 16 and rss > baseline_rss * 1.5
                            ):
                                raise RuntimeError("V5 resource growth guard")
                            if optimizer_stall(cell["updates"]):
                                raise RuntimeError("V5 retained optimizer stall")
                            if trainer.transitions - opening in (2048, 4096):
                                evaluate()
                        cell["status"] = "completed"
                        cell["mean_step_seconds"] = sum(step_timings) / len(step_timings)
                        assert sha(parent) == cell["parent_sha256"]
                        save()
        result["status"] = "completed"
        result["complete"] = len(result["cells"]) == 8 and result["training_transitions"] == 32768
    except BaseException as error:
        result.update(status="failed", complete=False, error=f"{type(error).__name__}: {error}")
        if result["cells"]:
            cell["status"] = "failed"
            try:
                trainer.checkpoint(args.output / f"{cell['id']}-failed.pt")
            except Exception as checkpoint_error:
                result["checkpoint_error"] = str(checkpoint_error)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        save()


if __name__ == "__main__":
    main()
