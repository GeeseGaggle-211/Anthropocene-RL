"""Exact value transfer and retained-critic adaptation on original TRAIN targets."""

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import torch

from anthropocene_rl.autonomous_v4.diagnostics import _layers, health
from anthropocene_rl.autonomous_v4.training import state_hash
from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3
from anthropocene_rl.training.autonomous_ppo_v4 import AutonomousTrainerV4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_config(parent_config, variant, horizon=100):
    return dict(
        parent_config,
        version="autonomous-ppo-critic-v4",
        model_version="autonomous-ppo-model-v4",
        critic_variant=variant,
        critic_lr=0.0001,
        horizon_years=horizon,
    )


def restore_v3(path):
    state = torch.load(path, weights_only=False)
    trainer = AutonomousTrainerV3(state["config"], 0, state["provenance"], state["width"])
    trainer.restore(path)
    return trainer


def fork_v4(path, variant, horizon=100, provenance=None):
    state = torch.load(path, weights_only=False)
    config = candidate_config(state["config"], variant, horizon)
    trainer = AutonomousTrainerV4(config, 0, provenance or state["provenance"], state["width"])
    trainer.warm_start(path, expected_sha256=sha(path))
    return trainer


def composite_health(model, obs):
    result = health(model, obs)
    result["adaptive_critic_layers"] = result.pop("critic_layers")
    with torch.enable_grad():
        x = obs["x"].clone().requires_grad_()
        adaptive = model.critic(x).squeeze(-1)
        derivative = torch.autograd.grad(adaptive.sum(), x)[0]
    result["adaptive_value_input_gradient"] = float(derivative.norm(dim=-1).mean())
    if model.critic_variant == "residual_layernorm":
        result["frozen_reference_layers"] = _layers(model.reference_critic[0], obs["x"])
        result["frozen_initial_layers"] = _layers(model.initial_normalized_critic[0], obs["x"])
        result["layernorm_scale_max"] = max(
            float(layer.weight.detach().abs().max())
            for layer in model.critic[0]
            if isinstance(layer, torch.nn.LayerNorm)
        )
        with torch.no_grad():
            result["residual_rms"] = float(
                (model.critic(obs["x"]) - model.initial_normalized_critic(obs["x"]))
                .square()
                .mean()
                .sqrt()
            )
    return result


def retained_critic_test(trainer, obs, labels, *, steps=64, target=None):
    """Disposable current-value-plus-label diagnostic with actual populated Adam."""
    before_actor, before_heads = state_hash(trainer.model.actor), state_hash(trainer.model.heads)
    frozen = {
        name: p.detach().clone()
        for name, p in trainer.model.named_parameters()
        if not p.requires_grad
    }
    with torch.no_grad():
        target = trainer.model.value(obs).detach() + labels if target is None else target.detach()
    trace = []
    for _ in range(steps):
        for _ in range(trainer.config["epochs"]):
            for ix in torch.randperm(len(labels), generator=trainer.shuffle).split(
                trainer.config["minibatch"]
            ):
                value = trainer.model.value({k: v[ix] for k, v in obs.items()})
                loss = 0.5 * (value - target[ix]).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite retained critic diagnostic")
                trainer.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.critic.parameters(), 0.5, error_if_nonfinite=True
                )
                trainer.optimizer.step()
        with torch.no_grad():
            trace.append(float((trainer.model.value(obs) - target).square().mean()))
    assert before_actor == state_hash(trainer.model.actor) and before_heads == state_hash(
        trainer.model.heads
    )
    params = dict(trainer.model.named_parameters())
    assert all(
        torch.equal(value, params[name]) and params[name].grad is None
        for name, value in frozen.items()
    )
    return dict(
        final_mse=trace[-1],
        maximum_last_eight_mse=max(trace[-8:]),
        passed=trace[-1] < 0.9 and max(trace[-8:]) < 0.9,
        mse_trace=trace,
        actor_heads_anchors_unchanged=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    assert cfg["maximum_wall_seconds"] == 300 and cfg["maximum_targets"] == 112
    for pin in cfg["inputs"].values():
        assert sha(pin["path"]) == pin["sha256"]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    obs = torch.load(cfg["inputs"]["probe"]["path"], weights_only=True)
    metadata = json.loads(Path(cfg["inputs"]["probe_metadata"]["path"]).read_text())
    bank = json.loads(Path(cfg["inputs"]["bank"]["path"]).read_text())
    by_id = {e["id"]: e for e in bank["entries"]}
    assert len(obs["x"]) == metadata["rows"] == len(metadata["episodes"]) * 8 == 192
    splits = {"construction_train": [], "disjoint_validation_train": []}
    for i, entry in enumerate(metadata["episodes"]):
        source = by_id[entry["id"]]
        assert source["partition"] == "train"
        split = (
            "construction_train"
            if source["world_seed"] in (831001, 831002)
            else "disjoint_validation_train"
        )
        if split == "disjoint_validation_train":
            assert source["world_seed"] == 831003
        splits[split].extend(range(i * 8, (i + 1) * 8))
    assert (
        len(splits["construction_train"]) == 128 and len(splits["disjoint_validation_train"]) == 64
    )
    comparison = json.loads(Path(cfg["inputs"]["original_targets"]["path"]).read_text())
    assert comparison["complete"] and len(comparison["targets"]) == 56
    started = time.monotonic()
    old_handler = signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(TimeoutError("Residual preflight deadline")),
    )
    signal.alarm(300)
    result = dict(
        version="autonomous-residual-preflight-v4",
        config=cfg,
        config_sha256=sha(args.config),
        fitting_updates=0,
        native_steps=0,
        action_draws=0,
        transfer=[],
        adaptation=[],
    )
    try:
        paths = sorted({t["checkpoint"]["path"] for t in comparison["targets"]})
        for path in paths:
            parent = restore_v3(path)
            for variant in ("small_step", "residual_layernorm"):
                trainer = fork_v4(path, variant)
                row = dict(checkpoint=path, checkpoint_sha256=sha(path), variant=variant, splits={})
                for name, indices in splits.items():
                    subset = {k: v[indices] for k, v in obs.items()}
                    with torch.no_grad():
                        previous, current = parent.model.value(subset), trainer.model.value(subset)
                    assert torch.equal(previous, current)
                    row["splits"][name] = dict(
                        rows=len(indices), rms_shift=0.0, maximum_shift=0.0, exact=True
                    )
                assert torch.equal(parent.sampler.get_state(), trainer.sampler.get_state())
                assert torch.equal(parent.shuffle.get_state(), trainer.shuffle.get_state())
                row["health"] = composite_health(trainer.model, obs)
                row["parameter_bytes"] = sum(
                    p.numel() * p.element_size() for p in trainer.model.parameters()
                )
                # Composite value-forward timing needs no policy sampling or optimization.
                with torch.no_grad():
                    for model in (parent.model, trainer.model):
                        for _ in range(4):
                            model.value(obs)
                    timings = []
                    for model in (parent.model, trainer.model):
                        opening = time.perf_counter()
                        for _ in range(64):
                            model.value(obs)
                        timings.append((time.perf_counter() - opening) / 64)
                row["value_seconds_parent_candidate"] = timings
                row["value_time_ratio"] = timings[1] / timings[0]
                result["transfer"].append(row)
        for original in comparison["targets"]:
            path = original["checkpoint"]["path"]
            assert sha(path) == original["checkpoint"]["sha256"]
            subset = {k: v[original["indices"]] for k, v in obs.items()}
            labels = torch.tensor(original["labels"])
            parent = restore_v3(path)
            with torch.no_grad():
                original_target = parent.model.value(subset).detach() + labels
            for variant in ("small_step", "residual_layernorm"):
                trainer = fork_v4(path, variant)
                result["adaptation"].append(
                    dict(
                        checkpoint=path,
                        tasks=original["tasks"],
                        variant=variant,
                        **retained_critic_test(trainer, subset, labels, target=original_target),
                    )
                )
                assert len(result["adaptation"]) <= 112
            assert sha(path) == original["checkpoint"]["sha256"]
        result.update(
            complete=True,
            wall_seconds=time.monotonic() - started,
            passed=all(t["passed"] for t in result["adaptation"])
            and all(t["value_time_ratio"] < 4.5 for t in result["transfer"]),
            critic_minibatches=112 * 64 * 4,
        )
    except BaseException as error:
        result.update(complete=False, passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"config", "transfer", "adaptation"}}
        )
    )


if __name__ == "__main__":
    main()
