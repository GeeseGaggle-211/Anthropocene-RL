"""Bounded critic-only attribution; no native steps, actor updates or saved policy."""

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from anthropocene_rl.autonomous_v4.training import state_hash
from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(parameters, *, gradients=False):
    return torch.cat([(p.grad if gradients else p.detach()).flatten() for p in parameters])


@contextmanager
def preserve_evidence(result, output):
    try:
        yield
    except BaseException as error:
        result.update(complete=False, status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never overwrite evidence")
    cfg = json.loads(args.config.read_text())
    if cfg["learning_rates"] != [0.0003, 0.0001, 0.00003] or cfg["updates"] != 64:
        raise ValueError("Unexpected attribution envelope")
    if cfg["maximum_wall_seconds"] != 60:
        raise ValueError("Unexpected deadline")
    for item in cfg["inputs"].values():
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Input digest mismatch: {item['path']}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = Path(cfg["inputs"]["checkpoint"]["path"])
    state = torch.load(checkpoint, weights_only=False)
    report = json.loads(Path(cfg["inputs"]["diagnostic"]["path"]).read_text())
    task = report["tasks"]["activity_4"]
    duplicate = report["tasks"]["activity_9"]
    assert task["indices"] == duplicate["indices"] and task["labels"] == duplicate["labels"]
    assert [x["critic_mse"] for x in task["updates"]] == [
        x["critic_mse"] for x in duplicate["updates"]
    ]
    probe = torch.load(cfg["inputs"]["probe"]["path"], weights_only=True)
    obs = {k: v[task["indices"]] for k, v in probe.items()}
    labels = torch.tensor(task["labels"])
    assert len(labels) == 32 and state["config"]["epochs"] == 4
    assert state["config"]["minibatch"] == 32
    started = time.monotonic()
    result = dict(
        version="autonomous-critic-step-attribution-v1",
        config=cfg,
        config_sha256=digest(args.config),
        native_steps=0,
        action_draws=0,
        actor_updates=0,
        cases=[],
        promotion_allowed=False,
    )
    with preserve_evidence(result, args.output):
        for lr in cfg["learning_rates"]:
            trainer = AutonomousTrainerV3(state["config"], 0, state["provenance"], state["width"])
            trainer.restore(checkpoint)
            assert trainer.optimizer.param_groups[1]["lr"] == 0.0003
            trainer.optimizer.param_groups[1]["lr"] = lr
            actor_before = state_hash(trainer.model.actor)
            parameters = list(trainer.model.critic.parameters())
            with torch.no_grad():
                target = trainer.model.value(obs).detach() + labels
            updates = []
            case = dict(learning_rate=lr, original_trace_exact=False, updates=updates)
            result["cases"].append(case)
            for update in range(cfg["updates"]):
                minibatches = []
                record = dict(update=update + 1, minibatches=minibatches)
                updates.append(record)
                for _ in range(trainer.config["epochs"]):
                    for ix in torch.randperm(len(labels), generator=trainer.shuffle).split(
                        trainer.config["minibatch"]
                    ):
                        if time.monotonic() - started >= cfg["maximum_wall_seconds"]:
                            raise TimeoutError("Critic attribution deadline")
                        selected = {k: v[ix] for k, v in obs.items()}
                        value = trainer.model.value(selected)
                        residual = value - target[ix]
                        loss = 0.5 * residual.square().mean()
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite critic loss")
                        trainer.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        gradient = vector(parameters, gradients=True).clone()
                        norm = torch.nn.utils.clip_grad_norm_(
                            parameters, 0.5, error_if_nonfinite=True
                        )
                        clipped_norm = float(vector(parameters, gradients=True).norm())
                        before = vector(parameters).clone()
                        trainer.optimizer.step()
                        delta = vector(parameters) - before
                        with torch.no_grad():
                            after_value = trainer.model.value(selected)
                            after_residual = after_value - target[ix]
                        minibatches.append(
                            dict(
                                residual_mean_before=float(residual.detach().mean()),
                                mse_before=float(residual.detach().square().mean()),
                                residual_mean_after=float(after_residual.mean()),
                                mse_after=float(after_residual.square().mean()),
                                gradient_norm=float(norm),
                                clipped_gradient_norm=clipped_norm,
                                parameter_step_norm=float(delta.norm()),
                                prediction_step_norm=float((after_value - value.detach()).norm()),
                                gradient_step_cosine=float(
                                    torch.nn.functional.cosine_similarity(
                                        gradient[None], delta[None]
                                    )[0]
                                ),
                            )
                        )
                with torch.no_grad():
                    mse = float((trainer.model.value(obs) - target).square().mean())
                record["critic_mse"] = mse
            assert actor_before == state_hash(trainer.model.actor)
            # Exact reproduction is required before attributing a learning-rate effect.
            if lr == 0.0003:
                case["original_trace_exact"] = [x["critic_mse"] for x in updates] == [
                    x["critic_mse"] for x in task["updates"]
                ]
                if not case["original_trace_exact"]:
                    raise AssertionError(
                        "Critic-only replay differs from original PPO critic trace"
                    )
            case.update(
                dict(
                    learning_rate=lr,
                    original_trace_exact=lr == 0.0003,
                    final_mse=updates[-1]["critic_mse"],
                    minimum_mse=min(x["critic_mse"] for x in updates),
                    maximum_last_eight_mse=max(x["critic_mse"] for x in updates[-8:]),
                    actor_unchanged=True,
                    updates=updates,
                )
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        assert digest(checkpoint) == cfg["inputs"]["checkpoint"]["sha256"]
        result.update(
            complete=True, wall_seconds=time.monotonic() - started, checkpoint_unchanged=True
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in c.items() if k != "updates"} for c in result["cases"]]))


if __name__ == "__main__":
    main()
