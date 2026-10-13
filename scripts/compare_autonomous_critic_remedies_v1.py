"""Discarded critic-only replicas; frozen original targets and retained Adam."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from audit_autonomous_critic_layernorm_v1 import digest, layernorm_copy
from diagnose_autonomous_critic_step_v1 import preserve_evidence

from anthropocene_rl.autonomous_v4.training import state_hash
from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3

VARIANTS = {
    "original": (0.0003, False),
    "lr_1e4": (0.0001, False),
    "lr_3e5": (0.00003, False),
    "ln_lr_1e4": (0.0001, True),
}


def convert(trainer, lr, layernorm):
    """Every old named parameter/moment survives; new LN affine states start empty."""
    if layernorm:
        old = trainer.model
        model = layernorm_copy(old)
        optimizer = torch.optim.Adam(
            [
                {"params": [*model.actor.parameters(), *model.heads.parameters()]},
                {"params": list(model.critic.parameters())},
            ],
            eps=1e-5,
        )
        for group, previous in zip(
            optimizer.param_groups, trainer.optimizer.param_groups, strict=True
        ):
            group.update({k: copy.deepcopy(v) for k, v in previous.items() if k != "params"})
        names = dict(model.named_parameters())
        for name, parameter in old.named_parameters():
            if parameter in trainer.optimizer.state:
                optimizer.state[names[name]] = copy.deepcopy(trainer.optimizer.state[parameter])
                for key, value in trainer.optimizer.state[parameter].items():
                    actual = optimizer.state[names[name]][key]
                    assert (
                        torch.equal(value, actual)
                        if isinstance(value, torch.Tensor)
                        else value == actual
                    )
        for name, parameter in model.named_parameters():
            if ".norm" in name and name.startswith("critic."):
                assert parameter not in optimizer.state
        trainer.model, trainer.optimizer = model, optimizer
    trainer.optimizer.param_groups[1]["lr"] = lr


def unique_targets(report):
    groups = {}
    for name, task in report["tasks"].items():
        if task["status"] != "completed" or len(task["updates"]) != 64:
            raise ValueError("Incomplete original task")
        key = (tuple(task["indices"]), tuple(task["labels"]))
        if key not in groups:
            groups[key] = dict(tasks=[], task=task)
        else:
            assert [u["critic_mse"] for u in task["updates"]] == [
                u["critic_mse"] for u in groups[key]["task"]["updates"]
            ]
        groups[key]["tasks"].append(name)
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never overwrite evidence")
    cfg = json.loads(args.config.read_text())
    assert cfg["variants"] == list(VARIANTS) and cfg["updates"] == 64
    assert cfg["maximum_wall_seconds"] == 300 and len(cfg["checkpoints"]) == 6
    assert cfg["final_mse_limit"] == cfg["maximum_last_eight_mse_limit"] == 0.9
    for item in cfg["inputs"].values():
        if digest(item["path"]) != item["sha256"]:
            raise ValueError(f"Input digest mismatch: {item['path']}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    probe = torch.load(cfg["inputs"]["probe"]["path"], weights_only=True)
    started = time.monotonic()
    result = dict(
        version="autonomous-critic-remedies-v1",
        config=cfg,
        config_sha256=digest(args.config),
        native_steps=0,
        action_draws=0,
        actor_updates=0,
        promotion_allowed=False,
        targets=[],
    )
    with preserve_evidence(result, args.output):
        for entry in cfg["checkpoints"]:
            for item in (entry["checkpoint"], entry["diagnostic"]):
                if digest(item["path"]) != item["sha256"]:
                    raise ValueError("Checkpoint or diagnostic digest mismatch")
            checkpoint = Path(entry["checkpoint"]["path"])
            report = json.loads(Path(entry["diagnostic"]["path"]).read_text())
            assert len(report["tasks"]) == 13 and not report.get("partial", False)
            state = torch.load(checkpoint, weights_only=False)
            for group in unique_targets(report):
                task = group["task"]
                obs = {k: v[task["indices"]] for k, v in probe.items()}
                labels = torch.tensor(task["labels"])
                assert len(labels) == 32 and state["config"]["epochs"] == 4
                assert state["config"]["minibatch"] == 32
                target_record = dict(
                    checkpoint=entry["checkpoint"],
                    tasks=group["tasks"],
                    indices=task["indices"],
                    labels=task["labels"],
                    variants={},
                )
                result["targets"].append(target_record)
                for name, (lr, ln) in VARIANTS.items():
                    trainer = AutonomousTrainerV3(
                        state["config"], 0, state["provenance"], state["width"]
                    )
                    trainer.restore(checkpoint)
                    with torch.no_grad():
                        # Common ORIGINAL target; never rebase to the normalized critic.
                        target = trainer.model.value(obs).detach() + labels
                    actor_before, heads_before = (
                        state_hash(trainer.model.actor),
                        state_hash(trainer.model.heads),
                    )
                    convert(trainer, lr, ln)
                    with torch.no_grad():
                        entry_mse = float((trainer.model.value(obs) - target).square().mean())
                    records = []
                    variant = dict(entry_mse=entry_mse, mse_trace=records, passed=False)
                    target_record["variants"][name] = variant
                    for _ in range(64):
                        for _ in range(4):
                            if time.monotonic() - started >= 300:
                                raise TimeoutError("Critic comparison deadline")
                            ix = torch.randperm(32, generator=trainer.shuffle)
                            value = trainer.model.value({k: v[ix] for k, v in obs.items()})
                            loss = 0.5 * (value - target[ix]).square().mean()
                            if not torch.isfinite(loss):
                                raise ValueError("Nonfinite critic loss")
                            trainer.optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                trainer.model.critic.parameters(), 0.5, error_if_nonfinite=True
                            )
                            trainer.optimizer.step()
                        with torch.no_grad():
                            records.append(
                                float((trainer.model.value(obs) - target).square().mean())
                            )
                    assert actor_before == state_hash(trainer.model.actor)
                    assert heads_before == state_hash(trainer.model.heads)
                    if name == "original":
                        variant["original_trace_exact"] = records == [
                            u["critic_mse"] for u in task["updates"]
                        ]
                        if not variant["original_trace_exact"]:
                            raise AssertionError("Original critic trace was not reproduced exactly")
                    variant.update(
                        final_mse=records[-1],
                        maximum_last_eight_mse=max(records[-8:]),
                        passed=records[-1] < 0.9 and max(records[-8:]) < 0.9,
                        actor_and_heads_exact=True,
                    )
                assert digest(checkpoint) == entry["checkpoint"]["sha256"]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        result.update(
            complete=True,
            wall_seconds=time.monotonic() - started,
            unique_targets=len(result["targets"]),
            original_tasks=sum(len(t["tasks"]) for t in result["targets"]),
            critic_minibatches=len(result["targets"]) * 4 * 64 * 4,
        )
        result["all_targets_pass"] = {
            name: all(t["variants"][name]["passed"] for t in result["targets"]) for name in VARIANTS
        }
    print(json.dumps({k: v for k, v in result.items() if k not in {"config", "targets"}}))


if __name__ == "__main__":
    main()
