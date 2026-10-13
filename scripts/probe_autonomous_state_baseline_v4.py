"""Versioned retained-Adam reversal tests for the composite v4 critic."""

import argparse
import json
import time
from pathlib import Path

import torch
from preflight_autonomous_v4_residual import sha
from probe_autonomous_state_baseline import (
    TASKS,
    TRADE_INDEX,
    enable_mechanism,
    expected_reward,
    probabilities,
    task_face,
)

from anthropocene_rl.autonomous_v4.training import state_hash
from anthropocene_rl.training.autonomous_ppo_v4 import AutonomousTrainerV4


def make_trainer(checkpoint):
    state = torch.load(checkpoint, weights_only=False)
    trainer = AutonomousTrainerV4(state["config"], 0, state["provenance"], width=state["width"])
    trainer.restore(checkpoint)
    return trainer


def run_task(checkpoint, obs, task, *, updates=64, deadline):
    trainer = make_trainer(checkpoint)
    original = state_hash(trainer.model)
    anchors = {
        n: p.detach().clone() for n, p in trainer.model.named_parameters() if not p.requires_grad
    }
    try:
        selected, mask, feature, indices = task_face(obs, task)
    except ValueError as error:
        return dict(
            task=task,
            status="insufficient_face",
            covered=False,
            passed=False,
            draws=0,
            updates=[],
            message=str(error),
        )
    choices = torch.where(mask)[0][:2]
    labels = torch.ones(len(indices))
    labels[: len(indices) // 2] = -1
    p = probabilities(trainer.model, selected, task, mask)
    gap = float(((p[:, choices[0]] - p[:, choices[1]]) * labels).mean())
    if gap > 0:
        labels = -labels  # Same task orientation for all architecture variants.
        gap = -gap
    initial_gap = gap
    with torch.no_grad():
        target_value = trainer.model.value(selected).detach() + labels
    initial_mechanism = enable_mechanism(trainer.model, selected, labels)
    records = []
    status = "completed"
    for step in range(updates):
        if time.monotonic() >= deadline:
            status = "time_limit"
            break
        with torch.no_grad():
            action, logp, _, _ = trainer.model(selected, generator=trainer.sampler)
            active = torch.ones(len(indices), dtype=torch.bool)
            if task == "workforce":
                sampled = action.workforce
            elif task.startswith("activity_"):
                sampled = action.activities[:, int(task.split("_")[1])]
            elif task.startswith("control_"):
                fraction = action.controls[:, int(task.split("_")[1])]
                sampled = torch.stack([fraction, 1 - fraction], -1)
            else:
                sampled = None
            if sampled is not None:
                reward = (sampled[:, choices[0]] - sampled[:, choices[1]]) * labels
                matching = None
            else:
                picked = (
                    action.movement if task == "movement" else action.trade[:, TRADE_INDEX[task]]
                )
                if task in TRADE_INDEX and task != "enabled":
                    active = action.trade[:, 0].bool()
                if task == "want":
                    # Both targets must be legal under the actually sampled offered good.
                    active = (
                        active
                        & (action.trade[:, 2] != choices[0])
                        & (action.trade[:, 2] != choices[1])
                    )
                reward = (
                    ((picked == choices[0]).float() - (picked == choices[1]).float())
                    * labels
                    * active
                )
                matching = int(((reward > 0) & active).sum())
        baseline = expected_reward(trainer.model, selected, task, mask, choices, labels)
        rows = [
            dict(
                obs={k: v[i : i + 1] for k, v in selected.items()},
                action=action.index(slice(i, i + 1)),
                logp=logp[i : i + 1],
                advantage=float(reward[i] - baseline[i]),
                **{"return": float(target_value[i])},
            )
            for i in range(len(indices))
        ]
        metrics = trainer.update(rows)
        params = dict(trainer.model.named_parameters())
        assert all(
            torch.equal(value, params[name]) and params[name].grad is None
            for name, value in anchors.items()
        )
        p = probabilities(trainer.model, selected, task, mask)
        with torch.no_grad():
            mse = float((trainer.model.value(selected) - target_value).square().mean())
        records.append(
            dict(
                update=step + 1,
                reward_std=float(reward.std(unbiased=False)),
                target_draws=matching,
                active_draws=int(active.sum()),
                active_positive_label=int((active & (labels > 0)).sum()),
                active_negative_label=int((active & (labels < 0)).sum()),
                rows=len(indices),
                separation=float(((p[:, choices[0]] - p[:, choices[1]]) * labels).mean()),
                critic_mse=mse,
                enable_mechanism=enable_mechanism(trainer.model, selected, labels),
                maximum_kl=max(m["kl"] for m in metrics),
                actor_minibatches=sum(not m["actor_stopped"] for m in metrics),
                minibatches=len(metrics),
                actor_grad_max=max(m["actor_grad"] for m in metrics),
                critic_grad_max=max(m["critic_grad"] for m in metrics),
                head_gradient_max={
                    k: max(m["head_grad"][k] for m in metrics) for k in metrics[0]["head_grad"]
                },
            )
        )
    final_gap = records[-1]["separation"] if records else initial_gap
    final_mse = records[-1]["critic_mse"] if records else 1.0
    last_eight_mse = max((r["critic_mse"] for r in records[-8:]), default=1.0)
    gain = final_gap - initial_gap
    exposed = (
        sum(r["active_draws"] for r in records) >= 128
        and sum(r["active_positive_label"] for r in records) >= 16
        and sum(r["active_negative_label"] for r in records) >= 16
    )
    return dict(
        task=task,
        status=status,
        rows=len(indices),
        indices=indices.tolist(),
        feature=feature,
        choices=choices.tolist(),
        labels=labels.tolist(),
        initial_model=original,
        final_model=state_hash(trainer.model),
        initial_enable_mechanism=initial_mechanism,
        initial_separation=initial_gap,
        final_separation=final_gap,
        separation_gain=gain,
        critic_mse=final_mse,
        critic_last_eight_max_mse=last_eight_mse,
        frozen_anchors_exact=True,
        covered=exposed,
        passed=status == "completed"
        and exposed
        and gain >= 0.01
        and final_gap >= 0.01
        and final_mse < 0.9
        and last_eight_mse < 0.9,
        draws=len(indices) * len(records),
        updates=records,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--maximum-seconds", type=float, default=900)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.maximum_seconds <= 900 or len(args.tasks) != len(set(args.tasks)):
        raise ValueError("Retained diagnostic envelope changed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    before = sha(args.checkpoint)
    obs = torch.load(args.probe, weights_only=True)
    started = time.monotonic()
    deadline = started + args.maximum_seconds
    sources = [
        Path(__file__),
        Path("scripts/probe_autonomous_state_baseline.py"),
        Path("src/anthropocene_rl/policies/autonomous_ppo_v4.py"),
        Path("src/anthropocene_rl/training/autonomous_ppo_v4.py"),
    ]
    result = dict(
        version="autonomous-retained-state-baseline-v4",
        checkpoint_sha256=before,
        probe_sha256=sha(args.probe),
        source_hashes={str(p): sha(p) for p in sources},
        tasks={},
        partial=True,
    )
    for task in args.tasks:
        result["tasks"][task] = run_task(args.checkpoint, obs, task, deadline=deadline)
        assert sha(args.checkpoint) == before
        result["wall_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        r = result["tasks"][task]
        print(
            json.dumps(
                dict(
                    task=task,
                    passed=r["passed"],
                    gain=r.get("separation_gain"),
                    critic_mse=r.get("critic_mse"),
                )
            ),
            flush=True,
        )
    result.update(
        partial=False,
        passed=all(r["passed"] for r in result["tasks"].values()),
        original_unchanged=True,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
