"""Discarded, on-policy reversal tests with the checkpoint's actual Adam/PPO state."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from anthropocene_rl.autonomous_v4.training import Trainer as TrainerV1
from anthropocene_rl.autonomous_v4.training import state_hash
from anthropocene_rl.training.autonomous_ppo_v2 import AutonomousTrainerV2 as TrainerV2
from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3 as TrainerV3

TASKS = [
    "workforce",
    "movement",
    "enabled",
    "partner",
    "give",
    "want",
    "lot",
    "price",
    "activity_0",
    "activity_4",
    "activity_5",
    "activity_9",
    "control_0",
]
TRADE_INDEX = {"enabled": 0, "partner": 1, "give": 2, "want": 3, "lot": 4, "price": 5}


def simplex_span(task):
    if task == "workforce":
        return slice(0, 13)
    if task.startswith("activity_"):
        r = int(task.split("_")[1])
        return slice(13 + r * 15, 28 + r * 15)
    if task.startswith("control_"):
        c = int(task.split("_")[1])
        return slice(208 + c * 2, 210 + c * 2)
    return None


def task_face(obs, task):
    give = obs["GoodMask"] & (obs["Stocks"] >= 1e-6)
    eligible = obs["CounterpartyMask"].any(-1) & give.any(-1) & (obs["GoodMask"].sum(-1) > 1)
    if task == "workforce":
        mask = obs["WorkforceMask"]
    elif task.startswith("activity_"):
        mask = obs["ActivityMask"][:, int(task.split("_")[1])]
    elif task.startswith("control_"):
        mask = obs["ControlMask"][:, int(task.split("_")[1]), None].expand(-1, 2)
    elif task == "movement":
        mask = obs["MovementMask"]
    elif task == "enabled":
        mask = torch.stack([torch.ones_like(eligible), eligible], -1)
    elif task == "partner":
        mask = obs["CounterpartyMask"] & eligible[:, None]
    elif task == "give":
        mask = give & eligible[:, None]
    elif task == "want":
        mask = obs["GoodMask"].clone() & eligible[:, None]
        mask.scatter_(1, give.long().argmax(-1)[:, None], False)
    elif task in ["lot", "price"]:
        mask = eligible[:, None].expand(-1, 3)
    else:
        raise ValueError(task)
    faces, inverse, counts = torch.unique(mask, dim=0, return_inverse=True, return_counts=True)
    counts[faces.sum(-1) < 2] = 0
    face = int(counts.argmax())
    if counts[face] < 16:
        raise ValueError("Insufficient nondegenerate coverage")
    ix = torch.where(inverse == face)[0]
    feature = int(obs["x"][ix].std(0).argmax())
    ix = ix[torch.argsort(obs["x"][ix, feature], stable=True)]
    n = min(32, len(ix) // 2 * 2)
    ix = ix[torch.linspace(0, len(ix) - 1, n).round().long()]
    return {k: v[ix] for k, v in obs.items()}, faces[face], feature, ix


@torch.no_grad()
def probabilities(model, obs, task, mask):
    h = model.actor(obs["x"])
    span = simplex_span(task)
    if span is not None:
        raw = model.heads["simplex"](h)[:, span]
        alpha = (torch.nn.functional.softplus(raw.clamp(-12, 12)) + 0.5) * mask
        return alpha / alpha.sum(-1, keepdim=True)
    q = model.heads[task](h).masked_fill(~mask, -torch.inf).softmax(-1)
    return (1 - getattr(model, "uniform_mix", 0.0)) * q + getattr(
        model, "uniform_mix", 0.0
    ) * mask / mask.sum(-1, keepdim=True)


@torch.no_grad()
def expected_reward(model, obs, task, mask, choices, labels):
    """Exact detached state baseline, including endogenous quote/give gates."""
    p = probabilities(model, obs, task, mask)
    difference = p[:, choices[0]] - p[:, choices[1]]
    if task in TRADE_INDEX and task != "enabled":
        enabled = probabilities(
            model, obs, "enabled", torch.ones(len(labels), 2, dtype=torch.bool)
        )[:, 1]
        if task == "want":
            give_mask = obs["GoodMask"] & (obs["Stocks"] >= 1e-6)
            offered = probabilities(model, obs, "give", give_mask)
            difference = torch.zeros_like(labels)
            for g in range(give_mask.shape[-1]):
                if g in choices:
                    continue
                wanted_mask = obs["GoodMask"].clone()
                wanted_mask[:, g] = False
                wanted = probabilities(model, obs, "want", wanted_mask)
                difference += offered[:, g] * (wanted[:, choices[0]] - wanted[:, choices[1]])
        difference = enabled * difference
    return labels * difference


def enable_mechanism(model, obs, labels):
    from anthropocene_rl.policies.autonomous_ppo_v2 import distribution

    with torch.enable_grad():
        logits = model.heads["enabled"](model.actor(obs["x"]))
        mask = torch.ones_like(logits, dtype=torch.bool)
        dist, penalty = distribution(logits, mask, getattr(model, "uniform_mix", 0.0))
        expected = (dist.probs[:, 0] - dist.probs[:, 1]) * labels
        parameters = list(model.heads["enabled"].parameters())
        task_grad = torch.autograd.grad(-expected.mean(), parameters, retain_graph=True)
        regularizer_grad = torch.autograd.grad(penalty.mean(), parameters)
    return dict(
        raw_gap=float((logits[:, 0] - logits[:, 1]).detach().abs().mean()),
        raw_max_probability=float(logits.detach().softmax(-1).max(-1).values.mean()),
        reverse_kl=float(penalty.detach().mean()),
        task_gradient_l2=float(torch.cat([g.flatten() for g in task_grad]).norm()),
        regularizer_gradient_l2=float(torch.cat([g.flatten() for g in regularizer_grad]).norm()),
    )


def make_trainer(checkpoint, variant, reverse_kl=None):
    state = torch.load(checkpoint, weights_only=False)
    if state["version"] == f"autonomous-ppo-checkpoint-{variant}" and variant != "v1":
        cls = TrainerV2 if variant == "v2" else TrainerV3
        trainer = cls(state["config"], 0, state["provenance"], width=state["width"])
        trainer.restore(checkpoint)
        if reverse_kl is not None and reverse_kl != state["config"]["legal_reverse_kl_coef"]:
            raise ValueError("Use native checkpoint configuration for retained probes")
    elif variant == "v1":
        trainer = TrainerV1(state["config"], 0, state["provenance"], width=state["width"])
        trainer.restore(checkpoint)
    else:
        name = "remedy-v2" if variant == "v2" else "layernorm-v3"
        config = json.loads(Path(f"configs/training/autonomous-ppo-{name}.json").read_text())
        if reverse_kl is not None:
            config["legal_reverse_kl_coef"] = reverse_kl
        cls = TrainerV2 if variant == "v2" else TrainerV3
        trainer = cls(config, 0, state["provenance"], width=state["width"])
        trainer.warm_start(
            checkpoint, expected_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        )
    return trainer


def run_task(checkpoint, obs, task, *, variant="v2", reverse_kl=None, updates=64, deadline):
    trainer = make_trainer(checkpoint, variant, reverse_kl)
    original = state_hash(trainer.model)
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
    reference_version = torch.load(checkpoint, weights_only=False)["version"].rsplit("-", 1)[-1]
    reference = make_trainer(checkpoint, reference_version)
    reference_p = probabilities(reference.model, selected, task, mask)
    reference_gap = float(
        ((reference_p[:, choices[0]] - reference_p[:, choices[1]]) * labels).mean()
    )
    if reference_gap > 0:
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
        covered=exposed,
        passed=status == "completed"
        and exposed
        and gain >= 0.01
        and final_gap >= 0.01
        and final_mse < 0.9,
        draws=len(indices) * len(records),
        updates=records,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse-kl", type=float)
    parser.add_argument("--variant", choices=["v1", "v2", "v3"], required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never overwrite diagnostic evidence")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    before = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    obs = torch.load(args.probe, weights_only=True)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        *sorted((root / "src/anthropocene_rl/autonomous_v4").glob("*.py")),
        root / "src/anthropocene_rl/autonomous/policy.py",
        root / "src/anthropocene_rl/training/returns.py",
        root / "src/anthropocene_rl/policies/autonomous_ppo_v2.py",
        root / "src/anthropocene_rl/training/autonomous_ppo_v2.py",
        root / "configs/training/autonomous-ppo-remedy-v2.json",
        root / "src/anthropocene_rl/policies/autonomous_ppo_v3.py",
        root / "src/anthropocene_rl/training/autonomous_ppo_v3.py",
        root / "configs/training/autonomous-ppo-layernorm-v3.json",
    ]
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    deadline = time.monotonic() + 900
    result = {}
    for task in args.tasks:
        result[task] = run_task(
            args.checkpoint,
            obs,
            task,
            variant=args.variant,
            reverse_kl=args.reverse_kl,
            deadline=deadline,
        )
        args.output.write_text(
            json.dumps(
                dict(
                    version="retained-ppo-state-baseline-v2",
                    variant=args.variant,
                    reverse_kl=args.reverse_kl,
                    partial=True,
                    checkpoint_sha256=before,
                    source_hashes=source_hashes,
                    tasks=result,
                ),
                indent=2,
            )
            + "\n"
        )
        print(
            task,
            result[task]["passed"],
            result[task].get("separation_gain"),
            result[task].get("critic_mse"),
            flush=True,
        )
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == before
    args.output.write_text(
        json.dumps(
            dict(
                version="retained-ppo-state-baseline-v2",
                variant=args.variant,
                reverse_kl=args.reverse_kl,
                checkpoint_sha256=before,
                source_hashes=source_hashes,
                probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),
                tasks=result,
                passed=all(r["passed"] for r in result.values()),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                task: {
                    k: v for k, v in r.items() if k in ["passed", "separation_gain", "critic_mse"]
                }
                for task, r in result.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
