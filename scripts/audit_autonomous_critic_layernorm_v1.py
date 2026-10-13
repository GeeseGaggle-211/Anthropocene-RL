"""Read-only, TRAIN-probe audit of critic LayerNorm conversion disruption."""

import argparse
import copy
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from anthropocene_rl.autonomous_v4.training import state_hash
from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def representation(model, obs):
    x = obs["x"].clone().requires_grad_()
    h = x
    layers = []
    for layer in model.critic[0]:
        h = layer(h)
        if isinstance(layer, nn.Tanh):
            activation = h.detach()
            singular = torch.linalg.svdvals(activation - activation.mean(0))
            mass = singular / singular.sum().clamp_min(1e-20)
            layers.append(
                dict(
                    saturated_fraction=float((activation.abs() >= 0.99).float().mean()),
                    mean_derivative=float((1 - activation.square()).mean()),
                    effective_rank=float(torch.exp(-(mass * mass.clamp_min(1e-20).log()).sum())),
                )
            )
    value = model.critic[1](h).squeeze(-1)
    gradient = torch.autograd.grad(value.sum(), x)[0].norm(dim=-1)
    return value.detach(), dict(
        layers=layers,
        value_mean=float(value.detach().mean()),
        value_rms=float(value.detach().square().mean().sqrt()),
        input_gradient_mean=float(gradient.mean()),
        input_gradient_maximum=float(gradient.max()),
    )


def layernorm_copy(model):
    candidate = copy.deepcopy(model)
    trunk = candidate.critic[0]
    candidate.critic[0] = nn.Sequential(
        OrderedDict(
            [
                ("0", trunk[0]),
                ("norm0", nn.LayerNorm(model.width, eps=1e-5)),
                ("1", trunk[1]),
                ("2", trunk[2]),
                ("norm1", nn.LayerNorm(model.width, eps=1e-5)),
                ("3", trunk[3]),
            ]
        )
    )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, dict(candidate.named_parameters())[name])
    assert state_hash(candidate.actor) == state_hash(model.actor)
    assert state_hash(candidate.heads) == state_hash(model.heads)
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never overwrite audit evidence")
    cfg = json.loads(args.config.read_text())
    for item in cfg["inputs"].values():
        if digest(item["path"]) != item["sha256"]:
            raise ValueError(f"Input digest mismatch: {item['path']}")
    assert len(cfg["checkpoints"]) == 6 and cfg["maximum_wall_seconds"] == 60
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    obs = torch.load(cfg["inputs"]["probe"]["path"], weights_only=True)
    started = time.monotonic()
    report = dict(
        version="autonomous-critic-layernorm-audit-v1",
        config=cfg,
        config_sha256=digest(args.config),
        checkpoints=[],
        native_steps=0,
        optimizer_updates=0,
        action_draws=0,
        promotion_allowed=False,
    )
    try:
        for item in cfg["checkpoints"]:
            if time.monotonic() - started >= cfg["maximum_wall_seconds"]:
                raise TimeoutError("LayerNorm audit deadline")
            checkpoint = Path(item["path"])
            if digest(checkpoint) != item["sha256"]:
                raise ValueError("Checkpoint digest mismatch")
            state = torch.load(checkpoint, weights_only=False)
            trainer = AutonomousTrainerV3(state["config"], 0, state["provenance"], state["width"])
            trainer.restore(checkpoint)
            actor_optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
            old, before = representation(trainer.model, obs)
            candidate = layernorm_copy(trainer.model)
            new, after = representation(candidate, obs)
            delta = new - old
            old_order = old[:, None] - old[None, :]
            new_order = new[:, None] - new[None, :]
            pairs = torch.triu(torch.ones_like(old_order, dtype=torch.bool), diagonal=1)
            pairs &= old_order != 0
            # No optimizer is attached to or updated on the conversion. Every original
            # actor/critic Adam tensor remains byte-for-byte equal in the restored trainer.
            for key, old_state in actor_optimizer_before["state"].items():
                for name, value in old_state.items():
                    actual = trainer.optimizer.state_dict()["state"][key][name]
                    assert (
                        torch.equal(value, actual)
                        if isinstance(value, torch.Tensor)
                        else value == actual
                    )
            assert digest(checkpoint) == item["sha256"]
            report["checkpoints"].append(
                dict(
                    checkpoint=item,
                    original=before,
                    layernorm=after,
                    value_shift_mean=float(delta.mean()),
                    value_shift_rms=float(delta.square().mean().sqrt()),
                    value_shift_maximum=float(delta.abs().max()),
                    value_correlation=float(torch.corrcoef(torch.stack([old, new]))[0, 1]),
                    pairwise_order_preserved=float(
                        (old_order[pairs].sign() == new_order[pairs].sign()).float().mean()
                    ),
                    actor_and_heads_exact=True,
                    original_adam_exact=True,
                    checkpoint_unchanged=True,
                    learning_rate_only_entry_values_exact=True,
                )
            )
        report.update(complete=True, wall_seconds=time.monotonic() - started)
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["checkpoints"]))


if __name__ == "__main__":
    main()
