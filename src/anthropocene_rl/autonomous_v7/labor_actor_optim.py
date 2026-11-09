"""The labor actor's PPO optimizer: Adam with two named parameter groups.

One Adam step moves every parameter by about ``lr`` whatever its gradient, and the
actor's ALR Gaussian specialty/activity heads are tight (std ~0.135, ~21 coordinates per
row), so a single step at the historical single rate 3e-4 over all ~865k parameters moved
the BC-initialised policy by thousands of nats (D7 rollout, 581 parents). ~92% of the
parameters sit in the shared encoder. This module gives the encoder and the heads their
own learning rates and an optional linear warm-up.

Contract (``labor-actor-optimizer-v1``):

* Groups, in this fixed order: ``encoder`` (parameter names starting ``encoder.``) and
  ``heads`` (every other parameter), each in ``named_parameters()`` order, each carrying
  its ``name`` and torch's ``param_names``. An empty group is kept, so the layout never
  depends on the architecture.
* Learning rates are configuration, not state. :func:`load_actor_optimizer_state` restores
  the per-parameter Adam moments and step counts of a saved state (legacy one-group, or
  two-group) and, when given an :class:`ActorLrConfig`, then re-applies the configured
  per-group rates: a checkpoint's saved ``lr`` never overrides the command line.
* An encoder rate of 0 freezes the encoder. Its parameters keep ``requires_grad`` and stay
  in the optimizer (Adam with lr 0 adds ``-0 * m/(sqrt(v)+eps)``, i.e. leaves them
  unchanged, while their moments and step counts keep accumulating), so the optimizer
  state layout, checkpoint compatibility, the shared grad-norm clip and the warm-up step
  count are the same whatever the rates, and unfreezing later resumes with warm moments.
* Warm-up: the rate used for an optimizer (minibatch) step is the configured rate times
  ``min(1, (n + 1) / W)``, where ``n`` is the number of Adam steps already taken, read
  from the persisted Adam state (the largest per-parameter ``step``). No new saved field:
  it survives checkpoint/resume, and a rollback that restores the Adam state restores
  ``n``, hence the rate. ``W = 0`` (default) is factor 1 exactly, so the configured rates
  are used unchanged. A fresh Adam state (new run, or ``pretrain-bc``'s fresh optimizer)
  restarts the warm-up; a resumed state with ``n >= W`` is past it.

With ``--actor-encoder-lr``/``--actor-head-lr`` unset (both fall back to ``--actor-lr``)
and no warm-up, every parameter's Adam update is the single-group update bit for bit:
Adam's per-parameter arithmetic depends only on that parameter's group hyperparameters,
which are then identical.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

ACTOR_OPTIMIZER_SCHEMA = "labor-actor-optimizer-v1"
ACTOR_PARAM_GROUPS = ("encoder", "heads")
ENCODER_PREFIX = "encoder."
DEFAULT_ACTOR_LR = 3e-4


@dataclass(frozen=True)
class ActorLrConfig:
    """Configured per-group learning rates and the warm-up length (optimizer steps)."""

    encoder_lr: float
    head_lr: float
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        for label, value in (("encoder", self.encoder_lr), ("head", self.head_lr)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"actor {label} lr must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"actor {label} lr must be finite and nonnegative")
        if not (float(self.encoder_lr) > 0 or float(self.head_lr) > 0):
            raise ValueError("at least one actor learning rate must be positive")
        if (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or self.warmup_steps < 0
        ):
            raise ValueError("actor lr warm-up steps must be a nonnegative integer")

    def base_lr(self, group: str) -> float:
        if group == "encoder":
            return float(self.encoder_lr)
        if group == "heads":
            return float(self.head_lr)
        raise KeyError(group)

    def factor(self, steps_taken: int) -> float:
        """Warm-up factor for the next step after ``steps_taken`` Adam steps."""
        if self.warmup_steps == 0:
            return 1.0
        return min(1.0, (int(steps_taken) + 1) / self.warmup_steps)

    def record(self) -> dict[str, Any]:
        return {
            "schema": ACTOR_OPTIMIZER_SCHEMA,
            "groups": list(ACTOR_PARAM_GROUPS),
            "encoderLr": float(self.encoder_lr),
            "headLr": float(self.head_lr),
            "warmupSteps": int(self.warmup_steps),
            "warmup": "lr x min(1, (n+1)/W), n = Adam steps already taken (from the Adam state)",
            "lrSource": "configuration; re-applied after loading any saved optimizer state",
        }


def actor_lr_config(args: argparse.Namespace) -> ActorLrConfig:
    """``--actor-encoder-lr``/``--actor-head-lr`` (None: ``--actor-lr``) and
    ``--actor-lr-warmup-steps`` (default 0)."""
    base = float(getattr(args, "actor_lr", DEFAULT_ACTOR_LR))
    encoder = getattr(args, "actor_encoder_lr", None)
    head = getattr(args, "actor_head_lr", None)
    warmup = getattr(args, "actor_lr_warmup_steps", 0)
    return ActorLrConfig(
        base if encoder is None else float(encoder),
        base if head is None else float(head),
        0 if warmup is None else int(warmup),
    )


def add_actor_lr_arguments(parser: argparse.ArgumentParser) -> None:
    """The per-group/warm-up flags (to sit next to an existing ``--actor-lr``)."""
    parser.add_argument(
        "--actor-encoder-lr",
        type=float,
        default=None,
        help="Adam lr of the actor's encoder.* parameters (default: --actor-lr; 0 freezes it)",
    )
    parser.add_argument(
        "--actor-head-lr",
        type=float,
        default=None,
        help="Adam lr of the actor's other (head) parameters (default: --actor-lr)",
    )
    parser.add_argument(
        "--actor-lr-warmup-steps",
        type=int,
        default=0,
        help="linear actor lr warm-up over the first W Adam (minibatch) steps, counted from "
        "the optimizer state (default 0: off)",
    )


def actor_parameter_names(actor: nn.Module) -> dict[str, list[str]]:
    """Parameter names per group, in ``named_parameters()`` order within each group."""
    groups: dict[str, list[str]] = {name: [] for name in ACTOR_PARAM_GROUPS}
    for name, _ in actor.named_parameters():
        groups["encoder" if name.startswith(ENCODER_PREFIX) else "heads"].append(name)
    return groups


def actor_group_counts(actor: nn.Module) -> dict[str, dict[str, int]]:
    parameters = dict(actor.named_parameters())
    return {
        group: {
            "tensors": len(names),
            "parameters": int(sum(parameters[name].numel() for name in names)),
        }
        for group, names in actor_parameter_names(actor).items()
    }


def build_actor_optimizer(actor: nn.Module, config: ActorLrConfig) -> torch.optim.Adam:
    """A fresh two-group Adam at the configured rates (warm-up factor for n = 0)."""
    parameters = dict(actor.named_parameters())
    if not parameters:
        raise ValueError("the actor has no parameters")
    groups = [
        {
            "params": [(name, parameters[name]) for name in names],
            "lr": config.base_lr(group),
            "name": group,
        }
        for group, names in actor_parameter_names(actor).items()
    ]
    optimizer = torch.optim.Adam(groups, lr=config.base_lr("heads"))
    apply_actor_lrs(optimizer, config)
    return optimizer


def _require_actor_layout(optimizer: torch.optim.Optimizer) -> None:
    if [group.get("name") for group in optimizer.param_groups] != list(ACTOR_PARAM_GROUPS):
        raise ValueError(
            f"actor optimizer must have the named groups {list(ACTOR_PARAM_GROUPS)} "
            "(build it with build_actor_optimizer)"
        )


def adam_steps_taken(optimizer: torch.optim.Optimizer) -> int:
    """The number of Adam steps already taken: the largest per-parameter ``step``."""
    steps = 0
    for state in optimizer.state.values():
        step = state.get("step")
        if step is not None:
            steps = max(steps, int(float(step)))
    return steps


def apply_actor_lrs(optimizer: torch.optim.Optimizer, config: ActorLrConfig) -> float:
    """Set every group's lr to its configured rate times the warm-up factor for the next
    step (from the optimizer's own Adam state); returns the factor."""
    _require_actor_layout(optimizer)
    factor = config.factor(adam_steps_taken(optimizer))
    for group in optimizer.param_groups:
        base = config.base_lr(group["name"])
        group["lr"] = base if factor == 1.0 else base * factor
    return factor


class ActorLrSchedule:
    """``update_labor_policy``'s ``lr_schedule``: applies :func:`apply_actor_lrs` before each
    minibatch step (and after the update) and returns the factor it used."""

    def __init__(self, config: ActorLrConfig) -> None:
        self.config = config

    def __call__(self, optimizer: torch.optim.Optimizer) -> float:
        return apply_actor_lrs(optimizer, self.config)

    def record(self) -> dict[str, Any]:
        return self.config.record()


def actor_group_lrs(optimizer_or_state: torch.optim.Optimizer | Mapping[str, Any]) -> dict:
    groups = (
        optimizer_or_state.param_groups
        if isinstance(optimizer_or_state, torch.optim.Optimizer)
        else optimizer_or_state["param_groups"]
    )
    if len(groups) == 1:
        return {"legacy-single-group": float(groups[0]["lr"])}
    return {str(group.get("name", index)): float(group["lr"]) for index, group in enumerate(groups)}


def _two_group_state(actor: nn.Module, saved: Mapping[str, Any]) -> dict[str, Any]:
    """``saved`` (legacy one-group or two-group) re-indexed to the two-group layout,
    each parameter's Adam state moved with its parameter (by name)."""
    if not isinstance(saved, Mapping) or not {"state", "param_groups"} <= set(saved):
        raise ValueError("actor optimizer state must have 'state' and 'param_groups'")
    natural = [name for name, _ in actor.named_parameters()]
    parameters = dict(actor.named_parameters())
    grouped = actor_parameter_names(actor)
    saved_groups = list(saved["param_groups"])
    # saved index -> parameter name
    saved_name: dict[int, str] = {}
    if len(saved_groups) == 1:
        group = saved_groups[0]
        indices = list(group["params"])
        if len(indices) != len(natural):
            raise ValueError(
                f"legacy actor optimizer state has {len(indices)} parameters; "
                f"the actor has {len(natural)}"
            )
        names = group.get("param_names")
        if names is not None and list(names) != natural:
            raise ValueError("legacy actor optimizer parameter names differ from the actor's")
        saved_name = dict(zip(indices, natural, strict=True))
        hyper = [
            {k: v for k, v in group.items() if k not in ("params", "param_names", "name")}
        ] * len(ACTOR_PARAM_GROUPS)
    elif len(saved_groups) == len(ACTOR_PARAM_GROUPS):
        if [group.get("name") for group in saved_groups] != list(ACTOR_PARAM_GROUPS):
            raise ValueError(
                f"actor optimizer groups must be named {list(ACTOR_PARAM_GROUPS)} in order"
            )
        hyper = []
        for group in saved_groups:
            names = group.get("param_names")
            expected = grouped[group["name"]]
            if names is None or len(group["params"]) != len(names):
                raise ValueError("two-group actor optimizer state lacks its parameter names")
            if list(names) != expected:
                raise ValueError(
                    f"actor optimizer group {group['name']!r} has {len(names)} parameters "
                    f"({'names differ' if len(names) == len(expected) else 'size differs'}); "
                    f"the actor's has {len(expected)}"
                )
            saved_name.update(zip(group["params"], names, strict=True))
            hyper.append(
                {k: v for k, v in group.items() if k not in ("params", "param_names", "name")}
            )
    else:
        raise ValueError(
            f"actor optimizer state has {len(saved_groups)} parameter groups; expected 1 "
            f"(legacy) or {len(ACTOR_PARAM_GROUPS)}"
        )
    if len(set(saved_name)) != len(saved_name) or set(saved_name.values()) != set(natural):
        raise ValueError("actor optimizer state does not cover each actor parameter once")
    new_index: dict[str, int] = {}
    param_groups = []
    for group, settings in zip(ACTOR_PARAM_GROUPS, hyper, strict=True):
        names = grouped[group]
        indices = list(range(len(new_index), len(new_index) + len(names)))
        new_index.update(zip(names, indices, strict=True))
        param_groups.append(
            {**settings, "params": indices, "param_names": list(names), "name": group}
        )
    state: dict[int, Any] = {}
    for index, value in saved["state"].items():
        if index not in saved_name:
            raise ValueError(f"actor optimizer state entry {index!r} names no parameter")
        name = saved_name[index]
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            moment = value.get(key)
            if moment is not None and tuple(moment.shape) != tuple(parameters[name].shape):
                raise ValueError(
                    f"actor optimizer {key} of {name} has shape {tuple(moment.shape)}; "
                    f"the parameter has {tuple(parameters[name].shape)}"
                )
        state[new_index[name]] = value
    return {"state": state, "param_groups": param_groups}


def load_actor_optimizer_state(
    optimizer: torch.optim.Optimizer,
    actor: nn.Module,
    saved: Mapping[str, Any],
    config: ActorLrConfig | None = None,
) -> dict[str, Any]:
    """Load ``saved`` (legacy one-group or two-group) into a :func:`build_actor_optimizer`
    optimizer, moving each parameter's Adam moments and step with that parameter; refuse
    any parameter/group size or name mismatch.

    With ``config`` the configured rates (times the warm-up factor for the restored step
    count) replace the saved ones; without it the saved rates are kept (pass-through
    commands that never step the actor). Returns a record of both."""
    _require_actor_layout(optimizer)
    if [id(p) for group in optimizer.param_groups for p in group["params"]] != [
        id(p) for _, p in _grouped_parameters(actor)
    ]:
        raise ValueError("the actor optimizer was not built for this actor")
    converted = _two_group_state(actor, saved)
    optimizer.load_state_dict(converted)
    record: dict[str, Any] = {
        "savedLayout": "legacy-single-group"
        if len(saved["param_groups"]) == 1
        else "two-group",
        "savedLrs": actor_group_lrs(saved),
        "adamStepsTaken": adam_steps_taken(optimizer),
    }
    if config is not None:
        record["lrFactor"] = apply_actor_lrs(optimizer, config)
        record["lrSource"] = "configuration (saved lrs ignored)"
    else:
        record["lrSource"] = "saved (pass-through; train re-applies its configuration)"
    record["appliedLrs"] = actor_group_lrs(optimizer)
    return record


def _grouped_parameters(actor: nn.Module) -> list[tuple[str, nn.Parameter]]:
    parameters = dict(actor.named_parameters())
    return [
        (name, parameters[name])
        for names in actor_parameter_names(actor).values()
        for name in names
    ]


def saved_actor_lr_config(
    saved: Mapping[str, Any], default: float = DEFAULT_ACTOR_LR
) -> ActorLrConfig:
    """The rates a saved state carries (one group: both groups at its rate); warm-up 0."""
    groups = list(saved.get("param_groups", [])) if isinstance(saved, Mapping) else []
    if not groups:
        return ActorLrConfig(default, default)
    if len(groups) == 1:
        lr = float(groups[0].get("lr", default))
        return ActorLrConfig(lr, lr)
    lrs = {group.get("name"): float(group.get("lr", default)) for group in groups}
    return ActorLrConfig(lrs.get("encoder", default), lrs.get("heads", default))


def rebuild_actor_optimizer(
    actor: nn.Module, saved: Mapping[str, Any], *, fresh: bool = False
) -> torch.optim.Adam:
    """For commands that never step the actor: a two-group optimizer at the saved rates,
    carrying the saved Adam state (``fresh``: empty state, e.g. after BC changed the actor)."""
    optimizer = build_actor_optimizer(actor, saved_actor_lr_config(saved))
    if not fresh:
        load_actor_optimizer_state(optimizer, actor, saved)
    return optimizer


def actor_optimizer_state_by_name(
    actor: nn.Module, saved: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Per-parameter Adam state keyed by parameter name (any supported layout)."""
    converted = _two_group_state(actor, saved)
    names = [n for group in converted["param_groups"] for n in group["param_names"]]
    return {names[index]: value for index, value in converted["state"].items()}


__all__ = [
    "ACTOR_OPTIMIZER_SCHEMA",
    "ACTOR_PARAM_GROUPS",
    "ENCODER_PREFIX",
    "ActorLrConfig",
    "ActorLrSchedule",
    "actor_group_counts",
    "actor_group_lrs",
    "actor_lr_config",
    "actor_optimizer_state_by_name",
    "actor_parameter_names",
    "adam_steps_taken",
    "add_actor_lr_arguments",
    "apply_actor_lrs",
    "build_actor_optimizer",
    "load_actor_optimizer_state",
    "rebuild_actor_optimizer",
    "saved_actor_lr_config",
]

