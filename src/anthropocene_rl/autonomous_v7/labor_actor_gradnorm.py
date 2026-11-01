"""Actor BC task decomposition and opt-in GradNorm balancing of its tasks.

The actor BC loss (``labor_bc._batch_loss_and_l1``) is a sum of head losses. For task
balancing it is decomposed into :data:`ACTOR_BC_TASKS`: the specialty CE, one activity CE per
role (``activity:<role>``, the role's share of the activity loss as weighted today: the
resident share n_r/P times the inverse-frequency activity weight, over the common activity
denominator, so the 13 role terms sum to the activity loss), the movement CE, the fission CE
and the trade-gate CE. The gauge anchors, the log-std regulariser and the trade value/share
regressions are not tasks: they always enter the loss unweighted.

Shared trunk: everything upstream of the head-specific output layers is ``actor.encoder``
(:func:`actor_trunk_parameters`; the heads are the small ``nn.Linear`` maps on the actor).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .labor_bridge import SPECIALTIES

ACTOR_ROLES: tuple[str, ...] = tuple(SPECIALTIES[1:])


def role_task(role: str) -> str:
    return f"activity:{role}"


ACTOR_ROLE_TASKS: tuple[str, ...] = tuple(role_task(r) for r in ACTOR_ROLES)
ACTOR_BC_TASKS: tuple[str, ...] = (
    "specialty",
    *ACTOR_ROLE_TASKS,
    "movement",
    "fission",
    "trade_gate",
)


def actor_trunk_parameters(actor: nn.Module) -> list[nn.Parameter]:
    """The whole shared trunk: every encoder parameter (upstream of all output layers)."""
    return list(actor.encoder.parameters())


def actor_gradnorm_parameters(actor: nn.Module) -> list[nn.Parameter]:
    """GradNorm's W for the actor: the last global attention layer and the global output norm.

    Every task's gradient reaches them (activity through the contextual specialty and village
    tokens, specialty through its tokens, movement/fission through the trunk of the village
    token, trade gates through the legacy trade rows), and a gradient there needs a backward
    pass through the heads and one attention layer only, not the whole encoder."""
    encoder = actor.encoder
    layers = getattr(getattr(encoder, "attention", None), "layers", None)
    if not layers or not isinstance(getattr(encoder, "norm", None), nn.LayerNorm):
        raise ValueError("actor gradnorm needs an encoder with global attention and its norm")
    return [*layers[-1].parameters(), *encoder.norm.parameters()]


# ---------------------------------------------------------------------------
# Opt-in GradNorm balancing (``pretrain-bc --loss-balancing gradnorm``)
# ---------------------------------------------------------------------------

GRADNORM_INITS = ("uniform", "sqrt", "share")
# Where the task gradient norms are measured: the whole encoder (the shared trunk; exact, a
# full backward per task) or its last global attention layer + output norm (cheaper; on the
# r5 pilot it under-reports the role activity tasks 2-7x against the trunk, because the
# activity tokens also reach the heads through the local-attention path that bypasses it).
GRADNORM_SHARED = ("trunk", "last-layer")


def gradnorm_shared_parameters_for(actor: nn.Module, shared: str) -> list[nn.Parameter]:
    if shared == "trunk":
        return actor_trunk_parameters(actor)
    if shared == "last-layer":
        return actor_gradnorm_parameters(actor)
    raise ValueError(f"actor gradnorm shared must be one of {GRADNORM_SHARED}")


@dataclass(frozen=True)
class ActorGradNorm:
    """``pretrain-bc --loss-balancing gradnorm``: GradNorm over :data:`ACTOR_BC_TASKS`.

    Scale. Each task i has a base scale s_i: a role task's is the role's mean resident share
    over the fitting rows (row-weighted, specialty-valid rows), every other task's is 1. The
    weights keep the invariant sum_i w_i s_i = sum_i s_i, so with all w_i = 1 (``init share``)
    the loss is the historical one and any weighting keeps the total loss scale.

    Initial weights (``init``): ``share`` all 1 (the current loss); ``uniform`` equal expected
    weight per role (w_r s_r equal for all roles: each role is one task whatever its resident
    share); ``sqrt`` w_r proportional to s_r^(-1/2). Non-role tasks start at 1; the role weights
    are scaled to the invariant (and clamped to [min_weight, max_weight]).

    Controller (every ``every`` steps, on the step's measurement, only for accepted steps):
    raw (unweighted) task gradient norms g_i on W (``shared``: the whole encoder trunk, the
    default, or its last layer; :func:`gradnorm_shared_parameters_for`) and task
    losses enter EMAs (rate ``ema``); a role task is measured, and its weight updated, only when
    the batch staffs it in at least ``min_rows`` rows (otherwise its EMAs and weight are held).
    Over the updated tasks U, with G_i = w_i g_i, loss ratios Lt_i = L_i / L_i(0) (L_i(0) the
    task's first EMA loss; a role's loss is its composition-normalised mean CE) and inverse
    rates r_i = Lt_i / mean_U(Lt), the GradNorm target is mean_U(G) r_i^alpha and the ideal
    weight w*_i = target_i / g_i (alpha 0: pure gradient-norm equalisation). Each weight takes
    a log-space step ln w <- (1 - lr) ln w + lr ln w*, is clamped to [min_weight, max_weight],
    and the updated weights are rescaled (held weights unchanged) to the invariant, clamping and
    rescaling again until both hold (at most 20 rounds; a saturated clamp can leave the
    invariant short, recorded as ``scale_sum``). No RNG; float64 arithmetic on the CPU. The
    weights a step uses are those of the previous update.
    """

    alpha: float = 0.0
    ema: float = 0.1
    lr: float = 0.25
    init: str = "uniform"
    min_rows: int = 4
    min_weight: float = 0.01
    max_weight: float = 100.0
    every: int = 1
    shared: str = "trunk"

    def __post_init__(self) -> None:
        for name in ("alpha", "ema", "lr", "min_weight", "max_weight"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"actor gradnorm {name} must be finite")
        if self.alpha < 0.0:
            raise ValueError("actor gradnorm alpha must be non-negative")
        if not 0.0 < self.ema <= 1.0 or not 0.0 < self.lr <= 1.0:
            raise ValueError("actor gradnorm ema and lr must be in (0, 1]")
        if self.init not in GRADNORM_INITS:
            raise ValueError(f"actor gradnorm init must be one of {GRADNORM_INITS}")
        if int(self.min_rows) < 1 or int(self.every) < 1:
            raise ValueError("actor gradnorm min_rows and every must be positive")
        if not 0.0 < self.min_weight <= 1.0 <= self.max_weight:
            raise ValueError("actor gradnorm needs 0 < min_weight <= 1 <= max_weight")
        if self.shared not in GRADNORM_SHARED:
            raise ValueError(f"actor gradnorm shared must be one of {GRADNORM_SHARED}")

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": "gradnorm",
            "tasks": list(ACTOR_BC_TASKS),
            "alpha": float(self.alpha),
            "ema": float(self.ema),
            "lr": float(self.lr),
            "init": str(self.init),
            "min_rows": int(self.min_rows),
            "min_weight": float(self.min_weight),
            "max_weight": float(self.max_weight),
            "every": int(self.every),
            "shared": str(self.shared),
        }


def role_resident_scales(
    target_specialty: torch.Tensor,
    row_weight: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
) -> dict[str, float]:
    """Each role task's base scale s_r: its row-weighted mean resident share."""
    shares = target_specialty[:, 1:].double()
    n = shares.shape[0]
    weight = (
        torch.ones(n, dtype=torch.float64)
        if row_weight is None
        else row_weight.reshape(n).double()
    )
    if valid is not None:
        weight = weight * valid.reshape(n).double()
    if float(weight.sum()) <= 0.0:
        weight = torch.ones(n, dtype=torch.float64)
    mean = (shares * weight.unsqueeze(-1)).sum(dim=0) / weight.sum()
    return {role_task(r): float(mean[i]) for i, r in enumerate(ACTOR_ROLES)}


class ActorGradNormState:
    """Runtime actor GradNorm weights and EMAs (see :class:`ActorGradNorm`)."""

    SCALE_FLOOR = 1e-6

    def __init__(self, config: ActorGradNorm, role_scales: Mapping[str, float]) -> None:
        self.config = config
        self.tasks = ACTOR_BC_TASKS
        self.scales = {
            t: (max(self.SCALE_FLOOR, float(role_scales[t])) if t in ACTOR_ROLE_TASKS else 1.0)
            for t in self.tasks
        }
        self.target_scale = sum(self.scales.values())
        self.weights = {t: 1.0 for t in self.tasks}
        if config.init != "share":
            power = 1.0 if config.init == "uniform" else 0.5
            for t in ACTOR_ROLE_TASKS:
                self.weights[t] = self.scales[t] ** -power
            self._project(set(ACTOR_ROLE_TASKS))
        self.initial_weights = dict(self.weights)
        self.ema_norm: dict[str, float | None] = {t: None for t in self.tasks}
        self.ema_loss: dict[str, float | None] = {t: None for t in self.tasks}
        self.initial_loss: dict[str, float | None] = {t: None for t in self.tasks}
        self.updates = 0
        self.task_updates = {t: 0 for t in self.tasks}
        self.task_holds = {t: 0 for t in self.tasks}
        self.last: dict[str, Any] | None = None

    def values(self) -> dict[str, float]:
        return dict(self.weights)

    _STATE = (
        "weights", "ema_norm", "ema_loss", "initial_loss", "updates", "task_updates",
        "task_holds", "last",
    )

    def snapshot(self) -> dict[str, Any]:
        """The mutable controller state (a rolled-back step restores it)."""
        return {name: copy.deepcopy(getattr(self, name)) for name in self._STATE}

    def restore(self, state: Mapping[str, Any]) -> None:
        for name in self._STATE:
            setattr(self, name, copy.deepcopy(state[name]))

    def scale_sum(self) -> float:
        return sum(self.weights[t] * self.scales[t] for t in self.tasks)

    def _project(self, free: set[str]) -> None:
        """Rescale the ``free`` weights to the scale invariant and clamp them (alternating)."""
        lo, hi = float(self.config.min_weight), float(self.config.max_weight)
        for _ in range(20):
            fixed = sum(self.weights[t] * self.scales[t] for t in self.tasks if t not in free)
            moving = sum(self.weights[t] * self.scales[t] for t in free)
            if not free or moving <= 0.0 or self.target_scale - fixed <= 0.0:
                break
            factor = (self.target_scale - fixed) / moving
            for t in free:
                self.weights[t] = min(hi, max(lo, self.weights[t] * factor))
            if abs(self.scale_sum() - self.target_scale) <= 1e-9 * self.target_scale:
                break

    def update(
        self,
        raw_norms: Mapping[str, float],
        losses: Mapping[str, float],
        staffed_rows: Mapping[str, int],
    ) -> dict[str, Any]:
        """One controller update from a step's raw W norms, task losses and staffing."""
        c = self.config
        eligible = []
        for t in self.tasks:
            norm, loss = float(raw_norms.get(t, math.nan)), float(losses.get(t, math.nan))
            rows_ok = t not in ACTOR_ROLE_TASKS or int(staffed_rows.get(t, 0)) >= c.min_rows
            if not rows_ok or not math.isfinite(norm) or not math.isfinite(loss) or norm <= 0.0:
                self.task_holds[t] += 1
                continue
            for store, value in ((self.ema_norm, norm), (self.ema_loss, loss)):
                held = store[t]
                store[t] = value if held is None else (1 - c.ema) * held + c.ema * value
            if self.initial_loss[t] is None:
                self.initial_loss[t] = self.ema_loss[t]
            eligible.append(t)
        entry: dict[str, Any] = {
            "weights_used": dict(self.weights),
            "raw_grad_norm": {t: float(raw_norms.get(t, math.nan)) for t in self.tasks},
            "loss": {t: float(losses.get(t, math.nan)) for t in self.tasks},
            "staffed_rows": {t: int(staffed_rows.get(t, 0)) for t in ACTOR_ROLE_TASKS},
            "updated": list(eligible),
        }
        if len(eligible) >= 2:
            g = {t: float(self.ema_norm[t]) for t in eligible}
            big = {t: self.weights[t] * g[t] for t in eligible}
            mean_g = sum(big.values()) / len(big)
            ratio = {
                t: (self.ema_loss[t] / self.initial_loss[t] if self.initial_loss[t] else 1.0)
                for t in eligible
            }
            mean_ratio = sum(ratio.values()) / len(ratio)
            ideal = {}
            for t in eligible:
                rate = ratio[t] / mean_ratio if mean_ratio > 0 else 1.0
                target = mean_g * max(0.0, rate) ** float(c.alpha)
                ideal[t] = min(c.max_weight, max(c.min_weight, target / g[t]))
                self.weights[t] = math.exp(
                    (1 - c.lr) * math.log(self.weights[t]) + c.lr * math.log(ideal[t])
                )
                self.task_updates[t] += 1
            self._project(set(eligible))
            self.updates += 1
            entry.update(
                ema_grad_norm=g,
                weighted_ema_grad_norm=big,
                mean_weighted_grad_norm=mean_g,
                ideal_weight=ideal,
            )
        entry["weights_next"] = dict(self.weights)
        entry["scale_sum"] = self.scale_sum()
        self.last = entry
        return entry

    def summary(self) -> dict[str, Any]:
        return {
            **self.config.state_dict(),
            "role_scales": {t: self.scales[t] for t in ACTOR_ROLE_TASKS},
            "scale_target": self.target_scale,
            "initial_weights": dict(self.initial_weights),
            "final_weights": dict(self.weights),
            "scale_sum": self.scale_sum(),
            "updates": int(self.updates),
            "task_updates": dict(self.task_updates),
            "task_holds": dict(self.task_holds),
            "ema_grad_norm": dict(self.ema_norm),
            "initial_loss": dict(self.initial_loss),
            "ema_loss": dict(self.ema_loss),
        }
