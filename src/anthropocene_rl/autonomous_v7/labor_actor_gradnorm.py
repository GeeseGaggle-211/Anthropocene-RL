"""Actor BC task decomposition (and, opt-in, GradNorm balancing of its tasks).

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
