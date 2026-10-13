"""Masked simplex actor. PPO scores requested shares before native apportionment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Dirichlet

from anthropocene_rl.autonomous.contracts import FEATURE_WIDTH


class MaskedDirichlet:
    """Dirichlet density on each active face, including point-mass faces.

    Inactive coordinates are canonical zero. Sampling visits only active
    coordinates; all absent and singleton faces consume no generator state.
    The density is with respect to the active simplex's (K-1) coordinates.
    """

    def __init__(self, concentration: Tensor, mask: Tensor):
        if concentration.shape != mask.shape or concentration.ndim < 1:
            raise ValueError("Concentration and mask shapes must match")
        self.alpha = concentration.to(torch.float64)
        self.mask = mask.to(torch.bool)
        if torch.any(~torch.isfinite(self.alpha[self.mask])) or torch.any(
            self.alpha[self.mask] <= 0
        ):
            raise ValueError("Active concentration must be finite and positive")

    def sample(self, *, generator: torch.Generator | None = None, deterministic=False) -> Tensor:
        rows = []
        for alpha, mask in zip(
            self.alpha.reshape(-1, self.alpha.shape[-1]),
            self.mask.reshape(-1, self.mask.shape[-1]),
            strict=True,
        ):
            row = torch.zeros_like(alpha)
            count = int(mask.sum())
            if count == 1:
                row[mask] = 1
            elif count > 1:
                active = alpha[mask]
                # CPU/GPU generator explicit; no draws for padded dimensions.
                values = (
                    active if deterministic else torch._standard_gamma(active, generator=generator)
                )
                values = values.clamp_min(torch.finfo(values.dtype).tiny)
                row[mask] = values / values.sum()
            rows.append(row)
        return torch.stack(rows).reshape(self.alpha.shape).detach()

    def log_prob(self, shares: Tensor) -> Tensor:
        if shares.shape != self.alpha.shape:
            raise ValueError("Requested shares shape mismatch")
        scores = []
        for alpha, mask, row in zip(
            self.alpha.reshape(-1, self.alpha.shape[-1]),
            self.mask.reshape(-1, self.mask.shape[-1]),
            shares.to(torch.float64).reshape(-1, shares.shape[-1]),
            strict=True,
        ):
            n = int(mask.sum())
            if (
                torch.any(~torch.isfinite(row))
                or torch.any(row[~mask] != 0)
                or torch.any(row[mask] <= 0)
                or not torch.isclose(row.sum(), row.new_tensor(float(n > 0)), atol=1e-8, rtol=0)
            ):
                raise ValueError("Requested action is outside its masked simplex")
            scores.append(
                Dirichlet(alpha[mask]).log_prob(row[mask]) if n > 1 else alpha[mask].sum() * 0
            )
        return torch.stack(scores).reshape(self.alpha.shape[:-1])

    def entropy(self) -> Tensor:
        values = [
            Dirichlet(a[m]).entropy() if int(m.sum()) > 1 else a[m].sum() * 0
            for a, m in zip(
                self.alpha.reshape(-1, self.alpha.shape[-1]),
                self.mask.reshape(-1, self.mask.shape[-1]),
                strict=True,
            )
        ]
        return torch.stack(values).reshape(self.alpha.shape[:-1])


@dataclass
class ActionBatch:
    workforce: Tensor
    activities: Tensor
    controls: Tensor

    def wire_action(self, index: int = 0) -> dict:
        return {
            "workforce": self.workforce[index].detach().cpu().numpy(),
            "activities": self.activities[index].detach().cpu().numpy(),
            "controls": self.controls[index].detach().cpu().numpy(),
        }


class AutonomousActorCritic(nn.Module):
    """Mechanics model; trade is explicitly scripted/disabled for this gate."""

    def __init__(self, width=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(FEATURE_WIDTH, width), nn.Tanh(), nn.Linear(width, 13 + 13 * 15 + 3 * 2)
        )
        self.critic = nn.Sequential(nn.Linear(FEATURE_WIDTH, width), nn.Tanh(), nn.Linear(width, 1))

    def distributions(
        self, features: Tensor, workforce_mask: Tensor, activity_mask: Tensor, control_mask: Tensor
    ):
        alpha = torch.nn.functional.softplus(self.actor(features).clamp(-12, 12)) + 0.5
        wf = MaskedDirichlet(alpha[..., :13], workforce_mask)
        activity = MaskedDirichlet(alpha[..., 13:208].reshape(-1, 13, 15), activity_mask)
        controls = MaskedDirichlet(
            alpha[..., 208:].reshape(-1, 3, 2), control_mask.unsqueeze(-1).expand(-1, -1, 2)
        )
        return wf, activity, controls

    def sample(
        self,
        features,
        workforce_mask,
        activity_mask,
        control_mask,
        *,
        generator=None,
        deterministic=False,
    ):
        wf, ac, co = self.distributions(features, workforce_mask, activity_mask, control_mask)
        actions = ActionBatch(
            wf.sample(generator=generator, deterministic=deterministic),
            ac.sample(generator=generator, deterministic=deterministic),
            co.sample(generator=generator, deterministic=deterministic)[..., 0],
        )
        log_prob, entropy, value = self.evaluate(
            features, workforce_mask, activity_mask, control_mask, actions
        )
        return actions, log_prob, entropy, value

    def evaluate(self, features, workforce_mask, activity_mask, control_mask, actions):
        wf, ac, co = self.distributions(features, workforce_mask, activity_mask, control_mask)
        controls = torch.stack(
            [
                actions.controls,
                torch.where(control_mask, 1 - actions.controls, torch.zeros_like(actions.controls)),
            ],
            dim=-1,
        )
        return (
            wf.log_prob(actions.workforce)
            + ac.log_prob(actions.activities).sum(-1)
            + co.log_prob(controls).sum(-1),
            wf.entropy() + ac.entropy().sum(-1) + co.entropy().sum(-1),
            self.critic(features).squeeze(-1),
        )


def ppo_mechanics_update(
    model,
    optimizer,
    features,
    workforce_mask,
    activity_mask,
    control_mask,
    actions,
    old_log_prob,
    advantages,
    returns,
    clip=0.2,
):
    """One bounded optimizer step; caller owns fixed rollout masks and requested actions."""
    logp, entropy, values = model.evaluate(
        features, workforce_mask, activity_mask, control_mask, actions
    )
    log_ratio = logp - old_log_prob.detach()
    ratio = log_ratio.exp()
    objective = torch.minimum(
        ratio * advantages.detach(), ratio.clamp(1 - clip, 1 + clip) * advantages.detach()
    )
    loss = (
        -objective.mean()
        + 0.5 * (values - returns.detach()).square().mean()
        - 0.001 * entropy.mean()
    )
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite PPO loss")
    optimizer.zero_grad()
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "grad_norm": float(grad),
        "approx_kl": float(((ratio - 1) - log_ratio).mean().detach()),
    }
