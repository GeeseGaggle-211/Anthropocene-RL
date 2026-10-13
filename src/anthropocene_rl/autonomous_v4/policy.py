"""Full requested-action likelihood; conditional trade heads have no utility veto."""

from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from anthropocene_rl.autonomous.policy import MaskedDirichlet
from anthropocene_rl.autonomous_v4.contracts import FEATURE_WIDTH, features


@dataclass
class Action:
    workforce: torch.Tensor
    activities: torch.Tensor
    controls: torch.Tensor
    movement: torch.Tensor
    trade: torch.Tensor  # enable, partner, give, want, lot, ratio

    def index(self, index):
        return Action(*(getattr(self, f.name)[index] for f in fields(self)))

    @staticmethod
    def cat(actions):
        return Action(*(torch.cat([getattr(a, f.name) for a in actions]) for f in fields(Action)))

    def wire_action(self, index=0):
        return {f.name: getattr(self, f.name)[index].detach().cpu().numpy() for f in fields(self)}


def to_native(action, obs):
    result = {k: v for k, v in action.items() if k != "trade"}
    choices = np.asarray(action.get("trade", np.zeros(6)), dtype=int)
    if choices[0]:
        _, partner, give, want, lot, price = choices
        ratio = (0.5, 1.0, 2.0)[price]
        quantity = max(
            1e-6, min(obs["Stocks"][give], 1e6 / max(1.0, ratio)) * (0.05, 0.2, 0.5)[lot]
        )
        # Both quantities meet the native lower bound, including microscopic stock.
        quantity = max(quantity, 1e-6 / ratio)
        result["quote"] = dict(
            Counterparty=int(partner),
            Give=int(give),
            Want=int(want),
            GiveQuantity=float(quantity),
            WantQuantity=float(quantity * ratio),
        )
    return result


def inputs(observations):
    return dict(
        x=torch.tensor(np.stack([features(o) for o in observations])),
        **{
            k: torch.tensor(np.stack([o[k] for o in observations]))
            for k in (
                "WorkforceMask",
                "ActivityMask",
                "ControlMask",
                "MovementMask",
                "CounterpartyMask",
                "GoodMask",
                "Stocks",
            )
        },
    )


def categorical(logits, mask, choice=None, generator=None, deterministic=False):
    if not torch.all(mask.any(-1)):
        raise ValueError("Categorical needs at least one legal choice")
    dist = Categorical(logits=logits.masked_fill(~mask, -torch.inf))
    if choice is None:
        choice = mask.long().argmax(-1)
        active = mask.sum(-1) > 1
        if active.any():
            choice[active] = (
                dist.probs[active].argmax(-1)
                if deterministic
                else torch.multinomial(dist.probs[active], 1, generator=generator).squeeze(-1)
            )
    if (
        torch.any(choice < 0)
        or torch.any(choice >= mask.shape[-1])
        or not torch.all(mask.gather(1, choice[:, None]))
    ):
        raise ValueError("Illegal categorical action")
    return choice, dist.log_prob(choice), dist.entropy()


class AutonomousPPO(nn.Module):
    """Existing Elder's independent 256x256 tanh architecture, new versioned heads."""

    def __init__(self, width=256):
        super().__init__()
        self.width = width

        def trunk():
            result = nn.Sequential(
                nn.Linear(FEATURE_WIDTH, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh()
            )
            for layer in result:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, 2**0.5)
                    nn.init.zeros_(layer.bias)
            return result

        self.actor = trunk()
        self.critic = nn.Sequential(trunk(), nn.Linear(width, 1))
        self.heads = nn.ModuleDict(
            {
                k: nn.Linear(width, n)
                for k, n in dict(
                    simplex=214, movement=8, enabled=2, partner=8, give=11, want=11, lot=3, price=3
                ).items()
            }
        )
        for head in self.heads.values():
            nn.init.orthogonal_(head.weight, 0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def value(self, obs):
        return self.critic(obs["x"]).squeeze(-1)

    def forward(self, obs, action=None, *, generator=None, deterministic=False):
        h = self.actor(obs["x"])
        alpha = torch.nn.functional.softplus(self.heads["simplex"](h).clamp(-12, 12)) + 0.5
        masks = obs["ControlMask"]
        wf = MaskedDirichlet(alpha[:, :13], obs["WorkforceMask"])
        ac = MaskedDirichlet(alpha[:, 13:208].reshape(-1, 13, 15), obs["ActivityMask"])
        co = MaskedDirichlet(alpha[:, 208:].reshape(-1, 3, 2), masks[..., None].expand(-1, -1, 2))
        kwargs = dict(generator=generator, deterministic=deterministic)
        if action is None:
            workforce, activities = wf.sample(**kwargs), ac.sample(**kwargs)
            controls = co.sample(**kwargs)[..., 0]
        else:
            workforce, activities, controls = action.workforce, action.activities, action.controls
        beta = torch.stack([controls, torch.where(masks, 1 - controls, 0)], -1)
        logp = wf.log_prob(workforce) + ac.log_prob(activities).sum(-1) + co.log_prob(beta).sum(-1)
        entropy = wf.entropy() + ac.entropy().sum(-1) + co.entropy().sum(-1)
        move, lp, ent = categorical(
            self.heads["movement"](h),
            obs["MovementMask"],
            None if action is None else action.movement,
            **kwargs,
        )
        logp, entropy = logp + lp, entropy + ent
        goods = obs["GoodMask"]
        give_mask = goods & (obs["Stocks"] >= 1e-6)
        enabled_mask = torch.stack(
            [
                torch.ones(len(h), dtype=torch.bool, device=h.device),
                obs["CounterpartyMask"].any(-1) & give_mask.any(-1) & (goods.sum(-1) > 1),
            ],
            -1,
        )
        choices = []
        for i, (name, mask) in enumerate(
            [
                ("enabled", enabled_mask),
                ("partner", obs["CounterpartyMask"]),
                ("give", give_mask),
                ("want", goods),
                ("lot", torch.ones(len(h), 3, dtype=torch.bool, device=h.device)),
                ("price", torch.ones(len(h), 3, dtype=torch.bool, device=h.device)),
            ]
        ):
            mask = mask.clone()
            if i == 3:
                mask.scatter_(1, choices[2][:, None], False)
            if i:
                disabled = choices[0] == 0
                mask[disabled] = False
                mask[disabled, 0] = True
            choice, lp, ent = categorical(
                self.heads[name](h), mask, None if action is None else action.trade[:, i], **kwargs
            )
            choices.append(choice)
            logp = logp + lp
            if i == 0:  # No biased upstream gradient from sampled conditional entropy.
                entropy = entropy + ent
        result = Action(workforce, activities, controls, move, torch.stack(choices, -1))
        return result, logp, entropy, self.value(obs)
