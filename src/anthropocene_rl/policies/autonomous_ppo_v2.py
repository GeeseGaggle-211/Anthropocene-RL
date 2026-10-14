"""Autonomous actor v2: exact mixed categoricals and legal reverse-KL recovery."""

import torch
from torch.distributions import Categorical

from anthropocene_rl.autonomous.policy import MaskedDirichlet
from anthropocene_rl.autonomous_v4.policy import Action, AutonomousPPO


def distribution(logits, mask, uniform_mix):
    if not torch.all(mask.any(-1)):
        raise ValueError("Empty categorical face")
    logq = logits.masked_fill(~mask, -torch.inf).log_softmax(-1)
    n = mask.sum(-1)
    uniform = mask / n[..., None]
    probs = (1 - uniform_mix) * logq.exp() + uniform_mix * uniform
    reverse_kl = -(logq.masked_fill(~mask, 0) * uniform).sum(-1) - n.log()
    # logits construction keeps singleton logp/entropy exactly zero; probs= clamps 1.
    mixed_logits = (
        probs.clamp_min(torch.finfo(probs.dtype).tiny).log().masked_fill(~mask, -torch.inf)
    )
    return Categorical(logits=mixed_logits), reverse_kl


def choose(logits, mask, choice, *, uniform_mix, generator, deterministic):
    dist, _ = distribution(logits, mask, uniform_mix)
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
        raise ValueError("Illegal mixed categorical choice")
    return choice, dist.log_prob(choice)


class AutonomousPPOV2(AutonomousPPO):
    """Same MLP/continuous heads and native v4 contract; new learner semantics."""

    def __init__(self, width=256, *, uniform_mix=0.05):
        super().__init__(width)
        if not 0 < uniform_mix < 1:
            raise ValueError("Exploration mixture must be in (0,1)")
        self.uniform_mix = uniform_mix

    def categorical_terms(self, obs, h=None):
        """Exact joint entropy; separate structural U||raw-policy regularizer."""
        h = self.actor(obs["x"]) if h is None else h
        goods = obs["GoodMask"]
        give = goods & (obs["Stocks"] >= 1e-6)
        eligible = obs["CounterpartyMask"].any(-1) & give.any(-1) & (goods.sum(-1) > 1)
        enabled = torch.stack([torch.ones_like(eligible), eligible], -1)

        def face(name, mask, active=None):
            mask = mask.clone()
            if active is not None:
                mask[~active] = False
                mask[~active, 0] = True
            return distribution(self.heads[name](h), mask, self.uniform_mix)

        move, r_move = face("movement", obs["MovementMask"])
        enable, r_enable = face("enabled", enabled)
        partner, r_partner = face("partner", obs["CounterpartyMask"], eligible)
        offered, r_give = face("give", give, eligible)
        lot, r_lot = face("lot", torch.ones(len(h), 3, dtype=torch.bool, device=h.device), eligible)
        price, r_price = face(
            "price", torch.ones(len(h), 3, dtype=torch.bool, device=h.device), eligible
        )
        # Enumerate every possible offered good, including the derivative of its weight.
        g = goods.shape[-1]
        want_mask = goods[:, None, :].expand(-1, g, -1).clone()
        want_mask &= ~torch.eye(g, dtype=torch.bool, device=h.device)[None]
        active_give = give & eligible[:, None]
        want_mask[~active_give] = False
        want_mask[..., 0] |= ~active_give
        wanted, r_want = distribution(
            self.heads["want"](h)[:, None, :].expand(-1, g, -1), want_mask, self.uniform_mix
        )
        want_entropy = (offered.probs * wanted.entropy()).sum(-1)
        joint_entropy = (
            move.entropy()
            + enable.entropy()
            + enable.probs[:, 1]
            * (
                partner.entropy()
                + offered.entropy()
                + want_entropy
                + lot.entropy()
                + price.entropy()
            )
        )
        # Uniform prospective-give weighting: no incentive to avoid a difficult face.
        want_regularizer = (r_want * give).sum(-1) / give.sum(-1).clamp_min(1)
        regularizer = (
            r_move + r_enable + eligible * (r_partner + r_give + want_regularizer + r_lot + r_price)
        )
        return joint_entropy, regularizer

    def regularization(self, obs):
        return self.categorical_terms(obs)[1]

    def forward(self, obs, action=None, *, generator=None, deterministic=False):
        h = self.actor(obs["x"])
        alpha = self.concentrations(self.heads["simplex"](h))
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
        move, lp = choose(
            self.heads["movement"](h),
            obs["MovementMask"],
            None if action is None else action.movement,
            uniform_mix=self.uniform_mix,
            **kwargs,
        )
        logp += lp
        goods = obs["GoodMask"]
        give = goods & (obs["Stocks"] >= 1e-6)
        enable = torch.stack(
            [
                torch.ones(len(h), dtype=torch.bool, device=h.device),
                obs["CounterpartyMask"].any(-1) & give.any(-1) & (goods.sum(-1) > 1),
            ],
            -1,
        )
        choices = []
        for i, (name, mask) in enumerate(
            [
                ("enabled", enable),
                ("partner", obs["CounterpartyMask"]),
                ("give", give),
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
            choice, lp = choose(
                self.heads[name](h),
                mask,
                None if action is None else action.trade[:, i],
                uniform_mix=self.uniform_mix,
                **kwargs,
            )
            choices.append(choice)
            logp = logp + lp
        entropy = entropy + self.categorical_terms(obs, h)[0]
        return (
            Action(workforce, activities, controls, move, torch.stack(choices, -1)),
            logp,
            entropy,
            self.value(obs),
        )

    def concentrations(self, raw):
        # The omitted-setting legacy path keeps the exact original operations.
        return torch.nn.functional.softplus(raw.clamp(-12, 12)) + 0.5
