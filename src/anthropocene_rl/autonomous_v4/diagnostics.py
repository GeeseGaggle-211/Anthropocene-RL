"""Read-only health metrics and disposable, observation-dependent recovery probes."""

import copy
import math

import torch

from anthropocene_rl.autonomous.policy import MaskedDirichlet


def _layers(trunk, x):
    result = []
    with torch.no_grad():
        for layer in trunk:
            x = layer(x)
            if isinstance(layer, torch.nn.Tanh):
                singular = torch.linalg.svdvals(x - x.mean(0)).double()
                p = singular / singular.sum().clamp_min(1e-20)
                rank = float((-(p * p.clamp_min(1e-20).log()).sum()).exp())
                result.append(
                    dict(
                        saturated_fraction=float((x.abs() >= 0.99).float().mean()),
                        mean_derivative=float((1 - x.square()).mean()),
                        fixed_saturated_units=float(
                            (torch.all(x.abs() >= 0.99, dim=0) & (x.std(0) < 1e-5)).float().mean()
                        ),
                        observation_std=float(x.std(0).mean()),
                        effective_rank=rank,
                    )
                )
    return result


def _distribution(logits, mask, uniform_mix=0.0):
    if uniform_mix:
        from anthropocene_rl.policies.autonomous_ppo_v2 import distribution

        return distribution(logits, mask, uniform_mix)[0]
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -torch.inf))


def _categorical(logits, mask, uniform_mix=0.0):
    eligible = mask.sum(-1) > 1
    if not eligible.any():
        return dict(eligible=0)
    m = mask[eligible]
    dist = _distribution(logits[eligible], m, uniform_mix)
    return dict(
        eligible=int(eligible.sum()),
        normalized_entropy=float((dist.entropy() / m.sum(-1).log()).mean()),
        maximum_probability=float(dist.probs.max(-1).values.mean()),
        probability_std=float(dist.probs.std(0, unbiased=False).mean()),
    )


@torch.no_grad()
def health(model, obs):
    h = model.actor(obs["x"])
    raw = model.heads["simplex"](h)
    alpha = torch.nn.functional.softplus(raw.clamp(-12, 12)) + 0.5
    active = torch.cat(
        [
            obs["WorkforceMask"],
            obs["ActivityMask"].flatten(1),
            obs["ControlMask"][..., None].expand(-1, -1, 2).flatten(1),
        ],
        -1,
    )
    simplex = {}
    for name, _offset, mask in [
        ("workforce", 0, obs["WorkforceMask"]),
        ("activities", 13, obs["ActivityMask"].reshape(-1, 15)),
        ("controls", 208, obs["ControlMask"][..., None].expand(-1, -1, 2).reshape(-1, 2)),
    ]:
        if name == "workforce":
            a = alpha[:, :13]
        elif name == "activities":
            a = alpha[:, 13:208].reshape(-1, 15)
        else:
            a = alpha[:, 208:].reshape(-1, 2)
        eligible = mask.sum(-1) > 1
        if not eligible.any():
            simplex[name] = dict(eligible=0)
            continue
        a = a[eligible] * mask[eligible]
        total = a.sum(-1, keepdim=True)
        mean = a / total
        variance = mean * (1 - mean) / (total + 1)
        simplex[name] = dict(
            eligible=int(eligible.sum()),
            concentration_mean=float(total.mean()),
            concentration_max=float(total.max()),
            coordinate_std=float(variance[mask[eligible]].sqrt().mean()),
            requested_mean_max=float(mean.max(-1).values.mean()),
        )
    goods = obs["GoodMask"]
    give = goods & (obs["Stocks"] >= 1e-6)
    enabled = torch.stack(
        [
            torch.ones(len(h), dtype=torch.bool),
            obs["CounterpartyMask"].any(-1) & give.any(-1) & (goods.sum(-1) > 1),
        ],
        -1,
    )
    # Prospective conditional heads: eligible even if this sampled policy declines a quote.
    want = goods.clone()
    offered = give.long().argmax(-1)
    want.scatter_(1, offered[:, None], False)
    masks = dict(
        movement=obs["MovementMask"],
        enabled=enabled,
        partner=obs["CounterpartyMask"],
        give=give,
        want=want,
        lot=torch.ones(len(h), 3, dtype=torch.bool),
        price=torch.ones(len(h), 3, dtype=torch.bool),
    )
    cats = {
        name: _categorical(model.heads[name](h), mask, getattr(model, "uniform_mix", 0.0))
        for name, mask in masks.items()
    }
    with torch.enable_grad():
        x = obs["x"].clone().requires_grad_()
        value = model.value(dict(obs, x=x))
        gradient = torch.autograd.grad(value.sum(), x)[0]
    return dict(
        actor_layers=_layers(model.actor, obs["x"]),
        critic_layers=_layers(model.critic[0], obs["x"]),
        active_clamp_fraction=float((raw[active].abs() >= 12).float().mean()),
        active_raw_min=float(raw[active].min()),
        active_raw_max=float(raw[active].max()),
        simplex=simplex,
        categorical=cats,
        value_std=float(value.std()),
        value_input_gradient=float(gradient.norm(dim=-1).mean()),
    )


def probe_face(obs):
    masks, inverse, counts = torch.unique(
        obs["WorkforceMask"], dim=0, return_inverse=True, return_counts=True
    )
    legal = counts.clone()
    legal[masks.sum(-1) < 2] = 0
    face = int(legal.argmax())
    ix = torch.where(inverse == face)[0]
    if len(ix) < 16:
        raise ValueError("Recovery probe needs >=16 rows on one nondegenerate legal face")
    subset = {k: v[ix] for k, v in obs.items()}
    feature = int(subset["x"].std(0).argmax())
    order = torch.argsort(subset["x"][:, feature], stable=True)
    # Exactly balanced labels; discard one row if needed.
    if len(order) % 2:
        order = order[:-1]
    subset = {k: v[order] for k, v in subset.items()}
    labels = torch.ones(len(order))
    labels[: len(order) // 2] = -1
    roles = torch.where(masks[face])[0][:2]
    return subset, labels, roles, feature


def recovery_probe(model, obs, *, steps=16, lr=0.001, critic_lr=None):
    obs, labels, roles, feature = probe_face(obs)
    actor = copy.deepcopy(model)
    critic = copy.deepcopy(model)
    actor_parameters = [*actor.actor.parameters(), *actor.heads["simplex"].parameters()]
    optimizer = torch.optim.Adam(actor_parameters, lr=lr, eps=1e-5)
    target = obs["WorkforceMask"].double()
    target = target * 0.15 / (target.sum(-1, keepdim=True) - 1)
    selected = torch.where(labels > 0, roles[0], roles[1])
    target.scatter_(1, selected[:, None], 0.85)

    def actor_state():
        raw = actor.heads["simplex"](actor.actor(obs["x"]))[:, :13]
        alpha = torch.nn.functional.softplus(raw.clamp(-12, 12)) + 0.5
        loss = -MaskedDirichlet(alpha, obs["WorkforceMask"]).log_prob(target).mean()
        mean = alpha * obs["WorkforceMask"]
        mean = mean / mean.sum(-1, keepdim=True)
        separation = ((mean[:, roles[0]] - mean[:, roles[1]]) * labels).mean()
        return loss, separation

    initial_loss, initial_gap = (float(x.detach()) for x in actor_state())
    norms = []
    for _ in range(steps):
        loss, _ = actor_state()
        optimizer.zero_grad()
        loss.backward()
        norms.append(
            float(torch.nn.utils.clip_grad_norm_(actor_parameters, 10.0, error_if_nonfinite=True))
        )
        optimizer.step()
    final_loss, final_gap = (float(x.detach()) for x in actor_state())
    with torch.no_grad():
        target_value = critic.value(obs).detach() + labels
    optimizer = torch.optim.Adam(
        critic.critic.parameters(), lr=lr if critic_lr is None else critic_lr, eps=1e-5
    )
    for _ in range(steps):
        loss = (critic.value(obs) - target_value).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.critic.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
    mse = float((critic.value(obs) - target_value).square().mean().detach())
    improvement = (initial_loss - final_loss) / max(1, abs(initial_loss))
    return dict(
        actor_pass=improvement >= 0.005 and final_gap - initial_gap >= 1e-4 and max(norms) > 0,
        critic_pass=math.isfinite(mse) and mse < 0.99,
        rows=len(labels),
        feature=feature,
        roles=roles.tolist(),
        actor_loss_before=initial_loss,
        actor_loss_after=final_loss,
        actor_relative_improvement=improvement,
        separation_before=initial_gap,
        separation_after=final_gap,
        actor_grad_max=max(norms),
        critic_residual_mse=mse,
    )


def categorical_recovery_probes(model, obs, *, steps=16, lr=0.001):
    """Prospective conditional heads, scored through their real masked likelihood."""
    with torch.no_grad():
        h = model.actor(obs["x"])
        give = obs["GoodMask"] & (obs["Stocks"] >= 1e-6)
        enable = obs["CounterpartyMask"].any(-1) & give.any(-1) & (obs["GoodMask"].sum(-1) > 1)
        want = obs["GoodMask"].clone()
        offered = give.long().argmax(-1)
        want.scatter_(1, offered[:, None], False)
        masks = dict(
            movement=obs["MovementMask"],
            enabled=torch.stack([torch.ones_like(enable), enable], -1),
            partner=obs["CounterpartyMask"],
            give=give,
            want=want,
            lot=torch.ones(len(h), 3, dtype=torch.bool),
            price=torch.ones(len(h), 3, dtype=torch.bool),
        )
        for name in ["partner", "give", "want", "lot", "price"]:
            masks[name] = masks[name] & enable[:, None]
    reports = {}
    for name, mask in masks.items():
        faces, inverse, counts = torch.unique(mask, dim=0, return_inverse=True, return_counts=True)
        counts[faces.sum(-1) < 2] = 0
        face = int(counts.argmax())
        ix = torch.where(inverse == face)[0]
        if counts[face] < 16:
            reports[name] = dict(covered=False, eligible=int((mask.sum(-1) > 1).sum()))
            continue
        x = obs["x"][ix]
        feature = int(x.std(0).argmax())
        order = torch.argsort(x[:, feature], stable=True)
        if len(order) % 2:
            order = order[:-1]
        x = x[order]
        m = mask[ix][order]
        labels = torch.ones(len(x))
        labels[: len(x) // 2] = -1
        choices = torch.where(faces[face])[0][:2]
        target = torch.where(labels > 0, choices[0], choices[1])
        clone = copy.deepcopy(model)
        parameters = [*clone.actor.parameters(), *clone.heads[name].parameters()]
        optimizer = torch.optim.Adam(parameters, lr=lr, eps=1e-5)

        def measure(
            clone=clone, name=name, x=x, m=m, target=target, choices=choices, labels=labels
        ):
            dist = _distribution(
                clone.heads[name](clone.actor(x)), m, getattr(clone, "uniform_mix", 0.0)
            )
            return -dist.log_prob(target).mean(), (
                (dist.probs[:, choices[0]] - dist.probs[:, choices[1]]) * labels
            ).mean()

        before, gap_before = (float(v.detach()) for v in measure())
        for _ in range(steps):
            loss, _ = measure()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True)
            optimizer.step()
        after, gap_after = (float(v.detach()) for v in measure())
        improvement = (before - after) / max(1, abs(before))
        reports[name] = dict(
            covered=True,
            rows=len(x),
            passed=improvement >= 0.005 and gap_after - gap_before >= 1e-4,
            loss_before=before,
            loss_after=after,
            separation_before=gap_before,
            separation_after=gap_after,
        )
    return reports


def conditional_simplex_probes(model, obs, *, steps=16, lr=0.001):
    """Every covered activity allocation and Beta control uses its actual density."""
    specs = [
        (f"activity_{r}", slice(13 + r * 15, 13 + (r + 1) * 15), obs["ActivityMask"][:, r])
        for r in range(13)
    ]
    specs += [
        (
            f"control_{c}",
            slice(208 + c * 2, 210 + c * 2),
            obs["ControlMask"][:, c, None].expand(-1, 2),
        )
        for c in range(3)
    ]
    reports = {}
    for name, span, mask in specs:
        faces, inverse, counts = torch.unique(mask, dim=0, return_inverse=True, return_counts=True)
        counts[faces.sum(-1) < 2] = 0
        face = int(counts.argmax())
        ix = torch.where(inverse == face)[0]
        if counts[face] < 16:
            reports[name] = dict(covered=False, eligible=int((mask.sum(-1) > 1).sum()))
            continue
        x = obs["x"][ix]
        feature = int(x.std(0).argmax())
        order = torch.argsort(x[:, feature], stable=True)
        if len(order) % 2:
            order = order[:-1]
        x = x[order]
        m = mask[ix][order]
        labels = torch.ones(len(x))
        labels[: len(x) // 2] = -1
        choices = torch.where(faces[face])[0][:2]
        target = m.double() * 0.15 / (m.sum(-1, keepdim=True) - 1)
        selected = torch.where(labels > 0, choices[0], choices[1])
        target.scatter_(1, selected[:, None], 0.85)
        clone = copy.deepcopy(model)
        parameters = [*clone.actor.parameters(), *clone.heads["simplex"].parameters()]
        optimizer = torch.optim.Adam(parameters, lr=lr, eps=1e-5)

        def measure(
            clone=clone, x=x, span=span, m=m, target=target, choices=choices, labels=labels
        ):
            raw = clone.heads["simplex"](clone.actor(x))[:, span]
            alpha = torch.nn.functional.softplus(raw.clamp(-12, 12)) + 0.5
            loss = -MaskedDirichlet(alpha, m).log_prob(target).mean()
            mean = alpha * m
            mean = mean / mean.sum(-1, keepdim=True)
            return loss, ((mean[:, choices[0]] - mean[:, choices[1]]) * labels).mean()

        before, gap_before = (float(v.detach()) for v in measure())
        for _ in range(steps):
            loss, _ = measure()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True)
            optimizer.step()
        after, gap_after = (float(v.detach()) for v in measure())
        reports[name] = dict(
            covered=True,
            rows=len(x),
            passed=(before - after) / max(1, abs(before)) >= 0.005
            and gap_after - gap_before >= 1e-4,
            loss_before=before,
            loss_after=after,
            separation_before=gap_before,
            separation_after=gap_after,
        )
    return reports


def full_recovery_probe(model, obs, *, steps=16, lr=0.001, critic_lr=None):
    result = recovery_probe(model, obs, steps=steps, lr=lr, critic_lr=critic_lr)
    result["categorical"] = categorical_recovery_probes(model, obs, steps=steps, lr=lr)
    result["conditional_simplex"] = conditional_simplex_probes(model, obs, steps=steps, lr=lr)
    result["passed"] = (
        result["actor_pass"]
        and result["critic_pass"]
        and all(
            r["passed"]
            for group in [result["categorical"], result["conditional_simplex"]]
            for r in group.values()
            if r["covered"]
        )
    )
    return result


def optimizer_stall(updates):
    if len(updates) < 8:
        return False
    recent = updates[-8:]
    actor = all(
        m["advantage_std"] > 1e-8
        and m["actor_first_layer_grad"] > 1e-10
        and m["actor_step_l2"] < 1e-12
        for m in recent
    )
    critic = all(
        m["critic_residual_mse"] > 1e-6
        and m["critic_first_layer_grad"] > 1e-10
        and m["critic_step_l2"] < 1e-12
        for m in recent
    )
    return actor or critic
