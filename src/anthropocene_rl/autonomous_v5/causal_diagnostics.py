"""Read-only fixed-state diagnostics and disposable two-direction adaptation probes."""

import copy

import torch

from anthropocene_rl.autonomous.policy import MaskedDirichlet
from anthropocene_rl.autonomous_v4.diagnostics import health, probe_face


@torch.no_grad()
def signature(model, obs):
    h = model.actor(obs["x"])
    raw = model.heads["simplex"](h)
    alpha = (
        model.concentrations(raw)
        if hasattr(model, "concentrations")
        else torch.nn.functional.softplus(raw.clamp(-12, 12)) + 0.5
    )
    result = {}
    for key, a, mask in [
        ("workforce", alpha[:, :13], obs["WorkforceMask"]),
        ("activities", alpha[:, 13:208].reshape(-1, 13, 15), obs["ActivityMask"]),
        ("controls", alpha[:, 208:].reshape(-1, 3, 2), obs["ControlMask"][..., None]),
    ]:
        a = a * mask
        result[key] = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
    # Raw categorical probabilities are descriptive; actual conditional masks
    # and exploration mixing are measured by health(), not these signatures.
    for name in ["movement", "enabled", "partner", "give", "want", "lot", "price"]:
        result[name] = model.heads[name](h).softmax(-1)
    return result


def snapshot_health(model, obs, mode):
    x = {k: v.clone() for k, v in obs.items()}
    if mode == "receipt_none":
        x["x"][:, 761:] = 0
    result = health(model, x)
    with torch.no_grad():
        ablated = dict(x, x=x["x"].clone())
        ablated["x"][:, 761:] = 0
        a, b = signature(model, x), signature(model, ablated)
        result["receipt_sensitivity"] = {k: float((a[k] - b[k]).abs().mean()) for k in a}
        result["receipt_value_sensitivity"] = float(
            (model.value(x) - model.value(ablated)).abs().mean()
        )
    return result


def signature_change(before, after):
    return {k: float((before[k] - after[k]).abs().mean()) for k in before}


def reversal_probe(model, obs, *, steps=16, lr=0.001):
    """Retained model never changes; test both directions through real densities."""
    subset, labels, roles, feature = probe_face(obs)
    clone = copy.deepcopy(model)
    parameters = [*clone.actor.parameters(), *clone.heads["simplex"].parameters()]
    optimizer = torch.optim.Adam(parameters, lr=lr, eps=1e-5)
    reports = []
    for direction in [1, -1]:
        selected = torch.where(labels * direction > 0, roles[0], roles[1])
        mask = subset["WorkforceMask"]
        target = mask.double() * 0.15 / (mask.sum(-1, keepdim=True) - 1)
        target.scatter_(1, selected[:, None], 0.85)

        def measure(mask=mask, target=target, direction=direction):
            raw = clone.heads["simplex"](clone.actor(subset["x"]))[:, :13]
            alpha = (
                model.concentrations(raw)
                if hasattr(model, "concentrations")
                else torch.nn.functional.softplus(raw.clamp(-12, 12)) + 0.5
            )
            loss = -MaskedDirichlet(alpha, mask).log_prob(target).mean()
            mean = alpha * mask
            mean = mean / mean.sum(-1, keepdim=True)
            gap = ((mean[:, roles[0]] - mean[:, roles[1]]) * labels * direction).mean()
            return loss, gap

        entry_loss, entry_gap = (float(v.detach()) for v in measure())
        for _ in range(steps):
            loss, _ = measure()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 0.5, error_if_nonfinite=True)
            optimizer.step()
        end_loss, end_gap = (float(v.detach()) for v in measure())
        improvement = (entry_loss - end_loss) / max(1, abs(entry_loss))
        reports.append(
            dict(
                direction=direction,
                loss_before=entry_loss,
                loss_after=end_loss,
                separation_before=entry_gap,
                separation_after=end_gap,
                relative_loss_improvement=improvement,
                final_gradient_norm=float(norm),
                passed=improvement >= 0.005 and end_gap - entry_gap >= 1e-4,
            )
        )
    return dict(
        passed=all(r["passed"] for r in reports),
        phases=reports,
        rows=len(labels),
        feature=feature,
        roles=roles.tolist(),
        disposable_optimizer_steps=steps * 2,
        retained_updates=0,
    )
