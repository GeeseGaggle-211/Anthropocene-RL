"""TRAIN-world CV of a fixed initial-trunk residual output; no validation tuning."""

import copy

import torch

from anthropocene_rl.autonomous_v6.readiness import protected_calibration

LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def displacement(phi, reference, target, strength, offset, scale):
    """Penalize output-weight displacement from the initial head; free intercept."""
    phi, reference, target = phi.double(), reference.double(), target.double()
    z = (target - offset) / scale - reference
    center = phi.mean(0)
    residual_mean = z.mean()
    a, b = phi - center, z - residual_mean
    weight = torch.linalg.solve(a.T @ a + strength * torch.eye(a.shape[1], dtype=a.dtype), a.T @ b)
    bias = residual_mean - center @ weight
    return weight, bias


def cv_select(phi, reference, target, worlds, *, lambdas=LAMBDAS, deadline_check=lambda: None):
    if len(target) != len(worlds) or len(torch.unique(worlds)) != 4:
        raise ValueError("Ridge CV requires four fitting worlds and aligned rows")
    records = []
    for strength in lambdas:
        errors = []
        for world in torch.unique(worlds).tolist():
            deadline_check()
            fitting, validation = worlds != world, worlds == world
            # Fold-local target statistics; no held-out-world normalization.
            offset = target[fitting].double().mean()
            scale = target[fitting].double().std(unbiased=False).clamp_min(1)
            weight, bias = displacement(
                phi[fitting], reference[fitting], target[fitting], strength, offset, scale
            )
            prediction = offset + scale * (
                reference[validation].double() + phi[validation].double() @ weight + bias
            )
            errors.append(
                dict(
                    world=world,
                    mse=float((prediction - target[validation]).square().mean()),
                    fitting_offset=float(offset),
                    fitting_scale=float(scale),
                )
            )
        records.append(
            dict(strength=strength, equal_world_mse=sum(e["mse"] for e in errors) / 4, folds=errors)
        )
    selected = min(records, key=lambda r: (r["equal_world_mse"], r["strength"]))
    return selected["strength"], records


def fit_ridge(trainer, fitting, worlds):
    before = protected_calibration(trainer)
    x, y = fitting
    with torch.no_grad():
        phi = trainer.model.initial_normalized_critic[0](x["x"])
        reference = trainer.model.reference_critic(x["x"]).squeeze(-1)
    strength, records = cv_select(
        phi, reference, y, worlds, deadline_check=trainer.check_update_deadline
    )
    weight, bias = displacement(
        phi,
        reference,
        y,
        strength,
        trainer.model.value_offset.double(),
        trainer.model.value_scale.double(),
    )
    trainer.model.critic.load_state_dict(
        copy.deepcopy(trainer.model.initial_normalized_critic.state_dict())
    )
    with torch.no_grad():
        trainer.model.critic[1].weight.add_(weight.to(trainer.model.critic[1].weight.dtype)[None])
        trainer.model.critic[1].bias.add_(bias.to(trainer.model.critic[1].bias.dtype))
    for p in trainer.model.critic.parameters():
        trainer.optimizer.state.pop(p, None)
    if protected_calibration(trainer) != before:
        raise ValueError("Ridge changed protected actor/anchors/Adam/RNG/config/value buffers")
    return dict(
        selected_strength=strength,
        training_world_cv=records,
        fitting_rows=len(y),
        features="fixed initial normalized critic trunk; receipt weights remain zero",
        actor_optimizer_reset=False,
        critic_optimizer_reset=True,
        protected_before=before,
        protected_after=protected_calibration(trainer),
        validation_used_for_selection=False,
    )
