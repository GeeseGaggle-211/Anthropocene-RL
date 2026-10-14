"""Strict raw-unit value readiness and actor-protected normalized critic fitting."""

import hashlib
import math
import time

import torch

from anthropocene_rl.autonomous_v5.causal import protected_state


def protected_calibration(trainer):
    digest = hashlib.sha256(protected_state(trainer).encode())
    digest.update(repr(trainer.config).encode())
    for name in ("value_offset", "value_scale"):
        digest.update(getattr(trainer.model, name).cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def metrics(trainer, fitting, validation):
    xt, yt = fitting
    xv, yv = validation
    if len(yt) == 0 or len(yv) < 2:
        raise ValueError("Nonempty TRAIN fitting and validation rows required")
    error = trainer.model.value(xv) - yv
    variance = float(yv.var(unbiased=False))
    if not math.isfinite(variance) or variance <= 1e-8:
        raise ValueError("Degenerate validation return variance")
    mse = float(error.square().mean())
    return dict(
        mse=mse,
        return_variance=variance,
        mse_over_return_variance=mse / variance,
        explained_variance=1 - float(error.var(unbiased=False)) / variance,
        mean_value_bias=float(error.mean()),
        constant_mse=float((yv - yt.mean()).square().mean()),
        fitting_mse=float((trainer.model.value(xt) - yt).square().mean()),
    )


def ready(value, original_mse):
    return (
        all(math.isfinite(v) for v in value.values())
        and value["explained_variance"] >= 0.60
        and value["mse_over_return_variance"] <= 0.40
        and abs(value["mean_value_bias"]) <= 2.0
        and value["mse"] <= original_mse
    )


@torch.no_grad()
def fit_intercept(trainer, fitting):
    before = protected_calibration(trainer)
    x, y = fitting
    if len(y) == 0 or not torch.isfinite(y).all():
        raise ValueError("Nonempty finite fitting TRAIN targets required")
    original_values = trainer.model.value(x).clone()
    head_before = trainer.model.critic[1].bias.clone()
    delta = (y - trainer.model.value(x)).mean()
    if not torch.isfinite(delta):
        raise ValueError("Nonfinite TRAIN intercept residual")
    trainer.model.critic[1].bias.add_(delta / trainer.model.value_scale)
    if protected_calibration(trainer) != before:
        raise ValueError("Intercept fit changed actor/anchors/optimizer/RNG/value scale")
    return dict(
        raw_unit_delta=float(delta),
        raw_unit_applied_delta=float((trainer.model.value(x) - original_values).mean()),
        normalized_head_bias_before=float(head_before),
        normalized_head_bias_after=float(trainer.model.critic[1].bias),
        normalized_head_delta=float(delta / trainer.model.value_scale),
        fitting_rows=len(y),
        fitting_residual_mean=float((trainer.model.value(x) - y).mean()),
    )


def fit_critic(
    trainer, fitting, validation, *, original_mse, max_epochs=256, seed=711, deadline=None
):
    before = protected_calibration(trainer)
    rng = torch.Generator().manual_seed(seed)
    x, y = fitting
    parameters = list(trainer.model.critic.parameters())
    actor_parameters = [*trainer.model.actor.parameters(), *trainer.model.heads.parameters()]
    actor_flags = [p.requires_grad for p in actor_parameters]
    for p in actor_parameters:
        p.requires_grad_(False)
    records = []
    steps = 0
    try:
        for epoch in range(max_epochs):
            for ix in torch.randperm(len(y), generator=rng).split(32):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Critic calibration exceeded the shared deadline")
                loss = trainer.critic_loss(
                    trainer.model.value({k: v[ix] for k, v in x.items()}), y[ix]
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite normalized critic loss")
                trainer.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 0.5, error_if_nonfinite=True)
                trainer.optimizer.step()
                steps += 1
            if (epoch + 1) % 32 == 0 or epoch + 1 == max_epochs:
                value = metrics(trainer, fitting, validation)
                records.append(
                    dict(
                        epoch=epoch + 1,
                        gradient_norm=float(norm),
                        **value,
                        ready=ready(value, original_mse),
                    )
                )
                if records[-1]["ready"]:
                    break
    finally:
        for p, flag in zip(actor_parameters, actor_flags, strict=True):
            p.requires_grad_(flag)
    if protected_calibration(trainer) != before:
        raise ValueError("Calibration changed protected actor/anchors/Adam/RNG/value scaling")
    return dict(
        trace=records,
        final=metrics(trainer, fitting, validation),
        epochs=epoch + 1,
        optimizer_steps=steps,
        protected_before=before,
        protected_after=protected_calibration(trainer),
        readiness="EV>=.60; MSE/Var(return)<=.40; |raw-unit mean bias|<=2; nonworse original",
    )
