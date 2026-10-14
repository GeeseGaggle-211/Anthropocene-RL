"""Single TRAIN-only least-squares intercept fit, with every other state protected."""

import hashlib

import torch

from anthropocene_rl.autonomous_v5.causal import protected_state


def other_state(trainer):
    digest = hashlib.sha256(protected_state(trainer).encode())

    def add(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, value.shape)).encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value, dict):
            for k in sorted(value, key=repr):
                digest.update(repr(k).encode())
                add(value[k])
        elif isinstance(value, (tuple, list)):
            for item in value:
                add(item)
        else:
            digest.update(repr(value).encode())

    for name, value in trainer.model.state_dict().items():
        if name != "critic.1.bias":
            digest.update(name.encode())
            add(value)
    add(trainer.optimizer.state_dict())
    return digest.hexdigest()


@torch.no_grad()
def fit_intercept(trainer, fitting):
    """No validation argument exists; neither validation bias nor reward guides delta."""
    x, targets = fitting
    if len(targets) == 0 or not torch.isfinite(targets).all():
        raise ValueError("Empty/nonfinite TRAIN target cohort")
    before = other_state(trainer)
    residual = targets - trainer.model.value(x)
    delta = float(residual.mean())
    if not torch.isfinite(residual).all():
        raise ValueError("Nonfinite TRAIN intercept fit")
    bias = trainer.model.critic[-1].bias
    original = float(bias)
    bias.add_(delta)
    actual = float(bias) - original
    following = targets - trainer.model.value(x)
    after = other_state(trainer)
    if before != after:
        raise ValueError("Analytic fit changed other model/optimizer/RNG/counter state")
    return dict(
        delta=delta,
        applied_delta=actual,
        bias_before=original,
        bias_after=float(bias),
        train_residual_before=float(residual.mean()),
        train_residual_after=float(following.mean()),
        train_mse_before=float(residual.square().mean()),
        train_mse_after=float(following.square().mean()),
        protected_state_before=before,
        protected_state_after=after,
        closed_form_fits=1,
        optimizer_steps=0,
        native_resolves=0,
        optimizer_policy="unchanged, including critic-bias Adam",
    )
