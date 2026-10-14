"""Explicit critic initialization transfer; actors and physical targets stay fixed."""

import copy
import hashlib

import torch

from anthropocene_rl.autonomous_v5.causal import protected_state


def protected_actor(trainer):
    # protected_state also covers anchors; remove them in a disposable copy so
    # this intentionally different initialization has an actor-only invariant.
    digest = hashlib.sha256()
    state = copy.deepcopy(trainer)
    with torch.no_grad():
        for module in [state.model.reference_critic, state.model.initial_normalized_critic]:
            for p in module.parameters():
                p.zero_()
    digest.update(protected_state(state).encode())
    digest.update(repr(trainer.config).encode())
    for name in ["value_offset", "value_scale"]:
        digest.update(getattr(trainer.model, name).numpy().tobytes())
    return digest.hexdigest()


def transfer_critic(target, source, probes):
    before = protected_actor(target)
    source_before = protected_state(source)
    ratio = float(source.model.value_scale / target.model.value_scale)
    shift = float(
        (source.model.value_offset - target.model.value_offset) / target.model.value_scale
    )
    for name in ["critic", "reference_critic", "initial_normalized_critic"]:
        module = getattr(target.model, name)
        module.load_state_dict(copy.deepcopy(getattr(source.model, name).state_dict()))
        with torch.no_grad():
            module[1].weight.mul_(ratio)
            module[1].bias.mul_(ratio)
            if name == "critic":
                module[1].bias.add_(shift)
        if name != "critic":
            module.requires_grad_(False)
    for p in target.model.critic.parameters():
        target.optimizer.state.pop(p, None)
    with torch.no_grad():
        errors = [
            float((target.model.value(x) - source.model.value(x)).abs().max()) for x in probes
        ]
    if (
        max(errors) > 1e-4
        or protected_actor(target) != before
        or protected_state(source) != source_before
    ):
        raise ValueError("Critic transfer raw parity/actor/source protection failed")
    return dict(
        value_max_abs_errors=errors,
        raw_value_parity_tolerance=1e-4,
        output_scale_ratio=ratio,
        live_bias_shift=shift,
        anchors_replaced_and_refrozen=True,
        critic_adam_reset=True,
        actor_adam_reset=False,
        protected_actor_before=before,
        protected_actor_after=protected_actor(target),
        independent_critic_initialization=False,
    )
