"""Explicit critic-only reset for expanded frozen-actor native calibration."""

import torch

from anthropocene_rl.autonomous_v5.causal import protected_state


def reset_critic_for_collection(trainer, probe):
    before = protected_state(trainer)
    trainer.model.critic.load_state_dict(trainer.model.initial_normalized_critic.state_dict())
    for parameter in trainer.model.critic.parameters():
        trainer.optimizer.state.pop(parameter, None)
    with torch.no_grad():
        trainer.model.value_offset.zero_()
        trainer.model.value_scale.fill_(1)
        actual = trainer.model.value(probe)
        reference = trainer.model.reference_critic(probe["x"]).squeeze(-1)
    if not torch.equal(actual, reference) or protected_state(trainer) != before:
        raise ValueError("Critic reset changed protected state or original frozen-tail semantics")
    return dict(
        protected_before=before,
        protected_after=protected_state(trainer),
        reference_value_bit_exact=True,
        actor_optimizer_reset=False,
        live_critic_optimizer_reset=True,
        native_resolves=0,
    )
