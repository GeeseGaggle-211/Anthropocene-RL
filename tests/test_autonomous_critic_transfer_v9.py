import copy

import torch
from test_autonomous_readiness_v7 import batch, config

from anthropocene_rl.autonomous_v6.critic_transfer import protected_actor, transfer_critic
from anthropocene_rl.training.autonomous_ppo_v7 import AutonomousTrainerV7


def test_critic_transfer_raw_parity_actor_adam_and_buffer_protection():
    torch.set_num_threads(1)
    source = AutonomousTrainerV7(config(), 101, {}, width=16)
    target = AutonomousTrainerV7(config(), 307, {}, width=16)
    for tr in [source, target]:
        sum(p.square().mean() for p in tr.model.parameters() if p.requires_grad).backward()
        tr.optimizer.step()
        tr.optimizer.zero_grad(set_to_none=True)
    source.model.fit_value_scale(torch.tensor([2.0, 7.0, 90.0]))
    target.model.fit_value_scale(torch.tensor([4.0, 25.0, 180.0]))
    source_state = copy.deepcopy(source.model.state_dict())
    before = protected_actor(target)
    x = batch()
    action, lp, _, _ = target.model(x, generator=torch.Generator().manual_seed(4))
    report = transfer_critic(target, source, [x])
    assert protected_actor(target) == before
    assert torch.equal(target.model(x, action)[1], lp)
    assert torch.allclose(target.model.value(x), source.model.value(x), atol=1e-4)
    assert all(torch.equal(v, source.model.state_dict()[k]) for k, v in source_state.items())
    assert not any(p.requires_grad for p in target.model.reference_critic.parameters())
    assert not any(p.requires_grad for p in target.model.initial_normalized_critic.parameters())
    assert all(p not in target.optimizer.state for p in target.model.critic.parameters())
    assert all(p in target.optimizer.state for p in target.model.actor.parameters())
    assert report["anchors_replaced_and_refrozen"]
