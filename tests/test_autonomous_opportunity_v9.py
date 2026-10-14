import copy

import pytest
import torch
from test_autonomous_direct_control_v6 import v6_config, v6_wire

from anthropocene_rl.autonomous_v6.contracts import observation
from anthropocene_rl.autonomous_v6.opportunity import (
    TRADE_HEADS,
    market_peer,
    own_address,
    protected_warmup_state,
    trade_target,
    warmup_trade,
)
from anthropocene_rl.autonomous_v6.policy import inputs_with_mode
from anthropocene_rl.training.autonomous_ppo_v6 import AutonomousTrainerV6


class Addresses:
    ordinal = 1
    requests = None

    def _request(self, request):
        self.requests = request
        return dict(metadataVersion="own-known-address-v1", ordinal=self.ordinal)


def test_fixed_market_targets_own_known_customer_without_fallback_or_foreign_stock():
    tr = Addresses()
    wire = v6_wire()
    wire["CounterpartyMask"] = [True, True, False, False, False, False, False, False]
    wire["Stocks"] = [100] * 11
    obs = observation(wire)
    quote = market_peer(obs, 9, 8, tr, {"market_focal_give": 6})["quote"]
    assert quote == dict(Counterparty=1, Give=3, Want=6, GiveQuantity=10, WantQuantity=10)
    assert tr.requests == dict(op="autonomous6_quote_address", observer=9, target=8)
    tr.ordinal = -1
    assert "quote" not in market_peer(obs, 9, 8, tr, {"market_focal_give": 6})
    tr.ordinal = 8
    with pytest.raises(ValueError):
        own_address(tr, 9, 8)
    tr.ordinal = True
    with pytest.raises(ValueError):
        own_address(tr, 9, 8)
    tr.ordinal = 1
    assert trade_target(obs, 1, 6) == [1, 1, 6, 3, 1, 1]
    obs["Stocks"][6] = 0
    assert trade_target(obs, 1, 6) == [0] * 6
    assert trade_target(obs, -1, 6) == [0] * 6


def test_head_only_warmup_preserves_populated_ppo_adam_rng_counters_and_other_heads():
    torch.set_num_threads(1)
    trainer = AutonomousTrainerV6(v6_config(), 59, {"test": "warmup"}, width=16)
    # Populated optimizer protection is stronger than preserving an empty Adam.
    sum(p.square().mean() for p in trainer.model.parameters() if p.requires_grad).backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.episodes, trainer.transitions, trainer.updates = 3, 80, 2
    before = protected_warmup_state(trainer)
    state = copy.deepcopy(trainer.model.state_dict())
    observations = []
    for i in range(12):
        wire = v6_wire()
        wire["CounterpartyMask"] = [True] + [False] * 7
        wire["Stocks"] = [10] * 11
        wire["Stocks"][3 if i % 2 else 6] = 10000
        observations.append(observation(wire))
    x = inputs_with_mode(observations)
    targets = torch.tensor(
        [trade_target(o, 0, 3 if i % 2 else 6) for i, o in enumerate(observations)]
    )
    report = warmup_trade(trainer, x, targets, seed=100, steps=4)
    assert protected_warmup_state(trainer) == before
    assert report["joint_logp_fixed_nontrade_after"] > report["joint_logp_fixed_nontrade_before"]
    assert report["trade_head_step_l2"] > 0
    assert report["native_resolves"] == report["retained_ppo_transitions"] == 0
    assert all(
        torch.equal(value, trainer.model.state_dict()[name])
        for name, value in state.items()
        if not any(name.startswith(f"heads.{h}.") for h in TRADE_HEADS)
    )
    assert all(p.grad is None for p in trainer.model.parameters())


def test_warmup_rejects_illegal_conditional_target_instead_of_repairing_it():
    trainer = AutonomousTrainerV6(v6_config(), 59, {}, width=16)
    obs = observation(v6_wire())
    obs["Stocks"][6] = 0
    x = inputs_with_mode([obs])
    before = protected_warmup_state(trainer)
    with pytest.raises(ValueError):
        warmup_trade(trainer, x, torch.tensor([[1, 0, 6, 3, 1, 1]]), seed=91, steps=4)
    assert protected_warmup_state(trainer) == before
