"""PPO scaffolding for the model9 learner (M7); no native bridge required."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts, data, ppo
from anthropocene_rl.autonomous_v7.model import Actor, Normalizer, to_tensors
from test_autonomous_v7 import batch


def setup(fishing: bool = True):
    b = batch()
    b["economy_mix_mask"][:, 2] = fishing
    normalizer = Normalizer.fit(b)
    torch.manual_seed(0)
    actor = Actor(b["core"].shape[1])
    t = to_tensors(b, normalizer)
    return actor, t


@pytest.mark.parametrize("stage", sorted(ppo.STAGES))
def test_sampled_draws_recompute_their_log_probability(stage: str) -> None:
    actor, t = setup()
    s = ppo.STAGES[stage]
    with torch.no_grad():
        out = actor(t)
        draw = ppo.sample(out, t, s, torch.Generator().manual_seed(3))
        lp1, entropy = ppo.log_prob(out, t, draw, s)
        lp2, _ = ppo.log_prob(actor(t), t, draw, s)
    assert torch.allclose(lp1, lp2) and torch.all(torch.isfinite(lp1)) and torch.all(torch.isfinite(entropy))
    actions = ppo.to_bridge_actions(draw, t)
    assert all(len(a["strategic"]) == len(contracts.STRATEGIC) for a in actions)
    assert all((a["movement"] == contracts.NATIVE_MOVEMENT) != s.movement for a in actions)
    assert all((a["clothingRoute"] == -1) != s.routes for a in actions)


def test_masked_economy_coordinates_are_never_chosen() -> None:
    actor, t = setup(fishing=False)
    with torch.no_grad():
        draw = ppo.sample(actor(t), t, ppo.STAGES["capability"], torch.Generator().manual_seed(1))
    actions = ppo.to_bridge_actions(draw, t)
    assert all(a["strategic"][4] == 0.0 for a in actions)
    mix = ppo.mix_from_ratios(draw["mix_z"], t["economy_mix_mask"])
    assert torch.allclose(mix.sum(dim=1), torch.ones(len(mix)))


def test_teacher_kl_vanishes_for_the_teacher_itself() -> None:
    actor, t = setup()
    with torch.no_grad():
        out = actor(t)
    assert torch.allclose(ppo.teacher_kl(out, out), torch.zeros(len(out["strategic_mean"])), atol=1e-6)


def test_gae_matches_a_hand_computation() -> None:
    d = {"world": np.array([1, 1]), "camp": np.array([7, 7]), "turn": np.array([1, 2]),
         "years": np.full(2, 20.0), "terminated": np.array([False, False])}
    reward = np.array([10.0, 12.0]); value = np.array([100.0, 110.0]); bootstrap = np.array([0.0, 120.0])
    adv, target = ppo.gae(d, reward, value, bootstrap, lam=0.5)
    gamma = float(data.discount(20.0))
    d1 = 12 + gamma * 120 - 110
    d0 = 10 + gamma * 110 - 100
    assert adv[1] == pytest.approx(d1) and adv[0] == pytest.approx(d0 + gamma * 0.5 * d1)
    assert np.allclose(target, adv + value)
