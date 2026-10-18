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


def test_regime_balance_gives_each_regime_equal_weight() -> None:
    rng = np.random.default_rng(0)
    regimes = np.array(["default"] * 90 + ["fishing"] * 10)
    combined = np.concatenate([rng.normal(5, 2, 90), rng.normal(-3, 1, 10)])
    out, weight = ppo.balance(combined, regimes, cap=20.0)
    for r in ("default", "fishing"):
        rows = regimes == r
        assert abs(out[rows].mean()) < 1e-9 and abs(out[rows].std() - 1) < 1e-9
    assert np.isclose(weight[regimes == "default"].sum(), weight[regimes == "fishing"].sum())
    assert np.isclose(weight.mean(), 1.0)
    capped = ppo.balance(combined, regimes, cap=4.0)[1]
    assert capped[regimes == "fishing"][0] / capped[regimes == "default"][0] <= 16 + 1e-9
    plain, unit = ppo.balance(combined, None, cap=4.0)
    assert np.allclose(unit, 1) and abs(plain.mean()) < 1e-9


def test_regimes_follow_the_gate_categories() -> None:
    grass = frozenset({4005})
    assert ppo.regime(1000, 0, grass) == "default" and ppo.regime(4005, 0, grass) == "grassland"
    for name, mask in contracts.CAPABILITY_MASKS.items():
        assert ppo.regime(1000, mask, grass) == name


def test_mix_change_is_total_variation_within_a_village() -> None:
    batch_ = {"world": np.array([0, 0, 0, 1]), "camp": np.array([5, 5, 5, 5]), "turn": np.array([0, 1, 2, 0]),
              "economy_mix_mask": np.ones((4, 3), bool)}
    z = torch.tensor([[0.0, 0.0, -50.0], [0.0, 0.0, -50.0], [50.0, 0.0, 0.0], [0.0, 50.0, 0.0]])
    change = ppo.mix_changes(batch_, {"mix_z": z})
    assert change[0] == 0 and change[3] == 0 and abs(change[1]) < 1e-6
    assert abs(change[2] - 0.5) < 1e-4


def test_expert_adapters_warm_start_identically_and_isolate_regimes() -> None:
    actor, t = setup()
    experts = Actor(actor.encoder.core[0].in_features // 2, experts=True)
    missing, unexpected = experts.load_state_dict(actor.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("experts.") for k in missing)
    with torch.no_grad():
        a, b = actor(t), experts(t)
    for key in ("strategic_mean", "mix_logits", "service_mean"):
        assert torch.allclose(a[key], b[key])
    # A fishing-only village's loss reaches only the fishing adapter.
    for adapter in experts.experts.adapters:
        torch.nn.init.normal_(adapter[2].weight, std=0.01)
    mask = torch.zeros_like(t["economy_mix_mask"]); mask[:, 2] = True
    t2 = dict(t); t2["economy_mix_mask"] = mask
    experts.zero_grad()
    experts(t2)["strategic_mean"].sum().backward()
    grads = [adapter[0].weight.grad for adapter in experts.experts.adapters]
    assert grads[0] is None or grads[0].abs().sum() == 0
    assert grads[1] is None or grads[1].abs().sum() == 0
    assert grads[2] is not None and grads[2].abs().sum() > 0
