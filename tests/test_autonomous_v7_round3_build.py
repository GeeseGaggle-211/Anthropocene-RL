"""Round-3 build (docs/MODEL9_M7.md, *Round 3 build: heads, diagnostics, provenance*); no native bridge required."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from test_autonomous_v7 import batch

from anthropocene_rl.autonomous_v7 import contracts, model, ppo
from anthropocene_rl.autonomous_v7.model import Actor, Normalizer, to_tensors


def setup(region_head: str | None = None, all_legal: bool = True, token: bool = False):
    b = batch()
    if all_legal:
        b["region_movement_mask"] = np.ones((len(b["core"]), contracts.REGION_MOVEMENT_SLOTS), bool)
    normalizer = Normalizer.fit(b)
    torch.manual_seed(0)
    actor = Actor(b["core"].shape[1], region_head=region_head, token=token)
    with torch.no_grad():  # make destinations differ, so groups and slots are not all tied
        actor.region_query.weight.normal_(0, 1.0)
        actor.region_native.weight.normal_(0, 0.3)
    return actor, to_tensors(b, normalizer)


# ------------------------------------------------------------------------------------------- 1. hierarchical head
def test_hierarchical_head_is_the_default_and_equals_the_flat_distribution_at_a_zero_offset() -> (
    None
):
    actor, t = setup()
    assert actor.region_head == "hierarchical"
    with torch.no_grad():
        out = actor(t)
    assert "region_group" in out
    flat = torch.log_softmax(out["region"], dim=1)
    assert torch.allclose(out["region_logp"], flat, atol=1e-5)
    # Group masses are the flat masses per group.
    p = flat.exp()
    index = model.region_group_index()
    masses = torch.stack([p[:, index == g].sum(dim=1) for g in range(3)], dim=1)
    assert torch.allclose(torch.softmax(out["region_group"], dim=1), masses, atol=1e-5)


def test_the_group_offset_moves_group_mass_but_not_the_odds_within_a_group() -> None:
    actor, t = setup()
    with torch.no_grad():
        before = actor(t)
        actor.region_group.bias[model.REGION_GROUP_GO] = 2.0
        after = actor(t)
    go = model.region_group_index() == model.REGION_GROUP_GO
    pb, pa = before["region_logp"].exp(), after["region_logp"].exp()
    assert torch.all(pa[:, go].sum(dim=1) > pb[:, go].sum(dim=1))
    within_b = pb[:, go] / pb[:, go].sum(dim=1, keepdim=True)
    within_a = pa[:, go] / pa[:, go].sum(dim=1, keepdim=True)
    assert torch.allclose(within_a, within_b, atol=1e-5)
    assert torch.allclose(pa.sum(dim=1), torch.ones(len(pa)), atol=1e-5)


def test_hierarchical_mode_chooses_the_group_first() -> None:
    # native 0.30, keep 0, stay 0.05, rows 1-7 share 0.65: a flat argmax picks native, the head moves to the best row.
    p = torch.tensor([[0.30, 1e-9, 0.05, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02]])
    region = torch.log(p)
    legal = torch.ones_like(region, dtype=torch.bool)
    group, joint = model.hierarchical_region(region, legal, torch.zeros(1, 3))
    out = {"region": region, "region_group": group, "region_logp": joint}
    assert int(model.region_mode(out)) == 3 and int(region.argmax()) == 0
    assert int(model.region_mode({"region": region})) == 0  # the flat head: plain argmax
    assert (
        int(model.region_mode({"region": region}, grouped=True)) == 3
    )  # the flat head's evaluation option


def test_flat_flag_reproduces_the_earlier_head_and_is_saved() -> None:
    actor, t = setup(region_head="flat")
    with torch.no_grad():
        out = actor(t)
    assert actor.region_head == "flat" and "region_group" not in out
    assert torch.allclose(out["region_logp"], torch.log_softmax(out["region"], dim=1))
    mode = model.act_deterministic(out, region=True)
    assert np.array_equal(mode["region"], out["region"].argmax(dim=1).numpy())
    restored = Actor(t["core"].shape[1] // 2)
    restored.load_state_dict(actor.state_dict())
    assert restored.region_head == "flat"


def test_checkpoints_saved_before_the_head_load_hierarchical_or_flat_by_environment(
    monkeypatch,
) -> None:
    actor, t = setup()
    state = {
        k: v
        for k, v in actor.state_dict().items()
        if not k.startswith("region_group") and k != "region_flat"
    }
    loaded = Actor(t["core"].shape[1] // 2)
    loaded.load_state_dict(state)  # strict: the pre-hook fills the head
    assert loaded.region_head == "hierarchical"
    assert torch.count_nonzero(loaded.region_group.weight) == 0
    with torch.no_grad():
        assert torch.allclose(loaded(t)["region_logp"], actor(t)["region_logp"], atol=1e-6)
    monkeypatch.setenv("ANTHROPOCENE_REGION_HEAD", "flat")
    old = Actor(t["core"].shape[1] // 2)
    old.load_state_dict(state)
    assert old.region_head == "flat"


def test_masked_slots_and_groups_carry_no_probability() -> None:
    actor, t = setup(all_legal=False)  # batch(): only native is legal
    with torch.no_grad():
        out = actor(t)
    p = out["region_logp"].exp()
    assert torch.allclose(p[:, contracts.REGION_NATIVE], torch.ones(len(p)))
    assert torch.all(out["region_group"][:, 1:] == model.MASKED)
    assert torch.all(torch.isfinite(out["region_logp"]))


@pytest.mark.parametrize("head", ["hierarchical", "flat"])
@pytest.mark.parametrize("token", [False, True])
def test_ppo_samples_and_scores_the_heads_joint_distribution(head: str, token: bool) -> None:
    actor, t = setup(region_head=head, token=token)
    stage = dataclasses.replace(ppo.STAGES["mixed"], movement=True, region_movement=True)
    with torch.no_grad():
        actor.region_group.bias[model.REGION_GROUP_GO] = 1.5  # inert on the flat head
        out = actor(t)
        draw = ppo.sample(out, t, stage, torch.Generator().manual_seed(4))
        lp, entropy = ppo.log_prob(out, t, draw, stage)
        lp0, _ = ppo.log_prob(out, t, draw, ppo.STAGES["mixed"])  # the same draw, region not scored
        expected = out["region_logp"].gather(1, draw["region"].unsqueeze(1)).squeeze(1)
        p = out["region_logp"].exp()
    assert torch.all(torch.isfinite(lp)) and torch.all(torch.isfinite(entropy))
    # The mix offset is zero-initialised, so the region part is exactly the head's joint log-probability of the draw.
    assert torch.allclose(lp - lp0, expected, atol=1e-4)
    if head == "hierarchical":  # the offset raised the go group's share above the flat one
        go = model.region_group_index() == model.REGION_GROUP_GO
        flat = torch.softmax(out["region"], dim=1)
        gain = p[:, go].sum(dim=1) - flat[:, go].sum(dim=1)
        assert torch.all(gain > -1e-6) and torch.any(gain > 1e-3)


def test_search_labels_supervise_both_levels() -> None:
    from anthropocene_rl.autonomous_v7.search_teacher import NATIVE, REGION

    actor, t = setup()
    out = actor(t)
    n = len(out["region"])
    kind = torch.full((n,), REGION, dtype=torch.long)
    kind[0] = NATIVE
    slot = torch.full((n,), 4, dtype=torch.long)
    slot[0] = contracts.REGION_NATIVE
    loss = ppo.search_loss(out, t, kind, slot, torch.full((n, 3), float("nan")))
    loss.backward()
    assert (
        actor.region_group.weight.grad is not None
        and actor.region_group.weight.grad.abs().sum() > 0
    )
    assert actor.region_query.weight.grad.abs().sum() > 0
    # The loss is the joint cross-entropy: -log p(group) - log p(slot | group).
    with torch.no_grad():
        o = actor(t)
        expected = -o["region_logp"].gather(1, slot.unsqueeze(1)).squeeze(1).mean()
        assert torch.allclose(
            ppo.search_loss(o, t, kind, slot, torch.full((n, 3), float("nan"))), expected, atol=1e-5
        )


# ------------------------------------------------------------------------------------------- 3. cost bootstrap
def test_guard_cost_critic_bootstraps_truncated_living_villages() -> None:
    b = batch()
    normalizer = Normalizer.fit(b)
    torch.manual_seed(0)
    cost_critic = ppo.CostCritic(b["core"].shape[1])
    with torch.no_grad():
        cost_critic.head.bias.fill_(0.5)
        cost_critic.head.weight.zero_()  # V_cost = 0.5 everywhere
    roll = ppo.Rollout()
    roll.bootstrap_obs.append({k: v[:2] for k, v in b.items()})
    # Two villages, two turns each; village 7 is truncated alive at the horizon, village 8 terminated.
    rows = {
        "world": np.array([0, 0, 0, 0]),
        "camp": np.array([7, 8, 7, 8]),
        "turn": np.array([1, 1, 2, 2]),
        "years": np.full(4, 20.0),
        "terminated": np.array([False, False, False, True]),
        "cost": np.array([1.0, 1.0, 1.0, 1.0]),
        "bootstrap_index": np.array([-1, -1, 0, -1]),
    }
    truncated = np.flatnonzero(rows["bootstrap_index"] >= 0)
    boot = ppo.cost_bootstrap(cost_critic, roll, rows, normalizer, truncated)
    assert np.allclose(boot, [0, 0, 0.5, 0])
    value = np.full(4, 0.5)
    _, with_boot = ppo.gae(rows, rows["cost"], value, boot)
    _, without = ppo.gae(rows, rows["cost"], value, np.zeros(4))
    gamma = float(ppo.data.discount(np.asarray([20.0]))[0])
    assert with_boot[2] == pytest.approx(1.0 + gamma * 0.5) and without[2] == pytest.approx(1.0)
    assert with_boot[3] == pytest.approx(without[3])  # a terminated village is not bootstrapped
    assert ppo.Hyper().cost_bootstrap
