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


def test_route_anchor_vanishes_for_the_teacher_and_is_positive_otherwise() -> None:
    actor, t = setup()
    with torch.no_grad():
        out = actor(t)
        assert torch.allclose(ppo.teacher_kl(out, out, True, routes=True), torch.zeros(len(out["clothing"])), atol=1e-5)
        shifted = dict(out); shifted["clothing"] = out["clothing"] + torch.tensor([3.0, 0.0])
        assert (ppo.teacher_kl(shifted, out, True, routes=True) >= -1e-6).all()
        assert ppo.teacher_kl(shifted, out, True, routes=True).sum() > ppo.teacher_kl(shifted, out, True).sum()


def test_contract8_checkpoints_upgrade_exactly() -> None:
    from anthropocene_rl.autonomous_v7.upgrade import upgrade
    from anthropocene_rl.autonomous_v7.model import Critic
    b = batch()
    old_width = b["core"].shape[1]
    torch.manual_seed(1)
    actor, critic = Actor(old_width), Critic(old_width)
    normalizer = Normalizer.fit(b)
    state = {"actor": {k: v for k, v in actor.state_dict().items() if not k.startswith("span.")}, "critic": critic.state_dict(),
             "normalizer": normalizer.state_dict(), "core_width": old_width, "contract": 8}
    up = upgrade(state)
    assert up["core_width"] == old_width + contracts.COMMITMENT_FEATURES and up["contract"] == contracts.NATIVE_CONTRACT
    new_actor, new_critic = Actor(up["core_width"]), Critic(up["core_width"])
    new_actor.load_state_dict(up["actor"]); new_critic.load_state_dict(up["critic"])
    b9 = dict(b)
    rng = np.random.default_rng(0)
    b9["core"] = np.concatenate([b["core"], rng.normal(size=(len(b["core"]), contracts.COMMITMENT_FEATURES)).astype(np.float32)], axis=1)
    b9["core_present"] = np.concatenate([b["core_present"], np.ones((len(b["core"]), contracts.COMMITMENT_FEATURES), bool)], axis=1)
    with torch.no_grad():
        a, a9 = actor(to_tensors(b, normalizer)), new_actor(to_tensors(b9, Normalizer.from_state(up["normalizer"])))
        c, c9 = critic(to_tensors(b, normalizer)), new_critic(to_tensors(b9, Normalizer.from_state(up["normalizer"])))
    for key in ("strategic_mean", "service_mean", "mix_logits", "clothing", "movement"):
        assert torch.allclose(a[key], a9[key], atol=1e-6)
    assert torch.allclose(c, c9, atol=1e-6)
    assert (a9["span"].argmax(dim=1) == contracts.SPAN_SINGLE).all()


def test_contract9_checkpoints_upgrade_exactly_to_10() -> None:
    from anthropocene_rl.autonomous_v7.upgrade import upgrade
    from anthropocene_rl.autonomous_v7.model import Critic
    b = batch(); normalizer = Normalizer.fit(b)
    torch.manual_seed(4)
    actor, critic = Actor(b["core"].shape[1]), Critic(b["core"].shape[1])
    pooled = 3 * actor.encoder.set_hidden + 1
    def as_contract9(net: dict) -> dict:
        net = {k: v for k, v in net.items() if ".sets.regions." not in k and not k.startswith(("region_", "mix_region"))}
        net["encoder.trunk.0.weight"] = net["encoder.trunk.0.weight"][:, :-pooled].clone()
        return net
    norm9 = {k: v for k, v in normalizer.state_dict().items() if not k.startswith("regions")}
    state9 = {"actor": as_contract9(actor.state_dict()), "critic": as_contract9(critic.state_dict()), "normalizer": norm9,
              "core_width": b["core"].shape[1], "contract": 9}
    up = upgrade(state9)
    assert up["contract"] == contracts.NATIVE_CONTRACT and up["upgraded_from"] == 9
    new_actor, new_critic = Actor(up["core_width"]), Critic(up["core_width"])
    new_actor.load_state_dict(up["actor"]); new_critic.load_state_dict(up["critic"])
    for key, value in state9["actor"].items():
        if key != "encoder.trunk.0.weight":
            assert torch.equal(up["actor"][key], value), key
    w = up["actor"]["encoder.trunk.0.weight"]
    assert torch.equal(w[:, :-pooled], state9["actor"]["encoder.trunk.0.weight"]) and torch.all(w[:, -pooled:] == 0)
    # Region rows, present or absent, change nothing an upgraded policy outputs; its region argmax is native.
    t = to_tensors(b, Normalizer.from_state(up["normalizer"]))
    absent = dict(t); absent["regions_present"] = torch.zeros_like(t["regions_present"])
    with torch.no_grad():
        a, a0 = new_actor(t), new_actor(absent)
        assert torch.allclose(new_critic(t), new_critic(absent))
    for key in ("strategic_mean", "service_mean", "mix_logits", "clothing", "movement", "span", "region"):
        assert torch.allclose(a[key], a0[key]), key
    assert torch.all(a["mix_offset"] == 0) and torch.all(a["region"].argmax(dim=1) == contracts.REGION_NATIVE)


def test_region_head_starts_native_and_respects_masks() -> None:
    from anthropocene_rl.autonomous_v7.model import NATIVE_START, region_code
    actor, t = setup()
    t = dict(t); mask = torch.ones_like(t["region_movement_mask"]); mask[:, contracts.REGION_KEEP] = False; mask[:, 2 + 5:] = False
    t["region_movement_mask"] = mask
    with torch.no_grad():
        p = torch.softmax(actor(t)["region"], dim=1)
    assert torch.all(p[:, contracts.REGION_KEEP] < 1e-9) and torch.all(p[:, 2 + 5:] < 1e-9)
    t_all = dict(t); t_all["region_movement_mask"] = torch.ones_like(mask)
    with torch.no_grad():
        p_all = torch.softmax(actor(t_all)["region"], dim=1)
    assert torch.allclose(p_all[:, contracts.REGION_NATIVE], torch.full((len(p_all),), NATIVE_START), atol=1e-5)
    assert [region_code(s) for s in (0, 1, 2, 9)] == [contracts.NATIVE_MOVEMENT, contracts.KEEP_INTENT,
                                                      contracts.REGION_BASE, contracts.REGION_BASE + 7]


def test_learned_movement_draws_recompute_their_log_probability_with_the_conditioned_mix() -> None:
    import dataclasses
    actor, t = setup()
    with torch.no_grad():
        actor.mix_region.weight.normal_(0, 1.0); actor.region_query.weight.normal_(0, 1.0)  # make conditioning matter
    stage = dataclasses.replace(ppo.STAGES["mixed"], movement=True, region_movement=True)
    with torch.no_grad():
        out = actor(t)
        draw = ppo.sample(out, t, stage, torch.Generator().manual_seed(9))
        lp1, entropy = ppo.log_prob(out, t, draw, stage)
        lp2, _ = ppo.log_prob(actor(t), t, draw, stage)
        # The mix density depends on the drawn slot: scoring the same draw under another slot changes it.
        other = dict(draw); other["region"] = (draw["region"] + 1) % contracts.REGION_MOVEMENT_SLOTS
        lp3, _ = ppo.log_prob(out, t, other, stage)
    assert torch.allclose(lp1, lp2) and torch.all(torch.isfinite(lp1)) and torch.all(torch.isfinite(entropy))
    assert not torch.allclose(lp1, lp3)
    codes = {a["movement"] for a in ppo.to_bridge_actions(draw, t)}
    allowed = {contracts.NATIVE_MOVEMENT, contracts.KEEP_INTENT} | {contracts.REGION_BASE + r for r in range(contracts.REGION_ROWS)}
    assert codes <= allowed


def test_learned_fission_is_sampled_scored_and_anchored() -> None:
    import dataclasses
    actor, t = setup()
    stage = dataclasses.replace(ppo.STAGES["mixed"], fission=True)
    with torch.no_grad():
        out = actor(t)
        draw = ppo.sample(out, t, stage, torch.Generator().manual_seed(4))
        lp1, _ = ppo.log_prob(out, t, draw, stage)
        other = dict(draw); other["fission"] = (draw["fission"] + 1) % contracts.FISSION_POLICIES
        lp2, _ = ppo.log_prob(out, t, other, stage)
        shifted = dict(out); shifted["fission"] = out["fission"] + torch.tensor([3.0, 0.0, 0.0])
        assert torch.allclose(ppo.teacher_kl(out, out, fission=True), ppo.teacher_kl(out, out), atol=1e-6)
        assert (ppo.teacher_kl(shifted, out, fission=True) > ppo.teacher_kl(shifted, out)).all()
    assert not torch.allclose(lp1, lp2)
    assert all(a["fission"] in (0, 1, 2) for a in ppo.to_bridge_actions(draw, t))
    native = ppo.sample(out, t, ppo.STAGES["mixed"], torch.Generator().manual_seed(4))
    assert all(a["fission"] == -1 for a in ppo.to_bridge_actions(native, t))


def test_observed_movement_control_stays_native() -> None:
    import dataclasses
    actor, t = setup()
    t = dict(t); t["region_movement_mask"] = torch.ones_like(t["region_movement_mask"])
    stage = dataclasses.replace(ppo.STAGES["mixed"], observe_movement=True)
    assert stage.bridge_movement and not stage.movement
    with torch.no_grad():
        draw = ppo.sample(actor(t), t, stage, torch.Generator().manual_seed(2))
    assert all(a["movement"] == contracts.NATIVE_MOVEMENT for a in ppo.to_bridge_actions(draw, t))


def test_evaluation_honours_the_training_commitments_setting(tmp_path, monkeypatch) -> None:
    from anthropocene_rl.autonomous_v7 import evaluate
    actor, t = setup()
    with torch.no_grad():
        actor.span.bias.zero_(); actor.span.bias[contracts.SPANS.index(6)] = 50.0   # an inherited head that commits
    monkeypatch.setattr(evaluate, "observation_batch", lambda live: None)
    monkeypatch.setattr(evaluate, "to_tensors", lambda b, normalizer: t)
    villages = [{"camp": i, "observation": {}} for i in range(len(t["core"]))]
    assert all(a["commit"] == 6 for a in evaluate.policy_actions(actor, None, villages).values())
    actor.commitments = False
    assert all(a["commit"] == 1 for a in evaluate.policy_actions(actor, None, villages).values())
    # The flag round-trips through checkpoints; checkpoints saved before it keep the span argmax.
    normalizer = Normalizer.fit(batch())
    for saved, expected in (({"commitments": False}, False), ({}, True)):
        path = tmp_path / f"c{len(saved)}.pt"
        torch.save({"actor": actor.state_dict(), "normalizer": normalizer.state_dict(), "core_width": t["core"].shape[1] // 2,
                    "contract": contracts.NATIVE_CONTRACT, **saved}, path)
        assert evaluate.load(path)[0].commitments is expected


def test_paired_compare_and_seed_lists() -> None:
    from anthropocene_rl.autonomous_v7 import paired_compare
    from anthropocene_rl.autonomous_v7.stage_eval import seed_list
    assert seed_list("2000-2002,2026,2030-2031") == [2000, 2001, 2002, 2026, 2030, 2031]
    rng = np.random.default_rng(0)
    difficulty = rng.uniform(1.0, 100.0, 40)                  # shared world difficulty must cancel in the pairing
    def row(scale: float, seeds: int) -> dict:
        return {f"{scale}-{s}": {"worlds": [[i, 0, float(d * scale * (1 + 0.01 * rng.normal())), 0] for i, d in enumerate(difficulty)],
                                 "ratio_vs_teacher": scale, "extinctions": 0} for s in range(seeds)}
    candidates = {**row(1.0, 3), **row(1.05, 3)}
    report = {"default": {"candidates": candidates}, "gate": {}}
    result = paired_compare.compare(report, [k for k in candidates if k.startswith("1.0-")],
                                    [k for k in candidates if k.startswith("1.05-")], draws=500)
    r = result["default"]["b_vs_a"]
    assert abs(r["ratio"] - 1.05) < 0.01 and r["lo"] < r["ratio"] < r["hi"] and r["lo"] > 1.03
    # Shards over the same worlds merge into one report.
    teacher = {"worlds": [[i, 0, 1.0, 0] for i in range(len(difficulty))]}
    names = sorted(candidates)
    shards = [{"default": {"teacher": teacher, "candidates": {k: candidates[k] for k in names[:3]}}, "gate": {}},
              {"default": {"teacher": teacher, "candidates": {k: candidates[k] for k in names[3:]}}, "gate": {}}]
    assert set(paired_compare.merge(shards)["default"]["candidates"]) == set(candidates)


def test_hold_rows_carry_no_capital_probability() -> None:
    import dataclasses
    actor, t = setup()
    t = dict(t); t["commit_hold_mask"] = torch.ones(len(t["core"]), dtype=torch.bool)
    stage = dataclasses.replace(ppo.STAGES["mixed"], commitments=True)
    with torch.no_grad():
        out = ppo.explore(actor(t), -2.0)
        draw = ppo.sample(out, t, stage, torch.Generator().manual_seed(5))
        draw["span"] = torch.zeros_like(draw["span"])           # every row holds
        lp_hold, _ = ppo.log_prob(out, t, draw, stage)
        moved = dict(draw); moved["strategic_z"] = draw["strategic_z"] + 3.0; moved["clothing"] = 1 - draw["clothing"].clamp(min=0)
        lp_moved, _ = ppo.log_prob(out, t, moved, stage)
    assert torch.allclose(lp_hold, lp_moved), "held capital draws must not change the log-probability"
    assert all(a["commit"] == 0 for a in ppo.to_bridge_actions(draw, t))


def test_vector_critic_with_default_weights_reproduces_the_scalar_critic() -> None:
    from anthropocene_rl.autonomous_v7.model import Critic, VectorCritic
    b = batch()
    normalizer = Normalizer.fit(b)
    torch.manual_seed(2)
    scalar = Critic(b["core"].shape[1])
    vector = VectorCritic.from_scalar(scalar, b["core"].shape[1])
    t = to_tensors(b, normalizer)
    with torch.no_grad():
        assert torch.allclose(scalar(t), vector(t)["welfare"])
    # The weighted advantage with the default weights is the welfare advantage itself.
    advantages = {"welfare": np.array([1.0, -2.0, 3.0]), "population_years": np.array([5.0, 5.0, 5.0])}
    assert np.allclose(ppo.combine(advantages, contracts.OBJECTIVE_WEIGHTS), advantages["welfare"])
    assert np.allclose(ppo.combine(advantages, {"welfare": 1.0, "population_years": 0.5}), advantages["welfare"] + 2.5)


def test_reward_components_split_welfare_from_population_years() -> None:
    d = {"world": np.array([1, 1, 1]), "camp": np.array([1, 2, 1]), "turn": np.array([0, 0, 1]), "society": np.array([5, 6, 5]),
         "parent_society": np.array([0, 0, 0]), "prior_population": np.array([100, 50, 110]), "qol": np.array([.8, .5, .9]),
         "years": np.full(3, 20.0), "terminated": np.array([False, False, False])}
    parts = data.reward_components(d)
    assert np.allclose(parts["welfare"], data.lineage_rewards(d))
    assert np.allclose(parts["population_years"], [100 * 20, 50 * 20, 110 * 20])


def test_span_labels_follow_the_heuristics_own_persistence() -> None:
    from anthropocene_rl.autonomous_v7 import span_labels
    n = 7
    strategic = np.tile(np.array([1.0, 1.0, 0.2, 0.5, 0.0], np.float32), (n, 1))
    strategic[4:, 2] = 0.6                     # from turn 4 the Development budget changes
    b = {"world": np.zeros(n, int), "camp": np.full(n, 3), "turn": np.arange(n),
         "strategic": strategic, "strategic_mask": np.ones((n, 5), bool),
         "clothing_route": np.zeros(n), "shelter_route": np.zeros(n),
         "clothing_route_label_mask": np.ones(n, bool), "shelter_route_label_mask": np.ones(n, bool)}
    d = span_labels.derive(b, 0.05)
    spans = [contracts.SPANS[i] for i in d["span_label"]]
    # Turns 0-3 repeat: a 4-turn commitment then three holds; turns 4-6 repeat the new budget: a 3-turn commitment.
    assert spans == [4, 0, 0, 0, 3, 0, 0]
    assert d["commit_hold_mask"].tolist() == [False, True, True, True, False, True, True]
    assert d["commitment_features"][1, :3].tolist() == [1.0, 3.0, 1.0]
    assert abs(d["commitment_features"][5, 5] - 0.6) < 1e-6
