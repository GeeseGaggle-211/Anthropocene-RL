"""Round 3 (docs/MODEL9_M7.md): stable personality scales, the daughter-start bank and specialist-share reporting.
Bridge tests are skipped unless the model9 bridge is built."""

from __future__ import annotations

import concurrent.futures
import os

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts, daughter_eval, evaluate, ppo, search_teacher
from anthropocene_rl.autonomous_v7.bridge import DEFAULT_BRIDGE, Model9Bridge
from anthropocene_rl.autonomous_v7.daughter_bank import DaughterBank, held_out

needs_bridge = pytest.mark.skipif(
    not os.path.exists(os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE", str(DEFAULT_BRIDGE))), reason="model9 bridge not built")


def advantages(n: int = 200) -> dict[str, np.ndarray]:
    return {name: np.random.default_rng(k).normal(loc=k, scale=10.0 ** k, size=n) for k, name in enumerate(contracts.OBJECTIVES)}


def update_input(advantage: np.ndarray, cost: np.ndarray, regimes: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """The advantage ``ppo.update`` trains on (its first two lines)."""
    combined = advantage / max(1e-8, advantage.std()) - 0.7 * cost / max(1e-8, cost.std() + 1e-8)
    return ppo.balance(combined, regimes, 4.0)


# --------------------------------------------------------------------------------------------- stable scales
def test_reference_scales_are_measured_and_positive() -> None:
    assert set(contracts.OBJECTIVE_REFERENCE_SCALES) == set(contracts.OBJECTIVES)
    assert all(s > 0 for s in contracts.OBJECTIVE_REFERENCE_SCALES.values())


@pytest.mark.parametrize("regimes", [None, np.array(["default", "herding", "fishing", "farming"] * 50)])
def test_reference_combine_reproduces_todays_update_at_the_welfare_personality(regimes) -> None:
    adv = advantages(); cost = np.random.default_rng(7).normal(size=200)
    welfare = np.tile(ppo.default_personality(), (200, 1))
    today, w_today = update_input(ppo.combine_personal(adv, welfare, "batch"), cost, regimes)
    stable, w_stable = update_input(ppo.combine_personal(adv, welfare, "reference"), cost, regimes)
    plain, _ = update_input(ppo.combine(adv, contracts.OBJECTIVE_WEIGHTS), cost, regimes)
    assert np.allclose(today, stable, rtol=1e-9, atol=1e-12) and np.allclose(w_today, w_stable)
    assert np.allclose(stable, plain, rtol=1e-9, atol=1e-12)


def test_reference_combine_uses_fixed_exchange_rates() -> None:
    adv = advantages(); s = contracts.OBJECTIVE_REFERENCE_SCALES
    p = np.tile(np.asarray([0.5, 0.5, 0, 0], np.float32), (200, 1))
    expected = 0.5 * adv["welfare"] / s["welfare"] + 0.5 * adv["population_years"] / s["population_years"]
    assert np.allclose(ppo.combine_personal(adv, p, "reference"), expected)
    # Batch content does not move the exchange rate: doubling one component's spread doubles its share exactly.
    doubled = dict(adv); doubled["population_years"] = 2 * adv["population_years"]
    expected2 = 0.5 * adv["welfare"] / s["welfare"] + adv["population_years"] / s["population_years"]
    assert np.allclose(ppo.combine_personal(doubled, p, "reference"), expected2)
    with pytest.raises(ValueError):
        ppo.combine_personal(adv, p, "other")


def test_clip_stops_an_untrained_head_from_dominating() -> None:
    s = contracts.OBJECTIVE_REFERENCE_SCALES; rng = np.random.default_rng(4)
    welfare = rng.normal(scale=s["welfare"], size=500)
    noise = rng.normal(scale=1500 * s["trade_utility"], size=500)  # an untrained trade head: scale_drift ~1500
    adv = {"welfare": welfare, "population_years": np.zeros(500), "trade_utility": noise, "qol_years": np.zeros(500)}
    p = np.tile(np.asarray([0.75, 0, 0.25, 0], np.float32), (500, 1))
    unclipped = ppo.combine_personal(adv, p, "reference")
    clipped = ppo.combine_personal(adv, p, "reference", clip=5.0)
    signal = 0.75 * welfare / s["welfare"]
    assert abs(np.corrcoef(unclipped, signal)[0, 1]) < 0.1        # without the clip the noise is the advantage
    # With it the welfare signal survives: a saturated head adds at most +/- p_k * clip (here 1.25 against a signal
    # spread of 0.75), so it is bounded, not removed.
    assert np.corrcoef(clipped, signal)[0, 1] > 0.4
    assert np.abs(clipped - signal).max() <= 0.25 * 5.0 + 1e-9     # the trade term is bounded by p_k * clip
    fractions = ppo.clip_fractions(adv, 5.0)
    assert fractions["trade_utility"] > 0.99 and fractions["welfare"] < 0.01 and ppo.clip_fractions(adv, 0)["welfare"] == 0
    assert ppo.Hyper().personality_clip == 5.0
    # The clip leaves ordinary-sized components untouched.
    calm = {k: rng.normal(scale=s[k], size=500) for k in contracts.OBJECTIVES}
    calm = {k: np.clip(v, -4.9 * s[k], 4.9 * s[k]) for k, v in calm.items()}
    assert np.allclose(ppo.combine_personal(calm, p, "reference", 5.0), ppo.combine_personal(calm, p, "reference"))


def test_scale_drift_is_batch_spread_over_reference_spread() -> None:
    adv = advantages()
    drift = ppo.scale_drift(adv)
    for name, a in adv.items():
        assert drift[name] == pytest.approx(np.std(a) / contracts.OBJECTIVE_REFERENCE_SCALES[name])
    assert ppo.Hyper().personality_scales == "reference"


def test_scale_reference_returns_are_discounted_per_sequence() -> None:
    from anthropocene_rl.autonomous_v7 import scale_reference
    batch = {"world": np.array([1, 1, 1, 1]), "camp": np.array([5, 5, 5, 6]), "turn": np.array([0, 1, 2, 0]),
             "society": np.array([1, 1, 1, 2]), "parent_society": np.zeros(4, int), "prior_population": np.full(4, 10.0),
             "population": np.full(4, 10.0), "qol": np.full(4, 0.5), "years": np.full(4, 20.0),
             "terminated": np.array([False, False, False, True]), "trade_utility": np.zeros(4)}
    g = scale_reference.discounted_returns(batch, np.ones(4))
    gamma = np.exp(-20 / contracts.HORIZON_YEARS)
    assert np.allclose(g, [1 + gamma + gamma ** 2, 1 + gamma, 1, 1])
    report = scale_reference.reference_scales(batch, full_window=2)
    assert list(scale_reference.later_turns(batch)) == [2, 1, 0, 2]
    assert report["rows"] == 4 and report["objectives"]["welfare"]["full_window_rows"] == 2
    assert report["objectives"]["qol_years"]["mean"] > 0


# --------------------------------------------------------------------------------------------- daughter bank
def test_bank_index_rotation_and_held_out_slice(tmp_path) -> None:
    bank = DaughterBank(tmp_path / "bank", cap=3)
    for i in range(5):
        bank.add(bytes([i]) * 100, seed=1000 + i, mask=0, turn=10 + i, society=50 + i, camp=60 + i, parent_society=30)
    entries = bank.entries()
    assert [e["id"] for e in entries] == [2, 3, 4]  # oldest first out
    assert sorted(p.name for p in (tmp_path / "bank").glob("*.snap")) == sorted(e["file"] for e in entries)
    reopened = DaughterBank(tmp_path / "bank", cap=3)  # persists across runs
    assert reopened.snapshot(reopened.get(3)) == bytes([3]) * 100
    assert reopened.get(4)["parent_society"] == 30 and reopened.get(4)["seed"] == 1004
    assert reopened.add("AAEC", seed=1, mask=2, turn=3, society=4, camp=5, parent_society=6)["id"] == 5  # base64 input
    assert reopened.snapshot(reopened.get(5)) == bytes([0, 1, 2])
    # Same-world, same-turn captures share one file, which survives until its last entry rotates out.
    shared = DaughterBank(tmp_path / "shared", cap=2)
    a = shared.add(b"world", seed=1, mask=0, turn=3, society=4, camp=5, parent_society=6)
    b = shared.add(b"world", seed=1, mask=0, turn=3, society=7, camp=8, parent_society=6)
    assert a["file"] == b["file"] and len(list((tmp_path / "shared").glob("*.snap"))) == 1
    shared.add(b"other", seed=1, mask=0, turn=4, society=9, camp=10, parent_society=6)
    assert shared.snapshot(shared.get(b["id"])) == b"world"
    shared.add(b"third", seed=1, mask=0, turn=5, society=11, camp=12, parent_society=6)
    assert len(list((tmp_path / "shared").glob("*.snap"))) == 2
    big = DaughterBank(tmp_path / "big")
    for i in range(24):  # two daughters per world seed, sharing their world's snapshot
        seed = 1000 + i // 2
        big.add(b"world-%d" % seed, seed=seed, mask=contracts.CAPABILITY_MASKS["fishing"] if seed % 2 else 0, turn=0,
                society=i, camp=i, parent_society=1)
    # The held-out family is whole world seeds (seed % 10 == 0): both of seed 1000's and of 1010's daughters.
    assert [e["seed"] for e in big.evaluation_entries()] == [1000, 1000, 1010, 1010]
    assert all(held_out(e) for e in big.evaluation_entries())
    train_files = {e["file"] for e in big.training_entries()}
    assert train_files.isdisjoint({e["file"] for e in big.evaluation_entries()})  # no snapshot on both sides
    rng = np.random.default_rng(0)
    draws = [big.sample(rng)["seed"] for _ in range(300)]
    assert 1000 not in draws and 1010 not in draws and len(set(draws)) == 10  # never the held-out family
    # An id split would have leaked: a snapshot shared across the partitions is refused.
    leaky = DaughterBank(tmp_path / "leaky")
    leaky.add(b"same", seed=1020, mask=0, turn=0, society=1, camp=1, parent_society=0)
    leaky.add(b"same", seed=1021, mask=0, turn=0, society=2, camp=2, parent_society=0)
    with pytest.raises(AssertionError, match="leak"):
        leaky.training_entries()
    with pytest.raises(ValueError, match="held-out family"):
        daughter_eval.run_entry(str(tmp_path / "big"), big.training_entries()[0]["id"], None, 1)
    fishing = {contracts.CAPABILITY_MASKS["fishing"]}
    assert all(big.sample(rng, fishing)["mask"] in fishing for _ in range(20))
    assert DaughterBank(tmp_path / "empty").sample(rng) is None


def test_reservoir_spreads_captures_over_the_episode(tmp_path) -> None:
    from anthropocene_rl.autonomous_v7.daughter_bank import Reservoir
    kept_turns, first_three = [], []
    for world in range(400):
        rng = np.random.default_rng(world)
        foundings = sorted(rng.integers(10, 50, size=int(rng.integers(4, 30))).tolist())  # founding turns, in order
        reservoir = Reservoir(3, np.random.default_rng([world, 0xDA]))
        for turn in foundings:
            reservoir.offer(turn)
        assert len(reservoir.items) == 3 and reservoir.seen == len(foundings)
        kept_turns += reservoir.items; first_three += foundings[:3]
    # Uniform over all foundings (mean founding turn about 30), not the earliest ones.
    assert abs(np.mean(kept_turns) - 29.5) < 1.5 and np.mean(first_three) < 22
    again = Reservoir(3, np.random.default_rng([7, 0xDA])); once = Reservoir(3, np.random.default_rng([7, 0xDA]))
    for t in range(20):
        assert again.offer(t) == once.offer(t)  # deterministic under the keyed generator
    bank = DaughterBank(tmp_path / "b")
    a = bank.add(b"one", seed=1, mask=0, turn=10, society=2, camp=3, parent_society=1)
    bank.add(b"two", seed=1, mask=0, turn=20, society=4, camp=5, parent_society=1)
    bank.remove(a["id"]); bank.remove(99)
    assert [e["turn"] for e in bank.entries()] == [20] and len(list((tmp_path / "b").glob("*.snap"))) == 1


def test_world_sampler_puts_bank_worlds_in_slot_zero_and_keeps_the_valley_slot(tmp_path) -> None:
    train, grass = np.arange(1000, 1060), np.array([1100, 1101])
    stage = ppo.dataclasses.replace(ppo.STAGES["mixed"], movement=True, region_movement=True)
    valley = [(1200, 4)]
    plain = ppo.world_sampler(stage, train, grass, np.random.default_rng(3), valley)(8)
    empty = ppo.world_sampler(stage, train, grass, np.random.default_rng(3), valley, DaughterBank(tmp_path / "e"))(8)
    assert plain == empty  # an empty bank changes nothing, the rng included
    bank = DaughterBank(tmp_path / "b")
    for i in range(1, 4):
        bank.add(b"s%d" % i, seed=1000 + i, mask=0, turn=12, society=70 + i, camp=80 + i, parent_society=20)
    worlds = ppo.world_sampler(stage, train, grass, np.random.default_rng(3), valley, bank)(8)
    for i, w in enumerate(worlds):
        if i % 4 == 0:
            assert len(w) == 4 and w[3] in (0, 1, 2) and w[0] == 1001 + w[3] and w[1] == 0
        else:
            assert len(w) < 4
        if i % 4 == 2:
            assert w == (1200, 0, 4)
    assert ppo.regime(worlds[0][0], worlds[0][1], frozenset()) == "default"
    # Stage 2 draws only capability-fixture entries; with none, its slot is the usual fixture.
    cap = ppo.world_sampler(ppo.STAGES["capability"], train, grass, np.random.default_rng(3), None, bank)(4)
    assert all(len(w) == 2 and w[1] in contracts.CAPABILITY_MASKS.values() for w in cap)
    bank.add(b"h", seed=1051, mask=contracts.CAPABILITY_MASKS["herding"], turn=12, society=9, camp=9, parent_society=2)
    cap = ppo.world_sampler(ppo.STAGES["capability"], train, grass, np.random.default_rng(3), None, bank)(4)
    assert cap[0] == (1051, contracts.CAPABILITY_MASKS["herding"], 0, 3)
    assert ppo.regime(cap[0][0], cap[0][1], frozenset()) == "herding"


def test_search_picks_one_young_daughter_when_present() -> None:
    live = [{"camp": c, "parentSociety": p} for c, p in ((1, 0), (2, 0), (3, 0), (4, 7), (5, 0))]
    first_seen = {1: 0, 2: 0, 3: 0, 4: 6, 5: 0}
    for seed in range(20):
        chosen = search_teacher.choose_villages(live, first_seen, 8, np.random.default_rng(seed), 2)
        assert chosen[0]["camp"] == 4 and chosen[1]["camp"] != 4 and len(chosen) == 2
        uniform = [live[i] for i in np.random.default_rng(seed).permutation(5)[:2]]
        assert search_teacher.choose_villages(live, first_seen, 8, np.random.default_rng(seed), 2, False) == uniform
        # An old daughter is not young: the uniform draw, exactly as before.
        assert search_teacher.choose_villages(live, first_seen, 10, np.random.default_rng(seed), 2) == uniform
    known = search_teacher.choose_villages(live, first_seen, 30, np.random.default_rng(0), 2, True, frozenset({2}))
    assert known[0]["camp"] == 2
    assert search_teacher.SearchConfig().daughter_search is True


# --------------------------------------------------------------------------------------------- specialist share
def test_specialist_share_defaults_to_zero_and_is_population_weighted(monkeypatch) -> None:
    villages = [{"population": 30}, {"population": 10}]
    assert evaluate.specialist_share(villages) == 0.0  # bridges without the field
    villages = [{"population": 30, "specialistShare": 0.1}, {"population": 10, "specialistShare": 0.5}]
    assert evaluate.specialist_share(villages) == pytest.approx(0.2)
    assert evaluate.specialist_share([]) == 0.0

    def fake(seed, *args):
        return {"welfare": 1.0, "population": 10.0 * seed, "qol": 0.5, "below_30": 0, "extinctions": 0, "settlements": 1,
                "interventions_per_decision": 0.0, **({"specialist_share": 0.1 * seed} if seed != 3 else {})}
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    monkeypatch.setattr(evaluate, "rollout", fake)
    result = ppo.evaluate_closed_loop(None, [(1, 0), (2, 0), (3, 0)], 1, workers=2)
    assert result["specialist_share"] == pytest.approx((10 * 0.1 + 20 * 0.2 + 30 * 0.0) / 60)


def test_daughter_eval_pairs_outcomes() -> None:
    villages = [{"society": 5, "terminated": False, "population": 40, "qol": 0.5},
                {"society": 5, "terminated": False, "population": 10, "qol": 1.0},
                {"society": 6, "terminated": False, "population": 99, "qol": 0.1}]
    out = daughter_eval.daughter_outcome(villages, 5)
    assert out["alive"] == 1 and out["population"] == 50 and out["qol"] == pytest.approx(0.6)
    assert daughter_eval.daughter_outcome(villages, 7)["alive"] == 0
    policy = [{"alive": 1.0, "growth": 2.0, "qol": 0.6, "population": 60}, {"alive": 0.0, "growth": 0.0, "qol": 0.0, "population": 0}]
    heur = [{"alive": 1.0, "growth": 1.5, "qol": 0.5, "population": 45}, {"alive": 1.0, "growth": 1.0, "qol": 0.4, "population": 30}]
    report = daughter_eval.paired(policy, heur)
    assert report["alive"]["difference"] == pytest.approx(-0.5) and report["growth"]["difference"] == pytest.approx(-0.25)
    assert report["qol_both_alive"] == {"entries": 1, "difference": pytest.approx(0.1)}


# --------------------------------------------------------------------------------------------- bridge smokes
@needs_bridge
def test_bridge_capture_and_reset_from_snapshot_continue_identically() -> None:
    with Model9Bridge() as a, Model9Bridge() as b:
        a.reset(1000)
        for _ in range(2):
            a.step({})
        snapshot = a.capture()
        live = a.step({})
        start = b.reset(1000, snapshot=snapshot)
        assert not any(v["founded"] for v in start["villages"])
        restarted = b.step({})
        assert live["turn"] == restarted["turn"]
        assert [v["camp"] for v in live["villages"]] == [v["camp"] for v in restarted["villages"]]
        assert live["fingerprint"] == restarted["fingerprint"]


@needs_bridge
def test_collect_banks_founded_daughters_and_replays_them_with_inherited_personalities(tmp_path) -> None:
    from anthropocene_rl.autonomous_v7.bridge import observation_batch
    from anthropocene_rl.autonomous_v7.model import Actor, Normalizer
    with Model9Bridge() as bridge:
        step = bridge.reset(1000)
        core_width = bridge.schema.core_width
    normalizer = Normalizer.fit(observation_batch([v for v in step["villages"] if v["observation"] is not None]))
    torch.manual_seed(0)
    actor = Actor(core_width, token=True); actor.eval()
    stage = ppo.STAGES["mixed"]; generator = torch.Generator().manual_seed(0)
    bank = DaughterBank(tmp_path / "bank")
    ppo.collect(actor, normalizer, [1000], 3, stage, generator, workers=1, bank=bank)
    entries = bank.entries()
    assert 1 <= len(entries) <= 3 and all(e["parent_society"] > 0 and e["seed"] == 1000 for e in entries)
    entry = entries[0]
    rng = np.random.default_rng(1)
    roll = ppo.collect(actor, normalizer, [(1000, 0, 0, entry["id"])], 2, stage, generator, workers=1,
                       personality=ppo.personality_sampler("dirichlet:1.0", rng), inherit=lambda p: p, bank=bank)
    batch, _, _ = ppo.flatten(roll)
    assert (batch["bank"] == entry["id"]).all() and (batch["seed"] == 1000).all()
    daughter = batch["camp"] == entry["camp"]
    parent = batch["society"] == entry["parent_society"]
    assert daughter.any() and parent.any()
    # The banked daughter inherits its parent's personality (an exact copy here), founders draw fresh ones.
    assert np.allclose(batch["personality"][daughter][0], batch["personality"][parent][0])
    assert len({tuple(p) for p in batch["personality"].round(6).tolist()}) > 1
    # The daughter evaluation runs from the same snapshot.
    out = daughter_eval.run_entry(str(tmp_path / "bank"), entry["id"], None, 1)
    assert out["initial_population"] > 0 and out["alive"] in (0.0, 1.0)


# --------------------------------------------------------------------------------------------- one utility
def test_search_scores_on_ppos_reference_utility_and_rescores_old_labels(tmp_path) -> None:
    from test_autonomous_v7 import batch
    s = contracts.OBJECTIVE_REFERENCE_SCALES
    returns = np.array([[1000.0, 2000.0, 0.0, 30.0], [1300.0, 1800.0, 0.0, 30.0], [1000.0, 2000.0, 0.0, 40.0]])
    p = np.array([0.5, 0, 0, 0.5])
    u = search_teacher.personality_utility(returns, p)
    assert np.allclose(u, 0.5 * returns[:, 0] / s["welfare"] + 0.5 * returns[:, 3] / s["qol_years"])
    assert np.allclose(search_teacher.personality_score(returns, p), u - u[0])
    # The same personality ranks alternatives as PPO's combine_personal would weight their returns.
    adv = {k: returns[:, i] - returns[0, i] for i, k in enumerate(contracts.OBJECTIVES)}
    assert np.allclose(ppo.combine_personal(adv, np.tile(p, (3, 1)), "reference"), u - u[0])
    # A version-1 file (relative scores) with return vectors is rescored; its label (the old argmax, alternative 1)
    # stays only where it is still the version-2 best.
    b = batch(2); n = 2
    old_scores = np.array([[1.0, 1.15, 1.10], [1.0, 1.15, 1.10]])  # v1: alternative 1 won both rows
    alt_returns = np.stack([returns, returns])
    personality = np.array([[1.0, 0, 0, 0], [0, 0, 0, 1.0]])  # row 0: welfare (alt 1 best); row 1: QoL (alt 2 best)
    v1 = dict(b, label_kind=np.full(n, search_teacher.REGION), label_slot=np.full(n, 4),
              label_mix=np.full((n, 3), np.nan), score_policy=np.ones(n), score_best=np.full(n, 1.15),
              alt_kind=np.array([[0, 2, 2]] * n), alt_slot=np.array([[-1, 4, 5]] * n), alt_score=old_scores,
              alt_returns=alt_returns, personality=personality, alt_return=alt_returns[:, :, 0])
    rescored = search_teacher.rescore(v1)
    assert rescored["utility_version"].tolist() == [2, 2] and rescored["label_is_best"].tolist() == [True, False]
    assert rescored["score_best"][0] == pytest.approx(300 / s["welfare"]) and (rescored["score_policy"] == 0).all()
    np.savez(tmp_path / "v1.npz", **v1)
    kept = ppo.load_search_labels([tmp_path / "v1.npz"], margin=0.004)
    assert kept is not None and len(kept["label_kind"]) == 1 and kept["personality"][0][0] == 1.0  # 300 / s_w > 0.004
    assert ppo.load_search_labels([tmp_path / "v1.npz"], margin=0.01) is None                     # in s_welfare_ref units
    summary = search_teacher.summarize(v1, 0.004)
    assert summary["utility_version"] == 2 and summary["beats_policy_share"] == 0.5
