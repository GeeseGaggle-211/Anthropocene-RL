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
    assert report["rows"] == 4 and report["objectives"]["welfare"]["full_window_rows"] == 1
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
    for i in range(12):
        big.add(b"x", seed=i, mask=contracts.CAPABILITY_MASKS["fishing"] if i % 2 else 0, turn=0, society=i, camp=i, parent_society=1)
    assert [e["id"] for e in big.evaluation_entries()] == [0, 10] and all(held_out(e) for e in big.evaluation_entries())
    rng = np.random.default_rng(0)
    draws = [big.sample(rng)["id"] for _ in range(200)]
    assert 0 not in draws and 10 not in draws and len(set(draws)) == 10  # never the held-out slice
    fishing = {contracts.CAPABILITY_MASKS["fishing"]}
    assert all(big.sample(rng, fishing)["mask"] in fishing for _ in range(20))
    assert DaughterBank(tmp_path / "empty").sample(rng) is None


def test_world_sampler_puts_bank_worlds_in_slot_zero_and_keeps_the_valley_slot(tmp_path) -> None:
    train, grass = np.arange(1000, 1060), np.array([1100, 1101])
    stage = ppo.dataclasses.replace(ppo.STAGES["mixed"], movement=True, region_movement=True)
    valley = [(1200, 4)]
    plain = ppo.world_sampler(stage, train, grass, np.random.default_rng(3), valley)(8)
    empty = ppo.world_sampler(stage, train, grass, np.random.default_rng(3), valley, DaughterBank(tmp_path / "e"))(8)
    assert plain == empty  # an empty bank changes nothing, the rng included
    bank = DaughterBank(tmp_path / "b")
    for i in range(1, 4):
        bank.add(b"s", seed=1000 + i, mask=0, turn=12, society=70 + i, camp=80 + i, parent_society=20)
    worlds = ppo.world_sampler(stage, train, grass, np.random.default_rng(3), valley, bank)(8)
    for i, w in enumerate(worlds):
        if i % 4 == 0:
            assert len(w) == 4 and w[3] in (1, 2) and w[0] == 1001 + w[3] and w[1] == 0  # id 0 is held out
        else:
            assert len(w) < 4
        if i % 4 == 2:
            assert w == (1200, 0, 4)
    assert ppo.regime(worlds[0][0], worlds[0][1], frozenset()) == "default"
    # Stage 2 draws only capability-fixture entries; with none, its slot is the usual fixture.
    cap = ppo.world_sampler(ppo.STAGES["capability"], train, grass, np.random.default_rng(3), None, bank)(4)
    assert all(len(w) == 2 and w[1] in contracts.CAPABILITY_MASKS.values() for w in cap)
    bank.add(b"s", seed=1050, mask=contracts.CAPABILITY_MASKS["herding"], turn=12, society=9, camp=9, parent_society=2)
    cap = ppo.world_sampler(ppo.STAGES["capability"], train, grass, np.random.default_rng(3), None, bank)(4)
    assert cap[0] == (1050, contracts.CAPABILITY_MASKS["herding"], 0, 3)
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
