"""Round 3: fission and trade probes, policy tails, critic branch shards, critic mixing and warm-up (no bridge)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import branch_bias, branch_shards, contracts, data, ppo, search_teacher as st
from anthropocene_rl.autonomous_v7.model import Actor, Critic, Normalizer, VectorCritic, to_tensors
from test_autonomous_v7 import batch

GROUPS = len(contracts.TRADE_GROUPS)


def observation(seed: int = 0, trade_rows=(0, 3), held=(5.0, 40.0)) -> dict:
    """A bridge-shaped observation (lists, as the JSON bridge sends) with trade rows present at ``trade_rows``."""
    b = batch(1, seed)
    sets, present = [], []
    for name, rows, width in contracts.SETS:
        x = b[name][0].copy(); p = b[name + "_present"][0].copy()
        if name == "trade":
            p[:] = False; x[:, st.TRADE_HELD] = 0.0
            for r, h in zip(trade_rows, held):
                p[r] = True; x[r, st.TRADE_HELD] = h
        sets.append(x.reshape(-1).tolist()); present.append(p.tolist())
    region = [True, True] + [False] * contracts.REGION_ROWS
    return {"core": b["core"][0].tolist(), "corePresent": b["core_present"][0].tolist(),
            "movementMask": [True] * contracts.MOVEMENT_SLOTS, "clothingRouteMask": [True, True],
            "shelterRouteMask": [True, True], "economyMixMask": [True, False, False], "commitHoldMask": False,
            "regionMovementMask": region, "sets": sets, "rowPresent": present}


def village(camp=7, society=3, parent=0, seed=0, **kw) -> dict:
    return {"camp": camp, "society": society, "parentSociety": parent, "observation": observation(seed, **kw),
            "privileged": [0.1 * seed, 1.0, 2.0], "terminated": False, "priorPopulation": 40, "population": 42,
            "qol": 0.8, "nutrition": 1.0, "years": 20.0, "welfare": 640.0, "tradeUtility": 0.5}


BASE = {"strategic": [0.5, 0.5, 0.5, 1.0, 0.0], "movement": contracts.NATIVE_MOVEMENT, "fission": -1}


# ------------------------------------------------------------------------------------------------ probes
def test_fission_and_trade_probes_join_the_alternatives() -> None:
    alts = st.alternatives(village(), BASE, st.SearchConfig())
    kinds = [a["kind"] for a in alts]
    assert [a["fission"] for a in alts if a["kind"] == st.FISSION] == [2, 1]
    trade = [a for a in alts if a["kind"] == st.TRADE]
    # The native posture already makes to order: that probe changes nothing and is skipped.
    assert [a["slot"] for a in trade] == [0, 1, 3, 4]
    assert len(alts) <= st.MAX_ALTERNATIVES and kinds[0] == st.POLICY
    withhold = trade[0]["trade"]
    assert withhold["gate"] == [0] * GROUPS and withhold["value"] == [1.0] * GROUPS
    # Only the present groups (rows 0 and 3) count as changed, and only their gates.
    assert withhold["mask"].tolist() == [st.TRADE_GATE_BIT, 0, 0, st.TRADE_GATE_BIT, 0, 0, 0]
    half, double = trade[2]["trade"], trade[3]["trade"]
    assert half["value"][3] == 0.5 and double["value"][3] == 2.0  # row 3 holds the most
    assert half["mask"].tolist() == [0, 0, 0, st.TRADE_VALUE_BIT, 0, 0, 0] and half["gate"] == [2] * GROUPS
    off = st.alternatives(village(), BASE, st.SearchConfig(fission=False, trade=False))
    assert not any(a["kind"] in (st.FISSION, st.TRADE) for a in off)


def test_probes_skip_the_policys_own_choice_and_villages_without_trade() -> None:
    learned = dict(BASE, fission=2)
    assert [a["fission"] for a in st.alternatives(village(), learned, st.SearchConfig()) if a["kind"] == st.FISSION] == [1]
    assert not [a for a in st.alternatives(village(trade_rows=(), held=()), BASE, st.SearchConfig()) if a["kind"] == st.TRADE]
    # A value already at the clamp is not tilted past it.
    posture = {"value": [4.0] * GROUPS, "gate": [2] * GROUPS, "share": 0.3}
    slots = [a["slot"] for a in st.trade_probes(village(), posture)]
    assert 4 not in slots and 3 in slots


def test_base_posture_prefers_the_policy_then_the_teacher_then_native() -> None:
    own = dict(BASE, tradeValue=[2.0] * GROUPS, tradeGate=[1] * GROUPS, tradeShare=0.4)
    teacher = {"tradeValue": [1.0] * GROUPS, "tradeGate": [2] * GROUPS, "tradeShare": 0.25}
    assert st.base_posture(own, teacher) == {"value": [2.0] * GROUPS, "gate": [1] * GROUPS, "share": 0.4}
    assert st.base_posture(BASE, teacher)["share"] == 0.25
    assert st.base_posture(BASE, None) == {"value": [1.0] * GROUPS, "gate": [2] * GROUPS, "share": -1.0}


def test_override_holds_fission_and_a_full_trade_posture_every_turn() -> None:
    actions = {7: dict(BASE), 8: dict(BASE)}
    v = village()
    for first in (True, False):
        f = st.override(actions, 7, {"kind": st.FISSION, "fission": 2}, v, first)
        assert f[7]["fission"] == 2 and f[8]["fission"] == -1
        probe = st.trade_probes(v, st.base_posture(BASE, None))[0]
        t = st.override(actions, 7, probe, v, first)[7]
        assert len(t["tradeValue"]) == GROUPS and len(t["tradeGate"]) == GROUPS and t["tradeShare"] == -1.0
    assert "tradeValue" not in actions[7] and actions[7]["fission"] == -1  # inputs unchanged


# ------------------------------------------------------------------------------------------------ labels
def test_label_arrays_store_fission_and_trade_labels() -> None:
    v = village()
    alts = st.alternatives(v, BASE, st.SearchConfig(regions=False, mixes=False, native=False))
    trade = next(i for i, a in enumerate(alts) if a["kind"] == st.TRADE)
    fission = next(i for i, a in enumerate(alts) if a["kind"] == st.FISSION)

    def results(win: int):
        out = []
        for i in range(len(alts)):
            r = np.array([100.0 + 10 * (i == win), 0, 0, 0])
            out.append({"returns": r, "return": float(r[0]), "bootstrap": 0.0, "bootstraps": np.zeros(4), "world": 1.0,
                        "native_mix": None})
        return out
    world = {"seed": 1, "withheld": 0, "rows": [
        {"village": v, "alternatives": alts, "results": results(trade), "turn": 2, "personality": [1.0, 0, 0, 0]},
        {"village": v, "alternatives": alts, "results": results(fission), "turn": 3, "personality": [1.0, 0, 0, 0]}]}
    labels = st.label_arrays([world])
    assert labels["label_kind"].tolist() == [st.TRADE, st.FISSION]
    assert labels["label_fission"].tolist() == [-1, alts[fission]["fission"]]
    assert labels["label_trade_gate"][0].tolist() == alts[trade]["trade"]["gate"]
    assert labels["label_trade_mask"][0].tolist() == alts[trade]["trade"]["mask"].tolist()
    assert (labels["label_trade_mask"][1] == 0).all() and np.isnan(labels["label_trade_value"][1]).all()
    assert labels["alt_kind"].shape[1] == st.MAX_ALTERNATIVES


def test_old_and_new_label_files_load_together(tmp_path) -> None:
    b = batch(4); n = 4
    common = dict(b, label_kind=np.full(n, st.REGION), label_slot=np.full(n, 4), label_mix=np.full((n, 3), np.nan),
                  score_policy=np.full(n, 1.0), score_best=np.full(n, 1.1))
    np.savez(tmp_path / "old.npz", **common, alt_kind=np.zeros((n, 12), int))
    new = dict(common, label_kind=np.full(n, st.FISSION), label_fission=np.full(n, 2), alt_kind=np.zeros((n, st.MAX_ALTERNATIVES), int),
               label_trade_value=np.full((n, GROUPS), np.nan), label_trade_gate=np.full((n, GROUPS), -1),
               label_trade_mask=np.zeros((n, GROUPS), np.int8))
    np.savez(tmp_path / "new.npz", **new)
    rows = ppo.load_search_labels([tmp_path / "old.npz", tmp_path / "new.npz"], 0.005)
    assert rows["label_fission"].tolist() == [-1] * 4 + [2] * 4
    assert rows["alt_kind"].shape == (8, st.MAX_ALTERNATIVES) and (rows["alt_kind"][:4, 12:] == -1).all()


def test_search_loss_learns_fission_and_only_the_masked_trade_components() -> None:
    b = batch(8); n = 8
    normalizer = Normalizer.fit(b)
    torch.manual_seed(0); actor = Actor(b["core"].shape[1])
    t = to_tensors(b, normalizer)
    kind = torch.full((n,), st.FISSION); kind[4:] = st.TRADE
    slot = torch.full((n,), -1); mix = torch.full((n, 3), float("nan"))
    fission = torch.tensor([2, 2, 1, 2, -1, -1, -1, -1])
    value = torch.full((n, GROUPS), float("nan")); value[4:, 3] = 2.0
    gate = torch.full((n, GROUPS), -1, dtype=torch.long); gate[4:] = 0
    mask = torch.zeros((n, GROUPS), dtype=torch.long); mask[4:, 3] = st.TRADE_VALUE_BIT; mask[4:, 0] = st.TRADE_GATE_BIT
    extra = dict(fission=fission, trade_value=value, trade_gate=gate, trade_mask=mask)

    # Gradients reach exactly the labelled components: no other value mean, no log-std, no other group's gate.
    out = {k: v.detach().requires_grad_(True) for k, v in actor(t).items()}
    ppo.search_loss(out, t, kind, slot, mix, **extra).backward()
    gv, gg = out["trade_value_mean"].grad, out["trade_gate"].grad
    assert (gv[4:, 3] != 0).all() and gv[:, [0, 1, 2, 4, 5, 6]].abs().sum() == 0 and gv[:4].abs().sum() == 0
    assert out["trade_value_log_std"].grad is None or out["trade_value_log_std"].grad.abs().sum() == 0
    assert gg[4:, 0].abs().sum() > 0 and gg[:, 1:].abs().sum() == 0
    assert out["fission"].grad[:4].abs().sum() > 0 and out["fission"].grad[4:].abs().sum() == 0

    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-2)
    first = float(ppo.search_loss(actor(t), t, kind, slot, mix, **extra))
    for _ in range(60):
        loss = ppo.search_loss(actor(t), t, kind, slot, mix, **extra); optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert float(ppo.search_loss(actor(t), t, kind, slot, mix, **extra)) < first
    with torch.no_grad():
        o = actor(t)
    assert (o["fission"][:4].argmax(dim=1) == fission[:4]).float().mean() >= 0.75
    assert torch.allclose(o["trade_value_mean"][4:, 3], torch.full((4,), float(np.log(2.0))), atol=0.2)
    assert (o["trade_gate"][4:, 0].argmax(dim=1) == 0).all()


# ------------------------------------------------------------------------------------------------ shards
def fake_tail(seed: int, camps=(7, 11), turns=(4, 5), boot=True, terminate_last=False) -> dict:
    rows = []
    for i, turn in enumerate(turns):
        for j, camp in enumerate(camps):
            dec = village(camp=camp, society=3 if j == 0 else 9, parent=0 if j == 0 else 3, seed=seed + i + j)
            out = dict(dec, priorPopulation=40 + i, population=41 + i, qol=0.7 + 0.05 * j, welfare=(40 + i) * (0.7 + 0.05 * j) * 20)
            out["terminated"] = bool(terminate_last and j == 1 and i == len(turns) - 1)
            rows.append({"decision": dec, "outcome": out, "turn": turn, "personality": [1.0, 0, 0, 0],
                         "founded_turn": 0 if j == 0 else 1})
    bootstrap = []
    if boot:
        for i, r in enumerate(rows):
            if r["turn"] == turns[-1] and not r["outcome"]["terminated"]:
                r["bootstrap_index"] = len(bootstrap)
                bootstrap.append({"village": village(camp=r["outcome"]["camp"], seed=99 + i), "personality": [1.0, 0, 0, 0]})
    return {"rows": rows, "bootstrap": bootstrap}


def write_two(tmp_path):
    root = tmp_path / "branch_shards"
    for a, kind in ((0, st.POLICY), (1, st.REGION)):
        meta = {"seed": 1001, "withheld": 0, "warmup": 0, "point": 0, "alternative": a, "kind": kind,
                "world_id": branch_shards.world_id(1001, 0, 0, 0, a), "calibration": a}
        branch_shards.write(root / f"b{a}", fake_tail(a, terminate_last=a == 1), meta)
    return root


def test_branch_shards_round_trip_through_data_load(tmp_path) -> None:
    root = write_two(tmp_path)
    rows = data.load(p for p in root.iterdir())
    for key in data.OBSERVATION_KEYS + data.LABEL_KEYS + data.OUTCOME_KEYS:
        assert key in rows
    assert not rows["strategic_mask"].any() and not rows["movement_label_mask"].any()  # never actor rows
    assert len(set(rows["world"].tolist())) == 2 and rows["world"].min() >= branch_shards.BRANCH_WORLD_BASE
    assert rows["alt_kind"].tolist() == [st.POLICY] * 4 + [st.REGION] * 4
    assert rows["calibration"].tolist() == [0] * 4 + [1] * 4
    assert rows["personality"].shape == (8, contracts.PERSONALITY_WIDTH) and rows["privileged"].shape == (8, 3)
    assert len(data.sequences(rows)) == 4  # (world, camp): branches never share a sequence
    components = data.reward_components(rows)
    assert set(components) == set(contracts.OBJECTIVES)
    # Bootstrap blocks are rebased onto one concatenated block.
    branch_rows, boot = branch_shards.load([root])
    index = branch_rows["bootstrap_index"]
    assert index.tolist() == [-1, -1, 0, 1, -1, -1, 2, -1] and len(boot["core"]) == 3
    assert branch_shards.world_id(1001, 0, 0, 0, 1) != branch_shards.world_id(1001, 0, 0, 1, 0)


def test_founding_turns_keep_the_lineage_window_of_the_full_path() -> None:
    d = {"world": np.array([5, 5]), "camp": np.array([1, 2]), "turn": np.array([10, 10]), "society": np.array([3, 9]),
         "parent_society": np.array([0, 3]), "prior_population": np.array([100, 20]), "qol": np.array([0.8, 0.4]),
         "years": np.array([20.0, 20.0]), "terminated": np.zeros(2, bool)}
    shared = data.lineage_rewards(d)  # first seen at turn 10: a "recent" daughter shares the parent's QoL
    assert shared[1] == pytest.approx(20 * (100 * 0.8 + 20 * 0.4) / 120 * 20)
    own = data.lineage_rewards(dict(d, founded_turn=np.array([0, 2])))  # founded at turn 2: its own QoL
    assert own[1] == pytest.approx(20 * 0.4 * 20)


def test_lambda_returns_bootstrap_after_the_last_tail_turn(tmp_path) -> None:
    d = {"world": np.zeros(2), "camp": np.zeros(2), "turn": np.array([0, 1]), "years": np.full(2, 20.0),
         "terminated": np.zeros(2, bool)}
    reward, value, boot = np.array([1.0, 2.0]), np.array([10.0, 11.0]), np.array([np.nan, 50.0])
    g = data.discount(np.array([20.0]))[0]
    out = data.lambda_returns(d, reward, value, 0.5, boot)
    assert out[1] == pytest.approx(2 + g * 50)
    assert out[0] == pytest.approx(1 + g * (0.5 * 11 + 0.5 * out[1]))
    assert data.lambda_returns(d, reward, value, 0.5)[1] == pytest.approx(2 + g * 11)  # unchanged without it
    # Branch targets: the same equations through the current critic, per head, in the head's own units.
    rows, bootstrap = branch_shards.load([write_two(tmp_path)])
    normalizer = Normalizer.fit(rows)
    torch.manual_seed(0); critic = VectorCritic(rows["core"].shape[1], privileged_width=3)
    targets = branch_shards.targets(critic, rows, bootstrap, normalizer)
    value_by, boot_by = branch_shards.values(critic, rows, bootstrap, normalizer)
    comp = data.reward_components(rows)
    for name in contracts.OBJECTIVES:
        expected = data.lambda_returns(rows, comp[name], value_by[name], 0.95, boot_by[name]) / ppo.head_scale(name, rows["prior_population"])
        assert np.allclose(targets[name], expected)
    assert np.isfinite(boot_by["welfare"][rows["bootstrap_index"] >= 0]).all()
    scalar = branch_shards.targets(Critic(rows["core"].shape[1], privileged_width=3), rows, bootstrap, normalizer)
    assert isinstance(scalar, np.ndarray) and scalar.shape == (8,)
    # Validation tool: per kind and head, relative bias of prediction against realised return plus bootstrap.
    pred, real = branch_bias.realised(critic, rows, bootstrap, normalizer)
    report = branch_bias.report(pred, real, rows)
    assert set(report["kinds"]) == {"policy", "region"} and report["kinds"]["region"]["welfare"]["branches"] == 1
    full = branch_bias.evaluate(pred, real, rows)
    # The calibration shard's 2-turn tail ends with its lineage alive: too short for an 8-turn check, so no pass.
    assert full["calibration"]["short"] == 1 and full["calibration"]["branches"] == 0 and full["pass"] is False
    json.dumps(full)


def calibration_rows(turns: int = 10, die_at: int | None = None) -> dict[str, np.ndarray]:
    """One calibration branch, one settlement, reward 100 per turn (welfare = P x Q x dt = 10 x 0.5 x 20)."""
    t = np.arange(turns)
    terminated = np.zeros(turns, bool)
    if die_at is not None:
        t = t[:die_at + 1]; terminated = np.zeros(len(t), bool); terminated[-1] = True
    n = len(t)
    return {"world": np.full(n, 77), "camp": np.zeros(n, int), "turn": t, "society": np.ones(n, int),
            "parent_society": np.zeros(n, int), "prior_population": np.full(n, 10), "population": np.full(n, 10),
            "qol": np.full(n, 0.5), "years": np.full(n, 20.0), "terminated": terminated,
            "welfare": np.full(n, 100.0), "alt_kind": np.full(n, st.NATIVE), "calibration": np.ones(n, np.int8)}


def test_calibration_compares_the_implied_8_turn_value_with_8_realised_turns() -> None:
    rows = calibration_rows()
    g = float(data.discount(np.array([20.0]))[0])
    realised = sum(100 * g ** k for k in range(8))
    full = 100 / (1 - g)  # a perfectly calibrated stationary value
    pred = {"welfare": np.full(len(rows["camp"]), full)}
    cal = branch_bias.calibration(pred, rows)
    assert cal["branches"] == 1
    assert abs(cal["kinds"]["native"]["welfare"]["mean"]) < 1e-9  # V0 - g^8 V8 is exactly the 8-turn return
    assert np.isclose(full - g ** 8 * full, realised)
    biased = branch_bias.calibration({"welfare": pred["welfare"] * 1.2}, rows)
    assert np.isclose(biased["kinds"]["native"]["welfare"]["median"], 0.2)
    real = {"welfare": np.zeros(len(rows["camp"]))}
    # One native branch proves the formula, not mandatory native/region/mix coverage or held-out provenance.
    assert branch_bias.evaluate(pred, real, rows)["pass"] is False
    assert branch_bias.evaluate({"welfare": pred["welfare"] * 1.2}, real, rows)["pass"] is False
    # A lineage that dies before turn 8: its whole realised return against V0 (nothing left to subtract).
    dead = calibration_rows(die_at=3)
    dead["welfare"][-1] = 0.0
    cal = branch_bias.calibration({"welfare": np.full(4, 100.0 * (1 + g + g * g))}, dead)
    assert cal["branches"] == 1 and abs(cal["kinds"]["native"]["welfare"]["fixed_mean"]) < 1e-9
    assert branch_bias.calibration(pred, dict(rows, calibration=np.zeros(10, np.int8)))["branches"] == 0


def test_calibration_branches_are_keyed_and_take_the_configured_share() -> None:
    draws = [st.calibration_branch(1000, 0, 0, turn, camp, a, 0.1) for turn in range(10) for camp in range(10) for a in range(10)]
    assert draws == [st.calibration_branch(1000, 0, 0, turn, camp, a, 0.1) for turn in range(10) for camp in range(10) for a in range(10)]
    assert 0.06 < np.mean(draws) < 0.14
    assert not any(st.calibration_branch(1000, 0, 0, 1, 2, a, 0.0) for a in range(50))
    assert st.SearchConfig().calibration_share == 0.1 and st.SearchConfig().calibration_tail == 12


def quiet(v: dict) -> dict:
    """The village with no trade activity in its contacts or trade rows."""
    obs = v["observation"]
    for index, columns in ((st.CONTACT_SET, st.CONTACT_TRADE), (st.TRADE_SET, st.TRADE_ACTIVITY)):
        _, rows, width = contracts.SETS[index]
        table = np.asarray(obs["sets"][index]).reshape(rows, width)
        table[:, list(columns)] = 0.0
        obs["sets"][index] = table.reshape(-1).tolist()
    return v


def test_trade_probes_run_only_for_villages_with_recent_trade() -> None:
    v = quiet(village())
    assert not st.trade_active(v, [0.0, 0.0, 0.0]) and not st.trade_active(v, None)
    assert st.trade_active(v, [0.0, 0.3, 0.0])
    history: dict = {}
    for u in (0.5, 0.0, 0.0, 0.0):
        st.note_trade(history, {"villages": [{"camp": 7, "tradeUtility": u}]})
    assert history[7] == [0.0, 0.0, 0.0] and not st.trade_active(v, history[7])  # the trade 4 turns ago has aged out
    contact = quiet(village())
    _, rows, width = contracts.SETS[st.CONTACT_SET]
    table = np.asarray(contact["observation"]["sets"][st.CONTACT_SET]).reshape(rows, width)
    present = contact["observation"]["rowPresent"][st.CONTACT_SET]
    present[:] = [False] * rows; present[2] = True; table[2, 4] = 0.01  # one known contact with trade flow
    contact["observation"]["sets"][st.CONTACT_SET] = table.reshape(-1).tolist()
    assert st.trade_active(contact, None)
    traded = quiet(village())
    _, rows, width = contracts.SETS[st.TRADE_SET]
    table = np.asarray(traded["observation"]["sets"][st.TRADE_SET]).reshape(rows, width)
    table[3, 6] = 2.0  # imported last turn on a present trade row
    traded["observation"]["sets"][st.TRADE_SET] = table.reshape(-1).tolist()
    assert st.trade_active(traded, None)
    kinds = [a["kind"] for a in st.alternatives(v, BASE, st.SearchConfig(), trade_ok=False)]
    assert st.TRADE not in kinds and st.FISSION in kinds


# ------------------------------------------------------------------------------------------------ PPO critic
def test_warmup_scheduling() -> None:
    assert ppo.warmup_iterations(False, 5, 0) == 0
    assert ppo.warmup_iterations(True, 5, 0) == 5
    assert ppo.warmup_iterations(False, 5, 3) == 3
    assert ppo.warmup_iterations(True, 5, 8) == 8 and ppo.warmup_iterations(True, 5, 2) == 5
    # The loop trains the actor from iteration `warmup`; a resume at or past it runs no further critic-only iteration.
    warm = ppo.warmup_iterations(True, 5, 8)
    assert [it >= warm for it in range(9, 12)] == [True] * 3


def ppo_fixture(n: int = 16):
    b = batch(n)
    b.update({"camp": np.arange(n) % 4, "turn": np.arange(n) // 4, "world": np.zeros(n, int), "years": np.full(n, 20.0),
              "prior_population": np.full(n, 50), "population": np.full(n, 50), "terminated": np.zeros(n, bool)})
    normalizer = Normalizer.fit(b)
    torch.manual_seed(0)
    actor, teacher = Actor(b["core"].shape[1]), Actor(b["core"].shape[1])
    critic, cost = VectorCritic(b["core"].shape[1]), ppo.CostCritic(b["core"].shape[1])
    stage = ppo.STAGES["capability"]
    t = to_tensors(b, normalizer)
    with torch.no_grad():
        draws = ppo.sample(actor(t), t, stage, torch.Generator().manual_seed(0))
        lp, _ = ppo.log_prob(actor(t), t, draws, stage)
    targets = {name: np.full(n, 3.0) for name in critic.heads}
    optimizers = (torch.optim.Adam(actor.parameters(), lr=1e-5), torch.optim.Adam(critic.parameters(), lr=1e-3),
                  torch.optim.Adam(cost.parameters(), lr=1e-3))
    return b, normalizer, actor, teacher, critic, cost, stage, draws, lp.numpy(), targets, optimizers


def run_update(hyper, branch=None):
    b, normalizer, actor, teacher, critic, cost, stage, draws, lp, targets, optimizers = ppo_fixture()
    steps = {"n": 0}
    step = optimizers[1].step

    def counted(*a, **k):
        steps["n"] += 1
        return step(*a, **k)
    optimizers[1].step = counted
    n = len(lp)
    stats = ppo.update(actor, critic, cost, teacher, normalizer, b, draws, lp, np.linspace(-1, 1, n), targets,
                       np.zeros(n), np.zeros(n), 0.0, stage, hyper, optimizers, np.random.default_rng(0), branch=branch)
    return stats, steps["n"]


def test_kl_refit_runs_one_extra_critic_epoch_only_when_kl_stopped_the_update() -> None:
    assert not ppo.kl_refit_due({"approx_kl": 1.0}, ppo.Hyper())  # off by default in code
    assert ppo.kl_refit_due({"approx_kl": 0.05}, ppo.Hyper(kl_refit=True, target_kl=0.03))
    assert not ppo.kl_refit_due({"approx_kl": 0.01}, ppo.Hyper(kl_refit=True, target_kl=0.03))
    stopped = dict(epochs=4, target_kl=-1.0)  # every update "exceeds" a negative target: stop after epoch 1
    stats, steps = run_update(ppo.Hyper(kl_refit=True, **stopped))
    assert stats["kl_refit"] is True and stats["epochs"] == 1 and steps == 2
    stats, steps = run_update(ppo.Hyper(kl_refit=False, **stopped))
    assert "kl_refit" not in stats and steps == 1
    stats, steps = run_update(ppo.Hyper(kl_refit=True, epochs=2, target_kl=10.0))
    assert stats["kl_refit"] is False and steps == 2


def test_critic_minibatches_mix_branch_rows(tmp_path) -> None:
    rows, boot = branch_shards.load([write_two(tmp_path)])
    b, normalizer, actor, teacher, critic, *_ = ppo_fixture()
    for key in ("privileged",):
        rows.pop(key)  # the fixture critic has no privileged block
    rows.pop("personality")
    targets = {name: np.full(len(rows["camp"]), 7.0) for name in contracts.OBJECTIVES}
    stats, steps = run_update(ppo.Hyper(epochs=1, branch_share=0.25), branch=(rows, targets))
    assert steps == 1 and np.isfinite(stats["value"])
