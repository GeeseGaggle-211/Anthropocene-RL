"""Learner-derived critic-only lineage token (``initialize --privileged-lineage``).

Off is bitwise the historical privileged critic; on adds one bounded token
(is-daughter, min(turns since founding, 4) / 4) derived offline from stored rows
and online from the rollout's founded turns and parentSociety.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
from test_labor_privileged_bridge import _raw
from test_labor_privileged_model import _critics, _layout, _payload, _restore, _village

from anthropocene_rl.autonomous_v7 import labor_pretrain_eval as pe
from anthropocene_rl.autonomous_v7.bridge import observation_batch
from anthropocene_rl.autonomous_v7.labor_model import (
    LaborActor,
    LaborCostCritics,
    LaborTaskCritic,
    bind_privileged_normalizer,
    critic_privileged_batch,
    labor_checkpoint_payload,
    labor_checkpoint_privileged_lineage,
    labor_policy_tensors,
    load_labor_checkpoint_models,
)
from anthropocene_rl.autonomous_v7.labor_privileged import (
    LINEAGE_CRITIC_KEYS,
    LINEAGE_RAW_KEY,
    YOUNG_DAUGHTER_ROW_AGE,
    offline_lineage,
    online_lineage,
)
from anthropocene_rl.autonomous_v7.labor_training import (
    _models_from_checkpoint,
    _settlement_guard,
    rollout_privileged_raw,
)
from anthropocene_rl.autonomous_v7.labor_value_calibration import labor_critic_digest
from anthropocene_rl.autonomous_v7.model import Normalizer
from anthropocene_rl.autonomous_v7.ppo import YOUNG_DAUGHTER_TURNS

B01 = Path(
    r"C:\Users\suyas\AppData\Local\Temp\claude\C--Users-suyas-Documents-ChatGPT-Anthropocene"
    r"\0cac0cc5-1bf8-4f14-ae95-34de833627e0\scratchpad\qbdata\fit8-b01"
)


def _lineage_critics(layout, seed: int = 3):
    from test_labor_privileged_bridge import _normalizer

    torch.manual_seed(seed)
    task = LaborTaskCritic(layout, width=16, token_width=16, privileged_lineage=True)
    costs = LaborCostCritics(layout, width=16, token_width=16, privileged_lineage=True)
    bind_privileged_normalizer(_normalizer(), task, costs)
    return task, costs


def _lineage_payload():
    village = _village(1)
    normalizer = Normalizer.fit_bounded(observation_batch([village]))
    _, tensors = labor_policy_tensors([village], normalizer)
    layout = _layout(True, core=int(tensors["core"].shape[1]))
    torch.manual_seed(5)
    actor = LaborActor(layout, width=16, token_width=16)
    task, costs = _lineage_critics(layout)
    return labor_checkpoint_payload(actor, task, costs, normalizer, layout), layout, tensors


def _restore_lineage(state, layout):
    actor = LaborActor(layout, width=16, token_width=16)
    lineage = labor_checkpoint_privileged_lineage(state)
    task = LaborTaskCritic(layout, width=16, token_width=16, privileged_lineage=lineage)
    costs = LaborCostCritics(layout, width=16, token_width=16, privileged_lineage=lineage)
    load_labor_checkpoint_models(state, actor, task, costs, expected_layout=layout)
    return actor, task, costs


# --- flag off: bitwise historical -------------------------------------------------


def test_lineage_off_checkpoint_has_no_lineage_bytes() -> None:
    state, layout, _ = _payload(True)
    assert "lineage" not in state["labor_model_contract"]["privilegedCritic"]
    assert not any("lineage" in key for key in state["task_critic"])
    assert not any("lineage" in key for key in state["cost_critics"])
    assert labor_checkpoint_privileged_lineage(state) is False
    _restore(state, layout)  # historical constructor path still loads


def test_lineage_off_critic_ignores_a_lineage_array_and_on_keeps_other_init() -> None:
    layout = _layout()
    task, _costs = _critics(layout)
    raw = _raw(3)
    plain = critic_privileged_batch(task, raw)
    with_key = critic_privileged_batch(task, dict(raw, **{LINEAGE_RAW_KEY: torch.ones(3, 2)}))
    assert set(plain) == set(with_key) and not set(LINEAGE_CRITIC_KEYS) & set(plain)
    for key in plain:
        assert torch.equal(plain[key], with_key[key])
    # Lineage parameters are built under a forked RNG after everything else.
    torch.manual_seed(11)
    off = LaborTaskCritic(layout, width=16, token_width=16).state_dict()
    torch.manual_seed(11)
    on = LaborTaskCritic(layout, width=16, token_width=16, privileged_lineage=True)
    extra = [k for k in on.state_dict() if k not in off]
    assert extra and all("lineage_" in k for k in extra)
    for key, value in off.items():
        assert torch.equal(value, on.state_dict()[key]), key
    # And the global RNG stream after construction is unchanged.
    torch.manual_seed(11)
    LaborTaskCritic(layout, width=16, token_width=16)
    a = torch.rand(4)
    torch.manual_seed(11)
    LaborTaskCritic(layout, width=16, token_width=16, privileged_lineage=True)
    assert torch.equal(a, torch.rand(4))


def test_lineage_needs_a_privileged_critic() -> None:
    with pytest.raises(ValueError, match="lineage token needs a privileged critic"):
        LaborTaskCritic(_layout(False), width=16, token_width=16, privileged_lineage=True)


# --- flag on: conditioning, checkpoint, resume, fingerprint -----------------------


def test_lineage_critic_reads_the_token_and_requires_it() -> None:
    from test_labor_model import _labor_tensors

    tensors = _labor_tensors(batch=2)
    task, costs = _lineage_critics(_layout())
    raw = _raw(2)
    with pytest.raises(ValueError, match="row identity arrays"):
        critic_privileged_batch(task, dict(raw))
    a = critic_privileged_batch(task, dict(raw, **{LINEAGE_RAW_KEY: torch.zeros(2, 2)}))
    b = critic_privileged_batch(
        task, dict(raw, **{LINEAGE_RAW_KEY: torch.tensor([[1.0, 0.25], [0.0, 1.0]])})
    )
    assert a[LINEAGE_CRITIC_KEYS[0]].shape == (2, 1, 2)
    with torch.no_grad():
        assert any(not torch.equal(task(tensors, a)[k], task(tensors, b)[k]) for k in ("qol_years",))
        ca, cb = costs(tensors, a), costs(tensors, b)
        assert not torch.equal(ca["settlement_guard"], cb["settlement_guard"])
    with pytest.raises(ValueError, match="keys differ"):
        task(tensors, critic_privileged_batch(_critics(_layout())[0], raw))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        critic_privileged_batch(task, dict(raw, **{LINEAGE_RAW_KEY: torch.full((2, 2), 2.0)}))


def test_lineage_checkpoint_round_trip_resume_and_digest() -> None:
    state, layout, tensors = _lineage_payload()
    record = state["labor_model_contract"]["privilegedCritic"]
    assert record["lineage"]["features"] == ["is-daughter", "turns-since-founding-over-4"]
    assert labor_checkpoint_privileged_lineage(state) is True
    _, task, costs = _restore_lineage(state, layout)
    _, task2, costs2 = _restore_lineage(copy.deepcopy(state), layout)
    raw = dict(_raw(1), **{LINEAGE_RAW_KEY: torch.tensor([[1.0, 0.5]])})
    p = critic_privileged_batch(task, dict(raw))
    with torch.no_grad():
        for name, value in task(tensors, p).items():
            assert torch.equal(value, task2(tensors, critic_privileged_batch(task2, dict(raw)))[name])
    # Fingerprint binds the lineage token.
    plain_state, plain_layout, _ = _payload(True)
    _, plain_task, plain_costs = _restore(plain_state, plain_layout)
    assert labor_critic_digest(task, costs) == labor_critic_digest(task2, costs2)
    assert labor_critic_digest(task, costs) != labor_critic_digest(plain_task, plain_costs)
    # The constructor must match the declared contract both ways.
    with pytest.raises(ValueError, match="different lineage token"):
        _restore(state, layout)
    stripped = copy.deepcopy(plain_state)
    with pytest.raises(ValueError, match="different lineage token"):
        actor = LaborActor(plain_layout, width=16, token_width=16)
        t = LaborTaskCritic(plain_layout, width=16, token_width=16, privileged_lineage=True)
        c = LaborCostCritics(plain_layout, width=16, token_width=16, privileged_lineage=True)
        load_labor_checkpoint_models(stripped, actor, t, c, expected_layout=plain_layout)
    tampered = copy.deepcopy(state)
    tampered["labor_model_contract"]["privilegedCritic"]["lineage"]["capTurns"] = 5
    with pytest.raises(ValueError, match="lineage record is unsupported"):
        _restore_lineage(tampered, layout)


def test_models_from_checkpoint_resumes_lineage_critics() -> None:
    """The trainer's resume constructor (default widths) rebuilds the lineage critics."""
    from test_labor_privileged_bridge import _normalizer

    village = _village(1)
    normalizer = Normalizer.fit_bounded(observation_batch([village]))
    _, tensors = labor_policy_tensors([village], normalizer)
    layout = _layout(True, core=int(tensors["core"].shape[1]))
    torch.manual_seed(5)
    actor = LaborActor(layout)
    task = LaborTaskCritic(layout, privileged_lineage=True)
    costs = LaborCostCritics(layout, privileged_lineage=True)
    bind_privileged_normalizer(_normalizer(), task, costs)
    state = labor_checkpoint_payload(actor, task, costs, normalizer, layout)
    _, _, task2, costs2, _ = _models_from_checkpoint(state)
    assert task2.privileged_lineage and costs2.privileged_lineage
    assert labor_critic_digest(task, costs) == labor_critic_digest(task2, costs2)
    for key, value in task.state_dict().items():
        assert torch.equal(value, task2.state_dict()[key]), key


# --- derivation and train/serve parity ---------------------------------------------


def test_offline_derivation_rules_and_edge_case() -> None:
    worlds = np.array([1, 1, 1, 1, 1, 2, 2])
    camps = np.array([10, 10, 11, 11, 11, 10, 10])
    turns = np.array([5, 6, 7, 8, 13, 9, 10])
    values, stats = offline_lineage(worlds, camps, turns)
    assert values[:, 0].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert values[:, 1].tolist() == [0, 0.25, 0, 0.25, 1.0, 0, 0.25]
    assert stats["window_original_rows"] == 4 and stats["window_original_camps"] == 2
    assert stats["young_daughter_rows"] == 2
    # A stored founded_turn (branch data) wins over first appearance.
    founded = np.array([5, 5, 6, 6, 6, 9, 9])
    values, stats = offline_lineage(worlds, camps, turns, founded)
    assert stats["founding_source"] == "stored-founded_turn"
    assert values[2].tolist() == [1.0, 0.25]


@pytest.mark.skipif(not B01.is_dir(), reason="local fit8-b01 data not available")
def test_b01_young_daughter_rows_carry_zero_guard() -> None:
    parts = {k: [] for k in ("row_world", "row_camp", "row_turn", "reward_settlement_guard")}
    for world in sorted(p for p in B01.iterdir() if p.is_dir()):
        for key in parts:
            parts[key].append(np.load(world / f"{key}.npy").reshape(-1))
    data = {k: np.concatenate(v) for k, v in parts.items()}
    values, stats = offline_lineage(data["row_world"], data["row_camp"], data["row_turn"])
    young = (values[:, 0] > 0.5) & (np.rint(values[:, 1] * 4) <= YOUNG_DAUGHTER_ROW_AGE)
    assert young.sum() == stats["young_daughter_rows"] > 1000
    assert (data["reward_settlement_guard"][young] == 0).all()
    assert stats["negative_age_rows"] == 0
    # The window-first camps (originals) are a minority; record the edge-case size.
    assert 0 < stats["window_original_rows"] < stats["rows"]


def _synthetic_episode():
    """A fresh episode: originals at reset, daughters appear in outcome steps."""
    founders = {1: 0, 2: 0}
    births = {11: (3, 7), 13: (4, 7), 14: (5, 8)}  # turn -> (camp, parentSociety)
    steps = []
    alive = dict(founders)
    for turn in range(9, 20):
        if turn in births:
            camp, parent = births[turn]
            alive[camp] = parent
        steps.append(
            {
                "turn": turn,
                "villages": [
                    {"camp": c, "parentSociety": p, "population": 20, "terminated": False}
                    for c, p in sorted(alive.items())
                ],
            }
        )
    return steps


def test_online_and_offline_lineage_agree_on_a_replayed_episode() -> None:
    steps = _synthetic_episode()
    # Online: the rollout's own bookkeeping (labor_training: seeded at reset,
    # setdefault on every outcome step, features at each decision turn).
    founded = {int(v["camp"]): int(steps[0]["turn"]) for v in steps[0]["villages"]}
    online, rows = [], []
    for step, next_step in itertools.pairwise(steps):
        live = step["villages"]
        online.append(online_lineage(live, founded, int(step["turn"])))
        raw = rollout_privileged_raw(live, founded, int(step["turn"]), lineage=True)
        assert raw is None  # no bridge view in the synthetic rows: lineage never stands alone
        for v in live:
            rows.append((7, int(v["camp"]), int(step["turn"])))
            guard = _settlement_guard(
                {**next_step["villages"][[x["camp"] for x in next_step["villages"]].index(v["camp"])],
                 "population": 5},
                founded.get(int(v["camp"]), int(step["turn"])),
                int(next_step["turn"]),
            )
            young = v["parentSociety"] > 0 and (
                int(step["turn"]) - founded[int(v["camp"])] <= YOUNG_DAUGHTER_ROW_AGE
            )
            assert (guard == 0.0) == young  # row age <= 2 <=> guard exemption
        for v in next_step["villages"]:
            founded.setdefault(int(v["camp"]), int(next_step["turn"]))
    online = np.concatenate(online)
    w, c, t = (np.array(x) for x in zip(*rows, strict=False))
    offline, _ = offline_lineage(w, c, t)
    np.testing.assert_array_equal(online, offline)
    assert YOUNG_DAUGHTER_TURNS == YOUNG_DAUGHTER_ROW_AGE + 1


def test_rollout_raw_adds_lineage_to_the_bridge_view() -> None:
    villages = [_village(1), _village(2)]
    villages[0].update(camp=4, parentSociety=0)
    villages[1].update(camp=9, parentSociety=3)
    raw = rollout_privileged_raw(villages, {4: 9, 9: 12}, 14, lineage=True)
    assert raw[LINEAGE_RAW_KEY].tolist() == [[0.0, 1.0], [1.0, 0.5]]
    assert LINEAGE_RAW_KEY not in rollout_privileged_raw(villages, {}, 14, lineage=False)


# --- conditioning diagnostic ---------------------------------------------------------


def _toy_dataset():
    rng = np.random.default_rng(0)
    rows = []
    for world in (90100, 90101):
        for camp, start in ((1, 9), (2, 9), (3, 11), (4, 12)):
            for turn in range(start, 20):
                rows.append((world, camp, turn))
    w, c, t = (torch.tensor(x) for x in zip(*rows, strict=False))
    n = len(rows)
    core = torch.zeros(n, 16)
    core[:, 12] = torch.as_tensor(rng.uniform(0, 5, n), dtype=torch.float32)
    lineage, _ = offline_lineage(w, c, t)
    young = (lineage[:, 0] > 0.5) & (np.rint(lineage[:, 1] * 4) <= 2)
    small = torch.as_tensor(rng.uniform(0, 1, n) < 0.3)
    guard = torch.as_tensor(~young, dtype=torch.float64) * small.double()
    return {
        "row_world": w, "row_camp": c, "row_turn": t,
        "row_quarantined": torch.zeros(n, dtype=torch.bool),
        "transition_terminal": torch.zeros(n, dtype=torch.bool),
        "transition_elapsed_years": torch.full((n,), 20.0, dtype=torch.float64),
        "reward_settlement_guard": guard,
        "reward_feasibility": torch.zeros(n, dtype=torch.float64),
        "reward_qol_years": torch.as_tensor(rng.uniform(0, 20, n)),
        "reward_population_years": torch.as_tensor(rng.uniform(0, 400, n)),
        "core": core,
        "collection": {},
    }


def test_lineage_ablation_on_a_toy_critic(monkeypatch) -> None:
    dataset = _toy_dataset()

    class Toy:
        privileged_lineage = True

    def toy_predictions(task, costs, data, rows, batch_size=256, observed_max=None):
        lin = data[LINEAGE_RAW_KEY].numpy()[rows] if LINEAGE_RAW_KEY in data else None
        assert lin is not None
        young = (lin[:, 0] > 0.5) & (np.rint(lin[:, 1] * 4) <= 2)
        base = np.full(len(rows), 0.4)
        return {
            "settlement_guard": np.where(young, 0.0, base),
            "qol_years": np.full(len(rows), 50.0),
            "population_years": 100.0 + 10 * lin[:, 0],
            "feasibility": np.zeros(len(rows)),
        }

    monkeypatch.setattr(pe, "critic_predictions", toy_predictions)
    report = pe.lineage_ablation_report(Toy(), Toy(), dataset, seed=1)
    guard = report["heads"]["settlement_guard"]
    assert guard["young_daughter"]["rows"] > 0
    # Zeroing turns every young daughter into an original: |delta| = 0.4 there, 0 elsewhere.
    assert guard["young_daughter"]["mean_abs_delta_zeroed"] == pytest.approx(0.4)
    assert guard["other"]["mean_abs_delta_zeroed"] == pytest.approx(0.0)
    assert report["heads"]["qol_years"]["all"]["mean_abs_delta_zeroed"] == 0.0
    assert report["heads"]["population_years"]["all"]["mean_abs_delta_shuffled"] >= 0.0
    for split in ("all", "young_daughter", "other", "fission_event", "non_event"):
        assert split in guard
    assert "ev_lambda0.95_base" in guard["other"]
    assert dataset["collection"]["lineage"]["window_original_rows"] > 0
    assert pe.summarize_lineage_ablation(report).startswith("lineage-ablation")


def test_shuffle_preserves_world_turn_multisets_and_lambda_one_is_mc() -> None:
    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    worlds = np.array([1, 1, 1, 2, 2, 2])
    turns = np.array([5, 5, 6, 5, 5, 5])
    out = pe.shuffle_within_world_turn(values, worlds, turns, seed=3)
    for key in {(1, 5), (1, 6), (2, 5)}:
        sel = (worlds == key[0]) & (turns == key[1])
        assert sorted(map(tuple, out[sel])) == sorted(map(tuple, values[sel]))
    rewards = np.array([1.0, 2.0, 3.0])
    gamma = np.full(3, 0.5)
    mc = np.array([1 + 0.5 * (2 + 0.5 * 3), 2 + 0.5 * 3, 3.0])
    groups = [np.array([0, 1, 2])]
    lam1 = pe.lambda_returns(np.zeros(3), rewards, mc, gamma, np.zeros(3, bool), groups, 1.0)
    np.testing.assert_allclose(lam1, mc)
