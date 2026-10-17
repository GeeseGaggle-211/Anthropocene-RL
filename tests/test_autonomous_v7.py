"""Unit tests for the model9 learner (no native bridge required)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts, data
from anthropocene_rl.autonomous_v7.model import Actor, Critic, Normalizer, SetEncoder, to_tensors


def batch(n: int = 6, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    core = 188
    b: dict[str, np.ndarray] = {
        "core": rng.normal(size=(n, core)).astype(np.float32),
        "core_present": rng.random((n, core)) > 0.2,
        "movement_mask": np.ones((n, contracts.MOVEMENT_SLOTS), bool),
        "clothing_route_mask": np.ones((n, 2), bool),
        "shelter_route_mask": np.ones((n, 2), bool),
    }
    for name, rows, width in contracts.SETS:
        b[name] = rng.normal(size=(n, rows, width)).astype(np.float32)
        b[name + "_present"] = rng.random((n, rows)) > 0.5
    return b


def test_missing_rows_do_not_change_the_encoding() -> None:
    torch.manual_seed(0)
    enc = SetEncoder(5)
    x = torch.randn(2, 4, 5)
    present = torch.tensor([[True, True, False, False], [True, False, False, False]])
    a, _ = enc(x, present)
    y = x.clone(); y[~present] = 99.0
    b, _ = enc(y, present)
    assert torch.allclose(a, b)


def test_two_hot_round_trips_through_the_expectation() -> None:
    critic = Critic(10)
    u = torch.tensor([0.0, 3.0, 280.0, 1500.0])
    target = critic.two_hot(u)
    assert torch.allclose(target.sum(dim=1), torch.ones(4))
    logits = torch.log(target.clamp(min=1e-12))
    recovered = critic.expectation(logits)
    assert torch.allclose(recovered, u, rtol=0.03, atol=0.05)


def test_actor_heads_respect_masks() -> None:
    b = batch()
    b["clothing_route_mask"][:, 0] = False
    b["movement_mask"][:] = False; b["movement_mask"][:, 0] = True
    normalizer = Normalizer.fit(b)
    actor = Actor(b["core"].shape[1])
    out = actor(to_tensors(b, normalizer))
    assert torch.all(out["clothing"].argmax(dim=1) == 1)
    assert torch.all(out["movement"].argmax(dim=1) == 0)
    assert out["service_mean"].shape == (6, len(contracts.SERVICES))


def lineage_fixture() -> dict[str, np.ndarray]:
    # World 1: parent society 5 (camp 10) and a daughter society 6 (camp 11) founded at turn 2.
    return {
        "world": np.array([1, 1, 1, 1]), "camp": np.array([10, 10, 11, 11]), "turn": np.array([1, 2, 2, 3]),
        "society": np.array([5, 5, 6, 6]), "parent_society": np.array([0, 0, 5, 5]),
        "prior_population": np.array([100, 120, 30, 32]), "qol": np.array([0.8, 0.7, 0.4, 0.5], np.float32),
        "years": np.full(4, 20.0, np.float32), "terminated": np.zeros(4, bool),
        "welfare": np.array([1600.0, 1680.0, 240.0, 320.0], np.float32),
    }


def test_lineage_rewards_conserve_realized_welfare() -> None:
    d = lineage_fixture()
    reward = data.lineage_rewards(d)
    turn2 = (d["turn"] == 2)
    assert reward[turn2].sum() == pytest.approx(d["welfare"][turn2].sum())
    # Both lineage members earn at the population-weighted lineage QoL at turn 2.
    q = (120 * 0.7 + 30 * 0.4) / 150
    assert reward[1] == pytest.approx(120 * q * 20) and reward[2] == pytest.approx(30 * q * 20)


def test_lambda_returns_discount_by_physical_years() -> None:
    d = lineage_fixture()
    reward = data.lineage_rewards(d)
    g = data.lambda_returns(d, reward, value=None, lam=1.0)
    gamma = np.exp(-20 / contracts.HORIZON_YEARS)
    assert g[0] == pytest.approx(reward[0] + gamma * g[1])
    assert np.all(data.scale_free(g, d) > 0)


def test_schema_verification_rejects_layout_drift() -> None:
    schema = {"nativeContract": 7, "actionSchema": 7, "observationSchema": 7, "environmentSchema": 7,
              "movementSlots": 8, "nativeMovement": -2,
              "sets": [{"name": n, "rows": r, "width": w} for n, r, w in contracts.SETS],
              "services": list(contracts.SERVICES), "strategic": list(contracts.STRATEGIC),
              "coreFeatures": ["a", "b"], "coreWidth": 2, "buildHash": "x"}
    assert contracts.verify_schema(schema).core_width == 2
    schema["sets"][0]["width"] = 13
    with pytest.raises(ValueError):
        contracts.verify_schema(schema)


def test_privileged_inputs_never_reach_the_actor() -> None:
    b = batch()
    b["privileged"] = np.random.default_rng(3).normal(size=(6, 10)).astype(np.float32)
    normalizer = Normalizer.fit(b)
    torch.manual_seed(1)
    actor, critic = Actor(b["core"].shape[1]), Critic(b["core"].shape[1], privileged_width=10)
    t = to_tensors(b, normalizer)
    a1, c1 = actor(t), critic(t)
    t["privileged"] = t["privileged"] + 5.0
    a2, c2 = actor(t), critic(t)
    assert all(torch.equal(a1[k], a2[k]) for k in a1)
    assert not torch.allclose(c1, c2)
