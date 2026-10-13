"""Full-action density, support, gradient and checkpoint tests (no training campaign)."""

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v4.contracts import FEATURE_WIDTH
from anthropocene_rl.autonomous_v4.policy import AutonomousPPO, categorical, to_native
from anthropocene_rl.autonomous_v4.training import Trainer


def batch(n=6):
    torch.manual_seed(42)
    result = dict(
        x=torch.randn(n, FEATURE_WIDTH),
        WorkforceMask=torch.ones(n, 13, dtype=torch.bool),
        ActivityMask=torch.zeros(n, 13, 15, dtype=torch.bool),
        ControlMask=torch.ones(n, 3, dtype=torch.bool),
        MovementMask=torch.ones(n, 8, dtype=torch.bool),
        CounterpartyMask=torch.ones(n, 8, dtype=torch.bool),
        GoodMask=torch.ones(n, 11, dtype=torch.bool),
        Stocks=torch.ones(n, 11) * 100,
    )
    result["ActivityMask"][:, 0, :4] = True
    result["ActivityMask"][:, 4, 4:8] = True
    return result


def test_categorical_density_mask_and_degenerate_rng():
    logits = torch.tensor([[1.0, 2.0, 9.0], [4.0, 5.0, 6.0]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.bool)
    generator = torch.Generator().manual_seed(91)
    _choice, lp, entropy = categorical(logits, mask, torch.tensor([0, 1]), generator)
    assert lp[0].item() == pytest.approx(torch.log_softmax(logits[0, :2], 0)[0].item())
    assert lp[1] == entropy[1] == 0
    (-lp.sum()).backward()
    assert logits.grad[0, 2] == 0 and torch.all(logits.grad[1] == 0)
    before = generator.get_state()
    categorical(logits[1:], mask[1:], generator=generator)
    assert torch.equal(before, generator.get_state())
    with pytest.raises(ValueError):
        categorical(logits, mask, torch.tensor([2, 1]))


def test_full_action_density_frozen_support_and_active_head_gradients():
    torch.set_num_threads(1)
    model, obs = AutonomousPPO(32), batch(12)
    generator = torch.Generator().manual_seed(102)
    with torch.no_grad():
        action, lp, _, _ = model(obs, generator=generator)
    _, scored, _, _ = model(obs, action)
    assert torch.equal(lp, scored)
    assert torch.isfinite(scored).all()
    enabled = action.trade[:, 0].bool()
    assert enabled.any() and torch.all(action.trade[~enabled] == 0)
    assert torch.all(action.trade[enabled, 2] != action.trade[enabled, 3])
    (-scored.mean()).backward()
    for head in model.heads.values():
        assert head.weight.grad.abs().sum() > 0
    bad = copy.deepcopy(action)
    bad.trade[~enabled, 2] = 1
    with pytest.raises(ValueError):
        model(obs, bad)
    # Likelihood responds to requested fractions even when staffing rounds identically.
    changed = copy.deepcopy(action)
    changed.workforce[:, 0] += 0.0001
    changed.workforce[:, 1] -= 0.0001
    assert not torch.equal(model(obs, changed)[1], lp)


def test_disabled_trade_movement_and_controls_have_zero_gradient():
    model, obs = AutonomousPPO(32), batch(2)
    obs["CounterpartyMask"][:] = False
    obs["ControlMask"][:] = False
    obs["MovementMask"][:] = False
    obs["MovementMask"][:, 0] = True
    action, lp, _, _ = model(obs)
    assert torch.all(action.trade == 0) and torch.all(action.movement == 0)
    assert torch.all(action.controls == 0)
    (-lp.mean()).backward()
    for key in ["enabled", "partner", "give", "want", "lot", "price", "movement"]:
        assert model.heads[key].weight.grad.abs().sum() == 0


def test_quote_transform_bounds_and_ratios():
    obs = {"Stocks": np.array([1e20] * 11)}
    for lot in range(3):
        for price, ratio in enumerate([0.5, 1, 2]):
            quote = to_native({"trade": np.array([1, 0, 0, 1, lot, price])}, obs)["quote"]
            assert 1e-6 <= quote["GiveQuantity"] <= 1e6
            assert 1e-6 <= quote["WantQuantity"] <= 1e6
            assert quote["WantQuantity"] / quote["GiveQuantity"] == pytest.approx(ratio)


def test_checkpoint_reproduces_next_actions_and_adam_update(tmp_path):
    config = json.loads(Path("configs/training/autonomous-ppo-gate-v1.json").read_text())
    config.update(epochs=2, minibatch=6)
    trainer = Trainer(config, 31, {"source": "test"}, width=32)
    obs = batch()

    def update(t):
        with torch.no_grad():
            a, lp, _, _ = t.model(obs, generator=t.sampler)
        rows = [
            dict(
                obs={k: v[i : i + 1] for k, v in obs.items()},
                action=a.index(slice(i, i + 1)),
                logp=lp[i : i + 1],
                advantage=i - 2.0,
                **{"return": float(i)},
            )
            for i in range(6)
        ]
        t.update(rows)
        return a, lp

    update(trainer)  # nonempty Adam moments
    trainer.checkpoint(tmp_path / "state.pt")
    restored = Trainer(config, 999, {"source": "test"}, width=32)
    restored.restore(tmp_path / "state.pt")
    a, lp = update(trainer)
    b, lq = update(restored)
    assert torch.equal(a.workforce, b.workforce) and torch.equal(a.trade, b.trade)
    assert torch.equal(lp, lq)
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v, restored.model.state_dict()[k])
    with pytest.raises(ValueError):
        Trainer(config, 31, {"source": "wrong"}, width=32).restore(tmp_path / "state.pt")


@pytest.mark.skipif(
    not os.environ.get("AUTONOMOUS_V4_BRIDGE"), reason="native v4 integration opt-in"
)
def test_native_full_action_replay_and_gym():
    from anthropocene_rl.autonomous_v4.env import AutonomousElderEnv, AutonomousTransport
    from anthropocene_rl.autonomous_v4.policy import inputs
    from anthropocene_rl.autonomous_v4.training import market_peer

    bank = Path(os.environ["AUTONOMOUS_V4_BANK"])
    entry = json.loads((bank / "manifest.json").read_text())["entries"][1]
    env = AutonomousElderEnv(
        AutonomousTransport(json.loads(os.environ["AUTONOMOUS_V4_BRIDGE"])),
        (bank / entry["snapshot"]).read_bytes(),
        entry["focal"],
        {entry["peer"]: market_peer},
        decision_limit=8,
    )
    try:
        model = AutonomousPPO(32)
        _, info = env.reset()
        for _ in range(8):
            with torch.no_grad():
                a, _, _, _ = model(inputs([info["native"]]))
            assert env.action_space.contains(a.wire_action())
            _, reward, terminal, truncated, info = env.step(a.wire_action())
            assert np.isfinite(reward)
            if terminal or truncated:
                break
        assert env.transport.verify_replay()["steps"] > 0
    finally:
        env.close()


def test_market_peer_microscopic_stock_quote_is_legal():
    from anthropocene_rl.autonomous_v4.training import market_peer

    obs = {k: v[0].numpy() for k, v in batch(1).items() if k != "x"}
    obs["Stocks"][3] = 1.5e-6
    quote = market_peer(obs)["quote"]
    assert 1e-6 <= quote["GiveQuantity"] <= obs["Stocks"][3]
    assert quote["WantQuantity"] >= 1e-6


def test_inference_plugin_checks_provenance_and_isolates_observations(tmp_path):
    from anthropocene_rl.autonomous_v4.inference import AutonomousElderPolicy

    config = json.loads(Path("configs/training/autonomous-ppo-gate-v1.json").read_text())
    trainer = Trainer(config, 17, {"source": "inference-test"}, width=32)
    trainer.checkpoint(tmp_path / "policy.pt")
    policy = AutonomousElderPolicy(
        tmp_path / "policy.pt", expected_provenance={"source": "inference-test"}, seed=71
    )
    obs = {k: v[0].tolist() for k, v in batch(1).items() if k != "x"}
    obs.update(SchemaVersion=4, Context=[0.0] * 512)
    other = copy.deepcopy(obs)
    other["Context"] = [100.0] * 512
    other["Stocks"] = [1e6] * 11
    alone = policy.act(obs, deterministic=True)
    together = policy.act_many({2: other, 1: obs}, deterministic=True)[1]
    assert np.allclose(alone["workforce"], together["workforce"], atol=1e-7, rtol=0)
    assert alone["movement"] == together["movement"]
    with pytest.raises(ValueError):
        AutonomousElderPolicy(tmp_path / "policy.pt", expected_provenance={}, seed=71)
    obs["SchemaVersion"] = 3
    with pytest.raises(ValueError):
        policy.act(obs)
