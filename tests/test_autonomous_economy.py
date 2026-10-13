import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.distributions import Dirichlet

from anthropocene_rl.autonomous.contracts import FEATURE_WIDTH
from anthropocene_rl.autonomous.env import AutonomousElderEnv, AutonomousTransport
from anthropocene_rl.autonomous.policy import (
    AutonomousActorCritic,
    MaskedDirichlet,
    ppo_mechanics_update,
)


def test_masked_density_and_degenerate_faces():
    alpha = torch.tensor([[2.0, 88.0, 3.0], [99.0, 2.0, 99.0], [3.0, 4.0, 5.0]], requires_grad=True)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=torch.bool)
    dist = MaskedDirichlet(alpha, mask)
    action = dist.sample(generator=torch.Generator().manual_seed(42))
    assert torch.all(action[~mask] == 0)
    expected = Dirichlet(alpha[0, [0, 2]].double()).log_prob(action[0, [0, 2]])
    assert torch.allclose(dist.log_prob(action)[0], expected)
    assert dist.log_prob(action)[1:].tolist() == [0, 0]
    assert dist.entropy()[1:].tolist() == [0, 0]
    (-dist.log_prob(action).sum()).backward()
    assert torch.all(alpha.grad[~mask] == 0)
    bad = action.clone()
    bad[0, 1] = 0.1
    with pytest.raises(ValueError):
        dist.log_prob(bad)


def test_absent_coordinates_do_not_draw_rng():
    ga = torch.Generator().manual_seed(100)
    gb = torch.Generator().manual_seed(100)
    a = MaskedDirichlet(torch.tensor([2.0, 3.0]), torch.tensor([True, True])).sample(generator=ga)
    b = MaskedDirichlet(torch.tensor([2.0, 19.0, 3.0]), torch.tensor([True, False, True])).sample(
        generator=gb
    )
    assert torch.equal(a, b[[0, 2]])
    MaskedDirichlet(torch.ones(3), torch.tensor([False, True, False])).sample(generator=ga)
    assert torch.equal(ga.get_state(), gb.get_state())


def test_ppo_requested_likelihood_and_update():
    torch.manual_seed(2)
    model = AutonomousActorCritic(32)
    x = torch.randn(8, FEATURE_WIDTH)
    wf = torch.ones(8, 13, dtype=torch.bool)
    wf[:, 3] = False
    act = torch.zeros(8, 13, 15, dtype=torch.bool)
    act[:, 0, :4] = True
    act[:, 4, 4:7] = True
    controls = torch.tensor([[1, 0, 0]] * 8, dtype=torch.bool)
    actions, logp, entropy, _ = model.sample(x, wf, act, controls)
    scored, scored_entropy, _ = model.evaluate(x, wf, act, controls, actions)
    assert torch.equal(logp, scored) and torch.equal(entropy, scored_entropy)
    assert torch.isfinite(logp).all()
    assert torch.all(actions.controls[:, 1:] == 0)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = ppo_mechanics_update(
        model,
        optimizer,
        x,
        wf,
        act,
        controls,
        actions,
        logp,
        torch.linspace(-1, 1, 8),
        torch.linspace(0, 1, 8),
    )
    assert result["approx_kl"] == pytest.approx(0, abs=1e-12)
    assert result["grad_norm"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    restored = AutonomousActorCritic(32)
    restored.load_state_dict(model.state_dict())
    assert torch.equal(
        model.evaluate(x, wf, act, controls, actions)[0],
        restored.evaluate(x, wf, act, controls, actions)[0],
    )


@pytest.mark.skipif(
    not os.environ.get("AUTONOMOUS_BRIDGE"), reason="explicit native v3 bridge required"
)
def test_native_wego_gym_and_ppo_mechanics():
    import json

    command = json.loads(os.environ["AUTONOMOUS_BRIDGE"])
    snapshot = Path(os.environ["AUTONOMOUS_FIXTURE"]).read_bytes()
    ids = [int(x) for x in os.environ["AUTONOMOUS_CAMPS"].split(",")]
    transport = AutonomousTransport(command)

    def peer(obs):
        workforce = obs["WorkforceMask"].astype(float)
        workforce /= workforce.sum()
        activities = obs["ActivityMask"].astype(float)
        activities /= np.maximum(1, activities.sum(-1, keepdims=True))
        return {"workforce": workforce, "activities": activities, "controls": np.zeros(3)}

    env = AutonomousElderEnv(transport, snapshot, ids[0], {ids[1]: peer}, decision_limit=3)
    try:
        x, info = env.reset(seed=10)
        assert env.observation_space.contains(x)
        model = AutonomousActorCritic(32)
        rows, masks, amasks, cmasks, samples, logps, rewards = [], [], [], [], [], [], []
        for _ in range(3):
            native = info["native"]
            inputs = (
                torch.tensor(x[None]),
                torch.tensor(native["WorkforceMask"][None]),
                torch.tensor(native["ActivityMask"][None]),
                torch.tensor(native["ControlMask"][None]),
            )
            action, logp, _, _ = model.sample(*inputs)
            assert env.action_space.contains(action.wire_action())
            rows.append(inputs[0])
            masks.append(inputs[1])
            amasks.append(inputs[2])
            cmasks.append(inputs[3])
            samples.append(action)
            logps.append(logp.detach())
            x, reward, terminated, truncated, info = env.step(action.wire_action())
            rewards.append(reward)
            assert info["elapsed_years"] > 0 and np.isfinite(reward)
            if terminated or truncated:
                break
        assert transport.verify_replay()["steps"] == len(samples)
        from anthropocene_rl.autonomous.policy import ActionBatch

        actions = ActionBatch(
            *(
                torch.cat([getattr(a, key) for a in samples])
                for key in ("workforce", "activities", "controls")
            )
        )
        result = ppo_mechanics_update(
            model,
            torch.optim.Adam(model.parameters(), lr=1e-4),
            torch.cat(rows),
            torch.cat(masks),
            torch.cat(amasks),
            torch.cat(cmasks),
            actions,
            torch.cat(logps),
            torch.linspace(-1, 1, len(samples)),
            torch.tensor(rewards).float() / 1000,
        )
        assert np.isfinite(result["loss"])
        # Reset is a fresh native boundary, with the same observation features.
        reset, _ = env.reset(seed=10)
        assert np.array_equal(reset, rows[0].numpy()[0])
    finally:
        env.close()


def test_requested_density_is_not_rounded_staffing_density():
    dist = MaskedDirichlet(torch.tensor([2.0, 4.0]), torch.tensor([True, True]))
    a, b = (
        torch.tensor([0.51, 0.49], dtype=torch.float64),
        torch.tensor([0.52, 0.48], dtype=torch.float64),
    )
    # Both requests staff one of each role with two residents. PPO still sees
    # distinct continuous densities at the actual requested simplex points.
    assert not torch.isclose(dist.log_prob(a), dist.log_prob(b))
