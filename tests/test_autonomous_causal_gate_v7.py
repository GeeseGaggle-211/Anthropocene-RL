import numpy as np
import pytest

from anthropocene_rl.autonomous_v5.head_influence import (
    CONTROL_MAX,
    CONTROL_MIN,
    shares,
)


def test_mean_reachability_is_distinct_from_native_endpoints():
    mask = np.ones(13, bool)
    endpoint = shares(mask, 1)
    representable = shares(mask, 1, representable=True)
    assert endpoint[1] == 1 and 0.67 < representable[1] < 0.68
    assert representable.sum() == pytest.approx(1)
    assert 0.038 < CONTROL_MIN < 0.039 and CONTROL_MAX < 1
    # A mean/control strictly below1 rounds down the native opening-herd target.
    assert int(np.floor(10 * CONTROL_MAX)) == 9
    mask[1] = False
    with pytest.raises(ValueError, match="unavailable"):
        shares(mask, 1)
    assert shares(np.zeros(15, bool)).sum() == 0


def test_physical_nstep_tail_and_terminal_are_not_confused():
    from anthropocene_rl.autonomous_v5.causal import n_step_targets

    for horizon in [100, 400]:
        r = n_step_targets([20.0] * 25, [20.0] * 25, [False] * 25, horizon, horizon, 1)
        assert r["target"][0] == pytest.approx(horizon)
        assert r["tail_weight"][0] == pytest.approx(np.exp(-500 / horizon))
        assert r["zero_tail"][0] == pytest.approx(horizon * (1 - np.exp(-500 / horizon)))
        terminated = n_step_targets(
            [20.0] * 25, [20.0] * 25, [False] * 24 + [True], horizon, horizon, 1
        )
        assert terminated["tail_weight"].sum() == 0
        assert terminated["target"][0] == pytest.approx(r["zero_tail"][0])


def test_receipt_ablation_preserves_native_memory_ema_masks_and_stocks():
    import torch
    from test_autonomous_history_v5 import wire

    from anthropocene_rl.autonomous_v5.contracts import observation
    from anthropocene_rl.autonomous_v5.policy import inputs_with_mode

    w = wire()
    w["Context"][84] = 1.0
    w["History"][-1] = 500
    obs = observation(w)
    full = inputs_with_mode([obs])
    ablated = inputs_with_mode([obs], "receipt_none")
    assert torch.equal(full["x"][:, :761], ablated["x"][:, :761])
    assert full["x"][:, 761:].abs().sum() > 0 and ablated["x"][:, 761:].sum() == 0
    assert obs["History"][-1] == 500 and obs["Context"][84] == 1
    assert all(torch.equal(full[k], ablated[k]) for k in full if k != "x")


def test_critic_only_calibration_preserves_actor_adam_rng_counters_and_anchors():
    import torch
    from test_autonomous_history_v5 import config, v5, wire

    from anthropocene_rl.autonomous_v5.causal import fit_critic, protected_state
    from anthropocene_rl.autonomous_v5.contracts import observation
    from anthropocene_rl.autonomous_v5.policy import inputs
    from anthropocene_rl.training.autonomous_ppo_v5 import AutonomousTrainerV5

    torch.set_num_threads(1)
    trainer = AutonomousTrainerV5(v5(config()), 31, {"test": 7}, width=16)
    x = inputs([observation(wire()) for _ in range(32)])
    x["x"][:, 0] = torch.linspace(-1, 1, 32)
    # Exercise preservation with populated Adam slots, not an empty optimizer.
    z = trainer.model.actor(x["x"])
    loss = z.square().mean() + sum(h(z).square().mean() for h in trainer.model.heads.values())
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    y = trainer.model.value(x).detach() + torch.linspace(-1, 1, 32)
    before = protected_state(trainer)
    result = fit_critic(trainer, (x, y), (x, y), mode="full", max_epochs=32)
    assert result["protected_state_before"] == result["protected_state_after"] == before
    assert result["optimizer_steps"] == 32 and result["final"]["mse"] <= result["entry"]["mse"]


def test_low_variance_reporting_floor_cannot_pass_a_worse_critic():
    from anthropocene_rl.autonomous_v5.causal import critic_ready

    value = dict(mse=0.8, normalized_mse=0.8, normalized_bias=0.0)
    assert not critic_ready(value, dict(mse=1.0), 0.1)
    value["mse"] = 0.09
    assert critic_ready(value, dict(mse=1.0), 0.1)
    assert not critic_ready(value, dict(mse=1.0), 0.0)
