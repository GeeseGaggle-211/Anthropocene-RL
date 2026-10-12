import math

import pytest

pytest.importorskip("torch")

import torch

from anthropocene_rl.policies.elder_mlp import ElderActorCritic, ElderActorCriticConfig
from anthropocene_rl.training.pilot import (
    _critic_evaluation_diagnostics,
    _critic_health_window,
    _inactive_head_gradient_audit,
)
from anthropocene_rl.training.ppo import PPOConfig, PPOTrainer, RolloutBatch
from anthropocene_rl.training.smoke import run_ppo_smoke


def test_ppo_mechanics_smoke_is_finite() -> None:
    metrics = run_ppo_smoke(batch_size=32)
    assert metrics
    assert all(math.isfinite(value) for value in metrics.values())


def test_ppo_keeps_structurally_inactive_continuous_heads_gradient_inert() -> None:
    torch.manual_seed(37)
    count = 32
    observation = torch.randn(count, 512) * 0.1
    observation[:, 80:84] = 0.0
    observation[:, 80] = 1.0  # field retention
    observation[:, 82] = 1.0  # field development
    observation[:, 482:490] = 0.0
    observation[:, 482] = 1.0  # explicit stay
    model = ElderActorCritic()
    with torch.no_grad():
        sampled = model.actor.sample(observation)
        baseline = model.critic(observation)
    advantage = torch.linspace(-1.0, 1.0, count)
    batch = RolloutBatch(
        observations=observation,
        continuous_actions=sampled.continuous.detach(),
        movement_indices=sampled.movement_index.detach(),
        old_log_prob=sampled.log_prob.detach(),
        advantages=advantage,
        returns=(baseline + advantage).detach(),
    )

    trainer = PPOTrainer(
        model,
        PPOConfig(
            epochs=2,
            minibatch_size=16,
            entropy_coefficient=0.0,
            target_kl=1e-12,
        ),
    )
    metrics = trainer.update(batch)

    assert metrics["continuous_head_0_grad_max"] > 0.0
    assert metrics["continuous_head_2_grad_max"] > 0.0
    assert metrics["continuous_head_1_grad_max"] == 0.0
    assert metrics["continuous_head_3_grad_max"] == 0.0
    assert metrics["movement_head_grad_max"] == 0.0
    assert metrics["optimization_epochs_completed"] == 2.0
    assert metrics["optimization_actor_epochs_completed"] == 1.0
    assert metrics["target_kl_early_stop"] == 1.0
    assert metrics["actor_pre_clip_grad_norm_max"] > 0.0
    assert metrics["critic_pre_clip_grad_norm_max"] > 0.0
    assert metrics["total_pre_clip_grad_norm_max"] > 0.0
    assert metrics["paired_actor_pre_clip_grad_norm_mean"] > 0.0
    assert metrics["paired_critic_pre_clip_grad_norm_mean"] > 0.0
    assert metrics["paired_total_pre_clip_grad_norm_mean"] > 0.0
    assert 0.0 < metrics["joint_clip_factor_min"] <= 1.0
    assert metrics["joint_clip_factor_min"] <= metrics["joint_clip_factor_mean"] <= 1.0
    assert metrics["actor_clip_factor_min"] == metrics["joint_clip_factor_min"]
    assert metrics["critic_clip_factor_min"] == metrics["joint_clip_factor_min"]
    assert metrics["actor_parameter_update_l2"] > 0.0
    assert metrics["actor_backbone_parameter_update_l2"] > 0.0
    assert metrics["actor_continuous_head_parameter_update_l2"] > 0.0
    assert metrics["critic_parameter_update_l2"] > 0.0
    assert metrics["critic_backbone_parameter_update_l2"] > 0.0
    assert metrics["critic_value_head_weight_l2"] > 0.0
    assert metrics["critic_value_head_bias_abs"] >= 0.0
    assert math.isfinite(metrics["post_update_critic_explained_variance"])
    assert math.isfinite(metrics["post_update_critic_normalized_bias"])
    for head in range(4):
        assert math.isfinite(metrics[f"continuous_head_{head}_log_std"])
        endpoint_rate = metrics[f"continuous_head_{head}_mean_endpoint_rate"]
        assert 0.0 <= endpoint_rate <= 1.0

    with torch.no_grad():
        evaluated = model.actor.evaluate_actions(
            batch.observations,
            batch.continuous_actions,
            batch.movement_indices,
        )
        log_ratio = evaluated.log_prob - batch.old_log_prob
        ratio = torch.exp(log_ratio)
        expected_kl = ((ratio - 1.0) - log_ratio).mean()
        expected_clip_fraction = (torch.abs(ratio - 1.0) > trainer.config.clip_ratio).float().mean()
        expected_value_mse = torch.square(model.critic(batch.observations) - batch.returns).mean()

    assert metrics["post_update_approx_kl"] == pytest.approx(float(expected_kl))
    assert metrics["post_update_clip_fraction"] == pytest.approx(float(expected_clip_fraction))
    assert metrics["post_update_value_mse"] == pytest.approx(float(expected_value_mse))


def test_ppo_can_clip_actor_and_critic_separately_for_shadow_audits() -> None:
    torch.manual_seed(41)
    count = 32
    observation = torch.randn(count, 512) * 0.2
    observation[:, 80:84] = 0.0
    observation[:, 80] = 1.0
    observation[:, 82] = 1.0
    observation[:, 482:490] = 0.0
    observation[:, 482] = 1.0
    model = ElderActorCritic()
    with torch.no_grad():
        sampled = model.actor.sample(observation)
        value = model.critic(observation)
    advantage = torch.linspace(-2.0, 2.0, count)
    batch = RolloutBatch(
        observations=observation,
        continuous_actions=sampled.continuous,
        movement_indices=sampled.movement_index,
        old_log_prob=sampled.log_prob,
        advantages=advantage,
        returns=value + 4.0 * advantage,
    )
    trainer = PPOTrainer(
        model,
        PPOConfig(
            epochs=1,
            minibatch_size=16,
            entropy_coefficient=0.0,
            target_kl=None,
            gradient_clip_mode="separate",
        ),
    )
    metrics = trainer.update(batch)

    assert 0.0 < metrics["actor_clip_factor_min"] <= 1.0
    assert 0.0 < metrics["critic_clip_factor_min"] <= 1.0
    assert 0.0 < metrics["joint_clip_factor_min"] <= 1.0
    assert metrics["actor_parameter_update_l2"] > 0.0
    assert metrics["critic_parameter_update_l2"] > 0.0


def test_popart_value_normalization_preserves_raw_predictions_and_round_trips() -> None:
    torch.manual_seed(53)
    observation = torch.randn(32, 512) * 0.2
    model = ElderActorCritic()
    trainer = PPOTrainer(
        model,
        PPOConfig(value_target_normalization=True),
    )
    with torch.no_grad():
        before = trainer.value(observation).clone()
    trainer._update_value_normalization(torch.linspace(40.0, 160.0, 32))
    with torch.no_grad():
        after = trainer.value(observation)
    assert torch.allclose(before, after, atol=2e-5, rtol=2e-5)
    assert trainer.value_count == 32.0
    assert trainer.value_std > 1.0

    clone = ElderActorCritic()
    clone.load_state_dict(model.state_dict())
    restored = PPOTrainer(
        clone,
        PPOConfig(value_target_normalization=True),
    )
    restored.load_state_dict(trainer.state_dict())
    with torch.no_grad():
        assert torch.equal(trainer.value(observation), restored.value(observation))


def test_per_state_inactive_head_gradient_audit_handles_broad_mixed_batch() -> None:
    torch.manual_seed(59)
    observation = torch.randn(8, 512) * 0.1
    observation[:, 80:84] = 0.0
    observation[:, 482:490] = 0.0
    observation[:, 482] = 1.0
    observation[2:, 80:84] = 1.0
    observation[4:, 483] = 1.0
    model = ElderActorCritic()
    with torch.no_grad():
        sampled = model.actor.sample(observation)
    batch = RolloutBatch(
        observations=observation,
        continuous_actions=sampled.continuous,
        movement_indices=sampled.movement_index,
        old_log_prob=sampled.log_prob,
        advantages=torch.zeros(8),
        returns=torch.zeros(8),
    )

    audit = _inactive_head_gradient_audit(model, batch)

    assert audit["max_abs_gradient"] == 0.0
    assert audit["per_head"]["continuous_0"]["inactive_rows"] == 2
    assert audit["per_head"]["movement"]["inactive_rows"] == 4


def test_hidden_activation_diagnostics_are_finite_and_layer_specific() -> None:
    torch.manual_seed(71)
    model = ElderActorCritic()
    diagnostics = model.hidden_activation_diagnostics(torch.randn(12, 512) * 0.1)

    assert diagnostics["observation_count"] == 12
    for network in ("actor", "critic"):
        assert diagnostics[network]["activation"] == "tanh"
        assert len(diagnostics[network]["layers"]) == 2
        for layer in diagnostics[network]["layers"]:
            assert math.isfinite(layer["preactivation"]["max_abs"])
            rate = layer["activation"]["abs_at_or_above_0_95_rate"]
            assert 0.0 <= rate <= 1.0
            derivative = layer["activation"]["mean_tanh_derivative"]
            assert 0.0 <= derivative <= 1.0


def test_zero_unseen_input_columns_neutralizes_both_networks() -> None:
    torch.manual_seed(73)
    model = ElderActorCritic()
    actor_first = model.actor.backbone.net[0]
    critic_first = model.critic.backbone.net[0]
    actor_control = actor_first.weight[:, 6].detach().clone()
    critic_control = critic_first.weight[:, 6].detach().clone()

    assert model.zero_input_columns((7, 200, 7)) == (7, 200)
    assert torch.count_nonzero(actor_first.weight[:, (7, 200)]) == 0
    assert torch.count_nonzero(critic_first.weight[:, (7, 200)]) == 0
    assert torch.equal(actor_first.weight[:, 6], actor_control)
    assert torch.equal(critic_first.weight[:, 6], critic_control)


def test_development_critic_diagnostic_uses_realized_horizon_returns() -> None:
    perfect = _critic_evaluation_diagnostics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert perfect["evaluable"] is True
    assert perfect["explained_variance"] == pytest.approx(1.0)
    assert perfect["normalized_mse"] == pytest.approx(0.0)
    assert perfect["mean_bias"] == pytest.approx(0.0)

    singleton = _critic_evaluation_diagnostics([1.0], [1.0])
    assert singleton["evaluable"] is False
    assert singleton["explained_variance"] is None


def test_critic_layer_norm_is_independent_and_diagnostic_safe() -> None:
    torch.manual_seed(89)
    baseline = ElderActorCritic(ElderActorCriticConfig(initial_log_std=-1.0))
    torch.manual_seed(89)
    normalized = ElderActorCritic(
        ElderActorCriticConfig(
            initial_log_std=-1.0,
            critic_layer_norm=True,
        )
    )

    baseline_actor = baseline.actor.state_dict()
    normalized_actor = normalized.actor.state_dict()
    assert baseline_actor.keys() == normalized_actor.keys()
    for key in baseline_actor:
        assert torch.equal(baseline_actor[key], normalized_actor[key])

    assert not any(isinstance(module, torch.nn.LayerNorm) for module in normalized.actor.modules())
    assert sum(
        isinstance(module, torch.nn.LayerNorm) for module in normalized.critic.modules()
    ) == 2

    diagnostics = normalized.hidden_activation_diagnostics(torch.randn(12, 512) * 3.0)
    assert diagnostics["actor"]["layer_norm"] is False
    assert diagnostics["critic"]["layer_norm"] is True
    for layer in diagnostics["critic"]["layers"]:
        assert math.isfinite(layer["linear_output"]["max_abs"])
        assert math.isfinite(layer["preactivation"]["max_abs"])


def test_legacy_actor_critic_config_defaults_to_no_layer_norm() -> None:
    config = ElderActorCriticConfig(
        **{"activation": "tanh", "initial_log_std": -1.0}
    )
    assert config.actor_layer_norm is False
    assert config.critic_layer_norm is False


def test_critic_health_window_requires_sustained_pre_update_fit() -> None:
    healthy = [
        {
            "update": update,
            "ppo": {
                "pre_update_critic_explained_variance": 0.4,
                "pre_update_critic_normalized_mse": 0.6,
                "pre_update_critic_normalized_bias": 0.1,
            },
        }
        for update in range(1, 9)
    ]
    assert _critic_health_window(healthy)["eligible"] is True

    unhealthy = [dict(row) for row in healthy]
    unhealthy[-1] = {
        "update": 8,
        "ppo": {
            "pre_update_critic_explained_variance": -0.1,
            "pre_update_critic_normalized_mse": 1.2,
            "pre_update_critic_normalized_bias": 0.9,
        },
    }
    assert _critic_health_window(unhealthy[:7])["eligible"] is False
    assert _critic_health_window(unhealthy)["eligible"] is True

    for index in range(3):
        unhealthy[index] = unhealthy[-1] | {"update": index + 1}
    assert _critic_health_window(unhealthy)["eligible"] is False
