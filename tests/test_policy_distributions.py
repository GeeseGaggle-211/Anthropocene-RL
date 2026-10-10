import pytest

torch = pytest.importorskip("torch")

from anthropocene_rl.policies.elder_mlp import ElderActor  # noqa: E402


def test_inactive_herd_and_split_heads_do_not_change_log_probability() -> None:
    observation = torch.zeros(1, 512)
    observation[:, 80] = 1.0
    observation[:, 82] = 1.0
    observation[:, 482] = 1.0
    actor = ElderActor()
    sampled = actor.sample(observation, deterministic=True)
    assert sampled.continuous[0, 1].item() == 0.0
    assert sampled.continuous[0, 3].item() == 1.0
    assert sampled.movement_index.item() == 0

    altered = sampled.continuous.clone()
    altered[:, 1] = 0.97
    altered[:, 3] = 0.03
    original_eval = actor.evaluate_actions(
        observation,
        sampled.continuous,
        sampled.movement_index,
    )
    altered_eval = actor.evaluate_actions(
        observation,
        altered,
        sampled.movement_index,
    )
    assert torch.allclose(original_eval.log_prob, altered_eval.log_prob, atol=1e-6)


def test_mixed_development_activates_split_head() -> None:
    observation = torch.zeros(1, 512)
    observation[:, 80:84] = 1.0
    observation[:, 482] = 1.0
    actor = ElderActor()
    sampled = actor.sample(observation, deterministic=True)
    altered = sampled.continuous.clone()
    altered[:, 3] = 0.9
    original_eval = actor.evaluate_actions(
        observation,
        sampled.continuous,
        sampled.movement_index,
    )
    altered_eval = actor.evaluate_actions(
        observation,
        altered,
        sampled.movement_index,
    )
    assert not torch.allclose(original_eval.log_prob, altered_eval.log_prob)
