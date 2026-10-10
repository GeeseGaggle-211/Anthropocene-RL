import pytest

torch = pytest.importorskip("torch")

from anthropocene_rl.policies.distributions import (  # noqa: E402
    evaluate_hybrid,
    sample_hybrid,
)
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


def test_sampled_bounded_action_log_probability_round_trips_at_saturation() -> None:
    torch.manual_seed(7)
    count = 1024
    observation = torch.zeros(count, 512)
    observation[:, 80:84] = 1.0
    observation[:, 482] = 1.0
    mean = torch.tensor([5.0, -5.0, 0.0, 2.5]).expand(count, -1)
    log_std = torch.full((4,), 0.5)
    logits = torch.zeros(count, 8)

    sampled = sample_hybrid(mean, log_std, logits, observation)
    evaluated = evaluate_hybrid(
        mean,
        log_std,
        logits,
        observation,
        sampled.continuous,
        sampled.movement_index,
    )

    assert torch.all(sampled.continuous > 0.0)
    assert torch.all(sampled.continuous < 1.0)
    assert torch.all(torch.isfinite(sampled.log_prob))
    assert torch.allclose(sampled.log_prob, evaluated.log_prob, atol=1e-5, rtol=1e-5)


def test_gaussian_parameter_envelope_rejects_saturation_prone_policy() -> None:
    observation = torch.zeros(1, 512)
    observation[:, 80:84] = 1.0
    observation[:, 482] = 1.0
    with pytest.raises(ValueError, match="log_std"):
        sample_hybrid(
            torch.zeros(1, 4),
            torch.full((4,), 2.0),
            torch.zeros(1, 8),
            observation,
        )


def test_entropy_depends_on_current_policy_not_scored_action() -> None:
    observation = torch.zeros(1, 512)
    observation[:, 80:84] = 1.0
    observation[:, 482] = 1.0
    mean = torch.zeros(1, 4)
    log_std = torch.zeros(4)
    logits = torch.zeros(1, 8)
    movement = torch.zeros(1, dtype=torch.long)

    low = evaluate_hybrid(
        mean,
        log_std,
        logits,
        observation,
        torch.full((1, 4), 0.2),
        movement,
    )
    high = evaluate_hybrid(
        mean,
        log_std,
        logits,
        observation,
        torch.full((1, 4), 0.65),
        movement,
    )
    assert not torch.allclose(low.log_prob, high.log_prob)
    assert torch.allclose(low.entropy, high.entropy)


def test_inactive_heads_are_gradient_inert() -> None:
    observation = torch.zeros(1, 512)
    observation[:, 80] = 1.0
    observation[:, 482] = 1.0
    mean = torch.zeros(1, 4, requires_grad=True)
    log_std = torch.zeros(4, requires_grad=True)
    logits = torch.zeros(1, 8, requires_grad=True)
    result = evaluate_hybrid(
        mean,
        log_std,
        logits,
        observation,
        torch.tensor([[0.4, 0.0, 0.0, 0.5]]),
        torch.zeros(1, dtype=torch.long),
    )
    (result.log_prob + result.entropy).sum().backward()
    assert torch.count_nonzero(mean.grad[0, 1:]) == 0
    assert torch.count_nonzero(log_std.grad[1:]) == 0
