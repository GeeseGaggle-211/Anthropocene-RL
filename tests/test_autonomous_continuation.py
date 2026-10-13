"""Frozen cumulative budgets and technical checkpoint selection cannot silently drift."""

import copy

import pytest

from anthropocene_rl.autonomous_v4.continuation import select_checkpoints, validate_budget


def fixture():
    config = dict(
        seeds=[29, 43],
        additional_transitions_per_seed=24576,
        maximum_total_transitions=49152,
        maximum_wall_seconds=12493,
        workers=4,
        torch_threads=1,
        evaluate_every_transitions=4096,
    )
    histories = [
        dict(
            seeds=[
                dict(seed=s, additional_transitions=n)
                for s, n in [(17, 32768), (29, 4096), (43, 4096)]
            ],
            additional_transitions=40960,
            wall_seconds=1533.183883661,
        ),
        dict(
            seeds=[dict(seed=s, additional_transitions=4096) for s in [29, 43]],
            additional_transitions=8192,
            wall_seconds=373.582114262,
        ),
    ]
    return config, histories


def test_budget_accounts_for_every_prior_seed_and_wall_duration():
    config, histories = fixture()
    actual = validate_budget(config, histories)
    assert actual["prior_transitions"] == 49152
    assert actual["prior_by_seed"] == {17: 32768, 29: 8192, 43: 8192}
    for key, value in [
        ("maximum_wall_seconds", 12494),
        ("additional_transitions_per_seed", 24577),
        ("maximum_total_transitions", 49153),
        ("seeds", [17, 29, 43]),
        ("workers", 8),
    ]:
        invalid = copy.deepcopy(config)
        invalid[key] = value
        with pytest.raises(ValueError):
            validate_budget(invalid, histories)
    with pytest.raises(ValueError):
        validate_budget(config, histories[1:])
    histories[0]["seeds"][0]["additional_transitions"] = -32768
    with pytest.raises(ValueError):
        validate_budget(config, histories)


def frame(boundary, actor, critic, confidence, eligible=32):
    return dict(
        additional_transitions=boundary,
        health=dict(
            actor_layers=[dict(saturated_fraction=actor)],
            critic_layers=[dict(saturated_fraction=critic)],
            categorical=dict(movement=dict(eligible=eligible, maximum_probability=confidence)),
        ),
    )


def test_selection_includes_critic_extreme_ignores_ineligible_and_breaks_ties_early():
    frames = [
        frame(0, 0.1, 0.8, 0.7),
        frame(4096, 0.2, 0.7, 0.9),
        frame(8192, 0.3, 0.8, 0.9),
        frame(12288, 0.4, 0.6, 1.0, eligible=0),
    ]
    selected = select_checkpoints(list(reversed(frames)))
    assert [r["transitions"] for r in selected] == [0, 4096, 12288]
    assert selected[0]["reasons"] == ["maximum_actor_or_critic_layer_saturation"]
    assert selected[1]["reasons"] == ["maximum_legal_categorical_confidence"]
    assert len(select_checkpoints([frames[0]])) == 1
    assert len(select_checkpoints([frames[0]])[0]["reasons"]) == 3
    with pytest.raises(ValueError):
        select_checkpoints([frames[0], frames[0]])
    with pytest.raises(ValueError):
        select_checkpoints([frame(0, float("nan"), 0.5, 0.9)])
