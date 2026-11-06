from __future__ import annotations

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts, evaluate
from anthropocene_rl.autonomous_v7.gate1_diagnostics import generator_state_sha256
from anthropocene_rl.autonomous_v7.legacy_contract8 import (
    SETS,
    Contract8Actor,
    deterministic_mode,
    load,
    to_tensors,
)


def checkpoint_state() -> dict:
    with torch.random.fork_rng():
        torch.manual_seed(81)
        actor = Contract8Actor(3)
    normalizer = {"core_mean": [0.0] * 3, "core_std": [1.0] * 3}
    for name, _, width in SETS:
        normalizer[name + "_mean"] = [0.0] * width
        normalizer[name + "_std"] = [1.0] * width
    return {"contract": 8, "core_width": 3, "actor": actor.state_dict(), "normalizer": normalizer}


def batch() -> dict:
    result = {
        "core": np.asarray([[1.0, 2.0, 3.0, 999.0]], np.float32),
        "core_present": np.ones((1, 4), bool),
        "movement_mask": np.ones((1, 8), bool),
        "clothing_route_mask": np.ones((1, 2), bool),
        "shelter_route_mask": np.ones((1, 2), bool),
        "economy_mix_mask": np.ones((1, 3), bool),
    }
    for name, rows, width in SETS:
        result[name] = np.zeros((1, rows, width), np.float32)
        result[name + "_present"] = np.zeros((1, rows), bool)
    return result


def test_pinned_contract8_loader_preserves_fixture_outputs_without_upgrade(tmp_path, monkeypatch):
    state = checkpoint_state()
    path = tmp_path / "contract8.pt"
    torch.save(state, path)
    # Frozen from the pinned commit-2f4cc895 fixture, rather than recomputing the
    # expectation with the adapter under test.
    expected = {
        "strategic": np.asarray(
            [
                [
                    0.48374128341674805,
                    0.2830224335193634,
                    0.1979578286409378,
                    0.5622341632843018,
                    0.3135809004306793,
                ]
            ],
            np.float32,
        ),
        "mix": np.asarray(
            [[0.38592827320098877, 0.3004908263683319, 0.3135809004306793]], np.float32
        ),
        "service": np.asarray(
            [
                [
                    0.507912814617157,
                    0.2386050522327423,
                    0.5190595984458923,
                    0.36563485860824585,
                    0.44817012548446655,
                    0.4539732336997986,
                ]
            ],
            np.float32,
        ),
        "clothing": np.asarray([0]),
        "shelter": np.asarray([0]),
        "fission": np.asarray([2]),
        "movement": np.asarray([0]),
    }
    monkeypatch.setattr(
        evaluate,
        "upgrade",
        lambda *_: (_ for _ in ()).throw(AssertionError("upgrade must not run")),
    )
    actor, normalizer = evaluate.load(
        path,
        strict=True,
        resolved_semantics={
            "experts": False,
            "token": False,
            "commitments": False,
            "learned_movement": False,
            "region_movement": False,
            "learned_fission": False,
            "learned_trade": False,
            "region_head": "flat",
            "grouped_region": False,
            "region_selection": None,
        },
        diagnostic_loading={"mode": "reviewed-adapter", "adapter": "pinned-contract-8"},
    )
    with torch.no_grad():
        actual = deterministic_mode(actor(to_tensors(batch(), normalizer)))
    for key in expected:
        assert np.array_equal(actual[key], expected[key])


def test_contract8_loader_rejects_missing_or_synthesized_tensor():
    state = checkpoint_state()
    state["actor"] = dict(state["actor"])
    state["actor"].pop("service.bias")
    with pytest.raises(ValueError, match="raw actor keys differ"):
        load(state)


@pytest.mark.parametrize("mutation", ["extra", "dtype", "shape"])
def test_contract8_loader_rejects_nonexact_actor_state(mutation):
    state = checkpoint_state()
    state["actor"] = dict(state["actor"])
    if mutation == "extra":
        state["actor"]["invented.weight"] = torch.zeros(1)
    elif mutation == "dtype":
        state["actor"]["service.bias"] = state["actor"]["service.bias"].double()
    else:
        state["actor"]["service.bias"] = state["actor"]["service.bias"][:-1]
    with pytest.raises(ValueError, match="raw actor keys differ|dtype/device/shape/layout"):
        load(state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("core_mean", [0.0, float("nan"), 0.0]),
        ("core_std", [1.0, 0.0, 1.0]),
        ("core_std", [1.0, -1.0, 1.0]),
    ],
)
def test_contract8_loader_rejects_invalid_normalizer_moments(field, value):
    state = checkpoint_state()
    state["normalizer"] = dict(state["normalizer"])
    state["normalizer"][field] = value
    with pytest.raises(ValueError, match="invalid moments"):
        load(state)


def test_current_loader_rejects_contract8_without_reviewed_adapter(tmp_path):
    path = tmp_path / "contract8.pt"
    torch.save(checkpoint_state(), path)
    with pytest.raises(ValueError, match="reviewed pinned adapter"):
        evaluate.load(path, strict=True, resolved_semantics={})


def test_seeded_stochastic_contract8_execution_is_repeatable_and_global_rng_neutral():
    actor, normalizer = load(checkpoint_state())
    observation = {
        "core": [1.0, 2.0, 3.0, 999.0],
        "corePresent": [True] * 4,
        "movementMask": [True] * 8,
        "clothingRouteMask": [True, True],
        "shelterRouteMask": [True, True],
        "economyMixMask": [True] * 3,
        "sets": [[[0.0] * width for _ in range(rows)] for _, rows, width in contracts.SETS],
        "rowPresent": [[False] * rows for _, rows, _ in contracts.SETS],
    }
    villages = [{"camp": 1, "society": 1, "terminated": False, "observation": observation}]
    seed = 1234
    generator = torch.Generator(device="cpu").manual_seed(seed)
    schedule = {
        "step": 0,
        "seed": seed,
        "state_sha256": generator_state_sha256(generator),
        "draw_order": ["mix", "strategic", "service", "clothing", "shelter"],
    }
    global_state = torch.random.get_rng_state().clone()
    first, first_receipt = evaluate.policy_actions_stochastic(
        actor,
        normalizer,
        villages,
        False,
        None,
        schedule,
    )
    second, second_receipt = evaluate.policy_actions_stochastic(
        actor,
        normalizer,
        villages,
        False,
        None,
        schedule,
    )
    assert first == second
    assert first_receipt == second_receipt
    assert torch.equal(torch.random.get_rng_state(), global_state)
