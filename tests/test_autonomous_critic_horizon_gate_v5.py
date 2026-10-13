"""Prospective budget, selection and scale-aware critic reporting checks."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture
def gate():
    directory = Path("scripts").resolve()
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "horizon_gate", directory / "run_autonomous_critic_horizon_v5.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(directory))


def test_gate_preserves_exhausted_history_and_rejects_unapproved_extra_native_work(gate):
    config = dict(
        variants=["small_step", "residual_layernorm"],
        horizons=[100, 400],
        seeds=[29, 43],
        transitions_per_cell=4096,
        maximum_training_transitions=32768,
        prior_additional_transitions=98304,
        maximum_cumulative_additional=131072,
        maximum_wall_seconds=2400,
        workers=4,
        torch_threads=1,
    )
    previous = dict(
        cumulative_transitions=98304,
        seeds=[dict(seed=seed, status="completed") for seed in (29, 43)],
    )
    assert gate.validate_gate(config, previous) == 32768
    for key, value in (
        ("transitions_per_cell", 8192),
        ("prior_additional_transitions", 0),
        ("maximum_cumulative_additional", 32768),
        ("maximum_wall_seconds", 2401),
        ("seeds", [17, 29, 43]),
    ):
        with pytest.raises(ValueError):
            gate.validate_gate(dict(config, **{key: value}), previous)


def test_extremes_cover_intermediate_horizons_and_only_trainable_actor_saturation(gate):
    def cell(name, variant, saturation, confidence):
        return dict(
            id=name,
            variant=variant,
            health=[
                dict(
                    transitions=2048,
                    checkpoint=name,
                    health=dict(
                        actor_layers=[dict(saturated_fraction=saturation)],
                        frozen_reference_layers=[dict(saturated_fraction=1.0)],
                        categorical=dict(
                            enabled=dict(eligible=32, maximum_probability=confidence),
                            partner=dict(eligible=0, maximum_probability=1.0),
                        ),
                    ),
                )
            ],
        )

    cells = [
        cell("early-h100", "small_step", 0.8, 0.6),
        cell("later-h400", "small_step", 0.2, 0.9),
        cell("tie-h400", "small_step", 0.8, 0.9),
        cell("residual", "residual_layernorm", 0.1, 0.4),
    ]
    selected = gate.health_extremes(cells)
    assert [r["cell"] for r in selected] == [
        "later-h400",
        "early-h100",
        "residual",
        "residual",
    ]


def test_critic_variance_diagnostics_use_real_scale_and_flag_constant_returns(gate):
    returns = torch.tensor([0.0, 0.2, 0.4])
    prediction = returns + torch.tensor([-0.1, 0.0, 0.1])
    report = gate.critic_return_diagnostics(prediction, returns, 0.01)
    assert report["explained_variance"] == pytest.approx(0.75)
    assert report["critic_mse_over_return_variance"] == pytest.approx(0.375)
    assert not report["near_constant_returns"]
    constant = gate.critic_return_diagnostics(torch.ones(3), torch.ones(3), 0.0)
    assert constant["near_constant_returns"]
    assert constant["variance_denominator"] == 1e-12
