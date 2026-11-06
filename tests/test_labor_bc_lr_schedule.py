"""Brief section 5.1 actor learning-rate schedule for pretrain-bc (--lr-schedule).

Pure checks: the warm-up/cosine factor, the CLI plumbing, the historical constant
default (no kwarg, bit-for-bit unchanged fit), the per-step rates a scheduled fit
applies, and a split fit (start step) continuing the same schedule.
"""

from __future__ import annotations

import math

import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_training as lt
from anthropocene_rl.autonomous_v7.labor_bc import lr_schedule_factor, pretrain_labor_actor_bc
from tests.test_labor_bc import (
    LABOR_ACTIVITY_FEATURE_WIDTH,
    LABOR_RESOURCE_FEATURE_WIDTH,
    LABOR_SPECIALTY_FEATURE_WIDTH,
    LaborActor,
    LaborEncoderLayout,
    contracts,
)


def test_factor_warms_up_then_decays_to_the_floor() -> None:
    total = 100
    values = [lr_schedule_factor(s, total, 0.1, 0.1) for s in range(total + 20)]
    assert values[0] == pytest.approx(0.1) and values[9] == pytest.approx(1.0)
    assert all(b > a for a, b in zip(values[:9], values[1:10], strict=False))  # linear warm-up
    assert all(
        b <= a + 1e-12 for a, b in zip(values[10:total], values[11 : total + 1], strict=False)
    )
    mid = lr_schedule_factor(10 + 45, total, 0.1, 0.1)
    assert mid == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.5)))
    assert values[total] == pytest.approx(0.1) and values[-1] == pytest.approx(0.1)
    assert lr_schedule_factor(0, 10, 0.0, 0.1) == pytest.approx(1.0)  # no warm-up


def test_cli_default_is_the_historical_constant_rate() -> None:
    parser = lt.argument_parser()
    base = ["pretrain-bc", "--checkpoint", "c.pt", "--dataset", "d", "--out", "o.pt"]
    args = parser.parse_args(base)
    assert args.lr_schedule == "constant" and lt._bc_lr_schedule_kwargs(args) == {}
    args = parser.parse_args(
        [
            *base,
            "--lr",
            "3e-5",
            "--lr-schedule",
            "warmup-cosine",
            "--lr-schedule-total-steps",
            "1000",
            "--lr-schedule-start-step",
            "250",
        ]
    )
    assert lt._bc_lr_schedule_kwargs(args) == {
        "lr_schedule": {
            "kind": "warmup-cosine",
            "warmup_fraction": 0.1,
            "final_fraction": 0.1,
            "start_step": 250,
            "total_steps": 1000,
        }
    }


def _dataset(B: int = 32) -> dict:
    torch.manual_seed(3)
    target_specialty = torch.zeros(B, 14)
    target_specialty[:, 1] = 0.8
    target_specialty[:, 2] = 0.2
    target_activity = torch.zeros(B, 13, 22)
    target_activity[:, 0, 1] = 1.0
    target_activity[:, 1:, 0] = 1.0
    data = {
        "core": torch.randn(B, 15),
        "labor_personality": torch.full((B, 2), 0.5),
        "movement_mask": torch.ones(B, 8, dtype=torch.bool),
        "region_movement_mask": torch.ones(B, 10, dtype=torch.bool),
        "labor_specialty_features": torch.randn(B, 14, LABOR_SPECIALTY_FEATURE_WIDTH),
        "labor_specialty_present": torch.ones(B, 14, dtype=torch.bool),
        "labor_specialty_mask": torch.ones(B, 14, dtype=torch.bool),
        "labor_activity_features": torch.randn(B, 13 * 22, LABOR_ACTIVITY_FEATURE_WIDTH),
        "labor_activity_present": torch.ones(B, 13 * 22, dtype=torch.bool),
        "labor_activity_mask": torch.ones(B, 13, 22, dtype=torch.bool),
        "labor_resource_features": torch.randn(B, 20, LABOR_RESOURCE_FEATURE_WIDTH),
        "labor_resource_present": torch.ones(B, 20, dtype=torch.bool),
        "target_specialty": target_specialty,
        "target_activity": target_activity,
        "target_movement_code": torch.full((B,), -1, dtype=torch.long),
        "target_fission": torch.zeros(B, dtype=torch.long),
        "target_trade_value_latent": torch.zeros(B, 7),
        "target_trade_gate": torch.ones(B, 7, dtype=torch.long),
        "target_trade_share_latent": torch.zeros(B),
        "collection": {"total_decisions": B, "initial_accepted": B},
    }
    for name, rows, columns in contracts.SETS:
        data[name] = torch.randn(B, rows, columns)
        data[name + "_present"] = torch.ones(B, rows, dtype=torch.bool)
    return data


def _actor() -> LaborActor:
    torch.manual_seed(11)
    return LaborActor(
        LaborEncoderLayout(
            15,
            LABOR_SPECIALTY_FEATURE_WIDTH,
            LABOR_ACTIVITY_FEATURE_WIDTH,
            LABOR_RESOURCE_FEATURE_WIDTH,
        )
    )


class _Recording(torch.optim.Adam):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.rates: list[float] = []

    def step(self, closure=None):
        self.rates.append(float(self.param_groups[0]["lr"]))
        return super().step(closure)


def _fit(schedule=None, epochs=4, start_actor=None):
    actor = start_actor or _actor()
    optimizer = _Recording(actor.parameters(), lr=1e-3)
    summary = pretrain_labor_actor_bc(
        actor=actor,
        dataset=_dataset(),
        optimizer=optimizer,
        epochs=epochs,
        batch_size=8,
        lr=1e-3,
        **({} if schedule is None else {"lr_schedule": schedule}),
    )
    return actor, optimizer.rates, summary


def test_constant_kind_equals_the_default_fit() -> None:
    default, rates_default, summary = _fit()
    constant, rates_constant, _ = _fit({"kind": "constant"})
    assert rates_default == rates_constant == [1e-3] * 16
    assert "lr_schedule" not in summary
    for a, b in zip(default.state_dict().values(), constant.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_scheduled_fit_applies_the_factor_every_step() -> None:
    _, rates, summary = _fit({"kind": "warmup-cosine"})
    expected = [1e-3 * lr_schedule_factor(s, 16, 0.1, 0.1) for s in range(16)]
    assert rates == pytest.approx(expected)
    record = summary["lr_schedule"]
    assert record["total_steps"] == 16 and record["steps_this_fit"] == 16
    assert record["end_step"] == 16 and record["peak_lr"] == 1e-3
    assert record["first_lr"] == pytest.approx(expected[0])
    assert record["last_lr"] == pytest.approx(expected[-1])


def test_a_split_fit_continues_the_same_schedule() -> None:
    _, first, one = _fit({"kind": "warmup-cosine", "total_steps": 16}, epochs=2)
    _, second, two = _fit(
        {"kind": "warmup-cosine", "total_steps": 16, "start_step": one["lr_schedule"]["end_step"]},
        epochs=2,
    )
    _, whole, _ = _fit({"kind": "warmup-cosine"}, epochs=4)
    assert first + second == pytest.approx(whole)
    assert two["lr_schedule"]["end_step"] == 16


def test_invalid_schedules_are_refused() -> None:
    with pytest.raises(ValueError, match="lr schedule must be"):
        _fit({"kind": "linear"})
    with pytest.raises(ValueError, match="fractions"):
        _fit({"kind": "warmup-cosine", "warmup_fraction": 1.0})
