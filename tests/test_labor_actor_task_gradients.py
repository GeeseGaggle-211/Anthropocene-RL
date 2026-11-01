from __future__ import annotations

import math

import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7.labor_actor_gradnorm import (
    ACTOR_BC_TASKS,
    ACTOR_ROLE_TASKS,
    actor_gradnorm_parameters,
    actor_trunk_parameters,
)
from anthropocene_rl.autonomous_v7.labor_actor_task_gradients import (
    format_table,
    task_gradient_report,
)
from anthropocene_rl.autonomous_v7.labor_bc import _batch_loss_and_l1
from anthropocene_rl.autonomous_v7.labor_bridge import (
    LABOR_ACTIVITY_FEATURE_WIDTH,
    LABOR_RESOURCE_FEATURE_WIDTH,
    LABOR_SPECIALTY_FEATURE_WIDTH,
)
from anthropocene_rl.autonomous_v7.labor_model import LaborActor, LaborEncoderLayout


def synthetic_actor_dataset(rows: int = 24, seed: int = 0) -> tuple[LaborActor, dict]:
    """A small actor and BC dataset in which Gatherer/Farmer carry most residents and a few
    rows staff rare craft roles (the r5 shape), every head label valid."""
    torch.manual_seed(seed)
    core_dim = 15
    layout = LaborEncoderLayout(
        core_dim, LABOR_SPECIALTY_FEATURE_WIDTH, LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
    )
    actor = LaborActor(layout)
    generator = torch.Generator().manual_seed(seed)
    target_specialty = torch.zeros(rows, 14)
    target_specialty[:, 1] = 0.55
    target_specialty[:, 2] = 0.43
    target_specialty[:, 8] = 0.02  # Weaver in every row
    target_specialty[::4, 2] -= 0.01
    target_specialty[::4, 7] = 0.01  # Potter in a quarter of the rows
    target_activity = torch.zeros(rows, 13, 22)
    choice = torch.randint(0, 22, (rows, 13), generator=generator)
    target_activity.scatter_(2, choice.unsqueeze(-1), 0.7)
    target_activity[..., 0] += 0.3
    dataset: dict = {
        "core": torch.randn(rows, core_dim, generator=generator),
        "labor_personality": torch.full((rows, 2), 0.5),
        "movement_mask": torch.ones(rows, 8, dtype=torch.bool),
        "region_movement_mask": torch.ones(rows, 10, dtype=torch.bool),
        "labor_specialty_features": torch.randn(
            rows, 14, LABOR_SPECIALTY_FEATURE_WIDTH, generator=generator
        ),
        "labor_specialty_present": torch.ones(rows, 14, dtype=torch.bool),
        "labor_specialty_mask": torch.ones(rows, 14, dtype=torch.bool),
        "labor_activity_features": torch.randn(
            rows, 13 * 22, LABOR_ACTIVITY_FEATURE_WIDTH, generator=generator
        ),
        "labor_activity_present": torch.ones(rows, 13 * 22, dtype=torch.bool),
        "labor_activity_mask": torch.ones(rows, 13, 22, dtype=torch.bool),
        "labor_resource_features": torch.randn(
            rows, 20, LABOR_RESOURCE_FEATURE_WIDTH, generator=generator
        ),
        "labor_resource_present": torch.ones(rows, 20, dtype=torch.bool),
        "target_specialty": target_specialty,
        "target_activity": target_activity,
        "target_movement_code": torch.full((rows,), -1, dtype=torch.long),
        "target_fission": torch.zeros(rows, dtype=torch.long),
        "target_trade_value_latent": torch.zeros(rows, 7),
        "target_trade_gate": torch.ones(rows, 7, dtype=torch.long),
        "target_trade_share_latent": torch.zeros(rows),
        "collection": {"total_decisions": rows, "initial_accepted": rows},
    }
    for name, count, columns in contracts.SETS:
        dataset[name] = torch.randn(rows, count, columns, generator=generator)
        dataset[name + "_present"] = torch.ones(rows, count, dtype=torch.bool)
    return actor, dataset


def test_task_terms_leave_the_loss_unchanged_and_sum_to_activity() -> None:
    actor, data = synthetic_actor_dataset()
    act_w = torch.rand(22) + 0.5
    base, spec_l1, act_l1, components = _batch_loss_and_l1(
        actor, data, target_log_std=-2.0, act_weights=act_w
    )
    terms: dict = {}
    again, spec_l1_b, act_l1_b, components_b = _batch_loss_and_l1(
        actor, data, target_log_std=-2.0, act_weights=act_w, task_terms=terms
    )
    assert torch.equal(base, again)
    assert (spec_l1, act_l1, components) == (spec_l1_b, act_l1_b, components_b)
    assert set(terms["loss"]) == set(ACTOR_BC_TASKS)
    role_sum = sum(float(terms["loss"][t]) for t in ACTOR_ROLE_TASKS)
    assert role_sum == pytest.approx(components["activity"], rel=1e-5)
    assert float(terms["loss"]["specialty"]) == pytest.approx(components["specialty"])
    assert terms["staffed_rows"]["activity:Gatherer"] == len(data["core"])
    assert terms["staffed_rows"]["activity:Potter"] == 6
    assert terms["staffed_rows"]["activity:ClayMiner"] == 0
    assert float(terms["loss"]["activity:ClayMiner"]) == 0.0


def test_uniform_role_weighting_reports_only() -> None:
    actor, data = synthetic_actor_dataset()
    share_terms: dict = {}
    uniform_terms: dict = {}
    base = _batch_loss_and_l1(actor, data, target_log_std=-2.0, task_terms=share_terms)[0]
    other = _batch_loss_and_l1(
        actor, data, target_log_std=-2.0, task_terms=uniform_terms, role_weighting="uniform"
    )[0]
    assert torch.equal(base, other)
    assert share_terms["staffed_rows"] == uniform_terms["staffed_rows"]
    share_w = float(share_terms["loss"]["activity:Weaver"])
    uniform_w = float(uniform_terms["loss"]["activity:Weaver"])
    assert uniform_w > 5 * share_w  # Weaver 2% of residents -> about 1/3 under uniform
    with pytest.raises(ValueError):
        _batch_loss_and_l1(actor, data, target_log_std=-2.0, role_weighting="uniform")


def test_task_weights_rebuild_the_total() -> None:
    actor, data = synthetic_actor_dataset()
    base = _batch_loss_and_l1(actor, data, target_log_std=-2.0)[0]
    ones = {t: 1.0 for t in ACTOR_BC_TASKS}
    rebuilt = _batch_loss_and_l1(actor, data, target_log_std=-2.0, task_weights=ones)[0]
    assert float(rebuilt) == pytest.approx(float(base), rel=1e-5)
    doubled = dict(ones, specialty=2.0)
    terms: dict = {}
    heavier = _batch_loss_and_l1(
        actor, data, target_log_std=-2.0, task_weights=doubled, task_terms=terms
    )[0]
    assert float(heavier) == pytest.approx(float(base) + float(terms["loss"]["specialty"]), rel=1e-5)
    with pytest.raises(ValueError, match="miss"):
        _batch_loss_and_l1(actor, data, target_log_std=-2.0, task_weights={"specialty": 1.0})


def test_task_gradient_report_shapes_and_shares() -> None:
    actor, data = synthetic_actor_dataset(rows=16)
    data["row_world"] = torch.tensor([90100] * 8 + [90101] * 8)
    data["row_weight"] = torch.ones(16)
    report = task_gradient_report(actor, data, batch_size=8)
    assert report["batches"] == 2 and report["rows"] == 16
    assert report["parameter_sets"]["trunk"] == sum(p.numel() for p in actor_trunk_parameters(actor))
    assert report["parameter_sets"]["gradnorm_w"] == sum(
        p.numel() for p in actor_gradnorm_parameters(actor)
    )
    for weighting in ("share", "uniform"):
        for pset in ("trunk", "gradnorm_w"):
            shares = report["weightings"][weighting][pset]["mean_batch_share"]
            assert set(shares) == set(ACTOR_BC_TASKS)
            assert math.isclose(sum(shares.values()), 1.0, rel_tol=1e-9)
    trunk = report["weightings"]["share"]["trunk"]["mean_batch_norm"]
    assert trunk["activity:ClayMiner"] == 0.0
    assert trunk["activity:Gatherer"] > trunk["activity:Weaver"] > 0.0
    relative = report["per_staffed_row"]["share"]["relative_to_gatherer"]
    assert relative["activity:Gatherer"] == pytest.approx(1.0)
    assert relative["activity:ClayMiner"] is None
    assert "activity:Potter" in format_table(report)
