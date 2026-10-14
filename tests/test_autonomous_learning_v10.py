import copy
from types import SimpleNamespace

import pytest

from anthropocene_rl.autonomous_v6.learning_guards import (
    endpoint_summary,
    validate_cohorts,
    validate_parent,
)


def cohort():
    selection = dict(fit_worlds=[1], validation_worlds=[2], development_worlds=[3])
    selected, entries = {}, {}
    for role in ["calibration", "ppo", "development"]:
        selection[role], selected[role] = [], []
        for j in range(24 if role == "calibration" else 1):
            world = (1 if j < 12 else 2) if role == "calibration" else (1 if role == "ppo" else 3)
            id_ = f"{role}-{j}"
            slot = dict(world_seed=world, family=str(j), ages=[0])
            if role == "calibration":
                slot["role"] = "fit" if j < 12 else "validation"
            selection[role].append(slot)
            selected[role].append(id_)
            entries[id_] = dict(
                world_seed=world,
                family=str(j),
                decision_index=0,
                partition="development" if role == "development" else "train",
            )
    old = {str(j): dict(world_seed=4, partition="train") for j in range(12)}
    return selection, selected, entries, old, list(old), [4]


def test_actual_partition_and_old_fitting_leakage_reject_before_collection():
    args = cohort()
    validate_cohorts(*args)
    broken = copy.deepcopy(args)
    broken[2]["calibration-12"]["world_seed"] = 3
    with pytest.raises(ValueError):
        validate_cohorts(*broken)
    broken = copy.deepcopy(args)
    broken[1]["calibration"][13] = broken[1]["calibration"][12]
    with pytest.raises(ValueError):
        validate_cohorts(*broken)
    broken = copy.deepcopy(args)
    broken[-1][:] = [2]
    with pytest.raises(ValueError):
        validate_cohorts(*broken)


def test_parent_ready_hash_and_resume_counters_are_required():
    tr = SimpleNamespace(episodes=50, transitions=1424, updates=8)
    cells = [
        dict(
            seed=s,
            passed=True,
            checkpoint_sha256=str(s),
            checkpoint_counters=dict(episodes=50, transitions=1424, updates=8),
            durable_transitions=1024,
            status="COMPLETE",
        )
        for s in [101, 211, 307]
    ]
    summary = dict(status="READY", calibrations=cells)
    validate_parent(summary, dict(sha256="101"), tr, 101, 0)
    with pytest.raises(ValueError):
        validate_parent(summary, dict(sha256="211"), tr, 101, 0)
    summary["calibrations"][2]["passed"] = False
    with pytest.raises(ValueError):
        validate_parent(summary, dict(sha256="101"), tr, 101, 0)
    staged = dict(status="STAGED_REVIEW_REQUIRED", cells=cells)
    validate_parent(staged, dict(sha256="101"), tr, 101, 1024)
    tr.transitions += 1
    with pytest.raises(ValueError):
        validate_parent(staged, dict(sha256="101"), tr, 101, 1024)
    with pytest.raises(ValueError):
        validate_parent(staged, dict(sha256="101"), tr, 101, 1000)


def test_endpoint_qol_distinguishes_realized_interval_and_extinct_focal():
    rows = [
        dict(id=str(i), family="survival", final_population=p, trace=[dict(qol=q)])
        for i, (p, q) in enumerate([(0, 0.8), (100, 0.6)])
    ]
    result = endpoint_summary(rows)
    assert result["mean_final_population"] == 50
    assert result["mean_final_realized_qol"] == pytest.approx(0.7)
    assert result["mean_living_endpoint_qol"] == pytest.approx(0.3)
