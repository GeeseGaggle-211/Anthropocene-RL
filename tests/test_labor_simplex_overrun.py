"""Activity rows within the staffing tolerance but past the simplex tolerance.

Production batch fit8-b03, world 90119 (collector b6a1290, simulator 8bd480b)
failed at tail decision 62 with ``activity[9] must sum to 1.0``: camp 327 at
turn 120 staffs one Builder (activity row 9) whose protected-shelter bundle
plans 10.000000108927148 worker-years each of goods.housing (15) and
goods.housing-material (20) over a 20-year turn. The shares sum to
1 + 1.09e-8: inside ``LABEL_TOLERANCE`` (1e-6, so the role was treated as in
budget and its idle residual clamped to 0) but outside ``SIMPLEX_TOLERANCE``
(1e-8), so ``_simplex`` refused the row. The fixture holds that decision's exact
converter inputs, extracted from a local resume of ckpt-000111.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from anthropocene_rl.autonomous_v7 import labor_bc
from anthropocene_rl.autonomous_v7.labor_bc import (
    LABEL_TOLERANCE,
    _label_tensors,
    convert_heuristic_labor_label,
    reasons_block_labor,
)
from anthropocene_rl.autonomous_v7.labor_bridge import (
    SIMPLEX_TOLERANCE,
    LaborCatalog,
    LaborMasks,
    LaborPolicySample,
    _simplex,
)
from tests.test_labor_bc_g1 import _catalog, _diag, _herding, _village

FIXTURE = Path(__file__).parent / "fixtures" / "labor_simplex_overrun_w90119_c327_t120.json.gz"
REASON = "activity-simplex-tolerance-overrun"


def _fixture() -> tuple[LaborCatalog, dict, dict, dict]:
    data = json.loads(gzip.decompress(FIXTURE.read_bytes()))
    c = data["catalog"]
    catalog = LaborCatalog(
        camp_id=c["camp_id"],
        turn=c["turn"],
        population=c["population"],
        turn_years=c["turn_years"],
        masks=LaborMasks.from_values(c["masks"]["specialty"], c["masks"]["activity"]),
        specialty_rows=tuple(c["specialty_rows"]),
        activity_rows=tuple(c["activity_rows"]),
        resources=tuple(c["resources"]),
        digest=c["digest"],
        semantic_rows_supported=c["semantic_rows_supported"],
    )
    return catalog, data["village_heur"], data["econ_diag"], data["pre_observation"]


def test_extracted_b03_label_is_quarantined_not_crashed_or_repaired() -> None:
    catalog, village, econ, pre = _fixture()
    assert (catalog.camp_id, catalog.turn, catalog.population) == (327, 120, 150)
    label = convert_heuristic_labor_label(catalog, village, econ, pre_observation=pre)

    # The explicit reason, and the labor label typed unavailable like over-budget work.
    assert f"{REASON}:role9" in label.reasons
    assert [r for r in label.reasons if r.startswith(REASON)] == [f"{REASON}:role9"]
    assert reasons_block_labor(label.reasons)
    assert label.over_budget
    assert not label.specialty_valid
    assert not label.activity_role_valid.any()
    assert not label.full_action_valid

    # The heuristic's own staffing and worker-years are stored unchanged.
    assert label.specialty_counts.tolist() == [0, 17, 125, 0, 0, 0, 1, 0, 4, 1, 1, 1, 0, 0]
    assert label.planned_worker_years[9, 15] == 10.000000108927148
    assert label.planned_worker_years[9, 20] == 10.000000108927148
    assert np.count_nonzero(label.planned_worker_years[9]) == 2

    # The raw row is genuinely not a simplex: the old in-budget path crashed here.
    raw = label.planned_worker_years[9] / (1 * catalog.turn_years)
    assert LABEL_TOLERANCE > math.fsum(raw) - 1.0 > SIMPLEX_TOLERANCE
    with pytest.raises(ValueError, match=r"activity\[9\] must sum to 1.0"):
        _simplex(tuple(raw), tuple(catalog.masks.activity[9]), "activity[9]")

    # The executable drive (never a target) is a valid schema-12 labor action.
    drive = LaborPolicySample.from_payload(
        {
            "laborSpecialtyAllocation": label.action["laborSpecialtyAllocation"],
            "laborActivityAllocation": label.action["laborActivityAllocation"],
        },
        catalog.masks,
    )
    row9 = drive.activity_allocation[9]
    assert math.isclose(row9[15], 0.5, abs_tol=1e-12) and math.isclose(row9[20], 0.5, abs_tol=1e-12)
    # Every other role keeps exactly what the in-budget conversion gives it.
    for role in range(len(catalog.masks.activity)):
        if role == 9:
            continue
        assert math.isclose(math.fsum(label.activity[role]), 1.0, abs_tol=SIMPLEX_TOLERANCE)

    # Row arrays are still emitted in full (the row is recorded, its labor heads masked).
    tensors = _label_tensors(label)
    assert not bool(tensors["label_specialty_valid"].item())
    assert not bool(tensors["label_activity_role_valid"].any())


def test_old_converter_semantics_raise_on_the_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the failure: without the overrun branch the same inputs reach _simplex and raise."""
    catalog, village, econ, pre = _fixture()
    monkeypatch.setattr(labor_bc, "SIMPLEX_TOLERANCE", math.inf)  # disables only the new branch
    with pytest.raises(ValueError, match=r"activity\[9\] must sum to 1.0"):
        convert_heuristic_labor_label(catalog, village, econ, pre_observation=pre)


@pytest.mark.parametrize(
    ("relative_overrun", "expected"),
    [
        (0.0, "valid"),
        (5e-9, "valid"),  # within the simplex tolerance: unchanged, canonicalized by _simplex
        (5e-8, REASON),  # the gap that crashed b03
        (9e-7, REASON),
        (2e-6, "activity-over-budget"),  # past the staffing tolerance: the pre-existing reason
    ],
)
def test_overrun_bands(relative_overrun: float, expected: str) -> None:
    # One Herder split over two activities, as the b03 Builder was: a single-activity
    # overrun is absorbed by the [0, 1] clip and never reached _simplex unbalanced.
    catalog = _catalog(population=2)
    half = 10.0 * (1.0 + relative_overrun)
    label = convert_heuristic_labor_label(catalog, _village(2), _diag(_herding(1, half, half, 0.0)))
    if expected == "valid":
        assert label.specialty_valid and not label.over_budget
        assert not any(r.startswith(("activity-over-budget", REASON)) for r in label.reasons)
    else:
        assert label.over_budget and not label.specialty_valid
        assert any(r.startswith(expected) for r in label.reasons)
        if expected == REASON:
            assert not any(r.startswith("activity-over-budget") for r in label.reasons)
    for role in range(label.activity.shape[0]):
        assert math.isclose(math.fsum(label.activity[role]), 1.0, abs_tol=SIMPLEX_TOLERANCE)
