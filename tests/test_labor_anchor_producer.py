import math

import pytest

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7.gate34_continuation_collector import (
    FrozenGeometricSchedule,
)
from anthropocene_rl.autonomous_v7.labor_anchor_producer import (
    PathHeadReturn,
    _combine_paths,
    summarize_labor_path,
)


def _schedule(*, cap: int = 8) -> FrozenGeometricSchedule:
    return FrozenGeometricSchedule(
        schedule_id="world-50000/path-0",
        survival_probability=0.95,
        stop_indices=(0, 1, 2, 3),
        hard_cap_terms=cap,
        rng_seed=80000,
    )


def test_terminal_path_is_exact_and_never_bootstraps() -> None:
    result = summarize_labor_path(
        [2.0, 3.0],
        [20.0, 20.0],
        [False, True],
        _schedule(),
        head="qol_years",
        prefix_terms=25,
        provenance_id="terminal",
    )
    gamma = math.exp(-contracts.TURN_YEARS / contracts.HORIZON_YEARS)
    assert result.value == pytest.approx(2.0 + gamma * 3.0)
    assert result.standard_error == 0.0
    assert result.method == "terminal-exact"
    assert not result.terminal_first


def test_living_path_uses_fixed_prefix_and_global_discounted_tail() -> None:
    schedule = _schedule()
    terms = 2 + schedule.required_terms
    result = summarize_labor_path(
        [1.0] * terms,
        [20.0] * terms,
        [False] * terms,
        schedule,
        head="settlement_guard",
        prefix_terms=2,
        provenance_id="living",
    )
    gamma = math.exp(-contracts.TURN_YEARS / contracts.HORIZON_YEARS)
    expected_prefix = 1.0 + gamma
    draws = []
    for stop in schedule.stop_indices:
        draws.append(
            sum(gamma ** (2 + index) / schedule.survival_probability**index
                for index in range(stop + 1))
        )
    assert result.value == pytest.approx(expected_prefix + sum(draws) / len(draws))
    assert result.standard_error is not None and result.standard_error > 0
    assert result.method == "independent-randomized-tail-path-mean"


def test_hard_cap_is_withheld_without_redrawing_or_zero_tail() -> None:
    schedule = FrozenGeometricSchedule(
        schedule_id="frozen-over-cap",
        survival_probability=0.95,
        stop_indices=(0, 5),
        hard_cap_terms=4,
        rng_seed=80001,
    )
    result = summarize_labor_path(
        [1.0] * 8,
        [20.0] * 8,
        [False] * 8,
        schedule,
        head="feasibility",
        prefix_terms=2,
        provenance_id="over-cap",
    )
    assert result.value is None
    assert result.standard_error is None
    assert result.method == "unavailable"


def test_path_combination_includes_between_path_and_tail_uncertainty() -> None:
    value, standard_error, method, available = _combine_paths(
        [
            PathHeadReturn(10.0, 2.0, "terminal-exact", 1.0, False),
            PathHeadReturn(
                14.0,
                2.0,
                "independent-randomized-tail-path-mean",
                1.0,
                False,
            ),
        ]
    )
    # Between-path variance/n plus the two within-path variances divided by n^2.
    assert value == 12.0
    assert standard_error == pytest.approx(math.sqrt(4.0 + 2.0))
    assert method == "independent-randomized-tail-path-mean"
    assert available
    assert _combine_paths(
        [PathHeadReturn(None, None, "unavailable", 0.0, False)]
    ) == (0.0, 0.0, "unavailable", False)
