import pytest

from anthropocene_rl.training.pilot import (
    _compare_evaluations,
    _self_sufficiency_diagnostic,
)


def _episode(
    snapshot_id: str,
    family: str,
    tags: list[str],
    field_operation: float,
    discounted_scaled_return: float = 1.0,
) -> dict[str, object]:
    return {
        "snapshot_id": snapshot_id,
        "scenario_family": family,
        "tags": tags,
        "discounted_scaled_return": discounted_scaled_return,
        "actions": {"requested": {"field_operation": field_operation}},
    }


def test_trade_contrast_is_unevaluable_without_both_f15_sides() -> None:
    diagnostic = _self_sufficiency_diagnostic(
        [
            _episode("migration", "F15", ["immigration"], 0.9),
            _episode("isolated-other", "F03", ["isolated"], 0.9),
        ]
    )

    assert diagnostic["trade_contrast_evaluable"] is False
    assert diagnostic["trade_contrast_assessment"] == "UNEVALUABLE"
    assert diagnostic["universal_self_sufficiency_signal"] is None


def test_trade_contrast_is_evaluable_with_disjoint_f15_cases() -> None:
    diagnostic = _self_sufficiency_diagnostic(
        [
            _episode("imports", "F15", ["trade-dependent"], 0.4),
            _episode("isolated", "F15", ["isolated"], 0.8),
        ]
    )

    assert diagnostic["trade_contrast_evaluable"] is True
    assert diagnostic["trade_contrast_assessment"] == "EVALUABLE"
    assert diagnostic["trade_dependent_minus_isolated_mean_field_operation"] == pytest.approx(
        -0.4
    )
    assert diagnostic["universal_self_sufficiency_signal"] is False


def test_paired_comparison_reports_family_relative_returns() -> None:
    baseline = {
        "episodes": [
            _episode("a", "F01", [], 0.5, 2.0),
            _episode("b", "F03", [], 0.5, 4.0),
        ]
    }
    learned = {
        "episodes": [
            _episode("a", "F01", [], 0.5, 2.2),
            _episode("b", "F03", [], 0.5, 3.0),
        ]
    }

    comparison = _compare_evaluations(learned, baseline)

    assert comparison["by_family_mean_relative_difference"]["F01"] == pytest.approx(0.1)
    assert comparison["by_family_mean_relative_difference"]["F03"] == pytest.approx(-0.25)

