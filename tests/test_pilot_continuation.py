from __future__ import annotations

import pytest

from anthropocene_rl.training.pilot import (
    PilotConfig,
    _checkpoint_selection_score,
    _heuristic_parity,
    _parity_plateau_reached,
)


def _evaluation(*, qol: float, population: float, scaled_return: float = 0.0) -> dict:
    return {
        "summary": {
            "final_qol": {"mean": qol},
            "final_population": {"mean": population},
            "discounted_scaled_return": {"mean": scaled_return},
        }
    }


def test_heuristic_parity_requires_both_qol_and_population() -> None:
    baseline = _evaluation(qol=0.8, population=100.0)
    learned = _evaluation(qol=0.76, population=89.0)

    parity = _heuristic_parity(learned, baseline, tolerance=0.10)

    assert parity["qol"]["relative_difference"] == pytest.approx(-0.05)
    assert parity["qol"]["within_tolerance"] is True
    assert parity["population"]["relative_difference"] == pytest.approx(-0.11)
    assert parity["population"]["within_tolerance"] is False
    assert parity["distance"] == pytest.approx(0.11)
    assert parity["within_tolerance"] is False


def test_parity_checkpoint_score_minimizes_worst_relative_gap() -> None:
    baseline = _evaluation(qol=0.8, population=100.0)
    learned = _evaluation(qol=0.76, population=92.0)

    score, diagnostics = _checkpoint_selection_score(
        learned,
        baseline,
        mode="development-heuristic-parity",
        tolerance=0.10,
    )

    assert diagnostics is not None
    assert diagnostics["distance"] == pytest.approx(0.08)
    assert score == pytest.approx(-0.08)


def test_parity_plateau_uses_running_best_over_declared_intervals() -> None:
    curve = [
        {"distance": distance}
        for distance in (0.20, 0.195, 0.193, 0.192, 0.191, 0.1905)
    ]
    assert _parity_plateau_reached(
        curve,
        resume_update=40,
        completed_update=160,
        evaluations=5,
        minimum_added_updates=100,
        minimum_improvement=0.01,
    )


def test_parity_stopping_rules_require_parity_selection() -> None:
    with pytest.raises(ValueError, match="Parity stopping rules"):
        PilotConfig(heuristic_parity_patience=2).validate()

