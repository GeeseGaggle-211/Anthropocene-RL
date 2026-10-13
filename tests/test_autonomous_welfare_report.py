"""Welfare aggregation preserves family weights and avoids multiplying marginal means."""

import runpy
from pathlib import Path

import pytest

aggregate = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/report_autonomous_welfare_v4.py")
)["aggregate"]


def row(family, population, qol):
    return dict(
        family=family,
        initial_population=population,
        final_population=population,
        elapsed_years=2,
        time_mean_population=population,
        time_mean_qol=qol,
        final_qol=qol,
        person_years=population * 2,
        raw_welfare=population * qol * 2,
        welfare=qol * 2,
    )


def test_welfare_is_equal_family_weighted_and_person_year_decomposition_is_exact():
    result = aggregate([row("small", 10, 1), row("large", 100, 0.5), row("large", 100, 0.5)])
    assert result["initial_population"] == 55
    assert result["raw_welfare"] == 60
    assert result["welfare"] == 1.5
    assert result["time_mean_qol"] == 0.75
    assert result["person_year_weighted_qol"] == pytest.approx(60 / 110)
    assert result["raw_welfare"] != result["initial_population"] * result["welfare"]
    assert result["raw_welfare"] == pytest.approx(
        result["time_mean_population"]
        * result["elapsed_years"]
        * result["person_year_weighted_qol"]
    )
