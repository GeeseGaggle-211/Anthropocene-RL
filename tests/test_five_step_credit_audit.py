import pytest

from anthropocene_rl.training.five_step_credit_audit import (
    _assess_delayed_effect,
    _digest,
    _profile_definitions,
    _seed_probe_summary,
)


def _probe(
    snapshot: str,
    *,
    relative_three: float,
    relative_five: float,
    requested_span: float = 0.2,
    accepted_span: float = 0.2,
) -> dict[str, object]:
    eligible = requested_span >= 0.1
    informative = eligible and accepted_span > 0.01 and relative_three <= 0.001
    return {
        "snapshot_id": snapshot,
        "family": "F03",
        "requested_span": requested_span,
        "accepted_span": accepted_span,
        "three_step_relative_physical_spread": relative_three,
        "five_step_relative_physical_spread": relative_five,
        "request_eligible": eligible,
        "informative_delayed_effect_probe": informative,
        "delayed_effect_observed": informative and relative_five > 0.001,
    }


def test_structural_profile_matrix_is_exact() -> None:
    profiles = _profile_definitions((1701, 1901, 2101))
    assert len(profiles) == 14
    assert sum(profile["kind"] == "constant" for profile in profiles) == 9
    assert sum(profile["kind"] == "alternating" for profile in profiles) == 2
    assert sum(profile["kind"] == "policy" for profile in profiles) == 3


def test_delayed_effect_gate_requires_independent_snapshots_and_rate_gain() -> None:
    passing = [
        _probe(f"snapshot-{index // 2}", relative_three=0.0, relative_five=0.01)
        for index in range(8)
    ]
    passing.extend(
        _probe(f"responsive-{index}", relative_three=0.01, relative_five=0.01)
        for index in range(4)
    )
    summary = _seed_probe_summary(passing)
    assert summary["informative_cohort_gate"] is True
    assert summary["delayed_effect_fraction"] == 1.0
    assert summary["f03_responsive_fraction_gain"] == pytest.approx(8 / 12)
    assert summary["delayed_effect_gate"] is True

    one_snapshot = [
        _probe("same", relative_three=0.0, relative_five=0.01) for _ in range(8)
    ]
    assert _seed_probe_summary(one_snapshot)["informative_cohort_gate"] is False


def test_campaign_assessment_requires_two_seed_delayed_support() -> None:
    passing_probes = [
        _probe(f"snapshot-{index // 2}", relative_three=0.0, relative_five=0.01)
        for index in range(8)
    ]
    passing_probes.extend(
        _probe(f"responsive-{index}", relative_three=0.01, relative_five=0.01)
        for index in range(4)
    )
    failing_probes = [
        _probe(f"snapshot-{index // 2}", relative_three=0.0, relative_five=0.0)
        for index in range(8)
    ]
    reports = [
        {"seed": 1701, "summary": _seed_probe_summary(passing_probes)},
        {"seed": 1901, "summary": _seed_probe_summary(passing_probes)},
        {"seed": 2101, "summary": _seed_probe_summary(failing_probes)},
    ]
    structural = {"summary": {"status": "PASS"}}
    assessment = _assess_delayed_effect(reports, structural)
    assert assessment["status"] == "FIVE_STEP_PPO_ABLATION_JUSTIFIED"
    assert assessment["delayed_effect_support_seeds"] == [1701, 1901]


def test_empty_probe_summary_is_canonical_json() -> None:
    summary = _seed_probe_summary([])
    assert summary["delayed_effect_fraction"] is None
    assert summary["f03_responsive_fraction_gain"] is None
    assert summary["delayed_effect_gate"] is False
    assert len(_digest({"summary": summary})) == 64
