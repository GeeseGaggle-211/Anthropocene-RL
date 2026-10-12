import json
from pathlib import Path

import pytest

from anthropocene_rl.training.action_credit_audit import (
    _campaign_assessment,
    _direction,
    _evidence_sha256,
    combine_action_credit_audits,
)


def _report(
    seed: int,
    *,
    acceptance_failure: float = 0.0,
    responsive: int = 24,
    agreement: float = 0.8,
    acceptance_responsive: float = 0.9,
    actor_ratio: float = 1.0,
    clipping_pass: bool = False,
) -> dict[str, object]:
    return {
        "seed": seed,
        "probe_summary": {
            "development_budget": {
                "physical_reward_responsive": responsive,
                "physical_directional_credit_agreement": agreement,
                "acceptance_responsive_fraction": acceptance_responsive,
            },
            "f03_development": {
                "acceptance_and_physical_unresponsive_fraction": acceptance_failure,
            },
        },
        "shadow_clipping_control": {
            "comparison": {
                "actor_parameter_displacement_ratio": actor_ratio,
                "joint_clipping_escalation_seed_pass": clipping_pass,
            }
        },
    }


def test_direction_uses_a_small_deadband() -> None:
    assert _direction(1.0) == 1
    assert _direction(-1.0) == -1
    assert _direction(1e-13) == 0


def test_campaign_assessment_requires_two_seed_replication() -> None:
    reports = [_report(1701), _report(1901), _report(2101, acceptance_responsive=0.5)]
    assessment = _campaign_assessment(reports)
    assert assessment["status"] == "CLEAR_DIAGNOSTIC_PASS"
    assert assessment["clear_pass_seeds"] == [1701, 1901]

    reports = [
        _report(1701, clipping_pass=True, actor_ratio=1.6),
        _report(1901, clipping_pass=True, actor_ratio=1.7),
        _report(2101),
    ]
    assessment = _campaign_assessment(reports)
    assert assessment["status"] == "JOINT_CLIPPING_ABLATION_JUSTIFIED"

    reports = [
        _report(1701, acceptance_failure=0.6),
        _report(1901, acceptance_failure=0.7),
        _report(2101),
    ]
    assessment = _campaign_assessment(reports)
    assert assessment["status"] == "ACTION_CREDIT_FAILURE"


def test_combines_isolated_seed_evidence_with_identity_checks(tmp_path: Path) -> None:
    inputs: list[Path] = []
    for seed in (1701, 1901, 2101):
        source = tmp_path / f"input-{seed}"
        source.mkdir()
        report = _report(seed)
        summary = {
            "status": "INCONCLUSIVE",
            "scope": "train-only-paired-action-credit-audit",
            "partitions_read": ["train"],
            "heldout_read": False,
            "development_read": False,
            "perturbation": 0.1,
            "thresholds": {"example": 1.0},
            "dataset": {"manifest_sha256": "a" * 64},
            "seeds": [report],
            "assessment": _campaign_assessment([report]),
        }
        summary["evidence_sha256"] = _evidence_sha256(summary)
        (source / f"seed-{seed}.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        (source / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        inputs.append(source)

    result = combine_action_credit_audits(
        input_directories=tuple(inputs),
        evidence_directory=tmp_path / "combined",
    )

    assert result["status"] == "CLEAR_DIAGNOSTIC_PASS"
    assert [report["seed"] for report in result["seeds"]] == [1701, 1901, 2101]
    assert result["evidence_sha256"] == _evidence_sha256(result)
    assert (tmp_path / "combined" / "summary.json").is_file()


def test_combine_rejects_mismatched_dataset_identity(tmp_path: Path) -> None:
    inputs: list[Path] = []
    for index, seed in enumerate((1701, 1901, 2101)):
        source = tmp_path / f"input-{seed}"
        source.mkdir()
        report = _report(seed)
        summary = {
            "status": "INCONCLUSIVE",
            "scope": "train-only-paired-action-credit-audit",
            "partitions_read": ["train"],
            "heldout_read": False,
            "development_read": False,
            "perturbation": 0.1,
            "thresholds": {"example": 1.0},
            "dataset": {"manifest_sha256": str(index)},
            "seeds": [report],
            "assessment": _campaign_assessment([report]),
        }
        summary["evidence_sha256"] = _evidence_sha256(summary)
        (source / f"seed-{seed}.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        (source / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        inputs.append(source)

    with pytest.raises(ValueError, match="dataset"):
        combine_action_credit_audits(
            input_directories=tuple(inputs),
            evidence_directory=tmp_path / "combined",
        )
