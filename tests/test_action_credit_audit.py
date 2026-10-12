from anthropocene_rl.training.action_credit_audit import _campaign_assessment, _direction


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
