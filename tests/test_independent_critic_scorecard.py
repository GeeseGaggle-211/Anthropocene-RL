import pytest

from anthropocene_rl.autonomous_v7.independent_critic_scorecard import (
    ActionAlternative,
    CriticObservation,
    IndependentTarget,
    score_action_rankings,
    score_critic_observations,
)


def target(kind: str, prefix: float, **kwargs) -> IndependentTarget:
    return IndependentTarget(
        prefix_return=prefix,
        tail_kind=kind,
        provenance_id=f"independent-{kind}",
        **kwargs,
    )


def observation(
    unit: str, prediction: float, independent: IndependentTarget, **signals
) -> CriticObservation:
    return CriticObservation(
        independent_unit_id=unit,
        state_id=f"state-{unit}",
        objective="qol_years",
        action_regime="frozen-policy",
        constraint="unconstrained",
        prediction=prediction,
        target=independent,
        units="qol-years",
        return_horizon="400-years",
        **signals,
    )


def alternative(
    unit: str,
    action: str,
    reward: float,
    successor: float,
    independent: IndependentTarget,
) -> ActionAlternative:
    return ActionAlternative(
        independent_unit_id=unit,
        state_id=f"state-{unit}",
        objective="qol_years",
        action_regime="frozen-policy",
        constraint="unconstrained",
        action_id=action,
        immediate_reward=reward,
        gamma=0.5,
        successor_value=successor,
        target=independent,
        units="qol-years",
        return_horizon="400-years",
    )


def test_unresolved_tail_withholds_full_value_metrics_and_duplicate_units_are_rejected() -> None:
    unresolved = target("unresolved", 4.0)
    score = score_critic_observations([observation("fork-1", 9.0, unresolved)])[0]
    assert score["prefixReturnMean"] == 4.0
    assert score["fullValueStatus"] == "withheld-unresolved-tail"
    for name in (
        "targetMean",
        "signedBias",
        "mae",
        "rmse",
        "explainedVariance",
        "calibrationSlope",
        "calibrationIntercept",
        "rawValueSignAgreement",
    ):
        assert score[name] is None

    with pytest.raises(ValueError, match="duplicate independent unit"):
        score_critic_observations(
            [observation("fork-1", 9.0, unresolved), observation("fork-1", 8.0, unresolved)]
        )


def test_terminal_and_estimated_targets_are_never_merged_and_uncertainty_propagates() -> None:
    rows = [
        observation("terminal", 6.0, target("terminal", 5.0)),
        observation(
            "estimated-a",
            8.0,
            target("independent-estimate", 5.0, tail_estimate=2.0, tail_standard_error=3.0),
        ),
        observation(
            "estimated-b",
            8.0,
            target("independent-estimate", 4.0, tail_estimate=2.0, tail_standard_error=4.0),
        ),
    ]
    scores = {row["targetClass"]: row for row in score_critic_observations(rows)}
    assert scores["terminal"]["independentUnitCount"] == 1
    assert scores["terminal"]["signedBias"] == 1.0
    estimated = scores["independent-estimate"]
    assert estimated["independentUnitCount"] == 2
    assert estimated["signedBias"] == pytest.approx(1.5)
    assert estimated["tailEstimateContributionToMeanBiasStandardError"] == pytest.approx(2.5)
    assert estimated["finiteUnitSamplingStandardError"] is None
    assert "conditional tail-estimate" in estimated["uncertaintyScope"]
    assert estimated["fullValueStatus"] == "estimated-with-propagated-uncertainty"
    terminal = scores["terminal"]
    assert terminal["tailEstimateContributionToMeanBiasStandardError"] is None
    assert terminal["finiteUnitSamplingStandardError"] is None


def test_tail_bounds_produce_metric_bounds_without_midpoint_point_metrics() -> None:
    bounded = target("bounded", 10.0, tail_lower=-2.0, tail_upper=6.0)
    score = score_critic_observations([observation("fork-1", 12.0, bounded)])[0]
    assert score["targetLowerMean"] == 8.0
    assert score["targetUpperMean"] == 16.0
    assert score["maeLower"] == 0.0
    assert score["maeUpper"] == 4.0
    assert score["signedBiasLower"] == -4.0
    assert score["signedBiasUpper"] == 4.0
    assert "targetMean" not in score
    assert score["calibrationSlope"] is None
    assert score["fullValueStatus"] == "bounded-no-midpoint-imputation"


def test_one_sided_tail_bound_can_establish_underestimation_without_point_bias() -> None:
    # The caller has explicitly proven a nonnegative tail for this raw objective. The
    # scorecard does not assume that shaped rewards have the same bound.
    lower_bounded = target("bounded", 10.0, tail_lower=2.0)
    score = score_critic_observations(
        [
            observation(
                "fork-1",
                9.0,
                lower_bounded,
                td_residual=1.0,
                gae_advantage=-1.0,
                policy_advantage=0.0,
            )
        ]
    )[0]
    assert score["targetLowerMean"] == 12.0
    assert score["targetUpperMean"] is None
    assert score["signedBiasLower"] is None
    assert score["signedBiasUpper"] == -3.0
    assert score["underestimationEstablishedFraction"] == 1.0
    assert score["maeLower"] == 3.0
    assert score["maeUpper"] is None
    assert score["calibrationSlope"] is None
    assert score["baselineEffectSignIdentifiedFraction"] == 1.0
    assert score["tdResidualSignAgreement"] == 1.0
    assert score["gaeAdvantageSignAgreement"] == 0.0
    assert score["policyAdvantageSignAgreement"] == 0.0


def test_uncertain_tail_withholds_baseline_effect_sign_comparisons() -> None:
    uncertain = target(
        "independent-estimate", 10.0, tail_estimate=0.0, tail_standard_error=5.0
    )
    score = score_critic_observations(
        [observation("fork-1", 10.0, uncertain, td_residual=1.0)]
    )[0]
    assert score["rawValueSignAgreement"] == 1.0
    assert score["baselineEffectSignIdentifiedFraction"] == 0.0
    assert score["baselineEffectAmbiguousCount"] == 1
    assert score["tdResidualSignComparedCount"] == 0
    assert score["tdResidualSignAgreement"] is None


def test_action_ranking_uses_reward_plus_discounted_successor_value() -> None:
    # A state value shared by both actions contains no ranking information. These predictions
    # differ only because each action has a separately evaluated reward/successor pair.
    alternatives = [
        alternative("fork-1", "a", 1.0, 10.0, target("terminal", 3.0)),
        alternative("fork-1", "b", 2.0, 2.0, target("terminal", 4.0)),
    ]
    score = score_action_rankings(alternatives)[0]
    assert alternatives[0].ranking_prediction == 6.0
    assert alternatives[1].ranking_prediction == 3.0
    assert score["pairWeightedAlternativeRankingAccuracy"] == 0.0
    assert score["equalIndependentUnitAlternativeRankingAccuracy"] == 0.0
    assert score["pairWeightedMatchedEffectSignAgreement"] == 0.0
    assert score["predictionDefinition"] == "immediate_reward + gamma * successor_value"


def test_ranking_exposes_exact_ties_and_uncertain_or_unresolved_effects() -> None:
    alternatives = [
        alternative("exact-tie", "a", 1.0, 2.0, target("terminal", 3.0)),
        alternative("exact-tie", "b", 2.0, 0.0, target("terminal", 3.0)),
        alternative(
            "uncertain",
            "a",
            1.0,
            2.0,
            target("independent-estimate", 3.0, tail_estimate=0.0, tail_standard_error=2.0),
        ),
        alternative(
            "uncertain",
            "b",
            2.0,
            0.0,
            target("independent-estimate", 3.0, tail_estimate=0.5, tail_standard_error=2.0),
        ),
        alternative("unresolved", "a", 0.0, 2.0, target("unresolved", 3.0)),
        alternative("unresolved", "b", 0.0, 1.0, target("terminal", 3.0)),
    ]
    score = score_action_rankings(alternatives)[0]
    assert score["independentUnitCount"] == 3
    assert score["pairCount"] == 3
    assert score["identifiedPairCount"] == 1
    assert score["targetTieCount"] == 1
    assert score["ambiguousPairCount"] == 2
    assert score["unresolvedTailPairCount"] == 1


def test_one_sided_bounds_can_identify_action_ranking_without_point_targets() -> None:
    alternatives = [
        alternative("bounded", "a", 3.0, 4.0, target("bounded", 10.0, tail_lower=0.0)),
        alternative("bounded", "b", 1.0, 2.0, target("bounded", 8.0, tail_upper=0.0)),
    ]
    score = score_action_rankings(alternatives)[0]
    assert score["identifiedPairCount"] == 1
    assert score["pairWeightedIdentifiedFraction"] == 1.0
    assert score["equalIndependentUnitIdentifiedFraction"] == 1.0
    assert score["pairWeightedMatchedEffectSignIdentifiedFraction"] == 1.0
    assert score["pairWeightedAlternativeRankingAccuracy"] == 1.0


def test_action_ranking_reports_pair_weighted_and_equal_unit_metrics_separately() -> None:
    alternatives = [
        alternative("many", "a", 3.0, 0.0, target("terminal", 3.0)),
        alternative("many", "b", 2.0, 0.0, target("terminal", 2.0)),
        alternative("many", "c", 1.0, 0.0, target("terminal", 1.0)),
        alternative("few", "a", 2.0, 0.0, target("terminal", 0.0)),
        alternative("few", "b", 1.0, 0.0, target("terminal", 1.0)),
    ]
    score = score_action_rankings(alternatives)[0]
    assert score["pairWeightedCorrectNumerator"] == 3
    assert score["pairWeightedIdentifiedDenominator"] == 4
    assert score["pairWeightedAlternativeRankingAccuracy"] == 0.75
    assert score["equalIndependentUnitRankingScoreSum"] == 1.0
    assert score["equalIndependentUnitRankingDenominator"] == 2
    assert score["equalIndependentUnitAlternativeRankingAccuracy"] == 0.5
    assert score["equalIndependentUnitIdentifiedFractionExcludedCount"] == 0
    assert score["equalIndependentUnitSignIdentifiedFractionScoreSum"] == 2.0
    assert score["equalIndependentUnitSignIdentifiedFractionDenominator"] == 2
    assert score["equalIndependentUnitSignIdentifiedFractionExcludedCount"] == 0
    assert score["pairWeightedRankCorrelation"] == 0.5
    assert score["equalIndependentUnitRankCorrelation"] == 0.0


def test_self_bootstrapped_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="critic under test"):
        IndependentTarget(
            prefix_return=1.0,
            tail_kind="terminal",
            provenance_id="bad",
            independent_from_critic=False,
        )
