"""Pure scoring primitives for Gate 3 independent-critic evidence.

The functions in this module do not collect continuations or evaluate a checkpoint.  They
only score already-bound, independent evidence.  In particular, an unresolved tail is not
turned into a point target and action alternatives are ranked with
``reward + gamma * V(successor)`` rather than a state value shared by all actions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from math import isfinite, sqrt
from typing import Literal

import numpy as np

TailKind = Literal["terminal", "independent-estimate", "bounded", "unresolved"]


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class IndependentTarget:
    """A realized prefix plus a tail that is independent of the critic under test.

    ``terminal`` has an exactly zero tail. ``independent-estimate`` is a separately
    labelled point estimate and must carry its standard error. ``bounded`` is interval
    evidence only. ``unresolved`` deliberately identifies no full-value target.
    """

    prefix_return: float
    tail_kind: TailKind
    provenance_id: str
    independent_from_critic: bool = True
    tail_estimate: float | None = None
    tail_standard_error: float | None = None
    tail_lower: float | None = None
    tail_upper: float | None = None

    def __post_init__(self) -> None:
        _finite(self.prefix_return, "prefix_return")
        if not self.provenance_id:
            raise ValueError("independent target requires provenance_id")
        if not self.independent_from_critic:
            raise ValueError("target may not bootstrap from the critic under test")
        if self.tail_kind == "terminal":
            self._require_absent("tail_estimate", self.tail_estimate)
            self._require_absent("tail_standard_error", self.tail_standard_error)
            self._require_absent("tail_lower", self.tail_lower)
            self._require_absent("tail_upper", self.tail_upper)
        elif self.tail_kind == "independent-estimate":
            if self.tail_estimate is None or self.tail_standard_error is None:
                raise ValueError("independent tail estimate requires estimate and standard error")
            _finite(self.tail_estimate, "tail_estimate")
            standard_error = _finite(self.tail_standard_error, "tail_standard_error")
            if standard_error < 0:
                raise ValueError("tail_standard_error must be nonnegative")
            self._require_absent("tail_lower", self.tail_lower)
            self._require_absent("tail_upper", self.tail_upper)
        elif self.tail_kind == "bounded":
            if self.tail_lower is None and self.tail_upper is None:
                raise ValueError("bounded tail requires at least one finite bound")
            lower = (
                _finite(self.tail_lower, "tail_lower") if self.tail_lower is not None else None
            )
            upper = (
                _finite(self.tail_upper, "tail_upper") if self.tail_upper is not None else None
            )
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("tail_lower exceeds tail_upper")
            self._require_absent("tail_estimate", self.tail_estimate)
            self._require_absent("tail_standard_error", self.tail_standard_error)
        elif self.tail_kind == "unresolved":
            self._require_absent("tail_estimate", self.tail_estimate)
            self._require_absent("tail_standard_error", self.tail_standard_error)
            self._require_absent("tail_lower", self.tail_lower)
            self._require_absent("tail_upper", self.tail_upper)
        else:
            raise ValueError(f"unsupported tail kind: {self.tail_kind}")

    @staticmethod
    def _require_absent(name: str, value: float | None) -> None:
        if value is not None:
            raise ValueError(f"{name} is not valid for this tail kind")

    @property
    def point_value(self) -> float | None:
        if self.tail_kind == "terminal":
            return float(self.prefix_return)
        if self.tail_kind == "independent-estimate":
            assert self.tail_estimate is not None
            return float(self.prefix_return + self.tail_estimate)
        return None

    def value_bounds(
        self, uncertainty_z: float = 1.96
    ) -> tuple[float | None, float | None]:
        """Return independently identified full-value bounds without midpoint imputation."""
        uncertainty_z = _finite(uncertainty_z, "uncertainty_z")
        if uncertainty_z < 0:
            raise ValueError("uncertainty_z must be nonnegative")
        point = self.point_value
        if self.tail_kind == "terminal":
            assert point is not None
            return point, point
        if self.tail_kind == "independent-estimate":
            assert point is not None and self.tail_standard_error is not None
            radius = uncertainty_z * self.tail_standard_error
            return point - radius, point + radius
        if self.tail_kind == "bounded":
            lower = (
                self.prefix_return + self.tail_lower if self.tail_lower is not None else None
            )
            upper = (
                self.prefix_return + self.tail_upper if self.tail_upper is not None else None
            )
            return lower, upper
        return None, None

    def value_interval(self, uncertainty_z: float = 1.96) -> tuple[float, float] | None:
        """Return a two-sided interval when both full-value bounds are identified."""
        lower, upper = self.value_bounds(uncertainty_z)
        return (lower, upper) if lower is not None and upper is not None else None


@dataclass(frozen=True)
class CriticObservation:
    """One already-aggregated independent scoring unit.

    Callers must aggregate descendants before constructing the row. Repeating village rows
    with the same independent-unit identity is rejected rather than counted as replication.
    """

    independent_unit_id: str
    state_id: str
    objective: str
    action_regime: str
    constraint: str
    prediction: float
    target: IndependentTarget
    units: str
    return_horizon: str
    td_residual: float | None = None
    gae_advantage: float | None = None
    policy_advantage: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "independent_unit_id",
            "state_id",
            "objective",
            "action_regime",
            "constraint",
            "units",
            "return_horizon",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        _finite(self.prediction, "prediction")
        for name in ("td_residual", "gae_advantage", "policy_advantage"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)


@dataclass(frozen=True)
class ActionAlternative:
    """One action alternative at a frozen state with a matched independent target."""

    independent_unit_id: str
    state_id: str
    objective: str
    action_regime: str
    constraint: str
    action_id: str
    immediate_reward: float
    gamma: float
    successor_value: float
    target: IndependentTarget
    units: str
    return_horizon: str

    def __post_init__(self) -> None:
        for name in (
            "independent_unit_id",
            "state_id",
            "objective",
            "action_regime",
            "constraint",
            "action_id",
            "units",
            "return_horizon",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        _finite(self.immediate_reward, "immediate_reward")
        gamma = _finite(self.gamma, "gamma")
        _finite(self.successor_value, "successor_value")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")

    @property
    def ranking_prediction(self) -> float:
        return float(self.immediate_reward + self.gamma * self.successor_value)


def _point_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    residual = prediction - target
    target_variance = float(target.var())
    prediction_variance = float(prediction.var())
    slope = (
        float(np.mean((prediction - prediction.mean()) * (target - target.mean())))
        / prediction_variance
        if prediction_variance > 0
        else None
    )
    return {
        "predictionMean": float(prediction.mean()),
        "targetMean": float(target.mean()),
        "signedBias": float(residual.mean()),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "explainedVariance": (
            1.0 - float(residual.var()) / target_variance if target_variance > 0 else None
        ),
        "calibrationSlope": slope,
        "calibrationIntercept": (
            float(target.mean() - slope * prediction.mean()) if slope is not None else None
        ),
        "rawValueSignAgreement": float(np.mean(np.sign(prediction) == np.sign(target))),
    }


def _base_score(records: Sequence[CriticObservation]) -> dict[str, object]:
    first = records[0]
    return {
        "objective": first.objective,
        "actionRegime": first.action_regime,
        "constraint": first.constraint,
        "targetClass": first.target.tail_kind,
        "units": first.units,
        "returnHorizon": first.return_horizon,
        "independentUnitCount": len(records),
        "rowCount": len(records),
        "tailCoverage": float(np.mean([row.target.tail_kind != "unresolved" for row in records])),
        "prefixReturnMean": float(np.mean([row.target.prefix_return for row in records])),
    }


def _baseline_effect_relation(
    row: CriticObservation, *, tolerance: float, uncertainty_z: float = 1.96
) -> int | None:
    lower, upper = row.target.value_bounds(uncertainty_z)
    effect_lower = lower - row.prediction if lower is not None else None
    effect_upper = upper - row.prediction if upper is not None else None
    if effect_lower is not None and effect_lower > tolerance:
        return 1
    if effect_upper is not None and effect_upper < -tolerance:
        return -1
    if (
        effect_lower is not None
        and effect_upper is not None
        and effect_lower == effect_upper
        and abs(effect_lower) <= tolerance
    ):
        return 0
    return None


def _signal_agreement(
    records: Sequence[CriticObservation], *, tolerance: float, uncertainty_z: float
) -> dict[str, object]:
    target_relations = [
        _baseline_effect_relation(
            row, tolerance=tolerance, uncertainty_z=uncertainty_z
        )
        for row in records
    ]
    output: dict[str, object] = {
        "baselineEffectDefinition": "independent_full_return - state_value_prediction",
        "baselineEffectSignIdentifiedCount": sum(
            relation is not None for relation in target_relations
        ),
        "baselineEffectSignIdentifiedFraction": float(
            np.mean([relation is not None for relation in target_relations])
        ),
        "baselineEffectTieCount": sum(relation == 0 for relation in target_relations),
        "baselineEffectAmbiguousCount": sum(
            relation is None for relation in target_relations
        ),
    }
    for attribute, label in (
        ("td_residual", "tdResidual"),
        ("gae_advantage", "gaeAdvantage"),
        ("policy_advantage", "policyAdvantage"),
    ):
        supplied = [getattr(row, attribute) for row in records]
        comparisons = [
            (relation, value)
            for relation, value in zip(target_relations, supplied, strict=True)
            if relation is not None and value is not None
        ]
        agreements = [
            relation == _relation(float(value), 0.0, tolerance)
            for relation, value in comparisons
        ]
        output[f"{label}MissingCount"] = sum(value is None for value in supplied)
        output[f"{label}SignComparedCount"] = len(comparisons)
        output[f"{label}SignComparedFraction"] = len(comparisons) / len(records)
        output[f"{label}SignAgreement"] = (
            float(np.mean(agreements)) if agreements else None
        )
    return output


def score_critic_observations(
    rows: Sequence[CriticObservation], *, sign_tolerance: float = 0.0, uncertainty_z: float = 1.96
) -> list[dict[str, object]]:
    """Score independent critic rows, keeping every tail evidence class separate."""
    sign_tolerance = _finite(sign_tolerance, "sign_tolerance")
    if sign_tolerance < 0:
        raise ValueError("sign_tolerance must be nonnegative")
    uncertainty_z = _finite(uncertainty_z, "uncertainty_z")
    if uncertainty_z < 0:
        raise ValueError("uncertainty_z must be nonnegative")
    seen: set[tuple[str, str, str, str]] = set()
    groups: dict[tuple[str, ...], list[CriticObservation]] = {}
    for row in rows:
        identity = (
            row.independent_unit_id,
            row.objective,
            row.action_regime,
            row.constraint,
        )
        if identity in seen:
            raise ValueError("duplicate independent unit; aggregate descendants before scoring")
        seen.add(identity)
        group_key = (
            row.objective,
            row.action_regime,
            row.constraint,
            row.target.tail_kind,
            row.units,
            row.return_horizon,
        )
        groups.setdefault(group_key, []).append(row)

    output: list[dict[str, object]] = []
    for score_group_key in sorted(groups):
        records = sorted(groups[score_group_key], key=lambda item: item.independent_unit_id)
        result = _base_score(records)
        kind = records[0].target.tail_kind
        prediction = np.asarray([row.prediction for row in records], dtype=np.float64)
        result["predictionMean"] = float(prediction.mean())
        if kind in {"terminal", "independent-estimate"}:
            target = np.asarray([row.target.point_value for row in records], dtype=np.float64)
            result.update(_point_metrics(prediction, target))
            if kind == "independent-estimate":
                standard_errors = np.asarray(
                    [row.target.tail_standard_error for row in records], dtype=np.float64
                )
                result["targetUncertaintyKind"] = "independent-standard-error"
                result["meanTailEstimateStandardError"] = float(standard_errors.mean())
                result["tailEstimateContributionToMeanBiasStandardError"] = float(
                    np.sqrt(np.sum(standard_errors**2)) / len(standard_errors)
                )
                result["finiteUnitSamplingStandardError"] = None
                result["uncertaintyScope"] = (
                    "conditional tail-estimate error propagation only; finite-unit "
                    "sampling uncertainty is not estimated"
                )
                result["fullValueStatus"] = "estimated-with-propagated-uncertainty"
            else:
                result["targetUncertaintyKind"] = "none-terminal"
                result["meanTailEstimateStandardError"] = None
                result["tailEstimateContributionToMeanBiasStandardError"] = None
                result["finiteUnitSamplingStandardError"] = None
                result["uncertaintyScope"] = (
                    "terminal tail is exact; finite-unit sampling uncertainty is not estimated"
                )
                result["fullValueStatus"] = "point-identified-terminal"
        elif kind == "bounded":
            bounds = [row.target.value_bounds() for row in records]
            lower_known = np.asarray([bound[0] is not None for bound in bounds], dtype=bool)
            upper_known = np.asarray([bound[1] is not None for bound in bounds], dtype=bool)
            lower = np.asarray(
                [bound[0] if bound[0] is not None else np.nan for bound in bounds],
                dtype=np.float64,
            )
            upper = np.asarray(
                [bound[1] if bound[1] is not None else np.nan for bound in bounds],
                dtype=np.float64,
            )
            error_lower = prediction - upper
            error_upper = prediction - lower
            absolute_lower = np.zeros(len(records), dtype=np.float64)
            absolute_lower[lower_known & (prediction < lower)] = (
                lower[lower_known & (prediction < lower)]
                - prediction[lower_known & (prediction < lower)]
            )
            absolute_lower[upper_known & (prediction > upper)] = np.maximum(
                absolute_lower[upper_known & (prediction > upper)],
                prediction[upper_known & (prediction > upper)]
                - upper[upper_known & (prediction > upper)],
            )
            two_sided = lower_known & upper_known
            absolute_upper = np.maximum(np.abs(error_lower), np.abs(error_upper))
            underestimation = lower_known & (error_upper < 0)
            overestimation = upper_known & (error_lower > 0)
            result.update(
                {
                    "targetLowerMean": (
                        float(lower[lower_known].mean()) if lower_known.any() else None
                    ),
                    "targetUpperMean": (
                        float(upper[upper_known].mean()) if upper_known.any() else None
                    ),
                    "tailLowerBoundCoverage": float(lower_known.mean()),
                    "tailUpperBoundCoverage": float(upper_known.mean()),
                    "signedBiasLower": (
                        float(error_lower.mean()) if upper_known.all() else None
                    ),
                    "signedBiasUpper": (
                        float(error_upper.mean()) if lower_known.all() else None
                    ),
                    "maeLower": float(absolute_lower.mean()),
                    "maeUpper": (
                        float(absolute_upper.mean()) if two_sided.all() else None
                    ),
                    "rmseLower": float(np.sqrt(np.mean(absolute_lower**2))),
                    "rmseUpper": (
                        float(np.sqrt(np.mean(absolute_upper**2))) if two_sided.all() else None
                    ),
                    "underestimationEstablishedCount": int(underestimation.sum()),
                    "underestimationEstablishedFraction": float(underestimation.mean()),
                    "overestimationEstablishedCount": int(overestimation.sum()),
                    "overestimationEstablishedFraction": float(overestimation.mean()),
                    "explainedVariance": None,
                    "calibrationSlope": None,
                    "calibrationIntercept": None,
                    "rawValueSignAgreement": None,
                    "fullValueStatus": "bounded-no-midpoint-imputation",
                }
            )
        else:
            result.update(
                {
                    "targetMean": None,
                    "signedBias": None,
                    "mae": None,
                    "rmse": None,
                    "explainedVariance": None,
                    "calibrationSlope": None,
                    "calibrationIntercept": None,
                    "rawValueSignAgreement": None,
                    "fullValueStatus": "withheld-unresolved-tail",
                    "exclusionReason": "full value is not identified by the realized prefix",
                }
            )
        result.update(
            _signal_agreement(
                records, tolerance=sign_tolerance, uncertainty_z=uncertainty_z
            )
        )
        output.append(result)
    return output


def _relation(first: float, second: float, tolerance: float) -> int:
    difference = first - second
    if difference > tolerance:
        return 1
    if difference < -tolerance:
        return -1
    return 0


def _target_relation(
    first: IndependentTarget,
    second: IndependentTarget,
    tolerance: float,
    uncertainty_z: float,
) -> int | None:
    first_lower, first_upper = first.value_bounds(uncertainty_z)
    second_lower, second_upper = second.value_bounds(uncertainty_z)
    if (
        first_lower is not None
        and second_upper is not None
        and first_lower > second_upper + tolerance
    ):
        return 1
    if (
        second_lower is not None
        and first_upper is not None
        and second_lower > first_upper + tolerance
    ):
        return -1
    first_point, second_point = first.point_value, second.point_value
    if (
        first_lower is not None
        and first_upper is not None
        and second_lower is not None
        and second_upper is not None
        and first_lower == first_upper
        and second_lower == second_upper
        and first_point is not None
        and second_point is not None
    ):
        return _relation(first_point, second_point, tolerance)
    return None


def score_action_rankings(
    alternatives: Sequence[ActionAlternative],
    *,
    prediction_tolerance: float = 0.0,
    target_tolerance: float = 0.0,
    uncertainty_z: float = 1.96,
) -> list[dict[str, object]]:
    """Score pairwise action rankings from one-step successor predictions.

    Ambiguous effects are excluded explicitly. Counts are frozen-state continuation units,
    never the number of village rows inside a continuation.
    """
    prediction_tolerance = _finite(prediction_tolerance, "prediction_tolerance")
    target_tolerance = _finite(target_tolerance, "target_tolerance")
    if prediction_tolerance < 0 or target_tolerance < 0:
        raise ValueError("tie tolerances must be nonnegative")
    groups: dict[tuple[str, ...], list[ActionAlternative]] = {}
    for item in alternatives:
        group_key = (
            item.objective,
            item.action_regime,
            item.constraint,
            item.units,
            item.return_horizon,
            item.independent_unit_id,
            item.state_id,
        )
        groups.setdefault(group_key, []).append(item)

    slice_rows: dict[tuple[str, ...], list[dict[str, int]]] = {}
    seen_units: set[tuple[str, str]] = set()
    for ranking_group_key in sorted(groups):
        choices = sorted(groups[ranking_group_key], key=lambda item: item.action_id)
        if len(choices) < 2:
            raise ValueError("action-ranking unit requires at least two matched alternatives")
        if len({item.action_id for item in choices}) != len(choices):
            raise ValueError("action-ranking unit repeats an action identity")
        unit_key = (choices[0].independent_unit_id, choices[0].objective)
        if unit_key in seen_units:
            raise ValueError("independent action-ranking unit appears in multiple states/slices")
        seen_units.add(unit_key)
        counts = {
            "pairs": 0,
            "identified": 0,
            "correct": 0,
            "concordant": 0,
            "discordant": 0,
            "predicted_ties": 0,
            "target_ties": 0,
            "both_ties": 0,
            "ambiguous": 0,
            "unresolved_tail": 0,
        }
        for first, second in combinations(choices, 2):
            counts["pairs"] += 1
            predicted = _relation(
                first.ranking_prediction, second.ranking_prediction, prediction_tolerance
            )
            actual = _target_relation(first.target, second.target, target_tolerance, uncertainty_z)
            if actual is None:
                counts["ambiguous"] += 1
                counts["unresolved_tail"] += int(
                    first.target.tail_kind == "unresolved"
                    or second.target.tail_kind == "unresolved"
                )
                continue
            counts["identified"] += 1
            counts["predicted_ties"] += int(predicted == 0)
            counts["target_ties"] += int(actual == 0)
            counts["both_ties"] += int(predicted == 0 and actual == 0)
            counts["correct"] += int(predicted == actual)
            if predicted != 0 and actual != 0:
                counts["concordant" if predicted == actual else "discordant"] += 1
        slice_rows.setdefault(ranking_group_key[:5], []).append(counts)

    output: list[dict[str, object]] = []
    for slice_key in sorted(slice_rows):
        unit_counts = slice_rows[slice_key]
        totals = {name: sum(row[name] for row in unit_counts) for name in unit_counts[0]}
        identified = totals["identified"]
        comparable = totals["concordant"] + totals["discordant"]
        tie_denominator_prediction = (
            comparable + totals["target_ties"] - totals["both_ties"]
        )
        tie_denominator_target = (
            comparable + totals["predicted_ties"] - totals["both_ties"]
        )
        rank_denominator = sqrt(tie_denominator_prediction * tie_denominator_target)
        unit_ranking_scores = [
            row["correct"] / row["identified"] for row in unit_counts if row["identified"]
        ]
        unit_identified_fractions = [
            row["identified"] / row["pairs"] for row in unit_counts if row["pairs"]
        ]
        unit_sign_scores = [
            row["concordant"] / (row["concordant"] + row["discordant"])
            for row in unit_counts
            if row["concordant"] + row["discordant"]
        ]
        unit_sign_identified_fractions = [
            (row["concordant"] + row["discordant"]) / row["pairs"]
            for row in unit_counts
            if row["pairs"]
        ]
        unit_rank_correlations = []
        for row in unit_counts:
            unit_comparable = row["concordant"] + row["discordant"]
            unit_prediction_denominator = (
                unit_comparable + row["target_ties"] - row["both_ties"]
            )
            unit_target_denominator = (
                unit_comparable + row["predicted_ties"] - row["both_ties"]
            )
            unit_denominator = sqrt(
                unit_prediction_denominator * unit_target_denominator
            )
            if unit_denominator:
                unit_rank_correlations.append(
                    (row["concordant"] - row["discordant"]) / unit_denominator
                )
        objective, regime, constraint, units, horizon = slice_key
        output.append(
            {
                "objective": objective,
                "actionRegime": regime,
                "constraint": constraint,
                "units": units,
                "returnHorizon": horizon,
                "independentUnitCount": len(unit_counts),
                "pairCount": totals["pairs"],
                "identifiedPairCount": identified,
                "pairWeightedIdentifiedNumerator": identified,
                "pairWeightedPairDenominator": totals["pairs"],
                "pairWeightedIdentifiedFraction": (
                    identified / totals["pairs"] if totals["pairs"] else None
                ),
                "equalIndependentUnitIdentifiedFractionScoreSum": float(
                    sum(unit_identified_fractions)
                ),
                "equalIndependentUnitIdentifiedFractionDenominator": len(
                    unit_identified_fractions
                ),
                "equalIndependentUnitIdentifiedFraction": (
                    float(np.mean(unit_identified_fractions))
                    if unit_identified_fractions
                    else None
                ),
                "equalIndependentUnitIdentifiedFractionExcludedCount": (
                    len(unit_counts) - len(unit_identified_fractions)
                ),
                "ambiguousPairCount": totals["ambiguous"],
                "unresolvedTailPairCount": totals["unresolved_tail"],
                "predictedTieCount": totals["predicted_ties"],
                "targetTieCount": totals["target_ties"],
                "pairWeightedCorrectNumerator": totals["correct"],
                "pairWeightedIdentifiedDenominator": identified,
                "pairWeightedAlternativeRankingAccuracy": (
                    totals["correct"] / identified if identified else None
                ),
                "equalIndependentUnitRankingScoreSum": (
                    float(sum(unit_ranking_scores)) if unit_ranking_scores else None
                ),
                "equalIndependentUnitRankingDenominator": len(unit_ranking_scores),
                "equalIndependentUnitAlternativeRankingAccuracy": (
                    float(np.mean(unit_ranking_scores)) if unit_ranking_scores else None
                ),
                "equalIndependentUnitRankingExcludedCount": (
                    len(unit_counts) - len(unit_ranking_scores)
                ),
                "pairWeightedSignAgreementNumerator": totals["concordant"],
                "pairWeightedSignAgreementDenominator": comparable,
                "pairWeightedMatchedEffectSignAgreement": (
                    totals["concordant"] / comparable if comparable else None
                ),
                "pairWeightedSignIdentifiedNumerator": comparable,
                "pairWeightedSignIdentifiedDenominator": totals["pairs"],
                "pairWeightedMatchedEffectSignIdentifiedFraction": (
                    comparable / totals["pairs"] if totals["pairs"] else None
                ),
                "equalIndependentUnitSignAgreementScoreSum": (
                    float(sum(unit_sign_scores)) if unit_sign_scores else None
                ),
                "equalIndependentUnitSignAgreementDenominator": len(unit_sign_scores),
                "equalIndependentUnitMatchedEffectSignAgreement": (
                    float(np.mean(unit_sign_scores)) if unit_sign_scores else None
                ),
                "equalIndependentUnitSignAgreementExcludedCount": (
                    len(unit_counts) - len(unit_sign_scores)
                ),
                "equalIndependentUnitMatchedEffectSignIdentifiedFraction": (
                    float(np.mean(unit_sign_identified_fractions))
                    if unit_sign_identified_fractions
                    else None
                ),
                "equalIndependentUnitSignIdentifiedFractionScoreSum": (
                    float(sum(unit_sign_identified_fractions))
                    if unit_sign_identified_fractions
                    else None
                ),
                "equalIndependentUnitSignIdentifiedFractionDenominator": len(
                    unit_sign_identified_fractions
                ),
                "equalIndependentUnitSignIdentifiedFractionExcludedCount": (
                    len(unit_counts) - len(unit_sign_identified_fractions)
                ),
                "pairWeightedRankCorrelation": (
                    (totals["concordant"] - totals["discordant"]) / rank_denominator
                    if rank_denominator
                    else None
                ),
                "equalIndependentUnitRankCorrelationScoreSum": (
                    float(sum(unit_rank_correlations))
                    if unit_rank_correlations
                    else None
                ),
                "equalIndependentUnitRankCorrelationDenominator": len(
                    unit_rank_correlations
                ),
                "equalIndependentUnitRankCorrelation": (
                    float(np.mean(unit_rank_correlations))
                    if unit_rank_correlations
                    else None
                ),
                "equalIndependentUnitRankCorrelationExcludedCount": (
                    len(unit_counts) - len(unit_rank_correlations)
                ),
                "predictionDefinition": "immediate_reward + gamma * successor_value",
                "targetDefinition": "independent matched continuation full return",
            }
        )
    return output
