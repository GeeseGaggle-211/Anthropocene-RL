from __future__ import annotations

import copy
import math

import pytest
from test_economic_diagnostics import update_pair, write_artifact
from test_gate2_accounting import transition_v2

from anthropocene_rl.autonomous_v7 import economic_trace_summary


def _artifact(tmp_path, *, debug: bool = True, actual_objective_fields: bool = True):
    first = transition_v2(turn=1)
    second = transition_v2(turn=2)
    for row, person_years in ((first, 17.0), (second, 19.0)):
        row["economicDiagnostics"]["domains"]["population"]["personYears"] = person_years
        row["economicDiagnostics"]["units"]["domains.population.personYears"] = "person-years"
        row["economicDiagnostics"]["domains"]["trade"].update(
            requisitions=1,
            honoured=0,
            countered=0,
            ignored=1,
            settledTrades=0,
            refund=0.25,
            storageRejection=0.5,
        )
    second["objectives"]["welfare"]["advantage"] = 1.0
    second["finalPolicyAdvantage"] = 0.5
    if actual_objective_fields:
        first["objectives"]["welfare"].update(
            weightedAdvantageContribution=-0.25,
            preClipEffectiveCombinationWeight=0.5,
            effectiveCombinationWeight=0.5,
            zeroShapedRewardNonzeroBootstrap=True,
        )
        second["objectives"]["welfare"].update(
            weightedAdvantageContribution=0.25,
            preClipEffectiveCombinationWeight=0.25,
            effectiveCombinationWeight=0.25,
            zeroShapedRewardNonzeroBootstrap=False,
        )
    updates = update_pair([first, second])
    if debug:
        updates[1]["optimizerDebug"] = {
            "schemaVersion": 1,
            "enabled": True,
            "scope": {
                "epoch": 0,
                "minibatch": 0,
                "minibatchRowCount": 2,
                "expensiveAudit": True,
                "selection": "first-rows-of-actual-minibatch",
                "requestedRowCap": 64,
                "sampledRowCount": 2,
                "auditTransitionIds": [first["transitionId"], second["transitionId"]],
            },
            "lossContributions": {
                "policyClippedCombined": -0.2,
                "entropyRegularizerWeighted": -0.01,
                "teacherAnchorWeighted": 0.03,
                "searchImitationWeighted": None,
                "totalActorLoss": -0.18,
            },
            "actualClip": {
                "actorStepPerformed": True,
                "actorStepRolledBack": False,
                "actorPreClipNorm": 0.5,
                "actorClipScale": 1.0,
                "criticPreClipNorm": 1.5,
                "criticClipScale": 2 / 3,
                "costPreClipNorm": 0.25,
                "costClipScale": 1.0,
            },
            "attributionScope": "pre-Adam gradient geometry; descriptive only",
            "sampledPolicyShift": {
                "measurementPhase": "pre-actor-step",
                "rowSelection": "entire-current-minibatch",
                "rowCount": 2,
                "nonnegativeKlEstimateMean": 0.005,
                "logRatioMean": 0.0,
                "logRatioMin": -0.1,
                "logRatioP10": -0.08,
                "logRatioMedian": 0.0,
                "logRatioP90": 0.08,
                "logRatioMax": 0.1,
                "interpretation": "sampled-action policy shift only",
            },
            "sameSampledAction": {
                "postStepSampledPolicyShift": {
                    "measurementPhase": "post-actor-step",
                    "rowSelection": "bounded-fixed-audit-rows",
                    "rowCount": 2,
                    "signedApproxKlMean": 0.0,
                    "nonnegativeKlEstimateMean": math.cosh(0.1) - 1.0,
                    "sampledProbabilityRatio": {
                        "definition": "exp(jointAfter-jointBefore)",
                        "count": 2,
                        "min": math.exp(-0.1),
                        "max": math.exp(0.1),
                    },
                    "interpretation": (
                        "post-step shift on bounded fixed audit rows; not exact KL"
                    ),
                },
                "joint": {"beforeMean": 0.5, "afterMean": 0.5, "meanDelta": 0.0},
                "perHead": {},
                "rows": [
                    {
                        "transitionId": first["transitionId"],
                        "jointBefore": 0.4,
                        "jointAfter": 0.5,
                        "jointDelta": 0.1,
                        "perHead": {
                            "strategic": {"before": 0.4, "after": 0.5, "delta": 0.1}
                        },
                        "perCoordinate": {
                            "strategic": {
                                "cultivation": {
                                    "nativeContractIndex": 0,
                                    "before": 0.4,
                                    "after": 0.5,
                                    "delta": 0.1,
                                    "beforeEntropy": 0.2,
                                    "afterEntropy": 0.19,
                                    "entropyDelta": -0.01,
                                    "policyScored": True,
                                    "nativeRecordedActive": False,
                                    "activityRelation": "scored-but-native-recorded-inactive",
                                },
                                "native-mask-unavailable": {
                                    "nativeContractIndex": 1,
                                    "before": 0.0,
                                    "after": 0.0,
                                    "delta": 0.0,
                                    "beforeEntropy": 0.0,
                                    "afterEntropy": 0.0,
                                    "entropyDelta": 0.0,
                                    "policyScored": True,
                                    "nativeRecordedActive": None,
                                    "activityRelation": "native-recorded-activity-unavailable",
                                }
                            }
                        },
                    },
                    {
                        "transitionId": second["transitionId"],
                        "jointBefore": 0.6,
                        "jointAfter": 0.5,
                        "jointDelta": -0.1,
                        "perHead": {
                            "strategic": {"before": 0.6, "after": 0.5, "delta": -0.1}
                        },
                        "perCoordinate": {
                            "strategic": {
                                "cultivation": {
                                    "nativeContractIndex": 0,
                                    "before": 0.6,
                                    "after": 0.5,
                                    "delta": -0.1,
                                    "beforeEntropy": 0.2,
                                    "afterEntropy": 0.19,
                                    "entropyDelta": -0.01,
                                    "policyScored": True,
                                    "nativeRecordedActive": True,
                                    "activityRelation": "scored-and-native-recorded-active",
                                }
                            }
                        },
                    },
                ],
            },
            "actorGradientGeometry": {
                "components": {"policyClippedCombined": {"loss": -0.2, "norm": 0.5}},
                "pairs": [],
            },
            "vectorCriticSharedEncoderGradientGeometry": None,
        }
    root = tmp_path / "trace"
    write_artifact(root, [first, second], updates=updates)
    return root


def test_summary_joins_control_outcomes_critic_and_optimizer_context(tmp_path):
    summary = economic_trace_summary.summarize_trace(_artifact(tmp_path))
    flow = summary["economicControlFlow"]
    assert flow["accountingTransitionDenominator"] == 2
    assert flow["completeAccountingTransitionCount"] == 2
    control = flow["controls"][0]
    assert control["actorRequested"]["count"] == 2
    assert control["phases"]["compilerGrant"]["sum"] == 1.4
    assert control["phaseUnits"]["compilerGrant"] == "operation-fraction"
    ratio = control["phaseRatios"]["actorCappedRequest_to_compilerGrant"]
    assert ratio["value"] == pytest.approx(0.875)
    assert ratio["denominatorUnit"] == ratio["numeratorUnit"] == "operation-fraction"

    outcomes = summary["outcomes"]
    assert outcomes["endingPopulation"]["count"] == 2
    assert outcomes["populationYears"]["sum"] == 36.0
    assert outcomes["objectives"]["welfare"]["tdResidual"]["count"] == 2
    objective = outcomes["objectives"]["welfare"]
    assert objective["weightedAdvantageContribution"]["sum"] == pytest.approx(0.0)
    assert objective["effectiveCombinationWeight"]["mean"] == pytest.approx(0.375)
    assert objective["preClipEffectiveCombinationWeight"]["count"] == 2
    bootstrap_flag = objective["zeroShapedRewardNonzeroBootstrap"]
    assert bootstrap_flag["observedCount"] == 2
    assert bootstrap_flag["trueCount"] == bootstrap_flag["falseCount"] == 1
    assert bootstrap_flag["unavailableReason"] is None
    assert outcomes["tradeLifecycle"]["endpointRecordTypeCounts"] == {"eligibility": 2}
    assert outcomes["tradeLifecycle"]["uniqueLifecycleRecordTypeCounts"] == {
        "eligibility": 1
    }
    assert outcomes["tradeLifecycle"]["reasonCounts"] == {
        "partnerSearchReason:no-eligible-remembered-partner-in-range": 2
    }
    assert outcomes["tradeLifecycle"]["domainEndpointCounts"]["refund"]["sum"] == 0.5
    assert (
        outcomes["tradeLifecycle"]["domainEndpointCounts"]["storageRejection"]["sum"]
        == 1.0
    )
    assert outcomes["criticInterpretation"] == "internal-bootstrap-or-GAE-target-error-only"
    assert outcomes["independentCriticCalibration"] is None

    optimizer = summary["optimizerUpdates"]
    assert optimizer["alignmentDenominator"] == 2
    assert optimizer["descriptiveSignAlignment"]["negativeAdvantageReinforcement"] == 1
    assert optimizer["descriptiveSignAlignment"]["positiveAdvantageSuppression"] == 1
    assert optimizer["positiveAdvantageBenefit"] is None
    assert optimizer["causalAttribution"] is None
    assert optimizer["teacherAndGradientContext"][0]["lossContributions"][
        "teacherAnchorWeighted"
    ] == 0.03
    coordinate = optimizer["perCoordinateDescriptiveSignAlignment"]["strategic.cultivation"]
    assert coordinate["policyScored"] == 2
    assert coordinate["nativeRecordedActive"] == 1
    assert coordinate["nativeRecordedInactive"] == 1
    unavailable_coordinate = optimizer["perCoordinateDescriptiveSignAlignment"][
        "strategic.native-mask-unavailable"
    ]
    assert unavailable_coordinate["nativeRecordedActivityUnavailable"] == 1
    assert (
        optimizer["nativeRecordedActiveMaskSemantics"]["automaticPolicyMaskingRecommendation"]
        is None
    )
    coverage = optimizer["updateCoverage"]["rows"][0]
    assert coverage["batchDrawCount"] == coverage["uniqueTransitionCount"] == 2
    assert coverage["repeatedDrawCount"] == coverage["notVisitedTransitionCount"] == 0
    assert coverage["completeSingleVisit"] is True
    assert optimizer["sampledPolicyShift"]["nonnegativeKlEstimateMean"] == 0.005
    assert optimizer["sampledPolicyShift"]["measurementPhase"] == "pre-actor-step"
    post_step = optimizer["postStepSampledPolicyShift"]
    assert post_step["measurementPhase"] == "post-actor-step"
    assert post_step["rowSelection"] == "bounded-fixed-audit-rows"
    assert post_step["signedApproxKlMean"] == pytest.approx(0.0)
    assert post_step["nonnegativeKlEstimateMean"] == pytest.approx(math.cosh(0.1) - 1.0)
    assert post_step["sampledProbabilityRatio"]["min"] == pytest.approx(math.exp(-0.1))
    assert "not exact KL" in post_step["interpretation"]
    assert optimizer["teacherAndGradientContext"][0]["sampledPolicyShift"]["rowCount"] == 2
    assert optimizer["teacherAndGradientContext"][0]["postStepSampledPolicyShift"][
        "rowCount"
    ] == 2
    finding = summary["economicControlFlow"]["requestGrantFindings"][0]
    assert finding["economicActivity"] == "cultivation"
    assert finding["requestShortfallCount"] == 2
    assert finding["pairedRequestGrantCount"] == 2
    assert finding["activeRequestShortfallCount"] == 0
    assert finding["inactiveDeniedRequestCount"] == 2
    assert finding["overallExampleTransitionIds"] == sorted(
        ["run:base:70:2000:1:7", "run:base:70:2000:2:7"]
    )
    assert "2/2 unique, 2 draws, 0 repeats, 0 unvisited" in summary["text"]
    assert "independent critic calibration is unavailable" in summary["text"]


def test_physical_events_group_exact_semantics_without_mixing_food_scopes(tmp_path):
    root = tmp_path / "trace"
    first = transition_v2(turn=1)
    second = transition_v2(turn=2)
    additions = (
        (first, "PlantFood", 0.25),
        (second, "PlantFood", 0.5),
        (second, "AnimalFood", 0.125),
    )
    for ordinal, (transition, item, quantity) in enumerate(additions):
        transition["economicDiagnostics"]["physicalEvents"].append(
            {
                "eventId": f"{transition['trajectoryId']}/typed-food/{ordinal}",
                "campId": transition["campId"],
                "category": "human-food-consumption",
                "item": item,
                "unit": "food",
                "quantity": quantity,
                "direction": "out",
                "phase": "needs-after-production",
            }
        )
    write_artifact(root, [first, second], updates=update_pair([first, second]))

    physical = economic_trace_summary.summarize_trace(root)["physicalEvents"]
    plant = next(
        row
        for row in physical["groups"]
        if row["eventClass"] == "typed-camp-food-component"
        and row["item"] == "PlantFood"
    )
    linked = next(
        row
        for row in physical["groups"]
        if row["eventClass"] == "linked-food-aggregate"
    )
    assert plant["eventCount"] == plant["transitionCount"] == 2
    assert plant["quantity"]["sum"] == pytest.approx(0.75)
    assert plant["phase"] == "needs-after-production"
    assert linked["item"] == "linked-food-reserves"
    assert linked["quantity"]["sum"] == pytest.approx(2.0)
    assert physical["linkedFoodAggregateKeptSeparate"] is True
    assert physical["sourceAttribution"] is None
    typed = physical["typedCampFoodConsumption"]
    assert typed["positiveEventCount"] == 3
    assert typed["transitionsWithObservedPositiveTypedEvent"] == 2
    assert typed["coverage"] is None


def test_old_trace_absence_does_not_claim_zero_or_typed_coverage(tmp_path):
    summary = economic_trace_summary.summarize_trace(_artifact(tmp_path))
    physical = summary["physicalEvents"]
    typed = physical["typedCampFoodConsumption"]
    assert typed["positiveEventCount"] == 0
    assert typed["transitionsWithObservedPositiveTypedEvent"] == 0
    assert typed["transitionsWithoutObservedPositiveTypedEvent"] == 2
    assert typed["coverage"] is None
    assert "zero consumption or unsupported older capture" in typed["coverageUnavailableReason"]
    assert "zero versus unsupported absence" in summary["text"]
    inputs = summary["policyInputHealth"]
    assert inputs["rowsWithTelemetry"] == 0
    assert inputs["rowsMissingTelemetry"] == 2
    assert inputs["missingTelemetryReasons"] == {
        "not-emitted-by-legacy-economic-trace": 2
    }


def test_policy_input_health_summary_surfaces_exact_known_and_bounded_outliers(tmp_path):
    root = _artifact(tmp_path)
    from anthropocene_rl.autonomous_v7 import economic_diagnostics

    _, rows, updates = economic_diagnostics.load_artifact(root)
    for ordinal, row in enumerate(rows):
        core_53 = 1791.759 if ordinal == 0 else 3.0
        row["policyInputHealth"] = {
            "schemaVersion": 1,
            "source": "exact-actor-input-tensors",
            "floatTensorScope": "actor inputs",
            "previousDecision": {
                "status": "complete",
                "expectedIndices": list(range(40, 56)),
                "absentIndices": [],
                "coordinates": [
                    {
                        "index": 53,
                        "label": "compiler-intervention-count",
                        "raw": 5.0,
                        "transformed": 1.791759,
                        "normalizerMean": 0.0,
                        "normalizerStd": 0.001,
                        "affineBeforeGuard": core_53,
                        "normalized": core_53,
                        "corePresent": True,
                        "guardEnabled": False,
                        "guardLimit": None,
                        "dead": None,
                        "clippedByGuard": False,
                        "zeroedByDeadMask": None,
                        "zeroedByPresenceMask": False,
                    }
                ],
            },
            "movement": {
                "status": "complete",
                "expectedIndex": 144,
                "coordinate": {
                    "index": 144,
                    "label": "movement-flag",
                    "raw": 1.0,
                    "transformed": math.log(2),
                    "normalizerMean": 0.0,
                    "normalizerStd": 1.0,
                    "affineBeforeGuard": math.log(2),
                    "normalized": math.log(2),
                    "corePresent": True,
                    "guardEnabled": False,
                    "guardLimit": None,
                    "dead": None,
                    "clippedByGuard": False,
                    "zeroedByDeadMask": None,
                    "zeroedByPresenceMask": False,
                },
            },
            "groups": [
                {
                    "group": "core",
                    "semantics": "symlog-affine-normalized-actual-actor-input",
                    "scalarCount": 188,
                    "finiteCount": 188,
                    "nonfiniteCount": 0,
                    "maximumAbsolute": core_53,
                },
                {
                    "group": "personality",
                    "semantics": "identity-actual-actor-input",
                    "scalarCount": 4,
                    "finiteCount": 4,
                    "nonfiniteCount": 0,
                    "maximumAbsolute": 1.0,
                },
            ],
            "actorFloatInputNonfiniteCount": 0,
            "normalizedInputNonfiniteCount": 0,
            "maximumAbsoluteNormalized": core_53,
            "topNormalizedOutliers": [
                {
                    "group": "core",
                    "coordinate": [53],
                    "normalized": core_53,
                    "absoluteNormalized": core_53,
                }
            ],
        }
    replacement = tmp_path / "policy-input-health"
    write_artifact(replacement, rows, updates=updates)

    summary = economic_trace_summary.summarize_trace(replacement)
    health = summary["policyInputHealth"]
    assert health["rowsWithTelemetry"] == health["transitionDenominator"] == 2
    assert health["rowsMissingTelemetry"] == 0
    assert health["maximumAbsoluteNormalized"] == pytest.approx(1791.759)
    assert health["topNormalizedOutliers"][0]["coordinate"] == [53]
    core_53_summary = next(
        value for value in health["knownCoreCoordinates"] if value["index"] == 53
    )
    assert core_53_summary["maximumRecord"]["normalizerStd"] == 0.001
    assert core_53_summary["maximumRecord"]["transitionId"] == rows[0]["transitionId"]
    assert health["absentLegacyFields"] == {}
    assert "max |normalized|=1791.759" in summary["text"]


def test_old_trace_absence_marks_new_objective_and_post_step_fields_unavailable(tmp_path):
    root = _artifact(tmp_path, actual_objective_fields=False)
    from anthropocene_rl.autonomous_v7 import economic_diagnostics

    _, rows, updates = economic_diagnostics.load_artifact(root)
    old_updates = copy.deepcopy(updates)
    del old_updates[1]["optimizerDebug"]["sameSampledAction"][
        "postStepSampledPolicyShift"
    ]
    replacement = tmp_path / "old-fields"
    write_artifact(replacement, rows, updates=old_updates)

    summary = economic_trace_summary.summarize_trace(replacement)
    objective = summary["outcomes"]["objectives"]["welfare"]
    assert objective["weightedAdvantageContribution"]["count"] == 0
    assert objective["weightedAdvantageContribution"]["sum"] is None
    assert objective["weightedAdvantageContribution"]["unavailableReason"] == (
        "not-emitted-by-trace"
    )
    assert objective["effectiveCombinationWeight"]["mean"] is None
    bootstrap_flag = objective["zeroShapedRewardNonzeroBootstrap"]
    assert bootstrap_flag["observedCount"] == 0
    assert bootstrap_flag["trueCount"] is None
    assert bootstrap_flag["falseCount"] is None
    assert bootstrap_flag["unavailableReason"] == "not-emitted-by-trace"
    optimizer = summary["optimizerUpdates"]
    assert optimizer["postStepSampledPolicyShift"] is None
    assert optimizer["postStepSampledPolicyShiftUnavailableReason"] == (
        "not-emitted-by-trace"
    )
    assert "post-step bounded-audit sampled shift unavailable" in summary["text"]


def test_summary_does_not_invent_update_attribution_without_audit_rows(tmp_path):
    summary = economic_trace_summary.summarize_trace(_artifact(tmp_path, debug=False))
    optimizer = summary["optimizerUpdates"]
    assert optimizer["alignmentDenominator"] == 0
    assert optimizer["descriptiveSignAlignment"]["negativeAdvantageReinforcement"] == 0
    assert optimizer["teacherAndGradientContext"] == []
    assert optimizer["unknownCounts"] == {"optimizerDebug-not-captured": 1}
    assert optimizer["causalAttribution"] is None


def test_summary_rejects_forged_action_probability_delta(tmp_path):
    root = _artifact(tmp_path)
    # Rebuild a validly sealed artifact with an internally inconsistent debug row.
    from anthropocene_rl.autonomous_v7 import economic_diagnostics

    _, rows, updates = economic_diagnostics.load_artifact(root)
    forged = copy.deepcopy(updates)
    forged[1]["optimizerDebug"]["sameSampledAction"]["rows"][0]["jointDelta"] = 0.2
    replacement = tmp_path / "forged"
    write_artifact(replacement, rows, updates=forged)
    with pytest.raises(ValueError, match="joint action delta differs"):
        economic_trace_summary.summarize_trace(replacement)


def test_summary_rejects_forged_post_step_sampled_shift(tmp_path):
    root = _artifact(tmp_path)
    from anthropocene_rl.autonomous_v7 import economic_diagnostics

    _, rows, updates = economic_diagnostics.load_artifact(root)
    forged = copy.deepcopy(updates)
    forged[1]["optimizerDebug"]["sameSampledAction"]["postStepSampledPolicyShift"][
        "nonnegativeKlEstimateMean"
    ] = 0.0
    replacement = tmp_path / "forged-post-step"
    write_artifact(replacement, rows, updates=forged)
    with pytest.raises(ValueError, match="post-step sampled policy shift differs"):
        economic_trace_summary.summarize_trace(replacement)


def test_phase_ratio_uses_only_paired_values_and_never_crosses_units():
    records = [
        ({}, {"phases": {"source": 4.0, "destination": 2.0}}),
        ({}, {"phases": {"source": 100.0, "destination": None}}),
    ]
    paired = economic_trace_summary._paired_ratio(
        records, "source", "destination", "good-units", "good-units"
    )
    assert paired["pairedValueCount"] == 1
    assert paired["value"] == 0.5
    mixed = economic_trace_summary._paired_ratio(
        records, "source", "destination", None, None
    )
    assert mixed["value"] is None
    assert mixed["unavailableReason"] == "mixed-or-missing-unit"


def test_request_instance_ids_group_by_stable_manufacturing_semantics():
    common = {
        "domain": "manufacturing",
        "actorHead": "service[3]",
        "nativeUnit": "good-units",
        "metadata": {
            "service": "herding tools",
            "serviceClass": "Continuity",
            "good": "HerdingTools",
            "rawMaterial": None,
            "role": None,
        },
    }
    first = {**common, "identity": "turn/9/camp/11/request/0"}
    second = copy.deepcopy({**common, "identity": "turn/10/camp/12/request/7"})
    assert economic_trace_summary._control_semantic_key(first) == (
        economic_trace_summary._control_semantic_key(second)
    )
    transition = {
        "economicDiagnostics": {
            "requests": [{"requestId": first["identity"], "unmetReason": "labor rationing"}]
        }
    }
    assert economic_trace_summary._request_unmet_reason(transition, first) == "labor rationing"
    assert economic_trace_summary._economic_activity(first) == (
        "herding",
        "metadata.service:herding tools",
    )


def test_update_coverage_exposes_repeats_and_unvisited_rows():
    transitions = [
        {
            "transitionId": f"row-{index}",
            "runId": "run",
            "arm": "base",
            "optimizerSeed": 70,
            "rollout": 2,
        }
        for index in range(4)
    ]
    identity = ("run", "base", 70, 2, 0)
    rows = economic_trace_summary._update_coverage(
        transitions,
        [(identity, ["row-0", "row-1"]), (identity, ["row-0", "row-2"])],
    )
    assert rows == [
        {
            "runId": "run",
            "arm": "base",
            "optimizerSeed": 70,
            "rollout": 2,
            "epoch": 0,
            "rolloutTransitionCount": 4,
            "minibatchCount": 2,
            "batchDrawCount": 4,
            "uniqueTransitionCount": 3,
            "repeatedDrawCount": 1,
            "notVisitedTransitionCount": 1,
            "visitedFraction": 0.75,
            "completeSingleVisit": False,
            "exampleRepeatedTransitionIds": ["row-0"],
            "exampleNotVisitedTransitionIds": ["row-3"],
        }
    ]


def test_request_grant_headline_is_active_unit_safe_and_transition_deduplicated():
    def record(identity, classification, request, grant, *, unit="workers"):
        transition = {
            "transitionId": identity,
            "economicDiagnostics": {"requests": []},
        }
        control = {
            "domain": "cultivation",
            "identity": "fieldDevelopment",
            "actorHead": "strategic[0]",
            "classification": classification,
            "phaseUnits": {
                "actorCappedRequest": "workers",
                "compilerGrant": unit,
            },
            "phases": {
                "actorCappedRequest": request,
                "compilerGrant": grant,
                "executedWorkerOrInputClaim": None,
                "committedPhysicalReceipt": None,
                "inventoryChange": None,
                "consumptionOrFinalService": None,
                "endingPopulation": 10.0 if identity == "t1" else 20.0,
                "endingQol": 0.5 if identity == "t1" else 0.75,
            },
        }
        return transition, control

    semantic = {
        "domain": "cultivation",
        "identity": "fieldDevelopment",
        "actorHead": "strategic[0]",
        "nativeUnit": "workers",
    }
    records = [
        record("t1", "active", 2.0, 0.0),
        record("t1", "active", 1.0, 0.0),
        record("t2", "inactive", 4.0, 0.0),
    ]
    finding = economic_trace_summary._request_grant_finding(semantic, records)
    assert finding["requestShortfallCount"] == 3
    assert finding["activeRequestShortfallCount"] == 2
    assert finding["inactiveDeniedRequestCount"] == 1
    assert finding["activeGrantToRequestRatio"] == 0.0
    context = finding["outcomeContextAmongActiveShortfalls"]
    assert context["requestRowCount"] == 2
    assert context["transitionCount"] == 1
    assert context["endingPopulation"]["count"] == 1

    incompatible = economic_trace_summary._request_grant_finding(
        semantic,
        [record("t1", "active", 2.0, 1.0, unit="operations")],
    )
    assert incompatible["activeGrantToRequestRatio"] is None
    assert (
        incompatible["activeGrantToRequestRatioUnavailableReason"]
        == "mixed-missing-or-incompatible-units"
    )


def test_cli_writes_one_json_and_refuses_overwrite(tmp_path):
    root = _artifact(tmp_path)
    output = tmp_path / "summary.json"
    assert economic_trace_summary.main(["--trace", str(root), "--output", str(output)]) == 0
    assert output.is_file()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        economic_trace_summary.main(["--trace", str(root), "--output", str(output)])
