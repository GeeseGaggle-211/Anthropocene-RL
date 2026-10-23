"""Recover the completed single-state path summary after its metadata-only audit abort.

This command is deliberately bound to the reviewed T3per-s70/world-50000 probe.  It
does not replay a continuation.  Its only native operations are Reset, fork and drop;
``bridge.step`` is never called.  The recovered result remains labelled separately
from the original runner result, which was not written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from .bridge import Model9Bridge
from .economic_trace import canonical_bytes
from .gate1_production_adapter import tensor_state_identity
from .gate34_checkpoint_collector import (
    FROZEN_ALTERNATIVES,
    FROZEN_PROTOCOL,
    PersonalityState,
    _critic_predictions,
    _derived_seed,
)
from .gate34_single_state_debug import (
    ENDPOINT_ID,
    EXPECTED_RESET_TURN,
    NATIVE_LABEL_PROBE_STEPS,
    REGIME_ID,
    WITHHELD_MASK,
    WORLD_SEED,
    _load_sealed_plan,
    _sha256,
    _verify_worker_inputs,
    _write_sealed,
)
from .provenance import ProvenanceError, digest

Json = dict[str, Any]
KIND = "gate34-single-state-recovered-path-summary"
ORIGINAL_ABORT = "bridge did not return to the exact frozen state after all paths"
NUMERICAL_EQUALITY_TOLERANCE = 1e-9


def _read_path_records(path: Path) -> list[Json]:
    records: list[Json] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ProvenanceError(f"blank recovered path record at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProvenanceError(f"recovered path record {line_number} is not an object")
            record = cast(Json, value)
            expected = digest(
                {key: item for key, item in record.items() if key != "recordDigest"}
            )
            if record.get("recordDigest") != expected:
                raise ProvenanceError(
                    f"recovered path record {line_number} digest differs"
                )
            records.append(record)
    return records


def _expected_schedule(plan: Json, replicate: int) -> Json:
    schedules = cast(list[Json], plan["schedules"])
    matches = [row for row in schedules if row.get("pathReplicate") == replicate]
    if len(matches) != 1:
        raise ProvenanceError("plan does not have exactly one schedule per path replicate")
    return matches[0]


def _validate_path_records(plan: Json, records: Sequence[Json]) -> Json:
    expected_pairs = [
        (replicate, alternative)
        for replicate in range(FROZEN_PROTOCOL.path_replicates)
        for alternative in FROZEN_ALTERNATIVES
    ]
    if len(records) != len(expected_pairs):
        raise ProvenanceError("recovered paths do not contain the complete eight records")
    actual_pairs = [
        (record.get("pathReplicate"), record.get("alternative")) for record in records
    ]
    if actual_pairs != expected_pairs:
        raise ProvenanceError("recovered path pair identities/order differ")

    objectives = set(cast(list[str], cast(Json, plan["runtime"])["objectives"]))
    cumulative = 0
    for sequence, record in enumerate(records):
        replicate = int(record["pathReplicate"])
        expected = _expected_schedule(plan, replicate)
        expected_schedule = {
            "id": expected["scheduleId"],
            "rngSeed": expected["rngSeed"],
            "stopIndices": expected["stopIndices"],
            "requiredTailTerms": expected["requiredTailTerms"],
            "exceedsHardCap": expected["exceedsHardCap"],
        }
        if record.get("sequence") != sequence:
            raise ProvenanceError("recovered path sequence is not contiguous")
        if record.get("schedule") != expected_schedule:
            raise ProvenanceError("recovered path schedule differs from persisted plan")
        maximum = expected["maximumStepsPerAlternative"]
        if record.get("maximumSteps") != maximum or record.get("collectedSteps") != maximum:
            raise ProvenanceError("recovered path is not complete to its planned bound")
        cumulative += int(record["collectedSteps"])
        if record.get("cumulativeCompletedBranchSteps") != cumulative:
            raise ProvenanceError("recovered path cumulative step count differs")
        if record.get("openingCampCount") != cast(Json, plan["traceResetBinding"])[
            "firstDecisionRowCount"
        ]:
            raise ProvenanceError("recovered path opening-camp denominator differs")
        if not isinstance(record.get("terminal"), bool):
            raise ProvenanceError("recovered path terminal flag is malformed")
        for name in ("prefixReturns", "tailEstimates", "rankingPredictions"):
            value = record.get(name)
            if not isinstance(value, dict) or set(value) != objectives:
                raise ProvenanceError(f"recovered path {name} objective coverage differs")
        for objective in objectives:
            if not math.isfinite(float(record["prefixReturns"][objective])) or not math.isfinite(
                float(record["rankingPredictions"][objective])
            ):
                raise ProvenanceError("recovered path contains non-finite point values")
            tail = record["tailEstimates"][objective]
            if not isinstance(tail, dict) or tail.get("scheduleId") != expected["scheduleId"]:
                raise ProvenanceError("recovered path tail schedule differs")
            if tail.get("status") == "identified":
                if (
                    not math.isfinite(float(tail["estimate"]))
                    or tail.get("drawCount") != len(expected["stopIndices"])
                ):
                    raise ProvenanceError("identified recovered tail is malformed")
            elif tail.get("estimate") is not None or tail.get("drawCount") != 0:
                raise ProvenanceError("unresolved recovered tail invents a point estimate")

    actual_steps = cumulative + NATIVE_LABEL_PROBE_STEPS
    if actual_steps != plan.get("maximumBridgeStepCalls") or actual_steps != 727:
        raise ProvenanceError("recovered path/native-label accounting is not exactly 727")
    return {
        "recordCount": len(records),
        "exactAlternativePairCount": len(expected_pairs) // len(FROZEN_ALTERNATIVES),
        "completedBranchSteps": cumulative,
        "nativeLabelProbeSteps": NATIVE_LABEL_PROBE_STEPS,
        "actualBridgeStepCallsInOriginalRun": actual_steps,
        "allRecordDigestsVerified": True,
        "allSchedulesVerified": True,
        "allPairsVerified": True,
    }


def _projection(step: Json) -> list[Json]:
    return [
        {
            "camp": int(village["camp"]),
            "observation": village["observation"],
        }
        for village in sorted(step["villages"], key=lambda row: int(row["camp"]))
    ]


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _without_fingerprint(step: Json) -> Json:
    return {key: value for key, value in step.items() if key != "fingerprint"}


def _difference_paths(left: Any, right: Any, path: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [f"{path}:type:{type(left).__name__}->{type(right).__name__}"]
    if isinstance(left, dict):
        result: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}.{key}"
            if key not in left:
                result.append(f"{child}:missing-left")
            elif key not in right:
                result.append(f"{child}:missing-right")
            else:
                result.extend(_difference_paths(left[key], right[key], child))
        return result
    if isinstance(left, list):
        result = []
        if len(left) != len(right):
            result.append(f"{path}:length:{len(left)}->{len(right)}")
        for index, (first, second) in enumerate(zip(left, right, strict=False)):
            result.extend(_difference_paths(first, second, f"{path}[{index}]"))
        return result
    return [] if left == right else [path]


def _model_hashes(runtime: Any) -> Json:
    return {
        "actorTypedContentSha256": tensor_state_identity(runtime.actor.state_dict())[
            "typed_content_sha256"
        ],
        "criticTypedContentSha256": tensor_state_identity(runtime.critic.state_dict())[
            "typed_content_sha256"
        ],
    }


def _opening_predictions(plan: Json, registry: Any, runtime: Any) -> tuple[Json, Json, Json]:
    before = _model_hashes(runtime)
    bridge: Model9Bridge | None = None
    try:
        dotnet = os.environ.get("ANTHROPOCENE_DOTNET", "dotnet")
        bridge = Model9Bridge(
            command=[dotnet, str(registry.bridge_path)], economic_diagnostics=True
        )
        reset = cast(
            Json,
            bridge.reset(
                WORLD_SEED,
                movement=runtime.stage.bridge_movement,
                withheld=WITHHELD_MASK,
            ),
        )
        expected_reset = cast(Json, plan["traceResetBinding"])["nativeResetIdentity"]
        actual_reset = {
            key: bridge.last_reset_identity.get(key)
            for key in (
                "configurationHash",
                "fingerprint",
                "buildHash",
                "rawMaterialsEnabled",
                "sourceSeed",
            )
        }
        if actual_reset != expected_reset:
            raise ProvenanceError("recovery live reset identity differs from persisted plan")
        if (
            int(reset["turn"]) != EXPECTED_RESET_TURN
            or str(reset["fingerprint"]) != str(expected_reset["fingerprint"])
        ):
            raise ProvenanceError("recovery reset decision identity differs")

        fork_id, forked = bridge.fork()
        try:
            reset_projection = _projection(reset)
            fork_projection = _projection(cast(Json, forked))
            if reset_projection != fork_projection:
                raise ProvenanceError("recovery fork changed canonical observations")
            differences = _difference_paths(
                _without_fingerprint(reset), _without_fingerprint(cast(Json, forked))
            )
            if differences != ["$.restoredObservation"]:
                raise ProvenanceError("recovery fork has more than restored-observation metadata")
            if str(reset["fingerprint"]) == str(forked["fingerprint"]):
                raise ProvenanceError("recovery did not reproduce reset-to-fork normalization")

            personality_seed = _derived_seed(
                "gate34-main-personality-v1", ENDPOINT_ID, WORLD_SEED, REGIME_ID
            )
            reset_personalities = PersonalityState(
                runtime.personality_spec, runtime.personality_drift, personality_seed
            )
            fork_personalities = PersonalityState(
                runtime.personality_spec, runtime.personality_drift, personality_seed
            )
            reset_values, reset_diagnostics = _critic_predictions(
                runtime, reset["villages"], reset_personalities
            )
            fork_values, fork_diagnostics = _critic_predictions(
                runtime, forked["villages"], fork_personalities
            )
            if reset_values != fork_values or reset_diagnostics != fork_diagnostics:
                raise ProvenanceError(
                    "recovery critic predictions differ across equivalent observations"
                )
        finally:
            bridge.drop(fork_id)
    finally:
        if bridge is not None:
            bridge.close()
    after = _model_hashes(runtime)
    if after != before:
        raise ProvenanceError("recovery summary mutated actor or critic tensors")
    audit = {
        "bridgeStepCalls": 0,
        "reset": {
            "turn": int(reset["turn"]),
            "fingerprint": str(reset["fingerprint"]),
            "restoredObservation": bool(reset.get("restoredObservation", False)),
            "canonicalObservationSha256": _canonical_sha(reset_projection),
        },
        "canonicalFork": {
            "turn": int(forked["turn"]),
            "fingerprint": str(forked["fingerprint"]),
            "restoredObservation": bool(forked.get("restoredObservation", False)),
            "canonicalObservationSha256": _canonical_sha(fork_projection),
        },
        "observationEquivalent": True,
        "stepWithoutFingerprintDifferencePaths": differences,
        "interpretation": (
            "The raw-reset fingerprint differs from the canonical-fork fingerprint while "
            "public observations and critic predictions are identical. This diagnostic does "
            "not isolate the mechanism that changes the fingerprint."
        ),
    }
    no_learning = {
        "optimizerConstructed": False,
        "optimizerStepCalled": False,
        "before": before,
        "after": after,
        "actorAndCriticUnchanged": True,
        "scope": (
            "recovery summarizer only; the aborted original runner did not serialize "
            "its check"
        ),
    }
    values_json = {
        objective: {str(camp): value for camp, value in values.items()}
        for objective, values in fork_values.items()
    }
    diagnostics_json = {
        objective: {str(camp): value for camp, value in values.items()}
        for objective, values in fork_diagnostics.items()
    }
    return audit, no_learning, {
        "criticValuesByCamp": values_json,
        "criticDistributionByCamp": diagnostics_json,
    }


def _relation(value: float) -> str:
    if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=NUMERICAL_EQUALITY_TOLERANCE):
        return "within-numerical-zero-tolerance"
    return "positive" if value > 0 else "negative"


def _full_return(record: Json, objective: str) -> float | None:
    tail = record["tailEstimates"][objective]
    if tail["status"] != "identified":
        return None
    return float(record["prefixReturns"][objective]) + float(tail["estimate"])


def _summary(records: Sequence[Json], opening: Json, objectives: Sequence[str]) -> Json:
    by_pair = {
        (int(record["pathReplicate"]), str(record["alternative"])): record
        for record in records
    }
    identified = [objective for objective in objectives if objective != "trade_utility"]
    calibration: Json = {}
    support: Json = {}
    for objective in identified:
        baseline_points = [
            _full_return(by_pair[(replicate, "sampled-policy")], objective)
            for replicate in range(FROZEN_PROTOCOL.path_replicates)
        ]
        if any(value is None for value in baseline_points):
            raise ProvenanceError(f"baseline {objective} full-return point is unresolved")
        points = [float(cast(float, value)) for value in baseline_points]
        critic_by_camp = cast(Json, opening["criticValuesByCamp"])[objective]
        critic = math.fsum(float(value) for value in critic_by_camp.values())
        point_mean = fmean(points)
        calibration[objective] = {
            "pathPointEstimates": points,
            "pathCount": len(points),
            "baselineFullReturnPointMean": point_mean,
            "openingCriticRawWorldValue": critic,
            "pointMeanMinusCritic": point_mean - critic,
            "uncertainty": None,
            "uncertaintyUnavailableReason": (
                "per-draw tail estimates/covariance were not serialized; four paths are nested "
                "within one world and no inferential standard error is claimed"
            ),
        }

        diagnostics = cast(Json, opening["criticDistributionByCamp"])[objective]
        lower = math.fsum(float(row["supportRaw"][0]) for row in diagnostics.values())
        upper = math.fsum(float(row["supportRaw"][1]) for row in diagnostics.values())
        support[objective] = {
            "aggregateOpeningRawSupportEnvelope": [lower, upper],
            "baselineFullReturnPointMean": point_mean,
            "pointMeanInsideAggregateEnvelope": lower <= point_mean <= upper,
            "perCampOpeningDiagnostics": diagnostics,
            "perCampTargetSupportComparison": None,
            "perCampTargetSupportComparisonUnavailableReason": (
                "completed progress records did not serialize per-camp prefix/tail returns"
            ),
        }

    paired: Json = {}
    for objective in identified:
        rows = []
        for replicate in range(FROZEN_PROTOCOL.path_replicates):
            baseline = by_pair[(replicate, "sampled-policy")]
            intervention = by_pair[(replicate, "development-budget-native-v1")]
            baseline_full = _full_return(baseline, objective)
            intervention_full = _full_return(intervention, objective)
            if baseline_full is None or intervention_full is None:
                raise ProvenanceError(f"paired {objective} full-return point is unresolved")
            predicted = float(intervention["rankingPredictions"][objective]) - float(
                baseline["rankingPredictions"][objective]
            )
            realised = intervention_full - baseline_full
            rows.append(
                {
                    "pathReplicate": replicate,
                    "predictedOneStepDelta": predicted,
                    "realizedFullReturnPointDelta": realised,
                    "predictedRelation": _relation(predicted),
                    "realizedRelation": _relation(realised),
                    "signAgreementAtNumericalTolerance": (
                        _relation(predicted) == _relation(realised)
                    ),
                }
            )
        predicted_mean = fmean(float(row["predictedOneStepDelta"]) for row in rows)
        realised_mean = fmean(float(row["realizedFullReturnPointDelta"]) for row in rows)
        paired[objective] = {
            "paths": rows,
            "pathCount": len(rows),
            "predictedDeltaPointMean": predicted_mean,
            "realizedFullReturnDeltaPointMean": realised_mean,
            "realizedMinusPredictedPointMean": realised_mean - predicted_mean,
            "signAgreementAtNumericalToleranceCount": sum(
                row["signAgreementAtNumericalTolerance"] for row in rows
            ),
            "uncertainty": None,
            "uncertaintyUnavailableReason": (
                "per-draw paired covariance was not serialized; no stopping-randomization or "
                "finite-world standard error is reconstructed"
            ),
        }

    trade_rows = []
    for replicate in range(FROZEN_PROTOCOL.path_replicates):
        baseline = by_pair[(replicate, "sampled-policy")]
        intervention = by_pair[(replicate, "development-budget-native-v1")]
        prefix_delta = float(intervention["prefixReturns"]["trade_utility"]) - float(
            baseline["prefixReturns"]["trade_utility"]
        )
        predicted = float(intervention["rankingPredictions"]["trade_utility"]) - float(
            baseline["rankingPredictions"]["trade_utility"]
        )
        trade_rows.append(
            {
                "pathReplicate": replicate,
                "predictedOneStepDelta": predicted,
                "observedFinite25PrefixDelta": prefix_delta,
                "predictedRelation": _relation(predicted),
                "prefixRelation": _relation(prefix_delta),
            }
        )
    trade = {
        "prefixTerms": FROZEN_PROTOCOL.prefix_terms,
        "paths": trade_rows,
        "predictedDeltaPointMean": fmean(
            float(row["predictedOneStepDelta"]) for row in trade_rows
        ),
        "observedFinite25PrefixDeltaPointMean": fmean(
            float(row["observedFinite25PrefixDelta"]) for row in trade_rows
        ),
        "numericalEqualityTolerance": NUMERICAL_EQUALITY_TOLERANCE,
        "fullReturn": None,
        "fullReturnUnavailableReason": (
            "trade tails are unresolved-absolute-integrability in every serialized path"
        ),
        "interpretation": "finite observed prefix only; not a trade full-return target",
    }
    return {
        "baselineFullReturnPointMeanVsOpeningCritic": calibration,
        "criticDistributionAndAggregateSupport": support,
        "pairedPredictedVsRealizedPointDeltas": paired,
        "tradeFinite25PrefixDelta": trade,
        "independentUnit": "one world",
        "nestedMeasurements": "four stochastic paths; paths are not independent worlds",
        "standardErrorsReported": False,
        "numericalEqualityTolerance": NUMERICAL_EQUALITY_TOLERANCE,
    }


def recover(source: Path, output: Path) -> Json:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite recovered result: {output}")
    plan_path = source / "plan.json"
    paths_path = source / "paths.jsonl"
    stderr_path = source / "worker.stderr.log"
    if (source / "result.json").exists():
        raise ProvenanceError("original runner already has a result; recovery is inapplicable")
    plan = _load_sealed_plan(plan_path)
    records = _read_path_records(paths_path)
    validation = _validate_path_records(plan, records)
    stderr = stderr_path.read_text(encoding="utf-8")
    if ORIGINAL_ABORT not in stderr:
        raise ProvenanceError("original runner did not fail at the reviewed final metadata audit")

    registry, runtime = _verify_worker_inputs(plan)
    native_audit, no_learning, opening = _opening_predictions(
        plan, registry, runtime
    )
    summary = _summary(records, opening, cast(list[str], plan["runtime"]["objectives"]))
    result: Json = {
        "kind": KIND,
        "schema": 1,
        "status": "recovered-complete-branch-evidence-original-final-audit-aborted",
        "planDigest": plan["digest"],
        "source": {
            "originalOutput": str(source.resolve()),
            "planPath": str(plan_path.resolve()),
            "planSha256": _sha256(plan_path),
            "pathsPath": str(paths_path.resolve()),
            "pathsSha256": _sha256(paths_path),
            "workerStderrPath": str(stderr_path.resolve()),
            "workerStderrSha256": _sha256(stderr_path),
            "originalResultPresent": False,
            "originalFinalAudit": "aborted",
            "originalFinalAuditReason": ORIGINAL_ABORT,
        },
        "validation": validation,
        "nativeRecoveryAudit": native_audit,
        "noLearningMutationProof": no_learning,
        "opening": opening,
        "summary": summary,
        "limitations": [
            "the original runner did not serialize result.json or its post-run tensor check",
            "this recovery independently recomputed opening critic predictions without bridge.step",
            "per-draw tail estimates and paired covariance were not serialized and are "
            "not fabricated",
            "trade full returns remain unresolved",
            "four paths are nested within one world and do not support finite-world inference",
            "this is not gate passage, readiness, selection or checkpoint promotion evidence",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_sealed(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = recover(args.source.resolve(), args.out.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "out": str(args.out.resolve()),
                "validatedRecords": result["validation"]["recordCount"],
                "recoveryBridgeStepCalls": result["nativeRecoveryAudit"]["bridgeStepCalls"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
