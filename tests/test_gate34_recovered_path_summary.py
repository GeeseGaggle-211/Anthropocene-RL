from __future__ import annotations

import copy
import json

import pytest

from anthropocene_rl.autonomous_v7.gate34_recovered_path_summary import (
    _read_path_records,
    _relation,
    _validate_path_records,
)
from anthropocene_rl.autonomous_v7.provenance import ProvenanceError, digest

OBJECTIVES = ["welfare", "population_years", "trade_utility", "qol_years"]
ALTERNATIVES = ["sampled-policy", "development-budget-native-v1"]


def _plan():
    schedules = []
    for replicate, maximum in enumerate((100, 78, 112, 73)):
        schedules.append(
            {
                "pathReplicate": replicate,
                "scheduleId": f"schedule/{replicate}",
                "rngSeed": 100 + replicate,
                "stopIndices": list(range(16)),
                "requiredTailTerms": 2,
                "maximumStepsPerAlternative": maximum,
                "exceedsHardCap": False,
            }
        )
    return {
        "runtime": {"objectives": OBJECTIVES},
        "schedules": schedules,
        "traceResetBinding": {"firstDecisionRowCount": 8},
        "maximumBridgeStepCalls": 727,
    }


def _records(plan):
    records = []
    cumulative = 0
    for replicate in range(4):
        schedule = plan["schedules"][replicate]
        for alternative in ALTERNATIVES:
            maximum = schedule["maximumStepsPerAlternative"]
            cumulative += maximum
            tails = {
                objective: {
                    "status": (
                        "unresolved-absolute-integrability"
                        if objective == "trade_utility"
                        else "identified"
                    ),
                    "scheduleId": schedule["scheduleId"],
                    "estimate": None if objective == "trade_utility" else 1.0,
                    "drawCount": 0 if objective == "trade_utility" else 16,
                }
                for objective in OBJECTIVES
            }
            record = {
                "sequence": len(records),
                "alternative": alternative,
                "pathReplicate": replicate,
                "schedule": {
                    "id": schedule["scheduleId"],
                    "rngSeed": schedule["rngSeed"],
                    "stopIndices": schedule["stopIndices"],
                    "requiredTailTerms": schedule["requiredTailTerms"],
                    "exceedsHardCap": False,
                },
                "maximumSteps": maximum,
                "collectedSteps": maximum,
                "cumulativeCompletedBranchSteps": cumulative,
                "openingCampCount": 8,
                "terminal": False,
                "prefixReturns": dict.fromkeys(OBJECTIVES, 1.0),
                "tailEstimates": tails,
                "rankingPredictions": dict.fromkeys(OBJECTIVES, 2.0),
            }
            record["recordDigest"] = digest(record)
            records.append(record)
    return records


def test_recovered_path_validation_accepts_exact_complete_pairs():
    plan = _plan()
    result = _validate_path_records(plan, _records(plan))
    assert result["recordCount"] == 8
    assert result["exactAlternativePairCount"] == 4
    assert result["actualBridgeStepCallsInOriginalRun"] == 727


def test_recovered_path_validation_rejects_record_tampering(tmp_path):
    plan = _plan()
    records = _records(plan)
    records[0]["prefixReturns"]["welfare"] = 999.0
    path = tmp_path / "paths.jsonl"
    path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="digest differs"):
        _read_path_records(path)


def test_recovered_path_validation_rejects_missing_exact_pair():
    plan = _plan()
    records = _records(plan)
    replacement = copy.deepcopy(records[-2])
    replacement["sequence"] = 7
    replacement["cumulativeCompletedBranchSteps"] = 726
    replacement["recordDigest"] = digest(
        {key: value for key, value in replacement.items() if key != "recordDigest"}
    )
    records[-1] = replacement
    with pytest.raises(ProvenanceError, match="pair identities/order differ"):
        _validate_path_records(plan, records)


def test_relation_retains_a_declared_numerical_zero_band():
    assert _relation(-2e-10) == "within-numerical-zero-tolerance"
    assert _relation(-2e-8) == "negative"
    assert _relation(2e-8) == "positive"
