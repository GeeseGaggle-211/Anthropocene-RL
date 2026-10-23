from __future__ import annotations

from typing import Any

import pytest

from anthropocene_rl.autonomous_v7 import scale_probe


def _trace(seed: int, *, mismatch: bool = False) -> dict[str, Any]:
    marker = f"{seed}-changed" if mismatch else str(seed)
    return {
        "fingerprints": [f"fingerprint-{seed}"],
        "observationsSha256": [f"observation-{marker}"],
        "rewardsSha256": [f"reward-{seed}"],
        "actionsSha256": [f"actions-{seed}"],
        "combinedSha256": f"combined-{marker}",
    }


def _fake_execute(
    profile: dict[str, Any],
    *,
    economic_diagnostics: bool,
    command: list[str] | None,
    missing_rss: bool = False,
    mismatch: bool = False,
) -> dict[str, Any]:
    del command
    workers = int(profile["workers"])
    median_step = 1.1 if economic_diagnostics else 1.0
    wall = 12.0 if economic_diagnostics else 10.0
    rss = None if missing_rss else workers * (110.0 if economic_diagnostics else 100.0)
    return {
        "workers": workers,
        "completedWorlds": workers,
        "completedSteps": workers * int(profile["horizon"]),
        "wallSeconds": wall,
        "worldsPerSecond": workers / wall,
        "stepsPerSecond": workers * int(profile["horizon"]) / wall,
        "measuredStepSecondsMedian": median_step,
        "workerPeakRss": [
            {
                "seed": seed,
                "peakRssMiB": None if missing_rss else rss / workers,
                "status": "unavailable" if missing_rss else "available",
            }
            for seed in profile["seeds"]
        ],
        "workerPeakRssSumMiB": rss,
        "workerPeakRssSumStatus": "unavailable" if missing_rss else "available",
        "traceBySeed": {
            str(seed): _trace(
                seed,
                mismatch=mismatch and economic_diagnostics and seed == profile["seeds"][0],
            )
            for seed in profile["seeds"]
        },
        "rawWorkers": [],
        "profileId": profile["id"],
    }


def test_frozen_paired_profiles_match_reviewed_worlds_and_limits() -> None:
    assert scale_probe.PAIRED_WARMUP_REPETITIONS == 2
    assert scale_probe.PAIRED_MEASURED_REPETITIONS == 7
    assert scale_probe.PAIRED_RSS_LIMIT_MIB == 256.0
    current, mature, workers = scale_probe.DIAGNOSTIC_PROFILES
    assert (current["seeds"], current["expectedResetTurn"], current["horizon"]) == (
        (50000,),
        9,
        25,
    )
    assert (mature["seeds"], mature["worldWarmupSteps"], mature["limit"]) == (
        (50001,),
        25,
        1.20,
    )
    assert workers["seeds"] == tuple(range(50000, 50008))
    assert (workers["workers"], workers["limit"]) == (8, 0.80)


def test_paired_profile_alternates_preserves_raw_runs_and_uses_paired_ratios() -> None:
    calls: list[tuple[str, bool]] = []

    def execute(profile: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append((profile["id"], kwargs["economic_diagnostics"]))
        return _fake_execute(profile, **kwargs)

    result = scale_probe.run_paired_profile(
        dict(scale_probe.DIAGNOSTIC_PROFILES[0]),
        baseline={"economicDiagnostics": False, "command": None},
        treatment={"economicDiagnostics": True, "command": None},
        execute=execute,
    )

    assert len(calls) == 2 * (2 + 7)
    assert [row["executionOrder"] for row in result["measuredPairs"]] == [
        ["baseline", "treatment"],
        ["treatment", "baseline"],
        ["baseline", "treatment"],
        ["treatment", "baseline"],
        ["baseline", "treatment"],
        ["treatment", "baseline"],
        ["baseline", "treatment"],
    ]
    assert result["medianOfPerPairRatios"] == pytest.approx(1.1)
    assert result["maximumPairedIncrementalPeakWorkerRssMiB"] == 10.0
    assert result["parityExact"] is True
    assert result["acceptanceStatus"] == "thresholds-met"
    assert result["measuredPairs"][0]["baseline"]["rawWorkers"] == []


def test_eight_worker_ratio_is_same_world_set_on_over_off_throughput() -> None:
    result = scale_probe.run_paired_profile(
        dict(scale_probe.DIAGNOSTIC_PROFILES[2]),
        baseline={"economicDiagnostics": False, "command": None},
        treatment={"economicDiagnostics": True, "command": None},
        execute=_fake_execute,
    )
    assert result["medianOfPerPairRatios"] == pytest.approx(10.0 / 12.0)
    assert result["maximumPairedIncrementalPeakWorkerRssMiB"] == 80.0
    first = result["measuredPairs"][0]
    assert len(first["baseline"]["workerPeakRss"]) == 8
    assert first["baseline"]["workerPeakRssSumMiB"] == 800.0
    assert result["acceptanceStatus"] == "thresholds-met"


def test_execution_summary_keeps_each_worker_rss_and_never_sums_missing_as_zero() -> None:
    profile = dict(scale_probe.DIAGNOSTIC_PROFILES[2])
    workers = []
    for seed in profile["seeds"]:
        workers.append(
            {
                "seed": seed,
                "measuredStepSeconds": [0.1, 0.2],
                "measuredStepsCompleted": 2,
                "workerPeakRssMiB": float(seed - 49900),
                "workerPeakRssStatus": "available",
                "trace": _trace(seed),
            }
        )
    summary = scale_probe._execution_summary(profile, workers, 4.0)
    assert [row["seed"] for row in summary["workerPeakRss"]] == list(range(50000, 50008))
    assert summary["workerPeakRssSumMiB"] == sum(range(100, 108))
    assert summary["worldsPerSecond"] == 2.0

    workers[3]["workerPeakRssMiB"] = None
    workers[3]["workerPeakRssStatus"] = "unavailable"
    missing = scale_probe._execution_summary(profile, workers, 4.0)
    assert missing["workerPeakRss"][3]["status"] == "unavailable"
    assert missing["workerPeakRssSumMiB"] is None
    assert missing["workerPeakRssSumStatus"] == "unavailable"


def test_missing_rss_is_unavailable_and_parity_mismatch_cannot_pass() -> None:
    profile = dict(scale_probe.DIAGNOSTIC_PROFILES[0])

    def missing(profile: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return _fake_execute(profile, missing_rss=True, **kwargs)

    unavailable = scale_probe.run_paired_profile(
        profile,
        baseline={"economicDiagnostics": False, "command": None},
        treatment={"economicDiagnostics": True, "command": None},
        execute=missing,
    )
    assert unavailable["maximumPairedIncrementalPeakWorkerRssMiB"] is None
    assert unavailable["rssThresholdMet"] is None
    assert unavailable["acceptanceStatus"] == "unavailable"

    def mismatched(profile: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return _fake_execute(profile, mismatch=True, **kwargs)

    failed = scale_probe.run_paired_profile(
        profile,
        baseline={"economicDiagnostics": False, "command": None},
        treatment={"economicDiagnostics": True, "command": None},
        execute=mismatched,
    )
    assert failed["parityExact"] is False
    assert failed["acceptanceStatus"] == "threshold-not-met"


def test_old_binary_baseline_is_explicitly_unavailable_when_not_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scale_probe,
        "_producer_identity",
        lambda command: {"command": command, "bridge": {"sha256": "a" * 64}},
    )
    report = scale_probe.paired_diagnostics_probe(execute=_fake_execute)
    assert report["oldBaseline"] == {
        "status": "unavailable",
        "reason": "no reviewed pre-change bridge binary was supplied",
        "requiredRatio": 0.98,
        "measuredValue": None,
    }
    assert report["protocol"]["missingRss"] == "unavailable-not-zero"
