"""Summary of a WP-B paired-branch file (``<rollout>.branches.pt``) as JSON.

    python -m anthropocene_rl.autonomous_v7.labor_branch_report --branches X.branches.pt \
        [--rollout X.pt] [--out report.json]

Reports: visited / branched states and the achieved selection rate against the mean
selection probability; per-head D6 triggers; pairs with an available target; mean |diff|
per objective (raw and 1/p-weighted) for the primary (folded, D3) and the exact target;
importance-weight effective sample size; which action heads differ between the two
branched actions; whether branch a reproduced the rollout's own successor; terminations;
and the cost driver: native steps of the rollout, the H12 shadow and the branches, with wall
seconds per stage and per branched state. With ``--rollout`` the file's rollout sha256 is
checked. Read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .labor_branch_collect import BRANCH_FILE_KIND, COMPONENTS, HEADS


def load_branches(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("kind") != BRANCH_FILE_KIND:
        raise ValueError(f"{path} is not a {BRANCH_FILE_KIND} file")
    return dict(payload)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _target_stats(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    rows = [(r["targets"][method]["diff"], r["selection"]["weight"])
            for r in records if r["targets"].get(method) is not None]
    if not rows:
        return {"pairs": 0, "meanAbsDiff": None, "weightedMeanAbsDiff": None, "meanDiff": None}
    diff = np.asarray([d for d, _ in rows], dtype=np.float64)
    w = np.asarray([x for _, x in rows], dtype=np.float64)
    absd = np.abs(diff)
    return {
        "pairs": len(rows),
        "meanAbsDiff": {h: float(absd[:, i].mean()) for i, h in enumerate(COMPONENTS)},
        "weightedMeanAbsDiff": {
            h: float((w * absd[:, i]).sum() / w.sum()) for i, h in enumerate(COMPONENTS)
        },
        "meanDiff": {h: float(diff[:, i].mean()) for i, h in enumerate(COMPONENTS)},
    }


def _cost_split(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Seconds and native steps of the per-camp b runs and of the shared a runs."""
    b_seconds = [r["work"]["b"]["seconds"] for r in records]
    b_steps = [r["work"]["b"]["nativeSteps"] for r in records]
    decisions: dict[tuple[int, int], Mapping[str, Any]] = {}
    for r in records:
        decisions.setdefault((int(r["world"]), int(r["turn"])), r["work"]["aShared"])
    a = list(decisions.values())
    return {
        "branchedDecisions": len(a),
        "meanCampsPerBranchedDecision": _mean([x["sharedBy"] for x in a]),
        "bSecondsPerBranchedState": _mean(b_seconds),
        "bNativeStepsPerBranchedState": _mean(b_steps),
        "aSecondsPerBranchedDecision": _mean([x["seconds"] for x in a]),
        "aNativeStepsPerBranchedDecision": _mean([x["nativeSteps"] for x in a]),
    }


def summarize(payload: Mapping[str, Any], *, rollout: Path | None = None) -> dict[str, Any]:
    records = list(payload["records"])
    states = list(payload["states"])
    config = dict(payload["config"])
    telemetry = dict(payload["telemetry"])
    weights = np.asarray([r["selection"]["weight"] for r in records], dtype=np.float64)
    probabilities = [s["probability"] for s in states]
    triggers: dict[str, int] = {}
    calibration = payload.get("calibration")
    for s in states:
        for h in s.get("triggers") or ():
            triggers[h] = triggers.get(h, 0) + 1
    differs = {h: sum(bool(r["headDiffers"].get(h)) for r in records) for h in HEADS}
    terminated = {
        name: sum(any(b["terminated"] for b in r["branches"][name]) for r in records)
        for name in ("a", "b")
    }
    out: dict[str, Any] = {
        "kind": payload["kind"],
        "schemaVersion": payload["schemaVersion"],
        "provenance": dict(payload["provenance"]),
        "config": config,
        "states": len(states),
        "branchedStates": len(records),
        "selectionRate": len(records) / len(states) if states else None,
        "meanSelectionProbability": _mean(probabilities),
        "calibratedStates": sum(bool(s.get("calibrated")) for s in states),
        "farStates": sum(bool(s.get("far")) for s in states),
        "headTriggers": triggers,
        "calibration": calibration,
        "targets": {"primary": "folded", "folded": _target_stats(records, "folded"),
                    "exact": _target_stats(records, "exact")},
        "unavailableTargets": sum(r["targets"].get("folded") is None for r in records),
        "importanceWeights": {
            "sum": float(weights.sum()) if len(weights) else 0.0,
            "mean": _mean(weights.tolist()),
            "max": float(weights.max()) if len(weights) else None,
            "effectiveSampleSize": (
                float(weights.sum() ** 2 / (weights**2).sum()) if len(weights) else 0.0
            ),
        },
        "headsDiffering": differs,
        "identicalActionPairs": sum(bool(r["identicalActions"]) for r in records),
        "aFirstMatchesRollout": {
            "true": sum(bool(r["aFirstMatchesRollout"]) for r in records),
            "false": sum(not r["aFirstMatchesRollout"] for r in records),
        },
        "terminatedBranches": terminated,
        "focalRetries": {
            "aRollout": sum(bool(r["actions"]["a"].get("retried")) for r in records),
            "branchDecisions": sum(
                bool(e["focalRetried"]) for r in records for name in ("a", "b")
                for b in r["branches"][name] for e in b["decisions"]
            ),
        },
        "telemetry": telemetry,
        "cost": {
            # Cost model per collect: decisions x shadow + branched decisions x shared a
            # + branched states x b (a is one run per (decision, stream) for all its camps).
            **_cost_split(records),
            "nativeSteps": telemetry["nativeSteps"],
            "branchStepsPerBranchedState": telemetry.get("branchStepsPerBranchedState"),
            "secondsPerBranchedState": telemetry.get("overheadSecondsPerBranchedState"),
            "branchSecondsPerBranchedState": telemetry.get("branchSecondsPerBranchedState"),
            "shadowSecondsPerDecision": (
                telemetry["seconds"]["shadow"] / telemetry["decisions"]
                if telemetry["decisions"] else None
            ),
            "rolloutSecondsPerDecision": (
                telemetry["seconds"]["rolloutDerived"] / telemetry["decisions"]
                if telemetry["decisions"] else None
            ),
        },
    }
    if rollout is not None:
        sha = hashlib.sha256(Path(rollout).read_bytes()).hexdigest()
        out["rolloutSha256Matches"] = sha == payload["provenance"].get("rolloutSha256")
    return out


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    report = summarize(load_branches(args.branches), rollout=args.rollout)
    text = json.dumps(report, indent=2, sort_keys=True, default=float)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    if args.rollout is not None and not report["rolloutSha256Matches"]:
        print("branch file is bound to another rollout", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
