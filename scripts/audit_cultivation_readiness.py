#!/usr/bin/env python3
"""Independently verify the ignored 3x3 readiness diagnostics/checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from anthropocene_rl.contracts import SIMULATOR_COMMIT
from anthropocene_rl.datasets.blobs import sha256_file

EXPECTED_LEARNER_SHA = "33a88006ba9f55f80c47a69b91331009973a75ac"
EXPECTED_FAMILIES = ("F01", "F03", "F13")
EXPECTED_SEEDS = (1701, 1901, 2101)
REQUIRED_RUN_GATES = (
    "inactive_heads_gradient_inert",
    "no_illegal_action_scored",
    "action_channels_preserved",
    "compiler_interventions_not_pathological",
    "finite_and_saturation_guards",
    "checkpoint_reload_policy_parity",
    "checkpoint_continuation_parity",
    "native_replay_valid",
    "movement_disabled",
    "fixed_policy_evaluation_contract",
    "critic_return_fit",
    "ppo_update_stability",
    "fixed_policy_outcome_improves",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path("artifacts/cultivation-readiness.json"),
    )
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path("artifacts/cultivation-readiness-runs"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("checkpoints/cultivation-readiness"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--learner-sha", default=EXPECTED_LEARNER_SHA)
    args = parser.parse_args()

    aggregate = _read_json(args.aggregate)
    _require(aggregate["status"] == "PASS", "aggregate status is not PASS")
    _require(
        all(bool(value) for value in aggregate["gates"].values()),
        "an aggregate gate is false",
    )
    _require(tuple(aggregate["matrix"]["seeds"]) == EXPECTED_SEEDS, "seed matrix drift")
    rows = aggregate["matrix"]["runs"]
    _require(len(rows) == 9, "readiness matrix does not contain nine runs")
    _require(
        {row["scenario_family"] for row in rows} == set(EXPECTED_FAMILIES),
        "scenario-family matrix drift",
    )
    verified_runs = []
    for row in rows:
        family = str(row["scenario_family"])
        seed = int(row["seed"])
        run_name = f"{family.lower()}-seed-{seed}"
        diagnostics_path = args.run_directory / f"{run_name}.json"
        checkpoint_path = args.checkpoint_directory / f"{run_name}.pt"
        diagnostics = _read_json(diagnostics_path)
        _verify_diagnostics_hash(diagnostics)
        _require(diagnostics["status"] == "PASS", f"{run_name}: diagnostics failed")
        _require(
            all(bool(diagnostics["gates"].get(gate)) for gate in REQUIRED_RUN_GATES),
            f"{run_name}: a required run gate failed",
        )
        identities = diagnostics["identities"]
        _require(identities["learner_commit"] == args.learner_sha, f"{run_name}: learner SHA")
        _require(
            identities["learner_tracked_files_dirty"] is False,
            f"{run_name}: learner checkout was dirty",
        )
        _require(
            identities["simulator_commit"] == SIMULATOR_COMMIT,
            f"{run_name}: simulator SHA",
        )
        _require(identities["snapshot_id"] == row["snapshot_id"], f"{run_name}: snapshot")
        _require(diagnostics["gates"] == row["gates"], f"{run_name}: aggregate gate copy")
        _require(
            diagnostics["deterministic_diagnostics_sha256"]
            == row["diagnostics_sha256"],
            f"{run_name}: aggregate diagnostics hash",
        )
        checkpoint_digest, checkpoint_size = sha256_file(checkpoint_path)
        checkpoint_evidence = diagnostics["checkpoint"]
        _require(
            checkpoint_digest == checkpoint_evidence["sha256"],
            f"{run_name}: checkpoint SHA-256",
        )
        _require(
            checkpoint_size == checkpoint_evidence["bytes"],
            f"{run_name}: checkpoint size",
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_identities = checkpoint["identities"]
        for key in (
            "learner_commit",
            "simulator_commit",
            "snapshot_manifest_sha256",
            "snapshot_id",
            "snapshot_content_sha256",
            "runtime_bridge_sha256",
        ):
            _require(
                checkpoint_identities[key] == identities[key],
                f"{run_name}: checkpoint identity {key}",
            )
        _require(checkpoint["overfit_config"]["seed"] == seed, f"{run_name}: seed")
        verified_runs.append(
            {
                "scenario_family": family,
                "seed": seed,
                "snapshot_id": row["snapshot_id"],
                "diagnostics_sha256": diagnostics["deterministic_diagnostics_sha256"],
                "diagnostics_file_sha256": sha256_file(diagnostics_path)[0],
                "checkpoint_sha256": checkpoint_digest,
                "checkpoint_bytes": checkpoint_size,
                "status": "PASS",
            }
        )
    _verify_aggregate_hash(aggregate)
    result = {
        "status": "PASS",
        "scope": "independent-artifact-integrity-and-gate-verification",
        "learner_commit": args.learner_sha,
        "native_simulator_commit": SIMULATOR_COMMIT,
        "aggregate_file_sha256": sha256_file(args.aggregate)[0],
        "aggregate_deterministic_sha256": aggregate[
            "deterministic_diagnostics_sha256"
        ],
        "matrix": {
            "families": list(EXPECTED_FAMILIES),
            "seeds": list(EXPECTED_SEEDS),
            "run_count": len(verified_runs),
            "runs": verified_runs,
        },
        "aggregate_gates": aggregate["gates"],
        "architecture_recommendation": aggregate["architecture_recommendation"],
    }
    if args.output is not None:
        _write_json_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


def _verify_diagnostics_hash(diagnostics: dict[str, Any]) -> None:
    expected = diagnostics["deterministic_diagnostics_sha256"]
    payload = {
        key: value
        for key, value in diagnostics.items()
        if key not in {"timing", "checkpoint", "deterministic_diagnostics_sha256"}
    }
    actual = _canonical_sha256(payload)
    _require(actual == expected, "per-run deterministic diagnostics hash mismatch")


def _verify_aggregate_hash(aggregate: dict[str, Any]) -> None:
    expected = aggregate["deterministic_diagnostics_sha256"]
    payload = {
        key: value
        for key, value in aggregate.items()
        if key not in {"elapsed_seconds", "deterministic_diagnostics_sha256"}
    }
    _require(_canonical_sha256(payload) == expected, "aggregate deterministic hash mismatch")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
