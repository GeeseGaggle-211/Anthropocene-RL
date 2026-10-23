"""One reviewed, no-learning Gate 3 debug continuation.

This entry point is intentionally not a configurable collector.  It binds the
T3per-s70 endpoint, current world 50000's default-regime first decision, and the
existing four-path/two-alternative randomized-continuation schedules.  It writes
the complete plan before starting a bridge subprocess and cannot claim a frozen
grid, readiness, selection, or gate passage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from .bridge import Model9Bridge
from .gate1_production_adapter import tensor_state_identity
from .gate34_checkpoint_collector import (
    FROZEN_ALTERNATIVES,
    FROZEN_PROTOCOL,
    EndpointRuntime,
    PersonalityState,
    _collect_frozen_state,
    _derived_seed,
    load_endpoint_registry,
    load_endpoint_runtime,
)
from .provenance import ProvenanceError, digest

Json = dict[str, Any]

ENDPOINT_ID = "t3per-s70"
WORLD_SEED = 50000
REGIME_ID = "default"
WITHHELD_MASK = 0
DECISION_STEP = 0
NATIVE_LABEL_PROBE_STEPS = 1
EXPECTED_RESET_TURN = 9
DEFAULT_TIMEOUT_SECONDS = 3600
MAX_TIMEOUT_SECONDS = 7200
PLAN_KIND = "gate34-single-state-debug-plan"
RESULT_KIND = "gate34-single-state-debug-result"
WORKER_MODULE = "anthropocene_rl.autonomous_v7.gate34_single_state_debug"
TORCH_NUM_THREADS = 4


def _read_json(path: Path) -> Json:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProvenanceError(f"JSON {path} must contain an object")
    return cast(Json, value)


def _write_sealed(path: Path, body: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite debug evidence: {path}")
    value = dict(body)
    if "digest" in value:
        raise ValueError("body is already sealed")
    value["digest"] = digest(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary evidence exists: {temporary}")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_sealed_plan(path: Path) -> Json:
    plan = _read_json(path)
    if plan.get("kind") != PLAN_KIND or plan.get("schema") != 1:
        raise ProvenanceError("single-state debug plan kind/schema differs")
    expected = digest({key: value for key, value in plan.items() if key != "digest"})
    if plan.get("digest") != expected:
        raise ProvenanceError("single-state debug plan digest differs")
    fixed = plan.get("fixedScope")
    if not isinstance(fixed, dict) or fixed != {
        "endpointId": ENDPOINT_ID,
        "worldSeed": WORLD_SEED,
        "regimeId": REGIME_ID,
        "withheldMask": WITHHELD_MASK,
        "decisionStep": DECISION_STEP,
        "alternatives": list(FROZEN_ALTERNATIVES),
        "pathReplicates": FROZEN_PROTOCOL.path_replicates,
        "captureEconomic": True,
    }:
        raise ProvenanceError("single-state debug plan scope is not the reviewed scope")
    return plan


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _trace_reset_binding(trace_root: Path, checkpoint_sha256: str, bridge_sha256: str) -> Json:
    """Bind the saved vector smoke's native reset identity, not its trajectory ID."""

    validation_path = trace_root / "validation.json"
    manifest_path = trace_root / "trace/manifest.json"
    transitions_path = trace_root / "trace/transitions.jsonl"
    validation = _read_json(validation_path)
    manifest = _read_json(manifest_path)
    if validation.get("status") != "pass" or validation.get("worldSeed") != WORLD_SEED:
        raise ProvenanceError("saved vector trace is not the approved world-50000 pass")
    checkpoint = validation.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("sha256") != checkpoint_sha256:
        raise ProvenanceError("saved vector trace checkpoint differs")
    if manifest.get("nonqualifying") is not True:
        raise ProvenanceError("saved trace must remain explicitly nonqualifying")

    first_turn_rows: list[Json] = []
    with transitions_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ProvenanceError("trace transition row is not an object")
            turn = int(row.get("decisionTurn", -1))
            if not first_turn_rows:
                if turn != EXPECTED_RESET_TURN:
                    raise ProvenanceError("trace first decision is not the reviewed reset turn")
            elif turn != EXPECTED_RESET_TURN:
                break
            first_turn_rows.append(cast(Json, row))
    if not first_turn_rows:
        raise ProvenanceError("saved vector trace has no first-decision rows")

    reset_keys = (
        "configurationHash",
        "fingerprint",
        "buildHash",
        "rawMaterialsEnabled",
        "sourceSeed",
    )
    identities: list[Json] = []
    snapshot_hashes: set[str] = set()
    normalizer_refs: set[str] = set()
    trajectory_ids: set[str] = set()
    for row in first_turn_rows:
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            raise ProvenanceError("trace row lacks provenance")
        if provenance.get("bridgeSha256") != bridge_sha256:
            raise ProvenanceError("trace row bridge differs from the frozen bridge")
        if row.get("worldSeed") != WORLD_SEED or row.get("policyVersion") != 0:
            raise ProvenanceError("trace first-decision scope differs")
        identity = {key: provenance.get(key) for key in reset_keys}
        if identity["fingerprint"] != provenance.get("preFingerprint"):
            raise ProvenanceError("trace reset fingerprint and pre-fingerprint differ")
        identities.append(identity)
        snapshot_hashes.add(str(provenance.get("resetSnapshotHash")))
        normalizer_refs.add(str(row.get("normalizerMomentsRef")))
        trajectory_ids.add(str(row.get("trajectoryId")))
    if any(identity != identities[0] for identity in identities[1:]):
        raise ProvenanceError("trace first-decision rows disagree on native reset identity")
    if len(snapshot_hashes) != 1 or len(normalizer_refs) != 1:
        raise ProvenanceError("trace first-decision rows disagree on immutable inputs")
    snapshot_hash = snapshot_hashes.pop()
    snapshot_path = trace_root / "trace/snapshots" / f"{snapshot_hash}.anthro"
    if _sha256(snapshot_path) != snapshot_hash:
        raise ProvenanceError("saved trace reset snapshot bytes differ")
    normalizer_reference = normalizer_refs.pop()
    normalizer_path = (
        trace_root / "trace/normalizers" / f"{normalizer_reference}.json"
    ).resolve()
    try:
        normalizer_path.relative_to((trace_root / "trace").resolve())
    except ValueError as error:
        raise ProvenanceError("trace normalizer reference escapes trace root") from error
    normalizer_hash = normalizer_path.stem
    if _sha256(normalizer_path) != normalizer_hash:
        raise ProvenanceError("saved trace normalizer bytes differ")
    return {
        "bindingScope": (
            "exact native reset-state identity only; this probe does not replay or "
            "label the smoke's sampled personalities, actions, or optimizer rows"
        ),
        "validationPath": str(validation_path.resolve()),
        "validationSha256": _sha256(validation_path),
        "manifestPath": str(manifest_path.resolve()),
        "manifestSha256": _sha256(manifest_path),
        "transitionsPath": str(transitions_path.resolve()),
        "transitionsSha256": _sha256(transitions_path),
        "snapshotPath": str(snapshot_path.resolve()),
        "snapshotSha256": snapshot_hash,
        "normalizerPath": str(normalizer_path),
        "normalizerSha256": normalizer_hash,
        "firstDecisionRowCount": len(first_turn_rows),
        "nativeResetIdentity": identities[0],
        "trajectoryIdsObservedButNotUsedAsFingerprint": sorted(trajectory_ids),
    }


def _runtime_json(runtime: EndpointRuntime) -> Json:
    return {
        "vectorCritic": runtime.vector_critic,
        "objectives": list(runtime.objectives),
        "personality": runtime.personality_spec,
        "personalityDrift": runtime.personality_drift,
        "logStdFloor": runtime.log_std_floor,
        "mixChangeCost": runtime.mix_change_cost,
        "breakCost": runtime.break_cost,
        "rewardSemantics": runtime.reward_semantics,
        "returnContract": runtime.return_contract_id,
        "stage": "mixed",
        "noOptimizerStep": True,
        "torchNumThreads": torch.get_num_threads(),
    }


def build_plan(
    freeze_root: Path,
    trace_root: Path,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Json:
    if (
        isinstance(timeout_seconds, bool)
        or int(timeout_seconds) != timeout_seconds
        or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        raise ValueError(f"timeout_seconds must be in [1, {MAX_TIMEOUT_SECONDS}]")
    torch.set_num_threads(TORCH_NUM_THREADS)
    registry = load_endpoint_registry(freeze_root)
    receipt = next(
        (item for item in registry.endpoints if item.node_id == ENDPOINT_ID), None
    )
    if receipt is None:
        raise ProvenanceError("reviewed endpoint is absent from the frozen registry")
    runtime = load_endpoint_runtime(registry, receipt)
    if not runtime.vector_critic or tuple(runtime.objectives) != (
        "welfare",
        "population_years",
        "trade_utility",
        "qol_years",
    ):
        raise ProvenanceError("reviewed endpoint is not the four-head vector critic")
    trace = _trace_reset_binding(
        trace_root.resolve(), receipt.checkpoint_sha256, registry.bridge_sha256
    )
    schedules = []
    path_randomness = []
    branch_maximum = 0
    for path_replicate in range(FROZEN_PROTOCOL.path_replicates):
        schedule = FROZEN_PROTOCOL.stopping_schedule(
            ENDPOINT_ID, WORLD_SEED, REGIME_ID, DECISION_STEP, path_replicate
        )
        maximum = FROZEN_PROTOCOL.prefix_terms + (
            0 if schedule.exceeds_hard_cap else schedule.required_terms
        )
        branch_maximum += len(FROZEN_ALTERNATIVES) * maximum
        schedules.append(
            {
                "pathReplicate": path_replicate,
                "scheduleId": schedule.schedule_id,
                "rngSeed": schedule.rng_seed,
                "stopIndices": list(schedule.stop_indices),
                "requiredTailTerms": schedule.required_terms,
                "exceedsHardCap": schedule.exceeds_hard_cap,
                "maximumStepsPerAlternative": maximum,
            }
        )
        path_randomness.append(
            {
                "pathReplicate": path_replicate,
                "actionSeed": _derived_seed(
                    "gate34-action-v1",
                    ENDPOINT_ID,
                    WORLD_SEED,
                    REGIME_ID,
                    DECISION_STEP,
                    path_replicate,
                ),
                "personalitySeed": _derived_seed(
                    "gate34-personality-v1",
                    ENDPOINT_ID,
                    WORLD_SEED,
                    REGIME_ID,
                    DECISION_STEP,
                    path_replicate,
                ),
            }
        )
    maximum_steps = NATIVE_LABEL_PROBE_STEPS + branch_maximum
    if maximum_steps != 727:
        raise ProvenanceError("reviewed single-state maximum-step calculation changed")
    return {
        "kind": PLAN_KIND,
        "schema": 1,
        "status": "planned-not-executed",
        "reviewAuthority": "root-reviewed-single-state-debug-2026-10-23",
        "fixedScope": {
            "endpointId": ENDPOINT_ID,
            "worldSeed": WORLD_SEED,
            "regimeId": REGIME_ID,
            "withheldMask": WITHHELD_MASK,
            "decisionStep": DECISION_STEP,
            "alternatives": list(FROZEN_ALTERNATIVES),
            "pathReplicates": FROZEN_PROTOCOL.path_replicates,
            "captureEconomic": True,
        },
        "registry": {
            "freezeRoot": str(freeze_root.resolve()),
            "registrySha256": registry.registry_sha256,
            "checkpointPath": str(receipt.checkpoint),
            "checkpointSha256": receipt.checkpoint_sha256,
            "criticTypedContentSha256": receipt.critic_typed_content_sha256,
            "bridgePath": str(registry.bridge_path),
            "bridgeSha256": registry.bridge_sha256,
            "sourceHashes": dict(registry.source_hashes),
        },
        "runtime": _runtime_json(runtime),
        "traceResetBinding": trace,
        "schedules": schedules,
        "continuationRandomness": {
            "scope": (
                "new frozen debug continuation schedules; not the saved smoke action "
                "or personality RNG stream"
            ),
            "mainPersonalitySeed": _derived_seed(
                "gate34-main-personality-v1", ENDPOINT_ID, WORLD_SEED, REGIME_ID
            ),
            "paths": path_randomness,
        },
        "maximumBridgeStepCalls": maximum_steps,
        "maximumStepAccounting": {
            "nativeLabelProbe": NATIVE_LABEL_PROBE_STEPS,
            "branchSteps": branch_maximum,
            "resetOperationExcluded": True,
        },
        "timeoutSeconds": int(timeout_seconds),
        "independentUnit": "one world",
        "nestedMeasurements": "four stochastic paths; paths are not independent worlds",
        "tradeTailRule": "withhold full-return sign/ranking unless terminal",
        "claimsExcluded": [
            "full-grid-completion",
            "gate-passage",
            "selection",
            "readiness",
            "checkpoint-promotion",
            "finite-world-population-inference",
            "optimizer-causality",
        ],
    }


def _verify_worker_inputs(plan: Json):
    torch.set_num_threads(TORCH_NUM_THREADS)
    registry_data = cast(Json, plan["registry"])
    freeze_root = Path(str(registry_data["freezeRoot"]))
    registry = load_endpoint_registry(freeze_root)
    if registry.registry_sha256 != registry_data["registrySha256"]:
        raise ProvenanceError("registry changed after plan persistence")
    receipt = next(item for item in registry.endpoints if item.node_id == ENDPOINT_ID)
    if (
        receipt.checkpoint_sha256 != registry_data["checkpointSha256"]
        or receipt.critic_typed_content_sha256
        != registry_data["criticTypedContentSha256"]
        or registry.bridge_sha256 != registry_data["bridgeSha256"]
    ):
        raise ProvenanceError("checkpoint or bridge changed after plan persistence")
    runtime = load_endpoint_runtime(registry, receipt)
    if _runtime_json(runtime) != plan["runtime"]:
        raise ProvenanceError("runtime contract changed after plan persistence")
    trace = cast(Json, plan["traceResetBinding"])
    for key in (
        "validationPath",
        "manifestPath",
        "transitionsPath",
        "snapshotPath",
        "normalizerPath",
    ):
        hash_key = key.removesuffix("Path") + "Sha256"
        if _sha256(Path(str(trace[key]))) != trace[hash_key]:
            raise ProvenanceError(f"planned trace input changed: {key}")
    return registry, runtime


def _append_progress(path: Path, record: Json) -> None:
    value = dict(record)
    value["recordDigest"] = digest(value)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_worker(plan_path: Path) -> int:
    plan = _load_sealed_plan(plan_path)
    output = plan_path.parent
    registry, runtime = _verify_worker_inputs(plan)
    progress_path = output / "paths.jsonl"
    if progress_path.exists() or (output / "result.json").exists():
        raise FileExistsError("single-state debug output already contains native results")
    bridge: Model9Bridge | None = None
    progress: list[Json] = []
    started = time.monotonic()
    before_models = {
        "actorTypedContentSha256": tensor_state_identity(
            runtime.actor.state_dict()
        )["typed_content_sha256"],
        "criticTypedContentSha256": tensor_state_identity(
            runtime.critic.state_dict()
        )["typed_content_sha256"],
    }
    fork_restore_verification: Json | None = None
    try:
        dotnet = os.environ.get("ANTHROPOCENE_DOTNET", "dotnet")
        bridge = Model9Bridge(
            command=[dotnet, str(registry.bridge_path)], economic_diagnostics=True
        )
        current = bridge.reset(
            WORLD_SEED,
            movement=runtime.stage.bridge_movement,
            withheld=WITHHELD_MASK,
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
            raise ProvenanceError("live native reset identity differs from saved vector trace")
        if (
            int(current["turn"]) != EXPECTED_RESET_TURN
            or str(current["fingerprint"]) != str(expected_reset["fingerprint"])
        ):
            raise ProvenanceError("live reset decision identity differs from saved trace")
        personality_seed = _derived_seed(
            "gate34-main-personality-v1", ENDPOINT_ID, WORLD_SEED, REGIME_ID
        )
        personalities = PersonalityState(
            runtime.personality_spec, runtime.personality_drift, personality_seed
        )

        def save_progress(record: Json) -> None:
            if int(record["collectedSteps"]) > int(record["maximumSteps"]):
                raise RuntimeError("path exceeded its persisted maximum step count")
            numbered = {
                "sequence": len(progress),
                **record,
                "cumulativeCompletedBranchSteps": sum(
                    int(item["collectedSteps"]) for item in progress
                )
                + int(record["collectedSteps"]),
            }
            progress.append(numbered)
            _append_progress(progress_path, numbered)

        state = _collect_frozen_state(
            bridge=bridge,
            current=current,
            runtime=runtime,
            protocol=FROZEN_PROTOCOL,
            world_seed=WORLD_SEED,
            regime_id=REGIME_ID,
            decision_step=DECISION_STEP,
            history=(),
            main_personalities=personalities,
            previous_mixes={},
            capture_economic=True,
            path_progress=save_progress,
        )
        verification_fork, restored = bridge.fork()
        try:
            restored_identity = {
                "turn": int(restored["turn"]),
                "fingerprint": str(restored["fingerprint"]),
            }
            expected_identity = {
                "turn": int(current["turn"]),
                "fingerprint": str(current["fingerprint"]),
            }
            if restored_identity != expected_identity:
                raise ProvenanceError(
                    "bridge did not return to the exact frozen state after all paths"
                )
            fork_restore_verification = {
                "status": "exact",
                "expected": expected_identity,
                "actual": restored_identity,
                "trajectoryIdUsedAsFingerprint": False,
            }
        finally:
            bridge.drop(verification_fork)
    finally:
        if bridge is not None:
            bridge.close()
    after_models = {
        "actorTypedContentSha256": tensor_state_identity(
            runtime.actor.state_dict()
        )["typed_content_sha256"],
        "criticTypedContentSha256": tensor_state_identity(
            runtime.critic.state_dict()
        )["typed_content_sha256"],
    }
    if after_models != before_models:
        raise RuntimeError("no-learning probe mutated actor or critic tensors")
    actual_steps = NATIVE_LABEL_PROBE_STEPS + sum(
        int(item["collectedSteps"]) for item in progress
    )
    if len(progress) != FROZEN_PROTOCOL.path_replicates * len(FROZEN_ALTERNATIVES):
        raise RuntimeError("single-state debug did not complete every planned path")
    if actual_steps > int(plan["maximumBridgeStepCalls"]):
        raise RuntimeError("actual bridge step calls exceed the persisted plan maximum")
    result = {
        "kind": RESULT_KIND,
        "schema": 1,
        "status": "single-state-debug-complete-no-gate-claim",
        "planDigest": plan["digest"],
        "fixedScope": plan["fixedScope"],
        "registry": plan["registry"],
        "runtime": plan["runtime"],
        "traceResetBinding": plan["traceResetBinding"],
        "stepAccounting": {
            "maximumBridgeStepCalls": plan["maximumBridgeStepCalls"],
            "actualBridgeStepCalls": actual_steps,
            "nativeLabelProbeSteps": NATIVE_LABEL_PROBE_STEPS,
            "completedPathCount": len(progress),
            "paths": progress,
        },
        "durationSeconds": time.monotonic() - started,
        "noLearningMutationProof": {
            "optimizerConstructed": False,
            "optimizerStepCalled": False,
            "before": before_models,
            "after": after_models,
            "actorAndCriticUnchanged": True,
        },
        "forkRestoration": fork_restore_verification,
        "state": state,
        "independentUnit": "one world",
        "nestedMeasurements": "four stochastic paths; paths are not independent worlds",
        "claimsExcluded": plan["claimsExcluded"],
    }
    _write_sealed(output / "result.json", result)
    return 0


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Stop the worker and its bridge child after the hard wall-clock timeout."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            timeout=30,
        )
        if completed.returncode != 0 and process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def run_single_state_debug(
    freeze_root: Path,
    trace_root: Path,
    output: Path,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Json:
    """Persist the reviewed plan, then execute it in one bounded worker process."""

    plan = build_plan(
        freeze_root.resolve(), trace_root.resolve(), timeout_seconds=timeout_seconds
    )
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "plan.json"
    _write_sealed(plan_path, plan)
    stdout_path = output / "worker.stdout.log"
    stderr_path = output / "worker.stderr.log"
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_root, environment.get("PYTHONPATH", ""))
        if value
    )
    command = _worker_command(plan_path)
    options: Json = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[3]),
            env=environment,
            stdout=stdout,
            stderr=stderr,
            **options,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            _terminate_process_tree(process)
            _write_sealed(
                output / "timeout.json",
                {
                    "kind": "gate34-single-state-debug-timeout",
                    "schema": 1,
                    "status": "timed-out-incomplete-no-gate-claim",
                    "planDigest": _load_sealed_plan(plan_path)["digest"],
                    "timeoutSeconds": timeout_seconds,
                    "partialPathsPreserved": (output / "paths.jsonl").exists(),
                },
            )
            raise TimeoutError("single-state debug exceeded its hard timeout") from error
        except BaseException:
            _terminate_process_tree(process)
            raise
    if return_code != 0:
        raise RuntimeError(
            f"single-state debug worker exited {return_code}; inspect {stderr_path}"
        )
    return _read_json(output / "result.json")


def _worker_command(plan_path: Path) -> list[str]:
    """Use the canonical module name even when the parent itself runs as ``__main__``."""

    return [
        sys.executable,
        "-m",
        WORKER_MODULE,
        "--worker-plan",
        str(plan_path),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the reviewed one-world, one-state no-learning critic debug probe"
    )
    parser.add_argument("--freeze-package", type=Path)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_plan is not None:
        if any(value is not None for value in (args.freeze_package, args.trace_root, args.out)):
            parser.error("--worker-plan is an internal exclusive mode")
        return _run_worker(args.worker_plan.resolve())
    if args.freeze_package is None or args.trace_root is None or args.out is None:
        parser.error("--freeze-package, --trace-root, and --out are required")
    run_single_state_debug(
        args.freeze_package,
        args.trace_root,
        args.out,
        timeout_seconds=args.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
