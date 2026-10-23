"""One-seed local wiring check for the prospective economic optimizer trace.

This is deliberately nonqualifying.  It runs one real PPO iteration twice from
the same checkpoint: once without economic tracing and once with tracing and the
bounded optimizer audit.  It never evaluates or selects a treatment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import ppo


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encoded(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _encoded(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_encoded(child) for child in value]
    return value


def _digest(value: Any) -> str:
    blob = json.dumps(
        _encoded(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def _different_tensors(before: dict, after: dict) -> int:
    if before.keys() != after.keys():
        raise ValueError("model state keys changed")
    return sum(_digest(before[name]) != _digest(after[name]) for name in before)


def _checkpoint_personality(state: dict, hyper: ppo.Hyper) -> dict[str, Any]:
    if not state.get("vector_critic"):
        return {
            "critic": "scalar",
            "personality": hyper.personality,
            "personalityScales": hyper.personality_scales,
            "personalityClip": hyper.personality_clip,
            "configurationSemantics": "current-validation-defaults",
        }
    saved = state.get("hyper", {})
    personality = saved.get("personality")
    if not isinstance(personality, str) or personality == "default":
        raise ValueError("vector validation checkpoint lacks its personality configuration")
    scales = saved.get("personality_scales", "batch")
    if scales not in ("batch", "reference"):
        raise ValueError("checkpoint has invalid personality scale semantics")
    clip = float(saved.get("personality_clip", 0.0 if scales == "batch" else 5.0))
    hyper.personality = personality
    hyper.personality_scales = scales
    hyper.personality_clip = clip
    hyper.personality_drift = float(saved.get("personality_drift", hyper.personality_drift))
    return {
        "critic": "vector",
        "personality": personality,
        "personalityScales": scales,
        "personalityClip": clip,
        "personalityDrift": hyper.personality_drift,
        "initializationKind": "diagnostic-from-checkpoint-weights-not-historical-resume",
        "configurationSemantics": (
            "checkpoint-explicit"
            if "personality_scales" in saved and "personality_clip" in saved
            else "smoke-explicit-batch-no-clip-checkpoint-fields-absent"
        ),
    }


def _models(init: Path, seed: int):
    from .checkpoint import actor_parameters, restore_actor
    from .model import Actor, Critic, VectorCritic, load_checkpoint_normalizer
    from .upgrade import upgrade

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    state = upgrade(torch.load(init, map_location="cpu", weights_only=False))
    if state.get("stage") != "mixed":
        raise ValueError("smoke requires an approved mixed-stage checkpoint")
    core, token = int(state["core_width"]), bool(state.get("token", False))
    actor = Actor(core, experts=bool(state.get("experts", False)), token=token)
    teacher = Actor(core, token=token)
    critic_class = VectorCritic if state.get("vector_critic") else Critic
    critic = critic_class(
        core, privileged_width=int(state.get("privileged_width", 0)), token=token
    )
    cost = ppo.CostCritic(core, token=token)
    missing, unexpected = restore_actor(actor, state, strict=False)
    if unexpected or any(not key.startswith("experts.") for key in missing):
        raise ValueError(f"checkpoint actor mismatch: missing={missing}, unexpected={unexpected}")
    critic.load_state_dict(state["critic"])
    cost.load_state_dict(state["cost_critic"])
    teacher.load_state_dict(state.get("teacher_actor", state["actor"]))
    teacher.eval()
    teacher.set_region_head(actor.region_head)
    normalizer = load_checkpoint_normalizer(state)
    hyper = ppo.Hyper(epochs=1, batch=16, critic_warmup=0, round_start_warmup=0, kl_refit=False)
    configuration = _checkpoint_personality(state, hyper)
    optimizers = (
        torch.optim.Adam(actor_parameters(actor), lr=hyper.lr),
        torch.optim.Adam(critic.parameters(), lr=hyper.critic_lr),
        torch.optim.Adam(cost.parameters(), lr=hyper.critic_lr),
    )
    return state, actor, critic, cost, teacher, normalizer, hyper, optimizers, configuration


def _rollout_record(roll, normalizer) -> dict[str, Any]:
    batch, draws, logp = ppo.flatten(roll)
    actions = ppo.to_bridge_actions(draws, ppo.to_tensors(batch, normalizer))
    rewards = ppo.data.lineage_rewards(batch)
    return {
        "digest": _digest(
            {
                "obs": roll.obs,
                "bootstrap": roll.bootstrap_obs,
                "draws": roll.draws,
                "logp": roll.logp,
                "outcome": roll.outcome,
                "actions": actions,
                "world": roll.world_contracts,
            }
        ),
        "initialObservationDigest": _digest(roll.obs[0]),
        "actionDigest": _digest(actions),
        "rewardDigest": _digest(rewards),
        "rewardSum": float(rewards.sum()),
        "decisions": len(logp),
        "producerMetadataDigest": _digest(roll.worker_contracts),
    }


def _lane(args) -> int:
    values = _models(args.init, args.optimizer_seed)
    state, actor, critic, cost, teacher, normalizer, hyper, optimizers, configuration = values
    traced = args.lane == "on"
    hyper.optimizer_audit_rows = 16 if traced else 0
    initial = {
        name: copy.deepcopy(net.state_dict())
        for name, net in (("actor", actor), ("critic", critic), ("cost", cost))
    }
    generator = torch.Generator().manual_seed(args.optimizer_seed)
    rng = np.random.default_rng(args.optimizer_seed)
    writer = None
    if traced:
        from .economic_trace import EconomicTraceWriter

        writer = EconomicTraceWriter(
            args.trace,
            {
                "runId": str(args.out.resolve()),
                "arm": "diagnostic",
                "optimizerSeed": args.optimizer_seed,
                "nonqualifying": True,
                "validationScope": "local-wiring-only-no-selection-readiness-or-causal-claim",
                "initialCheckpointSha256": _sha256(args.init),
            },
        )
    captured = []
    original_collect = ppo.collect
    retention = None
    if traced and getattr(args, "retain_audit", False):
        from .optimizer_exact_replay import SmokeAuditRetention

        retention = SmokeAuditRetention(
            ppo, actor, optimizers[0], generator, rng
        )

    def collect(*values, **options):
        roll = original_collect(*values, **options)
        captured.append(roll)
        return roll

    ppo.collect = collect
    try:
        if retention is not None:
            retention.__enter__()
        lagrange, stats = ppo.iteration(
            actor,
            critic,
            cost,
            teacher,
            normalizer,
            [args.world_seed],
            args.turns,
            ppo.STAGES["mixed"],
            hyper,
            optimizers,
            float(state.get("lagrange", 0.0)),
            generator,
            rng,
            economic_trace=writer,
        )
        if writer is not None:
            writer.close()
            writer = None
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        if retention is not None:
            retention.__exit__(None, None, None)
        ppo.collect = original_collect
    if len(captured) != 1:
        raise ValueError("expected exactly one real PPO collection")
    payload = {
        "initial": initial,
        "final": {
            name: copy.deepcopy(net.state_dict())
            for name, net in (("actor", actor), ("critic", critic), ("cost", cost))
        },
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "rng": {
            "torch": torch.get_rng_state(),
            "generator": generator.get_state(),
            "numpy": rng.bit_generator.state,
            "numpyGlobal": np.random.get_state(),
            "python": random.getstate(),
        },
        "rollout": _rollout_record(captured[0], normalizer),
        "stats": stats,
        "lagrange": lagrange,
        "configuration": configuration,
    }
    if retention is not None:
        payload["auditRetention"] = retention.payload(
            normalizer_state=normalizer.state_dict(),
            rollout=captured[0],
            stage=ppo.STAGES["mixed"],
            log_std_floor=hyper.log_std_floor,
        )
    torch.save(payload, args.state)
    return 0


def _compare(off: dict, on: dict) -> dict[str, Any]:
    def behavioral_stats(value):
        return {key: child for key, child in value.items() if key != "worker_contracts"}
    for name in ("initial", "final", "optimizers", "rng", "lagrange", "configuration"):
        if _digest(off[name]) != _digest(on[name]):
            raise ValueError(f"traced/untraced mismatch: {name}")
    rollout_fields = (
        "digest",
        "initialObservationDigest",
        "actionDigest",
        "rewardDigest",
        "rewardSum",
        "decisions",
    )
    if any(
        _digest(off["rollout"][name]) != _digest(on["rollout"][name]) for name in rollout_fields
    ):
        raise ValueError("traced/untraced mismatch: rollout")
    if _digest(behavioral_stats(off["stats"])) != _digest(behavioral_stats(on["stats"])):
        raise ValueError("traced/untraced mismatch: stats")
    changed = {
        name: _different_tensors(on["initial"][name], on["final"][name])
        for name in ("actor", "critic", "cost")
    }
    if changed["actor"] == 0 or changed["critic"] == 0:
        raise ValueError("smoke did not exercise both actor and critic parameter updates")
    rollout = {name: on["rollout"][name] for name in rollout_fields}
    return {
        "rollout": rollout,
        "producerMetadataDigests": {
            "off": off["rollout"].get("producerMetadataDigest"),
            "on": on["rollout"].get("producerMetadataDigest"),
        },
        "changedParameterTensors": changed,
        "finalModelDigest": _digest(on["final"]),
        "optimizerDigest": _digest(on["optimizers"]),
        "rngDigest": _digest(on["rng"]),
    }


def _trace_summary(path: Path, decisions: int, critic: str) -> dict[str, Any]:
    from .economic_diagnostics import load_artifact

    _, transitions, updates = load_artifact(path)
    if len(transitions) != decisions or any(
        row.get("economicDiagnostics", {}).get("schemaVersion") != 2 for row in transitions
    ):
        raise ValueError("trace transition count/schema differs from real rollout")
    starts = [row for row in updates if row.get("event") == "mutation-start"]
    completes = [row for row in updates if row.get("event") == "mutation-complete"]
    if (
        not starts
        or len(starts) != len(completes)
        or any(
            a["batchRowDigest"] != b["batchRowDigest"]
            for a, b in zip(starts, completes, strict=True)
        )
    ):
        raise ValueError("trace mutation brackets are incomplete")
    audit = completes[0].get("optimizerDebug", {})
    scope, same = audit.get("scope", {}), audit.get("sameSampledAction", {})
    expected_audit_rows = min(16, decisions)
    if (
        audit.get("schemaVersion") != 1
        or scope.get("expensiveAudit") is not True
        or scope.get("sampledRowCount") != expected_audit_rows
        or len(scope.get("auditTransitionIds", [])) != expected_audit_rows
        or len(same.get("rows", [])) != expected_audit_rows
        or "actorGradientGeometry" not in audit
        or "lossContributions" not in audit
        or "actualClip" not in audit
    ):
        raise ValueError("bounded optimizer audit is missing required evidence")
    from .contracts import OBJECTIVES

    head_diagnostics = audit.get("criticHeadDiagnostics", {})
    if critic == "vector":
        geometry = audit.get("vectorCriticSharedEncoderGradientGeometry", {})
        if (
            set(head_diagnostics) != set(OBJECTIVES)
            or set(geometry.get("components", {})) != set(OBJECTIVES)
            or len(geometry.get("pairs", [])) != 6
        ):
            raise ValueError("vector critic audit lacks all per-head diagnostics or gradient pairs")
    elif (
        set(head_diagnostics) != {"value"}
        or audit.get("unavailable", {}).get("vectorCriticSharedEncoderGradientGeometry")
        != "scalar-critic-no-per-head-shared-encoder-conflict"
    ):
        raise ValueError("scalar critic audit lacks its explicit unavailable classification")
    if not any(abs(float(row["jointDelta"])) > 0 for row in same["rows"]):
        raise ValueError("same-action probability audit recorded no actor change")
    if len([row for row in updates if row.get("event") == "iteration-summary"]) != 1:
        raise ValueError("trace must contain exactly one iteration summary")
    result = {
        "transitions": len(transitions),
        "updates": len(updates),
        "mutationBatches": len(completes),
        "optimizerAuditRows": expected_audit_rows,
        "jointLogProbabilityMeanDelta": same["joint"]["meanDelta"],
        "economicControlRows": sum(
            len(row["economicDiagnostics"].get("controls", [])) for row in transitions
        ),
        "physicalEventRows": sum(
            len(row["economicDiagnostics"].get("physicalEvents", [])) for row in transitions
        ),
    }
    if critic == "vector":
        geometry = audit["vectorCriticSharedEncoderGradientGeometry"]
        result["criticHeadSupport"] = {
            name: {
                "rowCount": value["rowCount"],
                "targetOutsideSupportFraction": value["targetOutsideSupportFraction"],
                "lowerEndpointProbabilityMean": value["lowerEndpointProbabilityMean"],
                "upperEndpointProbabilityMean": value["upperEndpointProbabilityMean"],
            }
            for name, value in sorted(head_diagnostics.items())
        }
        result["criticSharedEncoderGradientCosines"] = [
            {
                "first": pair["firstComponent"],
                "second": pair["secondComponent"],
                "cosine": pair["cosine"],
            }
            for pair in geometry["pairs"]
        ]
    return result


def _first_actor_audit(path: Path) -> dict[str, Any]:
    from .economic_diagnostics import load_artifact
    from .economic_trace import canonical_bytes

    _, transitions, updates = load_artifact(path)
    starts = [row for row in updates if row.get("event") == "mutation-start"]
    completes = [row for row in updates if row.get("event") == "mutation-complete"]
    if not starts or not completes:
        raise ValueError("optimizer replay trace has no completed actor minibatch")
    start, complete = starts[0], completes[0]
    audit = complete.get("optimizerDebug") or {}
    scope = audit.get("scope") or {}
    prefix = str(start["runId"]) + "/"

    def stable(identities):
        values = []
        for identity in identities:
            if not str(identity).startswith(prefix):
                raise ValueError("optimizer replay transition ID has a foreign run prefix")
            values.append(str(identity)[len(prefix) :])
        return values

    batch_stable = stable(start["batchTransitionIds"])
    result = {
        "transitionIds": [row["transitionId"] for row in transitions],
        "stableTransitionIds": stable(
            [row["transitionId"] for row in transitions]
        ),
        "batchTransitionIds": list(start["batchTransitionIds"]),
        "batchRowDigest": str(start["batchRowDigest"]),
        "stableBatchTransitionIds": batch_stable,
        "stableBatchRowDigest": hashlib.sha256(canonical_bytes(batch_stable)).hexdigest(),
        "auditTransitionIds": list(scope.get("auditTransitionIds", [])),
        "stableAuditTransitionIds": stable(scope.get("auditTransitionIds", [])),
        "actorHashBefore": str(start["beforeHashes"]["actor"]),
        "actorHashAfter": str(complete["afterHashes"]["actor"]),
    }
    if len(result["auditTransitionIds"]) != min(16, len(transitions)):
        raise ValueError("optimizer replay trace does not bind the expected 16 audit rows")
    return result


def _reference_replay_binding(
    reference: Path,
    current_trace: Path,
    comparison: dict[str, Any],
    retention: dict[str, Any],
    *,
    checkpoint_sha256: str,
    optimizer_seed: int,
    world_seed: int,
    turns: int,
) -> dict[str, Any]:
    from .optimizer_exact_replay import value_digest

    validation_path = reference / "validation.json"
    if not validation_path.is_file():
        raise ValueError("reference smoke lacks validation.json")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    expected_identity = (
        validation.get("checkpoint", {}).get("sha256"),
        validation.get("optimizerSeed"),
        validation.get("worldSeed"),
        validation.get("turns"),
    )
    actual_identity = (checkpoint_sha256, optimizer_seed, world_seed, turns)
    if expected_identity != actual_identity:
        raise ValueError("retention replay identity differs from the reference smoke")
    reference_rollout = validation.get("parity", {}).get("rollout", {})
    current_rollout = comparison["rollout"]
    rollout_fields = (
        "digest",
        "initialObservationDigest",
        "actionDigest",
        "rewardDigest",
        "rewardSum",
        "decisions",
    )
    mismatches = [
        field
        for field in rollout_fields
        if value_digest(reference_rollout.get(field))
        != value_digest(current_rollout.get(field))
    ]
    if mismatches:
        raise ValueError(f"retained rollout differs from reference smoke: {mismatches}")
    old_audit = _first_actor_audit(reference / "trace")
    new_audit = _first_actor_audit(current_trace)
    audit_fields = (
        "stableTransitionIds",
        "stableBatchTransitionIds",
        "stableBatchRowDigest",
        "stableAuditTransitionIds",
        "actorHashBefore",
        "actorHashAfter",
    )
    mismatches = [
        field
        for field in audit_fields
        if value_digest(old_audit[field]) != value_digest(new_audit[field])
    ]
    if mismatches:
        raise ValueError(f"retained first actor step differs from reference: {mismatches}")
    first_step = retention["firstActorStep"]
    if (
        first_step["traceActorHashBefore"] != new_audit["actorHashBefore"]
        or first_step["traceActorHashAfter"] != new_audit["actorHashAfter"]
    ):
        raise ValueError("retained actor weights differ from trace mutation hashes")
    adam_before = first_step["adamBefore"]
    if adam_before.get("state"):
        raise ValueError("prospective replay actor Adam was not fresh at first step")
    return {
        "status": "observed-smoke-match-before-deferred-independent-replay",
        "referenceRoot": str(reference.resolve()),
        "referenceValidationSha256": _sha256(validation_path),
        "rolloutFields": {field: current_rollout[field] for field in rollout_fields},
        "firstActorStep": new_audit,
        "pathDependentIdentityComparison": {
            "referenceBatchRowDigest": old_audit["batchRowDigest"],
            "currentBatchRowDigest": new_audit["batchRowDigest"],
            "stableSuffixBatchRowDigest": new_audit["stableBatchRowDigest"],
            "semantics": (
                "trace transition IDs include the output-root runId; replay comparison "
                "comparison removes only that prefix and rehashes the ordered IDs"
            ),
        },
        "retainedActorWeights": {
            "beforeDigest": value_digest(first_step["weightsBefore"]),
            "afterDigest": value_digest(first_step["weightsAfter"]),
        },
        "freshAdamProspective": {
            "historicalOptimizerRecovered": False,
            "beforeStateEntryCount": len(adam_before.get("state", {})),
            "beforeDigest": value_digest(adam_before),
            "afterDigest": value_digest(first_step["adamAfter"]),
        },
        "rngBinding": {
            "beforeDigest": value_digest(first_step["rngBefore"]),
            "afterDigest": value_digest(first_step["rngAfter"]),
        },
        "labelsPresent": False,
        "fullWorldEquivalenceProven": False,
        "optimizerDirectionReplayComplete": False,
        "labelJoinRule": (
            "later independent labels may join only these exact auditTransitionIds "
            "after a separately reviewed full-state replay method is available"
        ),
    }


def _bind_retained_trace_rows(
    retention: dict[str, Any], transitions: list[dict], audit: dict[str, Any]
) -> dict[str, Any]:
    reset_hashes = {
        str(row.get("provenance", {}).get("resetSnapshotHash"))
        for row in transitions
        if row.get("provenance", {}).get("resetSnapshotHash") is not None
    }
    trace_snapshot = str(retention["tracePostResetWorldSnapshotSha256"])
    if reset_hashes != {trace_snapshot}:
        raise ValueError("retained post-reset world capture differs from trace snapshot")
    by_decision = {}
    for row in transitions:
        key = (int(row["decisionTurn"]), int(row["campId"]))
        if key in by_decision:
            raise ValueError("trace repeats a retained turn/camp decision identity")
        by_decision[key] = str(row["transitionId"])
    retained_ids = []
    row_locations = {}
    for record_index, record in enumerate(retention["records"]):
        identities = []
        for row_index, camp in enumerate(record["campIdsInPolicyRowOrder"]):
            key = (int(record["preTurn"]), int(camp))
            if key not in by_decision:
                raise ValueError("retained policy row has no matching trace transition")
            identity = by_decision[key]
            identities.append(identity)
            row_locations[identity] = {
                "recordIndex": record_index,
                "policyRowIndex": row_index,
                "decisionTurn": key[0],
                "campId": key[1],
            }
        record["transitionIdsInPolicyRowOrder"] = identities
        retained_ids.extend(identities)
    audit_ids = list(audit["auditTransitionIds"])
    missing = [identity for identity in audit_ids if identity not in row_locations]
    if missing:
        raise ValueError("first-minibatch audit rows are absent from retained policy inputs")
    retention["firstActorBatchTransitionIds"] = list(audit["batchTransitionIds"])
    retention["auditTransitionIds"] = audit_ids
    retention["auditRowLocations"] = {
        identity: row_locations[identity] for identity in audit_ids
    }
    return {
        "retainedPolicyRowCount": len(retained_ids),
        "retainedTransitionIdDigest": _digest(retained_ids),
        "auditRowCount": len(audit_ids),
        "allAuditRowsBoundToRetainedInputs": True,
        "tracePostResetWorldSnapshotSha256": trace_snapshot,
        "postResetWorldCaptureMatchesTrace": True,
        "snapshotIsCompleteBridgeContext": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--optimizer-seed", type=int, default=4242)
    parser.add_argument("--world-seed", type=int, default=50000)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--lane-timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--retain-optimizer-audit",
        action="store_true",
        help="retain the bounded seed-4242 six-turn observed state/update context",
    )
    parser.add_argument(
        "--reference-smoke",
        type=Path,
        help="existing vector smoke whose recorded rollout/update digests must match",
    )
    parser.add_argument("--lane", choices=("off", "on"), help=argparse.SUPPRESS)
    parser.add_argument("--state", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--trace", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--retain-audit", action="store_true", help=argparse.SUPPRESS)
    return parser


def _run_child(command: list[str], timeout: int) -> None:
    options = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    process = subprocess.Popen(command, **options)
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def main() -> int:
    args = _parser().parse_args()
    from .protocol import seeds

    seeds([args.optimizer_seed, args.world_seed], "economic trace training smoke")
    if args.lane:
        return _lane(args)
    if args.out.exists() or not args.init.is_file() or not 4 <= args.turns <= 8:
        raise ValueError("output must be new, checkpoint must exist, and turns must be in [4, 8]")
    if args.retain_optimizer_audit:
        if (
            args.optimizer_seed != 4242
            or args.world_seed != 50000
            or args.turns != 6
        ):
            raise ValueError(
                "optimizer audit retention is fixed to seed 4242, world 50000, six turns"
            )
        if args.reference_smoke is None or not args.reference_smoke.is_dir():
            raise ValueError("optimizer audit retention requires an existing reference smoke")
    elif args.reference_smoke is not None:
        raise ValueError("--reference-smoke requires --retain-optimizer-audit")
    args.out.mkdir(parents=True)
    durations, states = {}, {}
    try:
        for lane in ("off", "on"):
            state = args.out / f".{lane}.pt"
            states[lane] = state
            command = [
                sys.executable,
                "-m",
                __package__ + ".economic_trace_training_smoke",
                "--init",
                str(args.init),
                "--out",
                str(args.out),
                "--optimizer-seed",
                str(args.optimizer_seed),
                "--world-seed",
                str(args.world_seed),
                "--turns",
                str(args.turns),
                "--lane",
                lane,
                "--state",
                str(state),
                "--trace",
                str(args.out / "trace"),
            ]
            if args.retain_optimizer_audit and lane == "on":
                command.append("--retain-audit")
            started = time.monotonic()
            _run_child(command, args.lane_timeout_seconds)
            durations[lane] = time.monotonic() - started
        off = torch.load(states["off"], map_location="cpu", weights_only=False)
        on = torch.load(states["on"], map_location="cpu", weights_only=False)
        comparison = _compare(off, on)
        trace = _trace_summary(
            args.out / "trace", int(on["rollout"]["decisions"]), on["configuration"]["critic"]
        )
        retention_manifest = None
        if args.retain_optimizer_audit:
            retained = on.get("auditRetention")
            if not isinstance(retained, dict):
                raise ValueError("traced lane omitted requested optimizer audit retention")
            binding = _reference_replay_binding(
                args.reference_smoke.resolve(),
                args.out / "trace",
                comparison,
                retained,
                checkpoint_sha256=_sha256(args.init),
                optimizer_seed=args.optimizer_seed,
                world_seed=args.world_seed,
                turns=args.turns,
            )
            from .economic_diagnostics import load_artifact

            _, retained_transitions, _ = load_artifact(args.out / "trace")
            binding["retainedTraceRows"] = _bind_retained_trace_rows(
                retained,
                retained_transitions,
                binding["firstActorStep"],
            )
            from .optimizer_exact_replay import write_retention_package

            retention_manifest = write_retention_package(
                args.out / "optimizer-audit-retention", retained, binding
            )
        result = {
            "schemaVersion": 1,
            "status": "pass",
            "scope": "local-wiring-validation-only",
            "claimsExcluded": [
                "arm-selection",
                "readiness",
                "treatment-comparison",
                "causal-credit",
            ],
            "checkpoint": {"path": str(args.init.resolve()), "sha256": _sha256(args.init)},
            "optimizerSeed": args.optimizer_seed,
            "worldSeed": args.world_seed,
            "turns": args.turns,
            "epochs": 1,
            "batch": 16,
            "configuration": on["configuration"],
            "durationsSeconds": {**durations, "total": sum(durations.values())},
            "parity": comparison,
            "trace": trace,
        }
        if retention_manifest is not None:
            result["optimizerAuditRetention"] = {
                "path": "optimizer-audit-retention/manifest.json",
                "digest": retention_manifest["digest"],
                "status": retention_manifest["status"],
            }
        (args.out / "validation.json").write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
    finally:
        for state in states.values():
            state.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
