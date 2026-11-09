"""Measure-only D7 for the A(s,a) control variate on one real rollout (no model is saved).

Runs exactly what ``labor_training train --a-mode gae`` runs at its first gae iteration
(iteration 0, eta = 0 everywhere, "first-gae-iteration-measure-only"), through the same
functions, minus every write:

1. the rollout/checkpoint binding checks of ``_command_train`` (source checkpoint SHA,
   behaviour policy, rollout critic and normalizer digests) and ``_load_rollout``;
2. ``labor_training._a_setup``: the A argument checks, ``labor_a_ppo.load_frozen_a`` (frozen
   ``--a-critic`` context provider + ``--a-warm-start`` head, digest-checked) and the lagged
   eta (``eta_for_iteration``);
3. ``labor_training._a_prepare``: ``labor_a_ppo.prepare_iteration`` and ``measure_d7`` on
   ``--d7-rows`` rows (default 64) with the behaviour (checkpoint) actor, plus the next eta;
4. ``labor_training._a_dry_run_kl``: the K-epoch (``--epochs``, default 4) per-head
   ``update_labor_policy`` on deep copies of the actor and its Adam state (never applied,
   no rollback), as ``train --a-telemetry-dry-run`` does;
5. ``labor_training._a_finish`` on a scratch dict: the frozen-A guard, the
   ``labor_a_ppo.warmup_report`` per-component calibration/TD telemetry and the warm-up
   switch decision, with the online V = the checkpoint's critic UNCHANGED.

Deliberately skipped (and why):

* the critic update. ``train`` requires an on-policy native-fork ``--anchor`` (produced by
  ``labor_anchor_producer``), which feeds ONLY the critic's independent-return loss. A, D7
  and the dry-run KL never read the online critic (A reads the frozen ``--a-critic``), so
  they equal what ``train`` measures; only the warm-up report differs, because ``train``
  evaluates it with the post-update online V while this script uses the checkpoint's
  (= at iteration 0 the frozen stage-1 critic itself).
* the actor update and the checkpoint write. Nothing is saved except the JSON report.

The dry run needs a ``LaborActorGate``. With ``--calibration-report`` the gate is built
from that report exactly as ``train`` does; without one a measurement-only binding to the
current policy/critic/normalizer digests is constructed (recorded as
``"gate": "measurement-only-binding"``). The gate only binds digests: it does not change
the update or its KL.

Usage::

    python -m anthropocene_rl.autonomous_v7.labor_a_measure \
        --checkpoint checkpoint-start.pt --rollout rollout.pt \
        --a-warm-start a_warm_start.pt --a-critic critic.pt --d7-rows 256 --out d7.json

Apply mode (``--apply``, :func:`apply_update`) is an ACTOR-ONLY step check. It runs the real
``update_labor_policy`` on the real actor exactly as ``train --a-mode off`` runs its actor
update (joint surrogate with ``_score_nonlabor``, the checkpoint's shuffle stream, train's
epochs/minibatch/clip/entropy/grad-clip/audit defaults, ``--target-kl`` default 0.02 with
rollback active), with the actor optimizer built by ``labor_actor_optim`` from the
checkpoint's saved Adam state and the configured per-group rates re-applied (moments and
the warm-up step count carry over). It never trains the critic (task/costs) and never
touches the critic optimizer: ``train``'s critic update needs a native-fork ``--anchor``
that feeds only the critic loss, and this check deliberately leaves V alone. It also skips
``train``'s calibration-report requirement: without ``--calibration-report`` the gate is the
measurement-only digest binding (the gate does not change the update).

The output checkpoint is the input state with ONLY ``actor`` and ``actor_optimizer``
replaced, plus the training-progress fields ``train`` itself advances after an update
(``iteration`` + 1; ``sampling_generator_state`` = the rollout's
``samplingGeneratorStateAfter``; ``shuffle_rng_state`` = the stream after the update), so the
next ``collect`` continues the sampling stream and the next apply the shuffle stream exactly
as after ``train``. ``critic_training_world_seeds``/``critic_training_independent_unit_ids``
stay unchanged (the critic saw no new data). ``last_update`` is replaced by this update's
record (``criticUpdated: false``) and ``actor_step_checks`` (a list, one entry appended per
apply) carries the provenance. ``collect`` recomputes the behaviour digest from the loaded
actor, so a rollout collected from the output is bound to the updated actor, and the next
apply checks that binding (source checkpoint SHA and behaviour digest) before updating::

    python -m anthropocene_rl.autonomous_v7.labor_a_measure --apply \
        --checkpoint checkpoint-start.pt --rollout rollout.pt \
        --actor-encoder-lr 1e-7 --actor-head-lr 3e-5 --actor-lr-warmup-steps 12 \
        --target-kl 0.02 --out-checkpoint checkpoint-1.pt --out apply-report.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

MEASURE_SCHEMA = "labor-a-d7-measurement-v1"
APPLY_SCHEMA = "labor-actor-apply-v1"
STEP_CHECKS_KEY = "actor_step_checks"


def _jsonable(value: Any) -> Any:
    from .labor_a_ppo import json_number

    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return json_number(value)


def measure(args: argparse.Namespace) -> dict[str, Any]:
    """Run the measurement with ``args.threads`` torch threads (restored afterwards).

    Float sums in D7 and the dry run depend on the torch thread count in the last digits
    (observed: ~1e-10 relative on variances, ~3e-5 on a dry-run KL), so equal thread
    counts are needed for a bitwise comparison with ``train``.
    """
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(args.threads)))
    try:
        return _measure(args)
    finally:
        torch.set_num_threads(previous_threads)


def _measure(args: argparse.Namespace) -> dict[str, Any]:
    from . import labor_a_ppo as ap
    from . import labor_training as training
    from .labor_actor_optim import (
        actor_lr_config,
        build_actor_optimizer,
        load_actor_optimizer_state,
    )
    from .labor_model import labor_checkpoint_personality
    from .labor_ppo import LaborActorGate
    from .labor_value_calibration import labor_critic_digest
    from .provenance import checkpoint_hash
    from .value_calibration import behavior_policy_digest

    seconds: dict[str, float] = {}
    started = time.perf_counter()

    # --- _command_train's loading and binding checks (identical order and messages) ----
    clock = time.perf_counter()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    training._require_trainer_checkpoint(state)
    layout, actor, task, costs, normalizer = training._models_from_checkpoint(state)
    personality = labor_checkpoint_personality(state)
    batch, _task_targets, _cost_targets, rollout_metadata = training._load_rollout(args.rollout)
    reset_record = training._log_rollout_reset(rollout_metadata, "labor_a_measure")
    behavior = training.labor_behavior_configuration(personality)
    checkpoint_sha = checkpoint_hash(args.checkpoint)
    if rollout_metadata.get("sourceCheckpointSha256") != checkpoint_sha:
        raise ValueError("labor rollout belongs to another checkpoint")
    if batch.behavior_policy_digest != behavior_policy_digest(actor, behavior):
        raise ValueError("labor rollout behavior policy is stale")
    if batch.advantage_critic_digest != labor_critic_digest(task, costs):
        raise ValueError("labor rollout advantages came from other critic weights")
    if batch.normalization_binding_digest != training._normalization_digest(normalizer):
        raise ValueError("labor rollout normalization binding is stale")
    if ap.A_STATE_KEY in state:
        raise ValueError(
            "the checkpoint already carries a D7 state: this script measures the first gae "
            "iteration only"
        )
    seconds["load"] = time.perf_counter() - clock

    a_args = argparse.Namespace(
        a_mode="gae",
        a_warm_start=args.a_warm_start,
        a_critic=args.a_critic,
        a_draws=args.a_draws,
        a_d7_rows=args.d7_rows,
        a_telemetry_dry_run=not args.no_dry_run,
        actor_lr=args.actor_lr,
        actor_encoder_lr=args.actor_encoder_lr,
        actor_head_lr=args.actor_head_lr,
        actor_lr_warmup_steps=args.actor_lr_warmup_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        clip=args.clip,
        entropy_weight=args.entropy_weight,
        max_grad_norm=args.max_grad_norm,
        audit_rows=args.audit_rows,
    )
    clock = time.perf_counter()
    a_run = training._a_setup(a_args, state, batch, rollout_metadata)
    seconds["setup"] = time.perf_counter() - clock
    if a_run is None:  # pragma: no cover - a_mode is gae
        raise AssertionError("gae setup returned no A run")

    # train builds the actor optimizer from the checkpoint before anything else
    # (saved moments; the configured per-group rates replace the saved ones).
    lr_config = actor_lr_config(a_args)
    actor_optimizer = build_actor_optimizer(actor, lr_config)
    load_actor_optimizer_state(actor_optimizer, actor, state["actor_optimizer"], lr_config)
    actor_before = {k: v.detach().clone() for k, v in actor.state_dict().items()}

    clock = time.perf_counter()
    prepared = training._a_prepare(a_run, actor, batch, state)
    seconds["prepareAndD7"] = time.perf_counter() - clock

    gate_record: dict[str, Any]
    if args.no_dry_run:
        gate_record = {"gate": "not used (--no-dry-run)"}
    else:
        if args.calibration_report is not None:
            report = training._load_json(args.calibration_report)
            training._gate_data_identity(report)
            if not report["actorGateOpen"]:
                raise ValueError("--calibration-report is closed; the dry run needs an open gate")
            gate = LaborActorGate.from_report(report)
            gate_record = {
                "gate": "calibration-report",
                "calibrationReportSha256": checkpoint_hash(args.calibration_report),
            }
        else:
            gate = LaborActorGate(
                behavior_policy_digest(actor, behavior),
                batch.advantage_critic_digest,
                batch.normalization_binding_digest,
                checkpoint_hash(args.rollout),
            )
            gate_record = {
                "gate": "measurement-only-binding",
                "note": "binds the current policy/critic/normalizer digests; no calibration "
                "claim; the gate does not change the update or its KL",
            }
        gate.require_current(
            actor,
            behavior,
            training.LABOR_REGISTRY_DIGEST,
            batch.advantage_critic_digest,
            batch.normalization_binding_digest,
        )
        clock = time.perf_counter()
        a_run["dryRunKL"] = training._a_dry_run_kl(
            prepared, actor, actor_optimizer, batch, state, a_args, gate, behavior
        )
        seconds["dryRunKL"] = time.perf_counter() - clock

    # _a_finish on a scratch dict: online V = the checkpoint's critic, unchanged.
    scratch: dict[str, Any] = {"last_update": {}}
    clock = time.perf_counter()
    training._a_finish(a_run, prepared, scratch, state, task, costs, batch, rollout_metadata)
    seconds["finish"] = time.perf_counter() - clock
    if any(not torch.equal(v, actor.state_dict()[k]) for k, v in actor_before.items()):
        raise AssertionError("the measurement changed the actor")
    record = scratch["last_update"]["aControlVariate"]

    decisions = rollout_metadata.get("decisionRows", [])
    retries = len(batch.retry_parent)
    per_world = Counter(int(row["world"]) for row in decisions)
    d7 = {
        head: {
            key: values.get(key)
            for key in (
                "varianceVOnly",
                "varianceControlVariate",
                "reduction",
                "admit",
                "aBlind",
                "rows",
                "threshold",
                "reason",
            )
            if key in values
        }
        for head, values in record["d7"].items()
    }
    seconds["total"] = time.perf_counter() - started
    seconds.update({f"a.{k}": v for k, v in a_run["seconds"].items()})
    return _jsonable(
        {
            "schema": MEASURE_SCHEMA,
            "scope": "measurement only: train --a-mode gae iteration 0 minus the critic update "
            "(no --anchor) and the actor update; no model saved; online V = checkpoint critic",
            "inputs": {
                "checkpoint": str(args.checkpoint),
                "checkpointSha256": checkpoint_sha,
                "rollout": str(args.rollout),
                "rolloutSha256": checkpoint_hash(args.rollout),
                "aWarmStartSha256": a_run["warmStartSha256"],
                "aCriticSha256": a_run["criticCheckpointSha256"],
                "iteration": int(state["iteration"]),
                "optimizerSeed": int(state["optimizer_seed"]),
                "behaviorPolicyDigest": batch.behavior_policy_digest,
                "advantageCriticDigest": batch.advantage_critic_digest,
                "normalizationBindingDigest": batch.normalization_binding_digest,
                "encoderLayout": layout.state_dict(),
            },
            "rollout": {
                "worldSeeds": rollout_metadata.get("worldSeeds"),
                "turns": rollout_metadata.get("turns"),
                "parents": int(batch.parent_count),
                "retries": retries,
                "parentsPerWorld": {str(w): n for w, n in sorted(per_world.items())},
                "terminated": int(sum(bool(r.get("terminated")) for r in decisions)),
                "costMultipliers": dict(batch.cost_multipliers or {}),
                "resetOptions": reset_record["rolloutResetOptions"],
                "resetOptionsSource": reset_record["rolloutResetOptionsSource"],
                "scope": rollout_metadata.get("scope"),
            },
            "config": {
                "draws": int(a_run["draws"]),
                "d7Rows": int(a_run["d7Rows"]),
                "d7Threshold": float(ap.D7_MIN_VARIANCE_REDUCTION),
                "dryRun": {
                    "epochs": int(args.epochs),
                    "batchSize": int(args.batch_size),
                    "clip": float(args.clip),
                    "entropyWeight": float(args.entropy_weight),
                    "maxGradNorm": float(args.max_grad_norm),
                    "actorLr": float(args.actor_lr),
                    "actorOptimizer": lr_config.record(),
                    "auditRows": int(args.audit_rows),
                    "targetKL": "1e30 (no rollback; as train's dry run)",
                    **gate_record,
                },
                "threads": int(torch.get_num_threads()),
            },
            "eta": {
                "used": record["etaUsed"],
                "source": record["etaSource"],
                "next": record["nextEta"],
            },
            "d7": d7,
            "frozen": record["frozen"],
            "counts": record["counts"],
            "warmup": {
                **record["warmup"],
                "onlineV": "checkpoint critic, unchanged (train would use the post-update V)",
            },
            "dryRunKL": {
                "perEpoch": a_run["dryRunKL"],
                "warmupKLCap": ap.s2.WARMUP_KL_CAP,
                "dryRunKLBound": ap.s2.DRY_RUN_KL_BOUND,
            },
            "storedStateTrainWouldWrite": scratch[ap.A_STATE_KEY],
            "seconds": seconds,
            "host": {"python": platform.python_version(), "torch": torch.__version__},
        }
    )


# ---------------------------------------------------------------------------
# Apply mode: the real actor-only update of train --a-mode off, written to a checkpoint
# ---------------------------------------------------------------------------

# Checkpoint fields an apply writes; every other field is the input's, bitwise.
APPLY_REPLACED_FIELDS = ("actor", "actor_optimizer")
APPLY_PROGRESS_FIELDS = ("iteration", "sampling_generator_state", "shuffle_rng_state")
APPLY_RECORD_FIELDS = ("last_update", STEP_CHECKS_KEY)
APPLY_DEFAULT_TARGET_KL = 0.02  # train's --target-kl default
LOGP_CHUNK = 256


def _identical(left: Any, right: Any) -> bool:
    """Bitwise equality of nested checkpoint values (tensors by dtype/shape/values)."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_identical(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_identical(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def policy_log_probs(actor: torch.nn.Module, batch: Any) -> dict[str, torch.Tensor]:
    """Current log-probs of the rollout's recorded actions, float64: per initial policy
    head and the initial joint over all parents, and the retry joint over all retry rows.

    Uses the scorers the PPO update uses (``score_labor_sample_heads``,
    ``labor_training._score_nonlabor_heads``, ``score_labor_sample``) on fixed chunks of
    ``LOGP_CHUNK`` rows, under ``no_grad``; no ratio is formed, so it never overflows.
    """
    from . import labor_training as training
    from .labor_ppo import (
        LABOR_POLICY_HEADS,
        NONLABOR_POLICY_HEADS,
        _rows,
        score_labor_sample,
        score_labor_sample_heads,
    )

    parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in (*LABOR_POLICY_HEADS, *NONLABOR_POLICY_HEADS, "initialJoint")
    }
    retry: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, batch.parent_count, LOGP_CHUNK):
            index = torch.arange(
                start, min(batch.parent_count, start + LOGP_CHUNK), dtype=torch.long
            )
            tensors = _rows(batch.initial_tensors, index)
            labor, _ = score_labor_sample_heads(
                actor, tensors, _rows(batch.initial_labor_sample, index)
            )
            nonlabor, _ = training._score_nonlabor_heads(
                actor, tensors, _rows(batch.initial_nonlabor_sample, index)
            )
            for name in LABOR_POLICY_HEADS:
                parts[name].append(labor[name])
            for name in NONLABOR_POLICY_HEADS:
                parts[name].append(nonlabor[name])
            parts["initialJoint"].append(labor["joint"] + nonlabor["joint"])
        retries = len(batch.retry_parent)
        for start in range(0, retries, LOGP_CHUNK):
            index = torch.arange(start, min(retries, start + LOGP_CHUNK), dtype=torch.long)
            logp, _ = score_labor_sample(
                actor, _rows(batch.retry_tensors, index), _rows(batch.retry_labor_sample, index)
            )
            retry.append(logp)
    result = {name: torch.cat(values).double() for name, values in parts.items()}
    result["retryJoint"] = (
        torch.cat(retry).double() if retry else torch.zeros(0, dtype=torch.float64)
    )
    return result


def behaviour_log_probs(batch: Any) -> dict[str, torch.Tensor]:
    """The rollout's stored behaviour log-probs in :func:`policy_log_probs`'s layout."""
    labor, nonlabor = batch.require_behaviour_head_logp()
    result = {name: value.double() for name, value in {**labor, **nonlabor}.items()}
    result["initialJoint"] = (
        batch.behaviour_initial_labor_logp.double()
        + batch.behaviour_initial_nonlabor_logp.double()
    )
    result["retryJoint"] = batch.behaviour_retry_labor_logp.double()
    return result


def log_prob_deltas(
    new: Mapping[str, torch.Tensor], old: Mapping[str, torch.Tensor]
) -> dict[str, dict[str, Any]]:
    """Per key: rows, mean |new - old|, max |new - old| and mean (new - old)."""
    out: dict[str, dict[str, Any]] = {}
    for name, value in new.items():
        difference = value - old[name]
        rows = int(difference.numel())
        out[name] = {
            "rows": rows,
            "meanAbs": float(difference.abs().mean()) if rows else None,
            "maxAbs": float(difference.abs().max()) if rows else None,
            "meanSigned": float(difference.mean()) if rows else None,
        }
    return out


def _per_epoch_kl(stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per epoch: minibatches planned/processed/accepted, the audit KL after the epoch's
    last ACCEPTED step (the state carried on) and the rejected step's KL, if any.

    ``sampledNonnegativeKL`` has one entry per attempted step (accepted ones, then the
    rolled-back one if the rollback came after a step); an overflow before a step adds a
    processed minibatch without a KL entry. A rollback ends the update, so only the last
    processed epoch can hold a rejection."""
    kl = list(stats["sampledNonnegativeKL"])
    attempted, accepted_total = int(stats["attemptedSteps"]), int(stats["acceptedSteps"])
    rejected_after_step = attempted - accepted_total
    if rejected_after_step not in (0, 1) or len(kl) != attempted:
        raise AssertionError("unexpected PPO step accounting")
    coverage = list(stats["epochCoverage"])
    out, cursor = [], 0
    for number, epoch in enumerate(coverage):
        accepted = int(epoch["accepted"]["minibatches"])
        entries = kl[cursor : cursor + accepted]
        cursor += accepted
        last = number == len(coverage) - 1
        rejected = kl[cursor] if last and rejected_after_step else None
        out.append(
            {
                "epoch": number,
                "plannedMinibatches": int(epoch["planned"]["minibatches"]),
                "processedMinibatches": int(epoch["processed"]["minibatches"]),
                "acceptedMinibatches": accepted,
                "complete": accepted == int(epoch["planned"]["minibatches"]),
                "auditKLAfterLastAcceptedStep": entries[-1] if entries else None,
                "rejectedStepAuditKL": rejected,
            }
        )
    if cursor + rejected_after_step != len(kl):
        raise AssertionError("per-epoch KL reconstruction does not cover every step")
    return out


def _kl_text(value: float) -> Any:
    return value if math.isfinite(value) else str(value)


def apply_update(args: argparse.Namespace) -> dict[str, Any]:
    """Apply mode with ``args.threads`` torch threads (restored afterwards); returns the
    report. Writes ``args.out_checkpoint`` (never overwrites an existing file)."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(args.threads)))
    try:
        return _apply(args)
    finally:
        torch.set_num_threads(previous_threads)


def _apply(args: argparse.Namespace) -> dict[str, Any]:
    from . import labor_a_ppo as ap
    from . import labor_training as training
    from .labor_actor_optim import (
        ActorLrSchedule,
        actor_group_counts,
        actor_group_lrs,
        actor_lr_config,
        adam_steps_taken,
        build_actor_optimizer,
        load_actor_optimizer_state,
    )
    from .labor_model import labor_checkpoint_personality
    from .labor_ppo import LaborActorGate, update_labor_policy
    from .labor_value_calibration import labor_critic_digest
    from .provenance import checkpoint_hash, code_commit
    from .value_calibration import behavior_policy_digest, policy_digest

    seconds: dict[str, float] = {}
    started = time.perf_counter()
    out_checkpoint = Path(args.out_checkpoint)
    if out_checkpoint.exists():
        raise ValueError(f"--out-checkpoint {out_checkpoint} exists; refusing to overwrite it")
    if out_checkpoint.resolve() in (
        Path(args.checkpoint).resolve(),
        Path(args.rollout).resolve(),
        Path(args.out).resolve(),
    ):
        raise ValueError("--out-checkpoint must differ from the inputs and the report")
    target_kl = float(args.target_kl)
    if not math.isfinite(target_kl) or target_kl <= 0:
        raise ValueError("--target-kl must be finite and positive")

    # --- _command_train's loading and binding checks (same order and messages) ---------
    clock = time.perf_counter()
    checkpoint_sha = checkpoint_hash(args.checkpoint)
    rollout_sha = checkpoint_hash(args.rollout)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    training._require_trainer_checkpoint(state)
    if ap.A_STATE_KEY in state:
        raise ValueError(
            "the checkpoint carries a D7/A state: apply mode is train --a-mode off only"
        )
    _layout, actor, task, costs, normalizer = training._models_from_checkpoint(state)
    personality = labor_checkpoint_personality(state)
    batch, _task_targets, _cost_targets, rollout_metadata = training._load_rollout(args.rollout)
    reset_record = training._log_rollout_reset(rollout_metadata, "labor_a_measure --apply")
    behavior = training.labor_behavior_configuration(personality)
    input_policy = behavior_policy_digest(actor, behavior)
    if rollout_metadata.get("sourceCheckpointSha256") != checkpoint_sha:
        raise ValueError("labor rollout belongs to another checkpoint")
    if batch.behavior_policy_digest != input_policy:
        raise ValueError("labor rollout behavior policy is stale")
    if rollout_metadata.get("behaviorPolicyDigest") != input_policy:
        raise ValueError("labor rollout metadata behavior policy is stale")
    critic_digest = labor_critic_digest(task, costs)
    if batch.advantage_critic_digest != critic_digest:
        raise ValueError("labor rollout advantages came from other critic weights")
    if batch.normalization_binding_digest != training._normalization_digest(normalizer):
        raise ValueError("labor rollout normalization binding is stale")
    if rollout_metadata.get("criticWarmup") is not None:
        raise ValueError("a critic-warmup rollout never feeds an actor update (train refuses)")
    loaded_actor = actor.state_dict()
    if set(loaded_actor) != set(state["actor"]) or any(
        not _identical(loaded_actor[k], state["actor"][k]) for k in loaded_actor
    ):
        raise AssertionError("the loaded actor differs from the checkpoint's actor tensors")
    seconds["load"] = time.perf_counter() - clock

    # --- train's actor optimizer: saved Adam moments/steps, configured per-group rates ---
    lr_config = actor_lr_config(args)
    actor_optimizer = build_actor_optimizer(actor, lr_config)
    optimizer_load = load_actor_optimizer_state(
        actor_optimizer, actor, state["actor_optimizer"], lr_config
    )
    steps_before = adam_steps_taken(actor_optimizer)
    lrs_before = actor_group_lrs(actor_optimizer)

    gate_record: dict[str, Any]
    if args.calibration_report is not None:
        report = training._load_json(args.calibration_report)
        training._gate_data_identity(report)
        if not report["actorGateOpen"]:
            raise ValueError("--calibration-report is closed; train would not update the actor")
        gate = LaborActorGate.from_report(report)
        gate_record = {
            "gate": "calibration-report",
            "calibrationReportSha256": checkpoint_hash(args.calibration_report),
        }
    else:
        gate = LaborActorGate(
            input_policy,
            batch.advantage_critic_digest,
            batch.normalization_binding_digest,
            rollout_sha,
        )
        gate_record = {
            "gate": "measurement-only-binding",
            "note": "binds the current policy/critic/normalizer digests; no calibration "
            "claim (train requires an open calibration report); the gate does not change "
            "the update or its KL",
        }
    gate.require_current(
        actor,
        behavior,
        training.LABOR_REGISTRY_DIGEST,
        batch.advantage_critic_digest,
        batch.normalization_binding_digest,
    )

    clock = time.perf_counter()
    behaviour = behaviour_log_probs(batch)
    logp_before = policy_log_probs(actor, batch)
    seconds["logProbsBefore"] = time.perf_counter() - clock
    actor_before = {k: v.detach().clone() for k, v in actor.state_dict().items()}

    # --- the update: train --a-mode off's update_labor_policy call -----------------------
    shuffle = np.random.default_rng()
    shuffle.bit_generator.state = copy.deepcopy(state["shuffle_rng_state"])
    clock = time.perf_counter()
    stats = update_labor_policy(
        policy_module=actor,
        labor_actor=actor,
        optimizer=actor_optimizer,
        batch=batch,
        nonlabor_scorer=lambda tensors, sample: training._score_nonlabor(actor, tensors, sample),
        rng=shuffle,
        epochs=args.epochs,
        batch_size=args.batch_size,
        clip=args.clip,
        entropy_weight=args.entropy_weight,
        max_grad_norm=args.max_grad_norm,
        target_kl=target_kl,
        audit_rows=args.audit_rows,
        actor_gate=gate,
        behavior_configuration=behavior,
        lr_schedule=ActorLrSchedule(lr_config),
    )
    seconds["update"] = time.perf_counter() - clock
    steps_after = adam_steps_taken(actor_optimizer)

    clock = time.perf_counter()
    logp_after = policy_log_probs(actor, batch)
    seconds["logProbsAfter"] = time.perf_counter() - clock
    actor_after = actor.state_dict()
    actor_changed = any(not torch.equal(actor_before[k], actor_after[k]) for k in actor_before)
    if labor_critic_digest(task, costs) != critic_digest:
        raise AssertionError("the actor update changed the critic")
    output_policy = behavior_policy_digest(actor, behavior)

    # --- the output checkpoint: the input with only the actor/its optimizer replaced -----
    sampling = torch.Generator()
    sampling.set_state(rollout_metadata["samplingGeneratorStateAfter"])
    flags = {
        "actorLr": float(args.actor_lr),
        "actorEncoderLr": None if args.actor_encoder_lr is None else float(args.actor_encoder_lr),
        "actorHeadLr": None if args.actor_head_lr is None else float(args.actor_head_lr),
        "actorLrWarmupSteps": int(args.actor_lr_warmup_steps or 0),
        "targetKL": target_kl,
        "epochs": int(args.epochs),
        "batchSize": int(args.batch_size),
        "clip": float(args.clip),
        "entropyWeight": float(args.entropy_weight),
        "maxGradNorm": float(args.max_grad_norm),
        "auditRows": int(args.audit_rows),
        "aMode": "off",
        "surrogate": "joint (train's default; no --per-head-surrogate)",
        "threads": int(torch.get_num_threads()),
        **gate_record,
    }
    commit = code_commit()
    provenance = {
        "schema": APPLY_SCHEMA,
        "tool": "labor_a_measure --apply",
        "codeCommit": commit,
        "inputCheckpoint": str(args.checkpoint),
        "inputCheckpointSha256": checkpoint_sha,
        "rollout": str(args.rollout),
        "rolloutSha256": rollout_sha,
        "rolloutWorldSeeds": list(rollout_metadata.get("worldSeeds", [])),
        "inputIteration": int(state["iteration"]),
        "outputIteration": int(state["iteration"]) + 1,
        "inputBehaviorPolicyDigest": input_policy,
        "outputBehaviorPolicyDigest": output_policy,
        "flags": flags,
        "criticUpdated": False,
        "criticOptimizerUpdated": False,
        "criticNote": "deliberately not trained: train's critic update needs a native-fork "
        "--anchor feeding only the critic loss; task/cost critics, critic_optimizer and the "
        "critic-training world/unit history are the input's, bitwise",
        "replacedFields": list(APPLY_REPLACED_FIELDS),
        "advancedFields": {
            "iteration": "input + 1 (as train)",
            "sampling_generator_state": "the rollout's samplingGeneratorStateAfter (as train)",
            "shuffle_rng_state": "the shuffle stream after this update (as train)",
        },
    }
    next_state = dict(state)
    next_state["actor"] = actor_after
    next_state["actor_optimizer"] = actor_optimizer.state_dict()
    next_state["iteration"] = int(state["iteration"]) + 1
    next_state["sampling_generator_state"] = sampling.get_state()
    next_state["shuffle_rng_state"] = copy.deepcopy(shuffle.bit_generator.state)
    next_state["last_update"] = {
        "rolloutSha256": rollout_sha,
        "anchorSha256": None,
        "criticEpochs": 0,
        "criticUpdated": False,
        "actor": stats,
        "actorOptimizer": {
            **lr_config.record(),
            "groupCounts": actor_group_counts(actor),
            "load": optimizer_load,
        },
        "scope": "actor-only step check (labor_a_measure --apply): train --a-mode off's actor "
        "update; critic not updated; not readiness",
    }
    next_state[STEP_CHECKS_KEY] = [*state.get(STEP_CHECKS_KEY, []), provenance]
    clock = time.perf_counter()
    training._save_checkpoint(out_checkpoint, next_state)
    seconds["save"] = time.perf_counter() - clock

    # --- verify the written file: collect's and apply's loaders; nothing else changed ----
    clock = time.perf_counter()
    written = torch.load(out_checkpoint, map_location="cpu", weights_only=True)
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    training._require_trainer_checkpoint(written)
    _, reloaded_actor, reloaded_task, reloaded_costs, reloaded_normalizer = (
        training._models_from_checkpoint(written)
    )
    reloaded_behavior = training.labor_behavior_configuration(
        labor_checkpoint_personality(written)
    )
    if behavior_policy_digest(reloaded_actor, reloaded_behavior) != output_policy:
        raise AssertionError("the written actor's behaviour digest differs from the updated actor")
    if labor_critic_digest(reloaded_task, reloaded_costs) != critic_digest:
        raise AssertionError("the written critic differs from the input critic")
    if training._normalization_digest(reloaded_normalizer) != batch.normalization_binding_digest:
        raise AssertionError("the written normalizer differs from the input normalizer")
    reloaded_optimizer = build_actor_optimizer(reloaded_actor, lr_config)
    load_actor_optimizer_state(
        reloaded_optimizer, reloaded_actor, written["actor_optimizer"], lr_config
    )
    if adam_steps_taken(reloaded_optimizer) != steps_after:
        raise AssertionError("the written actor optimizer lost its Adam step count")
    rewritten = set(APPLY_REPLACED_FIELDS) | set(APPLY_PROGRESS_FIELDS) | set(APPLY_RECORD_FIELDS)
    if set(written) - rewritten != set(original) - rewritten:
        raise AssertionError("the written checkpoint gained or lost a field")
    unchanged = sorted(set(original) - rewritten)
    differing = [key for key in unchanged if not _identical(original[key], written[key])]
    if differing:
        raise AssertionError(f"apply changed checkpoint fields it must copy: {differing}")
    seconds["verify"] = time.perf_counter() - clock
    output_sha = checkpoint_hash(out_checkpoint)

    seconds["total"] = time.perf_counter() - started
    optimizer_stats = dict(stats.get("actorOptimizer", {}))
    kl = [float(value) for value in stats["sampledNonnegativeKL"]]
    planned = sum(int(epoch["planned"]["minibatches"]) for epoch in stats["epochCoverage"])
    planned_total = int(args.epochs) * math.ceil(batch.parent_count / int(args.batch_size))
    rolled_back = bool(stats["actorStepRolledBack"])
    return _jsonable(
        {
            "schema": APPLY_SCHEMA,
            "scope": "actor-only step check: train --a-mode off's actor update applied for "
            "real (rollback active); the critic and its optimizer were NOT updated",
            "criticUpdated": False,
            "inputs": {
                "checkpoint": str(args.checkpoint),
                "checkpointSha256": checkpoint_sha,
                "rollout": str(args.rollout),
                "rolloutSha256": rollout_sha,
                "iteration": int(state["iteration"]),
                "optimizerSeed": int(state["optimizer_seed"]),
                "rolloutBehaviorPolicyDigest": batch.behavior_policy_digest,
                "checkpointBehaviorPolicyDigest": input_policy,
                "onPolicy": batch.behavior_policy_digest == input_policy,
                "advantageCriticDigest": batch.advantage_critic_digest,
                "normalizationBindingDigest": batch.normalization_binding_digest,
                "inputStepChecks": len(state.get(STEP_CHECKS_KEY, [])),
            },
            "rollout": {
                "worldSeeds": rollout_metadata.get("worldSeeds"),
                "turns": rollout_metadata.get("turns"),
                "parents": int(batch.parent_count),
                "retries": len(batch.retry_parent),
                "resetOptions": reset_record["rolloutResetOptions"],
                "resetOptionsSource": reset_record["rolloutResetOptionsSource"],
            },
            "config": {
                **flags,
                "actorOptimizer": lr_config.record(),
                "codeCommit": commit,
            },
            "update": {
                "plannedSteps": planned_total,
                "plannedStepsInProcessedEpochs": planned,
                "attemptedSteps": int(stats["attemptedSteps"]),
                "acceptedSteps": int(stats["acceptedSteps"]),
                "rolledBackSteps": int(stats["attemptedSteps"]) - int(stats["acceptedSteps"]),
                "rollback": rolled_back,
                "stoppedEarly": rolled_back or int(stats["attemptedSteps"]) < planned_total,
                "perStepAuditKL": [_kl_text(value) for value in kl],
                "perEpoch": [
                    {
                        **epoch,
                        **{
                            key: None if epoch[key] is None else _kl_text(float(epoch[key]))
                            for key in ("auditKLAfterLastAcceptedStep", "rejectedStepAuditKL")
                        },
                    }
                    for epoch in _per_epoch_kl(stats)
                ],
                "ratioOverflow": stats.get("ratioOverflow"),
                "lrFactorPerAttemptedStep": optimizer_stats.get("lrFactorPerAttemptedStep"),
                "nextLrFactor": optimizer_stats.get("nextLrFactor"),
                "lrsBefore": lrs_before,
                "lrsAfterUpdate": optimizer_stats.get("lrsAfterUpdate"),
                "losses": stats["losses"],
                "fixedAuditRows": len(stats["fixedAuditRows"]),
                "actorChanged": actor_changed,
            },
            "adam": {
                "savedLayout": optimizer_load["savedLayout"],
                "savedLrs": optimizer_load["savedLrs"],
                "stepsBefore": steps_before,
                "stepsAfter": steps_after,
                "groupCounts": actor_group_counts(actor),
            },
            "logProbs": {
                "units": "nats on the rollout's recorded actions; heads and initialJoint over "
                "all parents, retryJoint over all retry rows",
                "afterVsBefore": log_prob_deltas(logp_after, logp_before),
                "beforeVsBehaviour": log_prob_deltas(logp_before, behaviour),
                "afterVsBehaviour": log_prob_deltas(logp_after, behaviour),
            },
            "output": {
                "checkpoint": str(out_checkpoint),
                "checkpointSha256": output_sha,
                "iteration": int(state["iteration"]) + 1,
                "behaviorPolicyDigest": output_policy,
                "actorTensorDigest": policy_digest(actor),
                "replacedFields": list(APPLY_REPLACED_FIELDS),
                "advancedFields": list(APPLY_PROGRESS_FIELDS),
                "recordFields": list(APPLY_RECORD_FIELDS),
                "unchangedFieldsVerifiedBitwise": unchanged,
                "loadersVerified": [
                    "labor_training._require_trainer_checkpoint + _models_from_checkpoint "
                    "(collect)",
                    "build_actor_optimizer + load_actor_optimizer_state (train/apply)",
                ],
            },
            "seconds": seconds,
            "host": {"python": platform.python_version(), "torch": torch.__version__},
        }
    )


def argument_parser() -> argparse.ArgumentParser:
    from .labor_actor_optim import add_actor_lr_arguments

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument(
        "--a-warm-start", type=Path, default=None, help="measure mode (required there)"
    )
    parser.add_argument("--a-critic", type=Path, default=None, help="measure mode (required there)")
    parser.add_argument("--out", type=Path, required=True, help="JSON report")
    parser.add_argument("--d7-rows", type=int, default=64)
    parser.add_argument("--a-draws", type=int, default=16)
    parser.add_argument("--calibration-report", type=Path, default=None)
    parser.add_argument("--no-dry-run", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply mode: the real actor-only update of train --a-mode off (rollback "
        "active), written to --out-checkpoint; the critic is not updated",
    )
    parser.add_argument(
        "--out-checkpoint", type=Path, default=None,
        help="apply mode: the new checkpoint (refused if it exists)",
    )
    parser.add_argument(
        "--target-kl", type=float, default=None,
        help=f"apply mode: rollback threshold on the audit KL (default {APPLY_DEFAULT_TARGET_KL}, "
        "train's); measure mode's dry run always uses 1e30",
    )
    # train's actor-update defaults (the dry run and apply mode use them)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    add_actor_lr_arguments(parser)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--audit-rows", type=int, default=64)
    parser.add_argument(
        "--threads", type=int, default=max(1, torch.get_num_threads()),
        help="torch intra-op threads (default: torch's current count)",
    )
    return parser


def _write_json(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.apply:
        if args.out_checkpoint is None:
            parser.error("--apply requires --out-checkpoint")
        if args.a_warm_start is not None or args.a_critic is not None or args.no_dry_run:
            parser.error(
                "--apply is train --a-mode off: --a-warm-start/--a-critic/--no-dry-run do not apply"
            )
        if args.target_kl is None:
            args.target_kl = APPLY_DEFAULT_TARGET_KL
        result = apply_update(args)
        _write_json(args.out, result)
        update = result["update"]
        print(json.dumps({
            "attempted/accepted/rolledBack": [update["attemptedSteps"], update["acceptedSteps"],
                                              update["rolledBackSteps"]],
            "perStepAuditKL": update["perStepAuditKL"],
            "adamSteps": [result["adam"]["stepsBefore"], result["adam"]["stepsAfter"]],
            "jointMeanAbsDelta": result["logProbs"]["afterVsBefore"]["initialJoint"]["meanAbs"],
            "outputSha256": result["output"]["checkpointSha256"],
            "seconds": result["seconds"].get("total"),
        }, sort_keys=True))
        return 0
    if args.a_warm_start is None or args.a_critic is None:
        parser.error("measure mode requires --a-warm-start and --a-critic")
    if args.out_checkpoint is not None or args.target_kl is not None:
        parser.error("--out-checkpoint/--target-kl are apply-mode options (--apply)")
    result = measure(args)
    _write_json(args.out, result)
    summary = {
        head: (v.get("reduction"), v.get("admit"), v.get("aBlind"))
        for head, v in result["d7"].items()
    }
    print(json.dumps({"d7 (reduction, admit, aBlind)": summary,
                      "dryRunKL": result["dryRunKL"]["perEpoch"],
                      "seconds": result["seconds"].get("total")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "APPLY_SCHEMA",
    "MEASURE_SCHEMA",
    "STEP_CHECKS_KEY",
    "apply_update",
    "argument_parser",
    "behaviour_log_probs",
    "log_prob_deltas",
    "main",
    "measure",
    "policy_log_probs",
]
