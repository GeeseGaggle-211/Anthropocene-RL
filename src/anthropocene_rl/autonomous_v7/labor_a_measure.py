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
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

MEASURE_SCHEMA = "labor-a-d7-measurement-v1"


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

    # train builds the actor optimizer from the checkpoint before anything else.
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    actor_optimizer.load_state_dict(state["actor_optimizer"])
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


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--a-warm-start", type=Path, required=True)
    parser.add_argument("--a-critic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="JSON report")
    parser.add_argument("--d7-rows", type=int, default=64)
    parser.add_argument("--a-draws", type=int, default=16)
    parser.add_argument("--calibration-report", type=Path, default=None)
    parser.add_argument("--no-dry-run", action="store_true")
    # train's actor-update defaults (the dry run uses them)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
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


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    result = measure(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)
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


__all__ = ["MEASURE_SCHEMA", "argument_parser", "main", "measure"]
