"""Pilot-only provenance: the ungated actor update (``train --pilot-ungated-actor``).

A pilot PPO run may update the actor while the four-head actor-gate calibration report is
closed or absent. That is an explicit, opt-in bypass of the independent-return gate, so
every checkpoint it produces (and every checkpoint descended from one through ``train``
or ``labor_compose_checkpoint``) carries the persistent list :data:`PILOT_BYPASS_KEY`.
Readiness/acceptance tooling refuses such a checkpoint with
:func:`require_no_pilot_bypass` (``labor_r5_acceptance.evaluate_fit``). Diagnostics and
the PPO loop itself (collect, anchors, calibration, train) still read it.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

PILOT_BYPASS_KEY = "pilot_ungated_actor_updates"
PILOT_BYPASS_GATE = "pilot-bypass"
PILOT_BYPASS_SCOPE = "pilot-ungated-actor-update-never-readiness"


def pilot_bypass_history(state: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The checkpoint's bypass records (empty when it never used the bypass)."""

    if not isinstance(state, Mapping) or PILOT_BYPASS_KEY not in state:
        return []
    history = state[PILOT_BYPASS_KEY]
    if not isinstance(history, list) or not history or any(
        not isinstance(entry, Mapping) for entry in history
    ):
        raise ValueError(f"checkpoint {PILOT_BYPASS_KEY} is malformed")
    return [copy.deepcopy(dict(entry)) for entry in history]


def merged_pilot_bypass_history(*states: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Union of several checkpoints' records (order kept, exact duplicates dropped)."""

    merged: list[dict[str, Any]] = []
    for state in states:
        for entry in pilot_bypass_history(state):
            if entry not in merged:
                merged.append(entry)
    return merged


def require_no_pilot_bypass(state: Mapping[str, Any] | None, context: str) -> None:
    """Refuse a checkpoint whose actor ever updated through the pilot bypass."""

    history = pilot_bypass_history(state)
    if history:
        iterations = [entry.get("iteration") for entry in history]
        raise ValueError(
            f"{context} refuses a checkpoint whose actor was updated with "
            f"--pilot-ungated-actor (no open actor-gate calibration) at iterations "
            f"{iterations}; pilot checkpoints are never readiness or acceptance evidence"
        )


def calibration_report_status(
    report: Mapping[str, Any] | None, report_sha256: str | None
) -> dict[str, Any]:
    """The supplied calibration report's actual status (or that none was supplied)."""

    if report is None:
        return {"supplied": False}
    heads = report.get("heads") if isinstance(report.get("heads"), Mapping) else {}
    return {
        "supplied": True,
        "sha256": report_sha256,
        "actorGateOpen": report.get("actorGateOpen"),
        "policyCurrent": report.get("policyCurrent"),
        "normalizationCurrent": report.get("normalizationCurrent"),
        "registryCurrent": report.get("registryCurrent"),
        "dataSha256": report.get("dataSha256"),
        "worldSeeds": list(report.get("worldSeeds") or []),
        "headPass": {
            str(name): bool(isinstance(value, Mapping) and value.get("pass") is True)
            for name, value in sorted(heads.items())
        },
    }


# ---------------------------------------------------------------------------
# The PPO pilot loop's checks and summaries (ops/r5/pilot/ppo_pilot_job.sh).
# Pure Python over the files each stage writes; no native bridge.
# ---------------------------------------------------------------------------

PILOT_ROLLOUT_BLOCK = 91100
PILOT_GATE_BLOCK = 91200
PILOT_BLOCK_STRIDE = 10
PILOT_MAX_ITERATIONS = 9  # 91100-91182 / 91200-91282; 91190 and 91290 are the anchor probe's
PILOT_TRAIN_SEED_BASE = 10_000_000
PILOT_GATE_SEED_BASE = 20_000_000
PILOT_SEED_BASE_STRIDE = 100_000
PILOT_SUMMARY_SCHEMA = "r5-ppo-pilot-iteration-summary-v1"


def pilot_plan(
    iteration: int, *, rollout_worlds: int = 3, train_anchor_worlds: int = 3, gate_worlds: int = 2
) -> dict[str, Any]:
    """Deterministic worlds and continuation seed bases of pilot iteration ``iteration``."""

    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise ValueError("pilot iteration must be an integer")
    if not 0 <= iteration < PILOT_MAX_ITERATIONS:
        raise ValueError(f"pilot iteration must be in 0..{PILOT_MAX_ITERATIONS - 1}")
    for value, label, limit in (
        (rollout_worlds, "rollout_worlds", PILOT_BLOCK_STRIDE),
        (gate_worlds, "gate_worlds", PILOT_BLOCK_STRIDE),
        (train_anchor_worlds, "train_anchor_worlds", rollout_worlds),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
            raise ValueError(f"{label} must be in 1..{limit}")
    base = PILOT_ROLLOUT_BLOCK + PILOT_BLOCK_STRIDE * iteration
    rollout = [base + index for index in range(rollout_worlds)]
    return {
        "iteration": iteration,
        "rolloutWorlds": rollout,
        "trainAnchorWorlds": rollout[:train_anchor_worlds],
        "gateWorlds": [
            PILOT_GATE_BLOCK + PILOT_BLOCK_STRIDE * iteration + index
            for index in range(gate_worlds)
        ],
        "trainAnchorSeedBase": PILOT_TRAIN_SEED_BASE + PILOT_SEED_BASE_STRIDE * iteration,
        "gateSeedBase": PILOT_GATE_SEED_BASE + PILOT_SEED_BASE_STRIDE * iteration,
    }


def _reset_state(players: int, width: int, height: int, separation: int, movement: bool) -> dict:
    from .labor_bc import LaborResetOptions

    return LaborResetOptions(
        players=players, movement=movement, width=width, height=height, separation=separation
    ).state_dict()


def _load_trainer(path: Any) -> tuple[Any, ...]:
    """(state, actor, task, costs, normalizer, policy digest, critic digest, norm digest)."""

    import torch

    from .labor_model import labor_checkpoint_personality
    from .labor_training import (
        _models_from_checkpoint,
        _normalization_digest,
        _require_trainer_checkpoint,
        labor_behavior_configuration,
    )
    from .labor_value_calibration import labor_critic_digest
    from .value_calibration import behavior_policy_digest

    state = torch.load(path, map_location="cpu", weights_only=True)
    _require_trainer_checkpoint(state)
    _, actor, task, costs, normalizer = _models_from_checkpoint(state)
    behavior = labor_behavior_configuration(labor_checkpoint_personality(state))
    return (
        state, actor, task, costs, normalizer,
        behavior_policy_digest(actor, behavior),
        labor_critic_digest(task, costs),
        _normalization_digest(normalizer),
    )


def pilot_preflight(
    checkpoint: Any, plan: Mapping[str, Any], *, expected_iteration: int, anchor_paths: int = 4
) -> dict[str, Any]:
    """Refuse an iteration whose checkpoint or worlds would break the PPO loop's guards."""

    from .labor_anchor_producer import _path_rng_seeds
    from .labor_worlds import require_training_worlds
    from .protocol import seeds as checked_seeds
    from .provenance import checkpoint_hash

    state, *_, policy, critic, normalization = _load_trainer(checkpoint)
    if int(state["iteration"]) != int(expected_iteration):
        raise ValueError(
            f"pilot checkpoint iteration {state['iteration']} != expected {expected_iteration}"
        )
    rollout = [int(w) for w in plan["rolloutWorlds"]]
    train_anchor = [int(w) for w in plan["trainAnchorWorlds"]]
    gate = [int(w) for w in plan["gateWorlds"]]
    checked_seeds(rollout + train_anchor + gate, "pilot worlds")  # sealed 3000-3059 refused
    require_training_worlds(rollout + train_anchor, "pilot training worlds")
    require_training_worlds(gate, "pilot actor-gate worlds")  # reserved worlds never used here
    history = set(state["critic_training_world_seeds"])
    units = set(state["critic_training_independent_unit_ids"])
    if set(gate) & (history | set(rollout) | set(train_anchor)):
        raise ValueError("pilot actor-gate worlds overlap rollout/anchor or critic-training data")
    if {f"world-{world}" for world in gate} & units:
        raise ValueError("pilot actor-gate units are already critic-training units")
    reused = sorted(set(rollout + train_anchor) & history)
    if reused:
        raise ValueError(f"pilot rollout/anchor worlds {reused} already trained the critic")
    for base, worlds in ((plan["trainAnchorSeedBase"], train_anchor), (plan["gateSeedBase"], gate)):
        checked_seeds(
            [
                seed
                for index in range(len(worlds))
                for path in range(anchor_paths)
                for seed in _path_rng_seeds(int(base), index, path)
            ],
            "pilot continuation seeds",
        )
    return {
        "checkpointSha256": checkpoint_hash(checkpoint),
        "iteration": int(state["iteration"]),
        "behaviorPolicyDigest": policy,
        "criticDigest": critic,
        "normalizationDigest": normalization,
        "criticTrainingWorlds": len(history),
        "criticTrainingUnits": len(units),
        "pilotBypassUpdates": len(pilot_bypass_history(state)),
        "plan": dict(plan),
    }


def pilot_check_rollout(
    rollout: Any, checkpoint: Any, *, worlds: Sequence[int], turns: int, reset: Mapping[str, Any]
) -> dict[str, Any]:
    """The rollout is complete, on-policy for ``checkpoint`` and on the requested worlds."""

    from collections import Counter

    from .labor_training import _load_rollout
    from .provenance import checkpoint_hash

    *_, policy, critic, normalization = _load_trainer(checkpoint)
    batch, _, _, meta = _load_rollout(rollout)
    batch.require_behaviour_head_logp()
    rows = meta["decisionRows"]
    if len(rows) != batch.parent_count:
        raise ValueError("pilot rollout decision rows differ from its parents")
    if meta["worldSeeds"] != [int(w) for w in worlds] or int(meta["turns"]) != int(turns):
        raise ValueError(f"pilot rollout worlds/turns {meta['worldSeeds']}/{meta['turns']} differ")
    if meta["sourceCheckpointSha256"] != checkpoint_hash(checkpoint):
        raise ValueError("pilot rollout belongs to another checkpoint")
    if not batch.behavior_policy_digest == meta["behaviorPolicyDigest"] == policy:
        raise ValueError("pilot rollout is not on-policy for the checkpoint")
    if batch.advantage_critic_digest != critic or batch.normalization_binding_digest != normalization:
        raise ValueError("pilot rollout critic/normalizer binding differs from the checkpoint")
    if meta.get("resetOptions") != dict(reset) or meta.get("resetOptionsSource") != "explicit":
        raise ValueError(f"pilot rollout world options {meta.get('resetOptions')} != {dict(reset)}")
    if meta.get("criticWarmup") is not None:
        raise ValueError("pilot rollout is a critic-warmup rollout")
    per_world = Counter(int(r["world"]) for r in rows)
    missing = [w for w in worlds if per_world[int(w)] == 0]
    if missing:
        raise ValueError(f"pilot rollout worlds {missing} produced no decisions")
    return {
        "parents": int(batch.parent_count),
        "retries": int(len(batch.retry_parent)),
        "parentsPerWorld": {str(w): per_world[int(w)] for w in worlds},
        "decisionTurnsPerWorld": {
            str(w): len({r["turn"] for r in rows if int(r["world"]) == int(w)}) for w in worlds
        },
        "terminated": sum(bool(r["terminated"]) for r in rows),
        "behaviorPolicyDigest": policy,
        "resetOptions": meta["resetOptions"],
    }


def pilot_check_anchor(
    anchor: Any, checkpoint: Any, *, role: str, worlds: Sequence[int], reset: Mapping[str, Any]
) -> dict[str, Any]:
    """The anchor loads for its role and is bound to the checkpoint and the world options."""

    from .labor_value_calibration import load_labor_anchor
    from .provenance import checkpoint_hash

    *_, policy, _, normalization = _load_trainer(checkpoint)
    data = load_labor_anchor(anchor, role)
    source = data.metadata["source"]
    if source["sourceCheckpointSha256"] != checkpoint_hash(checkpoint):
        raise ValueError(f"pilot {role} anchor belongs to another checkpoint")
    if source["sourcePolicyDigest"] != policy:
        raise ValueError(f"pilot {role} anchor is stale for the checkpoint's policy")
    if source["normalizationBindingDigest"] != normalization:
        raise ValueError(f"pilot {role} anchor uses another normalizer")
    if [int(w) for w in source["worldSeeds"]] != [int(w) for w in worlds]:
        raise ValueError(f"pilot {role} anchor worlds {source['worldSeeds']} != {list(worlds)}")
    if data.metadata.get("resetOptions") != dict(reset):
        raise ValueError(
            f"pilot {role} anchor world options {data.metadata.get('resetOptions')} "
            f"!= {dict(reset)}"
        )
    ranking = data.ranking
    return {
        "role": role,
        "rows": len(data),
        "independentUnits": sorted(set(data.independent_unit_ids.tolist())),
        "rankRows": 0 if ranking is None else int(len(ranking.group_ids)),
        "rankGroups": 0 if ranking is None else int(len(set(ranking.group_ids.tolist()))),
        "targetsAvailable": {
            head: int(mask.sum()) for head, mask in sorted(data.target_masks.items())
        },
        "sha256": checkpoint_hash(anchor),
    }


def calibration_gate_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """The calibration report's gate status per head, compacted."""

    heads = {}
    for name, head in sorted((report.get("heads") or {}).items()):
        ranking = head.get("ranking") or {}
        heads[name] = {
            key: head.get(key)
            for key in (
                "pass", "status", "rows", "coveragePass", "magnitudePass",
                "zeroTargetMagnitudePass", "targetNearZero",
                "normalizedConservativeMeanAbsoluteError", "normalizedConservativeAbsoluteBias",
                "normalizedEqualIndependentUnitBias",
            )
            if key in head
        }
        heads[name]["ranking"] = {
            key: ranking.get(key)
            for key in (
                "status", "required", "informativeIndependentUnits",
                "equalIndependentUnitAccuracy", "uncertainTargetPairs", "pairCount",
            )
            if key in ranking
        }
    return {
        "actorGateOpen": report.get("actorGateOpen"),
        "policyCurrent": report.get("policyCurrent"),
        "normalizationCurrent": report.get("normalizationCurrent"),
        "registryCurrent": report.get("registryCurrent"),
        "worldSeeds": report.get("worldSeeds"),
        "heads": heads,
        "failingHeads": sorted(n for n, head in heads.items() if head.get("pass") is not True),
    }


def pilot_check_calibration(report_path: Any, checkpoint: Any, anchor: Any) -> dict[str, Any]:
    """The report scores this checkpoint on this actor-gate anchor."""

    import json
    from pathlib import Path

    from .provenance import checkpoint_hash

    *_, policy, critic, _ = _load_trainer(checkpoint)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("kind") != "model9-labor-value-calibration-report-v1":
        raise ValueError("not a labor calibration report")
    if report.get("dataRole") != "actor-gate" or report.get("dataSha256") != checkpoint_hash(anchor):
        raise ValueError("pilot calibration report scores another actor-gate anchor")
    if report.get("currentPolicyDigest") != policy or report.get("sourceCriticDigest") != critic:
        raise ValueError("pilot calibration report scores another checkpoint")
    return calibration_gate_summary(report)


def pilot_summary(
    *,
    iteration: int,
    checkpoint_in: Any,
    checkpoint_out: Any,
    rollout: Any,
    train_anchor: Any,
    gate_anchor: Any | None,
    calibration: Any | None,
    stage_seconds: Mapping[str, Any],
    bypass_expected: bool,
) -> dict[str, Any]:
    """One pilot iteration: data, gate, critic losses, actor steps, KL and log-prob moves."""

    import json
    from pathlib import Path

    from .labor_a_measure import (
        _per_epoch_kl,
        behaviour_log_probs,
        log_prob_deltas,
        policy_log_probs,
    )
    from .labor_training import _load_rollout
    from .provenance import checkpoint_hash

    state_in, actor_in, *_ = _load_trainer(checkpoint_in)
    state_out, actor_out, _, _, _, policy_out, critic_out, _ = _load_trainer(checkpoint_out)
    update = state_out.get("last_update") or {}
    if update.get("rolloutSha256") != checkpoint_hash(rollout):
        raise ValueError("pilot output checkpoint was trained on another rollout")
    if update.get("anchorSha256") != checkpoint_hash(train_anchor):
        raise ValueError("pilot output checkpoint was trained on another train anchor")
    if int(state_out["iteration"]) != int(state_in["iteration"]) + 1:
        raise ValueError("pilot output checkpoint does not advance the iteration by one")
    actor = dict(update.get("actor") or {})
    if bypass_expected and "pilotBypass" not in update:
        raise ValueError("pilot train ran without --pilot-ungated-actor")
    batch, _, _, _ = _load_rollout(rollout)
    before = policy_log_probs(actor_in, batch)
    after = policy_log_probs(actor_out, batch)
    moves = log_prob_deltas(after, before)
    sanity = log_prob_deltas(before, behaviour_log_probs(batch))
    epochs = _per_epoch_kl(actor) if "sampledNonnegativeKL" in actor else []
    from .labor_value_calibration import load_labor_anchor

    anchors = {}
    for label, path, role in (
        ("trainAnchor", train_anchor, "train-anchor"),
        ("gateAnchor", gate_anchor, "actor-gate"),
    ):
        if path is None:
            continue
        data = load_labor_anchor(path, role)
        ranking = getattr(data, "ranking", None)
        anchors[label] = {
            "rows": len(data),
            "independentUnits": len(set(data.independent_unit_ids.tolist())),
            "rankRows": 0 if ranking is None else int(len(ranking.group_ids)),
        }
    gate = None
    if calibration is not None:
        gate = calibration_gate_summary(
            json.loads(Path(calibration).read_text(encoding="utf-8"))
        )
    if "attemptedSteps" not in actor:
        mode = "closed-critic-only"
    else:
        mode = update.get("actorGate", "calibration-report")
    return {
        "schema": PILOT_SUMMARY_SCHEMA,
        "iteration": int(iteration),
        "checkpointIteration": {"in": int(state_in["iteration"]), "out": int(state_out["iteration"])},
        "sha256": {
            "checkpointIn": checkpoint_hash(checkpoint_in),
            "checkpointOut": checkpoint_hash(checkpoint_out),
            "rollout": checkpoint_hash(rollout),
            "trainAnchor": checkpoint_hash(train_anchor),
            "gateAnchor": None if gate_anchor is None else checkpoint_hash(gate_anchor),
            "calibration": None if calibration is None else checkpoint_hash(calibration),
        },
        "rolloutParents": int(batch.parent_count),
        "rolloutRetries": int(len(batch.retry_parent)),
        "anchors": anchors,
        "gate": gate,
        "actorGateMode": mode,
        "pilotBypass": update.get("pilotBypass"),
        "pilotBypassUpdatesInCheckpoint": len(pilot_bypass_history(state_out)),
        "critic": {
            "epochs": update.get("criticEpochs"),
            "initialLoss": update.get("initialCriticLoss"),
            "loss": update.get("criticLoss"),
            "independentHeadLosses": update.get("independentHeadLosses"),
            "digestOut": critic_out,
        },
        "actor": {
            "reason": actor.get("reason"),
            "attemptedSteps": actor.get("attemptedSteps"),
            "acceptedSteps": actor.get("acceptedSteps"),
            "rolledBack": actor.get("actorStepRolledBack"),
            "sampledNonnegativeKL": actor.get("sampledNonnegativeKL"),
            "perEpoch": epochs,
            "ratioOverflow": actor.get("ratioOverflow"),
            "lrsAfterUpdate": (actor.get("actorOptimizer") or {}).get("lrsAfterUpdate"),
            "behaviorPolicyDigestOut": policy_out,
        },
        "meanAbsDeltaLogp": {name: value["meanAbs"] for name, value in moves.items()},
        "maxAbsDeltaLogp": {name: value["maxAbs"] for name, value in moves.items()},
        "inputVsBehaviourMaxAbsLogp": {name: value["maxAbs"] for name, value in sanity.items()},
        "stageSeconds": dict(stage_seconds),
        "scope": PILOT_BYPASS_SCOPE if bypass_expected else "pilot-not-readiness",
    }


def _cli(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="r5 PPO pilot stage checks (no native bridge)")
    commands = parser.add_subparsers(dest="command", required=True)
    world_options = argparse.ArgumentParser(add_help=False)
    world_options.add_argument("--players", type=int, default=16)
    world_options.add_argument("--width", type=int, default=128)
    world_options.add_argument("--height", type=int, default=64)
    world_options.add_argument("--separation", type=int, default=9)
    world_options.add_argument("--no-movement", action="store_true", default=False)
    plan = commands.add_parser("plan")
    plan.add_argument("--iteration", type=int, required=True)
    plan.add_argument("--rollout-worlds", type=int, default=3)
    plan.add_argument("--train-anchor-worlds", type=int, default=3)
    plan.add_argument("--gate-worlds", type=int, default=2)
    pre = commands.add_parser("preflight")
    pre.add_argument("--checkpoint", type=Path, required=True)
    pre.add_argument("--plan", type=Path, required=True)
    pre.add_argument("--expected-iteration", type=int, required=True)
    pre.add_argument("--anchor-paths", type=int, default=4)
    roll = commands.add_parser("check-rollout", parents=[world_options])
    roll.add_argument("--rollout", type=Path, required=True)
    roll.add_argument("--checkpoint", type=Path, required=True)
    roll.add_argument("--worlds", required=True)
    roll.add_argument("--turns", type=int, required=True)
    anchor = commands.add_parser("check-anchor", parents=[world_options])
    anchor.add_argument("--anchor", type=Path, required=True)
    anchor.add_argument("--checkpoint", type=Path, required=True)
    anchor.add_argument("--role", choices=("train-anchor", "actor-gate"), required=True)
    anchor.add_argument("--worlds", required=True)
    calibration = commands.add_parser("check-calibration")
    calibration.add_argument("--report", type=Path, required=True)
    calibration.add_argument("--checkpoint", type=Path, required=True)
    calibration.add_argument("--anchor", type=Path, required=True)
    summary = commands.add_parser("summary")
    summary.add_argument("--iteration", type=int, required=True)
    summary.add_argument("--checkpoint-in", type=Path, required=True)
    summary.add_argument("--checkpoint-out", type=Path, required=True)
    summary.add_argument("--rollout", type=Path, required=True)
    summary.add_argument("--train-anchor", type=Path, required=True)
    summary.add_argument("--gate-anchor", type=Path, default=None)
    summary.add_argument("--calibration", type=Path, default=None)
    summary.add_argument("--stage-seconds", type=Path, required=True)
    summary.add_argument("--bypass-expected", action="store_true", default=False)
    args = parser.parse_args(argv)

    def worlds(text: str) -> list[int]:
        return [int(value) for value in text.split(",") if value]

    def reset() -> dict:
        return _reset_state(
            args.players, args.width, args.height, args.separation, not args.no_movement
        )

    if args.command == "plan":
        result = pilot_plan(
            args.iteration, rollout_worlds=args.rollout_worlds,
            train_anchor_worlds=args.train_anchor_worlds, gate_worlds=args.gate_worlds,
        )
    elif args.command == "preflight":
        result = pilot_preflight(
            args.checkpoint, json.loads(args.plan.read_text(encoding="utf-8")),
            expected_iteration=args.expected_iteration, anchor_paths=args.anchor_paths,
        )
    elif args.command == "check-rollout":
        result = pilot_check_rollout(
            args.rollout, args.checkpoint, worlds=worlds(args.worlds), turns=args.turns,
            reset=reset(),
        )
    elif args.command == "check-anchor":
        result = pilot_check_anchor(
            args.anchor, args.checkpoint, role=args.role, worlds=worlds(args.worlds),
            reset=reset(),
        )
    elif args.command == "check-calibration":
        result = pilot_check_calibration(args.report, args.checkpoint, args.anchor)
    else:
        result = pilot_summary(
            iteration=args.iteration, checkpoint_in=args.checkpoint_in,
            checkpoint_out=args.checkpoint_out, rollout=args.rollout,
            train_anchor=args.train_anchor, gate_anchor=args.gate_anchor,
            calibration=args.calibration,
            stage_seconds=json.loads(args.stage_seconds.read_text(encoding="utf-8")),
            bypass_expected=args.bypass_expected,
        )
    json.dump(result, sys.stdout, indent=1, sort_keys=True, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
