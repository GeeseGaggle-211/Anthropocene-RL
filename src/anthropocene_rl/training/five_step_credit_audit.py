"""TRAIN-only five-decision scope and delayed-cultivation-credit audit.

The audit performs no optimizer updates.  It stress-tests every TRAIN reset for
five decisions, then follows paired Development perturbations through a fixed
five-decision physical window.  DEVELOPMENT and HELDOUT are never selected.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anthropocene_rl.datasets.blobs import sha256_file
from anthropocene_rl.datasets.models import Partition, SnapshotRecord
from anthropocene_rl.observations.schema import cultivation_only_scope_violations
from anthropocene_rl.policies.elder_mlp import ElderActorCritic, ElderActorCriticConfig
from anthropocene_rl.training.action_credit_audit import (
    ACCEPTED_SPAN_FLOOR,
    PERTURBATION,
    PHYSICAL_RELATIVE_SPREAD_FLOOR,
    _action_values,
    _validate_checkpoint,
)
from anthropocene_rl.training.overfit import (
    _mapping,
    _observation_guard,
    _pick,
    _plain_mapping,
    _raw_observation,
    _requested_matches_sample,
    _sequence,
    _write_json,
)
from anthropocene_rl.training.pilot import (
    PILOT_FAMILIES,
    CampaignDataset,
    CampaignSelection,
    PilotConfig,
    _make_environment,
    _reset_request,
    _reward_denominator_violation,
    _validate_campaign_reset,
)
from anthropocene_rl.training.returns import discounted_interval_reward, physical_gamma

AUDIT_STEPS = 5
PROBE_DECISIONS = (0, 1, 2)
COHORT_PER_FAMILY = 8
INFORMATIVE_PROBE_MINIMUM = 8
INFORMATIVE_SNAPSHOT_MINIMUM = 4
DELAYED_EFFECT_FRACTION_MINIMUM = 0.50
RESPONSIVE_RATE_GAIN_MINIMUM = 0.15

ProgressCallback = Callable[[dict[str, Any]], None]


class AuditIntegrityError(RuntimeError):
    def __init__(self, message: str, payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.payload = {"message": message, **payload}


def run_five_step_credit_audit(
    *,
    checkpoint_directory: str | Path,
    previous_audit_summary: str | Path,
    evidence_directory: str | Path,
    bridge_dll: str | Path,
    dotnet: str,
    manifest_path: str | Path,
    blob_root: str | Path,
    normalizer_path: str | Path,
    snapshot_store_config: str | Path,
    dataset_manifest: str | Path | None,
    dataset_store_config: str | Path,
    repository_root: str | Path,
    seeds: tuple[int, ...] = (1701, 1901, 2101),
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the predeclared five-decision audit and persist hash-anchored evidence."""

    if tuple(seeds) != (1701, 1901, 2101):
        raise ValueError("The five-step audit requires seeds 1701, 1901, and 2101")
    evidence_dir = Path(evidence_directory)
    checkpoint_dir = Path(checkpoint_directory)
    bridge_path = Path(bridge_dll).resolve()
    dataset = CampaignDataset(
        manifest_path=manifest_path,
        blob_root=blob_root,
        normalizer_path=normalizer_path,
        snapshot_store_config=snapshot_store_config,
        dataset_manifest=dataset_manifest,
        dataset_store_config=dataset_store_config,
    )
    train = dataset.by_partition[Partition.TRAIN]
    if len(train) != 210:
        raise ValueError(f"Expected 210 frozen TRAIN snapshots, found {len(train)}")

    previous_path = Path(previous_audit_summary)
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    _validate_previous_audit(previous, dataset, seeds)
    models, payloads = _load_models(checkpoint_dir, dataset, seeds)
    cohorts = _select_counterfactual_cohorts(train, previous, seeds)
    cohort_payload = {
        "selected_before_outcomes": True,
        "partition": Partition.TRAIN.value,
        "snapshots_per_seed": 64,
        "snapshots_per_family": COHORT_PER_FAMILY,
        "preserved_previous_snapshots_per_family": 4,
        "seeds": {
            str(seed): [record.snapshot_id for record in cohorts[seed]] for seed in seeds
        },
    }
    _write_json(evidence_dir / "counterfactual-cohorts.json", cohort_payload)

    identities = _audit_identities(
        repository_root=Path(repository_root),
        bridge_path=bridge_path,
        dataset=dataset,
        normalizer_path=Path(normalizer_path),
        previous_path=previous_path,
        checkpoint_dir=checkpoint_dir,
        payloads=payloads,
        seeds=seeds,
    )
    config = {
        "steps": AUDIT_STEPS,
        "structural_profiles": [profile["name"] for profile in _profile_definitions(seeds)],
        "structural_train_snapshots": len(train),
        "counterfactual_snapshots_per_seed": 64,
        "counterfactual_snapshots_per_family": COHORT_PER_FAMILY,
        "probe_decisions_one_based": [1, 2, 3],
        "development_perturbation": PERTURBATION,
        "optimizer_updates": 0,
        "partitions_read": [Partition.TRAIN.value],
        "development_read": False,
        "heldout_read": False,
    }

    try:
        structural = _run_structural_component(
            dataset=dataset,
            train=train,
            models=models,
            bridge_path=bridge_path,
            dotnet=dotnet,
            progress=progress,
        )
        _write_json(evidence_dir / "structural.json", structural)
        seed_reports: list[dict[str, Any]] = []
        for seed in seeds:
            report = _run_counterfactual_seed(
                seed=seed,
                model=models[seed],
                payload=payloads[seed],
                records=cohorts[seed],
                dataset=dataset,
                bridge_path=bridge_path,
                dotnet=dotnet,
                progress=progress,
            )
            seed_reports.append(report)
            _write_json(evidence_dir / f"seed-{seed}.json", report)
    except AuditIntegrityError as error:
        failure = {
            "status": "STOP_NO_FIVE_STEP_PPO",
            "reason": "integrity-or-scope-failure",
            "failure": error.payload,
            "identities": identities,
            "configuration": config,
            "cohorts": cohort_payload,
        }
        failure["evidence_sha256"] = _digest(failure)
        _write_json(evidence_dir / "failure.json", failure)
        _write_json(evidence_dir / "summary.json", failure)
        return failure

    assessment = _assess_delayed_effect(seed_reports, structural)
    raw_files = {
        "structural.json": sha256_file(evidence_dir / "structural.json")[0],
        "counterfactual-cohorts.json": sha256_file(
            evidence_dir / "counterfactual-cohorts.json"
        )[0],
        **{
            f"seed-{seed}.json": sha256_file(evidence_dir / f"seed-{seed}.json")[0]
            for seed in seeds
        },
    }
    result = {
        "status": assessment["status"],
        "scope": "train-only-five-decision-structural-and-delayed-credit-audit",
        "identities": identities,
        "configuration": config,
        "structural_summary": structural["summary"],
        "seeds": [_compact_seed_report(report) for report in seed_reports],
        "assessment": assessment,
        "raw_file_sha256": raw_files,
    }
    result["evidence_sha256"] = _digest(result)
    _write_json(evidence_dir / "summary.json", result)
    return result


def _validate_previous_audit(
    previous: dict[str, Any],
    dataset: CampaignDataset,
    seeds: tuple[int, ...],
) -> None:
    if previous.get("evidence_sha256") != _digest(previous):
        raise ValueError("Previous action-credit audit evidence digest mismatch")
    if previous.get("status") != "ACTION_CREDIT_FAILURE":
        raise ValueError("Five-step audit requires the prior ACTION_CREDIT_FAILURE")
    if previous.get("heldout_read") or previous.get("development_read"):
        raise ValueError("Previous action-credit audit did not preserve tuning scope")
    if previous["dataset"]["manifest_sha256"] != dataset.manifest_sha256:
        raise ValueError("Previous action-credit audit used a different snapshot bank")
    if tuple(report["seed"] for report in previous["seeds"]) != seeds:
        raise ValueError("Previous action-credit audit seed identities differ")


def _load_models(
    checkpoint_dir: Path,
    dataset: CampaignDataset,
    seeds: tuple[int, ...],
) -> tuple[dict[int, ElderActorCritic], dict[int, dict[str, Any]]]:
    models: dict[int, ElderActorCritic] = {}
    payloads: dict[int, dict[str, Any]] = {}
    selected_updates: set[int] = set()
    for seed in seeds:
        payload = torch.load(
            checkpoint_dir / f"seed-{seed}.pt",
            map_location="cpu",
            weights_only=False,
        )
        _validate_checkpoint(payload, dataset, seed)
        config = ElderActorCriticConfig(**payload["actor_critic_config"])
        model = ElderActorCritic(config)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models[seed] = model
        payloads[seed] = payload
        selected_updates.add(int(payload["identities"]["selected_update"]))
    if selected_updates != {168}:
        raise ValueError(f"Expected frozen update-168 checkpoints, found {selected_updates}")
    return models, payloads


def _select_counterfactual_cohorts(
    train: tuple[SnapshotRecord, ...],
    previous: dict[str, Any],
    seeds: tuple[int, ...],
) -> dict[int, tuple[SnapshotRecord, ...]]:
    by_id = {record.snapshot_id: record for record in train}
    by_family = {
        family: sorted(
            (record for record in train if record.scenario_family == family),
            key=lambda record: record.snapshot_id,
        )
        for family in PILOT_FAMILIES
    }
    previous_by_seed = {int(report["seed"]): report for report in previous["seeds"]}
    result: dict[int, tuple[SnapshotRecord, ...]] = {}
    for seed in seeds:
        preserved_ids = previous_by_seed[seed]["batch"]["snapshot_ids"]
        preserved = [by_id[snapshot_id] for snapshot_id in preserved_ids]
        preserved_counts = Counter(record.scenario_family for record in preserved)
        if preserved_counts != Counter({family: 4 for family in PILOT_FAMILIES}):
            raise ValueError(f"Prior audit seed {seed} was not four-per-family balanced")
        if len({record.snapshot_id for record in preserved}) != 32:
            raise ValueError(f"Prior audit seed {seed} did not use distinct snapshots")

        selected = list(preserved)
        rng = np.random.default_rng(seed + 5_000_003)
        for family in PILOT_FAMILIES:
            used = {record.snapshot_id for record in preserved if record.scenario_family == family}
            candidates = [
                record for record in by_family[family] if record.snapshot_id not in used
            ]
            rng.shuffle(candidates)
            selected.extend(candidates[: COHORT_PER_FAMILY - len(used)])
        rng.shuffle(selected)
        counts = Counter(record.scenario_family for record in selected)
        if counts != Counter({family: COHORT_PER_FAMILY for family in PILOT_FAMILIES}):
            raise RuntimeError(f"Five-step cohort for seed {seed} is not balanced")
        if len({record.snapshot_id for record in selected}) != 64:
            raise RuntimeError(f"Five-step cohort for seed {seed} is not distinct")
        result[seed] = tuple(selected)
    return result


def _profile_definitions(seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for field in (0.0, 0.5, 1.0):
        for development in (0.0, 0.5, 1.0):
            profiles.append(
                {
                    "name": f"constant-field-{field:g}-development-{development:g}",
                    "kind": "constant",
                    "values": ((field, development),),
                }
            )
    profiles.extend(
        [
            {
                "name": "alternating-00-11",
                "kind": "alternating",
                "values": ((0.0, 0.0), (1.0, 1.0)),
            },
            {
                "name": "alternating-01-10",
                "kind": "alternating",
                "values": ((0.0, 1.0), (1.0, 0.0)),
            },
        ]
    )
    profiles.extend(
        {
            "name": f"policy-seed-{seed}",
            "kind": "policy",
            "seed": seed,
        }
        for seed in seeds
    )
    if len(profiles) != 14:
        raise AssertionError("Five-step structural profile count changed")
    return profiles


def _run_structural_component(
    *,
    dataset: CampaignDataset,
    train: tuple[SnapshotRecord, ...],
    models: dict[int, ElderActorCritic],
    bridge_path: Path,
    dotnet: str,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    profiles = _profile_definitions(tuple(models))
    env = _make_environment(
        bridge_path=bridge_path,
        dotnet=dotnet,
        selection=dataset.selection(train[0]),
        episode_limit=AUDIT_STEPS,
    )
    rows: list[dict[str, Any]] = []
    try:
        for profile_index, profile in enumerate(profiles, start=1):
            for record in train:
                rows.append(
                    _run_structural_episode(
                        env=env,
                        selection=dataset.selection(record),
                        profile=profile,
                        models=models,
                    )
                )
            _notify(
                progress,
                phase="structural",
                completed_profiles=profile_index,
                total_profiles=len(profiles),
                completed_episodes=profile_index * len(train),
            )
    finally:
        env.close()
    transition_count = sum(row["steps"] for row in rows)
    return {
        "scope": "all-train-five-decision-fourteen-profile-stress",
        "profiles": [profile["name"] for profile in profiles],
        "rows": rows,
        "summary": {
            "status": "PASS",
            "train_snapshots": len(train),
            "profiles": len(profiles),
            "episodes": len(rows),
            "transitions": transition_count,
            "scope_violations": 0,
            "replay_failures": 0,
            "request_mismatches": 0,
            "non_finite_values": 0,
            "reward_denominator_violations": 0,
        },
    }


def _run_structural_episode(
    *,
    env: Any,
    selection: CampaignSelection,
    profile: dict[str, Any],
    models: dict[int, ElderActorCritic],
) -> dict[str, Any]:
    observation, info = env.reset(options=_reset_request(selection, AUDIT_STEPS))
    _validate_campaign_reset(_plain_mapping(info["native_result"]), selection)
    prefix: list[dict[str, Any]] = []
    _validate_scope_and_observation(
        selection=selection,
        phase="reset",
        observation=observation,
        info=info,
        prefix=prefix,
        context={"profile": profile["name"]},
    )
    reset_state = _mapping(_plain_mapping(info["native_result"]), "StateDiagnostics")
    prior_population = int(_pick(reset_state, "Population"))
    rng_seed = _stable_seed(f"structural:{profile['name']}:{selection.record.snapshot_id}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(rng_seed)
        for decision in range(AUDIT_STEPS):
            continuous = _profile_action(
                profile=profile,
                decision=decision,
                observation=observation,
                models=models,
            )
            if not info["action_mask"].movement_candidates[0]:
                _raise_integrity(
                    "Explicit stay became illegal",
                    selection,
                    profile["name"],
                    decision,
                    prefix,
                )
            next_observation, _, terminated, truncated, next_info = env.step(
                {"continuous": continuous, "movement_index": 0}
            )
            result = _plain_mapping(next_info["native_result"])
            step = _validated_step(
                selection=selection,
                phase=f"post-step-{decision + 1}",
                observation=next_observation,
                info=next_info,
                result=result,
                continuous=continuous,
                movement=0,
                prior_population=prior_population,
                prefix=prefix,
                context={"profile": profile["name"], "decision": decision + 1},
                model=None,
                terminated=terminated,
                truncated=truncated,
            )
            prefix.append(step)
            state = _mapping(result, "StateDiagnostics")
            if bool(_pick(state, "Active")):
                prior_population = int(_pick(state, "Population"))
            observation = next_observation
            info = next_info
            if terminated or truncated:
                break
    replay = env.verify_replay()
    if not bool(replay.get("ok", True)) or int(replay.get("steps", 0)) != len(prefix):
        _raise_integrity(
            "Native structural replay failed",
            selection,
            profile["name"],
            len(prefix),
            prefix,
            extra={"native_replay": replay},
        )
    return {
        "snapshot_id": selection.record.snapshot_id,
        "family": selection.record.scenario_family,
        "profile": profile["name"],
        "steps": len(prefix),
        "terminated": bool(prefix[-1]["terminated"]),
        "truncated": bool(prefix[-1]["truncated"]),
        "native_replay": replay,
        "trajectory": prefix,
    }


def _profile_action(
    *,
    profile: dict[str, Any],
    decision: int,
    observation: np.ndarray,
    models: dict[int, ElderActorCritic],
) -> np.ndarray:
    if profile["kind"] == "policy":
        model = models[int(profile["seed"])]
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            sampled = model.actor.sample(tensor)
        return sampled.continuous[0].detach().cpu().numpy()
    values = profile["values"][decision % len(profile["values"])]
    return np.asarray((values[0], 0.0, values[1], 1.0), dtype=np.float64)


def _run_counterfactual_seed(
    *,
    seed: int,
    model: ElderActorCritic,
    payload: dict[str, Any],
    records: tuple[SnapshotRecord, ...],
    dataset: CampaignDataset,
    bridge_path: Path,
    dotnet: str,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    pilot_config = PilotConfig(**payload["pilot_config"])
    env = _make_environment(
        bridge_path=bridge_path,
        dotnet=dotnet,
        selection=dataset.selection(records[0]),
        episode_limit=AUDIT_STEPS,
    )
    episodes: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    try:
        for episode_index, record in enumerate(records, start=1):
            selection = dataset.selection(record)
            baseline = _run_policy_baseline(
                env=env,
                selection=selection,
                model=model,
                seed=seed,
            )
            episodes.append(baseline)
            for decision in PROBE_DECISIONS:
                if decision >= len(baseline["trajectory"]):
                    continue
                minus = _run_counterfactual_branch(
                    env=env,
                    selection=selection,
                    model=model,
                    baseline=baseline,
                    decision=decision,
                    delta=-PERTURBATION,
                )
                plus = _run_counterfactual_branch(
                    env=env,
                    selection=selection,
                    model=model,
                    baseline=baseline,
                    decision=decision,
                    delta=PERTURBATION,
                )
                probes.append(
                    _paired_probe(
                        baseline=baseline,
                        minus=minus,
                        plus=plus,
                        decision=decision,
                        horizon_years=pilot_config.horizon_years,
                        reward_scale=pilot_config.reward_scale,
                    )
                )
            if episode_index % 16 == 0 or episode_index == len(records):
                _notify(
                    progress,
                    phase="counterfactual",
                    seed=seed,
                    completed_episodes=episode_index,
                    total_episodes=len(records),
                )
    finally:
        env.close()
    return {
        "seed": seed,
        "checkpoint": {
            "selected_update": int(payload["identities"]["selected_update"]),
            "learner_commit": payload["identities"]["learner_commit"],
            "simulator_commit": payload["identities"]["simulator_commit"],
        },
        "cohort": {
            "episodes": len(episodes),
            "family_counts": dict(
                sorted(Counter(record.scenario_family for record in records).items())
            ),
            "snapshot_ids": [record.snapshot_id for record in records],
        },
        "episodes": episodes,
        "probes": probes,
        "summary": _seed_probe_summary(probes),
    }


def _run_policy_baseline(
    *,
    env: Any,
    selection: CampaignSelection,
    model: ElderActorCritic,
    seed: int,
) -> dict[str, Any]:
    observation, info = env.reset(options=_reset_request(selection, AUDIT_STEPS))
    _validate_campaign_reset(_plain_mapping(info["native_result"]), selection)
    trajectory: list[dict[str, Any]] = []
    _validate_scope_and_observation(
        selection=selection,
        phase="reset",
        observation=observation,
        info=info,
        prefix=trajectory,
        context={"seed": seed, "kind": "baseline"},
    )
    reset_state = _mapping(_plain_mapping(info["native_result"]), "StateDiagnostics")
    prior_population = int(_pick(reset_state, "Population"))
    rng_seed = _stable_seed(f"counterfactual:{seed}:{selection.record.snapshot_id}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(rng_seed)
        for decision in range(AUDIT_STEPS):
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
            with torch.no_grad():
                sampled = model.actor.sample(observation_tensor.unsqueeze(0))
            continuous = sampled.continuous[0].detach().cpu().numpy()
            movement = int(sampled.movement_index[0].detach().cpu())
            next_observation, _, terminated, truncated, next_info = env.step(
                {"continuous": continuous, "movement_index": movement}
            )
            result = _plain_mapping(next_info["native_result"])
            step = _validated_step(
                selection=selection,
                phase=f"baseline-post-step-{decision + 1}",
                observation=next_observation,
                info=next_info,
                result=result,
                continuous=continuous,
                movement=movement,
                prior_population=prior_population,
                prefix=trajectory,
                context={"seed": seed, "kind": "baseline", "decision": decision + 1},
                model=model,
                pre_observation=observation,
                terminated=terminated,
                truncated=truncated,
            )
            trajectory.append(step)
            state = _mapping(result, "StateDiagnostics")
            if bool(_pick(state, "Active")):
                prior_population = int(_pick(state, "Population"))
            observation = next_observation
            info = next_info
            if terminated or truncated:
                break
    replay = env.verify_replay()
    if not bool(replay.get("ok", True)) or int(replay.get("steps", 0)) != len(trajectory):
        _raise_integrity(
            "Native baseline replay failed",
            selection,
            f"policy-seed-{seed}",
            len(trajectory),
            trajectory,
            extra={"native_replay": replay},
        )
    return {
        "snapshot_id": selection.record.snapshot_id,
        "family": selection.record.scenario_family,
        "seed": seed,
        "trajectory": trajectory,
        "native_replay": replay,
    }


def _run_counterfactual_branch(
    *,
    env: Any,
    selection: CampaignSelection,
    model: ElderActorCritic,
    baseline: dict[str, Any],
    decision: int,
    delta: float,
) -> dict[str, Any]:
    observation, info = env.reset(options=_reset_request(selection, AUDIT_STEPS))
    _validate_campaign_reset(_plain_mapping(info["native_result"]), selection)
    trajectory: list[dict[str, Any]] = []
    _validate_scope_and_observation(
        selection=selection,
        phase="reset",
        observation=observation,
        info=info,
        prefix=trajectory,
        context={"seed": baseline["seed"], "kind": "branch", "delta": delta},
    )
    reset_state = _mapping(_plain_mapping(info["native_result"]), "StateDiagnostics")
    prior_population = int(_pick(reset_state, "Population"))
    for index, baseline_step in enumerate(baseline["trajectory"]):
        if index <= decision and _observation_sha256(observation) != baseline_step[
            "pre_observation_sha256"
        ]:
            _raise_integrity(
                "Counterfactual prefix failed deterministic replay",
                selection,
                f"policy-seed-{baseline['seed']}",
                index,
                trajectory,
                extra={"delta": delta, "perturbed_decision": decision + 1},
            )
        continuous = np.asarray(baseline_step["continuous"], dtype=np.float64).copy()
        if index == decision:
            continuous[2] = np.clip(continuous[2] + delta, 0.0, 1.0)
        movement = int(baseline_step["movement_index"])
        next_observation, _, terminated, truncated, next_info = env.step(
            {"continuous": continuous, "movement_index": movement}
        )
        result = _plain_mapping(next_info["native_result"])
        step = _validated_step(
            selection=selection,
            phase=f"branch-{decision + 1}-{delta:+g}-post-step-{index + 1}",
            observation=next_observation,
            info=next_info,
            result=result,
            continuous=continuous,
            movement=movement,
            prior_population=prior_population,
            prefix=trajectory,
            context={
                "seed": baseline["seed"],
                "kind": "branch",
                "delta": delta,
                "perturbed_decision": decision + 1,
            },
            model=model,
            pre_observation=observation,
            terminated=terminated,
            truncated=truncated,
        )
        trajectory.append(step)
        state = _mapping(result, "StateDiagnostics")
        if bool(_pick(state, "Active")):
            prior_population = int(_pick(state, "Population"))
        observation = next_observation
        info = next_info
        if terminated or truncated:
            break
    replay = env.verify_replay()
    if not bool(replay.get("ok", True)) or int(replay.get("steps", 0)) != len(trajectory):
        _raise_integrity(
            "Native counterfactual replay failed",
            selection,
            f"policy-seed-{baseline['seed']}",
            len(trajectory),
            trajectory,
            extra={
                "native_replay": replay,
                "delta": delta,
                "perturbed_decision": decision + 1,
            },
        )
    return {"trajectory": trajectory, "native_replay": replay, "delta": delta}


def _validated_step(
    *,
    selection: CampaignSelection,
    phase: str,
    observation: np.ndarray,
    info: dict[str, Any],
    result: dict[str, Any],
    continuous: np.ndarray,
    movement: int,
    prior_population: int,
    prefix: list[dict[str, Any]],
    context: dict[str, Any],
    model: ElderActorCritic | None,
    pre_observation: np.ndarray | None = None,
    terminated: bool = False,
    truncated: bool = False,
) -> dict[str, Any]:
    violations: list[str] = []
    if not _requested_matches_sample(result["Requested"], continuous, movement):
        violations.append("requested-action-mismatch")
    if int(_pick(result["Requested"], "MovementCandidate")) != -1:
        violations.append("non-stay-request")
    raw = _raw_observation(info)
    scope_violations = cultivation_only_scope_violations(raw)
    violations.extend(f"scope:{value}" for value in scope_violations)
    finite, _, _ = _observation_guard(selection.normalizer, raw)
    if not finite or not np.all(np.isfinite(np.asarray(observation, dtype=np.float64))):
        violations.append("non-finite-observation")
    reward = _mapping(result, "Reward")
    elapsed = float(_pick(result, "ElapsedYears"))
    reward_values = [
        float(_pick(reward, "RealizedQoL")),
        float(_pick(reward, "WelfareRate")),
        float(_pick(reward, "UndiscountedIntervalWelfare")),
        elapsed,
    ]
    if not all(math.isfinite(value) for value in reward_values) or elapsed <= 0:
        violations.append("non-finite-or-nonpositive-reward-time")
    if _reward_denominator_violation(reward, prior_population, elapsed):
        violations.append("reward-denominator-contract")
    if violations:
        _raise_integrity(
            "Five-step transition integrity failure",
            selection,
            str(context.get("profile", context.get("kind", "unknown"))),
            int(context.get("decision", len(prefix) + 1)),
            prefix,
            extra={"phase": phase, "violations": violations, **context},
        )

    post_value = None
    pre_value = None
    if model is not None:
        with torch.no_grad():
            if pre_observation is not None:
                pre_value = float(
                    model.critic(
                        torch.as_tensor(pre_observation, dtype=torch.float32).unsqueeze(0)
                    )[0]
                )
            post_value = 0.0 if terminated else float(
                model.critic(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))[0]
            )
    return {
        "decision": len(prefix) + 1,
        "pre_observation_sha256": (
            _observation_sha256(pre_observation)
            if pre_observation is not None
            else None
        ),
        "post_observation_sha256": _observation_sha256(observation),
        "continuous": [float(value) for value in continuous],
        "movement_index": movement,
        "requested": _action_values(_mapping(result, "Requested")),
        "accepted": _action_values(_mapping(result, "Accepted")),
        "realized": _action_values(_mapping(result, "Realized")),
        "interventions": list(_sequence(result, "Interventions")),
        "elapsed_years": elapsed,
        "reward": _reward_values(reward),
        "state": _state_values(_mapping(result, "StateDiagnostics")),
        "pre_value": pre_value,
        "post_value": post_value,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _validate_scope_and_observation(
    *,
    selection: CampaignSelection,
    phase: str,
    observation: np.ndarray,
    info: dict[str, Any],
    prefix: list[dict[str, Any]],
    context: dict[str, Any],
) -> None:
    raw = _raw_observation(info)
    violations = [f"scope:{value}" for value in cultivation_only_scope_violations(raw)]
    finite, _, _ = _observation_guard(selection.normalizer, raw)
    if not finite or not np.all(np.isfinite(np.asarray(observation, dtype=np.float64))):
        violations.append("non-finite-observation")
    if violations:
        _raise_integrity(
            "Five-step reset/scope integrity failure",
            selection,
            str(context.get("profile", context.get("kind", "unknown"))),
            0,
            prefix,
            extra={"phase": phase, "violations": violations, **context},
        )


def _paired_probe(
    *,
    baseline: dict[str, Any],
    minus: dict[str, Any],
    plus: dict[str, Any],
    decision: int,
    horizon_years: float,
    reward_scale: float,
) -> dict[str, Any]:
    outcomes = {
        "minus": _trajectory_outcome(
            minus["trajectory"], decision, horizon_years, reward_scale
        ),
        "baseline": _trajectory_outcome(
            baseline["trajectory"], decision, horizon_years, reward_scale
        ),
        "plus": _trajectory_outcome(
            plus["trajectory"], decision, horizon_years, reward_scale
        ),
    }
    requested = [
        outcomes[label]["decision_action"]["requested"]["development_budget"]
        for label in ("minus", "baseline", "plus")
    ]
    accepted = [
        outcomes[label]["decision_action"]["accepted"]["development_budget"]
        for label in ("minus", "baseline", "plus")
    ]
    through_three = [
        outcomes[label]["physical_return_through_decision_three"]
        for label in ("minus", "baseline", "plus")
    ]
    through_five = [
        outcomes[label]["physical_return_through_decision_five"]
        for label in ("minus", "baseline", "plus")
    ]
    scale = max(max(abs(value) for value in through_five), 1e-12)
    relative_three = (max(through_three) - min(through_three)) / scale
    relative_five = (max(through_five) - min(through_five)) / scale
    requested_span = max(requested) - min(requested)
    accepted_span = max(accepted) - min(accepted)
    request_eligible = requested_span >= PERTURBATION - 1e-9
    informative = bool(
        request_eligible
        and accepted_span > ACCEPTED_SPAN_FLOOR
        and relative_three <= PHYSICAL_RELATIVE_SPREAD_FLOOR
    )
    return {
        "snapshot_id": baseline["snapshot_id"],
        "family": baseline["family"],
        "decision_one_based": decision + 1,
        "requested_span": requested_span,
        "accepted_span": accepted_span,
        "five_step_physical_scale": scale,
        "three_step_relative_physical_spread": relative_three,
        "five_step_relative_physical_spread": relative_five,
        "request_eligible": request_eligible,
        "informative_delayed_effect_probe": informative,
        "delayed_effect_observed": bool(
            informative and relative_five > PHYSICAL_RELATIVE_SPREAD_FLOOR
        ),
        "outcomes": outcomes,
        "branches": {
            "minus": minus,
            "plus": plus,
        },
    }


def _trajectory_outcome(
    trajectory: list[dict[str, Any]],
    decision: int,
    horizon_years: float,
    reward_scale: float,
) -> dict[str, Any]:
    discount = 1.0
    through_three = 0.0
    through_five = 0.0
    decisions_four_five = 0.0
    for index, step in enumerate(trajectory[decision:], start=decision):
        interval = discounted_interval_reward(
            step["reward"]["welfare_rate"],
            step["elapsed_years"],
            horizon_years,
        )
        through_five += discount * interval
        if index <= 2:
            through_three += discount * interval
        else:
            decisions_four_five += discount * interval
        discount *= physical_gamma(step["elapsed_years"], horizon_years)

    if decision >= len(trajectory):
        raise ValueError("Perturbed decision is absent from the trajectory")
    decision_action = trajectory[decision]
    bellman = _decision_three_bellman(trajectory, horizon_years, reward_scale)
    return {
        "physical_return_through_decision_three": through_three,
        "physical_return_through_decision_five": through_five,
        "physical_return_decisions_four_five": decisions_four_five,
        "decision_action": {
            "requested": decision_action["requested"],
            "accepted": decision_action["accepted"],
            "realized": decision_action["realized"],
            "interventions": decision_action["interventions"],
        },
        "decision_three_bellman": bellman,
    }


def _decision_three_bellman(
    trajectory: list[dict[str, Any]],
    horizon_years: float,
    reward_scale: float,
) -> dict[str, float | None]:
    if len(trajectory) < 3:
        return {
            "critic_continuation_value": None,
            "observed_two_step_plus_bootstrap_target": None,
            "residual": None,
        }
    third = trajectory[2]
    critic_value = float(third["post_value"])
    continuation = 0.0
    discount = 1.0
    for step in trajectory[3:5]:
        physical = discounted_interval_reward(
            step["reward"]["welfare_rate"],
            step["elapsed_years"],
            horizon_years,
        )
        continuation += discount * physical * reward_scale
        discount *= physical_gamma(step["elapsed_years"], horizon_years)
    last = trajectory[-1]
    terminal_value = 0.0 if last["terminated"] else float(last["post_value"])
    continuation += discount * terminal_value
    return {
        "critic_continuation_value": critic_value,
        "observed_two_step_plus_bootstrap_target": continuation,
        "residual": continuation - critic_value,
    }


def _seed_probe_summary(probes: list[dict[str, Any]]) -> dict[str, Any]:
    f03 = [probe for probe in probes if probe["family"] == "F03"]
    eligible = [probe for probe in f03 if probe["request_eligible"]]
    informative = [probe for probe in f03 if probe["informative_delayed_effect_probe"]]
    delayed = [probe for probe in informative if probe["delayed_effect_observed"]]
    responsive_three = [
        probe
        for probe in eligible
        if probe["three_step_relative_physical_spread"]
        > PHYSICAL_RELATIVE_SPREAD_FLOOR
    ]
    responsive_five = [
        probe
        for probe in eligible
        if probe["five_step_relative_physical_spread"]
        > PHYSICAL_RELATIVE_SPREAD_FLOOR
    ]
    rate_three = _fraction(len(responsive_three), len(eligible))
    rate_five = _fraction(len(responsive_five), len(eligible))
    informative_snapshots = {probe["snapshot_id"] for probe in informative}
    informative_gate = bool(
        len(informative) >= INFORMATIVE_PROBE_MINIMUM
        and len(informative_snapshots) >= INFORMATIVE_SNAPSHOT_MINIMUM
    )
    delayed_fraction = _fraction(len(delayed), len(informative))
    responsive_gain = (
        rate_five - rate_three
        if rate_five is not None and rate_three is not None
        else None
    )
    delayed_gate = bool(
        informative_gate
        and delayed_fraction is not None
        and delayed_fraction >= DELAYED_EFFECT_FRACTION_MINIMUM
        and responsive_gain is not None
        and responsive_gain >= RESPONSIVE_RATE_GAIN_MINIMUM
    )
    return {
        "probes": len(probes),
        "f03_probes": len(f03),
        "f03_request_eligible": len(eligible),
        "informative_f03_probes": len(informative),
        "informative_f03_distinct_snapshots": len(informative_snapshots),
        "informative_cohort_gate": informative_gate,
        "delayed_effect_observed": len(delayed),
        "delayed_effect_fraction": delayed_fraction,
        "three_step_f03_responsive_fraction": rate_three,
        "five_step_f03_responsive_fraction": rate_five,
        "f03_responsive_fraction_gain": responsive_gain,
        "delayed_effect_gate": delayed_gate,
    }


def _assess_delayed_effect(
    seed_reports: list[dict[str, Any]],
    structural: dict[str, Any],
) -> dict[str, Any]:
    structural_pass = structural["summary"]["status"] == "PASS"
    informative_seeds = [
        report["seed"]
        for report in seed_reports
        if report["summary"]["informative_cohort_gate"]
    ]
    delayed_seeds = [
        report["seed"]
        for report in seed_reports
        if report["summary"]["delayed_effect_gate"]
    ]
    if not structural_pass:
        status = "STOP_NO_FIVE_STEP_PPO"
    elif len(informative_seeds) < 2:
        status = "INCONCLUSIVE"
    elif len(delayed_seeds) >= 2:
        status = "FIVE_STEP_PPO_ABLATION_JUSTIFIED"
    else:
        status = "DELAY_HYPOTHESIS_NOT_SUPPORTED"
    return {
        "status": status,
        "integrity_and_scope_gate": structural_pass,
        "informative_cohort_seeds": informative_seeds,
        "delayed_effect_support_seeds": delayed_seeds,
        "five_step_ppo_ablation_justified": status
        == "FIVE_STEP_PPO_ABLATION_JUSTIFIED",
        "architecture_change_justified": False,
        "normalizer_change_justified": False,
        "activation_change_justified": False,
        "bootstrap_or_gae_change_justified": False,
        "heldout_remains_sealed": True,
        "development_remains_unread": True,
    }


def _compact_seed_report(report: dict[str, Any]) -> dict[str, Any]:
    residuals: list[float] = []
    for probe in report["probes"]:
        for outcome in probe["outcomes"].values():
            residual = outcome["decision_three_bellman"]["residual"]
            if residual is not None:
                residuals.append(float(residual))
    return {
        "seed": report["seed"],
        "checkpoint": report["checkpoint"],
        "cohort": report["cohort"],
        "probe_summary": report["summary"],
        "decision_three_bellman_residual": _distribution(residuals),
    }


def _audit_identities(
    *,
    repository_root: Path,
    bridge_path: Path,
    dataset: CampaignDataset,
    normalizer_path: Path,
    previous_path: Path,
    checkpoint_dir: Path,
    payloads: dict[int, dict[str, Any]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    learner_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    tracked_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    )
    if tracked_dirty:
        raise ValueError("Five-step audit requires a clean tracked learner source tree")
    return {
        "audit_learner_commit": learner_commit,
        "training_learner_commits": sorted(
            {payloads[seed]["identities"]["learner_commit"] for seed in seeds}
        ),
        "native_simulator_commit": payloads[seeds[0]]["identities"][
            "simulator_commit"
        ],
        "snapshot_manifest_sha256": dataset.manifest_sha256,
        "normalizer_partition_id": dataset.normalizer.manifest.partition_id,
        "normalizer_sha256": sha256_file(normalizer_path)[0],
        "bridge_sha256": sha256_file(bridge_path)[0],
        "previous_action_credit_evidence_sha256": json.loads(
            previous_path.read_text(encoding="utf-8")
        )["evidence_sha256"],
        "checkpoint_sha256": {
            str(seed): sha256_file(checkpoint_dir / f"seed-{seed}.pt")[0]
            for seed in seeds
        },
    }


def _reward_values(reward: dict[str, Any]) -> dict[str, float | int]:
    return {
        "prior_population": int(_pick(reward, "PriorPopulation")),
        "realized_qol": float(_pick(reward, "RealizedQoL")),
        "welfare_rate": float(_pick(reward, "WelfareRate")),
        "undiscounted_interval_welfare": float(
            _pick(reward, "UndiscountedIntervalWelfare")
        ),
    }


def _state_values(state: dict[str, Any]) -> dict[str, float | int | bool]:
    return {
        "active": bool(_pick(state, "Active")),
        "population": int(_pick(state, "Population")),
        "qol": float(_pick(state, "QoL")),
        "field_q": float(_pick(state, "FieldQ")),
        "retained_field_capital": float(_pick(state, "RetainedFieldCapital")),
        "field_operation": float(_pick(state, "FieldOperation")),
        "maintenance_worker_years": float(_pick(state, "MaintenanceWorkerYears")),
        "food_reserve_years": float(_pick(state, "FoodReserveYears")),
        "materials_per_capita": float(_pick(state, "MaterialsPerCapita")),
        "farm_tools_per_capita": float(_pick(state, "FarmToolsPerCapita")),
    }


def _raise_integrity(
    message: str,
    selection: CampaignSelection,
    profile: str,
    decision: int,
    prefix: list[dict[str, Any]],
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    raise AuditIntegrityError(
        message,
        {
            "snapshot_id": selection.record.snapshot_id,
            "family": selection.record.scenario_family,
            "profile": profile,
            "decision": decision,
            "prefix": prefix,
            **(extra or {}),
        },
    )


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**63 - 1)


def _observation_sha256(observation: np.ndarray | None) -> str | None:
    if observation is None:
        return None
    return hashlib.sha256(np.asarray(observation, dtype=np.float32).tobytes()).hexdigest()


def _digest(payload: dict[str, Any]) -> str:
    content = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    return hashlib.sha256(
        json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95_abs": None, "max_abs": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95_abs": float(np.quantile(np.abs(array), 0.95)),
        "max_abs": float(np.max(np.abs(array))),
    }


def _notify(progress: ProgressCallback | None, **payload: Any) -> None:
    if progress is not None:
        progress(payload)
