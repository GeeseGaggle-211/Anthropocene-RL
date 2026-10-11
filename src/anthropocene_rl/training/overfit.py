"""Real one-snapshot PPO overfit and native-throughput gates.

This module deliberately stays on the learner side of the repository boundary.
All state transitions, legal repairs, rewards, and replay checks are delegated to
the pinned C# ``NativeElderEnvironment`` through the persistent bridge.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from anthropocene_rl.contracts import (
    ACTION_SCHEMA_VERSION,
    CRITIC_SIDECAR_SCHEMA_VERSION,
    ENVIRONMENT_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    SIMULATOR_COMMIT,
    TENSOR_FEATURE_COUNT,
)
from anthropocene_rl.datasets.blobs import sha256_file
from anthropocene_rl.datasets.models import (
    Partition,
    SnapshotBankManifest,
    SnapshotRecord,
)
from anthropocene_rl.datasets.object_store import (
    DEFAULT_DATASET_STORE_CONFIG,
    DatasetStoreConfig,
    build_object_store,
)
from anthropocene_rl.datasets.shards import load_dataset_reader
from anthropocene_rl.datasets.storage import (
    DEFAULT_SNAPSHOT_STORE_CONFIG,
    RemoteBlobStore,
    resolve_snapshot_blob,
)
from anthropocene_rl.env.gym_env import AnthropoceneElderEnv
from anthropocene_rl.env.transport import JsonLineProcessTransport
from anthropocene_rl.observations.normalization import (
    NormalizationManifest,
    SemanticNormalizer,
    semantic_normalizer_from_manifest,
)
from anthropocene_rl.observations.schema import EncodedObservation
from anthropocene_rl.policies.distributions import structural_head_mask
from anthropocene_rl.policies.elder_mlp import (
    ElderActorCritic,
    ElderActorCriticConfig,
)
from anthropocene_rl.training.ppo import PPOConfig, PPOTrainer, RolloutBatch
from anthropocene_rl.training.returns import (
    discounted_interval_reward,
    generalized_advantage_estimate,
    physical_gamma,
)

DEFAULT_MANIFEST = Path("datasets/manifests/elder-v2-stage0-v1.json")
DEFAULT_BLOB_ROOT = Path("datasets/blobs")
DEFAULT_NORMALIZER = Path("evidence/stage0-snapshot-bank/provisional-train-normalization.json")
DEFAULT_BRIDGE_DLL = Path("integration/bin/Release/net8.0/Anthropocene.RLBridge.dll")
DEFAULT_CHECKPOINT = Path("checkpoints/stage0-one-snapshot.pt")
DEFAULT_OVERFIT_DIAGNOSTICS = Path("artifacts/stage0-one-snapshot-overfit.json")
DEFAULT_THROUGHPUT_DIAGNOSTICS = Path("artifacts/stage0-native-throughput.json")
DEFAULT_STAGE0_GCS_DATASET_MANIFEST = (
    "gs://anthropocene-rl-datasets-508912/datasets/elder-v2-stage0/gcs-smoke-20261011/manifest.json"
)
DEFAULT_READINESS_DIAGNOSTICS = Path("artifacts/cultivation-readiness.json")
DEFAULT_READINESS_RUN_DIRECTORY = Path("artifacts/cultivation-readiness-runs")
DEFAULT_READINESS_CHECKPOINT_DIRECTORY = Path("checkpoints/cultivation-readiness")
DEFAULT_READINESS_SEEDS = (1701, 1901, 2101)
READINESS_SCENARIO_FAMILIES = ("F01", "F03", "F13")


@dataclass(frozen=True)
class OverfitConfig:
    seed: int = 1701
    device: str = "cpu"
    updates: int = 12
    episodes_per_update: int = 4
    episode_steps: int = 8
    horizon_years: float = 400.0
    lambda_ref: float = 0.95
    lambda_reference_years: float = 20.0
    reward_scale: float = 0.001
    minimum_critic_improvement: float = 0.50
    minimum_mean_critic_improvement: float = 0.40
    maximum_well_fit_critic_mse: float = 0.05
    minimum_actor_shift: float = 0.02
    minimum_actor_direction_consistency: float = 0.50
    maximum_intervention_rate_increase: float = 0.25
    maximum_intervention_count_increase: float = 0.25
    maximum_intervention_distance_increase: float = 0.05
    minimum_flat_return_repair_reduction: float = 0.01
    maximum_post_update_approx_kl: float = 0.15
    maximum_mean_post_update_clip_fraction: float = 0.60
    maximum_saturation_rate: float = 0.05

    def validate(self) -> None:
        if self.updates < 2:
            raise ValueError("The overfit gate requires at least two PPO updates")
        if self.episodes_per_update < 1 or self.episode_steps < 1:
            raise ValueError("Rollout episode counts and lengths must be positive")
        if self.horizon_years <= 0 or not 0 <= self.lambda_ref <= 1:
            raise ValueError("Invalid physical-time credit configuration")
        if self.lambda_reference_years <= 0 or self.reward_scale <= 0:
            raise ValueError("Reward and lambda reference scales must be positive")
        if not 0 <= self.minimum_critic_improvement < 1:
            raise ValueError("minimum_critic_improvement must be in [0,1)")
        if not 0 <= self.minimum_mean_critic_improvement < 1:
            raise ValueError("minimum_mean_critic_improvement must be in [0,1)")
        if self.maximum_well_fit_critic_mse <= 0:
            raise ValueError("maximum_well_fit_critic_mse must be positive")
        if self.minimum_actor_shift <= 0:
            raise ValueError("minimum_actor_shift must be positive")
        if not 0 <= self.minimum_actor_direction_consistency <= 1:
            raise ValueError("Invalid actor direction-consistency threshold")
        if not 0 <= self.maximum_intervention_rate_increase <= 1:
            raise ValueError("Invalid intervention-rate threshold")
        if self.maximum_intervention_count_increase < 0:
            raise ValueError("Invalid intervention-count threshold")
        if self.maximum_intervention_distance_increase < 0:
            raise ValueError("Invalid intervention-distance threshold")
        if self.minimum_flat_return_repair_reduction < 0:
            raise ValueError("Invalid repair-reduction threshold")
        if self.maximum_post_update_approx_kl <= 0:
            raise ValueError("Invalid post-update KL threshold")
        if not 0 <= self.maximum_mean_post_update_clip_fraction <= 1:
            raise ValueError("Invalid mean clip-fraction threshold")
        if not 0 <= self.maximum_saturation_rate <= 1:
            raise ValueError("Invalid saturation-rate threshold")


@dataclass(frozen=True)
class Stage0Selection:
    manifest: SnapshotBankManifest
    manifest_sha256: str
    record: SnapshotRecord
    snapshot_path: Path
    snapshot_bytes: bytes
    normalizer: SemanticNormalizer


@dataclass
class _TimingTotals:
    native_step_seconds: float = 0.0
    bridge_step_seconds: float = 0.0
    policy_inference_seconds: float = 0.0
    ppo_update_seconds: float = 0.0
    collection_seconds: float = 0.0


def load_stage0_selection(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    blob_root: str | Path = DEFAULT_BLOB_ROOT,
    normalizer_path: str | Path = DEFAULT_NORMALIZER,
    snapshot_id: str | None = None,
    remote_store: RemoteBlobStore | None = None,
    snapshot_store_config: str | Path = DEFAULT_SNAPSHOT_STORE_CONFIG,
    dataset_manifest: str | Path | None = None,
    dataset_store_config: str | Path = DEFAULT_DATASET_STORE_CONFIG,
    require_provisional_normalizer: bool = True,
) -> Stage0Selection:
    """Load one leakage-safe TRAIN reset and its frozen provisional normalizer."""

    manifest = SnapshotBankManifest.load(manifest_path)
    manifest_sha256 = manifest.sha256()
    normalization_manifest = NormalizationManifest.load(normalizer_path)
    allowed_partition_ids = {f"{manifest.bank_id}:train:provisional"}
    if not require_provisional_normalizer:
        allowed_partition_ids.add(f"{manifest.bank_id}:train:final")
    if normalization_manifest.partition_id not in allowed_partition_ids:
        requirement = "provisional" if require_provisional_normalizer else "provisional or final"
        raise ValueError(f"The normalizer is not a {requirement} TRAIN artifact")
    if normalization_manifest.partition_manifest_sha256 != manifest_sha256:
        raise ValueError("Normalizer and snapshot-bank manifest identities differ")

    if snapshot_id is None:
        candidates = [
            record
            for record in manifest.records
            if record.partition == Partition.TRAIN
            and record.scenario_family in {"F01", "F02"}
            and "cultivation-only" in record.actual_strata.tags
            and "fishing" not in record.actual_strata.tags
            and not record.actual_strata.owned_domestic_herd
        ]
        if not candidates:
            raise ValueError("No straightforward F01/F02 cultivation-only TRAIN snapshot exists")
        record = sorted(
            candidates,
            key=lambda item: (item.scenario_family != "F01", item.snapshot_id),
        )[0]
    else:
        matches = [record for record in manifest.records if record.snapshot_id == snapshot_id]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one snapshot named {snapshot_id!r}")
        record = matches[0]

    if record.partition != Partition.TRAIN:
        raise ValueError("The one-snapshot overfit gate may load only a TRAIN snapshot")
    if record.actual_strata.structural_masks.movement:
        raise ValueError("The first one-snapshot gate requires movement-disabled state")
    if not record.actual_strata.structural_masks.field_retention:
        raise ValueError("The first one-snapshot gate requires an active cultivation head")
    if record.actual_strata.owned_domestic_herd:
        raise ValueError("The first one-snapshot gate must not use a mixed/herd settlement")
    if record.actual_strata.fishing_ground_count > 0:
        raise ValueError("The first one-snapshot gate must not use a fishing settlement")

    identity_checks = {
        "configuration_hash": (
            normalization_manifest.configuration_hash,
            record.configuration_hash,
        ),
        "build_hash": (normalization_manifest.build_hash, record.native_build_hash),
        "observation_schema_hash": (
            normalization_manifest.observation_schema_hash,
            record.schema_identity.observation_schema_hash,
        ),
    }
    mismatches = {key: values for key, values in identity_checks.items() if values[0] != values[1]}
    if mismatches:
        raise ValueError(f"Snapshot/normalizer identity mismatch: {mismatches}")

    local_path = Path(blob_root) / record.blob_path
    if local_path.is_file() or dataset_manifest is None:
        snapshot_path = resolve_snapshot_blob(
            record,
            blob_root=blob_root,
            remote=remote_store,
            config_path=snapshot_store_config,
            manifest_path=manifest_path,
        )
    else:
        dataset_config = DatasetStoreConfig.load(dataset_store_config)
        reader = load_dataset_reader(
            dataset_manifest,
            store=build_object_store(dataset_config),
            cache_root=dataset_config.cache_root,
            max_cache_bytes=dataset_config.max_cache_bytes,
        )
        if reader.manifest.source_manifest_sha256 != manifest_sha256:
            raise ValueError("GCS dataset and snapshot-bank manifest identities differ")
        snapshot_path = reader.materialize_record(record.snapshot_id)
    return Stage0Selection(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        record=record,
        snapshot_path=snapshot_path,
        snapshot_bytes=snapshot_path.read_bytes(),
        normalizer=semantic_normalizer_from_manifest(normalization_manifest),
    )


def run_one_snapshot_overfit(
    *,
    bridge_dll: str | Path = DEFAULT_BRIDGE_DLL,
    dotnet: str = "dotnet",
    manifest_path: str | Path = DEFAULT_MANIFEST,
    blob_root: str | Path = DEFAULT_BLOB_ROOT,
    normalizer_path: str | Path = DEFAULT_NORMALIZER,
    dataset_manifest: str | Path | None = None,
    dataset_store_config: str | Path = DEFAULT_DATASET_STORE_CONFIG,
    snapshot_id: str | None = None,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    diagnostics_path: str | Path | None = DEFAULT_OVERFIT_DIAGNOSTICS,
    config: OverfitConfig | None = None,
    ppo_config: PPOConfig | None = None,
) -> dict[str, Any]:
    """Run and assert the real movement-disabled Stage-0 overfit gate."""

    config = config or OverfitConfig()
    config.validate()
    ppo_config = ppo_config or PPOConfig(
        learning_rate=1e-3,
        entropy_coefficient=0.0,
        epochs=8,
        minibatch_size=64,
    )
    device = _configure_determinism(config.seed, config.device)
    selection = load_stage0_selection(
        manifest_path=manifest_path,
        blob_root=blob_root,
        normalizer_path=normalizer_path,
        snapshot_id=snapshot_id,
        dataset_manifest=dataset_manifest,
        dataset_store_config=dataset_store_config,
    )
    bridge_path = Path(bridge_dll).resolve()
    if not bridge_path.is_file():
        raise FileNotFoundError(
            f"Native bridge is missing at {bridge_path}; build it against the frozen SHA"
        )

    actor_critic_config = ElderActorCriticConfig(initial_log_std=-1.0)
    model = ElderActorCritic(actor_critic_config).to(device)
    trainer = PPOTrainer(model, ppo_config)
    policy_evaluation_before = _evaluate_deterministic_policy(
        model=model,
        bridge_path=bridge_path,
        dotnet=dotnet,
        selection=selection,
        config=config,
        device=device,
    )
    env = _make_environment(
        bridge_path=bridge_path,
        dotnet=dotnet,
        selection=selection,
        episode_limit=config.episode_steps,
    )

    timings = _TimingTotals()
    transitions = 0
    replay_checks = 0
    illegal_action_count = 0
    non_finite_count = 0
    saturated = 0
    saturation_denominator = 0
    intervention_flags: list[bool] = []
    intervention_counts: list[int] = []
    intervention_kinds: Counter[str] = Counter()
    requested_accepted_distances: list[float] = []
    accepted_realized_distances: list[float] = []
    intervention_updates: list[dict[str, Any]] = []
    requested_accepted_differences = 0
    accepted_realized_differences = 0
    channel_object_failures = 0
    requested_sample_mismatches = 0
    non_stay_requested_count = 0
    action_examples: list[dict[str, Any]] = []
    update_metrics: list[dict[str, float]] = []
    actor_path: list[list[float]] = []
    inactive_head_gradient_max = [0.0] * 4
    reset_identity: dict[str, Any] | None = None
    initial_observation: Tensor | None = None
    training_started = time.perf_counter()

    try:
        for update in range(config.updates):
            collection_started = time.perf_counter()
            observations: list[Tensor] = []
            actions: list[Tensor] = []
            movement_indices: list[Tensor] = []
            log_probabilities: list[Tensor] = []
            values: list[float] = []
            next_values: list[float] = []
            scaled_rewards: list[float] = []
            elapsed_years: list[float] = []
            terminated_flags: list[bool] = []
            truncated_flags: list[bool] = []
            update_intervention_flags: list[bool] = []
            update_intervention_counts: list[int] = []
            update_intervention_kinds: Counter[str] = Counter()
            update_requested_accepted_distances: list[float] = []
            update_accepted_realized_distances: list[float] = []

            for episode in range(config.episodes_per_update):
                observation, info = env.reset(seed=config.seed + episode)
                current_reset_identity = _plain_mapping(info["native_result"])
                _validate_reset_identity(current_reset_identity, selection)
                if reset_identity is None:
                    reset_identity = current_reset_identity
                raw = _raw_observation(info)
                finite, sat, denominator = _observation_guard(
                    selection.normalizer,
                    raw,
                )
                non_finite_count += 0 if finite else 1
                saturated += sat
                saturation_denominator += denominator
                observation_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                )
                if initial_observation is None:
                    initial_observation = observation_tensor.detach().clone()
                    actor_path.append(_deterministic_action(model, initial_observation))

                for _ in range(config.episode_steps):
                    inference_started = _timing_start(device)
                    with torch.no_grad():
                        sampled = model.actor.sample(observation_tensor.unsqueeze(0))
                        value = model.critic(observation_tensor.unsqueeze(0))[0]
                    timings.policy_inference_seconds += _timing_stop(
                        inference_started,
                        device,
                    )
                    continuous = sampled.continuous[0].detach()
                    movement = sampled.movement_index[0].detach()
                    old_log_probability = sampled.log_prob[0].detach()
                    if not _all_finite(
                        continuous,
                        old_log_probability,
                        value,
                    ):
                        non_finite_count += 1

                    movement_value = int(movement.cpu())
                    action_mask = info["action_mask"]
                    if not action_mask.movement_candidates[movement_value]:
                        illegal_action_count += 1
                    action = {
                        "continuous": continuous.cpu().numpy(),
                        "movement_index": movement_value,
                    }
                    bridge_started = time.perf_counter()
                    next_observation, _, terminated, truncated, next_info = env.step(action)
                    timings.bridge_step_seconds += time.perf_counter() - bridge_started
                    result = _plain_mapping(next_info["native_result"])
                    native_seconds = float(_pick(result, "NativeStepSeconds"))
                    timings.native_step_seconds += native_seconds

                    reward_record = _mapping(result, "Reward")
                    welfare_rate = float(_pick(reward_record, "WelfareRate"))
                    elapsed = float(_pick(result, "ElapsedYears"))
                    physical_reward = discounted_interval_reward(
                        welfare_rate,
                        elapsed,
                        config.horizon_years,
                    )
                    scaled_reward = physical_reward * config.reward_scale
                    next_tensor = torch.as_tensor(
                        next_observation,
                        dtype=torch.float32,
                        device=device,
                    )
                    with torch.no_grad():
                        next_value = (
                            0.0 if terminated else float(model.critic(next_tensor.unsqueeze(0))[0])
                        )
                    raw_next = _raw_observation(next_info)
                    finite, sat, denominator = _observation_guard(
                        selection.normalizer,
                        raw_next,
                    )
                    non_finite_count += 0 if finite else 1
                    saturated += sat
                    saturation_denominator += denominator

                    requested = _mapping(result, "Requested")
                    accepted = _mapping(result, "Accepted")
                    realized = _mapping(result, "Realized")
                    if requested is accepted or accepted is realized or requested is realized:
                        channel_object_failures += 1
                    if not _requested_matches_sample(
                        requested,
                        continuous.cpu().numpy(),
                        movement_value,
                    ):
                        requested_sample_mismatches += 1
                    if int(_pick(requested, "MovementCandidate")) != -1:
                        non_stay_requested_count += 1
                    if _action_signature(requested) != _action_signature(accepted):
                        requested_accepted_differences += 1
                    if _action_signature(accepted) != _action_signature(realized):
                        accepted_realized_differences += 1
                    requested_accepted_distance = _action_l2(requested, accepted)
                    accepted_realized_distance = _action_l2(accepted, realized)
                    requested_accepted_distances.append(requested_accepted_distance)
                    accepted_realized_distances.append(accepted_realized_distance)
                    update_requested_accepted_distances.append(requested_accepted_distance)
                    update_accepted_realized_distances.append(accepted_realized_distance)
                    interventions = _sequence(result, "Interventions")
                    intervention_flags.append(bool(interventions))
                    intervention_counts.append(len(interventions))
                    update_intervention_flags.append(bool(interventions))
                    update_intervention_counts.append(len(interventions))
                    for intervention in interventions:
                        if isinstance(intervention, dict):
                            kind = str(intervention.get("Kind", "unknown"))
                            intervention_kinds[kind] += 1
                            update_intervention_kinds[kind] += 1
                    if len(action_examples) < 8:
                        action_examples.append(
                            {
                                "requested": _plain_mapping(requested),
                                "accepted": _plain_mapping(accepted),
                                "realized": _plain_mapping(realized),
                                "interventions": interventions,
                            }
                        )

                    observations.append(observation_tensor.detach())
                    actions.append(continuous)
                    movement_indices.append(movement)
                    log_probabilities.append(old_log_probability)
                    values.append(float(value))
                    next_values.append(next_value)
                    scaled_rewards.append(scaled_reward)
                    elapsed_years.append(elapsed)
                    terminated_flags.append(bool(terminated))
                    truncated_flags.append(bool(truncated))
                    transitions += 1

                    observation_tensor = next_tensor
                    info = next_info
                    if terminated or truncated:
                        break

                replay = env.verify_replay()
                if int(replay.get("steps", 0)) < 1:
                    raise RuntimeError("Native replay verification returned no steps")
                replay_checks += 1

            timings.collection_seconds += time.perf_counter() - collection_started
            advantages, returns = generalized_advantage_estimate(
                np.asarray(scaled_rewards, dtype=np.float64),
                np.asarray(values, dtype=np.float64),
                np.asarray(next_values, dtype=np.float64),
                np.asarray(elapsed_years, dtype=np.float64),
                np.asarray(terminated_flags, dtype=bool),
                np.asarray(truncated_flags, dtype=bool),
                horizon_years=config.horizon_years,
                lambda_ref=config.lambda_ref,
                lambda_reference_years=config.lambda_reference_years,
            )
            batch = RolloutBatch(
                observations=torch.stack(observations),
                continuous_actions=torch.stack(actions),
                movement_indices=torch.stack(movement_indices).to(torch.long),
                old_log_prob=torch.stack(log_probabilities),
                advantages=torch.as_tensor(
                    advantages,
                    dtype=torch.float32,
                    device=device,
                ),
                returns=torch.as_tensor(
                    returns,
                    dtype=torch.float32,
                    device=device,
                ),
            )
            pre_update_mse = _value_mse(model, batch)
            update_started = _timing_start(device)
            metrics = trainer.update(batch)
            timings.ppo_update_seconds += _timing_stop(update_started, device)
            post_update_mse = _value_mse(model, batch)
            metrics.update(
                {
                    "update": float(update),
                    "pre_update_value_mse": pre_update_mse,
                    "post_update_value_mse": post_update_mse,
                }
            )
            update_metrics.append(metrics)
            intervention_updates.append(
                {
                    "update": update,
                    "transition_rate": float(np.mean(update_intervention_flags)),
                    "mean_count_per_transition": float(np.mean(update_intervention_counts)),
                    "requested_to_accepted_mean_l2": float(
                        np.mean(update_requested_accepted_distances)
                    ),
                    "accepted_to_realized_mean_l2": float(
                        np.mean(update_accepted_realized_distances)
                    ),
                    "kinds": dict(sorted(update_intervention_kinds.items())),
                }
            )
            for head in range(4):
                inactive_head_gradient_max[head] = max(
                    inactive_head_gradient_max[head],
                    metrics[f"continuous_head_{head}_grad_max"],
                )
            if initial_observation is None:
                raise AssertionError("The rollout did not produce an observation")
            actor_path.append(_deterministic_action(model, initial_observation))
            if any(not math.isfinite(value) for value in metrics.values()):
                non_finite_count += 1
            if any(not torch.all(torch.isfinite(parameter)) for parameter in model.parameters()):
                non_finite_count += 1
    finally:
        env.close()

    training_seconds = time.perf_counter() - training_started
    policy_evaluation_after = _evaluate_deterministic_policy(
        model=model,
        bridge_path=bridge_path,
        dotnet=dotnet,
        selection=selection,
        config=config,
        device=device,
    )
    if not update_metrics or initial_observation is None:
        raise RuntimeError("The overfit gate collected no training batches")
    pre_update_mse = [metric["pre_update_value_mse"] for metric in update_metrics]
    post_update_mse = [metric["post_update_value_mse"] for metric in update_metrics]
    aggregate_critic_improvement = 1.0 - sum(post_update_mse) / max(
        sum(pre_update_mse),
        1e-12,
    )
    post_fit_progress = 1.0 - post_update_mse[-1] / max(post_update_mse[0], 1e-12)
    per_update_critic_improvement = [
        1.0 - post / max(pre, 1e-12)
        for pre, post in zip(pre_update_mse, post_update_mse, strict=True)
    ]
    mean_critic_improvement = float(np.mean(per_update_critic_improvement))
    final_critic_improvement = per_update_critic_improvement[-1]
    active_heads = [
        index
        for index, active in enumerate(
            structural_head_mask(initial_observation.unsqueeze(0))[0].tolist()
        )
        if active
    ]
    inactive_heads = [index for index in range(4) if index not in active_heads]
    actor_shift, direction_consistency = _actor_progress(actor_path, active_heads)
    inactive_gradient = max(
        (inactive_head_gradient_max[index] for index in inactive_heads),
        default=0.0,
    )
    early_rate, late_rate = _edge_rates(intervention_flags)
    early_count, late_count = _edge_means(intervention_counts)
    early_requested_accepted_l2, late_requested_accepted_l2 = _edge_means(
        requested_accepted_distances
    )
    early_accepted_realized_l2, late_accepted_realized_l2 = _edge_means(accepted_realized_distances)
    policy_return_improvement = (
        policy_evaluation_after["discounted_scaled_return"]
        - policy_evaluation_before["discounted_scaled_return"]
    )
    policy_repair_reduction = (
        policy_evaluation_before["requested_to_accepted_mean_l2"]
        - policy_evaluation_after["requested_to_accepted_mean_l2"]
    )
    evaluation_action_shift = _mean_action_path_shift(
        policy_evaluation_before["action_path"],
        policy_evaluation_after["action_path"],
        active_heads,
    )
    effective_actor_shift = max(actor_shift, evaluation_action_shift)
    flat_policy_return = math.isclose(
        policy_return_improvement,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    maximum_post_update_kl = max(metric["post_update_approx_kl"] for metric in update_metrics)
    mean_post_update_clip_fraction = float(
        np.mean([metric["post_update_clip_fraction"] for metric in update_metrics])
    )
    saturation_rate = saturated / max(saturation_denominator, 1)

    identities = _checkpoint_identities(selection, reset_identity, bridge_path)
    parity = _save_and_verify_checkpoint(
        path=Path(checkpoint_path),
        model=model,
        trainer=trainer,
        actor_critic_config=actor_critic_config,
        ppo_config=ppo_config,
        overfit_config=config,
        normalizer=selection.normalizer,
        identities=identities,
        observation=initial_observation,
        continuation_batch=batch,
        device=device,
    )
    gates = {
        "critic_return_fit": (
            mean_critic_improvement >= config.minimum_mean_critic_improvement
            and (
                final_critic_improvement >= config.minimum_critic_improvement
                or post_update_mse[-1] <= config.maximum_well_fit_critic_mse
            )
            and post_fit_progress >= config.minimum_critic_improvement
        ),
        "actor_moved_consistently": (
            effective_actor_shift >= config.minimum_actor_shift
            and direction_consistency >= config.minimum_actor_direction_consistency
        ),
        "inactive_heads_gradient_inert": inactive_gradient <= 1e-12,
        "no_illegal_action_scored": illegal_action_count == 0,
        "action_channels_preserved": (
            channel_object_failures == 0
            and requested_sample_mismatches == 0
            and requested_accepted_differences > 0
        ),
        "compiler_interventions_not_pathological": (
            late_rate <= early_rate + config.maximum_intervention_rate_increase
            and late_count <= early_count + config.maximum_intervention_count_increase
            and late_requested_accepted_l2
            <= early_requested_accepted_l2 + config.maximum_intervention_distance_increase
        ),
        "finite_and_saturation_guards": (
            non_finite_count == 0 and saturation_rate <= config.maximum_saturation_rate
        ),
        "checkpoint_reload_policy_parity": parity["max_abs_error"] == 0.0,
        "checkpoint_continuation_parity": (
            parity["stochastic_policy_max_abs_error"] == 0.0
            and parity["continuation_model_max_abs_error"] == 0.0
            and parity["continuation_metrics_max_abs_error"] == 0.0
        ),
        "native_replay_valid": replay_checks == config.updates * config.episodes_per_update,
        "movement_disabled": non_stay_requested_count == 0,
        "fixed_policy_outcome_improves": (
            policy_return_improvement > 1e-9
            or (
                flat_policy_return
                and policy_repair_reduction >= config.minimum_flat_return_repair_reduction
            )
        ),
        "fixed_policy_evaluation_contract": (
            policy_evaluation_before["contract_valid"] and policy_evaluation_after["contract_valid"]
        ),
        "ppo_update_stability": (
            maximum_post_update_kl <= config.maximum_post_update_approx_kl
            and mean_post_update_clip_fraction <= config.maximum_mean_post_update_clip_fraction
        ),
    }
    diagnostics: dict[str, Any] = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "mechanics-overfit-only-not-gameplay-performance",
        "identities": identities,
        "configuration": {
            "overfit": asdict(config),
            "ppo": asdict(ppo_config),
            "actor_critic": asdict(actor_critic_config),
        },
        "gates": gates,
        "critic": {
            "mean_pre_update_return_mse": float(np.mean(pre_update_mse)),
            "mean_post_update_return_mse": float(np.mean(post_update_mse)),
            "aggregate_fit_improvement_fraction": aggregate_critic_improvement,
            "first_post_update_return_mse": post_update_mse[0],
            "final_post_update_return_mse": post_update_mse[-1],
            "post_fit_progress_fraction": post_fit_progress,
            "per_update_fit_improvement_fractions": per_update_critic_improvement,
            "mean_per_update_fit_improvement_fraction": mean_critic_improvement,
            "final_update_fit_improvement_fraction": final_critic_improvement,
            "required_improvement_fraction": config.minimum_critic_improvement,
            "required_mean_improvement_fraction": (config.minimum_mean_critic_improvement),
            "well_fit_mse_threshold": config.maximum_well_fit_critic_mse,
        },
        "actor": {
            "active_heads": active_heads,
            "inactive_heads": inactive_heads,
            "deterministic_action_path": actor_path,
            "active_action_shift_l2": actor_shift,
            "evaluation_trajectory_mean_action_shift_l2": (evaluation_action_shift),
            "effective_action_shift_l2": effective_actor_shift,
            "direction_consistency": direction_consistency,
            "inactive_head_gradient_max": inactive_gradient,
            "per_head_gradient_max": inactive_head_gradient_max,
        },
        "fixed_policy_evaluation": {
            "before": policy_evaluation_before,
            "after": policy_evaluation_after,
            "discounted_scaled_return_improvement": policy_return_improvement,
            "requested_to_accepted_l2_reduction": policy_repair_reduction,
            "flat_return_repair_reduction_required": (config.minimum_flat_return_repair_reduction),
        },
        "ppo_stability": {
            "maximum_post_update_approx_kl": maximum_post_update_kl,
            "allowed_maximum_post_update_approx_kl": (config.maximum_post_update_approx_kl),
            "mean_post_update_clip_fraction": mean_post_update_clip_fraction,
            "allowed_mean_post_update_clip_fraction": (
                config.maximum_mean_post_update_clip_fraction
            ),
            "target_kl_early_stop_count": int(
                sum(metric["target_kl_early_stop"] for metric in update_metrics)
            ),
        },
        "actions": {
            "transition_count": transitions,
            "illegal_action_count": illegal_action_count,
            "requested_sample_mismatches": requested_sample_mismatches,
            "non_stay_requested_count": non_stay_requested_count,
            "channel_object_failures": channel_object_failures,
            "requested_accepted_difference_count": requested_accepted_differences,
            "accepted_realized_difference_count": accepted_realized_differences,
            "examples": action_examples,
        },
        "interventions": {
            "early_transition_rate": early_rate,
            "late_transition_rate": late_rate,
            "early_count_per_transition": early_count,
            "late_count_per_transition": late_count,
            "mean_count_per_transition": float(np.mean(intervention_counts)),
            "requested_to_accepted_mean_l2": float(np.mean(requested_accepted_distances)),
            "requested_to_accepted_early_mean_l2": early_requested_accepted_l2,
            "requested_to_accepted_late_mean_l2": late_requested_accepted_l2,
            "accepted_to_realized_mean_l2": float(np.mean(accepted_realized_distances)),
            "accepted_to_realized_early_mean_l2": early_accepted_realized_l2,
            "accepted_to_realized_late_mean_l2": late_accepted_realized_l2,
            "kinds": dict(sorted(intervention_kinds.items())),
            "per_update": intervention_updates,
        },
        "numerics": {
            "non_finite_guard_count": non_finite_count,
            "normalizer_saturation_count": saturated,
            "normalizer_saturation_denominator": saturation_denominator,
            "normalizer_saturation_rate": saturation_rate,
        },
        "replay": {
            "verified_episodes": replay_checks,
            "expected_episodes": config.updates * config.episodes_per_update,
        },
        "checkpoint": {
            "path": str(Path(checkpoint_path)),
            **parity,
        },
        "timing": _timing_report(timings, transitions, training_seconds),
        "updates": update_metrics,
    }
    deterministic_payload = {
        key: value for key, value in diagnostics.items() if key not in {"timing", "checkpoint"}
    }
    diagnostics["deterministic_diagnostics_sha256"] = hashlib.sha256(
        json.dumps(
            deterministic_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if diagnostics_path is not None:
        _write_json(Path(diagnostics_path), diagnostics)
    return diagnostics


def run_cultivation_readiness_matrix(
    *,
    bridge_dll: str | Path = DEFAULT_BRIDGE_DLL,
    dotnet: str = "dotnet",
    manifest_path: str | Path = DEFAULT_MANIFEST,
    blob_root: str | Path = DEFAULT_BLOB_ROOT,
    normalizer_path: str | Path = DEFAULT_NORMALIZER,
    dataset_manifest: str | Path | None = None,
    dataset_store_config: str | Path = DEFAULT_DATASET_STORE_CONFIG,
    snapshot_ids: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = DEFAULT_READINESS_SEEDS,
    run_directory: str | Path = DEFAULT_READINESS_RUN_DIRECTORY,
    checkpoint_directory: str | Path = DEFAULT_READINESS_CHECKPOINT_DIRECTORY,
    diagnostics_path: str | Path | None = DEFAULT_READINESS_DIAGNOSTICS,
    config: OverfitConfig | None = None,
    ppo_config: PPOConfig | None = None,
) -> dict[str, Any]:
    """Replicate the fixed-capability PPO gate across snapshots and neural seeds."""

    if len(set(seeds)) < 3:
        raise ValueError("Cultivation readiness requires at least three distinct seeds")
    selected_ids = snapshot_ids or _default_readiness_snapshot_ids(manifest_path)
    if len(set(selected_ids)) < 3:
        raise ValueError("Cultivation readiness requires at least three snapshots")
    base_config = config or OverfitConfig(updates=24)
    base_config.validate()
    ppo_config = ppo_config or PPOConfig(
        learning_rate=1e-3,
        entropy_coefficient=0.0,
        epochs=8,
        minibatch_size=64,
        target_kl=0.05,
    )
    run_root = Path(run_directory)
    checkpoint_root = Path(checkpoint_directory)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for snapshot_id in selected_ids:
        selection = load_stage0_selection(
            manifest_path=manifest_path,
            blob_root=blob_root,
            normalizer_path=normalizer_path,
            snapshot_id=snapshot_id,
            dataset_manifest=dataset_manifest,
            dataset_store_config=dataset_store_config,
        )
        for seed in seeds:
            run_name = f"{selection.record.scenario_family.lower()}-seed-{seed}"
            result = run_one_snapshot_overfit(
                bridge_dll=bridge_dll,
                dotnet=dotnet,
                manifest_path=manifest_path,
                blob_root=blob_root,
                normalizer_path=normalizer_path,
                snapshot_id=snapshot_id,
                dataset_manifest=dataset_manifest,
                dataset_store_config=dataset_store_config,
                checkpoint_path=checkpoint_root / f"{run_name}.pt",
                diagnostics_path=run_root / f"{run_name}.json",
                config=replace(base_config, seed=seed),
                ppo_config=ppo_config,
            )
            rows.append(_readiness_row(result, seed))

    gates = _readiness_matrix_gates(rows, selected_ids, seeds)
    critic_improvements = [row["critic_mean_batch_improvement"] for row in rows]
    critic_progress = [row["critic_fresh_target_progress"] for row in rows]
    reward_improvements = [row["fixed_policy_return_improvement"] for row in rows]
    repair_reductions = [row["fixed_policy_repair_reduction"] for row in rows]
    elapsed = time.perf_counter() - started
    status = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "status": status,
        "scope": "replicated-cultivation-readiness-not-gameplay-performance",
        "capabilities": {
            "cultivation": True,
            "movement": False,
            "herding": False,
            "fishing": False,
        },
        "matrix": {
            "snapshot_ids": list(selected_ids),
            "seeds": list(seeds),
            "run_count": len(rows),
            "runs": rows,
        },
        "aggregate_convergence": {
            "critic_mean_batch_improvement_min": min(critic_improvements),
            "critic_mean_batch_improvement_mean": float(np.mean(critic_improvements)),
            "critic_fresh_target_progress_min": min(critic_progress),
            "critic_fresh_target_progress_mean": float(np.mean(critic_progress)),
            "positive_fixed_policy_return_runs": sum(value > 1e-9 for value in reward_improvements),
            "flat_fixed_policy_return_runs": sum(
                math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-9) for value in reward_improvements
            ),
            "negative_fixed_policy_return_runs": sum(
                value < -1e-9 for value in reward_improvements
            ),
            "fixed_policy_return_improvement_mean": float(np.mean(reward_improvements)),
            "fixed_policy_repair_reduction_mean": float(np.mean(repair_reductions)),
            "maximum_post_update_approx_kl": max(
                row["maximum_post_update_approx_kl"] for row in rows
            ),
            "mean_post_update_clip_fraction": float(
                np.mean([row["mean_post_update_clip_fraction"] for row in rows])
            ),
        },
        "gates": gates,
        "architecture_recommendation": (
            "retain-separate-feedforward-mlp"
            if status == "PASS"
            else "do-not-escalate-architecture-until-failed-gates-are-diagnosed"
        ),
        "elapsed_seconds": elapsed,
        "configuration": {
            "overfit": asdict(base_config),
            "ppo": asdict(ppo_config),
        },
    }
    deterministic_payload = {
        key: value for key, value in result.items() if key not in {"elapsed_seconds"}
    }
    result["deterministic_diagnostics_sha256"] = hashlib.sha256(
        json.dumps(
            deterministic_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if diagnostics_path is not None:
        _write_json(Path(diagnostics_path), result)
    return result


def benchmark_native_throughput(
    *,
    bridge_dll: str | Path = DEFAULT_BRIDGE_DLL,
    dotnet: str = "dotnet",
    manifest_path: str | Path = DEFAULT_MANIFEST,
    blob_root: str | Path = DEFAULT_BLOB_ROOT,
    normalizer_path: str | Path = DEFAULT_NORMALIZER,
    dataset_manifest: str | Path | None = None,
    dataset_store_config: str | Path = DEFAULT_DATASET_STORE_CONFIG,
    snapshot_id: str | None = None,
    checkpoint_path: str | Path | None = None,
    worker_counts: tuple[int, ...] = (1, 2, 4, 8),
    transitions_per_environment: int = 8,
    seed: int = 1701,
    device_name: str = "cpu",
    diagnostics_path: str | Path | None = DEFAULT_THROUGHPUT_DIAGNOSTICS,
) -> dict[str, Any]:
    """Measure fixed-semantics throughput across parallel bridge processes."""

    if not worker_counts or any(count < 1 for count in worker_counts):
        raise ValueError("Worker counts must be positive")
    if transitions_per_environment < 1:
        raise ValueError("transitions_per_environment must be positive")
    device = _configure_determinism(seed, device_name)
    selection = load_stage0_selection(
        manifest_path=manifest_path,
        blob_root=blob_root,
        normalizer_path=normalizer_path,
        snapshot_id=snapshot_id,
        dataset_manifest=dataset_manifest,
        dataset_store_config=dataset_store_config,
        require_provisional_normalizer=False,
    )
    bridge_path = Path(bridge_dll).resolve()
    if not bridge_path.is_file():
        raise FileNotFoundError(f"Native bridge is missing at {bridge_path}")

    actor_config = ElderActorCriticConfig(initial_log_std=-1.0)
    checkpoint_identity: dict[str, Any] | None = None
    checkpoint_payload: dict[str, Any] | None = None
    if checkpoint_path is not None:
        checkpoint_payload = torch.load(
            Path(checkpoint_path),
            map_location=device,
            weights_only=False,
        )
        actor_config = ElderActorCriticConfig(**checkpoint_payload["actor_critic_config"])
        checkpoint_identity = _validate_benchmark_checkpoint(
            checkpoint_payload,
            selection,
            bridge_path,
        )
    model = ElderActorCritic(actor_config).to(device)
    if checkpoint_payload is not None:
        model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()
    inference_lock = threading.Lock()
    rows: list[dict[str, Any]] = []

    for workers in worker_counts:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _benchmark_worker,
                    worker=index,
                    model=model,
                    inference_lock=inference_lock,
                    device=device,
                    bridge_path=bridge_path,
                    dotnet=dotnet,
                    selection=selection,
                    transitions=transitions_per_environment,
                    seed=seed,
                )
                for index in range(workers)
            ]
            worker_results = [future.result() for future in futures]
        wall_seconds = time.perf_counter() - started
        transition_count = sum(int(result["transitions"]) for result in worker_results)
        native_seconds = sum(float(result["native_step_seconds"]) for result in worker_results)
        inference_seconds = sum(
            float(result["policy_inference_seconds"]) for result in worker_results
        )
        bridge_seconds = sum(float(result["bridge_step_seconds"]) for result in worker_results)
        replay_checks = sum(int(result["replay_checks"]) for result in worker_results)
        illegal_actions = sum(int(result["illegal_action_count"]) for result in worker_results)
        rows.append(
            {
                "parallel_environments": workers,
                "transitions": transition_count,
                "transitions_per_environment": transitions_per_environment,
                "native_step_seconds_total": native_seconds,
                "native_step_seconds_mean": native_seconds / transition_count,
                "native_step_transitions_per_second_serial_equivalent": (
                    transition_count / native_seconds
                ),
                "policy_inference_seconds_total": inference_seconds,
                "policy_inference_transitions_per_second": (transition_count / inference_seconds),
                "bridge_step_seconds_total": bridge_seconds,
                "end_to_end_seconds": wall_seconds,
                "end_to_end_transitions_per_second": transition_count / wall_seconds,
                "replay_checks": replay_checks,
                "illegal_action_count": illegal_actions,
            }
        )

    baseline = rows[0]["end_to_end_transitions_per_second"]
    for row in rows:
        row["end_to_end_speedup_vs_first"] = row["end_to_end_transitions_per_second"] / baseline
    result: dict[str, Any] = {
        "status": "PASS" if all(row["illegal_action_count"] == 0 for row in rows) else "FAIL",
        "scope": "local-native-throughput-not-gameplay-performance",
        "identities": {
            "simulator_commit": SIMULATOR_COMMIT,
            "snapshot_manifest_sha256": selection.manifest_sha256,
            "snapshot_id": selection.record.snapshot_id,
            "snapshot_content_sha256": selection.record.content_sha256,
            "checkpoint": checkpoint_identity,
        },
        "device": str(device),
        "rows": rows,
    }
    deterministic_payload = {
        "identities": result["identities"],
        "device": result["device"],
        "worker_counts": list(worker_counts),
        "transitions_per_environment": transitions_per_environment,
        "status": result["status"],
    }
    result["deterministic_diagnostics_sha256"] = hashlib.sha256(
        json.dumps(
            deterministic_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if diagnostics_path is not None:
        _write_json(Path(diagnostics_path), result)
    return result


def _default_readiness_snapshot_ids(
    manifest_path: str | Path,
) -> tuple[str, ...]:
    manifest = SnapshotBankManifest.load(manifest_path)
    selected: list[SnapshotRecord] = []
    for family in READINESS_SCENARIO_FAMILIES:
        candidates = [
            record
            for record in manifest.records
            if record.partition == Partition.TRAIN
            and record.scenario_family == family
            and "cultivation-only" in record.actual_strata.tags
            and record.actual_strata.structural_masks.field_retention
            and record.actual_strata.structural_masks.field_development
            and not record.actual_strata.structural_masks.movement
            and not record.actual_strata.owned_domestic_herd
            and record.actual_strata.fishing_ground_count == 0
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one cultivation-only TRAIN snapshot for {family}; "
                f"found {len(candidates)}"
            )
        selected.append(candidates[0])
    if len({record.world_seed for record in selected}) != len(selected):
        raise ValueError("Readiness snapshots must use distinct native world seeds")
    return tuple(record.snapshot_id for record in selected)


def _readiness_row(result: dict[str, Any], seed: int) -> dict[str, Any]:
    identities = result["identities"]
    gates = result["gates"]
    return {
        "snapshot_id": identities["snapshot_id"],
        "scenario_family": identities["scenario_family"],
        "seed": seed,
        "status": result["status"],
        "gates": gates,
        "critic_mean_batch_improvement": result["critic"][
            "mean_per_update_fit_improvement_fraction"
        ],
        "critic_fresh_target_progress": result["critic"]["post_fit_progress_fraction"],
        "actor_shift_l2": result["actor"]["effective_action_shift_l2"],
        "actor_direction_consistency": result["actor"]["direction_consistency"],
        "fixed_policy_return_improvement": result["fixed_policy_evaluation"][
            "discounted_scaled_return_improvement"
        ],
        "fixed_policy_repair_reduction": result["fixed_policy_evaluation"][
            "requested_to_accepted_l2_reduction"
        ],
        "maximum_post_update_approx_kl": result["ppo_stability"]["maximum_post_update_approx_kl"],
        "mean_post_update_clip_fraction": result["ppo_stability"]["mean_post_update_clip_fraction"],
        "diagnostics_sha256": result["deterministic_diagnostics_sha256"],
    }


def _readiness_matrix_gates(
    rows: list[dict[str, Any]],
    snapshot_ids: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, bool]:
    hard_gate_names = (
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
    )
    required_seed_passes = math.ceil(2 * len(seeds) / 3)
    rows_by_snapshot = {
        snapshot_id: [row for row in rows if row["snapshot_id"] == snapshot_id]
        for snapshot_id in snapshot_ids
    }
    return {
        "complete_three_by_three_matrix": (
            len(snapshot_ids) >= 3
            and len(set(seeds)) >= 3
            and len(rows) == len(snapshot_ids) * len(seeds)
        ),
        "all_contract_and_safety_gates": all(
            all(row["gates"][name] for name in hard_gate_names) for row in rows
        ),
        "all_on_policy_critic_fits_converge": all(
            row["gates"]["critic_return_fit"] for row in rows
        ),
        "all_ppo_updates_stable": all(row["gates"]["ppo_update_stability"] for row in rows),
        "actor_motion_replicates_per_snapshot": all(
            sum(row["gates"]["actor_moved_consistently"] for row in snapshot_rows)
            >= required_seed_passes
            for snapshot_rows in rows_by_snapshot.values()
        ),
        "fixed_policy_outcomes_do_not_regress": all(
            row["fixed_policy_return_improvement"] >= -1e-9 for row in rows
        ),
        "positive_return_signal_replicates": any(
            sum(row["fixed_policy_return_improvement"] > 1e-9 for row in snapshot_rows)
            >= required_seed_passes
            for snapshot_rows in rows_by_snapshot.values()
        ),
    }


def _make_environment(
    *,
    bridge_path: Path,
    dotnet: str,
    selection: Stage0Selection,
    episode_limit: int,
) -> AnthropoceneElderEnv:
    transport = JsonLineProcessTransport([dotnet, str(bridge_path)])
    record = selection.record
    request = {
        "Seed": record.world_seed,
        "RandomStream": 0,
        "Snapshot": selection.snapshot_bytes,
        "ControlledCampId": record.controlled.camp_id,
        "ControlledPlayer": record.controlled.player_id,
        "BackgroundPolicy": 0,
        "Options": {
            "EnvironmentSchemaVersion": ENVIRONMENT_SCHEMA_VERSION,
            "EpisodeDecisionLimit": episode_limit,
            "MovementEnabled": False,
            "SyntheticShocksEnabled": False,
            "IncludeTrainingCriticSidecar": False,
        },
        "ExpectedSimulatorVersion": record.native_simulator_version,
        "ExpectedConfigurationHash": record.configuration_hash,
        "ExpectedObservationSchemaVersion": OBSERVATION_SCHEMA_VERSION,
        "ExpectedActionSchemaVersion": ACTION_SCHEMA_VERSION,
        "ExpectedEnvironmentSchemaVersion": ENVIRONMENT_SCHEMA_VERSION,
    }
    return AnthropoceneElderEnv(
        transport,
        request,
        normalizer=selection.normalizer,
    )


def _evaluate_deterministic_policy(
    *,
    model: ElderActorCritic,
    bridge_path: Path,
    dotnet: str,
    selection: Stage0Selection,
    config: OverfitConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one reproducible native episode without changing learner RNG."""

    env = _make_environment(
        bridge_path=bridge_path,
        dotnet=dotnet,
        selection=selection,
        episode_limit=config.episode_steps,
    )
    discounted_return = 0.0
    discount = 1.0
    undiscounted_interval_welfare = 0.0
    elapsed_total = 0.0
    intervention_counts: list[int] = []
    intervention_kinds: Counter[str] = Counter()
    requested_accepted_distances: list[float] = []
    accepted_realized_distances: list[float] = []
    requested_sample_mismatches = 0
    illegal_actions = 0
    non_stay_requests = 0
    non_finite_observations = 0
    action_path: list[list[float]] = []
    steps = 0
    replay_valid = False
    was_training = model.training
    model.eval()
    try:
        observation, info = env.reset(seed=config.seed)
        _validate_reset_identity(_plain_mapping(info["native_result"]), selection)
        for _ in range(config.episode_steps):
            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                sampled = model.actor.sample(
                    observation_tensor.unsqueeze(0),
                    deterministic=True,
                )
            continuous = sampled.continuous[0].detach()
            movement = int(sampled.movement_index[0].detach().cpu())
            if not info["action_mask"].movement_candidates[movement]:
                illegal_actions += 1
            action_path.append([float(value) for value in continuous.cpu()])
            observation, _, terminated, truncated, info = env.step(
                {
                    "continuous": continuous.cpu().numpy(),
                    "movement_index": movement,
                }
            )
            result = _plain_mapping(info["native_result"])
            requested = _mapping(result, "Requested")
            accepted = _mapping(result, "Accepted")
            realized = _mapping(result, "Realized")
            if not _requested_matches_sample(
                requested,
                continuous.cpu().numpy(),
                movement,
            ):
                requested_sample_mismatches += 1
            if int(_pick(requested, "MovementCandidate")) != -1:
                non_stay_requests += 1
            requested_accepted_distances.append(_action_l2(requested, accepted))
            accepted_realized_distances.append(_action_l2(accepted, realized))
            interventions = _sequence(result, "Interventions")
            intervention_counts.append(len(interventions))
            for intervention in interventions:
                if isinstance(intervention, dict):
                    intervention_kinds[str(intervention.get("Kind", "unknown"))] += 1

            reward = _mapping(result, "Reward")
            welfare_rate = float(_pick(reward, "WelfareRate"))
            elapsed = float(_pick(result, "ElapsedYears"))
            interval_welfare = discounted_interval_reward(
                welfare_rate,
                elapsed,
                config.horizon_years,
            )
            discounted_return += discount * interval_welfare * config.reward_scale
            undiscounted_interval_welfare += float(_pick(reward, "UndiscountedIntervalWelfare"))
            discount *= physical_gamma(elapsed, config.horizon_years)
            elapsed_total += elapsed
            raw = _raw_observation(info)
            finite, _, _ = _observation_guard(selection.normalizer, raw)
            non_finite_observations += 0 if finite else 1
            steps += 1
            if terminated or truncated:
                break
        replay = env.verify_replay()
        replay_valid = int(replay.get("steps", 0)) == steps and steps > 0
    finally:
        env.close()
        model.train(was_training)

    contract_valid = (
        replay_valid
        and illegal_actions == 0
        and requested_sample_mismatches == 0
        and non_stay_requests == 0
        and non_finite_observations == 0
    )
    return {
        "discounted_scaled_return": discounted_return,
        "undiscounted_interval_welfare": undiscounted_interval_welfare,
        "elapsed_years": elapsed_total,
        "steps": steps,
        "action_path": action_path,
        "intervention_transition_rate": float(
            np.mean([count > 0 for count in intervention_counts])
        ),
        "intervention_mean_count": float(np.mean(intervention_counts)),
        "intervention_kinds": dict(sorted(intervention_kinds.items())),
        "requested_to_accepted_mean_l2": float(np.mean(requested_accepted_distances)),
        "accepted_to_realized_mean_l2": float(np.mean(accepted_realized_distances)),
        "illegal_action_count": illegal_actions,
        "requested_sample_mismatches": requested_sample_mismatches,
        "non_stay_requested_count": non_stay_requests,
        "non_finite_observation_count": non_finite_observations,
        "replay_valid": replay_valid,
        "contract_valid": contract_valid,
    }


def _benchmark_worker(
    *,
    worker: int,
    model: ElderActorCritic,
    inference_lock: threading.Lock,
    device: torch.device,
    bridge_path: Path,
    dotnet: str,
    selection: Stage0Selection,
    transitions: int,
    seed: int,
) -> dict[str, float | int]:
    episode_limit = min(32, transitions)
    env = _make_environment(
        bridge_path=bridge_path,
        dotnet=dotnet,
        selection=selection,
        episode_limit=episode_limit,
    )
    native_seconds = 0.0
    bridge_seconds = 0.0
    inference_seconds = 0.0
    replay_checks = 0
    illegal_actions = 0
    completed = 0
    try:
        observation, info = env.reset(seed=seed + worker)
        _validate_reset_identity(_plain_mapping(info["native_result"]), selection)
        while completed < transitions:
            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=device,
            )
            with inference_lock:
                inference_started = _timing_start(device)
                with torch.no_grad():
                    sampled = model.actor.sample(
                        observation_tensor.unsqueeze(0),
                        deterministic=True,
                    )
                inference_seconds += _timing_stop(inference_started, device)
            movement = int(sampled.movement_index[0].cpu())
            if not info["action_mask"].movement_candidates[movement]:
                illegal_actions += 1
            action = {
                "continuous": sampled.continuous[0].cpu().numpy(),
                "movement_index": movement,
            }
            bridge_started = time.perf_counter()
            observation, _, terminated, truncated, info = env.step(action)
            bridge_seconds += time.perf_counter() - bridge_started
            result = _plain_mapping(info["native_result"])
            native_seconds += float(_pick(result, "NativeStepSeconds"))
            completed += 1
            if terminated or truncated:
                replay = env.verify_replay()
                if int(replay.get("steps", 0)) < 1:
                    raise RuntimeError("Native throughput replay contained no steps")
                replay_checks += 1
                if completed < transitions:
                    observation, info = env.reset(seed=seed + worker)
                    _validate_reset_identity(
                        _plain_mapping(info["native_result"]),
                        selection,
                    )
        if not terminated and not truncated:
            replay = env.verify_replay()
            if int(replay.get("steps", 0)) < 1:
                raise RuntimeError("Native throughput replay contained no steps")
            replay_checks += 1
    finally:
        env.close()
    return {
        "transitions": completed,
        "native_step_seconds": native_seconds,
        "bridge_step_seconds": bridge_seconds,
        "policy_inference_seconds": inference_seconds,
        "replay_checks": replay_checks,
        "illegal_action_count": illegal_actions,
    }


def _configure_determinism(seed: int, device_name: str) -> torch.device:
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available to PyTorch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    return torch.device(device_name)


def _deterministic_action(model: ElderActorCritic, observation: Tensor) -> list[float]:
    with torch.no_grad():
        sampled = model.actor.sample(observation.unsqueeze(0), deterministic=True)
    return [float(value) for value in sampled.continuous[0].cpu()]


def _actor_progress(path: list[list[float]], active_heads: list[int]) -> tuple[float, float]:
    values = np.asarray(path, dtype=np.float64)[:, active_heads]
    direction = values[-1] - values[0]
    shift = float(np.linalg.norm(direction))
    if shift <= 1e-12 or len(values) < 3:
        return shift, 0.0
    progress = ((values - values[0]) @ direction) / float(direction @ direction)
    deltas = np.diff(progress)
    consistency = float(np.mean(deltas >= -0.02))
    return shift, consistency


def _mean_action_path_shift(
    before: list[list[float]],
    after: list[list[float]],
    active_heads: list[int],
) -> float:
    count = min(len(before), len(after))
    if count == 0 or not active_heads:
        return 0.0
    before_values = np.asarray(before[:count], dtype=np.float64)[:, active_heads]
    after_values = np.asarray(after[:count], dtype=np.float64)[:, active_heads]
    return float(np.mean(np.linalg.norm(after_values - before_values, axis=1)))


def _value_mse(model: ElderActorCritic, batch: RolloutBatch) -> float:
    with torch.no_grad():
        predicted = model.critic(batch.observations)
        return float(torch.square(predicted - batch.returns).mean())


def _observation_guard(
    normalizer: SemanticNormalizer,
    raw: EncodedObservation,
) -> tuple[bool, int, int]:
    transformed = np.asarray(normalizer.transform(raw), dtype=np.float64)
    _, present = normalizer.semantic_coordinates(raw)
    eligible = np.asarray(
        [
            is_present and kind not in {"binary", "bounded01"}
            for is_present, kind in zip(
                present,
                normalizer.manifest.kinds,
                strict=True,
            )
        ],
        dtype=bool,
    )
    finite = bool(np.all(np.isfinite(transformed)))
    saturated = int(np.count_nonzero(np.abs(transformed[eligible]) >= 0.95))
    return finite, saturated, int(np.count_nonzero(eligible))


def _requested_matches_sample(
    requested: dict[str, Any],
    continuous: np.ndarray,
    movement_index: int,
) -> bool:
    expected = (
        float(continuous[0]),
        float(continuous[1]),
        float(continuous[2]),
        float(continuous[3]),
        movement_index - 1,
    )
    actual = _action_signature(requested)
    return all(
        left == right if isinstance(left, int) else math.isclose(left, right, abs_tol=1e-12)
        for left, right in zip(actual, expected, strict=True)
    )


def _action_signature(action: dict[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        float(_pick(action, "RetainedFieldOperation")),
        float(_pick(action, "RetainedHerdFraction")),
        float(_pick(action, "DevelopmentBudget")),
        float(_pick(action, "FieldDevelopmentShare")),
        int(_pick(action, "MovementCandidate")),
    )


def _action_l2(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_values = np.asarray(_action_signature(left)[:4], dtype=np.float64)
    right_values = np.asarray(_action_signature(right)[:4], dtype=np.float64)
    return float(np.linalg.norm(left_values - right_values))


def _edge_rates(flags: list[bool]) -> tuple[float, float]:
    if not flags:
        return 0.0, 0.0
    width = max(1, len(flags) // 4)
    early = sum(flags[:width]) / width
    late = sum(flags[-width:]) / width
    return float(early), float(late)


def _edge_means(values: list[float] | list[int]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    width = max(1, len(values) // 4)
    return float(np.mean(values[:width])), float(np.mean(values[-width:]))


def _validate_reset_identity(
    reset: dict[str, Any],
    selection: Stage0Selection,
) -> None:
    record = selection.record
    checks = {
        "initial_state_fingerprint": (
            str(_pick(reset, "InitialStateFingerprint")).upper(),
            record.native_fingerprint.upper(),
        ),
        "configuration_hash": (
            str(_pick(reset, "ConfigurationHash")).upper(),
            record.configuration_hash.upper(),
        ),
        "simulator_version": (
            str(_pick(reset, "SimulatorVersion")),
            record.native_simulator_version,
        ),
        "controlled_camp_id": (
            int(_pick(reset, "ControlledCampId")),
            record.controlled.camp_id,
        ),
        "controlled_player": (
            int(_pick(reset, "ControlledPlayer")),
            record.controlled.player_id,
        ),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise RuntimeError(f"Native reset identity mismatch: {mismatches}")


def _validate_benchmark_checkpoint(
    payload: dict[str, Any],
    selection: Stage0Selection,
    bridge_path: Path,
) -> dict[str, Any]:
    identities = _plain_mapping(payload["identities"])
    expected = {
        "simulator_commit": SIMULATOR_COMMIT,
        "snapshot_manifest_sha256": selection.manifest_sha256,
        "snapshot_id": selection.record.snapshot_id,
        "snapshot_content_sha256": selection.record.content_sha256,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "tensor_feature_count": TENSOR_FEATURE_COUNT,
        "runtime_bridge_sha256": sha256_file(bridge_path)[0],
    }
    mismatches = {
        key: {"checkpoint": identities.get(key), "runtime": value}
        for key, value in expected.items()
        if identities.get(key) != value
    }
    checkpoint_normalizer = NormalizationManifest(**payload["normalizer"])
    checkpoint_normalizer.validate()
    if asdict(checkpoint_normalizer) != asdict(selection.normalizer.manifest):
        mismatches["normalizer"] = {
            "checkpoint": checkpoint_normalizer.partition_id,
            "runtime": selection.normalizer.manifest.partition_id,
        }
    if mismatches:
        raise ValueError(f"Checkpoint cannot benchmark this native setup: {mismatches}")
    return identities


def _checkpoint_identities(
    selection: Stage0Selection,
    reset_identity: dict[str, Any] | None,
    bridge_path: Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    )
    record = selection.record
    return {
        "learner_commit": commit,
        "learner_tracked_files_dirty": dirty,
        "simulator_commit": SIMULATOR_COMMIT,
        "snapshot_manifest_sha256": selection.manifest_sha256,
        "snapshot_id": record.snapshot_id,
        "scenario_family": record.scenario_family,
        "snapshot_content_sha256": record.content_sha256,
        "snapshot_native_fingerprint": record.native_fingerprint,
        "normalizer_partition_id": selection.normalizer.manifest.partition_id,
        "normalizer_build_hash": selection.normalizer.manifest.build_hash,
        "configuration_hash": record.configuration_hash,
        "observation_schema_hash": record.schema_identity.observation_schema_hash,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "critic_sidecar_schema_version": CRITIC_SIDECAR_SCHEMA_VERSION,
        "tensor_feature_count": TENSOR_FEATURE_COUNT,
        "runtime_bridge_sha256": sha256_file(bridge_path)[0],
        "runtime_reset": reset_identity,
    }


def _save_and_verify_checkpoint(
    *,
    path: Path,
    model: ElderActorCritic,
    trainer: PPOTrainer,
    actor_critic_config: ElderActorCriticConfig,
    ppo_config: PPOConfig,
    overfit_config: OverfitConfig,
    normalizer: SemanticNormalizer,
    identities: dict[str, Any],
    observation: Tensor,
    continuation_batch: RolloutBatch,
    device: torch.device,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema_version": "elder-ppo-checkpoint-v2",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "actor_critic_config": asdict(actor_critic_config),
        "ppo_config": asdict(ppo_config),
        "overfit_config": asdict(overfit_config),
        "normalizer": asdict(normalizer.manifest),
        "identities": identities,
        "rng": _capture_rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)

    expected = _policy_vector(model, observation)
    loaded = torch.load(path, map_location=device, weights_only=False)
    loaded_normalizer = NormalizationManifest(**loaded["normalizer"])
    loaded_normalizer.validate()
    if asdict(loaded_normalizer) != asdict(normalizer.manifest):
        raise RuntimeError("Checkpoint changed the frozen normalizer")
    reloaded_model = ElderActorCritic(ElderActorCriticConfig(**loaded["actor_critic_config"])).to(
        device
    )
    reloaded_model.load_state_dict(loaded["model_state_dict"])
    reloaded_trainer = PPOTrainer(
        reloaded_model,
        PPOConfig(**loaded["ppo_config"]),
    )
    reloaded_trainer.optimizer.load_state_dict(copy.deepcopy(loaded["optimizer_state_dict"]))
    actual = _policy_vector(reloaded_model, observation)
    maximum_error = float(torch.max(torch.abs(expected - actual)).cpu())

    reference_model = ElderActorCritic(ElderActorCriticConfig(**loaded["actor_critic_config"])).to(
        device
    )
    reference_model.load_state_dict(loaded["model_state_dict"])
    reference_trainer = PPOTrainer(
        reference_model,
        PPOConfig(**loaded["ppo_config"]),
    )
    reference_trainer.optimizer.load_state_dict(copy.deepcopy(loaded["optimizer_state_dict"]))

    _restore_rng_state(loaded["rng"])
    expected_stochastic = _stochastic_policy_vector(reference_model, observation)
    _restore_rng_state(loaded["rng"])
    actual_stochastic = _stochastic_policy_vector(reloaded_model, observation)
    stochastic_error = float(torch.max(torch.abs(expected_stochastic - actual_stochastic)).cpu())

    _restore_rng_state(loaded["rng"])
    reference_metrics = reference_trainer.update(continuation_batch)
    _restore_rng_state(loaded["rng"])
    reloaded_metrics = reloaded_trainer.update(continuation_batch)
    continuation_model_error = _model_state_max_abs_error(
        reference_model,
        reloaded_model,
    )
    continuation_metrics_error = max(
        abs(reference_metrics[key] - reloaded_metrics[key]) for key in reference_metrics
    )
    digest, size = sha256_file(path)
    return {
        "sha256": digest,
        "bytes": size,
        "max_abs_error": maximum_error,
        "stochastic_policy_max_abs_error": stochastic_error,
        "continuation_model_max_abs_error": continuation_model_error,
        "continuation_metrics_max_abs_error": continuation_metrics_error,
        "optimizer_state_reloaded": bool(reloaded_trainer.optimizer.state_dict()["state"]),
        "rng_state_fields": sorted(loaded["rng"]),
    }


def _policy_vector(model: ElderActorCritic, observation: Tensor) -> Tensor:
    with torch.no_grad():
        mean, log_std, movement_logits = model.actor.parameters_for(observation.unsqueeze(0))
        sampled = model.actor.sample(observation.unsqueeze(0), deterministic=True)
        value = model.critic(observation.unsqueeze(0)).unsqueeze(-1)
    return torch.cat(
        (
            mean.to(torch.float64),
            log_std.expand_as(mean).to(torch.float64),
            movement_logits.to(torch.float64),
            sampled.continuous.to(torch.float64),
            sampled.movement_index.unsqueeze(-1).to(torch.float64),
            sampled.log_prob.unsqueeze(-1).to(torch.float64),
            value.to(torch.float64),
        ),
        dim=-1,
    ).cpu()


def _stochastic_policy_vector(model: ElderActorCritic, observation: Tensor) -> Tensor:
    with torch.no_grad():
        sampled = model.actor.sample(observation.unsqueeze(0), deterministic=False)
    return torch.cat(
        (
            sampled.continuous.to(torch.float64),
            sampled.movement_index.unsqueeze(-1).to(torch.float64),
            sampled.log_prob.unsqueeze(-1).to(torch.float64),
            sampled.entropy.unsqueeze(-1).to(torch.float64),
        ),
        dim=-1,
    ).cpu()


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def _model_state_max_abs_error(
    expected: ElderActorCritic,
    actual: ElderActorCritic,
) -> float:
    errors = [
        float(
            torch.max(
                torch.abs(
                    expected.state_dict()[key].to(torch.float64)
                    - actual.state_dict()[key].to(torch.float64)
                )
            ).cpu()
        )
        for key in expected.state_dict()
    ]
    return max(errors, default=0.0)


def _timing_report(
    timings: _TimingTotals,
    transitions: int,
    end_to_end_seconds: float,
) -> dict[str, float | int]:
    return {
        "transitions": transitions,
        "native_step_seconds": timings.native_step_seconds,
        "native_step_transitions_per_second": transitions / timings.native_step_seconds,
        "bridge_step_seconds": timings.bridge_step_seconds,
        "policy_inference_seconds": timings.policy_inference_seconds,
        "policy_inference_transitions_per_second": transitions / timings.policy_inference_seconds,
        "ppo_update_seconds": timings.ppo_update_seconds,
        "ppo_update_transitions_per_second": transitions / timings.ppo_update_seconds,
        "rollout_collection_seconds": timings.collection_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "end_to_end_transitions_per_second": transitions / end_to_end_seconds,
    }


def _timing_start(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _timing_stop(started: float, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started


def _all_finite(*values: Tensor) -> bool:
    return all(bool(torch.all(torch.isfinite(value))) for value in values)


def _raw_observation(info: dict[str, Any]) -> EncodedObservation:
    raw = info.get("raw_observation")
    if not isinstance(raw, EncodedObservation):
        raise TypeError("Gym adapter did not preserve the raw native observation")
    return raw


def _plain_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object from the native bridge")
    return value


def _mapping(mapping: dict[str, Any], *names: str) -> dict[str, Any]:
    value = _pick(mapping, *names)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping for {names!r}")
    return value


def _sequence(mapping: dict[str, Any], *names: str) -> list[Any]:
    value = _pick(mapping, *names)
    if not isinstance(value, list):
        raise TypeError(f"Expected list for {names!r}")
    return value


def _pick(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
        camel = name[:1].lower() + name[1:]
        if camel in mapping:
            return mapping[camel]
    raise KeyError(f"Missing required native field; tried {names!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
