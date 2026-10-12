"""Balanced multi-snapshot Elder PPO campaign and native evaluation.

The campaign keeps storage, sampling, learning, and evaluation on the learner
side.  Native dynamics, action compilation, reward semantics, and the Option-A
comparison all remain authoritative in the pinned simulator assembly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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
from anthropocene_rl.datasets.shards import DatasetReader, load_dataset_reader
from anthropocene_rl.datasets.storage import (
    DEFAULT_SNAPSHOT_STORE_CONFIG,
    resolve_snapshot_blob,
)
from anthropocene_rl.env.gym_env import AnthropoceneElderEnv
from anthropocene_rl.env.transport import JsonLineProcessTransport
from anthropocene_rl.observations.normalization import (
    NormalizationManifest,
    SemanticNormalizer,
    semantic_normalizer_from_manifest,
)
from anthropocene_rl.observations.schema import (
    EncodedObservation,
    broad_domain_scope_violations,
    cultivation_only_scope_violations,
)
from anthropocene_rl.policies.distributions import movement_mask, structural_head_mask
from anthropocene_rl.policies.elder_mlp import ElderActorCritic, ElderActorCriticConfig
from anthropocene_rl.training.overfit import (
    _action_l2,
    _all_finite,
    _capture_rng_state,
    _configure_determinism,
    _mapping,
    _model_state_max_abs_error,
    _observation_guard,
    _pick,
    _plain_mapping,
    _policy_vector,
    _raw_observation,
    _requested_matches_sample,
    _restore_rng_state,
    _sequence,
    _stochastic_policy_vector,
    _timing_start,
    _timing_stop,
    _write_json,
)
from anthropocene_rl.training.ppo import (
    PPOConfig,
    PPOTrainer,
    RolloutBatch,
    critic_fit_diagnostics,
)
from anthropocene_rl.training.returns import (
    discounted_interval_reward,
    generalized_advantage_estimate,
    physical_gamma,
)

PILOT_FAMILIES = ("F01", "F02", "F03", "F13", "F15", "F16", "F17", "F18")
BROAD_DOMAIN_FAMILIES = (
    "F01",
    "F02",
    "F03",
    "F04",
    "F05",
    "F06",
    "F07",
    "F13",
    "F14",
    "F15",
    "F16",
    "F17",
    "F18",
)
CampaignScope = Literal["cultivation-only", "broad-domain"]
DEFAULT_MANIFEST = Path("datasets/manifests/elder-v2-cultivation-pilot-v1.json")
DEFAULT_NORMALIZER = Path("evidence/cultivation-pilot-v1/final-train-normalization.json")
DEFAULT_BLOB_ROOT = Path("datasets/blobs")
DEFAULT_BRIDGE_DLL = Path("integration/bin/Release/net8.0/Anthropocene.RLBridge.dll")
DEFAULT_EVIDENCE_DIRECTORY = Path("evidence/cultivation-ppo-pilot-v1")
DEFAULT_CHECKPOINT_DIRECTORY = Path("checkpoints/cultivation-ppo-pilot-v1")
DEFAULT_SEEDS = (1701, 1901, 2101)
DEFAULT_GCS_DATASET_MANIFEST = (
    "gs://anthropocene-rl-datasets-508912/datasets/elder-v2-cultivation-pilot-v1/"
    "2026-10-11-manifest-9d33a9e5/manifest.json"
)


@dataclass(frozen=True)
class PilotConfig:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    device: str = "cpu"
    updates: int = 96
    episodes_per_update: int = 8
    rollouts_per_snapshot: int = 1
    group_relative_first_decision: bool = False
    episode_steps: int = 3
    evaluation_interval: int = 24
    evaluation_steps: int = 3
    horizon_years: float = 400.0
    lambda_ref: float = 0.95
    lambda_reference_years: float = 20.0
    reward_scale: float = 0.001
    train_probe_per_family: int = 2
    evaluate_heldout: bool = True
    campaign_scope: CampaignScope = "cultivation-only"
    replay_verification_interval: int = 1
    collector_workers: int = 1

    @property
    def families(self) -> tuple[str, ...]:
        return (
            BROAD_DOMAIN_FAMILIES
            if self.campaign_scope == "broad-domain"
            else PILOT_FAMILIES
        )

    def validate(self) -> None:
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("The cultivation pilot requires three distinct neural seeds")
        if self.campaign_scope not in ("cultivation-only", "broad-domain"):
            raise ValueError("Unsupported campaign scope")
        if self.updates < 2 or self.episodes_per_update < len(self.families):
            raise ValueError("Pilot updates must expose every family at least once")
        if self.episodes_per_update % len(self.families) != 0:
            raise ValueError("episodes_per_update must be a multiple of campaign families")
        if self.episode_steps < 1 or self.evaluation_steps < 1:
            raise ValueError("Episode lengths must be positive")
        if self.rollouts_per_snapshot < 1:
            raise ValueError("rollouts_per_snapshot must be positive")
        if self.group_relative_first_decision and self.rollouts_per_snapshot < 2:
            raise ValueError("Group-relative advantages require at least two rollouts per snapshot")
        if self.campaign_scope == "broad-domain" and (
            self.episode_steps != 50 or self.evaluation_steps != 50
        ):
            raise ValueError(
                "The broad-domain campaign is pinned to 50-decision training and evaluation"
            )
        if self.evaluation_interval < 1 or self.evaluation_interval > self.updates:
            raise ValueError("evaluation_interval must fall within the run")
        if self.horizon_years <= 0 or not 0 <= self.lambda_ref <= 1:
            raise ValueError("Invalid physical-time credit configuration")
        if self.lambda_reference_years <= 0 or self.reward_scale <= 0:
            raise ValueError("Invalid reward or lambda reference scale")
        if self.train_probe_per_family < 1:
            raise ValueError("At least one TRAIN probe per family is required")
        if self.replay_verification_interval < 1:
            raise ValueError("Replay verification interval must be positive")
        if self.collector_workers < 1 or self.collector_workers > self.episodes_per_update:
            raise ValueError("collector_workers must fit within episodes_per_update")
        if self.device != "cpu" and self.collector_workers != 1:
            raise ValueError("Parallel native collection currently requires the CPU policy")


@dataclass
class CampaignSelection:
    manifest: SnapshotBankManifest
    manifest_sha256: str
    record: SnapshotRecord
    snapshot_path: Path
    snapshot_bytes: bytes
    normalizer: SemanticNormalizer


class CampaignDataset:
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        blob_root: str | Path,
        normalizer_path: str | Path,
        snapshot_store_config: str | Path = DEFAULT_SNAPSHOT_STORE_CONFIG,
        dataset_manifest: str | Path | None = DEFAULT_GCS_DATASET_MANIFEST,
        dataset_store_config: str | Path = DEFAULT_DATASET_STORE_CONFIG,
        campaign_scope: CampaignScope = "cultivation-only",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.blob_root = Path(blob_root)
        self.normalizer_path = Path(normalizer_path)
        self.snapshot_store_config = Path(snapshot_store_config)
        self.dataset_manifest = str(dataset_manifest) if dataset_manifest is not None else None
        self.dataset_store_config = Path(dataset_store_config)
        self.campaign_scope = campaign_scope
        self._dataset_reader: DatasetReader | None = None
        self.manifest = SnapshotBankManifest.load(self.manifest_path)
        self.manifest_sha256 = self.manifest.sha256()
        normalization_manifest = NormalizationManifest.load(self.normalizer_path)
        expected_partition = f"{self.manifest.bank_id}:train:final"
        if normalization_manifest.partition_id != expected_partition:
            raise ValueError("Pilot requires the frozen final TRAIN-only normalizer")
        if normalization_manifest.partition_manifest_sha256 != self.manifest_sha256:
            raise ValueError("Pilot normalizer and bank manifest identities differ")
        self.normalizer = semantic_normalizer_from_manifest(normalization_manifest)
        self._bytes: dict[str, tuple[Path, bytes]] = {}
        self.by_partition = {
            partition: tuple(
                sorted(
                    (record for record in self.manifest.records if record.partition == partition),
                    key=lambda item: item.snapshot_id,
                )
            )
            for partition in Partition
        }
        self._validate_scope()

    def _validate_scope(self) -> None:
        families = {record.scenario_family for record in self.manifest.records}
        expected_families = (
            set(BROAD_DOMAIN_FAMILIES)
            if self.campaign_scope == "broad-domain"
            else set(PILOT_FAMILIES)
        )
        if families != expected_families:
            raise ValueError(f"Unexpected pilot families: {sorted(families)}")
        for record in self.manifest.records:
            strata = record.actual_strata
            if record.native_simulator_sha != SIMULATOR_COMMIT:
                raise ValueError("Campaign record changed the frozen simulator SHA")
            if self.campaign_scope == "cultivation-only":
                if (
                    strata.structural_masks.movement
                    or strata.structural_masks.herd_retention
                    or strata.structural_masks.herd_development
                    or strata.owned_domestic_herd
                    or strata.fishing_available
                    or strata.fishing_ground_count
                ):
                    raise ValueError(
                        f"Snapshot escaped cultivation-only scope: {record.snapshot_id}"
                    )
                if not (
                    strata.structural_masks.field_retention
                    and strata.structural_masks.field_development
                    and "cultivation-only" in strata.tags
                ):
                    raise ValueError(
                        f"Snapshot lacks active cultivation controls: {record.snapshot_id}"
                    )
            elif strata.fishing_available or strata.fishing_ground_count:
                raise ValueError(
                    f"Snapshot escaped fishing-excluded broad scope: {record.snapshot_id}"
                )
            if record.configuration_hash != self.normalizer.manifest.configuration_hash:
                raise ValueError("Snapshot configuration differs from the normalizer cohort")
            if record.native_build_hash != self.normalizer.manifest.build_hash:
                raise ValueError("Snapshot build differs from the normalizer cohort")
            if (
                record.schema_identity.observation_schema_hash
                != self.normalizer.manifest.observation_schema_hash
            ):
                raise ValueError("Snapshot observation schema differs from the normalizer")

    def selection(self, record: SnapshotRecord) -> CampaignSelection:
        cached = self._bytes.get(record.content_sha256)
        if cached is None:
            local_path = self.blob_root / record.blob_path
            if local_path.is_file() or self.dataset_manifest is None:
                path = resolve_snapshot_blob(
                    record,
                    blob_root=self.blob_root,
                    config_path=self.snapshot_store_config,
                    manifest_path=self.manifest_path,
                )
            else:
                path = self._gcs_reader().materialize_record(record.snapshot_id)
            cached = (path, path.read_bytes())
            self._bytes[record.content_sha256] = cached
        return CampaignSelection(
            manifest=self.manifest,
            manifest_sha256=self.manifest_sha256,
            record=record,
            snapshot_path=cached[0],
            snapshot_bytes=cached[1],
            normalizer=self.normalizer,
        )

    def _gcs_reader(self) -> DatasetReader:
        if self._dataset_reader is not None:
            return self._dataset_reader
        if self.dataset_manifest is None:
            raise FileNotFoundError("No GCS dataset manifest is configured")
        config = DatasetStoreConfig.load(self.dataset_store_config)
        reader = load_dataset_reader(
            self.dataset_manifest,
            store=build_object_store(config),
            cache_root=config.cache_root,
            max_cache_bytes=config.max_cache_bytes,
        )
        if reader.manifest.source_manifest_sha256 != self.manifest_sha256:
            raise ValueError("GCS dataset and snapshot-bank manifest identities differ")
        expected_ids = {record.snapshot_id for record in self.manifest.records}
        if set(reader.record_ids()) != expected_ids:
            raise ValueError("GCS dataset record cohort differs from the frozen snapshot bank")
        self._dataset_reader = reader
        return reader


class BalancedFamilySampler:
    def __init__(
        self,
        records: tuple[SnapshotRecord, ...],
        seed: int,
        families: tuple[str, ...] = PILOT_FAMILIES,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.families = families
        self.records: dict[str, list[SnapshotRecord]] = {}
        self.cursor: dict[str, int] = {}
        for family in self.families:
            family_records = sorted(
                (record for record in records if record.scenario_family == family),
                key=lambda item: item.snapshot_id,
            )
            if not family_records:
                raise ValueError(f"TRAIN contains no {family} snapshots")
            self.rng.shuffle(family_records)
            self.records[family] = family_records
            self.cursor[family] = 0

    def sample(self, count: int) -> tuple[SnapshotRecord, ...]:
        repeats = count // len(self.families)
        families = list(self.families) * repeats
        self.rng.shuffle(families)
        selected: list[SnapshotRecord] = []
        for family in families:
            items = self.records[family]
            cursor = self.cursor[family]
            if cursor >= len(items):
                self.rng.shuffle(items)
                cursor = 0
            selected.append(items[cursor])
            self.cursor[family] = cursor + 1
        return tuple(selected)


@dataclass
class _Timing:
    native_step_seconds: float = 0.0
    bridge_step_seconds: float = 0.0
    policy_inference_seconds: float = 0.0
    ppo_update_seconds: float = 0.0
    collection_seconds: float = 0.0
    collector_worker_seconds: float = 0.0
    collector_workers_peak: int = 1


def run_cultivation_pilot(
    *,
    bridge_dll: str | Path = DEFAULT_BRIDGE_DLL,
    dotnet: str = "dotnet",
    manifest_path: str | Path = DEFAULT_MANIFEST,
    blob_root: str | Path = DEFAULT_BLOB_ROOT,
    normalizer_path: str | Path = DEFAULT_NORMALIZER,
    snapshot_store_config: str | Path = DEFAULT_SNAPSHOT_STORE_CONFIG,
    dataset_manifest: str | Path | None = DEFAULT_GCS_DATASET_MANIFEST,
    dataset_store_config: str | Path = DEFAULT_DATASET_STORE_CONFIG,
    evidence_directory: str | Path = DEFAULT_EVIDENCE_DIRECTORY,
    checkpoint_directory: str | Path = DEFAULT_CHECKPOINT_DIRECTORY,
    config: PilotConfig | None = None,
    ppo_config: PPOConfig | None = None,
    actor_critic_config: ElderActorCriticConfig | None = None,
) -> dict[str, Any]:
    config = config or PilotConfig()
    config.validate()
    ppo_config = ppo_config or PPOConfig(
        learning_rate=1e-3,
        actor_learning_rate=3e-4,
        entropy_coefficient=0.0,
        epochs=8,
        minibatch_size=64,
        target_kl=0.05,
    )
    actor_critic_config = actor_critic_config or ElderActorCriticConfig(initial_log_std=-1.0)
    evidence_dir = Path(evidence_directory)
    checkpoint_dir = Path(checkpoint_directory)
    bridge_path = Path(bridge_dll).resolve()
    if not bridge_path.is_file():
        raise FileNotFoundError(f"Native bridge is missing at {bridge_path}")
    dataset = CampaignDataset(
        manifest_path=manifest_path,
        blob_root=blob_root,
        normalizer_path=normalizer_path,
        snapshot_store_config=snapshot_store_config,
        dataset_manifest=dataset_manifest,
        dataset_store_config=dataset_store_config,
        campaign_scope=config.campaign_scope,
    )
    train = dataset.by_partition[Partition.TRAIN]
    development = dataset.by_partition[Partition.DEVELOPMENT]
    heldout = dataset.by_partition[Partition.HELDOUT]
    train_probe = _probe_records(
        train,
        config.train_probe_per_family,
        families=config.families,
    )
    development_probe = _probe_records(
        development,
        1,
        families=config.families,
    )

    campaign_started = time.perf_counter()
    option_a = {
        "train_probe": _evaluate_records(
            dataset=dataset,
            records=train_probe,
            bridge_path=bridge_path,
            dotnet=dotnet,
            episode_steps=config.evaluation_steps,
            horizon_years=config.horizon_years,
            reward_scale=config.reward_scale,
            policy="option-a",
        ),
        "development": _evaluate_records(
            dataset=dataset,
            records=development,
            bridge_path=bridge_path,
            dotnet=dotnet,
            episode_steps=config.evaluation_steps,
            horizon_years=config.horizon_years,
            reward_scale=config.reward_scale,
            policy="option-a",
        ),
    }
    if config.evaluate_heldout:
        option_a["heldout"] = _evaluate_records(
            dataset=dataset,
            records=heldout,
            bridge_path=bridge_path,
            dotnet=dotnet,
            episode_steps=config.evaluation_steps,
            horizon_years=config.horizon_years,
            reward_scale=config.reward_scale,
            policy="option-a",
        )
    _write_json(evidence_dir / "option-a-baseline.json", option_a)

    seed_reports: list[dict[str, Any]] = []
    for seed in config.seeds:
        report = _run_seed(
            seed=seed,
            dataset=dataset,
            train=train,
            train_probe=train_probe,
            development=development,
            development_probe=development_probe,
            heldout=heldout,
            option_a=option_a,
            bridge_path=bridge_path,
            dotnet=dotnet,
            checkpoint_path=checkpoint_dir / f"seed-{seed}.pt",
            config=config,
            ppo_config=ppo_config,
            actor_config=actor_critic_config,
        )
        seed_reports.append(report)
        _write_json(evidence_dir / f"seed-{seed}.json", report)

    trajectory = _cultivation_trajectory(seed_reports)
    validation = _validation_assessment(seed_reports)
    promotion = _promotion_assessment(seed_reports)
    project_root = Path(__file__).resolve().parents[3]
    identities = _campaign_identities(dataset, bridge_path, project_root)
    result = {
        "status": (
            "PASS"
            if all(validation.values()) and all(promotion.values())
            else ("NO_GO" if all(validation.values()) else "FAIL")
        ),
        "validation_status": "PASS" if all(validation.values()) else "FAIL",
        "promotion_status": (
            "PASS" if all(validation.values()) and all(promotion.values()) else "NO_GO"
        ),
        "scope": (
            "movement-husbandry-cultivation-open-economy-ppo-pilot"
            if config.campaign_scope == "broad-domain"
            else "fixed-location-cultivation-control-open-economy-ppo-pilot"
        ),
        "identities": identities,
        "configuration": {
            "pilot": asdict(config),
            "ppo": asdict(ppo_config),
            "actor_critic": asdict(actor_critic_config),
            "reset_sampler": {
                "kind": "exact-balanced-family-within-family-shuffled-cycle",
                "families": list(config.families),
                "episodes_per_update": config.episodes_per_update,
                "rollouts_per_snapshot": config.rollouts_per_snapshot,
                "group_relative_first_decision": config.group_relative_first_decision,
                "rollout_episodes_per_update": (
                    config.episodes_per_update * config.rollouts_per_snapshot
                ),
                "optimization_partition": Partition.TRAIN.value,
                "development_use": "checkpoint-selection-and-debugging-only",
                "heldout_use": (
                    "single-final-evaluation-only"
                    if config.evaluate_heldout
                    else "sealed-not-evaluated"
                ),
            },
            "structural_scope": {
                "asserted": "reset-and-every-post-step-state",
                "maximum_episode_decisions": config.episode_steps,
                "admitted": (
                    ["movement", "husbandry", "herds", "cultivation"]
                    if config.campaign_scope == "broad-domain"
                    else ["cultivation"]
                ),
                "excluded": (
                    ["fishing"]
                    if config.campaign_scope == "broad-domain"
                    else ["movement", "husbandry", "herds", "fishing"]
                ),
                "failure_mode": "fail-fast",
            },
        },
        "bank": {
            "records": len(dataset.manifest.records),
            "partition_counts": {
                partition.value: len(records) for partition, records in dataset.by_partition.items()
            },
            "family_counts": dict(
                sorted(
                    Counter(record.scenario_family for record in dataset.manifest.records).items()
                )
            ),
        },
        "option_a_baseline": {key: value["summary"] for key, value in option_a.items()},
        "seeds": seed_reports,
        "cultivation_intensity_trajectory": trajectory,
        "validation_gates": validation,
        "promotion_gates": promotion,
        "elapsed_seconds": time.perf_counter() - campaign_started,
    }
    result["evidence_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    _write_json(evidence_dir / "campaign-summary.json", result)
    return result


def _run_seed(
    *,
    seed: int,
    dataset: CampaignDataset,
    train: tuple[SnapshotRecord, ...],
    train_probe: tuple[SnapshotRecord, ...],
    development: tuple[SnapshotRecord, ...],
    development_probe: tuple[SnapshotRecord, ...],
    heldout: tuple[SnapshotRecord, ...],
    option_a: dict[str, dict[str, Any]],
    bridge_path: Path,
    dotnet: str,
    checkpoint_path: Path,
    config: PilotConfig,
    ppo_config: PPOConfig,
    actor_config: ElderActorCriticConfig,
) -> dict[str, Any]:
    device = _configure_determinism(seed, config.device)
    model = ElderActorCritic(actor_config).to(device)
    trainer = PPOTrainer(model, ppo_config)
    sampler = BalancedFamilySampler(train, seed, config.families)
    timings = _Timing()
    first_selection = dataset.selection(train[0])
    envs = tuple(
        _make_environment(
            bridge_path=bridge_path,
            dotnet=dotnet,
            selection=first_selection,
            episode_limit=config.episode_steps,
            campaign_scope=config.campaign_scope,
        )
        for _ in range(config.collector_workers)
    )
    learning_curve: list[dict[str, Any]] = []
    evaluation_curve: list[dict[str, Any]] = []
    visitation_snapshots: Counter[str] = Counter()
    visitation_transitions: Counter[str] = Counter()
    strata_visitation: Counter[str] = Counter()
    inactive_head_gradient_max = [0.0] * 4
    movement_head_gradient_max = 0.0
    totals: Counter[str] = Counter()
    last_batch: RolloutBatch | None = None
    best_score = -math.inf
    best_update = 0
    best_state: dict[str, Any] | None = None
    best_health_score = -math.inf
    best_health_update = 0
    best_health_state: dict[str, Any] | None = None
    initial_activation_diagnostics: dict[str, object] | None = None
    fixed_training_observations: Tensor | None = None
    fixed_critic_targets: Tensor | None = None
    activation_probe_curve: list[dict[str, Any]] = []
    critic_fit_curve: list[dict[str, Any]] = []

    initial_evaluation = _evaluate_records(
        dataset=dataset,
        records=development_probe,
        bridge_path=bridge_path,
        dotnet=dotnet,
        episode_steps=config.evaluation_steps,
        horizon_years=config.horizon_years,
        reward_scale=config.reward_scale,
        policy="learned",
        model=model,
        device=device,
    )
    evaluation_curve.append(
        {"update": 0, "partition": "development-probe", **initial_evaluation["summary"]}
    )
    initial_development_probe_score = float(
        initial_evaluation["summary"]["discounted_scaled_return"]["mean"]
    )
    fixed_development_observations = _fixed_reset_observations(
        dataset=dataset,
        records=development_probe,
        bridge_path=bridge_path,
        dotnet=dotnet,
        episode_steps=config.evaluation_steps,
        device=device,
    )

    training_started = time.perf_counter()
    try:
        for update in range(1, config.updates + 1):
            records = sampler.sample(config.episodes_per_update)
            collected = _collect_update_parallel(
                envs=envs,
                dataset=dataset,
                records=records,
                model=model,
                trainer=trainer,
                device=device,
                config=config,
                timings=timings,
                neural_seed=seed,
                update=update,
            )
            batch = collected.pop("batch")
            last_batch = batch
            if fixed_training_observations is None:
                fixed_training_observations = batch.observations.detach().clone()
                fixed_critic_targets = batch.returns.detach().clone()
                initial_activation_diagnostics = model.hidden_activation_diagnostics(
                    fixed_training_observations
                )
                activation_probe_curve.append(
                    _activation_probe_row(
                        update=0,
                        model=model,
                        train_observations=fixed_training_observations,
                        development_observations=fixed_development_observations,
                    )
                )
                critic_fit_curve.append(
                    {
                        "update": 0,
                        **_critic_probe_diagnostics(
                            model,
                            fixed_training_observations,
                            fixed_critic_targets,
                            trainer=trainer,
                        ),
                    }
                )
            update_started = _timing_start(device)
            metrics = trainer.update(batch)
            timings.ppo_update_seconds += _timing_stop(update_started, device)
            row = {
                "update": update,
                **collected,
                "ppo": metrics,
            }
            learning_curve.append(row)
            for family, count in collected["snapshot_visitation"].items():
                visitation_snapshots[family] += int(count)
            for family, count in collected["transition_visitation"].items():
                visitation_transitions[family] += int(count)
            for tag, count in collected["strata_visitation"].items():
                strata_visitation[tag] += int(count)
            for key in (
                "illegal_action_count",
                "requested_sample_mismatches",
                "non_stay_requested_count",
                "movement_request_count",
                "non_finite_count",
                "replay_failures",
                "reward_denominator_violations",
                "replay_verified_episodes",
                "replay_skipped_episodes",
            ):
                totals[key] += int(collected[key])
            for head in range(4):
                inactive_head_gradient_max[head] = max(
                    inactive_head_gradient_max[head],
                    metrics[f"continuous_head_{head}_grad_max"],
                )
            movement_head_gradient_max = max(
                movement_head_gradient_max,
                metrics["movement_head_grad_max"],
            )
            if update % config.evaluation_interval == 0 or update == config.updates:
                evaluated = _evaluate_records(
                    dataset=dataset,
                    records=development_probe,
                    bridge_path=bridge_path,
                    dotnet=dotnet,
                    episode_steps=config.evaluation_steps,
                    horizon_years=config.horizon_years,
                    reward_scale=config.reward_scale,
                    policy="learned",
                    model=model,
                    device=device,
                )
                critic_health = _critic_health_window(learning_curve)
                evaluation_curve.append(
                    {
                        "update": update,
                        "partition": "development-probe",
                        "critic_health": critic_health,
                        **evaluated["summary"],
                    }
                )
                assert fixed_training_observations is not None
                assert fixed_critic_targets is not None
                activation_probe_curve.append(
                    _activation_probe_row(
                        update=update,
                        model=model,
                        train_observations=fixed_training_observations,
                        development_observations=fixed_development_observations,
                    )
                )
                critic_fit_curve.append(
                    {
                        "update": update,
                        **_critic_probe_diagnostics(
                            model,
                            fixed_training_observations,
                            fixed_critic_targets,
                            trainer=trainer,
                        ),
                    }
                )
                score = float(evaluated["summary"]["discounted_scaled_return"]["mean"])
                if score > best_score:
                    best_score = score
                    best_update = update
                    best_state = _capture_training_state(model, trainer)
                if critic_health["eligible"] and score > best_health_score:
                    best_health_score = score
                    best_health_update = update
                    best_health_state = _capture_training_state(model, trainer)
    finally:
        for env in envs:
            env.close()
    training_seconds = time.perf_counter() - training_started
    if (
        last_batch is None
        or best_state is None
        or fixed_training_observations is None
        or fixed_critic_targets is None
        or initial_activation_diagnostics is None
    ):
        raise RuntimeError("Pilot seed produced no selected checkpoint")
    final_activation_probe = _activation_probe_row(
        update=config.updates,
        model=model,
        train_observations=fixed_training_observations,
        development_observations=fixed_development_observations,
    )
    final_critic_fit = _critic_probe_diagnostics(
        model,
        fixed_training_observations,
        fixed_critic_targets,
        trainer=trainer,
    )
    final_critic_health = _critic_health_window(learning_curve)
    terminal_development_evaluation = _evaluate_records(
        dataset=dataset,
        records=development,
        bridge_path=bridge_path,
        dotnet=dotnet,
        episode_steps=config.evaluation_steps,
        horizon_years=config.horizon_years,
        reward_scale=config.reward_scale,
        policy="learned",
        model=model,
        device=device,
    )
    terminal_checkpoint = {
        "update": config.updates,
        "development": terminal_development_evaluation,
        "baseline_comparison": _compare_evaluations(
            terminal_development_evaluation,
            option_a["development"],
        ),
    }
    shadow_health_checkpoint: dict[str, Any] | None = None
    if best_health_state is not None:
        model.load_state_dict(best_health_state["model"])
        health_evaluation = _evaluate_records(
            dataset=dataset,
            records=development,
            bridge_path=bridge_path,
            dotnet=dotnet,
            episode_steps=config.evaluation_steps,
            horizon_years=config.horizon_years,
            reward_scale=config.reward_scale,
            policy="learned",
            model=model,
            device=device,
        )
        shadow_health_checkpoint = {
            "update": best_health_update,
            "development_probe_score": best_health_score,
            "critic_health": _critic_health_window(
                learning_curve[:best_health_update]
            ),
            "development": health_evaluation,
            "baseline_comparison": _compare_evaluations(
                health_evaluation,
                option_a["development"],
            ),
        }
    model.load_state_dict(best_state["model"])
    trainer.load_state_dict(best_state["trainer"])
    _restore_rng_state(best_state["rng"])
    selected_activation_probe = _activation_probe_row(
        update=best_update,
        model=model,
        train_observations=fixed_training_observations,
        development_observations=fixed_development_observations,
    )
    selected_activation_diagnostics = selected_activation_probe["train"]
    selected_critic_fit = _critic_probe_diagnostics(
        model,
        fixed_training_observations,
        fixed_critic_targets,
        trainer=trainer,
    )
    selected_critic_health = _critic_health_window(learning_curve[:best_update])
    inactive_head_audit = _inactive_head_gradient_audit(model, last_batch)

    final_evaluation = {
        "train_probe": _evaluate_records(
            dataset=dataset,
            records=train_probe,
            bridge_path=bridge_path,
            dotnet=dotnet,
            episode_steps=config.evaluation_steps,
            horizon_years=config.horizon_years,
            reward_scale=config.reward_scale,
            policy="learned",
            model=model,
            device=device,
        ),
        "development": _evaluate_records(
            dataset=dataset,
            records=development,
            bridge_path=bridge_path,
            dotnet=dotnet,
            episode_steps=config.evaluation_steps,
            horizon_years=config.horizon_years,
            reward_scale=config.reward_scale,
            policy="learned",
            model=model,
            device=device,
        ),
    }
    if config.evaluate_heldout:
        final_evaluation["heldout"] = _evaluate_records(
            dataset=dataset,
            records=heldout,
            bridge_path=bridge_path,
            dotnet=dotnet,
            episode_steps=config.evaluation_steps,
            horizon_years=config.horizon_years,
            reward_scale=config.reward_scale,
            policy="learned",
            model=model,
            device=device,
        )
    comparisons = {
        key: _compare_evaluations(final_evaluation[key], option_a[key]) for key in final_evaluation
    }
    identities = _seed_checkpoint_identities(dataset, bridge_path, seed, best_update)
    checkpoint = _save_and_verify_campaign_checkpoint(
        path=checkpoint_path,
        model=model,
        trainer=trainer,
        actor_config=actor_config,
        ppo_config=ppo_config,
        pilot_config=config,
        normalizer=dataset.normalizer,
        identities=identities,
        observation=last_batch.observations[0],
        continuation_batch=last_batch,
        device=device,
    )
    maximum_kl = max(row["ppo"]["post_update_approx_kl"] for row in learning_curve)
    mean_clip = float(np.mean([row["ppo"]["post_update_clip_fraction"] for row in learning_curve]))
    critic_start = float(learning_curve[0]["ppo"]["post_update_value_mse"])
    critic_end = float(learning_curve[-1]["ppo"]["post_update_value_mse"])
    actor_activation_healthy = _activation_probe_healthy(
        selected_activation_probe,
        network="actor",
    ) and _activation_probe_healthy(final_activation_probe, network="actor")
    critic_activation_healthy = _activation_probe_healthy(
        selected_activation_probe,
        network="critic",
    ) and _activation_probe_healthy(final_activation_probe, network="critic")
    if config.campaign_scope == "broad-domain":
        active_heads: list[int | str] = [0, 1, 2, 3, "movement"]
        # Every head is active somewhere in the broad-domain cohort, so audit
        # gradients on the rows where each head is structurally absent.
        inactive_gradient = float(inactive_head_audit["max_abs_gradient"])
    else:
        active_heads = [0, 2]
        inactive_gradient = max(
            inactive_head_gradient_max[1],
            inactive_head_gradient_max[3],
            movement_head_gradient_max,
        )
    contract_failure_keys = (
        "illegal_action_count",
        "requested_sample_mismatches",
        "non_finite_count",
        "replay_failures",
        "reward_denominator_violations",
    )
    if config.campaign_scope == "cultivation-only":
        contract_failure_keys += ("non_stay_requested_count",)
    validation_gates = {
        "contract_and_numerics": all(totals[key] == 0 for key in contract_failure_keys),
        "structural_scope_preserved": True,
        "inactive_heads_gradient_inert": inactive_gradient <= 1e-12,
        "checkpoint_reload_parity": checkpoint["max_abs_error"] == 0.0,
        "checkpoint_continuation_parity": (
            checkpoint["stochastic_policy_max_abs_error"] == 0.0
            and checkpoint["continuation_model_max_abs_error"] == 0.0
            and checkpoint["continuation_metrics_max_abs_error"] == 0.0
            and checkpoint["value_normalization_state_reloaded"]
        ),
        "ppo_kl_stable": maximum_kl <= 0.15 and mean_clip <= 0.60,
        "critic_finite": math.isfinite(critic_start) and math.isfinite(critic_end),
        "actor_activations_healthy": actor_activation_healthy,
        "critic_activations_healthy": critic_activation_healthy,
        "selected_checkpoint_critic_health": bool(selected_critic_health["eligible"]),
        "final_checkpoint_critic_health": bool(final_critic_health["eligible"]),
    }
    development_comparison = comparisons["development"]
    self_sufficiency = final_evaluation["development"]["self_sufficiency"]
    family_relative = development_comparison["by_family_mean_relative_difference"]
    promotion_gates = {
        "selected_checkpoint_not_worse_than_initial_probe": (
            best_score >= initial_development_probe_score - 1e-12
        ),
        "development_mean_relative_return_at_least_minus_two_percent": (
            development_comparison["relative_return_difference"]["mean"] >= -0.02
        ),
        "development_median_relative_return_at_or_above_baseline": (
            development_comparison["relative_return_difference"]["p50"] >= 0.0
        ),
        "development_all_family_mean_relative_returns_at_least_minus_five_percent": (
            bool(family_relative) and min(family_relative.values()) >= -0.05
        ),
        "development_trade_contrast_evaluable": bool(
            self_sufficiency["trade_contrast_evaluable"]
        ),
        "no_universal_self_sufficiency_signal": (
            self_sufficiency["universal_self_sufficiency_signal"] is False
        ),
    }
    return {
        "seed": seed,
        "status": (
            "PASS"
            if all(validation_gates.values()) and all(promotion_gates.values())
            else ("NO_GO" if all(validation_gates.values()) else "FAIL")
        ),
        "validation_status": "PASS" if all(validation_gates.values()) else "FAIL",
        "promotion_status": (
            "PASS"
            if all(validation_gates.values()) and all(promotion_gates.values())
            else "NO_GO"
        ),
        "selected_update": best_update,
        "initial_development_probe_score": initial_development_probe_score,
        "selected_development_probe_score": best_score,
        "selected_minus_initial_development_probe_score": (
            best_score - initial_development_probe_score
        ),
        "gates": {**validation_gates, **promotion_gates},
        "validation_gates": validation_gates,
        "promotion_gates": promotion_gates,
        "training": {
            "transitions": sum(int(row["transitions"]) for row in learning_curve),
            "elapsed_seconds": training_seconds,
            "learning_curve": learning_curve,
            "evaluation_curve": evaluation_curve,
            "visitation": {
                "snapshots_by_family": dict(sorted(visitation_snapshots.items())),
                "transitions_by_family": dict(sorted(visitation_transitions.items())),
                "episodes_by_stratum": dict(sorted(strata_visitation.items())),
            },
            "contract_totals": dict(totals),
            "inactive_head_gradient_max": inactive_gradient,
            "inactive_head_gradient_audit": inactive_head_audit,
            "inactive_head_gradient_gate": (
                "per-state-structural-mask-gradient-audit"
                if config.campaign_scope == "broad-domain"
                else "globally-inactive-heads"
            ),
            "per_head_gradient_max": inactive_head_gradient_max,
            "movement_head_gradient_max": movement_head_gradient_max,
            "active_heads": active_heads,
            "maximum_post_update_approx_kl": maximum_kl,
            "mean_post_update_clip_fraction": mean_clip,
            "critic_first_post_update_mse": critic_start,
            "critic_final_post_update_mse": critic_end,
            "hidden_activation_diagnostics": {
                "initial_policy": initial_activation_diagnostics,
                "final_policy": final_activation_probe["train"],
                "selected_policy": selected_activation_diagnostics,
                "tanh_saturation_trigger": 0.10,
                "tanh_derivative_floor": 0.10,
                "layernorm_ablation_recommended": any(
                    float(layer["activation"]["abs_at_or_above_0_95_rate"]) >= 0.10
                    for network in ("actor", "critic")
                    for layer in selected_activation_diagnostics[network]["layers"]
                ),
                "fixed_probe": {
                    "definition": (
                        "first TRAIN rollout observations plus deterministic reset "
                        "observations from one DEVELOPMENT snapshot per family"
                    ),
                    "activation_curve": activation_probe_curve,
                    "final": final_activation_probe,
                    "selected": selected_activation_probe,
                    "critic_fit_targets": "frozen returns from the first TRAIN rollout batch",
                    "critic_fit_promotion_use": False,
                    "critic_fit_warning": (
                        "historical diagnostic only: bootstrapped value targets become stale as "
                        "the continuation-value estimate changes"
                    ),
                    "critic_fit_curve": critic_fit_curve,
                    "final_critic_fit": final_critic_fit,
                    "selected_critic_fit": selected_critic_fit,
                },
            },
            "critic_health_screen": {
                "definition": (
                    "preceding eight ordinary rollout batches, measured before each PPO update"
                ),
                "selected": selected_critic_health,
                "final": final_critic_health,
                "shadow_health_eligible_checkpoint": shadow_health_checkpoint,
            },
            "terminal_checkpoint": terminal_checkpoint,
            "timing": _timing_report(timings, learning_curve, training_seconds),
        },
        "evaluation": final_evaluation,
        "baseline_comparison": comparisons,
        "checkpoint": checkpoint,
    }


def _collect_update_parallel(
    *,
    envs: tuple[AnthropoceneElderEnv, ...],
    dataset: CampaignDataset,
    records: tuple[SnapshotRecord, ...],
    model: ElderActorCritic,
    trainer: PPOTrainer,
    device: torch.device,
    config: PilotConfig,
    timings: _Timing,
    neural_seed: int,
    update: int,
) -> dict[str, Any]:
    """Collect independent snapshot groups across persistent native bridge workers."""

    if len(envs) == 1:
        before = timings.collection_seconds
        result = _collect_update(
            env=envs[0],
            dataset=dataset,
            records=records,
            model=model,
            trainer=trainer,
            device=device,
            config=config,
            timings=timings,
            policy_seed=_collection_worker_seed(neural_seed, update, 0),
        )
        elapsed = timings.collection_seconds - before
        timings.collector_worker_seconds += elapsed
        timings.collector_workers_peak = max(timings.collector_workers_peak, 1)
        result["collector_workers"] = 1
        return result
    if len(envs) > len(records):
        raise ValueError("Collector workers cannot exceed base snapshot groups")

    # Resolve blobs and update the cache access ledger before worker threads
    # touch the read-only in-memory selection cache.
    for record in records:
        dataset.selection(record)

    quotient, remainder = divmod(len(records), len(envs))
    chunks: list[tuple[SnapshotRecord, ...]] = []
    cursor = 0
    for worker in range(len(envs)):
        size = quotient + int(worker < remainder)
        chunks.append(records[cursor : cursor + size])
        cursor += size

    lock = threading.Lock()
    local_timings = [_Timing() for _ in envs]
    replay_offsets: list[int] = []
    episode_cursor = 0
    for chunk in chunks:
        replay_offsets.append(episode_cursor)
        episode_cursor += len(chunk) * config.rollouts_per_snapshot

    started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=len(envs),
        thread_name_prefix="elder-collector",
    ) as executor:
        futures = [
            executor.submit(
                _collect_update,
                env=env,
                dataset=dataset,
                records=chunk,
                model=model,
                trainer=trainer,
                device=device,
                config=config,
                timings=local_timing,
                policy_seed=_collection_worker_seed(neural_seed, update, worker),
                inference_lock=lock,
                replay_episode_offset=replay_offsets[worker],
            )
            for worker, (env, chunk, local_timing) in enumerate(
                zip(envs, chunks, local_timings, strict=True)
            )
        ]
        collected = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - started

    timings.native_step_seconds += sum(row.native_step_seconds for row in local_timings)
    timings.bridge_step_seconds += sum(row.bridge_step_seconds for row in local_timings)
    timings.policy_inference_seconds += sum(
        row.policy_inference_seconds for row in local_timings
    )
    timings.collection_seconds += wall_seconds
    timings.collector_worker_seconds += sum(row.collection_seconds for row in local_timings)
    timings.collector_workers_peak = max(timings.collector_workers_peak, len(envs))
    return _merge_collected_updates(collected, len(envs))


def _merge_collected_updates(
    collected: list[dict[str, Any]],
    worker_count: int,
) -> dict[str, Any]:
    batches = [row.pop("batch") for row in collected]
    batch = RolloutBatch(
        observations=torch.cat([row.observations for row in batches]),
        continuous_actions=torch.cat([row.continuous_actions for row in batches]),
        movement_indices=torch.cat([row.movement_indices for row in batches]),
        old_log_prob=torch.cat([row.old_log_prob for row in batches]),
        advantages=torch.cat([row.advantages for row in batches]),
        returns=torch.cat([row.returns for row in batches]),
    )
    transitions = sum(int(row["transitions"]) for row in collected)
    episode_counts = [sum(row["snapshot_visitation"].values()) for row in collected]
    episodes = sum(episode_counts)

    def merged_counter(key: str) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for row in collected:
            counter.update({name: int(value) for name, value in row[key].items()})
        return dict(sorted(counter.items()))

    snapshot_visitation = merged_counter("snapshot_visitation")
    families = sorted(snapshot_visitation)
    family_returns: dict[str, float] = {}
    for family in families:
        numerator = 0.0
        denominator = 0
        for row in collected:
            count = int(row["snapshot_visitation"].get(family, 0))
            if count:
                numerator += row["discounted_scaled_return_by_family"][family] * count
                denominator += count
        family_returns[family] = numerator / denominator

    def transition_weighted(key: str) -> float:
        return sum(
            float(row[key]) * int(row["transitions"]) for row in collected
        ) / max(transitions, 1)

    action_channels: dict[str, dict[str, float]] = {}
    for channel in ("requested", "accepted", "realized"):
        action_channels[channel] = {
            head: sum(
                float(row["actions"][channel][head]) * int(row["transitions"])
                for row in collected
            )
            / max(transitions, 1)
            for head in (
                "field_operation",
                "herd_retention",
                "development_budget",
                "field_development_share",
            )
        }

    group_rows = [row["group_relative_advantages"] for row in collected]
    group_count = sum(int(row["count"]) for row in group_rows)

    def pooled(mean_key: str, std_key: str) -> tuple[float | None, float | None]:
        if not group_count:
            return None, None
        total = sum(float(row[mean_key]) * int(row["count"]) for row in group_rows)
        mean = total / group_count
        second = sum(
            int(row["count"])
            * (float(row[std_key]) ** 2 + float(row[mean_key]) ** 2)
            for row in group_rows
        ) / group_count
        return mean, max(second - mean * mean, 0.0) ** 0.5

    original_mean, original_std = pooled(
        "first_decision_original_mean",
        "first_decision_original_std",
    )
    adjusted_mean, adjusted_std = pooled(
        "adjusted_mean",
        "first_decision_adjusted_std",
    )
    total_keys = (
        "illegal_action_count",
        "requested_sample_mismatches",
        "non_stay_requested_count",
        "movement_request_count",
        "non_finite_count",
        "replay_failures",
        "reward_denominator_violations",
        "replay_verified_episodes",
        "replay_skipped_episodes",
    )
    return {
        "batch": batch,
        "transitions": transitions,
        "collector_workers": worker_count,
        "snapshot_visitation": snapshot_visitation,
        "transition_visitation": merged_counter("transition_visitation"),
        "strata_visitation": merged_counter("strata_visitation"),
        "discounted_scaled_return_by_family": family_returns,
        "discounted_scaled_return_mean": sum(
            float(row["discounted_scaled_return_mean"]) * episode_count
            for row, episode_count in zip(collected, episode_counts, strict=True)
        )
        / max(episodes, 1),
        "actions": action_channels,
        "intervention_transition_rate": transition_weighted(
            "intervention_transition_rate"
        ),
        "intervention_mean_count": transition_weighted("intervention_mean_count"),
        "requested_to_accepted_mean_l2": transition_weighted(
            "requested_to_accepted_mean_l2"
        ),
        "accepted_to_realized_mean_l2": transition_weighted(
            "accepted_to_realized_mean_l2"
        ),
        "normalizer_saturation_rate": transition_weighted(
            "normalizer_saturation_rate"
        ),
        "group_relative_advantages": {
            "enabled": bool(group_count),
            "rollouts_per_snapshot": group_rows[0]["rollouts_per_snapshot"],
            "first_decision_original_mean": original_mean,
            "first_decision_original_std": original_std,
            "first_decision_adjusted_std": adjusted_std,
            "adjusted_mean": adjusted_mean,
            "count": group_count,
        },
        "worker_diagnostics": [
            {
                "worker": worker,
                "transitions": int(row["transitions"]),
                "episodes": episode_counts[worker],
            }
            for worker, row in enumerate(collected)
        ],
        **{key: sum(int(row[key]) for row in collected) for key in total_keys},
    }


def _collection_worker_seed(neural_seed: int, update: int, worker: int) -> int:
    material = f"elder-policy-worker-v1:{neural_seed}:{update}:{worker}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _episode_policy_seed(policy_seed: int, episode_index: int, snapshot_id: str) -> int:
    material = f"{policy_seed}:{episode_index}:{snapshot_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _collect_update(
    *,
    env: AnthropoceneElderEnv,
    dataset: CampaignDataset,
    records: tuple[SnapshotRecord, ...],
    model: ElderActorCritic,
    trainer: PPOTrainer,
    device: torch.device,
    config: PilotConfig,
    timings: _Timing,
    policy_seed: int | None = None,
    inference_lock: threading.Lock | None = None,
    replay_episode_offset: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()
    observations: list[Tensor] = []
    actions: list[Tensor] = []
    movement_indices: list[Tensor] = []
    log_probabilities: list[Tensor] = []
    advantages_all: list[np.ndarray] = []
    returns_all: list[np.ndarray] = []
    action_channels: dict[str, list[list[float]]] = {
        name: [[], [], [], []] for name in ("requested", "accepted", "realized")
    }
    family_returns: dict[str, list[float]] = defaultdict(list)
    snapshot_visitation: Counter[str] = Counter()
    transition_visitation: Counter[str] = Counter()
    strata_visitation: Counter[str] = Counter()
    intervention_flags: list[bool] = []
    intervention_counts: list[int] = []
    requested_accepted: list[float] = []
    accepted_realized: list[float] = []
    totals: Counter[str] = Counter()
    saturation_count = 0
    saturation_denominator = 0
    transition_count = 0
    original_first_advantages: list[float] = []
    group_relative_first_advantages: list[float] = []

    episode_records = tuple(
        record
        for record in records
        for _ in range(config.rollouts_per_snapshot)
    )
    for episode_index, record in enumerate(episode_records):
        policy_generator: torch.Generator | None = None
        if policy_seed is not None:
            policy_generator = torch.Generator(device=device.type)
            policy_generator.manual_seed(
                _episode_policy_seed(policy_seed, episode_index, record.snapshot_id)
            )
        selection = dataset.selection(record)
        observation, info = env.reset(
            options=_reset_request(
                selection,
                config.episode_steps,
                config.campaign_scope,
            )
        )
        _validate_campaign_reset(_plain_mapping(info["native_result"]), selection)
        raw = _raw_observation(info)
        _assert_campaign_scope(
            raw,
            campaign_scope=config.campaign_scope,
            snapshot_id=record.snapshot_id,
            phase="reset",
        )
        finite, saturated, denominator = _observation_guard(dataset.normalizer, raw)
        totals["non_finite_count"] += 0 if finite else 1
        saturation_count += saturated
        saturation_denominator += denominator
        snapshot_visitation[record.scenario_family] += 1
        for tag in record.actual_strata.tags:
            strata_visitation[tag] += 1

        episode_observations: list[Tensor] = []
        episode_actions: list[Tensor] = []
        episode_movements: list[Tensor] = []
        episode_log_probabilities: list[Tensor] = []
        values: list[float] = []
        next_values: list[float] = []
        rewards: list[float] = []
        elapsed_years: list[float] = []
        terminated_flags: list[bool] = []
        truncated_flags: list[bool] = []
        episode_return = 0.0
        discount = 1.0
        reset_state = _mapping(_plain_mapping(info["native_result"]), "StateDiagnostics")
        prior_population = int(_pick(reset_state, "Population"))

        for _ in range(config.episode_steps):
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
            inference_started = _timing_start(device)
            with (
                inference_lock if inference_lock is not None else nullcontext(),
                torch.no_grad(),
            ):
                sampled = model.actor.sample(
                    observation_tensor.unsqueeze(0),
                    generator=policy_generator,
                )
                value = trainer.value(observation_tensor.unsqueeze(0))[0]
            timings.policy_inference_seconds += _timing_stop(inference_started, device)
            continuous = sampled.continuous[0].detach()
            movement = sampled.movement_index[0].detach()
            old_log_probability = sampled.log_prob[0].detach()
            if not _all_finite(continuous, old_log_probability, value):
                totals["non_finite_count"] += 1
            movement_value = int(movement.cpu())
            if not info["action_mask"].movement_candidates[movement_value]:
                totals["illegal_action_count"] += 1
            bridge_started = time.perf_counter()
            next_observation, _, terminated, truncated, next_info = env.step(
                {
                    "continuous": continuous.cpu().numpy(),
                    "movement_index": movement_value,
                }
            )
            timings.bridge_step_seconds += time.perf_counter() - bridge_started
            result = _plain_mapping(next_info["native_result"])
            timings.native_step_seconds += float(_pick(result, "NativeStepSeconds"))
            requested = _mapping(result, "Requested")
            accepted = _mapping(result, "Accepted")
            realized = _mapping(result, "Realized")
            if not _requested_matches_sample(requested, continuous.cpu().numpy(), movement_value):
                totals["requested_sample_mismatches"] += 1
            if int(_pick(requested, "MovementCandidate")) != -1:
                totals["movement_request_count"] += 1
                if config.campaign_scope == "cultivation-only":
                    totals["non_stay_requested_count"] += 1
            for index, key in enumerate(
                (
                    "RetainedFieldOperation",
                    "RetainedHerdFraction",
                    "DevelopmentBudget",
                    "FieldDevelopmentShare",
                )
            ):
                action_channels["requested"][index].append(float(_pick(requested, key)))
                action_channels["accepted"][index].append(float(_pick(accepted, key)))
                action_channels["realized"][index].append(float(_pick(realized, key)))
            requested_accepted.append(_action_l2(requested, accepted))
            accepted_realized.append(_action_l2(accepted, realized))
            interventions = _sequence(result, "Interventions")
            intervention_flags.append(bool(interventions))
            intervention_counts.append(len(interventions))
            reward = _mapping(result, "Reward")
            elapsed = float(_pick(result, "ElapsedYears"))
            welfare_rate = float(_pick(reward, "WelfareRate"))
            physical_reward = discounted_interval_reward(
                welfare_rate, elapsed, config.horizon_years
            )
            scaled_reward = physical_reward * config.reward_scale
            episode_return += discount * scaled_reward
            discount *= physical_gamma(elapsed, config.horizon_years)
            next_tensor = torch.as_tensor(next_observation, dtype=torch.float32, device=device)
            with (
                inference_lock if inference_lock is not None else nullcontext(),
                torch.no_grad(),
            ):
                next_value = (
                    0.0
                    if terminated
                    else float(trainer.value(next_tensor.unsqueeze(0))[0])
                )
            if _reward_denominator_violation(reward, prior_population, elapsed):
                totals["reward_denominator_violations"] += 1
            state = _mapping(result, "StateDiagnostics")
            if bool(_pick(state, "Active")):
                prior_population = int(_pick(state, "Population"))
            raw_next = _raw_observation(next_info)
            _assert_campaign_scope(
                raw_next,
                campaign_scope=config.campaign_scope,
                snapshot_id=record.snapshot_id,
                phase=f"post-step-{len(rewards) + 1}",
            )
            finite, saturated, denominator = _observation_guard(dataset.normalizer, raw_next)
            totals["non_finite_count"] += 0 if finite else 1
            saturation_count += saturated
            saturation_denominator += denominator

            episode_observations.append(observation_tensor.detach())
            episode_actions.append(continuous)
            episode_movements.append(movement)
            episode_log_probabilities.append(old_log_probability)
            values.append(float(value))
            next_values.append(next_value)
            rewards.append(scaled_reward)
            elapsed_years.append(elapsed)
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            transition_count += 1
            transition_visitation[record.scenario_family] += 1
            observation = next_observation
            info = next_info
            if terminated or truncated:
                break

        if (
            replay_episode_offset + episode_index
        ) % config.replay_verification_interval == 0:
            replay = env.verify_replay()
            totals["replay_verified_episodes"] += 1
            if int(replay.get("steps", 0)) != len(rewards) or not rewards:
                totals["replay_failures"] += 1
        else:
            totals["replay_skipped_episodes"] += 1
        episode_advantages, episode_returns = generalized_advantage_estimate(
            np.asarray(rewards, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            np.asarray(next_values, dtype=np.float64),
            np.asarray(elapsed_years, dtype=np.float64),
            np.asarray(terminated_flags, dtype=bool),
            np.asarray(truncated_flags, dtype=bool),
            horizon_years=config.horizon_years,
            lambda_ref=config.lambda_ref,
            lambda_reference_years=config.lambda_reference_years,
        )
        observations.extend(episode_observations)
        actions.extend(episode_actions)
        movement_indices.extend(episode_movements)
        log_probabilities.extend(episode_log_probabilities)
        advantages_all.append(episode_advantages)
        returns_all.append(episode_returns)
        family_returns[record.scenario_family].append(episode_return)

    if config.group_relative_first_decision:
        (
            advantages_all,
            original_first_advantages,
            group_relative_first_advantages,
        ) = _apply_group_relative_first_decision(
            advantages_all,
            config.rollouts_per_snapshot,
        )

    timings.collection_seconds += time.perf_counter() - started
    batch = RolloutBatch(
        observations=torch.stack(observations),
        continuous_actions=torch.stack(actions),
        movement_indices=torch.stack(movement_indices).to(torch.long),
        old_log_prob=torch.stack(log_probabilities),
        advantages=torch.as_tensor(
            np.concatenate(advantages_all), dtype=torch.float32, device=device
        ),
        returns=torch.as_tensor(np.concatenate(returns_all), dtype=torch.float32, device=device),
    )
    return {
        "batch": batch,
        "transitions": transition_count,
        "snapshot_visitation": dict(sorted(snapshot_visitation.items())),
        "transition_visitation": dict(sorted(transition_visitation.items())),
        "strata_visitation": dict(sorted(strata_visitation.items())),
        "discounted_scaled_return_by_family": {
            family: float(np.mean(values)) for family, values in sorted(family_returns.items())
        },
        "discounted_scaled_return_mean": float(
            np.mean([value for values in family_returns.values() for value in values])
        ),
        "actions": {
            channel: {
                "field_operation": float(np.mean(values[0])),
                "herd_retention": float(np.mean(values[1])),
                "development_budget": float(np.mean(values[2])),
                "field_development_share": float(np.mean(values[3])),
            }
            for channel, values in action_channels.items()
        },
        "intervention_transition_rate": float(np.mean(intervention_flags)),
        "intervention_mean_count": float(np.mean(intervention_counts)),
        "requested_to_accepted_mean_l2": float(np.mean(requested_accepted)),
        "accepted_to_realized_mean_l2": float(np.mean(accepted_realized)),
        "normalizer_saturation_rate": saturation_count / max(saturation_denominator, 1),
        "group_relative_advantages": {
            "enabled": config.group_relative_first_decision,
            "rollouts_per_snapshot": config.rollouts_per_snapshot,
            "first_decision_original_std": (
                float(np.std(original_first_advantages))
                if original_first_advantages
                else None
            ),
            "first_decision_original_mean": (
                float(np.mean(original_first_advantages))
                if original_first_advantages
                else None
            ),
            "first_decision_adjusted_std": (
                float(np.std(group_relative_first_advantages))
                if group_relative_first_advantages
                else None
            ),
            "adjusted_mean": (
                float(np.mean(group_relative_first_advantages))
                if group_relative_first_advantages
                else None
            ),
            "count": len(group_relative_first_advantages),
        },
        **{
            key: int(totals[key])
            for key in (
                "illegal_action_count",
                "requested_sample_mismatches",
                "non_stay_requested_count",
                "movement_request_count",
                "non_finite_count",
                "replay_failures",
                "reward_denominator_violations",
                "replay_verified_episodes",
                "replay_skipped_episodes",
            )
        },
    }


def _apply_group_relative_first_decision(
    advantages_all: list[np.ndarray],
    group_size: int,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """Replace reset-state advantages with a leave-one-out within-snapshot contrast."""

    if group_size < 2 or len(advantages_all) % group_size:
        raise ValueError("Group-relative advantages require complete groups of at least two")
    adjusted_all = [advantages.copy() for advantages in advantages_all]
    original: list[float] = []
    adjusted_first: list[float] = []
    for start in range(0, len(adjusted_all), group_size):
        group = adjusted_all[start : start + group_size]
        if any(len(advantages) == 0 for advantages in group):
            raise ValueError("Group-relative advantage episode cannot be empty")
        first = np.asarray([advantages[0] for advantages in group], dtype=np.float64)
        original.extend(float(value) for value in first)
        total = float(np.sum(first))
        for offset, advantages in enumerate(group):
            leave_one_out = (total - first[offset]) / (group_size - 1)
            advantages[0] = first[offset] - leave_one_out
            adjusted_first.append(float(advantages[0]))
    return adjusted_all, original, adjusted_first


def _fixed_reset_observations(
    *,
    dataset: CampaignDataset,
    records: tuple[SnapshotRecord, ...],
    bridge_path: Path,
    dotnet: str,
    episode_steps: int,
    device: torch.device,
) -> Tensor:
    """Capture deterministic reset observations for architecture diagnostics."""

    if not records:
        raise ValueError("A fixed reset probe requires at least one snapshot")
    env = _make_environment(
        bridge_path=bridge_path,
        dotnet=dotnet,
        selection=dataset.selection(records[0]),
        episode_limit=episode_steps,
        campaign_scope=dataset.campaign_scope,
    )
    observations: list[np.ndarray] = []
    try:
        for record in records:
            selection = dataset.selection(record)
            observation, info = env.reset(
                options=_reset_request(
                    selection,
                    episode_steps,
                    dataset.campaign_scope,
                )
            )
            _validate_campaign_reset(_plain_mapping(info["native_result"]), selection)
            _assert_campaign_scope(
                _raw_observation(info),
                campaign_scope=dataset.campaign_scope,
                snapshot_id=selection.record.snapshot_id,
                phase="fixed-diagnostic-reset",
            )
            observations.append(np.asarray(observation, dtype=np.float32))
    finally:
        env.close()
    return torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device)


def _activation_probe_row(
    *,
    update: int,
    model: ElderActorCritic,
    train_observations: Tensor,
    development_observations: Tensor,
) -> dict[str, Any]:
    return {
        "update": update,
        "train": model.hidden_activation_diagnostics(train_observations),
        "development": model.hidden_activation_diagnostics(development_observations),
    }


@torch.no_grad()
def _critic_probe_diagnostics(
    model: ElderActorCritic,
    observations: Tensor,
    targets: Tensor,
    *,
    trainer: PPOTrainer | None = None,
) -> dict[str, float]:
    prediction = model.critic(observations) if trainer is None else trainer.value(observations)
    return critic_fit_diagnostics(prediction, targets)


def _inactive_head_gradient_audit(
    model: ElderActorCritic,
    batch: RolloutBatch,
) -> dict[str, Any]:
    """Measure head gradients only on rows where native structure masks that head."""

    actor = model.actor
    active = structural_head_mask(batch.observations)
    per_head: dict[str, dict[str, Any]] = {}
    maxima: list[float] = []
    for head in range(4):
        rows = ~active[:, head]
        count = int(torch.sum(rows).detach().cpu())
        gradient = 0.0
        if count:
            actor.zero_grad(set_to_none=True)
            evaluated = actor.evaluate_actions(
                batch.observations[rows],
                batch.continuous_actions[rows],
                batch.movement_indices[rows],
            )
            (evaluated.log_prob.mean() + evaluated.entropy.mean()).backward()
            weight = actor.continuous_mean.weight.grad
            bias = actor.continuous_mean.bias.grad
            log_std = actor.log_std.grad
            if weight is None or bias is None or log_std is None:
                raise RuntimeError("Inactive-head audit did not produce actor gradients")
            gradient = max(
                float(torch.max(torch.abs(weight[head])).detach().cpu()),
                float(torch.abs(bias[head]).detach().cpu()),
                float(torch.abs(log_std[head]).detach().cpu()),
            )
            maxima.append(gradient)
        per_head[f"continuous_{head}"] = {
            "inactive_rows": count,
            "max_abs_gradient": gradient if count else None,
        }

    legal_movement = movement_mask(batch.observations)
    movement_rows = torch.sum(legal_movement, dim=-1) == 1
    movement_count = int(torch.sum(movement_rows).detach().cpu())
    movement_gradient = 0.0
    if movement_count:
        actor.zero_grad(set_to_none=True)
        evaluated = actor.evaluate_actions(
            batch.observations[movement_rows],
            batch.continuous_actions[movement_rows],
            batch.movement_indices[movement_rows],
        )
        (evaluated.log_prob.mean() + evaluated.entropy.mean()).backward()
        weight = actor.movement_logits.weight.grad
        bias = actor.movement_logits.bias.grad
        if weight is None or bias is None:
            raise RuntimeError("Inactive movement audit did not produce actor gradients")
        movement_gradient = max(
            float(torch.max(torch.abs(weight)).detach().cpu()),
            float(torch.max(torch.abs(bias)).detach().cpu()),
        )
        maxima.append(movement_gradient)
    per_head["movement"] = {
        "inactive_rows": movement_count,
        "max_abs_gradient": movement_gradient if movement_count else None,
    }
    actor.zero_grad(set_to_none=True)
    return {
        "max_abs_gradient": max(maxima, default=0.0),
        "per_head": per_head,
        "threshold": 1e-12,
    }


def _capture_training_state(
    model: ElderActorCritic,
    trainer: PPOTrainer,
) -> dict[str, Any]:
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "trainer": copy.deepcopy(trainer.state_dict()),
        "rng": _capture_rng_state(),
    }


def _critic_health_window(
    learning_curve: list[dict[str, Any]],
    *,
    window: int = 8,
) -> dict[str, Any]:
    rows = learning_curve[-window:]
    explained_variance = np.asarray(
        [row["ppo"]["pre_update_critic_explained_variance"] for row in rows],
        dtype=np.float64,
    )
    normalized_mse = np.asarray(
        [row["ppo"]["pre_update_critic_normalized_mse"] for row in rows],
        dtype=np.float64,
    )
    normalized_bias = np.asarray(
        [row["ppo"]["pre_update_critic_normalized_bias"] for row in rows],
        dtype=np.float64,
    )
    positive_fit = (explained_variance > 0.0) & (normalized_mse < 1.0)
    enough_rows = len(rows) == window
    median_explained_variance = (
        float(np.median(explained_variance)) if len(rows) else None
    )
    median_normalized_mse = float(np.median(normalized_mse)) if len(rows) else None
    return {
        "window": window,
        "observed_updates": len(rows),
        "start_update": int(rows[0]["update"]) if rows else None,
        "end_update": int(rows[-1]["update"]) if rows else None,
        "median_pre_update_explained_variance": median_explained_variance,
        "median_pre_update_normalized_mse": median_normalized_mse,
        "median_abs_pre_update_normalized_bias": (
            float(np.median(np.abs(normalized_bias))) if len(rows) else None
        ),
        "positive_fit_updates": int(np.sum(positive_fit)),
        "required_positive_fit_updates": 6,
        "eligible": bool(
            enough_rows
            and median_explained_variance is not None
            and median_explained_variance >= 0.20
            and median_normalized_mse is not None
            and median_normalized_mse <= 0.80
            and int(np.sum(positive_fit)) >= 6
        ),
        "thresholds": {
            "median_pre_update_explained_variance_min": 0.20,
            "median_pre_update_normalized_mse_max": 0.80,
            "per_update_positive_fit": "EV > 0 and nMSE < 1",
        },
    }


def _network_activation_healthy(
    diagnostics: dict[str, Any],
    *,
    saturation_limit: float,
    derivative_floor: float,
) -> bool:
    for layer in diagnostics["layers"]:
        activation = layer["activation"]
        if float(activation["abs_at_or_above_0_95_rate"]) >= saturation_limit:
            return False
        derivative = activation["mean_tanh_derivative"]
        if derivative is None or float(derivative) <= derivative_floor:
            return False
    return True


def _activation_probe_healthy(
    probe: dict[str, Any],
    *,
    network: Literal["actor", "critic"],
    saturation_limit: float = 0.10,
    derivative_floor: float = 0.10,
) -> bool:
    return all(
        _network_activation_healthy(
            probe[partition][network],
            saturation_limit=saturation_limit,
            derivative_floor=derivative_floor,
        )
        for partition in ("train", "development")
    )


def _evaluate_records(
    *,
    dataset: CampaignDataset,
    records: tuple[SnapshotRecord, ...],
    bridge_path: Path,
    dotnet: str,
    episode_steps: int,
    horizon_years: float,
    reward_scale: float,
    policy: Literal["learned", "option-a"],
    model: ElderActorCritic | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if policy == "learned" and (model is None or device is None):
        raise ValueError("Learned evaluation requires a model and device")
    if not records:
        raise ValueError("Evaluation record set cannot be empty")
    selection = dataset.selection(records[0])
    env = _make_environment(
        bridge_path=bridge_path,
        dotnet=dotnet,
        selection=selection,
        episode_limit=episode_steps,
        campaign_scope=dataset.campaign_scope,
    )
    episodes: list[dict[str, Any]] = []
    was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
    try:
        for record in records:
            episodes.append(
                _evaluate_episode(
                    env=env,
                    selection=dataset.selection(record),
                    model=model,
                    device=device,
                    policy=policy,
                    episode_steps=episode_steps,
                    horizon_years=horizon_years,
                    reward_scale=reward_scale,
                    campaign_scope=dataset.campaign_scope,
                )
            )
    finally:
        env.close()
        if model is not None:
            model.train(was_training)
    return {
        "policy": (
            "feed-forward-elder-ppo" if policy == "learned" else "option-a-through-shared-compiler"
        ),
        "summary": _summarize_episodes(episodes),
        "by_family": _group_episode_summaries(episodes, "scenario_family"),
        "by_world": _group_episode_summaries(episodes, "world_seed"),
        "behavior_slices": _behavior_slices(episodes),
        "self_sufficiency": _self_sufficiency_diagnostic(episodes),
        "episodes": episodes,
    }


def _evaluate_episode(
    *,
    env: AnthropoceneElderEnv,
    selection: CampaignSelection,
    model: ElderActorCritic | None,
    device: torch.device | None,
    policy: Literal["learned", "option-a"],
    episode_steps: int,
    horizon_years: float,
    reward_scale: float,
    campaign_scope: CampaignScope,
) -> dict[str, Any]:
    observation, info = env.reset(
        options=_reset_request(selection, episode_steps, campaign_scope)
    )
    _validate_campaign_reset(_plain_mapping(info["native_result"]), selection)
    _assert_campaign_scope(
        _raw_observation(info),
        campaign_scope=campaign_scope,
        snapshot_id=selection.record.snapshot_id,
        phase="reset",
    )
    reset_result = _plain_mapping(info["native_result"])
    start_state = _mapping(reset_result, "StateDiagnostics")
    prior_population = int(_pick(start_state, "Population"))
    discounted_return = 0.0
    discount = 1.0
    raw_welfare = 0.0
    elapsed_total = 0.0
    intervention_counts: list[int] = []
    intervention_kinds: Counter[str] = Counter()
    requested_accepted: list[float] = []
    accepted_realized: list[float] = []
    entropies: list[float] = []
    saturation_flags: list[bool] = []
    channels: dict[str, list[list[float]]] = {
        name: [[], [], [], []] for name in ("requested", "accepted", "realized")
    }
    state_path: list[dict[str, Any]] = [start_state]
    totals: Counter[str] = Counter()
    fission_events = 0
    trade_total = 0.0
    food_imports = 0.0
    material_tool_imports = 0.0
    migration_total = 0.0
    steps = 0

    for _ in range(episode_steps):
        if policy == "learned":
            assert model is not None and device is not None
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
            with torch.no_grad():
                sampled = model.actor.sample(tensor.unsqueeze(0), deterministic=True)
            continuous = sampled.continuous[0].detach()
            movement = int(sampled.movement_index[0].detach().cpu())
            entropies.append(float(sampled.entropy[0].detach().cpu()))
            if not info["action_mask"].movement_candidates[movement]:
                totals["illegal_action_count"] += 1
            observation, _, terminated, truncated, info = env.step(
                {
                    "continuous": continuous.cpu().numpy(),
                    "movement_index": movement,
                }
            )
        else:
            continuous = None
            movement = 0
            observation, _, terminated, truncated, info = env.option_a_step()
        result = _plain_mapping(info["native_result"])
        requested = _mapping(result, "Requested")
        accepted = _mapping(result, "Accepted")
        realized = _mapping(result, "Realized")
        if continuous is not None and not _requested_matches_sample(
            requested, continuous.cpu().numpy(), movement
        ):
            totals["requested_sample_mismatches"] += 1
        requested_movement = int(_pick(requested, "MovementCandidate"))
        accepted_movement = int(_pick(accepted, "MovementCandidate"))
        realized_movement = int(_pick(realized, "MovementCandidate"))
        if requested_movement != -1:
            totals["movement_request_count"] += 1
            if campaign_scope == "cultivation-only":
                totals["non_stay_requested_count"] += 1
        if accepted_movement != -1:
            totals["movement_accept_count"] += 1
        if realized_movement != -1:
            totals["movement_realized_count"] += 1
        for index, key in enumerate(
            (
                "RetainedFieldOperation",
                "RetainedHerdFraction",
                "DevelopmentBudget",
                "FieldDevelopmentShare",
            )
        ):
            channels["requested"][index].append(float(_pick(requested, key)))
            channels["accepted"][index].append(float(_pick(accepted, key)))
            channels["realized"][index].append(float(_pick(realized, key)))
        saturation_flags.extend(
            [
                channels["requested"][0][-1] <= 0.02 or channels["requested"][0][-1] >= 0.98,
                channels["requested"][2][-1] <= 0.02 or channels["requested"][2][-1] >= 0.98,
            ]
        )
        requested_accepted.append(_action_l2(requested, accepted))
        accepted_realized.append(_action_l2(accepted, realized))
        interventions = _sequence(result, "Interventions")
        intervention_counts.append(len(interventions))
        for intervention in interventions:
            if isinstance(intervention, dict):
                intervention_kinds[str(intervention.get("Kind", "unknown"))] += 1
        reward = _mapping(result, "Reward")
        elapsed = float(_pick(result, "ElapsedYears"))
        welfare_rate = float(_pick(reward, "WelfareRate"))
        interval = discounted_interval_reward(welfare_rate, elapsed, horizon_years)
        discounted_return += discount * interval
        discount *= physical_gamma(elapsed, horizon_years)
        raw_welfare += float(_pick(reward, "UndiscountedIntervalWelfare"))
        elapsed_total += elapsed
        if _reward_denominator_violation(reward, prior_population, elapsed):
            totals["reward_denominator_violations"] += 1
        state = _mapping(result, "StateDiagnostics")
        state_path.append(state)
        if bool(_pick(state, "Active")):
            prior_population = int(_pick(state, "Population"))
            trade_total += float(_pick(state, "DeliveredTrade"))
            food_imports += float(_pick(state, "DeliveredFoodImports"))
            material_tool_imports += float(_pick(state, "DeliveredMaterialToolImports"))
            migration_total += float(_pick(state, "SignedMigration"))
            fission_events += int(bool(_pick(state, "FissionEvent")))
        raw = _raw_observation(info)
        _assert_campaign_scope(
            raw,
            campaign_scope=campaign_scope,
            snapshot_id=selection.record.snapshot_id,
            phase=f"post-step-{steps + 1}",
        )
        finite, _, _ = _observation_guard(selection.normalizer, raw)
        totals["non_finite_count"] += 0 if finite else 1
        steps += 1
        if terminated or truncated:
            break
    replay = env.verify_replay()
    if int(replay.get("steps", 0)) != steps or steps == 0:
        totals["replay_failures"] += 1
    end_state = state_path[-1]
    active_end = bool(_pick(end_state, "Active"))
    return {
        "snapshot_id": selection.record.snapshot_id,
        "scenario_family": selection.record.scenario_family,
        "world_seed": selection.record.world_seed,
        "lineage_id": selection.record.lineage_id,
        "counterfactual_group_id": selection.record.counterfactual_group_id,
        "classification": selection.record.classification.value,
        "tags": list(selection.record.actual_strata.tags),
        "steps": steps,
        "elapsed_years": elapsed_total,
        "discounted_return": discounted_return,
        "discounted_scaled_return": discounted_return * reward_scale,
        "raw_prior_cohort_welfare": raw_welfare,
        "survived": active_end,
        "start_state": start_state,
        "end_state": end_state,
        "delivered_trade": trade_total,
        "delivered_food_imports": food_imports,
        "delivered_material_tool_imports": material_tool_imports,
        "signed_migration": migration_total,
        "fission_events": fission_events,
        "movement": {
            "requested_count": int(totals["movement_request_count"]),
            "accepted_count": int(totals["movement_accept_count"]),
            "realized_count": int(totals["movement_realized_count"]),
            "requested_rate": totals["movement_request_count"] / max(steps, 1),
            "accepted_rate": totals["movement_accept_count"] / max(steps, 1),
            "realized_rate": totals["movement_realized_count"] / max(steps, 1),
        },
        "actions": {
            channel: {
                "field_operation": float(np.mean(values[0])),
                "herd_retention": float(np.mean(values[1])),
                "development_budget": float(np.mean(values[2])),
                "field_development_share": float(np.mean(values[3])),
            }
            for channel, values in channels.items()
        },
        "reset_actions": {
            channel: {
                "field_operation": values[0][0],
                "herd_retention": values[1][0],
                "development_budget": values[2][0],
                "field_development_share": values[3][0],
            }
            for channel, values in channels.items()
        },
        "action_entropy": float(np.mean(entropies)) if entropies else None,
        "action_saturation_rate": float(np.mean(saturation_flags)),
        "intervention_transition_rate": float(
            np.mean([count > 0 for count in intervention_counts])
        ),
        "intervention_mean_count": float(np.mean(intervention_counts)),
        "intervention_kinds": dict(sorted(intervention_kinds.items())),
        "requested_to_accepted_mean_l2": float(np.mean(requested_accepted)),
        "accepted_to_realized_mean_l2": float(np.mean(accepted_realized)),
        "contract": {
            "illegal_action_count": int(totals["illegal_action_count"]),
            "requested_sample_mismatches": int(totals["requested_sample_mismatches"]),
            "non_stay_requested_count": int(totals["non_stay_requested_count"]),
            "non_finite_count": int(totals["non_finite_count"]),
            "replay_failures": int(totals["replay_failures"]),
            "reward_denominator_violations": int(totals["reward_denominator_violations"]),
        },
    }


def _summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    def values(path: tuple[str, ...], default: float = 0.0) -> list[float]:
        result: list[float] = []
        for episode in episodes:
            value: Any = episode
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    value = default
                    break
                value = value[key]
            result.append(float(value if value is not None else default))
        return result

    fields = {
        "discounted_return": ("discounted_return",),
        "discounted_scaled_return": ("discounted_scaled_return",),
        "raw_prior_cohort_welfare": ("raw_prior_cohort_welfare",),
        "final_qol": ("end_state", "QoL"),
        "final_population": ("end_state", "Population"),
        "final_food": ("end_state", "FoodNeedsMet"),
        "final_water": ("end_state", "WaterNeedsMet"),
        "final_clothing": ("end_state", "ClothingCoverage"),
        "final_housing": ("end_state", "HousingCoverage"),
        "final_field_q": ("end_state", "FieldQ"),
        "final_retained_field_capital": ("end_state", "RetainedFieldCapital"),
        "final_field_operation": ("end_state", "FieldOperation"),
        "final_maintenance_worker_years": ("end_state", "MaintenanceWorkerYears"),
        "final_food_reserve_years": ("end_state", "FoodReserveYears"),
        "final_hand_tools_per_capita": ("end_state", "HandToolsPerCapita"),
        "final_farm_tools_per_capita": ("end_state", "FarmToolsPerCapita"),
        "final_materials_per_capita": ("end_state", "MaterialsPerCapita"),
        "delivered_trade": ("delivered_trade",),
        "delivered_food_imports": ("delivered_food_imports",),
        "signed_migration": ("signed_migration",),
        "fission_events": ("fission_events",),
        "requested_field_operation": ("actions", "requested", "field_operation"),
        "accepted_field_operation": ("actions", "accepted", "field_operation"),
        "realized_field_operation": ("actions", "realized", "field_operation"),
        "requested_herd_retention": ("actions", "requested", "herd_retention"),
        "accepted_herd_retention": ("actions", "accepted", "herd_retention"),
        "realized_herd_retention": ("actions", "realized", "herd_retention"),
        "requested_development_budget": (
            "actions",
            "requested",
            "development_budget",
        ),
        "reset_requested_field_operation": (
            "reset_actions",
            "requested",
            "field_operation",
        ),
        "reset_requested_development_budget": (
            "reset_actions",
            "requested",
            "development_budget",
        ),
        "accepted_development_budget": (
            "actions",
            "accepted",
            "development_budget",
        ),
        "realized_development_budget": (
            "actions",
            "realized",
            "development_budget",
        ),
        "requested_field_development_share": (
            "actions",
            "requested",
            "field_development_share",
        ),
        "accepted_field_development_share": (
            "actions",
            "accepted",
            "field_development_share",
        ),
        "realized_field_development_share": (
            "actions",
            "realized",
            "field_development_share",
        ),
        "movement_requested_rate": ("movement", "requested_rate"),
        "movement_accepted_rate": ("movement", "accepted_rate"),
        "movement_realized_rate": ("movement", "realized_rate"),
        "intervention_transition_rate": ("intervention_transition_rate",),
        "intervention_mean_count": ("intervention_mean_count",),
        "requested_to_accepted_mean_l2": ("requested_to_accepted_mean_l2",),
        "accepted_to_realized_mean_l2": ("accepted_to_realized_mean_l2",),
        "action_saturation_rate": ("action_saturation_rate",),
    }
    summary: dict[str, Any] = {
        "episodes": len(episodes),
        "survival_rate": float(np.mean([episode["survived"] for episode in episodes])),
        "contract_failure_count": sum(
            sum(int(value) for value in episode["contract"].values()) for episode in episodes
        ),
    }
    for name, path in fields.items():
        summary[name] = _distribution(values(path))
    entropies = [episode["action_entropy"] for episode in episodes]
    if all(value is not None for value in entropies):
        summary["action_entropy"] = _distribution([float(value) for value in entropies])
    return summary


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _group_episode_summaries(episodes: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[str(episode[key])].append(episode)
    return {group: _summarize_episodes(rows) for group, rows in sorted(grouped.items())}


def _behavior_slices(episodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    predicates: dict[str, Callable[[dict[str, Any]], bool]] = {
        "low-q": lambda row: float(_pick(row["start_state"], "FieldQ")) < 0.3,
        "high-q": lambda row: float(_pick(row["start_state"], "FieldQ")) >= 0.5,
        "high-maintenance": lambda row: (
            float(_pick(row["start_state"], "MaintenanceWorkerYears")) >= 10
        ),
        "trade-dependent": lambda row: "trade-dependent" in row["tags"],
        "isolated": lambda row: "isolated" in row["tags"],
        "immigration": lambda row: "immigration" in row["tags"],
        "emigration": lambda row: "emigration" in row["tags"],
        "fission-opportunity": lambda row: (
            float(_pick(row["start_state"], "FissionExitOpportunity")) > 0
        ),
        "young-memory": lambda row: "young-memory" in row["tags"],
        "mature-memory": lambda row: "mature-memory" in row["tags"],
        "primitive-frontier": lambda row: "primitive-frontier" in row["tags"],
        "plough-frontier": lambda row: "plough-frontier" in row["tags"],
    }
    result: dict[str, dict[str, Any]] = {}
    for name, predicate in predicates.items():
        selected = [episode for episode in episodes if predicate(episode)]
        if selected:
            result[name] = _summarize_episodes(selected)
    return result


def _self_sufficiency_diagnostic(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    requested = np.asarray(
        [episode["actions"]["requested"]["field_operation"] for episode in episodes],
        dtype=np.float64,
    )
    trade_dependent = np.asarray(
        [
            episode["actions"]["requested"]["field_operation"]
            for episode in episodes
            if "trade-dependent" in episode["tags"]
        ],
        dtype=np.float64,
    )
    isolated = np.asarray(
        [
            episode["actions"]["requested"]["field_operation"]
            for episode in episodes
            if "isolated" in episode["tags"]
        ],
        dtype=np.float64,
    )
    f15_trade_dependent_count = sum(
        episode["scenario_family"] == "F15" and "trade-dependent" in episode["tags"]
        for episode in episodes
    )
    f15_isolated_count = sum(
        episode["scenario_family"] == "F15" and "isolated" in episode["tags"]
        for episode in episodes
    )
    trade_contrast_evaluable = bool(
        f15_trade_dependent_count > 0 and f15_isolated_count > 0
    )
    high_rate = float(np.mean(requested >= 0.9))
    contrast = (
        float(np.mean(trade_dependent) - np.mean(isolated))
        if len(trade_dependent) and len(isolated)
        else None
    )
    return {
        "requested_field_operation_at_or_above_0_9_rate": high_rate,
        "trade_dependent_count": len(trade_dependent),
        "isolated_count": len(isolated),
        "f15_trade_dependent_count": f15_trade_dependent_count,
        "f15_isolated_count": f15_isolated_count,
        "trade_contrast_evaluable": trade_contrast_evaluable,
        "trade_contrast_assessment": (
            "EVALUABLE" if trade_contrast_evaluable else "UNEVALUABLE"
        ),
        "trade_dependent_minus_isolated_mean_field_operation": contrast,
        "universal_self_sufficiency_signal": (
            bool(high_rate >= 0.80 and contrast is not None and contrast >= -0.02)
            if trade_contrast_evaluable
            else None
        ),
    }


def _compare_evaluations(learned: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_by_id = {row["snapshot_id"]: row for row in baseline["episodes"]}
    differences: list[float] = []
    relative: list[float] = []
    by_family: dict[str, list[float]] = defaultdict(list)
    by_family_relative: dict[str, list[float]] = defaultdict(list)
    for row in learned["episodes"]:
        reference = baseline_by_id[row["snapshot_id"]]
        delta = row["discounted_scaled_return"] - reference["discounted_scaled_return"]
        differences.append(delta)
        relative_delta = delta / max(abs(reference["discounted_scaled_return"]), 1e-9)
        relative.append(relative_delta)
        by_family[row["scenario_family"]].append(delta)
        by_family_relative[row["scenario_family"]].append(relative_delta)
    mean_relative = float(np.mean(relative))
    return {
        "matched_snapshots": len(differences),
        "discounted_scaled_return_difference": _distribution(differences),
        "relative_return_difference": _distribution(relative),
        "fraction_at_or_above_baseline": float(np.mean(np.asarray(differences) >= -1e-12)),
        "by_family_mean_difference": {
            family: float(np.mean(values)) for family, values in sorted(by_family.items())
        },
        "by_family_mean_relative_difference": {
            family: float(np.mean(values))
            for family, values in sorted(by_family_relative.items())
        },
        "competitive_with_baseline": bool(mean_relative >= -0.10),
    }


def _cultivation_trajectory(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    update_count = len(seed_reports[0]["training"]["learning_curve"])
    rows: list[dict[str, Any]] = []
    for index in range(update_count):
        requested_field = [
            report["training"]["learning_curve"][index]["actions"]["requested"]["field_operation"]
            for report in seed_reports
        ]
        realized_field = [
            report["training"]["learning_curve"][index]["actions"]["realized"]["field_operation"]
            for report in seed_reports
        ]
        requested_development = [
            report["training"]["learning_curve"][index]["actions"]["requested"][
                "development_budget"
            ]
            for report in seed_reports
        ]
        rows.append(
            {
                "update": index + 1,
                "requested_field_operation_mean": float(np.mean(requested_field)),
                "requested_field_operation_by_seed": requested_field,
                "realized_field_operation_mean": float(np.mean(realized_field)),
                "realized_field_operation_by_seed": realized_field,
                "requested_development_budget_mean": float(np.mean(requested_development)),
                "requested_development_budget_by_seed": requested_development,
            }
        )
    field = np.asarray([row["requested_field_operation_mean"] for row in rows], dtype=np.float64)
    development = np.asarray(
        [row["requested_development_budget_mean"] for row in rows], dtype=np.float64
    )
    return {
        "updates": rows,
        "requested_field_operation": _trend(field),
        "requested_development_budget": _trend(development),
        "steady_increase_in_cultivation_intensity": bool(
            _trend(field)["steady_increase"] and _trend(development)["steady_increase"]
        ),
        "interpretation_rule": (
            "steady requires a positive endpoint change and at least 75% nonnegative "
            "update-to-update changes for both field operation and Development budget"
        ),
    }


def _trend(values: np.ndarray) -> dict[str, Any]:
    deltas = np.diff(values)
    endpoint = float(values[-1] - values[0])
    nonnegative = float(np.mean(deltas >= -1e-3)) if len(deltas) else 0.0
    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
    return {
        "first": float(values[0]),
        "last": float(values[-1]),
        "endpoint_change": endpoint,
        "linear_slope_per_update": slope,
        "nonnegative_step_fraction": nonnegative,
        "steady_increase": bool(endpoint > 0.02 and nonnegative >= 0.75),
    }


def _validation_assessment(seed_reports: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "all_three_seeds_structural_scope_preserved": all(
            report["validation_gates"]["structural_scope_preserved"]
            for report in seed_reports
        ),
        "all_three_seeds_contract_clean": all(
            report["validation_gates"]["contract_and_numerics"]
            for report in seed_reports
        ),
        "all_three_seeds_inactive_heads_inert": all(
            report["validation_gates"]["inactive_heads_gradient_inert"]
            for report in seed_reports
        ),
        "all_three_seeds_checkpoint_parity": all(
            report["validation_gates"]["checkpoint_reload_parity"]
            and report["validation_gates"]["checkpoint_continuation_parity"]
            for report in seed_reports
        ),
        "all_three_seeds_ppo_stable": all(
            report["validation_gates"]["ppo_kl_stable"] for report in seed_reports
        ),
        "all_three_seeds_critic_finite": all(
            report["validation_gates"]["critic_finite"] for report in seed_reports
        ),
        "all_three_seeds_actor_activations_healthy": all(
            report["validation_gates"]["actor_activations_healthy"]
            for report in seed_reports
        ),
        "all_three_seeds_critic_activations_healthy": all(
            report["validation_gates"]["critic_activations_healthy"]
            for report in seed_reports
        ),
        "all_three_selected_checkpoints_critic_healthy": all(
            report["validation_gates"]["selected_checkpoint_critic_health"]
            for report in seed_reports
        ),
        "all_three_final_checkpoints_critic_healthy": all(
            report["validation_gates"]["final_checkpoint_critic_health"]
            for report in seed_reports
        ),
    }


def _promotion_assessment(seed_reports: list[dict[str, Any]]) -> dict[str, bool]:
    gate_names = tuple(seed_reports[0]["promotion_gates"])
    result = {
        f"all_three_seeds_{name}": all(
            report["promotion_gates"][name] for report in seed_reports
        )
        for name in gate_names
    }
    result["all_three_seeds_promotion_ready"] = all(
        report["promotion_status"] == "PASS"
        for report in seed_reports
    )
    return result


def _probe_records(
    records: tuple[SnapshotRecord, ...],
    per_family: int,
    *,
    families: tuple[str, ...] = PILOT_FAMILIES,
) -> tuple[SnapshotRecord, ...]:
    selected: list[SnapshotRecord] = []
    for family in families:
        family_records = [record for record in records if record.scenario_family == family]
        selected.extend(family_records[:per_family])
    return tuple(selected)


def _make_environment(
    *,
    bridge_path: Path,
    dotnet: str,
    selection: CampaignSelection,
    episode_limit: int,
    campaign_scope: CampaignScope = "cultivation-only",
) -> AnthropoceneElderEnv:
    return AnthropoceneElderEnv(
        JsonLineProcessTransport([dotnet, str(bridge_path)]),
        _reset_request(selection, episode_limit, campaign_scope),
        normalizer=selection.normalizer,
    )


def _reset_request(
    selection: CampaignSelection,
    episode_limit: int,
    campaign_scope: CampaignScope = "cultivation-only",
) -> dict[str, Any]:
    record = selection.record
    return {
        "Seed": record.world_seed,
        "RandomStream": 0,
        "Snapshot": selection.snapshot_bytes,
        "ControlledCampId": record.controlled.camp_id,
        "ControlledPlayer": record.controlled.player_id,
        "BackgroundPolicy": 0,
        "Options": {
            "EnvironmentSchemaVersion": ENVIRONMENT_SCHEMA_VERSION,
            "EpisodeDecisionLimit": episode_limit,
            "MovementEnabled": campaign_scope == "broad-domain",
            "SyntheticShocksEnabled": False,
            "IncludeTrainingCriticSidecar": False,
        },
        "ExpectedSimulatorVersion": record.native_simulator_version,
        "ExpectedConfigurationHash": record.configuration_hash,
        "ExpectedObservationSchemaVersion": OBSERVATION_SCHEMA_VERSION,
        "ExpectedActionSchemaVersion": ACTION_SCHEMA_VERSION,
        "ExpectedEnvironmentSchemaVersion": ENVIRONMENT_SCHEMA_VERSION,
    }


def _validate_campaign_reset(reset: dict[str, Any], selection: CampaignSelection) -> None:
    record = selection.record
    checks: dict[str, tuple[Any, Any]] = {
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
        raise RuntimeError(f"Native campaign reset identity mismatch: {mismatches}")


def _assert_cultivation_scope(
    observation: EncodedObservation,
    *,
    snapshot_id: str,
    phase: str,
) -> None:
    """Fail before an excluded learned domain can enter a campaign transition."""

    violations = cultivation_only_scope_violations(observation)
    if violations:
        detail = ", ".join(violations)
        raise RuntimeError(f"Cultivation-only scope escaped for {snapshot_id} at {phase}: {detail}")


def _assert_campaign_scope(
    observation: EncodedObservation,
    *,
    campaign_scope: CampaignScope,
    snapshot_id: str,
    phase: str,
) -> None:
    """Fail before an excluded learned domain can enter a campaign transition."""

    if campaign_scope == "cultivation-only":
        _assert_cultivation_scope(
            observation,
            snapshot_id=snapshot_id,
            phase=phase,
        )
        return
    violations = broad_domain_scope_violations(observation)
    if violations:
        detail = ", ".join(violations)
        raise RuntimeError(
            f"Broad-domain scope escaped for {snapshot_id} at {phase}: {detail}"
        )


def _reward_denominator_violation(
    reward: dict[str, Any], prior_population: int, elapsed: float
) -> bool:
    recorded_population = int(_pick(reward, "PriorPopulation"))
    quality = float(_pick(reward, "RealizedQoL"))
    welfare_rate = float(_pick(reward, "WelfareRate"))
    interval = float(_pick(reward, "UndiscountedIntervalWelfare"))
    expected_rate = max(0, prior_population) * min(1.0, max(0.0, quality))
    return bool(
        recorded_population != prior_population
        or not math.isclose(welfare_rate, expected_rate, rel_tol=1e-12, abs_tol=1e-12)
        or not math.isclose(
            interval, expected_rate * max(0.0, elapsed), rel_tol=1e-12, abs_tol=1e-9
        )
    )


def _campaign_identities(
    dataset: CampaignDataset, bridge_path: Path, project_root: Path
) -> dict[str, Any]:
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
    normalizer_sha = sha256_file(dataset.normalizer_path)[0]
    return {
        "learner_commit": commit,
        "learner_tracked_files_dirty": dirty,
        "simulator_commit": SIMULATOR_COMMIT,
        "snapshot_manifest_sha256": dataset.manifest_sha256,
        "normalizer_partition_id": dataset.normalizer.manifest.partition_id,
        "normalizer_sha256": normalizer_sha,
        "configuration_hash": dataset.normalizer.manifest.configuration_hash,
        "native_build_hash": dataset.normalizer.manifest.build_hash,
        "observation_schema_hash": dataset.normalizer.manifest.observation_schema_hash,
        "runtime_bridge_sha256": sha256_file(bridge_path)[0],
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "critic_sidecar_schema_version": CRITIC_SIDECAR_SCHEMA_VERSION,
        "tensor_feature_count": TENSOR_FEATURE_COUNT,
    }


def _seed_checkpoint_identities(
    dataset: CampaignDataset, bridge_path: Path, seed: int, selected_update: int
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    return {
        **_campaign_identities(dataset, bridge_path, project_root),
        "neural_seed": seed,
        "selected_update": selected_update,
    }


def _save_and_verify_campaign_checkpoint(
    *,
    path: Path,
    model: ElderActorCritic,
    trainer: PPOTrainer,
    actor_config: ElderActorCriticConfig,
    ppo_config: PPOConfig,
    pilot_config: PilotConfig,
    normalizer: SemanticNormalizer,
    identities: dict[str, Any],
    observation: Tensor,
    continuation_batch: RolloutBatch,
    device: torch.device,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema_version": "elder-ppo-campaign-checkpoint-v2",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "trainer_state_dict": trainer.state_dict(),
        "actor_critic_config": asdict(actor_config),
        "ppo_config": asdict(ppo_config),
        "pilot_config": asdict(pilot_config),
        "normalizer": asdict(normalizer.manifest),
        "identities": identities,
        "rng": _capture_rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    loaded = torch.load(path, map_location=device, weights_only=False)
    loaded_normalizer = NormalizationManifest(**loaded["normalizer"])
    loaded_normalizer.validate()
    if asdict(loaded_normalizer) != asdict(normalizer.manifest):
        raise RuntimeError("Campaign checkpoint changed the frozen normalizer")
    expected = _policy_vector(model, observation)
    reloaded_model = ElderActorCritic(ElderActorCriticConfig(**loaded["actor_critic_config"])).to(
        device
    )
    reloaded_model.load_state_dict(loaded["model_state_dict"])
    reloaded_trainer = PPOTrainer(reloaded_model, PPOConfig(**loaded["ppo_config"]))
    reloaded_trainer.load_state_dict(copy.deepcopy(loaded["trainer_state_dict"]))
    actual = _policy_vector(reloaded_model, observation)
    reference_model = ElderActorCritic(ElderActorCriticConfig(**loaded["actor_critic_config"])).to(
        device
    )
    reference_model.load_state_dict(loaded["model_state_dict"])
    reference_trainer = PPOTrainer(reference_model, PPOConfig(**loaded["ppo_config"]))
    reference_trainer.load_state_dict(copy.deepcopy(loaded["trainer_state_dict"]))
    _restore_rng_state(loaded["rng"])
    expected_stochastic = _stochastic_policy_vector(reference_model, observation)
    _restore_rng_state(loaded["rng"])
    actual_stochastic = _stochastic_policy_vector(reloaded_model, observation)
    _restore_rng_state(loaded["rng"])
    reference_metrics = reference_trainer.update(continuation_batch)
    _restore_rng_state(loaded["rng"])
    reloaded_metrics = reloaded_trainer.update(continuation_batch)
    digest, size = sha256_file(path)
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": size,
        "max_abs_error": float(torch.max(torch.abs(expected - actual)).cpu()),
        "stochastic_policy_max_abs_error": float(
            torch.max(torch.abs(expected_stochastic - actual_stochastic)).cpu()
        ),
        "continuation_model_max_abs_error": _model_state_max_abs_error(
            reference_model, reloaded_model
        ),
        "continuation_metrics_max_abs_error": max(
            abs(reference_metrics[key] - reloaded_metrics[key]) for key in reference_metrics
        ),
        "optimizer_state_reloaded": bool(reloaded_trainer.optimizer.state_dict()["state"]),
        "value_normalization_state_reloaded": (
            reloaded_trainer.value_count == reference_trainer.value_count
            and reloaded_trainer.value_mean == reference_trainer.value_mean
            and reloaded_trainer.value_m2 == reference_trainer.value_m2
        ),
    }


def _timing_report(
    timings: _Timing, learning_curve: list[dict[str, Any]], elapsed: float
) -> dict[str, Any]:
    transitions = sum(int(row["transitions"]) for row in learning_curve)
    return {
        "transitions": transitions,
        "native_step_seconds": timings.native_step_seconds,
        "native_step_transitions_per_second": transitions / max(timings.native_step_seconds, 1e-12),
        "bridge_step_seconds": timings.bridge_step_seconds,
        "policy_inference_seconds": timings.policy_inference_seconds,
        "policy_inference_transitions_per_second": transitions
        / max(timings.policy_inference_seconds, 1e-12),
        "ppo_update_seconds": timings.ppo_update_seconds,
        "ppo_update_transitions_per_second": transitions / max(timings.ppo_update_seconds, 1e-12),
        "rollout_collection_seconds": timings.collection_seconds,
        "collector_worker_seconds": timings.collector_worker_seconds,
        "collector_workers_peak": timings.collector_workers_peak,
        "collector_parallel_efficiency": (
            timings.collector_worker_seconds
            / max(
                timings.collection_seconds * timings.collector_workers_peak,
                1e-12,
            )
        ),
        "end_to_end_seconds": elapsed,
        "end_to_end_transitions_per_second": transitions / max(elapsed, 1e-12),
    }
