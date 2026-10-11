"""Declared snapshot-bank generation-plan schema and partition resolution."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from anthropocene_rl.config import load_yaml
from anthropocene_rl.contracts import SIMULATOR_COMMIT
from anthropocene_rl.datasets.models import (
    Partition,
    SnapshotClassification,
    StrictModel,
)
from anthropocene_rl.datasets.partition import DeterministicPartitioner, LineageIdentity
from anthropocene_rl.observations.normalization import (
    CAPABILITIES_WITH_PROGRESSION_THRESHOLDS,
)


class WorldGenerationConfig(StrictModel):
    width: int = Field(ge=24)
    height: int = Field(ge=16)
    players: int = Field(ge=2)
    start_separation: int = Field(ge=3)
    annual_human_growth_rate: float = Field(ge=0)


class PartitionConfig(StrictModel):
    salt: str
    train_fraction: float = 0.6
    development_fraction: float = 0.2
    heldout_fraction: float = 0.2


class TrajectoryPlan(StrictModel):
    trajectory_id: str
    scenario_family: str = Field(pattern=r"^F(?:0[1-9]|1[0-8])$")
    scenario_version: str = "stage0-v1"
    world_seed: int
    trajectory_family: str
    lineage_id: str
    counterfactual_group_id: str | None = None
    source_controller: str = "bounded-exploratory-elder-v1"
    classification: SnapshotClassification = SnapshotClassification.CONTROLLED_VALID_FIXTURE
    historical_or_consumed_diagnostic: bool = False
    snapshot_steps: tuple[int, ...] = Field(min_length=1)
    economy: Literal[
        "cultivation",
        "husbandry",
        "mixed",
        "fishing",
        "cultivation-fishing",
        "husbandry-fishing",
        "all-three",
        "knowledge-only-husbandry",
    ]
    population: int = Field(ge=50)
    field_q: float = Field(default=0, ge=0, le=0.65)
    herd_support_ratio: float = Field(default=0, ge=0, le=1.8)
    reserve_intervals: float = Field(default=1, ge=0, le=4)
    fishing_ground_target: int = Field(default=0, ge=0, le=3)
    site_support_quantile: float = Field(default=0.5, ge=0, le=1)
    tool_abundance: float = Field(default=1, ge=0, le=2)
    amenity_coverage: float = Field(default=1, ge=0, le=1)
    frontier: Literal[
        "primitive", "plough", "technology-transition-cultivation"
    ] = "primitive"
    requested_tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_trajectory(self) -> Self:
        if tuple(sorted(set(self.snapshot_steps))) != self.snapshot_steps:
            raise ValueError("snapshot_steps must be unique and increasing")
        if any(step < 0 for step in self.snapshot_steps):
            raise ValueError("snapshot_steps cannot be negative")
        cultivation = self.economy in {
            "cultivation",
            "mixed",
            "cultivation-fishing",
            "all-three",
        }
        husbandry = self.economy in {
            "husbandry",
            "mixed",
            "husbandry-fishing",
            "all-three",
            "knowledge-only-husbandry",
        }
        fishing = self.economy in {
            "fishing",
            "cultivation-fishing",
            "husbandry-fishing",
            "all-three",
        }
        if (self.field_q > 0) != cultivation:
            raise ValueError("field_q must be positive exactly when cultivation is enabled")
        if self.herd_support_ratio > 0 and not husbandry:
            raise ValueError("A requested herd requires husbandry capability")
        if self.economy == "knowledge-only-husbandry" and self.herd_support_ratio != 0:
            raise ValueError("Knowledge-only husbandry cannot request an owned herd")
        if fishing != (self.fishing_ground_target > 0):
            raise ValueError("Fishing economies require a positive fishing-ground target")
        if self.frontier == "plough" and not cultivation:
            raise ValueError("The Plough frontier requires cultivation")
        return self


class ResolvedTrajectoryPlan(TrajectoryPlan):
    partition: Partition


class SnapshotBankGenerationConfig(StrictModel):
    config_schema_version: str = "snapshot-bank-generation-v1"
    bank_kind: Literal["stage0", "cultivation-pilot"] = "stage0"
    bank_id: str
    bank_version: str
    native_generator_version: str
    generation_seed: int
    world: WorldGenerationConfig
    partition: PartitionConfig
    calibration_rollout_steps: int = Field(default=3, ge=1, le=5)
    verify_deterministic_regeneration: bool = True
    local_movement_radius: float = Field(gt=0)
    capability_experience_thresholds: dict[str, float]
    trajectories: tuple[TrajectoryPlan, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bank_plan(self) -> Self:
        if self.config_schema_version != "snapshot-bank-generation-v1":
            raise ValueError("Unsupported generation config schema")
        ids = [trajectory.trajectory_id for trajectory in self.trajectories]
        if len(ids) != len(set(ids)):
            raise ValueError("trajectory_id values must be unique")
        snapshots = sum(len(trajectory.snapshot_steps) for trajectory in self.trajectories)
        if self.bank_kind == "stage0" and not 24 <= snapshots <= 48:
            raise ValueError("Stage-0 generation plan must request 24-48 snapshots")
        if self.bank_kind == "cultivation-pilot" and not 200 <= snapshots <= 500:
            raise ValueError(
                "Cultivation pilot generation plan must request 200-500 snapshots"
            )
        if set(self.capability_experience_thresholds) != set(
            CAPABILITIES_WITH_PROGRESSION_THRESHOLDS
        ):
            raise ValueError(
                "Stage-0 must declare the three native semantic capability thresholds"
            )
        if any(
            not math.isfinite(threshold) or threshold <= 0
            for threshold in self.capability_experience_thresholds.values()
        ):
            raise ValueError("Capability Experience thresholds must be finite and positive")
        if self.bank_kind == "cultivation-pilot":
            allowed_families = {"F01", "F02", "F03", "F13", "F17", "F18"}
            for trajectory in self.trajectories:
                if trajectory.scenario_family not in allowed_families:
                    raise ValueError(
                        "Cultivation pilot contains a non-cultivation scenario family"
                    )
                if (
                    trajectory.economy != "cultivation"
                    or trajectory.herd_support_ratio != 0
                    or trajectory.fishing_ground_target != 0
                ):
                    raise ValueError(
                        "Cultivation pilot must keep herding and fishing structurally absent"
                    )
                if trajectory.historical_or_consumed_diagnostic:
                    raise ValueError(
                        "Cultivation pilot cannot reuse consumed historical diagnostics"
                    )
        return self

    def canonical_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        # Stage-0 predates the explicit bank-kind discriminator.  Preserve its
        # committed generator identity while still emitting ``bank_kind`` to
        # the current native plan validator.
        if self.bank_kind == "stage0" and "bank_kind" not in self.model_fields_set:
            payload.pop("bank_kind", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def resolve_partitions(
        self,
    ) -> tuple[tuple[ResolvedTrajectoryPlan, ...], DeterministicPartitioner]:
        partitioner = DeterministicPartitioner(
            salt=self.partition.salt,
            train_fraction=self.partition.train_fraction,
            development_fraction=self.partition.development_fraction,
            heldout_fraction=self.partition.heldout_fraction,
        )
        resolved = []
        for plan in self.trajectories:
            partition = partitioner.assign(
                LineageIdentity(
                    plan.world_seed,
                    plan.trajectory_family,
                    plan.lineage_id,
                    plan.counterfactual_group_id,
                    plan.historical_or_consumed_diagnostic,
                )
            )
            resolved.append(
                ResolvedTrajectoryPlan.model_validate(
                    {**plan.model_dump(mode="python"), "partition": partition}
                )
            )
        return tuple(resolved), partitioner

    @classmethod
    def load(cls, path: str | Path) -> SnapshotBankGenerationConfig:
        return cls.model_validate(load_yaml(path))

    def native_payload(self) -> dict[str, Any]:
        resolved, _ = self.resolve_partitions()
        payload = self.model_dump(mode="json", exclude={"trajectories"})
        payload["native_simulator_sha"] = SIMULATOR_COMMIT
        payload["generator_config_sha256"] = self.canonical_sha256()
        payload["trajectories"] = [plan.model_dump(mode="json") for plan in resolved]
        return payload
