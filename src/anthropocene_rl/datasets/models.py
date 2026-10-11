"""Versioned snapshot-bank manifest contracts.

The manifest contains reset provenance and measured native state. It never
contains hand-authored actor tensors or action labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from anthropocene_rl.contracts import SIMULATOR_COMMIT, SIMULATOR_REPOSITORY

MANIFEST_SCHEMA_VERSION = "snapshot-bank-v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Partition(StrEnum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    HELDOUT = "heldout"


class SnapshotClassification(StrEnum):
    NATURAL = "natural"
    CONTROLLED_VALID_FIXTURE = "controlled-valid-fixture"
    SYNTHETIC_STRESS = "synthetic-stress"


class StructuralActionMasks(StrictModel):
    field_retention: bool
    herd_retention: bool
    field_development: bool
    herd_development: bool
    movement: bool
    movement_candidates: tuple[bool, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        if not self.movement_candidates[0]:
            raise ValueError("The explicit stay movement candidate must be legal")
        if self.herd_development and not self.herd_retention:
            raise ValueError("Herd Development cannot be active without a retained herd")
        if not self.movement and any(self.movement_candidates[1:]):
            raise ValueError("Movement-disabled states cannot expose destination candidates")
        return self


class InstitutionalMemoryMetadata(StrictModel):
    age_years: float = Field(ge=0)
    maturity_20y: float = Field(ge=0, le=1)
    maturity_60y: float = Field(ge=0, le=1)
    maturity_180y: float = Field(ge=0, le=1)


class MeasuredStrata(StrictModel):
    population: int = Field(ge=0)
    connected_consumers: int = Field(ge=0)
    next_interval_years: float = Field(gt=0)
    food_reserve_years: float = Field(ge=0)
    reserve_horizon_ratio: float = Field(ge=0)
    field_q: float = Field(ge=0, le=1)
    fishing_q: float | None = Field(default=None, ge=0, le=1)
    herd_animals: int = Field(ge=0)
    herd_zero_displacement_capacity: float = Field(ge=0)
    herd_absolute_capacity: float = Field(ge=0)
    herd_water_capacity: float = Field(ge=0)
    herd_zero_displacement_ratio: float | None = Field(default=None, ge=0)
    herd_absolute_ratio: float | None = Field(default=None, ge=0)
    herd_water_ratio: float | None = Field(default=None, ge=0)
    fishing_ground_count: int = Field(ge=0, le=3)
    current_qol: float = Field(ge=0, le=1)
    food_needs_met: float = Field(ge=0, le=1)
    water_needs_met: float = Field(ge=0, le=1)
    clothing_coverage: float = Field(ge=0, le=1)
    housing_coverage: float = Field(ge=0, le=1)
    retained_field_capital: float = Field(default=0, ge=0)
    retained_field_operation: float = Field(default=0, ge=0, le=1)
    maintenance_worker_years: float = Field(default=0, ge=0)
    hand_tools_per_capita: float = Field(default=0, ge=0)
    farm_tools_per_capita: float = Field(default=0, ge=0)
    materials_per_capita: float = Field(default=0, ge=0)
    delivered_trade_total: float = Field(default=0, ge=0)
    delivered_import_total: float = Field(default=0, ge=0)
    delivered_export_total: float = Field(default=0, ge=0)
    delivered_food_import: float = Field(default=0, ge=0)
    delivered_food_export: float = Field(default=0, ge=0)
    delivered_material_tool_import: float = Field(default=0, ge=0)
    import_dependence: float = Field(default=0, ge=0)
    signed_migration: float = 0
    fission_candidate_present: bool = False
    fission_recent_split: bool = False
    fission_pressure: float = Field(default=0, ge=0, le=1)
    fission_exit_opportunity: float = Field(default=0, ge=0, le=1)
    fission_daughter_population: int = Field(default=0, ge=0)
    parent_population_relief: float = Field(default=0, ge=0)
    field_frontier: str
    cultivation_available: bool
    husbandry_available: bool
    fishing_available: bool
    owned_domestic_herd: bool
    structural_masks: StructuralActionMasks
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_measured_relations(self) -> Self:
        expected_reserve = self.food_reserve_years / self.next_interval_years
        if abs(expected_reserve - self.reserve_horizon_ratio) > 1e-8 * max(1.0, expected_reserve):
            raise ValueError("Measured reserve-horizon ratio disagrees with physical durations")
        if self.owned_domestic_herd != (self.herd_animals > 0):
            raise ValueError("Owned-herd stratum disagrees with measured animal count")
        if self.fishing_available != (self.fishing_ground_count > 0):
            raise ValueError("Fishing availability disagrees with observed fishing rows")
        if (self.fishing_q is not None) != (self.fishing_ground_count > 0):
            raise ValueError("Fishing q presence disagrees with observed fishing rows")
        if self.structural_masks.field_development != self.cultivation_available:
            raise ValueError("Field Development mask disagrees with cultivation capability")
        if self.structural_masks.herd_retention != self.owned_domestic_herd:
            raise ValueError("Herd Retention mask disagrees with owned-herd state")
        if self.structural_masks.herd_development != (
            self.husbandry_available and self.owned_domestic_herd
        ):
            raise ValueError("Herd Development mask disagrees with husbandry and owned herd")
        for label, capacity, ratio in (
            (
                "zero-displacement",
                self.herd_zero_displacement_capacity,
                self.herd_zero_displacement_ratio,
            ),
            ("absolute", self.herd_absolute_capacity, self.herd_absolute_ratio),
            ("water", self.herd_water_capacity, self.herd_water_ratio),
        ):
            expected_ratio = self.herd_animals / capacity if capacity > 0 else None
            if expected_ratio is None:
                if ratio is not None:
                    raise ValueError(f"{label} herd ratio must be missing at zero capacity")
            elif ratio is None or abs(ratio - expected_ratio) > 1e-8 * max(1.0, expected_ratio):
                raise ValueError(f"{label} herd ratio disagrees with measured capacity")
        if self.cultivation_available == (self.field_frontier == "none"):
            raise ValueError("Field frontier identity disagrees with cultivation capability")
        if tuple(sorted(set(self.tags))) != self.tags:
            raise ValueError("Measured stratum tags must be unique and sorted")
        return self


class ControlledMetadata(StrictModel):
    camp_id: int = Field(ge=0)
    society_id: int = Field(ge=0)
    player_id: int = Field(ge=0)


class NativeSchemaIdentity(StrictModel):
    observation_schema_version: int
    action_schema_version: int
    environment_schema_version: int
    critic_sidecar_schema_version: int
    tensor_feature_count: int
    observation_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class GeneratorProvenance(StrictModel):
    name: str
    version: str
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_seed: int
    rng_algorithm: str
    rng_stream: str


class RemoteBlobMetadata(StrictModel):
    """Mutable storage metadata that is excluded from snapshot-bank identity."""

    backend: Literal["google-drive"]
    file_id: str = Field(min_length=1)
    remote_key: str = Field(pattern=r"^sha256/[0-9a-f]{2}/[0-9a-f]{64}\.snapshot$")
    upload_status: Literal["uploaded"] = "uploaded"
    verification_status: Literal["metadata-verified", "download-verified"]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_bytes: int = Field(gt=0)
    md5_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    provider_sha256_checksum: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    uploaded_at: str | None = None
    version: str | None = None

    @model_validator(mode="after")
    def validate_remote_key(self) -> Self:
        expected = f"sha256/{self.content_sha256[:2]}/{self.content_sha256}.snapshot"
        if self.remote_key != expected:
            raise ValueError("Remote object key disagrees with its content SHA-256")
        return self


class SnapshotStoragePolicy(StrictModel):
    """Durability and local-cache policy without authentication material."""

    local_mode: Literal["bounded-cache"] = "bounded-cache"
    max_local_bytes: int = Field(default=5 * 1024**3, gt=0)
    eviction_policy: Literal["lru"] = "lru"
    remote_backend: Literal["google-drive"] = "google-drive"
    remote_root_id: str = Field(min_length=1)
    remote_root_path: str = Field(min_length=1)


class SnapshotValidation(StrictModel):
    save_load_fingerprint: bool
    native_reset: bool
    legal_native_step: bool
    replay_verified: bool
    finite_positive_elapsed_years: bool
    schema_configuration_hash_match: bool
    structural_masks_valid: bool
    measured_strata_match: bool
    elapsed_years: float = Field(gt=0)
    post_step_fingerprint: str = Field(pattern=r"^[0-9A-Fa-f]{64}$")

    @model_validator(mode="after")
    def all_gates_pass(self) -> Self:
        gates = (
            self.save_load_fingerprint,
            self.native_reset,
            self.legal_native_step,
            self.replay_verified,
            self.finite_positive_elapsed_years,
            self.schema_configuration_hash_match,
            self.structural_masks_valid,
            self.measured_strata_match,
        )
        if not all(gates):
            raise ValueError("Accepted snapshot records must pass every native validation gate")
        return self


class SnapshotRecord(StrictModel):
    snapshot_id: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_bytes: int = Field(gt=0)
    blob_path: str
    native_fingerprint: str = Field(pattern=r"^[0-9A-Fa-f]{64}$")
    partition: Partition
    scenario_family: str = Field(pattern=r"^F(?:0[1-9]|1[0-8])$")
    scenario_version: str
    world_seed: int
    trajectory_family: str
    lineage_id: str
    counterfactual_group_id: str | None = None
    source_controller: str
    controlled: ControlledMetadata
    physical_years: float = Field(ge=0)
    decision_index: int = Field(ge=0)
    native_simulator_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    native_simulator_version: str
    configuration_hash: str = Field(pattern=r"^[0-9A-Fa-f]{64}$")
    native_build_hash: str = Field(pattern=r"^[0-9A-Fa-f]{32,64}$")
    schema_identity: NativeSchemaIdentity = Field(alias="schema", serialization_alias="schema")
    actual_strata: MeasuredStrata
    institutional_memory: InstitutionalMemoryMetadata
    classification: SnapshotClassification
    generator: GeneratorProvenance
    selection_metadata: dict[str, Any] = Field(default_factory=dict)
    observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_observation_count: int = Field(ge=1)
    validation: SnapshotValidation
    remote_storage: RemoteBlobMetadata | None = None

    @model_validator(mode="after")
    def validate_frozen_native_identity(self) -> Self:
        if self.native_simulator_sha != SIMULATOR_COMMIT:
            raise ValueError("Snapshot record was not generated by the frozen simulator SHA")
        if self.remote_storage is not None and (
            self.remote_storage.content_sha256 != self.content_sha256
            or self.remote_storage.content_bytes != self.content_bytes
        ):
            raise ValueError("Remote storage identity disagrees with snapshot content")
        return self


class RejectedCandidate(StrictModel):
    candidate_id: str
    scenario_family: str
    scenario_version: str
    world_seed: int
    trajectory_family: str
    lineage_id: str
    counterfactual_group_id: str | None = None
    requested_partition: Partition
    reason_code: str
    reason: str
    selection_metadata: dict[str, Any] = Field(default_factory=dict)


class PartitionContract(StrictModel):
    algorithm: str
    salt: str
    train_fraction: float = Field(gt=0, lt=1)
    development_fraction: float = Field(gt=0, lt=1)
    heldout_fraction: float = Field(gt=0, lt=1)
    grouping_fields: tuple[str, ...]
    isolate_world_seed: bool = True
    historical_partition: Partition = Partition.DEVELOPMENT

    @model_validator(mode="after")
    def validate_fractions(self) -> Self:
        total = self.train_fraction + self.development_fraction + self.heldout_fraction
        if abs(total - 1.0) > 1e-12:
            raise ValueError("Partition fractions must sum to one")
        return self


class SnapshotBankManifest(StrictModel):
    manifest_schema_version: str = MANIFEST_SCHEMA_VERSION
    bank_id: str
    bank_version: str
    generator_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    native_simulator_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    native_simulator_repository: str
    blob_storage: str
    partition_contract: PartitionContract
    records: tuple[SnapshotRecord, ...]
    rejected_candidates: tuple[RejectedCandidate, ...] = ()
    storage_policy: SnapshotStoragePolicy | None = None

    @model_validator(mode="after")
    def validate_manifest_identity(self) -> Self:
        if self.manifest_schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("Unsupported snapshot-bank manifest schema")
        if self.native_simulator_sha != SIMULATOR_COMMIT:
            raise ValueError("Snapshot bank does not use the frozen simulator SHA")
        if self.native_simulator_repository != SIMULATOR_REPOSITORY:
            raise ValueError("Snapshot bank names a different native simulator repository")
        return self

    def canonical_bytes(self) -> bytes:
        # Remote file IDs, upload timestamps, and cache limits are storage
        # metadata.  They must not invalidate normalizers or checkpoints whose
        # authoritative identity is the native content/provenance cohort.
        payload = self._document_payload()
        payload.pop("storage_policy", None)
        for record in payload["records"]:
            record.pop("remote_storage", None)
        return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self._document_payload()
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def save_atomic(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self._document_payload()
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _document_payload(self) -> dict[str, Any]:
        # Preserve the field set of older snapshot-bank documents.  The v1
        # schema has gained additive, defaulted measured-strata fields; writing
        # those defaults into a legacy document would otherwise change its
        # frozen manifest identity merely because it was loaded by newer code.
        payload = self.model_dump(mode="json", by_alias=True, exclude_unset=True)
        if self.storage_policy is None:
            payload.pop("storage_policy", None)
        for record, serialized in zip(self.records, payload["records"], strict=True):
            if record.remote_storage is None:
                serialized.pop("remote_storage", None)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> SnapshotBankManifest:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
