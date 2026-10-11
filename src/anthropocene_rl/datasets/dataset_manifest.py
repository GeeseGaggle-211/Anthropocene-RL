"""Versioned, provider-neutral manifests for immutable RL dataset runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from anthropocene_rl.datasets.models import StrictModel

DATASET_SCHEMA_VERSION: Literal["anthropocene-dataset-v1"] = (
    "anthropocene-dataset-v1"
)


class DatasetRecord(StrictModel):
    """One logical state/sample stored as a member of an immutable shard."""

    record_id: str = Field(min_length=1)
    member_path: str = Field(pattern=r"^records/[A-Za-z0-9._-]+\.snapshot$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_bytes: int = Field(gt=0)
    trajectory_id: str
    world_seed: int
    sample_count: int = Field(default=1, ge=1)


class DatasetShard(StrictModel):
    name: str = Field(pattern=r"^shard-[0-9]{6}\.tar$")
    object_name: str = Field(min_length=1)
    gcs_uri: str | None = Field(default=None, pattern=r"^gs://[^/]+/.+$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    payload_bytes: int = Field(gt=0)
    record_count: int = Field(gt=0)
    trajectory_count: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    records: tuple[DatasetRecord, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.record_count != len(self.records):
            raise ValueError("Shard record_count does not match records")
        if self.payload_bytes != sum(record.content_bytes for record in self.records):
            raise ValueError("Shard payload_bytes does not match records")
        if self.sample_count != sum(record.sample_count for record in self.records):
            raise ValueError("Shard sample_count does not match records")
        if self.trajectory_count != len({record.trajectory_id for record in self.records}):
            raise ValueError("Shard trajectory_count does not match records")
        return self


class DatasetCounts(StrictModel):
    simulations: int = Field(ge=0)
    trajectories: int = Field(ge=0)
    states: int = Field(ge=0)
    samples: int = Field(ge=0)


class SerializationContract(StrictModel):
    format: Literal["anthropocene-savecodec-snapshot"]
    version: str
    member_compression: Literal["gzip"] = "gzip"
    shard_archive: Literal["tar"] = "tar"
    shard_compression: Literal["none-already-compressed-members"] = (
        "none-already-compressed-members"
    )


class DatasetRunManifest(StrictModel):
    """Authoritative publication contract for one immutable dataset run."""

    dataset_schema_version: Literal["anthropocene-dataset-v1"] = DATASET_SCHEMA_VERSION
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    generated_at: str
    anthropocene_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    anthropocene_git_branch: str | None = None
    training_git_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    source_manifest_path: str
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulation_configuration: dict[str, Any]
    map_world_configuration: dict[str, Any]
    random_seed_information: dict[str, Any]
    counts: DatasetCounts
    serialization: SerializationContract
    target_shard_bytes: int = Field(gt=0)
    shards: tuple[DatasetShard, ...]
    publication_status: Literal["staged", "complete"] = "staged"
    storage_backend: Literal["local", "gcs"] = "local"
    storage_root: str

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        records = tuple(record for shard in self.shards for record in shard.records)
        record_ids = [record.record_id for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("Dataset record IDs must be unique")
        if self.counts.states != len(records):
            raise ValueError("Dataset state count does not match shard records")
        if self.counts.samples != sum(record.sample_count for record in records):
            raise ValueError("Dataset sample count does not match shard records")
        if self.counts.trajectories != len({record.trajectory_id for record in records}):
            raise ValueError("Dataset trajectory count does not match shard records")
        if self.counts.simulations != len({record.world_seed for record in records}):
            raise ValueError("Dataset simulation count does not match world seeds")
        expected_names = [f"shard-{index:06d}.tar" for index in range(len(self.shards))]
        if [shard.name for shard in self.shards] != expected_names:
            raise ValueError("Dataset shards must be contiguous and ordered")
        if (
            self.publication_status == "complete"
            and self.storage_backend == "gcs"
            and any(shard.gcs_uri is None for shard in self.shards)
        ):
            raise ValueError("Published GCS shards require immutable gs:// URIs")
        return self

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def save_atomic(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(self.model_dump(mode="json"), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> Self:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("dataset_schema_version") if isinstance(payload, dict) else None
        if version != DATASET_SCHEMA_VERSION:
            raise ValueError(f"Unsupported dataset schema version: {version!r}")
        return cls.model_validate(payload)


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
