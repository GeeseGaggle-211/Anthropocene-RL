"""Deterministic snapshot shards and checksum-addressed on-demand readers."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from anthropocene_rl.datasets.blobs import sha256_file
from anthropocene_rl.datasets.dataset_manifest import (
    DatasetCounts,
    DatasetRecord,
    DatasetRunManifest,
    DatasetShard,
    SerializationContract,
    utc_timestamp,
)
from anthropocene_rl.datasets.models import SnapshotBankManifest, SnapshotRecord
from anthropocene_rl.datasets.object_store import GcsObjectStore, ImmutableObjectStore


@dataclass(frozen=True)
class SourceRecord:
    record: SnapshotRecord
    path: Path


@dataclass(frozen=True)
class PackResult:
    manifest_path: Path
    manifest_sha256: str
    shard_count: int
    state_count: int
    payload_bytes: int
    shard_bytes: int
    elapsed_seconds: float

    @property
    def serialization_mib_per_second(self) -> float:
        return (self.payload_bytes / 1024**2) / max(self.elapsed_seconds, 1e-9)


def pack_snapshot_bank(
    *,
    snapshot_manifest_path: str | Path,
    blob_root: str | Path,
    staging_root: str | Path,
    dataset_id: str,
    run_id: str,
    target_shard_bytes: int = 64 * 1024**2,
    anthropocene_git_branch: str | None = None,
    training_git_commit: str | None = None,
) -> PackResult:
    """Pack a snapshot bank without recompressing its gzip SaveCodec payloads."""

    if target_shard_bytes <= 0:
        raise ValueError("target_shard_bytes must be positive")
    started = time.perf_counter()
    source_path = Path(snapshot_manifest_path)
    source = SnapshotBankManifest.load(source_path)
    root = Path(blob_root).resolve()
    output = Path(staging_root) / dataset_id / run_id
    shard_directory = output / "shards"
    shard_directory.mkdir(parents=True, exist_ok=True)

    sources = [_source_record(record, root) for record in source.records]
    groups = _trajectory_groups(sources)
    shard_groups: list[list[SourceRecord]] = []
    current: list[SourceRecord] = []
    current_bytes = 0
    for group in groups:
        group_bytes = sum(item.record.content_bytes for item in group)
        if current and current_bytes + group_bytes > target_shard_bytes:
            shard_groups.append(current)
            current = []
            current_bytes = 0
        current.extend(group)
        current_bytes += group_bytes
    if current:
        shard_groups.append(current)

    shards: list[DatasetShard] = []
    for index, items in enumerate(shard_groups):
        name = f"shard-{index:06d}.tar"
        path = shard_directory / name
        records = tuple(_dataset_record(item.record) for item in items)
        _write_shard(path, items, records)
        digest, size = sha256_file(path)
        shards.append(
            DatasetShard(
                name=name,
                object_name=(
                    f"objects/sha256/{digest[:2]}/{digest}.tar"
                ),
                sha256=digest,
                size_bytes=size,
                payload_bytes=sum(record.content_bytes for record in records),
                record_count=len(records),
                trajectory_count=len({record.trajectory_id for record in records}),
                sample_count=sum(record.sample_count for record in records),
                records=records,
            )
        )

    records = tuple(record for shard in shards for record in shard.records)
    manifest = DatasetRunManifest(
        dataset_id=dataset_id,
        run_id=run_id,
        generated_at=utc_timestamp(),
        anthropocene_git_commit=source.native_simulator_sha,
        anthropocene_git_branch=anthropocene_git_branch,
        training_git_commit=training_git_commit,
        source_manifest_path=source_path.as_posix(),
        source_manifest_sha256=source.sha256(),
        simulation_configuration={
            "snapshot_bank_id": source.bank_id,
            "snapshot_bank_version": source.bank_version,
            "generator_config_sha256": source.generator_config_sha256,
            "configuration_hashes": sorted(
                {record.configuration_hash for record in source.records}
            ),
        },
        map_world_configuration={
            "scenario_families": sorted(
                {record.scenario_family for record in source.records}
            ),
            "partitions": sorted({record.partition.value for record in source.records}),
        },
        random_seed_information={
            "world_seeds": sorted({record.world_seed for record in source.records}),
            "generation_seeds": sorted(
                {record.generator.generation_seed for record in source.records}
            ),
            "rng_algorithms": sorted(
                {record.generator.rng_algorithm for record in source.records}
            ),
        },
        counts=DatasetCounts(
            simulations=len({record.world_seed for record in records}),
            trajectories=len({record.trajectory_id for record in records}),
            states=len(records),
            samples=sum(record.sample_count for record in records),
        ),
        serialization=SerializationContract(
            format="anthropocene-savecodec-snapshot",
            version=_serialization_version(source),
        ),
        target_shard_bytes=target_shard_bytes,
        shards=tuple(shards),
        publication_status="staged",
        storage_backend="local",
        storage_root=output.resolve().as_uri(),
    )
    manifest_path = output / "manifest.json"
    manifest.save_atomic(manifest_path)
    elapsed = time.perf_counter() - started
    return PackResult(
        manifest_path=manifest_path,
        manifest_sha256=manifest.sha256(),
        shard_count=len(shards),
        state_count=len(records),
        payload_bytes=sum(record.content_bytes for record in records),
        shard_bytes=sum(shard.size_bytes for shard in shards),
        elapsed_seconds=elapsed,
    )


def _source_record(record: SnapshotRecord, root: Path) -> SourceRecord:
    path = (root / record.blob_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Snapshot blob path escapes the configured root") from error
    digest, size = sha256_file(path)
    if digest != record.content_sha256 or size != record.content_bytes:
        raise ValueError(f"Snapshot {record.snapshot_id!r} failed SHA-256/size validation")
    return SourceRecord(record=record, path=path)


def _trajectory_groups(sources: Iterable[SourceRecord]) -> tuple[tuple[SourceRecord, ...], ...]:
    ordered = sorted(
        sources,
        key=lambda item: (
            item.record.world_seed,
            item.record.lineage_id,
            item.record.decision_index,
            item.record.snapshot_id,
        ),
    )
    groups: list[list[SourceRecord]] = []
    key: tuple[int, str] | None = None
    for item in ordered:
        candidate = (item.record.world_seed, item.record.lineage_id)
        if key != candidate:
            groups.append([])
            key = candidate
        groups[-1].append(item)
    return tuple(tuple(group) for group in groups)


def _dataset_record(record: SnapshotRecord) -> DatasetRecord:
    return DatasetRecord(
        record_id=record.snapshot_id,
        member_path=f"records/{record.snapshot_id}.snapshot",
        content_sha256=record.content_sha256,
        content_bytes=record.content_bytes,
        trajectory_id=record.lineage_id,
        world_seed=record.world_seed,
        sample_count=record.calibration_observation_count,
    )


def _write_shard(
    path: Path,
    sources: list[SourceRecord],
    records: tuple[DatasetRecord, ...],
) -> None:
    index_bytes = (
        json.dumps(
            {
                "shard_index_schema_version": "anthropocene-shard-index-v1",
                "records": [record.model_dump(mode="json") for record in records],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    with tarfile.open(path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        index_info = _tar_info("shard-index.json", len(index_bytes))
        archive.addfile(index_info, io.BytesIO(index_bytes))
        for source, record in zip(sources, records, strict=True):
            info = _tar_info(record.member_path, record.content_bytes)
            with source.path.open("rb") as handle:
                archive.addfile(info, handle)


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o444
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _serialization_version(source: SnapshotBankManifest) -> str:
    versions = sorted({record.native_simulator_version for record in source.records})
    return ",".join(versions)


class DatasetReader:
    """Read records through a manifest while caching immutable shards by SHA-256."""

    def __init__(
        self,
        manifest: DatasetRunManifest,
        *,
        store: ImmutableObjectStore,
        cache_root: str | Path,
    ) -> None:
        self.manifest = manifest
        self.store = store
        self.cache_root = Path(cache_root)
        self.cache_hits = 0
        self.cache_misses = 0
        self._records = {
            record.record_id: (shard, record)
            for shard in manifest.shards
            for record in shard.records
        }

    def record_ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    def ensure_shard(self, shard: DatasetShard) -> Path:
        path = self.cache_root / "shards" / shard.sha256[:2] / f"{shard.sha256}.tar"
        if path.is_file():
            digest, size = sha256_file(path)
            if digest == shard.sha256 and size == shard.size_bytes:
                self.cache_hits += 1
                return path
            path.unlink()
        self.cache_misses += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=".download-",
            suffix=".download",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            info = self.store.download(shard.object_name, temporary)
            if info.size != shard.size_bytes:
                raise ValueError(f"Downloaded shard {shard.name} has the wrong provider size")
            digest, size = sha256_file(temporary)
            if digest != shard.sha256 or size != shard.size_bytes:
                raise ValueError(f"Downloaded shard {shard.name} failed SHA-256/size validation")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def read_record(self, record_id: str) -> bytes:
        try:
            shard, record = self._records[record_id]
        except KeyError as error:
            raise KeyError(f"Unknown dataset record: {record_id}") from error
        path = self.ensure_shard(shard)
        with tarfile.open(path, mode="r:") as archive:
            member = archive.getmember(record.member_path)
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Shard member {record.member_path!r} is not a file")
            payload = handle.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record.content_sha256 or len(payload) != record.content_bytes:
            raise ValueError(f"Dataset record {record_id!r} failed SHA-256/size validation")
        return payload

    def materialize_record(self, record_id: str) -> Path:
        _, record = self._records[record_id]
        path = (
            self.cache_root
            / "records"
            / "sha256"
            / record.content_sha256[:2]
            / f"{record.content_sha256}.snapshot"
        )
        if path.is_file():
            digest, size = sha256_file(path)
            if digest == record.content_sha256 and size == record.content_bytes:
                return path
            path.unlink()
        payload = self.read_record(record_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=".record-",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def iter_records(self, record_ids: Iterable[str] | None = None) -> Iterator[bytes]:
        for record_id in record_ids or self.record_ids():
            yield self.read_record(record_id)

    def cache_statistics(self) -> dict[str, int | float]:
        total = self.cache_hits + self.cache_misses
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_ratio": self.cache_hits / total if total else 0.0,
        }


def discover_runs(store: ImmutableObjectStore, dataset_id: str) -> tuple[str, ...]:
    prefix = f"datasets/{dataset_id}/"
    suffix = "/manifest.json"
    return tuple(
        sorted(
            info.name[len(prefix) : -len(suffix)]
            for info in store.list(prefix)
            if info.name.startswith(prefix) and info.name.endswith(suffix)
        )
    )


def load_dataset_reader(
    reference: str | Path,
    *,
    store: ImmutableObjectStore,
    cache_root: str | Path,
) -> DatasetReader:
    """Load a local manifest path or a configured immutable ``gs://`` manifest."""

    value = str(reference)
    if not value.startswith("gs://"):
        manifest = DatasetRunManifest.load(value)
        return DatasetReader(manifest, store=store, cache_root=cache_root)
    if not isinstance(store, GcsObjectStore):
        raise ValueError("A gs:// manifest requires the configured GCS backend")
    without_scheme = value[5:]
    bucket, separator, object_name = without_scheme.partition("/")
    if not separator or bucket != store.config.bucket:
        raise ValueError("Manifest URI bucket differs from the configured GCS bucket")
    prefix = store.config.prefix
    if prefix:
        expected = prefix + "/"
        if not object_name.startswith(expected):
            raise ValueError("Manifest URI is outside the configured GCS prefix")
        object_name = object_name[len(expected) :]
    cache_path = Path(cache_root) / "manifests" / hashlib.sha256(value.encode()).hexdigest()
    cache_path = cache_path.with_suffix(".json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=cache_path.parent,
        prefix=".manifest.",
        suffix=".download",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        store.download(object_name, temporary)
        manifest = DatasetRunManifest.load(temporary)
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return DatasetReader(manifest, store=store, cache_root=cache_root)
