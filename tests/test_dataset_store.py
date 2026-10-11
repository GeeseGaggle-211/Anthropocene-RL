from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from anthropocene_rl.datasets.blobs import sha256_file
from anthropocene_rl.datasets.dataset_manifest import DatasetRunManifest
from anthropocene_rl.datasets.models import SnapshotBankManifest, SnapshotRecord
from anthropocene_rl.datasets.object_store import (
    DatasetPublisher,
    DatasetStoreConfig,
    GcsObjectStore,
    ImmutableObjectStore,
    LocalObjectStore,
    ObjectCollisionError,
    ObjectInfo,
)
from anthropocene_rl.datasets.shards import (
    DatasetReader,
    discover_runs,
    pack_snapshot_bank,
)
from anthropocene_rl.observations.normalization import NormalizationManifest
from anthropocene_rl.training.overfit import load_stage0_selection


def _snapshot_bank(tmp_path: Path) -> tuple[Path, Path, dict[str, bytes]]:
    template = SnapshotBankManifest.load("datasets/manifests/elder-v2-stage0-v1.json")
    root = tmp_path / "blobs"
    payloads: dict[str, bytes] = {}
    records: list[SnapshotRecord] = []
    for index in range(6):
        record_id = f"dataset-test-{index}"
        payload = gzip.compress((f"state-{index}-" * 80).encode(), mtime=0)
        digest = hashlib.sha256(payload).hexdigest()
        relative = f"sha256/{digest[:2]}/{digest}.snapshot"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        raw = template.records[index].model_dump(mode="python", by_alias=True)
        raw.update(
            {
                "snapshot_id": record_id,
                "content_sha256": digest,
                "content_bytes": len(payload),
                "blob_path": relative,
                "world_seed": 100 + index // 2,
                "lineage_id": f"lineage-{index // 2}",
                "decision_index": index % 2,
                "remote_storage": None,
            }
        )
        records.append(SnapshotRecord.model_validate(raw))
        payloads[record_id] = payload
    manifest = template.model_copy(
        update={"records": tuple(records), "rejected_candidates": (), "storage_policy": None}
    )
    manifest_path = tmp_path / "snapshot-bank.json"
    manifest.save(manifest_path)
    return manifest_path, root, payloads


@dataclass
class Stored:
    payload: bytes
    info: ObjectInfo


class FakeGcsStore(ImmutableObjectStore):
    backend = "gcs"

    def __init__(self, *, fail_once: set[str] | None = None, delay: float = 0) -> None:
        self.objects: dict[str, Stored] = {}
        self.fail_once = set(fail_once or ())
        self.attempts: dict[str, int] = {}
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def stat(self, name: str) -> ObjectInfo | None:
        value = self.objects.get(name)
        return None if value is None else value.info

    def upload(self, source: Path, *, name: str, sha256: str, size: int) -> ObjectInfo:
        with self.lock:
            self.attempts[name] = self.attempts.get(name, 0) + 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if name in self.fail_once:
                self.fail_once.remove(name)
                raise ConnectionError("transient test failure")
            payload = source.read_bytes()
            existing = self.objects.get(name)
            if existing is not None:
                if existing.info.sha256 != sha256 or existing.info.size != size:
                    raise ObjectCollisionError("immutable object collision")
                return existing.info
            assert len(payload) == size
            assert hashlib.sha256(payload).hexdigest() == sha256
            info = ObjectInfo(
                name=name,
                size=size,
                sha256=sha256,
                generation=str(len(self.objects) + 1),
                uri=f"gs://test-bucket/{name}",
            )
            self.objects[name] = Stored(payload, info)
            return info
        finally:
            with self.lock:
                self.active -= 1

    def download(self, name: str, destination: Path) -> ObjectInfo:
        stored = self.objects[name]
        destination.write_bytes(stored.payload)
        return stored.info

    def list(self, prefix: str) -> tuple[ObjectInfo, ...]:
        return tuple(
            stored.info for name, stored in sorted(self.objects.items()) if name.startswith(prefix)
        )


def _config(tmp_path: Path, **updates: object) -> DatasetStoreConfig:
    values: dict[str, object] = {
        "backend": "gcs",
        "project": "test-project",
        "bucket": "test-bucket",
        "local_root": (tmp_path / "remote").as_posix(),
        "staging_root": (tmp_path / "staging").as_posix(),
        "cache_root": (tmp_path / "cache").as_posix(),
        "upload_workers": 2,
        "max_pending_uploads": 2,
        "upload_attempts": 3,
        "initial_backoff_seconds": 0,
    }
    values.update(updates)
    return DatasetStoreConfig.model_validate(values)


def _pack(tmp_path: Path, *, target: int = 1) -> tuple[Path, dict[str, bytes]]:
    source, root, payloads = _snapshot_bank(tmp_path)
    result = pack_snapshot_bank(
        snapshot_manifest_path=source,
        blob_root=root,
        staging_root=tmp_path / "staging",
        dataset_id="elder-test",
        run_id="run-001",
        target_shard_bytes=target,
        anthropocene_git_branch="feature/elder-rl-scaffold",
        training_git_commit="1" * 40,
    )
    return result.manifest_path, payloads


def test_serialization_shard_readback_manifest_and_cache(tmp_path: Path) -> None:
    manifest_path, payloads = _pack(tmp_path)
    staged = DatasetRunManifest.load(manifest_path)
    assert staged.dataset_schema_version == "anthropocene-dataset-v1"
    assert staged.publication_status == "staged"
    assert staged.counts.simulations == 3
    assert staged.counts.trajectories == 3
    assert staged.counts.states == 6
    assert staged.counts.samples >= 6
    assert len(staged.shards) == 3  # two decisions from each lineage stay together
    assert all(shard.record_count == 2 for shard in staged.shards)
    assert all(
        shard.object_name.endswith(f"/{shard.sha256[:2]}/{shard.sha256}.tar")
        for shard in staged.shards
    )
    assert staged.serialization.member_compression == "gzip"
    assert staged.serialization.shard_compression == "none-already-compressed-members"

    store = FakeGcsStore()
    DatasetPublisher(_config(tmp_path), store).publish(manifest_path)
    published = DatasetRunManifest.load(manifest_path)
    reader = DatasetReader(published, store=store, cache_root=tmp_path / "cache")
    first, second = published.shards[0].records
    assert reader.read_record(first.record_id) == payloads[first.record_id]
    assert reader.cache_statistics() == {"hits": 0, "misses": 1, "hit_ratio": 0.0}
    assert reader.read_record(second.record_id) == payloads[second.record_id]
    assert reader.cache_statistics()["hits"] == 1
    materialized = reader.materialize_record(first.record_id)
    assert materialized.read_bytes() == payloads[first.record_id]
    assert discover_runs(store, "elder-test") == ("run-001",)


def test_checksum_validation_rejects_corrupt_cache(tmp_path: Path) -> None:
    manifest_path, payloads = _pack(tmp_path)
    store = FakeGcsStore()
    DatasetPublisher(_config(tmp_path), store).publish(manifest_path)
    manifest = DatasetRunManifest.load(manifest_path)
    reader = DatasetReader(manifest, store=store, cache_root=tmp_path / "cache")
    record = manifest.shards[0].records[0]
    assert reader.read_record(record.record_id) == payloads[record.record_id]
    cached = reader.ensure_shard(manifest.shards[0])
    cached.write_bytes(b"corrupt")
    assert reader.read_record(record.record_id) == payloads[record.record_id]
    assert reader.cache_misses == 2


def test_dataset_reader_enforces_lru_cache_limit_without_touching_remote(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _pack(tmp_path)
    store = FakeGcsStore()
    DatasetPublisher(_config(tmp_path), store).publish(manifest_path)
    manifest = DatasetRunManifest.load(manifest_path)
    limit = max(shard.size_bytes for shard in manifest.shards)
    reader = DatasetReader(
        manifest,
        store=store,
        cache_root=tmp_path / "cache",
        max_cache_bytes=limit,
    )

    first_path = reader.ensure_shard(manifest.shards[0])
    assert first_path.is_file()
    second_path = reader.ensure_shard(manifest.shards[1])

    assert not first_path.exists()
    assert second_path.is_file()
    assert store.objects


def test_local_backend_has_the_same_publish_and_read_contract(tmp_path: Path) -> None:
    manifest_path, payloads = _pack(tmp_path)
    config = DatasetStoreConfig(
        backend="local",
        local_root=(tmp_path / "remote").as_posix(),
        cache_root=(tmp_path / "cache").as_posix(),
        upload_workers=2,
        max_pending_uploads=2,
        initial_backoff_seconds=0,
    )
    store = LocalObjectStore(config.local_root)
    result = DatasetPublisher(config, store).publish(manifest_path)
    assert result["status"] == "PASS"
    manifest = DatasetRunManifest.load(manifest_path)
    assert manifest.storage_backend == "local"
    reader = DatasetReader(manifest, store=store, cache_root=config.cache_root)
    first = manifest.shards[0].records[0]
    assert reader.read_record(first.record_id) == payloads[first.record_id]


def test_overfit_loader_fetches_a_missing_snapshot_from_dataset_cache(
    tmp_path: Path,
) -> None:
    source_manifest_path, blob_root, _ = _snapshot_bank(tmp_path)
    source_manifest = SnapshotBankManifest.load(source_manifest_path)
    normalizer = NormalizationManifest.load(
        "evidence/stage0-snapshot-bank/provisional-train-normalization.json"
    )
    normalizer_path = tmp_path / "normalizer.json"
    replace(
        normalizer,
        partition_manifest_sha256=source_manifest.sha256(),
    ).save(normalizer_path)
    config = DatasetStoreConfig(
        backend="local",
        local_root=(tmp_path / "remote").as_posix(),
        staging_root=(tmp_path / "staging").as_posix(),
        cache_root=(tmp_path / "cache").as_posix(),
        max_cache_gib=0.01,
        upload_workers=1,
        max_pending_uploads=1,
        initial_backoff_seconds=0,
    )
    packed = pack_snapshot_bank(
        snapshot_manifest_path=source_manifest_path,
        blob_root=blob_root,
        staging_root=config.staging_root,
        dataset_id="elder-stage0-test",
        run_id="run-001",
        target_shard_bytes=64 * 1024**2,
    )
    DatasetPublisher(config, LocalObjectStore(config.local_root)).publish(packed.manifest_path)
    config_path = tmp_path / "dataset-store.json"
    config_path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")

    selection = load_stage0_selection(
        manifest_path=source_manifest_path,
        blob_root=tmp_path / "empty-local-blobs",
        normalizer_path=normalizer_path,
        dataset_manifest=packed.manifest_path,
        dataset_store_config=config_path,
    )

    assert selection.snapshot_path.is_file()
    assert hashlib.sha256(selection.snapshot_bytes).hexdigest() == (selection.record.content_sha256)


def test_retry_bounded_concurrency_partial_restart_and_duplicate(tmp_path: Path) -> None:
    manifest_path, _ = _pack(tmp_path)
    manifest = DatasetRunManifest.load(manifest_path)
    failing_name = manifest.shards[1].object_name
    store = FakeGcsStore(fail_once={failing_name}, delay=0.01)
    result = DatasetPublisher(_config(tmp_path), store).publish(manifest_path)
    assert result["status"] == "PASS"
    assert store.attempts[failing_name] == 2
    assert store.max_active <= 2

    # Restarting a fully or partially uploaded run is idempotent and deduplicates.
    restarted = DatasetPublisher(_config(tmp_path), store).publish(manifest_path)
    assert restarted["uploaded"] == 0
    assert restarted["deduplicated"] == len(manifest.shards)


def test_interrupted_upload_recovers_from_journal_and_remote_state(tmp_path: Path) -> None:
    manifest_path, _ = _pack(tmp_path)
    manifest = DatasetRunManifest.load(manifest_path)
    failing = manifest.shards[1].object_name
    store = FakeGcsStore(fail_once={failing})
    config = _config(tmp_path, upload_workers=1, max_pending_uploads=1, upload_attempts=1)
    with pytest.raises(RuntimeError, match="transient test failure"):
        DatasetPublisher(config, store).publish(manifest_path)
    journal = json.loads((manifest_path.parent / "upload-state.json").read_text())
    assert journal["objects"][failing]["status"] == "failed"
    result = DatasetPublisher(config.model_copy(update={"upload_attempts": 3}), store).publish(
        manifest_path
    )
    assert result["status"] == "PASS"


def test_differing_data_at_same_object_identity_is_rejected(tmp_path: Path) -> None:
    manifest_path, _ = _pack(tmp_path)
    manifest = DatasetRunManifest.load(manifest_path)
    store = FakeGcsStore()
    first = manifest.shards[0]
    store.objects[first.object_name] = Stored(
        b"different",
        ObjectInfo(
            name=first.object_name,
            size=9,
            sha256="0" * 64,
            generation="1",
            uri=f"gs://test-bucket/{first.object_name}",
        ),
    )
    with pytest.raises(RuntimeError, match="collision"):
        DatasetPublisher(_config(tmp_path), store).publish(manifest_path)


def test_missing_remote_checksum_metadata_is_rejected(tmp_path: Path) -> None:
    manifest_path, _ = _pack(tmp_path)
    manifest = DatasetRunManifest.load(manifest_path)
    store = FakeGcsStore()
    first = manifest.shards[0]
    store.objects[first.object_name] = Stored(
        b"x" * first.size_bytes,
        ObjectInfo(
            name=first.object_name,
            size=first.size_bytes,
            sha256=None,
            generation="1",
            uri=f"gs://test-bucket/{first.object_name}",
        ),
    )
    with pytest.raises(RuntimeError, match="collision"):
        DatasetPublisher(_config(tmp_path), store).publish(manifest_path)


def test_gcs_adapter_uses_generation_precondition_and_sha_metadata(tmp_path: Path) -> None:
    pytest.importorskip("google.cloud.storage")
    from google.api_core.exceptions import NotFound

    class Blob:
        def __init__(self, bucket: Bucket, name: str) -> None:
            self.bucket = bucket
            self.name = name
            self.payload = b""
            self.metadata: dict[str, str] = {}
            self.generation: str | None = None
            self.size = 0
            self.upload_options: dict[str, object] = {}
            self.download_ranges: list[tuple[int, int]] = []

        def reload(self, **_kwargs: object) -> None:
            if self.name not in self.bucket.objects:
                raise NotFound("missing")

        def upload_from_filename(self, filename: str, **kwargs: object) -> None:
            self.upload_options = kwargs
            self.payload = Path(filename).read_bytes()
            self.size = len(self.payload)
            self.generation = "1"
            self.bucket.objects[self.name] = self

        def download_as_bytes(self, *, start: int, end: int, **_kwargs: object) -> bytes:
            self.download_ranges.append((start, end))
            return self.payload[start : end + 1]

    class Bucket:
        def __init__(self) -> None:
            self.objects: dict[str, Blob] = {}
            self.blob_options: list[dict[str, object]] = []

        def blob(self, name: str, **kwargs: object) -> Blob:
            self.blob_options.append(kwargs)
            return self.objects.get(name, Blob(self, name))

    class Client:
        def __init__(self) -> None:
            self.value = Bucket()

        def bucket(self, _name: str) -> Bucket:
            return self.value

        def list_blobs(self, bucket: Bucket, *, prefix: str) -> list[Blob]:
            return [blob for name, blob in bucket.objects.items() if name.startswith(prefix)]

    source = tmp_path / "shard.tar"
    source.write_bytes(b"immutable-shard")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    client = Client()
    store = GcsObjectStore(_config(tmp_path), client=client)
    info = store.upload(source, name="objects/sha256/test.tar", sha256=digest, size=15)
    blob = client.value.objects["objects/sha256/test.tar"]
    assert info.sha256 == digest
    assert blob.metadata["anthropocene-sha256"] == digest
    assert blob.upload_options["if_generation_match"] == 0
    assert blob.upload_options["checksum"] == "crc32c"
    destination = tmp_path / "downloaded.tar"
    store.download("objects/sha256/test.tar", destination)
    assert destination.read_bytes() == b"immutable-shard"
    assert blob.download_ranges == [(0, 14)]
    assert store.list("objects/sha256/") == (info,)


def test_unknown_dataset_schema_is_rejected(tmp_path: Path) -> None:
    path, _ = _pack(tmp_path)
    payload = json.loads(path.read_text())
    payload["dataset_schema_version"] = "anthropocene-dataset-v999"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported dataset schema"):
        DatasetRunManifest.load(path)


def test_tar_is_deterministic_and_source_blobs_remain(tmp_path: Path) -> None:
    source, root, _ = _snapshot_bank(tmp_path)
    first = pack_snapshot_bank(
        snapshot_manifest_path=source,
        blob_root=root,
        staging_root=tmp_path / "one",
        dataset_id="elder-test",
        run_id="run-001",
        target_shard_bytes=1000000,
    )
    second = pack_snapshot_bank(
        snapshot_manifest_path=source,
        blob_root=root,
        staging_root=tmp_path / "two",
        dataset_id="elder-test",
        run_id="run-001",
        target_shard_bytes=1000000,
    )
    one = DatasetRunManifest.load(first.manifest_path).shards[0]
    two = DatasetRunManifest.load(second.manifest_path).shards[0]
    assert one.sha256 == two.sha256
    assert one.size_bytes == two.size_bytes
    assert len(list(root.rglob("*.snapshot"))) == 6
    assert sha256_file(first.manifest_path.parent / "shards" / one.name)[0] == one.sha256
