from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from anthropocene_rl.datasets.models import (
    RemoteBlobMetadata,
    SnapshotBankManifest,
    SnapshotRecord,
    SnapshotStoragePolicy,
)
from anthropocene_rl.datasets.storage import (
    RemoteObject,
    SnapshotStoreConfig,
    SnapshotStoreManager,
    md5_file,
    remote_key_for,
    resolve_snapshot_blob,
)


class FakeRemoteStore:
    backend = "google-drive"

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, RemoteObject]] = {}
        self.download_count = 0

    def find(self, key: str) -> RemoteObject | None:
        for _, remote in self.objects.values():
            if remote.key == key:
                return remote
        return None

    def get(self, file_id: str) -> RemoteObject:
        return self.objects[file_id][1]

    def upload(
        self,
        source: Path,
        *,
        key: str,
        content_sha256: str,
        content_bytes: int,
    ) -> RemoteObject:
        payload = source.read_bytes()
        file_id = f"drive-{len(self.objects) + 1}"
        remote = RemoteObject(
            file_id=file_id,
            key=key,
            size=content_bytes,
            md5_checksum=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
            version="1",
            modified_time="2026-10-11T00:00:00Z",
            content_sha256=content_sha256,
            sha256_checksum=content_sha256,
        )
        self.objects[file_id] = (payload, remote)
        return remote

    def download(self, file_id: str, destination: Path) -> RemoteObject:
        payload, remote = self.objects[file_id]
        destination.write_bytes(payload)
        self.download_count += 1
        return remote


def _mini_manifest(
    tmp_path: Path,
) -> tuple[SnapshotBankManifest, SnapshotRecord, Path, bytes]:
    source = SnapshotBankManifest.load("datasets/manifests/elder-v2-stage0-v1.json")
    payload = b"immutable SaveCodec test payload"
    digest = hashlib.sha256(payload).hexdigest()
    relative = f"sha256/{digest[:2]}/{digest}.snapshot"
    record_payload = source.records[0].model_dump(mode="python", by_alias=True)
    record_payload.update(
        {
            "snapshot_id": "storage-test-snapshot",
            "content_sha256": digest,
            "content_bytes": len(payload),
            "blob_path": relative,
            "remote_storage": None,
        }
    )
    record = SnapshotRecord.model_validate(record_payload)
    manifest = source.model_copy(
        update={"records": (record,), "rejected_candidates": (), "storage_policy": None}
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    blob = tmp_path / "cache" / relative
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    return manifest, record, manifest_path, payload


def _config(tmp_path: Path) -> SnapshotStoreConfig:
    return SnapshotStoreConfig(
        folder_id="stage0-folder",
        folder_path="Anthropocene / RL Snapshot Banks / Elder v2 / Stage0 and Pilot Banks",
        local_cache_root=(tmp_path / "cache").as_posix(),
        max_local_gb=5.0,
    )


def test_remote_metadata_does_not_change_snapshot_bank_identity() -> None:
    manifest = SnapshotBankManifest.load("datasets/manifests/elder-v2-stage0-v1.json")
    record = manifest.records[0]
    metadata = RemoteBlobMetadata(
        backend="google-drive",
        file_id="drive-file",
        remote_key=remote_key_for(record.content_sha256),
        verification_status="metadata-verified",
        content_sha256=record.content_sha256,
        content_bytes=record.content_bytes,
        md5_checksum="0" * 32,
    )
    records = (record.model_copy(update={"remote_storage": metadata}), *manifest.records[1:])
    with_storage = SnapshotBankManifest.model_validate(
        {
            **manifest.model_dump(mode="python"),
            "records": records,
            "storage_policy": SnapshotStoragePolicy(
                remote_root_id="stage0-folder",
                remote_root_path="Stage0 and Pilot Banks",
            ),
        }
    )

    assert with_storage.sha256() == manifest.sha256()


def test_push_fetch_cache_hit_and_safe_prune(tmp_path: Path) -> None:
    manifest, record, manifest_path, payload = _mini_manifest(tmp_path)
    remote = FakeRemoteStore()
    manager = SnapshotStoreManager(
        manifest_path=manifest_path,
        blob_root=tmp_path / "cache",
        config=_config(tmp_path),
        remote=remote,
    )

    pushed = manager.push()
    assert pushed["status"] == "PASS"
    assert pushed["uploaded"] == 1
    assert pushed["cache_prune"]["status"] == "PASS"
    updated = SnapshotBankManifest.load(manifest_path)
    assert updated.sha256() == manifest.sha256()
    assert updated.records[0].remote_storage is not None
    remote_md5 = next(iter(remote.objects.values()))[1].md5_checksum
    assert md5_file(manager.cache.path_for(record)) == remote_md5

    manager.cache.path_for(record).unlink()
    fetched = manager.fetch(record.snapshot_id)
    assert fetched["fetched"] is True
    assert fetched["cache_prune"]["status"] == "PASS"
    assert manager.cache.path_for(record).read_bytes() == payload
    assert remote.download_count == 1

    resolved = resolve_snapshot_blob(
        SnapshotBankManifest.load(manifest_path).records[0],
        blob_root=tmp_path / "cache",
        remote=remote,
    )
    assert resolved.read_bytes() == payload
    assert remote.download_count == 1

    pruned = manager.prune(max_local_bytes=0)
    assert pruned["status"] == "PASS"
    assert pruned["evicted_snapshot_ids"] == [record.snapshot_id]
    assert not manager.cache.path_for(record).exists()
    assert remote.objects


def test_status_reports_incomplete_remote_durability(tmp_path: Path) -> None:
    _, record, manifest_path, _ = _mini_manifest(tmp_path)
    manager = SnapshotStoreManager(
        manifest_path=manifest_path,
        blob_root=tmp_path / "cache",
        config=_config(tmp_path),
        remote=FakeRemoteStore(),
    )

    result = manager.status(check_remote=False)

    assert result["status"] == "PARTIAL"
    assert result["remote"]["missing_metadata_snapshot_ids"] == [record.snapshot_id]
    assert result["unrecoverable_snapshot_ids"] == []

    manager.cache.path_for(record).unlink()
    missing = manager.status(check_remote=False)
    assert missing["status"] == "FAIL"
    assert missing["unrecoverable_snapshot_ids"] == [record.snapshot_id]


def test_push_enforces_configured_cache_limit_after_remote_verification(
    tmp_path: Path,
) -> None:
    _, record, manifest_path, _ = _mini_manifest(tmp_path)
    config = _config(tmp_path).model_copy(update={"max_local_gb": 1 / 1024**3})
    manager = SnapshotStoreManager(
        manifest_path=manifest_path,
        blob_root=tmp_path / "cache",
        config=config,
        remote=FakeRemoteStore(),
    )

    result = manager.push()

    assert result["status"] == "PASS"
    assert result["cache_prune"]["evicted_snapshot_ids"] == [record.snapshot_id]
    assert not manager.cache.path_for(record).exists()


def test_prune_refuses_a_changed_remote_object_version(tmp_path: Path) -> None:
    _, record, manifest_path, _ = _mini_manifest(tmp_path)
    remote = FakeRemoteStore()
    manager = SnapshotStoreManager(
        manifest_path=manifest_path,
        blob_root=tmp_path / "cache",
        config=_config(tmp_path),
        remote=remote,
    )
    manager.push()
    file_id, (payload, remote_object) = next(iter(remote.objects.items()))
    remote.objects[file_id] = (payload, replace(remote_object, version="2"))

    result = manager.prune(max_local_bytes=0)

    assert result["status"] == "PARTIAL"
    assert "version differs" in result["skipped"][record.snapshot_id]
    assert manager.cache.path_for(record).is_file()


def test_sync_refreshes_version_when_verified_local_bytes_are_unchanged(
    tmp_path: Path,
) -> None:
    _, _record, manifest_path, _ = _mini_manifest(tmp_path)
    remote = FakeRemoteStore()
    manager = SnapshotStoreManager(
        manifest_path=manifest_path,
        blob_root=tmp_path / "cache",
        config=_config(tmp_path),
        remote=remote,
    )
    manager.push()
    file_id, (payload, remote_object) = next(iter(remote.objects.items()))
    remote.objects[file_id] = (payload, replace(remote_object, version="2"))

    result = manager.sync()

    assert result["status"] == "PASS"
    updated = SnapshotBankManifest.load(manifest_path)
    assert updated.records[0].remote_storage is not None
    assert updated.records[0].remote_storage.version == "2"
    assert manager.status()["status"] == "PASS"


def test_prune_refuses_a_provider_sha256_mismatch(tmp_path: Path) -> None:
    _, record, manifest_path, _ = _mini_manifest(tmp_path)
    remote = FakeRemoteStore()
    manager = SnapshotStoreManager(
        manifest_path=manifest_path,
        blob_root=tmp_path / "cache",
        config=_config(tmp_path),
        remote=remote,
    )
    manager.push()
    file_id, (payload, remote_object) = next(iter(remote.objects.items()))
    remote.objects[file_id] = (
        payload,
        replace(remote_object, sha256_checksum="0" * 64),
    )

    result = manager.prune(max_local_bytes=0)

    assert result["status"] == "PARTIAL"
    assert "provider SHA-256 differs" in result["skipped"][record.snapshot_id]
    assert manager.cache.path_for(record).is_file()


def test_prune_never_removes_the_only_copy(tmp_path: Path) -> None:
    _, record, manifest_path, _ = _mini_manifest(tmp_path)
    manager = SnapshotStoreManager(
        manifest_path=manifest_path,
        blob_root=tmp_path / "cache",
        config=_config(tmp_path),
        remote=FakeRemoteStore(),
    )

    result = manager.prune(max_local_bytes=0)

    assert result["status"] == "PARTIAL"
    assert record.snapshot_id in result["skipped"]
    assert manager.cache.path_for(record).is_file()
