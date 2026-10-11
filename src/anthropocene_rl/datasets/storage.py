"""Durable remote snapshot storage with a bounded local content cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field

from anthropocene_rl.config import load_yaml
from anthropocene_rl.datasets.blobs import sha256_file
from anthropocene_rl.datasets.models import (
    RemoteBlobMetadata,
    SnapshotBankManifest,
    SnapshotRecord,
    SnapshotStoragePolicy,
    StrictModel,
)

DEFAULT_SNAPSHOT_STORE_CONFIG = Path("configs/storage/google-drive-v1.yaml")


class SnapshotStoreConfig(StrictModel):
    config_schema_version: Literal["snapshot-store-v1"] = "snapshot-store-v1"
    backend: Literal["google-drive"] = "google-drive"
    folder_id: str = Field(min_length=1)
    folder_path: str = Field(min_length=1)
    remote_prefix: Literal["sha256"] = "sha256"
    local_cache_root: str = "datasets/blobs"
    max_local_gb: float = Field(default=5.0, gt=0)
    eviction_policy: Literal["lru"] = "lru"

    @property
    def max_local_bytes(self) -> int:
        return int(self.max_local_gb * 1024**3)

    @property
    def storage_policy(self) -> SnapshotStoragePolicy:
        return SnapshotStoragePolicy(
            max_local_bytes=self.max_local_bytes,
            remote_root_id=self.folder_id,
            remote_root_path=self.folder_path,
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SNAPSHOT_STORE_CONFIG) -> Self:
        return cls.model_validate(load_yaml(path))


@dataclass(frozen=True)
class RemoteObject:
    file_id: str
    key: str
    size: int
    md5_checksum: str | None
    version: str | None
    modified_time: str | None
    content_sha256: str | None


class RemoteBlobStore(Protocol):
    backend: str

    def find(self, key: str) -> RemoteObject | None: ...

    def get(self, file_id: str) -> RemoteObject: ...

    def upload(
        self,
        source: Path,
        *,
        key: str,
        content_sha256: str,
        content_bytes: int,
    ) -> RemoteObject: ...

    def download(self, file_id: str, destination: Path) -> RemoteObject: ...


def remote_key_for(content_sha256: str) -> str:
    if len(content_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in content_sha256
    ):
        raise ValueError("Expected a lowercase SHA-256 digest")
    return f"sha256/{content_sha256[:2]}/{content_sha256}.snapshot"


def md5_file(path: str | Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class _CacheAccessIndex:
    def __init__(self, root: Path) -> None:
        self.path = root / ".snapshot-cache-access.json"

    def last_access(self, digest: str, fallback: float) -> float:
        return self._load().get(digest, fallback)

    def touch(self, digest: str) -> None:
        values = self._load()
        values[digest] = time.time()
        self._save(values)

    def remove(self, digest: str) -> None:
        values = self._load()
        if digest in values:
            del values[digest]
            self._save(values)

    def _load(self) -> dict[str, float]:
        if not self.path.is_file():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Snapshot cache access index must be a JSON object")
        return {str(key): float(value) for key, value in payload.items()}

    def _save(self, values: dict[str, float]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(self.path, dict(sorted(values.items())))


class SnapshotCache:
    def __init__(self, root: str | Path, remote: RemoteBlobStore | None = None) -> None:
        self.root = Path(root)
        self.remote = remote
        self.access = _CacheAccessIndex(self.root)

    def path_for(self, record: SnapshotRecord) -> Path:
        path = (self.root / record.blob_path).resolve()
        root = self.root.resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Snapshot blob path escapes the configured cache") from error
        return path

    def ensure_local(self, record: SnapshotRecord) -> tuple[Path, bool, RemoteObject | None]:
        path = self.path_for(record)
        if path.is_file():
            _verify_local(path, record)
            self.access.touch(record.content_sha256)
            return path, False, None
        metadata = record.remote_storage
        if metadata is None:
            raise FileNotFoundError(
                f"Snapshot {record.snapshot_id!r} is absent locally and has no remote identity"
            )
        if self.remote is None:
            raise FileNotFoundError(
                f"Snapshot {record.snapshot_id!r} requires the configured Google Drive store"
            )
        remote_object = self.remote.get(metadata.file_id)
        errors = _remote_errors(record, remote_object, expected_file_id=metadata.file_id)
        if errors:
            raise ValueError("Remote snapshot metadata mismatch: " + "; ".join(errors))
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".download",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            downloaded = self.remote.download(metadata.file_id, temporary)
            download_errors = _remote_errors(
                record,
                downloaded,
                expected_file_id=metadata.file_id,
            )
            if download_errors:
                raise ValueError(
                    "Downloaded remote metadata mismatch: " + "; ".join(download_errors)
                )
            _verify_local(temporary, record)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.access.touch(record.content_sha256)
        return path, True, remote_object


class SnapshotStoreManager:
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        blob_root: str | Path,
        config: SnapshotStoreConfig,
        remote: RemoteBlobStore,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.blob_root = Path(blob_root)
        self.config = config
        self.remote = remote
        self.cache = SnapshotCache(self.blob_root, remote)

    def push(self) -> dict[str, object]:
        return self._reconcile("push")

    def sync(self) -> dict[str, object]:
        return self._reconcile("sync")

    def fetch(self, selector: str) -> dict[str, object]:
        manifest = SnapshotBankManifest.load(self.manifest_path)
        record = _select_record(manifest, selector)
        path, fetched, remote_object = self.cache.ensure_local(record)
        if fetched and remote_object is not None:
            metadata = _remote_metadata(record, remote_object, "download-verified")
            records = tuple(
                candidate.model_copy(update={"remote_storage": metadata})
                if candidate.snapshot_id == record.snapshot_id
                else candidate
                for candidate in manifest.records
            )
            updated = SnapshotBankManifest.model_validate(
                {
                    **manifest.model_dump(mode="python"),
                    "records": records,
                    "storage_policy": self.config.storage_policy,
                }
            )
            if updated.sha256() != manifest.sha256():
                raise AssertionError("Storage metadata changed snapshot-bank identity")
            updated.save_atomic(self.manifest_path)
        cache_prune = self.prune(
            max_local_bytes=self.config.max_local_bytes,
            protected_snapshot_ids=frozenset({record.snapshot_id}),
        )
        return {
            "status": "PASS" if cache_prune["status"] == "PASS" else "PARTIAL",
            "snapshot_id": record.snapshot_id,
            "content_sha256": record.content_sha256,
            "path": path.as_posix(),
            "fetched": fetched,
            "content_bytes": record.content_bytes,
            "cache_prune": cache_prune,
        }

    def status(self, *, check_remote: bool = True) -> dict[str, object]:
        manifest = SnapshotBankManifest.load(self.manifest_path)
        local_bytes = 0
        local_count = 0
        local_missing: list[str] = []
        local_invalid: list[str] = []
        remote_verified = 0
        remote_metadata_missing: list[str] = []
        remote_invalid: list[str] = []
        for record in manifest.records:
            path = self.cache.path_for(record)
            if path.is_file():
                try:
                    _verify_local(path, record)
                    local_bytes += record.content_bytes
                    local_count += 1
                except ValueError:
                    local_invalid.append(record.snapshot_id)
            else:
                local_missing.append(record.snapshot_id)
            if record.remote_storage is None:
                remote_metadata_missing.append(record.snapshot_id)
            elif check_remote:
                try:
                    remote_object = self.remote.get(record.remote_storage.file_id)
                    errors = _remote_errors(
                        record,
                        remote_object,
                        expected_file_id=record.remote_storage.file_id,
                    )
                except Exception as error:  # provider errors belong in the audit result
                    errors = [str(error)]
                if errors:
                    remote_invalid.append(record.snapshot_id)
                else:
                    remote_verified += 1
        unrecoverable = sorted(set(local_missing) & set(remote_metadata_missing))
        over_cache_limit = local_bytes > self.config.max_local_bytes
        if local_invalid or remote_invalid or unrecoverable:
            status = "FAIL"
        elif remote_metadata_missing or over_cache_limit:
            status = "PARTIAL"
        else:
            status = "PASS"
        return {
            "status": status,
            "manifest_identity_sha256": manifest.sha256(),
            "record_count": len(manifest.records),
            "local": {
                "count": local_count,
                "bytes": local_bytes,
                "missing_snapshot_ids": local_missing,
                "invalid_snapshot_ids": local_invalid,
                "configured_max_bytes": self.config.max_local_bytes,
                "over_configured_limit": over_cache_limit,
            },
            "remote": {
                "metadata_count": sum(
                    record.remote_storage is not None for record in manifest.records
                ),
                "live_verified_count": remote_verified if check_remote else None,
                "missing_metadata_snapshot_ids": remote_metadata_missing,
                "invalid_snapshot_ids": remote_invalid,
                "backend": self.remote.backend,
                "folder_id": self.config.folder_id,
            },
            "unrecoverable_snapshot_ids": unrecoverable,
        }

    def prune(
        self,
        *,
        max_local_bytes: int,
        protected_snapshot_ids: frozenset[str] = frozenset(),
    ) -> dict[str, object]:
        if max_local_bytes < 0:
            raise ValueError("max_local_bytes cannot be negative")
        manifest = SnapshotBankManifest.load(self.manifest_path)
        all_blobs = tuple(self.blob_root.glob("sha256/*/*.snapshot"))
        total_before = sum(path.stat().st_size for path in all_blobs if path.is_file())
        total = total_before
        candidates: list[tuple[float, SnapshotRecord, Path]] = []
        skipped: dict[str, str] = {}
        for record in manifest.records:
            path = self.cache.path_for(record)
            if not path.is_file():
                continue
            if record.snapshot_id in protected_snapshot_ids:
                skipped[record.snapshot_id] = "protected-current-fetch"
                continue
            metadata = record.remote_storage
            if metadata is None:
                skipped[record.snapshot_id] = "no-verified-remote-identity"
                continue
            fallback = path.stat().st_mtime
            candidates.append(
                (self.cache.access.last_access(record.content_sha256, fallback), record, path)
            )
        candidates.sort(key=lambda item: (item[0], item[1].snapshot_id))
        evicted: list[str] = []
        reclaimed = 0
        for _, record, path in candidates:
            if total <= max_local_bytes:
                break
            try:
                _verify_local(path, record)
                metadata = record.remote_storage
                assert metadata is not None
                remote_object = self.remote.get(metadata.file_id)
                errors = _remote_errors(
                    record,
                    remote_object,
                    expected_file_id=metadata.file_id,
                )
                if errors:
                    skipped[record.snapshot_id] = "; ".join(errors)
                    continue
            except Exception as error:
                skipped[record.snapshot_id] = str(error)
                continue
            size = path.stat().st_size
            path.unlink()
            self.cache.access.remove(record.content_sha256)
            total -= size
            reclaimed += size
            evicted.append(record.snapshot_id)
        return {
            "status": "PASS" if total <= max_local_bytes else "PARTIAL",
            "max_local_bytes": max_local_bytes,
            "bytes_before": total_before,
            "bytes_after": total,
            "bytes_reclaimed": reclaimed,
            "evicted_snapshot_ids": evicted,
            "skipped": skipped,
        }

    def _reconcile(self, operation: str) -> dict[str, object]:
        manifest = SnapshotBankManifest.load(self.manifest_path)
        identity = manifest.sha256()
        records: list[SnapshotRecord] = []
        uploaded = 0
        deduplicated = 0
        remote_only = 0
        errors: list[str] = []
        for record in manifest.records:
            path = self.cache.path_for(record)
            key = remote_key_for(record.content_sha256)
            if path.is_file():
                try:
                    _verify_local(path, record)
                    local_md5 = md5_file(path)
                    remote_object = self.remote.find(key)
                    if remote_object is None:
                        remote_object = self.remote.upload(
                            path,
                            key=key,
                            content_sha256=record.content_sha256,
                            content_bytes=record.content_bytes,
                        )
                        uploaded += 1
                    else:
                        deduplicated += 1
                    remote_errors = _remote_errors(
                        record,
                        remote_object,
                        expected_md5=local_md5,
                    )
                except Exception as error:
                    errors.append(f"{record.snapshot_id}: {error}")
                    records.append(record)
                    continue
            elif record.remote_storage is not None:
                try:
                    remote_object = self.remote.get(record.remote_storage.file_id)
                    remote_errors = _remote_errors(
                        record,
                        remote_object,
                        expected_file_id=record.remote_storage.file_id,
                    )
                    remote_only += 1
                except Exception as error:
                    errors.append(f"{record.snapshot_id}: {error}")
                    records.append(record)
                    continue
            else:
                errors.append(f"{record.snapshot_id}: no local or remote blob copy")
                records.append(record)
                continue
            if remote_errors:
                errors.append(f"{record.snapshot_id}: " + "; ".join(remote_errors))
                records.append(record)
                continue
            verification: Literal["metadata-verified", "download-verified"] = (
                "download-verified"
                if record.remote_storage is not None
                and record.remote_storage.verification_status == "download-verified"
                else "metadata-verified"
            )
            records.append(
                record.model_copy(
                    update={
                        "remote_storage": _remote_metadata(
                            record,
                            remote_object,
                            verification,
                        )
                    }
                )
            )
        if errors:
            raise RuntimeError(
                f"Snapshot-store {operation} failed without changing the manifest: "
                + " | ".join(errors)
            )
        updated = SnapshotBankManifest.model_validate(
            {
                **manifest.model_dump(mode="python"),
                "records": tuple(records),
                "storage_policy": self.config.storage_policy,
            }
        )
        if updated.sha256() != identity:
            raise AssertionError("Storage metadata changed snapshot-bank identity")
        updated.save_atomic(self.manifest_path)
        cache_prune = self.prune(max_local_bytes=self.config.max_local_bytes)
        return {
            "status": "PASS" if cache_prune["status"] == "PASS" else "PARTIAL",
            "operation": operation,
            "manifest_identity_sha256": identity,
            "records": len(records),
            "uploaded": uploaded,
            "deduplicated": deduplicated,
            "remote_only": remote_only,
            "folder_id": self.config.folder_id,
            "cache_prune": cache_prune,
        }


def build_snapshot_store_manager(
    *,
    manifest_path: str | Path,
    config_path: str | Path = DEFAULT_SNAPSHOT_STORE_CONFIG,
    blob_root: str | Path | None = None,
    credentials_path: str | Path | None = None,
) -> SnapshotStoreManager:
    from anthropocene_rl.datasets.google_drive import GoogleDriveRemoteStore

    config = SnapshotStoreConfig.load(config_path)
    credentials = credentials_path or os.environ.get(
        "ANTHROPOCENE_GOOGLE_DRIVE_CREDENTIALS"
    )
    remote = GoogleDriveRemoteStore(
        root_folder_id=config.folder_id,
        credentials_path=credentials,
    )
    return SnapshotStoreManager(
        manifest_path=manifest_path,
        blob_root=blob_root or config.local_cache_root,
        config=config,
        remote=remote,
    )


def resolve_snapshot_blob(
    record: SnapshotRecord,
    *,
    blob_root: str | Path,
    remote: RemoteBlobStore | None = None,
    config_path: str | Path = DEFAULT_SNAPSHOT_STORE_CONFIG,
    credentials_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    cache = SnapshotCache(blob_root, remote)
    path = cache.path_for(record)
    if path.is_file():
        return cache.ensure_local(record)[0]
    if record.remote_storage is None:
        return cache.ensure_local(record)[0]
    if remote is None:
        manager = build_snapshot_store_manager(
            manifest_path=Path("unused-by-loader.json"),
            config_path=config_path,
            blob_root=blob_root,
            credentials_path=credentials_path,
        )
        cache = manager.cache
    path, fetched, _ = cache.ensure_local(record)
    if fetched and manifest_path is not None:
        config = SnapshotStoreConfig.load(config_path)
        assert cache.remote is not None
        manager = SnapshotStoreManager(
            manifest_path=manifest_path,
            blob_root=blob_root,
            config=config,
            remote=cache.remote,
        )
        manager.prune(
            max_local_bytes=config.max_local_bytes,
            protected_snapshot_ids=frozenset({record.snapshot_id}),
        )
    return path


def _select_record(manifest: SnapshotBankManifest, selector: str) -> SnapshotRecord:
    exact = [
        record
        for record in manifest.records
        if record.snapshot_id == selector or record.content_sha256 == selector
    ]
    if len(exact) == 1:
        return exact[0]
    if not exact and 8 <= len(selector) < 64:
        prefixed = [
            record for record in manifest.records if record.content_sha256.startswith(selector)
        ]
        if len(prefixed) == 1:
            return prefixed[0]
        if len(prefixed) > 1:
            raise ValueError(f"SHA selector {selector!r} is ambiguous")
    raise ValueError(f"Expected one snapshot matching {selector!r}; found {len(exact)}")


def _verify_local(path: Path, record: SnapshotRecord) -> None:
    digest, size = sha256_file(path)
    if digest != record.content_sha256 or size != record.content_bytes:
        raise ValueError(
            f"Local blob for {record.snapshot_id!r} does not match its SHA-256/size"
        )


def _remote_errors(
    record: SnapshotRecord,
    remote: RemoteObject,
    *,
    expected_file_id: str | None = None,
    expected_md5: str | None = None,
) -> list[str]:
    expected_key = remote_key_for(record.content_sha256)
    errors: list[str] = []
    if expected_file_id is not None and remote.file_id != expected_file_id:
        errors.append("Drive file ID changed")
    if remote.key != expected_key:
        errors.append("remote key changed")
    if remote.size != record.content_bytes:
        errors.append("remote size differs from manifest")
    if remote.content_sha256 != record.content_sha256:
        errors.append("remote SHA-256 app property differs from manifest")
    if expected_md5 is not None and remote.md5_checksum != expected_md5:
        errors.append("Drive MD5 differs from local bytes")
    metadata = record.remote_storage
    if (
        metadata is not None
        and metadata.md5_checksum is not None
        and remote.md5_checksum != metadata.md5_checksum
    ):
        errors.append("Drive MD5 differs from recorded remote identity")
    if (
        metadata is not None
        and metadata.version is not None
        and remote.version != metadata.version
    ):
        errors.append("Drive object version differs from recorded remote identity")
    return errors


def _remote_metadata(
    record: SnapshotRecord,
    remote: RemoteObject,
    verification_status: Literal["metadata-verified", "download-verified"],
) -> RemoteBlobMetadata:
    return RemoteBlobMetadata(
        backend="google-drive",
        file_id=remote.file_id,
        remote_key=remote.key,
        verification_status=verification_status,
        content_sha256=record.content_sha256,
        content_bytes=record.content_bytes,
        md5_checksum=remote.md5_checksum,
        uploaded_at=remote.modified_time,
        version=remote.version,
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
