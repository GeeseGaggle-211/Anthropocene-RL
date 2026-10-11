"""Immutable local/GCS objects and bounded, restartable dataset publication."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast

from pydantic import Field, model_validator

from anthropocene_rl.config import load_yaml
from anthropocene_rl.datasets.blobs import sha256_file
from anthropocene_rl.datasets.dataset_manifest import DatasetRunManifest
from anthropocene_rl.datasets.models import StrictModel

DEFAULT_DATASET_STORE_CONFIG = Path("configs/storage/gcs-v1.yaml")


class DatasetStoreConfig(StrictModel):
    config_schema_version: Literal["dataset-store-v1"] = "dataset-store-v1"
    backend: Literal["local", "gcs"] = "local"
    project: str | None = None
    bucket: str | None = None
    location: str | None = None
    storage_class: Literal["STANDARD", "NEARLINE", "COLDLINE", "ARCHIVE"] = (
        "STANDARD"
    )
    prefix: str = ""
    local_root: str = "datasets/remote"
    staging_root: str = "datasets/staging"
    cache_root: str = "datasets/cache"
    target_shard_mib: int = Field(default=64, ge=1)
    upload_workers: int = Field(default=4, ge=1, le=32)
    max_pending_uploads: int = Field(default=8, ge=1, le=128)
    upload_attempts: int = Field(default=5, ge=1, le=20)
    initial_backoff_seconds: float = Field(default=0.5, ge=0, le=60)
    resumable_chunk_mib: int = Field(default=16, ge=1, le=256)
    request_timeout_seconds: int = Field(default=120, ge=1)

    @model_validator(mode="after")
    def validate_provider(self) -> Self:
        if self.max_pending_uploads < self.upload_workers:
            raise ValueError("max_pending_uploads must be at least upload_workers")
        if self.backend == "gcs" and (not self.project or not self.bucket):
            raise ValueError("GCS storage requires project and bucket")
        if self.prefix.startswith("/") or self.prefix.endswith("/"):
            raise ValueError("prefix must be empty or omit leading/trailing slashes")
        return self

    @property
    def target_shard_bytes(self) -> int:
        return self.target_shard_mib * 1024**2

    @classmethod
    def load(cls, path: str | Path = DEFAULT_DATASET_STORE_CONFIG) -> Self:
        return cls.model_validate(load_yaml(path))


@dataclass(frozen=True)
class ObjectInfo:
    name: str
    size: int
    sha256: str | None
    generation: str | None
    uri: str


class ImmutableObjectStore(Protocol):
    backend: str

    def stat(self, name: str) -> ObjectInfo | None: ...

    def upload(self, source: Path, *, name: str, sha256: str, size: int) -> ObjectInfo: ...

    def download(self, name: str, destination: Path) -> ObjectInfo: ...

    def list(self, prefix: str) -> tuple[ObjectInfo, ...]: ...


class ObjectCollisionError(RuntimeError):
    """An immutable object name already contains different bytes."""


def _validate_info(
    info: ObjectInfo,
    *,
    name: str,
    sha256: str,
    size: int,
) -> None:
    errors: list[str] = []
    if info.name != name:
        errors.append("object name differs")
    if info.size != size:
        errors.append("object size differs")
    if info.sha256 is None:
        errors.append("object SHA-256 metadata is missing")
    elif info.sha256 != sha256:
        errors.append("object SHA-256 differs")
    if errors:
        raise ObjectCollisionError(
            f"Immutable object collision at {info.uri}: " + "; ".join(errors)
        )


class LocalObjectStore:
    backend = "local"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("Object name escapes local store root") from error
        return path

    def stat(self, name: str) -> ObjectInfo | None:
        path = self._path(name)
        if not path.is_file():
            return None
        digest, size = sha256_file(path)
        return ObjectInfo(
            name=name,
            size=size,
            sha256=digest,
            generation=str(path.stat().st_mtime_ns),
            uri=path.as_uri(),
        )

    def upload(self, source: Path, *, name: str, sha256: str, size: int) -> ObjectInfo:
        existing = self.stat(name)
        if existing is not None:
            _validate_info(existing, name=name, sha256=sha256, size=size)
            return existing
        destination = self._path(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=".upload-",
            suffix=".upload",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            digest, copied_size = sha256_file(temporary)
            if digest != sha256 or copied_size != size:
                raise ValueError("Staged local object changed while it was copied")
            try:
                # Windows rename is atomic within a filesystem and refuses an
                # existing destination, preserving create-only semantics.
                os.rename(temporary, destination)
            except FileExistsError:
                existing = self.stat(name)
                assert existing is not None
                _validate_info(existing, name=name, sha256=sha256, size=size)
            info = self.stat(name)
            assert info is not None
            _validate_info(info, name=name, sha256=sha256, size=size)
            return info
        finally:
            if temporary.exists():
                temporary.unlink()

    def download(self, name: str, destination: Path) -> ObjectInfo:
        info = self.stat(name)
        if info is None:
            raise FileNotFoundError(name)
        source = self._path(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return info

    def list(self, prefix: str) -> tuple[ObjectInfo, ...]:
        base = self._path(prefix)
        if base.is_file():
            info = self.stat(prefix)
            return () if info is None else (info,)
        if not base.exists():
            return ()
        values: list[ObjectInfo] = []
        for path in sorted(candidate for candidate in base.rglob("*") if candidate.is_file()):
            name = path.relative_to(self.root).as_posix()
            info = self.stat(name)
            assert info is not None
            values.append(info)
        return tuple(values)


class GcsObjectStore:
    """Google Cloud Storage adapter with create-only resumable uploads."""

    backend = "gcs"

    def __init__(self, config: DatasetStoreConfig, *, client: object | None = None) -> None:
        if config.backend != "gcs" or config.bucket is None:
            raise ValueError("GcsObjectStore requires a GCS configuration")
        try:
            from google.cloud import storage  # type: ignore[import-untyped]
        except ImportError as error:
            raise RuntimeError(
                "Install the GCS extra: pip install -e '.[gcs]'"
            ) from error
        self.config = config
        self.client: Any = client or storage.Client(project=config.project)
        self.bucket = self.client.bucket(config.bucket)

    def _name(self, name: str) -> str:
        return f"{self.config.prefix}/{name}" if self.config.prefix else name

    def _info(self, blob: object, logical_name: str) -> ObjectInfo:
        metadata = getattr(blob, "metadata", None) or {}
        return ObjectInfo(
            name=logical_name,
            size=int(getattr(blob, "size", 0)),
            sha256=metadata.get("anthropocene-sha256"),
            generation=str(getattr(blob, "generation", "")) or None,
            uri=f"gs://{self.config.bucket}/{self._name(logical_name)}",
        )

    def stat(self, name: str) -> ObjectInfo | None:
        from google.api_core.exceptions import NotFound

        blob = self.bucket.blob(self._name(name))
        try:
            blob.reload(timeout=self.config.request_timeout_seconds)
        except NotFound:
            return None
        return self._info(blob, name)

    def upload(self, source: Path, *, name: str, sha256: str, size: int) -> ObjectInfo:
        from google.api_core.exceptions import PreconditionFailed
        from google.cloud.storage.retry import (  # type: ignore[import-untyped]
            DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
        )

        existing = self.stat(name)
        if existing is not None:
            _validate_info(existing, name=name, sha256=sha256, size=size)
            return existing
        blob = self.bucket.blob(
            self._name(name),
            chunk_size=self.config.resumable_chunk_mib * 1024**2,
        )
        blob.metadata = {
            "anthropocene-sha256": sha256,
            "anthropocene-immutable": "true",
        }
        try:
            blob.upload_from_filename(
                str(source),
                content_type="application/x-tar" if name.endswith(".tar") else "application/json",
                if_generation_match=0,
                checksum="crc32c",
                retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
                timeout=self.config.request_timeout_seconds,
            )
        except PreconditionFailed:
            existing = self.stat(name)
            if existing is None:
                raise
            _validate_info(existing, name=name, sha256=sha256, size=size)
            return existing
        blob.reload(timeout=self.config.request_timeout_seconds)
        info = self._info(blob, name)
        _validate_info(info, name=name, sha256=sha256, size=size)
        return info

    def download(self, name: str, destination: Path) -> ObjectInfo:
        from google.cloud.storage.retry import DEFAULT_RETRY  # type: ignore[import-untyped]

        blob = self.bucket.blob(self._name(name))
        blob.reload(timeout=self.config.request_timeout_seconds)
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(
            str(destination),
            checksum="auto",
            retry=DEFAULT_RETRY,
            timeout=self.config.request_timeout_seconds,
        )
        return self._info(blob, name)

    def list(self, prefix: str) -> tuple[ObjectInfo, ...]:
        remote_prefix = self._name(prefix)
        values: list[ObjectInfo] = []
        for blob in self.client.list_blobs(self.bucket, prefix=remote_prefix):
            logical_name = blob.name
            if self.config.prefix:
                logical_name = logical_name[len(self.config.prefix) + 1 :]
            values.append(self._info(blob, logical_name))
        return tuple(values)


class UploadJournal:
    """Crash-safe local record of verified immutable uploads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._payload: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"journal_schema_version": "dataset-upload-journal-v1", "objects": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Upload journal must be a JSON object")
        if payload.get("journal_schema_version") != "dataset-upload-journal-v1":
            raise ValueError("Unsupported upload journal version")
        return cast(dict[str, Any], payload)

    def verified(self, name: str, sha256: str, size: int) -> bool:
        with self._lock:
            entry = self._payload["objects"].get(name)
            return bool(
                isinstance(entry, dict)
                and entry.get("status") == "verified"
                and entry.get("sha256") == sha256
                and entry.get("size") == size
            )

    def update(self, name: str, **values: object) -> None:
        with self._lock:
            objects = self._payload["objects"]
            assert isinstance(objects, dict)
            entry = dict(objects.get(name, {}))
            entry.update(values)
            objects[name] = entry
            _write_json_atomic(self.path, self._payload)


class BoundedUploader:
    """Bounded concurrent uploader that can run while new shards are produced."""

    def __init__(
        self,
        store: ImmutableObjectStore,
        *,
        journal: UploadJournal,
        workers: int,
        max_pending: int,
        attempts: int,
        initial_backoff_seconds: float,
    ) -> None:
        self.store = store
        self.journal = journal
        self.attempts = attempts
        self.initial_backoff_seconds = initial_backoff_seconds
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gcs-upload")
        self.pending_limit = threading.Semaphore(max_pending)
        self.futures: list[Future[tuple[ObjectInfo, bool]]] = []

    def submit(
        self,
        source: Path,
        *,
        name: str,
        sha256: str,
        size: int,
    ) -> None:
        self.pending_limit.acquire()
        future = self.executor.submit(self._upload, source, name, sha256, size)
        future.add_done_callback(lambda _future: self.pending_limit.release())
        self.futures.append(future)

    def _upload(
        self, source: Path, name: str, sha256: str, size: int
    ) -> tuple[ObjectInfo, bool]:
        if self.journal.verified(name, sha256, size):
            existing = self.store.stat(name)
            if existing is not None:
                _validate_info(existing, name=name, sha256=sha256, size=size)
                return existing, True
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            self.journal.update(
                name,
                status="uploading",
                sha256=sha256,
                size=size,
                attempts=attempt,
            )
            try:
                existing = self.store.stat(name)
                deduplicated = existing is not None
                info = self.store.upload(source, name=name, sha256=sha256, size=size)
                _validate_info(info, name=name, sha256=sha256, size=size)
                self.journal.update(
                    name,
                    status="verified",
                    sha256=sha256,
                    size=size,
                    generation=info.generation,
                    uri=info.uri,
                    attempts=attempt,
                    error=None,
                )
                return info, deduplicated
            except ObjectCollisionError:
                raise
            except Exception as error:  # provider/network errors are retried here
                last_error = error
                self.journal.update(name, status="retrying", error=str(error), attempts=attempt)
                if attempt < self.attempts:
                    time.sleep(self.initial_backoff_seconds * (2 ** (attempt - 1)))
        assert last_error is not None
        self.journal.update(name, status="failed", error=str(last_error))
        raise last_error

    def wait(self) -> dict[str, object]:
        uploaded = 0
        deduplicated = 0
        errors: list[str] = []
        for future in as_completed(self.futures):
            try:
                _, was_deduplicated = future.result()
                if was_deduplicated:
                    deduplicated += 1
                else:
                    uploaded += 1
            except Exception as error:
                errors.append(str(error))
        self.executor.shutdown(wait=True)
        if errors:
            raise RuntimeError("Dataset upload failed: " + " | ".join(errors))
        return {"uploaded": uploaded, "deduplicated": deduplicated}


class DatasetPublisher:
    def __init__(self, config: DatasetStoreConfig, store: ImmutableObjectStore) -> None:
        self.config = config
        self.store = store

    def publish(self, manifest_path: str | Path) -> dict[str, object]:
        path = Path(manifest_path)
        manifest = DatasetRunManifest.load(path)
        staged_shards: list[tuple[Path, str, str, int]] = []
        for shard in manifest.shards:
            source = path.parent / "shards" / shard.name
            digest, size = sha256_file(source)
            if digest != shard.sha256 or size != shard.size_bytes:
                raise ValueError(f"Staged shard {shard.name} failed SHA-256/size validation")
            staged_shards.append((source, shard.object_name, shard.sha256, shard.size_bytes))

        journal = UploadJournal(path.parent / "upload-state.json")
        uploader = BoundedUploader(
            self.store,
            journal=journal,
            workers=self.config.upload_workers,
            max_pending=self.config.max_pending_uploads,
            attempts=self.config.upload_attempts,
            initial_backoff_seconds=self.config.initial_backoff_seconds,
        )
        for source, object_name, sha256, size in staged_shards:
            uploader.submit(
                source,
                name=object_name,
                sha256=sha256,
                size=size,
            )
        upload_summary = uploader.wait()

        shards = tuple(
            shard.model_copy(
                update={
                    "gcs_uri": (
                        f"gs://{self.config.bucket}/{self._remote_name(shard.object_name)}"
                        if self.config.backend == "gcs"
                        else None
                    )
                }
            )
            for shard in manifest.shards
        )
        completed = manifest.model_copy(
            update={
                "shards": shards,
                "publication_status": "complete",
                "storage_backend": self.config.backend,
                "storage_root": (
                    f"gs://{self.config.bucket}/{self.config.prefix}".rstrip("/")
                    if self.config.backend == "gcs"
                    else Path(self.config.local_root).resolve().as_uri()
                ),
            }
        )
        completed.save_atomic(path)
        manifest_name = (
            f"datasets/{completed.dataset_id}/{completed.run_id}/manifest.json"
        )
        digest, size = sha256_file(path)
        manifest_info = self.store.upload(
            path,
            name=manifest_name,
            sha256=digest,
            size=size,
        )
        return {
            "status": "PASS",
            "dataset_id": completed.dataset_id,
            "run_id": completed.run_id,
            "manifest_uri": manifest_info.uri,
            "manifest_sha256": digest,
            "shards": len(shards),
            "bytes": sum(shard.size_bytes for shard in shards),
            **upload_summary,
        }

    def _remote_name(self, logical_name: str) -> str:
        return f"{self.config.prefix}/{logical_name}" if self.config.prefix else logical_name


def build_object_store(config: DatasetStoreConfig) -> ImmutableObjectStore:
    if config.backend == "local":
        return LocalObjectStore(config.local_root)
    return GcsObjectStore(config)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
