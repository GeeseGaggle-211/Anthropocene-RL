"""Deterministic, immutable publication of training-run artifacts."""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from anthropocene_rl.datasets.blobs import sha256_file
from anthropocene_rl.datasets.models import StrictModel
from anthropocene_rl.datasets.object_store import DatasetStoreConfig, ImmutableObjectStore

RUN_ARTIFACT_SCHEMA_VERSION = "anthropocene-run-artifacts-v1"


class RunArtifactFile(StrictModel):
    path: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_bytes: int = Field(ge=0)


class RunArtifactBundle(StrictModel):
    object_name: str = Field(min_length=1)
    uri: str | None = None
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_bytes: int = Field(gt=0)
    generation: str | None = None


class RunArtifactManifest(StrictModel):
    run_artifact_schema_version: Literal["anthropocene-run-artifacts-v1"] = (
        RUN_ARTIFACT_SCHEMA_VERSION
    )
    campaign_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    captured_at: str
    source_branch: str
    starting_learner_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    execution_learner_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    native_simulator_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset_manifest_uri: str = Field(pattern=r"^gs://[^/]+/.+$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[RunArtifactFile, ...]
    bundle: RunArtifactBundle
    publication_status: Literal["staged", "complete"] = "staged"
    storage_backend: Literal["local", "gcs"] = "local"

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        paths = [item.path for item in self.files]
        if paths != sorted(paths):
            raise ValueError("Run artifact files must be sorted")
        if len(paths) != len(set(paths)):
            raise ValueError("Run artifact file paths must be unique")
        if self.publication_status == "complete" and self.bundle.uri is None:
            raise ValueError("A complete run artifact manifest requires a bundle URI")
        return self

    def save_atomic(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=".run-manifest.",
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
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def stage_run_artifacts(
    *,
    repository_root: str | Path,
    includes: tuple[str | Path, ...],
    staging_root: str | Path,
    campaign_id: str,
    run_id: str,
    captured_at: str,
    source_branch: str,
    starting_learner_sha: str,
    execution_learner_sha: str,
    native_simulator_sha: str,
    dataset_manifest_uri: str,
    dataset_manifest_sha256: str,
) -> Path:
    root = Path(repository_root).resolve()
    output = Path(staging_root) / "run-artifacts" / campaign_id / run_id
    output.mkdir(parents=True, exist_ok=True)
    paths = _collect_files(root, includes)
    bundle_path = output / "bundle.tar"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output,
        prefix=".bundle.",
        suffix=".tar",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    files: list[RunArtifactFile] = []
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for relative, source in paths:
                digest, size = sha256_file(source)
                files.append(
                    RunArtifactFile(
                        path=relative,
                        content_sha256=digest,
                        content_bytes=size,
                    )
                )
                info = archive.gettarinfo(str(source), arcname=relative)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mode = 0o644
                with source.open("rb") as handle:
                    archive.addfile(info, handle)
        os.replace(temporary, bundle_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    bundle_sha256, bundle_bytes = sha256_file(bundle_path)
    object_name = f"run-objects/sha256/{bundle_sha256[:2]}/{bundle_sha256}.tar"
    manifest = RunArtifactManifest(
        campaign_id=campaign_id,
        run_id=run_id,
        captured_at=captured_at,
        source_branch=source_branch,
        starting_learner_sha=starting_learner_sha,
        execution_learner_sha=execution_learner_sha,
        native_simulator_sha=native_simulator_sha,
        dataset_manifest_uri=dataset_manifest_uri,
        dataset_manifest_sha256=dataset_manifest_sha256,
        files=tuple(files),
        bundle=RunArtifactBundle(
            object_name=object_name,
            content_sha256=bundle_sha256,
            content_bytes=bundle_bytes,
        ),
    )
    manifest_path = output / "manifest.json"
    manifest.save_atomic(manifest_path)
    return manifest_path


def publish_run_artifacts(
    manifest_path: str | Path,
    *,
    config: DatasetStoreConfig,
    store: ImmutableObjectStore,
) -> dict[str, object]:
    path = Path(manifest_path)
    manifest = RunArtifactManifest.load(path)
    bundle_path = path.parent / "bundle.tar"
    digest, size = sha256_file(bundle_path)
    if digest != manifest.bundle.content_sha256 or size != manifest.bundle.content_bytes:
        raise ValueError("Staged run artifact bundle failed SHA-256/size validation")
    existing = store.stat(manifest.bundle.object_name)
    bundle_info = store.upload(
        bundle_path,
        name=manifest.bundle.object_name,
        sha256=digest,
        size=size,
    )
    completed = manifest.model_copy(
        update={
            "bundle": manifest.bundle.model_copy(
                update={
                    "uri": bundle_info.uri,
                    "generation": bundle_info.generation,
                }
            ),
            "publication_status": "complete",
            "storage_backend": config.backend,
        }
    )
    completed.save_atomic(path)
    manifest_sha256, manifest_bytes = sha256_file(path)
    manifest_name = f"runs/{completed.campaign_id}/{completed.run_id}/manifest.json"
    manifest_info = store.upload(
        path,
        name=manifest_name,
        sha256=manifest_sha256,
        size=manifest_bytes,
    )
    return {
        "status": "PASS",
        "campaign_id": completed.campaign_id,
        "run_id": completed.run_id,
        "files": len(completed.files),
        "payload_bytes": sum(item.content_bytes for item in completed.files),
        "bundle_bytes": completed.bundle.content_bytes,
        "bundle_sha256": completed.bundle.content_sha256,
        "bundle_uri": completed.bundle.uri,
        "bundle_generation": completed.bundle.generation,
        "manifest_bytes": manifest_bytes,
        "manifest_sha256": manifest_sha256,
        "manifest_uri": manifest_info.uri,
        "manifest_generation": manifest_info.generation,
        "deduplicated": existing is not None,
    }


def _collect_files(
    root: Path,
    includes: tuple[str | Path, ...],
) -> tuple[tuple[str, Path], ...]:
    collected: dict[str, Path] = {}
    for value in includes:
        candidate = (root / value).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Run artifact include escapes repository root: {value}") from error
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        sources = (
            [candidate]
            if candidate.is_file()
            else sorted(path for path in candidate.rglob("*") if path.is_file())
        )
        for source in sources:
            relative = source.relative_to(root).as_posix()
            collected[relative] = source
    if not collected:
        raise ValueError("At least one run artifact file is required")
    return tuple(sorted(collected.items()))
