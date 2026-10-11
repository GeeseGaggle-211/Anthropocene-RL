from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

from anthropocene_rl.datasets.object_store import DatasetStoreConfig, LocalObjectStore
from anthropocene_rl.datasets.run_artifacts import (
    RunArtifactManifest,
    publish_run_artifacts,
    stage_run_artifacts,
)


def _stage(tmp_path: Path, staging: str) -> Path:
    root = tmp_path / "repository"
    (root / "evidence").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    (root / "evidence" / "summary.json").write_text('{"status":"NO_GO"}\n')
    (root / "checkpoints" / "seed.pt").write_bytes(b"checkpoint")
    return stage_run_artifacts(
        repository_root=root,
        includes=("evidence", "checkpoints/seed.pt"),
        staging_root=tmp_path / staging,
        campaign_id="cultivation-pilot-v1",
        run_id="run-001",
        captured_at="2026-10-11T00:00:00Z",
        source_branch="feature/test",
        starting_learner_sha="1" * 40,
        execution_learner_sha="2" * 40,
        native_simulator_sha="3" * 40,
        dataset_manifest_uri="gs://bucket/datasets/pilot/manifest.json",
        dataset_manifest_sha256="4" * 64,
    )


def test_run_artifact_tar_is_deterministic_and_publishes_manifest_last(tmp_path: Path) -> None:
    first_path = _stage(tmp_path, "first")
    second_path = _stage(tmp_path, "second")
    first = RunArtifactManifest.load(first_path)
    second = RunArtifactManifest.load(second_path)
    assert first.bundle.content_sha256 == second.bundle.content_sha256
    assert [item.path for item in first.files] == [
        "checkpoints/seed.pt",
        "evidence/summary.json",
    ]

    config = DatasetStoreConfig(
        backend="local",
        local_root=(tmp_path / "remote").as_posix(),
        staging_root=(tmp_path / "staging").as_posix(),
    )
    store = LocalObjectStore(config.local_root)
    result = publish_run_artifacts(first_path, config=config, store=store)
    assert result["status"] == "PASS"
    assert result["files"] == 2
    assert store.stat(first.bundle.object_name) is not None
    assert store.stat("runs/cultivation-pilot-v1/run-001/manifest.json") is not None

    completed = RunArtifactManifest.load(first_path)
    assert completed.publication_status == "complete"
    assert completed.bundle.uri is not None
    with tarfile.open(first_path.parent / "bundle.tar") as archive:
        assert archive.getnames() == ["checkpoints/seed.pt", "evidence/summary.json"]
    assert hashlib.sha256((first_path.parent / "bundle.tar").read_bytes()).hexdigest() == (
        completed.bundle.content_sha256
    )
