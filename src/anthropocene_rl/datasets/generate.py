"""Orchestrate native SaveCodec snapshot-bank generation.

Python owns provenance, partitioning, content-addressed storage, and audit
artifacts. World construction and every physical/economic transition remain in
the pinned native simulator executable.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropocene_rl.contracts import (
    ACTION_SCHEMA_VERSION,
    CRITIC_SIDECAR_SCHEMA_VERSION,
    ENVIRONMENT_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    SIMULATOR_COMMIT,
    SIMULATOR_REPOSITORY,
    TENSOR_FEATURE_COUNT,
)
from anthropocene_rl.datasets.audit import SnapshotBankAuditReport, audit_snapshot_bank
from anthropocene_rl.datasets.blobs import ContentAddressedBlobStore
from anthropocene_rl.datasets.coverage import save_coverage_report
from anthropocene_rl.datasets.generation_config import SnapshotBankGenerationConfig
from anthropocene_rl.datasets.models import (
    RejectedCandidate,
    SnapshotBankManifest,
    SnapshotRecord,
)
from anthropocene_rl.datasets.normalization_audit import (
    CalibrationObservation,
    fit_provisional_and_report,
)
from anthropocene_rl.observations.schema import EncodedObservation


@dataclass(frozen=True)
class GenerationPaths:
    manifest: Path
    blob_root: Path
    evidence_directory: Path


@dataclass(frozen=True)
class GenerationSummary:
    manifest_sha256: str
    manifest_path: str
    blob_root: str
    snapshot_count: int
    partition_counts: dict[str, int]
    rejection_counts: dict[str, int]
    deterministic_regeneration: bool
    native_result_sha256: str
    audit_status: str


def generate_snapshot_bank(
    *,
    config_path: str | Path,
    paths: GenerationPaths,
    simulator_root: str | Path,
    dotnet: str = "dotnet",
    project_root: str | Path = ".",
) -> GenerationSummary:
    root = Path(project_root).resolve()
    simulator = Path(simulator_root).resolve()
    config = SnapshotBankGenerationConfig.load(config_path)
    _validate_frozen_simulator(simulator)
    dll = _build_native_generator(root, simulator, dotnet)

    native_payload = config.native_payload()
    with tempfile.TemporaryDirectory(prefix="anthropocene-snapshot-bank-") as temporary:
        temporary_root = Path(temporary)
        resolved = temporary_root / "resolved-generation-plan.json"
        resolved.write_text(
            json.dumps(native_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        first_result, first_directory = _native_run(
            dotnet=dotnet,
            dll=dll,
            resolved_config=resolved,
            output=temporary_root / "first",
            project_root=root,
        )
        deterministic = True
        if config.verify_deterministic_regeneration:
            second_result, _ = _native_run(
                dotnet=dotnet,
                dll=dll,
                resolved_config=resolved,
                output=temporary_root / "second",
                project_root=root,
            )
            deterministic = _canonical_json(first_result) == _canonical_json(second_result)
            if not deterministic:
                raise ValueError(
                    "Native regeneration changed for the same declared seed/config"
                )

        _validate_native_result(first_result, config)
        manifest, samples = _ingest_native_result(
            first_result,
            native_directory=first_directory,
            blob_root=paths.blob_root,
            config=config,
        )

    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.evidence_directory.mkdir(parents=True, exist_ok=True)
    manifest.save(paths.manifest)
    coverage_json = paths.evidence_directory / "coverage.json"
    coverage_markdown = paths.evidence_directory / "coverage.md"
    save_coverage_report(
        manifest,
        json_path=coverage_json,
        markdown_path=coverage_markdown,
    )
    audit = audit_snapshot_bank(manifest, blob_root=paths.blob_root)
    _save_json(paths.evidence_directory / "audit.json", audit.model_dump(mode="json"))
    if not audit.passed:
        raise ValueError(
            "Generated snapshot bank failed audit: "
            + "; ".join(f"{issue.code}: {issue.message}" for issue in audit.errors)
        )

    fit_provisional_and_report(
        samples,
        manifest=manifest,
        capability_experience_thresholds=config.capability_experience_thresholds,
        local_movement_radius=config.local_movement_radius,
        normalizer_path=paths.evidence_directory
        / "provisional-train-normalization.json",
        diagnostics_path=paths.evidence_directory
        / "provisional-normalization-diagnostics.json",
        rng_provenance=(
            f"{config.native_generator_version}/generation-seed:{config.generation_seed};"
            "bounded native calibration directives"
        ),
    )

    native_sha = hashlib.sha256(_canonical_json(first_result)).hexdigest()
    summary = GenerationSummary(
        manifest_sha256=manifest.sha256(),
        manifest_path=paths.manifest.as_posix(),
        blob_root=paths.blob_root.as_posix(),
        snapshot_count=len(manifest.records),
        partition_counts=audit.partition_counts,
        rejection_counts=audit.rejection_counts,
        deterministic_regeneration=deterministic,
        native_result_sha256=native_sha,
        audit_status=audit.status,
    )
    _save_json(
        paths.evidence_directory / "generation-summary.json",
        {
            **summary.__dict__,
            "generator_config_sha256": config.canonical_sha256(),
            "normalization_status": (
                "PROVISIONAL_STAGE0_PIPELINE_TEST_NOT_FINAL_CAMPAIGN_NORMALIZER"
            ),
        },
    )
    return summary


def audit_manifest_file(
    *,
    manifest_path: str | Path,
    blob_root: str | Path | None = None,
    report_path: str | Path | None = None,
    coverage_json_path: str | Path | None = None,
    coverage_markdown_path: str | Path | None = None,
    require_stage0_coverage: bool = True,
    verify_blobs: bool = True,
) -> SnapshotBankAuditReport:
    manifest = SnapshotBankManifest.load(manifest_path)
    resolved_blob_root = Path(blob_root or manifest.blob_storage)
    report = audit_snapshot_bank(
        manifest,
        blob_root=resolved_blob_root,
        require_stage0_coverage=require_stage0_coverage,
        verify_blobs=verify_blobs,
    )
    if report_path is not None:
        _save_json(report_path, report.model_dump(mode="json"))
    if coverage_json_path is not None or coverage_markdown_path is not None:
        json_path = coverage_json_path or Path(manifest_path).with_suffix(".coverage.json")
        markdown_path = coverage_markdown_path or Path(manifest_path).with_suffix(
            ".coverage.md"
        )
        save_coverage_report(
            manifest,
            json_path=json_path,
            markdown_path=markdown_path,
        )
    return report


def _validate_frozen_simulator(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"Frozen simulator checkout is missing: {root}")
    head = _run(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
    if head != SIMULATOR_COMMIT:
        raise ValueError(f"Simulator pin mismatch: {head} != {SIMULATOR_COMMIT}")
    dirty = _run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]
    ).strip()
    if dirty:
        raise ValueError("Frozen simulator tracked files are modified")


def _build_native_generator(root: Path, simulator: Path, dotnet: str) -> Path:
    project = root / "integration/Anthropocene.SnapshotBank.csproj"
    _run(
        [
            dotnet,
            "build",
            str(project),
            "-c",
            "Release",
            "--nologo",
            "-v:q",
            f"-p:AnthropoceneSimRoot={simulator}",
        ],
        cwd=root,
    )
    dll = root / "integration/bin/Release/net8.0/Anthropocene.SnapshotBank.dll"
    if not dll.is_file():
        raise FileNotFoundError(f"Native snapshot generator did not produce {dll}")
    return dll


def _native_run(
    *,
    dotnet: str,
    dll: Path,
    resolved_config: Path,
    output: Path,
    project_root: Path,
) -> tuple[dict[str, Any], Path]:
    _run(
        [
            dotnet,
            str(dll),
            "generate",
            "--config",
            str(resolved_config),
            "--output",
            str(output),
        ],
        cwd=project_root,
    )
    result_path = output / "native-result.json"
    if not result_path.is_file():
        raise FileNotFoundError("Native generator did not emit native-result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Native generator result must be a JSON object")
    return result, output


def _validate_native_result(
    result: dict[str, Any], config: SnapshotBankGenerationConfig
) -> None:
    expected = {
        "status": "PASS",
        "native_simulator_sha": SIMULATOR_COMMIT,
        "generator_config_sha256": config.canonical_sha256(),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "critic_sidecar_schema_version": CRITIC_SIDECAR_SCHEMA_VERSION,
        "tensor_feature_count": TENSOR_FEATURE_COUNT,
    }
    mismatches = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Native generator identity mismatch: {mismatches}")
    if not isinstance(result.get("candidates"), list) or not isinstance(
        result.get("rejected_candidates"), list
    ):
        raise ValueError("Native result candidate collections are malformed")


def _ingest_native_result(
    result: dict[str, Any],
    *,
    native_directory: Path,
    blob_root: Path,
    config: SnapshotBankGenerationConfig,
) -> tuple[SnapshotBankManifest, list[CalibrationObservation]]:
    store = ContentAddressedBlobStore(blob_root)
    records: list[SnapshotRecord] = []
    samples: list[CalibrationObservation] = []
    for raw_candidate in result["candidates"]:
        if not isinstance(raw_candidate, dict):
            raise ValueError("Native candidate must be a JSON object")
        raw_record = raw_candidate.get("record")
        snapshot_file = raw_candidate.get("snapshot_file")
        raw_samples = raw_candidate.get("calibration_observations")
        if not isinstance(raw_record, dict) or not isinstance(snapshot_file, str):
            raise ValueError("Native candidate record/blob reference is malformed")
        if not isinstance(raw_samples, list):
            raise ValueError("Native candidate calibration observations are malformed")
        source = (native_directory / snapshot_file).resolve()
        try:
            source.relative_to(native_directory.resolve())
        except ValueError as error:
            raise ValueError("Native snapshot path escapes its generation directory") from error
        reference = store.put_file(source)
        if raw_record.get("content_sha256") != reference.sha256:
            raise ValueError("Native-declared content SHA-256 does not match SaveCodec bytes")
        if raw_record.get("content_bytes") != reference.size:
            raise ValueError("Native-declared content size does not match SaveCodec bytes")
        record = SnapshotRecord.model_validate(
            {**raw_record, "blob_path": reference.relative_path}
        )
        records.append(record)
        for raw_sample in raw_samples:
            if not isinstance(raw_sample, dict) or not isinstance(
                raw_sample.get("source"), str
            ):
                raise ValueError("Native calibration sample is malformed")
            tensor = raw_sample.get("tensor")
            if not isinstance(tensor, dict):
                raise ValueError("Native calibration tensor is malformed")
            samples.append(
                CalibrationObservation(
                    snapshot_id=record.snapshot_id,
                    partition=record.partition,
                    source=raw_sample["source"],
                    observation=EncodedObservation.from_native(tensor),
                    # Measured stratum tags describe the authoritative saved
                    # state. Do not silently carry them onto later rollout
                    # states whose physical strata may have changed.
                    slices=(
                        frozenset(record.actual_strata.tags)
                        if raw_sample["source"] == "reset"
                        else frozenset()
                    ),
                )
            )

    rejected = tuple(
        RejectedCandidate.model_validate(candidate)
        for candidate in result["rejected_candidates"]
    )
    _, partitioner = config.resolve_partitions()
    manifest = SnapshotBankManifest(
        bank_id=config.bank_id,
        bank_version=config.bank_version,
        generator_config_sha256=config.canonical_sha256(),
        native_simulator_sha=SIMULATOR_COMMIT,
        native_simulator_repository=SIMULATOR_REPOSITORY,
        blob_storage=blob_root.as_posix(),
        partition_contract=partitioner.contract,
        records=tuple(sorted(records, key=lambda record: record.snapshot_id)),
        rejected_candidates=rejected,
    )
    return manifest, samples


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    environment = os.environ.copy()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    return completed.stdout
