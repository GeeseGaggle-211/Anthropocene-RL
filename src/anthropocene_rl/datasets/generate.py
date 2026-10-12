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
from collections import Counter, defaultdict
from collections.abc import Callable
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
    fit_train_and_report,
)
from anthropocene_rl.datasets.pilot_plan import (
    PILOT_ACCEPTED_FAMILY_COUNTS,
    PILOT_ACCEPTED_PARTITION_COUNTS,
)
from anthropocene_rl.observations.schema import (
    EncodedObservation,
    broad_domain_scope_violations,
    cultivation_only_scope_violations,
)


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
    paths.evidence_directory.mkdir(parents=True, exist_ok=True)

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
        _validate_native_result(first_result, config)
        _save_json(
            paths.evidence_directory / "native-candidate-summary.json",
            _native_candidate_summary(first_result, config),
        )
        # This field means regeneration was actually checked and matched.  A
        # disabled check must not be reported as successful verification.
        deterministic = False
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
                raise ValueError("Native regeneration changed for the same declared seed/config")

        manifest, samples = _ingest_native_result(
            first_result,
            native_directory=first_directory,
            blob_root=paths.blob_root,
            config=config,
        )

    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(paths.manifest)
    coverage_json = paths.evidence_directory / "coverage.json"
    coverage_markdown = paths.evidence_directory / "coverage.md"
    save_coverage_report(
        manifest,
        json_path=coverage_json,
        markdown_path=coverage_markdown,
    )
    audit = audit_snapshot_bank(
        manifest,
        blob_root=paths.blob_root,
        require_stage0_coverage=config.bank_kind == "stage0",
        require_cultivation_pilot_coverage=config.bank_kind == "cultivation-pilot",
        require_broad_domain_coverage=config.bank_kind == "elder-broad-domain",
    )
    _save_json(paths.evidence_directory / "audit.json", audit.model_dump(mode="json"))
    if not audit.passed:
        raise ValueError(
            "Generated snapshot bank failed audit: "
            + "; ".join(f"{issue.code}: {issue.message}" for issue in audit.errors)
        )

    rng_provenance = (
        f"{config.native_generator_version}/generation-seed:{config.generation_seed};"
        "bounded native calibration directives"
    )
    if config.bank_kind == "stage0":
        fit_provisional_and_report(
            samples,
            manifest=manifest,
            capability_experience_thresholds=config.capability_experience_thresholds,
            local_movement_radius=config.local_movement_radius,
            normalizer_path=paths.evidence_directory / "provisional-train-normalization.json",
            diagnostics_path=paths.evidence_directory
            / "provisional-normalization-diagnostics.json",
            rng_provenance=rng_provenance,
        )
        normalization_status = "PROVISIONAL_STAGE0_PIPELINE_TEST_NOT_FINAL_CAMPAIGN_NORMALIZER"
    else:
        fit_train_and_report(
            samples,
            manifest=manifest,
            capability_experience_thresholds=config.capability_experience_thresholds,
            local_movement_radius=config.local_movement_radius,
            normalizer_path=paths.evidence_directory / "final-train-normalization.json",
            diagnostics_path=paths.evidence_directory / "final-normalization-diagnostics.json",
            rng_provenance=rng_provenance,
            partition_suffix="final",
            artifact_status="FINAL_PILOT_TRAIN_ONLY_NORMALIZER",
            normalization_version=config.normalization_version,
        )
        normalization_status = f"FINAL_PILOT_TRAIN_ONLY_NORMALIZER:{config.normalization_version}"

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
            "normalization_status": normalization_status,
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
    require_cultivation_pilot_coverage: bool = False,
    require_broad_domain_coverage: bool = False,
    verify_blobs: bool = True,
) -> SnapshotBankAuditReport:
    manifest = SnapshotBankManifest.load(manifest_path)
    resolved_blob_root = Path(blob_root or manifest.blob_storage)
    report = audit_snapshot_bank(
        manifest,
        blob_root=resolved_blob_root,
        require_stage0_coverage=require_stage0_coverage,
        require_cultivation_pilot_coverage=require_cultivation_pilot_coverage,
        require_broad_domain_coverage=require_broad_domain_coverage,
        verify_blobs=verify_blobs,
    )
    if report_path is not None:
        _save_json(report_path, report.model_dump(mode="json"))
    if coverage_json_path is not None or coverage_markdown_path is not None:
        json_path = coverage_json_path or Path(manifest_path).with_suffix(".coverage.json")
        markdown_path = coverage_markdown_path or Path(manifest_path).with_suffix(".coverage.md")
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
    dirty = _run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]).strip()
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


def _validate_native_result(result: dict[str, Any], config: SnapshotBankGenerationConfig) -> None:
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
        key: (result.get(key), value) for key, value in expected.items() if result.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Native generator identity mismatch: {mismatches}")
    if not isinstance(result.get("candidates"), list) or not isinstance(
        result.get("rejected_candidates"), list
    ):
        raise ValueError("Native result candidate collections are malformed")


def _native_candidate_summary(
    result: dict[str, Any], config: SnapshotBankGenerationConfig
) -> dict[str, Any]:
    """Persist compact native feasibility evidence before promotion selection."""

    candidates = result["candidates"]
    rejected = result["rejected_candidates"]
    accepted_families: Counter[str] = Counter()
    accepted_partitions: Counter[str] = Counter()
    for candidate in candidates:
        record = candidate.get("record", {})
        accepted_families[str(record.get("scenario_family", "missing"))] += 1
        accepted_partitions[str(record.get("partition", "missing"))] += 1
    rejection_reasons = Counter(
        str(candidate.get("reason_code", "missing")) for candidate in rejected
    )
    rejected_families = Counter(
        str(candidate.get("scenario_family", "missing")) for candidate in rejected
    )
    return {
        "status": str(result.get("status", "missing")),
        "bank_id": config.bank_id,
        "generator_config_sha256": config.canonical_sha256(),
        "native_simulator_sha": result.get("native_simulator_sha"),
        "native_build_hash": result.get("native_build_hash"),
        "candidate_count": len(candidates),
        "native_rejection_count": len(rejected),
        "candidate_family_counts": dict(sorted(accepted_families.items())),
        "candidate_partition_counts": dict(sorted(accepted_partitions.items())),
        "native_rejection_family_counts": dict(sorted(rejected_families.items())),
        "native_rejection_reason_counts": dict(sorted(rejection_reasons.items())),
    }


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
    raw_candidates = result["candidates"]
    selection_rejections: tuple[RejectedCandidate, ...] = ()
    if config.bank_kind == "cultivation-pilot":
        raw_candidates, selection_rejections = _select_pilot_candidates(
            raw_candidates,
            config=config,
        )
    elif config.bank_kind == "elder-broad-domain":
        raw_candidates, selection_rejections = _select_broad_domain_candidates(
            raw_candidates,
            config=config,
        )
    for raw_candidate in raw_candidates:
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
        record = SnapshotRecord.model_validate({**raw_record, "blob_path": reference.relative_path})
        records.append(record)
        for raw_sample in raw_samples:
            if not isinstance(raw_sample, dict) or not isinstance(raw_sample.get("source"), str):
                raise ValueError("Native calibration sample is malformed")
            tensor = raw_sample.get("tensor")
            if not isinstance(tensor, dict):
                raise ValueError("Native calibration tensor is malformed")
            observation = EncodedObservation.from_native(tensor)
            violations = cultivation_only_scope_violations(observation)
            if config.bank_kind == "cultivation-pilot" and violations:
                raise ValueError(
                    f"Selected cultivation candidate {record.snapshot_id} contains "
                    f"an excluded calibration domain: {', '.join(violations)}"
                )
            broad_violations = broad_domain_scope_violations(observation)
            if config.bank_kind == "elder-broad-domain" and broad_violations:
                raise ValueError(
                    f"Selected broad-domain candidate {record.snapshot_id} contains "
                    f"an excluded calibration domain: {', '.join(broad_violations)}"
                )
            samples.append(
                CalibrationObservation(
                    snapshot_id=record.snapshot_id,
                    partition=record.partition,
                    source=raw_sample["source"],
                    observation=observation,
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

    rejected = (
        tuple(
            RejectedCandidate.model_validate(candidate)
            for candidate in result["rejected_candidates"]
        )
        + selection_rejections
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


def _select_pilot_candidates(
    candidates: list[Any],
    *,
    config: SnapshotBankGenerationConfig,
) -> tuple[list[dict[str, Any]], tuple[RejectedCandidate, ...]]:
    """Promote exact quotas without splitting native causal groups.

    The campaign intentionally requests surplus trajectories in families whose
    achieved physical strata can be rejected by the native simulator.  This
    selector operates only after native validation and keeps a group iff every
    requested timestamp and counterfactual sibling survived.  F15 additionally
    preserves five complete pairs of each declared causal contrast.
    """

    return _select_grouped_candidates(
        candidates,
        config=config,
        accepted_family_counts=PILOT_ACCEPTED_FAMILY_COUNTS,
        accepted_partition_counts=(
            config.pilot_accepted_partition_counts or PILOT_ACCEPTED_PARTITION_COUNTS
        ),
        calibration_scope_violations=_candidate_calibration_scope_violations,
        scope_reason_code="cultivation-scope-escape-during-calibration",
        selection_metadata_key="pilot_selection_policy",
    )


def _select_broad_domain_candidates(
    candidates: list[Any],
    *,
    config: SnapshotBankGenerationConfig,
) -> tuple[list[dict[str, Any]], tuple[RejectedCandidate, ...]]:
    accepted = config.broad_domain_accepted_partition_counts
    if accepted is None:
        raise ValueError("Broad-domain selection requires accepted partition counts")
    accepted_families = {
        family: sum(partitions.values())
        for family, partitions in accepted.items()
    }
    return _select_grouped_candidates(
        candidates,
        config=config,
        accepted_family_counts=accepted_families,
        accepted_partition_counts=accepted,
        calibration_scope_violations=_candidate_broad_domain_scope_violations,
        scope_reason_code="broad-domain-scope-escape-during-calibration",
        selection_metadata_key="broad_domain_selection_policy",
    )


def _select_grouped_candidates(
    candidates: list[Any],
    *,
    config: SnapshotBankGenerationConfig,
    accepted_family_counts: dict[str, int],
    accepted_partition_counts: dict[str, dict[str, int]],
    calibration_scope_violations: Callable[[dict[str, Any]], tuple[str, ...]],
    scope_reason_code: str,
    selection_metadata_key: str,
) -> tuple[list[dict[str, Any]], tuple[RejectedCandidate, ...]]:
    expected_counts: dict[tuple[str, str], int] = defaultdict(int)
    for trajectory in config.trajectories:
        group = trajectory.counterfactual_group_id or trajectory.lineage_id
        expected_counts[(trajectory.scenario_family, group)] += len(trajectory.snapshot_steps)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("record"), dict):
            raise ValueError("Native candidate must contain a record object")
        record = candidate["record"]
        family = record.get("scenario_family")
        group = record.get("counterfactual_group_id") or record.get("lineage_id")
        if not isinstance(family, str) or not isinstance(group, str):
            raise ValueError("Native candidate lacks a causal group identity")
        key = (family, group)
        if key not in expected_counts:
            raise ValueError(f"Native candidate has an undeclared group: {key}")
        grouped[key].append(candidate)

    duplicate_identity_groups: set[tuple[str, str]] = set()
    scope_escape_groups: dict[tuple[str, str], tuple[str, ...]] = {}
    complete: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, group_candidates in grouped.items():
        if len(group_candidates) != expected_counts[key]:
            continue
        violations = sorted(
            {
                violation
                for candidate in group_candidates
                for violation in calibration_scope_violations(candidate)
            }
        )
        if violations:
            scope_escape_groups[key] = tuple(violations)
            continue
        fingerprints = {
            str(candidate["record"]["native_fingerprint"]).lower() for candidate in group_candidates
        }
        content_hashes = {
            str(candidate["record"]["content_sha256"]).lower() for candidate in group_candidates
        }
        if len(fingerprints) != len(group_candidates) or len(content_hashes) != len(
            group_candidates
        ):
            duplicate_identity_groups.add(key)
            continue
        complete[key] = group_candidates

    selected_keys: set[tuple[str, str]] = set()
    selected_fingerprints: set[str] = set()
    selected_content_hashes: set[str] = set()
    identity_conflict_groups: set[tuple[str, str]] = set()
    eligible: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, group_candidates in sorted(complete.items()):
        fingerprints = {
            str(candidate["record"]["native_fingerprint"]).lower() for candidate in group_candidates
        }
        content_hashes = {
            str(candidate["record"]["content_sha256"]).lower() for candidate in group_candidates
        }
        if fingerprints & selected_fingerprints or content_hashes & selected_content_hashes:
            identity_conflict_groups.add(key)
            continue
        eligible[key] = group_candidates
        selected_fingerprints.update(fingerprints)
        selected_content_hashes.update(content_hashes)

    # Reuse these sets below for identities in promoted groups.  Every eligible
    # group is already globally unique, so partition selection cannot create a
    # duplicate by construction.
    selected_fingerprints.clear()
    selected_content_hashes.clear()
    shortages: list[str] = []
    for family, quotas in accepted_partition_counts.items():
        if family == "F15":
            continue
        for partition, required in quotas.items():
            available = sum(
                len(group_candidates)
                for key, group_candidates in eligible.items()
                if key[0] == family and _group_partition(group_candidates) == partition
            )
            if available < required:
                complete_count = sum(
                    len(group_candidates)
                    for key, group_candidates in complete.items()
                    if key[0] == family
                    and _group_partition(group_candidates) == partition
                )
                scope_escape_count = sum(
                    len(grouped[key])
                    for key in scope_escape_groups
                    if key[0] == family
                    and _group_partition(grouped[key]) == partition
                )
                duplicate_identity_count = sum(
                    len(grouped[key])
                    for key in duplicate_identity_groups | identity_conflict_groups
                    if key[0] == family
                    and _group_partition(grouped[key]) == partition
                )
                incomplete_count = sum(
                    len(group_candidates)
                    for key, group_candidates in grouped.items()
                    if key[0] == family
                    and key not in complete
                    and key not in scope_escape_groups
                    and key not in duplicate_identity_groups
                    and _group_partition(group_candidates) == partition
                )
                shortages.append(
                    f"{family}/{partition}: eligible {available}, required {required}; "
                    f"complete {complete_count}, scope-escape {scope_escape_count}, "
                    f"identity-conflict {duplicate_identity_count}, "
                    f"incomplete-records {incomplete_count}"
                )
    if shortages:
        raise ValueError("Insufficient complete scope-safe groups: " + "; ".join(shortages))
    for family, target_count in accepted_family_counts.items():
        family_groups = sorted(
            (
                (key, group_candidates)
                for key, group_candidates in eligible.items()
                if key[0] == family
            ),
            key=lambda item: item[0][1],
        )
        quotas = accepted_partition_counts[family]
        if sum(quotas.values()) != target_count:
            raise ValueError(f"Accepted partition quotas do not sum for {family}")
        if family == "F15":
            chosen = _select_f15_partitioned_groups(family_groups, quotas)
        else:
            chosen = []
            for partition in ("train", "development", "heldout"):
                candidates_for_partition = [
                    item for item in family_groups if _group_partition(item[1]) == partition
                ]
                count = 0
                for item in candidates_for_partition:
                    size = len(item[1])
                    if count + size > quotas[partition]:
                        continue
                    chosen.append(item)
                    count += size
                    if count == quotas[partition]:
                        break
                if count != quotas[partition]:
                    raise ValueError(
                        f"Insufficient complete {family}/{partition} groups: "
                        f"selected {count}, required {quotas[partition]}"
                    )
        for key, group_candidates in chosen:
            selected_keys.add(key)
            selected_fingerprints.update(
                str(candidate["record"]["native_fingerprint"]).lower()
                for candidate in group_candidates
            )
            selected_content_hashes.update(
                str(candidate["record"]["content_sha256"]).lower() for candidate in group_candidates
            )

    selected: list[dict[str, Any]] = []
    rejected: list[RejectedCandidate] = []
    for key, group_candidates in sorted(grouped.items()):
        if key in selected_keys:
            selected.extend(group_candidates)
            continue
        complete_group = key in complete
        duplicate_identity = key in duplicate_identity_groups or key in identity_conflict_groups
        if key in scope_escape_groups:
            reason_code = scope_reason_code
            reason = (
                "A legal calibration rollout activated an excluded campaign domain: "
                + ", ".join(scope_escape_groups[key])
                + "."
            )
        elif duplicate_identity:
            reason_code = "duplicate-native-identity-group"
            reason = "A causal group repeated a native fingerprint or content identity."
        elif complete_group:
            reason_code = "coverage-quota-not-selected"
            reason = "Valid native group exceeded the frozen accepted-family quota."
        else:
            reason_code = "incomplete-causal-group"
            reason = "A requested timestamp or counterfactual sibling failed native validation."
        for candidate in group_candidates:
            record = candidate["record"]
            metadata = dict(record.get("selection_metadata", {}))
            metadata[selection_metadata_key] = (
                "complete-groups-family-partition-exact-quota-v3"
            )
            rejected.append(
                RejectedCandidate(
                    candidate_id=str(record["snapshot_id"]),
                    scenario_family=str(record["scenario_family"]),
                    scenario_version=str(record["scenario_version"]),
                    world_seed=int(record["world_seed"]),
                    trajectory_family=str(record["trajectory_family"]),
                    lineage_id=str(record["lineage_id"]),
                    counterfactual_group_id=record.get("counterfactual_group_id"),
                    requested_partition=record["partition"],
                    reason_code=reason_code,
                    reason=reason,
                    selection_metadata=metadata,
                )
            )
    return selected, tuple(rejected)


def _candidate_calibration_scope_violations(candidate: dict[str, Any]) -> tuple[str, ...]:
    raw_samples = candidate.get("calibration_observations")
    if not isinstance(raw_samples, list):
        raise ValueError("Native candidate calibration observations are malformed")
    violations: set[str] = set()
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, dict) or not isinstance(raw_sample.get("tensor"), dict):
            raise ValueError("Native candidate calibration observation is malformed")
        observation = EncodedObservation.from_native(raw_sample["tensor"])
        violations.update(cultivation_only_scope_violations(observation))
    return tuple(sorted(violations))


def _candidate_broad_domain_scope_violations(
    candidate: dict[str, Any],
) -> tuple[str, ...]:
    raw_samples = candidate.get("calibration_observations")
    if not isinstance(raw_samples, list):
        raise ValueError("Native candidate calibration observations are malformed")
    violations: set[str] = set()
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, dict) or not isinstance(raw_sample.get("tensor"), dict):
            raise ValueError("Native candidate calibration observation is malformed")
        observation = EncodedObservation.from_native(raw_sample["tensor"])
        violations.update(broad_domain_scope_violations(observation))
    return tuple(sorted(violations))


def _group_partition(group_candidates: list[dict[str, Any]]) -> str:
    partitions = {str(candidate["record"]["partition"]) for candidate in group_candidates}
    if len(partitions) != 1:
        raise ValueError("A causal group crossed snapshot-bank partitions")
    return next(iter(partitions))


def _select_f15_partitioned_groups(
    family_groups: list[tuple[tuple[str, str], list[dict[str, Any]]]],
    record_quotas: dict[str, int],
) -> list[tuple[tuple[str, str], list[dict[str, Any]]]]:
    """Balance F15 contrasts while meeting split quotas exactly."""

    group_quotas = {partition: count // 2 for partition, count in record_quotas.items()}
    if any(count % 2 for count in record_quotas.values()):
        raise ValueError("F15 record quotas must preserve two-snapshot counterfactual pairs")
    by_contrast: dict[
        tuple[str, ...],
        dict[str, list[tuple[tuple[str, str], list[dict[str, Any]]]]],
    ] = defaultdict(lambda: defaultdict(list))
    for item in family_groups:
        contrast = tuple(
            sorted(
                str(candidate["record"]["selection_metadata"].get("causal_variant", "none"))
                for candidate in item[1]
            )
        )
        by_contrast[contrast][_group_partition(item[1])].append(item)
    if len(by_contrast) != 5:
        availability = {
            " + ".join(contrast): {
                partition: len(groups)
                for partition, groups in sorted(partitions.items())
            }
            for contrast, partitions in sorted(by_contrast.items())
        }
        raise ValueError(
            "F15 must expose exactly five causal contrasts after native identity "
            f"de-duplication; observed={availability}"
        )
    contrasts = sorted(by_contrast)
    partitions = ("train", "development", "heldout")
    selected_pair_count = sum(group_quotas.values())
    if selected_pair_count % len(contrasts):
        raise ValueError("F15 pair quota must divide evenly across its five contrasts")
    pairs_per_contrast = selected_pair_count // len(contrasts)
    allocation: dict[tuple[str, ...], dict[str, int]] = {}

    def search(index: int, remaining: dict[str, int]) -> bool:
        if index == len(contrasts):
            return all(value == 0 for value in remaining.values())
        contrast = contrasts[index]
        available = {
            partition: len(by_contrast[contrast].get(partition, [])) for partition in partitions
        }
        for train in range(pairs_per_contrast + 1):
            for development in range(pairs_per_contrast - train + 1):
                heldout = pairs_per_contrast - train - development
                proposal = {
                    "train": train,
                    "development": development,
                    "heldout": heldout,
                }
                if any(
                    proposal[partition] > available[partition]
                    or proposal[partition] > remaining[partition]
                    for partition in partitions
                ):
                    continue
                next_remaining = {
                    partition: remaining[partition] - proposal[partition]
                    for partition in partitions
                }
                allocation[contrast] = proposal
                if search(index + 1, next_remaining):
                    return True
        allocation.pop(contrast, None)
        return False

    if not search(0, dict(group_quotas)):
        availability = {
            " + ".join(contrast): {
                partition: len(by_contrast[contrast].get(partition, [])) for partition in partitions
            }
            for contrast in contrasts
        }
        raise ValueError(
            "Complete F15 groups cannot meet contrast and partition quotas; "
            f"requested_groups={group_quotas}; available_groups={availability}"
        )
    selected = []
    for contrast in contrasts:
        for partition in partitions:
            selected.extend(
                by_contrast[contrast].get(partition, [])[: allocation[contrast][partition]]
            )
    if len(selected) != selected_pair_count:
        raise AssertionError(
            f"F15 selection must preserve {selected_pair_count} complete pairs"
        )
    return selected


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
