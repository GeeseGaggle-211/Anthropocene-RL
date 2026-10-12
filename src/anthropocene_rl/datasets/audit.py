"""Leakage, provenance, native-gate, and blob-integrity audits."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from pydantic import Field

from anthropocene_rl.contracts import (
    ACTION_SCHEMA_VERSION,
    CRITIC_SIDECAR_SCHEMA_VERSION,
    ENVIRONMENT_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    SIMULATOR_COMMIT,
    TENSOR_FEATURE_COUNT,
)
from anthropocene_rl.datasets.blobs import sha256_file
from anthropocene_rl.datasets.broad_domain_plan import (
    BROAD_DOMAIN_ACCEPTED_FAMILY_COUNTS,
    BROAD_DOMAIN_ACCEPTED_PARTITION_COUNTS,
)
from anthropocene_rl.datasets.models import (
    Partition,
    SnapshotBankManifest,
    SnapshotRecord,
    StrictModel,
)
from anthropocene_rl.datasets.partition import DeterministicPartitioner, LineageIdentity
from anthropocene_rl.datasets.pilot_plan import PILOT_ACCEPTED_FAMILY_COUNTS


class AuditIssue(StrictModel):
    code: str
    message: str
    snapshot_ids: tuple[str, ...] = ()


class SnapshotBankAuditReport(StrictModel):
    status: str
    manifest_sha256: str
    snapshot_count: int = Field(ge=0)
    partition_counts: dict[str, int]
    rejection_counts: dict[str, int]
    errors: tuple[AuditIssue, ...]
    warnings: tuple[AuditIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


def audit_snapshot_bank(
    manifest: SnapshotBankManifest,
    *,
    blob_root: str | Path,
    require_stage0_coverage: bool = True,
    require_cultivation_pilot_coverage: bool = False,
    require_broad_domain_coverage: bool = False,
    verify_blobs: bool = True,
) -> SnapshotBankAuditReport:
    errors: list[AuditIssue] = []
    warnings: list[AuditIssue] = []
    records = manifest.records
    root = Path(blob_root).resolve()

    _duplicates(records, "snapshot_id", errors)
    _duplicates(records, "native_fingerprint", errors)
    _duplicates(records, "content_sha256", errors)
    _check_identity(manifest, errors)
    _check_partitions(manifest, errors)
    _check_strata(records, errors)
    if verify_blobs:
        _check_blobs(records, root, errors)
    if require_stage0_coverage:
        _check_stage0_coverage(records, errors, warnings)
    if require_cultivation_pilot_coverage:
        _check_cultivation_pilot_coverage(records, errors)
    if require_broad_domain_coverage:
        _check_broad_domain_coverage(records, errors)

    rejection_counts = Counter(rejection.reason_code for rejection in manifest.rejected_candidates)
    partition_counts = Counter(record.partition.value for record in records)
    return SnapshotBankAuditReport(
        status="PASS" if not errors else "FAIL",
        manifest_sha256=manifest.sha256(),
        snapshot_count=len(records),
        partition_counts=dict(sorted(partition_counts.items())),
        rejection_counts=dict(sorted(rejection_counts.items())),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _duplicates(
    records: tuple[SnapshotRecord, ...], attribute: str, errors: list[AuditIssue]
) -> None:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[str(getattr(record, attribute)).lower()].append(record.snapshot_id)
    for value, snapshot_ids in sorted(grouped.items()):
        if len(snapshot_ids) > 1:
            errors.append(
                AuditIssue(
                    code=f"duplicate-{attribute.replace('_', '-')}",
                    message=f"Duplicate {attribute} {value}",
                    snapshot_ids=tuple(sorted(snapshot_ids)),
                )
            )


def _check_identity(manifest: SnapshotBankManifest, errors: list[AuditIssue]) -> None:
    expected_schema = (
        OBSERVATION_SCHEMA_VERSION,
        ACTION_SCHEMA_VERSION,
        ENVIRONMENT_SCHEMA_VERSION,
        CRITIC_SIDECAR_SCHEMA_VERSION,
        TENSOR_FEATURE_COUNT,
    )
    configuration_hashes: set[str] = set()
    build_hashes: set[str] = set()
    simulator_versions: set[str] = set()
    schema_hashes: set[str] = set()
    generator_identities: set[tuple[str, str, str, int, str]] = set()
    for record in manifest.records:
        actual_schema = (
            record.schema_identity.observation_schema_version,
            record.schema_identity.action_schema_version,
            record.schema_identity.environment_schema_version,
            record.schema_identity.critic_sidecar_schema_version,
            record.schema_identity.tensor_feature_count,
        )
        if record.native_simulator_sha != SIMULATOR_COMMIT or actual_schema != expected_schema:
            errors.append(
                AuditIssue(
                    code="native-schema-mismatch",
                    message="Record does not match the pinned native/schema cohort",
                    snapshot_ids=(record.snapshot_id,),
                )
            )
        configuration_hashes.add(record.configuration_hash)
        build_hashes.add(record.native_build_hash)
        simulator_versions.add(record.native_simulator_version)
        schema_hashes.add(record.schema_identity.observation_schema_hash)
        generator_identities.add(
            (
                record.generator.name,
                record.generator.version,
                record.generator.code_sha256,
                record.generator.generation_seed,
                record.generator.rng_algorithm,
            )
        )
    if len(configuration_hashes) > 1:
        errors.append(
            AuditIssue(
                code="configuration-cohort-drift",
                message=f"Bank contains {len(configuration_hashes)} native configuration hashes",
            )
        )
    if len(schema_hashes) > 1:
        errors.append(
            AuditIssue(
                code="observation-schema-hash-drift",
                message=f"Bank contains {len(schema_hashes)} observation schema hashes",
            )
        )
    if len(build_hashes) > 1 or len(simulator_versions) > 1:
        errors.append(
            AuditIssue(
                code="native-build-cohort-drift",
                message=(
                    "Bank contains multiple native build hashes or simulator versions"
                ),
            )
        )
    if len(generator_identities) > 1:
        errors.append(
            AuditIssue(
                code="generator-cohort-drift",
                message="Bank contains multiple native generator identities",
            )
        )


def _check_partitions(manifest: SnapshotBankManifest, errors: list[AuditIssue]) -> None:
    contract = manifest.partition_contract
    partitioner = DeterministicPartitioner(
        salt=contract.salt,
        train_fraction=contract.train_fraction,
        development_fraction=contract.development_fraction,
        heldout_fraction=contract.heldout_fraction,
    )
    by_world: dict[int, set[Partition]] = defaultdict(set)
    by_lineage: dict[tuple[int, str, str, str | None], set[Partition]] = defaultdict(set)
    by_counterfactual: dict[tuple[int, str], set[Partition]] = defaultdict(set)
    for record in manifest.records:
        historical = bool(
            record.selection_metadata.get("historical_or_consumed_diagnostic", False)
        )
        identity = LineageIdentity(
            record.world_seed,
            record.trajectory_family,
            record.lineage_id,
            record.counterfactual_group_id,
            historical,
        )
        expected = partitioner.assign(identity)
        if expected != record.partition:
            errors.append(
                AuditIssue(
                    code="partition-assignment-mismatch",
                    message=f"Expected {expected.value}, observed {record.partition.value}",
                    snapshot_ids=(record.snapshot_id,),
                )
            )
        by_world[record.world_seed].add(record.partition)
        by_lineage[identity.complete_group_key()].add(record.partition)
        if record.counterfactual_group_id:
            by_counterfactual[(record.world_seed, record.counterfactual_group_id)].add(
                record.partition
            )
        if historical and record.partition is not Partition.DEVELOPMENT:
            errors.append(
                AuditIssue(
                    code="consumed-diagnostic-outside-development",
                    message="Historical/consumed diagnostics must remain DEVELOPMENT",
                    snapshot_ids=(record.snapshot_id,),
                )
            )
    for label, groups in (
        ("world", by_world),
        ("lineage", by_lineage),
        ("counterfactual", by_counterfactual),
    ):
        for key, partitions in groups.items():
            if len(partitions) > 1:
                errors.append(
                    AuditIssue(
                        code=f"{label}-partition-leakage",
                        message=f"{label.title()} group {key!r} crosses partitions",
                    )
                )


def _check_blobs(
    records: tuple[SnapshotRecord, ...], root: Path, errors: list[AuditIssue]
) -> None:
    for record in records:
        path = (root / record.blob_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(
                AuditIssue(
                    code="blob-path-escape",
                    message="Blob path escapes the configured dataset root",
                    snapshot_ids=(record.snapshot_id,),
                )
            )
            continue
        if not path.is_file():
            errors.append(
                AuditIssue(
                    code="blob-missing",
                    message=f"Snapshot blob is missing: {record.blob_path}",
                    snapshot_ids=(record.snapshot_id,),
                )
            )
            continue
        digest, size = sha256_file(path)
        if digest != record.content_sha256 or size != record.content_bytes:
            errors.append(
                AuditIssue(
                    code="blob-integrity",
                    message="Snapshot blob SHA-256 or size does not match its record",
                    snapshot_ids=(record.snapshot_id,),
                )
            )


def _check_strata(records: tuple[SnapshotRecord, ...], errors: list[AuditIssue]) -> None:
    by_lineage: dict[
        tuple[int, str, str, str | None], list[SnapshotRecord]
    ] = defaultdict(list)
    for record in records:
        by_lineage[
            (
                record.world_seed,
                record.trajectory_family,
                record.lineage_id,
                record.counterfactual_group_id,
            )
        ].append(record)
    for record in records:
        strata = record.actual_strata
        q = max(strata.field_q, strata.fishing_q or 0.0)
        herd_ratio = strata.herd_zero_displacement_ratio
        lineage = by_lineage[
            (
                record.world_seed,
                record.trajectory_family,
                record.lineage_id,
                record.counterfactual_group_id,
            )
        ]
        technology_transition = bool(
            record.selection_metadata.get("frontier_request")
            == "technology-transition-cultivation"
            and record.decision_index > 0
            and strata.cultivation_available
            and any(
                ancestor.decision_index < record.decision_index
                and not ancestor.actual_strata.cultivation_available
                for ancestor in lineage
            )
        )
        expected_tags = {
            "cultivation-only": (
                strata.cultivation_available and not strata.husbandry_available
            ),
            "husbandry-only": (
                strata.husbandry_available and not strata.cultivation_available
            ),
            "mixed-cultivation-husbandry": (
                strata.cultivation_available and strata.husbandry_available
            ),
            "fishing": strata.fishing_ground_count > 0,
            "fishing-only": (
                strata.fishing_ground_count > 0
                and not strata.cultivation_available
                and not strata.husbandry_available
            ),
            "no-herd": strata.herd_animals == 0,
            "young-memory": record.institutional_memory.age_years < 20,
            "mature-memory": record.institutional_memory.age_years >= 180,
            "near-knee-q": 0.35 <= q <= 0.45,
            "high-q": q >= 0.5,
            "overstocked-herd": herd_ratio is not None and herd_ratio >= 1.3,
            "amenity-bottleneck": (
                strata.reserve_horizon_ratio >= 1
                and min(strata.clothing_coverage, strata.housing_coverage) < 0.5
            ),
            "no-development": not (
                strata.structural_masks.field_development
                or strata.structural_masks.herd_development
            ),
            "knowledge-only-husbandry-no-herd": (
                strata.husbandry_available and strata.herd_animals == 0
            ),
            "technology-transition": technology_transition,
        }
        tags = set(strata.tags)
        for tag, expected in expected_tags.items():
            if (tag in tags) != expected:
                errors.append(
                    AuditIssue(
                        code="measured-stratum-tag-mismatch",
                        message=f"Tag {tag!r} disagrees with saved-state measurements",
                        snapshot_ids=(record.snapshot_id,),
                    )
                )


def _check_stage0_coverage(
    records: tuple[SnapshotRecord, ...],
    errors: list[AuditIssue],
    warnings: list[AuditIssue],
) -> None:
    if not 24 <= len(records) <= 48:
        errors.append(
            AuditIssue(
                code="stage0-size",
                message=f"Stage-0 requires 24-48 snapshots; observed {len(records)}",
            )
        )
    families = Counter(record.scenario_family for record in records)
    required = {f"F{index:02d}" for index in range(1, 14)} | {"F17", "F18"}
    missing = sorted(required - set(families))
    if missing:
        errors.append(
            AuditIssue(
                code="stage0-family-coverage",
                message=f"Stage-0 is missing required scenario families: {missing}",
            )
        )
    priority = {f"F{index:02d}" for index in range(1, 8)} | {"F13", "F17", "F18"}
    fishing = {f"F{index:02d}" for index in range(8, 13)}
    if sum(families[family] for family in priority) <= sum(
        families[family] for family in fishing
    ):
        errors.append(
            AuditIssue(
                code="stage0-priority-balance",
                message="Priority F01-F07/F13/F17/F18 states must outnumber F08-F12 states",
            )
        )
    partitions = {record.partition for record in records}
    if partitions != set(Partition):
        errors.append(
            AuditIssue(
                code="stage0-partitions",
                message="Stage-0 must exercise TRAIN, DEVELOPMENT, and HELDOUT",
            )
        )
    all_tags = Counter(tag for record in records for tag in record.actual_strata.tags)
    required_tags = {
        "cultivation-only",
        "husbandry-only",
        "mixed-cultivation-husbandry",
        "no-development",
        "knowledge-only-husbandry-no-herd",
        "fishing-only",
        "young-memory",
        "mature-memory",
        "near-knee-q",
        "high-q",
        "overstocked-herd",
        "amenity-bottleneck",
        "technology-transition",
    }
    missing_tags = sorted(tag for tag in required_tags if not all_tags[tag])
    if missing_tags:
        errors.append(
            AuditIssue(
                code="stage0-measured-strata-coverage",
                message=f"Stage-0 is missing measured strata: {missing_tags}",
            )
        )
    ground_counts = Counter(record.actual_strata.fishing_ground_count for record in records)
    if not ground_counts[1] or not (ground_counts[2] or ground_counts[3]):
        errors.append(
            AuditIssue(
                code="stage0-fishing-ground-coverage",
                message="Stage-0 requires one-ground and multi-ground fishing states",
            )
        )
    for family in ("F14", "F15", "F16"):
        if families[family]:
            warnings.append(
                AuditIssue(
                    code="later-stage-family-present",
                    message=(
                        f"{family} is present although it is not prioritized in "
                        "fixed-location Stage-0"
                    ),
                )
            )


def _check_cultivation_pilot_coverage(
    records: tuple[SnapshotRecord, ...],
    errors: list[AuditIssue],
) -> None:
    if not 200 <= len(records) <= 500:
        errors.append(
            AuditIssue(
                code="cultivation-pilot-size",
                message=(
                    "Cultivation pilot requires 200-500 accepted snapshots; "
                    f"observed {len(records)}"
                ),
            )
        )
    required_counts = PILOT_ACCEPTED_FAMILY_COUNTS
    required_families = set(required_counts)
    families = Counter(record.scenario_family for record in records)
    if set(families) != required_families:
        errors.append(
            AuditIssue(
                code="cultivation-pilot-family-coverage",
                message=(
                    "Cultivation pilot family set differs from F01/F02/F03/F13/F17/F18"
                ),
            )
        )
    if {record.partition for record in records} != set(Partition):
        errors.append(
            AuditIssue(
                code="cultivation-pilot-partitions",
                message="Cultivation pilot must include TRAIN, DEVELOPMENT, and HELDOUT",
            )
        )
    for record in records:
        strata = record.actual_strata
        invalid = (
            not strata.cultivation_available
            or strata.husbandry_available
            or strata.owned_domestic_herd
            or strata.fishing_ground_count != 0
            or strata.structural_masks.movement
            or not strata.structural_masks.field_retention
            or not strata.structural_masks.field_development
        )
        if invalid:
            errors.append(
                AuditIssue(
                    code="cultivation-pilot-structural-scope",
                    message=(
                        "Pilot record enables movement/herding/fishing or lacks cultivation"
                    ),
                    snapshot_ids=(record.snapshot_id,),
                )
            )
    if dict(sorted(families.items())) != required_counts:
        errors.append(
            AuditIssue(
                code="cultivation-pilot-family-counts",
                message=(
                    "Cultivation pilot accepted family counts differ from the frozen "
                    f"300-state design: {dict(sorted(families.items()))}"
                ),
            )
        )
    required_measured_tags = {
        "F01": "young-memory",
        "F02": "near-knee-q",
        "F03": "high-q",
        "F13": "amenity-bottleneck",
    }
    for family, tag in required_measured_tags.items():
        if not any(
            record.scenario_family == family and tag in record.actual_strata.tags
            for record in records
        ):
            errors.append(
                AuditIssue(
                    code="cultivation-pilot-measured-strata",
                    message=f"{family} lacks an accepted measured {tag!r} state",
                )
            )
    memory_ages = {
        round(record.institutional_memory.age_years, 8)
        for record in records
        if record.scenario_family == "F17"
    }
    if len(memory_ages) < 2:
        errors.append(
            AuditIssue(
                code="cultivation-pilot-memory-contrast",
                message="F17 does not contain distinct measured memory ages",
            )
        )
    frontier_values = {
        record.actual_strata.field_frontier
        for record in records
        if record.scenario_family == "F18"
    }
    if not {"primitive", "plough"}.issubset(frontier_values):
        errors.append(
            AuditIssue(
                code="cultivation-pilot-frontier-contrast",
                message="F18 lacks measured primitive/plough frontier contrast",
            )
        )
    f15_tags = Counter(
        tag
        for record in records
        if record.scenario_family == "F15"
        for tag in record.actual_strata.tags
    )
    required_f15_tags = {
        "isolated",
        "trade-active",
        "food-import",
        "trade-dependent",
        "food-export",
        "material-tool-import",
        "migration-positive",
        "migration-negative",
    }
    if not required_f15_tags.issubset(f15_tags):
        errors.append(
            AuditIssue(
                code="cultivation-pilot-open-economy-coverage",
                message=(
                    "F15 lacks measured isolated/trade/import-or-export/signed-migration "
                    f"coverage: {dict(sorted(f15_tags.items()))}"
                ),
            )
        )
    f16 = [record for record in records if record.scenario_family == "F16"]
    if not any(record.actual_strata.fission_recent_split for record in f16) or not any(
        not record.actual_strata.fission_recent_split for record in f16
    ):
        errors.append(
            AuditIssue(
                code="cultivation-pilot-fission-causal-coverage",
                message="F16 requires both recent native fission and no-split states",
            )
        )


def _check_broad_domain_coverage(
    records: tuple[SnapshotRecord, ...],
    errors: list[AuditIssue],
) -> None:
    target = sum(BROAD_DOMAIN_ACCEPTED_FAMILY_COUNTS.values())
    if len(records) != target:
        errors.append(
            AuditIssue(
                code="broad-domain-size",
                message=f"Broad-domain cohort requires {target} snapshots; observed {len(records)}",
            )
        )
    families = Counter(record.scenario_family for record in records)
    if dict(sorted(families.items())) != BROAD_DOMAIN_ACCEPTED_FAMILY_COUNTS:
        errors.append(
            AuditIssue(
                code="broad-domain-family-counts",
                message=(
                    "Broad-domain accepted family counts differ from the frozen design: "
                    f"{dict(sorted(families.items()))}"
                ),
            )
        )
    actual_partition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        actual_partition_counts[record.scenario_family][record.partition.value] += 1
    if {
        family: dict(sorted(counts.items()))
        for family, counts in sorted(actual_partition_counts.items())
    } != {
        family: dict(sorted(counts.items()))
        for family, counts in sorted(BROAD_DOMAIN_ACCEPTED_PARTITION_COUNTS.items())
    }:
        errors.append(
            AuditIssue(
                code="broad-domain-partition-counts",
                message="Broad-domain family/partition counts differ from the frozen quotas",
            )
        )

    for record in records:
        strata = record.actual_strata
        if strata.fishing_available or strata.fishing_ground_count != 0:
            errors.append(
                AuditIssue(
                    code="broad-domain-fishing-scope",
                    message="Broad-domain v1 must not expose fishing as a policy domain",
                    snapshot_ids=(record.snapshot_id,),
                )
            )

    def family_records(family: str) -> list[SnapshotRecord]:
        return [record for record in records if record.scenario_family == family]

    for family in BROAD_DOMAIN_ACCEPTED_FAMILY_COUNTS:
        selected = family_records(family)
        if selected and not any(
            record.actual_strata.structural_masks.movement for record in selected
        ):
            errors.append(
                AuditIssue(
                    code="broad-domain-movement-coverage",
                    message=f"{family} has no accepted state with a legal non-stay movement choice",
                )
            )

    for family in ("F04", "F05"):
        invalid = [
            record.snapshot_id
            for record in family_records(family)
            if not (
                record.actual_strata.husbandry_available
                and not record.actual_strata.cultivation_available
                and record.actual_strata.owned_domestic_herd
                and record.actual_strata.structural_masks.herd_retention
                and record.actual_strata.structural_masks.herd_development
            )
        ]
        if invalid:
            errors.append(
                AuditIssue(
                    code="broad-domain-husbandry-only-contract",
                    message=(
                        f"{family} contains a state outside the husbandry-only "
                        "owned-herd contract"
                    ),
                    snapshot_ids=tuple(invalid),
                )
            )
    if not any(
        "overstocked-herd" in record.actual_strata.tags for record in family_records("F05")
    ):
        errors.append(
            AuditIssue(
                code="broad-domain-herd-overshoot-coverage",
                message="F05 lacks an accepted measured overstocked-herd state",
            )
        )

    for family in ("F06", "F07"):
        invalid = [
            record.snapshot_id
            for record in family_records(family)
            if not (
                record.actual_strata.cultivation_available
                and record.actual_strata.husbandry_available
                and record.actual_strata.owned_domestic_herd
                and record.actual_strata.structural_masks.field_development
                and record.actual_strata.structural_masks.herd_development
            )
        ]
        if invalid:
            errors.append(
                AuditIssue(
                    code="broad-domain-mixed-contract",
                    message=f"{family} contains a state outside the mixed field/herd contract",
                    snapshot_ids=tuple(invalid),
                )
            )

    f14 = family_records("F14")
    variants = {
        str(record.selection_metadata.get("causal_variant", "none")) for record in f14
    }
    if variants != {"low-lock-in", "high-lock-in"} or any(
        not record.actual_strata.structural_masks.movement for record in f14
    ):
        errors.append(
            AuditIssue(
                code="broad-domain-movement-lock-in-coverage",
                message="F14 lacks complete low/high lock-in pairs with legal movement",
            )
        )

    f15_tags = Counter(
        tag for record in family_records("F15") for tag in record.actual_strata.tags
    )
    required_f15_tags = {
        "isolated",
        "trade-active",
        "food-import",
        "trade-dependent",
        "food-export",
        "material-tool-import",
        "migration-positive",
        "migration-negative",
    }
    if not required_f15_tags.issubset(f15_tags):
        errors.append(
            AuditIssue(
                code="broad-domain-open-economy-coverage",
                message=f"F15 lacks required measured external-state tags: {dict(f15_tags)}",
            )
        )

    f16 = family_records("F16")
    if not any(record.actual_strata.fission_recent_split for record in f16) or not any(
        not record.actual_strata.fission_recent_split for record in f16
    ):
        errors.append(
            AuditIssue(
                code="broad-domain-fission-causal-coverage",
                message="F16 requires both recent native fission and no-split states",
            )
        )

    memory_ages = {
        round(record.institutional_memory.age_years, 8)
        for record in family_records("F17")
    }
    if len(memory_ages) < 3:
        errors.append(
            AuditIssue(
                code="broad-domain-memory-coverage",
                message="F17 requires young, intermediate, and mature legal memory ages",
            )
        )

    frontier_values = {
        record.actual_strata.field_frontier for record in family_records("F18")
    }
    if not {"primitive", "plough"}.issubset(frontier_values):
        errors.append(
            AuditIssue(
                code="broad-domain-frontier-coverage",
                message="F18 lacks measured primitive/plough frontier contrast",
            )
        )
