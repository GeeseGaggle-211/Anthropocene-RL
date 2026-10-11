from collections import Counter
from pathlib import Path

import pytest

from anthropocene_rl.datasets.audit import audit_snapshot_bank
from anthropocene_rl.datasets.blobs import ContentAddressedBlobStore
from anthropocene_rl.datasets.generation_config import SnapshotBankGenerationConfig
from anthropocene_rl.datasets.models import Partition, SnapshotBankManifest
from anthropocene_rl.datasets.normalization_audit import (
    CalibrationObservation,
    fit_train_and_report,
    normalization_diagnostics,
)
from anthropocene_rl.datasets.partition import DeterministicPartitioner, LineageIdentity
from anthropocene_rl.datasets.pilot_plan import (
    PILOT_ACCEPTED_PARTITION_COUNTS,
    PILOT_V3_ACCEPTED_PARTITION_COUNTS,
    build_cultivation_pilot_plan,
    build_normalization_v3_remediation_plan,
    build_normalization_v3_scope_safe_plan,
    cultivation_pilot_plan_summary,
)
from anthropocene_rl.observations.normalization import (
    NORMALIZATION_VERSION_V2,
    NORMALIZATION_VERSION_V3,
    SemanticNormalizerV2,
)
from anthropocene_rl.observations.schema import EncodedObservation


def test_content_addressed_blob_store_deduplicates(tmp_path: Path) -> None:
    source = tmp_path / "candidate.snapshot"
    source.write_bytes(b"authoritative SaveCodec bytes")
    store = ContentAddressedBlobStore(tmp_path / "blobs")

    first = store.put_file(source)
    second = store.put_file(source)

    assert first == second
    assert store.verify(first)
    assert first.relative_path == f"sha256/{first.sha256[:2]}/{first.sha256}.snapshot"


def test_partitioning_is_deterministic_and_isolates_complete_world() -> None:
    partitioner = DeterministicPartitioner(salt="stage0-test")
    original = LineageIdentity(1001, "bounded", "lineage-a", "counterfactual-a")
    sibling_branch = LineageIdentity(1001, "heuristic", "lineage-b", None)

    assert partitioner.assign(original) == partitioner.assign(original)
    assert partitioner.assign(original) == partitioner.assign(sibling_branch)


def test_historical_or_consumed_diagnostics_are_development() -> None:
    partitioner = DeterministicPartitioner(salt="stage0-test")
    historical = LineageIdentity(
        9301,
        "elder-v2-handoff",
        "native-fixture-9301",
        historical_or_consumed_diagnostic=True,
    )

    assert partitioner.assign(historical) is Partition.DEVELOPMENT


def test_stage0_plan_partitions_before_timestamp_selection() -> None:
    config = SnapshotBankGenerationConfig.load("configs/datasets/stage0-v1.yaml")
    resolved, _ = config.resolve_partitions()
    manifest = SnapshotBankManifest.load("datasets/manifests/elder-v2-stage0-v1.json")

    assert sum(len(plan.snapshot_steps) for plan in resolved) == 28
    assert config.canonical_sha256() == manifest.generator_config_sha256
    assert {plan.scenario_family for plan in resolved} == {
        *(f"F{index:02d}" for index in range(1, 14)),
        "F17",
        "F18",
    }
    world_partitions: dict[int, set[Partition]] = {}
    for plan in resolved:
        world_partitions.setdefault(plan.world_seed, set()).add(plan.partition)
    assert all(len(partitions) == 1 for partitions in world_partitions.values())


def test_cultivation_pilot_plan_is_pairwise_bounded_and_partitioned_first() -> None:
    config = build_cultivation_pilot_plan()
    committed = SnapshotBankGenerationConfig.load("configs/datasets/cultivation-pilot-v1.yaml")
    resolved, _ = config.resolve_partitions()
    summary = cultivation_pilot_plan_summary(config)

    assert committed.canonical_sha256() == config.canonical_sha256()
    assert config.bank_kind == "cultivation-pilot"
    assert sum(len(plan.snapshot_steps) for plan in resolved) == 500
    assert len({plan.world_seed for plan in resolved}) == 299
    assert {plan.scenario_family for plan in resolved} == {
        "F01",
        "F02",
        "F03",
        "F13",
        "F15",
        "F16",
        "F17",
        "F18",
    }
    assert all(plan.economy == "cultivation" for plan in resolved)
    assert all(plan.herd_support_ratio == 0 for plan in resolved)
    assert all(plan.fishing_ground_target == 0 for plan in resolved)
    world_partitions: dict[int, set[Partition]] = {}
    for plan in resolved:
        world_partitions.setdefault(plan.world_seed, set()).add(plan.partition)
    assert all(len(partitions) == 1 for partitions in world_partitions.values())
    assert set(summary["partition_world_counts"]) == {
        "train",
        "development",
        "heldout",
    }
    assert summary["requested_snapshot_count"] == 500
    assert summary["target_accepted_snapshot_count"] == 300
    assert summary["target_accepted_partition_counts"] == PILOT_ACCEPTED_PARTITION_COUNTS
    assert summary["family_snapshot_counts"] == {
        "F01": 30,
        "F02": 35,
        "F03": 35,
        "F13": 40,
        "F15": 218,
        "F16": 60,
        "F17": 36,
        "F18": 46,
    }
    assert summary["target_accepted_family_counts"] == {
        "F01": 30,
        "F02": 35,
        "F03": 35,
        "F13": 40,
        "F15": 50,
        "F16": 40,
        "F17": 24,
        "F18": 46,
    }
    assert summary["requested_pairwise_coverage"]["F01"]["fraction"] > 0.9


def test_normalization_v3_remediation_plan_is_fresh_and_disjoint() -> None:
    original = build_cultivation_pilot_plan()
    remediation = build_normalization_v3_remediation_plan()
    original_worlds = {trajectory.world_seed for trajectory in original.trajectories}
    remediation_worlds = {trajectory.world_seed for trajectory in remediation.trajectories}
    summary = cultivation_pilot_plan_summary(remediation)

    assert original.normalization_version == NORMALIZATION_VERSION_V2
    assert remediation.normalization_version == NORMALIZATION_VERSION_V3
    assert remediation_worlds.isdisjoint(original_worlds)
    assert len(remediation_worlds) == 393
    assert summary["requested_snapshot_count"] == 698
    assert summary["target_accepted_snapshot_count"] == 300
    assert summary["target_accepted_partition_counts"] == (PILOT_V3_ACCEPTED_PARTITION_COUNTS)
    assert Counter(
        {
            partition: sum(
                counts[partition] for counts in PILOT_V3_ACCEPTED_PARTITION_COUNTS.values()
            )
            for partition in ("train", "development", "heldout")
        }
    ) == Counter({"train": 210, "development": 46, "heldout": 44})
    assert remediation.calibration_rollout_steps == 5
    assert summary["normalization_rule"].startswith("fit final elder-normalization-v3")


def test_scope_safe_plan_is_a_new_generation_of_the_same_candidate_universe() -> None:
    remediation = build_normalization_v3_remediation_plan()
    scope_safe = build_normalization_v3_scope_safe_plan()

    assert scope_safe.bank_id != remediation.bank_id
    assert scope_safe.bank_version == "3.1.0-scope-safe"
    assert scope_safe.canonical_sha256() != remediation.canonical_sha256()
    assert sum(len(plan.snapshot_steps) for plan in scope_safe.trajectories) == 796
    assert {plan.world_seed for plan in remediation.trajectories}.issubset(
        {plan.world_seed for plan in scope_safe.trajectories}
    )


def test_frozen_cultivation_pilot_has_every_family_in_every_partition() -> None:
    manifest = SnapshotBankManifest.load("datasets/manifests/elder-v2-cultivation-pilot-v1.json")
    observed = Counter(
        (record.scenario_family, record.partition.value) for record in manifest.records
    )
    expected = {
        (family, partition): count
        for family, partitions in PILOT_ACCEPTED_PARTITION_COUNTS.items()
        for partition, count in partitions.items()
    }

    assert observed == expected
    assert Counter(record.partition.value for record in manifest.records) == {
        "train": 210,
        "development": 45,
        "heldout": 45,
    }


def test_audit_detects_world_lineage_and_counterfactual_leakage() -> None:
    manifest = SnapshotBankManifest.load("datasets/manifests/elder-v2-stage0-v1.json")
    pair_indices = [
        index for index, record in enumerate(manifest.records) if record.world_seed == 21150
    ]
    assert len(pair_indices) == 2
    records = list(manifest.records)
    records[pair_indices[0]] = records[pair_indices[0]].model_copy(
        update={"partition": Partition.HELDOUT}
    )
    leaking = manifest.model_copy(update={"records": tuple(records)})

    report = audit_snapshot_bank(
        leaking,
        blob_root="datasets/blobs",
        require_stage0_coverage=False,
        verify_blobs=False,
    )
    codes = {issue.code for issue in report.errors}

    assert "partition-assignment-mismatch" in codes
    assert "world-partition-leakage" in codes
    assert "lineage-partition-leakage" in codes
    assert "counterfactual-partition-leakage" in codes


def test_native_snake_case_tensor_payload_is_accepted() -> None:
    observation = EncodedObservation.from_native(
        {
            "schema_version": 2,
            "values": [0.0] * 512,
            "present": [False] * 512,
        }
    )

    assert len(observation.values) == 512


def test_normalization_diagnostics_separates_fishing_only_slice() -> None:
    observation = EncodedObservation(
        values=(0.0,) * 512,
        present=(False,) * 512,
    )
    normalizer = SemanticNormalizerV2.fit(
        [observation],
        partition_id="unit-train",
    )
    samples = [
        CalibrationObservation(
            snapshot_id="fishing-only",
            partition=Partition.TRAIN,
            source="reset",
            observation=observation,
            slices=frozenset({"fishing", "fishing-only"}),
        ),
        CalibrationObservation(
            snapshot_id="cultivation-fishing",
            partition=Partition.TRAIN,
            source="reset",
            observation=observation,
            slices=frozenset({"fishing", "cultivation-only"}),
        ),
    ]
    manifest = SnapshotBankManifest.load("datasets/manifests/elder-v2-stage0-v1.json")

    diagnostics = normalization_diagnostics(
        samples,
        normalizer=normalizer,
        manifest=manifest,
    )

    assert diagnostics["sample_counts"]["fishing"] == 2
    assert diagnostics["sample_counts"]["fishing-only"] == 1
    assert diagnostics["saturation_gate_pass"] is True


def test_final_pilot_normalizer_fits_train_only(tmp_path: Path) -> None:
    observation = EncodedObservation(
        values=(0.0,) * 512,
        present=(False,) * 512,
    )
    manifest = SnapshotBankManifest.load("datasets/manifests/elder-v2-stage0-v1.json")
    samples = [
        CalibrationObservation(
            snapshot_id="train",
            partition=Partition.TRAIN,
            source="reset",
            observation=observation,
            slices=frozenset(),
        ),
        CalibrationObservation(
            snapshot_id="development",
            partition=Partition.DEVELOPMENT,
            source="calibration-step-1",
            observation=observation,
            slices=frozenset(),
        ),
    ]
    config = SnapshotBankGenerationConfig.load("configs/datasets/stage0-v1.yaml")

    normalizer, diagnostics = fit_train_and_report(
        samples,
        manifest=manifest,
        capability_experience_thresholds=config.capability_experience_thresholds,
        local_movement_radius=config.local_movement_radius,
        normalizer_path=tmp_path / "normalizer.json",
        diagnostics_path=tmp_path / "diagnostics.json",
        rng_provenance="unit-test",
        partition_suffix="final",
        artifact_status="FINAL_PILOT_TRAIN_ONLY_NORMALIZER",
    )

    assert normalizer.manifest.partition_id.endswith(":train:final")
    assert normalizer.manifest.fit_observation_count == 1
    assert diagnostics["status"] == "FINAL_PILOT_TRAIN_ONLY_NORMALIZER"
    assert diagnostics["heldout_or_development_used_for_fit"] is False


def test_normalization_v3_rejects_missing_sparse_causal_support(tmp_path: Path) -> None:
    manifest = SnapshotBankManifest.load("datasets/manifests/elder-v2-stage0-v1.json")
    record = next(record for record in manifest.records if record.partition is Partition.TRAIN)
    observation = EncodedObservation(
        values=(0.0,) * 512,
        present=(True,) * 512,
    )
    config = SnapshotBankGenerationConfig.load("configs/datasets/stage0-v1.yaml")

    with pytest.raises(ValueError, match="sparse causal TRAIN support"):
        fit_train_and_report(
            [
                CalibrationObservation(
                    snapshot_id=record.snapshot_id,
                    partition=Partition.TRAIN,
                    source="reset",
                    observation=observation,
                    slices=frozenset(),
                )
            ],
            manifest=manifest,
            capability_experience_thresholds=config.capability_experience_thresholds,
            local_movement_radius=config.local_movement_radius,
            normalizer_path=tmp_path / "normalizer-v3.json",
            diagnostics_path=tmp_path / "diagnostics-v3.json",
            rng_provenance="unit-test",
            partition_suffix="v3-gate",
            artifact_status="UNIT_TEST",
            normalization_version=NORMALIZATION_VERSION_V3,
        )
