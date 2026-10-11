from pathlib import Path

from anthropocene_rl.datasets.audit import audit_snapshot_bank
from anthropocene_rl.datasets.blobs import ContentAddressedBlobStore
from anthropocene_rl.datasets.generation_config import SnapshotBankGenerationConfig
from anthropocene_rl.datasets.models import Partition, SnapshotBankManifest
from anthropocene_rl.datasets.normalization_audit import (
    CalibrationObservation,
    normalization_diagnostics,
)
from anthropocene_rl.datasets.partition import DeterministicPartitioner, LineageIdentity
from anthropocene_rl.observations.normalization import SemanticNormalizerV2
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

    assert sum(len(plan.snapshot_steps) for plan in resolved) == 28
    assert {plan.scenario_family for plan in resolved} == {
        *(f"F{index:02d}" for index in range(1, 14)),
        "F17",
        "F18",
    }
    world_partitions: dict[int, set[Partition]] = {}
    for plan in resolved:
        world_partitions.setdefault(plan.world_seed, set()).add(plan.partition)
    assert all(len(partitions) == 1 for partitions in world_partitions.values())


def test_audit_detects_world_lineage_and_counterfactual_leakage() -> None:
    manifest = SnapshotBankManifest.load(
        "datasets/manifests/elder-v2-stage0-v1.json"
    )
    pair_indices = [
        index
        for index, record in enumerate(manifest.records)
        if record.world_seed == 21150
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
    manifest = SnapshotBankManifest.load(
        "datasets/manifests/elder-v2-stage0-v1.json"
    )

    diagnostics = normalization_diagnostics(
        samples,
        normalizer=normalizer,
        manifest=manifest,
    )

    assert diagnostics["sample_counts"]["fishing"] == 2
    assert diagnostics["sample_counts"]["fishing-only"] == 1
