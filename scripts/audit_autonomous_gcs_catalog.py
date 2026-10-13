"""Read-only metadata audit: never admits unmaterialized legacy snapshots to training."""

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def main():
    manifest = Path("datasets/manifests/elder-v2-broad-domain-50d-scaleout-v1.json")
    bank = json.loads(manifest.read_text())
    records = bank["records"]
    current = json.loads(Path("datasets/autonomous-ppo-gate-v1/manifest.json").read_text())[
        "entries"
    ]
    current_seeds = {e["world_seed"] for e in current}
    current_hashes = {e["sha256"] for e in current}
    collisions = {}
    for key in ["world_seed", "lineage_id", "counterfactual_group_id", "content_sha256"]:
        groups = defaultdict(set)
        for row in records:
            if row.get(key) is not None:
                groups[row[key]].add(row["partition"])
        collisions[key] = sum(len(partitions) > 1 for partitions in groups.values())
    families = defaultdict(Counter)
    for row in records:
        families[row["scenario_family"]][row["partition"]] += 1
    # Inspect physical metadata for TRAIN only; sealed evaluation outcomes are not fitted.
    train = [r for r in records if r["partition"] == "train"]
    candidates = [
        r
        for r in train
        if r["actual_strata"].get("cultivation_available")
        and not r["actual_strata"].get("husbandry_available")
        and not r["actual_strata"].get("fishing_available")
    ]
    # Preserve transitive whole groups across every identity, not just each field alone.
    parents = list(range(len(train)))

    def root(index):
        while parents[index] != index:
            index = parents[index]
        return index

    for key in ["world_seed", "lineage_id", "counterfactual_group_id", "content_sha256"]:
        seen = {}
        for index, row in enumerate(train):
            if row.get(key) is None:
                continue
            if row[key] in seen:
                parents[root(index)] = root(seen[row[key]])
            else:
                seen[row[key]] = index
    groups = defaultdict(list)
    for index, row in enumerate(train):
        groups[root(index)].append(row)
    candidate_ids = {r["snapshot_id"] for r in candidates}
    complete = [
        group for group in groups.values() if all(r["snapshot_id"] in candidate_ids for r in group)
    ]
    selected = []
    for family in sorted({r["scenario_family"] for r in candidates}):
        row = min(
            (r for r in candidates if r["scenario_family"] == family),
            key=lambda r: r["content_sha256"],
        )
        selected.append(
            {
                k: row[k]
                for k in [
                    "snapshot_id",
                    "content_sha256",
                    "world_seed",
                    "partition",
                    "scenario_family",
                    "controlled",
                    "blob_path",
                ]
            }
        )
    audit = json.loads(
        Path("evidence/elder-broad-domain-50d-scaleout-v1/storage-audit.json").read_text()
    )
    ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                "/workspace/Anthropocene-elder",
                "merge-base",
                "--is-ancestor",
                bank["native_simulator_sha"],
                "HEAD",
            ],
            capture_output=True,
        ).returncode
        == 0
    )
    report = dict(
        version="autonomous-gcs-catalog-audit-v1",
        evidence_scope="retained metadata only; no live GCS listing or blob validation",
        live_access=(
            "unavailable: no configured GCP identity, Storage endpoint permission "
            "or installed GCP SDK"
        ),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        bank_id=bank["bank_id"],
        source_native_commit=bank["native_simulator_sha"],
        source_is_ancestor=ancestor,
        records=len(records),
        partitions=dict(Counter(r["partition"] for r in records)),
        family_partitions=families,
        partition_group_collisions=collisions,
        current_bank_world_seed_overlap=sorted(current_seeds & {r["world_seed"] for r in records}),
        current_bank_content_overlap=len(current_hashes & {r["content_sha256"] for r in records}),
        local_blobs_present=sum(
            (Path(bank["blob_storage"]) / r["blob_path"]).is_file() for r in records
        ),
        legacy_schema=records[0]["schema"],
        current_schema=dict(action=3, observation=4, environment=4, replay=4, features=761),
        train_metadata_cultivation_only_candidates=len(candidates),
        complete_train_metadata_candidate_groups=len(complete),
        records_in_complete_train_metadata_candidate_groups=sum(len(g) for g in complete),
        train_candidates_by_family=dict(Counter(r["scenario_family"] for r in candidates)),
        prospective_train_smoke_sample=selected,
        gcs_published_manifest=audit["dataset"]["published_manifest"],
        gcs_shards=audit["dataset"]["shards"],
        current_training_integration="none; frozen reviewed bank retained",
        future_gate=[
            "materialize pinned GCS generations and verify shard/member SHA256",
            "preserve world/lineage/counterfactual/content partition groups; keep HELDOUT sealed",
            "load unchanged saves with pinned native v4 bridge; verify schema and physical state",
            "rebuild native observations/masks; never reuse legacy actions or normalizer",
            "verify chronology, finite action probabilities, native consequences and save/replay",
            "audit new recipe/control masks, demographic/trade flags, "
            "and multi-Elder quote opportunities",
            "create a new derived manifest; independently review before adding TRAIN records",
        ],
    )
    Path("evidence/autonomous-continuation-v4/gcs-catalog-audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "records",
                    "partitions",
                    "source_is_ancestor",
                    "partition_group_collisions",
                    "current_bank_world_seed_overlap",
                    "local_blobs_present",
                    "train_metadata_cultivation_only_candidates",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
