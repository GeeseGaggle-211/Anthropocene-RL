from __future__ import annotations

from collections import Counter

from anthropocene_rl.datasets.broad_domain_plan import (
    BROAD_DOMAIN_ACCEPTED_FAMILY_COUNTS,
    BROAD_DOMAIN_ACCEPTED_PARTITION_COUNTS,
    broad_domain_plan_summary,
    build_elder_broad_domain_plan,
    save_elder_broad_domain_plan,
)
from anthropocene_rl.datasets.generation_config import SnapshotBankGenerationConfig


def test_broad_domain_plan_is_deterministic_and_quota_safe(tmp_path) -> None:
    first = build_elder_broad_domain_plan()
    second = build_elder_broad_domain_plan()
    assert first.canonical_sha256() == second.canonical_sha256()
    assert first.bank_kind == "elder-broad-domain"
    assert len(first.trajectories) > 400
    assert sum(len(item.snapshot_steps) for item in first.trajectories) == 800
    assert sum(BROAD_DOMAIN_ACCEPTED_FAMILY_COUNTS.values()) == 499

    resolved, _ = first.resolve_partitions()
    capacity: dict[str, Counter[str]] = {
        family: Counter() for family in BROAD_DOMAIN_ACCEPTED_FAMILY_COUNTS
    }
    for trajectory in resolved:
        capacity[trajectory.scenario_family][trajectory.partition.value] += len(
            trajectory.snapshot_steps
        )
    for family, quotas in BROAD_DOMAIN_ACCEPTED_PARTITION_COUNTS.items():
        assert sum(quotas.values()) == BROAD_DOMAIN_ACCEPTED_FAMILY_COUNTS[family]
        for partition, required in quotas.items():
            assert capacity[family][partition] >= required

    assert {item.economy for item in first.trajectories} == {
        "cultivation",
        "husbandry",
        "mixed",
    }
    assert all(item.fishing_ground_target == 0 for item in first.trajectories)
    assert "broad_domain_accepted_partition_counts" not in first.native_payload()


def test_broad_domain_plan_serialization_roundtrip(tmp_path) -> None:
    config_path = tmp_path / "plan.yaml"
    summary_path = tmp_path / "summary.json"
    summary = save_elder_broad_domain_plan(
        config_path=config_path,
        summary_path=summary_path,
    )
    loaded = SnapshotBankGenerationConfig.load(config_path)
    assert loaded.canonical_sha256() == summary["generator_config_sha256"]
    assert summary == broad_domain_plan_summary(loaded)
    assert summary["primary_episode_decisions"] == 50
    assert summary["active_policy_domains"] == [
        "cultivation",
        "husbandry",
        "movement",
    ]
