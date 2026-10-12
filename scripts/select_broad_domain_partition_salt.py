"""Choose an outcome-blind partition salt for the bounded broad-domain retry.

The first native pass established which original causal groups are complete,
scope-safe, and identity-unique.  This script searches salts using only that
eligibility ledger plus the declared supplemental F15 construction groups.  It
never reads rewards, welfare, or policy-evaluation outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropocene_rl.datasets.broad_domain_plan import (
    BROAD_DOMAIN_ACCEPTED_PARTITION_COUNTS,
)
from anthropocene_rl.datasets.generation_config import SnapshotBankGenerationConfig

PARTITIONS = ("train", "development", "heldout")
SALT_PREFIX = "elder-v2-broad-domain-50d-partitions-v2-"

# These margins are construction safeguards, not acceptance criteria.  They
# ensure the prospective hard-trade supplements are spread across all splits
# before the one bounded native retry is run.
SUPPLEMENT_PARTITION_MARGINS = {
    ("food-shortfall-contact", "material-tool-import"): {
        "train": 15,
        "development": 4,
        "heldout": 4,
    },
    ("mild-import", "surplus-export"): {
        "train": 7,
        "development": 2,
        "heldout": 2,
    },
}


@dataclass(frozen=True)
class Group:
    family: str
    group_id: str
    world_seed: int
    records: int
    contrast: tuple[str, ...]
    prospective_supplement: bool


def _partition(salt: str, world_seed: int) -> str:
    material = f"{salt}\x00world-seed\x00{world_seed}".encode()
    unit = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64
    if unit < 0.70:
        return "train"
    if unit < 0.85:
        return "development"
    return "heldout"


def _exact_sum_possible(sizes: list[int], target: int) -> bool:
    reachable = {0}
    for size in sizes:
        reachable |= {subtotal + size for subtotal in tuple(reachable) if subtotal + size <= target}
    return target in reachable


def _f15_allocation(
    availability: dict[tuple[str, ...], dict[str, int]],
) -> dict[str, dict[str, int]] | None:
    contrasts = sorted(availability)
    if len(contrasts) != 5:
        return None
    remaining = {
        partition: BROAD_DOMAIN_ACCEPTED_PARTITION_COUNTS["F15"][partition] // 2
        for partition in PARTITIONS
    }
    allocation: dict[tuple[str, ...], dict[str, int]] = {}

    def search(index: int) -> bool:
        if index == len(contrasts):
            return all(value == 0 for value in remaining.values())
        contrast = contrasts[index]
        for train in range(6):
            for development in range(6 - train):
                proposal = {
                    "train": train,
                    "development": development,
                    "heldout": 5 - train - development,
                }
                if any(
                    proposal[partition] > availability[contrast].get(partition, 0)
                    or proposal[partition] > remaining[partition]
                    for partition in PARTITIONS
                ):
                    continue
                for partition in PARTITIONS:
                    remaining[partition] -= proposal[partition]
                allocation[contrast] = proposal
                if search(index + 1):
                    return True
                allocation.pop(contrast, None)
                for partition in PARTITIONS:
                    remaining[partition] += proposal[partition]
        return False

    if not search(0):
        return None
    return {" + ".join(contrast): allocation[contrast] for contrast in contrasts}


def _groups(
    config: SnapshotBankGenerationConfig,
    ledger: dict[str, Any],
) -> tuple[list[Group], set[str]]:
    observed_eligibility = {
        str(group["group_id"]): bool(group["eligible"]) for group in ledger["groups"]
    }
    plans: dict[str, list[Any]] = defaultdict(list)
    for trajectory in config.trajectories:
        group_id = trajectory.counterfactual_group_id or trajectory.lineage_id
        plans[group_id].append(trajectory)

    result: list[Group] = []
    undeclared_in_ledger: set[str] = set()
    for group_id, trajectories in sorted(plans.items()):
        family = trajectories[0].scenario_family
        prospective = family in {"F03", "F15"} and "-supplement-" in group_id
        if not prospective and group_id not in observed_eligibility:
            undeclared_in_ledger.add(group_id)
            continue
        if not prospective and not observed_eligibility[group_id]:
            continue
        result.append(
            Group(
                family=family,
                group_id=group_id,
                world_seed=trajectories[0].world_seed,
                records=sum(len(trajectory.snapshot_steps) for trajectory in trajectories),
                contrast=tuple(sorted(trajectory.causal_variant for trajectory in trajectories)),
                prospective_supplement=prospective,
            )
        )
    return result, undeclared_in_ledger


def _evaluate(salt: str, groups: list[Group]) -> tuple[list[str], dict[str, Any]]:
    assigned = [(group, _partition(salt, group.world_seed)) for group in groups]
    reasons: list[str] = []
    family_capacity: dict[str, dict[str, int]] = {}
    for family, quotas in BROAD_DOMAIN_ACCEPTED_PARTITION_COUNTS.items():
        if family == "F15":
            continue
        family_capacity[family] = {}
        for partition, target in quotas.items():
            sizes = [
                group.records
                for group, assigned_partition in assigned
                if group.family == family and assigned_partition == partition
            ]
            capacity = sum(sizes)
            family_capacity[family][partition] = capacity
            if capacity < target:
                reasons.append(f"{family}/{partition}:capacity={capacity}<target={target}")
            elif not _exact_sum_possible(sizes, target):
                reasons.append(f"{family}/{partition}:exact-sum-unreachable:{sizes}->{target}")

    f15_availability: dict[tuple[str, ...], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    supplement_availability: dict[tuple[str, ...], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for group, partition in assigned:
        if group.family != "F15":
            continue
        f15_availability[group.contrast][partition] += 1
        if group.prospective_supplement:
            supplement_availability[group.contrast][partition] += 1
    allocation = _f15_allocation(f15_availability)
    if allocation is None:
        reasons.append("F15:contrast-plus-partition-allocation-unreachable")
    for contrast, margins in SUPPLEMENT_PARTITION_MARGINS.items():
        for partition, minimum in margins.items():
            available = supplement_availability[contrast][partition]
            if available < minimum:
                reasons.append(
                    "F15:supplement-margin:"
                    f"{' + '.join(contrast)}/{partition}={available}<minimum={minimum}"
                )

    diagnostics = {
        "family_record_capacity": family_capacity,
        "f15_pair_availability": {
            " + ".join(contrast): dict(sorted(counts.items()))
            for contrast, counts in sorted(f15_availability.items())
        },
        "f15_prospective_supplement_pair_availability": {
            " + ".join(contrast): dict(sorted(counts.items()))
            for contrast, counts in sorted(supplement_availability.items())
        },
        "f15_selected_allocation": allocation,
    }
    return reasons, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-salts", type=int, default=100_000)
    args = parser.parse_args()

    config = SnapshotBankGenerationConfig.load(args.config)
    ledger_bytes = args.ledger.read_bytes()
    ledger = json.loads(ledger_bytes)
    groups, missing = _groups(config, ledger)
    if missing:
        raise ValueError(f"Non-supplement groups absent from eligibility ledger: {sorted(missing)}")

    rejected: list[dict[str, Any]] = []
    selected_salt: str | None = None
    selected_diagnostics: dict[str, Any] | None = None
    for index in range(args.max_salts):
        salt = f"{SALT_PREFIX}{index}"
        reasons, diagnostics = _evaluate(salt, groups)
        if not reasons:
            selected_salt = salt
            selected_diagnostics = diagnostics
            break
        rejected.append({"salt": salt, "reasons": reasons})
    if selected_salt is None or selected_diagnostics is None:
        raise RuntimeError(f"No feasible salt in first {args.max_salts} candidates")

    selected_config = config.model_copy(
        update={
            "partition": config.partition.model_copy(update={"salt": selected_salt})
        }
    )

    payload = {
        "status": "SALT_SELECTED_NOT_NATIVE_VALIDATED",
        "selection_rule": (
            "first salt satisfying exact family/partition subset feasibility, F15 exact "
            "five-pair-per-contrast allocation, and predeclared hard-trade supplement margins"
        ),
        "outcome_blind_fields": [
            "world_seed",
            "causal_group_id",
            "record_count",
            "native_complete_scope_safe_identity_unique_eligibility",
            "declared_causal_variant",
        ],
        "prohibited_selection_fields": [
            "reward",
            "welfare",
            "policy_action",
            "training_metric",
            "evaluation_metric",
        ],
        "source_config_sha256": config.canonical_sha256(),
        "selected_config_sha256": selected_config.canonical_sha256(),
        "source_eligibility_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "eligible_or_prospective_group_count": len(groups),
        "supplement_partition_margins": {
            " + ".join(contrast): margins
            for contrast, margins in SUPPLEMENT_PARTITION_MARGINS.items()
        },
        "selected_salt": selected_salt,
        "evaluated_salt_count": len(rejected) + 1,
        "rejected_salts": rejected,
        "selected_diagnostics": selected_diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
