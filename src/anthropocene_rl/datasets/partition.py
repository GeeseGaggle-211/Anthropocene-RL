"""Deterministic leakage-safe snapshot-bank partitioning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from anthropocene_rl.datasets.models import Partition, PartitionContract

PARTITION_ALGORITHM = "sha256-world-seed-v1"
GROUPING_FIELDS = (
    "world_seed",
    "trajectory_family",
    "lineage_id",
    "counterfactual_group_id",
)


@dataclass(frozen=True)
class LineageIdentity:
    world_seed: int
    trajectory_family: str
    lineage_id: str
    counterfactual_group_id: str | None = None
    historical_or_consumed_diagnostic: bool = False

    def complete_group_key(self) -> tuple[int, str, str, str | None]:
        return (
            self.world_seed,
            self.trajectory_family,
            self.lineage_id,
            self.counterfactual_group_id,
        )


class DeterministicPartitioner:
    """Assign complete worlds before any trajectory timestamp is selected.

    World-seed isolation is intentionally stricter than the minimum lineage
    tuple contract: branches from one generated physical world cannot cross a
    partition even if their lineage labels differ.
    """

    def __init__(
        self,
        *,
        salt: str,
        train_fraction: float = 0.6,
        development_fraction: float = 0.2,
        heldout_fraction: float = 0.2,
    ) -> None:
        self.contract = PartitionContract(
            algorithm=PARTITION_ALGORITHM,
            salt=salt,
            train_fraction=train_fraction,
            development_fraction=development_fraction,
            heldout_fraction=heldout_fraction,
            grouping_fields=GROUPING_FIELDS,
            isolate_world_seed=True,
        )

    def assign(self, identity: LineageIdentity) -> Partition:
        if identity.historical_or_consumed_diagnostic:
            return Partition.DEVELOPMENT
        material = f"{self.contract.salt}\x00world-seed\x00{identity.world_seed}".encode()
        digest = hashlib.sha256(material).digest()
        unit = int.from_bytes(digest[:8], "big") / 2**64
        if unit < self.contract.train_fraction:
            return Partition.TRAIN
        if unit < self.contract.train_fraction + self.contract.development_fraction:
            return Partition.DEVELOPMENT
        return Partition.HELDOUT
