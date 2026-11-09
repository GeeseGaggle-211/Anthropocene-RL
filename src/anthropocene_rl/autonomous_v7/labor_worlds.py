"""Shared world-partition guard for every labor training path (G0 manifest sections 4, 8, 9).

The reserved fitting worlds 90154-90159 (amendment r5-b) never train any
network: not the actor (BC, DAgger collection and refit, PPO rollouts), not the
critics (V, A, Q) and not the search labels that feed them. They may be
searched or evaluated only in an explicit held-out-test mode whose outputs
carry :data:`HELD_OUT_TEST_PARTITION` in their source record and in every shard
meta; every training loader refuses such outputs.

This module is the single definition of the reserved range and its guards.
``labor_privileged.RESERVED_HELDOUT_WORLDS`` re-exports the same object.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

RESERVED_HELDOUT_WORLDS = range(90154, 90160)
RESERVED_HELDOUT_TEXT = "90154-90159"
# Partition tag written into source records and shard metas.
FITTING_PARTITION = "fitting"
HELD_OUT_TEST_PARTITION = "held-out-test"
# Curated calibration scenario sets (labor_scenarios, worlds 90180-90199): evaluation
# data only; every training loader refuses them like held-out-test data.
CALIBRATION_PARTITION = "calibration"


def reserved_worlds(worlds: Iterable[int]) -> list[int]:
    """The reserved held-out worlds among ``worlds`` (sorted, unique)."""
    return sorted({int(w) for w in worlds} & set(RESERVED_HELDOUT_WORLDS))


def require_training_worlds(worlds: Iterable[int], context: str) -> list[int]:
    """Refuse any reserved held-out world as training data; returns the worlds unchanged."""
    worlds = [int(w) for w in worlds]
    reserved = reserved_worlds(worlds)
    if reserved:
        raise ValueError(
            f"{context} refuses reserved held-out worlds {reserved} ({RESERVED_HELDOUT_TEXT}); "
            "they never train any network (use the explicit --held-out-test mode to search or "
            "evaluate them)"
        )
    return worlds


def require_held_out_test_worlds(worlds: Iterable[int], context: str) -> list[int]:
    """Held-out-test mode runs on reserved worlds only, so its outputs stay pure test data."""
    worlds = [int(w) for w in worlds]
    if not worlds:
        raise ValueError(f"{context}: held-out-test mode needs at least one world")
    outside = sorted({w for w in worlds if w not in RESERVED_HELDOUT_WORLDS})
    if outside:
        raise ValueError(
            f"{context}: held-out-test mode accepts only the reserved worlds "
            f"{RESERVED_HELDOUT_TEXT}, got {outside[:5]}"
        )
    return worlds


def partition_for(held_out_test: bool) -> str:
    return HELD_OUT_TEST_PARTITION if held_out_test else FITTING_PARTITION


def is_held_out_test(record: Mapping[str, Any] | None) -> bool:
    """Whether a source record, its options or a shard meta is tagged held-out test."""
    if not isinstance(record, Mapping):
        return False
    if record.get("partition") == HELD_OUT_TEST_PARTITION or record.get("held_out_test") is True:
        return True
    options = record.get("options")
    return isinstance(options, Mapping) and (
        options.get("partition") == HELD_OUT_TEST_PARTITION or options.get("held_out_test") is True
    )


def is_calibration_set(record: Mapping[str, Any] | None) -> bool:
    """Whether a source record, its options or a shard meta is tagged calibration-only."""
    if not isinstance(record, Mapping):
        return False
    if record.get("partition") == CALIBRATION_PARTITION:
        return True
    options = record.get("options")
    return isinstance(options, Mapping) and options.get("partition") == CALIBRATION_PARTITION


def require_training_record(record: Mapping[str, Any] | None, context: str) -> None:
    """Refuse a held-out-test or calibration-set tagged dataset/shard for training."""
    if is_calibration_set(record):
        raise ValueError(
            f"{context} refuses calibration-set data (partition {CALIBRATION_PARTITION!r}); "
            "calibration scenario sets never train any network"
        )
    if is_held_out_test(record):
        raise ValueError(
            f"{context} refuses held-out-test data (partition {HELD_OUT_TEST_PARTITION!r}); "
            "test shards from the reserved worlds never train any network"
        )


# Diagnostic posture-override outputs (labor_posture_probe): the probe forces the actor's
# non-labor posture (fission policy, trade share/gate/value) for every camp before
# submission, so its decisions were never sampled by any policy. Its outputs carry this
# scope and every training/anchor/calibration loader refuses them.
DIAGNOSTIC_POSTURE_SCOPE = "diagnostic-posture-override"


def is_diagnostic_posture(record: Any) -> bool:
    """Whether a loaded record (or its ``metadata``/``source``/``options``) carries the scope."""
    if not isinstance(record, Mapping):
        return False
    if record.get("scope") == DIAGNOSTIC_POSTURE_SCOPE:
        return True
    return any(
        isinstance(record.get(key), Mapping)
        and record[key].get("scope") == DIAGNOSTIC_POSTURE_SCOPE
        for key in ("metadata", "source", "options")
    )


def refuse_diagnostic_posture(record: Any, context: str) -> None:
    """Refuse a posture-override probe output wherever training data/inputs are loaded."""
    if is_diagnostic_posture(record):
        raise ValueError(
            f"{context} refuses {DIAGNOSTIC_POSTURE_SCOPE!r} data: posture-probe decisions "
            "were forced, not sampled by a policy, and never train or calibrate anything"
        )
