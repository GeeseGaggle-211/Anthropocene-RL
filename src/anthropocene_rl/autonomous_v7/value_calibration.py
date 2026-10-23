"""Independent observed-return anchors for the opt-in stage-one critic.

The format intentionally separates two kinds of evidence:

* complete native trajectories provide ordinary ``V(s)`` targets through observed termination;
* optional paired-fork witnesses rank actions as ``r + gamma * V(successor)``
  against independently observed branch returns.  They never ask a state-value
  critic to rank alternative actions at the same parent state.

Lambda returns, critic bootstraps and a truncated tail silently set to zero are
rejected by construction.  This is a training/debug gate, not a local
statistical-readiness claim.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contracts
from .model import Normalizer, to_tensors
from .provenance import checkpoint_hash, digest

LEARNER_CONTRACT = "observed-anchor-v1"
SOURCE_KIND = "native-complete-terminal-trajectory-v1"
ESTIMATOR_SOURCE_KIND = "native-randomized-continuation-source-v1"
DATASET_KIND = "independent-observed-return-anchor-v1"
REWARD_SEMANTICS = "ppo-value-head-rewards-v1"
TERMINATION_SEMANTICS = "camp-terminal-after-recorded-transition"
DISCOUNT_SEMANTICS = "exp(-years/horizonYears)"
RANK_SEMANTICS = "same-parent-distinct-successor-r-plus-gamma-v-v1"
ROLES = frozenset(("train-anchor", "actor-gate"))
PER_RESIDENT = frozenset(("welfare", "population_years", "trade_utility"))
GUARD_COST = "guard_cost"
VALUE_HEADS = (*contracts.OBJECTIVES, GUARD_COST)
REFERENCE_SCALES = {
    **contracts.OBJECTIVE_REFERENCE_SCALES,
    GUARD_COST: 1.0 / (1.0 - math.exp(-contracts.TURN_YEARS / contracts.HORIZON_YEARS)),
}


@dataclass(frozen=True)
class RankingWitness:
    successor_inputs: dict[str, np.ndarray]
    immediate_raw: dict[str, np.ndarray]
    realized_raw: dict[str, np.ndarray]
    realized_standard_errors: dict[str, np.ndarray]
    available: dict[str, np.ndarray]
    discount: np.ndarray
    terminal: np.ndarray
    group_ids: np.ndarray
    parent_digests: np.ndarray
    independent_unit_ids: np.ndarray


@dataclass(frozen=True)
class ObservedReturnData:
    path: Path
    role: str
    metadata: dict[str, Any]
    inputs: dict[str, np.ndarray]
    targets_raw: dict[str, np.ndarray]
    target_masks: dict[str, np.ndarray]
    target_standard_errors: dict[str, np.ndarray]
    target_methods: dict[str, np.ndarray]
    row_ids: np.ndarray
    independent_unit_ids: np.ndarray
    ranking: RankingWitness | None

    def __len__(self) -> int:
        return len(self.row_ids)


@dataclass(frozen=True)
class CalibrationThresholds:
    """Declared engineering thresholds; they do not turn a local set into readiness evidence."""

    max_normalized_mae: float = 1.0
    max_normalized_abs_bias: float = 0.5
    max_zero_target_prediction: float = 0.25
    rank_tolerance: float = 0.02
    min_informative_rank_groups: int = 2
    min_equal_group_rank_accuracy: float = 0.6
    min_independent_units: int = 1
    min_living_continuation_units: int = 1
    stopping_randomization_z: float = 1.96

    def __post_init__(self) -> None:
        values = (
            self.max_normalized_mae,
            self.max_normalized_abs_bias,
            self.max_zero_target_prediction,
            self.rank_tolerance,
            self.stopping_randomization_z,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("calibration thresholds must be finite and non-negative")
        if self.min_informative_rank_groups < 0:
            raise ValueError("min_informative_rank_groups must be non-negative")
        if self.min_independent_units <= 0 or self.min_living_continuation_units <= 0:
            raise ValueError("calibration unit thresholds must be positive")
        if not 0 <= self.min_equal_group_rank_accuracy <= 1:
            raise ValueError("rank accuracy threshold must be in [0, 1]")


def normalization_binding(
    normalizer_state: Mapping[str, Any], normalization_contract: Mapping[str, Any]
) -> dict[str, str]:
    """Content binding used by both a producer source and the checkpoint that consumes it."""
    contract_digest = digest(dict(normalization_contract))
    state_digest = digest(dict(normalizer_state))
    return {
        "contractDigest": contract_digest,
        "stateDigest": state_digest,
        "bindingDigest": digest({"contractDigest": contract_digest, "stateDigest": state_digest}),
    }


def policy_digest(actor) -> str:
    """The actor tensor digest, one component of an effective policy binding."""
    from .economic_trace import state_digest

    return state_digest(actor)


def behavior_policy_digest(actor, configuration: Mapping[str, Any]) -> str:
    """Bind tensors plus rollout-time action and personality settings.

    Equal weights with a different exploration floor, enabled action heads, or
    personality sampler are different sampled behaviour policies.
    """
    if not isinstance(configuration, Mapping) or not configuration:
        raise ValueError("effective behaviour policy configuration must be non-empty")
    return digest(
        {
            "semantics": "effective-sampled-policy-digest-v1",
            "actorTensorDigest": policy_digest(actor),
            "configuration": dict(configuration),
        }
    )


def _metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    try:
        raw = arrays["metadata_json"]
    except KeyError as exc:
        raise ValueError("observed-return file lacks metadata_json") from exc
    if np.asarray(raw).shape != ():
        raise ValueError("metadata_json must be one scalar JSON string")
    try:
        value = json.loads(str(np.asarray(raw).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("metadata_json is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("metadata_json must decode to an object")
    return value


def _one_dimensional(values: np.ndarray, name: str, n: int | None = None) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or (n is not None and len(result) != n):
        expected = "one-dimensional" if n is None else f"one-dimensional with {n} rows"
        raise ValueError(f"{name} must be {expected}")
    return result


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value.lower())
    )


def _validate_common_source_metadata(meta: Mapping[str, Any]) -> None:
    from .protocol import scan, seeds

    # These identities are data, not a caller assertion: applying the shared
    # protocol scanner here keeps the sealed confirmation cohort unavailable
    # to both terminal and randomized-tail converters.
    scan(dict(meta), "observed-return source metadata")
    expected = {
        "schemaVersion": 1,
        "nativeContract": contracts.NATIVE_CONTRACT,
        "rewardSemantics": REWARD_SEMANTICS,
        "terminationSemantics": TERMINATION_SEMANTICS,
        "discountSemantics": DISCOUNT_SEMANTICS,
        "turnYears": contracts.TURN_YEARS,
        "horizonYears": contracts.HORIZON_YEARS,
        "policyDigestSemantics": "effective-sampled-policy-digest-v1",
    }
    for key, wanted in expected.items():
        if meta.get(key) != wanted:
            raise ValueError(f"observed-return source {key} {meta.get(key)!r} != {wanted!r}")
    if not _valid_sha256(meta.get("sourceCheckpointSha256")):
        raise ValueError("sourceCheckpointSha256 must be a content SHA-256")
    if not _valid_sha256(meta.get("sourcePolicyDigest")):
        raise ValueError("sourcePolicyDigest must bind the frozen collection policy")
    if not _valid_sha256(meta.get("sourceActorTensorDigest")):
        raise ValueError("sourceActorTensorDigest must bind the frozen actor tensors")
    policy_configuration = meta.get("sourcePolicyConfiguration")
    if not isinstance(policy_configuration, dict) or not policy_configuration:
        raise ValueError("sourcePolicyConfiguration must bind rollout-time policy settings")
    expected_policy_digest = digest(
        {
            "semantics": "effective-sampled-policy-digest-v1",
            "actorTensorDigest": meta["sourceActorTensorDigest"],
            "configuration": policy_configuration,
        }
    )
    if meta["sourcePolicyDigest"] != expected_policy_digest:
        raise ValueError("sourcePolicyDigest differs from actor tensors/policy configuration")
    binding = meta.get("normalizationBinding")
    if not isinstance(binding, dict) or any(
        not _valid_sha256(binding.get(key))
        for key in ("contractDigest", "stateDigest", "bindingDigest")
    ):
        raise ValueError(
            "normalizationBinding must bind the exact input contract and normalizer state"
        )
    if not isinstance(meta.get("simulator"), dict) or not meta["simulator"]:
        raise ValueError("source metadata must bind the native simulator/runtime identity")
    declared: dict[str, list[int]] = {}
    for key in ("worldSeeds", "optimizerSeeds", "continuationSeeds"):
        value = meta.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"source metadata {key} must be a non-empty explicit list")
        checked = seeds(value, f"observed-return source {key}")
        if len(checked) != len(set(checked)):
            raise ValueError(f"source metadata {key} contains duplicates")
        declared[key] = checked
    units = meta.get("independentUnits")
    if not isinstance(units, list) or not units:
        raise ValueError("source metadata must bind independent world/branch units")
    unit_ids: set[str] = set()
    observed = {key: set() for key in declared}
    bound_worlds: set[int] = set()
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("each independent unit binding must be an object")
        unit_id = unit.get("unitId")
        parents = unit.get("parentIdentities")
        branches = unit.get("branchIdentities")
        if not isinstance(unit_id, str) or not unit_id or unit_id in unit_ids:
            raise ValueError("independent unit IDs must be non-empty and unique")
        if (
            not isinstance(parents, list)
            or not parents
            or any(not isinstance(parent, str) or not parent for parent in parents)
            or len(parents) != len(set(parents))
        ):
            raise ValueError("independent unit parentIdentities must be non-empty and unique")
        if (
            not isinstance(branches, list)
            or not branches
            or any(not isinstance(branch, str) or not branch for branch in branches)
            or len(branches) != len(set(branches))
        ):
            raise ValueError("independent unit branchIdentities must be non-empty and unique")
        world = seeds([unit.get("worldSeed")], f"independent unit {unit_id} world")[0]
        if world in bound_worlds:
            raise ValueError("all states/branches from one world must share one independent unit")
        bound_worlds.add(world)
        optimizer = seeds([unit.get("optimizerSeed")], f"independent unit {unit_id} optimizer")[0]
        continuation = unit.get("continuationSeeds")
        if not isinstance(continuation, list) or not continuation:
            raise ValueError("independent unit continuationSeeds must be non-empty")
        continuation = seeds(continuation, f"independent unit {unit_id} continuation")
        unit_ids.add(unit_id)
        observed["worldSeeds"].add(world)
        observed["optimizerSeeds"].add(optimizer)
        observed["continuationSeeds"].update(continuation)
    for key, values in declared.items():
        if observed[key] != set(values):
            raise ValueError(f"source metadata {key} differs from independent-unit bindings")


def _validate_row_unit_bindings(meta: Mapping[str, Any], row_units: np.ndarray) -> None:
    declared = {str(unit["unitId"]) for unit in meta["independentUnits"]}
    observed = set(np.asarray(row_units).astype(str).tolist())
    if observed != declared:
        raise ValueError("independent_unit_id rows differ from source world/branch unit bindings")


def _validate_source_metadata(meta: Mapping[str, Any]) -> None:
    _validate_common_source_metadata(meta)
    expected = {
        "kind": SOURCE_KIND,
        "tailSemantics": "observed-through-terminal",
        "truncated": False,
    }
    for key, wanted in expected.items():
        if meta.get(key) != wanted:
            raise ValueError(f"observed-return source {key} {meta.get(key)!r} != {wanted!r}")


def _validate_estimator_metadata(meta: Mapping[str, Any]) -> None:
    _validate_common_source_metadata(meta)
    expected = {
        "kind": ESTIMATOR_SOURCE_KIND,
        "tailSemantics": "fixed-prefix-plus-geometric-russian-roulette",
        "truncated": False,
        "criticBootstrapUsed": False,
        "continuationPane": "sampled-training-policy",
        "discountedTailTerms": "discounted-from-original-frozen-state",
    }
    for key, wanted in expected.items():
        if meta.get(key) != wanted:
            raise ValueError(f"randomized-return source {key} {meta.get(key)!r} != {wanted!r}")
    schedules = meta.get("schedules")
    if not isinstance(schedules, list) or len(schedules) < 2:
        raise ValueError("randomized-return source needs at least two independent path schedules")


def _source_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def build_terminal_dataset(source: Path, output: Path, role: str) -> dict[str, Any]:
    """Convert complete native terminal trajectories into exact discounted value anchors.

    The source contains flattened model inputs plus ``reward__<head>``, sequence
    identifiers/steps, ``years`` and a terminal bit.  Optional fork ranking needs
    ``rank_group_id``, ``rank_parent_digest`` and ``successor_index``.
    A non-terminal final sequence row is an error, never a zero tail.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    if output.exists():
        raise FileExistsError(output)
    arrays = _source_arrays(source)
    meta = _metadata(arrays)
    _validate_source_metadata(meta)
    sequence = _one_dimensional(arrays.get("sequence_id"), "sequence_id")
    n = len(sequence)
    if n == 0:
        raise ValueError("observed-return source is empty")
    step = _one_dimensional(arrays.get("sequence_step"), "sequence_step", n).astype(np.int64)
    years = _one_dimensional(arrays.get("years"), "years", n).astype(np.float64)
    terminal = _one_dimensional(arrays.get("terminated"), "terminated", n).astype(bool)
    if not np.isfinite(years).all() or bool((years <= 0).any()):
        raise ValueError("years must be finite and positive")
    row_ids = _one_dimensional(arrays.get("row_id"), "row_id", n).astype(str)
    if len(set(row_ids.tolist())) != n:
        raise ValueError("row_id values must be unique")
    unit_ids = _one_dimensional(arrays.get("independent_unit_id"), "independent_unit_id", n).astype(
        str
    )
    if any(not value for value in unit_ids):
        raise ValueError("independent_unit_id values must be non-empty")
    _validate_row_unit_bindings(meta, unit_ids)
    reward = {}
    for name in VALUE_HEADS:
        values = _one_dimensional(arrays.get(f"reward__{name}"), f"reward__{name}", n).astype(
            np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError(f"reward__{name} contains non-finite values")
        reward[name] = values
    targets = {name: np.zeros(n, np.float64) for name in VALUE_HEADS}
    successor_expected = np.full(n, -1, np.int64)
    for sequence_id in np.unique(sequence):
        rows = np.flatnonzero(sequence == sequence_id)
        order = rows[np.argsort(step[rows], kind="stable")]
        if len(np.unique(step[order])) != len(order):
            raise ValueError(f"sequence {sequence_id!r} repeats a sequence_step")
        if bool(terminal[order[:-1]].any()) or not bool(terminal[order[-1]]):
            raise ValueError(
                f"sequence {sequence_id!r} is truncated or continues after termination"
            )
        successor_expected[order[:-1]] = order[1:]
        for name in VALUE_HEADS:
            continuation = 0.0
            for row in order[::-1]:
                continuation = (
                    float(reward[name][row])
                    + math.exp(-float(years[row]) / float(meta["horizonYears"])) * continuation
                )
                targets[name][row] = continuation
    aux = {
        "metadata_json",
        "sequence_id",
        "sequence_step",
        "row_id",
        "independent_unit_id",
        "terminated",
        "successor_index",
        "rank_group_id",
        "rank_parent_digest",
        *(f"reward__{name}" for name in VALUE_HEADS),
    }
    input_arrays = {key: np.asarray(value) for key, value in arrays.items() if key not in aux}
    for key, value in input_arrays.items():
        if value.ndim == 0 or len(value) != n:
            raise ValueError(f"model input {key!r} does not have {n} rows")
    if "prior_population" not in input_arrays:
        raise ValueError("model inputs need prior_population for value-head units")
    out: dict[str, np.ndarray] = {f"input__{key}": value for key, value in input_arrays.items()}
    out.update({f"target__{name}": value for name, value in targets.items()})
    for name in VALUE_HEADS:
        out[f"target_mask__{name}"] = np.ones(n, bool)
        out[f"target_standard_error__{name}"] = np.zeros(n, np.float64)
        out[f"target_method__{name}"] = np.full(n, "terminal-exact")
    out.update({"row_id": row_ids, "independent_unit_id": unit_ids})
    rank_count = 0
    if "rank_group_id" in arrays:
        groups = _one_dimensional(arrays["rank_group_id"], "rank_group_id", n).astype(str)
        selected = np.flatnonzero(np.char.str_len(groups) > 0)
        if len(selected):
            if "successor_index" not in arrays or "rank_parent_digest" not in arrays:
                raise ValueError("ranking witnesses need successor_index and rank_parent_digest")
            successors = _one_dimensional(arrays["successor_index"], "successor_index", n).astype(
                np.int64
            )
            parents = _one_dimensional(
                arrays["rank_parent_digest"], "rank_parent_digest", n
            ).astype(str)
            if not np.array_equal(successors, successor_expected):
                raise ValueError(
                    "successor_index must be the next row of the same complete "
                    "trajectory, or -1 at terminal"
                )
            for group in np.unique(groups[selected]):
                members = selected[groups[selected] == group]
                if (
                    len(members) < 2
                    or len(set(parents[members].tolist())) != 1
                    or not parents[members[0]]
                ):
                    raise ValueError(
                        "each rank group needs at least two alternatives and one "
                        "shared parent digest"
                    )
            successor_rows = np.where(successors[selected] >= 0, successors[selected], selected)
            for key, value in input_arrays.items():
                out[f"rank_input__{key}"] = value[successor_rows]
            out.update(
                {
                    "rank_terminal": terminal[selected],
                    "rank_discount": np.exp(-years[selected] / float(meta["horizonYears"])),
                    "rank_group_id": groups[selected],
                    "rank_parent_digest": parents[selected],
                    "rank_independent_unit_id": unit_ids[selected],
                }
            )
            for name in VALUE_HEADS:
                out[f"rank_immediate__{name}"] = reward[name][selected]
                out[f"rank_realized__{name}"] = targets[name][selected]
                out[f"rank_standard_error__{name}"] = np.zeros(len(selected), np.float64)
                out[f"rank_mask__{name}"] = np.ones(len(selected), bool)
            rank_count = len(selected)
    dataset_meta = {
        "schemaVersion": 1,
        "kind": DATASET_KIND,
        "learnerContract": LEARNER_CONTRACT,
        "role": role,
        "valueHeads": list(VALUE_HEADS),
        "targetKind": "independent-observed-full-return",
        "rankSemantics": RANK_SEMANTICS if rank_count else "absent",
        "sourcePath": str(source.resolve()),
        "sourceSha256": checkpoint_hash(source),
        "source": meta,
        "rowCount": n,
        "rankRowCount": rank_count,
        "readinessScope": "training-debug-only-no-local-readiness-claim",
    }
    out["metadata_json"] = np.asarray(
        json.dumps(dataset_meta, sort_keys=True, separators=(",", ":"))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out)
    return {
        **dataset_meta,
        "outputPath": str(output.resolve()),
        "outputSha256": checkpoint_hash(output),
    }


def build_randomized_estimator_dataset(source: Path, output: Path, role: str) -> dict[str, Any]:
    """Build living-state anchors with the reviewed independent randomized-tail machinery.

    Each source row is a frozen state.  ``path_prefix__<head>`` has shape
    ``[state, independent_path]``.  A synthetic/source adapter may supply discounted
    tail terms, while the native fork adapter supplies the actual predeclared
    stopping-draw estimates plus their statuses.  In both cases schedules are frozen
    before outcomes and every stochastic path keeps its own prefix.  Heads without a
    source-backed bound remain unavailable unless every path is an exact observed
    terminal suffix.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    if output.exists():
        raise FileExistsError(output)
    arrays = _source_arrays(source)
    meta = _metadata(arrays)
    _validate_estimator_metadata(meta)
    row_ids = _one_dimensional(arrays.get("row_id"), "row_id").astype(str)
    n = len(row_ids)
    if n == 0 or len(set(row_ids.tolist())) != n:
        raise ValueError("randomized-return source needs unique nonempty row_id values")
    unit_ids = _one_dimensional(arrays.get("independent_unit_id"), "independent_unit_id", n).astype(
        str
    )
    if any(not value for value in unit_ids):
        raise ValueError("independent_unit_id values must be non-empty")
    _validate_row_unit_bindings(meta, unit_ids)
    living = _one_dimensional(
        arrays.get("survived_fixed_prefix"), "survived_fixed_prefix", n
    ).astype(bool)
    if not bool(living.any()):
        raise ValueError(
            "randomized-return source must include living states, not only eventual extinctions"
        )

    from .gate34_continuation_collector import (
        FrozenGeometricSchedule,
        bounded_geometric_tail_assumptions,
        combine_independent_path_returns,
        estimate_randomized_tail,
    )

    schedules = tuple(
        FrozenGeometricSchedule(
            schedule_id=str(item["scheduleId"]),
            survival_probability=float(item["survivalProbability"]),
            stop_indices=tuple(int(value) for value in item["stopIndices"]),
            hard_cap_terms=int(item["hardCapTerms"]),
            rng_seed=None if item.get("rngSeed") is None else int(item["rngSeed"]),
            generator=str(item.get("generator", "predeclared-explicit")),
            frozen_before_collection=bool(item.get("frozenBeforeCollection", False)),
        )
        for item in meta["schedules"]
    )
    if len({schedule.schedule_id for schedule in schedules}) != len(schedules):
        raise ValueError("independent paths need distinct frozen stopping schedules")
    gamma = math.exp(-float(meta["turnYears"]) / float(meta["horizonYears"]))
    bounds = meta.get("tailAbsoluteBounds")
    if not isinstance(bounds, dict):
        raise ValueError("randomized-return source must declare source-backed per-head tail bounds")
    aux_prefixes = ("path_", "rank_")
    aux = {
        "metadata_json",
        "row_id",
        "independent_unit_id",
        "survived_fixed_prefix",
        "rank_group_id",
        "rank_parent_digest",
        "rank_successor_digest",
        "rank_discount",
        "rank_terminal",
        *(f"rank_immediate__{name}" for name in VALUE_HEADS),
    }
    input_arrays = {
        key: np.asarray(value)
        for key, value in arrays.items()
        if key not in aux and not key.startswith(aux_prefixes)
    }
    if "prior_population" not in input_arrays:
        raise ValueError("model inputs need prior_population for value-head units")
    for key, value in input_arrays.items():
        if value.ndim == 0 or len(value) != n:
            raise ValueError(f"model input {key!r} does not have {n} rows")
    out: dict[str, np.ndarray] = {f"input__{key}": value for key, value in input_arrays.items()}
    unavailable = []
    for name in VALUE_HEADS:
        targets = np.zeros(n, np.float64)
        errors = np.zeros(n, np.float64)
        mask = np.zeros(n, bool)
        methods = np.full(n, "unavailable-no-source-backed-tail-bound")
        proof = bounds.get(name)
        draw_key = f"path_tail_draws__{name}"
        if draw_key in arrays:
            prefixes = np.asarray(arrays.get(f"path_prefix__{name}"), dtype=np.float64)
            draws = np.asarray(arrays[draw_key], dtype=np.float64)
            statuses = np.asarray(arrays.get(f"path_tail_status__{name}")).astype(str)
            exact = np.asarray(arrays.get(f"path_tail_exact_terminal__{name}")).astype(bool)
            schedule_ids = np.asarray(arrays.get("path_schedule_id")).astype(str)
            if (
                prefixes.ndim != 2
                or prefixes.shape[0] != n
                or draws.ndim != 3
                or draws.shape[:2] != prefixes.shape
                or draws.shape[2] < 2
                or statuses.shape != prefixes.shape
                or exact.shape != prefixes.shape
                or schedule_ids.shape != prefixes.shape
            ):
                raise ValueError(f"native stopping-draw arrays for {name} have the wrong shape")
            if not np.isfinite(prefixes).all() or not np.isfinite(draws).all():
                raise ValueError(
                    f"native stopping-draw arrays for {name} contain non-finite values"
                )
            for row in range(n):
                tails = []
                row_available = True
                for path in range(prefixes.shape[1]):
                    status = statuses[row, path]
                    if status != "identified":
                        row_available = False
                        break
                    if proof is None and not exact[row, path]:
                        row_available = False
                        break
                    values = draws[row, path]
                    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
                    from .gate34_continuation_collector import RandomizedTailEstimate

                    tails.append(
                        RandomizedTailEstimate(
                            status="identified",
                            schedule_id=schedule_ids[row, path],
                            estimate=float(values.mean()),
                            standard_error=standard_error,
                            draw_estimates=tuple(float(value) for value in values),
                            standard_error_scope=(
                                "exact observed finite terminal suffix"
                                if exact[row, path]
                                else "stopping-randomization Monte Carlo conditional on one "
                                "observed continuation path"
                            ),
                            stopping_randomization_contribution_to_standard_error=(
                                0.0 if exact[row, path] else standard_error
                            ),
                        )
                    )
                if not row_available:
                    continue
                provenance = (
                    str(proof["provenanceId"])
                    if isinstance(proof, dict) and proof.get("provenanceId")
                    else "exact-terminal"
                )
                combined = combine_independent_path_returns(
                    prefixes[row], tails, f"{provenance}:{row_ids[row]}"
                )
                if not combined.identified:
                    continue
                assert combined.estimate is not None and combined.standard_error is not None
                targets[row] = combined.estimate
                errors[row] = combined.standard_error
                mask[row] = True
                methods[row] = "independent-randomized-tail-path-mean"
        elif proof is not None:
            if not isinstance(proof, dict) or not proof.get("provenanceId"):
                raise ValueError(f"tail bound for {name} needs a provenanceId")
            prefixes = np.asarray(arrays.get(f"path_prefix__{name}"), dtype=np.float64)
            terms = np.asarray(arrays.get(f"path_discounted_tail_terms__{name}"), dtype=np.float64)
            expected_prefix = (n, len(schedules))
            if (
                prefixes.shape != expected_prefix
                or terms.ndim != 3
                or terms.shape[:2] != expected_prefix
            ):
                raise ValueError(f"randomized path arrays for {name} have the wrong shape")
            if not np.isfinite(prefixes).all() or not np.isfinite(terms).all():
                raise ValueError(f"randomized path arrays for {name} contain non-finite values")
            for row in range(n):
                tails = []
                for path, schedule in enumerate(schedules):
                    assumptions = bounded_geometric_tail_assumptions(
                        per_turn_absolute_bound=float(proof["perTurnAbsoluteBound"]),
                        discount=gamma,
                        survival_probability=schedule.survival_probability,
                        provenance_id=str(proof["provenanceId"]),
                    )
                    tails.append(estimate_randomized_tail(terms[row, path], schedule, assumptions))
                combined = combine_independent_path_returns(
                    prefixes[row], tails, f"{proof['provenanceId']}:{row_ids[row]}"
                )
                if not combined.identified:
                    raise ValueError(
                        f"predeclared randomized tail for {name}/{row_ids[row]} is unresolved"
                    )
                assert combined.estimate is not None and combined.standard_error is not None
                targets[row] = combined.estimate
                errors[row] = combined.standard_error
                mask[row] = True
                methods[row] = "independent-randomized-tail-path-mean"
        if not bool(mask.any()):
            unavailable.append(name)
        out[f"target__{name}"] = targets
        out[f"target_mask__{name}"] = mask
        out[f"target_standard_error__{name}"] = errors
        out[f"target_method__{name}"] = methods
    out.update({"row_id": row_ids, "independent_unit_id": unit_ids})
    rank_count = 0
    rank_semantics = "absent"
    if "rank_group_id" in arrays:
        if meta.get("rankWitnessSemantics") != RANK_SEMANTICS:
            raise ValueError(
                "randomized source ranking metadata does not declare successor-action semantics"
            )
        explicit_rank_inputs = any(key.startswith("rank_input__") for key in arrays)
        groups = _one_dimensional(arrays["rank_group_id"], "rank_group_id").astype(str)
        if explicit_rank_inputs:
            selected = np.arange(len(groups))
            rank_n = len(groups)
        else:
            if len(groups) != n:
                raise ValueError("row-aligned rank_group_id has the wrong length")
            selected = np.flatnonzero(np.char.str_len(groups) > 0)
            rank_n = n
        if len(selected):
            parents = _one_dimensional(
                arrays.get("rank_parent_digest"), "rank_parent_digest", rank_n
            ).astype(str)
            successors = _one_dimensional(
                arrays.get("rank_successor_digest"), "rank_successor_digest", rank_n
            ).astype(str)
            discounts = _one_dimensional(
                arrays.get("rank_discount"), "rank_discount", rank_n
            ).astype(np.float64)
            terminals = _one_dimensional(
                arrays.get("rank_terminal"), "rank_terminal", rank_n
            ).astype(bool)
            if (
                not np.isfinite(discounts[selected]).all()
                or bool((discounts[selected] < 0).any())
                or bool((discounts[selected] > 1).any())
            ):
                raise ValueError("rank discounts must be finite probabilities")
            for group in np.unique(groups[selected]):
                members = selected[groups[selected] == group]
                if (
                    len(members) < 2
                    or len(set(parents[members].tolist())) != 1
                    or not parents[members[0]]
                    or len(set(successors[members].tolist())) != len(members)
                    or any(not value for value in successors[members])
                ):
                    raise ValueError(
                        "rank group needs one parent and distinct bound successor states"
                    )
            if explicit_rank_inputs:
                rank_inputs = {
                    key[len("rank_input__") :]: np.asarray(value)
                    for key, value in arrays.items()
                    if key.startswith("rank_input__")
                }
                if set(rank_inputs) != set(input_arrays) or any(
                    value.ndim == 0 or len(value) != rank_n for value in rank_inputs.values()
                ):
                    raise ValueError(
                        "explicit ranking inputs differ from parent model input schema"
                    )
                for key, value in rank_inputs.items():
                    out[f"rank_input__{key}"] = value[selected]
                rank_units = _one_dimensional(
                    arrays.get("rank_independent_unit_id"),
                    "rank_independent_unit_id",
                    rank_n,
                ).astype(str)
                if not set(rank_units).issubset(set(unit_ids)):
                    raise ValueError("ranking witnesses reference undeclared independent units")
            else:
                for key, value in input_arrays.items():
                    out[f"rank_input__{key}"] = value[selected]
                rank_units = unit_ids
            out.update(
                {
                    "rank_terminal": terminals[selected],
                    "rank_discount": discounts[selected],
                    "rank_group_id": groups[selected],
                    "rank_parent_digest": parents[selected],
                    "rank_independent_unit_id": rank_units[selected],
                }
            )
            for name in VALUE_HEADS:
                immediate = _one_dimensional(
                    arrays.get(f"rank_immediate__{name}"),
                    f"rank_immediate__{name}",
                    rank_n,
                ).astype(np.float64)[selected]
                out[f"rank_immediate__{name}"] = immediate
                if explicit_rank_inputs:
                    out[f"rank_realized__{name}"] = _one_dimensional(
                        arrays.get(f"rank_realized__{name}"),
                        f"rank_realized__{name}",
                        rank_n,
                    ).astype(np.float64)[selected]
                    out[f"rank_standard_error__{name}"] = _one_dimensional(
                        arrays.get(f"rank_standard_error__{name}"),
                        f"rank_standard_error__{name}",
                        rank_n,
                    ).astype(np.float64)[selected]
                    out[f"rank_mask__{name}"] = _one_dimensional(
                        arrays.get(f"rank_mask__{name}"),
                        f"rank_mask__{name}",
                        rank_n,
                    ).astype(bool)[selected]
                else:
                    available = out[f"target_mask__{name}"][selected]
                    out[f"rank_realized__{name}"] = (
                        immediate + discounts[selected] * out[f"target__{name}"][selected]
                    )
                    out[f"rank_standard_error__{name}"] = (
                        discounts[selected] * out[f"target_standard_error__{name}"][selected]
                    )
                    out[f"rank_mask__{name}"] = available
            rank_count = len(selected)
            rank_semantics = RANK_SEMANTICS
    dataset_meta = {
        "schemaVersion": 1,
        "kind": DATASET_KIND,
        "learnerContract": LEARNER_CONTRACT,
        "role": role,
        "valueHeads": list(VALUE_HEADS),
        "targetKind": "independent-observed-full-return-estimate",
        "rankSemantics": rank_semantics,
        "sourcePath": str(source.resolve()),
        "sourceSha256": checkpoint_hash(source),
        "source": meta,
        "rowCount": n,
        "rankRowCount": rank_count,
        "unavailableHeads": unavailable,
        "readinessScope": "training-debug-only-no-local-readiness-claim",
    }
    out["metadata_json"] = np.asarray(
        json.dumps(dataset_meta, sort_keys=True, separators=(",", ":"))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out)
    return {
        **dataset_meta,
        "outputPath": str(output.resolve()),
        "outputSha256": checkpoint_hash(output),
    }


def load_dataset(
    path: Path, role: str, *, expected_binding: Mapping[str, str] | None = None
) -> ObservedReturnData:
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    arrays = _source_arrays(path)
    meta = _metadata(arrays)
    expected = {
        "schemaVersion": 1,
        "kind": DATASET_KIND,
        "learnerContract": LEARNER_CONTRACT,
        "role": role,
        "valueHeads": list(VALUE_HEADS),
    }
    for key, wanted in expected.items():
        if meta.get(key) != wanted:
            raise ValueError(f"anchor {key} {meta.get(key)!r} != {wanted!r}")
    source = meta.get("source")
    if not isinstance(source, dict):
        raise ValueError("anchor metadata lacks its validated source metadata")
    if source.get("kind") == SOURCE_KIND:
        _validate_source_metadata(source)
    elif source.get("kind") == ESTIMATOR_SOURCE_KIND:
        _validate_estimator_metadata(source)
    else:
        raise ValueError("anchor source kind is not an approved independent-return producer")
    if meta.get("targetKind") not in {
        "independent-observed-full-return",
        "independent-observed-full-return-estimate",
    }:
        raise ValueError("anchor target kind is not an approved independent return")
    if expected_binding is not None and source["normalizationBinding"] != dict(expected_binding):
        raise ValueError("anchor input-normalization binding differs from the consuming checkpoint")
    row_ids = _one_dimensional(arrays.get("row_id"), "row_id")
    n = len(row_ids)
    units = _one_dimensional(arrays.get("independent_unit_id"), "independent_unit_id", n).astype(
        str
    )
    _validate_row_unit_bindings(source, units)
    inputs = {
        key[len("input__") :]: np.asarray(value)
        for key, value in arrays.items()
        if key.startswith("input__")
    }
    if not inputs or any(value.ndim == 0 or len(value) != n for value in inputs.values()):
        raise ValueError("anchor model inputs are absent or row-misaligned")
    targets, masks, errors, methods = {}, {}, {}, {}
    for name in VALUE_HEADS:
        target = _one_dimensional(arrays.get(f"target__{name}"), f"target__{name}", n).astype(
            np.float64
        )
        if not np.isfinite(target).all():
            raise ValueError(f"anchor target {name} contains non-finite values")
        targets[name] = target
        masks[name] = _one_dimensional(
            arrays.get(f"target_mask__{name}"), f"target_mask__{name}", n
        ).astype(bool)
        errors[name] = _one_dimensional(
            arrays.get(f"target_standard_error__{name}"), f"target_standard_error__{name}", n
        ).astype(np.float64)
        methods[name] = _one_dimensional(
            arrays.get(f"target_method__{name}"), f"target_method__{name}", n
        ).astype(str)
        if not np.isfinite(errors[name]).all() or bool((errors[name] < 0).any()):
            raise ValueError(f"anchor standard error {name} is invalid")
    ranking = None
    if meta.get("rankSemantics") != "absent":
        if meta.get("rankSemantics") != RANK_SEMANTICS:
            raise ValueError("unknown ranking witness semantics")
        groups = _one_dimensional(arrays.get("rank_group_id"), "rank_group_id")
        r = len(groups)
        rank_inputs = {
            key[len("rank_input__") :]: np.asarray(value)
            for key, value in arrays.items()
            if key.startswith("rank_input__")
        }
        if not rank_inputs or any(
            value.ndim == 0 or len(value) != r for value in rank_inputs.values()
        ):
            raise ValueError("ranking successor inputs are absent or row-misaligned")
        immediate, realized, realized_errors, available = {}, {}, {}, {}
        for name in VALUE_HEADS:
            immediate[name] = _one_dimensional(
                arrays.get(f"rank_immediate__{name}"), f"rank_immediate__{name}", r
            ).astype(np.float64)
            realized[name] = _one_dimensional(
                arrays.get(f"rank_realized__{name}"), f"rank_realized__{name}", r
            ).astype(np.float64)
            realized_errors[name] = _one_dimensional(
                arrays.get(f"rank_standard_error__{name}"),
                f"rank_standard_error__{name}",
                r,
            ).astype(np.float64)
            if not np.isfinite(realized_errors[name]).all() or bool(
                (realized_errors[name] < 0).any()
            ):
                raise ValueError(f"ranking standard error {name} is invalid")
            available[name] = _one_dimensional(
                arrays.get(f"rank_mask__{name}"), f"rank_mask__{name}", r
            ).astype(bool)
        ranking = RankingWitness(
            successor_inputs=rank_inputs,
            immediate_raw=immediate,
            realized_raw=realized,
            realized_standard_errors=realized_errors,
            available=available,
            discount=_one_dimensional(arrays.get("rank_discount"), "rank_discount", r).astype(
                np.float64
            ),
            terminal=_one_dimensional(arrays.get("rank_terminal"), "rank_terminal", r).astype(bool),
            group_ids=groups.astype(str),
            parent_digests=_one_dimensional(
                arrays.get("rank_parent_digest"), "rank_parent_digest", r
            ).astype(str),
            independent_unit_ids=_one_dimensional(
                arrays.get("rank_independent_unit_id"), "rank_independent_unit_id", r
            ).astype(str),
        )
    return ObservedReturnData(
        path.resolve(),
        role,
        meta,
        inputs,
        targets,
        masks,
        errors,
        methods,
        row_ids.astype(str),
        units,
        ranking,
    )


def validate_disjoint(train: ObservedReturnData, gate: ObservedReturnData) -> None:
    overlap = set(train.independent_unit_ids.tolist()) & set(gate.independent_unit_ids.tolist())
    if overlap:
        raise ValueError(
            f"train-anchor and actor-gate share independent units: {sorted(overlap)[:5]}"
        )


def write_refresh_request(
    path: Path,
    checkpoint: Path,
    actor,
    policy_configuration: Mapping[str, Any],
    binding: Mapping[str, str],
    enabled_heads: tuple[str, ...],
    reason: str,
) -> dict[str, Any]:
    """Persist the bounded handoff required before a stale/closed actor gate can resume.

    This is deliberately a request, not an invented independent target.  The
    existing fork/continuation collector must collect sampled-policy paths from
    the frozen checkpoint and the producer in this module must validate them.
    """
    record = {
        "schemaVersion": 1,
        "kind": "observed-anchor-refresh-request-v1",
        "learnerContract": LEARNER_CONTRACT,
        "reason": reason,
        "frozenCheckpoint": str(checkpoint.resolve()),
        "frozenCheckpointSha256": checkpoint_hash(checkpoint),
        "sourcePolicyDigest": behavior_policy_digest(actor, policy_configuration),
        "sourceActorTensorDigest": policy_digest(actor),
        "sourcePolicyConfiguration": dict(policy_configuration),
        "normalizationBinding": dict(binding),
        "requiredValueHeads": list(enabled_heads),
        "requiredRoles": ["train-anchor", "actor-gate"],
        "collectorSemantics": {
            "pane": "sampled-training-policy",
            "target": "fixed-prefix-plus-independent-geometric-randomized-tail-or-exact-terminal",
            "ranking": RANK_SEMANTICS,
            "criticBootstrap": False,
            "truncatedTailAsZero": False,
            "independentUnit": "world",
        },
        "producerCommand": (
            "python -m anthropocene_rl.autonomous_v7.value_calibration "
            "--source <collector-source.npz> --out <anchor.npz> --role <train-anchor|actor-gate>"
        ),
        "status": "collection-required-no-readiness-claim",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return record


def head_scale(name: str, prior_population: np.ndarray) -> np.ndarray:
    population = np.asarray(prior_population, dtype=np.float64)
    return (
        np.maximum(1.0, population)
        if name in PER_RESIDENT
        else np.ones(len(population), np.float64)
    )


def anchor_loss(
    critic, data: ObservedReturnData, normalizer: Normalizer, rows: np.ndarray
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    selected = np.asarray(rows, dtype=np.int64)
    tensors = to_tensors({key: value[selected] for key, value in data.inputs.items()}, normalizer)
    logits = critic(tensors)
    if not isinstance(logits, dict) or set(logits) != set(contracts.OBJECTIVES):
        raise ValueError("observed-anchor-v1 requires one vector-critic head per objective")
    scale = {
        name: head_scale(name, data.inputs["prior_population"][selected])
        for name in contracts.OBJECTIVES
    }
    losses = {}
    for name in contracts.OBJECTIVES:
        available = torch.as_tensor(data.target_masks[name][selected], dtype=torch.bool)
        if bool(available.any()):
            target = torch.as_tensor(
                data.targets_raw[name][selected] / scale[name], dtype=torch.float32
            )
            losses[name] = (
                -(
                    critic.two_hot(target[available])
                    * torch.log_softmax(logits[name][available], dim=-1)
                )
                .sum(-1)
                .mean()
            )
    if not losses:
        raise ValueError("anchor minibatch has no independently identified target")
    return sum(losses.values()), losses


def cost_anchor_loss(
    cost_critic, data: ObservedReturnData, normalizer: Normalizer, rows: np.ndarray
) -> torch.Tensor:
    selected = np.asarray(rows, dtype=np.int64)
    available_np = data.target_masks[GUARD_COST][selected]
    if not bool(available_np.any()):
        raise ValueError("anchor minibatch has no independently identified guard-cost target")
    tensors = to_tensors({key: value[selected] for key, value in data.inputs.items()}, normalizer)
    available = torch.as_tensor(available_np, dtype=torch.bool)
    target = torch.as_tensor(data.targets_raw[GUARD_COST][selected], dtype=torch.float32)
    return ((cost_critic(tensors)[available] - target[available]) ** 2).mean()


def _predict_raw(
    critic, inputs: dict[str, np.ndarray], normalizer: Normalizer, batch_size: int = 512
) -> dict[str, np.ndarray]:
    n = len(inputs["prior_population"])
    parts = {name: [] for name in contracts.OBJECTIVES}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            logits = critic(
                to_tensors({key: value[start:stop] for key, value in inputs.items()}, normalizer)
            )
            if not isinstance(logits, dict) or set(logits) != set(contracts.OBJECTIVES):
                raise ValueError("observed-anchor-v1 requires one vector-critic head per objective")
            for name in contracts.OBJECTIVES:
                raw = critic.expectation(logits[name]).detach().cpu().numpy().astype(np.float64)
                raw *= head_scale(name, inputs["prior_population"][start:stop])
                parts[name].append(raw)
    return {name: np.concatenate(values) for name, values in parts.items()}


def _predict_cost(
    cost_critic, inputs: dict[str, np.ndarray], normalizer: Normalizer, batch_size: int = 512
) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, len(inputs["prior_population"]), batch_size):
            stop = min(len(inputs["prior_population"]), start + batch_size)
            value = cost_critic(
                to_tensors({key: array[start:stop] for key, array in inputs.items()}, normalizer)
            )
            parts.append(value.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(parts)


def _relation(value: float, tolerance: float) -> int:
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def calibration_report(
    critic,
    cost_critic,
    data: ObservedReturnData,
    normalizer: Normalizer,
    thresholds: CalibrationThresholds | None = None,
    enabled_heads: tuple[str, ...] = VALUE_HEADS,
    current_policy_digest: str | None = None,
) -> dict[str, Any]:
    """Held-out magnitude/bias and successor-action ranking by equal rank group."""
    thresholds = CalibrationThresholds() if thresholds is None else thresholds
    predicted = _predict_raw(critic, data.inputs, normalizer)
    predicted[GUARD_COST] = _predict_cost(cost_critic, data.inputs, normalizer)
    rank_scores = rank_realized = None
    if data.ranking is not None:
        successor = _predict_raw(critic, data.ranking.successor_inputs, normalizer)
        successor[GUARD_COST] = _predict_cost(
            cost_critic, data.ranking.successor_inputs, normalizer
        )
        rank_scores = {
            name: data.ranking.immediate_raw[name]
            + data.ranking.discount * np.where(data.ranking.terminal, 0.0, successor[name])
            for name in VALUE_HEADS
        }
        rank_realized = data.ranking.realized_raw
    heads = {}
    for name in VALUE_HEADS:
        scale = float(REFERENCE_SCALES[name])
        selected = data.target_masks[name]
        if not bool(selected.any()):
            heads[name] = {
                "rows": 0,
                "requiredForActor": name in enabled_heads,
                "pass": False,
                "status": "no-independently-identified-targets",
            }
            continue
        error = predicted[name][selected] - data.targets_raw[name][selected]
        selected_targets = data.targets_raw[name][selected]
        selected_standard_errors = data.target_standard_errors[name][selected]
        mean_standard_error = float(
            np.sqrt(np.square(selected_standard_errors).sum()) / len(selected_standard_errors)
        )
        conservative_mae = float(
            np.mean(np.abs(error) + thresholds.stopping_randomization_z * selected_standard_errors)
        )
        conservative_abs_bias = float(
            abs(np.mean(error)) + thresholds.stopping_randomization_z * mean_standard_error
        )
        target_near_zero = bool(
            np.max(np.abs(selected_targets)) <= thresholds.rank_tolerance * scale
        )
        magnitude_pass = bool(
            conservative_mae / scale <= thresholds.max_normalized_mae
            and conservative_abs_bias / scale <= thresholds.max_normalized_abs_bias
        )
        zero_pass = (
            not target_near_zero
            or float(np.mean(np.abs(predicted[name][selected])) / scale)
            <= thresholds.max_zero_target_prediction
        )
        units = set(data.independent_unit_ids[selected].tolist())
        living_mask = selected & (
            data.target_methods[name] == "independent-randomized-tail-path-mean"
        )
        living_units = set(data.independent_unit_ids[living_mask].tolist())
        coverage_pass = bool(
            len(units) >= thresholds.min_independent_units
            and len(living_units) >= thresholds.min_living_continuation_units
        )
        rank_record: dict[str, Any] = {
            "semantics": RANK_SEMANTICS,
            "available": data.ranking is not None,
            "required": not target_near_zero,
            "status": "not-required-near-zero-target" if target_near_zero else "missing",
            "informativeGroups": 0,
            "equalIndependentGroupAccuracy": None,
            "targetTies": 0,
            "predictedTies": 0,
            "uncertainTargetPairs": 0,
        }
        rank_pass = target_near_zero
        if data.ranking is not None and rank_scores is not None and rank_realized is not None:
            accuracies = []
            target_ties = predicted_ties = uncertain_pairs = pairs = 0
            for group in np.unique(data.ranking.group_ids):
                rows = np.flatnonzero(data.ranking.group_ids == group)
                rows = rows[data.ranking.available[name][rows]]
                correct = informative = 0
                for offset, left in enumerate(rows):
                    for right in rows[offset + 1 :]:
                        target_difference = float(
                            rank_realized[name][left] - rank_realized[name][right]
                        )
                        target_uncertainty = thresholds.stopping_randomization_z * float(
                            data.ranking.realized_standard_errors[name][left]
                            + data.ranking.realized_standard_errors[name][right]
                        )
                        if target_uncertainty and abs(target_difference) <= (
                            thresholds.rank_tolerance * scale + target_uncertainty
                        ):
                            target_relation = None
                            uncertain_pairs += 1
                        else:
                            target_relation = _relation(
                                target_difference, thresholds.rank_tolerance * scale
                            )
                        predicted_relation = _relation(
                            float(rank_scores[name][left] - rank_scores[name][right]),
                            thresholds.rank_tolerance * scale,
                        )
                        pairs += 1
                        target_ties += int(target_relation == 0)
                        predicted_ties += int(predicted_relation == 0)
                        if target_relation not in (None, 0):
                            informative += 1
                            correct += int(target_relation == predicted_relation)
                if informative:
                    accuracies.append(correct / informative)
            rank_record.update(
                {
                    "status": (
                        "informative"
                        if accuracies
                        else "uncertain-target-pairs"
                        if uncertain_pairs
                        else "all-realized-pairs-tied"
                        if pairs
                        else "no-available-pairs"
                    ),
                    "pairCount": pairs,
                    "informativeGroups": len(accuracies),
                    "equalIndependentGroupAccuracy": float(np.mean(accuracies))
                    if accuracies
                    else None,
                    "targetTies": target_ties,
                    "predictedTies": predicted_ties,
                    "uncertainTargetPairs": uncertain_pairs,
                }
            )
            if accuracies:
                rank_pass = bool(
                    len(accuracies) >= thresholds.min_informative_rank_groups
                    and np.mean(accuracies) >= thresholds.min_equal_group_rank_accuracy
                )
        heads[name] = {
            "rows": int(selected.sum()),
            "independentUnits": len(units),
            "livingContinuationUnits": len(living_units),
            "coveragePass": coverage_pass,
            "requiredForActor": name in enabled_heads,
            "normalizedMeanAbsoluteError": float(np.mean(np.abs(error)) / scale),
            "normalizedBias": float(np.mean(error) / scale),
            "normalizedTargetEstimateStandardErrorMean": float(
                np.mean(selected_standard_errors) / scale
            ),
            "normalizedConservativeMeanAbsoluteError": conservative_mae / scale,
            "normalizedConservativeAbsoluteBias": conservative_abs_bias / scale,
            "targetUncertaintyScope": (
                "stopping-randomization/path-mean standard error supplied by the "
                "independent producer; excludes finite-world/policy uncertainty"
            ),
            "normalizedMeanAbsolutePrediction": float(
                np.mean(np.abs(predicted[name][selected])) / scale
            ),
            "targetNearZero": target_near_zero,
            "magnitudePass": magnitude_pass,
            "zeroTargetMagnitudePass": bool(zero_pass),
            "ranking": rank_record,
            "pass": bool(coverage_pass and magnitude_pass and zero_pass and rank_pass),
        }
    unknown = set(enabled_heads) - set(VALUE_HEADS)
    if unknown:
        raise ValueError(f"unknown enabled value heads: {sorted(unknown)}")
    source_policy = str(data.metadata["source"]["sourcePolicyDigest"])
    policy_current = current_policy_digest is not None and current_policy_digest == source_policy
    return {
        "schemaVersion": 1,
        "learnerContract": LEARNER_CONTRACT,
        "dataRole": data.role,
        "dataSha256": checkpoint_hash(data.path),
        "localReadinessClaim": False,
        "thresholds": thresholds.__dict__,
        "heads": heads,
        "enabledHeads": list(enabled_heads),
        "sourcePolicyDigest": source_policy,
        "currentPolicyDigest": current_policy_digest,
        "policyCurrent": policy_current,
        "stalePolicyAction": "actor-gate-closed-refresh-required",
        "actorGateOpen": bool(
            policy_current and all(heads[name]["pass"] for name in enabled_heads)
        ),
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build exact terminal observed-return anchors")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="NPZ native complete-terminal trajectory/fork source (never a truncated rollout)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    return parser


def main() -> int:
    args = argument_parser().parse_args()
    arrays = _source_arrays(args.source)
    kind = _metadata(arrays).get("kind")
    if kind == SOURCE_KIND:
        result = build_terminal_dataset(args.source, args.out, args.role)
    elif kind == ESTIMATOR_SOURCE_KIND:
        result = build_randomized_estimator_dataset(args.source, args.out, args.role)
    else:
        raise ValueError(f"unsupported observed-return source kind: {kind!r}")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
