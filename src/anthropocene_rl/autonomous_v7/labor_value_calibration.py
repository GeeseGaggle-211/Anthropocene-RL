"""Independent-return calibration for the opt-in labor/retry value heads.

This module consumes already packed, normalization-bound labor model inputs.
It never constructs targets from a critic bootstrap or interprets a truncated
trajectory as terminal.  Exact terminal returns and independently randomized
continuation estimates are the only accepted target methods.

The actor gate covers the two task heads and both constraint-cost critics.  A
ranking witness scores ``reward + discount * V(successor)`` for distinct
successor states; it never asks ``V(parent)`` to rank actions at one state.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contracts
from .labor_bridge import LABOR_REGISTRY_DIGEST
from .labor_model import (
    LABOR_COST_REFERENCE_SCALES,
    LABOR_COSTS,
    LABOR_MODEL_INPUT_CONTRACT,
    LABOR_OBJECTIVE_REFERENCE_SCALES,
    LABOR_OBJECTIVES,
)
from .provenance import checkpoint_hash, digest
from .value_calibration import CalibrationThresholds

LABOR_VALUE_HEADS = (*LABOR_OBJECTIVES, *LABOR_COSTS)
LABOR_ANCHOR_KIND = "model9-labor-independent-return-anchor-v1"
LABOR_ANCHOR_SOURCE_KINDS = frozenset(
    {
        "native-labor-complete-terminal-return-source-v1",
        "native-labor-randomized-continuation-source-v1",
    }
)
LABOR_RANK_SEMANTICS = "same-parent-distinct-successor-r-plus-gamma-v-v1"
LABOR_TARGET_METHODS = frozenset(
    {"terminal-exact", "independent-randomized-tail-path-mean", "unavailable"}
)
LABOR_ANCHOR_ROLES = frozenset({"train-anchor", "actor-gate"})


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _vector(value: object, label: str, n: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or (n is not None and len(result) != n):
        suffix = "" if n is None else f" with {n} rows"
        raise ValueError(f"{label} must be one-dimensional{suffix}")
    return result


def _metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    raw = arrays.get("metadata_json")
    if raw is None or np.asarray(raw).shape != ():
        raise ValueError("labor anchor metadata_json must be one scalar JSON string")
    try:
        value = json.loads(str(np.asarray(raw).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("labor anchor metadata_json is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("labor anchor metadata_json must decode to an object")
    return value


def _source_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _reference_scales(source: Mapping[str, Any]) -> dict[str, float]:
    task = source.get("objectiveReferenceScales")
    if task != LABOR_OBJECTIVE_REFERENCE_SCALES:
        raise ValueError("labor anchor objective reference scales differ from the model contract")
    costs = source.get("costReferenceScales")
    if costs != LABOR_COST_REFERENCE_SCALES:
        raise ValueError("labor anchor cost reference scales differ from the model contract")
    result = {name: float(LABOR_OBJECTIVE_REFERENCE_SCALES[name]) for name in LABOR_OBJECTIVES}
    for name in LABOR_COSTS:
        result[name] = float(LABOR_COST_REFERENCE_SCALES[name])
    return result


def _validate_source(source: Mapping[str, Any]) -> None:
    from .protocol import scan, seeds

    scan(dict(source), "labor independent-return source metadata")
    expected = {
        "schemaVersion": 1,
        "modelInputContract": LABOR_MODEL_INPUT_CONTRACT,
        "laborRegistryDigest": LABOR_REGISTRY_DIGEST,
        "criticBootstrapUsed": False,
        "truncatedTailAsZero": False,
        "independentUnit": "world",
        "policyDigestSemantics": "effective-sampled-policy-digest-v1",
        "rewardSemantics": "labor-qol-population-and-two-cost-returns-v1",
        "terminationSemantics": "camp-terminal-after-recorded-physical-transition",
        "physicalDiscountSemantics": "exp(-years/horizonYears)",
        "horizonYears": contracts.HORIZON_YEARS,
        "internalRetryDiscount": 1.0,
        "internalRetryTraceContinuation": 1.0,
        "physicalTaskRewardEdges": 1,
        "continuationPane": "sampled-training-policy",
    }
    for key, wanted in expected.items():
        if source.get(key) != wanted:
            raise ValueError(f"labor anchor source {key} {source.get(key)!r} != {wanted!r}")
    if source.get("kind") not in LABOR_ANCHOR_SOURCE_KINDS:
        raise ValueError("labor anchor source is not an approved independent-return producer")
    if source.get("kind") == "native-labor-complete-terminal-return-source-v1":
        if source.get("returnEstimator") != "observed-through-terminal":
            raise ValueError("terminal labor anchors must be observed through termination")
    else:
        if source.get("returnEstimator") != "fixed-prefix-plus-geometric-russian-roulette":
            raise ValueError("living labor anchors require the reviewed randomized-tail estimator")
        tail = source.get("randomizedTail")
        if not isinstance(tail, Mapping) or set(tail) != {
            "prefixTerms",
            "drawCount",
            "survivalProbability",
            "hardCapTerms",
            "scheduleDigest",
            "absoluteIntegrabilityEstablished",
            "finiteVarianceEstablished",
            "droppedDraws",
        }:
            raise ValueError("randomized labor anchors must bind the exact stopping schedule")
        positive_integers = (tail["prefixTerms"], tail["drawCount"], tail["hardCapTerms"])
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in positive_integers
            )
            or tail["drawCount"] < 2
            or isinstance(tail["survivalProbability"], bool)
            or not isinstance(tail["survivalProbability"], (int, float))
            or not 0.0 < float(tail["survivalProbability"]) < 1.0
            or not _valid_sha256(tail["scheduleDigest"])
            or tail["absoluteIntegrabilityEstablished"] is not True
            or tail["finiteVarianceEstablished"] is not True
            or tail["droppedDraws"] != 0
        ):
            raise ValueError("randomized labor stopping schedule is invalid or unresolved")
    for key in (
        "sourceCheckpointSha256",
        "sourcePolicyDigest",
        "sourceActorTensorDigest",
        "normalizationBindingDigest",
        "laborRegistryDigest",
    ):
        if not _valid_sha256(source.get(key)):
            raise ValueError(f"labor anchor source {key} must be a lowercase SHA-256")
    configuration = source.get("sourcePolicyConfiguration")
    if not isinstance(configuration, Mapping) or not configuration:
        raise ValueError("labor anchor source must bind the effective behavior configuration")
    expected_policy = digest(
        {
            "semantics": "effective-sampled-policy-digest-v1",
            "actorTensorDigest": source["sourceActorTensorDigest"],
            "configuration": dict(configuration),
        }
    )
    if source["sourcePolicyDigest"] != expected_policy:
        raise ValueError("labor source policy digest differs from its tensors/configuration")
    _reference_scales(source)
    declared: dict[str, list[int]] = {}
    for key in ("worldSeeds", "optimizerSeeds", "continuationSeeds"):
        values = source.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"labor anchor source {key} must be a non-empty list")
        checked = seeds(values, f"labor anchor source {key}")
        if len(checked) != len(set(checked)):
            raise ValueError(f"labor anchor source {key} contains duplicates")
        declared[key] = checked
    units = source.get("independentUnits")
    if not isinstance(units, list) or not units:
        raise ValueError("labor anchor source must bind independent world units")
    seen_ids: set[str] = set()
    seen_worlds: set[int] = set()
    observed = {key: set() for key in declared}
    for unit in units:
        if not isinstance(unit, Mapping):
            raise ValueError("each labor independent unit must be an object")
        unit_id = unit.get("unitId")
        if not isinstance(unit_id, str) or not unit_id or unit_id in seen_ids:
            raise ValueError("labor independent unit IDs must be unique non-empty strings")
        world = seeds([unit.get("worldSeed")], f"labor unit {unit_id} world")[0]
        optimizer = seeds([unit.get("optimizerSeed")], f"labor unit {unit_id} optimizer")[0]
        continuations = unit.get("continuationSeeds")
        if not isinstance(continuations, list) or not continuations:
            raise ValueError("labor independent units need continuation seed identities")
        continuations = seeds(continuations, f"labor unit {unit_id} continuations")
        if world in seen_worlds:
            raise ValueError("all branches from one world must share one independent unit")
        seen_ids.add(unit_id)
        seen_worlds.add(world)
        observed["worldSeeds"].add(world)
        observed["optimizerSeeds"].add(optimizer)
        observed["continuationSeeds"].update(continuations)
    for key, values in declared.items():
        if observed[key] != set(values):
            raise ValueError(f"labor anchor source {key} differs from its unit bindings")


@dataclass(frozen=True)
class LaborRankingWitness:
    successor_tensors: dict[str, np.ndarray]
    immediate_raw: dict[str, np.ndarray]
    realized_raw: dict[str, np.ndarray]
    realized_standard_errors: dict[str, np.ndarray]
    available: dict[str, np.ndarray]
    discount: np.ndarray
    terminal: np.ndarray
    group_ids: np.ndarray
    parent_digests: np.ndarray
    successor_digests: np.ndarray
    independent_unit_ids: np.ndarray


@dataclass(frozen=True)
class LaborObservedReturnData:
    path: Path
    role: str
    metadata: dict[str, Any]
    tensors: dict[str, np.ndarray]
    targets_raw: dict[str, np.ndarray]
    target_masks: dict[str, np.ndarray]
    target_standard_errors: dict[str, np.ndarray]
    target_methods: dict[str, np.ndarray]
    row_ids: np.ndarray
    independent_unit_ids: np.ndarray
    ranking: LaborRankingWitness | None

    def __len__(self) -> int:
        return len(self.row_ids)


def load_labor_anchor(path: Path, role: str) -> LaborObservedReturnData:
    """Load a strict four-head target file emitted by the native fork collector adapter."""

    if role not in LABOR_ANCHOR_ROLES:
        raise ValueError(f"labor anchor role must be one of {sorted(LABOR_ANCHOR_ROLES)}")
    arrays = _source_arrays(path)
    metadata = _metadata(arrays)
    expected = {
        "schemaVersion": 1,
        "kind": LABOR_ANCHOR_KIND,
        "role": role,
        "valueHeads": list(LABOR_VALUE_HEADS),
    }
    for key, wanted in expected.items():
        if metadata.get(key) != wanted:
            raise ValueError(f"labor anchor {key} {metadata.get(key)!r} != {wanted!r}")
    source = metadata.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("labor anchor lacks source metadata")
    _validate_source(source)
    rows = _vector(arrays.get("row_id"), "row_id").astype(str)
    n = len(rows)
    if not n or len(set(rows.tolist())) != n:
        raise ValueError("labor anchor row IDs must be non-empty and unique")
    units = _vector(arrays.get("independent_unit_id"), "independent_unit_id", n).astype(str)
    declared_units = {str(unit["unitId"]) for unit in source["independentUnits"]}
    if set(units.tolist()) != declared_units:
        raise ValueError("labor anchor rows differ from declared independent world units")
    tensors = {
        key.removeprefix("input__"): np.asarray(value)
        for key, value in arrays.items()
        if key.startswith("input__")
    }
    if not tensors or any(value.ndim == 0 or len(value) != n for value in tensors.values()):
        raise ValueError("labor anchor model tensors are absent or row-misaligned")
    for name, value in tensors.items():
        if value.dtype.kind not in "bif" or (
            value.dtype.kind == "f" and not np.isfinite(value).all()
        ):
            raise ValueError(f"labor anchor tensor {name} is not finite numeric/bool data")
    targets: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    errors: dict[str, np.ndarray] = {}
    methods: dict[str, np.ndarray] = {}
    for head in LABOR_VALUE_HEADS:
        targets[head] = _vector(arrays.get(f"target__{head}"), f"target__{head}", n).astype(
            np.float64
        )
        masks[head] = _vector(arrays.get(f"target_mask__{head}"), f"target_mask__{head}", n).astype(
            bool
        )
        errors[head] = _vector(
            arrays.get(f"target_standard_error__{head}"),
            f"target_standard_error__{head}",
            n,
        ).astype(np.float64)
        methods[head] = _vector(
            arrays.get(f"target_method__{head}"), f"target_method__{head}", n
        ).astype(str)
        if (
            not np.isfinite(targets[head]).all()
            or not np.isfinite(errors[head]).all()
            or bool((errors[head] < 0).any())
            or not set(methods[head].tolist()) <= LABOR_TARGET_METHODS
            or bool((masks[head] & (methods[head] == "unavailable")).any())
            or bool((~masks[head] & (methods[head] != "unavailable")).any())
        ):
            raise ValueError(f"labor anchor target contract is invalid for {head}")
    ranking = None
    rank_semantics = metadata.get("rankSemantics", "absent")
    if rank_semantics != "absent":
        if rank_semantics != LABOR_RANK_SEMANTICS:
            raise ValueError("labor anchor ranking semantics are unsupported")
        groups = _vector(arrays.get("rank_group_id"), "rank_group_id").astype(str)
        r = len(groups)
        successor_tensors = {
            key.removeprefix("rank_input__"): np.asarray(value)
            for key, value in arrays.items()
            if key.startswith("rank_input__")
        }
        if not successor_tensors or any(
            value.ndim == 0 or len(value) != r for value in successor_tensors.values()
        ):
            raise ValueError("labor ranking successor tensors are absent or row-misaligned")
        parents = _vector(arrays.get("rank_parent_digest"), "rank_parent_digest", r).astype(str)
        successors = _vector(
            arrays.get("rank_successor_digest"), "rank_successor_digest", r
        ).astype(str)
        rank_units = _vector(
            arrays.get("rank_independent_unit_id"), "rank_independent_unit_id", r
        ).astype(str)
        for group in np.unique(groups):
            members = np.flatnonzero(groups == group)
            if (
                len(members) < 2
                or len(set(parents[members].tolist())) != 1
                or len(set(successors[members].tolist())) < 2
                or len(set(rank_units[members].tolist())) != 1
            ):
                raise ValueError(
                    "each labor rank group needs one parent, one independent world unit, "
                    "and distinct successor states"
                )
        immediate: dict[str, np.ndarray] = {}
        realized: dict[str, np.ndarray] = {}
        errors_rank: dict[str, np.ndarray] = {}
        available: dict[str, np.ndarray] = {}
        for head in LABOR_VALUE_HEADS:
            immediate[head] = _vector(
                arrays.get(f"rank_immediate__{head}"), f"rank_immediate__{head}", r
            ).astype(np.float64)
            realized[head] = _vector(
                arrays.get(f"rank_realized__{head}"), f"rank_realized__{head}", r
            ).astype(np.float64)
            errors_rank[head] = _vector(
                arrays.get(f"rank_standard_error__{head}"),
                f"rank_standard_error__{head}",
                r,
            ).astype(np.float64)
            available[head] = _vector(
                arrays.get(f"rank_mask__{head}"), f"rank_mask__{head}", r
            ).astype(bool)
            if not all(
                np.isfinite(values).all()
                for values in (immediate[head], realized[head], errors_rank[head])
            ) or bool((errors_rank[head] < 0).any()):
                raise ValueError(f"labor ranking values are invalid for {head}")
        ranking = LaborRankingWitness(
            successor_tensors,
            immediate,
            realized,
            errors_rank,
            available,
            _vector(arrays.get("rank_discount"), "rank_discount", r).astype(np.float64),
            _vector(arrays.get("rank_terminal"), "rank_terminal", r).astype(bool),
            groups,
            parents,
            successors,
            rank_units,
        )
        if (
            not np.isfinite(ranking.discount).all()
            or bool((ranking.discount < 0).any())
            or bool((ranking.discount > 1).any())
            or not set(ranking.independent_unit_ids.tolist()) <= declared_units
        ):
            raise ValueError("labor ranking discounts/unit bindings are invalid")
    return LaborObservedReturnData(
        path.resolve(),
        role,
        metadata,
        tensors,
        targets,
        masks,
        errors,
        methods,
        rows,
        units,
        ranking,
    )


def _torch_rows(
    tensors: Mapping[str, np.ndarray], rows: np.ndarray | None = None
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        selected = value if rows is None else value[rows]
        dtype = torch.bool if selected.dtype.kind == "b" else torch.float32
        result[name] = torch.as_tensor(selected, dtype=dtype)
    return result


def labor_anchor_loss(
    task_critic: torch.nn.Module,
    cost_critics: torch.nn.Module,
    data: LaborObservedReturnData,
    rows: np.ndarray,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train all available heads only against independently observed returns."""

    selected = np.asarray(rows, dtype=np.int64)
    if (
        selected.ndim != 1
        or not len(selected)
        or bool((selected < 0).any())
        or bool((selected >= len(data)).any())
    ):
        raise ValueError("labor anchor rows must be a non-empty valid index vector")
    tensors = _torch_rows(data.tensors, selected)
    task_logits = task_critic(tensors)
    costs = cost_critics(tensors)
    if set(task_logits) != set(LABOR_OBJECTIVES) or set(costs) != set(LABOR_COSTS):
        raise ValueError("labor critic heads differ from the four-head contract")
    selected_units = data.independent_unit_ids[selected]

    def equal_unit_mean(row_loss: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        unit_losses = []
        for unit in sorted(set(selected_units[available.cpu().numpy()].tolist())):
            member = torch.as_tensor(selected_units == unit, dtype=torch.bool) & available
            unit_losses.append(row_loss[member].mean())
        if not unit_losses:
            raise ValueError("labor anchor head has no available independent unit")
        return torch.stack(unit_losses).mean()

    losses: dict[str, torch.Tensor] = {}
    for head in LABOR_OBJECTIVES:
        available = torch.as_tensor(data.target_masks[head][selected], dtype=torch.bool)
        if bool(available.any()):
            target = torch.as_tensor(data.targets_raw[head][selected], dtype=torch.float32)
            row_loss = torch.zeros(len(selected), dtype=task_logits[head].dtype)
            row_loss[available] = -(
                task_critic.two_hot(head, target[available])
                * torch.log_softmax(task_logits[head][available], dim=-1)
            ).sum(-1)
            losses[head] = equal_unit_mean(row_loss, available)
    for head in LABOR_COSTS:
        available = torch.as_tensor(data.target_masks[head][selected], dtype=torch.bool)
        if bool(available.any()):
            target = torch.as_tensor(data.targets_raw[head][selected], dtype=torch.float32)
            row_loss = torch.zeros(len(selected), dtype=costs[head].dtype)
            row_loss[available] = (costs[head][available] - target[available]).square()
            losses[head] = equal_unit_mean(row_loss, available)
    if not losses:
        raise ValueError("labor anchor minibatch has no independently identified target")
    return sum(losses.values()), losses


def labor_critic_digest(task_critic: torch.nn.Module, cost_critics: torch.nn.Module) -> str:
    from .economic_trace import state_digest

    return digest(
        {
            "semantics": "labor-four-head-critic-tensor-digest-v1",
            "taskCritic": state_digest(task_critic),
            "costCritics": state_digest(cost_critics),
        }
    )


def _predict(
    task_critic: torch.nn.Module,
    cost_critics: torch.nn.Module,
    tensors: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    packed = _torch_rows(tensors)
    with torch.no_grad():
        task_logits = task_critic(packed)
        costs = cost_critics(packed)
        if set(task_logits) != set(LABOR_OBJECTIVES) or set(costs) != set(LABOR_COSTS):
            raise ValueError("labor critic heads differ from the four-head contract")
        result = {
            head: task_critic.expectation(head, task_logits[head])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            for head in LABOR_OBJECTIVES
        }
        result.update(
            {head: costs[head].detach().cpu().numpy().astype(np.float64) for head in LABOR_COSTS}
        )
    return result


def _relation(value: float, tolerance: float) -> int:
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def labor_calibration_report(
    task_critic: torch.nn.Module,
    cost_critics: torch.nn.Module,
    data: LaborObservedReturnData,
    *,
    current_policy_digest: str,
    current_normalization_binding_digest: str,
    current_registry_digest: str,
    thresholds: CalibrationThresholds | None = None,
) -> dict[str, Any]:
    """Score held-out full-return evidence without making a readiness claim."""

    thresholds = CalibrationThresholds() if thresholds is None else thresholds
    source = data.metadata["source"]
    scales = _reference_scales(source)
    predictions = _predict(task_critic, cost_critics, data.tensors)
    rank_scores = None
    if data.ranking is not None:
        successor = _predict(task_critic, cost_critics, data.ranking.successor_tensors)
        rank_scores = {
            head: data.ranking.immediate_raw[head]
            + data.ranking.discount * np.where(data.ranking.terminal, 0.0, successor[head])
            for head in LABOR_VALUE_HEADS
        }
    heads: dict[str, Any] = {}
    for head in LABOR_VALUE_HEADS:
        selected = data.target_masks[head]
        if not bool(selected.any()):
            heads[head] = {
                "rows": 0,
                "requiredForActor": True,
                "pass": False,
                "status": "no-independently-identified-targets",
            }
            continue
        scale = scales[head]
        target = data.targets_raw[head][selected]
        error = predictions[head][selected] - target
        standard_error = data.target_standard_errors[head][selected]
        selected_units = data.independent_unit_ids[selected]
        unit_ids = sorted(set(selected_units.tolist()))
        unit_mean_errors = []
        unit_mean_abs_errors = []
        unit_mean_abs_predictions = []
        unit_standard_errors = []
        selected_predictions = predictions[head][selected]
        for unit in unit_ids:
            member = selected_units == unit
            unit_mean_errors.append(float(np.mean(error[member])))
            unit_mean_abs_errors.append(float(np.mean(np.abs(error[member]))))
            unit_mean_abs_predictions.append(float(np.mean(np.abs(selected_predictions[member]))))
            # Same-world rows are correlated. Treat their stopping-error
            # contributions as perfectly correlated before combining worlds.
            unit_standard_errors.append(float(np.mean(standard_error[member])))
        mean_se = float(np.sqrt(np.square(unit_standard_errors).sum()) / len(unit_standard_errors))
        conservative_mae = float(
            np.mean(
                np.asarray(unit_mean_abs_errors)
                + thresholds.stopping_randomization_z * np.asarray(unit_standard_errors)
            )
        )
        equal_unit_bias = float(np.mean(unit_mean_errors))
        conservative_bias = float(
            abs(equal_unit_bias) + thresholds.stopping_randomization_z * mean_se
        )
        near_zero = bool(np.max(np.abs(target)) <= thresholds.rank_tolerance * scale)
        magnitude_pass = bool(
            conservative_mae / scale <= thresholds.max_normalized_mae
            and conservative_bias / scale <= thresholds.max_normalized_abs_bias
        )
        zero_pass = bool(
            not near_zero
            or np.mean(unit_mean_abs_predictions) / scale <= thresholds.max_zero_target_prediction
        )
        units = set(unit_ids)
        living = selected & (data.target_methods[head] == "independent-randomized-tail-path-mean")
        living_units = set(data.independent_unit_ids[living].tolist())
        coverage_pass = bool(
            len(units) >= thresholds.min_independent_units
            and len(living_units) >= thresholds.min_living_continuation_units
        )
        rank_record: dict[str, Any] = {
            "semantics": LABOR_RANK_SEMANTICS,
            "available": data.ranking is not None,
            "required": not near_zero,
            "status": "not-required-near-zero-target" if near_zero else "missing",
            "informativeGroups": 0,
            "informativeIndependentUnits": 0,
            "equalIndependentUnitAccuracy": None,
            "targetTies": 0,
            "predictedTies": 0,
            "uncertainTargetPairs": 0,
        }
        rank_pass = near_zero
        if data.ranking is not None and rank_scores is not None:
            group_accuracies_by_unit: dict[str, list[float]] = {}
            pairs = target_ties = predicted_ties = uncertain = 0
            for group in np.unique(data.ranking.group_ids):
                group_rows = np.flatnonzero(data.ranking.group_ids == group)
                group_rows = group_rows[data.ranking.available[head][group_rows]]
                unit = (
                    str(data.ranking.independent_unit_ids[group_rows[0]])
                    if len(group_rows)
                    else None
                )
                correct = informative = 0
                for offset, left in enumerate(group_rows):
                    for right in group_rows[offset + 1 :]:
                        difference = float(
                            data.ranking.realized_raw[head][left]
                            - data.ranking.realized_raw[head][right]
                        )
                        uncertainty = thresholds.stopping_randomization_z * float(
                            data.ranking.realized_standard_errors[head][left]
                            + data.ranking.realized_standard_errors[head][right]
                        )
                        if uncertainty and abs(difference) <= (
                            thresholds.rank_tolerance * scale + uncertainty
                        ):
                            realized_relation = None
                            uncertain += 1
                        else:
                            realized_relation = _relation(
                                difference, thresholds.rank_tolerance * scale
                            )
                        predicted_relation = _relation(
                            float(rank_scores[head][left] - rank_scores[head][right]),
                            thresholds.rank_tolerance * scale,
                        )
                        pairs += 1
                        target_ties += int(realized_relation == 0)
                        predicted_ties += int(predicted_relation == 0)
                        if realized_relation not in (None, 0):
                            informative += 1
                            correct += int(realized_relation == predicted_relation)
                if informative:
                    assert unit is not None
                    group_accuracies_by_unit.setdefault(unit, []).append(correct / informative)
            unit_accuracies = [
                float(np.mean(values)) for values in group_accuracies_by_unit.values()
            ]
            informative_groups = sum(len(values) for values in group_accuracies_by_unit.values())
            rank_record.update(
                {
                    "status": (
                        "informative"
                        if unit_accuracies
                        else "uncertain-target-pairs"
                        if uncertain
                        else "all-realized-pairs-tied"
                        if pairs
                        else "no-available-pairs"
                    ),
                    "pairCount": pairs,
                    "informativeGroups": informative_groups,
                    "informativeIndependentUnits": len(unit_accuracies),
                    "equalIndependentUnitAccuracy": float(np.mean(unit_accuracies))
                    if unit_accuracies
                    else None,
                    "targetTies": target_ties,
                    "predictedTies": predicted_ties,
                    "uncertainTargetPairs": uncertain,
                }
            )
            if unit_accuracies:
                rank_pass = bool(
                    len(unit_accuracies) >= thresholds.min_informative_rank_groups
                    and np.mean(unit_accuracies) >= thresholds.min_equal_group_rank_accuracy
                )
        heads[head] = {
            "rows": int(selected.sum()),
            "independentUnits": len(units),
            "livingContinuationUnits": len(living_units),
            "coveragePass": coverage_pass,
            "requiredForActor": True,
            "normalizedEqualIndependentUnitMeanAbsoluteError": float(
                np.mean(unit_mean_abs_errors) / scale
            ),
            "normalizedEqualIndependentUnitBias": equal_unit_bias / scale,
            "normalizedConservativeMeanAbsoluteError": conservative_mae / scale,
            "normalizedConservativeAbsoluteBias": conservative_bias / scale,
            "normalizedEqualIndependentUnitMeanAbsolutePrediction": float(
                np.mean(unit_mean_abs_predictions) / scale
            ),
            "targetNearZero": near_zero,
            "magnitudePass": magnitude_pass,
            "zeroTargetMagnitudePass": zero_pass,
            "ranking": rank_record,
            "pass": bool(coverage_pass and magnitude_pass and zero_pass and rank_pass),
            "targetUncertaintyScope": (
                "stopping-randomization/path-mean SE only; excludes finite-world and "
                "policy sampling uncertainty"
            ),
        }
    policy_current = current_policy_digest == source["sourcePolicyDigest"]
    normalization_current = (
        current_normalization_binding_digest == source["normalizationBindingDigest"]
    )
    registry_current = current_registry_digest == source["laborRegistryDigest"]
    critic_digest = labor_critic_digest(task_critic, cost_critics)
    return {
        "schemaVersion": 1,
        "kind": "model9-labor-value-calibration-report-v1",
        "dataRole": data.role,
        "dataSha256": checkpoint_hash(data.path),
        "localReadinessClaim": False,
        "enabledHeads": list(LABOR_VALUE_HEADS),
        "thresholds": asdict(thresholds),
        "worldSeeds": list(source["worldSeeds"]),
        "independentUnitIds": sorted(set(data.independent_unit_ids.tolist())),
        "heads": heads,
        "sourcePolicyDigest": source["sourcePolicyDigest"],
        "currentPolicyDigest": current_policy_digest,
        "policyCurrent": policy_current,
        "sourceCriticDigest": critic_digest,
        "normalizationBindingDigest": source["normalizationBindingDigest"],
        "normalizationCurrent": normalization_current,
        "registryCurrent": registry_current,
        "actorGateOpen": bool(
            data.role == "actor-gate"
            and policy_current
            and normalization_current
            and registry_current
            and all(heads[head]["pass"] for head in LABOR_VALUE_HEADS)
        ),
        "stalePolicyAction": "actor-gate-closed-refresh-required",
        "scope": (
            "current-policy four-head calibration on independent world units; "
            "not a local readiness or treatment-selection result"
        ),
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a labor actor-gate anchor against one exact learner checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-normalized-mae", type=float, default=1.0)
    parser.add_argument("--max-normalized-abs-bias", type=float, default=0.5)
    parser.add_argument("--max-zero-target-prediction", type=float, default=0.25)
    parser.add_argument("--rank-tolerance", type=float, default=0.02)
    parser.add_argument("--min-informative-rank-groups", type=int, default=2)
    parser.add_argument("--min-equal-group-rank-accuracy", type=float, default=0.6)
    parser.add_argument("--min-independent-units", type=int, default=1)
    parser.add_argument("--min-living-continuation-units", type=int, default=1)
    parser.add_argument("--stopping-randomization-z", type=float, default=1.96)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Emit a machine-consumable current-policy gate report; never a readiness claim."""

    # Local imports avoid a labor_training -> labor_value_calibration cycle.
    from .labor_model import labor_checkpoint_personality
    from .labor_training import (
        _models_from_checkpoint,
        _normalization_digest,
        labor_behavior_configuration,
    )
    from .value_calibration import behavior_policy_digest

    torch.set_num_threads(4)
    args = argument_parser().parse_args(argv)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _, actor, task_critic, cost_critics, normalizer = _models_from_checkpoint(state)
    personality = labor_checkpoint_personality(state)
    behavior = labor_behavior_configuration(personality)
    data = load_labor_anchor(args.anchor, "actor-gate")
    thresholds = CalibrationThresholds(
        max_normalized_mae=args.max_normalized_mae,
        max_normalized_abs_bias=args.max_normalized_abs_bias,
        max_zero_target_prediction=args.max_zero_target_prediction,
        rank_tolerance=args.rank_tolerance,
        min_informative_rank_groups=args.min_informative_rank_groups,
        min_equal_group_rank_accuracy=args.min_equal_group_rank_accuracy,
        min_independent_units=args.min_independent_units,
        min_living_continuation_units=args.min_living_continuation_units,
        stopping_randomization_z=args.stopping_randomization_z,
    )
    report = labor_calibration_report(
        task_critic,
        cost_critics,
        data,
        current_policy_digest=behavior_policy_digest(actor, behavior),
        current_normalization_binding_digest=_normalization_digest(normalizer),
        current_registry_digest=LABOR_REGISTRY_DIGEST,
        thresholds=thresholds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
