"""Five-step, no-learning successor sensitivity probe for the reviewed T3per state.

This is a narrow model-input diagnostic, not another Gate 3/4 collector.  It
reuses the single-state debug endpoint, reset state and replicate-zero action /
personality seeds.  One native-label step and two steps on each of two exact
forks test whether the frozen critic responds to the previous *requested*
development control when the immediate physical transition does not.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from . import contracts, ppo
from .bridge import Model9Bridge
from .gate1_production_adapter import tensor_state_identity
from .gate34_checkpoint_collector import (
    DEVELOPMENT_NATIVE_INTERVENTION,
    EndpointRuntime,
    PersonalityState,
    SampledActions,
    _canonical_sha256,
    _critic_predictions,
    _derived_seed,
    _live_villages,
    apply_first_action_intervention,
    load_endpoint_registry,
    load_endpoint_runtime,
    probe_native_labels,
    sample_training_actions,
)
from .gate34_continuation_collector import ForkedAlternatives
from .gate34_single_state_debug import (
    DECISION_STEP,
    ENDPOINT_ID,
    EXPECTED_RESET_TURN,
    REGIME_ID,
    TORCH_NUM_THREADS,
    WITHHELD_MASK,
    WORLD_SEED,
    _trace_reset_binding,
)
from .model import symexp, to_tensors
from .provenance import ProvenanceError, digest

Json = dict[str, Any]

REQUESTED_DEVELOPMENT_CORE_INDEX = 47
ACCEPTED_DEVELOPMENT_CORE_INDEX = 48
REALIZED_DEVELOPMENT_CORE_INDEX = 49
PREVIOUS_DECISION_CORE_START = 40
PREVIOUS_DECISION_CORE_STOP = 56
MAX_INDIVIDUAL_CORE_PROBES = 16
EXPECTED_BRIDGE_STEPS = 5
RESULT_KIND = "gate34-successor-request-sensitivity"
CORE_SEMANTICS = {
    REQUESTED_DEVELOPMENT_CORE_INDEX: "previous-requested-development-budget",
    ACCEPTED_DEVELOPMENT_CORE_INDEX: "previous-accepted-development-budget",
    REALIZED_DEVELOPMENT_CORE_INDEX: "previous-realized-development-budget",
    53: "previous-intervention-count",
}


def _model_identity(runtime: EndpointRuntime) -> Json:
    return {
        "actorTypedContentSha256": tensor_state_identity(runtime.actor.state_dict())[
            "typed_content_sha256"
        ],
        "criticTypedContentSha256": tensor_state_identity(runtime.critic.state_dict())[
            "typed_content_sha256"
        ],
    }


def _packed_step(
    step: Json, personalities: PersonalityState
) -> tuple[list[int], Json]:
    live = sorted(_live_villages(step), key=lambda village: int(village["camp"]))
    camps = [int(village["camp"]) for village in live]
    batch = cast(Json, ppo.village_batch(live))
    personalities.add_to_batch(batch, live)
    batch["prior_population"] = np.asarray(
        [max(1, int(village["population"])) for village in live], dtype=np.float64
    )
    return camps, batch


def _head_values(runtime: EndpointRuntime, camps: Sequence[int], batch: Json) -> Json:
    tensors = to_tensors(batch, runtime.normalizer)
    with torch.no_grad():
        output = runtime.critic(tensors)
    if not isinstance(output, dict):
        raise ProvenanceError("successor sensitivity requires the vector critic")
    bins = runtime.critic.state_dict()["bins"].detach()
    raw_bins = symexp(bins)
    population = np.asarray(batch["prior_population"], dtype=np.float64)
    result: Json = {}
    for objective in runtime.objectives:
        logits = output[objective]
        per_resident = (
            (torch.softmax(logits, dim=-1) * raw_bins.to(logits.device))
            .sum(dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        scale = (
            np.maximum(1.0, population)
            if objective in ppo.PER_RESIDENT
            else np.ones_like(population)
        )
        result[objective] = {
            str(camp): {
                "perResidentValue": float(value),
                "populationScaledRawValue": float(value * scale[row]),
                "populationScale": float(scale[row]),
            }
            for row, (camp, value) in enumerate(
                zip(camps, per_resident, strict=True)
            )
        }
    return result


def _copy_batch(batch: Json) -> Json:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
        for key, value in batch.items()
    }


def _swap_raw_core_coordinates(
    left: Json, right: Json, coordinates: Sequence[int]
) -> Json:
    """Copy selected raw core coordinates from ``right`` before packing."""

    if np.asarray(left["core"]).shape != np.asarray(right["core"]).shape:
        raise ValueError("paired successor core shapes differ")
    swapped = _copy_batch(left)
    width = int(np.asarray(left["core"]).shape[1])
    selected = tuple(dict.fromkeys(int(index) for index in coordinates))
    if any(index < 0 or index >= width for index in selected):
        raise IndexError("raw core swap coordinate is outside the model contract")
    if selected:
        swapped["core"][:, selected] = right["core"][:, selected]
        swapped["core_present"][:, selected] = right["core_present"][:, selected]
    return swapped


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _model_input_keys(batch: Json) -> set[str]:
    keys = {
        "core",
        "core_present",
        "movement_mask",
        "clothing_route_mask",
        "shelter_route_mask",
        "economy_mix_mask",
        "region_movement_mask",
        "commit_hold_mask",
        "personality",
        "privileged",
    }
    for name, _, _ in contracts.SETS:
        keys.update((name, name + "_present"))
    return keys & set(batch)


def _changed_input_coordinates(
    baseline: Json, intervention: Json, camps: Sequence[int]
) -> list[Json]:
    """Enumerate every exact changed numeric/bool model-input coordinate."""

    changes: list[Json] = []
    keys = sorted(_model_input_keys(baseline) | _model_input_keys(intervention))
    for key in keys:
        left, right = baseline.get(key), intervention.get(key)
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            continue
        if left.shape != right.shape:
            raise ValueError(f"paired model input shape differs for {key}")
        if left.ndim == 0 or left.shape[0] != len(camps):
            continue
        equal = left == right
        if np.issubdtype(left.dtype, np.floating) and np.issubdtype(
            right.dtype, np.floating
        ):
            equal = equal | (np.isnan(left) & np.isnan(right))
        for coordinate in np.argwhere(~equal):
            row = int(coordinate[0])
            index = [int(value) for value in coordinate[1:]]
            change: Json = {
                "camp": int(camps[row]),
                "input": key,
                "index": index,
                "baseline": _json_scalar(left[tuple(coordinate)]),
                "intervention": _json_scalar(right[tuple(coordinate)]),
            }
            if key == "core" and len(index) == 1 and index[0] in CORE_SEMANTICS:
                change["semantic"] = CORE_SEMANTICS[index[0]]
            changes.append(change)
    return changes


def _core_coordinate_stages(
    runtime: EndpointRuntime,
    batch: Json,
    camps: Sequence[int],
    coordinates: Sequence[int],
) -> Json:
    """Expose the exact raw -> symlog -> affine -> packed core path."""

    selected = tuple(dict.fromkeys(int(index) for index in coordinates))
    raw = np.asarray(batch["core"])
    present = np.asarray(batch["core_present"])
    packed = to_tensors(batch, runtime.normalizer)["core"].detach().cpu().numpy()
    result: Json = {}
    for index in selected:
        transformed = np.sign(raw[:, index]) * np.log1p(np.abs(raw[:, index]))
        mean = float(runtime.normalizer.core_mean[index])
        std = float(runtime.normalizer.core_std[index])
        affine = (transformed - mean) / std
        dead = runtime.normalizer.dead.get("core")
        result[str(index)] = {
            "normalizer": {
                "mean": mean,
                "std": std,
                "guard": runtime.normalizer.guard,
                "dead": bool(dead[index]) if dead is not None else False,
            },
            "rows": [
                {
                    "camp": int(camp),
                    "present": bool(present[row, index]),
                    "raw": float(raw[row, index]),
                    "symlogTransformed": float(transformed[row]),
                    "affineNormalizedBeforeGuard": float(affine[row]),
                    "packedNormalized": float(packed[row, index]),
                }
                for row, camp in enumerate(camps)
            ],
        }
    return result


def _hybrid_value_probe(
    runtime: EndpointRuntime,
    camps: Sequence[int],
    baseline: Json,
    intervention: Json,
    coordinates: Sequence[int],
) -> Json:
    selected = tuple(dict.fromkeys(int(index) for index in coordinates))
    baseline_swap = _swap_raw_core_coordinates(baseline, intervention, selected)
    intervention_swap = _swap_raw_core_coordinates(intervention, baseline, selected)
    return {
        "rawCoreCoordinates": list(selected),
        "baselineWithInterventionCoordinates": _head_values(
            runtime, camps, baseline_swap
        ),
        "interventionWithBaselineCoordinates": _head_values(
            runtime, camps, intervention_swap
        ),
        "inputStages": {
            "baselineWithInterventionCoordinates": _core_coordinate_stages(
                runtime, baseline_swap, camps, selected
            ),
            "interventionWithBaselineCoordinates": _core_coordinate_stages(
                runtime, intervention_swap, camps, selected
            ),
        },
    }


def _observation_comparison(
    baseline_step: Json, intervention_step: Json, camps: Sequence[int]
) -> Json:
    baseline = {int(row["camp"]): row for row in _live_villages(baseline_step)}
    intervention = {
        int(row["camp"]): row for row in _live_villages(intervention_step)
    }
    rows = []
    for camp in camps:
        left = baseline[camp]["observation"]
        right = intervention[camp]["observation"]
        left_hash = _canonical_sha256(left)
        right_hash = _canonical_sha256(right)
        rows.append(
            {
                "camp": int(camp),
                "baselineSha256": left_hash,
                "interventionSha256": right_hash,
                "equal": left_hash == right_hash,
            }
        )
    return {
        "byCamp": rows,
        "allEqual": all(bool(row["equal"]) for row in rows),
        "semantics": (
            "observation payloads include bridge previous-decision memory; the "
            "native world fingerprint is reported separately and need not bind it"
        ),
    }


def _successor_comparison(
    runtime: EndpointRuntime,
    baseline_step: Json,
    intervention_step: Json,
    baseline_personalities: PersonalityState,
    intervention_personalities: PersonalityState,
) -> Json:
    baseline_camps, baseline = _packed_step(baseline_step, baseline_personalities)
    intervention_camps, intervention = _packed_step(
        intervention_step, intervention_personalities
    )
    if baseline_camps != intervention_camps:
        raise ValueError("paired successor live camp identities differ")
    changes = _changed_input_coordinates(baseline, intervention, baseline_camps)
    request_changes = [
        change
        for change in changes
        if change["input"] == "core"
        and change["index"] == [REQUESTED_DEVELOPMENT_CORE_INDEX]
    ]
    other_changes = [change for change in changes if change not in request_changes]
    changed_core = sorted(
        {
            int(change["index"][0])
            for change in changes
            if change["input"] == "core" and len(change["index"]) == 1
        }
    )
    individual: Json
    if len(changed_core) <= MAX_INDIVIDUAL_CORE_PROBES:
        individual = {
            "status": "evaluated",
            "maximum": MAX_INDIVIDUAL_CORE_PROBES,
            "probes": {
                str(index): _hybrid_value_probe(
                    runtime, baseline_camps, baseline, intervention, (index,)
                )
                for index in changed_core
            },
        }
    else:
        individual = {
            "status": "withheld-too-many-changed-core-coordinates",
            "maximum": MAX_INDIVIDUAL_CORE_PROBES,
            "changedCoordinateCount": len(changed_core),
            "probes": {},
        }
    return {
        "camps": baseline_camps,
        "observations": _observation_comparison(
            baseline_step, intervention_step, baseline_camps
        ),
        "fingerprints": {
            "baseline": str(baseline_step["fingerprint"]),
            "intervention": str(intervention_step["fingerprint"]),
            "equal": str(baseline_step["fingerprint"])
            == str(intervention_step["fingerprint"]),
            "semantics": (
                "native world fingerprint; bridge-local previous-decision memory is "
                "represented in observations and is not independently bound by this hash"
            ),
        },
        "changedInputCoordinates": changes,
        "requestedDevelopmentCoordinateChanges": request_changes,
        "otherChangedInputCoordinates": other_changes,
        "changedRawCoreCoordinates": changed_core,
        "changedRawCoreInputStages": {
            "baseline": _core_coordinate_stages(
                runtime, baseline, baseline_camps, changed_core
            ),
            "intervention": _core_coordinate_stages(
                runtime, intervention, baseline_camps, changed_core
            ),
        },
        "values": {
            "baseline": _head_values(runtime, baseline_camps, baseline),
            "intervention": _head_values(runtime, baseline_camps, intervention),
        },
        "rawCoreHybridProbes": {
            "requestedDevelopmentCore47": _hybrid_value_probe(
                runtime,
                baseline_camps,
                baseline,
                intervention,
                (REQUESTED_DEVELOPMENT_CORE_INDEX,),
            ),
            "previousDecisionBlock40To55": _hybrid_value_probe(
                runtime,
                baseline_camps,
                baseline,
                intervention,
                range(PREVIOUS_DECISION_CORE_START, PREVIOUS_DECISION_CORE_STOP),
            ),
            "allChangedRawCoreCoordinates": _hybrid_value_probe(
                runtime, baseline_camps, baseline, intervention, changed_core
            ),
            "individualChangedRawCoreCoordinates": individual,
        },
        "requestedDevelopmentCore47InputStages": {
            "baseline": _core_coordinate_stages(
                runtime,
                baseline,
                baseline_camps,
                (REQUESTED_DEVELOPMENT_CORE_INDEX,),
            )[str(REQUESTED_DEVELOPMENT_CORE_INDEX)],
            "intervention": _core_coordinate_stages(
                runtime,
                intervention,
                baseline_camps,
                (REQUESTED_DEVELOPMENT_CORE_INDEX,),
            )[str(REQUESTED_DEVELOPMENT_CORE_INDEX)],
        },
        "swapInterpretation": (
            "two-direction raw-input hybrids packed through the checkpoint's normalizer; "
            "they are artificial model forward probes, not simulator states or "
            "additive attribution, and nonlinear probe differences must not be summed"
        ),
    }


def _policy_domain(step: Json) -> tuple[list[int], Json]:
    camps, batch = _packed_step(step, PersonalityState("default", 0.0, 0))
    keys = (
        "movement_mask",
        "clothing_route_mask",
        "shelter_route_mask",
        "economy_mix_mask",
        "region_movement_mask",
        "commit_hold_mask",
    )
    return camps, {
        key: np.asarray(batch[key]).tolist() for key in keys if key in batch
    }


def _require_shared_second_action_domain(baseline: Json, intervention: Json) -> Json:
    baseline_camps, baseline_masks = _policy_domain(baseline)
    intervention_camps, intervention_masks = _policy_domain(intervention)
    if baseline_camps != intervention_camps:
        raise ValueError("step-two paired camp identities differ")
    if baseline_masks != intervention_masks:
        raise ValueError("step-two action legality masks differ")
    return {
        "campIds": baseline_camps,
        "policyMasks": baseline_masks,
        "status": "identical-domain-fixed-action-map-permitted",
    }


def _event_groups(rows: Sequence[Json]) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], float] = {}
    for row in rows:
        camp = int(row["campId"])
        for event in cast(list[Json], row.get("physicalEvents", [])):
            key = (
                camp,
                event.get("category"),
                event.get("item"),
                event.get("unit"),
                event.get("direction"),
                event.get("phase"),
            )
            grouped[key] = grouped.get(key, 0.0) + float(event["quantity"])
    return grouped


def _physical_event_comparison(baseline: Sequence[Json], intervention: Sequence[Json]) -> Json:
    left, right = _event_groups(baseline), _event_groups(intervention)
    differences = []
    for key in sorted(set(left) | set(right), key=lambda value: repr(value)):
        before, after = left.get(key, 0.0), right.get(key, 0.0)
        if before != after:
            differences.append(
                {
                    "camp": key[0],
                    "category": key[1],
                    "item": key[2],
                    "unit": key[3],
                    "direction": key[4],
                    "phase": key[5],
                    "baselineQuantity": before,
                    "interventionQuantity": after,
                    "difference": after - before,
                }
            )
    return {
        "groupKey": ["camp", "category", "item", "unit", "direction", "phase"],
        "branchAndEventIdsExcluded": True,
        "baselineGroupCount": len(left),
        "interventionGroupCount": len(right),
        "differingGroups": differences,
        "allGroupedQuantitiesEqual": not differences,
    }


def _economic_rows(bridge: Model9Bridge) -> list[Json]:
    rows = getattr(bridge, "economic_diagnostics", None)
    if not isinstance(rows, dict):
        raise ValueError("bridge lacks economic diagnostics")
    return [copy.deepcopy(rows[camp]) for camp in sorted(rows)]


def _first_actions(
    runtime: EndpointRuntime,
    current: Json,
    main_personalities: PersonalityState,
    action_seed: int,
    personality_seed: int,
) -> tuple[SampledActions, PersonalityState, torch.Generator]:
    personalities = main_personalities.clone_for_path(personality_seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(action_seed)
    sampled = sample_training_actions(runtime, current, generator, personalities)
    return sampled, personalities, generator


def _run_bound_probe(
    bridge: Model9Bridge,
    runtime: EndpointRuntime,
    current: Json,
) -> Json:
    main_seed = _derived_seed(
        "gate34-main-personality-v1", ENDPOINT_ID, WORLD_SEED, REGIME_ID
    )
    main_personalities = PersonalityState(
        runtime.personality_spec, runtime.personality_drift, main_seed
    )
    # The existing frozen-state collector initializes lineage traits through its
    # opening critic prediction before cloning continuation paths.
    opening = sorted(_live_villages(current), key=lambda village: int(village["camp"]))
    _critic_predictions(runtime, opening, main_personalities)
    opening_personality_context = [
        {
            "camp": int(village["camp"]),
            "society": int(village["society"]),
            "parentSociety": int(village.get("parentSociety", 0)),
            "weights": (
                [
                    float(value)
                    for value in main_personalities.traits[int(village["society"])]
                ]
                if main_personalities.active
                else None
            ),
        }
        for village in opening
    ]
    action_seed = _derived_seed(
        "gate34-action-v1", ENDPOINT_ID, WORLD_SEED, REGIME_ID, DECISION_STEP, 0
    )
    personality_seed = _derived_seed(
        "gate34-personality-v1", ENDPOINT_ID, WORLD_SEED, REGIME_ID, DECISION_STEP, 0
    )

    step_count = 0
    with ForkedAlternatives(bridge) as forks:
        label_state = forks.begin("native-label-probe")
        labels = probe_native_labels(bridge, label_state)
        step_count += 1

        baseline_state = forks.begin("replicate-0/baseline")
        baseline_first, baseline_personalities, baseline_generator = _first_actions(
            runtime,
            baseline_state,
            main_personalities,
            action_seed,
            personality_seed,
        )
        baseline_successor = bridge.step(dict(baseline_first.actions))
        step_count += 1
        baseline_first_economic = _economic_rows(bridge)
        baseline_second = sample_training_actions(
            runtime, baseline_successor, baseline_generator, baseline_personalities
        )
        baseline_second_actions = copy.deepcopy(dict(baseline_second.actions))
        baseline_end = bridge.step(baseline_second_actions)
        step_count += 1
        baseline_second_economic = _economic_rows(bridge)

        intervention_state = forks.begin("replicate-0/development-budget-native")
        intervention_first, intervention_personalities, intervention_generator = (
            _first_actions(
                runtime,
                intervention_state,
                main_personalities,
                action_seed,
                personality_seed,
            )
        )
        if _canonical_sha256(intervention_first.actions) != _canonical_sha256(
            baseline_first.actions
        ):
            raise RuntimeError("paired replicate-zero first action samples differ")
        intervention_actions: dict[int, Json] = {}
        receipts = []
        by_camp = {
            int(village["camp"]): village for village in _live_villages(intervention_state)
        }
        for camp, action in sorted(intervention_first.actions.items()):
            changed, receipt = apply_first_action_intervention(
                DEVELOPMENT_NATIVE_INTERVENTION,
                camp_id=camp,
                action=cast(Json, action),
                native_label=labels[camp],
                village=by_camp[camp],
            )
            intervention_actions[camp] = changed
            receipts.append(
                {
                    "camp": camp,
                    "beforeSha256": receipt.before_sha256,
                    "afterSha256": receipt.after_sha256,
                    "changedPaths": list(receipt.changed_paths),
                    "exactZeroChange": receipt.exact_zero_change,
                }
            )
        intervention_successor = bridge.step(intervention_actions)
        step_count += 1
        intervention_first_economic = _economic_rows(bridge)

        successor_one = _successor_comparison(
            runtime,
            baseline_successor,
            intervention_successor,
            baseline_personalities,
            intervention_personalities,
        )
        shared_domain = _require_shared_second_action_domain(
            baseline_successor, intervention_successor
        )
        if set(baseline_second_actions) != set(shared_domain["campIds"]):
            raise ValueError("fixed second action map does not cover the shared live camps")
        # Populate the intervention branch's successor lineage traits without
        # sampling another action or changing the fixed action map.
        _packed_step(intervention_successor, intervention_personalities)
        intervention_end = bridge.step(copy.deepcopy(baseline_second_actions))
        del intervention_generator  # constructed only to bind identical first-draw semantics
        step_count += 1
        intervention_second_economic = _economic_rows(bridge)

        successor_two = _successor_comparison(
            runtime,
            baseline_end,
            intervention_end,
            baseline_personalities,
            intervention_personalities,
        )

    if step_count != EXPECTED_BRIDGE_STEPS:
        raise RuntimeError("successor sensitivity probe bridge-step accounting changed")
    return {
        "randomness": {
            "pathReplicate": 0,
            "mainPersonalitySeed": main_seed,
            "actionSeed": action_seed,
            "personalitySeed": personality_seed,
            "identityScope": "existing single-state debug replicate-zero identities",
        },
        "openingPersonalityContext": opening_personality_context,
        "personalityContextScope": (
            "traits initialized by the frozen-state critic prediction and inherited "
            "by both branch clones; this RNG schedule differs from the saved smoke trace"
        ),
        "firstAction": {
            "baselineSha256": _canonical_sha256(baseline_first.actions),
            "interventionSha256": _canonical_sha256(intervention_actions),
            "receipts": receipts,
        },
        "secondAction": {
            "sha256": _canonical_sha256(baseline_second_actions),
            "sameCompleteActionMapAppliedToBothBranches": True,
            "legality": shared_domain,
        },
        "successorOne": {
            **successor_one,
            "physicalEvents": _physical_event_comparison(
                baseline_first_economic, intervention_first_economic
            ),
        },
        "successorTwoAfterIdenticalRequest": {
            **successor_two,
            "physicalEvents": _physical_event_comparison(
                baseline_second_economic, intervention_second_economic
            ),
        },
        "stepAccounting": {
            "nativeLabelProbe": 1,
            "baseline": 2,
            "intervention": 2,
            "actualBridgeStepCalls": step_count,
            "expectedBridgeStepCalls": EXPECTED_BRIDGE_STEPS,
        },
    }


def _write_result(path: Path, value: Json) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite successor sensitivity result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = {**value, "digest": digest(value)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(sealed, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_successor_sensitivity(
    freeze_root: Path, trace_root: Path, output: Path
) -> Json:
    """Run the fixed five-step probe.  No optimizer is constructed or stepped."""

    torch.set_num_threads(TORCH_NUM_THREADS)
    registry = load_endpoint_registry(freeze_root.resolve())
    receipt = next(
        (item for item in registry.endpoints if item.node_id == ENDPOINT_ID), None
    )
    if receipt is None:
        raise ProvenanceError("reviewed endpoint is absent from the frozen registry")
    runtime = load_endpoint_runtime(registry, receipt)
    if not runtime.vector_critic:
        raise ProvenanceError("successor sensitivity requires the vector checkpoint")
    reset_binding = _trace_reset_binding(
        trace_root.resolve(), receipt.checkpoint_sha256, registry.bridge_sha256
    )
    before = _model_identity(runtime)
    bridge: Model9Bridge | None = None
    try:
        dotnet = os.environ.get("ANTHROPOCENE_DOTNET", "dotnet")
        bridge = Model9Bridge(
            command=[dotnet, str(registry.bridge_path)], economic_diagnostics=True
        )
        current = bridge.reset(
            WORLD_SEED,
            movement=runtime.stage.bridge_movement,
            withheld=WITHHELD_MASK,
        )
        expected = cast(Json, reset_binding["nativeResetIdentity"])
        if int(current["turn"]) != EXPECTED_RESET_TURN or str(
            current["fingerprint"]
        ) != str(expected["fingerprint"]):
            raise ProvenanceError("live reset differs from the reviewed fixed state")
        probe = _run_bound_probe(bridge, runtime, current)
    finally:
        if bridge is not None:
            bridge.close()
    after = _model_identity(runtime)
    if after != before:
        raise RuntimeError("no-learning successor probe mutated actor or critic")
    result: Json = {
        "kind": RESULT_KIND,
        "schema": 1,
        "status": "complete-model-input-sensitivity-no-gate-claim",
        "scope": {
            "endpoint": ENDPOINT_ID,
            "worldSeed": WORLD_SEED,
            "regime": REGIME_ID,
            "decisionStep": DECISION_STEP,
            "pathReplicate": 0,
            "interpretation": (
                "model-input sensitivity and causal closed-loop effect of the first "
                "request; not value calibration or a proven historical cause"
            ),
        },
        "registry": {
            "checkpointSha256": receipt.checkpoint_sha256,
            "criticTypedContentSha256": receipt.critic_typed_content_sha256,
            "bridgeSha256": registry.bridge_sha256,
            "sourceHashes": dict(registry.source_hashes),
        },
        "traceResetBinding": reset_binding,
        "sourceCoordinateContract": {
            "requestedDevelopmentBudgetRawCoreIndex": REQUESTED_DEVELOPMENT_CORE_INDEX,
            "acceptedDevelopmentBudgetRawCoreIndex": ACCEPTED_DEVELOPMENT_CORE_INDEX,
            "realizedDevelopmentBudgetRawCoreIndex": REALIZED_DEVELOPMENT_CORE_INDEX,
            "previousDecisionRawCoreHalfOpenRange": [
                PREVIOUS_DECISION_CORE_START,
                PREVIOUS_DECISION_CORE_STOP,
            ],
            "maximumIndividualChangedRawCoreProbes": MAX_INDIVIDUAL_CORE_PROBES,
            "nativeCoreV2Mapping": "direct index-preserving copy into learner raw core",
            "swapPhase": "raw batch before normalizer packing",
            "packing": "symlog then checkpoint affine, optional guard/dead mask, present mask",
        },
        "runtime": {
            "torchNumThreads": torch.get_num_threads(),
            "bridgeEconomicDiagnostics": True,
        },
        "probe": probe,
        "noLearningMutationProof": {
            "optimizerConstructed": False,
            "optimizerStepCalled": False,
            "before": before,
            "after": after,
            "actorAndCriticUnchanged": True,
        },
        "claimsExcluded": [
            "value-calibration",
            "historical-causality",
            "gate-passage",
            "selection",
            "readiness",
            "independent-world-inference",
        ],
    }
    _write_result(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-package", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run_successor_sensitivity(args.freeze_package, args.trace_root, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
