import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7 import value_calibration as calibration
from anthropocene_rl.autonomous_v7.gate34_checkpoint_collector import BranchResult
from anthropocene_rl.autonomous_v7.gate34_continuation_collector import (
    CampReward,
    FrozenGeometricSchedule,
    RandomizedTailEstimate,
)
from anthropocene_rl.autonomous_v7.provenance import ProvenanceError, digest
from anthropocene_rl.autonomous_v7.value_anchor_collector import (
    FrozenStateCapture,
    build_native_source_arrays,
)

SHA = "a" * 64


def metadata(kind: str, unit_ids: list[str]) -> dict:
    policy_configuration = {
        "schemaVersion": 1,
        "samplingSemantics": "autonomous-v7-explore-and-sample-v1",
        "stage": {"name": "mixed"},
        "logStdFloor": -2.0,
        "personality": "default",
    }
    source_policy = digest(
        {
            "semantics": "effective-sampled-policy-digest-v1",
            "actorTensorDigest": SHA,
            "configuration": policy_configuration,
        }
    )
    unit_bindings = [
        {
            "unitId": unit_id,
            "worldSeed": 1000 + offset,
            "optimizerSeed": 70,
            "continuationSeeds": [50000 + offset],
            "parentIdentities": [f"world-parent-{offset}"],
            "branchIdentities": [f"baseline-{offset}"],
        }
        for offset, unit_id in enumerate(unit_ids)
    ]
    result = {
        "schemaVersion": 1,
        "kind": kind,
        "nativeContract": contracts.NATIVE_CONTRACT,
        "rewardSemantics": calibration.REWARD_SEMANTICS,
        "terminationSemantics": calibration.TERMINATION_SEMANTICS,
        "discountSemantics": calibration.DISCOUNT_SEMANTICS,
        "turnYears": contracts.TURN_YEARS,
        "horizonYears": contracts.HORIZON_YEARS,
        "policyDigestSemantics": "effective-sampled-policy-digest-v1",
        "truncated": False,
        "sourceCheckpointSha256": SHA,
        "sourcePolicyDigest": source_policy,
        "sourceActorTensorDigest": SHA,
        "sourcePolicyConfiguration": policy_configuration,
        "normalizationBinding": {
            "contractDigest": SHA,
            "stateDigest": SHA,
            "bindingDigest": SHA,
        },
        "simulator": {"sha256": SHA},
        "worldSeeds": [unit["worldSeed"] for unit in unit_bindings],
        "optimizerSeeds": [70],
        "continuationSeeds": [unit["continuationSeeds"][0] for unit in unit_bindings],
        "independentUnits": unit_bindings,
    }
    if kind == calibration.SOURCE_KIND:
        result["tailSemantics"] = "observed-through-terminal"
    else:
        result.update(
            {
                "tailSemantics": "fixed-prefix-plus-geometric-russian-roulette",
                "criticBootstrapUsed": False,
                "continuationPane": "sampled-training-policy",
                "discountedTailTerms": "discounted-from-original-frozen-state",
            }
        )
    return result


def write_terminal_source(path: Path, *, truncated: bool = False) -> None:
    n = 4
    arrays = {
        "metadata_json": np.asarray(
            json.dumps(metadata(calibration.SOURCE_KIND, ["world-a", "world-b"]))
        ),
        "sequence_id": np.asarray(["a", "a", "b", "b"]),
        "sequence_step": np.asarray([0, 1, 0, 1]),
        "years": np.full(n, contracts.TURN_YEARS),
        "terminated": np.asarray([False, not truncated, False, True]),
        "successor_index": np.asarray([1, -1, 3, -1]),
        "row_id": np.asarray([f"row-{i}" for i in range(n)]),
        "independent_unit_id": np.asarray(["world-a", "world-a", "world-b", "world-b"]),
        "rank_group_id": np.asarray(["fork", "", "fork", ""]),
        "rank_parent_digest": np.asarray([SHA, "", SHA, ""]),
        "prior_population": np.full(n, 10.0),
        "core": np.arange(n * 2, dtype=np.float32).reshape(n, 2),
    }
    for offset, name in enumerate(calibration.VALUE_HEADS):
        arrays[f"reward__{name}"] = np.asarray([1.0 + offset, 2.0, 3.0 + offset, 4.0])
    np.savez(path, **arrays)


def test_terminal_producer_computes_observed_discounted_returns_and_successor_ranking(tmp_path):
    source, output = tmp_path / "source.npz", tmp_path / "anchor.npz"
    write_terminal_source(source)
    calibration.build_terminal_dataset(source, output, "train-anchor")
    data = calibration.load_dataset(output, "train-anchor")
    gamma = math.exp(-contracts.TURN_YEARS / contracts.HORIZON_YEARS)
    assert data.targets_raw["welfare"][0] == pytest.approx(1.0 + gamma * 2.0)
    assert data.targets_raw["welfare"][2] == pytest.approx(3.0 + gamma * 4.0)
    assert data.ranking is not None
    assert data.ranking.successor_inputs["core"].tolist() == [[2.0, 3.0], [6.0, 7.0]]
    assert data.target_methods[calibration.GUARD_COST].tolist() == ["terminal-exact"] * 4


def test_terminal_producer_rejects_truncation_instead_of_zeroing_tail(tmp_path):
    source, output = tmp_path / "truncated.npz", tmp_path / "anchor.npz"
    write_terminal_source(source, truncated=True)
    with pytest.raises(ValueError, match="truncated"):
        calibration.build_terminal_dataset(source, output, "actor-gate")


def write_estimator_source(path: Path) -> None:
    meta = metadata(calibration.ESTIMATOR_SOURCE_KIND, ["world-1"])
    meta.update(
        {
            "schedules": [
                {
                    "scheduleId": "path-0",
                    "survivalProbability": 0.95,
                    "stopIndices": [0, 1],
                    "hardCapTerms": 2,
                    "frozenBeforeCollection": True,
                },
                {
                    "scheduleId": "path-1",
                    "survivalProbability": 0.95,
                    "stopIndices": [1, 0],
                    "hardCapTerms": 2,
                    "frozenBeforeCollection": True,
                },
            ],
            "tailAbsoluteBounds": {
                name: {"perTurnAbsoluteBound": 100.0, "provenanceId": f"bound-{name}"}
                for name in calibration.VALUE_HEADS
                if name != "trade_utility"
            },
            "rankWitnessSemantics": calibration.RANK_SEMANTICS,
        }
    )
    arrays = {
        "metadata_json": np.asarray(json.dumps(meta)),
        "row_id": np.asarray(["successor-a", "successor-b"]),
        "independent_unit_id": np.asarray(["world-1", "world-1"]),
        "survived_fixed_prefix": np.ones(2, bool),
        "prior_population": np.asarray([10.0, 20.0]),
        "core": np.asarray([[1.0], [2.0]], np.float32),
        "rank_group_id": np.asarray(["parent-1", "parent-1"]),
        "rank_parent_digest": np.asarray([SHA, SHA]),
        "rank_successor_digest": np.asarray(["b" * 64, "c" * 64]),
        "rank_discount": np.full(2, math.exp(-1.0 / 20.0)),
        "rank_terminal": np.zeros(2, bool),
    }
    for offset, name in enumerate(calibration.VALUE_HEADS):
        arrays[f"path_prefix__{name}"] = np.asarray(
            [[1.0 + offset, 1.5 + offset], [4.0 + offset, 4.5 + offset]]
        )
        arrays[f"path_discounted_tail_terms__{name}"] = np.full((2, 2, 2), 0.1 + offset)
        arrays[f"rank_immediate__{name}"] = np.asarray([0.25, 0.5])
    np.savez(path, **arrays)


def test_randomized_producer_uses_existing_tail_math_and_withholds_unbounded_trade(tmp_path):
    source, output = tmp_path / "rr-source.npz", tmp_path / "gate.npz"
    write_estimator_source(source)
    calibration.build_randomized_estimator_dataset(source, output, "actor-gate")
    data = calibration.load_dataset(output, "actor-gate")
    assert data.target_masks["welfare"].all()
    assert data.target_masks[calibration.GUARD_COST].all()
    assert not data.target_masks["trade_utility"].any()
    assert np.all(data.target_standard_errors["welfare"] >= 0)
    assert data.ranking is not None
    assert not data.ranking.available["trade_utility"].any()


def test_native_fork_adapter_retains_stopping_draws_inputs_guard_and_successor_ranks(
    tmp_path: Path,
):
    schedules = [
        FrozenGeometricSchedule(
            schedule_id=f"native-path-{path}",
            survival_probability=0.95,
            stop_indices=(0,) * 16,
            hard_cap_terms=200,
            rng_seed=700 + path,
            generator="synthetic-test",
        )
        for path in range(4)
    ]
    paths = []
    for path_index, schedule in enumerate(schedules):
        for alternative_index, alternative in enumerate(
            ("sampled-policy", "development-budget-native-v1")
        ):
            prefix = {
                name: {1: float(path_index + alternative_index + offset + 1)}
                for offset, name in enumerate(contracts.OBJECTIVES)
            }
            tails = {}
            one_step = {}
            for offset, name in enumerate(contracts.OBJECTIVES):
                if name == "trade_utility":
                    tails[name] = {
                        1: RandomizedTailEstimate(
                            "unresolved-absolute-integrability", schedule.schedule_id
                        )
                    }
                else:
                    value = float(offset + path_index + alternative_index + 0.25)
                    tails[name] = {
                        1: RandomizedTailEstimate(
                            "identified",
                            schedule.schedule_id,
                            estimate=value,
                            standard_error=0.0,
                            draw_estimates=(value,) * 16,
                            standard_error_scope="synthetic stopping-randomization",
                            stopping_randomization_contribution_to_standard_error=0.0,
                        )
                    }
                one_step[name] = {1: CampReward(1, 0, float(offset + 1), 20.0, False)}
            branch_rows = tuple(
                {
                    "camp": 1,
                    "turn": 9 + step,
                    "parent_society": 0,
                    "population": 20,
                    "years": 20.0,
                    "terminated": False,
                }
                for step in range(26)
            )
            paths.append(
                BranchResult(
                    alternative,
                    path_index,
                    schedule,
                    {name: values[1] for name, values in prefix.items()},
                    {name: values[1] for name, values in tails.items()},
                    prefix,
                    tails,
                    {name: 0.0 for name in contracts.OBJECTIVES},
                    (),
                    26,
                    1,
                    False,
                    None,
                    anchor_successor_inputs={
                        1: {
                            "core": np.asarray([10.0 + alternative_index], dtype=np.float32),
                            "prior_population": np.asarray(20.0),
                        }
                    },
                    anchor_branch_rows=branch_rows,
                    anchor_one_step_rewards=one_step,
                    anchor_successor_world_digest=(f"successor-{path_index}-{alternative_index}"),
                )
            )
    capture = FrozenStateCapture(
        1000,
        70,
        "default",
        5,
        14,
        SHA,
        (1,),
        {1: 9},
        {
            "core": np.asarray([[1.0]], dtype=np.float32),
            "prior_population": np.asarray([20.0]),
        },
        tuple(paths),
    )
    runtime = SimpleNamespace(
        reward_semantics="raw-stationary",
        receipt=SimpleNamespace(node_id="synthetic"),
        mix_change_cost=0.0,
        break_cost=0.0,
        personality_spec="default",
        stage=SimpleNamespace(commitments=False),
    )
    meta = metadata(calibration.ESTIMATOR_SOURCE_KIND, ["world-1000"])
    continuation = [schedule.rng_seed for schedule in schedules]
    meta.update(
        {
            "schedules": [
                {
                    "scheduleId": schedule.schedule_id,
                    "survivalProbability": schedule.survival_probability,
                    "stopIndices": list(schedule.stop_indices),
                    "hardCapTerms": schedule.hard_cap_terms,
                    "rngSeed": schedule.rng_seed,
                    "generator": schedule.generator,
                    "frozenBeforeCollection": True,
                }
                for schedule in schedules
            ],
            "tailAbsoluteBounds": {
                name: {"perTurnAbsoluteBound": 100.0, "provenanceId": f"bound-{name}"}
                for name in calibration.VALUE_HEADS
                if name != "trade_utility"
            },
            "rankWitnessSemantics": calibration.RANK_SEMANTICS,
            "continuationSeeds": continuation,
            "independentUnits": [
                {
                    "unitId": "world-1000",
                    "worldSeed": 1000,
                    "optimizerSeed": 70,
                    "continuationSeeds": continuation,
                    "parentIdentities": [SHA],
                    "branchIdentities": [f"branch-{i}" for i in range(8)],
                }
            ],
        }
    )
    arrays = build_native_source_arrays([capture], runtime, meta)
    assert arrays["path_tail_draws__welfare"].shape == (1, 4, 16)
    assert arrays["rank_input__core"].shape == (8, 1)
    source, output = tmp_path / "native-source.npz", tmp_path / "anchor.npz"
    np.savez_compressed(source, **arrays)
    calibration.build_randomized_estimator_dataset(source, output, "actor-gate")
    data = calibration.load_dataset(output, "actor-gate")
    assert data.target_masks[calibration.GUARD_COST].tolist() == [True]
    assert data.target_masks["welfare"].tolist() == [True]
    assert data.target_masks["trade_utility"].tolist() == [False]
    assert data.ranking is not None
    assert len(data.ranking.group_ids) == 8
    assert data.ranking.available[calibration.GUARD_COST].all()


def test_disjoint_units_are_required(tmp_path):
    source = tmp_path / "source.npz"
    write_estimator_source(source)
    train_path, gate_path = tmp_path / "train.npz", tmp_path / "gate.npz"
    calibration.build_randomized_estimator_dataset(source, train_path, "train-anchor")
    calibration.build_randomized_estimator_dataset(source, gate_path, "actor-gate")
    train = calibration.load_dataset(train_path, "train-anchor")
    gate = calibration.load_dataset(gate_path, "actor-gate")
    with pytest.raises(ValueError, match="share independent units"):
        calibration.validate_disjoint(train, gate)


def test_source_metadata_rejects_sealed_confirmation_seed():
    meta = metadata(calibration.SOURCE_KIND, ["sealed-world"])
    meta["worldSeeds"] = [3000]
    meta["independentUnits"][0]["worldSeed"] = 3000
    with pytest.raises(ProvenanceError, match="confirmation seed denied"):
        calibration._validate_source_metadata(meta)


def test_effective_policy_digest_changes_with_exploration_configuration():
    actor = torch.nn.Linear(2, 2)
    first = calibration.behavior_policy_digest(
        actor, {"stage": "mixed", "logStdFloor": -2.0, "personality": "default"}
    )
    second = calibration.behavior_policy_digest(
        actor, {"stage": "mixed", "logStdFloor": -3.0, "personality": "default"}
    )
    assert first != second
