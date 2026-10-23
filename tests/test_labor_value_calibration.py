from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7.labor_bridge import LABOR_REGISTRY_DIGEST
from anthropocene_rl.autonomous_v7.labor_model import (
    LABOR_COST_REFERENCE_SCALES,
    LABOR_MODEL_INPUT_CONTRACT,
    LABOR_OBJECTIVE_REFERENCE_SCALES,
    LaborPersonality,
)
from anthropocene_rl.autonomous_v7.labor_ppo import LaborActorGate
from anthropocene_rl.autonomous_v7.labor_value_calibration import (
    LABOR_ANCHOR_KIND,
    LABOR_RANK_SEMANTICS,
    LABOR_VALUE_HEADS,
    labor_anchor_loss,
    labor_calibration_report,
    labor_critic_digest,
    load_labor_anchor,
    main,
)
from anthropocene_rl.autonomous_v7.provenance import digest
from anthropocene_rl.autonomous_v7.value_calibration import CalibrationThresholds


class _TaskCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(2))

    def forward(self, tensors):
        values = tensors["predictions"]
        return {
            "qol_years": torch.stack(
                [values[:, 0] + self.offset[0], torch.zeros_like(values[:, 0])], -1
            ),
            "population_years": torch.stack(
                [values[:, 1] + self.offset[1], torch.zeros_like(values[:, 1])], -1
            ),
        }

    def expectation(self, objective: str, logits: torch.Tensor) -> torch.Tensor:
        del objective
        return logits[:, 0]

    def two_hot(self, objective: str, values: torch.Tensor) -> torch.Tensor:
        del objective
        result = torch.zeros(len(values), 2)
        result[:, 0] = 1.0
        return result


class _CostCritics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(2))

    def forward(self, tensors):
        values = tensors["predictions"]
        return {
            "feasibility": values[:, 2] + self.offset[0],
            "settlement_guard": values[:, 3] + self.offset[1],
        }


def _source() -> dict:
    actor_digest = "a" * 64
    configuration = {
        "laborRegistryDigest": LABOR_REGISTRY_DIGEST,
        "explorationFloor": -2.0,
        "personality": {"qol_years": 0.5, "population_years": 0.5},
        "enabledHeads": ["labor", "movement", "fission", "trade"],
    }
    return {
        "schemaVersion": 1,
        "kind": "native-labor-randomized-continuation-source-v1",
        "modelInputContract": LABOR_MODEL_INPUT_CONTRACT,
        "laborRegistryDigest": LABOR_REGISTRY_DIGEST,
        "criticBootstrapUsed": False,
        "truncatedTailAsZero": False,
        "returnEstimator": "fixed-prefix-plus-geometric-russian-roulette",
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
        "sourceCheckpointSha256": "f" * 64,
        "sourceActorTensorDigest": actor_digest,
        "sourcePolicyConfiguration": configuration,
        "sourcePolicyDigest": digest(
            {
                "semantics": "effective-sampled-policy-digest-v1",
                "actorTensorDigest": actor_digest,
                "configuration": configuration,
            }
        ),
        "normalizationBindingDigest": "d" * 64,
        "objectiveReferenceScales": dict(LABOR_OBJECTIVE_REFERENCE_SCALES),
        "costReferenceScales": dict(LABOR_COST_REFERENCE_SCALES),
        "randomizedTail": {
            "prefixTerms": 25,
            "drawCount": 16,
            "survivalProbability": 0.95,
            "hardCapTerms": 200,
            "scheduleDigest": "9" * 64,
            "absoluteIntegrabilityEstablished": True,
            "finiteVarianceEstablished": True,
            "droppedDraws": 0,
        },
        "worldSeeds": [50000, 50001],
        "optimizerSeeds": [70],
        "continuationSeeds": [80000, 80001, 80002, 80003],
        "independentUnits": [
            {
                "unitId": "world-50000",
                "worldSeed": 50000,
                "optimizerSeed": 70,
                "continuationSeeds": [80000, 80001],
            },
            {
                "unitId": "world-50001",
                "worldSeed": 50001,
                "optimizerSeed": 70,
                "continuationSeeds": [80002, 80003],
            },
        ],
    }


def _write(path: Path, *, duplicate_successors: bool = False) -> dict:
    source = _source()
    metadata = {
        "schemaVersion": 1,
        "kind": LABOR_ANCHOR_KIND,
        "role": "actor-gate",
        "valueHeads": list(LABOR_VALUE_HEADS),
        "rankSemantics": LABOR_RANK_SEMANTICS,
        "source": source,
    }
    predictions = np.asarray([[0.0, 0.0, 0.0, 0.0], [5.0, 5000.0, 0.0, 0.0]] * 2, np.float32)
    units = np.asarray(["world-50000", "world-50000", "world-50001", "world-50001"])
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        "row_id": np.asarray(["a", "b", "c", "d"]),
        "independent_unit_id": units,
        "input__predictions": predictions,
        "rank_input__predictions": predictions,
        "rank_group_id": np.asarray(["g0", "g0", "g1", "g1"]),
        "rank_parent_digest": np.asarray(["p0", "p0", "p1", "p1"]),
        "rank_successor_digest": np.asarray(
            ["s0", "s0" if duplicate_successors else "s1", "s2", "s3"]
        ),
        "rank_independent_unit_id": units,
        "rank_discount": np.ones(4, np.float64),
        "rank_terminal": np.zeros(4, bool),
    }
    targets = {
        "qol_years": predictions[:, 0].astype(np.float64),
        "population_years": predictions[:, 1].astype(np.float64),
        "feasibility": np.zeros(4, np.float64),
        "settlement_guard": np.zeros(4, np.float64),
    }
    for head, values in targets.items():
        arrays[f"target__{head}"] = values
        arrays[f"target_mask__{head}"] = np.ones(4, bool)
        arrays[f"target_standard_error__{head}"] = np.zeros(4, np.float64)
        arrays[f"target_method__{head}"] = np.full(4, "independent-randomized-tail-path-mean")
        arrays[f"rank_immediate__{head}"] = np.zeros(4, np.float64)
        arrays[f"rank_realized__{head}"] = values
        arrays[f"rank_standard_error__{head}"] = np.zeros(4, np.float64)
        arrays[f"rank_mask__{head}"] = np.ones(4, bool)
    np.savez_compressed(path, **arrays)
    return source


def test_four_head_report_opens_only_for_bound_current_policy_norm_registry(tmp_path: Path) -> None:
    path = tmp_path / "labor-anchor.npz"
    source = _write(path)
    data = load_labor_anchor(path, "actor-gate")
    task, costs = _TaskCritic(), _CostCritics()
    report = labor_calibration_report(
        task,
        costs,
        data,
        current_policy_digest=source["sourcePolicyDigest"],
        current_normalization_binding_digest=source["normalizationBindingDigest"],
        current_registry_digest=source["laborRegistryDigest"],
        thresholds=CalibrationThresholds(min_informative_rank_groups=2),
    )
    assert report["actorGateOpen"]
    assert report["sourceCriticDigest"] == labor_critic_digest(task, costs)
    assert LaborActorGate.from_report(report).source_critic_digest == report["sourceCriticDigest"]
    assert report["heads"]["feasibility"]["targetNearZero"]
    assert report["heads"]["feasibility"]["ranking"]["required"] is False
    assert report["heads"]["qol_years"]["ranking"]["informativeGroups"] == 2
    assert report["heads"]["qol_years"]["ranking"]["informativeIndependentUnits"] == 2
    assert report["worldSeeds"] == [50000, 50001]
    assert report["independentUnitIds"] == ["world-50000", "world-50001"]
    assert report["thresholds"]["min_informative_rank_groups"] == 2

    stale = labor_calibration_report(
        task,
        costs,
        data,
        current_policy_digest=source["sourcePolicyDigest"],
        current_normalization_binding_digest="0" * 64,
        current_registry_digest=source["laborRegistryDigest"],
        thresholds=CalibrationThresholds(min_informative_rank_groups=2),
    )
    assert stale["normalizationCurrent"] is False
    assert stale["actorGateOpen"] is False


def test_ranking_requires_distinct_successor_states(tmp_path: Path) -> None:
    path = tmp_path / "bad-rank.npz"
    _write(path, duplicate_successors=True)
    with pytest.raises(ValueError, match="distinct successor"):
        load_labor_anchor(path, "actor-gate")


def test_split_rank_groups_from_one_world_do_not_inflate_independent_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split-groups.npz"
    source = _write(path)
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    arrays["rank_independent_unit_id"] = np.full(4, "world-50000")
    np.savez_compressed(path, **arrays)
    report = labor_calibration_report(
        _TaskCritic(),
        _CostCritics(),
        load_labor_anchor(path, "actor-gate"),
        current_policy_digest=source["sourcePolicyDigest"],
        current_normalization_binding_digest=source["normalizationBindingDigest"],
        current_registry_digest=source["laborRegistryDigest"],
        thresholds=CalibrationThresholds(min_informative_rank_groups=2),
    )
    ranking = report["heads"]["qol_years"]["ranking"]
    assert ranking["informativeGroups"] == 2
    assert ranking["informativeIndependentUnits"] == 1
    assert report["heads"]["qol_years"]["pass"] is False
    assert report["actorGateOpen"] is False


def test_rank_group_cannot_mix_world_units(tmp_path: Path) -> None:
    path = tmp_path / "mixed-group.npz"
    _write(path)
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    arrays["rank_independent_unit_id"] = np.asarray(
        ["world-50000", "world-50001", "world-50001", "world-50001"]
    )
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="one independent world unit"):
        load_labor_anchor(path, "actor-gate")


def test_magnitude_metrics_weight_world_units_not_camp_rows(tmp_path: Path) -> None:
    path = tmp_path / "unequal-camps.npz"
    _write(path)
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    arrays["independent_unit_id"] = np.asarray(
        ["world-50000", "world-50000", "world-50000", "world-50001"]
    )
    arrays["target__qol_years"] = arrays["input__predictions"][:, 0].astype(np.float64)
    arrays["target__qol_years"][:3] -= 3.0
    np.savez_compressed(path, **arrays)
    source = _source()
    report = labor_calibration_report(
        _TaskCritic(),
        _CostCritics(),
        load_labor_anchor(path, "actor-gate"),
        current_policy_digest=source["sourcePolicyDigest"],
        current_normalization_binding_digest=source["normalizationBindingDigest"],
        current_registry_digest=source["laborRegistryDigest"],
        thresholds=CalibrationThresholds(min_informative_rank_groups=2),
    )
    observed = report["heads"]["qol_years"]["normalizedEqualIndependentUnitMeanAbsoluteError"]
    assert observed == pytest.approx(1.5 / LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"])


def test_anchor_loss_uses_independent_targets_for_all_four_heads(tmp_path: Path) -> None:
    path = tmp_path / "labor-anchor.npz"
    _write(path)
    data = load_labor_anchor(path, "actor-gate")
    loss, heads = labor_anchor_loss(
        _TaskCritic(), _CostCritics(), data, np.arange(len(data), dtype=np.int64)
    )
    assert torch.isfinite(loss)
    assert set(heads) == set(LABOR_VALUE_HEADS)


def test_anchor_loss_weights_independent_worlds_not_camp_rows(tmp_path: Path) -> None:
    path = tmp_path / "unequal-anchor-loss.npz"
    _write(path)
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    arrays["independent_unit_id"] = np.asarray(
        ["world-50000", "world-50000", "world-50000", "world-50001"]
    )
    arrays["input__predictions"][:, 2] = np.asarray([2.0, 2.0, 2.0, 0.0])
    arrays["target__feasibility"] = np.zeros(4, np.float64)
    np.savez_compressed(path, **arrays)
    _, heads = labor_anchor_loss(
        _TaskCritic(),
        _CostCritics(),
        load_labor_anchor(path, "actor-gate"),
        np.arange(4, dtype=np.int64),
    )
    # World 50000 has mean squared error four, world 50001 zero.
    assert float(heads["feasibility"].detach()) == pytest.approx(2.0)


def test_terminal_only_rows_do_not_open_living_state_actor_gate(tmp_path: Path) -> None:
    path = tmp_path / "terminal-only.npz"
    source = _write(path)
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    for head in LABOR_VALUE_HEADS:
        arrays[f"target_method__{head}"] = np.full(4, "terminal-exact")
    np.savez_compressed(path, **arrays)
    report = labor_calibration_report(
        _TaskCritic(),
        _CostCritics(),
        load_labor_anchor(path, "actor-gate"),
        current_policy_digest=source["sourcePolicyDigest"],
        current_normalization_binding_digest=source["normalizationBindingDigest"],
        current_registry_digest=source["laborRegistryDigest"],
        thresholds=CalibrationThresholds(min_informative_rank_groups=2),
    )
    assert report["actorGateOpen"] is False
    assert all(report["heads"][head]["livingContinuationUnits"] == 0 for head in LABOR_VALUE_HEADS)


def test_sealed_confirmation_world_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sealed.npz"
    _write(path)
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["source"]["worldSeeds"] = [3000, 50001]
    metadata["source"]["independentUnits"][0]["worldSeed"] = 3000
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    with pytest.raises(Exception, match="confirmation seed denied"):
        load_labor_anchor(path, "actor-gate")


class _FakeNormalizer:
    def state_dict(self) -> dict[str, int]:
        return {"fixture": 1}


def test_calibration_cli_loads_exact_checkpoint_and_writes_gate_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anthropocene_rl.autonomous_v7 import labor_training, labor_value_calibration

    checkpoint = tmp_path / "checkpoint.pt"
    anchor = tmp_path / "gate.npz"
    output = tmp_path / "gate.json"
    checkpoint.write_bytes(b"")
    anchor.write_bytes(b"")
    state = {
        "labor_model_contract": {"personality": LaborPersonality().state_dict()}
    }
    actor = nn.Linear(1, 1)
    task, costs = _TaskCritic(), _CostCritics()
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        labor_training,
        "_models_from_checkpoint",
        lambda loaded: ("layout", actor, task, costs, _FakeNormalizer()),
    )
    monkeypatch.setattr(labor_value_calibration, "load_labor_anchor", lambda path, role: object())

    captured: dict[str, object] = {}

    def fake_report(*args, **kwargs):
        captured.update(kwargs)
        return {
            "schemaVersion": 1,
            "kind": "model9-labor-value-calibration-report-v1",
            "localReadinessClaim": False,
            "actorGateOpen": False,
        }

    monkeypatch.setattr(labor_value_calibration, "labor_calibration_report", fake_report)
    main(
        [
            "--checkpoint",
            str(checkpoint),
            "--anchor",
            str(anchor),
            "--out",
            str(output),
            "--min-independent-units",
            "2",
        ]
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["localReadinessClaim"] is False
    assert captured["thresholds"].min_independent_units == 2
    assert captured["current_registry_digest"] == LABOR_REGISTRY_DIGEST
