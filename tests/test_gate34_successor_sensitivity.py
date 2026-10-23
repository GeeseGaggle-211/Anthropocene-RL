from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7 import gate34_successor_sensitivity as sensitivity
from anthropocene_rl.autonomous_v7.gate34_checkpoint_collector import SampledActions
from anthropocene_rl.autonomous_v7.model import Normalizer


class _TinyVectorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("bins", torch.tensor([-1.0, 0.0, 1.0]))

    def forward(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        requested = tensors["core"][:, sensitivity.REQUESTED_DEVELOPMENT_CORE_INDEX]
        logits = torch.stack((-requested, torch.zeros_like(requested), requested), dim=1)
        return {objective: logits for objective in contracts.OBJECTIVES}


def _normalizer(width: int = 60) -> Normalizer:
    result = Normalizer(
        core_mean=np.zeros(width, dtype=np.float32),
        core_std=np.full(width, 0.001, dtype=np.float32),
    )
    for name, _, set_width in contracts.SETS:
        result.set_mean[name] = np.zeros(set_width, dtype=np.float32)
        result.set_std[name] = np.ones(set_width, dtype=np.float32)
    return result


def _batch(*, requested: float, intervention_count: float, population: float) -> dict:
    rows, width = 1, 60
    batch: dict[str, np.ndarray] = {
        "core": np.zeros((rows, width), dtype=np.float32),
        "core_present": np.ones((rows, width), dtype=bool),
        "movement_mask": np.ones((rows, contracts.MOVEMENT_SLOTS), dtype=bool),
        "clothing_route_mask": np.ones((rows, 3), dtype=bool),
        "shelter_route_mask": np.ones((rows, 3), dtype=bool),
        "economy_mix_mask": np.ones((rows, len(contracts.ECONOMY_MIX)), dtype=bool),
        "region_movement_mask": np.ones(
            (rows, contracts.REGION_MOVEMENT_SLOTS), dtype=bool
        ),
        "commit_hold_mask": np.zeros(rows, dtype=bool),
        "personality": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "prior_population": np.asarray([population], dtype=np.float64),
    }
    batch["core"][:, sensitivity.REQUESTED_DEVELOPMENT_CORE_INDEX] = requested
    batch["core"][:, 53] = intervention_count
    for name, set_rows, set_width in contracts.SETS:
        batch[name] = np.zeros((rows, set_rows, set_width), dtype=np.float32)
        batch[name + "_present"] = np.zeros((rows, set_rows), dtype=bool)
    return batch


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        critic=_TinyVectorCritic(),
        normalizer=_normalizer(),
        objectives=contracts.OBJECTIVES,
        personality_spec="default",
        personality_drift=0.0,
    )


def _step(side: str) -> dict:
    return {
        "turn": 10,
        "fingerprint": side * 64,
        "villages": [
            {
                "camp": 1,
                "society": 101,
                "population": 10,
                "terminated": False,
                "observation": {"side": side},
            }
        ],
    }


def test_successor_comparison_swaps_raw_inputs_through_normal_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _batch(requested=0.0, intervention_count=0.0, population=10.0)
    intervention = _batch(requested=1.0, intervention_count=2.0, population=20.0)

    def packed(step: dict, _personalities: object) -> tuple[list[int], dict]:
        return [1], copy.deepcopy(baseline if step["fingerprint"][0] == "b" else intervention)

    monkeypatch.setattr(sensitivity, "_packed_step", packed)
    result = sensitivity._successor_comparison(
        _runtime(), _step("b"), _step("i"), object(), object()
    )

    assert result["changedRawCoreCoordinates"] == [47, 53]
    assert not any(
        row["input"] == "prior_population"
        for row in result["changedInputCoordinates"]
    )
    assert set(
        result["rawCoreHybridProbes"]["individualChangedRawCoreCoordinates"][
            "probes"
        ]
    ) == {"47", "53"}
    assert result["rawCoreHybridProbes"]["previousDecisionBlock40To55"][
        "rawCoreCoordinates"
    ] == list(range(40, 56))

    stage = result["requestedDevelopmentCore47InputStages"]["intervention"]
    assert stage["normalizer"] == {
        "mean": 0.0,
        "std": pytest.approx(0.001),
        "guard": None,
        "dead": False,
    }
    assert stage["rows"][0]["raw"] == 1.0
    assert stage["rows"][0]["symlogTransformed"] == pytest.approx(np.log(2.0))
    assert stage["rows"][0]["affineNormalizedBeforeGuard"] == pytest.approx(
        np.log(2.0) / 0.001
    )
    assert stage["rows"][0]["packedNormalized"] == pytest.approx(
        np.log(2.0) / 0.001, rel=1e-6
    )

    request_probe = result["rawCoreHybridProbes"]["requestedDevelopmentCore47"]
    before = result["values"]["baseline"]["welfare"]["1"]["perResidentValue"]
    swapped = request_probe["baselineWithInterventionCoordinates"]["welfare"]["1"]
    assert swapped["perResidentValue"] > before
    assert swapped["populationScaledRawValue"] == pytest.approx(
        swapped["perResidentValue"] * 10.0
    )
    assert not result["observations"]["allEqual"]
    assert "not simulator states" in result["swapInterpretation"]


def test_many_changed_core_coordinates_keep_group_probe_and_withhold_individuals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _batch(requested=0.0, intervention_count=0.0, population=10.0)
    intervention = copy.deepcopy(baseline)
    intervention["core"][:, :17] = 1.0

    def packed(step: dict, _personalities: object) -> tuple[list[int], dict]:
        return [1], copy.deepcopy(baseline if step["fingerprint"][0] == "b" else intervention)

    monkeypatch.setattr(sensitivity, "_packed_step", packed)
    result = sensitivity._successor_comparison(
        _runtime(), _step("b"), _step("i"), object(), object()
    )

    probes = result["rawCoreHybridProbes"]
    assert probes["allChangedRawCoreCoordinates"]["rawCoreCoordinates"] == list(
        range(17)
    )
    individual = probes["individualChangedRawCoreCoordinates"]
    assert individual["status"] == "withheld-too-many-changed-core-coordinates"
    assert individual["changedCoordinateCount"] == 17
    assert individual["probes"] == {}


def test_physical_event_comparison_ignores_branch_and_event_ids() -> None:
    baseline = [
        {
            "campId": 1,
            "physicalEvents": [
                {
                    "eventId": "baseline-1",
                    "branchId": "baseline",
                    "category": "flow",
                    "item": "Wood",
                    "unit": "units",
                    "direction": "in",
                    "phase": "economy",
                    "quantity": 2.0,
                },
                {
                    "eventId": "baseline-2",
                    "category": "flow",
                    "item": "Wood",
                    "unit": "units",
                    "direction": "in",
                    "phase": "economy",
                    "quantity": 3.0,
                },
            ],
        }
    ]
    intervention = copy.deepcopy(baseline)
    intervention[0]["physicalEvents"] = [
        {
            **baseline[0]["physicalEvents"][0],
            "eventId": "intervention",
            "branchId": "intervention",
            "quantity": 5.0,
        }
    ]

    equal = sensitivity._physical_event_comparison(baseline, intervention)
    assert equal["allGroupedQuantitiesEqual"]
    assert equal["baselineGroupCount"] == equal["interventionGroupCount"] == 1

    intervention[0]["physicalEvents"][0]["quantity"] = 4.5
    different = sensitivity._physical_event_comparison(baseline, intervention)
    assert not different["allGroupedQuantitiesEqual"]
    assert different["differingGroups"][0]["difference"] == -0.5


@pytest.mark.parametrize(
    ("baseline", "intervention", "message"),
    [
        (([1], {"movement_mask": [[True]]}), ([2], {"movement_mask": [[True]]}), "camp"),
        (([1], {"movement_mask": [[True]]}), ([1], {"movement_mask": [[False]]}), "legality"),
    ],
)
def test_second_action_domain_aborts_on_identity_or_mask_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    baseline: tuple[list[int], dict],
    intervention: tuple[list[int], dict],
    message: str,
) -> None:
    monkeypatch.setattr(
        sensitivity,
        "_policy_domain",
        lambda step: baseline if step["side"] == "baseline" else intervention,
    )
    with pytest.raises(ValueError, match=message):
        sensitivity._require_shared_second_action_domain(
            {"side": "baseline"}, {"side": "intervention"}
        )


class _FakeForkBridge:
    def __init__(self, current: dict) -> None:
        self.current = copy.deepcopy(current)
        self.restore_count = 0
        self.step_calls: list[tuple[int, dict]] = []
        self.economic_diagnostics = {
            1: {"campId": 1, "physicalEvents": []}
        }

    def fork(self) -> tuple[int, dict]:
        return 7, copy.deepcopy(self.current)

    def restore(self, _handle: int) -> dict:
        self.restore_count += 1
        return copy.deepcopy(self.current)

    def drop(self, _handle: int) -> None:
        return None

    def step(self, actions: dict) -> dict:
        self.step_calls.append((self.restore_count, copy.deepcopy(actions)))
        return _step("s")


def test_bound_probe_uses_five_steps_and_same_complete_second_action_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "turn": 9,
        "fingerprint": "f" * 64,
        "villages": [
            {
                "camp": 1,
                "society": 101,
                "parentSociety": 0,
                "population": 10,
                "terminated": False,
                "observation": {"opening": True},
            }
        ],
    }
    bridge = _FakeForkBridge(current)
    sampled = [
        SampledActions({1: {"action": "first"}}, {}, {}, {}),
        SampledActions({1: {"action": "second"}}, {}, {}, {}),
        SampledActions({1: {"action": "first"}}, {}, {}, {}),
    ]

    monkeypatch.setattr(sensitivity, "_critic_predictions", lambda *_args: ({}, {}))
    monkeypatch.setattr(
        sensitivity,
        "sample_training_actions",
        lambda *_args: sampled.pop(0),
    )

    def native_labels(fake: _FakeForkBridge, _state: dict) -> dict:
        fake.step({})
        return {1: {}}

    monkeypatch.setattr(sensitivity, "probe_native_labels", native_labels)
    monkeypatch.setattr(
        sensitivity,
        "apply_first_action_intervention",
        lambda _spec, *, camp_id, action, native_label, village: (
            {**action, "intervened": True},
            SimpleNamespace(
                before_sha256="a" * 64,
                after_sha256="b" * 64,
                changed_paths=("intervened",),
                exact_zero_change=False,
            ),
        ),
    )
    monkeypatch.setattr(
        sensitivity,
        "_successor_comparison",
        lambda *_args: {"comparison": True},
    )
    monkeypatch.setattr(
        sensitivity,
        "_require_shared_second_action_domain",
        lambda *_args: {"campIds": [1], "policyMasks": {}, "status": "identical"},
    )
    monkeypatch.setattr(
        sensitivity,
        "_packed_step",
        lambda *_args: ([1], {}),
    )

    result = sensitivity._run_bound_probe(bridge, _runtime(), current)

    assert result["stepAccounting"]["actualBridgeStepCalls"] == 5
    assert len(bridge.step_calls) == 5
    baseline_second = bridge.step_calls[2][1]
    intervention_second = bridge.step_calls[4][1]
    assert baseline_second == intervention_second == {1: {"action": "second"}}
    assert result["secondAction"]["sameCompleteActionMapAppliedToBothBranches"]
