"""Golden actor/heuristic/search equivalence in the pretrained action units."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_labor_bridge import _catalog_payload, _payload

from anthropocene_rl.autonomous_v7.labor_action_adapter import (
    action_adapter_contract,
    action_adapter_digest,
    actor_action_features,
    canonical_action_features,
)
from anthropocene_rl.autonomous_v7.labor_advantage import action_features, action_layout_state
from anthropocene_rl.autonomous_v7.labor_advantage_eval import _reordered
from anthropocene_rl.autonomous_v7.labor_bc import _retained_action, convert_heuristic_labor_label
from anthropocene_rl.autonomous_v7.labor_bridge import LaborCatalog
from anthropocene_rl.autonomous_v7.labor_search_payload import compiled_action


def _fixture(value=3.0, share=0.5):
    catalog = LaborCatalog.from_payload(_catalog_payload())
    action = {
        "schemaVersion": 12,
        **_payload(catalog.masks),
        "movement": -1,
        "fission": 1,
        "tradeGate": [2] * 7,
        "tradeValue": [value] * 7,
        "tradeShare": share,
    }
    retained = _retained_action(
        SimpleNamespace(action=action), catalog, torch.ones(1, 1, dtype=torch.bool)
    )
    return catalog, action, retained


@pytest.mark.parametrize("value", [0.25, 1.0, 1.5, 3.0, 4.0])
def test_actor_latents_and_heuristic_search_wire_units_have_same_features(value):
    catalog, action, retained = _fixture(value)
    actor = actor_action_features(
        retained.labor_sample,
        retained.nonlabor_sample,
        catalog,
        expected_layout=action_layout_state(),
    )
    # Search training's parser invokes this same historical encoder on the wire action.
    search = action_features(action)
    np.testing.assert_allclose(actor.numpy(), search, rtol=0, atol=1e-7)
    assert not actor.requires_grad
    # Native counts/worker-years are explanatory export fields, not encoder inputs.
    compiled = compiled_action(action, [1, 3] + [0] * 12, 20.0)
    assert compiled["specialty_counts"][1] == 3
    assert actor[1] == 0.75  # three out of four residents, not the raw number three
    reordered, specialties, activities = _reordered(action)
    np.testing.assert_array_equal(
        search, action_features(reordered, specialty_ids=specialties, activity_ids=activities)
    )


def test_different_wire_trade_values_are_not_arbitrarily_rescaled():
    catalog, action, retained = _fixture(3.0)
    encoded_three = canonical_action_features(action, catalog)
    other = dict(action, tradeValue=[1.5] * 7)
    encoded_one_half = canonical_action_features(other, catalog)
    assert not torch.equal(encoded_three, encoded_one_half)
    # An actor log-latent log(3) maps to the heuristic request 3 via exp, not x2.
    assert retained.nonlabor_sample["trade_value_latent"][0, 0] == pytest.approx(np.log(3.0))
    assert action_adapter_contract()["criticLayout"] == action_layout_state()
    assert len(action_adapter_digest()) == 64


def test_production_heuristic_count_and_worker_year_converter_matches_actor_units():
    from test_labor_bc_g1 import _catalog, _diag, _herding, _village

    catalog = _catalog(population=4)
    label = convert_heuristic_labor_label(catalog, _village(4), _diag(_herding(1, 20.0, 0.0, 5.0)))
    assert label.specialty_valid
    assert label.specialty_counts[1] == 3
    assert label.action["laborSpecialtyAllocation"][1] == 0.75
    assert label.action["laborActivityAllocation"][2][7] == 1.0
    retained = _retained_action(label, catalog, torch.ones(1, 1, dtype=torch.bool))
    actor = actor_action_features(retained.labor_sample, retained.nonlabor_sample, catalog)
    np.testing.assert_allclose(actor.numpy(), action_features(label.action), rtol=0, atol=1e-7)
    # Coverage/output differs from assigned worker-years; it must not rescale inputs.
    better_tools = convert_heuristic_labor_label(
        catalog, _village(4), _diag(_herding(1, 20.0, 0.0, 20.0))
    )
    np.testing.assert_array_equal(
        action_features(label.action), action_features(better_tools.action)
    )


@pytest.mark.parametrize("share", [0.0, 1.0])
def test_boundary_trade_shares_decode_with_declared_actor_transform(share):
    catalog, action, retained = _fixture()
    sample = dict(retained.nonlabor_sample)
    # Large finite logit reaches the float32 sigmoid boundary exactly.
    sample["trade_share_latent"] = torch.tensor([-100.0 if share == 0.0 else 100.0])
    features = actor_action_features(retained.labor_sample, sample, catalog)
    action["tradeShare"] = share
    np.testing.assert_array_equal(features.numpy(), action_features(action))


@pytest.mark.parametrize(
    "mutation", ["missing", "counts", "masked", "category", "bool_category", "nan", "schema"]
)
def test_adapter_rejects_wrong_units_masks_missing_and_invalid_inputs(mutation):
    catalog, action, _ = _fixture()
    action = copy.deepcopy(action)
    if mutation == "missing":
        del action["tradeShare"]
    elif mutation == "counts":
        action["laborSpecialtyAllocation"] = [1.0, 3.0] + [0.0] * 12
    elif mutation == "masked":
        action["laborActivityAllocation"][0] = [0.0] * 22
        action["laborActivityAllocation"][0][-1] = 1.0
    elif mutation == "category":
        action["tradeGate"][0] = 1.5
    elif mutation == "bool_category":
        action["tradeGate"][0] = True
    elif mutation == "nan":
        action["tradeValue"][0] = float("nan")
    else:
        action["schemaVersion"] = 99
    with pytest.raises(ValueError):
        canonical_action_features(action, catalog)


def test_pretrained_layout_mismatch_fails_closed():
    catalog, action, _ = _fixture()
    layout = action_layout_state()
    layout["tradeValueTransform"] = "unverified-multiply-by-two"
    with pytest.raises(ValueError, match="pretrained action layout"):
        canonical_action_features(action, catalog, expected_layout=layout)
