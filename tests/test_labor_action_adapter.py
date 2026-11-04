"""Golden actor/heuristic/search equivalence in the pretrained action units."""

from __future__ import annotations

import copy
import math
from dataclasses import replace
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
from anthropocene_rl.autonomous_v7.labor_advantage import (
    action_features,
    action_layout_state,
    parse_candidate_shard,
    synthetic_candidate_shard,
)
from anthropocene_rl.autonomous_v7.labor_advantage_eval import _reordered
from anthropocene_rl.autonomous_v7.labor_bc import _retained_action, convert_heuristic_labor_label
from anthropocene_rl.autonomous_v7.labor_bridge import SIMPLEX_TOLERANCE, LaborCatalog
from anthropocene_rl.autonomous_v7.labor_model import (
    LaborActor,
    sample_labor_policy,
    sample_nonlabor_policy,
    schema12_initial_action,
    schema12_retry_action,
)
from anthropocene_rl.autonomous_v7.labor_search import compose_payload, labor_state
from anthropocene_rl.autonomous_v7.labor_search_payload import compiled_action
from anthropocene_rl.autonomous_v7.provenance import digest


def _fixture(value=3.0, share=0.5, population=4):
    catalog = replace(LaborCatalog.from_payload(_catalog_payload()), population=population)
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
        expected_adapter_digest=action_adapter_digest(),
    )
    # Search training's parser invokes this same historical encoder on the wire action.
    search = action_features(action, population=catalog.population)
    np.testing.assert_allclose(actor.numpy(), search, rtol=0, atol=1e-7)
    assert not actor.requires_grad
    # Native counts/worker-years are explanatory export fields, not encoder inputs.
    compiled = compiled_action(action, [1, 3] + [0] * 12, 20.0)
    assert compiled["specialty_counts"][1] == 3
    assert actor[1] == 0.75  # three out of four residents, not the raw number three
    reordered, specialties, activities = _reordered(action)
    np.testing.assert_array_equal(
        search,
        action_features(
            reordered,
            specialty_ids=specialties,
            activity_ids=activities,
            population=catalog.population,
        ),
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
    np.testing.assert_allclose(
        actor.numpy(),
        action_features(label.action, population=catalog.population),
        rtol=0,
        atol=1e-7,
    )
    # Coverage/output differs from assigned worker-years; it must not rescale inputs.
    better_tools = convert_heuristic_labor_label(
        catalog, _village(4), _diag(_herding(1, 20.0, 0.0, 20.0))
    )
    np.testing.assert_array_equal(
        action_features(label.action, population=catalog.population),
        action_features(better_tools.action, population=catalog.population),
    )


@pytest.mark.parametrize("share", [0.0, 1.0])
def test_boundary_trade_shares_decode_with_declared_actor_transform(share):
    catalog, action, retained = _fixture()
    sample = dict(retained.nonlabor_sample)
    # Large finite logit reaches the float32 sigmoid boundary exactly.
    sample["trade_share_latent"] = torch.tensor([-100.0 if share == 0.0 else 100.0])
    features = actor_action_features(retained.labor_sample, sample, catalog)
    action["tradeShare"] = share
    np.testing.assert_array_equal(
        features.numpy(), action_features(action, population=catalog.population)
    )


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


def test_actual_actor_sampler_batch_alignment_retry_and_gradient_rng_isolation():
    from test_labor_model import _labor_tensors, _layout

    catalog, _, _ = _fixture()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        actor = LaborActor(_layout(), width=16, token_width=16)
        tensors = _labor_tensors(batch=3)
        tensors["labor_specialty_mask"] = torch.tensor(catalog.masks.specialty).repeat(3, 1)
        tensors["labor_activity_mask"] = torch.tensor(catalog.masks.activity).repeat(3, 1, 1)
        output = actor(tensors)
        generator = torch.Generator().manual_seed(71)
        labor = sample_labor_policy(output, tensors, generator=generator)
        nonlabor = sample_nonlabor_policy(output, generator=generator)
        retry = sample_labor_policy(output, tensors, generator=generator)
        # Real float32 samples can exceed the strict wire tolerance. The
        # serializer's one-coordinate roundoff correction must remain usable.
        assert any(
            abs(math.fsum(row) - 1.0) > SIMPLEX_TOLERANCE
            for row in labor["specialty_allocation"].detach().tolist()
        )
        global_rng, sample_rng = torch.random.get_rng_state().clone(), generator.get_state().clone()
        originals = {
            group: {k: v.detach().clone() for k, v in values.items()}
            for group, values in (("labor", labor), ("nonlabor", nonlabor))
        }
        vectors = []
        for index in range(3):
            selected_catalog = replace(catalog, population=(1, 7, 23)[index])
            vector = actor_action_features(
                labor,
                nonlabor,
                selected_catalog,
                index=index,
                expected_layout=action_layout_state(),
                expected_adapter_digest=action_adapter_digest(),
            )
            wire = schema12_initial_action(labor, nonlabor, index)
            np.testing.assert_array_equal(
                vector.numpy(), action_features(wire, population=selected_catalog.population)
            )
            assert torch.equal(vector, canonical_action_features(wire, selected_catalog))
            assert vector.shape == (349,) and vector.dtype == torch.float32
            assert vector.device.type == "cpu" and not vector.requires_grad
            vectors.append(vector)
        assert not torch.equal(vectors[0], vectors[1])
        retry_catalog = replace(catalog, population=7)
        retry_vector = actor_action_features(retry, nonlabor, retry_catalog, index=1)
        retry_wire = dict(
            schema12_initial_action(labor, nonlabor, 1), **schema12_retry_action(retry, 1)
        )
        np.testing.assert_array_equal(
            retry_vector.numpy(), action_features(retry_wire, population=retry_catalog.population)
        )
        assert torch.equal(retry_vector[300:], vectors[1][300:])
        assert not torch.equal(retry_vector[:300], vectors[1][:300])
        assert torch.equal(global_rng, torch.random.get_rng_state())
        assert torch.equal(sample_rng, generator.get_state())
        for group, values in (("labor", labor), ("nonlabor", nonlabor)):
            assert all(torch.equal(v, originals[group][k]) for k, v in values.items())
        assert all(p.grad is None for p in actor.parameters())


@pytest.mark.parametrize(
    "mutation",
    [
        "fractional_movement",
        "fractional_fission",
        "fractional_gate",
        "boolean_gate",
        "infinite_value",
        "infinite_share",
        "nan_allocation",
        "infinite_labor_latent",
        "raw_counts",
        "bad_activity_mass",
        "catalog_mask",
        "movement_mask",
        "movement_code",
        "movement_index",
        "gate_range",
        "fission_range",
        "batch_mismatch",
        "value_width",
        "missing",
        "negative_index",
        "bool_index",
    ],
)
def test_actor_samples_are_validated_before_lossy_wire_decoding(mutation):
    catalog, _, retained = _fixture()
    labor = {k: v.clone() for k, v in retained.labor_sample.items()}
    nonlabor = {k: v.clone() for k, v in retained.nonlabor_sample.items()}
    index = 0
    if mutation.startswith("fractional_"):
        key = {"movement": "movement", "fission": "fission", "gate": "trade_gate"}[
            mutation.split("_")[1]
        ]
        nonlabor[key] = torch.full_like(nonlabor[key], 1.5, dtype=torch.float32)
    elif mutation == "boolean_gate":
        nonlabor["trade_gate"] = nonlabor["trade_gate"].bool()
    elif mutation == "infinite_value":
        nonlabor["trade_value_latent"][0, 0] = float("inf")
    elif mutation == "infinite_share":
        nonlabor["trade_share_latent"][0] = -float("inf")
    elif mutation == "nan_allocation":
        labor["specialty_allocation"][0, 0] = float("nan")
    elif mutation == "infinite_labor_latent":
        labor["specialty_latent"][0, 0] = float("inf")
    elif mutation == "raw_counts":
        labor["specialty_allocation"][0, :2] = torch.tensor([1.0, 3.0])
    elif mutation == "bad_activity_mass":
        labor["activity_allocation"][0, 0] *= 2
    elif mutation == "catalog_mask":
        labor["specialty_mask"][0, 0] = False
    elif mutation == "movement_mask":
        nonlabor["movement_mask"][:] = False
    elif mutation == "movement_code":
        nonlabor["movement"][0] = 0
    elif mutation == "movement_index":
        nonlabor["movement_index"][0] = 999
    elif mutation == "gate_range":
        nonlabor["trade_gate"][0, 0] = 3
    elif mutation == "fission_range":
        nonlabor["fission"][0] = 3
    elif mutation == "batch_mismatch":
        nonlabor["fission"] = nonlabor["fission"].repeat(2)
    elif mutation == "value_width":
        nonlabor["trade_value_latent"] = nonlabor["trade_value_latent"][:, :6]
    elif mutation == "missing":
        del nonlabor["trade_share_latent"]
    elif mutation == "negative_index":
        index = -1
    else:
        index = True
    with pytest.raises(ValueError):
        actor_action_features(labor, nonlabor, catalog, index=index)


@pytest.mark.parametrize("latent,wire_value", [(-100.0, 0.25), (100.0, 4.0)])
def test_finite_actor_trade_extremes_use_pretrained_clamp(latent, wire_value):
    catalog, action, retained = _fixture()
    sample = dict(retained.nonlabor_sample)
    sample["trade_value_latent"] = torch.full((1, 7), latent)
    action["tradeValue"] = [wire_value] * 7
    np.testing.assert_array_equal(
        actor_action_features(retained.labor_sample, sample, catalog),
        action_features(action, population=catalog.population),
    )


def test_trade_group_stable_ids_restore_canonical_feature_order():
    catalog, action, _ = _fixture()
    action["tradeValue"] = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    action["tradeGate"] = [0, 1, 2, 0, 1, 2, 1]
    groups = action_layout_state()["tradeGroups"]
    order = [6, 2, 0, 5, 4, 1, 3]
    reordered = dict(
        action,
        tradeGate=[action["tradeGate"][i] for i in order],
        tradeValue=[action["tradeValue"][i] for i in order],
    )
    np.testing.assert_array_equal(
        canonical_action_features(action, catalog).numpy(),
        action_features(
            reordered, trade_groups=[groups[i] for i in order], population=catalog.population
        ),
    )


@pytest.mark.parametrize("population_source", ["root", "compiled"])
def test_h12_nonintegral_staffing_matches_actual_search_reader_bitwise(population_source):
    catalog, h12, retained = _fixture(population=23, value=1.0)
    h12["laborActivityAllocation"][0] = [0.25, 0.75] + [0.0] * 20
    retained = _retained_action(
        SimpleNamespace(action=h12), catalog, torch.ones(1, 1, dtype=torch.bool)
    )
    # These are valid requested shares, not multiples of one resident at P=23.
    assert h12["laborSpecialtyAllocation"][1] * catalog.population == 17.25
    counts, _ = labor_state(h12, catalog)
    twin = compose_payload(h12, catalog)
    shard = synthetic_candidate_shard(
        worlds=(90100,), roots_per_world=1, candidates=2, horizons=(1,), streams=0
    )
    root = shard["roots"][0]
    root["candidates"] = root["candidates"][:2]
    if population_source == "root":
        root["population"] = catalog.population
    for candidate, action in zip(root["candidates"], (h12, twin), strict=True):
        candidate["action"] = action
        candidate["bookkeeping_twin"] = None
        candidate["compiled"] = compiled_action(action, counts, catalog.turn_years)
    parsed = parse_candidate_shard(shard).roots[0]
    assert parsed.population == catalog.population
    search = parsed.candidates[0].features
    adapted = canonical_action_features(h12, catalog)
    np.testing.assert_array_equal(adapted.numpy(), search)
    np.testing.assert_array_equal(search, parsed.candidates[1].features)
    np.testing.assert_array_equal(
        actor_action_features(retained.labor_sample, retained.nonlabor_sample, catalog), search
    )
    assert not np.array_equal(search, action_features(h12))  # regression: omitted population
    np.testing.assert_array_equal(search[:14], (counts / catalog.population).astype(np.float32))
    activity = search[14:300].reshape(13, 22)
    for row, residents in zip(activity, counts[1:], strict=True):
        if residents == 0:
            np.testing.assert_array_equal(row, [1.0] + [0.0] * 21)


def test_zero_population_is_explicit_and_matches_reader_neutral_activity():
    catalog, action, retained = _fixture(population=0)
    result = canonical_action_features(action, catalog)
    np.testing.assert_array_equal(result.numpy(), action_features(action, population=0))
    assert torch.count_nonzero(result[:14]) == 0
    expected_idle = torch.zeros(13, 22)
    expected_idle[:, 0] = 1
    assert torch.equal(result[14:300].reshape(13, 22), expected_idle)
    assert torch.equal(
        result, actor_action_features(retained.labor_sample, retained.nonlabor_sample, catalog)
    )
    assert not torch.equal(
        result, canonical_action_features(action, replace(catalog, population=1))
    )


@pytest.mark.parametrize(
    "population", [None, -1, True, np.bool_(False), 1.5, 3.0, float("inf"), float("nan"), "23"]
)
def test_invalid_population_cannot_silently_select_or_coerce_legacy_path(population):
    catalog, action, retained = _fixture(population=population)
    with pytest.raises(ValueError, match="population must be a nonnegative integer"):
        canonical_action_features(action, catalog)
    with pytest.raises(ValueError, match="population must be a nonnegative integer"):
        actor_action_features(retained.labor_sample, retained.nonlabor_sample, catalog)


def test_old_or_missing_checkpoint_binding_rejected_despite_identical_layout():
    catalog, action, retained = _fixture()
    old = action_adapter_contract()
    old["kind"] = "labor-canonical-actor-action-adapter-v1"
    for key in ("population", "specialty", "unstaffedActivity"):
        del old[key]
    assert old["criticLayout"] == action_layout_state()
    assert digest(old) != action_adapter_digest()
    for binding in (None, digest(old)):
        with pytest.raises(ValueError, match="adapter digest"):
            canonical_action_features(
                action,
                catalog,
                expected_layout=old["criticLayout"],
                expected_adapter_digest=binding,
            )
        with pytest.raises(ValueError, match="adapter digest"):
            actor_action_features(
                retained.labor_sample,
                retained.nonlabor_sample,
                catalog,
                expected_layout=old["criticLayout"],
                expected_adapter_digest=binding,
            )
    assert torch.equal(
        canonical_action_features(action, catalog),
        canonical_action_features(
            action,
            catalog,
            expected_layout=action_layout_state(),
            expected_adapter_digest=action_adapter_digest(),
        ),
    )
