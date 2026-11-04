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

from anthropocene_rl.autonomous_v7 import contracts, labor_advantage
from anthropocene_rl.autonomous_v7.labor_action_adapter import (
    CATEGORICAL_HEADS,
    _decode_row,
    _labor_lists,
    action_adapter_contract,
    action_adapter_digest,
    actor_action_features,
    actor_action_features_batch,
    canonical_action_features,
    categorical_counterfactuals,
    movement_union_legal,
)
from anthropocene_rl.autonomous_v7.labor_advantage import (
    MOVEMENT_UNION_CODES,
    action_features,
    action_layout_state,
    block_slices,
    parse_candidate_shard,
    reparameterized_action_features,
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

BIND = {
    "expected_layout": action_layout_state(),
    "expected_adapter_digest": action_adapter_digest(),
}


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
    encoded_three = canonical_action_features(action, catalog, **BIND)
    other = dict(action, tradeValue=[1.5] * 7)
    encoded_one_half = canonical_action_features(other, catalog, **BIND)
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
    actor = actor_action_features(retained.labor_sample, retained.nonlabor_sample, catalog, **BIND)
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
    features = actor_action_features(retained.labor_sample, sample, catalog, **BIND)
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
        canonical_action_features(action, catalog, **BIND)


def test_pretrained_layout_mismatch_fails_closed():
    catalog, action, _ = _fixture()
    layout = action_layout_state()
    layout["tradeValueTransform"] = "unverified-multiply-by-two"
    with pytest.raises(ValueError, match="pretrained action layout"):
        canonical_action_features(
            action, catalog, expected_layout=layout, expected_adapter_digest=action_adapter_digest()
        )


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
            assert torch.equal(vector, canonical_action_features(wire, selected_catalog, **BIND))
            assert vector.shape == (349,) and vector.dtype == torch.float32
            assert vector.device.type == "cpu" and not vector.requires_grad
            vectors.append(vector)
        assert not torch.equal(vectors[0], vectors[1])
        retry_catalog = replace(catalog, population=7)
        retry_vector = actor_action_features(retry, nonlabor, retry_catalog, index=1, **BIND)
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
        actor_action_features(labor, nonlabor, catalog, index=index, **BIND)


@pytest.mark.parametrize("latent,wire_value", [(-100.0, 0.25), (100.0, 4.0)])
def test_finite_actor_trade_extremes_use_pretrained_clamp(latent, wire_value):
    catalog, action, retained = _fixture()
    sample = dict(retained.nonlabor_sample)
    sample["trade_value_latent"] = torch.full((1, 7), latent)
    action["tradeValue"] = [wire_value] * 7
    np.testing.assert_array_equal(
        actor_action_features(retained.labor_sample, sample, catalog, **BIND),
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
        canonical_action_features(action, catalog, **BIND).numpy(),
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
    adapted = canonical_action_features(h12, catalog, **BIND)
    np.testing.assert_array_equal(adapted.numpy(), search)
    np.testing.assert_array_equal(search, parsed.candidates[1].features)
    np.testing.assert_array_equal(
        actor_action_features(retained.labor_sample, retained.nonlabor_sample, catalog, **BIND),
        search,
    )
    assert not np.array_equal(search, action_features(h12))  # regression: omitted population
    np.testing.assert_array_equal(search[:14], (counts / catalog.population).astype(np.float32))
    activity = search[14:300].reshape(13, 22)
    for row, residents in zip(activity, counts[1:], strict=True):
        if residents == 0:
            np.testing.assert_array_equal(row, [1.0] + [0.0] * 21)


def test_zero_population_is_explicit_and_matches_reader_neutral_activity():
    catalog, action, retained = _fixture(population=0)
    result = canonical_action_features(action, catalog, **BIND)
    np.testing.assert_array_equal(result.numpy(), action_features(action, population=0))
    assert torch.count_nonzero(result[:14]) == 0
    expected_idle = torch.zeros(13, 22)
    expected_idle[:, 0] = 1
    assert torch.equal(result[14:300].reshape(13, 22), expected_idle)
    assert torch.equal(
        result,
        actor_action_features(retained.labor_sample, retained.nonlabor_sample, catalog, **BIND),
    )
    assert not torch.equal(
        result, canonical_action_features(action, replace(catalog, population=1), **BIND)
    )


@pytest.mark.parametrize(
    "population", [None, -1, True, np.bool_(False), 1.5, 3.0, float("inf"), float("nan"), "23"]
)
def test_invalid_population_cannot_silently_select_or_coerce_legacy_path(population):
    catalog, action, retained = _fixture(population=population)
    with pytest.raises(ValueError, match="population must be a nonnegative integer"):
        canonical_action_features(action, catalog, **BIND)
    with pytest.raises(ValueError, match="population must be a nonnegative integer"):
        actor_action_features(retained.labor_sample, retained.nonlabor_sample, catalog, **BIND)


def test_old_or_missing_checkpoint_binding_rejected_despite_identical_layout():
    catalog, action, retained = _fixture()
    # The pre-merge v2 contract repeated the specialty/unstaffed semantics now in
    # the layout; v1 also lacked population. Both differ from the current digest.
    pre_merge = action_adapter_contract()
    pre_merge["kind"] = "labor-canonical-actor-action-adapter-v2"
    pre_merge["specialty"] = "post-native-largest-remainder-apportionment"
    pre_merge["unstaffedActivity"] = "idle"
    v1 = {
        k: v
        for k, v in pre_merge.items()
        if k not in ("population", "specialty", "unstaffedActivity")
    }
    v1["kind"] = "labor-canonical-actor-action-adapter-v1"
    for old in (pre_merge, v1):
        assert old["criticLayout"] == action_layout_state()
        assert digest(old) != action_adapter_digest()
    for binding in (None, digest(pre_merge), digest(v1)):
        with pytest.raises(ValueError, match="adapter digest"):
            canonical_action_features(
                action,
                catalog,
                expected_layout=action_layout_state(),
                expected_adapter_digest=binding,
            )
        with pytest.raises(ValueError, match="adapter digest"):
            actor_action_features(
                retained.labor_sample,
                retained.nonlabor_sample,
                catalog,
                expected_layout=action_layout_state(),
                expected_adapter_digest=binding,
            )
    # A v1 critic layout (raw requested shares) is rejected before the digest.
    old_layout = dict(action_layout_state(), version="labor-action-semantic-vector-v1")
    del old_layout["specialtyEncoding"], old_layout["unstaffedActivity"]
    with pytest.raises(ValueError, match="pretrained action layout"):
        canonical_action_features(
            action,
            catalog,
            expected_layout=old_layout,
            expected_adapter_digest=action_adapter_digest(),
        )
    assert "specialty" not in action_adapter_contract()
    assert action_adapter_contract()["criticLayout"]["specialtyEncoding"] == (
        "post-native-largest-remainder-apportionment"
    )


# ---------------------------------------------------------------------------
# Batched encoding, mandatory binding, counterfactuals, reparameterized parity
# ---------------------------------------------------------------------------


def _actor_batch(batch, populations, *, seed=2, sample_seed=71, value_std_shift=0.0):
    """Real actor outputs and retained samples over one catalog's masks."""
    from test_labor_model import _labor_tensors, _layout

    catalog, _, _ = _fixture()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = LaborActor(_layout(), width=16, token_width=16)
        tensors = _labor_tensors(batch=batch)
    tensors["labor_specialty_mask"] = torch.tensor(catalog.masks.specialty).repeat(batch, 1)
    tensors["labor_activity_mask"] = torch.tensor(catalog.masks.activity).repeat(batch, 1, 1)
    with torch.no_grad():
        output = dict(actor(tensors))
    output["trade_value_log_std"] = output["trade_value_log_std"] + value_std_shift
    generator = torch.Generator().manual_seed(sample_seed)
    labor = sample_labor_policy(output, tensors, generator=generator)
    nonlabor = sample_nonlabor_policy(output, generator=generator)
    catalogs = [replace(catalog, population=int(p)) for p in populations]
    return catalogs, labor, nonlabor, output, tensors


def test_batch_api_is_bitwise_equal_to_per_row_and_to_the_wire_serializer():
    populations = [0, 1, 7, 23, 23, 100, 3, 48]
    catalogs, labor, nonlabor, _, _ = _actor_batch(len(populations), populations)
    batch = actor_action_features_batch(labor, nonlabor, catalogs, **BIND)
    assert batch.shape == (8, 349) and batch.dtype == torch.float32
    assert batch.device.type == "cpu" and not batch.requires_grad
    for index, catalog in enumerate(catalogs):
        # The adapter's decoder reproduces the actual wire serializer exactly.
        wire = schema12_initial_action(labor, nonlabor, index)
        assert _decode_row(_labor_lists(labor, slice(0, 8)), index, nonlabor, index) == wire
        per_row = actor_action_features(labor, nonlabor, catalog, index=index, **BIND)
        assert torch.equal(batch[index], per_row)
        np.testing.assert_array_equal(
            per_row.numpy(), action_features(wire, population=catalog.population)
        )
    # Per-row populations are honoured: the same retained row at another population differs.
    swapped = actor_action_features_batch(labor, nonlabor, catalogs[::-1], **BIND)
    assert not torch.equal(swapped[0], batch[0])
    empty = {k: v[:0] for k, v in labor.items()}
    empty_non = {k: (v if k == "movement_codes" else v[:0]) for k, v in nonlabor.items()}
    assert actor_action_features_batch(empty, empty_non, [], **BIND).shape == (0, 349)


def test_batch_validates_each_row_against_its_own_catalog_and_per_row_only_its_row():
    catalogs, labor, nonlabor, _, _ = _actor_batch(3, [5, 5, 5])
    with pytest.raises(ValueError, match="one catalog per row"):
        actor_action_features_batch(labor, nonlabor, catalogs[:2], **BIND)
    with pytest.raises(ValueError, match="one catalog per row"):
        actor_action_features_batch(labor, nonlabor, catalogs[0], **BIND)
    # Row 2's catalog masks one specialty the retained row-2 sample allowed.
    specialty = list(catalogs[2].masks.specialty)
    specialty[13] = False
    masked = replace(catalogs[2], masks=replace(catalogs[2].masks, specialty=tuple(specialty)))
    with pytest.raises(ValueError, match="specialty mask differs"):
        actor_action_features_batch(labor, nonlabor, [*catalogs[:2], masked], **BIND)
    actor_action_features(labor, nonlabor, catalogs[0], index=0, **BIND)
    with pytest.raises(ValueError, match="specialty mask differs"):
        actor_action_features(labor, nonlabor, masked, index=2, **BIND)
    # A bad value in another row neither costs nor fails the selected row in the
    # per-row path; the batch rejects it.
    bad = {k: v.clone() for k, v in nonlabor.items()}
    bad["trade_share_latent"][1] = float("nan")
    actor_action_features(labor, bad, catalogs[0], index=0, **BIND)
    with pytest.raises(ValueError, match="finite"):
        actor_action_features(labor, bad, catalogs[1], index=1, **BIND)
    with pytest.raises(ValueError, match="finite"):
        actor_action_features_batch(labor, bad, catalogs, **BIND)
    # Batch-wide structure is still checked by the per-row path.
    short = dict(nonlabor, fission=nonlabor["fission"][:2])
    with pytest.raises(ValueError, match="shape"):
        actor_action_features(labor, short, catalogs[0], index=0, **BIND)


def _entry_points():
    catalog, action, retained = _fixture()
    features = canonical_action_features(action, catalog, **BIND).unsqueeze(0)
    return {
        "canonical": lambda **kw: canonical_action_features(action, catalog, **kw),
        "per_row": lambda **kw: actor_action_features(
            retained.labor_sample, retained.nonlabor_sample, catalog, **kw
        ),
        "batch": lambda **kw: actor_action_features_batch(
            retained.labor_sample, retained.nonlabor_sample, [catalog], **kw
        ),
        "counterfactual": lambda **kw: categorical_counterfactuals(features, "fission", **kw),
    }


@pytest.mark.parametrize("entry", ["canonical", "per_row", "batch", "counterfactual"])
def test_binding_is_mandatory_on_every_public_entry_point(entry):
    call = _entry_points()[entry]
    assert call(**BIND) is not None
    for omitted in ({}, {"expected_layout": action_layout_state()}):
        with pytest.raises(TypeError, match="expected_"):
            call(**omitted)
    with pytest.raises(TypeError, match="expected_layout"):
        call(expected_adapter_digest=action_adapter_digest())
    with pytest.raises(ValueError, match="action layout is required"):
        call(expected_layout=None, expected_adapter_digest=action_adapter_digest())
    with pytest.raises(ValueError, match="adapter digest"):
        call(expected_layout=action_layout_state(), expected_adapter_digest=None)
    with pytest.raises(ValueError, match="adapter digest differs"):
        call(expected_layout=action_layout_state(), expected_adapter_digest="0" * 64)
    layout = dict(action_layout_state(), tradeValueTransform="unverified-multiply-by-two")
    with pytest.raises(ValueError, match="pretrained action layout differs"):
        call(expected_layout=layout, expected_adapter_digest=action_adapter_digest())


def _set_category(action, head, group, k):
    out = copy.deepcopy(action)
    if head == "movement":
        out["movement"] = MOVEMENT_UNION_CODES[k]
    elif head == "fission":
        out["fission"] = k
    else:
        out["tradeGate"][group] = k
    return out


def _counterfactual_fixtures():
    populations = [0, 1, 7, 23, 64]
    catalogs, labor, nonlabor, _, _ = _actor_batch(5, populations, value_std_shift=2.0)
    actions = [schema12_initial_action(labor, nonlabor, i) for i in range(5)]
    catalog, h12, _ = _fixture(population=23)
    h12["tradeGate"] = [0, 1, 2, 0, 1, 2, 1]
    h12["tradeValue"] = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    return [*actions, h12], [*catalogs, catalog]


def _encode_all(actions, catalogs):
    return torch.stack(
        [canonical_action_features(a, c, **BIND) for a, c in zip(actions, catalogs, strict=True)]
    )


_HEADS = [("movement", None), ("fission", None)] + [
    ("trade_gate", g) for g in range(len(contracts.TRADE_GROUPS))
]


@pytest.mark.parametrize("head,group", _HEADS)
def test_categorical_counterfactuals_equal_full_reencoding_bitwise(head, group):
    actions, catalogs = _counterfactual_fixtures()
    features = _encode_all(actions, catalogs)
    before = features.clone()
    out = categorical_counterfactuals(features, head, group=group, **BIND)
    categories = CATEGORICAL_HEADS[head][1]
    assert out.shape == (len(actions), categories, 349) and out.dtype == torch.float32
    for n, (action, catalog) in enumerate(zip(actions, catalogs, strict=True)):
        for k in range(categories):
            expected = canonical_action_features(
                _set_category(action, head, group, k), catalog, **BIND
            )
            assert torch.equal(out[n, k], expected), (head, group, n, k)
    assert torch.equal(features, before)  # inputs are not mutated


def test_movement_union_legality_follows_the_counterfactual_category_axis():
    _, _, nonlabor, _, _ = _actor_batch(3, [1, 1, 1])
    legal = movement_union_legal(nonlabor)
    assert legal.shape == (3, len(MOVEMENT_UNION_CODES)) and legal.dtype == torch.bool
    for row in range(3):
        for slot, code in enumerate(nonlabor["movement_codes"].tolist()):
            union = MOVEMENT_UNION_CODES.index(code)
            assert bool(legal[row, union]) == bool(nonlabor["movement_mask"][row, slot])
        assert legal[row, MOVEMENT_UNION_CODES.index(int(nonlabor["movement"][row]))]
    order = torch.arange(len(MOVEMENT_UNION_CODES) - 1, -1, -1)
    reordered = dict(
        nonlabor,
        movement_codes=nonlabor["movement_codes"][order],
        movement_mask=nonlabor["movement_mask"][:, order],
    )
    assert torch.equal(movement_union_legal(reordered), legal)


def test_counterfactuals_refuse_noncategorical_relaxed_and_coupled_heads(monkeypatch):
    actions, catalogs = _counterfactual_fixtures()
    features = _encode_all(actions, catalogs)
    for head in ("trade_value", "specialty_share", "trade_share"):
        with pytest.raises(ValueError, match="categorical heads"):
            categorical_counterfactuals(features, head, **BIND)
    with pytest.raises(ValueError, match="trade group"):
        categorical_counterfactuals(features, "trade_gate", **BIND)
    with pytest.raises(ValueError, match="trade group"):
        categorical_counterfactuals(features, "trade_gate", group=7, **BIND)
    with pytest.raises(ValueError, match="no trade group"):
        categorical_counterfactuals(features, "fission", group=0, **BIND)
    relaxed = features.clone()
    relaxed[:, block_slices()["fission_onehot"]] = 1.0 / 3.0
    with pytest.raises(ValueError, match="exact fission one-hot"):
        categorical_counterfactuals(relaxed, "fission", **BIND)
    with pytest.raises(ValueError, match="float32"):
        categorical_counterfactuals(features.double(), "fission", **BIND)

    # A coupled encoder (a closed gate suppressing its group's trade value) is detected.
    original = labor_advantage.action_features
    value_columns = block_slices()["trade_value_log"]

    def coupled(action, **kwargs):
        vector = original(action, **kwargs)
        if int(action["tradeGate"][0]) == 0:
            vector[value_columns.start] = 0.0
        return vector

    monkeypatch.setattr(labor_advantage, "action_features", coupled)
    with pytest.raises(ValueError, match="couples the trade_gate head"):
        categorical_counterfactuals(features, "trade_gate", group=0, **BIND)
    # Uncoupled heads and groups remain usable under the same encoder.
    categorical_counterfactuals(features, "trade_gate", group=1, **BIND)
    categorical_counterfactuals(features, "movement", **BIND)


def _replayed_reparameterized_draw(output, tensors, populations, seed):
    """Run reparameterized_action_features, then replay its generator consumption
    to build the retained sample of the same draw with the categoricals made hard.

    The reparameterized path draws, in order: specialty and activity ALR noise
    (exactly the draws of sample_labor_policy), Gumbel uniforms for movement,
    fission and the trade gates, then trade-value and trade-share Gaussian noise.
    """
    generator = torch.Generator().manual_seed(seed)
    state = generator.get_state()
    vector = reparameterized_action_features(
        output, tensors, generator=generator, population=populations
    )
    replay = torch.Generator()
    replay.set_state(state)
    labor = sample_labor_policy(output, tensors, generator=replay)

    def gumbel(logits, mask):
        uniform = torch.rand(logits.shape, dtype=logits.dtype, generator=replay)
        noisy = logits - torch.log(-torch.log(uniform.clamp(1e-10, 1.0 - 1e-10)))
        if mask is not None:
            noisy = noisy.masked_fill(~mask, -torch.inf)
        return torch.softmax(noisy, dim=-1), noisy.argmax(dim=-1)

    movement_soft, movement = gumbel(output["movement_union_logits"], output["movement_union_mask"])
    fission_soft, fission = gumbel(output["fission"], None)
    gate_soft, gate = gumbel(output["trade_gate"], None)
    value_noise = torch.randn(output["trade_value_mean"].shape, generator=replay)
    share_noise = torch.randn(output["trade_share_mean"].shape, generator=replay)
    nonlabor = {
        "movement_index": movement,
        "movement": output["movement_codes"][movement],
        "fission": fission,
        "trade_gate": gate,
        # Same expressions as sample_nonlabor_policy, on the replayed noise.
        "trade_value_latent": output["trade_value_mean"]
        + value_noise * output["trade_value_log_std"].exp(),
        "trade_share_latent": output["trade_share_mean"]
        + share_noise * output["trade_share_log_std"].exp(),
        "movement_mask": output["movement_union_mask"].clone(),
        "movement_codes": output["movement_codes"].clone(),
    }
    return vector, labor, nonlabor, (movement_soft, fission_soft, gate_soft)


def test_reparameterized_forward_matches_executed_encoding_block_by_block_on_a_shared_draw():
    populations = [0, 1, 2, 5, 7, 23, 23, 40, 64, 100, 3, 11] * 8
    batch = len(populations)
    catalogs, _, _, output, tensors = _actor_batch(batch, populations, value_std_shift=2.5)
    vector, labor, nonlabor, soft = _replayed_reparameterized_draw(
        output, tensors, populations, seed=11
    )
    blocks = block_slices()
    # The replay is aligned: identical relaxed categoricals (and labor latents below).
    for name, relaxed in zip(
        ("movement_onehot", "fission_onehot", "trade_gate_onehot"), soft, strict=True
    ):
        assert torch.equal(vector[:, blocks[name]], relaxed.reshape(batch, -1))
    executed = actor_action_features_batch(labor, nonlabor, catalogs, **BIND)

    # Specialty: post-apportionment shares agree bitwise on this draw.
    spec = blocks["specialty_share"]
    assert torch.equal(executed[:, spec], vector[:, spec])
    # Activity: unstaffed rows are bitwise neutral on both sides; staffed rows
    # differ only where the wire serializer re-derived its reference coordinate
    # (1 - fsum(others)): at most one coordinate per row, within 2.5e-7.
    a = executed[:, blocks["activity_share"]].reshape(batch, 13, 22)
    r = vector[:, blocks["activity_share"]].reshape(batch, 13, 22)
    staffed = torch.stack(
        [
            torch.as_tensor(labor_advantage.apportion_specialty_counts(row, p)[1:] > 0)
            for row, p in zip(
                labor["specialty_allocation"].double().numpy(), populations, strict=True
            )
        ]
    )
    assert bool((~staffed).any())
    assert torch.equal(a[~staffed], r[~staffed])
    assert float((a - r).abs().max()) <= 2.5e-7
    assert int(((a != r).sum(dim=-1) > 1).sum()) == 0
    # Trade value: log(clamp(exp(x))) vs clamp(x) agree within one float32 ulp;
    # clamped coordinates agree bitwise.
    value = blocks["trade_value_log"]
    value_a, value_r = executed[:, value], vector[:, value]
    latent = nonlabor["trade_value_latent"]
    clamped = (latent <= labor_advantage._LOG_TRADE_LOW) | (
        latent >= labor_advantage._LOG_TRADE_HIGH
    )
    assert bool(clamped.any()) and bool((~clamped).any())
    assert torch.equal(value_a[clamped], value_r[clamped])
    assert float((value_a - value_r).abs().max()) <= 1.2e-7
    # Trade share: the serializer's 0-d sigmoid vs the batched sigmoid kernel.
    share_a, share_r = executed[:, -1], vector[:, -1]
    assert float((share_a - share_r).abs().max()) <= 4e-9
    assert torch.equal(
        share_a, torch.stack([torch.sigmoid(x) for x in nonlabor["trade_share_latent"]])
    )
    # Categoricals: hard(argmax) of the relaxation is the executed one-hot.
    for name, width in (("movement_onehot", 0), ("fission_onehot", 0), ("trade_gate_onehot", 3)):
        relaxed = vector[:, blocks[name]]
        relaxed = relaxed.reshape(batch, -1, width) if width else relaxed.unsqueeze(1)
        hard = torch.nn.functional.one_hot(relaxed.argmax(-1), relaxed.shape[-1]).float()
        assert torch.equal(hard.reshape(batch, -1), executed[:, blocks[name]])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="defect: reparameterized categoricals forward soft Gumbel-softmax values",
)
def test_reparameterized_categorical_forward_value_is_the_executed_one_hot():
    populations = [7] * 16
    catalogs, _, _, output, tensors = _actor_batch(16, populations)
    vector, labor, nonlabor, _ = _replayed_reparameterized_draw(output, tensors, populations, 3)
    executed = actor_action_features_batch(labor, nonlabor, catalogs, **BIND)
    blocks = block_slices()
    columns = slice(blocks["movement_onehot"].start, blocks["trade_gate_onehot"].stop)
    assert torch.equal(vector[:, columns], executed[:, columns])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="defect: reparameterized specialty renormalizes off-tolerance float32 shares",
)
def test_reparameterized_specialty_apportionment_matches_executed_wire_at_near_ties():
    # Float32 shares 1.2e-7 off the simplex (inside the actor's 1e-6, outside the
    # wire's 1e-8): the reparameterized path renormalizes every coordinate, the
    # executed wire corrects only the last, and a one-resident apportionment flips.
    raw = torch.zeros(1, 14)
    raw[0, :3] = torch.tensor([0.14285710453987122, 0.42857152223587036, 0.42857155203819275])
    catalog, _, retained = _fixture(population=1)
    labor = dict(retained.labor_sample, specialty_allocation=raw)
    executed = actor_action_features_batch(labor, retained.nonlabor_sample, [catalog], **BIND)
    # reparameterized_action_features forwards apportioned_specialty of its
    # continuous float32 share (its own composition, called directly here).
    reparameterized = labor_advantage.apportioned_specialty(raw[0].double().numpy(), 1)
    np.testing.assert_array_equal(executed[0, :14].numpy(), reparameterized.astype(np.float32))
