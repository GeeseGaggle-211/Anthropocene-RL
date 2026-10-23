from __future__ import annotations

import copy

import pytest
import torch
from test_autonomous_v7_round3_search import observation as legacy_observation
from test_labor_bridge import _attempt_payload, _catalog_payload

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7.bridge import observation_batch
from anthropocene_rl.autonomous_v7.labor_bridge import (
    ACTIVITY_ROWS,
    LABOR_ACTIVITY_FEATURE_WIDTH,
    LABOR_RESOURCE_FEATURE_WIDTH,
    LABOR_SPECIALTY_FEATURE_WIDTH,
    ROLE_ROWS,
    SPECIALTY_ROWS,
    LaborProducerContract,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    DEFAULT_LABOR_PERSONALITY,
    LABOR_ARCHITECTURE,
    LABOR_ARTIFACT_KIND,
    LABOR_COST_REFERENCE_SCALES,
    LABOR_COSTS,
    LABOR_MODEL_INPUT_CONTRACT,
    LABOR_OBJECTIVES,
    LaborActor,
    LaborCostCritics,
    LaborEncoderLayout,
    LaborModelContract,
    LaborPersonality,
    LaborTaskCritic,
    deterministic_labor_policy,
    deterministic_nonlabor_policy,
    labor_checkpoint_contract,
    labor_checkpoint_payload,
    labor_entropy,
    labor_log_prob,
    labor_policy_tensors,
    load_labor_checkpoint_models,
    nonlabor_entropy,
    nonlabor_log_prob,
    sample_labor_policy,
    sample_nonlabor_policy,
    schema12_initial_action,
    schema12_retry_action,
)
from anthropocene_rl.autonomous_v7.model import BINS, BOUNDED_FEATURE_CONTRACT, Normalizer


def test_labor_personality_is_fixed_two_objective_simplex() -> None:
    personality = LaborPersonality()
    assert personality.state_dict() == DEFAULT_LABOR_PERSONALITY
    assert personality.tensor().tolist() == [0.5, 0.5]
    combined = personality.combine(
        {
            "qol_years": torch.tensor([contracts.OBJECTIVE_REFERENCE_SCALES["qol_years"]]),
            "population_years": torch.tensor(
                [contracts.OBJECTIVE_REFERENCE_SCALES["population_years"]]
            ),
        }
    )
    assert combined.tolist() == pytest.approx([1.0])


@pytest.mark.parametrize(
    "value",
    [
        {"qol_years": 1.0, "population_years": 1.0},
        {"qol_years": -0.1, "population_years": 1.1},
        {"qol_years": True, "population_years": 0.0},
        {"qol_years": float("nan"), "population_years": 0.0},
        {"qol_years": 0.5, "population_years": 0.5, "welfare": 0.0},
    ],
)
def test_labor_personality_rejects_learned_or_malformed_weights(value: dict) -> None:
    with pytest.raises(ValueError):
        LaborPersonality.from_mapping(value)


def _checkpoint() -> dict:
    contract = LaborModelContract()
    layout = _layout()
    return {
        "artifact_kind": LABOR_ARTIFACT_KIND,
        "feature_contract": BOUNDED_FEATURE_CONTRACT,
        "model_input_contract": LABOR_MODEL_INPUT_CONTRACT,
        "labor_model_contract": contract.state_dict(),
        "native_producer_contract": LaborProducerContract().state_dict(),
        "labor_encoder_layout": layout.state_dict(),
    }


def test_labor_checkpoint_contract_is_explicit_and_rejects_legacy() -> None:
    parsed = labor_checkpoint_contract(_checkpoint())
    assert parsed.state_dict()["architecture"] == LABOR_ARCHITECTURE
    labor_checkpoint_contract(_checkpoint(), expected_layout=_layout())
    with pytest.raises(ValueError, match="encoder layout"):
        labor_checkpoint_contract(
            _checkpoint(),
            expected_layout=LaborEncoderLayout(
                11,
                LABOR_SPECIALTY_FEATURE_WIDTH,
                LABOR_ACTIVITY_FEATURE_WIDTH,
                LABOR_RESOURCE_FEATURE_WIDTH,
            ),
        )
    with pytest.raises(ValueError, match="not an explicit"):
        labor_checkpoint_contract({"actor": {}})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state.update(artifact_kind="model9"),
        lambda state: state.update(feature_contract="legacy-implicit-symlog-affine"),
        lambda state: state.update(model_input_contract="other"),
        lambda state: state.pop("native_producer_contract"),
        lambda state: state["labor_encoder_layout"].update(activity_feature_width=1),
        lambda state: state["native_producer_contract"].update(activityRows=15),
        lambda state: state["labor_model_contract"].update(objectives=["welfare"]),
        lambda state: state["labor_model_contract"].update(costCritics=["feasibility"]),
        lambda state: state["labor_model_contract"].update(extra=True),
    ],
)
def test_labor_checkpoint_contract_fails_closed(mutation) -> None:
    state = copy.deepcopy(_checkpoint())
    mutation(state)
    with pytest.raises(ValueError):
        labor_checkpoint_contract(state)


def test_labor_task_and_cost_critics_are_separate_and_strict() -> None:
    layout = _layout()
    task = LaborTaskCritic(layout, width=16, token_width=16)
    costs = LaborCostCritics(layout, width=12, token_width=16)
    inputs = _labor_tensors(batch=5)
    task_out = task(inputs)
    cost_out = costs(inputs)
    assert tuple(task_out) == LABOR_OBJECTIVES
    assert all(value.shape == (5, BINS) for value in task_out.values())
    assert tuple(cost_out) == LABOR_COSTS
    assert all(value.shape == (5,) for value in cost_out.values())
    assert not set(map(id, costs.encoders["feasibility"].parameters())).intersection(
        map(id, costs.encoders["settlement_guard"].parameters())
    )
    values = torch.tensor([0.0, 1.0, -1.0, 20.0, 100000.0])
    target = task.two_hot("population_years", values)
    assert target.shape == (5, BINS)
    assert torch.allclose(target.sum(dim=1), torch.ones(5))
    decoded = task.expectation("population_years", target.log().nan_to_num(-30.0))
    assert torch.isfinite(decoded).all()
    inputs["core"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        task(inputs)


def test_contract_does_not_modify_legacy_registries() -> None:
    assert LABOR_OBJECTIVES == ("qol_years", "population_years")
    assert contracts.OBJECTIVES == ("welfare", "population_years", "trade_utility", "qol_years")
    assert len(contracts.SETS) == 8
    assert LABOR_COST_REFERENCE_SCALES == {"feasibility": 1.0, "settlement_guard": 1.0}


def _layout() -> LaborEncoderLayout:
    return LaborEncoderLayout(
        10,
        LABOR_SPECIALTY_FEATURE_WIDTH,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
    )


def _labor_tensors(
    batch: int = 3, specialties: int = SPECIALTY_ROWS, activities: int = ACTIVITY_ROWS
) -> dict[str, torch.Tensor]:
    specialty = torch.ones(batch, specialties, dtype=torch.bool)
    activity = torch.zeros(batch, specialties - 1, activities, dtype=torch.bool)
    activity[:, :, 0] = True
    activity[:, :, 1:] = True
    tensors = {
        "core": torch.zeros(batch, 10),
        "labor_personality": torch.full((batch, 2), 0.5),
        "labor_specialty_features": torch.randn(batch, specialties, LABOR_SPECIALTY_FEATURE_WIDTH),
        "labor_specialty_present": torch.ones(batch, specialties, dtype=torch.bool),
        "labor_activity_features": torch.randn(
            batch, (specialties - 1) * activities, LABOR_ACTIVITY_FEATURE_WIDTH
        ),
        "labor_activity_present": torch.ones(
            batch, (specialties - 1) * activities, dtype=torch.bool
        ),
        "labor_resource_features": torch.randn(batch, 20, LABOR_RESOURCE_FEATURE_WIDTH),
        "labor_resource_present": torch.ones(batch, 20, dtype=torch.bool),
        "labor_specialty_mask": specialty,
        "labor_activity_mask": activity,
        "movement_mask": torch.ones(batch, contracts.MOVEMENT_SLOTS, dtype=torch.bool),
        "region_movement_mask": torch.ones(
            batch, contracts.REGION_MOVEMENT_SLOTS, dtype=torch.bool
        ),
    }
    for name, rows, width in contracts.SETS:
        tensors[name] = torch.zeros(batch, rows, width)
        tensors[name + "_present"] = torch.zeros(batch, rows, dtype=torch.bool)
    return tensors


def test_labor_actor_samples_and_scores_each_head_once_without_global_rng() -> None:
    actor = LaborActor(_layout(), width=20, token_width=16)
    tensors = _labor_tensors()
    output = actor(tensors)
    global_before = torch.random.get_rng_state().clone()
    first = sample_labor_policy(output, tensors, generator=torch.Generator().manual_seed(71))
    second = sample_labor_policy(output, tensors, generator=torch.Generator().manual_seed(71))
    assert torch.equal(global_before, torch.random.get_rng_state())
    assert torch.equal(first["specialty_allocation"], second["specialty_allocation"])
    assert torch.equal(first["activity_allocation"], second["activity_allocation"])
    assert torch.equal(first["specialty_mask"], tensors["labor_specialty_mask"])
    assert torch.allclose(first["specialty_allocation"].sum(dim=1), torch.ones(3))
    assert torch.allclose(first["activity_allocation"].sum(dim=2), torch.ones(3, ROLE_ROWS))
    score = labor_log_prob(output, first, tensors)
    entropy = labor_entropy(output, tensors)
    assert set(score) == {"specialty", "activity", "joint"}
    assert set(entropy) == set(score)
    assert torch.allclose(score["joint"], score["specialty"] + score["activity"])
    assert torch.allclose(entropy["joint"], entropy["specialty"] + entropy["activity"])


def test_deterministic_labor_action_obeys_masks_and_keeps_idle_explicit() -> None:
    actor = LaborActor(_layout(), width=8, token_width=16)
    tensors = _labor_tensors(1)
    tensors["labor_specialty_mask"][0, 3] = False
    tensors["labor_activity_mask"][0, 2, 1:] = False
    output = actor(tensors)
    sample = deterministic_labor_policy(output, tensors)
    assert sample.specialty_allocation[3] == 0.0
    assert sample.activity_allocation[2][0] == 1.0
    assert sum(sample.specialty_allocation) == pytest.approx(1.0)
    assert all(sum(row) == pytest.approx(1.0) for row in sample.activity_allocation)


def test_labor_actor_rejects_masks_that_hide_idle_or_expose_masked_roles() -> None:
    actor = LaborActor(_layout(), width=8, token_width=16)
    tensors = _labor_tensors(1)
    tensors["labor_specialty_mask"][0, 0] = False
    with pytest.raises(ValueError, match="idle"):
        actor(tensors)
    tensors = _labor_tensors(1)
    tensors["labor_specialty_mask"][0, 1] = False
    with pytest.raises(ValueError, match="masked specialties"):
        actor(tensors)


def test_two_coordinate_labor_log_prob_is_exact_alr_latent_density() -> None:
    tensors = _labor_tensors(1)
    tensors["labor_specialty_mask"][:, 2:] = False
    tensors["labor_activity_mask"][:, :, 1:] = False
    output = {
        "specialty_logits": torch.zeros(1, SPECIALTY_ROWS),
        "specialty_log_std": torch.zeros(1, SPECIALTY_ROWS),
        "activity_logits": torch.zeros(1, ROLE_ROWS, ACTIVITY_ROWS),
        "activity_log_std": torch.zeros(1, ROLE_ROWS, ACTIVITY_ROWS),
    }
    latent = torch.zeros(1, SPECIALTY_ROWS)
    latent[0, 1] = 0.7
    sample = {
        "specialty_allocation": torch.softmax(
            latent.masked_fill(~tensors["labor_specialty_mask"], -torch.inf), dim=1
        ),
        "activity_allocation": torch.zeros(1, ROLE_ROWS, ACTIVITY_ROWS),
        "specialty_latent": latent,
        "activity_latent": torch.zeros(1, ROLE_ROWS, ACTIVITY_ROWS),
        "specialty_mask": tensors["labor_specialty_mask"].clone(),
        "activity_mask": tensors["labor_activity_mask"].clone(),
    }
    sample["activity_allocation"][:, :, 0] = 1.0
    score = labor_log_prob(output, sample, tensors)
    expected = -0.5 * 0.7**2 - 0.5 * __import__("math").log(2 * __import__("math").pi)
    assert score["specialty"].item() == pytest.approx(expected, abs=1e-6)
    assert score["activity"].item() == 0.0


def test_dynamic_shared_scorers_support_new_tokens_and_are_permutation_equivariant() -> None:
    actor = LaborActor(_layout(), width=16, token_width=16).eval()
    tensors = _labor_tensors(1, specialties=16, activities=24)
    original = actor(tensors)
    # Keep global idle first and permute every specialty together with its
    # activity block. No learned output ordinal should change the paired score.
    role_permutation = torch.tensor([2, 0, 14, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    specialty_permutation = torch.cat([torch.tensor([0]), role_permutation + 1])
    changed = {key: value.clone() for key, value in tensors.items()}
    changed["labor_specialty_features"] = tensors["labor_specialty_features"][
        :, specialty_permutation
    ]
    changed["labor_specialty_present"] = tensors["labor_specialty_present"][
        :, specialty_permutation
    ]
    changed["labor_specialty_mask"] = tensors["labor_specialty_mask"][:, specialty_permutation]
    activity_features = tensors["labor_activity_features"].view(1, 15, 24, -1)
    changed["labor_activity_features"] = activity_features[:, role_permutation].reshape(
        1, 15 * 24, -1
    )
    changed["labor_activity_present"] = (
        tensors["labor_activity_present"].view(1, 15, 24)[:, role_permutation].reshape(1, 15 * 24)
    )
    changed["labor_activity_mask"] = tensors["labor_activity_mask"][:, role_permutation]
    permuted = actor(changed)
    assert torch.allclose(
        permuted["specialty_logits"],
        original["specialty_logits"][:, specialty_permutation],
        atol=1e-6,
    )
    assert torch.allclose(
        permuted["activity_logits"], original["activity_logits"][:, role_permutation], atol=1e-6
    )


def test_padded_absent_specialty_is_finite_and_does_not_change_active_tokens() -> None:
    actor = LaborActor(_layout(), width=16, token_width=16).eval()
    padded = _labor_tensors(1, specialties=16, activities=24)
    padded["labor_specialty_features"][:, -1].zero_()
    padded["labor_specialty_present"][:, -1] = False
    padded["labor_specialty_mask"][:, -1] = False
    activity = padded["labor_activity_features"].view(1, 15, 24, -1)
    activity[:, -1].zero_()
    padded["labor_activity_present"].view(1, 15, 24)[:, -1] = False
    padded["labor_activity_mask"][:, -1] = False
    truncated = {name: value.clone() for name, value in padded.items()}
    truncated["labor_specialty_features"] = padded["labor_specialty_features"][:, :-1]
    truncated["labor_specialty_present"] = padded["labor_specialty_present"][:, :-1]
    truncated["labor_specialty_mask"] = padded["labor_specialty_mask"][:, :-1]
    truncated["labor_activity_features"] = activity[:, :-1].reshape(1, 14 * 24, -1)
    truncated["labor_activity_present"] = (
        padded["labor_activity_present"].view(1, 15, 24)[:, :-1].reshape(1, 14 * 24)
    )
    truncated["labor_activity_mask"] = padded["labor_activity_mask"][:, :-1]
    with torch.no_grad():
        with_padding = actor(padded)
        without_padding = actor(truncated)
    assert all(
        torch.isfinite(value).all()
        for value in with_padding.values()
        if isinstance(value, torch.Tensor)
    )
    assert torch.allclose(
        with_padding["specialty_logits"][:, :-1], without_padding["specialty_logits"], atol=1e-6
    )
    assert torch.allclose(
        with_padding["activity_logits"][:, :-1], without_padding["activity_logits"], atol=1e-6
    )
    invalid = {name: value.clone() for name, value in padded.items()}
    invalid["labor_activity_mask"][:, -1, 0] = True
    with pytest.raises(ValueError, match="activity actions require present"):
        actor(invalid)


def test_schema12_nonlabor_union_sampling_scoring_and_payload_are_exact() -> None:
    actor = LaborActor(_layout(), width=16, token_width=16).eval()
    tensors = _labor_tensors(2)
    tensors["movement_mask"].zero_()
    tensors["movement_mask"][:, 0] = True
    tensors["movement_mask"][:, 3] = True
    tensors["region_movement_mask"].zero_()
    tensors["region_movement_mask"][:, 1] = True
    tensors["region_movement_mask"][:, 5] = True
    output = actor(tensors)
    assert output["movement_codes"].tolist() == [
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        -3,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
    ]
    legal_codes = output["movement_codes"][output["movement_union_mask"][0]].tolist()
    assert legal_codes == [-1, 2, -3, 103]
    global_before = torch.random.get_rng_state().clone()
    sample = sample_nonlabor_policy(output, generator=torch.Generator().manual_seed(191))
    repeated = sample_nonlabor_policy(output, generator=torch.Generator().manual_seed(191))
    assert torch.equal(global_before, torch.random.get_rng_state())
    assert all(torch.equal(sample[name], repeated[name]) for name in sample)
    assert set(sample["movement"].tolist()).issubset(set(legal_codes))
    score = nonlabor_log_prob(output, sample)
    entropy = nonlabor_entropy(output)
    components = {"movement", "fission", "trade_value", "trade_gate", "trade_share"}
    assert set(score) == components | {"joint"}
    assert set(entropy) == set(score)
    assert torch.allclose(score["joint"], sum(score[name] for name in components))
    labor = sample_labor_policy(output, tensors, generator=torch.Generator().manual_seed(192))
    initial = schema12_initial_action(labor, sample, 0)
    assert set(initial) == {
        "schemaVersion",
        "movement",
        "fission",
        "tradeValue",
        "tradeGate",
        "tradeShare",
        "laborSpecialtyAllocation",
        "laborActivityAllocation",
    }
    assert initial["schemaVersion"] == 12 and initial["movement"] in legal_codes
    assert set(schema12_retry_action(labor, 0)) == {
        "laborSpecialtyAllocation",
        "laborActivityAllocation",
    }
    deterministic = deterministic_nonlabor_policy(output)
    nonlabor_log_prob(output, deterministic)


def test_nonidle_permutation_preserves_idle_gauge_density() -> None:
    tensors = _labor_tensors(1)
    output = {
        "specialty_logits": torch.randn(1, SPECIALTY_ROWS),
        "specialty_log_std": torch.zeros(1, SPECIALTY_ROWS),
        "activity_logits": torch.randn(1, ROLE_ROWS, ACTIVITY_ROWS),
        "activity_log_std": torch.zeros(1, ROLE_ROWS, ACTIVITY_ROWS),
    }
    sample = sample_labor_policy(output, tensors, generator=torch.Generator().manual_seed(33))
    original = labor_log_prob(output, sample, tensors)["joint"]
    specialty_permutation = torch.tensor([0, 3, 1, 2, *range(4, SPECIALTY_ROWS)])
    activity_permutation = torch.tensor([0, 4, 1, 2, 3, *range(5, ACTIVITY_ROWS)])
    changed_tensors = {name: value.clone() for name, value in tensors.items()}
    for name in ("labor_specialty_features", "labor_specialty_present", "labor_specialty_mask"):
        changed_tensors[name] = tensors[name][:, specialty_permutation]
    role_permutation = specialty_permutation[1:] - 1
    activity_features = tensors["labor_activity_features"].view(1, ROLE_ROWS, ACTIVITY_ROWS, -1)
    changed_tensors["labor_activity_features"] = activity_features[:, role_permutation][
        :, :, activity_permutation
    ].reshape(1, ROLE_ROWS * ACTIVITY_ROWS, -1)
    activity_present = tensors["labor_activity_present"].view(1, ROLE_ROWS, ACTIVITY_ROWS)
    changed_tensors["labor_activity_present"] = activity_present[:, role_permutation][
        :, :, activity_permutation
    ].reshape(1, ROLE_ROWS * ACTIVITY_ROWS)
    changed_tensors["labor_activity_mask"] = tensors["labor_activity_mask"][:, role_permutation][
        :, :, activity_permutation
    ]
    changed_output = {
        "specialty_logits": output["specialty_logits"][:, specialty_permutation],
        "specialty_log_std": output["specialty_log_std"][:, specialty_permutation],
        "activity_logits": output["activity_logits"][:, role_permutation][
            :, :, activity_permutation
        ],
        "activity_log_std": output["activity_log_std"][:, role_permutation][
            :, :, activity_permutation
        ],
    }
    changed_sample = {
        "specialty_allocation": sample["specialty_allocation"][:, specialty_permutation],
        "activity_allocation": sample["activity_allocation"][:, role_permutation][
            :, :, activity_permutation
        ],
        "specialty_latent": sample["specialty_latent"][:, specialty_permutation],
        "activity_latent": sample["activity_latent"][:, role_permutation][
            :, :, activity_permutation
        ],
        "specialty_mask": sample["specialty_mask"][:, specialty_permutation],
        "activity_mask": sample["activity_mask"][:, role_permutation][:, :, activity_permutation],
    }
    assert torch.allclose(
        labor_log_prob(changed_output, changed_sample, changed_tensors)["joint"],
        original,
        atol=1e-6,
    )


def test_specialty_scores_receive_only_their_owned_activity_semantics() -> None:
    actor = LaborActor(_layout(), width=16, token_width=16).eval()
    tensors = _labor_tensors(1)
    tensors["labor_specialty_features"].zero_()
    tensors["labor_activity_features"].zero_()
    # Two otherwise identical craft-capable roles differ only in the physical
    # recipe/capacity feature carried by one owned activity token.
    first = 4 * ACTIVITY_ROWS + 11
    second = 5 * ACTIVITY_ROWS + 12
    tensors["labor_activity_features"][0, first, 15] = 1.0
    tensors["labor_activity_features"][0, second, 16] = 2.0
    output = actor(tensors)
    assert not torch.allclose(output["specialty_logits"][0, 5], output["specialty_logits"][0, 6])


def test_strict_json_to_bounded_tensor_to_policy_vertical_slice() -> None:
    raw = legacy_observation(seed=91)
    raw["core"][40:56] = [0.0] * 16
    raw["core"][144] = 0.0
    catalog = _catalog_payload()
    raw.update(laborCatalog=catalog, laborPrevious=None)
    village = {"observation": raw}
    normalizer = Normalizer.fit_bounded(observation_batch([village]))
    retry = {"laborRetry": _attempt_payload(catalog, rejected=True)}
    contexts, tensors = labor_policy_tensors([village], normalizer, retry_responses=[retry])
    assert contexts[0][2] is not None and not contexts[0][2].valid
    actor = LaborActor(
        LaborEncoderLayout(
            tensors["core"].shape[1],
            LABOR_SPECIALTY_FEATURE_WIDTH,
            LABOR_ACTIVITY_FEATURE_WIDTH,
            LABOR_RESOURCE_FEATURE_WIDTH,
        ),
        width=16,
        token_width=16,
    )
    output = actor(tensors)
    sample = deterministic_labor_policy(output, tensors)
    encoded = sample.payload()
    parsed = type(sample).from_payload(encoded, contexts[0][0].masks)
    assert parsed.payload() == encoded
    assert set(output) >= {
        "movement_union_logits",
        "movement_union_mask",
        "movement_codes",
        "fission",
        "trade_value_mean",
        "trade_gate",
        "trade_share_mean",
    }


def test_labor_checkpoint_creation_and_strict_restore_do_not_upgrade_legacy() -> None:
    raw = legacy_observation(seed=92)
    raw["core"][40:56] = [0.0] * 16
    raw["core"][144] = 0.0
    raw.update(laborCatalog=_catalog_payload(), laborPrevious=None)
    village = {"observation": raw}
    normalizer = Normalizer.fit_bounded(observation_batch([village]))
    _, tensors = labor_policy_tensors([village], normalizer)
    layout = LaborEncoderLayout(
        tensors["core"].shape[1],
        LABOR_SPECIALTY_FEATURE_WIDTH,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
    )
    actor = LaborActor(layout, width=16, token_width=16)
    task = LaborTaskCritic(layout, width=16, token_width=16)
    costs = LaborCostCritics(layout, width=16, token_width=16)
    state = labor_checkpoint_payload(actor, task, costs, normalizer, layout)
    assert set(state) == {
        "artifact_kind",
        "feature_contract",
        "model_input_contract",
        "labor_model_contract",
        "native_producer_contract",
        "labor_encoder_layout",
        "normalizer",
        "actor",
        "task_critic",
        "cost_critics",
    }
    restored = (
        LaborActor(layout, width=16, token_width=16),
        LaborTaskCritic(layout, width=16, token_width=16),
        LaborCostCritics(layout, width=16, token_width=16),
    )
    _, restored_normalizer = load_labor_checkpoint_models(state, *restored, expected_layout=layout)
    assert restored_normalizer.state_dict() == normalizer.state_dict()
    changed = copy.deepcopy(state)
    changed["actor"]["unexpected"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        load_labor_checkpoint_models(changed, *restored, expected_layout=layout)
    with pytest.raises(ValueError, match="explicit labor"):
        load_labor_checkpoint_models({"actor": {}}, *restored, expected_layout=layout)
