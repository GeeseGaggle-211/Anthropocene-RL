"""Intervention A (learner): privileged critic encoders and checkpoint contract.

Actor outputs are bitwise invariant under privilege-only perturbation, the
three critics respond to it, the actor encoder class refuses a privileged
block, old checkpoints keep their bytes, schema-version mismatches are rejected
at load and the strict-load migration hook is explicit.
"""

from __future__ import annotations

import copy

import pytest
import torch
from test_autonomous_v7_round3_search import observation as legacy_observation
from test_labor_bridge import _catalog_payload
from test_labor_model import _labor_tensors
from test_labor_privileged_bridge import LAYOUT, _normalizer, _raw, _view

from anthropocene_rl.autonomous_v7.bridge import observation_batch, privileged_batch
from anthropocene_rl.autonomous_v7.labor_bridge import (
    LABOR_ACTIVITY_FEATURE_WIDTH,
    LABOR_RESOURCE_FEATURE_WIDTH,
    LABOR_SPECIALTY_FEATURE_WIDTH,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    LaborActor,
    LaborCostCritics,
    LaborCriticEncoder,
    LaborEncoderLayout,
    LaborObservationEncoder,
    LaborTaskCritic,
    bind_privileged_normalizer,
    critic_privileged_batch,
    labor_checkpoint_payload,
    labor_policy_tensors,
    load_labor_checkpoint_models,
)
from anthropocene_rl.autonomous_v7.labor_privileged import CRITIC_PREFIX
from anthropocene_rl.autonomous_v7.labor_value_calibration import labor_critic_digest
from anthropocene_rl.autonomous_v7.model import Normalizer


def _layout(privileged: bool = True, core: int = 10) -> LaborEncoderLayout:
    return LaborEncoderLayout(
        core,
        LABOR_SPECIALTY_FEATURE_WIDTH,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
        privileged=LAYOUT if privileged else None,
    )


def _critics(layout: LaborEncoderLayout, normalizer=None, seed: int = 3):
    torch.manual_seed(seed)
    task = LaborTaskCritic(layout, width=16, token_width=16)
    costs = LaborCostCritics(layout, width=16, token_width=16)
    if layout.privileged is not None:
        bind_privileged_normalizer(normalizer or _normalizer(), task, costs)
    return task, costs


def _village(privileged_seed: int, scale: float = 1.0) -> dict:
    raw = legacy_observation(seed=92)
    raw["core"][40:56] = [0.0] * 16
    raw["core"][144] = 0.0
    raw.update(laborCatalog=_catalog_payload(), laborPrevious=None)
    return {
        "observation": raw,
        "privilegedSchemaVersion": 1,
        "privilegedView": _view(privileged_seed, scale=scale),
    }


def _payload(privileged: bool):
    village = _village(1)
    normalizer = Normalizer.fit_bounded(observation_batch([village]))
    _, tensors = labor_policy_tensors([village], normalizer)
    layout = _layout(privileged, core=int(tensors["core"].shape[1]))
    torch.manual_seed(5)
    actor = LaborActor(layout, width=16, token_width=16)
    task, costs = _critics(layout)
    return labor_checkpoint_payload(actor, task, costs, normalizer, layout), layout, tensors


def _restore(state, layout):
    actor = LaborActor(layout, width=16, token_width=16)
    task = LaborTaskCritic(layout, width=16, token_width=16)
    costs = LaborCostCritics(layout, width=16, token_width=16)
    load_labor_checkpoint_models(state, actor, task, costs, expected_layout=layout)
    return actor, task, costs


def test_actor_output_is_bitwise_invariant_under_privilege_only_perturbation() -> None:
    base, perturbed = _village(1), _village(2, scale=7.0)
    normalizer = Normalizer.fit_bounded(observation_batch([base]))
    _, tensors_a = labor_policy_tensors([base], normalizer)
    _, tensors_b = labor_policy_tensors([perturbed], normalizer)
    assert set(tensors_a) == set(tensors_b)
    assert not any(key.startswith("privileged") for key in tensors_a)
    for key in tensors_a:
        assert torch.equal(tensors_a[key], tensors_b[key]), key
    layout = _layout(core=int(tensors_a["core"].shape[1]))
    torch.manual_seed(0)
    actor = LaborActor(layout, width=16, token_width=16)
    out_a, out_b = actor(tensors_a), actor(tensors_b)
    for key in out_a:
        assert torch.equal(out_a[key], out_b[key]), key
    # The actor graph has no privileged parameters and refuses privileged inputs.
    assert not any("privileged" in name for name, _ in actor.named_parameters())
    leaked = dict(tensors_a, **{CRITIC_PREFIX + "core": torch.zeros(1, 1, 10)})
    with pytest.raises(AssertionError, match="privileged"):
        actor(leaked)
    with pytest.raises(AssertionError, match="privileged"):
        actor(dict(tensors_a, privileged=torch.zeros(1, 10)))


def test_critic_output_changes_under_the_same_perturbation() -> None:
    base, perturbed = _village(1), _village(2, scale=7.0)
    normalizer = Normalizer.fit_bounded(observation_batch([base]))
    _, tensors = labor_policy_tensors([base], normalizer)
    layout = _layout(core=int(tensors["core"].shape[1]))
    task, costs = _critics(layout)
    raw_a = {k: torch.from_numpy(v) for k, v in privileged_batch([base]).items()}
    raw_b = {k: torch.from_numpy(v) for k, v in privileged_batch([perturbed]).items()}
    pa, pb = critic_privileged_batch(task, raw_a), critic_privileged_batch(task, raw_b)
    assert set(pa).isdisjoint(tensors)  # a separate critic-only dictionary
    with torch.no_grad():
        ta, tb = task(tensors, pa), task(tensors, pb)
        ca, cb = costs(tensors, pa), costs(tensors, pb)
    assert any(not torch.equal(ta[k], tb[k]) for k in ta)
    assert all(not torch.equal(ca[k], cb[k]) for k in ca)
    with pytest.raises(ValueError, match="critic-only privileged batch"):
        task(tensors)
    with pytest.raises(ValueError, match="keys differ"):
        task(tensors, {k: v for k, v in pa.items() if "herds" not in k})


def test_actor_encoder_class_rejects_a_privileged_width() -> None:
    with pytest.raises(ValueError, match="rejects a nonzero privileged width"):
        LaborObservationEncoder(_layout(), width=16, token_width=16)
    encoder = LaborCriticEncoder(_layout(), width=16, token_width=16)
    assert set(encoder.privileged_project) == {f.name for f in LAYOUT.families}
    assert set(encoder.privileged_overflow) == {"fields", "herds", "unlocks", "economy"}
    actor = LaborActor(_layout(), width=16, token_width=16)
    assert actor.encoder.layout.privileged is None


def test_privileged_tokens_join_the_global_set_under_film() -> None:
    tensors = _labor_tensors(batch=2)
    task, _ = _critics(_layout())
    privileged = critic_privileged_batch(task, _raw(2))
    film = task.encoder.film_global[0]
    with torch.no_grad():
        base = task(tensors, privileged)
        film.weight.normal_()
        conditioned = dict(tensors, labor_personality=torch.tensor([[0.9, 0.1], [0.2, 0.8]]))
        changed = task(conditioned, privileged)
        # With a nonzero condition the global FiLM also acts on the privileged tokens:
        # perturbing only privileged inputs still changes the conditioned output.
        other = critic_privileged_batch(task, _raw(2, seed=40))
        assert any(not torch.equal(changed[k], task(conditioned, other)[k]) for k in changed)
    assert any(not torch.equal(base[k], changed[k]) for k in base)


def test_old_checkpoints_keep_their_bytes_and_load_bit_identically() -> None:
    state, layout, tensors = _payload(False)
    assert "privileged" not in state["labor_encoder_layout"]
    assert "privilegedCritic" not in state["labor_model_contract"]
    assert "privileged_normalizer" not in state
    assert not any("privileged" in key for key in state["task_critic"])
    _, task, costs = _restore(state, layout)
    assert task.privileged_layout is None
    with torch.no_grad():
        again = _restore(copy.deepcopy(state), layout)[1]
        for name, value in task(tensors).items():
            assert torch.equal(value, again(tensors)[name])
    # A non-privileged critic's digest has no privileged field (historical semantics).
    body = labor_critic_digest(task, costs)
    assert isinstance(body, str)


def test_privileged_parameters_do_not_shift_the_historical_initialization() -> None:
    torch.manual_seed(11)
    plain = LaborTaskCritic(_layout(False), width=16, token_width=16)
    torch.manual_seed(11)
    privileged = LaborTaskCritic(_layout(True), width=16, token_width=16)
    reference = plain.state_dict()
    for name, value in privileged.state_dict().items():
        if "privileged_" in name:
            continue
        assert torch.equal(value, reference[name]), name


def test_privileged_checkpoint_round_trip_and_digest_binds_schema() -> None:
    state, layout, _tensors = _payload(True)
    assert state["labor_encoder_layout"]["privileged"]["schemaVersion"] == 1
    record = state["labor_model_contract"]["privilegedCritic"]
    assert record["schemaVersion"] == 1 and record["actorPrivilegedWidth"] == 0
    assert record["widths"] == {"taskEncoder": 16, "costEncoder": 16, "token": 16}
    _actor, task, costs = _restore(state, layout)
    assert task.privileged_normalizer is costs.privileged_normalizer
    plain_state, plain_layout, _ = _payload(False)
    _, plain_task, plain_costs = _restore(plain_state, plain_layout)
    assert labor_critic_digest(task, costs) != labor_critic_digest(plain_task, plain_costs)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda s: s["labor_encoder_layout"]["privileged"].update(schemaVersion=2),
            "schema version",
        ),
        (
            lambda s: s["labor_model_contract"]["privilegedCritic"].update(schemaVersion=2),
            "schema version",
        ),
        (lambda s: s["privileged_normalizer"].update(schemaVersion=2), "schema version"),
        (lambda s: s["privileged_normalizer"]["layout"].update(schemaVersion=2), "schema version"),
        (lambda s: s.pop("privileged_normalizer"), "lacks its privileged"),
        (lambda s: s["labor_model_contract"].pop("privilegedCritic"), "lacks its privileged"),
        (
            lambda s: s["privileged_normalizer"]["families"]["core"].update(mean=[0.5] * 10),
            "normalizer differs",
        ),
    ],
)
def test_schema_version_mismatch_is_rejected_at_load(mutation, message) -> None:
    state, _layout, _ = _payload(True)
    broken = copy.deepcopy(state)
    mutation(broken)
    with pytest.raises(ValueError, match=message):
        layout_state = broken["labor_encoder_layout"]
        loaded = LaborEncoderLayout.from_state(layout_state)
        _restore(broken, loaded)


def test_non_privileged_checkpoint_cannot_smuggle_privileged_state() -> None:
    state, layout, _ = _payload(False)
    privileged_state, _, _ = _payload(True)
    broken = dict(state, privileged_normalizer=privileged_state["privileged_normalizer"])
    with pytest.raises(ValueError, match="without a privileged block"):
        _restore(broken, layout)


def test_strict_load_migration_hook_is_explicit() -> None:
    plain, _ = _critics(_layout(False), seed=8)
    migrated, _ = _critics(_layout(True), seed=8)
    with pytest.raises(RuntimeError, match="privileged_"):
        migrated.load_state_dict(plain.state_dict(), strict=True)
    fresh = {k: v.clone() for k, v in migrated.state_dict().items() if "privileged_" in k}
    migrated.encoder.privileged_migration = True
    migrated.load_state_dict(plain.state_dict(), strict=True)
    migrated.encoder.privileged_migration = False
    for name, value in migrated.state_dict().items():
        expected = fresh[name] if "privileged_" in name else plain.state_dict()[name]
        assert torch.equal(value, expected), name
