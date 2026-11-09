"""WP-A0: per-head log-prob plumbing for the stage-2 A(s, a) head (default path unchanged).

Small CPU fixtures built on the privileged-PPO collection fixture (fake bridge execution,
real ``LaborActor``). Covers: retained per-head behaviour log-probs, persisted GAE
combination weights, old-rollout compatibility, exposed current per-head log-probs and
the opt-in per-head initial-edge surrogate (gradient-equal to the joint path at eta = 0).
"""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_labor_bridge import _attempt_payload
from test_labor_privileged_model import _critics, _layout, _village

from anthropocene_rl.autonomous_v7 import labor_a_stage2 as s2
from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7.bridge import observation_batch
from anthropocene_rl.autonomous_v7.labor_bridge import LaborAttempt
from anthropocene_rl.autonomous_v7.labor_model import (
    LABOR_COSTS,
    LaborActor,
    LaborPersonality,
    Normalizer,
    labor_log_prob,
    labor_policy_tensors,
    nonlabor_log_prob,
    sample_labor_policy,
    sample_nonlabor_policy,
)
from anthropocene_rl.autonomous_v7.labor_ppo import (
    INITIAL_POLICY_HEADS,
    LABOR_POLICY_HEADS,
    NONLABOR_POLICY_HEADS,
    LaborActorGate,
    check_head_logp_consistency,
    combine_labor_advantages,
    labor_policy_minibatch_terms,
    update_labor_policy,
)

MULTIPLIERS = {"feasibility": 0.25, "settlement_guard": 0.5}
WORLD = 50000  # fixture label only; no sealed (3000-3059) or reserved (90154-90159) world


def _village_at(camp: int) -> dict:
    return dict(
        _village(1),
        camp=camp,
        terminated=False,
        society=70,
        parentSociety=0,
        priorPopulation=100,
        population=100,
        qol=0.5,
        years=20.0,
        welfare=0.5,
    )


def _models():
    village = _village_at(7)
    normalizer = Normalizer.fit_bounded(observation_batch([village]))
    _, tensors = labor_policy_tensors([village], normalizer)
    layout = _layout(core=tensors["core"].shape[1])
    torch.manual_seed(17)
    actor = LaborActor(layout, width=16, token_width=16)
    task, costs = _critics(layout)
    return actor, task, costs


def _collect(monkeypatch, retry, models, *, seed=71, world=WORLD, generators=None):
    """One decision through ``_execute_single_step_detailed`` with a fake execution."""
    actor, task, costs = models
    village = _village_at(7)
    normalizer = Normalizer.fit_bounded(observation_batch([village]))
    successor = copy.deepcopy(village)
    successor["privilegedView"] = _village(9)["privilegedView"]
    next_step = {"turn": 4, "villages": [successor]}

    def execute(_bridge, catalogs, retained, retry_factory):
        catalog = village["observation"]["laborCatalog"]
        initial = LaborAttempt.from_payload(_attempt_payload(catalog, rejected=retry), catalogs[7])
        retried = {7: retry_factory(7, catalogs[7], initial)} if retry else {}
        final = dataclasses.replace(initial, attempt=1, remaining_proposals=0) if retry else initial
        return SimpleNamespace(
            step=next_step, initial_attempts={7: initial}, final_attempts={7: final}, retry=retried
        )

    monkeypatch.setattr(training, "execute_labor_decision", execute)
    generator = torch.Generator().manual_seed(seed)
    _, rows, _ = training._execute_single_step_detailed(
        bridge=object(),
        step={"turn": 3, "villages": [village]},
        actor=actor,
        task_critic=task,
        cost_critics=costs,
        normalizer=normalizer,
        personality=LaborPersonality(),
        generator=generator,
        world_seed=world,
        world_index=0,
        founded_turns={7: 0},
        action_binding=None,
    )
    if generators is not None:
        generators.append(generator)
    tail = training._bootstrap_values(next_step, normalizer, task, costs, LaborPersonality())
    return rows[0], tail[7], normalizer, village


def _batch(rows, bootstrap, models, multipliers=MULTIPLIERS):
    actor, task, costs = models
    from anthropocene_rl.autonomous_v7.labor_value_calibration import labor_critic_digest
    from anthropocene_rl.autonomous_v7.value_calibration import behavior_policy_digest

    return training.build_labor_training_batch(
        rows,
        bootstrap,
        personality=LaborPersonality(),
        cost_multipliers=multipliers,
        behavior_policy_sha256=behavior_policy_digest(
            actor, training.labor_behavior_configuration(LaborPersonality())
        ),
        critic_sha256=labor_critic_digest(task, costs),
        normalization_sha256="c" * 64,
    )


# Rows: (retry, generator seed). Distinct worlds keep each row its own sequence.
ROW_PLAN = ((False, 71), (True, 72), (False, 73), (False, 74), (True, 75), (False, 76),
            (False, 77))


def _rows(monkeypatch, models, plan=ROW_PLAN):
    rows, bootstrap = [], {}
    for index, (retry, seed) in enumerate(plan):
        row, tail, *_ = _collect(monkeypatch, retry, models, seed=seed, world=WORLD + index)
        rows.append(row)
        bootstrap[(WORLD + index, 7)] = tail
    return rows, bootstrap


def _gate(batch):
    return LaborActorGate(
        batch.behavior_policy_digest,
        batch.advantage_critic_digest,
        batch.normalization_binding_digest,
        "d" * 64,
    )


def _heads_scorer(actor):
    return lambda tensors, sample: training._score_nonlabor_heads(actor, tensors, sample)


def _joint_scorer(actor):
    return lambda tensors, sample: training._score_nonlabor(actor, tensors, sample)


# ---------------------------------------------------------------------------
# Deliverable 1: behaviour per-head log-probs retained at rollout
# ---------------------------------------------------------------------------


def test_head_names_are_the_log_prob_functions_keys(monkeypatch):
    models = _models()
    row, *_ = _collect(monkeypatch, False, models)
    actor = models[0]
    output = actor(row.initial_tensors)
    labor = labor_log_prob(output, row.initial_labor_sample, row.initial_tensors)
    nonlabor = nonlabor_log_prob(output, row.initial_nonlabor_sample)
    assert tuple(labor) == (*LABOR_POLICY_HEADS, "joint")
    assert tuple(nonlabor) == (*NONLABOR_POLICY_HEADS, "joint")
    assert (*LABOR_POLICY_HEADS, *NONLABOR_POLICY_HEADS) == INITIAL_POLICY_HEADS


@pytest.mark.parametrize("retry", [False, True])
def test_rollout_retains_behaviour_heads_that_sum_to_the_stored_joints(
    monkeypatch, tmp_path, retry
):
    models = _models()
    row, tail, *_ = _collect(monkeypatch, retry, models)
    assert set(row.behavior_initial_labor_head_logp) == set(LABOR_POLICY_HEADS)
    assert set(row.behavior_initial_nonlabor_head_logp) == set(NONLABOR_POLICY_HEADS)
    # float32 sum in the log_prob functions' own order reproduces the stored joint bitwise.
    labor = torch.tensor([row.behavior_initial_labor_head_logp[h] for h in LABOR_POLICY_HEADS])
    nonlabor = torch.tensor(
        [row.behavior_initial_nonlabor_head_logp[h] for h in NONLABOR_POLICY_HEADS]
    )
    assert float(labor[0] + labor[1]) == row.behavior_initial_labor_logp
    total = torch.zeros((), dtype=torch.float32)
    for value in nonlabor:
        total = total + value
    assert float(total) == row.behavior_initial_nonlabor_logp
    # ... and matches a rescoring of the retained sample with the unchanged actor.
    with torch.no_grad():
        output = models[0](row.initial_tensors)
        rescored = {
            **labor_log_prob(output, row.initial_labor_sample, row.initial_tensors),
            **{
                k: v
                for k, v in nonlabor_log_prob(output, row.initial_nonlabor_sample).items()
                if k != "joint"
            },
        }
    for head in LABOR_POLICY_HEADS:
        assert row.behavior_initial_labor_head_logp[head] == pytest.approx(
            float(rescored[head][0]), rel=1e-5, abs=1e-5
        )
    for head in NONLABOR_POLICY_HEADS:
        assert row.behavior_initial_nonlabor_head_logp[head] == pytest.approx(
            float(rescored[head][0]), rel=1e-5, abs=1e-5
        )

    batch, tt, ct = _batch([row], {(WORLD, 7): tail}, models)
    labor_heads, nonlabor_heads = batch.require_behaviour_head_logp()
    # (float64 check of float32 values: the error is the float32 sum's own rounding)
    assert check_head_logp_consistency(
        labor_heads, batch.behaviour_initial_labor_logp, LABOR_POLICY_HEADS, "t"
    ) < 1e-5
    assert check_head_logp_consistency(
        nonlabor_heads, batch.behaviour_initial_nonlabor_logp, NONLABOR_POLICY_HEADS, "t"
    ) < 1e-5
    path = tmp_path / "rollout.pt"
    torch.save(training._state_dict_rollout(batch, tt, ct, {}), path)
    restored, *_ = training._load_rollout(path)  # weights_only=True
    for name in LABOR_POLICY_HEADS:
        assert torch.equal(restored.behaviour_initial_labor_head_logp[name], labor_heads[name])
    for name in NONLABOR_POLICY_HEADS:
        assert torch.equal(
            restored.behaviour_initial_nonlabor_head_logp[name], nonlabor_heads[name]
        )
    assert restored.cost_multipliers == MULTIPLIERS


@pytest.mark.parametrize("retry", [False, True])
def test_rollout_sampling_stream_is_only_the_samplers(monkeypatch, retry):
    """Scoring consumes no generator draws: the stream after a decision equals a
    reference that only samples (initial labor, non-labor, then the retry labor)."""
    models = _models()
    generators = []
    row, _, normalizer, village = _collect(monkeypatch, retry, models, generators=generators)
    actor = models[0]
    reference = torch.Generator().manual_seed(71)
    _, tensors = labor_policy_tensors([village], normalizer, personality=LaborPersonality())
    with torch.no_grad():
        output = actor(tensors)
        labor = sample_labor_policy(output, tensors, generator=reference)
        nonlabor = sample_nonlabor_policy(output, generator=reference)
        if retry:
            retry_output = actor(row.retry_tensors)
            retry_sample = sample_labor_policy(retry_output, row.retry_tensors, generator=reference)
            for name, value in retry_sample.items():
                assert torch.equal(value, row.retry_labor_sample[name])
    for name, value in labor.items():
        assert torch.equal(value, row.initial_labor_sample[name])
    for name, value in nonlabor.items():
        assert torch.equal(value, row.initial_nonlabor_sample[name])
    assert torch.equal(generators[0].get_state(), reference.get_state())


def test_batch_rejects_per_head_logp_that_do_not_sum_to_the_joint(monkeypatch):
    models = _models()
    row, tail, *_ = _collect(monkeypatch, False, models)
    batch, *_ = _batch([row], {(WORLD, 7): tail}, models)
    broken = dict(batch.behaviour_initial_nonlabor_head_logp)
    broken["fission"] = broken["fission"] + 0.01
    with pytest.raises(ValueError, match="do not sum to the joint"):
        dataclasses.replace(batch, behaviour_initial_nonlabor_head_logp=broken)
    with pytest.raises(ValueError, match="both labor and non-labor"):
        dataclasses.replace(batch, behaviour_initial_labor_head_logp=None)


# ---------------------------------------------------------------------------
# Deliverable 2: GAE combination weights persisted
# ---------------------------------------------------------------------------


def test_cost_multipliers_and_row_weights_recombine_the_gae_bitwise(monkeypatch):
    models = _models()
    rows, bootstrap = _rows(monkeypatch, models, plan=ROW_PLAN[:3])
    captured = []
    original = training.combine_labor_advantages

    def spy(task, costs, **kwargs):
        captured.append((dict(task), dict(costs), dict(kwargs)))
        return original(task, costs, **kwargs)

    monkeypatch.setattr(training, "combine_labor_advantages", spy)
    batch, *_ = _batch(rows, bootstrap, models)
    assert len(captured) == 2  # initial edge, retry edge
    multipliers = batch.require_cost_multipliers()
    assert multipliers == MULTIPLIERS
    assert set(multipliers) == set(LABOR_COSTS)
    initial_weights, retry_weights = batch.row_personality_weights()
    assert len(retry_weights) == len(batch.retry_parent) == 1
    for (task, costs, kwargs), weights, stored in (
        (captured[0], initial_weights, batch.initial_advantage),
        (captured[1], retry_weights, batch.retry_advantage),
    ):
        assert kwargs["cost_multipliers"] == multipliers
        assert torch.equal(kwargs["row_weights"], weights)
        recombined = combine_labor_advantages(
            task,
            costs,
            personality=LaborPersonality(),
            cost_multipliers=multipliers,
            row_weights=weights,
        )
        assert torch.equal(recombined, stored)


def test_old_rollout_without_new_fields_loads_and_per_head_users_refuse(monkeypatch, tmp_path):
    models = _models()
    rows, bootstrap = _rows(monkeypatch, models, plan=ROW_PLAN[:2])
    batch, tt, ct = _batch(rows, bootstrap, models)
    state = training._state_dict_rollout(batch, tt, ct, {})
    for name in (
        "behaviour_initial_labor_head_logp",
        "behaviour_initial_nonlabor_head_logp",
        "cost_multipliers",
    ):
        del state["batch"][name]
    path = tmp_path / "old.pt"
    torch.save(state, path)
    old, *_ = training._load_rollout(path)
    assert old.behaviour_initial_labor_head_logp is None
    assert old.behaviour_initial_nonlabor_head_logp is None
    assert old.cost_multipliers is None
    with pytest.raises(ValueError, match="predates per-head behaviour"):
        old.require_behaviour_head_logp()
    with pytest.raises(ValueError, match="predates persisted cost multipliers"):
        old.require_cost_multipliers()
    actor = models[0]
    index = torch.arange(old.parent_count)
    with pytest.raises(ValueError, match="predates per-head behaviour"):
        labor_policy_minibatch_terms(
            actor, old, index, _heads_scorer(actor), clip=0.2, entropy_weight=0.0,
            initial_head_advantage={h: old.initial_advantage for h in INITIAL_POLICY_HEADS},
        )
    with pytest.raises(ValueError, match="predates per-head behaviour"):
        _update(actor, old, per_head=True)
    # The default joint path still trains on the old file.
    _update(copy.deepcopy(actor), old, per_head=False)


# ---------------------------------------------------------------------------
# Deliverable 3: current per-head log-probs exposed
# ---------------------------------------------------------------------------


def test_minibatch_terms_expose_current_heads_without_changing_the_joint(monkeypatch):
    models = _models()
    rows, bootstrap = _rows(monkeypatch, models, plan=ROW_PLAN[:3])
    batch, *_ = _batch(rows, bootstrap, models)
    actor = models[0]
    index = torch.arange(batch.parent_count)
    joint = labor_policy_minibatch_terms(
        actor, batch, index, _joint_scorer(actor), clip=0.2, entropy_weight=0.01
    )
    mapped = labor_policy_minibatch_terms(
        actor, batch, index, _heads_scorer(actor), clip=0.2, entropy_weight=0.01
    )
    assert joint.initial_surrogate == mapped.initial_surrogate == "joint"
    assert set(joint.current_initial_labor_head_logp) == set(LABOR_POLICY_HEADS)
    assert joint.current_initial_nonlabor_head_logp is None  # tensor-form scorer
    assert set(mapped.current_initial_nonlabor_head_logp) == set(NONLABOR_POLICY_HEADS)
    for name in ("loss", "policy_loss", "entropy", "initial_ratio", "retry_ratio",
                 "parent_surrogate", "initial_joint_logp", "retry_labor_logp"):
        assert torch.equal(getattr(joint, name), getattr(mapped, name)), name
    labor = mapped.current_initial_labor_head_logp
    nonlabor = mapped.current_initial_nonlabor_head_logp
    assert all(value.requires_grad for value in (*labor.values(), *nonlabor.values()))
    with torch.no_grad():
        check_head_logp_consistency(
            {**labor, **nonlabor},
            mapped.initial_joint_logp,
            INITIAL_POLICY_HEADS,
            "current",
        )


# ---------------------------------------------------------------------------
# Deliverable 4: per-head surrogate (opt-in)
# ---------------------------------------------------------------------------

# ratio = exp(current - behaviour) ~ exp(-delta): rows 0/4 clipped (zero gradient),
# rows 3/5/6 outside the band but unclipped (pessimistic side), rows 1/2 inside.
DELTAS = (-0.6, -0.05, 0.0, 0.5, 0.6, -0.4, 0.45)
ADVANTAGES = (1.0, -0.7, 0.5, 0.8, -0.9, -1.1, 1.2)


def _shifted(batch):
    """Move behaviour log-probs (joint and the movement head together) to spread ratios."""
    delta = torch.tensor(DELTAS)
    nonlabor = dict(batch.behaviour_initial_nonlabor_head_logp)
    nonlabor["movement"] = nonlabor["movement"] + delta
    return dataclasses.replace(
        batch,
        behaviour_initial_nonlabor_logp=batch.behaviour_initial_nonlabor_logp + delta,
        behaviour_initial_nonlabor_head_logp=nonlabor,
        initial_advantage=torch.tensor(ADVANTAGES),
    )


def _grads(actor, batch, *, head_advantage, entropy_weight=0.01):
    actor.zero_grad()
    terms = labor_policy_minibatch_terms(
        actor,
        batch,
        torch.arange(batch.parent_count),
        _heads_scorer(actor),
        clip=0.2,
        entropy_weight=entropy_weight,
        initial_head_advantage=head_advantage,
    )
    terms.loss.backward()
    grads = {
        name: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for name, p in actor.named_parameters()
    }
    return terms, grads


def test_per_head_surrogate_at_eta_zero_has_the_joint_gradient(monkeypatch):
    models = _models()
    rows, bootstrap = _rows(monkeypatch, models)
    batch = _shifted(_batch(rows, bootstrap, models)[0])
    actor = models[0]
    joint, joint_grads = _grads(actor, batch, head_advantage=None)
    shared = {h: batch.initial_advantage for h in INITIAL_POLICY_HEADS}
    per_head, head_grads = _grads(actor, batch, head_advantage=shared)
    assert joint.initial_surrogate == "joint" and per_head.initial_surrogate == "per-head"

    # The fixture exercises both clip regimes and the retry edge.
    ratio = joint.initial_ratio.detach()
    advantage = batch.initial_advantage
    clipped = ((ratio > 1.2) & (advantage > 0)) | ((ratio < 0.8) & (advantage < 0))
    unclipped_outside = ((ratio > 1.2) & (advantage < 0)) | ((ratio < 0.8) & (advantage > 0))
    assert int(clipped.sum()) >= 2 and int(unclipped_outside.sum()) >= 2
    assert int(((ratio > 0.8) & (ratio < 1.2)).sum()) >= 2
    assert len(batch.retry_parent) == 2

    # Identical ratios (bitwise: r_h's forward value is the joint ratio) and retry edge.
    assert torch.equal(joint.initial_ratio, per_head.initial_ratio)
    assert torch.equal(joint.retry_ratio, per_head.retry_ratio)
    assert torch.equal(joint.entropy, per_head.entropy)
    # Loss VALUE: head mean of seven identical per-head terms -> equals the joint value
    # up to the rounding of (7 x)/7; it is not 7x the joint value.
    assert torch.allclose(per_head.parent_surrogate, joint.parent_surrogate, rtol=1e-6, atol=1e-7)
    assert torch.allclose(per_head.loss, joint.loss, rtol=1e-6, atol=1e-7)
    # GRADIENT: equal up to autograd accumulation order.
    nonzero = 0
    for name, grad in joint_grads.items():
        assert torch.allclose(head_grads[name], grad, rtol=1e-5, atol=1e-8), name
        nonzero += int(grad.abs().sum() > 0)
    assert nonzero > 0


def test_per_head_surrogate_matches_stage2_contract_with_distinct_coefficients(monkeypatch):
    """With different coefficients per head the gradient is the stage-2 library's
    ``per_head_surrogate`` gradient; the value is its head mean (s2 returns the sum)."""
    models = _models()
    rows, bootstrap = _rows(monkeypatch, models)
    batch = _shifted(_batch(rows, bootstrap, models)[0])
    actor = models[0]
    generator = torch.Generator().manual_seed(5)
    coefficients = {
        h: torch.randn(batch.parent_count, generator=generator) for h in INITIAL_POLICY_HEADS
    }
    # No retry rows and no entropy: compare the initial-edge surrogate alone.
    plain = dataclasses.replace(
        batch,
        retry_parent=batch.retry_parent[:0],
        retry_tensors={k: v[:0] for k, v in batch.retry_tensors.items()},
        retry_labor_sample={k: v[:0] for k, v in batch.retry_labor_sample.items()},
        behaviour_retry_labor_logp=batch.behaviour_retry_labor_logp[:0],
        retry_advantage=batch.retry_advantage[:0],
        retry_privileged_raw={k: v[:0] for k, v in batch.retry_privileged_raw.items()},
        retry_actor_outputs={k: v[:0] for k, v in batch.retry_actor_outputs.items()},
    )
    ours, our_grads = _grads(actor, plain, head_advantage=coefficients, entropy_weight=0.0)

    actor.zero_grad()
    index = torch.arange(plain.parent_count)
    tensors = {k: v.index_select(0, index) for k, v in plain.initial_tensors.items()}
    output = actor(tensors)
    current = {
        **{k: v for k, v in labor_log_prob(output, plain.initial_labor_sample, tensors).items()
           if k != "joint"},
        **{k: v for k, v in nonlabor_log_prob(output, plain.initial_nonlabor_sample).items()
           if k != "joint"},
    }
    labor_b, nonlabor_b = plain.require_behaviour_head_logp()
    reference = s2.per_head_surrogate(current, {**labor_b, **nonlabor_b}, coefficients, 0.2)
    (-reference.mean()).backward()
    for name, p in actor.named_parameters():
        expected = p.grad if p.grad is not None else torch.zeros_like(p)
        # float32 cancellation in shared parameters: scale-aware absolute tolerance
        scale = float(expected.abs().max()) if expected.numel() else 0.0
        assert torch.allclose(our_grads[name], expected, rtol=1e-4, atol=1e-5 * scale + 1e-8), name
    heads = len(INITIAL_POLICY_HEADS)
    assert torch.allclose(
        ours.parent_surrogate * heads, reference.detach(), rtol=1e-5, atol=1e-6
    )


def test_per_head_option_refuses_ambiguous_or_incomplete_inputs(monkeypatch):
    models = _models()
    rows, bootstrap = _rows(monkeypatch, models, plan=ROW_PLAN[:2])
    batch, *_ = _batch(rows, bootstrap, models)
    actor = models[0]
    index = torch.arange(batch.parent_count)
    shared = {h: batch.initial_advantage for h in INITIAL_POLICY_HEADS}
    with pytest.raises(ValueError, match="returns per-head log-probs"):
        labor_policy_minibatch_terms(
            actor, batch, index, _joint_scorer(actor), clip=0.2, entropy_weight=0.0,
            initial_head_advantage=shared,
        )
    with pytest.raises(ValueError, match="must name exactly"):
        labor_policy_minibatch_terms(
            actor, batch, index, _heads_scorer(actor), clip=0.2, entropy_weight=0.0,
            initial_head_advantage={h: batch.initial_advantage for h in LABOR_POLICY_HEADS},
        )
    with pytest.raises(ValueError, match="per_head_surrogate is off"):
        _update(actor, batch, per_head=False, initial_head_advantage=shared)


def _update(actor, batch, *, per_head, lr=1e-5, **kwargs):
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    extra = {"per_head_surrogate": True} if per_head else {}
    stats = update_labor_policy(
        policy_module=actor,
        labor_actor=actor,
        optimizer=optimizer,
        batch=batch,
        nonlabor_scorer=_heads_scorer(actor) if per_head else _joint_scorer(actor),
        rng=np.random.default_rng(3),
        epochs=2,
        batch_size=3,
        clip=0.2,
        entropy_weight=0.01,
        max_grad_norm=1.0,
        target_kl=10.0,
        audit_rows=4,
        actor_gate=_gate(batch),
        behavior_configuration=training.labor_behavior_configuration(LaborPersonality()),
        **extra,
        **kwargs,
    )
    return stats, optimizer


def test_update_with_per_head_option_at_eta_zero_tracks_the_joint_update(monkeypatch):
    models = _models()
    rows, bootstrap = _rows(monkeypatch, models)
    batch, *_ = _batch(rows, bootstrap, models)
    joint_actor = copy.deepcopy(models[0])
    head_actor = copy.deepcopy(models[0])
    joint_stats, _ = _update(joint_actor, batch, per_head=False)
    head_stats, _ = _update(head_actor, batch, per_head=True)
    assert "initialSurrogate" not in joint_stats  # default report unchanged
    assert head_stats.pop("initialSurrogate")["coefficients"] == "gae-all-heads"
    assert head_stats["attemptedSteps"] == joint_stats["attemptedSteps"] == 6
    assert head_stats["acceptedSteps"] == joint_stats["acceptedSteps"]
    assert head_stats["epochCoverage"] == joint_stats["epochCoverage"]
    np.testing.assert_allclose(head_stats["losses"], joint_stats["losses"], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(
        head_stats["sampledNonnegativeKL"], joint_stats["sampledNonnegativeKL"],
        rtol=1e-3, atol=1e-9,
    )
    moved = 0
    for (name, before), after_joint, after_head in zip(
        models[0].state_dict().items(),
        joint_actor.state_dict().values(),
        head_actor.state_dict().values(),
        strict=True,
    ):
        assert torch.allclose(after_head, after_joint, rtol=1e-4, atol=1e-6), name
        moved += int(not torch.equal(before, after_joint))
    assert moved > 0


def test_train_cli_per_head_option_defaults_off():
    parser = training.argument_parser()
    base = ["train", "--checkpoint", "c.pt", "--rollout", "r.pt", "--anchor", "a.pt",
            "--out", "o.pt"]
    assert parser.parse_args(base).per_head_surrogate is False
    assert parser.parse_args([*base, "--per-head-surrogate"]).per_head_surrogate is True


def test_default_update_is_bitwise_independent_of_the_new_fields(monkeypatch):
    """End-to-end default path: the joint update with the WP-A0 fields present equals the
    update of the same batch with them stripped (as an older rollout), bitwise."""
    models = _models()
    rows, bootstrap = _rows(monkeypatch, models)
    batch, *_ = _batch(rows, bootstrap, models)
    stripped = dataclasses.replace(
        batch,
        behaviour_initial_labor_head_logp=None,
        behaviour_initial_nonlabor_head_logp=None,
        cost_multipliers=None,
    )
    results = []
    for candidate in (batch, stripped):
        actor = copy.deepcopy(models[0])
        stats, optimizer = _update(actor, candidate, per_head=False)
        results.append((stats, actor.state_dict(), optimizer.state_dict()["state"]))
    (stats_a, actor_a, opt_a), (stats_b, actor_b, opt_b) = results
    assert stats_a == stats_b
    assert stats_a["acceptedSteps"] == 6
    for name, value in actor_a.items():
        assert torch.equal(value, actor_b[name]), name
    for key, state in opt_a.items():
        for name, value in state.items():
            assert torch.equal(value, opt_b[key][name])
