"""WP-A1: the stage-1 A(s,a) head wired into labor PPO as a Q-Prop-style control variate.

Synthetic CPU fixtures only (fake bridge execution, real ``LaborActor`` and critics,
random A heads); no native DLL, no sealed (3000-3059) or reserved (90154-90159) worlds.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_labor_ppo_perhead import (
    ROW_PLAN,
    WORLD,
    _batch,
    _collect,
    _gate,
    _heads_scorer,
    _joint_scorer,
    _models,
    _update,
    _village_at,
)
from test_labor_privileged_model import _layout

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7 import labor_a_ppo as ap
from anthropocene_rl.autonomous_v7 import labor_a_stage1 as s1
from anthropocene_rl.autonomous_v7 import labor_a_stage2 as s2
from anthropocene_rl.autonomous_v7 import labor_advantage as la
from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7.bridge import observation_batch
from anthropocene_rl.autonomous_v7.labor_model import (
    LABOR_COSTS,
    LABOR_OBJECTIVES,
    LaborActor,
    LaborCostCritics,
    LaborPersonality,
    LaborTaskCritic,
    Normalizer,
    bind_privileged_normalizer,
    labor_policy_tensors,
    nonlabor_log_prob,
    schema12_initial_action,
)
from anthropocene_rl.autonomous_v7.labor_ppo import (
    INITIAL_POLICY_HEADS,
    LABOR_GATE_HEADS,
    combine_labor_advantages,
    labor_policy_minibatch_terms,
    update_labor_policy,
)
from anthropocene_rl.autonomous_v7.labor_value_calibration import labor_critic_digest
from anthropocene_rl.autonomous_v7.value_calibration import behavior_policy_digest

REPO = Path(__file__).resolve().parents[1]
PRECHANGE = "94ba951"  # WP-A0, the parent of WP-A1
ZEROED = ("feasibility", "settlement_guard")  # as in the current stage-1 warm start


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _a_payload(task, costs, *, seed=11, zero=ZEROED, masked=la.STAGE1_MASKED_BLOCKS, **widths):
    torch.manual_seed(seed)
    head = la.LaborAdvantageHead.for_critics(task, costs, **widths)
    with torch.no_grad():
        for name in la.COMPONENTS:  # the fitted output layers are zero-initialised
            head.heads[name][2].weight.normal_(0.0, 0.5)
            head.heads[name][2].bias.normal_(0.0, 0.1)
    head = s1.zero_heads(head, zero)
    head.set_masked_blocks(masked)
    payload = la.advantage_checkpoint_payload(
        head,
        critic_digest=labor_critic_digest(task, costs),
        config=la.AdvantageFitConfig(),
        fit={},
        task_context=la.task_context_record(task),
    )
    payload["warm_start"] = {
        "schema": s1.WARM_START_SCHEMA,
        "status": "partial",
        "seed": 0,
        "decisions": {n: ("zero" if n in zero else "load") for n in la.COMPONENTS},
    }
    return payload


def _frozen(task, costs, **kwargs) -> ap.FrozenA:
    """A FrozenA on a frozen copy of the critics (as load_frozen_a, without checkpoints)."""
    payload = _a_payload(task, costs, **kwargs)
    frozen_task, frozen_costs = copy.deepcopy((task, costs))
    for module in (frozen_task, frozen_costs):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
    head, _ = la.load_advantage_head(payload, frozen_task, frozen_costs)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return ap.FrozenA(
        frozen_task,
        frozen_costs,
        head,
        labor_critic_digest(frozen_task, frozen_costs),
        ap._module_digest(head),
        {"decisions": payload["warm_start"]["decisions"]},
    )


def _small(monkeypatch, plan=ROW_PLAN, multipliers=None):
    """Small-width models, a 7-parent batch (two retries) and the parent populations."""
    models = _models()
    rows, bootstrap = [], {}
    for index, (retry, seed) in enumerate(plan):
        row, tail, *_ = _collect(monkeypatch, retry, models, seed=seed, world=WORLD + index)
        rows.append(row)
        bootstrap[(WORLD + index, 7)] = tail
    kwargs = {} if multipliers is None else {"multipliers": multipliers}
    batch, *_ = _batch(rows, bootstrap, models, **kwargs)
    return models, rows, batch, [row.initial_population for row in rows]


def _prepare(models, batch, populations, *, eta=None, key=(5, 0), draws=6, **frozen_kwargs):
    actor, task, costs = models
    frozen = _frozen(task, costs, **frozen_kwargs)
    return frozen, ap.prepare_iteration(
        frozen, actor, batch, populations, eta=eta or ap.zero_eta(), key=key, draws=draws
    )


def _grads(actor, loss):
    actor.zero_grad()
    loss.backward()
    return {
        n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for n, p in actor.named_parameters()
    }


def _close(a, b, rtol=1e-5, atol=1e-7):
    for name in a:
        scale = float(b[name].abs().max()) if b[name].numel() else 0.0
        assert torch.allclose(a[name], b[name], rtol=rtol, atol=atol + 1e-5 * scale), name


# ---------------------------------------------------------------------------
# 1. --a-mode off: bitwise the pre-change code (git 94ba951), WP-A0 default unchanged
# ---------------------------------------------------------------------------


def _prechange(name: str):
    """Import the pre-change (WP-A0) module ``name`` from git under a private name."""
    modname = f"anthropocene_rl.autonomous_v7._wpa1_prechange_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    if name != "labor_ppo":
        _prechange("labor_ppo")  # the pre-change training module imports this one
    try:
        source = subprocess.run(
            ["git", "show", f"{PRECHANGE}:src/anthropocene_rl/autonomous_v7/{name}.py"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("pre-change source is not available from git")
    source = source.replace(
        "from .labor_ppo import", "from ._wpa1_prechange_labor_ppo import"
    )
    spec = importlib.util.spec_from_loader(modname, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "anthropocene_rl.autonomous_v7"
    sys.modules[modname] = module
    exec(compile(source, f"<{PRECHANGE}:{name}.py>", "exec"), module.__dict__)
    return module


def _old_batch(old_ppo, batch):
    return old_ppo.LaborAttemptPolicyBatch(
        **{
            f.name: getattr(batch, f.name)
            for f in dataclasses.fields(old_ppo.LaborAttemptPolicyBatch)
        }
    )


@pytest.mark.parametrize("per_head", [False, True])
def test_default_update_is_bitwise_the_prechange_update(monkeypatch, per_head):
    old_ppo = _prechange("labor_ppo")
    models, _, batch, _ = _small(monkeypatch)
    results = []
    for module, candidate in ((old_ppo, _old_batch(old_ppo, batch)), (None, batch)):
        actor = copy.deepcopy(models[0])
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
        update = update_labor_policy if module is None else module.update_labor_policy
        stats = update(
            policy_module=actor, labor_actor=actor, optimizer=optimizer, batch=candidate,
            nonlabor_scorer=(_heads_scorer if per_head else _joint_scorer)(actor),
            rng=np.random.default_rng(3), epochs=2, batch_size=3, clip=0.2,
            entropy_weight=0.01, max_grad_norm=1.0, target_kl=10.0, audit_rows=4,
            actor_gate=_gate(batch),
            behavior_configuration=training.labor_behavior_configuration(LaborPersonality()),
            **({"per_head_surrogate": True} if per_head else {}),
        )
        results.append((stats, actor.state_dict(), optimizer.state_dict()["state"]))
    (stats_old, actor_old, opt_old), (stats_new, actor_new, opt_new) = results
    assert stats_old == stats_new and stats_new["acceptedSteps"] == 6
    for name, value in actor_old.items():
        assert torch.equal(value, actor_new[name]), name
    for key, state in opt_old.items():
        for name, value in state.items():
            assert torch.equal(value, opt_new[key][name])


def test_stage2_per_head_surrogate_calls_through_with_the_original_formula():
    """S2's ``per_head_surrogate`` now delegates to labor_ppo's one implementation; it must
    still equal the original stage-2 formula (value: head sum; gradient) at d8f1e86."""

    def original(current, behaviour, advantages, clip):
        heads = list(advantages)
        old = sum(behaviour[h].detach() for h in heads)
        total = None
        for h in heads:
            joint = current[h] + sum(current[j].detach() for j in heads if j != h)
            ratio = torch.exp(joint - old)
            adv = advantages[h].detach()
            term = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
            total = term if total is None else total + term
        return total

    generator = torch.Generator().manual_seed(4)
    theta = torch.randn(3, 6, generator=generator, requires_grad=True)
    behaviour = {h: torch.randn(6, generator=generator) * 0.3 for h in ("a", "b", "c")}
    advantages = {h: torch.randn(6, generator=generator) for h in ("a", "b", "c")}

    def current():
        return {h: theta[i] * 0.4 for i, h in enumerate(("a", "b", "c"))}

    ours = s2.per_head_surrogate(current(), behaviour, advantages, 0.2)
    (g_ours,) = torch.autograd.grad(ours.sum(), theta)
    ref = original(current(), behaviour, advantages, 0.2)
    (g_ref,) = torch.autograd.grad(ref.sum(), theta)
    assert torch.allclose(ours, ref, atol=1e-6) and torch.allclose(g_ours, g_ref, atol=1e-6)
    ratio = torch.exp(sum(current()[h] for h in "abc") - sum(behaviour.values()))
    assert bool(((ratio > 1.2) | (ratio < 0.8)).any())  # the clip is exercised


# ---------------------------------------------------------------------------
# Units, combination, executed-action encoding
# ---------------------------------------------------------------------------


def test_combine_head_units_is_the_gae_combination(monkeypatch):
    _, _, batch, _ = _small(monkeypatch, multipliers={"feasibility": 0.25,
                                                           "settlement_guard": 0.5})
    weights, _ = batch.row_personality_weights()
    generator = torch.Generator().manual_seed(1)
    head_units = {n: torch.randn(batch.parent_count, generator=generator) for n in la.COMPONENTS}
    raw = {n: head_units[n] * la.REFERENCE_SCALES[n] for n in la.COMPONENTS}
    expected = combine_labor_advantages(
        {n: raw[n] for n in LABOR_OBJECTIVES}, {n: raw[n] for n in LABOR_COSTS},
        personality=LaborPersonality(), cost_multipliers=batch.require_cost_multipliers(),
        row_weights=weights,
    )
    got = ap.combine_head_units(head_units, weights, batch.require_cost_multipliers())
    assert torch.equal(got, expected)
    # [N, K] candidates use row i's weights for every candidate of row i.
    stacked = {n: torch.stack([head_units[n], 2 * head_units[n]], dim=1) for n in la.COMPONENTS}
    pair = ap.combine_head_units(stacked, weights, batch.require_cost_multipliers())
    assert torch.allclose(pair[:, 0], expected) and torch.allclose(pair[:, 1], 2 * expected)


def test_executed_encoding_matches_the_recorded_rollout_features(monkeypatch):
    """Population from the rollout reproduces the producer's recorded action features."""
    from test_labor_privileged_ppo import _collect_one

    actor, task, costs, row, tail, _ = _collect_one(monkeypatch, True)
    assert row.initial_action_features is not None and row.initial_population is not None
    batch, *_ = training.build_labor_training_batch(
        [row], {(50000, 7): tail[7]}, personality=LaborPersonality(),
        cost_multipliers={"feasibility": 1.0, "settlement_guard": 1.0},
        behavior_policy_sha256="a" * 64, critic_sha256="b" * 64, normalization_sha256="c" * 64,
    )
    frozen = _frozen(task, costs)
    prepared = ap.prepare_iteration(
        frozen, actor, batch, [row.initial_population], eta=ap.zero_eta(), key=(1, 0), draws=3
    )
    assert prepared.a_executed.shape == (1,)
    with pytest.raises(ValueError, match="recorded action features"):
        ap.prepare_iteration(
            frozen, actor, batch, [row.initial_population + 1], eta=ap.zero_eta(), key=(1, 0),
            draws=3,
        )


# ---------------------------------------------------------------------------
# 2. gae with eta = 0 everywhere has the off gradient; eta only moves its own head
# ---------------------------------------------------------------------------


def test_gae_with_eta_zero_has_the_off_gradient(monkeypatch):
    models, _, batch, populations = _small(monkeypatch)
    _, prepared = _prepare(models, batch, populations)
    assert prepared.correction_fn(models[0]) is None
    actor = models[0]
    index = torch.arange(batch.parent_count)
    off = labor_policy_minibatch_terms(
        actor, batch, index, _joint_scorer(actor), clip=0.2, entropy_weight=0.01
    )
    off_grads = _grads(actor, off.loss)
    gae = labor_policy_minibatch_terms(
        actor, batch, index, _heads_scorer(actor), clip=0.2, entropy_weight=0.01,
        initial_head_advantage=prepared.coefficients(),
    )
    gae_grads = _grads(actor, gae.loss)
    assert all(torch.equal(c, batch.initial_advantage) for c in prepared.coefficients().values())
    assert torch.allclose(gae.loss, off.loss, rtol=1e-6, atol=1e-7)
    _close(gae_grads, off_grads)
    # And a whole update tracks the off update.
    off_actor, gae_actor = copy.deepcopy(actor), copy.deepcopy(actor)
    off_stats, _ = _update(off_actor, batch, per_head=False)
    gae_stats, _ = _update(gae_actor, batch, per_head=True,
                           initial_head_advantage=prepared.coefficients(),
                           initial_correction=prepared.correction_fn(gae_actor))
    assert off_stats["acceptedSteps"] == gae_stats["acceptedSteps"] == 6
    for name, value in off_actor.state_dict().items():
        assert torch.allclose(gae_actor.state_dict()[name], value, rtol=1e-4, atol=1e-6), name


@pytest.mark.parametrize("head", ["movement", "fission", "trade_gate", "specialty", "activity"])
def test_eta_on_one_head_changes_only_that_heads_parameters(monkeypatch, head):
    """The control variate for head h changes the gradient only through head h's
    log-prob/distribution: actor modules owned by other heads get the off gradient."""
    models, _, batch, populations = _small(monkeypatch)
    eta = dict(ap.zero_eta(), **{head: 1.0})
    _, prepared = _prepare(models, batch, populations, eta=eta)
    actor = models[0]
    index = torch.arange(batch.parent_count)
    base = labor_policy_minibatch_terms(
        actor, batch, index, _heads_scorer(actor), clip=0.2, entropy_weight=0.0,
        initial_head_advantage={h: batch.initial_advantage for h in INITIAL_POLICY_HEADS},
    )
    base_grads = _grads(actor, base.loss)
    cv = labor_policy_minibatch_terms(
        actor, batch, index, _heads_scorer(actor), clip=0.2, entropy_weight=0.0,
        initial_head_advantage=prepared.coefficients(),
        initial_correction=prepared.correction_fn(actor),
    )
    cv_grads = _grads(actor, cv.loss)
    owned = {
        "movement": ("stay.", "site_query.", "region_keep.", "region_query."),
        "fission": ("fission.",),
        "trade_gate": ("trade_token.",),
        "specialty": ("specialty.",),
        "activity": ("activity.",),
    }[head]
    others = {
        "fission.", "trade_share.", "stay.", "site_query.", "specialty.", "activity.",
        "trade_token.", "region_keep.", "region_query.",
    } - set(owned)
    moved = 0
    for name in cv_grads:
        if any(name.startswith(prefix) for prefix in others):
            assert torch.allclose(cv_grads[name], base_grads[name], rtol=1e-5, atol=1e-7), name
        if any(name.startswith(prefix) for prefix in owned):
            moved += int(not torch.allclose(cv_grads[name], base_grads[name], atol=1e-9))
    assert moved > 0


# ---------------------------------------------------------------------------
# 3. Categorical E_h equals brute-force enumeration (full re-encoding)
# ---------------------------------------------------------------------------


def _brute_scores(frozen, batch, populations, mutate):
    weights, _ = batch.row_personality_weights()
    contexts = frozen.contexts(batch.initial_tensors, batch.initial_privileged_raw)
    rows = []
    for i in range(batch.parent_count):
        base = schema12_initial_action(batch.initial_labor_sample, batch.initial_nonlabor_sample, i)
        rows.append(
            [la.action_features(a, population=populations[i]) for a in mutate(base)]
        )
    n, k = len(rows), len(rows[0])
    features = torch.from_numpy(np.asarray(rows, dtype=np.float32).reshape(n * k, -1))
    raw = frozen.score(contexts, features, torch.arange(n).repeat_interleave(k))
    return ap.combine_head_units(
        {c: v.reshape(n, k) for c, v in raw.items()}, weights, batch.require_cost_multipliers()
    )


def test_categorical_expectations_equal_brute_force(monkeypatch):
    models, _, batch, populations = _small(monkeypatch)
    frozen, prepared = _prepare(models, batch, populations)
    actor = models[0]
    with torch.no_grad():
        output = actor(batch.initial_tensors)
    movement = _brute_scores(
        frozen, batch, populations,
        lambda a: [dict(a, movement=code) for code in la.MOVEMENT_UNION_CODES],
    )
    fission = _brute_scores(
        frozen, batch, populations,
        lambda a: [dict(a, fission=k) for k in range(contracts.FISSION_POLICIES)],
    )
    assert torch.allclose(prepared.categorical["movement"], movement, rtol=1e-6, atol=1e-6)
    assert torch.allclose(prepared.categorical["fission"], fission, rtol=1e-6, atol=1e-6)
    mask = output["movement_union_mask"]
    p_move = torch.softmax(output["movement_union_logits"].masked_fill(~mask, -torch.inf), 1)
    assert torch.allclose(prepared.expected["movement"], (p_move * movement).sum(1), atol=1e-6)
    p_fission = torch.softmax(output["fission"], 1)
    assert torch.allclose(prepared.expected["fission"], (p_fission * fission).sum(1), atol=1e-6)

    groups = len(contracts.TRADE_GROUPS)
    p_gate = torch.softmax(output["trade_gate"], 2)
    for g in range(groups):
        def mutate(a, g=g):
            out = []
            for k in range(contracts.TRADE_GATES):
                gates = list(a["tradeGate"])
                gates[g] = k
                out.append(dict(a, tradeGate=gates))
            return out

        brute = _brute_scores(frozen, batch, populations, mutate)
        assert torch.allclose(prepared.categorical["trade_gate"][:, g], brute, atol=1e-6)
        assert torch.allclose(
            prepared.expected["trade_gate"][:, g], (p_gate[:, g] * brute).sum(1), atol=1e-6
        )
    # The executed action is one of the enumerated candidates.
    executed_move = [
        la.MOVEMENT_UNION_CODES.index(int(c)) for c in batch.initial_nonlabor_sample["movement"]
    ]
    assert torch.allclose(
        movement[torch.arange(batch.parent_count), executed_move], prepared.a_executed, atol=1e-6
    )
    # Trade-gate factor log-probs sum to the head's log-prob.
    terms = prepared.head_terms(
        actor(batch.initial_tensors), batch.initial_tensors, torch.arange(batch.parent_count),
        ["trade_gate"],
    )
    group_logp = terms["trade_gate"][1][0]
    full = nonlabor_log_prob(actor(batch.initial_tensors), batch.initial_nonlabor_sample)
    assert torch.allclose(group_logp.sum(1), full["trade_gate"], atol=1e-5)


def test_categorical_correction_gradient_is_the_exact_expectation_gradient(monkeypatch):
    models, _, batch, populations = _small(monkeypatch)
    _, prepared = _prepare(models, batch, populations)
    actor = models[0]
    index = torch.arange(batch.parent_count)
    output = actor(batch.initial_tensors)
    term = prepared.head_terms(output, batch.initial_tensors, index, ["fission"])["fission"][0]
    got = torch.autograd.grad(term.sum(), actor.fission.weight)[0]
    manual = (torch.softmax(actor(batch.initial_tensors)["fission"], 1)
              * prepared.categorical["fission"]).sum()
    want = torch.autograd.grad(manual, actor.fission.weight)[0]
    assert torch.allclose(got, want, atol=1e-7)


# ---------------------------------------------------------------------------
# 4. The sampled-head correction is unbiased (leave-one-out baseline)
# ---------------------------------------------------------------------------


def _toy_estimates(mu_start, mu_current, *, baseline, rows=20000, draws=16, sigma=0.7):
    """Per-row gradient estimates d/dmu of E_{N(mu, sigma)}[A], A(x) = x^2 + sin(3x),
    with keyed draws from N(mu_start, sigma) (one keyed generator per row)."""
    noise = torch.stack(
        [
            torch.randn(draws, generator=ap.draw_generator((77,), r, "trade_share"),
                        dtype=torch.float64)
            for r in range(rows)
        ]
    )
    x = mu_start + sigma * noise
    scores = x**2 + torch.sin(3 * x)
    mu = torch.full((rows, 1), float(mu_current), dtype=torch.float64, requires_grad=True)
    normal = lambda m: torch.distributions.Normal(m, sigma)  # noqa: E731
    current = normal(mu).log_prob(x)
    start = normal(torch.full_like(x, mu_start)).log_prob(x)
    if baseline == "loo":
        term = ap.sampled_correction(current, start, scores)
    else:  # plain mean including the sample itself (biased by (M - 1) / M)
        centred = scores - scores.mean(dim=1, keepdim=True)
        term = (torch.exp(current - start) * centred).mean(dim=1)
    (grad,) = torch.autograd.grad(term.sum(), mu)
    return grad.squeeze(1).numpy()


def _toy_exact(mu, sigma=0.7):
    # d/dmu E[x^2 + sin(3x)] = 2 mu + 3 cos(3 mu) exp(-9 sigma^2 / 2)
    return 2 * mu + 3 * math.cos(3 * mu) * math.exp(-4.5 * sigma**2)


def test_sampled_correction_is_unbiased_with_the_leave_one_out_baseline():
    exact = _toy_exact(0.6)
    est = _toy_estimates(0.6, 0.6, baseline="loo")
    se = est.std(ddof=1) / math.sqrt(len(est))
    assert abs(est.mean() - exact) < 4 * se, (est.mean(), exact, se)
    # The plain mean baseline is biased by the factor (M - 1) / M and the test sees it.
    plain = _toy_estimates(0.6, 0.6, baseline="plain")
    se_plain = plain.std(ddof=1) / math.sqrt(len(plain))
    assert abs(plain.mean() - exact * 15 / 16) < 4 * se_plain
    assert abs(plain.mean() - exact) > 4 * se_plain
    # Later epochs: draws from the iteration-start policy, importance-weighted to the
    # current one, stay unbiased for the CURRENT policy's gradient.
    moved = _toy_estimates(0.6, 0.75, baseline="loo")
    se_moved = moved.std(ddof=1) / math.sqrt(len(moved))
    assert abs(moved.mean() - _toy_exact(0.75)) < 4 * se_moved
    # Value: the plain estimate of E_h; the carrier adds nothing to the value.
    scores = torch.tensor([[1.0, 2.0, 4.0]])
    logp = torch.zeros(1, 3, requires_grad=True)
    value = ap.sampled_correction(logp, logp.detach(), scores)
    assert torch.allclose(value, torch.tensor([7.0 / 3.0]))
    (g,) = torch.autograd.grad(value.sum(), logp)
    assert torch.allclose(g, ap.leave_one_out_centred(scores) / 3)
    assert torch.allclose(ap.leave_one_out_centred(scores), torch.tensor([[-2.0, -0.5, 2.5]]))


# ---------------------------------------------------------------------------
# 8. Determinism and keyed draws
# ---------------------------------------------------------------------------


def test_draws_are_keyed_per_row_and_head_and_order_free(monkeypatch):
    models, _, batch, populations = _small(monkeypatch)
    _, first = _prepare(models, batch, populations, key=(5, 3))
    _, again = _prepare(models, batch, populations, key=(5, 3))
    _, other = _prepare(models, batch, populations, key=(5, 4))
    for head in ("specialty", "activity"):
        assert torch.equal(first.draws[head].scores, again.draws[head].scores)
        for name, value in first.draws[head].sample.items():
            assert torch.equal(value, again.draws[head].sample[name])
        assert not torch.equal(first.draws[head].scores, other.draws[head].scores)
    assert set(first.draws) == {"specialty", "activity"}  # trade value/share are A-blind
    assert set(first.blind) == {"trade_value", "trade_share"}
    # Row 2's draws come from its own keyed generator (independent of the other rows).
    m = first.draws["specialty"].draws
    output = models[0](batch.initial_tensors)
    from anthropocene_rl.autonomous_v7.labor_model import sample_labor_policy
    from anthropocene_rl.autonomous_v7.labor_ppo import _rows

    local = torch.full((m,), 2, dtype=torch.long)
    masks = _rows({k: batch.initial_tensors[k] for k in ap._LABOR_MASK_KEYS}, local)
    with torch.no_grad():
        drawn = sample_labor_policy(
            _rows(output, local), masks, generator=ap.draw_generator((5, 3), 2, "specialty")
        )
    assert torch.equal(
        first.draws["specialty"].sample["specialty_latent"][2 * m:3 * m],
        drawn["specialty_latent"],
    )
    # The estimator does not depend on the order of the M draws.
    actor = models[0]
    index = torch.arange(batch.parent_count)
    term = first.head_terms(actor(batch.initial_tensors), batch.initial_tensors, index,
                            ["specialty"])["specialty"][0]
    grads = torch.autograd.grad(term.sum(), actor.specialty.weight)[0]
    permuted = copy.copy(first)
    perm = torch.randperm(m, generator=torch.Generator().manual_seed(0))
    draws = first.draws["specialty"]
    order = (torch.arange(batch.parent_count)[:, None] * m + perm[None, :]).reshape(-1)
    permuted.draws = dict(first.draws)
    permuted.draws["specialty"] = dataclasses.replace(
        draws,
        sample={k: v.index_select(0, order) for k, v in draws.sample.items()},
        start_logp=draws.start_logp[:, perm],
        scores=draws.scores[:, perm],
    )
    term_p = permuted.head_terms(actor(batch.initial_tensors), batch.initial_tensors, index,
                                 ["specialty"])["specialty"][0]
    grads_p = torch.autograd.grad(term_p.sum(), actor.specialty.weight)[0]
    assert torch.allclose(term, term_p, atol=1e-6)
    assert torch.allclose(grads, grads_p, rtol=1e-5, atol=1e-8)


def test_d7_variance_is_the_stage2_gradient_variance(monkeypatch):
    """The streamed trace equals ``labor_a_stage2.gradient_variance`` on the explicit
    per-row gradient matrices; A-blind heads have exactly zero reduction."""
    models, _, batch, populations = _small(monkeypatch)
    _, prepared = _prepare(models, batch, populations)
    actor = models[0]
    measured, _ = ap.measure_d7(prepared, actor, rows=batch.parent_count)
    params = [p for p in actor.parameters() if p.requires_grad]
    rows = {h: ([], []) for h in INITIAL_POLICY_HEADS}
    for r in range(batch.parent_count):
        for h, (v_only, cv) in ap.per_row_head_estimates(prepared, actor, r).items():
            rows[h][0].append(ap._flat_gradient(v_only, params).double())
            rows[h][1].append(ap._flat_gradient(cv, params).double())
    ones = torch.ones(batch.parent_count, dtype=torch.float64)
    for h in INITIAL_POLICY_HEADS:
        v = s2.gradient_variance(torch.stack(rows[h][0]), ones)
        c = s2.gradient_variance(torch.stack(rows[h][1]), ones)
        assert measured[h]["varianceVOnly"] == pytest.approx(v, rel=1e-6, abs=1e-12)
        assert measured[h]["varianceControlVariate"] == pytest.approx(c, rel=1e-6, abs=1e-12)
        if h in prepared.blind:
            assert measured[h]["reduction"] == 0.0 and not measured[h]["admit"]
    # V-only estimate of a head = grad log pi_h(a) * GAE.
    assert measured["fission"]["varianceVOnly"] > 0


# ---------------------------------------------------------------------------
# 7. Warm-up switch (S2, B5)
# ---------------------------------------------------------------------------


def test_warmup_constants_and_no_events_exclusion():
    assert s2.TD_SPLIT_BIAS_FRACTION == 0.05
    assert s2.DRY_RUN_KL_BOUND == 0.02 and s2.WARMUP_KL_CAP == 0.005
    assert s2.V_SLOPE_BAND == (0.8, 1.2) and s2.V_BIAS_FRACTION == 0.25

    def report(spread_feas=0.0, td_feas=None, td_qol=0.04, kl=(0.01,), slope_feas=None):
        return s2.WarmupReport(
            calibration_slope={"qol_years": 1.0, "feasibility": slope_feas},
            calibration_bias={"qol_years": 0.1, "feasibility": None},
            realised_spread={"qol_years": 1.0, "feasibility": spread_feas},
            td_by_world={"qol_years": {1: td_qol, 2: -td_qol},
                         "feasibility": td_feas or {}},
            dry_run_kl=kl,
        )

    sw = s2.WarmupSwitch()
    # A no-events component is excluded (not failed) even though its slope is missing.
    d = sw.update(report())
    assert d.passed and d.phase == "gae" and d.no_events == ("feasibility",)
    assert any("feasibility: no events" in r for r in d.reasons)
    # Per-component TD: the qol bound is 0.05 x qol's own spread.
    assert not s2.WarmupSwitch().update(report(td_qol=0.06)).gates["td_split"]
    # A component WITH events is gated on its own spread and TD.
    with_events = report(spread_feas=2.0, slope_feas=1.0, td_feas={1: 0.2, 2: 0.0})
    bad = s2.WarmupSwitch().update(dataclasses.replace(
        with_events, calibration_bias={"qol_years": 0.1, "feasibility": 0.0}))
    assert not bad.gates["td_split"] and any("feasibility fails in worlds [1]" in r
                                              for r in bad.reasons)
    ok = s2.WarmupSwitch().update(dataclasses.replace(
        with_events, calibration_bias={"qol_years": 0.1, "feasibility": 0.0},
        td_by_world={"qol_years": {1: 0.0}, "feasibility": {1: 0.1, 2: -0.1}}))
    assert ok.passed
    # No component with events: no vacuous pass.
    empty = s2.WarmupReport({"qol_years": None}, {"qol_years": None}, {"qol_years": 0.0},
                            {"qol_years": {}}, (0.0,))
    none = s2.WarmupSwitch().update(empty)
    assert not none.gates["v_calibration"] and not none.gates["td_split"] and not none.passed
    # Dry-run KL not measured fails closed.
    unmeasured = s2.WarmupSwitch().update(report(kl=None))
    assert not unmeasured.gates["dry_run_kl"] and "dry-run KL not measured" in unmeasured.reasons
    assert not s2.WarmupSwitch().update(report(kl=(0.03,))).gates["dry_run_kl"]


def test_warmup_report_realised_returns_and_td():
    """Hand-checked realised returns, TD(0) and world-balanced calibration."""
    g = math.exp(-20.0 / contracts.HORIZON_YEARS)
    rewards = lambda q: {"qol_years": q, "population_years": 0.0, "feasibility": 0.0,  # noqa
                         "settlement_guard": 0.0}
    meta = {
        "decisionRows": [
            {"world": 1, "camp": 7, "turn": 3, "years": 20.0, "terminated": False,
             "population": 10, "rewards": rewards(1.0)},
            {"world": 1, "camp": 7, "turn": 4, "years": 20.0, "terminated": False,
             "population": 10, "rewards": rewards(2.0)},
            {"world": 2, "camp": 7, "turn": 3, "years": 20.0, "terminated": True,
             "population": 10, "rewards": rewards(3.0)},
        ],
        "bootstrapValues": [{"world": 1, "camp": 7, "values": {
            "qol_years": 5.0, "population_years": 0.0, "feasibility": 0.0,
            "settlement_guard": 0.0}}],
    }
    values = {n: torch.zeros(3) for n in (*LABOR_OBJECTIVES, *LABOR_COSTS)}
    values["qol_years"] = torch.tensor([4.0, 6.0, 1.0])
    report, detail = ap.warmup_report(values, meta, dry_run_kl=None)
    g1 = 2.0 + g * 5.0
    g0 = 1.0 + g * g1
    assert detail["components"]["qol_years"]["perWorld"]["2"]["spread"] == 0.0
    td = detail["components"]["qol_years"]["tdByWorld"]
    assert td["1"]["mean"] == pytest.approx(((1 + g * 6 - 4) + (2 + g * 5 - 6)) / 2)
    assert td["2"]["mean"] == pytest.approx(3.0 - 1.0)
    assert report.realised_spread["qol_years"] == pytest.approx(
        math.sqrt((np.std([g0, g1]) ** 2 + 0.0) / 2)
    )
    assert report.realised_spread["feasibility"] == 0.0  # no events
    assert report.dry_run_kl is None and detail["dryRunKL"] == "not measured"
    old = {"decisionRows": [{"world": 1, "camp": 7, "turn": 3, "years": 20.0,
                             "terminated": True}]}
    assert ap.warmup_report(values, old, dry_run_kl=None)[0] is None


# ---------------------------------------------------------------------------
# 5, 6, 9. End to end: train --a-mode gae (default-width models, synthetic rollout)
# ---------------------------------------------------------------------------


E2E_PLAN = ((False, 71), (True, 72), (False, 73), (False, 74), (True, 75))
# (world, turn) per row: two 2-decision sequences and one single decision.
E2E_IDENTITY = ((50000, 3), (50000, 4), (50001, 3), (50001, 4), (50002, 3))


def _default_models(seed=1):
    from test_labor_privileged_bridge import _normalizer

    village = _village_at(7)
    normalizer = Normalizer.fit_bounded(observation_batch([village]))
    _, tensors = labor_policy_tensors([village], normalizer)
    layout = _layout(core=tensors["core"].shape[1])
    torch.manual_seed(seed)
    actor, task, costs = LaborActor(layout), LaborTaskCritic(layout), LaborCostCritics(layout)
    bind_privileged_normalizer(_normalizer(), task, costs)
    return actor, task, costs, normalizer, layout


def _trainer_state(actor, task, costs, normalizer, layout, iteration=0):
    return training._trainer_checkpoint(
        actor, task, costs, normalizer, layout,
        torch.optim.Adam(actor.parameters(), lr=1e-6),  # the loaded state sets the lr
        torch.optim.Adam([*task.parameters(), *costs.parameters()]),
        personality=LaborPersonality(), optimizer_seed=1, iteration=iteration,
        sampling_generator=torch.Generator().manual_seed(1),
        shuffle_rng=np.random.default_rng(1),
        critic_training_world_seeds=[], critic_training_independent_unit_ids=[],
    )


def _e2e_rows(monkeypatch, models):
    rows, bootstrap = [], {}
    for (retry, seed), (world, turn) in zip(E2E_PLAN, E2E_IDENTITY, strict=True):
        row, tail, *_ = _collect(monkeypatch, retry, models, seed=seed, world=world)
        row.turn = turn
        rows.append(row)
        bootstrap[(world, 7)] = tail  # the last decision's successor of each sequence
    return rows, bootstrap


def _iteration_files(monkeypatch, tmp_path, state, tag):
    """Collect a rollout from ``state`` (real collect command, fake bridge) plus its gate
    report and anchor. Returns (checkpoint, rollout, anchor, gate) paths."""
    _, actor, task, costs, normalizer = training._models_from_checkpoint(state)
    checkpoint = tmp_path / f"{tag}.pt"
    torch.save(state, checkpoint)
    rows, bootstrap = _e2e_rows(monkeypatch, (actor, task, costs))
    monkeypatch.setattr(training, "collect_labor_rollout", lambda **kw: (rows, bootstrap))
    rollout = tmp_path / f"{tag}-rollout.pt"
    training.main(["collect", "--checkpoint", str(checkpoint), "--worlds",
                   "50000,50001,50002", "--turns", "2", "--out", str(rollout)])
    behavior = behavior_policy_digest(
        actor, training.labor_behavior_configuration(LaborPersonality())
    )
    source = {
        "sourcePolicyDigest": behavior,
        "normalizationBindingDigest": training._normalization_digest(normalizer),
        "privilegedNormalizerDigest": task.privileged_normalizer.digest(),
        "worldSeeds": [50005],
    }
    row = rows[0]

    class Anchor:
        def __init__(self):
            self.metadata = {"source": source}
            self.independent_unit_ids = np.array(["world-50005"])
            self.privileged_raw = row.initial_privileged_raw
            self.tensors = {k: v.numpy() for k, v in row.initial_tensors.items()}
            self.targets_raw = {n: np.array([0.0])
                                for n in (*row.initial_task_values, *row.initial_cost_values)}
            self.target_masks = {n: np.array([True]) for n in self.targets_raw}

        def __len__(self):
            return 1

    anchor = Anchor()
    monkeypatch.setattr(training, "load_labor_anchor", lambda *args: anchor)
    anchor_path = tmp_path / f"{tag}-anchor.npz"
    anchor_path.write_bytes(b"synthetic anchor unit fixture")
    gate = {
        "kind": "model9-labor-value-calibration-report-v1",
        "dataRole": "actor-gate",
        "localReadinessClaim": False,
        "actorGateOpen": True,
        "policyCurrent": True,
        "normalizationCurrent": True,
        "registryCurrent": True,
        "enabledHeads": list(LABOR_GATE_HEADS),
        "heads": {name: {"pass": True} for name in LABOR_GATE_HEADS},
        "sourcePolicyDigest": behavior,
        "sourceCriticDigest": labor_critic_digest(task, costs),
        "normalizationBindingDigest": training._normalization_digest(normalizer),
        "dataSha256": "e" * 64,
        "worldSeeds": [50010],
        "independentUnitIds": ["world-50010"],
    }
    gate_path = tmp_path / f"{tag}-gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return checkpoint, rollout, anchor_path, gate_path


def _train_args(checkpoint, rollout, anchor, gate, out, *extra):
    return ["train", "--checkpoint", str(checkpoint), "--rollout", str(rollout),
            "--anchor", str(anchor), "--calibration-report", str(gate), "--out", str(out),
            "--epochs", "2", "--batch-size", "2", "--target-kl", "10", "--actor-lr", "1e-6",
            *extra]


def _same(a, b, path="state"):
    assert type(a) is type(b), path
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert set(a) == set(b), path
        for key in a:
            _same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


@pytest.fixture()
def e2e(monkeypatch, tmp_path):
    actor, task, costs, normalizer, layout = _default_models()
    state0 = _trainer_state(actor, task, costs, normalizer, layout)
    files0 = _iteration_files(monkeypatch, tmp_path, state0, "it0")
    warm = tmp_path / "warm-start.pt"
    torch.save(_a_payload(task, costs), warm)
    return {"state0": state0, "files0": files0, "warm": warm, "tmp": tmp_path,
            "critic": files0[0]}


def test_train_off_is_bitwise_the_prechange_train_and_refuses_a_arguments(e2e, monkeypatch):
    old = _prechange("labor_training")
    checkpoint, rollout, anchor, gate = e2e["files0"]
    monkeypatch.setattr(old, "load_labor_anchor", training.load_labor_anchor)
    tmp = e2e["tmp"]
    outputs = {}
    for tag, main, extra in (
        ("old", old.main, ()),
        ("default", training.main, ()),
        ("off", training.main, ("--a-mode", "off")),
    ):
        out = tmp / f"{tag}.pt"
        main(_train_args(checkpoint, rollout, anchor, gate, out, *extra))
        outputs[tag] = torch.load(out, weights_only=True)
    assert outputs["old"]["last_update"]["actor"]["acceptedSteps"] > 0
    _same(outputs["default"], outputs["off"])
    # Since the two-group actor optimizer (labor_actor_optim) the checkpoint's Adam state is
    # laid out in named encoder/heads groups and train records its optimizer configuration;
    # the per-parameter moments/steps and everything else are still the pre-change bits.
    from anthropocene_rl.autonomous_v7.labor_actor_optim import actor_optimizer_state_by_name

    old_state, new_state = outputs["old"], outputs["default"]
    for key in set(old_state) | set(new_state):
        if key not in ("actor_optimizer", "last_update"):
            _same(old_state[key], new_state[key], key)
    old_update, new_update = dict(old_state["last_update"]), dict(new_state["last_update"])
    new_update.pop("actorOptimizer")
    new_update["actor"] = {k: v for k, v in new_update["actor"].items() if k != "actorOptimizer"}
    _same(old_update, new_update, "last_update")
    _, actor, *_ = training._models_from_checkpoint(new_state)
    _same(
        actor_optimizer_state_by_name(actor, old_state["actor_optimizer"]),
        actor_optimizer_state_by_name(actor, new_state["actor_optimizer"]),
        "actor_optimizer.state",
    )
    assert ap.A_STATE_KEY not in outputs["off"]
    with pytest.raises(ValueError, match="require --a-mode gae"):
        training.main(_train_args(checkpoint, rollout, anchor, gate, tmp / "x.pt",
                                  "--a-warm-start", str(e2e["warm"])))
    with pytest.raises(ValueError, match="requires --a-warm-start and --a-critic"):
        training.main(_train_args(checkpoint, rollout, anchor, gate, tmp / "x.pt",
                                  "--a-mode", "gae", "--a-warm-start", str(e2e["warm"])))


def test_train_gae_end_to_end_lagged_d7_frozen_critic_and_exact_resume(
    e2e, monkeypatch, capsys
):
    checkpoint, rollout, anchor, gate = e2e["files0"]
    tmp = e2e["tmp"]
    gae = ("--a-mode", "gae", "--a-warm-start", str(e2e["warm"]), "--a-critic", str(e2e["critic"]))
    # Every measurable non-blind head is admitted (to exercise the lag with eta = 1).
    monkeypatch.setattr(ap, "D7_MIN_VARIANCE_REDUCTION", -1e30)
    out1 = tmp / "it1.pt"
    training.main(_train_args(checkpoint, rollout, anchor, gate, out1, *gae))
    printed = json.loads(next(
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"aControlVariate"')
    ))
    assert set(printed["aControlVariate"]["seconds"]) >= {"prepare", "d7"}
    state1 = torch.load(out1, weights_only=True)
    record = state1["last_update"]["aControlVariate"]
    # Iteration 0: eta = 0 everywhere; measured only.
    assert record["etaUsed"] == ap.zero_eta()
    assert record["etaSource"] == "first-gae-iteration-measure-only"
    assert state1["last_update"]["actor"]["acceptedSteps"] > 0
    stored = state1[ap.A_STATE_KEY]
    assert stored["measuredIteration"] == 0 and stored["nextEta"] == record["nextEta"]
    blind = {"trade_value", "trade_share"}
    admitted = {h for h, e in stored["nextEta"].items() if e == 1.0}
    assert admitted == set(INITIAL_POLICY_HEADS) - blind
    for h in INITIAL_POLICY_HEADS:
        entry = record["d7"][h]
        assert set(entry) >= {"varianceVOnly", "varianceControlVariate", "reduction", "admit"}
        assert entry["aBlind"] == (h in blind)
    # The frozen critic is the iteration-0 critic and unchanged; the online V moved.
    _, _, task0, costs0, _ = training._models_from_checkpoint(e2e["state0"])
    _, _, task1, costs1, _ = training._models_from_checkpoint(state1)
    assert record["frozen"]["criticDigest"] == labor_critic_digest(task0, costs0)
    assert labor_critic_digest(task1, costs1) != labor_critic_digest(task0, costs0)
    assert record["frozen"]["unchangedAfterIteration"] is True
    warmup = record["warmup"]
    assert warmup["status"] == "measured" and warmup["dryRunKL"] == "not measured"
    assert warmup["effect"] == "telemetry-only-never-changes-training"
    assert warmup["decision"]["gates"]["dry_run_kl"] is False
    assert {"feasibility", "settlement_guard"} <= set(warmup["components"])

    # Iteration 1 applies the eta measured on iteration 0's batch.
    files1 = _iteration_files(monkeypatch, tmp, state1, "it1src")
    outs = []
    for tag in ("a", "b"):  # a resume from the same checkpoint reproduces it exactly
        out = tmp / f"it2-{tag}.pt"
        training.main(_train_args(*files1, out, *gae))
        outs.append(torch.load(out, weights_only=True))
    record2 = outs[0]["last_update"]["aControlVariate"]
    assert record2["etaUsed"] == stored["nextEta"]
    assert record2["etaSource"] == "lagged-d7-from-previous-batch"
    assert outs[0]["last_update"]["actor"]["acceptedSteps"] > 0
    _same(outs[0], outs[1])
    assert outs[0][ap.A_STATE_KEY]["measuredIteration"] == 1
    # With heads admitted the update differs from plain GAE PPO on the same files.
    plain = tmp / "it2-off.pt"
    training.main(_train_args(*files1, plain))
    plain_state = torch.load(plain, weights_only=True)
    _same(plain_state["task_critic"], outs[0]["task_critic"])  # the online V path is untouched
    assert any(
        not torch.equal(value, outs[0]["actor"][name])
        for name, value in plain_state["actor"].items()
    )

    # The telemetry dry run never changes training.
    out = tmp / "it2-dry.pt"
    training.main(_train_args(*files1, out, *gae, "--a-telemetry-dry-run"))
    dry = torch.load(out, weights_only=True)
    kl = dry["last_update"]["aControlVariate"]["warmup"]["dryRunKL"]
    assert len(kl) == 2 and all(value >= 0 for value in kl)
    for key in ("actor", "task_critic", "cost_critics", "actor_optimizer"):
        _same(dry[key], outs[0][key])
    assert dry["last_update"]["actor"] == outs[0]["last_update"]["actor"]

    # A stale or mismatched D7 state is refused, not silently reset.
    _, _, _, _, normalizer = training._models_from_checkpoint(state1)
    frozen = ap.load_frozen_a(
        torch.load(e2e["warm"], weights_only=True), e2e["state0"],
        normalization_binding_digest=training._normalization_digest(normalizer),
    )
    assert ap.eta_for_iteration(stored, iteration=1, frozen=frozen) == (
        stored["nextEta"], "lagged-d7-from-previous-batch"
    )
    with pytest.raises(ValueError, match="preceding iteration"):
        ap.eta_for_iteration(stored, iteration=3, frozen=frozen)
    with pytest.raises(ValueError, match="another frozen A"):
        ap.eta_for_iteration(dict(stored, headDigest="0" * 64), iteration=1, frozen=frozen)


def test_frozen_a_loads_only_against_its_critic_and_normalizer(e2e, tmp_path):
    state0 = e2e["state0"]
    payload = torch.load(e2e["warm"], weights_only=True)
    *_, normalizer = training._models_from_checkpoint(state0)
    binding = training._normalization_digest(normalizer)
    frozen = ap.load_frozen_a(payload, state0, normalization_binding_digest=binding)
    assert not any(p.requires_grad for p in frozen._parameters())
    assert not frozen.task_critic.training and not frozen.head.training
    frozen.assert_unchanged()
    other_actor, other_task, other_costs, _, other_layout = _default_models(seed=2)
    other = _trainer_state(other_actor, other_task, other_costs, normalizer, other_layout)
    with pytest.raises(ValueError, match="different critic"):
        ap.load_frozen_a(payload, other, normalization_binding_digest=binding)
    with pytest.raises(ValueError, match="observation normalizer"):
        ap.load_frozen_a(payload, state0, normalization_binding_digest="f" * 64)
    plain = {k: v for k, v in payload.items() if k != "warm_start"}
    with pytest.raises(ValueError, match="warm-start artifact"):
        ap.load_frozen_a(plain, state0, normalization_binding_digest=binding)
    with torch.no_grad():
        next(frozen.task_critic.parameters()).add_(1.0)
    with pytest.raises(AssertionError, match="frozen A critic changed"):
        frozen.assert_unchanged()


def test_gae_refuses_rollouts_without_populations(e2e, monkeypatch):
    checkpoint, rollout, anchor, gate = e2e["files0"]
    recorded = torch.load(rollout, weights_only=True)
    for row in recorded["metadata"]["decisionRows"]:
        del row["population"]
    old = e2e["tmp"] / "old-rollout.pt"
    torch.save(recorded, old)
    gae = ("--a-mode", "gae", "--a-warm-start", str(e2e["warm"]), "--a-critic", str(e2e["critic"]))
    with pytest.raises(ValueError, match="predates recorded parent populations"):
        training.main(_train_args(checkpoint, old, anchor, gate, e2e["tmp"] / "x.pt", *gae))


def test_train_cli_a_mode_defaults_off():
    parser = training.argument_parser()
    base = ["train", "--checkpoint", "c.pt", "--rollout", "r.pt", "--anchor", "a.pt",
            "--out", "o.pt"]
    args = parser.parse_args(base)
    assert args.a_mode == "off" and args.a_warm_start is None and args.a_critic is None
    assert args.a_draws == ap.DEFAULT_DRAWS == 16 and args.a_d7_rows == ap.DEFAULT_D7_ROWS
    assert args.a_telemetry_dry_run is False
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--a-mode", "q"])
