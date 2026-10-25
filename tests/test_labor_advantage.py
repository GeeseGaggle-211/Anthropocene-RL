"""Intervention B (model side): action vector, dueling A head, pairs, fit, evaluator, ranker.

Covers the Q-pi isolation (closed gate: no gradient reaches the actor; the
actor batch never carries privileged keys), the frozen critic (V untouched),
same-root/same-horizon/same-stream pairing, the world split and reserved-world
refusal, the preregistered evaluator on synthetic payloads in the consumer
schema, the search ranker hook and the advantage checkpoint binding.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
from test_labor_model import _labor_tensors
from test_labor_privileged_bridge import _raw
from test_labor_privileged_model import _critics, _layout

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7 import labor_advantage as la
from anthropocene_rl.autonomous_v7 import labor_advantage_eval as lae
from anthropocene_rl.autonomous_v7.labor_bridge import ACTIVITIES, SPECIALTIES
from anthropocene_rl.autonomous_v7.labor_model import (
    LaborActor,
    LaborCostCritics,
    LaborTaskCritic,
)
from anthropocene_rl.autonomous_v7.labor_privileged import CRITIC_PREFIX, SHARD_PREFIX
from anthropocene_rl.autonomous_v7.labor_value_calibration import labor_critic_digest


def _plain_critics(seed: int = 3):
    return _critics(_layout(False), seed=seed)


def _action(seed: int = 0, **kwargs) -> dict:
    return la._synthetic_action(np.random.default_rng(seed), **kwargs)


def _open_gate() -> la.QPiGate:
    return la.QPiGate(
        report_digest="test",
        worlds=tuple(la.RESERVED_HELDOUT_WORLDS),
        ranking={"ranking:qol_years": 0.9, "ranking:utility:root": 0.9},
        false_improvement=0.05,
        idle_control=True,
        delayed_benefit=True,
        authorized_by="unit-test",
    )


# --- action vector -------------------------------------------------------------------------


def test_action_vector_layout_is_fixed_and_bound_by_stable_id() -> None:
    assert la.ACTION_FEATURE_WIDTH == 14 + 13 * 22 + 17 + 3 + 21 + 7 + 1 == 349
    assert dict(la.ACTION_BLOCKS)["movement_onehot"] == 17
    assert la.MOVEMENT_UNION_CODES[0] == -1 and la.MOVEMENT_UNION_CODES[8] == contracts.KEEP_INTENT
    action = _action(1)
    vector = la.action_features(action)
    assert vector.shape == (349,) and vector.dtype == np.float32
    spec = np.asarray(action["laborSpecialtyAllocation"], dtype=np.float32)
    assert np.allclose(vector[:14], spec)
    # The same action declared in another stable-ID order lands in the same slots.
    reordered, spec_ids, act_ids = lae._reordered(action)
    assert spec_ids != list(SPECIALTIES) and act_ids != list(ACTIVITIES)
    assert np.array_equal(
        la.action_features(reordered, specialty_ids=spec_ids, activity_ids=act_ids), vector
    )
    # Reading the reordered rows as if canonical gives a different vector.
    assert not np.array_equal(la.action_features(reordered), vector)
    # Trade groups bind by ID as well.
    groups = list(reversed(contracts.TRADE_GROUPS))
    swapped = dict(
        action, tradeValue=action["tradeValue"][::-1], tradeGate=action["tradeGate"][::-1]
    )
    assert np.array_equal(la.action_features(swapped, trade_groups=groups), vector)
    # Bookkeeping keys never change the vector.
    noisy = dict(action, description="x", digest="y", schemaVersion=99)
    assert np.array_equal(la.action_features(noisy), vector)


def test_action_vector_blocks_and_validation() -> None:
    action = _action(2)
    action.update(movement=contracts.REGION_BASE + 3, fission=2, tradeShare=0.25)
    action["tradeValue"] = [10.0, 0.01, 1.0, 2.0, 0.5, 4.0, 0.25]
    action["tradeGate"] = [0, 1, 2, 0, 1, 2, 0]
    vector = la.action_features(action)
    start = dict()
    cursor = 0
    for name, width in la.ACTION_BLOCKS:
        start[name] = (cursor, cursor + width)
        cursor += width
    movement = vector[slice(*start["movement_onehot"])]
    assert movement.sum() == 1.0 and movement[9 + 3] == 1.0
    assert vector[slice(*start["fission_onehot"])].tolist() == [0.0, 0.0, 1.0]
    gate = vector[slice(*start["trade_gate_onehot"])].reshape(7, 3)
    assert gate.argmax(axis=1).tolist() == action["tradeGate"]
    logs = vector[slice(*start["trade_value_log"])]
    assert math.isclose(float(logs[0]), 1.0, abs_tol=1e-6)  # clipped at 4
    assert math.isclose(float(logs[1]), -1.0, abs_tol=1e-6)  # clipped at 0.25
    assert vector[-1] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="movement union"):
        la.action_features(dict(action, movement=contracts.NATIVE_MOVEMENT))
    with pytest.raises(ValueError, match="fission"):
        la.action_features(dict(action, fission=3))
    with pytest.raises(ValueError, match="schema-12 fields"):
        la.action_features({k: v for k, v in action.items() if k != "tradeShare"})
    with pytest.raises(ValueError, match="permutation"):
        la.action_features(action, specialty_ids=["idle", *SPECIALTIES[1:-1], "Unknown"])


# --- Q-pi path and isolation --------------------------------------------------------------


def _actor_and_head(privileged: bool):
    layout = _layout(privileged)
    torch.manual_seed(11)
    actor = LaborActor(layout, width=16, token_width=16)
    task, costs = _critics(layout)
    head = la.LaborAdvantageHead.for_critics(task, costs, embedding=16, hidden=16)
    # Nonzero last layer so A depends on the action (zero-init would hide gradients).
    for module in head.heads.values():
        torch.nn.init.normal_(module[2].weight, std=0.5)
    return actor, task, costs, head


def test_reparameterized_sample_with_closed_gate_sends_no_gradient_to_the_actor() -> None:
    actor, task, costs, head = _actor_and_head(privileged=True)
    tensors = _labor_tensors(batch=2)
    raw = _raw(2)
    output = actor(tensors)
    features = la.reparameterized_action_features(
        output, tensors, generator=torch.Generator().manual_seed(3)
    )
    assert features.shape == (2, la.ACTION_FEATURE_WIDTH)
    assert not features.requires_grad
    contexts = la.critic_contexts(task, costs, tensors, raw)
    values = head(contexts, features, torch.arange(2))
    sum(v.sum() for v in values.values()).backward()
    assert all(p.grad is None for p in actor.parameters())
    assert all(p.grad is None for p in task.parameters()) and all(
        p.grad is None for p in costs.parameters()
    )
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_open_gate_is_the_only_differentiable_q_pi_path() -> None:
    actor, task, costs, head = _actor_and_head(privileged=False)
    tensors = _labor_tensors(batch=2)
    gate = _open_gate()
    assert gate.open and la.q_pi_enabled(gate)
    for closed in (
        None,
        la.QPiGate(**{**gate.__dict__, "authorized_by": ""}),
        la.QPiGate(**{**gate.__dict__, "worlds": (90100,)}),
        la.QPiGate(**{**gate.__dict__, "false_improvement": 0.2}),
        la.QPiGate(**{**gate.__dict__, "idle_control": False}),
    ):
        assert not la.q_pi_enabled(closed)
    features = la.reparameterized_action_features(
        actor(tensors), tensors, generator=torch.Generator().manual_seed(3), gate=gate
    )
    assert features.requires_grad
    contexts = la.critic_contexts(task, costs, tensors)
    sum(v.sum() for v in head(contexts, features, torch.arange(2)).values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
    # Even an open gate never lets A reach the critic it reads.
    assert all(p.grad is None for p in task.parameters())


def test_actor_batch_never_carries_privileged_keys() -> None:
    actor, task, costs, _ = _actor_and_head(privileged=True)
    tensors = _labor_tensors(batch=2)
    raw = _raw(2)
    output = actor(tensors)
    leaked = dict(tensors, **{SHARD_PREFIX + "core": raw[SHARD_PREFIX + "core"]})
    with pytest.raises(AssertionError, match="privileged"):
        la.reparameterized_action_features(output, leaked, generator=torch.Generator())
    with pytest.raises(AssertionError, match="privileged"):
        la.critic_contexts(task, costs, dict(tensors, **{CRITIC_PREFIX + "core": torch.zeros(2)}))
    with pytest.raises(AssertionError, match="privileged"):
        actor(leaked)
    # Privileged state reaches the critic contexts only through the separate argument.
    with_priv = la.critic_contexts(task, costs, tensors, raw)
    other = la.critic_contexts(task, costs, tensors, _raw(2, seed=9))
    assert not torch.equal(with_priv["qol_years"], other["qol_years"])
    # The actor's outputs, hence its features, do not depend on privileged state.
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    assert torch.equal(
        la.reparameterized_action_features(actor(tensors), tensors, generator=g1),
        la.reparameterized_action_features(actor(tensors), tensors, generator=g2),
    )


def test_reparameterized_sample_respects_masks_and_simplexes() -> None:
    actor, *_ = _actor_and_head(privileged=False)
    tensors = _labor_tensors(batch=3)
    tensors["labor_specialty_mask"][:, 5] = False
    tensors["labor_activity_mask"][:, 4, 1:] = False
    tensors["movement_mask"][:, 3] = False
    features = la.reparameterized_action_features(
        actor(tensors), tensors, generator=torch.Generator().manual_seed(1)
    )
    assert torch.allclose(features[:, :14].sum(-1), torch.ones(3))
    assert (features[:, 5] == 0).all()
    activity = features[:, 14 : 14 + 286].view(3, 13, 22)
    assert torch.allclose(activity.sum(-1), torch.ones(3, 13))
    movement = features[:, 300:317]
    assert torch.allclose(movement.sum(-1), torch.ones(3)) and (movement[:, 3] == 0).all()


# --- head, centring, Q = V + A, frozen critic ---------------------------------------------


def test_untrained_head_is_zero_and_q_equals_v_and_centring_is_per_root() -> None:
    task, costs = _plain_critics()
    head = la.LaborAdvantageHead.for_critics(task, costs, embedding=16, hidden=16)
    tensors = _labor_tensors(batch=2)
    contexts = la.critic_contexts(task, costs, tensors)
    features = torch.as_tensor(np.stack([la.action_features(_action(s)) for s in range(5)]))
    index = torch.tensor([0, 0, 0, 1, 1])
    raw = head(contexts, features, index)
    assert all(torch.equal(v, torch.zeros(5)) for v in raw.values())
    v = la.state_values(task, costs, tensors)
    q = la.q_values(v, raw, index, 2)
    for name in la.COMPONENTS:
        assert torch.equal(q[name], v[name][index])
    values = torch.tensor([1.0, 2.0, 6.0, -1.0, 3.0])
    centred = la.centre(values, index, 2)
    assert torch.allclose(centred, torch.tensor([-2.0, -1.0, 3.0, -2.0, 2.0]))


def test_head_widths_and_parameter_counts() -> None:
    task = LaborTaskCritic(_layout(False))
    costs = LaborCostCritics(_layout(False))
    head = la.LaborAdvantageHead.for_critics(task, costs)
    assert head.widths.context == {
        "qol_years": 256 + 128,
        "population_years": 256 + 128,
        "feasibility": 128 + 128,
        "settlement_guard": 128 + 128,
    }
    counts = la.parameter_counts(head)
    assert counts["action_encoder"] == 349 * 256 + 256 + 2 * 256 + 256 * 128 + 128
    assert counts["head:qol_years"] == (384 + 128) * 256 + 256 + 257
    assert counts["head:feasibility"] == (256 + 128) * 256 + 256 + 257
    assert counts["total"] == counts["action_encoder"] + sum(
        counts[f"head:{n}"] for n in la.COMPONENTS
    )


# --- pairs and loss ------------------------------------------------------------------------


def _root(evals_by_candidate, pair_sd=None) -> la.Root:
    candidates = []
    for i, evals in enumerate(evals_by_candidate):
        evaluations = {
            (h, s): la.Evaluation(h, s, np.asarray(r, dtype=np.float64), None)
            for (h, s), r in evals.items()
        }
        candidates.append(
            la.CandidateRecord(i, "x", i == 0, "", {}, np.zeros(349), None, None, evaluations)
        )
    return la.Root("r", 90100, 0, 1, "h12", 0, (0.5, 0.5), False, pair_sd or {}, candidates)


def _ret(q: float) -> list[float]:
    return [q * la.REFERENCE_SCALES["qol_years"], 0.0, 0.0, 0.0]


def test_pairs_same_horizon_prefer_reevaluation_and_never_mix() -> None:
    root = _root(
        [
            # candidate 0: re-evaluated at h=2 (streams 1-3) and selection at h=1, h=2
            {(1, 0): _ret(0.0), (2, 0): _ret(0.0), (2, 1): _ret(1.0), (2, 2): _ret(1.1),
             (2, 3): _ret(0.9)},
            # candidate 1: re-evaluated at h=2 lower, selection at h=1 higher
            {(1, 0): _ret(0.5), (2, 0): _ret(5.0), (2, 1): _ret(0.0), (2, 2): _ret(0.1),
             (2, 3): _ret(-0.1)},
            # candidate 2: selection only at h=2 (cannot pair with re-evaluated ones by reeval)
            {(2, 0): _ret(2.0)},
        ],
        pair_sd={1: np.array([0.01 * la.REFERENCE_SCALES["qol_years"], 0, 0, 0]),
                 2: np.array([0.01 * la.REFERENCE_SCALES["qol_years"], 0, 0, 0])},
    )
    pairs = la.build_pairs([root], "qol_years")
    rows = {(int(i), int(k), int(h)): (bool(r), float(d)) for i, k, h, r, d in zip(
        pairs.i, pairs.k, pairs.horizon, pairs.reevaluated, pairs.delta, strict=True)}
    # 0 vs 1 at h=2 uses the paired re-evaluation streams (+1.0), not stream 0 (-5.0).
    assert rows[(0, 1, 2)] == (True, pytest.approx(1.0))
    # 0 vs 1 at h=1: selection stream only.
    assert rows[(0, 1, 1)] == (False, pytest.approx(-0.5))
    # 0 vs 2 and 1 vs 2 at h=2: both on the selection stream.
    assert rows[(0, 2, 2)] == (False, pytest.approx(-2.0))
    assert rows[(1, 2, 2)][0] is False
    # Never pooled across horizons: one pair per (i, k, horizon).
    assert len(rows) == len(pairs) == 4
    assert all(pairs.admitted)
    # Horizon-specific targets: only h=1 for this objective.
    only_one = la.build_pairs([root], "qol_years", horizons=(1,))
    assert set(only_one.horizon.tolist()) == {1}
    # Without declared single-rollout noise, selection pairs are not comparable.
    root.pair_sd = {}
    assert len(la.build_pairs([root], "qol_years")) == 1


def test_pairs_within_two_standard_errors_are_excluded() -> None:
    noisy = _root(
        [
            {(1, 1): _ret(1.0), (1, 2): _ret(0.0), (1, 3): _ret(2.0)},
            {(1, 1): _ret(0.9), (1, 2): _ret(0.2), (1, 3): _ret(1.6)},
        ]
    )
    pairs = la.build_pairs([noisy], "qol_years")
    assert len(pairs) == 1 and not pairs.admitted[0]
    assert pairs.se[0] > 0 and abs(pairs.delta[0]) < 2 * pairs.se[0]


def test_pairwise_loss_matches_the_preregistered_formula() -> None:
    a_i = torch.tensor([0.3, -0.2])
    a_k = torch.tensor([0.1, 0.4])
    delta = torch.tensor([0.5, 0.2])
    diff = a_i - a_k
    expected = (
        torch.nn.functional.softplus(-torch.sign(delta) * diff) + 0.1 * (diff - delta) ** 2
    ).mean()
    assert torch.allclose(la.pairwise_loss(a_i, a_k, delta), expected)


# --- payload schema, fit, evaluator --------------------------------------------------------


def test_candidate_schema_is_strict() -> None:
    shard = la.synthetic_candidate_shard(worlds=(90100,), roots_per_world=2, candidates=4)
    parsed = la.parse_candidate_shard(shard)
    assert len(parsed.roots) == 2 and parsed.roots[0].h12 == 0
    assert parsed.roots[0].candidates[0].twin is not None
    bad = copy.deepcopy(shard)
    bad["tensors"]["privileged_core"] = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="privileged keys"):
        la.parse_candidate_shard(bad)
    bad = copy.deepcopy(shard)
    bad["roots"][0]["candidates"][1]["is_h12"] = True
    with pytest.raises(ValueError, match="exactly one H12"):
        la.parse_candidate_shard(bad)
    bad = copy.deepcopy(shard)
    bad["component_order"] = bad["component_order"][::-1]
    with pytest.raises(ValueError, match="component_order"):
        la.parse_candidate_shard(bad)
    # A truncated (not terminated) branch is not comparable at its declared horizon.
    truncated = copy.deepcopy(shard)
    evaluation = truncated["roots"][0]["candidates"][0]["evaluations"][-1]
    evaluation["reached_horizon"] = evaluation["horizon"] - 1
    key = (evaluation["horizon"], evaluation["stream"])
    assert key not in la.parse_candidate_shard(truncated).roots[0].candidates[0].evaluations
    evaluation["terminated"] = True
    assert key in la.parse_candidate_shard(truncated).roots[0].candidates[0].evaluations


def test_fit_refuses_reserved_calibration_development_and_sealed_worlds() -> None:
    task, costs = _plain_critics()
    for world in (90154, 90159, 90180, 2000, 3000):
        shard = la.synthetic_candidate_shard(worlds=(90100, world), roots_per_world=1, candidates=3)
        with pytest.raises(ValueError, match="advantage fit refuses"):
            la.fit_advantage_head(
                task, costs, la.parse_candidate_shard(shard), la.AdvantageFitConfig(epochs=1)
            )


def test_world_split_happens_before_pairs_and_is_deterministic() -> None:
    worlds = list(range(90100, 90110))
    fit, held = la.split_worlds(worlds, 0.2, 0)
    assert len(held) == 2 and not set(fit) & set(held) and sorted(fit + held) == worlds
    assert la.split_worlds(worlds, 0.2, 0) == (fit, held)
    assert la.split_worlds(worlds, 0.2, 1) != (fit, held)


def _smoke_fit():
    task, costs = _plain_critics()
    shard = la.synthetic_candidate_shard(
        worlds=tuple(range(90100, 90110)), roots_per_world=6, candidates=8, seed=1
    )
    candidates = la.parse_candidate_shard(shard)
    config = la.AdvantageFitConfig(epochs=60, batch_roots=8, lr=2e-3, seed=0, hidden=64)
    digest = labor_critic_digest(task, costs)
    head, summary = la.fit_advantage_head(task, costs, candidates, config)
    assert labor_critic_digest(task, costs) == digest  # V untouched
    return task, costs, head, summary, config


def test_synthetic_smoke_fit_and_preregistered_evaluation(tmp_path) -> None:
    task, costs, head, summary, config = _smoke_fit()
    assert summary["fit_pairs"]["qol_years"] > 0
    assert summary["loss_history"][-1] < summary["loss_history"][0]
    assert summary["heldout_ranking"]["qol_years"] >= 0.7
    # Evaluate on the reserved worlds (allowed) with a fresh synthetic shard.
    reserved = la.parse_candidate_shard(
        la.synthetic_candidate_shard(
            worlds=tuple(la.RESERVED_HELDOUT_WORLDS), roots_per_world=4, candidates=8, seed=1
        )
    )
    report = lae.evaluate_advantage(
        task, costs, head, reserved, horizons=config.horizons, fit_worlds=summary["fit_worlds"]
    )
    assert report["reserved_worlds_only"]
    for key in ("ranking:qol_years", "ranking:population_years", "ranking:utility:root"):
        assert report["gates"][key]["value"] >= 0.7, key
    assert report["gates"]["false_improvement"]["value"] <= 0.15
    assert report["gates"]["idle_control"]["pass"]
    assert report["gates"]["delayed_benefit"]["pass"]
    assert report["gates"]["bookkeeping_invariance"]["pass"]
    assert report["bookkeeping"]["data_twin_pairs"] > 0
    assert report["pruning_regret"]["top_k_only"]["roots"] > 0
    assert report["status"] == "PASS"
    gate = la.QPiGate.from_report(report)
    assert not gate.open  # authorization is a separate, explicit user decision
    assert la.QPiGate.from_report(report, authorized_by="user").open
    # A world the head was fitted on is refused as an evaluation world.
    fitted = la.parse_candidate_shard(
        la.synthetic_candidate_shard(worlds=(summary["fit_worlds"][0],), roots_per_world=1)
    )
    with pytest.raises(ValueError, match="overlap the advantage fit worlds"):
        lae.evaluate_advantage(
            task, costs, head, fitted, horizons=config.horizons, fit_worlds=summary["fit_worlds"]
        )
    # Checkpoint round trip, bound to the critic digest.
    payload = la.advantage_checkpoint_payload(
        head, critic_digest=labor_critic_digest(task, costs), config=config, fit=summary
    )
    path = tmp_path / "advantage.pt"
    torch.save(payload, path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    restored, restored_config = la.load_advantage_head(state, task, costs)
    assert restored_config == config
    for name, value in head.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    other_task, other_costs = _plain_critics(seed=99)
    with pytest.raises(ValueError, match="different critic"):
        la.load_advantage_head(state, other_task, other_costs)


def test_untrained_head_fails_the_ranking_gates() -> None:
    task, costs = _plain_critics()
    head = la.LaborAdvantageHead.for_critics(task, costs, embedding=16, hidden=16)
    candidates = la.parse_candidate_shard(
        la.synthetic_candidate_shard(worlds=(90154,), roots_per_world=4, candidates=6)
    )
    report = lae.evaluate_advantage(
        task, costs, head, candidates, horizons=la.DEFAULT_OBJECTIVE_HORIZONS
    )
    # All A = 0: every pair is a tie (0.5), idle cannot be strictly last.
    assert report["gates"]["ranking:qol_years"]["value"] == 0.5
    assert not report["gates"]["idle_control"]["pass"]
    assert report["status"] == "FAIL"


# --- ranker hook ---------------------------------------------------------------------------


def test_ranker_evaluates_top_four_plus_h12_and_a_keyed_full_fraction() -> None:
    task, costs = _plain_critics()
    head = la.LaborAdvantageHead.for_critics(task, costs, embedding=16, hidden=16)
    torch.manual_seed(0)
    for module in head.heads.values():
        torch.nn.init.normal_(module[2].weight, std=1.0)
    tensors = _labor_tensors(batch=1)
    actions = [_action(s) for s in range(10)]
    ranker = la.AdvantageRanker(task, costs, head, full_fraction=0.0)
    plan = ranker.plan(tensors, actions, h12_index=7, root_key=(1, 2))
    assert not plan.full_evaluation
    assert 7 in plan.evaluate and set(plan.order[:4]) <= set(plan.evaluate)
    assert len(plan.evaluate) in (4, 5)
    assert math.isclose(sum(plan.advantage["qol_years"]), 0.0, abs_tol=1e-6)
    assert plan.order[0] == int(np.argmax(plan.utility[0]))
    record = plan.record(plan.evaluate[0], [1.0, 2.0, 0.0, 0.0], horizon=3)
    first = plan.evaluate[0]
    assert record["predicted_advantage"]["qol_years"] == plan.advantage["qol_years"][first]
    assert la.AdvantageRanker(task, costs, head, full_fraction=1.0).plan(
        tensors, actions
    ).evaluate == list(range(10))
    # Keyed and reproducible draw: about 20% of roots get full evaluation.
    default = la.AdvantageRanker(task, costs, head, key=4)
    draws = [default.plan(tensors, actions[:5], root_key=(r,)).full_evaluation for r in range(200)]
    assert draws == [default.plan(tensors, actions[:5], root_key=(r,)).full_evaluation
                     for r in range(200)]
    assert 0.1 < np.mean(draws) < 0.3
    # Multiple personalities: the order keeps each personality's best first.
    multi = ranker.plan(tensors, actions, personalities=((0.9, 0.1), (0.1, 0.9)))
    bests = {int(np.argmax(row)) for row in multi.utility}
    assert bests <= set(multi.order[: len(bests)])
    with pytest.raises(ValueError, match="one root"):
        ranker.plan(_labor_tensors(batch=2), actions)


# --- FiLM still conditions the critic context the head reads -----------------------------


def test_film_personality_reaches_the_advantage_context() -> None:
    task, costs = _plain_critics()
    for film in task.encoder._films():
        torch.nn.init.normal_(film.weight, std=0.2)
    tensors = _labor_tensors(batch=1)
    balanced = la.critic_contexts(task, costs, tensors)["qol_years"]
    tilted = dict(tensors, labor_personality=torch.tensor([[0.9, 0.1]]))
    assert not torch.equal(la.critic_contexts(task, costs, tilted)["qol_years"], balanced)
    # The context is exactly the critic's own encoder computation.
    hidden = task.encoder(tensors)[0]
    assert torch.equal(balanced[:, : hidden.shape[1]], hidden.detach())


# --- CLI -----------------------------------------------------------------------------------


def test_fit_and_eval_command_lines(tmp_path) -> None:
    from test_labor_qol_head import _normalizer

    from anthropocene_rl.autonomous_v7.labor_model import (
        LaborEncoderLayout,
        LaborQoLHead,
        labor_checkpoint_payload,
    )

    layout = LaborEncoderLayout(*_layout(False).state_dict().values())
    torch.manual_seed(0)
    checkpoint = labor_checkpoint_payload(
        LaborActor(layout),
        LaborTaskCritic(layout, qol_head=LaborQoLHead()),
        LaborCostCritics(layout),
        _normalizer(),
        layout,
    )
    torch.save(checkpoint, tmp_path / "critic.pt")
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    torch.save(
        la.synthetic_candidate_shard(worlds=(90100, 90101, 90102), roots_per_world=2, candidates=4),
        fit_dir / "a.pt",
    )
    torch.save(
        la.synthetic_candidate_shard(worlds=(90154,), roots_per_world=2, candidates=4),
        tmp_path / "reserved.pt",
    )
    assert la.main(
        ["fit", "--checkpoint", str(tmp_path / "critic.pt"), "--payloads", str(fit_dir),
         "--out", str(tmp_path / "adv.pt"), "--epochs", "2", "--horizons", "qol_years=1,2"]
    ) == 0
    state = torch.load(tmp_path / "adv.pt", map_location="cpu", weights_only=True)
    assert state["config"]["horizons"]["qol_years"] == [1, 2]
    assert state["critic_trained"] is False
    assert lae.main(
        ["--checkpoint", str(tmp_path / "critic.pt"), "--advantage", str(tmp_path / "adv.pt"),
         "--payloads", str(tmp_path / "reserved.pt"), "--out", str(tmp_path / "report.json")]
    ) == 0
    with pytest.raises(ValueError, match="reserved"):
        la.main(
            ["fit", "--checkpoint", str(tmp_path / "critic.pt"), "--payloads",
             str(tmp_path / "reserved.pt"), "--out", str(tmp_path / "x.pt"), "--epochs", "1"]
        )
