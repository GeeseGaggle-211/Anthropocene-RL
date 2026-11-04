"""Stage 1 of the A(s,a) design: targets, loss, encoding, masks, V guard, gate report."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import torch
from test_labor_model import _labor_tensors
from test_labor_privileged_model import _critics, _layout

from anthropocene_rl.autonomous_v7 import labor_a_stage1 as s1
from anthropocene_rl.autonomous_v7 import labor_advantage as la
from anthropocene_rl.autonomous_v7.labor_bridge import SPECIALTIES
from anthropocene_rl.autonomous_v7.labor_model import LaborActor
from anthropocene_rl.autonomous_v7.labor_value_calibration import labor_critic_digest

G = math.exp(-20.0 / 400.0)
QS = la.REFERENCE_SCALES["qol_years"]


def _plain_critics(seed: int = 3):
    return _critics(_layout(False), seed=seed)


# --- hand-computed targets -----------------------------------------------------------------


def _stream(stream, rewards, statuses, rows):
    """rewards in reference units on qol only; rows: decision -> V (qol, reference units)."""
    return {
        "stream": stream,
        "step_rewards": [[r * QS, 0.0, 0.0, 0.0] for r in rewards],
        "step_years": [20.0] * len(rewards),
        "terminated": "terminated" in statuses.values(),
        "boundaries": [
            {"decision": d, "turn": 10 + d, "status": statuses[d], "row": rows.get(d)}
            for d in sorted(statuses)
        ],
    }


def _hand_shard():
    """One root (population 10). V rows are stored in successor core[:, 0] (qol)."""
    core: list[float] = []

    def row(v):
        core.append(v)
        return len(core) - 1

    live3 = {1: "live", 2: "live", 3: "live"}

    def survivor_streams(rewards_by_stream, v1, v3_by_stream):
        return [
            _stream(s, rewards_by_stream[s], live3, {1: row(v1), 3: row(v3_by_stream[s])})
            for s in sorted(rewards_by_stream)
        ]

    rng = np.random.default_rng(4)
    actions = [la._synthetic_action(rng) for _ in range(6)]
    cands = []

    def cand(i, family, evals, streams):
        cands.append(
            {
                "index": i,
                "family": family,
                "is_h12": i == 0,
                "digest": str(i),
                "action": actions[i],
                "control": None,
                "bookkeeping_twin": None,
                "evaluations": [
                    {
                        "horizon": h,
                        "reached_horizon": reached,
                        "terminated": term,
                        "stream": s,
                        "returns": [0.0, 0.0, 0.0, 0.0],
                        "step_years": [20.0] * h,
                        "se": None,
                    }
                    for (h, s, reached, term) in evals
                ],
                "successor_streams": streams,
            }
        )

    full = [(1, 0, 1, False), (3, 0, 3, False), (3, 1, 3, False), (3, 2, 3, False)]
    ones = {s: [1.0, 1.0, 1.0] for s in (0, 1, 2)}
    cand(0, "h12", full, survivor_streams(ones, 2.0, {0: 4.0, 1: 4.0, 2: 4.0}))
    a_rewards = {s: [2.0, 1.0, 1.0] for s in (0, 1, 2)}
    cand(1, "transfer", full, survivor_streams(a_rewards, 3.0, {0: 5.0, 1: 5.5, 2: 4.5}))
    # B: non-survivor (horizon 1 only, stream 0).
    cand(2, "activity", [(1, 0, 1, False)], [_stream(0, [3.0], {1: "live"}, {1: row(1.0)})])
    # C: survivor that terminates at decision 2 (closes with zero future).
    term = [(1, 0, 1, False), (3, 0, 2, True), (3, 1, 2, True), (3, 2, 2, True)]
    cand(
        3,
        "movement",
        term,
        [
            _stream(s, [1.0, r2], {1: "live", 2: "terminated"}, {1: row(1.0)})
            for s, r2 in ((0, 3.0), (1, 3.0), (2, 2.0))
        ],
    )
    # D: non-survivor, inactive at decision 1 -> unavailable.
    cand(4, "transfer", [(1, 0, 1, False)], [_stream(0, [1.0], {1: "inactive"}, {})])
    # E: survivor on stream 0 only whose decision-3 row is missing -> unavailable.
    cand(
        5,
        "activity",
        [(1, 0, 1, False), (3, 0, 3, False)],
        [_stream(0, [1.0, 1.0, 1.0], live3, {1: row(1.0)})],
    )
    tensors = la.synthetic_root_tensors(1)
    succ = la.synthetic_root_tensors(len(core), seed=5)
    succ["core"][:, 0] = torch.tensor(core, dtype=torch.float32)
    succ["core"][:, 1:4] = 0.0
    return {
        "schema": la.CANDIDATE_PAYLOAD_SCHEMA,
        "producer": {"kind": "hand"},
        "component_order": list(la.COMPONENTS),
        "returns_units": la.RETURNS_UNITS,
        "discount_years": 400.0,
        "specialty_ids": list(SPECIALTIES),
        "tensors": tensors,
        "privileged": None,
        "roots": [
            {
                "root_id": "90100/hand",
                "world": 90100,
                "turn": 3,
                "camp": 1,
                "society": 1,
                "source": "h12",
                "tensor_row": 0,
                "personality": [0.5, 0.5],
                "full_evaluation": True,
                "pair_sd": None,
                "population": 10,
                "successor_decisions": [1, 3],
                "final_horizon": 3,
                "candidates": cands,
            }
        ],
        "successor_tensors": succ,
        "successor_privileged": None,
        "successor_row_count": len(core),
    }


def _exact_values(data):
    """V straight from the stored float64 numbers (float32 core is exact for these)."""
    return s1.synthetic_values(data)


def test_target_arithmetic_on_a_hand_computed_root() -> None:
    data = s1.stage1_data([_hand_shard()])
    targets = s1.build_stage1_targets(data, _exact_values)
    assert targets.candidates_seen == 5
    assert targets.dropped == {
        "non-survivor:one-step-target-unavailable": 1,  # D: inactive at decision 1
        "survivor:folded-target-unavailable": 1,  # E: missing decision-3 row
    }
    by = {int(c): i for i, c in enumerate(targets.candidate)}
    assert sorted(by) == [1, 2, 3]
    t1_h, tk_h = 1 + 2 * G, 1 + G + G * G + 4 * G**3
    # A: survivor, folded 0.05 / 0.95 on re-evaluation streams 1, 2 paired with H12.
    d = [
        0.05 * ((2 + 3 * G) - t1_h) + 0.95 * ((2 + G + G * G + v3 * G**3) - tk_h)
        for v3 in (5.5, 4.5)
    ]
    a = by[1]
    assert targets.survivor[a] and targets.target_class[a] == s1.FOLDED
    assert targets.source[a] == "reeval" and targets.streams[a] == 2
    assert targets.y[a, 0] == pytest.approx(np.mean(d), rel=1e-12)
    # C: terminated at decision 2: the K-step target is r0 + g r1 (zero future).
    dc = [0.05 * ((1 + G) - t1_h) + 0.95 * ((1 + G * r2) - tk_h) for r2 in (3.0, 2.0)]
    c = by[3]
    assert targets.survivor[c] and targets.y[c, 0] == pytest.approx(np.mean(dc), rel=1e-12)
    # Pooled single-stream folded variance: dof-weighted over A and C (one dof each).
    var_a, var_c = np.var(d, ddof=1), np.var(dc, ddof=1)
    pooled = (var_a + var_c) / 2
    assert targets.pooled_variance["folded"]["qol_years"] == pytest.approx(pooled, rel=1e-9)
    assert targets.se[a, 0] == pytest.approx(
        math.sqrt(max(var_a, la.VARIANCE_FLOOR_FRACTION * pooled) / 2), rel=1e-9
    )
    # B: non-survivor: the folding degenerates to the one-step target on both sides
    # (weights 1, 0): y = (3 + g V1) - (1 + g V1_h12) on stream 0.
    b = by[2]
    assert not targets.survivor[b] and targets.target_class[b] == s1.ONE_STEP
    assert targets.source[b] == "stream0"
    assert targets.y[b, 0] == pytest.approx((3 + G * 1.0) - t1_h, rel=1e-12)
    # One-step paired differences are identical across re-evaluation streams here, so
    # the pooled one-step variance is 0 and B's SE is the absolute floor.
    assert targets.pooled_variance["one-step"]["qol_years"] == 0.0
    assert targets.se[b, 0] == la.SE_ABS_FLOOR
    # Other objectives carry zero rewards and zero V: y = 0, SE at the floor.
    assert np.all(targets.y[:, 1:] == 0.0) and np.all(targets.se[:, 1:] == la.SE_ABS_FLOOR)
    summary = targets.summary()
    assert summary["survivors"] == 2 and summary["non_survivors"] == 1


def test_targets_refuse_v1_shards_and_roots_without_population() -> None:
    v1 = la.synthetic_candidate_shard(worlds=(90100,), roots_per_world=1, candidates=3)
    with pytest.raises(ValueError, match="export-v2"):
        s1.stage1_data([v1])
    shard = _hand_shard()
    del shard["roots"][0]["population"]
    with pytest.raises(ValueError, match="population"):
        s1.build_stage1_targets(s1.stage1_data([shard]), _exact_values)


# --- loss ------------------------------------------------------------------------------------


def test_anchored_huber_is_inverse_variance_weighted_with_se_scaled_delta() -> None:
    a_i = torch.tensor([0.5, 0.3, 2.0, -1.0], requires_grad=True)
    a_h = torch.tensor([0.1, 0.1, 0.0, 0.0])
    y = torch.tensor([0.3, 0.0, 0.0, 0.0])
    se = torch.tensor([0.1, 0.2, 0.5, 0.25])
    c = la.HUBER_SE_MULTIPLIER
    loss = s1.anchored_huber_loss(a_i, a_h, y, se)
    r = (a_i - a_h - y).detach()

    def huber(x, delta):
        return 0.5 * x * x if abs(x) <= delta else delta * (abs(x) - 0.5 * delta)

    expected = np.mean([huber(float(r[k]), c * float(se[k])) / float(se[k]) ** 2 for k in range(4)])
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-6)
    # Quadratic regime: gradient = r / se^2 / N (inverse noise variance weight).
    loss.backward()
    assert float(a_i.grad[0]) == pytest.approx(float(r[0]) / 0.1**2 / 4, rel=1e-5)
    # Linear regime beyond 1.345 SE: gradient magnitude c / se / N.
    assert abs(float(a_i.grad[2])) == pytest.approx(c / 0.5 / 4, rel=1e-5)
    assert (
        float(
            s1.anchored_huber_loss(torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0))
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="positive"):
        s1.anchored_huber_loss(a_i, a_h, y, torch.zeros(4))
    # The historical pairwise loss stays callable for comparison.
    assert torch.isfinite(la.pairwise_loss(a_i.detach(), a_h, y))


# --- encoding after apportionment ------------------------------------------------------------


def _fractional_action(population: int):
    """An H12-like action whose specialty shares are not multiples of 1 / population."""
    action = la._synthetic_action(np.random.default_rng(12))
    raw = np.asarray(action["laborSpecialtyAllocation"])
    counts = la.apportion_specialty_counts(raw, population)
    assert not np.allclose(raw, counts / population)  # apportionment changes the block
    return action


def test_h12_action_encodes_identically_on_search_and_actor_sides() -> None:
    population = 23
    shard = s1.synthetic_stage1_shard(worlds=(90100,), roots_per_world=1, population=population)
    root = shard["roots"][0]
    h12 = _fractional_action(population)
    root["candidates"][0]["action"] = h12
    parsed = la.parse_candidate_shard(shard)
    search = parsed.roots[0].candidates[0].features
    assert parsed.roots[0].population == population
    # Actor-side encoding of the concrete action (encode_actions, the ranker's path).
    actor_side = la.encode_actions([h12], population)[0]
    assert np.array_equal(actor_side, search)
    spec = la.block_slices()["specialty_share"]
    expected = (
        la.apportion_specialty_counts(h12["laborSpecialtyAllocation"], population) / population
    ).astype(np.float32)
    assert np.array_equal(search[spec], expected)
    assert not np.array_equal(la.action_features(h12)[spec], search[spec])  # raw differs
    # A bookkeeping-style perturbation that apportions to the same residents encodes the same.
    shifted = copy.deepcopy(h12)
    alloc = np.asarray(shifted["laborSpecialtyAllocation"], dtype=np.float64)
    big = int(np.argmax(alloc))
    small = int(np.argsort(alloc)[-2])
    alloc[big] -= 1e-4
    alloc[small] += 1e-4
    shifted["laborSpecialtyAllocation"] = alloc.tolist()
    assert np.array_equal(la.encode_actions([shifted], population)[0], search)
    # Ranker path (AdvantageRanker.plan) with the population uses the same encoding.
    task, costs = _plain_critics()
    head = la.LaborAdvantageHead.for_critics(task, costs, embedding=16, hidden=16)
    plan = la.AdvantageRanker(task, costs, head, full_fraction=0.0).plan(
        _labor_tensors(batch=1), [h12, shifted], population=population
    )
    assert plan.advantage["qol_years"][0] == plan.advantage["qol_years"][1]
    # Reparameterized path: an actor output concentrated on H12's shares reproduces the
    # search encoding of the specialty block exactly (forward = post-apportionment).
    shares = torch.tensor([h12["laborSpecialtyAllocation"]], dtype=torch.float32)
    torch.manual_seed(0)
    actor = LaborActor(_layout(False), width=16, token_width=16)
    tensors = _labor_tensors(batch=1)
    output = dict(actor(tensors))
    output["specialty_logits"] = torch.log(shares.clamp_min(1e-12))
    output["specialty_log_std"] = torch.full_like(shares, -30.0)
    gate = la.QPiGate("t", tuple(la.RESERVED_HELDOUT_WORLDS), {"a": 0.9}, 0.0, True, True, "test")
    output["specialty_logits"].requires_grad_(True)
    vector = la.reparameterized_action_features(
        output,
        tensors,
        generator=torch.Generator().manual_seed(1),
        gate=gate,
        population=[population],
    )
    assert torch.equal(vector[0, spec], torch.as_tensor(search[spec]))
    # Straight-through: the gradient reaches the continuous shares.
    (vector[0, spec] * torch.arange(14.0)).sum().backward()
    assert output["specialty_logits"].grad is not None
    assert output["specialty_logits"].grad.abs().sum() > 0


def test_payload_reader_checks_compiled_residents() -> None:
    shard = s1.synthetic_stage1_shard(worlds=(90100,), roots_per_world=1, population=23)
    cand = shard["roots"][0]["candidates"][1]
    counts = la.apportion_specialty_counts(cand["action"]["laborSpecialtyAllocation"], 23)
    cand["compiled"] = {"specialty_counts": counts.tolist(), "population": 23}
    la.parse_candidate_shard(shard)
    bad = counts.copy()
    bad[0] += 1
    bad[int(np.argmax(bad[1:])) + 1] -= 1
    cand["compiled"]["specialty_counts"] = bad.tolist()
    with pytest.raises(ValueError, match="compiled residents"):
        la.parse_candidate_shard(shard)


# --- fit, masks, V guard -----------------------------------------------------------------------


def _fit_data(worlds=tuple(range(90100, 90108)), seed=0, roots=4):
    data = s1.stage1_data(
        [s1.synthetic_stage1_shard(worlds=worlds, roots_per_world=roots, seed=seed)]
    )
    return data, s1.build_stage1_targets(data, s1.synthetic_values)


def _config(**kw):
    base = dict(
        epochs=40,
        batch_roots=8,
        lr=2e-3,
        hidden=64,
        seed=0,
        target_source=la.LAMBDA_TARGET_SOURCE,
        support_floor=5,
    )
    base.update(kw)
    return la.AdvantageFitConfig(**base)


def test_fit_masks_trade_blocks_and_holds_unsupported_blocks_at_zero() -> None:
    task, costs = _plain_critics()
    data, targets = _fit_data()
    support = s1.block_support(data.candidates.roots, targets)
    # Synthetic search perturbs every block (random actions), including trade value/share.
    assert support["trade_value_log"] > 0 and support["specialty_share"] > 0
    floor = support["fission_onehot"] + 1
    config = _config(epochs=3, support_floor=floor)
    head, summary = la.fit_advantage_head(
        task, costs, data.candidates, config, targets=targets, stability=None
    )
    masked = set(summary["masked_blocks"])
    assert {"trade_value_log", "trade_share", "fission_onehot"} <= masked
    assert "fission_onehot" in summary["masked_for_low_support"]
    assert set(head.masked_blocks) == masked
    tensors, priv = data.candidates.root_tensors()
    contexts = la.critic_contexts(task, costs, tensors, priv)
    response = s1.block_response(head, contexts, data.candidates.roots)
    for block in masked:
        assert all(v == 0.0 for v in response[block].values()), block
    assert any(v > 0 for v in response["specialty_share"].values())
    # Changing a masked block of any action leaves every A exactly unchanged.
    feats = torch.as_tensor(np.stack([c.features for c in data.candidates.roots[0].candidates]))
    index = torch.zeros(len(feats), dtype=torch.long)
    first = {k: v for k, v in contexts.items()}
    moved = feats.clone()
    sl = la.block_slices()["trade_share"]
    moved[:, sl] = 0.9
    with torch.no_grad():
        a, b = head(first, feats, index), head(first, moved, index)
    assert all(torch.equal(a[n], b[n]) for n in la.COMPONENTS)
    # Masks survive the checkpoint round trip.
    payload = la.advantage_checkpoint_payload(
        head, critic_digest=labor_critic_digest(task, costs), config=config, fit=summary
    )
    restored, restored_config = la.load_advantage_head(
        json.loads(json.dumps({k: v for k, v in payload.items() if k != "state_dict"}))
        | {"state_dict": payload["state_dict"]},
        task,
        costs,
    )
    assert restored.masked_blocks == head.masked_blocks
    assert restored_config == config


def test_v_parameters_and_outputs_are_bitwise_identical_after_an_a_fit() -> None:
    task, costs = _plain_critics()
    data, targets = _fit_data()
    tensors, priv = data.candidates.root_tensors()
    params = {k: v.clone() for k, v in [*task.state_dict().items(), *costs.state_dict().items()]}
    before = la.state_values(task, costs, tensors, priv)
    successor_before = s1.critic_values(task, costs)(data.successor_dataset)
    head, summary = la.fit_advantage_head(
        task, costs, data.candidates, _config(epochs=5), targets=targets, stability=None
    )
    after = la.state_values(task, costs, tensors, priv)
    for name in la.COMPONENTS:
        assert torch.equal(before[name], after[name])
    successor_after = s1.critic_values(task, costs)(data.successor_dataset)
    for name in la.COMPONENTS:
        assert np.array_equal(successor_before[name], successor_after[name])
    now = {k: v for k, v in [*task.state_dict().items(), *costs.state_dict().items()]}
    assert all(torch.equal(params[k], now[k]) for k in params)
    assert all(p.grad is None for p in [*task.parameters(), *costs.parameters()])
    assert summary["v_guard"]["identical"] and summary["v_guard"]["outputs_identical"]
    assert any(p.abs().sum() > 0 for p in head.heads["qol_years"][2].parameters())


def test_anchor_is_exact_and_the_fit_learns_the_planted_targets() -> None:
    task, costs = _plain_critics()
    data, targets = _fit_data()
    head, summary = la.fit_advantage_head(task, costs, data.candidates, _config(), targets=targets)
    assert summary["loss_history"][-1] < summary["loss_history"][0]
    assert summary["fit_rows_by_survival"]["non_survivors"] > 0  # never survivors only
    assert summary["stability"]["admitted"] in (True, False)
    tensors, priv = data.candidates.root_tensors()
    contexts = la.critic_contexts(task, costs, tensors, priv)
    pred = s1.anchored_predictions(head, contexts, data.candidates.roots)
    for name in la.COMPONENTS:
        for r, root in enumerate(data.candidates.roots):
            assert pred[name][r][root.h12] == 0.0
            assert np.any(pred[name][r] != 0.0)
    assert summary["heldout_sign_accuracy"]["qol_years"] > 0.65
    with pytest.raises(ValueError, match="stage-1 targets"):
        la.fit_advantage_head(task, costs, data.candidates, _config(epochs=1))
    with pytest.raises(ValueError, match="target_source"):
        la.fit_advantage_head(
            task, costs, data.candidates, la.AdvantageFitConfig(epochs=1), targets=targets
        )
    # The pair-source config keeps its historical state dict exactly.
    assert "target_source" not in la.AdvantageFitConfig().state_dict()


# --- gate report ---------------------------------------------------------------------------------


def test_gate_report_schema_with_two_seeds_and_ppo_dry_run() -> None:
    task, costs = _plain_critics()
    data, targets = _fit_data()
    heads, summaries = [], []
    for seed in (0, 1):
        head, summary = la.fit_advantage_head(
            task, costs, data.candidates, _config(epochs=40, seed=seed), targets=targets
        )
        heads.append(head)
        summaries.append(summary)
    reserved, reserved_targets = _fit_data(
        worlds=tuple(la.RESERVED_HELDOUT_WORLDS), seed=7, roots=3
    )
    torch.manual_seed(5)
    actor = LaborActor(_layout(False), width=16, token_width=16)
    actor_before = {k: v.clone() for k, v in actor.state_dict().items()}
    report = s1.stage1_gate_report(
        heads,
        task,
        costs,
        reserved.candidates,
        reserved_targets,
        summaries,
        actor=actor,
        ppo_epochs=3,
    )
    json.dumps(report)  # plain JSON
    assert report["schema"] == s1.GATE_SCHEMA and report["reserved_worlds_only"]
    assert report["thresholds"] == {
        "calibration_slope_range": [0.8, 1.2],
        "calibration_abs_bias_over_spread": 0.25,
        "ordering_sign_accuracy": 0.65,
        "no_single_path_sign_accuracy_floor": 0.5,
        "seed_rank_correlation": 0.8,
        "overfit_gap": 0.10,
        "overfit_heldout_drop": 0.03,
    }
    gates = report["gates"]
    expected = {
        f"{g}:{n}" for g in ("calibration", "ordering", "no_single_path") for n in la.COMPONENTS
    }
    expected |= {
        "block_support",
        "v_non_regression",
        "seed_agreement",
        "stability",
        "ppo_readiness",
    }
    assert set(gates) == expected
    allowed = {"PASS", "FAIL", "NO_SIGNAL", "NOT_RUN", "SKIPPED", "REPORT_ONLY"}
    assert all(g["status"] in allowed for g in gates.values())
    assert gates["v_non_regression"]["status"] == "PASS"
    assert gates["block_support"]["masked_response_exactly_zero"]
    assert gates["seed_agreement"]["seeds"] == 2
    cal = gates["calibration:qol_years"]
    assert {"slope", "mae", "bias", "realised_spread", "abs_bias_over_spread"} <= set(cal)
    # Survival split in every evaluation.
    for name in la.COMPONENTS:
        assert set(report["calibration"][name]) == set(s1.SPLITS)
        assert set(report["ordering"][name]) == set(s1.SPLITS)
        assert {"no_controls", "with_controls", "top1_no_controls"} == set(
            report["ordering"][name]["all"]
        )
    worst = gates["no_single_path:qol_years"]
    assert "worst_world" in worst and "worst_family" in worst
    ppo = gates["ppo_readiness"]
    assert ppo["status"] == "REPORT_ONLY" and len(ppo["per_epoch"]) == 3
    assert ppo["per_epoch"][0]["approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert ppo["actor_untouched"]
    assert all(torch.equal(actor_before[k], v) for k, v in actor.state_dict().items())
    # End to end on the planted structure (same structure seed, unseen worlds/actions):
    # every gate with signal passes.
    assert report["status"] == "PASS", report["failed"]
    assert gates["ordering:qol_years"]["value"] >= s1.ORDERING_SIGN_ACCURACY
    assert gates["calibration:qol_years"]["status"] == "PASS"
    no_actor = s1.stage1_gate_report(
        heads[:1], task, costs, reserved.candidates, reserved_targets, summaries[:1]
    )
    assert no_actor["gates"]["ppo_readiness"]["status"] == "SKIPPED"
    assert no_actor["gates"]["seed_agreement"]["status"] == "NOT_RUN"
    assert "seed_agreement" in no_actor["failed"]
    # The fit worlds are never gate worlds.
    with pytest.raises(ValueError, match="overlap"):
        s1.stage1_gate_report(heads, task, costs, data.candidates, targets, summaries)


def test_command_line_fits_seeds_and_writes_the_gate_report(tmp_path) -> None:
    from test_labor_qol_head import _normalizer

    from anthropocene_rl.autonomous_v7.labor_model import (
        LaborCostCritics,
        LaborEncoderLayout,
        LaborQoLHead,
        LaborTaskCritic,
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
    fit_dir, eval_dir = tmp_path / "fit", tmp_path / "eval"
    fit_dir.mkdir()
    eval_dir.mkdir()
    torch.save(
        s1.synthetic_stage1_shard(worlds=(90100, 90101, 90102), roots_per_world=2), fit_dir / "a.pt"
    )
    torch.save(
        s1.synthetic_stage1_shard(worlds=(90154,), roots_per_world=2, seed=3), eval_dir / "r.pt"
    )
    code = s1.main(
        [
            "fit",
            "--checkpoint",
            str(tmp_path / "critic.pt"),
            "--payloads",
            str(fit_dir),
            "--eval-payloads",
            str(eval_dir),
            "--out",
            str(tmp_path / "a1.pt"),
            "--report",
            str(tmp_path / "gates.json"),
            "--epochs",
            "2",
            "--seeds",
            "0,1",
            "--support-floor",
            "1",
            "--actor-checkpoint",
            str(tmp_path / "critic.pt"),
            "--ppo-epochs",
            "2",
        ]
    )
    assert code in (0, 1)
    report = json.loads((tmp_path / "gates.json").read_text())
    assert report["schema"] == s1.GATE_SCHEMA and report["worlds"] == [90154]
    assert report["gates"]["ppo_readiness"]["status"] == "REPORT_ONLY"
    assert (tmp_path / "a1.pt").exists() and (tmp_path / "a1.seed1.pt").exists()
    state = torch.load(tmp_path / "a1.pt", map_location="cpu", weights_only=True)
    assert state["config"]["target_source"] == la.LAMBDA_TARGET_SOURCE
    assert set(la.STAGE1_MASKED_BLOCKS) <= set(state["masked_blocks"])
    with pytest.raises(ValueError, match="reserved"):
        s1.main(
            [
                "fit",
                "--checkpoint",
                str(tmp_path / "critic.pt"),
                "--payloads",
                str(eval_dir),
                "--out",
                str(tmp_path / "x.pt"),
                "--report",
                str(tmp_path / "x.json"),
                "--epochs",
                "1",
                "--seeds",
                "0",
            ]
        )


# --- review round 2: unstaffed roles, signal rules, provisional constants -----------------


def _h12_with_unstaffed_role(population: int):
    rng = np.random.default_rng(21)
    counts = np.zeros(len(SPECIALTIES), dtype=np.int64)
    counts[0], counts[1], counts[3] = 5, 12, population - 17  # role rows 1 and 3 staffed
    action = la._synthetic_action(rng)
    action["laborSpecialtyAllocation"] = (counts / population).tolist()
    unstaffed = SPECIALTIES.index("Farmer") - 1  # role row of Farmer (zero residents)
    assert counts[unstaffed + 1] == 0
    return action, unstaffed


def test_control_twin_of_h12_encodes_bitwise_equal_and_has_zero_advantage() -> None:
    population = 23
    h12, role = _h12_with_unstaffed_role(population)
    twin = copy.deepcopy(h12)
    activity = np.asarray(twin["laborActivityAllocation"], dtype=np.float64)
    activity[role] = np.roll(activity[role], 3)  # unstaffed role: bookkeeping only
    twin["laborActivityAllocation"] = activity.tolist()
    shares = np.asarray(twin["laborSpecialtyAllocation"])
    shares[1] -= 0.25 / population  # a quarter resident: same apportioned residents
    shares[3] += 0.25 / population
    twin["laborSpecialtyAllocation"] = shares.tolist()
    assert not np.array_equal(la.action_features(h12), la.action_features(twin))  # raw differs
    shard = s1.synthetic_stage1_shard(worlds=(90100,), roots_per_world=1, population=population)
    root = shard["roots"][0]
    root["candidates"][0]["action"] = h12
    root["candidates"][1]["action"] = twin
    root["candidates"][1]["family"] = "control-twin"
    parsed = la.parse_candidate_shard(shard).roots[0]
    f_h12, f_twin = parsed.candidates[0].features, parsed.candidates[1].features
    assert np.array_equal(f_h12, f_twin)
    sl = la.block_slices()["activity_share"]
    rows = f_h12[sl].reshape(-1, len(la.ACTIVITIES))
    assert rows[role].tolist() == [1.0] + [0.0] * (len(la.ACTIVITIES) - 1)
    assert np.array_equal(la.encode_actions([h12, twin], population), np.stack([f_h12, f_twin]))
    task, costs = _plain_critics()
    head = la.LaborAdvantageHead.for_critics(task, costs, embedding=16, hidden=16)
    torch.manual_seed(2)
    for module in head.heads.values():
        torch.nn.init.normal_(module[2].weight, std=1.0)
    tensors, priv = la.parse_candidate_shard(shard).root_tensors()
    contexts = la.critic_contexts(task, costs, tensors, priv)
    pred = s1.anchored_predictions(head, contexts, [parsed])
    for name in la.COMPONENTS:
        assert pred[name][0][1] == 0.0
        assert np.any(pred[name][0] != 0.0)
    plan = la.AdvantageRanker(task, costs, head, full_fraction=0.0).plan(
        _labor_tensors(batch=1), [h12, twin], population=population
    )
    assert all(plan.advantage[n][0] == plan.advantage[n][1] for n in la.COMPONENTS)
    # Reparameterized forward: H12-concentrated output gives the same specialty and
    # activity blocks, the unstaffed row exactly neutral.
    torch.manual_seed(0)
    actor = LaborActor(_layout(False), width=16, token_width=16)
    tensors1 = _labor_tensors(batch=1)
    output = dict(actor(tensors1))
    spec_t = torch.tensor([h12["laborSpecialtyAllocation"]], dtype=torch.float32)
    output["specialty_logits"] = torch.log(spec_t.clamp_min(1e-12))
    output["specialty_log_std"] = torch.full_like(spec_t, -30.0)
    vector = la.reparameterized_action_features(
        output, tensors1, generator=torch.Generator().manual_seed(1), population=[population]
    )
    spec = la.block_slices()["specialty_share"]
    assert torch.equal(vector[0, spec], torch.as_tensor(f_h12[spec]))
    act = vector[0, sl].reshape(-1, len(la.ACTIVITIES))
    assert torch.equal(act[role], torch.as_tensor(rows[role], dtype=torch.float32))


def test_task_objectives_without_signal_fail_and_cost_heads_may_have_none() -> None:
    task, costs = _plain_critics()
    data, targets = _fit_data()
    head, summary = la.fit_advantage_head(
        task, costs, data.candidates, _config(epochs=3), targets=targets, stability=None
    )
    reserved, rt = _fit_data(worlds=tuple(la.RESERVED_HELDOUT_WORLDS), seed=7, roots=3)
    for j in (la.COMPONENTS.index("qol_years"), la.COMPONENTS.index("feasibility")):
        rt.y[:, j] = 0.0
    report = s1.stage1_gate_report([head], task, costs, reserved.candidates, rt, [summary])
    gates = report["gates"]
    for g in ("calibration", "ordering", "no_single_path"):
        assert gates[f"{g}:qol_years"]["status"] == "FAIL"
        assert gates[f"{g}:qol_years"]["no_signal"] is True
        assert f"{g}:qol_years" in report["failed"]
        assert gates[f"{g}:feasibility"]["status"] == "NO_SIGNAL"
        assert f"{g}:feasibility" in report["no_signal"]
        assert f"{g}:feasibility" not in report["failed"]
    assert report["status"] == "FAIL"
    assert report["provisional_constants"] == {
        "block_support_floor_rows": 30,
        "variance_floor_fraction_of_pooled": 0.25,
        "se_abs_floor_head_units": 1e-3,
        "huber_se_multiplier": 1.345,
        "status": "provisional",
    }
