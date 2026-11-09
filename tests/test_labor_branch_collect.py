"""WP-B: paired actor branches recorded during PPO collection (labor_branch_collect).

The real rollout path and the real branch collector run over ``labor_wpb_fake`` (a fake
multi-camp labor bridge with exact fork/restore; not a model of the game). Covers: default
off is bitwise the old collect and branching never perturbs the rollout; common random
numbers (re-running reproduces exactly; identical actions give identical branches; the
continuation key ignores the branch); importance weights and the D6 selection; the folded
lambda target against a hand computation; the selection-rate control; the B4 actor
continuation as default; the report.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
import torch
from labor_wpb_fake import FakeLaborBridge, fake_shadow_factory, install
from test_labor_ppo_perhead import _models, _village_at

from anthropocene_rl.autonomous_v7 import labor_a_stage2 as s2
from anthropocene_rl.autonomous_v7 import labor_branch_collect as bc
from anthropocene_rl.autonomous_v7 import labor_branch_report as report
from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7.bridge import observation_batch
from anthropocene_rl.autonomous_v7.labor_model import Normalizer

WORLD = 50001  # fixture label; no sealed (3000-3059) or reserved (90154-90159) world


@pytest.fixture(scope="module")
def models():
    actor, task, costs = _models()
    normalizer = Normalizer.fit_bounded(observation_batch([_village_at(7)]))
    return actor, task, costs, normalizer


def _collector(models, config, shadow=None):
    actor, task, costs, normalizer = models
    return bc.PairedBranchCollector(
        config, actor=actor, task_critic=task, cost_critics=costs, normalizer=normalizer,
        reset_options=training.LEGACY_COLLECT_RESET,
        action_binding=training._producer_action_binding()[0],
        shadow_factory=shadow,
    )


def _collect(models, brancher=None, *, turns=2, bridge=FakeLaborBridge, seed=3):
    actor, task, costs, normalizer = models
    extra = {} if brancher is None else {"brancher": brancher}
    return training.collect_labor_rollout(
        actor=actor, task_critic=task, cost_critics=costs, normalizer=normalizer,
        world_seeds=[WORLD], turns=turns, generator=torch.Generator().manual_seed(seed),
        bridge_factory=bridge, action_binding=training._producer_action_binding()[0], **extra,
    )


def _row_key(row):
    return (row.world_seed, row.camp_id, row.turn, row.behavior_initial_labor_logp,
            row.behavior_initial_nonlabor_logp, row.feasibility_initial, row.feasibility_retry,
            tuple(sorted(row.physical_task_rewards.items())))


# ---------------------------------------------------------------------------
# 1. default off: the old collect, bitwise; branching does not perturb the rollout
# ---------------------------------------------------------------------------


def test_default_off_is_bitwise_and_branching_leaves_the_rollout_unchanged(
    monkeypatch, tmp_path
):
    from test_labor_a_ppo_wiring import _default_models, _trainer_state

    install(monkeypatch)
    actor, task, costs, normalizer, layout = _default_models()
    checkpoint = tmp_path / "c.pt"
    torch.save(_trainer_state(actor, task, costs, normalizer, layout), checkpoint)
    original = training.collect_labor_rollout
    seen = []

    def fake_collect(**kw):
        seen.append(sorted(kw))
        return original(**{**kw, "bridge_factory": FakeLaborBridge})

    monkeypatch.setattr(training, "collect_labor_rollout", fake_collect)
    common = ["collect", "--checkpoint", str(checkpoint), "--worlds", str(WORLD),
              "--turns", "2"]
    # Same basename in each directory: torch.save records it inside the zip archive.
    plain_out, off_out, on_out = (tmp_path / d / "rollout.pt" for d in ("plain", "off", "on"))
    training.main([*common, "--out", str(plain_out)])
    training.main([*common, "--branch-select", "off", "--out", str(off_out)])
    training.main([*common, "--branch-select", "uniform", "--branch-rate", "0.5",
                   "--branch-horizon", "2", "--out", str(on_out)])
    plain = plain_out.read_bytes()
    assert off_out.read_bytes() == plain
    assert on_out.read_bytes() == plain  # branching never touched the rollout
    # The off calls pass exactly the historical keywords (no brancher keyword at all).
    assert "brancher" not in seen[0] and seen[0] == seen[1]
    assert "brancher" in seen[2]
    assert not (tmp_path / "plain" / "rollout.branches.pt").exists()
    assert not (tmp_path / "off" / "rollout.branches.pt").exists()
    path = tmp_path / "on" / "rollout.branches.pt"
    payload = report.load_branches(path)
    assert payload["kind"] == bc.BRANCH_FILE_KIND and payload["schemaVersion"] == 1
    prov = payload["provenance"]
    assert prov["rolloutSha256"] == hashlib.sha256(plain).hexdigest()
    rollout = torch.load(on_out, weights_only=True)
    meta = rollout["metadata"]
    assert prov["behaviorPolicyDigest"] == meta["behaviorPolicyDigest"]
    assert prov["bootstrapCriticDigest"] == meta["advantageCriticDigest"]
    assert payload["config"]["continuation"] == "actor" and payload["config"]["horizon"] == 2
    assert payload["records"], "rate 0.5 over the fixture's states should branch"
    for record in payload["records"]:
        assert record["selection"]["weight"] == pytest.approx(2.0)
        assert record["aFirstMatchesRollout"]
        a_features, b_features = (record["actions"][x]["features"] for x in ("a", "b"))
        assert a_features.shape == b_features.shape
    out = tmp_path / "report.json"
    assert report.main(["--branches", str(path), "--rollout", str(on_out),
                        "--out", str(out)]) == 0
    summary = json.loads(out.read_text())
    assert summary["rolloutSha256Matches"] is True
    assert summary["branchedStates"] == len(payload["records"])
    assert summary["cost"]["nativeSteps"]["rollout"] == payload["telemetry"]["decisions"]


# ---------------------------------------------------------------------------
# 2. common random numbers
# ---------------------------------------------------------------------------


def _strip(records):
    out = []
    for r in records:
        out.append({
            "key": (r["world"], r["turn"], r["camp"]),
            "b": json.dumps(r["actions"]["b"]["payload"], sort_keys=True, default=float),
            "targets": r["targets"],
            "digests": [[e["digest"] for e in b["decisions"]]
                        for name in ("a", "b") for b in r["branches"][name]],
            "rewards": [[e["rewards"] for e in b["decisions"]]
                        for name in ("a", "b") for b in r["branches"][name]],
        })
    return out


def test_rerunning_reproduces_every_pair_exactly(monkeypatch, models):
    install(monkeypatch)
    config = bc.BranchConfig(select="uniform", rate=0.5, horizon=2, record_digests=True)
    first, second = _collector(models, config), _collector(models, config)
    rows1, _ = _collect(models, first)
    rows2, _ = _collect(models, second)
    assert [_row_key(r) for r in rows1] == [_row_key(r) for r in rows2]
    assert first.records and _strip(first.records) == _strip(second.records)
    # Branching is invisible to the rollout itself.
    rows0, _ = _collect(models)
    assert [_row_key(r) for r in rows0] == [_row_key(r) for r in rows1]


class _SameAction(bc.PairedBranchCollector):
    """b := a: with common random numbers both branches must be identical."""

    def _second_sample(self, plan, focal, output_row, tensors_row, a):
        return dict(a)


def test_identical_actions_give_identical_branches(monkeypatch, models):
    install(monkeypatch)
    config = bc.BranchConfig(select="uniform", rate=0.5, horizon=3, streams=2,
                             record_digests=True)
    actor, task, costs, normalizer = models
    collector = _SameAction(
        config, actor=actor, task_critic=task, cost_critics=costs, normalizer=normalizer,
        reset_options=training.LEGACY_COLLECT_RESET,
        action_binding=training._producer_action_binding()[0],
    )
    _collect(models, collector)
    # A focal camp whose initial action the rollout saw rejected retries with the rollout's
    # own retry in a but a keyed actor retry in b (both on-policy; not common numbers).
    shared = [r for r in collector.records if not r["actions"]["a"]["retried"]]
    assert shared
    for record in shared:
        assert record["identicalActions"]
        for sa_, sb in zip(record["branches"]["a"], record["branches"]["b"], strict=True):
            for field in ("digest", "rewards", "value"):
                assert [e[field] for e in sa_["decisions"]] == [e[field] for e in sb["decisions"]]
        assert record["targets"]["folded"]["diff"] == [0.0, 0.0, 0.0, 0.0]
        # Streams differ only after the forked decision (their continuation keys differ).
        s0, s1 = record["branches"]["a"]
        assert s0["decisions"][0]["digest"] == s1["decisions"][0]["digest"]


def test_keyed_generator_is_branch_independent():
    def draws(*key):
        return torch.rand(4, generator=bc.keyed_generator(bc.CONTINUATION_TAG, *key))

    assert torch.equal(draws(0, WORLD, 4, 7, 0, 1, 8), draws(0, WORLD, 4, 7, 0, 1, 8))
    assert not torch.equal(draws(0, WORLD, 4, 7, 0, 1, 8), draws(0, WORLD, 4, 7, 0, 1, 9))
    assert not torch.equal(draws(0, WORLD, 4, 7, 0, 1, 8), draws(0, WORLD, 4, 7, 1, 1, 8))
    # Order independence: drawing other keys first changes nothing.
    draws(1, 2, 3)
    assert torch.equal(draws(0, WORLD, 4, 7, 0, 1, 8), draws(0, WORLD, 4, 7, 0, 1, 8))


# ---------------------------------------------------------------------------
# 3. importance weights and D6 selection
# ---------------------------------------------------------------------------


def test_adaptive_weights_are_inverse_selection_probabilities(monkeypatch, models):
    install(monkeypatch)
    actor, _, _, normalizer = models
    config = bc.BranchConfig(select="adaptive", rate=0.25, floor=0.0625, horizon=2,
                             calibration_min_states=3)
    collector = _collector(models, config, shadow=fake_shadow_factory(actor, normalizer))
    _collect(models, collector, turns=3)
    states = collector.states
    assert len(states) == collector.telemetry.camp_states == 9
    # Before calibration (first decision: 3 states) the rate is uniform 0.25.
    assert all(s["probability"] == 0.25 and not s["calibrated"] for s in states[:3])
    assert all(s["calibrated"] for s in states[3:])
    for s in states[3:]:
        assert s["probability"] == (1.0 if s["far"] else 0.0625)
        assert bool(s["triggers"]) == s["far"]
    # The draw is the keyed Bernoulli of labor_a_stage2 with these probabilities.
    chosen, weights = s2.select_branches(
        [s["probability"] for s in states], [(s["world"], s["turn"], s["camp"]) for s in states],
        0,
    )
    assert [s["selected"] for s in states] == chosen.tolist()
    assert [s["weight"] for s in states] == pytest.approx(weights.tolist())
    for record in collector.records:
        sel = record["selection"]
        assert sel["weight"] == pytest.approx(1.0 / sel["probability"])
    # One shadow resolve per rollout decision (all camps at once), none in branches (actor).
    assert collector.telemetry.shadow_steps == collector.telemetry.decisions == 3


def test_selection_rate_control():
    rng = np.random.default_rng(0)

    def distances(n):
        return [
            {h: float(x) for h, x in zip(bc.HEADS, rng.exponential(size=len(bc.HEADS)),
                                          strict=True)}
            for _ in range(n)
        ]

    selector = bc.BranchSelector(bc.BranchConfig(select="adaptive", rate=0.25, floor=1 / 16))
    selector.observe(distances(2000))
    new = distances(4000)
    selection = selector.probabilities(new)
    assert selection.calibrated
    keys = [(WORLD, 4, i) for i in range(len(new))]
    chosen, _ = s2.select_branches(selection.probability, keys, 0)
    assert abs(selection.probability.mean() - 0.25) < 0.02
    assert abs(chosen.mean() - 0.25) < 0.03
    uniform = bc.BranchSelector(bc.BranchConfig(select="uniform", rate=0.1))
    p = uniform.probabilities([{}] * 4000).probability
    assert (p == 0.1).all()
    chosen, w = s2.select_branches(p, keys, 1)
    assert abs(chosen.mean() - 0.1) < 0.02 and (w == 10.0).all()
    fixed = bc.BranchSelector(bc.BranchConfig(
        select="adaptive", rate=0.25, floor=0.05, thresholds={h: 1.0 for h in bc.HEADS}
    ))
    near = {h: 0.5 for h in bc.HEADS}
    far = dict(near, movement=2.0)
    assert fixed.probabilities([near, far]).probability.tolist() == [0.05, 1.0]


# ---------------------------------------------------------------------------
# 4. folded lambda target vs a hand computation
# ---------------------------------------------------------------------------


def _branch(stream, rewards, years, values, statuses):
    return {
        "stream": stream,
        "terminated": "terminated" in statuses,
        "decisions": [
            {"decision": j, "rewards": list(r), "years": y, "value": v, "status": st}
            for j, (r, y, v, st) in enumerate(zip(rewards, years, values, statuses, strict=True))
        ],
    }


def test_folded_lambda_target_matches_a_hand_computation():
    lam, dy = 0.95, 400.0
    ra = [[1.0, 10.0, 0.1, 0.0], [2.0, 20.0, 0.0, 0.0], [3.0, 30.0, 0.5, 1.0]]
    rb = [[0.5, 12.0, 0.0, 0.0], [1.5, 18.0, 0.2, 0.0], [2.5, 33.0, 0.0, 0.0]]
    va = [[100.0, 1000.0, 1.0, 0.1], [110.0, 1100.0, 1.1, 0.2], [120.0, 1200.0, 1.2, 0.3]]
    vb = [[90.0, 900.0, 0.9, 0.0], [95.0, 950.0, 0.8, 0.1], [99.0, 990.0, 0.7, 0.2]]
    years = [20.0, 20.0, 20.0]
    live = ["live"] * 3
    branches = {"a": [_branch(0, ra, years, va, live)], "b": [_branch(0, rb, years, vb, live)]}
    targets = bc.branch_targets(branches, 3, lam=lam, discount_years=dy)
    g = math.exp(-20.0 / 400.0)
    assert g == pytest.approx(0.951229, abs=1e-6)

    def by_hand(r, v):
        r, v = np.asarray(r), np.asarray(v)
        t1 = r[0] + g * v[0]
        t3 = r[0] + g * r[1] + g * g * r[2] + g**3 * v[2]
        folded = 0.05 * t1 + 0.95 * t3
        g2 = r[1] + g * ((1 - lam) * v[1] + lam * (r[2] + g * v[2]))
        exact = r[0] + g * ((1 - lam) * v[0] + lam * g2)
        return folded, exact

    fa, ea = by_hand(ra, va)
    fb, eb = by_hand(rb, vb)
    assert targets["primary"] == "folded"
    assert targets["folded"]["a"] == pytest.approx(fa.tolist())
    assert targets["folded"]["diff"] == pytest.approx((fa - fb).tolist())
    assert targets["exact"]["diff"] == pytest.approx((ea - eb).tolist())
    # k = 2: the fold and the exact recursion coincide.
    two = bc.branch_targets(branches, 2, lam=lam, discount_years=dy)
    assert two["folded"]["diff"] == pytest.approx(two["exact"]["diff"])
    # A focal camp terminated at decision 1 closes with zero future on that side.
    dead = {"a": [_branch(0, ra[:1], years[:1], [None], ["terminated"])],
            "b": [_branch(0, rb, years, vb, live)]}
    t = bc.branch_targets(dead, 3, lam=lam, discount_years=dy)
    assert t["folded"]["a"] == pytest.approx(ra[0])
    # An inactive focal camp makes the pair's target unavailable.
    gone = {"a": [_branch(0, ra[:1], years[:1], [None], ["inactive"])],
            "b": [_branch(0, rb, years, vb, live)]}
    assert bc.branch_targets(gone, 3)["folded"] is None


# ---------------------------------------------------------------------------
# 5. B4: the actor continues by default; H12 continuation is available
# ---------------------------------------------------------------------------


def _refusing_shadow(bridge, world, options):
    raise AssertionError("uniform actor branching must not start an H12 shadow")


def test_actor_continuation_is_the_default_and_needs_no_shadow(monkeypatch, models):
    install(monkeypatch)
    assert bc.BranchConfig().continuation == "actor"
    args = training.argument_parser().parse_args(
        ["collect", "--checkpoint", "c.pt", "--worlds", "1", "--out", "r.pt",
         "--branch-select", "uniform"]
    )
    config = training._branch_config(args)
    assert config.continuation == "actor" and not config.uses_shadow
    collector = _collector(models, bc.BranchConfig(select="uniform", rate=0.5, horizon=3),
                           shadow=_refusing_shadow)
    _collect(models, collector)
    assert collector.records and collector.telemetry.shadow_steps == 0
    assert all(r["continuation"] == "actor" for r in collector.records)


def test_h12_continuation_uses_the_shadow_after_the_fork(monkeypatch, models):
    install(monkeypatch)
    actor, _, _, normalizer = models
    collector = _collector(
        models, bc.BranchConfig(select="uniform", rate=0.5, horizon=3, continuation="h12"),
        shadow=fake_shadow_factory(actor, normalizer),
    )
    _collect(models, collector)
    assert collector.records
    t = collector.telemetry
    # One shadow per rollout decision plus one per branch decision after the forked one.
    assert t.shadow_steps == t.decisions + t.branch_steps - t.branch_runs
    assert t.branch_steps > t.branch_runs
    assert all(r["continuation"] == "h12" for r in collector.records)


def test_branch_a_is_shared_by_the_focal_camps_of_a_decision(monkeypatch, models):
    install(monkeypatch)
    collector = _collector(models, bc.BranchConfig(select="uniform", rate=1.0, horizon=2))
    _collect(models, collector, turns=1)
    t = collector.telemetry
    assert len(collector.records) == 3  # every camp of the one decision
    assert t.branch_runs == 1 + 3  # one shared a + one b per focal camp
    a_runs = [r["branches"]["a"][0] for r in collector.records]
    assert len({json.dumps([e["digest"] for e in a["decisions"]]) for a in a_runs}) == 1
    for record in collector.records:
        assert record["work"]["aShared"]["sharedBy"] == 3


# ---------------------------------------------------------------------------
# 6. guards
# ---------------------------------------------------------------------------


def test_configuration_guards(tmp_path):
    with pytest.raises(ValueError, match="shadow"):
        bc.BranchConfig(select="adaptive", shadow=False)
    with pytest.raises(ValueError, match="one stream"):
        bc.BranchConfig(select="uniform", continuation="h12", streams=2)
    with pytest.raises(ValueError, match="between floor"):
        bc.BranchConfig(select="adaptive", rate=0.01, floor=0.0625)
    with pytest.raises(ValueError, match="horizon"):
        bc.BranchConfig(select="uniform", horizon=4)
    args = training.argument_parser().parse_args(
        ["collect", "--checkpoint", "c.pt", "--worlds", "1", "--out", "r.pt",
         "--branch-select", "uniform", "--economic-trace", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="economic-trace"):
        training._branch_config(args)
    args = training.argument_parser().parse_args(
        ["collect", "--checkpoint", "c.pt", "--worlds", "1", "--out", "r.pt"]
    )
    assert training._branch_config(args) is None
    assert bc.branch_path_for("x/rollout.pt").name == "rollout.branches.pt"
    with pytest.raises(NotImplementedError):
        bc.warmup_actor_update()


class _Leaky(FakeLaborBridge):
    """A bridge whose restore is not exact (it nudges a camp after restoring)."""

    def restore(self, fork):
        super().restore(fork)
        self.state["camps"][8]["population"] += 1
        return self._step()


def test_an_inexact_restore_is_refused(monkeypatch, models):
    install(monkeypatch)
    collector = _collector(models, bc.BranchConfig(select="uniform", rate=1.0, horizon=2))
    with pytest.raises(RuntimeError, match="changed the rollout"):
        _collect(models, collector, bridge=_Leaky)


def test_head_distances_and_differences(models):
    actor, _, _, normalizer = models
    village = _village_at(7)
    contexts, tensors = training.labor_policy_tensors([village], normalizer)
    catalog = contexts[0][0]
    with torch.no_grad():
        out = actor(tensors)
    from anthropocene_rl.autonomous_v7.labor_model import (
        sample_labor_policy,
        sample_nonlabor_policy,
        schema12_initial_action,
    )

    g = torch.Generator().manual_seed(1)
    payload = schema12_initial_action(
        sample_labor_policy(out, tensors, generator=g), sample_nonlabor_policy(out, generator=g), 0
    )
    zero = bc.head_distances(payload, payload, catalog)
    assert set(zero) == set(bc.HEADS) and all(v == 0.0 for v in zero.values())
    other = dict(payload, fission=(int(payload["fission"]) + 1) % 3, tradeShare=0.75)
    d = bc.head_distances(payload, other, catalog)
    assert d["fission"] == 1.0
    assert d["trade_share"] == pytest.approx(abs(0.75 - payload["tradeShare"]))
    diff = bc.head_differences(payload, other, catalog)
    assert diff["fission"] and diff["trade_share"] and not diff["specialty"]
