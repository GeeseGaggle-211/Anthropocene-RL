"""Stage-2 A(s, a) library: branch selection, paired targets, fit, control variate, D7,
warm-up switch. Synthetic data only; every random draw is seeded."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
from test_labor_model import _labor_tensors
from test_labor_privileged_model import _critics, _layout

from anthropocene_rl.autonomous_v7 import labor_a_stage2 as s2
from anthropocene_rl.autonomous_v7 import labor_advantage as la
from anthropocene_rl.autonomous_v7 import labor_sa_advantage as sa
from anthropocene_rl.autonomous_v7.labor_sa_advantage import SuccStream

G20 = math.exp(-20.0 / 400.0)


# --- 1. branch selection ---------------------------------------------------------------------


def test_head_distances_per_kind() -> None:
    assert s2.head_distance("categorical", [1, 2, 3], [1, 0, 3]).tolist() == [0, 1, 0]
    multi = s2.head_distance("categorical", [[0, 1], [1, 1]], [[0, 0], [0, 0]])
    assert multi.tolist() == [0.5, 1.0]
    tv = s2.head_distance("simplex", [[0.5, 0.5, 0.0]], [[0.0, 0.5, 0.5]])
    assert tv.tolist() == [0.5]
    rows = s2.head_distance("simplex", [[[1, 0], [0, 1]]], [[[1, 0], [1, 0]]])
    assert rows.tolist() == [0.5]  # mean over rows of (0, 1)
    counts = s2.head_distance("counts", [[4, 4, 2]], [[5, 4, 1]])
    assert counts.tolist() == [pytest.approx(0.1)]  # 1 of 10 workers moved
    scalar = s2.head_distance("scalar", [[0.2, 0.4]], [[0.0, 0.0]], scale=2.0)
    assert scalar.tolist() == [pytest.approx(0.15)]
    with pytest.raises(ValueError, match="unknown distance kind"):
        s2.head_distance("ordinal", [0], [0])


def _distances(n: int = 4000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "specialty": rng.beta(1, 6, n),
        "activity": rng.beta(1, 3, n) * 0.5,
        "movement": (rng.random(n) < 0.15).astype(float),
    }


def test_calibration_hits_the_target_rate_including_the_floor() -> None:
    d = _distances()
    cal = s2.calibrate_thresholds(d)
    # far fraction aimed at: (0.25 - 1/16) / (1 - 1/16) = 0.2
    assert cal.far_rate == pytest.approx(0.2, abs=0.01)
    assert cal.expected_rate == pytest.approx(0.25, abs=0.01)
    assert set(cal.thresholds) == set(d)
    probs = s2.branch_probabilities(d, cal.thresholds)
    assert set(np.unique(probs).tolist()) == {1.0 / 16.0, 1.0}
    assert probs.mean() == pytest.approx(cal.expected_rate)
    # Deterministic.
    assert s2.calibrate_thresholds(d) == cal


def test_calibration_with_only_discrete_heads_reports_the_closest_rate() -> None:
    d = {"fission": np.r_[np.ones(10), np.zeros(90)]}
    cal = s2.calibrate_thresholds(d)
    # Only "all differing states" (10%) or none can be far: 10% is closest to 20%.
    assert cal.far_rate == pytest.approx(0.10)
    assert cal.head_triggers == {"fission": 10}


def test_keyed_selection_is_order_free_and_weights_are_inverse_probability() -> None:
    d = _distances(4000, seed=1)
    cal = s2.calibrate_thresholds(d)
    probs = s2.branch_probabilities(d, cal.thresholds)
    keys = [(90154, t, c) for t in range(40) for c in range(100)]
    sel, w = s2.select_branches(probs, keys, seed=5)
    perm = np.random.default_rng(3).permutation(len(keys))
    sel_p, _ = s2.select_branches(probs[perm], [keys[i] for i in perm], seed=5)
    assert np.array_equal(sel_p, sel[perm])
    assert np.array_equal(w, 1.0 / probs)
    # Empirical rate within 4 binomial SE of the expected rate.
    se = math.sqrt((probs * (1 - probs)).sum()) / len(probs)
    assert abs(sel.mean() - probs.mean()) < 4 * se
    # Horvitz-Thompson: the 1/p-weighted mean over selected states is unbiased.
    f = d["specialty"]
    ht = (f * sel * w).sum() / len(f)
    ht_se = math.sqrt((f**2 * (1 - probs) / probs).sum()) / len(f)
    assert abs(ht - f.mean()) < 4 * ht_se
    assert not np.array_equal(sel, s2.select_branches(probs, keys, seed=6)[0])


# --- 2. paired targets -----------------------------------------------------------------------


def _stream(stream: int, rewards, *, rows=(1, 3), status=None, terminated=False) -> SuccStream:
    rewards = np.asarray(rewards, dtype=float).reshape(-1, 4)
    d = len(rewards)
    status = status or {j: "live" for j in range(1, d + 1)}
    pairs = zip((1, d), rows, strict=True)
    row = {j: r for j, r in pairs if r is not None and status.get(j) == "live"}
    return SuccStream(stream, rewards, np.full(d, 20.0), terminated, status, row)


def _v_succ(values: dict[int, float]) -> dict[str, np.ndarray]:
    arr = np.zeros(max(values) + 1)
    for k, v in values.items():
        arr[k] = v
    return {n: arr * (i + 1) for i, n in enumerate(s2.COMPONENTS)}


def test_folded_lambda_return_and_paired_difference() -> None:
    v = _v_succ({1: 10.0, 3: 20.0, 11: 5.0, 13: 7.0})
    a = _stream(0, [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], rows=(1, 3))
    b = _stream(0, [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], rows=(11, 13))
    scale = np.arange(1, 5)

    def folded(r, v1, v3):
        t1 = r[0] + G20 * v1 * scale
        t3 = r[0] + G20 * r[1] + G20**2 * r[2] + G20**3 * v3 * scale
        return 0.05 * t1 + 0.95 * t3

    ga = folded([1, 2, 3], 10.0, 20.0)
    gb = folded([0, 1, 1], 5.0, 7.0)
    assert np.allclose(s2.folded_lambda_return(a, 3, v), ga)
    pt = s2.paired_target([a], [b], 3, v, selection_probability=0.25)
    assert np.allclose(pt.value, ga - gb)
    assert pt.weight == 4.0 and pt.streams == (0,) and pt.k == 3
    one = s2.one_step_paired_signal([a], [b], v, selection_probability=1 / 16)
    assert np.allclose(one.value, (1 + G20 * 10 * scale) - (0 + G20 * 5 * scale))
    assert one.weight == 16.0
    # k = 2 needs a row at decision 2, which export v2 does not record.
    assert s2.paired_target([a], [b], 2, v, selection_probability=1.0) is None
    with pytest.raises(ValueError):
        s2.paired_target([a], [b], 4, v, selection_probability=1.0)


def test_paired_targets_terminated_inactive_and_crn_pairing() -> None:
    v = _v_succ({1: 10.0, 3: 20.0, 21: 1.0, 23: 2.0})
    # Focal camp terminated after the first step: zero future, both T1 and T3 = r0.
    dead = _stream(0, [[4, 4, 4, 4]], rows=(None, None), status={1: "terminated"}, terminated=True)
    live = _stream(0, [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    assert np.allclose(s2.folded_lambda_return(dead, 3, v), 4.0)
    assert s2.paired_target([dead], [live], 3, v, selection_probability=1.0) is not None
    # Inactive at decision 2: the 3-step target is unavailable, so the pair is.
    inactive = _stream(0, [[1] * 4, [1] * 4, [1] * 4], status={1: "live", 2: "inactive"})
    assert s2.paired_target([inactive], [live], 3, v, selection_probability=1.0) is None
    assert s2.one_step_paired_signal([inactive], [live], v, selection_probability=1.0) is not None
    # CRN: only shared stream indices are paired; none shared -> None.
    a1 = _stream(1, [[1] * 4, [1] * 4, [1] * 4], rows=(1, 3))
    b2 = _stream(2, [[0] * 4, [0] * 4, [0] * 4], rows=(21, 23))
    assert s2.paired_target([a1], [b2], 3, v, selection_probability=1.0) is None
    b1 = _stream(1, [[0] * 4, [0] * 4, [0] * 4], rows=(21, 23))
    pt = s2.paired_target([a1, live], [b1, b2], 3, v, selection_probability=1.0)
    assert pt.streams == (1,)


def test_paired_policy_advantages_and_single_action_target() -> None:
    d = torch.tensor([1.5, -2.0])
    pa, pb = s2.paired_policy_advantages(d)
    assert torch.equal(pa, d) and torch.equal(pb, -d)
    _, db = s2.paired_policy_advantages({"qol_years": d})
    assert torch.equal(db["qol_years"], -d)
    v = _v_succ({1: 10.0, 3: 20.0})
    a = _stream(0, [[1] * 4, [2] * 4, [3] * 4])
    g = s2.folded_lambda_return(a, 3, v)
    y = s2.single_action_target([a], 3, np.ones(4), v)
    assert np.allclose(y, g - 1.0)
    assert s2.single_action_target([], 3, np.ones(4), v) is None
    raw = np.asarray([la.REFERENCE_SCALES[n] for n in s2.COMPONENTS])
    assert np.allclose(s2.to_head_units(raw), 1.0)
    assert s2.to_raw_units({n: 1.0 for n in s2.COMPONENTS})["qol_years"] == raw[0]


# --- 3. stage-2 fit --------------------------------------------------------------------------


def _head_setup(seed: int = 3, states: int = 6):
    task, costs = _critics(_layout(False), seed=seed)
    torch.manual_seed(seed)
    head = la.LaborAdvantageHead.for_critics(task, costs, embedding=16, hidden=16)
    tensors = _labor_tensors(batch=states)
    contexts = la.critic_contexts(task, costs, tensors)
    return task, costs, head, tensors, contexts


def _features(n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.as_tensor(
        np.stack([la.action_features(la._synthetic_action(rng)) for _ in range(n)])
    )


def _pairs(states: int, per_state: int, seed: int) -> s2.PairBatch:
    root = torch.arange(states).repeat_interleave(per_state)
    fa, fb = _features(len(root), seed), _features(len(root), seed + 1)
    # Synthetic truth: A depends on the idle share (feature 0) linearly, per objective.
    slope = torch.tensor([1.0, -0.5, 0.2, 0.3])
    target = (fa[:, :1] - fb[:, :1]) * slope
    return s2.PairBatch(root, fa, fb, target, torch.ones(len(root)))


def test_fit_moves_only_a_and_v_is_untouched() -> None:
    task, costs, head, tensors, contexts = _head_setup()
    v_before = la.state_values(task, costs, tensors)
    params_before = [copy.deepcopy(m.state_dict()) for m in (task, costs)]
    head_before = copy.deepcopy(head.state_dict())
    pairs = _pairs(6, 8, seed=10)
    anchor = s2.AnchorBatch(pairs.root, pairs.features_a)
    single = s2.SingleBatch(torch.arange(6), _features(6, 30), torch.zeros(6, 4), torch.ones(6))
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    delta = {n: 1.0 for n in s2.COMPONENTS}
    first = None
    for _ in range(60):
        rep = s2.stage2_fit_step(
            head,
            opt,
            contexts,
            pairs,
            delta=delta,
            single=single,
            anchor=anchor,
            frozen=(task, costs),
        )
        first = first if first is not None else rep.loss
    assert rep.loss < 0.5 * first
    v_after = la.state_values(task, costs, tensors)
    for n in s2.COMPONENTS:
        assert torch.equal(v_before[n], v_after[n])
    for module, before in zip((task, costs), params_before, strict=True):
        for key, value in module.state_dict().items():
            assert torch.equal(value, before[key])
    assert any(not torch.equal(v, head_before[k]) for k, v in head.state_dict().items())


def test_stop_gradient_holds_even_for_differentiable_contexts() -> None:
    _, _, head, _, contexts = _head_setup()
    for module in head.heads.values():
        torch.nn.init.normal_(module[2].weight, std=0.5)
    live = {n: c.clone().requires_grad_(True) for n, c in contexts.items()}
    pairs = _pairs(6, 4, seed=11)
    loss, _ = s2.stage2_loss(head, live, pairs, delta={n: 1.0 for n in s2.COMPONENTS})
    loss.backward()
    assert all(c.grad is None for c in live.values())
    # The optimizer guard refuses non-head parameters.
    task, _, head2, _, ctx2 = _head_setup()
    opt = torch.optim.SGD([*head2.parameters(), *task.parameters()], lr=0.1)
    with pytest.raises(ValueError, match="only hold A-head"):
        s2.stage2_fit_step(head2, opt, ctx2, pairs, delta={n: 1.0 for n in s2.COMPONENTS})


def test_actor_centring_is_the_anchor_mean_and_pairs_ignore_it() -> None:
    _, _, head, _, contexts = _head_setup()
    for module in head.heads.values():
        torch.nn.init.normal_(module[2].weight, std=0.5)
    root = torch.tensor([0, 0, 0, 1, 1, 2])
    feats = _features(6, 40)
    anchor = s2.AnchorBatch(root, feats)
    centred = s2.actor_centred_advantage(head, contexts, feats, root, anchor)
    for n in s2.COMPONENTS:
        mean = torch.zeros(6).index_add(0, root, centred[n]) / torch.bincount(
            root, minlength=6
        ).clamp_min(1)
        assert torch.allclose(mean[:3], torch.zeros(3), atol=1e-6)
    raw = head(contexts, feats, root)
    for n in s2.COMPONENTS:
        assert torch.allclose(centred[n][0] - centred[n][1], raw[n][0] - raw[n][1], atol=1e-6)
    with pytest.raises(ValueError, match="anchor sample"):
        s2.actor_centred_advantage(head, contexts, feats, torch.tensor([3, 3, 3, 3, 3, 3]), anchor)


# --- 4. control variate: unbiased whatever A is ---------------------------------------------

# Toy factorised policy: categorical head (3 logits) x Gaussian head (mean mu, sd 1).
THETA = np.array([0.2, -0.3, 0.5])
MU = 0.4
R_TAB = np.array([1.0, -0.5, 0.3])
W_TAB = np.array([-2.0, 1.0, 0.5])  # the deliberately wrong A
C_W, D_W, E_W = 0.8, -1.5, 2.0


def _pi() -> np.ndarray:
    e = np.exp(THETA - THETA.max())
    return e / e.sum()


def _reward_mean(a1, a2):
    return R_TAB[a1] + 2 * a2 - 0.5 * a2**2 + 0.7 * (a1 == 1) * a2


def _wrong_a(a1, a2):
    return W_TAB[a1] + C_W * a2**2 + D_W * a2 + E_W * (a1 == 2) * a2


def _true_gradient() -> np.ndarray:
    pi = _pi()
    m = R_TAB + 2 * MU - 0.5 * (MU**2 + 1) + 0.7 * (np.arange(3) == 1) * MU
    return np.r_[pi * (m - pi @ m), 2 - MU + 0.7 * pi[1]]


def _toy_batch(n: int, seed: int, a_fn):
    rng = np.random.default_rng(seed)
    pi = _pi()
    a1 = rng.choice(3, size=n, p=pi)
    a2 = MU + rng.standard_normal(n)
    reward = _reward_mean(a1, a2) + 0.5 * rng.standard_normal(n)
    expected_r = pi @ (R_TAB + 2 * MU - 0.5 * (MU**2 + 1) + 0.7 * (np.arange(3) == 1) * MU)
    gae = reward - expected_r  # V-only, exact V
    score1 = np.eye(3)[a1] - pi
    score2 = (a2 - MU)[:, None]
    a_all = np.stack([a_fn(np.full(n, k), a2) for k in range(3)], axis=1)  # [n, 3]
    e1 = a_all @ pi
    grad_e1 = pi * (a_all - e1[:, None])
    # E_2 and its mu-gradient by Gauss-Hermite quadrature (exact for these polynomials).
    x, w = np.polynomial.hermite_e.hermegauss(12)
    w = w / w.sum()
    pts = MU + x
    e2 = np.stack([a_fn(a1, np.full(n, p)) for p in pts], axis=1) @ w
    grad_e2 = (np.stack([a_fn(a1, np.full(n, p)) for p in pts], axis=1) * x) @ w
    joint = sum(pi[k] * (a_fn(np.full(len(pts), k), pts) @ w) for k in range(3))
    t = lambda z: torch.as_tensor(z, dtype=torch.float64)  # noqa: E731
    return {
        "a_mean": t(np.full(n, joint)),
        "gae": t(gae),
        "a_exec": t(a_fn(a1, a2)),
        "a_exp": {"cat": t(e1), "gauss": t(e2)},
        "score": {"cat": t(score1), "gauss": t(score2)},
        "corr": {"cat": t(grad_e1), "gauss": t(grad_e2[:, None])},
    }


def _estimates(batch, eta):
    coef = s2.control_variate_advantages(batch["gae"], batch["a_exec"], batch["a_mean"], eta)
    out = []
    for h in ("cat", "gauss"):
        est = batch["score"][h] * coef[h][:, None] + eta[h] * batch["corr"][h]
        out.append(est)
    return torch.cat(out, dim=1).numpy()


def test_control_variate_is_unbiased_with_a_deliberately_wrong_a() -> None:
    """400k draws, fixed seed. Tolerance: 4 standard errors of the per-component Monte-Carlo
    mean (two-sided false-alarm ~6e-5 per component for a normal mean); the same tolerance
    flags the estimator WITHOUT the analytic correction, so the check has power."""
    true = _true_gradient()
    batch = _toy_batch(400_000, seed=2026, a_fn=_wrong_a)
    for eta in ({"cat": 0.0, "gauss": 0.0}, {"cat": 1.0, "gauss": 1.0}, {"cat": 1.0, "gauss": 0.0}):
        est = _estimates(batch, eta)
        se = est.std(axis=0, ddof=1) / math.sqrt(len(est))
        assert (np.abs(est.mean(axis=0) - true) < 4 * se).all(), (eta, est.mean(axis=0), true, se)
    # Dropping the correction term (score * residual only) is biased and detected.
    coef = s2.control_variate_advantages(
        batch["gae"], batch["a_exec"], batch["a_mean"], {"cat": 1.0, "gauss": 1.0}
    )
    biased = torch.cat([batch["score"][h] * coef[h][:, None] for h in ("cat", "gauss")], 1).numpy()
    se = biased.std(axis=0, ddof=1) / math.sqrt(len(biased))
    assert (np.abs(biased.mean(axis=0) - true) > 4 * se).any()


def test_surrogate_and_correction_autograd_match_the_per_sample_estimator() -> None:
    n = 64
    rng = np.random.default_rng(7)
    a1 = torch.as_tensor(rng.choice(3, size=n, p=_pi()))
    # Recompute with tensors that carry the policy parameters.
    theta = torch.tensor(THETA, requires_grad=True)
    mu = torch.tensor(MU, dtype=torch.float64, requires_grad=True)
    a2 = torch.as_tensor(MU + rng.standard_normal(n))
    logp_cat = torch.log_softmax(theta, 0)[a1]
    logp_gauss = -0.5 * (a2 - mu) ** 2
    gae = torch.as_tensor(rng.standard_normal(n))
    a_exec = torch.as_tensor(_wrong_a(a1.numpy(), a2.numpy()))
    a_all = torch.stack([torch.as_tensor(_wrong_a(np.full(n, k), a2.numpy())) for k in range(3)], 1)
    e_cat = s2.categorical_expectation(torch.softmax(theta, 0).expand(n, 3), a_all)
    e_gauss = torch.as_tensor(W_TAB)[a1] + C_W * (mu**2 + 1) + D_W * mu + E_W * (a1 == 2) * mu
    eta = {"cat": 1.0, "gauss": 1.0}
    a_mean = torch.full((n,), 0.37, dtype=torch.float64)  # any state-only value
    coef = s2.control_variate_advantages(gae, a_exec, a_mean, eta)
    cur = {"cat": logp_cat, "gauss": logp_gauss}
    old = {k: v.detach() for k, v in cur.items()}
    objective = s2.per_head_surrogate(cur, old, coef, clip=0.2).mean()
    objective = objective + s2.control_variate_correction({"cat": e_cat, "gauss": e_gauss}, eta)
    objective.backward()
    pi = torch.softmax(theta.detach(), 0)
    score_cat = torch.nn.functional.one_hot(a1, 3).double() - pi
    score_gauss = a2 - MU
    grad_e_cat = pi * (a_all - e_cat.detach()[:, None])
    grad_e_gauss = 2 * C_W * MU + D_W + E_W * (a1 == 2).double()
    want_theta = (score_cat * coef["cat"][:, None] + grad_e_cat).mean(0)
    want_mu = (score_gauss * coef["gauss"] + grad_e_gauss).mean()
    assert torch.allclose(theta.grad, want_theta, atol=1e-10)
    assert torch.allclose(mu.grad, want_mu, atol=1e-10)


def test_per_head_surrogate_with_equal_advantages_matches_the_joint_surrogate() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 3, generator=g, requires_grad=True)
    old = {"a": torch.randn(32, generator=g), "b": torch.randn(32, generator=g)}
    adv = torch.randn(32, generator=g)

    def current():
        return {"a": x[:, 0] * 2.0 + old["a"], "b": x[:, 1] * 0.3 + old["b"]}

    s2.per_head_surrogate(current(), old, {"a": adv, "b": adv}, clip=0.2).sum().backward()
    split = x.grad.clone()
    x.grad = None
    cur = current()
    ratio = torch.exp(cur["a"] + cur["b"] - old["a"] - old["b"])
    torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).sum().backward()
    assert torch.allclose(split, x.grad, atol=1e-6)
    assert bool(((ratio < 0.8) | (ratio > 1.2)).any())  # clipping is exercised


# --- 5. D7 per-head variance -----------------------------------------------------------------


def _true_a(a1, a2):
    # Q(a) - V for the toy reward: the ideal control variate.
    return _reward_mean(a1, a2)


def test_d7_adopts_a_good_a_and_rejects_a_wrong_one() -> None:
    good = _toy_batch(20_000, seed=11, a_fn=_true_a)
    bad = _toy_batch(20_000, seed=11, a_fn=_wrong_a)
    rep_good, mask_good = s2.measure_head_variance(
        good["score"], good["gae"], good["a_exec"], good["a_mean"], good["corr"]
    )
    rep_bad, mask_bad = s2.measure_head_variance(
        bad["score"], bad["gae"], bad["a_exec"], bad["a_mean"], bad["corr"]
    )
    assert mask_good == {"cat": True, "gauss": True}
    assert all(r.reduction >= s2.D7_VARIANCE_REDUCTION for r in rep_good.values())
    assert mask_bad == {"cat": False, "gauss": False}
    assert all(r.reduction < 0 for r in rep_bad.values())
    assert s2.connection_eta(mask_good) == {"cat": 1.0, "gauss": 1.0}
    # V-only variance is the same whatever A is (same draws).
    for h in ("cat", "gauss"):
        assert rep_good[h].variance_v_only == pytest.approx(rep_bad[h].variance_v_only)


def test_per_sample_gradients_and_keyed_projection() -> None:
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    out = x @ w
    grads = s2.per_sample_gradients(out, [w])
    assert torch.equal(grads, x)
    p1 = s2.per_sample_gradients(out, [w], projection_dim=8, key=4)
    p2 = s2.per_sample_gradients(out, [w], projection_dim=8, key=4)
    assert torch.equal(p1, p2) and p1.shape == (2, 8)
    # Trace preservation in expectation: average over keys.
    norms = np.mean(
        [
            float(s2.per_sample_gradients(out, [w], projection_dim=8, key=k)[1].square().sum())
            for k in range(400)
        ]
    )
    assert norms == pytest.approx(2.0, rel=0.1)


# --- 6. warm-up switch -----------------------------------------------------------------------


def _report(slope=1.0, bias=0.1, td=(0.05, 0.02), kl=(0.01, 0.02)) -> s2.WarmupReport:
    # World labels only; per value component since Amendment B5 (8 Nov 2026).
    return s2.WarmupReport(
        calibration_slope={"qol_years": slope, "population_years": 1.05},
        calibration_bias={"qol_years": bias, "population_years": 0.0},
        realised_spread={"qol_years": 1.0, "population_years": 1.0},
        td_by_world={
            "qol_years": {50001: td[0], 50002: td[1]},
            "population_years": {50001: 0.0, 50002: 0.0},
        },
        dry_run_kl=kl,
    )


def test_warmup_switch_fails_closed_and_latches() -> None:
    # An unset threshold still fails its gate closed.
    unset = s2.WarmupSwitch(s2.WarmupThresholds(td_split_fraction=None, kl_bound=None))
    d = unset.update(_report())
    assert d.phase == "warmup" and not d.gates["td_split"] and not d.gates["dry_run_kl"]
    assert any("unset" in r for r in d.reasons)
    # Amendment B5 defaults: TD split 0.05 x spread, dry-run KL 0.02.
    assert s2.WarmupSwitch().update(_report()).passed  # |0.05| <= 0.05 x spread 1.0
    assert not s2.WarmupSwitch().update(_report(td=(0.06, 0.02))).gates["td_split"]
    assert not s2.WarmupSwitch().update(_report(kl=(0.01, 0.03))).gates["dry_run_kl"]
    sw = s2.WarmupSwitch(s2.WarmupThresholds(td_split_fraction=0.1, kl_bound=0.05, consecutive=2))
    assert sw.update(_report(slope=1.3)).gates["v_calibration"] is False
    assert sw.update(_report(td=(0.2, 0.0))).gates["td_split"] is False
    assert sw.update(_report(kl=(0.01, 0.09))).gates["dry_run_kl"] is False
    first = sw.update(_report())
    assert first.passed and first.phase == "warmup" and first.streak == 1
    assert sw.update(_report()).phase == "gae"
    assert sw.update(_report(slope=2.0)).phase == "gae"  # latched
    td = s2.per_world_td([1, 1, 2], [0.1, 0.3, -0.2])
    assert td[1]["mean"] == pytest.approx(0.2) and td[2]["n"] == 1


# --- follow-up: exact lambda-return ---------------------------------------------------------


def test_exact_lambda_return_matches_hand_recursion_and_limits() -> None:
    lam = 0.95
    v = _v_succ({1: 10.0, 2: 15.0, 3: 20.0})
    scale = np.arange(1, 5)
    r = np.array([[1.0] * 4, [2.0] * 4, [3.0] * 4])
    full = SuccStream(
        0, r, np.full(3, 20.0), False, {1: "live", 2: "live", 3: "live"}, {1: 1, 2: 2, 3: 3}
    )
    g3 = r[2] + G20 * 20.0 * scale
    g2 = r[1] + G20 * ((1 - lam) * 15.0 * scale + lam * g3)
    g1 = r[0] + G20 * ((1 - lam) * 10.0 * scale + lam * g2)
    value, exact = s2.lambda_return(full, 3, v, lam=lam)
    assert exact and np.allclose(value, g1)
    # lam = 1 is the plain k-step target, lam = 0 the one-step target.
    one = s2.exact_lambda_return(full, 3, v, lam=1.0)
    assert np.allclose(one, sa.successor_target(full, 3, v))
    zero = s2.exact_lambda_return(full, 3, v, lam=0.0)
    assert np.allclose(zero, sa.successor_target(full, 1, v))
    # Terminated at decision 2: G_1 = r_1, no row needed after it.
    dead = SuccStream(0, r[:2], np.full(2, 20.0), True, {1: "live", 2: "terminated"}, {1: 1})
    want = r[0] + G20 * ((1 - lam) * 10.0 * scale + lam * r[1])
    value, exact = s2.lambda_return(dead, 3, v, lam=lam)
    assert exact and np.allclose(value, want)
    # Inactive -> unavailable.
    gone = SuccStream(0, r, np.full(3, 20.0), False, {1: "live", 2: "inactive"}, {1: 1})
    assert s2.lambda_return(gone, 3, v) == (None, False)


def test_lambda_return_falls_back_to_the_fold_without_intermediate_rows() -> None:
    v = _v_succ({1: 10.0, 3: 20.0, 11: 5.0, 12: 6.0, 13: 7.0})
    a = _stream(0, [[1] * 4, [2] * 4, [3] * 4], rows=(1, 3))  # export v2: s_1 and s_k only
    value, exact = s2.lambda_return(a, 3, v)
    assert not exact
    assert np.allclose(value, s2.folded_lambda_return(a, 3, v))
    b = SuccStream(
        0,
        np.zeros((3, 4)),
        np.full(3, 20.0),
        False,
        {1: "live", 2: "live", 3: "live"},
        {1: 11, 2: 12, 3: 13},
    )
    pt = s2.paired_target([a], [b], 3, v, selection_probability=0.5)
    assert not pt.exact  # one side folded
    assert np.allclose(pt.value, s2.folded_lambda_return(a, 3, v) - s2.exact_lambda_return(b, 3, v))
    pt_b = s2.paired_target([b], [b], 3, v, selection_probability=0.5)
    assert pt_b.exact and np.allclose(pt_b.value, 0.0)


# --- follow-up: autoregressive two-head control variate -------------------------------------

BETA = 0.8  # head 2 (Gaussian) mean = MU + BETA * a1: a child of head 1


def _ar_true_gradient() -> np.ndarray:
    pi = _pi()
    k = np.arange(3)
    m = MU + BETA * k
    big_m = R_TAB + 2 * m - 0.5 * (m**2 + 1) + 0.7 * (k == 1) * m
    j = pi @ big_m
    dm = 2 - m + 0.7 * (k == 1)
    return np.r_[pi * (big_m - j), pi @ dm, pi @ (dm * k)]


def _ar_batch(n: int, seed: int, a_fn, *, integrate_children: bool = True):
    rng = np.random.default_rng(seed)
    pi = _pi()
    a1 = rng.choice(3, size=n, p=pi)
    mean2 = MU + BETA * a1
    a2 = mean2 + rng.standard_normal(n)
    reward = _reward_mean(a1, a2) + 0.5 * rng.standard_normal(n)
    gae = reward - _ar_value()  # V-only, exact V
    x, w = np.polynomial.hermite_e.hermegauss(12)
    w = w / w.sum()
    # n_k = E_{a2 ~ N(MU + BETA k)} A(k, a2): head 1's E integrates over its child.
    n_k = np.array([a_fn(np.full(len(x), k), MU + BETA * k + x) @ w for k in range(3)])
    if integrate_children:
        a_all = np.broadcast_to(n_k, (n, 3))
    else:  # wrong for this actor: child held at its sample
        a_all = np.stack([a_fn(np.full(n, k), a2) for k in range(3)], axis=1)
    e1 = a_all @ pi
    grad_e1 = pi * (a_all - e1[:, None])
    vals = np.stack([a_fn(a1, mean2 + p) for p in x], axis=1)  # [n, q]
    stein = (vals * x) @ w  # d/dmean E_{N(mean,1)} A(a1, .)
    grad_e2 = np.stack([stein, stein * a1], axis=1)  # wrt (MU, BETA), parent a1 fixed
    score2 = (a2 - mean2)[:, None] * np.stack([np.ones(n), a1], axis=1)
    t = lambda z: torch.as_tensor(z, dtype=torch.float64)  # noqa: E731
    return {
        "gae": t(gae),
        "a_exec": t(a_fn(a1, a2)),
        "a_mean": t(np.full(n, pi @ n_k)),
        "score": {"cat": t(np.eye(3)[a1] - pi), "gauss": t(score2)},
        "corr": {"cat": t(grad_e1), "gauss": t(grad_e2)},
    }


def _ar_value() -> float:
    pi = _pi()
    k = np.arange(3)
    m = MU + BETA * k
    return float(pi @ (R_TAB + 2 * m - 0.5 * (m**2 + 1) + 0.7 * (k == 1) * m))


def test_control_variate_is_unbiased_for_an_autoregressive_two_head_policy() -> None:
    """Head 2's mean depends on head 1's sample. Same 4-SE tolerance as the factorised
    test. Head 1's E must integrate over its child head; holding the child at its sample
    (the factorised formula) is biased here and the test detects it."""
    true = _ar_true_gradient()
    batch = _ar_batch(400_000, seed=2027, a_fn=_wrong_a)
    for eta in ({"cat": 0.0, "gauss": 0.0}, {"cat": 1.0, "gauss": 1.0}, {"cat": 0.0, "gauss": 1.0}):
        est = _estimates(batch, eta)
        se = est.std(axis=0, ddof=1) / math.sqrt(len(est))
        assert (np.abs(est.mean(axis=0) - true) < 4 * se).all(), (eta, est.mean(axis=0), true)
    naive = _ar_batch(400_000, seed=2027, a_fn=_wrong_a, integrate_children=False)
    est = _estimates(naive, {"cat": 1.0, "gauss": 1.0})
    se = est.std(axis=0, ddof=1) / math.sqrt(len(est))
    assert (np.abs(est.mean(axis=0)[:3] - true[:3]) > 4 * se[:3]).any()
    # And a good A still cuts the variance of both heads by the D7 margin.
    good = _ar_batch(20_000, seed=5, a_fn=_true_a)
    _, mask = s2.measure_head_variance(
        good["score"], good["gae"], good["a_exec"], good["a_mean"], good["corr"]
    )
    assert mask == {"cat": True, "gauss": True}


def test_warmup_alarm_after_the_one_way_switch() -> None:
    sw = s2.WarmupSwitch(s2.WarmupThresholds(td_split_fraction=0.1, kl_bound=0.05))
    first = sw.update(_report())
    assert first.phase == "gae" and not first.alarm
    bad = sw.update(_report(slope=2.0))
    assert bad.phase == "gae" and bad.alarm and not bad.passed
    assert not sw.update(_report()).alarm
    unset = s2.WarmupThresholds(td_split_fraction=None, kl_bound=None)
    assert not s2.WarmupSwitch(unset).update(_report()).alarm  # no alarm while warming up
