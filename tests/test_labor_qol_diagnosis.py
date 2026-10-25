"""Metric math of the read-only QoL critic diagnosis (synthetic arrays only)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_pretrain_eval as ev
from anthropocene_rl.autonomous_v7 import labor_qol_diagnosis as qd
from anthropocene_rl.autonomous_v7.labor_bc import CRITIC_DISCOUNT_DAYS, discounted_returns


def test_symlog_round_trip_and_rankdata() -> None:
    x = np.array([-1e4, -2.0, 0.0, 0.3, 50.0])
    assert np.allclose(qd.symexp(qd.symlog(x)), x)
    assert qd.rankdata(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [2.5, 0.0, 2.5, 1.0]
    assert qd.spearman(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 15.0])) == pytest.approx(
        0.5
    )
    assert qd.pearson(np.array([1.0, 1.0]), np.array([1.0, 2.0])) is None


def test_variance_decomposition_sums_and_known_split() -> None:
    values = np.array([0.0, 2.0, 10.0, 12.0])
    groups = np.array([0, 0, 1, 1])
    d = qd.variance_decomposition(values, groups)
    assert d["within"] == pytest.approx(1.0)
    assert d["total"] == pytest.approx(values.var())
    assert d["between"] == pytest.approx(25.0)
    assert d["within"] + d["between"] == pytest.approx(d["total"])
    assert d["within_fraction"] == pytest.approx(1.0 / 26.0)


def test_within_group_r2_ignores_group_offsets() -> None:
    target = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 13.0])
    groups = np.array([0, 0, 0, 1, 1, 1])
    assert qd.within_group_r2(
        target + np.array([5, 5, 5, -3, -3, -3]), target, groups
    ) == pytest.approx(1.0)
    means = np.array([1.0, 1.0, 1.0, 34 / 3, 34 / 3, 34 / 3])
    assert qd.within_group_r2(means, target, groups) == pytest.approx(0.0)
    assert qd.within_group_r2(-target, target, groups) < 0.0


def test_pair_index_counts_and_mask() -> None:
    groups = np.array([0, 0, 0, 1, 1, 2])
    worlds = np.array([7, 7, 7, 8, 8, 9])
    pairs = qd.PairIndex(groups, worlds)
    assert len(pairs) == 3 + 1
    assert np.all(groups[pairs.i] == groups[pairs.j])
    masked = qd.PairIndex(groups, worlds, np.array([True, True, False, True, True, True]))
    assert len(masked) == 2
    assert set(pairs.world.tolist()) == {7, 8}


def test_pair_ranking_matches_existing_evaluator_and_splits_margins() -> None:
    rng = np.random.default_rng(4)
    n = 300
    worlds = rng.integers(0, 4, n)
    turns = rng.integers(0, 5, n)
    target = rng.normal(0, 1, n)
    pred = target + rng.normal(0, 1, n)
    pred[:10] = pred[10]  # some prediction ties
    groups = qd.group_ids(worlds, turns)
    pairs = qd.PairIndex(groups, worlds)
    mine = qd.pair_ranking(pred, target, pairs, 0.05, bootstrap=200, seed=1)
    theirs = ev.ranking_accuracy(pred, target, worlds, turns, 0.05)
    assert mine["accuracy"] == pytest.approx(theirs["accuracy"])
    assert mine["informative_pairs"] == theirs["informative_pairs"]
    assert mine["world_balanced_accuracy"] == pytest.approx(theirs["world_balanced_accuracy"])
    assert sum(b["pairs"] for b in mine["by_margin"]) == mine["informative_pairs"]
    low, high = mine["bootstrap_ci95"]
    assert low <= mine["accuracy"] <= high
    # A perfect predictor is right at every margin; a constant one scores one half.
    perfect = qd.pair_ranking(target, target, pairs, 0.05)
    assert perfect["accuracy"] == 1.0
    assert qd.pair_ranking(np.zeros(n), target, pairs, 0.05)["accuracy"] == 0.5


def test_informative_pair_structure_in_bins() -> None:
    width = 3.0 / 254.0
    # Two groups of two rows, normalized targets chosen to differ by known symlog amounts.
    y = np.array([1.0, 1.0 + 0.5 * width, 1.0, 1.0 + 3.0 * width])
    normalized = qd.symexp(y)
    groups = np.array([0, 0, 1, 1])
    pairs = qd.PairIndex(groups, np.zeros(4, int))
    s = qd.informative_pair_structure(normalized, pairs, threshold=0.0, bin_width=width)
    assert s["informative_pairs"] == 2
    assert s["fraction_below_bins"]["1"] == pytest.approx(0.5)
    assert s["fraction_below_bins"]["2"] == pytest.approx(0.5)
    assert s["fraction_below_bins"]["5"] == pytest.approx(1.0)
    # The default threshold (0.05 reference scales) is ~1.6 bins at y=1: it keeps only the 3-bin pair.
    assert (
        qd.informative_pair_structure(normalized, pairs, bin_width=width)["informative_pairs"] == 1
    )


def test_group_spread_detects_collapse() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 5)
    target = rng.normal(0, 1, 100) + groups
    collapsed = qd.group_spread(groups.astype(float), target, groups)
    assert collapsed["pooled_sd_ratio"] == pytest.approx(0.0)
    assert collapsed["collapse_fraction"] == 1.0
    exact = qd.group_spread(target, target, groups)
    assert exact["pooled_sd_ratio"] == pytest.approx(1.0)
    assert exact["within_slope"] == pytest.approx(1.0)
    assert exact["mean_within_correlation"] == pytest.approx(1.0)
    shrunk = qd.group_spread(0.5 * target, target, groups)
    assert shrunk["within_slope"] == pytest.approx(0.5)


def _trajectory_dataset() -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(2)
    worlds, camps, turns, terminal = [], [], [], []
    for w in (1, 2):
        for c in (10, 11, 12):
            for t in range(6):
                worlds.append(w)
                camps.append(c)
                turns.append(t)
                terminal.append(c == 11 and t == 3)
    n = len(worlds)
    order = rng.permutation(n)  # rows need not be sorted
    data = {
        "row_world": torch.tensor(worlds)[order],
        "row_camp": torch.tensor(camps)[order],
        "row_turn": torch.tensor(turns)[order],
        "transition_terminal": torch.tensor(terminal)[order],
        "transition_elapsed_days": torch.as_tensor(rng.uniform(15, 25, n))[order],
        "reward_qol_years": torch.as_tensor(rng.normal(5, 2, n))[order],
        "reward_population_years": torch.as_tensor(rng.normal(900, 50, n))[order],
        "reward_feasibility": torch.zeros(n, dtype=torch.float64),
        "reward_settlement_guard": torch.zeros(n, dtype=torch.float64),
    }
    return data


def test_windowed_returns_reproduce_discounted_returns_and_truncate() -> None:
    data = _trajectory_dataset()
    reference = discounted_returns(data)["qol_years"].double().numpy()
    args = (
        data["reward_qol_years"].numpy(),
        data["transition_elapsed_days"].numpy(),
        data["transition_terminal"].numpy(),
        qd.group_ids(data["row_world"].numpy(), data["row_camp"].numpy()),
        data["row_turn"].numpy(),
    )
    full, remaining, ends = qd.windowed_returns(*args, window=None, discount=CRITIC_DISCOUNT_DAYS)
    assert np.allclose(full, reference, rtol=1e-6)
    one, _, _ = qd.windowed_returns(*args, window=1, discount=CRITIC_DISCOUNT_DAYS)
    assert np.allclose(one, data["reward_qol_years"].numpy())
    long, _, _ = qd.windowed_returns(*args, window=100, discount=CRITIC_DISCOUNT_DAYS)
    assert np.allclose(long, full)
    turns, camps = data["row_turn"].numpy(), data["row_camp"].numpy()
    # Camp 11 ends in a terminal at turn 3: turn 0 has 4 decisions left, turn 4 starts a new segment.
    assert remaining[(camps == 11) & (turns == 0)].tolist() == [4, 4]
    assert ends[(camps == 11) & (turns == 0)].all()
    assert remaining[(camps == 10) & (turns == 5)].tolist() == [1, 1]
    assert not ends[camps == 10].any()
    # Two-step window by hand.
    three, _, _ = qd.windowed_returns(*args, window=2, discount=CRITIC_DISCOUNT_DAYS)
    rows = np.flatnonzero((camps == 12) & (data["row_world"].numpy() == 1))
    rows = rows[np.argsort(turns[rows])]
    r, e = args[0][rows], args[1][rows]
    assert three[rows[0]] == pytest.approx(r[0] + math.exp(-e[0] / CRITIC_DISCOUNT_DAYS) * r[1])


def test_remaining_and_magnitude_labels() -> None:
    assert qd.remaining_labels(np.array([1, 2, 3, 6, 7, 12, 25])).tolist() == [
        "1",
        "2-3",
        "2-3",
        "4-6",
        "7-12",
        "7-12",
        "13+",
    ]
    labels, edges = qd.magnitude_labels(np.arange(10.0), 5)
    assert labels.tolist() == ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4", "q5", "q5"]
    assert edges[0] == 0.0 and edges[-1] == 9.0


def test_group_folds_keep_units_together() -> None:
    units = np.repeat(np.arange(9), 4)
    folds = qd.group_folds(units, 3, seed=5)
    for u in range(9):
        assert len(set(folds[units == u].tolist())) == 1
    assert sorted(set(folds.tolist())) == [0, 1, 2]


def test_ridge_matches_closed_form_and_drops_constant_columns() -> None:
    rng = np.random.default_rng(3)
    n, d = 200, 6
    x = rng.normal(0, 1, (n, d))
    x[:, 4] = 3.0  # constant column
    x[:, 5] *= 100.0  # badly scaled column
    y = np.stack([x[:, 0] - 2 * x[:, 1] + 0.01 * x[:, 5] + rng.normal(0, 0.1, n), x[:, 2]], axis=1)
    xt, yt = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y)
    rows = np.arange(n)
    stats = qd.GramStats.compute(xt, yt, rows, chunk=37)
    model = qd.RidgeModel(stats)
    assert model.effective_features == d - 1
    alpha = 0.1
    pred = model.predict_projected(model.project(xt, rows), alpha).numpy()
    keep = [0, 1, 2, 3, 5]
    xs = x[:, keep].astype(np.float32).astype(np.float64)
    mu, sd = xs.mean(0), xs.std(0)
    z = (xs - mu) / sd
    lam = alpha * model.scale
    w = np.linalg.solve(z.T @ z + lam * np.eye(len(keep)), z.T @ (y - y.mean(0)))
    assert np.allclose(pred, z @ w + y.mean(0), atol=1e-6)
    # Subtracting a fold's statistics equals computing on the remaining rows.
    fold = rows < 50
    direct = qd.GramStats.compute(xt, yt, rows[~fold])
    diff = stats - qd.GramStats.compute(xt, yt, rows[fold])
    assert diff.n == direct.n
    assert torch.allclose(diff.gram, direct.gram, rtol=1e-9, atol=1e-6)
    assert torch.allclose(diff.xty, direct.xty, rtol=1e-9, atol=1e-6)


def test_cross_validated_ridge_and_knn_recover_signal() -> None:
    rng = np.random.default_rng(8)
    units = np.repeat(np.arange(10), 30)
    x = rng.normal(0, 1, (300, 5))
    y = np.stack([x[:, 0] + 0.05 * rng.normal(0, 1, 300)], axis=1)
    xt, yt = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y)
    ridge, knn = qd.cross_validated(xt, yt, units, 5, [1e-3], 8, 0)
    assert np.isfinite(ridge).all() and np.isfinite(knn).all()
    assert qd.r2(ridge[:, 0], y[:, 0]) > 0.95
    assert qd.r2(knn[:, 0], y[:, 0]) > 0.5
    alphas, _ = qd.select_alpha(xt, yt, np.arange(300), units, 3, 0)
    assert len(alphas) == 1 and alphas[0] in qd.RIDGE_ALPHAS
    exact = qd.knn_predict(torch.eye(3), torch.tensor([[1.0], [2.0], [3.0]]), torch.eye(3)[[2]], 1)
    assert exact.item() == 3.0


def _record(qol_streams: dict[int, list[float]]) -> dict:
    """A two-horizon root with three candidates; candidate 0 is H12."""

    def cand(index, steps, reeval=None):
        returns = qd.candidate_horizon_returns(steps, len(steps), math.exp(-20.0 / 400.0))
        return {
            "index": index,
            "alive": True,
            "family": "h12" if index == 0 else "actor-sample",
            "step_rewards": steps,
            "returns": dict(
                zip(
                    ("qol_years", "population_years", "feasibility", "settlement_guard"),
                    returns.tolist(),
                    strict=True,
                )
            ),
            "reeval_returns": {} if reeval is None else reeval,
        }

    scale_q = 43.152
    reeval0 = {str(s): [v, 1000.0, 0.0, 0.0] for s, v in enumerate(qol_streams[0], 1)}
    reeval1 = {str(s): [v, 1000.0, 0.0, 0.0] for s, v in enumerate(qol_streams[1], 1)}
    return {
        "world": 1,
        "turn": 3,
        "final_horizon": 2,
        "personalities": [{"name": "balanced", "vector": [0.5, 0.5]}],
        "candidates": [
            cand(0, [[1.0 * scale_q, 500.0, 0.0, 0.0], [1.0 * scale_q, 500.0, 0.0, 0.0]], reeval0),
            cand(1, [[1.2 * scale_q, 500.0, 0.0, 0.0], [1.2 * scale_q, 500.0, 0.0, 0.0]], reeval1),
            cand(2, [[1.0 * scale_q, 900.0, 0.0, 0.0]]),  # eliminated after horizon 1
        ],
    }


def test_root_disparity_paired_se_and_horizons() -> None:
    config = {"horizons": [1, 2], "feasibility_multiplier": 1.0, "guard_multiplier": 1.0}
    scale_q = 43.152
    streams = {0: [86.0, 87.0, 88.0], 1: [96.0, 98.0, 97.0]}
    root = qd.root_disparity(_record(streams), config, math.exp(-20.0 / 400.0))
    assert root["reconstruction_max_rel_error"] < 1e-12
    d = (np.array(streams[0]) - np.array(streams[1])) / scale_q
    mean, se = root["exact_pairs"]["qol_years"][0]
    assert mean == pytest.approx(d.mean())
    assert se == pytest.approx(d.std(ddof=1) / math.sqrt(3))
    assert root["noise"]["paired_stream_sd"]["qol_years"] == pytest.approx(d.std(ddof=1))
    # Horizon 1 has all three candidates, horizon 2 only the two survivors.
    assert root["horizons"]["1"]["candidates"] == 3
    assert root["horizons"]["2"]["candidates"] == 2
    assert root["horizons"]["1"]["qol_years"]["range"] == pytest.approx(0.2)
    assert root["gains"]["qol"] == pytest.approx([0.2 * (1 + math.exp(-0.05))])


def test_search_disparity_reads_tables(tmp_path) -> None:
    shard = tmp_path / "world-1"
    shard.mkdir()
    near = _record({0: [86.0, 87.0, 88.0], 1: [86.5, 87.5, 88.0]})  # within noise
    far = _record({0: [86.0, 87.0, 88.0], 1: [96.0, 98.0, 97.0]})  # beyond 2 SE
    (shard / "candidates.jsonl").write_text(json.dumps(near) + "\n" + json.dumps(far) + "\n")
    report = qd.search_disparity(tmp_path)
    assert report["roots"] == 2
    exact = report["components"]["qol_years"]["reevaluated"]["share_roots_any_pair_beyond_2se"]
    assert exact["0"] == pytest.approx(0.5)
    assert exact["0.05"] == pytest.approx(0.5)
    assert "utility:balanced" in report["components"]
    assert any("Successor tensors" in note for note in report["notes"])


def _rank(value):
    return {"ranking": {"accuracy": value, "by_margin": []}, "within_group_r2": 0.2}


def test_verdict_maps_numbers_to_hypotheses() -> None:
    report = {
        "critic": {
            "calibration": {"qol_years": _rank(0.58), "population_years": _rank(0.9)},
            "fitting": {"qol_years": _rank(0.60)},
        },
        "noise_floor": {
            "models": {
                "qol_years/full": {
                    "critic": _rank(0.58),
                    "ridge_inputs/cal-cv": _rank(0.72),
                    "ridge_critic_trunk/fit->cal": _rank(0.60),
                },
                "qol_years/h1": {"ridge_inputs/cal-cv": _rank(0.70)},
            }
        },
    }
    v = qd.verdict(report)
    assert v["hypotheses"]["iii_critic_optimisation"]["status"] == "supported"
    assert v["hypotheses"]["ii_input_information"]["status"] == "not supported"
    assert v["hypotheses"]["iv_action_level_disparity"]["status"] == "inconclusive"
    report["noise_floor"]["models"]["qol_years/full"]["ridge_inputs/cal-cv"] = _rank(0.57)
    report["noise_floor"]["models"]["qol_years/h1"]["ridge_inputs/cal-cv"] = _rank(0.60)
    v = qd.verdict(report)
    assert v["hypotheses"]["ii_input_information"]["status"] == "supported"
    assert v["hypotheses"]["iii_critic_optimisation"]["status"] == "not supported"
    report["noise_floor"]["models"]["qol_years/h1"]["ridge_inputs/cal-cv"] = _rank(0.75)
    v = qd.verdict(report)
    assert v["hypotheses"]["i_noise_or_resolution"]["status"] == "supported"
    assert (
        v["hypotheses"]["i_noise_or_resolution"]["evidence"]["flags"]["short_horizon_better"]
        is True
    )
