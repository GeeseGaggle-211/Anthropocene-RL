"""Toy-payload tests for the per-(state, action) advantage-error check."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_sa_advantage as sa
from anthropocene_rl.autonomous_v7.labor_bridge import ACTIVITIES

FIELD_OP = ACTIVITIES.index("field.operation")


def _eval(h, s, ret):
    return {
        "horizon": h,
        "reached_horizon": h,
        "terminated": False,
        "stream": s,
        "returns": list(ret),
        "step_years": [20.0] * h,
        "se": None,
    }


def _cand(i, family, r1, g3=None, field=0.0, is_h12=False):
    wy = np.zeros((13, len(ACTIVITIES)))
    wy[0, 1] = 100.0 * (1 - field)
    wy[1, FIELD_OP] = 100.0 * field
    evals = [_eval(1, 0, r1)]
    for s, ret in (g3 or {}).items():
        evals.append(_eval(3, s, ret))
    return {
        "index": i,
        "family": family,
        "is_h12": is_h12,
        "control": None,
        "bookkeeping_twin": None,
        "evaluations": evals,
        "compiled": {"worker_years": wy.tolist()},
    }


def toy_shard(world=90154):
    roots = []
    for r in range(2):
        base = 10.0 * (r + 1)
        cands = [
            _cand(
                0,
                "h12",
                [base, 100, 0, 0],
                {0: [base * 3, 300, 0, 0], 1: [base * 3, 300, 0, 0], 2: [base * 3 + 1, 300, 0, 0]},
                0.1,
                True,
            ),
            _cand(
                1,
                "transfer",
                [base + 2, 100, 0, 0],
                {
                    0: [base * 3 + 6, 310, 0, 0],
                    1: [base * 3 + 6, 310, 0, 0],
                    2: [base * 3 + 5, 310, 0, 0],
                },
                0.6,
            ),
            _cand(
                2,
                "activity",
                [base - 1, 100, 0, 0],
                {
                    0: [base * 3 - 4, 290, 0, 0],
                    1: [base * 3 - 4, 290, 0, 0],
                    2: [base * 3 - 2, 290, 0, 0],
                },
                0.0,
            ),
            _cand(3, "movement", [base - 5, 100, 0, 0]),
            _cand(4, "control-twin", [base, 100, 0, 0], {0: [base * 3, 300, 0, 0]}),
        ]
        cands[4]["bookkeeping_twin"] = 0
        roots.append(
            {
                "root_id": f"{world}/{r}",
                "world": world,
                "turn": 10 + 20 * r,
                "camp": 5,
                "tensor_row": r,
                "control_root": False,
                "final_horizon": 3,
                "candidates": cands,
            }
        )
    return {
        "component_order": list(sa.COMPONENTS),
        "discount_years": 400.0,
        "activity_ids": list(ACTIVITIES),
        "tensors": {"core": torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)},
        "roots": roots,
    }


def test_parse_skips_plans_and_marks_twins():
    shard = toy_shard()
    shard["roots"][0]["candidates"].append(dict(_cand(5, "plan", [0, 0, 0, 0]), plan_role="plan"))
    roots = sa.parse_payload(shard)
    assert len(roots) == 2 and len(roots[0].cands) == 5
    assert [c.twin for c in roots[0].cands] == [False, False, False, False, True]
    assert roots[0].h12.r1[0] == 10.0
    assert math.isclose(roots[0].cands[1].field_share, 0.6)


def test_realised_rows_baselines():
    roots = sa.parse_payload(toy_shard())
    h1 = sa.realised_rows(roots, horizon=1, baseline="h12")
    assert len(h1["root"]) == 8  # twin excluded, 4 per root
    assert h1["real"][h1["is_h12"]][:, 0].tolist() == [0.0, 0.0]
    np.testing.assert_allclose(h1["real"], h1["reward_only"])
    h3 = sa.realised_rows(roots, horizon=3, baseline="h12", streams="reeval")
    assert len(h3["root"]) == 6  # survivors only
    # transfer at root 0: mean(36,35) - mean(30,31) = 5
    k = (h3["root"] == "90154/0") & (h3["family"] == "transfer")
    assert h3["real"][k][0, 0] == pytest.approx(5.0)
    mean = sa.realised_rows(roots, horizon=3, baseline="mean", streams="0")
    for gid in np.unique(mean["root"]):
        assert abs(mean["real"][mean["root"] == gid].sum(axis=0)).max() < 1e-9


def test_within_root_metrics_perfect_and_reversed():
    real = np.array([0.0, 1.0, -1.0, 0.0, 2.0, -2.0])
    group = np.array(["a"] * 3 + ["b"] * 3)
    worlds = np.array([1, 1, 1, 2, 2, 2])
    good = sa.within_root_metrics(real * 2, real, group, worlds, margin=0.1)["world_balanced"]
    assert good["corr"] == pytest.approx(1.0) and good["pair_sign_accuracy"] == 1.0
    assert good["top1"] == 1.0 and good["slope"] == pytest.approx(0.5)
    bad = sa.within_root_metrics(-real, real, group, worlds, margin=0.1)["world_balanced"]
    assert bad["pair_sign_accuracy"] == 0.0 and bad["spearman"] == pytest.approx(-1.0)


def test_flip_rates():
    diffs = np.array([1.0, -1.0, 0.5])
    assert sa.flip_rate_empirical(diffs, np.zeros(10)) == 0.0
    big = np.linspace(-1000, 1000, 101)
    assert sa.flip_rate_empirical(diffs, big) == pytest.approx(0.5, abs=0.02)
    assert sa.flip_rate_gaussian(np.array([0.0]), 1.0) is None
    assert sa.flip_rate_gaussian(np.array([1e-9]), 1.0) == pytest.approx(0.5, abs=1e-6)


def test_event_flags():
    core = np.zeros((6, 20))
    assert sa.event_flags(core[:5]) is None
    assert sa.event_flags(core) == {
        "fission": False,
        "emigration": False,
        "immigration": False,
        "event": False,
    }
    core[:3, 12] = 1.0  # drop of 1.0 between rows 2 and 3
    core[4, 13] = 0.05
    flags = sa.event_flags(core)
    assert flags["fission"] and flags["immigration"] and not flags["emigration"] and flags["event"]


def test_nstep_targets_rules():
    g = math.exp(-20 / 400)
    v = np.array([5.0, 4.0, 3.0, 2.0])
    r = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.full(4, 20.0)
    no = np.zeros(4, dtype=bool)
    tail = np.full(4, 7.0)
    ok = np.ones(4, dtype=bool)
    assert sa.nstep_targets(v, r, y, no, tail, ok, 1) == pytest.approx(1 + g * 4)
    assert sa.nstep_targets(v, r, y, no, tail, ok, 3) == pytest.approx(1 + g + g * g + g**3 * 2)
    term = no.copy()
    term[1] = True
    assert sa.nstep_targets(v, r, y, term, tail, ok, 3) == pytest.approx(1 + g)
    # living end at row 1 uses its stored tail; without a tail it is unavailable
    assert sa.nstep_targets(v[:2], r[:2], y[:2], no[:2], tail[:2], ok[:2], 3) == pytest.approx(
        1 + g + g * g * 7
    )
    assert sa.nstep_targets(v[:2], r[:2], y[:2], no[:2], tail[:2], ~ok[:2], 3) is None


def test_combine_matches_ppo_and_rescales():
    values = {
        "qol_years": np.array([43.152]),
        "population_years": np.array([85133.98]),
        "feasibility": np.array([0.25]),
        "settlement_guard": np.array([0.0]),
    }
    assert sa.combine(values, sa.FROZEN_SCALES)[0] == pytest.approx(0.5 + 0.5 - 0.25)
    r5 = {"qol_years": 2 * 43.152, "population_years": 85133.98}
    assert sa.combine(values, r5)[0] == pytest.approx(0.25 + 0.5 - 0.25)


def test_noise_and_field_separation():
    roots = sa.parse_payload(toy_shard())
    noise = sa.stream_noise(roots)
    assert noise["pairs"] == 4
    assert noise["per_head"]["population_years"]["identical_streams_fraction"] == 1.0
    rows = sa.realised_rows(roots, horizon=3, baseline="mean", streams="reeval")
    sep = sa.field_separation(rows, rows["real"][:, 0])
    assert sep["roots"] == 2 and sep["corr"] > 0.9 and sep["heavy_minus_light"] > 0


def test_run_without_trajectory_is_deterministic(tmp_path):
    payloads = tmp_path / "search" / "w" / "world-90154"
    payloads.mkdir(parents=True)
    torch.save(toy_shard(), payloads / "advantage-candidates.pt")
    outs = []
    for name in ("a.json", "b.json"):
        sa.main(["--payloads", str(tmp_path / "search"), "--out", str(tmp_path / name)])
        outs.append((tmp_path / name).read_text())
    assert outs[0] == outs[1]
    report = json.loads(outs[0])
    assert report["roots"] == 2 and report["successor_observations_in_payload"] is False
    assert "critic" not in report
    floor = report["reward_only_floor"]["metrics"]["H3/h12/reeval"]["qol_years"]["all"]
    assert floor["pair_sign_accuracy"] == 1.0
