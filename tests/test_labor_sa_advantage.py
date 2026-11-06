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


# ---------------------------------------------------------------------------
# Successor rows (search export v2)
# ---------------------------------------------------------------------------

G = math.exp(-20 / 400)


def _succ_stream(stream, rewards, rows, statuses=None):
    statuses = statuses or {d: "live" for d in range(1, len(rewards) + 1)}
    return {
        "stream": stream,
        "step_rewards": [list(r) for r in rewards],
        "step_years": [20.0] * len(rewards),
        "terminated": "terminated" in statuses.values(),
        "boundaries": [
            {"decision": d, "turn": 10 + d, "status": statuses[d], "row": rows.get(d)}
            for d in sorted(statuses)
        ],
    }


def toy_shard_v2(world=90154):
    """toy_shard plus successor rows. Successor row core = [qol V, population V, 0, 0]
    under the fake critic below (V reads core[0], core[1])."""
    shard = toy_shard(world)
    succ_core = []

    def row(qv, pv):
        succ_core.append([qv, pv, 0.0, 0.0])
        return len(succ_core) - 1

    for root in shard["roots"]:
        root["successor_decisions"] = [1, 3]
        for c in root["candidates"]:
            r1 = next(e["returns"] for e in c["evaluations"] if e["horizon"] == 1)
            streams = []
            for s in sorted({e["stream"] for e in c["evaluations"]}):
                # step rewards: r1 on every step (stream-independent), V(s1) = 2 * r1,
                # V(s3) = r1 + candidate index (population: 1000 * index)
                rows = {
                    1: row(2 * r1[0], 2 * r1[1]),
                    3: row(r1[0] + c["index"], 1000.0 * c["index"]),
                }
                if c["index"] == 3:  # H1-only candidate: one step, row at decision 1
                    streams.append(_succ_stream(s, [r1], {1: rows[1]}))
                else:
                    streams.append(_succ_stream(s, [r1, r1, r1], rows))
            c["successor_streams"] = streams
        root["successor_rows"] = None
    shard["successor_tensors"] = {"core": torch.tensor(succ_core, dtype=torch.float32)}
    shard["successor_privileged"] = None
    shard["successor_row_count"] = len(succ_core)
    shard["tensors"] = {"core": torch.zeros(2, 4)}  # V(s) = 0 at both roots
    return shard


def _fake_values(_paths, datasets):
    class M:
        path = "fake"
        sha256 = "0"

    class E:
        members = [M(), M()]

    members = []
    for scale in (1.0, 1.0):
        per = []
        for data in datasets:
            core = data["core"].double().numpy() if data else np.zeros((0, 4))
            per.append(
                {
                    "qol_years": scale * core[:, 0],
                    "population_years": scale * core[:, 1],
                    "feasibility": np.zeros(len(core)),
                    "settlement_guard": np.zeros(len(core)),
                }
            )
        members.append(per)
    return E(), members


def test_parse_successors_and_target_rules():
    shard = toy_shard_v2()
    succ = sa.parse_successors(shard["roots"][0], row_offset=100)
    assert sorted(succ) == [0, 1, 2, 3, 4]
    s0 = succ[1][0]
    assert s0.stream == 0 and s0.row[1] >= 100 and set(s0.row) == {1, 3}
    v = {h: np.arange(200, dtype=np.float64) for h in sa.COMPONENTS}
    t1 = sa.successor_target(s0, 1, v)
    assert t1[0] == pytest.approx(s0.rewards[0][0] + G * s0.row[1])
    t3 = sa.successor_target(s0, 3, v)
    assert t3[0] == pytest.approx(s0.rewards[0][0] * (1 + G + G * G) + G**3 * s0.row[3])
    assert sa.successor_target(succ[3][0], 3, v) is None  # H1-only: no decision-3 row
    dead = sa.SuccStream(
        0, np.ones((2, 4)), np.full(2, 20.0), True, {1: "live", 2: "terminated"}, {1: 0}
    )
    assert sa.successor_target(dead, 3, v)[0] == pytest.approx(1 + G)
    asleep = sa.SuccStream(0, np.ones((1, 4)), np.full(1, 20.0), False, {1: "inactive"}, {})
    assert sa.successor_target(asleep, 1, v) is None


def test_successor_section_full_advantage(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "_ensemble_values", _fake_values)
    payloads = tmp_path / "search" / "w" / "world-90154"
    payloads.mkdir(parents=True)
    torch.save(toy_shard_v2(), payloads / "advantage-candidates.pt")
    out = tmp_path / "r.json"
    sa.main(
        [
            "--payloads",
            str(tmp_path / "search"),
            "--out",
            str(out),
            "--arm",
            "p=x",
            "--margin",
            "1e-6",
        ]
    )
    report = json.loads(out.read_text())
    assert report["successor_observations_in_payload"] is True
    check = report["successor_check"]
    assert check["roots"] == 2 and check["status_counts"] == {"live": 62}  # 2 x (3 x 3 x 3 + 1 + 3)
    res = check["arms"]["p"]["results"]
    # k=1 stream 0: pred(a) = r1 + g * 2 r1 - 0 for survivors (H12, transfer, activity)
    k1 = res["k1/streams-0"]
    assert k1["rows"] == 6 and k1["dropped"] == 0
    # QoL: pred ranks as r1 does; realised H3 QoL: transfer > h12 > activity at both roots
    q = k1["metrics"]["qol_years"]["all"]
    assert q["pair_sign_accuracy"] == 1.0 and q["top1"] == 1.0
    # population: r1 identical -> k=1 prediction identical -> no sign skill (ties)
    p1 = k1["metrics"]["population_years"]["all"]
    assert p1["pair_sign_accuracy"] == 0.0
    # k=3: V(s3) population = 1000 * index -> activity (2) > transfer (1) > h12 (0), but
    # realised H3 population: transfer (310) > h12 (300) > activity (290): pairs with
    # activity are wrong, transfer vs h12 right
    p3 = res["k3/streams-0"]["metrics"]["population_years"]["all"]
    assert p3["pair_sign_accuracy"] == pytest.approx(1 / 3)
    flips = res["k3/streams-0"]["measured_sign_flips"]["with_controls"]["frozen"]
    assert flips["world_balanced"]["pairs"] > 0
    assert 0.0 <= flips["world_balanced"]["population_only_flip_rate"] <= 1.0
    # all-candidate predicted spread exists for every head
    allc = res["k1/all-candidates/stream-0"]
    assert allc["rows"] == 8  # twins excluded, 4 per root
    assert allc["predicted_within_root_sd"]["population_years"] == pytest.approx(0.0)
    # reeval mode uses streams 1, 2
    assert res["k1/streams-reeval"]["rows"] == 6


def test_successor_metrics_value_by_hand(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "_ensemble_values", _fake_values)
    shard = toy_shard_v2()
    roots = sa.parse_payload(shard)
    data = sa.SuccessorData()
    data.add(shard, roots)
    root_data, succ_data = data.datasets()
    _, members = _fake_values(None, [root_data, succ_data])
    rows = sa.successor_rows_table(data.roots, members[0][0], members[0][1], k=1, mode="0")
    k = (rows["root"] == "90154/0") & (rows["family"] == "transfer")
    r1 = 12.0  # base 10 + 2
    assert rows["pred"][k][0, 0] == pytest.approx(r1 + G * 2 * r1)
    assert rows["real"][k][0, 0] == pytest.approx(36.0 - 30.0)


def test_old_payload_output_has_no_successor_section(tmp_path):
    payloads = tmp_path / "search" / "w" / "world-90154"
    payloads.mkdir(parents=True)
    torch.save(toy_shard(), payloads / "advantage-candidates.pt")
    out = tmp_path / "r.json"
    sa.main(["--payloads", str(tmp_path / "search"), "--out", str(out)])
    report = json.loads(out.read_text())
    assert "successor_check" not in report
    assert report["successor_observations_in_payload"] is False
