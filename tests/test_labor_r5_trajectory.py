"""r5 trajectory diagnostics T1 (myopia), T3 (adaptation lag) and T4 (value tracking).

Synthetic fixtures with a known answer: a myopic A head that follows the first
decision on sign-conflict pairs, an adapted versus a lagging critic on
beat-the-teacher states, and value curves that separate at a known decision.
Plus the TD-error identity, the DAgger/continuation join, the receipt section
(never gating), the checkpoint comparison, the T4 recorder flag (off by
default) and an end-to-end fit_diagnostics run on a small critic.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_advantage as la
from anthropocene_rl.autonomous_v7 import labor_r5_acceptance as ra
from anthropocene_rl.autonomous_v7 import labor_r5_trajectory as rt

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

SCALES = rt._reference_scales()
GAMMA = math.exp(-20.0 / 400.0)


# --- T1: candidate pairs --------------------------------------------------------------------


def _contrib(first: float, delayed: float, length: int = 6) -> np.ndarray:
    """Per-step contributions (raw) with the QoL first part and the delayed part spread."""
    out = np.zeros((length, 4))
    out[0, 0] = first * SCALES["qol_years"]
    out[1:, 0] = delayed * SCALES["qol_years"] / (length - 1)
    return out


def _t1_fixture():
    """Roots of candidates whose first decision and delayed outcome disagree in sign."""
    roots, table = [], rt.StepTable()
    rng = np.random.default_rng(7)
    truth = []
    for r in range(12):
        parts = []
        candidates = []
        for c in range(4):
            first = float(rng.uniform(0.2, 1.0)) * (1 if c % 2 == 0 else -1)
            delayed = -2.0 * first + float(rng.normal(scale=0.05))  # delayed dominates, opposite
            parts.append((first, delayed))
            table.add(f"r{r}", c, _contrib(first, delayed), False, True)
            candidates.append(SimpleNamespace(index=c, evaluations={}))
        roots.append(
            SimpleNamespace(
                root_id=f"r{r}",
                world=90154 + r % 3,
                personality=(1.0, 0.0),
                candidates=candidates,
            )
        )
        truth.append(parts)
    return roots, table, truth


def _prediction(truth, which: str) -> dict:
    out = {name: [] for name in rt.COMPONENTS}
    for parts in truth:
        if which == "myopic":
            qol = np.array([f for f, _ in parts])
        else:
            qol = np.array([f + d for f, d in parts])
        out["qol_years"].append(qol)
        for name in rt.COMPONENTS[1:]:
            out[name].append(np.zeros(len(parts)))
    return out


def test_myopic_advantage_fails_sign_conflict_pairs_and_farsighted_passes() -> None:
    roots, table, truth = _t1_fixture()
    myopic = rt.t1_advantage_report(roots, _prediction(truth, "myopic"), table)
    farsighted = rt.t1_advantage_report(roots, _prediction(truth, "farsighted"), table)
    for report in (myopic, farsighted):
        for target in ("utility:root", "qol_years"):
            s = report["targets"][target]
            assert s["pairs"] == 12 * 6
            assert s["sign_conflict"]["pairs"] > 0
            assert s["sign_conflict"]["delayed_dominated_pairs"] == s["sign_conflict"]["pairs"]
    m = myopic["targets"]["qol_years"]
    f = farsighted["targets"]["qol_years"]
    assert m["sign_conflict"]["follows_first"] == 1.0
    assert m["sign_conflict"]["follows_delayed"] == 0.0
    assert m["sign_conflict"]["delayed_dominated_follows_total"] == 0.0
    assert m["sign_conflict"]["myopic"] is True
    assert m["correlation"]["first"] == pytest.approx(1.0)
    assert m["correlation"]["total"] < 0
    assert m["loadings"]["delayed_to_first_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert f["sign_conflict"]["follows_delayed"] == 1.0
    assert f["sign_conflict"]["myopic"] is False
    assert f["correlation"]["total"] == pytest.approx(1.0)
    assert f["loadings"]["first"] == pytest.approx(1.0)
    assert f["loadings"]["delayed_to_first_ratio"] == pytest.approx(1.0)
    assert f["delayed_share_of_abs_difference"] > 0.6
    # Root utility with personality (1, 0) is the QoL component here.
    assert myopic["targets"]["utility:root"]["sign_conflict"]["follows_first"] == 1.0
    # Population has no signal: nothing to rank.
    assert myopic["targets"]["population_years"]["sign_conflict"]["pairs"] == 0


def test_parts_split_first_two_to_five_and_beyond_at_the_common_horizon() -> None:
    contrib = np.arange(8 * 4, dtype=float).reshape(8, 4)
    parts = rt._parts(contrib, 7)
    assert np.allclose(parts["first"], contrib[0])
    assert np.allclose(parts["mid"], contrib[1:5].sum(axis=0))
    assert np.allclose(parts["late"], contrib[5:7].sum(axis=0))
    # A terminated branch earns zero afterwards; a living one bounds the common horizon.
    assert rt._common_length((np.zeros((3, 4)), True), (np.zeros((6, 4)), False)) == 6
    assert rt._common_length((np.zeros((3, 4)), False), (np.zeros((6, 4)), False)) == 3
    assert rt._common_length((np.zeros((3, 4)), True), (np.zeros((2, 4)), True)) == 3


def test_step_rewards_are_discounted_by_physical_years() -> None:
    contrib = rt.discounted_contributions([[1, 0, 0, 0]] * 3, [20.0, 40.0, 20.0])
    assert contrib[:, 0] == pytest.approx([1.0, math.exp(-20 / 400), math.exp(-60 / 400)])
    shard = {
        "discount_years": 400.0,
        "roots": [
            {
                "root_id": "a",
                "candidates": [
                    {"index": 0, "step_rewards": [[1, 0, 0, 0]], "step_years": [20.0]},
                    {"index": 1},  # no step extension: falls back
                    {
                        "index": 2,
                        "plan_role": "plan",
                        "step_rewards": [[5, 0, 0, 0]],
                        "step_years": [20.0],
                    },
                ],
            }
        ],
    }
    table = rt.step_table_from_shards([shard])
    assert len(table) == 1 and table.get("a", 0)[2] is True


def test_synthetic_payload_runs_through_the_cumulative_horizon_fallback(tmp_path) -> None:
    shard = la.synthetic_candidate_shard(worlds=(90154,), roots_per_world=3, candidates=5)
    path = tmp_path / "shard.pt"
    torch.save(shard, path)
    candidates = la.load_candidate_shards([path])
    table = rt.load_step_table([path])
    assert len(table) == 0  # the synthetic shard has no step extension
    oracle = {
        name: [
            np.array([c.evaluations[(3, 0)].returns[i] / SCALES[name] for c in root.candidates])
            for root in candidates.roots
        ]
        for i, name in enumerate(rt.COMPONENTS)
    }
    report = rt.t1_advantage_report(candidates.roots, oracle, table)
    s = report["targets"]["qol_years"]
    assert s["pairs"] > 0 and s["inexact_part_boundaries"] == 0
    assert s["correlation"]["total"] == pytest.approx(1.0)
    assert s["ranking_total"]["accuracy"] == 1.0


# --- T1: V's TD errors ------------------------------------------------------------------------


def _trajectory(n: int = 9, reward: float = 1.0):
    rewards = np.full(n, reward)
    terminal = np.zeros(n, dtype=bool)
    terminal[-1] = True
    exact = np.zeros(n)
    g = 0.0
    for t in reversed(range(n)):
        g = rewards[t] + (0.0 if terminal[t] else GAMMA * g)
        exact[t] = g
    return rewards, terminal, exact


def _split(values, rewards, terminal):
    n = len(values)
    return rt.td_error_split(
        values,
        rewards,
        np.full(n, 20.0),
        terminal,
        np.arange(n),
        np.full(n, 90180),
        np.ones(n),
        np.full(n, -1),
        zero_tail_declared=False,
    )


def test_td_split_is_zero_for_an_exact_critic_and_sums_to_the_monte_carlo_error() -> None:
    rewards, terminal, exact = _trajectory()
    split = _split(exact, rewards, terminal)
    for part in rt.PARTS:
        assert np.allclose(split["parts"][part], 0.0)
        assert split["available"][part].all()  # terminated: every part identified
    noisy = exact + np.random.default_rng(0).normal(size=len(exact))
    split = _split(noisy, rewards, terminal)
    total = sum(split["parts"][p] for p in rt.PARTS)
    assert np.allclose(total, exact - noisy)
    # A myopic V (immediate reward only) is one-step wrong by the discounted next
    # reward, and the delayed parts carry most of its error early in the trajectory.
    myopic = rewards.copy()
    split = _split(myopic, rewards, terminal)
    assert split["parts"]["first"][0] == pytest.approx(GAMMA * rewards[1])
    assert split["parts"]["mid"][0] + split["parts"]["late"][0] > split["parts"]["first"][0]


def test_td_split_living_end_without_a_tail_is_unavailable_not_zero() -> None:
    rewards, _, exact = _trajectory(n=8)
    living = np.zeros(8, dtype=bool)
    split = _split(exact, rewards, living)
    assert not split["available"]["late"].any()
    assert split["available"]["mid"][:2].all() and not split["available"]["mid"][3:].any()
    assert split["available"]["first"][:-1].all() and not split["available"]["first"][-1]


# --- T3 ---------------------------------------------------------------------------------------


def _rows_dataset(rows: list[dict]) -> dict:
    from test_labor_model import _labor_tensors

    n = len(rows)
    dataset = dict(_labor_tensors(batch=n))

    def col(key, dtype=torch.long, default=0):
        return torch.tensor([r.get(key, default) for r in rows], dtype=dtype)

    dataset.update(
        row_world=col("world"),
        row_camp=col("camp"),
        row_turn=col("turn"),
        row_source=col("source"),
        row_round=col("round"),
        row_start_index=col("start", default=-1),
        row_branch_step=col("step", default=-1),
        row_trajectory=col("trajectory", default=-1),
        row_quarantined=torch.zeros(n, dtype=torch.bool),
        transition_elapsed_years=torch.full((n,), 20.0, dtype=torch.float64),
        transition_terminal=col("terminal", torch.bool, True),
    )
    for name in rt.COMPONENTS:
        dataset["reward_" + name] = col(name, torch.float64, 0.0)
    dataset["collection"] = {"return_tail": "truncated-zero-at-collection-horizon"}
    return dataset


def _t3_sets(lag: float):
    """Teacher rows, actor-visited continuation rows (half of them beat the teacher)."""
    q = SCALES["qol_years"]
    rows, beat = [], {}
    for k in range(8):
        rows.append(
            {"world": 90180 + k % 4, "camp": 10 + k, "turn": k, "source": 0, "qol_years": q}
        )
    for k in range(8):
        key = dict(
            world=90100 + k % 4, camp=20 + k, turn=k, source=3, round=1 + k % 2, start=k, step=1
        )
        rows.append({**key, "trajectory": k * 2 + 1, "qol_years": 2.0 * q})
        beat[(key["world"], key["camp"], key["round"], k, 1)] = 1.0 if k % 2 == 0 else -1.0
    dataset = _rows_dataset(rows)
    target = dataset["reward_qol_years"].numpy().copy()
    pred = target.copy()
    is_beat = np.array([False] * 8 + [k % 2 == 0 for k in range(8)])
    pred[is_beat] += lag * q  # a lagging critic stays wrong where the actor improved
    predictions = {name: np.zeros(len(rows)) for name in rt.COMPONENTS}
    predictions["qol_years"] = pred
    return [("synthetic", dataset, predictions)], beat


def test_lagging_critic_errs_on_beat_the_teacher_states_and_adapted_does_not() -> None:
    sets, beat = _t3_sets(0.0)
    adapted = rt.v_adaptation_report(sets, beat=beat)
    sets, beat = _t3_sets(0.5)
    lagging = rt.v_adaptation_report(sets, beat=beat)
    a, g = adapted["heads"]["qol_years"], lagging["heads"]["qol_years"]
    assert a["groups"]["teacher"]["rows"] == 8 and a["groups"]["beat"]["rows"] == 4
    assert a["groups"]["actor"]["rows"] == 8
    assert a["beat_minus_teacher_nmae"] == pytest.approx(0.0, abs=1e-6)
    assert g["groups"]["teacher"]["nmae"] == pytest.approx(0.0, abs=1e-6)
    assert g["groups"]["beat"]["nmae"] == pytest.approx(0.5, abs=1e-6)
    assert g["groups"]["beat"]["bias"] == pytest.approx(0.5, abs=1e-6)
    assert g["beat_minus_teacher_nmae"] == pytest.approx(0.5, abs=1e-6)
    assert g["actor_minus_teacher_nmae"] == pytest.approx(0.25, abs=1e-6)
    assert set(g["groups"]) >= {"actor:round-1", "actor:round-2", "beat:round-1"}
    assert lagging["beat_branches_above_margin"] == 4
    # Checkpoint comparison flags the lag; the series marks the head lagging.
    base = {"arm": "p", "seed": 0, "trajectory_diagnostics": {"t3": {"value": adapted}}}
    worse = {"arm": "p", "seed": 0, "trajectory_diagnostics": {"t3": {"value": lagging}}}
    comparison = rt.compare(base, worse)
    flagged = comparison["arms"]["p"]["flagged"]
    assert "t3.v.qol_years.beat_minus_teacher_nmae" in flagged
    assert "t3.v.qol_years.teacher.nmae" not in flagged
    assert rt.compare(worse, base)["flagged_count"] == 0
    series = rt.adaptation_series([("refit-1", worse), ("refit-2", worse)])
    assert series["heads"]["qol_years"]["lagging"] is True
    assert (
        rt.adaptation_series([("refit-1", worse), ("refit-2", base)])["heads"]["qol_years"][
            "lagging"
        ]
        is False
    )


def test_branch_join_marks_actor_branches_that_beat_the_teacher() -> None:
    q = SCALES["qol_years"]
    rows = [
        # Teacher trajectory from turn 0 (its own return is 1 q).
        {"world": 90100, "camp": 1, "turn": 0, "source": 0, "qol_years": 1.0 * q},
        # DAgger actor decisions at the start (turn 0) and the next decision.
        {
            "world": 90100,
            "camp": 1,
            "turn": 0,
            "source": 1,
            "round": 1,
            "start": 0,
            "step": 0,
            "qol_years": 2.0 * q,
            "terminal": False,
        },
        {
            "world": 90100,
            "camp": 1,
            "turn": 1,
            "source": 1,
            "round": 1,
            "start": 0,
            "step": 1,
            "qol_years": 0.0,
            "terminal": False,
        },
        # Teacher continuation after one and after two actor decisions.
        {
            "world": 90100,
            "camp": 1,
            "turn": 1,
            "source": 3,
            "round": 1,
            "start": 0,
            "step": 1,
            "trajectory": 1,
            "qol_years": 0.5 * q,
        },
        {
            "world": 90100,
            "camp": 1,
            "turn": 2,
            "source": 3,
            "round": 1,
            "start": 0,
            "step": 2,
            "trajectory": 2,
            "qol_years": -6.0 * q,
        },
    ]
    keys = rt.branch_beat_keys([_rows_dataset(rows)])
    # Utility weights 0.5 QoL by default: (2 + g * 0.5 - 1) * 0.5 and (2 + g^2 (-6) - 1) * 0.5.
    assert keys[(90100, 1, 1, 0, 1)] == pytest.approx(0.5 * (1.0 + GAMMA * 0.5))
    assert keys[(90100, 1, 1, 0, 2)] == pytest.approx(0.5 * (1.0 - 6.0 * GAMMA**2))


def test_advantage_adaptation_by_root_source_and_search_beats() -> None:
    shard = la.synthetic_candidate_shard(worlds=(90154, 90155), roots_per_world=4, candidates=5)
    for r, root in enumerate(shard["roots"]):
        root["source"] = "actor" if r % 2 else "h12"
    candidates = la.parse_candidate_shard(shard)
    roots = candidates.roots

    def oracle(sign_for_root):
        return {
            name: [
                sign_for_root(root)
                * np.array(
                    [c.evaluations[(3, 0)].returns[i] / SCALES[name] for c in root.candidates]
                )
                for root in roots
            ]
            for i, name in enumerate(rt.COMPONENTS)
        }

    beat = rt.search_beat_roots(roots)
    assert beat  # the planted delayed-benefit candidates beat H12
    good = rt.a_adaptation_report(roots, oracle(lambda root: 1.0))
    lag = rt.a_adaptation_report(
        roots, oracle(lambda root: -1.0 if roots.index(root) in beat else 1.0)
    )
    assert good["groups"]["beat"]["beat_candidate_above_h12"] == 1.0
    assert good["groups"]["beat"]["utility:root"]["accuracy"] > 0.95
    assert lag["groups"]["beat"]["beat_candidate_above_h12"] == 0.0
    assert lag["groups"]["beat"]["utility:root"]["accuracy"] < 0.05
    assert good["groups"]["teacher"]["roots"] + good["groups"]["actor"]["roots"] == len(roots)


# --- T4 ---------------------------------------------------------------------------------------


def _trace(offset_from: int | None, decisions: int = 20, offset: float = 0.5, world: int = 2000):
    """One camp with constant reward; exact value r/(1-g); optional offset from a decision."""
    r = {name: 1.0 * SCALES[name] for name in rt.COMPONENTS}
    exact = {name: r[name] / (1.0 - GAMMA) for name in rt.COMPONENTS}
    out = []
    for d in range(decisions):
        shift = offset if offset_from is not None and d >= offset_from else 0.0
        camps = [
            {
                "camp": 1,
                "society": 1,
                "parent_society": 0,
                "farmer_unlocked": d >= 12,
                "values": {n: exact[n] + shift * SCALES[n] for n in rt.COMPONENTS},
                "rewards": dict(r),
                "elapsed": 20.0,
                "terminal": False,
            }
        ]
        if d >= 10:  # a daughter of society 1 appears at decision 10
            camps.append(
                {
                    "camp": 2,
                    "society": 2,
                    "parent_society": 1,
                    "farmer_unlocked": False,
                    "values": dict(exact),
                    "rewards": dict(r),
                    "elapsed": 20.0,
                    "terminal": False,
                }
            )
        out.append({"decision": d, "turn": d, "camps": camps})
    return {
        "schema": rt.TRACE_SCHEMA,
        "world": world,
        "mode": "deterministic",
        "critic_digest": "c",
        "decisions": out,
        "final": {
            "turn": decisions,
            "camps": [{"camp": c, "values": dict(exact)} for c in (1, 2)],
        },
    }


def test_value_curves_separate_at_the_known_decision() -> None:
    report = rt.t4_report([_trace(12), _trace(12, world=2001)])
    head = report["heads"]["qol_years"]
    assert head["separation_decision"] == 12
    assert set(head["per_world_separation"].values()) == {12}
    points = {p["decision"]: p for p in head["curve"]}
    assert points[5]["gap"] == pytest.approx(0.0, abs=1e-9)
    # Two camps from decision 12: the offset camp and the exact daughter average to 0.25.
    assert points[15]["gap"] == pytest.approx(0.25, abs=1e-6)
    assert points[3]["realised_truncated"] < points[3]["realised_closed"]
    stages = head["stages"]
    assert stages["early"]["mae"] == pytest.approx(0.0, abs=1e-9)
    assert stages["late"]["mae"] > 0.2
    assert stages["unlock"]["rows"] > 0 and stages["unlock"]["mae"] > 0
    assert stages["fission"]["rows"] > 0
    assert rt.t4_report([_trace(None)])["heads"]["qol_years"]["separation_decision"] is None


def test_terminal_camps_close_their_return_with_zero() -> None:
    trace = _trace(None, decisions=4)
    for decision in trace["decisions"]:
        decision["camps"] = decision["camps"][:1]
    trace["decisions"][-1]["camps"][0]["terminal"] = True
    trace["final"]["camps"] = []
    rows = rt._trace_rows(trace)
    last = next(r for r in rows if r["decision"] == 3)
    assert last["closed"][0] == pytest.approx(SCALES["qol_years"])
    assert last["truncation_weight"] == 0.0


def test_recorder_flag_is_off_by_default_and_traces_feed_t4() -> None:
    from anthropocene_rl.autonomous_v7 import labor_pretrain_eval as lpe

    parser = lpe.argument_parser()
    args = parser.parse_args(
        ["autonomous", "--checkpoint", "x.pt", "--worlds", "2000", "--out", "o"]
    )
    assert args.record_critic_trace is False
    args = parser.parse_args(
        [
            "autonomous",
            "--checkpoint",
            "x.pt",
            "--worlds",
            "2000",
            "--out",
            "o",
            "--record-critic-trace",
        ]
    )
    assert args.record_critic_trace is True
    report = {"runs": [{"world": 2000, "mode": "deterministic", "critic_trace": _trace(12)}]}
    traces = rt.traces_from_reports([report])
    assert len(traces) == 1 and traces[0]["world"] == 2000
    assert rt.t4_report(traces)["heads"]["qol_years"]["separation_decision"] == 12


def test_recorder_logs_values_rewards_and_the_final_state(monkeypatch) -> None:
    from anthropocene_rl.autonomous_v7 import labor_training

    def fake_rewards(step, camps, founded, *, world_index, decision_turn):
        for v in step["villages"]:
            founded.setdefault(int(v["camp"]), int(step["turn"]))
        return (
            {c: {"qol_years": 3.0, "population_years": 4.0} for c in camps},
            {c: False for c in camps},
            {c: 20.0 for c in camps},
        )

    monkeypatch.setattr(labor_training, "_transition_rewards", fake_rewards)
    recorder = rt.CriticTraceRecorder(SimpleNamespace(privileged_layout=None), SimpleNamespace())
    monkeypatch.setattr(
        recorder,
        "predict",
        lambda live, tensors: {n: np.full(len(live), 7.0) for n in rt.COMPONENTS},
    )
    village = {"camp": 5, "society": 2, "parentSociety": 0, "population": 30, "terminated": False}
    nxt = {"turn": 1, "villages": [dict(village, population=31)]}
    execution = SimpleNamespace(
        step=nxt, initial_attempts={5: SimpleNamespace(feasibility_cost=0.25)}
    )
    live = [village]
    recorder.record(
        step={"turn": 0},
        live=live,
        tensors={},
        values=recorder.predict(live, {}),
        execution=execution,
    )
    recorder.finish(nxt, nxt["villages"], {})
    trace = recorder.trace()
    camp = trace["decisions"][0]["camps"][0]
    assert camp["values"]["qol_years"] == 7.0 and camp["rewards"]["qol_years"] == 3.0
    assert camp["rewards"]["feasibility"] == 0.25 and camp["farmer_unlocked"] is None
    assert trace["final"]["camps"][0]["values"]["population_years"] == 7.0
    assert trace["schema"] == rt.TRACE_SCHEMA


# --- receipt ----------------------------------------------------------------------------------


def _diagnostics(lag: float, separation_from: int | None) -> dict:
    sets, beat = _t3_sets(lag)
    roots, table, truth = _t1_fixture()
    return {
        "schema": rt.SCHEMA,
        "t1": {"advantage": rt.t1_advantage_report(roots, _prediction(truth, "farsighted"), table)},
        "t3": {"value": rt.v_adaptation_report(sets, beat=beat)},
        "t4": rt.t4_report([_trace(separation_from)]),
    }


def test_receipt_reports_trajectory_diagnostics_beside_an_unchanged_verdict() -> None:
    from test_labor_r5_acceptance import _evidence_set, _receipt

    plain = _receipt(_evidence_set())
    diag = _diagnostics(0.0, None)

    def attach(evidence, value):
        for record in evidence:
            record["trajectory_diagnostics"] = value(record["seed"])

    good = _evidence_set(mutate=lambda e: attach(e, lambda s: _diagnostics(0.01 * s, None)))
    receipt = _receipt(good)
    section = receipt["trajectory_diagnostics"]
    assert receipt["machine_verdict"] == plain["machine_verdict"]
    assert section["gating"] is False and section["schema"] == rt.SCHEMA
    summary = section["arms"]["privileged-e4"]["summary"]
    key = "t3.v.qol_years.beat_minus_teacher_nmae"
    assert summary[key]["seeds"] == 5
    assert summary[key]["spread"] == pytest.approx(0.04)
    assert summary["t4.qol_years.separation_decision"]["mean"] == 20  # never separated
    assert plain["trajectory_diagnostics"]["arms"]["privileged-e4"]["missing_seeds"] == [
        0,
        1,
        2,
        3,
        4,
    ]
    # A later checkpoint that lags and separates early is flagged against the baseline.
    bad = _evidence_set(mutate=lambda e: attach(e, lambda s: _diagnostics(0.5 + 0.01 * s, 12)))
    later = ra.build_receipt(
        bad,
        arms=ra.DEFAULT_ARMS,
        seeds=[0, 1, 2, 3, 4],
        seed_rule="all-seeds",
        required_seeds=5,
        determinism=None,
        configuration={},
        attempt_id="t-2",
        source={"commit": "t"},
        trajectory_baseline=receipt,
    )
    flagged = later["trajectory_diagnostics"]["comparison"]["arms"]["privileged-e4"]["flagged"]
    assert key in flagged and "t4.qol_years.separation_decision" in flagged
    assert "t1.a.qol_years.conflict_follows_delayed" not in flagged
    assert rt.compare(receipt, later)["flagged_count"] > 0
    assert ra.summarize(later).count("non-gating") == 1
    assert diag["t4"]["heads"]["qol_years"]["separation_decision"] is None
    # A baseline receipt from before section 4a records an error, never blocks the receipt.
    old = {k: v for k, v in receipt.items() if k != "trajectory_diagnostics"}
    section = rt.receipt_section(good, arms=["privileged-e4"], seeds=[0], baseline=old)
    assert "error" in section["comparison"] and section["comparison"]["flagged_count"] == 0


def test_compare_cli_and_labelled_payload_arguments(tmp_path) -> None:
    base = {"arm": "p", "seed": 0, "trajectory_diagnostics": _diagnostics(0.0, None)}
    cand = {"arm": "p", "seed": 0, "trajectory_diagnostics": _diagnostics(0.5, 12)}
    (tmp_path / "b.json").write_text(json.dumps(base, default=float))
    (tmp_path / "c.json").write_text(json.dumps(cand, default=float))
    out = tmp_path / "cmp.json"
    assert (
        rt.main(
            [
                "compare",
                "--baseline",
                str(tmp_path / "b.json"),
                "--candidate",
                str(tmp_path / "c.json"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    result = json.loads(out.read_text())
    assert result["gating"] is False and result["flagged_count"] >= 2
    labelled = ra._labelled_paths(["round-1=a.pt,b.pt", "round-2=c.pt", "round-1=d.pt"])
    assert labelled == [
        ("round-1", [Path("a.pt"), Path("b.pt"), Path("d.pt")]),
        ("round-2", [Path("c.pt")]),
    ]
    with pytest.raises(ValueError):
        ra._labelled_paths(["nolabel"])
    args = ra.argument_parser().parse_args(
        [
            "receipt",
            "--evidence",
            "e.json",
            "--seeds",
            "0",
            "--attempt-id",
            "a",
            "--out",
            "o",
            "--trajectory-baseline",
            "b.json",
        ]
    )
    assert args.trajectory_baseline == Path("b.json")


# --- end to end on a small critic -------------------------------------------------------------


def test_fit_diagnostics_end_to_end_on_a_small_critic(tmp_path) -> None:
    from test_labor_privileged_model import _critics, _layout
    from test_labor_qol_head import _training_dataset

    task, costs = _critics(_layout(False))
    head = la.LaborAdvantageHead.for_critics(task, costs, embedding=16, hidden=16)
    shard = la.synthetic_candidate_shard(worlds=(90154,), roots_per_world=3, candidates=5)
    for root in shard["roots"]:
        for cand in root["candidates"]:
            ev = next(e for e in cand["evaluations"] if e["horizon"] == 1 and e["stream"] == 0)
            cand["step_rewards"] = [ev["returns"]] * 6
            cand["step_years"] = [20.0] * 6
            cand["terminated"] = False
    path = tmp_path / "reserved.pt"
    torch.save(shard, path)
    reserved = la.load_candidate_shards([path])
    report = rt.fit_diagnostics(
        task=task,
        costs=costs,
        head=head,
        calibration_dataset=_training_dataset(),
        reserved=reserved,
        reserved_paths=[path],
        payload_sets=[("round-1", reserved)],
        traces=[_trace(12)],
        critic_digest="c",
    )
    assert report["gating"] is False
    t1a = report["t1"]["advantage"]["targets"]["qol_years"]
    assert t1a["pairs"] > 0 and t1a["skipped_pairs_without_steps"] == 0
    assert set(report["t1"]["value"]["heads"]) == set(rt.COMPONENTS)
    assert report["t1"]["value"]["heads"]["qol_years"]["first"]["rows"] == 6
    assert report["t3"]["value"]["heads"]["qol_years"]["groups"]["teacher"]["rows"] == 6
    assert set(report["t3"]["advantage"]) == {"reserved", "round-1"}
    assert report["t4"]["critic_digest_matches"] is True
    assert rt.headline_metrics(report)
    json.dumps(report, default=float)
