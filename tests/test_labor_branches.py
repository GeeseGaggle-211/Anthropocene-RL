"""Paired-branch runner over the synthetic stand-in (no simulator binary is used).

The unmodified ``NativeSearch.extend`` runs; only the simulator primitives (fork/restore,
resolve, H12 labels, catalogs, dither) are stubbed in ``labor_branches_synthetic``.
"""

from __future__ import annotations

import numpy as np
import pytest

from anthropocene_rl.autonomous_v7 import labor_horizon_check as lhc
from anthropocene_rl.autonomous_v7.labor_search import Branch, NativeSearch
from anthropocene_rl.autonomous_v7.labor_branches import (
    KeyedActorContinuation,
    cumulative_returns,
    h12_continuation,
    paired_differences,
    run_paired_branches,
)
from anthropocene_rl.autonomous_v7.labor_branches_synthetic import SYNTH_FOCAL, SyntheticNative
from anthropocene_rl.autonomous_v7.labor_sa_advantage import parse_successors, successor_target

A = {"synthetic_vector": [0.5, -0.2, 0.1]}
B = {"synthetic_vector": [-0.4, 0.3, 0.6]}


def _pair(a=A, b=B, k=3, streams=(0,), continuation=None, seed=0, death=None):
    native = SyntheticNative(seed=seed, death_radius=death)
    fork = native.make_root(0)
    pair = run_paired_branches(
        native,
        root_fork=fork,
        root_h12=native.root_h12(),
        root_key=[0, seed, 0, SYNTH_FOCAL],
        focal=SYNTH_FOCAL,
        society=1,
        founded={SYNTH_FOCAL: 0},
        first_a=a,
        first_b=b,
        k=k,
        streams=streams,
        continuation=continuation,
    )
    return native, pair


def _bits(pair):
    out = []
    for name in ("a", "b"):
        for s in pair.streams[name]:
            out.append(np.asarray(s["step_rewards"]).tobytes())
    return out


def test_real_extend_runs_over_the_synthetic_primitives():
    assert type(SyntheticNative()).extend is NativeSearch.extend


def test_identical_actions_are_bitwise_identical_branches():
    _, pair = _pair(a=A, b=dict(A), streams=(0, 1, 2))
    assert pair.identical_actions
    for s in pair.stream_ids:
        sa, sb = pair.stream("a", s), pair.stream("b", s)
        assert np.asarray(sa["step_rewards"]).tobytes() == np.asarray(sb["step_rewards"]).tobytes()
        assert sa == sb  # includes boundary rows (deduplicated into one store)
    assert not np.any(paired_differences(pair))


def test_different_actions_diverge_and_dither_streams_differ():
    _, pair = _pair(streams=(0, 1))
    d = paired_differences(pair)
    assert d.shape == (2, 3, 4)
    assert np.abs(d[0, 0]).max() > 0
    s0, s1 = pair.stream("a", 0), pair.stream("a", 1)
    assert s0["step_rewards"] != s1["step_rewards"]  # dither acts on streams >= 1


def test_crn_noise_is_shared_between_branches():
    # The same exogenous noise hits both branches: with a == b the difference is zero
    # under any stream, which would fail if the branches drew different noise.
    for stream in (0, 3):
        _, pair = _pair(a=B, b=dict(B), streams=(stream,))
        assert not np.any(paired_differences(pair))


def test_determinism_across_reruns():
    _, p1 = _pair(streams=(0, 1), k=5)
    _, p2 = _pair(streams=(0, 1), k=5)
    assert _bits(p1) == _bits(p2)
    assert p1.streams["a"] == p2.streams["a"]
    assert [r["tensors"]["core"].tolist() for r in p1.store.rows] == [
        r["tensors"]["core"].tolist() for r in p2.store.rows
    ]


def test_rows_at_every_decision_and_parse_successors_round_trip():
    k = 6
    native, pair = _pair(k=k, streams=(0, 1))
    assert native.successor_every_decision is False  # restored
    parsed = parse_successors(pair.raw_root())
    assert sorted(parsed) == [0, 1]
    for streams in parsed.values():
        for st in streams:
            assert st.rewards.shape == (k, 4) and st.years.shape == (k,)
            assert sorted(st.row) == list(range(1, k + 1))  # a row at EVERY decision
            assert set(st.status.values()) == {"live"} and not st.terminated
            assert all(0 <= r < len(pair.store.rows) for r in st.row.values())
    # Rows carry the successor critic input (state + turn) and the root personality.
    st = parsed[0][0]
    row = pair.store.rows[st.row[2]]["tensors"]
    assert row["core"].shape == (1, 4) and row["labor_personality"].shape == (1, 2)
    # successor_target reads them at any decision k' (identity V for the check).
    v = {h: np.arange(len(pair.store.rows), dtype=float) for h in
         ("qol_years", "population_years", "feasibility", "settlement_guard")}
    for kk in (1, 3, k):
        assert successor_target(st, kk, v) is not None


def test_flag_off_keeps_search_export_decisions():
    # With the hook off, record_boundary exports rows only at config.successor_decisions.
    native = SyntheticNative()
    fork = native.make_root(0)
    br = Branch(0, 0, record=True)
    native.extend(br, 3, root_fork=fork, focal=SYNTH_FOCAL, candidate=A,
                  root_h12=native.root_h12(), root_key=[0], keep_fork=False)
    rows = {b["decision"]: b["row"] is not None for b in br.boundaries}
    assert rows == {1: True, 2: False, 3: False}


def test_k1_and_k20_with_early_termination():
    _, p1 = _pair(k=1)
    assert paired_differences(p1).shape == (1, 1, 4)
    _, p20 = _pair(k=20, streams=(0, 2))
    d = paired_differences(p20)
    assert d.shape == (2, 20, 4)
    for s in p20.streams["a"]:
        assert len(s["step_rewards"]) == 20 and len(s["boundaries"]) == 20
    # terminated branches close with zero future: cumulative returns are held constant
    _, pd = _pair(k=20, death=1.2)
    term = [s for s in pd.streams["a"] + pd.streams["b"] if s["terminated"]]
    assert term, "fixture should terminate at least one branch"
    s = term[0]
    n = len(s["step_rewards"])
    assert n < 20 and s["boundaries"][-1]["status"] == "terminated"
    cum = cumulative_returns(s, 20)
    assert np.array_equal(cum[n - 1], cum[19])
    parsed = parse_successors(pd.raw_root())
    assert any(st.terminated for sts in parsed.values() for st in sts)


def test_cumulative_matches_branch_returns_and_prefix_property():
    _, p20 = _pair(k=20)
    _, p5 = _pair(k=5)
    c20 = cumulative_returns(p20.stream("a", 0), 20)
    c5 = cumulative_returns(p5.stream("a", 0), 5)
    assert np.array_equal(c20[:5], c5)  # a longer run extends, never alters, a shorter one


def test_policy_is_applied_and_crn_safe():
    calls = []

    def act(decision, label, catalog, step, rng):
        calls.append(decision)
        return {"synthetic_vector": list(rng.normal(size=3) * 0.2)}

    pol = KeyedActorContinuation(act, [1, 2])
    _, base = _pair(continuation=h12_continuation, k=4)
    _, none = _pair(continuation=None, k=4)
    assert _bits(base) == _bits(none)  # H12 continuation == default
    _, p1 = _pair(continuation=pol, k=4)
    _, p2 = _pair(continuation=pol, k=4)
    assert _bits(p1) == _bits(p2) and _bits(p1) != _bits(base)
    assert sorted(set(calls)) == [1, 2, 3]  # decisions >= 1 only
    _, same = _pair(a=A, b=dict(A), continuation=pol, k=4)
    assert not np.any(paired_differences(same))  # identical actions stay identical


def test_validation_errors():
    native = SyntheticNative()
    fork = native.make_root(0)
    kw = dict(root_fork=fork, root_h12=native.root_h12(), root_key=[0], focal=SYNTH_FOCAL,
              society=1, founded={SYNTH_FOCAL: 0}, first_a=A, first_b=B)
    with pytest.raises(ValueError):
        run_paired_branches(native, k=0, **kw)
    with pytest.raises(ValueError):
        run_paired_branches(native, streams=(0, 0), **kw)
    native.successor_view = None
    with pytest.raises(ValueError):
        run_paired_branches(native, **kw)


def test_cost_estimator_and_refusals(capsys, tmp_path):
    est = lhc.cost_estimate(10, 20, 1)
    assert est["per_state"]["labor_decisions"] == 40
    assert est["per_state"]["native_resolves"] == 2 * 20 + 2 * 19
    assert est["total"]["native_resolves"] == 780
    assert lhc.main(["plan", "--states", "10", "--k", "20"]) == 0
    assert "total:" in capsys.readouterr().out
    states = tmp_path / "s.jsonl"
    states.write_text('{"world": 3010, "turn": 4, "camp": 1}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        lhc.main(["plan", "--states-file", str(states)])
    with pytest.raises(SystemExit):
        lhc.main(["execute", "--states-file", str(states)])
    ok = tmp_path / "ok.jsonl"
    ok.write_text('{"world": 90100, "turn": 4, "camp": 1}\n', encoding="utf-8")
    with pytest.raises(SystemExit):  # real worlds without approval
        lhc.main(["execute", "--states-file", str(ok)])


def test_synthetic_cli_writes_per_k_differences(tmp_path):
    out = tmp_path / "r.json"
    assert lhc.main(["synthetic", "--states", "3", "--k", "20", "--streams", "2",
                     "--out", str(out)]) == 0
    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["records"]) == 3
    diffs = np.asarray(data["records"][0]["cumulative_paired_difference"])
    assert diffs.shape == (2, 20, 4)
