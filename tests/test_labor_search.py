from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7.labor_bc import (
    HeuristicLaborLabel,
    _label_tensors,
    convert_heuristic_labor_label,
)
from anthropocene_rl.autonomous_v7.labor_bridge import (
    ACTIVITIES,
    SPECIALTIES,
    LaborCatalog,
    labor_catalog_digest,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    LABOR_OBJECTIVE_REFERENCE_SCALES,
)
from anthropocene_rl.autonomous_v7.labor_search import (
    COMPONENTS,
    PERTURBATION_FAMILIES,
    Candidate,
    SearchConfig,
    candidate_label,
    compose_payload,
    conserves_labor,
    dither_payload,
    focal_order,
    halving_order,
    labor_state,
    paired_acceptance,
    physical_digest,
    root_turns,
    round_robin,
    select_survivors,
    structured_candidates,
    trade_opportunities,
    utility,
    utility_matrix,
    validate_search_worlds,
)
from tests.test_labor_bridge import _catalog_payload

S = {name: i for i, name in enumerate(SPECIALTIES)}
A = {name: i for i, name in enumerate(ACTIVITIES)}


def _catalog(population: int = 100) -> LaborCatalog:
    raw = _catalog_payload()
    raw["population"] = population
    raw["turnYears"] = 20.0
    raw["digest"] = labor_catalog_digest(raw)
    return LaborCatalog.from_payload(raw)


def _observation(trade_groups: tuple[int, ...] = (1, 3)) -> dict:
    sets = []
    present = []
    for _, rows, width in contracts.SETS:
        sets.append([0.0] * (rows * width))
        present.append([False] * rows)
    trade = [name for name, _, _ in contracts.SETS].index("trade")
    width = contracts.SETS[trade][2]
    for group in trade_groups:
        present[trade][group] = True
        sets[trade][group * width] = 3.0  # held stock
    present[trade][5] = True  # present but no stock or activity: not an opportunity
    return {
        "movementMask": [True] * 8,
        "regionMovementMask": [True, False] + [True] * contracts.REGION_ROWS,
        "sets": sets,
        "rowPresent": present,
    }


def _base_payload(catalog: LaborCatalog) -> dict:
    counts = np.zeros(len(SPECIALTIES), dtype=np.int64)
    counts[S["Gatherer"]] = 80
    counts[S["Farmer"]] = 10
    counts[S["Craftsperson"]] = 10
    activity = np.zeros((len(SPECIALTIES) - 1, len(ACTIVITIES)))
    activity[:, 0] = 1.0
    activity[S["Gatherer"] - 1] = 0.0
    activity[S["Gatherer"] - 1, A["food.gather"]] = 0.9
    activity[S["Gatherer"] - 1, A["stock.wood"]] = 0.1
    activity[S["Farmer"] - 1] = 0.0
    activity[S["Farmer"] - 1, A["field.operation"]] = 0.7
    activity[S["Farmer"] - 1, A["field.expansion"]] = 0.2
    activity[S["Farmer"] - 1, 0] = 0.1
    activity[S["Craftsperson"] - 1] = 0.0
    activity[S["Craftsperson"] - 1, A["goods.housing"]] = 0.6
    activity[S["Craftsperson"] - 1, A["goods.cloth"]] = 0.4
    seed = {
        "movement": -1,
        "fission": 0,
        "tradeValue": [1.0] * len(contracts.TRADE_GROUPS),
        "tradeGate": [2] * len(contracts.TRADE_GROUPS),
        "tradeShare": 0.5,
        "laborSpecialtyAllocation": [float(c) / 100 for c in counts],
        "laborActivityAllocation": activity.tolist(),
    }
    return compose_payload(seed, catalog, counts=counts, activity=activity)


def _h12_label(catalog: LaborCatalog, payload: dict, *, trade_valid: bool) -> HeuristicLaborLabel:
    counts, activity = labor_state(payload, catalog)
    groups = len(contracts.TRADE_GROUPS)
    return HeuristicLaborLabel(
        specialty=counts / 100.0,
        activity=activity,
        action=payload,
        specialty_counts=counts,
        planned_worker_years=counts[1:, None] * 20.0 * activity,
        specialty_valid=True,
        activity_role_valid=counts[1:] > 0,
        movement_code=-1,
        movement_valid=True,
        fission_valid=True,
        trade_gate_valid=np.full(groups, trade_valid),
        trade_value_active=np.full(groups, trade_valid),
        trade_share_active=trade_valid,
        over_budget=False,
    )


def test_structured_candidates_conserve_residents_and_worker_time() -> None:
    catalog = _catalog()
    base = _base_payload(catalog)
    base_counts, base_activity = labor_state(base, catalog)
    families = structured_candidates(base, catalog, _observation(), key=[1, 2, 3])
    assert set(families) == set(PERTURBATION_FAMILIES)
    for family in PERTURBATION_FAMILIES:
        assert families[family], f"no {family} candidates"
    for family, cands in families.items():
        for cand in cands:
            assert conserves_labor(cand.payload, catalog), cand.description
            counts, activity = labor_state(cand.payload, catalog)
            assert int(counts.sum()) == 100
            np.testing.assert_allclose(activity.sum(axis=1), 1.0, atol=1e-9)
            if family in ("movement", "fission", "trade"):
                np.testing.assert_array_equal(counts, base_counts)
                np.testing.assert_allclose(activity, base_activity, atol=1e-12)
            if family == "activity":
                np.testing.assert_array_equal(counts, base_counts)
    for cand in families["transfer"]:
        counts, _ = labor_state(cand.payload, catalog)
        _, amount, pair = cand.description.split(" ")
        src, dst = pair.split("->")
        assert counts[S[src]] == base_counts[S[src]] - int(amount)
        assert counts[S[dst]] == base_counts[S[dst]] + int(amount)
    gb = [c for c in families["transfer"] if c.description.endswith("Gatherer->Builder")]
    assert gb, "Builder transfer from Gatherer"
    _, activity = labor_state(gb[0].payload, catalog)
    assert activity[S["Builder"] - 1, A["goods.housing"]] == pytest.approx(1.0)


def test_trade_and_movement_families_respect_opportunities_and_masks() -> None:
    catalog = _catalog()
    base = _base_payload(catalog)
    assert trade_opportunities(_observation((1, 3))) == [1, 3]
    families = structured_candidates(base, catalog, _observation((1, 3)), key=[7])
    flipped = sorted(
        next(
            i
            for i, (a, b) in enumerate(zip(c.payload["tradeGate"], base["tradeGate"], strict=True))
            if a != b
        )
        for c in families["trade"]
    )
    assert flipped == [1, 3]
    assert all(0 in c.payload["tradeGate"] for c in families["trade"])
    moves = [c.payload["movement"] for c in families["movement"]]
    assert len(moves) == 3 and all(m >= contracts.REGION_BASE + 1 for m in moves)
    assert sorted(c.payload["fission"] for c in families["fission"]) == [1, 2]


def test_candidates_are_deterministic_by_key() -> None:
    catalog = _catalog()
    base = _base_payload(catalog)
    one = round_robin(structured_candidates(base, catalog, _observation(), key=[5, 9]))
    two = round_robin(structured_candidates(base, catalog, _observation(), key=[5, 9]))
    other = round_robin(structured_candidates(base, catalog, _observation(), key=[5, 10]))
    assert [c.digest() for c in one] == [c.digest() for c in two]
    assert [c.digest() for c in one] != [c.digest() for c in other]
    # The key reorders families; only the keyed movement subset (and the joint that
    # reuses its first move) may change membership.
    fixed = [f for f in PERTURBATION_FAMILIES if f not in ("movement", "joint")]
    assert sorted(c.digest() for c in one if c.family in fixed) == sorted(
        c.digest() for c in other if c.family in fixed
    )
    # Round robin covers every family within the first len(families) slots.
    assert {c.family for c in one[: len(PERTURBATION_FAMILIES)]} == set(PERTURBATION_FAMILIES)


def test_keep_counts_and_halving_bookkeeping() -> None:
    config = SearchConfig()
    assert config.keep_counts(16) == [16, 8, 4, 2]
    assert SearchConfig(candidates=8, horizons=(1, 2), keep=("all", "1/2")).keep_counts(8) == [8, 4]
    # Candidate 0 is the protected reference. Personality columns disagree.
    u = np.asarray(
        [
            [0.0, 0.0],
            [5.0, -9.0],  # best for personality 0 only
            [-9.0, 5.0],  # best for personality 1 only
            [3.0, 3.0],  # second for both
            [1.0, 1.0],
            [-1.0, -1.0],
        ]
    )
    order = halving_order(u, range(6))
    assert order[:2] == [1, 2]
    assert order[2] == 3
    survivors = select_survivors(u, list(range(6)), 2, protected=0)
    assert survivors == [0, 1, 2]
    survivors = select_survivors(u, [0, 3, 4, 5], 1, protected=0)
    assert survivors == [0, 3]
    # Exact ties share a rank and are ordered by the keyed tie-break, not by id.
    tied = np.zeros((4, 2))
    assert halving_order(tied, range(4), {0: 3, 1: 2, 2: 0, 3: 1}) == [2, 3, 1, 0]
    assert select_survivors(
        tied, list(range(4)), 1, protected=0, tiebreak={0: 3, 1: 2, 2: 0, 3: 1}
    ) == [0, 2]


def test_personality_rescoring_from_one_return_vector() -> None:
    config = SearchConfig()
    returns = np.asarray(
        [
            [40.0, 8000.0, 0.0, 0.0],  # H12
            [44.0, 7000.0, 0.0, 0.0],  # more QoL, fewer person-years
            [38.0, 12000.0, 0.0, 0.0],  # more person-years
            [60.0, 20000.0, 1.0, 1.0],  # costs
        ]
    )
    grid = [(0.5, 0.5), (0.9, 0.1), (0.1, 0.9)]
    u = utility_matrix(returns, grid, config)
    sq = LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"]
    sp = LABOR_OBJECTIVE_REFERENCE_SCALES["population_years"]
    expected = 0.9 * 44.0 / sq + 0.1 * 7000.0 / sp
    assert u[1, 1] == pytest.approx(expected)
    assert u[3, 0] == pytest.approx(0.5 * 60 / sq + 0.5 * 20000 / sp - 2.0)
    assert int(np.argmax(u[:, 1])) == 1
    assert int(np.argmax(u[:, 2])) == 2
    np.testing.assert_allclose(utility(returns, grid[2], config), u[:, 2])
    assert len(COMPONENTS) == returns.shape[1]


def test_paired_acceptance_margin_and_standard_error() -> None:
    ok = paired_acceptance([1.10, 1.11, 1.12], [1.0, 1.0, 1.01], 0.005, 2.0)
    assert ok["accepted"] and ok["mean_gain"] == pytest.approx(0.1066666, rel=1e-4)
    noisy = paired_acceptance([1.3, 0.8, 1.2], [1.0, 1.0, 1.0], 0.005, 2.0)
    assert not noisy["accepted"]
    small = paired_acceptance([1.004, 1.004], [1.0, 1.0], 0.005, 2.0)
    assert not small["accepted"]  # below the 0.5% margin with zero SE
    with pytest.raises(ValueError):
        paired_acceptance([1.0], [1.0], 0.005, 2.0)


def test_candidate_label_matches_collector_format_and_validity() -> None:
    catalog = _catalog()
    base = _base_payload(catalog)
    obs = _observation()
    h12 = _h12_label(catalog, base, trade_valid=False)
    families = structured_candidates(base, catalog, obs, key=[3])
    reference = _label_tensors(h12)
    for family in ("transfer", "trade", "movement"):
        cand = families[family][0]
        label = candidate_label(cand, catalog, h12, obs)
        tensors = _label_tensors(label)
        assert set(tensors) == set(reference)
        for key, value in tensors.items():
            assert value.shape == reference[key].shape and value.dtype == reference[key].dtype, key
        if family == "transfer":
            assert bool(tensors["label_specialty_valid"])
            assert not bool(tensors["label_trade_gate_valid"].any())  # inherited H12 drive default
            counts, _ = labor_state(cand.payload, catalog)
            np.testing.assert_allclose(label.specialty, counts / 100.0)
        if family == "trade":
            assert bool(tensors["label_trade_gate_valid"].all())
    h12_cand = Candidate(dict(base), "h12", "converted heuristic", frozenset())
    assert candidate_label(h12_cand, catalog, h12, obs) is h12


def test_converted_label_tensors_match_search_label_keys() -> None:
    # A converted collector label and a search label share one tensor schema.
    catalog = _catalog()
    converted = convert_heuristic_labor_label(catalog, {}, {}, pre_observation=None)
    base = _base_payload(catalog)
    h12 = _h12_label(catalog, base, trade_valid=True)
    cand = structured_candidates(base, catalog, _observation(), key=[1])["activity"][0]
    a, b = _label_tensors(converted), _label_tensors(candidate_label(cand, catalog, h12, None))
    assert set(a) == set(b)
    assert all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype for k in a)


def test_dither_common_random_numbers_and_conservation() -> None:
    catalog = _catalog()
    base = _base_payload(catalog)
    draws = np.random.default_rng([1, 2, 3]).random(3)
    first = dither_payload(base, catalog, [0.1, 0.2, 0.9], probability=0.5, fraction=0.05)
    again = dither_payload(base, catalog, [0.1, 0.2, 0.9], probability=0.5, fraction=0.05)
    assert first == again and first is not None
    assert conserves_labor(first, catalog)
    counts, _ = labor_state(first, catalog)
    base_counts, _ = labor_state(base, catalog)
    assert int(np.abs(counts - base_counts).sum()) == 10  # 5 residents moved
    assert dither_payload(base, catalog, [0.9, 0.2, 0.9], probability=0.5, fraction=0.05) is None
    assert len(draws) == 3


def test_root_turns_and_focal_cycling_are_keyed() -> None:
    assert root_turns(90100, 0, 16, 8) == root_turns(90100, 0, 16, 8)
    assert root_turns(90100, 0, 16, 8) != root_turns(90101, 0, 16, 8)
    assert len(set(root_turns(90100, 0, 16, 8))) == 8
    assert root_turns(90100, 0, 4, 8) == [0, 1, 2, 3]
    order = focal_order([3, 1, 2], {}, [1])
    assert sorted(order) == [1, 2, 3]
    assert focal_order([3, 1, 2], {order[0]: 1}, [1])[-1] == order[0]


def test_search_worlds_are_fitting_only() -> None:
    assert validate_search_worlds([90100, 90159]) == [90100, 90159]
    for bad in ([3000], [2000], [90160]):
        with pytest.raises(ValueError):
            validate_search_worlds(bad)


def test_physical_digest_ignores_identity_and_timing_tags() -> None:
    a = {
        "nativeMs": 1.0,
        "trajectoryId": "x1",
        "villages": [{"q": 1, "decisionDiagnostics": {"TrajectoryId": "a"}}],
    }
    b = {
        "nativeMs": 9.0,
        "trajectoryId": "x2",
        "villages": [{"q": 1, "decisionDiagnostics": {"TrajectoryId": "b"}}],
    }
    c = {
        "nativeMs": 9.0,
        "trajectoryId": "x2",
        "villages": [{"q": 2, "decisionDiagnostics": {"TrajectoryId": "b"}}],
    }
    assert physical_digest(a) == physical_digest(b)
    assert physical_digest(a) != physical_digest(c)


def test_search_shard_identity_includes_label_variant(tmp_path: Path) -> None:
    from anthropocene_rl.autonomous_v7.labor_teacher_data import verify_shard, write_shard
    from anthropocene_rl.autonomous_v7.provenance import ProvenanceError

    arrays = {
        "row_world": np.asarray([90100, 90100]),
        "row_turn": np.asarray([9, 9]),
        "row_camp": np.asarray([15, 15]),
        "row_attempt": np.asarray([0, 0]),
    }
    write_shard(tmp_path / "plain", arrays, {"world_seed": 90100})
    with pytest.raises(ProvenanceError):
        verify_shard(tmp_path / "plain")
    write_shard(
        tmp_path / "variant",
        {**arrays, "row_label_variant": np.asarray([0, 1])},
        {"world_seed": 90100},
    )
    assert verify_shard(tmp_path / "variant")["rows"] == 2


BRIDGE = os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE")


@pytest.mark.skipif(not BRIDGE or not Path(BRIDGE).is_file(), reason="native bridge unavailable")
def test_native_fork_restore_repeats_the_same_successor() -> None:
    from anthropocene_rl.autonomous_v7.bridge import Model9Bridge
    from anthropocene_rl.autonomous_v7.labor_bc import LaborResetOptions
    from anthropocene_rl.autonomous_v7.labor_bridge import LaborModel9Bridge
    from anthropocene_rl.autonomous_v7.labor_search import NativeSearch, verify_exact_restore

    options = LaborResetOptions(players=4)
    with LaborModel9Bridge() as bridge, Model9Bridge(economic_diagnostics=True) as heuristic:
        native = NativeSearch(
            bridge, heuristic, world=90100, options=options, config=SearchConfig()
        )
        step = bridge.reset(90100, **options.reset_kwargs())
        labels = native.shadow_h12(step, native.catalogs())
        payloads = {camp: label.action for camp, label in labels.items()}
        check = verify_exact_restore(native, payloads)
        assert check["equal"]
        # The restored root still executes to the same successor as a fresh branch.
        fork = native.fork()
        a = physical_digest(native.execute(native.catalogs(), payloads).step)
        native.restore(fork)
        b = physical_digest(native.execute(native.catalogs(), payloads).step)
        native.drop(fork)
        assert a == b == check["digest"]
        assert torch.is_tensor(torch.zeros(1))
