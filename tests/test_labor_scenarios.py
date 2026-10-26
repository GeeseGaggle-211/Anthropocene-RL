"""Curated Neolithic scenario bank (labor_scenarios): detection, quotas, partitions, shards."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_bc, labor_scenarios as ls
from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.labor_privileged import LABOR_PRIVILEGED_LAYOUT_V1 as LAYOUT
from anthropocene_rl.autonomous_v7.labor_scenarios import (
    FAMILIES,
    CampState,
    DetectionRules,
    FamilyTracker,
)
from anthropocene_rl.autonomous_v7.labor_worlds import require_training_record

FARMER, HERDER, FISHER = (True, False, False), (False, True, False), (False, False, True)


def _state(camp, turn=8, heads=FARMER, residents=300.0, neolithic=True, housing=1.0,
           clothing=1.0, trade=0.0):
    return CampState(camp, turn, heads, residents, neolithic, housing, clothing, trade)


def _run(sequence, rules=None):
    """sequence: list of decisions, each a list of CampState; returns per-decision observations."""
    tracker = FamilyTracker(rules)
    return [tracker.observe(d, states) for d, states in enumerate(sequence)]


# Detection rules on synthetic states -------------------------------------------------------


def test_reset_camp_head_window_is_its_first_two_decisions() -> None:
    out = _run([[_state(1)], [_state(1)], [_state(1)]])
    assert ["farming-unlocked" in o[1].tags for o in out] == [True, True, False]
    assert out[0][1].entries >= {"farming-unlocked"}
    assert out[0][1].subtypes["farming-unlocked"] == "reset"
    assert "farming-unlocked" not in out[1][1].entries  # one root per run


def test_in_trajectory_unlock_and_mixed_window() -> None:
    both = (True, True, False)
    seq = [[_state(1, heads=FARMER)]] * 2 + [[_state(1, heads=both)]] * 4
    out = _run(seq)
    herding = ["herding-unlocked" in o[1].tags for o in out]
    mixed = ["mixed" in o[1].tags for o in out]
    assert herding == [False, False, True, True, False, False]
    assert mixed == [False, False, True, True, False, False]
    assert out[2][1].subtypes == {"herding-unlocked": "in-trajectory", "mixed": "in-trajectory"}
    # Masked again then re-legal: a new window and a new root.
    seq = [[_state(1, heads=FISHER)], [_state(1, heads=(False, False, False))],
           [_state(1, heads=FISHER)]]
    out = _run(seq)
    assert "fishing-unlocked" in out[2][1].entries


def test_daughter_window_and_no_inherited_head_window() -> None:
    seq = [[_state(1)], [_state(1)], [_state(1), _state(9)], [_state(1), _state(9)],
           [_state(1), _state(9)]]
    out = _run(seq)
    assert ["fission-daughter" in o[9].tags for o in out[2:]] == [True, True, False]
    # The daughter inherits Farmer: no farming-unlocked window of its own.
    assert not any("farming-unlocked" in o[9].tags for o in out[2:])
    assert "fission-daughter" not in out[0][1].tags


def test_state_families_thresholds_and_run_entries() -> None:
    rules = DetectionRules()
    small = _run([[_state(1, residents=99.0)], [_state(1, residents=100.0)]])
    assert "small-camp" in small[0][1].tags and "small-camp" not in small[1][1].tags
    large = _run([[_state(1, residents=601.0)], [_state(1, residents=600.0)]])
    assert "large-camp" in large[0][1].tags and "large-camp" not in large[1][1].tags
    housing = [0.0, 0.1, 0.29, 0.3, 0.2, 0.1]
    out = _run([[_state(1, housing=h)] for h in housing])
    assert ["under-housed" in o[1].tags for o in out] == [True, True, True, False, True, True]
    entries = [d for d, o in enumerate(out) if "under-housed" in o[1].entries]
    assert entries == [0, 4]
    assert out[0][1].subtypes["under-housed"] == "initial"
    assert out[4][1].subtypes["under-housed"] == "relapse"
    cloth = _run([[_state(1, clothing=0.25)]])
    assert "under-clothed" in cloth[0][1].tags
    trade = _run([[_state(1, trade=0.0)], [_state(1, trade=1.0)]])
    assert "trade-eligible" not in trade[0][1].tags and "trade-eligible" in trade[1][1].entries
    assert rules.state_dict()["housing_below"] == 0.3


def test_paleolithic_states_carry_no_family_unless_any_era() -> None:
    paleo = [[_state(1, residents=24.0, neolithic=False, housing=0.0)]]
    assert _run(paleo)[0][1].tags == frozenset()
    any_era = _run(paleo, DetectionRules(require_neolithic=False))[0][1].tags
    assert {"small-camp", "under-housed"} <= any_era


def _synthetic_arrays(rows):
    """rows: (world, turn, camp, heads, residents, housing, clothing, trade, neolithic)."""
    n = len(rows)
    economy = LAYOUT.family("economy")
    unlocks = LAYOUT.family("unlocks")
    mask = np.zeros((n, 14), dtype=bool)
    mask[:, 0] = mask[:, 1] = True
    econ = np.zeros((n, economy.rows, economy.width), dtype=np.float32)
    unl = np.zeros((n, unlocks.rows, unlocks.width), dtype=np.float32)
    for i, (_, _, _, heads, res, house, cloth, trade, neo) in enumerate(rows):
        mask[i, 2:5] = heads
        econ[i, 0, economy.features.index("residents")] = res
        econ[i, 0, economy.features.index("housing-per-capita")] = house
        econ[i, 0, economy.features.index("cloth-per-capita")] = cloth
        econ[i, 0, economy.features.index("trade-partners")] = trade
        unl[i, 0, unlocks.features.index("era-neolithic")] = float(neo)
    return {
        "row_world": np.asarray([r[0] for r in rows], dtype=np.int64),
        "row_turn": np.asarray([r[1] for r in rows], dtype=np.int64),
        "row_camp": np.asarray([r[2] for r in rows], dtype=np.int64),
        "labor_specialty_mask": mask,
        "privileged_economy": econ,
        "privileged_economy_present": np.ones((n, economy.rows), dtype=bool),
        "privileged_unlocks": unl,
        "privileged_unlocks_present": np.ones((n, unlocks.rows), dtype=bool),
    }


def test_array_path_matches_the_tracker() -> None:
    rows = [
        (90100, 8, 1, FARMER, 50.0, 0.0, 1.0, 0.0, 1),
        (90100, 8, 2, HERDER, 700.0, 1.0, 0.1, 1.0, 1),
        (90100, 9, 1, FARMER, 50.0, 1.0, 1.0, 0.0, 1),
        (90100, 9, 2, (False, True, True), 700.0, 1.0, 0.1, 1.0, 1),
        (90100, 9, 7, FARMER, 30.0, 1.0, 1.0, 0.0, 1),
    ]
    arrays = _synthetic_arrays(rows)
    states = ls.states_from_arrays(arrays, [0, 1])
    assert states[0] == CampState(1, 8, FARMER, 50.0, True, 0.0, 1.0, 0.0)
    tags = ls.row_family_tags({k: torch.as_tensor(v) for k, v in arrays.items()})
    names = [set(ls.family_names(int(t))) for t in tags]
    assert names[0] == {"farming-unlocked", "small-camp", "under-housed"}
    assert names[1] == {"herding-unlocked", "large-camp", "under-clothed", "trade-eligible"}
    assert names[3] == {"herding-unlocked", "fishing-unlocked", "mixed", "large-camp",
                        "under-clothed", "trade-eligible"}
    assert names[4] == {"small-camp", "fission-daughter"}
    assert ls.family_mask(names[0]) == int(tags[0])


# Quotas ----------------------------------------------------------------------------------


def test_quota_sampling_is_keyed_and_order_independent() -> None:
    candidates = [(d, c, ["under-housed", "small-camp"]) for d in range(20) for c in range(10)]
    candidates.append((5, 3, ["mixed"]))  # rare family
    caps = {"under-housed": 4, "small-camp": 2, "mixed": 3}
    first = ls.select_roots(candidates, caps, key=7, world=90100)
    shuffled = list(candidates)
    random.Random(3).shuffle(shuffled)
    assert ls.select_roots(shuffled, caps, key=7, world=90100) == first
    counts = {f: sum(f in fams for fams in first.values()) for f in caps}
    assert counts == {"under-housed": 4, "small-camp": 2, "mixed": 1}  # rare kept, common capped
    assert ls.select_roots(candidates, caps, key=8, world=90100) != first
    assert ls.select_roots(candidates, caps, key=7, world=90101) != first
    # The kept roots are the lowest keyed priorities.
    pool = sorted(ls.root_priority(7, 90100, "small-camp", d, c) for d, c, _ in candidates[:-1])
    kept = sorted(p["small-camp"] for p in first.values() if "small-camp" in p)
    assert kept == pool[:2]


def test_quota_caps_fill_only_the_gaps() -> None:
    coverage = {
        "minima": {"fitting": [32, 4]},
        "partitions": {
            "fitting": {
                "families": {
                    name: {"met": name != "mixed" and name != "trade-eligible",
                           "deficit_roots": 30 if name == "mixed" else 0}
                    for name in FAMILIES
                    if name != "trade-eligible"
                }
            }
        },
    }
    caps = ls.quota_caps(coverage, "fitting", 60)
    assert caps["mixed"] == 1  # ceil(1.5 * 30 / 60)
    assert caps["trade-eligible"] == 1  # absent from the audit: the full minimum, spread
    assert all(caps[name] == 0 for name in FAMILIES if name not in ("mixed", "trade-eligible"))
    assert ls.quota_caps(coverage, "fitting", 2, families=["mixed"]) == {"mixed": 8}
    assert set(ls.quota_caps(None, "fitting", 1, per_world=3).values()) == {3}
    with pytest.raises(ValueError, match="coverage"):
        ls.quota_caps(None, "fitting", 1)


# Partitions ------------------------------------------------------------------------------


def test_partition_guards() -> None:
    assert ls.validate_scenario_worlds([90100, 90165], "fitting") == [90100, 90165]
    assert ls.validate_scenario_worlds([90180], "calibration") == [90180]
    assert ls.validate_scenario_worlds([90154], "held-out-test", held_out_test=True) == [90154]
    with pytest.raises(ValueError, match="reserved"):
        ls.validate_scenario_worlds([90154], "fitting")
    with pytest.raises(ValueError, match="not fitting worlds"):
        ls.validate_scenario_worlds([90180], "fitting")
    with pytest.raises(ValueError, match="not calibration worlds"):
        ls.validate_scenario_worlds([90100], "calibration")
    with pytest.raises(ValueError, match="held-out-test"):
        ls.validate_scenario_worlds([90154], "held-out-test")
    with pytest.raises(ValueError, match="held-out-test"):
        ls.validate_scenario_worlds([90100], "fitting", held_out_test=True)
    with pytest.raises(ValueError, match="development"):
        ls.validate_scenario_worlds([2000], "fitting")
    with pytest.raises(ValueError, match="not fitting worlds"):
        ls.validate_scenario_worlds([5000], "fitting")
    assert [ls.world_partition(w) for w in (90100, 90160, 90155, 90180, 2000, 3000, 5000)] == [
        "fitting", "fitting", "held-out-test", "calibration", "development", "sealed", "other"]
    with pytest.raises(ValueError, match="calibration-set"):
        require_training_record({"options": {"partition": "calibration"}}, "loader")
    with pytest.raises(ValueError, match="held-out-test"):
        require_training_record({"partition": "held-out-test"}, "loader")
    require_training_record({"partition": "fitting"}, "loader")


# Source kind, weighting and critic admission ----------------------------------------------


def _critic_rows(n, source, trajectory, *, tails=True):
    data = {
        "row_world": torch.full((n,), 90100, dtype=torch.long),
        "row_camp": torch.arange(n, dtype=torch.long),
        "row_turn": torch.full((n,), 3, dtype=torch.long),
        "row_quarantined": torch.zeros(n, dtype=torch.bool),
        "row_source": torch.full((n,), source, dtype=torch.long),
        labor_bc.TRAJECTORY_FIELD: torch.full((n,), trajectory, dtype=torch.long),
    }
    if tails:
        data["tail_method"] = torch.full((n,), labor_bc.TAIL_RANDOMIZED)
    return data


def test_scenario_source_kind_weighting_and_critic_admission() -> None:
    assert ltd.ROW_SOURCES[ls.SOURCE_KIND] == ls.ROW_SOURCE == labor_bc.SCENARIO_ROW_SOURCE
    for key in ("row_scenario_family", "row_scenario_tags", "row_root_turn"):
        assert key in ltd.AGGREGATE_IDENTITY_DEFAULTS
    # Equal source mass: 90 teacher rows and 10 scenario rows carry the same total weight.
    sources = torch.tensor([0] * 90 + [4] * 10)
    worlds = torch.tensor([90100] * 45 + [90101] * 45 + [90100] * 10)
    weights = ltd.source_world_balanced_weights(sources, worlds)
    assert torch.isclose(weights[:90].sum(), weights[90:].sum())
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    data = _critic_rows(2, ls.ROW_SOURCE, ls.TRAJECTORY_BASE)
    data["collection"] = {"source_kind": ls.SOURCE_KIND, "actor_only": False}
    with pytest.raises(ValueError, match="curated-scenarios"):
        labor_bc.require_critic_teacher_dataset(data)
    labor_bc.require_critic_teacher_dataset(data, allow_curated_scenarios=True)
    mixed = {k: torch.cat([a, b]) for (k, a), b in zip(
        _critic_rows(2, 0, -1).items(), _critic_rows(2, 4, ls.TRAJECTORY_BASE).values(),
        strict=True)}
    mixed["collection"] = {
        "source_kind": "aggregated",
        "component_collections": [
            {"source_kind": labor_bc.CRITIC_TEACHER_SOURCE_KIND,
             "return_tail": labor_bc.RETURN_TAIL_RANDOMIZED},
            {"source_kind": ls.SOURCE_KIND, "return_tail": labor_bc.RETURN_TAIL_RANDOMIZED},
        ],
    }
    labor_bc.require_critic_teacher_dataset(mixed, allow_curated_scenarios=True)
    with pytest.raises(ValueError, match="aggregated"):
        labor_bc.require_critic_teacher_dataset(mixed, allow_learner_state_continuation=True)
    lost = {k: v for k, v in mixed.items() if k != "tail_method"}
    with pytest.raises(ValueError, match="curated scenario rows lost"):
        labor_bc.require_critic_teacher_dataset(lost, allow_curated_scenarios=True)
    truncated = dict(lost, collection=dict(mixed["collection"], component_collections=[
        dict(c, return_tail=labor_bc.RETURN_TAIL_TRUNCATED_ZERO)
        for c in mixed["collection"]["component_collections"]]))
    labor_bc.require_critic_teacher_dataset(truncated, allow_curated_scenarios=True)
    # Scenario trajectories never merge with learner-state continuation ids.
    assert ls.TRAJECTORY_BASE > 10_000


# Live native smoke (local bridge): 1 world, 8 players, harvest 6, 2 roots, 2-decision scenarios


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    from anthropocene_rl.autonomous_v7.labor_model import BOUNDED_FEATURE_CONTRACT
    from anthropocene_rl.autonomous_v7.labor_training import main as labor_training_main
    from tests.test_labor_bc import _fit_test_normalizer

    root = tmp_path_factory.mktemp("scenarios")
    norm = root / "norm.pt"
    torch.save(
        {"feature_contract": BOUNDED_FEATURE_CONTRACT,
         "normalizer": _fit_test_normalizer().state_dict()},
        norm,
    )
    init = root / "init.pt"
    labor_training_main(["initialize", "--normalizer-checkpoint", str(norm), "--world-seed",
                         "90100", "--optimizer-seed", "1", "--campaign-r5", "--out", str(init)])
    teacher = root / "teacher"
    assert ltd.main(["--normalizer-checkpoint", str(norm), "--worlds", "90100", "--turns", "2",
                     "--players", "8", "--jobs", "1", "--return-tail", "truncated-zero",
                     "--allow-dirty", "--out", str(teacher)]) == 0
    scen = root / "scen"
    common = ["collect", "--normalizer-checkpoint", str(norm), "--partition", "fitting",
              "--worlds", "90100", "--players", "8", "--decisions", "6",
              "--scenario-decisions", "2", "--families", "under-housed",
              "--quota-per-world", "2", "--return-tail", "truncated-zero", "--jobs", "1",
              "--allow-dirty"]
    assert ls.main([*common, "--out", str(scen)]) == 0
    return root, norm, init, teacher, scen, common


def test_live_smoke_shards_roots_and_coverage(live) -> None:
    root, _, _, teacher, scen, common = live
    manifest = ltd.verify_dataset(scen)
    assert manifest["source_record"]["source_kind"] == ls.SOURCE_KIND
    meta = ltd.verify_shard(scen / "world-90100")
    assert len(meta["roots"]) == 2
    assert all("under-housed" in r["selected_for"] for r in meta["roots"])
    data = ltd.load_labor_dataset(scen, require_privileged=True)
    assert data["collection"]["source_kind"] == ls.SOURCE_KIND
    assert bool((data["row_source"] == ls.ROW_SOURCE).all())
    assert set(data["row_start_index"].tolist()) == {0, 1}
    assert set(data[labor_bc.TRAJECTORY_FIELD].tolist()) == {ls.TRAJECTORY_BASE,
                                                             ls.TRAJECTORY_BASE + 1}
    assert set(data["row_branch_step"].tolist()) <= {0, 1}
    bit = ls.FAMILY_BITS["under-housed"]
    assert bool(((data["row_scenario_family"] & bit) != 0).all())
    # Each root's step-0 row is the bank row of the same state (deterministic teacher).
    bank = ltd.load_labor_dataset(teacher)
    for r in meta["roots"]:
        step0 = (data["row_camp"] == r["camp"]) & (data["row_branch_step"] == 0)
        same = (bank["row_camp"] == r["camp"]) & (bank["row_turn"] == r["turn"])
        if bool(same.any()):
            for key in ("labor_specialty_mask", "core", "target_specialty"):
                assert torch.equal(data[key][step0], bank[key][same]), key
    roots = ls.load_scenario_roots(scen)
    assert [r["root_id"] for r in roots["roots"]] == [r["root_id"] for r in meta["roots"]]
    # Coverage of teacher bank plus scenario set.
    report = ls.coverage_audit([teacher, scen])
    fitting = report["partitions"]["fitting"]["families"]
    assert fitting["under-housed"]["roots"] >= 2
    assert fitting["under-housed"]["roots_by_dataset"]
    assert "first_active_use" in report["partitions"]["fitting"]
    # Excluded bank states: same reset options only.
    excluded = ls.excluded_states([teacher], meta["reset_options"])
    assert len(excluded[90100]) == int(bank["row_world"].shape[0])
    assert ls.excluded_states([teacher], dict(meta["reset_options"], players=16)) == {}
    # Idempotent: rerunning the same command is a no-op on the published output.
    assert ls.main([*common, "--out", str(scen)]) == 0
    assert ltd.verify_dataset(scen)["dataset_digest"] == manifest["dataset_digest"]
    cost = json.loads((scen / "summary.json").read_text())["cost"]
    assert cost["roots"] == 2 and cost["scenario_plus_tail_native_resolves_per_root"] > 0


def test_live_scenario_rows_train_bc_and_critic_with_source_weighting(live) -> None:
    from anthropocene_rl.autonomous_v7.labor_training import main as labor_training_main

    root, _, init, teacher, scen, _ = live
    both = ltd.load_labor_datasets([teacher, scen])
    sources = both["collection"]["sources"]
    assert set(sources) == {labor_bc.CRITIC_TEACHER_SOURCE_KIND, ls.SOURCE_KIND}
    assert sources[ls.SOURCE_KIND]["weight_mass"] == pytest.approx(
        sources[labor_bc.CRITIC_TEACHER_SOURCE_KIND]["weight_mass"], rel=1e-5
    )
    assert "row_scenario_family" in both
    bc = root / "bc.pt"
    labor_training_main(["pretrain-bc", "--checkpoint", str(init), "--dataset", str(teacher),
                         "--dataset", str(scen), "--epochs", "1", "--out", str(bc)])
    critic = root / "critic.pt"
    with pytest.raises(ValueError, match="curated-scenarios"):
        labor_training_main(["pretrain-critic", "--checkpoint", str(init), "--dataset",
                             str(scen), "--epochs", "1", "--out", str(critic)])
    labor_training_main(["pretrain-critic", "--checkpoint", str(init), "--dataset", str(teacher),
                         "--dataset", str(scen), "--curated-scenarios", "--epochs", "1",
                         "--batch-size", "16", "--out", str(critic)])
    summary = torch.load(critic, map_location="cpu", weights_only=True)[
        "heuristic_critic_pretraining"
    ]
    assert summary["curated_scenario_rows"] > 0
    assert summary["sampling"].startswith("equal-source-mass")


def test_live_scenario_roots_feed_search_payloads(live) -> None:
    from anthropocene_rl.autonomous_v7 import labor_search
    from anthropocene_rl.autonomous_v7.labor_advantage import load_candidate_shards
    from anthropocene_rl.autonomous_v7.labor_search_payload import candidate_shard_paths

    root, norm, _, _, scen, _ = live
    out = root / "search"
    assert labor_search.main([
        "--normalizer-checkpoint", str(norm), "--worlds", "90100", "--source", "scenario",
        "--scenario-roots", str(scen), "--players", "8", "--candidates", "2",
        "--horizons", "1", "--keep", "all", "--personalities", "balanced",
        "--reeval-streams", "2", "--control-fraction", "0", "--max-resolves", "12",
        "--no-verify-restore", "--allow-dirty", "--out", str(out),
    ]) == 0
    paths = candidate_shard_paths(out)
    candidates = load_candidate_shards(paths)
    roots = ls.load_scenario_roots(scen)["roots"]
    assert {r.camp for r in candidates.roots} <= {r["camp"] for r in roots}
    assert all("under-housed" in r.families for r in candidates.roots)
    assert all(r.root_id.endswith("/scenario") for r in candidates.roots)


def test_calibration_scenario_sets_are_refused_for_training(tmp_path: Path) -> None:
    source = {"options": {"partition": "calibration"}, "source_kind": ls.SOURCE_KIND}
    with pytest.raises(ValueError, match="calibration-set"):
        ltd._training_view({"source_record": source, "shards": {}}, tmp_path, training=True,
                           drop=False)
