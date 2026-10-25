"""Search candidate-payload export (labor_search_payload v1) and the A-ranker hook.

A fake native search (deterministic per-candidate rewards, no bridge) drives
the real ``search_root`` so the payload, the final-survivor re-evaluation and
the ranker pruning are checked end to end; a monkeypatched world driver checks
the shard binding, verification and loader.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_search
from anthropocene_rl.autonomous_v7 import labor_search_payload as lsp
from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.labor_bc import LaborResetOptions
from anthropocene_rl.autonomous_v7.labor_bridge import ACTIVITIES, SPECIALTIES
from anthropocene_rl.autonomous_v7.labor_search import (
    COMPONENTS,
    Branch,
    NativeSearch,
    SearchConfig,
    labor_state,
    search_root,
)
from anthropocene_rl.autonomous_v7.labor_worlds import HELD_OUT_TEST_PARTITION
from anthropocene_rl.autonomous_v7.provenance import digest
from tests.test_labor_search import _base_payload, _catalog, _h12_label, _observation

FOCAL = 7
YEARS = 20.0


class FakeNative(NativeSearch):
    """Deterministic rewards per candidate payload; stream adds a small offset."""

    def __init__(self, catalog, config: SearchConfig, world: int = 90100) -> None:
        super().__init__(None, None, world=world, options=LaborResetOptions(), config=config)
        self._catalog = catalog
        self.extended: list[tuple[int, int, int]] = []

    def catalogs(self):
        return {FOCAL: self._catalog}

    def validate(self, camp, payload):
        self.work.validations += 1
        return SimpleNamespace(valid=True, violations=[], feasibility_cost=0.0)

    def fork(self) -> int:
        return 1

    def drop(self, fork) -> None:
        return None

    @staticmethod
    def value(payload) -> float:
        return int(labor_search.payload_digest(payload)[:6], 16) / float(0xFFFFFF)

    def extend(self, branch: Branch, target: int, *, candidate, keep_fork, **_) -> None:
        self.extended.append((branch.candidate, branch.stream, target))
        v = self.value(candidate)
        while branch.horizon < target and branch.alive:
            r = np.asarray([v + 0.01 * branch.stream, 10.0 * v, 0.0, 0.0])
            branch.returns = branch.returns + branch.discount * r
            branch.steps.append(r.tolist())
            branch.years.append(YEARS)
            branch.discount *= math.exp(-YEARS / self.config.discount_years)
            branch.horizon += 1
            self.work.labor_steps += 1
            self.work.shadow_steps += 1


def _root(config: SearchConfig, *, ranker=None, root_index: int = 0, world: int = 90100):
    catalog = _catalog()
    base = _base_payload(catalog)
    h12 = _h12_label(catalog, base, trade_valid=True)
    village = {"camp": FOCAL, "society": 3, "observation": _observation()}
    native = FakeNative(catalog, config, world)
    result = search_root(
        native,
        step={"turn": 4, "villages": [village]},
        root_h12={FOCAL: h12},
        focal=FOCAL,
        founded={FOCAL: 0},
        tensors={"probe": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
        row=1,
        root_pair=(0.3, 0.7),
        actor=None,
        root_fork=0,
        key=[world, 0, 4, FOCAL],
        root_source="actor",
        root_index=root_index,
        ranker=ranker,
    )
    return native, result, catalog


def test_movement_union_matches_the_actor_head() -> None:
    assert len(lsp.MOVEMENT_UNION_CODES) == 17
    assert lsp.movement_slot(-1) == 0 and lsp.movement_slot(107) == 16
    with pytest.raises(ValueError):
        lsp.movement_slot(-2)  # the forbidden native sentinel is not an actor action


def test_returns_by_horizon_and_reeval_summary() -> None:
    steps = [[1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 0.5, 0.0]]
    g = math.exp(-20.0 / 400.0)
    by_h, ok = lsp.returns_by_horizon(steps, [20.0, 20.0], [1, 2, 3], terminated=False,
                                      discount_years=400.0)
    assert ok.tolist() == [True, True, False]
    assert np.allclose(by_h[1], [1 + g, 2 + 2 * g, 0.5 * g, 0.0])
    by_h, ok = lsp.returns_by_horizon(steps, [20.0, 20.0], [1, 2, 3], terminated=True,
                                      discount_years=400.0)
    assert ok.tolist() == [True, True, True] and np.allclose(by_h[2], by_h[1])
    streams = {s: {"returns": [s, 0, 0, 0], "horizon": 2, "terminated": False} for s in (1, 2, 3)}
    summary = lsp.reeval_summary(streams, 3)
    assert bool(summary["cand_reeval_available"])
    assert np.isclose(summary["cand_reeval_se"][0], 1.0 / math.sqrt(3))
    assert not bool(lsp.reeval_summary({}, 3)["cand_reeval_available"])


def test_action_arrays_carry_raw_and_compiled_units() -> None:
    catalog = _catalog()
    payload = _base_payload(catalog)
    counts, activity = labor_state(payload, catalog)
    arrays = lsp.action_arrays(payload, counts, 20.0)
    assert set(arrays) == set(lsp.ACTION_KEYS)
    assert arrays["action_specialty_allocation"].shape == (len(SPECIALTIES),)
    assert arrays["action_activity_allocation"].shape == (len(SPECIALTIES) - 1, len(ACTIVITIES))
    assert int(arrays["action_population"]) == 100 == int(arrays["action_specialty_counts"].sum())
    assert np.allclose(arrays["action_worker_years"], counts[1:, None] * 20.0 * activity)
    assert np.isclose(arrays["action_worker_years"].sum(), 100 * 20.0)  # idle is a role column


def test_search_root_exports_every_candidate_and_reevaluates_final_survivors() -> None:
    config = SearchConfig(candidates=8, actor_samples=0, horizons=(1, 2), keep=("all", "2"))
    _, result, _ = _root(config)
    root, cands = result.payload_root, result.payload_candidates
    n = int(root["root_candidates"])
    assert len(cands) == n >= 3
    assert int(root["root_source"]) == lsp.ROOT_SOURCE_CODES["actor"]
    assert root["probe"].tolist() == [3.0, 4.0, 5.0]  # the focal row of the actor tensors
    assert np.allclose(root["labor_personality"], [0.3, 0.7])
    assert int(root["root_ranker_mode"]) == lsp.RANKER_OFF
    final = sorted(result.table["reeval_candidates"])
    assert 0 in final and len(final) == 3  # top two plus the protected H12
    for i, row in enumerate(cands):
        assert int(row["cand_index"]) == i and bool(row["cand_natively_evaluated"])
        assert bool(row["cand_is_h12"]) == (i == 0)
        reached = int(row["cand_horizon_reached"])
        assert row["cand_step_mask"].sum() == reached
        assert np.allclose(row["cand_step_years"][:reached], YEARS)
        assert bool(row["cand_return_available"][0])
        assert bool(row["cand_return_available"][1]) == (i in final)
        payload = result.table["candidates"][i]["payload"]
        v = FakeNative.value(payload)
        assert np.isclose(row["cand_returns_by_horizon"][0][0], v)
        assert bytes(row["cand_digest"]).hex() == labor_search.payload_digest(payload)
        assert bool(row["cand_reeval_available"]) == (i in final)
        if i in final:
            assert row["cand_reeval_horizon"].tolist() == [2, 2, 2]
            g = math.exp(-YEARS / config.discount_years)
            expected = [(v + 0.01 * s) * (1 + g) for s in (1, 2, 3)]
            assert np.allclose(row["cand_reeval_returns"][:, 0], expected)
            assert row["cand_reeval_se"][0] > 0
    # The label test still compares only the contenders with H12.
    for decision in result.table["decisions"]:
        if "reeval" in decision:
            assert decision["best_found"] != 0
    legacy = SearchConfig(candidates=8, actor_samples=0, horizons=(1, 2), keep=("all", "2"),
                          reeval_scope="winners")
    _, old, _ = _root(legacy)
    assert set(old.table["reeval_candidates"]) <= set(final)
    assert [d["label"] for d in old.table["decisions"]] == [
        d["label"] for d in result.table["decisions"]
    ]


class FakeRanker:
    """A = minus the candidate index on QoL: lower indices rank higher."""

    def __init__(self) -> None:
        self.calls = []

    def score(self, root, candidates):
        self.calls.append((dict(root), dict(candidates)))
        n = candidates["action_specialty_allocation"].shape[0]
        scores = torch.zeros(n, 4)
        scores[:, 0] = -torch.arange(n, dtype=torch.float32)
        return scores


def test_ranker_prunes_to_top_k_plus_h12_and_records_predictions() -> None:
    config = SearchConfig(candidates=10, actor_samples=0, horizons=(1, 2), keep=("all", "2"),
                          ranker=True, ranker_top=4, ranker_full_fraction=0.0)
    ranker = FakeRanker()
    native, result, _ = _root(config, ranker=ranker)
    (root_tensors, action_tensors), = ranker.calls
    assert set(action_tensors) == set(lsp.ACTION_KEYS)
    assert torch.allclose(root_tensors["labor_personality"], torch.tensor([[0.3, 0.7]]))
    n = int(result.payload_root["root_candidates"])
    assert n > 5 and int(result.payload_root["root_ranker_mode"]) == lsp.RANKER_PRUNED
    evaluated = [i for i, row in enumerate(result.payload_candidates)
                 if bool(row["cand_natively_evaluated"])]
    assert evaluated == [0, 1, 2, 3, 4]
    assert {c for c, _, _ in native.extended} <= set(evaluated)
    for i, row in enumerate(result.payload_candidates):
        assert bool(row["cand_predicted_available"])
        assert np.isclose(row["cand_predicted_advantage"][0], (n - 1) / 2 - i)  # centered
        if i not in evaluated:
            assert int(row["cand_horizon_reached"]) == 0 and int(row["cand_last_round"]) == -1
    full = SearchConfig(candidates=10, actor_samples=0, horizons=(1, 2), keep=("all", "2"),
                        ranker=True, ranker_top=4, ranker_full_fraction=1.0)
    _, everything, _ = _root(full, ranker=FakeRanker())
    assert int(everything.payload_root["root_ranker_mode"]) == lsp.RANKER_FULL
    assert all(bool(r["cand_natively_evaluated"]) for r in everything.payload_candidates)
    # Off by default: a ranker object without the declared configuration is not used.
    unused = FakeRanker()
    _, off, _ = _root(SearchConfig(candidates=10, actor_samples=0, horizons=(1, 2),
                                   keep=("all", "2")), ranker=unused)
    assert not unused.calls and int(off.payload_root["root_ranker_mode"]) == lsp.RANKER_OFF


def _fake_world(job):
    rows, roots, cands, tables = [], [], [], []
    for index in range(2):
        config = SearchConfig.from_mapping(job["config"])
        _, result, _ = _root(config, root_index=index, world=int(job["world"]))
        for row in result.rows:
            row["row_turn"] = torch.tensor([4 + index])
        result.payload_root["row_turn"] = np.asarray(4 + index)
        rows.extend(result.rows)
        tables.append(result.table)
        roots.append(result.payload_root)
        cands.extend(result.payload_candidates)
    return {
        "world": int(job["world"]), "rows": rows, "tables": tables, "skipped": {},
        "reset_identity": {}, "root_turn_indices": [4, 5], "restore_checks": [],
        "work": {}, "main_path_resolves": 0, "wall_seconds": 0.0,
        "payload_roots": roots, "payload_candidates": cands,
    }


def _publish(tmp_path: Path, monkeypatch, *, partition: str = "fitting", world: int = 90100):
    monkeypatch.setattr(labor_search, "search_world", _fake_world)
    out = tmp_path / "search"
    out.mkdir()
    config = SearchConfig(candidates=6, actor_samples=0, horizons=(1, 2), keep=("all", "2"))
    body = {"schema": 1, "source_kind": labor_search.SOURCE_KIND,
            "options": {"turns": 8, "reset_options": {}, "partition": partition}}
    source = dict(body, source_run=digest(body))
    job = {"world": world, "out": str(out), "source_run": source["source_run"],
           "config": config.state_dict(), "turns": 8, "reset_options": {},
           "normalizer_digest": "n", "source": "h12", "partition": partition}
    summary = labor_search.write_search_shard(job)
    ltd.publish_dataset(out, source, [
        {k: summary[k] for k in ("path", "world_seed", "meta_sha256", "data_digest", "rows")}
    ])
    return out


def test_shard_binds_and_loader_reads_the_payload(tmp_path, monkeypatch) -> None:
    out = _publish(tmp_path, monkeypatch)
    shard = out / "world-90100"
    meta = ltd.verify_shard(shard)
    binding = meta["candidate_payload"]
    assert binding["schema"] == lsp.PAYLOAD_SCHEMA and binding["schema_version"] == 1
    assert binding["horizons"] == [1, 2] and binding["components"] == list(COMPONENTS)
    assert not (out / "world-90100.payload-staging").exists()
    loaded = lsp.load_search_payload(out)
    roots, cands = loaded["roots"], loaded["candidates"]
    assert roots["root_index"].tolist() == [0, 1]
    assert set(cands["cand_root"].tolist()) == {0, 1}
    assert len(cands["cand_index"]) == int(roots["root_candidates"].sum())
    # Label rows are unchanged and still load for actor BC.
    labels = ltd.load_labor_dataset(out)
    assert labels["collection"]["source_kind"] == labor_search.SOURCE_KIND
    # Tampering with a payload table is detected.
    np.save(shard / lsp.CANDIDATES_DIR / "cand_index.npy",
            np.zeros_like(cands["cand_index"].numpy()))
    with pytest.raises(ltd.ProvenanceError):
        lsp.load_search_payload(out)


def test_held_out_test_payload_is_refused_for_training(tmp_path, monkeypatch) -> None:
    out = _publish(tmp_path, monkeypatch, partition=HELD_OUT_TEST_PARTITION, world=90155)
    with pytest.raises(ValueError, match="held-out-test"):
        lsp.load_search_payload(out)
    with pytest.raises(ValueError, match="held-out-test"):
        ltd.load_labor_dataset(out)
    loaded = lsp.load_search_payload(out, training=False)
    assert loaded["collection"]["partition"] == HELD_OUT_TEST_PARTITION
    assert loaded["collection"]["world_seeds"] == [90155]
