"""Search candidate-payload export (labor-advantage-candidates-v1) and the A-ranker hook.

A fake native search (deterministic per-candidate rewards, no bridge) drives
the real ``search_root`` so the payload, the final-survivor re-evaluation and
the ranker pruning are checked end to end; a monkeypatched world driver checks
the shard binding, verification and the training guard. When the model side's
``labor_advantage`` is importable its strict reader parses the exported shard.
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
from anthropocene_rl.autonomous_v7.labor_bridge import SPECIALTIES
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
        step={"turn": 4 + root_index, "villages": [village]},
        root_h12={FOCAL: h12},
        focal=FOCAL,
        founded={FOCAL: 0},
        tensors={"core": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
        row=1,
        root_pair=(0.3, 0.7),
        actor=None,
        root_fork=0,
        key=[world, 0, 4 + root_index, FOCAL],
        root_source="actor",
        root_index=root_index,
        ranker=ranker,
    )
    return native, result, catalog


def _evaluations(candidate, stream=None):
    return [e for e in candidate["evaluations"] if stream is None or e["stream"] == stream]


def test_movement_union_and_compiled_units() -> None:
    assert len(lsp.MOVEMENT_UNION_CODES) == 17
    assert lsp.movement_slot(-1) == 0 and lsp.movement_slot(107) == 16
    with pytest.raises(ValueError):
        lsp.movement_slot(-2)  # the forbidden native sentinel is not an actor action
    catalog = _catalog()
    payload = _base_payload(catalog)
    counts, activity = labor_state(payload, catalog)
    compiled = lsp.compiled_action(payload, counts, 20.0)
    assert compiled["population"] == 100 == sum(compiled["specialty_counts"])
    assert np.allclose(compiled["worker_years"], counts[1:, None] * 20.0 * activity)
    assert np.isclose(np.sum(compiled["worker_years"]), 100 * 20.0)
    assert len(compiled["specialty_counts"]) == len(SPECIALTIES)


def test_selection_evaluations_per_horizon() -> None:
    steps = [[1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 0.5, 0.0]]
    g = math.exp(-20.0 / 400.0)
    evals = lsp.selection_evaluations(steps, [20.0, 20.0], [1, 2, 3], terminated=False,
                                      discount_years=400.0)
    assert [e["horizon"] for e in evals] == [1, 2]  # horizon 3 was not reached
    assert np.allclose(evals[1]["returns"], [1 + g, 2 + 2 * g, 0.5 * g, 0.0])
    assert evals[1]["step_years"] == [20.0, 20.0] and not evals[1]["terminated"]
    ended = lsp.selection_evaluations(steps, [20.0, 20.0], [1, 2, 3], terminated=True,
                                      discount_years=400.0)
    assert [(e["horizon"], e["reached_horizon"], e["terminated"]) for e in ended] == [
        (1, 1, False), (2, 2, True), (3, 2, True)]
    assert np.allclose(ended[2]["returns"], ended[1]["returns"])


def test_search_root_exports_every_candidate_and_reevaluates_final_survivors() -> None:
    config = SearchConfig(candidates=8, actor_samples=0, horizons=(1, 2), keep=("all", "2"))
    _, result, _ = _root(config)
    payload = result.payload_root
    record = payload["record"]
    assert set(payload["tensors"]) == {"core", "labor_personality"}
    assert payload["tensors"]["core"].tolist() == [[3.0, 4.0, 5.0]]  # the focal row
    assert payload["tensors"]["labor_personality"].tolist() == [[pytest.approx(0.3),
                                                                 pytest.approx(0.7)]]
    assert payload["privileged"] is None  # the fake village has no privileged view
    assert record["source"] == "actor" and record["full_evaluation"]
    assert record["ranker_mode"] == lsp.RANKER_OFF and record["final_horizon"] == 2
    cands = record["candidates"]
    final = sorted(result.table["reeval_candidates"])
    assert 0 in final and len(final) == 3  # top two plus the protected H12
    g = math.exp(-YEARS / config.discount_years)
    for i, cand in enumerate(cands):
        assert cand["index"] == i and cand["is_h12"] == (i == 0)
        assert cand["digest"] == labor_search.payload_digest(cand["action"])
        assert cand["natively_evaluated"] and cand["control"] is None
        v = FakeNative.value(cand["action"])
        selection = _evaluations(cand, 0)
        assert selection[0]["horizon"] == 1 and np.isclose(selection[0]["returns"][0], v)
        assert len(selection) == (2 if i in final else 1)
        streams = _evaluations(cand)[len(selection):]
        if i in final:
            assert [e["stream"] for e in streams] == [1, 2, 3]
            assert all(e["horizon"] == 2 == e["reached_horizon"] for e in streams)
            expected = [(v + 0.01 * s) * (1 + g) for s in (1, 2, 3)]
            assert np.allclose([e["returns"][0] for e in streams], expected)
        else:
            assert streams == []
    for decision in result.table["decisions"]:
        if "reeval" in decision:  # the label test still compares contenders with H12
            assert decision["best_found"] != 0
    legacy = SearchConfig(candidates=8, actor_samples=0, horizons=(1, 2), keep=("all", "2"),
                          reeval_scope="winners")
    _, old, _ = _root(legacy)
    assert set(old.table["reeval_candidates"]) <= set(final)
    assert [d["label"] for d in old.table["decisions"]] == [
        d["label"] for d in result.table["decisions"]
    ]


class FakeRanker:
    """AdvantageRanker.plan stand-in: A = minus the candidate index on QoL."""

    def __init__(self, *, full: bool = False, top: int = 4) -> None:
        self.full, self.top, self.calls = full, top, []

    def plan(self, root_tensors, actions, *, h12_index, personalities, privileged_raw,
             root_key):
        self.calls.append((dict(root_tensors), list(actions), privileged_raw, list(root_key)))
        n = len(actions)
        centred = [(n - 1) / 2 - i for i in range(n)]
        advantage = {c: (centred if c == "qol_years" else [0.0] * n) for c in COMPONENTS}
        evaluate = list(range(n)) if self.full else sorted({h12_index, *range(1, 1 + self.top)})
        return SimpleNamespace(advantage=advantage, evaluate=evaluate, full_evaluation=self.full)


def test_ranker_prunes_to_top_k_plus_h12_and_records_predictions() -> None:
    config = SearchConfig(candidates=10, actor_samples=0, horizons=(1, 2), keep=("all", "2"),
                          ranker=True)
    ranker = FakeRanker()
    native, result, _ = _root(config, ranker=ranker)
    (root_tensors, actions, privileged, _), = ranker.calls
    assert torch.allclose(root_tensors["labor_personality"], torch.tensor([[0.3, 0.7]]))
    assert privileged is None and actions[0] == result.payload_root["record"]["candidates"][0][
        "action"]
    record = result.payload_root["record"]
    n = len(record["candidates"])
    assert n > 5 and record["ranker_mode"] == lsp.RANKER_PRUNED and not record["full_evaluation"]
    evaluated = [c["index"] for c in record["candidates"] if c["natively_evaluated"]]
    assert evaluated == [0, 1, 2, 3, 4]
    assert {c for c, _, _ in native.extended} <= set(evaluated)
    for cand in record["candidates"]:
        assert np.isclose(cand["predicted_advantage"][0], (n - 1) / 2 - cand["index"])
        if cand["index"] not in evaluated:
            assert cand["evaluations"] == [] and cand["last_round"] == -1
    _, everything, _ = _root(config, ranker=FakeRanker(full=True))
    full = everything.payload_root["record"]
    assert full["ranker_mode"] == lsp.RANKER_FULL and full["full_evaluation"]
    assert all(c["natively_evaluated"] for c in full["candidates"])
    # Off by default: a ranker object without the declared configuration is not used.
    unused = FakeRanker()
    _, off, _ = _root(SearchConfig(candidates=10, actor_samples=0, horizons=(1, 2),
                                   keep=("all", "2")), ranker=unused)
    assert not unused.calls and off.payload_root["record"]["ranker_mode"] == lsp.RANKER_OFF


def _fake_world(job):
    rows, roots, tables = [], [], []
    for index in range(2):
        config = SearchConfig.from_mapping(job["config"])
        _, result, _ = _root(config, root_index=index, world=int(job["world"]))
        rows.extend(result.rows)
        tables.append(result.table)
        roots.append(result.payload_root)
    return {
        "world": int(job["world"]), "rows": rows, "tables": tables, "skipped": {},
        "reset_identity": {}, "root_turn_indices": [4, 5], "restore_checks": [],
        "work": {}, "main_path_resolves": 0, "wall_seconds": 0.0, "payload_roots": roots,
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


def test_shard_binds_the_payload_in_the_model_schema(tmp_path, monkeypatch) -> None:
    out = _publish(tmp_path, monkeypatch)
    shard_dir = out / "world-90100"
    meta = ltd.verify_shard(shard_dir)
    binding = meta["candidate_payload"]
    assert binding["schema"] == lsp.CANDIDATE_PAYLOAD_SCHEMA == "labor-advantage-candidates-v1"
    assert binding["roots"] == 2 and binding["file"] == lsp.PAYLOAD_FILE
    assert not (out / "world-90100.payload-staging").exists()
    (path,) = lsp.candidate_shard_paths(out)
    shard = lsp.read_candidate_shard(path)
    assert shard["component_order"] == list(COMPONENTS)
    assert shard["returns_units"] == "raw-discounted-component-return"
    assert shard["discount_years"] == 400.0
    assert [r["tensor_row"] for r in shard["roots"]] == [0, 1]
    assert shard["tensors"]["core"].shape == (2, 3)
    assert shard["producer"]["export_schema"] == lsp.EXPORT_SCHEMA
    assert binding["candidates"] == sum(len(r["candidates"]) for r in shard["roots"])
    # Label rows still load for actor BC.
    assert ltd.load_labor_dataset(out)["collection"]["source_kind"] == labor_search.SOURCE_KIND
    try:
        from anthropocene_rl.autonomous_v7 import labor_advantage
    except ImportError:
        labor_advantage = None
    if labor_advantage is not None:  # the model side's strict reader
        parsed = labor_advantage.load_candidate_shards([shard_dir])
        assert len(parsed.roots) == 2
        assert labor_advantage.CANDIDATE_PAYLOAD_SCHEMA == lsp.CANDIDATE_PAYLOAD_SCHEMA
    # Tampering with the payload is detected.
    torch.save({"schema": "x"}, path)
    with pytest.raises(ltd.ProvenanceError):
        lsp.candidate_shard_paths(out)


def test_held_out_test_payload_is_refused_for_training(tmp_path, monkeypatch) -> None:
    out = _publish(tmp_path, monkeypatch, partition=HELD_OUT_TEST_PARTITION, world=90155)
    with pytest.raises(ValueError, match="held-out-test"):
        lsp.candidate_shard_paths(out)
    with pytest.raises(ValueError, match="held-out-test"):
        ltd.load_labor_dataset(out)
    (path,) = lsp.candidate_shard_paths(out, training=False)
    assert lsp.read_candidate_shard(path)["producer"]["partition"] == HELD_OUT_TEST_PARTITION


def test_shard_assembly_refuses_malformed_roots() -> None:
    config = SearchConfig(candidates=6, actor_samples=0, horizons=(1, 2), keep=("all", "2"))
    _, a, _ = _root(config, root_index=0)
    _, b, _ = _root(config, root_index=0)
    with pytest.raises(ValueError, match="tensor_row"):
        lsp.candidate_shard([a.payload_root, b.payload_root], producer={}, discount_years=400.0)
    leaked = dict(a.payload_root, tensors=dict(a.payload_root["tensors"],
                                               privileged_core=torch.zeros(1, 1)))
    with pytest.raises(ValueError, match="privileged keys"):
        lsp.candidate_shard([leaked], producer={}, discount_years=400.0)
    with pytest.raises(ValueError, match="400-year"):
        lsp.candidate_shard([a.payload_root], producer={}, discount_years=300.0)


def test_control_candidates_idle_delayed_and_twin() -> None:
    from anthropocene_rl.autonomous_v7.labor_search import compose_payload, control_candidates

    catalog = _catalog()
    base = _base_payload(catalog)
    families = {c.family: c for c in control_candidates(base, catalog)}
    assert set(families) == {"control-idle", "control-twin"}  # H12 already staffs Farmers
    idle_counts, _ = labor_state(families["control-idle"].payload, catalog)
    assert int(idle_counts[0]) == 100
    twin_counts, twin_activity = labor_state(families["control-twin"].payload, catalog)
    base_counts, base_activity = labor_state(base, catalog)
    assert np.array_equal(twin_counts, base_counts)  # same residents: physics of H12
    differs = np.flatnonzero(np.abs(twin_activity - base_activity).sum(axis=1) > 0)
    assert len(differs) == 1 and base_counts[differs[0] + 1] == 0  # only an unstaffed role
    farmer = SPECIALTIES.index("Farmer")
    counts = base_counts.copy()
    counts[SPECIALTIES.index("Gatherer")] += counts[farmer]
    counts[farmer] = 0
    activity = base_activity.copy()
    activity[farmer - 1] = 0.0
    activity[farmer - 1, 0] = 1.0
    no_farm = compose_payload(base, catalog, counts=counts, activity=activity)
    delayed = {c.family: c for c in control_candidates(no_farm, catalog)}["control-delayed"]
    assert labor_state(delayed.payload, catalog)[0][farmer] == 10


def test_control_roots_carry_tagged_controls_to_the_final_horizon() -> None:
    config = SearchConfig(candidates=10, actor_samples=0, horizons=(1, 2, 3),
                          keep=("all", "2", "1"), control_fraction=1.0)
    _, result, _ = _root(config)
    record = result.payload_root["record"]
    assert record["control_root"]
    by_family = {c["family"]: c for c in record["candidates"]}
    assert by_family["control-idle"]["control"] == "all-idle"
    assert by_family["control-twin"]["bookkeeping_twin"] == 0
    for name in ("control-idle", "control-twin"):
        cand = by_family[name]
        assert cand["horizon_reached"] == 3  # protected through halving like H12
        assert [e["stream"] for e in cand["evaluations"] if e["horizon"] == 3] == [0, 1, 2, 3]
    off = SearchConfig(candidates=10, actor_samples=0, horizons=(1, 2), keep=("all", "2"))
    _, plain, _ = _root(off)
    assert not plain.payload_root["record"]["control_root"]
    assert not any(c["family"].startswith("control") for c in
                   plain.payload_root["record"]["candidates"])
