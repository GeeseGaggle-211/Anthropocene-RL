"""T2 plan candidates and preemption-safe --resume for the labor search teacher.

Plans: the pure templates conserve residents and worker-time at every step; a
policy-aware fake native search drives the real ``search_root`` so the plan and
paired first-step-control rows, their per-step returns and the candidate budget
are checked end to end, then round-tripped through the payload writer and the
A-head's strict reader (old shards still parse; plan rows stay out of the
first-action candidates).

Resume: ``run_search_jobs`` with a monkeypatched world driver is interrupted mid
world (a staged shard left behind) and resumed: the verified shard is kept
byte-for-byte, the partial world is redone into a fresh shard, untouched worlds
run, and dataset.json appears only once every world is complete. A changed source
record is refused.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from anthropocene_rl.autonomous_v7 import labor_search
from anthropocene_rl.autonomous_v7 import labor_search_payload as lsp
from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.labor_bridge import ACTIVITIES, SPECIALTIES
from anthropocene_rl.autonomous_v7.labor_search import (
    PLAN_TEMPLATES,
    Branch,
    SearchConfig,
    conserves_labor,
    labor_state,
    plan_specs,
    plan_step_payload,
    search_root,
)
from anthropocene_rl.autonomous_v7.provenance import ProvenanceError, checkpoint_hash, digest
from tests.test_labor_search import _base_payload, _catalog, _h12_label, _observation
from tests.test_labor_search_payload import FOCAL, YEARS, FakeNative, _fake_world

S = {name: i for i, name in enumerate(SPECIALTIES)}
A = {name: i for i, name in enumerate(ACTIVITIES)}


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def _shifted(catalog, base, population_shift: int = 0):
    """An H12 action at a later state: fewer Farmers/Craftspersons, idle Gatherer time."""
    counts, activity = labor_state(base, catalog)
    counts = counts.copy()
    counts[S["Farmer"]] -= 6
    counts[S["Gatherer"]] += 6
    activity = activity.copy()
    activity[S["Gatherer"] - 1] = 0.0
    activity[S["Gatherer"] - 1, A["food.gather"]] = 0.8
    activity[S["Gatherer"] - 1, 0] = 0.2
    return labor_search.compose_payload(base, catalog, counts=counts, activity=activity)


@pytest.mark.parametrize("template", PLAN_TEMPLATES)
def test_plan_steps_conserve_and_hold_their_course(template) -> None:
    catalog = _catalog()
    base = _base_payload(catalog)
    built = plan_step_payload(template, 0, base, catalog, amount=10)
    assert built is not None
    first, _ = built
    assert conserves_labor(first, catalog)
    spec = S[labor_search.PLAN_SPECIALTY[template]]
    counts0 = labor_state(first, catalog)[0]
    assert counts0[spec] == labor_state(base, catalog)[0][spec] + 10
    assert counts0[S["Gatherer"]] == labor_state(base, catalog)[0][S["Gatherer"]] - 10
    later = _shifted(catalog, base)
    for step in range(1, 4):
        payload, description = plan_step_payload(
            template, step, later, catalog, amount=10, anchor=int(counts0[spec])
        )
        assert conserves_labor(payload, catalog), description
        counts, activity = labor_state(payload, catalog)
        assert int(counts.sum()) == catalog.population
        assert np.allclose(activity.sum(axis=1), 1.0)
        if template == "housing-grow":
            assert payload["fission"] == labor_search.DISCOURAGE_FISSION
            assert activity[S["Gatherer"] - 1, 0] == 0.0  # idle time to food.gather
        else:
            assert counts[spec] >= counts0[spec]  # the adopted specialists are held
            expansion = A[labor_search.PLAN_EXPANSION[template]]
            assert activity[spec - 1, expansion] >= 0.5 - 1e-9


def test_housing_first_step_mixes_housing_and_material() -> None:
    catalog = _catalog()
    base = _base_payload(catalog)
    payload, _ = plan_step_payload("housing-grow", 0, base, catalog, amount=10)
    counts, activity = labor_state(payload, catalog)
    row = activity[S["Craftsperson"] - 1]
    # 10 existing (0.6 housing, 0.4 cloth) + 10 new (0.5 housing, 0.5 material).
    assert counts[S["Craftsperson"]] == 20
    assert math.isclose(row[A["goods.housing"]], 0.55)
    assert math.isclose(row[A["goods.housing-material"]], 0.25)
    assert math.isclose(row[A["goods.cloth"]], 0.2)


def test_plan_specs_are_keyed_and_bounded() -> None:
    catalog = _catalog()
    base = _base_payload(catalog)
    a = plan_specs(base, catalog, count=2, steps=3, key=[1, 2, 3])
    b = plan_specs(base, catalog, count=2, steps=3, key=[1, 2, 3])
    assert [s.plan_id for s in a] == [s.plan_id for s in b] and len(a) == 2
    assert len({s.template for s in a}) == 2
    assert plan_specs(base, catalog, count=0, steps=3, key=[1]) == []
    every = plan_specs(base, catalog, count=5, steps=2, key=[9])
    assert sorted(s.template for s in every) == sorted(PLAN_TEMPLATES)
    assert all(s.plan_id.endswith("/P2") for s in every)


def test_plan_budget_counts_toward_m() -> None:
    assert SearchConfig(candidates=16, plan_candidates=2).single_budget == 12
    with pytest.raises(ValueError, match="candidate budget"):
        SearchConfig(candidates=5, plan_candidates=2)
    assert SearchConfig().state_dict()["plan_candidates"] == 0  # programmatic default
    parser = labor_search.argument_parser()
    args = parser.parse_args(["--normalizer-checkpoint", "n", "--worlds", "90100", "--out", "o"])
    assert args.plan_candidates == 2 and args.plan_steps == 3 and not args.resume


# ---------------------------------------------------------------------------
# search_root with a policy-aware fake
# ---------------------------------------------------------------------------


class PlanFake(FakeNative):
    """FakeNative that honours focal_policy and records every executed focal action."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.executed: list[tuple[int, int, int, dict]] = []

    def extend(
        self, branch: Branch, target: int, *, candidate, keep_fork, focal_policy=None,
        root_h12=None, focal=None, **_,
    ) -> None:
        self.extended.append((branch.candidate, branch.stream, target))
        while branch.horizon < target and branch.alive:
            payload = candidate
            if branch.horizon > 0:
                chosen = None
                if focal_policy is not None:
                    chosen = focal_policy(branch.horizon, root_h12[focal], self._catalog, {})
                payload = chosen if chosen is not None else root_h12[focal].action
            self.executed.append((branch.candidate, branch.stream, branch.horizon, dict(payload)))
            v = self.value(payload)
            r = np.asarray([v + 0.01 * branch.stream, 10.0 * v, 0.0, 0.0])
            branch.returns = branch.returns + branch.discount * r
            branch.steps.append(r.tolist())
            branch.years.append(YEARS)
            branch.discount *= math.exp(-YEARS / self.config.discount_years)
            branch.horizon += 1
            self.work.labor_steps += 1
            self.work.shadow_steps += 1


def _plan_root(config: SearchConfig, *, world: int = 90100, root_index: int = 0):
    catalog = _catalog()
    base = _base_payload(catalog)
    h12 = _h12_label(catalog, base, trade_valid=True)
    village = {"camp": FOCAL, "society": 3, "observation": _observation()}
    native = PlanFake(catalog, config, world)
    import torch

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
        root_source="h12",
        root_index=root_index,
    )
    return native, result, catalog, h12


def test_search_root_exports_plans_with_paired_controls() -> None:
    config = SearchConfig(
        candidates=10, actor_samples=0, horizons=(1, 2), keep=("all", "2"),
        plan_candidates=2, plan_steps=3, reeval_streams=2,
    )
    native, result, catalog, h12 = _plan_root(config)
    record = result.payload_root["record"]
    rows = record["candidates"]
    singles = [c for c in rows if not c.get("plan_role")]
    plans = [c for c in rows if c.get("plan_role")]
    assert len(singles) <= config.single_budget == 6 and len(rows) <= config.candidates
    assert len(plans) == 4 and [c["index"] for c in rows] == list(range(len(rows)))
    assert rows[len(singles):] == plans  # plan rows follow every first-action candidate
    assert record["plan_candidates"] == 2 and record["plan_horizon"] == 3  # max(final 2, P 3)
    for plan, control in zip(plans[::2], plans[1::2], strict=True):
        assert plan["plan_role"] == "plan" and control["plan_role"] == "first-step-control"
        assert plan["family"] == "plan" and control["family"] == "plan-control"
        assert plan["plan_id"] == control["plan_id"]
        assert plan["action"] == control["action"] and plan["digest"] == control["digest"]
        assert len(plan["plan_steps"]) == 3 and len(control["plan_steps"]) == 1
        for entry in plan["plan_steps"]:
            assert entry["valid"] and entry["fallback"] is None and entry["conserves"]
            assert conserves_labor(entry["action"], catalog)
            assert sum(entry["specialty_counts"]) == catalog.population
        # Streams 0..S, per-step returns: the shared first step is identical, later
        # plan steps execute the plan's (different) actions, the control H12.
        assert sorted(plan["step_returns"]) == ["0", "1", "2"]
        for s in ("0", "1", "2"):
            p, c = plan["step_returns"][s], control["step_returns"][s]
            assert len(p) == len(c) == 3 and p[0] == c[0] and p[1] != c[1]
        streams = sorted({e["stream"] for e in plan["evaluations"]})
        assert streams == [0, 1, 2]
        assert sorted(e["horizon"] for e in plan["evaluations"] if e["stream"] == 0) == [1, 2, 3]
    # Every executed focal action (plan and control, every stream) conserves and the
    # control executes H12 after the shared first step.
    control_rows = {c["index"] for c in plans if c["plan_role"] == "first-step-control"}
    for cand, _stream, horizon, payload in native.executed:
        assert conserves_labor(payload, catalog)
        if cand in control_rows and horizon >= 1:
            assert payload == h12.action
    # Plans never enter halving or the label decision.
    assert all(d["label"] < len(singles) for d in result.table["decisions"])
    table_plans = result.table["plans"]
    assert len(table_plans) == 2 and result.table["work"]["plan_resolves"] > 0
    # Payload round trip through the writer and the A-head's strict reader.
    shard = lsp.candidate_shard(
        [result.payload_root], producer={"test": True}, discount_years=400.0
    )
    return_trip(shard, singles=len(singles), plans=4)


def return_trip(shard, *, singles: int, plans: int, tmp: Path | None = None) -> None:
    import tempfile

    from anthropocene_rl.autonomous_v7 import labor_advantage as la

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / lsp.PAYLOAD_FILE
        binding = lsp.write_candidate_shard(path, shard)
        assert binding["plan_rows"] == plans
        loaded = lsp.read_candidate_shard(path)
    assert loaded["roots"] == shard["roots"]
    parsed = la.parse_candidate_shard(loaded)
    (root,) = parsed.roots
    assert len(root.candidates) == singles and len(root.plans) == plans
    assert [c.index for c in root.candidates] == list(range(singles))
    assert {p.plan_role for p in root.plans} == ({"plan", "first-step-control"} if plans else set())
    for p in root.plans:
        assert p.plan_id and set(p.step_returns) == {0, 1, 2}
        assert all(v.shape[1] == 4 for v in p.step_returns.values())
    la.build_pairs(parsed.roots, "qol_years")  # plan rows never enter pairs


def test_old_shards_without_plans_still_parse() -> None:
    config = SearchConfig(candidates=6, actor_samples=0, horizons=(1, 2), keep=("all", "2"))
    _, result, _, _ = _plan_root(config)
    rows = result.payload_root["record"]["candidates"]
    assert not any("plan_role" in c for c in rows)
    shard = lsp.candidate_shard([result.payload_root], producer={}, discount_years=400.0)
    return_trip(shard, singles=len(rows), plans=0)


def test_reader_refuses_misplaced_plan_rows() -> None:
    from anthropocene_rl.autonomous_v7 import labor_advantage as la

    config = SearchConfig(
        candidates=8, actor_samples=0, horizons=(1, 2), keep=("all", "2"), plan_candidates=1,
        plan_steps=2,
    )
    _, result, _, _ = _plan_root(config)
    shard = lsp.candidate_shard([result.payload_root], producer={}, discount_years=400.0)
    rows = shard["roots"][0]["candidates"]
    moved = [rows[-1], *rows[:-1]]
    for i, row in enumerate(moved):
        row["index"] = i
    moved[0]["is_h12"] = False
    shard["roots"][0]["candidates"] = moved
    with pytest.raises(ValueError):
        la.parse_candidate_shard(shard)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

WORLDS = (90100, 90101, 90102)
_REAL_WRITE_SHARD = ltd.write_shard


def _source(tag: str = "a") -> dict:
    body = {"schema": 1, "source_kind": labor_search.SOURCE_KIND, "tag": tag,
            "options": {"turns": 8, "reset_options": {}, "partition": "fitting"}}
    return dict(body, source_run=digest(body))


def _jobs(out: Path, source: dict, worlds=WORLDS) -> list[dict]:
    config = SearchConfig(
        candidates=8, actor_samples=0, horizons=(1, 2), keep=("all", "2"), plan_candidates=1,
        plan_steps=2,
    )
    return [
        {"world": w, "out": str(out), "source_run": source["source_run"],
         "config": config.state_dict(), "turns": 8, "reset_options": {},
         "normalizer_digest": "n", "source": "h12", "partition": "fitting"}
        for w in worlds
    ]


class Killed(RuntimeError):
    pass


def _counting(monkeypatch, *, kill_world: int | None = None, empty_world: int | None = None):
    calls: list[int] = []

    def world(job):
        calls.append(int(job["world"]))
        result = _fake_world(job)
        if int(job["world"]) == empty_world:
            result.update(rows=[], payload_roots=[], tables=[])
        return result

    monkeypatch.setattr(labor_search, "search_world", world)
    real = _REAL_WRITE_SHARD

    def write_shard(directory, arrays, meta):
        record = real(directory, arrays, meta)
        if int(meta["world_seed"]) == kill_world:
            raise Killed("preempted after staging the shard")  # mid-world kill
        return record

    monkeypatch.setattr(ltd, "write_shard", write_shard)
    return calls


def test_resume_after_a_mid_world_kill(tmp_path, monkeypatch) -> None:
    out = tmp_path / "search"
    source = _source()
    jobs = _jobs(out, source)
    calls = _counting(monkeypatch, kill_world=90101)
    with pytest.raises(Killed):
        labor_search.run_search_jobs(out, source, jobs)
    assert calls == [90100, 90101]
    assert (out / "world-90100" / "meta.json").is_file()
    assert (out / "world-90101.staging").is_dir() and not (out / "world-90101").exists()
    assert not (out / "dataset.json").exists()
    kept = {p.name: checkpoint_hash(p) for p in (out / "world-90100").iterdir()}

    # Without --resume the existing output is refused.
    with pytest.raises(ProvenanceError, match="--resume"):
        labor_search.run_search_jobs(out, source, jobs)
    # A changed source record (code, bridge, checkpoint or options) is refused.
    other = _source("b")
    with pytest.raises(ProvenanceError, match="source record differs"):
        labor_search.run_search_jobs(out, other, _jobs(out, other), resume=True)

    calls = _counting(monkeypatch)
    report = labor_search.run_search_jobs(out, source, jobs, resume=True)
    assert calls == [90101, 90102]  # published world skipped, partial world redone
    assert report["resumed_worlds"] == [90100] and report["worlds_this_run"] == [90101, 90102]
    assert not (out / "world-90101.staging").exists()
    assert {p.name: checkpoint_hash(p) for p in (out / "world-90100").iterdir()} == kept
    manifest = ltd.verify_dataset(out)
    assert sorted(manifest["shards"]) == [str(w) for w in WORLDS]
    assert len(lsp.candidate_shard_paths(out)) == 3
    assert report["roots"] == 6  # resumed world's tables are in the report

    # Rerunning a complete output with --resume republishes the same manifest.
    calls = _counting(monkeypatch)
    again = labor_search.run_search_jobs(out, source, jobs, resume=True)
    assert calls == [] and again["resumed_worlds"] == list(WORLDS)
    assert ltd.verify_dataset(out)["dataset_digest"] == manifest["dataset_digest"]


def test_resume_redoes_a_corrupt_shard_and_keeps_empty_worlds(tmp_path, monkeypatch) -> None:
    out = tmp_path / "search"
    source = _source()
    jobs = _jobs(out, source)
    calls = _counting(monkeypatch, empty_world=90102)
    labor_search.run_search_jobs(out, source, jobs)
    assert calls == list(WORLDS)
    marker = labor_search.empty_marker(out, 90102)
    assert json.loads(marker.read_text())["source_run"] == source["source_run"]
    assert sorted(ltd.verify_dataset(out)["shards"]) == ["90100", "90101"]
    # Simulate an interrupted run whose shard bytes were damaged; dataset.json
    # would not exist for an unfinished run.
    (out / "dataset.json").unlink()
    (out / "world-90101" / "candidates.jsonl").write_text("{}\n")
    calls = _counting(monkeypatch, empty_world=90102)
    report = labor_search.run_search_jobs(out, source, jobs, resume=True)
    assert calls == [90101]  # the corrupt shard is redone; the empty world stays done
    assert report["resumed_worlds"] == [90100, 90102]
    assert sorted(ltd.verify_dataset(out)["shards"]) == ["90100", "90101"]


def test_resume_refuses_undeclared_directories(tmp_path, monkeypatch) -> None:
    out = tmp_path / "search"
    source = _source()
    _counting(monkeypatch, kill_world=90100)
    with pytest.raises(Killed):
        labor_search.run_search_jobs(out, source, _jobs(out, source))
    (out / "world-99999").mkdir()
    with pytest.raises(ProvenanceError, match="undeclared"):
        labor_search.run_search_jobs(out, source, _jobs(out, source), resume=True)
