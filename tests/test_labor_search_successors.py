"""Successor critic-input rows in the search candidate payload (export v2, --successor-rows).

A fake native search drives the real ``search_root`` and calls the real
``NativeSearch.record_boundary`` at every decision boundary, so the successor streams,
their row indices, statuses and per-stream rewards are checked end to end through the
shard assembly, the writer and the strict model-side reader. With the flag off the export
is the v1 payload unchanged; with it on every v1 field keeps its value.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_search
from anthropocene_rl.autonomous_v7 import labor_search_payload as lsp
from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.labor_search import Branch, SearchConfig, search_root
from anthropocene_rl.autonomous_v7.provenance import digest
from tests.test_labor_search import _base_payload, _catalog, _h12_label, _observation
from tests.test_labor_search_payload import FOCAL, YEARS, FakeNative

DIES = 0.25  # candidates with value below this terminate at decision 2


class SuccessorFake(FakeNative):
    """FakeNative whose branches pass each boundary to the real ``record_boundary``."""

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
            terminal = branch.horizon == 2 and v < DIES
            if terminal:
                branch.terminated = True
                branch.alive = False
            turn = 4 + branch.horizon
            village = {
                "camp": FOCAL,
                "observation": {"v": v, "stream": branch.stream, "decision": branch.horizon},
                "terminated": terminal,
            }
            step = {"turn": turn, "villages": [village]}
            if branch.record and self.successor_view is not None:
                self.record_boundary(branch, step, FOCAL, terminal=terminal, live=not terminal)


def _view(village):
    o = village["observation"]
    return {
        "tensors": {"core": torch.tensor([[o["v"], float(o["stream"]), float(o["decision"])]])},
        "privileged": None,
    }


def _root(config: SearchConfig, *, root_index: int = 0, world: int = 90100, view=True):
    catalog = _catalog()
    base = _base_payload(catalog)
    h12 = _h12_label(catalog, base, trade_valid=True)
    village = {"camp": FOCAL, "society": 3, "observation": _observation()}
    native = SuccessorFake(catalog, config, world)
    if view:
        native.successor_view = _view
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
    return native, result


def _config(successors: bool) -> SearchConfig:
    return SearchConfig(
        candidates=8,
        actor_samples=0,
        horizons=(1, 3),
        keep=("all", "3"),
        reeval_streams=2,
        successor_rows=successors,
    )


PRODUCER = {"exporter": "labor_search", "world": 90100}
NEW_ROOT_KEYS = {"successor_decisions", "successor_rows"}
NEW_CAND_KEYS = {"successor_streams"}
NEW_SHARD_KEYS = {"successor_tensors", "successor_privileged", "successor_row_count"}


def _shard(successors: bool, roots: int = 2) -> dict:
    payloads = [_root(_config(successors), root_index=i)[1].payload_root for i in range(roots)]
    return lsp.candidate_shard(payloads, producer=PRODUCER, discount_years=400.0)


def _strip_v2(shard: dict) -> dict:
    out = {k: v for k, v in shard.items() if k not in NEW_SHARD_KEYS}
    out["producer"] = dict(out["producer"], export_schema=lsp.EXPORT_SCHEMA)
    out["roots"] = [
        dict(
            {k: v for k, v in r.items() if k not in NEW_ROOT_KEYS},
            candidates=[
                {k: v for k, v in c.items() if k not in NEW_CAND_KEYS} for c in r["candidates"]
            ],
        )
        for r in shard["roots"]
    ]
    return out


def _strip(value):
    """Drop wall-clock fields (anything named *seconds*), as the live resume test does."""
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if "seconds" not in str(k)}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _equal(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=False))
    return a == b


def test_config_records_the_flag_only_when_on() -> None:
    off, on = _config(False), _config(True)
    assert "successor_rows" not in off.state_dict()
    assert on.state_dict()["successor_rows"] is True
    assert SearchConfig.from_mapping(on.state_dict()) == on
    assert SearchConfig.from_mapping(off.state_dict()) == off
    assert on.successor_decisions == (1, 3)
    assert SearchConfig(horizons=(1,), keep=("all",)).successor_decisions == (1,)
    assert lsp.export_schema(False) == lsp.EXPORT_SCHEMA
    assert lsp.export_schema(True) == lsp.EXPORT_SCHEMA_SUCCESSORS


def test_flag_off_is_v1_and_on_keeps_every_v1_field() -> None:
    off = _shard(False)
    assert off["producer"]["export_schema"] == lsp.EXPORT_SCHEMA
    assert not NEW_SHARD_KEYS & set(off) and not lsp.has_successors(off)
    for r in off["roots"]:
        assert not NEW_ROOT_KEYS & set(r)
        assert all(not NEW_CAND_KEYS & set(c) for c in r["candidates"])
    on = _shard(True)
    assert on["producer"]["export_schema"] == lsp.EXPORT_SCHEMA_SUCCESSORS
    assert lsp.has_successors(on)
    # Every v1 field is identical apart from the declared config flag and export schema.
    stripped = _strip_v2(on)
    stripped["producer"] = dict(stripped["producer"])
    assert _equal(stripped, off)
    # The search itself is unchanged: labels, utilities, rounds and work.
    _, a = _root(_config(False))
    _, b = _root(_config(True))
    assert json.dumps(a.table["decisions"]) == json.dumps(b.table["decisions"])
    assert json.dumps(a.table["rounds"]) == json.dumps(b.table["rounds"])
    assert _strip(a.table["work"]) == _strip(b.table["work"])
    assert all(_equal(x, y) for x, y in zip(a.rows, b.rows, strict=False))


def test_successor_streams_rows_and_statuses() -> None:
    native, result = _root(_config(True))
    record = result.payload_root["record"]
    rows = result.payload_root["successors"]
    assert record["successor_decisions"] == [1, 3]
    final = set(result.table["reeval_candidates"])
    seen_dead = seen_live3 = 0
    for cand in record["candidates"]:
        if cand.get("plan_role"):
            assert "successor_streams" not in cand
            continue
        streams = cand["successor_streams"]
        v = FakeNative.value(cand["action"])
        expected = [0] + ([1, 2] if cand["index"] in final else [])
        assert [s["stream"] for s in streams] == expected
        for entry in streams:
            s = entry["stream"]
            decisions = [b["decision"] for b in entry["boundaries"]]
            # per-stream rewards are the branch's raw steps
            assert len(entry["step_rewards"]) == len(decisions) == len(entry["step_years"])
            for k, reward in enumerate(entry["step_rewards"]):
                assert reward[0] == pytest.approx(v + 0.01 * s)
            for b in entry["boundaries"]:
                if b["status"] == "terminated":
                    seen_dead += 1
                    assert b["decision"] == 2 and v < DIES and b["row"] is None
                    assert entry["terminated"]
                elif b["decision"] in (1, 3):
                    assert b["status"] == "live" and b["row"] is not None
                    row = rows[b["row"]]
                    assert row["tensors"]["core"].tolist() == [
                        [pytest.approx(v), float(s), float(b["decision"])]
                    ]
                    assert row["tensors"]["labor_personality"].tolist() == [
                        [pytest.approx(0.3), pytest.approx(0.7)]
                    ]
                    seen_live3 += b["decision"] == 3
                else:
                    assert b["status"] == "live" and b["row"] is None  # decision 2: status only
                assert b["turn"] == 4 + b["decision"]
            # returns_by_horizon of the stream's steps equals the stored evaluations
            if s == 0:
                h1 = next(e for e in cand["evaluations"] if e["stream"] == 0 and e["horizon"] == 1)
                assert h1["returns"] == pytest.approx(entry["step_rewards"][0])
    assert seen_live3 > 0
    # Rows are deduplicated: distinct (value, stream, decision) only.
    keys = {tuple(r["tensors"]["core"].reshape(-1).tolist()) for r in rows}
    assert len(keys) == len(rows)
    assert native.successor_seconds >= 0.0
    del seen_dead  # termination depends on the keyed candidate values


def test_terminated_branch_records_status_without_row() -> None:
    config = _config(True)
    native = SuccessorFake(_catalog(), config)
    native.successor_view = _view
    branch = Branch(0, 0, record=True)
    payload = _base_payload(_catalog())
    # force a dying candidate
    native.value = lambda _payload: 0.1  # type: ignore[method-assign]
    native.extend(branch, 3, candidate=payload, keep_fork=False)
    assert [b["status"] for b in branch.boundaries] == ["live", "terminated"]
    assert branch.boundaries[0]["row"] is not None and branch.boundaries[1]["row"] is None
    assert branch.terminated and not branch.alive


def test_branch_without_record_flag_and_plans_record_nothing() -> None:
    native = SuccessorFake(_catalog(), _config(True))
    native.successor_view = _view
    branch = Branch(0, 0)
    native.extend(branch, 3, candidate=_base_payload(_catalog()), keep_fork=False)
    assert branch.boundaries == []


def test_shard_assembly_offsets_rows_across_roots_and_round_trips(tmp_path: Path) -> None:
    shard = _shard(True, roots=3)
    total = shard["successor_row_count"]
    assert shard["successor_tensors"]["core"].shape == (total, 3)
    assert shard["successor_privileged"] is None
    assert sum(r["successor_rows"] for r in shard["roots"]) == total
    offset = 0
    for r in shard["roots"]:
        used = {
            b["row"]
            for c in r["candidates"]
            for e in c.get("successor_streams") or []
            for b in e["boundaries"]
            if b["row"] is not None
        }
        assert used == set(range(offset, offset + r["successor_rows"]))
        offset += r["successor_rows"]
    path = tmp_path / lsp.PAYLOAD_FILE
    binding = lsp.write_candidate_shard(path, shard)
    assert binding["export_schema"] == lsp.EXPORT_SCHEMA_SUCCESSORS
    assert binding["successor_rows"] == total
    loaded = lsp.read_candidate_shard(path)
    assert _equal(loaded["successor_tensors"], shard["successor_tensors"])
    assert loaded["roots"] == shard["roots"]
    v1_binding = lsp.write_candidate_shard(tmp_path / "v1" / lsp.PAYLOAD_FILE, _shard(False))
    assert "successor_rows" not in v1_binding
    assert v1_binding["export_schema"] == lsp.EXPORT_SCHEMA
    try:
        from anthropocene_rl.autonomous_v7 import labor_advantage
    except ImportError:  # pragma: no cover - model side absent
        return
    parsed = labor_advantage.parse_candidate_shard(loaded)  # the strict reader ignores v2 keys
    assert len(parsed.roots) == 3


def test_shard_assembly_refuses_mixed_roots() -> None:
    a = _root(_config(True), root_index=0)[1].payload_root
    b = _root(_config(False), root_index=1)[1].payload_root
    with pytest.raises(ValueError, match="mix successor"):
        lsp.candidate_shard([a, b], producer=PRODUCER, discount_years=400.0)


def test_successor_rows_store_deduplicates_and_validates() -> None:
    store = lsp.SuccessorRows((0.5, 0.5))
    row = {"tensors": {"core": torch.ones(1, 3)}, "privileged": None}
    assert store.add(row) == 0 and store.add(row) == 0
    assert store.add({"tensors": {"core": torch.zeros(1, 3)}, "privileged": None}) == 1
    with pytest.raises(ValueError, match="only a live"):
        lsp.successor_stream(
            0,
            [[0, 0, 0, 0]],
            [20.0],
            True,
            [{"decision": 1, "turn": 5, "status": "terminated", "row": row}],
            store,
        )
    with pytest.raises(ValueError, match="status"):
        lsp.successor_stream(
            0,
            [[0, 0, 0, 0]],
            [20.0],
            False,
            [{"decision": 1, "turn": 5, "status": "asleep", "row": None}],
            store,
        )


def _fake_world(job):
    config = SearchConfig.from_mapping(job["config"])
    rows, roots, tables = [], [], []
    for index in range(2):
        _, result = _root(config, root_index=index, world=int(job["world"]))
        rows.extend(result.rows)
        tables.append(result.table)
        roots.append(result.payload_root)
    return {
        "world": int(job["world"]),
        "rows": rows,
        "tables": tables,
        "skipped": {},
        "reset_identity": {},
        "root_turn_indices": [4, 5],
        "restore_checks": [],
        "work": {},
        "main_path_resolves": 0,
        "wall_seconds": 0.0,
        "payload_roots": roots,
    }


def _publish(tmp_path: Path, monkeypatch, successors: bool) -> Path:
    monkeypatch.setattr(labor_search, "search_world", _fake_world)
    out = tmp_path / f"search-{int(successors)}"
    out.mkdir()
    config = _config(successors)
    body = {
        "schema": 1,
        "source_kind": labor_search.SOURCE_KIND,
        "options": {"turns": 8, "reset_options": {}, "partition": "fitting"},
    }
    source = dict(body, source_run=digest(body))
    job = {
        "world": 90100,
        "out": str(out),
        "source_run": source["source_run"],
        "config": config.state_dict(),
        "turns": 8,
        "reset_options": {},
        "normalizer_digest": "n",
        "source": "h12",
        "partition": "fitting",
    }
    summary = labor_search.write_search_shard(job)
    ltd.publish_dataset(
        out,
        source,
        [{k: summary[k] for k in ("path", "world_seed", "meta_sha256", "data_digest", "rows")}],
    )
    return out


def test_published_shard_label_rows_and_tables_are_identical(tmp_path, monkeypatch) -> None:
    off = _publish(tmp_path, monkeypatch, False) / "world-90100"
    on = _publish(tmp_path, monkeypatch, True) / "world-90100"
    for path in sorted(off.glob("*.npy")):
        assert path.read_bytes() == (on / path.name).read_bytes(), path.name
    tables = [
        [_strip(json.loads(line)) for line in (d / "candidates.jsonl").read_text().splitlines()]
        for d in (off, on)
    ]
    assert tables[0] == tables[1]  # up to wall-clock fields
    meta_off = json.loads((off / "meta.json").read_text())
    meta_on = json.loads((on / "meta.json").read_text())
    b_off, b_on = meta_off.pop("candidate_payload"), meta_on.pop("candidate_payload")
    meta_off.pop("candidates_sha256"), meta_on.pop("candidates_sha256")  # wall-clock fields
    assert meta_off == meta_on
    assert b_on["export_schema"] == lsp.EXPORT_SCHEMA_SUCCESSORS and b_on["successor_rows"] > 0
    assert {
        k: v for k, v in b_on.items() if k not in ("export_schema", "sha256", "successor_rows")
    } == {k: v for k, v in b_off.items() if k not in ("export_schema", "sha256")}
    (path,) = lsp.candidate_shard_paths(on.parent)
    assert lsp.has_successors(lsp.read_candidate_shard(path))
