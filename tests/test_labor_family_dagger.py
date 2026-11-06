"""Family tags for actor-from-reset DAgger rows (labor_scenarios.row_family_tags).

Actor-from-reset DAgger rollouts are the actor's own chronological episodes from the world
reset, so the tracker's history-dependent detection applies to them exactly as to a
teacher trajectory. Teacher-prefix DAgger branches (keyed starts inside a teacher
trajectory) stay untagged, and teacher-only datasets are tagged exactly as before.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_dagger
from anthropocene_rl.autonomous_v7 import labor_scenarios as ls
from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.provenance import ProvenanceError, digest
from tests.test_labor_scenarios import FARMER, HERDER, _synthetic_arrays

BOTH = (True, True, False)
# (world, turn, camp, heads, residents, housing, clothing, trade, neolithic): two reset camps
# (one under-clothed with a relapse, one gaining herding = mixed) and a daughter at turn 10.
ROWS = [
    (90100, 8, 1, FARMER, 50.0, 1.0, 0.2, 0.0, 1),
    (90100, 8, 2, FARMER, 300.0, 1.0, 1.0, 0.0, 1),
    (90100, 9, 1, FARMER, 50.0, 1.0, 0.5, 0.0, 1),
    (90100, 9, 2, BOTH, 300.0, 1.0, 1.0, 0.0, 1),
    (90100, 10, 1, FARMER, 60.0, 1.0, 0.5, 0.0, 1),
    (90100, 10, 2, BOTH, 300.0, 1.0, 1.0, 0.0, 1),
    (90100, 10, 7, FARMER, 40.0, 1.0, 1.0, 0.0, 1),
    (90100, 11, 1, FARMER, 60.0, 1.0, 0.1, 0.0, 1),
    (90100, 11, 2, BOTH, 300.0, 1.0, 0.25, 0.0, 1),
    (90100, 11, 7, HERDER, 40.0, 1.0, 1.0, 0.0, 1),
]
TURNS = sorted({r[1] for r in ROWS})


def _teacher(rows=ROWS) -> dict:
    data = {k: torch.as_tensor(v) for k, v in _synthetic_arrays(rows).items()}
    data["row_source"] = torch.zeros(len(rows), dtype=torch.long)
    data["row_trajectory"] = torch.full((len(rows),), -1, dtype=torch.long)
    return data


def _dagger(rows=ROWS, *, prefix="actor-from-reset", start=0, start_index=0, round_=1,
            collection=True, steps=None) -> dict:
    data = {k: torch.as_tensor(v) for k, v in _synthetic_arrays(rows).items()}
    n = len(rows)
    turns = [r[1] for r in rows]
    unique = sorted(set(turns))
    step = [unique.index(t) for t in turns] if steps is None else steps
    data.update(
        row_source=torch.full((n,), labor_dagger.DAGGER_ROW_SOURCE, dtype=torch.long),
        row_round=torch.full((n,), round_, dtype=torch.long),
        row_depth=torch.full((n,), len(unique), dtype=torch.long),
        row_start=torch.full((n,), start, dtype=torch.long),
        row_start_index=torch.full((n,), start_index, dtype=torch.long),
        row_branch_step=torch.as_tensor(step, dtype=torch.long),
    )
    if collection:
        data["collection"] = {"source_kind": labor_dagger.DAGGER_SOURCE_KIND, "rows": n}
        if prefix is not None:
            data["collection"]["prefix"] = prefix
    return data


def _aggregate(*parts: dict) -> dict:
    """Concatenated parts with the loader's identity defaults (as _load_labor_datasets)."""
    parts = tuple(dict(p) for p in parts)
    for p in parts:
        for key, default in ltd.AGGREGATE_IDENTITY_DEFAULTS.items():
            if key not in p:
                p[key] = torch.full((len(p["row_world"]),), default, dtype=torch.long)
    keys = set.intersection(*(set(k for k, v in p.items() if isinstance(v, torch.Tensor))
                              for p in parts))
    out = {k: torch.cat([p[k] for p in parts]) for k in sorted(keys)}
    out["collection"] = {
        "source_kind": "aggregated",
        "component_collections": [
            p.get("collection") or {"source_kind": ltd.SOURCE_KIND, "rows": len(p["row_world"])}
            for p in parts
        ],
    }
    return out


def test_constants_match_the_collector() -> None:
    assert ls.DAGGER_ROW_SOURCE == labor_dagger.DAGGER_ROW_SOURCE == 1
    assert ls.DAGGER_SOURCE_KIND == labor_dagger.DAGGER_SOURCE_KIND
    assert ls.DAGGER_PREFIX_TEACHER == labor_dagger.PREFIX_TEACHER
    assert ls.DAGGER_PREFIX_ACTOR_FROM_RESET == labor_dagger.PREFIX_ACTOR_FROM_RESET
    assert ltd.ROW_SOURCES[ls.DAGGER_SOURCE_KIND] == ls.DAGGER_ROW_SOURCE


def test_actor_from_reset_rows_get_the_teacher_trajectory_tags() -> None:
    teacher = ls.row_family_tags(_teacher())
    names = [set(ls.family_names(int(t))) for t in teacher]
    # The fixture exercises history-dependent families.
    assert "under-clothed" in names[0] and "under-clothed" in names[7]
    assert "mixed" in names[3] and "fission-daughter" in names[6]
    dagger = ls.row_family_tags(_dagger())
    assert dagger.tolist() == teacher.tolist()
    # Reversed row order: grouping and turn sorting, not storage order, decide.
    order = list(reversed(range(len(ROWS))))
    shuffled = _dagger([ROWS[i] for i in order])
    assert ls.row_family_tags(shuffled).tolist() == [int(teacher[i]) for i in order]


def test_teacher_prefix_dagger_rows_stay_untagged() -> None:
    for data in (_dagger(prefix="teacher"), _dagger(prefix=None, start=3)):
        assert not ls.row_family_tags(data).any()
    # Opting out leaves every DAgger row untagged and needs no metadata.
    assert not ls.row_family_tags(_dagger(collection=False), dagger_from_reset=False).any()


def test_dagger_rows_without_metadata_or_inconsistent_rollouts_are_refused() -> None:
    with pytest.raises(ValueError, match="prefix"):
        ls.row_family_tags(_dagger(collection=False))
    with pytest.raises(ValueError, match="row_start"):
        ls.row_family_tags(_dagger(start=2))
    bad = [0, 1, 1, 2, 2, 2, 2, 3, 3, 4]  # the last decision skips branch step 3
    with pytest.raises(ValueError, match="consecutive"):
        ls.row_family_tags(_dagger(steps=bad))
    with pytest.raises(ValueError, match="unknown DAgger prefix"):
        ls.row_family_tags(_dagger(prefix="something-else"))
    mixed = _aggregate(_teacher(), _dagger())
    mixed["collection"]["component_collections"][1]["rows"] -= 1
    with pytest.raises(ValueError, match="attributed"):
        ls.row_family_tags(mixed)


def test_aggregate_tags_each_rollout_separately_and_keeps_teacher_tags() -> None:
    teacher = _teacher()
    reference = ls.row_family_tags(teacher)
    from_reset = _dagger()
    second = _dagger(start_index=1)  # a second (stochastic) rollout of the same world
    branches = _dagger(prefix=None, start=3)  # teacher-prefix branches
    mixed = _aggregate(teacher, from_reset, second, branches)
    tags = ls.row_family_tags(mixed)
    n = len(ROWS)
    assert tags[:n].tolist() == reference.tolist()
    assert tags[n:2 * n].tolist() == reference.tolist()
    assert tags[2 * n:3 * n].tolist() == reference.tolist()  # its own tracker, not a continuation
    assert not tags[3 * n:].any()


def test_teacher_only_datasets_are_tagged_exactly_as_before() -> None:
    teacher = _teacher()
    before = ls.row_family_tags(teacher, dagger_from_reset=False)  # the historical path
    assert ls.row_family_tags(teacher).tolist() == before.tolist()
    # No metadata is read without DAgger rows: no collection, or a teacher collection.
    plain = {k: v for k, v in teacher.items() if k not in ("row_source", "row_trajectory")}
    assert ls.row_family_tags(plain).tolist() == before.tolist()
    teacher["collection"] = {"source_kind": ltd.SOURCE_KIND}
    assert ls.row_family_tags(teacher).tolist() == before.tolist()
    assert ls.row_family_tags.__defaults__[1] == (0,)


def test_reweighting_keeps_source_and_world_mass_and_records_shares() -> None:
    teacher, dagger = _teacher(), _dagger()
    mixed = _aggregate(teacher, dagger)
    n = 2 * len(ROWS)
    mixed["row_quarantined"] = torch.zeros(n, dtype=torch.bool)
    mixed["row_weight"] = torch.ones(n)
    base = mixed["row_weight"].clone().double().numpy()
    record = ls.reweight_dataset(mixed, {"under-clothed": 2.0})
    new = mixed["row_weight"].double().numpy()
    sources = mixed["row_source"].numpy()
    for code in (0, 1):
        group = sources == code
        assert new[group].sum() == pytest.approx(base[group].sum())
    clothed = (mixed["row_family_tags"].numpy() & ls.FAMILY_BITS["under-clothed"]) != 0
    dagger_rows = sources == 1
    assert (clothed & dagger_rows).any()
    up = new[clothed & dagger_rows][0]
    flat = new[~clothed & dagger_rows][0]
    assert up == pytest.approx(2.0 * flat)  # the up-weight now reaches DAgger rows
    assert record["tagged_sources"] == ls.TAGGED_SOURCES_DESCRIPTION
    assert "actor-from-reset" in record["tagged_sources"]
    by_source = record["shares_by_source"]
    assert set(by_source) == {"0", "1"}
    assert by_source["1"]["tagged_kept_rows"] == by_source["0"]["tagged_kept_rows"] > 0
    assert by_source["1"]["shares"]["under-clothed"]["weight_share_after"] > by_source["1"][
        "shares"]["under-clothed"]["weight_share_before"]
    # One source: the record has no per-source block (unchanged layout).
    single = _teacher()
    single["row_quarantined"] = torch.zeros(len(ROWS), dtype=torch.bool)
    single["row_weight"] = torch.ones(len(ROWS))
    assert "shares_by_source" not in ls.reweight_dataset(single, {"under-clothed": 2.0})


# Loader: the DAgger dataset's prefix reaches the loaded collection ---------------------------


def _publish(directory, kind, rows, *, prefix=None, meta_prefix="same", dagger=False):
    options = {"turns": len(TURNS), "reset_options": {"players": 8}, "worlds": [rows[0][0]]}
    if dagger:
        options["actor_only"] = True
    if prefix is not None:
        options["prefix"] = prefix
    body = {"options": options, "source_kind": kind, "schema": ltd.SCHEMA}
    source = dict(body, source_run=digest(body))
    directory.mkdir(parents=True)
    world = rows[0][0]
    synthetic = _synthetic_arrays(rows)
    n = len(rows)
    arrays = {
        "row_world": synthetic["row_world"],
        "row_camp": synthetic["row_camp"],
        "row_turn": synthetic["row_turn"],
        "labor_specialty_mask": synthetic["labor_specialty_mask"],
        "row_attempt": np.zeros(n, np.int64),
        "row_quarantined": np.zeros(n, dtype=bool),
        "row_weight": np.ones(n, np.float32),
        "target_specialty": np.full((n, 14), 1 / 14, np.float32),
    }
    if dagger:
        unique = sorted({r[1] for r in rows})
        arrays.update(
            row_source=np.ones(n, np.int64),
            row_round=np.ones(n, np.int64),
            row_depth=np.full(n, len(unique), np.int64),
            row_start=np.zeros(n, np.int64),
            row_start_index=np.zeros(n, np.int64),
            row_branch_step=np.asarray([unique.index(r[1]) for r in rows], np.int64),
        )
    meta = {"source_run": source["source_run"], "world_seed": world, "normalizer_digest": "n",
            "counters": {"total_decisions": n}, "label_unavailable_reasons": {},
            "quarantine": []}
    shard_prefix = prefix if meta_prefix == "same" else meta_prefix
    if shard_prefix is not None:
        meta["prefix"] = shard_prefix
    record = ltd.write_shard(directory / f"world-{world}", arrays, meta)
    return ltd.publish_dataset(directory, source, [{
        "path": f"world-{world}", "world_seed": world, "rows": n,
        "meta_sha256": ltd.checkpoint_hash(directory / f"world-{world}" / "meta.json"),
        "data_digest": record["data_digest"],
    }])


def test_loaded_teacher_plus_actor_from_reset_dagger_is_tagged(tmp_path) -> None:
    teacher_dir, dagger_dir = tmp_path / "teacher", tmp_path / "dagger"
    _publish(teacher_dir, ltd.SOURCE_KIND, ROWS)
    _publish(dagger_dir, labor_dagger.DAGGER_SOURCE_KIND, ROWS, prefix="actor-from-reset",
             dagger=True)
    alone = ltd.load_labor_datasets([teacher_dir])
    assert "prefix" not in alone["collection"]  # teacher collections are unchanged
    dagger_alone = ltd.load_labor_datasets([dagger_dir])
    assert dagger_alone["collection"]["prefix"] == "actor-from-reset"
    mixed = ltd.load_labor_datasets([teacher_dir, dagger_dir])  # pretrain-bc's call
    components = mixed["collection"]["component_collections"]
    assert [c.get("prefix") for c in components] == [None, "actor-from-reset"]
    tags = ls.row_family_tags(mixed)
    n = len(ROWS)
    reference = ls.row_family_tags(alone)
    assert reference.any()
    assert tags[:n].tolist() == reference.tolist()
    assert tags[n:].tolist() == reference.tolist()
    assert ls.row_family_tags(dagger_alone).tolist() == reference.tolist()
    weighted = ltd.load_labor_datasets([teacher_dir, dagger_dir],
                                       family_weights={"fission-daughter": 2.0})
    shares = weighted["collection"]["family_weighting"]["shares_by_source"]
    assert shares["1"]["tagged_kept_rows"] == shares["0"]["tagged_kept_rows"]


def test_loaded_teacher_prefix_dagger_stays_untagged_and_meta_mismatch_fails(tmp_path) -> None:
    teacher_dir, branches = tmp_path / "teacher", tmp_path / "branches"
    _publish(teacher_dir, ltd.SOURCE_KIND, ROWS)
    _publish(branches, labor_dagger.DAGGER_SOURCE_KIND, ROWS, dagger=True)  # default prefix
    mixed = ltd.load_labor_datasets([teacher_dir, branches])
    tags = ls.row_family_tags(mixed)
    assert tags[: len(ROWS)].any() and not tags[len(ROWS):].any()
    bad = tmp_path / "bad"
    _publish(bad, labor_dagger.DAGGER_SOURCE_KIND, ROWS, prefix="actor-from-reset",
             meta_prefix=None, dagger=True)
    with pytest.raises(ProvenanceError, match="prefix"):
        ltd.load_labor_datasets([bad])
    stray = tmp_path / "stray"
    _publish(stray, labor_dagger.DAGGER_SOURCE_KIND, ROWS, meta_prefix="actor-from-reset",
             dagger=True)
    with pytest.raises(ProvenanceError, match="prefix"):
        ltd.load_labor_datasets([stray])
