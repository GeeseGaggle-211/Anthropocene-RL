"""G4 learner-state recovery (DAgger): start draws, aggregation weights and a live round trip."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.labor_bc import world_balanced_weights
from anthropocene_rl.autonomous_v7.labor_dagger import draw_start_turns
from anthropocene_rl.autonomous_v7.provenance import ProvenanceError, digest


def test_start_turns_are_keyed_stratified_and_in_range() -> None:
    a = draw_start_turns(4243, 90100, 1, 3, 25, 1)
    assert a == draw_start_turns(4243, 90100, 1, 3, 25, 1)
    assert len(a) == 3 and a == sorted(a)
    span = 25 - 1 + 1
    for index, start in enumerate(a):
        assert (index * span) // 3 <= start < ((index + 1) * span) // 3
    for depth in range(1, 6):
        for world in range(90100, 90160):
            starts = draw_start_turns(4243, world, depth, 3, 25, depth)
            assert all(0 <= s <= 25 - depth for s in starts)
    rounds = {tuple(draw_start_turns(4243, 90100, r, 3, 25, 1)) for r in range(1, 6)}
    assert len(rounds) > 1  # the round enters the key
    assert draw_start_turns(1, 90100, 1, 1, 2, 1)[0] in (0, 1)
    assert draw_start_turns(1, 90100, 1, 4, 2, 2) == [0, 0, 0, 0]  # degenerate span stays legal


def test_source_world_weights_give_equal_source_then_world_mass() -> None:
    # Source 0: worlds 1 (4 rows) and 2 (2 rows); source 1: worlds 1 (1 row), 3 (3 rows), one
    # quarantined.
    sources = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    worlds = torch.tensor([1, 1, 1, 1, 2, 2, 1, 3, 3, 3, 3])
    keep = torch.tensor([True] * 10 + [False])
    w = ltd.source_world_balanced_weights(sources, worlds, keep)
    total = float(w.sum())
    assert math.isclose(total, float(keep.sum()), rel_tol=1e-6)  # mean weight one over kept rows
    assert float(w[~keep].sum()) == 0.0
    assert math.isclose(float(w[sources == 0].sum()), total / 2, rel_tol=1e-6)
    assert math.isclose(float(w[sources == 1].sum()), total / 2, rel_tol=1e-6)
    for source, world in ((0, 1), (0, 2), (1, 1), (1, 3)):
        mass = float(w[(sources == source) & (worlds == world)].sum())
        assert math.isclose(mass, total / 4, rel_tol=1e-6)
    # Within one world of one source every kept row has the same weight.
    rows = (sources == 1) & (worlds == 3) & keep
    assert torch.allclose(w[rows], w[rows][0].expand(int(rows.sum())))
    # One source reduces exactly to the original world balance.
    single = ltd.source_world_balanced_weights(torch.zeros(11, dtype=torch.long), worlds, keep)
    assert torch.equal(single, world_balanced_weights(worlds, keep))


def _fake_dataset(
    directory, kind: str, worlds: list[int], rows_per_world: int, extra: bool
) -> dict:
    options = {"turns": 2, "reset_options": {"players": 8}, "worlds": worlds}
    if extra:
        options["actor_only"] = True
    body = {"options": options, "source_kind": kind, "schema": ltd.SCHEMA}
    source = dict(body, source_run=digest(body))
    directory.mkdir(parents=True)
    shards = []
    for world in worlds:
        n = rows_per_world
        arrays = {
            "row_world": np.full(n, world, np.int64),
            "row_camp": np.arange(n, dtype=np.int64),
            "row_turn": np.full(n, 7, np.int64),
            "row_attempt": np.zeros(n, np.int64),
            "row_quarantined": np.asarray([False] * (n - 1) + [True]),
            "row_weight": np.ones(n, np.float32),
            "target_specialty": np.random.default_rng(world).random((n, 14)).astype(np.float32),
        }
        if extra:
            arrays.update(
                row_source=np.ones(n, np.int64),
                row_round=np.full(n, 1, np.int64),
                row_depth=np.full(n, 1, np.int64),
                row_start=np.full(n, 3, np.int64),
                row_start_index=np.zeros(n, np.int64),
                row_branch_step=np.zeros(n, np.int64),
            )
        meta = {
            "source_run": source["source_run"],
            "world_seed": world,
            "normalizer_digest": "n",
            "counters": {"total_decisions": n},
            "label_unavailable_reasons": {},
            "quarantine": [],
        }
        record = ltd.write_shard(directory / f"world-{world}", arrays, meta)
        shards.append(
            {
                "path": f"world-{world}",
                "world_seed": world,
                "meta_sha256": ltd.checkpoint_hash(directory / f"world-{world}" / "meta.json"),
                "data_digest": record["data_digest"],
                "rows": n,
            }
        )
    return ltd.publish_dataset(directory, source, shards)


def test_aggregated_datasets_mix_sources_half_and_half(tmp_path) -> None:
    teacher = tmp_path / "teacher"
    dagger1 = tmp_path / "dagger1"
    dagger2 = tmp_path / "dagger2"
    _fake_dataset(teacher, ltd.SOURCE_KIND, [5000, 5002, 5004], 10, extra=False)
    _fake_dataset(dagger1, "learner_state_dagger", [5000], 3, extra=True)
    _fake_dataset(dagger2, "learner_state_dagger", [5002, 5004], 2, extra=True)

    single = ltd.load_labor_datasets([teacher])
    reference = ltd.load_labor_dataset(teacher)
    assert set(single) == set(reference)
    for key, value in reference.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(single[key], value), key

    mixed = ltd.load_labor_datasets([teacher, dagger1, dagger2])
    w, src, world = mixed["row_weight"], mixed["row_source"], mixed["row_world"]
    assert len(w) == 30 + 3 + 4
    total = float(w.sum())
    assert math.isclose(float(w[src == 0].sum()), total / 2, rel_tol=1e-6)
    assert math.isclose(float(w[src == 1].sum()), total / 2, rel_tol=1e-6)
    for seed in (5000, 5002, 5004):
        assert math.isclose(float(w[(src == 0) & (world == seed)].sum()), total / 6, rel_tol=1e-6)
        assert math.isclose(float(w[(src == 1) & (world == seed)].sum()), total / 6, rel_tol=1e-6)
    assert float(w[mixed["row_quarantined"]].sum()) == 0.0
    assert bool((mixed["row_start"][src == 0] == -1).all())
    assert bool((mixed["row_round"][src == 1] == 1).all())
    collection = mixed["collection"]
    assert collection["actor_only"] is True
    assert collection["sources"]["learner_state_dagger"]["kept_rows"] == 2 + 1 + 1
    assert collection["sources"][ltd.SOURCE_KIND]["worlds"] == [5000, 5002, 5004]
    json.dumps(collection)
    with pytest.raises(ProvenanceError):
        ltd.load_labor_datasets([teacher, teacher])


def test_learner_state_shards_separate_branch_identities(tmp_path) -> None:
    arrays = {
        "row_world": np.asarray([5000, 5000]),
        "row_camp": np.asarray([1, 1]),
        "row_turn": np.asarray([3, 3]),
        "row_attempt": np.asarray([0, 0]),
        "row_start_index": np.asarray([0, 1]),
    }
    shard = tmp_path / "world-5000"
    ltd.write_shard(shard, arrays, {"world_seed": 5000, "source_run": "x"})
    assert ltd.verify_shard(shard)["rows"] == 2
    np.save(shard / "row_start_index.npy", np.asarray([1, 1]))
    with pytest.raises(ProvenanceError):
        ltd.verify_shard(shard)


@pytest.fixture(scope="module")
def bounded_normalizer():
    from tests.test_labor_bc import _fit_test_normalizer

    return _fit_test_normalizer()


def test_live_dagger_round_feeds_aggregated_bc(tmp_path, bounded_normalizer) -> None:
    """Teacher data, one learner-state round, aggregated pretrain-bc; critic refuses actor-only data."""  # noqa: E501 - docstring/prose line kept verbatim
    from anthropocene_rl.autonomous_v7 import labor_dagger
    from anthropocene_rl.autonomous_v7.labor_model import BOUNDED_FEATURE_CONTRACT
    from anthropocene_rl.autonomous_v7.labor_training import main as labor_training_main

    norm = tmp_path / "norm.pt"
    torch.save(
        {
            "feature_contract": BOUNDED_FEATURE_CONTRACT,
            "normalizer": bounded_normalizer.state_dict(),
        },
        norm,
    )
    init = tmp_path / "init.pt"
    labor_training_main(
        [
            "initialize",
            "--normalizer-checkpoint",
            str(norm),
            "--world-seed",
            "5000",
            "--optimizer-seed",
            "1",
            "--out",
            str(init),
        ]
    )
    teacher = tmp_path / "teacher"
    assert (
        ltd.main(
            [
                "--normalizer-checkpoint",
                str(norm),
                "--worlds",
                "5000",
                "--turns",
                "2",
                "--players",
                "8",
                "--jobs",
                "1",
                "--allow-dirty",
                "--return-tail",
                "truncated-zero",
                "--out",
                str(teacher),
            ]
        )
        == 0
    )
    dagger = tmp_path / "dagger"
    common = [
        "collect",
        "--checkpoint",
        str(init),
        "--worlds",
        "5000",
        "--round",
        "1",
        "--depth",
        "1",
        "--starts",
        "2",
        "--horizon",
        "2",
        "--players",
        "8",
        "--jobs",
        "1",
        "--allow-dirty",
    ]
    starts = draw_start_turns(labor_dagger.DEFAULT_START_KEY, 5000, 1, 2, 2, 1)
    assert labor_dagger.main(common + ["--out", str(dagger)]) == 0  # noqa: RUF005 - concatenation kept (operand type not guaranteed list)
    manifest = ltd.verify_dataset(dagger)
    assert manifest["source_record"]["source_kind"] == labor_dagger.DAGGER_SOURCE_KIND
    assert manifest["source_record"]["options"]["actor_only"] is True
    assert labor_dagger.main(common + ["--out", str(dagger)]) == 0  # published: idempotent no-op  # noqa: RUF005 - concatenation kept (operand type not guaranteed list)

    learner = ltd.load_labor_dataset(dagger)
    reference = ltd.load_labor_dataset(teacher)
    assert learner["collection"]["actor_only"] is True
    assert bool((learner["row_source"] == 1).all())
    # Both sources store the critic-only privileged view (schema 1) beside the observation;
    # the equality loop below then binds the learner rows' privileged arrays to the teacher's.
    from anthropocene_rl.autonomous_v7.labor_privileged import shard_keys

    assert set(shard_keys()) <= set(learner) and set(shard_keys()) <= set(reference)
    assert (
        ltd.load_labor_dataset(dagger, require_privileged=True)["collection"][
            "privileged_schema_version"
        ]
        == 1
    )
    # Branch step 0 is the teacher-prefix state: inputs and labels equal the collector's rows there.
    skip = {
        "row_weight",
        "row_source",
        "row_round",
        "row_depth",
        "row_start",
        "row_start_index",
        "row_branch_step",
    }
    for index, start in enumerate(starts):
        sel_d = learner["row_start_index"] == index
        assert bool((learner["row_start"][sel_d] == start).all())
        turn = int(learner["row_turn"][sel_d][0])
        sel_t = reference["row_turn"] == turn
        for key, value in reference.items():
            if (
                not isinstance(value, torch.Tensor)
                or key in skip
                or key.startswith(("reward_", "transition_"))
            ):
                continue
            assert torch.equal(value[sel_t], learner[key][sel_d]), key

    bc = tmp_path / "bc.pt"
    labor_training_main(
        [
            "pretrain-bc",
            "--checkpoint",
            str(init),
            "--dataset",
            str(teacher),
            "--dataset",
            str(dagger),
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--out",
            str(bc),
        ]
    )
    summary = torch.load(bc, map_location="cpu", weights_only=True)["heuristic_bc_pretraining"]
    sources = summary["collection"]["sources"]
    assert set(sources) == {ltd.SOURCE_KIND, labor_dagger.DAGGER_SOURCE_KIND}
    assert math.isclose(
        sources[ltd.SOURCE_KIND]["weight_mass"],
        sources[labor_dagger.DAGGER_SOURCE_KIND]["weight_mass"],
        rel_tol=1e-5,
    )
    with pytest.raises(ValueError, match="actor-only"):
        labor_training_main(
            [
                "pretrain-critic",
                "--checkpoint",
                str(init),
                "--dataset",
                str(dagger),
                "--epochs",
                "1",
                "--batch-size",
                "16",
                "--out",
                str(tmp_path / "c.pt"),
            ]
        )
