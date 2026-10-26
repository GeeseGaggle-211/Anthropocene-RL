"""Reserved held-out worlds 90154-90159 never train any network (amendment r5-b).

One shared constant and guard (``labor_worlds``) is enforced by the actor
side too: BC, DAgger collection and refit, search label collection, PPO
rollouts and every training loader. Search and teacher collection may run on
the reserved worlds only in an explicit held-out-test mode whose shards are
tagged and refused by every training loader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_privileged, labor_worlds
from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.labor_bc import (
    pretrain_labor_actor_bc,
    validate_fitting_worlds,
)
from anthropocene_rl.autonomous_v7.labor_search import validate_search_worlds
from anthropocene_rl.autonomous_v7.provenance import digest


def test_single_shared_constant_and_guard() -> None:
    assert labor_privileged.RESERVED_HELDOUT_WORLDS is labor_worlds.RESERVED_HELDOUT_WORLDS
    assert list(labor_worlds.RESERVED_HELDOUT_WORLDS) == list(range(90154, 90160))
    assert labor_worlds.require_training_worlds([90100, 90153, 90160], "x") == [90100, 90153, 90160]
    for world in (90154, 90159):
        with pytest.raises(ValueError, match="reserved held-out"):
            labor_worlds.require_training_worlds([90100, world], "x")
    assert labor_worlds.require_held_out_test_worlds([90154, 90155], "x") == [90154, 90155]
    with pytest.raises(ValueError, match="only the reserved"):
        labor_worlds.require_held_out_test_worlds([90154, 90100], "x")
    with pytest.raises(ValueError, match="reserved held-out"):
        labor_privileged.require_critic_training_worlds([90157])
    tagged = {"options": {"partition": labor_worlds.HELD_OUT_TEST_PARTITION}}
    assert labor_worlds.is_held_out_test(tagged)
    assert labor_worlds.is_held_out_test({"partition": "held-out-test"})
    assert not labor_worlds.is_held_out_test({"options": {"partition": "fitting"}})
    with pytest.raises(ValueError, match="held-out-test"):
        labor_worlds.require_training_record(tagged, "x")


def test_collection_world_validation_refuses_reserved_worlds() -> None:
    assert validate_fitting_worlds([90100, 90153], "c") == [90100, 90153]
    with pytest.raises(ValueError, match="reserved held-out"):
        validate_fitting_worlds([90100, 90154], "c")
    assert validate_fitting_worlds([90154, 90159], "c", held_out_test=True) == [90154, 90159]
    with pytest.raises(ValueError, match="only the reserved"):
        validate_fitting_worlds([90100], "c", held_out_test=True)
    with pytest.raises(ValueError, match=r"denied|sealed"):
        validate_fitting_worlds([3000], "c", held_out_test=True)
    assert validate_search_worlds([90100, 90153]) == [90100, 90153]
    with pytest.raises(ValueError, match="reserved held-out"):
        validate_search_worlds([90100, 90159])
    assert validate_search_worlds([90156], held_out_test=True) == [90156]


def test_collection_clis_refuse_reserved_worlds_before_any_work(tmp_path) -> None:
    from anthropocene_rl.autonomous_v7 import labor_dagger, labor_search

    missing = str(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="reserved held-out"):
        labor_dagger.main(["collect", "--checkpoint", missing, "--worlds", "90150-90155",
                           "--round", "1", "--depth", "1", "--out", str(tmp_path / "d")])
    with pytest.raises(ValueError, match="reserved held-out"):
        labor_search.main(["--normalizer-checkpoint", missing, "--worlds", "90154",
                           "--out", str(tmp_path / "s")])
    with pytest.raises(ValueError, match="only the reserved"):
        labor_search.main(["--normalizer-checkpoint", missing, "--worlds", "90100",
                           "--held-out-test", "--out", str(tmp_path / "s")])
    with pytest.raises(ValueError, match="reserved held-out"):
        ltd.main(["--normalizer-checkpoint", missing, "--worlds", "90158",
                  "--out", str(tmp_path / "t")])
    assert not any(tmp_path.iterdir())


def test_actor_bc_refuses_reserved_rows_and_held_out_test_data() -> None:
    dataset = {
        "target_specialty": torch.zeros(2, 14),
        "row_world": torch.tensor([90100, 90155]),
    }
    with pytest.raises(ValueError, match="reserved held-out"):
        pretrain_labor_actor_bc(actor=None, dataset=dataset)
    tagged = {
        "target_specialty": torch.zeros(1, 14),
        "row_world": torch.tensor([90100]),
        "collection": {"partition": labor_worlds.HELD_OUT_TEST_PARTITION},
    }
    with pytest.raises(ValueError, match="held-out-test"):
        pretrain_labor_actor_bc(actor=None, dataset=tagged)


def _dataset(
    directory: Path, worlds, *, partition: str = "fitting", shard_partition: str | None = None
) -> Path:
    body = {
        "schema": ltd.SCHEMA,
        "source_kind": ltd.SOURCE_KIND,
        "options": {"turns": 1, "reset_options": {}, "partition": partition},
    }
    source = dict(body, source_run=digest(body))
    directory.mkdir(parents=True)
    shards = []
    for world in worlds:
        n = 2
        arrays = {
            "row_world": np.full(n, world, dtype=np.int64),
            "row_camp": np.arange(n, dtype=np.int64),
            "row_turn": np.zeros(n, dtype=np.int64),
            "row_attempt": np.zeros(n, dtype=np.int64),
            "row_quarantined": np.zeros(n, dtype=bool),
        }
        meta = {
            "source_run": source["source_run"],
            "world_seed": world,
            "normalizer_digest": "n",
            "counters": {},
            "label_unavailable_reasons": {},
            "quarantine": [],
            "partition": shard_partition or partition,
        }
        record = ltd.write_shard(directory / f"world-{world}", arrays, meta)
        shards.append(
            {
                "path": f"world-{world}",
                "world_seed": world,
                "meta_sha256": ltd.checkpoint_hash(directory / f"world-{world}" / "meta.json"),
                "data_digest": record["data_digest"],
            }
        )
    ltd.publish_dataset(directory, source, shards)
    return directory


def test_training_loaders_refuse_reserved_shards_unless_dropped(tmp_path) -> None:
    mixed = _dataset(tmp_path / "mixed", [90100, 90154])
    with pytest.raises(ValueError, match="reserved held-out"):
        ltd.load_labor_dataset(mixed)
    with pytest.raises(ValueError, match="reserved held-out"):
        ltd.load_labor_datasets([mixed, _dataset(tmp_path / "clean", [90101])])
    dropped = ltd.load_labor_dataset(mixed, drop_reserved=True)
    assert dropped["row_world"].tolist() == [90100, 90100]
    assert dropped["collection"]["dropped_reserved_worlds"] == [90154]
    assert dropped["collection"]["world_seeds"] == [90100]
    # Evaluation readers see everything.
    full = ltd.load_labor_dataset(mixed, training=False)
    assert sorted(set(full["row_world"].tolist())) == [90100, 90154]
    only_reserved = _dataset(tmp_path / "reserved", [90155])
    with pytest.raises(ltd.ProvenanceError, match="no training shards"):
        ltd.load_labor_dataset(only_reserved, drop_reserved=True)


def test_held_out_test_shards_are_refused_by_every_training_loader(tmp_path) -> None:
    test = _dataset(
        tmp_path / "test", [90154, 90155], partition=labor_worlds.HELD_OUT_TEST_PARTITION
    )
    for kwargs in ({}, {"drop_reserved": True}, {"require_privileged": False}):
        with pytest.raises(ValueError, match="held-out-test"):
            ltd.load_labor_dataset(test, **kwargs)
    with pytest.raises(ValueError, match="held-out-test"):
        ltd.load_labor_datasets([test, _dataset(tmp_path / "clean", [90101])])
    loaded = ltd.load_labor_dataset(test, training=False)
    assert loaded["collection"]["partition"] == labor_worlds.HELD_OUT_TEST_PARTITION
    # A test-tagged shard inside an otherwise fitting manifest is refused too.
    sneaky = _dataset(
        tmp_path / "sneaky", [90100], shard_partition=labor_worlds.HELD_OUT_TEST_PARTITION
    )
    with pytest.raises(ValueError, match="held-out-test"):
        ltd.load_labor_dataset(sneaky)
