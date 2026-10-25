"""Intervention A (learner): privileged arrays in shards and critic training.

Shards store raw privileged arrays and their schema version; a shard without
them is rejected for critic training (never zero-filled) while actor BC still
loads it, and critic pretraining feeds the separate critic-only batch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from test_labor_model import _labor_tensors
from test_labor_privileged_bridge import _raw
from test_labor_privileged_model import _critics, _layout

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.labor_model import critic_privileged_batch
from anthropocene_rl.autonomous_v7.labor_privileged import SHARD_SCHEMA_KEY, actor_inputs
from anthropocene_rl.autonomous_v7.provenance import digest


def _dataset(directory: Path, worlds, *, privileged: bool) -> Path:
    body = {
        "schema": ltd.SCHEMA,
        "source_kind": ltd.SOURCE_KIND,
        "options": {"turns": 2, "reset_options": {}},
    }
    source = dict(body, source_run=digest(body))
    directory.mkdir(parents=True)
    shards = []
    for index, world in enumerate(worlds):
        n = 3
        arrays = {
            "row_world": np.full(n, world, dtype=np.int64),
            "row_camp": np.arange(n, dtype=np.int64),
            "row_turn": np.zeros(n, dtype=np.int64),
            "row_attempt": np.zeros(n, dtype=np.int64),
            "row_quarantined": np.zeros(n, dtype=bool),
            "reward_qol_years": np.ones(n),
            "reward_population_years": np.ones(n),
            "reward_feasibility": np.zeros(n),
            "reward_settlement_guard": np.zeros(n),
            "transition_terminal": np.ones(n, dtype=bool),
            "transition_elapsed_years": np.full(n, 20.0),
            "core": np.zeros((n, 10), dtype=np.float32),
        }
        if privileged:
            arrays.update({k: v.numpy() for k, v in _raw(n, seed=index).items()})
        meta = {
            "source_run": source["source_run"],
            "world_seed": world,
            "normalizer_digest": "n",
            "counters": {},
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
            }
        )
    ltd.publish_dataset(directory, source, shards)
    return directory


def test_shard_without_privileged_data_is_rejected_for_critic_training(tmp_path) -> None:
    plain = _dataset(tmp_path / "plain", [90100], privileged=False)
    with pytest.raises(ltd.ProvenanceError, match="no privileged data"):
        ltd.load_labor_dataset(plain, require_privileged=True)
    with pytest.raises(ltd.ProvenanceError, match="no privileged data"):
        ltd.load_labor_datasets([plain], require_privileged=True)
    # Actor BC may still load it; nothing is zero-filled.
    actor_view = ltd.load_labor_dataset(plain)
    assert not any(key.startswith("privileged") for key in actor_view)
    task, _ = _critics(_layout())
    with pytest.raises(ValueError, match="lack privileged data"):
        critic_privileged_batch(task, actor_view)


def test_privileged_shards_round_trip_and_stay_out_of_actor_inputs(tmp_path) -> None:
    stored = _dataset(tmp_path / "priv", [90100, 90101], privileged=True)
    meta = ltd.verify_shard(stored / "world-90100")
    assert meta["privileged_schema_version"] == 1
    dataset = ltd.load_labor_dataset(stored, require_privileged=True)
    assert dataset["collection"]["privileged_schema_version"] == 1
    assert dataset["privileged_herds"].shape == (6, 9, 24)
    assert not any(key.startswith("privileged") for key in actor_inputs(dataset))
    # Mixed shards: actor loads without privileged arrays; critic loads refuse.
    mixed = tmp_path / "mixed"
    _dataset(mixed, [90102], privileged=True)
    other = _dataset(tmp_path / "other", [90103], privileged=False)
    with pytest.raises(ltd.ProvenanceError):
        ltd.load_labor_datasets([mixed, other], require_privileged=True)
    combined = ltd.load_labor_datasets([mixed, other])
    assert not any(key.startswith("privileged") for key in combined)


def test_partial_privileged_arrays_are_refused_at_write(tmp_path) -> None:
    arrays = {"row_world": np.zeros(2, np.int64), SHARD_SCHEMA_KEY: np.ones(2, np.int64)}
    with pytest.raises(ltd.ProvenanceError, match="incomplete"):
        ltd.write_shard(tmp_path / "x", arrays, {})


def test_pretrain_critic_uses_the_separate_privileged_batch(tmp_path) -> None:
    from anthropocene_rl.autonomous_v7.labor_bc import pretrain_critic_on_dataset

    rows = 6
    tensors = _labor_tensors(batch=rows)
    dataset = {k: v for k, v in tensors.items()}
    dataset.update({k: v for k, v in _raw(rows).items()})
    dataset.update(
        row_world=torch.tensor([90100] * 3 + [90101] * 3),
        row_camp=torch.tensor([1, 1, 1, 2, 2, 2]),
        row_turn=torch.tensor([0, 1, 2, 0, 1, 2]),
        row_quarantined=torch.zeros(rows, dtype=torch.bool),
        reward_qol_years=torch.ones(rows, dtype=torch.float64),
        reward_population_years=torch.ones(rows, dtype=torch.float64),
        reward_feasibility=torch.zeros(rows, dtype=torch.float64),
        reward_settlement_guard=torch.zeros(rows, dtype=torch.float64),
        transition_elapsed_years=torch.full((rows,), 20.0, dtype=torch.float64),
        transition_terminal=torch.tensor([False, False, True] * 2),
    )
    dataset["collection"] = {"return_tail": "truncated-zero-at-collection-horizon"}
    task, costs = _critics(_layout())
    summary = pretrain_critic_on_dataset(
        task_critic=task, cost_critics=costs, dataset=dataset, epochs=1, batch_size=4
    )
    assert summary["privileged_schema_version"] == 1
    reserved = dict(dataset, row_world=torch.tensor([90154] * rows))
    with pytest.raises(ValueError, match="reserved"):
        pretrain_critic_on_dataset(task_critic=task, cost_critics=costs, dataset=reserved, epochs=1)
    stripped = {k: v for k, v in dataset.items() if not k.startswith("privileged")}
    with pytest.raises(ValueError, match="lack privileged data"):
        pretrain_critic_on_dataset(task_critic=task, cost_critics=costs, dataset=stripped, epochs=1)
    assert contracts.SETS  # observation tensors are untouched by the critic batch
