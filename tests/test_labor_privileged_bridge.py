"""Intervention A (learner): the critic-only privileged view, bridge to normalizer.

Layout v1 (rows per family), the bridge's separate critic-only batcher, the
frozen per-family normalizer with its dead-mask report, and the fit-world
policy (never sealed, development, reserved 90154-90159 or calibration worlds).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from test_autonomous_v7_round3_search import observation as legacy_observation

from anthropocene_rl.autonomous_v7.bridge import DEFAULT_BRIDGE, observation_batch, privileged_batch
from anthropocene_rl.autonomous_v7.labor_privileged import (
    CRITIC_PREFIX,
    LABOR_PRIVILEGED_LAYOUT_V1,
    SHARD_PREFIX,
    SHARD_SCHEMA_KEY,
    LaborPrivilegedLayout,
    LaborPrivilegedNormalizer,
    dead_mask_audit,
    privileged_raw_batch,
    require_critic_training_worlds,
    shard_keys,
)

LAYOUT = LABOR_PRIVILEGED_LAYOUT_V1
FIT_WORLDS = [90100, 90101]


def _view(seed: int, *, scale: float = 1.0) -> dict:
    """A bridge-shaped privilegedView: every family partly present, absent rows zero."""
    rng = np.random.default_rng(seed)
    rows, present = [], []
    for family in LAYOUT.families:
        mask = np.zeros(family.rows, dtype=bool)
        mask[: max(1, family.cap // 2)] = True
        values = scale * rng.uniform(0.0, 5.0, size=(family.rows, family.width)).astype(np.float32)
        values[~mask] = 0.0
        rows.append(values.reshape(-1).tolist())
        present.append(mask.tolist())
    return {"schemaVersion": 1, "rows": rows, "present": present}


def _raw(batch: int, seed: int = 0, *, constant_column: tuple[str, int] | None = None) -> dict:
    villages = [
        {"privilegedSchemaVersion": 1, "privilegedView": _view(seed + row)} for row in range(batch)
    ]
    raw = privileged_raw_batch(villages)
    if constant_column is not None:
        name, column = constant_column
        values = raw[SHARD_PREFIX + name]
        mask = raw[SHARD_PREFIX + name + "_present"]
        values[..., column] = np.where(mask, 3.0, 0.0)
    return {key: torch.from_numpy(value) for key, value in raw.items()}


def _normalizer(seed: int = 0) -> LaborPrivilegedNormalizer:
    return LaborPrivilegedNormalizer.fit(_raw(16, seed), worlds=FIT_WORLDS)


def test_privileged_layout_v1_rows_and_widths() -> None:
    rows = {f.name: (f.rows, f.width, f.overflow) for f in LAYOUT.families}
    assert rows == {
        "core": (1, 10, False),
        "fields": (17, 19, True),
        "herds": (9, 24, True),
        "unlocks": (2, 28, True),
        "economy": (2, 64, True),
    }
    assert LAYOUT.total_rows == 31
    assert LAYOUT.schema_version == 1
    assert LaborPrivilegedLayout.from_state(LAYOUT.state_dict()) == LAYOUT
    future = LAYOUT.state_dict()
    future["schemaVersion"] = 2
    with pytest.raises(ValueError, match="schema version"):
        LaborPrivilegedLayout.from_state(future)


def test_observation_batch_drops_privileged_and_critic_batcher_parses_it() -> None:
    raw = legacy_observation(seed=3)
    village = {
        "observation": raw,
        "privileged": [1.0] * 10,
        "privilegedSchemaVersion": 1,
        "privilegedView": _view(5),
    }
    batch = observation_batch([village])
    assert not any(key.startswith("privileged") for key in batch)
    critic = privileged_batch([village])
    assert set(critic) == set(shard_keys())
    assert critic["privileged_fields"].shape == (1, 17, 19)
    with pytest.raises(ValueError, match="schema"):
        privileged_batch([dict(village, privilegedSchemaVersion=2)])
    with pytest.raises(ValueError, match="no privileged view"):
        privileged_batch([{"observation": raw}])
    broken = copy.deepcopy(village)
    broken["privilegedView"]["rows"][1][-1] = 9.0  # absent overflow row must be zero
    with pytest.raises(ValueError, match="absent rows must be zero"):
        privileged_batch([broken])


@pytest.mark.skipif(not DEFAULT_BRIDGE.is_file(), reason="model9 bridge is not built")
def test_live_bridge_advertises_and_emits_layout_v1() -> None:
    from anthropocene_rl.autonomous_v7.bridge import Model9Bridge

    with Model9Bridge() as bridge:
        if bridge.privileged_layout is None:
            pytest.skip("bridge predates the privileged view")
        assert bridge.privileged_layout == LAYOUT
        step = bridge.reset(90100)
        live = [v for v in step["villages"] if v.get("observation") is not None]
        raw = privileged_batch(live)
        assert raw[SHARD_SCHEMA_KEY].tolist() == [1] * len(live)
        assert raw["privileged_core_present"].all()


def test_normalizer_dead_mask_report_names_constant_fields() -> None:
    raw = _raw(12, constant_column=("herds", 2))  # herds species-index constant
    normalizer = LaborPrivilegedNormalizer.fit(raw, worlds=FIT_WORLDS)
    report = normalizer.dead_mask_report()
    assert report["schema_version"] == 1
    assert "species-index" in report["families"]["herds"]["dead_fields"]
    assert report["families"]["fields"]["dead_count"] == 0
    assert report["dead_total"] >= 1
    audit = dead_mask_audit(raw)
    assert audit["families"]["herds"]["fields"]["species-index"]["constant_value"] == 3.0
    assert "species-index" in audit["families"]["herds"]["dead_fields"]
    batch = normalizer.critic_batch(raw)
    assert torch.equal(batch[CRITIC_PREFIX + "herds"][..., 2], torch.zeros(12, 9))
    # Absent rows stay exactly zero after normalization.
    absent = ~batch[CRITIC_PREFIX + "fields_present"]
    assert torch.equal(batch[CRITIC_PREFIX + "fields"][absent], torch.zeros(int(absent.sum()), 19))
    # Frozen: the state round-trips exactly and a later fit does not alter it.
    restored = LaborPrivilegedNormalizer.from_state(normalizer.state_dict())
    assert restored.digest() == normalizer.digest()


@pytest.mark.parametrize("world", [90154, 90159, 90180, 3000, 2000])
def test_privileged_normalizer_fits_on_fitting_worlds_only(world) -> None:
    with pytest.raises(ValueError, match="refuses"):
        LaborPrivilegedNormalizer.fit(_raw(4), worlds=[90100, world])
    with pytest.raises(ValueError, match="calibration"):
        LaborPrivilegedNormalizer.fit(_raw(4), worlds=[90100, 90120], excluded_worlds=[90120])


def test_reserved_and_calibration_worlds_never_train_a_critic() -> None:
    require_critic_training_worlds([90100, 90153, 90160])
    for world in (90154, 90157, 90185):
        with pytest.raises(ValueError, match="refuses"):
            require_critic_training_worlds([90100, world])
