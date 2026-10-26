"""Privileged normalizer schema v2: declared per-feature transforms, float-tolerance dead rule.

Local data showed the v1 rule (symlog std < 1e-3) zeroing fields that vary
but are tiny (per-day regrowth rates) or large and nearly constant (stocks),
and the structural overflow row-count. v2 declares a transform per feature and
zeroes a field only when it is constant within float tolerance after it.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from test_labor_privileged_bridge import FIT_WORLDS, _raw

from anthropocene_rl.autonomous_v7.labor_privileged import (
    CRITIC_PREFIX,
    LABOR_PRIVILEGED_LAYOUT_V1,
    LEGACY_PRIVILEGED_NORMALIZER_KIND,
    PRIVILEGED_NORMALIZER_KIND,
    PRIVILEGED_NORMALIZER_SCHEMA_VERSION,
    PRIVILEGED_TRANSFORMS,
    SHARD_PREFIX,
    LaborPrivilegedNormalizer,
    dead_mask_audit,
    transform_specs,
)

LAYOUT = LABOR_PRIVILEGED_LAYOUT_V1


def _column(family: str, feature: str) -> int:
    return LAYOUT.family(family).features.index(feature)


def _set(raw, family: str, feature: str, values) -> None:
    array = raw[SHARD_PREFIX + family]
    mask = raw[SHARD_PREFIX + family + "_present"]
    column = _column(family, feature)
    filled = torch.as_tensor(np.broadcast_to(values, mask.shape).copy(), dtype=array.dtype)
    array[..., column] = torch.where(mask, filled, torch.zeros_like(filled))


def _fixture(rows: int = 24):
    raw = _raw(rows, seed=4)
    rng = np.random.default_rng(0)
    shape = raw[SHARD_PREFIX + "fields_present"].shape
    _set(raw, "fields", "mean-food-recovery-per-day", rng.uniform(1e-4, 5e-4, size=shape))
    _set(raw, "fields", "land-condition", 1.0)
    _set(raw, "fields", "row-count", 1.0)
    eshape = raw[SHARD_PREFIX + "economy_present"].shape
    _set(raw, "economy", "stone", 5e6 + rng.normal(0.0, 1e3, size=eshape))
    _set(raw, "economy", "stock-HousingMaterial", rng.uniform(0.0, 1.4e-14, size=eshape))
    return raw


def test_transform_registry_names_real_fields_and_defaults_to_symlog() -> None:
    specs = transform_specs()
    assert set(specs) == {f.name for f in LAYOUT.families}
    for family in LAYOUT.families:
        assert len(specs[family.name]) == family.width
    assert specs["fields"][_column("fields", "mean-food-recovery-per-day")].startswith(
        "per-year-log1p"
    )
    assert specs["core"] == ("symlog",) * 10
    for family, names in PRIVILEGED_TRANSFORMS.items():
        if LAYOUT.family(family).overflow:
            assert names["row-count"] == "count-log1p"
    with pytest.raises(ValueError, match="unknown"):
        transform_specs(declared={"fields": {"no-such-field": "symlog"}})
    with pytest.raises(ValueError, match="unknown privileged transform"):
        transform_specs(declared={"fields": {"access": "cube-root"}})


def test_v2_keeps_tiny_and_near_constant_fields_that_v1_zeroed() -> None:
    raw = _fixture()
    v1 = LaborPrivilegedNormalizer.fit(raw, worlds=FIT_WORLDS, version=1)
    v2 = LaborPrivilegedNormalizer.fit(raw, worlds=FIT_WORLDS)
    assert v2.version == PRIVILEGED_NORMALIZER_SCHEMA_VERSION
    old, new = v1.dead_mask_report()["families"], v2.dead_mask_report()["families"]
    for family, name in (
        ("fields", "mean-food-recovery-per-day"),
        ("economy", "stone"),
        ("fields", "row-count"),
    ):
        assert name in old[family]["dead_fields"], name
        assert name not in new[family]["dead_fields"], name
    # Genuinely constant fields and float-rounding noise stay dead.
    assert "land-condition" in new["fields"]["dead_fields"]
    assert "stock-HousingMaterial" in new["economy"]["dead_fields"]
    assert v2.resolved["economy"][_column("economy", "stone")] == "identity"
    batch = v2.critic_batch(raw)
    present = batch[CRITIC_PREFIX + "economy_present"]
    stone = batch[CRITIC_PREFIX + "economy"][..., _column("economy", "stone")][present]
    deviation = raw[SHARD_PREFIX + "economy"][..., _column("economy", "stone")][present] - 5e6
    assert 0.5 < float(stone.std()) < 2.0
    assert float(np.corrcoef(stone.numpy(), deviation.double().numpy())[0, 1]) > 0.99
    fields_present = batch[CRITIC_PREFIX + "fields_present"]
    rate = batch[CRITIC_PREFIX + "fields"][..., _column("fields", "mean-food-recovery-per-day")]
    assert float(rate[fields_present].std()) > 0.5
    # The structural row-count is never zeroed: fixed log1p(count) on present rows.
    count = batch[CRITIC_PREFIX + "fields"][..., _column("fields", "row-count")]
    assert torch.allclose(count[fields_present], torch.full_like(count[fields_present], np.log(2)))
    assert torch.equal(count[~fields_present], torch.zeros_like(count[~fields_present]))


def test_v2_state_round_trips_and_rejects_tampering() -> None:
    raw = _fixture()
    v2 = LaborPrivilegedNormalizer.fit(raw, worlds=FIT_WORLDS)
    state = v2.state_dict()
    assert state["kind"] == PRIVILEGED_NORMALIZER_KIND
    assert state["normalizerSchemaVersion"] == 2
    restored = LaborPrivilegedNormalizer.from_state(state)
    assert restored.digest() == v2.digest()
    for key, value in v2.critic_batch(raw).items():
        assert torch.equal(value, restored.critic_batch(raw)[key]), key
    bad = copy.deepcopy(state)
    column = _column("economy", "stone")
    bad["families"]["economy"]["resolvedTransforms"][column] = "count-log1p"
    with pytest.raises(ValueError, match="resolved transform"):
        LaborPrivilegedNormalizer.from_state(bad)
    bad = copy.deepcopy(state)
    bad["normalizerSchemaVersion"] = 3
    with pytest.raises(ValueError, match="unsupported"):
        LaborPrivilegedNormalizer.from_state(bad)
    bad = copy.deepcopy(state)
    bad["deadRelativeTolerance"] = 1e-3
    with pytest.raises(ValueError, match="numeric contract"):
        LaborPrivilegedNormalizer.from_state(bad)


def test_legacy_v1_states_load_and_normalize_exactly_as_before() -> None:
    raw = _fixture()
    v1 = LaborPrivilegedNormalizer.fit(raw, worlds=FIT_WORLDS, version=1)
    state = v1.state_dict()
    assert state["kind"] == LEGACY_PRIVILEGED_NORMALIZER_KIND
    assert "normalizerSchemaVersion" not in state  # the original v1 field set
    restored = LaborPrivilegedNormalizer.from_state(state)
    assert restored.version == 1 and restored.digest() == v1.digest()
    batch = restored.critic_batch(raw)
    for family in LAYOUT.families:
        x = raw[SHARD_PREFIX + family.name].double().numpy()
        mask = raw[SHARD_PREFIX + family.name + "_present"].numpy()
        z = (np.sign(x) * np.log1p(np.abs(x)) - v1.mean[family.name]) / v1.std[family.name]
        z = np.where(v1.dead[family.name], 0.0, np.clip(z, -10, 10))
        z = np.where(mask[..., None], z, 0.0)
        assert torch.equal(
            batch[CRITIC_PREFIX + family.name], torch.as_tensor(z, dtype=torch.float32)
        )


def test_audit_reports_a_reason_for_every_dead_field() -> None:
    raw = _fixture()
    audit = dead_mask_audit(raw)
    fields = audit["families"]["fields"]["fields"]
    assert fields["land-condition"]["reason"] == "constant-in-data"
    assert fields["land-condition"]["constant_value"] == 1.0
    assert fields["mean-food-recovery-per-day"]["rescued_by_v2"]
    assert fields["row-count"]["dead"] is False and fields["row-count"]["legacy_dead"]
    economy = audit["families"]["economy"]["fields"]
    assert economy["stock-HousingMaterial"]["reason"] == "varies-only-at-float-rounding"
    assert economy["stone"]["transform"] == "identity" and economy["stone"]["rescued_by_v2"]
    for family in audit["families"].values():
        for name in family["dead_fields"]:
            assert family["fields"][name]["reason"] is not None
    assert audit["families"]["fields"]["overflow_rows_present"] == 0
    assert audit["rescued_total"] >= 3
