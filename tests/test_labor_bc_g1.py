"""G1 corrective fixtures for heuristic labor BC (brief v1.9 sections 4.2, 5.1, 5.4)."""

from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts, labor_bc
from anthropocene_rl.autonomous_v7.labor_bc import (
    LaborResetOptions,
    _batch_loss_and_l1,
    convert_heuristic_labor_label,
    discounted_returns,
    masked_allocation_cross_entropy,
    validate_fitting_worlds,
    world_balanced_weights,
)
from anthropocene_rl.autonomous_v7.labor_bridge import (
    LABOR_ACTIVITY_FEATURE_WIDTH,
    LABOR_RESOURCE_FEATURE_WIDTH,
    LABOR_SPECIALTY_FEATURE_WIDTH,
    LaborCatalog,
    labor_catalog_digest,
)
from anthropocene_rl.autonomous_v7.labor_model import LaborActor, LaborEncoderLayout
from anthropocene_rl.autonomous_v7.labor_ppo import validate_schema12_action
from tests.test_labor_bridge import _catalog_payload


def _catalog(population: int = 1, years: float = 20.0) -> LaborCatalog:
    raw = _catalog_payload()
    raw["population"] = population
    raw["turnYears"] = years
    raw["digest"] = labor_catalog_digest(raw)
    return LaborCatalog.from_payload(raw)


def _village(population: int) -> dict:
    core = [0.0] * 218
    return {
        "camp": 7,
        "population": population,
        "observation": {
            "core": core,
            "corePresent": [True] * 218,
            "movementMask": [True] + [False] * 7,
            "regionMovementMask": [True] + [False] * 9,
        },
        "teacher": {
            "movement": -1,
            "movementLabel": True,
            "fission": 0,
            "tradeValue": [1.0] * 7,
            "tradeGate": [2] * 7,
            "tradeShare": 0.5,
            "tradeLabel": True,
        },
    }


def _herding(assigned: float, management: float, development: float, covered: float) -> dict:
    return {
        "assignedHerderResidents": assigned,
        "managementHerderResidents": assigned,
        "developmentHerderResidents": 0.0,
        "managementHerderWorkerYears": management,
        "developmentHerderWorkerYears": development,
        "assignedHerderWorkerYears": management + development,
        "plannedHerderYears": management,
        "coveredHerderYears": covered,
        "funnels": [],
    }


def _diag(herding: dict, goods: list | None = None) -> dict:
    return {
        "domains": {
            "cultivation": {"assignedFarmerYears": 0.0, "expansionWorkerYears": 0.0},
            "herding": herding,
        },
        "goods": goods or [],
        "materials": [],
        "requests": [],
    }


# Task 1 ---------------------------------------------------------------------


def test_stable_allocation_ce_keeps_saturated_positive_target_gradient() -> None:
    logits = torch.tensor([[0.0, 40.0, 0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.90, 0.05, 0.05]])
    mask = torch.ones_like(target, dtype=torch.bool)
    smoothing = 1e-6
    smoothed = (1.0 - smoothing) * target + smoothing * mask.float() / 4.0
    masked_allocation_cross_entropy(logits, smoothed, mask).sum().backward()
    farmer = float(logits.grad[0, 2])
    assert math.isclose(farmer, -0.0500002, rel_tol=0.0, abs_tol=1e-7)

    # The clamped formula it replaces loses that corrective gradient.
    legacy_logits = logits.detach().clone().requires_grad_(True)
    probabilities = torch.softmax(legacy_logits, dim=-1)
    (-(smoothed * torch.log(probabilities.clamp(min=1e-12))).sum()).backward()
    assert abs(float(legacy_logits.grad[0, 2])) < 1e-12


def test_stable_allocation_ce_ignores_illegal_and_empty_rows() -> None:
    logits = torch.tensor([[0.0, 1.0, 50.0], [3.0, -2.0, 0.5]], requires_grad=True)
    target = torch.tensor([[0.25, 0.75, 0.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [False, False, False]])
    loss = masked_allocation_cross_entropy(logits, target, mask)
    assert torch.isfinite(loss).all()
    assert float(loss[1]) == 0.0
    expected = -(0.25 * math.log(1 / (1 + math.e)) + 0.75 * math.log(math.e / (1 + math.e)))
    assert math.isclose(float(loss[0]), expected, rel_tol=1e-6)
    loss.sum().backward()
    assert float(logits.grad[0, 2]) == 0.0 and torch.all(logits.grad[1] == 0)


# Task 2 ---------------------------------------------------------------------


def test_identical_herders_with_different_tools_have_identical_staffing_labels() -> None:
    catalog = _catalog(population=10)
    village = _village(10)
    well_tooled = convert_heuristic_labor_label(
        catalog, village, _diag(_herding(3, 60.0, 0.0, 60.0))
    )
    no_tools = convert_heuristic_labor_label(catalog, village, _diag(_herding(3, 60.0, 0.0, 12.5)))
    assert well_tooled.specialty_valid and no_tools.specialty_valid
    assert np.array_equal(well_tooled.specialty, no_tools.specialty)
    assert np.array_equal(well_tooled.activity, no_tools.activity)
    assert np.array_equal(well_tooled.specialty_counts, no_tools.specialty_counts)
    assert well_tooled.specialty_counts[3] == 3
    assert math.isclose(well_tooled.activity[2, 7], 1.0)


def test_missing_herder_staffing_is_unavailable_not_zero() -> None:
    catalog = _catalog(population=10)
    herding = {"plannedHerderYears": 20.0, "coveredHerderYears": 20.0, "funnels": []}
    label = convert_heuristic_labor_label(catalog, _village(10), _diag(herding))
    assert not label.specialty_valid
    assert "herder-staffing-unavailable" in label.reasons
    # A recorded zero stays a valid zero.
    zero = convert_heuristic_labor_label(catalog, _village(10), _diag(_herding(0, 0.0, 0.0, 0.0)))
    assert zero.specialty_valid and zero.specialty_counts[3] == 0


# Task 4 ---------------------------------------------------------------------


def test_one_worker_activity_shares_preserve_idle_time() -> None:
    catalog = _catalog(population=1)
    goods = [
        {"good": "HandTools", "workerYears": 10.0, "plannedOutputCeiling": 0.0},
        {"good": "Pots", "workerYears": 5.0, "plannedOutputCeiling": 0.0},
    ]
    label = convert_heuristic_labor_label(catalog, _village(1), _diag(_herding(0, 0, 0, 0), goods))
    assert label.specialty_valid, label.reasons
    assert label.specialty_counts[6] == 1 and label.specialty_counts.sum() == 1
    assert math.isclose(label.activity[5, 11], 0.50, abs_tol=1e-12)
    assert math.isclose(label.activity[5, 12], 0.25, abs_tol=1e-12)
    assert math.isclose(label.activity[5, 0], 0.25, abs_tol=1e-12)
    # Executable and identical to the label.
    decoded = validate_schema12_action(label.action, catalog, retry=False)
    assert np.allclose(decoded.activity_allocation[5], label.activity[5])


def test_unknown_or_unmapped_work_is_unavailable_not_gatherer() -> None:
    catalog = _catalog(population=5)
    goods = [{"good": "Mystery", "workerYears": 8.0, "plannedOutputCeiling": 0.0}]
    label = convert_heuristic_labor_label(catalog, _village(5), _diag(_herding(0, 0, 0, 0), goods))
    assert not label.specialty_valid
    assert any(reason.startswith("craft-work-unmapped") for reason in label.reasons)


def test_native_residual_gathering_stays_gatherer_work() -> None:
    catalog = _catalog(population=4)
    label = convert_heuristic_labor_label(catalog, _village(4), _diag(_herding(1, 20.0, 0.0, 5.0)))
    assert label.specialty_valid
    assert label.specialty_counts[1] == 3 and label.specialty_counts[3] == 1
    assert math.isclose(label.activity[0, 1], 1.0)


def test_over_budget_role_work_is_flagged_not_rescaled() -> None:
    catalog = _catalog(population=2)
    label = convert_heuristic_labor_label(catalog, _village(2), _diag(_herding(1, 30.0, 0.0, 0.0)))
    assert label.over_budget and not label.specialty_valid
    assert any(reason.startswith("activity-over-budget") for reason in label.reasons)


# Task 7 ---------------------------------------------------------------------


def test_movement_label_resolves_against_pre_action_catalog() -> None:
    catalog = _catalog(population=4)
    village = _village(4)
    village["teacher"]["movement"] = 2
    shadow = copy.deepcopy(village["observation"])
    shadow["movementMask"] = [True] * 8  # successor mask would allow it
    pre = copy.deepcopy(village["observation"])
    village["observation"] = shadow
    label = convert_heuristic_labor_label(
        catalog, village, _diag(_herding(0, 0, 0, 0)), pre_observation=pre
    )
    assert not label.movement_valid and label.action["movement"] == -1
    assert label.specialty_valid  # labor label is independent of the movement gap
    village["teacher"]["movement"] = -2
    sentinel = convert_heuristic_labor_label(
        catalog, village, _diag(_herding(0, 0, 0, 0)), pre_observation=pre
    )
    assert not sentinel.movement_valid


def test_missing_trade_label_is_invalid_not_default_filled() -> None:
    catalog = _catalog(population=4)
    village = _village(4)
    del village["teacher"]["tradeGate"]
    label = convert_heuristic_labor_label(catalog, village, _diag(_herding(0, 0, 0, 0)))
    assert not label.trade_gate_valid.any() and not label.trade_value_active.any()
    assert not label.trade_share_active
    village = _village(4)
    village["teacher"]["tradeGate"] = [0, 2, 0, 1, 0, 0, 0]
    label = convert_heuristic_labor_label(catalog, village, _diag(_herding(0, 0, 0, 0)))
    assert label.trade_gate_valid.all()
    assert label.trade_value_active.tolist() == [False, True, False, True, False, False, False]
    assert label.trade_share_active


def _synthetic_batch(B: int = 6) -> tuple[LaborActor, dict]:
    torch.manual_seed(3)
    core_dim = 15
    layout = LaborEncoderLayout(
        core_dim,
        LABOR_SPECIALTY_FEATURE_WIDTH,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
    )
    actor = LaborActor(layout)
    target_specialty = torch.zeros(B, 14)
    target_specialty[:, 1] = 0.5
    target_specialty[:, 2] = 0.5
    target_activity = torch.zeros(B, 13, 22)
    target_activity[:, :, 0] = 1.0
    batch = {
        "core": torch.randn(B, core_dim),
        "labor_personality": torch.full((B, 2), 0.5),
        "movement_mask": torch.ones(B, 8, dtype=torch.bool),
        "region_movement_mask": torch.ones(B, 10, dtype=torch.bool),
        "labor_specialty_features": torch.randn(B, 14, LABOR_SPECIALTY_FEATURE_WIDTH),
        "labor_specialty_present": torch.ones(B, 14, dtype=torch.bool),
        "labor_specialty_mask": torch.ones(B, 14, dtype=torch.bool),
        "labor_activity_features": torch.randn(B, 13 * 22, LABOR_ACTIVITY_FEATURE_WIDTH),
        "labor_activity_present": torch.ones(B, 13 * 22, dtype=torch.bool),
        "labor_activity_mask": torch.ones(B, 13, 22, dtype=torch.bool),
        "labor_resource_features": torch.randn(B, 20, LABOR_RESOURCE_FEATURE_WIDTH),
        "labor_resource_present": torch.ones(B, 20, dtype=torch.bool),
        "target_specialty": target_specialty,
        "target_activity": target_activity,
        "target_movement_code": torch.full((B,), -1, dtype=torch.long),
        "target_fission": torch.zeros(B, dtype=torch.long),
        "target_trade_value_latent": torch.zeros(B, 7),
        "target_trade_gate": torch.ones(B, 7, dtype=torch.long),
        "target_trade_share_latent": torch.zeros(B),
    }
    for name, rows, columns in contracts.SETS:
        batch[name] = torch.randn(B, rows, columns)
        batch[name + "_present"] = torch.ones(B, rows, dtype=torch.bool)
    return actor, batch


def test_unmatched_movement_label_contributes_no_loss() -> None:
    actor, batch = _synthetic_batch()
    batch["target_movement_code"] = torch.full((len(batch["core"]),), 999, dtype=torch.long)
    _, _, _, components = _batch_loss_and_l1(actor, batch, target_log_std=-2.0)
    assert components["count_movement"] == 0 and components["movement"] == 0.0
    batch["target_movement_code"][:] = -1
    batch["label_movement_valid"] = torch.zeros(len(batch["core"]), dtype=torch.bool)
    _, _, _, components = _batch_loss_and_l1(actor, batch, target_log_std=-2.0)
    assert components["count_movement"] == 0 and components["movement"] == 0.0


def test_inactive_trade_labels_are_masked_out_of_trade_losses() -> None:
    actor, batch = _synthetic_batch()
    B = len(batch["core"])
    batch["label_trade_gate_valid"] = torch.ones(B, 7, dtype=torch.bool)
    batch["label_trade_value_active"] = torch.zeros(B, 7, dtype=torch.bool)
    batch["label_trade_value_active"][:, 2] = True
    batch["label_trade_share_active"] = torch.zeros(B, dtype=torch.bool)
    base = _batch_loss_and_l1(actor, batch, target_log_std=-2.0)
    changed = dict(batch)
    changed["target_trade_value_latent"] = batch["target_trade_value_latent"].clone()
    changed["target_trade_value_latent"][:, [0, 1, 3, 4, 5, 6]] = 5.0
    changed["target_trade_share_latent"] = torch.full((B,), 7.0)
    other = _batch_loss_and_l1(actor, changed, target_log_std=-2.0)
    assert math.isclose(float(base[0]), float(other[0]), rel_tol=0, abs_tol=1e-6)
    assert base[3]["count_trade_share"] == 0
    changed["target_trade_value_latent"][:, 2] = 3.0
    moved = _batch_loss_and_l1(actor, changed, target_log_std=-2.0)
    assert float(moved[0]) > float(base[0])
    # Invalid gates are excluded from gate CE as well.
    batch["label_trade_gate_valid"] = torch.zeros(B, 7, dtype=torch.bool)
    assert _batch_loss_and_l1(actor, batch, target_log_std=-2.0)[3]["trade_gate"] == 0.0


def test_quarantined_and_invalid_rows_contribute_no_labor_loss() -> None:
    actor, batch = _synthetic_batch()
    B = len(batch["core"])
    batch["row_quarantined"] = torch.ones(B, dtype=torch.bool)
    _, _, _, components = _batch_loss_and_l1(actor, batch, target_log_std=-2.0)
    assert components["specialty"] == 0.0 and components["count_specialty"] == 0
    batch["row_quarantined"][:] = False
    batch["label_specialty_valid"] = torch.zeros(B, dtype=torch.bool)
    _, _, _, components = _batch_loss_and_l1(actor, batch, target_log_std=-2.0)
    assert components["specialty"] == 0.0 and components["activity"] == 0.0


# Task 6 ---------------------------------------------------------------------


def test_world_balanced_weights_equalize_worlds() -> None:
    worlds = torch.tensor([5, 5, 5, 9])
    weights = world_balanced_weights(worlds)
    assert math.isclose(float(weights[:3].sum()), float(weights[3:].sum()))
    assert math.isclose(float(weights.mean()), 1.0)
    keep = torch.tensor([True, False, True, True])
    kept = world_balanced_weights(worlds, keep)
    assert float(kept[1]) == 0.0 and math.isclose(float(kept[[0, 2]].sum()), float(kept[3]))


def test_discounted_returns_follow_each_camp_trajectory() -> None:
    dataset = {
        "row_world": torch.tensor([1, 1, 1]),
        "row_camp": torch.tensor([4, 4, 8]),
        "row_turn": torch.tensor([2, 1, 1]),
        "transition_elapsed_years": torch.tensor([400.0, 400.0, 400.0], dtype=torch.float64),
        "transition_terminal": torch.tensor([False, False, True]),
        "reward_qol_years": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        "reward_population_years": torch.zeros(3, dtype=torch.float64),
        "reward_feasibility": torch.zeros(3, dtype=torch.float64),
        "reward_settlement_guard": torch.zeros(3, dtype=torch.float64),
    }
    returns = discounted_returns(dataset)["qol_years"]
    assert math.isclose(float(returns[0]), 1.0)
    assert math.isclose(float(returns[1]), 2.0 + math.exp(-1.0) * 1.0, rel_tol=1e-6)
    assert math.isclose(float(returns[2]), 3.0)


def test_elapsed_field_is_years_and_legacy_days_name_loads_unchanged(tmp_path) -> None:
    from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
    from anthropocene_rl.autonomous_v7.provenance import digest

    # The field always held physical years: 20 per turn, discounted at tau = 400 years.
    assert labor_bc.CRITIC_DISCOUNT_YEARS == contracts.HORIZON_YEARS == 400.0
    assert labor_bc.CRITIC_DISCOUNT_DAYS == labor_bc.CRITIC_DISCOUNT_YEARS
    assert labor_bc.ELAPSED_YEARS_FIELD == "transition_elapsed_years"

    def arrays(elapsed_key: str, world: int) -> dict[str, np.ndarray]:
        return {
            "row_world": np.full(2, world, dtype=np.int64),
            "row_camp": np.array([4, 4], dtype=np.int64),
            "row_turn": np.array([0, 1], dtype=np.int64),
            "row_attempt": np.zeros(2, dtype=np.int64),
            "row_quarantined": np.zeros(2, dtype=bool),
            "reward_qol_years": np.array([1.0, 2.0]),
            "reward_population_years": np.zeros(2),
            "reward_feasibility": np.zeros(2),
            "reward_settlement_guard": np.zeros(2),
            "transition_terminal": np.array([False, True]),
            elapsed_key: np.full(2, contracts.TURN_YEARS),
        }

    body = {
        "schema": ltd.SCHEMA,
        "source_kind": ltd.SOURCE_KIND,
        "options": {"turns": 2, "reset_options": {}},
    }
    source = dict(body, source_run=digest(body))
    out = tmp_path / "mixed"
    out.mkdir()
    shards = []
    for world, key in ((5000, "transition_elapsed_days"), (5001, "transition_elapsed_years")):
        meta = {
            "source_run": source["source_run"],
            "world_seed": world,
            "normalizer_digest": "n",
            "counters": {},
            "label_unavailable_reasons": {},
            "quarantine": [],
        }
        record = ltd.write_shard(out / f"world-{world}", arrays(key, world), meta)
        shards.append(
            {
                "path": f"world-{world}",
                "world_seed": world,
                "meta_sha256": ltd.checkpoint_hash(out / f"world-{world}" / "meta.json"),
                "data_digest": record["data_digest"],
            }
        )
    ltd.publish_dataset(out, source, shards)
    dataset = ltd.load_labor_dataset(out)
    assert "transition_elapsed_days" not in dataset
    assert torch.equal(
        dataset["transition_elapsed_years"],
        torch.full((4,), contracts.TURN_YEARS, dtype=torch.float64),
    )
    returns = discounted_returns(dataset)["qol_years"]
    assert math.isclose(float(returns[0]), 1.0 + math.exp(-20.0 / 400.0) * 2.0, rel_tol=1e-6)
    # In-memory legacy datasets still discount, and the migration keeps values unchanged.
    legacy = {k: torch.from_numpy(v) for k, v in arrays("transition_elapsed_days", 5000).items()}
    assert torch.equal(discounted_returns(legacy)["qol_years"], returns[:2])
    migrated = labor_bc.migrate_legacy_elapsed_field(dict(legacy))
    renamed = (set(legacy) - {"transition_elapsed_days"}) | {"transition_elapsed_years"}
    assert set(migrated) == renamed
    clash = dict(legacy, transition_elapsed_years=torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="different values"):
        labor_bc.migrate_legacy_elapsed_field(clash)


def test_pretrain_critic_rejects_search_dagger_and_aggregated_shards() -> None:
    from anthropocene_rl.autonomous_v7 import labor_dagger, labor_search
    from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd

    assert labor_bc.CRITIC_TEACHER_SOURCE_KIND == ltd.SOURCE_KIND
    assert ltd.ROW_SOURCES[labor_search.SOURCE_KIND] == 2
    teacher = {
        "collection": {"source_kind": ltd.SOURCE_KIND, "actor_only": False},
        "row_source": torch.zeros(3, dtype=torch.long),
    }
    labor_bc.require_critic_teacher_dataset(teacher)
    labor_bc.require_critic_teacher_dataset({"collection": {}})  # in-process heuristic collection
    search = {"collection": {"source_kind": labor_search.SOURCE_KIND, "actor_only": False}}
    with pytest.raises(ValueError, match="search shards"):
        labor_bc.require_critic_teacher_dataset(search)
    # The library entry point refuses before computing any target.
    with pytest.raises(ValueError, match="search shards"):
        labor_bc.pretrain_critic_on_dataset(
            task_critic=None,
            cost_critics=None,
            dataset=search,
            epochs=0,  # type: ignore[arg-type]
        )
    dagger = {"collection": {"source_kind": labor_dagger.DAGGER_SOURCE_KIND, "actor_only": True}}
    with pytest.raises(ValueError, match="actor-only"):
        labor_bc.require_critic_teacher_dataset(dagger)
    dagger_unflagged = {"collection": {"source_kind": labor_dagger.DAGGER_SOURCE_KIND}}
    with pytest.raises(ValueError, match="DAgger"):
        labor_bc.require_critic_teacher_dataset(dagger_unflagged)
    with pytest.raises(ValueError, match="aggregated"):
        labor_bc.require_critic_teacher_dataset({"collection": {"source_kind": "aggregated"}})
    # A teacher-labelled dataset smuggling search rows is refused by row_source.
    mixed = {
        "collection": {"source_kind": ltd.SOURCE_KIND},
        "row_source": torch.tensor([0, 2, 0], dtype=torch.long),
    }
    with pytest.raises(ValueError, match=r"row_source \[2\]"):
        labor_bc.require_critic_teacher_dataset(mixed)


def test_fitting_world_partition_rejects_sealed_and_development_worlds() -> None:
    assert validate_fitting_worlds([5000, 5001], "fit") == [5000, 5001]
    with pytest.raises(ValueError, match="development"):
        validate_fitting_worlds([2003], "fit")
    with pytest.raises(Exception):  # noqa: B017 - blind Exception assertion kept (test expectation unchanged)
        validate_fitting_worlds([3010], "fit")
    with pytest.raises(ValueError, match="repeats"):
        validate_fitting_worlds([5000, 5000], "fit")


def test_reset_options_default_to_movement_on() -> None:
    options = LaborResetOptions(players=32, width=192, height=96, separation=9)
    assert options.movement is True
    assert options.reset_kwargs() == {
        "players": 32,
        "movement": True,
        "withheld": 0,
        "size": (192, 96, 9),
    }
    assert LaborResetOptions.from_mapping(options.state_dict()) == options


# Task 5 (live) ----------------------------------------------------------------


def test_live_rejected_initial_rows_are_quarantined(monkeypatch, bounded_normalizer) -> None:
    real = labor_bc.execute_labor_decision
    rejected_camp: dict[str, int] = {}

    def reject_one(runtime, catalogs, initial, retry_factory):
        execution = real(runtime, catalogs, initial, retry_factory)
        camp = min(execution.initial_attempts)
        rejected_camp.setdefault("camp", camp)
        attempts = dict(execution.initial_attempts)
        attempts[camp] = dataclasses.replace(attempts[camp], valid=False)
        return dataclasses.replace(execution, initial_attempts=attempts)

    monkeypatch.setattr(labor_bc, "execute_labor_decision", reject_one)
    dataset = labor_bc.collect_heuristic_bc_dataset(
        normalizer=bounded_normalizer, world_seeds=[50000], turns=1
    )
    collection = dataset["collection"]
    assert collection["quarantined_rows"] == 1 == collection["initial_rejected"]
    assert collection["quarantine"][0]["camp"] == rejected_camp["camp"]
    quarantined = dataset["row_quarantined"]
    assert int(quarantined.sum()) == 1
    assert float(dataset["row_weight"][quarantined].sum()) == 0.0
    camp_row = int(torch.nonzero(dataset["row_camp"] == rejected_camp["camp"])[0])
    assert bool(quarantined[camp_row])
    assert collection["reset_options"]["movement"] is True
    assert set(dataset) >= {
        "row_world",
        "row_camp",
        "row_turn",
        "row_attempt",
        "label_specialty_valid",
    }


@pytest.fixture(scope="module")
def bounded_normalizer():
    from tests.test_labor_bc import _fit_test_normalizer

    return _fit_test_normalizer()


# Task 6: immutable shards, CLI and downstream pretraining ---------------------


def test_shard_roundtrip_detects_tampering(tmp_path) -> None:
    from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
    from anthropocene_rl.autonomous_v7.provenance import ProvenanceError

    arrays = {
        "row_world": np.asarray([5000, 5000]),
        "row_camp": np.asarray([1, 2]),
        "row_turn": np.asarray([3, 3]),
        "row_attempt": np.asarray([0, 0]),
        "row_quarantined": np.asarray([False, True]),
        "target_specialty": np.zeros((2, 14), np.float32),
    }
    meta = {"world_seed": 5000, "source_run": "x"}
    shard = tmp_path / "world-5000"
    record = ltd.write_shard(shard, arrays, meta)
    assert ltd.verify_shard(shard)["rows"] == 2 == record["rows"]
    with pytest.raises(ProvenanceError):
        ltd.write_shard(shard, arrays, meta)
    np.save(shard / "row_camp.npy", np.asarray([1, 1]))
    with pytest.raises(ProvenanceError):
        ltd.verify_shard(shard)


def test_labor_teacher_cli_shards_feed_bc_and_critic(tmp_path, bounded_normalizer) -> None:
    import json

    from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
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
    out = tmp_path / "teacher"
    assert (
        ltd.main(
            [
                "--normalizer-checkpoint",
                str(norm),
                "--worlds",
                "5000,5002",
                "--turns",
                "1",
                "--players",
                "8",
                "--jobs",
                "2",
                "--allow-dirty",
                "--return-tail",
                "truncated-zero",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    manifest = ltd.verify_dataset(out)
    assert set(manifest["shards"]) == {"5000", "5002"}
    assert manifest["source_record"]["options"]["reset_options"]["movement"] is True
    dataset = ltd.load_labor_dataset(out)
    weights = dataset["row_weight"]
    for world in (5000, 5002):
        assert math.isclose(
            float(weights[dataset["row_world"] == world].sum()),
            float(weights.sum()) / 2,
            rel_tol=1e-5,
        )
    with pytest.raises(Exception):  # noqa: B017 - blind Exception assertion kept (test expectation unchanged)
        ltd.main(
            [
                "--normalizer-checkpoint",
                str(norm),
                "--worlds",
                "2001",
                "--turns",
                "1",
                "--allow-dirty",
                "--out",
                str(tmp_path / "dev"),
            ]
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
    bc = tmp_path / "bc.pt"
    labor_training_main(
        [
            "pretrain-bc",
            "--checkpoint",
            str(init),
            "--dataset",
            str(out),
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--out",
            str(bc),
        ]
    )
    state = torch.load(bc, map_location="cpu", weights_only=True)
    summary = state["heuristic_bc_pretraining"]
    assert summary["collection"]["dataset_digest"] == manifest["dataset_digest"]
    assert summary["components_after"]["count_specialty"] > 0
    critic = tmp_path / "critic.pt"
    labor_training_main(
        [
            "pretrain-critic",
            "--checkpoint",
            str(bc),
            "--dataset",
            str(out),
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--out",
            str(critic),
        ]
    )
    state = torch.load(critic, map_location="cpu", weights_only=True)
    assert state["heuristic_critic_pretraining"]["dataset_digest"] == manifest["dataset_digest"]
    assert sorted(state["critic_training_world_seeds"]) == [5000, 5002]
    json.dumps(summary["collection"])
