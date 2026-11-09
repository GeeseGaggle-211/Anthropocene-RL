from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7.bridge import DEFAULT_BRIDGE, observation_batch
from anthropocene_rl.autonomous_v7.labor_bc import (
    LABOR_BC_CONTRACT,
    collect_heuristic_bc_dataset,
    convert_heuristic_labor_label,
    extract_heuristic_labor_action,
    pretrain_labor_actor_bc,
)
from anthropocene_rl.autonomous_v7.labor_bridge import (
    LABOR_ACTIVITY_FEATURE_WIDTH,
    LABOR_RESOURCE_FEATURE_WIDTH,
    LABOR_SPECIALTY_FEATURE_WIDTH,
    LaborCatalog,
    LaborModel9Bridge,
    labor_catalog_digest,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    BOUNDED_FEATURE_CONTRACT,
    LaborActor,
    LaborEncoderLayout,
    Normalizer,
)
from anthropocene_rl.autonomous_v7.labor_ppo import validate_schema12_action
from anthropocene_rl.autonomous_v7.labor_training import (
    _require_trainer_checkpoint,
)
from anthropocene_rl.autonomous_v7.labor_training import (
    main as labor_training_main,
)
from tests.test_labor_bridge import _catalog_payload


def _fit_test_normalizer() -> Normalizer:
    if not Path(os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE", str(DEFAULT_BRIDGE))).is_file():
        pytest.skip("model9 bridge is not built")
    with LaborModel9Bridge() as bridge:
        step = bridge.reset(50000)
        live = [
            v
            for v in step["villages"]
            if v.get("observation") is not None and not v.get("terminated")
        ]
        return Normalizer.fit_bounded(observation_batch(live))


@pytest.fixture(scope="module")
def bounded_normalizer_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Normalizer, Path]:
    tmp_path = tmp_path_factory.mktemp("norm")
    norm_path = tmp_path / "bounded-normalizer.pt"
    norm = _fit_test_normalizer()
    torch.save(
        {
            "feature_contract": BOUNDED_FEATURE_CONTRACT,
            "normalizer": norm.state_dict(),
        },
        norm_path,
    )
    return norm, norm_path


def test_extract_heuristic_labor_action_synthetic_domains() -> None:
    raw_cat = _catalog_payload()
    raw_cat["population"] = 100
    raw_cat["turnYears"] = 20.0
    raw_cat["digest"] = labor_catalog_digest(raw_cat)
    catalog = LaborCatalog.from_payload(raw_cat)

    # 1. Cultivation: assignedFarmerYears = 40.0, expansion = 10.0 -> Farmer (role=1, spec=2)
    # 2. Herding: plannedHerderYears = 20.0, funnels = 1.0 work (20 wy) -> Herder (role=2, spec=3)
    # 3. Fisher (role=3, spec=4): core[202] = 0.05, core[206] = 0.05
    #    -> 5 harvest, 5 dev -> 10 fishers
    # 4. Craft goods: HandTools (good 0) -> Toolmaker (role=5, spec=6): 10 wy (1 person)
    #                 Pots (good 1) -> Craftsperson (role=4, spec=5): 5 wy (1 person)
    # 5. Raw materials: Stone (good 0) -> Stonecutter (role=11, spec=12): 6 wy (1 person)
    #                   Wood (good 2) -> Woodcutter (role=10, spec=11): 4 wy (1 person)
    # Non-gatherers = 2 + 2 + 10 + 1 + 1 + 1 + 1 = 18.
    # Gatherers = 100 - 18 = 82.
    core = [0.0] * 218
    core_present = [True] * 218
    core[202] = 0.05
    core[206] = 0.05
    village_heur = {
        "camp": 7,
        "population": 100,
        "observation": {
            "core": core,
            "corePresent": core_present,
            "movementMask": [True] + [False] * 7,
            "regionMovementMask": [True] + [False] * 9,
        },
        "teacher": {
            "movement": -1,
            "fission": 1,
            "tradeValue": [2.0] * 7,
            "tradeGate": [2] * 7,
            "tradeShare": 0.8,
        },
    }

    goods = [
        {"good": "HandTools", "workerYears": 10.0, "plannedOutputCeiling": 0.0},
        {"good": "Pots", "workerYears": 5.0, "plannedOutputCeiling": 0.0},
    ] + [{"good": "none", "workerYears": 0.0, "plannedOutputCeiling": 0.0} for _ in range(9)]

    materials = [
        {"material": "Stone", "workerYears": 6.0},
        {"material": "Clay", "workerYears": 0.0},
        {"material": "Wood", "workerYears": 4.0},
    ]

    econ_diag = {
        "domains": {
            "cultivation": {
                "assignedFarmerYears": 40.0,
                "expansionWorkerYears": 10.0,
                "funnels": [],
            },
            "herding": {
                "plannedHerderYears": 20.0,
                "coveredHerderYears": 20.0,
                "assignedHerderResidents": 2.0,
                "managementHerderResidents": 1.0,
                "developmentHerderResidents": 1.0,
                "managementHerderWorkerYears": 20.0,
                "developmentHerderWorkerYears": 20.0,
                "assignedHerderWorkerYears": 40.0,
                "funnels": [
                    {
                        "metric": "herdDevelopment",
                        "executedWork": 1.0,
                        "compilerGrant": 1.0,
                    }
                ],
            },
        },
        "goods": goods,
        "materials": materials,
        "requests": [],
    }

    spec, act, action = extract_heuristic_labor_action(catalog, village_heur, econ_diag)

    # Specialty simplex assertions
    assert spec.shape == (14,)
    assert math.isclose(float(spec.sum()), 1.0, abs_tol=1e-7)
    assert spec[0] == 0.0  # idle
    assert math.isclose(spec[1], 0.83, abs_tol=1e-4)  # Gatherer (83 / 100)
    assert math.isclose(spec[2], 0.02, abs_tol=1e-4)  # Farmer (2 / 100)
    assert math.isclose(spec[3], 0.02, abs_tol=1e-4)  # Herder (2 / 100)
    assert math.isclose(spec[4], 0.10, abs_tol=1e-4)  # Fisher (10 / 100)
    assert math.isclose(spec[5], 0.00, abs_tol=1e-4)  # Craftsperson (0 / 100)
    assert math.isclose(spec[6], 0.01, abs_tol=1e-4)  # Toolmaker (1 / 100)
    assert math.isclose(spec[11], 0.01, abs_tol=1e-4)  # Woodcutter (1 / 100)
    assert math.isclose(spec[12], 0.01, abs_tol=1e-4)  # Stonecutter (1 / 100)

    # Activity simplex assertions
    assert act.shape == (13, 22)
    for r in range(13):
        assert math.isclose(float(act[r].sum()), 1.0, abs_tol=1e-7)

    # Farmer activities: act 5 (field.op) = 30 wy, act 6 (field.exp) = 10 wy -> 0.75 / 0.25
    assert math.isclose(act[1, 5], 0.75, abs_tol=1e-4)
    assert math.isclose(act[1, 6], 0.25, abs_tol=1e-4)

    # Herder activities: act 7 (herd.mgmt) = 20 wy, act 8 (herd.expansion) = 20 wy -> 0.5 / 0.5
    assert math.isclose(act[2, 7], 0.50, abs_tol=1e-4)
    assert math.isclose(act[2, 8], 0.50, abs_tol=1e-4)

    # Fisher activities: act 9 (harvest) = 5 people, act 10 (dev) = 5 people -> 0.5 / 0.5
    assert math.isclose(act[3, 9], 0.50, abs_tol=1e-4)
    assert math.isclose(act[3, 10], 0.50, abs_tol=1e-4)

    # Toolmaker: one worker over 20 years with 10 and 5 worker-years keeps its
    # unused time as role idle (0.50 / 0.25 / idle 0.25), not 2/3 and 1/3.
    assert math.isclose(act[5, 11], 0.50, abs_tol=1e-9)
    assert math.isclose(act[5, 12], 0.25, abs_tol=1e-9)
    assert math.isclose(act[5, 0], 0.25, abs_tol=1e-9)

    # Woodcutter: 4 of 20 worker-years on stock.wood, the rest idle
    assert math.isclose(act[10, 4], 0.20, abs_tol=1e-9)
    assert math.isclose(act[10, 0], 0.80, abs_tol=1e-9)

    # Stonecutter: 6 of 20 worker-years on stock.stone
    assert math.isclose(act[11, 2], 0.30, abs_tol=1e-9)
    assert math.isclose(act[11, 0], 0.70, abs_tol=1e-9)

    # Unstaffed roles emit canonical role idle (masked out of the activity loss)
    assert act[4, 0] == 1.0  # Craftsperson has no staff

    label = convert_heuristic_labor_label(catalog, village_heur, econ_diag)
    assert label.specialty_valid and not label.over_budget, label.reasons
    assert label.specialty_counts.sum() == 100
    assert bool(label.activity_role_valid[5]) and not bool(label.activity_role_valid[4])

    # Schema-12 action checks
    assert action["schemaVersion"] == 12
    assert action["movement"] == -1
    assert action["fission"] == 1
    assert action["tradeValue"] == [2.0] * 7
    assert action["tradeGate"] == [2] * 7
    assert math.isclose(action["tradeShare"], 0.8, abs_tol=1e-5)

    # Wire validation must pass cleanly
    decoded = validate_schema12_action(action, catalog, retry=False)
    assert decoded.specialty_allocation == tuple(spec)


def test_extract_heuristic_labor_action_scaled_population_budget() -> None:
    raw_cat = _catalog_payload()
    raw_cat["population"] = 10  # small population
    raw_cat["turnYears"] = 20.0
    raw_cat["digest"] = labor_catalog_digest(raw_cat)
    catalog = LaborCatalog.from_payload(raw_cat)

    # Demands 40 farmers (2p) + 40 herders (2p) + 200 fishers (10p) = 14 people > P=10
    core = [0.0] * 218
    core_present = [True] * 218
    core[202] = 1.0  # 10 fishers
    village_heur = {
        "camp": 7,
        "population": 10,
        "observation": {"core": core, "corePresent": core_present},
        "teacher": {"movement": -1, "fission": 0},
    }
    econ_diag = {
        "domains": {
            "cultivation": {"assignedFarmerYears": 40.0, "expansionWorkerYears": 0.0},
            "herding": {
                "plannedHerderYears": 40.0,
                "assignedHerderResidents": 2.0,
                "managementHerderWorkerYears": 40.0,
                "developmentHerderWorkerYears": 0.0,
                "funnels": [],
            },
        },
    }

    label = convert_heuristic_labor_label(catalog, village_heur, econ_diag)
    # Over-budget staffing is flagged, never rescaled into new Gatherers.
    assert label.over_budget
    assert not label.specialty_valid
    assert not label.activity_role_valid.any()
    assert "over-budget-staffing" in label.reasons
    # The drive action stays executable for advancing the world only.
    decoded = validate_schema12_action(label.action, catalog, retry=False)
    assert len(decoded.specialty_allocation) == 14


def test_pretrain_labor_actor_bc_synthetic_optimization() -> None:
    B = 32
    core_dim = 15
    layout = LaborEncoderLayout(
        core_dim,
        LABOR_SPECIALTY_FEATURE_WIDTH,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
    )
    actor = LaborActor(layout)

    # Build synthetic dataset
    spec_mask = torch.ones(B, 14, dtype=torch.bool)
    act_mask = torch.ones(B, 13, 22, dtype=torch.bool)

    target_specialty = torch.zeros(B, 14, dtype=torch.float32)
    target_specialty[:, 1] = 0.8
    target_specialty[:, 2] = 0.2

    target_activity = torch.zeros(B, 13, 22, dtype=torch.float32)
    target_activity[:, 0, 1] = 1.0
    target_activity[:, 1, 5] = 1.0
    target_activity[:, 2:, 0] = 1.0

    movement_mask = torch.ones(B, 8, dtype=torch.bool)
    region_mask = torch.ones(B, 10, dtype=torch.bool)

    dataset: dict[str, torch.Tensor | dict] = {
        "core": torch.randn(B, core_dim),
        "labor_personality": torch.full((B, 2), 0.5),
        "movement_mask": movement_mask,
        "region_movement_mask": region_mask,
        "labor_specialty_features": torch.randn(B, 14, LABOR_SPECIALTY_FEATURE_WIDTH),
        "labor_specialty_present": torch.ones(B, 14, dtype=torch.bool),
        "labor_specialty_mask": spec_mask,
        "labor_activity_features": torch.randn(B, 13 * 22, LABOR_ACTIVITY_FEATURE_WIDTH),
        "labor_activity_present": torch.ones(B, 13 * 22, dtype=torch.bool),
        "labor_activity_mask": act_mask,
        "labor_resource_features": torch.randn(B, 20, LABOR_RESOURCE_FEATURE_WIDTH),
        "labor_resource_present": torch.ones(B, 20, dtype=torch.bool),
        "target_specialty": target_specialty,
        "target_activity": target_activity,
        "target_movement_code": torch.full((B,), -1, dtype=torch.long),
        "target_fission": torch.zeros(B, dtype=torch.long),
        "target_trade_value_latent": torch.zeros(B, 7, dtype=torch.float32),
        "target_trade_gate": torch.ones(B, 7, dtype=torch.long),
        "target_trade_share_latent": torch.zeros(B, dtype=torch.float32),
        "collection": {"total_decisions": B, "initial_accepted": B},
    }

    # Add legacy token sets required by LaborObservationEncoder
    for name, rows, columns in contracts.SETS:
        dataset[name] = torch.randn(B, rows, columns)
        dataset[name + "_present"] = torch.ones(B, rows, dtype=torch.bool)

    summary = pretrain_labor_actor_bc(
        actor=actor,
        dataset=dataset,
        epochs=12,
        batch_size=16,
        lr=5e-3,
        target_log_std=-2.0,
    )

    assert summary["contract"] == LABOR_BC_CONTRACT
    assert summary["rows"] == B
    assert summary["epochs"] == 12
    assert summary["final_loss"] < summary["initial_loss"]
    assert summary["specialty_l1_after"] < summary["specialty_l1_before"]
    assert summary["activity_l1_after"] <= summary["activity_l1_before"]

    # Verify log_std regularization drove outputs toward target_log_std
    with torch.no_grad():
        out = actor(dataset)
        spec_std = out["specialty_log_std"].mean().item()
        assert -3.0 < spec_std < -1.0


def test_live_collect_heuristic_bc_dataset_and_pretrain(
    bounded_normalizer_fixture: tuple[Normalizer, Path],
) -> None:
    normalizer, _ = bounded_normalizer_fixture
    dataset = collect_heuristic_bc_dataset(
        normalizer=normalizer,
        world_seeds=[50000],
        turns=2,
    )

    assert dataset["target_specialty"].shape[0] > 0
    collection = dataset["collection"]
    assert collection["total_decisions"] > 0
    assert collection["initial_accepted"] == collection["total_decisions"]
    assert collection["initial_rejected"] == 0

    torch.manual_seed(42)
    layout = LaborEncoderLayout(
        int(dataset["core"].shape[1]),
        LABOR_SPECIALTY_FEATURE_WIDTH,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
    )
    actor = LaborActor(layout)
    summary = pretrain_labor_actor_bc(
        actor=actor,
        dataset=dataset,
        epochs=5,
        batch_size=16,
        lr=2e-3,
    )

    assert summary["contract"] == LABOR_BC_CONTRACT
    assert summary["final_loss"] < summary["initial_loss"]
    assert summary["specialty_l1_after"] < summary["specialty_l1_before"]


def test_pretrain_bc_cli_end_to_end(
    tmp_path: Path,
    bounded_normalizer_fixture: tuple[Normalizer, Path],
) -> None:
    _, normalizer_path = bounded_normalizer_fixture
    init_out = tmp_path / "checkpoint-000.pt"
    bc_out = tmp_path / "checkpoint-bc.pt"
    rollout_out = tmp_path / "rollout-000.pt"

    # 1. Initialize trainer checkpoint
    labor_training_main(
        [
            "initialize",
            "--normalizer-checkpoint",
            str(normalizer_path),
            "--world-seed",
            "50000",
            "--optimizer-seed",
            "42",
            "--out",
            str(init_out),
        ]
    )
    assert init_out.exists()

    # 2. Run pretrain-bc CLI
    labor_training_main(
        [
            "pretrain-bc",
            "--checkpoint",
            str(init_out),
            "--worlds",
            "50000",
            "--turns",
            "2",
            "--epochs",
            "5",
            "--batch-size",
            "16",
            "--out",
            str(bc_out),
        ]
    )
    assert bc_out.exists()

    # 3. Verify trainer checkpoint requirements & metadata
    state = torch.load(bc_out, map_location="cpu", weights_only=True)
    _require_trainer_checkpoint(state)
    assert "heuristic_bc_pretraining" in state
    summary = state["heuristic_bc_pretraining"]
    assert summary["contract"] == LABOR_BC_CONTRACT
    assert summary["final_loss"] < summary["initial_loss"]
    assert state["critic_training_world_seeds"] == []
    assert state["critic_training_independent_unit_ids"] == []

    # 4. Verify collect command runs seamlessly on the pre-trained checkpoint
    labor_training_main(
        [
            "collect",
            "--checkpoint",
            str(bc_out),
            "--worlds",
            "50000",
            "--turns",
            "1",
            "--out",
            str(rollout_out),
        ]
    )
    assert rollout_out.exists()


def test_initialize_with_pretrain_bc_turns_cli(
    tmp_path: Path,
    bounded_normalizer_fixture: tuple[Normalizer, Path],
) -> None:
    _, normalizer_path = bounded_normalizer_fixture
    init_bc_out = tmp_path / "checkpoint-init-bc.pt"

    labor_training_main(
        [
            "initialize",
            "--normalizer-checkpoint",
            str(normalizer_path),
            "--world-seed",
            "50000",
            "--optimizer-seed",
            "42",
            "--pretrain-bc-turns",
            "2",
            "--pretrain-bc-epochs",
            "3",
            "--out",
            str(init_bc_out),
        ]
    )
    assert init_bc_out.exists()

    state = torch.load(init_bc_out, map_location="cpu", weights_only=True)
    _require_trainer_checkpoint(state)
    assert "heuristic_bc_pretraining" in state
    assert state["heuristic_bc_pretraining"]["contract"] == LABOR_BC_CONTRACT


def test_extract_heuristic_action_registry_driven_fallback() -> None:
    raw_cat = _catalog_payload()
    raw_cat["population"] = 50
    raw_cat["turnYears"] = 20.0
    raw_cat["digest"] = labor_catalog_digest(raw_cat)
    catalog = LaborCatalog.from_payload(raw_cat)

    core = [0.0] * 218
    core_present = [True] * 218
    village_heur = {
        "camp": 7,
        "population": 50,
        "observation": {
            "core": core,
            "corePresent": core_present,
            "movementMask": [True] + [False] * 7,
            "regionMovementMask": [True] + [False] * 9,
        },
        "teacher": {
            "movement": -1,
            "fission": 0,
            "tradeValue": [1.0] * 7,
            "tradeGate": [1] * 7,
            "tradeShare": 0.5,
        },
    }

    # Completely empty diagnostics: missing work is unavailable, never an
    # all-Gatherer demonstration.
    label = convert_heuristic_labor_label(catalog, village_heur, {})
    assert not label.specialty_valid
    assert "economic-diagnostics-unavailable" in label.reasons
    assert "herder-staffing-unavailable" in label.reasons
    assert not label.activity_role_valid.any()
    # Nonlabor heads remain separately valid.
    assert label.movement_valid and label.fission_valid and label.trade_gate_valid.all()

    # Roles without staff emit canonical idle, not fabricated operation defaults.
    for r in (1, 2, 3):
        assert label.activity[r, 0] == 1.0

    # The drive action still validates.
    validate_schema12_action(label.action, catalog, retry=False)


def test_pretrain_labor_actor_bc_inactive_role_weight_and_inverse_freq() -> None:
    B = 32
    core_dim = 15
    layout = LaborEncoderLayout(
        core_dim,
        LABOR_SPECIALTY_FEATURE_WIDTH,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
    )
    actor = LaborActor(layout)

    spec_mask = torch.ones(B, 14, dtype=torch.bool)
    act_mask = torch.ones(B, 13, 22, dtype=torch.bool)

    target_specialty = torch.zeros(B, 14, dtype=torch.float32)
    target_specialty[:, 1] = 0.9
    target_specialty[:, 2] = 0.1

    target_activity = torch.zeros(B, 13, 22, dtype=torch.float32)
    target_activity[:, 0, 1] = 1.0
    target_activity[:, 1, 5] = 1.0
    target_activity[:, 2:, 0] = 1.0

    dataset: dict[str, torch.Tensor | dict] = {
        "core": torch.randn(B, core_dim),
        "labor_personality": torch.full((B, 2), 0.5),
        "movement_mask": torch.ones(B, 8, dtype=torch.bool),
        "region_movement_mask": torch.ones(B, 10, dtype=torch.bool),
        "labor_specialty_features": torch.randn(B, 14, LABOR_SPECIALTY_FEATURE_WIDTH),
        "labor_specialty_present": torch.ones(B, 14, dtype=torch.bool),
        "labor_specialty_mask": spec_mask,
        "labor_activity_features": torch.randn(B, 13 * 22, LABOR_ACTIVITY_FEATURE_WIDTH),
        "labor_activity_present": torch.ones(B, 13 * 22, dtype=torch.bool),
        "labor_activity_mask": act_mask,
        "labor_resource_features": torch.randn(B, 20, LABOR_RESOURCE_FEATURE_WIDTH),
        "labor_resource_present": torch.ones(B, 20, dtype=torch.bool),
        "target_specialty": target_specialty,
        "target_activity": target_activity,
        "target_movement_code": torch.full((B,), -1, dtype=torch.long),
        "target_fission": torch.zeros(B, dtype=torch.long),
        "target_trade_value_latent": torch.zeros(B, 7, dtype=torch.float32),
        "target_trade_gate": torch.ones(B, 7, dtype=torch.long),
        "target_trade_share_latent": torch.zeros(B, dtype=torch.float32),
        "collection": {"total_decisions": B, "initial_accepted": B},
    }
    for name, rows, columns in contracts.SETS:
        dataset[name] = torch.randn(B, rows, columns)
        dataset[name + "_present"] = torch.ones(B, rows, dtype=torch.bool)

    summary = pretrain_labor_actor_bc(
        actor=actor,
        dataset=dataset,
        epochs=8,
        batch_size=16,
        lr=5e-3,
        inactive_role_weight=0.0,
        label_smoothing=1e-6,
        inverse_freq_power=0.5,
    )

    assert summary["contract"] == LABOR_BC_CONTRACT
    assert summary["inactive_role_weight"] == 0.0
    assert summary["label_smoothing"] == 1e-6
    assert summary["inverse_freq_power"] == 0.5
    assert summary["final_loss"] < summary["initial_loss"]


def test_allow_vm_scale_boundary_in_bc(
    bounded_normalizer_fixture: tuple[Normalizer, Path],
) -> None:
    normalizer, _ = bounded_normalizer_fixture
    # 4 world seeds exceeds local limit of 3
    with pytest.raises(ValueError, match=r"1\.\.3 world seeds"):
        collect_heuristic_bc_dataset(
            normalizer=normalizer,
            world_seeds=[50000, 50001, 50002, 50003],
            turns=1,
            allow_vm_scale=False,
        )
