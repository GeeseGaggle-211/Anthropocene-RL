from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7.labor_model import (
    BOUNDED_FEATURE_CONTRACT,
    LaborPersonality,
)
from anthropocene_rl.autonomous_v7.labor_training import (
    LABOR_ROLLOUT_KIND,
    LaborRolloutRow,
    _critic_td_losses,
    _load_rollout,
    _parse_worlds,
    _sample_world_step,
    _state_dict_rollout,
    _transition_rewards,
    _validate_training_data_separation,
    argument_parser,
    build_labor_training_batch,
)
from anthropocene_rl.autonomous_v7.labor_training import (
    main as labor_training_main,
)


def _tensors(rows: int) -> dict[str, torch.Tensor]:
    t = {
        "core": torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
        "mask": torch.ones(rows, 2, dtype=torch.bool),
        "labor_personality": torch.tensor([[1.0, 0.0]], dtype=torch.float32).repeat(rows, 1),
    }
    from anthropocene_rl.autonomous_v7 import contracts
    from anthropocene_rl.autonomous_v7.labor_bridge import (
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
        LABOR_SPECIALTY_FEATURE_WIDTH,
    )
    for name, set_rows, width in contracts.SETS:
        t[name] = torch.zeros(rows, set_rows, width)
        t[name + "_present"] = torch.zeros(rows, set_rows, dtype=torch.bool)
    t["labor_specialty_features"] = torch.randn(rows, 2, LABOR_SPECIALTY_FEATURE_WIDTH)
    t["labor_specialty_present"] = torch.ones(rows, 2, dtype=torch.bool)
    t["labor_activity_features"] = torch.randn(rows, 2, LABOR_ACTIVITY_FEATURE_WIDTH)
    t["labor_activity_present"] = torch.ones(rows, 2, dtype=torch.bool)
    t["labor_resource_features"] = torch.randn(rows, 20, LABOR_RESOURCE_FEATURE_WIDTH)
    t["labor_resource_present"] = torch.ones(rows, 20, dtype=torch.bool)
    return t


def _labor_sample(rows: int) -> dict[str, torch.Tensor]:
    return {
        "specialty_allocation": torch.tensor([[1.0, 0.0]]).repeat(rows, 1),
        "activity_allocation": torch.tensor([[[1.0, 0.0]]]).repeat(rows, 1, 1),
        "specialty_latent": torch.zeros(rows, 2),
        "activity_latent": torch.zeros(rows, 1, 2),
        "specialty_mask": torch.ones(rows, 2, dtype=torch.bool),
        "activity_mask": torch.ones(rows, 1, 2, dtype=torch.bool),
    }


def _nonlabor_sample(rows: int) -> dict[str, torch.Tensor]:
    return {
        "movement_index": torch.zeros(rows, dtype=torch.long),
        "movement": torch.full((rows,), -1, dtype=torch.long),
        "fission": torch.zeros(rows, dtype=torch.long),
        "trade_value_latent": torch.zeros(rows, 7),
        "trade_gate": torch.zeros(rows, 7, dtype=torch.long),
        "trade_share_latent": torch.zeros(rows),
        "movement_mask": torch.ones(rows, 1, dtype=torch.bool),
        "movement_codes": torch.full((rows, 1), -1, dtype=torch.long),
    }


def _row(
    turn: int,
    *,
    retry: bool,
    terminated: bool = False,
    physical_qol: float = 2.0,
    physical_population: float = 20.0,
) -> LaborRolloutRow:
    return LaborRolloutRow(
        world_seed=50000,
        camp_id=7,
        turn=turn,
        years=20.0,
        terminated=terminated,
        initial_tensors=_tensors(1),
        initial_labor_sample=_labor_sample(1),
        initial_nonlabor_sample=_nonlabor_sample(1),
        behavior_initial_labor_logp=-1.0,
        behavior_initial_nonlabor_logp=-2.0,
        initial_task_values={"qol_years": 10.0, "population_years": 100.0},
        initial_cost_values={"feasibility": 1.0, "settlement_guard": 0.25},
        retry_tensors=_tensors(1) if retry else None,
        retry_labor_sample=_labor_sample(1) if retry else None,
        behavior_retry_labor_logp=-3.0 if retry else None,
        retry_task_values={"qol_years": 11.0, "population_years": 110.0} if retry else None,
        retry_cost_values={"feasibility": 0.5, "settlement_guard": 0.5} if retry else None,
        physical_task_rewards={
            "qol_years": physical_qol,
            "population_years": physical_population,
        },
        feasibility_initial=3.0 if retry else 0.0,
        feasibility_retry=0.0,
        settlement_guard=1.0,
    )


def test_training_batch_scores_nonlabor_once_and_physical_reward_once() -> None:
    direct = _row(9, retry=False)
    retried = _row(10, retry=True, terminated=True, physical_qol=4.0, physical_population=40.0)
    batch, task_targets, cost_targets = build_labor_training_batch(
        [direct, retried],
        {
            (50000, 7): {
                "qol_years": 0.0,
                "population_years": 0.0,
                "feasibility": 0.0,
                "settlement_guard": 0.0,
            }
        },
        personality=LaborPersonality(),
        cost_multipliers={"feasibility": 1.0, "settlement_guard": 1.0},
        behavior_policy_sha256="a" * 64,
        critic_sha256="b" * 64,
        normalization_sha256="c" * 64,
        lam=0.0,
    )
    gamma = torch.exp(torch.tensor(-20.0 / 400.0))
    # The retry target is the one physical reward. The initial target carries
    # zero physical reward and bootstraps through the retry value exactly once.
    assert batch.retry_parent.tolist() == [1]
    assert task_targets["initial"]["qol_years"][1] == pytest.approx(4.0)
    assert task_targets["retry"]["qol_years"][0] == pytest.approx(4.0)
    expected_retry_task = 0.5 * (4.0 - 11.0) / 43.152 + 0.5 * (40.0 - 110.0) / 85133.98
    # Feasibility advantage is -0.5 and guard advantage is +0.5; both enter
    # with a negative multiplier, so they cancel in this fixture.
    assert batch.retry_advantage[0] == pytest.approx(expected_retry_task)
    assert cost_targets["initial"]["feasibility"][1] == pytest.approx(3.0)
    assert cost_targets["retry"]["feasibility"][0] == pytest.approx(0.0)
    # The previous direct edge links to the next physical decision, while a
    # terminal retry has no outer bootstrap.
    assert task_targets["initial"]["qol_years"][0] == pytest.approx(
        2.0 + float(gamma) * 10.0
    )
    assert batch.behaviour_initial_nonlabor_logp.tolist() == pytest.approx([-2.0, -2.0])
    assert len(batch.behaviour_retry_labor_logp) == 1


def test_living_sequence_cannot_silently_bootstrap_from_zero() -> None:
    with pytest.raises(ValueError, match="complete four-head successor bootstrap"):
        build_labor_training_batch(
            [_row(9, retry=False, terminated=False)],
            {},
            personality=LaborPersonality(),
            cost_multipliers={"feasibility": 1.0, "settlement_guard": 1.0},
            behavior_policy_sha256="a" * 64,
            critic_sha256="b" * 64,
            normalization_sha256="c" * 64,
        )


def test_complete_extinction_transition_keeps_the_acted_terminal_row() -> None:
    rewards, terminal, years = _transition_rewards(
        {
            "turn": 10,
            "villages": [
                {
                    "camp": 7,
                    "society": 70,
                    "parentSociety": 0,
                    "priorPopulation": 12,
                    "population": 0,
                    "qol": 0.0,
                    "years": 20.0,
                    "welfare": 0.0,
                    "terminated": True,
                    "tradeUtility": 0.0,
                }
            ],
        },
        [7],
        {7: 9},
        world_index=0,
        decision_turn=9,
    )
    assert terminal == {7: True}
    assert years == {7: 20.0}
    assert rewards[7] == {"qol_years": 0.0, "population_years": 0.0}


def test_empty_retry_surface_is_typed_and_has_no_nonlabor_retry_rows() -> None:
    batch, _, _ = build_labor_training_batch(
        [_row(9, retry=False, terminated=True)],
        {},
        personality=LaborPersonality(),
        cost_multipliers={"feasibility": 0.1, "settlement_guard": 0.2},
        behavior_policy_sha256="a" * 64,
        critic_sha256="b" * 64,
        normalization_sha256="c" * 64,
    )
    assert batch.retry_parent.numel() == 0
    assert batch.retry_tensors["core"].shape == (0, 3)
    assert batch.retry_labor_sample["activity_allocation"].shape == (0, 1, 2)
    assert not hasattr(batch, "retry_nonlabor_sample")


def test_rollout_roundtrip_uses_weights_only_payload(tmp_path: Path) -> None:
    batch, task, costs = build_labor_training_batch(
        [_row(9, retry=True, terminated=True)],
        {},
        personality=LaborPersonality(),
        cost_multipliers={"feasibility": 1.0, "settlement_guard": 1.0},
        behavior_policy_sha256="a" * 64,
        critic_sha256="b" * 64,
        normalization_sha256="c" * 64,
    )
    path = tmp_path / "rollout.pt"
    torch.save(
        _state_dict_rollout(
            batch,
            task,
            costs,
            {"sourceCheckpointSha256": "d" * 64, "scope": "diagnostic"},
        ),
        path,
    )
    loaded, loaded_task, loaded_costs, metadata = _load_rollout(path)
    assert loaded.behavior_policy_digest == "a" * 64
    assert torch.equal(loaded.initial_advantage, batch.initial_advantage)
    assert torch.equal(loaded_task["initial"]["qol_years"], task["initial"]["qol_years"])
    assert torch.equal(
        loaded_costs["retry"]["feasibility"], costs["retry"]["feasibility"]
    )
    assert metadata["scope"] == "diagnostic"
    assert torch.load(path, weights_only=True)["kind"] == LABOR_ROLLOUT_KIND


def test_train_cli_requires_independent_anchor() -> None:
    parser = argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "--checkpoint",
                "checkpoint.pt",
                "--rollout",
                "rollout.pt",
                "--out",
                "next.pt",
            ]
        )
    parsed = parser.parse_args(
        [
            "train",
            "--checkpoint",
            "checkpoint.pt",
            "--rollout",
            "rollout.pt",
            "--anchor",
            "anchor.npz",
            "--out",
            "next.pt",
        ]
    )
    assert parsed.calibration_report is None


def test_retry_critic_losses_use_physical_parent_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anthropocene_rl.autonomous_v7 import labor_training

    batch, task_targets, cost_targets = build_labor_training_batch(
        [_row(9, retry=False), _row(10, retry=True, terminated=True)],
        {},
        personality=LaborPersonality(),
        cost_multipliers={"feasibility": 1.0, "settlement_guard": 1.0},
        behavior_policy_sha256="a" * 64,
        critic_sha256="b" * 64,
        normalization_sha256="c" * 64,
    )

    def task_loss(_critic, tensors, _targets):
        return torch.tensor(4.0 if len(tensors["core"]) == 2 else 10.0)

    def cost_loss(_critic, tensors, _targets):
        return torch.tensor(2.0 if len(tensors["core"]) == 2 else 6.0)

    monkeypatch.setattr(labor_training, "labor_task_value_loss", task_loss)
    monkeypatch.setattr(labor_training, "labor_cost_value_loss", cost_loss)
    task, cost = _critic_td_losses(
        object(), object(), batch, task_targets, cost_targets
    )
    assert float(task) == pytest.approx(9.0)
    assert float(cost) == pytest.approx(5.0)


def test_training_data_separation_rejects_off_policy_and_gate_overlap() -> None:
    batch, _, _ = build_labor_training_batch(
        [_row(9, retry=False, terminated=True)],
        {},
        personality=LaborPersonality(),
        cost_multipliers={"feasibility": 1.0, "settlement_guard": 1.0},
        behavior_policy_sha256="a" * 64,
        critic_sha256="b" * 64,
        normalization_sha256="c" * 64,
    )
    state = {
        "critic_training_world_seeds": [49999],
        "critic_training_independent_unit_ids": ["world-49999"],
    }
    off_policy = SimpleNamespace(
        metadata={
            "source": {"sourcePolicyDigest": "d" * 64, "worldSeeds": [50001]}
        },
        independent_unit_ids=np.asarray(["world-50001"]),
    )
    with pytest.raises(ValueError, match="off-policy"):
        _validate_training_data_separation(
            off_policy, batch, {"worldSeeds": [50000]}, state, None
        )

    anchor = SimpleNamespace(
        metadata={
            "source": {"sourcePolicyDigest": "a" * 64, "worldSeeds": [50001]}
        },
        independent_unit_ids=np.asarray(["world-50001"]),
    )
    def gate_report(worlds: list[int]) -> dict[str, object]:
        return {
            "kind": "model9-labor-value-calibration-report-v1",
            "dataRole": "actor-gate",
            "localReadinessClaim": False,
            "actorGateOpen": False,
            "enabledHeads": [
                "qol_years",
                "population_years",
                "feasibility",
                "settlement_guard",
            ],
            "heads": {
                name: {"pass": False}
                for name in (
                    "qol_years",
                    "population_years",
                    "feasibility",
                    "settlement_guard",
                )
            },
            "sourcePolicyDigest": "a" * 64,
            "sourceCriticDigest": "b" * 64,
            "normalizationBindingDigest": "c" * 64,
            "dataSha256": "d" * 64,
            "worldSeeds": worlds,
            "independentUnitIds": [f"world-{world}" for world in worlds],
        }

    overlap = gate_report([50000, 50002])
    with pytest.raises(ValueError, match="overlap"):
        _validate_training_data_separation(
            anchor, batch, {"worldSeeds": [50000]}, state, overlap
        )

    worlds, units = _validate_training_data_separation(
        anchor,
        batch,
        {"worldSeeds": [50000]},
        state,
        gate_report([50002]),
    )
    assert worlds == {49999, 50000, 50001}
    assert units == {"world-49999", "world-50001"}


def test_argument_parser_critic_epochs_and_vm_scale() -> None:
    parser = argument_parser()
    train_args = parser.parse_args(
        [
            "train",
            "--checkpoint",
            "chk.pt",
            "--rollout",
            "roll.pt",
            "--anchor",
            "anc.npz",
            "--critic-epochs",
            "5",
            "--out",
            "out.pt",
        ]
    )
    assert train_args.critic_epochs == 5

    collect_args = parser.parse_args(
        [
            "collect",
            "--checkpoint",
            "chk.pt",
            "--worlds",
            "50000",
            "--max-self-destruct-rewinds",
            "2",
            "--allow-vm-scale",
            "--out",
            "roll.pt",
        ]
    )
    assert collect_args.max_self_destruct_rewinds == 2
    assert collect_args.allow_vm_scale is True

    fit_norm_args = parser.parse_args(
        [
            "fit-normalizer",
            "--worlds",
            "50000,50001",
            "--turns",
            "10",
            "--allow-vm-scale",
            "--out",
            "norm.pt",
        ]
    )
    assert fit_norm_args.worlds == "50000,50001"
    assert fit_norm_args.turns == 10
    assert fit_norm_args.allow_vm_scale is True
    assert fit_norm_args.out == Path("norm.pt")


def test_parse_worlds_allow_vm_scale() -> None:
    # 4 world seeds: without allow_vm_scale must fail
    with pytest.raises(ValueError, match=r"1\.\.3 seeds"):
        _parse_worlds("50000,50001,50002,50003", allow_vm_scale=False)

    # with allow_vm_scale must succeed
    parsed = _parse_worlds("50000,50001,50002,50003", allow_vm_scale=True)
    assert parsed == [50000, 50001, 50002, 50003]


def test_multi_epoch_critic_loss_reduction() -> None:
    from anthropocene_rl.autonomous_v7.labor_bridge import (
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
        LABOR_SPECIALTY_FEATURE_WIDTH,
    )
    from anthropocene_rl.autonomous_v7.labor_model import (
        LaborCostCritics,
        LaborEncoderLayout,
        LaborTaskCritic,
    )
    from anthropocene_rl.autonomous_v7.labor_value_calibration import labor_anchor_loss

    core_dim = 3
    layout = LaborEncoderLayout(
        core_dim,
        LABOR_SPECIALTY_FEATURE_WIDTH,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
    )
    task = LaborTaskCritic(layout)
    costs = LaborCostCritics(layout)
    optimizer = torch.optim.Adam(list(task.parameters()) + list(costs.parameters()), lr=1e-2)

    batch = build_labor_training_batch(
        [_row(9, retry=False, terminated=True)],
        {},
        personality=LaborPersonality(),
        cost_multipliers={"feasibility": 1.0, "settlement_guard": 1.0},
        behavior_policy_sha256="a" * 64,
        critic_sha256="b" * 64,
        normalization_sha256="c" * 64,
    )[0]
    task_targets = {
        "initial": {
            "qol_years": torch.tensor([5.0]),
            "population_years": torch.tensor([50.0]),
        }
    }
    cost_targets = {
        "initial": {
            "feasibility": torch.tensor([1.0]),
            "settlement_guard": torch.tensor([0.0]),
        }
    }

    class SyntheticAnchor:
        def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
            self.tensors = {
                k: v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                for k, v in tensors.items()
            }
            self.row_ids = np.array(["r0"])
            self.independent_unit_ids = np.array(["unit-0"])
            self.targets_raw = {
                "qol_years": np.array([5.0]),
                "population_years": np.array([50.0]),
                "feasibility": np.array([1.0]),
                "settlement_guard": np.array([0.0]),
            }
            self.target_masks = {
                "qol_years": np.array([True]),
                "population_years": np.array([True]),
                "feasibility": np.array([True]),
                "settlement_guard": np.array([True]),
            }

        def __len__(self) -> int:
            return 1

    anchor = SyntheticAnchor(batch.initial_tensors)
    anchor_rows = np.arange(1, dtype=np.int64)

    initial_loss = None
    final_loss = None
    for _epoch in range(5):
        optimizer.zero_grad()
        td_task, td_cost = _critic_td_losses(task, costs, batch, task_targets, cost_targets)
        indep, _ = labor_anchor_loss(task, costs, anchor, anchor_rows)
        loss = td_task + td_cost + indep
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    assert final_loss < initial_loss


def test_self_destruct_rewind_synthetic() -> None:
    # Verify that rewound self-destruct rows get synthetic negative camp_id and terminated=True,
    # and combine cleanly in build_labor_training_batch
    failed_turn = _row(9, retry=False, terminated=False, physical_qol=0.0)
    failed_turn.settlement_guard = 1.0  # self-destruct
    # Rewound row
    synth_failed = LaborRolloutRow(
        world_seed=50000,
        camp_id=-(7 * 1_000_000 + 9 * 100 + 0 + 1),
        turn=9,
        years=20.0,
        terminated=True,
        initial_tensors=failed_turn.initial_tensors,
        initial_labor_sample=failed_turn.initial_labor_sample,
        initial_nonlabor_sample=failed_turn.initial_nonlabor_sample,
        behavior_initial_labor_logp=failed_turn.behavior_initial_labor_logp,
        behavior_initial_nonlabor_logp=failed_turn.behavior_initial_nonlabor_logp,
        initial_task_values=failed_turn.initial_task_values,
        initial_cost_values=failed_turn.initial_cost_values,
        retry_tensors=None,
        retry_labor_sample=None,
        behavior_retry_labor_logp=None,
        retry_task_values=None,
        retry_cost_values=None,
        physical_task_rewards={"qol_years": 0.0, "population_years": 0.0},
        feasibility_initial=0.0,
        feasibility_retry=0.0,
        settlement_guard=1.0,
    )
    # Surviving replacement turn
    surviving_turn = _row(9, retry=False, terminated=True, physical_qol=3.0)

    # build_labor_training_batch processes both sequences cleanly
    batch, task_targets, cost_targets = build_labor_training_batch(
        [synth_failed, surviving_turn],
        {},
        personality=LaborPersonality(),
        cost_multipliers={"feasibility": 1.0, "settlement_guard": 1.0},
        behavior_policy_sha256="a" * 64,
        critic_sha256="b" * 64,
        normalization_sha256="c" * 64,
    )
    assert len(batch.initial_advantage) == 2
    assert cost_targets["initial"]["settlement_guard"][0] == pytest.approx(1.0)
    assert task_targets["initial"]["qol_years"][1] == pytest.approx(3.0)


def test_selective_self_destruct_rewind_multivillage() -> None:

    mock_bridge = MagicMock()
    mock_fork = 123
    mock_bridge.fork.return_value = (mock_fork, None)
    mock_bridge.restore.return_value = {"turn": 5, "villages": []}

    # Step 1: Camp 1 self-destructs (settlement_guard = 1.0, terminal = True),
    # while Camp 2 is healthy (settlement_guard = 0.0, terminal = False).
    row_c1_failed = _row(5, retry=False, terminated=False)
    row_c1_failed.camp_id = 1
    row_c1_failed.settlement_guard = 1.0

    row_c2_healthy = _row(5, retry=False, terminated=False)
    row_c2_healthy.camp_id = 2
    row_c2_healthy.settlement_guard = 0.0

    # Step 2 (after rewind): Both survive cleanly
    row_c1_retry = _row(5, retry=False, terminated=True)
    row_c1_retry.camp_id = 1
    row_c1_retry.settlement_guard = 0.0

    row_c2_retry = _row(5, retry=False, terminated=True)
    row_c2_retry.camp_id = 2
    row_c2_retry.settlement_guard = 0.0

    call_count = 0

    def mock_execute(
        *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any], list[LaborRolloutRow], dict[int, bool]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"turn": 6}, [row_c1_failed, row_c2_healthy], {1: True, 2: False}
        else:
            return {"turn": 6}, [row_c1_retry, row_c2_retry], {1: False, 2: False}

    founded = {1: 0, 2: 0}
    with patch(
        "anthropocene_rl.autonomous_v7.labor_training._execute_single_step_detailed",
        side_effect=mock_execute,
    ):
        _next_step, rows = _sample_world_step(
            bridge=mock_bridge,
            step={"turn": 5, "villages": []},
            actor=MagicMock(),
            task_critic=MagicMock(),
            cost_critics=MagicMock(),
            normalizer=MagicMock(),
            personality=MagicMock(),
            generator=MagicMock(),
            world_seed=50000,
            world_index=0,
            founded_turns=founded,
            max_self_destruct_rewinds=1,
        )

    # Must contain exactly 3 rows:
    # 1. Rewound failed row ONLY for Camp 1
    #    (synth_camp = -(1 * 1_000_000 + 5 * 100 + 0 + 1) = -1000501, terminated=True)
    # 2. Retried Camp 1 row (camp_id = 1, terminated=True)
    # 3. Retried Camp 2 row (camp_id = 2, terminated=True)
    # Crucially, Camp 2 must NOT have a failed terminated=True row from the first attempt!
    assert len(rows) == 3
    synth_rows = [r for r in rows if r.camp_id < 0]
    assert len(synth_rows) == 1
    expected_synth_camp = -(1 * 1_000_000 + 5 * 100 + 0 + 1)
    assert synth_rows[0].camp_id == expected_synth_camp
    assert synth_rows[0].terminated is True
    assert synth_rows[0].settlement_guard == 1.0

    camp2_rows = [r for r in rows if r.camp_id == 2]
    assert len(camp2_rows) == 1
    assert camp2_rows[0] is row_c2_retry


def test_fit_normalizer_cli_and_initialize_e2e(tmp_path: Path) -> None:
    norm_path = tmp_path / "fitted-normalizer.pt"
    # Run fit-normalizer CLI for 1 turn on world 50000
    labor_training_main([
        "fit-normalizer",
        "--worlds", "50000",
        "--turns", "1",
        "--out", str(norm_path),
    ])

    assert norm_path.is_file()
    saved = torch.load(norm_path, map_location="cpu", weights_only=True)
    assert "feature_contract" in saved
    assert saved["feature_contract"] == BOUNDED_FEATURE_CONTRACT
    assert "normalizer" in saved

    # Verify that initialize CLI loads it cleanly
    init_path = tmp_path / "trainer-init.pt"
    labor_training_main([
        "initialize",
        "--normalizer-checkpoint", str(norm_path),
        "--world-seed", "50000",
        "--optimizer-seed", "42",
        "--out", str(init_path),
    ])
    assert init_path.is_file()
    init_state = torch.load(init_path, map_location="cpu", weights_only=True)
    assert "actor" in init_state
    assert "task_critic" in init_state
    assert "cost_critics" in init_state
