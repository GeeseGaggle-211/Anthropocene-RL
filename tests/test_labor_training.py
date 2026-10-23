from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7.labor_model import LaborPersonality
from anthropocene_rl.autonomous_v7.labor_training import (
    LABOR_ROLLOUT_KIND,
    LaborRolloutRow,
    _critic_td_losses,
    _load_rollout,
    _state_dict_rollout,
    _transition_rewards,
    _validate_training_data_separation,
    argument_parser,
    build_labor_training_batch,
)


def _tensors(rows: int) -> dict[str, torch.Tensor]:
    return {
        "core": torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
        "mask": torch.ones(rows, 2, dtype=torch.bool),
    }


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
