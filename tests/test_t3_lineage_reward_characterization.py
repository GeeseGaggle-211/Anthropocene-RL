"""Synthetic characterization of the recovered T3 per-camp reward semantics.

These tests pin observable behavior; they do not assert that a centralized lineage objective
was intended, and they do not establish that the behavior caused the measured T3 outcome.
The recovered producer omitted ``founded_turn``. The current compatibility fallback is the
same batch-first-observation rule, so these fixtures deliberately omit that optional field.
"""

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import data


def rows(records: list[dict]) -> dict[str, np.ndarray]:
    types = {
        "world": np.int64,
        "camp": np.int64,
        "turn": np.int64,
        "society": np.int64,
        "parent_society": np.int64,
        "prior_population": np.float64,
        "qol": np.float64,
        "years": np.float64,
        "terminated": np.bool_,
        "trade_utility": np.float64,
    }
    return {
        name: np.asarray([record.get(name, 0) for record in records], dtype=dtype)
        for name, dtype in types.items()
    }


def record(
    camp: int,
    turn: int,
    society: int,
    parent: int,
    population: float,
    qol: float,
    *,
    terminated=False,
):
    return {
        "world": 1,
        "camp": camp,
        "turn": turn,
        "society": society,
        "parent_society": parent,
        "prior_population": population,
        "qol": qol,
        "years": 20.0,
        "terminated": terminated,
        "trade_utility": 0.0,
    }


def test_missing_founding_time_makes_lineage_grouping_depend_on_batch_truncation() -> None:
    full = rows(
        [
            record(2, 0, 20, 10, 100, 0.0),
            record(2, 1, 20, 10, 100, 0.0),
            record(2, 2, 20, 10, 100, 0.0),
            record(1, 3, 10, 0, 100, 1.0),
            record(2, 3, 20, 10, 100, 0.0),
        ]
    )
    truncated = {name: values[3:] for name, values in full.items()}

    full_reward = data.lineage_rewards(full)
    truncated_reward = data.lineage_rewards(truncated)
    # In full history the daughter is three turns old and no longer grouped. In the
    # truncated batch its first visible row is treated as its founding turn and it is grouped.
    assert full_reward[3:].tolist() == [2000.0, 0.0]
    assert truncated_reward.tolist() == [1000.0, 1000.0]


def _shared_parameter_gradient(objective_rows: np.ndarray) -> float:
    parameter = torch.tensor(1.0, requires_grad=True)
    loss = -(parameter * torch.as_tensor(objective_rows)).mean()
    return float(torch.autograd.grad(loss, parameter)[0])


def test_fission_duplicates_per_camp_qol_rows_and_changes_row_mean_gradient_weight() -> None:
    no_fission = rows(
        [
            record(1, 5, 10, 0, 100, 1.0),
            record(3, 5, 30, 0, 100, 0.0),
        ]
    )
    fission = rows(
        [
            record(1, 5, 10, 0, 50, 1.0),
            record(2, 5, 20, 10, 50, 1.0),
            record(3, 5, 30, 0, 100, 0.0),
        ]
    )
    before = data.reward_components(no_fission)["qol_years"]
    after = data.reward_components(fission)["qol_years"]

    # The target lineage's physical population-weighted QoL-years remains 1 * 20 = 20.
    # The implemented per-camp objective emits that same 20 on each living camp row.
    assert before.tolist() == [20.0, 0.0]
    assert after.tolist() == [20.0, 20.0, 0.0]
    assert before[0] == after[0] == after[1]
    assert len(data.sequences(fission)) == 3  # GAE topology remains one sequence per camp.
    # For a synthetic shared log-probability derivative, the row-mean PPO coefficient changes
    # solely because the daughter adds another objective row.
    assert _shared_parameter_gradient(before) == pytest.approx(-10.0)
    assert _shared_parameter_gradient(after) == pytest.approx(-40.0 / 3.0)


def test_terminated_relative_reduces_survivor_reward_and_breaks_literal_sum_claim() -> None:
    transition = rows(
        [
            record(1, 5, 10, 0, 100, 1.0),
            record(2, 5, 20, 10, 100, 1.0, terminated=True),
        ]
    )
    reward = data.lineage_rewards(transition)
    # Dead prior population remains in the QoL denominator, while its own reward row is then
    # zeroed. This can be interpreted as a shared extinction penalty, but it is not literally
    # conserved realized welfare for the surviving 100 people (100 * 1 * 20 = 2000).
    assert reward.tolist() == [1000.0, 0.0]
    assert reward.sum() != 2000.0
