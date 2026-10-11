from __future__ import annotations

from copy import deepcopy

from anthropocene_rl.training.overfit import (
    _default_readiness_snapshot_ids,
    _readiness_matrix_gates,
)


def test_default_readiness_snapshots_are_three_distinct_cultivation_states() -> None:
    snapshot_ids = _default_readiness_snapshot_ids(
        "datasets/manifests/elder-v2-stage0-v1.json"
    )
    assert len(snapshot_ids) == 3
    assert len(set(snapshot_ids)) == 3
    assert tuple(value[:3] for value in snapshot_ids) == ("f01", "f03", "f13")


def test_readiness_matrix_requires_replicated_actor_and_return_signal() -> None:
    snapshot_ids = ("f01-snapshot", "f03-snapshot", "f13-snapshot")
    seeds = (1701, 1901, 2101)
    gate_names = (
        "inactive_heads_gradient_inert",
        "no_illegal_action_scored",
        "action_channels_preserved",
        "compiler_interventions_not_pathological",
        "finite_and_saturation_guards",
        "checkpoint_reload_policy_parity",
        "checkpoint_continuation_parity",
        "native_replay_valid",
        "movement_disabled",
        "fixed_policy_evaluation_contract",
        "critic_return_fit",
        "ppo_update_stability",
        "actor_moved_consistently",
    )
    rows = []
    for snapshot_id in snapshot_ids:
        for seed in seeds:
            gates = {name: True for name in gate_names}
            if seed == seeds[-1]:
                gates["actor_moved_consistently"] = False
            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "gates": gates,
                    "fixed_policy_return_improvement": (
                        0.5 if snapshot_id == "f03-snapshot" else 0.0
                    ),
                }
            )

    passing = _readiness_matrix_gates(rows, snapshot_ids, seeds)
    assert all(passing.values())

    regressed = deepcopy(rows)
    regressed[0]["fixed_policy_return_improvement"] = -0.1
    failed = _readiness_matrix_gates(regressed, snapshot_ids, seeds)
    assert not failed["fixed_policy_outcomes_do_not_regress"]
