import math

import pytest

from anthropocene_rl.autonomous_v7.gate34_continuation_collector import (
    CampOneStepValue,
    CampReward,
    ContinuationPlan,
    ContinuationRegimeBinding,
    ForkedAlternatives,
    FrozenGeometricSchedule,
    PolicyExpectationPlan,
    RandomizedTailEstimate,
    TailAssumptions,
    aggregate_camp_prefix,
    aggregate_ranking_prediction,
    bounded_geometric_tail_assumptions,
    close_prefix_target,
    combine_independent_path_returns,
    estimate_paired_tail_difference,
    estimate_randomized_tail,
    freeze_geometric_schedule,
    geometric_stop_indices,
    raw_camp_reward_absolute_bound,
    shaped_nonnegative_component_absolute_bound,
    shaped_welfare_reward_absolute_bound,
    shaped_welfare_reward_at_fork,
    target_from_randomized_tail,
)
from anthropocene_rl.autonomous_v7.provenance import ProvenanceError


class FakeBridge:
    def __init__(self, *, bad_restore: bool = False, restore_error: bool = False):
        self.calls = []
        self.bad_restore = bad_restore
        self.restore_error = restore_error

    def fork(self):
        self.calls.append(("fork",))
        return 17, {"turn": 9, "fingerprint": "fixed"}

    def restore(self, fork):
        self.calls.append(("restore", fork))
        if self.restore_error:
            raise RuntimeError("restore failed")
        return {
            "turn": 9,
            "fingerprint": "changed" if self.bad_restore else "fixed",
        }

    def drop(self, fork):
        self.calls.append(("drop", fork))


def test_fork_scope_restores_before_every_alternative_and_before_drop():
    bridge = FakeBridge()
    with ForkedAlternatives(bridge) as branches:
        assert branches.decision_identity == (9, "fixed")
        assert branches.begin("policy")["turn"] == 9
        assert branches.begin("native")["fingerprint"] == "fixed"
        with pytest.raises(ValueError, match="duplicate"):
            branches.begin("policy")
    assert bridge.calls == [
        ("fork",),
        ("restore", 17),
        ("restore", 17),
        ("restore", 17),
        ("drop", 17),
    ]


def test_fork_scope_drops_on_branch_exception_and_on_failed_cleanup_restore():
    bridge = FakeBridge()
    with pytest.raises(LookupError, match="branch failed"), ForkedAlternatives(bridge) as branches:
        branches.begin("policy")
        raise LookupError("branch failed")
    assert bridge.calls[-2:] == [("restore", 17), ("drop", 17)]

    broken = FakeBridge(restore_error=True)
    with pytest.raises(RuntimeError, match="restore failed"), ForkedAlternatives(broken):
        pass
    assert broken.calls[-1] == ("drop", 17)


def test_fork_scope_rejects_restore_identity_drift_but_still_drops():
    bridge = FakeBridge(bad_restore=True)
    with (
        pytest.raises(ValueError, match="changed decision identity"),
        ForkedAlternatives(bridge) as branches,
    ):
        branches.begin("policy")
    assert bridge.calls[-1] == ("drop", 17)


def test_camp_prefix_follows_only_opening_camp_sequences_and_discounts_each():
    gamma = math.exp(-20.0 / 400.0)
    rows = [
        CampReward(10, 0, 2.0, 20.0, False),
        CampReward(10, 1, 3.0, 20.0, False),
        CampReward(20, 0, 5.0, 20.0, True),
        # A daughter can be required to derive a contemporaneous lineage reward, but it
        # is not a successor in either opening camp's PPO sequence.
        CampReward(99, 0, 1000.0, 20.0, False),
        CampReward(99, 1, 1000.0, 20.0, False),
    ]
    prefix = aggregate_camp_prefix(rows, [10, 20], 2)
    assert prefix.prefix_return == pytest.approx(2.0 + gamma * 3.0 + 5.0)
    assert prefix.opening_camp_count == 2
    assert prefix.terminated_camp_count == 1
    assert prefix.live_tail_discounts == pytest.approx((gamma**2,))


def test_camp_prefix_rejects_missing_or_post_termination_outcomes():
    with pytest.raises(ValueError, match="incomplete prefix"):
        aggregate_camp_prefix([CampReward(1, 0, 1, 20, False)], [1], 2)
    with pytest.raises(ValueError, match="after termination"):
        aggregate_camp_prefix(
            [CampReward(1, 0, 0, 20, True), CampReward(1, 1, 1, 20, False)],
            [1],
            2,
        )


def test_ranking_prediction_uses_each_actions_successor_and_ignores_new_daughters():
    gamma = math.exp(-20.0 / 400.0)
    terms = aggregate_ranking_prediction(
        [
            CampOneStepValue(10, 2.0, 20.0, 11.0, False),
            CampOneStepValue(20, 5.0, 20.0, 0.0, True),
            CampOneStepValue(99, 1000.0, 20.0, 1000.0, False),
        ],
        [10, 20],
    )
    assert terms.immediate_reward == 7.0
    assert terms.successor_value == 11.0
    assert terms.value == pytest.approx(7.0 + gamma * 11.0)


def test_ranking_prediction_rejects_a_value_on_a_terminated_successor():
    with pytest.raises(ValueError, match="zero successor"):
        CampOneStepValue(10, 1.0, 20.0, 3.0, True)


def test_tail_closure_is_exact_only_for_terminal_sequences():
    terminal = aggregate_camp_prefix([CampReward(1, 0, 7, 20, True)], [1], 5)
    target = close_prefix_target(terminal, "trade_utility", "terminal-proof", shaped=True)
    assert target.tail_kind == "terminal"
    assert target.point_value == 7.0


def test_raw_qol_and_guard_have_two_sided_geometric_tail_bounds():
    gamma = math.exp(-20.0 / 400.0)
    rows = [CampReward(1, i, 20.0, 20.0, False) for i in range(25)]
    prefix = aggregate_camp_prefix(rows, [1], 25)
    qol = close_prefix_target(prefix, "qol_years", "raw-qol", shaped=False)
    assert qol.tail_kind == "bounded"
    assert qol.tail_lower == 0.0
    assert qol.tail_upper == pytest.approx(gamma**25 * 20.0 / (1.0 - gamma))

    hundred = aggregate_camp_prefix(
        [CampReward(1, i, 0.0, 20.0, False) for i in range(100)], [1], 100
    )
    guard = close_prefix_target(hundred, "guard_cost", "raw-guard", shaped=False)
    assert guard.tail_upper == pytest.approx(gamma**100 / (1.0 - gamma))


def test_raw_population_and_welfare_are_lower_bounded_but_signed_or_shaped_are_not():
    prefix = aggregate_camp_prefix([CampReward(1, 0, 4.0, 20.0, False)], [1], 1)
    for objective in ("welfare", "population_years"):
        target = close_prefix_target(prefix, objective, objective, shaped=False)
        assert target.tail_kind == "bounded"
        assert target.tail_lower == 0.0
        assert target.tail_upper is None
    assert (
        close_prefix_target(prefix, "trade_utility", "trade", shaped=False).tail_kind
        == "unresolved"
    )
    assert close_prefix_target(prefix, "qol_years", "shaped", shaped=True).tail_kind == "unresolved"


def test_plan_counts_worlds_not_snapshots_and_has_a_bounded_step_budget():
    plan = ContinuationPlan(
        world_seeds=tuple(range(50000, 50016)),
        regime_ids=("default", "farming", "fishing", "herding"),
        decision_steps=(5, 15, 25),
        alternative_ids=("sampled-policy-first-action", "native-first-step"),
        continuation_turns=100,
    )
    assert plan.summary() == {
        "independentWorldCount": 16,
        "regimeCountPerWorld": 4,
        "frozenStateCount": 192,
        "branchCount": 384,
        "continuationTurnsPerBranch": 100,
        "bridgeStepUpperBoundPerCheckpoint": 40000,
    }


def test_plan_rejects_confirmation_seeds_and_duplicate_state_axes():
    with pytest.raises(ProvenanceError, match="confirmation seed denied"):
        ContinuationPlan((3000,), ("default",), (5,), ("policy",), 10)
    with pytest.raises(ValueError, match="decision_steps"):
        ContinuationPlan((50000,), ("default",), (5, 5), ("policy",), 10)
    with pytest.raises(ValueError, match="alternative_ids"):
        ContinuationPlan((50000,), ("default",), (5,), ("policy", "policy"), 10)
    with pytest.raises(ValueError, match="regime_ids"):
        ContinuationPlan((50000,), ("default", "default"), (5,), ("policy",), 10)


def test_geometric_stop_transform_has_declared_inclusive_survival_semantics():
    assert geometric_stop_indices((0.0, 0.49, 0.5, 0.75, 0.875), 0.5) == (
        0,
        0,
        1,
        2,
        3,
    )
    first = freeze_geometric_schedule(
        schedule_id="path-0",
        rng_seed=90210,
        draw_count=16,
        survival_probability=0.95,
        hard_cap_terms=200,
    )
    second = freeze_geometric_schedule(
        schedule_id="path-0",
        rng_seed=90210,
        draw_count=16,
        survival_probability=0.95,
        hard_cap_terms=200,
    )
    assert first == second
    assert first.generator == "numpy.random.PCG64"
    assert first.expected_terms_per_draw == pytest.approx(20.0)


def test_bounded_geometric_proof_requires_survival_above_squared_discount():
    gamma = math.exp(-20.0 / 400.0)
    assumptions = bounded_geometric_tail_assumptions(
        per_turn_absolute_bound=20.0,
        discount=gamma,
        survival_probability=0.95,
        provenance_id="raw-qol-bound",
    )
    assert assumptions.absolute_integrability_proven
    assert assumptions.finite_variance_proven
    with pytest.raises(ValueError, match=r"discount\*\*2"):
        bounded_geometric_tail_assumptions(
            per_turn_absolute_bound=20.0,
            discount=gamma,
            survival_probability=0.90,
            provenance_id="insufficient-survival",
        )


def test_raw_camp_reward_bounds_use_fixed_opening_camps_and_int32_population():
    maximum = 2_147_483_647
    for objective in ("welfare", "population_years"):
        assert raw_camp_reward_absolute_bound(objective, 2) == 2 * maximum * 20.0
    assert raw_camp_reward_absolute_bound("qol_years", 2) == 40.0
    assert raw_camp_reward_absolute_bound("guard_cost", 2) == 2.0
    with pytest.raises(ValueError, match="signed trade utility"):
        raw_camp_reward_absolute_bound("trade_utility", 2)


def test_recovered_shaping_bounds_respect_inactive_commitments_stage():
    # Recovered T3 stored break_cost=.01 but stage=mixed had commitments=False.
    assert shaped_welfare_reward_absolute_bound(
        100.0,
        80.0,
        mix_change_cost=0.02,
        break_cost=0.01,
        commitments_active=False,
    ) == pytest.approx(101.6)
    assert shaped_nonnegative_component_absolute_bound(
        100.0,
        mix_change_cost=0.02,
        break_cost=0.01,
        commitments_active=False,
    ) == pytest.approx(102.0)
    assert shaped_welfare_reward_absolute_bound(
        100.0,
        80.0,
        mix_change_cost=0.02,
        break_cost=0.01,
        commitments_active=True,
    ) == pytest.approx(102.4)


def test_shaped_fork_reward_requires_prior_mix_unless_camp_sequence_initial():
    with pytest.raises(ValueError, match="do not reset mix-change shaping"):
        shaped_welfare_reward_at_fork(
            100.0,
            80.0,
            (0.25, 0.25, 0.5),
            previous_policy_mix=None,
            sequence_initial_decision=False,
            mix_change_cost=0.02,
        )

    # TV((.5,.5,0),(.25,.25,.5))=.5, hence a .01 share of native welfare.
    assert shaped_welfare_reward_at_fork(
        100.0,
        80.0,
        (0.25, 0.25, 0.5),
        previous_policy_mix=(0.5, 0.5, 0.0),
        sequence_initial_decision=False,
        mix_change_cost=0.02,
    ) == pytest.approx(99.2)
    # The same sequence-initial rule applies to a newly born camp's first decision.
    assert shaped_welfare_reward_at_fork(
        100.0,
        80.0,
        (0.25, 0.25, 0.5),
        previous_policy_mix=None,
        sequence_initial_decision=True,
        mix_change_cost=0.02,
    ) == 100.0


def test_randomized_tail_uses_inverse_survival_weights_and_labels_se_scope():
    schedule = FrozenGeometricSchedule("known", 0.5, (0, 1), 2)
    assumptions = TailAssumptions(True, True, "bounded-geometric-proof")
    result = estimate_randomized_tail((2.0, 3.0), schedule, assumptions)
    assert result.status == "identified"
    assert result.draw_estimates == pytest.approx((2.0, 8.0))
    assert result.estimate == pytest.approx(5.0)
    assert result.standard_error == pytest.approx(3.0)
    assert result.stopping_randomization_contribution_to_standard_error == pytest.approx(3.0)
    assert "conditional on one observed continuation path" in result.standard_error_scope


def test_randomized_tail_withholds_on_assumptions_or_hard_cap_without_redraw():
    schedule = FrozenGeometricSchedule("too-long", 0.95, (0, 200), 200)
    proven = TailAssumptions(True, True, "proof")
    capped = estimate_randomized_tail((), schedule, proven)
    assert capped.status == "unresolved-hard-cap"
    assert capped.estimate is None
    assert schedule.stop_indices == (0, 200)

    ordinary = FrozenGeometricSchedule("ordinary", 0.95, (0, 1), 200)
    assert (
        estimate_randomized_tail(
            (), ordinary, TailAssumptions(False, False, "no-growth-proof")
        ).status
        == "unresolved-absolute-integrability"
    )
    assert (
        estimate_randomized_tail(
            (), ordinary, TailAssumptions(True, False, "no-second-moment-proof")
        ).status
        == "unresolved-finite-variance"
    )


def test_randomized_tail_rejects_an_incomplete_collected_path():
    schedule = FrozenGeometricSchedule("known", 0.5, (0, 1), 2)
    with pytest.raises(ValueError, match="need 2, received 1"):
        estimate_randomized_tail((2.0,), schedule, TailAssumptions(True, True, "proof"))

    one_draw = FrozenGeometricSchedule("one-draw", 0.5, (0,), 1)
    assert (
        estimate_randomized_tail((2.0,), one_draw, TailAssumptions(True, True, "proof")).status
        == "unresolved-insufficient-draws"
    )


def test_paired_tail_difference_uses_one_schedule_for_both_alternatives():
    schedule = FrozenGeometricSchedule("paired", 0.5, (0, 1), 2)
    result = estimate_paired_tail_difference(
        (2.0, 3.0),
        (1.0, 3.0),
        schedule,
        TailAssumptions(True, True, "paired-proof"),
    )
    assert result.draw_estimates == pytest.approx((1.0, 1.0))
    assert result.estimate == pytest.approx(1.0)
    assert result.standard_error == 0.0
    assert "paired stopping-randomization" in result.standard_error_scope


def test_independent_policy_paths_combine_each_own_prefix_and_tail_within_world():
    paths = [
        RandomizedTailEstimate("identified", "path-0", 2.0, 0.2, (2.0, 2.0), "path scope", 0.2),
        RandomizedTailEstimate("identified", "path-1", 8.0, 0.3, (8.0, 8.0), "path scope", 0.3),
    ]
    # Full path returns are 10+2=12 and -2+8=6.  Averaging tails first and adding
    # one shared prefix would instead erase prefix variation and covariance.
    result = combine_independent_path_returns((10.0, -2.0), paths, "world-50000-state-5")
    assert result.estimate == pytest.approx(9.0)
    assert result.standard_error == pytest.approx(3.0)
    assert result.estimate_kind == "full-return"
    assert result.stopping_randomization_contribution_to_standard_error == pytest.approx(
        math.sqrt(0.2**2 + 0.3**2) / 2
    )
    assert "conditional on one world" in result.standard_error_scope
    assert "cluster by world" in result.standard_error_scope

    with pytest.raises(ValueError, match="distinct frozen stopping schedules"):
        combine_independent_path_returns(
            (0.0, 0.0), [paths[0], paths[0]], "not-independent-paths"
        )
    with pytest.raises(ValueError, match="own fixed prefix"):
        combine_independent_path_returns((0.0,), paths, "missing-prefix")


def test_randomized_tail_becomes_independent_target_only_when_identified():
    identified = RandomizedTailEstimate(
        "identified", "path-0", 4.0, 0.5, (3.0, 5.0), "path scope", 0.5
    )
    target = target_from_randomized_tail(10.0, identified, "target-proof")
    assert target.tail_kind == "independent-estimate"
    assert target.point_value == 14.0
    assert target.tail_standard_error == 0.5

    unresolved = RandomizedTailEstimate("unresolved-hard-cap", "path-1")
    withheld = target_from_randomized_tail(10.0, unresolved, "cap-proof")
    assert withheld.tail_kind == "unresolved"
    assert withheld.point_value is None


def test_continuation_binding_separates_training_policy_from_deterministic_pane():
    hashes = {
        "actor_checkpoint_sha256": "a" * 64,
        "critic_checkpoint_sha256": "b" * 64,
        "checkpoint_metadata_sha256": "f" * 64,
        "normalizer_sha256": "c" * 64,
        "simulator_sha256": "1" * 64,
        "bridge_sha256": "2" * 64,
        "configuration_sha256": "3" * 64,
        "producer_sha256": "d" * 64,
        "reward_contract_sha256": "e" * 64,
        "checkpoint_resolution_id": "raw-schema12-no-upgrade-no-defaults",
        "critic_return_contract_id": "camp-sequence-discounted-continuation-v1",
        "stage_id": "checkpoint-bound-stage",
        "personality_id": "frozen-personality",
        "shaping_id": "frozen-shaping",
        "reward_semantics": "raw-stationary",
        "simulator_rng_schedule_id": "sim-paths-v1",
    }
    sampled = ContinuationRegimeBinding(
        pane="sampled-training-policy",
        action_rng_schedule_id="actor-paths-v1",
        exploration_contract_id="ppo.explore-clamp-then-sample-v1",
        log_std_floor=-4.0,
        **hashes,
    )
    assert sampled.supports_full_critic_calibration

    deterministic = ContinuationRegimeBinding(pane="deterministic-intervention", **hashes)
    assert not deterministic.supports_full_critic_calibration
    with pytest.raises(ValueError, match="requires exact exploration"):
        ContinuationRegimeBinding(
            pane="sampled-training-policy",
            action_rng_schedule_id="actor-paths-v1",
            **hashes,
        )

    shaped = dict(hashes)
    shaped["reward_semantics"] = "history-conditioned-shaped"
    shaped_sampled = ContinuationRegimeBinding(
        pane="sampled-training-policy",
        action_rng_schedule_id="actor-paths-v1",
        exploration_contract_id="ppo.explore-clamp-then-sample-v1",
        log_std_floor=-2.0,
        shaping_history_id="exact-prior-sampled-mix-on-main-path-v1",
        rollout_boundary_rule_id="zero-only-at-fresh-world-first-decision-v1",
        **shaped,
    )
    assert not shaped_sampled.supports_full_critic_calibration
    assert shaped_sampled.supports_history_conditioned_critic_diagnostic

    with pytest.raises(ValueError, match="prior-policy history"):
        ContinuationRegimeBinding(
            pane="sampled-training-policy",
            action_rng_schedule_id="actor-paths-v1",
            exploration_contract_id="ppo.explore-clamp-then-sample-v1",
            log_std_floor=-2.0,
            **shaped,
        )


def test_policy_expectation_plan_counts_paths_but_worlds_remain_independent_units():
    grid = ContinuationPlan(
        world_seeds=tuple(range(50000, 50016)),
        regime_ids=("default", "farming", "fishing", "herding"),
        decision_steps=(5, 15, 25),
        alternative_ids=("sampled-policy-first-action", "native-first-step"),
        continuation_turns=225,
    )
    plan = PolicyExpectationPlan(
        grid,
        independent_path_replicates=4,
        fixed_prefix_terms=25,
        hard_cap_tail_terms_per_path=200,
        stopping_draws_per_path=16,
        paired_alternative_paths=True,
    )
    assert plan.summary() == {
        "independentUnit": "world",
        "independentWorldCount": 16,
        "regimeCountPerWorld": 4,
        "frozenStateCount": 192,
        "alternativeCountPerState": 2,
        "independentPathReplicatesPerAlternative": 4,
        "fixedPrefixTermsPerPath": 25,
        "stoppingDrawsPerPath": 16,
        "pairedAlternativePaths": True,
        "hardCapTailTermsPerPath": 200,
        "bridgeStepHardCapPerCheckpoint": 347200,
    }
    with pytest.raises(ValueError, match="must equal"):
        PolicyExpectationPlan(
            grid,
            independent_path_replicates=4,
            fixed_prefix_terms=25,
            hard_cap_tail_terms_per_path=100,
            stopping_draws_per_path=16,
            paired_alternative_paths=True,
        )
    with pytest.raises(ValueError, match="paired path"):
        PolicyExpectationPlan(
            grid,
            independent_path_replicates=4,
            fixed_prefix_terms=25,
            hard_cap_tail_terms_per_path=200,
            stopping_draws_per_path=16,
            paired_alternative_paths=False,
        )
