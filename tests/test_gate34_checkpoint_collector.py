import copy
import json
from math import exp
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from anthropocene_rl.autonomous_v7 import gate34_checkpoint_collector as collector
from anthropocene_rl.autonomous_v7.gate34_checkpoint_collector import (
    DEVELOPMENT_NATIVE_INTERVENTION,
    FROZEN_ENDPOINT_IDS,
    FROZEN_PROTOCOL,
    ActionInterventionReceipt,
    EndpointRegistry,
    FrozenCollectionProtocol,
    _recovered_lineage_rewards,
    apply_first_action_intervention,
    direction_counts,
    load_endpoint_registry,
    load_endpoint_runtime,
)
from anthropocene_rl.autonomous_v7.gate34_continuation_collector import (
    CampReward,
    FrozenGeometricSchedule,
    TailAssumptions,
)
from anthropocene_rl.autonomous_v7.provenance import ProvenanceError

REPOSITORY = Path(__file__).resolve().parents[1]
FINAL2 = REPOSITORY / "evidence/model9-gate1-freeze-candidate-final2-20261023"


def test_real_final2_registry_schema_resolves_exact_historical_endpoints():
    if not FINAL2.is_dir():
        pytest.skip("real final2 Gate-1 package is not present")
    registry = load_endpoint_registry(FINAL2, verify_checkpoint_bytes=False)
    assert isinstance(registry, EndpointRegistry)
    assert tuple(endpoint.node_id for endpoint in registry.endpoints) == FROZEN_ENDPOINT_IDS
    assert len(registry.endpoints) == 16
    assert registry.endpoints[0].arm_id is None
    assert registry.endpoints[0].optimizer_seed is None
    assert [endpoint.optimizer_seed for endpoint in registry.endpoints[1:]] == list(
        range(70, 85)
    )
    assert {endpoint.loading["adapter"] for endpoint in registry.endpoints} == {
        "recovered-contract12-inactive-region"
    }
    assert len(registry.registry_sha256) == 64


def test_real_final2_all_checkpoint_actor_critic_contracts_load_strictly():
    if not FINAL2.is_dir():
        pytest.skip("real final2 Gate-1 package is not present")
    registry = load_endpoint_registry(FINAL2)
    loaded = [load_endpoint_runtime(registry, endpoint) for endpoint in registry.endpoints]
    assert [runtime.vector_critic for runtime in loaded] == [
        False,
        *([True] * 5),
        *([False] * 5),
        *([True] * 5),
    ]
    assert [runtime.personality_spec for runtime in loaded] == [
        "default",
        *(["dirichlet:1.0"] * 5),
        *(["default"] * 5),
        *(["dirichlet:1.0"] * 5),
    ]
    for runtime in loaded:
        bins = runtime.critic.state_dict()["bins"]
        assert bins.ndim == 1
        assert np.all(np.diff(bins.numpy()) > 0)
        assert runtime.log_std_floor == -2.0


def test_registry_rejects_noncanonical_nested_artifact_digest(tmp_path: Path):
    if not FINAL2.is_dir():
        pytest.skip("real final2 Gate-1 package is not present")
    for name in (
        "freeze-candidate.json",
        "artifact-dag.json",
        "recovered-contract12-adapter-protocol.json",
    ):
        (tmp_path / name).write_bytes((FINAL2 / name).read_bytes())
    artifact = json.loads((tmp_path / "artifact-dag.json").read_text(encoding="utf-8"))
    artifact["digest"] = "0" * 64
    (tmp_path / "artifact-dag.json").write_text(
        json.dumps(artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    candidate = json.loads((tmp_path / "freeze-candidate.json").read_text(encoding="utf-8"))
    # Keep the outer reference byte hash current: the canonical inner digest must still fail.
    import hashlib

    candidate["proposals"]["artifact_dag"]["sha256"] = hashlib.sha256(
        (tmp_path / "artifact-dag.json").read_bytes()
    ).hexdigest()
    from anthropocene_rl.autonomous_v7.provenance import digest

    candidate["digest"] = digest(
        {key: value for key, value in candidate.items() if key != "digest"}
    )
    (tmp_path / "freeze-candidate.json").write_text(
        json.dumps(candidate, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ProvenanceError, match="canonical schema-1 sealed"):
        load_endpoint_registry(tmp_path, verify_checkpoint_bytes=False)


def test_protocol_is_exact_and_accounts_for_native_label_probe_steps():
    assert FROZEN_PROTOCOL.frozen_state_count == 16 * 4 * 3
    assert FROZEN_PROTOCOL.native_label_probe_steps_per_checkpoint == 192
    assert FROZEN_PROTOCOL.bridge_step_hard_cap_per_checkpoint == 347_392
    assert FROZEN_PROTOCOL.as_dict()["independentUnit"] == "world"
    assert FROZEN_PROTOCOL.as_dict()["decisionSteps"] == [5, 15, 25]
    with pytest.raises(ProvenanceError, match="immutable"):
        FrozenCollectionProtocol(world_seeds=(50000,))


def _decoded_action(development: float) -> dict:
    return {
        "strategic": [0.1, 0.2, development, 0.4, 0.5],
        "service": [0.0] * 6,
        "clothingRoute": 1,
        "shelterRoute": 2,
        "movement": -1,
        "fission": -1,
        "commit": 1,
    }


def test_development_intervention_proves_complete_equality_outside_one_leaf():
    before = _decoded_action(0.3)
    native = {"strategic": [0.9, 0.8, 0.75, 0.6, 0.5]}
    after, receipt = apply_first_action_intervention(
        DEVELOPMENT_NATIVE_INTERVENTION,
        camp_id=7,
        action=before,
        native_label=native,
        village={"camp": 7},
    )
    assert after["strategic"][2] == 0.75
    assert receipt.changed_paths == ("strategic[2]",)
    restored = copy.deepcopy(after)
    restored["strategic"][2] = before["strategic"][2]
    assert restored == before


def test_exact_zero_intervention_and_direction_ties_remain_in_denominators():
    action = _decoded_action(0.3)
    after, receipt = apply_first_action_intervention(
        DEVELOPMENT_NATIVE_INTERVENTION,
        camp_id=7,
        action=action,
        native_label={"strategic": [0.0, 0.0, 0.3, 0.0, 0.0]},
        village={"camp": 7},
    )
    assert after == action
    assert receipt == ActionInterventionReceipt(
        camp_id=7,
        before_sha256=receipt.before_sha256,
        after_sha256=receipt.after_sha256,
        changed_paths=(),
        exact_zero_change=True,
    )
    counts = direction_counts((1.0, 0.0, -1.0, 0.0))
    assert counts["denominator"] == 4
    assert counts["exactZeroUninformativeButRetained"] == 2
    assert counts["exactZeroFraction"] == 0.5


def test_recovered_reward_mirror_uses_full_history_for_daughter_age():
    rows = [
        {
            "world": 1,
            "camp": 1,
            "turn": 9,
            "society": 10,
            "parent_society": 0,
            "prior_population": 100,
            "qol": 1.0,
            "years": 20,
            "terminated": False,
        },
        {
            "world": 1,
            "camp": 2,
            "turn": 9,
            "society": 20,
            "parent_society": 10,
            "prior_population": 100,
            "qol": 0.0,
            "years": 20,
            "terminated": False,
        },
        {
            "world": 1,
            "camp": 1,
            "turn": 13,
            "society": 10,
            "parent_society": 0,
            "prior_population": 100,
            "qol": 1.0,
            "years": 20,
            "terminated": False,
        },
        {
            "world": 1,
            "camp": 2,
            "turn": 13,
            "society": 20,
            "parent_society": 10,
            "prior_population": 100,
            "qol": 0.0,
            "years": 20,
            "terminated": False,
        },
    ]
    full = _recovered_lineage_rewards(rows)
    truncated = _recovered_lineage_rewards(rows[2:])
    # With full history the daughter is older than LINEAGE_TURNS and separates;
    # truncated input falsely treats its first visible row as its founding row.
    assert tuple(full[2:]) == (2000.0, 0.0)
    assert tuple(truncated) == (1000.0, 1000.0)


class _BranchBridge:
    def __init__(self, terminate_after: int | None) -> None:
        self.terminate_after = terminate_after
        self.calls = 0

    def step(self, _actions: object) -> dict:
        self.calls += 1
        terminated = (
            self.terminate_after is not None and self.calls >= self.terminate_after
        )
        return {
            "turn": 9 + self.calls,
            "villages": [
                {
                    "camp": 1,
                    "society": 1,
                    "parentSociety": 0,
                    "observation": [0.0],
                    "terminated": terminated,
                }
            ],
        }


def test_execute_branch_slices_prefix_and_closes_observed_terminal_trade_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the 25+75 branch boundary without invoking actor or native code."""

    def sampled(*_args: object, **_kwargs: object) -> collector.SampledActions:
        return collector.SampledActions(
            actions={1: {"test": 0.0}},
            mixes={1: (1.0,)},
            spans={1: 20},
            personality={},
        )

    def outcome_rows(**kwargs: object) -> list[dict]:
        before = kwargs["before"]
        after = kwargs["after"]
        assert isinstance(before, dict) and isinstance(after, dict)
        return [
            {
                "camp": 1,
                "turn": int(before["turn"]),
                "step_index": int(before["turn"]) - 9,
                "years": 20.0,
                "terminated": bool(after["villages"][0]["terminated"]),
            }
        ]

    def branch_rewards(
        _runtime: object,
        _history: object,
        rows: object,
        _opening: object,
    ) -> dict[str, list[CampReward]]:
        assert isinstance(rows, list)
        return {
            "trade_utility": [
                CampReward(
                    camp_id=1,
                    step_index=int(row["step_index"]),
                    reward=1.0,
                    years=20.0,
                    terminated=bool(row["terminated"]),
                )
                for row in rows
            ]
        }

    monkeypatch.setattr(collector, "sample_training_actions", sampled)
    monkeypatch.setattr(collector, "_outcome_rows", outcome_rows)
    monkeypatch.setattr(collector, "_branch_rewards", branch_rewards)
    monkeypatch.setattr(
        collector,
        "_critic_values",
        lambda *_args, **_kwargs: {"trade_utility": {1: 0.0}},
    )
    monkeypatch.setattr(
        collector,
        "_tail_assumptions",
        lambda *_args, **_kwargs: TailAssumptions(True, True, "synthetic-bound"),
    )

    schedule = FrozenGeometricSchedule(
        schedule_id="synthetic-25-plus-75",
        survival_probability=0.95,
        stop_indices=(74, 74),
        hard_cap_terms=200,
    )
    decision = {
        "turn": 9,
        "villages": [
            {
                "camp": 1,
                "society": 1,
                "parentSociety": 0,
                "observation": [0.0],
                "terminated": False,
            }
        ],
    }
    runtime = SimpleNamespace(objectives=("trade_utility",))
    personality = collector.PersonalityState("default", 0.0, 7)

    def execute(terminate_after: int | None) -> collector.BranchResult:
        return collector._execute_branch(
            bridge=_BranchBridge(terminate_after),
            decision_step=decision,
            runtime=runtime,
            protocol=FROZEN_PROTOCOL,
            world_seed=50000,
            history=(),
            main_personalities=personality,
            previous_mixes={},
            native_labels={},
            alternative_id="sampled-policy",
            path_replicate=0,
            schedule=schedule,
            action_seed=11,
            personality_seed=13,
            capture_economic=False,
        )

    gamma = exp(-20.0 / 400.0)
    expected_prefix_25 = sum(gamma**index for index in range(25))

    living = execute(None)
    assert living.collected_steps == 100
    assert living.terminal is False
    assert living.prefix_returns["trade_utility"] == pytest.approx(expected_prefix_25)

    before_prefix = execute(10)
    assert before_prefix.collected_steps == 10
    assert before_prefix.terminal is True
    assert before_prefix.prefix_returns["trade_utility"] == pytest.approx(
        sum(gamma**index for index in range(10))
    )
    assert before_prefix.tail_estimates["trade_utility"].estimate == 0.0
    assert before_prefix.tail_estimates["trade_utility"].standard_error == 0.0

    after_prefix = execute(30)
    exact_tail = sum(gamma**index for index in range(25, 30))
    assert after_prefix.collected_steps == 30
    assert after_prefix.terminal is True
    assert after_prefix.prefix_returns["trade_utility"] == pytest.approx(
        expected_prefix_25
    )
    tail = after_prefix.tail_estimates["trade_utility"]
    assert tail.status == "identified"
    assert tail.estimate == pytest.approx(exact_tail)
    assert tail.standard_error == 0.0
    assert tail.draw_estimates == pytest.approx((exact_tail, exact_tail))
    assert tail.standard_error_scope.startswith("exact observed finite terminal suffix")
    assert after_prefix.camp_tail_estimates["trade_utility"][1] == tail
