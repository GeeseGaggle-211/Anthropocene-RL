"""V^rho targets at learner-visited states: DAgger teacher continuations (amendment r5-b item 6).

Every learner-visited state gets one teacher (H12) continuation closed by the
randomized tail, in the teacher collector's row format with the same typed
tail fields and a distinct source kind that only ``pretrain-critic
--learner-state-continuations`` accepts. The actor rows are unchanged.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts, labor_bc, labor_dagger
from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.labor_bc import (
    CONTINUATION_ROW_SOURCE,
    CONTINUATION_SOURCE_KIND,
    CRITIC_TEACHER_SOURCE_KIND,
    TAIL_HEADS,
    TAIL_NONE_TERMINAL,
    TAIL_RANDOMIZED,
    TRAJECTORY_FIELD,
    RandomizedTailSettings,
    discounted_returns,
    pretrain_labor_actor_bc,
    require_critic_teacher_dataset,
)

GAMMA = math.exp(-contracts.TURN_YEARS / contracts.HORIZON_YEARS)


def _rows(camps, *, trajectory, source, terminal=False, tail=10.0):
    n = len(camps)
    data = {
        "row_world": torch.full((n,), 90100, dtype=torch.long),
        "row_camp": torch.tensor(camps, dtype=torch.long),
        "row_turn": torch.full((n,), 3, dtype=torch.long),
        "row_quarantined": torch.zeros(n, dtype=torch.bool),
        "row_source": torch.full((n,), source, dtype=torch.long),
        TRAJECTORY_FIELD: torch.tensor(trajectory, dtype=torch.long),
        "transition_elapsed_years": torch.full((n,), contracts.TURN_YEARS, dtype=torch.float64),
        "transition_terminal": torch.full((n,), terminal, dtype=torch.bool),
        "reward_qol_years": torch.ones(n, dtype=torch.float64),
        "reward_population_years": torch.zeros(n, dtype=torch.float64),
        "reward_feasibility": torch.zeros(n, dtype=torch.float64),
        "reward_settlement_guard": torch.zeros(n, dtype=torch.float64),
        "tail_method": torch.full((n,), TAIL_NONE_TERMINAL if terminal else TAIL_RANDOMIZED),
        "tail_cutoff_turn": torch.full((n,), 4, dtype=torch.long),
    }
    for head in TAIL_HEADS:
        value = 0.0 if terminal else (tail if head == "qol_years" else 0.0)
        data["tail_value__" + head] = torch.full((n,), value, dtype=torch.float64)
        data["tail_standard_error__" + head] = torch.zeros(n, dtype=torch.float64)
    return data


def test_each_continuation_is_its_own_trajectory() -> None:
    # Two continuations of the same (world, camp, turn) from different branches.
    data = _rows([4, 4], trajectory=[0, 3], source=CONTINUATION_ROW_SOURCE)
    data["tail_value__qol_years"] = torch.tensor([10.0, 20.0], dtype=torch.float64)
    returns = discounted_returns(data)["qol_years"]
    expected = torch.tensor([1 + GAMMA * 10, 1 + GAMMA * 20], dtype=torch.float64)
    assert torch.allclose(returns.double(), expected, atol=1e-5)
    # Without the trajectory key the two rows would be one (world, camp) trajectory.
    merged = {k: v for k, v in data.items() if k != TRAJECTORY_FIELD}
    with pytest.raises(ValueError, match="tail value differs"):
        discounted_returns(merged)


def test_critic_accepts_continuations_only_with_the_flag() -> None:
    data = _rows([4], trajectory=[0], source=CONTINUATION_ROW_SOURCE)
    data["collection"] = {"source_kind": CONTINUATION_SOURCE_KIND, "actor_only": False}
    with pytest.raises(ValueError, match="learner-state-continuations"):
        require_critic_teacher_dataset(data)
    require_critic_teacher_dataset(data, allow_learner_state_continuation=True)
    mixed = _rows([4, 5], trajectory=[-1, 0], source=0)
    mixed["row_source"] = torch.tensor([0, CONTINUATION_ROW_SOURCE])
    mixed["collection"] = {
        "source_kind": "aggregated",
        "actor_only": False,
        "component_collections": [
            {"source_kind": CRITIC_TEACHER_SOURCE_KIND},
            {"source_kind": CONTINUATION_SOURCE_KIND},
        ],
    }
    require_critic_teacher_dataset(mixed, allow_learner_state_continuation=True)
    with pytest.raises(ValueError, match="aggregated"):
        require_critic_teacher_dataset(mixed)
    dagger = dict(mixed, collection=dict(mixed["collection"], component_collections=[
        {"source_kind": CRITIC_TEACHER_SOURCE_KIND}, {"source_kind": "learner_state_dagger"}]))
    with pytest.raises(ValueError, match="learner_state_dagger"):
        require_critic_teacher_dataset(dagger, allow_learner_state_continuation=True)
    actor_only = dict(data, collection={"source_kind": CONTINUATION_SOURCE_KIND,
                                        "actor_only": True})
    with pytest.raises(ValueError, match="actor-only"):
        require_critic_teacher_dataset(actor_only, allow_learner_state_continuation=True)
    # Continuation rows whose tails were dropped by aggregation are one-step returns: refused.
    lost = {k: v for k, v in mixed.items() if not k.startswith("tail_")}
    with pytest.raises(ValueError, match="lost their randomized tail"):
        require_critic_teacher_dataset(lost, allow_learner_state_continuation=True)
    with pytest.raises(ValueError, match="row_source"):
        require_critic_teacher_dataset(
            dict(mixed, row_source=torch.tensor([0, 1])), allow_learner_state_continuation=True
        )


def test_actor_bc_refuses_continuation_rows() -> None:
    data = {
        "target_specialty": torch.zeros(1, 14),
        "row_world": torch.tensor([90100]),
        "row_source": torch.tensor([CONTINUATION_ROW_SOURCE]),
    }
    with pytest.raises(ValueError, match="not actor labels"):
        pretrain_labor_actor_bc(actor=None, dataset=data)
    aggregated = {
        "target_specialty": torch.zeros(1, 14),
        "row_world": torch.tensor([90100]),
        "collection": {"source_kind": "aggregated",
                       "component_collections": [{"source_kind": CONTINUATION_SOURCE_KIND}]},
    }
    with pytest.raises(ValueError, match="not actor labels"):
        pretrain_labor_actor_bc(actor=None, dataset=aggregated)
    assert ltd.ROW_SOURCES[CONTINUATION_SOURCE_KIND] == CONTINUATION_ROW_SOURCE
    assert ltd.AGGREGATE_IDENTITY_DEFAULTS[TRAJECTORY_FIELD] == -1


class FakeBridge:
    def __init__(self) -> None:
        self.events: list[str] = []

    def fork(self):
        self.events.append("fork")
        return 7, None

    def restore(self, fork):
        self.events.append(f"restore-{fork}")
        return {"turn": 3}

    def drop(self, fork):
        self.events.append(f"drop-{fork}")


def test_teacher_continuation_records_typed_tails_and_restores(monkeypatch) -> None:
    settings = RandomizedTailSettings(draws=2, rho=0.91)
    required = settings.schedule(90100).required_terms
    calls = {"n": 0}

    def fake_transition(**kwargs):
        term = calls["n"]
        calls["n"] += 1
        kwargs["personality"].table["daughter"] = term  # continuation mutates the lineage table
        return SimpleNamespace(
            decision_turn=3 + term,
            contexts=[(SimpleNamespace(camp_id=camp),) for camp in (1, 2)],
            elapsed={1: contracts.TURN_YEARS, 2: contracts.TURN_YEARS},
            terminal={1: False, 2: term == 0},
            head_rewards=lambda camp, founded, term=term: {
                "qol_years": 1.0 + term, "population_years": 10.0, "feasibility": 0.0,
                "settlement_guard": 0.0,
            },
            next_step={"turn": 4 + term},
        )

    def fake_record(result, transition, world, founded):
        for camp in (1, 2):
            result.rows.append({
                "row_world": torch.tensor([world]),
                "row_camp": torch.tensor([camp]),
                "row_turn": torch.tensor([transition.decision_turn]),
                "transition_terminal": torch.tensor([transition.terminal[camp]]),
            })

    monkeypatch.setattr(labor_bc, "_teacher_transition", fake_transition)
    monkeypatch.setattr(labor_bc, "record_teacher_transition", fake_record)
    bridge = FakeBridge()
    personality = SimpleNamespace(table={"root": 1})
    identity = {"row_round": 1, "row_depth": 1, "row_start": 2, "row_start_index": 0,
                "row_branch_step": 1, TRAJECTORY_FIELD: 1}
    collected = labor_dagger.teacher_continuation(
        labor_bridge=bridge, heur_bridge=None,
        step={"turn": 3, "villages": [{"observation": {}, "terminated": False, "camp": 1}]},
        world=90100, options=None, normalizer=None, personality=personality, founded={},
        tail=settings, identity=identity,
    )
    assert bridge.events == ["fork", "restore-7", "drop-7"]
    assert personality.table == {"root": 1}  # lineage table restored with the fork
    assert calls["n"] == 1 + required  # first decision plus the tail's required terms
    assert collected.native_resolves == 2 and collected.tail_native_resolves == 2 * required
    rows = {int(r["row_camp"]): r for r in collected.rows}
    assert int(rows[1]["tail_method"]) == TAIL_RANDOMIZED  # living after the first decision
    assert int(rows[2]["tail_method"]) == TAIL_NONE_TERMINAL  # terminated by it
    for row in collected.rows:
        assert int(row["row_source"]) == CONTINUATION_ROW_SOURCE
        assert {k: int(row[k]) for k in identity} == identity
    assert collected.tail_record["prefix_terms"] == 1


# Live native (local bridge) -----------------------------------------------------


@pytest.fixture(scope="module")
def live_paths(tmp_path_factory):
    from anthropocene_rl.autonomous_v7.labor_model import BOUNDED_FEATURE_CONTRACT
    from anthropocene_rl.autonomous_v7.labor_training import main as labor_training_main
    from tests.test_labor_bc import _fit_test_normalizer

    root = tmp_path_factory.mktemp("continuation")
    norm = root / "norm.pt"
    torch.save(
        {"feature_contract": BOUNDED_FEATURE_CONTRACT,
         "normalizer": _fit_test_normalizer().state_dict()},
        norm,
    )
    init = root / "init.pt"
    labor_training_main(["initialize", "--normalizer-checkpoint", str(norm), "--world-seed",
                         "5000", "--optimizer-seed", "1", "--campaign-r5", "--out", str(init)])
    return root, norm, init


def test_live_continuations_leave_actor_rows_unchanged_and_feed_the_critic(live_paths) -> None:
    from anthropocene_rl.autonomous_v7.labor_privileged import shard_keys
    from anthropocene_rl.autonomous_v7.labor_training import main as labor_training_main

    root, _, init = live_paths
    tail = ["--tail-draws", "2", "--tail-rho", "0.91"]
    common = ["collect", "--checkpoint", str(init), "--worlds", "5000", "--round", "1",
              "--depth", "2", "--starts", "1", "--horizon", "3", "--players", "8",
              "--jobs", "1", "--allow-dirty"]
    plain, with_c, cont = root / "plain", root / "with", root / "cont"
    assert labor_dagger.main([*common, "--out", str(plain)]) == 0
    assert labor_dagger.main([*common, *tail, "--teacher-continuation", "all-visited",
                              "--continuation-out", str(cont), "--out", str(with_c)]) == 0
    a, b = ltd.load_labor_dataset(plain), ltd.load_labor_dataset(with_c)
    for key, value in a.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, b[key]), key  # actor rows unchanged by the continuations
    manifest = ltd.verify_dataset(cont)
    assert manifest["source_record"]["source_kind"] == CONTINUATION_SOURCE_KIND
    c = ltd.load_labor_dataset(cont, require_privileged=True)
    assert c["collection"]["return_tail"] == labor_bc.RETURN_TAIL_RANDOMIZED
    assert bool((c["row_source"] == CONTINUATION_ROW_SOURCE).all())
    assert set(c["row_branch_step"].tolist()) == {1, 2}
    assert set(shard_keys()) <= set(c)
    # Inputs at the learner-visited state after one actor decision equal that DAgger row's.
    sel_c = c["row_branch_step"] == 1
    sel_d = a["row_branch_step"] == 1
    assert bool(sel_c.any()) and int(sel_c.sum()) == int(sel_d.sum())
    for key, value in a.items():
        if not isinstance(value, torch.Tensor) or key.startswith(
            ("reward_", "transition_", "row_", "target_", "label_", "tail_")
        ):
            continue
        assert torch.equal(value[sel_d], c[key][sel_c]), key
    living = ~c["transition_terminal"]
    assert bool((c["tail_method"][living] != TAIL_NONE_TERMINAL).all())
    meta = ltd.verify_shard(cont / "world-5000")
    assert meta["return_tail_record"]["prefix_terms"] == 1
    assert len(meta["continuation_costs"]) == 2  # one per learner-visited state

    critic = root / "critic.pt"
    with pytest.raises(ValueError, match="learner-state-continuations"):
        labor_training_main(["pretrain-critic", "--checkpoint", str(init), "--dataset",
                             str(cont), "--epochs", "1", "--out", str(critic)])
    labor_training_main(["pretrain-critic", "--checkpoint", str(init), "--dataset", str(cont),
                         "--learner-state-continuations", "--require-randomized-tail",
                         "--epochs", "1", "--batch-size", "16", "--out", str(critic)])
    summary = torch.load(critic, map_location="cpu", weights_only=True)[
        "heuristic_critic_pretraining"
    ]
    assert summary["learner_state_continuation_rows"] > 0
    with pytest.raises(ValueError, match="not actor labels"):
        labor_training_main(["pretrain-bc", "--checkpoint", str(init), "--dataset", str(cont),
                             "--epochs", "1", "--out", str(root / "bc.pt")])
