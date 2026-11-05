"""DAgger ``--prefix actor-from-reset``: options, validation, state fidelity and BC aggregation.

Bridge-free. The collector and the autonomous evaluator run on the same fake world, whose
state is the history of decisions (plus a hidden counter a state-touching shadow query
would perturb), so the actor-from-reset rollout must visit exactly the evaluator's states.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import bridge as bridge_module
from anthropocene_rl.autonomous_v7 import (
    labor_bc,
    labor_dagger,
    labor_model,
    labor_personality,
    labor_training,
)
from anthropocene_rl.autonomous_v7 import labor_bridge as labor_bridge_module
from anthropocene_rl.autonomous_v7 import labor_pretrain_eval as ev
from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.labor_bc import LaborResetOptions
from anthropocene_rl.autonomous_v7.labor_bridge import SPECIALTIES, SPECIALTY_ROWS
from anthropocene_rl.autonomous_v7.provenance import digest

FARMER = SPECIALTIES.index("Farmer")
WORLD = 90100

# ---------------------------------------------------------------------------
# Source options: the default prefix is unchanged
# ---------------------------------------------------------------------------


class _State:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    def state_dict(self) -> dict[str, Any]:
        return dict(self.value)


def _parse(*extra: str) -> argparse.Namespace:
    return labor_dagger.argument_parser().parse_args(
        ["collect", "--checkpoint", "x.pt", "--worlds", "90100-90102", "--round", "2",
         "--depth", "3", "--starts", "2", "--horizon", "25", "--out", "o", *extra]
    )


def _options(args: argparse.Namespace, worlds: list[int]) -> dict[str, Any]:
    return labor_dagger._source_options(
        args,
        worlds,
        labor_dagger._branch_starts(args, worlds),
        checkpoint_sha="c" * 64,
        reset=LaborResetOptions(players=16, width=128, height=64, separation=9),
        reference=_State({"qol_years": 0.5, "population_years": 0.5}),
        conditioning=_State({"distribution": "fixed"}),
        personality=SimpleNamespace(iteration=0),
        normalizer_digest="n" * 64,
        tail=None,
    )


LEGACY_OPTIONS_DIGEST = "d24a717db546101291cab822365418291d6945d85660ace96c2fe32930e9756d"


def test_default_prefix_source_options_are_unchanged() -> None:
    """The pre-``--prefix`` options dict, built verbatim; its digest is pinned."""
    args = _parse()
    assert args.prefix == "teacher"
    worlds = [90100, 90101, 90102]
    starts = {w: labor_dagger.draw_start_turns(4243, w, 2, 2, 25, 3) for w in worlds}
    legacy = {
        "contract": labor_dagger.DAGGER_CONTRACT,
        "worlds": worlds,
        "turns": 25,
        "round": 2,
        "depth": 3,
        "starts_per_world": 2,
        "start_key": 4243,
        "starts": {str(w): s for w, s in starts.items()},
        "actor_mode": "deterministic",
        "policy_seed": 0,
        "checkpoint_sha256": "c" * 64,
        "reset_options": LaborResetOptions(
            players=16, width=128, height=64, separation=9
        ).state_dict(),
        "personality": {"qol_years": 0.5, "population_years": 0.5},
        "personality_conditioning": {"distribution": "fixed"},
        "personality_iteration": 0,
        "normalizer_digest": "n" * 64,
        "teacher": "h12-converted-native-heuristic-shadow-step",
        "actor_only": True,
        "partition": "fitting",
    }
    options = _options(args, worlds)
    assert options == legacy and "prefix" not in options
    assert json.dumps(options, sort_keys=True) == json.dumps(legacy, sort_keys=True)
    assert digest(options) == LEGACY_OPTIONS_DIGEST
    # The explicit default spelling is the same record.
    assert _options(_parse("--prefix", "teacher"), worlds) == legacy
    # With a continuation the record gains only the (unchanged) teacher_continuation entry.
    tail = labor_bc.RandomizedTailSettings()
    cont = labor_dagger._source_options(
        _parse("--teacher-continuation", "branch-end"), worlds,
        labor_dagger._branch_starts(args, worlds), checkpoint_sha="c" * 64,
        reset=LaborResetOptions(players=16, width=128, height=64, separation=9),
        reference=_State({"qol_years": 0.5, "population_years": 0.5}),
        conditioning=_State({"distribution": "fixed"}), personality=SimpleNamespace(iteration=0),
        normalizer_digest="n" * 64, tail=tail,
    )
    assert cont == dict(
        legacy, teacher_continuation={"states": "branch-end", "return_tail": tail.state_dict()}
    )


def test_actor_from_reset_options_and_validation() -> None:
    worlds = [90100, 90101]
    args = _parse("--prefix", "actor-from-reset", "--depth", "50", "--horizon", "50",
                  "--starts", "1")
    labor_dagger._validate_prefix(args)
    assert labor_dagger._branch_starts(args, worlds) == {90100: [0], 90101: [0]}
    options = _options(args, worlds)
    assert options["prefix"] == "actor-from-reset" and options["starts"] == {
        "90100": [0], "90101": [0]
    }
    assert digest(options) != digest(_options(_parse(), worlds))
    with pytest.raises(ValueError, match="--depth must equal --horizon"):
        labor_dagger._validate_prefix(
            _parse("--prefix", "actor-from-reset", "--depth", "10", "--horizon", "50",
                   "--starts", "1")
        )
    with pytest.raises(ValueError, match="--starts 1"):
        labor_dagger._validate_prefix(
            _parse("--prefix", "actor-from-reset", "--depth", "50", "--horizon", "50",
                   "--starts", "3")
        )
    stochastic = _parse("--prefix", "actor-from-reset", "--depth", "50", "--horizon", "50",
                        "--starts", "3", "--stochastic")
    labor_dagger._validate_prefix(stochastic)
    assert labor_dagger._branch_starts(stochastic, worlds)[90100] == [0, 0, 0]
    assert _options(stochastic, worlds)["actor_mode"] == "stochastic"
    # The teacher prefix keeps any depth <= horizon and its keyed draws.
    labor_dagger._validate_prefix(_parse())
    with pytest.raises(SystemExit):
        _parse("--prefix", "actor")


def test_run_collect_refuses_bad_actor_from_reset_before_any_work(tmp_path) -> None:
    for extra in (["--depth", "10", "--horizon", "50", "--starts", "1"],
                  ["--depth", "50", "--horizon", "50", "--starts", "2"]):
        args = labor_dagger.argument_parser().parse_args(
            ["collect", "--checkpoint", str(tmp_path / "missing.pt"), "--worlds", "90100",
             "--round", "1", "--prefix", "actor-from-reset", *extra, "--out", str(tmp_path / "o")]
        )
        with pytest.raises(ValueError):
            labor_dagger.run_collect(args)
        assert not (tmp_path / "o").exists()


# ---------------------------------------------------------------------------
# Fake world shared by the collector and the evaluator
# ---------------------------------------------------------------------------

CAMPS = (1, 2, 3)
RESET = {"players": 3, "movement": True, "withheld": 0}


@dataclass
class _Attempt:
    valid: bool = True
    violations: tuple = ()
    feasibility_cost: float = 0.0
    accepted_specialty_residents: tuple[int, ...] = field(
        default_factory=lambda: tuple([0] * SPECIALTY_ROWS)
    )


@dataclass
class _Execution:
    step: dict[str, Any]
    initial_attempts: dict[int, _Attempt]
    retry: dict[int, Any]
    final_attempts: dict[int, _Attempt]


@dataclass(frozen=True)
class _Catalog:
    camp_id: int
    population: int


class _FakeLaborBridge:
    """Toy world: the state is (decision history, hidden counter). Camp 3 dies at turn 2."""

    perturb_on_capture = False

    def __init__(self) -> None:
        self.history: list[str] = []
        self.hidden = 0
        self.forks: dict[int, tuple[list[str], int]] = {}
        self._bridge = self
        self.last_reset_identity = None
        self.captures = 0

    def __enter__(self) -> _FakeLaborBridge:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def capture(self) -> tuple[Any, ...]:
        self.captures += 1
        if self.perturb_on_capture:
            self.hidden += 1  # a state-touching shadow query (what the probe must exclude)
        return (tuple(self.history), self.hidden)

    def _step(self) -> dict[str, Any]:
        n = len(self.history)
        signature = sum((i + 1) * (3 if c == "actor" else 5) for i, c in enumerate(self.history))
        villages = []
        for camp in CAMPS:
            dead = camp == 3 and n >= 2
            population = 0 if dead else 10 + camp * n + self.hidden
            villages.append(
                {
                    "camp": camp,
                    "priorPopulation": 10 + camp * max(0, n - 1),
                    "population": population,
                    "welfare": float(signature * camp + self.hidden) / 7.0,
                    "years": 20.0,
                    "housing": 0.5 + 0.01 * n,
                    "clothing": 0.25 + 0.02 * (n + self.hidden),
                    "nutrition": 0.5 if camp == 3 else 1.0,
                    "terminated": dead,
                    "observation": None if dead else {"camp": camp},
                }
            )
        return {"turn": n, "villages": villages}

    def reset(self, world: int, **kwargs: Any) -> dict[str, Any]:
        self.history, self.hidden = [], 0
        self.last_reset_identity = {"world": world, **{k: str(v) for k, v in kwargs.items()}}
        return self._step()

    def fork(self) -> tuple[int, None]:
        key = len(self.forks) + 1
        self.forks[key] = (list(self.history), self.hidden)
        return key, None

    def restore(self, key: int) -> dict[str, Any]:
        self.history, self.hidden = list(self.forks[key][0]), self.forks[key][1]
        return self._step()

    def drop(self, key: int) -> None:
        del self.forks[key]

    def validate_labor(self, actions: dict[int, Any], attempt: int = 0) -> dict[int, _Attempt]:
        return {camp: _Attempt() for camp in actions}

    def advance(self, catalogs: dict[int, Any]) -> _Execution:
        self.history.append("actor")
        farmers = 1 + len(self.history) % 3
        return _Execution(
            step=self._step(),
            initial_attempts={c: _Attempt(valid=c != 1) for c in catalogs},
            retry={1: object()},
            final_attempts={
                c: _Attempt(
                    accepted_specialty_residents=tuple(
                        farmers if i == FARMER else 0 for i in range(SPECIALTY_ROWS)
                    )
                )
                for c in catalogs
            },
        )


class _PerturbingLaborBridge(_FakeLaborBridge):
    perturb_on_capture = True


class _FakeHeur:
    def __init__(self, economic_diagnostics: bool = True) -> None:
        self.resets = 0

    def __enter__(self) -> _FakeHeur:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def reset(self, world: int, *, movement: bool, withheld: int, snapshot: Any) -> None:
        self.resets += 1

    def step(self, action: Any) -> dict[str, Any]:
        return {"villages": [{"camp": c} for c in CAMPS], "economicDiagnostics": {}}


@pytest.fixture
def world(monkeypatch):
    seen: dict[str, list[Any]] = {"actor": [], "rows": [], "generators": []}

    def tensors(live, normalizer, *, personality=None, **kwargs):
        assert normalizer == "NORM" and personality == "actor-p"
        contexts = tuple(
            (_Catalog(int(v["camp"]), int(v["population"])), None, None) for v in live
        )
        return contexts, {"movement_mask": torch.ones((len(live), 4), dtype=torch.bool)}

    def actor_decision(bridge, actor, normalizer, personality, live, catalogs, t, **kw):
        assert actor == "ACTOR" and personality == "actor-p"
        seen["actor"].append((bridge.history[:], bridge.hidden))
        seen["generators"].append(kw["generator"].initial_seed())
        return bridge.advance(catalogs)

    class _Label:
        reasons = ()

    class _Action:
        payload: ClassVar[dict[str, Any]] = {}

        def validate(self, catalog: Any, retry: bool = False) -> None:
            return None

    monkeypatch.setattr(labor_model, "labor_policy_tensors", tensors)
    monkeypatch.setattr(labor_model, "labor_checkpoint_personality", lambda state: "actor-p")
    monkeypatch.setattr(labor_personality, "personality_for_world", lambda p, w: p)
    monkeypatch.setattr(ev, "actor_labor_decision", actor_decision)
    monkeypatch.setattr(
        ev, "_load_models", lambda path: ({"state": 1}, "ACTOR", "TASK", "COSTS", "NORM")
    )
    monkeypatch.setattr(bridge_module, "Model9Bridge", _FakeHeur)
    monkeypatch.setattr(
        labor_bc, "convert_heuristic_labor_label", lambda *a, **k: _Label()
    )
    monkeypatch.setattr(labor_dagger, "_retained_action", lambda label, catalog, mask: _Action())
    monkeypatch.setattr(labor_dagger, "privileged_row_tensors", lambda live: {})
    monkeypatch.setattr(
        labor_dagger,
        "_record_row",
        lambda result, tensors, row, label, **kw: seen["rows"].append(
            (kw["start_index"], kw["branch_step"], kw["decision_turn"], kw["camp"])
        ),
    )
    monkeypatch.setattr(
        labor_training,
        "_transition_rewards",
        lambda step, camps, founded, **kw: (
            {c: {"qol_years": 0.0, "population_years": 0.0} for c in camps},
            {c: False for c in camps},
            {c: 20.0 for c in camps},
        ),
    )
    monkeypatch.setattr(labor_training, "_settlement_guard", lambda *a: 0.0)
    return seen


def _collect(labor_factory, *, depth=4, starts=(0,), stochastic=False):
    rollouts: list[dict[str, Any]] = []
    heur: list[_FakeHeur] = []

    def heur_factory():
        heur.append(_FakeHeur())
        return heur[-1]

    result = labor_dagger.collect_learner_world(
        actor="ACTOR", normalizer="NORM", personality="actor-p", world_seed=WORLD,
        round_index=1, depth=depth, starts=list(starts), stochastic=stochastic,
        reset_options=LaborResetOptions.from_mapping(RESET),
        labor_bridge_factory=labor_factory, heuristic_bridge_factory=heur_factory,
        rollouts=rollouts,
    )
    return result, rollouts, heur[0]


def _evaluate(monkeypatch, labor_factory, turns=4):
    monkeypatch.setattr(labor_bridge_module, "LaborModel9Bridge", labor_factory)
    return ev._run_job(
        {"checkpoint": "x.pt", "turns": turns, "policy_seed": 0, "record_critic_trace": False,
         "world": WORLD, "mode": "deterministic",
         "reset_options": dict(RESET)}
    )


def test_actor_from_reset_visits_the_evaluator_states(world, monkeypatch) -> None:
    result, rollouts, heur = _collect(_FakeLaborBridge)
    collector_states = list(world["actor"])
    world["actor"].clear()
    run = _evaluate(monkeypatch, _FakeLaborBridge)
    assert collector_states == world["actor"]  # the same states before every actor decision
    assert len(rollouts) == 1 and rollouts[0]["start_index"] == 0 and rollouts[0]["start"] == 0
    assert labor_dagger.compare_rollout(rollouts[0], run["turns"], 4) is None
    assert rollouts[0]["decision"] == [0, 1, 2, 3]
    assert rollouts[0]["live_settlements"] == [3, 2, 2, 2]
    assert rollouts[0]["terminations"] == [0, 1, 0, 0]
    assert rollouts[0]["starvation_extinctions"] == [0, 1, 0, 0]
    columns = labor_dagger.rollout_columns(run["turns"])
    assert rollouts[0] == {"start_index": 0, "start": 0, **columns}
    # Every visited state was teacher-labelled (one shadow query per decision), with no prefix.
    assert heur.resets == 4 and result.counters["teacher_prefix_decisions"] == 0
    expected = [(0, 0)] * 3 + [(1, 1)] * 3 + [(2, 2)] * 2 + [(3, 3)] * 2
    assert [r[1:3] for r in world["rows"]] == expected
    assert result.decisions == 4 and result.counters["branches"] == 1


def test_a_state_touching_shadow_query_is_detected(world, monkeypatch) -> None:
    _, rollouts, _ = _collect(_PerturbingLaborBridge)
    run = _evaluate(monkeypatch, _PerturbingLaborBridge)  # the evaluator never captures
    difference = labor_dagger.compare_rollout(rollouts[0], run["turns"], 4)
    assert difference is not None and difference["decision"] == 0
    assert {f["field"] for f in difference["fields"]} >= {"residents", "welfare"}
    # Shorter evaluator output is a length difference, not a pass.
    _, clean, _ = _collect(_FakeLaborBridge)
    short = _evaluate(monkeypatch, _FakeLaborBridge, turns=2)
    assert labor_dagger.compare_rollout(clean[0], short["turns"], 4)["fields"] == ["<length>"]
    assert labor_dagger.compare_rollout(clean[0], short["turns"], 2) is None


def test_stochastic_rollouts_from_reset_use_independent_keyed_generators(world) -> None:
    result, rollouts, _ = _collect(_FakeLaborBridge, depth=2, starts=(0, 0, 0), stochastic=True)
    assert [r["start_index"] for r in rollouts] == [0, 1, 2]
    assert all(r["start"] == 0 for r in rollouts)
    # Every branch starts from the reset: the fork restores it between branches.
    assert [s for s, _ in world["actor"]] == [[], ["actor"]] * 3
    seeds = world["generators"][::2]
    assert seeds == [labor_dagger._stream_seed(0, WORLD, 1, k) for k in range(3)]
    assert len(set(seeds)) == 3
    assert result.counters["branches"] == 3


# ---------------------------------------------------------------------------
# rollout-compare CLI on a published dataset
# ---------------------------------------------------------------------------


def _publish(directory, rollouts, *, prefix="actor-from-reset", checkpoint_sha="", turns=4):
    options = {"prefix": prefix, "actor_mode": "deterministic", "starts_per_world": 1,
               "turns": turns, "checkpoint_sha256": checkpoint_sha,
               "reset_options": {"players": 3}}
    if prefix is None:
        del options["prefix"]
    body = {"options": options, "source_kind": labor_dagger.DAGGER_SOURCE_KIND,
            "schema": ltd.SCHEMA}
    source = dict(body, source_run=digest(body))
    directory.mkdir(parents=True)
    shard = directory / f"world-{WORLD}"
    meta = {"source_run": source["source_run"], "world_seed": WORLD,
            "reset_identity": {"world": WORLD}, "rollouts": rollouts}
    arrays = {"row_world": np.full(2, WORLD, np.int64), "row_camp": np.arange(2, dtype=np.int64),
              "row_turn": np.zeros(2, np.int64), "row_attempt": np.zeros(2, np.int64)}
    record = ltd.write_shard(shard, arrays, meta)
    ltd.publish_dataset(directory, source, [{
        "path": shard.name, "world_seed": WORLD, "rows": 2,
        "meta_sha256": ltd.checkpoint_hash(shard / "meta.json"),
        "data_digest": record["data_digest"],
    }])


def test_rollout_compare_cli(tmp_path, world, monkeypatch) -> None:
    _, rollouts, _ = _collect(_FakeLaborBridge)
    run = _evaluate(monkeypatch, _FakeLaborBridge)
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"weights")
    sha = ltd.checkpoint_hash(checkpoint)
    report = {"checkpoint": str(checkpoint), "reset_options": {"players": 3},
              "runs": [{"world": WORLD, "mode": "heuristic", "turns": []},
                       dict(run, reset_identity={"world": WORLD})]}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    good = tmp_path / "good"
    _publish(good, rollouts, checkpoint_sha=sha)
    out = tmp_path / "cmp.json"
    argv = ["rollout-compare", "--dagger", str(good), "--autonomous", str(report_path),
            "--out", str(out)]
    assert labor_dagger.main(argv) == 0
    assert json.loads(out.read_text())["ok"] is True
    assert labor_dagger.main([*argv[:-2], "--decisions", "2"]) == 0

    bad_rollout = json.loads(json.dumps(rollouts))
    bad_rollout[0]["residents"][2] += 1
    bad = tmp_path / "bad"
    _publish(bad, bad_rollout, checkpoint_sha=sha)
    assert labor_dagger.main(["rollout-compare", "--dagger", str(bad), "--autonomous",
                              str(report_path), "--out", str(out)]) == 3
    first = json.loads(out.read_text())["worlds"][str(WORLD)]["first_difference"]
    assert first["decision"] == 2 and first["fields"][0]["field"] == "residents"
    # Wrong checkpoint, teacher prefix or too few evaluator turns: refused.
    other = tmp_path / "other"
    _publish(other, rollouts, checkpoint_sha="0" * 64)
    assert labor_dagger.main(["rollout-compare", "--dagger", str(other), "--autonomous",
                              str(report_path)]) == 3
    teacher = tmp_path / "teacher"
    _publish(teacher, rollouts, prefix=None, checkpoint_sha=sha)
    assert labor_dagger.main(["rollout-compare", "--dagger", str(teacher), "--autonomous",
                              str(report_path)]) == 3
    assert labor_dagger.main([*argv[:-2], "--decisions", "5"]) == 3


# ---------------------------------------------------------------------------
# BC aggregation: four teacher batches + one DAgger dataset
# ---------------------------------------------------------------------------


def test_teacher_batches_plus_dagger_get_equal_source_mass(tmp_path) -> None:
    from tests.test_labor_dagger import _fake_dataset

    batches = []
    for index, worlds in enumerate(([5000, 5002], [5004, 5006])):
        batch = tmp_path / f"teacher-{index}"
        _fake_dataset(batch, ltd.SOURCE_KIND, worlds, 10 + 4 * index, extra=False)
        batches.append(batch)
    dagger = tmp_path / "dagger"
    _fake_dataset(dagger, labor_dagger.DAGGER_SOURCE_KIND, [5000, 5002, 5004, 5006], 5, extra=True)
    mixed = ltd.load_labor_datasets([*batches, dagger])  # what pretrain-bc reads (no merge)
    w, src, worlds = mixed["row_weight"], mixed["row_source"], mixed["row_world"]
    total = float(w.sum())
    # The teacher batches pool into one source (row_source 0): DAgger gets half the mass ...
    assert math.isclose(float(w[src == 0].sum()), total / 2, rel_tol=1e-6)
    assert math.isclose(float(w[src == 1].sum()), total / 2, rel_tol=1e-6)
    # ... and every (source, world) an equal share within its source.
    for seed in (5000, 5002, 5004, 5006):
        for code in (0, 1):
            mass = float(w[(src == code) & (worlds == seed)].sum())
            assert math.isclose(mass, total / 8, rel_tol=1e-6)
    assert len(w) == 2 * 10 + 2 * 14 + 4 * 5  # BC iterates every row (steps = ceil(N / batch))
    # Family multipliers never touch untagged (DAgger) rows; mass is kept per (source, world).
    from anthropocene_rl.autonomous_v7.labor_scenarios import (
        FAMILY_BITS,
        apply_family_weights,
        row_family_tags,
    )

    base = np.ones(6)
    tags = np.asarray([FAMILY_BITS["under-clothed"], 0, 0, 0, 0, 0])
    sources = np.asarray([0, 0, 0, 1, 1, 1])
    out = apply_family_weights(base, tags, sources, np.full(6, 5000), {"under-clothed": 2.0})
    assert out[0] == pytest.approx(1.5) and out[1] == pytest.approx(0.75)
    assert out[:3].sum() == pytest.approx(3.0) and list(out[3:]) == [1.0, 1.0, 1.0]
    assert row_family_tags.__defaults__[1] == (0,)  # tracked sources: teacher trajectories only
