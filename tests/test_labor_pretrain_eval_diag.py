"""Bridge-free checks of the autonomous diagnostic modes ``takeover:K`` and ``swap:GROUPS``.

The closed-loop runners are exercised with fake bridges, models and decisions that
encode who decided each turn into the simulated trajectory, so the tests check the
control flow of ``_run_h12``/``_run_actor``/``_run_takeover``/``_run_swap`` (which
controller decides which turn, counting, records), not the native simulation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import bridge as bridge_module
from anthropocene_rl.autonomous_v7 import labor_bc, labor_model, labor_ppo
from anthropocene_rl.autonomous_v7 import labor_bridge as labor_bridge_module
from anthropocene_rl.autonomous_v7 import labor_pretrain_eval as ev
from anthropocene_rl.autonomous_v7.labor_bridge import (
    ACTIVITY_ROWS,
    ROLE_ROWS,
    SPECIALTIES,
    SPECIALTY_ROWS,
)

FARMER = SPECIALTIES.index("Farmer")

# ---------------------------------------------------------------------------
# Mode parsing
# ---------------------------------------------------------------------------


def test_parse_autonomous_mode_accepts_canonical_modes() -> None:
    assert ev.parse_autonomous_mode("h12") == ("h12", None)
    assert ev.parse_autonomous_mode("deterministic") == ("deterministic", None)
    assert ev.parse_autonomous_mode("stochastic") == ("stochastic", None)
    assert ev.parse_autonomous_mode("takeover:0") == ("takeover", 0)
    assert ev.parse_autonomous_mode("takeover:25") == ("takeover", 25)
    assert ev.parse_autonomous_mode("swap:labor") == ("swap", frozenset({"specialty", "activity"}))
    assert ev.parse_autonomous_mode("swap:specialty+movement") == (
        "swap",
        frozenset({"specialty", "movement"}),
    )
    assert ev.parse_autonomous_mode("swap:trade+fission+movement+labor")[1] == frozenset(
        {"specialty", "activity", "movement", "fission", "trade"}
    )


@pytest.mark.parametrize(
    "mode",
    [
        "heuristic",
        "h13",
        "",
        "takeover",
        "takeover:",
        "takeover:-1",
        "takeover:01",
        "takeover:1.5",
        "takeover: 3",
        "takeover:" + chr(0xFF13),  # fullwidth digit three
        "swap",
        "swap:",
        "swap:labor+specialty",
        "swap:activity+labor",
        "swap:movement+movement",
        "swap:foo",
        "swap:Labor",
        "swap:labor+",
        "swap:+labor",
        "swap:tradeValue",
    ],
)
def test_parse_autonomous_mode_rejects(mode: str) -> None:
    with pytest.raises(ValueError):
        ev.parse_autonomous_mode(mode)


def test_validate_autonomous_modes() -> None:
    assert ev.validate_autonomous_modes("h12, takeover:3,swap:labor+trade") == [
        "h12",
        "takeover:3",
        "swap:labor+trade",
    ]
    assert ev.validate_autonomous_modes("deterministic,stochastic") == [
        "deterministic",
        "stochastic",
    ]
    for bad in ("", ",", "h12,h12", "takeover:1,takeover:1", "swap:", "deterministic,swap:x"):
        with pytest.raises(ValueError):
            ev.validate_autonomous_modes(bad)


def test_run_autonomous_refuses_critic_trace_with_diagnostic_modes(tmp_path) -> None:
    args = ev.argument_parser().parse_args(
        [
            "autonomous",
            "--checkpoint",
            str(tmp_path / "x.pt"),
            "--worlds",
            "2000",
            "--modes",
            "takeover:1",
            "--record-critic-trace",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    with pytest.raises(ValueError, match="critic-trace"):
        ev.run_autonomous(args)


# ---------------------------------------------------------------------------
# Payload merge
# ---------------------------------------------------------------------------


def _activity(rows: dict[int, list[float]]) -> list[list[float]]:
    result = []
    for role in range(ROLE_ROWS):
        row = [0.0] * ACTIVITY_ROWS
        values = rows.get(role, [1.0])
        row[: len(values)] = values
        result.append(row)
    return result


def _specialty(shares: dict[int, float]) -> list[float]:
    row = [0.0] * SPECIALTY_ROWS
    for index, value in shares.items():
        row[index] = value
    return row


def _h12_payload() -> dict[str, Any]:
    # Gatherer (role 0) and Farmer (role 1) staffed; every other role idle.
    return {
        "schemaVersion": 12,
        "movement": -1,
        "fission": 0,
        "tradeValue": [1.0] * 7,
        "tradeGate": [1] * 7,
        "tradeShare": 0.5,
        "laborSpecialtyAllocation": _specialty({1: 0.5, 2: 0.5}),
        "laborActivityAllocation": _activity({0: [0.2, 0.8], 1: [0.1, 0.4, 0.5]}),
    }


def _actor_payload() -> dict[str, Any]:
    # Over population 10: idle 0.5->1, Gatherer 3, Farmer 3, Herder 3, Fisher 0.1->0,
    # Craftsperson 0.4->0 (largest remainder gives the leftover resident to idle).
    return {
        "schemaVersion": 12,
        "movement": 4,
        "fission": 2,
        "tradeValue": [0.5, 2.0, 1.5, 1.0, 1.0, 3.0, 0.25],
        "tradeGate": [2, 0, 1, 1, 0, 2, 1],
        "tradeShare": 0.25,
        "laborSpecialtyAllocation": _specialty({0: 0.05, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.01, 5: 0.04}),
        "laborActivityAllocation": [
            [0.4, 0.3, 0.3] + [0.0] * (ACTIVITY_ROWS - 3) for _ in range(ROLE_ROWS)
        ],
    }


def test_swap_group_keys_partition_the_schema12_action() -> None:
    assert ev._check_swap_key_partition() == labor_ppo.LABOR_ACTION_FIELDS
    assert ev.SWAP_GROUP_KEYS == {
        "specialty": ("laborSpecialtyAllocation",),
        "activity": ("laborActivityAllocation",),
        "movement": ("movement",),
        "fission": ("fission",),
        "trade": ("tradeValue", "tradeGate", "tradeShare"),
    }
    assert ev.SWAP_SHARED_KEYS == ("schemaVersion",)


def test_merge_no_groups_is_h12_and_all_groups_is_actor() -> None:
    h12, actor = _h12_payload(), _actor_payload()
    before = copy.deepcopy((h12, actor))
    merged, substituted = ev.merge_action_payloads(h12, actor, frozenset(), population=10)
    assert merged == h12 and substituted == [] and merged is not h12
    every = ev.parse_swap_groups("labor+movement+fission+trade")
    merged, substituted = ev.merge_action_payloads(h12, actor, every, population=10)
    assert merged == actor and substituted == []
    assert (h12, actor) == before  # inputs untouched


def test_merge_takes_exactly_the_named_groups() -> None:
    h12, actor = _h12_payload(), _actor_payload()
    merged, _ = ev.merge_action_payloads(h12, actor, frozenset({"trade", "fission"}), population=10)
    for key in ("tradeValue", "tradeGate", "tradeShare", "fission"):
        assert merged[key] == actor[key]
    for key in ("movement", "laborSpecialtyAllocation", "laborActivityAllocation", "schemaVersion"):
        assert merged[key] == h12[key]
    merged, substituted = ev.merge_action_payloads(
        h12, actor, ev.parse_swap_groups("labor"), population=10
    )
    assert substituted == []
    assert merged["laborSpecialtyAllocation"] == actor["laborSpecialtyAllocation"]
    assert merged["laborActivityAllocation"] == actor["laborActivityAllocation"]
    assert merged["movement"] == h12["movement"]


def test_merge_specialty_only_substitutes_idle_rows_of_staffed_roles() -> None:
    h12, actor = _h12_payload(), _actor_payload()
    merged, substituted = ev.merge_action_payloads(
        h12, actor, frozenset({"specialty"}), population=10
    )
    # The actor staffs Herder (role 2, 3 residents) whose h12 row is role-idle: taken
    # from the actor. Gatherer/Farmer keep h12's rows; Fisher/Craftsperson get positive
    # shares but no resident, so they keep h12's idle rows.
    assert substituted == [2]
    rows = merged["laborActivityAllocation"]
    assert rows[2] == actor["laborActivityAllocation"][2]
    assert rows[0] == h12["laborActivityAllocation"][0]
    assert rows[1] == h12["laborActivityAllocation"][1]
    for role in range(3, ROLE_ROWS):
        assert rows[role] == h12["laborActivityAllocation"][role]
    assert merged["laborSpecialtyAllocation"] == actor["laborSpecialtyAllocation"]
    # A larger population staffs Craftsperson (role 4) too.
    _, substituted = ev.merge_action_payloads(h12, actor, frozenset({"specialty"}), population=100)
    assert substituted == [2, 3, 4]


def test_merge_activity_only_substitutes_from_h12_specialty_source() -> None:
    h12, actor = _h12_payload(), _actor_payload()
    actor["laborActivityAllocation"][1] = [1.0] + [0.0] * (ACTIVITY_ROWS - 1)
    merged, substituted = ev.merge_action_payloads(
        h12, actor, frozenset({"activity"}), population=10
    )
    # h12 staffs Farmer (role 1); the actor's Farmer row is role-idle -> h12's row.
    assert substituted == [1]
    assert merged["laborActivityAllocation"][1] == h12["laborActivityAllocation"][1]
    assert merged["laborActivityAllocation"][0] == actor["laborActivityAllocation"][0]
    assert merged["laborSpecialtyAllocation"] == h12["laborSpecialtyAllocation"]


def test_merge_payload_is_executable() -> None:
    masks = labor_bridge_module.LaborMasks.from_values(
        [True] * SPECIALTY_ROWS,
        [[index < 3 for index in range(ACTIVITY_ROWS)] for _ in range(ROLE_ROWS)],
    )
    for groups in ("specialty", "activity", "labor", "movement+trade"):
        merged, _ = ev.merge_action_payloads(
            _h12_payload(), _actor_payload(), ev.parse_swap_groups(groups), population=10
        )
        labor_bridge_module.LaborPolicySample.from_payload(
            {
                "laborSpecialtyAllocation": merged["laborSpecialtyAllocation"],
                "laborActivityAllocation": merged["laborActivityAllocation"],
            },
            masks,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(strategy=1),
        lambda p: p.pop("tradeShare"),
        lambda p: p.update(schemaVersion=11),
    ],
)
def test_merge_fails_loudly_on_unexpected_payloads(mutate) -> None:
    for side in ("h12", "actor"):
        h12, actor = _h12_payload(), _actor_payload()
        mutate(h12 if side == "h12" else actor)
        with pytest.raises(ValueError):
            ev.merge_action_payloads(h12, actor, frozenset({"movement"}), population=10)
    with pytest.raises(ValueError):
        ev.merge_action_payloads(_h12_payload(), _actor_payload(), {"labor"}, population=10)


# ---------------------------------------------------------------------------
# Closed-loop control flow with fakes
# ---------------------------------------------------------------------------

CAMPS = (1, 2)


@dataclass
class _Attempt:
    valid: bool = True
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
    """A deterministic toy world whose state is the history of controllers."""

    instances: ClassVar[list[_FakeLaborBridge]] = []

    def __init__(self) -> None:
        self.history: list[str] = []
        self._bridge = self
        self.last_reset_identity = None
        _FakeLaborBridge.instances.append(self)

    def __enter__(self) -> _FakeLaborBridge:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def capture(self) -> tuple[str, ...]:
        return tuple(self.history)

    def _step(self) -> dict[str, Any]:
        signature = sum((i + 1) * (1 if c == "actor" else 2) for i, c in enumerate(self.history))
        villages = []
        for camp in CAMPS:
            villages.append(
                {
                    "camp": camp,
                    "priorPopulation": 10,
                    "population": 10,
                    "welfare": float(signature + camp),
                    "years": 20.0,
                    "housing": 0.5,
                    "clothing": 0.25,
                    "nutrition": 1.0,
                    "terminated": False,
                    "observation": {"camp": camp},
                }
            )
        return {"turn": len(self.history), "villages": villages}

    def reset(self, world: int, **kwargs: Any) -> dict[str, Any]:
        self.history = []
        self.last_reset_identity = {"world": world, **{k: str(v) for k, v in kwargs.items()}}
        return self._step()

    def advance(self, controller: str, retained: dict[int, Any]) -> _Execution:
        assert set(retained) == set(CAMPS)
        self.history.append(controller)
        farmers = 3 if controller == "actor" else 5
        reject = {"actor": {1}, "h12": {2}, "swap": {1, 2}}[controller]
        return _Execution(
            step=self._step(),
            initial_attempts={c: _Attempt(valid=c not in reject) for c in CAMPS},
            retry={c: object() for c in reject},
            final_attempts={
                c: _Attempt(
                    accepted_specialty_residents=tuple(
                        farmers if i == FARMER else 0 for i in range(SPECIALTY_ROWS)
                    )
                )
                for c in CAMPS
            },
        )


class _FakeHeur:
    def __init__(self, economic_diagnostics: bool = False) -> None:
        assert economic_diagnostics
        self.resets: list[Any] = []

    def __enter__(self) -> _FakeHeur:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def reset(self, world: int, *, movement: bool, withheld: int, snapshot: Any) -> None:
        self.resets.append(snapshot)

    def step(self, action: Any) -> dict[str, Any]:
        return {"villages": [{"camp": c} for c in CAMPS], "economicDiagnostics": {}}


@dataclass(frozen=True)
class _Label:
    action: dict[str, Any]
    specialty_valid: bool = True
    activity_role_valid: Any = field(default_factory=lambda: np.ones(ROLE_ROWS, dtype=bool))
    specialty_counts: Any = field(default_factory=lambda: np.array([0, 4, 6] + [0] * 11))
    over_budget: bool = False
    movement_valid: bool = False
    fission_valid: bool = True
    trade_gate_valid: Any = field(default_factory=lambda: np.ones(7, dtype=bool))
    full_action_valid: bool = False
    reasons: tuple[str, ...] = ("movement-label-unavailable:-2",)


@pytest.fixture
def fakes(monkeypatch):
    calls: list[tuple[str, Any]] = []
    _FakeLaborBridge.instances.clear()
    monkeypatch.setattr(labor_bridge_module, "LaborModel9Bridge", _FakeLaborBridge)
    monkeypatch.setattr(bridge_module, "Model9Bridge", _FakeHeur)
    monkeypatch.setattr(
        ev, "_load_models", lambda path: ({"state": 1}, "ACTOR", "TASK", "COSTS", "NORM")
    )
    monkeypatch.setattr(labor_model, "labor_checkpoint_personality", lambda state: "actor-p")
    monkeypatch.setattr(labor_model, "LaborPersonality", lambda: "h12-p")

    def tensors(live, normalizer, *, personality=None, **kwargs):
        assert normalizer == "NORM"
        calls.append(("tensors", personality))
        contexts = tuple(
            (_Catalog(int(v["camp"]), int(v["population"])), None, None) for v in live
        )
        return contexts, {
            "movement_mask": torch.ones((len(live), 4), dtype=torch.bool),
            "personality": personality,
        }

    monkeypatch.setattr(labor_model, "labor_policy_tensors", tensors)

    def actor_decision(bridge, actor, normalizer, personality, live, catalogs, t, **kw):
        assert actor == "ACTOR" and personality == "actor-p" and t["personality"] == "actor-p"
        assert kw["stochastic"] is False and isinstance(kw["generator"], torch.Generator)
        calls.append(("actor", len(bridge.history)))
        return bridge.advance("actor", {c: None for c in catalogs})

    monkeypatch.setattr(ev, "actor_labor_decision", actor_decision)

    def convert(catalog, village_heur, econ, *, pre_observation):
        assert pre_observation == {"camp": catalog.camp_id}
        return _Label(action=dict(_h12_payload(), movement=-1))

    monkeypatch.setattr(labor_bc, "convert_heuristic_labor_label", convert)

    def retained_action(label, catalog, movement_mask):
        assert movement_mask.shape == (1, 4)
        return ("h12-retained", catalog.camp_id)

    monkeypatch.setattr(labor_bc, "_retained_action", retained_action)

    def retained_payload(payload, catalog, movement_mask):
        calls.append(("merged", payload))
        return ("swap-retained", catalog.camp_id)

    monkeypatch.setattr(labor_bc, "_retained_payload_action", retained_payload)

    def execute(bridge, catalogs, retained, retry_factory):
        assert retry_factory is labor_bc._idle_retry
        kind = next(iter(retained.values()))[0]
        controller = "h12" if kind == "h12-retained" else "swap"
        calls.append((controller, len(bridge.history)))
        return bridge.advance(controller, retained)

    monkeypatch.setattr(labor_ppo, "execute_labor_decision", execute)

    def actor_payloads(actor, t, rows):
        assert actor == "ACTOR" and t["personality"] == "actor-p"
        return [_actor_payload() for _ in range(rows)]

    monkeypatch.setattr(ev, "actor_deterministic_payloads", actor_payloads)
    return calls


def _job(mode: str, turns: int = 4) -> dict[str, Any]:
    return {
        "checkpoint": "x.pt",
        "turns": turns,
        "reset_options": {"players": 2, "movement": True, "withheld": 0},
        "policy_seed": 0,
        "record_critic_trace": False,
        "world": 2000,
        "mode": mode,
    }


def test_h12_and_actor_paths_use_their_controllers(fakes) -> None:
    h12 = ev._run_job(_job("h12"))
    assert [c for c in fakes if c[0] in ("actor", "h12", "swap")] == [("h12", t) for t in range(4)]
    assert ("tensors", "h12-p") in fakes and ("tensors", "actor-p") not in fakes
    assert h12["mode"] == "h12" and len(h12["turns"]) == 4
    assert h12["initial_rejections"] == 4 and h12["retries"] == 4
    fb = h12["h12_fallbacks"]
    assert fb["decisions"] == 8 and fb["activity_roles"] == 16
    assert fb["counts"]["movement"] == 8 and fb["counts"]["full_action"] == 8
    assert fb["counts"]["specialty"] == 0
    assert fb["reasons"] == {"movement-label-unavailable": 8}
    assert h12["turns"][0]["farmer_share"] == pytest.approx(10 / 20)
    fakes.clear()
    det = ev._run_job(_job("deterministic"))
    assert [c for c in fakes if c[0] in ("actor", "h12", "swap")] == [
        ("actor", t) for t in range(4)
    ]
    assert det["initial_rejections"] == 4 and det["retries"] == 4
    assert "h12_fallbacks" not in det
    assert det["turns"][0]["farmer_share"] == pytest.approx(6 / 20)
    assert det["turns"] != h12["turns"]


def test_takeover_endpoints_reproduce_h12_and_deterministic(fakes) -> None:
    h12 = ev._run_job(_job("h12"))
    det = ev._run_job(_job("deterministic"))
    t0 = ev._run_job(_job("takeover:0"))
    t_all = ev._run_job(_job("takeover:4"))
    t_more = ev._run_job(_job("takeover:99"))
    assert t0["turns"] == h12["turns"]
    assert t_all["turns"] == det["turns"] and t_more["turns"] == det["turns"]
    assert t0["reset_identity"] == h12["reset_identity"] == det["reset_identity"]
    assert t0["initial_rejections"] == h12["initial_rejections"]
    assert t0["h12_phase"]["h12_fallbacks"] == h12["h12_fallbacks"]
    assert t0["actor_phase"] == {"decisions": 0, "initial_rejections": 0, "retries": 0}
    assert t_more["h12_phase"]["decisions"] == 0 and t_more["handover_turn"] == 99
    assert t_more["initial_rejections"] == det["initial_rejections"]


def test_takeover_hands_over_after_k_decisions(fakes) -> None:
    run = ev._run_job(_job("takeover:2"))
    assert [c for c in fakes if c[0] in ("actor", "h12", "swap")] == [
        ("actor", 0),
        ("actor", 1),
        ("h12", 2),
        ("h12", 3),
    ]
    assert run["mode"] == "takeover:2" and run["handover_turn"] == 2
    assert run["actor_phase"] == {"decisions": 2, "initial_rejections": 2, "retries": 2}
    assert run["h12_phase"]["decisions"] == 2 and run["h12_phase"]["initial_rejections"] == 2
    assert run["h12_phase"]["h12_fallbacks"]["decisions"] == 4
    assert run["initial_rejections"] == 4 and run["retries"] == 4
    assert [t["farmer_share"] for t in run["turns"]] == pytest.approx([0.3, 0.3, 0.5, 0.5])
    # The h12 shadow resumes from the actor-controlled state.
    assert len(run["turns"]) == 4


def test_swap_submits_merged_payloads(fakes) -> None:
    run = ev._run_job(_job("swap:specialty", turns=2))
    assert [c for c in fakes if c[0] in ("actor", "h12", "swap")] == [("swap", 0), ("swap", 1)]
    merged = [c[1] for c in fakes if c[0] == "merged"]
    assert len(merged) == 4
    expected, substituted = ev.merge_action_payloads(
        _h12_payload(), _actor_payload(), frozenset({"specialty"}), population=10
    )
    assert merged[0] == expected and substituted == [2]
    assert run["mode"] == "swap:specialty"
    swap = run["swap"]
    assert swap["groups"] == ["specialty"]
    assert swap["activity_row_substitutions"] == 4  # Herder, every camp and turn
    assert swap["decisions_with_activity_row_substitution"] == 4
    assert swap["decisions_where_swapped_group_differs"] == {"specialty": 4}
    assert swap["h12_fallbacks"]["decisions"] == 4
    assert run["initial_rejections"] == 4 and run["retries"] == 4
    assert ("tensors", "actor-p") in fakes and ("tensors", "h12-p") in fakes


def test_autonomous_report_with_diagnostic_modes() -> None:
    def run(world, mode, welfare, **extra):
        turn = {
            "welfare": welfare,
            "person_years": 200.0,
            "qol": welfare / 200.0,
            "farmer_share": 0.1,
            "housing_coverage": 1.0,
            "clothing_coverage": 1.0,
            "residents": 10,
            "terminations": [],
            "starvation_extinctions": [],
        }
        return dict(
            {"world": world, "mode": mode, "turns": [turn], "reset_identity": {"f": world}},
            **extra,
        )

    base = [run(1, "heuristic", 100.0), run(1, "takeover:3", 90.0, handover_turn=3)]
    report = ev.autonomous_report(base, ["takeover:3"])
    assert "actor_vs_h12" not in report
    assert report["aggregate"]["takeover:3"]["ratios"]["welfare"] == pytest.approx(0.9)
    report.update({"worlds": [1], "turns": 1, "modes": ["takeover:3"]})
    text = ev.summarize_autonomous(report)
    assert "takeover:3 W=0.900" in text and "farmer heur/takeover:3=" in text
    fallbacks = {"decisions": 1, "activity_roles": 1, "counts": {}, "reasons": {}}
    results = [
        *base,
        run(1, "h12", 80.0, h12_fallbacks=fallbacks),
        run(1, "swap:labor", 85.0),
    ]
    report = ev.autonomous_report(results, ["h12", "takeover:3", "swap:labor"])
    assert report["actor_vs_h12"]["takeover:3"]["ratios"]["welfare"] == pytest.approx(90 / 80)
    assert report["actor_vs_h12"]["swap:labor"]["ratios"]["welfare"] == pytest.approx(85 / 80)
    report.update({"worlds": [1], "turns": 1, "modes": ["h12", "takeover:3", "swap:labor"]})
    text = ev.summarize_autonomous(report)
    assert "h12 W=" in text
    assert "swap:labor/h12 welfare=1.062" in text
