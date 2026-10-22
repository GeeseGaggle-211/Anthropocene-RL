"""Evaluating with movement on (docs/MODEL9_TOKENS.md, "Evaluating with movement on"): the default bridge movement
setting of every evaluation path, the shared setting of policy and reference, the policy's movement head under it,
and the train/evaluation input-shift check. No native bridge required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts, evaluate, input_shift, paired_compare, stage_eval
from anthropocene_rl.autonomous_v7.model import Actor, Normalizer, to_tensors
from test_autonomous_v7 import batch


def save(path: Path, **state) -> str:
    torch.save(state, path)
    return str(path)


def test_trained_movement_follows_the_training_regime_of_contract12_checkpoints() -> None:
    tm = evaluate.trained_movement
    assert tm({"token": True}) is True                                   # stage-1 clone: teacher shards, movement on
    assert tm({}) is None and tm({"token": False, "stage": "mixed"}) is None  # pre-contract-12: no default
    assert tm({"token": True, "stage": "capability"}) is False           # PPO with native movement, bridge off
    assert tm({"token": True, "stage": "mixed", "region_movement": True}) is True   # learned movement
    assert tm({"token": True, "stage": "mixed", "observe_movement": True}) is True  # observed movement
    assert tm({"token": True, "stage": "movement"}) is True


def test_evaluation_movement_defaults_flags_and_conflicts(tmp_path: Path) -> None:
    clone = save(tmp_path / "clone.pt", token=True)
    legacy = save(tmp_path / "legacy.pt", contract=contracts.NATIVE_CONTRACT)
    native_ppo = save(tmp_path / "ppo.pt", token=True, stage="capability")
    em = evaluate.evaluation_movement
    assert em(None, [clone]) is True
    assert em(None, [legacy]) is False and em(None, [None]) is False
    assert em(None, [legacy, clone]) is True            # the legacy init joins the clone's regime (one setting)
    assert em(None, [native_ppo]) is False
    assert em(False, [clone]) is False and em(True, [legacy]) is True   # --no-movement / --movement
    with pytest.raises(SystemExit):
        em(None, [clone, native_ppo])                    # disagreeing regimes need an explicit flag
    assert em(True, [clone, native_ppo]) is True


def test_stage_eval_runs_the_teacher_under_the_policies_movement_setting(tmp_path: Path, monkeypatch) -> None:
    clone = save(tmp_path / "clone.pt", token=True)
    legacy = save(tmp_path / "legacy.pt", contract=contracts.NATIVE_CONTRACT)
    calls = []

    def fake(checkpoint, worlds, turns, workers=6, movement=False, region=None, grouped=False, **_):
        calls.append((checkpoint, movement))
        return {"welfare": 1.0, "population": 1.0, "qol": 1.0, "specialist_share": 0.0, "below_30": 0.0,
                "extinctions": 0.0, "worlds": [[w[0], w[1], 1.0, 0] for w in worlds]}

    monkeypatch.setattr(stage_eval, "evaluate_closed_loop", fake)
    for extra, expected in (([], True), (["--no-movement"], False)):
        calls.clear()
        out = tmp_path / f"eval{len(extra)}.json"
        monkeypatch.setattr(sys, "argv", ["stage_eval", "--init", legacy, "--candidates", clone, "--default-seeds", "2000",
                                          "--fixture-seeds", "2000", "--out", str(out), *extra])
        assert stage_eval.main() == 0
        assert calls and {m for _, m in calls} == {expected}
        assert any(c is None for c, _ in calls)          # the teacher (reference) is among them
        report = json.loads(out.read_text())
        assert all(row["movement"] is expected for name, row in report.items() if name != "gate")


def test_merging_reports_of_different_movement_settings_is_refused() -> None:
    teacher = {"worlds": [[2000, 0, 1.0, 0]]}
    a = {"default": {"teacher": teacher, "candidates": {"x": {}}, "movement": True}, "gate": {}}
    b = {"default": {"teacher": teacher, "candidates": {"y": {}}, "movement": False}, "gate": {}}
    with pytest.raises(ValueError, match="movement"):
        paired_compare.merge([a, b])
    assert set(paired_compare.merge([a, {**b, "default": {**b["default"], "movement": True}}])["default"]["candidates"]) == {"x", "y"}


def test_policy_moves_through_its_cloned_head_only_with_movement_on(monkeypatch) -> None:
    b = batch()
    normalizer = Normalizer.fit(b)
    torch.manual_seed(0)
    actor = Actor(b["core"].shape[1])
    with torch.no_grad():
        actor.stay.bias.fill_(-50.0)                     # the cloned head prefers a site row over staying
    t = to_tensors(b, normalizer)
    t["movement_mask"] = torch.ones_like(t["movement_mask"])
    monkeypatch.setattr(evaluate, "observation_batch", lambda live: None)
    monkeypatch.setattr(evaluate, "to_tensors", lambda batch, normalizer: t)
    villages = [{"camp": i, "observation": {}} for i in range(len(t["core"]))]
    on = evaluate.policy_actions(actor, None, villages, movement=True)
    off = evaluate.policy_actions(actor, None, villages, movement=False)
    assert all(0 <= a["movement"] < contracts.MOVEMENT_SLOTS - 1 for a in on.values())
    assert all(a["movement"] == contracts.NATIVE_MOVEMENT for a in off.values())


def test_observe_movement_arms_stay_native_through_the_region_head(tmp_path: Path) -> None:
    b = batch()
    normalizer = Normalizer.fit(b)
    actor = Actor(b["core"].shape[1])
    common = {"actor": actor.state_dict(), "normalizer": normalizer.state_dict(), "core_width": b["core"].shape[1],
              "contract": contracts.NATIVE_CONTRACT}
    assert evaluate.load(Path(save(tmp_path / "o.pt", **common, observe_movement=True)))[0].region_movement is True
    assert evaluate.load(Path(save(tmp_path / "n.pt", **common)))[0].region_movement is False


def test_input_shift_flags_a_constant_training_input_that_moves_at_evaluation() -> None:
    train = batch(40, 1)
    train["core"][:, 3] = 1.0                            # the v2-480 kind: constant in the teacher shards
    evaluated = {k: v.copy() for k, v in batch(12, 2).items()}
    moments = input_shift.moments(train)
    evaluated["core"][:, 3] = 1.0
    assert not any(r["dead_shifted"] for r in input_shift.shift(moments, evaluated).get("core", []))
    evaluated["core"][:, 3] = 0.0                        # the old evaluation regime
    rows = {r["column"]: r for r in input_shift.shift(moments, evaluated)["core"]}
    assert rows[3]["dead_shifted"] and rows[3]["dead_in_train"] and rows[3]["train_range"] == [1.0, 1.0]
    evaluated["core"][:, 3] = 1.0
    evaluated["core"][0, 5] = 1e12                       # far outside a varying input's range
    rows = {r["column"]: r for r in input_shift.shift(moments, evaluated)["core"]}
    assert 5 in rows and not rows[5]["dead_shifted"] and rows[5]["max_abs_z"] > 10
    names = [f"v2-{i}" for i in range(train["core"].shape[1])]
    named = input_shift.named_columns(moments, evaluated, ["v2-3"], names)
    assert named[0]["dead_in_train"] and named[0]["eval_range"] == [1.0, 1.0] and named[0]["max_abs_z"] < 1e-6
