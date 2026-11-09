"""train --pilot-ungated-actor: the opt-in pilot bypass of a closed/absent actor gate.

Synthetic CPU fixtures (the WP-A1 e2e helpers: real collect/train commands, fake bridge
rows, synthetic anchor); no native DLL, no sealed (3000-3059) or reserved worlds.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_labor_a_ppo_wiring import (
    _default_models,
    _iteration_files,
    _same,
    _train_args,
    _trainer_state,
)

from anthropocene_rl.autonomous_v7 import labor_compose_checkpoint as compose
from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7.labor_pilot import (
    PILOT_BYPASS_GATE,
    PILOT_BYPASS_KEY,
    pilot_bypass_history,
    require_no_pilot_bypass,
)
from anthropocene_rl.autonomous_v7.provenance import checkpoint_hash

REPO = Path(__file__).resolve().parents[1]
PREPILOT = "68ae43d"  # feature/r5-ppo before the pilot bypass


def _prepilot_training():
    """labor_training as of PREPILOT, under a private module name."""
    modname = "anthropocene_rl.autonomous_v7._prepilot_labor_training"
    if modname in sys.modules:
        return sys.modules[modname]
    try:
        source = subprocess.run(
            ["git", "show", f"{PREPILOT}:src/anthropocene_rl/autonomous_v7/labor_training.py"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("pre-pilot source is not available from git")
    spec = importlib.util.spec_from_loader(modname, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "anthropocene_rl.autonomous_v7"
    sys.modules[modname] = module
    exec(compile(source, f"<{PREPILOT}:labor_training.py>", "exec"), module.__dict__)
    return module


@pytest.fixture()
def files(monkeypatch, tmp_path):
    actor, task, costs, normalizer, layout = _default_models()
    state0 = _trainer_state(actor, task, costs, normalizer, layout)
    checkpoint, rollout, anchor, gate = _iteration_files(monkeypatch, tmp_path, state0, "it0")
    report = json.loads(gate.read_text(encoding="utf-8"))
    closed = tmp_path / "it0-gate-closed.json"
    closed.write_text(json.dumps({
        **report,
        "actorGateOpen": False,
        "heads": {**report["heads"], "feasibility": {"pass": False}},
    }), encoding="utf-8")
    return {"state0": state0, "checkpoint": checkpoint, "rollout": rollout, "anchor": anchor,
            "open": gate, "closed": closed, "tmp": tmp_path}


def _run(files, name, report, *extra, main=None):
    out = files["tmp"] / f"{name}.pt"
    args = _train_args(files["checkpoint"], files["rollout"], files["anchor"],
                       files[report] if report else None, out, *extra)
    if report is None:  # drop "--calibration-report None"
        index = args.index("--calibration-report")
        del args[index:index + 2]
    (main or training.main)(args)
    return torch.load(out, weights_only=True)


@pytest.mark.parametrize("report", ["open", "closed", None])
def test_default_train_is_bitwise_the_prepilot_train(files, monkeypatch, report):
    old = _prepilot_training()
    monkeypatch.setattr(old, "load_labor_anchor", training.load_labor_anchor)
    before = _run(files, f"old-{report}", report, main=old.main)
    after = _run(files, f"new-{report}", report)
    _same(before, after)
    assert PILOT_BYPASS_KEY not in after and "pilotBypass" not in after["last_update"]
    assert (after["last_update"]["actor"].get("acceptedSteps", 0) > 0) == (report == "open")


def test_bypass_with_closed_report_updates_actor_like_an_open_gate(files):
    closed = _run(files, "closed", "closed")
    opened = _run(files, "open", "open")
    bypass = _run(files, "bypass", "closed", "--pilot-ungated-actor")
    # The critic step is unchanged by the flag; the actor step is exactly the open-gate step.
    for key in ("task_critic", "cost_critics", "critic_optimizer"):
        _same(bypass[key], closed[key], key)
    for key in ("actor", "actor_optimizer", "shuffle_rng_state"):
        _same(bypass[key], opened[key], key)
    assert not all(torch.equal(v, closed["actor"][k]) for k, v in bypass["actor"].items())
    update = bypass["last_update"]
    assert update["actorGate"] == PILOT_BYPASS_GATE
    assert update["actor"]["acceptedSteps"] == opened["last_update"]["actor"]["acceptedSteps"] > 0
    assert update["actor"]["actorGate"]["mode"] == PILOT_BYPASS_GATE
    status = update["pilotBypass"]["calibrationReport"]
    assert update["pilotBypass"]["used"] is True and status["supplied"] is True
    assert status["actorGateOpen"] is False and status["headPass"]["feasibility"] is False
    assert status["sha256"] == checkpoint_hash(files["closed"])
    (entry,) = bypass[PILOT_BYPASS_KEY]
    assert entry["iteration"] == 0 and entry["actorGate"] == PILOT_BYPASS_GATE
    assert entry["rolloutSha256"] == checkpoint_hash(files["rollout"])
    assert entry["inputCheckpointSha256"] == checkpoint_hash(files["checkpoint"])
    assert entry["acceptedSteps"] == update["actor"]["acceptedSteps"]
    with pytest.raises(ValueError, match="pilot-ungated-actor"):
        require_no_pilot_bypass(bypass, "test")


def test_bypass_without_report_binds_the_rollout(files):
    bypass = _run(files, "absent", None, "--pilot-ungated-actor")
    opened = _run(files, "open", "open")
    _same(bypass["actor"], opened["actor"])
    update = bypass["last_update"]
    assert update["pilotBypass"]["calibrationReport"] == {"supplied": False}
    assert update["actor"]["actorGate"]["dataSha256"] == checkpoint_hash(files["rollout"])
    assert len(bypass[PILOT_BYPASS_KEY]) == 1


def test_flag_with_an_open_report_uses_the_real_gate(files):
    flagged = _run(files, "open-flag", "open", "--pilot-ungated-actor")
    opened = _run(files, "open", "open")
    for key in ("actor", "task_critic", "cost_critics", "actor_optimizer", "critic_optimizer"):
        _same(flagged[key], opened[key], key)
    assert flagged["last_update"]["actorGate"] == "calibration-report"
    assert flagged["last_update"]["pilotBypass"]["used"] is False
    assert PILOT_BYPASS_KEY not in flagged
    require_no_pilot_bypass(flagged, "test")


def test_the_record_persists_through_later_updates_and_composition(files, monkeypatch):
    bypass = _run(files, "bypass", "closed", "--pilot-ungated-actor")
    history = pilot_bypass_history(bypass)
    tmp = files["tmp"]
    # A later, ordinary (no flag, closed gate) update carries the record unchanged.
    checkpoint1, rollout1, anchor1, gate1 = _iteration_files(monkeypatch, tmp, bypass, "it1")
    out = tmp / "it2.pt"
    training.main(_train_args(checkpoint1, rollout1, anchor1, gate1, out))
    later = torch.load(out, weights_only=True)
    assert pilot_bypass_history(later) == history
    # A second bypass appends.
    closed = json.loads(gate1.read_text(encoding="utf-8"))
    closed["actorGateOpen"] = False
    gate1.write_text(json.dumps(closed), encoding="utf-8")
    out = tmp / "it2-bypass.pt"
    training.main(_train_args(checkpoint1, rollout1, anchor1, gate1, out, "--pilot-ungated-actor"))
    again = pilot_bypass_history(torch.load(out, weights_only=True))
    assert again[:1] == history and len(again) == 2 and again[1]["iteration"] == 1
    # Composing a clean critic with the pilot actor (or the reverse) keeps the record.
    for base, source in ((files["state0"], bypass), (bypass, files["state0"])):
        composed = compose.compose_trainer_checkpoint(base, source)
        assert pilot_bypass_history(composed) == history
        report = compose.verify_composed(composed, base, source)
        assert report["pilotBypassUpdates"] == 1
    clean = compose.compose_trainer_checkpoint(files["state0"], files["state0"])
    assert PILOT_BYPASS_KEY not in clean
    assert "pilotBypassUpdates" not in compose.verify_composed(
        clean, files["state0"], files["state0"]
    )


def test_bypass_refusals(files, tmp_path):
    with pytest.raises(ValueError, match="plain-GAE pilot flag"):
        _run(files, "gae", "closed", "--pilot-ungated-actor", "--a-mode", "gae",
             "--a-warm-start", str(tmp_path / "w.pt"), "--a-critic", str(tmp_path / "c.pt"))
    args = training.argument_parser().parse_args(
        ["train", "--checkpoint", "c", "--rollout", "r", "--anchor", "a", "--out", "o"]
    )
    assert args.pilot_ungated_actor is False
    with pytest.raises(ValueError, match="malformed"):
        pilot_bypass_history({PILOT_BYPASS_KEY: []})


def test_acceptance_evaluate_refuses_a_pilot_checkpoint(files, monkeypatch):
    from anthropocene_rl.autonomous_v7 import labor_pretrain_eval
    from anthropocene_rl.autonomous_v7 import labor_r5_acceptance as acceptance

    bypass = _run(files, "bypass", "closed", "--pilot-ungated-actor")
    models = training._models_from_checkpoint(bypass)
    monkeypatch.setattr(
        labor_pretrain_eval, "_load_models", lambda path: (bypass, *models[1:])
    )
    monkeypatch.setattr(
        labor_pretrain_eval, "run_critic_calibration",
        lambda *a, **k: pytest.fail("calibration ran on a pilot checkpoint"),
    )
    with pytest.raises(ValueError, match="never readiness or acceptance"):
        acceptance.evaluate_fit(
            arm="a", seed=0, checkpoint=files["tmp"] / "bypass.pt", advantage=None,
            calibration_data=files["tmp"] / "none", fit_data=None, reserved_payloads=None,
        )
