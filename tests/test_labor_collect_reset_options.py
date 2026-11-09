"""collect's native world options (players/size/movement), with a fake bridge.

The default (no option) path must stay the historical ``bridge.reset(seed)`` exactly; explicit
options must reach ``bridge.reset`` through ``LaborResetOptions.reset_kwargs`` and be recorded
in the rollout metadata. No native DLL; no sealed or reserved worlds.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
import torch
from test_labor_a_ppo_wiring import _default_models, _e2e_rows, _same, _trainer_state

from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7.labor_bc import LaborResetOptions

REPO = Path(__file__).resolve().parents[1]
BEFORE = "64bf8a5"  # WP-A1: collect without world options
TRAINING_WORLD = ["--players", "16", "--width", "128", "--height", "64", "--separation", "9"]
TRAINING_RESET = {"players": 16, "movement": True, "withheld": 0,
                  "width": 128, "height": 64, "separation": 9}


def _before():
    """labor_training as of 64bf8a5, imported under a private name."""
    modname = "anthropocene_rl.autonomous_v7._reset_before_labor_training"
    if modname in sys.modules:
        return sys.modules[modname]
    try:
        source = subprocess.run(
            ["git", "show", f"{BEFORE}:src/anthropocene_rl/autonomous_v7/labor_training.py"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("the 64bf8a5 source is not available from git")
    spec = importlib.util.spec_from_loader(modname, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "anthropocene_rl.autonomous_v7"
    sys.modules[modname] = module
    exec(compile(source, f"<{BEFORE}:labor_training.py>", "exec"), module.__dict__)
    return module


def _parse(*extra):
    return training.argument_parser().parse_args(
        ["collect", "--checkpoint", "c.pt", "--worlds", "50000", "--out", "r.pt", *extra]
    )


def test_cli_options_default_explicit_and_refusals():
    assert training._collect_reset_options(_parse()) is None
    options = training._collect_reset_options(_parse(*TRAINING_WORLD, "--movement"))
    assert options == LaborResetOptions(players=16, movement=True, width=128, height=64,
                                        separation=9)
    assert options.state_dict() == TRAINING_RESET
    assert options.reset_kwargs() == {"players": 16, "movement": True, "withheld": 0,
                                      "size": (128, 64, 9)}
    fixture = training._collect_reset_options(_parse("--no-movement"))
    assert fixture == LaborResetOptions(players=8, movement=False)
    # pretrain-bc defaults movement ON, collect historically OFF: never assumed silently.
    with pytest.raises(ValueError, match="explicit --movement"):
        training._collect_reset_options(_parse("--players", "16"))
    with pytest.raises(SystemExit):
        _parse("--movement", "--no-movement")
    assert training.LEGACY_COLLECT_RESET.reset_kwargs() == {
        "players": 8, "movement": False, "withheld": 0
    }


class _Bridge:
    calls: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        type(self).calls.append((args, kwargs))
        village = {"camp": 1, "society": 1, "parentSociety": 0, "observation": {}}
        return {"turn": 0, "villages": [village]}

    def close(self):
        pass


def _run_collect(module, **kwargs):
    _Bridge.calls = []

    def step(**kw):
        return kw["step"], [object()]

    with (
        patch.object(module, "_sample_world_step", side_effect=step),
        patch.object(module, "_bootstrap_values", return_value={}),
    ):
        module.collect_labor_rollout(
            actor=None, task_critic=None, cost_critics=None, normalizer=None,
            world_seeds=[50000, 50001], turns=2, generator=torch.Generator(),
            bridge_factory=_Bridge, **kwargs,
        )
    return list(_Bridge.calls)


def test_options_reach_bridge_reset_and_default_call_is_unchanged():
    before = _run_collect(_before())
    default = _run_collect(training)
    assert default == before == [((50000,), {}), ((50001,), {})]
    options = LaborResetOptions(players=16, movement=True, width=128, height=64, separation=9)
    explicit = _run_collect(training, reset_options=options)
    kwargs = {"players": 16, "movement": True, "withheld": 0, "size": (128, 64, 9)}
    assert explicit == [((50000,), kwargs), ((50001,), kwargs)]


def _collect_cli(module, monkeypatch, tmp_path, tag, extra=()):
    actor, task, costs, normalizer, layout = _default_models()
    state = _trainer_state(actor, task, costs, normalizer, layout)
    checkpoint = tmp_path / "checkpoint.pt"
    if not checkpoint.exists():
        torch.save(state, checkpoint)
    rows, bootstrap = _e2e_rows(monkeypatch, (actor, task, costs))
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return rows, bootstrap

    monkeypatch.setattr(module, "collect_labor_rollout", fake)
    out = tmp_path / f"{tag}.pt"
    module.main(["collect", "--checkpoint", str(checkpoint), "--worlds", "50000,50001,50002",
                 "--turns", "2", "--out", str(out), *extra])
    return torch.load(out, weights_only=True), seen


def test_collect_records_options_and_default_rollout_is_unchanged(monkeypatch, tmp_path):
    old, old_seen = _collect_cli(_before(), monkeypatch, tmp_path, "before")
    new, new_seen = _collect_cli(training, monkeypatch, tmp_path, "default")
    assert "reset_options" not in old_seen and new_seen["reset_options"] is None
    meta = new["metadata"]
    assert meta.pop("resetOptions") == training.LEGACY_COLLECT_RESET.state_dict()
    assert meta.pop("resetOptionsSource") == "legacy-default-bridge.reset(seed)"
    _same(old, new)  # everything else, batch and targets included, is bitwise the same

    explicit, seen = _collect_cli(training, monkeypatch, tmp_path, "explicit",
                                  (*TRAINING_WORLD, "--movement"))
    assert seen["reset_options"].state_dict() == TRAINING_RESET
    assert explicit["metadata"]["resetOptions"] == TRAINING_RESET
    assert explicit["metadata"]["resetOptionsSource"] == "explicit"
    for key in ("batch", "task_targets", "cost_targets"):
        _same(explicit[key], old[key])


def test_train_side_logging_handles_old_and_new_rollouts(capsys):
    old = training._log_rollout_reset({}, "labor train")
    assert old["rolloutResetOptions"] == "not recorded (older rollout)"
    new = training._log_rollout_reset(
        {"resetOptions": TRAINING_RESET, "resetOptionsSource": "explicit"}, "labor train"
    )
    assert new["rolloutResetOptions"] == TRAINING_RESET
    assert '"rolloutResetOptions"' in capsys.readouterr().err
