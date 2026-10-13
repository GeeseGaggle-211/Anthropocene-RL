"""Retained-optimizer diagnostic uses real legal branches without mutating originals."""

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import torch

from anthropocene_rl.autonomous_v4.training import Trainer


def probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts/probe_autonomous_retained_ppo.py"
    spec = importlib.util.spec_from_file_location("retained_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_faces_use_legal_choices_and_frozen_feature_order():
    from test_autonomous_v4 import batch

    module = probe_module()
    obs = batch(40)
    obs["MovementMask"][:, 1] = False  # Native current-site candidate is masked.
    for r in [5, 9]:
        obs["ActivityMask"][:, r, :2] = True
    for task in module.TASKS:
        first = module.task_face(obs, task)
        second = module.task_face(obs, task)
        assert torch.equal(first[3], second[3]) and len(first[3]) == 32
        assert first[1].sum() >= 2
    assert torch.where(module.task_face(obs, "movement")[1])[0][:2].tolist() == [0, 2]


def test_disabled_quotes_do_not_fake_conditional_probe_exposure(tmp_path):
    from test_autonomous_v4 import batch

    torch.set_num_threads(1)
    module = probe_module()
    config = json.loads(Path("configs/training/autonomous-ppo-gate-v1.json").read_text())
    config.update(epochs=1, actor_lr=0, critic_lr=0)
    trainer = Trainer(config, 31, {}, width=16)
    with torch.no_grad():
        trainer.model.heads["enabled"].weight.zero_()
        trainer.model.heads["enabled"].bias.copy_(torch.tensor([100.0, -100.0]))
    path = tmp_path / "fixed.pt"
    trainer.checkpoint(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report = module.run_task(path, batch(32), "partner", updates=1, deadline=time.monotonic() + 30)
    assert report["initial_separation"] <= 0
    assert report["updates"][0]["active_draws"] == 0
    assert report["updates"][0]["reward_std"] == 0
    assert not report["covered"] and not report["passed"]
    assert report["initial_model"] == report["final_model"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    timed = module.run_task(path, batch(32), "workforce", deadline=time.monotonic() - 1)
    assert timed["status"] == "time_limit" and timed["draws"] == 0
    assert not timed["passed"] and timed["updates"] == []
    obs = batch(32)
    obs["MovementMask"][:] = False
    obs["MovementMask"][:, 0] = True
    uncovered = module.run_task(path, obs, "movement", deadline=time.monotonic() + 30)
    assert uncovered["status"] == "insufficient_face"
    assert not uncovered["covered"] and not uncovered["passed"]
