import copy

import pytest
import torch
from test_autonomous_readiness_v7 import batch, config

from anthropocene_rl.autonomous_v6.readiness import protected_calibration
from anthropocene_rl.autonomous_v6.ridge import cv_select, displacement, fit_ridge
from anthropocene_rl.training.autonomous_ppo_v7 import AutonomousTrainerV7


def test_ridge_value_equation_and_protected_populated_state():
    torch.set_num_threads(1)
    trainer = AutonomousTrainerV7(config(), 17, {}, width=16)
    sum(p.square().mean() for p in trainer.model.parameters() if p.requires_grad).backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    x = batch()
    target = torch.linspace(10, 70, 8)
    trainer.model.fit_value_scale(target)
    before = protected_calibration(trainer)
    anchors = copy.deepcopy(trainer.model.initial_normalized_critic.state_dict())
    report = fit_ridge(trainer, (x, target), torch.arange(8) % 4)
    assert protected_calibration(trainer) == before
    assert not report["validation_used_for_selection"]
    assert all(
        torch.equal(v, trainer.model.initial_normalized_critic.state_dict()[k])
        for k, v in anchors.items()
    )
    assert all(p not in trainer.optimizer.state for p in trainer.model.critic.parameters())
    assert all(p in trainer.optimizer.state for p in trainer.model.actor.parameters())
    with torch.no_grad():
        phi = trainer.model.initial_normalized_critic[0](x["x"])
        ref = trainer.model.reference_critic(x["x"]).squeeze(-1)
        w, b = displacement(
            phi,
            ref,
            target,
            report["selected_strength"],
            trainer.model.value_offset,
            trainer.model.value_scale,
        )
        expected = trainer.model.value_offset.double() + trainer.model.value_scale.double() * (
            ref.double() + phi.double() @ w + b
        )
        assert torch.allclose(trainer.model.value(x).double(), expected, atol=1e-4)


def test_world_cv_statistics_do_not_use_held_out_targets_and_ties_are_stable():
    phi = torch.zeros(8, 3)
    ref = torch.zeros(8)
    target = torch.arange(8).double()
    worlds = torch.arange(8) % 4
    strength, rows = cv_select(phi, ref, target, worlds)
    assert strength == 0.01
    for row in rows:
        for fold in row["folds"]:
            allowed = target[worlds != fold["world"]]
            assert fold["fitting_offset"] == float(allowed.mean())
            assert fold["fitting_scale"] == float(allowed.std(unbiased=False).clamp_min(1))
    with pytest.raises(ValueError):
        cv_select(phi, ref, target, torch.zeros(8))
