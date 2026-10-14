from pathlib import Path

import torch
from test_autonomous_readiness_v7 import batch
from test_autonomous_reward_objectives_v8 import cfg

from anthropocene_rl.autonomous_v6.objective_collection import context
from anthropocene_rl.autonomous_v6.readiness import protected_calibration
from anthropocene_rl.autonomous_v6.reward_objectives import EpisodeObjectiveState, Objective
from anthropocene_rl.training.autonomous_ppo_v8 import AutonomousTrainerV8


def test_fixed_train_dose_ignores_validation_and_preserves_populated_actor_state(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from repair_autonomous_current_critic_v11 import fixed_fit

    torch.set_num_threads(1)
    trainer = AutonomousTrainerV8(cfg("superlinear"), 53, {}, width=16)
    sum(p.square().mean() for p in trainer.model.parameters() if p.requires_grad).backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.episodes, trainer.transitions, trainer.updates = 5, 130, 2
    before = protected_calibration(trainer)
    x = context(batch(), EpisodeObjectiveState(100, Objective("superlinear")), 100)
    y = torch.linspace(10, 80, 8)
    critic = {k: v.clone() for k, v in trainer.model.critic.state_dict().items()}
    records = fixed_fit(trainer, (x, y), epochs=2, seed=79, deadline=float("inf"))
    assert len(records) == 2 and records[-1]["epoch"] == 2
    assert protected_calibration(trainer) == before
    assert any(not torch.equal(v, trainer.model.critic.state_dict()[k]) for k, v in critic.items())
    assert all(r["gradient"] > 0 for r in records)
