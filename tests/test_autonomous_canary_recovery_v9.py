import json
from pathlib import Path

import pytest
import torch
from test_autonomous_readiness_v7 import batch, config

from anthropocene_rl.autonomous_v6.canary_recovery import persist_rollout, validate_probe
from anthropocene_rl.autonomous_v6.contracts import observation
from anthropocene_rl.autonomous_v6.policy import inputs_with_mode
from anthropocene_rl.training.autonomous_ppo_v7 import AutonomousTrainerV7


def test_real_fitting_probe_passes_existing_guard_and_small_or_constant_probes_fail():
    p = (
        Path(__file__).parents[1]
        / "evidence/autonomous-opportunity-v9/fitting-adaptation-probe-v9.6.json"
    )
    x = inputs_with_mode([observation(w) for w in json.loads(p.read_text())])
    assert validate_probe(x)["rows"] >= 16
    with pytest.raises(ValueError):
        validate_probe(batch())
    constant = {k: torch.cat([v[:1]] * 16) for k, v in batch().items()}
    with pytest.raises(ValueError):
        validate_probe(constant)


def test_completed_rollout_is_restorable_before_a_diagnostic_exception(tmp_path):
    tr = AutonomousTrainerV7(config(), 41, {}, width=16)
    tr.transitions, tr.episodes, tr.updates = 48, 2, 1
    cell = dict(id="test", transitions=48)
    path = persist_rollout(
        tr, tmp_path, cell, [dict(trace=[], final_obs=dict(x=torch.ones(2)), steps=48)]
    )
    with pytest.raises(ValueError):
        validate_probe(batch())
    restored = AutonomousTrainerV7(config(), 1, {}, width=16)
    restored.restore(path)
    assert restored.transitions == 48 and restored.episodes == 2 and restored.updates == 1
    assert torch.equal(tr.sampler.get_state(), restored.sampler.get_state())
    assert cell["durable_transitions"] == 48
    assert "final_obs" not in json.loads((tmp_path / "test-training-traces.json").read_text())[0]
