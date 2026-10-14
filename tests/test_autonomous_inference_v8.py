import copy

import numpy as np
import pytest
import torch
from test_autonomous_direct_control_v6 import v6_wire
from test_autonomous_reward_objectives_v8 import cfg

from anthropocene_rl.policies.autonomous_inference_v8 import AutonomousElderPolicyV8
from anthropocene_rl.training.autonomous_ppo_v8 import AutonomousTrainerV8


def same_actions(left, right):
    assert left.keys() == right.keys()
    for camp in left:
        assert left[camp].keys() == right[camp].keys()
        for key in left[camp]:
            if isinstance(left[camp][key], np.ndarray):
                assert np.array_equal(left[camp][key], right[camp][key])
            else:
                assert left[camp][key] == right[camp][key]


def test_sorted_wego_context_sidecar_resumes_sampler_and_preserves_reset_p0(tmp_path):
    torch.set_num_threads(1)
    trainer = AutonomousTrainerV8(cfg("superlinear"), 19, {"test": "sidecar"}, width=16)
    path = tmp_path / "model.pt"
    trainer.checkpoint(path)
    policy = AutonomousElderPolicyV8(path, expected_provenance=trainer.provenance, seed=701)
    other = AutonomousElderPolicyV8(path, expected_provenance=trainer.provenance, seed=701)
    a, b = v6_wire(), v6_wire()
    a["Context"][0], b["Context"][0] = 100, 400
    opening = {"11": a, "2": b}
    with pytest.raises(ValueError):
        policy.act_many(opening)
    policy.start_episode(opening)
    other.start_episode({2: b, 11: a})
    same_actions(policy.act_many(opening), other.act_many({2: b, 11: a}))
    sidecar = policy.capture_sidecar()
    a["Context"][0], b["Context"][0] = 25, 300
    expected = policy.act_many(opening)
    restored = AutonomousElderPolicyV8(path, expected_provenance=trainer.provenance, seed=1)
    restored.restore_sidecar(sidecar)
    same_actions(expected, restored.act_many(opening))
    assert restored.episode_states[11].initial_population == 100
    assert restored.episode_states[2].initial_population == 400
    assert torch.equal(policy.generator.get_state(), restored.generator.get_state())
    invalid = copy.deepcopy(sidecar)
    invalid["episode_states"]["2"]["initial_population"] = 0
    with pytest.raises(ValueError):
        restored.restore_sidecar(invalid)
    invalid = dict(sidecar, checkpoint_sha256="changed")
    with pytest.raises(ValueError):
        restored.restore_sidecar(invalid)
    for states in [
        {"-1": sidecar["episode_states"]["2"]},
        {"0": sidecar["episode_states"]["2"], "00": sidecar["episode_states"]["11"]},
    ]:
        with pytest.raises(ValueError):
            restored.restore_sidecar(dict(sidecar, episode_states=states))
    with pytest.raises(ValueError):
        policy.act_many({11: a, "11": b})


def test_inference_context_requires_exact_own_population_domain(tmp_path):
    trainer = AutonomousTrainerV8(cfg("original"), 19, {}, width=16)
    path = tmp_path / "model.pt"
    trainer.checkpoint(path)
    policy = AutonomousElderPolicyV8(path, expected_provenance={}, seed=3)
    for amount in [0, 2.5, 2**24 + 2]:
        wire = v6_wire()
        wire["Context"][0] = amount
        with pytest.raises(ValueError):
            policy.start_episode({0: wire})
