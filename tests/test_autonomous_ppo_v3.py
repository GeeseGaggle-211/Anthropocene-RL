"""Architecture conversion must retain old weights/moments and collect fresh likelihoods."""

import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn

from anthropocene_rl.autonomous_v4.training import Trainer
from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3


def probe_module():
    spec = importlib.util.spec_from_file_location(
        "probe", "scripts/probe_autonomous_state_baseline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_layernorm_conversion_preserves_every_old_adam_state_and_roundtrips(tmp_path):
    from test_autonomous_v4 import batch

    torch.set_num_threads(1)
    obs = batch(32)
    config = json.loads(Path("configs/training/autonomous-ppo-gate-v1.json").read_text())
    old = Trainer(config, 7, {}, width=16)
    with torch.no_grad():
        action, logp, _, _ = old.model(obs, generator=old.sampler)
    rows = [
        dict(
            obs={k: v[i : i + 1] for k, v in obs.items()},
            action=action.index(slice(i, i + 1)),
            logp=logp[i : i + 1],
            advantage=float(i % 2 * 2 - 1),
            **{"return": float(i)},
        )
        for i in range(32)
    ]
    old.update(rows)
    path = tmp_path / "old.pt"
    old.checkpoint(path)
    config = json.loads(Path("configs/training/autonomous-ppo-layernorm-v3.json").read_text())
    new = AutonomousTrainerV3(config, 11, {}, width=16)
    new.warm_start(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert sum(isinstance(m, nn.LayerNorm) for m in new.model.actor) == 2
    assert not any(isinstance(m, nn.LayerNorm) for m in new.model.critic)
    parameters = dict(new.model.named_parameters())
    for name, p in old.model.named_parameters():
        assert torch.equal(p, parameters[name])
        for k, v in old.optimizer.state[p].items():
            assert torch.equal(v, new.optimizer.state[parameters[name]][k])
    assert all(p not in new.optimizer.state for n, p in parameters.items() if ".norm" in n)
    assert not torch.equal(old.model(obs, action)[1], new.model(obs, action)[1])
    with torch.no_grad():
        fresh, lp, _, _ = new.model(obs, generator=new.sampler)
    for i, row in enumerate(rows):
        row.update(action=fresh.index(slice(i, i + 1)), logp=lp[i : i + 1])
    assert torch.equal(lp, new.model(obs, fresh)[1])
    new.update(rows)
    path = tmp_path / "new.pt"
    new.checkpoint(path)
    resumed = AutonomousTrainerV3(config, 19, {}, width=16)
    resumed.restore(path)
    assert new.update(rows) == resumed.update(rows)
    assert all(
        torch.equal(v, resumed.model.state_dict()[k]) for k, v in new.model.state_dict().items()
    )
    from anthropocene_rl.policies.autonomous_inference_v3 import AutonomousElderPolicyV3

    policy = AutonomousElderPolicyV3(path, expected_provenance={}, seed=5)
    resumed.restore(path)
    assert torch.equal(policy.model(obs, fresh)[1], resumed.model(obs, fresh)[1])


def test_exact_state_baseline_matches_enumerated_conditional_reward():
    from test_autonomous_v4 import batch

    from anthropocene_rl.policies.autonomous_ppo_v3 import AutonomousPPOV3

    module = probe_module()
    obs = batch(2)
    obs["GoodMask"][:, 3:] = False
    model = AutonomousPPOV3(16)
    choices = torch.tensor([0, 1])
    labels = torch.tensor([-1.0, 1.0])
    mask = obs["GoodMask"].clone()
    mask[:, 2] = False
    actual = module.expected_reward(model, obs, "want", mask, choices, labels)
    enabled = module.probabilities(model, obs, "enabled", torch.ones(2, 2, dtype=torch.bool))[:, 1]
    give = module.probabilities(model, obs, "give", obs["GoodMask"])
    wanted = module.probabilities(model, obs, "want", mask)
    expected = labels * enabled * give[:, 2] * (wanted[:, 0] - wanted[:, 1])
    assert torch.allclose(actual, expected)
    assert not actual.requires_grad
