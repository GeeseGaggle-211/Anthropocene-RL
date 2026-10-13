"""Critic-only conversion preserves actual populated optimizer state and policy."""

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest
import torch

from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("compare_autonomous_critic_remedies_v1")
    yield module
    sys.modules.pop("compare_autonomous_critic_remedies_v1", None)


def test_critic_conversion_keeps_policy_and_every_populated_named_adam_moment(comparison):
    from test_autonomous_v4 import batch

    torch.set_num_threads(1)
    config = json.loads(
        Path("configs/training/autonomous-ppo-layernorm-v3-recovery.json").read_text()
    )
    trainer = AutonomousTrainerV3(config, 123, {}, width=16)
    obs = batch(32)
    with torch.no_grad():
        action, logp, _, _ = trainer.model(obs, generator=trainer.sampler)
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
    trainer.update(rows)
    old_parameters = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    old_moments = {
        n: copy.deepcopy(trainer.optimizer.state[p]) for n, p in trainer.model.named_parameters()
    }
    sampler, shuffle = trainer.sampler.get_state(), trainer.shuffle.get_state()
    with torch.no_grad():
        old_logp = trainer.model(obs, action)[1]
    comparison.convert(trainer, 0.0001, True)
    parameters = dict(trainer.model.named_parameters())
    for name, value in old_parameters.items():
        assert torch.equal(value, parameters[name])
        for key, moment in old_moments[name].items():
            assert torch.equal(moment, trainer.optimizer.state[parameters[name]][key])
    added = set(parameters) - set(old_parameters)
    assert added == {f"critic.0.norm{i}.{key}" for i in (0, 1) for key in ("weight", "bias")}
    assert all(parameters[n] not in trainer.optimizer.state for n in added)
    assert torch.equal(old_logp, trainer.model(obs, action)[1])
    assert torch.equal(sampler, trainer.sampler.get_state())
    assert torch.equal(shuffle, trainer.shuffle.get_state())
    assert trainer.optimizer.param_groups[0]["lr"] == config["actor_lr"]
    assert trainer.optimizer.param_groups[1]["lr"] == 0.0001

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model.value(obs).square().mean().backward()
    trainer.optimizer.step()
    for name in old_parameters:
        if name.startswith(("actor.", "heads.")):
            assert torch.equal(parameters[name], old_parameters[name])
            for key, moment in old_moments[name].items():
                assert torch.equal(moment, trainer.optimizer.state[parameters[name]][key])
    assert all(parameters[n] in trainer.optimizer.state for n in added)


def test_target_deduplication_preserves_order_labels_and_original_trace(comparison):
    task = dict(
        status="completed", indices=[1, 2], labels=[-1, 1], updates=[dict(critic_mse=0.5)] * 64
    )
    report = dict(
        tasks={
            "one": task,
            "same": copy.deepcopy(task),
            "opposite": {**task, "labels": [1, -1]},
            "reordered": {**task, "indices": [2, 1]},
        }
    )
    groups = comparison.unique_targets(report)
    assert [g["tasks"] for g in groups] == [["one", "same"], ["opposite"], ["reordered"]]
    report["tasks"]["same"]["updates"][0]["critic_mse"] = 0.6
    with pytest.raises(AssertionError):
        comparison.unique_targets(report)


def test_error_preserves_partial_measurements(comparison, tmp_path):
    output = tmp_path / "partial.json"
    result = {"targets": [{"mse_trace": [1, 0.5]}]}
    with pytest.raises(TimeoutError), comparison.preserve_evidence(result, output):
        raise TimeoutError("bounded")
    saved = json.loads(output.read_text())
    assert saved["targets"][0]["mse_trace"] == [1, 0.5]
    assert saved["complete"] is False and "TimeoutError" in saved["error"]
