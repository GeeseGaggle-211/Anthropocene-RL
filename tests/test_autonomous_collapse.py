"""Collapse detector controls and schedule-independent native collection."""

import copy
import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch

from anthropocene_rl.autonomous_v4.collection import Collector, DeadlineTransport, episode_seed
from anthropocene_rl.autonomous_v4.diagnostics import (
    full_recovery_probe,
    health,
    optimizer_stall,
    recovery_probe,
)
from anthropocene_rl.autonomous_v4.training import Trainer, state_hash


def _obs():
    from test_autonomous_v4 import batch

    torch.manual_seed(11)
    return batch(32)


def test_collapsed_network_and_zero_lr_fail_recovery_without_mutating_model():
    from anthropocene_rl.autonomous_v4.policy import AutonomousPPO

    torch.set_num_threads(1)
    model = AutonomousPPO(32)
    obs = _obs()
    before = state_hash(model)
    assert not recovery_probe(model, obs, lr=0)["actor_pass"]
    assert not recovery_probe(model, obs, lr=0)["critic_pass"]
    assert state_hash(model) == before
    dead = copy.deepcopy(model)
    with torch.no_grad():
        for trunk in [dead.actor, dead.critic[0]]:
            for layer in trunk:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.zero_()
                    layer.bias.fill_(100)
    report = recovery_probe(dead, obs)
    assert not report["actor_pass"] and not report["critic_pass"]
    stats = health(dead, obs)
    assert all(r["saturated_fraction"] == 1 for r in stats["actor_layers"] + stats["critic_layers"])


def test_native_ipc_deadline_kills_unresponsive_process():
    start = time.monotonic()
    with DeadlineTransport(
        [sys.executable, "-c", "import time;time.sleep(10)"],
        deadline=start + 2,
        request_timeout=0.1,
    ) as transport:
        with pytest.raises(TimeoutError):
            transport.schema()
        assert transport._process.poll() is not None
    assert time.monotonic() - start < 2


def test_partial_recipe_and_control_collapse_cannot_hide_behind_healthy_workforce():
    from anthropocene_rl.autonomous_v4.policy import AutonomousPPO

    torch.set_num_threads(1)
    obs = _obs()
    model = AutonomousPPO(32)
    assert full_recovery_probe(model, obs, critic_lr=0.0003)["passed"]
    with torch.no_grad():
        model.heads["simplex"].weight[13:].zero_()
        model.heads["simplex"].bias[13:].fill_(100)
    report = full_recovery_probe(model, obs, critic_lr=0.0003)
    assert report["actor_pass"] and report["critic_pass"]
    assert all(r["passed"] for r in report["categorical"].values())
    assert not report["passed"]
    for name in ["activity_0", "activity_4", "control_0", "control_1", "control_2"]:
        assert report["conditional_simplex"][name]["covered"]
        assert not report["conditional_simplex"][name]["passed"]


def test_retained_critic_optimizer_stall_requires_residual_and_gradient():
    healthy = dict(
        advantage_std=1,
        actor_first_layer_grad=1,
        actor_step_l2=0.01,
        critic_residual_mse=1,
        critic_first_layer_grad=1,
        critic_step_l2=0.01,
    )
    assert not optimizer_stall([healthy] * 8)
    stalled = dict(healthy, critic_step_l2=0)
    assert not optimizer_stall([stalled] * 7)
    assert optimizer_stall([stalled] * 8)
    assert not optimizer_stall([dict(stalled, critic_residual_mse=0)] * 8)
    assert not optimizer_stall([dict(stalled, critic_first_layer_grad=0)] * 8)


def test_zero_lr_retained_critic_is_detected_on_real_ppo_updates():
    # Import the campaign instrumentation without running its CLI.
    import importlib.util

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "canary_measurement", scripts / "run_autonomous_collapse_canary.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    config = json.loads(Path("configs/training/autonomous-ppo-gate-v1.json").read_text())
    config.update(epochs=1, minibatch=32)
    trainer = Trainer(config, 31, {}, width=32)
    obs = _obs()

    def update():
        with torch.no_grad():
            action, lp, _, value = trainer.model(obs, generator=trainer.sampler)
        rows = [
            dict(
                obs={k: v[i : i + 1] for k, v in obs.items()},
                action=action.index(slice(i, i + 1)),
                logp=lp[i : i + 1],
                advantage=float(i % 2 * 2 - 1),
                **{"return": float(value[i]) + 2},
            )
            for i in range(32)
        ]
        return module.measured_update(trainer, rows)

    assert update()["critic_step_l2"] > 0  # Populate Adam moments first.
    trainer.optimizer.param_groups[1]["lr"] = 0
    measurements = [update() for _ in range(8)]
    assert all(r["critic_first_layer_grad"] > 0 for r in measurements)
    assert optimizer_stall(measurements)


def test_episode_keys_are_reproducible_and_distinct():
    assert episode_seed(17, 41) == episode_seed(17, 41)
    assert (
        len({episode_seed(seed, episode) for seed in [17, 29, 43] for episode in range(50)}) == 150
    )


@pytest.mark.skipif(
    not os.environ.get("AUTONOMOUS_V4_BRIDGE"), reason="native v4 integration opt-in"
)
def test_one_four_workers_and_nonempty_optimizer_checkpoint(tmp_path):
    torch.set_num_threads(1)
    config = json.loads(Path("configs/training/autonomous-ppo-gate-v1.json").read_text())
    bank = Path(os.environ["AUTONOMOUS_V4_BANK"])
    entries = json.loads((bank / "manifest.json").read_text())["entries"][:12]
    command = json.loads(os.environ["AUTONOMOUS_V4_BRIDGE"])
    a = Trainer(config, 17, {}, width=32)
    b = Trainer(config, 17, {}, width=32)
    for phase in range(2):
        with Collector(command, workers=1, deadline=time.monotonic() + 120) as collector:
            rows_a, summaries_a = collector.rollout(a, bank, entries, 17, 64, verify=True)
        with Collector(command, workers=4, deadline=time.monotonic() + 120) as collector:
            rows_b, summaries_b = collector.rollout(b, bank, entries, 17, 64, verify=True)
        assert [s["fingerprint"] for s in summaries_a] == [s["fingerprint"] for s in summaries_b]
        for x, y in zip(rows_a, rows_b, strict=True):
            assert torch.equal(x["logp"], y["logp"])
            for key in ["workforce", "activities", "controls", "movement", "trade"]:
                assert torch.equal(getattr(x["action"], key), getattr(y["action"], key))
        assert a.update(rows_a) == b.update(rows_b)
        assert state_hash(a.model) == state_hash(b.model)
        if phase == 0:
            a.checkpoint(tmp_path / "populated.pt")
            b = Trainer(config, 29, {}, width=32)
            b.restore(tmp_path / "populated.pt")
        else:
            for key, value in a.optimizer.state_dict()["state"].items():
                for field, item in value.items():
                    other = b.optimizer.state_dict()["state"][key][field]
                    assert (
                        torch.equal(item, other)
                        if isinstance(item, torch.Tensor)
                        else item == other
                    )
