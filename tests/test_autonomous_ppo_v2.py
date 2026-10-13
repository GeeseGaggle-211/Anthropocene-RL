"""Exact mixed likelihood, conditional entropy, and saturated-logit recovery gradients."""

import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import pytest
import torch

from anthropocene_rl.policies.autonomous_ppo_v2 import AutonomousPPOV2, choose, distribution
from anthropocene_rl.training.autonomous_ppo_v2 import AutonomousTrainerV2


def test_mixture_has_legal_support_exact_singletons_and_live_reverse_kl_gradient():
    logits = torch.tensor([[100.0, -100.0, 999.0], [1.0, 2.0, 3.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, True, False]])
    dist, penalty = distribution(logits, mask, 0.05)
    assert dist.probs[0, 1].item() == pytest.approx(0.025)
    assert dist.probs[0, 2] == 0
    assert dist.log_prob(torch.tensor([0, 1]))[1] == 0
    assert dist.entropy()[1] == penalty[1] == 0
    penalty.sum().backward()
    assert torch.allclose(logits.grad[0], torch.tensor([0.5, -0.5, 0.0]))
    generator = torch.Generator().manual_seed(9)
    before = generator.get_state()
    choose(logits[1:], mask[1:], None, uniform_mix=0.05, generator=generator, deterministic=False)
    assert torch.equal(before, generator.get_state())


def test_joint_entropy_and_gradients_match_exhaustive_trade_enumeration():
    from test_autonomous_v4 import batch

    torch.set_num_threads(1)
    obs = batch(1)
    obs["MovementMask"][:, 2:] = False
    obs["CounterpartyMask"][:, 2:] = False
    obs["GoodMask"][:, 3:] = False
    model = AutonomousPPOV2(16)
    with torch.no_grad():
        for head in model.heads.values():
            head.bias.add_(torch.linspace(-2, 2, len(head.bias)))
    h = model.actor(obs["x"])

    def d(name, mask):
        return distribution(model.heads[name](h), mask, 0.05)[0]

    move = d("movement", obs["MovementMask"])
    enable = d("enabled", torch.ones(1, 2, dtype=torch.bool))
    partner = d("partner", obs["CounterpartyMask"])
    give = d("give", obs["GoodMask"])
    lot = d("lot", torch.ones(1, 3, dtype=torch.bool))
    price = d("price", torch.ones(1, 3, dtype=torch.bool))
    logs = []
    for m in range(2):
        ml = move.log_prob(torch.tensor([m]))
        logs.append(ml + enable.log_prob(torch.tensor([0])))
        for p, g, size, r in itertools.product(range(2), range(3), range(3), range(3)):
            mask = obs["GoodMask"].clone()
            mask[:, g] = False
            want = d("want", mask)
            for w in range(3):
                if w == g:
                    continue
                logs.append(
                    ml
                    + enable.log_prob(torch.tensor([1]))
                    + partner.log_prob(torch.tensor([p]))
                    + give.log_prob(torch.tensor([g]))
                    + want.log_prob(torch.tensor([w]))
                    + lot.log_prob(torch.tensor([size]))
                    + price.log_prob(torch.tensor([r]))
                )
    logp = torch.cat(logs)
    assert float(logp.exp().sum().detach()) == pytest.approx(1, abs=1e-6)
    enumerated = -(logp.exp() * logp).sum()
    exact = model.categorical_terms(obs)[0].sum()
    assert float(exact.detach()) == pytest.approx(float(enumerated.detach()), abs=1e-6)
    parameters = [model.heads[k].bias for k in ["enabled", "give", "want"]]
    ga = torch.autograd.grad(exact, parameters, retain_graph=True)
    gb = torch.autograd.grad(enumerated, parameters)
    assert all(torch.allclose(a, b, atol=2e-6) for a, b in zip(ga, gb, strict=True))


def test_v2_full_scoring_gradient_and_versioned_checkpoint(tmp_path):
    from test_autonomous_v4 import batch

    torch.set_num_threads(1)
    obs = batch(32)
    config = json.loads(Path("configs/training/autonomous-ppo-gate-v1.json").read_text())
    config.update(
        uniform_mix=0.05,
        legal_reverse_kl_coef=0.01,
        critic_lr=0.0003,
        model_version="autonomous-ppo-model-v2",
    )
    trainer = AutonomousTrainerV2(config, 9, {}, width=32)
    with torch.no_grad():
        actions, old, _, _ = trainer.model(obs, generator=trainer.sampler)
    assert torch.equal(old, trainer.model(obs, actions)[1])
    rows = [
        dict(
            obs={k: v[i : i + 1] for k, v in obs.items()},
            action=actions.index(slice(i, i + 1)),
            logp=old[i : i + 1],
            advantage=float(i % 2 * 2 - 1),
            **{"return": float(i)},
        )
        for i in range(32)
    ]
    metrics = trainer.update(rows)
    assert all(torch.isfinite(torch.tensor(m["loss"])) for m in metrics)
    path = tmp_path / "v2.pt"
    trainer.checkpoint(path)
    restored = AutonomousTrainerV2(config, 99, {}, width=32)
    restored.restore(path)
    assert trainer.update(rows) == restored.update(rows)
    assert all(
        torch.equal(v, restored.model.state_dict()[k])
        for k, v in trainer.model.state_dict().items()
    )
    from anthropocene_rl.autonomous_v4.training import Trainer

    with pytest.raises(ValueError):
        Trainer(config, 9, {}, width=32).restore(path)
    from anthropocene_rl.policies.autonomous_inference_v2 import AutonomousElderPolicyV2

    policy = AutonomousElderPolicyV2(path, expected_provenance={}, seed=91)
    # Compare the saved boundary, before the extra discarded continuation update.
    restored.restore(path)
    assert torch.equal(policy.model(obs, actions)[1], restored.model(obs, actions)[1])
    with pytest.raises(ValueError):
        AutonomousElderPolicyV2(path, expected_provenance={"wrong": True}, seed=91)


@pytest.mark.skipif(not os.environ.get("AUTONOMOUS_V4_BRIDGE"), reason="native integration opt-in")
def test_v2_native_parallel_warm_start_and_populated_resume(tmp_path):
    from anthropocene_rl.autonomous_v4.collection import Collector
    from anthropocene_rl.autonomous_v4.training import state_hash

    torch.set_num_threads(1)
    config = json.loads(Path("configs/training/autonomous-ppo-remedy-v2.json").read_text())
    bank = Path(os.environ["AUTONOMOUS_V4_BANK"])
    entries = [
        e
        for e in json.loads((bank / "manifest.json").read_text())["entries"]
        if e["partition"] == "train"
    ]
    parent = Path(os.environ["AUTONOMOUS_V2_PARENT"])
    digest = hashlib.sha256(parent.read_bytes()).hexdigest()
    trainers = [AutonomousTrainerV2(config, seed, {"parent_sha256": digest}) for seed in [17, 29]]
    for trainer in trainers:
        trainer.warm_start(parent, expected_sha256=digest)
    assert trainers[0].optimizer.state  # Native-canary Adam state survives conversion.
    for phase in range(2):
        collected = []
        for trainer, workers in zip(trainers, [1, 4], strict=True):
            with Collector(
                json.loads(os.environ["AUTONOMOUS_V4_BRIDGE"]),
                workers=workers,
                deadline=time.monotonic() + 120,
            ) as collector:
                rows, episodes = collector.rollout(trainer, bank, entries, 17, 16, verify=True)
            collected.append((rows, episodes, trainer.update(rows)))
        a, b = collected
        assert a[2] == b[2]
        assert [e["fingerprint"] for e in a[1]] == [e["fingerprint"] for e in b[1]]
        for x, y in zip(a[0], b[0], strict=True):
            assert torch.equal(x["logp"], y["logp"])
            for key in ["workforce", "activities", "controls", "movement", "trade"]:
                assert torch.equal(getattr(x["action"], key), getattr(y["action"], key))
        assert state_hash(trainers[0].model) == state_hash(trainers[1].model)
        if phase == 0:
            path = tmp_path / "native-populated-v2.pt"
            trainers[0].checkpoint(path)
            trainers[1] = AutonomousTrainerV2(config, 43, {"parent_sha256": digest})
            trainers[1].restore(path)
        else:
            for key, value in trainers[0].optimizer.state_dict()["state"].items():
                for field, item in value.items():
                    other = trainers[1].optimizer.state_dict()["state"][key][field]
                    assert (
                        torch.equal(item, other)
                        if isinstance(item, torch.Tensor)
                        else item == other
                    )
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == digest
