"""labor_head_temper: exact fission/gate temperature, trade-share/log-std bias shifts.

Synthetic CPU fixtures (WP-A1 models with a collapsed posture, the real collect command
over the fake-bridge rows). Checks: logits/T equality, targets within 1e-3 on the
reference rows (recomputed by the checkpoint's own actor), every other parameter and
checkpoint field bitwise unchanged, Adam entries of the changed tensors reset, the
provenance record, the non-collector (F2b) path, refusals, and that collect/train load the
result.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from test_labor_a_ppo_wiring import _default_models, _iteration_files, _train_args

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7 import labor_head_temper as temper
from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7.labor_actor_optim import actor_optimizer_state_by_name
from anthropocene_rl.autonomous_v7.labor_model import LaborPersonality
from anthropocene_rl.autonomous_v7.provenance import checkpoint_hash


def _collapsed_state(fission_bias=(9.0, -14.5, -14.7), seed=1):
    """A trainer state whose non-labor posture is collapsed like the pilot's, with Adam
    moments on every actor parameter."""
    actor, task, costs, normalizer, layout = _default_models(seed)
    generator = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        actor.fission.weight.mul_(0.01)
        actor.fission.bias.copy_(torch.tensor(fission_bias))
        actor.trade_token.bias[2 + contracts.TRADE_MAKE] = 24.0
        actor.trade_token.weight[2:].normal_(0.0, 0.01, generator=generator)
        actor.trade_share.weight.normal_(0.0, 0.05, generator=generator)
        actor.trade_share.bias.copy_(torch.tensor([-9.2, -1.5]))
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-6)
    optimizer.zero_grad()
    sum(p.sum() for p in actor.parameters()).backward()
    optimizer.step()  # moments and step counts on every parameter (lr tiny)
    optimizer.zero_grad()
    state = training._trainer_checkpoint(
        actor, task, costs, normalizer, layout, optimizer,
        torch.optim.Adam([*task.parameters(), *costs.parameters()]),
        personality=LaborPersonality(), optimizer_seed=1, iteration=0,
        sampling_generator=torch.Generator().manual_seed(1),
        shuffle_rng=np.random.default_rng(1),
        critic_training_world_seeds=[], critic_training_independent_unit_ids=[],
    )
    return state


@pytest.fixture()
def collected(monkeypatch, tmp_path):
    state = _collapsed_state()
    checkpoint, rollout, anchor, gate = _iteration_files(monkeypatch, tmp_path, state, "c")
    return {"state": state, "checkpoint": checkpoint, "rollout": rollout, "anchor": anchor,
            "gate": gate, "tmp": tmp_path}


def _outputs(state, rollout):
    _, actor, *_ = training._models_from_checkpoint(state)
    batch, *_ = training._load_rollout(rollout)
    with torch.no_grad():
        return actor(batch.initial_tensors), batch


def test_fission_temperature_divides_logits_and_touches_nothing_else(collected):
    state, rollout = collected["state"], collected["rollout"]
    tempered, record = temper.temper_checkpoint(state, rollout, fission_temperature=4.0)
    before, _ = _outputs(state, rollout)
    after, _ = _outputs(tempered, rollout)
    assert torch.allclose(after["fission"], before["fission"] / 4.0, rtol=1e-6, atol=1e-6)
    for key, value in before.items():
        if key != "fission":
            assert torch.equal(value, after[key]), key
    assert set(record["changedParameters"]) == {"fission.weight", "fission.bias"}
    for name, value in state["actor"].items():
        if name not in record["changedParameters"]:
            assert torch.equal(value, tempered["actor"][name]), name
    from anthropocene_rl.autonomous_v7.labor_compose_checkpoint import _equal

    assert set(tempered) == set(state) | {temper.TEMPER_KEY}
    for key in state:
        if key not in ("actor", "actor_optimizer"):
            assert _equal(state[key], tempered[key]), key
    _, actor, *_ = training._models_from_checkpoint(state)
    old = actor_optimizer_state_by_name(actor, state["actor_optimizer"])
    new = actor_optimizer_state_by_name(actor, tempered["actor_optimizer"])
    assert set(old) - set(new) == {"fission.weight", "fission.bias"}
    for name, entry in new.items():
        for k, v in entry.items():
            assert torch.equal(v, old[name][k]), name
    assert record["actorOptimizer"]["removedAdamEntries"] == {
        "fission.weight": 1, "fission.bias": 1
    }
    assert tempered[temper.TEMPER_KEY][-1] is record


def test_targets_are_achieved_and_recorded(collected):
    state, rollout = collected["state"], collected["rollout"]
    tempered, record = temper.temper_checkpoint(
        state, rollout, fission_explore=0.02, share_explore=0.1, share_log_std=-1.0,
        gate_explore=0.05, value_log_std=-1.0,
    )
    achieved = record["achievedAfter"]
    for key, target in record["targets"].items():
        assert abs(achieved[key] - target) < 1e-3, key
    assert record["measuredBefore"]["fissionNonNeutral"] < 1e-8
    assert record["measuredBefore"]["tradeGateNonMake"] < 1e-8
    assert record["transform"]["fissionTemperature"] > 1
    assert set(record["changedParameters"]) == {
        "fission.weight", "fission.bias", "trade_token.weight", "trade_token.bias",
        "trade_share.bias",
    }
    for name, value in state["actor"].items():
        if name not in record["changedParameters"]:
            assert torch.equal(value, tempered["actor"][name]), name
        else:
            digests = record["changedParameters"][name]
            assert digests["before"] == temper.tensor_digest(value)
            assert digests["after"] == temper.tensor_digest(tempered["actor"][name])
    # trade_share.weight and the trade_token value-mean row are untouched
    before_actor, after_actor = state["actor"], tempered["actor"]
    assert torch.equal(before_actor["trade_share.weight"], after_actor["trade_share.weight"])
    token = "trade_token.weight"
    assert torch.equal(before_actor[token][:2], after_actor[token][:2])
    assert torch.equal(before_actor["trade_token.bias"][0], after_actor["trade_token.bias"][0])
    reference = record["referenceRollout"]
    assert reference["sha256"] == checkpoint_hash(rollout)
    assert reference["storedOutputCrossCheck"]["checkpointIsCollector"] is True
    assert reference["storedOutputCrossCheck"]["match"] is True
    batch, *_ = training._load_rollout(rollout)
    assert record["sourceActor"]["behaviorPolicyDigest"] == batch.behavior_policy_digest
    assert reference["collectorBehaviorPolicyDigest"] == batch.behavior_policy_digest
    assert record["temperedActor"]["behaviorPolicyDigest"] != batch.behavior_policy_digest
    # independent Monte-Carlo check of the expected export share (sigmoid of the sampled latent)
    after, _ = _outputs(tempered, rollout)
    noise = torch.randn((20000, batch.parent_count), generator=torch.Generator().manual_seed(5),
                        dtype=torch.float64)
    std = after["trade_share_log_std"].double().exp()
    latent = after["trade_share_mean"].double() + noise * std
    assert abs(float(torch.sigmoid(latent).mean()) - 0.1) < 5e-3
    assert abs(float(after["trade_share_log_std"].double().mean()) + 1.0) < 1e-3


def test_cli_writes_a_checkpoint_collect_and_train_load(collected, monkeypatch):
    tmp = collected["tmp"]
    out, report = tmp / "tempered.pt", tmp / "temper.json"
    temper.main(["--checkpoint", str(collected["checkpoint"]), "--reference-rollout",
                 str(collected["rollout"]), "--fission-explore", "0.05",
                 "--trade-share-explore", "0.2", "--out", str(out), "--report", str(report)])
    record = json.loads(report.read_text(encoding="utf-8"))
    assert record["outSha256"] == checkpoint_hash(out)
    assert record["provenance"]["inputCheckpointSha256"] == checkpoint_hash(collected["checkpoint"])
    with pytest.raises(ValueError, match="never overwrites"):
        temper.main(["--checkpoint", str(collected["checkpoint"]), "--reference-rollout",
                     str(collected["rollout"]), "--fission-temperature", "2", "--out", str(out)])
    state = torch.load(out, weights_only=True)
    training._require_trainer_checkpoint(state)
    # collect (real command, fake-bridge rows) and train from the tempered checkpoint
    checkpoint, rollout, anchor, gate = _iteration_files(monkeypatch, tmp, state, "t")
    trained = tmp / "trained.pt"
    training.main(_train_args(checkpoint, rollout, anchor, gate, trained))
    assert torch.load(trained, weights_only=True)["last_update"]["actor"]["attemptedSteps"] > 0


def test_non_collector_checkpoint_uses_its_own_outputs(collected):
    """F2b case: the start actor differs from the rollout's collector."""
    other = _collapsed_state(fission_bias=(7.0, -11.0, -12.5))
    other["normalizer"] = collected["state"]["normalizer"]
    tempered, record = temper.temper_checkpoint(other, collected["rollout"], fission_explore=0.03)
    cross = record["referenceRollout"]["storedOutputCrossCheck"]
    assert cross["checkpointIsCollector"] is False and cross["match"] is False
    own, _ = _outputs(other, collected["rollout"])
    expected = 1.0 - float(torch.softmax(own["fission"].double(), dim=1)[:, 0].mean())
    assert record["measuredBefore"]["fissionNonNeutral"] == pytest.approx(expected, rel=1e-9)
    assert abs(record["achievedAfter"]["fissionNonNeutral"] - 0.03) < 1e-3
    after, _ = _outputs(tempered, collected["rollout"])
    t = record["transform"]["fissionTemperature"]
    assert torch.allclose(after["fission"], own["fission"] / t, rtol=1e-5, atol=1e-6)


def test_refusals(collected):
    state, rollout, tmp = collected["state"], collected["rollout"], collected["tmp"]
    raw = torch.load(rollout, weights_only=True)
    raw["batch"]["initial_actor_outputs"]["fission"] = (
        raw["batch"]["initial_actor_outputs"]["fission"] + 1.0
    )
    tampered = tmp / "tampered.pt"
    torch.save(raw, tampered)
    with pytest.raises(ValueError, match="stored outputs"):
        temper.temper_checkpoint(state, tampered, fission_explore=0.02)
    raw = torch.load(rollout, weights_only=True)
    raw["batch"]["normalization_binding_digest"] = "f" * 64
    renormed = tmp / "renormed.pt"
    torch.save(raw, renormed)
    with pytest.raises(ValueError, match="normalization binding"):
        temper.temper_checkpoint(state, renormed, fission_explore=0.02)
    with pytest.raises(ValueError, match="exclusive"):
        temper.temper_checkpoint(state, rollout, fission_explore=0.02, fission_temperature=2.0)
    with pytest.raises(ValueError, match="no tempering"):
        temper.temper_checkpoint(state, rollout)
    explored = _collapsed_state(fission_bias=(0.0, 0.0, 0.0))
    explored["normalizer"] = state["normalizer"]
    with pytest.raises(ValueError, match="already explores"):
        temper.temper_checkpoint(explored, rollout, fission_explore=0.02)


def test_expected_share_quadrature():
    mean, log_std = np.array([-2.0, 0.0, 3.0]), np.array([0.0, -1.0, 0.5])
    rng = np.random.default_rng(0)
    z = mean[:, None] + np.exp(log_std)[:, None] * rng.standard_normal((3, 400000))
    assert temper.expected_share(mean, log_std) == pytest.approx(
        float((1 / (1 + np.exp(-z))).mean()), abs=2e-3
    )
