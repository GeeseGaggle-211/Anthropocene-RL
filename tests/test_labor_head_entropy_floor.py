"""Optional per-head entropy floors in the labor PPO actor loss (--head-entropy-floor).

Default off must be bitwise the 78616b6 update (update-level and train-level, open and
closed gate, --a-mode off). With floors the term w * mean(relu(H_min - H_head)) enters the
same minibatch loss as the surrogate, and its gradient reaches only the floored head (plus
the shared encoder) when the surrogate is zeroed. Synthetic CPU fixtures only.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_labor_a_ppo_wiring import (
    _default_models,
    _iteration_files,
    _same,
    _small,
    _train_args,
    _trainer_state,
)
from test_labor_ppo_perhead import _gate, _heads_scorer, _joint_scorer

from anthropocene_rl.autonomous_v7 import labor_ppo as ppo
from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7.labor_model import LaborPersonality, nonlabor_entropy

REPO = Path(__file__).resolve().parents[1]
BASE = "78616b6"  # feature/r5-ppo before the exploration work


def _base(name: str):
    """Import the 78616b6 module ``name`` from git under a private name."""
    modname = f"anthropocene_rl.autonomous_v7._explore_base_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    if name != "labor_ppo":
        _base("labor_ppo")
    try:
        source = subprocess.run(
            ["git", "show", f"{BASE}:src/anthropocene_rl/autonomous_v7/{name}.py"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("base source is not available from git")
    source = source.replace("from .labor_ppo import", "from ._explore_base_labor_ppo import")
    spec = importlib.util.spec_from_loader(modname, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "anthropocene_rl.autonomous_v7"
    sys.modules[modname] = module
    exec(compile(source, f"<{BASE}:{name}.py>", "exec"), module.__dict__)
    return module


def _entropy_heads_scorer(actor, per_head=False):
    return lambda tensors, sample: training._score_nonlabor_entropy_heads(
        actor, tensors, sample, per_head_logp=per_head
    )


# --- parsing --------------------------------------------------------------------------------


def test_parse_head_entropy_floor():
    assert ppo.parse_head_entropy_floor(None) is None
    assert ppo.parse_head_entropy_floor("  ") is None
    assert ppo.parse_head_entropy_floor("fission=0.3:0.1,trade_share=-1.5:2") == {
        "fission": (0.3, 0.1), "trade_share": (-1.5, 2.0),
    }
    for bad in ("specialty=1:1", "fission=1", "fission=1:-1", "fission=nan:1",
                "fission=1:1,fission=2:1", "fission:1"):
        with pytest.raises(ValueError):
            ppo.parse_head_entropy_floor(bad)


def test_floor_loss_on_hand_built_entropies():
    entropy = {
        "fission": torch.tensor([0.5, 1.5, 3.0]),
        "trade_share": torch.tensor([-2.0, -1.0, 0.0]),
        "movement": torch.tensor([0.0, 0.0, 0.0]),
    }
    floors = {"fission": (1.0, 2.0), "trade_share": (-0.5, 0.5)}
    # fission: 2 * mean(0.5, 0, 0) = 1/3; share: 0.5 * mean(1.5, 0.5, 0) = 1/3
    value = ppo.head_entropy_floor_loss(entropy, floors)
    assert torch.allclose(value, torch.tensor(2.0 / 3.0))
    assert float(ppo.head_entropy_floor_loss(entropy, {"movement": (0.0, 5.0)})) == 0.0


# --- minibatch loss ---------------------------------------------------------------------------


def test_floor_enters_the_minibatch_loss_exactly(monkeypatch):
    models, _, batch, _ = _small(monkeypatch)
    actor = models[0]
    index = torch.arange(batch.parent_count, dtype=torch.long)
    floors = {"fission": (5.0, 0.3), "trade_share": (2.0, 0.7), "trade_gate": (0.1, 1.0)}
    base = ppo.labor_policy_minibatch_terms(
        actor, batch, index, _joint_scorer(actor), clip=0.2, entropy_weight=0.01
    )
    floored = ppo.labor_policy_minibatch_terms(
        actor, batch, index, _entropy_heads_scorer(actor), clip=0.2, entropy_weight=0.01,
        head_entropy_floor=floors,
    )
    with torch.no_grad():
        entropy = nonlabor_entropy(actor(batch.initial_tensors))
    expected = sum(
        w * torch.relu(h - entropy[name]).mean() for name, (h, w) in floors.items()
    )
    assert torch.allclose(floored.head_entropy_floor_loss.detach(), expected.detach(), atol=1e-6)
    assert float(floored.head_entropy_floor_loss.detach()) > 0
    assert torch.equal(floored.policy_loss, base.policy_loss)
    assert torch.equal(floored.entropy, base.entropy)
    assert torch.equal(floored.loss, base.loss + floored.head_entropy_floor_loss)
    # the scorer's joint is the other scorers' bit for bit
    assert base.head_entropy_floor_loss is None
    with pytest.raises(ValueError, match="per-head entropies"):
        ppo.labor_policy_minibatch_terms(
            actor, batch, index, _joint_scorer(actor), clip=0.2, entropy_weight=0.0,
            head_entropy_floor=floors,
        )


HEAD_MODULES = ("specialty", "activity", "stay", "site_query", "region_keep", "region_query",
                "fission", "trade_token", "trade_share")


@pytest.mark.parametrize("head,module", [("fission", "fission"), ("trade_share", "trade_share"),
                                         ("trade_gate", "trade_token")])
def test_floor_gradient_reaches_only_the_floored_head(monkeypatch, head, module):
    models, _, batch, _ = _small(monkeypatch)
    actor = copy.deepcopy(models[0])
    zeroed = dataclasses.replace(
        batch,
        initial_advantage=torch.zeros_like(batch.initial_advantage),
        retry_advantage=torch.zeros_like(batch.retry_advantage),
    )
    index = torch.arange(batch.parent_count, dtype=torch.long)
    terms = ppo.labor_policy_minibatch_terms(
        actor, zeroed, index, _entropy_heads_scorer(actor), clip=0.2, entropy_weight=0.0,
        head_entropy_floor={head: (50.0, 1.0)},
    )
    actor.zero_grad()
    terms.loss.backward()
    grads = {name: p.grad for name, p in actor.named_parameters()}
    floored = [n for n in grads if n.startswith(module + ".")]
    assert any(grads[n] is not None and grads[n].abs().sum() > 0 for n in floored)
    for name, grad in grads.items():
        if name.startswith("encoder.") or name in floored:
            continue
        assert grad is None or not grad.any(), name
    if head == "trade_share":  # differential entropy depends on the log-std row only
        assert not grads["trade_share.bias"][0] and grads["trade_share.bias"][1]
    if head == "trade_gate":  # the gate logits only, not the value mean/log-std rows
        assert not grads["trade_token.bias"][:2].any() and grads["trade_token.bias"][2:].any()


def test_update_with_floor_records_and_is_bounded_by_the_kl_audit(monkeypatch):
    models, _, batch, _ = _small(monkeypatch)
    actor = copy.deepcopy(models[0])
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    stats = ppo.update_labor_policy(
        policy_module=actor, labor_actor=actor, optimizer=optimizer, batch=batch,
        nonlabor_scorer=_entropy_heads_scorer(actor), rng=np.random.default_rng(3), epochs=2,
        batch_size=3, clip=0.2, entropy_weight=0.0, max_grad_norm=1.0, target_kl=1e-9,
        audit_rows=4, actor_gate=_gate(batch),
        behavior_configuration=training.labor_behavior_configuration(LaborPersonality()),
        head_entropy_floor={"fission": (5.0, 10.0)},
    )
    record = stats["headEntropyFloor"]
    assert record["floors"] == {"fission": {"hMin": 5.0, "weight": 10.0}}
    assert len(record["floorLossPerAttemptedStep"]) == stats["attemptedSteps"] >= 1
    # the KL audit still rolls the floored step back
    assert stats["actorStepRolledBack"] and stats["acceptedSteps"] == 0
    for name, value in models[0].state_dict().items():
        assert torch.equal(value, actor.state_dict()[name]), name


# --- default off: bitwise the base ------------------------------------------------------------


@pytest.mark.parametrize("per_head", [False, True])
def test_default_update_is_bitwise_the_base_update(monkeypatch, per_head):
    old_ppo = _base("labor_ppo")
    models, _, batch, _ = _small(monkeypatch)
    results = []
    for module in (old_ppo, ppo):
        candidate = (
            batch if module is ppo else old_ppo.LaborAttemptPolicyBatch(
                **{f.name: getattr(batch, f.name)
                   for f in dataclasses.fields(old_ppo.LaborAttemptPolicyBatch)}
            )
        )
        actor = copy.deepcopy(models[0])
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
        stats = module.update_labor_policy(
            policy_module=actor, labor_actor=actor, optimizer=optimizer, batch=candidate,
            nonlabor_scorer=(_heads_scorer if per_head else _joint_scorer)(actor),
            rng=np.random.default_rng(3), epochs=2, batch_size=3, clip=0.2,
            entropy_weight=0.01, max_grad_norm=1.0, target_kl=10.0, audit_rows=4,
            actor_gate=_gate(batch),
            behavior_configuration=training.labor_behavior_configuration(LaborPersonality()),
            **({"per_head_surrogate": True} if per_head else {}),
        )
        results.append((stats, actor.state_dict(), optimizer.state_dict()["state"]))
    (stats_old, actor_old, opt_old), (stats_new, actor_new, opt_new) = results
    assert stats_old == stats_new and stats_new["acceptedSteps"] == 6
    for name, value in actor_old.items():
        assert torch.equal(value, actor_new[name]), name
    for key, state in opt_old.items():
        for name, value in state.items():
            assert torch.equal(value, opt_new[key][name])


@pytest.fixture()
def files(monkeypatch, tmp_path):
    actor, task, costs, normalizer, layout = _default_models()
    state = _trainer_state(actor, task, costs, normalizer, layout)
    return _iteration_files(monkeypatch, tmp_path, state, "it0")


@pytest.mark.parametrize("gate_open", [True, False])
def test_default_train_is_bitwise_the_base_train(files, monkeypatch, tmp_path, gate_open):
    old = _base("labor_training")
    monkeypatch.setattr(old, "load_labor_anchor", training.load_labor_anchor)
    checkpoint, rollout, anchor, gate = files
    if not gate_open:
        report = json.loads(gate.read_text(encoding="utf-8"))
        report["actorGateOpen"] = False
        gate.write_text(json.dumps(report), encoding="utf-8")
    outputs = {}
    for tag, main in (("base", old.main), ("now", training.main)):
        out = tmp_path / f"{tag}-{gate_open}.pt"
        main(_train_args(checkpoint, rollout, anchor, gate, out, "--a-mode", "off"))
        outputs[tag] = torch.load(out, weights_only=True)
    _same(outputs["base"], outputs["now"])
    assert ("acceptedSteps" in outputs["now"]["last_update"]["actor"]) is gate_open
    assert "headEntropyFloor" not in outputs["now"]["last_update"]


def test_train_with_floor_records_entropies(files, tmp_path):
    checkpoint, rollout, anchor, gate = files
    out = tmp_path / "floor.pt"
    training.main(_train_args(checkpoint, rollout, anchor, gate, out,
                              "--head-entropy-floor", "fission=5:0.1,trade_share=3:0.1"))
    update = torch.load(out, weights_only=True)["last_update"]
    record = update["headEntropyFloor"]
    assert set(record["floors"]) == {"fission", "trade_share"}
    assert set(record["entropyBefore"]) == set(ppo.INITIAL_POLICY_HEADS)
    assert set(record["entropyAfter"]) == set(ppo.INITIAL_POLICY_HEADS)
    assert record["actorUpdated"] is bool(update["actor"]["acceptedSteps"])
    assert update["actor"]["headEntropyFloor"]["floorLossPerAttemptedStep"]
    with pytest.raises(ValueError, match="must be one of"):
        training.main(_train_args(checkpoint, rollout, anchor, gate, tmp_path / "bad.pt",
                                  "--head-entropy-floor", "specialty=1:1"))


def test_apply_with_floor_is_train_with_floor(files, tmp_path):
    from anthropocene_rl.autonomous_v7 import labor_a_measure as measure

    checkpoint, rollout, anchor, gate = files
    floor = "fission=5:0.2,trade_gate=3:0.1"
    trained = tmp_path / "train-floor.pt"
    training.main(_train_args(checkpoint, rollout, anchor, gate, trained,
                              "--head-entropy-floor", floor))
    applied = tmp_path / "apply-floor.pt"
    measure.main(["--apply", "--checkpoint", str(checkpoint), "--rollout", str(rollout),
                  "--out-checkpoint", str(applied), "--out", str(tmp_path / "apply.json"),
                  "--epochs", "2", "--batch-size", "2", "--actor-lr", "1e-6",
                  "--target-kl", "10", "--threads", str(torch.get_num_threads()),
                  "--head-entropy-floor", floor])
    trained_state = torch.load(trained, weights_only=True)
    applied_state = torch.load(applied, weights_only=True)
    _same(trained_state["actor"], applied_state["actor"])
    left = trained_state["last_update"]["headEntropyFloor"]
    right = applied_state["last_update"]["headEntropyFloor"]
    assert left["floors"] == right["floors"] and left["entropyAfter"] == right["entropyAfter"]
    assert (trained_state["last_update"]["actor"]["headEntropyFloor"]
            == applied_state["last_update"]["actor"]["headEntropyFloor"])
