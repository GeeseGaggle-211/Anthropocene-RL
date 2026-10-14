"""History contract, exact transfer and real retained PPO mechanics; no campaign."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v5.contracts import (
    FEATURE_WIDTH,
    HISTORY_NAMES,
    HISTORY_WIDTH,
    SLICES,
    features,
    observation,
)
from anthropocene_rl.autonomous_v5.policy import inputs
from anthropocene_rl.policies.autonomous_inference_v5 import AutonomousElderPolicyV5
from anthropocene_rl.training.autonomous_ppo_v4 import AutonomousTrainerV4
from anthropocene_rl.training.autonomous_ppo_v5 import AutonomousTrainerV5


def wire():
    activity = np.zeros((13, 15), dtype=bool)
    activity[0, :4] = True
    return dict(
        SchemaVersion=5,
        Context=[0.0] * 512,
        History=[0.0] * 298 + [1.0, 0.0],
        WorkforceMask=[True] * 13,
        ActivityMask=activity.tolist(),
        ControlMask=[True] * 3,
        MovementMask=[True] * 8,
        CounterpartyMask=[True] * 8,
        GoodMask=[True] * 11,
        Stocks=[100.0] * 11,
    )


def config(variant="residual_layernorm"):
    c = json.loads(Path("configs/training/autonomous-ppo-layernorm-v3-recovery.json").read_text())
    return dict(
        c,
        version="autonomous-ppo-residual-v4",
        model_version="autonomous-ppo-model-v4",
        critic_variant=variant,
        critic_lr=0.0001,
        horizon_years=400,
    )


def v5(c):
    return dict(
        c,
        version="autonomous-history-v5",
        model_version="autonomous-ppo-model-v5",
        observation_version=5,
    )


def rows(model, obs, generator):
    with torch.no_grad():
        a, lp, _, value = model(obs, generator=generator)
    return [
        dict(
            obs={k: v[i : i + 1] for k, v in obs.items()},
            action=a.index(slice(i, i + 1)),
            logp=lp[i : i + 1],
            advantage=float(i % 2 * 2 - 1),
            **{"return": float(value[i] + i % 2 * 2 - 1)},
        )
        for i in range(len(value))
    ]


def test_history_schema_labels_and_fixed_feature_compression():
    assert HISTORY_WIDTH == 300 and FEATURE_WIDTH == 1061 and len(set(HISTORY_NAMES)) == 300
    w = wire()
    w["History"][-1] = 1e100
    x = features(observation(w))
    assert x.shape == (1061,) and np.isfinite(x).all()
    assert x[-2] == 1 and x[-1] == pytest.approx(np.log1p(1e100))
    for invalid in (
        dict(w, SchemaVersion=4),
        dict(w, History=[0] * 299),
        dict(w, History=[float("nan"), *w["History"][1:]]),
    ):
        with pytest.raises(ValueError):
            observation(invalid)
    w["Context"][40] = 1
    with pytest.raises(ValueError):
        observation(w)


@pytest.mark.parametrize("variant", ["small_step", "residual_layernorm"])
def test_history_fork_preserves_populated_adam_probabilities_values_rng_and_anchors(
    tmp_path, variant
):
    from test_autonomous_v4 import batch

    torch.set_num_threads(1)
    c = config(variant)
    old = AutonomousTrainerV4(c, 31, {"parent": 1}, width=16)
    old.optimizer.param_groups[0]["lr"] = 0.00017
    base = batch(32)
    old.update(rows(old.model, base, old.sampler))
    p = tmp_path / "old.pt"
    old.checkpoint(p)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    t = AutonomousTrainerV5(v5(c), 29, {"history": 5}, width=16)
    t.warm_start(p, expected_sha256=digest)
    assert t.optimizer.param_groups[0]["lr"] == 0.00017
    obs = dict(base, x=torch.cat((base["x"], torch.randn(32, 300)), 1))
    old_state = old.model.state_dict()
    assert all(torch.equal(v, t.model.state_dict()[n]) for n, v in old_state.items())
    assert all(not v.any() for n, v in t.model.state_dict().items() if n.endswith("history_weight"))
    assert torch.equal(old.sampler.get_state(), t.sampler.get_state())
    a, lp, ent, value = old.model(base, generator=old.sampler)
    b, newlp, newent, newvalue = t.model(obs, generator=t.sampler)
    for name in ("workforce", "activities", "controls", "movement", "trade"):
        assert torch.equal(getattr(a, name), getattr(b, name))
    assert torch.equal(lp, newlp) and torch.equal(ent, newent) and torch.equal(value, newvalue)
    parameters = dict(t.model.named_parameters())
    for name, parameter in old.model.named_parameters():
        for key, v in old.optimizer.state.get(parameter, {}).items():
            actual = t.optimizer.state[parameters[name]][key]
            assert torch.equal(v, actual) if isinstance(v, torch.Tensor) else v == actual
    anchors = {n: p.clone() for n, p in parameters.items() if not p.requires_grad}
    t.update(rows(t.model, obs, t.sampler))
    assert t.model.actor[0].history_weight.abs().sum() > 0
    assert t.model.critic[0][0].history_weight.abs().sum() > 0
    for n, v in anchors.items():
        assert torch.equal(v, parameters[n]) and parameters[n].grad is None
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest


def test_history_checkpoint_next_update_and_inference_are_exact(tmp_path):
    torch.set_num_threads(1)
    c = v5(config())
    t = AutonomousTrainerV5(c, 7, {"v": 5}, width=16)
    obs = inputs([observation(wire()) for _ in range(32)])
    t.update(rows(t.model, obs, t.sampler))
    p = tmp_path / "v5.pt"
    t.checkpoint(p)
    r = AutonomousTrainerV5(c, 17, {"v": 5}, width=16)
    r.restore(p)
    batch_rows = rows(t.model, obs, torch.Generator().manual_seed(19))
    assert t.update(batch_rows) == r.update(batch_rows)
    assert all(torch.equal(v, r.model.state_dict()[k]) for k, v in t.model.state_dict().items())
    policy = AutonomousElderPolicyV5(p, expected_provenance={"v": 5}, seed=19)
    r.restore(p)
    assert torch.equal(policy.model.value(obs), r.model.value(obs))
    result = policy.act_many({3: wire(), 1: wire()}, deterministic=True)
    assert list(result) == [1, 3]
    wrong = torch.load(p, weights_only=False)
    wrong["history_names"] = tuple(reversed(HISTORY_NAMES))
    torch.save(wrong, p)
    with pytest.raises(ValueError):
        r.restore(p)


def test_receipt_is_not_validated_against_changed_current_masks():
    w = wire()
    h = np.asarray(w["History"])
    h[0:3] = [1, 20, 20]
    h[-1] = 20
    h[SLICES["requested_workforce"]][0] = 1
    h[SLICES["requested_activities"]][0] = 1
    h[SLICES["requested_movement"]][1] = 1
    h[SLICES["compiled_staffing"]][0] = 10
    h[SLICES["population_start"]] = 10
    h[SLICES["quote_present"]] = 1
    h[SLICES["quoted_partner_ordinal"]][3] = 1
    h[SLICES["quoted_give"]][8] = 1
    h[SLICES["quoted_want"]][9] = 1
    h[SLICES["quoted_quantities"]] = [3, 6]
    w["History"] = h.tolist()
    w["MovementMask"] = [True] + [False] * 7
    w["CounterpartyMask"] = [False] * 8
    decoded = observation(w)
    assert decoded["History"][SLICES["quoted_partner_ordinal"]][3] == 1
    bad = copy.deepcopy(w)
    bad["History"][1] = 10
    with pytest.raises(ValueError):
        observation(bad)


@pytest.mark.parametrize("bad", [2, float("nan"), "false", 0])
def test_history_masks_reject_nonboolean_values(bad):
    w = wire()
    w["WorkforceMask"][0] = bad
    with pytest.raises(ValueError):
        observation(w)


def test_curriculum_rejects_unaudited_quarantined_and_heldout_states(tmp_path):
    from anthropocene_rl.autonomous_v5.curriculum import load_curriculum

    entries = [
        dict(id="cold", partition="train", focal=1, peer=2, neural_training_eligible=True),
        dict(id="herder", partition="train", focal=1, peer=None, neural_training_eligible=False),
        dict(id="held", partition="heldout", focal=3, peer=None, neural_training_eligible=True),
    ]
    manifest = dict(version="autonomous-history-fresh-v5", status="PASS", entries=entries)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    audit = dict(
        version="autonomous-history-bank-audit-v5",
        status="PASS",
        manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        native_steps=0,
        model_calls=0,
        entries=[dict(e, controlled=[e["focal"]]) for e in entries],
    )
    a = tmp_path / "audit.json"
    a.write_text(json.dumps(audit))
    selected = load_curriculum(tmp_path, a)
    assert [e["id"] for e in selected] == ["cold"] and selected[0]["peer"] is None
    with pytest.raises(ValueError, match="HELDOUT"):
        load_curriculum(tmp_path, a, partition="heldout")
    manifest["entries"][0]["partition"] = "heldout"
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash-matched"):
        load_curriculum(tmp_path, a)


def test_native_progression_stops_core_experiment_before_unvalidated_model_input():
    from anthropocene_rl.autonomous_v5.collection import core_inputs

    w = wire()
    w["WorkforceMask"][2:4] = [False, False]
    w["ControlMask"][1:3] = [False, False]
    assert core_inputs([observation(w)])["x"].shape == (1, 1061)
    w["WorkforceMask"][3] = True
    with pytest.raises(ValueError, match="unvalidated"):
        core_inputs([observation(w)])


@pytest.mark.parametrize("bad", [None, 0, "false"])
def test_unvalidated_faces_cannot_be_admitted_by_falsy_configuration(bad):
    from anthropocene_rl.autonomous_v5.collection import episode

    with pytest.raises(ValueError, match="explicit boolean"):
        episode(None, None, None, None, None, {"require_core_action_faces": bad})
