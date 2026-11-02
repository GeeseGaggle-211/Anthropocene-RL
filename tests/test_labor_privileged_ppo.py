"""Small CPU fixtures for PPO's privileged channel and decision warmup boundary."""

from __future__ import annotations

import copy
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_labor_bridge import _attempt_payload
from test_labor_model import _labor_tensors
from test_labor_privileged_bridge import _raw
from test_labor_privileged_model import _critics, _layout, _village
from test_labor_training import _row
from test_labor_value_calibration import _write

from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7.bridge import observation_batch
from anthropocene_rl.autonomous_v7.labor_advantage import action_features
from anthropocene_rl.autonomous_v7.labor_bridge import LABOR_REGISTRY_DIGEST, LaborAttempt
from anthropocene_rl.autonomous_v7.labor_model import (
    LaborActor,
    LaborPersonality,
    Normalizer,
    labor_policy_tensors,
)
from anthropocene_rl.autonomous_v7.labor_ppo import LaborActorGate, update_labor_policy
from anthropocene_rl.autonomous_v7.labor_value_calibration import (
    labor_anchor_loss,
    labor_calibration_report,
    labor_critic_digest,
    load_labor_anchor,
)
from anthropocene_rl.autonomous_v7.value_calibration import behavior_policy_digest


def _batch(rows, bootstrap=None, actor=None, task=None, costs=None):
    return training.build_labor_training_batch(
        rows,
        bootstrap or {},
        personality=LaborPersonality(),
        cost_multipliers={"feasibility": 1.0, "settlement_guard": 1.0},
        behavior_policy_sha256=(
            "a" * 64
            if actor is None
            else behavior_policy_digest(
                actor, training.labor_behavior_configuration(LaborPersonality())
            )
        ),
        critic_sha256="b" * 64 if task is None else labor_critic_digest(task, costs),
        normalization_sha256="c" * 64,
    )


def _collect_one(monkeypatch, retry, models=None):
    village = dict(
        _village(1),
        camp=7,
        terminated=False,
        society=70,
        parentSociety=0,
        priorPopulation=100,
        population=100,
        qol=0.5,
        years=20.0,
        welfare=0.5,
    )
    normalizer = Normalizer.fit_bounded(observation_batch([village]))
    _, tensors = labor_policy_tensors([village], normalizer)
    layout = _layout(core=tensors["core"].shape[1])
    if models is None:
        actor = LaborActor(layout, width=16, token_width=16)
        task, costs = _critics(layout)
    else:
        actor, task, costs = models
    successor = copy.deepcopy(village)
    successor["privilegedView"] = _village(9)["privilegedView"]
    next_step = {"turn": 4, "villages": [successor]}
    captured = {}

    def execute(_bridge, catalogs, retained, retry_factory):
        catalog = village["observation"]["laborCatalog"]
        initial = LaborAttempt.from_payload(_attempt_payload(catalog, rejected=retry), catalogs[7])
        retried = {7: retry_factory(7, catalogs[7], initial)} if retry else {}
        final = dataclasses.replace(initial, attempt=1, remaining_proposals=0) if retry else initial
        captured.update(initial=retained, retry=retried)
        return SimpleNamespace(
            step=next_step, initial_attempts={7: initial}, final_attempts={7: final}, retry=retried
        )

    monkeypatch.setattr(training, "execute_labor_decision", execute)
    _, rows, _ = training._execute_single_step_detailed(
        bridge=object(),
        step={"turn": 3, "villages": [village]},
        actor=actor,
        task_critic=task,
        cost_critics=costs,
        normalizer=normalizer,
        personality=LaborPersonality(),
        generator=torch.Generator().manual_seed(71),
        world_seed=50000,
        world_index=0,
        founded_turns={7: 0},
    )
    tail = training._bootstrap_values(next_step, normalizer, task, costs, LaborPersonality())
    return actor, task, costs, rows[0], tail, captured


@pytest.mark.parametrize("retry", [False, True])
def test_collection_retry_bootstrap_and_replay_are_aligned(monkeypatch, tmp_path, retry):
    actor, task, costs, row, tail, captured = _collect_one(monkeypatch, retry)
    raw = _raw(1, 1)
    for key in raw:
        assert torch.equal(row.initial_privileged_raw[key], raw[key])
        if retry:
            assert torch.equal(row.retry_privileged_raw[key], raw[key])
    assert not any(key.startswith("privileged") for key in row.initial_tensors)
    for key, value in row.initial_actor_outputs.items():
        assert not value.requires_grad
        assert torch.equal(value, actor(row.initial_tensors)[key])
    expected = action_features(captured["initial"][7].payload)
    np.testing.assert_array_equal(row.initial_action_features["features"][0], expected)
    if retry:
        combined = dict(captured["initial"][7].payload, **captured["retry"][7].payload)
        np.testing.assert_array_equal(
            row.retry_action_features["features"][0], action_features(combined)
        )
        # Last 49 entries are the retained movement/fission/trade posture.
        assert torch.equal(
            row.initial_action_features["features"][:, 300:],
            row.retry_action_features["features"][:, 300:],
        )
    expected_tail = training._task_values(task, row.initial_tensors, _raw(1, 9))
    assert tail[7]["qol_years"] == pytest.approx(float(expected_tail["qol_years"][0]))
    batch, tt, ct = _batch([row], {(50000, 7): tail[7]})
    assert len(batch.retry_privileged_raw["privileged_core"]) == int(retry)
    path = tmp_path / "rollout.pt"
    torch.save(training._state_dict_rollout(batch, tt, ct, {}), path)
    restored, _, _, _ = training._load_rollout(path)
    assert restored.privileged_normalizer_digest == task.privileged_normalizer.digest()
    assert torch.equal(
        restored.initial_action_features["features"], batch.initial_action_features["features"]
    )
    losses = training._critic_td_losses(task, costs, restored, tt, ct)
    assert all(torch.isfinite(loss) for loss in losses)
    with pytest.raises(ValueError, match="normalization binding"):
        training._critic_td_losses(
            task, costs, dataclasses.replace(batch, privileged_normalizer_digest="0" * 64), tt, ct
        )


def test_warmup_counts_world_decisions_not_camps_and_bootstraps(monkeypatch):
    calls = []

    class Bridge:
        def reset(self, seed):
            return {"turn": 0, "villages": [{"camp": 7}, {"camp": 8}]}

        def close(self):
            pass

    def step(**kw):
        turn = kw["step"]["turn"]
        calls.append(turn)
        rows = [_row(turn, retry=False), dataclasses.replace(_row(turn, retry=True), camp_id=8)]
        return {"turn": turn + 1, "villages": [{"camp": 7}, {"camp": 8}]}, rows

    observed_tail = []
    values = dict(qol_years=12.0, population_years=120.0, feasibility=0.2, settlement_guard=0.3)

    def bootstrap(step, *args):
        observed_tail.append(step["turn"])
        return {7: values, 8: values}

    monkeypatch.setattr(training, "_sample_world_step", step)
    monkeypatch.setattr(training, "_bootstrap_values", bootstrap)
    rows, tail = training.collect_labor_rollout(
        actor=None,
        task_critic=None,
        cost_critics=None,
        normalizer=None,
        world_seeds=[50000],
        turns=100,
        warmup_decisions=9,
        generator=torch.Generator(),
        bridge_factory=Bridge,
    )
    assert calls == list(range(9)) and len(rows) == 18 and observed_tail == [9]
    batch, tt, _ = _batch(rows, tail)
    gamma = np.exp(-20.0 / 400.0)
    assert tt["initial"]["qol_years"][-1] == pytest.approx(2.0 + gamma * 12.0)
    assert batch.parent_count == 18
    for bad in (0, 10, True, 1.5):
        with pytest.raises(ValueError, match=r"1\.\.9"):
            training._warmup_limit(bad)


def test_privileged_warmup_gradient_isolation_then_gated_actor_update(monkeypatch):
    actor, task, costs, row, tail, _ = _collect_one(monkeypatch, True)
    batch, tt, ct = _batch([row], {(50000, 7): tail[7]}, actor, task, costs)
    actor_before = copy.deepcopy(actor.state_dict())
    critic_before = copy.deepcopy(task.state_dict())
    optimizer = torch.optim.Adam([*task.parameters(), *costs.parameters()], lr=1e-3)
    optimizer.zero_grad()
    sum(training._critic_td_losses(task, costs, batch, tt, ct)).backward()
    optimizer.step()
    assert all(torch.equal(actor_before[k], v) for k, v in actor.state_dict().items())
    assert all(p.grad is None for p in actor.parameters())
    assert any(not torch.equal(critic_before[k], v) for k, v in task.state_dict().items())
    # Recollect values/advantages using the warmed critics. The gate here is a
    # synthetic unit-test admission, never a real calibration/readiness claim.
    stale_gate = LaborActorGate(
        batch.behavior_policy_digest,
        batch.advantage_critic_digest,
        batch.normalization_binding_digest,
        "d" * 64,
    )
    _, _, _, row, tail, _ = _collect_one(monkeypatch, True, (actor, task, costs))
    batch, _, _ = _batch([row], {(50000, 7): tail[7]}, actor, task, costs)
    with pytest.raises(ValueError, match="other rollout-advantage critic weights"):
        stale_gate.require_current(
            actor,
            training.labor_behavior_configuration(LaborPersonality()),
            LABOR_REGISTRY_DIGEST,
            batch.advantage_critic_digest,
            batch.normalization_binding_digest,
        )
    gate = LaborActorGate(
        batch.behavior_policy_digest,
        batch.advantage_critic_digest,
        batch.normalization_binding_digest,
        "d" * 64,
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    update_labor_policy(
        policy_module=actor,
        labor_actor=actor,
        optimizer=actor_optimizer,
        batch=batch,
        nonlabor_scorer=lambda tensors, sample: training._score_nonlabor(actor, tensors, sample),
        rng=np.random.default_rng(1),
        epochs=1,
        batch_size=1,
        clip=0.2,
        entropy_weight=0.0,
        max_grad_norm=1.0,
        target_kl=10.0,
        audit_rows=1,
        actor_gate=gate,
        behavior_configuration=training.labor_behavior_configuration(LaborPersonality()),
    )
    assert any(not torch.equal(actor_before[k], v) for k, v in actor.state_dict().items())


def test_anchor_privileged_roundtrip_training_and_calibration(tmp_path):
    path = tmp_path / "anchor.npz"
    _write(path)
    with np.load(path) as data:
        arrays = dict(data)
    task, costs = _critics(_layout())
    metadata = json.loads(str(arrays["metadata_json"]))
    metadata["source"]["privilegedNormalizerDigest"] = task.privileged_normalizer.digest()
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    del arrays["input__predictions"], arrays["rank_input__predictions"]
    for key, value in _labor_tensors(4).items():
        arrays["input__" + key] = value.numpy()
        arrays["rank_input__" + key] = value.numpy()
    for key, value in _raw(4).items():
        arrays[key] = value.numpy()
        arrays["rank_" + key] = value.numpy()
    np.savez(path, **arrays)
    anchor = load_labor_anchor(path, "actor-gate")
    assert set(anchor.tensors).isdisjoint(anchor.privileged_raw)
    loss, heads = labor_anchor_loss(task, costs, anchor, np.array([3, 1]))
    assert torch.isfinite(loss) and len(heads) == 4
    source = metadata["source"]
    report = labor_calibration_report(
        task,
        costs,
        anchor,
        current_policy_digest=source["sourcePolicyDigest"],
        current_normalization_binding_digest=source["normalizationBindingDigest"],
        current_registry_digest=LABOR_REGISTRY_DIGEST,
    )
    assert report["sourceCriticDigest"] == labor_critic_digest(task, costs)
    bad = copy.deepcopy(anchor)
    bad.metadata["source"]["privilegedNormalizerDigest"] = "0" * 64
    with pytest.raises(ValueError, match="normalization binding"):
        labor_anchor_loss(task, costs, bad, np.array([0]))
    del arrays["privileged_core"]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="lack privileged data"):
        load_labor_anchor(path, "actor-gate")


def test_rollout_rejects_malformed_side_channels(monkeypatch):
    _, _, _, row, tail, _ = _collect_one(monkeypatch, False)
    batch, _, _ = _batch([row], {(50000, 7): tail[7]})
    with pytest.raises(ValueError, match="both initial and retry"):
        dataclasses.replace(batch, retry_privileged_raw=None)
    with pytest.raises(ValueError, match="SHA-256"):
        dataclasses.replace(batch, privileged_normalizer_digest="bad")
    with pytest.raises(ValueError, match="contract"):
        dataclasses.replace(batch, action_feature_contract="future")
    for value in (torch.zeros(1, 348), torch.full((1, 349), float("nan"))):
        with pytest.raises(ValueError, match="finite N x 349"):
            dataclasses.replace(batch, initial_action_features={"features": value})
    with pytest.raises(ValueError, match="detached"):
        dataclasses.replace(
            batch, initial_action_features={"features": torch.zeros(1, 349, requires_grad=True)}
        )


def test_warmup_cli_roundtrip_keeps_actor_and_requires_new_gate(monkeypatch, tmp_path):
    from test_labor_privileged_bridge import _normalizer

    from anthropocene_rl.autonomous_v7.labor_model import (
        LaborCostCritics,
        LaborTaskCritic,
        bind_privileged_normalizer,
    )

    village = dict(_village(1), camp=7)
    normalizer = Normalizer.fit_bounded(observation_batch([village]))
    _, tensors = labor_policy_tensors([village], normalizer)
    layout = _layout(core=tensors["core"].shape[1])
    torch.manual_seed(1)
    actor, task, costs = LaborActor(layout), LaborTaskCritic(layout), LaborCostCritics(layout)
    bind_privileged_normalizer(_normalizer(), task, costs)
    _, _, _, row, tail, _ = _collect_one(monkeypatch, False, (actor, task, costs))
    state = training._trainer_checkpoint(
        actor,
        task,
        costs,
        normalizer,
        layout,
        torch.optim.Adam(actor.parameters()),
        torch.optim.Adam([*task.parameters(), *costs.parameters()]),
        personality=LaborPersonality(),
        optimizer_seed=1,
        iteration=0,
        sampling_generator=torch.Generator().manual_seed(1),
        shuffle_rng=np.random.default_rng(1),
        critic_training_world_seeds=[],
        critic_training_independent_unit_ids=[],
    )
    checkpoint, rollout, out = (tmp_path / name for name in ("source.pt", "rollout.pt", "warm.pt"))
    torch.save(state, checkpoint)
    monkeypatch.setattr(
        training, "collect_labor_rollout", lambda **kw: ([row], {(50000, 7): tail[7]})
    )
    training.main(
        [
            "collect",
            "--checkpoint",
            str(checkpoint),
            "--worlds",
            "50000",
            "--warmup-decisions",
            "9",
            "--out",
            str(rollout),
        ]
    )
    source = {
        "sourcePolicyDigest": behavior_policy_digest(
            actor, training.labor_behavior_configuration(LaborPersonality())
        ),
        "normalizationBindingDigest": training._normalization_digest(normalizer),
        "privilegedNormalizerDigest": task.privileged_normalizer.digest(),
        "worldSeeds": [50001],
    }

    class Anchor:
        def __init__(self):
            self.metadata = {"source": source}
            self.independent_unit_ids = np.array(["world-50001"])
            self.privileged_raw = row.initial_privileged_raw
            self.tensors = {key: value.numpy() for key, value in row.initial_tensors.items()}
            self.targets_raw = {
                name: np.array([0.0])
                for name in (*row.initial_task_values, *row.initial_cost_values)
            }
            self.target_masks = {name: np.array([True]) for name in self.targets_raw}

        def __len__(self):
            return 1

    monkeypatch.setattr(training, "load_labor_anchor", lambda *args: Anchor())
    anchor_path = tmp_path / "anchor.npz"
    anchor_path.write_bytes(b"synthetic anchor unit fixture")
    args = [
        "train",
        "--checkpoint",
        str(checkpoint),
        "--rollout",
        str(rollout),
        "--anchor",
        str(anchor_path),
        "--out",
        str(out),
    ]
    with pytest.raises(ValueError, match="cannot perform an actor update"):
        training.main([*args, "--calibration-report", str(tmp_path / "gate.json")])
    training.main(args)
    updated = torch.load(out, weights_only=True)
    assert all(torch.equal(value, updated["actor"][key]) for key, value in state["actor"].items())
    assert updated["actor_optimizer"] == state["actor_optimizer"]
    assert any(
        not torch.equal(value, updated["task_critic"][key])
        for key, value in state["task_critic"].items()
    )
    assert updated["last_update"]["criticWarmup"]["readiness"] is False
    assert updated["last_update"]["actor"]["reason"] == "critic-warmup-requires-fresh-calibration"
    _, first, first_task, _, _ = training._models_from_checkpoint(updated)
    _, second, second_task, _, _ = training._models_from_checkpoint(updated)
    assert next(first.parameters()).data_ptr() != next(second.parameters()).data_ptr()
    with torch.no_grad():
        next(first_task.parameters()).add_(1)
    assert not torch.equal(next(first_task.parameters()), next(second_task.parameters()))


def test_anchor_producer_keeps_parent_and_successor_raw_channels(monkeypatch, tmp_path):
    from anthropocene_rl.autonomous_v7 import labor_anchor_producer as producer

    actor, task, costs, row, _, _ = _collect_one(monkeypatch, False)
    village = dict(_village(1), camp=7, terminated=False)
    normalizer = Normalizer.fit_bounded(observation_batch([village]))
    step = {"turn": 3, "villages": [village]}

    class Bridge:
        def reset(self, *args, **kw):
            return step

        def runtime_identity(self):
            return {"fixture": True}

        def fork(self):
            return "fork", step

        def restore(self, fork):
            return step

        def drop(self, fork):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        producer, "_models_from_checkpoint", lambda state: (None, actor, task, costs, normalizer)
    )
    monkeypatch.setattr(producer, "labor_checkpoint_personality", lambda state: LaborPersonality())

    def continuation(**kw):
        # Distinct successor features/privilege for the paired branch; two terms
        # close at a real fixture terminal, without any critic return target.
        index = 9 if kw["alternative"] == "sampled-policy" else 10
        raw = {k: v[0].numpy() for k, v in _raw(1, index).items()}
        inputs = {k: v[0].numpy() for k, v in row.initial_tensors.items()}
        return producer.LaborContinuationPath(
            50000,
            kw["path_index"],
            kw["alternative"],
            kw["schedule"],
            {7: inputs},
            {7: str(index)},
            {7: {name: (1.0, 2.0) for name in producer.LABOR_VALUE_HEADS}},
            {7: (20.0, 20.0)},
            {7: (False, True)},
            {7: raw},
        )

    monkeypatch.setattr(producer, "_run_continuation", continuation)
    source = tmp_path / "source.pt"
    source.write_bytes(b"synthetic source")
    arrays = producer.collect_labor_anchor_arrays(
        checkpoint_path=source,
        state={},
        world_seeds=[50000],
        optimizer_seed=1,
        continuation_seed_base=80000,
        role="train-anchor",
        path_count=2,
        bridge_factory=Bridge,
    )
    np.testing.assert_array_equal(arrays["privileged_core"], _raw(1, 1)["privileged_core"])
    assert len(arrays["rank_privileged_core"]) == 4
    np.testing.assert_array_equal(
        arrays["rank_privileged_core"][0], _raw(1, 9)["privileged_core"][0]
    )
    path = tmp_path / "anchor.npz"
    np.savez(path, **arrays)
    anchor = load_labor_anchor(path, "train-anchor")
    loss, _ = labor_anchor_loss(task, costs, anchor, np.array([0]))
    assert torch.isfinite(loss)
