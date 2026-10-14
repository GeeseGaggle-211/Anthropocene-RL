import copy
import hashlib
import json
import time

import numpy as np
import pytest
import torch
from test_autonomous_history_v5 import config, v5, wire

from anthropocene_rl.autonomous_v5.causal import protected_state
from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger, BudgetTransport
from anthropocene_rl.autonomous_v5.collection import DeadlineHistoryTransport
from anthropocene_rl.autonomous_v6.contracts import action_wire, features, observation
from anthropocene_rl.autonomous_v6.policy import inputs_with_mode, physical_controls, to_native
from anthropocene_rl.training.autonomous_ppo_v6 import AutonomousTrainerV6


def v6_wire():
    w = wire()
    w["SchemaVersion"] = 6
    w["History"].append(0)
    return w


def v6_config():
    return dict(v5(config()), model_version="autonomous-ppo-model-v6", observation_version=6)


def test_gym_schema_layout_guard_and_control_conversion_once():
    from anthropocene_rl.autonomous_v6.contracts import HISTORY_NAMES, validate_schema
    from anthropocene_rl.autonomous_v6.env import AutonomousElderEnv

    schema = dict(
        actionVersion=4,
        observationVersion=6,
        environmentVersion=6,
        replayVersion=6,
        historyWidth=301,
        featureWidth=1062,
        historyFeatures=list(HISTORY_NAMES),
        saveEnvelope="ANTHROPOCENE-13",
        controlTransform="clip((latent-.05)/.90,0,1)",
        herdRetention="post-growth",
    )
    validate_schema(schema)
    for key in schema:
        bad = dict(schema)
        bad[key] = None
        with pytest.raises(ValueError):
            validate_schema(bad)

    class Stub:
        action = None

        def schema(self):
            return schema

        def reset_world(self, *_):
            return dict(observations={"5": v6_wire()}, fingerprint="opening")

        def step_world(self, actions):
            self.action = action_wire(actions[5])
            return dict(
                result=dict(
                    Observations={"5": v6_wire()},
                    Rewards={"5": 123},
                    ElapsedYears=20,
                    Truncated=False,
                    Staffing={"5": [0] * 13},
                    Fingerprint="closing",
                    Telemetry={},
                ),
                nativeStepSeconds=0.1,
            )

    tr = Stub()
    env = AutonomousElderEnv(tr, b"unused", 5)
    f, _ = env.reset()
    assert f.shape == (1062,)
    a = dict(
        workforce=np.ones(13) / 13,
        activities=np.zeros((13, 15)),
        controls=np.array([0.6, 0, 0]),
        movement=0,
        trade=np.zeros(6),
    )
    _, reward, _, _, _ = env.step(a)
    assert reward == 123 and tr.action["SchemaVersion"] == 4
    assert tr.action["FieldExpansionFraction"] == pytest.approx((0.6 - 0.05) / 0.9)
    assert a["controls"][0] == 0.6


def test_control_endpoints_and_masked_absence_are_reachable_without_mutation():
    assert physical_controls([0.05, 0.95]).tolist() == [0, 1]
    assert 0 < physical_controls([np.nextafter(0.05, 1)])[0] < 1
    assert 0 < physical_controls([np.nextafter(0.95, 0)])[0] < 1
    for x, y in [(0, 0), (0.038462, 0), (0.05, 0), (0.5, 0.5), (0.95, 1), (0.961538, 1), (1, 1)]:
        assert physical_controls([x])[0] == pytest.approx(y)
    for x in [[np.nan], [-0.1], [1.1]]:
        with pytest.raises(ValueError):
            physical_controls(x)
    o = observation(v6_wire())
    controls = np.array([0.96, 0, 0])
    action = dict(
        workforce=np.ones(13) / 13,
        activities=np.zeros((13, 15)),
        controls=controls,
        movement=0,
        trade=np.zeros(6),
    )
    native = to_native(action, o)
    assert native["controls"].tolist() == [1, 0, 0]
    assert np.array_equal(controls, [0.96, 0, 0])
    assert action_wire(native)["SchemaVersion"] == 4


def test_semantic_discriminator_preserves_old_coordinates_and_rejects_stale_contracts():
    w = v6_wire()
    f = features(observation(w))
    assert f.shape == (1062,) and f[-1] == 0
    for value in [1, 3, 4, np.nan]:
        bad = copy.deepcopy(w)
        bad["History"][-1] = value
        with pytest.raises(ValueError):
            observation(bad)
    with pytest.raises(ValueError):
        observation(wire())
    obs = observation(w)
    a, b = inputs_with_mode([obs]), inputs_with_mode([obs], "receipt_none")
    assert torch.equal(a["x"][:, :761], b["x"][:, :761])
    assert b["x"][:, 761:].abs().sum() == 0


def test_ppo_likelihood_is_latent_and_checkpoint_reproduces_sampler(tmp_path):
    torch.set_num_threads(1)
    trainer = AutonomousTrainerV6(v6_config(), 31, {"test": 8}, width=16)
    obs = inputs_with_mode([observation(v6_wire()) for _ in range(8)])
    action, lp, _, _ = trainer.model(obs, generator=trainer.sampler)
    _, repeated, _, _ = trainer.model(obs, action)
    assert torch.equal(lp, repeated)
    for i in range(8):
        to_native(action.wire_action(i), observation(v6_wire()))
    # Mapping receipts must not replace the latent tensor used by PPO.
    _, after, _, _ = trainer.model(obs, action)
    assert torch.equal(lp, after) and torch.equal((after - lp).exp(), torch.ones_like(lp))
    trainer.config["feature_mode"] = "receipt_none"
    path = tmp_path / "new.pt"
    trainer.checkpoint(path)
    clone = AutonomousTrainerV6(trainer.config, 0, trainer.provenance, width=16)
    clone.restore(path)
    assert protected_state(clone) == protected_state(trainer)
    assert torch.equal(
        trainer.model(obs, generator=trainer.sampler)[0].controls,
        clone.model(obs, generator=clone.sampler)[0].controls,
    )
    from anthropocene_rl.policies.autonomous_inference_v6 import AutonomousElderPolicyV6

    policy = AutonomousElderPolicyV6(path, expected_provenance=trainer.provenance, seed=41)
    assert policy.act(v6_wire())["controls"].shape == (3,)
    for key, value in [
        ("model_version", "autonomous-ppo-model-v5"),
        ("observation_version", 5),
        ("feature_mode", "unknown"),
    ]:
        bad = torch.load(path, weights_only=False)
        bad["config"][key] = value
        invalid = tmp_path / f"invalid-{key}.pt"
        torch.save(bad, invalid)
        with pytest.raises(ValueError):
            AutonomousElderPolicyV6(invalid, expected_provenance=trainer.provenance, seed=41)


def test_budget_charges_attempts_replay_and_rejects_exhaustion(tmp_path, monkeypatch):
    ledger = BudgetLedger(
        tmp_path / "budget.json",
        dict(training=1, training_replay=1),
        prior_resolves=1024,
        prior_wall_seconds=42,
        deadline=time.monotonic() + 60,
        native_version=6,
    )
    ledger.phase = "training"
    tr = object.__new__(BudgetTransport)
    tr.ledger, tr.episode_attempts = ledger, 0
    monkeypatch.setattr(DeadlineHistoryTransport, "_request", lambda _, p: p)
    payload = tr._request(dict(op="autonomous_step", actions={"5": {"SchemaVersion": 3}}))
    assert payload["op"] == "autonomous6_step" and payload["actions"]["5"]["SchemaVersion"] == 4
    tr._request(dict(op="autonomous_verify_replay"))
    with pytest.raises(RuntimeError):
        tr._request(dict(op="autonomous_step"))
    assert json.loads(ledger.path.read_text())["total_resolves"] == 1026
    ledger.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        ledger.reserve("training", 0)


def test_real_parent_conversion_preserves_inherited_tensors_adam_rng_anchors():
    from pathlib import Path

    parent = Path(
        "runs/autonomous-critic-horizon-gate-v5/residual_layernorm-h100-seed29-step-4096.pt"
    )
    if not parent.exists():
        pytest.skip("Historical parent artifact unavailable")
    state = torch.load(parent, map_location="cpu", weights_only=False)
    cfg = dict(
        state["config"],
        version="autonomous-ppo-direct-control-v6",
        model_version="autonomous-ppo-model-v6",
        observation_version=6,
    )
    trainer = AutonomousTrainerV6(cfg, 0, {"test": "parent"}, width=state["width"])
    trainer.warm_start(parent, expected_sha256=hashlib.sha256(parent.read_bytes()).hexdigest())
    assert all(torch.equal(v, trainer.model.state_dict()[k]) for k, v in state["model"].items())
    assert torch.equal(trainer.sampler.get_state(), state["sampler"])
    assert torch.equal(trainer.shuffle.get_state(), state["shuffle"])
    assert trainer.transitions == state["transitions"]
    assert all(
        p.shape[-1] == 301 and p.count_nonzero() == 0
        for n, p in trainer.model.named_parameters()
        if n.endswith("history_weight")
    )
    parameters = dict(trainer.model.named_parameters())
    names = [n for n, p in parameters.items() if p in trainer.optimizer.state]
    assert names and trainer.optimizer.state[parameters["actor.0.weight"]]["step"] > 0


def test_intercept_fit_uses_train_only_and_preserves_all_other_state():
    from anthropocene_rl.autonomous_v6.calibration import fit_intercept, other_state

    trainer = AutonomousTrainerV6(v6_config(), 9, {"test": "intercept"}, width=16)
    x = inputs_with_mode([observation(v6_wire()) for _ in range(16)])
    x["x"][:, 0] = torch.linspace(-1, 1, 16)
    loss = trainer.model.value(x).square().mean()
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    target = trainer.model.value(x).detach() + 3 + torch.linspace(-1, 1, 16)
    before = other_state(trainer)
    report = fit_intercept(trainer, (x, target))
    assert before == other_state(trainer) == report["protected_state_after"]
    assert report["delta"] == pytest.approx(3)
    assert report["train_residual_after"] == pytest.approx(0, abs=1e-6)
    assert report["train_mse_before"] - report["train_mse_after"] == pytest.approx(9, abs=1e-5)
    assert report["native_resolves"] == report["optimizer_steps"] == 0


@pytest.mark.parametrize("dead", [False, True])
def test_native_episode_to_gae_and_mechanical_ppo_update(tmp_path, dead):
    from anthropocene_rl.autonomous_v5.causal import episode

    class NativeStub:
        steps = 0

        def reset_world(self, *_):
            return dict(observations={"1": v6_wire()})

        def step_world(self, actions):
            self.steps += 1
            h = np.zeros(301)
            h[0] = 1
            return dict(
                nativeStepSeconds=0,
                result=dict(
                    Observations={} if self.steps == 2 and dead else {"1": v6_wire()},
                    Truncated=self.steps == 2 and not dead,
                    ElapsedYears=20.0,
                    Rewards={"1": 1600.0 + self.steps * 100},
                    CompletedHistory={"1": h.tolist()},
                    Staffing={"1": [8] + [0] * 12},
                    Fingerprint="unit-stub",
                ),
            )

    torch.set_num_threads(1)
    trainer = AutonomousTrainerV6(
        dict(v6_config(), episode_limit=2, horizon_years=400, require_core_action_faces=False),
        9,
        {"test": "gae"},
        width=16,
    )
    blob = b"unit source"
    (tmp_path / "source.snapshot").write_bytes(blob)
    entry = dict(
        snapshot="source.snapshot",
        sha256=hashlib.sha256(blob).hexdigest(),
        focal=1,
        peer=None,
        id="unit",
        family="survival",
        world_seed=1,
        population=8,
    )
    rows, summary = episode(
        trainer.model,
        NativeStub(),
        tmp_path,
        entry,
        torch.Generator().manual_seed(17),
        trainer.config,
    )
    assert len(rows) == summary["steps"] == 2
    assert rows[-1]["truncated"] == (not dead) and rows[-1]["terminal"] == dead
    assert all(np.isfinite(r["advantage"]) and np.isfinite(r["return"]) for r in rows)
    actor_before = trainer.model.actor[0].weight.detach().clone()
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_autonomous_continuation_v4 import measured_update

    metrics = measured_update(trainer, rows)
    assert trainer.transitions == 2 and trainer.updates == 1 and metrics
    assert not torch.equal(actor_before, trainer.model.actor[0].weight)
