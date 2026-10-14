import copy

import pytest
import torch
from test_autonomous_direct_control_v6 import v6_config, v6_wire

from anthropocene_rl.autonomous_v5.causal import protected_state
from anthropocene_rl.autonomous_v6.contracts import observation
from anthropocene_rl.autonomous_v6.demographics import summarize
from anthropocene_rl.autonomous_v6.policy import inputs_with_mode
from anthropocene_rl.autonomous_v6.readiness import fit_critic, metrics, ready
from anthropocene_rl.policies.autonomous_ppo_v6 import AutonomousPPOV6
from anthropocene_rl.policies.autonomous_ppo_v7 import AutonomousPPOV7
from anthropocene_rl.training.autonomous_ppo_v7 import AutonomousTrainerV7


def config():
    return dict(
        v6_config(),
        model_version="autonomous-ppo-model-v7",
        value_normalization="fixed-train-affine",
        concentration_floor=0.1,
    )


def batch():
    wires = []
    for i in range(8):
        w = v6_wire()
        w["Context"][0] = i
        wires.append(observation(w))
    return inputs_with_mode(wires)


def test_legacy_default_floor_matches_old_actions_logp_values_rng_exactly():
    torch.manual_seed(19)
    old = AutonomousPPOV6(16, critic_variant="residual_layernorm")
    new = AutonomousPPOV7(16, critic_variant="residual_layernorm")
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert set(missing) == {"value_offset", "value_scale"} and not unexpected
    a, b = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    left, right = old(batch(), generator=a), new(batch(), generator=b)
    for field in ["workforce", "activities", "controls", "movement", "trade"]:
        assert torch.equal(getattr(left[0], field), getattr(right[0], field))
    assert all(torch.equal(x, y) for x, y in zip(left[1:], right[1:], strict=True))
    assert torch.equal(a.get_state(), b.get_state())


def test_lower_simplex_floor_scores_actual_samples_and_expands_mean_without_changing_beta():
    model = AutonomousPPOV7(16, concentration_floor=0.1)
    raw = torch.full((1, 214), -12.0)
    raw[0, 0] = 12
    actual = model.concentrations(raw)
    expected = torch.nn.functional.softplus(raw.clamp(-12, 12)) + 0.5
    assert torch.equal(actual[:, 208:], expected[:, 208:])
    assert float(actual[0, 0] / actual[0, :13].sum()) > 0.90
    x = batch()
    a, lp, entropy, value = model(x, generator=torch.Generator().manual_seed(4))
    _, scored, _, _ = model(x, a)
    assert torch.equal(lp, scored)
    assert torch.isfinite(torch.cat([lp, entropy, value])).all()
    (-scored.mean()).backward()
    assert model.heads["simplex"].weight.grad.norm() > 0


def test_actor_frozen_scaled_critic_protects_populated_adam_probabilities_and_rng():
    torch.set_num_threads(1)
    tr = AutonomousTrainerV7(config(), 31, {}, width=16)
    sum(p.square().mean() for p in tr.model.parameters() if p.requires_grad).backward()
    tr.optimizer.step()
    tr.optimizer.zero_grad(set_to_none=True)
    x = batch()
    y = torch.linspace(20, 400, 8)
    before = protected_state(tr)
    action, lp, _, _ = tr.model(x, generator=torch.Generator().manual_seed(41))
    tr.model.fit_value_scale(y)
    assert float(tr.model.value_offset) == float(y.mean())
    report = fit_critic(tr, (x, y), (x, y), original_mse=1e6, max_epochs=2)
    assert protected_state(tr) == before
    _, afterward, _, _ = tr.model(x, action)
    assert torch.equal(lp, afterward)
    assert report["protected_before"] == report["protected_after"]
    assert torch.isfinite(tr.model.value(x)).all()


def test_strict_critic_gate_uses_actual_variance_and_raw_bias():
    tr = AutonomousTrainerV7(config(), 31, {}, width=16)
    x = batch()
    y = torch.arange(8, dtype=torch.float32)
    actual = metrics(tr, (x, y), (x, y))
    assert actual["mse_over_return_variance"] == pytest.approx(actual["mse"] / 5.25)
    passed = dict(
        actual, explained_variance=0.7, mse_over_return_variance=0.3, mean_value_bias=1.9, mse=1.0
    )
    assert ready(passed, 2)
    for field, value in [
        ("explained_variance", 0.59),
        ("mse_over_return_variance", 0.41),
        ("mean_value_bias", -2.01),
    ]:
        assert not ready(dict(passed, **{field: value}), 2)
    with pytest.raises(ValueError):
        metrics(tr, (x, y), (x, torch.ones(8)))


def test_history_masked_forward_backward_isolation():
    model = AutonomousPPOV7(16)
    x = batch()
    with torch.no_grad():
        model.actor[0].history_weight.fill_(0.1)
    left = dict(x, x=x["x"].clone())
    right = dict(x, x=x["x"].clone())
    left["x"][:, 761:] = 0
    right["x"][:, 761:] = 0
    assert torch.equal(model.actor(left["x"]), model.actor(right["x"]))
    model.actor(left["x"]).square().sum().backward()
    assert torch.count_nonzero(model.actor[0].history_weight.grad) == 0
    x["x"][:, 761:] = 1
    assert not torch.equal(model.actor(x["x"]), model.actor(left["x"]))


def test_demographic_decomposition_distinguishes_native_outflows_and_focal_welfare():
    trace = []
    for start, end, growth, deaths, incoming, outgoing, daughters, qol in [
        (100, 90, 8, 3, 2, 5, 12, 0.5),
        (90, 93, 6, 1, 0, 2, 0, 0.9),
    ]:
        staffing = [0] * 13
        staffing[0] = start
        trace.append(
            dict(
                population_start=start,
                population_end=end,
                qol=qol,
                elapsed=20,
                reward=start * qol * 20,
                staffing=staffing,
                requested_workforce=[1] + [0] * 12,
                requested_controls=[0.2, 0, 0],
                demographics=dict(
                    SchemaVersion=1,
                    OpeningPopulation=start,
                    EndingPopulation=end,
                    ResidentGrowth=growth,
                    StarvationDeaths=deaths,
                    Immigrants=incoming,
                    Emigrants=outgoing,
                    FissionHeadcount=daughters,
                    OtherResidentChange=0,
                    FissionByTrigger={"Crowding": daughters, "Distress": 0},
                    FoodReserveYears=2,
                    MeanFieldIntensity=0.4,
                ),
            )
        )
    result = summarize(trace)
    assert result["person_years"] == 3800
    assert result["raw_welfare"] == 2620
    assert result["person_year_weighted_qol"] == pytest.approx(2620 / 3800)
    assert result["outflows"]["StarvationDeaths"] == 4
    assert result["outflows"]["Emigrants"] == 7
    assert result["outflows"]["FissionHeadcount"] == 12
    assert result["population_accounting_residual"] == 0
    malformed = copy.deepcopy(trace)
    malformed[0]["demographics"]["StarvationDeaths"] += 1
    with pytest.raises(ValueError):
        summarize(malformed)


def test_model7_restore_checks_numeric_semantics_and_rollout_mixture_schedule(tmp_path):
    cfg = dict(config(), uniform_mix_schedule=[[0, 0.05], [100, 0.01]])
    tr = AutonomousTrainerV7(cfg, 41, {"source": "test"}, width=16)
    tr.model.fit_value_scale(torch.tensor([10.0, 50.0, 200.0]))
    tr.transitions = 100
    tr.set_rollout_mix()
    assert tr.model.uniform_mix == 0.01
    path = tmp_path / "model.pt"
    tr.checkpoint(path)
    new = AutonomousTrainerV7(cfg, 5, tr.provenance, width=16)
    new.restore(path)
    assert new.model.uniform_mix == 0.01
    assert all(torch.equal(v, new.model.state_dict()[k]) for k, v in tr.model.state_dict().items())
    state = torch.load(path, weights_only=False)
    state["concentration_floor"] = 0.5
    with pytest.raises(ValueError):
        AutonomousTrainerV7.validate_checkpoint(state, cfg, tr.provenance)


def test_fixed_numeric_scale_floor_and_deadline_preserve_actor_flags():
    tr = AutonomousTrainerV7(config(), 41, {}, width=16)
    x, y = batch(), torch.ones(8) * 10
    tr.model.fit_value_scale(y)
    assert float(tr.model.value_scale) == 1.0
    flags = [p.requires_grad for p in tr.model.actor.parameters()]
    before = protected_state(tr)
    with pytest.raises(TimeoutError):
        fit_critic(tr, (x, y), (x, torch.arange(8).float()), original_mse=1e6, deadline=0)
    assert flags == [p.requires_grad for p in tr.model.actor.parameters()]
    assert protected_state(tr) == before


def test_coverage_reset_restores_original_tail_and_only_clears_live_critic_adam():
    from anthropocene_rl.autonomous_v6.coverage import reset_critic_for_collection

    tr = AutonomousTrainerV7(config(), 41, {}, width=16)
    sum(p.square().mean() for p in tr.model.parameters() if p.requires_grad).backward()
    tr.optimizer.step()
    tr.optimizer.zero_grad(set_to_none=True)
    tr.model.fit_value_scale(torch.tensor([100.0, 500.0, 900.0]))
    before = protected_state(tr)
    report = reset_critic_for_collection(tr, batch())
    assert protected_state(tr) == before
    assert report["reference_value_bit_exact"]
    assert float(tr.model.value_offset) == 0 and float(tr.model.value_scale) == 1
    assert all(p in tr.optimizer.state for p in tr.model.actor.parameters())
    assert all(p not in tr.optimizer.state for p in tr.model.critic.parameters())


def test_scaled_low_floor_ppo_updates_and_exact_checkpoint_continuation(tmp_path):
    torch.set_num_threads(1)
    cfg = dict(config(), epochs=1, minibatch=4)
    tr = AutonomousTrainerV7(cfg, 41, {}, width=16)
    x = batch()
    tr.model.fit_value_scale(torch.linspace(50, 500, 8))
    with torch.no_grad():
        action, lp, _, value = tr.model(x, generator=tr.sampler)
    rows = [
        dict(
            obs={k: v[i : i + 1] for k, v in x.items()},
            action=action.index(torch.tensor([i])),
            logp=lp[i : i + 1],
            **{"return": float(value[i]) + i * 10},
            advantage=float(i - 3),
        )
        for i in range(8)
    ]
    before = copy.deepcopy(tr.model.state_dict())
    evidence = tr.update(rows)
    assert any(
        not torch.equal(v, tr.model.state_dict()[k])
        for k, v in before.items()
        if k.startswith("actor.")
    )
    assert torch.equal(before["value_scale"], tr.model.value_scale)
    assert all(e["actor_grad"] > 0 and e["critic_grad"] > 0 for e in evidence)
    path = tmp_path / "checkpoint.pt"
    tr.checkpoint(path)
    restored = AutonomousTrainerV7(cfg, 7, {}, width=16)
    restored.restore(path)
    assert torch.equal(tr.sampler.get_state(), restored.sampler.get_state())
    assert tr.update(rows) == restored.update(rows)
    assert all(
        torch.equal(v, restored.model.state_dict()[k]) for k, v in tr.model.state_dict().items()
    )
