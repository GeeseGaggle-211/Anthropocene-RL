"""labor_compose_checkpoint on synthetic trainer states (no native DLL, no real artifacts)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from test_labor_a_ppo_wiring import _a_payload, _default_models, _trainer_state

from anthropocene_rl.autonomous_v7 import labor_a_ppo as ap
from anthropocene_rl.autonomous_v7 import labor_compose_checkpoint as compose
from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7.labor_value_calibration import labor_critic_digest


def _states():
    actor, task, costs, normalizer, layout = _default_models(seed=1)
    base = _trainer_state(actor, task, costs, normalizer, layout)
    base["critic_training_world_seeds"] = [50100, 50101]
    base["critic_optimizer"] = torch.optim.Adam(
        [*task.parameters(), *costs.parameters()], lr=3e-4
    ).state_dict()
    s_actor, s_task, s_costs, s_norm, s_layout = _default_models(seed=2)
    source = _trainer_state(s_actor, s_task, s_costs, s_norm, s_layout)
    # A non-fresh actor optimizer in the base, to see it rebuilt.
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    for p in actor.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    base["actor_optimizer"] = optimizer.state_dict()
    return base, source, (task, costs)


def test_compose_replaces_only_the_actor_and_rebuilds_adam():
    base, source, (task, costs) = _states()
    composed = compose.compose_trainer_checkpoint(base, source, actor_lr=3e-4)
    for name, value in source["actor"].items():
        assert torch.equal(composed["actor"][name], value)
    assert not all(torch.equal(composed["actor"][n], v) for n, v in base["actor"].items())
    groups = composed["actor_optimizer"]["param_groups"]
    assert composed["actor_optimizer"]["state"] == {} and groups[0]["lr"] == 3e-4
    _, fresh_actor, *_ = training._models_from_checkpoint(composed)
    assert len(groups[0]["params"]) == len(list(fresh_actor.parameters()))
    for key in base:
        if key not in ("actor", "actor_optimizer"):
            assert compose._equal(composed[key], base[key]), key
    assert composed[compose.COMPOSE_KEY]["schema"] == compose.COMPOSE_SCHEMA
    training._require_trainer_checkpoint(composed)
    _, _, c_task, c_costs, _ = training._models_from_checkpoint(composed)
    assert labor_critic_digest(c_task, c_costs) == labor_critic_digest(task, costs)
    report = compose.verify_composed(
        composed, base, source, a_warm_start=_a_payload(task, costs),
        check_worlds=[51000, 51001, 51002, 51003],
    )
    assert report["criticDigestUnchanged"] and report["actorEqualsSource"]
    assert report["behaviorPolicyDigestEqualsSource"] and report["otherKeysUnchanged"]
    assert report["collectWorlds"]["worlds"] == [51000, 51001, 51002, 51003]


def test_compose_refuses_contract_normalizer_and_layout_differences():
    base, source, _ = _states()
    other = dict(source, labor_model_contract=dict(
        source["labor_model_contract"], personality={"qol_years": 0.9, "population_years": 0.1}
    ))
    with pytest.raises(ValueError, match="labor_model_contract differs"):
        compose.compose_trainer_checkpoint(base, other)
    normalizer = dict(source["normalizer"])
    normalizer["core_mean"] = [value + 1.0 for value in normalizer["core_mean"]]
    with pytest.raises(ValueError, match="normalizer differs"):
        compose.compose_trainer_checkpoint(base, dict(source, normalizer=normalizer))
    actor = dict(source["actor"])
    name = next(k for k, v in actor.items() if v.is_floating_point() and v.ndim == 2)
    actor[name] = torch.zeros(actor[name].shape[0] + 1, actor[name].shape[1])
    with pytest.raises(ValueError, match="shape/dtype"):
        compose.compose_trainer_checkpoint(base, dict(source, actor=actor))
    actor = dict(source["actor"])
    actor.pop(name)
    with pytest.raises(ValueError, match="keys"):
        compose.compose_trainer_checkpoint(base, dict(source, actor=actor))


def test_verify_refuses_reserved_sealed_and_critic_worlds():
    base, source, _ = _states()
    composed = compose.compose_trainer_checkpoint(base, source)
    with pytest.raises(ValueError, match="reserved held-out"):
        compose.verify_composed(composed, base, source, check_worlds=[90154])
    with pytest.raises(Exception, match="denied"):
        compose.verify_composed(composed, base, source, check_worlds=[3001])
    with pytest.raises(ValueError, match="critic-training worlds"):
        compose.verify_composed(composed, base, source, check_worlds=[50100])
    with pytest.raises(ValueError, match="critic digest"):
        compose.verify_composed(composed, base, source, expect_critic_digest="0" * 16)


def test_cli_writes_a_checkpoint_the_loaders_accept(tmp_path):
    base, source, (task, costs) = _states()
    paths = {name: tmp_path / f"{name}.pt" for name in ("base", "source", "warm")}
    torch.save(base, paths["base"])
    torch.save(source, paths["source"])
    torch.save(_a_payload(task, costs), paths["warm"])
    out, report = tmp_path / "start.pt", tmp_path / "report.json"
    *_, normalizer = training._models_from_checkpoint(base)
    norm = training._normalization_digest(normalizer)
    assert compose.main([
        "--base", str(paths["base"]), "--actor-from", str(paths["source"]), "--out", str(out),
        "--expect-critic-digest", labor_critic_digest(task, costs)[:16],
        "--expect-normalizer-digest", norm[:8], "--a-warm-start", str(paths["warm"]),
        "--check-worlds", "51000,51001", "--report", str(report),
    ]) == 0
    written = torch.load(out, weights_only=True)
    training._require_trainer_checkpoint(written)
    _, actor, *_ = training._models_from_checkpoint(written)
    assert all(torch.equal(v, source["actor"][k]) for k, v in actor.state_dict().items())
    frozen = ap.load_frozen_a(
        torch.load(paths["warm"], weights_only=True), torch.load(paths["base"], weights_only=True),
        normalization_binding_digest=norm,
    )
    assert frozen.critic_digest == labor_critic_digest(task, costs)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["outSha256"] == training.checkpoint_hash(out)
    assert data["provenance"]["actorSourceSha256"] == training.checkpoint_hash(paths["source"])
    sampling = torch.Generator()
    sampling.set_state(written["sampling_generator_state"])
    assert np.isfinite(float(torch.rand(1, generator=sampling)))
