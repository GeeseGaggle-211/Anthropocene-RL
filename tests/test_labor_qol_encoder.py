"""Separate QoL encoder for the r5 task critic (contract field ``qolEncoder``).

The shared (historical) path is checked bit for bit against a17abc5: the golden
digests below are the pre-change source's values on the same fixtures (RNG-only
quantities, so platform-stable). The forward pass and a monitored, weighted,
privileged fit were also compared bitwise against a17abc5 when this was written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from test_labor_critic_support import _fixture
from test_labor_model import _labor_tensors
from test_labor_privileged_bridge import _normalizer, _raw
from test_labor_privileged_model import _layout, _payload
from test_labor_qol_head import _training_dataset

from anthropocene_rl.autonomous_v7 import labor_advantage as la
from anthropocene_rl.autonomous_v7 import labor_training as lt
from anthropocene_rl.autonomous_v7.labor_bc import AutoQoLLossWeight, pretrain_critic_on_dataset
from anthropocene_rl.autonomous_v7.labor_critic_support import (
    convert_privileged_critics,
    convert_qol_encoder,
    convert_task_critic_support,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    LABOR_OBJECTIVES,
    LaborActor,
    LaborCostCritics,
    LaborModelContract,
    LaborPersonality,
    LaborQoLHead,
    LaborTaskCritic,
    LaborValueSupport,
    bind_privileged_normalizer,
    critic_privileged_batch,
    labor_checkpoint_payload,
    labor_checkpoint_qol_encoder,
    labor_qol_encoder_seed,
    load_labor_checkpoint_models,
)
from anthropocene_rl.autonomous_v7.labor_ppo import labor_task_value_losses
from anthropocene_rl.autonomous_v7.labor_stability import StabilityConfig
from anthropocene_rl.autonomous_v7.labor_value_calibration import labor_critic_digest


def _digest(tensors) -> str:
    h = hashlib.sha256()
    for key in sorted(tensors):
        value = tensors[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()[:24]


def _equal(a, b) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _task(mode: str = "separate", *, privileged: bool = True, seed: int = 11, **kwargs):
    torch.manual_seed(seed)
    return LaborTaskCritic(
        _layout(privileged),
        width=16,
        token_width=16,
        qol_head=LaborQoLHead(**kwargs),
        qol_encoder=mode,
    )


def _pair(mode: str = "separate", *, privileged: bool = True, seed: int = 0, **kwargs):
    task = _task(mode, privileged=privileged, seed=seed, **kwargs)
    costs = LaborCostCritics(_layout(privileged), width=16, token_width=16)
    if privileged:
        bind_privileged_normalizer(_normalizer(), task, costs)
    return task, costs


def _dataset(privileged: bool = True) -> dict:
    data = _training_dataset()
    if privileged:
        data.update(_raw(6))
    data["collection"] = dict(data["collection"], dataset_digest="qol-encoder")
    return data


# --- 1. old contract: bitwise ------------------------------------------------------------------

# a17abc5 values of tests/../qol_golden.py-style digests (see module docstring).
GOLDEN = {
    "after_rng": "9169e030f8ecbdd49d0cd508",
    "init": "c612acf7d1a834932d339b36",
    "plain_init": "06e3ae771bb0214158225435",
    "payload_contract": "abfc56ff1955a018ea0c67db",
    "payload_task": "0fbd6fcad656540e4b696e80",
}


def test_shared_path_is_the_historical_critic_bitwise() -> None:
    torch.manual_seed(11)
    task = LaborTaskCritic(_layout(True), width=16, token_width=16, qol_head=LaborQoLHead())
    assert _digest({"r": torch.rand(5)}) == GOLDEN["after_rng"]
    assert _digest(task.state_dict()) == GOLDEN["init"]
    assert task.qol_encoder is None and not task.separate_qol_encoder
    assert not any(k.startswith("qol_encoder") for k in task.state_dict())
    torch.manual_seed(7)
    plain = LaborTaskCritic(_layout(False), width=16, token_width=16)
    assert _digest(plain.state_dict()) == GOLDEN["plain_init"]
    state, _, _ = _payload(False)
    contract = json.dumps(state["labor_model_contract"], sort_keys=True).encode()
    assert hashlib.sha256(contract).hexdigest()[:24] == GOLDEN["payload_contract"]
    assert _digest(state["task_critic"]) == GOLDEN["payload_task"]
    assert "qolEncoder" not in state["labor_model_contract"]
    assert labor_checkpoint_qol_encoder(state) == "shared"
    # Explicit "shared" is the default object, forward and fit included.
    torch.manual_seed(11)
    explicit = LaborTaskCritic(
        _layout(True), width=16, token_width=16, qol_head=LaborQoLHead(), qol_encoder="shared"
    )
    assert _equal(task.state_dict(), explicit.state_dict())


def test_shared_fit_ignores_the_new_code_paths() -> None:
    def fit(mode_kwargs):
        task, costs = _pair("shared", seed=0, target="td-lambda")
        summary = pretrain_critic_on_dataset(
            task_critic=task,
            cost_critics=costs,
            dataset=_dataset(),
            epochs=2,
            batch_size=2,
            rng=np.random.default_rng(7),
            stability=StabilityConfig(log_every=1),
            **mode_kwargs,
        )
        return {**task.state_dict(), **costs.state_dict()}, summary

    a, summary = fit({})
    b, _ = fit({"task_loss_weights": {"qol_years": 1.0}})
    assert _equal(a, b)
    assert "qol_encoder" not in summary
    assert set(summary["stability"]["checks"]["privileged_attention"]["encoders"]) == {
        "task",
        "cost:feasibility",
        "cost:settlement_guard",
    }


def test_contract_field_is_canonical() -> None:
    assert "qolEncoder" not in LaborModelContract().state_dict()
    state = LaborModelContract(qol_encoder="separate").state_dict()
    assert state["qolEncoder"] == "separate"
    assert LaborModelContract.from_state(state).qol_encoder == "separate"
    shared = dict(state, qolEncoder="shared")
    with pytest.raises(ValueError, match="qolEncoder"):
        LaborModelContract.from_state(shared)
    with pytest.raises(ValueError, match="QoL encoder"):
        LaborModelContract(qol_encoder="twin")
    with pytest.raises(ValueError, match="QoL encoder"):
        _task("twin")


# --- 2. new contract: initialization ------------------------------------------------------------


def test_separate_init_is_deterministic_and_leaves_the_historical_stream() -> None:
    shared = _task("shared")
    after_shared = torch.rand(4)
    separate = _task("separate")
    after_separate = torch.rand(4)
    assert torch.equal(after_shared, after_separate)  # global RNG stream unchanged
    reference = shared.state_dict()
    keys = list(separate.state_dict())
    assert keys[: len(reference)] == list(reference)  # historical keys first, same order
    for key, value in separate.state_dict().items():
        if key.startswith("qol_encoder."):
            continue
        assert torch.equal(value, reference[key]), key
    again = _task("separate")
    assert _equal(separate.state_dict(), again.state_dict())
    # A full copy of the task encoder architecture, privileged path included, own tensors.
    qol = {k[len("qol_encoder.") :]: v for k, v in separate.state_dict().items()
           if k.startswith("qol_encoder.")}
    enc = separate.encoder.state_dict()
    assert qol.keys() == enc.keys() and any("privileged_" in k for k in qol)
    assert all(qol[k].shape == enc[k].shape for k in qol)
    assert not all(torch.equal(qol[k], enc[k]) for k in qol)
    assert {p.data_ptr() for p in separate.qol_encoder.parameters()}.isdisjoint(
        p.data_ptr() for p in separate.encoder.parameters()
    )
    # Seed-dependent; an explicit seed is independent of the global stream.
    other = _task("separate", seed=12)
    assert not torch.equal(
        other.state_dict()["qol_encoder.village.weight"], qol["village.weight"]
    )
    torch.manual_seed(1)
    x = LaborTaskCritic(_layout(True), 16, 16, qol_encoder="separate", qol_encoder_seed=5)
    torch.manual_seed(2)
    y = LaborTaskCritic(_layout(True), 16, 16, qol_encoder="separate", qol_encoder_seed=5)
    assert _equal(x.qol_encoder.state_dict(), y.qol_encoder.state_dict())


def test_parameter_count() -> None:
    shared, separate = _task("shared"), _task("separate")
    n_shared = sum(p.numel() for p in shared.parameters())
    n_separate = sum(p.numel() for p in separate.parameters())
    assert n_separate - n_shared == sum(p.numel() for p in shared.encoder.parameters())


# --- 3. forward, backward, isolation ------------------------------------------------------------


def _inputs(task):
    tensors = _labor_tensors(batch=3)
    return tensors, critic_privileged_batch(task, _raw(3))


def test_qol_head_reads_only_its_encoder_and_population_only_the_task_encoder() -> None:
    task, _ = _pair("separate", target="monte-carlo")
    tensors, privileged = _inputs(task)
    out = task(tensors, privileged)
    assert out["qol_years"].shape == (3, 1) and out["population_years"].shape == (3, 255)
    targets = {
        "qol_years": torch.full((3,), 10.0),
        "population_years": torch.full((3,), 4.0),
    }
    _, heads, _ = labor_task_value_losses(task, tensors, targets, privileged)
    qol_grads = torch.autograd.grad(
        heads["qol_years"].mean(), list(task.parameters()), allow_unused=True, retain_graph=True
    )
    pop_grads = torch.autograd.grad(
        heads["population_years"].mean(), list(task.parameters()), allow_unused=True
    )
    names = [n for n, _ in task.named_parameters()]

    def touched(grads):
        return {n for n, g in zip(names, grads, strict=True) if g is not None and g.abs().sum() > 0}

    qol_touched, pop_touched = touched(qol_grads), touched(pop_grads)
    assert qol_touched and all(
        n.startswith(("qol_encoder.", "heads.qol_years.")) for n in qol_touched
    )
    assert any(n.startswith("qol_encoder.privileged_") for n in qol_touched)
    assert pop_touched and all(
        n.startswith(("encoder.", "heads.population_years.")) for n in pop_touched
    )
    # Population-years does not depend on the QoL encoder (and vice versa).
    with torch.no_grad():
        before = task(tensors, privileged)
        for p in task.qol_encoder.parameters():
            p.add_(0.5 * torch.randn_like(p))
        after = task(tensors, privileged)
    assert torch.equal(before["population_years"], after["population_years"])
    assert not torch.equal(before["qol_years"], after["qol_years"])
    with torch.no_grad():
        for p in task.encoder.parameters():
            p.add_(0.5 * torch.randn_like(p))
        last = task(tensors, privileged)
    assert torch.equal(after["qol_years"], last["qol_years"])


def test_full_backward_reaches_both_encoders_and_is_finite() -> None:
    task, _costs = _pair("separate", target="td-lambda")
    tensors, privileged = _inputs(task)
    targets = {
        "qol_years": torch.full((3,), 10.0),
        "population_years": torch.full((3,), 4.0),
    }
    loss, _, _ = labor_task_value_losses(task, tensors, targets, privileged)
    loss.backward()
    for encoder in (task.encoder, task.qol_encoder):
        grads = [p.grad for p in encoder.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


# --- 4. training: monitor, weights, resume ------------------------------------------------------


def _fit(mode="separate", **kwargs):
    task, costs = _pair(mode, target=kwargs.pop("target", "td-lambda"))
    summary = pretrain_critic_on_dataset(
        task_critic=task,
        cost_critics=costs,
        dataset=kwargs.pop("dataset", _dataset()),
        epochs=kwargs.pop("epochs", 3),
        batch_size=2,
        rng=np.random.default_rng(7),
        **kwargs,
    )
    return task, costs, summary


def test_monitored_separate_fit_blocks_probes_and_summary(tmp_path) -> None:
    log = tmp_path / "s.jsonl"
    task, _, summary = _fit(stability=StabilityConfig(log_every=1, jsonl=str(log)))
    stability = summary["stability"]
    checks = stability["checks"]
    assert checks["finite_updates"]["status"] == "PASS"
    assert checks["gradient_dominance"]["status"] in ("PASS", "FAIL")
    assert summary["qol_encoder"]["mode"] == "separate"
    steps = [json.loads(line) for line in log.read_text().splitlines()]
    steps = [s for s in steps if s["kind"] == "step"]
    assert steps and all("qol_encoder" in s["grad_norm_preclip"] for s in steps)
    assert all(s["grad_norm_preclip"]["qol_encoder"] > 0 for s in steps)
    assert all(s["shared_head_grad_norm"] == {} for s in steps)  # no shared trunk
    assert all("global_clip" in s["contender_grad_norm"] for s in steps)
    attention = checks["privileged_attention"]["encoders"]
    dead = checks["dead_features"]["encoders"]
    assert set(attention) == {"task", "task_qol", "cost:feasibility", "cost:settlement_guard"}
    assert attention["task_qol"]["max_village_ratio"] > 0
    assert set(dead) == set(attention) and dead["task_qol"]["privileged"] is not None
    for value in task.state_dict().values():
        assert torch.isfinite(value).all()


def test_monitored_fit_is_bitwise_the_unmonitored_fit() -> None:
    plain, plain_costs, _ = _fit()
    monitored, monitored_costs, _ = _fit(stability=StabilityConfig(log_every=1))
    assert _equal(plain.state_dict(), monitored.state_dict())
    assert _equal(plain_costs.state_dict(), monitored_costs.state_dict())


def test_fixed_qol_weight_works_and_auto_is_refused(tmp_path) -> None:
    unweighted, _, _ = _fit(epochs=1)
    weighted, _, summary = _fit(
        epochs=1,
        task_loss_weights={"qol_years": 30.0},
        stability=StabilityConfig(log_every=1, jsonl=str(tmp_path / "w.jsonl")),
    )
    assert summary["task_loss_weights"]["final"]["qol_years"] == 30.0
    assert not _equal(unweighted.state_dict(), weighted.state_dict())
    with pytest.raises(ValueError, match="separate QoL encoder"):
        _fit(
            epochs=1,
            qol_loss_weight_auto=AutoQoLLossWeight(),
            stability=StabilityConfig(log_every=1),
        )


def test_dominance_excess_loss_is_accepted_and_reported_ineffective() -> None:
    with pytest.warns(UserWarning, match="no effect with a separate QoL encoder"):
        _, _, summary = _fit(
            epochs=1, stability=StabilityConfig(log_every=1, dominance_excess_loss=True)
        )
    assert summary["qol_encoder"]["dominance_excess_loss_effective"] is False
    assert summary["final_qol_excess_bce"] >= -1e-6


def test_exact_resume_matches_the_uninterrupted_fit(tmp_path) -> None:
    reference, _, _ = _fit(stability=StabilityConfig(log_every=1))
    path = str(tmp_path / "resume.pt")
    _fit(stability=StabilityConfig(log_every=1), max_steps=4, resume_path=path, resume_every=2)
    resumed, _, summary = _fit(
        stability=StabilityConfig(log_every=1), resume_path=path, resume_every=2
    )
    assert _equal(reference.state_dict(), resumed.state_dict())
    assert summary["stability"]["steps"] == 9
    state = torch.load(path, weights_only=True)
    assert any(k.startswith("qol_encoder.") for k in state["task"])
    # A shared critic cannot resume a separate fit (strict load).
    task, costs = _pair("shared", target="td-lambda")
    with pytest.raises(RuntimeError, match="qol_encoder"):
        pretrain_critic_on_dataset(
            task_critic=task,
            cost_critics=costs,
            dataset=_dataset(),
            epochs=3,
            batch_size=2,
            rng=np.random.default_rng(7),
            stability=StabilityConfig(log_every=1),
            resume_path=path,
        )


# --- 5. advantage head -------------------------------------------------------------------------


def test_advantage_head_reads_each_objectives_own_encoder(tmp_path) -> None:
    task, costs = _pair("separate", privileged=False, seed=3)
    head = la.LaborAdvantageHead.for_critics(task, costs, embedding=16, hidden=16)
    assert head.widths.context["qol_years"] == 32 and head.widths.context["population_years"] == 32
    tensors = _labor_tensors(batch=3)
    contexts = la.critic_contexts(task, costs, tensors)
    with torch.no_grad():
        task.train()
        qol = task.qol_encoder.context(tensors)
        pop = task.encoder.context(tensors)
    assert torch.equal(
        contexts["qol_years"], torch.cat([qol["hidden"], qol["privileged_readout"]], -1)
    )
    assert torch.equal(
        contexts["population_years"], torch.cat([pop["hidden"], pop["privileged_readout"]], -1)
    )
    assert not torch.equal(contexts["qol_years"], contexts["population_years"])
    shard = la.synthetic_candidate_shard(
        worlds=tuple(range(90100, 90110)), roots_per_world=6, candidates=8, seed=1
    )
    config = la.AdvantageFitConfig(epochs=20, batch_roots=8, lr=2e-3, seed=0, hidden=32)
    digest = labor_critic_digest(task, costs)
    fitted, summary = la.fit_advantage_head(task, costs, la.parse_candidate_shard(shard), config)
    assert labor_critic_digest(task, costs) == digest  # V untouched
    assert summary["loss_history"][-1] < summary["loss_history"][0]
    payload = la.advantage_checkpoint_payload(
        fitted,
        critic_digest=digest,
        config=config,
        fit=summary,
        task_context=la.task_context_record(task),
    )
    assert payload["task_context"] == la.TASK_CONTEXT_PER_OBJECTIVE
    torch.save(payload, tmp_path / "a.pt")
    state = torch.load(tmp_path / "a.pt", weights_only=True)
    restored, _ = la.load_advantage_head(state, task, costs)
    assert _equal(restored.state_dict(), fitted.state_dict())
    stripped = {k: v for k, v in state.items() if k != "task_context"}
    with pytest.raises(ValueError, match="task context"):
        la.load_advantage_head(stripped, task, costs)
    # A shared critic's payload keeps its historical keys.
    shared_task, _shared_costs = _pair("shared", privileged=False, seed=3)
    assert la.task_context_record(shared_task) is None


# --- 6. checkpoints and conversions -------------------------------------------------------------


def _trainer_state(mode: str, qol_head=None, *, seed: int = 11) -> tuple[dict, dict]:
    from anthropocene_rl.autonomous_v7.labor_training import _trainer_checkpoint

    normalizer, layout, tensors = _fixture()
    torch.manual_seed(3)
    actor = LaborActor(layout)
    task = LaborTaskCritic(
        layout, qol_head=qol_head, qol_encoder=mode, qol_encoder_seed=labor_qol_encoder_seed(seed)
    )
    costs = LaborCostCritics(layout)
    state = _trainer_checkpoint(
        actor,
        task,
        costs,
        normalizer,
        layout,
        torch.optim.Adam(actor.parameters(), lr=3e-4),
        torch.optim.Adam(list(task.parameters()) + list(costs.parameters()), lr=2e-4),
        personality=LaborPersonality(),
        optimizer_seed=seed,
        iteration=4,
        sampling_generator=torch.Generator().manual_seed(seed),
        shuffle_rng=np.random.default_rng(seed),
        critic_training_world_seeds=(90100,),
        critic_training_independent_unit_ids=(),
    )
    return state, tensors


def test_checkpoint_round_trip_and_mismatch_rejection() -> None:
    from anthropocene_rl.autonomous_v7.labor_training import _models_from_checkpoint

    state, tensors = _trainer_state("separate", LaborQoLHead())
    assert state["labor_model_contract"]["qolEncoder"] == "separate"
    assert labor_checkpoint_qol_encoder(state) == "separate"
    layout, _, task, _, _ = _models_from_checkpoint(state)
    assert task.separate_qol_encoder
    assert _equal(task.state_dict(), state["task_critic"])
    task.train()
    with torch.no_grad():
        first = task(tensors)
        again = _models_from_checkpoint(copy.deepcopy(state))[2]
        again.train()
        second = again(tensors)
    assert all(torch.equal(first[k], second[k]) for k in LABOR_OBJECTIVES)
    with pytest.raises(ValueError, match="different QoL encoder"):
        load_labor_checkpoint_models(
            state,
            LaborActor(layout),
            LaborTaskCritic(layout, qol_head=LaborQoLHead()),
            LaborCostCritics(layout),
            expected_layout=layout,
        )
    shared_state, _ = _trainer_state("shared", LaborQoLHead())
    with pytest.raises(ValueError, match="different QoL encoder"):
        load_labor_checkpoint_models(
            shared_state,
            LaborActor(layout),
            LaborTaskCritic(layout, qol_head=LaborQoLHead(), qol_encoder="separate"),
            LaborCostCritics(layout),
            expected_layout=layout,
        )
    # Smuggling encoder tensors into a shared checkpoint fails strict loading.
    smuggled = copy.deepcopy(shared_state)
    smuggled["task_critic"].update(
        {k: v for k, v in state["task_critic"].items() if k.startswith("qol_encoder.")}
    )
    with pytest.raises(RuntimeError, match="qol_encoder"):
        _models_from_checkpoint(smuggled)


def test_privileged_payload_round_trip() -> None:
    from test_labor_privileged_model import _village

    from anthropocene_rl.autonomous_v7.bridge import observation_batch
    from anthropocene_rl.autonomous_v7.labor_model import labor_policy_tensors
    from anthropocene_rl.autonomous_v7.model import Normalizer

    village = _village(1)
    normalizer = Normalizer.fit_bounded(observation_batch([village]))
    _, tensors = labor_policy_tensors([village], normalizer)
    layout = _layout(True, core=int(tensors["core"].shape[1]))
    torch.manual_seed(5)
    actor = LaborActor(layout, width=16, token_width=16)
    task = LaborTaskCritic(layout, 16, 16, qol_encoder="separate")
    costs = LaborCostCritics(layout, 16, 16)
    bind_privileged_normalizer(_normalizer(), task, costs)
    payload = labor_checkpoint_payload(actor, task, costs, normalizer, layout)
    assert payload["labor_model_contract"]["qolEncoder"] == "separate"
    restored = LaborTaskCritic(layout, 16, 16, qol_encoder="separate")
    load_labor_checkpoint_models(
        payload,
        LaborActor(layout, width=16, token_width=16),
        restored,
        LaborCostCritics(layout, 16, 16),
        expected_layout=layout,
    )
    assert _equal(restored.state_dict(), task.state_dict())
    assert restored.privileged_normalizer is not None


def test_convert_copy_is_a_warm_start_with_identical_outputs() -> None:
    from anthropocene_rl.autonomous_v7.labor_training import _models_from_checkpoint

    state, tensors = _trainer_state("shared", LaborQoLHead())
    converted = convert_qol_encoder(state, "copy")
    assert labor_checkpoint_qol_encoder(converted) == "separate"
    _, _, old, _, _ = _models_from_checkpoint(state)
    _, _, new, _, _ = _models_from_checkpoint(converted)
    for key, value in state["task_critic"].items():
        assert torch.equal(converted["task_critic"][key], value), key
    assert _equal(new.qol_encoder.state_dict(), old.encoder.state_dict())
    old.train()
    new.train()
    with torch.no_grad():
        a, b = old(tensors), new(tensors)
    assert all(torch.equal(a[k], b[k]) for k in LABOR_OBJECTIVES)
    for section in ("actor", "cost_critics"):
        assert _equal(converted[section], state[section])
    assert converted["critic_optimizer"]["state"] == {}
    assert converted["critic_qol_encoder_conversions"][-1]["init"] == "copy"
    with pytest.raises(ValueError, match="already declares"):
        convert_qol_encoder(converted, "copy")


def test_convert_fresh_equals_initializing_separate() -> None:
    shared, _ = _trainer_state("shared", LaborQoLHead(), seed=11)
    direct, _ = _trainer_state("separate", LaborQoLHead(), seed=11)
    converted = convert_qol_encoder(shared, "fresh")
    assert _equal(converted["task_critic"], direct["task_critic"])
    assert converted["labor_model_contract"] == direct["labor_model_contract"]
    other = convert_qol_encoder(shared, "fresh", seed=12)
    assert not _equal(other["task_critic"], direct["task_critic"])


def test_support_and_privileged_conversions_keep_the_qol_encoder() -> None:
    from anthropocene_rl.autonomous_v7.labor_training import _models_from_checkpoint

    state, _ = _trainer_state("separate")
    narrow = LaborValueSupport(-1.0, 4.0, 255)
    converted = convert_task_critic_support(state, narrow, seed=5)
    assert labor_checkpoint_qol_encoder(converted) == "separate"
    _, _, task, _, _ = _models_from_checkpoint(converted)
    for key, value in state["task_critic"].items():
        if key.startswith("qol_encoder."):
            assert torch.equal(task.state_dict()[key], value), key
    privileged = convert_privileged_critics(state, _normalizer(), seed=2)
    assert labor_checkpoint_qol_encoder(privileged) == "separate"
    keys = privileged["task_critic"]
    assert any(k.startswith("qol_encoder.privileged_") for k in keys)
    for key, value in state["task_critic"].items():
        assert torch.equal(keys[key], value), key


# --- 7. command line ----------------------------------------------------------------------------


def test_initialize_flags() -> None:
    base = ["initialize", "--normalizer-checkpoint", "n.pt", "--world-seed", "5000",
            "--optimizer-seed", "1", "--out", "o.pt"]
    parse = lt.argument_parser().parse_args
    assert lt._initialize_qol_encoder(parse(base)) == "shared"
    assert lt._initialize_qol_encoder(parse([*base, "--campaign-r5"])) == "separate"
    assert lt._initialize_qol_encoder(
        parse([*base, "--campaign-r5", "--qol-encoder", "shared"])
    ) == "shared"
    assert lt._initialize_qol_encoder(parse([*base, "--qol-encoder", "separate"])) == "separate"
    with pytest.raises(SystemExit):
        parse([*base, "--qol-encoder", "twin"])
    assert lt._initialize_qol_encoder(argparse.Namespace()) == "shared"


# --- 8. gate v2 coexistence (TD rule v2, early stop, n-step report) ----------------------------


def test_td_rule_v2_and_early_stop_with_a_separate_encoder() -> None:
    plain, plain_costs, _ = _fit(stability=StabilityConfig(log_every=1))
    early, early_costs, early_summary = _fit(
        stability=StabilityConfig(log_every=1, early_stop=True)
    )
    # Without a trip, early stop changes nothing (the separate encoder included).
    assert _equal(plain.state_dict(), early.state_dict())
    assert _equal(plain_costs.state_dict(), early_costs.state_dict())
    assert "early_stop" not in early_summary["stability"]
    td = early_summary["stability"]["checks"]["td_divergence"]
    assert td["td_rule"] == "v2-floor-or-declining" and td["refreshes"]
    # Nine steps are too short for a v2 PASS on this toy fixture (also for the shared
    # critic): the end-of-run trajectory judges it, never a live stop.
    assert all(a["rule"] == "v2-floor-or-declining" for a in td["alarms"])
    assert all(a["reason"].endswith("not-declining-or-stationary") for a in td["alarms"])


def test_nstep_critic_report_on_a_separate_critic() -> None:
    from anthropocene_rl.autonomous_v7 import labor_pretrain_eval as pe

    task, costs = _pair("separate", privileged=False, seed=0)
    dataset = _dataset(privileged=False)
    pretrain_critic_on_dataset(
        task_critic=task, cost_critics=costs, dataset=dataset, epochs=1, batch_size=2
    )
    reference = {"value": 10.0, "source": "artifact", "digest": "a" * 64}
    report = pe.critic_dataset_report(task, costs, dataset, batch_size=3, qol_reference=reference)
    assert report["qol_nstep"]["rows"] == report["rows"]
    assert report["population_nstep"]["gating"] is False
