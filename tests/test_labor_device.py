"""--device support for the stored-data fits (labor_device).

CPU tests always run: device resolution, device-agnostic (CPU) saves, the
finite check, the CLI flags and the device-aware determinism tolerance of the
r5 acceptance harness. The CPU-versus-CUDA parity tests are skipped when no
GPU is present; on the GPU fitting VM they are the bootstrap smoke test:

* one forward/backward step agrees within ``CROSS_DEVICE_STEP_TOLERANCE``
  (relative, fp32, TF32 off) for the critics, the A head and the actor BC loss;
* short CUDA fits (critic with stability monitor and TD refresh, A head, BC)
  finish admitted, save CPU-only state and stay close to the CPU fit;
* two identical CUDA critic fits agree within the PROPOSED
  ``CUDA_DETERMINISM_TOLERANCE``.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from test_labor_privileged_model import _layout
from test_labor_qol_head import _training_dataset

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7 import labor_advantage as la
from anthropocene_rl.autonomous_v7 import labor_device as ld
from anthropocene_rl.autonomous_v7 import labor_r5_acceptance as ra
from anthropocene_rl.autonomous_v7.labor_bc import (
    _batch_loss_and_l1,
    pretrain_critic_on_dataset,
    pretrain_labor_actor_bc,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    LABOR_ACTIVITY_FEATURE_WIDTH,
    LABOR_RESOURCE_FEATURE_WIDTH,
    LABOR_SPECIALTY_FEATURE_WIDTH,
    LaborActor,
    LaborCostCritics,
    LaborEncoderLayout,
    LaborQoLHead,
    LaborTaskCritic,
)
from anthropocene_rl.autonomous_v7.labor_ppo import (
    labor_cost_value_losses,
    labor_task_value_losses,
)
from anthropocene_rl.autonomous_v7.labor_stability import StabilityConfig

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="CPU-vs-CUDA parity needs a GPU")


# --- CPU: resolution, saves, finite checks ------------------------------------------------------


def test_resolve_device() -> None:
    assert ld.resolve_device("cpu") == torch.device("cpu")
    assert ld.resolve_device(None) == torch.device("cpu")
    assert ld.resolve_device("auto").type == ("cuda" if CUDA else "cpu")
    with pytest.raises(ValueError):
        ld.resolve_device("tpu")
    if not CUDA:
        with pytest.raises(RuntimeError, match="cuda"):
            ld.resolve_device("cuda")


def test_configure_leaves_cpu_untouched() -> None:
    before = (
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    )
    record = ld.configure(torch.device("cpu"))
    assert record["device"] == "cpu"
    assert (
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    ) == before


def test_to_device_and_cpu_copy_are_identity_on_cpu() -> None:
    tensor = torch.arange(3.0)
    nested = {"a": tensor, "b": [tensor, (tensor, 1)], "c": None}
    moved = ld.to_device(nested, torch.device("cpu"))
    assert moved["a"] is tensor and moved["b"][0] is tensor and moved["b"][1][0] is tensor
    state = torch.nn.Linear(2, 2).state_dict()
    copied = ld.cpu_copy(state)
    assert getattr(copied, "_metadata", None) == getattr(state, "_metadata", None)
    assert all(copied[k] is state[k] for k in state)


def test_all_finite_matches_the_per_tensor_check() -> None:
    good = [torch.ones(3), torch.zeros(2, 2)]
    assert ld.all_finite(good) and ld.all_finite([])
    assert not ld.all_finite([*good, torch.tensor([1.0, float("nan")])])
    assert not ld.all_finite([torch.tensor([float("inf")])])


def test_critic_resume_state_is_saved_on_cpu(tmp_path) -> None:
    torch.manual_seed(0)
    task = LaborTaskCritic(
        _layout(False), width=16, token_width=16, qol_head=LaborQoLHead(target="td-lambda")
    )
    costs = LaborCostCritics(_layout(False), width=16, token_width=16)
    data = _training_dataset()
    data["collection"] = dict(data["collection"], dataset_digest="fixture")
    path = tmp_path / "resume.pt"
    pretrain_critic_on_dataset(
        task_critic=task,
        cost_critics=costs,
        dataset=data,
        epochs=3,
        batch_size=2,
        max_steps=4,
        resume_path=str(path),
        resume_every=2,
    )
    _assert_all_cpu(torch.load(path, map_location=None, weights_only=True))


def _assert_all_cpu(value) -> None:
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu"
    elif isinstance(value, dict):
        for item in value.values():
            _assert_all_cpu(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_all_cpu(item)


def test_cli_parsers_accept_device() -> None:
    from anthropocene_rl.autonomous_v7 import labor_training as lt

    parser = lt.argument_parser()
    critic = parser.parse_args(
        ["pretrain-critic", "--checkpoint", "c.pt", "--dataset", "d", "--out", "o.pt"]
    )
    assert critic.device == "cpu"
    critic = parser.parse_args(
        ["pretrain-critic", "--checkpoint", "c.pt", "--dataset", "d", "--out", "o.pt",
         "--device", "cuda"]
    )
    assert critic.device == "cuda"
    bc = parser.parse_args(
        ["pretrain-bc", "--checkpoint", "c.pt", "--dataset", "d", "--out", "o.pt",
         "--device", "auto"]
    )
    assert bc.device == "auto"
    fit = la.argument_parser().parse_args(
        ["fit", "--checkpoint", "c.pt", "--payloads", "p", "--out", "o.pt", "--device", "cuda"]
    )
    assert fit.device == "cuda"
    evaluate = ra.argument_parser().parse_args(
        ["evaluate", "--arm", "a", "--seed", "0", "--checkpoint", "c", "--advantage", "a",
         "--calibration-data", "d", "--reserved-payloads", "r", "--out", "o", "--device", "cuda"]
    )
    assert evaluate.device == "cuda"


def test_game_bound_collection_refuses_a_gpu(tmp_path) -> None:
    from anthropocene_rl.autonomous_v7 import labor_training as lt

    args = lt.argument_parser().parse_args(
        ["pretrain-critic", "--checkpoint", str(tmp_path / "missing.pt"), "--worlds", "90100",
         "--out", str(tmp_path / "o.pt"), "--device", "cuda"]
    )
    with pytest.raises((RuntimeError, ValueError)):
        lt._command_pretrain_critic(args)


def test_device_flag_is_absent_on_cpu() -> None:
    import argparse

    assert ra._device_flag(argparse.Namespace()) == []
    assert ra._device_flag(argparse.Namespace(device="cpu")) == []
    assert ra._device_flag(argparse.Namespace(device="cuda")) == ["--device", "cuda"]


def _determinism(device, relative, bitwise=False):
    report = {"comparisons": {"critic:task_critic": {"bitwise": bitwise, "max_relative": relative}}}
    if device is not None:
        report["device"] = device
    return ra.determinism_checks(report)[0]["status"]


def test_determinism_tolerance_is_device_aware() -> None:
    assert ra.threshold("determinism_tolerance") == ld.CPU_DETERMINISM_TOLERANCE == 1e-6
    assert ra.threshold("determinism_tolerance_cuda") == ld.CUDA_DETERMINISM_TOLERANCE
    assert ra.PREREGISTERED["determinism_tolerance_cuda"]["status"].startswith("approved")
    # CPU (default when the report predates the device field): the approved 1e-6.
    assert _determinism(None, 5e-7) == "PASS"
    assert _determinism("cpu", 5e-6) == "FAIL"
    # CUDA: bitwise passes; within the approved 1e-5 tolerance PASSes; beyond FAILs.
    assert _determinism("cuda", 0.0, bitwise=True) == "PASS"
    assert _determinism("cuda", 5e-6) == "PASS"
    assert _determinism("cuda", 1e-5) == "PASS"
    assert _determinism("cuda", 2e-5) == "FAIL"
    assert _determinism("cuda", 1e-3) == "FAIL"


# --- CPU vs CUDA parity (skipped without a GPU) --------------------------------------------------


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).abs().max() / a.abs().max().clamp_min(1e-12))


def _critic_pair(seed: int = 0, target: str = "td-lambda"):
    torch.manual_seed(seed)
    task = LaborTaskCritic(
        _layout(False), width=16, token_width=16, qol_head=LaborQoLHead(target=target)
    )
    costs = LaborCostCritics(_layout(False), width=16, token_width=16)
    return task, costs


@needs_cuda
def test_critic_step_parity_cpu_vs_cuda() -> None:
    cuda = ld.setup("cuda")
    task, costs = _critic_pair()
    task_g, costs_g = copy.deepcopy(task).to(cuda), copy.deepcopy(costs).to(cuda)
    data = _training_dataset()
    from anthropocene_rl.autonomous_v7.labor_bc import discounted_returns
    from anthropocene_rl.autonomous_v7.labor_model import LABOR_COSTS, LABOR_OBJECTIVES
    from anthropocene_rl.autonomous_v7.labor_privileged import actor_inputs

    returns = discounted_returns(data)
    tensors = {
        k: v
        for k, v in actor_inputs(data).items()
        if isinstance(v, torch.Tensor)
        and not k.startswith(("target_", "label_", "reward_", "transition_", "row_", "tail_"))
    }
    task_targets = {n: returns[n] for n in LABOR_OBJECTIVES}
    cost_targets = {n: returns[n] for n in LABOR_COSTS}
    losses = []
    for t, c, device in ((task, costs, torch.device("cpu")), (task_g, costs_g, cuda)):
        batch = ld.to_device(tensors, device)
        task_loss, _, _ = labor_task_value_losses(t, batch, ld.to_device(task_targets, device))
        cost_loss, _ = labor_cost_value_losses(c, batch, ld.to_device(cost_targets, device))
        loss = task_loss + cost_loss
        loss.backward()
        losses.append(loss)
    assert _relative(losses[0], losses[1]) <= ld.CROSS_DEVICE_STEP_TOLERANCE
    for module, other in ((task, task_g), (costs, costs_g)):
        for (name, p), q in zip(module.named_parameters(), other.parameters(), strict=True):
            if p.grad is not None and p.grad.abs().max() > 0:
                assert _relative(p.grad, q.grad) <= 10 * ld.CROSS_DEVICE_STEP_TOLERANCE, name


def _short_critic_fit(device: torch.device, resume=None):
    task, costs = _critic_pair()
    task.to(device)
    costs.to(device)
    data = _training_dataset()
    data["collection"] = dict(data["collection"], dataset_digest="fixture")
    summary = pretrain_critic_on_dataset(
        task_critic=task,
        cost_critics=costs,
        dataset=data,
        epochs=3,
        batch_size=2,
        rng=np.random.default_rng(7),
        stability=StabilityConfig(log_every=1),
        resume_path=resume,
        resume_every=2,
    )
    task.to("cpu")
    costs.to("cpu")
    return summary, {**task.state_dict(), **{f"c.{k}": v for k, v in costs.state_dict().items()}}


@needs_cuda
def test_critic_fit_on_cuda_is_close_to_cpu_and_repeatable(tmp_path) -> None:
    cuda = ld.setup("cuda")
    _, cpu_state = _short_critic_fit(torch.device("cpu"))
    summary, gpu_state = _short_critic_fit(cuda, resume=str(tmp_path / "r.pt"))
    _, gpu_again = _short_critic_fit(cuda)
    assert summary["stability"]["checks"]["finite_updates"]["status"] == "PASS"
    _assert_all_cpu(torch.load(tmp_path / "r.pt", weights_only=True))
    assert all(v.device.type == "cpu" for v in gpu_state.values())
    # Nine Adam steps: cross-device drift stays small (not a gate, fp32 reassociation only).
    worst = max(
        _relative(cpu_state[k], gpu_state[k])
        for k in cpu_state
        if cpu_state[k].is_floating_point()
    )
    assert worst <= 1e-3
    repeat = ra.compare_state_dicts(gpu_state, gpu_again)
    assert repeat["bitwise"] or repeat["max_relative"] <= ld.CUDA_DETERMINISM_TOLERANCE


@needs_cuda
def test_critic_resume_crosses_devices(tmp_path) -> None:
    cuda = ld.setup("cuda")
    path = str(tmp_path / "resume.pt")
    task, costs = _critic_pair()
    data = _training_dataset()
    data["collection"] = dict(data["collection"], dataset_digest="fixture")
    pretrain_critic_on_dataset(
        task_critic=task.to(cuda), cost_critics=costs.to(cuda), dataset=data, epochs=3,
        batch_size=2, rng=np.random.default_rng(7), max_steps=4, resume_path=path, resume_every=2,
    )
    task_b, costs_b = _critic_pair()  # resumed on CPU from a CUDA-written state
    summary = pretrain_critic_on_dataset(
        task_critic=task_b, cost_critics=costs_b, dataset=data, epochs=3, batch_size=2,
        rng=np.random.default_rng(7), resume_path=path, resume_every=2,
    )
    assert summary["stability"]["steps"] == 9


@needs_cuda
def test_advantage_fit_parity_cpu_vs_cuda() -> None:
    cuda = ld.setup("cuda")
    from test_labor_privileged_model import _critics

    shard = la.synthetic_candidate_shard(
        worlds=tuple(range(90100, 90106)), roots_per_world=3, candidates=5, seed=1
    )
    config = la.AdvantageFitConfig(epochs=3, batch_roots=4, lr=2e-3, seed=0, hidden=32)
    heads = []
    for device in (torch.device("cpu"), cuda):
        task, costs = _critics(_layout(False))
        for module in (task, costs):
            module.to(device)
            for p in module.parameters():
                p.requires_grad_(False)
        head, summary = la.fit_advantage_head(
            task, costs, la.parse_candidate_shard(shard), config,
            stability=StabilityConfig(log_every=1),
        )
        assert summary["stability"]["checks"]["finite_updates"]["status"] == "PASS"
        assert summary["stability"]["checks"]["frozen_gradients_zero"]["status"] == "PASS"
        heads.append(head.to("cpu").state_dict())
    worst = max(
        _relative(heads[0][k], heads[1][k]) for k in heads[0] if heads[0][k].abs().max() > 0
    )
    assert worst <= 1e-3


def _bc_dataset(rows: int = 32, core_dim: int = 15):
    torch.manual_seed(5)
    target_specialty = torch.zeros(rows, 14)
    target_specialty[:, 1], target_specialty[:, 2] = 0.8, 0.2
    target_activity = torch.zeros(rows, 13, 22)
    target_activity[:, 0, 1] = 1.0
    target_activity[:, 1, 5] = 1.0
    target_activity[:, 2:, 0] = 1.0
    dataset = {
        "core": torch.randn(rows, core_dim),
        "labor_personality": torch.full((rows, 2), 0.5),
        "movement_mask": torch.ones(rows, 8, dtype=torch.bool),
        "region_movement_mask": torch.ones(rows, 10, dtype=torch.bool),
        "labor_specialty_features": torch.randn(rows, 14, LABOR_SPECIALTY_FEATURE_WIDTH),
        "labor_specialty_present": torch.ones(rows, 14, dtype=torch.bool),
        "labor_specialty_mask": torch.ones(rows, 14, dtype=torch.bool),
        "labor_activity_features": torch.randn(rows, 13 * 22, LABOR_ACTIVITY_FEATURE_WIDTH),
        "labor_activity_present": torch.ones(rows, 13 * 22, dtype=torch.bool),
        "labor_activity_mask": torch.ones(rows, 13, 22, dtype=torch.bool),
        "labor_resource_features": torch.randn(rows, 20, LABOR_RESOURCE_FEATURE_WIDTH),
        "labor_resource_present": torch.ones(rows, 20, dtype=torch.bool),
        "target_specialty": target_specialty,
        "target_activity": target_activity,
        "target_movement_code": torch.full((rows,), -1, dtype=torch.long),
        "target_fission": torch.zeros(rows, dtype=torch.long),
        "target_trade_value_latent": torch.zeros(rows, 7),
        "target_trade_gate": torch.ones(rows, 7, dtype=torch.long),
        "target_trade_share_latent": torch.zeros(rows),
        "collection": {"total_decisions": rows, "initial_accepted": rows},
    }
    for name, count, columns in contracts.SETS:
        dataset[name] = torch.randn(rows, count, columns)
        dataset[name + "_present"] = torch.ones(rows, count, dtype=torch.bool)
    layout = LaborEncoderLayout(
        core_dim, LABOR_SPECIALTY_FEATURE_WIDTH, LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
    )
    return layout, dataset


@needs_cuda
def test_bc_parity_cpu_vs_cuda() -> None:
    cuda = ld.setup("cuda")
    layout, dataset = _bc_dataset()
    torch.manual_seed(0)
    actor = LaborActor(layout)
    actor_g = copy.deepcopy(actor).to(cuda)
    tensors = {k: v for k, v in dataset.items() if isinstance(v, torch.Tensor)}
    losses = [
        _batch_loss_and_l1(model, ld.to_device(tensors, device), target_log_std=-2.0)[0]
        for model, device in ((actor, torch.device("cpu")), (actor_g, cuda))
    ]
    assert _relative(losses[0], losses[1]) <= ld.CROSS_DEVICE_STEP_TOLERANCE
    states = []
    for model in (actor, actor_g):
        summary = pretrain_labor_actor_bc(
            actor=model, dataset=dataset, epochs=3, batch_size=16, lr=5e-3,
            rng=np.random.default_rng(3), stability=StabilityConfig(log_every=1),
        )
        assert summary["stability"]["checks"]["finite_updates"]["status"] == "PASS"
        states.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
    worst = max(
        _relative(states[0][k], states[1][k])
        for k in states[0]
        if states[0][k].is_floating_point() and states[0][k].abs().max() > 0
    )
    assert worst <= 1e-3
