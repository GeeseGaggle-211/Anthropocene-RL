"""Cross-device parity machinery on CPU: replica fits, drift metrics, native-JIT guard."""

from __future__ import annotations

import pytest
import torch
from labor_parity_problems import AdvantageProblem, CriticProblem, Recorder, run, teleport

from anthropocene_rl.autonomous_v7 import labor_device as ld
from anthropocene_rl.autonomous_v7.labor_device_parity import (
    function_drift,
    repeat_check,
    state_drift,
)


def _replica_matches(problem) -> None:
    cpu = torch.device("cpu")
    real = problem.real_fit(cpu)
    replica = run(problem, cpu, "adam", "default", 10.0, False, Recorder())
    assert repeat_check(replica["trajectory"][-1], real)["bitwise"]


def test_critic_replica_equals_real_fit() -> None:
    _replica_matches(CriticProblem("monte-carlo"))


def test_advantage_replica_equals_real_fit() -> None:
    _replica_matches(AdvantageProblem())


def test_teleport_to_the_same_device_is_exact() -> None:
    problem = AdvantageProblem()
    result = teleport(problem, problem.steps // 2, torch.device("cpu"))
    assert result["loss_relative"] == 0.0 and result["global_l2_relative"] == 0.0


def test_drift_metrics_and_gauge_mask() -> None:
    init = {"w": torch.zeros(6), "b": torch.tensor([1.0, 2.0])}
    ref = {"w": torch.tensor([0.0, 0.0, 1e-3, 1e-3, 0.0, 0.0]), "b": torch.tensor([1.0, 2.0])}
    test = {"w": torch.tensor([0.0, 0.0, 1e-3, 3e-3, 0.0, 0.0]), "b": torch.tensor([1.0, 2.5])}

    def gauge(key, tensor):
        if key == "w":
            mask = torch.zeros(6, dtype=torch.bool)
            mask[2:4] = True
            return mask
        return None

    d = state_drift(test, ref, init, gauge)
    w = d["per_tensor"]["w"]
    assert abs(w["plain"] - 2.0) < 1e-6 and abs(w["update"] - 2.0) < 1e-6
    assert w["plain_non_gauge"] == 0.0 and w["gauge_elements"] == 2
    assert d["per_tensor"]["b"]["update"] == float("inf")  # moved where the reference did not
    assert d["worst_update_non_gauge"]["tensor"] == "b"


def test_function_drift_separates_unmoved_outputs() -> None:
    init = {"moved": torch.zeros(3), "still": torch.ones(2)}
    ref = {"moved": torch.tensor([1.0, 2.0, 4.0]), "still": torch.ones(2)}
    test = {"moved": torch.tensor([1.0, 2.0, 4.004]), "still": torch.tensor([1.0, 1.0 + 1e-6])}
    d = function_drift(test, ref, init)
    assert abs(d["worst_change_relative"]["value"] - 1e-3) < 1e-6
    assert d["worst_change_relative"]["output"] == "moved"
    assert abs(d["worst_unmoved_plain"]["value"] - 1e-6) < 1e-7
    assert d["worst_unmoved_plain"]["output"] == "still"


def test_native_jit_status_reports_this_torch() -> None:
    status = ld.native_jit_status()
    assert set(status) >= {"env", "registered"}
    if status.get("torch_native"):
        assert isinstance(status["disabled"], bool)


def test_configure_refuses_cuda_with_native_jit_kernels(monkeypatch) -> None:
    monkeypatch.setattr(
        ld, "native_jit_status", lambda: {"env": None, "registered": {"triton": ["bmm"]}}
    )
    with pytest.raises(RuntimeError, match="TORCH_DISABLE_NATIVE_JIT=1"):
        ld.configure(torch.device("cuda"))
    assert ld.configure(torch.device("cpu"))["device"] == "cpu"  # CPU untouched
