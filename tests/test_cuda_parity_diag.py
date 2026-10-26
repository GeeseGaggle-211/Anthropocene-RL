"""tools/cuda_parity_diag.py: its replica fit loops equal the real fits on CPU (bitwise)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_PATH = Path(__file__).resolve().parents[1] / "tools" / "cuda_parity_diag.py"
_spec = importlib.util.spec_from_file_location("cuda_parity_diag", _PATH)
diag = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("cuda_parity_diag", diag)
_spec.loader.exec_module(diag)


def _replica_matches(problem) -> None:
    cpu = torch.device("cpu")
    real = problem.real_fit(cpu)
    replica = diag.run(problem, cpu, "adam", "default", 10.0, False, diag.Recorder())
    assert diag.repeat_check(replica["trajectory"][-1], real)["bitwise"]


def test_critic_replica_equals_real_fit() -> None:
    _replica_matches(diag.CriticProblem("monte-carlo"))


def test_advantage_replica_equals_real_fit() -> None:
    _replica_matches(diag.AdvantageProblem())


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

    d = diag.state_drift(test, ref, init, gauge)
    w = d["per_tensor"]["w"]
    assert abs(w["plain"] - 2.0) < 1e-6 and abs(w["update"] - 2.0) < 1e-6
    assert w["plain_non_gauge"] == 0.0 and w["gauge_elements"] == 2
    assert d["per_tensor"]["b"]["update"] == float("inf")  # moved where the reference did not
    assert d["worst_update_non_gauge"]["tensor"] == "b"
