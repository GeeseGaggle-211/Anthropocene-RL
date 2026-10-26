"""Device selection for the stored-data fits (pretrain-critic, pretrain-bc, A fit, r5 evaluate).

Only the fitting stages that read stored shards may run on a GPU; collection,
search and evaluation against the game stay on CPU. The contract:

* ``--device cpu`` (the default) is the historical path, bit for bit: every
  ``to_device`` call is a no-op on CPU tensors and no global flag changes.
* ``--device cuda`` moves the modules once and every minibatch per step; the
  stored datasets stay in host memory (a critic dataset is ~11 GB, so several
  fits can share one GPU). ``--device auto`` picks CUDA when available.
* Checkpoints are device-agnostic: every saved tensor is on CPU
  (:func:`cpu_copy`), so a CUDA-fitted checkpoint loads with
  ``map_location="cpu"`` exactly like a CPU one, and resume states written on
  one device resume on the other (the trajectory then differs, see below).
* Model initialization (``labor_training initialize``, the A head under
  ``torch.manual_seed``) always runs on CPU, so initial parameters are identical
  across devices; only the optimization trajectory can differ.

Determinism. CPU keeps the approved bitwise run-to-run guarantee (same seed,
data, thread count and code). CUDA, with :func:`configure` applied (cuDNN
deterministic, no benchmark autotuning, TF32 off, ``CUBLAS_WORKSPACE_CONFIG``
and ``torch.use_deterministic_algorithms(True, warn_only=True)``) is expected
to be run-to-run reproducible on the same GPU model, driver and torch build,
but this is not guaranteed for every kernel (``warn_only`` lets an op without
a deterministic CUDA kernel run with a warning instead of failing). The r5
determinism probe therefore uses :data:`CUDA_DETERMINISM_TOLERANCE`
(PROPOSED, not approved) for CUDA repeats. CPU-versus-CUDA results are never
expected to be bitwise equal; their parity is checked by
``tests/test_labor_device.py`` at :data:`CROSS_DEVICE_STEP_TOLERANCE` for one
forward/backward step and is not a gate.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch

DEVICE_CHOICES = ("cpu", "cuda", "auto")

# Approved: max relative drift between two identical CPU fits (labor_r5_acceptance).
CPU_DETERMINISM_TOLERANCE = 1e-6
# PROPOSED (not approved): max relative drift between two identical CUDA fits on the
# same GPU model / driver / torch build after the 50-step critic probe. Rationale: the
# deterministic flags make repeats bitwise in the common case; the residual allowance
# covers warn_only kernels (atomic scatter/index_add backward) whose float32
# summation order may vary, ~1e-7 per step compounded through 50 Adam steps.
CUDA_DETERMINISM_TOLERANCE = 1e-5
CUDA_DETERMINISM_TOLERANCE_STATUS = "proposed (awaiting co-architect approval)"
# Parity test only (not a gate): relative loss/gradient agreement of one fp32
# forward/backward step on CPU and CUDA with TF32 disabled.
CROSS_DEVICE_STEP_TOLERANCE = 1e-4


def add_device_argument(parser: Any) -> None:
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="cpu",
        help="cpu (default, bitwise historical path), cuda, or auto (cuda when available); "
        "data stays in host memory, checkpoints are always saved on CPU",
    )


def resolve_device(name: str | torch.device | None) -> torch.device:
    if name is None:
        return torch.device("cpu")
    if isinstance(name, torch.device):
        return name
    text = str(name)
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cpu":
        return torch.device("cpu")
    if text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {text} requested but torch.cuda.is_available() is False "
                f"(torch {torch.__version__}, built for CUDA {torch.version.cuda})"
            )
        return torch.device(text)
    raise ValueError(f"unknown device {text!r}; expected one of {DEVICE_CHOICES}")


def configure(device: torch.device) -> dict[str, Any]:
    """Cheap determinism settings for CUDA; CPU is left untouched (bitwise path)."""
    record: dict[str, Any] = {"device": str(device), "torch": torch.__version__}
    if device.type != "cuda":
        return record
    # Must be set before the first cuBLAS handle is created in this process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    record.update(
        {
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "cudnn_deterministic": True,
            "tf32": False,
            "deterministic_algorithms": "warn_only",
            "determinism_tolerance": CUDA_DETERMINISM_TOLERANCE,
            "determinism_tolerance_status": CUDA_DETERMINISM_TOLERANCE_STATUS,
        }
    )
    return record


def setup(name: str | None) -> torch.device:
    """resolve_device + configure, for CLI entry points."""
    device = resolve_device(name)
    configure(device)
    return device


def module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def to_device(value: Any, device: torch.device) -> Any:
    """Tensors (also inside dicts/lists/tuples) to ``device``; a no-op on the same device."""
    if isinstance(value, torch.Tensor):
        return value if value.device == device else value.to(device)
    if isinstance(value, Mapping):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


def cpu_copy(value: Any) -> Any:
    """A CPU copy of nested state (state dicts, optimizer state) for device-agnostic saves.

    CPU tensors are returned as they are (``torch.save`` of a CPU state is unchanged).
    """
    cpu = torch.device("cpu")
    if isinstance(value, torch.Tensor):
        return value if value.device == cpu else value.detach().to(cpu)
    if isinstance(value, dict):
        try:
            out = type(value)()
        except TypeError:
            out = {}
        for key, item in value.items():
            out[key] = cpu_copy(item)
        metadata = getattr(value, "_metadata", None)  # module state_dict versions
        if metadata is not None:
            out._metadata = metadata  # type: ignore[attr-defined]
        return out
    if isinstance(value, Mapping):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def all_finite(tensors: list[torch.Tensor]) -> bool:
    """One host sync for many tensors on an accelerator; the per-tensor loop on CPU."""
    groups: dict[torch.device, list[torch.Tensor]] = {}
    for tensor in tensors:
        groups.setdefault(tensor.device, []).append(tensor)
    for device, group in groups.items():
        if device.type == "cpu":
            # Historical per-tensor loop (Adam's step counters live here even on CUDA).
            if not all(bool(torch.isfinite(t).all()) for t in group):
                return False
        elif not bool(torch.stack([torch.isfinite(t).all() for t in group]).all()):
            return False
    return True


def device_record(device: torch.device) -> dict[str, Any]:
    """Recorded in a fit summary when it ran off CPU: GPU and peak device memory."""
    record: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        record.update(
            gpu=torch.cuda.get_device_name(device),
            max_memory_allocated_gib=round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
            max_memory_reserved_gib=round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
            deterministic_algorithms="warn_only",
            tf32=False,
        )
    return record
