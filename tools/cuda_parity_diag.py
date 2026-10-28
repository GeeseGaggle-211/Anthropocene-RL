#!/usr/bin/env python
"""CPU-versus-CUDA parity diagnostic for the stored-data fits (critic, A head, actor BC).

Purpose: decide whether the multi-step failures of ``tests/test_labor_device.py``
(critic fit drift 0.0779, A-head drift 6.36, both against a 1e-3 per-tensor
relative threshold) are a numerical defect on CUDA or an ill-posed metric /
optimizer amplification of fp32 kernel differences.

It uses the failing tests' own fixtures (``_critic_pair``/``_training_dataset``,
``_critics``/``synthetic_candidate_shard``, ``_bc_dataset``) and replays each fit
with an explicit step loop (a "replica") that is checked bitwise against the real
fit function on CPU. Every run starts from the same CPU-initialised parameters
and uses the same CPU-drawn batch plan. For each fit it reports:

(a) per-step losses on CPU and CUDA and their relative difference;
(b) per-tensor drift of the parameters under three metrics: plain relative
    max|a-b|/max|b| (the tests' metric), scale-normalised
    max|a-b|/(max|b|+max|init|) and update-relative max|a-b|/max|b-init|,
    plus global L2 versions;
(c) the same with plain SGD instead of Adam/AdamW;
(d) a second CUDA run (bitwise, or max relative against the approved 1e-5);
(e) all of (a)-(d) again under ``sdpa_kernel(SDPBackend.MATH)``;
(f) "teleported" one-step parity: the CPU trajectory's state at step k
    (parameters, optimizer state, TD targets) is loaded on both devices and one
    step is taken; this measures the per-step kernel error separately from
    accumulated trajectory divergence.

It also re-runs the real fit functions exactly as the tests do and reports the
tests' own metric, so the replica can be matched to the observed failures.

Triton/C compiler: torch 2.14's ``torch._native`` registers a Triton override for
``aten::bmm`` outer products on CUDA (``torch/_native/ops/bmm_outer_product``),
which the actor's einsum backward hits; Triton needs a C compiler at runtime to
build its launcher. torch reads ``TORCH_DISABLE_NATIVE_JIT=1``
(``torch/_native/common_utils.check_native_jit_disabled``, cached, read at
``import torch``) and then registers no native DSL override, so ``aten::bmm`` stays
on the stock cuBLAS kernel. This script sets it before importing torch, records
whether it took effect, and (with ``--probe-native``) runs a subprocess without
it to show whether the compiler failure reproduces.

The fixtures and replica loops live in ``tests/labor_parity_problems.py`` and the
metrics in ``anthropocene_rl.autonomous_v7.labor_device_parity`` (shared with
``tests/test_labor_device.py``).

Usage (GPU VM): ``cd <repo> && PYTHONPATH=src python tools/cuda_parity_diag.py --out diag.json``.
On a CPU-only host the CUDA parts are skipped and only the CPU runs and the
replica-versus-real validation execute.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # Must precede ``import torch``: torch._native reads it once while importing
    # (labor_device.configure refuses CUDA otherwise).
    os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

REPO_DEFAULT = Path(__file__).resolve().parents[1]


def _repo_from_argv(argv: list[str]) -> Path:
    for i, arg in enumerate(argv):
        if arg == "--repo" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
        if arg.startswith("--repo="):
            return Path(arg.split("=", 1)[1]).resolve()
    return REPO_DEFAULT


# The shared fixtures and replica loops live in <repo>/tests (not a package).
sys.path.insert(0, str(_repo_from_argv(sys.argv[1:]) / "tests"))
from labor_parity_problems import PROBLEMS, Recorder, drift_all, one_step, run  # noqa: E402

from anthropocene_rl.autonomous_v7.labor_device_parity import (  # noqa: E402
    grad_compare,
    rel,
    repeat_check,
    state_drift,
)


def key_metrics(d: dict[str, Any]) -> dict[str, Any]:
    """The headline numbers of a drift record (parameter and output space)."""
    out = {
        "worst_plain": d["worst_plain"],
        "worst_plain_nonzero_ref": d["worst_plain_nonzero_ref"],
        "worst_scale": d["worst_scale"],
        "worst_update": d["worst_update"],
        "worst_plain_non_gauge": d["worst_plain_non_gauge"],
        "worst_update_non_gauge": d["worst_update_non_gauge"],
        "global_l2_relative": d["global_l2_relative"],
        "global_l2_update_relative": d["global_l2_update_relative"],
    }
    if "function" in d:
        out["function"] = {
            k: d["function"][k]
            for k in ("worst_change_relative", "worst_plain", "global_l2_change_relative")
        }
    return out


def trajectory_report(problem, cpu, gpu) -> list[dict[str, Any]]:
    init = cpu["trajectory"][0]
    out = []
    for k in range(len(cpu["losses"])):
        d = state_drift(gpu["trajectory"][k + 1], cpu["trajectory"][k + 1], init, problem.gauge)
        out.append(
            {
                "step": k,
                "loss_cpu": cpu["losses"][k],
                "loss_cuda": gpu["losses"][k],
                "loss_relative": rel(gpu["losses"][k], cpu["losses"][k]),
                "grad_norm_cpu": cpu["grad_norms"][k],
                "grad_norm_cuda": gpu["grad_norms"][k],
                "worst_plain": d["worst_plain"]["value"],
                "worst_scale": d["worst_scale"]["value"],
                "worst_update": d["worst_update"]["value"],
                "worst_update_non_gauge": d["worst_update_non_gauge"]["value"],
                "global_l2_relative": d["global_l2_relative"],
                "global_l2_update_relative": d["global_l2_update_relative"],
            }
        )
    return out


def teleport_report(problem, cpu, kind, backend, sgd_scale, cuda, recorder) -> list[dict[str, Any]]:
    out = []
    cpu_dev = torch.device("cpu")
    for k, snap in enumerate(cpu["snapshots"]):
        a = one_step(problem, snap, k, cpu_dev, kind, backend, sgd_scale, recorder)
        b = one_step(problem, snap, k, cuda, kind, backend, sgd_scale, recorder)
        replay = all(
            torch.equal(a["state"][key], cpu["trajectory"][k + 1][key]) for key in a["state"]
        )
        after = state_drift(b["state"], a["state"], snap["state"], problem.gauge)
        out.append(
            {
                "step": k,
                "cpu_teleport_matches_cpu_trajectory": replay,
                "loss_relative": rel(b["loss"], a["loss"]),
                "grad": grad_compare(b["grads"], a["grads"]),
                "after_step_worst_plain": after["worst_plain"],
                "after_step_worst_update": after["worst_update"],
                "after_step_worst_update_non_gauge": after["worst_update_non_gauge"],
                "after_step_global_l2_update_relative": after["global_l2_update_relative"],
            }
        )
    return out


def diagnose(problem, args, cuda, recorder) -> dict[str, Any]:
    cpu_dev = torch.device("cpu")
    report: dict[str, Any] = {"steps": problem.steps, "configs": {}}
    # Replica fidelity: on CPU the replica Adam loop must equal the real fit bitwise.
    with recorder.capture():
        real_cpu = problem.real_fit(cpu_dev)
    replica = run(problem, cpu_dev, "adam", "default", args.sgd_lr_scale, False, recorder)
    report["replica_matches_real_fit_cpu"] = repeat_check(replica["trajectory"][-1], real_cpu)
    init = replica["trajectory"][0]
    real_gpu = None
    if cuda is not None:
        # The failing test, reproduced with the real fit functions and its own metric.
        with recorder.capture():
            real_gpu = problem.real_fit(cuda)
        real_gpu_again = None
        if problem.name.startswith("critic"):
            with recorder.capture():
                real_gpu_again = problem.real_fit(cuda)
        d = drift_all(problem, real_gpu, real_cpu, init)
        report["real_fit_cpu_vs_cuda"] = {
            **key_metrics(d),
            "test_metric": (
                d["worst_plain"]
                if problem.name.startswith("critic")
                else d["worst_plain_nonzero_ref"]
            ),
            "cuda_repeat": (
                None if real_gpu_again is None else repeat_check(real_gpu, real_gpu_again)
            ),
        }
    for backend in args.backends:
        for kind in ("adam", "sgd"):
            label = f"{backend}/{kind}"
            entry: dict[str, Any] = {}
            cpu = run(
                problem, cpu_dev, kind, backend, args.sgd_lr_scale, cuda is not None, recorder
            )
            cpu_again = run(problem, cpu_dev, kind, backend, args.sgd_lr_scale, False, recorder)
            entry["cpu_losses"] = cpu["losses"]
            entry["cpu_repeat"] = repeat_check(cpu["trajectory"][-1], cpu_again["trajectory"][-1])
            entry["cpu_seconds"] = cpu["seconds"]
            entry["cpu_update_l2"] = math.sqrt(
                sum(
                    float((cpu["trajectory"][-1][k].double() - init[k].double()).square().sum())
                    for k in init
                    if init[k].is_floating_point()
                )
            )
            if backend == "default" and args.proxy_levels:
                # CPU-only amplification proxy: same fit, gradients perturbed per step.
                entry["cpu_perturbation_proxy"] = {}
                for mode, level in (
                    (m, x) for m in ("multiplicative", "additive") for x in args.proxy_levels
                ):
                    noisy = run(
                        problem,
                        cpu_dev,
                        kind,
                        backend,
                        args.sgd_lr_scale,
                        False,
                        recorder,
                        perturb=level,
                        perturb_mode=mode,
                    )
                    d = drift_all(problem, noisy["trajectory"][-1], cpu["trajectory"][-1], init)
                    tag = str(level) if mode == "multiplicative" else f"additive:{level}"
                    entry["cpu_perturbation_proxy"][tag] = {
                        **key_metrics(d),
                        "max_step_loss_relative": max(
                            rel(x, y) for x, y in zip(noisy["losses"], cpu["losses"], strict=True)
                        ),
                        "per_step_worst_plain": [
                            state_drift(noisy["trajectory"][k], cpu["trajectory"][k], init)[
                                "worst_plain"
                            ]["value"]
                            for k in range(1, len(cpu["trajectory"]))
                        ],
                    }
            if cuda is None:
                entry["cuda"] = "skipped: torch.cuda.is_available() is False"
                report["configs"][label] = entry
                continue
            gpu = run(problem, cuda, kind, backend, args.sgd_lr_scale, False, recorder)
            gpu_again = run(problem, cuda, kind, backend, args.sgd_lr_scale, False, recorder)
            final = drift_all(problem, gpu["trajectory"][-1], cpu["trajectory"][-1], init)
            entry.update(
                {
                    "cuda_seconds": gpu["seconds"],
                    "per_step": trajectory_report(problem, cpu, gpu),
                    "final_drift": final,
                    "cuda_repeat": {
                        **repeat_check(gpu["trajectory"][-1], gpu_again["trajectory"][-1]),
                        "losses_bitwise": gpu["losses"] == gpu_again["losses"],
                        "tolerance": 1e-5,
                    },
                    "teleport": teleport_report(
                        problem, cpu, kind, backend, args.sgd_lr_scale, cuda, recorder
                    ),
                }
            )
            if backend == "default" and kind == "adam":
                entry["replica_cuda_vs_real_fit_cuda"] = repeat_check(
                    gpu["trajectory"][-1], real_gpu
                )
            report["configs"][label] = entry
    report["summary"] = summarize(report)
    return report


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for label, entry in report["configs"].items():
        proxy = {
            level: {
                k: v[k]
                for k in (
                    "worst_plain",
                    "worst_update_non_gauge",
                    "global_l2_update_relative",
                    "function",
                )
            }
            for level, v in entry.get("cpu_perturbation_proxy", {}).items()
        }
        if "final_drift" not in entry:
            out[label] = {"cpu_repeat_bitwise": entry["cpu_repeat"]["bitwise"]}
            if proxy:
                out[label]["cpu_perturbation_proxy"] = proxy
            continue
        tele = entry["teleport"]
        out[label] = {
            "final": key_metrics(entry["final_drift"]),
            "max_step_loss_relative": max(s["loss_relative"] for s in entry["per_step"]),
            "teleport_max_loss_relative": max(t["loss_relative"] for t in tele),
            "teleport_max_grad_global_l2_relative": max(
                t["grad"].get("global_l2_relative", math.inf) for t in tele
            ),
            "teleport_max_grad_worst_tensor_relative": max(
                t["grad"].get("worst_tensor_relative", math.inf) for t in tele
            ),
            "teleport_max_after_step_global_l2_update_relative": max(
                (t["after_step_global_l2_update_relative"] or 0.0) for t in tele
            ),
            "teleport_cpu_replays_trajectory": all(
                t["cpu_teleport_matches_cpu_trajectory"] for t in tele
            ),
            "cuda_repeat_bitwise": entry["cuda_repeat"]["bitwise"],
            "cuda_repeat_max_relative": entry["cuda_repeat"]["max_relative"],
        }
        if proxy:
            out[label]["cpu_perturbation_proxy"] = proxy
    return out


# --------------------------------------------------------------------------------------
# Environment and the native-JIT probe
# --------------------------------------------------------------------------------------

PROBE = r"""
import torch
a = torch.randn(4, 8, 1, device="cuda", requires_grad=True)
b = torch.randn(4, 1, 16, device="cuda", requires_grad=True)
torch.bmm(a, b).sum().backward()
torch.cuda.synchronize()
print("bmm outer product forward/backward OK")
"""


def native_state() -> dict[str, Any]:
    out: dict[str, Any] = {"TORCH_DISABLE_NATIVE_JIT": os.environ.get("TORCH_DISABLE_NATIVE_JIT")}
    try:
        from torch._native.common_utils import check_native_jit_disabled

        out["check_native_jit_disabled"] = bool(check_native_jit_disabled())
    except Exception as error:
        out["check_native_jit_disabled"] = f"unavailable: {error!r}"
    try:
        from torch._native import registry

        out["triton_ops_registered"] = list(registry.get_dsl_operations("triton"))
    except Exception as error:
        out["triton_ops_registered"] = f"unavailable: {error!r}"
    try:
        from torch._native import triton_utils

        out["triton_runtime_available"] = bool(triton_utils.runtime_available())
    except Exception as error:
        out["triton_runtime_available"] = f"unavailable: {error!r}"
    return out


def probe_native(timeout: float) -> dict[str, Any]:
    results = {}
    for label, disable in (("native_jit_enabled", False), ("native_jit_disabled", True)):
        env = dict(os.environ)
        env.pop("TORCH_DISABLE_NATIVE_JIT", None)
        if disable:
            env["TORCH_DISABLE_NATIVE_JIT"] = "1"
        try:
            proc = subprocess.run(
                [sys.executable, "-c", PROBE],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
            results[label] = {"returncode": proc.returncode, "tail": tail}
        except subprocess.TimeoutExpired:
            results[label] = {"returncode": None, "tail": [f"timeout after {timeout}s"]}
    return results


def environment(repo: Path) -> dict[str, Any]:
    import anthropocene_rl

    cuda = torch.cuda.is_available()
    record = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "num_threads": torch.get_num_threads(),
        "anthropocene_rl": str(Path(anthropocene_rl.__file__).resolve().parent),
        "repo": str(repo),
        "native": native_state(),
    }
    try:
        record["git_head"] = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        ).stdout.strip()
    except Exception:
        record["git_head"] = None
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT, help="checkout with tests/")
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    parser.add_argument(
        "--problems",
        default="critic,critic_monte_carlo,advantage,bc",
        help="comma list from critic, critic_monte_carlo, critic_separate_qol, advantage, bc",
    )
    parser.add_argument("--backends", default="default,math", help="SDPA: default and/or math")
    parser.add_argument(
        "--sgd-lr-scale",
        type=float,
        default=10.0,
        help="SGD learning rate = fit learning rate x this (Adam steps are ~lr per element)",
    )
    parser.add_argument(
        "--probe-native",
        action="store_true",
        help="CUDA only: run a bmm outer-product backward in subprocesses "
        "with and without TORCH_DISABLE_NATIVE_JIT",
    )
    parser.add_argument("--probe-timeout", type=float, default=600.0)
    parser.add_argument("--cpu-only", action="store_true", help="skip CUDA even if available")
    parser.add_argument(
        "--self-test-second-device-cpu",
        action="store_true",
        help="run the CUDA code paths with CPU as the second device (sanity)",
    )
    parser.add_argument(
        "--proxy-levels",
        default="1e-7,1e-6",
        help="relative gradient perturbations for the CPU amplification proxy ('' to skip)",
    )
    args = parser.parse_args(argv)
    args.backends = [b for b in args.backends.split(",") if b]
    args.proxy_levels = [float(x) for x in args.proxy_levels.split(",") if x]
    if set(args.backends) - {"default", "math"}:
        parser.error("backends must be default and/or math")
    repo = args.repo.resolve()

    from anthropocene_rl.autonomous_v7 import labor_device as ld

    cuda = None
    report: dict[str, Any] = {"environment": environment(repo)}
    if args.self_test_second_device_cpu:
        # Exercises every CUDA code path with CPU as the "second device": all drifts are 0.
        cuda = torch.device("cpu")
        report["self_test_second_device_cpu"] = True
    elif torch.cuda.is_available() and not args.cpu_only:
        cuda = ld.setup("cuda")  # exactly as the tests: cuDNN deterministic, TF32 off, warn_only
        report["configure"] = ld.configure(cuda)
        if args.probe_native:
            report["native_probe"] = probe_native(args.probe_timeout)
    report["sgd_lr_scale"] = args.sgd_lr_scale
    report["problems"] = {}
    recorder = Recorder()
    for name in args.problems.split(","):
        started = time.time()
        torch.manual_seed(0)
        problem = PROBLEMS[name]()
        try:
            report["problems"][name] = diagnose(problem, args, cuda, recorder)
        except Exception as error:
            import traceback

            report["problems"][name] = {
                "error": repr(error),
                "traceback": traceback.format_exc().splitlines()[-12:],
            }
        report["problems"][name]["seconds"] = time.time() - started
        print(f"[diag] {name}: {time.time() - started:.1f}s", file=sys.stderr, flush=True)
    report["warnings"] = dict(recorder.warnings)
    report["native_after"] = native_state()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    brief = {
        name: (
            {"error": value["error"]}
            if "error" in value
            else {
                "replica_matches_real_fit_cpu": value["replica_matches_real_fit_cpu"]["bitwise"],
                **(
                    {"real_fit_test_metric": value["real_fit_cpu_vs_cuda"]["test_metric"]}
                    if "real_fit_cpu_vs_cuda" in value
                    else {}
                ),
                "summary": value["summary"],
            }
        )
        for name, value in report["problems"].items()
    }
    print(json.dumps(brief, indent=1, default=str))
    return 0 if all("error" not in v for v in report["problems"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
