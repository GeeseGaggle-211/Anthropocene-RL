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
on the stock cuBLAS kernel. This script sets it before importing torch unless
``--keep-native-jit`` is given, records whether it took effect, and (with
``--probe-native``) runs a subprocess without it to show whether the compiler
failure reproduces.

Usage (GPU VM): ``cd <repo> && PYTHONPATH=src python tools/cuda_parity_diag.py --out diag.json``.
On a CPU-only host the CUDA parts are skipped and only the CPU runs and the
replica-versus-real validation execute.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__" and "--keep-native-jit" not in sys.argv:
    # Must precede ``import torch``: torch._native reads it once while importing.
    os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

import argparse
import contextlib
import copy
import json
import math
import platform
import subprocess
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_DEFAULT = Path(__file__).resolve().parents[1]
MAX_GRAD_NORM = 1.0
EPS = 1e-12


def _fixtures(repo: Path) -> None:
    tests = str(repo / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def _f(x: float | None) -> float | None:
    if x is None or not math.isfinite(x):
        return None if x is None else (float("inf") if x > 0 else float("-inf"))
    return float(x)


def state_drift(
    test: dict[str, torch.Tensor],
    ref: dict[str, torch.Tensor],
    init: dict[str, torch.Tensor],
    gauge=None,
) -> dict[str, Any]:
    """Drift of ``test`` (CUDA) against ``ref`` (CPU) with ``init`` the shared start.

    ``gauge(key, tensor)`` returns a boolean mask of elements whose loss gradient is
    analytically zero (they do not change the fitted function); the ``*_non_gauge``
    metrics exclude them.
    """
    per: dict[str, dict[str, Any]] = {}
    diff_sq = ref_sq = upd_sq = 0.0
    for key, b in ref.items():
        if not b.is_floating_point():
            continue
        a = test[key].double()
        b = b.double()
        i = init[key].double()
        d = float((a - b).abs().max()) if b.numel() else 0.0
        bmax = float(b.abs().max()) if b.numel() else 0.0
        imax = float(i.abs().max()) if b.numel() else 0.0
        umax = float((b - i).abs().max()) if b.numel() else 0.0
        diff_sq += float((a - b).square().sum())
        ref_sq += float(b.square().sum())
        upd_sq += float((b - i).square().sum())
        per[key] = {
            "numel": int(b.numel()),
            "max_abs_diff": d,
            "max_abs_ref": bmax,
            "max_abs_init": imax,
            "max_abs_update": umax,
            "zero_init": imax == 0.0,
            "plain": d / max(bmax, EPS),
            "scale": d / max(bmax + imax, EPS),
            "update": (d / umax) if umax > 0 else (0.0 if d == 0 else math.inf),
        }
        mask = None if gauge is None else gauge(key, b)
        if mask is not None and bool(mask.any()):
            keep = ~mask
            per[key]["gauge_elements"] = int(mask.sum())
            if bool(keep.any()):
                dn = float((a - b)[keep].abs().max())
                bn = float(b[keep].abs().max())
                un = float((b - i)[keep].abs().max())
                per[key]["plain_non_gauge"] = dn / max(bn, EPS)
                per[key]["update_non_gauge"] = (
                    (dn / un) if un > 0 else (0.0 if dn == 0 else math.inf)
                )
            else:
                per[key]["plain_non_gauge"] = per[key]["update_non_gauge"] = 0.0

    def worst(metric: str, keys=None) -> dict[str, Any]:
        base = metric.replace("_non_gauge", "")
        items = [(k, v.get(metric, v[base])) for k, v in per.items() if keys is None or k in keys]
        if not items:
            return {"value": None, "tensor": None}
        k, v = max(items, key=lambda kv: kv[1])
        return {"value": _f(v), "tensor": k}

    nonzero_ref = {k for k, v in per.items() if v["max_abs_ref"] > 0}
    return {
        "worst_plain": worst("plain"),
        "worst_plain_nonzero_ref": worst("plain", nonzero_ref),  # tests 2/3 exclude zero refs
        "worst_scale": worst("scale"),
        "worst_update": worst("update"),
        "worst_plain_non_gauge": worst("plain_non_gauge"),
        "worst_update_non_gauge": worst("update_non_gauge"),
        "global_l2_relative": math.sqrt(diff_sq) / max(math.sqrt(ref_sq), EPS),
        "global_l2_update_relative": (
            math.sqrt(diff_sq) / math.sqrt(upd_sq) if upd_sq > 0 else None
        ),
        "per_tensor": per,
    }


def function_drift(
    test: dict[str, torch.Tensor], ref: dict[str, torch.Tensor], init: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Output-space drift on a fixed probe batch (every fixture row), evaluated on CPU."""
    per = {}
    diff_sq = change_sq = ref_sq = 0.0
    for key, b in ref.items():
        a, i = test[key], init[key]
        d = float((a - b).abs().max())
        change = float((b - i).abs().max())
        diff_sq += float((a - b).square().sum())
        change_sq += float((b - i).square().sum())
        ref_sq += float(b.square().sum())
        per[key] = {
            "plain": d / max(float(b.abs().max()), EPS),
            "change_relative": (d / change) if change > 0 else (0.0 if d == 0 else math.inf),
            "max_abs_change": change,
        }
    worst_change = max(per.items(), key=lambda kv: kv[1]["change_relative"])
    worst_plain = max(per.items(), key=lambda kv: kv[1]["plain"])
    return {
        "worst_change_relative": {
            "value": _f(worst_change[1]["change_relative"]),
            "output": worst_change[0],
        },
        "worst_plain": {"value": _f(worst_plain[1]["plain"]), "output": worst_plain[0]},
        "global_l2_relative": math.sqrt(diff_sq) / max(math.sqrt(ref_sq), EPS),
        "global_l2_change_relative": (
            math.sqrt(diff_sq) / math.sqrt(change_sq) if change_sq > 0 else None
        ),
        "per_output": per,
    }


def compact(drift: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in drift.items() if k not in ("per_tensor", "per_output")}


def drift_all(problem, test_state, ref_state, init_state) -> dict[str, Any]:
    """Parameter drift (all three metrics, gauge-aware) plus output-space drift."""
    d = state_drift(test_state, ref_state, init_state, problem.gauge)
    d["function"] = compact(
        function_drift(
            problem.outputs(test_state), problem.outputs(ref_state), problem.outputs(init_state)
        )
    )
    return d


def repeat_check(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    bitwise = all(torch.equal(a[k], b[k]) for k in a)
    worst = 0.0
    for k in a:
        if a[k].is_floating_point() and not torch.equal(a[k], b[k]):
            d = float((a[k].double() - b[k].double()).abs().max())
            worst = max(worst, d / max(float(a[k].double().abs().max()), EPS))
    return {"bitwise": bitwise, "max_relative": worst}


def grad_compare(
    test: dict[str, torch.Tensor | None], ref: dict[str, torch.Tensor | None]
) -> dict[str, Any]:
    worst, worst_key = 0.0, None
    diff_sq = ref_sq = 0.0
    sign_flips = elements = big_rel = 0
    for key, g in ref.items():
        h = test[key]
        if g is None or h is None:
            if (g is None) != (h is None):
                return {"error": f"gradient presence differs at {key}"}
            continue
        g, h = g.double(), h.double()
        gmax = float(g.abs().max())
        d = float((g - h).abs().max())
        diff_sq += float((g - h).square().sum())
        ref_sq += float(g.square().sum())
        nz = (g != 0) | (h != 0)
        elements += int(nz.sum())
        sign_flips += int((torch.sign(g) != torch.sign(h))[nz].sum())
        big_rel += int(((g - h).abs() > 0.1 * g.abs())[nz].sum())
        if gmax > 0 and d / gmax > worst:
            worst, worst_key = d / gmax, key
    return {
        "worst_tensor_relative": worst,
        "worst_tensor": worst_key,
        "global_l2_relative": math.sqrt(diff_sq) / max(math.sqrt(ref_sq), EPS),
        "nonzero_elements": elements,
        "sign_disagreements": sign_flips,
        "elements_rel_diff_gt_10pct": big_rel,
    }


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), EPS)


# --------------------------------------------------------------------------------------
# Problems: one fixture each, a deterministic per-step loss, CPU-initialised modules
# --------------------------------------------------------------------------------------


class Problem:
    name = ""
    steps = 0
    lr = 0.0
    adam = torch.optim.Adam

    def modules(self) -> dict[str, nn.Module]:
        """Fresh CPU copies of the initial modules (prefix -> module)."""
        return {k: copy.deepcopy(v) for k, v in self._init.items()}

    def build(self, device: torch.device, state: dict[str, torch.Tensor] | None = None):
        mods = self.modules()
        if state is not None:
            for prefix, module in self.trainable(mods).items():
                module.load_state_dict(
                    {
                        k[len(prefix) + 1 :]: v
                        for k, v in state.items()
                        if k.startswith(prefix + ".")
                    }
                )
        for module in mods.values():
            module.to(device)
            module.train()
        return self.attach(mods, device)

    def attach(self, mods, device):  # device-dependent extras (A-head contexts)
        return mods

    def trainable(self, mods) -> dict[str, nn.Module]:
        return mods

    def named_params(self, mods) -> list[tuple[str, nn.Parameter]]:
        return [
            (f"{prefix}.{n}", p)
            for prefix, m in self.trainable(mods).items()
            for n, p in m.named_parameters()
        ]

    def state(self, mods) -> dict[str, torch.Tensor]:
        return {
            f"{prefix}.{k}": v.detach().cpu().clone()
            for prefix, m in self.trainable(mods).items()
            for k, v in m.state_dict().items()
        }

    def optimizer(self, params, kind: str, sgd_scale: float):
        if kind == "adam":
            return self.adam(params, lr=self.lr)
        if kind == "sgd":
            return torch.optim.SGD(params, lr=self.lr * sgd_scale)
        raise ValueError(kind)

    def initial_extra(self) -> dict[str, Any]:
        return {}

    def gauge(self, key: str, tensor: torch.Tensor) -> torch.Tensor | None:
        """Elements with an analytically zero loss gradient (default: attention key bias).

        Softmax attention is invariant to the key-projection bias, so its true gradient is
        exactly zero and the computed one is pure roundoff.
        """
        if key.endswith("self_attn.in_proj_bias"):
            e = tensor.shape[0] // 3
            mask = torch.zeros(tensor.shape, dtype=torch.bool)
            mask[e : 2 * e] = True
            return mask
        return None

    def outputs(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Probe outputs of the model in ``state``, on CPU, for every fixture row."""
        mods = self.build(torch.device("cpu"), state)
        with torch.no_grad():
            return {k: v.detach().double() for k, v in self._outputs(mods).items()}

    def _outputs(self, mods) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def step_loss(self, mods, k: int, extra: dict[str, Any], device) -> torch.Tensor:
        raise NotImplementedError

    def real_fit(self, device: torch.device) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class CriticProblem(Problem):
    """tests: test_critic_fit_on_cuda_is_close_to_cpu_and_repeatable (9 Adam steps)."""

    adam = torch.optim.Adam
    lr = 1e-3

    def __init__(self, target: str = "td-lambda") -> None:
        from test_labor_device import _critic_pair
        from test_labor_qol_head import _training_dataset

        from anthropocene_rl.autonomous_v7.labor_bc import (
            critic_target_available,
            discounted_returns,
            world_balanced_weights,
        )
        from anthropocene_rl.autonomous_v7.labor_model import LABOR_COSTS, LABOR_OBJECTIVES
        from anthropocene_rl.autonomous_v7.labor_privileged import actor_inputs

        self.name = "critic" if target == "td-lambda" else "critic_monte_carlo"
        self.target = target
        self._pair = lambda: _critic_pair(0, target)
        task, costs = self._pair()
        self._init = {"task": task, "cost": costs}
        data = _training_dataset()
        data["collection"] = dict(data["collection"], dataset_digest="fixture")
        self.data = data
        returns = discounted_returns(data)
        n = int(data["row_world"].shape[0])
        self.n = n
        keep = ~data["row_quarantined"].reshape(n).bool() & critic_target_available(data)
        kept = torch.nonzero(keep).reshape(-1)
        self.kept = kept
        self.tensor_keys = [
            k
            for k, v in actor_inputs(data).items()
            if isinstance(v, torch.Tensor)
            and not k.startswith(("target_", "label_", "reward_", "transition_", "row_", "tail_"))
        ]
        self.tensors = {k: data[k][kept] for k in self.tensor_keys}
        self.task_targets = {name: returns[name][kept] for name in LABOR_OBJECTIVES}
        self.cost_targets = {name: returns[name][kept] for name in LABOR_COSTS}
        weights = world_balanced_weights(data["row_world"][kept]).double().numpy()
        probabilities = weights / weights.sum()
        N, batch, epochs = len(kept), 2, 3
        generator = np.random.default_rng(7)
        self.batches = []
        for _ in range(epochs):
            draws = generator.choice(N, size=N, replace=True, p=probabilities)
            for start in range(0, N, batch):
                self.batches.append(torch.as_tensor(draws[start : start + batch], dtype=torch.long))
        self.steps = len(self.batches)
        steps_per_epoch = max(1, -(-N // batch))
        self.td_interval = max(1, -(-steps_per_epoch // 4))
        head = getattr(task, "qol_head", None)
        self.td = head is not None and head.target == "td-lambda"

    def _outputs(self, mods):
        task, costs = mods["task"], mods["cost"]
        batch = {k: self.data[k] for k in self.tensor_keys}
        out = {f"task:{n}": task.expectation(n, v) for n, v in task(batch, None).items()}
        out.update({f"cost:{n}": v for n, v in costs(batch, None).items()})
        return out

    def _qol_logits(self, task, device) -> torch.Tensor:
        from anthropocene_rl.autonomous_v7.labor_device import to_device
        from anthropocene_rl.autonomous_v7.labor_model import critic_privileged_batch

        rows = torch.arange(self.n)
        parts = []
        with torch.no_grad():
            for start in range(0, len(rows), 512):
                index = rows[start : start + 512]
                batch = to_device({k: self.data[k][index] for k in self.tensor_keys}, device)
                p = critic_privileged_batch(task, self.data, index, device=device)
                parts.append(task(batch, p)["qol_years"].cpu())
        return torch.cat(parts)

    def step_loss(self, mods, k, extra, device):
        from anthropocene_rl.autonomous_v7.labor_advantage import qol_td_lambda_targets
        from anthropocene_rl.autonomous_v7.labor_device import to_device
        from anthropocene_rl.autonomous_v7.labor_ppo import (
            labor_cost_value_losses,
            labor_task_value_losses,
        )

        task, costs = mods["task"], mods["cost"]
        if self.td and k % self.td_interval == 0:  # k == 0: the fit's initial refresh
            values = task.expectation("qol_years", self._qol_logits(task, device))
            extra["qol"] = qol_td_lambda_targets(self.data, values, task.qol_head.td_lambda)[
                self.kept
            ]
        targets = dict(self.task_targets)
        if self.td:
            targets["qol_years"] = extra["qol"]
        idx = self.batches[k]
        b = to_device({key: v[idx] for key, v in self.tensors.items()}, device)
        task_loss, _, _ = labor_task_value_losses(
            task, b, to_device({key: v[idx] for key, v in targets.items()}, device), None
        )
        cost_loss, _ = labor_cost_value_losses(
            costs, b, to_device({key: v[idx] for key, v in self.cost_targets.items()}, device), None
        )
        return task_loss + 1.0 * cost_loss

    def real_fit(self, device):
        from anthropocene_rl.autonomous_v7.labor_bc import pretrain_critic_on_dataset
        from anthropocene_rl.autonomous_v7.labor_stability import StabilityConfig

        task, costs = self._pair()
        task.to(device)
        costs.to(device)
        pretrain_critic_on_dataset(
            task_critic=task,
            cost_critics=costs,
            dataset=self.data,
            epochs=3,
            batch_size=2,
            rng=np.random.default_rng(7),
            stability=StabilityConfig(log_every=1),
        )
        return self.state({"task": task, "cost": costs})


class AdvantageProblem(Problem):
    """tests: test_advantage_fit_parity_cpu_vs_cuda (AdamW, frozen critics)."""

    name = "advantage"
    adam = torch.optim.AdamW

    def __init__(self) -> None:
        from test_labor_privileged_model import _critics, _layout

        from anthropocene_rl.autonomous_v7 import labor_advantage as la

        self.la = la
        self._critics = lambda: _critics(_layout(False))
        shard = la.synthetic_candidate_shard(
            worlds=tuple(range(90100, 90106)), roots_per_world=3, candidates=5, seed=1
        )
        config = la.AdvantageFitConfig(epochs=3, batch_roots=4, lr=2e-3, seed=0, hidden=32)
        self.config = config
        self.lr = config.lr
        self.candidates = la.parse_candidate_shard(shard)
        fit_worlds, _ = la.split_worlds(
            self.candidates.worlds, config.holdout_fraction, config.seed
        )
        fit_set = self.candidates.subset(fit_worlds)
        self.root_tensors, self.root_privileged = fit_set.root_tensors()
        pairs = {
            name: la.build_pairs(
                fit_set.roots,
                name,
                horizons=config.horizons[name],
                se_multiplier=config.se_multiplier,
            )
            for name in la.COMPONENTS
        }
        flat = la._flatten(fit_set.roots)
        self.flat = flat
        by_root: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for name, p in pairs.items():
            for idx in np.flatnonzero(p.admitted):
                r = int(p.root[idx])
                by_root[r][name].append(
                    (
                        flat.offsets[r] + int(p.i[idx]),
                        flat.offsets[r] + int(p.k[idx]),
                        float(p.delta[idx]),
                    )
                )
        trainable = sorted(by_root)
        task, costs = self._critics()
        torch.manual_seed(config.seed)
        head = la.LaborAdvantageHead.for_critics(
            task, costs, embedding=config.embedding, hidden=config.hidden
        )
        self._init = {"head": head, "task": task, "cost": costs}
        rng = np.random.default_rng([la.SHUFFLE_TAG, int(config.seed)])
        self.batches = []
        for _ in range(config.epochs if trainable else 0):
            order = [trainable[i] for i in rng.permutation(len(trainable))]
            for start in range(0, len(order), config.batch_roots):
                batch = order[start : start + config.batch_roots]
                members = torch.cat(
                    [
                        torch.arange(
                            flat.offsets[r], flat.offsets[r] + len(fit_set.roots[r].candidates)
                        )
                        for r in batch
                    ]
                )
                local = torch.full((len(flat.root_index),), -1, dtype=torch.long)
                local[members] = torch.arange(len(members))
                components = {}
                for name in la.COMPONENTS:
                    triples = [t for r in batch for t in by_root[r][name]]
                    if triples:
                        components[name] = (
                            local[
                                torch.as_tensor([[t[0], t[1]] for t in triples], dtype=torch.long)
                            ],
                            torch.as_tensor([t[2] for t in triples], dtype=torch.float32),
                        )
                self.batches.append((members, components))
        self.steps = len(self.batches)

    def optimizer(self, params, kind, sgd_scale):
        if kind == "adam":  # the fit's AdamW with its configured (zero) weight decay
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.config.weight_decay)
        return super().optimizer(params, kind, sgd_scale)

    def attach(self, mods, device):
        for module in (mods["task"], mods["cost"]):
            for p in module.parameters():
                p.requires_grad_(False)
        mods["contexts"] = self.la.critic_contexts(
            mods["task"], mods["cost"], self.root_tensors, self.root_privileged
        )
        mods["features"] = self.flat.features.to(device)
        mods["root_index"] = self.flat.root_index.to(device)
        return mods

    def trainable(self, mods):
        return {"head": mods["head"]}

    def gauge(self, key, tensor):
        # Last-layer bias of each A head: the pairwise loss sees only A_i - A_k and use is
        # per-root centred, so the bias has exactly zero true gradient. (The head has no
        # attention, so no key bias.)
        if key.startswith("head.heads.") and key.endswith(".2.bias"):
            return torch.ones(tensor.shape, dtype=torch.bool)
        return None

    def _outputs(self, mods):
        raw = mods["head"](mods["contexts"], mods["features"], mods["root_index"])
        roots = len(self.flat.offsets)
        return {
            f"centred_A:{n}": self.la.centre(v, mods["root_index"], roots) for n, v in raw.items()
        }

    def step_loss(self, mods, k, extra, device):
        members, components = self.batches[k]
        on = members.to(device)
        raw = mods["head"](mods["contexts"], mods["features"][on], mods["root_index"][on])
        loss = raw["qol_years"].sum() * 0.0
        for name in self.la.COMPONENTS:
            if name not in components:
                continue
            index, delta = components[name]
            index, delta = index.to(device), delta.to(device)
            loss = loss + self.la.pairwise_loss(
                raw[name][index[:, 0]],
                raw[name][index[:, 1]],
                delta,
                regression_weight=self.config.regression_weight,
            )
        return loss

    def real_fit(self, device):
        from anthropocene_rl.autonomous_v7.labor_stability import StabilityConfig

        task, costs = self._critics()
        for module in (task, costs):
            module.to(device)
            for p in module.parameters():
                p.requires_grad_(False)
        head, _ = self.la.fit_advantage_head(
            task, costs, self.candidates, self.config, stability=StabilityConfig(log_every=1)
        )
        return self.state({"head": head})


class BCProblem(Problem):
    """tests: test_bc_parity_cpu_vs_cuda (6 Adam steps, lr 5e-3)."""

    name = "bc"
    adam = torch.optim.Adam
    lr = 5e-3

    def __init__(self) -> None:
        from test_labor_device import _bc_dataset

        from anthropocene_rl.autonomous_v7.labor_model import LaborActor

        self.layout, self.dataset = _bc_dataset()
        torch.manual_seed(0)
        self._init = {"actor": LaborActor(self.layout)}
        self._actor = lambda: copy.deepcopy(self._init["actor"])
        d = self.dataset
        N = int(d["target_specialty"].shape[0])
        self.tensor_keys = [k for k, v in d.items() if isinstance(v, torch.Tensor)]
        keep = torch.ones(N, dtype=torch.bool)
        valid = d.get("label_specialty_valid")
        if valid is not None:
            keep = valid.reshape(N).bool()
        if "row_quarantined" in d:
            keep = keep & ~d["row_quarantined"].reshape(N).bool()
        if not bool(keep.any()):
            keep = torch.ones(N, dtype=torch.bool)
        mean_spec = d["target_specialty"][keep].mean(dim=0)
        spec = (mean_spec + 0.02).pow(-0.5)
        self.spec_weights = spec / spec.mean().clamp(min=1e-8)
        mean_act = d["target_activity"][keep].mean(dim=0)
        act = (mean_act + 0.02).pow(-0.5)
        self.act_weights = act / act.mean().clamp(min=1e-8)
        generator = np.random.default_rng(3)
        self.batches = []
        for _ in range(3):
            indices = generator.permutation(N)
            for start in range(0, N, 16):
                self.batches.append(
                    torch.as_tensor(indices[start : min(N, start + 16)], dtype=torch.long)
                )
        self.steps = len(self.batches)

    def _outputs(self, mods):
        from anthropocene_rl.autonomous_v7.labor_privileged import actor_inputs

        output = mods["actor"](actor_inputs({k: self.dataset[k] for k in self.tensor_keys}))
        return {
            f"actor:{k}": v
            for k, v in output.items()
            if isinstance(v, torch.Tensor) and v.is_floating_point()
        }

    def step_loss(self, mods, k, extra, device):
        from anthropocene_rl.autonomous_v7.labor_bc import _batch_loss_and_l1
        from anthropocene_rl.autonomous_v7.labor_device import to_device

        idx = self.batches[k]
        batch = to_device({key: self.dataset[key][idx] for key in self.tensor_keys}, device)
        loss, _, _, _ = _batch_loss_and_l1(
            mods["actor"],
            batch,
            target_log_std=-2.0,
            inactive_role_weight=0.0,
            label_smoothing=1e-6,
            spec_weights=to_device(self.spec_weights, device),
            act_weights=to_device(self.act_weights, device),
        )
        return loss

    def real_fit(self, device):
        from anthropocene_rl.autonomous_v7.labor_bc import pretrain_labor_actor_bc
        from anthropocene_rl.autonomous_v7.labor_stability import StabilityConfig

        actor = self._actor().to(device)
        pretrain_labor_actor_bc(
            actor=actor,
            dataset=self.dataset,
            epochs=3,
            batch_size=16,
            lr=5e-3,
            rng=np.random.default_rng(3),
            stability=StabilityConfig(log_every=1),
        )
        return self.state({"actor": actor})


PROBLEMS = {
    "critic": lambda: CriticProblem("td-lambda"),
    "critic_monte_carlo": lambda: CriticProblem("monte-carlo"),
    "advantage": AdvantageProblem,
    "bc": BCProblem,
}


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------


def backend_context(backend: str):
    if backend == "default":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(SDPBackend.MATH)


class Recorder:
    def __init__(self) -> None:
        self.warnings: dict[str, int] = defaultdict(int)

    @contextlib.contextmanager
    def capture(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
        for w in caught:
            self.warnings[f"{w.category.__name__}: {str(w.message)[:200]}"] += 1


def clone_extra(extra: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in extra.items()
    }


def run(
    problem, device, kind, backend, sgd_scale, snapshots, recorder, perturb: float = 0.0
) -> dict[str, Any]:
    """One fit. ``perturb`` > 0 multiplies every gradient element by (1 + perturb * N(0, 1))
    before clipping (a CPU stand-in for fp32 kernel reassociation error)."""
    from anthropocene_rl.autonomous_v7.labor_device import cpu_copy

    noise = torch.Generator().manual_seed(1234)
    started = time.time()
    with backend_context(backend), recorder.capture():
        mods = problem.build(device)
        named = problem.named_params(mods)
        params = [p for _, p in named]
        opt = problem.optimizer(params, kind, sgd_scale)
        extra = problem.initial_extra()
        trajectory = [problem.state(mods)]
        snaps, losses, norms = [], [], []
        for k in range(problem.steps):
            if snapshots:
                snaps.append(
                    {
                        "state": trajectory[-1],
                        "opt": cpu_copy(copy.deepcopy(opt.state_dict())),
                        "extra": clone_extra(extra),
                    }
                )
            opt.zero_grad()
            loss = problem.step_loss(mods, k, extra, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{problem.name}: nonfinite loss at step {k}")
            loss.backward()
            if perturb > 0:
                with torch.no_grad():
                    for p in params:
                        if p.grad is not None:
                            p.grad.mul_(
                                1 + perturb * torch.randn(p.grad.shape, generator=noise).to(p.grad)
                            )
            norms.append(float(nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)))
            opt.step()
            losses.append(float(loss.detach()))
            trajectory.append(problem.state(mods))
    return {
        "losses": losses,
        "grad_norms": norms,
        "trajectory": trajectory,
        "snapshots": snaps,
        "seconds": time.time() - started,
    }


def one_step(problem, snap, k, device, kind, backend, sgd_scale, recorder) -> dict[str, Any]:
    with backend_context(backend), recorder.capture():
        mods = problem.build(device, snap["state"])
        named = problem.named_params(mods)
        params = [p for _, p in named]
        opt = problem.optimizer(params, kind, sgd_scale)
        opt.load_state_dict(copy.deepcopy(snap["opt"]))
        extra = clone_extra(snap["extra"])
        opt.zero_grad()
        loss = problem.step_loss(mods, k, extra, device)
        loss.backward()
        grads = {n: None if p.grad is None else p.grad.detach().cpu().clone() for n, p in named}
        nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
        opt.step()
        return {"loss": float(loss.detach()), "grads": grads, "state": problem.state(mods)}


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
                for level in args.proxy_levels:
                    noisy = run(
                        problem,
                        cpu_dev,
                        kind,
                        backend,
                        args.sgd_lr_scale,
                        False,
                        recorder,
                        perturb=level,
                    )
                    d = drift_all(problem, noisy["trajectory"][-1], cpu["trajectory"][-1], init)
                    entry["cpu_perturbation_proxy"][str(level)] = {
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
        help=f"comma list from {sorted(PROBLEMS)}",
    )
    parser.add_argument("--backends", default="default,math", help="SDPA: default and/or math")
    parser.add_argument(
        "--sgd-lr-scale",
        type=float,
        default=10.0,
        help="SGD learning rate = fit learning rate x this (Adam steps are ~lr per element)",
    )
    parser.add_argument(
        "--keep-native-jit",
        action="store_true",
        help="do not set TORCH_DISABLE_NATIVE_JIT=1 (BC then needs a C compiler)",
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
    _fixtures(repo)

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
