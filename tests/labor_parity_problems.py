"""Replica step loops of the three device-parity fixtures (critic, A head, actor BC).

Each problem replays a stored-data fit of ``tests/test_labor_device.py`` with an explicit
step loop that is bitwise equal to the real fit function on CPU, so a state (parameters,
optimizer moments, TD targets) at any step can be loaded on another device and stepped
once ("teleported" parity). Shared by ``tests/test_labor_device.py``,
``tests/test_cuda_parity_diag.py`` and ``tools/cuda_parity_diag.py``. Not a test module.
"""

from __future__ import annotations

import contextlib
import copy
import time
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch import nn

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7.labor_device_parity import (
    compact,
    function_drift,
    state_drift,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    LABOR_ACTIVITY_FEATURE_WIDTH,
    LABOR_RESOURCE_FEATURE_WIDTH,
    LABOR_SPECIALTY_FEATURE_WIDTH,
    LaborCostCritics,
    LaborEncoderLayout,
    LaborQoLHead,
    LaborTaskCritic,
)

MAX_GRAD_NORM = 1.0


def _critic_pair(seed: int = 0, target: str = "td-lambda"):
    from test_labor_privileged_model import _layout

    torch.manual_seed(seed)
    task = LaborTaskCritic(
        _layout(False), width=16, token_width=16, qol_head=LaborQoLHead(target=target)
    )
    costs = LaborCostCritics(_layout(False), width=16, token_width=16)
    return task, costs


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
        core_dim,
        LABOR_SPECIALTY_FEATURE_WIDTH,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
    )
    return layout, dataset


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
    problem,
    device,
    kind="adam",
    backend="default",
    sgd_scale=10.0,
    snapshots=False,
    recorder=None,
    perturb: float = 0.0,
    perturb_mode: str = "multiplicative",
) -> dict[str, Any]:
    """One fit, as a CPU-comparable trajectory of states.

    ``perturb`` > 0 perturbs every gradient element before clipping, a CPU stand-in for
    fp32 kernel error: ``multiplicative`` scales it by (1 + perturb * N(0, 1)), which keeps
    every sign; ``additive`` adds perturb * rms(tensor gradient) * N(0, 1) to every nonzero
    element, which flips the sign of elements whose gradient is roundoff-sized (as a
    reassociated sum can); exact zeros stay zero, as they do on every device.
    """
    from anthropocene_rl.autonomous_v7.labor_device import cpu_copy

    recorder = recorder or Recorder()

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
                        if p.grad is None:
                            continue
                        z = torch.randn(p.grad.shape, generator=noise).to(p.grad)
                        if perturb_mode == "additive":
                            # Exact zeros (unused parameters, masked paths) stay exact.
                            scale = perturb * p.grad.square().mean().sqrt()
                            p.grad.add_(scale * z * (p.grad != 0))
                        else:
                            p.grad.mul_(1 + perturb * z)
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


def one_step(
    problem, snap, k, device, kind="adam", backend="default", sgd_scale=10.0, recorder=None
) -> dict[str, Any]:
    """One step on ``device`` from a trajectory snapshot (state, optimizer, extra) at step k."""
    recorder = recorder or Recorder()
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


def drift_all(problem, test_state, ref_state, init_state) -> dict[str, Any]:
    """Parameter drift (all three metrics, gauge-aware) plus output-space drift."""
    d = state_drift(test_state, ref_state, init_state, problem.gauge)
    d["function"] = compact(
        function_drift(
            problem.outputs(test_state), problem.outputs(ref_state), problem.outputs(init_state)
        )
    )
    return d


def teleport(problem, k: int, device, kind: str = "adam") -> dict[str, Any]:
    """Step k of the CPU trajectory replayed on CPU and on ``device`` from the same state.

    Returns the loss relative difference and :func:`grad_compare` of the pre-clip gradients
    (``global_l2_relative`` is the gated number; ``worst_tensor_relative`` hits gauge
    tensors and is reported only).
    """
    from anthropocene_rl.autonomous_v7.labor_device_parity import grad_compare, rel

    cpu = torch.device("cpu")
    trajectory = run(problem, cpu, kind, snapshots=True)
    snap = trajectory["snapshots"][k]
    a = one_step(problem, snap, k, cpu, kind)
    b = one_step(problem, snap, k, device, kind)
    return {
        "step": k,
        "loss_relative": rel(b["loss"], a["loss"]),
        **grad_compare(b["grads"], a["grads"]),
    }
