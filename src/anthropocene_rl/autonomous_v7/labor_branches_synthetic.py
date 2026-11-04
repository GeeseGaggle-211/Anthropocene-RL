"""Synthetic stand-in for the simulator, for tests and the horizon check's synthetic mode.

``SyntheticNative`` is a real ``NativeSearch`` subclass whose ``extend`` mirrors the real
one's structure (restore the root fork, first action then H12-or-policy continuation,
keyed stream dither, ``record_boundary`` at each boundary) over a small deterministic
dynamical system instead of the simulator. It is NOT a model of the game and carries
no evidence about it; it exists to exercise the paired-branch plumbing (CRN, export
format, row recording, determinism) without the simulator binary.

State: x in R^3 (one numpy vector). Per decision with action vector u:
``x' = 0.8 x + u + n`` where the noise ``n`` is keyed by (root key, decision) only, hence
common to every branch (CRN). Reward components are smooth in (x, u).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from . import labor_search
from .labor_bc import LaborResetOptions
from .labor_search import DITHER_TAG, Branch, NativeSearch, SearchConfig

NOISE_TAG = 0x534E  # "SN"
ROOT_TAG = 0x5352  # "SR"
YEARS = 20.0
SYNTH_FOCAL = 7
H12_PAYLOAD = {"synthetic": "h12"}


def action_vector(payload: Mapping[str, Any]) -> np.ndarray:
    """``synthetic_vector`` if given, else three digest-derived values in [-1, 1]."""
    if "synthetic_vector" in payload:
        return np.asarray(payload["synthetic_vector"], dtype=np.float64)
    h = labor_search.payload_digest(payload)
    return np.asarray([int(h[i : i + 6], 16) / 0xFFFFFF * 2.0 - 1.0 for i in (0, 6, 12)])


class SyntheticNative(NativeSearch):
    def __init__(
        self,
        *,
        seed: int = 0,
        world: int = -1,
        config: SearchConfig | None = None,
        death_radius: float | None = None,
    ) -> None:
        config = config or SearchConfig(candidates=4, actor_samples=0)
        super().__init__(None, None, world=world, options=LaborResetOptions(), config=config)
        self.seed = int(seed)
        self.death_radius = death_radius
        self.state = {"x": np.zeros(3), "turn": 0}
        self._forks: dict[int, dict[str, Any]] = {}
        self._next = 1
        self.successor_view = self._view

    # -- state -----------------------------------------------------------------
    def make_root(self, index: int) -> int:
        """A deterministic synthetic root state; returns its fork id."""
        x = np.random.default_rng([ROOT_TAG, self.seed, int(index)]).normal(size=3)
        self.state = {"x": x, "turn": 4 + int(index)}
        return self.fork()

    def fork(self) -> int:
        self.work.forks += 1
        fid = self._next
        self._next += 1
        self._forks[fid] = {"x": self.state["x"].copy(), "turn": self.state["turn"]}
        return fid

    def restore(self, fork: int) -> Mapping[str, Any]:
        self.work.restores += 1
        saved = self._forks[fork]
        self.state = {"x": saved["x"].copy(), "turn": saved["turn"]}
        return self._step(False)

    def drop(self, fork: int | None) -> None:
        if fork is not None:
            self._forks.pop(fork, None)

    def catalogs(self) -> dict[int, Any]:
        return {SYNTH_FOCAL: None}

    def root_h12(self) -> dict[int, Any]:
        return {SYNTH_FOCAL: SimpleNamespace(action=dict(H12_PAYLOAD))}

    @staticmethod
    def _view(village: Mapping[str, Any]) -> dict[str, Any]:
        o = village["observation"]
        return {
            "tensors": {"core": torch.tensor([[*o["x"], float(o["turn"])]], dtype=torch.float32)},
            "privileged": None,
        }

    def _step(self, terminal: bool) -> dict[str, Any]:
        village = {
            "camp": SYNTH_FOCAL,
            "observation": {"x": [float(v) for v in self.state["x"]], "turn": self.state["turn"]},
            "terminated": bool(terminal),
        }
        return {"turn": self.state["turn"], "villages": [village]}

    # -- the branch loop (mirrors NativeSearch.extend) ---------------------------
    def extend(
        self,
        branch: Branch,
        target: int,
        *,
        root_fork: int,
        focal: int,
        candidate: Mapping[str, Any],
        root_h12: Mapping[int, Any],
        root_key: Sequence[int],
        keep_fork: bool,
        focal_policy: Any | None = None,
    ) -> None:
        if branch.horizon >= target or not branch.alive:
            return
        step = self.restore(branch.fork if branch.fork is not None else root_fork)
        before = self.work.resolves
        while branch.horizon < target and branch.alive:
            exempt = branch.horizon == 0
            if branch.horizon == 0:
                payload = dict(candidate)
            else:
                self.work.shadow_steps += 1
                x = self.state["x"]
                label = SimpleNamespace(action={"synthetic_vector": list(-0.3 * x)})
                payload = dict(label.action)
                chosen = None
                if focal_policy is not None:
                    chosen = focal_policy(branch.horizon, label, None, step)
                if chosen is not None:
                    payload, exempt = dict(chosen), True
            u = action_vector(payload)
            if branch.stream > 0 and not exempt:
                draws = np.random.default_rng(
                    [DITHER_TAG, *root_key, branch.stream, branch.horizon, focal]
                ).normal(size=3)
                u = u + self.config.dither_fraction * draws
            noise = 0.1 * np.random.default_rng(
                [NOISE_TAG, self.seed, *root_key, branch.horizon]
            ).normal(size=3)
            x = 0.8 * self.state["x"] + u + noise
            self.state = {"x": x, "turn": self.state["turn"] + 1}
            self.work.labor_steps += 1
            r = np.asarray([-float(x @ x), float(x.sum()), -0.1 * float(u @ u), 0.0])
            terminal = self.death_radius is not None and float(np.linalg.norm(x)) > self.death_radius
            branch.returns = branch.returns + branch.discount * r
            branch.steps.append([float(v) for v in r])
            branch.years.append(YEARS)
            if terminal:
                branch.terminated = True
                branch.alive = False
            branch.discount *= math.exp(-YEARS / self.config.discount_years)
            branch.horizon += 1
            step = self._step(terminal)
            if branch.record and self.successor_view is not None:
                self.record_boundary(branch, step, focal, terminal=terminal, live=not terminal)
        branch.resolves += self.work.resolves - before
