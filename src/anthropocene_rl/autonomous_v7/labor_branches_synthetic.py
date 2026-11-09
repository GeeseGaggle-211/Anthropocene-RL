"""Synthetic stand-in for the simulator, for tests and the horizon check's synthetic mode.

``SyntheticNative`` is a ``NativeSearch`` subclass that overrides ONLY the simulator
primitives ``extend`` calls: ``fork``/``restore``/``drop``, ``catalogs``, ``shadow_h12``
(the H12 labels), ``execute`` (the one-decision resolve) and ``dithered``. The real,
unmodified ``NativeSearch.extend`` runs over them, including the real
``_transition_rewards``, ``_settlement_guard``, ``_lineage_components``,
``record_boundary`` and the focal-policy call. It is NOT a model of the game and carries
no evidence about it. Stubbed: the simulator, label conversion, catalogs (None) and
``dithered`` (a keyed perturbation of the action vector standing in for the real
``dither_payload`` + validation; it does check that ``extend`` passes the right key and
exemption).

State: x in R^3 (one numpy vector). Per decision with action vector u:
``x' = 0.8 x + u + n`` where the noise ``n`` is keyed by (root key, decision) only, hence
common to every branch (CRN). Reward components are smooth in (x, u).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from . import labor_search
from .labor_bc import LaborResetOptions
from .labor_search import DITHER_TAG, NativeSearch, SearchConfig

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

    def _step(self, terminal: bool, x_prev: np.ndarray | None = None, u=None) -> dict[str, Any]:
        x = self.state["x"]
        prior = 50
        qol = float(1.0 - 0.1 * (x @ x))
        village = {
            "camp": SYNTH_FOCAL,
            "society": 1,
            "parentSociety": 0,
            "priorPopulation": prior,
            "population": 0 if terminal else prior,
            "qol": qol,
            "years": YEARS,
            "welfare": qol * prior * YEARS,
            "terminated": bool(terminal),
            "observation": {"x": [float(v) for v in x], "turn": self.state["turn"]},
            "reward_x": None if u is None else [float(v) for v in u],
        }
        return {"turn": self.state["turn"], "villages": [village]}

    # -- stubbed simulator primitives (extend itself is NativeSearch.extend) ------
    def shadow_h12(self, step: Mapping[str, Any], catalogs: Mapping[int, Any]) -> dict[int, Any]:
        self.work.shadow_steps += 1
        x = self.state["x"]
        return {SYNTH_FOCAL: SimpleNamespace(action={"synthetic_vector": list(-0.3 * x)})}

    def dithered(self, payloads, catalogs, *, key, exempt):
        out = {camp: dict(p) for camp, p in payloads.items()}
        for camp in sorted(payloads):
            draws = np.random.default_rng([DITHER_TAG, *key, camp]).normal(size=3)
            if camp == exempt:
                continue
            u = action_vector(payloads[camp]) + self.config.dither_fraction * draws
            out[camp] = {"synthetic_vector": [float(v) for v in u]}
        return out

    def execute(self, catalogs, payloads):
        self.work.labor_steps += 1
        u = action_vector(payloads[SYNTH_FOCAL])
        noise = 0.1 * np.random.default_rng([NOISE_TAG, self.seed, self.state["turn"]]).normal(
            size=3
        )
        x = 0.8 * self.state["x"] + u + noise
        self.state = {"x": x, "turn": self.state["turn"] + 1}
        terminal = self.death_radius is not None and float(np.linalg.norm(x)) > self.death_radius
        return SimpleNamespace(
            step=self._step(terminal, u=u),
            initial_attempts={
                SYNTH_FOCAL: SimpleNamespace(valid=True, feasibility_cost=0.1 * float(u @ u))
            },
        )
