"""A small fake multi-camp labor world for WP-B (paired branches during collection).

``FakeLaborBridge`` stands in for ``LaborModel9Bridge`` in ``collect_labor_rollout``: reset,
fork/restore/drop with exact state copies, and a deterministic transition. Its execution
replaces ``labor_training.execute_labor_decision`` (``install(monkeypatch)``), so the REAL
rollout path (``_execute_single_step_detailed`` with the real actor, critics, retry
factory and reward helpers) and the REAL branch collector run over it. It is NOT a model
of the game: villages are the privileged-PPO test fixture with per-camp catalogs; the
privileged view (critic input) varies with the state; the transition is keyed noise plus
an action-digest effect. Engine randomness is keyed by (world seed, turn, camp) only, so a
restored fork replays exactly (the native contract the branches rely on).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
from test_labor_bridge import _attempt_payload
from test_labor_privileged_bridge import _view
from test_labor_privileged_model import _village

from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7.labor_bridge import LaborAttempt, labor_catalog_digest

CAMPS = (7, 8, 9)
NOISE_TAG = 0x4657  # "FW"


def _catalog_for(camp: int) -> dict:
    catalog = copy.deepcopy(_village(1)["observation"]["laborCatalog"])
    catalog["campId"] = int(camp)
    catalog["digest"] = "0" * 64
    catalog["digest"] = labor_catalog_digest(catalog)
    return catalog


_BASE = _village(1)
_CATALOGS = {camp: _catalog_for(camp) for camp in CAMPS}


def payload_digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=float).encode()).hexdigest()


def _effect(payload: dict) -> float:
    h = payload_digest(payload)
    return int(h[:8], 16) / 0xFFFFFFFF * 2.0 - 1.0


class FakeLaborBridge:
    """Deterministic fake labor bridge with exact fork/restore."""

    def __init__(self, *, reject_rate: float = 0.3, death: dict | None = None, **_: Any) -> None:
        self.reject_rate = float(reject_rate)
        self.death = dict(death or {})  # camp -> turn at which it terminates
        self.state: dict[str, Any] = {}
        self.forks: dict[int, dict[str, Any]] = {}
        self.next_fork = 1
        self.steps = 0
        self.closed = False

    # -- lifecycle -------------------------------------------------------------
    def reset(self, seed: int, **_: Any) -> dict:
        self.state = {
            "seed": int(seed),
            "turn": 3,
            "camps": {c: {"population": 100, "qol": 0.5, "prior": 100, "terminated": False}
                      for c in CAMPS},
        }
        return self._step()

    def close(self) -> None:
        self.closed = True

    def fork(self) -> tuple[int, dict]:
        fid = self.next_fork
        self.next_fork += 1
        self.forks[fid] = copy.deepcopy(self.state)
        return fid, self._step()

    def restore(self, fork: int) -> dict:
        self.state = copy.deepcopy(self.forks[fork])
        return self._step()

    def drop(self, fork: int) -> None:
        self.forks.pop(fork)

    # -- observation -----------------------------------------------------------------
    def _village(self, camp: int, s: dict) -> dict:
        observation = copy.deepcopy(_BASE["observation"])
        observation["laborCatalog"] = copy.deepcopy(_CATALOGS[camp])
        return {
            "camp": camp,
            "society": 70 + camp,
            "parentSociety": 0,
            "priorPopulation": int(s["prior"]),
            "population": int(s["population"]),
            "qol": float(s["qol"]),
            "years": 20.0,
            "welfare": float(s["qol"]) * float(s["prior"]) * 20.0,
            "terminated": bool(s["terminated"]),
            "observation": None if s["terminated"] else observation,
            "privilegedSchemaVersion": 1,
            "privilegedView": _view(
                int(s["population"]) % 7 + camp, scale=1.0 + 0.01 * int(s["population"])
            ),
        }

    def _step(self) -> dict:
        return {
            "turn": int(self.state["turn"]),
            "villages": [self._village(c, self.state["camps"][c]) for c in CAMPS],
        }

    def _noise(self, camp: int) -> np.ndarray:
        return np.random.default_rng(
            [NOISE_TAG, self.state["seed"], self.state["turn"], camp]
        ).normal(size=2)

    def _rejected(self, camp: int, payload: dict) -> bool:
        h = hashlib.sha256(f"{self.state['seed']}:{self.state['turn']}:{camp}:"
                           f"{payload_digest(payload)}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF < self.reject_rate

    # -- execution (installed as labor_training.execute_labor_decision) ---------------
    def execute(self, catalogs, retained, retry_factory):
        if set(retained) != set(catalogs):
            raise ValueError("initial labor actions must cover exactly the current camps")
        initial, final, retries = {}, {}, {}
        executed = {}
        for camp in sorted(catalogs):
            payload = dict(retained[camp].payload)
            rejected = self._rejected(camp, payload)
            attempt_payload = _attempt_payload(_CATALOGS[camp], rejected=rejected)
            attempt_payload["campId"] = camp
            attempt = LaborAttempt.from_payload(attempt_payload, catalogs[camp])
            initial[camp] = attempt
            if rejected:
                retries[camp] = retry_factory(camp, catalogs[camp], attempt)
                final[camp] = dataclasses.replace(attempt, attempt=1, remaining_proposals=0)
                merged = dict(payload)
                merged.update(retries[camp].payload)
                executed[camp] = merged
            else:
                final[camp] = attempt
                executed[camp] = payload
        for camp in sorted(catalogs):
            s = self.state["camps"][camp]
            noise = self._noise(camp)
            effect = _effect(executed[camp])
            s["prior"] = s["population"]
            s["population"] = max(0, round(s["population"] + 4.0 * effect + 2.0 * noise[0]))
            s["qol"] = float(np.clip(0.5 + 0.2 * effect + 0.05 * noise[1], 0.0, 1.0))
            if self.death.get(camp) == self.state["turn"] or s["population"] <= 0:
                s["terminated"] = True
        self.state["turn"] += 1
        self.steps += 1
        return SimpleNamespace(
            step=self._step(), initial_attempts=initial, final_attempts=final, retry=retries
        )


def fake_execute(bridge, catalogs, retained, retry_factory):
    return bridge.execute(catalogs, retained, retry_factory)


def install(monkeypatch) -> None:
    monkeypatch.setattr(training, "execute_labor_decision", fake_execute)


def fake_shadow_factory(actor, normalizer):
    """A stand-in H12 shadow: a keyed actor sample per (world, turn, camp) (a valid
    schema-12 payload; not H12). Counts its calls on the bridge."""
    import torch

    from anthropocene_rl.autonomous_v7.labor_branch_collect import keyed_generator
    from anthropocene_rl.autonomous_v7.labor_model import (
        labor_policy_tensors,
        sample_labor_policy,
        sample_nonlabor_policy,
        schema12_initial_action,
    )

    def factory(bridge, world, options):
        def shadow(step, catalogs):
            bridge.shadow_calls = getattr(bridge, "shadow_calls", 0) + 1
            live = [v for v in step["villages"]
                    if v.get("observation") is not None and not v.get("terminated")]
            _, tensors = labor_policy_tensors(live, normalizer)
            with torch.no_grad():
                output = actor(tensors)
            labels = {}
            for row, village in enumerate(live):
                camp = int(village["camp"])
                one = training._one_row(output, row)
                t = training._one_row(tensors, row)
                g = keyed_generator(0x4848, world, int(step["turn"]), camp)
                labor = sample_labor_policy(one, t, generator=g)
                nonlabor = sample_nonlabor_policy(one, generator=g)
                labels[camp] = SimpleNamespace(action=schema12_initial_action(labor, nonlabor, 0))
            return labels

        return shadow, lambda: None

    return factory
