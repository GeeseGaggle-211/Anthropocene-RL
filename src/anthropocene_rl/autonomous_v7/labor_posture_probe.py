"""Diagnostic posture probe: forced fission/trade posture arms on native worlds (never training).

The PPO pilot's iteration-0 rollout (worlds 91100-91102, 580 decision rows) showed the
actor's non-labor posture collapsed: P(fission non-neutral) ~ 4e-10, export trade share
sigmoid(-9.2) ~ 1e-4, and 2 of 580 rows realised any trade. This probe asks what the
engine does when that posture is forced: from a checkpoint it runs N worlds x T decision
turns with the native bridge under a matrix of posture "arms". Each arm overrides the
submitted non-labor action of EVERY camp (after the policy sampled it):

* ``fission``: the fission policy lever (0 Neutral, 1 Discourage, 2 Encourage);
* ``share``: the export labour share ``tradeShare`` in [0, 1] (one scalar per camp in the
  schema-12 action: the actor's ``trade_share`` latent is one-dimensional, so a per-group
  share does not exist on the wire);
* ``gate``: the per-group offer gate (0 withhold, 1 offer stock, 2 offer and make to order),
  one value for all 7 groups or 7 slash-separated values;
* ``value``: the per-group relative value in ``contracts.TRADE_VALUE_RANGE``, likewise.

Labor actions stay exactly what the policy samples. Each world's sampling generator is
seeded from (``--sampling-seed``, world) and consumed in collect's order (initial labor,
initial non-labor, then any retry labor), so the arms of one world differ only by the
override until their states diverge. The override is applied to the wire payload only;
the retained samples/log-probs are the policy's own (the probe never forms a ratio).

Per arm, world and decision turn it records: live camps and population before/after,
terminated camps, vanished camps (live at k, absent at k+1), fission daughters (camps new
at k+1 whose ``parentSociety`` > 0; there is no separate engine fission report in the
step), the mean QoL-years/population-years transition rewards of the acted camps, the
successor villages' mean QoL, and realised trade read from the RAW successor observation:
the trade set (``contracts.SETS`` "trade", 7 groups x 10, simulator
``ElderModel9.TradeFeatureNames``) columns 6/7/8 = imported/exported/realised utility,
summed per group over present rows, plus counts of camps with any trade (any nonzero
6/7/8 or a nonzero village ``tradeUtility``) and camps with any present contact row. It also
records the policy's own sampled posture next to the submitted one, and, when the native
bridge returns decision diagnostics, the engine's realised fission/tradeShare.

Feasibility (does forced export labour make the labor compiler reject or repair more?): per
turn, the feasibility-rejected first proposals (retries), retries still invalid, initial
attempts carrying violations, native fallbacks with their reasons, the typed violation
reasons (``LaborViolation.reason|phase``) of initial and retry attempts, and the
``feasibility_initial`` / ``feasibility_retry`` / ``settlement_guard`` values exactly as
``labor_training`` stores them per rollout row (sum, mean, nonzero count). Default: 16
decision turns (the next PPO run's rollout length).

Scope: every output carries ``scope: "diagnostic-posture-override"``
(:data:`labor_worlds.DIAGNOSTIC_POSTURE_SCOPE`); train, collect, the anchor producer and
calibration refuse any input with that scope. The probe writes no rollout, anchor or
checkpoint. Sealed (3000-3059) and reserved held-out (90154-90159) worlds are refused.

Usage::

    python -m anthropocene_rl.autonomous_v7.labor_posture_probe --checkpoint start.pt \
        --worlds 91300,91301 --turns 8 --players 16 --width 128 --height 64 \
        --separation 9 [--arm name:share=0.2,fission=2 ...] [--jobs 7] [--threads 1] \
        [--progress] --out probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contracts
from .labor_bc import LaborResetOptions
from .labor_model import (
    labor_checkpoint_personality,
    labor_log_prob,
    labor_policy_tensors,
    nonlabor_log_prob,
    sample_labor_policy,
    sample_nonlabor_policy,
    schema12_initial_action,
    schema12_retry_action,
)
from .labor_personality import personality_for_world
from .labor_ppo import RetainedLaborAction, execute_labor_decision
from .labor_worlds import DIAGNOSTIC_POSTURE_SCOPE, require_training_worlds
from .protocol import seeds as checked_seeds
from .provenance import checkpoint_hash

PROBE_SCHEMA = "labor-posture-probe-v1"
SET_NAMES = [name for name, _, _ in contracts.SETS]
TRADE_SET = SET_NAMES.index("trade")
CONTACT_SET = SET_NAMES.index("contacts")
TRADE_COLUMNS = {"imported": 6, "exported": 7, "realisedUtility": 8}
FISSION_NAMES = ("neutral", "discourage", "encourage")
GROUPS = len(contracts.TRADE_GROUPS)
DEFAULT_WORLDS = (91300, 91301)
DEFAULT_TURNS = 16  # the next PPO run's rollout length (approved 16 decision turns)


@dataclass(frozen=True)
class PostureArm:
    """One forced posture; ``None`` fields keep the policy's own sample."""

    name: str
    fission: int | None = None
    trade_share: float | None = None
    trade_gate: tuple[int, ...] | None = None
    trade_value: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or any(c in self.name for c in ":,= "):
            raise ValueError(f"posture arm name {self.name!r} must be nonempty without ':,= '")
        if self.fission is not None and (
            isinstance(self.fission, bool)
            or not isinstance(self.fission, int)
            or not 0 <= self.fission < contracts.FISSION_POLICIES
        ):
            raise ValueError("posture fission must be 0 (neutral), 1 (discourage) or 2 (encourage)")
        if self.trade_share is not None and not (
            isinstance(self.trade_share, (int, float))
            and not isinstance(self.trade_share, bool)
            and math.isfinite(float(self.trade_share))
            and 0.0 <= float(self.trade_share) <= 1.0
        ):
            raise ValueError("posture trade share must be in [0, 1]")
        if self.trade_gate is not None and (
            len(self.trade_gate) != GROUPS
            or any(
                isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < contracts.TRADE_GATES
                for v in self.trade_gate
            )
        ):
            raise ValueError(f"posture trade gate must be {GROUPS} values in 0..2")
        low, high = contracts.TRADE_VALUE_RANGE
        if self.trade_value is not None and (
            len(self.trade_value) != GROUPS
            or any(
                not math.isfinite(float(v)) or not low <= float(v) <= high
                for v in self.trade_value
            )
        ):
            raise ValueError(f"posture trade value must be {GROUPS} values in [{low}, {high}]")

    @property
    def is_baseline(self) -> bool:
        return all(
            value is None
            for value in (self.fission, self.trade_share, self.trade_gate, self.trade_value)
        )

    def apply(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """The submitted initial payload: ``payload`` with this arm's fields replaced."""
        out = dict(payload)
        if self.fission is not None:
            out["fission"] = int(self.fission)
        if self.trade_share is not None:
            out["tradeShare"] = float(self.trade_share)
        if self.trade_gate is not None:
            out["tradeGate"] = [int(v) for v in self.trade_gate]
        if self.trade_value is not None:
            out["tradeValue"] = [float(v) for v in self.trade_value]
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fission": self.fission,
            "tradeShare": None if self.trade_share is None else float(self.trade_share),
            "tradeGate": None if self.trade_gate is None else list(self.trade_gate),
            "tradeValue": (
                None if self.trade_value is None else [float(v) for v in self.trade_value]
            ),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> PostureArm:
        return cls(
            state["name"],
            state.get("fission"),
            state.get("tradeShare"),
            None if state.get("tradeGate") is None else tuple(int(v) for v in state["tradeGate"]),
            None
            if state.get("tradeValue") is None
            else tuple(float(v) for v in state["tradeValue"]),
        )

    @classmethod
    def parse(cls, text: str) -> PostureArm:
        """``name`` or ``name:key=value,...`` with keys fission (0/1/2 or neutral/
        discourage/encourage), share, gate (one value or 7 slash-separated), value (same)."""
        name, _, rest = text.partition(":")
        fields: dict[str, Any] = {}
        for item in filter(None, rest.split(",")):
            key, eq, value = item.partition("=")
            if not eq:
                raise ValueError(f"posture arm field {item!r} must be key=value")
            key = key.strip()
            if key == "fission":
                fields["fission"] = (
                    FISSION_NAMES.index(value) if value in FISSION_NAMES else int(value)
                )
            elif key == "share":
                fields["trade_share"] = float(value)
            elif key in ("gate", "value"):
                parts = value.split("/")
                if len(parts) == 1:
                    parts = parts * GROUPS
                cast = int if key == "gate" else float
                fields["trade_gate" if key == "gate" else "trade_value"] = tuple(
                    cast(p) for p in parts
                )
            else:
                raise ValueError(f"unknown posture arm field {key!r}")
        return cls(name, **fields)


DEFAULT_ARMS = (
    PostureArm("baseline"),
    PostureArm("share-0.05", trade_share=0.05),
    PostureArm("share-0.2", trade_share=0.2),
    PostureArm("share-0.5", trade_share=0.5),
    PostureArm("fission-encourage", fission=2),
    PostureArm("fission-discourage", fission=1),
    PostureArm("share-0.2-encourage", trade_share=0.2, fission=2),
)


def world_sampling_seed(sampling_seed: int, world: int) -> int:
    """The per-world sampling seed: identical for every arm of that world."""
    sequence = np.random.SeedSequence([int(sampling_seed), int(world), 0x5057])
    return int(sequence.generate_state(1)[0])


def _live(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        village
        for village in step["villages"]
        if village.get("observation") is not None and not village.get("terminated")
    ]


def raw_trade(village: Mapping[str, Any]) -> np.ndarray:
    """[7 groups, 3] raw imported/exported/realised utility from the village observation
    (absent rows are zero)."""
    out = np.zeros((GROUPS, len(TRADE_COLUMNS)), dtype=np.float64)
    obs = village.get("observation") or {}
    sets, present = obs.get("sets"), obs.get("rowPresent")
    if not sets or not present or len(sets) <= TRADE_SET:
        return out
    _, rows, width = contracts.SETS[TRADE_SET]
    table = np.asarray(sets[TRADE_SET], dtype=np.float64).reshape(rows, width)
    live = np.asarray(present[TRADE_SET], dtype=bool)[:rows]
    columns = list(TRADE_COLUMNS.values())
    out[live] = table[live][:, columns]
    return out


def contact_rows(village: Mapping[str, Any]) -> int:
    obs = village.get("observation") or {}
    present = obs.get("rowPresent")
    if not present or len(present) <= CONTACT_SET:
        return 0
    _, rows, _ = contracts.SETS[CONTACT_SET]
    return int(np.asarray(present[CONTACT_SET], dtype=bool)[:rows].sum())


def _posture_summary(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not payloads:
        return {"camps": 0}
    shares = [float(p["tradeShare"]) for p in payloads]
    gates = np.asarray([p["tradeGate"] for p in payloads], dtype=np.int64)
    values = np.asarray([p["tradeValue"] for p in payloads], dtype=np.float64)
    return {
        "camps": len(payloads),
        "fission": {
            FISSION_NAMES[k]: int(sum(int(p["fission"]) == k for p in payloads))
            for k in range(contracts.FISSION_POLICIES)
        },
        "tradeShareMean": float(np.mean(shares)),
        "tradeShareMax": float(np.max(shares)),
        "tradeGateCounts": {
            name: [int((gates[:, g] == k).sum()) for k in range(contracts.TRADE_GATES)]
            for g, name in enumerate(contracts.TRADE_GROUPS)
        },
        "tradeValueMean": [float(v) for v in values.mean(axis=0)],
    }


def _engine_realised(bridge: Any) -> dict[str, Any] | None:
    """The native decision diagnostics' realised fission/tradeShare, when exposed."""
    inner = getattr(bridge, "_bridge", None)
    records = getattr(inner, "decision_diagnostics", None)
    if not records:
        return None
    fission: Counter[int] = Counter()
    shares: list[float] = []
    for record in records:
        realised = record.get("realised") if isinstance(record, Mapping) else None
        if not isinstance(realised, Mapping):
            continue
        if "fission" in realised:
            fission[int(np.atleast_1d(realised["fission"])[0])] += 1
        if "tradeShare" in realised:
            shares.append(float(np.atleast_1d(realised["tradeShare"])[0]))
    return {
        "records": len(records),
        "realisedFission": {str(k): v for k, v in sorted(fission.items())},
        "realisedTradeShareMean": float(np.mean(shares)) if shares else None,
    }


def _violation_reasons(attempts: Sequence[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for attempt in attempts:
        for violation in getattr(attempt, "violations", ()) or ():
            counts[f"{violation.reason}|{violation.phase}"] += 1
    return dict(sorted(counts.items()))


def feasibility_record(
    execution: Any,
    acted: Sequence[int],
    present: Mapping[int, Mapping[str, Any]],
    founded: Mapping[int, int],
    successor_turn: int,
) -> dict[str, Any]:
    """The labor compiler's interventions for one decision, as collect stores them per row:
    ``feasibility_initial`` (the initial attempt's feasibility cost), ``feasibility_retry``
    (the final attempt's cost when the first proposal was rejected, else 0) and
    ``settlement_guard`` (``labor_training._settlement_guard`` on the successor), plus the
    rejected first proposals (retries), retries still invalid, fallbacks and the typed
    violation reasons (``LaborViolation.reason|phase``) of initial and retry attempts."""
    from .labor_training import _settlement_guard

    initial = dict(getattr(execution, "initial_attempts", {}) or {})
    final = dict(getattr(execution, "final_attempts", {}) or {})
    retried = set(getattr(execution, "retry", {}) or {})
    feedback = dict(getattr(execution, "feedback", {}) or {})
    feas_initial = [float(initial[c].feasibility_cost) for c in acted if c in initial]
    feas_retry = [
        float(final[c].feasibility_cost) if c in retried and c in final else 0.0
        for c in acted if c in initial
    ]
    guard = [
        _settlement_guard(present[c], founded.get(c, successor_turn), successor_turn)
        for c in acted if c in present
    ]
    fallback_reasons: Counter[str] = Counter(
        row.fallback_reason for row in feedback.values() if getattr(row, "fallback", False)
    )

    def stats(values: Sequence[float]) -> dict[str, float | int]:
        return {
            "sum": float(np.sum(values)) if values else 0.0,
            "mean": float(np.mean(values)) if values else 0.0,
            "nonzero": int(sum(v != 0 for v in values)),
        }

    return {
        "attemptsKnown": bool(initial),
        "rejectedFirstProposals": int(sum(not initial[c].valid for c in initial)),
        "retriesStillInvalid": int(sum(not final[c].valid for c in retried if c in final)),
        "initialWithViolations": int(sum(bool(a.violations) for a in initial.values())),
        "fallbacks": int(sum(fallback_reasons.values())),
        "fallbackReasons": dict(sorted(fallback_reasons.items())),
        "initialViolationReasons": _violation_reasons(list(initial.values())),
        "retryViolationReasons": _violation_reasons(
            [final[c] for c in retried if c in final]
        ),
        "feasibilityInitial": stats(feas_initial),
        "feasibilityRetry": stats(feas_retry),
        "settlementGuard": stats(guard),
    }


def probe_world(
    *,
    bridge: Any,
    actor: Any,
    normalizer: Any,
    personality: Any,
    arm: PostureArm,
    world: int,
    world_index: int,
    turns: int,
    reset: LaborResetOptions,
    sampling_seed: int,
    execute: Callable[..., Any] = execute_labor_decision,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one world for ``turns`` decisions under ``arm``; returns its per-turn records."""
    from .labor_training import _one_row, _transition_rewards

    clock = time.perf_counter()
    step = bridge.reset(world, **reset.reset_kwargs())
    opening = [int(v["camp"]) for v in _live(step)]
    founded = {int(v["camp"]): int(step["turn"]) for v in step["villages"]}
    world_personality = personality_for_world(personality, world)
    generator = torch.Generator().manual_seed(world_sampling_seed(sampling_seed, world))
    records: list[dict[str, Any]] = []
    for _ in range(turns):
        live = _live(step)
        if not live:
            break
        contexts, tensors = labor_policy_tensors(live, normalizer, personality=world_personality)
        catalogs = {context[0].camp_id: context[0] for context in contexts}
        if len(catalogs) != len(live):
            raise ValueError("posture probe: duplicate live camp identities")
        with torch.no_grad():
            output = actor(tensors)
            labor = sample_labor_policy(output, tensors, generator=generator)
            nonlabor = sample_nonlabor_policy(output, generator=generator)
            labor_lp = labor_log_prob(output, labor, tensors)["joint"]
            nonlabor_lp = nonlabor_log_prob(output, nonlabor)["joint"]
        retained: dict[int, RetainedLaborAction] = {}
        sampled_payloads, submitted_payloads = [], []
        village_by_camp = {}
        for row, (village, context) in enumerate(zip(live, contexts, strict=True)):
            camp = int(village["camp"])
            if context[0].camp_id != camp:
                raise ValueError("posture probe: observation/catalog camp mismatch")
            village_by_camp[camp] = village
            sampled = schema12_initial_action(labor, nonlabor, row)
            submitted = arm.apply(sampled)
            sampled_payloads.append(sampled)
            submitted_payloads.append(submitted)
            retained[camp] = RetainedLaborAction(
                submitted,
                _one_row(labor, row),
                float(labor_lp[row]),
                _one_row(nonlabor, row),
                float(nonlabor_lp[row]),
            )

        def retry_factory(
            camp: int, catalog: Any, attempt: Any, villages: Mapping[int, Any] = village_by_camp
        ) -> RetainedLaborAction:
            retry_contexts, retry_tensors = labor_policy_tensors(
                [villages[camp]],
                normalizer,
                personality=world_personality,
                rejected_attempts=[attempt],
            )
            if retry_contexts[0][0].digest != catalog.digest:
                raise ValueError("posture probe: retry context differs from the rejection")
            with torch.no_grad():
                retry_output = actor(retry_tensors)
                sample = sample_labor_policy(retry_output, retry_tensors, generator=generator)
                logp = float(labor_log_prob(retry_output, sample, retry_tensors)["joint"][0])
            return RetainedLaborAction(
                schema12_retry_action(sample, 0), _one_row(sample, 0), logp, None, None
            )

        execution = execute(bridge, catalogs, retained, retry_factory)
        successor = execution.step
        decision_turn = int(step["turn"])
        rewards, terminal, _ = _transition_rewards(
            successor, tuple(catalogs), founded, world_index=world_index,
            decision_turn=decision_turn,
        )
        before = {int(v["camp"]) for v in step["villages"]}
        after_live = _live(successor)
        present = {int(v["camp"]): v for v in successor["villages"]}
        acted = list(catalogs)
        daughters = [
            camp for camp, v in present.items()
            if camp not in before and int(v.get("parentSociety", 0) or 0) > 0
        ]
        other_new = [
            camp for camp, v in present.items()
            if camp not in before and int(v.get("parentSociety", 0) or 0) <= 0
        ]
        trade = np.zeros((GROUPS, len(TRADE_COLUMNS)), dtype=np.float64)
        rows_with_trade = 0
        trade_utility = 0.0
        for village in after_live:
            table = raw_trade(village)
            trade += table
            utility = float(village.get("tradeUtility", 0.0) or 0.0)
            trade_utility += utility
            rows_with_trade += int(bool(np.any(table != 0)) or utility != 0.0)
        record = {
            "turn": decision_turn,
            "campsBefore": len(live),
            "populationBefore": int(sum(int(v["population"]) for v in live)),
            "campsAfter": len(after_live),
            "populationAfter": int(sum(int(v["population"]) for v in after_live)),
            "terminated": sorted(c for c in acted if terminal.get(c)),
            "vanished": sorted(c for c in acted if c not in present),
            "fissionDaughters": [
                {"camp": c, "parentSociety": int(present[c]["parentSociety"])}
                for c in sorted(daughters)
            ],
            "otherNewCamps": sorted(other_new),
            "meanQolReward": float(np.mean([rewards[c]["qol_years"] for c in acted])),
            "meanPopulationReward": float(
                np.mean([rewards[c]["population_years"] for c in acted])
            ),
            "meanQolAfter": (
                float(np.mean([float(v["qol"]) for v in after_live])) if after_live else None
            ),
            "trade": {
                name: {
                    group: float(trade[g, column])
                    for g, group in enumerate(contracts.TRADE_GROUPS)
                }
                for column, name in enumerate(TRADE_COLUMNS)
            },
            "tradeTotals": {
                name: float(trade[:, column].sum()) for column, name in enumerate(TRADE_COLUMNS)
            },
            "campsWithTrade": rows_with_trade,
            "villageTradeUtility": trade_utility,
            "campsWithContactBefore": int(sum(contact_rows(v) > 0 for v in live)),
            "retries": len(getattr(execution, "retry", {}) or {}),
            "feasibility": feasibility_record(
                execution, acted, present, founded, int(successor["turn"])
            ),
            "policyPosture": _posture_summary(sampled_payloads),
            "submittedPosture": _posture_summary(submitted_payloads),
            "engineRealised": _engine_realised(bridge),
        }
        records.append(record)
        if progress is not None:
            progress(
                {"arm": arm.name, "world": world, "turn": decision_turn,
                 "campsAfter": record["campsAfter"], "campsWithTrade": rows_with_trade,
                 "daughters": len(daughters), "seconds": round(time.perf_counter() - clock, 3)}
            )
        step = successor
    final_live = {int(v["camp"]) for v in _live(step)}
    return {
        "arm": arm.name,
        "world": int(world),
        "samplingSeed": world_sampling_seed(sampling_seed, world),
        "openingCamps": opening,
        "decisions": len(records),
        "turns": records,
        "final": {
            "camps": len(final_live),
            "population": int(sum(int(v["population"]) for v in _live(step))),
            "foundersAlive": int(sum(camp in final_live for camp in opening)),
        },
        "seconds": time.perf_counter() - clock,
    }


def summarize_arm(worlds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Totals of one arm over its worlds."""
    turns = [t for w in worlds for t in w["turns"]]
    camp_decisions = sum(t["campsBefore"] for t in turns)
    trade = {
        name: {
            group: float(sum(t["trade"][name][group] for t in turns))
            for group in contracts.TRADE_GROUPS
        }
        for name in TRADE_COLUMNS
    }

    def weighted(key: str) -> float | None:
        if not camp_decisions:
            return None
        return float(sum(t[key] * t["campsBefore"] for t in turns) / camp_decisions)

    return {
        "worlds": len(worlds),
        "decisionTurns": len(turns),
        "campDecisions": camp_decisions,
        "campsWithTrade": int(sum(t["campsWithTrade"] for t in turns)),
        "tradeRowShare": (
            None if not camp_decisions
            else sum(t["campsWithTrade"] for t in turns) / camp_decisions
        ),
        "trade": trade,
        "tradeTotals": {name: float(sum(trade[name].values())) for name in TRADE_COLUMNS},
        "villageTradeUtility": float(sum(t["villageTradeUtility"] for t in turns)),
        "campsWithContact": int(sum(t["campsWithContactBefore"] for t in turns)),
        "fissionDaughters": int(sum(len(t["fissionDaughters"]) for t in turns)),
        "otherNewCamps": int(sum(len(t["otherNewCamps"]) for t in turns)),
        "terminated": int(sum(len(t["terminated"]) for t in turns)),
        "vanished": int(sum(len(t["vanished"]) for t in turns)),
        "meanQolReward": weighted("meanQolReward"),
        "meanPopulationReward": weighted("meanPopulationReward"),
        "finalCamps": int(sum(w["final"]["camps"] for w in worlds)),
        "finalPopulation": int(sum(w["final"]["population"] for w in worlds)),
        "foundersAlive": int(sum(w["final"]["foundersAlive"] for w in worlds)),
        "openingCamps": int(sum(len(w["openingCamps"]) for w in worlds)),
        "feasibility": {
            "rejectedFirstProposals": int(
                sum(t["feasibility"]["rejectedFirstProposals"] for t in turns)
            ),
            "rejectedShare": (
                None if not camp_decisions
                else sum(t["feasibility"]["rejectedFirstProposals"] for t in turns)
                / camp_decisions
            ),
            "retriesStillInvalid": int(sum(t["feasibility"]["retriesStillInvalid"] for t in turns)),
            "initialWithViolations": int(
                sum(t["feasibility"]["initialWithViolations"] for t in turns)
            ),
            "fallbacks": int(sum(t["feasibility"]["fallbacks"] for t in turns)),
            "initialViolationReasons": _merge_counts(
                t["feasibility"]["initialViolationReasons"] for t in turns
            ),
            "retryViolationReasons": _merge_counts(
                t["feasibility"]["retryViolationReasons"] for t in turns
            ),
            "fallbackReasons": _merge_counts(t["feasibility"]["fallbackReasons"] for t in turns),
            **{
                key: {
                    "sum": float(sum(t["feasibility"][key]["sum"] for t in turns)),
                    "perCampDecision": (
                        None if not camp_decisions
                        else sum(t["feasibility"][key]["sum"] for t in turns) / camp_decisions
                    ),
                    "nonzero": int(sum(t["feasibility"][key]["nonzero"] for t in turns)),
                }
                for key in ("feasibilityInitial", "feasibilityRetry", "settlementGuard")
            },
        },
    }


def _merge_counts(items: Any) -> dict[str, int]:
    total: Counter[str] = Counter()
    for item in items:
        total.update(item)
    return dict(sorted(total.items()))


# --- process plumbing (one spawned process + bridge per (arm, world), as the anchor producer)


def _probe_unit(job: Mapping[str, Any]) -> dict[str, Any]:
    from .labor_training import _models_from_checkpoint, labor_behavior_configuration
    from .value_calibration import behavior_policy_digest

    path = Path(job["checkpointPath"])
    if checkpoint_hash(path) != job["checkpointSha256"]:
        raise ValueError("posture probe worker: checkpoint changed during the probe")
    state = torch.load(path, map_location="cpu", weights_only=True)
    _, actor, _, _, normalizer = _models_from_checkpoint(state)
    personality = labor_checkpoint_personality(state)
    behavior = labor_behavior_configuration(personality)
    if behavior_policy_digest(actor, behavior) != job["policyDigest"]:
        raise ValueError("posture probe worker loaded another sampled policy")

    def progress(item: dict[str, Any]) -> None:
        print(json.dumps({"postureProgress": item}, sort_keys=True), file=sys.stderr, flush=True)

    bridge = job["bridgeFactory"]()
    try:
        return probe_world(
            bridge=bridge, actor=actor, normalizer=normalizer, personality=personality,
            arm=PostureArm.from_state(job["arm"]), world=job["world"],
            world_index=job["worldIndex"], turns=job["turns"], reset=job["reset"],
            sampling_seed=job["samplingSeed"], execute=job.get("execute", execute_labor_decision),
            progress=progress if job["progress"] else None,
        )
    finally:
        bridge.close()


def _default_executor(max_workers: int, threads: int | None) -> Any:
    from .labor_anchor_producer import _default_executor as anchor_executor

    return anchor_executor(max_workers, threads)


class _InlineExecutor:
    """``--jobs 1``: the units in this process, in canonical order."""

    def __init__(self, workers: int, threads: int | None) -> None:
        del workers, threads

    def submit(self, fn: Callable[..., Any], *args: Any) -> Any:
        from concurrent.futures import Future

        future: Future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as error:
            future.set_exception(error)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        del wait, cancel_futures


def run_posture_probe(
    *,
    checkpoint_path: Path,
    worlds: Sequence[int],
    turns: int,
    arms: Sequence[PostureArm] = DEFAULT_ARMS,
    reset: LaborResetOptions,
    sampling_seed: int = 0,
    jobs: int = 1,
    threads: int | None = None,
    progress: bool = False,
    bridge_factory: Any = None,
    executor_factory: Any = None,
    execute: Callable[..., Any] | None = None,
    unit_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from .labor_training import _models_from_checkpoint, labor_behavior_configuration
    from .value_calibration import behavior_policy_digest

    worlds = require_training_worlds(
        checked_seeds(list(worlds), "posture probe worlds"), "posture probe"
    )
    if not worlds or len(set(worlds)) != len(worlds):
        raise ValueError("posture probe needs distinct world seeds")
    if isinstance(turns, bool) or not isinstance(turns, int) or turns <= 0:
        raise ValueError("posture probe turns must be a positive integer")
    if not arms or len({arm.name for arm in arms}) != len(arms):
        raise ValueError("posture probe arms must be nonempty with distinct names")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("--jobs must be a positive integer")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _, actor, _, _, _ = _models_from_checkpoint(state)
    personality = labor_checkpoint_personality(state)
    policy_digest = behavior_policy_digest(actor, labor_behavior_configuration(personality))
    sha = checkpoint_hash(checkpoint_path)
    if bridge_factory is None:
        from .labor_bridge import LaborModel9Bridge

        bridge_factory = LaborModel9Bridge
    job_list = []
    for arm in arms:
        for world_index, world in enumerate(worlds):
            job = {
                "checkpointPath": str(checkpoint_path),
                "checkpointSha256": sha,
                "policyDigest": policy_digest,
                "bridgeFactory": bridge_factory,
                "arm": arm.state_dict(),
                "world": int(world),
                "worldIndex": world_index,
                "turns": int(turns),
                "reset": reset,
                "samplingSeed": int(sampling_seed),
                "progress": bool(progress),
            }
            if execute is not None:
                job["execute"] = execute
            job_list.append(job)
    if executor_factory is None:
        executor_factory = _InlineExecutor if jobs == 1 else _default_executor
    executor = executor_factory(min(jobs, len(job_list)), threads)
    try:
        futures = [executor.submit(_probe_unit, job) for job in job_list]
        if unit_sink is not None:  # each finished unit as it lands (partial results on a cap)
            from concurrent.futures import as_completed

            for future in as_completed(futures):
                unit_sink(future.result())
        results = [future.result() for future in futures]  # canonical (arm, world) order
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    by_arm: dict[str, list[dict[str, Any]]] = {arm.name: [] for arm in arms}
    for result in results:
        by_arm[result["arm"]].append(result)
    return {
        "schema": PROBE_SCHEMA,
        "scope": DIAGNOSTIC_POSTURE_SCOPE,
        "trainingUse": "forbidden: posture forced for every camp; not policy samples",
        "checkpointSha256": sha,
        "behaviorPolicyDigest": policy_digest,
        "checkpointIteration": int(state.get("iteration", -1)),
        "worlds": [int(w) for w in worlds],
        "turns": int(turns),
        "resetOptions": reset.state_dict(),
        "samplingSeed": int(sampling_seed),
        "samplingOrder": "per world: initial labor, initial non-labor, retry labor (collect's)",
        "override": "applied to the submitted wire payload of every camp after sampling",
        "tradeColumns": dict(TRADE_COLUMNS),
        "tradeGroups": list(contracts.TRADE_GROUPS),
        "fissionDetection": "camps new in the successor step with parentSociety > 0",
        "arms": [arm.state_dict() for arm in arms],
        "results": {
            name: {str(r["world"]): r for r in rows} for name, rows in by_arm.items()
        },
        "summary": {name: summarize_arm(rows) for name, rows in by_arm.items()},
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--worlds", default=",".join(map(str, DEFAULT_WORLDS)), help="comma-separated seeds"
    )
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    parser.add_argument(
        "--arm", action="append", default=None,
        help="posture arm 'name[:fission=0|1|2,share=S,gate=G[/G..x7],value=V[/V..x7]]'; "
        "repeatable (default: baseline, share 0.05/0.2/0.5, fission encourage/discourage, "
        "share 0.2 + encourage)",
    )
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--players", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--separation", type=int, default=9)
    parser.add_argument("--no-movement", action="store_true", default=False)
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="parallel (arm, world) units, one spawned process + bridge each (default 1: inline)",
    )
    parser.add_argument("--threads", type=int, default=1, help="torch threads per process")
    parser.add_argument("--progress", action="store_true", default=False)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _write_unit(out: Path, unit: Mapping[str, Any]) -> None:
    """``<out>.units/<arm>-<world>.json``: one finished (arm, world) unit, scope-tagged, so a
    run cut by its time cap still leaves every completed unit."""
    directory = out.with_name(out.name + ".units")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{unit['arm']}-{unit['world']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"scope": DIAGNOSTIC_POSTURE_SCOPE, "schema": PROBE_SCHEMA, "unit": unit},
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    torch.set_num_threads(args.threads)
    arms = DEFAULT_ARMS if not args.arm else tuple(PostureArm.parse(a) for a in args.arm)
    result = run_posture_probe(
        checkpoint_path=args.checkpoint,
        worlds=[int(w) for w in args.worlds.split(",") if w],
        turns=args.turns,
        arms=arms,
        reset=LaborResetOptions(
            players=args.players, movement=not args.no_movement, width=args.width,
            height=args.height, separation=args.separation,
        ),
        sampling_seed=args.sampling_seed,
        jobs=args.jobs,
        threads=args.threads,
        progress=args.progress,
        unit_sink=lambda unit: _write_unit(args.out, unit),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps({"postureSummary": result["summary"]}, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
