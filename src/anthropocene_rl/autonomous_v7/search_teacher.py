"""Search teacher (expert iteration, docs/MODEL9_SEARCH_TEACHER.md): improves on a policy by simulation and writes the
winning alternatives as imitation labels.

At sampled decision points the bridge forks the world (op fork; every restore replays the live path exactly). For one
village it rolls out each alternative for ``horizon`` turns: the policy itself, the native heuristic, each legal region
move (then keep the intent) and each pure economy mix (held for the horizon). Every other village follows the
policy deterministically, so branches differ only by the alternative (paired comparisons with no world noise). The
score is the village lineage's discounted welfare over the horizon plus the critic's value of the lineage at the end
(self-interested credit: the lineage's own settlements, founded daughters included). The label row is the village's
observation at the fork, the best alternative, and every alternative's score; PPO imitates rows whose gain clears a
margin (``ppo --search-labels``).

    python -m anthropocene_rl.autonomous_v7.search_teacher --checkpoint <pt> --seeds 1000-1011 --out <dir>
"""  # noqa: E501 - docstring/prose line kept verbatim

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contracts, data
from .bridge import Model9Bridge, observation_batch
from .evaluate import load, policy_actions
from .model import Critic, VectorCritic, mix_shares, to_tensors
from .ppo import head_scale, personality_sampler, village_batch
from .upgrade import upgrade

# Alternative kinds (label_kind): the policy's own action, the heuristic, a region move, a pure
# economy mix, a fission
# policy held for the horizon, a trade posture held for the horizon.
POLICY, NATIVE, REGION, MIX, FISSION, TRADE = 0, 1, 2, 3, 4, 5
KINDS = ("policy", "native", "region", "mix", "fission", "trade")
# Fission probes: the bridge's fission codes (Neutral 0, Discourage 1, Encourage 2).
FISSION_PROBES = (2, 1)
# Trade probes (slot = probe id): every gate withheld, offered from stock, made to order; the value
# of the group with
# the largest holdings halved and doubled.
TRADE_PROBES = ("withhold", "stock", "make", "value_half", "value_double")
TRADE_SET = [name for name, _, _ in contracts.SETS].index("trade")
TRADE_HELD = 0  # ElderModel9Encoder.TradeFeatureNames()[0] == "held" (bridge schema tradeFeatures)
# label_trade_mask bits per group: which posture components the winning probe changed.
TRADE_VALUE_BIT, TRADE_GATE_BIT = 1, 2
MAX_ALTERNATIVES = (
    2
    + contracts.REGION_ROWS
    - 1
    + len(contracts.ECONOMY_MIX)
    + len(FISSION_PROBES)
    + len(TRADE_PROBES)
)


@dataclass
class SearchConfig:
    turns: int = 25  # episode length of the main path
    horizon: int = 6  # turns each alternative is rolled out (120 years) before the critic bootstrap
    points: int = 4  # decision turns searched per world
    villages: int = 2  # villages searched per decision turn
    first_turn: int = 1
    regions: bool = True
    mixes: bool = True
    native: bool = True
    # Add the critic's end-of-horizon lineage value to the score. Off by default: the first labels
    # (20 October 2026)
    # showed the critic, trained on Elder-acting states, scores native and moved states
    # off-distribution (native
    # branches lost 15% on the bootstrap at equal returns). The bootstrap is always recorded as a
    # diagnostic.
    bootstrap: bool = False
    # Personalities (docs/MODEL9_PERSONALITY.md): each society is drawn from this distribution
    # ("default", "fixed:..."
    # or "dirichlet:a", as in ppo --personality) and daughters inherit their parent's draw. Every
    # alternative returns
    # its lineage's discounted return per objective (contracts.OBJECTIVES), and a decision point's
    # label is scored for
    # its society's personality: 1 + sum_k p_k (R_k(alt) - R_k(policy)) / |R_k(policy)|. The policy,
    # the critic and
    # the imitation loss all read that personality, through the same FiLM-conditioned encoder.
    personality: str = "default"
    # Round 3: when a decision point has a young daughter (parentSociety > 0 and first seen within
    # YOUNG_TURNS turns,
    # or the daughter of a bank start), one of the searched villages is a young daughter. 0 keeps
    # the uniform draw.
    daughter_search: bool = True
    # Round 3 (docs/MODEL9_SEARCH_TEACHER.md, *Probes*): fission probes (Encourage and Discourage
    # submitted every turn of
    # the horizon) and trade-posture probes (TRADE_PROBES, a full posture every turn of the
    # horizon).
    fission: bool = True
    trade: bool = True
    # Pure-policy turns after each branch's horizon, written as critic branch shards
    # (<out>/branch_shards); 0 off.
    tail: int = 4
    # Calibration tails: this share of branches (a keyed draw per branch, calibration_branch) plays
    # calibration_tail
    # turns instead, and its shard rows are marked calibration = 1, so branch_bias can test the
    # critic against 8
    # realised turns rather than against its own bootstrap. Only when tail > 0.
    calibration_share: float = 0.1
    calibration_tail: int = 12
    # Round 3 (review 5.5(3)): coverage. Each world calibrates at least this many branches of every
    # alternative kind it
    # searches (from a keyed decision point per kind on, one of that kind's alternatives by a keyed
    # draw), on top of
    # the keyed share, so rare kinds (fission, trade) are never absent from calibration. 0 off.
    calibration_min_per_kind: int = 1
    calibration_manifest: str = ""


YOUNG_TURNS = 3


def load_critic(checkpoint: Path):
    state = upgrade(torch.load(checkpoint, map_location="cpu", weights_only=False))
    core, privileged, token = (
        int(state["core_width"]),
        int(state.get("privileged_width", 0)),
        bool(state.get("token", False)),
    )
    critic = (VectorCritic if state.get("vector_critic") else Critic)(
        core, privileged_width=privileged, token=token
    )
    critic.load_state_dict(state["critic"])
    critic.eval()
    return critic


def lineage_values(
    critic,
    normalizer,
    villages: list[dict[str, Any]],
    lineage: set[int],
    personalities: dict[int, Any] | None = None,
) -> np.ndarray:
    """The critic's value of the lineage's living settlements per objective (contracts.OBJECTIVES; a scalar critic
    fills welfare only, the rest NaN), each head unscaled as ppo.head_scale says, at each settlement's personality."""  # noqa: E501 - docstring/prose line kept verbatim
    out_values = np.full(len(contracts.OBJECTIVES), np.nan)
    out_values[0] = 0.0
    rows = [
        v
        for v in villages
        if v.get("observation") is not None and not v.get("terminated") and v["society"] in lineage
    ]
    if not rows:
        return np.zeros(len(contracts.OBJECTIVES))  # no living settlement: nothing left to value
    batch = village_batch(rows)
    if personalities:
        default = [contracts.OBJECTIVE_WEIGHTS[k] for k in contracts.OBJECTIVES]
        batch["personality"] = np.asarray(
            [personalities.get(int(v["society"]), default) for v in rows], np.float32
        )
    population = np.asarray([v["population"] for v in rows], dtype=np.float64)
    with torch.no_grad():
        out = critic(to_tensors(batch, normalizer))
    heads = out if isinstance(out, dict) else {"welfare": out}
    for k, name in enumerate(contracts.OBJECTIVES):
        if name in heads:
            u = critic.expectation(heads[name]).numpy().astype(np.float64)
            out_values[k] = float((u * head_scale(name, population)).sum())
    return out_values


def lineage_value(critic, normalizer, villages: list[dict[str, Any]], lineage: set[int]) -> float:
    """The critic's welfare value of the lineage's living settlements: u x residents, summed."""
    return float(lineage_values(critic, normalizer, villages, lineage)[0])


def component_rewards(villages: list[dict[str, Any]], lineage: set[int]) -> np.ndarray:
    """This step's lineage reward per objective (contracts.OBJECTIVES order): welfare P x Q x dt and
    population-years P x dt summed over living settlements, realised trade utility, and QoL-years as the lineage's
    population-weighted QoL x dt. That last one is one number per lineage, so splitting into more settlements is not
    rewarded as more QoL."""  # noqa: E501 - docstring/prose line kept verbatim
    out = np.zeros(len(contracts.OBJECTIVES))
    people = weighted = years = 0.0
    for v in villages:
        if v["society"] not in lineage:
            continue
        alive = not v["terminated"] and v["priorPopulation"] > 0
        dt = float(v.get("years", 0.0))
        out[0] += float(v["welfare"])
        out[2] += float(v.get("tradeUtility", 0.0))
        if alive:
            out[1] += float(v["priorPopulation"]) * dt
            people += float(v["priorPopulation"])
            weighted += float(v["priorPopulation"]) * float(v["qol"])
            years = dt
    out[3] = weighted / people * years if people > 0 else 0.0
    return out


# Search utility versions (docs/MODEL9_PERSONALITY.md, *One utility for PPO and search*). Version 1
# (labels before
# 22 October 2026): 1 + sum_k p_k (G_k - G_k(policy)) / |G_k(policy)|, a relative change against the
# policy's own
# return. Version 2: the utility PPO's advantage uses, U_p = sum_k p_k G_k / s_k_ref
# (contracts.OBJECTIVE_REFERENCE_
# SCALES); an alternative scores U_p(alt) - U_p(policy), so the policy scores 0 and a gain is in
# reference spreads (for
# the welfare personality, a fraction of s_welfare_ref). Label files keep every alternative's return
# vector
# (alt_returns), so version-1 labels are rescored (``rescore``).
UTILITY_VERSION = 2


def personality_utility(returns: np.ndarray, p: np.ndarray) -> np.ndarray:
    """U_p = sum_k p_k G_k / s_k_ref per row of ``returns`` (objectives in contracts.OBJECTIVES order). A component
    missing (NaN) for any alternative gets no weight, so every alternative is scored on the same components."""  # noqa: E501 - docstring/prose line kept verbatim
    scale = np.asarray(
        [contracts.OBJECTIVE_REFERENCE_SCALES[k] for k in contracts.OBJECTIVES], np.float64
    )
    weights = np.where(np.isnan(returns).any(axis=0), 0.0, np.asarray(p, np.float64))
    return (np.nan_to_num(returns) / scale * weights).sum(axis=1)


def personality_score(returns: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Version-2 scores of a decision point's alternatives (rows of ``returns``, the policy first) for personality
    ``p``: U_p(alt) - U_p(policy). The policy scores 0."""  # noqa: E501 - docstring/prose line kept verbatim
    u = personality_utility(returns, p)
    return u - u[0]


def rescore(labels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Label arrays on the version-2 utility. Version-2 files pass through. Version-1 files with per-alternative
    return vectors (alt_returns, personality) are rescored: alt_score, score_policy (0) and score_best (the stored
    label's alternative, the version-1 argmax); label_is_best marks rows whose label is still the version-2 argmax.
    Files without return vectors keep version 1 (utility_version 1) and are judged by the legacy relative margin."""  # noqa: E501 - docstring/prose line kept verbatim
    n = len(labels["label_kind"])
    if (
        "utility_version" in labels
        and (np.asarray(labels["utility_version"]) == UTILITY_VERSION).all()
    ):
        return labels
    out = dict(labels)
    if not all(k in labels for k in ("alt_returns", "personality", "alt_score", "alt_kind")):
        out["utility_version"] = np.full(n, 1)
        out["label_is_best"] = np.ones(n, bool)
        return out
    present = np.asarray(labels["alt_kind"]) >= 0
    old_best = np.argmax(
        np.where(present, np.nan_to_num(labels["alt_score"], nan=-np.inf), -np.inf), axis=1
    )
    scores = np.full(present.shape, np.nan)
    for i in range(n):
        m = present[i]
        scores[i, m] = personality_score(
            np.asarray(labels["alt_returns"][i][m], np.float64), labels["personality"][i]
        )
    new_best = np.argmax(np.where(present, scores, -np.inf), axis=1)
    out.update(
        alt_score=scores,
        score_policy=np.zeros(n),
        score_best=scores[np.arange(n), old_best],
        label_is_best=old_best == new_best,
        utility_version=np.full(n, UTILITY_VERSION),
    )
    return out


def base_posture(action: dict[str, Any] | None, teacher: dict[str, Any] | None) -> dict[str, Any]:
    """The trade posture probes perturb: the policy's own when it learns trade (its action carries one), else the
    heuristic's teacher label (the main-path step whose observation is the fork's), else the native posture (values
    1, every gate make-to-order, the configuration's export share: tradeShare -1)."""  # noqa: E501 - docstring/prose line kept verbatim
    groups = len(contracts.TRADE_GROUPS)
    for source in (action, teacher):
        if source and source.get("tradeValue") is not None and source.get("tradeGate") is not None:
            share = source.get("tradeShare")
            return {
                "value": [float(x) for x in source["tradeValue"]],
                "gate": [int(g) for g in source["tradeGate"]],
                "share": float(share) if share is not None and share >= 0 else -1.0,
            }
    return {"value": [1.0] * groups, "gate": [contracts.TRADE_MAKE] * groups, "share": -1.0}


def trade_probes(village: dict[str, Any], posture: dict[str, Any]) -> list[dict[str, Any]]:
    """TRADE alternatives for one village: each a full posture (every group's value and gate, and the share) and the
    mask of the groups and components it changes from ``posture`` (TRADE_VALUE_BIT, TRADE_GATE_BIT). Only groups
    present in the village's trade-set observation count as changed; a village with no trade row gets no probe, and a
    probe that changes nothing (say make-to-order on the native posture) is skipped."""  # noqa: E501 - docstring/prose line kept verbatim
    obs = village.get("observation") or {}
    sets, present = obs.get("sets"), obs.get("rowPresent")
    if not sets or not present or len(sets) <= TRADE_SET:
        return []
    _, rows, width = contracts.SETS[TRADE_SET]
    active = np.asarray(present[TRADE_SET], bool)[:rows]
    if not active.any():
        return []
    table = np.asarray(sets[TRADE_SET], np.float64).reshape(rows, width)
    value, gate = list(posture["value"]), list(posture["gate"])
    out = []

    def probe(slot: int, new_value: list[float], new_gate: list[int]) -> None:
        mask = np.zeros(rows, np.int8)
        mask |= ((np.asarray(new_value) != np.asarray(value)) & active).astype(
            np.int8
        ) * TRADE_VALUE_BIT
        mask |= ((np.asarray(new_gate) != np.asarray(gate)) & active).astype(
            np.int8
        ) * TRADE_GATE_BIT
        if mask.any():
            out.append(
                {
                    "kind": TRADE,
                    "slot": slot,
                    "mix": None,
                    "trade": {
                        "value": new_value,
                        "gate": new_gate,
                        "share": float(posture["share"]),
                        "mask": mask,
                    },
                }
            )

    for code in range(contracts.TRADE_GATES):  # withhold, stock, make
        probe(code, value, [code] * rows)
    largest = int(np.argmax(np.where(active, table[:, TRADE_HELD], -np.inf)))
    lo, hi = contracts.TRADE_VALUE_RANGE
    for slot, factor in ((3, 0.5), (4, 2.0)):
        tilted = list(value)
        tilted[largest] = float(np.clip(value[largest] * factor, lo, hi))
        probe(slot, tilted, gate)
    return out


TRADE_RECENT_TURNS = 3
CONTACT_SET = [name for name, _, _ in contracts.SETS].index("contacts")
CONTACT_TRADE = (
    3,
    4,
)  # contacts row: recent trade utility and trade flow per capita-year (ElderObservationSchema)
TRADE_ACTIVITY = (6, 7, 8)  # trade row: imported, exported, realised utility (TradeFeatureNames)


def note_trade(history: dict[int, list[float]], step: dict[str, Any]) -> None:
    """Each settlement's realised trade utility on the last TRADE_RECENT_TURNS main-path steps."""
    for v in step["villages"]:
        recent = history.setdefault(int(v["camp"]), [])
        recent.append(float(v.get("tradeUtility", 0.0) or 0.0))
        del recent[:-TRADE_RECENT_TURNS]


def trade_active(village: dict[str, Any], recent: list[float] | None) -> bool:
    """Whether trade probes are worth running for this village: it traded in the last TRADE_RECENT_TURNS main-path
    turns (non-zero tradeUtility), or its observation shows a known trading contact (a present contacts row with
    recent trade utility or flow) or last turn's trade in a present trade-set row (imported, exported or realised
    utility). A present trade row alone is not enough: nearly every eligible village has one."""  # noqa: E501 - docstring/prose line kept verbatim
    if recent and any(abs(float(x)) > 0 for x in recent[-TRADE_RECENT_TURNS:]):
        return True
    obs = village.get("observation") or {}
    sets, present = obs.get("sets"), obs.get("rowPresent")
    if not sets or not present:
        return False
    for index, columns in ((CONTACT_SET, CONTACT_TRADE), (TRADE_SET, TRADE_ACTIVITY)):
        if index >= len(sets):
            continue
        _, rows, width = contracts.SETS[index]
        table = np.asarray(sets[index], np.float64).reshape(rows, width)
        live = np.asarray(present[index], bool)[:rows]
        if (np.abs(table[live][:, list(columns)]) > 0).any():
            return True
    return False


def calibration_branch(
    seed: int, withheld: int, warmup: int, turn: int, camp: int, alternative: int, share: float
) -> bool:
    """A keyed, deterministic draw: whether this branch plays a calibration tail."""
    if share <= 0:
        return False
    return bool(
        np.random.default_rng([seed, withheld, warmup, 17, turn, camp, alternative]).random()
        < share
    )


def force_from_points(seed: int, withheld: int, warmup: int, points: list[int]) -> dict[int, int]:
    """Per alternative kind, the keyed decision point from which a world forces that kind's calibration coverage."""  # noqa: E501 - docstring/prose line kept verbatim
    if not points:
        return {}
    return {
        k: points[int(np.random.default_rng([seed, withheld, warmup, 23, k]).integers(len(points)))]
        for k in range(len(KINDS))
    }


def ensure_kind_coverage(
    alts: list[dict[str, Any]],
    calibrated: list[bool],
    counts: dict[int, int],
    turn: int,
    force_from: dict[int, int],
    minimum: int,
    key: list[int],
) -> list[bool]:
    """Calibration flags with coverage forced: a kind present among ``alts`` whose world count (``counts``) is still
    below ``minimum``, at or after its forcing point, and with no calibrated alternative here, gets one (a keyed draw
    among its alternatives). Pure; the caller updates ``counts``."""  # noqa: E501 - docstring/prose line kept verbatim
    out = list(calibrated)
    if minimum <= 0:
        return out
    for k in sorted({int(a["kind"]) for a in alts}):
        members = [i for i, a in enumerate(alts) if int(a["kind"]) == k]
        if counts.get(k, 0) >= minimum or any(out[i] for i in members):
            continue
        pick = members[int(np.random.default_rng([*key, 29, k]).integers(len(members)))]
        out[pick] = True
    return out


def calibration_coverage(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    """Calibration branches per alternative kind, and per kind and objective the branches whose horizon return on that
    objective is non-zero (a zero return cannot calibrate its head). The full kind x objective x population-band check
    on the written shards is branch_bias.coverage."""  # noqa: E501 - docstring/prose line kept verbatim
    branches = {name: 0 for name in KINDS}
    by_objective = {name: {o: 0 for o in contracts.OBJECTIVES} for name in KINDS}
    for world in worlds:
        for row in world["rows"]:
            for alt, result in zip(row["alternatives"], row["results"], strict=False):
                if not result.get("calibration"):
                    continue
                name = KINDS[int(alt["kind"])]
                branches[name] += 1
                for k, o in enumerate(contracts.OBJECTIVES):
                    if abs(float(np.nan_to_num(result["returns"][k]))) > 1e-9:
                        by_objective[name][o] += 1
    return {
        "branches": branches,
        "nonzero_objective": by_objective,
        "missing_kinds": [name for name, n in branches.items() if n == 0],
    }


def alternatives(
    village: dict[str, Any],
    base: dict[str, Any],
    config: SearchConfig,
    posture: dict[str, Any] | None = None,
    trade_ok: bool = True,
) -> list[dict[str, Any]]:
    """The alternatives searched for one village at a fork, the policy first. ``posture``: the trade posture probes
    perturb (base_posture; None takes the policy's own or the native one). ``trade_ok`` False (trade_active) skips
    the trade probes."""  # noqa: E501 - docstring/prose line kept verbatim
    obs = village["observation"]
    out = [{"kind": POLICY, "slot": -1, "mix": None}]
    if config.native:
        out.append({"kind": NATIVE, "slot": contracts.REGION_NATIVE, "mix": None})
    if config.regions:
        mask = obs.get("regionMovementMask") or []
        for row in range(1, contracts.REGION_ROWS):  # row 0 is the current tile (stay)
            slot = 2 + row
            if slot < len(mask) and mask[slot]:
                out.append({"kind": REGION, "slot": slot, "mix": None})
    if config.mixes:
        legal = obs["economyMixMask"]
        s, f = base["strategic"][3], base["strategic"][4]
        policy_mix = np.asarray([(1 - f) * s, (1 - f) * (1 - s), f])
        for j, ok in enumerate(legal):
            if ok and sum(legal) > 1:
                vertex = np.eye(len(contracts.ECONOMY_MIX))[j]
                if 0.5 * np.abs(vertex - policy_mix).sum() > 0.05:
                    out.append({"kind": MIX, "slot": -1, "mix": vertex})
    if config.fission:
        for code in FISSION_PROBES:
            if (
                int(base.get("fission", -1)) != code
            ):  # the policy's own learned choice is the policy branch
                out.append({"kind": FISSION, "slot": code, "mix": None, "fission": code})
    if config.trade and trade_ok:
        out.extend(
            trade_probes(village, posture if posture is not None else base_posture(base, None))
        )
    return out


def override(
    actions: dict[int, dict[str, Any]],
    camp: int,
    alt: dict[str, Any],
    village: dict[str, Any] | None,
    first: bool,
) -> dict[int, dict[str, Any]]:
    """This turn's actions with the alternative applied to ``camp`` (the policy's elsewhere)."""
    actions = dict(actions)
    if alt["kind"] == POLICY or camp not in actions:
        return actions
    if alt["kind"] == NATIVE:
        actions.pop(camp)
        return actions
    action = dict(actions[camp])
    if alt["kind"] == REGION:
        mask = (village or {}).get("observation", {}).get("regionMovementMask") or []
        if first:
            action["movement"] = contracts.REGION_BASE + alt["slot"] - 2
        elif len(mask) > contracts.REGION_KEEP and mask[contracts.REGION_KEEP]:
            action["movement"] = contracts.KEEP_INTENT
    elif alt["kind"] == MIX:
        s, f = mix_shares(alt["mix"][None, :])
        strategic = list(action["strategic"])
        strategic[3] = float(s[0])
        strategic[4] = float(f[0])
        action["strategic"] = strategic
    elif alt["kind"] == FISSION:  # every turn of the horizon
        action["fission"] = int(alt["fission"])
    elif alt["kind"] == TRADE:
        # Always a full posture (every group's value and gate): ElderTrade.From fills an absent
        # vector with the native
        # values or gates, so a partial posture would silently mix the probe with native trade.
        posture = alt["trade"]
        action["tradeValue"] = [float(x) for x in posture["value"]]
        action["tradeGate"] = [int(g) for g in posture["gate"]]
        action["tradeShare"] = float(posture["share"])
    actions[camp] = action
    return actions


def inherit(personalities: dict[int, Any] | None, villages: list[dict[str, Any]]) -> None:
    """Daughters founded in a branch take their parent's personality exactly, so every branch of a fork scores under
    the same personalities (training adds drift; search keeps branches paired)."""  # noqa: E501 - docstring/prose line kept verbatim
    if personalities is None:
        return
    for v in villages:
        society, parent = int(v["society"]), int(v.get("parentSociety", 0))
        if society not in personalities and parent in personalities:
            personalities[society] = personalities[parent]


def note_founded(founded: dict[int, int] | None, step: dict[str, Any]) -> None:
    """Each settlement's founding turn: the turn of the step it first appears in (its first decision turn)."""  # noqa: E501 - docstring/prose line kept verbatim
    if founded is not None:
        for v in step["villages"]:
            founded.setdefault(int(v["camp"]), int(step["turn"]))


def default_vector() -> list[float]:
    return [contracts.OBJECTIVE_WEIGHTS[k] for k in contracts.OBJECTIVES]


def policy_tail(
    bridge: Model9Bridge,
    step: dict[str, Any],
    actor,
    normalizer,
    movement: bool,
    turns: int,
    lineage: set[int],
    personalities: dict[int, Any] | None,
    founded: dict[int, int],
) -> dict[str, Any]:
    """``turns`` more turns of pure policy after a branch's horizon. Rows: every lineage settlement's tail decision
    (observation at the decision, outcome after it, the personality it acted at, its founding turn), plus each living
    settlement's observation after the last tail turn as its bootstrap. Daughters founded in the tail join the
    lineage (they appear as rows from their first decision)."""  # noqa: E501 - docstring/prose line kept verbatim
    rows: list[dict[str, Any]] = []
    final_turn = None
    for _ in range(turns):
        if not any(not v["terminated"] for v in step["villages"]):
            break
        decision = step
        final_turn = int(decision["turn"])
        acting = [
            v
            for v in decision["villages"]
            if v.get("observation") is not None and not v.get("terminated")
        ]
        step = bridge.step(
            policy_actions(
                actor, normalizer, decision["villages"], movement, personality=personalities
            )
        )
        for v in step["villages"]:
            if v.get("parentSociety", 0) in lineage:
                lineage.add(int(v["society"]))
        inherit(personalities, step["villages"])
        note_founded(founded, step)
        outcome = {int(v["camp"]): v for v in step["villages"]}
        for v in acting:
            society, out = int(v["society"]), outcome.get(int(v["camp"]))
            if out is None or society not in lineage:
                continue
            rows.append(
                {
                    "decision": v,
                    "outcome": out,
                    "turn": int(decision["turn"]),
                    "personality": (personalities or {}).get(society, default_vector()),
                    "founded_turn": founded.get(int(v["camp"]), int(decision["turn"])),
                }
            )
    last = {int(r["outcome"]["camp"]): i for i, r in enumerate(rows)}
    bootstrap = []
    for v in step["villages"]:
        i = last.get(int(v["camp"]))
        # Only a sequence that reaches the last tail turn alive is truncated (and bootstrapped).
        if v.get("observation") is None or v.get("terminated") or int(v["society"]) not in lineage:
            continue
        if i is not None and not rows[i]["outcome"]["terminated"] and rows[i]["turn"] == final_turn:
            rows[i]["bootstrap_index"] = len(bootstrap)
        bootstrap.append(
            {
                "village": v,
                "personality": (personalities or {}).get(int(v["society"]), default_vector()),
            }
        )
    return {"rows": rows, "bootstrap": bootstrap, "endpoint_turn": int(step["turn"])}


def branch(
    bridge: Model9Bridge,
    fork: int,
    actor,
    normalizer,
    critic,
    camp: int,
    society: int,
    alt: dict[str, Any],
    horizon: int,
    movement: bool,
    use_bootstrap: bool = False,
    personalities: dict[int, Any] | None = None,
    tail: int = 0,
    founded: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Roll one alternative out from the fork: its lineage's discounted return per objective (and the critic's
    per-objective bootstrap, added when ``use_bootstrap``), the whole world's discounted welfare (to tell a lineage
    gain that only takes from neighbours), and the heuristic's mix on a native branch. ``tail`` > 0 then plays that
    many turns of pure policy and returns them as ``tail`` (policy_tail; never part of the score)."""  # noqa: E501 - docstring/prose line kept verbatim
    step = bridge.restore(fork)
    personalities = None if personalities is None else dict(personalities)
    founded = dict(founded or {})
    lineage = {society}
    returns = np.zeros(len(contracts.OBJECTIVES))
    world = 0.0
    discount = 1.0
    native_mix = None
    for h in range(horizon):
        by_camp = {int(v["camp"]): v for v in step["villages"]}
        actions = override(
            policy_actions(
                actor, normalizer, step["villages"], movement, personality=personalities
            ),
            camp,
            alt,
            by_camp.get(camp),
            h == 0,
        )
        step = bridge.step(actions)
        for v in step["villages"]:
            if v.get("parentSociety", 0) in lineage:
                lineage.add(int(v["society"]))
        inherit(personalities, step["villages"])
        note_founded(founded, step)
        returns += discount * component_rewards(step["villages"], lineage)
        world += discount * sum(float(v["welfare"]) for v in step["villages"])
        discount *= float(data.discount(np.asarray([step["years"]]))[0])
        if h == 0 and alt["kind"] == NATIVE:
            teacher = next(
                (v.get("teacher") for v in step["villages"] if int(v["camp"]) == camp), None
            )
            if teacher is not None:
                s, f = teacher["strategic"][3], teacher["strategic"][4]
                native_mix = [(1 - f) * s, (1 - f) * (1 - s), f]
        if not any(not v["terminated"] for v in step["villages"]):
            break
    bootstraps = discount * lineage_values(
        critic, normalizer, step["villages"], lineage, personalities
    )
    total = returns + (np.nan_to_num(bootstraps) if use_bootstrap else 0.0)
    result = {
        "returns": total,
        "return": float(returns[0]),
        "bootstrap": float(bootstraps[0]),
        "bootstraps": bootstraps,
        "world": world,
        "native_mix": native_mix,
    }
    if tail > 0:
        result["tail"] = policy_tail(
            bridge, step, actor, normalizer, movement, tail, lineage, personalities, founded
        )
    return result


def choose_villages(
    live: list[dict[str, Any]],
    first_seen: dict[int, int],
    turn: int,
    rng: np.random.Generator,
    count: int,
    daughter_search: bool = True,
    known: frozenset = frozenset(),
) -> list[dict[str, Any]]:
    """The villages searched at a decision point: a uniform draw of ``count``, except that with ``daughter_search``
    and a young daughter present, one is a uniformly drawn young daughter and the rest a uniform draw of the others.
    Young: parentSociety > 0 and first seen at most YOUNG_TURNS turns ago, or a ``known`` (bank) daughter camp.
    Without a young daughter the draw is the uniform one, so earlier searches reproduce."""  # noqa: E501 - docstring/prose line kept verbatim
    young = [
        i
        for i, v in enumerate(live)
        if int(v["camp"]) in known
        or (
            int(v.get("parentSociety", 0)) > 0
            and turn - first_seen.get(int(v["camp"]), turn) <= YOUNG_TURNS
        )
    ]
    if not daughter_search or not young or count <= 0:
        return [live[i] for i in rng.permutation(len(live))[:count]]
    first = young[int(rng.integers(len(young)))]
    rest = [i for i in rng.permutation(len(live)) if i != first][: count - 1]
    return [live[first]] + [live[i] for i in rest]


def search_world(
    checkpoint: str,
    seed: int,
    withheld: int,
    warmup: int,
    config: SearchConfig,
    bank: str | None = None,
    entry: int = -1,
    expected_producer=None,
    world_contract=None,
) -> dict[str, Any]:
    """One world's main path with the policy, searching at sampled decision points. ``bank`` and ``entry``: start from
    that daughter-bank snapshot instead of ``seed``'s opening (its daughter counts as young for selection)."""  # noqa: E501 - docstring/prose line kept verbatim
    torch.set_num_threads(1)
    actor, normalizer = load(Path(checkpoint))
    critic = load_critic(Path(checkpoint))
    movement = bool(getattr(actor, "region_movement", False))
    rng = np.random.default_rng([seed, withheld, 9])
    draw = personality_sampler(config.personality, np.random.default_rng([seed, withheld, 11]))
    personalities: dict[int, Any] | None = None if config.personality == "default" else {}

    def assign(
        villages: list[dict[str, Any]],
    ) -> None:  # first sight: inherit from a known parent, else draw
        if personalities is None:
            return
        inherit(personalities, villages)
        for v in villages:
            personalities.setdefault(int(v["society"]), draw().tolist())

    last = max(config.first_turn + 1, config.turns - config.horizon)
    turns = set(
        rng.choice(
            np.arange(config.first_turn, last),
            size=min(config.points, last - config.first_turn),
            replace=False,
        ).tolist()
    )
    rows: list[dict[str, Any]] = []
    clock = time.perf_counter()
    branches = 0
    known: frozenset = frozenset()  # the banked daughter of a bank start
    first_seen: dict[
        int, int
    ] = {}  # main-path turn index each camp was first seen at (daughter selection)
    founded: dict[int, int] = {}
    traded: dict[int, list[float]] = {}
    trade_points = {"trade_points_probed": 0, "trade_points_skipped": 0}
    force_from = force_from_points(seed, withheld, warmup, sorted(turns))
    calibrated_kinds: dict[int, int] = {}
    snapshot_hash = None
    with Model9Bridge() as bridge:
        from .provenance import simulator_identity

        simulator = simulator_identity(bridge.schema)
        from .worker_contract import validate, world

        snapshot = None
        if entry >= 0:
            from .daughter_bank import DaughterBank

            store = DaughterBank(bank)
            record = store.get(entry)
            snapshot = store.snapshot(record)
            import hashlib

            snapshot_hash = hashlib.sha256(snapshot).hexdigest()
            known = frozenset({int(record["camp"])})
        world_contract = world_contract or world(
            seed,
            withheld,
            snapshot_sha256=snapshot_hash or "",
            turn=record["turn"] if entry >= 0 else None,
            movement=True,
            head=actor.region_head,
        )
        worker = validate(
            bridge,
            expected_producer or getattr(actor, "worker_contract", None),
            world_contract,
            snapshot=snapshot,
            head=actor.region_head,
        )
        step = bridge.reset(seed, withheld=withheld, movement=True, snapshot=snapshot)
        from .evaluate import producer_identity

        producer = producer_identity(bridge)
        note_founded(founded, step)
        for _ in range(warmup):
            step = bridge.step({})
            note_founded(founded, step)
            note_trade(traded, step)
        for turn in range(config.turns):
            for v in step["villages"]:
                first_seen.setdefault(int(v["camp"]), turn)
            assign(step["villages"])
            if turn in turns:
                # The heuristic's trade labels on the main-path step whose observation the fork
                # restores (a restored
                # step carries none).
                teachers = {int(v["camp"]): v.get("teacher") for v in step["villages"]}
                fork, decision = bridge.fork()
                base = policy_actions(
                    actor, normalizer, decision["villages"], movement, personality=personalities
                )
                live = [
                    v
                    for v in decision["villages"]
                    if v.get("observation") is not None
                    and not v["terminated"]
                    and int(v["camp"]) in base
                ]
                for v in choose_villages(
                    live, first_seen, turn, rng, config.villages, config.daughter_search, known
                ):
                    camp, society = int(v["camp"]), int(v["society"])
                    trade_ok = trade_active(v, traded.get(camp))
                    if config.trade:
                        trade_points[
                            "trade_points_probed" if trade_ok else "trade_points_skipped"
                        ] += 1
                    alts = alternatives(
                        v,
                        base[camp],
                        config,
                        base_posture(base[camp], teachers.get(camp)),
                        trade_ok,
                    )
                    calibrated = [
                        config.tail > 0
                        and calibration_branch(
                            seed, withheld, warmup, turn, camp, i, config.calibration_share
                        )
                        for i in range(len(alts))
                    ]
                    if config.tail > 0:
                        calibrated = ensure_kind_coverage(
                            alts,
                            calibrated,
                            calibrated_kinds,
                            turn,
                            force_from,
                            config.calibration_min_per_kind,
                            [seed, withheld, warmup, turn, camp],
                        )
                        for alt_i, c in zip(alts, calibrated, strict=False):
                            calibrated_kinds[int(alt_i["kind"])] = calibrated_kinds.get(
                                int(alt_i["kind"]), 0
                            ) + int(c)
                    results = [
                        branch(
                            bridge,
                            fork,
                            actor,
                            normalizer,
                            critic,
                            camp,
                            society,
                            a,
                            config.horizon,
                            movement,
                            config.bootstrap,
                            personalities,
                            config.calibration_tail if c else config.tail,
                            founded,
                        )
                        for a, c in zip(alts, calibrated, strict=False)
                    ]
                    for r, c in zip(results, calibrated, strict=False):
                        r["calibration"] = bool(c)
                    branches += len(alts)
                    p = (
                        personalities.get(society)
                        if personalities
                        else [contracts.OBJECTIVE_WEIGHTS[k] for k in contracts.OBJECTIVES]
                    )
                    rows.append(
                        {
                            "village": v,
                            "alternatives": alts,
                            "results": results,
                            "turn": turn,
                            "personality": p,
                        }
                    )
                step = bridge.restore(fork)
                bridge.drop(fork)
            step = bridge.step(
                policy_actions(
                    actor, normalizer, step["villages"], movement, personality=personalities
                )
            )
            note_founded(founded, step)
            note_trade(traded, step)
            if not any(not v["terminated"] for v in step["villages"]):
                break
        from .teacher_data import binary_identity

        if binary_identity()["sha256"] != producer["bridge_sha256"]:
            raise ValueError("search worker bridge changed during collection")
    return {
        "seed": seed,
        "withheld": withheld,
        "warmup": warmup,
        "entry": entry,
        "snapshot_hash": snapshot_hash,
        "simulator": simulator,
        "producer": producer,
        "worker_contract": worker,
        "world_contract": world_contract,
        "region_head": actor.region_head,
        "rows": rows,
        "seconds": time.perf_counter() - clock,
        "branches": branches,
        **trade_points,
    }


def validate_producers(worlds, expected):
    if not worlds or any(world.get("producer") != expected for world in worlds):
        raise ValueError(
            "registered search workers used different actual bridge/flags or enabled raw materials"
        )


def validate_registered_head(actor):
    if actor.region_head != "hierarchical":
        from .provenance import ProvenanceError

        raise ProvenanceError(
            "registered search/calibration requires hierarchical region inference"
        )


def label_arrays(
    worlds: list[dict[str, Any]], source: dict[str, Any] | None = None
) -> dict[str, np.ndarray]:
    """Label rows in the shard layout: observation arrays plus label_kind, label_slot, label_mix, the policy's and best
    scores, and every alternative's kind, slot and score (padded to MAX_ALTERNATIVES).

    Round 3 provenance (provenance.py): every row also carries its world (seed, withheld, warmup), the source
    checkpoint hash and code commit (``source``, provenance.source; absent: unknown), the provenance version, and each
    alternative's calibration flag (alt_calibration). The turn, kinds and personality are already per row."""  # noqa: E501 - docstring/prose line kept verbatim
    from .provenance import UNKNOWN

    source = source or {}
    villages, extra = (
        [],
        {
            k: []
            for k in (
                "seed",
                "turn",
                "camp",
                "label_kind",
                "label_slot",
                "label_mix",
                "score_policy",
                "score_best",
                "alt_kind",
                "alt_slot",
                "alt_score",
                "alt_return",
                "alt_bootstrap",
                "alt_world",
                "personality",
                "alt_returns",
                "alt_bootstraps",
                "label_fission",
                "label_trade_value",
                "label_trade_gate",
                "label_trade_mask",
                "utility_version",
                "label_is_best",
                "withheld",
                "warmup",
                "alt_calibration",
                "source_checkpoint",
                "code_commit",
                "provenance_version",
                "source_run",
                "source_record",
                "bank_entry",
                "snapshot_hash",
                "heldout_calibration",
                "calibration_manifest_digest",
                "trajectory_id",
                "decision_id",
            )
        },
    )
    width = len(contracts.OBJECTIVES)
    groups = len(contracts.TRADE_GROUPS)
    for world in worlds:
        for row in world["rows"]:
            p = np.asarray(row["personality"], dtype=np.float64)
            returns = np.stack([r["returns"] for r in row["results"]])
            scores = personality_score(returns, p)
            best = int(np.argmax(scores))
            alt = row["alternatives"][best]
            mix = (
                alt["mix"]
                if alt["kind"] == MIX
                else row["results"][best]["native_mix"]
                if alt["kind"] == NATIVE
                else None
            )
            villages.append(row["village"])
            extra["seed"].append(world["seed"])
            extra["turn"].append(row["turn"])
            extra["camp"].append(int(row["village"]["camp"]))
            extra["label_kind"].append(alt["kind"])
            extra["label_slot"].append(alt["slot"])
            extra["label_mix"].append(
                np.full(3, np.nan) if mix is None else np.asarray(mix, dtype=np.float64)
            )
            # Fission and trade labels: the winning probe's code, or its full posture and the mask
            # of what it changed.
            extra["label_fission"].append(int(alt["fission"]) if alt["kind"] == FISSION else -1)
            trade = alt.get("trade") if alt["kind"] == TRADE else None
            extra["label_trade_value"].append(
                np.full(groups, np.nan) if trade is None else np.asarray(trade["value"], np.float64)
            )
            extra["label_trade_gate"].append(
                np.full(groups, -1) if trade is None else np.asarray(trade["gate"], np.int64)
            )
            extra["label_trade_mask"].append(
                np.zeros(groups, np.int8) if trade is None else np.asarray(trade["mask"], np.int8)
            )
            extra["score_policy"].append(scores[0])
            extra["score_best"].append(scores[best])
            extra["utility_version"].append(UTILITY_VERSION)
            extra["label_is_best"].append(True)
            pad = MAX_ALTERNATIVES - len(scores)
            extra["alt_kind"].append([a["kind"] for a in row["alternatives"]] + [-1] * pad)
            extra["alt_slot"].append([a["slot"] for a in row["alternatives"]] + [-1] * pad)
            extra["alt_score"].append(list(scores) + [np.nan] * pad)
            extra["alt_return"].append([r["return"] for r in row["results"]] + [np.nan] * pad)
            extra["alt_bootstrap"].append([r["bootstrap"] for r in row["results"]] + [np.nan] * pad)
            extra["alt_world"].append([r["world"] for r in row["results"]] + [np.nan] * pad)
            # The value vector of every alternative, so labels can be rescored for any personality
            # later.
            extra["personality"].append(p.astype(np.float32))
            extra["alt_returns"].append(np.vstack([returns, np.full((pad, width), np.nan)]))
            extra["alt_bootstraps"].append(
                np.vstack(
                    [
                        np.stack([r["bootstraps"] for r in row["results"]]),
                        np.full((pad, width), np.nan),
                    ]
                )
            )
            extra["withheld"].append(int(world.get("withheld", 0)))
            extra["warmup"].append(int(world.get("warmup", 0)))
            extra["alt_calibration"].append(
                [int(bool(r.get("calibration", False))) for r in row["results"]] + [-1] * pad
            )
            extra["source_checkpoint"].append(str(source.get("source_checkpoint", UNKNOWN)))
            extra["code_commit"].append(str(source.get("code_commit", UNKNOWN)))
            extra["provenance_version"].append(int(source.get("provenance_version", 0)))
            extra["source_run"].append(str(source.get("source_run", UNKNOWN)))
            extra["source_record"].append(json.dumps(source, sort_keys=True))
            extra["bank_entry"].append(int(world.get("entry", -1)))
            extra["snapshot_hash"].append(str(world.get("snapshot_hash") or ""))
            from .calibration import held_out

            held = source.get("search_config", {}).get("calibration_manifest") or {}
            extra["heldout_calibration"].append(
                held_out(held, int(world["seed"]), world.get("snapshot_hash"))
            )
            extra["calibration_manifest_digest"].append(held.get("manifest_digest", ""))
            from .provenance import label_identity

            trajectory, identity = label_identity(
                {
                    "source_run": source.get("source_run", UNKNOWN),
                    "seed": int(world["seed"]),
                    "withheld": int(world.get("withheld", 0)),
                    "warmup": int(world.get("warmup", 0)),
                    "bank_entry": int(world.get("entry", -1)),
                    "snapshot_hash": str(world.get("snapshot_hash") or ""),
                    "turn": int(row["turn"]),
                    "camp": int(row["village"]["camp"]),
                }
            )
            extra["trajectory_id"].append(trajectory)
            extra["decision_id"].append(identity)
    if not villages:
        return {}
    arrays = observation_batch(villages)
    arrays.update({k: np.asarray(v) for k, v in extra.items()})
    return arrays


def write_branch_shards(
    worlds: list[dict[str, Any]], root: Path, source: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Every branch's policy tail as one critic shard (branch_shards.write), keyed by (seed, decision point,
    alternative); returns shard and row counts per alternative kind. ``source`` (provenance.source) is recorded in
    every shard's meta with the searched village's personality."""  # noqa: E501 - docstring/prose line kept verbatim
    from . import branch_shards

    counts = {name: 0 for name in KINDS}
    shards = calibration = 0
    for world in worlds:
        seed, withheld, warmup = (
            int(world["seed"]),
            int(world["withheld"]),
            int(world.get("warmup", 0)),
        )
        for point, row in enumerate(world["rows"]):
            for a, (alt, result) in enumerate(
                zip(row["alternatives"], row["results"], strict=False)
            ):
                tail = result.get("tail")
                if not tail or not tail["rows"]:
                    continue
                meta = {
                    "seed": seed,
                    "withheld": withheld,
                    "warmup": warmup,
                    "point": point,
                    "alternative": a,
                    "kind": int(alt["kind"]),
                    "slot": int(alt.get("slot", -1)),
                    "fork_turn": int(row["turn"]),
                    "camp": int(row["village"]["camp"]),
                    "world_id": branch_shards.world_id(seed, withheld, warmup, point, a),
                    "calibration": int(bool(result.get("calibration", False))),
                    "personality": [float(x) for x in row["personality"]],
                    **(source or {}),
                    "source_record": source or {},
                    "bank_entry": int(world.get("entry", -1)),
                    "snapshot_hash": world.get("snapshot_hash"),
                }
                from .calibration import held_out

                held = (source or {}).get("search_config", {}).get("calibration_manifest") or {}
                meta["heldout_calibration"] = held_out(held, seed, world.get("snapshot_hash"))
                meta["calibration_manifest_digest"] = held.get("manifest_digest", "")
                identity = branch_shards.trajectory_id(meta)
                meta["trajectory_id"] = identity
                counts[KINDS[alt["kind"]]] += branch_shards.write(root / identity, tail, meta)
                shards += 1
                calibration += meta["calibration"]
    return {"branch_shards": shards, "branch_rows": counts, "calibration_shards": calibration}


def label_gain(
    labels: dict[str, np.ndarray], margin: float, legacy_margin: float = 0.005
) -> tuple[np.ndarray, np.ndarray]:
    """Each label row's gain over the policy and whether it clears its margin. Version-2 rows: score_best -
    score_policy in reference spreads against ``margin``, and only while the label is the version-2 argmax. Version-1
    rows that could not be rescored: the relative score gain against ``legacy_margin``."""  # noqa: E501 - docstring/prose line kept verbatim
    n = len(labels["score_best"])
    version = np.asarray(labels.get("utility_version", np.ones(n)))
    absolute = labels["score_best"] - labels["score_policy"]
    relative = absolute / np.maximum(1e-9, np.abs(labels["score_policy"]))
    best = np.asarray(labels.get("label_is_best", np.ones(n, bool)), bool)
    v2 = version == UTILITY_VERSION
    return np.where(v2, absolute, relative), np.where(
        v2, (absolute > margin) & best, relative > legacy_margin
    )


def summarize(labels: dict[str, np.ndarray], margin: float) -> dict[str, Any]:
    """How often and by how much search beats the policy, by winning kind. Version-2 gains are in reference spreads
    (U_p units), and ``margin`` is in the same units."""  # noqa: E501 - docstring/prose line kept verbatim
    if not labels:
        return {"rows": 0}
    labels = rescore(labels)
    gain, kept = label_gain(labels, margin)
    out = {
        "rows": len(gain),
        "beats_policy_share": float(np.mean(kept)),
        "margin": margin,
        "utility_version": UTILITY_VERSION,
        "gain_median_when_kept": float(np.median(gain[kept])) if kept.any() else 0.0,
        "gain_p90": float(np.quantile(gain, 0.9)),
    }
    for k, name in enumerate(KINDS):
        sel = kept & (labels["label_kind"] == k)
        out[f"wins_{name}"] = int(sel.sum())

    # Per alternative kind: mean relative score against the policy (paired).
    # Per alternative kind: mean relative score against the policy (paired), and the same for the
    # horizon return alone
    # (no critic bootstrap), so a gain the critic invents shows as a gap between the two.
    def relative(key: str) -> np.ndarray:
        base = labels[key][:, :1]
        return (labels[key] - base) / np.maximum(1e-9, np.abs(base))

    # alt_score is already a difference from the policy in U_p units (version 2); alt_return stays
    # relative.
    rel, ret = labels["alt_score"] - labels["alt_score"][:, :1], relative("alt_return")
    if "alt_world" in labels:
        # The winners' effect on the whole world: a lineage gain with a world loss takes from
        # neighbours.
        best = np.nanargmax(np.where(labels["alt_kind"] >= 0, labels["alt_score"], -np.inf), axis=1)
        world = labels["alt_world"]
        world_gain = (world[np.arange(len(best)), best] - world[:, 0]) / np.maximum(
            1e-9, np.abs(world[:, 0])
        )
        out["winner_world_gain_median"] = float(np.median(world_gain[kept])) if kept.any() else 0.0
        out["winner_world_loss_share"] = float(np.mean(world_gain[kept] < 0)) if kept.any() else 0.0
    for k, name in enumerate(KINDS[1:], start=1):
        sel = labels["alt_kind"] == k
        out[f"mean_gain_{name}"] = float(np.nanmean(rel[sel])) if sel.any() else None
        out[f"mean_return_gain_{name}"] = float(np.nanmean(ret[sel])) if sel.any() else None
        out[f"count_{name}"] = int(sel.sum())
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--seeds", default="1000-1011", help="TRAIN seed range (never selection or gate seeds)"
    )
    parser.add_argument(
        "--valley-seeds", type=Path, default=None, help="valley fixture bank (lines 'seed,warmup')"
    )
    parser.add_argument("--withheld", type=int, default=0)
    parser.add_argument(
        "--daughter-bank", type=Path, default=None, help="daughter-start bank (ppo --daughter-bank)"
    )
    parser.add_argument(
        "--worlds-manifest",
        type=Path,
        default=None,
        help="immutable reviewed exact search/calibration cohort",
    )
    parser.add_argument(
        "--bank-worlds",
        type=int,
        default=0,
        help="also search this many training entries of --daughter-bank (newest first), from "
        "their snapshots",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.004,
        help="U_p gain a label must clear, in reference spreads (a fraction of s_welfare_ref "
        "for the "
        "welfare personality; summary only)",
    )
    parser.add_argument("--out", type=Path, required=True)
    for name, value in asdict(SearchConfig()).items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=type(value) if not isinstance(value, bool) else int,
            default=value,
        )
    from .checkpoint import parse

    args = parse(parser)
    config = SearchConfig(
        **{
            k: (bool(getattr(args, k)) if isinstance(v, bool) else getattr(args, k))
            for k, v in asdict(SearchConfig()).items()
        }
    )
    a, b = (int(x) for x in args.seeds.split("-"))
    worlds = [(s, args.withheld, 0, -1) for s in range(a, b + 1)]
    if args.valley_seeds:
        worlds += [
            (int(s), 0, int(w), -1)
            for s, w in (
                line.split(",") for line in args.valley_seeds.read_text().split() if line.strip()
            )
        ]
    if args.daughter_bank and args.bank_worlds > 0:
        from .daughter_bank import DaughterBank

        entries = DaughterBank(args.daughter_bank).training_entries()[::-1][: args.bank_worlds]
        worlds += [(int(e["seed"]), int(e["mask"]), 0, int(e["id"])) for e in entries]
    cohort = None
    if args.worlds_manifest:
        from .calibration import manifest
        from .daughter_bank import DaughterBank
        from .protocol import search_cohort

        if not config.calibration_manifest:
            raise ValueError("frozen cohort requires exclusion manifest")
        cohort, worlds = search_cohort(
            args.worlds_manifest,
            DaughterBank(args.daughter_bank) if args.daughter_bank else None,
            config,
            manifest(Path(config.calibration_manifest)),
        )
        if args.bank_worlds or args.valley_seeds:
            raise ValueError("frozen cohort forbids dynamic additional worlds")
        if args.out.exists():
            raise ValueError(
                "one-shot frozen search output already exists; cannot top up or overwrite"
            )
    elif config.calibration_manifest:
        raise ValueError("qualification search requires --worlds-manifest exact frozen cohort")
    args.out.mkdir(parents=True, exist_ok=True)
    clock = time.time()
    # Freeze identities before any worker reads the checkpoint or forks a world.
    from .provenance import checkpoint_hash, code_commit
    from .provenance import source as provenance_source

    configuration = asdict(config)
    if cohort is not None:
        configuration["worlds_manifest"] = cohort
    configuration["sim_rules"] = os.environ.get("ANTHROPOCENE_SIM_RULES", "")
    configuration["sim_statics"] = os.environ.get("ANTHROPOCENE_SIM_STATICS", "")
    from .teacher_data import binary_identity

    configuration["bridge_sha256"] = binary_identity()["sha256"]
    actor, _ = load(Path(args.checkpoint))
    if cohort is not None:
        validate_registered_head(actor)
    configuration["region_head"] = actor.region_head
    if config.calibration_manifest:
        from .calibration import manifest

        configuration["calibration_manifest"] = manifest(Path(config.calibration_manifest))
    with Model9Bridge() as probe:
        from .provenance import simulator_identity

        simulator = simulator_identity(probe.schema)
        from .worker_contract import from_state, observe, world

        expected_worker = observe(probe)
    source = provenance_source(args.checkpoint, configuration, simulator)
    if cohort is not None:
        from .lineage import checkpoint_evidence
        from .provenance import ProvenanceError

        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if from_state(state, required=True) != expected_worker:
            raise ProvenanceError(
                "registered search probe differs from checkpoint worker contract before collection"
            )
        held = configuration["calibration_manifest"]
        if (
            state.get("artifact_kind") != "promotion"
            or not checkpoint_evidence(state, held)["verified"]
        ):
            raise ProvenanceError(
                "registered search/calibration requires the already-promoted verified candidate"
            )
        if len(source["code_commit"]) != 40 or not simulator.get("clean"):
            raise ProvenanceError(
                "registered search requires clean immutable learner/simulator identities"
            )
    descriptors = []
    for seed, mask, warm, entry_id in worlds:
        frozen = (
            next(
                (
                    row
                    for row in cohort["worlds"]
                    if (row["seed"], row["withheld"], row["warmup"], row.get("bank_entry", -1))
                    == (seed, mask, warm, entry_id)
                ),
                None,
            )
            if cohort
            else None
        )
        bank_entry = None
        if entry_id >= 0:
            from .daughter_bank import DaughterBank

            bank_entry = DaughterBank(args.daughter_bank).get(entry_id)
        descriptors.append(
            world(
                seed,
                mask,
                snapshot_sha256=frozen.get("snapshot_sha256", "")
                if frozen
                else (bank_entry or {}).get("snapshot_sha256", ""),
                turn=bank_entry["turn"] if bank_entry else None,
                movement=True,
                head=actor.region_head,
                cohort_digest=cohort["digest"] if cohort else "",
            )
        )
    with ProcessPoolExecutor(max_workers=min(args.workers, len(worlds))) as pool:
        results = list(
            pool.map(
                search_world,
                [args.checkpoint] * len(worlds),
                [w[0] for w in worlds],
                [w[1] for w in worlds],
                [w[2] for w in worlds],
                [config] * len(worlds),
                [None if args.daughter_bank is None else str(args.daughter_bank)] * len(worlds),
                [w[3] for w in worlds],
                [expected_worker] * len(worlds),
                descriptors,
            )
        )
    simulators = {json.dumps(w["simulator"], sort_keys=True) for w in results}
    if simulators != {json.dumps(simulator, sort_keys=True)}:
        raise ValueError("search workers used different simulator builds")
    expected_producer = {
        "simulator": simulator,
        "bridge_sha256": configuration["bridge_sha256"],
        "flags": {key: configuration[key] for key in ("sim_rules", "sim_statics")},
        "raw_materials": False,
    }
    if cohort is not None:
        validate_producers(results, expected_producer)
    if any(result.get("worker_contract") != expected_worker for result in results):
        raise ValueError("search worker pre-execution contract differs from frozen producer")
    if len(results) != len(descriptors) or any(
        result.get("world_contract") != descriptor
        for result, descriptor in zip(results, descriptors, strict=False)
    ):
        raise ValueError("search worker world/cohort contract differs from frozen descriptor")
    heads = {w["region_head"] for w in results}
    if heads != {configuration["region_head"]}:
        raise ValueError("search workers resolved different region heads")
    if cohort is not None:
        expected = {
            (
                r["seed"],
                r["withheld"],
                r["warmup"],
                r.get("bank_entry", -1),
                r.get("snapshot_sha256") or None,
            )
            for r in cohort["worlds"]
        }
        actual = {
            (r["seed"], r["withheld"], r["warmup"], r["entry"], r.get("snapshot_hash"))
            for r in results
        }
        if actual != expected:
            raise ValueError("search workers did not consume the exact frozen snapshot cohort")
    if (
        checkpoint_hash(args.checkpoint) != source["source_checkpoint"]
        or code_commit() != source["code_commit"]
    ):
        raise ValueError("source code or checkpoint changed during search")
    labels = label_arrays(results, source)
    if labels:
        np.savez_compressed(args.out / "labels.npz", **labels)
    tails = (
        write_branch_shards(results, args.out / "branch_shards", source) if config.tail > 0 else {}
    )
    if cohort is not None and cohort["role"] == "calibration":
        from .calibration_audit import freeze

        frozen_inputs = freeze(
            [args.out / "branch_shards"],
            source["source_checkpoint"],
            configuration["calibration_manifest"],
            cohort,
        )
        (args.out / "calibration-inputs.json").write_text(json.dumps(frozen_inputs, sort_keys=True))
    effective_config = asdict(config)
    if configuration.get("calibration_manifest"):
        effective_config["calibration_manifest"] = configuration["calibration_manifest"]
    summary = {
        "checkpoint": args.checkpoint,
        "provenance": source,
        "config": effective_config,
        "worlds": len(worlds),
        "seconds": time.time() - clock,
        "calibration_coverage": calibration_coverage(results),
        "branches": sum(r["branches"] for r in results),
        **tails,
        # Trade-probe gate (trade_active): decision points whose village was probed or skipped.
        **{
            key: sum(r.get(key, 0) for r in results)
            for key in ("trade_points_probed", "trade_points_skipped")
        },
        "seconds_per_branch": sum(r["seconds"] for r in results)
        / max(1, sum(r["branches"] for r in results)),
        **summarize(labels, args.margin),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
