"""Critic branch shards (docs/MODEL9_SEARCH_TEACHER.md, *Policy tails*): the pure-policy tail turns that follow each
search branch's held horizon, written in the teacher-shard layout ``data.load`` reads, so the critic sees states the
search reaches (native-acting, moved, re-mixed, probed) under the policy that will act from them.

One directory per branch (seed, decision point, alternative). Each row is one lineage settlement's tail decision: its
observation arrays, privileged features and personality, the outcome keys ``data.reward_components`` needs, the
branch's unique world id (``world.npy``; sequences never collide across branches), its alternative kind
(``alt_kind.npy``), the turn each settlement was founded on the full path (``founded_turn.npy``, for the lineage
sharing window) and a bootstrap index into the branch's ``bootstrap/`` block: the observation after the last tail turn
of each living settlement. Label arrays are present with every mask false: the actor never trains on these rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from . import contracts, data
from .bridge import observation_batch
from .model import VectorCritic, to_tensors

# World ids of branch rows live far above any seed, so branch and teacher rows never share a sequence key.
BRANCH_WORLD_BASE = 1 << 56


def world_id(seed: int, withheld: int, warmup: int, point: int, alternative: int) -> int:
    """A unique int64 world id per (seed, withheld mask, warm-up, decision point, alternative)."""
    if not (0 <= point < 1024 and 0 <= alternative < 64 and 0 <= withheld < 16 and 0 <= seed < (1 << 29)):
        raise ValueError("branch world id out of range")
    return BRANCH_WORLD_BASE + ((((seed * 64 + warmup % 64) * 16 + withheld) * 1024 + point) * 64 + alternative)


def _observations(villages: list[dict[str, Any]], personalities: list[Any]) -> dict[str, np.ndarray]:
    batch = observation_batch(villages)
    if villages and villages[0].get("privileged") is not None:
        batch["privileged"] = np.asarray([v["privileged"] for v in villages], dtype=np.float32)
    batch["personality"] = np.asarray(personalities, dtype=np.float32)
    return batch


def _dummy_labels(n: int) -> dict[str, np.ndarray]:
    services, strategic = len(contracts.SERVICES), len(contracts.STRATEGIC)
    return {"strategic": np.zeros((n, strategic), np.float32), "strategic_mask": np.zeros((n, strategic), bool),
            "service_levels": np.zeros((n, services), np.float32), "service_mask": np.zeros((n, services), bool),
            "clothing_route": np.full(n, -1, np.int32), "clothing_route_label_mask": np.zeros(n, bool),
            "shelter_route": np.full(n, -1, np.int32), "shelter_route_label_mask": np.zeros(n, bool),
            "movement": np.zeros(n, np.int32), "movement_label_mask": np.zeros(n, bool),
            "fission_policy": np.full(n, -1, np.int32), "repair": np.zeros(n, np.float32)}


def write(directory: Path, tail: dict[str, Any], meta: dict[str, Any]) -> int:
    """Write one branch's tail (``search_teacher.branch`` returns it as ``tail``) as a shard; returns its rows."""
    rows = tail["rows"]
    if not rows:
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    arrays = _observations([r["decision"] for r in rows], [r["personality"] for r in rows])
    arrays.update(_dummy_labels(n))
    out = [r["outcome"] for r in rows]
    arrays.update({
        "camp": np.asarray([v["camp"] for v in out], np.int32), "turn": np.asarray([r["turn"] for r in rows], np.int32),
        "society": np.asarray([v["society"] for v in out], np.int32),
        "parent_society": np.asarray([v.get("parentSociety", 0) for v in out], np.int32),
        "prior_population": np.asarray([v["priorPopulation"] for v in out], np.int32),
        "population": np.asarray([v["population"] for v in out], np.int32),
        "qol": np.asarray([v["qol"] for v in out], np.float32), "nutrition": np.asarray([v["nutrition"] for v in out], np.float32),
        "years": np.asarray([v["years"] for v in out], np.float32), "welfare": np.asarray([v["welfare"] for v in out], np.float32),
        "terminated": np.asarray([bool(v["terminated"]) for v in out], bool),
        "trade_utility": np.asarray([float(v.get("tradeUtility", 0.0)) for v in out], np.float32),
        "world": np.full(n, int(meta["world_id"]), np.int64),
        "alt_kind": np.full(n, int(meta["kind"]), np.int32),
        "founded_turn": np.asarray([r["founded_turn"] for r in rows], np.int32),
        "bootstrap_index": np.asarray([r.get("bootstrap_index", -1) for r in rows], np.int32),
        # 1 on a calibration tail (SearchConfig.calibration_tail turns), checked by branch_bias against realised turns.
        "calibration": np.full(n, int(meta.get("calibration", 0)), np.int8),
    })
    for key, value in arrays.items():
        np.save(directory / f"{key}.npy", value)
    boot = tail.get("bootstrap") or []
    if boot:
        (directory / "bootstrap").mkdir(exist_ok=True)
        for key, value in _observations([b["village"] for b in boot], [b["personality"] for b in boot]).items():
            np.save(directory / "bootstrap" / f"{key}.npy", value)
    (directory / "meta.json").write_text(json.dumps({**meta, "rows": n, "branch": True,
                                                     "contract": contracts.NATIVE_CONTRACT}))
    return n


def directories(roots: Iterable[Path]) -> list[Path]:
    """Branch directories under each root (``<search out>/branch_shards``), or a root that is itself one."""
    out: list[Path] = []
    for root in roots:
        root = Path(root)
        if (root / "meta.json").exists():
            out.append(root)
        elif root.exists():
            out.extend(sorted(p for p in root.iterdir() if (p / "meta.json").exists()))
    return out


def load(roots: Iterable[Path]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """All branch rows under ``roots`` and their bootstrap observations; ``bootstrap_index`` is rebased onto the
    concatenated bootstrap block (-1: none)."""
    worlds, boots, offset = [], [], 0
    for d in directories(roots):
        w = data.load_world(d)
        if len(w["camp"]) == 0:
            continue
        w.setdefault("calibration", np.zeros(len(w["camp"]), np.int8))  # shards written before calibration tails
        index = w["bootstrap_index"].astype(np.int64)
        if (d / "bootstrap" / "core.npy").exists():
            b = {p.stem: np.load(p) for p in sorted((d / "bootstrap").glob("*.npy"))}
            boots.append(b)
            w["bootstrap_index"] = np.where(index >= 0, index + offset, -1)
            offset += len(b["core"])
        else:
            w["bootstrap_index"] = np.full(len(index), -1, np.int64)
        worlds.append(w)
    if not worlds:
        raise ValueError("no branch shards found")
    keys = [k for k in worlds[0] if all(k in w for w in worlds)]
    rows = {k: np.concatenate([w[k] for w in worlds]) for k in keys}
    boot = None
    if boots:
        bkeys = [k for k in boots[0] if all(k in b for b in boots)]
        boot = {k: np.concatenate([b[k] for b in boots]) for k in bkeys}
    return rows, boot


def _head_values(critic, batch: dict[str, np.ndarray], normalizer, batch_size: int = 4096) -> dict[str, np.ndarray]:
    """The critic's per-resident (or raw, for qol_years) expectation per head; a scalar critic as ``welfare``."""
    parts: dict[str, list[np.ndarray]] = {}
    n = len(batch["core"])
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sub = {k: v[start:start + batch_size] for k, v in batch.items()}
            out = critic(to_tensors(sub, normalizer))
            heads = out if isinstance(out, dict) else {"welfare": out}
            for name, logits in heads.items():
                parts.setdefault(name, []).append(critic.expectation(logits).numpy().astype(np.float64))
    return {name: np.concatenate(p) for name, p in parts.items()}


def head_scale(name: str, population: np.ndarray) -> np.ndarray:
    from .ppo import head_scale as scale
    return scale(name, population)


def values(critic, rows: dict[str, np.ndarray], boot: dict[str, np.ndarray] | None,
           normalizer) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Raw-unit values per head of every row, and of each row's bootstrap observation (NaN where it has none)."""
    u = _head_values(critic, rows, normalizer)
    value = {name: x * head_scale(name, rows["prior_population"]) for name, x in u.items()}
    bootstrap = {name: np.full(len(rows["camp"]), np.nan) for name in u}
    index = rows["bootstrap_index"].astype(np.int64)
    has = index >= 0
    if boot is not None and has.any():
        ub = _head_values(critic, boot, normalizer)
        for name, x in ub.items():
            # The bootstrap observation is taken at the row's end population.
            bootstrap[name][has] = x[index[has]] * head_scale(name, rows["population"][has])
    return value, bootstrap


def targets(critic, rows: dict[str, np.ndarray], boot: dict[str, np.ndarray] | None, normalizer,
            lam: float = 0.95) -> dict[str, np.ndarray] | np.ndarray:
    """Critic targets for branch rows: lambda-returns over each branch sequence with the current critic's values and
    its bootstrap after the last tail turn, in the critic's own units (per resident, qol_years raw). A vector critic
    gets one target per head (a dict); a scalar critic its welfare target (an array)."""
    value, bootstrap = values(critic, rows, boot, normalizer)
    components = data.reward_components(rows)
    out = {name: data.lambda_returns(rows, components[name], value[name], lam, bootstrap[name])
           / head_scale(name, rows["prior_population"]) for name in value}
    return out if isinstance(critic, VectorCritic) else out["welfare"]
