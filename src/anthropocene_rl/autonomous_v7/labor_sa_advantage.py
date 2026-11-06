"""Per-(state, action) advantage-error check of the labor critic on search payloads.

PPO needs the critic's per-candidate advantage ``A(s, a) = r(s, a) + g V(s'_a) - V(s)``
to be right; ordering metrics along the teacher trajectory hide that error. This module
compares, at every search root of a ``labor-advantage-candidates-v1`` payload
(``labor_search_payload``), what can be predicted with the critic against the
search-realised candidate returns.

What the payload supports (see ``labor_search_payload`` module docstring):

* realised returns per candidate: stream 0 (deterministic H12 continuation) at every
  declared horizon the branch reached, and re-evaluation streams 1..S at the final horizon
  for the survivors of successive halving (``evaluations``); horizon 1 on stream 0 is the
  first-step reward ``r(s, a)`` exactly (discount 1 on the first step);
* the ROOT observation row only (``tensors``/``privileged``). No successor observation of
  any candidate is stored, so ``V(s'_a)`` and hence the V-based per-candidate advantage is
  not computable from the payload for any candidate.

What this module computes instead:

1. realised within-root advantage structure per head (H1 over all candidates, H3 over the
   survivors), the search noise between re-evaluation streams, field separation;
2. the "reward-only" one-step advantage ``r(s, a) - r(s, h12)`` (what PPO's one-step
   advantage reduces to when the critic's successor values are equal across candidates)
   scored against the realised H3 advantage: the critic-free floor;
3. when the H12 trajectory dataset the roots lie on is given (reserved-world heldout
   collection; roots matched exactly by observation row), the critic's per-(s, a) error on
   the executed H12 candidate. Under H12 the native step and the continuation are
   deterministic, so the true A^{H12}(s, h12) = 0 and the n-step estimate
   ``sum g r + g^n V(s_{t+n}) - V(s)`` *is* the error of PPO's advantage for that (s, a);
4. signal-to-noise of that error against the realised within-root advantage spread, and a
   modelled sign-flip rate of candidate-pair advantage differences (per-candidate errors
   drawn from the measured executed-candidate error distribution; labelled a model).
5. when the payloads carry successor rows (search export v2, ``--successor-rows``): the
   full V-based per-candidate advantage ``A_k(s, a) = sum_{j<k} g r_j + g^k V(s_{a,k}) -
   V(s)`` for k = 1 and the branch end (k = 3), per stream, with V(s) from the payload's
   root row, scored against the realised H3 advantage ``G3(a) - G3(h12)`` per head and
   combined: MAE/bias, within-root calibration slope and correlation, pair sign accuracy
   (with and without the control candidates), top-1, the *measured* population-only
   sign-flip rate, field separation, event/non-event split, per-member spread
   (``successor_check``). Payloads without successor rows give the earlier output exactly.

Combined objective: ``labor_ppo.combine_labor_advantages`` with the balanced personality
and unit cost multipliers, with the frozen reference scales and (by rescaling the inputs)
with the measured r5 scales.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

COMPONENTS = ("qol_years", "population_years", "feasibility", "settlement_guard")
HEADS = COMPONENTS
FROZEN_SCALES = {"qol_years": 43.152, "population_years": 85133.98}
R5_POPULATION_SCALE = 135589.0  # measured r5 population scale (brief; no artifact)
FIELD_ACTIVITIES = ("field.expansion", "field.operation")
EVENT_WINDOW = 5
SCHEMA = "labor-sa-advantage-check-v1"


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------


@dataclass
class Cand:
    index: int
    family: str
    is_h12: bool
    twin: bool  # control-twin: bookkeeping twin of H12 (identical physics)
    control: str | None
    r1: np.ndarray  # [4] stream-0 horizon-1 return == first-step reward
    g3: dict[int, np.ndarray]  # stream -> [4] horizon-3 return (survivors only)
    field_share: float
    field_expansion_share: float
    field_operation_share: float


@dataclass
class SuccStream:
    """One evaluated stream of a candidate with its successor rows (export v2)."""

    stream: int
    rewards: np.ndarray  # [d, 4] raw per-step rewards
    years: np.ndarray  # [d]
    terminated: bool
    status: dict[int, str]  # decision -> live / terminated / inactive
    row: dict[int, int]  # decision -> global successor row (live, exported decisions)


@dataclass
class RootRec:
    root_id: str
    world: int
    turn: int
    camp: int
    control_root: bool
    tensor_row: int
    shard: str
    cands: list[Cand]
    final_horizon: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def h12(self) -> Cand:
        return next(c for c in self.cands if c.is_h12)


def _field_shares(
    cand: Mapping[str, Any], activity_ids: Sequence[str]
) -> tuple[float, float, float]:
    compiled = cand.get("compiled") or {}
    wy = np.asarray(compiled.get("worker_years") or [], dtype=np.float64)
    if wy.size == 0 or wy.sum() <= 0:
        return 0.0, 0.0, 0.0
    total = float(wy.sum())
    exp = float(wy[:, activity_ids.index("field.expansion")].sum()) / total
    op = float(wy[:, activity_ids.index("field.operation")].sum()) / total
    return exp + op, exp, op


def parse_successors(
    raw_root: Mapping[str, Any], row_offset: int = 0
) -> dict[int, list[SuccStream]]:
    """Per first-action candidate index: its successor streams (empty for v1 payloads).
    ``row_offset`` maps shard rows to rows of a concatenated multi-shard dataset."""
    out: dict[int, list[SuccStream]] = {}
    for c in raw_root["candidates"]:
        if c.get("plan_role") is not None or "successor_streams" not in c:
            continue
        streams = []
        for e in c["successor_streams"]:
            status, row = {}, {}
            for b in e["boundaries"]:
                status[int(b["decision"])] = str(b["status"])
                if b.get("row") is not None:
                    row[int(b["decision"])] = int(b["row"]) + int(row_offset)
            streams.append(
                SuccStream(
                    int(e["stream"]),
                    np.asarray(e["step_rewards"], dtype=np.float64).reshape(-1, len(COMPONENTS)),
                    np.asarray(e["step_years"], dtype=np.float64),
                    bool(e["terminated"]),
                    status,
                    row,
                )
            )
        out[int(c["index"])] = streams
    return out


def parse_payload(shard: Mapping[str, Any], label: str = "", horizon: int = 3) -> list[RootRec]:
    """First-action candidates of each root (plan rows are skipped, as
    ``labor_advantage.parse_candidate_shard`` keeps them apart). ``Cand.g3`` holds the
    returns at ``horizon`` (the search's final horizon; 3 for the r5 searches)."""
    if list(shard["component_order"]) != list(COMPONENTS):
        raise ValueError("unexpected component order")
    if float(shard["discount_years"]) != 400.0:
        raise ValueError("payload discount must be 400 years")
    activity_ids = list(shard["activity_ids"])
    out = []
    for raw in shard["roots"]:
        cands = []
        for c in raw["candidates"]:
            if c.get("plan_role") is not None:
                continue
            r1 = None
            g3: dict[int, np.ndarray] = {}
            for ev in c["evaluations"]:
                if int(ev["reached_horizon"]) < int(ev["horizon"]) and not ev["terminated"]:
                    continue  # truncated branch (labor_advantage parse rule)
                if int(ev["horizon"]) == 1 and int(ev["stream"]) == 0:
                    r1 = np.asarray(ev["returns"], dtype=np.float64)
                if int(ev["horizon"]) == int(horizon):
                    g3[int(ev["stream"])] = np.asarray(ev["returns"], dtype=np.float64)
            if r1 is None:
                continue
            fs, fe, fo = _field_shares(c, activity_ids)
            cands.append(
                Cand(
                    int(c["index"]),
                    str(c["family"]),
                    bool(c["is_h12"]),
                    str(c["family"]) == "control-twin" or c.get("bookkeeping_twin") is not None,
                    c.get("control"),
                    r1,
                    g3,
                    fs,
                    fe,
                    fo,
                )
            )
        if sum(c.is_h12 for c in cands) != 1:
            raise ValueError(f"root {raw['root_id']} needs exactly one H12 candidate")
        out.append(
            RootRec(
                str(raw["root_id"]),
                int(raw["world"]),
                int(raw["turn"]),
                int(raw["camp"]),
                bool(raw.get("control_root", False)),
                int(raw["tensor_row"]),
                label,
                cands,
                int(raw.get("final_horizon") or 0),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Units and combination
# ---------------------------------------------------------------------------


def combine(values: Mapping[str, np.ndarray], scales: Mapping[str, float]) -> np.ndarray:
    """``labor_ppo.combine_labor_advantages`` (balanced 0.5/0.5, unit cost multipliers)
    applied in the given objective scales. The function itself divides by the frozen
    reference scales; other scales are applied by rescaling its inputs."""
    import torch

    from .labor_model import LaborPersonality
    from .labor_ppo import combine_labor_advantages

    task = {
        name: torch.as_tensor(
            np.asarray(values[name], dtype=np.float64) * FROZEN_SCALES[name] / float(scales[name])
        )
        for name in ("qol_years", "population_years")
    }
    costs = {
        name: torch.as_tensor(np.asarray(values[name], dtype=np.float64))
        for name in ("feasibility", "settlement_guard")
    }
    personality = LaborPersonality(qol_years=0.5, population_years=0.5)
    out = combine_labor_advantages(
        task,
        costs,
        personality=personality,
        cost_multipliers={"feasibility": 1.0, "settlement_guard": 1.0},
    )
    return out.double().numpy()


def vec_to_heads(arr: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1, len(COMPONENTS))
    return {name: arr[:, k] for k, name in enumerate(COMPONENTS)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def within_root_metrics(
    pred: np.ndarray,
    real: np.ndarray,
    group: np.ndarray,
    worlds: np.ndarray,
    *,
    margin: float,
) -> dict[str, Any]:
    """Pred vs realised advantages (same units). World-balanced aggregates.

    ``group``: root id per row. Calibration slope/correlation use per-root demeaned
    values; Spearman is the mean within-root rank correlation over roots with >= 3
    candidates; pairwise sign accuracy counts same-root pairs with |realised diff| >
    ``margin``; top-1: the predicted best candidate equals the realised best (roots with a
    unique realised best by more than ``margin``)."""
    pred = np.asarray(pred, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    per_world: dict[int, dict[str, Any]] = {}
    for w in sorted(set(int(x) for x in worlds)):
        m = worlds == w
        p, r, g = pred[m], real[m], group[m]
        err = p - r
        dp = p.copy()
        dr = r.copy()
        spear, pairs_ok, pairs_n, top_ok, top_n = [], 0, 0, 0, 0
        for gid in np.unique(g):
            k = g == gid
            dp[k] -= p[k].mean()
            dr[k] -= r[k].mean()
            pk, rk = p[k], r[k]
            if k.sum() >= 3:
                c = _corr(_rank(pk), _rank(rk))
                if c is not None:
                    spear.append(c)
            n = len(pk)
            for i in range(n):
                for j in range(i + 1, n):
                    d = rk[i] - rk[j]
                    if abs(d) > margin:
                        pairs_n += 1
                        pairs_ok += int(np.sign(pk[i] - pk[j]) == np.sign(d))
            if n >= 2:
                srt = np.sort(rk)
                if srt[-1] - srt[-2] > margin:
                    top_n += 1
                    top_ok += int(np.argmax(pk) == np.argmax(rk))
        var = float(np.dot(dp, dp))
        per_world[w] = {
            "rows": int(m.sum()),
            "roots": len(np.unique(g)),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "slope": None if var == 0 else float(np.dot(dp, dr) / var),
            "corr": _corr(dp, dr),
            "spearman": None if not spear else float(np.mean(spear)),
            "pair_sign_accuracy": None if pairs_n == 0 else pairs_ok / pairs_n,
            "pairs": pairs_n,
            "top1": None if top_n == 0 else top_ok / top_n,
            "top1_roots": top_n,
        }
    return {"world_balanced": world_mean(per_world), "per_world": per_world}


def world_mean(per_world: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    keys = set().union(*[set(v) for v in per_world.values()]) if per_world else set()
    out: dict[str, Any] = {}
    for key in sorted(keys):
        vals = [v.get(key) for v in per_world.values()]
        vals = [x for x in vals if isinstance(x, (int, float)) and x is not None]
        if key in ("rows", "roots", "pairs", "top1_roots"):
            out[key] = int(sum(vals))
        else:
            out[key] = None if not vals else float(np.mean(vals))
    out["worlds"] = len(per_world)
    return out


def flip_rate_empirical(diffs: np.ndarray, errors: np.ndarray) -> float | None:
    """Mean over pairs (true diff d) of P[sign(d + e_i - e_j) != sign(d)], e_i, e_j drawn
    independently from the empirical error sample (all ordered i != j)."""
    diffs = np.asarray(diffs, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    if diffs.size == 0 or e.size < 2:
        return None
    de = (e[:, None] - e[None, :])[~np.eye(e.size, dtype=bool)]
    de = np.sort(de)
    rates = []
    for d in diffs:
        if d == 0:
            continue
        # flip when d + de has the opposite sign of d: de < -d (d > 0) or de > -d (d < 0)
        if d > 0:
            rates.append(np.searchsorted(de, -d, side="left") / de.size)
        else:
            rates.append(1.0 - np.searchsorted(de, -d, side="right") / de.size)
    return None if not rates else float(np.mean(rates))


def flip_rate_gaussian(diffs: np.ndarray, sigma: float) -> float | None:
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[diffs != 0]
    if diffs.size == 0 or sigma <= 0:
        return None
    z = np.abs(diffs) / (math.sqrt(2.0) * sigma)
    return float(np.mean([0.5 * math.erfc(x / math.sqrt(2.0)) for x in z]))


def event_flags(core_future: np.ndarray, *, window: int = EVENT_WINDOW) -> dict[str, bool] | None:
    """Event definitions of g0/event-split-diagnostic.md section 3 applied to the focal
    camp's rows t..t+window (row 0 = the root): fission when fission-age core[12] drops by
    more than 0.5 from one row to the next; emigration / immigration when realised
    migration core[13] < -0.01 / > +0.01 on a successor row. ``None`` when fewer than
    ``window`` later rows exist."""
    core_future = np.asarray(core_future, dtype=np.float64)
    if core_future.shape[0] < window + 1:
        return None
    fa = core_future[: window + 1, 12]
    mig = core_future[1 : window + 1, 13]
    fission = bool(np.any(fa[:-1] - fa[1:] > 0.5))
    emig = bool(np.any(mig < -0.01))
    immig = bool(np.any(mig > 0.01))
    return {
        "fission": fission,
        "emigration": emig,
        "immigration": immig,
        "event": fission or emig or immig,
    }


# ---------------------------------------------------------------------------
# Realised advantages
# ---------------------------------------------------------------------------


def realised_rows(
    roots: Sequence[RootRec], *, horizon: int, baseline: str, streams: str = "reeval"
) -> dict[str, np.ndarray]:
    """Per-candidate realised advantages (raw units, [4]) at ``horizon``.

    horizon 1: every candidate, stream 0 (the first-step reward). horizon 3: survivors;
    ``streams="reeval"`` averages the shared re-evaluation streams >= 1 (as
    ``labor_advantage.root_value`` prefers), ``"0"`` uses the deterministic stream 0.
    ``baseline``: ``"h12"`` (minus the H12 candidate's return, paired per stream) or
    ``"mean"`` (minus the mean over the root's included candidates). Twins of H12 are
    excluded. Also returns the reward-only predictor ``r1(a) - baseline(r1)``.
    """
    rows: dict[str, list[Any]] = defaultdict(list)
    for root in roots:
        cands = [c for c in root.cands if not c.twin]
        if horizon != 1:
            cands = [c for c in cands if c.g3]
            if not any(c.is_h12 for c in cands):
                continue
        if len(cands) < 2:
            continue
        if horizon == 1:
            value = np.stack([c.r1 for c in cands])
        else:
            if streams == "0":
                value = np.stack([c.g3[0] for c in cands])
            else:
                shared = sorted(set.intersection(*[{s for s in c.g3 if s >= 1} for c in cands]))
                if not shared:
                    continue
                value = np.stack([np.mean([c.g3[s] for s in shared], axis=0) for c in cands])
        r1 = np.stack([c.r1 for c in cands])
        h = next(i for i, c in enumerate(cands) if c.is_h12)
        if baseline == "h12":
            adv = value - value[h]
            adv1 = r1 - r1[h]
        else:
            adv = value - value.mean(axis=0)
            adv1 = r1 - r1.mean(axis=0)
        for i, c in enumerate(cands):
            rows["root"].append(root.root_id)
            rows["world"].append(root.world)
            rows["turn"].append(root.turn)
            rows["is_h12"].append(c.is_h12)
            rows["family"].append(c.family)
            rows["control"].append(c.control is not None)
            rows["field"].append(c.field_share)
            rows["real"].append(adv[i])
            rows["reward_only"].append(adv1[i])
    if not rows:
        return {}
    return {
        "root": np.asarray(rows["root"]),
        "world": np.asarray(rows["world"], dtype=np.int64),
        "turn": np.asarray(rows["turn"], dtype=np.int64),
        "is_h12": np.asarray(rows["is_h12"], dtype=bool),
        "family": np.asarray(rows["family"]),
        "control": np.asarray(rows["control"], dtype=bool),
        "field": np.asarray(rows["field"], dtype=np.float64),
        "real": np.stack(rows["real"]),
        "reward_only": np.stack(rows["reward_only"]),
    }


def stream_noise(roots: Sequence[RootRec]) -> dict[str, Any]:
    """Search noise of a same-root pair difference vs H12 between the two re-evaluation
    streams (common random numbers): sd_stream = |d1 - d2| / sqrt(2) per pair, and the SE of
    their mean |d1 - d2| / 2. Raw units per head."""
    per_head: dict[str, list[float]] = defaultdict(list)
    zero = defaultdict(int)
    n = 0
    for root in roots:
        h = root.h12
        if not {1, 2} <= set(h.g3):
            continue
        for c in root.cands:
            if c.is_h12 or c.twin or not {1, 2} <= set(c.g3):
                continue
            d1 = c.g3[1] - h.g3[1]
            d2 = c.g3[2] - h.g3[2]
            n += 1
            for k, name in enumerate(COMPONENTS):
                per_head[name].append(abs(d1[k] - d2[k]) / math.sqrt(2.0))
                zero[name] += int(d1[k] == d2[k])
    return {
        "pairs": n,
        "per_head": {
            name: {
                "rms_single_stream_sd": float(np.sqrt(np.mean(np.square(v)))) if v else None,
                "identical_streams_fraction": zero[name] / n if n else None,
            }
            for name, v in per_head.items()
        },
    }


def within_root_spread(rows: Mapping[str, np.ndarray], values: np.ndarray) -> float | None:
    """RMS over roots of the within-root SD (ddof=0) of ``values``."""
    sds = []
    for gid in np.unique(rows["root"]):
        k = rows["root"] == gid
        if k.sum() >= 2:
            sds.append(float(np.std(values[k])))
    return None if not sds else float(np.sqrt(np.mean(np.square(sds))))


def field_separation(rows: Mapping[str, np.ndarray], values: np.ndarray) -> dict[str, Any]:
    """Within-root association between candidate field share and ``values``: demeaned
    correlation, and mean(heavy) - mean(light) where heavy is above the root median field
    share (roots whose candidates differ in field share)."""
    df, dv, gaps = [], [], []
    for gid in np.unique(rows["root"]):
        k = rows["root"] == gid
        f, v = rows["field"][k], values[k]
        if k.sum() < 2 or np.ptp(f) < 1e-9:
            continue
        df.append(f - f.mean())
        dv.append(v - v.mean())
        med = np.median(f)
        heavy, light = f > med, f <= med
        if heavy.any() and light.any():
            gaps.append(float(v[heavy].mean() - v[light].mean()))
    if not df:
        return {"roots": 0}
    df_, dv_ = np.concatenate(df), np.concatenate(dv)
    return {
        "roots": len(df),
        "corr": _corr(df_, dv_),
        "heavy_minus_light": None if not gaps else float(np.mean(gaps)),
        "heavy_minus_light_positive_fraction": None
        if not gaps
        else float(np.mean(np.asarray(gaps) > 0)),
    }


# ---------------------------------------------------------------------------
# Trajectory matching and critic evaluation
# ---------------------------------------------------------------------------


TRAJECTORY_KEYS_SKIP = ("target_", "label_")


def trajectory_index(heldout: Path) -> dict[int, dict[str, Any]]:
    """Per world: the shard dir and the row turn/camp arrays of the H12 trajectory dataset."""
    out = {}
    for shard in sorted(Path(heldout).glob("world-*")):
        world = int(shard.name.split("-")[1])
        turns = np.load(shard / "row_turn.npy")
        camps = np.load(shard / "row_camp.npy")
        out[world] = {
            "dir": shard,
            "turns": turns,
            "camps": camps,
        }
    return out


def camp_future_rows(traj: Mapping[str, Any], turn: int, camp: int, count: int) -> list[int]:
    """Rows of ``camp`` at decisions t, t+1, ... (consecutive decision turns of that camp)."""
    turns = traj["turns"]
    rows = np.nonzero(traj["camps"] == camp)[0]
    rows = rows[np.argsort(turns[rows], kind="mergesort")]
    rows = rows[turns[rows] >= turn]
    return [int(r) for r in rows[:count]]


def gather_rows(shard_dir: Path, rows: Sequence[int]) -> dict[str, np.ndarray]:
    meta = json.loads((shard_dir / "meta.json").read_text())
    out = {}
    idx = np.asarray(rows, dtype=np.int64)
    for name in meta["payload"]:
        key = name[: -len(".npy")]
        if key.startswith(TRAJECTORY_KEYS_SKIP):
            continue
        array = np.load(shard_dir / name, mmap_mode="r")
        out[key] = np.ascontiguousarray(array[idx])
    return out


def nstep_targets(
    values: np.ndarray,
    reward: np.ndarray,
    years: np.ndarray,
    terminal: np.ndarray,
    tail: np.ndarray,
    tail_ok: np.ndarray,
    k: int,
    discount_years: float = 400.0,
) -> float | None:
    """sum_{j<k} prod g r_{t+j} + prod g * V(s_{t+k}) over the rows t..t+k of one camp
    (index 0 = root), the rule of ``labor_r5_acceptance.nstep_residuals``: a terminal row
    closes with zero, the trajectory's living end with its stored tail (None if no tail)."""
    total, discount = 0.0, 1.0
    size = len(values)
    for j in range(k):
        if j >= size:
            return None
        total += discount * float(reward[j])
        discount *= math.exp(-float(years[j]) / discount_years)
        if terminal[j]:
            return total
        if j + 1 == size:
            if not tail_ok[j]:
                return None
            return total + discount * float(tail[j])
    return total + discount * float(values[k])


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    payload_files = sorted(Path(args.payloads).glob("**/advantage-candidates.pt"))
    H = int(getattr(args, "horizon", 3))  # the search's final horizon ("H3" when 3)
    roots: list[RootRec] = []
    succ = SuccessorData()
    for f in payload_files:
        shard = torch.load(f, map_location="cpu", weights_only=True)
        label = str(f.relative_to(args.payloads)).replace("\\", "/")
        roots.extend(parse_payload(shard, label, H))
        for root in roots[-len(shard["roots"]) :]:
            root.extra["core"] = np.asarray(shard["tensors"]["core"][root.tensor_row])
        succ.add(shard, roots[-len(shard["roots"]) :])
    roots.sort(key=lambda r: (r.world, r.turn, r.camp))
    worlds_seen = sorted({r.world for r in roots})
    if any(3000 <= w <= 3059 for w in worlds_seen):
        raise ValueError("worlds 3000-3059 are never used here")
    qol_scale = float(args.qol_scale)
    s_units = {
        "qol_years": qol_scale,
        "population_years": FROZEN_SCALES["population_years"],
        "feasibility": 1.0,
        "settlement_guard": 1.0,
    }
    scale_sets = {
        "frozen": dict(FROZEN_SCALES),
        "r5": {"qol_years": qol_scale, "population_years": float(args.population_scale)},
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "payload_files": [
            str(f.relative_to(args.payloads)).replace("\\", "/") for f in payload_files
        ],
        "roots": len(roots),
        "candidates_first_action": sum(len(r.cands) for r in roots),
        "worlds": worlds_seen,
        "roots_per_world": {str(w): sum(r.world == w for r in roots) for w in worlds_seen},
        "s_units": s_units,
        "combined_scales": scale_sets,
        "successor_observations_in_payload": succ.present,
    }

    # ---- trajectory match, events, critic -----------------------------------
    traj = trajectory_index(Path(args.trajectory)) if args.trajectory else {}
    match_ok = 0
    root_rows: list[dict[str, Any]] = []
    for root in roots:
        t = traj.get(root.world)
        entry: dict[str, Any] = {"root": root.root_id, "matched": False}
        if t is not None:
            future = camp_future_rows(t, root.turn, root.camp, EVENT_WINDOW + 1)
            if future and t["turns"][future[0]] == root.turn:
                core0 = np.load(t["dir"] / "core.npy", mmap_mode="r")[future[0]]
                same = bool(np.array_equal(np.asarray(core0), root.extra["core"]))
                if same:
                    match_ok += 1
                    entry.update(matched=True, world=root.world, rows=future)
        root_rows.append(entry)
    report["trajectory_match"] = {
        "matched_roots": match_ok,
        "rule": "exact equality of the root core observation row with the trajectory row "
        "at the same (world, turn, camp)",
    }

    # Event split
    events: dict[str, dict[str, bool] | None] = {}
    if traj:
        for root, entry in zip(roots, root_rows, strict=True):
            if not entry["matched"]:
                events[root.root_id] = None
                continue
            core = np.load(traj[root.world]["dir"] / "core.npy", mmap_mode="r")
            events[root.root_id] = event_flags(np.asarray(core[entry["rows"]]))
    report["event_counts"] = {
        "known": sum(v is not None for v in events.values()),
        "event": sum(bool(v and v["event"]) for v in events.values()),
        "fission": sum(bool(v and v["fission"]) for v in events.values()),
        "emigration": sum(bool(v and v["emigration"]) for v in events.values()),
        "immigration": sum(bool(v and v["immigration"]) for v in events.values()),
    }
    median_turn = float(np.median([r.turn for r in roots]))
    report["early_late_split_turn"] = median_turn

    def split_masks(rows: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        ev = np.asarray(
            [
                (None if events.get(r) is None else events[r]["event"]) if events else None
                for r in rows["root"]
            ],
            dtype=object,
        )
        masks = {
            "all": np.ones(len(rows["root"]), dtype=bool),
            "early": rows["turn"] < median_turn,
            "late": rows["turn"] >= median_turn,
            "no_controls": ~rows["control"],
        }
        if events:
            masks["event"] = ev == True  # noqa: E712
            masks["non_event"] = ev == False  # noqa: E712
        for w in worlds_seen:
            masks[f"world_{w}"] = rows["world"] == w
        return masks

    # ---- realised structure, reward-only floor --------------------------------
    realised: dict[str, Any] = {}
    cache: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    for horizon in (1, H):
        for baseline in ("h12", "mean"):
            for streams in ("reeval", "0") if horizon != 1 else ("0",):
                rows = realised_rows(roots, horizon=horizon, baseline=baseline, streams=streams)
                cache[(horizon, baseline, streams)] = rows
                key = f"H{horizon}/{baseline}/streams-{streams}"
                if not rows:  # e.g. no survivor reached the final horizon
                    realised[key] = {"rows": 0}
                    continue
                block: dict[str, Any] = {"rows": len(rows["root"]), "heads": {}}
                for k, name in enumerate(COMPONENTS):
                    real = rows["real"][:, k]
                    block["heads"][name] = {
                        "within_root_sd_natural": within_root_spread(rows, real),
                        "within_root_sd_S": None
                        if within_root_spread(rows, real) is None
                        else within_root_spread(rows, real) / s_units[name],
                        "nonzero_fraction": float(np.mean(real != 0)),
                    }
                for sname, scales in scale_sets.items():
                    comb = combine(vec_to_heads(rows["real"]), scales)
                    block["heads"][f"combined_{sname}"] = {
                        "within_root_sd": within_root_spread(rows, comb),
                        "nonzero_fraction": float(np.mean(comb != 0)),
                    }
                realised[key] = block
    report["realised"] = realised
    report["search_noise"] = stream_noise(roots)
    for name, v in report["search_noise"]["per_head"].items():
        if v["rms_single_stream_sd"] is not None:
            v["rms_single_stream_sd_S"] = v["rms_single_stream_sd"] / s_units[name]

    def head_series(rows: Mapping[str, np.ndarray], which: str) -> dict[str, np.ndarray]:
        heads = {name: rows[which][:, k] / s_units[name] for k, name in enumerate(COMPONENTS)}
        for sname, scales in scale_sets.items():
            heads[f"combined_{sname}"] = combine(vec_to_heads(rows[which]), scales)
        return heads

    floor: dict[str, Any] = {}
    for key in ((f"H{H}", "h12", "reeval"), (f"H{H}", "mean", "reeval"), (f"H{H}", "h12", "0")):
        rows = cache[(H, key[1], key[2])]
        if not rows:
            continue
        pred = head_series(rows, "reward_only")
        real = head_series(rows, "real")
        masks = split_masks(rows)
        block = {}
        for head in pred:
            block[head] = {}
            for mname, m in masks.items():
                if m.sum() < 2:
                    continue
                block[head][mname] = within_root_metrics(
                    pred[head][m],
                    real[head][m],
                    rows["root"][m],
                    rows["world"][m],
                    margin=args.margin,
                )["world_balanced"]
        floor["/".join(key)] = block
    report["reward_only_floor"] = {
        "definition": "predicted A(a) = r1(a) - r1(baseline): PPO's one-step advantage when "
        "g V(s'_a) is equal across a root's candidates (critic-free floor); realised = H3 "
        "advantage with the same baseline. Units: S per head; combined in PPO units.",
        "margin": args.margin,
        "metrics": floor,
    }

    # Field separation (realised H1 / H3, reward-only)
    fsep = {}
    for label, rows, which in (
        ("H1_all/real", cache[(1, "mean", "0")], "real"),
        (f"H{H}_survivors/real", cache[(H, "mean", "reeval")], "real"),
        (f"H{H}_survivors/reward_only", cache[(H, "mean", "reeval")], "reward_only"),
    ):
        if not rows:
            continue
        series = head_series(rows, which)
        fsep[label] = {head: field_separation(rows, v) for head, v in series.items()}
    report["field_separation"] = fsep

    # ---- successor rows: full V-based per-candidate advantage ----------------------
    if succ.present:
        report["successor_check"] = successor_section(
            args, roots, succ, root_rows, traj, events, s_units, scale_sets, median_turn
        )

    # ---- critic ------------------------------------------------------------
    if traj and args.arm:
        report["critic"] = critic_section(
            args,
            roots,
            root_rows,
            traj,
            cache,
            s_units,
            scale_sets,
            events,
            split_masks,
            median_turn,
        )
    return report


def critic_section(
    args, roots, root_rows, traj, cache, s_units, scale_sets, events, split_masks, median_turn
):
    import torch

    from .labor_critic_ensemble import load_critic_ensemble
    from .labor_pretrain_eval import critic_predictions

    # rows t..t+3 of each matched root's camp, per world
    needed: dict[int, list[int]] = defaultdict(list)
    for entry in root_rows:
        if entry["matched"]:
            needed[entry["world"]].extend(entry["rows"][:4])
    data_parts: list[dict[str, np.ndarray]] = []
    position: dict[tuple[int, int], int] = {}
    offset = 0
    for world in sorted(needed):
        rows = sorted(set(needed[world]))
        part = gather_rows(traj[world]["dir"], rows)
        for i, r in enumerate(rows):
            position[(world, r)] = offset + i
        offset += len(rows)
        data_parts.append(part)
    keys = set(data_parts[0])
    for p in data_parts[1:]:
        keys &= set(p)
    dataset = {
        k: torch.from_numpy(np.concatenate([p[k] for p in data_parts])) for k in sorted(keys)
    }
    n = offset
    every = np.arange(n)
    # per-row natural-unit arrays
    reward = {name: dataset[f"reward_{name}"].double().numpy() for name in COMPONENTS}
    years = dataset["transition_elapsed_years"].double().numpy()
    terminal = dataset["transition_terminal"].numpy().astype(bool)
    tail = {name: dataset[f"tail_value__{name}"].double().numpy() for name in COMPONENTS}
    tail_ok = np.isin(dataset["tail_method"].numpy(), (1, 2))

    # check: H12 candidate first-step reward == trajectory reward at the root row
    rdiff = 0.0
    for root, entry in zip(roots, root_rows, strict=True):
        if entry["matched"]:
            i = position[(entry["world"], entry["rows"][0])]
            for k, name in enumerate(COMPONENTS):
                rdiff = max(rdiff, abs(root.h12.r1[k] - reward[name][i]))

    out: dict[str, Any] = {"h12_reward_max_abs_diff": rdiff, "arms": {}}
    for arm in args.arm:
        name_, _, paths = arm.partition("=")
        paths = [Path(p) for p in paths.split(",")]
        ensemble = load_critic_ensemble(paths)
        member_values = []
        for member in ensemble.members:
            preds = critic_predictions(member.task, member.costs, dataset, every, 256)
            member_values.append({h: np.asarray(preds[h], dtype=np.float64) for h in COMPONENTS})
        mean_values = {h: np.mean([m[h] for m in member_values], axis=0) for h in COMPONENTS}
        out["arms"][name_] = arm_metrics(
            name_,
            ensemble,
            member_values,
            mean_values,
            roots,
            root_rows,
            position,
            reward,
            years,
            terminal,
            tail,
            tail_ok,
            cache,
            s_units,
            scale_sets,
            events,
            split_masks,
            args,
        )
    return out


def _nstep_errors(values, roots, root_rows, position, reward, years, terminal, tail, tail_ok, k):
    """Per matched root: target - V(s) per head (natural units), the error of PPO's k-step
    advantage for the executed H12 candidate (true value 0)."""
    out = {}
    for root, entry in zip(roots, root_rows, strict=True):
        if not entry["matched"]:
            continue
        idx = [position[(entry["world"], r)] for r in entry["rows"][: k + 1]]
        errs = []
        for name in COMPONENTS:
            target = nstep_targets(
                values[name][idx],
                reward[name][idx],
                years[idx],
                terminal[idx],
                tail[name][idx],
                tail_ok[idx],
                k,
            )
            if target is None:
                errs = None
                break
            errs.append(target - values[name][idx[0]])
        if errs is not None:
            out[root.root_id] = np.asarray(errs)
    return out


def _err_stats(errs: Mapping[str, np.ndarray], world_of: Mapping[str, int], s_units, scale_sets):
    ids = sorted(errs)
    if not ids:
        return {}, {}
    arr = np.stack([errs[i] for i in ids])
    worlds = np.asarray([world_of[i] for i in ids])
    series = {name: arr[:, k] / s_units[name] for k, name in enumerate(COMPONENTS)}
    for sname, scales in scale_sets.items():
        series[f"combined_{sname}"] = combine(vec_to_heads(arr), scales)
    stats = {}
    for head, v in series.items():
        per_world = {
            int(w): {
                "mae": float(np.mean(np.abs(v[worlds == w]))),
                "bias": float(np.mean(v[worlds == w])),
                "rms": float(np.sqrt(np.mean(np.square(v[worlds == w])))),
                "rows": int((worlds == w).sum()),
            }
            for w in np.unique(worlds)
        }
        stats[head] = {"world_balanced": world_mean(per_world), "per_world": per_world}
    return stats, {head: dict(zip(ids, v.tolist(), strict=True)) for head, v in series.items()}


def arm_metrics(
    name,
    ensemble,
    member_values,
    mean_values,
    roots,
    root_rows,
    position,
    reward,
    years,
    terminal,
    tail,
    tail_ok,
    cache,
    s_units,
    scale_sets,
    events,
    split_masks,
    args,
):
    world_of = {r.root_id: r.world for r in roots}
    turn_of = {r.root_id: r.turn for r in roots}
    result: dict[str, Any] = {
        "members": [str(m.path) for m in ensemble.members],
        "member_sha256": [m.sha256 for m in ensemble.members],
    }
    errors: dict[int, Any] = {}
    for k in (1, 3):
        errs = _nstep_errors(
            mean_values, roots, root_rows, position, reward, years, terminal, tail, tail_ok, k
        )
        stats, per_root = _err_stats(errs, world_of, s_units, scale_sets)
        # splits
        splits = {}
        for head, by_root in per_root.items():
            ids = np.asarray(sorted(by_root))
            v = np.asarray([by_root[i] for i in ids])
            masks = {
                "early": np.asarray(
                    [turn_of[i] < float(np.median([r.turn for r in roots])) for i in ids]
                ),
                "event": np.asarray([bool(events.get(i) and events[i]["event"]) for i in ids]),
                "non_event": np.asarray(
                    [events.get(i) is not None and not events[i]["event"] for i in ids]
                ),
            }
            masks["late"] = ~masks["early"]
            splits[head] = {
                m: None
                if not mask.any()
                else {
                    "mae": float(np.mean(np.abs(v[mask]))),
                    "rms": float(np.sqrt(np.mean(np.square(v[mask])))),
                    "rows": int(mask.sum()),
                }
                for m, mask in masks.items()
            }
        # per member
        members = []
        for mv in member_values:
            e_m = _nstep_errors(
                mv, roots, root_rows, position, reward, years, terminal, tail, tail_ok, k
            )
            s_m, _ = _err_stats(e_m, world_of, s_units, scale_sets)
            members.append({h: s_m[h]["world_balanced"]["mae"] for h in s_m})
        result[f"executed_error_k{k}"] = {
            "definition": f"{k}-step target (trajectory rewards, physical-time discount, "
            f"bootstrap V(s_t+{k}); terminal 0; living end stored tail) minus V(s_t) at the "
            "root: the error of PPO's advantage for the executed H12 candidate (true 0 under "
            "the deterministic H12 step and continuation). S units per head; combined PPO units.",
            "stats": stats,
            "splits": splits,
            "per_member_mae": members,
            "per_member_mae_spread": {
                h: {
                    "min": float(min(m[h] for m in members)),
                    "max": float(max(m[h] for m in members)),
                    "sd": float(np.std([m[h] for m in members])),
                }
                for h in members[0]
            }
            if members
            else {},
        }
        errors[k] = per_root

    # ensemble member disagreement of V at roots (proxy for successor-value error scale)
    root_idx = [position[(e["world"], e["rows"][0])] for e in root_rows if e["matched"]]
    disagreement = {}
    for head in COMPONENTS:
        stack = np.stack([mv[head][root_idx] for mv in member_values])
        disagreement[head] = float(np.sqrt(np.mean(np.var(stack, axis=0)))) / s_units[head]
    for sname, scales in scale_sets.items():
        stack = np.stack(
            [combine({h: mv[h][root_idx] for h in COMPONENTS}, scales) for mv in member_values]
        )
        disagreement[f"combined_{sname}"] = float(np.sqrt(np.mean(np.var(stack, axis=0))))
    result["member_disagreement_V_root_rms_sd"] = disagreement

    # SNR and modelled flips
    snr: dict[str, Any] = {}
    flips: dict[str, Any] = {}
    rows = cache[(int(getattr(args, "horizon", 3)), "h12", "reeval")]
    real = {name: rows["real"][:, k] / s_units[name] for k, name in enumerate(COMPONENTS)}
    for sname, scales in scale_sets.items():
        real[f"combined_{sname}"] = combine(vec_to_heads(rows["real"]), scales)
    for head, v in real.items():
        signal = within_root_spread(rows, v)
        e1 = np.asarray(list(errors[1][head].values()))
        e3 = np.asarray(list(errors[3][head].values()))
        rms1 = float(np.sqrt(np.mean(e1**2))) if e1.size else None
        rms3 = float(np.sqrt(np.mean(e3**2))) if e3.size else None
        mae1 = float(np.mean(np.abs(e1))) if e1.size else None
        med1 = float(np.median(np.abs(e1))) if e1.size else None
        snr[head] = {
            "signal_within_root_sd_H3": signal,
            "error_rms_k1": rms1,
            "error_rms_k3": rms3,
            "error_mae_k1": mae1,
            "error_median_abs_k1": med1,
            "snr_mae_k1": None if not mae1 or signal is None else signal / mae1,
            "snr_median_k1": None if not med1 or signal is None else signal / med1,
            "snr_k1": None if not rms1 or signal is None else signal / rms1,
            "snr_k3": None if not rms3 or signal is None else signal / rms3,
            "member_disagreement": disagreement.get(head),
        }
    # pair differences (non-H12 vs every other survivor in the root, i<j)
    for sname, scales in scale_sets.items():
        comb = real[f"combined_{sname}"]
        pop_coeff = 0.5 / scales["population_years"] * s_units["population_years"]
        diffs = []
        for gid in np.unique(rows["root"]):
            kk = np.nonzero(rows["root"] == gid)[0]
            for a in range(len(kk)):
                for b in range(a + 1, len(kk)):
                    diffs.append(comb[kk[a]] - comb[kk[b]])
        diffs = np.asarray(diffs)
        keep = np.abs(diffs) > args.margin
        e1c = np.asarray(list(errors[1][f"combined_{sname}"].values()))
        e1p = np.asarray(list(errors[1]["population_years"].values())) * pop_coeff
        e3c = np.asarray(list(errors[3][f"combined_{sname}"].values()))
        e3p = np.asarray(list(errors[3]["population_years"].values())) * pop_coeff
        flips[sname] = {
            "pairs": int(diffs.size),
            "pairs_above_margin": int(keep.sum()),
            "combined_error_k1_empirical": flip_rate_empirical(diffs[keep], e1c),
            "combined_error_k3_empirical": flip_rate_empirical(diffs[keep], e3c),
            "population_only_error_k1_empirical": flip_rate_empirical(diffs[keep], e1p),
            "population_only_error_k3_empirical": flip_rate_empirical(diffs[keep], e3p),
            "combined_error_k1_gaussian": flip_rate_gaussian(
                diffs[keep], float(np.sqrt(np.mean(e1c**2)))
            ),
            "population_only_error_k1_gaussian": flip_rate_gaussian(
                diffs[keep], float(np.sqrt(np.mean(e1p**2)))
            ),
        }
    result["snr"] = snr
    result["modelled_sign_flips"] = {
        "definition": "same-root survivor pairs (H3, re-evaluated, |realised combined diff| > "
        "margin); each candidate's advantage error drawn independently from the measured "
        "executed-candidate error sample (all roots). Population-only: the population head's "
        "error times its combined coefficient. A model, not a measurement: successor errors "
        "of one root are likely positively correlated, which lowers the true flip rate.",
        "margin": args.margin,
        "rates": flips,
    }
    result["per_root_errors"] = {f"k{k}": errors[k] for k in errors}
    return result


# ---------------------------------------------------------------------------
# Successor rows (search export v2): the full V-based per-candidate advantage
# ---------------------------------------------------------------------------


class SuccessorData:
    """Root and successor critic-input rows of every v2 payload, concatenated."""

    def __init__(self) -> None:
        self.root_parts: list[dict[str, Any]] = []
        self.succ_parts: list[dict[str, Any]] = []
        self.root_rows = 0
        self.succ_rows = 0
        self.roots: list[RootRec] = []

    @property
    def present(self) -> bool:
        return bool(self.roots)

    @staticmethod
    def _merge(tensors: Mapping[str, Any], privileged: Mapping[str, Any] | None) -> dict:
        out = dict(tensors)
        if privileged is not None:
            out.update(privileged)
        return out

    def add(self, shard: Mapping[str, Any], roots: Sequence[RootRec]) -> None:
        if "successor_row_count" not in shard or shard.get("successor_tensors") is None:
            return
        raw = {str(r["root_id"]): r for r in shard["roots"]}
        for root in roots:
            root.extra["succ"] = parse_successors(raw[root.root_id], self.succ_rows)
            root.extra["root_global_row"] = self.root_rows + root.tensor_row
            self.roots.append(root)
        self.root_parts.append(self._merge(shard["tensors"], shard.get("privileged")))
        self.root_rows += int(shard["tensors"]["core"].shape[0])
        count = int(shard["successor_row_count"])
        if count:
            self.succ_parts.append(
                self._merge(shard["successor_tensors"], shard.get("successor_privileged"))
            )
        self.succ_rows += count

    @staticmethod
    def _concat(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        if not parts:
            return {}
        keys = set(parts[0])
        for p in parts[1:]:
            keys &= set(p)
        return {k: torch.cat([torch.as_tensor(p[k]) for p in parts]) for k in sorted(keys)}

    def datasets(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._concat(self.root_parts), self._concat(self.succ_parts)


def _ensemble_values(paths: Sequence[Path], datasets: Sequence[Mapping[str, Any]]):
    """Per member, per dataset: decoded predictions per head (value units)."""
    from .labor_critic_ensemble import load_critic_ensemble
    from .labor_pretrain_eval import critic_predictions

    ensemble = load_critic_ensemble([Path(p) for p in paths])
    members = []
    for member in ensemble.members:
        per = []
        for data in datasets:
            n = int(data["core"].shape[0]) if data else 0
            if n == 0:
                per.append({h: np.zeros(0) for h in COMPONENTS})
                continue
            preds = critic_predictions(member.task, member.costs, data, np.arange(n), 256)
            per.append({h: np.asarray(preds[h], dtype=np.float64) for h in COMPONENTS})
        members.append(per)
    return ensemble, members


def successor_target(
    stream: SuccStream, k: int, v_succ: Mapping[str, np.ndarray], discount_years: float = 400.0
) -> np.ndarray | None:
    """``sum_{j<k} prod g r_j + prod g * V(s_{a,k})`` per head (raw units) on one stream: a
    terminated focal camp closes with zero; an inactive one, a missing step or a missing
    row at decision k make the target unavailable (None)."""
    total = np.zeros(len(COMPONENTS))
    discount = 1.0
    for j in range(k):
        if j >= len(stream.rewards):
            return None
        total = total + discount * stream.rewards[j]
        discount *= math.exp(-float(stream.years[j]) / discount_years)
        status = stream.status.get(j + 1)
        if status == "terminated":
            return total
        if status != "live":
            return None
    row = stream.row.get(k)
    if row is None:
        return None
    return total + discount * np.asarray([v_succ[h][row] for h in COMPONENTS])


def successor_rows_table(
    roots: Sequence[RootRec],
    v_root: Mapping[str, np.ndarray],
    v_succ: Mapping[str, np.ndarray],
    *,
    k: int,
    mode: str,
) -> dict[str, np.ndarray]:
    """Per-candidate predicted ``A_k = target_k - V(s)`` and realised H3 advantage
    ``G3(a) - G3(h12)`` (raw units, [4]); H3 survivors (twins excluded). ``mode``:
    ``"reeval"`` averages both over the streams >= 1 every included candidate shares
    (paired with H12 per stream); ``"0"`` uses the deterministic stream 0. Also counts
    candidates dropped for an unavailable target."""
    rows: dict[str, list[Any]] = defaultdict(list)
    dropped = 0
    for root in roots:
        succ = root.extra.get("succ") or {}
        vs = np.asarray([v_root[h][root.extra["root_global_row"]] for h in COMPONENTS])
        cands = [c for c in root.cands if not c.twin and c.g3 and c.index in succ]
        if not any(c.is_h12 for c in cands):
            continue
        if mode == "0":
            streams = [0]
            cands = [c for c in cands if 0 in c.g3]
        else:
            sets = [{s for s in c.g3 if s >= 1} & {e.stream for e in succ[c.index]} for c in cands]
            streams = sorted(set.intersection(*sets)) if sets else []
        if not streams or not any(c.is_h12 for c in cands):
            continue
        preds, keep = [], []
        for c in cands:
            by = {e.stream: e for e in succ[c.index]}
            targets = [successor_target(by[s], k, v_succ) if s in by else None for s in streams]
            if any(t is None for t in targets):
                dropped += 1
                continue
            preds.append(np.mean(targets, axis=0) - vs)
            keep.append(c)
        if len(keep) < 2 or not any(c.is_h12 for c in keep):
            continue
        h = next(i for i, c in enumerate(keep) if c.is_h12)
        value = np.stack([np.mean([c.g3[s] for s in streams], axis=0) for c in keep])
        real = value - value[h]
        for i, c in enumerate(keep):
            rows["root"].append(root.root_id)
            rows["world"].append(root.world)
            rows["turn"].append(root.turn)
            rows["is_h12"].append(c.is_h12)
            rows["family"].append(c.family)
            rows["control"].append(c.control is not None or c.family.startswith("control"))
            rows["field"].append(c.field_share)
            rows["real"].append(real[i])
            rows["pred"].append(preds[i])
    out: dict[str, Any] = {"dropped": dropped}
    if not rows:
        return out
    out.update(
        {
            "root": np.asarray(rows["root"]),
            "world": np.asarray(rows["world"], dtype=np.int64),
            "turn": np.asarray(rows["turn"], dtype=np.int64),
            "is_h12": np.asarray(rows["is_h12"], dtype=bool),
            "family": np.asarray(rows["family"]),
            "control": np.asarray(rows["control"], dtype=bool),
            "field": np.asarray(rows["field"], dtype=np.float64),
            "real": np.stack(rows["real"]),
            "pred": np.stack(rows["pred"]),
        }
    )
    return out


def _series(arr: np.ndarray, s_units, scale_sets) -> dict[str, np.ndarray]:
    out = {name: arr[:, k] / s_units[name] for k, name in enumerate(COMPONENTS)}
    for sname, scales in scale_sets.items():
        out[f"combined_{sname}"] = combine(vec_to_heads(arr), scales)
    return out


def _subset(rows: Mapping[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in rows.items()}


def population_only_flips(rows: Mapping[str, Any], scale_sets, *, margin: float) -> dict[str, Any]:
    """Measured sign flips of same-root pair differences of the combined advantage
    (|realised combined diff| > margin) when only the population head is the critic's
    prediction (every other head realised), and when every head is predicted."""
    out: dict[str, Any] = {}
    for sname, scales in scale_sets.items():
        real = combine(vec_to_heads(rows["real"]), scales)
        mixed_arr = rows["real"].copy()
        mixed_arr[:, 1] = rows["pred"][:, 1]
        mixed = combine(vec_to_heads(mixed_arr), scales)
        full = combine(vec_to_heads(rows["pred"]), scales)
        per_world: dict[int, dict[str, Any]] = {}
        for w in sorted(set(int(x) for x in rows["world"])):
            m = rows["world"] == w
            n = flips_pop = flips_all = 0
            for gid in np.unique(rows["root"][m]):
                kk = np.nonzero(m & (rows["root"] == gid))[0]
                for a in range(len(kk)):
                    for b in range(a + 1, len(kk)):
                        d = real[kk[a]] - real[kk[b]]
                        if abs(d) <= margin:
                            continue
                        n += 1
                        flips_pop += int(np.sign(mixed[kk[a]] - mixed[kk[b]]) != np.sign(d))
                        flips_all += int(np.sign(full[kk[a]] - full[kk[b]]) != np.sign(d))
            per_world[w] = {
                "pairs": n,
                "population_only_flip_rate": None if n == 0 else flips_pop / n,
                "all_heads_flip_rate": None if n == 0 else flips_all / n,
            }
        out[sname] = {"world_balanced": world_mean(per_world), "per_world": per_world}
    return out


def successor_section(
    args, roots, succ: SuccessorData, root_rows, traj, events, s_units, scale_sets, median_turn
) -> dict[str, Any]:
    """The full V-based per-candidate advantage on v2 payloads (see the module docstring)."""
    root_data, succ_data = succ.datasets()
    used = succ.roots
    out: dict[str, Any] = {
        "definition": "predicted A_k(s,a) = sum_{j<k} g r_j(a) + g^k V(s_{a,k}) - V(s) on each "
        "stream (g = exp(-years/400); a terminated focal camp closes with 0; inactive or "
        "missing rows drop the candidate), V(s) from the payload's root row; realised = H3 "
        "advantage G3(a) - G3(h12) (truncated, no tail) on the same streams. reeval: mean "
        "over the shared streams >= 1; 0: deterministic stream 0. S units per head; combined "
        "in PPO units. Ensemble = per-row member mean.",
        "roots": len(used),
        "successor_rows": succ.succ_rows,
        "status_counts": dict(
            sorted(
                Counter(
                    st
                    for r in used
                    for streams in r.extra["succ"].values()
                    for e in streams
                    for st in e.status.values()
                ).items()
            )
        ),
        "margin": args.margin,
        "arms": {},
    }
    # Consistency: the H12 candidate's stream-0 decision-1 successor row is the trajectory's
    # next row of the focal camp (deterministic H12 step) when the root lies on it.
    if traj:
        checked = equal = 0
        entry_of = {e["root"]: e for e in root_rows}
        for r in used:
            e = entry_of.get(r.root_id)
            h12 = r.h12
            streams = {s.stream: s for s in r.extra["succ"].get(h12.index, [])}
            if not e or not e.get("matched") or len(e["rows"]) < 2 or 0 not in streams:
                continue
            row = streams[0].row.get(1)
            if row is None:
                continue
            core = np.load(traj[r.world]["dir"] / "core.npy", mmap_mode="r")[e["rows"][1]]
            checked += 1
            equal += int(np.array_equal(np.asarray(core), succ_data["core"][row].numpy()))
        out["h12_successor_equals_trajectory_next_row"] = {"checked": checked, "equal": equal}
    if not args.arm:
        out["note"] = "no --arm: successor rows present but no critic to evaluate"
        return out

    def masks_for(rows):
        ev = np.asarray(
            [
                None if not events or events.get(r) is None else events[r]["event"]
                for r in rows["root"]
            ],
            dtype=object,
        )
        m = {
            "all": np.ones(len(rows["root"]), dtype=bool),
            "no_controls": ~rows["control"],
            "early": rows["turn"] < median_turn,
            "late": rows["turn"] >= median_turn,
        }
        if events:
            m["event"] = ev == True  # noqa: E712
            m["non_event"] = ev == False  # noqa: E712
        return m

    for arm in args.arm:
        name_, _, paths = arm.partition("=")
        ensemble, members = _ensemble_values(paths.split(","), [root_data, succ_data])
        mean_root = {h: np.mean([m[0][h] for m in members], axis=0) for h in COMPONENTS}
        mean_succ = {h: np.mean([m[1][h] for m in members], axis=0) for h in COMPONENTS}
        block: dict[str, Any] = {
            "members": [str(m.path) for m in ensemble.members],
            "member_sha256": [m.sha256 for m in ensemble.members],
        }
        # member disagreement of V(s'_a) (RMS SD over successor rows), S / PPO units
        dis = {}
        if succ.succ_rows:
            for h in COMPONENTS:
                stack = np.stack([m[1][h] for m in members])
                dis[h] = float(np.sqrt(np.mean(np.var(stack, axis=0)))) / s_units[h]
            for sname, scales in scale_sets.items():
                stack = np.stack(
                    [combine({h: m[1][h] for h in COMPONENTS}, scales) for m in members]
                )
                dis[f"combined_{sname}"] = float(np.sqrt(np.mean(np.var(stack, axis=0))))
        block["member_disagreement_V_successor_rms_sd"] = dis
        results: dict[str, Any] = {}
        for k in sorted({1, int(getattr(args, "horizon", 3))}):
            for mode in ("reeval", "0"):
                rows = successor_rows_table(used, mean_root, mean_succ, k=k, mode=mode)
                key = f"k{k}/streams-{mode}"
                if "root" not in rows:
                    results[key] = {"rows": 0, "dropped": rows["dropped"]}
                    continue
                pred = _series(rows["pred"], s_units, scale_sets)
                real = _series(rows["real"], s_units, scale_sets)
                metrics: dict[str, Any] = {}
                for head in pred:
                    metrics[head] = {}
                    for mname, m in masks_for(rows).items():
                        if m.sum() < 2:
                            continue
                        metrics[head][mname] = within_root_metrics(
                            pred[head][m],
                            real[head][m],
                            rows["root"][m],
                            rows["world"][m],
                            margin=args.margin,
                        )["world_balanced"]
                field = {
                    head: {
                        "predicted": field_separation(rows, pred[head]),
                        "realised": field_separation(rows, real[head]),
                    }
                    for head in pred
                }
                flips = {
                    "with_controls": population_only_flips(rows, scale_sets, margin=args.margin),
                    "no_controls": population_only_flips(
                        _subset(rows, ~rows["control"]), scale_sets, margin=args.margin
                    ),
                }
                # per member: MAE and pair sign accuracy (all, no controls)
                member_stats = []
                for m in members:
                    mr = successor_rows_table(
                        used,
                        {h: m[0][h] for h in COMPONENTS},
                        {h: m[1][h] for h in COMPONENTS},
                        k=k,
                        mode=mode,
                    )
                    if "root" not in mr:
                        continue
                    mp = _series(mr["pred"], s_units, scale_sets)
                    mreal = _series(mr["real"], s_units, scale_sets)
                    nc = ~mr["control"]
                    stat = {}
                    for head in mp:
                        allm = within_root_metrics(
                            mp[head], mreal[head], mr["root"], mr["world"], margin=args.margin
                        )["world_balanced"]
                        ncm = within_root_metrics(
                            mp[head][nc],
                            mreal[head][nc],
                            mr["root"][nc],
                            mr["world"][nc],
                            margin=args.margin,
                        )["world_balanced"]
                        stat[head] = {
                            "mae": allm["mae"],
                            "pair_sign_accuracy": allm["pair_sign_accuracy"],
                            "pair_sign_accuracy_no_controls": ncm["pair_sign_accuracy"],
                        }
                    member_stats.append(stat)
                spread = {}
                if member_stats:
                    for head in member_stats[0]:
                        spread[head] = {}
                        for stat_name in member_stats[0][head]:
                            vals = [
                                ms[head][stat_name]
                                for ms in member_stats
                                if ms[head][stat_name] is not None
                            ]
                            spread[head][stat_name] = (
                                None
                                if not vals
                                else {
                                    "min": float(min(vals)),
                                    "max": float(max(vals)),
                                    "sd": float(np.std(vals)),
                                }
                            )
                results[key] = {
                    "rows": int(len(rows["root"])),
                    "roots": int(len(np.unique(rows["root"]))),
                    "dropped": rows["dropped"],
                    "metrics": metrics,
                    "field_separation": field,
                    "measured_sign_flips": flips,
                    "per_member": member_stats,
                    "member_spread": spread,
                }
        # all candidates, stream 0, k=1: the predicted within-root spread per head (the
        # population head's first-step reward is identical across candidates, so its spread
        # comes from g V(s'_a) alone)
        allrows: dict[str, list[Any]] = defaultdict(list)
        for root in used:
            sc = root.extra["succ"]
            vs = np.asarray([mean_root[h][root.extra["root_global_row"]] for h in COMPONENTS])
            for c in root.cands:
                if c.twin or c.index not in sc:
                    continue
                by = {e.stream: e for e in sc[c.index]}
                t = successor_target(by[0], 1, mean_succ) if 0 in by else None
                if t is None:
                    continue
                allrows["root"].append(root.root_id)
                allrows["pred"].append(t - vs)
                allrows["field"].append(c.field_share)
        if allrows:
            ar = {
                "root": np.asarray(allrows["root"]),
                "field": np.asarray(allrows["field"]),
            }
            ps = _series(np.stack(allrows["pred"]), s_units, scale_sets)
            results["k1/all-candidates/stream-0"] = {
                "rows": len(allrows["root"]),
                "predicted_within_root_sd": {h: within_root_spread(ar, v) for h, v in ps.items()},
                "predicted_field_separation": {h: field_separation(ar, v) for h, v in ps.items()},
            }
        block["results"] = results
        out["arms"][name_] = block
    return out


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else float(f"{value:.6g}")
    if isinstance(value, dict):
        return {str(k): _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    if isinstance(value, (np.floating,)):
        return _round(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--payloads", type=Path, required=True, help="directory of search shards")
    p.add_argument("--trajectory", type=Path, help="H12 trajectory dataset the roots lie on")
    p.add_argument(
        "--arm",
        action="append",
        default=[],
        help="name=ckpt1,ckpt2,... (an ensemble; repeat per arm)",
    )
    p.add_argument("--qol-scale-artifact", type=Path, help="QoL reference-scale artifact")
    p.add_argument("--population-scale", type=float, default=R5_POPULATION_SCALE)
    p.add_argument("--margin", type=float, default=0.01, help="pair margin (S / PPO units)")
    p.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="the search's final horizon (realised H returns; successor k = 1 and this)",
    )
    p.add_argument("--out", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.qol_scale_artifact is not None:
        from .labor_pretrain_eval import load_qol_reference_scale

        args.qol_scale = load_qol_reference_scale(args.qol_scale_artifact)["value"]
    else:
        args.qol_scale = FROZEN_SCALES["qol_years"]
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_round(report), sort_keys=True, indent=1) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
