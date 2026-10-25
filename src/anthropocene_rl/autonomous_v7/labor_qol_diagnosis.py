"""Read-only diagnosis of the QoL-years task-critic head (why it ranks poorly, or that it does not).

One command writes a JSON report (``--out``) and a markdown summary (``--summary``,
default ``<out>.md``)::

    python -m anthropocene_rl.autonomous_v7.labor_qol_diagnosis \\
        --checkpoint critic.pt --dataset CAL [--fit-dataset FIT --fit-worlds 10] \\
        [--search-dir SEARCH] --out report.json

Nothing is trained or written except the two report files. The sections are:

1. Target structure, for QoL-years and population-years. It reports the variance
   decomposition within and between (world, turn) groups, informative pairs and
   their size in reference-scale and symlog units against the value-support bin
   width, split by decisions remaining to the window end and by camp size.
2. Critic error decomposition. It reports error and ranking by stratum and by
   target magnitude, ranking accuracy against the |dG| margin, the entropy of the
   predicted two-hot distribution, the within-group spread of predictions against
   targets (collapse), and the decoded tail above the observed target maximum.
3. Noise floor. Cheap regressors are fitted on actor-visible inputs: ridge on
   flattened and pooled input tensors, k-NN on their principal components, and a
   ridge probe on the critic's own trunk features. They are cross-validated by world
   and evaluated against the critic, for the full-window target and for the next
   1, 3 and 5 decisions.
4. Search candidate disparity, when ``--search-dir`` is given. It reports the
   spread of candidate returns per root and horizon against the re-evaluation
   standard error.
5. Verdict. The numbers are mapped to (i) noise or resolution, (ii) input
   information, (iii) critic optimisation, or (iv) genuine action-level disparity.
   Explicit thresholds are listed in ``VERDICT_THRESHOLDS``.

The within-(world, turn) ranking compares different camps at the same decision,
not different actions in one state. Action-level evidence comes only from the
search export. That export stores no successor tensors, so the V-based advantage
of a candidate cannot be computed from it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPORT_SCHEMA = "labor-qol-diagnosis-v1"
HEADS = ("qol_years", "population_years")
REWARD_KEYS = {"qol_years": "reward_qol_years", "population_years": "reward_population_years"}
SHORT_HORIZONS = (1, 3, 5)
INFORMATIVE_FRACTION = 0.05  # same as CRITIC_GATES["informative_delta_fraction"]
MARGIN_EDGES = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, math.inf)  # |dG| in reference scales
BIN_MULTIPLES = (1.0, 2.0, 5.0)
BIN_MARGIN_EDGES = (0.0, 1.0, 2.0, 5.0, 10.0, math.inf)  # |d symlog| in bins
REMAINING_BINS: tuple[tuple[str, int, int], ...] = (
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-6", 4, 6),
    ("7-12", 7, 12),
    ("13+", 13, 10**9),
)
# Reserved Q-pi worlds (manifest section 8): never used here, not even to fit a probe.
RESERVED_WORLDS = frozenset(range(90154, 90160))
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)  # times the mean eigenvalue
PCA_COMPONENTS = 32
COLLAPSE_RATIO = 0.25
VERDICT_THRESHOLDS = {
    "rank_pass": 0.70,  # G2C ranking gate
    "rank_weak": 0.65,  # at or below: "does not discriminate"
    "rank_gain": 0.05,  # a regressor "beats" another by at least this much
    "within_r2_noise": 0.05,  # within-group R2 at or below: no row-level signal
    "short_horizon_gain": 0.08,  # short-horizon ranking above the full window by this much
    "remaining_gain": 0.08,  # many-remaining rows rank better than few-remaining rows by this much
    "fit_gap": 0.10,  # fitting minus calibration ranking above this: generalisation gap
    "small_margin_bins": 2.0,  # informative pairs below this many bins count as resolution-bound
    "small_margin_share": 0.5,
    "search_root_share": 0.30,  # share of roots with a QoL pair beyond 2 SE and materiality
    "search_materiality": 0.05,  # normalized (reference-scale) materiality of a pair difference
    "qol_pop_corr_redundant": 0.80,
}


def _log(message: str) -> None:
    print(f"[qol-diagnosis {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Pure metric math (unit tested)
# ---------------------------------------------------------------------------


def symlog(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def symexp(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return np.sign(y) * np.expm1(np.abs(y))


def group_ids(*keys: np.ndarray) -> np.ndarray:
    """Dense integer id per row for the tuple of keys."""
    stacked = np.stack([np.asarray(k).reshape(-1).astype(np.int64) for k in keys], axis=1)
    _, inverse = np.unique(stacked, axis=0, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64)


def quantiles(values: np.ndarray, qs: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {str(q): None for q in qs}
    return {str(q): float(np.quantile(values, q)) for q in qs}


def variance_decomposition(values: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """Population variance split into within-group and between-group parts (they sum)."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if not len(values):
        return {"total": None, "within": None, "between": None, "within_fraction": None}
    total = float(values.var())
    sums = np.bincount(groups, weights=values)
    counts = np.bincount(groups)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    within = float(((values - means[groups]) ** 2).mean())
    between = total - within
    singleton = counts[groups] == 1
    return {
        "total": total,
        "within": within,
        "between": between,
        "within_fraction": within / total if total > 0 else None,
        "within_sd": math.sqrt(within),
        "groups": int((counts > 0).sum()),
        "rows_in_singleton_groups": int(singleton.sum()),
    }


def within_group_r2(pred: np.ndarray, target: np.ndarray, groups: np.ndarray) -> float | None:
    """1 - SS(demeaned residual) / SS(demeaned target); group mean scores 0."""
    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    groups = np.asarray(groups).reshape(-1)
    counts = np.bincount(groups)
    dp = pred - (np.bincount(groups, weights=pred) / np.maximum(counts, 1))[groups]
    dt = target - (np.bincount(groups, weights=target) / np.maximum(counts, 1))[groups]
    ss = float((dt**2).sum())
    if not ss > 0:
        return None
    return 1.0 - float(((dt - dp) ** 2).sum()) / ss


def r2(pred: np.ndarray, target: np.ndarray) -> float | None:
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    ss = float(((target - target.mean()) ** 2).sum()) if len(target) else 0.0
    return None if not ss > 0 else 1.0 - float(((target - pred) ** 2).sum()) / ss


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), 0-based."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 2:
        return None
    a, b = a - a.mean(), b - b.mean()
    denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return None if not denominator > 0 else float((a * b).sum()) / denominator


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    return pearson(rankdata(a), rankdata(b)) if len(a) >= 2 else None


class PairIndex:
    """All unordered row pairs within each group (rows restricted by ``mask``)."""

    def __init__(self, groups: np.ndarray, worlds: np.ndarray, mask: np.ndarray | None = None):
        groups = np.asarray(groups).reshape(-1)
        worlds = np.asarray(worlds).reshape(-1)
        keep = np.ones(len(groups), bool) if mask is None else np.asarray(mask, bool).reshape(-1)
        rows = np.flatnonzero(keep)
        order = rows[np.argsort(groups[rows], kind="mergesort")]
        firsts, seconds = [], []
        if len(order):
            boundaries = np.flatnonzero(np.diff(groups[order])) + 1
            for members in np.split(order, boundaries):
                if len(members) < 2:
                    continue
                a, b = np.triu_indices(len(members), k=1)
                firsts.append(members[a])
                seconds.append(members[b])
        self.i = np.concatenate(firsts).astype(np.int64) if firsts else np.zeros(0, np.int64)
        self.j = np.concatenate(seconds).astype(np.int64) if seconds else np.zeros(0, np.int64)
        self.world = worlds[self.i] if len(self.i) else np.zeros(0, worlds.dtype)
        self.group = groups[self.i] if len(self.i) else np.zeros(0, groups.dtype)

    def __len__(self) -> int:
        return len(self.i)


def pair_ranking(
    pred: np.ndarray,
    target: np.ndarray,
    pairs: PairIndex,
    threshold: float,
    *,
    margin_edges: Sequence[float] = MARGIN_EDGES,
    margin_values: np.ndarray | None = None,
    bootstrap: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Informative pairwise ordering accuracy (ties in prediction count one half).

    Matches ``labor_pretrain_eval.ranking_accuracy`` for the pooled accuracy. Adds a
    split by margin (``|dG|`` by default, or ``margin_values`` per pair) and a world
    cluster bootstrap interval.
    """
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    dg = target[pairs.i] - target[pairs.j]
    dp = pred[pairs.i] - pred[pairs.j]
    informative = np.abs(dg) > threshold
    agree = np.where(dp == 0.0, 0.5, (np.sign(dp) == np.sign(dg)).astype(np.float64))
    hits = agree[informative]
    count = int(informative.sum())
    out: dict[str, Any] = {
        "accuracy": float(hits.mean()) if count else None,
        "informative_pairs": count,
        "pairs": len(pairs),
        "informative_fraction": count / len(pairs) if len(pairs) else None,
        "prediction_tie_fraction": float((dp[informative] == 0.0).mean()) if count else None,
    }
    margin = np.abs(dg) if margin_values is None else np.asarray(margin_values, np.float64)
    margin = margin[informative]
    by_margin = []
    for low, high in itertools.pairwise(margin_edges):
        inside = (margin > low) & (margin <= high)
        n = int(inside.sum())
        by_margin.append(
            {
                "low": low,
                "high": high if math.isfinite(high) else None,
                "pairs": n,
                "accuracy": float(hits[inside].mean()) if n else None,
            }
        )
    out["by_margin"] = by_margin
    if count:
        worlds = pairs.world[informative]
        unique, inverse = np.unique(worlds, return_inverse=True)
        world_hits = np.bincount(inverse, weights=hits, minlength=len(unique))
        world_counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
        out["world_balanced_accuracy"] = float(np.mean(world_hits / world_counts))
        out["worlds"] = len(unique)
        if bootstrap and len(unique) >= 2:
            rng = np.random.default_rng(seed)
            draws = rng.integers(0, len(unique), size=(bootstrap, len(unique)))
            values = world_hits[draws].sum(axis=1) / np.maximum(world_counts[draws].sum(axis=1), 1)
            out["bootstrap_ci95"] = [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
    return out


def informative_pair_structure(
    normalized: np.ndarray,
    pairs: PairIndex,
    *,
    threshold: float = INFORMATIVE_FRACTION,
    bin_width: float,
) -> dict[str, Any]:
    """Informative pair share and |dG| in reference scales, symlog units and bins."""
    normalized = np.asarray(normalized, np.float64)
    dg = np.abs(normalized[pairs.i] - normalized[pairs.j])
    informative = dg > threshold
    y = symlog(normalized)
    dy = np.abs(y[pairs.i] - y[pairs.j])[informative]
    bins = dy / bin_width
    return {
        "pairs": len(pairs),
        "informative_pairs": int(informative.sum()),
        "informative_fraction": float(informative.mean()) if len(pairs) else None,
        "abs_delta_reference_scales": quantiles(dg[informative]),
        "abs_delta_symlog": quantiles(dy),
        "abs_delta_bins": quantiles(bins),
        "fraction_below_bins": {
            f"{m:g}": float((bins < m).mean()) if len(bins) else None for m in BIN_MULTIPLES
        },
    }


def group_spread(
    pred: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    collapse_ratio: float = COLLAPSE_RATIO,
) -> dict[str, Any]:
    """Within-group prediction spread against target spread (collapse detector)."""
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    groups = np.asarray(groups).reshape(-1)
    counts = np.bincount(groups)
    multi = counts[groups] >= 2
    dp = pred - (np.bincount(groups, weights=pred) / np.maximum(counts, 1))[groups]
    dt = target - (np.bincount(groups, weights=target) / np.maximum(counts, 1))[groups]
    # Per-group second moments of the demeaned values (population sd, as numpy .std()).
    n_g = np.maximum(counts, 1)
    sd_t_g = np.sqrt(np.bincount(groups, weights=dt * dt, minlength=len(counts)) / n_g)
    sd_p_g = np.sqrt(np.bincount(groups, weights=dp * dp, minlength=len(counts)) / n_g)
    cov_g = np.bincount(groups, weights=dp * dt, minlength=len(counts)) / n_g
    scored = (counts >= 3) & (sd_t_g > 0)
    ratios = (sd_p_g[scored] / sd_t_g[scored]).tolist()
    with_corr = scored & (sd_p_g > 0)
    correlations = (cov_g[with_corr] / (sd_p_g[with_corr] * sd_t_g[with_corr])).tolist()
    var_t = float((dt[multi] ** 2).mean()) if multi.any() else 0.0
    var_p = float((dp[multi] ** 2).mean()) if multi.any() else 0.0
    cov = float((dp[multi] * dt[multi]).mean()) if multi.any() else 0.0
    ratios_a = np.asarray(ratios)
    return {
        "groups_scored": len(ratios),
        "pooled_sd_ratio": math.sqrt(var_p / var_t) if var_t > 0 else None,
        "median_sd_ratio": float(np.median(ratios_a)) if len(ratios_a) else None,
        "collapse_fraction": float((ratios_a < collapse_ratio).mean()) if len(ratios_a) else None,
        "collapse_ratio": collapse_ratio,
        "within_slope": cov / var_t if var_t > 0 else None,
        "mean_within_correlation": float(np.mean(correlations)) if correlations else None,
        "within_pred_sd": math.sqrt(var_p),
        "within_target_sd": math.sqrt(var_t),
    }


def world_balanced(worlds: np.ndarray) -> np.ndarray:
    worlds = np.asarray(worlds).reshape(-1)
    unique, inverse = np.unique(worlds, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    return len(worlds) / (len(unique) * counts[inverse])


def error_metrics(pred: np.ndarray, target: np.ndarray, worlds: np.ndarray) -> dict[str, Any]:
    """nMAE / bias in reference scales (inputs already normalized), world-balanced."""
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    if not len(pred):
        return {"rows": 0, "nmae": None, "bias": None}
    w = world_balanced(worlds)
    return {
        "rows": len(pred),
        "nmae": float((np.abs(pred - target) * w).sum() / w.sum()),
        "bias": float(((pred - target) * w).sum() / w.sum()),
        "target_mean": float((target * w).sum() / w.sum()),
        "rmse_symlog": float(np.sqrt(((symlog(pred) - symlog(target)) ** 2).mean())),
    }


def windowed_returns(
    reward: np.ndarray,
    elapsed: np.ndarray,
    terminal: np.ndarray,
    trajectory: np.ndarray,
    turns: np.ndarray,
    *,
    window: int | None,
    discount: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-trajectory discounted returns over the next ``window`` decisions.

    ``window=None`` reproduces ``labor_bc.discounted_returns`` exactly (a terminal
    row ends the sum, there is no tail after the last row). Also returns, per row,
    the decisions remaining in its segment (itself included) and whether the
    segment ended in a terminal row.
    """
    reward = np.asarray(reward, np.float64).reshape(-1)
    gamma = np.exp(-np.asarray(elapsed, np.float64).reshape(-1) / discount)
    terminal = np.asarray(terminal, bool).reshape(-1)
    trajectory = np.asarray(trajectory).reshape(-1)
    turns = np.asarray(turns).reshape(-1)
    n = len(reward)
    out = np.zeros(n, np.float64)
    remaining = np.zeros(n, np.int64)
    ends_terminal = np.zeros(n, bool)
    order = np.lexsort((turns, trajectory))
    boundaries = np.flatnonzero(np.diff(trajectory[order])) + 1
    for rows in np.split(order, boundaries):
        m = len(rows)
        if not m:
            continue
        r, g, t = reward[rows], gamma[rows], terminal[rows]
        # Segment end (inclusive) per position: the first terminal at or after it.
        end = np.empty(m, np.int64)
        nxt = m - 1
        for p in range(m - 1, -1, -1):
            if t[p]:
                nxt = p
            end[p] = nxt
        for p in range(m):
            stop = end[p] if window is None else min(end[p], p + window - 1)
            total, factor = 0.0, 1.0
            for q in range(p, stop + 1):
                total += factor * r[q]
                factor *= g[q]
            out[rows[p]] = total
            remaining[rows[p]] = end[p] - p + 1
            ends_terminal[rows[p]] = bool(t[end[p]])
    return out, remaining, ends_terminal


def remaining_labels(remaining: np.ndarray) -> np.ndarray:
    remaining = np.asarray(remaining).reshape(-1)
    labels = np.empty(len(remaining), dtype=object)
    for name, low, high in REMAINING_BINS:
        labels[(remaining >= low) & (remaining <= high)] = name
    return labels


def population_labels(population: np.ndarray) -> np.ndarray:
    from .labor_pretrain_eval import population_bin

    return population_bin(population)


def magnitude_labels(target: np.ndarray, count: int = 5) -> tuple[np.ndarray, list[float]]:
    target = np.asarray(target, np.float64)
    edges = (
        np.quantile(target, np.linspace(0, 1, count + 1)) if len(target) else np.zeros(count + 1)
    )
    labels = np.clip(np.searchsorted(edges[1:-1], target, side="right"), 0, count - 1)
    return np.asarray([f"q{k + 1}" for k in labels], dtype=object), [float(e) for e in edges]


def group_folds(units: np.ndarray, folds: int, seed: int = 0) -> np.ndarray:
    """Fold id per row such that every unit (world) lies in exactly one fold."""
    units = np.asarray(units).reshape(-1)
    unique = np.unique(units)
    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(len(unique))]
    assignment = {u: k % folds for k, u in enumerate(shuffled.tolist())}
    return np.asarray([assignment[u] for u in units.tolist()], dtype=np.int64)


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    p = probabilities.double()
    return -(p * torch.log(p.clamp_min(1e-300))).sum(dim=-1)


# ---------------------------------------------------------------------------
# Ridge / k-NN on dense features (float64, closed form via eigendecomposition)
# ---------------------------------------------------------------------------


class GramStats:
    """Sufficient statistics of raw features (and targets) over a row subset."""

    def __init__(
        self,
        n: int,
        s1: torch.Tensor,
        gram: torch.Tensor,
        s2: torch.Tensor,
        xty: torch.Tensor,
        ysum: torch.Tensor,
    ) -> None:
        self.n, self.s1, self.gram, self.s2, self.xty, self.ysum = n, s1, gram, s2, xty, ysum

    @classmethod
    def compute(
        cls, x: torch.Tensor, y: torch.Tensor, rows: np.ndarray, chunk: int = 4096
    ) -> GramStats:
        d, t = x.shape[1], y.shape[1]
        s1 = torch.zeros(d, dtype=torch.float64)
        s2 = torch.zeros(d, dtype=torch.float64)
        gram = torch.zeros(d, d, dtype=torch.float64)
        xty = torch.zeros(d, t, dtype=torch.float64)
        ysum = torch.zeros(t, dtype=torch.float64)
        rows_t = torch.as_tensor(np.asarray(rows, np.int64))
        for start in range(0, len(rows_t), chunk):
            index = rows_t[start : start + chunk]
            xb = x[index].double()
            yb = y[index].double()
            s1 += xb.sum(0)
            s2 += (xb * xb).sum(0)
            gram += xb.T @ xb
            xty += xb.T @ yb
            ysum += yb.sum(0)
        return cls(len(rows_t), s1, gram, s2, xty, ysum)

    def __sub__(self, other: GramStats) -> GramStats:
        return GramStats(
            self.n - other.n,
            self.s1 - other.s1,
            self.gram - other.gram,
            self.s2 - other.s2,
            self.xty - other.xty,
            self.ysum - other.ysum,
        )


class RidgeModel:
    """Standardized ridge on precomputed statistics; all alphas share one eigh."""

    def __init__(self, stats: GramStats, *, tol: float = 1e-8) -> None:
        n = stats.n
        if n < 2:
            raise ValueError("ridge needs at least two training rows")
        mean = stats.s1 / n
        sd = (stats.s2 / n - mean**2).clamp_min(0.0).sqrt()
        # Constant columns (padding, unused slots) are dropped, not regularized.
        self.active = torch.nonzero(sd > tol).reshape(-1)
        if not len(self.active):
            self.active = torch.zeros(1, dtype=torch.long)
        a = self.active
        self.mean = mean[a]
        self.inv_sd = 1.0 / sd[a].clamp_min(tol)
        self.y_mean = stats.ysum / n
        centered = stats.gram[a][:, a] - n * torch.outer(self.mean, self.mean)
        gram = self.inv_sd[:, None] * centered * self.inv_sd[None, :]
        gram = 0.5 * (gram + gram.T)
        xty_c = stats.xty[a] - torch.outer(stats.s1[a], self.y_mean)
        xty_s = self.inv_sd[:, None] * xty_c
        try:
            eigenvalues, vectors = torch.linalg.eigh(gram)
        except RuntimeError:  # ill-conditioned (few rows): the PSD SVD gives the same basis
            u, s, _ = torch.linalg.svd(gram)
            eigenvalues, vectors = s.flip(0), u.flip(1)
        self.eigenvalues = eigenvalues.clamp_min(0.0)
        self.vectors = vectors
        self.b = vectors.T @ xty_s
        self.effective_features = len(a)
        self.scale = max(float(self.eigenvalues.sum()) / max(1, self.effective_features), 1e-12)

    def project(
        self, x: torch.Tensor, rows: np.ndarray, chunk: int = 4096, components: int | None = None
    ) -> torch.Tensor:
        vectors = self.vectors if components is None else self.vectors[:, -components:]
        out = []
        rows_t = torch.as_tensor(np.asarray(rows, np.int64))
        for start in range(0, len(rows_t), chunk):
            xb = x[rows_t[start : start + chunk]][:, self.active].double()
            out.append(((xb - self.mean) * self.inv_sd) @ vectors)
        return torch.cat(out) if out else torch.zeros(0, vectors.shape[1], dtype=torch.float64)

    def predict_projected(self, z: torch.Tensor, alpha: float) -> torch.Tensor:
        shrink = 1.0 / (self.eigenvalues + alpha * self.scale)
        return z @ (self.b * shrink[:, None]) + self.y_mean


def knn_predict(
    train_z: torch.Tensor, train_y: torch.Tensor, test_z: torch.Tensor, k: int, chunk: int = 1024
) -> torch.Tensor:
    k = max(1, min(k, len(train_z)))
    out = []
    for start in range(0, len(test_z), chunk):
        distance = torch.cdist(test_z[start : start + chunk].float(), train_z.float())
        index = distance.topk(k, dim=1, largest=False).indices
        out.append(train_y[index].double().mean(dim=1))
    return torch.cat(out) if out else torch.zeros(0, train_y.shape[1], dtype=torch.float64)


def select_alpha(
    x: torch.Tensor, y: torch.Tensor, rows: np.ndarray, units: np.ndarray, folds: int, seed: int
) -> tuple[list[float], dict[str, Any]]:
    """Per-target alpha minimizing held-out MSE over unit (world) folds of ``rows``."""
    fold = group_folds(units[rows], folds, seed)
    total = GramStats.compute(x, y, rows)
    errors = np.zeros((len(RIDGE_ALPHAS), y.shape[1]))
    for k in range(folds):
        test = rows[fold == k]
        if not len(test) or len(test) == len(rows):
            continue
        model = RidgeModel(total - GramStats.compute(x, y, test))
        z = model.project(x, test)
        truth = y[torch.as_tensor(test)].double()
        for a, alpha in enumerate(RIDGE_ALPHAS):
            errors[a] += ((model.predict_projected(z, alpha) - truth) ** 2).sum(0).numpy()
    best = [float(RIDGE_ALPHAS[int(np.argmin(errors[:, t]))]) for t in range(y.shape[1])]
    return best, {"alphas": list(RIDGE_ALPHAS), "chosen": best, "folds": folds}


def cross_validated(
    x: torch.Tensor,
    y: torch.Tensor,
    units: np.ndarray,
    folds: int,
    alphas: Sequence[float],
    knn_k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold ridge and k-NN predictions (symlog targets) with unit-held-out folds."""
    n = len(units)
    rows = np.arange(n)
    fold = group_folds(units, folds, seed)
    total = GramStats.compute(x, y, rows)
    ridge = np.full((n, y.shape[1]), np.nan)
    knn = np.full((n, y.shape[1]), np.nan)
    for k in range(folds):
        test = rows[fold == k]
        train = rows[fold != k]
        if not len(test) or len(train) < 2:
            continue
        model = RidgeModel(total - GramStats.compute(x, y, test))
        z = model.project(x, test)
        for t, alpha in enumerate(alphas):
            ridge[test, t] = model.predict_projected(z, alpha)[:, t].numpy()
        components = min(PCA_COMPONENTS, model.vectors.shape[1])
        train_z = model.project(x, train, components=components)
        knn[test] = knn_predict(
            train_z, y[torch.as_tensor(train)], z[:, -components:], knn_k
        ).numpy()
    return ridge, knn


def transfer(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_cal: torch.Tensor,
    alphas: Sequence[float],
    knn_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on all fitting rows, predict every calibration row."""
    model = RidgeModel(GramStats.compute(x_fit, y_fit, np.arange(len(x_fit))))
    z = model.project(x_cal, np.arange(len(x_cal)))
    ridge = np.stack(
        [model.predict_projected(z, alpha)[:, t].numpy() for t, alpha in enumerate(alphas)], axis=1
    )
    components = min(PCA_COMPONENTS, model.vectors.shape[1])
    train_z = model.project(x_fit, np.arange(len(x_fit)), components=components)
    knn = knn_predict(train_z, y_fit, z[:, -components:], knn_k).numpy()
    return ridge, knn


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------


def _input_tensors(dataset: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    from .labor_pretrain_eval import _input_tensors as tensors

    return {k: v for k, v in tensors(dataset).items() if not k.startswith("search_")}


def load_subset(path: Path, normalizer: Any, max_worlds: int | None) -> dict[str, Any]:
    """Verified dataset load; with ``max_worlds`` only the first N eligible worlds.

    Reserved Q-pi worlds and sealed worlds are never loaded.
    """
    from .labor_pretrain_eval import _check_world_policy
    from .labor_teacher_data import ProvenanceError, checkpoint_hash, verify_shard
    from .labor_training import _normalization_digest

    manifest = json.loads((Path(path) / "dataset.json").read_text())
    worlds = sorted(int(w) for w in manifest["shards"] if int(w) not in RESERVED_WORLDS)
    if max_worlds is not None:
        worlds = worlds[:max_worlds]
    _check_world_policy(worlds, f"dataset {path}")
    expected = _normalization_digest(normalizer)
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for world in worlds:
        entry = manifest["shards"][str(world)]
        shard = Path(path) / entry["path"]
        if checkpoint_hash(shard / "meta.json") != entry["meta_sha256"]:
            raise ProvenanceError(f"shard {shard} meta hash mismatch")
        meta = verify_shard(shard)
        if meta["normalizer_digest"] != expected:
            raise ProvenanceError(
                f"shard {shard} normalizer differs from the checkpoint normalizer"
            )
        for name in meta["payload"]:
            arrays[name[: -len(".npy")]].append(np.load(shard / name, allow_pickle=False))
    dataset: dict[str, Any] = {k: torch.from_numpy(np.concatenate(v)) for k, v in arrays.items()}
    dataset["collection"] = {
        "dataset_digest": manifest["dataset_digest"],
        "worlds_loaded": worlds,
        "worlds_available": len(manifest["shards"]),
    }
    return dataset


def _np(dataset: Mapping[str, Any], key: str) -> np.ndarray:
    return (
        dataset[key].numpy() if isinstance(dataset[key], torch.Tensor) else np.asarray(dataset[key])
    )


def row_frame(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Kept (non-quarantined) rows with targets for every horizon and strata labels."""
    from .labor_bc import CRITIC_DISCOUNT_DAYS
    from .labor_model import LABOR_OBJECTIVE_REFERENCE_SCALES

    keep = ~_np(dataset, "row_quarantined").reshape(-1).astype(bool)
    worlds_all = _np(dataset, "row_world").reshape(-1)
    camps_all = _np(dataset, "row_camp").reshape(-1)
    turns_all = _np(dataset, "row_turn").reshape(-1)
    trajectory = group_ids(worlds_all, camps_all)
    elapsed = _np(dataset, "transition_elapsed_days").reshape(-1)
    terminal = _np(dataset, "transition_terminal").reshape(-1)
    targets: dict[str, dict[str, np.ndarray]] = {}
    remaining = ends_terminal = None
    for head in HEADS:
        reward = _np(dataset, REWARD_KEYS[head]).reshape(-1)
        scale = LABOR_OBJECTIVE_REFERENCE_SCALES[head]
        targets[head] = {}
        for window in (None, *SHORT_HORIZONS):
            values, rem, ends = windowed_returns(
                reward,
                elapsed,
                terminal,
                trajectory,
                turns_all,
                window=window,
                discount=CRITIC_DISCOUNT_DAYS,
            )
            targets[head]["full" if window is None else f"h{window}"] = values[keep] / scale
            if window is None:
                remaining, ends_terminal = rem, ends
    rows = np.flatnonzero(keep)
    worlds, turns = worlds_all[rows], turns_all[rows]
    if "target_specialty_counts" in dataset:
        population = _np(dataset, "target_specialty_counts")[rows].sum(axis=1).astype(np.float64)
    else:
        population = np.full(len(rows), np.nan)
    return {
        "rows": rows,
        "worlds": worlds,
        "turns": turns,
        "camps": camps_all[rows],
        "groups": group_ids(worlds, turns),
        "targets": targets,
        "remaining": remaining[rows],
        "ends_terminal": ends_terminal[rows],
        "population": population,
        "excluded_quarantined": int((~keep).sum()),
    }


def critic_forward(
    task: Any,
    dataset: Mapping[str, Any],
    rows: np.ndarray,
    observed_max: Mapping[str, float],
    chunk: int,
) -> dict[str, np.ndarray]:
    """Decoded normalized predictions, distribution diagnostics and trunk features."""
    from .labor_model import LABOR_OBJECTIVES
    from .model import symexp as torch_symexp

    tensors = _input_tensors(dataset)
    bins = task.bins.double()
    width = float(bins[1] - bins[0])
    values = torch_symexp(bins)
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    started = time.perf_counter()
    with torch.inference_mode():
        for count, start in enumerate(range(0, len(rows), chunk)):
            index = torch.as_tensor(rows[start : start + chunk], dtype=torch.long)
            batch = {k: v[index] for k, v in tensors.items()}
            hidden = task.encoder(batch)[0]
            out["hidden"].append(hidden.float().numpy())
            for name in LABOR_OBJECTIVES:
                logits = task.heads[name](hidden).double()
                p = torch.softmax(logits, dim=-1)
                mean_y = (p * bins).sum(-1)
                out[f"{name}/pred"].append((p * values).sum(-1).numpy())
                out[f"{name}/entropy"].append(entropy(p).numpy())
                out[f"{name}/sd_bins"].append(
                    ((p * (bins - mean_y[:, None]) ** 2).sum(-1).sqrt() / width).numpy()
                )
                upper = values > float(observed_max[name])
                out[f"{name}/upper_mass"].append((p * upper).sum(-1).numpy())
                out[f"{name}/upper_excess"].append(
                    (p * upper * (values - float(observed_max[name]))).sum(-1).numpy()
                )
                out[f"{name}/top_bin_mass"].append(p[:, -1].numpy())
            if count % 10 == 0:
                done = min(len(rows), start + chunk)
                _log(
                    f"critic forward {done}/{len(rows)} rows, {time.perf_counter() - started:.0f}s"
                )
    return {k: np.concatenate(v) for k, v in out.items()}


def input_features(
    dataset: Mapping[str, Any], rows: np.ndarray, chunk: int = 2048
) -> tuple[torch.Tensor, dict[str, int]]:
    """Dense actor-visible features: flattened tensors, with activity tokens pooled.

    The 286 activity tokens are reduced to a masked mean per owning specialty and a
    global masked mean and max; every other input tensor is flattened as-is.
    """
    tensors = _input_tensors(dataset)
    keys = sorted(tensors)
    blocks: dict[str, int] = {}
    parts: list[torch.Tensor] = []
    rows_t = torch.as_tensor(np.asarray(rows, np.int64))
    roles = (
        int(tensors["labor_specialty_features"].shape[1]) - 1
        if "labor_specialty_features" in tensors
        else 0
    )
    for start in range(0, len(rows_t), chunk):
        index = rows_t[start : start + chunk]
        chunk_parts = []
        for key in keys:
            value = tensors[key][index]
            if (
                key == "labor_activity_features"
                and roles > 0
                and "labor_activity_present" in tensors
            ):
                present = tensors["labor_activity_present"][index].float()
                n, tokens, width = value.shape
                v = value.float() * present[..., None]
                per_role = v.view(n, roles, tokens // roles, width).sum(2)
                per_role = per_role / present.view(n, roles, -1).sum(2).clamp_min(1.0)[..., None]
                mean = v.sum(1) / present.sum(1).clamp_min(1.0)[:, None]
                peak = value.float().masked_fill(present[..., None] == 0, -1e9).amax(1)
                peak = torch.where(peak < -1e8, torch.zeros_like(peak), peak)
                flat = torch.cat([per_role.reshape(n, -1), mean, peak], dim=1)
            else:
                flat = value.reshape(len(index), -1).float()
            if start == 0:
                blocks[key] = int(flat.shape[1])
            chunk_parts.append(flat)
        parts.append(torch.cat(chunk_parts, dim=1))
    return torch.cat(parts), blocks


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _strata(frame: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "remaining": remaining_labels(frame["remaining"]),
        "camp_size": population_labels(frame["population"]),
    }


def target_structure(frame: Mapping[str, Any], bin_width: float) -> dict[str, Any]:
    strata = _strata(frame)
    report: dict[str, Any] = {
        "bin_width_symlog": bin_width,
        "rows": len(frame["rows"]),
        "groups": len(np.unique(frame["groups"])),
        "mean_rows_per_group": float(np.bincount(frame["groups"]).mean())
        if len(frame["groups"])
        else None,
        "remaining_distribution": {
            name: int((strata["remaining"] == name).sum()) for name, _, _ in REMAINING_BINS
        },
        "segments_ending_terminal_fraction": float(frame["ends_terminal"].mean())
        if len(frame["rows"])
        else None,
    }
    pairs = PairIndex(frame["groups"], frame["worlds"])
    for head in HEADS:
        normalized = frame["targets"][head]["full"]
        head_report: dict[str, Any] = {
            "normalized": variance_decomposition(normalized, frame["groups"]),
            "symlog": variance_decomposition(symlog(normalized), frame["groups"]),
            "target_quantiles_reference_scales": quantiles(normalized, (0.0, 0.1, 0.5, 0.9, 1.0)),
            "pairs": informative_pair_structure(normalized, pairs, bin_width=bin_width),
        }
        head_report["within_sd_bins"] = head_report["symlog"]["within_sd"] / bin_width
        for stratum, labels in strata.items():
            split = {}
            for label in sorted({str(v) for v in labels.tolist()}):
                mask = labels.astype(str) == label
                sub_pairs = PairIndex(frame["groups"], frame["worlds"], mask)
                split[label] = {
                    "rows": int(mask.sum()),
                    "within_fraction": variance_decomposition(
                        normalized[mask], frame["groups"][mask]
                    )["within_fraction"],
                    "target_mean": float(normalized[mask].mean()) if mask.any() else None,
                    "pairs": informative_pair_structure(normalized, sub_pairs, bin_width=bin_width),
                }
            head_report[f"by_{stratum}"] = split
        horizons = {}
        for key, values in frame["targets"][head].items():
            horizons[key] = {
                "normalized": variance_decomposition(values, frame["groups"]),
                "informative_fraction": informative_pair_structure(
                    values, pairs, bin_width=bin_width
                )["informative_fraction"],
                "within_corr_with_full": group_spread(values, normalized, frame["groups"])[
                    "mean_within_correlation"
                ],
            }
        head_report["by_horizon"] = horizons
        report[head] = head_report
    return report


def prediction_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    frame: Mapping[str, Any],
    pairs: PairIndex,
    bin_width: float,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    out = error_metrics(pred, target, frame["worlds"])
    out["r2"] = r2(pred, target)
    out["within_group_r2"] = within_group_r2(pred, target, frame["groups"])
    out["within_group_r2_symlog"] = within_group_r2(symlog(pred), symlog(target), frame["groups"])
    out["ranking"] = pair_ranking(
        pred, target, pairs, INFORMATIVE_FRACTION, bootstrap=bootstrap, seed=seed
    )
    y = symlog(target)
    out["ranking_by_bins"] = pair_ranking(
        pred,
        target,
        pairs,
        INFORMATIVE_FRACTION,
        margin_edges=BIN_MARGIN_EDGES,
        margin_values=np.abs(y[pairs.i] - y[pairs.j]) / bin_width,
    )["by_margin"]
    out["spread"] = group_spread(pred, target, frame["groups"])
    return out


def critic_section(
    frame: Mapping[str, Any],
    preds: Mapping[str, np.ndarray],
    bin_width: float,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    pairs = PairIndex(frame["groups"], frame["worlds"])
    strata = _strata(frame)
    report: dict[str, Any] = {}
    for head in HEADS:
        target = frame["targets"][head]["full"]
        pred = preds[f"{head}/pred"]
        weights = world_balanced(frame["worlds"])
        head_report = prediction_metrics(
            pred, target, frame, pairs, bin_width, bootstrap=bootstrap, seed=seed
        )
        head_report["distribution"] = {
            "mean_entropy_nats": float(preds[f"{head}/entropy"].mean()),
            "mean_effective_bins": float(np.exp(preds[f"{head}/entropy"]).mean()),
            "mean_sd_bins": float(preds[f"{head}/sd_bins"].mean()),
            "upper_mass": float((preds[f"{head}/upper_mass"] * weights).sum() / weights.sum()),
            "upper_excess_reference_scales": float(
                (preds[f"{head}/upper_excess"] * weights).sum() / weights.sum()
            ),
            "upper_excess_max": float(preds[f"{head}/upper_excess"].max()),
            "top_bin_mass": float(preds[f"{head}/top_bin_mass"].mean()),
            "observed_max_reference_scales": float(target.max()),
        }
        labels_by = dict(strata)
        labels_by["target_magnitude"], edges = magnitude_labels(target)
        for stratum, labels in labels_by.items():
            split = {}
            for label in sorted({str(v) for v in labels.tolist()}):
                mask = labels.astype(str) == label
                sub_pairs = PairIndex(frame["groups"], frame["worlds"], mask)
                entry = error_metrics(pred[mask], target[mask], frame["worlds"][mask])
                ranking = pair_ranking(pred, target, sub_pairs, INFORMATIVE_FRACTION)
                entry["ranking"] = ranking["accuracy"]
                entry["informative_pairs"] = ranking["informative_pairs"]
                entry["within_group_r2"] = within_group_r2(
                    pred[mask], target[mask], frame["groups"][mask]
                )
                split[label] = entry
            head_report[f"by_{stratum}"] = split
        head_report["target_magnitude_edges"] = edges
        head_report["within_corr_by_horizon"] = {
            key: group_spread(pred, values, frame["groups"])["mean_within_correlation"]
            for key, values in frame["targets"][head].items()
        }
        report[head] = head_report
    return report


def _target_matrix(frame: Mapping[str, Any]) -> tuple[torch.Tensor, list[tuple[str, str]]]:
    columns = [(head, key) for head in HEADS for key in frame["targets"][head]]
    y = np.stack([symlog(frame["targets"][h][k]) for h, k in columns], axis=1)
    return torch.from_numpy(y), columns


def noise_floor(
    cal: Mapping[str, Any],
    fit: Mapping[str, Any] | None,
    x_cal: torch.Tensor,
    x_fit: torch.Tensor | None,
    h_cal: torch.Tensor,
    h_fit: torch.Tensor | None,
    critic_preds: Mapping[str, np.ndarray],
    bin_width: float,
    *,
    folds: int,
    knn_k: int,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    y_cal, columns = _target_matrix(cal)
    units = cal["worlds"]
    fold_unit = "world"
    if len(np.unique(units)) < 3:
        units = group_ids(cal["worlds"], cal["camps"])
        fold_unit = "trajectory (fewer than 3 worlds: smoke only, optimistic)"
    folds = max(2, min(folds, len(np.unique(units))))
    pairs = PairIndex(cal["groups"], cal["worlds"])
    report: dict[str, Any] = {
        "fold_unit": fold_unit,
        "folds": folds,
        "features": int(x_cal.shape[1]),
        "hidden_features": int(h_cal.shape[1]),
        "knn_k": knn_k,
        "pca_components": PCA_COMPONENTS,
        "targets": "symlog(G / reference scale); predictions decoded with symexp",
        "group_mean_loo_nmae": {},
        "models": {},
    }
    predictions: dict[str, np.ndarray] = {}
    for name, x, xf in (("inputs", x_cal, x_fit), ("critic_trunk", h_cal, h_fit)):
        y_fit = None
        if fit is not None and xf is not None:
            y_fit, _ = _target_matrix(fit)
            fit_units = (
                fit["worlds"]
                if len(np.unique(fit["worlds"])) >= 3
                else group_ids(fit["worlds"], fit["camps"])
            )
            alphas, selection = select_alpha(
                xf,
                y_fit,
                np.arange(len(xf)),
                fit_units,
                max(2, min(3, len(np.unique(fit_units)))),
                seed,
            )
            selection["selected_on"] = "fitting dataset (world folds)"
        else:
            alphas, selection = select_alpha(x, y_cal, np.arange(len(x)), units, folds, seed)
            selection["selected_on"] = "calibration dataset (optimistic: no fitting dataset given)"
        report[f"alpha_{name}"] = selection
        _log(f"noise floor: {name} cross-validation ({x.shape[1]} features)")
        ridge, knn = cross_validated(x, y_cal, units, folds, alphas, knn_k, seed)
        predictions[f"ridge_{name}/cal-cv"] = ridge
        if name == "inputs":
            predictions[f"knn_{name}/cal-cv"] = knn
        if y_fit is not None:
            _log(f"noise floor: {name} fit -> calibration transfer")
            ridge_t, knn_t = transfer(xf, y_fit, x, alphas, knn_k)
            predictions[f"ridge_{name}/fit->cal"] = ridge_t
            if name == "inputs":
                predictions[f"knn_{name}/fit->cal"] = knn_t
    for t, (head, key) in enumerate(columns):
        target = cal["targets"][head][key]
        counts = np.bincount(cal["groups"])
        sums = np.bincount(cal["groups"], weights=target)
        multi = counts[cal["groups"]] >= 2
        loo = (sums[cal["groups"]] - target) / np.maximum(counts[cal["groups"]] - 1, 1)
        report["group_mean_loo_nmae"][f"{head}/{key}"] = error_metrics(
            loo[multi], target[multi], cal["worlds"][multi]
        )["nmae"]
        entries = {}
        if key == "full":
            entries["critic"] = prediction_metrics(
                critic_preds[f"{head}/pred"],
                target,
                cal,
                pairs,
                bin_width,
                bootstrap=bootstrap,
                seed=seed,
            )
        for model, values in predictions.items():
            column = values[:, t]
            valid = np.isfinite(column)
            if not valid.all():
                entries[model] = {"rows_missing": int((~valid).sum())}
                continue
            entries[model] = prediction_metrics(
                symexp(column), target, cal, pairs, bin_width, bootstrap=bootstrap, seed=seed
            )
        report["models"][f"{head}/{key}"] = entries
    return report


# ---------------------------------------------------------------------------
# Search candidate disparity
# ---------------------------------------------------------------------------


def _search_config(path: Path) -> tuple[dict[str, Any], str]:
    from .labor_search import SearchConfig

    for parent in (path, *path.parents):
        manifest = parent / "dataset.json"
        if manifest.is_file():
            try:
                options = json.loads(manifest.read_text())["source_record"]["options"]
                if "search" in options:
                    return SearchConfig.from_mapping(options["search"]).state_dict(), str(manifest)
            except (KeyError, ValueError, TypeError):
                continue
    return (
        SearchConfig().state_dict(),
        "SearchConfig defaults (no dataset.json with a search record found)",
    )


def candidate_horizon_returns(
    steps: Sequence[Sequence[float]], horizon: int, gamma: float
) -> np.ndarray:
    """Discounted component returns over the first ``horizon`` stored steps."""
    total = np.zeros(4, np.float64)
    factor = 1.0
    for step in list(steps)[:horizon]:
        total += factor * np.asarray(step, np.float64)
        factor *= gamma
    return total


def _utility(returns: np.ndarray, vector: Sequence[float], config: Mapping[str, Any]) -> np.ndarray:
    from .labor_model import LABOR_COST_REFERENCE_SCALES, LABOR_OBJECTIVE_REFERENCE_SCALES

    r = np.asarray(returns, np.float64).reshape(-1, 4)
    return (
        float(vector[0]) * r[:, 0] / LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"]
        + float(vector[1]) * r[:, 1] / LABOR_OBJECTIVE_REFERENCE_SCALES["population_years"]
        - float(config["feasibility_multiplier"])
        * r[:, 2]
        / LABOR_COST_REFERENCE_SCALES["feasibility"]
        - float(config["guard_multiplier"])
        * r[:, 3]
        / LABOR_COST_REFERENCE_SCALES["settlement_guard"]
    )


def _components(
    returns: np.ndarray, record: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    from .labor_model import LABOR_OBJECTIVE_REFERENCE_SCALES

    r = np.asarray(returns, np.float64).reshape(-1, 4)
    out = {
        "qol_years": r[:, 0] / LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"],
        "population_years": r[:, 1] / LABOR_OBJECTIVE_REFERENCE_SCALES["population_years"],
    }
    for entry in record.get("personalities") or [{"name": "balanced", "vector": [0.5, 0.5]}]:
        out[f"utility:{entry['name']}"] = _utility(r, entry["vector"], config)
    return out


def root_disparity(
    record: Mapping[str, Any], config: Mapping[str, Any], gamma: float
) -> dict[str, Any]:
    """Per-root candidate spread against the paired re-evaluation noise."""
    candidates = record["candidates"]
    final_h = int(record["final_horizon"])
    horizons = [h for h in config["horizons"] if h <= final_h] or [final_h]
    mismatch = 0.0
    for c in candidates:
        recon = candidate_horizon_returns(c["step_rewards"], len(c["step_rewards"]), gamma)
        stored = np.asarray(
            [
                c["returns"][k]
                for k in ("qol_years", "population_years", "feasibility", "settlement_guard")
            ]
        )
        mismatch = max(
            mismatch, float(np.max(np.abs(recon - stored) / np.maximum(1.0, np.abs(stored))))
        )
    # Paired re-evaluation streams at the final horizon (H12 = candidate 0 is the reference).
    reevaluated = {
        int(c["index"]): c["reeval_returns"] for c in candidates if c.get("reeval_returns")
    }
    noise: dict[str, Any] = {"reevaluated_candidates": sorted(reevaluated)}
    exact_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)  # (mean diff, se)
    pair_sd: dict[str, list[float]] = defaultdict(list)
    ids = sorted(reevaluated)
    for a_pos, a in enumerate(ids):
        for b in ids[a_pos + 1 :]:
            streams = sorted(set(reevaluated[a]) & set(reevaluated[b]))
            if len(streams) < 2:
                continue
            ra = np.asarray([reevaluated[a][s] for s in streams], np.float64)
            rb = np.asarray([reevaluated[b][s] for s in streams], np.float64)
            ca, cb = _components(ra, record, config), _components(rb, record, config)
            for name in ca:
                d = ca[name] - cb[name]
                sd = float(d.std(ddof=1))
                exact_pairs[name].append((float(d.mean()), sd / math.sqrt(len(streams))))
                pair_sd[name].append(sd)
    sigma = {name: math.sqrt(float(np.mean(np.square(v)))) for name, v in pair_sd.items()}
    noise["paired_stream_sd"] = sigma
    noise["exact_pairs"] = {name: len(v) for name, v in exact_pairs.items()}
    per_h: dict[str, Any] = {}
    for h in horizons:
        present = [c for c in candidates if len(c["step_rewards"]) >= h or not c.get("alive", True)]
        if len(present) < 2:
            continue
        returns = np.stack(
            [candidate_horizon_returns(c["step_rewards"], h, gamma) for c in present]
        )
        comps = _components(returns, record, config)
        entry: dict[str, Any] = {"candidates": len(present)}
        for name, values in comps.items():
            diff = np.abs(values[:, None] - values[None, :])[np.triu_indices(len(values), 1)]
            entry[name] = {
                "range": float(values.max() - values.min()),
                "sd": float(values.std()),
                "max_pair_diff": float(diff.max()),
                "diffs": diff,
            }
        per_h[str(h)] = entry
    # QoL against population-years gains over H12 at the final horizon (single rollout).
    final = [c for c in candidates if len(c["step_rewards"]) >= final_h or not c.get("alive", True)]
    h12 = next((c for c in candidates if int(c["index"]) == 0), None)
    gains = None
    if h12 is not None and len(final) >= 2:
        returns = np.stack(
            [candidate_horizon_returns(c["step_rewards"], final_h, gamma) for c in final]
        )
        comps = _components(returns, record, config)
        reference = [i for i, c in enumerate(final) if int(c["index"]) == 0]
        if reference:
            r0 = reference[0]
            others = [i for i in range(len(final)) if i != r0]
            gains = {
                "qol": (comps["qol_years"][others] - comps["qol_years"][r0]).tolist(),
                "pop": (comps["population_years"][others] - comps["population_years"][r0]).tolist(),
                "argmax_differs": bool(
                    int(np.argmax(comps["qol_years"])) != int(np.argmax(comps["population_years"]))
                ),
            }
    return {
        "world": record.get("world"),
        "turn": record.get("turn"),
        "final_horizon": final_h,
        "reconstruction_max_rel_error": mismatch,
        "noise": noise,
        "exact_pairs": exact_pairs,
        "horizons": per_h,
        "gains": gains,
    }


def _iter_candidate_tables(search_dir: Path) -> Iterable[tuple[Path, Mapping[str, Any]]]:
    for path in sorted(Path(search_dir).rglob("candidates.jsonl")):
        if any(part.endswith(".staging") for part in path.parts):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield path, json.loads(line)


def search_disparity(
    search_dir: Path, materiality: Sequence[float] = (0.0, 0.01, 0.05)
) -> dict[str, Any]:
    from . import contracts

    config, config_source = _search_config(Path(search_dir))
    gamma = math.exp(-contracts.TURN_YEARS / float(config["discount_years"]))
    roots = []
    tables = set()
    for path, record in _iter_candidate_tables(Path(search_dir)):
        tables.add(str(path.parent))
        roots.append(root_disparity(record, config, gamma))
    report: dict[str, Any] = {
        "search_dir": str(search_dir),
        "config_source": config_source,
        "shards": len(tables),
        "roots": len(roots),
        "per_step_discount": gamma,
        "max_return_reconstruction_rel_error": max(
            (r["reconstruction_max_rel_error"] for r in roots), default=None
        ),
        "materiality_reference_scales": list(materiality),
        "notes": [
            "Successor tensors are not stored, so the V-based (state-value) advantage of each candidate "
            "cannot be computed from this export; only native returns are compared.",
            "Re-evaluation (dithered H12 continuation streams) exists only at the final horizon and only for "
            "H12 and the personality winners beyond the selection margin.",
            "Single-rollout candidate returns come from the deterministic stream 0; the pairwise noise used for "
            "them is the root's paired re-evaluation stream SD at the final horizon (an upper-bound proxy for "
            "shorter horizons when noise grows with horizon).",
        ],
    }
    if not roots:
        report["status"] = "no candidates.jsonl found"
        return report
    names = sorted(
        {name for r in roots for name in r["noise"]["paired_stream_sd"]}
        | {
            name
            for r in roots
            for entry in r["horizons"].values()
            for name in entry
            if name != "candidates"
        }
        | {"qol_years", "population_years"}
    )
    components: dict[str, Any] = {}
    for name in names:
        comp: dict[str, Any] = {}
        # Exact: re-evaluated pairs at the final horizon, mean paired difference beyond 2 SE.
        with_pairs = [r for r in roots if r["exact_pairs"].get(name)]
        zero_se = [r for r in with_pairs if all(se == 0.0 for _, se in r["exact_pairs"][name])]
        exact = {}
        for m in materiality:
            hits = [
                any(abs(d) > 2.0 * se and abs(d) > m for d, se in r["exact_pairs"][name])
                for r in with_pairs
            ]
            exact[f"{m:g}"] = float(np.mean(hits)) if hits else None
        ratios = [
            abs(d) / (2.0 * se) for r in with_pairs for d, se in r["exact_pairs"][name] if se > 0
        ]
        comp["reevaluated"] = {
            "roots_with_pairs": len(with_pairs),
            "roots_with_zero_se": len(zero_se),
            "share_roots_any_pair_beyond_2se": exact,
            "abs_diff_over_2se": quantiles(np.asarray(ratios)),
            "pair_se": quantiles(
                np.asarray([se for r in with_pairs for _, se in r["exact_pairs"][name]])
            ),
        }
        # Proxy: every candidate pair at each horizon against 2x the paired single-stream SD.
        by_h = {}
        for h in sorted({h for r in roots for h in r["horizons"]}, key=int):
            usable = [r for r in roots if h in r["horizons"] and name in r["horizons"][h]]
            with_sigma = [r for r in usable if name in r["noise"]["paired_stream_sd"]]
            spread = np.asarray([r["horizons"][h][name]["range"] for r in usable])
            sigma = np.asarray([r["noise"]["paired_stream_sd"][name] for r in with_sigma])
            shares = {}
            for m in materiality:
                hits = [
                    bool(
                        np.any(
                            (
                                r["horizons"][h][name]["diffs"]
                                > 2.0 * r["noise"]["paired_stream_sd"][name]
                            )
                            & (r["horizons"][h][name]["diffs"] > m)
                        )
                    )
                    for r in with_sigma
                ]
                shares[f"{m:g}"] = float(np.mean(hits)) if hits else None
            material_only = {
                f"{m:g}": float(
                    np.mean([r["horizons"][h][name]["max_pair_diff"] > m for r in usable])
                )
                if usable
                else None
                for m in materiality
            }
            by_h[h] = {
                "roots": len(usable),
                "roots_with_noise_estimate": len(with_sigma),
                "candidate_range": quantiles(spread),
                "paired_stream_sd": quantiles(sigma),
                "range_over_2sd": quantiles(
                    np.asarray(
                        [
                            r["horizons"][h][name]["range"]
                            / (2.0 * r["noise"]["paired_stream_sd"][name])
                            for r in with_sigma
                            if r["noise"]["paired_stream_sd"][name] > 0
                        ]
                    )
                ),
                "share_roots_any_pair_beyond_2sd": shares,
                "share_roots_any_pair_beyond_materiality": material_only,
            }
        comp["by_horizon"] = by_h
        components[name] = comp
    report["components"] = components
    q = [g for r in roots if r["gains"] for g in r["gains"]["qol"]]
    p = [g for r in roots if r["gains"] for g in r["gains"]["pop"]]
    per_root = [
        spearman(np.asarray(r["gains"]["qol"]), np.asarray(r["gains"]["pop"]))
        for r in roots
        if r["gains"] and len(r["gains"]["qol"]) >= 3
    ]
    per_root = [v for v in per_root if v is not None]
    q_a, p_a = np.asarray(q), np.asarray(p)
    both = (q_a != 0) & (p_a != 0)
    report["qol_vs_population_gains"] = {
        "candidates": len(q),
        "pearson": pearson(q_a, p_a),
        "spearman": spearman(q_a, p_a),
        "median_per_root_spearman": float(np.median(per_root)) if per_root else None,
        "opposite_sign_fraction": float((np.sign(q_a[both]) != np.sign(p_a[both])).mean())
        if both.any()
        else None,
        "share_roots_argmax_differs": float(
            np.mean([r["gains"]["argmax_differs"] for r in roots if r["gains"]])
        )
        if any(r["gains"] for r in roots)
        else None,
    }
    return report


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def _get(mapping: Any, *path: Any) -> Any:
    for key in path:
        if not isinstance(mapping, Mapping) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


def _best(models: Mapping[str, Any], prefix: str) -> tuple[str | None, float | None]:
    best_name, best = None, None
    for name, entry in models.items():
        if not name.startswith(prefix):
            continue
        value = _get(entry, "ranking", "accuracy")
        if value is not None and (best is None or value > best):
            best_name, best = name, value
    return best_name, best


def verdict(report: Mapping[str, Any]) -> dict[str, Any]:
    t = VERDICT_THRESHOLDS
    out: dict[str, Any] = {"thresholds": t, "hypotheses": {}}
    models = _get(report, "noise_floor", "models") or {}
    critic_rank = _get(report, "critic", "calibration", "qol_years", "ranking", "accuracy")
    pop_rank = _get(report, "critic", "calibration", "population_years", "ranking", "accuracy")
    fit_rank = _get(report, "critic", "fitting", "qol_years", "ranking", "accuracy")
    full = models.get("qol_years/full", {})
    _, input_rank = _best(full, "ridge_inputs") if full else (None, None)
    _, knn_rank = _best(full, "knn_inputs") if full else (None, None)
    _, trunk_rank = _best(full, "ridge_critic_trunk") if full else (None, None)
    trivial = max([v for v in (input_rank, knn_rank) if v is not None], default=None)
    within_r2 = [
        _get(entry, "within_group_r2")
        for entry in full.values()
        if isinstance(entry, Mapping) and _get(entry, "within_group_r2") is not None
    ]
    short = {
        key: _best(models.get(f"qol_years/{key}", {}), "ridge_inputs")[1]
        for key in ("h1", "h3", "h5")
    }
    short_best = max([v for v in short.values() if v is not None], default=None)
    remaining = _get(report, "critic", "calibration", "qol_years", "by_remaining") or {}
    many = [_get(remaining, k, "ranking") for k in ("13+", "7-12")]
    few = [_get(remaining, k, "ranking") for k in ("1", "2-3", "4-6")]
    many = [v for v in many if v is not None]
    few = [v for v in few if v is not None]
    below = _get(
        report,
        "target_structure",
        "qol_years",
        "pairs",
        "fraction_below_bins",
        f"{t['small_margin_bins']:g}",
    )
    margins = _get(report, "critic", "calibration", "qol_years", "ranking", "by_margin") or []
    large = [m["accuracy"] for m in margins if m["accuracy"] is not None and m["low"] >= 0.5]
    small = [m["accuracy"] for m in margins if m["accuracy"] is not None and m["low"] < 0.1]
    collapse = _get(report, "critic", "calibration", "qol_years", "spread", "pooled_sd_ratio")

    def hyp(name: str, supported: bool | None, evidence: Mapping[str, Any], meaning: str) -> None:
        out["hypotheses"][name] = {
            "status": "inconclusive"
            if supported is None
            else ("supported" if supported else "not supported"),
            "evidence": dict(evidence),
            "meaning": meaning,
        }

    noise_dominated = bool(within_r2) and max(within_r2) <= t["within_r2_noise"]
    horizon_gain = (
        None
        if short_best is None or trivial is None
        else short_best - trivial >= t["short_horizon_gain"]
    )
    tail_gain = (
        None
        if not many or not few
        else float(np.mean(many)) - float(np.mean(few)) >= t["remaining_gain"]
    )
    resolution = None if below is None else below >= t["small_margin_share"]
    margin_shape = (
        None
        if not large or not small
        else (max(large) >= t["rank_pass"] and max(small) < t["rank_weak"])
    )
    flags = [v for v in (noise_dominated, horizon_gain, tail_gain, margin_shape) if v is not None]
    hyp(
        "i_noise_or_resolution",
        None if not flags else bool(any(flags)),
        {
            "max_within_group_r2_any_model": max(within_r2) if within_r2 else None,
            "short_horizon_ridge_ranking": short,
            "full_window_trivial_ranking": trivial,
            "ranking_many_remaining": float(np.mean(many)) if many else None,
            "ranking_few_remaining": float(np.mean(few)) if few else None,
            "share_informative_pairs_below_bins": below,
            "critic_rank_small_margins": small,
            "critic_rank_large_margins": large,
            "resolution_share_flag": resolution,
            "flags": {
                "noise_dominated": noise_dominated,
                "short_horizon_better": horizon_gain,
                "tail_truncation": tail_gain,
                "fails_only_at_small_margins": margin_shape,
            },
        },
        "Supported when every regressor has within-group R2 <= within_r2_noise, or short-horizon targets rank "
        ">= short_horizon_gain better than the full window, or rows far from the window end rank >= remaining_gain "
        "better than late rows, or ranking passes at large margins but not at small ones. Points to more bins or a "
        "shorter-horizon / randomized-tail target. The bin-share is reported but is not decisive alone: the decoded "
        "expectation is continuous.",
    )
    input_limited = None
    if trivial is not None and critic_rank is not None:
        input_limited = (
            critic_rank <= t["rank_weak"]
            and trivial <= t["rank_weak"]
            and (short_best is None or short_best <= t["rank_weak"])
            and (pop_rank is None or pop_rank >= t["rank_pass"])
        )
    hyp(
        "ii_input_information",
        input_limited,
        {
            "critic": critic_rank,
            "best_trivial_inputs": trivial,
            "short_horizon_best": short_best,
            "population_head": pop_rank,
        },
        "Supported when the critic, the trivial input regressors and the short-horizon regressors all rank <= "
        "rank_weak while the population head (same inputs) passes. Points to intervention A (privileged state); "
        "it cannot be separated from intrinsic outcome noise without privileged features.",
    )
    best_alt = max([v for v in (trivial, trunk_rank) if v is not None], default=None)
    under = None
    if critic_rank is not None and best_alt is not None:
        under = best_alt >= critic_rank + t["rank_gain"] or (
            fit_rank is not None
            and fit_rank < t["rank_pass"]
            and best_alt >= critic_rank + t["rank_gain"] / 2
        )
    hyp(
        "iii_critic_optimisation",
        under,
        {
            "critic": critic_rank,
            "critic_on_fitting": fit_rank,
            "best_trivial_inputs": trivial,
            "trunk_probe": trunk_rank,
            "pooled_sd_ratio": collapse,
            "fit_minus_cal": None
            if fit_rank is None or critic_rank is None
            else fit_rank - critic_rank,
        },
        "Supported when a trivial regressor or a linear probe on the critic's own trunk ranks >= rank_gain above "
        "the critic head (the information is there but the head does not read it out), or by half that margin "
        "when the critic also fails on its own fitting worlds. A fit-cal gap above fit_gap is a generalisation "
        "problem, reported separately.",
    )
    out["generalisation_gap"] = (
        None if fit_rank is None or critic_rank is None else fit_rank - critic_rank >= t["fit_gap"]
    )
    search = report.get("search")
    action = None
    evidence: dict[str, Any] = {}
    if search and search.get("components"):
        m = f"{t['search_materiality']:g}"
        qol = search["components"].get("qol_years", {})
        share_exact = _get(qol, "reevaluated", "share_roots_any_pair_beyond_2se", m)
        horizons = _get(qol, "by_horizon") or {}
        final = horizons[max(horizons, key=int)] if horizons else {}
        share_proxy = _get(final, "share_roots_any_pair_beyond_2sd", m)
        corr = _get(search, "qol_vs_population_gains", "spearman")
        evidence = {
            "share_exact_reevaluated": share_exact,
            "share_proxy_final_horizon": share_proxy,
            "qol_pop_gain_spearman": corr,
            "materiality": t["search_materiality"],
        }
        shares = [v for v in (share_exact, share_proxy) if v is not None]
        if shares:
            action = max(shares) >= t["search_root_share"]
        evidence["qol_redundant_with_population"] = (
            None if corr is None else corr >= t["qol_pop_corr_redundant"]
        )
    hyp(
        "iv_action_level_disparity",
        action,
        evidence,
        "Supported when at least search_root_share of roots have a candidate pair whose QoL return differs beyond "
        "2 SE and beyond search_materiality reference scales. This is escalation-gate evidence that real QoL "
        "disparity between actions exists. Whether existing advantage signals capture it cannot be measured "
        "here (no successor tensors). A high QoL/population gain correlation means a population advantage "
        "already carries most of it.",
    )
    statuses = {k: v["status"] for k, v in out["hypotheses"].items()}
    path = []
    if statuses["iii_critic_optimisation"] == "supported":
        path.append(
            "fix critic optimisation / readout first (neither (a) nor (b) is interpretable yet)"
        )
    if statuses["i_noise_or_resolution"] == "supported":
        path.append(
            "(b) value representation / target: more bins or shorter-horizon / randomized-tail targets"
        )
    if statuses["ii_input_information"] == "supported":
        path.append("intervention A (privileged critic state)")
    if statuses["iv_action_level_disparity"] == "supported":
        path.append("(a) escalation-gate evidence: real action-level QoL disparity exists")
    out["recommended_paths"] = path or [
        "no hypothesis cleared its threshold; collect more evidence"
    ]
    out["caveats"] = [
        "Within-(world, turn) ranking compares camps, not actions; action-level evidence is the search section only.",
        "Noise and missing input information are not separable from observational data alone; the privileged "
        "re-collection (intervention A) is the decisive test of (ii).",
        "Ridge and k-NN are lower bounds on input information; a nonlinear model could extract more.",
        "Bootstrap intervals resample worlds; with 20 calibration worlds they are wide.",
    ]
    return out


# ---------------------------------------------------------------------------
# Driver and markdown
# ---------------------------------------------------------------------------


def _strip(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _strip(v) for k, v in value.items() if k != "diffs"}
    if isinstance(value, (list, tuple)):
        return [_strip(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_diagnosis(
    task: Any,
    cal_dataset: Mapping[str, Any],
    fit_dataset: Mapping[str, Any] | None,
    search_dir: Path | None,
    *,
    chunk: int,
    folds: int,
    knn_k: int,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    support = task.support
    bin_width = (support.high - support.low) / (support.bins - 1)
    timings: dict[str, float] = {}
    cal = row_frame(cal_dataset)
    fit = row_frame(fit_dataset) if fit_dataset is not None else None
    _log(
        f"calibration rows {len(cal['rows'])}, fitting rows {0 if fit is None else len(fit['rows'])}"
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "value_support": {
            "low": support.low,
            "high": support.high,
            "bins": support.bins,
            "bin_width_symlog": bin_width,
        },
        "informative_threshold_reference_scales": INFORMATIVE_FRACTION,
        "calibration_worlds": sorted(int(w) for w in np.unique(cal["worlds"])),
        "fitting_worlds": None if fit is None else sorted(int(w) for w in np.unique(fit["worlds"])),
        "excluded_quarantined_rows": {
            "calibration": cal["excluded_quarantined"],
            "fitting": None if fit is None else fit["excluded_quarantined"],
        },
    }
    report["target_structure"] = target_structure(cal, bin_width)
    if fit is not None:
        report["target_structure_fitting"] = {
            head: {
                "normalized": variance_decomposition(fit["targets"][head]["full"], fit["groups"])
            }
            for head in HEADS
        }
    timings["targets"] = time.perf_counter() - started
    observed = {head: float(cal["targets"][head]["full"].max()) for head in HEADS}
    t0 = time.perf_counter()
    cal_preds = critic_forward(task, cal_dataset, cal["rows"], observed, chunk)
    fit_preds = None
    if fit is not None:
        fit_observed = {head: float(fit["targets"][head]["full"].max()) for head in HEADS}
        fit_preds = critic_forward(task, fit_dataset, fit["rows"], fit_observed, chunk)
    timings["critic_forward"] = time.perf_counter() - t0
    report["critic"] = {
        "calibration": critic_section(cal, cal_preds, bin_width, bootstrap=bootstrap, seed=seed)
    }
    if fit is not None:
        report["critic"]["fitting"] = critic_section(
            fit, fit_preds, bin_width, bootstrap=bootstrap, seed=seed
        )
    t0 = time.perf_counter()
    x_cal, blocks = input_features(cal_dataset, cal["rows"])
    x_fit = input_features(fit_dataset, fit["rows"])[0] if fit is not None else None
    timings["features"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    h_cal = torch.from_numpy(cal_preds["hidden"])
    h_fit = torch.from_numpy(fit_preds["hidden"]) if fit_preds is not None else None
    report["noise_floor"] = noise_floor(
        cal,
        fit,
        x_cal,
        x_fit,
        h_cal,
        h_fit,
        cal_preds,
        bin_width,
        folds=folds,
        knn_k=knn_k,
        bootstrap=bootstrap,
        seed=seed,
    )
    report["noise_floor"]["feature_blocks"] = blocks
    timings["noise_floor"] = time.perf_counter() - t0
    if search_dir is not None:
        t0 = time.perf_counter()
        report["search"] = search_disparity(search_dir)
        timings["search"] = time.perf_counter() - t0
    report["verdict"] = verdict(report)
    timings["total"] = time.perf_counter() - started
    report["timings_seconds"] = timings
    return _strip(report)


def _f(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(report: Mapping[str, Any]) -> str:
    lines = [
        "# QoL critic diagnosis",
        "",
        f"Support [{report['value_support']['low']:g}, {report['value_support']['high']:g}] x "
        f"{report['value_support']['bins']} (bin {report['value_support']['bin_width_symlog']:.4f} symlog). "
        f"Calibration worlds {len(report['calibration_worlds'])}; fitting worlds "
        f"{len(report['fitting_worlds'] or [])}.",
        "",
        "## 1. Target structure (calibration)",
        "",
        "| head | var total | within frac | within sd (bins) | informative pairs | median abs dG (ref) | median abs dG (bins) | < 1 / 2 / 5 bins |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ts = report["target_structure"]
    for head in HEADS:
        h = ts[head]
        p = h["pairs"]
        below = p["fraction_below_bins"]
        lines.append(
            f"| {head} | {_f(h['normalized']['total'])} | {_f(h['normalized']['within_fraction'])} | "
            f"{_f(h['within_sd_bins'], 1)} | {_f(p['informative_fraction'])} | "
            f"{_f(p['abs_delta_reference_scales']['0.5'])} | {_f(p['abs_delta_bins']['0.5'], 1)} | "
            f"{_f(below['1'])} / {_f(below['2'])} / {_f(below['5'])} |"
        )
    lines += ["", "Informative share and median abs dG (bins) by decisions remaining:", ""]
    for head in HEADS:
        parts = [
            f"{k}: {_f(v['pairs']['informative_fraction'])} / {_f(v['pairs']['abs_delta_bins']['0.5'], 1)} (n={v['rows']})"
            for k, v in ts[head]["by_remaining"].items()
        ]
        lines.append(f"- {head}: " + "; ".join(parts))
    lines += ["", "By camp size:", ""]
    for head in HEADS:
        parts = [
            f"{k}: {_f(v['pairs']['informative_fraction'])} / {_f(v['pairs']['abs_delta_bins']['0.5'], 1)} (n={v['rows']})"
            for k, v in ts[head]["by_camp_size"].items()
        ]
        lines.append(f"- {head}: " + "; ".join(parts))
    lines += ["", "## 2. Critic error decomposition", ""]
    for label, section in report["critic"].items():
        lines += [
            f"### {label}",
            "",
            "| head | nMAE | bias | R2 | within R2 | rank (CI) | pooled sd ratio | collapse frac | within slope | entropy (eff bins) | upper mass |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for head in HEADS:
            m = section[head]
            ci = m["ranking"].get("bootstrap_ci95")
            ci_s = "" if not ci else f" ({ci[0]:.2f}-{ci[1]:.2f})"
            lines.append(
                f"| {head} | {_f(m['nmae'])} | {_f(m['bias'])} | {_f(m['r2'])} | {_f(m['within_group_r2'])} | "
                f"{_f(m['ranking']['accuracy'])}{ci_s} | {_f(m['spread']['pooled_sd_ratio'])} | "
                f"{_f(m['spread']['collapse_fraction'])} | {_f(m['spread']['within_slope'])} | "
                f"{_f(m['distribution']['mean_effective_bins'], 1)} | {_f(m['distribution']['upper_mass'], 4)} |"
            )
        lines.append("")
        for head in HEADS:
            m = section[head]
            margin = "; ".join(
                f"({b['low']:g},{_f(b['high'], 2) if b['high'] is not None else 'inf'}]: {_f(b['accuracy'])} n={b['pairs']}"
                for b in m["ranking"]["by_margin"]
            )
            lines.append(f"- {head} ranking by abs dG (ref scales): {margin}")
            for stratum in ("remaining", "camp_size", "target_magnitude"):
                parts = "; ".join(
                    f"{k}: nMAE {_f(v['nmae'])} bias {_f(v['bias'])} rank {_f(v['ranking'])}"
                    for k, v in m[f"by_{stratum}"].items()
                )
                lines.append(f"- {head} by {stratum}: {parts}")
        lines.append("")
    nf = report["noise_floor"]
    lines += [
        "## 3. Noise floor",
        "",
        f"Folds by {nf['fold_unit']} ({nf['folds']}); {nf['features']} input features, "
        f"{nf['hidden_features']} trunk features; k-NN k={nf['knn_k']} on {nf['pca_components']} PCs.",
        "",
        "| target | model | nMAE | within R2 | rank | rank CI |",
        "|---|---|---|---|---|---|",
    ]
    for target, entries in nf["models"].items():
        for model, m in entries.items():
            if "ranking" not in m:
                continue
            ci = m["ranking"].get("bootstrap_ci95")
            lines.append(
                f"| {target} | {model} | {_f(m['nmae'])} | {_f(m['within_group_r2'])} | "
                f"{_f(m['ranking']['accuracy'])} | {'-' if not ci else f'{ci[0]:.2f}-{ci[1]:.2f}'} |"
            )
        lines.append(
            f"| {target} | group mean (LOO) | {_f(nf['group_mean_loo_nmae'].get(target))} | 0 | 0.5 | |"
        )
    search = report.get("search")
    if search:
        lines += [
            "",
            "## 4. Search candidate disparity",
            "",
            f"{search.get('roots', 0)} roots from {search.get('shards', 0)} shards ({search.get('config_source')}).",
            "",
        ]
        for note in search.get("notes", []):
            lines.append(f"- {note}")
        for name, comp in (search.get("components") or {}).items():
            re = comp["reevaluated"]
            lines.append(
                f"- {name}: re-evaluated roots {re['roots_with_pairs']} (zero SE {re['roots_with_zero_se']}), "
                f"share beyond 2 SE at materiality 0/0.01/0.05: "
                + " / ".join(
                    _f(re["share_roots_any_pair_beyond_2se"].get(k)) for k in ("0", "0.01", "0.05")
                )
            )
            for h, entry in comp["by_horizon"].items():
                lines.append(
                    f"  - h={h}: roots {entry['roots']}, median range {_f(entry['candidate_range']['0.5'], 4)}, "
                    f"median paired sd {_f(entry['paired_stream_sd']['0.5'], 4)}, share beyond 2 sd at 0/0.01/0.05: "
                    + " / ".join(
                        _f(entry["share_roots_any_pair_beyond_2sd"].get(k))
                        for k in ("0", "0.01", "0.05")
                    )
                )
        g = search.get("qol_vs_population_gains")
        if g:
            lines.append(
                f"- QoL vs population-years gains over H12: Pearson {_f(g['pearson'])}, Spearman {_f(g['spearman'])}, "
                f"median per-root Spearman {_f(g['median_per_root_spearman'])}, opposite sign {_f(g['opposite_sign_fraction'])}, "
                f"argmax differs {_f(g['share_roots_argmax_differs'])} (n={g['candidates']})"
            )
    v = report["verdict"]
    lines += ["", "## 5. Verdict", ""]
    for name, h in v["hypotheses"].items():
        lines.append(f"- **{name}: {h['status']}**. {h['meaning']}")
        lines.append(f"  - evidence: `{json.dumps(h['evidence'], default=str)}`")
    lines.append(f"- generalisation gap flag: {v['generalisation_gap']}")
    lines += ["", "Recommended: " + "; ".join(v["recommended_paths"]), "", "Caveats:", ""]
    lines += [f"- {c}" for c in v["caveats"]]
    lines += [
        "",
        f"Thresholds: `{json.dumps(v['thresholds'])}`",
        "",
        f"Timings (s): `{json.dumps({k: round(x, 1) for k, x in report['timings_seconds'].items()})}`",
    ]
    return "\n".join(lines) + "\n"


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="calibration dataset")
    parser.add_argument("--fit-dataset", type=Path, default=None)
    parser.add_argument(
        "--fit-worlds",
        type=int,
        default=10,
        help="first N eligible fitting worlds (reserved Q-pi worlds excluded); 0 = all",
    )
    parser.add_argument("--search-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    torch.set_num_threads(max(1, min(8, args.threads)))
    from .labor_pretrain_eval import _load_models

    _, _, task, _, normalizer = _load_models(args.checkpoint)
    from .labor_pretrain_eval import _load_dataset

    _log("loading calibration dataset")
    cal = _load_dataset(args.dataset, normalizer)
    fit = None
    if args.fit_dataset is not None:
        _log("loading fitting subset")
        fit = load_subset(args.fit_dataset, normalizer, args.fit_worlds or None)
    report = run_diagnosis(
        task,
        cal,
        fit,
        args.search_dir,
        chunk=args.chunk,
        folds=args.folds,
        knn_k=args.knn_k,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    report["inputs"] = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "dataset_digest": cal["collection"]["dataset_digest"],
        "fit_dataset": None if args.fit_dataset is None else str(args.fit_dataset),
        "fit_dataset_digest": None if fit is None else fit["collection"]["dataset_digest"],
        "search_dir": None if args.search_dir is None else str(args.search_dir),
        "threads": torch.get_num_threads(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=1, sort_keys=True))
    os.replace(temporary, args.out)
    summary_path = args.summary or args.out.with_suffix(".md")
    summary_path.write_text(summarize(report), encoding="utf-8")
    print(summarize(report))
    print(f"report: {args.out}\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
