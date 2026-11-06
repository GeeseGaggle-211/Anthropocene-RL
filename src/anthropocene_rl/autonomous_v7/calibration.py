"""Held-out eight-turn calibration and the frozen round-3 numerical gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import contracts, data
from .provenance import ProvenanceError, digest

KINDS = ("policy", "native", "region", "mix", "fission", "trade")
MANDATORY = {"native/welfare", "region/welfare", "mix/welfare"}
MINIMUM = 20
BANDS = ((0, 30), (30, 100), (100, 300), (300, float("inf")))


def manifest(path: Path | dict) -> dict:
    obj = json.loads(Path(path).read_text()) if not isinstance(path, dict) else dict(path)
    from .protocol import scan

    scan(obj)
    for dimension in ("seeds", "snapshots"):
        groups = [
            set(obj.get(f"{role}_{dimension}", []))
            for role in ("calibration", "training", "selection")
        ]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ProvenanceError(f"calibration manifest {dimension} families overlap")
    if not obj.get("calibration_seeds") and not obj.get("calibration_snapshots"):
        raise ProvenanceError("calibration manifest has no held-out families")
    if not any(k in obj for k in ("training_seeds", "training_snapshots")) or not any(
        k in obj for k in ("selection_seeds", "selection_snapshots")
    ):
        raise ProvenanceError("calibration manifest must declare training and selection exclusions")
    waivers = set(obj.get("waivers", []))
    if waivers & MANDATORY or any(w.startswith(k + "/") for w in waivers for k in MANDATORY):
        raise ProvenanceError("mandatory welfare calibration cannot be waived")
    obj["manifest_digest"] = digest({k: v for k, v in obj.items() if k != "manifest_digest"})
    return obj


def held_out(obj: dict, seed: int, snapshot: str | None = None) -> bool:
    return seed in set(obj.get("calibration_seeds", [])) or bool(
        snapshot and snapshot in set(obj.get("calibration_snapshots", []))
    )


def assert_training(obj: dict, seeds=(), snapshots=()) -> None:
    if set(seeds) & set(obj.get("calibration_seeds", [])) or set(snapshots) & set(
        obj.get("calibration_snapshots", [])
    ):
        raise ProvenanceError("held-out calibration family was supplied to training or selection")


def assert_training_rows(obj: dict, rows: dict) -> None:
    """Check attributable families; `world` is only a rebased trajectory/sequence key."""
    if "source_seed" not in rows:
        raise ProvenanceError("training rows lack attributable source_seed identity")
    assert_training(
        obj,
        np.asarray(rows["source_seed"]).tolist(),
        np.asarray(rows.get("source_snapshot", [])).tolist(),
    )


def windows(rows: dict, turns: int = 8):
    cal = np.asarray(rows.get("calibration", np.zeros(len(rows["camp"]))), bool)
    for w in sorted(set(rows["world"][cal].tolist())):
        m = rows["world"] == w
        ts = np.unique(rows["turn"][m])
        # Missing turns never become an apparently complete eight-turn window.
        if not np.array_equal(ts, np.arange(ts[0], ts[0] + len(ts))):
            yield w, m, None, None
            continue
        endpoint = ts[turns] if len(ts) > turns else None
        external = rows.get("endpoint_turn", np.full(len(m), -1))[m][0]
        dead = rows["terminated"][m & (rows["turn"] == ts[-1])].all() and external < 0
        if len(ts) >= turns and (endpoint is not None or external == ts[0] + turns):
            yield w, m, ts[:turns], int(ts[0] + turns)
        elif len(ts) <= turns and dead:
            yield w, m, ts, None
        else:
            yield w, m, None, None


def sample_returns(rows: dict, window, m, components):
    gamma = 1.0
    returns = {name: 0.0 for name in components}
    for turn in window:
        at = m & (rows["turn"] == turn)
        for name in returns:
            returns[name] += gamma * float(components[name][at].sum())
        gamma *= float(data.discount(rows["years"][at][:1])[0])
    return returns, gamma


def statistics(pred, real, objective):
    p, r = np.asarray(pred, float), np.asarray(real, float)
    scale = contracts.OBJECTIVE_REFERENCE_SCALES[objective]
    near = np.abs(r) <= 0.01 * scale
    fixed = (p - r) / scale
    rel = (p[~near] - r[~near]) / r[~near]
    near_abs = np.abs(fixed[near])
    finite = bool(np.isfinite(p).all() and np.isfinite(r).all())
    mean = float(rel.mean()) if len(rel) else None
    median = float(np.median(rel)) if len(rel) else None
    near_mean = float(near_abs.mean()) if len(near_abs) else None
    near_median = float(np.median(near_abs)) if len(near_abs) else None
    return {
        "branches": len(r),
        "nonzero": int(np.count_nonzero(r)),
        "non_near_zero": int((~near).sum()),
        "near_zero": int(near.sum()),
        "tau": 0.01 * scale,
        "mean": mean,
        "median": median,
        "fixed_mean": float(fixed.mean()),
        "fixed_median": float(np.median(fixed)),
        "near_absolute_mean": near_mean,
        "near_absolute_median": near_median,
        "near_pass": finite and (not len(near_abs) or (near_mean <= 0.10 and near_median <= 0.10)),
        "relative_pass": finite
        and len(rel) >= MINIMUM
        and abs(mean) <= 0.10
        and abs(median) <= 0.10,
    }


def calibration(pred: dict, rows: dict, turns: int = 8, endpoints: dict | None = None) -> dict:
    components = data.reward_components(rows)
    samples = {}
    short = total = 0
    for w, m, window, end in windows(rows, turns):
        if window is None:
            short += 1
            continue
        values, gamma = sample_returns(rows, window, m, components)
        kind = KINDS[int(rows["alt_kind"][m][0])]
        cells = samples.setdefault(kind, {name: ([], []) for name in pred})
        for name in pred:
            v0 = float(pred[name][m & (rows["turn"] == window[0])].sum())
            at_end = m & (rows["turn"] == end) if end is not None else np.zeros(len(m), bool)
            if at_end.any():
                v_end = float(pred[name][at_end].sum())
            elif end is not None:
                if endpoints is None or (w, end, name) not in endpoints:
                    raise ValueError("calibration endpoint observation has no evaluated value")
                v_end = endpoints[w, end, name]
            else:
                v_end = 0.0
            cells[name][0].append(v0 - gamma * v_end)
            cells[name][1].append(values[name])
        total += 1
    return {
        "turns": turns,
        "target": 0.10,
        "branches": total,
        "short": short,
        "kinds": {
            kind: {name: statistics(p, r, name) for name, (p, r) in cells.items()}
            for kind, cells in samples.items()
        },
    }


def band_name(lo, hi):
    return f"{lo}+" if hi == float("inf") else f"{lo}-{int(hi) - 1}"


def coverage(
    rows: dict, turns: int = 8, minimum: int = 20, band_minimum: int = 5, warn: bool = True
) -> dict:
    import warnings

    components = data.reward_components(rows)
    cells = {k: {o: 0 for o in contracts.OBJECTIVES} for k in KINDS}
    nonzero = {k: dict(v) for k, v in cells.items()}
    bands = {k: {band_name(lo, hi): 0 for lo, hi in BANDS} for k in KINDS}
    branches = {k: 0 for k in KINDS}
    short = 0
    for _, m, window, _ in windows(rows, turns):
        if window is None:
            short += 1
            continue
        kind = KINDS[int(rows["alt_kind"][m][0])]
        branches[kind] += 1
        returns, _ = sample_returns(rows, window, m, components)
        for name, value in returns.items():
            cells[kind][name] += 1
            nonzero[kind][name] += int(abs(value) > 1e-9)
        population = float(rows["prior_population"][m & (rows["turn"] == window[0])].sum())
        for lo, hi in BANDS:
            if lo <= population < hi:
                bands[kind][band_name(lo, hi)] += 1
    problems = [f"{k}/{o}" for k, by in cells.items() for o, n in by.items() if n < minimum]
    problems += [
        f"{k}/population/{b}" for k, by in bands.items() for b, n in by.items() if n < band_minimum
    ]
    if warn and problems:
        warnings.warn(f"calibration coverage: {len(problems)} insufficient cells", stacklevel=2)
    return {
        "minimum": minimum,
        "band_minimum": band_minimum,
        "turns": turns,
        "branches": branches,
        "objective_cells": cells,
        "nonzero_objective_cells": nonzero,
        "population_bands": bands,
        "warnings": problems,
        "complete": not problems,
        "short": short,
    }


def verdict(cal: dict, cov: dict, waivers=()) -> dict:
    waived = set(waivers)
    if waived & MANDATORY:
        raise ValueError("mandatory welfare calibration cannot be waived")
    failures = []
    for key in sorted(MANDATORY):
        kind, objective = key.split("/")
        cell = cal["kinds"].get(kind, {}).get(objective)
        if cell is None or cell["branches"] < MINIMUM or cell["non_near_zero"] < MINIMUM:
            failures.append(key + ": insufficient_calibration")
        elif not cell["relative_pass"] or not cell["near_pass"]:
            failures.append(key + ": biased")
    mandatory_pass = not failures
    for kind, by in cal["kinds"].items():
        for objective, cell in by.items():
            key = kind + "/" + objective
            if not cell["near_pass"] and key not in waived:
                failures.append(key + ": near_zero_error")
            if 0 < cell["non_near_zero"] < MINIMUM and key not in waived:
                failures.append(key + ": insufficient_calibration")
            if cell["non_near_zero"] >= MINIMUM and not cell["relative_pass"] and key not in waived:
                failures.append(key + ": relative_error")
    uncovered = [k for k in cov["warnings"] if k not in waived]
    return {
        "pass": not failures and not uncovered,
        "mandatory_pass": mandatory_pass,
        "failures": failures,
        "unwaived_coverage": uncovered,
        "waivers": sorted(waived),
    }
