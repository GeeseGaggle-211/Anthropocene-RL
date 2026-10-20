"""Paired comparison of two training arms from one stage_eval report, with bootstrap confidence intervals.

    python -m anthropocene_rl.autonomous_v7.paired_compare --report <stage_eval json> --a a1.pt,a2.pt --b b1.pt,b2.pt

Every policy in the report ran on the same worlds. Per regime, the effect is the ratio of arm B's to arm A's
welfare, each arm's welfare being its mean over training seeds summed over worlds. The interval resamples training
seeds (within each arm) and worlds (shared by both arms, so world difficulty cancels) together, so it covers both
seed noise and world sampling. ``--reference`` adds each arm's ratio against one further policy (e.g. their init).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def welfare_matrix(row: dict, checkpoints: list[str]) -> np.ndarray:
    """Seeds x worlds welfare for one regime."""
    return np.array([[w[2] for w in row["candidates"][c]["worlds"]] for c in checkpoints], dtype=np.float64)


def ratio_interval(a: np.ndarray, b: np.ndarray, draws: int, rng: np.random.Generator) -> dict[str, float]:
    point = b.mean(axis=0).sum() / max(1e-9, a.mean(axis=0).sum())
    worlds = a.shape[1]
    samples = np.empty(draws)
    for i in range(draws):
        w = rng.integers(0, worlds, worlds)
        sa = a[rng.integers(0, len(a), len(a))][:, w]
        sb = b[rng.integers(0, len(b), len(b))][:, w]
        samples[i] = sb.mean(axis=0).sum() / max(1e-9, sa.mean(axis=0).sum())
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"ratio": float(point), "lo": float(lo), "hi": float(hi)}


def extinction_matrix(row: dict, checkpoints: list[str]) -> np.ndarray:
    return np.array([[w[3] for w in row["candidates"][c]["worlds"]] for c in checkpoints], dtype=np.float64)


def difference_interval(a: np.ndarray, b: np.ndarray, draws: int, rng: np.random.Generator) -> dict[str, float]:
    """Arm B minus arm A in mean extinct settlements per world (E3: a rate with an interval, pooled over seeds),
    resampling seeds within each arm and the shared worlds."""
    point = b.mean() - a.mean()
    worlds = a.shape[1]
    samples = np.empty(draws)
    for i in range(draws):
        w = rng.integers(0, worlds, worlds)
        samples[i] = b[rng.integers(0, len(b), len(b))][:, w].mean() - a[rng.integers(0, len(a), len(a))][:, w].mean()
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"difference": float(point), "lo": float(lo), "hi": float(hi), "a_rate": float(a.mean()), "b_rate": float(b.mean())}


def compare(report: dict, arm_a: list[str], arm_b: list[str], reference: str | None = None, draws: int = 4000,
            seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for name, row in report.items():
        if name == "gate":
            continue
        a, b = welfare_matrix(row, arm_a), welfare_matrix(row, arm_b)
        entry = {"b_vs_a": ratio_interval(a, b, draws, rng), "worlds": int(a.shape[1]),
                 "extinctions_b_minus_a": difference_interval(extinction_matrix(row, arm_a), extinction_matrix(row, arm_b), draws, rng),
                 "per_seed_vs_teacher": {"a": [float(row["candidates"][c]["ratio_vs_teacher"]) for c in arm_a],
                                         "b": [float(row["candidates"][c]["ratio_vs_teacher"]) for c in arm_b]},
                 "extinctions": {"a": [row["candidates"][c]["extinctions"] for c in arm_a],
                                 "b": [row["candidates"][c]["extinctions"] for c in arm_b]}}
        if reference is not None:
            r = welfare_matrix(row, [reference])
            entry["a_vs_reference"] = ratio_interval(r, a, draws, rng)
            entry["b_vs_reference"] = ratio_interval(r, b, draws, rng)
        out[name] = entry
    return out


def merge(reports: list[dict]) -> dict:
    """stage_eval reports over the same worlds (e.g. candidate shards run in parallel) as one report."""
    out = json.loads(json.dumps(reports[0]))
    for other in reports[1:]:
        for name, row in other.items():
            if name == "gate":
                out["gate"].update(row)
                continue
            if [w[:2] for w in row["teacher"]["worlds"]] != [w[:2] for w in out[name]["teacher"]["worlds"]]:
                raise ValueError(f"{name}: reports cover different worlds")
            out[name]["candidates"].update(row["candidates"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, nargs="+", required=True, help="one or more stage_eval reports (merged)")
    parser.add_argument("--a", required=True, help="comma-separated checkpoints of arm A (control)")
    parser.add_argument("--b", required=True, help="comma-separated checkpoints of arm B (treatment)")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = merge([json.loads(p.read_text()) for p in args.report])
    result = compare(report, args.a.split(","), args.b.split(","), args.reference, args.draws)
    for name, e in result.items():
        r = e["b_vs_a"]
        x = e["extinctions_b_minus_a"]
        line = (f"{name:10s} worlds {e['worlds']:3d}  B/A {r['ratio']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]"
                f"  extinct/world A {x['a_rate']:.3f} B {x['b_rate']:.3f} diff [{x['lo']:+.3f}, {x['hi']:+.3f}]")
        for key in ("a_vs_reference", "b_vs_reference"):
            if key in e:
                q = e[key]; line += f"  {key} {q['ratio']:.4f} [{q['lo']:.4f}, {q['hi']:.4f}]"
        print(line)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
