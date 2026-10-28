"""Merge the per-world development-panel reports of the actor pilot and apply the actor gates.

Input: <panel>/world-<seed>.json, one ``labor_pretrain_eval autonomous`` report per
development world (modes heuristic H, h12, deterministic, stochastic). The runs of every
world are pooled and re-aggregated with the evaluator's own ``autonomous_report`` (so the
numbers equal one 24-world invocation), then judged:

* manifest section 8 (r5, primary): against H12 (the converted heuristic through the
  action-12 interface), welfare and person-years >= 0.95 in aggregate and >= 0.90 in every
  world, deterministic and stochastic separately, and no starvation extinction beyond H12's;
* brief S1 (reported): the evaluator's H-relative gate (aggregate >= 0.90, per world >= 0.85,
  no additional starvation extinction versus H);
* interface quality (reported): H12 / H welfare and person-years (0.965 / 1.082 in G0).

It is a pilot on 4 of the 8 fitting batches: indicative, never gate evidence, and the
24-world x 50-decision panel is not comparable with G0's 8 x 8 panel (manifest section 11).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MANIFEST_GATE = {"aggregate_min": 0.95, "per_world_min": 0.90}
METRICS = ("welfare", "person_years")


def load(panel: Path) -> tuple[list[dict], list[int], dict]:
    runs, worlds, meta = [], [], {}
    for path in sorted(panel.glob("world-*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        runs.extend(report["runs"])
        worlds.extend(int(w) for w in report["worlds"])
        for key in ("turns", "reset_options", "policy_seed", "checkpoint"):
            value = report.get(key)
            if key in meta and meta[key] != value:
                raise ValueError(f"{path.name}: {key} {value!r} differs from {meta[key]!r}")
            meta[key] = value
    if not runs:
        raise ValueError(f"no world-*.json reports in {panel}")
    if len(set(worlds)) != len(worlds):
        raise ValueError("a world appears in two panel reports")
    return runs, sorted(worlds), meta


def judge(report: dict, modes: list[str]) -> dict:
    per_world = report["per_world"]
    out: dict = {}
    for mode in [m for m in modes if m != "h12"]:
        vs = (report.get("actor_vs_h12") or {}).get(mode)
        if vs is None:
            out[mode] = {"status": "HOLD", "reason": "no h12 mode in the panel"}
            continue
        checks = {}
        for metric in METRICS:
            aggregate = vs["ratios"][metric]
            worlds = {w: r[metric] for w, r in vs["per_world"].items()}
            low = {w: r for w, r in worlds.items() if r is None or r < MANIFEST_GATE["per_world_min"]}
            checks[f"{metric}_aggregate"] = {
                "value": aggregate,
                "threshold": MANIFEST_GATE["aggregate_min"],
                "pass": aggregate is not None and aggregate >= MANIFEST_GATE["aggregate_min"],
            }
            checks[f"{metric}_per_world"] = {
                "min": min((r for r in worlds.values() if r is not None), default=None),
                "threshold": MANIFEST_GATE["per_world_min"],
                "worlds_below": sorted(low),
                "pass": not low,
            }
        extra = {
            w: per_world[w][mode]["starvation_extinctions"] - per_world[w]["h12"]["starvation_extinctions"]
            for w in per_world
        }
        checks["no_additional_starvation_vs_h12"] = {
            "worlds_above": sorted(w for w, d in extra.items() if d > 0),
            "pass": all(d <= 0 for d in extra.values()),
        }
        out[mode] = {
            "status": "PASS" if all(c["pass"] for c in checks.values()) else "FAIL",
            "ratios_vs_h12": vs["ratios"],
            "checks": checks,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--modes", default="h12,deterministic,stochastic")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from anthropocene_rl.autonomous_v7.labor_pretrain_eval import (
        AUTONOMOUS_GATES,
        autonomous_report,
    )

    modes = [m for m in args.modes.split(",") if m]
    runs, worlds, meta = load(args.panel)
    report = autonomous_report(runs, modes)
    h12_vs_h = None
    if "h12" in modes:
        h12_vs_h = report["aggregate"]["h12"]["ratios"]
    result = {
        "schema": "r5-actor-pilot-panel-v1",
        "worlds": worlds,
        "panel": {**meta, "modes": modes, "world_count": len(worlds)},
        "manifest_gate_vs_h12": {"thresholds": MANIFEST_GATE, "modes": judge(report, modes)},
        "brief_s1_gate_vs_h": {
            "thresholds": AUTONOMOUS_GATES,
            "modes": {m: report["aggregate"][m]["gate"] for m in modes if m != "h12"},
            "checks": {m: report["aggregate"][m]["checks"] for m in modes if m != "h12"},
        },
        "interface_h12_over_h": h12_vs_h,
        "h12_fallbacks": report.get("h12_fallbacks"),
        "aggregate": report["aggregate"],
        "actor_vs_h12": report.get("actor_vs_h12"),
        "per_world": report["per_world"],
        "note": "pilot (BC on fit8-b01..b04, 1 seed): indicative, not gate evidence",
    }
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True, default=float), encoding="utf-8")
    lines = [f"panel worlds={len(worlds)} turns={meta.get('turns')} modes={','.join(modes)}"]
    if h12_vs_h:
        lines.append(
            f"  H12/H welfare={h12_vs_h['welfare']:.3f} person-years={h12_vs_h['person_years']:.3f}"
        )
    for mode, verdict in result["manifest_gate_vs_h12"]["modes"].items():
        r = verdict.get("ratios_vs_h12") or {}
        lines.append(
            f"  {mode}: vs H12 {verdict['status']} welfare={r.get('welfare')} "
            f"person-years={r.get('person_years')} qol={r.get('qol')}; brief S1 vs H "
            f"{result['brief_s1_gate_vs_h']['modes'].get(mode)}"
        )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
