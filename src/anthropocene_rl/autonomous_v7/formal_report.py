"""Reviewed paired personality statistics. Complete per-world evidence is mandatory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .protocol import ROLES, artifact, scan
from .provenance import ProvenanceError


def worlds(report: dict, point: str, *, focal=False) -> dict:
    scan(report)
    rows = report.get(point, {}).get("per_seed", [])
    if len(rows) != 60 or sorted(r.get("seed", -1) for r in rows) != ROLES["development"]:
        raise ProvenanceError("formal report missing/duplicate/unregistered per-world evidence")
    required = ("population", "qol", "welfare", "extinctions")
    if focal:
        required += (
            "focal_population",
            "focal_qol",
            "focal_welfare",
            "focal_extinctions",
            "focal_root_lineages",
            "focal_population_denominator",
            "focal_societies",
        )
    for row in rows:
        if row.get("withheld") != 0 or row.get("warmup") != 0:
            raise ProvenanceError("formal report is not a fresh default-world comparison")
        from .worker_contract import world

        if row.get("world_contract") != world(
            row["seed"],
            movement=report.get("_meta", {}).get("movement", False),
            head=report.get("_meta", {}).get("region_head", "hierarchical"),
        ):
            raise ProvenanceError(
                "formal per-world evidence lacks exact pre-execution world contract"
            )
        from .personality_sweep import GRID

        if row.get("personality") != GRID[point]:
            raise ProvenanceError("formal per-world personality differs from canonical grid")
        if any(k not in row for k in required):
            raise ProvenanceError("formal focal/per-world evidence incomplete")
        if any(not np.isfinite(row[k]) for k in required if k != "focal_societies"):
            raise ProvenanceError("formal report contains nonfinite evidence")
        if focal and (
            row["focal_population_denominator"] != row["focal_population"]
            or row["focal_root_lineages"] <= 0
        ):
            raise ProvenanceError("invalid focal population/lineage denominator")
    return {r["seed"]: r for r in rows}


def ordered(reports: list[dict]) -> list[dict]:
    if sorted(r.get("_meta", {}).get("optimizer_seed", -1) for r in reports) != list(
        range(170, 175)
    ):
        raise ProvenanceError(
            "formal report requires five uniquely matched optimizer seeds 170-174"
        )
    return sorted(reports, key=lambda r: r["_meta"]["optimizer_seed"])


def separation(reports: list[dict], *, focal=False) -> dict:
    q, p = [], []
    for report in ordered(reports):
        a, b = worlds(report, "qol", focal=focal), worlds(report, "population", focal=focal)
        prefix = "focal_" if focal else ""
        q.append(
            float(
                np.mean([a[s][prefix + "qol"] - b[s][prefix + "qol"] for s in ROLES["development"]])
            )
        )
        p.append(
            float(
                np.mean(
                    [
                        b[s][prefix + "population"] - a[s][prefix + "population"]
                        for s in ROLES["development"]
                    ]
                )
            )
        )
    sq, sp = float(np.std(q, ddof=1)), float(np.std(p, ddof=1))
    success = [x > sq and y > sp for x, y in zip(q, p, strict=False)]
    return {
        "pass": sum(success) >= 4,
        "qol_gaps": q,
        "population_gaps": p,
        "qol_sample_sd": sq,
        "population_sample_sd": sp,
        "success": success,
        "succeeded": sum(success),
    }


def paired_interval(a, b, *, difference=False, draws=4000, seed=0) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (
        a.ndim != 2
        or a.shape != b.shape
        or a.shape[0] != 5
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ProvenanceError(
            "paired interval requires finite same-shaped five-seed x world matrices"
        )
    if not a.shape[1] or (not difference and a.sum() <= 0):
        raise ProvenanceError("invalid paired denominator/world count")
    rng = np.random.default_rng(seed)
    point = float((b - a).mean()) if difference else float(b.sum() / a.sum())
    values = []
    for _ in range(draws):
        s = rng.integers(0, 5, 5)
        w = rng.integers(0, a.shape[1], a.shape[1])
        aa, bb = a[s][:, w], b[s][:, w]
        if not difference and aa.sum() <= 0:
            raise ProvenanceError("bootstrap denominator is not positive")
        values.append(float((bb - aa).mean()) if difference else float(bb.sum() / aa.sum()))
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "draws": draws,
        "seed": seed,
        "paired_optimizer_seeds": True,
    }


def default_comparison(a: list[dict], b: list[dict]) -> dict:
    aa, bb = ordered(a), ordered(b)

    def matrix(reports, key):
        checked = [worlds(r, "welfare") for r in reports]
        return [[r[s][key] for s in ROLES["development"]] for r in checked]

    ratio = paired_interval(matrix(aa, "welfare"), matrix(bb, "welfare"))
    extinction = paired_interval(
        matrix(aa, "extinctions"), matrix(bb, "extinctions"), difference=True
    )
    return {
        "pass": ratio["lo"] >= 0.8 and ratio["hi"] <= 1.2 and extinction["hi"] <= 0,
        "welfare_ratio": ratio,
        "extinction_difference": extinction,
    }


def evaluate(spec: dict, base: Path) -> dict:
    from .provenance import digest
    from .qualified_cohort import load

    qualified, frozen_runs = load(spec.get("qualification"), base)
    reports = {}
    checkpoint_arms = {}
    for arm in ("R3def", "R3per", "R3lo"):
        reports[arm] = {}
        for form in ("best", "latest"):
            reports[arm][form] = {}
            for mode in ("all",) if arm == "R3def" else ("all", "focal"):
                refs = spec["arms"][arm][form][mode]
                loaded = []
                for ref in refs:
                    path, _ = artifact(ref, base)
                    obj = json.loads(path.read_text())
                    meta = obj.get("_meta", {})
                    if (
                        meta.get("schema") != 1
                        or meta.get("checkpoint_form") != form
                        or meta.get("focal") != (mode == "focal")
                        or meta.get("turns") != 25
                    ):
                        raise ProvenanceError("formal report form/mode identity mismatch")
                    from .personality_sweep import GRID

                    if set(obj) != set(GRID) | {"_meta"} or meta.get("points") != list(GRID):
                        raise ProvenanceError("formal report requires the exact seven-point grid")
                    if any(obj[name].get("personality") != vector for name, vector in GRID.items()):
                        raise ProvenanceError(
                            "formal grid labels do not match actual canonical personalities"
                        )
                    for name in GRID:
                        worlds(obj, name, focal=mode == "focal")
                    checkpoint_path, _ = artifact(
                        {"path": meta["checkpoint"], "sha256": meta["checkpoint_sha256"]},
                        path.parent,
                    )
                    import torch

                    from .checkpoint import saved_head
                    from .promotion import semantic_hash

                    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    from .completion import validate

                    validate(meta.get("completion"), checkpoint_path, form, path.parent)
                    frozen = state.get("run_manifest", {})
                    if digest(frozen) != frozen_runs.get((arm, meta.get("optimizer_seed"))):
                        raise ProvenanceError(
                            "formal checkpoint/completion does not map to the one qualified cohort"
                        )
                    if meta.get("worker_contract") != frozen.get("worker_contract"):
                        raise ProvenanceError(
                            "formal sweep lacks frozen pre-execution worker evidence"
                        )
                    actual_arm = frozen.get("options", {}).get("arm")
                    if actual_arm != arm or meta.get("arm") != arm:
                        raise ProvenanceError(
                            "formal report/checkpoint does not belong to requested arm"
                        )
                    options = frozen.get("options", {})
                    hyper = frozen.get("resolved_hyper", {})
                    if (
                        options.get("personality")
                        != ("default" if arm == "R3def" else "dirichlet:1")
                        or hyper.get("teacher_kl") != (0.002 if arm == "R3lo" else 0.01)
                        or hyper.get("personality") != options["personality"]
                    ):
                        raise ProvenanceError(
                            "checkpoint treatment configuration does not match requested arm"
                        )
                    sha = meta["checkpoint_sha256"]
                    if sha in checkpoint_arms and checkpoint_arms[sha] != arm:
                        raise ProvenanceError(
                            "formal report aliases one checkpoint across treatment arms"
                        )
                    checkpoint_arms[sha] = arm
                    from .personality_sweep import validate_producers

                    validate_producers([{"producer": meta.get("producer")}], state)
                    if (
                        meta.get("movement") is not True
                        or meta.get("region_head") != "hierarchical"
                        or saved_head(state) != "hierarchical"
                        or meta.get("model_semantics") != semantic_hash(state)
                        or meta.get("simulator") != frozen.get("simulator")
                        or meta.get("bridge_sha256") != frozen.get("bridge_sha256")
                        or meta.get("optimizer_seed") != frozen.get("options", {}).get("seed")
                    ):
                        raise ProvenanceError("formal report model/producer identity mismatch")
                    loaded.append(obj)
                reports[arm][form][mode] = ordered(loaded)
    sep, costs, low_vs_per = {}, {}, {}
    for arm in ("R3per", "R3lo"):
        sep[arm], costs[arm] = {}, {}
        for form in ("best", "latest"):
            sep[arm][form] = {
                mode: separation(reports[arm][form][mode], focal=mode == "focal")
                for mode in ("all", "focal")
            }
            costs[arm][form] = default_comparison(
                reports["R3def"][form]["all"], reports[arm][form]["all"]
            )
    for form in ("best", "latest"):
        low_vs_per[form] = default_comparison(
            reports["R3per"][form]["all"], reports["R3lo"][form]["all"]
        )
    passed = all(
        c["pass"] for by in sep.values() for forms in by.values() for c in forms.values()
    ) and all(c["pass"] for by in costs.values() for c in by.values())
    return {
        "schema": 1,
        "pass": passed,
        "separation": sep,
        "default_cost": costs,
        "lo_vs_per": low_vs_per,
        "qualification": spec["qualification"],
        "submission_digest": qualified["submission_digest"],
        "lo_vs_per_role": "report-only; not an additional registered pass gate",
        "scope": "personality/default-world gates only; global regime/reference guards require their own verified receipts",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.inputs.read_text()), args.inputs.parent)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
