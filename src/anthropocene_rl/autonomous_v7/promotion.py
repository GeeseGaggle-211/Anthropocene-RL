"""Deterministic, non-resumable clone promotion; every candidate and report is evidence, not just the winner."""  # noqa: E501 - docstring/prose line kept verbatim

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .calibration import manifest
from .protocol import ROLES, artifact, scan
from .provenance import ProvenanceError, checkpoint_hash, digest

SELECTOR = {
    "name": "max-eligible-movement-welfare-v1",
    "band": [0.8, 1.2],
    "tie_break": ["optimizer_seed_ascending", "checkpoint_sha256_lexical"],
    "worlds": ROLES["clone_selection"],
}
SEMANTIC_KEYS = (
    "actor",
    "critic",
    "cost_critic",
    "teacher_actor",
    "normalizer",
    "region_head",
    "core_width",
    "privileged_width",
    "token",
    "experts",
    "vector_critic",
    "contract",
    "commitments",
    "region_movement",
    "observe_movement",
    "learned_fission",
    "learned_trade",
)


def semantic_hash(state: dict) -> str:
    h = hashlib.sha256()

    def visit(obj):
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
        if isinstance(obj, np.ndarray):
            h.update(b"array")
            h.update(str(obj.dtype).encode())
            h.update(str(obj.shape).encode())
            h.update(obj.tobytes())
        elif isinstance(obj, dict):
            h.update(b"dict")
            for key in sorted(obj):
                h.update(str(key).encode() + b"\0")
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            h.update(b"list")
            for item in obj:
                visit(item)
        else:
            h.update(json.dumps(obj, sort_keys=True, allow_nan=False).encode() + b"\0")

    visit({k: state[k] for k in SEMANTIC_KEYS if k in state})
    return h.hexdigest()


def seal(state: dict) -> dict:
    state["model_semantics"] = semantic_hash(state)
    return state


def cohort_config(state):
    run = dict(state["run_manifest"])
    late = ("clone_protocol", "preregistration", "protocol_approval")
    run["options"] = {
        k: v for k, v in run.get("options", {}).items() if k not in ("seed", "out", *late)
    }
    run["inputs"] = {k: v for k, v in run.get("inputs", {}).items() if k not in late}
    for key in ("config_digest", "clone_protocol", "protocol_evidence"):
        run.pop(key, None)
    return {
        "projection_version": 1,
        "run": run,
        "bc_config": state.get("bc_config"),
        "token": state.get("token"),
        "vector_critic": state.get("vector_critic"),
        "region_head": state.get("region_head"),
    }


def validate_cohort(state):
    path, _ = artifact(state.get("run_manifest", {}).get("clone_protocol", {}))
    planned = json.loads(path.read_text())
    from .protocol import verify_preregistration

    evidence = state["run_manifest"].get("protocol_evidence", {})
    protocol = verify_preregistration(evidence.get("preregistration"), evidence.get("approval"))
    if (
        planned.get("schema") != 1
        or planned.get("projection_version") != 1
        or planned.get("kind") != "guarded_clone_cohort"
        or protocol.get("digest") != state["run_manifest"].get("protocol_digest")
        or protocol.get("digest") != state.get("calibration_manifest", {}).get("protocol_digest")
        or planned.get("protocol_digest") != protocol["digest"]
        or protocol.get("clone_protocol_sha256")
        != state["run_manifest"]["clone_protocol"]["sha256"]
        or planned.get("cohort_digest") != digest(cohort_config(state))
    ):
        raise ProvenanceError(
            "candidate differs from its pre-frozen clone configuration/input cohort"
        )
    return state["run_manifest"]["clone_protocol"]["sha256"]


def validate_report(
    report: dict, checkpoint_sha: str, semantics: str, frozen: dict
) -> tuple[float, dict]:
    scan(report)
    if (
        report.get("schema") != 1
        or report.get("kind") != "clone_evaluation"
        or report.get("checkpoint_sha256") != checkpoint_sha
        or report.get("model_semantics") != semantics
        or report.get("movement") is not True
        or report.get("region_head") != "hierarchical"
        or report.get("simulator") != frozen.get("simulator")
        or report.get("bridge_sha256") != frozen.get("bridge_sha256")
        or report.get("flags") != {k: frozen[k] for k in ("sim_rules", "sim_statics")}
    ):
        raise ProvenanceError("clone report identity/configuration mismatch")
    rows = report.get("worlds", [])
    if len(rows) != 20 or sorted(r["seed"] for r in rows) != ROLES["clone_selection"]:
        raise ProvenanceError("clone selection report missing/duplicate/unregistered worlds")
    for row in rows:
        if (
            row.get("withheld") != 0
            or row.get("warmup") != 0
            or row.get("snapshot", "")
            or not all(
                isinstance(row.get(k), (int, float)) and np.isfinite(row[k]) and row[k] >= 0
                for k in ("policy_welfare", "heuristic_welfare")
            )
        ):
            raise ProvenanceError("invalid clone per-world evidence")
    denominator = sum(r["heuristic_welfare"] for r in rows)
    if denominator <= 0:
        raise ProvenanceError("clone report has no positive heuristic denominator")
    ratio = sum(r["policy_welfare"] for r in rows) / denominator
    if report.get("ratio") != ratio:
        raise ProvenanceError("clone ratio differs from per-world evidence")
    return ratio, {"seeds": [r["seed"] for r in rows], "snapshots": []}


def validate_embedded(state: dict) -> None:
    p = state.get("promotion") or {}
    if (
        p.get("schema") != 1
        or p.get("selector") != SELECTOR
        or p.get("digest") != digest({k: v for k, v in p.items() if k != "digest"})
        or state.get("artifact_kind") != "promotion"
        or state.get("resumable") is not False
        or any(
            k in state for k in ("optimizers", "optimizer", "numpy_rng", "torch_rng", "generator")
        )
    ):
        raise ProvenanceError("invalid/non-immutable promotion artifact")
    candidates = p.get("candidates", [])
    if sorted(c.get("seed", -1) for c in candidates) != list(range(10)):
        raise ProvenanceError("promotion lacks all ten candidates")
    for c in candidates:
        validate_report(c["report"], c["checkpoint_sha256"], c["model_semantics"], c["frozen_run"])
    eligible = [c for c in candidates if 0.8 <= c["report"]["ratio"] <= 1.2]
    if not eligible:
        raise ProvenanceError("no eligible promoted clone")
    chosen = min(eligible, key=lambda c: (-c["report"]["ratio"], c["seed"], c["checkpoint_sha256"]))
    parents = state.get("family_lineage", {}).get("parents", [])
    by_hash = {parent["checkpoint"]: parent["proof"] for parent in parents}
    actual_candidates = []
    for parent in parents:
        path, sha = artifact(parent.get("artifact", {}))
        if sha != parent["checkpoint"]:
            raise ProvenanceError("promotion candidate artifact hash differs")
        candidate = torch.load(path, map_location="cpu", weights_only=False)
        validate_cohort(candidate)
        actual_candidates.append(candidate)
    if (
        len({validate_cohort(candidate) for candidate in actual_candidates}) != 1
        or len({digest(cohort_config(candidate)) for candidate in actual_candidates}) != 1
    ):
        raise ProvenanceError("recursive promotion candidates differ from one pre-frozen cohort")
    if any(
        c["checkpoint_sha256"] not in by_hash
        or c["frozen_run"] != by_hash[c["checkpoint_sha256"]].get("run_manifest")
        or c["model_semantics"] != by_hash[c["checkpoint_sha256"]].get("model_semantics")
        for c in candidates
    ):
        raise ProvenanceError("promotion report candidate differs from inherited proof")
    if (
        {c["checkpoint_sha256"] for c in candidates} != {r["checkpoint"] for r in parents}
        or p.get("selected") != chosen["checkpoint_sha256"]
        or state.get("model_semantics") != chosen["model_semantics"]
        or any(c.get("report_sha256") != digest(c["report"]) for c in candidates)
    ):
        raise ProvenanceError("promotion winner/selection closure binding mismatch")


def promote(inputs: list[dict], held: dict) -> dict:
    from .lineage import build, checkpoint_evidence, families, merge

    candidates, parents, states, selection = [], [], {}, []
    for item in inputs:
        path, sha = artifact(item["checkpoint"])
        report_path, _ = artifact(item["report"])
        state = torch.load(path, map_location="cpu", weights_only=False)
        scan(state)
        check = checkpoint_evidence(state, held)
        if not check["verified"]:
            raise ProvenanceError("candidate lineage unverified: " + "; ".join(check["failures"]))
        semantics = semantic_hash(state)
        if state.get("model_semantics") != semantics:
            raise ProvenanceError("candidate model/normalizer seal mismatch")
        validate_cohort(state)
        from .checkpoint import saved_head
        from .contracts import OBJECTIVES

        if (
            not state.get("token")
            or not state.get("vector_critic")
            or saved_head(state) != "hierarchical"
            or state.get("normalizer", {}).get("guard") != 10.0
            or any(
                f"heads.{name}.{parameter}" not in state.get("critic", {})
                for name in OBJECTIVES
                for parameter in ("weight", "bias")
            )
        ):
            raise ProvenanceError(
                "promotion requires guarded token/hierarchical clones with all four critic heads"
            )
        report = json.loads(report_path.read_text())
        _ratio, group = validate_report(report, sha, semantics, state["run_manifest"])
        seed = state.get("seed")
        if seed not in range(10) or sha in states:
            raise ProvenanceError("duplicate/unregistered clone candidate")
        candidates.append(
            {
                "seed": seed,
                "checkpoint_sha256": sha,
                "model_semantics": semantics,
                "report_file_sha256": checkpoint_hash(report_path),
                "report_sha256": digest(report),
                "report": report,
                "frozen_run": state["run_manifest"],
            }
        )
        parents.append((sha, state, item["checkpoint"]))
        states[sha] = state
        selection.append(group)
    if sorted(c["seed"] for c in candidates) != list(range(10)):
        raise ProvenanceError("promotion requires all ten clones")
    if (
        len({digest(cohort_config(state)) for state in states.values()}) != 1
        or len({validate_cohort(state) for state in states.values()}) != 1
    ):
        raise ProvenanceError(
            "clone candidates differ from one frozen non-seed configuration/input cohort"
        )
    eligible = [c for c in candidates if 0.8 <= c["report"]["ratio"] <= 1.2]
    if not eligible:
        raise ProvenanceError("no clone within mechanics sanity band")
    chosen = min(eligible, key=lambda c: (-c["report"]["ratio"], c["seed"], c["checkpoint_sha256"]))
    winner = states[chosen["checkpoint_sha256"]]
    promoted = {
        k: v
        for k, v in winner.items()
        if k not in ("optimizers", "optimizer", "numpy_rng", "torch_rng", "generator")
    }
    promoted.update(
        artifact_kind="promotion",
        resumable=False,
        family_lineage=build(held, families(), merge(*selection), parents),
    )
    p = {
        "schema": 1,
        "selector": SELECTOR,
        "selected": chosen["checkpoint_sha256"],
        "candidates": sorted(candidates, key=lambda c: c["seed"]),
    }
    p["digest"] = digest(p)
    promoted["promotion"] = p
    if semantic_hash(promoted) != semantic_hash(winner):
        raise ProvenanceError("promotion changed model semantics")
    validate_embedded(promoted)
    return promoted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    args = parser.parse_args()
    from .protocol import enforce_qualification_execution

    enforce_qualification_execution(args, "anthropocene_rl.autonomous_v7.promotion")
    if args.out.exists():
        raise ProvenanceError("promotion output already exists")
    result = promote(
        json.loads(args.inputs.read_text())["candidates"], manifest(args.calibration_manifest)
    )
    tmp = args.out.with_suffix(".tmp")
    torch.save(result, tmp)
    tmp.replace(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
