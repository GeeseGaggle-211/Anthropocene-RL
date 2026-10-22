"""Content-bound, complete movement-on clone selection reports for deterministic promotion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from . import contracts
from .calibration import manifest
from .evaluate import rollout
from .lineage import checkpoint_evidence
from .promotion import semantic_hash, validate_report
from .protocol import ROLES
from .provenance import checkpoint_hash, simulator_identity, ProvenanceError
from .teacher_data import binary_identity


def evaluate(path: Path, held: dict) -> dict:
    initial_sha = checkpoint_hash(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    check = checkpoint_evidence(state, held)
    if not check["verified"] or state.get("model_semantics") != semantic_hash(state):
        raise ProvenanceError("clone evaluation requires a verified sealed candidate")
    frozen = state["run_manifest"]
    from .checkpoint import resolve_head
    from .bridge import Model9Bridge
    if resolve_head(state) != "hierarchical":
        raise ProvenanceError("clone selection requires actual hierarchical inference")
    flags = {"sim_rules": os.environ.get("ANTHROPOCENE_SIM_RULES", ""),
             "sim_statics": os.environ.get("ANTHROPOCENE_SIM_STATICS", "")}
    with Model9Bridge() as probe:
        live = simulator_identity(probe.schema, production=True)
    if live != frozen["simulator"] or flags != {k: frozen[k] for k in flags}:
        raise ProvenanceError("clone evaluation live simulator/flags differ from candidate")
    if binary_identity()["sha256"] != frozen["bridge_sha256"]:
        raise ProvenanceError("clone evaluation bridge differs from candidate's frozen producer")
    rows = []
    from .worker_contract import from_state, world
    expected_worker = from_state(state, required=True)
    for seed in ROLES["clone_selection"]:
        policy = rollout(seed, 25, str(path), movement=True, region=True, expected_producer=expected_worker,
                         world_contract=world(seed, movement=True, head="hierarchical"))
        teacher = rollout(seed, 25, None, movement=True, expected_producer=expected_worker, world_contract=world(seed, movement=True))
        expected_producer = {"simulator": live, "bridge_sha256": frozen["bridge_sha256"], "flags": flags, "raw_materials": False}
        if any(result.get("producer") != expected_producer or result.get("worker_contract") != expected_worker for result in (policy, teacher)):
            raise ProvenanceError("clone evaluation worker differs from the actual frozen producer")
        rows.append({"seed": seed, "withheld": 0, "warmup": 0, "snapshot": "",
                     "policy_welfare": policy["welfare"], "heuristic_welfare": teacher["welfare"]})
    result = {"schema": 1, "kind": "clone_evaluation", "checkpoint_sha256": initial_sha,
              "model_semantics": semantic_hash(state), "movement": True, "region_head": "hierarchical",
              "simulator": frozen["simulator"], "bridge_sha256": frozen["bridge_sha256"],
              "flags": {k: frozen[k] for k in ("sim_rules", "sim_statics")}, "worlds": rows,
              "ratio": sum(r["policy_welfare"] for r in rows) / sum(r["heuristic_welfare"] for r in rows)}
    if checkpoint_hash(path) != result["checkpoint_sha256"]: raise ProvenanceError("candidate changed during evaluation")
    validate_report(result, result["checkpoint_sha256"], result["model_semantics"], frozen)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    args = parser.parse_args()
    from .protocol import enforce_qualification_execution
    enforce_qualification_execution(args, "anthropocene_rl.autonomous_v7.clone_evaluate")
    if args.out.exists(): raise ProvenanceError("selection report already exists")
    result = evaluate(args.checkpoint, manifest(args.calibration_manifest))
    tmp = args.out.with_suffix(".tmp"); tmp.write_text(json.dumps(result, sort_keys=True)); tmp.replace(args.out); return 0


if __name__ == "__main__": raise SystemExit(main())
