"""One immutable qualified 3x5 submission binds every formal checkpoint/completion."""
import json
from .protocol import artifact, execution_design, verify_preregistration
from .provenance import ProvenanceError, digest


def load(ref, base):
    path, _ = artifact(ref, base); qualified = json.loads(path.read_text())
    if qualified.get("schema") != 1 or qualified.get("kind") != "qualified_submission" or qualified.get("status") != "pass":
        raise ProvenanceError("formal reports require an immutable qualified submission receipt")
    spec_path, spec_sha = artifact(qualified.get("submission"), path.parent); spec = json.loads(spec_path.read_text())
    from .qualification import validate_qualified_submission
    try: validated = validate_qualified_submission(spec, spec_path.parent, check_live=False)
    except (KeyError, TypeError, OSError) as error:
        raise ProvenanceError("qualified status shell lacks reopened semantic qualification evidence") from error
    if {k: v for k, v in qualified.items() if k != "submission"} != validated:
        raise ProvenanceError("qualified receipt differs from shared semantic revalidation")
    if (spec.get("digest") != digest({k: v for k, v in spec.items() if k != "digest"})
            or qualified.get("submission_digest") != spec["digest"] or qualified.get("protocol_digest") != spec.get("protocol_digest")
            or qualified.get("initialization") != spec.get("initialization")):
        raise ProvenanceError("qualified submission/protocol/initialization binding differs")
    if not execution_design(spec["execution"])["readiness_bearing"]: raise ProvenanceError("formal cohort is not the reviewed VM 3x5 design")
    artifact(spec["initialization"], spec_path.parent)
    protocol_ref = spec["gates"]["protocol"]
    protocol_path, _ = artifact(protocol_ref, spec_path.parent); protocol_gate = json.loads(protocol_path.read_text())
    approved = verify_preregistration(protocol_gate["evidence"]["preregistration"], protocol_gate["evidence"]["review"], protocol_path.parent)
    if approved["digest"] != spec["protocol_digest"]: raise ProvenanceError("formal protocol differs from approved preregistration")
    from .qualification import GATES
    if set(spec.get("gates", {})) != set(GATES) or set(qualified.get("gates", {})) != set(GATES):
        raise ProvenanceError("qualified formal submission lacks complete gate receipts")
    for kind, gate_ref in spec["gates"].items():
        gate_path, _ = artifact(gate_ref, spec_path.parent); body = json.loads(gate_path.read_text())
        if qualified["gates"][kind] != digest(body): raise ProvenanceError("qualified gate receipt changed")
        for output in body.get("outputs", []): artifact(output, gate_path.parent)
    runs = spec.get("runs", []); frozen = qualified.get("frozen_runs", {})
    required = {(arm, seed) for arm in ("R3def", "R3per", "R3lo") for seed in range(170, 175)}
    if len(runs) != 15 or {(r["arm"], r["seed"]) for r in runs} != required or set(frozen) != {r["phase"] for r in runs}:
        raise ProvenanceError("qualified formal cohort lacks exact fifteen run identities")
    common, banks, cohorts, result = [], set(), {}, {}
    for row in runs:
        entry = frozen[row["phase"]]; run = entry["run_manifest"]
        config_path, _ = artifact(row["resolved_config"], spec_path.parent); config = json.loads(config_path.read_text())
        options = {key: value for key, value in config["options"].items() if key != "resume"}
        options["region_head"] = "hierarchical"; options["cost_bootstrap"] = True
        if (entry.get("arm") != row["arm"] or entry.get("seed") != row["seed"]
                or entry.get("run_manifest_digest") != digest(run) or run.get("options") != options
                or run.get("resolved_hyper") != config["resolved_hyper"] or run.get("resolved_stage") != config["resolved_stage"]
                or run.get("effective_iterations") != {"critic_only": 5, "actor": 50, "total": 55}
                or run.get("inputs", {}).get("init") != spec["initialization"]["sha256"]
                or run.get("inputs", {}).get("execution_manifest") != spec_sha
                or run.get("inputs", {}).get("bank_image") != row["bank_image_sha256"]
                or run.get("code") != spec["learner_commit"] or run.get("simulator") != spec["simulator"]
                or run.get("bridge_sha256") != spec["bridge_sha256"]
                or run.get("worker_contract") != qualified.get("live_probe", {}).get("worker_contract")
                or run.get("protocol_digest") != spec["protocol_digest"]):
            raise ProvenanceError("qualified run/config/root/code/binary/bank/protocol binding differs")
        excluded = {"arm", "seed", "out", "daughter_bank", "personality", "teacher_kl", "search_labels", "branch_shards"}
        hyper = {k: v for k, v in config["resolved_hyper"].items() if k not in ("personality", "teacher_kl")}
        common.append(digest({"options": {k: v for k, v in options.items() if k not in excluded}, "hyper": hyper,
                              "stage": config["resolved_stage"]}))
        banks.add(config["bank_image_digest"]); cohorts.setdefault(row["arm"], set()).add(row["search_cohort_digest"])
        result[(row["arm"], row["seed"])] = entry["run_manifest_digest"]
    if len(set(common)) != 1 or len(banks) != 1 or any(len(v) != 1 for v in cohorts.values()) or cohorts["R3per"] != cohorts["R3lo"]:
        raise ProvenanceError("formal cohort mixes non-treatment configuration/banks/search cohorts")
    return qualified, result
