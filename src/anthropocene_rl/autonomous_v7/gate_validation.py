"""Gate-specific content verification. A status/pass shell is never qualification evidence."""
import json
import math
import zlib
from pathlib import Path
import numpy as np
from .protocol import artifact, scan, ROLES, search_cohort, verify_inventory
from .provenance import ProvenanceError, checkpoint_hash, digest, compatible_sources


def read(ref, base):
    path, _ = artifact(ref, base)
    obj = json.loads(path.read_text()); scan(obj)
    return obj, path


class FrozenBank:
    def __init__(self, ref, base):
        self.index, _ = artifact(ref, base)
    def entries(self): return [json.loads(x) for x in self.index.read_text().splitlines() if x.strip()]
    def snapshot(self, entry): return zlib.decompress((self.index.parent / entry["file"]).read_bytes())


def search_evidence(evidence, expected, held, base, *, calibration=False):
    from .search_teacher import SearchConfig
    from . import branch_shards, ppo
    cohort, _ = read(evidence["cohort"], base)
    summary, summary_path = read(evidence["summary"], base)
    source = summary.get("provenance", {})
    config = SearchConfig(**summary["config"])
    validate_search_producer(source, summary, cohort, summary_path.parent)
    bank = FrozenBank(evidence["bank_index"], base) if evidence.get("bank_index") else None
    search_cohort(Path(artifact(evidence["cohort"], base)[0]), bank, config, held)
    if (source.get("source_checkpoint") != expected["initialization_sha256"]
            or source.get("search_config", {}).get("worlds_manifest") != cohort
            or source.get("search_config", {}).get("bridge_sha256") != expected["bridge_sha256"]
            or summary.get("worlds") != len(cohort["worlds"])):
        raise ProvenanceError("search summary is not the exact promoted candidate/cohort")
    roots = [artifact(ref, base)[0].parent for ref in evidence["branch_indexes"]]
    report = {}
    rows, boot = branch_shards.load(roots, "reject", report, purpose="analysis" if calibration else "training", calibration_manifest=held)
    actual = {(m["seed"], m["withheld"], m["warmup"], m.get("bank_entry", -1), m.get("snapshot_hash") or None) for m in report["branches"]}
    frozen = {(m["seed"], m["withheld"], m["warmup"], m.get("bank_entry", -1), m.get("snapshot_sha256") or None) for m in cohort["worlds"]}
    if (actual != frozen or summary.get("branches") != len(report["branches"])
            or any(m.get("source_record") != source for m in report["branches"])):
        raise ProvenanceError("search completion is a subset/extra or has inconsistent producer payloads")
    compatible_sources(report["records"], expected["simulator"], flags=expected["flags"])
    labels = artifact(evidence["labels"], base)[0]
    if not calibration:
        loaded = {}; ppo.load_search_labels([labels], 0., provenance="reject", report=loaded, calibration_manifest=held)
        if any(record != source for record in loaded["records"]): raise ProvenanceError("search labels use a different producer")
    else:
        with np.load(labels) as contents:
            if not len(contents["heldout_calibration"]) or not contents["heldout_calibration"].all():
                raise ProvenanceError("calibration label file is not excluded from training")
    return cohort, roots, report


def calibration_evidence(evidence, expected, held, state, base):
    from .calibration_audit import inventory
    from . import branch_shards, branch_bias
    from .model import load_checkpoint_normalizer
    from .search_teacher import load_critic
    cohort, roots, provenance = search_evidence(evidence, expected, held, base, calibration=True)
    inputs, _ = read(evidence["inputs"], base)
    binding = inventory(roots, expected["initialization_sha256"], held, inputs)
    report, _ = read(evidence["report"], base)
    claim, claim_path = read(report.get("audit_claim"), base)
    if any(claim.get(k) != v for k, v in binding.items()) or claim.get("audit_key") != digest(binding):
        raise ProvenanceError("calibration central claim does not bind the frozen audit")
    registry = Path(held["audit_registry"]).resolve()
    if claim_path.parent != registry or claim_path.name != digest(binding) + ".json":
        raise ProvenanceError("calibration claim is outside its frozen central registry")
    lock = registry / (digest({k: binding[k] for k in ("protocol_digest", "checkpoint_sha256")}) + ".lock.json")
    if json.loads(lock.read_text()) != dict(binding, audit_key=digest(binding)):
        raise ProvenanceError("calibration protocol/candidate lock missing/inconsistent")
    cp, _ = artifact(evidence["checkpoint"], base)
    if checkpoint_hash(cp) != expected["initialization_sha256"]: raise ProvenanceError("calibration evaluated another checkpoint")
    normalizer = load_checkpoint_normalizer(state); critic = load_critic(cp)
    rows, boot = branch_shards.load(roots, "reject", purpose="analysis")
    pred, real = branch_bias.realised(critic, rows, boot, normalizer)
    endpoints = branch_shards.endpoint_values(critic, boot, normalizer)
    actual = branch_bias.evaluate(pred, real, rows, endpoints, True, [])
    for key in ("pass", "mandatory_pass", "failures", "unwaived_coverage", "waivers", "calibration", "coverage"):
        if report.get(key) != actual[key]: raise ProvenanceError("calibration verdict/statistics differ from actual held-out payloads")
    if not actual["pass"] or report.get("audit_binding") != binding:
        raise ProvenanceError("calibration numerical/coverage/empty-waiver gate failed")
    return binding


def clone_evidence(evidence, expected, held, state, base):
    from .promotion import promote, semantic_hash
    promoted = artifact(evidence["promoted"], base)[0]
    if checkpoint_hash(promoted) != expected["initialization_sha256"]: raise ProvenanceError("clone gate names a different promotion")
    inputs = []
    for row in evidence.get("candidates", []):
        inputs.append({k: {"path": str(artifact(row[k], base)[0]), "sha256": row[k]["sha256"]} for k in ("checkpoint", "report")})
    rebuilt = promote(inputs, held)
    if (rebuilt["promotion"] != state["promotion"] or semantic_hash(rebuilt) != semantic_hash(state)
            or rebuilt["family_lineage"] != state["family_lineage"]):
        raise ProvenanceError("clone gate cannot reproduce the selected all-ten promotion")
    if any({k: c["frozen_run"].get(k) for k in expected["flags"]} != expected["flags"] for c in state["promotion"]["candidates"]):
        raise ProvenanceError("clone candidates use different frozen simulator flags")
    return digest(rebuilt["promotion"])


def _finite(value): return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_search_producer(source, summary, cohort, base=None):
    declared = source.get("search_config", {})
    effective = dict(summary["config"])
    from .calibration import manifest
    raw = effective.get("calibration_manifest")
    if isinstance(raw, str) and raw:
        path = Path(raw); raw = path if path.is_absolute() else (base or Path.cwd()) / path
    effective["calibration_manifest"] = manifest(raw) if raw else {}
    if (declared.get("region_head") != "hierarchical" or declared.get("worlds_manifest") != cohort
            or {key: declared.get(key) for key in effective} != effective):
        raise ProvenanceError("search producer head/config differs from frozen summary/cohort")


def validate_training_cohorts(cohorts):
    # Role, personality and their consequent digest differ; every other declaration must match exactly.
    projections = [{key: value for key, value in cohort.items() if key not in ("role", "personality", "digest")}
                   for cohort in cohorts.values()]
    if len(projections) != 2 or projections[0] != projections[1]:
        raise ProvenanceError("training search cohorts must share identical worlds/snapshots/extra-root declarations")


def comparator(ref, base):
    body, path = read(ref, base)
    if body.get("schema") != 1 or body.get("kind") != "historical_comparator" or not body.get("comparisons"):
        raise ProvenanceError("historical comparator lacks actual paired metrics")
    for comparison in body["comparisons"]:
        if comparison.get("status") == "insufficient":
            if not comparison.get("reason") or not comparison.get("inputs"):
                raise ProvenanceError("insufficient comparator must identify its missing evidence and actual available inputs")
            for source in comparison["inputs"]: artifact(source, path.parent)
            continue
        a, _ = read(comparison["a"], path.parent); b, _ = read(comparison["b"], path.parent)
        aa, bb = a.get("per_seed", []), b.get("per_seed", [])
        if (not aa or len({r["seed"] for r in aa}) != len(aa) or {r["seed"] for r in aa} != {r["seed"] for r in bb}
                or len(bb) != len(aa) or any(not _finite(r.get("welfare")) for r in aa + bb)):
            raise ProvenanceError("comparator missing/duplicate/nonfinite paired per-world welfare")
        denominator = sum(r["welfare"] for r in aa)
        if denominator <= 0 or comparison.get("welfare_ratio") != sum(r["welfare"] for r in bb) / denominator:
            raise ProvenanceError("comparator metrics differ from content-bound source reports")
    return body


def movement_evidence(evidence, expected, base):
    from .movement_audit_summary import world, reduce
    groups = {name: [] for name in ("grouped", "native", "stay")}
    for ref in evidence.get("rollouts", []):
        raw, path = read(ref, base)
        producer = raw.get("producer", {})
        if (raw.get("policy") not in groups or raw.get("checkpoint_sha256") != expected["initialization_sha256"]
                or producer.get("simulator") != expected["simulator"] or producer.get("bridge_sha256") != expected["bridge_sha256"]
                or producer.get("flags") != expected["flags"] or producer.get("raw_materials") is not False
                or raw.get("turns") != 25 or not raw.get("replay") or not raw.get("rows", [])[-1].get("final")):
            raise ProvenanceError("movement rollout missing controls/complete replay/producer identity")
        groups[raw["policy"]].append(world(path))
    seed_groups = [{row["seed"] for row in group} for group in groups.values()]
    if not seed_groups[0] or any(group != seed_groups[0] for group in seed_groups) or any(len(g) != len({r["seed"] for r in g}) for g in groups.values()):
        raise ProvenanceError("movement controls must have complete identical paired world families")
    summary = reduce(groups["grouped"])
    metrics = {k: summary[k] for k in ("physical_move_share", "move_share_settled", "move_share_nomadic", "welfare", "welfare_e1")}
    metrics["capital_per_move"] = {k: summary["capital_per_move"][k] for k in ("n", "fixed_left", "fixed_waiting", "field_capital", "local_capital")}
    metrics.update({key: {k: summary[key][k] for k in ("n", "land", "fish", "herd", "total")}
                    for key in ("harvest_fresh", "harvest_steady")})
    def check(node):
        if isinstance(node, dict): return all(check(v) for v in node.values())
        return _finite(node)
    if not check(metrics) or any(metrics[k]["n"] <= 0 for k in ("capital_per_move", "harvest_fresh", "harvest_steady")):
        raise ProvenanceError("movement audit lacks non-vacuous settled/nomadic/capital/fresh/steady measurements")
    forks, _ = read(evidence["forks"], base)
    pairs = forks.get("pairs", [])
    if not pairs or any(p.get("checkpoint_sha256") != expected["initialization_sha256"]
                        or p.get("producer", {}).get("simulator") != expected["simulator"]
                        or p.get("producer", {}).get("bridge_sha256") != expected["bridge_sha256"]
                        or p.get("producer", {}).get("flags") != expected["flags"] or p.get("producer", {}).get("raw_materials") is not False
                        or not all(_finite(p.get(side, {}).get(k)) for side in ("move", "stay")
                                   for k in ("welfare", "moves", "land", "produced")) for p in pairs):
        raise ProvenanceError("movement fork evidence incomplete/unbound")
    judgment, _ = read(evidence["judgment"], base)
    if (judgment.get("kind") != "movement_nondegeneracy_review" or judgment.get("nondegenerate") is not True
            or not judgment.get("reason") or judgment.get("metrics_digest") != digest(metrics)
            or judgment.get("forks_digest") != digest(pairs)):
        raise ProvenanceError("movement nondegeneracy judgment does not bind the actual metrics/controls")
    for ref in evidence.get("integration_reports", []):
        report, _ = read(ref, base)
        if report.get("failed") != 0 or not report.get("passed", 0): raise ProvenanceError("integration regression evidence failed/vacuous")
    if not evidence.get("integration_reports"): raise ProvenanceError("movement gate lacks integration regression evidence")
    return digest(metrics)


def gate(gate_name, body, expected, held, state, base, spec):
    evidence = body.get("evidence")
    if not isinstance(evidence, dict): raise ProvenanceError("gate needs typed semantic evidence, not a pass shell")
    required = []
    def refs(node):
        if isinstance(node, dict):
            if "path" in node and "sha256" in node: required.append(artifact(node, base)[1])
            else:
                for value in node.values(): refs(value)
        elif isinstance(node, list):
            for value in node: refs(value)
    refs(evidence)
    if not set(required) <= {artifact(ref, base)[1] for ref in body.get("outputs", [])}:
        raise ProvenanceError("gate done/output inventory omits actual typed evidence artifacts")
    if gate_name == "clone": return clone_evidence(evidence, expected, held, state, base)
    if gate_name == "calibration": return calibration_evidence(evidence, expected, held, state, base)
    if gate_name == "movement-integration": return movement_evidence(evidence, expected, base)
    if gate_name == "source":
        path, _ = artifact(evidence["inventory"], base); verified = verify_inventory(path)
        from .teacher_data import verify_dataset, validate_source
        datasets = []
        for ref in evidence.get("datasets", []):
            path, _ = artifact(ref, base); obj = verify_dataset(path.parent)
            validate_source(obj["source_record"], live=expected["simulator"], binary_sha256=expected["bridge_sha256"])
            if obj["source_record"]["flags"] != expected["flags"]: raise ProvenanceError("teacher input flags differ from frozen live flags")
            datasets.append(obj["dataset_digest"])
        if not datasets: raise ProvenanceError("source gate needs actual regenerated teacher datasets")
        inherited = {d["dataset_digest"] for parent in state["family_lineage"]["parents"]
                     for proof in parent["proof"]["family_lineage"]["teacher_inputs"].values() for d in proof["datasets"]}
        if set(datasets) != inherited: raise ProvenanceError("source gate datasets differ from all clone input closures")
        return verified["digest"]
    if gate_name == "search":
        if set(evidence) != {"training-default", "training-dirichlet"}: raise ProvenanceError("search needs the exact two training cohorts")
        cohorts = {name: search_evidence(item, expected, held, base)[0] for name, item in evidence.items()}
        validate_training_cohorts(cohorts)
        return {name: cohort["digest"] for name, cohort in cohorts.items()}
    if gate_name == "protocol":
        from .protocol import verify_preregistration
        protocol = verify_preregistration(evidence["preregistration"], evidence["review"], base)
        protocol_path, _ = artifact(evidence["preregistration"], base)
        approval, _ = read(evidence["review"], base)
        if (protocol.get("schema") != 1 or protocol.get("kind") != "round3_preregistration" or protocol.get("waivers") != []
                or protocol.get("seed_roles") != ROLES or protocol.get("digest") != expected["protocol_digest"]
                or approval.get("decision") != "APPROVE" or approval.get("protocol_digest") != protocol["digest"]
                or approval.get("preregistration_sha256") != checkpoint_hash(protocol_path)
                or approval.get("reviewer") != "Astra"):
            raise ProvenanceError("protocol receipt does not contain the frozen reviewed preregistration")
        if protocol.get("clone_protocol_sha256") != spec.get("clone_protocol_sha256"):
            raise ProvenanceError("reviewed preregistration names a different guarded-clone cohort")
        from .promotion import SELECTOR
        if (protocol.get("selector") != SELECTOR or protocol.get("execution") != spec.get("execution")
                or any(protocol.get(key) != expected[key] for key in ("learner_commit", "simulator", "bridge_sha256", "flags"))
                or not protocol.get("hypotheses") or not protocol.get("resource_envelope")
                or protocol.get("budgets") != {"critic_only": 5, "actor": 50, "worlds": 12, "turns": 25}
                or protocol.get("statistics") != {"draws": 4000, "seed": 0, "paired": True, "band": [.8, 1.2],
                    "separation": "both-differences-exceed-across-seed-sample-sd", "required_seeds": 4}):
            raise ProvenanceError("protocol lacks frozen hypotheses/selector/budgets/statistics/resource envelope")
        return protocol["digest"]
    if gate_name == "submission":
        # Exact job/argv/config validation is performed by qualification.validate_runs, not a self-attested gate.
        if evidence.get("queue_job") != spec.get("queue_job"): raise ProvenanceError("submission gate names another queue manifest")
        artifact(evidence["queue_job"], base)
        return evidence["queue_job"]["sha256"]
    raise ProvenanceError("unknown qualification gate")
