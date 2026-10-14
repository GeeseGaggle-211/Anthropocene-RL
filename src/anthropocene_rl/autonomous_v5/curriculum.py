"""Require the independent schema/action-face audit before using fresh resets."""

import hashlib
import json


def load_curriculum(bank, audit_path, *, partition="train", allow_heldout=False):
    if partition not in ("train", "development", "heldout"):
        raise ValueError("Unknown curriculum partition")
    if partition == "heldout" and not allow_heldout:
        raise ValueError("HELDOUT is reserved for a separately frozen evaluation gate")
    manifest_path = bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (
        manifest.get("version") != "autonomous-history-fresh-v5"
        or manifest["status"] not in ("PASS", "PASS_WITH_GAPS")
        or audit.get("version") != "autonomous-history-bank-audit-v5"
        or audit.get("status") != "PASS"
        or audit["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        or audit["native_steps"] != 0
        or audit["model_calls"] != 0
    ):
        raise ValueError("Completed hash-matched native schema audit required")
    checked = {e["id"]: e for e in audit["entries"]}
    if set(checked) != {e["id"] for e in manifest["entries"]}:
        raise ValueError("Audit does not cover the complete bank")
    selected = []
    for source in manifest["entries"]:
        check = checked[source["id"]]
        if source["partition"] != partition or not check["neural_training_eligible"]:
            continue
        if not source["neural_training_eligible"] or check["partition"] != partition:
            raise ValueError("Manifest/audit eligibility or partition mismatch")
        entry = dict(source)
        entry["peer"] = next((c for c in check["controlled"] if c != entry["focal"]), None)
        selected.append(entry)
    if not selected:
        raise ValueError("No audited core curriculum states in this partition")
    return selected
