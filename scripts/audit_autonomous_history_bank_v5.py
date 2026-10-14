"""Native reset/schema audit of the regenerated bank; no policy or world steps."""

import argparse
import collections
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from anthropocene_rl.autonomous_v5.collection import DeadlineHistoryTransport
from anthropocene_rl.autonomous_v5.contracts import HISTORY_NAMES, features, observation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] not in ("PASS", "PASS_WITH_GAPS"):
        raise ValueError("Generation must be completed before auditing")
    if tuple(
        manifest[k]
        for k in ("action_version", "observation_version", "environment_version", "replay_version")
    ) != (3, 5, 5, 5):
        raise ValueError("Bank schema mismatch")
    entries = manifest["entries"]
    by_id = {entry["id"]: entry for entry in entries}
    if len(by_id) != len(entries):
        raise ValueError("Duplicate snapshot IDs")
    for key in ("world_seed", "lineage_id", "counterfactual_group_id", "sha256"):
        partitions = collections.defaultdict(set)
        for entry in entries:
            partitions[entry[key]].add(entry["partition"])
        if any(len(p) != 1 for p in partitions.values()):
            raise ValueError(f"Partition leakage via {key}")
    unique_unavailable = {(u["id"], u["decision_index"]) for u in manifest["unavailable"]}
    if len(entries) + len(unique_unavailable) != 576:
        raise ValueError("Incomplete frozen endpoint accounting")
    start = time.monotonic()
    result = dict(
        version="autonomous-history-bank-audit-v5",
        status="RUNNING",
        manifest_sha256=sha(manifest_path),
        bridge_sha256=sha(args.bridge),
        audit_script_sha256=sha(Path(__file__)),
        native_steps=0,
        replay_resolves=0,
        model_calls=0,
        optimizer_updates=0,
        heldout_neural_use=0,
        raw_unavailable_rows=len(manifest["unavailable"]),
        unique_unavailable_endpoints=len(unique_unavailable),
        entries=[],
        reset_requests=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    command = ["/workspace/Anthropocene/.tools/dotnet/dotnet", str(args.bridge.resolve())]
    try:
        with DeadlineHistoryTransport(command, deadline=start + 600) as transport:
            schema = transport.schema()
            if tuple(schema["historyFeatures"]) != HISTORY_NAMES or schema["featureWidth"] != 1061:
                raise ValueError("Native history layout mismatch")
            for entry in entries:
                path = args.bank / entry["snapshot"]
                blob = path.read_bytes()
                if sha(path) != entry["sha256"] or not gzip.decompress(blob).startswith(
                    b"\x0fANTHROPOCENE-12"
                ):
                    raise ValueError(f"Snapshot digest/envelope mismatch: {entry['id']}")
                ids = [entry["focal"]] + ([entry["peer"]] if entry["peer"] is not None else [])
                result["reset_requests"] += 1
                try:
                    reset = transport.reset_world(blob, ids, 25)
                except RuntimeError as error:
                    # An extinct original peer is never resurrected. Retry the
                    # surviving focal only and retain this enrollment correction.
                    if (
                        len(ids) != 2
                        or str(error) != "ArgumentException: Controlled camp lacks active economy."
                    ):
                        raise
                    ids = ids[:1]
                    result["reset_requests"] += 1
                    reset = transport.reset_world(blob, ids, 25)
                if reset["fingerprint"] != entry["fingerprint"]:
                    raise ValueError(f"Reset changed persistent state: {entry['id']}")
                if reset["configurationHash"] != entry["configuration_hash"]:
                    raise ValueError("Configuration hash mismatch")
                obs = observation(reset["observations"][str(entry["focal"])])
                if not np.isfinite(features(obs)).all():
                    raise ValueError("Nonfinite fixed feature encoding")
                age = 20 * entry["decision_index"]
                if obs["History"][-1] != age or obs["History"][0] != (age > 0):
                    raise ValueError("Persistent history/presence mismatch")
                expected = -np.expm1(-age / np.array([20, 60, 180]))
                if not np.allclose(obs["Context"][117:120], expected, atol=1e-7, rtol=0):
                    raise ValueError("EMA maturity mismatch")
                if not np.array_equal(
                    obs["WorkforceMask"], entry["workforce_mask"]
                ) or not np.array_equal(obs["ControlMask"], entry["control_mask"]):
                    raise ValueError("Current native masks differ from manifest")
                eligible = (
                    not obs["WorkforceMask"][[2, 3]].any() and not obs["ControlMask"][[1, 2]].any()
                )
                eligible = eligible and entry["family"] not in ("husbandry", "fishing")
                if eligible != entry["neural_training_eligible"]:
                    raise ValueError("Eligibility does not follow actual action faces")
                result["entries"].append(
                    dict(
                        id=entry["id"],
                        partition=entry["partition"],
                        decision_index=entry["decision_index"],
                        neural_training_eligible=eligible,
                        controlled=ids,
                        peer_enrollment_changed=ids != [entry["focal"], entry["peer"]]
                        if entry["peer"] is not None
                        else False,
                        fingerprint=reset["fingerprint"],
                        schema_history_masks="PASS",
                    )
                )
        result["status"] = "PASS"
        result["wall_seconds"] = time.monotonic() - start
        result["partitions"] = dict(collections.Counter(e["partition"] for e in entries))
        result["eligible"] = dict(
            collections.Counter(
                e["partition"] for e in result["entries"] if e["neural_training_eligible"]
            )
        )
        result["train_eligible_by_decision"] = dict(
            collections.Counter(
                e["decision_index"]
                for e in result["entries"]
                if e["partition"] == "train" and e["neural_training_eligible"]
            )
        )
        save()
        print(json.dumps({k: v for k, v in result.items() if k != "entries"}))
    except Exception as error:
        result["status"] = "FAILED"
        result["error"] = str(error)
        result["wall_seconds"] = time.monotonic() - start
        save()
        raise


if __name__ == "__main__":
    main()
