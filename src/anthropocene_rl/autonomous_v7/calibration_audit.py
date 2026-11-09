"""Exact held-out payload binding and central, protocol/candidate-scoped one-shot claims."""

import json
import re
from pathlib import Path

from .protocol import scan
from .provenance import ProvenanceError, checkpoint_hash, digest


def freeze(roots, checkpoint_sha, held, cohort):
    from . import branch_shards

    branches = []
    for directory in branch_shards.directories(roots):
        meta = json.loads((directory / "meta.json").read_text())
        payload = {
            p.relative_to(directory).as_posix(): checkpoint_hash(p)
            for p in sorted(directory.rglob("*.npy"))
        }
        branches.append(
            {
                "trajectory_id": meta["trajectory_id"],
                "meta_sha256": checkpoint_hash(directory / "meta.json"),
                "payload_digest": digest(payload),
            }
        )
    body = {
        "schema": 1,
        "protocol_digest": held.get("protocol_digest"),
        "checkpoint_sha256": checkpoint_sha,
        "cohort": cohort,
        "branches": sorted(branches, key=lambda row: row["trajectory_id"]),
    }
    body["digest"] = digest(body)
    inventory(roots, checkpoint_sha, held, body)
    return body


def inventory(roots, checkpoint_sha, held, frozen):
    from . import branch_shards

    scan(frozen)
    if not re.fullmatch(r"[0-9a-f]{64}", str(held.get("protocol_digest", ""))):
        raise ProvenanceError("calibration requires its pre-frozen protocol digest")
    if (
        frozen.get("schema") != 1
        or frozen.get("checkpoint_sha256") != checkpoint_sha
        or frozen.get("protocol_digest") != held.get("protocol_digest")
        or frozen.get("digest") != digest({k: v for k, v in frozen.items() if k != "digest"})
    ):
        raise ProvenanceError("calibration frozen input inventory identity mismatch")
    cohort = frozen.get("cohort", {})
    if cohort.get("role") != "calibration" or cohort.get("digest") != digest(
        {k: v for k, v in cohort.items() if k != "digest"}
    ):
        raise ProvenanceError("calibration full cohort missing/inconsistent")
    expected = {
        (
            r["seed"],
            r["withheld"],
            r["warmup"],
            r.get("bank_entry", -1),
            r.get("snapshot_sha256") or None,
        )
        for r in cohort.get("worlds", [])
    }
    if not expected:
        raise ProvenanceError("calibration cohort empty")
    actual, worlds = [], set()
    for directory in branch_shards.directories(roots):
        meta = json.loads((directory / "meta.json").read_text())
        scan(meta)
        source = meta.get("source_record", {})
        if (
            source.get("source_checkpoint") != checkpoint_sha
            or source.get("search_config", {}).get("worlds_manifest") != cohort
            or meta.get("calibration_manifest_digest") != held["manifest_digest"]
            or not meta.get("heldout_calibration")
            or not meta.get("calibration")
        ):
            raise ProvenanceError(
                "calibration shard producer/cohort/exclusions differ from audited candidate"
            )
        payload = {
            p.relative_to(directory).as_posix(): checkpoint_hash(p)
            for p in sorted(directory.rglob("*.npy"))
        }
        if not payload or digest(payload) != meta.get("data_digest"):
            raise ProvenanceError("calibration payload missing/mutated")
        actual.append(
            {
                "trajectory_id": meta["trajectory_id"],
                "meta_sha256": checkpoint_hash(directory / "meta.json"),
                "payload_digest": digest(payload),
            }
        )
        worlds.add(
            (
                meta["seed"],
                meta["withheld"],
                meta["warmup"],
                meta.get("bank_entry", -1),
                meta.get("snapshot_hash") or None,
            )
        )
    actual.sort(key=lambda row: row["trajectory_id"])
    if (
        len({r["trajectory_id"] for r in actual}) != len(actual)
        or actual != frozen.get("branches")
        or worlds != expected
    ):
        raise ProvenanceError(
            "calibration input subset/extra/duplicate differs from full frozen "
            "cohort/payload inventory"
        )
    return {
        "protocol_digest": held["protocol_digest"],
        "checkpoint_sha256": checkpoint_sha,
        "cohort_digest": cohort["digest"],
        "payload_inventory_digest": frozen["digest"],
    }


def claim(held, binding):
    if not re.fullmatch(r"[0-9a-f]{64}", str(held.get("protocol_digest", ""))) or not held.get(
        "audit_registry"
    ):
        raise ProvenanceError(
            "calibration requires frozen protocol digest and central audit_registry"
        )
    root = Path(held["audit_registry"])
    if not root.is_absolute():
        raise ProvenanceError("central calibration registry must be absolute and frozen")
    root.mkdir(parents=True, exist_ok=True)
    key = digest(binding)
    # The protocol/candidate lock also forbids changing the payload to obtain a new one-shot key.
    lock = root / (
        digest({k: binding[k] for k in ("protocol_digest", "checkpoint_sha256")}) + ".lock.json"
    )
    try:
        with lock.open("x") as stream:
            json.dump(dict(binding, audit_key=key), stream, sort_keys=True)
    except FileExistsError as error:
        raise ProvenanceError(
            "one-shot calibration already claimed for protocol/candidate; changing output "
            "or inputs is forbidden"
        ) from error
    path = root / (key + ".json")
    with path.open("x") as stream:
        json.dump(dict(binding, audit_key=key, status="claimed"), stream, sort_keys=True)
    return {"path": str(path), "sha256": checkpoint_hash(path)}
