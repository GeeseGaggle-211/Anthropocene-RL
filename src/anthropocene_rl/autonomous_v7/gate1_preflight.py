"""Read-only Gate 1 retrieval inventory and historical-answerability preflight.

The preflight deliberately stops before evaluation.  It validates an exact
historical endpoint inventory, rehashes every declared byte stream, inspects
checkpoint metadata without upgrading it, and reports which historical
questions the recovered bytes can actually answer.  A successful preflight is
only a candidate for the independently reviewed production freeze.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from pathlib import Path
from typing import Any

import torch

from . import economic_diagnostics
from .initialization_lineage import inspect_checkpoint
from .protocol import scan
from .provenance import ProvenanceError, checkpoint_hash, digest

SCHEMA = 1
KIND = "gate1_retrieval_inventory"
REPORT_KIND = "gate1_retrieval_preflight_report"
BASELINE_COMMIT = "2a6565e46c43b425b1f9cc6d220995818d42061d"

FILE_REF_KEYS = {"path", "sha256", "size_bytes", "local_mtime_ns", "acquisition"}
ACQUISITION_KEYS = {
    "source_uri",
    "source_generation",
    "source_size_bytes",
    "source_mtime",
    "source_mtime_unavailable_reason",
    "source_etag",
    "method",
    "receipt_sha256",
}
ACQUISITION_METHODS = {"pre-existing-local", "read-only-copy", "read-only-download"}
EXECUTION = {
    "tier": "local",
    "purpose": "artifact-preflight",
    "selection_allowed": False,
    "readiness_claimed": False,
    "evaluation_authorized": False,
}
SLOTS = (
    "evaluation-checkpoint",
    "selection-checkpoint",
    "resume-state",
    "run-manifest",
    "producer-artifact-inventory",
    "args-config",
    "producer-build-provenance",
    "normalizer",
    "critic",
    "adapter-default-migration",
    "training-logs",
    "evaluation-raw",
    "sweep-raw",
    "transition-update-trace",
)
REQUIRED_ENDPOINT_SLOTS = {
    "evaluation-checkpoint",
    "run-manifest",
    "producer-artifact-inventory",
    "args-config",
    "producer-build-provenance",
    "normalizer",
}
RESUME_KEYS = {
    "actor",
    "critic",
    "optimizers",
    "iteration",
    "python_rng",
    "numpy_rng",
    "torch_rng",
    "simulator_rng",
    "dataloader_state",
    "scheduler",
    "scaler",
    "run_manifest",
}
CRITIC_FIELDS = {"prediction", "target", "tdResidual", "advantage"}
EMBEDDED_SLOT_KEYS = {
    "run-manifest": "run_manifest",
    "args-config": "run_manifest.options",
    "producer-build-provenance": "run_manifest",
    "normalizer": "normalizer",
    "critic": "critic",
}


def _expected_nodes() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {
        "prior-reference": {
            "role": "prior-reference",
            "historical_label": "stage1-seed2",
            "arm_id": None,
            "optimizer_seed": None,
            "pairing_status": "not-applicable",
            "checkpoint_contract": 8,
        },
        "guarded-clone": {
            "role": "guarded-clone",
            "historical_label": "stage1-token-s6",
            "arm_id": None,
            "optimizer_seed": None,
            "pairing_status": "not-applicable",
            "checkpoint_contract": 12,
        },
    }
    for arm, canonical, seeds in (
        ("T3per", "t3per", range(70, 75)),
        ("T3def", "t3def", range(75, 80)),
        ("T3lo", "t3lo", range(80, 85)),
    ):
        for seed in seeds:
            node_id = f"{canonical}-s{seed}"
            rows[node_id] = {
                "role": "trained-t3",
                "historical_label": f"{arm}-s{seed}",
                "arm_id": arm,
                "optimizer_seed": seed,
                "pairing_status": "unpaired",
                "checkpoint_contract": 12,
            }
    return rows


EXPECTED_NODES = _expected_nodes()


def _fail(message: str) -> None:
    raise ProvenanceError("Gate 1 retrieval preflight: " + message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _reparse(metadata: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and getattr(metadata, "st_file_attributes", 0) & flag)


def _plain_file(path: Path, what: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        _fail(f"{what} is missing or unreadable: {error}")
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink() or _reparse(metadata):
        _fail(f"{what} must be a regular non-reparse file: {path}")
    _plain_ancestors(path, what)
    return metadata


def _plain_ancestors(path: Path, what: str) -> None:
    current = path.parent
    while True:
        try:
            metadata = os.lstat(current)
        except OSError as error:
            _fail(f"{what} ancestor is missing or unreadable: {error}")
        if not stat.S_ISDIR(metadata.st_mode) or current.is_symlink() or _reparse(metadata):
            _fail(f"{what} path has a symlink/reparse ancestor: {current}")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _plain_directory(path: Path, what: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        _fail(f"{what} is missing or unreadable: {error}")
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink() or _reparse(metadata):
        _fail(f"{what} must be a regular non-reparse directory: {path}")
    _plain_ancestors(path, what)


def _sealed_json(path: Path, *, kind: str | None = None) -> dict[str, Any]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"invalid JSON evidence {path}: {error}")
    if not isinstance(body, dict):
        _fail(f"JSON evidence is not an object: {path}")
    scan(body, str(path))
    if kind is not None and (body.get("schema") != 1 or body.get("kind") != kind):
        _fail(f"{path} is not schema-1 {kind} evidence")
    if "digest" in body:
        try:
            expected = digest({key: value for key, value in body.items() if key != "digest"})
        except (TypeError, ValueError):
            _fail(f"JSON evidence is not canonical: {path}")
        if body["digest"] != expected:
            _fail(f"JSON evidence digest mismatch: {path}")
    return body


def _file_ref(value: Any, base: Path, what: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != FILE_REF_KEYS:
        _fail(f"{what} must declare exact path/hash/size/source metadata")
    if (
        not isinstance(value.get("path"), str)
        or not value["path"].strip()
        or not _is_sha(value.get("sha256"))
        or isinstance(value.get("size_bytes"), bool)
        or not isinstance(value.get("size_bytes"), int)
        or value["size_bytes"] < 0
        or not isinstance(value.get("acquisition"), dict)
        or isinstance(value.get("local_mtime_ns"), bool)
        or not isinstance(value.get("local_mtime_ns"), int)
        or value["local_mtime_ns"] < 0
    ):
        _fail(f"{what} file metadata is malformed")
    acquisition = value["acquisition"]
    if (
        set(acquisition) != ACQUISITION_KEYS
        or not isinstance(acquisition.get("source_uri"), str)
        or not acquisition["source_uri"].strip()
        or acquisition.get("method") not in ACQUISITION_METHODS
        or (
            acquisition.get("source_generation") is not None
            and (
                not isinstance(acquisition["source_generation"], str)
                or not acquisition["source_generation"].strip()
            )
        )
        or (
            acquisition.get("source_size_bytes") is not None
            and (
                isinstance(acquisition["source_size_bytes"], bool)
                or not isinstance(acquisition["source_size_bytes"], int)
                or acquisition["source_size_bytes"] < 0
            )
        )
        or (
            acquisition.get("source_mtime") is not None
            and (
                not isinstance(acquisition["source_mtime"], str)
                or not acquisition["source_mtime"].strip()
            )
        )
        or (
            acquisition.get("source_mtime_unavailable_reason") is not None
            and (
                not isinstance(acquisition["source_mtime_unavailable_reason"], str)
                or not acquisition["source_mtime_unavailable_reason"].strip()
            )
        )
        or (
            acquisition.get("source_mtime") is None
            and acquisition.get("source_mtime_unavailable_reason") is None
        )
        or (
            acquisition.get("source_mtime") is not None
            and acquisition.get("source_mtime_unavailable_reason") is not None
        )
        or (
            acquisition.get("source_etag") is not None
            and (
                not isinstance(acquisition["source_etag"], str)
                or not acquisition["source_etag"].strip()
            )
        )
        or (
            acquisition.get("receipt_sha256") is not None
            and not _is_sha(acquisition["receipt_sha256"])
        )
        or (
            acquisition.get("method") != "pre-existing-local"
            and acquisition.get("receipt_sha256") is None
        )
    ):
        _fail(f"{what} acquisition provenance is malformed")
    path = Path(value["path"])
    if not path.is_absolute():
        path = base / path
    path = Path(os.path.abspath(path))
    metadata = _plain_file(path, what)
    if acquisition["method"] == "pre-existing-local":
        source_path = Path(acquisition["source_uri"])
        if (
            not source_path.is_absolute()
            or Path(os.path.abspath(source_path)) != path
            or acquisition["source_generation"] is not None
            or acquisition["source_mtime"] != str(metadata.st_mtime_ns)
            or acquisition["source_mtime_unavailable_reason"] is not None
            or acquisition["source_etag"] is not None
            or acquisition["receipt_sha256"] is not None
        ):
            _fail(f"{what} pre-existing-local provenance is inconsistent: {path}")
    before = checkpoint_hash(path)
    if (
        metadata.st_size != value["size_bytes"]
        or before != value["sha256"]
        or (
            acquisition["source_size_bytes"] is not None
            and acquisition["source_size_bytes"] != metadata.st_size
        )
    ):
        _fail(f"{what} size or SHA-256 differs: {path}")
    if (
        value["local_mtime_ns"] is not None
        and value["local_mtime_ns"] != metadata.st_mtime_ns
    ):
        _fail(f"{what} local modification time differs: {path}")
    if checkpoint_hash(path) != before:
        _fail(f"{what} changed while it was inspected: {path}")
    return path, before


def file_reference(
    path: Path,
    *,
    source_path: str | None = None,
    acquisition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create retrieval metadata for an already-local regular file.

    This convenience function does not copy, retrieve or execute the artifact.
    """
    path = Path(os.path.abspath(path))
    metadata = _plain_file(path, "retrieved artifact")
    origin = acquisition or {
        "source_uri": source_path or str(path),
        "source_generation": None,
        "source_size_bytes": metadata.st_size,
        "source_mtime": str(getattr(metadata, "st_mtime_ns", "")) or None,
        "source_mtime_unavailable_reason": None,
        "source_etag": None,
        "method": "pre-existing-local",
        "receipt_sha256": None,
    }
    value = {
        "path": str(path),
        "sha256": checkpoint_hash(path),
        "size_bytes": metadata.st_size,
        "local_mtime_ns": getattr(metadata, "st_mtime_ns", None),
        "acquisition": origin,
    }
    _file_ref(value, path.parent, "retrieved artifact")
    return value


def _hash_value(value: Any) -> str:
    """Hash a typed value with unambiguous length-delimited recursive framing."""
    def frame(tag: bytes, *parts: bytes) -> bytes:
        return b"".join(
            [len(tag).to_bytes(8, "big"), tag]
            + [len(part).to_bytes(8, "big") + part for part in parts]
        )

    def encode(item: Any) -> bytes:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu()
            if tensor.layout != torch.strided:
                _fail("embedded checkpoint values require strided tensors")
            raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            return frame(
                b"tensor",
                str(tensor.dtype).encode("utf-8"),
                _canonical(list(tensor.shape)),
                raw,
            )
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                _fail("embedded checkpoint mapping has a non-string key")
            entries = [
                frame(b"entry", key.encode("utf-8"), encode(item[key]))
                for key in sorted(item)
            ]
            return frame(b"dict", len(entries).to_bytes(8, "big"), *entries)
        if isinstance(item, list):
            return frame(
                b"list", len(item).to_bytes(8, "big"), *(encode(child) for child in item)
            )
        if isinstance(item, tuple):
            return frame(
                b"tuple", len(item).to_bytes(8, "big"), *(encode(child) for child in item)
            )
        if item is None:
            return frame(b"none")
        if isinstance(item, bool):
            return frame(b"bool", b"1" if item else b"0")
        if isinstance(item, int):
            return frame(b"int", str(item).encode("ascii"))
        if isinstance(item, float):
            if not math.isfinite(item):
                _fail("embedded checkpoint value is nonfinite")
            return frame(b"float64", struct.pack(">d", item))
        if isinstance(item, str):
            return frame(b"string", item.encode("utf-8"))
        _fail(f"unsupported embedded checkpoint value: {type(item).__name__}")

    return hashlib.sha256(encode(value)).hexdigest()


def embedded_reference(value: Any, *, container_slot: str, key: str) -> dict[str, Any]:
    """Build an exact declaration for one named checkpoint section."""
    return {
        "kind": "embedded",
        "container_slot": container_slot,
        "key": key,
        "sha256": _hash_value(value),
    }


def _embedded_value(state: dict[str, Any], key: str) -> Any:
    value: Any = state
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            _fail(f"embedded checkpoint key is absent: {key}")
        value = value[part]
    return value


def _finite_vector(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            and (not positive or float(item) > 0)
            for item in value
        )
    )


def _validate_embedded_slot(name: str, key: str, value: Any, node: dict[str, Any]) -> None:
    expected_key = EMBEDDED_SLOT_KEYS[name]
    if key != expected_key:
        _fail(
            f"node {node['id']} slot {name} must bind checkpoint section {expected_key}"
        )
    identity = node["run_identity"]
    if name == "normalizer":
        if (
            not isinstance(value, dict)
            or set(value) < {"core_mean", "core_std"}
            or not _finite_vector(value["core_mean"])
            or not _finite_vector(value["core_std"], positive=True)
            or len(value["core_mean"]) != len(value["core_std"])
        ):
            _fail(f"node {node['id']} normalizer slot has no valid moment schema")
    elif name == "critic":
        if (
            not isinstance(value, dict)
            or not value
            or not all(isinstance(key_name, str) and key_name for key_name in value)
            or not any(isinstance(item, torch.Tensor) for item in value.values())
        ):
            _fail(f"node {node['id']} critic slot has no typed state-dict schema")
    elif name == "args-config":
        if not isinstance(value, dict) or not value:
            _fail(f"node {node['id']} args-config slot is not a configuration mapping")
        run_id = identity["producer_run_id"]
        if run_id is not None and value.get("run_id") != run_id:
            _fail(f"node {node['id']} args-config does not bind the producer run")
        for field, expected in (
            ("arm", identity["arm_id"]), ("seed", identity["optimizer_seed"])
        ):
            if value.get(field) != expected:
                _fail(f"node {node['id']} args-config {field} differs")
    else:
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("options"), dict)
            or not isinstance(value.get("code"), dict)
            or not isinstance(value.get("simulator"), dict)
            or not re.fullmatch(r"[0-9a-f]{40}", str(value["code"].get("commit", "")))
            or type(value["code"].get("clean")) is not bool
            or not re.fullmatch(
                r"[0-9a-f]{40}", str(value["simulator"].get("commit", ""))
            )
            or not _is_sha(value["simulator"].get("buildHash"))
        ):
            _fail(f"node {node['id']} slot {name} has no producer/run schema")


def _missing_slot(
    slot: dict[str, Any], node_id: str, name: str, base: Path = Path.cwd()
) -> dict[str, Any]:
    status = slot.get("status")
    if status == "missing":
        if set(slot) != {"status", "reason"} or not str(slot.get("reason", "")).strip():
            _fail(f"node {node_id} slot {name} missingness is malformed")
        return {"status": status, "reason": slot["reason"]}
    if status != "not-applicable" or set(slot) != {"status", "reason", "support"}:
        _fail(f"node {node_id} slot {name} status is malformed")
    if not str(slot.get("reason", "")).strip():
        _fail(f"node {node_id} slot {name} not-applicable status needs a reason")
    support_path, support_sha = _file_ref(slot["support"], base, f"{name} support")
    support = _sealed_json(support_path, kind="gate1_historical_slot_applicability")
    if (
        support.get("node_id") != node_id
        or support.get("slot") != name
        or support.get("status") != "not-applicable"
        or not str(support.get("finding", "")).strip()
        or set(support) != {
            "schema",
            "kind",
            "node_id",
            "slot",
            "status",
            "finding",
            "source_artifacts",
            "digest",
        }
        or not isinstance(support.get("source_artifacts"), list)
        or not support["source_artifacts"]
    ):
        _fail(f"node {node_id} slot {name} applicability support is not content-backed")
    source_sha256s = []
    for index, ref in enumerate(support["source_artifacts"]):
        _, sha = _file_ref(ref, support_path.parent, f"{name} support source {index}")
        source_sha256s.append(sha)
    return {
        "status": status,
        "reason": slot["reason"],
        "review_status": "unverified",
        "support_sha256": support_sha,
        "support_source_sha256s": sorted(source_sha256s),
    }


def _generic_present(
    source: dict[str, Any], base: Path, state: dict[str, Any], node: dict[str, Any], name: str
) -> dict[str, Any]:
    node_id = node["id"]
    kind = source.get("kind")
    if name in EMBEDDED_SLOT_KEYS and kind != "embedded":
        _fail(f"node {node_id} slot {name} must use its typed checkpoint section")
    if kind == "file":
        if set(source) != {"kind", "artifact"}:
            _fail(f"node {node_id} slot {name} file source is malformed")
        path, sha = _file_ref(source["artifact"], base, f"node {node_id} slot {name}")
        if path.suffix.lower() == ".json":
            _sealed_json(path)
        return {"status": "present", "source_kind": kind, "sha256": sha}
    if kind == "embedded":
        if set(source) != {"kind", "container_slot", "key", "sha256"}:
            _fail(f"node {node_id} slot {name} embedded source is malformed")
        if source["container_slot"] != "evaluation-checkpoint" or not _is_sha(source["sha256"]):
            _fail(f"node {node_id} slot {name} has an invalid embedded container/hash")
        value = _embedded_value(state, source.get("key", ""))
        if name in EMBEDDED_SLOT_KEYS:
            _validate_embedded_slot(name, source["key"], value, node)
        actual = _hash_value(value)
        if actual != source["sha256"]:
            _fail(f"node {node_id} slot {name} embedded content hash differs")
        return {
            "status": "present",
            "source_kind": kind,
            "sha256": actual,
            "key": source["key"],
        }
    _fail(f"node {node_id} slot {name} uses an unsupported source kind")


def _trace_present(
    source: dict[str, Any], base: Path, node: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    node_id = node["id"]
    if node["run_identity"]["producer_run_id"] is None:
        _fail(f"node {node_id} trace cannot claim an unknown producer run identity")
    if (
        set(source) != {"kind", "root", "files", "snapshots", "bindings"}
        or source.get("kind") != "economic-trace"
    ):
        _fail(f"node {node_id} trace source is malformed")
    bindings = source.get("bindings")
    if (
        not isinstance(bindings, dict)
        or set(bindings) != {
            "checkpoint_sha256",
            "endpoint_actor_sha256",
            "normalizer_sha256",
            "run_manifest_sha256",
            "producer_provenance_sha256",
            "transitions_sha256",
            "updates_sha256",
        }
        or not all(_is_sha(value) for value in bindings.values())
    ):
        _fail(f"node {node_id} trace bindings are malformed")
    root = Path(source.get("root", ""))
    if not root.is_absolute():
        root = base / root
    root = Path(os.path.abspath(root))
    _plain_directory(root, f"node {node_id} economic trace root")
    files = source.get("files")
    if not isinstance(files, dict) or set(files) != {"manifest", "done", "transitions", "updates"}:
        _fail(f"node {node_id} trace files are incomplete")
    resolved: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name, ref in files.items():
        path, sha = _file_ref(ref, base, f"node {node_id} trace {name}")
        paths[name] = path
        resolved[name] = sha
    if (
        bindings["transitions_sha256"] != resolved["transitions"]
        or bindings["updates_sha256"] != resolved["updates"]
    ):
        _fail(f"node {node_id} trace bindings do not bind transition/update bytes")
    if paths["manifest"] != root / "manifest.json" or paths["done"] != root / "done.json":
        _fail(f"node {node_id} trace manifest/done paths do not bind the declared root")
    manifest_body = _sealed_json(paths["manifest"])
    if manifest_body.get("gate1Bindings") != bindings:
        _fail(f"node {node_id} trace manifest does not bind its endpoint/provenance")
    declared_files = manifest_body.get("files", {})
    transition_name = declared_files.get("transitions", "transitions.jsonl")
    update_name = declared_files.get("updates", "updates.jsonl")
    if paths["transitions"] != root / transition_name or paths["updates"] != root / update_name:
        _fail(f"node {node_id} trace payload paths differ from the manifest")
    snapshot_refs = source.get("snapshots")
    if not isinstance(snapshot_refs, list):
        _fail(f"node {node_id} trace snapshots must be an exact list")
    snapshot_hashes: list[str] = []
    for index, ref in enumerate(snapshot_refs):
        path, sha = _file_ref(ref, base, f"node {node_id} trace snapshot {index}")
        if path.parent != root / "snapshots" or path.name != f"{sha}.anthro":
            _fail(f"node {node_id} trace snapshot is not stored by digest")
        snapshot_hashes.append(sha)
    done = _sealed_json(paths["done"])
    if sorted(done.get("snapshotHashes", [])) != sorted(snapshot_hashes):
        _fail(f"node {node_id} trace snapshot inventory differs from done.json")
    try:
        _, transitions, _validated_updates = economic_diagnostics.load_traces([root])
        _, _, updates = economic_diagnostics.load_artifact(root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        _fail(f"node {node_id} economic trace is invalid: {error}")
    identity = node["run_identity"]
    for row in [*transitions, *updates]:
        if row.get("runId") != identity["producer_run_id"]:
            _fail(f"node {node_id} trace run identity differs")
        if identity["arm_id"] is not None and row.get("arm") != identity["arm_id"]:
            _fail(f"node {node_id} trace arm identity differs")
        if identity["optimizer_seed"] is not None and row.get("optimizerSeed") != identity["optimizer_seed"]:
            _fail(f"node {node_id} trace optimizer seed differs")
    return {
        "status": "present",
        "source_kind": "economic-trace",
        "file_sha256s": resolved,
        "snapshot_sha256s": sorted(snapshot_hashes),
        "bindings": bindings,
        "transition_count": len(transitions),
        "update_count": len(updates),
    }, transitions, updates


def _node_identity(node: dict[str, Any]) -> None:
    node_id = node.get("id")
    expected = EXPECTED_NODES.get(node_id)
    if expected is None:
        _fail(f"unexpected historical endpoint node: {node_id!r}")
    if set(node) != {"id", "role", "run_identity", "artifacts"}:
        _fail(f"node {node_id} has an incomplete or extra inventory field")
    identity = node.get("run_identity")
    if not isinstance(identity, dict) or set(identity) != {
        "historical_label",
        "producer_run_id",
        "arm_id",
        "optimizer_seed",
        "pairing_status",
        "pairing_group",
    }:
        _fail(f"node {node_id} run identity is malformed")
    for name in ("role", "historical_label", "arm_id", "optimizer_seed", "pairing_status"):
        actual = node.get(name) if name == "role" else identity.get(name)
        if actual != expected[name]:
            _fail(f"node {node_id} {name} differs from the reviewed historical inventory")
    producer_run_id = identity.get("producer_run_id")
    if producer_run_id is not None and (
        not isinstance(producer_run_id, str) or not producer_run_id.strip()
    ):
        _fail(f"node {node_id} producer run ID must be null or a nonempty exact identity")
    if identity.get("pairing_group") is not None:
        _fail(f"node {node_id} may not invent a historical pairing group")
    artifacts = node.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(SLOTS):
        _fail(f"node {node_id} artifact slots differ from the production protocol")


def _checkpoint_identity(node: dict[str, Any], state: dict[str, Any]) -> None:
    identity = node["run_identity"]
    seed = identity["optimizer_seed"]
    raw_seed = state.get("seed")
    if seed is not None and raw_seed is not None and raw_seed != seed:
        _fail(f"node {node['id']} checkpoint seed differs from its optimizer identity")
    run = state.get("run_manifest")
    if not isinstance(run, dict):
        return
    options = run.get("options")
    if not isinstance(options, dict):
        return
    if seed is not None and options.get("seed") is not None and options.get("seed") != seed:
        _fail(f"node {node['id']} checkpoint manifest seed differs")
    if identity["arm_id"] is not None and options.get("arm") is not None:
        if options["arm"] != identity["arm_id"]:
            _fail(f"node {node['id']} checkpoint manifest arm differs")
    if (
        identity["producer_run_id"] is not None
        and options.get("run_id") != identity["producer_run_id"]
    ):
        _fail(f"node {node['id']} checkpoint manifest run ID differs")


def _present_slot(
    slot: Any,
    base: Path,
    state: dict[str, Any],
    node: dict[str, Any],
    name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(slot, dict) or slot.get("status") != "present":
        _fail(f"node {node['id']} slot {name} is not a present record")
    if set(slot) != {"status", "source"} or not isinstance(slot.get("source"), dict):
        _fail(f"node {node['id']} slot {name} present record is malformed")
    if name == "producer-artifact-inventory":
        source = slot["source"]
        if (
            set(source) != {"kind", "artifact", "acquisition_receipt"}
            or source.get("kind") != "file"
        ):
            _fail(f"node {node['id']} producer inventory source is malformed")
        _, sha = _file_ref(
            source["artifact"], base, f"node {node['id']} producer artifact inventory"
        )
        return {
            "status": "present",
            "source_kind": "file",
            "sha256": sha,
        }, [], []
    if name == "transition-update-trace":
        descriptor, transitions, updates = _trace_present(slot["source"], base, node)
        return descriptor, transitions, updates
    return _generic_present(slot["source"], base, state, node, name), [], []


def _declared_file_refs(
    node: dict[str, Any]
) -> list[tuple[str, str, dict[str, Any]]]:
    refs: list[tuple[str, str, dict[str, Any]]] = []
    for name in SLOTS:
        if name == "producer-artifact-inventory":
            continue
        slot = node["artifacts"][name]
        if slot.get("status") != "present":
            continue
        source = slot.get("source", {})
        if source.get("kind") == "file":
            refs.append((name, "artifact", source["artifact"]))
        elif source.get("kind") == "economic-trace":
            refs.extend(
                (name, f"trace-{role}", ref)
                for role, ref in sorted(source["files"].items())
            )
            refs.extend(
                (name, f"snapshot-{index}", ref)
                for index, ref in enumerate(source["snapshots"])
            )
    return refs


def _producer_artifact_inventory(
    node: dict[str, Any], base: Path
) -> tuple[dict[str, Any], list[str]]:
    slot = node["artifacts"]["producer-artifact-inventory"]
    if slot.get("status") != "present":
        return {}, []
    if node["run_identity"]["producer_run_id"] is None:
        _fail(f"node {node['id']} producer inventory cannot claim an unknown run identity")
    source = slot.get("source", {})
    if (
        source.get("kind") != "file"
        or set(source) != {"kind", "artifact", "acquisition_receipt"}
    ):
        _fail(f"node {node['id']} producer artifact inventory must be a file source")
    inventory_ref = source.get("artifact")
    path, sha = _file_ref(
        inventory_ref, base, f"node {node['id']} producer artifact inventory"
    )
    inventory_acquisition = inventory_ref["acquisition"]
    inventory_receipt_sha: str | None = None
    if inventory_acquisition["method"] == "pre-existing-local":
        if source["acquisition_receipt"] is not None:
            _fail(f"node {node['id']} local producer inventory invents an acquisition receipt")
    else:
        receipt_ref = source["acquisition_receipt"]
        if not isinstance(receipt_ref, dict):
            _fail(f"node {node['id']} remote producer inventory lacks its acquisition receipt")
        receipt_path, inventory_receipt_sha = _file_ref(
            receipt_ref, base, f"node {node['id']} producer inventory acquisition receipt"
        )
        if inventory_receipt_sha != inventory_acquisition["receipt_sha256"]:
            _fail(f"node {node['id']} producer inventory acquisition receipt hash differs")
        receipt = _sealed_json(receipt_path, kind="gate1_read_only_acquisition_receipt")
        if (
            set(receipt) != {
                "schema",
                "kind",
                "status",
                "source",
                "destination",
                "method",
                "acquired_at",
                "tool_identity",
                "digest",
            }
            or receipt.get("status") != "complete"
            or receipt.get("source")
            != {
                "uri": inventory_acquisition["source_uri"],
                "generation": inventory_acquisition["source_generation"],
                "size_bytes": inventory_acquisition["source_size_bytes"],
                "mtime": inventory_acquisition["source_mtime"],
                "mtime_unavailable_reason": inventory_acquisition[
                    "source_mtime_unavailable_reason"
                ],
                "etag": inventory_acquisition["source_etag"],
            }
            or receipt.get("destination")
            != {
                "node_id": node["id"],
                "slot": "producer-artifact-inventory",
                "artifact_role": "artifact",
                "path": str(path),
                "sha256": sha,
                "size_bytes": inventory_ref["size_bytes"],
            }
            or receipt.get("method") != inventory_acquisition["method"]
            or not isinstance(receipt.get("acquired_at"), str)
            or not receipt["acquired_at"].strip()
            or not isinstance(receipt.get("tool_identity"), dict)
            or set(receipt["tool_identity"]) != {"name", "version"}
            or not all(
                isinstance(value, str) and value.strip()
                for value in receipt["tool_identity"].values()
            )
        ):
            _fail(f"node {node['id']} producer inventory acquisition receipt is invalid")
    body = _sealed_json(path, kind="gate1_retrieved_artifact_inventory")
    if (
        set(body) != {
            "schema",
            "kind",
            "node_id",
            "producer_run_id",
            "declared_source_root",
            "complete",
            "artifacts",
            "digest",
        }
        or body.get("node_id") != node["id"]
        or body.get("producer_run_id") != node["run_identity"]["producer_run_id"]
        or not isinstance(body.get("declared_source_root"), str)
        or not body["declared_source_root"].strip()
        or body.get("complete") is not True
        or not isinstance(body.get("artifacts"), list)
        or not body["artifacts"]
    ):
        _fail(f"node {node['id']} producer artifact inventory is incomplete")
    declared: list[str] = []
    declared_files: list[tuple[str, Path, dict[str, Any]]] = []
    identities: set[tuple[str, str]] = set()
    for index, ref in enumerate(body["artifacts"]):
        declared_path, declared_sha = _file_ref(
            ref, path.parent, f"node {node['id']} producer artifact {index}"
        )
        identity = (str(declared_path), declared_sha)
        if identity in identities:
            _fail(f"node {node['id']} producer artifact inventory repeats a file")
        identities.add(identity)
        declared.append(declared_sha)
        declared_files.append((declared_sha, declared_path, ref))
    required_bindings: dict[tuple[str, str], tuple[str, str, dict[str, Any]]] = {}
    for index, (slot_name, artifact_role, ref) in enumerate(_declared_file_refs(node)):
        required_path, required_sha = _file_ref(
            ref, base, f"node {node['id']} bound slot artifact {index}"
        )
        identity = (str(required_path), required_sha)
        if identity in required_bindings:
            other_slot, other_role, _ = required_bindings[identity]
            _fail(
                f"node {node['id']} aliases one file across {other_slot}/{other_role} "
                f"and {slot_name}/{artifact_role}"
            )
        required_bindings[identity] = (slot_name, artifact_role, ref)
        exact = [
            candidate
            for candidate in declared_files
            if candidate[0] == required_sha
            and candidate[1] == required_path
            and candidate[2] == ref
        ]
        if len(exact) != 1:
            _fail(
                f"node {node['id']} producer inventory does not exactly bind "
                f"slot {slot_name}/{artifact_role}"
            )
    for artifact_sha, artifact_path, artifact_ref in declared_files:
        acquisition = artifact_ref["acquisition"]
        receipt_sha = acquisition.get("receipt_sha256")
        if receipt_sha is None:
            continue
        target = required_bindings.get((str(artifact_path), artifact_sha))
        if target is None:
            _fail(f"node {node['id']} remote artifact is not bound to an exact slot")
        slot_name, artifact_role, _ = target
        receipt_candidates = [
            (candidate_path, candidate_ref)
            for candidate_sha, candidate_path, candidate_ref in declared_files
            if candidate_sha == receipt_sha
        ]
        if len(receipt_candidates) != 1:
            _fail(f"node {node['id']} acquisition receipt identity is ambiguous")
        receipt_path, _ = receipt_candidates[0]
        receipt = _sealed_json(receipt_path, kind="gate1_read_only_acquisition_receipt")
        if (
            set(receipt) != {
                "schema",
                "kind",
                "status",
                "source",
                "destination",
                "method",
                "acquired_at",
                "tool_identity",
                "digest",
            }
            or receipt.get("status") != "complete"
            or receipt.get("source")
            != {
                "uri": acquisition["source_uri"],
                "generation": acquisition["source_generation"],
                "size_bytes": acquisition["source_size_bytes"],
                "mtime": acquisition["source_mtime"],
                "mtime_unavailable_reason": acquisition[
                    "source_mtime_unavailable_reason"
                ],
                "etag": acquisition["source_etag"],
            }
            or receipt.get("destination")
            != {
                "node_id": node["id"],
                "slot": slot_name,
                "artifact_role": artifact_role,
                "path": str(artifact_path),
                "sha256": artifact_sha,
                "size_bytes": artifact_ref["size_bytes"],
            }
            or receipt.get("method") != acquisition["method"]
            or not isinstance(receipt.get("acquired_at"), str)
            or not receipt["acquired_at"].strip()
            or not isinstance(receipt.get("tool_identity"), dict)
            or set(receipt["tool_identity"]) != {"name", "version"}
            or not all(
                isinstance(value, str) and value.strip()
                for value in receipt["tool_identity"].values()
            )
        ):
            _fail(
                f"node {node['id']} acquisition receipt does not bind {artifact_path}"
            )
    return {
        "inventory_sha256": sha,
        "declared_artifact_count": len(declared),
        "declared_sha256s": sorted(declared),
        "authoritative_completeness": False,
    }, [sha, *([inventory_receipt_sha] if inventory_receipt_sha else []), *declared]


def _question(status: str, reasons: list[str], evidence: list[str]) -> dict[str, Any]:
    if status not in {"answerable", "blocked", "not-applicable"}:
        _fail("internal answerability status is invalid")
    return {"status": status, "reasons": reasons, "evidence_sha256s": sorted(set(evidence))}


def _critic_complete(transitions: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    if not transitions:
        return False, []
    objectives: set[str] = set()
    for row in transitions:
        values = row.get("objectives")
        if not isinstance(values, dict) or not values:
            return False, []
        for name, record in values.items():
            if not isinstance(record, dict) or not CRITIC_FIELDS <= set(record):
                return False, []
            if any(
                isinstance(record[field], bool)
                or not isinstance(record[field], (int, float))
                or not math.isfinite(float(record[field]))
                for field in CRITIC_FIELDS
            ):
                return False, []
            objectives.add(str(name))
    return True, sorted(objectives)


def _complete_update_receipts(
    updates: list[dict[str, Any]], joined: list[dict[str, Any]]
) -> tuple[bool, list[str], list[str]]:
    if not joined:
        return False, ["no complete optimizer mutation/refit join"], []
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in updates:
        event = str(row.get("event", ""))
        family = "mutation" if event.startswith("mutation-") else (
            "critic-refit" if event.startswith("critic-refit-") else None
        )
        if family is None:
            continue
        key = (
            row.get("runId"),
            row.get("arm"),
            row.get("optimizerSeed"),
            row.get("rollout"),
            row.get("epoch"),
            row.get("minibatch", 0),
            row.get("batchRowDigest"),
            family,
        )
        side = "start" if event.endswith("-start") else "complete"
        if side in pairs.setdefault(key, {}):
            return False, [f"duplicate {family} {side} receipt"], []
        pairs[key][side] = row
    if len(pairs) != len(joined):
        return False, ["raw update-receipt cardinality differs from joined updates"], []
    required_hashes = {"actor", "critic", "optimizer"}
    for key, pair in pairs.items():
        if set(pair) != {"start", "complete"}:
            return False, [f"unpaired raw optimizer receipt: {key}"], []
        start, complete = pair["start"], pair["complete"]
        before, after = start.get("beforeHashes"), complete.get("afterHashes")
        if (
            not isinstance(before, dict)
            or not isinstance(after, dict)
            or not required_hashes <= set(before)
            or not required_hashes <= set(after)
            or not all(_is_sha(before[name]) and _is_sha(after[name]) for name in required_hashes)
        ):
            return False, ["update receipt lacks exact actor/critic/optimizer pre/post hashes"], []
        before_version = start.get("policyVersionBefore")
        after_version = complete.get("policyVersionAfter")
        if (
            isinstance(before_version, bool)
            or not isinstance(before_version, int)
            or isinstance(after_version, bool)
            or not isinstance(after_version, int)
            or after_version - before_version not in {0, 1}
        ):
            return False, ["update receipt policy-version linkage is invalid"], []
        if (
            start.get("batchRowDigest") != complete.get("batchRowDigest")
            or start.get("batchTransitionIds") != complete.get("batchTransitionIds")
        ):
            return False, ["update receipt batch linkage differs across start/complete"], []
    return True, [], sorted(required_hashes)


def _trace_actor_chain(
    transitions: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    endpoint_actor_sha256: str,
) -> tuple[bool, list[str]]:
    versions: dict[int, str] = {}
    completed: list[tuple[int, str]] = []
    ordered_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pending: tuple[str, dict[str, Any]] | None = None

    for row in updates:
        event = str(row.get("event", ""))
        family = (
            "mutation"
            if event.startswith("mutation-")
            else "critic-refit"
            if event.startswith("critic-refit-")
            else None
        )
        if family is None:
            continue
        if event.endswith("-start"):
            if pending is not None:
                return False, ["update start precedes completion of the prior update"]
            pending = (family, row)
            continue
        if not event.endswith("-complete"):
            continue
        if pending is None:
            return False, ["update completion precedes its start"]
        start_family, start = pending
        if start_family != family:
            return False, ["update completion kind differs from its start"]
        ordered_pairs.append((start, row))
        pending = None
    if pending is not None:
        return False, ["final update start has no following completion"]

    def bind(version: Any, actor_sha: Any, context: str) -> str | None:
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version < 0
            or not _is_sha(actor_sha)
        ):
            return f"{context} has no typed policy-version/actor identity"
        previous = versions.setdefault(version, actor_sha)
        if previous != actor_sha:
            return f"policy version {version} maps to conflicting actor identities"
        return None

    previous_after_version: int | None = None
    previous_after_hashes: dict[str, Any] | None = None
    for start, complete in ordered_pairs:
        before_version = start.get("policyVersionBefore")
        after_version = complete.get("policyVersionAfter")
        before_hashes = start.get("beforeHashes")
        after_hashes = complete.get("afterHashes")
        if (
            isinstance(before_version, bool)
            or not isinstance(before_version, int)
            or isinstance(after_version, bool)
            or not isinstance(after_version, int)
            or after_version - before_version not in {0, 1}
        ):
            return False, ["update policy-version delta is not zero or one"]
        if not isinstance(before_hashes, dict) or not isinstance(after_hashes, dict):
            return False, ["update boundary hashes are absent"]
        if previous_after_version is not None and (
            before_version != previous_after_version
            or before_hashes != previous_after_hashes
        ):
            return False, ["adjacent update boundaries are not contiguous"]
        error = bind(before_version, before_hashes.get("actor"), "update start")
        if error is not None:
            return False, [error]
        error = bind(after_version, after_hashes.get("actor"), "update completion")
        if error is not None:
            return False, [error]
        completed.append((after_version, after_hashes["actor"]))
        previous_after_version = after_version
        previous_after_hashes = after_hashes
    if completed and completed[-1][1] != endpoint_actor_sha256:
        return False, ["final traced actor identity differs from the endpoint actor bytes"]
    transitions_by_id: dict[str, dict[str, Any]] = {}
    for row in transitions:
        transition_id = row.get("transitionId")
        if (
            not isinstance(transition_id, str)
            or not transition_id
            or transition_id in transitions_by_id
        ):
            return False, ["trace transition identity is absent or duplicated"]
        transitions_by_id[transition_id] = row
        version = row.get("policyVersion")
        actor_sha = row.get("actorSha256")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version < 0
            or not _is_sha(actor_sha)
        ):
            return False, ["transition has no typed policy-version/actor identity"]
        if version not in versions:
            return False, ["transition policy version has no bound optimizer receipt"]
        if versions[version] != actor_sha:
            return False, ["transition actor identity conflicts with its policy version"]
    for row in updates:
        if not str(row.get("event", "")).endswith("-start"):
            continue
        version = row.get("policyVersionBefore")
        actor_sha = (row.get("beforeHashes") or {}).get("actor")
        transition_ids = row.get("batchTransitionIds")
        if not isinstance(transition_ids, list) or not transition_ids:
            return False, ["update start has no exact input transition batch"]
        for transition_id in transition_ids:
            transition = transitions_by_id.get(transition_id)
            if transition is None:
                return False, ["update batch names a missing transition"]
            behavior_version = transition.get("policyVersion")
            if behavior_version > version:
                return False, [
                    "update batch transition was produced by a future policy version"
                ]
    return True, []


def _resume_answerability(
    descriptor: dict[str, Any], checkpoint_state: dict[str, Any], base: Path, slot: dict[str, Any]
) -> tuple[bool, list[str]]:
    if descriptor.get("source_kind") != "file":
        return False, ["resume-state must be a separate weights-only file"]
    path, _ = _file_ref(slot["source"]["artifact"], base, "resume state")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        return False, [f"resume state is not weights-only inspectable: {error}"]
    if not isinstance(state, dict):
        return False, ["resume state root is not a mapping"]
    missing = sorted(RESUME_KEYS - set(state))
    if missing:
        return False, ["resume state lacks exact reconstruction sections: " + ", ".join(missing)]
    optimizers = state.get("optimizers")
    if not isinstance(optimizers, dict) or not optimizers:
        return False, ["resume state has no optimizer state"]
    for name, optimizer in optimizers.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(optimizer, dict)
            or set(optimizer) != {"state", "param_groups"}
            or not isinstance(optimizer["state"], dict)
            or not isinstance(optimizer["param_groups"], list)
            or not optimizer["param_groups"]
            or any(
                not isinstance(group, dict)
                or not isinstance(group.get("params"), list)
                or not group["params"]
                or any(type(parameter) is not int or parameter < 0 for parameter in group["params"])
                for group in optimizer["param_groups"]
            )
        ):
            return False, [f"resume optimizer {name!r} is null or structurally incomplete"]
    if (
        not isinstance(state.get("iteration"), int)
        or isinstance(state.get("iteration"), bool)
        or state["iteration"] < 0
    ):
        return False, ["resume state iteration is invalid"]
    python_rng = state.get("python_rng")
    if (
        not isinstance(python_rng, tuple)
        or len(python_rng) != 3
        or type(python_rng[0]) is not int
        or not isinstance(python_rng[1], tuple)
        or not python_rng[1]
        or python_rng[2] is not None
        and not isinstance(python_rng[2], float)
    ):
        return False, ["resume Python RNG state is absent or malformed"]
    numpy_rng = state.get("numpy_rng")
    if (
        not isinstance(numpy_rng, dict)
        or not isinstance(numpy_rng.get("bit_generator"), str)
        or not numpy_rng["bit_generator"]
        or not isinstance(numpy_rng.get("state"), dict)
        or not numpy_rng["state"]
    ):
        return False, ["resume NumPy RNG state is absent or malformed"]
    torch_rng = state.get("torch_rng")
    if (
        not isinstance(torch_rng, torch.Tensor)
        or torch_rng.dtype != torch.uint8
        or torch_rng.ndim != 1
        or torch_rng.numel() == 0
    ):
        return False, ["resume Torch RNG state is absent or malformed"]
    for name in ("simulator_rng", "dataloader_state"):
        if not isinstance(state.get(name), dict) or not state[name]:
            return False, [f"resume {name} is absent or malformed"]
    for name in ("scheduler", "scaler"):
        value = state.get(name)
        explicit_na = (
            isinstance(value, dict)
            and set(value) == {"status", "reason"}
            and value.get("status") == "not-applicable"
            and isinstance(value.get("reason"), str)
            and bool(value["reason"].strip())
        )
        if not explicit_na and (not isinstance(value, dict) or not value):
            return False, [f"resume {name} is absent without explicit not-applicable evidence"]
    for name in ("actor", "critic"):
        if name not in checkpoint_state or _hash_value(state.get(name)) != _hash_value(checkpoint_state[name]):
            return False, [f"resume {name} does not equal the endpoint {name}"]
    if state.get("run_manifest") != checkpoint_state.get("run_manifest"):
        return False, ["resume run manifest does not equal the endpoint run manifest"]
    return True, []


def _inspect_node(node: dict[str, Any], base: Path) -> dict[str, Any]:
    _node_identity(node)
    node_id = node["id"]
    slots = node["artifacts"]
    endpoint_slot = slots["evaluation-checkpoint"]
    descriptors: dict[str, dict[str, Any]] = {}
    transitions: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    checkpoint_state: dict[str, Any] = {}
    checkpoint_raw: dict[str, Any] | None = None
    checkpoint_sha: str | None = None

    if endpoint_slot.get("status") == "present":
        if set(endpoint_slot) != {"status", "source"} or endpoint_slot["source"].get("kind") != "file":
            _fail(f"node {node_id} evaluation checkpoint must be a file source")
        path, checkpoint_sha = _file_ref(
            endpoint_slot["source"].get("artifact"), base, f"node {node_id} evaluation checkpoint"
        )
        checkpoint_raw, checkpoint_state, _ = inspect_checkpoint(
            {"path": str(path), "sha256": checkpoint_sha}, base
        )
        expected_contract = EXPECTED_NODES[node_id]["checkpoint_contract"]
        if checkpoint_raw["checkpoint_contract"] != expected_contract:
            _fail(
                f"node {node_id} checkpoint contract differs from the historical slot "
                f"({checkpoint_raw['checkpoint_contract']} != {expected_contract})"
            )
        _checkpoint_identity(node, checkpoint_state)
        descriptors["evaluation-checkpoint"] = {
            "status": "present",
            "source_kind": "file",
            "sha256": checkpoint_sha,
            "raw_schema": checkpoint_raw,
        }
    else:
        descriptors["evaluation-checkpoint"] = _missing_slot(
            endpoint_slot, node_id, "evaluation-checkpoint", base
        )

    for name in SLOTS[1:]:
        slot = slots[name]
        if not isinstance(slot, dict):
            _fail(f"node {node_id} slot {name} is not an object")
        if slot.get("status") == "present":
            if checkpoint_raw is None and slot.get("source", {}).get("kind") == "embedded":
                _fail(f"node {node_id} slot {name} cannot embed into a missing checkpoint")
            descriptor, found_transitions, found_updates = _present_slot(
                slot, base, checkpoint_state, node, name
            )
            descriptors[name] = descriptor
            if name == "transition-update-trace":
                transitions, updates = found_transitions, found_updates
        else:
            descriptors[name] = _missing_slot(slot, node_id, name, base)

    trace_descriptor = descriptors["transition-update-trace"]
    if trace_descriptor["status"] == "present":
        endpoint_actor_sha = (
            _hash_value(checkpoint_state["actor"])
            if isinstance(checkpoint_state.get("actor"), dict)
            else None
        )
        expected_bindings = {
            "checkpoint_sha256": descriptors["evaluation-checkpoint"].get("sha256"),
            "endpoint_actor_sha256": endpoint_actor_sha,
            "normalizer_sha256": descriptors["normalizer"].get("sha256"),
            "run_manifest_sha256": descriptors["run-manifest"].get("sha256"),
            "producer_provenance_sha256": descriptors["producer-build-provenance"].get(
                "sha256"
            ),
            "transitions_sha256": trace_descriptor["file_sha256s"].get("transitions"),
            "updates_sha256": trace_descriptor["file_sha256s"].get("updates"),
        }
        if not all(_is_sha(value) for value in expected_bindings.values()):
            _fail(f"node {node_id} trace cannot bind a missing endpoint/provenance input")
        if trace_descriptor["bindings"] != expected_bindings:
            _fail(f"node {node_id} trace endpoint/provenance bindings differ")

    producer_inventory, producer_hashes = _producer_artifact_inventory(node, base)
    if producer_inventory:
        descriptors["producer-artifact-inventory"].update(producer_inventory)
    identity_bound = bool(producer_inventory) and node["run_identity"][
        "producer_run_id"
    ] is not None
    historical_provenance_verified = bool(
        identity_bound
        and producer_inventory
        and producer_inventory["authoritative_completeness"]
    )

    holds: list[str] = []
    if node["run_identity"]["producer_run_id"] is None:
        holds.append("exact producer run identity is not yet recovered")
    if producer_inventory and not producer_inventory["authoritative_completeness"]:
        holds.append(
            "producer artifact completeness lacks separately reviewed provenance"
        )
    for name in sorted(REQUIRED_ENDPOINT_SLOTS):
        if descriptors[name]["status"] != "present":
            holds.append(f"required slot {name} is {descriptors[name]['status']}")
    if checkpoint_raw is not None and checkpoint_raw["checkpoint_contract"] < 12:
        if descriptors["adapter-default-migration"]["status"] != "present":
            holds.append("legacy endpoint lacks a bound adapter/default/migration bundle")

    actual_critic = checkpoint_state.get("critic")
    critic_present = isinstance(actual_critic, dict) and bool(actual_critic)
    declared_critic = descriptors["critic"]["status"]
    if critic_present and declared_critic != "present":
        _fail(f"node {node_id} has checkpoint critic bytes but declares critic {declared_critic}")
    if not critic_present and declared_critic == "present":
        _fail(f"node {node_id} declares critic present but checkpoint has no critic bytes")
    if not critic_present and declared_critic == "missing":
        holds.append("expected critic bytes are missing/lost")

    evidence = [
        value
        for descriptor in descriptors.values()
        for value in (
            [descriptor["sha256"]] if _is_sha(descriptor.get("sha256")) else []
        )
    ]
    for descriptor in descriptors.values():
        evidence.extend(descriptor.get("file_sha256s", {}).values())
        evidence.extend(descriptor.get("snapshot_sha256s", []))
        if _is_sha(descriptor.get("support_sha256")):
            evidence.append(descriptor["support_sha256"])
        evidence.extend(descriptor.get("support_source_sha256s", []))
    evidence.extend(producer_hashes)

    endpoint_status = "answerable" if not holds else "blocked"
    questions: dict[str, dict[str, Any]] = {
        "endpoint_artifact_inventory": _question(endpoint_status, holds, evidence),
        "strict_evaluator_eligibility": _question(
            "blocked",
            [
                "retrieval preflight does not replace the independently reviewed loader, "
                "adapter, producer-comparator and evaluation/claim protocols"
            ],
            evidence,
        ),
    }
    if critic_present:
        questions["critic_state_identity"] = _question(
            "answerable" if declared_critic == "present" and identity_bound else "blocked",
            []
            if declared_critic == "present" and identity_bound
            else ["critic bytes lack exact producer/acquisition inventory binding"],
            evidence,
        )
    elif declared_critic == "not-applicable":
        questions["critic_state_identity"] = _question(
            "blocked",
            ["historical critic nonexistence lacks separately reviewed provenance"],
            evidence,
        )
    else:
        questions["critic_state_identity"] = _question(
            "blocked", ["critic bytes are absent without reviewed not-applicable support"], evidence
        )

    resume = descriptors["resume-state"]
    if resume["status"] == "present" and checkpoint_raw is not None and identity_bound:
        complete, reasons = _resume_answerability(resume, checkpoint_state, base, slots["resume-state"])
        questions["resume_sections_inventoried"] = _question(
            "answerable" if complete else "blocked", reasons, evidence
        )
    else:
        reason = (
            "complete exact endpoint resume state was not recovered"
            if resume["status"] != "present"
            else "resume state lacks exact producer/acquisition inventory binding"
        )
        questions["resume_sections_inventoried"] = _question("blocked", [reason], evidence)
    questions["exact_endpoint_resume_eligibility"] = _question(
        "blocked",
        [
            "no separately reviewed historical restoration contract binds supported "
            "optimizer, RNG, data-order and simulator-state schemas"
        ],
        evidence,
    )
    questions["historical_optimizer_sequence_reconstruction"] = _question(
        "blocked",
        [
            "an endpoint resume state cannot reconstruct the ordered pre-update optimizer "
            "states, batches and RNG chain that produced the endpoint"
        ],
        evidence,
    )

    trace = descriptors["transition-update-trace"]
    endpoint_actor_sha = (
        _hash_value(checkpoint_state["actor"])
        if isinstance(checkpoint_state.get("actor"), dict)
        else ""
    )
    actor_chain_complete, actor_chain_reasons = _trace_actor_chain(
        transitions, updates, endpoint_actor_sha
    ) if transitions or updates else (False, ["no actor-bound transition/update chain"])
    structurally_bound_rows = bool(transitions) and identity_bound and actor_chain_complete and all(
        row.get("evidenceClass") == "original_training_trace" for row in transitions
    )
    structural_transition_reasons = (
        []
        if structurally_bound_rows
        else actor_chain_reasons or ["no complete actor-bound transition trace"]
    )
    questions["transition_trace_structural_consistency"] = _question(
        "answerable" if structurally_bound_rows else "blocked",
        structural_transition_reasons,
        evidence,
    )
    historical_rows = structurally_bound_rows and historical_provenance_verified
    historical_transition_reasons = (
        []
        if historical_rows
        else (
            ["historical trace provenance/completeness lacks separately reviewed authority"]
            if structurally_bound_rows
            else structural_transition_reasons
        )
    )
    questions["historical_transition_diagnostics"] = _question(
        "answerable" if historical_rows else "blocked",
        historical_transition_reasons,
        evidence,
    )
    critic_complete, objectives = _critic_complete(
        transitions if structurally_bound_rows else []
    )
    critic_trace_reasons = [] if critic_complete else [
        "original trace lacks complete finite prediction/target/TD-residual/advantage fields"
    ]
    questions["historical_optimization_target_diagnostics"] = _question(
        "answerable" if historical_rows and critic_complete else "blocked",
        (
            []
            if historical_rows and critic_complete
            else (
                ["historical trace provenance/completeness lacks separately reviewed authority"]
                if structurally_bound_rows and critic_complete
                else critic_trace_reasons
            )
        ),
        evidence,
    )
    if objectives:
        questions["historical_optimization_target_diagnostics"][
            "objective_identities"
        ] = objectives
    questions["independent_return_critic_accuracy"] = _question(
        "blocked",
        [
            "no frozen independently identified continuation-return evidence is bound; "
            "optimization targets cannot establish critic accuracy or course-correction"
        ],
        evidence,
    )
    joined: list[dict[str, Any]] = []
    update_error: str | None = None
    if structurally_bound_rows:
        try:
            joined = economic_diagnostics.optimizer_update_rows(transitions, updates)
        except ValueError as error:
            update_error = str(error)
    receipts_complete, receipt_reasons, hash_dimensions = _complete_update_receipts(
        updates, joined
    )
    structural_update_reasons = (
        []
        if structurally_bound_rows and receipts_complete
        else (
            structural_transition_reasons
            if not structurally_bound_rows
            else [update_error or receipt_reasons[0]]
        )
    )
    questions["update_trace_structural_consistency"] = _question(
        "answerable" if structurally_bound_rows and receipts_complete else "blocked",
        structural_update_reasons,
        evidence,
    )
    if receipts_complete:
        questions["update_trace_structural_consistency"]["joined_update_count"] = len(joined)
    update_reasons = (
        []
        if historical_rows and receipts_complete
        else (
            ["historical trace provenance/completeness lacks separately reviewed authority"]
            if structurally_bound_rows and receipts_complete
            else structural_update_reasons
        )
    )
    questions["historical_update_diagnostics"] = _question(
        "answerable" if historical_rows and receipts_complete else "blocked",
        update_reasons,
        evidence,
    )
    if receipts_complete:
        questions["historical_update_diagnostics"]["joined_update_count"] = len(joined)
        questions["historical_update_diagnostics"]["update_kinds"] = sorted(
            {str(row["updateKind"]) for row in joined}
        )
        questions["historical_update_diagnostics"]["hash_dimensions"] = hash_dimensions

    if trace["status"] != "present":
        questions["historical_transition_diagnostics"]["reasons"] = [
            "transition/update trace was not recovered"
        ]
    return {
        "id": node_id,
        "role": node["role"],
        "run_identity": node["run_identity"],
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_raw_schema": checkpoint_raw,
        "artifact_slots": descriptors,
        "answerability": questions,
        "artifact_sha256s": sorted(set(evidence)),
    }


def _conversion(value: Any, base: Path) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("status") not in {"missing", "not-applicable"}:
        _fail("conversion must be explicit missing/not-applicable until a reviewed present schema exists")
    return _missing_slot(value, "conversion", "conversion", base)


def run_preflight(path: str | Path) -> dict[str, Any]:
    """Validate one retrieval inventory and return its deterministic HOLD report."""
    inventory_path = Path(os.path.abspath(path))
    _plain_file(inventory_path, "retrieval inventory")
    inventory_sha = checkpoint_hash(inventory_path)
    inventory = _sealed_json(inventory_path, kind=KIND)
    if set(inventory) != {
        "schema",
        "kind",
        "baseline_commit",
        "execution",
        "conversion",
        "nodes",
        "digest",
    }:
        _fail("retrieval inventory has incomplete or extra top-level fields")
    if inventory.get("baseline_commit") != BASELINE_COMMIT:
        _fail("retrieval inventory baseline commit differs from the approved capability commit")
    if inventory.get("execution") != EXECUTION:
        _fail("retrieval inventory must deny evaluation, selection and readiness authority")
    nodes = inventory.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != len(EXPECTED_NODES):
        _fail("retrieval inventory must contain the exact 17 historical endpoints")
    ids = [node.get("id") for node in nodes if isinstance(node, dict)]
    if len(ids) != len(nodes) or set(ids) != set(EXPECTED_NODES) or len(ids) != len(set(ids)):
        _fail("retrieval inventory endpoint IDs differ from the reviewed 17-node inventory")
    conversion = _conversion(inventory["conversion"], inventory_path.parent)
    reports = [_inspect_node(node, inventory_path.parent) for node in nodes]
    reports.sort(key=lambda row: row["id"])

    endpoint_hashes = [
        row["checkpoint_sha256"] for row in reports if row["checkpoint_sha256"] is not None
    ]
    if len(endpoint_hashes) != len(set(endpoint_hashes)):
        _fail("distinct historical endpoint nodes may not alias one checkpoint artifact")

    holds: list[str] = []
    if conversion["status"] == "missing":
        holds.append("historical conversion status/artifact remains missing")
    else:
        holds.append(
            "historical conversion nonexistence lacks separately reviewed provenance"
        )
    for node in reports:
        for question, row in node["answerability"].items():
            if question == "endpoint_artifact_inventory" and row["status"] != "answerable":
                holds.append(f"{node['id']}: endpoint artifact inventory is incomplete")
    artifacts = sorted(
        {
            inventory_sha,
            *(
                sha
                for node in reports
                for sha in node["artifact_sha256s"]
            ),
            *(
                [conversion["support_sha256"]]
                if _is_sha(conversion.get("support_sha256"))
                else []
            ),
            *conversion.get("support_source_sha256s", []),
        }
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": REPORT_KIND,
        "baseline_commit": BASELINE_COMMIT,
        "inventory_sha256": inventory_sha,
        "execution": EXECUTION,
        "conversion": conversion,
        "nodes": reports,
        "artifact_sha256s": artifacts,
        "preflight_status": "PASS-PREFLIGHT-ONLY" if not holds else "HOLD",
        "preflight_status_scope": "retrieval-inventory-and-answerability-only",
        "ready_for_protocol_freeze": not holds,
        "evaluation_authorized": False,
        "gate1_readiness_claimed": False,
        "production_gate1": "HOLD",
        "holds": holds,
        "conclusion": (
            "content-addressed retrieval candidate is complete for independent freeze review"
            if not holds
            else "retrieval evidence is incomplete; evaluation remains prohibited"
        ),
    }
    report["digest"] = digest(report)
    return report
