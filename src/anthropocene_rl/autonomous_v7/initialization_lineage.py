"""Gate 1: strict, content-backed initialization-lineage diagnostics.

This module deliberately does not reuse :mod:`upgrade`.  A diagnostic inventory
must describe the bytes that actually exist, not the checkpoint that today's
loader could manufacture from them.  Legacy execution therefore needs a named
pinned implementation or reviewed adapter, while reconstructed migrations remain
separate artifacts and never establish that a historical conversion occurred.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import stat
import struct
import subprocess
import sys
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import torch

from . import contracts
from .protocol import DENIED, ROLES, scan
from .protocol import artifact as _protocol_artifact
from .provenance import ProvenanceError, checkpoint_hash, digest

SCHEMA = 1
KIND = "initialization_lineage"
ARCHITECTURE_COMMIT = "a3f64427d536e66eab9e0a016d1e9c2ef5f9a8f0"
EVIDENCE_CLASSES = {
    "historical-record",
    "checkpoint-evaluation",
    "instrumented-replay",
    "matched-intervention",
    "prospective-update-audit",
}
RELATIONS = {
    "parameter_parent",
    "teacher_data_source",
    "evaluation_reference",
    "schema_conversion",
    "normalizer_change",
    "clone_fit",
    "critic_initialization",
    "ppo_continuation",
}
EQUIVALENCE_KINDS = {
    "exact_tensor_preservation",
    "mapped_distribution_numerical_equivalence",
    "deterministic_behavioral_equivalence",
    "stochastic_behavioral_equivalence",
}
NODE_STATUSES = {"present", "missing", "not-applicable"}
EDGE_STATUSES = {"present", "missing", "not-applicable"}
LOADING_MODES = {"native-strict", "pinned-historical", "reviewed-adapter"}
PANE_KINDS = {
    "historical-semantics",
    "standardized-flags-intervention",
    "fresh-common-producer-diagnostic",
}
SUPPORTED_RAW_CONTRACTS = frozenset(range(8, contracts.NATIVE_CONTRACT + 1))
MANDATORY_METRICS = frozenset(
    {
        "welfare",
        "population_years",
        "ending_residents",
        "qol_years",
        "trade_utility",
        "extinctions",
        "settlements",
        "service_outcomes",
        "compiler_interventions",
        "trajectory_digest",
    }
)
CURRENT_METRICS = MANDATORY_METRICS
SEMANTIC_FIELDS = (
    "experts",
    "token",
    "commitments",
    "learned_movement",
    "region_movement",
    "learned_fission",
    "learned_trade",
    "region_head",
    "grouped_region",
    "region_selection",
)
INTERVENTION_FIELDS = frozenset(SEMANTIC_FIELDS) - {"experts", "token"}
BOOLEAN_SEMANTIC_FIELDS = frozenset(
    {
        "experts",
        "token",
        "commitments",
        "learned_movement",
        "region_movement",
        "learned_fission",
        "learned_trade",
        "grouped_region",
    }
)
REPOSITORY = Path(__file__).resolve().parents[3]
IMPLEMENTATION_SOURCES = (
    "docs/MODEL9_DIAGNOSTIC_CLOSURE_ARCHITECTURE.md",
    "docs/MODEL9_ECONOMIC_OBSERVABILITY_ARCHITECTURE.md",
    "docs/MODEL9_GATE1_PRODUCTION_PROTOCOL.md",
    "docs/MODEL9_GATE1_IMPLEMENTATION_STATUS.md",
    "src/anthropocene_rl/autonomous_v7/evaluate.py",
    "src/anthropocene_rl/autonomous_v7/bridge.py",
    "src/anthropocene_rl/autonomous_v7/model.py",
    "src/anthropocene_rl/autonomous_v7/gate1_diagnostics.py",
    "src/anthropocene_rl/autonomous_v7/gate1_preflight.py",
    "src/anthropocene_rl/autonomous_v7/legacy_contract8.py",
    "src/anthropocene_rl/autonomous_v7/gate1_production_adapter.py",
    "src/anthropocene_rl/autonomous_v7/gate1_freeze_candidate.py",
    "tests/test_autonomous_v7_bridge.py",
    "tests/test_gate1_diagnostics.py",
    "tests/test_gate1_preflight.py",
    "tests/test_gate1_legacy_contract8.py",
    "tests/test_gate1_freeze_candidate.py",
    "src/anthropocene_rl/autonomous_v7/initialization_lineage.py",
    "tests/test_initialization_lineage.py",
)
CURRENT_BACKGROUND_POLICY = {"kind": "none", "actor_scope": "all-live-villages"}
EXECUTION_TIERS = {"local", "vm"}
COUNT_METRICS = frozenset(
    {"ending_residents", "extinctions", "settlements", "compiler_interventions"}
)
SCALAR_METRICS = MANDATORY_METRICS - {"service_outcomes", "trajectory_digest"}
LINEAGE_SLOTS = ("prior-reference", "conversion", "guarded-clone", "trained-t3")
PAIRING_STATUSES = {"matched", "unpaired", "not-applicable"}


class ReviewPacket(dict):
    """Canonical packet plus transient source paths used to materialize its CAS."""

    artifact_sources: dict[str, Path]


_STRICT_PACKET_CAS: ContextVar[Path | None] = ContextVar(
    "initialization_lineage_strict_packet_cas",
    default=None,
)


def _fail(message: str) -> None:
    raise ProvenanceError("initialization lineage: " + message)


def _is_sha(value: Any, width: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{width}}}", value) is not None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _native_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def _reparse_point(metadata: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and getattr(metadata, "st_file_attributes", 0) & flag)


def _plain_file(path: Path, what: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        _fail(f"{what} is missing or unreadable: {error}")
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink() or _reparse_point(metadata):
        _fail(f"{what} must be a regular non-reparse file: {path}")


def _plain_directory(path: Path, what: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        _fail(f"{what} is missing or unreadable: {error}")
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink() or _reparse_point(metadata):
        _fail(f"{what} must be a regular non-reparse directory: {path}")


def _strict_cas_artifact(ref: dict, root: Path) -> tuple[Path, str]:
    """Resolve one artifact by digest only from a prevalidated packet CAS."""
    if (
        not isinstance(ref, dict)
        or not isinstance(ref.get("path"), str)
        or not _is_sha(ref.get("sha256"))
    ):
        _fail("packet evidence requires a content path and SHA-256")
    sha = ref["sha256"]
    native = _native_path(root / "cas" / sha)
    _plain_file(native, "packet CAS artifact")
    if checkpoint_hash(native) != sha:
        _fail(f"packet CAS artifact content hash mismatch: {sha}")
    return native, sha


@contextmanager
def _packet_cas_only(root: Path):
    """Make all nested artifact resolution packet-local during reconstruction."""
    token = _STRICT_PACKET_CAS.set(root.resolve())
    try:
        yield
    finally:
        _STRICT_PACKET_CAS.reset(token)


def artifact(ref: dict, base: Path | None = None) -> tuple[Path, str]:
    """Resolve ordinary evidence, falling back to a packet-local SHA-addressed CAS."""
    strict_root = _STRICT_PACKET_CAS.get()
    if strict_root is not None:
        # Reconstruction deliberately ignores ref["path"].  It is retained as
        # provenance, but cannot become an escape hatch to a readable original.
        return _strict_cas_artifact(ref, strict_root)
    sha = ref.get("sha256") if isinstance(ref, dict) else None
    if base is not None and _is_sha(sha):
        cas_path = _native_path(Path(base) / "cas" / sha)
        if cas_path.is_file() and checkpoint_hash(cas_path) == sha:
            return cas_path.resolve(), sha
    try:
        return _protocol_artifact(ref, base)
    except ProvenanceError as original:
        if base is None or not _is_sha(sha):
            raise original
        path = Path(base) / "cas" / sha
        if path.is_file() and checkpoint_hash(path) == sha:
            return path.resolve(), sha
        raise original


def _read_json_ref(ref: dict, base: Path, *, kind: str | None = None) -> tuple[dict, Path]:
    path, _ = artifact(ref, base)
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"invalid JSON evidence {path}: {error}")
    scan(value, str(path))
    if kind is not None and (value.get("schema") != 1 or value.get("kind") != kind):
        _fail(f"{path} is not schema-1 {kind} evidence")
    if "digest" in value:
        try:
            calculated = digest({k: v for k, v in value.items() if k != "digest"})
        except (TypeError, ValueError):
            _fail(f"evidence contains nonfinite or noncanonical JSON values: {path}")
        if value["digest"] != calculated:
            _fail(f"evidence digest mismatch: {path}")
    return value, path


def _execution_declaration(value: Any, what: str) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != {"tier", "selection_allowed", "readiness_claimed"}
        or value.get("tier") not in EXECUTION_TIERS
        or value.get("selection_allowed") is not False
        or value.get("readiness_claimed") is not False
    ):
        _fail(f"{what} must declare local/vm execution and deny selection/readiness claims")
    return value


def _reviewed_protocol(
    body: dict, base: Path, proposal_kind: str, required_scope: set[str]
) -> dict:
    review, _ = _read_json_ref(body.get("review"), base, kind="independent_protocol_review")
    reviewer, _ = _read_json_ref(
        review.get("reviewer_identity"), base, kind="independent_reviewer_identity"
    )
    if (
        set(reviewer) != {"schema", "kind", "provider", "model", "version", "digest"}
        or reviewer.get("provider") != "openai"
        or reviewer.get("model") != "astra-high"
        or reviewer.get("version") != "2026-10-22"
        or any(
            not isinstance(reviewer.get(key), str) or not reviewer[key].strip()
            for key in ("provider", "model", "version")
        )
    ):
        _fail(f"{proposal_kind} reviewer identity is malformed")
    proposal_digest = digest(
        {key: value for key, value in body.items() if key not in {"review", "digest"}}
    )
    if (
        set(review)
        != {
            "schema",
            "kind",
            "reviewer_identity",
            "decision",
            "proposal_kind",
            "proposal_digest",
            "scope",
            "findings",
            "digest",
        }
        or review.get("decision") != "approved"
        or review.get("proposal_kind") != proposal_kind
        or review.get("proposal_digest") != proposal_digest
        or not isinstance(review.get("scope"), list)
        or set(review["scope"]) != required_scope
        or not isinstance(review.get("findings"), list)
        or not review["findings"]
        or any(not isinstance(item, str) or not item.strip() for item in review["findings"])
    ):
        _fail(f"{proposal_kind} review is unbound, incomplete or not approved")
    return review


def _service_schema(pane: dict, base: Path) -> dict:
    body, _ = _read_json_ref(
        pane.get("service_outcome_schema"), base, kind="service_outcome_schema"
    )
    if set(body) != {"schema", "kind", "fields", "digest"} or not _is_sha(body.get("digest")):
        _fail(f"pane {pane.get('id')} service outcome schema is not canonical/content-addressed")
    fields = body.get("fields")
    field_sets = ({"name", "unit"}, {"name", "unit", "source"})
    if (
        not isinstance(fields, list)
        or not fields
        or any(
            not isinstance(row, dict)
            or set(row) not in field_sets
            or not isinstance(row["name"], str)
            or not row["name"].strip()
            or not isinstance(row["unit"], str)
            or not row["unit"].strip()
            or (
                "source" in row
                and (not isinstance(row["source"], str) or not row["source"].strip())
            )
            for row in fields
        )
        or len({row["name"] for row in fields}) != len(fields)
    ):
        _fail(f"pane {pane.get('id')} service outcome schema has invalid fields/units")
    return body


def _verification_protocol(document: dict, base: Path) -> tuple[dict, str]:
    body, _ = _read_json_ref(
        document.get("verification_protocol"), base, kind="gate1_verification_protocol"
    )
    if (
        set(body)
        != {
            "schema",
            "kind",
            "architecture_commit",
            "review",
            "tests",
            "coverage",
            "parity",
            "benchmarks",
            "digest",
        }
        or body.get("architecture_commit") != ARCHITECTURE_COMMIT
    ):
        _fail("Gate 1 verification protocol has a noncanonical field set")
    _reviewed_protocol(
        body,
        base,
        "gate1_verification_protocol",
        {"tests", "coverage", "diagnostic-off-parity", "benchmarks", "budgets"},
    )
    tests = body.get("tests")
    if (
        not isinstance(tests, list)
        or not tests
        or any(
            not isinstance(row, dict)
            or set(row) != {"id", "command"}
            or not isinstance(row["id"], str)
            or not row["id"].strip()
            or not isinstance(row["command"], list)
            or not row["command"]
            or any(not isinstance(part, str) or not part for part in row["command"])
            for row in tests
        )
        or len({row["id"] for row in tests}) != len(tests)
    ):
        _fail("Gate 1 verification protocol has malformed required test IDs/commands")
    coverage = body.get("coverage")
    if (
        not isinstance(coverage, dict)
        or set(coverage) != {"minimum_percent", "scope", "denominator", "exclusions", "tool"}
        or type(coverage.get("minimum_percent")) not in (int, float)
        or not math.isfinite(coverage["minimum_percent"])
        or coverage["minimum_percent"] <= 0
        or coverage["minimum_percent"] > 100
        or not isinstance(coverage.get("scope"), list)
        or not coverage["scope"]
        or any(not isinstance(item, str) or not item for item in coverage["scope"])
        or not isinstance(coverage.get("denominator"), str)
        or not coverage["denominator"].strip()
        or not isinstance(coverage.get("exclusions"), list)
        or any(not isinstance(item, str) for item in coverage["exclusions"])
        or len(coverage["scope"]) != len(set(coverage["scope"]))
    ):
        _fail("Gate 1 verification protocol has an invalid coverage scope/denominator")
    _implementation_artifact(
        coverage.get("tool"), base, "verification_tool_identity", "coverage tool"
    )
    parity = body.get("parity")
    if (
        not isinstance(parity, dict)
        or set(parity) != {"method", "scenarios"}
        or parity.get("method") != "bitwise-diagnostic-off"
        or not isinstance(parity.get("scenarios"), list)
        or not parity["scenarios"]
        or any(not isinstance(item, str) or not item for item in parity["scenarios"])
        or len(parity["scenarios"]) != len(set(parity["scenarios"]))
    ):
        _fail("Gate 1 verification protocol has invalid diagnostic-off parity scenarios")
    benchmarks = body.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        _fail("Gate 1 verification protocol has no benchmark scenarios")
    benchmark_ids = set()
    for row in benchmarks:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "scenario", "baseline", "throughput", "memory"}
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or row["id"] in benchmark_ids
            or not isinstance(row.get("scenario"), str)
            or not row["scenario"]
            or not isinstance(row.get("throughput"), dict)
            or set(row["throughput"]) != {"work_unit", "unit", "minimum"}
            or not isinstance(row["throughput"].get("work_unit"), str)
            or not row["throughput"]["work_unit"]
            or not isinstance(row["throughput"].get("unit"), str)
            or not row["throughput"]["unit"]
            or type(row["throughput"].get("minimum")) not in (int, float)
            or not math.isfinite(row["throughput"]["minimum"])
            or row["throughput"]["minimum"] <= 0
            or not isinstance(row.get("memory"), dict)
            or set(row["memory"]) != {"method", "unit", "maximum"}
            or not isinstance(row["memory"].get("method"), str)
            or not row["memory"]["method"]
            or row["memory"].get("unit") != "MiB"
            or type(row["memory"].get("maximum")) not in (int, float)
            or not math.isfinite(row["memory"]["maximum"])
            or row["memory"]["maximum"] <= 0
        ):
            _fail("Gate 1 verification protocol has an invalid benchmark/budget")
        artifact(row.get("baseline"), base)
        benchmark_ids.add(row["id"])
    return body, document["verification_protocol"]["sha256"]


def _run_identity(value: Any, what: str) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != {"run_id", "arm_id", "optimizer_seed", "pairing_status", "pairing_group"}
        or not isinstance(value.get("run_id"), str)
        or not value["run_id"].strip()
        or value.get("pairing_status") not in PAIRING_STATUSES
    ):
        _fail(f"{what} has an invalid run/optimizer identity")
    arm = value["arm_id"]
    seed, pairing, group = value["optimizer_seed"], value["pairing_status"], value["pairing_group"]
    if seed is None:
        if arm is not None or pairing != "not-applicable" or group is not None:
            _fail(
                f"{what} without an optimizer seed must have not-applicable arm/pairing semantics"
            )
    elif type(seed) is not int or seed < 0 or pairing == "not-applicable":
        _fail(f"{what} optimizer seed/pairing status is invalid")
    elif not isinstance(arm, str) or not arm.strip():
        _fail(f"{what} optimizer identity needs an explicit treatment arm")
    elif pairing == "matched":
        if not isinstance(group, str) or not group.strip():
            _fail(f"{what} matched optimizer identity needs a pairing group")
    elif group is not None:
        _fail(f"{what} unpaired optimizer identity cannot name a pairing group")
    return value


def _equivalence_descriptor(claim: Any) -> dict | None:
    if claim is None:
        return None
    if not isinstance(claim, dict):
        _fail("equivalence claim descriptor is malformed")
    return {
        "required": True,
        "kind": claim.get("kind"),
        "claim_sha256": _claim_digest(claim),
        "corpus_sha256": claim.get("corpus", {}).get("sha256"),
        "tolerances": claim.get("tolerances"),
        "numeric_runtime": claim.get("numeric_runtime"),
        "execution_protocol_sha256": claim.get("execution_protocol", {}).get("sha256"),
    }


def _evaluation_claim_projection(document: dict) -> dict:
    """The complete pre-result Gate-1 surface covered by independent review."""
    nodes = [json.loads(json.dumps(row, sort_keys=True)) for row in document.get("nodes", [])]
    edges = []
    for row in document.get("edges", []):
        frozen = json.loads(json.dumps(row, sort_keys=True))
        claim = frozen.get("equivalence")
        if isinstance(claim, dict):
            # Execution results are collected after preregistration.  Every
            # other declaration (including corpus, tolerances, implementations,
            # runtime and migration evidence) is independently reviewed.
            claim.pop("result", None)
        edges.append(frozen)
    return {
        "nodes": nodes,
        "edges": edges,
        "panes": document.get("evaluation", {}).get("panes", []),
    }


def _evaluation_claim_protocol(document: dict, base: Path) -> tuple[dict, str]:
    body, _ = _read_json_ref(
        document.get("evaluation_claim_protocol"), base, kind="gate1_evaluation_claim_protocol"
    )
    required = {
        "schema",
        "kind",
        "architecture_commit",
        "review",
        "nodes",
        "edges",
        "panes",
        "digest",
    }
    if set(body) != required or body.get("architecture_commit") != ARCHITECTURE_COMMIT:
        _fail("Gate 1 evaluation/claim protocol is noncanonical")
    _reviewed_protocol(
        body,
        base,
        "gate1_evaluation_claim_protocol",
        {"artifacts", "equivalence", "evaluation-cohort", "semantics", "implementations"},
    )
    projection = _evaluation_claim_projection(document)
    if any(body.get(key) != projection[key] for key in ("nodes", "edges", "panes")):
        _fail("manifest artifact/equivalence/evaluation surface differs from reviewed protocol")
    return body, document["evaluation_claim_protocol"]["sha256"]


def _lineage_inventory(document: dict, base: Path) -> tuple[dict, str]:
    body, _ = _read_json_ref(
        document.get("lineage_inventory"), base, kind="gate1_lineage_inventory_protocol"
    )
    required = {
        "schema",
        "kind",
        "architecture_commit",
        "review",
        "slots",
        "nodes",
        "edges",
        "required_nodes",
        "required_edges",
        "sequences",
        "comparison_order",
        "digest",
    }
    if set(body) != required or body.get("architecture_commit") != ARCHITECTURE_COMMIT:
        _fail("lineage inventory protocol is noncanonical or bound to another architecture")
    _reviewed_protocol(
        body,
        base,
        "gate1_lineage_inventory_protocol",
        {"actual-inventory", "slots", "applicability", "chronology", "run-identities"},
    )
    slots = body.get("slots")
    if not isinstance(slots, list) or [
        row.get("slot") for row in slots if isinstance(row, dict)
    ] != list(LINEAGE_SLOTS):
        _fail("lineage inventory must prescribe the four user-facing slots in order")
    for row in slots:
        if (
            set(row) != {"slot", "status", "node_ids", "edge_ids", "reason", "historical_support"}
            or row["status"] not in NODE_STATUSES
            or not isinstance(row["node_ids"], list)
            or not isinstance(row["edge_ids"], list)
            or len(row["node_ids"]) != len(set(row["node_ids"]))
            or len(row["edge_ids"]) != len(set(row["edge_ids"]))
        ):
            _fail(f"lineage slot {row.get('slot')} is malformed")
        if row["slot"] == "conversion" and row["status"] == "not-applicable":
            if row["node_ids"] or row["edge_ids"] or not str(row.get("reason", "")).strip():
                _fail("not-applicable conversion slot cannot invent conversion artifacts")
            support, _ = _read_json_ref(
                row.get("historical_support"), base, kind="historical_absence_evidence"
            )
            if (
                support.get("slot") != "conversion"
                or support.get("finding") != "not-applicable"
                or not isinstance(support.get("sources"), list)
                or not support["sources"]
            ):
                _fail("not-applicable conversion lacks content-backed historical support")
            for ref in support["sources"]:
                artifact(ref, base)
        elif row["slot"] == "conversion":
            if row["status"] not in {"present", "missing"} or not (
                row["node_ids"] or row["edge_ids"]
            ):
                _fail(
                    "conversion slot must name expected artifacts or supported "
                    "not-applicable history"
                )
            if row["status"] == "missing":
                if not str(row.get("reason", "")).strip() or not isinstance(
                    row.get("historical_support"), dict
                ):
                    _fail("missing conversion slot needs content-backed historical support")
                artifact(row["historical_support"], base)
            elif row["reason"] is not None or row["historical_support"] is not None:
                _fail("present conversion slot cannot carry missingness evidence")
        else:
            expected_count = 15 if row["slot"] == "trained-t3" else 1
            if (
                len(row["node_ids"]) != expected_count
                or row["edge_ids"]
                or row["status"] not in {"present", "missing"}
            ):
                _fail(f"lineage slot {row['slot']} has the wrong reviewed node cardinality")
            if row["status"] == "missing":
                if not str(row.get("reason", "")).strip() or not isinstance(
                    row.get("historical_support"), dict
                ):
                    _fail(f"missing lineage slot {row['slot']} needs historical support")
                artifact(row["historical_support"], base)
            elif row["reason"] is not None or row["historical_support"] is not None:
                _fail(f"present lineage slot {row['slot']} cannot carry a missingness reason")
    protocol_nodes = body.get("nodes")
    protocol_edges = body.get("edges")
    if (
        not isinstance(protocol_nodes, list)
        or not protocol_nodes
        or not isinstance(protocol_edges, list)
        or not protocol_edges
    ):
        _fail("lineage inventory has no exact node/edge inventory")
    node_map = {}
    for row in protocol_nodes:
        if (
            not isinstance(row, dict)
            or set(row)
            != {"id", "role", "status", "historical_panes", "standardized_panes", "run_identity"}
            or not isinstance(row.get("id"), str)
            or row["id"] in node_map
            or not isinstance(row.get("role"), str)
            or not row["role"].strip()
            or row.get("status") not in NODE_STATUSES
            or not isinstance(row.get("historical_panes"), list)
            or not isinstance(row.get("standardized_panes"), list)
            or len(row["historical_panes"]) != len(set(row["historical_panes"]))
            or len(row["standardized_panes"]) != len(set(row["standardized_panes"]))
        ):
            _fail("lineage inventory contains a malformed/duplicate node")
        _run_identity(row["run_identity"], f"inventory node {row['id']}")
        node_map[row["id"]] = row
    edge_map = {}
    for row in protocol_edges:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "id",
                "source",
                "destination",
                "relation",
                "status",
                "historical_panes",
                "standardized_panes",
                "sequence",
                "position",
                "equivalence",
            }
            or not isinstance(row.get("id"), str)
            or row["id"] in edge_map
            or row.get("source") not in node_map
            or row.get("destination") not in node_map
            or row["source"] == row["destination"]
            or row.get("relation") not in RELATIONS
            or row.get("status") not in EDGE_STATUSES
            or not isinstance(row.get("historical_panes"), list)
            or not isinstance(row.get("standardized_panes"), list)
            or type(row.get("position")) is not int
            or row["position"] < 0
        ):
            _fail("lineage inventory contains a malformed/duplicate edge")
        edge_map[row["id"]] = row
    allowed_roles = {
        "prior-reference": {"prior-policy", "evaluation-reference", "teacher-reference"},
        "guarded-clone": {"guarded-clone"},
        "trained-t3": {"trained-t3", "trained-policy"},
    }
    mapped_nodes = []
    mapped_edges = []
    for slot in slots:
        if any(node_id not in node_map for node_id in slot["node_ids"]):
            _fail(f"lineage slot {slot['slot']} names a nonexistent node")
        if any(edge_id not in edge_map for edge_id in slot["edge_ids"]):
            _fail(f"lineage slot {slot['slot']} names a nonexistent edge")
        if slot["slot"] == "conversion":
            for node_id in slot["node_ids"]:
                if (
                    node_map[node_id]["role"] not in {"conversion", "converted-policy"}
                    or node_map[node_id]["status"] != slot["status"]
                ):
                    _fail("conversion slot maps a nonconversion node or mismatched status")
            for edge_id in slot["edge_ids"]:
                if (
                    edge_map[edge_id]["relation"] != "schema_conversion"
                    or edge_map[edge_id]["status"] != slot["status"]
                ):
                    _fail("conversion slot maps a nonconversion edge or mismatched status")
        else:
            slot_nodes = [node_map[node_id] for node_id in slot["node_ids"]]
            if any(node["role"] not in allowed_roles[slot["slot"]] for node in slot_nodes):
                _fail(f"lineage slot {slot['slot']} maps the wrong role")
            statuses = {node["status"] for node in slot_nodes}
            expected_status = "present" if statuses == {"present"} else "missing"
            if not statuses <= {"present", "missing"} or slot["status"] != expected_status:
                _fail(f"lineage slot {slot['slot']} maps the wrong aggregate status")
        mapped_nodes.extend(slot["node_ids"])
        mapped_edges.extend(slot["edge_ids"])
    if len(mapped_nodes) != len(set(mapped_nodes)) or len(mapped_edges) != len(set(mapped_edges)):
        _fail("user-facing lineage slots duplicate a node or edge")
    trained_nodes = [node_map[node_id] for node_id in slots[-1]["node_ids"]]
    trained_runs = [node["run_identity"] for node in trained_nodes]
    arms = {row["arm_id"] for row in trained_runs}
    seeds_by_arm = {
        arm: {row["optimizer_seed"] for row in trained_runs if row["arm_id"] == arm} for arm in arms
    }
    expected_historical_seeds = {
        "T3per": set(range(70, 75)),
        "T3def": set(range(75, 80)),
        "T3lo": set(range(80, 85)),
    }
    if (
        arms != {"T3per", "T3def", "T3lo"}
        or any(len([row for row in trained_runs if row["arm_id"] == arm]) != 5 for arm in arms)
        or len({row["run_id"] for row in trained_runs}) != 15
        or seeds_by_arm != expected_historical_seeds
        or any(
            row["pairing_status"] != "unpaired" or row["pairing_group"] is not None
            for row in trained_runs
        )
    ):
        _fail("trained-t3 slot must bind the reviewed three-arm disjoint unpaired 5-seed history")
    manifest_nodes = {row.get("id"): row for row in document.get("nodes", [])}
    manifest_edges = {row.get("id"): row for row in document.get("edges", [])}
    if set(manifest_nodes) != set(node_map) or set(manifest_edges) != set(edge_map):
        _fail("manifest node/edge inventory differs from the reviewed lineage inventory")
    for node_id, prescribed in node_map.items():
        actual = manifest_nodes[node_id]
        if (
            actual.get("role") != prescribed["role"]
            or actual.get("status") != prescribed["status"]
            or actual.get("run_identity") != prescribed["run_identity"]
        ):
            _fail(f"manifest node {node_id} differs from reviewed role/status/run identity")
    for edge_id, prescribed in edge_map.items():
        actual = manifest_edges[edge_id]
        if any(
            actual.get(key) != prescribed[key]
            for key in ("source", "destination", "relation", "status")
        ):
            _fail(f"manifest edge {edge_id} differs from reviewed lineage inventory")
        if _equivalence_descriptor(actual.get("equivalence")) != prescribed["equivalence"]:
            _fail(f"manifest edge {edge_id} equivalence differs from reviewed lineage inventory")
    required_nodes, required_edges = body.get("required_nodes"), body.get("required_edges")
    if (
        not isinstance(required_nodes, list)
        or len(required_nodes) != len(set(required_nodes))
        or not isinstance(required_edges, list)
        or len(required_edges) != len(set(required_edges))
        or set(required_nodes) - set(node_map)
        or set(required_edges) - set(edge_map)
        or document.get("required_nodes") != required_nodes
        or document.get("required_edges") != required_edges
    ):
        _fail("manifest required registries differ from reviewed lineage inventory")
    sequences = body.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        _fail("lineage inventory has no reviewed branch/path sequences")
    sequenced = []
    sequence_ids = set()
    for sequence in sequences:
        if (
            not isinstance(sequence, dict)
            or set(sequence) != {"id", "branch", "edges"}
            or not isinstance(sequence.get("id"), str)
            or not sequence["id"]
            or sequence["id"] in sequence_ids
            or not isinstance(sequence.get("branch"), str)
            or not sequence["branch"]
            or not isinstance(sequence.get("edges"), list)
            or not sequence["edges"]
        ):
            _fail("lineage inventory has a malformed/duplicate sequence")
        sequence_ids.add(sequence["id"])
        previous = None
        for position, edge_id in enumerate(sequence["edges"]):
            edge = edge_map.get(edge_id)
            if edge is None or edge["sequence"] != sequence["id"] or edge["position"] != position:
                _fail(f"lineage sequence {sequence['id']} does not bind edge positions")
            if previous is not None and previous["destination"] != edge["source"]:
                _fail(f"lineage sequence {sequence['id']} is discontinuous or reversed")
            previous = edge
            sequenced.append(edge_id)
    if len(sequenced) != len(set(sequenced)) or set(sequenced) != set(required_edges):
        _fail("reviewed sequences must cover every required edge exactly once")
    if body.get("comparison_order") != sequenced or document.get("comparison_order") != sequenced:
        _fail("comparison order differs from reviewed continuous lineage sequences")
    positions = {edge_id: position for position, edge_id in enumerate(sequenced)}
    for upstream in protocol_edges:
        for downstream in protocol_edges:
            if (
                upstream["id"] != downstream["id"]
                and upstream["destination"] == downstream["source"]
                and positions[upstream["id"]] >= positions[downstream["id"]]
            ):
                _fail("comparison order reverses DAG ancestry")
    panes = {pane.get("id"): pane for pane in document.get("evaluation", {}).get("panes", [])}
    expected_nodes = defaultdict(set)
    for row in protocol_nodes:
        if (
            row["status"] == "present"
            and row["id"] in required_nodes
            and not row["historical_panes"]
        ):
            _fail(f"required present node {row['id']} lacks prescribed historical coverage")
        for pane_id in row["historical_panes"]:
            if pane_id not in panes or panes[pane_id].get("kind") not in {
                "historical-semantics",
                "fresh-common-producer-diagnostic",
            }:
                _fail(f"node {row['id']} historical applicability names a nonhistorical pane")
            expected_nodes[pane_id].add(row["id"])
        for pane_id in row["standardized_panes"]:
            if (
                pane_id not in panes
                or panes[pane_id].get("kind") != "standardized-flags-intervention"
            ):
                _fail(f"node {row['id']} standardized applicability names a nonstandardized pane")
            expected_nodes[pane_id].add(row["id"])
    for row in protocol_edges:
        if (
            row["status"] == "present"
            and row["id"] in required_edges
            and not row["historical_panes"]
        ):
            _fail(f"required present edge {row['id']} lacks prescribed historical coverage")
        for field, kinds in (
            ("historical_panes", {"historical-semantics", "fresh-common-producer-diagnostic"}),
            ("standardized_panes", {"standardized-flags-intervention"}),
        ):
            for pane_id in row[field]:
                if pane_id not in panes or panes[pane_id].get("kind") not in kinds:
                    _fail(f"edge {row['id']} applicability names the wrong pane kind")
                if (
                    row["source"] not in expected_nodes[pane_id]
                    or row["destination"] not in expected_nodes[pane_id]
                ):
                    _fail(f"edge {row['id']} applicability omits an endpoint from pane {pane_id}")
    if set(panes) != set(expected_nodes):
        _fail("manifest panes differ from reviewed lineage applicability")
    for pane_id, pane in panes.items():
        if set(pane.get("nodes", [])) != expected_nodes[pane_id]:
            _fail(f"pane {pane_id} nodes differ from reviewed lineage applicability")
    return body, document["lineage_inventory"]["sha256"]


def _raw_load(path: Path) -> dict:
    """Read a tensor checkpoint without importing arbitrary pickled classes."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # weights-only rejection is a diagnostic rejection
        _fail(f"checkpoint is not weights-only inspectable: {path}: {error}")
    if not isinstance(state, dict):
        _fail(f"checkpoint root is not a mapping: {path}")
    return state


def _implementation_artifact(ref: dict, base: Path, kind: str, what: str) -> dict:
    body, _ = _read_json_ref(ref, base, kind=kind)
    if (
        set(body) != {"schema", "kind", "identity", "commit", "source", "digest"}
        or not isinstance(body.get("identity"), str)
        or not body["identity"].strip()
        or not _is_sha(body.get("commit"), 40)
    ):
        _fail(f"{what} implementation identity is malformed")
    artifact(body.get("source"), base)
    return body


def _raw_region_head(state: dict) -> str | None:
    actor = state.get("actor", state)
    if isinstance(actor, dict) and "region_flat" in actor:
        raw = actor["region_flat"]
        if isinstance(raw, torch.Tensor) and raw.numel() == 1:
            return "flat" if bool(raw.item()) else "hierarchical"
        if isinstance(raw, (bool, int)):
            return "flat" if bool(raw) else "hierarchical"
        _fail("region_flat metadata is not scalar")
    value = state.get("region_head")
    return value if value in ("flat", "hierarchical") else None


def inspect_checkpoint(ref: dict, base: Path) -> tuple[dict, dict, Path]:
    """Hash and inspect raw metadata before any model is constructed."""
    path, sha = artifact(ref, base)
    before = checkpoint_hash(path)
    state = _raw_load(path)
    contract = state.get("contract")
    if isinstance(contract, bool) or not isinstance(contract, int):
        _fail(f"checkpoint {path} has no explicit integer contract; legacy defaults are forbidden")
    if contract not in SUPPORTED_RAW_CONTRACTS:
        _fail(f"checkpoint {path} has unknown/future contract {contract}")
    core_width = state.get("core_width")
    if isinstance(core_width, bool) or not isinstance(core_width, int) or core_width <= 0:
        _fail(f"checkpoint {path} has no explicit positive core_width")
    actor = state.get("actor", state)
    if not isinstance(actor, dict) or not actor or not all(isinstance(k, str) for k in actor):
        _fail(f"checkpoint {path} has no raw actor state")
    normalizer = state.get("normalizer")
    if not isinstance(normalizer, dict) or not normalizer:
        _fail(f"checkpoint {path} has no raw normalizer state")
    run = state.get("run_manifest") if isinstance(state.get("run_manifest"), dict) else {}
    raw = {
        "checkpoint_contract": contract,
        "core_width": core_width,
        "token": state.get("token") if "token" in state else None,
        "experts": state.get("experts") if "experts" in state else None,
        "vector_critic": state.get("vector_critic") if "vector_critic" in state else None,
        "commitments": state.get("commitments") if "commitments" in state else None,
        "learned_movement": state.get("learned_movement") if "learned_movement" in state else None,
        "region_movement": state.get("region_movement") if "region_movement" in state else None,
        "learned_fission": state.get("learned_fission") if "learned_fission" in state else None,
        "learned_trade": state.get("learned_trade") if "learned_trade" in state else None,
        "region_head": _raw_region_head(state),
        "actor_keys_sha256": digest(sorted(actor)),
        "normalizer_keys_sha256": digest(sorted(normalizer)),
        "producer_contract_digest": (run.get("worker_contract") or {}).get("digest"),
    }
    after = checkpoint_hash(path)
    if before != sha or after != sha:
        _fail(f"mutable checkpoint changed during raw inspection: {path}")
    return raw, state, path


def _status(record: dict, allowed: set[str], what: str) -> str:
    status = record.get("status")
    if status not in allowed:
        _fail(f"{what} has unknown status {status!r}")
    if status != "present" and not str(record.get("reason", "")).strip():
        _fail(f"{what} {status} status needs a reason")
    return status


def _contract_declaration(value: Any, name: str) -> None:
    if isinstance(value, str) and value.strip():
        return
    if (
        isinstance(value, dict)
        and value
        and (value.get("status") == "not-applicable" or value.get("schema") is not None)
    ):
        return
    _fail(f"node {name} contract is not explicit")


def _defaults(node: dict, base: Path, checkpoint_sha: str) -> dict:
    loading = node.get("loading")
    if not isinstance(loading, dict) or loading.get("mode") not in LOADING_MODES:
        _fail(f"node {node.get('id')} lacks an explicit loading mode")
    record, _ = _read_json_ref(
        loading.get("default_resolution"), base, kind="checkpoint_default_resolution"
    )
    if (
        record.get("checkpoint_sha256") != checkpoint_sha
        or record.get("implicit_defaults") != []
        or not isinstance(record.get("resolved"), dict)
    ):
        _fail(f"node {node.get('id')} default resolution permits implicit or unbound defaults")
    mode = loading["mode"]
    if mode in {"pinned-historical", "reviewed-adapter"}:
        _, implementation_sha = artifact(loading.get("implementation"), base)
    if mode == "reviewed-adapter":
        migration, _ = _read_json_ref(
            loading.get("migration_record"), base, kind="checkpoint_migration"
        )
        if (
            migration.get("status") != "reconstructed"
            or migration.get("historical_conversion") is not False
            or migration.get("destination_checkpoint_sha256") != checkpoint_sha
            or migration.get("adapter", {}).get("sha256") != implementation_sha
        ):
            _fail(
                "reviewed adapters must declare reconstructed, non-historical migration semantics"
            )
    return record


def _validate_node(node: dict, base: Path) -> tuple[dict, dict | None, Path | None, list[str]]:
    node_id = node.get("id")
    if not isinstance(node_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", node_id):
        _fail("node id must be a stable lowercase identifier")
    if not isinstance(node.get("role"), str) or not node["role"].strip():
        _fail(f"node {node_id} lacks a role")
    if node.get("evidence_class") not in EVIDENCE_CLASSES:
        _fail(f"node {node_id} lacks a valid evidence class")
    run_identity = _run_identity(node.get("run_identity"), f"node {node_id}")
    status = _status(node, NODE_STATUSES, f"node {node_id}")
    if status != "present":
        return (
            {
                "id": node_id,
                "status": status,
                "role": node["role"],
                "evidence_class": node["evidence_class"],
                "reason": node["reason"],
                "run_identity": run_identity,
            },
            None,
            None,
            [
                f"node:{node_id}:{status}",
            ],
        )
    historical = node.get("historical_status")
    if historical not in {"documented", "reference-only", "reconstructed"}:
        _fail(f"node {node_id} has invalid historical_status")
    if historical == "reconstructed" and node.get("claims_historical_conversion") is not False:
        _fail(f"reconstructed node {node_id} must explicitly deny historical conversion")
    producer_status = node.get("historical_producer_status", "embedded-worker-contract")
    if producer_status not in {"embedded-worker-contract", "unresolved"}:
        _fail(f"node {node_id} has invalid historical producer status")
    raw, state, path = inspect_checkpoint(node.get("artifact"), base)
    declared = node.get("raw_schema")
    if not isinstance(declared, dict) or declared != raw:
        _fail(f"node {node_id} raw schema differs from checkpoint metadata")
    contracts_decl = node.get("contracts")
    if not isinstance(contracts_decl, dict) or set(contracts_decl) != {
        "actor",
        "critic",
        "normalizer",
    }:
        _fail(f"node {node_id} must bind actor, critic and normalizer contracts")
    for name, value in contracts_decl.items():
        _contract_declaration(value, f"{node_id}/{name}")
    learner = node.get("producing_learner")
    if (
        not isinstance(learner, dict)
        or not str(learner.get("identity", "")).strip()
        or not _is_sha(learner.get("commit"), 40)
    ):
        _fail(f"node {node_id} lacks a full producing learner identity")
    _implementation_artifact(
        node.get("inference_implementation"),
        base,
        "checkpoint_inference_implementation",
        f"node {node_id}",
    )
    for name in ("producer_evidence", "run_evidence"):
        artifact(node.get(name), base)
    sources = node.get("source_artifacts")
    if not isinstance(sources, list):
        _fail(f"node {node_id} source_artifacts must be an explicit list")
    for ref in sources:
        artifact(ref, base)
    default_record = _defaults(node, base, node["artifact"]["sha256"])
    semantics = node.get("historical_semantics")
    if not isinstance(semantics, dict) or set(semantics) != set(SEMANTIC_FIELDS):
        _fail(f"node {node_id} must resolve every historical decoder/head/default semantic")
    if (
        any(type(semantics[k]) is not bool for k in BOOLEAN_SEMANTIC_FIELDS)
        or semantics["region_head"] not in ("flat", "hierarchical")
        or semantics["region_selection"] not in (None, "argmax", "stay")
    ):
        _fail(f"node {node_id} has invalid historical semantics")
    for key in (
        "experts",
        "token",
        "commitments",
        "learned_movement",
        "region_movement",
        "learned_fission",
        "learned_trade",
    ):
        if key in state:
            if type(state[key]) is not bool:
                _fail(f"node {node_id} raw checkpoint semantic {key} is not boolean")
            if state[key] != semantics[key]:
                _fail(
                    f"node {node_id} historical semantic {key} differs from raw checkpoint metadata"
                )
    if raw["region_head"] is not None and raw["region_head"] != semantics["region_head"]:
        _fail(f"node {node_id} historical region head differs from raw checkpoint metadata")
    resolved = default_record["resolved"]
    for key in SEMANTIC_FIELDS:
        if (
            key not in resolved
            or not isinstance(resolved[key], dict)
            or resolved[key].get("value") != semantics[key]
        ):
            _fail(f"node {node_id} semantic {key} is not bound by default resolution")
    return (
        {
            "id": node_id,
            "status": status,
            "role": node["role"],
            "sha256": node["artifact"]["sha256"],
            "raw_schema": raw,
            "historical_status": historical,
            "semantics": semantics,
            "run_identity": run_identity,
            "historical_producer_status": producer_status,
            "inference_implementation": node["inference_implementation"],
        },
        state,
        path,
        [],
    )


def _tensor(state: dict, dotted: str) -> torch.Tensor:
    """Resolve a dotted declaration without splitting dots inside state-dict keys."""
    parts = dotted.split(".")
    matches: list[tuple[tuple[str, ...], Any]] = []

    def visit(value: Any, start: int, route: tuple[str, ...]) -> None:
        if not isinstance(value, dict):
            return
        for end in range(start + 1, len(parts) + 1):
            key = ".".join(parts[start:end])
            if key not in value:
                continue
            child = value[key]
            child_route = (*route, key)
            if end == len(parts):
                matches.append((child_route, child))
            else:
                visit(child, end, child_route)

    visit(state, 0, ())
    if not matches:
        _fail(f"declared tensor {dotted!r} is missing")
    if len(matches) != 1:
        _fail(f"declared tensor {dotted!r} is ambiguous across mapping/state-dict keys")
    value = matches[0][1]
    if not isinstance(value, torch.Tensor):
        _fail(f"declared tensor {dotted!r} is not a tensor")
    return value


def _claim_digest(claim: dict) -> str:
    return digest({k: v for k, v in claim.items() if k != "result"})


def _claim_protocol_digest(claim: dict) -> str:
    return digest({k: v for k, v in claim.items() if k not in {"result", "execution_protocol"}})


def _equivalence_corpus(ref: dict, base: Path) -> tuple[dict[str, dict], str]:
    corpus, _ = _read_json_ref(ref, base, kind="equivalence_corpus")
    if set(corpus) != {"schema", "kind", "cases", "digest"} or not isinstance(
        corpus.get("cases"), list
    ):
        _fail("equivalence corpus is noncanonical")
    cases = {}
    for row in corpus["cases"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "input"}
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or row["id"] in cases
        ):
            _fail("equivalence corpus contains a malformed/duplicate case")
        artifact(row["input"], base)
        cases[row["id"]] = row
    if not cases:
        _fail("equivalence corpus is empty")
    return cases, ref["sha256"]


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _action_contract(schema: Any, what: str = "decoded-action schema") -> dict:
    if not isinstance(schema, dict) or schema.get("type") not in {
        "integer",
        "number",
        "boolean",
        "string",
        "array",
        "object",
        "null",
    }:
        _fail(f"{what} is malformed")
    kind = schema["type"]
    if kind == "integer":
        if (
            set(schema) != {"type", "minimum", "maximum"}
            or type(schema["minimum"]) is not int
            or type(schema["maximum"]) is not int
            or schema["minimum"] > schema["maximum"]
        ):
            _fail(f"{what} integer bounds are malformed")
    elif kind == "number":
        if (
            set(schema) != {"type", "minimum", "maximum"}
            or not _finite_number(schema["minimum"])
            or not _finite_number(schema["maximum"])
            or schema["minimum"] > schema["maximum"]
        ):
            _fail(f"{what} numeric bounds are malformed")
    elif kind == "boolean" or kind == "null":
        if set(schema) != {"type"}:
            _fail(f"{what} {kind} declaration is malformed")
    elif kind == "string":
        enum = schema.get("enum")
        if (
            set(schema) != {"type", "enum"}
            or not isinstance(enum, list)
            or not enum
            or any(not isinstance(item, str) for item in enum)
            or len(enum) != len(set(enum))
        ):
            _fail(f"{what} string enum is malformed")
    elif kind == "array":
        if (
            set(schema) != {"type", "length", "items"}
            or type(schema.get("length")) is not int
            or schema["length"] < 0
        ):
            _fail(f"{what} array declaration is malformed")
        _action_contract(schema.get("items"), what + "/items")
    else:
        fields = schema.get("fields")
        if (
            set(schema) != {"type", "fields"}
            or not isinstance(fields, dict)
            or any(not isinstance(name, str) or not name for name in fields)
        ):
            _fail(f"{what} object declaration is malformed")
        for name, child in fields.items():
            _action_contract(child, what + "/" + name)
    return schema


def _action_matches(value: Any, schema: dict) -> bool:
    kind = schema["type"]
    if kind == "integer":
        return type(value) is int and schema["minimum"] <= value <= schema["maximum"]
    if kind == "number":
        return _finite_number(value) and schema["minimum"] <= value <= schema["maximum"]
    if kind == "boolean":
        return type(value) is bool
    if kind == "string":
        return isinstance(value, str) and value in schema["enum"]
    if kind == "null":
        return value is None
    if kind == "array":
        return (
            isinstance(value, list)
            and len(value) == schema["length"]
            and all(_action_matches(item, schema["items"]) for item in value)
        )
    return (
        isinstance(value, dict)
        and set(value) == set(schema["fields"])
        and all(_action_matches(value[name], child) for name, child in schema["fields"].items())
    )


def _output_contracts(
    ref: dict,
    base: Path,
    cases: dict[str, dict],
    scopes: list[str],
    behavioral: bool,
    runtime_dtype: str,
) -> tuple[dict[tuple[str, str], dict], str]:
    body, _ = _read_json_ref(ref, base, kind="equivalence_output_contracts")
    if set(body) != {"schema", "kind", "contracts", "digest"} or not isinstance(
        body.get("contracts"), list
    ):
        _fail("equivalence output contracts are noncanonical")
    contracts_by_key = {}
    for row in body["contracts"]:
        required = {"case_id", "scope", "dtype", "shape", "components", "mask", "decoded_action"}
        shape = row.get("shape") if isinstance(row, dict) else None
        components = row.get("components") if isinstance(row, dict) else None
        key = (row.get("case_id"), row.get("scope")) if isinstance(row, dict) else (None, None)
        if (
            not isinstance(row, dict)
            or set(row) != required
            or key in contracts_by_key
            or key[0] not in cases
            or key[1] not in scopes
            or row.get("dtype") not in {"float32", "float64"}
            or row.get("dtype") != runtime_dtype
            or not isinstance(shape, list)
            or any(type(width) is not int or width <= 0 for width in shape)
            or not isinstance(components, list)
            or any(not isinstance(label, str) or not label for label in components)
            or len(components) != len(set(components))
            or len(components) != (math.prod(shape) if shape else 1)
        ):
            _fail("equivalence output contract has an invalid dtype/shape/component layout")
        mask = row["mask"]
        action = row["decoded_action"]
        if behavioral:
            if (
                not isinstance(mask, dict)
                or set(mask) != {"shape", "components"}
                or not isinstance(mask.get("shape"), list)
                or any(type(width) is not int or width <= 0 for width in mask["shape"])
                or not isinstance(mask.get("components"), list)
                or any(not isinstance(label, str) or not label for label in mask["components"])
                or len(mask["components"]) != len(set(mask["components"]))
                or len(mask["components"]) != (math.prod(mask["shape"]) if mask["shape"] else 1)
            ):
                _fail("behavioral output contract has an invalid mask schema")
            _action_contract(action)
        elif mask is not None or action is not None:
            _fail("nonbehavioral output contract must declare null mask/action schemas")
        contracts_by_key[key] = row
    expected = {(case_id, scope) for case_id in cases for scope in scopes}
    if set(contracts_by_key) != expected:
        _fail("equivalence output contracts omit or duplicate a case/scope")
    return contracts_by_key, ref["sha256"]


def _normalized_numeric(value: Any, dtype: str) -> float:
    if type(value) not in (int, float):
        _fail("equivalence raw numeric output is malformed or nonfinite")
    try:
        source = float(value)
        if not math.isfinite(source):
            _fail("equivalence raw numeric output is malformed or nonfinite")
        packed = struct.pack(">f" if dtype == "float32" else ">d", source)
        normalized = struct.unpack(">f" if dtype == "float32" else ">d", packed)[0]
    except (OverflowError, struct.error):
        _fail(f"equivalence raw numeric output exceeds declared {dtype} range")
    if not math.isfinite(normalized):
        _fail(f"equivalence raw numeric output is nonfinite after {dtype} cast")
    if type(value) is int:
        # JSON integers arrive with arbitrary precision.  Comparing only with
        # ``source`` would compare against an already-rounded Python float and
        # accept, for example, 2**53+1 as float64 or 2**60+1 as float32.
        representable = normalized.is_integer() and int(normalized) == value
    else:
        representable = normalized == source and not (
            normalized == 0 and math.copysign(1.0, normalized) != math.copysign(1.0, source)
        )
    if not representable:
        _fail(f"equivalence raw numeric output is not canonically representable as {dtype}")
    return normalized


def _equivalence_context(row: dict, output_contract_sha: str) -> str:
    fields = (
        "claim_sha256",
        "protocol_sha256",
        "case_id",
        "scope",
        "side",
        "checkpoint_sha256",
        "input",
        "producer_implementation_sha256",
        "executor_implementation_sha256",
        "numeric_runtime",
        "rng_schedule_sha256",
    )
    return digest(
        {
            **{field: row.get(field) for field in fields},
            "output_contract_sha256": output_contract_sha,
        }
    )


def _equivalence_output(
    ref: dict,
    base: Path,
    scope: str,
    behavioral: bool,
    contract: dict,
    contract_sha: str,
    context_sha: str,
) -> dict:
    output, _ = _read_json_ref(ref, base, kind="equivalence_raw_output")
    required = {
        "schema",
        "kind",
        "scope",
        "dtype",
        "shape",
        "components",
        "values",
        "mask",
        "decoded_action",
        "output_contract_sha256",
        "context_sha256",
        "digest",
    }
    shape = output.get("shape")
    values = output.get("values")
    if (
        set(output) != required
        or output.get("scope") != scope
        or output.get("dtype") != contract["dtype"]
        or shape != contract["shape"]
        or output.get("components") != contract["components"]
        or output.get("output_contract_sha256") != contract_sha
        or output.get("context_sha256") != context_sha
        or not isinstance(values, list)
        or len(values) != len(contract["components"])
    ):
        _fail(
            "equivalence raw output differs from its reviewed "
            "dtype/shape/component/context contract"
        )
    output["values"] = [_normalized_numeric(value, contract["dtype"]) for value in values]
    width = math.prod(shape) if shape else 1
    if len(values) != width:
        _fail("equivalence raw output values do not match declared shape")
    mask, action = output["mask"], output["decoded_action"]
    if behavioral:
        mask_contract = contract["mask"]
        if (
            not isinstance(mask, dict)
            or set(mask) != {"shape", "components", "values"}
            or mask.get("shape") != mask_contract["shape"]
            or mask.get("components") != mask_contract["components"]
            or not isinstance(mask.get("values"), list)
            or any(type(item) is not bool for item in mask["values"])
            or len(mask["values"]) != len(mask_contract["components"])
            or not _action_matches(action, contract["decoded_action"])
        ):
            _fail("behavioral equivalence raw mask/action output is malformed")
    elif mask is not None or action is not None:
        _fail("nonbehavioral equivalence output cannot assert mask/action evidence")
    return output


def _literal_tensor_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare tensor storage values literally after dtype/shape agreement."""
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    supported = {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    }
    if (
        left.layout != torch.strided
        or right.layout != torch.strided
        or left.is_quantized
        or right.is_quantized
    ):
        _fail("exact tensor-byte comparison supports only dense strided nonquantized tensors")
    if left.dtype not in supported:
        _fail(f"exact tensor-byte comparison does not support dtype {left.dtype}")
    left_bytes = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    right_bytes = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return left_bytes == right_bytes


def _validate_equivalence(
    edge: dict, source: dict, destination: dict, states: dict[str, dict], base: Path
) -> dict:
    claim = edge["equivalence"]
    required = {
        "kind",
        "source_schema",
        "destination_schema",
        "numeric_runtime",
        "mapped_tensors",
        "mapped_outputs",
        "appended_outputs",
        "action_decoder",
        "active_heads",
        "delegated_heads",
        "normalizer_mapping",
        "tolerances",
        "corpus",
    }
    if not isinstance(claim, dict) or claim.get("kind") not in EQUIVALENCE_KINDS:
        _fail(f"edge {edge['id']} equivalence claim is incomplete")
    expected_fields = set(required)
    if claim["kind"] != "exact_tensor_preservation":
        expected_fields.update(("execution_protocol", "result"))
    if claim["kind"] == "stochastic_behavioral_equivalence":
        expected_fields.add("rng_schedule")
    if set(claim) != expected_fields:
        _fail(f"edge {edge['id']} equivalence claim has a noncanonical field set")
    if (
        claim["source_schema"] != source["raw_schema"]
        or claim["destination_schema"] != destination["raw_schema"]
    ):
        _fail(f"edge {edge['id']} equivalence schemas do not bind the raw artifacts")
    runtime = claim["numeric_runtime"]
    if (
        not isinstance(runtime, dict)
        or not all(
            str(runtime.get(k, "")).strip()
            for k in ("framework", "version", "device", "dtype", "determinism")
        )
        or runtime.get("dtype") not in {"float32", "float64"}
    ):
        _fail(f"edge {edge['id']} lacks frozen numeric runtime")
    mappings = claim["mapped_tensors"]
    if not isinstance(mappings, list) or any(set(m) != {"source", "destination"} for m in mappings):
        _fail(f"edge {edge['id']} tensor mapping is malformed")
    if not isinstance(claim["mapped_outputs"], list) or len(set(claim["mapped_outputs"])) != len(
        claim["mapped_outputs"]
    ):
        _fail(f"edge {edge['id']} output scope is malformed")
    appended = claim["appended_outputs"]
    if not isinstance(appended, list) or any(
        not isinstance(x, dict)
        or set(x) != {"name", "disposition"}
        or x["disposition"] not in {"disabled", "delegated"}
        for x in appended
    ):
        _fail(f"edge {edge['id']} appended outputs must be disabled or delegated")
    active, delegated = claim["active_heads"], claim["delegated_heads"]
    if (
        not isinstance(active, list)
        or not isinstance(delegated, list)
        or set(active) & set(delegated)
        or any(x["disposition"] == "delegated" and x["name"] not in delegated for x in appended)
    ):
        _fail(f"edge {edge['id']} active/delegated head scope conflicts")
    if not isinstance(claim["action_decoder"], dict) or not isinstance(
        claim["normalizer_mapping"], dict
    ):
        _fail(f"edge {edge['id']} lacks decoder/normalizer mapping")
    tolerance = claim["tolerances"]
    if (
        not isinstance(tolerance, dict)
        or set(tolerance) != {"absolute", "relative"}
        or any(
            type(tolerance[k]) not in (int, float)
            or not math.isfinite(tolerance[k])
            or tolerance[k] < 0
            for k in tolerance
        )
    ):
        _fail(f"edge {edge['id']} has invalid tolerances")
    corpus_cases, corpus_sha = _equivalence_corpus(claim["corpus"], base)
    result = {
        "kind": claim["kind"],
        "claim_sha256": _claim_digest(claim),
        "corpus_sha256": corpus_sha,
        "status": "PASS",
        "checks": [],
    }
    if claim["kind"] == "exact_tensor_preservation":
        if tolerance != {"absolute": 0, "relative": 0} or not mappings:
            _fail(
                f"edge {edge['id']} exact tensor claim needs zero tolerance and a nonempty mapping"
            )
        for mapping in mappings:
            left = _tensor(states[source["id"]], mapping["source"])
            right = _tensor(states[destination["id"]], mapping["destination"])
            equal = _literal_tensor_equal(left, right)
            result["checks"].append(
                {**mapping, "equal": equal, "dtype": str(left.dtype), "shape": list(left.shape)}
            )
        if not all(row["equal"] for row in result["checks"]):
            result["status"] = "FAIL"
        if "result" in claim:
            _fail(
                f"edge {edge['id']} exact tensor result is computed from raw artifacts, "
                "not accepted by assertion"
            )
        return result
    if not claim["mapped_outputs"]:
        _fail(f"edge {edge['id']} forward/behavioral claim has empty mapped output scope")
    behavioral = claim["kind"] in {
        "deterministic_behavioral_equivalence",
        "stochastic_behavioral_equivalence",
    }
    protocol, _ = _read_json_ref(
        claim["execution_protocol"],
        base,
        kind="equivalence_execution_protocol",
    )
    protocol_fields = {
        "schema",
        "kind",
        "claim_projection_sha256",
        "executor_implementation",
        "source_implementation",
        "destination_implementation",
        "numeric_runtime",
        "scopes",
        "behavioral",
        "rng_schedule",
        "output_contracts",
        "digest",
    }
    if (
        set(protocol) != protocol_fields
        or protocol.get("claim_projection_sha256") != _claim_protocol_digest(claim)
        or protocol.get("source_implementation") != source["inference_implementation"]
        or protocol.get("destination_implementation") != destination["inference_implementation"]
        or protocol.get("numeric_runtime") != runtime
        or protocol.get("scopes") != claim["mapped_outputs"]
        or protocol.get("behavioral") is not behavioral
    ):
        _fail(f"edge {edge['id']} execution protocol is not bound to claim/runtime/implementations")
    _implementation_artifact(
        protocol.get("executor_implementation"),
        base,
        "equivalence_executor_implementation",
        f"edge {edge['id']} equivalence executor",
    )
    output_contracts, output_contract_sha = _output_contracts(
        protocol.get("output_contracts"),
        base,
        corpus_cases,
        claim["mapped_outputs"],
        behavioral,
        runtime["dtype"],
    )
    rng_sha = None
    if claim["kind"] == "stochastic_behavioral_equivalence":
        schedule, _ = _read_json_ref(claim["rng_schedule"], base, kind="equivalence_rng_schedule")
        if set(schedule) != {"schema", "kind", "cases", "digest"} or not isinstance(
            schedule.get("cases"), list
        ):
            _fail(f"edge {edge['id']} stochastic RNG schedule is malformed")
        scheduled = {}
        for row in schedule["cases"]:
            if (
                not isinstance(row, dict)
                or set(row) != {"case_id", "draws"}
                or row.get("case_id") in scheduled
                or not isinstance(row.get("draws"), list)
                or not row["draws"]
                or any(type(value) is not int or value < 0 for value in row["draws"])
            ):
                _fail(f"edge {edge['id']} stochastic RNG case is malformed")
            scheduled[row["case_id"]] = row["draws"]
        if (
            set(scheduled) != set(corpus_cases)
            or protocol.get("rng_schedule") != claim["rng_schedule"]
        ):
            _fail(f"edge {edge['id']} stochastic RNG schedule omits/substitutes corpus cases")
        rng_sha = claim["rng_schedule"]["sha256"]
    elif protocol.get("rng_schedule") is not None:
        _fail(f"edge {edge['id']} deterministic execution protocol carries an RNG schedule")
    evidence, _ = _read_json_ref(
        claim["result"],
        base,
        kind="equivalence_execution_result",
    )
    result_fields = {
        "schema",
        "kind",
        "claim_sha256",
        "protocol_sha256",
        "corpus_sha256",
        "executions",
        "digest",
    }
    if (
        set(evidence) != result_fields
        or evidence.get("claim_sha256") != result["claim_sha256"]
        or evidence.get("protocol_sha256") != claim["execution_protocol"]["sha256"]
        or evidence.get("corpus_sha256") != corpus_sha
        or not isinstance(evidence.get("executions"), list)
    ):
        _fail(f"edge {edge['id']} equivalence execution result is assertion-only or unbound")
    executions = {}
    for execution_ref in evidence["executions"]:
        row, _ = _read_json_ref(execution_ref, base, kind="equivalence_execution")
        row_fields = {
            "schema",
            "kind",
            "claim_sha256",
            "protocol_sha256",
            "case_id",
            "scope",
            "side",
            "checkpoint_sha256",
            "producer_implementation_sha256",
            "executor_implementation_sha256",
            "numeric_runtime",
            "rng_schedule_sha256",
            "input",
            "output_contract_sha256",
            "context_sha256",
            "output",
            "digest",
        }
        key = (row.get("case_id"), row.get("scope"), row.get("side"))
        endpoint = source if row.get("side") == "source" else destination
        expected_case = corpus_cases.get(row.get("case_id"))
        expected_context = _equivalence_context(row, output_contract_sha)
        if (
            set(row) != row_fields
            or key in executions
            or expected_case is None
            or row.get("scope") not in claim["mapped_outputs"]
            or row.get("side") not in {"source", "destination"}
            or row.get("claim_sha256") != result["claim_sha256"]
            or row.get("protocol_sha256") != claim["execution_protocol"]["sha256"]
            or row.get("checkpoint_sha256") != endpoint["sha256"]
            or row.get("producer_implementation_sha256")
            != endpoint["inference_implementation"]["sha256"]
            or row.get("executor_implementation_sha256")
            != protocol["executor_implementation"]["sha256"]
            or row.get("numeric_runtime") != runtime
            or row.get("rng_schedule_sha256") != rng_sha
            or row.get("input") != expected_case["input"]
            or row.get("output_contract_sha256") != output_contract_sha
            or row.get("context_sha256") != expected_context
        ):
            _fail(f"edge {edge['id']} equivalence execution is duplicated, malformed or unbound")
        artifact(row["input"], base)
        output = _equivalence_output(
            row["output"],
            base,
            row["scope"],
            behavioral,
            output_contracts[key[:2]],
            output_contract_sha,
            expected_context,
        )
        executions[key] = output
    expected_keys = {
        (case_id, scope, side)
        for case_id in corpus_cases
        for scope in claim["mapped_outputs"]
        for side in ("source", "destination")
    }
    if set(executions) != expected_keys:
        _fail(f"edge {edge['id']} equivalence execution omits a case/scope/side")
    checks = []
    for scope in claim["mapped_outputs"]:
        absolute_errors = []
        relative_errors = []
        masks_equal = True
        actions_equal = True
        for case_id in corpus_cases:
            left = executions[(case_id, scope, "source")]
            right = executions[(case_id, scope, "destination")]
            if left["dtype"] != right["dtype"] or left["shape"] != right["shape"]:
                _fail(f"edge {edge['id']} equivalence output dtype/shape differs across sides")
            for source_value, destination_value in zip(
                left["values"], right["values"], strict=True
            ):
                absolute = abs(destination_value - source_value)
                denominator = max(abs(source_value), abs(destination_value))
                relative = 0.0 if denominator == 0 else absolute / denominator
                if not math.isfinite(absolute) or not math.isfinite(relative):
                    _fail(f"edge {edge['id']} equivalence error computation is nonfinite")
                absolute_errors.append(absolute)
                relative_errors.append(relative)
            if behavioral:
                masks_equal &= left["mask"] == right["mask"]
                actions_equal &= left["decoded_action"] == right["decoded_action"]
        check = {
            "scope": scope,
            "case_count": len(corpus_cases),
            "value_count": len(absolute_errors),
            "max_absolute_error": max(absolute_errors),
            "max_relative_error": max(relative_errors),
            "masks_equal": (masks_equal if behavioral else None),
            "decoded_actions_equal": (actions_equal if behavioral else None),
        }
        if (
            check["max_absolute_error"] > tolerance["absolute"]
            and check["max_relative_error"] > tolerance["relative"]
        ):
            result["status"] = "FAIL"
        if behavioral and (not masks_equal or not actions_equal):
            result["status"] = "FAIL"
        checks.append(check)
    result["checks"] = checks
    return result


def _validate_migration(edge: dict, source: dict, destination: dict, base: Path) -> dict:
    record, _ = _read_json_ref(edge.get("migration_record"), base, kind="checkpoint_migration")
    if (
        record.get("source_checkpoint_sha256") != source["sha256"]
        or record.get("destination_checkpoint_sha256") != destination["sha256"]
        or record.get("source_contract") != source["raw_schema"]["checkpoint_contract"]
        or record.get("destination_contract") != destination["raw_schema"]["checkpoint_contract"]
        or record.get("status") not in {"historical", "reconstructed"}
    ):
        _fail(f"edge {edge['id']} migration record is not bound to its endpoints")
    artifact(record.get("adapter"), base)
    artifact(record.get("default_resolution"), base)
    if record["status"] == "reconstructed":
        if (
            record.get("historical_conversion") is not False
            or destination.get("historical_status") != "reconstructed"
        ):
            _fail(f"edge {edge['id']} reconstructed migration cannot claim historical conversion")
    elif record.get("historical_conversion") is not True:
        _fail(f"edge {edge['id']} historical migration needs affirmative contemporaneous evidence")
    if record["status"] == "historical":
        artifact(record.get("historical_evidence"), base)
    return record


def _topological(node_ids: set[str], edges: list[dict]) -> list[str]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    degree = {node: 0 for node in node_ids}
    for edge in edges:
        if edge["status"] != "present":
            continue
        source, destination = edge["source"], edge["destination"]
        adjacency[source].append(destination)
        degree[destination] += 1
    queue = deque(sorted(node for node, count in degree.items() if count == 0))
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in sorted(adjacency[node]):
            degree[child] -= 1
            if degree[child] == 0:
                queue.append(child)
    if len(order) != len(node_ids):
        _fail("artifact lineage contains a cycle")
    return order


def _validate_pane(pane: dict, nodes: dict[str, dict], base: Path) -> dict:
    pane_id = pane.get("id")
    if not isinstance(pane_id, str) or not pane_id:
        _fail("evaluation pane lacks an id")
    if pane.get("kind") not in PANE_KINDS:
        _fail(f"pane {pane_id} has unknown semantics")
    selected = pane.get("nodes")
    if (
        not isinstance(selected, list)
        or not selected
        or any(node not in nodes for node in selected)
    ):
        _fail(f"pane {pane_id} names unknown/empty nodes")
    worlds = pane.get("worlds")
    if not isinstance(worlds, list) or not worlds:
        _fail(f"pane {pane_id} has no frozen worlds")
    reserved = {seed for values in ROLES.values() for seed in values} | DENIED
    identities = set()
    for world in worlds:
        if not isinstance(world, dict) or world.get("family") != "evaluation":
            _fail(f"pane {pane_id} worlds must explicitly use the evaluation family")
        seed = world.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in reserved:
            _fail(f"pane {pane_id} uses a training/selection/calibration/confirmation seed")
        if (
            type(world.get("withheld")) is not int
            or world["withheld"] < 0
            or type(world.get("warmup")) is not int
            or world["warmup"] < 0
            or type(world.get("horizon")) is not int
            or world["horizon"] <= 0
        ):
            _fail(f"pane {pane_id} world controls are incomplete")
        turn = world.get("turn")
        reset_turn = world.get("reset_turn")
        players = world.get("players", 8)
        cohort_digest = world.get("cohort_digest", "")
        if (
            type(reset_turn) is not int
            or reset_turn < 0
            or (turn is not None and (type(turn) is not int or turn < 0))
            or type(players) is not int
            or players <= 0
            or not isinstance(cohort_digest, str)
            or (cohort_digest and not _is_sha(cohort_digest))
        ):
            _fail(f"pane {pane_id} world descriptor inputs are invalid")
        snapshot = world.get("snapshot")
        snapshot_sha = ""
        if snapshot is not None:
            _, snapshot_sha = artifact(snapshot, base)
            if turn != reset_turn:
                _fail(f"pane {pane_id} snapshot turn differs from its reviewed reset turn")
        identity = (
            seed,
            world["withheld"],
            world["warmup"],
            world["horizon"],
            snapshot_sha,
            turn,
            reset_turn,
            players,
            cohort_digest,
        )
        if identity in identities:
            _fail(f"pane {pane_id} contains a duplicate world")
        identities.add(identity)
    if pane.get("action_mode") not in {"deterministic", "seeded-stochastic"}:
        _fail(f"pane {pane_id} has no explicit action mode")
    from .gate1_diagnostics import (
        draw_order_from_semantics,
        validate_capture_bounds,
        validate_rng_schedule,
    )

    capture_bounds = pane.get("capture_bounds")
    for world_record in worlds:
        try:
            validate_capture_bounds(capture_bounds, world_record["horizon"])
        except ValueError as exc:
            _fail(f"pane {pane_id} {exc}")
    if pane["action_mode"] == "seeded-stochastic":
        schedule, _ = _read_json_ref(
            pane.get("policy_rng_schedule"), base, kind="gate1_policy_rng_schedule"
        )
        scheduled_worlds = schedule.get("worlds")
        expected_world_ids = {_world_id(world_record) for world_record in worlds}
        if (
            not isinstance(scheduled_worlds, list)
            or len(scheduled_worlds) != len(expected_world_ids)
            or {row.get("world_id") for row in scheduled_worlds if isinstance(row, dict)}
            != expected_world_ids
        ):
            _fail(
                f"pane {pane_id} policy RNG schedule world coverage differs from the "
                "reviewed cohort"
            )
        for scheduled_world in scheduled_worlds:
            scheduled_nodes = scheduled_world.get("nodes")
            if (
                not isinstance(scheduled_nodes, list)
                or len(scheduled_nodes) != len(selected)
                or {row.get("node_id") for row in scheduled_nodes if isinstance(row, dict)}
                != set(selected)
            ):
                _fail(
                    f"pane {pane_id} policy RNG schedule node coverage differs from the "
                    "reviewed executions"
                )
        for world_record in worlds:
            for node_id in selected:
                if "semantics" not in nodes[node_id]:
                    _fail(
                        f"pane {pane_id} cannot schedule stochastic execution for an "
                        "unavailable node"
                    )
                semantics = dict(nodes[node_id]["semantics"])
                if pane["kind"] == "standardized-flags-intervention":
                    semantics.update(pane.get("standardized_overrides", {}))
                try:
                    validate_rng_schedule(
                        schedule,
                        _world_id(world_record),
                        world_record["horizon"],
                        node_id=node_id,
                        expected_draw_order=draw_order_from_semantics(semantics),
                    )
                except ValueError as exc:
                    _fail(f"pane {pane_id} {exc}")
    elif pane.get("policy_rng_schedule") is not None:
        _fail(f"deterministic pane {pane_id} cannot carry a policy RNG schedule")
    metrics = pane.get("metrics")
    if not isinstance(metrics, list) or not metrics or len(metrics) != len(set(metrics)):
        _fail(f"pane {pane_id} has an invalid metric request")
    if any(metric not in MANDATORY_METRICS for metric in metrics):
        _fail(f"pane {pane_id} requests an unknown metric")
    observation_core_width = pane.get("observation_core_width")
    if (
        type(observation_core_width) is not int
        or observation_core_width <= 0
        or observation_core_width > 1_000_000
    ):
        _fail(f"pane {pane_id} has no valid frozen live observation core width")
    service_schema = _service_schema(pane, base)
    if "service_outcomes" in metrics:
        from .gate1_diagnostics import service_outcome_registry

        if service_schema["fields"] != service_outcome_registry():
            _fail(f"pane {pane_id} does not bind the complete Gate 1 service registry")
    execution = _execution_declaration(pane.get("execution"), f"pane {pane_id}")
    background = pane.get("background_policy")
    if not isinstance(background, dict) or background.get("digest") != digest(
        {k: v for k, v in background.items() if k != "digest"}
    ):
        _fail(f"pane {pane_id} background policy is not frozen")
    if {k: v for k, v in background.items() if k != "digest"} != CURRENT_BACKGROUND_POLICY:
        _fail(f"pane {pane_id} background policy is unsupported by the all-village evaluator")
    personality = pane.get("personality")
    if (
        not isinstance(personality, list)
        or len(personality) != contracts.PERSONALITY_WIDTH
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in personality)
    ):
        _fail(f"pane {pane_id} personality must be one supported frozen vector")
    worker, _ = _read_json_ref(pane.get("evaluation_worker_contract"), base, kind="worker_contract")
    if worker.get("contract", {}).get("digest") != digest(
        {k: v for k, v in worker.get("contract", {}).items() if k != "digest"}
    ):
        _fail(f"pane {pane_id} worker contract is unsealed")
    _implementation_artifact(
        pane.get("evaluation_implementation"),
        base,
        "checkpoint_inference_implementation",
        f"pane {pane_id}",
    )
    overrides = pane.get("environment_overrides")
    if not isinstance(overrides, dict) or set(overrides) != {
        "ANTHROPOCENE_REGION_HEAD",
        "ANTHROPOCENE_SIM_RULES",
        "ANTHROPOCENE_SIM_STATICS",
    }:
        _fail(f"pane {pane_id} must explicitly freeze every environment override")
    if pane["kind"] in {"historical-semantics", "fresh-common-producer-diagnostic"}:
        if pane.get("standardized_overrides") not in ({}, None):
            _fail(f"non-intervention pane {pane_id} cannot standardize flags")
        if pane["kind"] == "fresh-common-producer-diagnostic" and (
            pane.get("semantic_comparator_protocol") is not None
            or pane.get("semantic_comparator") is not None
        ):
            _fail(
                f"fresh common-producer pane {pane_id} cannot claim historical producer equivalence"
            )
    else:
        standardized = pane.get("standardized_overrides")
        if (
            not isinstance(standardized, dict)
            or not standardized
            or not pane.get("intervention_label")
        ):
            _fail(f"standardized pane {pane_id} needs explicit intervention overrides/label")
        if set(standardized) - INTERVENTION_FIELDS:
            _fail(
                f"standardized pane {pane_id} attempts an unsupported model-construction override"
            )
        for key, value in standardized.items():
            if (
                key
                in {
                    "commitments",
                    "learned_movement",
                    "region_movement",
                    "learned_fission",
                    "learned_trade",
                    "grouped_region",
                }
                and type(value) is not bool
            ):
                _fail(f"standardized pane {pane_id} has a non-boolean flag")
            if key == "region_head" and value not in {"flat", "hierarchical"}:
                _fail(f"standardized pane {pane_id} has an invalid region head")
            if key == "region_selection" and value not in {None, "argmax", "stay"}:
                _fail(f"standardized pane {pane_id} has an invalid region selection")
    return {
        "id": pane_id,
        "kind": pane["kind"],
        "nodes": selected,
        "worlds": worlds,
        "metrics": metrics,
        "worker_contract": worker["contract"],
        "service_outcome_schema_sha256": pane["service_outcome_schema"]["sha256"],
        "service_fields": service_schema["fields"],
        "execution": execution,
        "capture_bounds": capture_bounds,
        "observation_core_width": observation_core_width,
        "evaluation_implementation_sha256": pane["evaluation_implementation"]["sha256"],
    }


def validate_manifest(document: dict, base: Path | str = ".") -> dict:
    """Validate and re-hash the complete artifact DAG without constructing a model."""
    base = Path(base)
    scan(document, "initialization lineage manifest")
    if document.get("schema") != SCHEMA or document.get("kind") != KIND:
        _fail("unknown manifest schema/kind")
    if document.get("architecture_commit") != ARCHITECTURE_COMMIT:
        _fail("manifest does not bind the approved architecture commit")
    if document.get("digest") != digest({k: v for k, v in document.items() if k != "digest"}):
        _fail("manifest digest mismatch")
    required_nodes, required_edges = document.get("required_nodes"), document.get("required_edges")
    if (
        not isinstance(required_nodes, list)
        or not required_nodes
        or len(required_nodes) != len(set(required_nodes))
        or not isinstance(required_edges, list)
        or len(required_edges) != len(set(required_edges))
    ):
        _fail("required node/edge registries are missing or duplicated")
    lineage_inventory, lineage_inventory_sha = _lineage_inventory(document, base)
    node_rows: dict[str, dict] = {}
    states: dict[str, dict] = {}
    paths: dict[str, Path] = {}
    holds: list[str] = []
    for node in document.get("nodes", []):
        row, state, path, node_holds = _validate_node(node, base)
        if row["id"] in node_rows:
            _fail(f"duplicate node id {row['id']}")
        node_rows[row["id"]] = row
        if row["id"] in required_nodes:
            holds.extend(node_holds)
        if state is not None:
            states[row["id"]] = state
        if path is not None:
            paths[row["id"]] = path
    if not set(required_nodes) <= set(node_rows):
        _fail("manifest omits a required node record")
    hashes: dict[str, list[dict]] = defaultdict(list)
    source_nodes = {node["id"]: node for node in document["nodes"]}
    for node_id, row in node_rows.items():
        if row["status"] == "present":
            hashes[row["sha256"]].append(source_nodes[node_id])
    # Artifact-backed imports and packet replay must prove the same raw loader
    # eligibility as live evaluation.  A report is never allowed to stand in
    # for missing actor/normalizer tensors or an unexecutable adapter.
    for node_id, row in node_rows.items():
        if row["status"] == "present":
            _diagnostic_loading(source_nodes[node_id], base)
    for sha, copies in hashes.items():
        signatures = {
            digest(
                {
                    k: n.get(k)
                    for k in (
                        "raw_schema",
                        "contracts",
                        "producing_learner",
                        "historical_status",
                        "historical_semantics",
                        "loading",
                    )
                }
            )
            for n in copies
        }
        if len(signatures) > 1:
            _fail(f"checkpoint hash {sha} has conflicting semantic declarations")
    edges_by_id: dict[str, dict] = {}
    present_edges = []
    equivalence = []
    for edge in document.get("edges", []):
        edge_id = edge.get("id")
        if not isinstance(edge_id, str) or not edge_id or edge_id in edges_by_id:
            _fail("edge ids must be unique and nonempty")
        status = _status(edge, EDGE_STATUSES, f"edge {edge_id}")
        relation = edge.get("relation")
        if relation not in RELATIONS:
            _fail(f"edge {edge_id} has unknown relation")
        source, destination = edge.get("source"), edge.get("destination")
        if source not in node_rows or destination not in node_rows or source == destination:
            _fail(f"edge {edge_id} has invalid endpoints")
        row = {
            "id": edge_id,
            "status": status,
            "relation": relation,
            "source": source,
            "destination": destination,
        }
        if status != "present":
            row["reason"] = edge["reason"]
        edges_by_id[edge_id] = row
        if status != "present":
            if edge_id in required_edges:
                holds.append(f"edge:{edge_id}:{status}")
            continue
        if (
            node_rows[source]["status"] != "present"
            or node_rows[destination]["status"] != "present"
        ):
            _fail(f"edge {edge_id} cannot connect missing/not-applicable nodes")
        artifact(edge.get("evidence"), base)
        present_edges.append(row)
        if edge["relation"] == "schema_conversion":
            migration = _validate_migration(edge, node_rows[source], node_rows[destination], base)
            row["migration_status"] = migration["status"]
        if "equivalence" in edge:
            result = _validate_equivalence(
                edge, node_rows[source], node_rows[destination], states, base
            )
            result["edge"] = edge_id
            equivalence.append(result)
            if result["status"] != "PASS":
                holds.append(f"equivalence:{edge_id}:FAIL")
    if not set(required_edges) <= set(edges_by_id):
        _fail("manifest omits a required edge record")
    order = _topological(set(node_rows), present_edges)
    panes = [
        _validate_pane(pane, node_rows, base)
        for pane in document.get("evaluation", {}).get("panes", [])
    ]
    if (
        not panes
        or len({pane["id"] for pane in panes}) != len(panes)
        or not any(
            pane["kind"] in {"historical-semantics", "fresh-common-producer-diagnostic"}
            for pane in panes
        )
    ):
        _fail(
            "evaluation surface needs unique panes and a historical or approved fresh "
            "common-producer pane"
        )
    comparison_order = document.get("comparison_order")
    if (
        not isinstance(comparison_order, list)
        or not comparison_order
        or len(comparison_order) != len(set(comparison_order))
        or any(edge not in edges_by_id for edge in comparison_order)
    ):
        _fail("comparison_order must be a nonempty ordered set of registered edges")
    if not set(required_edges) <= set(comparison_order):
        _fail("comparison_order omits a required edge")
    evaluation_claim_protocol, evaluation_claim_protocol_sha = _evaluation_claim_protocol(
        document, base
    )
    verification_protocol, verification_protocol_sha = _verification_protocol(document, base)
    artifact_hashes = sorted({ref["sha256"] for ref in _artifact_refs(document)})
    return {
        "schema": 1,
        "kind": "initialization_lineage_validation",
        "manifest_digest": document["digest"],
        "nodes": [node_rows[k] for k in sorted(node_rows)],
        "edges": [edges_by_id[k] for k in sorted(edges_by_id)],
        "topological_order": order,
        "equivalence": sorted(equivalence, key=lambda x: x["edge"]),
        "panes": panes,
        "artifact_hashes": artifact_hashes,
        "holds": sorted(holds),
        "lineage_inventory": lineage_inventory,
        "lineage_inventory_sha256": lineage_inventory_sha,
        "evaluation_claim_protocol_sha256": evaluation_claim_protocol_sha,
        "verification_protocol_sha256": verification_protocol_sha,
        "strict_no_upgrade": True,
        "digest": digest(
            {
                "manifest": document["digest"],
                "artifacts": artifact_hashes,
                "nodes": [node_rows[k] for k in sorted(node_rows)],
                "edges": [edges_by_id[k] for k in sorted(edges_by_id)],
                "equivalence": sorted(equivalence, key=lambda x: x["edge"]),
                "evaluation_claim_protocol": evaluation_claim_protocol,
                "verification_protocol": verification_protocol,
            }
        ),
    }


def _artifact_refs(value: Any):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and _is_sha(value.get("sha256")):
            yield value
        else:
            for item in value.values():
                yield from _artifact_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _artifact_refs(item)


def _transitive_artifacts(roots: list[Any], base: Path) -> dict[str, Path]:
    queue = [ref for root in roots for ref in _artifact_refs(root)]
    seen: set[tuple[str, str]] = set()
    artifacts: dict[str, Path] = {}
    while queue:
        ref = queue.pop()
        try:
            path, sha = artifact(ref, base)
        except ProvenanceError:
            if _STRICT_PACKET_CAS.get() is not None:
                # Reconstruction must never consult the live repository when
                # preserved identity-bound source bytes are absent or corrupt.
                raise
            if ref.get("path") not in IMPLEMENTATION_SOURCES:
                raise
            try:
                path, sha = _protocol_artifact(ref, REPOSITORY)
            except ProvenanceError:
                # A forged/mismatched declared source is itself a packet HOLD;
                # it cannot be preserved as an artifact under bytes that do not exist.
                continue
        key = (str(path.resolve()), sha)
        if key in seen:
            continue
        seen.add(key)
        prior = artifacts.get(sha)
        if prior is not None and checkpoint_hash(prior) != checkpoint_hash(path):
            _fail(f"artifact SHA collision between {prior} and {path}")
        artifacts[sha] = path
        try:
            body = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        queue.extend(_artifact_refs(body))
    return artifacts


def _transitive_artifact_hashes(roots: list[Any], base: Path) -> list[str]:
    return sorted(_transitive_artifacts(roots, base))


def load_manifest(path: Path | str) -> tuple[dict, dict, Path]:
    path = Path(path).resolve()
    before = checkpoint_hash(path)
    document = json.loads(path.read_text())
    validation = validate_manifest(document, path.parent)
    if checkpoint_hash(path) != before:
        _fail("manifest changed during validation")
    return document, validation, path


def _semantic_comparator(
    protocol_ref: dict | None,
    receipt_ref: dict | None,
    base: Path,
    source: dict,
    live: dict,
    source_implementation: dict,
    live_implementation: dict,
) -> None:
    if source == live:
        return
    dimensions = ("reset", "configuration", "action", "observation", "reward", "state_trajectory")
    protocol, _ = _read_json_ref(protocol_ref, base, kind="producer_comparator_protocol")
    if set(protocol) != {
        "schema",
        "kind",
        "dimensions",
        "implementation",
        "corpus",
        "review",
        "digest",
    } or protocol.get("dimensions") != list(dimensions):
        _fail("producer comparator protocol does not freeze the complete semantic surface")
    _reviewed_protocol(
        protocol,
        base,
        "producer_comparator_protocol",
        {"reset", "configuration", "action", "observation", "reward", "state_trajectory"},
    )
    _implementation_artifact(
        protocol.get("implementation"),
        base,
        "producer_comparator_implementation",
        "producer comparator",
    )
    corpus, _ = _read_json_ref(protocol.get("corpus"), base, kind="producer_comparator_corpus")
    if (
        set(corpus) != {"schema", "kind", "cases", "digest"}
        or not isinstance(corpus.get("cases"), list)
        or not corpus["cases"]
    ):
        _fail("producer comparator corpus is malformed or empty")
    corpus_inputs = {}
    for case in corpus["cases"]:
        if (
            not isinstance(case, dict)
            or set(case) != {"id", "inputs"}
            or not isinstance(case.get("id"), str)
            or not case["id"]
            or case["id"] in corpus_inputs
            or not isinstance(case.get("inputs"), dict)
            or set(case["inputs"]) != set(dimensions)
        ):
            _fail("producer comparator corpus case is malformed")
        for ref in case["inputs"].values():
            artifact(ref, base)
        corpus_inputs[case["id"]] = case["inputs"]
    receipt, _ = _read_json_ref(receipt_ref, base, kind="producer_comparator_receipt")
    if (
        set(receipt)
        != {
            "schema",
            "kind",
            "protocol_sha256",
            "source_worker_digest",
            "live_worker_digest",
            "executions",
            "digest",
        }
        or receipt.get("protocol_sha256") != protocol_ref["sha256"]
        or receipt.get("source_worker_digest") != source.get("digest")
        or receipt.get("live_worker_digest") != live.get("digest")
        or not isinstance(receipt.get("executions"), list)
    ):
        _fail("producer comparator receipt is malformed or bound to another producer/protocol")
    executions = {}
    for ref in receipt["executions"]:
        row, _ = _read_json_ref(ref, base, kind="producer_comparator_execution")
        required = {
            "schema",
            "kind",
            "protocol_sha256",
            "case_id",
            "dimension",
            "side",
            "worker_digest",
            "producer_implementation_sha256",
            "comparator_implementation_sha256",
            "input",
            "output",
            "digest",
        }
        key = (row.get("case_id"), row.get("dimension"), row.get("side"))
        expected_worker = source if row.get("side") == "source" else live
        expected_implementation = (
            source_implementation if row.get("side") == "source" else live_implementation
        )
        if (
            set(row) != required
            or key in executions
            or row.get("protocol_sha256") != protocol_ref["sha256"]
            or row.get("case_id") not in corpus_inputs
            or row.get("dimension") not in dimensions
            or row.get("side") not in {"source", "live"}
            or row.get("worker_digest") != expected_worker.get("digest")
            or row.get("producer_implementation_sha256") != expected_implementation.get("sha256")
            or row.get("comparator_implementation_sha256") != protocol["implementation"]["sha256"]
            or row.get("input", {}).get("sha256")
            != corpus_inputs[row["case_id"]][row["dimension"]].get("sha256")
        ):
            _fail("producer comparator execution receipt is unbound, duplicated or malformed")
        artifact(row["input"], base)
        artifact(row["output"], base)
        executions[key] = row
    expected_keys = {
        (case_id, dimension, side)
        for case_id in corpus_inputs
        for dimension in dimensions
        for side in ("source", "live")
    }
    if set(executions) != expected_keys:
        _fail("producer comparator receipt omits a case/dimension/side execution")
    for case_id in corpus_inputs:
        for dimension in dimensions:
            source_row = executions[(case_id, dimension, "source")]
            live_row = executions[(case_id, dimension, "live")]
            if source_row["output"]["sha256"] != live_row["output"]["sha256"]:
                _fail(f"producer comparator found unequal {dimension} output for case {case_id}")


def _producer_identity_for_worker(worker: dict) -> dict:
    feature_flags = worker.get("feature_flags")
    raw_materials = (
        feature_flags.get("SimulationConfig.ElderRawMaterialsEnabled")
        if isinstance(feature_flags, dict)
        else None
    )
    if (
        not isinstance(worker.get("simulator"), dict)
        or not worker["simulator"]
        or not _is_sha(worker.get("bridge_sha256"))
        or not isinstance(worker.get("flags"), dict)
        or type(raw_materials) is not bool
    ):
        _fail("evaluation worker contract cannot bind the reported producer identity")
    return {
        "simulator": worker["simulator"],
        "bridge_sha256": worker["bridge_sha256"],
        "flags": worker["flags"],
        "raw_materials": raw_materials,
    }


def _pane(document: dict, pane_id: str) -> dict:
    panes = [pane for pane in document["evaluation"]["panes"] if pane["id"] == pane_id]
    if len(panes) != 1:
        _fail(f"unknown pane {pane_id}")
    return panes[0]


def _node(document: dict, node_id: str) -> dict:
    nodes = [node for node in document["nodes"] if node["id"] == node_id]
    if len(nodes) != 1:
        _fail(f"unknown node {node_id}")
    return nodes[0]


def _world_id(world: dict) -> str:
    return digest(_frozen_world_record(world))


def _frozen_world_record(world: dict) -> dict:
    snapshot = world.get("snapshot")
    return {
        "family": world["family"],
        "seed": world["seed"],
        "withheld": world["withheld"],
        "warmup": world["warmup"],
        "horizon": world["horizon"],
        "reset_turn": world["reset_turn"],
        "snapshot_sha256": "" if snapshot is None else snapshot["sha256"],
        "turn": world.get("turn"),
        "players": world.get("players", 8),
        "cohort_digest": world.get("cohort_digest", ""),
    }


def _expected_world_descriptor(world_spec: dict, semantics: dict) -> dict:
    from .worker_contract import world

    frozen = _frozen_world_record(world_spec)
    return world(
        frozen["seed"],
        frozen["withheld"],
        snapshot_sha256=frozen["snapshot_sha256"],
        turn=frozen["turn"],
        players=frozen["players"],
        movement=semantics["learned_movement"],
        head=semantics["region_head"],
        cohort_digest=frozen["cohort_digest"],
    )


def _evaluation_context(node: dict, pane: dict, base: Path) -> tuple[dict, Path, dict, dict]:
    """Bind a checkpoint's historical producer to the requested live worker."""
    _, state, path = inspect_checkpoint(node["artifact"], base)
    from .worker_contract import from_state

    historical_worker = from_state(state, required=False)
    live_wrapper, _ = _read_json_ref(
        pane["evaluation_worker_contract"], base, kind="worker_contract"
    )
    live_worker = live_wrapper["contract"]
    producer = _producer_identity_for_worker(live_worker)
    if pane["kind"] == "fresh-common-producer-diagnostic":
        _fresh_common_producer_scope(node, pane)
        return state, path, live_worker, producer
    if historical_worker is None:
        _fail("historical/standardized evaluation requires an embedded historical worker contract")
    _semantic_comparator(
        pane.get("semantic_comparator_protocol"),
        pane.get("semantic_comparator"),
        base,
        historical_worker,
        live_worker,
        node["inference_implementation"],
        pane["evaluation_implementation"],
    )
    return state, path, live_worker, producer


def _fresh_common_producer_scope(node: dict, pane: dict) -> None:
    if (
        node.get("historical_producer_status") != "unresolved"
        or pane.get("semantic_comparator_protocol") is not None
        or pane.get("semantic_comparator") is not None
    ):
        _fail(
            "fresh common-producer evaluation must preserve unresolved historical producer identity"
        )


def _diagnostic_loading(node: dict, base: Path) -> dict | None:
    """Validate diagnostic loader eligibility without advancing global RNGs."""
    from .gate1_diagnostics import preserve_global_rng

    with preserve_global_rng():
        return _diagnostic_loading_preserved(node, base)


def _diagnostic_loading_preserved(node: dict, base: Path) -> dict | None:
    """Implementation of loader eligibility, called only under RNG preservation."""
    _, checkpoint_state, _ = inspect_checkpoint(node["artifact"], base)
    if node["loading"]["mode"] == "native-strict":
        if node["raw_schema"].get("checkpoint_contract") != contracts.NATIVE_CONTRACT:
            _fail("native-strict loading cannot execute a legacy checkpoint")
        from .evaluate import load_strict_state

        try:
            load_strict_state(
                checkpoint_state,
                resolved_semantics=node["historical_semantics"],
                diagnostic_loading=None,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            _fail(f"native checkpoint is not executable under its strict declared loader: {exc}")
        return None
    if (
        node["loading"].get("mode") == "reviewed-adapter"
        and node["loading"].get("adapter") == "t3-initialization-bc-composite"
    ):
        protocol, _ = _read_json_ref(
            node["loading"].get("implementation"),
            base,
            kind="gate1_bc_phase_adapter_protocol",
        )
        if (
            protocol.get("composite_scope") != "current-diagnostic-only"
            or protocol.get("historical_bc_normalizer_claim") is not False
            or node["loading"].get("protocol") != node["loading"].get("implementation")
        ):
            _fail("BC phase adapter overclaims historical normalizer provenance")
        diagnostic_loading = {
            "mode": "reviewed-adapter",
            "adapter": "t3-initialization-bc-composite",
            "protocol": node["loading"]["protocol"],
            "support_checkpoint": node["loading"].get("support_checkpoint"),
        }
        from .gate1_production_adapter import load_bc_phase

        try:
            assembled = load_bc_phase(
                Path(node["artifact"]["path"])
                if Path(node["artifact"]["path"]).is_absolute()
                else base / node["artifact"]["path"],
                diagnostic_loading,
                base=base,
            )
            from .evaluate import load_strict_state

            load_strict_state(
                assembled, resolved_semantics=node["historical_semantics"], diagnostic_loading=None
            )
        except (KeyError, TypeError, ValueError, RuntimeError, ProvenanceError) as exc:
            _fail(f"BC phase checkpoint is not executable under its reviewed composite: {exc}")
        return diagnostic_loading
    if (
        node["loading"].get("mode") == "reviewed-adapter"
        and node["loading"].get("adapter") == "t2m-guarded-bc-actor-only"
    ):
        protocol, _ = _read_json_ref(
            node["loading"].get("implementation"),
            base,
            kind="gate1_guarded_bc_adapter_protocol",
        )
        if (
            protocol.get("checkpoint_sha256") != node["artifact"].get("sha256")
            or protocol.get("critic_status") != "absent-in-actor-only-bc-phase"
            or node["loading"].get("protocol") != node["loading"].get("implementation")
        ):
            _fail("guarded BC adapter overclaims critic/completion provenance")
        diagnostic_loading = {
            "mode": "reviewed-adapter",
            "adapter": "t2m-guarded-bc-actor-only",
            "protocol": node["loading"]["protocol"],
        }
        from .gate1_production_adapter import load_guarded_bc

        try:
            load_guarded_bc(
                Path(node["artifact"]["path"])
                if Path(node["artifact"]["path"]).is_absolute()
                else base / node["artifact"]["path"],
                diagnostic_loading,
                base=base,
            )
        except (KeyError, TypeError, ValueError, RuntimeError, ProvenanceError) as exc:
            _fail(f"guarded BC checkpoint is not executable under its reviewed adapter: {exc}")
        return diagnostic_loading
    if (
        node["loading"]["mode"] != "reviewed-adapter"
        or node["raw_schema"].get("checkpoint_contract") != 8
    ):
        _fail("legacy checkpoint lacks a supported reviewed adapter")
    protocol, protocol_sha = _read_json_ref(
        node["loading"].get("implementation"),
        base,
        kind="gate1_legacy_adapter_protocol",
    )
    required = {
        "schema",
        "kind",
        "source_contract",
        "historical_source_commit",
        "source",
        "input_mapping",
        "decoder_defaults",
        "default_resolution",
        "review",
        "digest",
    }
    expected_mapping = {
        "core": "raw-checkpoint-core-width-prefix",
        "sets": ["sites", "fishing", "herds", "commodities", "contacts"],
        "appended_sets": "excluded-from-contract-8-actor-input",
    }
    expected_decoder = {
        "commit": 1,
        "movement": contracts.NATIVE_MOVEMENT,
        "fission": -1,
        "trade_posture": "native",
    }
    if (
        set(protocol) != required
        or protocol.get("source_contract") != 8
        or protocol.get("historical_source_commit") != "2f4cc895e6bcd897200d340851d69be98fd18e00"
        or protocol.get("input_mapping") != expected_mapping
        or protocol.get("decoder_defaults") != expected_decoder
        or protocol.get("default_resolution") != node["loading"]["default_resolution"]
    ):
        _fail("contract-8 adapter protocol is noncanonical or not bound to its defaults")
    _reviewed_protocol(
        protocol,
        base,
        "gate1_legacy_adapter_protocol",
        {"historical-source", "input-mapping", "decoder", "default-resolution"},
    )
    source_path, source_sha = artifact(protocol.get("source"), base)
    expected_relative = Path("src/anthropocene_rl/autonomous_v7/legacy_contract8.py")
    expected_source = (
        REPOSITORY / "src/anthropocene_rl/autonomous_v7/legacy_contract8.py"
    ).resolve()
    declared_source = Path(protocol["source"].get("path", ""))
    if declared_source.is_absolute():
        try:
            declared_relative = declared_source.resolve().relative_to(REPOSITORY.resolve())
        except ValueError:
            _fail("contract-8 adapter source identity is outside the repository")
    else:
        declared_relative = declared_source
    if (
        declared_relative.as_posix() != expected_relative.as_posix()
        or checkpoint_hash(source_path) != source_sha
    ):
        _fail("contract-8 adapter protocol does not bind the executing adapter source")
    if _STRICT_PACKET_CAS.get() is None and checkpoint_hash(expected_source) != source_sha:
        _fail("contract-8 adapter protocol differs from the live executing adapter source")
    semantics = node["historical_semantics"]
    expected_semantics = {
        "experts": False,
        "token": False,
        "commitments": False,
        "learned_movement": False,
        "region_movement": False,
        "learned_fission": False,
        "learned_trade": False,
        "region_head": "flat",
        "grouped_region": False,
        "region_selection": None,
    }
    if semantics != expected_semantics:
        _fail("contract-8 adapter cannot synthesize post-contract-8 heads or decoder semantics")
    diagnostic_loading = {
        "mode": "reviewed-adapter",
        "adapter": "pinned-contract-8",
        "protocol_sha256": protocol_sha,
    }
    from .evaluate import load_strict_state

    try:
        load_strict_state(
            checkpoint_state,
            resolved_semantics=node["historical_semantics"],
            diagnostic_loading=diagnostic_loading,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        _fail(f"legacy checkpoint is not executable under its reviewed adapter: {exc}")
    return diagnostic_loading


def evaluate_node(
    document: dict,
    base: Path | str,
    node_id: str,
    pane_id: str,
    runner: Callable[..., dict] | None = None,
) -> dict:
    """Execute the exact reviewed Gate-1 pane; unsupported contracts fail closed."""
    base = Path(base)
    validate_manifest(document, base)
    node, pane = _node(document, node_id), _pane(document, pane_id)
    if node["status"] != "present" or node_id not in pane["nodes"]:
        _fail("evaluation node is unavailable or outside the pane")
    diagnostic_loading = _diagnostic_loading(node, base)
    if pane["kind"] == "standardized-flags-intervention":
        unknown = set(pane["standardized_overrides"]) - INTERVENTION_FIELDS
        if unknown:
            _fail(
                "standardized pane requests unsupported actor overrides: "
                + ", ".join(sorted(unknown))
            )
    _, path, live, expected_producer = _evaluation_context(node, pane, base)
    for key, value in pane["environment_overrides"].items():
        if os.environ.get(key, "") != value:
            _fail(f"live environment override {key} differs from frozen pane")
    historical_semantics = dict(node["historical_semantics"])
    semantics = dict(historical_semantics)
    if pane["kind"] == "standardized-flags-intervention":
        semantics.update(pane["standardized_overrides"])
    runner = (
        runner or __import__("anthropocene_rl.autonomous_v7.evaluate", fromlist=["rollout"]).rollout
    )
    service_schema = _service_schema(pane, base)
    rng_schedule = None
    if pane["action_mode"] == "seeded-stochastic":
        rng_schedule, _ = _read_json_ref(
            pane["policy_rng_schedule"], base, kind="gate1_policy_rng_schedule"
        )
    from .gate1_diagnostics import (
        draw_order_from_semantics,
        validate_rng_schedule,
        validate_trajectory_evidence,
    )

    rows = []
    for frozen in pane["worlds"]:
        descriptor = _expected_world_descriptor(frozen, semantics)
        snapshot = None
        if frozen.get("snapshot") is not None:
            snapshot_path, snapshot_sha = artifact(frozen["snapshot"], base)
            snapshot = snapshot_path.read_bytes()
            if snapshot_sha != descriptor["snapshot_sha256"]:
                _fail("snapshot bytes differ from the exact frozen world descriptor")
        policy_rng_steps = None
        if rng_schedule is not None:
            try:
                policy_rng_steps = validate_rng_schedule(
                    rng_schedule,
                    _world_id(frozen),
                    frozen["horizon"],
                    node_id=node_id,
                    expected_draw_order=draw_order_from_semantics(semantics),
                )
            except ValueError as exc:
                _fail(str(exc))
        result = runner(
            frozen["seed"],
            frozen["horizon"],
            str(path),
            frozen["withheld"],
            semantics["learned_movement"],
            semantics["commitments"],
            semantics["region_movement"],
            frozen["warmup"],
            semantics["learned_fission"],
            semantics["learned_trade"],
            pane["personality"],
            semantics["grouped_region"],
            False,
            live,
            descriptor,
            strict_checkpoint=True,
            resolved_semantics=historical_semantics,
            evaluation_semantics=semantics,
            snapshot=snapshot,
            action_mode=pane["action_mode"],
            policy_rng_steps=policy_rng_steps,
            requested_metrics=pane["metrics"],
            service_fields=service_schema["fields"],
            diagnostic_loading=diagnostic_loading,
            capture_bounds=pane["capture_bounds"],
            evaluation_world_id=_world_id(frozen),
            expected_reset_turn=frozen["reset_turn"],
            observation_core_width=pane["observation_core_width"],
        )
        if (
            result.get("worker_contract") != live
            or result.get("world_contract") != descriptor
            or result.get("producer") != expected_producer
            or result.get("snapshot_sha256", "") != descriptor["snapshot_sha256"]
            or result.get("action_mode", "deterministic") != pane["action_mode"]
        ):
            _fail(
                "evaluation result producer/worker/world identity differs from the frozen request"
            )
        try:
            recomputed = validate_trajectory_evidence(
                result.get("trajectory_evidence"),
                world_id=_world_id(frozen),
                snapshot_sha256=descriptor["snapshot_sha256"],
                horizon=frozen["horizon"],
                bounds=pane["capture_bounds"],
                action_mode=pane["action_mode"],
                rng_steps=policy_rng_steps,
                service_fields=service_schema["fields"],
                expected_opening_turn=frozen["reset_turn"] + frozen["warmup"],
                expected_reset_turn=frozen["reset_turn"],
                observation_core_width=pane["observation_core_width"],
                policy_semantics=semantics,
                expected_decoder=(
                    "pinned-contract-8" if diagnostic_loading is not None else "native-contract-12"
                ),
            )
        except (TypeError, ValueError) as exc:
            _fail(f"live evaluation trajectory evidence is malformed: {exc}")
        values = {metric: recomputed[metric] for metric in pane["metrics"]}
        if (
            result.get("policy_rng_receipts", []) != recomputed["policy_rng_receipts"]
            or result.get("evaluation_steps") != recomputed["evaluation_steps"]
            or result.get("capture_receipt") != recomputed["capture_receipt"]
            or result.get("compiler_intervention_summary")
            != recomputed["compiler_intervention_summary"]
        ):
            _fail("live evaluation summaries differ from complete raw trajectory evidence")
        rows.append(
            {
                "world_id": _world_id(frozen),
                "world": _frozen_world_record(frozen),
                "metrics": values,
                "compiler_intervention_summary": result.get("compiler_intervention_summary"),
                "trajectory_evidence": result.get("trajectory_evidence"),
                "policy_rng_receipts": result.get("policy_rng_receipts", []),
                "evaluation_steps": result.get("evaluation_steps", frozen["horizon"]),
                "capture_receipt": result.get("capture_receipt"),
                "producer": result.get("producer"),
                "worker_contract": result["worker_contract"],
                "world_contract": result["world_contract"],
            }
        )
    payload = {
        "schema": 1,
        "kind": "initialization_evaluation",
        "evidence_class": "checkpoint-evaluation",
        "manifest_digest": document["digest"],
        "node": node_id,
        "pane": pane_id,
        "checkpoint_sha256": node["artifact"]["sha256"],
        "semantics": semantics,
        "metrics": pane["metrics"],
        "personality": pane["personality"],
        "background_policy": pane["background_policy"],
        "service_outcome_schema": pane["service_outcome_schema"],
        "inference_implementation": node["inference_implementation"],
        "evaluation_implementation": pane["evaluation_implementation"],
        "producer_scope": pane["kind"],
        "historical_producer_status": node.get(
            "historical_producer_status", "embedded-worker-contract"
        ),
        "execution": pane["execution"],
        "rows": rows,
    }
    payload["loading_protocol"] = node["loading"].get("implementation")
    payload["action_mode"] = pane["action_mode"]
    payload["policy_rng_schedule"] = pane.get("policy_rng_schedule")
    payload["capture_bounds"] = pane["capture_bounds"]
    payload["observation_core_width"] = pane["observation_core_width"]
    payload["digest"] = digest(payload)
    if checkpoint_hash(path) != node["artifact"]["sha256"]:
        _fail("checkpoint changed during evaluation")
    return payload


def _metric(metric: str, value: Any, service_schema: dict) -> bool:
    if metric == "trajectory_digest":
        return _is_sha(value)
    if metric == "service_outcomes":
        names = {row["name"] for row in service_schema["fields"]}
        if not isinstance(value, dict) or set(value) != names:
            return False
        for measurement in value.values():
            if not isinstance(measurement, dict) or set(measurement) != {
                "value",
                "availability_reason",
            }:
                return False
            observed, reason = measurement["value"], measurement["availability_reason"]
            if observed is None:
                if not isinstance(reason, str) or not reason.strip():
                    return False
            elif (
                type(observed) not in (int, float)
                or not math.isfinite(observed)
                or reason is not None
            ):
                return False
        return True
    if metric in COUNT_METRICS:
        return type(value) is int and value >= 0
    if type(value) not in (int, float) or not math.isfinite(value):
        return False
    return metric not in {"population_years", "qol_years"} or value >= 0


def _compiler_summary(value: Any, total: int) -> bool:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "kind", "total", "by_domain_reason", "raw_rows", "digest"}
        or value.get("schema") != 1
        or value.get("kind") != "gate1_compiler_intervention_summary"
        or value.get("total") != total
        or value.get("digest")
        != digest({key: item for key, item in value.items() if key != "digest"})
        or not isinstance(value.get("by_domain_reason"), list)
        or not isinstance(value.get("raw_rows"), list)
        or len(value["raw_rows"]) != total
    ):
        return False
    expected = defaultdict(int)
    previous = None
    for row in value["raw_rows"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"transitionId", "decisionTurn", "campId", "domain", "reason", "source"}
            or not all(
                isinstance(row.get(key), str) and row[key] for key in ("domain", "reason", "source")
            )
        ):
            return False
        ordering = (
            str(row["transitionId"]),
            int(row["decisionTurn"] or 0),
            int(row["campId"] or 0),
            row["domain"],
            row["reason"],
            row["source"],
        )
        if previous is not None and ordering < previous:
            return False
        previous = ordering
        expected[(row["domain"], row["reason"])] += 1
    return value["by_domain_reason"] == [
        {"domain": domain, "reason": reason, "count": count}
        for (domain, reason), count in sorted(expected.items())
    ]


def _evaluation_index(
    document: dict, evaluations: list[dict], base: Path
) -> dict[tuple[str, str, str], dict]:
    index = {}
    for report in evaluations:
        if (
            report.get("schema") != 1
            or report.get("kind") != "initialization_evaluation"
            or report.get("manifest_digest") != document["digest"]
            or report.get("digest") != digest({k: v for k, v in report.items() if k != "digest"})
        ):
            _fail("evaluation report is malformed or bound to another manifest")
        node, pane = _node(document, report.get("node")), _pane(document, report.get("pane"))
        diagnostic_loading = _diagnostic_loading(node, base)
        if (
            node["status"] != "present"
            or node["id"] not in pane["nodes"]
            or report.get("evidence_class") != "checkpoint-evaluation"
            or report.get("checkpoint_sha256") != node["artifact"]["sha256"]
            or report.get("metrics") != pane["metrics"]
            or report.get("personality") != pane["personality"]
            or report.get("background_policy") != pane["background_policy"]
            or report.get("service_outcome_schema") != pane["service_outcome_schema"]
            or report.get("inference_implementation") != node["inference_implementation"]
            or report.get("evaluation_implementation") != pane["evaluation_implementation"]
            or report.get("execution") != pane["execution"]
            or report.get("loading_protocol") != node["loading"].get("implementation")
            or report.get("action_mode") != pane["action_mode"]
            or report.get("policy_rng_schedule") != pane.get("policy_rng_schedule")
            or report.get("capture_bounds") != pane.get("capture_bounds")
        ):
            _fail(
                "evaluation report node/checkpoint/evidence/metric/personality/background "
                "scope differs from request"
            )
        if pane["kind"] == "fresh-common-producer-diagnostic" and (
            report.get("producer_scope") != "fresh-common-producer-diagnostic"
            or report.get("historical_producer_status") != "unresolved"
        ):
            _fail("fresh common-producer report overclaims historical producer provenance")
        if report.get("observation_core_width") != pane["observation_core_width"]:
            _fail("evaluation report observation layout differs from the reviewed pane")
        _execution_declaration(report["execution"], "evaluation report")
        semantics = dict(node["historical_semantics"])
        if pane["kind"] == "standardized-flags-intervention":
            semantics.update(pane["standardized_overrides"])
        if report.get("semantics") != semantics:
            _fail("evaluation report silently changed historical/standardized semantics")
        expected_worlds = {_world_id(world): world for world in pane["worlds"]}
        _, _, worker, expected_producer = _evaluation_context(node, pane, base)
        service_schema = _service_schema(pane, base)
        if not isinstance(report.get("rows"), list):
            _fail("evaluation report rows are not a list")
        for row in report["rows"]:
            world_id = row.get("world_id")
            frozen = expected_worlds.get(world_id)
            if frozen is None or row.get("world") != _frozen_world_record(frozen):
                _fail("evaluation report contains an unrequested or altered world")
            metrics = row.get("metrics")
            if (
                not isinstance(metrics, dict)
                or set(metrics) != set(pane["metrics"])
                or any(not _metric(name, value, service_schema) for name, value in metrics.items())
                or row.get("producer") != expected_producer
                or row.get("worker_contract") != worker
            ):
                _fail("evaluation row lacks finite scoped metrics or its exact bound producer")
            executed_steps = row.get("evaluation_steps")
            if (
                type(executed_steps) is not int
                or executed_steps < 0
                or executed_steps > frozen["horizon"]
            ):
                _fail("evaluation row has an invalid executed-step count")
            evidence = row.get("trajectory_evidence")
            schedule_steps = None
            if pane["action_mode"] == "seeded-stochastic":
                schedule, _ = _read_json_ref(
                    pane["policy_rng_schedule"], base, kind="gate1_policy_rng_schedule"
                )
                from .gate1_diagnostics import draw_order_from_semantics, validate_rng_schedule

                try:
                    schedule_steps = validate_rng_schedule(
                        schedule,
                        world_id,
                        frozen["horizon"],
                        node_id=node["id"],
                        expected_draw_order=draw_order_from_semantics(semantics),
                    )
                except ValueError as exc:
                    _fail(f"evaluation row policy RNG schedule is invalid: {exc}")
            from .gate1_diagnostics import validate_trajectory_evidence

            try:
                recomputed = validate_trajectory_evidence(
                    evidence,
                    world_id=world_id,
                    snapshot_sha256=_frozen_world_record(frozen)["snapshot_sha256"],
                    horizon=frozen["horizon"],
                    bounds=pane["capture_bounds"],
                    action_mode=pane["action_mode"],
                    rng_steps=schedule_steps,
                    service_fields=service_schema["fields"],
                    expected_opening_turn=frozen["reset_turn"] + frozen["warmup"],
                    expected_reset_turn=frozen["reset_turn"],
                    observation_core_width=pane["observation_core_width"],
                    policy_semantics=semantics,
                    expected_decoder=(
                        "pinned-contract-8"
                        if diagnostic_loading is not None
                        else "native-contract-12"
                    ),
                )
            except (TypeError, ValueError) as exc:
                _fail(f"evaluation row diagnostic surface is malformed: {exc}")
            if (
                any(metrics[name] != recomputed[name] for name in metrics)
                or row.get("compiler_intervention_summary")
                != recomputed["compiler_intervention_summary"]
                or row.get("policy_rng_receipts", []) != recomputed["policy_rng_receipts"]
                or executed_steps != recomputed["evaluation_steps"]
                or row.get("capture_receipt") != recomputed["capture_receipt"]
            ):
                _fail(
                    "evaluation row outcomes/receipts differ from complete raw trajectory evidence"
                )
            from .worker_contract import checked

            expected_descriptor = _expected_world_descriptor(frozen, semantics)
            if checked(row.get("world_contract")) != expected_descriptor:
                _fail("evaluation row world contract differs from the exact frozen request")
            key = (report["node"], report["pane"], world_id)
            if key in index:
                _fail("duplicate evaluation node/pane/world result")
            index[key] = {"report": report, "row": row}
        if {key[2] for key in index if key[:2] == (report["node"], report["pane"])} != set(
            expected_worlds
        ):
            _fail("evaluation report is not a complete requested world cohort")
    return index


def _completeness_audit(document: dict, index: dict, inventory: dict) -> list[dict]:
    """Audit the entire ordered surface before any first-change classification."""
    issues = []
    panes = {pane["id"]: pane for pane in document["evaluation"]["panes"]}
    required_nodes = set(inventory["required_nodes"])
    for node in inventory["nodes"]:
        if node["status"] != "present" or node["id"] not in required_nodes:
            continue
        applicable = [(pane_id, "historical") for pane_id in node["historical_panes"]]
        applicable += [(pane_id, "standardized") for pane_id in node["standardized_panes"]]
        for pane_id, surface in applicable:
            pane = panes[pane_id]
            for world in pane["worlds"]:
                world_id = _world_id(world)
                result = index.get((node["id"], pane_id, world_id))
                if result is None:
                    issues.append(
                        {
                            "node": node["id"],
                            "pane": pane_id,
                            "surface": surface,
                            "world_id": world_id,
                            "reason": "missing evaluation",
                        }
                    )
                    continue
                values = result["row"]["metrics"]
                missing = sorted(MANDATORY_METRICS - set(values))
                if missing:
                    issues.append(
                        {
                            "node": node["id"],
                            "pane": pane_id,
                            "surface": surface,
                            "world_id": world_id,
                            "reason": "missing mandatory metrics",
                            "metrics": missing,
                        }
                    )
                    continue
                unavailable = sorted(
                    name
                    for name, measurement in values["service_outcomes"].items()
                    if measurement["value"] is None
                )
                if unavailable:
                    issues.append(
                        {
                            "node": node["id"],
                            "pane": pane_id,
                            "surface": surface,
                            "world_id": world_id,
                            "reason": "required service outcomes unavailable",
                            "services": unavailable,
                        }
                    )
    required_edges = set(inventory["required_edges"])
    for edge in inventory["edges"]:
        if edge["status"] != "present" or edge["id"] not in required_edges:
            continue
        applicable = [(pane_id, "historical") for pane_id in edge["historical_panes"]]
        applicable += [(pane_id, "standardized") for pane_id in edge["standardized_panes"]]
        for pane_id, surface in applicable:
            pane = panes[pane_id]
            for world in pane["worlds"]:
                world_id = _world_id(world)
                endpoint_rows = {
                    endpoint: index.get((endpoint, pane_id, world_id))
                    for endpoint in (edge["source"], edge["destination"])
                }
                if any(value is None for value in endpoint_rows.values()):
                    issues.append(
                        {
                            "edge": edge["id"],
                            "pane": pane_id,
                            "surface": surface,
                            "world_id": world_id,
                            "reason": "missing paired evaluation",
                        }
                    )
                    continue
                for endpoint, result in endpoint_rows.items():
                    values = result["row"]["metrics"]
                    missing = sorted(MANDATORY_METRICS - set(values))
                    if missing:
                        issues.append(
                            {
                                "edge": edge["id"],
                                "endpoint": endpoint,
                                "pane": pane_id,
                                "surface": surface,
                                "world_id": world_id,
                                "reason": "missing mandatory metrics",
                                "metrics": missing,
                            }
                        )
                        continue
                    unavailable = sorted(
                        name
                        for name, measurement in values["service_outcomes"].items()
                        if measurement["value"] is None
                    )
                    if unavailable:
                        issues.append(
                            {
                                "edge": edge["id"],
                                "endpoint": endpoint,
                                "pane": pane_id,
                                "surface": surface,
                                "world_id": world_id,
                                "reason": "required service outcomes unavailable",
                                "services": unavailable,
                            }
                        )
    return issues


def _ratio(destination: int | float | None, source: int | float | None) -> dict:
    if destination is None or source is None:
        return {"status": "unavailable", "value": None}
    if source == 0:
        return {"status": "zero-denominator", "value": None}
    return {"status": "defined", "value": destination / source}


def _paired_interval(values: list[float], identity: dict) -> dict:
    if len(values) < 2:
        return {"status": "unavailable", "lower": None, "upper": None}
    seed = int(digest({"identity": identity, "values": values})[:16], 16)
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(1000)
    )
    return {"status": "defined", "lower": means[24], "upper": means[974]}


def _paired_aggregation(document: dict, index: dict, inventory: dict | None = None) -> dict:
    pairs = []
    grouped: dict[tuple[str, str, str], dict] = {}
    edges = {edge["id"]: edge for edge in document["edges"]}
    prescribed = (
        {edge["id"]: edge for edge in inventory["edges"]} if inventory is not None else None
    )
    for edge_id in document["comparison_order"]:
        edge = edges[edge_id]
        if edge["status"] != "present":
            continue
        source_node, destination_node = (
            _node(document, edge["source"]),
            _node(document, edge["destination"]),
        )
        for pane in document["evaluation"]["panes"]:
            if prescribed is not None:
                field = (
                    "historical_panes"
                    if pane["kind"] in {"historical-semantics", "fresh-common-producer-diagnostic"}
                    else "standardized_panes"
                )
                if pane["id"] not in prescribed[edge_id][field]:
                    continue
            if edge["source"] not in pane["nodes"] or edge["destination"] not in pane["nodes"]:
                continue
            service_schema = None
            for world in pane["worlds"]:
                world_id = _world_id(world)
                left = index.get((edge["source"], pane["id"], world_id))
                right = index.get((edge["destination"], pane["id"], world_id))
                if left is None or right is None:
                    continue
                source_values, destination_values = left["row"]["metrics"], right["row"]["metrics"]
                derived = {}
                for metric in sorted(SCALAR_METRICS & set(source_values) & set(destination_values)):
                    source_value, destination_value = (
                        source_values[metric],
                        destination_values[metric],
                    )
                    derived[metric] = {
                        "source": source_value,
                        "destination": destination_value,
                        "difference": destination_value - source_value,
                        "ratio": _ratio(destination_value, source_value),
                    }
                if "service_outcomes" in source_values and "service_outcomes" in destination_values:
                    service_schema = source_values["service_outcomes"]
                    for name in sorted(service_schema):
                        source_value = source_values["service_outcomes"][name]["value"]
                        destination_value = destination_values["service_outcomes"][name]["value"]
                        key = f"service_outcomes.{name}"
                        derived[key] = {
                            "source": source_value,
                            "destination": destination_value,
                            "difference": (
                                None
                                if source_value is None or destination_value is None
                                else destination_value - source_value
                            ),
                            "ratio": _ratio(destination_value, source_value),
                        }
                pair = {
                    "edge": edge_id,
                    "pane": pane["id"],
                    "source_node": edge["source"],
                    "destination_node": edge["destination"],
                    "world_id": world_id,
                    "world": _frozen_world_record(world),
                    "source_metrics": source_values,
                    "destination_metrics": destination_values,
                    "derived": derived,
                }
                pairs.append(pair)
                for metric, values in derived.items():
                    key = (edge_id, pane["id"], metric)
                    group = grouped.setdefault(
                        key,
                        {
                            "edge": edge_id,
                            "pane": pane["id"],
                            "source_node": edge["source"],
                            "destination_node": edge["destination"],
                            "metric": metric,
                            "source_learner": source_node["producing_learner"],
                            "destination_learner": destination_node["producing_learner"],
                            "source_run_identity": source_node["run_identity"],
                            "destination_run_identity": destination_node["run_identity"],
                            "differences": [],
                            "defined_ratios": [],
                            "zero_denominators": 0,
                            "unavailable": 0,
                        },
                    )
                    if values["difference"] is None:
                        group["unavailable"] += 1
                    else:
                        group["differences"].append(values["difference"])
                    if values["ratio"]["status"] == "defined":
                        group["defined_ratios"].append(values["ratio"]["value"])
                    elif values["ratio"]["status"] == "zero-denominator":
                        group["zero_denominators"] += 1
    summaries = []
    for key in sorted(grouped):
        group = grouped[key]
        differences = group.pop("differences")
        ratios = group.pop("defined_ratios")
        identity = {
            name: group[name]
            for name in ("edge", "pane", "source_node", "destination_node", "metric")
        }
        summaries.append(
            {
                **group,
                "paired_world_count": len(differences) + group["unavailable"],
                "difference": {
                    "mean": (None if not differences else sum(differences) / len(differences)),
                    "interval": _paired_interval(
                        differences, {**identity, "statistic": "difference"}
                    ),
                },
                "ratio": {
                    "mean": (None if not ratios else sum(ratios) / len(ratios)),
                    "interval": _paired_interval(ratios, {**identity, "statistic": "ratio"}),
                    "defined_count": len(ratios),
                },
            }
        )
    return {
        "resampling": {
            "unit": "paired-world",
            "method": "deterministic-bootstrap",
            "replicates": 1000,
            "optimizer_identities_pooled": False,
            "uncertainty_scope": "world-only; optimizer uncertainty is not estimated",
        },
        "pairs": pairs,
        "summaries": summaries,
    }


def _classify(document: dict, index: dict, metric_kind: str) -> dict:
    field = "trajectory_digest" if metric_kind == "behavior" else None
    present_comparisons = 0
    for edge_id in document["comparison_order"]:
        edge = next(e for e in document["edges"] if e["id"] == edge_id)
        if edge["status"] != "present":
            if edge_id in document["required_edges"]:
                return {
                    "classification": "insufficient/incompatible",
                    "boundary": edge_id,
                    "reason": f"required edge is {edge['status']}",
                }
            continue
        present_comparisons += 1
        comparable = False
        changed = False
        for pane in document["evaluation"]["panes"]:
            if edge["source"] not in pane["nodes"] or edge["destination"] not in pane["nodes"]:
                continue
            for world in pane["worlds"]:
                world_id = _world_id(world)
                left = index.get((edge["source"], pane["id"], world_id))
                right = index.get((edge["destination"], pane["id"], world_id))
                if left is None or right is None:
                    return {
                        "classification": "insufficient/incompatible",
                        "boundary": edge_id,
                        "reason": "missing paired evaluation",
                    }
                comparable = True
                if metric_kind == "behavior":
                    if field not in left["row"]["metrics"] or field not in right["row"]["metrics"]:
                        return {
                            "classification": "insufficient/incompatible",
                            "boundary": edge_id,
                            "reason": "trajectory digest unavailable",
                        }
                    changed |= left["row"]["metrics"][field] != right["row"]["metrics"][field]
                else:
                    metrics = MANDATORY_METRICS - {"trajectory_digest"}
                    if not metrics <= set(left["row"]["metrics"]) or not metrics <= set(
                        right["row"]["metrics"]
                    ):
                        return {
                            "classification": "insufficient/incompatible",
                            "boundary": edge_id,
                            "reason": "mandatory outcomes unavailable",
                        }
                    changed |= any(
                        left["row"]["metrics"][name] != right["row"]["metrics"][name]
                        for name in metrics
                    )
        if not comparable:
            return {
                "classification": "insufficient/incompatible",
                "boundary": edge_id,
                "reason": "no common applicable pane",
            }
        if changed:
            return {"classification": "first-detected-change", "boundary": edge_id}
    if not present_comparisons:
        return {
            "classification": "insufficient/incompatible",
            "boundary": None,
            "reason": "comparison order contains no present comparable edge",
        }
    return {"classification": "no-detected-change", "boundary": None}


def _classify_path(
    document: dict, index: dict, metric_kind: str, pane_id: str, edge_ids: list[str]
) -> dict:
    pane = _pane(document, pane_id)
    edges = {edge["id"]: edge for edge in document["edges"]}
    if not edge_ids:
        return {
            "classification": "insufficient/incompatible",
            "boundary": None,
            "reason": "path has no applicable edge",
        }
    for edge_id in edge_ids:
        edge = edges[edge_id]
        if edge["status"] != "present":
            return {
                "classification": "insufficient/incompatible",
                "boundary": edge_id,
                "reason": f"reviewed edge is {edge['status']}",
            }
        changed = False
        for world in pane["worlds"]:
            world_id = _world_id(world)
            left = index.get((edge["source"], pane_id, world_id))
            right = index.get((edge["destination"], pane_id, world_id))
            if left is None or right is None:
                return {
                    "classification": "insufficient/incompatible",
                    "boundary": edge_id,
                    "reason": "missing paired evaluation",
                }
            if metric_kind == "behavior":
                if (
                    "trajectory_digest" not in left["row"]["metrics"]
                    or "trajectory_digest" not in right["row"]["metrics"]
                ):
                    return {
                        "classification": "insufficient/incompatible",
                        "boundary": edge_id,
                        "reason": "trajectory digest unavailable",
                    }
                changed |= (
                    left["row"]["metrics"]["trajectory_digest"]
                    != right["row"]["metrics"]["trajectory_digest"]
                )
            else:
                metrics = MANDATORY_METRICS - {"trajectory_digest"}
                if not metrics <= set(left["row"]["metrics"]) or not metrics <= set(
                    right["row"]["metrics"]
                ):
                    return {
                        "classification": "insufficient/incompatible",
                        "boundary": edge_id,
                        "reason": "mandatory outcomes unavailable",
                    }
                changed |= any(
                    left["row"]["metrics"][name] != right["row"]["metrics"][name]
                    for name in metrics
                )
        if changed:
            return {"classification": "first-detected-change", "boundary": edge_id}
    return {"classification": "no-detected-change", "boundary": None}


def _classification_matrix(
    document: dict, index: dict, inventory: dict, metric_kind: str
) -> list[dict]:
    edge_map = {edge["id"]: edge for edge in inventory["edges"]}
    rows = []
    for sequence in inventory["sequences"]:
        for pane in document["evaluation"]["panes"]:
            field = (
                "historical_panes"
                if pane["kind"] in {"historical-semantics", "fresh-common-producer-diagnostic"}
                else "standardized_panes"
            )
            edges = [
                edge_id for edge_id in sequence["edges"] if pane["id"] in edge_map[edge_id][field]
            ]
            if not edges:
                continue
            result = _classify_path(document, index, metric_kind, pane["id"], edges)
            rows.append(
                {
                    "metric_kind": metric_kind,
                    "surface": pane["kind"],
                    "pane": pane["id"],
                    "intervention_label": pane.get("intervention_label"),
                    "sequence": sequence["id"],
                    "branch": sequence["branch"],
                    "edges": edges,
                    **result,
                }
            )
    return rows


def _classification_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "classification": "insufficient/incompatible",
            "boundary": None,
            "reason": "no reviewed applicable classification surface",
        }
    insufficient = next(
        (row for row in rows if row["classification"] == "insufficient/incompatible"), None
    )
    if insufficient is not None:
        return {
            "classification": "insufficient/incompatible",
            "boundary": insufficient.get("boundary"),
            "reason": "one or more branch/pane classifications are insufficient",
        }
    if len(rows) == 1:
        return {key: rows[0][key] for key in ("classification", "boundary")}
    return {
        "classification": "branch/path-specific",
        "boundary": None,
        "results_sha256": digest(rows),
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", f"--work-tree={REPOSITORY}", *args],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        _fail(f"cannot inspect executing implementation source tree: {error}")


def implementation_source_tree() -> dict:
    """Return recomputed hashes and Git dirty-state proof for executing Gate-1 sources."""
    commit = _git("rev-parse", "HEAD").strip()
    sources = [
        {"path": name, "sha256": checkpoint_hash(REPOSITORY / name)}
        for name in IMPLEMENTATION_SOURCES
    ]
    if not _is_sha(commit, 40) or any(not _is_sha(row["sha256"]) for row in sources):
        _fail("executing implementation has an unknown commit or missing required source")
    output = _git("status", "--porcelain=v1", "--untracked-files=all")
    entries = output.splitlines()
    dirty_state = {
        "clean": not entries,
        "scope": "entire-repository",
        "porcelain": entries,
        "porcelain_sha256": digest(entries),
    }
    return {"commit": commit, "source_files": sources, "dirty_state": dirty_state}


def _implementation_identity(
    ref: dict, base: Path, observed_source_tree: dict | None = None
) -> tuple[dict, list[str]]:
    identity, _ = _read_json_ref(ref, base, kind="gate1_implementation_identity")
    if "digest" not in identity:
        _fail("implementation identity lacks its canonical digest")
    observed = (
        implementation_source_tree() if observed_source_tree is None else observed_source_tree
    )
    reasons = []
    if identity.get("architecture_commit") != ARCHITECTURE_COMMIT:
        reasons.append("implementation identity does not bind approved architecture")
    if not _is_sha(identity.get("commit"), 40):
        reasons.append("implementation identity lacks a full clean commit")
    elif identity["commit"] != observed["commit"]:
        reasons.append("implementation commit differs from executing source tree")
    declared_sources = identity.get("source_files")
    expected_paths = list(IMPLEMENTATION_SOURCES)
    if (
        not isinstance(declared_sources, list)
        or [row.get("path") for row in declared_sources if isinstance(row, dict)] != expected_paths
        or any(
            not isinstance(row, dict)
            or set(row) != {"path", "sha256"}
            or not _is_sha(row.get("sha256"))
            for row in declared_sources
        )
    ):
        reasons.append("implementation identity lacks the complete required source-file inventory")
    else:
        observed_hashes = {row["path"]: row["sha256"] for row in observed["source_files"]}
        for row in declared_sources:
            if row["sha256"] != observed_hashes[row["path"]]:
                reasons.append(
                    f"implementation source hash differs from executing source: {row['path']}"
                )
    declared_dirty = identity.get("dirty_state")
    if (
        not isinstance(declared_dirty, dict)
        or set(declared_dirty) != {"clean", "scope", "porcelain", "porcelain_sha256"}
        or type(declared_dirty.get("clean")) is not bool
        or declared_dirty.get("scope") != "entire-repository"
        or not isinstance(declared_dirty.get("porcelain"), list)
        or any(not isinstance(row, str) for row in declared_dirty.get("porcelain", []))
        or declared_dirty.get("porcelain_sha256") != digest(declared_dirty.get("porcelain", []))
        or declared_dirty.get("clean") != (not declared_dirty.get("porcelain", []))
    ):
        reasons.append("implementation identity has invalid dirty-state proof")
    elif declared_dirty != observed["dirty_state"]:
        reasons.append("implementation dirty-state proof differs from executing source tree")
    if not observed["dirty_state"]["clean"]:
        reasons.append("executing implementation source tree is not clean")
    bound = {
        "artifact_sha256": ref.get("sha256"),
        "identity": identity,
        "observed_source_tree": observed,
        "source_tree_sha256": digest(observed),
    }
    return bound, reasons


def _verification_evidence(
    ref: dict | None, base: Path, implementation_commit: Any, protocol: dict, protocol_sha: str
) -> tuple[dict | None, list[str]]:
    if ref is None:
        return None, ["Gate 1 verification artifact is missing"]
    body, _ = _read_json_ref(ref, base, kind="gate1_verification")
    required = {
        "schema",
        "kind",
        "implementation_commit",
        "protocol_sha256",
        "execution",
        "machine",
        "test_receipts",
        "coverage",
        "missingness",
        "parity",
        "benchmarks",
        "digest",
    }
    if set(body) != required or not _is_sha(body.get("digest")):
        _fail("Gate 1 verification artifact has a noncanonical field set")
    execution = _execution_declaration(body["execution"], "Gate 1 verification artifact")
    reasons = []
    if not _is_sha(body.get("implementation_commit"), 40):
        _fail("Gate 1 verification artifact has an invalid implementation commit")
    if body["implementation_commit"] != implementation_commit:
        reasons.append("verification artifact is bound to another implementation commit")
    if body.get("protocol_sha256") != protocol_sha:
        _fail("verification artifact is bound to another frozen verification protocol")
    artifact_hashes = {ref["sha256"], protocol_sha}
    machine = body.get("machine")
    if (
        not isinstance(machine, dict)
        or set(machine) != {"name", "os", "cpu", "digest"}
        or any(
            not isinstance(machine.get(key), str) or not machine[key].strip()
            for key in ("name", "os", "cpu")
        )
        or machine.get("digest") != digest({k: v for k, v in machine.items() if k != "digest"})
    ):
        _fail("Gate 1 verification machine identity is invalid")
    test_receipts = body.get("test_receipts")
    if not isinstance(test_receipts, list):
        _fail("Gate 1 verification test receipts are malformed")
    required_tests = {row["id"]: row for row in protocol["tests"]}
    checked_runs = []
    seen_tests = set()
    for receipt_ref in test_receipts:
        run, _ = _read_json_ref(receipt_ref, base, kind="gate1_test_execution_receipt")
        if (
            set(run)
            != {
                "schema",
                "kind",
                "test_id",
                "command",
                "exit_code",
                "machine_digest",
                "started_at",
                "finished_at",
                "output",
                "digest",
            }
            or run.get("test_id") not in required_tests
            or run["test_id"] in seen_tests
            or run.get("command") != required_tests[run["test_id"]]["command"]
            or type(run.get("exit_code")) is not int
            or run.get("machine_digest") != machine["digest"]
            or not isinstance(run.get("started_at"), str)
            or not run["started_at"]
            or not isinstance(run.get("finished_at"), str)
            or not run["finished_at"]
        ):
            _fail("Gate 1 verification test receipt substitutes/omits a prescribed test command")
        _, output_sha = artifact(run["output"], base)
        artifact_hashes.update((receipt_ref["sha256"], output_sha))
        seen_tests.add(run["test_id"])
        checked_runs.append(
            {
                "id": run["test_id"],
                "command": run["command"],
                "exit_code": run["exit_code"],
                "output_sha256": output_sha,
                "receipt_sha256": receipt_ref["sha256"],
            }
        )
        if run["exit_code"] != 0:
            reasons.append("verification test command did not exit zero")
    if seen_tests != set(required_tests):
        _fail("Gate 1 verification omits a prescribed test ID")
    coverage, _ = _read_json_ref(body.get("coverage"), base, kind="gate1_coverage_report")
    if (
        set(coverage)
        != {
            "schema",
            "kind",
            "tool_sha256",
            "scope",
            "denominator",
            "exclusions",
            "covered",
            "total",
            "raw_measurements",
            "digest",
        }
        or coverage.get("tool_sha256") != protocol["coverage"]["tool"]["sha256"]
        or coverage.get("scope") != protocol["coverage"]["scope"]
        or coverage.get("denominator") != protocol["coverage"]["denominator"]
        or coverage.get("exclusions") != protocol["coverage"]["exclusions"]
        or type(coverage.get("covered")) is not int
        or coverage["covered"] < 0
        or type(coverage.get("total")) is not int
        or coverage["total"] <= 0
        or coverage["covered"] > coverage["total"]
    ):
        _fail("Gate 1 coverage report substitutes scope/tool/denominator or has invalid counts")
    coverage_raw, _ = _read_json_ref(coverage["raw_measurements"], base, kind="gate1_coverage_raw")
    if (
        set(coverage_raw)
        != {
            "schema",
            "kind",
            "tool_sha256",
            "scope",
            "denominator",
            "exclusions",
            "files",
            "digest",
        }
        or coverage_raw.get("tool_sha256") != protocol["coverage"]["tool"]["sha256"]
        or coverage_raw.get("scope") != protocol["coverage"]["scope"]
        or coverage_raw.get("denominator") != protocol["coverage"]["denominator"]
        or coverage_raw.get("exclusions") != protocol["coverage"]["exclusions"]
        or not isinstance(coverage_raw.get("files"), list)
    ):
        _fail("Gate 1 raw coverage measurements substitute scope/tool/denominator")
    raw_covered = raw_total = 0
    covered_paths = set()
    for row in coverage_raw["files"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "covered_lines", "missing_lines"}
            or row.get("path") in covered_paths
            or not isinstance(row.get("covered_lines"), list)
            or not isinstance(row.get("missing_lines"), list)
            or any(
                type(line) is not int or line <= 0
                for line in row["covered_lines"] + row["missing_lines"]
            )
            or len(set(row["covered_lines"])) != len(row["covered_lines"])
            or len(set(row["missing_lines"])) != len(row["missing_lines"])
            or set(row["covered_lines"]) & set(row["missing_lines"])
        ):
            _fail("Gate 1 raw coverage line measurements are malformed")
        covered_paths.add(row["path"])
        raw_covered += len(row["covered_lines"])
        raw_total += len(row["covered_lines"]) + len(row["missing_lines"])
    if (
        covered_paths != set(protocol["coverage"]["scope"])
        or raw_covered != coverage["covered"]
        or raw_total != coverage["total"]
    ):
        _fail("Gate 1 coverage summary does not derive from raw line measurements")
    artifact_hashes.update((body["coverage"]["sha256"], coverage["raw_measurements"]["sha256"]))
    observed_coverage = 100.0 * coverage["covered"] / coverage["total"]
    if observed_coverage < protocol["coverage"]["minimum_percent"]:
        reasons.append("verification coverage is below the frozen minimum")
    missingness = body.get("missingness")
    if (
        not isinstance(missingness, dict)
        or set(missingness) != {"required_missing"}
        or type(missingness.get("required_missing")) is not int
        or missingness["required_missing"] < 0
    ):
        _fail("Gate 1 verification missingness measurement is invalid")
    if missingness["required_missing"] != 0:
        reasons.append("verification reports required missingness")
    parity, _ = _read_json_ref(body.get("parity"), base, kind="gate1_diagnostic_off_parity")
    if (
        set(parity) != {"schema", "kind", "method", "scenarios", "digest"}
        or parity.get("method") != protocol["parity"]["method"]
        or not isinstance(parity.get("scenarios"), list)
    ):
        _fail("Gate 1 diagnostic-off parity report is malformed")
    parity_ids = set()
    for row in parity["scenarios"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "baseline", "diagnostic_off"}
            or row.get("id") in parity_ids
        ):
            _fail("Gate 1 diagnostic-off parity scenario is malformed/duplicated")
        _, baseline_sha = artifact(row["baseline"], base)
        _, diagnostic_sha = artifact(row["diagnostic_off"], base)
        artifact_hashes.update((baseline_sha, diagnostic_sha))
        parity_ids.add(row["id"])
        if baseline_sha != diagnostic_sha:
            reasons.append(f"diagnostic-off parity differs for scenario {row['id']}")
    if parity_ids != set(protocol["parity"]["scenarios"]):
        _fail("Gate 1 diagnostic-off parity substitutes/omits a prescribed scenario")
    artifact_hashes.add(body["parity"]["sha256"])
    benchmark_protocol = {row["id"]: row for row in protocol["benchmarks"]}
    benchmark_refs = body.get("benchmarks")
    if not isinstance(benchmark_refs, list):
        _fail("Gate 1 benchmark receipts are malformed")
    benchmark_summaries = []
    seen_benchmarks = set()
    for benchmark_ref in benchmark_refs:
        row, _ = _read_json_ref(benchmark_ref, base, kind="gate1_benchmark_receipt")
        required_fields = {
            "schema",
            "kind",
            "benchmark_id",
            "scenario",
            "baseline_sha256",
            "machine_digest",
            "elapsed_seconds",
            "work_units",
            "peak_memory_mib",
            "memory_method",
            "raw_measurements",
            "digest",
        }
        benchmark_id = row.get("benchmark_id")
        prescribed = benchmark_protocol.get(benchmark_id)
        if (
            set(row) != required_fields
            or prescribed is None
            or benchmark_id in seen_benchmarks
            or row.get("scenario") != prescribed["scenario"]
            or row.get("baseline_sha256") != prescribed["baseline"]["sha256"]
            or row.get("machine_digest") != machine["digest"]
            or type(row.get("elapsed_seconds")) not in (int, float)
            or not math.isfinite(row["elapsed_seconds"])
            or row["elapsed_seconds"] <= 0
            or type(row.get("work_units")) is not int
            or row["work_units"] <= 0
            or type(row.get("peak_memory_mib")) not in (int, float)
            or not math.isfinite(row["peak_memory_mib"])
            or row["peak_memory_mib"] < 0
            or row.get("memory_method") != prescribed["memory"]["method"]
        ):
            _fail("Gate 1 benchmark receipt substitutes a scenario/baseline/method or is malformed")
        raw, _ = _read_json_ref(row["raw_measurements"], base, kind="gate1_benchmark_raw")
        if (
            set(raw)
            != {
                "schema",
                "kind",
                "benchmark_id",
                "scenario",
                "machine_digest",
                "memory_method",
                "samples",
                "digest",
            }
            or raw.get("benchmark_id") != benchmark_id
            or raw.get("scenario") != prescribed["scenario"]
            or raw.get("machine_digest") != machine["digest"]
            or raw.get("memory_method") != prescribed["memory"]["method"]
            or not isinstance(raw.get("samples"), list)
            or not raw["samples"]
        ):
            _fail("Gate 1 raw benchmark measurements substitute scenario/machine/method")
        raw_elapsed = 0.0
        raw_work = 0
        raw_peak = 0.0
        iterations = set()
        for sample in raw["samples"]:
            if (
                not isinstance(sample, dict)
                or set(sample) != {"iteration", "elapsed_seconds", "work_units", "peak_memory_mib"}
                or type(sample.get("iteration")) is not int
                or sample["iteration"] < 0
                or sample["iteration"] in iterations
                or type(sample.get("elapsed_seconds")) not in (int, float)
                or not math.isfinite(sample["elapsed_seconds"])
                or sample["elapsed_seconds"] <= 0
                or type(sample.get("work_units")) is not int
                or sample["work_units"] <= 0
                or type(sample.get("peak_memory_mib")) not in (int, float)
                or not math.isfinite(sample["peak_memory_mib"])
                or sample["peak_memory_mib"] < 0
            ):
                _fail("Gate 1 raw benchmark sample is malformed")
            iterations.add(sample["iteration"])
            raw_elapsed += sample["elapsed_seconds"]
            raw_work += sample["work_units"]
            raw_peak = max(raw_peak, sample["peak_memory_mib"])
        if (
            not math.isclose(raw_elapsed, row["elapsed_seconds"], rel_tol=0, abs_tol=0)
            or raw_work != row["work_units"]
            or raw_peak != row["peak_memory_mib"]
        ):
            _fail("Gate 1 benchmark summary does not derive from raw samples")
        artifact_hashes.update((benchmark_ref["sha256"], row["raw_measurements"]["sha256"]))
        throughput = row["work_units"] / row["elapsed_seconds"]
        if throughput < prescribed["throughput"]["minimum"]:
            reasons.append(f"verification throughput is below budget for {benchmark_id}")
        if row["peak_memory_mib"] > prescribed["memory"]["maximum"]:
            reasons.append(f"verification peak memory exceeds budget for {benchmark_id}")
        benchmark_summaries.append(
            {
                "id": benchmark_id,
                "throughput": throughput,
                "peak_memory_mib": row["peak_memory_mib"],
                "receipt_sha256": benchmark_ref["sha256"],
            }
        )
        seen_benchmarks.add(benchmark_id)
    if seen_benchmarks != set(benchmark_protocol):
        _fail("Gate 1 verification omits a prescribed benchmark")
    return {
        "artifact_sha256": ref["sha256"],
        "evidence": body,
        "test_runs": checked_runs,
        "coverage_percent": observed_coverage,
        "benchmarks": benchmark_summaries,
        "execution": execution,
        "artifact_hashes": sorted(artifact_hashes),
    }, reasons


def _evaluation_inputs(values: list[dict], base: Path) -> tuple[list[dict], list[dict], list[str]]:
    reports = []
    records = []
    reasons = []
    for value in values:
        if (
            isinstance(value, dict)
            and set(value) == {"path", "sha256"}
            and isinstance(value.get("path"), str)
            and _is_sha(value.get("sha256"))
        ):
            report, _ = _read_json_ref(value, base, kind="initialization_evaluation")
            reports.append(report)
            records.append(
                {
                    "artifact_sha256": value["sha256"],
                    "ref": value,
                    "canonical_sha256": digest(report),
                    "content": report,
                }
            )
        elif isinstance(value, dict):
            reports.append(value)
            records.append(
                {
                    "artifact_sha256": None,
                    "ref": None,
                    "canonical_sha256": digest(value),
                    "content": value,
                }
            )
            reasons.append("evaluation report is unreferenced in-memory evidence")
        else:
            _fail("evaluation inputs must be report artifacts or report mappings")
    return reports, records, reasons


def _build_review_packet(
    document: dict,
    evaluations: list[dict],
    base: Path | str,
    *,
    cli: list[str],
    verification: dict | None,
    implementation: dict,
    manifest_ref: dict | None,
    observed_source_tree: dict | None,
) -> dict:
    base = Path(base)
    validation = validate_manifest(document, base)
    if (
        not isinstance(cli, list)
        or not cli
        or not all(isinstance(x, str) for x in cli)
        or not isinstance(implementation, dict)
    ):
        _fail("review packet needs canonical CLI and implementation identity")
    implementation_identity, implementation_reasons = _implementation_identity(
        implementation,
        base,
        observed_source_tree,
    )
    verification_protocol, verification_protocol_sha = _verification_protocol(document, base)
    verification_evidence, verification_reasons = _verification_evidence(
        verification,
        base,
        implementation_identity["identity"].get("commit"),
        verification_protocol,
        verification_protocol_sha,
    )
    reports, evaluation_records, evaluation_reasons = _evaluation_inputs(evaluations, base)
    index = _evaluation_index(document, reports, base)
    missing = []
    for pane in document["evaluation"]["panes"]:
        for node in pane["nodes"]:
            if _node(document, node)["status"] != "present":
                continue
            for world in pane["worlds"]:
                if (node, pane["id"], _world_id(world)) not in index:
                    missing.append({"node": node, "pane": pane["id"], "seed": world["seed"]})
    inventory = validation["lineage_inventory"]
    completeness = _completeness_audit(document, index, inventory)
    behavior_rows = _classification_matrix(document, index, inventory, "behavior")
    outcome_rows = _classification_matrix(document, index, inventory, "outcome")
    behavior, outcome = (
        _classification_summary(behavior_rows),
        _classification_summary(outcome_rows),
    )
    if completeness:
        boundary = completeness[0].get("edge")
        behavior = {
            "classification": "insufficient/incompatible",
            "boundary": boundary,
            "reason": "full comparison surface is incomplete",
        }
        outcome = dict(behavior)
    failed_equivalence = [
        row["edge"] for row in validation["equivalence"] if row["status"] != "PASS"
    ]
    if failed_equivalence:
        reason = "failed expected equivalence: " + ", ".join(failed_equivalence)
        behavior = {
            "classification": "insufficient/incompatible",
            "boundary": failed_equivalence[0],
            "reason": reason,
        }
        outcome = {
            "classification": "insufficient/incompatible",
            "boundary": failed_equivalence[0],
            "reason": reason,
        }
    reasons = list(validation["holds"])
    if missing:
        reasons.append("incomplete evaluation cohort")
    if completeness:
        reasons.append("mandatory comparison metrics/services are incomplete")
    if behavior["classification"] == "insufficient/incompatible":
        reasons.append("behavior classification insufficient/incompatible")
    if outcome["classification"] == "insufficient/incompatible":
        reasons.append("outcome classification insufficient/incompatible")
    reasons.extend(implementation_reasons)
    reasons.extend(verification_reasons)
    reasons.extend(evaluation_reasons)
    if manifest_ref is None:
        reasons.append("manifest is not a content-addressed packet input")
        manifest_artifact_sha = None
    else:
        manifest_path, manifest_artifact_sha = artifact(manifest_ref, base)
        if json.loads(manifest_path.read_text()) != document:
            _fail("manifest artifact content differs from packet document")
    raw_rows = [value["row"] for _, value in sorted(index.items())]
    paired_aggregation = _paired_aggregation(document, index, inventory)
    aggregates = {
        "behavior": behavior,
        "outcome": outcome,
        "missing": missing,
        "classifications": {"behavior": behavior_rows, "outcome": outcome_rows},
        "completeness": completeness,
        "paired": paired_aggregation,
    }
    artifact_roots = [document, implementation]
    if verification is not None:
        artifact_roots.append(verification)
    if manifest_ref is not None:
        artifact_roots.append(manifest_ref)
    artifact_roots.extend(
        record["ref"] for record in evaluation_records if record["ref"] is not None
    )
    artifact_sources = _transitive_artifacts(artifact_roots, base)
    packet_artifact_hashes = set(artifact_sources)
    request = {
        "manifest_ref": manifest_ref,
        "evaluation_inputs": [
            (record["ref"] if record["ref"] is not None else record["content"])
            for record in evaluation_records
        ],
        "verification_ref": verification,
        "implementation_ref": implementation,
        "cli": cli,
    }
    inputs = {
        "manifest": {
            "artifact_sha256": manifest_artifact_sha,
            "canonical_sha256": digest(document),
            "content": document,
        },
        "evaluation_reports": evaluation_records,
        "raw_rows": {"canonical_sha256": digest(raw_rows), "content": raw_rows},
        "verification": verification_evidence,
        "implementation_identity": implementation_identity,
        "reconstruction_request": request,
    }
    packet = {
        "schema": 1,
        "kind": "gate1_review_packet",
        "decision": "HOLD" if reasons else "PASS",
        "architecture_commit": ARCHITECTURE_COMMIT,
        "implementation": implementation_identity,
        "manifest_digest": document["digest"],
        "validation_digest": validation["digest"],
        "execution": {
            "tiers": sorted(
                {pane["execution"]["tier"] for pane in document["evaluation"]["panes"]}
            ),
            "selection_allowed": False,
            "readiness_claimed": False,
        },
        "cli": cli,
        "expanded_evaluation": [
            {"pane": p["id"], "nodes": p["nodes"], "worlds": p["worlds"]}
            for p in document["evaluation"]["panes"]
        ],
        "artifact_hashes": sorted(packet_artifact_hashes),
        "raw_row_sha256": digest(raw_rows),
        "aggregate_sha256": digest(aggregates),
        "behavior": behavior,
        "outcome": outcome,
        "classifications": {"behavior": behavior_rows, "outcome": outcome_rows},
        "equivalence": validation["equivalence"],
        "missing": missing,
        "completeness": completeness,
        "paired_aggregation": paired_aggregation,
        "verification": verification_evidence,
        "inputs": inputs,
        "supported_claims": (
            [] if reasons else ["complete Gate 1 initialization-loss decomposition"]
        ),
        "unsupported_claims": sorted(set(reasons)),
        "unresolved_claims": sorted(set(reasons)),
    }
    packet["digest"] = digest(packet)
    result = ReviewPacket(packet)
    result.artifact_sources = artifact_sources
    return result


def build_review_packet(
    document: dict,
    evaluations: list[dict],
    base: Path | str,
    *,
    cli: list[str],
    verification: dict | None,
    implementation: dict,
    manifest_ref: dict | None = None,
) -> dict:
    """Build a packet while always binding identity to the live executing source tree."""
    return _build_review_packet(
        document,
        evaluations,
        base,
        cli=cli,
        verification=verification,
        implementation=implementation,
        manifest_ref=manifest_ref,
        observed_source_tree=None,
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path = _native_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive creation makes content-addressed packets append-only.  A
        # temporary-file rename is not portable into Windows AppContainer test
        # directories, where the temporary file can receive a different label.
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        _fail(f"refusing to overwrite review artifact {path}")


def write_review_packet(packet: dict, root: Path | str) -> Path:
    root = Path(root)
    directory = root / packet["digest"]
    sources = getattr(packet, "artifact_sources", None)
    if not isinstance(sources, dict) or set(sources) != set(packet["artifact_hashes"]):
        _fail("review packet lacks the complete transitive artifact source inventory")
    for sha, source in sorted(sources.items()):
        if checkpoint_hash(source) != sha:
            _fail(f"packet artifact changed before preservation: {source}")
        _atomic_write(directory / "cas" / sha, source.read_bytes())
    report = _canonical(packet) + b"\n"
    markdown = (
        f"# Gate 1 decision: {packet['decision']}\n\n"
        f"Packet: `{packet['digest']}`\n\n"
        f"Behavior: `{packet['behavior']['classification']}`\n\n"
        f"Outcome: `{packet['outcome']['classification']}`\n\n"
        + (
            "## HOLD reasons\n\n" + "\n".join(f"- {x}" for x in packet["unsupported_claims"]) + "\n"
            if packet["unsupported_claims"]
            else "## Supported\n\n- Complete Gate 1 packet.\n"
        )
    )
    _atomic_write(directory / "report.json", report)
    _atomic_write(directory / "decision.md", markdown.encode())
    inputs = packet["inputs"]
    _atomic_write(directory / "inputs" / "manifest.json", _canonical(inputs["manifest"]) + b"\n")
    _atomic_write(
        directory / "inputs" / "evaluation-reports.json",
        _canonical(inputs["evaluation_reports"]) + b"\n",
    )
    _atomic_write(directory / "inputs" / "raw-rows.json", _canonical(inputs["raw_rows"]) + b"\n")
    _atomic_write(
        directory / "inputs" / "verification.json", _canonical(inputs["verification"]) + b"\n"
    )
    _atomic_write(
        directory / "inputs" / "implementation-identity.json",
        _canonical(inputs["implementation_identity"]) + b"\n",
    )
    _atomic_write(
        directory / "inputs" / "reconstruction-request.json",
        _canonical(inputs["reconstruction_request"]) + b"\n",
    )
    input_files = sorted((directory / "inputs").glob("*.json"))
    completion = {
        "schema": 1,
        "kind": "gate1_completion",
        "packet_digest": packet["digest"],
        "decision": packet["decision"],
        "report_sha256": checkpoint_hash(directory / "report.json"),
        "decision_sha256": checkpoint_hash(directory / "decision.md"),
        "artifact_hashes": sorted(packet["artifact_hashes"]),
        "input_hashes": {path.name: checkpoint_hash(path) for path in input_files},
    }
    completion["digest"] = digest(completion)
    _atomic_write(directory / "completion.json", _canonical(completion) + b"\n")
    return directory


def _preserved_implementation_source_tree(ref: dict, directory: Path) -> dict:
    """Recompute a reconstruction-only source tree from identity-bound CAS bytes."""
    identity, _ = _read_json_ref(ref, directory, kind="gate1_implementation_identity")
    source_files = identity.get("source_files")
    if not isinstance(source_files, list) or [
        row.get("path") for row in source_files if isinstance(row, dict)
    ] != list(IMPLEMENTATION_SOURCES):
        _fail("preserved implementation identity lacks the required source inventory")
    checked_sources = []
    for row in source_files:
        if set(row) != {"path", "sha256"} or not _is_sha(row.get("sha256")):
            _fail("preserved implementation source identity is malformed")
        cas_path, _ = artifact(row, directory)
        checked_sources.append({"path": row["path"], "sha256": checkpoint_hash(cas_path)})
    return {
        "commit": identity.get("commit"),
        "source_files": checked_sources,
        "dirty_state": identity.get("dirty_state"),
    }


def _validate_packet_cas(directory: Path, expected: Any) -> set[str]:
    """Validate the completion-bound CAS in full before replay reads evidence."""
    if (
        not isinstance(expected, list)
        or not expected
        or expected != sorted(set(expected))
        or any(not _is_sha(value) for value in expected)
    ):
        _fail("packet completion record has a malformed CAS inventory")
    _plain_directory(_native_path(directory), "packet directory")
    cas = _native_path(directory / "cas")
    _plain_directory(cas, "packet CAS")
    entries = list(cas.iterdir())
    for entry in entries:
        _plain_file(entry, "packet CAS entry")
        if not _is_sha(entry.name):
            _fail(f"packet CAS entry has a non-digest filename: {entry.name}")
    expected_set = set(expected)
    actual_set = {entry.name for entry in entries}
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        _fail(f"packet CAS inventory differs from completion (missing={missing}, extra={extra})")
    for sha in expected:
        path = _native_path(cas / sha)
        if checkpoint_hash(path) != sha:
            _fail(f"packet CAS artifact content hash mismatch: {sha}")
    return expected_set


def reconstruct_review_packet(directory: Path | str) -> dict:
    directory = Path(directory)
    completion_path = directory / "completion.json"
    completion = json.loads(completion_path.read_text())
    if (
        completion.get("digest")
        != digest({key: value for key, value in completion.items() if key != "digest"})
        or completion.get("packet_digest") != directory.name
    ):
        _fail("packet completion record is unbound or directory identity differs")
    cas_inventory = _validate_packet_cas(directory, completion.get("artifact_hashes"))
    for name, expected in completion.get("input_hashes", {}).items():
        path = directory / "inputs" / name
        if checkpoint_hash(path) != expected:
            _fail(f"preserved packet input changed: {name}")
    report_path = directory / "report.json"
    decision_path = directory / "decision.md"
    if checkpoint_hash(report_path) != completion.get("report_sha256") or checkpoint_hash(
        decision_path
    ) != completion.get("decision_sha256"):
        _fail("packet report or decision rendering changed after completion")
    stored = json.loads(report_path.read_text())
    request = json.loads((directory / "inputs" / "reconstruction-request.json").read_text())
    required_request = {
        "manifest_ref",
        "evaluation_inputs",
        "verification_ref",
        "implementation_ref",
        "cli",
    }
    if set(request) != required_request or request["manifest_ref"] is None:
        _fail("packet reconstruction request is incomplete")
    with _packet_cas_only(directory):
        manifest_path, _ = artifact(request["manifest_ref"], directory)
        document = json.loads(manifest_path.read_text())
        preserved_tree = _preserved_implementation_source_tree(
            request["implementation_ref"],
            directory,
        )
        rebuilt = _build_review_packet(
            document,
            request["evaluation_inputs"],
            directory,
            cli=request["cli"],
            verification=request["verification_ref"],
            implementation=request["implementation_ref"],
            manifest_ref=request["manifest_ref"],
            observed_source_tree=preserved_tree,
        )
    if _canonical(rebuilt) != _canonical(stored) or rebuilt["digest"] != directory.name:
        _fail("packet derived outputs do not reconstruct from preserved raw inputs")
    if set(rebuilt["artifact_hashes"]) != cas_inventory:
        _fail("packet CAS differs from the reconstructed transitive artifact inventory")
    return rebuilt


def _write_json(path: Path, value: dict) -> None:
    _atomic_write(
        path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict Gate 1 artifact-lineage diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--node", required=True)
    evaluate.add_argument("--pane", required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    packet = sub.add_parser("packet")
    packet.add_argument("--manifest", type=Path, required=True)
    packet.add_argument("--evaluations", type=Path, required=True)
    packet.add_argument("--verification", type=Path, required=True)
    packet.add_argument("--implementation", type=Path, required=True)
    packet.add_argument("--out", type=Path, required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--inventory", type=Path, required=True)
    preflight.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from .protocol import command

    command(sys.argv[1:])
    if args.command == "preflight":
        from .gate1_preflight import run_preflight

        result = run_preflight(args.inventory)
        _write_json(args.out, result)
        return 0 if result["ready_for_protocol_freeze"] else 2
    document, validation, manifest_path = load_manifest(args.manifest)
    if args.command == "validate":
        _write_json(args.out, validation)
        return 0
    if args.command == "evaluate":
        _write_json(args.out, evaluate_node(document, manifest_path.parent, args.node, args.pane))
        return 0
    evaluations = json.loads(args.evaluations.read_text())
    if not isinstance(evaluations, list):
        _fail("--evaluations must contain a JSON list of artifact refs")
    manifest_ref = {"path": str(manifest_path), "sha256": checkpoint_hash(manifest_path)}
    verification = {
        "path": str(args.verification.resolve()),
        "sha256": checkpoint_hash(args.verification),
    }
    implementation = {
        "path": str(args.implementation.resolve()),
        "sha256": checkpoint_hash(args.implementation),
    }
    result = build_review_packet(
        document,
        evaluations,
        manifest_path.parent,
        cli=sys.argv[1:],
        verification=verification,
        implementation=implementation,
        manifest_ref=manifest_ref,
    )
    output = write_review_packet(result, args.out)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
