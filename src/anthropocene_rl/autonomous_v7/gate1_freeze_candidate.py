"""Build and validate the Model9 Gate-1 production freeze candidate.

This module performs only read-only artifact inspection and proposal creation.
It never evaluates a checkpoint, starts the simulator or collects benchmark
treatment measurements.  Every generated JSON document is schema-1 and sealed
by a digest over all fields other than ``digest``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contracts
from .gate1_production_adapter import (
    CONTRACT8_ACTIONS,
    CONTRACT8_OUTPUTS,
    GUARDED_BC_ACTIONS,
    GUARDED_BC_OUTPUTS,
    comparator_runtime_identity,
    guard_only_protocol_fields,
    inactive_region_state,
    load_recovered_contract12,
    run_contract8_comparator,
    run_guarded_bc_comparator,
    tensor_state_identity,
    validate_contract8_corpus,
)
from .protocol import DENIED, ROLES
from .provenance import ProvenanceError, checkpoint_hash, digest

SCHEMA = 1
PINNED_CONTRACT8_COMMIT = "2f4cc895e6bcd897200d340851d69be98fd18e00"
FAMILY_SIZE = 16
T2M_ARCHIVE_SHA256 = "25189c8237287508368d67e71461e071b38bd1d79b74850fd775807fe03c7d8c"
MASKS = (
    ("default", 0),
    ("farming", contracts.HUSBANDRY | contracts.WATERCRAFT),
    ("fishing", contracts.HUSBANDRY | contracts.CULTIVATION),
    ("herding", contracts.CULTIVATION | contracts.WATERCRAFT),
)
MANDATORY_METRICS = (
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
)
PROPOSAL_DECLARATIONS = {
    "docs/MODEL9_GATE1_PRODUCTION_PROTOCOL.md",
    "src/anthropocene_rl/autonomous_v7/gate1_freeze_candidate.py",
    "src/anthropocene_rl/autonomous_v7/scale_probe.py",
    "tests/test_gate1_freeze_candidate.py",
    "tests/test_scale_probe.py",
}
IMPLEMENTATION_SOURCES = (
    "AGENTS.md",
    "docs/MODEL9_DIAGNOSTIC_CLOSURE_ARCHITECTURE.md",
    "docs/MODEL9_ECONOMIC_OBSERVABILITY_ARCHITECTURE.md",
    "docs/MODEL9_GATE1_PRODUCTION_PROTOCOL.md",
    "docs/MODEL9_GATE1_IMPLEMENTATION_STATUS.md",
    "src/anthropocene_rl/autonomous_v7/initialization_lineage.py",
    "src/anthropocene_rl/autonomous_v7/evaluate.py",
    "src/anthropocene_rl/autonomous_v7/gate1_diagnostics.py",
    "src/anthropocene_rl/autonomous_v7/gate1_preflight.py",
    "src/anthropocene_rl/autonomous_v7/legacy_contract8.py",
    "src/anthropocene_rl/autonomous_v7/gate1_production_adapter.py",
    "src/anthropocene_rl/autonomous_v7/gate1_freeze_candidate.py",
    "src/anthropocene_rl/autonomous_v7/scale_probe.py",
    "tests/test_initialization_lineage.py",
    "tests/test_gate1_diagnostics.py",
    "tests/test_gate1_preflight.py",
    "tests/test_gate1_legacy_contract8.py",
    "tests/test_gate1_freeze_candidate.py",
    "tests/test_scale_probe.py",
)
EXECUTION_SOURCES = (
    "src/anthropocene_rl/autonomous_v7/bridge.py",
    "src/anthropocene_rl/autonomous_v7/checkpoint.py",
    "src/anthropocene_rl/autonomous_v7/contracts.py",
    "src/anthropocene_rl/autonomous_v7/diagnostics.py",
    "src/anthropocene_rl/autonomous_v7/economic_accounting.py",
    "src/anthropocene_rl/autonomous_v7/economic_trace.py",
    "src/anthropocene_rl/autonomous_v7/evaluate.py",
    "src/anthropocene_rl/autonomous_v7/gate1_diagnostics.py",
    "src/anthropocene_rl/autonomous_v7/gate1_freeze_candidate.py",
    "src/anthropocene_rl/autonomous_v7/gate1_production_adapter.py",
    "src/anthropocene_rl/autonomous_v7/legacy_contract8.py",
    "src/anthropocene_rl/autonomous_v7/model.py",
    "src/anthropocene_rl/autonomous_v7/protocol.py",
    "src/anthropocene_rl/autonomous_v7/provenance.py",
    "src/anthropocene_rl/autonomous_v7/scale_probe.py",
    "src/anthropocene_rl/autonomous_v7/teacher_data.py",
    "src/anthropocene_rl/autonomous_v7/upgrade.py",
    "src/anthropocene_rl/autonomous_v7/worker_contract.py",
)


def _fail(message: str) -> None:
    raise ProvenanceError("Gate 1 freeze candidate: " + message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    if "digest" in value:
        _fail("cannot reseal a document carrying a digest")
    value = dict(value)
    value["digest"] = digest(value)
    return value


def _validate_seal(value: Any, kind: str | None = None) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA
        or (kind is not None and value.get("kind") != kind)
        or value.get("digest")
        != digest({key: item for key, item in value.items() if key != "digest"})
    ):
        _fail(f"{kind or 'schema-1 document'} is unsealed or malformed")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")


def _ref(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": checkpoint_hash(path),
        "size": path.stat().st_size,
    }


def _read_ref(ref: dict[str, Any], base: Path, *, kind: str | None = None) -> dict[str, Any]:
    if not isinstance(ref, dict) or set(ref) not in (
        {"path", "sha256"},
        {"path", "sha256", "size"},
    ):
        _fail("artifact reference has a noncanonical field set")
    path = base / ref["path"]
    if (
        not path.is_file()
        or checkpoint_hash(path) != ref["sha256"]
        or ("size" in ref and path.stat().st_size != ref["size"])
    ):
        _fail(f"artifact reference is missing or changed: {ref.get('path')}")
    value = json.loads(path.read_text(encoding="utf-8"))
    return _validate_seal(value, kind)


def _adapter_ref(ref: dict[str, Any], base: Path, *, kind: str) -> dict[str, str]:
    """Validate a production proposal ref and narrow it for adapter APIs.

    Proposal references also bind byte size.  The adapter boundary deliberately
    accepts only path/hash, so normalize only after validating all proposal
    fields and resolve the path relative to the freeze package here.
    """
    _read_ref(ref, base, kind=kind)
    path = Path(ref["path"])
    if not path.is_absolute():
        path = (base / path).resolve()
    return {"path": str(path), "sha256": ref["sha256"]}


def _file_identity(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix() if root is not None else str(path),
        "sha256": checkpoint_hash(path),
        "size": path.stat().st_size,
    }


def _git(rl_root: Path, *args: str) -> str:
    command = ["git", f"--work-tree={rl_root}", *args]
    return subprocess.check_output(
        command, cwd=rl_root, text=True, encoding="utf-8", errors="strict"
    ).strip()


def _git_bytes(rl_root: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", f"--work-tree={rl_root}", "show", f"{commit}:{path}"],
        cwd=rl_root,
    )


def _package_sums(package: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    sums_path = package / "SHA256SUMS"
    if not sums_path.is_file() or sums_path.is_symlink():
        _fail(f"evidence package lacks a regular SHA256SUMS: {package}")
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ", 1)
        if name in sums or not re.fullmatch(r"[0-9a-f]{64}", sha):
            _fail(f"evidence SHA256SUMS is malformed: {package}")
        path = package / name
        if not path.is_file() or path.is_symlink() or checkpoint_hash(path) != sha:
            _fail(f"corrected evidence file changed: {name}")
        sums[name] = sha
    return sums


def _evidence_package(package: Path, retrieval: Path, review_path: Path) -> dict[str, Any]:
    """Validate the accepted v2 correction and its factual v1 raw-data input.

    The v1 package remains the only convenient 396-file/checkpoint-schema table,
    but its guarded-clone interpretation is explicitly superseded by v2.  No v1
    label is allowed to establish the DAG.  The root review binds the exact v2
    checksum manifest accepted for factual inventory/source-data use only.
    """
    sums = _package_sums(package)
    required = {
        "retrieval-inventory.json",
        "checkpoint-inventory.json",
        "lineage.json",
        "evaluation-summary.json",
        "REPORT.md",
    }
    v2_required = {
        "analysis-environment.json",
        "build_evidence.py",
        "checkpoint-inventory.json",
        "lineage.json",
        "REPORT.md",
        "retrieval-integrity.json",
        "reward-architecture-characterization.json",
        "source-log-alignment.json",
    }
    if set(sums) != v2_required:
        _fail("accepted v2 evidence package inventory is incomplete or expanded")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        not review_path.is_file()
        or review_path.is_symlink()
        or review.get("schema") != 1
        or review.get("kind") != "root_recovered_evidence_review"
        or review.get("decision")
        != "accepted-for-factual-inventory-and-source-data-characterization"
        or review.get("production_gate1") != "HOLD"
        or review.get("readiness_claimed") is not False
        or review.get("independently_verified_package_files") != len(sums)
        or review.get("sha256sums_sha256") != checkpoint_hash(package / "SHA256SUMS")
    ):
        _fail("root review does not bind the exact accepted v2 factual package")
    v2_lineage = json.loads((package / "lineage.json").read_text(encoding="utf-8"))
    v2_checkpoints = json.loads((package / "checkpoint-inventory.json").read_text(encoding="utf-8"))
    v2_integrity = json.loads((package / "retrieval-integrity.json").read_text(encoding="utf-8"))
    if (
        v2_lineage.get("schema") != "model9-gate1-vm-retrieval-analysis/v2"
        or v2_checkpoints.get("schema") != "model9-gate1-vm-retrieval-analysis/v2"
        or v2_integrity.get("schema") != "model9-gate1-vm-retrieval-analysis/v2"
        or v2_lineage.get("guarded_t2m_is_not_t3_ancestor", {}).get("value") is not True
        or v2_lineage.get("unguarded_t3_initializer", {}).get("role")
        != "unguarded clone used as recovered T3 initialization/teacher binding"
        or len(v2_lineage.get("t3_branch_bindings", {}).get("records", ())) != 30
    ):
        _fail("accepted v2 evidence omits its factual lineage correction")

    raw_package = package.parent / "model9-gate1-vm-retrieval-analysis"
    raw_sums = _package_sums(raw_package)
    if not required <= set(raw_sums):
        _fail("v1 raw-data package needed for the 396-file table is incomplete")
    inventory = json.loads((raw_package / "retrieval-inventory.json").read_text(encoding="utf-8"))
    if inventory.get("status") != "verified-byte-inventory" or inventory.get("file_count") != 396:
        _fail("retrieval evidence does not prove the exact 396-file copy")
    declared_generated = {
        "producer/" + row
        for row in v2_integrity.get("nested_source_archive", {}).get(
            "additional_runtime_generated_source_tree_files", ()
        )
    }
    generated = {
        row["path"]
        for row in inventory["files"]
        if "__pycache__" in Path(row["path"]).parts and row["path"].endswith(".pyc")
    }
    if not declared_generated <= generated:
        _fail("v2 runtime-generated retrieval declarations escape the raw pycache inventory")
    generated_observed = []
    for row in inventory["files"]:
        path = retrieval / Path(row["path"])
        if not path.is_file() or path.is_symlink():
            _fail(f"retrieved artifact differs from verified inventory: {row['path']}")
        if row["path"] in generated:
            generated_observed.append(_file_identity(path, root=retrieval))
        elif path.stat().st_size != row["size"] or checkpoint_hash(path) != row["sha256"]:
            _fail(f"retrieved artifact differs from verified inventory: {row['path']}")
    if {row["path"] for row in generated_observed} != generated:
        _fail("runtime-generated retrieval pycache inventory differs")
    raw_checkpoints = json.loads(
        (raw_package / "checkpoint-inventory.json").read_text(encoding="utf-8"),
    )
    raw_lineage = json.loads((raw_package / "lineage.json").read_text(encoding="utf-8"))
    records = {row["path"]: row for row in raw_checkpoints.get("checkpoints", ())}
    initializer = v2_lineage["unguarded_t3_initializer"]
    if records.get("stage1-token-s6.pt", {}).get("sha256") != initializer.get(
        "sha256"
    ) or records.get("t2/stage1-token-s6.bc.pt", {}).get("sha256") != initializer.get(
        "original_bc_phase_sha256"
    ):
        _fail("v1 raw checkpoint table disagrees with the accepted v2 initializer correction")
    for row in v2_lineage["t3_branch_bindings"]["records"]:
        if records.get(row["path"], {}).get("sha256") != row.get("sha256"):
            _fail(f"v1 raw checkpoint table disagrees with v2: {row.get('path')}")
    return {
        "sums": sums,
        "raw_sums": raw_sums,
        "retrieval": inventory,
        "checkpoints": raw_checkpoints,
        "lineage": raw_lineage,
        "v2_lineage": v2_lineage,
        "v2_checkpoints": v2_checkpoints,
        "v2_integrity": v2_integrity,
        "package": package,
        "raw_package": raw_package,
        "root_review": review,
        "root_review_path": review_path,
        "runtime_generated_observed": generated_observed,
    }


def _copy_evidence_binding(evidence: dict[str, Any], output: Path) -> Path:
    """Copy the accepted factual evidence inputs into the portable package."""
    copied: dict[str, list[dict[str, Any]]] = {}
    for label, package, sums in (
        ("accepted-v2", evidence["package"], evidence["sums"]),
        ("v1-raw-data", evidence["raw_package"], evidence["raw_sums"]),
    ):
        destination = output / "recovered-evidence" / label
        destination.mkdir(parents=True, exist_ok=True)
        rows = []
        for name, expected in sorted(sums.items()):
            source = package / name
            target = destination / name
            shutil.copyfile(source, target)
            if checkpoint_hash(target) != expected:
                _fail(f"evidence copy changed: {label}/{name}")
            rows.append(_ref(target, output))
        sums_source = package / "SHA256SUMS"
        sums_target = destination / "SHA256SUMS"
        shutil.copyfile(sums_source, sums_target)
        rows.append(_ref(sums_target, output))
        copied[label] = rows
    review_target = output / "recovered-evidence" / "root-review.json"
    shutil.copyfile(evidence["root_review_path"], review_target)
    binding = _seal(
        {
            "schema": 1,
            "kind": "gate1_recovered_evidence_binding",
            "accepted_scope": "factual-inventory-and-source-data-characterization-only",
            "production_gate1": "HOLD",
            "v2_sha256sums_sha256": checkpoint_hash(
                evidence["package"] / "SHA256SUMS",
            ),
            "v1_labels_superseded": ["guarded clone", "guarded contract-12 clone"],
            "v1_usage": "raw 396-file/checkpoint schema table only; no lineage label authority",
            "packages": copied,
            "root_review": _ref(review_target, output),
        }
    )
    path = output / "recovered-evidence-binding.json"
    _write_json(path, binding)
    return path


def _copy_expert_expansion_evidence(package: Path, retrieval: Path, output: Path) -> Path:
    sums = _package_sums(package)
    expected = {
        "analysis-environment.json",
        "audit_expert_expansion.py",
        "expert-expansion-audit.json",
        "REPORT.md",
    }
    if set(sums) != expected:
        _fail("expert-expansion evidence inventory is incomplete or expanded")
    audit = json.loads((package / "expert-expansion-audit.json").read_text(encoding="utf-8"))
    initializer = audit.get("actual_initializer", {})
    summary = audit.get("summary", {})
    sources = audit.get("sources", {}).get("files", ())
    if (
        audit.get("schema") != "model9-gate1-expert-expansion-equivalence/v1"
        or audit.get("status") != "equivalent"
        or initializer.get("path") != "gate1-vm-retrieval/stage1-token-s6.pt"
        or checkpoint_hash(retrieval / "stage1-token-s6.pt") != initializer.get("sha256")
        or summary.get("all_seed_and_batch_forward_outputs_and_decoded_modes_equivalent")
        is not True
        or summary.get("all_seeds_only_missing_expert_parameters") is not True
        or summary.get("all_seeds_copied_tensors_bit_exact") is not True
        or summary.get("all_seeds_final_adapters_exactly_zero") is not True
        or summary.get("maximum_finite_forward_error") != 0.0
        or summary.get("optimizer_seed_count") != 15
    ):
        _fail("expert-expansion evidence does not establish the exact virtual entry equivalence")
    for row in sources:
        prefix = "gate1-vm-retrieval/"
        relative = row.get("path", "")
        if not isinstance(relative, str) or not relative.startswith(prefix):
            _fail("expert-expansion source reference escapes the retrieval")
        path = retrieval / relative[len(prefix) :]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("size")
            or checkpoint_hash(path) != row.get("sha256")
        ):
            _fail(f"expert-expansion source changed: {relative}")
    destination = output / "expert-expansion-evidence"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, sha in sorted(sums.items()):
        target = destination / name
        shutil.copyfile(package / name, target)
        if checkpoint_hash(target) != sha:
            _fail(f"expert-expansion evidence copy changed: {name}")
        rows.append(_ref(target, output))
    sums_target = destination / "SHA256SUMS"
    shutil.copyfile(package / "SHA256SUMS", sums_target)
    rows.append(_ref(sums_target, output))
    binding = _seal(
        {
            "schema": 1,
            "kind": "gate1_expert_expansion_equivalence_binding",
            "status": "equivalent-on-frozen-corpus",
            "scope": "virtual PPO-entry parameter expansion only; no rollout node or outcome claim",
            "initializer_sha256": initializer["sha256"],
            "optimizer_seeds": list(range(70, 85)),
            "maximum_finite_forward_error": 0.0,
            "files": rows,
        }
    )
    path = output / "expert-expansion-equivalence-binding.json"
    _write_json(path, binding)
    return path


def _t2m_inventory(root: Path) -> dict[str, Any]:
    """Bind the exact recovered guarded-BC attempt without inferring completion."""
    archive = root.parent / "model9-t2m-evidence-20261023.tar.gz"
    if (
        not root.is_dir()
        or root.is_symlink()
        or not archive.is_file()
        or checkpoint_hash(archive) != T2M_ARCHIVE_SHA256
    ):
        _fail("T2m retrieval/archive identity is absent or changed")
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            _fail("T2m retrieval contains a symlink")
        files.append(_file_identity(path, root=root))
    checkpoints = [
        row
        for row in files
        if row["path"].startswith("t2m/runs/stage1-token-s") and row["path"].endswith(".bc.pt")
    ]
    seeds = set()
    for row in checkpoints:
        match = re.fullmatch(r"t2m/runs/stage1-token-s([0-9]+)\.bc\.pt", row["path"])
        if match is None:
            _fail("T2m guarded BC checkpoint path is malformed")
        seed = int(match.group(1))
        seeds.add(seed)
        state = torch.load(root / row["path"], map_location="cpu", weights_only=True)
        if (
            not isinstance(state, dict)
            or set(state) != {"actor", "bc_config", "log", "normalizer"}
            or len(state["log"]) != 12
            or state["normalizer"].get("guard") != 10.0
        ):
            _fail(f"T2m guarded BC checkpoint schema differs: seed {seed}")
    if seeds != {0, 1, 2, 3, 4, 6, 7, 8, 9}:
        _fail("T2m guarded BC recovered seed inventory differs")
    completed = [
        row for row in files if row["path"].endswith(".pt") and not row["path"].endswith(".bc.pt")
    ]
    if completed:
        _fail("T2m retrieval contains an unclassified completed checkpoint")
    return _seal(
        {
            "schema": 1,
            "kind": "gate1_t2m_retrieval_inventory",
            "archive": {
                "path": str(archive),
                "sha256": T2M_ARCHIVE_SHA256,
                "size": archive.stat().st_size,
                "remote_hash_match_assertion": True,
            },
            "files": files,
            "file_count": len(files),
            "guarded_bc_seeds": sorted(seeds),
            "selected_actor_seed": 6,
            "selection_basis": "same initialization seed as recovered historical T3 "
            "initialization; "
            "not "
            "outcome selection",
            "completed_guarded_critic_checkpoint_recovered": False,
            "scope_limit": "absence in this archive is not proof of global historical "
            "non-production",
        }
    )


def _repo_files(rl_root: Path) -> list[str]:
    result = _git(rl_root, "ls-files", "--cached", "--others", "--exclude-standard")
    return sorted({row.replace("\\", "/") for row in result.splitlines() if row.strip()})


def _seed_pattern(seeds: Iterable[int]) -> re.Pattern[bytes]:
    options = b"|".join(str(seed).encode() for seed in seeds)
    # A decimal-looking substring inside a hexadecimal digest is not evidence
    # that a simulator seed was used. Keep those strings in the scanned-file
    # inventory through their file hashes, but only recognize complete textual
    # scalar tokens here. Seed-labelled forms such as ``s50000`` are handled
    # separately below.
    return re.compile(rb"(?<![A-Za-z0-9])(?:" + options + rb")(?![A-Za-z0-9])")


def _labelled_seed_pattern(seeds: Iterable[int]) -> re.Pattern[bytes]:
    options = b"|".join(str(seed).encode() for seed in seeds)
    return re.compile(
        rb"(?i)(?<![A-Za-z0-9])(?:seed(?:s)?|world[_-]?seed|s)"
        rb"[_:= -]*(?:" + options + rb")(?![0-9])"
    )


def _seed_scan(
    rl_root: Path,
    retrieval: Path,
    package: Path,
    output: Path,
    seeds: list[int],
    retrieval_inventory: dict[str, Any],
    t2m_retrieval: Path | None = None,
    t2m_inventory: dict[str, Any] | None = None,
) -> tuple[dict, dict]:
    pattern = _seed_pattern(seeds)
    labelled_pattern = _labelled_seed_pattern(seeds)
    rows = []
    occurrences = []
    candidates = []
    output_relative = None
    try:  # noqa: SIM105 - explicit try/except kept verbatim
        output_relative = output.resolve().relative_to(rl_root.resolve()).as_posix()
    except ValueError:
        pass

    def inspect(
        scope: str,
        relative: str,
        path: Path,
        expected: str | None,
        *,
        excluded: bool = False,
        reason: str | None = None,
    ) -> None:
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if expected is not None and sha != expected:
            _fail(f"seed-scan input changed: {scope}/{relative}")
        row = {
            "scope": scope,
            "path": relative,
            "size": len(raw),
            "sha256": sha,
            "scan": "excluded-proposal-declaration" if excluded else "raw-byte-decimal-token",
        }
        if excluded:
            row["reason"] = reason
        else:
            for match in pattern.finditer(raw):
                line_start = raw.rfind(b"\n", 0, match.start()) + 1
                line_end = raw.find(b"\n", match.end())
                if line_end < 0:
                    line_end = len(raw)
                line = raw[line_start:line_end]
                candidate = {
                    "scope": scope,
                    "path": relative,
                    "seed": int(match.group()),
                    "byte_offset": match.start(),
                }
                candidates.append(candidate)
                # Prior use needs seed-bearing context. Arbitrary numeric
                # output, population values and digest substrings are not seed
                # evidence. The complete candidate list remains reviewable.
                if re.search(rb"(?i)\b(?:seed(?:s)?|world[_-]?seed)\b", line):
                    occurrences.append({**candidate, "evidence": "seed-labelled-context"})
            for match in labelled_pattern.finditer(raw):
                digits = re.search(rb"[0-9]+", match.group())
                assert digits is not None
                candidate = {
                    "scope": scope,
                    "path": relative,
                    "seed": int(digits.group()),
                    "byte_offset": match.start(),
                    "evidence": "seed-labelled-token",
                }
                if candidate not in occurrences:
                    occurrences.append(candidate)
            path_match = labelled_pattern.search(relative.encode())
            if path_match:
                digits = re.search(rb"[0-9]+", path_match.group())
                assert digits is not None
                occurrences.append(
                    {
                        "scope": scope,
                        "path": relative,
                        "seed": int(digits.group()),
                        "location": "path",
                        "evidence": "seed-labelled-path",
                    }
                )
        rows.append(row)

    for relative in _repo_files(rl_root):
        if output_relative and (
            relative == output_relative or relative.startswith(output_relative + "/")
        ):
            continue
        path = rl_root / Path(relative)
        if not path.is_file():
            continue
        generated_proposal = (
            relative.startswith("evidence/model9-gate1-freeze-candidate")
            and "/post-review/" not in relative
        )
        excluded = relative in PROPOSAL_DECLARATIONS or generated_proposal
        inspect(
            "rl-worktree",
            relative,
            path,
            None,
            excluded=excluded,
            reason=(
                "this file is a preregistration/generated HOLD proposal and is not "
                "prior-use evidence"
                if excluded
                else None
            ),
        )
    expected_rows = {row["path"]: row for row in retrieval_inventory["files"]}
    for relative, expected in sorted(expected_rows.items()):
        inspect("vm-retrieval", relative, retrieval / Path(relative), expected["sha256"])
    if t2m_retrieval is not None and t2m_inventory is not None:
        for expected in t2m_inventory["files"]:
            inspect(
                "t2m-retrieval",
                expected["path"],
                t2m_retrieval / Path(expected["path"]),
                expected["sha256"],
            )
    # The corrected evidence package is also prior-use evidence and is distinct
    # from the retrieved byte inventory.
    for path in sorted(package.iterdir()):
        if path.is_file():
            inspect("retrieval-analysis", path.name, path, None)

    forbidden = {seed for values in ROLES.values() for seed in values} | DENIED
    if len(seeds) != FAMILY_SIZE or len(seeds) != len(set(seeds)) or set(seeds) & forbidden:
        _fail(
            "candidate family overlaps a known train/dev/selection/calibration/confirmation family"
        )
    if occurrences:
        _fail(f"candidate family has prior-use occurrences: {occurrences[:8]}")
    files = _seal(
        {
            "schema": 1,
            "kind": "gate1_seed_scan_file_inventory",
            "scanned_files": rows,
            "scanned_file_count": len(rows),
            "decimal_token_candidates": candidates,
            "scope_note": (
                "Every Git tracked/untracked nonignored RL file and every file in the verified "
                "T3 and T2m retrievals plus corrected analysis package was raw-byte scanned; only "
                "the named preregistration declaration files were excluded and remain listed. "
                "All standalone decimal candidates are retained, while rejection requires "
                "seed-labelled schema, source, CLI or path context."
            ),
        }
    )
    audit = _seal(
        {
            "schema": 1,
            "kind": "gate1_evaluation_seed_audit",
            "candidate_family": {
                "id": f"gate1-evaluation-only-{seeds[0]}-{seeds[-1]}",
                "seeds": seeds,
                "selection_basis": "first reviewed candidate supplied to builder",
            },
            "known_excluded_roles": {**ROLES, "confirmation": sorted(DENIED)},
            "prior_use_occurrences": [],
            "status": "fresh-in-scanned-scope",
            "limits": (
                "Freshness is a repository/retrieval prior-use statement, not a simulator-quality "
                "or readiness result. Astra approval is required before any world is reset."
            ),
        }
    )
    return files, audit


def _corpus(
    *, core_width: int = 207, kind: str = "gate1_contract8_corpus", case_prefix: str = "contract8"
) -> dict[str, Any]:
    from .legacy_contract8 import ECONOMY_MIX, SETS

    cases = []
    for case_index, batch in enumerate((1, 1, 2, 3)):
        inputs = {}
        core = np.asarray(
            [
                [((index + 3 * case_index + 5 * row) % 19 - 9) / 4 for index in range(core_width)]
                for row in range(batch)
            ],
            dtype=np.float32,
        )
        present = np.asarray(
            [
                [
                    case_index == 1 or (index + case_index + row) % 7 != 0
                    for index in range(core_width)
                ]
                for row in range(batch)
            ],
            dtype=bool,
        )
        inputs["core"] = {"dtype": "float32", "shape": [batch, core_width], "values": core.tolist()}
        inputs["core_present"] = {
            "dtype": "bool",
            "shape": [batch, core_width],
            "values": present.tolist(),
        }
        for set_index, (name, count, width) in enumerate(SETS):
            values = np.asarray(
                [
                    [
                        [
                            ((row + column + case_index + set_index + sample) % 13 - 6) / 3
                            for column in range(width)
                        ]
                        for row in range(count)
                    ]
                    for sample in range(batch)
                ],
                dtype=np.float32,
            )
            mask = np.asarray(
                [
                    [(row + case_index + set_index + sample) % 3 != 0 for row in range(count)]
                    for sample in range(batch)
                ],
                dtype=bool,
            )
            inputs[name] = {
                "dtype": "float32",
                "shape": [batch, count, width],
                "values": values.tolist(),
            }
            inputs[name + "_present"] = {
                "dtype": "bool",
                "shape": [batch, count],
                "values": mask.tolist(),
            }
        if case_index == 0:
            movement = [[False] * 8 for _ in range(batch)]
            clothing = shelter = [[False, False] for _ in range(batch)]
            mix = [[False] * ECONOMY_MIX for _ in range(batch)]
        elif case_index == 1:
            movement = [[True] * 8 for _ in range(batch)]
            clothing = shelter = [[True, True] for _ in range(batch)]
            mix = [[True] * ECONOMY_MIX for _ in range(batch)]
        else:
            movement = [
                [slot == (sample + case_index) % 8 for slot in range(8)] for sample in range(batch)
            ]
            clothing = [
                [(sample + route + case_index) % 2 == 0 for route in range(2)]
                for sample in range(batch)
            ]
            shelter = [
                [(sample + route + case_index) % 2 != 0 for route in range(2)]
                for sample in range(batch)
            ]
            mix = [
                [slot == (sample + case_index) % ECONOMY_MIX for slot in range(ECONOMY_MIX)]
                for sample in range(batch)
            ]
        inputs["movement_mask"] = {"dtype": "bool", "shape": [batch, 8], "values": movement}
        inputs["clothing_route_mask"] = {"dtype": "bool", "shape": [batch, 2], "values": clothing}
        inputs["shelter_route_mask"] = {"dtype": "bool", "shape": [batch, 2], "values": shelter}
        inputs["economy_mix_mask"] = {"dtype": "bool", "shape": [batch, ECONOMY_MIX], "values": mix}
        cases.append({"id": f"{case_prefix}-case-{case_index}", "inputs": inputs})
    return _seal(
        {
            "schema": 1,
            "kind": kind,
            "cases": cases,
            "purpose": "pinned historical-source versus diagnostic-adapter inference",
        }
    )


def _semantics(*, experts: bool, token: bool) -> dict[str, Any]:
    return {
        "experts": experts,
        "token": token,
        "commitments": False,
        "learned_movement": False,
        "region_movement": False,
        "learned_fission": False,
        "learned_trade": False,
        "region_head": "flat",
        "grouped_region": False,
        "region_selection": None,
    }


def _worlds(seeds: list[int]) -> tuple[list[dict[str, Any]], str]:
    rows = []
    for seed in seeds:
        for name, withheld in MASKS:
            rows.append(
                {
                    "id": f"s{seed}-{name}",
                    "family": f"gate1-evaluation-only-{seeds[0]}-{seeds[-1]}",
                    "seed": seed,
                    "regime": name,
                    "withheld": withheld,
                    "snapshot": {"kind": "none", "sha256": None},
                    "reset_turn": None,
                    "turn": None,
                    "reset_turn_policy": "bind-native-post-reset-boundary",
                    "warmup": 0,
                    "horizon": 25,
                    "players": 8,
                    "early_stop": "typed-terminal-only",
                }
            )
    cohort = digest(rows)
    return [{**row, "cohort_digest": cohort} for row in rows], cohort


def _rng_schedule(nodes: list[dict[str, Any]], worlds: list[dict[str, Any]]) -> dict[str, Any]:
    from .gate1_diagnostics import draw_order_from_semantics, generator_state_sha256

    rows = []
    for world in worlds:
        executions = []
        for node in nodes:
            draw_order = draw_order_from_semantics(node["historical_semantics"])
            steps = []
            for step in range(world["horizon"]):
                seed = int.from_bytes(
                    hashlib.sha256(
                        f"gate1-policy-rng-v2|{world['id']}|{step}".encode(),
                    ).digest()[:8],
                    "big",
                ) & ((1 << 63) - 1)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                steps.append(
                    {"step": step, "seed": seed, "state_sha256": generator_state_sha256(generator)}
                )
            executions.append(
                {
                    "node_id": node["id"],
                    "draw_order": draw_order,
                    "initial_state_coupling": "same-world-and-step-across-nodes",
                    "post_draw_coupling_claim": False,
                    "steps": steps,
                }
            )
        rows.append({"world_id": world["id"], "nodes": executions})
    return _seal(
        {
            "schema": 1,
            "kind": "gate1_policy_rng_schedule",
            "algorithm": "torch.Generator.cpu-default",
            "torch_version": torch.__version__,
            "device": "cpu",
            "coupling_scope": (
                "identical initial generator state per world/step; no claim "
                "after node-specific masks, camp counts or draw orders diverge"
            ),
            "worlds": rows,
        }
    )


def _runtime() -> dict[str, Any]:
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "missing"

    # Preserve the invoked venv command path. Resolving this symlink on Linux
    # would put the base uv-managed interpreter into every generated command
    # and silently drop the venv site-packages. Bind the resolved binary as a
    # separate identity instead.
    executable = Path(sys.executable).absolute()
    resolved_executable = executable.resolve()
    dotnet_name = shutil.which("dotnet")
    if dotnet_name is None:
        _fail("dotnet runtime is missing")
    dotnet = Path(dotnet_name).resolve()
    dotnet_info = subprocess.check_output(
        [str(dotnet), "--info"],
        text=True,
        encoding="utf-8",
    )
    host_match = re.search(r"(?ms)^Host:\s*\n\s*Version:\s*([^\s]+)", dotnet_info)
    if host_match is None:
        _fail("dotnet host version could not be identified")
    installed_runtimes = (
        subprocess.check_output(
            [str(dotnet), "--list-runtimes"],
            text=True,
            encoding="utf-8",
        )
        .strip()
        .splitlines()
    )
    if not installed_runtimes:
        _fail("dotnet has no installed runtimes")
    return {
        "python": {
            "path": str(executable),
            "sha256": checkpoint_hash(executable),
            "resolved_path": str(resolved_executable),
            "resolved_sha256": checkpoint_hash(resolved_executable),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "version": sys.version,
        },
        "dotnet": {
            "path": str(dotnet),
            "sha256": checkpoint_hash(dotnet),
            "version": host_match.group(1),
            "installed_runtimes": installed_runtimes,
        },
        "packages": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pytest": version("pytest"),
            "ruff": version("ruff"),
            "coverage": version("coverage"),
        },
        "platform": sys.platform,
    }


def _implementation_identity(
    rl_root: Path, generated_output: Path, portable_identity: dict[str, Any] | None = None
) -> dict[str, Any]:
    files = []
    for relative in IMPLEMENTATION_SOURCES:
        path = rl_root / relative
        if not path.is_file():
            _fail(f"implementation source is missing: {relative}")
        files.append(
            {"path": relative, "sha256": checkpoint_hash(path), "size": path.stat().st_size}
        )
    if portable_identity is None:
        status = _git(rl_root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
        commit = _git(rl_root, "rev-parse", "HEAD")
        materialization = "source-worktree"
    else:
        expected = portable_identity.get("source_files")
        if not isinstance(expected, list) or expected != files:
            _fail("VM materialization source bytes differ from the portable reviewed inventory")
        status = portable_identity.get("full_repository_dirty_state")
        commit = portable_identity.get("commit")
        if not isinstance(status, list) or not isinstance(commit, str) or not commit:
            _fail("portable implementation identity lacks commit/dirty-state evidence")
        materialization = "portable-source-manifest-verified-without-git-history"
    try:
        generated_prefix = (
            generated_output.resolve().relative_to(rl_root.resolve()).as_posix() + "/"
        )
    except ValueError:
        generated_prefix = ""
    status = [
        row
        for row in status
        if not generated_prefix or not row[3:].replace("\\", "/").startswith(generated_prefix)
    ]
    by_path = {row["path"]: row for row in files}
    execution_files = []
    for relative in EXECUTION_SOURCES:
        path = rl_root / relative
        if not path.is_file():
            _fail(f"executing source is missing: {relative}")
        execution_files.append(
            {"path": relative, "sha256": checkpoint_hash(path), "size": path.stat().st_size}
        )
    # Where a source also belongs to the broader review set, the two identities
    # must literally agree rather than becoming independent assertions.
    if any(
        relative in by_path and by_path[relative] != row
        for relative, row in ((entry["path"], entry) for entry in execution_files)
    ):
        _fail("implementation/execution source identity is inconsistent")
    return _seal(
        {
            "schema": 1,
            "kind": "gate1_freeze_implementation_identity",
            "commit": commit,
            "source_files": files,
            "repository_root": str(rl_root),
            "execution_source_files": execution_files,
            "full_repository_dirty_state": status,
            "generated_output_exclusion": generated_prefix or None,
            "materialization": materialization,
            "note": (
                "Exact executing source hashes and the repository dirty-state proof are bound. "
                "The generated package excludes only its own output path; cleanliness is "
                "not a gate."
            ),
        }
    )


def _bridge_binding() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from .bridge import DEFAULT_BRIDGE

    bridge = DEFAULT_BRIDGE.resolve()
    if not bridge.is_file():
        _fail("current diagnostic bridge binary is missing")
    dependencies = []
    for path in (
        bridge,
        bridge.with_suffix(".deps.json"),
        bridge.with_suffix(".runtimeconfig.json"),
    ):
        if not path.is_file():
            _fail(f"current diagnostic bridge dependency is missing: {path.name}")
        dependencies.append(_file_identity(path))
    return dependencies[0], dependencies


def _artifact_record(retrieval: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = retrieval / Path(record["path"])
    if checkpoint_hash(path) != record["sha256"] or path.stat().st_size != record["size"]:
        _fail(f"checkpoint changed: {record['path']}")
    return {
        "root_id": "t3",
        "path": record["path"],
        "sha256": record["sha256"],
        "size": record["size"],
        "top_level_keys": record["top_level_keys"],
        "metadata": record["metadata"],
        "components": record["components"],
    }


def _build_dag(
    retrieval: Path,
    evidence: dict[str, Any],
    equivalence_ref: dict[str, Any],
    bc_adapter_ref: dict[str, Any],
    guarded_artifact: dict[str, Any],
    guarded_adapter_ref: dict[str, Any],
    guard_only_ref: dict[str, Any],
    recovered_native_ref: dict[str, Any],
    expert_expansion_ref: dict[str, Any],
) -> dict[str, Any]:
    records = {row["path"]: row for row in evidence["checkpoints"]["checkpoints"]}
    lineage = evidence["lineage"]
    nodes = []
    prior = _artifact_record(retrieval, records["producer/inputs/stage1-seed2.pt"])
    bc = _artifact_record(retrieval, records["t2/stage1-token-s6.bc.pt"])
    initialization = _artifact_record(retrieval, records["stage1-token-s6.pt"])
    nodes.extend(
        [
            {
                "id": "prior-reference",
                "role": "evaluation-reference",
                "status": "present",
                "artifact": prior,
                "contract": 8,
                "historical_semantics": _semantics(experts=False, token=False),
                "historical_producer_status": "unresolved",
                "run_identity": {
                    "run_id": "stage1-seed2",
                    "arm_id": None,
                    "optimizer_seed": None,
                    "pairing_status": "not-applicable",
                    "pairing_group": None,
                },
                "loading": {"mode": "reviewed-adapter", "adapter": "pinned-contract-8"},
            },
            {
                "id": "bc-phase",
                "role": "learned-bc-phase",
                "status": "present",
                "artifact": bc,
                "contract": "actor-only; current diagnostic composite uses unguarded T3 "
                "initialization "
                "support after identity proof",
                "historical_semantics": _semantics(experts=False, token=True),
                "historical_producer_status": "unresolved",
                "run_identity": {
                    "run_id": "stage1-token-s6-bc",
                    "arm_id": None,
                    "optimizer_seed": None,
                    "pairing_status": "not-applicable",
                    "pairing_group": None,
                },
                "loading": {
                    "mode": "reviewed-adapter",
                    "adapter": "t3-initialization-bc-composite",
                    "protocol": bc_adapter_ref,
                    "support_checkpoint": initialization,
                },
            },
            {
                "id": "t3-initialization-clone",
                "role": "t3-initialization-clone",
                "status": "present",
                "artifact": initialization,
                "normalizer_semantics": "unguarded",
                "contract": 12,
                "historical_semantics": _semantics(experts=False, token=True),
                "historical_producer_status": "unresolved",
                "run_identity": {
                    "run_id": "stage1-token-s6",
                    "arm_id": None,
                    "optimizer_seed": None,
                    "pairing_status": "not-applicable",
                    "pairing_group": None,
                },
                "loading": {
                    "mode": "reviewed-adapter",
                    "adapter": "recovered-contract12-inactive-region",
                    "protocol": recovered_native_ref,
                    "node_id": "t3-initialization-clone",
                },
            },
            {
                "id": "t2m-guarded-bc-seed6",
                "role": "guarded-bc-actor-phase",
                "status": "present",
                "artifact": guarded_artifact,
                "normalizer_semantics": "guarded-10.0",
                "contract": "actor-only BC phase; no critic/completed-clone claim",
                "historical_semantics": _semantics(experts=False, token=True),
                "historical_producer_status": "unresolved",
                "run_identity": {
                    "run_id": "t2m-stage1-token-s6-bc",
                    "arm_id": None,
                    "optimizer_seed": 6,
                    "pairing_status": "not-applicable",
                    "pairing_group": None,
                },
                "loading": {
                    "mode": "reviewed-adapter",
                    "adapter": "t2m-guarded-bc-actor-only",
                    "protocol": guarded_adapter_ref,
                },
                "critic_status": "absent-in-actor-only-bc-phase",
                "selection_basis": "seed identity 6 matches historical T3 initialization; no "
                "outcome "
                "selection",
            },
            {
                "id": "t3-init-guard-only-intervention",
                "role": "prospective-normalizer-intervention",
                "status": "present",
                "artifact": initialization,
                "normalizer_semantics": "t2m-seed6-guard-dead-only",
                "contract": 12,
                "historical_semantics": _semantics(experts=False, token=True),
                "historical_producer_status": "unresolved",
                "run_identity": {
                    "run_id": None,
                    "arm_id": None,
                    "optimizer_seed": None,
                    "pairing_status": "not-applicable",
                    "pairing_group": None,
                },
                "loading": {
                    "mode": "reviewed-adapter",
                    "adapter": "t3-init-guard-only-intervention",
                    "protocol": guard_only_ref,
                    "guard_source_checkpoint": guarded_artifact,
                },
                "composition_scope": "prospective-current-diagnostic-only",
                "historical_artifact_claim": False,
                "historical_ancestry_claim": False,
                "selection_basis": (
                    "pre-outcome cloning-stability diagnosis plus exact affine-moment "
                    "identity; no fresh outcome selection"
                ),
            },
        ]
    )
    branch_rows = []
    for branch in lineage["t3_branches"]:
        arm, seed = branch["arm"], branch["optimizer_seed"]
        prefix = "t2lo" if arm == "T3lo" else "t2"
        latest_path = f"{prefix}/{arm}-s{seed}/latest.pt"
        best_path = f"{prefix}/{arm}-s{seed}/best.pt"
        latest = _artifact_record(retrieval, records[latest_path])
        best = _artifact_record(retrieval, records[best_path])
        node_id = f"{arm.lower()}-s{seed}"
        nodes.append(
            {
                "id": node_id,
                "role": "trained-t3",
                "status": "present",
                "artifact": latest,
                "selection_evidence_only": best,
                "contract": 12,
                "historical_semantics": _semantics(experts=True, token=True),
                "historical_producer_status": "unresolved",
                "run_identity": {
                    "run_id": branch["run_id"],
                    "arm_id": arm,
                    "optimizer_seed": seed,
                    "pairing_status": "unpaired",
                    "pairing_group": None,
                },
                "loading": {
                    "mode": "reviewed-adapter",
                    "adapter": "recovered-contract12-inactive-region",
                    "protocol": recovered_native_ref,
                    "node_id": node_id,
                },
                "critic_treatment": {
                    "vector": branch["vector_critic"],
                    "critic_only_warmup_iterations": branch["critic_only_warmup_iterations"],
                    "actor_update_iterations": branch["actor_update_iterations"],
                },
            }
        )
        branch_rows.append((node_id, arm, seed))
    edges = [
        {
            "id": "prior-reference-to-initialization",
            "source": "prior-reference",
            "destination": "t3-initialization-clone",
            "relation": "evaluation_reference",
            "equivalence": None,
            "interpretation": "descriptive boundary, not parameter/data lineage",
        },
        {
            "id": "bc-phase-to-initialization",
            "source": "bc-phase",
            "destination": "t3-initialization-clone",
            "relation": "critic_initialization",
            "equivalence": equivalence_ref,
            "interpretation": "actor tensors only; no behavioral-equivalence claim",
        },
        {
            "id": "t3-initialization-to-guard-only",
            "source": "t3-initialization-clone",
            "destination": "t3-init-guard-only-intervention",
            "relation": "normalizer_change",
            "equivalence": None,
            "interpretation": (
                "prospective guard/dead-only intervention; actor and affine moments "
                "are exact, behavior equality is not expected or claimed"
            ),
        },
    ]
    for node_id, arm, seed in branch_rows:
        edges.append(
            {
                "id": f"initialization-to-{node_id}",
                "source": "t3-initialization-clone",
                "destination": node_id,
                "status": "present",
                "relation": "ppo_continuation",
                "equivalence": None,
                "parameter_expansion_substep": {
                    "kind": "reconstructed-experts-architecture-expansion",
                    "source_semantics": "experts=false",
                    "destination_entry_semantics": "experts=true",
                    "equivalence_evidence": expert_expansion_ref,
                    "equivalence_scope": "frozen teacher-state/mask corpus only",
                    "rollout_node_added": False,
                },
                "interpretation": (
                    "bundled historical boundary: reconstructed zero-output "
                    "expert parameter expansion followed by PPO training; "
                    "not a pure-gradient or pure-PPO causal effect unless the "
                    "separate expansion audit is accepted"
                ),
                "branch": {"arm_id": arm, "optimizer_seed": seed, "pairing_status": "unpaired"},
            }
        )
    return _seal(
        {
            "schema": 1,
            "kind": "gate1_artifact_dag_proposal",
            "review_state": "awaiting-astra-high",
            "nodes": nodes,
            "edges": edges,
            "user_slots": {
                "prior-reference": ["prior-reference"],
                "conversion": {
                    "status": "not-applicable",
                    "support": [
                        "producer/t2.sh",
                        "producer/src/anthropocene_rl/autonomous_v7/train.py",
                    ],
                    "reason": "recovered code constructs a fresh token actor and fits BC; no prior "
                    "conversion artifact",
                },
                "learned-bc-phase": ["bc-phase"],
                "t3-initialization-clone": ["t3-initialization-clone"],
                "guarded-clone": {
                    "status": "actor-phase-present-completed-critic-clone-unresolved",
                    "node_ids": ["t2m-guarded-bc-seed6"],
                    "reason": (
                        "guarded actor and original guarded normalizer are recovered; "
                        "critic fitting did not yield a recovered completed clone"
                    ),
                    "interpretation": (
                        "actor-only fresh evaluation is supported; this node is "
                        "not a T3 ancestor or a completed critic checkpoint"
                    ),
                },
                "prospective-normalizer-intervention": {
                    "status": "present-diagnostic-composition",
                    "node_ids": ["t3-init-guard-only-intervention"],
                    "reason": (
                        "isolates T2m seed-6 guard/dead semantics on the "
                        "bitwise T3 initialization actor and identical affine moments"
                    ),
                    "historical_artifact_claim": False,
                },
                "trained-t3": [row[0] for row in branch_rows],
            },
            "clone_fit": {
                "parameter_parent": "fresh token actor constructed by recovered train.py",
                "dataset_arguments": ["~/t1/teacher/train", "~/t1/teacher/dev"],
                "data_provenance": (
                    "recovered teacher arrays have a hash-verified 7,920-member "
                    "inventory, but their exact linkage to the original BC fit "
                    "and BC-time normalizer remains unproved"
                ),
                "epochs": 12,
                "historical_conversion_claim": False,
            },
            "best_checkpoint_policy": "selection-evidence-only; never substitutes for latest "
            "endpoint",
            "lineage_conflict": (
                "Recovered T3 scripts/checkpoints expose an unguarded stage1-token-s6 "
                "initialization. The independent T2m seed-6 BC phase supplies a guarded actor "
                "and original normalizer for fresh evaluation, but is not a T3 ancestor and "
                "does not supply a completed critic checkpoint. The guard-only node is an "
                "explicit prospective composition, not a recovered historical artifact."
            ),
            "ppo_entry_expansion_status": (
                "behavior-preserving on the bound frozen corpus; "
                "closed-loop effect remains outside this equivalence"
            ),
            "optimizer_uncertainty": "15 disjoint unpaired runs; no cross-arm pairing claim",
        }
    )


def _verification_protocol(runtime: dict[str, Any], worlds: list[dict[str, Any]]) -> dict[str, Any]:
    python = runtime["python"]["path"]
    tests = [
        "tests/test_initialization_lineage.py",
        "tests/test_gate1_preflight.py",
        "tests/test_gate1_diagnostics.py",
        "tests/test_gate1_legacy_contract8.py",
        "tests/test_gate1_freeze_candidate.py",
        "tests/test_autonomous_v7_worker_contracts.py",
        "tests/test_autonomous_v7_ppo.py",
        "tests/test_autonomous_v7_bridge.py",
        "tests/test_economic_trace.py",
        "tests/test_economic_diagnostics.py",
        "tests/test_autonomous_v7_round3_contracts.py",
        "tests/test_autonomous_v7_round3_protocol.py",
    ]
    profiles = [
        {
            "id": "diagnostic-off-current",
            "fixture": worlds[0],
            "workers": 1,
            "metric": "throughput-ratio",
            "baseline": "same binary/fixture with pre-Gate1 rollout path",
            "treatment": "same binary/fixture with diagnostic capture disabled",
            "ratio_definition": "treatment completed-steps-per-second / paired baseline "
            "completed-steps-per-second",
            "rule": {"direction": ">=", "limit": 0.98},
        },
        {
            "id": "diagnostic-on-current",
            "fixture": worlds[0],
            "workers": 1,
            "metric": "step-time-ratio",
            "baseline": "same binary/fixture with diagnostic capture disabled",
            "treatment": "same binary/fixture with full Gate1 diagnostic capture enabled",
            "ratio_definition": "treatment median step time / paired baseline median step time",
            "rule": {"direction": "<=", "limit": 1.15},
        },
        {
            "id": "diagnostic-on-mature",
            "fixture": {**worlds[4], "warmup": 25, "fixture_id": "mature-reset-plus-25-warmups"},
            "workers": 1,
            "metric": "step-time-ratio",
            "baseline": "same mature fixture with diagnostic capture disabled",
            "treatment": "same mature fixture with full Gate1 diagnostic capture enabled",
            "ratio_definition": "treatment median step time / paired baseline median step time",
            "rule": {"direction": "<=", "limit": 1.20},
        },
        {
            "id": "eight-worker-end-to-end",
            "fixture": {
                "world_ids": [row["id"] for row in worlds if row["regime"] == "default"][:8]
            },
            "workers": 8,
            "metric": "throughput-ratio",
            "baseline": "same eight worlds, eight workers and binary with diagnostic capture "
            "disabled",
            "treatment": "same eight worlds, eight workers and binary with full Gate1 diagnostic "
            "capture enabled",
            "ratio_definition": "treatment completed-worlds-per-second / paired baseline "
            "completed-worlds-per-second",
            "scaling_efficiency_claim": False,
            "rule": {"direction": ">=", "limit": 0.80},
        },
    ]
    for profile in profiles:
        profile.update(
            {
                "warmup_repetitions": 2,
                "measured_repetitions": 7,
                "pair_order": [
                    "baseline-treatment" if i % 2 == 0 else "treatment-baseline" for i in range(7)
                ],
                "aggregation": "median-of-per-pair-ratios",
                "memory": {
                    "method": "paired-incremental-peak-worker-rss",
                    "aggregation": "maximum-of-per-pair-deltas",
                    "unit": "MiB",
                    "limit": 256,
                },
            }
        )
    return _seal(
        {
            "schema": 1,
            "kind": "gate1_verification_protocol_proposal",
            "review_state": "awaiting-astra-high",
            "runtime": runtime,
            "test_commands": [[python, "-m", "pytest", "-q", path] for path in tests],
            "coverage": {
                "tool": "coverage.py",
                "version": runtime["packages"]["coverage"],
                "scope": [
                    "src/anthropocene_rl/autonomous_v7/initialization_lineage.py",
                    "src/anthropocene_rl/autonomous_v7/evaluate.py",
                    "src/anthropocene_rl/autonomous_v7/gate1_diagnostics.py",
                    "src/anthropocene_rl/autonomous_v7/gate1_preflight.py",
                    "src/anthropocene_rl/autonomous_v7/legacy_contract8.py",
                    "src/anthropocene_rl/autonomous_v7/gate1_production_adapter.py",
                    "src/anthropocene_rl/autonomous_v7/gate1_freeze_candidate.py",
                ],
                "line_minimum_percent": 90.0,
                "branch_minimum_percent": 85.0,
                "exclusions": [],
                "raw_reports_required": True,
            },
            "diagnostic_off_parity": {
                "required": True,
                "surface": ["observations", "actions", "rewards", "trajectory-fingerprints"],
                "equality": "literal-bytes",
            },
            "performance": {
                "acceptance": "relative-paired-only",
                "baseline_calibration": None,
                "profiles": profiles,
                "treatment_measurements_before_review": "forbidden",
            },
        }
    )


def _execution_command_artifact(
    runtime: dict[str, Any], output: Path, evaluation: dict[str, Any], *, execution_tier: str
) -> dict[str, Any]:
    python = runtime["python"]["path"]
    candidate_path = output / "freeze-candidate.json"
    review_path = output / "astra-review.json"
    worker_path = output / "post-review/current-worker.json"
    comparator_result_path = output / "post-review/contract8-comparator-result.json"
    guarded_result_path = output / "post-review/guarded-bc-comparator-result.json"
    evaluation_commands = []
    for pane in evaluation["panes"]:
        for node_id in pane["nodes"]:
            destination = output / "post-review/evaluations" / f"{node_id}--{pane['id']}.json"
            evaluation_commands.append(
                {
                    "node": node_id,
                    "pane": pane["id"],
                    "out": str(destination),
                    "command": [
                        python,
                        "-m",
                        "anthropocene_rl.autonomous_v7.gate1_freeze_candidate",
                        "evaluate-reviewed",
                        "--candidate",
                        str(candidate_path),
                        "--review",
                        str(review_path),
                        "--worker",
                        str(worker_path),
                        "--contract8-result",
                        str(comparator_result_path),
                        "--guarded-bc-result",
                        str(guarded_result_path),
                        "--node",
                        node_id,
                        "--pane",
                        pane["id"],
                        "--out",
                        str(destination),
                    ],
                }
            )
    return _seal(
        {
            "schema": 1,
            "kind": "gate1_post_review_execution_commands",
            "status": "blocked-pending-astra-review",
            "review_path": str(review_path),
            "execution_tier": execution_tier,
            "collection_execution_policy": {
                "pre_calibration_max_concurrent_node_pane_processes": 1,
                "proposed_post_calibration_max_concurrent_node_pane_processes": 8,
                "post_calibration_authority": (
                    "requires accepted paired incremental-RSS and throughput "
                    "receipts on the exact VM runtime; the proposal is not authority"
                ),
                "world_retention": "write one sealed world artifact then release it",
                "maximum_in_memory_world_results_per_process": 1,
                "reason": (
                    "streaming bounds retained result count, but concurrent Python/native RSS must "
                    "be measured before increasing from one to the proposed eight processes"
                ),
            },
            "commands": {
                "validate_candidate": [
                    python,
                    "-m",
                    "anthropocene_rl.autonomous_v7.gate1_freeze_candidate",
                    "validate",
                    "--candidate",
                    str(candidate_path),
                ],
                "contract8_comparator": [
                    python,
                    "-m",
                    "anthropocene_rl.autonomous_v7.gate1_freeze_candidate",
                    "compare-contract8",
                    "--candidate",
                    str(candidate_path),
                    "--review",
                    str(review_path),
                    "--out",
                    str(comparator_result_path),
                ],
                "guarded_bc_comparator": [
                    python,
                    "-m",
                    "anthropocene_rl.autonomous_v7.gate1_freeze_candidate",
                    "compare-guarded-bc",
                    "--candidate",
                    str(candidate_path),
                    "--review",
                    str(review_path),
                    "--out",
                    str(guarded_result_path),
                ],
                "capture_current_worker": [
                    python,
                    "-m",
                    "anthropocene_rl.autonomous_v7.gate1_freeze_candidate",
                    "capture-current-worker",
                    "--candidate",
                    str(candidate_path),
                    "--review",
                    str(review_path),
                    "--out",
                    str(worker_path),
                ],
                "checkpoint_evaluations": evaluation_commands,
            },
            "pre_review_prohibitions": [
                "checkpoint evaluation",
                "simulator reset",
                "performance treatment measurement",
                "training",
                "selection",
                "confirmation seed use",
            ],
        }
    )


def build_candidate(
    rl_root: Path,
    retrieval: Path,
    t2m_retrieval: Path,
    evidence_package: Path,
    evidence_review: Path,
    expert_expansion_evidence: Path,
    output: Path,
    seed_start: int,
) -> Path:
    rl_root = rl_root.resolve()
    retrieval = retrieval.resolve()
    t2m_retrieval = t2m_retrieval.resolve()
    evidence_package = evidence_package.resolve()
    evidence_review = evidence_review.resolve()
    expert_expansion_evidence = expert_expansion_evidence.resolve()
    output = output.resolve()
    current_bridge_record, bridge_dependencies = _bridge_binding()
    evidence = _evidence_package(evidence_package, retrieval, evidence_review)
    t2m_inventory = _t2m_inventory(t2m_retrieval)
    seeds = list(range(seed_start, seed_start + FAMILY_SIZE))
    scan_retrieval = json.loads(json.dumps(evidence["retrieval"]))
    scan_rows = {row["path"]: row for row in scan_retrieval["files"]}
    for observed in evidence["runtime_generated_observed"]:
        scan_rows[observed["path"]].update(
            {
                "sha256": observed["sha256"],
                "size": observed["size"],
            }
        )
    scan_files, seed_audit = _seed_scan(
        rl_root,
        retrieval,
        evidence_package,
        output,
        seeds,
        scan_retrieval,
        t2m_retrieval,
        t2m_inventory,
    )
    output.mkdir(parents=True, exist_ok=True)
    evidence_binding_path = _copy_evidence_binding(evidence, output)
    expert_expansion_path = _copy_expert_expansion_evidence(
        expert_expansion_evidence,
        retrieval,
        output,
    )
    t2m_inventory_path = output / "t2m-retrieval-inventory.json"
    _write_json(t2m_inventory_path, t2m_inventory)
    scan_path = output / "seed-scan-files.json"
    _write_json(scan_path, scan_files)
    seed_audit["scan_file_inventory"] = _ref(scan_path, output)
    seed_audit = _seal({key: value for key, value in seed_audit.items() if key != "digest"})
    seed_path = output / "seed-use-audit.json"
    _write_json(seed_path, seed_audit)

    historical_root = output / "historical-source"
    historical_root.mkdir(parents=True, exist_ok=True)
    for leaf in ("model.py", "contracts.py"):
        source = _git_bytes(
            rl_root, PINNED_CONTRACT8_COMMIT, f"src/anthropocene_rl/autonomous_v7/{leaf}"
        )
        (historical_root / leaf).write_bytes(source)
    corpus_path = output / "contract8-corpus.json"
    _write_json(corpus_path, _corpus())

    guarded_source_root = output / "guarded-historical-source"
    guarded_source_root.mkdir(parents=True, exist_ok=True)
    t2m_source_root = t2m_retrieval / "t2m/src/anthropocene_rl/autonomous_v7"
    for leaf in ("model.py", "contracts.py"):
        shutil.copyfile(t2m_source_root / leaf, guarded_source_root / leaf)
    guarded_checkpoint = t2m_retrieval / "t2m/runs/stage1-token-s6.bc.pt"
    guarded_state = torch.load(guarded_checkpoint, map_location="cpu", weights_only=True)
    guarded_corpus_path = output / "guarded-bc-corpus.json"
    _write_json(
        guarded_corpus_path,
        _corpus(
            core_width=len(guarded_state["normalizer"]["core_mean"]),
            kind="gate1_guarded_bc_corpus",
            case_prefix="guarded-bc",
        ),
    )
    from .model import Actor

    adapter_actor = Actor(
        len(guarded_state["normalizer"]["core_mean"]), token=True, region_head="flat"
    )
    adapter_state = adapter_actor.state_dict()
    appended = tensor_state_identity(
        {
            "region_flat": torch.tensor(True),
            "region_group.weight": torch.zeros_like(adapter_state["region_group.weight"]),
            "region_group.bias": torch.zeros_like(adapter_state["region_group.bias"]),
        }
    )
    guarded_protocol = _seal(
        {
            "schema": 1,
            "kind": "gate1_guarded_bc_adapter_protocol",
            "review_state": "awaiting-astra-high",
            "checkpoint_sha256": checkpoint_hash(guarded_checkpoint),
            "checkpoint_size": guarded_checkpoint.stat().st_size,
            "core_width": len(guarded_state["normalizer"]["core_mean"]),
            "bc_config": guarded_state["bc_config"],
            "bc_epochs": 12,
            "critic_status": "absent-in-actor-only-bc-phase",
            "historical_source_root": "guarded-historical-source",
            "historical_sources": {
                leaf: checkpoint_hash(guarded_source_root / leaf)
                for leaf in ("model.py", "contracts.py")
            },
            "runtime": comparator_runtime_identity(),
            "corpus_sha256": checkpoint_hash(guarded_corpus_path),
            "mapped_outputs": list(GUARDED_BC_OUTPUTS),
            "decoded_actions": list(GUARDED_BC_ACTIONS),
            "appended_inactive_region_state": appended,
            "appended_region_scope": "inactive-under-learned_movement=false",
            "current_upgrade_allowed": False,
            "historical_t3_ancestry_claim": False,
            "completed_critic_checkpoint_claim": False,
        }
    )
    guarded_protocol_path = output / "guarded-bc-adapter-protocol.json"
    _write_json(guarded_protocol_path, guarded_protocol)
    guarded_artifact = {
        "root_id": "t2m",
        "path": "t2m/runs/stage1-token-s6.bc.pt",
        "sha256": checkpoint_hash(guarded_checkpoint),
        "size": guarded_checkpoint.stat().st_size,
        "top_level_keys": sorted(guarded_state),
        "metadata": {
            "seed": 6,
            "core_width": len(guarded_state["normalizer"]["core_mean"]),
            "guard": guarded_state["normalizer"]["guard"],
            "bc_epochs": 12,
        },
        "components": {
            "actor": tensor_state_identity(guarded_state["actor"]),
            "normalizer_keys": sorted(guarded_state["normalizer"]),
        },
    }

    records = {row["path"]: row for row in evidence["checkpoints"]["checkpoints"]}
    prior = records["producer/inputs/stage1-seed2.pt"]
    bc = records["t2/stage1-token-s6.bc.pt"]
    initialization = records["stage1-token-s6.pt"]
    adapter_source = rl_root / "src/anthropocene_rl/autonomous_v7/legacy_contract8.py"
    comparator = _seal(
        {
            "schema": 1,
            "kind": "gate1_contract8_comparator_protocol",
            "review_state": "awaiting-astra-high",
            "checkpoint_sha256": prior["sha256"],
            "source_contract": 8,
            "historical_source_commit": PINNED_CONTRACT8_COMMIT,
            "historical_source_root": "historical-source",
            "historical_sources": {
                leaf: checkpoint_hash(historical_root / leaf)
                for leaf in ("model.py", "contracts.py")
            },
            "adapter_source": {
                "repository_path": adapter_source.relative_to(rl_root).as_posix(),
                "sha256": checkpoint_hash(adapter_source),
            },
            "runtime": comparator_runtime_identity(),
            "corpus_sha256": checkpoint_hash(corpus_path),
            "mapped_outputs": list(CONTRACT8_OUTPUTS),
            "decoded_actions": list(CONTRACT8_ACTIONS),
            "numeric_rule": {
                "dtype": "float32",
                "absolute_tolerance": 0.0,
                "relative_tolerance": 0.0,
            },
            "current_upgrade_allowed": False,
            "result": "not-executed-pending-review",
        }
    )
    comparator_path = output / "contract8-comparator-protocol.json"
    _write_json(comparator_path, comparator)

    bc_state = torch.load(retrieval / bc["path"], map_location="cpu", weights_only=True)
    initialization_state = torch.load(
        retrieval / initialization["path"],
        map_location="cpu",
        weights_only=True,
    )
    bc_actor = tensor_state_identity(bc_state["actor"])
    initialization_actor = tensor_state_identity(initialization_state["actor"])
    normalizer_state = initialization_state.get("normalizer")
    if (
        not isinstance(normalizer_state, dict)
        or "guard" in normalizer_state
        or any(name.endswith("_dead") for name in normalizer_state)
    ):
        _fail("stage1-token-s6 is not the proved unguarded T3 initialization checkpoint")
    if bc_actor != initialization_actor:
        _fail("recovered BC and T3 initialization actors are not literally identical")
    bc_protocol = _seal(
        {
            "schema": 1,
            "kind": "gate1_bc_phase_adapter_protocol",
            "review_state": "awaiting-astra-high",
            "bc_checkpoint_sha256": bc["sha256"],
            "support_checkpoint_sha256": initialization["sha256"],
            "bc_actor_identity": bc_actor,
            "support_actor_identity": initialization_actor,
            "metadata_and_normalizer_source": (
                "hash-bound unguarded T3 initialization checkpoint for current diagnostic composite"
            ),
            "support_normalizer_semantics": "unguarded",
            "support_proof": [
                "recovered .bc.pt contains actor/log only",
                "BC and T3 initialization actor tensors are literally identical",
                "support normalizer has no guard/dead fields",
                "recovered teacher arrays are not proved to be the exact BC-fit input",
                "the original BC-time fitted-normalizer identity is unproved",
            ],
            "composite_scope": "current-diagnostic-only",
            "appended_inactive_region_state": inactive_region_state(
                int(initialization_state["core_width"]),
                experts=False,
                token=True,
            )["identity"],
            "appended_region_scope": "inactive-under-learned_movement=false",
            "historical_bc_normalizer_claim": False,
            "missing_guarded_clone_claim": False,
            "implicit_defaults": [],
            "equivalence_scope": "actor-tensors-only",
            "behavioral_equivalence_claim": False,
            "current_upgrade_allowed": False,
        }
    )
    bc_path = output / "bc-phase-adapter-protocol.json"
    _write_json(bc_path, bc_protocol)
    equivalence = _seal(
        {
            "schema": 1,
            "kind": "gate1_exact_tensor_equivalence_proposal",
            "review_state": "awaiting-astra-high",
            "edge_id": "bc-phase-to-initialization",
            "kind_name": "exact_tensor_preservation",
            "scope": "actor-tensors-only",
            "source_checkpoint_sha256": bc["sha256"],
            "destination_checkpoint_sha256": initialization["sha256"],
            "source_actor_identity": bc_actor,
            "destination_actor_identity": initialization_actor,
            "literal_byte_identity": True,
            "behavioral_equivalence_claim": False,
        }
    )
    equivalence_path = output / "bc-initialization-actor-equivalence.json"
    _write_json(equivalence_path, equivalence)
    allowed_native = {"t3-initialization-clone": initialization["sha256"]}
    for branch in evidence["lineage"]["t3_branches"]:
        arm, seed = branch["arm"], branch["optimizer_seed"]
        prefix = "t2lo" if arm == "T3lo" else "t2"
        allowed_native[f"{arm.lower()}-s{seed}"] = records[f"{prefix}/{arm}-s{seed}/latest.pt"][
            "sha256"
        ]
    recovered_native = _seal(
        {
            "schema": 1,
            "kind": "gate1_recovered_contract12_adapter_protocol",
            "review_state": "awaiting-astra-high",
            "allowed_checkpoints": allowed_native,
            "raw_missing_actor_keys": sorted(
                {"region_flat", "region_group.weight", "region_group.bias"}
            ),
            "appended_inactive_region_state": {
                "experts-false-token-true": inactive_region_state(
                    int(initialization_state["core_width"]),
                    experts=False,
                    token=True,
                )["identity"],
                "experts-true-token-true": inactive_region_state(
                    int(initialization_state["core_width"]),
                    experts=True,
                    token=True,
                )["identity"],
            },
            "appended_region_scope": "inactive-under-learned_movement=false",
            "historical_worker_contract_claim": False,
            "current_upgrade_allowed": False,
            "implicit_defaults": [],
        }
    )
    recovered_native_path = output / "recovered-contract12-adapter-protocol.json"
    _write_json(recovered_native_path, recovered_native)
    guard_only_protocol = _seal(
        {
            "schema": 1,
            "kind": "gate1_guard_only_intervention_protocol",
            "review_state": "awaiting-astra-high",
            "base_checkpoint_sha256": initialization["sha256"],
            "guard_source_checkpoint_sha256": checkpoint_hash(guarded_checkpoint),
            **guard_only_protocol_fields(initialization_state, guarded_state),
            "composition_scope": "prospective-current-diagnostic-only",
            "normalizer_change_only": True,
            "historical_artifact_claim": False,
            "historical_ancestry_claim": False,
            "outcome_selection_used": False,
            "motivation": (
                "pre-outcome cloning-stability diagnosis: guard dead constant inputs "
                "and clip normalized values without changing actor or affine moments"
            ),
            "implicit_defaults": [],
        }
    )
    guard_only_path = output / "guard-only-intervention-protocol.json"
    _write_json(guard_only_path, guard_only_protocol)
    dag = _build_dag(
        retrieval,
        evidence,
        _ref(equivalence_path, output),
        _ref(bc_path, output),
        guarded_artifact,
        _ref(guarded_protocol_path, output),
        _ref(guard_only_path, output),
        _ref(recovered_native_path, output),
        _ref(expert_expansion_path, output),
    )
    dag_path = output / "artifact-dag.json"
    _write_json(dag_path, dag)

    worlds, cohort = _worlds(seeds)
    present_nodes = [node for node in dag["nodes"] if node["status"] == "present"]
    node_ids = [node["id"] for node in present_nodes]
    rng_path = output / "policy-rng-schedule.json"
    _write_json(rng_path, _rng_schedule(present_nodes, worlds))
    evaluation = _seal(
        {
            "schema": 1,
            "kind": "gate1_evaluation_claim_protocol_proposal",
            "review_state": "awaiting-astra-high",
            "seed_audit": _ref(seed_path, output),
            "cohort": {
                "id": f"gate1-evaluation-only-{seeds[0]}-{seeds[-1]}",
                "digest": cohort,
                "worlds": worlds,
                "selection_allowed": False,
                "readiness_claimed": False,
            },
            "panes": [
                {
                    "id": "fresh-common-producer-deterministic",
                    "kind": "fresh-common-producer-diagnostic",
                    "action_mode": "deterministic",
                    "nodes": node_ids,
                    "background": {"kind": "none", "actor_scope": "all-live-villages"},
                },
                {
                    "id": "fresh-common-producer-seeded-stochastic",
                    "kind": "fresh-common-producer-diagnostic",
                    "action_mode": "seeded-stochastic",
                    "nodes": node_ids,
                    "background": {"kind": "none", "actor_scope": "all-live-villages"},
                    "rng": {
                        "implementation": "torch.Generator.cpu-default",
                        "draw_semantics": "active recovered actor heads only",
                        "schedule": _ref(rng_path, output),
                    },
                },
            ],
            "standardized_interventions": [],
            "standardized_intervention_reason": (
                "fresh common-producer comparison only; "
                "the separately labelled guard-only node is a "
                "prospective normalizer intervention, not historical replay"
            ),
            "mask_semantics": {
                "technology_masks": [{"id": name, "withheld": value} for name, value in MASKS],
                "action_masks": "captured from each source observation and validated against the "
                "exact "
                "decoder receipt; never overridden",
            },
            "personality": [contracts.OBJECTIVE_WEIGHTS[name] for name in contracts.OBJECTIVES],
            "capture_bounds": {
                "max_steps": 25,
                "max_villages_per_boundary": 4096,
                "max_economic_rows_per_step": 4096,
                "max_canonical_bytes": 256 * 1024 * 1024,
            },
            "mandatory_metrics": list(MANDATORY_METRICS),
            "horizon": 25,
            "snapshot_policy": (
                "no snapshots; exact seeded reset; native post-reset turn "
                "and fingerprint bound inline; warmup=0"
            ),
            "early_stop": "typed terminal witness only",
        }
    )
    evaluation_path = output / "evaluation-protocol.json"
    _write_json(evaluation_path, evaluation)

    runtime = _runtime()
    verification = _verification_protocol(runtime, worlds)
    verification_path = output / "verification-protocol.json"
    _write_json(verification_path, verification)
    implementation = _implementation_identity(rl_root, output)
    implementation_path = output / "implementation-identity.json"
    _write_json(implementation_path, implementation)

    candidate_path = output / "freeze-candidate.json"
    commands = _execution_command_artifact(
        runtime,
        output,
        evaluation,
        execution_tier="local-portable-staging",
    )
    commands_path = output / "execution-commands.json"
    _write_json(commands_path, commands)
    candidate = _seal(
        {
            "schema": 1,
            "kind": "gate1_production_freeze_candidate",
            "status": "HOLD-awaiting-vm-materialization",
            "production_gate1": "HOLD",
            "execution_tier": "local-portable-staging",
            "evaluation_authorized": False,
            "selection_allowed": False,
            "readiness_claimed": False,
            "inputs": {
                "retrieval_root": str(retrieval),
                "t2m_retrieval_root": str(t2m_retrieval),
                "current_bridge": current_bridge_record,
                "current_bridge_dependencies": bridge_dependencies,
                "retrieval_inventory_sha256": evidence["raw_sums"]["retrieval-inventory.json"],
                "checkpoint_inventory_sha256": evidence["raw_sums"]["checkpoint-inventory.json"],
                "corrected_lineage_sha256": evidence["sums"]["lineage.json"],
                "accepted_v2_sha256sums_sha256": checkpoint_hash(
                    evidence_package / "SHA256SUMS",
                ),
                "root_evidence_review_sha256": checkpoint_hash(
                    evidence_review,
                ),
            },
            "proposals": {
                "artifact_dag": _ref(dag_path, output),
                "bc_phase_adapter": _ref(bc_path, output),
                "contract8_corpus": _ref(corpus_path, output),
                "contract8_comparator": _ref(comparator_path, output),
                "guarded_bc_corpus": _ref(guarded_corpus_path, output),
                "guarded_bc_adapter": _ref(guarded_protocol_path, output),
                "guard_only_intervention": _ref(guard_only_path, output),
                "recovered_contract12_adapter": _ref(
                    recovered_native_path,
                    output,
                ),
                "recovered_evidence": _ref(
                    evidence_binding_path,
                    output,
                ),
                "expert_expansion_equivalence": _ref(
                    expert_expansion_path,
                    output,
                ),
                "t2m_retrieval_inventory": _ref(t2m_inventory_path, output),
                "bc_initialization_equivalence": _ref(equivalence_path, output),
                "policy_rng_schedule": _ref(rng_path, output),
                "seed_audit": _ref(seed_path, output),
                "evaluation": _ref(evaluation_path, output),
                "verification": _ref(verification_path, output),
                "implementation_identity": _ref(implementation_path, output),
                "execution_commands": _ref(commands_path, output),
            },
            "required_review": {
                "reviewer": "OpenAI gpt-6-astra/high",
                "decision": "approved",
                "scope": [
                    "artifacts",
                    "legacy-adapter",
                    "producer-comparator",
                    "equivalence",
                    "evaluation-cohort",
                    "RNG",
                    "metrics",
                    "verification",
                    "performance",
                    "execution-tier",
                ],
            },
            "unsupported_historical_claims": [
                "contract-8 prior is not the BC parameter/data parent",
                "recovered teacher arrays are not proved to be the exact original BC-fit input",
                "the original BC-time fitted-normalizer identity remains unproved",
                "stage1-token-s6 and recovered T3 normalizers are unguarded",
                "no completed T2m guarded critic checkpoint was recovered",
                "the guard-only node is a prospective diagnostic composition, not a "
                "historical artifact",
                "no BC-to-initialization behavioral-equivalence claim",
                "recovered T3 optimizer seeds are disjoint and unpaired",
                "producer source variants do not cryptographically bind every historical iteration",
            ],
        }
    )
    _write_json(candidate_path, candidate)
    readme = output / "README.md"
    readme.write_text(
        "# Model9 Gate 1 production freeze candidate\n\n"
        "Status: **HOLD pending VM materialization and Astra High review**. This portable "
        "staging package "
        "contains no checkpoint evaluation, simulator run, "
        "performance treatment measurement, training or selection result. `latest.pt` is "
        "the endpoint; `best.pt` is "
        "selection evidence only. The historical T3 arms remain disjoint and unpaired.\n\n"
        "The contract-8 prior is an evaluation reference, not the learned clone's "
        "parameter/data parent. The BC phase "
        "is a learned 12-epoch actor checkpoint. Recovered teacher arrays are inventoried, "
        "but their exact linkage to "
        "the original BC fit and BC-time normalizer remains unresolved. Its only registered "
        "equivalence is literal actor-tensor identity with the recovered unguarded T3 "
        "initialization checkpoint. "
        "No completed T2m guarded critic checkpoint was recovered: the available T2m "
        "record stops during critic fitting. "
        "Its seed-6 actor-only BC checkpoint and original guarded normalizer are a factual "
        "independent evaluation node, "
        "not a T3 ancestor. One additional prospective node places those exact guard/dead "
        "settings on the bitwise T3-init "
        "actor after proving every affine moment identical; it is not a fabricated "
        "historical artifact. Completed-clone "
        "critic provenance remains an explicit HOLD. The final candidate must be "
        "materialized on the VM to bind its Python/.NET runtime, exact source bytes, "
        "rebuilt bridge dependency "
        "set, retrieval location and runnable commands before review.\n",
        encoding="utf-8",
    )
    validate_candidate(candidate_path)
    names = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    sums = "".join(
        f"{checkpoint_hash(path)}  {path.relative_to(output).as_posix()}\n" for path in names
    )
    (output / "SHA256SUMS").write_text(sums, encoding="utf-8")
    return candidate_path


def materialize_vm_candidate(
    portable_candidate_path: Path, rl_root: Path, retrieval: Path, t2m_retrieval: Path, output: Path
) -> Path:
    """Bind a portable staging proposal to the exact VM runtime and paths.

    This performs no bridge construction, reset, checkpoint inference or
    benchmark.  The portable package supplies pinned historical source blobs,
    exhaustive seed-audit evidence and reviewed source identities, so the VM
    requires no Git history.  The resulting candidate still requires a new
    independent review before any execution command is legal.
    """
    portable_base = portable_candidate_path.resolve().parent
    portable = _validate_seal(
        json.loads(portable_candidate_path.read_text(encoding="utf-8")),
        "gate1_production_freeze_candidate",
    )
    if (
        portable.get("execution_tier") != "local-portable-staging"
        or portable.get("status") != "HOLD-awaiting-vm-materialization"
        or portable.get("evaluation_authorized") is not False
    ):
        _fail("VM materialization input is not a portable HOLD proposal")
    rl_root = rl_root.resolve()
    retrieval = retrieval.resolve()
    t2m_retrieval = t2m_retrieval.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        _fail("VM materialization output must be an empty directory")
    output.mkdir(parents=True, exist_ok=True)

    portable_refs = portable["proposals"]
    copied: dict[str, dict[str, Any]] = {}
    static_kinds = {
        "artifact_dag": "gate1_artifact_dag_proposal",
        "bc_phase_adapter": "gate1_bc_phase_adapter_protocol",
        "contract8_corpus": "gate1_contract8_corpus",
        "guarded_bc_corpus": "gate1_guarded_bc_corpus",
        "guard_only_intervention": "gate1_guard_only_intervention_protocol",
        "recovered_contract12_adapter": "gate1_recovered_contract12_adapter_protocol",
        "recovered_evidence": "gate1_recovered_evidence_binding",
        "expert_expansion_equivalence": "gate1_expert_expansion_equivalence_binding",
        "bc_initialization_equivalence": "gate1_exact_tensor_equivalence_proposal",
        "policy_rng_schedule": "gate1_policy_rng_schedule",
        "seed_audit": "gate1_evaluation_seed_audit",
        "evaluation": "gate1_evaluation_claim_protocol_proposal",
    }
    bodies = {}
    for key, kind in static_kinds.items():
        bodies[key] = _read_ref(portable_refs[key], portable_base, kind=kind)
        source = portable_base / portable_refs[key]["path"]
        destination = output / Path(portable_refs[key]["path"]).name
        shutil.copyfile(source, destination)
        copied[key] = _ref(destination, output)
    portable_evidence = portable_base / "recovered-evidence"
    if (
        not portable_evidence.is_dir()
        or portable_evidence.is_symlink()
        or (output / "recovered-evidence").exists()
    ):
        _fail("portable recovered evidence directory is missing or unsafe")
    shutil.copytree(portable_evidence, output / "recovered-evidence")
    portable_expansion = portable_base / "expert-expansion-evidence"
    if (
        not portable_expansion.is_dir()
        or portable_expansion.is_symlink()
        or (output / "expert-expansion-evidence").exists()
    ):
        _fail("portable expert-expansion evidence directory is missing or unsafe")
    shutil.copytree(portable_expansion, output / "expert-expansion-evidence")
    scan_ref = bodies["seed_audit"]["scan_file_inventory"]
    _read_ref(scan_ref, portable_base, kind="gate1_seed_scan_file_inventory")
    scan_source = portable_base / scan_ref["path"]
    scan_destination = output / Path(scan_ref["path"]).name
    shutil.copyfile(scan_source, scan_destination)
    if _ref(scan_destination, output) != scan_ref:
        _fail("portable seed-scan reference changed during VM materialization")

    historical_source = output / "historical-source"
    historical_source.mkdir()
    portable_comparator = _read_ref(
        portable_refs["contract8_comparator"],
        portable_base,
        kind="gate1_contract8_comparator_protocol",
    )
    portable_historical = portable_base / portable_comparator["historical_source_root"]
    for name in ("model.py", "contracts.py"):
        source = portable_historical / name
        if checkpoint_hash(source) != portable_comparator["historical_sources"][name]:
            _fail("portable pinned historical source changed before VM materialization")
        shutil.copyfile(source, historical_source / name)

    guarded_source = output / "guarded-historical-source"
    guarded_source.mkdir()
    portable_guarded = _read_ref(
        portable_refs["guarded_bc_adapter"],
        portable_base,
        kind="gate1_guarded_bc_adapter_protocol",
    )
    portable_guarded_source = portable_base / portable_guarded["historical_source_root"]
    for name in ("model.py", "contracts.py"):
        source = portable_guarded_source / name
        if checkpoint_hash(source) != portable_guarded["historical_sources"][name]:
            _fail("portable pinned guarded source changed before VM materialization")
        shutil.copyfile(source, guarded_source / name)

    portable_identity = _read_ref(
        portable_refs["implementation_identity"],
        portable_base,
        kind="gate1_freeze_implementation_identity",
    )
    implementation = _implementation_identity(
        rl_root,
        output,
        portable_identity=portable_identity,
    )
    implementation_path = output / "implementation-identity.json"
    _write_json(implementation_path, implementation)
    runtime = _runtime()
    comparator = {key: value for key, value in portable_comparator.items() if key != "digest"}
    comparator.update(
        {
            "runtime": comparator_runtime_identity(),
            "historical_source_root": "historical-source",
            "historical_sources": {
                name: checkpoint_hash(historical_source / name)
                for name in ("model.py", "contracts.py")
            },
            "adapter_source": {
                "repository_path": "src/anthropocene_rl/autonomous_v7/legacy_contract8.py",
                "sha256": checkpoint_hash(
                    rl_root / "src/anthropocene_rl/autonomous_v7/legacy_contract8.py"
                ),
            },
            "result": "not-executed-pending-review",
        }
    )
    comparator = _seal(comparator)
    comparator_path = output / "contract8-comparator-protocol.json"
    _write_json(comparator_path, comparator)
    copied["contract8_comparator"] = _ref(comparator_path, output)

    guarded_protocol = {key: value for key, value in portable_guarded.items() if key != "digest"}
    guarded_protocol.update(
        {
            "runtime": comparator_runtime_identity(),
            "historical_source_root": "guarded-historical-source",
            "historical_sources": {
                name: checkpoint_hash(guarded_source / name)
                for name in ("model.py", "contracts.py")
            },
        }
    )
    guarded_protocol = _seal(guarded_protocol)
    guarded_protocol_path = output / "guarded-bc-adapter-protocol.json"
    _write_json(guarded_protocol_path, guarded_protocol)
    copied["guarded_bc_adapter"] = _ref(guarded_protocol_path, output)
    t2m_inventory = _t2m_inventory(t2m_retrieval)
    t2m_inventory_path = output / "t2m-retrieval-inventory.json"
    _write_json(t2m_inventory_path, t2m_inventory)
    copied["t2m_retrieval_inventory"] = _ref(t2m_inventory_path, output)

    # The guarded source protocol is runtime-bound at materialization, so its
    # DAG reference must be resealed as part of the same immutable proposal.
    dag_body = {key: value for key, value in bodies["artifact_dag"].items() if key != "digest"}
    guarded_nodes = [row for row in dag_body["nodes"] if row["id"] == "t2m-guarded-bc-seed6"]
    if len(guarded_nodes) != 1:
        _fail("portable DAG lacks the exact guarded actor node")
    guarded_nodes[0]["loading"]["protocol"] = copied["guarded_bc_adapter"]
    dag_body = _seal(dag_body)
    dag_path = output / "artifact-dag.json"
    _write_json(dag_path, dag_body)
    copied["artifact_dag"] = _ref(dag_path, output)

    verification = _verification_protocol(runtime, bodies["evaluation"]["cohort"]["worlds"])
    verification_path = output / "verification-protocol.json"
    _write_json(verification_path, verification)
    copied["verification"] = _ref(verification_path, output)
    copied["implementation_identity"] = _ref(implementation_path, output)
    commands = _execution_command_artifact(
        runtime,
        output,
        bodies["evaluation"],
        execution_tier="vm",
    )
    commands_path = output / "execution-commands.json"
    _write_json(commands_path, commands)
    copied["execution_commands"] = _ref(commands_path, output)

    current_bridge, bridge_dependencies = _bridge_binding()
    candidate = _seal(
        {
            "schema": 1,
            "kind": "gate1_production_freeze_candidate",
            "status": "HOLD-awaiting-astra-high",
            "production_gate1": "HOLD",
            "execution_tier": "vm",
            "evaluation_authorized": False,
            "selection_allowed": False,
            "readiness_claimed": False,
            "portable_source_candidate_digest": portable["digest"],
            "inputs": {
                "retrieval_root": str(retrieval),
                "t2m_retrieval_root": str(t2m_retrieval),
                "current_bridge": current_bridge,
                "current_bridge_dependencies": bridge_dependencies,
                "retrieval_inventory_sha256": portable["inputs"]["retrieval_inventory_sha256"],
                "checkpoint_inventory_sha256": portable["inputs"]["checkpoint_inventory_sha256"],
                "corrected_lineage_sha256": portable["inputs"]["corrected_lineage_sha256"],
                "accepted_v2_sha256sums_sha256": portable["inputs"][
                    "accepted_v2_sha256sums_sha256"
                ],
                "root_evidence_review_sha256": portable["inputs"]["root_evidence_review_sha256"],
            },
            "proposals": copied,
            "required_review": portable["required_review"],
            "unsupported_historical_claims": portable["unsupported_historical_claims"],
        }
    )
    candidate_path = output / "freeze-candidate.json"
    _write_json(candidate_path, candidate)
    (output / "README.md").write_text(
        "# Model9 Gate 1 VM-materialized freeze candidate\n\n"
        "Status: **HOLD pending Astra High review**. Runtime, source, bridge dependencies, "
        "retrieval paths and commands are bound to this VM. No checkpoint evaluation, simulator "
        "reset or benchmark has run. Historical producer identity remains unresolved; all future "
        "reports are scoped to `fresh-common-producer-diagnostic`.\n",
        encoding="utf-8",
    )
    validate_candidate(candidate_path)
    names = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    sums = "".join(
        f"{checkpoint_hash(path)}  {path.relative_to(output).as_posix()}\n" for path in names
    )
    (output / "SHA256SUMS").write_text(sums, encoding="utf-8")
    return candidate_path


def _verify_execution_identity(candidate: dict[str, Any], base: Path) -> None:
    proposals = candidate.get("proposals", {})
    identity = _read_ref(
        proposals.get("implementation_identity"), base, kind="gate1_freeze_implementation_identity"
    )
    root_value = identity.get("repository_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        _fail("implementation identity lacks an absolute repository root")
    root = Path(root_value)
    expected_module = root / "src/anthropocene_rl/autonomous_v7/gate1_freeze_candidate.py"
    if Path(__file__).resolve() != expected_module.resolve():
        _fail("executing Gate-1 module is not loaded from the reviewed source root")
    records = identity.get("execution_source_files")
    if not isinstance(records, list) or [row.get("path") for row in records] != list(
        EXECUTION_SOURCES
    ):
        _fail("implementation identity does not bind the exact executing source inventory")
    for row in records:
        if set(row) != {"path", "sha256", "size"}:
            _fail("executing source identity is malformed")
        path = root / row["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row["size"]
            or checkpoint_hash(path) != row["sha256"]
        ):
            _fail(f"executing source changed after freeze: {row['path']}")
    verification = _read_ref(
        proposals.get("verification"), base, kind="gate1_verification_protocol_proposal"
    )
    if verification.get("runtime") != _runtime():
        _fail("Python/.NET/package execution runtime changed after freeze")
    inputs = candidate.get("inputs")
    dependencies = inputs.get("current_bridge_dependencies") if isinstance(inputs, dict) else None
    if (
        not isinstance(dependencies, list)
        or len(dependencies) != 3
        or len({row.get("path") for row in dependencies if isinstance(row, dict)}) != 3
    ):
        _fail("current bridge dependency inventory is incomplete")
    observed = {}
    for row in dependencies:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}:
            _fail("current bridge dependency identity is malformed")
        path = Path(row["path"])
        if (
            not path.is_absolute()
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row["size"]
            or checkpoint_hash(path) != row["sha256"]
        ):
            _fail(f"current bridge dependency changed after freeze: {row['path']}")
        observed[path.name] = row
    bridge = inputs.get("current_bridge")
    bridge_path = Path(bridge.get("path", "")) if isinstance(bridge, dict) else Path()
    required_names = {
        bridge_path.name,
        bridge_path.with_suffix(".deps.json").name,
        bridge_path.with_suffix(".runtimeconfig.json").name,
    }
    if set(observed) != required_names or observed.get(bridge_path.name) != bridge:
        _fail("current bridge/dependency identity is internally inconsistent")


def _validate_evidence_binding(ref: dict[str, Any], base: Path) -> dict[str, Any]:
    binding = _read_ref(ref, base, kind="gate1_recovered_evidence_binding")
    if (
        binding.get("accepted_scope") != "factual-inventory-and-source-data-characterization-only"
        or binding.get("production_gate1") != "HOLD"
        or binding.get("v1_usage")
        != "raw 396-file/checkpoint schema table only; no lineage label authority"
        or binding.get("v1_labels_superseded") != ["guarded clone", "guarded contract-12 clone"]
    ):
        _fail("recovered evidence binding overclaims its accepted scope")
    packages = binding.get("packages")
    if not isinstance(packages, dict) or set(packages) != {"accepted-v2", "v1-raw-data"}:
        _fail("recovered evidence binding omits a required package")
    expected_names = {
        "accepted-v2": {
            "analysis-environment.json",
            "build_evidence.py",
            "checkpoint-inventory.json",
            "lineage.json",
            "REPORT.md",
            "retrieval-integrity.json",
            "reward-architecture-characterization.json",
            "source-log-alignment.json",
            "SHA256SUMS",
        },
        "v1-raw-data": {
            "analysis-environment.json",
            "build_evidence.py",
            "checkpoint-inventory.json",
            "clone-training.json",
            "evaluation-summary.json",
            "lineage.json",
            "REPORT.md",
            "retrieval-inventory.json",
            "trajectory-summary.json",
            "SHA256SUMS",
        },
    }
    for label, rows in packages.items():
        if (
            not isinstance(rows, list)
            or {Path(row.get("path", "")).name for row in rows} != expected_names[label]
        ):
            _fail(f"recovered evidence package inventory differs: {label}")
        observed: dict[str, str] = {}
        sums_path: Path | None = None
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}:
                _fail("recovered evidence file reference is malformed")
            relative = Path(row["path"])
            path = base / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != row["size"]
                or checkpoint_hash(path) != row["sha256"]
            ):
                _fail(f"recovered evidence file changed: {row['path']}")
            if path.name == "SHA256SUMS":
                sums_path = path
            else:
                observed[path.name] = row["sha256"]
        if sums_path is None:
            _fail("recovered evidence package lacks SHA256SUMS")
        listed = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            sha, name = line.split("  ", 1)
            listed[name] = sha
        if listed != observed:
            _fail(f"recovered evidence SHA256SUMS differs: {label}")
        if label == "accepted-v2" and checkpoint_hash(sums_path) != binding.get(
            "v2_sha256sums_sha256"
        ):
            _fail("accepted v2 SHA256SUMS is not the reviewed manifest")
    review_ref = binding.get("root_review")
    if not isinstance(review_ref, dict) or set(review_ref) != {"path", "sha256", "size"}:
        _fail("root evidence review reference is malformed")
    review_path = base / review_ref["path"]
    if (
        not review_path.is_file()
        or review_path.is_symlink()
        or checkpoint_hash(review_path) != review_ref["sha256"]
        or review_path.stat().st_size != review_ref["size"]
    ):
        _fail("root evidence review changed")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("kind") != "root_recovered_evidence_review"
        or review.get("decision")
        != "accepted-for-factual-inventory-and-source-data-characterization"
        or review.get("sha256sums_sha256") != binding["v2_sha256sums_sha256"]
        or review.get("production_gate1") != "HOLD"
        or review.get("readiness_claimed") is not False
    ):
        _fail("root evidence review does not accept this exact factual package")
    return binding


def _validate_expert_expansion_binding(ref: dict[str, Any], base: Path) -> dict[str, Any]:
    binding = _read_ref(ref, base, kind="gate1_expert_expansion_equivalence_binding")
    if (
        binding.get("status") != "equivalent-on-frozen-corpus"
        or binding.get("scope")
        != "virtual PPO-entry parameter expansion only; no rollout node or outcome claim"
        or binding.get("optimizer_seeds") != list(range(70, 85))
        or binding.get("maximum_finite_forward_error") != 0.0
    ):
        _fail("expert-expansion binding overclaims or changes its exact scope")
    rows = binding.get("files")
    expected = {
        "analysis-environment.json",
        "audit_expert_expansion.py",
        "expert-expansion-audit.json",
        "REPORT.md",
        "SHA256SUMS",
    }
    if not isinstance(rows, list) or {Path(row.get("path", "")).name for row in rows} != expected:
        _fail("expert-expansion binding file inventory differs")
    listed = {}
    observed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}:
            _fail("expert-expansion evidence reference is malformed")
        relative = Path(row["path"])
        path = base / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row["size"]
            or checkpoint_hash(path) != row["sha256"]
        ):
            _fail(f"expert-expansion evidence changed: {row['path']}")
        if path.name == "SHA256SUMS":
            for line in path.read_text(encoding="utf-8").splitlines():
                sha, name = line.split("  ", 1)
                listed[name] = sha
        else:
            observed[path.name] = row["sha256"]
    if listed != observed:
        _fail("expert-expansion SHA256SUMS differs from copied evidence")
    audit_path = next(
        base / row["path"]
        for row in rows
        if Path(row["path"]).name == "expert-expansion-audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "equivalent"
        or audit.get("actual_initializer", {}).get("sha256") != binding.get("initializer_sha256")
        or audit.get("summary", {}).get(
            "all_seed_and_batch_forward_outputs_and_decoded_modes_equivalent"
        )
        is not True
        or audit.get("summary", {}).get("maximum_finite_forward_error") != 0.0
    ):
        _fail("expert-expansion audit differs from its binding")
    return binding


def validate_candidate(candidate_path: Path) -> dict[str, Any]:
    base = candidate_path.resolve().parent
    candidate = _validate_seal(
        json.loads(candidate_path.read_text(encoding="utf-8")), "gate1_production_freeze_candidate"
    )
    tier = candidate.get("execution_tier")
    expected_status = {
        "local-portable-staging": "HOLD-awaiting-vm-materialization",
        "vm": "HOLD-awaiting-astra-high",
    }.get(tier)
    if (
        expected_status is None
        or candidate.get("status") != expected_status
        or candidate.get("evaluation_authorized") is not False
        or candidate.get("selection_allowed") is not False
        or candidate.get("readiness_claimed") is not False
    ):
        _fail("freeze candidate overclaims authority")
    proposals = candidate.get("proposals")
    expected = {
        "artifact_dag",
        "bc_phase_adapter",
        "contract8_corpus",
        "contract8_comparator",
        "guarded_bc_corpus",
        "guarded_bc_adapter",
        "guard_only_intervention",
        "recovered_contract12_adapter",
        "recovered_evidence",
        "expert_expansion_equivalence",
        "t2m_retrieval_inventory",
        "bc_initialization_equivalence",
        "policy_rng_schedule",
        "seed_audit",
        "evaluation",
        "verification",
        "implementation_identity",
        "execution_commands",
    }
    if not isinstance(proposals, dict) or set(proposals) != expected:
        _fail("freeze candidate proposal inventory is incomplete")
    evidence_binding = _validate_evidence_binding(proposals["recovered_evidence"], base)
    expansion_binding = _validate_expert_expansion_binding(
        proposals["expert_expansion_equivalence"],
        base,
    )
    inputs = candidate.get("inputs", {})
    if (
        inputs.get("accepted_v2_sha256sums_sha256") != evidence_binding["v2_sha256sums_sha256"]
        or inputs.get("root_evidence_review_sha256") != evidence_binding["root_review"]["sha256"]
    ):
        _fail("freeze candidate inputs do not bind the accepted factual evidence/review")
    dag = _read_ref(proposals["artifact_dag"], base, kind="gate1_artifact_dag_proposal")
    nodes = dag["nodes"]
    if (
        len(nodes) != 20
        or len({row["id"] for row in nodes}) != 20
        or sum(row["role"] == "trained-t3" for row in nodes) != 15
    ):
        _fail(
            "artifact DAG does not contain the five reviewed initialization surfaces and "
            "15 T3 endpoints"
        )
    trained = [row for row in nodes if row["role"] == "trained-t3"]
    guarded = [row for row in nodes if row["id"] == "t2m-guarded-bc-seed6"]
    if (
        len(guarded) != 1
        or guarded[0].get("role") != "guarded-bc-actor-phase"
        or guarded[0].get("critic_status") != "absent-in-actor-only-bc-phase"
        or guarded[0].get("selection_basis")
        != "seed identity 6 matches historical T3 initialization; no outcome selection"
        or guarded[0].get("loading", {}).get("adapter") != "t2m-guarded-bc-actor-only"
    ):
        _fail("T2m guarded actor-only phase is absent or overclaims completion/selection")
    initialization = next(row for row in nodes if row["id"] == "t3-initialization-clone")
    if (
        initialization.get("status") != "present"
        or initialization.get("normalizer_semantics") != "unguarded"
        or initialization.get("loading", {}).get("adapter")
        != "recovered-contract12-inactive-region"
        or initialization.get("loading", {}).get("protocol")
        != proposals["recovered_contract12_adapter"]
    ):
        _fail("stage1-token-s6 was not preserved as the unguarded T3 initialization")
    interventions = [row for row in nodes if row["id"] == "t3-init-guard-only-intervention"]
    if (
        len(interventions) != 1
        or interventions[0].get("role") != "prospective-normalizer-intervention"
        or interventions[0].get("artifact") != initialization.get("artifact")
        or interventions[0].get("composition_scope") != "prospective-current-diagnostic-only"
        or interventions[0].get("historical_artifact_claim") is not False
        or interventions[0].get("historical_ancestry_claim") is not False
        or interventions[0].get("loading", {}).get("adapter") != "t3-init-guard-only-intervention"
        or interventions[0].get("loading", {}).get("protocol")
        != proposals["guard_only_intervention"]
        or interventions[0].get("loading", {}).get("guard_source_checkpoint")
        != guarded[0].get("artifact")
    ):
        _fail("guard-only intervention does not bind the reviewed T3 actor and T2m guard source")
    trained_ids = {row["id"] for row in nodes if row["role"] == "trained-t3"}
    edges = dag.get("edges")
    if (
        not isinstance(edges, list)
        or len(edges) != 3 + len(trained_ids)
        or {
            (row.get("source"), row.get("destination"))
            for row in edges
            if row.get("relation") == "ppo_continuation"
        }
        != {("t3-initialization-clone", node_id) for node_id in trained_ids}
        or any(
            row.get("status") != "present"
            for row in edges
            if row.get("relation") == "ppo_continuation"
        )
        or any(
            row.get("parameter_expansion_substep")
            != {
                "kind": "reconstructed-experts-architecture-expansion",
                "source_semantics": "experts=false",
                "destination_entry_semantics": "experts=true",
                "equivalence_evidence": proposals["expert_expansion_equivalence"],
                "equivalence_scope": "frozen teacher-state/mask corpus only",
                "rollout_node_added": False,
            }
            or "not a pure-gradient or pure-PPO causal effect" not in row.get("interpretation", "")
            for row in edges
            if row.get("relation") == "ppo_continuation"
        )
        or len(
            [
                row
                for row in edges
                if row.get("relation") == "normalizer_change"
                and row.get("source") == "t3-initialization-clone"
                and row.get("destination") == "t3-init-guard-only-intervention"
                and row.get("equivalence") is None
            ]
        )
        != 1
        or not isinstance(dag.get("lineage_conflict"), str)
        or not dag["lineage_conflict"].strip()
        or expansion_binding.get("initializer_sha256") != initialization["artifact"]["sha256"]
    ):
        _fail("artifact DAG does not preserve the factual unguarded T3 initialization branches")
    slots = dag.get("user_slots", {})
    if (
        slots.get("t3-initialization-clone") != ["t3-initialization-clone"]
        or not isinstance(slots.get("guarded-clone"), dict)
        or slots["guarded-clone"].get("status")
        != "actor-phase-present-completed-critic-clone-unresolved"
        or slots["guarded-clone"].get("node_ids") != ["t2m-guarded-bc-seed6"]
        or not slots["guarded-clone"].get("reason")
        or not slots["guarded-clone"].get("interpretation")
        or slots.get("prospective-normalizer-intervention", {}).get("node_ids")
        != ["t3-init-guard-only-intervention"]
        or slots["prospective-normalizer-intervention"].get("historical_artifact_claim")
        is not False
    ):
        _fail("user-facing lineage slots hide or overclaim the incomplete guarded attempt")
    roots = {"t3": inputs.get("retrieval_root"), "t2m": inputs.get("t2m_retrieval_root")}
    if any(not isinstance(value, str) or not Path(value).is_absolute() for value in roots.values()):
        _fail("candidate lacks its absolute tier-specific retrieval roots")
    for node in nodes:
        for field in ("artifact", "selection_evidence_only"):
            artifact = node.get(field)
            if artifact is None:
                continue
            relative = Path(artifact.get("path", "")) if isinstance(artifact, dict) else Path()
            root_id = artifact.get("root_id") if isinstance(artifact, dict) else None
            path = Path(roots.get(root_id, "")) / relative
            if (
                root_id not in roots
                or relative.is_absolute()
                or ".." in relative.parts
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != artifact.get("size")
                or checkpoint_hash(path) != artifact.get("sha256")
            ):
                _fail(
                    f"reviewed checkpoint artifact is missing or changed: {node.get('id')}/{field}"
                )
    t2m_inventory = _read_ref(
        proposals["t2m_retrieval_inventory"],
        base,
        kind="gate1_t2m_retrieval_inventory",
    )
    if t2m_inventory != _t2m_inventory(Path(roots["t2m"])):
        _fail("T2m retrieval inventory differs from the reviewed bytes")
    guarded_protocol = _read_ref(
        proposals["guarded_bc_adapter"],
        base,
        kind="gate1_guarded_bc_adapter_protocol",
    )
    guarded_corpus = _read_ref(
        proposals["guarded_bc_corpus"],
        base,
        kind="gate1_guarded_bc_corpus",
    )
    if (
        guarded_protocol.get("checkpoint_sha256") != guarded[0]["artifact"]["sha256"]
        or guarded_protocol.get("runtime") != comparator_runtime_identity()
        or tuple(guarded_protocol.get("mapped_outputs", ())) != GUARDED_BC_OUTPUTS
        or tuple(guarded_protocol.get("decoded_actions", ())) != GUARDED_BC_ACTIONS
        or guarded_protocol.get("corpus_sha256") != proposals["guarded_bc_corpus"]["sha256"]
        or guarded_protocol.get("critic_status") != "absent-in-actor-only-bc-phase"
        or guarded_protocol.get("historical_t3_ancestry_claim") is not False
        or guarded_protocol.get("completed_critic_checkpoint_claim") is not False
    ):
        _fail("guarded BC adapter protocol overclaims or differs from its reviewed surface")
    guarded_source_root = Path(guarded_protocol.get("historical_source_root", ""))
    guarded_sources = guarded_protocol.get("historical_sources")
    guarded_source_path = base / guarded_source_root
    if (
        guarded_source_root.is_absolute()
        or guarded_source_root.as_posix() != "guarded-historical-source"
        or not isinstance(guarded_sources, dict)
        or set(guarded_sources) != {"model.py", "contracts.py"}
        or not guarded_source_path.is_dir()
        or guarded_source_path.is_symlink()
        or {path.name for path in guarded_source_path.iterdir() if path.is_file()}
        != {"model.py", "contracts.py"}
        or any(
            checkpoint_hash(guarded_source_path / name) != sha
            for name, sha in guarded_sources.items()
        )
    ):
        _fail("guarded BC pinned source inventory is incomplete or changed")
    validate_contract8_corpus(
        guarded_corpus,
        core_width=guarded_protocol.get("core_width"),
    )
    guard_only = _read_ref(
        proposals["guard_only_intervention"],
        base,
        kind="gate1_guard_only_intervention_protocol",
    )
    initialization_path = Path(roots["t3"]) / initialization["artifact"]["path"]
    guarded_path = Path(roots["t2m"]) / guarded[0]["artifact"]["path"]
    base_state = torch.load(initialization_path, map_location="cpu", weights_only=True)
    guarded_state = torch.load(guarded_path, map_location="cpu", weights_only=True)
    expected_guard_only = guard_only_protocol_fields(base_state, guarded_state)
    if (
        guard_only.get("base_checkpoint_sha256") != initialization["artifact"]["sha256"]
        or guard_only.get("guard_source_checkpoint_sha256") != guarded[0]["artifact"]["sha256"]
        or any(guard_only.get(name) != value for name, value in expected_guard_only.items())
        or guard_only.get("composition_scope") != "prospective-current-diagnostic-only"
        or guard_only.get("normalizer_change_only") is not True
        or guard_only.get("historical_artifact_claim") is not False
        or guard_only.get("historical_ancestry_claim") is not False
        or guard_only.get("outcome_selection_used") is not False
        or guard_only.get("implicit_defaults") != []
    ):
        _fail("guard-only intervention protocol differs from its two exact source artifacts")
    recovered_native = _read_ref(
        proposals["recovered_contract12_adapter"],
        base,
        kind="gate1_recovered_contract12_adapter_protocol",
    )
    expected_native = {
        initialization["id"]: initialization["artifact"]["sha256"],
        **{row["id"]: row["artifact"]["sha256"] for row in trained},
    }
    expected_appended = {
        "experts-false-token-true": inactive_region_state(
            expected_guard_only["core_width"],
            experts=False,
            token=True,
        )["identity"],
        "experts-true-token-true": inactive_region_state(
            expected_guard_only["core_width"],
            experts=True,
            token=True,
        )["identity"],
    }
    if (
        recovered_native.get("allowed_checkpoints") != expected_native
        or set(recovered_native.get("raw_missing_actor_keys", ()))
        != {"region_flat", "region_group.weight", "region_group.bias"}
        or recovered_native.get("appended_inactive_region_state") != expected_appended
        or recovered_native.get("appended_region_scope") != "inactive-under-learned_movement=false"
        or recovered_native.get("historical_worker_contract_claim") is not False
        or recovered_native.get("current_upgrade_allowed") is not False
        or recovered_native.get("implicit_defaults") != []
    ):
        _fail("recovered contract-12 adapter protocol differs from the exact endpoint inventory")
    from .evaluate import load_strict_state
    from .gate1_diagnostics import preserve_global_rng

    protocol_ref = _adapter_ref(
        proposals["recovered_contract12_adapter"],
        base,
        kind="gate1_recovered_contract12_adapter_protocol",
    )
    with preserve_global_rng():
        for node in [initialization, *trained]:
            checkpoint = Path(roots[node["artifact"]["root_id"]]) / node["artifact"]["path"]
            raw = load_recovered_contract12(
                checkpoint,
                {"protocol": protocol_ref, "node_id": node["id"]},
                node["historical_semantics"],
                base=base,
            )
            try:
                load_strict_state(
                    raw,
                    resolved_semantics=node["historical_semantics"],
                    diagnostic_loading=None,
                )
            except (KeyError, TypeError, ValueError, RuntimeError, ProvenanceError) as exc:
                _fail(f"reviewed recovered endpoint is not strict-loadable: {node['id']}: {exc}")
    expected_seeds = {
        "T3per": set(range(70, 75)),
        "T3def": set(range(75, 80)),
        "T3lo": set(range(80, 85)),
    }
    actual = {
        arm: {
            row["run_identity"]["optimizer_seed"]
            for row in trained
            if row["run_identity"]["arm_id"] == arm
        }
        for arm in expected_seeds
    }
    if actual != expected_seeds or any(
        row["run_identity"]["pairing_status"] != "unpaired" for row in trained
    ):
        _fail("T3 historical optimizer identities were paired, omitted or relabelled")
    if any(
        row.get("loading", {}).get("adapter") != "recovered-contract12-inactive-region"
        or row["loading"].get("protocol") != proposals["recovered_contract12_adapter"]
        or row["loading"].get("node_id") != row["id"]
        for row in trained
    ):
        _fail("T3 endpoint loader bypasses the reviewed pre-region actor adapter")
    if any(row["artifact"]["path"].endswith("best.pt") for row in trained):
        _fail("best.pt substituted for a latest endpoint")
    equivalence = _read_ref(
        proposals["bc_initialization_equivalence"],
        base,
        kind="gate1_exact_tensor_equivalence_proposal",
    )
    if (
        equivalence.get("scope") != "actor-tensors-only"
        or equivalence.get("edge_id") != "bc-phase-to-initialization"
        or equivalence.get("behavioral_equivalence_claim") is not False
        or equivalence.get("literal_byte_identity") is not True
        or any(
            edge.get("equivalence") is not None and edge["id"] != "bc-phase-to-initialization"
            for edge in dag["edges"]
        )
    ):
        _fail("artifact DAG asserts an unsupported equivalence")
    seed_audit = _read_ref(proposals["seed_audit"], base, kind="gate1_evaluation_seed_audit")
    scan = _read_ref(seed_audit["scan_file_inventory"], base, kind="gate1_seed_scan_file_inventory")
    if (
        seed_audit.get("status") != "fresh-in-scanned-scope"
        or seed_audit.get("prior_use_occurrences") != []
        or scan.get("scanned_file_count") != len(scan["scanned_files"])
    ):
        _fail("evaluation seed family lacks a complete no-prior-use scan")
    seeds = seed_audit["candidate_family"]["seeds"]
    if set(seeds) & ({seed for values in ROLES.values() for seed in values} | DENIED):
        _fail("evaluation cohort overlaps a forbidden family")
    evaluation = _read_ref(
        proposals["evaluation"], base, kind="gate1_evaluation_claim_protocol_proposal"
    )
    worlds = evaluation["cohort"]["worlds"]
    if (
        len(worlds) != len(seeds) * len(MASKS)
        or {(row["seed"], row["withheld"]) for row in worlds}
        != {(seed, mask) for seed in seeds for _, mask in MASKS}
        or any(
            row["horizon"] != 25
            or row["warmup"] != 0
            or row.get("reset_turn") is not None
            or row.get("turn") is not None
            or row.get("reset_turn_policy") != "bind-native-post-reset-boundary"
            or row["players"] != 8
            or row["snapshot"] != {"kind": "none", "sha256": None}
            for row in worlds
        )
        or evaluation.get("snapshot_policy")
        != (
            "no snapshots; exact seeded reset; native post-reset turn "
            "and fingerprint bound inline; warmup=0"
        )
        or evaluation.get("mandatory_metrics") != list(MANDATORY_METRICS)
    ):
        _fail("evaluation cohort/mask/reset/horizon/metric surface is incomplete")
    panes = evaluation.get("panes")
    if (
        [row.get("action_mode") for row in panes] != ["deterministic", "seeded-stochastic"]
        or any(row.get("kind") != "fresh-common-producer-diagnostic" for row in panes)
        or evaluation.get("standardized_interventions") != []
    ):
        _fail("evaluation action-mode/pane separation differs from the proposal")
    schedule_ref = panes[1]["rng"].get("schedule")
    if schedule_ref != proposals["policy_rng_schedule"]:
        _fail("evaluation pane substituted its stochastic schedule")
    schedule = _read_ref(schedule_ref, base, kind="gate1_policy_rng_schedule")
    schedules = schedule["worlds"]
    present_node_ids = [node["id"] for node in nodes if node["status"] == "present"]
    if (
        len(schedules) != len(worlds)
        or {row["world_id"] for row in schedules} != {row["id"] for row in worlds}
        or any(
            [entry["node_id"] for entry in row["nodes"]] != present_node_ids for row in schedules
        )
        or any(
            entry.get("initial_state_coupling") != "same-world-and-step-across-nodes"
            or entry.get("post_draw_coupling_claim") is not False
            for row in schedules
            for entry in row["nodes"]
        )
        or any(
            any(
                any(
                    len({entry["steps"][step][field] for entry in row["nodes"]}) != 1
                    for field in ("seed", "state_sha256")
                )
                for step in range(len(row["nodes"][0]["steps"]))
            )
            for row in schedules
        )
        or any(pane.get("nodes") != present_node_ids for pane in panes)
    ):
        _fail("seeded-stochastic schedule does not exactly cover compatible nodes/worlds")
    verification = _read_ref(
        proposals["verification"], base, kind="gate1_verification_protocol_proposal"
    )
    if (
        verification["performance"].get("baseline_calibration") is not None
        or verification["performance"].get("acceptance") != "relative-paired-only"
    ):
        _fail("verification substitutes calibration/absolute thresholds for final paired evidence")
    limits = {row["id"]: row["rule"]["limit"] for row in verification["performance"]["profiles"]}
    if limits != {
        "diagnostic-off-current": 0.98,
        "diagnostic-on-current": 1.15,
        "diagnostic-on-mature": 1.20,
        "eight-worker-end-to-end": 0.80,
    }:
        _fail("performance proposal differs from the reviewed relative budgets")
    for key in (
        "bc_phase_adapter",
        "contract8_corpus",
        "contract8_comparator",
        "guarded_bc_corpus",
        "guarded_bc_adapter",
        "guard_only_intervention",
        "recovered_contract12_adapter",
        "recovered_evidence",
        "expert_expansion_equivalence",
        "t2m_retrieval_inventory",
        "implementation_identity",
        "execution_commands",
    ):
        _read_ref(proposals[key], base)
    if any(row.get("historical_producer_status") != "unresolved" for row in nodes):
        _fail("fresh common-producer proposal fabricated a historical producer identity")
    bc_protocol = _read_ref(
        proposals["bc_phase_adapter"], base, kind="gate1_bc_phase_adapter_protocol"
    )
    if (
        bc_protocol.get("composite_scope") != "current-diagnostic-only"
        or bc_protocol.get("historical_bc_normalizer_claim") is not False
        or bc_protocol.get("support_normalizer_semantics") != "unguarded"
        or bc_protocol.get("missing_guarded_clone_claim") is not False
        or bc_protocol.get("appended_inactive_region_state")
        != expected_appended["experts-false-token-true"]
        or bc_protocol.get("appended_region_scope") != "inactive-under-learned_movement=false"
    ):
        _fail("BC phase proposal overclaims its reconstructed normalizer scope")
    comparator = _read_ref(
        proposals["contract8_comparator"], base, kind="gate1_contract8_comparator_protocol"
    )
    if (
        set(comparator.get("historical_sources", {})) != {"model.py", "contracts.py"}
        or comparator.get("mapped_outputs") != list(CONTRACT8_OUTPUTS)
        or comparator.get("decoded_actions") != list(CONTRACT8_ACTIONS)
        or comparator.get("runtime") != comparator_runtime_identity()
    ):
        _fail("contract-8 comparator source/runtime/surface is incomplete")
    corpus = _read_ref(proposals["contract8_corpus"], base, kind="gate1_contract8_corpus")
    validate_contract8_corpus(corpus)
    commands = _read_ref(
        proposals["execution_commands"], base, kind="gate1_post_review_execution_commands"
    )
    if commands.get("execution_tier") != tier:
        _fail("post-review commands target another execution tier")
    command_set = commands.get("commands")
    policy = commands.get("collection_execution_policy")
    if (
        not isinstance(command_set, dict)
        or not isinstance(policy, dict)
        or policy.get("pre_calibration_max_concurrent_node_pane_processes") != 1
        or policy.get("proposed_post_calibration_max_concurrent_node_pane_processes") != 8
        or not isinstance(policy.get("post_calibration_authority"), str)
        or policy.get("maximum_in_memory_world_results_per_process") != 1
        or policy.get("world_retention") != "write one sealed world artifact then release it"
    ):
        _fail("post-review command inventory is malformed")
    evaluations = command_set.get("checkpoint_evaluations")
    expected_pairs = {(node_id, pane["id"]) for pane in panes for node_id in pane["nodes"]}
    if (
        not isinstance(evaluations, list)
        or len(evaluations) != len(expected_pairs)
        or {(row.get("node"), row.get("pane")) for row in evaluations} != expected_pairs
    ):
        _fail("post-review commands do not cover every exact node/pane pair")
    all_commands = [
        command_set.get("validate_candidate"),
        command_set.get("contract8_comparator"),
        command_set.get("guarded_bc_comparator"),
        command_set.get("capture_current_worker"),
    ]
    all_commands.extend(row.get("command") for row in evaluations)
    if any(
        not isinstance(command, list)
        or not command
        or command[0] != verification["runtime"]["python"]["path"]
        or any(not isinstance(part, str) or "<" in part or ">" in part for part in command)
        for command in all_commands
    ):
        _fail("post-review command inventory contains placeholders or another runtime")
    _verify_execution_identity(candidate, base)
    return {
        "schema": 1,
        "kind": "gate1_freeze_candidate_validation",
        "status": "HOLD-PROPOSAL-ONLY",
        "candidate_digest": candidate["digest"],
        "evaluation_authorized": False,
        "production_gate1": "HOLD",
        "required_missing": ["guarded-clone-completed-critic-checkpoint"],
        "node_count": len(nodes),
        "world_count": len(worlds),
        "seed_scan_file_count": scan["scanned_file_count"],
    }


def _load_reviewed_candidate(
    candidate_path: Path, review_path: Path
) -> tuple[dict[str, Any], Path]:
    validate_candidate(candidate_path)
    candidate = _validate_seal(
        json.loads(candidate_path.read_text(encoding="utf-8")), "gate1_production_freeze_candidate"
    )
    if candidate.get("execution_tier") != "vm":
        _fail("checkpoint execution requires a VM-materialized candidate")
    _review(candidate, review_path)
    return candidate, candidate_path.resolve().parent


def capture_current_worker(candidate_path: Path, review_path: Path, out: Path) -> dict[str, Any]:
    """Capture the exact common producer after, and only after, proposal review.

    Bridge construction reads the schema but does not reset a world.  The
    resulting contract is later re-observed by every rollout before reset.
    """
    candidate, _ = _load_reviewed_candidate(candidate_path, review_path)
    frozen = candidate["inputs"]["current_bridge"]
    bridge_path = Path(frozen["path"])
    if (
        not bridge_path.is_file()
        or bridge_path.stat().st_size != frozen["size"]
        or checkpoint_hash(bridge_path) != frozen["sha256"]
    ):
        _fail("current diagnostic bridge differs from the reviewed binary")
    from .bridge import Model9Bridge, dotnet_command

    command = dotnet_command()
    if Path(command[-1]).resolve() != bridge_path.resolve():
        _fail("configured bridge command differs from the reviewed current binary")
    from .worker_contract import observe

    with Model9Bridge(command=command) as bridge:
        worker = observe(bridge)
        core_width = bridge.schema.core_width
    body = _seal(
        {
            "schema": 1,
            "kind": "gate1_current_worker_capture",
            "candidate_digest": candidate["digest"],
            "bridge": frozen,
            "worker_contract": worker,
            "observation_core_width": core_width,
            "scope": "fresh-common-producer-diagnostic",
            "historical_producer_equivalence_claim": False,
        }
    )
    _write_json(out, body)
    return body


def _worker_capture(ref_path: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    body = _validate_seal(
        json.loads(ref_path.read_text(encoding="utf-8")), "gate1_current_worker_capture"
    )
    if (
        body.get("candidate_digest") != candidate["digest"]
        or body.get("bridge") != candidate["inputs"]["current_bridge"]
        or body.get("scope") != "fresh-common-producer-diagnostic"
        or body.get("historical_producer_equivalence_claim") is not False
    ):
        _fail("current worker capture is bound to another proposal or overclaims history")
    from .worker_contract import checked

    checked(body.get("worker_contract"))
    if type(body.get("observation_core_width")) is not int or body["observation_core_width"] <= 0:
        _fail("current worker capture lacks its observation width")
    return body


def evaluate_reviewed_node(
    candidate_path: Path,
    review_path: Path,
    worker_path: Path,
    contract8_result_path: Path,
    guarded_result_path: Path,
    node_id: str,
    pane_id: str,
    out: Path,
) -> dict[str, Any]:
    """Run one exact reviewed node/pane on the common current producer."""
    candidate, base = _load_reviewed_candidate(candidate_path, review_path)
    worker_capture = _worker_capture(worker_path, candidate)
    proposals = candidate["proposals"]
    dag = _read_ref(proposals["artifact_dag"], base, kind="gate1_artifact_dag_proposal")
    evaluation = _read_ref(
        proposals["evaluation"], base, kind="gate1_evaluation_claim_protocol_proposal"
    )
    nodes = [row for row in dag["nodes"] if row["id"] == node_id]
    panes = [row for row in evaluation["panes"] if row["id"] == pane_id]
    if len(nodes) != 1 or len(panes) != 1 or node_id not in panes[0]["nodes"]:
        _fail("reviewed evaluation names an unknown node/pane")
    node, pane = nodes[0], panes[0]
    if pane.get("kind") != "fresh-common-producer-diagnostic":
        _fail("production runner only supports the approved fresh common-producer pane")
    root_key = {"t3": "retrieval_root", "t2m": "t2m_retrieval_root"}.get(
        node["artifact"].get("root_id")
    )
    if root_key is None:
        _fail("reviewed node uses an unknown retrieval root")
    retrieval = Path(candidate["inputs"][root_key])
    checkpoint = retrieval / node["artifact"]["path"]
    if (
        checkpoint_hash(checkpoint) != node["artifact"]["sha256"]
        or checkpoint.stat().st_size != node["artifact"]["size"]
    ):
        _fail("reviewed checkpoint changed before evaluation")
    diagnostic_loading = None
    if node_id == "prior-reference":
        result = _validate_seal(
            json.loads(contract8_result_path.read_text(encoding="utf-8")),
            "gate1_contract8_comparator_result",
        )
        if (
            result.get("status") != "PASS"
            or result.get("checkpoint_sha256") != node["artifact"]["sha256"]
            or result.get("protocol_sha256") != proposals["contract8_comparator"]["sha256"]
            or result.get("corpus_sha256") != proposals["contract8_corpus"]["sha256"]
        ):
            _fail("contract-8 source/adapter comparator is absent or bound to another input")
        diagnostic_loading = {
            "mode": "reviewed-adapter",
            "adapter": "pinned-contract-8",
            "protocol_sha256": proposals["contract8_comparator"]["sha256"],
        }
    elif node_id == "bc-phase":
        protocol_ref = _adapter_ref(
            proposals["bc_phase_adapter"],
            base,
            kind="gate1_bc_phase_adapter_protocol",
        )
        diagnostic_loading = {
            "mode": "reviewed-adapter",
            "adapter": "t3-initialization-bc-composite",
            "protocol": protocol_ref,
            "support_checkpoint": {
                "path": str(
                    retrieval
                    / next(row for row in dag["nodes"] if row["id"] == "t3-initialization-clone")[
                        "artifact"
                    ]["path"]
                ),
                "sha256": next(
                    row for row in dag["nodes"] if row["id"] == "t3-initialization-clone"
                )["artifact"]["sha256"],
            },
        }
    elif node_id == "t2m-guarded-bc-seed6":
        result = _validate_seal(
            json.loads(guarded_result_path.read_text(encoding="utf-8")),
            "gate1_guarded_bc_comparator_result",
        )
        if (
            result.get("status") != "PASS"
            or result.get("checkpoint_sha256") != node["artifact"]["sha256"]
            or result.get("protocol_sha256") != proposals["guarded_bc_adapter"]["sha256"]
            or result.get("corpus_sha256") != proposals["guarded_bc_corpus"]["sha256"]
        ):
            _fail("guarded BC source/adapter comparator is absent or bound to another input")
        diagnostic_loading = {
            "mode": "reviewed-adapter",
            "adapter": "t2m-guarded-bc-actor-only",
            "protocol": _adapter_ref(
                proposals["guarded_bc_adapter"],
                base,
                kind="gate1_guarded_bc_adapter_protocol",
            ),
        }
    elif node_id == "t3-init-guard-only-intervention":
        guard_source = node["loading"]["guard_source_checkpoint"]
        diagnostic_loading = {
            "mode": "reviewed-adapter",
            "adapter": "t3-init-guard-only-intervention",
            "protocol": _adapter_ref(
                proposals["guard_only_intervention"],
                base,
                kind="gate1_guard_only_intervention_protocol",
            ),
            "guard_source_checkpoint": {
                "path": str(Path(candidate["inputs"]["t2m_retrieval_root"]) / guard_source["path"]),
                "sha256": guard_source["sha256"],
            },
        }
    elif node.get("loading", {}).get("adapter") == "recovered-contract12-inactive-region":
        diagnostic_loading = {
            "mode": "reviewed-adapter",
            "adapter": "recovered-contract12-inactive-region",
            "protocol": _adapter_ref(
                proposals["recovered_contract12_adapter"],
                base,
                kind="gate1_recovered_contract12_adapter_protocol",
            ),
            "node_id": node_id,
        }
    from .gate1_diagnostics import service_outcome_registry, validate_rng_schedule

    rng = None
    if pane["action_mode"] == "seeded-stochastic":
        rng = _read_ref(proposals["policy_rng_schedule"], base, kind="gate1_policy_rng_schedule")
    from . import evaluate
    from .worker_contract import world as world_contract

    bounds = evaluation["capture_bounds"]
    personality = evaluation["personality"]
    if out.exists():
        _fail("evaluation output already exists")
    world_root = out.parent / f"{out.stem}.worlds"
    if world_root.exists():
        _fail("evaluation world-artifact directory already exists")
    world_root.mkdir(parents=True)
    row_artifacts = []
    for frozen in evaluation["cohort"]["worlds"]:
        policy_rng_steps = None
        if rng is not None:
            from .gate1_diagnostics import draw_order_from_semantics

            policy_rng_steps = validate_rng_schedule(
                rng,
                frozen["id"],
                frozen["horizon"],
                node_id=node_id,
                expected_draw_order=draw_order_from_semantics(node["historical_semantics"]),
            )
        descriptor = world_contract(
            frozen["seed"],
            frozen["withheld"],
            # A generated world has no pre-reset turn: reset_metadata reports
            # ``null`` until generation.  evaluate.rollout binds the native
            # post-reset boundary returned for this exact seed/mask.  Snapshot
            # cohorts instead bind their already-materialized source turn.
            turn=(None if frozen["snapshot"]["kind"] == "none" else frozen["turn"]),
            players=frozen["players"],
            movement=node["historical_semantics"]["learned_movement"],
            head=node["historical_semantics"]["region_head"],
            cohort_digest=frozen["cohort_digest"],
        )
        result = evaluate.rollout(
            frozen["seed"],
            frozen["horizon"],
            str(checkpoint),
            frozen["withheld"],
            node["historical_semantics"]["learned_movement"],
            node["historical_semantics"]["commitments"],
            node["historical_semantics"]["region_movement"],
            frozen["warmup"],
            node["historical_semantics"]["learned_fission"],
            node["historical_semantics"]["learned_trade"],
            personality,
            node["historical_semantics"]["grouped_region"],
            False,
            worker_capture["worker_contract"],
            descriptor,
            strict_checkpoint=True,
            resolved_semantics=node["historical_semantics"],
            evaluation_semantics=node["historical_semantics"],
            action_mode=pane["action_mode"],
            policy_rng_steps=policy_rng_steps,
            requested_metrics=evaluation["mandatory_metrics"],
            service_fields=service_outcome_registry(),
            diagnostic_loading=diagnostic_loading,
            capture_bounds=bounds,
            evaluation_world_id=frozen["id"],
            # Fresh seed generation has no pre-reset turn.  Bind the native
            # bridge's observed post-reset turn inside the sealed world result;
            # snapshot cohorts retain their reviewed exact turn.
            expected_reset_turn=(
                None if frozen["snapshot"]["kind"] == "none" else frozen["reset_turn"]
            ),
            bind_observed_reset_turn=frozen["snapshot"]["kind"] == "none",
            observation_core_width=worker_capture["observation_core_width"],
        )
        world_body = _seal(
            {
                "schema": 1,
                "kind": "gate1_fresh_common_producer_world",
                "candidate_digest": candidate["digest"],
                "node": node_id,
                "pane": pane_id,
                "world_id": frozen["id"],
                "result": result,
            }
        )
        world_path = world_root / f"{frozen['id']}.json"
        _write_json(world_path, world_body)
        row_artifacts.append(_ref(world_path, out.parent))
        del result, world_body
    body = _seal(
        {
            "schema": 1,
            "kind": "gate1_fresh_common_producer_evaluation",
            "candidate_digest": candidate["digest"],
            "node": node_id,
            "pane": pane_id,
            "checkpoint_sha256": node["artifact"]["sha256"],
            "worker_capture_sha256": checkpoint_hash(worker_path),
            "producer_scope": "fresh-common-producer-diagnostic",
            "node_evidence_scope": node.get("composition_scope", "recovered-artifact"),
            "historical_artifact_claim": node.get("historical_artifact_claim", True),
            "historical_producer_status": "unresolved",
            "historical_replay_claim": False,
            "selection_allowed": False,
            "readiness_claimed": False,
            "world_count": len(row_artifacts),
            "world_artifacts": row_artifacts,
            "streaming": {
                "maximum_in_memory_world_results": 1,
                "canonical_capture_bound_per_world_bytes": bounds["max_canonical_bytes"],
            },
        }
    )
    _write_json(out, body)
    return body


def _review(candidate: dict[str, Any], review_path: Path) -> None:
    review = _validate_seal(
        json.loads(review_path.read_text(encoding="utf-8")), "independent_gate1_freeze_review"
    )
    approved_reviewers = {
        "OpenAI gpt-6-astra/high": "independent-architecture-reviewer",
        "user-designated-root-reviewer": "user-designated-root-reviewer",
    }
    if (
        approved_reviewers.get(review.get("reviewer_identity")) != review.get("reviewer_role")
        or review.get("decision") != "approved"
        or review.get("candidate_digest") != candidate["digest"]
        or review.get("production_gate1") != "HOLD"
    ):
        _fail("independent review does not approve this exact freeze candidate")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build/validate the Gate-1 production freeze candidate"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--rl-root", type=Path, required=True)
    build.add_argument("--retrieval", type=Path, required=True)
    build.add_argument("--t2m-retrieval", type=Path, required=True)
    build.add_argument("--evidence-package", type=Path, required=True)
    build.add_argument("--evidence-review", type=Path, required=True)
    build.add_argument("--expert-expansion-evidence", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed-start", type=int, required=True)
    materialize = sub.add_parser("materialize-vm")
    materialize.add_argument("--portable-candidate", type=Path, required=True)
    materialize.add_argument("--rl-root", type=Path, required=True)
    materialize.add_argument("--retrieval", type=Path, required=True)
    materialize.add_argument("--t2m-retrieval", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--candidate", type=Path, required=True)
    compare = sub.add_parser("compare-contract8")
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--review", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare_guarded = sub.add_parser("compare-guarded-bc")
    compare_guarded.add_argument("--candidate", type=Path, required=True)
    compare_guarded.add_argument("--review", type=Path, required=True)
    compare_guarded.add_argument("--out", type=Path, required=True)
    capture = sub.add_parser("capture-current-worker")
    capture.add_argument("--candidate", type=Path, required=True)
    capture.add_argument("--review", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    execute = sub.add_parser("evaluate-reviewed")
    execute.add_argument("--candidate", type=Path, required=True)
    execute.add_argument("--review", type=Path, required=True)
    execute.add_argument("--worker", type=Path, required=True)
    execute.add_argument("--contract8-result", type=Path, required=True)
    execute.add_argument("--guarded-bc-result", type=Path, required=True)
    execute.add_argument("--node", required=True)
    execute.add_argument("--pane", required=True)
    execute.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        path = build_candidate(
            args.rl_root,
            args.retrieval,
            args.t2m_retrieval,
            args.evidence_package,
            args.evidence_review,
            args.expert_expansion_evidence,
            args.output,
            args.seed_start,
        )
        print(path)
        return 0
    if args.command == "materialize-vm":
        path = materialize_vm_candidate(
            args.portable_candidate,
            args.rl_root,
            args.retrieval,
            args.t2m_retrieval,
            args.output,
        )
        print(path)
        return 0
    result = validate_candidate(args.candidate)
    if args.command == "validate":
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "capture-current-worker":
        body = capture_current_worker(args.candidate, args.review, args.out)
        print(json.dumps({"path": str(args.out), "digest": body["digest"]}, sort_keys=True))
        return 0
    if args.command == "evaluate-reviewed":
        body = evaluate_reviewed_node(
            args.candidate,
            args.review,
            args.worker,
            args.contract8_result,
            args.guarded_bc_result,
            args.node,
            args.pane,
            args.out,
        )
        print(json.dumps({"path": str(args.out), "digest": body["digest"]}, sort_keys=True))
        return 0
    candidate, base = _load_reviewed_candidate(args.candidate, args.review)
    proposals = candidate["proposals"]
    dag = _read_ref(proposals["artifact_dag"], base, kind="gate1_artifact_dag_proposal")
    if args.command == "compare-guarded-bc":
        node = next(node for node in dag["nodes"] if node["id"] == "t2m-guarded-bc-seed6")
        checkpoint = Path(candidate["inputs"]["t2m_retrieval_root"]) / node["artifact"]["path"]
        corpus_ref = _adapter_ref(
            proposals["guarded_bc_corpus"], base, kind="gate1_guarded_bc_corpus"
        )
        protocol_ref = _adapter_ref(
            proposals["guarded_bc_adapter"], base, kind="gate1_guarded_bc_adapter_protocol"
        )
        output = run_guarded_bc_comparator(checkpoint, corpus_ref, protocol_ref, base=base)
    else:
        prior = next(node for node in dag["nodes"] if node["id"] == "prior-reference")
        checkpoint = Path(candidate["inputs"]["retrieval_root"]) / prior["artifact"]["path"]
        corpus_ref = _adapter_ref(
            proposals["contract8_corpus"], base, kind="gate1_contract8_corpus"
        )
        protocol_ref = _adapter_ref(
            proposals["contract8_comparator"], base, kind="gate1_contract8_comparator_protocol"
        )
        output = run_contract8_comparator(checkpoint, corpus_ref, protocol_ref, base=base)
    _write_json(args.out, output)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
