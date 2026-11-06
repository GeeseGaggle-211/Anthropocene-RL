"""Branch and label provenance (docs/MODEL9_M7.md, *Round 3 build: heads, diagnostics, provenance*; review 5.5(2)).

Every search label row and every critic branch shard records where it came from: the source checkpoint's content hash,
the world (seed, withheld mask, warm-up), the decision turn, the alternative kind, the personality it was scored at,
whether it played a calibration tail, and the learner code commit. Loaders group what they load by (checkpoint, code
commit) and refuse, or flag, a load that mixes sources or contains rows of unknown source (files written before
provenance existed):

* ``reject`` (the ppo and train CLI default): raise ``ProvenanceError``;
* ``flag`` (the library default, so existing tools keep loading old files): warn and describe the mix in the report;
* ``allow``: report only.

A deliberate mix, such as last round's branch shards beside this round's, is loaded with ``flag`` and its report kept.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROVENANCE_VERSION = 2
UNKNOWN = "unknown"
POLICIES = ("reject", "flag", "allow")
REPOSITORY = Path(__file__).resolve().parents[3]


class ProvenanceError(ValueError):
    """A load mixes label or branch sources, or contains rows whose source is unknown."""


def _file_hash(path: str, size: int, mtime: float) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hash(path: str | Path | None) -> str:
    """The full checkpoint SHA-256 (content, not path), or ``unknown``."""
    if path is None:
        return UNKNOWN
    p = Path(path)
    try:
        stat = p.stat()
    except OSError:
        return UNKNOWN
    return _file_hash(str(p.resolve()), stat.st_size, stat.st_mtime)


def code_commit() -> str:
    """The learner code commit: ANTHROPOCENE_CODE_COMMIT when set (detached run checkouts record it in PROVENANCE),
    else the local Git commit with a content digest of any dirty code, else ``unknown``. Failed Git calls never
    produce an apparently clean identity. Generated datasets and run outputs are outside the code digest."""
    try:
        head = subprocess.run(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        if not head:
            return UNKNOWN
        code_paths = [
            "src",
            "integration",
            "configs",
            "scripts",
            "tools",
            "tests",
            "docs",
            "pyproject.toml",
        ]
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY),
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                *code_paths,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        untracked = subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY),
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                *code_paths,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.splitlines()
        if dirty or untracked:
            diff = subprocess.run(
                ["git", "-C", str(REPOSITORY), "diff", "HEAD", "--binary", "--", *code_paths],
                capture_output=True,
                timeout=10,
                check=True,
            ).stdout
            digest = hashlib.sha256(diff)
            for name in sorted(untracked):
                digest.update(name.encode())
                digest.update((REPOSITORY / name).read_bytes())
            return head + "+dirty:" + digest.hexdigest()
        expected = os.environ.get("ANTHROPOCENE_CODE_COMMIT")
        return head if not expected or head == expected else UNKNOWN
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def label_identity(row: dict) -> tuple[str, str]:
    """One main-path search decision; its winning alternative is deliberately excluded from identity."""
    context = {
        k: row[k]
        for k in ("source_run", "seed", "withheld", "warmup", "bank_entry", "snapshot_hash")
    }
    trajectory = digest(context)
    return trajectory, digest(
        {"trajectory": trajectory, "turn": int(row["turn"]), "camp": int(row["camp"])}
    )


def normalize_label_identity(rows: dict, file_hash: str, strict: bool) -> tuple[list[str], bool]:
    import numpy as np
    from .protocol import seeds

    if "seed" in rows:
        seeds(rows["seed"], "search labels")
    fields = (
        "source_run",
        "seed",
        "withheld",
        "warmup",
        "bank_entry",
        "snapshot_hash",
        "turn",
        "camp",
    )
    complete = all(k in rows for k in fields)
    if strict and not complete:
        raise ProvenanceError("search labels lack attributable decision/family identity")
    ids = []
    for i in range(len(rows["label_kind"])):
        if complete:
            row = {
                k: rows[k][i].item() if isinstance(rows[k][i], np.generic) else rows[k][i]
                for k in fields
            }
            trajectory, identity = label_identity(row)
            if ("decision_id" in rows and str(rows["decision_id"][i]) != identity) or (
                "trajectory_id" in rows and str(rows["trajectory_id"][i]) != trajectory
            ):
                raise ProvenanceError(
                    "search decision identity differs from its attributable fields"
                )
        else:
            trajectory, identity = "unknown", digest({"legacy_file": file_hash, "row": i})
        ids.append(identity)
    if complete:
        if "source_seed" in rows and not np.array_equal(rows["source_seed"], rows["seed"]):
            raise ProvenanceError("search source_seed differs from decision seed")
        if "source_snapshot" in rows and not np.array_equal(
            rows["source_snapshot"].astype(str), rows["snapshot_hash"].astype(str)
        ):
            raise ProvenanceError("search source snapshot differs from decision snapshot")
        if any(
            int(b) >= 0 and not re.fullmatch(r"[0-9a-f]{64}", str(s))
            for b, s in zip(rows["bank_entry"], rows["snapshot_hash"])
        ):
            raise ProvenanceError("daughter search decisions require a full snapshot SHA-256")
        rows["source_seed"] = np.asarray(rows["seed"], np.int64)
        rows["source_snapshot"] = np.asarray(rows["snapshot_hash"]).astype(str)
    rows["decision_id"] = np.asarray(ids)
    return ids, complete


def compatibility_mapping(path: Path | None, flags: dict) -> list[dict]:
    """Only an explicit frozen comparator attestation can bridge distinct immutable binaries."""
    mappings = json.loads(Path(path).read_text()).get("simulator_compatibility", []) if path else []
    for entry in mappings:
        identities = (entry.get("source", {}), entry.get("live", {}))
        if (
            any(
                set(identity) != {"commit", "buildHash", "clean"}
                or identity.get("clean") is not True
                or not re.fullmatch(r"[0-9a-f]{40}", str(identity.get("commit", "")))
                or not re.fullmatch(r"[0-9a-f]{32,64}", str(identity.get("buildHash", "")))
                for identity in identities
            )
            or not str(entry.get("reason", "")).strip()
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("evidence_digest", "")))
            or entry.get("flags") != flags
        ):
            raise ProvenanceError(
                "invalid pre-registered simulator compatibility mapping or frozen flags"
            )
        from .protocol import receipt

        try:
            ref = entry.get("evidence", {})
            if ref.get("sha256") != entry["evidence_digest"]:
                raise ProvenanceError("comparator evidence digest does not bind its content")
            body = receipt(
                ref,
                {
                    "kind": "simulator_semantic_comparison",
                    "source": entry["source"],
                    "live": entry["live"],
                    "flags": flags,
                },
                base=Path(path).parent,
            )
            comparisons = body.get("comparisons")
            if (
                not isinstance(comparisons, list)
                or not comparisons
                or any(
                    row.get("equal") is not True
                    or not row.get("seed")
                    or not row.get("turns")
                    or row.get("source_fingerprint") != row.get("live_fingerprint")
                    or not row.get("source_fingerprint")
                    for row in comparisons
                )
            ):
                raise ProvenanceError("comparator lacks complete equal replay evidence")
            from .protocol import artifact

            evidence_path, _ = artifact(ref, Path(path).parent)
            declared_outputs = {output.get("sha256") for output in body["outputs"]}
            from .semantic_trace import verify, complete_payload, same_experiment

            for row in comparisons:
                loaded = []
                for side, identity in (("source", entry["source"]), ("live", entry["live"])):
                    trace_ref = row.get(side + "_trace", {})
                    trace_path, trace_sha = artifact(trace_ref, evidence_path.parent)
                    if trace_sha not in declared_outputs:
                        raise ProvenanceError(
                            "comparator trace omitted from complete output inventory"
                        )
                    trace = json.loads(trace_path.read_text())
                    loaded.append((trace, trace_path.parent, identity, side))
                same_experiment(
                    loaded[0][0], loaded[0][1], loaded[1][0], loaded[1][1], declared_outputs
                )
                traces = [
                    verify(trace, base, identity, flags, row, side, declared_outputs)
                    for trace, base, identity, side in loaded
                ]
                if any(
                    complete_payload(a.read_bytes())[0] != complete_payload(b.read_bytes())[0]
                    for a, b in zip(*traces)
                ):
                    raise ProvenanceError("simulator complete normalized save payloads differ")
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ProvenanceError(
                "invalid simulator compatibility mapping evidence: " + str(error)
            ) from error
    return mappings


def compatible_sources(
    records: Iterable[dict], simulator: dict, mappings=(), flags: dict | None = None
) -> dict:
    records = list(records)
    validate_records(records, "reject")
    if flags is not None and any(
        {key: record.get("flags", record.get("search_config", {})).get(key) for key in flags}
        != flags
        for record in records
    ):
        raise ProvenanceError("offline simulator flags differ from the frozen live rules/statics")
    used = [
        entry
        for record in records
        for entry in mappings
        if entry["source"] == record["simulator"] and entry["live"] == simulator
    ]
    if any(
        record["simulator"] != simulator
        and not any(
            entry["source"] == record["simulator"] and entry["live"] == simulator
            for entry in mappings
        )
        for record in records
    ):
        raise ProvenanceError("offline inputs differ from the frozen live simulator commit/build")
    return {
        "sources": len(records),
        "simulator": simulator,
        "flags": flags,
        "compatible": True,
        "compatibility_mappings": used,
    }


def source_exclusion(record: dict) -> dict:
    raw = record.get("search_config", {}).get("calibration_manifest") or {}
    if not isinstance(raw, dict):
        raise ProvenanceError("source exclusion manifest must be embedded, not a historical path")
    if raw:
        from .calibration import manifest

        normalized = manifest(raw)
        if raw.get("manifest_digest") != normalized["manifest_digest"]:
            raise ProvenanceError("source exclusion manifest digest is inconsistent")
        return normalized
    return {}


def simulator_identity(schema, production: bool = False) -> dict:
    """Identity embedded while compiling the same bridge binary that supplied this live schema."""
    record = {
        "buildHash": schema.build_hash,
        "commit": schema.simulator_commit,
        "clean": schema.simulator_source_clean,
    }
    pin = os.environ.get("ANTHROPOCENE_SIM_COMMIT")
    if pin is not None and pin != record["commit"]:
        raise ProvenanceError(
            "live bridge simulator commit differs from ANTHROPOCENE_SIM_COMMIT pin"
        )
    if production and (
        not re.fullmatch(r"[0-9a-f]{40}", record["commit"])
        or not record["clean"]
        or not record["buildHash"]
        or record["buildHash"] == UNKNOWN
    ):
        raise ProvenanceError(
            "production requires a live bridge built from a full clean simulator commit"
        )
    return record


def source(
    checkpoint: str | Path | None, config: dict | None = None, simulator: dict | None = None
) -> dict[str, Any]:
    """The run-level provenance a search writes into every label row and branch shard."""
    from . import contracts

    record = {
        "source_checkpoint": checkpoint_hash(checkpoint),
        "code_commit": code_commit(),
        "provenance_version": PROVENANCE_VERSION,
        "contract": contracts.NATIVE_CONTRACT,
        "utility_version": 2,
        "config_digest": digest(config or {}),
        "search_config": config or {},
        "simulator": simulator or {"buildHash": UNKNOWN, "commit": UNKNOWN},
    }
    record["source_run"] = digest(record)
    return record


def issue(message: str, policy: str) -> None:
    if policy == "reject":
        raise ProvenanceError(message)
    if policy == "flag":
        warnings.warn(message, stacklevel=2)


def validate_records(records: Iterable[dict], policy: str = "flag") -> dict:
    from . import contracts

    records = list(records)
    problems = []
    for record in records:
        from .protocol import scan

        scan(record, "source provenance")
        if record.get("source_kind") == "native_teacher":
            from .teacher_data import validate_source

            try:
                validate_source(record)
            except ProvenanceError:
                problems.append(record.get("source_run", UNKNOWN))
            continue
        body = {k: v for k, v in record.items() if k != "source_run"}
        # Callers pass the run-level record, never branch metadata mixed into the signed content.
        if (
            record.get("provenance_version") != PROVENANCE_VERSION
            or record.get("contract") != contracts.NATIVE_CONTRACT
            or record.get("utility_version") != 2
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("source_checkpoint", "")))
            or not re.fullmatch(r"[0-9a-f]{40}", str(record.get("code_commit", "")))
            or record.get("source_run") != digest(body)
            or record.get("config_digest") != digest(record.get("search_config", {}))
            or not re.fullmatch(r"[0-9a-f]{40}", str(record.get("simulator", {}).get("commit", "")))
            or record.get("simulator", {}).get("clean") is not True
            or record.get("simulator", {}).get("buildHash", UNKNOWN) in (UNKNOWN, "", None)
        ):
            problems.append(record.get("source_run", UNKNOWN))
    if problems:
        issue(f"unattributable or dirty production sources: {problems}", policy)
    builds = {
        (
            r.get("contract", r.get("schemas", {}).get("contract")),
            r.get("utility_version", r.get("schemas", {}).get("utility")),
            r.get("simulator", {}).get("buildHash"),
            r.get("simulator", {}).get("commit"),
        )
        for r in records
    }
    if len(builds) > 1:
        issue("inputs mix incompatible simulator/contract/utility identities", policy)
    return {"identity_valid": not problems and len(builds) <= 1, "invalid_sources": problems}


def declared_sources(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = json.loads(Path(path).read_text()).get("source_runs", [])
    if not ids or not all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) for x in ids):
        raise ProvenanceError("source manifest requires full source_runs SHA-256 identities")
    return set(ids)


def key(record: dict[str, Any] | None) -> tuple[str, str, str]:
    """Checkpoint, learner commit and canonical source-run identity; old records carry ``unknown``."""
    record = record or {}
    return (
        str(record.get("source_checkpoint") or UNKNOWN),
        str(record.get("code_commit") or UNKNOWN),
        str(record.get("source_run") or UNKNOWN),
    )


def verify_link(record: dict, row: dict, policy: str):
    if any(
        str(row.get(k, UNKNOWN)) != str(record.get(k, UNKNOWN))
        for k in ("source_checkpoint", "code_commit", "source_run")
    ):
        issue("row/shard provenance differs from its full source record", policy)


def check(
    sources: Iterable[tuple[str, ...]],
    policy: str = "flag",
    what: str = "rows",
    counts: Iterable[int] | None = None,
    declared: set[str] | None = None,
) -> dict[str, Any]:
    """Apply ``policy`` to the (checkpoint, commit) sources of one load. Returns the report: every distinct source with
    its row count, and whether the load is mixed or holds unknown rows."""
    if policy not in POLICIES:
        raise ValueError(f"unknown provenance policy {policy!r}")
    sources = list(sources)
    counts = list(counts) if counts is not None else [1] * len(sources)
    if len(counts) != len(sources):
        raise ValueError("source counts do not match source identities")
    tally: Counter = Counter()
    for s, n in zip(sources, counts):
        tally[s] += int(n)
    unknown = sum(n for identity, n in tally.items() if UNKNOWN in identity)
    report = {
        "what": what,
        "policy": policy,
        "sources": [
            {
                "source_checkpoint": identity[0],
                "code_commit": identity[1],
                "source_run": identity[2] if len(identity) > 2 else UNKNOWN,
                "rows": n,
            }
            for identity, n in sorted(tally.items())
        ],
        "mixed": len(tally) > 1,
        "unknown_rows": int(unknown),
    }
    report["declared_mix"] = declared is not None and all(
        len(k) > 2 and k[2] in declared for k in tally
    )
    problem = []
    if report["mixed"] and not report["declared_mix"]:
        problem.append(f"{len(tally)} sources")
    if declared is not None and not report["declared_mix"]:
        problem.append("sources outside the frozen declaration")
    if unknown:
        problem.append(f"{unknown} rows of unknown source")
    report["ok"] = not problem
    if problem and policy != "allow":
        message = f"{what}: mixed or unknown provenance ({', '.join(problem)}): {report['sources']}"
        if policy == "reject":
            raise ProvenanceError(
                message + " (load with provenance 'flag' to accept a deliberate mix)"
            )
        warnings.warn(message, stacklevel=2)
    return report
