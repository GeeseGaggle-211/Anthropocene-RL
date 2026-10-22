"""Deterministic aggregation for model9 economic/learner diagnostic traces.

The input is deliberately stricter than an ordinary evaluation log.  A trace is a
content-addressed artifact (manifest.json, transitions.jsonl, updates.jsonl and
done.json); incomplete files are never summarized.  All correlations emitted by
this module are descriptive.  Causal effects require an explicitly identified
matched-fork record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
EVIDENCE_CLASSES = {
    "original_training_trace",
    "checkpoint_evaluation",
    "instrumented_replay_reconstruction",
    "prospective_diagnostic_training_trace",
}
CONFIRMATION_SEEDS = range(3000, 3060)
PHASES = (
    "nativeDemand",
    "actorCappedRequest",
    "compilerGrant",
    "executedWork",
    "physicalReceipt",
    "finalService",
)
IDENTITY_COLUMNS = (
    "evidenceClass",
    "runId",
    "arm",
    "optimizerSeed",
    "worldSeed",
    "trajectoryId",
    "decisionTurn",
    "campId",
    "transitionId",
)
UPDATE_EVENTS = {
    "mutation-start",
    "mutation-complete",
    "critic-refit-start",
    "critic-refit-complete",
    "iteration-summary",
}


def canonical_bytes(value: Any) -> bytes:
    """Canonical UTF-8 representation used by trace Merkle leaves."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merkle_root(rows: Sequence[dict[str, Any]]) -> str:
    """SHA-256 binary Merkle tree; the final odd node is duplicated."""
    if not rows:
        return hashlib.sha256(b"").hexdigest()
    level = [hashlib.sha256(canonical_bytes(row)).digest() for row in rows]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            output.append(value)
    return output


def _artifact_file(manifest: dict[str, Any], kind: str) -> str:
    files = manifest.get("files", {})
    if isinstance(files, dict) and isinstance(files.get(kind), str):
        return str(files[kind])
    value = manifest.get(kind)
    if isinstance(value, str):
        return value
    return f"{kind}.jsonl"


def _verify_payload(path: Path, rows: list[dict[str, Any]], proof: Any) -> None:
    if not isinstance(proof, dict):
        raise ValueError(f"missing completion proof for {path.name}")
    if proof.get("rows") != len(rows):
        raise ValueError(f"completion row count differs for {path.name}")
    expected = proof.get("sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected.lower():
        raise ValueError(f"completion digest differs for {path.name}")


def load_artifact(
    path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(path)
    manifest_path, done_path = root / "manifest.json", root / "done.json"
    if not manifest_path.is_file() or not done_path.is_file():
        raise ValueError(f"incomplete economic trace artifact: {root}")
    manifest, done = _json(manifest_path), _json(done_path)
    if (
        manifest.get("schemaVersion") != SCHEMA_VERSION
        or done.get("schemaVersion") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported economic trace artifact schema")
    transition_path = root / _artifact_file(manifest, "transitions")
    update_path = root / _artifact_file(manifest, "updates")
    if not transition_path.is_file() or not update_path.is_file():
        raise ValueError(f"economic trace payload is incomplete: {root}")
    transitions, updates = _jsonl(transition_path), _jsonl(update_path)
    _verify_payload(transition_path, transitions, done.get("transitions"))
    _verify_payload(update_path, updates, done.get("updates"))
    expected_root = done.get("merkleRoot")
    actual_root = merkle_root([*transitions, *updates])
    if not isinstance(expected_root, str) or expected_root.lower() != actual_root:
        raise ValueError("economic trace Merkle root differs")
    manifest = dict(manifest)
    manifest["artifactPath"] = str(root.resolve())
    manifest["manifestSha256"] = sha256_file(manifest_path)
    manifest["doneSha256"] = sha256_file(done_path)
    return manifest, transitions, updates


def _finite_numbers(value: Any, path: str = "") -> Iterator[tuple[str, float]]:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"nonfinite diagnostic value: {path}")
        yield path, number
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _finite_numbers(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _finite_numbers(child, f"{path}[{index}]")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _require_nonnegative(record: dict[str, Any], names: Iterable[str], label: str) -> None:
    for name in names:
        if name not in record or record[name] is None:
            continue
        value = _number(record[name])
        if value is None or value < 0:
            raise ValueError(f"{label}.{name} must be finite and nonnegative")


def _unavailable_reason(row: dict[str, Any], economic: dict[str, Any], path: str) -> str | None:
    candidates = [path]
    prefix = "economicDiagnostics."
    if path.startswith(prefix):
        candidates.append(path[len(prefix) :])
    for reasons in (economic.get("unavailable", {}), row.get("unavailable", {})):
        if not isinstance(reasons, dict):
            continue
        for candidate in candidates:
            reason = reasons.get(candidate)
            if isinstance(reason, str) and reason.strip():
                return reason
    return None


def _null_paths(value: Any, path: str = "") -> Iterator[str]:
    if value is None:
        yield path
    elif isinstance(value, dict):
        for key, child in value.items():
            if key == "unavailable":
                continue
            yield from _null_paths(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _null_paths(child, f"{path}[{index}]")


def validate_transition(row: dict[str, Any]) -> None:
    required = {
        "schemaVersion": int,
        "transitionId": str,
        "trajectoryId": str,
        "decisionTurn": int,
        "campId": int,
        "societyId": int,
        "runId": str,
        "arm": str,
        "optimizerSeed": int,
        "worldSeed": int,
        "policyVersion": int,
        "evidenceClass": str,
        "economicDiagnostics": dict,
        "objectives": dict,
        "provenance": dict,
    }
    for name, kind in required.items():
        if not isinstance(row.get(name), kind) or isinstance(row.get(name), bool):
            raise ValueError(f"economic trace missing/invalid {name}")
    if row["schemaVersion"] != SCHEMA_VERSION:
        raise ValueError("unsupported economic transition schema")
    if row["evidenceClass"] not in EVIDENCE_CLASSES:
        raise ValueError("unknown economic trace evidence class")
    if row["optimizerSeed"] in CONFIRMATION_SEEDS or row["worldSeed"] in CONFIRMATION_SEEDS:
        raise ValueError("confirmation seed appears in economic diagnostic trace")
    years, gamma, lam = (_number(row.get(name)) for name in ("years", "gamma", "gaeLambda"))
    if years is None or years <= 0 or gamma is None or not 0 <= gamma <= 1:
        raise ValueError("economic transition duration/discount is invalid")
    if lam is None or not 0 <= lam <= 1:
        raise ValueError("economic transition lambda is invalid")
    if not isinstance(row.get("terminated"), bool) or not isinstance(row.get("truncated"), bool):
        raise ValueError("economic transition termination flags are invalid")
    economic = row["economicDiagnostics"]
    for name in ("transitionId", "trajectoryId", "decisionTurn", "campId", "societyId"):
        if economic.get(name) != row[name]:
            raise ValueError(f"economic sidecar identity differs: {name}")
    if economic.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unsupported economic sidecar schema")
    for _path, _value in _finite_numbers(row):
        pass
    for path in _null_paths(economic, "economicDiagnostics"):
        if _unavailable_reason(row, economic, path) is None:
            raise ValueError(f"null economic field has no unavailable reason: {path}")
    for good in economic.get("goods", []):
        if not isinstance(good, dict) or "good" not in good:
            raise ValueError("invalid physical good diagnostic")
        _require_nonnegative(
            good,
            (
                "usefulDemand",
                "plannedOutputCeiling",
                "output",
                "workerIntervals",
                "workerYears",
                "equipmentCoveredWork",
                "stone",
                "clay",
                "wood",
                "plantFood",
                "animalFood",
                "animalFabric",
                "cloth",
                "housingMaterial",
                "tools",
            ),
            "goods",
        )
    for material in economic.get("materials", []):
        if not isinstance(material, dict) or "material" not in material:
            raise ValueError("invalid material diagnostic")
        _require_nonnegative(
            material,
            (
                "requested",
                "granted",
                "received",
                "workerIntervals",
                "workerYears",
                "extractionRate",
            ),
            "materials",
        )
        requested, granted, received = (
            _number(material.get(name)) for name in ("requested", "granted", "received")
        )
        if requested is not None and granted is not None and granted > requested + 1e-8:
            raise ValueError("material grant exceeds request")
        if granted is not None and received is not None and received > granted + 1e-8:
            raise ValueError("material receipt exceeds grant")
    for trade in economic.get("trades", []):
        if not isinstance(trade, dict):
            raise ValueError("invalid trade diagnostic")
        _require_nonnegative(
            trade,
            (
                "requestedQuantity",
                "agreedQuantity",
                "deliveredQuantity",
                "acceptedQuantity",
                "stockQuantity",
                "madeToOrderQuantity",
                "distance",
            ),
            "trades",
        )


def validate_update(row: dict[str, Any], transition_ids: set[str]) -> None:
    """Validate an optimizer event and its exact join to pre-update transitions."""
    required = {
        "schemaVersion": int,
        "event": str,
        "runId": str,
        "arm": str,
        "optimizerSeed": int,
        "rollout": int,
        "evidenceClass": str,
    }
    for name, kind in required.items():
        if not isinstance(row.get(name), kind) or isinstance(row.get(name), bool):
            raise ValueError(f"optimizer trace missing/invalid {name}")
    if row["schemaVersion"] != SCHEMA_VERSION or row["event"] not in UPDATE_EVENTS:
        raise ValueError("unsupported optimizer trace schema/event")
    if row["evidenceClass"] not in EVIDENCE_CLASSES:
        raise ValueError("unknown optimizer trace evidence class")
    if row["optimizerSeed"] in CONFIRMATION_SEEDS:
        raise ValueError("confirmation seed appears in optimizer diagnostic trace")
    for _path, _value in _finite_numbers(row):
        pass

    event = row["event"]
    if event == "iteration-summary":
        if not isinstance(row.get("metrics"), dict):
            raise ValueError("iteration summary has no metrics")
        return

    for name in ("epoch",):
        if not isinstance(row.get(name), int) or isinstance(row.get(name), bool):
            raise ValueError(f"optimizer event missing/invalid {name}")
    batch_ids = row.get("batchTransitionIds")
    if not isinstance(batch_ids, list) or not batch_ids or not all(
        isinstance(identity, str) for identity in batch_ids
    ):
        raise ValueError("optimizer event has no exact batch transition identities")
    if len(batch_ids) != len(set(batch_ids)):
        raise ValueError("optimizer event repeats a transition identity")
    if any(identity not in transition_ids for identity in batch_ids):
        raise ValueError("optimizer event references a transition outside its artifact")
    expected_digest = hashlib.sha256(canonical_bytes(batch_ids)).hexdigest()
    if row.get("batchRowDigest") != expected_digest:
        raise ValueError("optimizer event batch-row digest differs")
    if event.startswith("mutation-") and (
        not isinstance(row.get("minibatch"), int)
        or isinstance(row.get("minibatch"), bool)
    ):
        raise ValueError("optimizer mutation has no minibatch identity")
    if event.endswith("complete"):
        for group in ("gradientMetrics",):
            values = row.get(group)
            if not isinstance(values, dict):
                raise ValueError(f"optimizer completion has no {group}")
            for name, value in values.items():
                number = _number(value)
                if number is None or number < 0:
                    raise ValueError(f"optimizer completion has invalid {group}.{name}")
                if name.endswith("ClippingScale") and number > 1:
                    raise ValueError("optimizer clipping scale exceeds one")
        if event == "mutation-complete":
            for name in ("losses", "parameterUpdateNorms", "afterHashes"):
                if not isinstance(row.get(name), dict):
                    raise ValueError(f"optimizer mutation has no {name}")
            _require_nonnegative(
                row["parameterUpdateNorms"], row["parameterUpdateNorms"], "updates"
            )


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(name, "") for name in IDENTITY_COLUMNS)


def load_traces(
    paths: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifests: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted((Path(value) for value in paths), key=lambda value: str(value.resolve())):
        manifest, rows, mutation_rows = load_artifact(path)
        manifests.append(manifest)
        for row in rows:
            validate_transition(row)
            identity = str(row["transitionId"])
            if identity in seen:
                raise ValueError(f"duplicate economic transition: {identity}")
            seen.add(identity)
            transitions.append(row)
        artifact_ids = {str(row["transitionId"]) for row in rows}
        for update in mutation_rows:
            validate_update(update, artifact_ids)
        updates.extend(mutation_rows)
    transitions.sort(key=_sort_key)
    updates.sort(key=lambda row: canonical_bytes(row))
    return manifests, transitions, updates


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten(value[key], path))
    elif isinstance(value, list):
        output[prefix] = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    else:
        output[prefix] = value
    return output


def flatten_fields(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten collection members for schema/unit inspection (unlike CSV serialization)."""
    output: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_fields(value[key], path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.update(flatten_fields(child, f"{prefix}[{index}]"))
    else:
        output[prefix] = value
    return output


def _update_family(event: str) -> str | None:
    if event.startswith("mutation-"):
        return "mutation"
    if event.startswith("critic-refit-"):
        return "critic-refit"
    return None


def optimizer_update_rows(
    transitions: Sequence[dict[str, Any]], updates: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join mutation boundaries to their exact pre-update transition batch."""
    transition_by_id = {str(row["transitionId"]): row for row in transitions}
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in updates:
        family = _update_family(str(row.get("event", "")))
        if family is None:
            continue
        key = (
            row["runId"],
            row["arm"],
            row["optimizerSeed"],
            row["rollout"],
            row["epoch"],
            row.get("minibatch", 0),
            row["batchRowDigest"],
            family,
        )
        side = "start" if str(row["event"]).endswith("-start") else "complete"
        if side in pairs[key]:
            raise ValueError(f"duplicate optimizer {side} boundary: {key}")
        pairs[key][side] = row

    output: list[dict[str, Any]] = []
    for key in sorted(pairs, key=lambda value: tuple(str(item) for item in value)):
        pair = pairs[key]
        if set(pair) != {"start", "complete"}:
            raise ValueError(f"unpaired optimizer mutation boundary: {key}")
        start, complete = pair["start"], pair["complete"]
        if start["batchTransitionIds"] != complete["batchTransitionIds"]:
            raise ValueError("optimizer mutation boundaries reference different batch rows")
        run, arm, seed, rollout, epoch, minibatch, digest, family = key
        batch = [transition_by_id[identity] for identity in start["batchTransitionIds"]]
        if any(
            row.get("runId") != run
            or row.get("arm") != arm
            or row.get("optimizerSeed") != seed
            or row.get("rollout") != rollout
            or row.get("evidenceClass") != complete["evidenceClass"]
            for row in batch
        ):
            raise ValueError("optimizer event identity differs from its transition batch")
        result: dict[str, Any] = {
            "evidenceClass": complete["evidenceClass"],
            "runId": run,
            "arm": arm,
            "optimizerSeed": seed,
            "rollout": rollout,
            "epoch": epoch,
            "minibatch": minibatch,
            "updateKind": family,
            "batchRowDigest": digest,
            "batchTransitionCount": len(batch),
            "batchTransitionIds": json.dumps(
                start["batchTransitionIds"], separators=(",", ":"), ensure_ascii=False
            ),
            "policyVersionBefore": start.get("policyVersionBefore"),
            "policyVersionAfter": complete.get("policyVersionAfter"),
            "constraintRate": float(np.mean([bool(_constraint_reasons(row)) for row in batch])),
        }
        before = start.get("beforeHashes", {})
        after = complete.get("afterHashes", {})
        for component in sorted(set(before) | set(after)):
            result[f"{component}HashBefore"] = before.get(component)
            result[f"{component}HashAfter"] = after.get(component)
            result[f"{component}HashChanged"] = (
                before.get(component) != after.get(component)
                if before.get(component) is not None and after.get(component) is not None
                else None
            )
        for name in (
            "ppoRatioMean",
            "ppoClipFraction",
            "approxKl",
            "losses",
            "gradientMetrics",
            "parameterUpdateNorms",
            "unavailable",
        ):
            if name in complete:
                result.update(flatten(complete[name], name))
        objective_names = sorted(
            {
                str(name)
                for row in batch
                for name in row.get("objectives", {})
                if isinstance(row.get("objectives"), dict)
            }
        )
        for objective in objective_names:
            values = [
                row["objectives"][objective]
                for row in batch
                if isinstance(row.get("objectives", {}).get(objective), dict)
            ]
            for field in ("prediction", "target", "tdResidual", "advantage"):
                numbers = [
                    number
                    for value in values
                    if (number := _number(value.get(field))) is not None
                ]
                if numbers:
                    result[f"batch.{objective}.{field}Mean"] = float(np.mean(numbers))
                    result[f"batch.{objective}.{field}AbsMean"] = float(
                        np.mean(np.abs(numbers))
                    )
        output.append(result)
    return output


def optimizer_update_rollup(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate mutations by arm and by optimizer seed without inflating seed n."""
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        common = (row["evidenceClass"], row["arm"], row["updateKind"])
        groups[(*common, "overall", "all")].append(row)
        groups[(*common, "optimizer_seed", str(row["optimizerSeed"]))].append(row)
    excluded = {
        "optimizerSeed",
        "rollout",
        "epoch",
        "minibatch",
        "policyVersionBefore",
        "policyVersionAfter",
    }
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        evidence, arm, update_kind, slice_name, slice_value = key
        records = groups[key]
        result: dict[str, Any] = {
            "evidenceClass": evidence,
            "arm": arm,
            "updateKind": update_kind,
            "slice": slice_name,
            "sliceValue": slice_value,
            "mutationCount": len(records),
            "optimizerSeedCount": len({row["optimizerSeed"] for row in records}),
        }
        numeric_names = sorted(
            {
                name
                for row in records
                for name, value in row.items()
                if name not in excluded
                and not isinstance(value, bool)
                and _number(value) is not None
                and (
                    name.startswith("losses.")
                    or name.startswith("gradientMetrics.")
                    or name.startswith("parameterUpdateNorms.")
                    or name.startswith("batch.")
                    or name in {"ppoRatioMean", "ppoClipFraction", "approxKl", "constraintRate"}
                )
            }
        )
        for name in numeric_names:
            values = [_number(row.get(name)) for row in records]
            present = [value for value in values if value is not None]
            if present:
                result[f"{name}.mean"] = float(np.mean(present))
        for component in ("actor", "critic", "cost"):
            values = [
                row.get(f"{component}HashChanged")
                for row in records
                if isinstance(row.get(f"{component}HashChanged"), bool)
            ]
            result[f"{component}HashChangedRate"] = (
                float(np.mean(values)) if values else None
            )
        critic_updates = [
            _number(row.get("parameterUpdateNorms.critic")) for row in records
        ]
        known_critic_updates = [value for value in critic_updates if value is not None]
        result["zeroCriticUpdateRate"] = (
            float(np.mean([value == 0 for value in known_critic_updates]))
            if known_critic_updates
            else None
        )
        result["insufficient"] = (
            sum(int(row["batchTransitionCount"]) for row in records) < 30
            or result["optimizerSeedCount"] < 3
        )
        output.append(result)
    return output


def _write_csv(
    path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        names = sorted({name for row in rows for name in row})
    else:
        names = list(columns)
        names.extend(sorted({name for row in rows for name in row if name not in names}))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=names, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {name: "" if row.get(name) is None else row.get(name, "") for name in names}
            )


def _source_phase(path: str) -> str:
    if path.startswith("objectives.") or path.startswith("cost."):
        return "learner_pre_update"
    if path.startswith("provenance."):
        return "artifact_provenance"
    if ".goods" in path or ".materials" in path or ".trades" in path:
        return "committed_physical_receipt"
    if path.startswith("economicDiagnostics."):
        return "simulator_sidecar"
    if path.startswith("actor."):
        return "actor_sample"
    return "transition_identity"


def _sign_contract(path: str) -> str:
    lower = path.lower()
    signed = ("residual", "advantage", "utility", "net", "bias", "logprob", "delta")
    if any(token in lower for token in signed):
        return "signed"
    nonnegative = (
        "population",
        "birth",
        "death",
        "years",
        "food",
        "stock",
        "output",
        "input",
        "requested",
        "granted",
        "received",
        "animals",
        "capital",
        "distance",
    )
    return "nonnegative" if any(token in lower for token in nonnegative) else "field_specific"


def field_dictionary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    units: dict[str, set[str]] = defaultdict(set)
    reasons: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        economic = row.get("economicDiagnostics", {})
        for name, unit in economic.get("units", {}).items():
            if isinstance(unit, str):
                units[str(name)].add(unit)
                units[f"economicDiagnostics.{name}"].add(unit)
        for source in (economic.get("unavailable", {}), row.get("unavailable", {})):
            if isinstance(source, dict):
                for name, reason in source.items():
                    if isinstance(reason, str):
                        reasons[str(name)].add(reason)
                        reasons[f"economicDiagnostics.{name}"].add(reason)
        flat = flatten_fields(row)
        for name, value in flat.items():
            cell = cells.setdefault(
                name,
                {
                    "field": name,
                    "sourcePhase": _source_phase(name),
                    "signRange": _sign_contract(name),
                    "present": 0,
                    "null": 0,
                    "unbounded": 0,
                    "types": set(),
                    "minimum": None,
                    "maximum": None,
                },
            )
            cell["present"] += 1
            if value is None:
                cell["null"] += 1
                continue
            cell["types"].add(type(value).__name__)
            number = _number(value)
            if number is not None:
                if flat.get(f"{name}BoundState") in {
                    "positive-unbounded",
                    "negative-unbounded",
                }:
                    cell["unbounded"] += 1
                    continue
                cell["minimum"] = (
                    number if cell["minimum"] is None else min(cell["minimum"], number)
                )
                cell["maximum"] = (
                    number if cell["maximum"] is None else max(cell["maximum"], number)
                )
    fields = []
    for name in sorted(cells):
        cell = cells[name]
        cell["types"] = sorted(cell["types"])
        cell["units"] = sorted(units.get(name, set()))
        cell["unavailableReasons"] = sorted(reasons.get(name, set()))
        denominator = f"{name}Denominator"
        cell["denominatorField"] = denominator if denominator in cells else None
        fields.append(cell)
    return {"schemaVersion": SCHEMA_VERSION, "rows": len(rows), "fields": fields}


def _phase_unit(item: dict[str, Any], phase: str, default: str | None) -> str | None:
    phase_units = item.get("phaseUnits")
    if isinstance(phase_units, dict) and isinstance(phase_units.get(phase), str):
        return str(phase_units[phase])
    return default


def _phase_number(item: dict[str, Any], phase: str) -> float | None:
    if item.get(f"{phase}BoundState") in {"positive-unbounded", "negative-unbounded"}:
        return None
    return _number(item.get(phase))


def _explicit_funnels(domain: str, block: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for item in block.get("funnels", []):
        if not isinstance(item, dict):
            continue
        result = dict(item)
        result["domain"] = domain
        result.setdefault("metric", "service")
        yield result


_PREFIXES = {
    "native": "nativeDemand",
    "requested": "actorCappedRequest",
    "accepted": "compilerGrant",
    "granted": "compilerGrant",
    "executed": "executedWork",
    "realised": "physicalReceipt",
    "realized": "physicalReceipt",
    "fulfilled": "finalService",
}


def _implicit_funnels(domain: str, block: dict[str, Any]) -> Iterator[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for name, value in block.items():
        if isinstance(value, (dict, list, bool)) or _number(value) is None:
            continue
        match = re.match(
            r"^(native|requested|accepted|granted|executed|realised|realized|fulfilled)[_-]?(.*)$",
            name,
            re.I,
        )
        if not match:
            continue
        metric = match.group(2) or "service"
        grouped[metric[:1].lower() + metric[1:]][_PREFIXES[match.group(1).lower()]] = value
    for metric, values in grouped.items():
        if len(values) >= 2:
            yield {"domain": domain, "metric": metric, **values}


def _row_funnels(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    economic = row["economicDiagnostics"]
    for domain, block in economic.get("domains", {}).items():
        if not isinstance(block, dict):
            continue
        yield from _explicit_funnels(str(domain), block)
        yield from _implicit_funnels(str(domain), block)
    for good in economic.get("goods", []):
        if isinstance(good, dict):
            yield {
                "domain": "manufacturing",
                "metric": f"good:{good.get('good')}",
                "unit": good.get("unit", "good_units"),
                "nativeDemand": good.get("usefulDemand"),
                "compilerGrant": good.get("plannedOutputCeiling"),
                "physicalReceipt": good.get("output"),
            }
    for material in economic.get("materials", []):
        if isinstance(material, dict):
            yield {
                "domain": "manufacturing",
                "metric": f"material:{material.get('material')}",
                "unit": material.get("unit", "material_units"),
                "actorCappedRequest": material.get("requested"),
                "compilerGrant": material.get("granted"),
                "physicalReceipt": material.get("received"),
            }


def _loss(before: float, after: float) -> float:
    return (before - after) / before


def domain_summary(
    rows: Sequence[dict[str, Any]], manifests: Sequence[dict[str, Any]] = ()
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        for funnel in _row_funnels(row):
            for slice_name, slice_value in _slices(row, manifests):
                key = (
                    row["evidenceClass"],
                    row["arm"],
                    funnel["domain"],
                    funnel.get("metric", "service"),
                    funnel.get("unit", ""),
                    slice_name,
                    slice_value,
                )
                groups[key].append((row, funnel))
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: tuple(str(part) for part in value)):
        evidence, arm, domain, metric, default_unit, slice_name, slice_value = key
        records = groups[key]
        result: dict[str, Any] = {
            "evidenceClass": evidence,
            "arm": arm,
            "domain": domain,
            "metric": metric,
            "slice": slice_name,
            "sliceValue": slice_value,
            "capabilityRegime": slice_value if slice_name == "capability_regime" else "all",
            "unit": default_unit,
            "count": len(records),
            "optimizerSeedCount": len({row["optimizerSeed"] for row, _ in records}),
        }
        for phase in PHASES:
            values = [_phase_number(item, phase) for _, item in records]
            valid = [value for value in values if value is not None]
            result[f"{phase}Count"] = len(valid)
            result[f"{phase}Sum"] = sum(valid) if valid else None
        for before, after in itertools.pairwise(PHASES):
            ratios: list[float] = []
            before_sum = after_sum = 0.0
            same_unit = True
            denominator_rows = 0
            for _, item in records:
                a, b = _phase_number(item, before), _phase_number(item, after)
                unit_a = _phase_unit(item, before, str(default_unit) if default_unit else None)
                unit_b = _phase_unit(item, after, str(default_unit) if default_unit else None)
                if unit_a != unit_b:
                    same_unit = False
                    continue
                if a is None or b is None:
                    continue
                if a > 0:
                    ratios.append(_loss(a, b))
                    before_sum += a
                    after_sum += b
                    denominator_rows += 1
            stem = f"{before}To{after[:1].upper() + after[1:]}"
            if not same_unit:
                result[f"{stem}RatioOfSumsLoss"] = None
                result[f"{stem}MeanOfRowLosses"] = None
                result[f"{stem}UndefinedReason"] = "unit-mismatch"
            elif not denominator_rows or before_sum == 0:
                result[f"{stem}RatioOfSumsLoss"] = None
                result[f"{stem}MeanOfRowLosses"] = None
                result[f"{stem}UndefinedReason"] = "zero-denominator"
            else:
                ratio = _loss(before_sum, after_sum)
                result[f"{stem}RatioOfSumsLoss"] = ratio
                result[f"{stem}MeanOfRowLosses"] = float(np.mean(ratios))
                result[f"{stem}UndefinedReason"] = None
                result[f"{stem}LossAbove10Percent"] = ratio > 0.1
        result["insufficient"] = result["count"] < 30 or result["optimizerSeedCount"] < 3
        output.append(result)
    return output


def _personality_band(row: dict[str, Any]) -> str | None:
    explicit = row.get("personalityBand")
    if isinstance(explicit, str):
        return explicit
    personality = row.get("personality")
    if not isinstance(personality, list) or not personality:
        return None
    values = np.asarray(personality, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return None
    return f"dominant-{int(np.argmax(values))}"


def _nested(block: Any, *names: str) -> Any:
    current = block
    for name in names:
        if not isinstance(current, dict):
            return None
        current = current.get(name)
    return current


def _bin(value: float | None, boundaries: Any, prefix: str) -> str | None:
    if value is None or not isinstance(boundaries, list) or len(boundaries) < 2:
        return None
    points = [float(item) for item in boundaries]
    if any(not math.isfinite(item) for item in points) or points != sorted(points):
        return None
    index = int(np.searchsorted(points[1:-1], value, side="right"))
    return f"{prefix}{index + 1}"


def _manifest_option(manifests: Sequence[dict[str, Any]], name: str) -> Any:
    presence = [name in manifest for manifest in manifests]
    if any(presence) and not all(presence):
        raise ValueError(f"trace manifests disagree on presence of {name}")
    values = [manifest.get(name) for manifest in manifests if name in manifest]
    if not values:
        return None
    canonical = {canonical_bytes(value) for value in values}
    if len(canonical) != 1:
        raise ValueError(f"trace manifests disagree on {name}")
    return values[0]


def _constraint_reasons(row: dict[str, Any]) -> list[str]:
    reasons: set[str] = set()
    domains = _nested(row, "economicDiagnostics", "domains")
    if isinstance(domains, dict):
        for block in domains.values():
            if not isinstance(block, dict):
                continue
            for name in ("constraintReasons", "interventionReasons"):
                value = block.get(name, [])
                if isinstance(value, str):
                    reasons.add(value)
                elif isinstance(value, list):
                    reasons.update(str(item) for item in value if item)
            for funnel in block.get("funnels", []):
                if not isinstance(funnel, dict):
                    continue
                value = funnel.get("constraintReasons", [])
                if isinstance(value, str):
                    reasons.add(value)
                elif isinstance(value, list):
                    reasons.update(str(item) for item in value if item)
    return sorted(reasons)


def _slices(row: dict[str, Any], manifests: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    result = [("overall", "all")]
    for name, value in (
        ("optimizer_seed", row.get("optimizerSeed")),
        ("world_seed", row.get("worldSeed")),
        ("capability_regime", row.get("capabilityRegime")),
        ("personality_band", _personality_band(row)),
    ):
        if value is not None:
            result.append((name, str(value)))
    boundaries = _manifest_option(manifests, "sliceBoundaries") or {}
    population_block = _nested(row, "economicDiagnostics", "domains", "population")
    population = None
    qol = None
    if isinstance(population_block, dict):
        population = population_block.get(
            "committedEndingLinked", population_block.get("endingLinked")
        )
        qol = population_block.get("endingQol", population_block.get("qol"))
    for name, value, key, prefix in (
        ("population_quintile", _number(population), "population", "q"),
        ("qol_quintile", _number(qol), "qol", "q"),
        ("turn_band", _number(row.get("decisionTurn")), "turn", "t"),
    ):
        label = _bin(value, boundaries.get(key) if isinstance(boundaries, dict) else None, prefix)
        if label is not None:
            result.append((name, label))
    for reason in _constraint_reasons(row):
        result.append(("intervention_reason", reason))
    return result


def _calibration(values: Sequence[tuple[float, float]]) -> dict[str, Any]:
    prediction = np.asarray([item[0] for item in values], dtype=np.float64)
    target = np.asarray([item[1] for item in values], dtype=np.float64)
    residual = prediction - target
    result: dict[str, Any] = {
        "count": len(values),
        "predictionMean": float(prediction.mean()),
        "targetMean": float(target.mean()),
        "signedBias": float(residual.mean()),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "residualP05": float(np.quantile(residual, 0.05)),
        "residualP50": float(np.quantile(residual, 0.50)),
        "residualP95": float(np.quantile(residual, 0.95)),
    }
    target_var, prediction_var = float(target.var()), float(prediction.var())
    result["explainedVariance"] = 1 - float(residual.var()) / target_var if target_var > 0 else None
    if prediction_var > 0:
        slope = float(
            np.mean((prediction - prediction.mean()) * (target - target.mean())) / prediction_var
        )
        result["slope"] = slope
        result["intercept"] = float(target.mean() - slope * prediction.mean())
    else:
        result["slope"] = None
        result["intercept"] = None
    return result


def critic_calibration(
    rows: Sequence[dict[str, Any]], manifests: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[tuple[dict[str, Any], float, float, float | None]]] = (
        defaultdict(list)
    )
    for row in rows:
        objectives = row.get("objectives", {})
        if not isinstance(objectives, dict):
            continue
        for objective, values in objectives.items():
            if not isinstance(values, dict):
                continue
            prediction = _number(values.get("prediction"))
            targets = [("lambda_target", _number(values.get("target")))]
            independent = _number(values.get("independentReturn"))
            if independent is not None:
                targets.append(("independent_return", independent))
            scale = _number(values.get("referenceScale"))
            for target_class, target in targets:
                if prediction is None or target is None:
                    continue
                for slice_name, slice_value in _slices(row, manifests):
                    key = (
                        row["evidenceClass"],
                        row["arm"],
                        str(objective),
                        target_class,
                        slice_name,
                        slice_value,
                    )
                    groups[key].append((row, prediction, target, scale))
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        evidence, arm, objective, target_class, slice_name, slice_value = key
        records = groups[key]
        result = {
            "evidenceClass": evidence,
            "arm": arm,
            "objective": objective,
            "targetClass": target_class,
            "slice": slice_name,
            "sliceValue": slice_value,
            **_calibration([(prediction, target) for _, prediction, target, _ in records]),
        }
        seeds = {row["optimizerSeed"] for row, *_ in records}
        scales = [scale for *_, scale in records if scale is not None and scale > 0]
        result["optimizerSeedCount"] = len(seeds)
        result["referenceScale"] = float(np.mean(scales)) if scales else None
        result["absoluteBiasAbove10Percent"] = (
            abs(float(result["signedBias"])) > 0.1 * float(result["referenceScale"])
            if result["referenceScale"]
            else None
        )
        result["insufficient"] = result["count"] < 30 or len(seeds) < 3
        output.append(result)
    return output


def _actor_values(row: dict[str, Any]) -> Iterator[tuple[str, float]]:
    # Prospective traces store the bridge action at the transition root.  Keep
    # accepting the older nested fixture shape so checkpoint/reconstruction
    # artifacts can be aggregated without relabelling them.
    source = row.get("sampledAction")
    if not isinstance(source, (dict, list)):
        actor = row.get("actor")
        if not isinstance(actor, dict):
            return
        source = actor.get("sampledAction", actor.get("action", {}))

    def walk(value: Any, path: str = "") -> Iterator[tuple[str, float]]:
        number = _number(value)
        if number is not None:
            yield path, number
        elif isinstance(value, dict):
            for name, child in sorted(value.items()):
                yield from walk(child, f"{path}.{name}" if path else str(name))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, f"{path}[{index}]")

    yield from walk(source)


def _alignment_outcomes(row: dict[str, Any]) -> Iterator[tuple[str, str, str, float]]:
    final = _number(row.get("finalPolicyAdvantage"))
    if final is not None:
        yield "final_policy_advantage", "learner", "combined", final
    objectives = row.get("objectives", {})
    if isinstance(objectives, dict):
        for objective, values in sorted(objectives.items()):
            if not isinstance(values, dict):
                continue
            for field in ("advantage", "tdResidual"):
                value = _number(values.get(field))
                if value is not None:
                    yield f"objective_{field}", "critic", str(objective), value
    for funnel in _row_funnels(row):
        for phase in ("executedWork", "physicalReceipt", "finalService"):
            value = _number(funnel.get(phase))
            if value is not None:
                yield (
                    f"economic_{phase}",
                    str(funnel["domain"]),
                    str(funnel.get("metric", "service")),
                    value,
                )


def action_alignment(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, str, str, str, str], list[tuple[dict[str, Any], float, float]]
    ] = defaultdict(list)
    for row in rows:
        for head, action in _actor_values(row):
            for relationship, domain, metric, outcome in _alignment_outcomes(row):
                groups[
                    (row["evidenceClass"], row["arm"], head, relationship, domain, metric)
                ].append((row, action, outcome))
    output = []
    for (evidence, arm, head, relationship, domain, metric), records in sorted(
        groups.items()
    ):
        action = np.asarray([item[1] for item in records])
        outcome = np.asarray([item[2] for item in records])
        correlation = None
        if len(records) > 1 and action.var() > 0 and outcome.var() > 0:
            correlation = float(np.corrcoef(action, outcome)[0, 1])
        seeds = {item[0]["optimizerSeed"] for item in records}
        output.append(
            {
                "evidenceClass": evidence,
                "arm": arm,
                "head": head,
                "relationship": f"descriptive_action_to_{relationship}",
                "outcomeDomain": domain,
                "outcomeMetric": metric,
                "count": len(records),
                "optimizerSeedCount": len(seeds),
                "pearson": correlation,
                "interventionRate": float(
                    np.mean([bool(_constraint_reasons(item[0])) for item in records])
                ),
                "causal": False,
                "insufficient": len(records) < 30 or len(seeds) < 3,
            }
        )
    return output


def matched_fork_effects(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        matched = row.get("matchedFork")
        if not isinstance(matched, dict) or not isinstance(matched.get("effects"), dict):
            continue
        for metric, effect in sorted(matched["effects"].items()):
            number = _number(effect)
            if number is None:
                continue
            output.append(
                {
                    "evidenceClass": row["evidenceClass"],
                    "arm": row["arm"],
                    "optimizerSeed": row["optimizerSeed"],
                    "worldSeed": row["worldSeed"],
                    "pairId": matched.get("pairId"),
                    "perturbedHead": matched.get("perturbedHead"),
                    "metric": metric,
                    "effect": number,
                    "sameSnapshot": matched.get("sameSnapshot"),
                    "sameRandomStream": matched.get("sameRandomStream"),
                    "causal": bool(matched.get("sameSnapshot") and matched.get("sameRandomStream")),
                }
            )
    return output


def paired_arm_contrasts(
    rows: Sequence[dict[str, Any]], manifests: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    baseline = _manifest_option(manifests, "baselineArm")
    if not isinstance(baseline, str):
        return []
    cells: dict[tuple[int, int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for objective, values in row.get("objectives", {}).items():
            if isinstance(values, dict):
                reward = _number(values.get("reward"))
                if reward is not None:
                    cells[
                        (row["optimizerSeed"], row["worldSeed"], row["arm"], str(objective))
                    ].append(reward)
    means = {key: float(np.mean(value)) for key, value in cells.items()}
    alternatives = sorted({key[2] for key in cells if key[2] != baseline})
    objectives = sorted({key[3] for key in cells})
    output: list[dict[str, Any]] = []
    seed_differences: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for alternative in alternatives:
        for objective in objectives:
            pairs = sorted(
                {(key[0], key[1]) for key in cells if key[2] == alternative and key[3] == objective}
            )
            for seed, world in pairs:
                base = means.get((seed, world, baseline, objective))
                candidate = means.get((seed, world, alternative, objective))
                if base is None or candidate is None:
                    continue
                difference = candidate - base
                seed_differences[(alternative, objective, seed)].append(difference)
                output.append(
                    {
                        "level": "optimizer_seed_world",
                        "baselineArm": baseline,
                        "alternativeArm": alternative,
                        "objective": objective,
                        "optimizerSeed": seed,
                        "worldSeed": world,
                        "difference": difference,
                    }
                )
    across: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (alternative, objective, seed), values in sorted(seed_differences.items()):
        difference = float(np.mean(values))
        across[(alternative, objective)].append(difference)
        output.append(
            {
                "level": "optimizer_seed",
                "baselineArm": baseline,
                "alternativeArm": alternative,
                "objective": objective,
                "optimizerSeed": seed,
                "worldSeed": None,
                "difference": difference,
            }
        )
    for (alternative, objective), values in sorted(across.items()):
        output.append(
            {
                "level": "across_optimizer_seeds",
                "baselineArm": baseline,
                "alternativeArm": alternative,
                "objective": objective,
                "optimizerSeed": None,
                "worldSeed": None,
                "difference": float(np.mean(values)),
                "medianDifference": float(np.median(values)),
                "optimizerSeedCount": len(values),
                "insufficient": len(values) < 3,
            }
        )
    return output


def _trade_signal(row: dict[str, Any]) -> float:
    trade = _nested(row, "economicDiagnostics", "domains", "trade")
    if isinstance(trade, dict):
        for name in ("realisedUtility", "realizedUtility", "settledValue", "deliveredQuantity"):
            value = _number(trade.get(name))
            if value is not None:
                return value
    total = 0.0
    for record in _nested(row, "economicDiagnostics", "trades") or []:
        if isinstance(record, dict):
            value = _number(record.get("realisedUtility", record.get("realizedUtility")))
            if value is not None:
                total += value
    return total


def trade_credit_latency(
    rows: Sequence[dict[str, Any]], manifests: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    max_lag = _manifest_option(manifests, "tradeCreditMaxLag")
    if not isinstance(max_lag, int) or max_lag < 0:
        return []
    sequences: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["runId"],
            row["arm"],
            row["optimizerSeed"],
            row["worldSeed"],
            row["trajectoryId"],
            row["campId"],
        )
        sequences[key].append(row)
    groups: dict[tuple[str, str, int], list[tuple[float, float, float, int]]] = defaultdict(list)
    for sequence in sequences.values():
        sequence.sort(key=lambda row: row["decisionTurn"])
        for index, row in enumerate(sequence):
            signal = _trade_signal(row)
            for lag in range(max_lag + 1):
                if index + lag >= len(sequence):
                    break
                future = sequence[index + lag]
                influence = 1.0
                for cursor in range(index, index + lag):
                    influence *= float(sequence[cursor]["gamma"]) * float(
                        sequence[cursor]["gaeLambda"]
                    )
                for objective, values in future.get("objectives", {}).items():
                    if not isinstance(values, dict):
                        continue
                    reward = _number(values.get("reward"))
                    if reward is not None:
                        groups[(row["arm"], str(objective), lag)].append(
                            (signal, reward, influence, row["optimizerSeed"])
                        )
    output = []
    for (arm, objective, lag), values in sorted(groups.items()):
        trade = np.asarray([item[0] for item in values])
        reward = np.asarray([item[1] for item in values])
        correlation = None
        if len(values) > 1 and trade.var() > 0 and reward.var() > 0:
            correlation = float(np.corrcoef(trade, reward)[0, 1])
        seeds = {item[3] for item in values}
        output.append(
            {
                "arm": arm,
                "objective": objective,
                "lag": lag,
                "count": len(values),
                "optimizerSeedCount": len(seeds),
                "pearson": correlation,
                "meanGammaLambdaInfluence": float(np.mean([item[2] for item in values])),
                "causal": False,
                "insufficient": len(values) < 30 or len(seeds) < 3,
            }
        )
    return output


def _conservation_failures(
    rows: Sequence[dict[str, Any]], tolerance: float = 1e-8
) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        conservation = _nested(row, "economicDiagnostics", "conservation")
        if not isinstance(conservation, dict):
            continue
        for name, value in conservation.items():
            residual = _number(value)
            if residual is not None and abs(residual) > tolerance:
                failures.append(
                    {
                        "transitionId": row["transitionId"],
                        "field": name,
                        "residual": residual,
                    }
                )
    return failures


def _missing_capabilities(
    manifests: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    calibration: Sequence[dict[str, Any]],
    forks: Sequence[dict[str, Any]],
    contrasts: Sequence[dict[str, Any]],
    latency: Sequence[dict[str, Any]],
    optimizer_rows: Sequence[dict[str, Any]],
) -> list[str]:
    missing = []
    if not any(row["evidenceClass"] == "original_training_trace" for row in rows):
        missing.append(
            "historical T3 optimizer attribution unavailable: no original training trace"
        )
    if not any(item["targetClass"] == "independent_return" for item in calibration):
        missing.append("independent-return critic calibration unavailable")
    if not forks:
        missing.append("matched-fork causal effects unavailable")
    if not contrasts:
        missing.append("paired arm contrasts unavailable or baselineArm not preregistered")
    if not latency:
        missing.append("trade credit latency unavailable or tradeCreditMaxLag not preregistered")
    if not optimizer_rows:
        missing.append("joined optimizer mutation diagnostics unavailable")
    boundaries = _manifest_option(manifests, "sliceBoundaries")
    if not isinstance(boundaries, dict) or not all(
        name in boundaries for name in ("population", "qol", "turn")
    ):
        missing.append("frozen population/QoL/turn slice boundaries unavailable")
    return missing


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Economic diagnostic report",
        "",
        f"Complete trace rows: {report['transitions']}; update rows: {report['updates']}; "
        f"joined mutations: {report['counts']['optimizerMutations']}.",
        f"Evidence classes: {', '.join(sorted(report['evidenceClasses'])) or 'none'}.",
        f"Conservation failures: {len(report['conservationFailures'])}.",
        "",
        "## Interpretation boundary",
        "",
        "Action correlations and trade latency are descriptive. Only rows in "
        "matched_fork_effects.csv "
        "with sameSnapshot=true and sameRandomStream=true support a causal contrast.",
        "",
        "## Unavailable evidence",
        "",
    ]
    if report["missingCapabilities"]:
        lines.extend(f"- {item}" for item in report["missingCapabilities"])
    else:
        lines.append("- None.")
    lines.extend(["", "## High-signal flags", ""])
    flags = report["flags"]
    if flags:
        lines.extend(f"- {item}" for item in flags)
    else:
        lines.append("- None at the configured thresholds.")
    return "\n".join(lines) + "\n"


def aggregate_trace_directories(paths: Sequence[str | Path], output: str | Path) -> dict[str, Any]:
    manifests, transitions, updates = load_traces(paths)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)

    decision_rows = [flatten(row) for row in transitions]
    dictionary = field_dictionary(transitions)
    domains = domain_summary(transitions, manifests)
    calibration = critic_calibration(transitions, manifests)
    alignment = action_alignment(transitions)
    forks = matched_fork_effects(transitions)
    contrasts = paired_arm_contrasts(transitions, manifests)
    latency = trade_credit_latency(transitions, manifests)
    mutation_rows = optimizer_update_rows(transitions, updates)
    mutation_rollup = optimizer_update_rollup(mutation_rows)
    failures = _conservation_failures(transitions)

    (destination / "field_dictionary.json").write_text(
        json.dumps(dictionary, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_csv(destination / "decision_rows.csv", decision_rows, IDENTITY_COLUMNS)
    _write_csv(destination / "optimizer_updates.csv", [flatten(row) for row in updates])
    _write_csv(destination / "optimizer_mutations.csv", mutation_rows)
    _write_csv(destination / "optimizer_update_summary.csv", mutation_rollup)
    _write_csv(destination / "domain_summary.csv", domains)
    _write_csv(destination / "critic_calibration.csv", calibration)
    _write_csv(destination / "action_alignment.csv", alignment)
    _write_csv(
        destination / "matched_fork_effects.csv",
        forks,
        (
            "evidenceClass",
            "arm",
            "optimizerSeed",
            "worldSeed",
            "pairId",
            "perturbedHead",
            "metric",
            "effect",
            "sameSnapshot",
            "sameRandomStream",
            "causal",
        ),
    )
    _write_csv(
        destination / "paired_arm_contrasts.csv",
        contrasts,
        (
            "level",
            "baselineArm",
            "alternativeArm",
            "objective",
            "optimizerSeed",
            "worldSeed",
            "difference",
        ),
    )
    _write_csv(
        destination / "trade_credit_latency.csv",
        latency,
        (
            "arm",
            "objective",
            "lag",
            "count",
            "optimizerSeedCount",
            "pearson",
            "meanGammaLambdaInfluence",
            "causal",
            "insufficient",
        ),
    )

    flags = []
    for item in domains:
        for name, value in item.items():
            if name.endswith("LossAbove10Percent") and value is True:
                flags.append(f"{item['arm']}/{item['domain']}/{item['metric']}: {name}")
    for item in calibration:
        if item.get("absoluteBiasAbove10Percent") is True:
            flags.append(
                f"{item['arm']}/{item['objective']}/{item['targetClass']}/"
                f"{item['slice']}={item['sliceValue']}: "
                "critic bias above 10% reference scale"
            )
    if failures:
        flags.append(f"{len(failures)} non-zero conservation residual(s)")
    for item in mutation_rows:
        if item["updateKind"] not in {"mutation", "critic-refit"}:
            continue
        critic_norm = _number(item.get("parameterUpdateNorms.critic"))
        critic_changed = item.get("criticHashChanged")
        if critic_norm == 0 or critic_changed is False:
            flags.append(
                f"{item['runId']}/rollout-{item['rollout']}/epoch-{item['epoch']}/"
                f"minibatch-{item['minibatch']}: critic did not mutate"
            )
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "artifacts": [
            {
                name: manifest.get(name)
                for name in ("artifactPath", "manifestSha256", "doneSha256", "runId", "arm")
            }
            for manifest in manifests
        ],
        "transitions": len(transitions),
        "updates": len(updates),
        "evidenceClasses": sorted({row["evidenceClass"] for row in transitions}),
        "optimizerSeeds": sorted({row["optimizerSeed"] for row in transitions}),
        "worldSeeds": sorted({row["worldSeed"] for row in transitions}),
        "historicalAttributionAvailable": any(
            row["evidenceClass"] == "original_training_trace" for row in transitions
        ),
        "conservationFailures": failures,
        "missingCapabilities": _missing_capabilities(
            manifests, transitions, calibration, forks, contrasts, latency, mutation_rows
        ),
        "counts": {
            "domainSummary": len(domains),
            "criticCalibration": len(calibration),
            "actionAlignment": len(alignment),
            "matchedForkEffects": len(forks),
            "pairedArmContrasts": len(contrasts),
            "tradeCreditLatency": len(latency),
            "optimizerMutations": len(mutation_rows),
            "optimizerUpdateSummary": len(mutation_rollup),
        },
        "flags": sorted(flags),
    }
    (destination / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (destination / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate complete model9 economic diagnostic traces"
    )
    parser.add_argument("traces", nargs="+", help="complete trace artifact directories")
    parser.add_argument("--out", required=True, help="output report directory")
    args = parser.parse_args(argv)
    aggregate_trace_directories(args.traces, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
