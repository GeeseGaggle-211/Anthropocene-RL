"""Authoritative decision sidecars. Deliberately never imported by observation tensor construction."""  # noqa: E501 - docstring/prose line kept verbatim

from __future__ import annotations

import numpy as np


def canonical(value):
    if isinstance(value, dict):
        return {k[:1].lower() + k[1:]: canonical(v) for k, v in value.items()}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def validate(
    step: dict, turn: int, camps, *, required: bool, trajectory: str | None = None
) -> list[dict]:
    records, seen, identities = [], set(), set()
    by_camp = {int(v["camp"]): v for v in step["villages"]}
    for camp in camps:
        raw = by_camp.get(int(camp), {}).get("decisionDiagnostics")
        if raw is None:
            if required:
                raise ValueError(f"missing decision diagnostics: camp {camp}, turn {turn}")
            continue
        d = canonical(raw)
        if (
            d.get("schemaVersion") != 1
            or d.get("campId") != int(camp)
            or d.get("decisionTurn") != int(turn)
        ):
            raise ValueError("decision diagnostics schema/turn/camp mismatch")
        identity = d.get("trajectoryId")
        if not identity or (trajectory is not None and identity != trajectory):
            raise ValueError("decision diagnostics trajectory mismatch")
        key = (identity, turn, camp)
        if key in seen:
            raise ValueError("duplicate decision diagnostics")
        seen.add(key)
        identities.add(identity)
        for name in (
            "requested",
            "accepted",
            "realised",
            "requestedMask",
            "acceptedMask",
            "realisedMask",
            "activeMask",
        ):
            if not isinstance(d.get(name), dict):
                raise ValueError(f"missing decision diagnostics {name}")
            for head, width in {
                "strategic": 5,
                "service": 6,
                "clothingRoute": 1,
                "shelterRoute": 1,
                "movement": 1,
                "intentTile": 1,
                "fission": 1,
                "commit": 1,
                "tradeValue": 7,
                "tradeGate": 7,
                "tradeShare": 1,
            }.items():
                if head not in d[name] or np.atleast_1d(d[name][head]).shape != (width,):
                    raise ValueError(f"decision diagnostics {name}/{head} layout mismatch")
        records.append(d)
    if len(identities) > 1:
        raise ValueError("mixed trajectories in one bridge step")
    return records


def aggregate(records: list[dict]) -> dict:
    totals = {}
    for raw in records:
        d = canonical(raw)
        for before, after in (("requested", "accepted"), ("accepted", "realised")):
            for head, values in d[before].items():
                if head not in d[after]:
                    continue
                a, b = np.atleast_1d(values), np.atleast_1d(d[after][head])
                masks = [
                    np.atleast_1d(d[n].get(head, False)).astype(bool)
                    for n in (before + "Mask", after + "Mask", "activeMask")
                ]
                if a.shape != b.shape or any(m.shape != a.shape for m in masks):
                    raise ValueError(f"decision diagnostics shape mismatch: {head}")
                valid = masks[0] & masks[1] & masks[2] & np.isfinite(a) & np.isfinite(b)
                for i in range(len(a)):
                    key = f"{head}[{i}]" if len(a) > 1 else head
                    cell = totals.setdefault(key, {}).setdefault(
                        before + "_to_" + after,
                        {
                            "count": 0,
                            "changed": 0,
                            "absolute_sum": 0.0,
                            "inactive": 0,
                            "before_available": 0,
                            "after_available": 0,
                            "mask_changes": 0,
                            "uncomparable": 0,
                        },
                    )
                    if not masks[2][i]:
                        cell["inactive"] += 1
                        continue
                    cell["before_available"] += int(masks[0][i])
                    cell["after_available"] += int(masks[1][i])
                    cell["mask_changes"] += int(masks[0][i] != masks[1][i])
                    if not valid[i]:
                        cell["uncomparable"] += 1
                        continue
                    delta = abs(float(b[i]) - float(a[i]))
                    cell["count"] += 1
                    cell["changed"] += int(delta > 1e-7)
                    cell["absolute_sum"] += delta
    for by in totals.values():
        for cell in by.values():
            cell["change_rate"] = cell["changed"] / cell["count"] if cell["count"] else None
            cell["mean_magnitude"] = cell["absolute_sum"] / cell["count"] if cell["count"] else None
    return {"decisions": len(records), "heads": totals}


def validate_raw_materials(
    step: dict,
    turn: int | None,
    *,
    required: bool,
    initial: bool = False,
    raw_enabled: bool | None = None,
) -> dict:
    """Physical receipt sidecar, keyed by camp/material; never reconstructed from head controls or grant alone."""  # noqa: E501 - docstring/prose line kept verbatim
    if "rawMaterialDiagnostics" not in step or "diagnosticSchemaVersion" not in step:
        if required:
            raise ValueError("missing physical diagnostic sidecar/schema")
        return {}
    version = step["diagnosticSchemaVersion"]
    records = step["rawMaterialDiagnostics"]
    if version not in (0, 1, 2, 3) or not isinstance(records, dict):
        raise ValueError("physical diagnostic schema/layout mismatch")
    if version == 1 and (raw_enabled is True or records):
        raise ValueError("provisional schema-1 raw-material semantics cannot continue as schema 2")
    if initial or version == 0:
        if records or (required and not initial):
            raise ValueError(
                "physical diagnostics absent/reset schema cannot carry preceding receipts"
            )
        return {}
    trajectory = step.get("trajectoryId")
    if not isinstance(trajectory, str) or not trajectory:
        raise ValueError("physical diagnostics missing trajectory identity")
    by_camp = {int(v["camp"]): v for v in step["villages"]}
    output = {}
    for key, raw in records.items():
        d = canonical(raw)
        camp, material = d.get("campId"), d.get("material")
        if (
            d.get("schemaVersion") != 2
            or version not in (2, 3)
            or not isinstance(camp, int)
            or not isinstance(material, int)
            or camp not in by_camp
            or str(key) != f"{camp}:{material}"
            or d.get("decisionTurn") != turn
            or not isinstance(d.get("requestId"), str)
            or not d["requestId"]
        ):
            raise ValueError("physical diagnostics camp/material/turn/request identity mismatch")
        values = [d.get(name) for name in ("requestedUnits", "acceptedUnits", "realisedUnits")]
        if any(
            not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
            for value in values
        ):
            raise ValueError("physical diagnostics require finite nonnegative receipt units")
        if values[1] > values[0] + 1e-8 or values[2] > values[1] + 1e-8:
            raise ValueError("physical diagnostics receipt exceeds request/grant")
        output[str(key)] = d
    for village in step["villages"]:
        if village.get("decisionDiagnostics") is not None:
            d = canonical(village["decisionDiagnostics"])
            if d.get("trajectoryId") != trajectory or d.get("decisionTurn") != turn:
                raise ValueError("physical/head diagnostic trajectory/turn mismatch")
    return output
