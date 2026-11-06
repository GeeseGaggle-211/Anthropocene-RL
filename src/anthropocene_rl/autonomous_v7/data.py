"""Teacher shards (``--model9-teacher``) -> training arrays, lineage rewards and scale-free value targets."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from . import contracts

OBSERVATION_KEYS = [
    "core",
    "core_present",
    "movement_mask",
    "clothing_route_mask",
    "shelter_route_mask",
    "economy_mix_mask",
] + [key for name, _, _ in contracts.SETS for key in (name, name + "_present")]
OPTIONAL_KEYS = {
    key for name in ("regions", "trade", "activities") for key in (name, name + "_present")
}


def normalize_optional(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normalize each file before joining vintages; never discard another file's conditioning."""
    out = dict(arrays)
    n = len(out["core"])
    for name, rows, width in contracts.SETS:
        out.setdefault(name, np.zeros((n, rows, width), np.float32))
        out.setdefault(name + "_present", np.zeros((n, rows), bool))
    native = np.zeros((n, contracts.REGION_MOVEMENT_SLOTS), bool)
    native[:, contracts.REGION_NATIVE] = True
    out.setdefault("region_movement_mask", native)
    out.setdefault("commit_hold_mask", np.zeros(n, bool))
    out.setdefault(
        "personality",
        np.tile(
            np.asarray([contracts.OBJECTIVE_WEIGHTS[k] for k in contracts.OBJECTIVES], np.float32),
            (n, 1),
        ),
    )
    return out


LABEL_KEYS = [
    "strategic",
    "strategic_mask",
    "service_levels",
    "service_mask",
    "clothing_route",
    "clothing_route_label_mask",
    "shelter_route",
    "shelter_route_label_mask",
    "movement",
    "movement_label_mask",
    "fission_policy",
    "repair",
]
OUTCOME_KEYS = [
    "camp",
    "turn",
    "society",
    "parent_society",
    "prior_population",
    "population",
    "qol",
    "nutrition",
    "years",
    "welfare",
    "terminated",
]


def load_world(directory: Path) -> dict[str, np.ndarray]:
    meta = json.loads((directory / "meta.json").read_text())
    from .protocol import scan

    scan(meta, "dataset metadata")
    # Sets and masks added after a shard was written (contracts 10-12) are optional: to_tensors treats them as absent.
    arrays = {
        key: np.load(directory / f"{key}.npy")
        for key in OBSERVATION_KEYS + LABEL_KEYS + OUTCOME_KEYS
        if key not in OPTIONAL_KEYS or (directory / f"{key}.npy").exists()
    }
    for key in ("region_movement_mask", "commit_hold_mask", "trade_utility"):
        if (directory / f"{key}.npy").exists():
            arrays[key] = np.load(directory / f"{key}.npy")
    if (directory / "privileged.npy").exists():
        arrays["privileged"] = np.load(directory / "privileged.npy")
    arrays["world"] = np.full(len(arrays["camp"]), int(meta["seed"]), dtype=np.int32)
    # Source family identity is independent of sequence IDs, which branch loaders rebase.
    arrays["source_seed"] = np.full(len(arrays["camp"]), int(meta["seed"]), np.int64)
    arrays["source_snapshot"] = np.full(len(arrays["camp"]), str(meta.get("snapshot_hash") or ""))
    # Search branch shards (branch_shards.py) carry a unique world id per (seed, decision point, alternative), each
    # row's personality, alternative kind and founding turn, and a bootstrap index into their own bootstrap block.
    for key in BRANCH_KEYS:
        if (directory / f"{key}.npy").exists():
            arrays[key] = np.load(directory / f"{key}.npy")
    return arrays


# Optional per-row arrays of search branch shards; "world" replaces the meta seed as the sequence key.
BRANCH_KEYS = ("world", "personality", "alt_kind", "founded_turn", "bootstrap_index", "calibration")


def load(
    directories: Iterable[Path],
    *,
    strict: bool = False,
    report: dict | None = None,
    declared_sources=None,
    calibration_manifest=None,
) -> dict[str, np.ndarray]:
    directories = sorted(directories)
    from .protocol import scan

    for directory in directories:
        if (directory / "meta.json").exists():
            scan(json.loads((directory / "meta.json").read_text()))
    verified, metadata = {}, []
    if strict:
        from . import provenance as prov
        from .teacher_data import verify_dataset, verify_shard

        for directory in directories:
            root = directory.parent
            if root not in verified:
                verified[root] = verify_dataset(root)
            metadata.append(verify_shard(directory))
        ids = [m["trajectory_id"] for m in metadata]
        if len(ids) != len(set(ids)):
            raise prov.ProvenanceError("duplicate teacher trajectory")
        records = [v["source_record"] for v in verified.values()]
        checked = prov.check(
            [("native_teacher", r["learner_commit"], r["source_run"]) for r in records],
            "reject",
            "native teacher",
            declared=declared_sources,
        )
        proof = {
            **checked,
            "schema": 1,
            "kind": "native_teacher",
            "verified": True,
            "datasets": list(verified.values()),
            "records": records,
            "dataset_refs": [
                {
                    "path": str((root / "dataset.json").resolve()),
                    "sha256": prov.checkpoint_hash(root / "dataset.json"),
                }
                for root in verified
            ],
        }
        proof["digest"] = prov.digest(proof)
        if report is not None:
            report.update(proof)
    worlds = [load_world(d) for d in directories if (d / "meta.json").exists()]
    if strict:
        for i, world in enumerate(worlds):
            world["world"] = np.full(len(world["camp"]), i, np.int64)
    if calibration_manifest:
        from .calibration import assert_training_rows

        for world in worlds:
            assert_training_rows(calibration_manifest, world)
    worlds = [normalize_optional(w) for w in worlds if len(w["camp"]) > 0]
    if not worlds:
        raise ValueError("no teacher shards found")
    keys = set.union(*(set(w) for w in worlds))
    for key in keys:
        template = next(w[key] for w in worlds if key in w)
        for world in worlds:
            if key not in world:
                world[key] = np.zeros((len(world["camp"]),) + template.shape[1:], template.dtype)
    return {key: np.concatenate([w[key] for w in worlds]) for key in keys}


def lineage_rewards(data: dict[str, np.ndarray], qol: bool = True) -> np.ndarray:
    """Population-weighted lineage sharing (decision D4): each settlement earns P_i x Q_lineage x dt, where the
    lineage is a parent and any daughter founded in the last ``LINEAGE_TURNS`` turns; rewards sum to realized welfare."""
    reward = np.zeros(len(data["camp"]), dtype=np.float64)
    first_turn: dict[tuple[int, int], int] = {}
    for i in np.argsort(data["turn"], kind="stable"):
        key = (int(data["world"][i]), int(data["camp"][i]))
        # Branch shards start mid-episode, so their settlements carry the turn they were founded on the full path.
        first_turn.setdefault(
            key, int(data["founded_turn"][i]) if "founded_turn" in data else int(data["turn"][i])
        )
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i in range(len(reward)):
        world, camp, turn = int(data["world"][i]), int(data["camp"][i]), int(data["turn"][i])
        parent = int(data["parent_society"][i])
        recent = parent > 0 and turn - first_turn[(world, camp)] < contracts.LINEAGE_TURNS
        group = parent if recent else int(data["society"][i])
        groups.setdefault((world, turn, group), []).append(i)
    for rows in groups.values():
        p = np.array([max(0, data["prior_population"][i]) for i in rows], dtype=np.float64)
        alive = np.array([not data["terminated"][i] for i in rows])
        q = (
            np.array([data["qol"][i] for i in rows], dtype=np.float64)
            if qol
            else np.ones(len(rows))
        ) * alive
        q_lineage = float((p * q).sum() / p.sum()) if p.sum() > 0 else 0.0
        for i, pi, a in zip(rows, p, alive, strict=False):
            reward[i] = pi * q_lineage * float(data["years"][i]) if a else 0.0
    return reward


def reward_components(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-decision rewards for each objective in ``contracts.OBJECTIVES``: lineage welfare (P x Q_lineage x dt),
    lineage population-years (P x dt for living settlements) and (contract 11) the village's own realised trade
    utility, in person-QoL-years on true valuations; self-interested: a partner's gain is never credited."""
    n = len(data["camp"])
    trade = (
        np.asarray(data["trade_utility"], dtype=np.float64)
        if "trade_utility" in data
        else np.zeros(n)
    )
    # qol_years: the lineage QoL over the interval regardless of headcount (living settlements); welfare is P times it.
    welfare = lineage_rewards(data)
    p = np.maximum(1.0, np.asarray(data["prior_population"], dtype=np.float64))
    alive = ~np.asarray(data["terminated"], dtype=bool)
    qol_years = np.where(alive & (np.asarray(data["prior_population"]) > 0), welfare / p, 0.0)
    return {
        "welfare": welfare,
        "population_years": lineage_rewards(data, qol=False),
        "trade_utility": trade,
        "qol_years": qol_years,
    }


def sequences(data: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Row indices per (world, camp), ordered by turn."""
    order = np.lexsort((data["turn"], data["camp"], data["world"]))
    keys = np.stack([data["world"][order], data["camp"][order]], axis=1)
    breaks = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1
    return np.split(order, breaks)


def discount(years: np.ndarray) -> np.ndarray:
    return np.exp(-np.asarray(years, dtype=np.float64) / contracts.HORIZON_YEARS)


def lambda_returns(
    data: dict[str, np.ndarray],
    reward: np.ndarray,
    value: np.ndarray | None,
    lam: float = 0.95,
    bootstrap: np.ndarray | None = None,
) -> np.ndarray:
    """TD(lambda) returns in raw welfare units. ``value`` (raw units, per row) bootstraps truncated sequences and the
    lambda mixture; without it the truncated tail bootstraps from a stationary geometric continuation. ``bootstrap``
    (raw units, per row; NaN where absent) is the value of the observation after a truncated sequence's last row
    (search branch shards); it replaces the row's own value there whenever ``value`` is given."""
    target = np.zeros(len(reward), dtype=np.float64)
    for rows in sequences(data):
        g_next = 0.0
        for position in range(len(rows) - 1, -1, -1):
            i = rows[position]
            gamma = float(discount(data["years"][i]))
            if data["terminated"][i]:
                g = reward[i]
            elif position == len(rows) - 1:
                # Truncated: continue at this turn's reward rate (or the critic's estimate) for the remaining horizon.
                tail = value[i] if value is not None else reward[i] / max(1e-9, 1 - gamma)
                if value is not None and bootstrap is not None and np.isfinite(bootstrap[i]):
                    tail = bootstrap[i]
                g = reward[i] + gamma * tail
            else:
                j = rows[position + 1]
                v_next = value[j] if value is not None else g_next
                g = reward[i] + gamma * ((1 - lam) * v_next + lam * g_next)
            target[i] = g
            g_next = g
    return target


def scale_free(target: np.ndarray, data: dict[str, np.ndarray]) -> np.ndarray:
    """u = G / P: return per resident at the decision, so 20- and 800-person villages train alike."""
    return target / np.maximum(1.0, data["prior_population"].astype(np.float64))


def symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def summary(data: dict[str, np.ndarray]) -> dict[str, float]:
    worlds = len(np.unique(data["world"]))
    return {
        "rows": float(len(data["camp"])),
        "worlds": float(worlds),
        "mean_population": float(np.mean(data["prior_population"])),
        "mean_qol": float(np.mean(data["qol"])),
        "hungry_share": float(np.mean(data["nutrition"] < 0.8)),
        "terminated_share": float(np.mean(data["terminated"])),
        "welfare_per_world": float(np.sum(data["welfare"]) / max(1, worlds)),
        "gamma_per_turn": float(math.exp(-contracts.TURN_YEARS / contracts.HORIZON_YEARS)),
    }
